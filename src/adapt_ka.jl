# =============================================================================
# Backend-generic blob adaptation — log-space posterior IRLS on any KA device
# =============================================================================
#
# Port of the CUDA implementation (formerly ext/adapt_cuda.jl) to pure
# KernelAbstractions, so `adapt` runs on CPU, CUDA, Metal, or ROCm.
#
# Design (each point audited against the legacy kernels, 2026-08-10):
#
# - GPU kernels are `cpu=false`: they use group-shared memory + barriers,
#   which KA's CPU transform cannot compile. KA.CPU() instead runs
#   `_adapt_cpu` — a per-blob threaded host implementation of the SAME math
#   (same T-storage quantization, same sweep, same LU, same gates), the
#   split ext/segmented_reduce.jl pioneered.
#
# - Accumulation precision `Ta` dispatches on `KA.supports_float64(device)`:
#   Float64 on CUDA/CPU (arithmetic identical to the legacy kernels),
#   Float32 on Metal (which has no Float64 at all — a DISTINCT numeric
#   regime, never claimed as parity). Per-pixel storage stays
#   Float32 staging (the deleted `AdaptConfig.store`'s only value) exactly as before.
#
# - `lgamma` is supplied TO the kernels as an argument: the CUDA extension
#   supplies the `__nv_lgamma` libdevice ccall (bit-identical to legacy);
#   every other backend gets the pure-Julia Lanczos below
#   (SpecialFunctions.loggamma does not GPU-compile).
#
# - The 6×6 normal-equation solve is the `_solve_6x6_batched!` hook: cuBLAS
#   getrf/getrs on CUDA (bit-identical, including which near-degenerate
#   blobs its `info` invalidates), a per-blob register LU with partial
#   pivoting everywhere else.
#
# - The legacy fused kernel's thread-1 `break` past a barrier (formally UB
#   on CUDA, a deadlock elsewhere) is replaced by a shared-memory stop flag
#   all threads read — identical iteration counts and ν values.
#
# - The write-only f_hat/r buffers of the legacy kernels are gone (~2×
#   total_pixels of allocation + 2 stores/pixel/iteration); every consumer
#   reads r², which is kept.
#
# - Pixel slots are assigned by an atomic counter, so the flat-buffer pixel
#   ORDER is scheduler-dependent — exactly as on legacy CUDA, whose output
#   already differs in final ulps between identical runs. Parity testing is
#   therefore frame-level with tolerances, never bitwise.
# =============================================================================

using KernelAbstractions: @kernel, @Const, @context, @index, @groupsize,
    @localmem, @synchronize, @uniform
using SpecialFunctions: loggamma

# The per-blob scan samples a canonical (2·31 + 1)² = 3969-point grid, which
# is what `_SORT_PADDED` (the next power of two) sizes the sweep's bitonic
# sort for: 32 KB of Float64 group-shared memory on CUDA, 16 KB of Float32 on
# Metal (whose threadgroup cap is exactly 32 KB).
#
# `_PATCH_HALFWIDTH` is the grid halfwidth in SAMPLES, not in pixels. A blob
# whose support fits within it is sampled one sample per pixel (`spacing == 1`,
# bit-identical to a plain pixel scan); a bigger blob keeps the same sample
# count and widens the spacing instead — see `_scan_grid`.
#
# It was formerly `_SORT_HW_CAP`, a hard CLAMP on the pixel halfwidth named
# after the sort that motivated it. That silently fitted every blob with
# σ > 31/cutoff on a crop of its own core: at σ ≈ 96 px the window covered
# 0.4σ and the fit returned ~0.4× the true scale, unstable run to run. The
# sort bound is real, but it bounds the SAMPLE COUNT, never the extent.
const _SORT_PADDED = Int32(4096)
const _PATCH_HALFWIDTH = Int32(31)

"""
    _scan_grid(R) -> (halfwidth, spacing)

Sample geometry for a blob whose fitting support is `±R` pixels: a
`(2·halfwidth + 1)²` grid with `spacing` pixels between samples, always covering
`±R` and never exceeding `_SORT_PADDED` samples.

`R ≤ _PATCH_HALFWIDTH` gives `spacing == 1` — integer pixel positions, the exact
scan this replaced. Beyond that the grid stops growing and the spacing opens
up, so per-blob cost is constant in σ rather than quadratic. A blob that big
is oversampled by ~15 samples per σ even at the widest spacing, far above
Nyquist for a Gaussian, and decimating (rather than area-averaging) keeps the
samples effectively independent — which the sweep's Jeffreys score assumes.
"""
@inline function _scan_grid(R::Real)
    hwc = Int(_PATCH_HALFWIDTH)
    R <= hwc && return (clamp(ceil(Int, R), _MIN_REGION_HALFWIDTH, hwc), 1.0)
    return (hwc, Float64(R) / hwc)
end

# The clamped bilinear sample, the device-safe lgamma/digamma/trigamma series,
# the in-register 6×6 LU and the workgroup tree reduction are VisualGeometryToolkitCore's
# since 0.21.1 (hoisted verbatim from here — same op sequences).
using VisualGeometryToolkitCore: bilinear_sample_clamped, lgamma_lanczos, digamma_stirling,
    trigamma_stirling, lu_solve!, tree_sum!


# =============================================================================
# Backend hooks
# =============================================================================

"""
    _device_lgamma(backend) -> function

The lgamma the adaptation kernels use on `backend`. The CUDA extension
overrides this with the `__nv_lgamma` libdevice ccall so CUDA results stay
bit-identical to the legacy kernels; the default is the device-safe Lanczos
approximation below. Every call site has argument ≥ 1, so no reflection.
"""
_device_lgamma(::KA.Backend) = lgamma_lanczos

@kernel inbounds = true cpu = false unsafe_indices = true function _lu6_kernel!(
        AtWA, AtWz, info, @Const(valid), @Const(converged)
    )
    blob = @index(Global, Linear)
    if blob <= length(info) && valid[blob] && converged[blob] == Int32(0)
        Ta = eltype(AtWA)
        A = StaticArrays.MMatrix{6, 6, Ta}(undef)
        b = StaticArrays.MVector{6, Ta}(undef)
        for j in 1:6, i in 1:6
            A[i, j] = AtWA[i, j, blob]
        end
        for i in 1:6
            b[i] = AtWz[i, 1, blob]
        end
        inf = lu_solve!(A, b)
        info[blob] = inf
        if inf == Int32(0)
            for i in 1:6
                AtWz[i, 1, blob] = b[i]
            end
        end
    end
end

"""
    _solve_6x6_batched!(backend, AtWA, AtWz, pivot, info, valid, converged)

Per-blob 6×6 solve: θ_new replaces `AtWz[:, 1, blob]`; `info[blob] ≠ 0`
flags a singular system (the apply/check step invalidates those blobs).
Generic method: one-thread-per-blob register LU (skips invalid/converged
blobs, whose results the apply step ignores anyway). The CUDA extension
overrides with cuBLAS getrf/getrs_strided_batched! — bit-identical to the
legacy implementation; `pivot` exists for that method alone.
"""
function _solve_6x6_batched!(backend::KA.Backend, AtWA, AtWz, pivot, info, valid, converged)
    N = length(info)
    _lu6_kernel!(backend, 256)(AtWA, AtWz, info, valid, converged; ndrange = 256 * cld(N, 256))
    return nothing
end

# =============================================================================
# Device kernels (cpu=false: group-shared memory + barriers)
# =============================================================================
#
# All per-workgroup branches below (`valid[blob]`, `converged[blob]`) are
# GROUP-UNIFORM — every thread in the group reads the same flag — so the
# barriers inside them are convergent.


# Stage 1: per-blob background (max unsaturated intensity, shared-memory max
# reduction). Stage 2: collect valid pixels into the flat buffers, slots
# assigned by an atomic counter (order scheduler-dependent, as on legacy).
@kernel inbounds = true cpu = false unsafe_indices = true function _scan_ka!(
        A, f_obs, v, cnt, bg,
        @Const(img), @Const(cx), @Const(cy), @Const(hw), @Const(spacing), @Const(valid),
        @Const(offsets), H::Int32, W::Int32, sat_lo, sat_hi, min_int
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(bg)
    smem = @localmem eltype(bg) (256,)
    if valid[blob]
        bx = cx[blob]
        by = cy[blob]
        bhw = hw[blob]
        # Canonical grid anchored on the same integer pixel the pixel scan
        # used, so `st == 1` lands exactly on those pixels.
        st = Ta(spacing[blob])
        ax = floor(bx)
        ay = floor(by)
        side = Int32(2) * bhw + Int32(1)
        npix = side * side
        base = offsets[blob]

        local_max = typemin(Ta)
        i = tid
        while i <= npix
            gx = (i - Int32(1)) % side
            gy = (i - Int32(1)) ÷ side
            xf = ax + Ta(gx - bhw) * st
            yf = ay + Ta(gy - bhw) * st
            ok, val = bilinear_sample_clamped(Ta, img, yf, xf, H, W)
            if ok && val >= sat_lo && val <= sat_hi && val > local_max
                local_max = val
            end
            i += nt
        end
        smem[tid] = local_max
        @synchronize()
        s = Int32(128)
        while s >= Int32(1)
            if tid <= s && tid + s <= nt
                smem[tid] = max(smem[tid], smem[tid + s])
            end
            @synchronize()
            s >>= Int32(1)
        end
        background = smem[1]
        if tid == Int32(1)
            bg[blob] = background
            cnt[blob] = Int32(0)
        end
        @synchronize()

        if background > zero(Ta)
            i = tid
            while i <= npix
                gx = (i - Int32(1)) % side
                gy = (i - Int32(1)) ÷ side
                xf = ax + Ta(gx - bhw) * st
                yf = ay + Ta(gy - bhw) * st
                ok, val = bilinear_sample_clamped(Ta, img, yf, xf, H, W)
                if ok && val >= sat_lo && val <= sat_hi
                    f = background - val
                    if f > min_int
                        slot = KA.@atomic cnt[blob] += Int32(1)
                        flat = base + slot
                        # Offsets stay in IMAGE pixels regardless of `st`, so
                        # θ, r² and outlier_halfwidth need no rescale downstream.
                        dx = xf - Ta(0.5) - bx
                        dy = yf - Ta(0.5) - by
                        f_obs[flat] = f
                        v[flat] = val
                        A[flat, 1] = one(Ta)
                        A[flat, 2] = dx
                        A[flat, 3] = dy
                        A[flat, 4] = dx * dx
                        A[flat, 5] = Ta(2.0) * dx * dy
                        A[flat, 6] = dy * dy
                    end
                end
                i += nt
            end
        end
    end
end

# Per-blob max(f_obs) — strided shared-memory reduction, one group per blob.
@kernel inbounds = true cpu = false unsafe_indices = true function _segmented_maxf_ka!(
        maxf, @Const(f_obs), @Const(cnt), @Const(valid), @Const(offsets)
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(maxf)
    smem = @localmem eltype(maxf) (256,)
    if valid[blob]
        n = cnt[blob]
        base = offsets[blob]
        m = typemin(Ta)
        j = tid
        while j <= n
            fv = Ta(f_obs[base + j])
            fv > m && (m = fv)
            j += nt
        end
        smem[tid] = m
        @synchronize()
        s = Int32(128)
        while s >= Int32(1)
            if tid <= s
                smem[tid] = max(smem[tid], smem[tid + s])
            end
            @synchronize()
            s >>= Int32(1)
        end
        tid == Int32(1) && (maxf[blob] = smem[1])
    else
        tid == Int32(1) && (maxf[blob] = zero(Ta))
    end
end

# Log-space whitened residuals r² = ((log(bg−v) − Aθ)·(bg−v))².
# =============================================================================
# Shared scalar core — the per-element algebra of the Student-t/uniform mixture
# =============================================================================
#
# Used VERBATIM by both realizations of the adaptation — the `cpu = false`
# workgroup kernels (shared-memory bitonic sort + barriers) and the threaded
# host path (`_adapt_cpu_one!`). The two orchestrations are shaped by their
# hardware and stay separate; the ALGEBRA lives here exactly once.
#
# EXPRESSION TREES ARE LOAD-BEARING. The host path is bit-compared against a
# recorded baseline and the device path against the legacy CUDA trajectory —
# do not re-associate, do not introduce muladd/fma, do not "simplify".
# `Ta` is the accumulation type (Float64 on CPU/CUDA, Float32 on Metal);
# `lgamma_f` is the backend's loggamma (libdevice `__nv_lgamma` on CUDA,
# SpecialFunctions.loggamma on the host).

# Squared log-space residual of one pixel: r = (log f − z_pred)·f, guarded for
# non-positive inverted intensity.
@inline function _adapt_r2(fi::Ta, z_pred::Ta) where {Ta}
    if fi > Ta(1.0e-10)
        ri = (log(fi) - z_pred) * fi
        return ri * ri
    else
        return zero(Ta)
    end
end

# RSDR's Jeffreys sweep score at partition size kk (jdof = kk − m > 0):
# S = lgamma(j/2) − (j/2)·log(π·RSS) − (n − kk)·log_two_outlier_halfwidth.
@inline function _sweep_score(jdof::Integer, RSS::Ta, outlier_term::Ta, lgamma_f::F) where {Ta, F}
    RSS_safe = max(RSS, Ta(1.0e-30))
    ν_half = Ta(jdof) / Ta(2.0)
    lg_j = if jdof == 1
        Ta(0.5) * log(Ta(3.141592653589793))
    elseif jdof == 2
        zero(Ta)
    else
        lgamma_f(ν_half)
    end
    return lg_j - ν_half * log(Ta(3.141592653589793) * RSS_safe) - outlier_term
end

# log C of the Student-t density at (ν, ŝ²) — the constant of the E-step.
@inline _weight_logC(ν::Ta, s2::Ta, lgamma_f::F) where {Ta, F} =
    lgamma_f((ν + Ta(1.0)) / Ta(2.0)) - lgamma_f(ν / Ta(2.0)) -
    Ta(0.5) * log(ν * Ta(3.141592653589793) * s2)

# Posterior responsibility and IRLS weight of one pixel (spatial factor is the
# caller's — it reads the θ layout).
@inline function _posterior_weight(r2i::Ta, ν::Ta, s2::Ta, half_νp1::Ta, log_C::Ta, log_pout::Ta) where {Ta}
    ti = r2i / (ν * s2)
    log_pin = log_C - half_νp1 * log1p(ti)
    p_in = Ta(1.0) / (Ta(1.0) + exp(log_pout - log_pin))
    u_i = (ν + Ta(1.0)) / (ν * s2 + r2i)
    return p_in, p_in * u_i
end

# One pixel's contribution to the ν-Newton objective: (dq_i, d2q_data_i).
@inline function _nu_point_terms(τ_i::Ta, ν::Ta, inv_ν::Ta, inv_ν2::Ta, ψ_half_νp1::Ta, ψ_half_ν::Ta) where {Ta}
    inv_1pτ = Ta(1.0) / (Ta(1.0) + τ_i)
    dq_i = Ta(0.5) * ψ_half_νp1 - Ta(0.5) * ψ_half_ν - Ta(0.5) * inv_ν -
        Ta(0.5) * log1p(τ_i) + (ν + Ta(1.0)) * τ_i * inv_1pτ * Ta(0.5) * inv_ν
    d2q_data_i = Ta(0.5) * τ_i * inv_1pτ * inv_ν -
        Ta(0.5) * τ_i * (ν + Ta(2.0) + τ_i) * inv_1pτ * inv_1pτ * inv_ν2
    return dq_i, d2q_data_i
end

# The ν-Newton step from the reduced sums: returns (ν_new, stop::Bool).
@inline function _next_nu(ν::Ta, dQ::Ta, d2Q_data::Ta, sum_w::Ta, ψ1_half_νp1::Ta, ψ1_half_ν::Ta, inv_ν2::Ta) where {Ta}
    sum_w < Ta(1.0e-10) && return ν, true
    d2Q = sum_w * (Ta(0.25) * ψ1_half_νp1 - Ta(0.25) * ψ1_half_ν + Ta(0.5) * inv_ν2) + d2Q_data
    abs(d2Q) < Ta(1.0e-30) && return ν, true
    ν_new = clamp(ν - dQ / d2Q, Ta(2.0), Ta(1.0e4))
    return ν_new, abs(dQ / sum_w) < Ta(1.0e-8)
end

# The apply-time shape gate on (θ₄, θ₅, θ₆): the fitted quadratic must be a
# concave paraboloid (tr < 0, det > 0) with eigenvalue ratio within the aspect
# cap. Returns true when the update is ACCEPTABLE.
@inline function _shape_ok(θ4::Ta, θ5::Ta, θ6::Ta, max_aspect_sq) where {Ta}
    tr_C = θ4 + θ6
    det_C = θ4 * θ6 - θ5 * θ5
    (tr_C < zero(Ta) && det_C > zero(Ta)) || return false
    disc = tr_C * tr_C - Ta(4.0) * det_C
    disc < zero(Ta) && (disc = zero(Ta))
    sqrt_disc = sqrt(disc)
    denom = tr_C + sqrt_disc
    if abs(denom) > Ta(1.0e-30)
        λ_ratio = (tr_C - sqrt_disc) / denom
        λ_ratio > max_aspect_sq && return false
    end
    return true
end

# PAIRED COPY (2026-09 audit): the residual + soft-weight arithmetic lives in
# _residuals_ka!/_soft_weights_ka! (pre-loop init) AND re-inlined in
# _fused_ka! (the per-iteration kernel). Verified not-drifted; any change to
# the shared atoms' call shape must land in both. Deliberately NOT collapsed:
# adapt is nondeterministic on CUDA, so a de-inline cannot be certified
# bitwise — only frame-level — and this is the quality-critical path.
@kernel inbounds = true cpu = false unsafe_indices = true function _residuals_ka!(
        r2, @Const(A), @Const(v), @Const(theta), @Const(bg),
        @Const(cnt), @Const(valid), @Const(offsets)
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(theta)
    if valid[blob]
        n = cnt[blob]
        base = offsets[blob]
        bg_val = bg[blob]
        i = tid
        while i <= n
            gid = base + i
            z_pred = Ta(A[gid, 1]) * theta[1, blob] + Ta(A[gid, 2]) * theta[2, blob] +
                Ta(A[gid, 3]) * theta[3, blob] + Ta(A[gid, 4]) * theta[4, blob] +
                Ta(A[gid, 5]) * theta[5, blob] + Ta(A[gid, 6]) * theta[6, blob]
            r2[gid] = eltype(r2)(_adapt_r2(bg_val - Ta(v[gid]), z_pred))
            i += nt
        end
    end
end

# Per-blob bitonic sort of r² in shared memory → outlier_halfwidth = √median(r²) →
# marginal posterior sweep (RSDR's Jeffreys `_sweep_score` algebra) → ν, ŝ².
@kernel inbounds = true cpu = false unsafe_indices = true function _sort_sweep_ka!(
        nu, s_hat_sq, outlier_halfwidth, @Const(r2), @Const(np),
        @Const(cnt), @Const(valid), @Const(offsets), lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(nu)
    smem = @localmem eltype(nu) (4096,)
    if valid[blob]
        base = offsets[blob]
        n = cnt[blob]

        i = tid
        while i <= _SORT_PADDED
            smem[i] = (i <= n) ? Ta(r2[base + i]) : typemax(Ta)
            i += nt
        end
        @synchronize()

        k = Int32(2)
        while k <= _SORT_PADDED
            j = k >> Int32(1)
            while j > Int32(0)
                idx = tid - Int32(1)
                while idx < _SORT_PADDED
                    ixj = idx ⊻ j
                    if ixj > idx
                        asc = (idx & k) == Int32(0)
                        a = smem[idx + Int32(1)]
                        b = smem[ixj + Int32(1)]
                        if asc ? a > b : a < b
                            smem[idx + Int32(1)] = b
                            smem[ixj + Int32(1)] = a
                        end
                    end
                    idx += nt
                end
                @synchronize()
                j >>= Int32(1)
            end
            k <<= Int32(1)
        end

        if tid == Int32(1)
            mid = cld(n, Int32(2))
            blob_outlier_halfwidth = max(sqrt(smem[mid]), Ta(1.0e-30))
            outlier_halfwidth[blob] = blob_outlier_halfwidth

            m = Int32(np[blob]) + Int32(1)
            log_two_outlier_halfwidth = log(Ta(2.0) * blob_outlier_halfwidth)

            RSS = zero(Ta)
            best_S = Ta(-1.0e30)
            best_k = m + Int32(1)

            kk = Int32(1)
            while kk <= n
                RSS += smem[kk]
                if kk > m
                    S = _sweep_score(kk - m, RSS, Ta(n - kk) * log_two_outlier_halfwidth, lgamma_f)
                    if S > best_S
                        best_S = S
                        best_k = kk
                    end
                end
                kk += Int32(1)
            end

            ν = max(Ta(best_k - m), Ta(2.0))
            RSS_k = zero(Ta)
            kk = Int32(1)
            while kk <= best_k
                RSS_k += smem[kk]
                kk += Int32(1)
            end
            nu[blob] = ν
            s_hat_sq[blob] = max(RSS_k / ν, Ta(1.0e-30))
        end
    end
end

# Soft posterior weights + optional spatial factor (initial weights; the
# per-iteration update is fused into `_fused_ka!`).
# PAIRED COPY (2026-09 audit): the residual + soft-weight arithmetic lives in
# _residuals_ka!/_soft_weights_ka! (pre-loop init) AND re-inlined in
# _fused_ka! (the per-iteration kernel). Verified not-drifted; any change to
# the shared atoms' call shape must land in both. Deliberately NOT collapsed:
# adapt is nondeterministic on CUDA, so a de-inline cannot be certified
# bitwise — only frame-level — and this is the quality-critical path.
@kernel inbounds = true cpu = false unsafe_indices = true function _soft_weights_ka!(
        w, pin, @Const(r2), @Const(A), @Const(theta), @Const(nu),
        @Const(s_hat_sq), @Const(outlier_halfwidth),
        @Const(cnt), @Const(valid), @Const(offsets), spatial::Bool, lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(theta)
    if valid[blob]
        n = cnt[blob]
        base = offsets[blob]
        ν = nu[blob]
        s2 = max(s_hat_sq[blob], Ta(1.0e-30))
        half_νp1 = (ν + Ta(1.0)) / Ta(2.0)
        log_C = _weight_logC(ν, s2, lgamma_f)
        log_pout = -log(Ta(2.0) * outlier_halfwidth[blob])
        i = tid
        while i <= n
            gid = base + i
            p_in, wi = _posterior_weight(Ta(r2[gid]), ν, s2, half_νp1, log_C, log_pout)
            pin[gid] = p_in
            if spatial
                q = theta[4, blob] * Ta(A[gid, 4]) +
                    theta[5, blob] * Ta(A[gid, 5]) +
                    theta[6, blob] * Ta(A[gid, 6])
                wi *= exp(q)
            end
            w[gid] = wi
            i += nt
        end
    end
end

# 6×6 A'WA / A'Wz accumulate with W = w·f² — 27 thread-local Float-Ta
# accumulators + the shared-memory tree reduction, np-masked write with
# identity padding on inactive params. Arithmetic mirrors legacy exactly.
@kernel inbounds = true cpu = false unsafe_indices = true function _accumulate_ka!(
        AtWA, AtWz, @Const(A), @Const(v), @Const(w), @Const(bg),
        @Const(cnt), @Const(valid), @Const(offsets), @Const(np),
        @Const(converged)
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(AtWA)
    smem = @localmem eltype(AtWA) (256,)
    if valid[blob] && converged[blob] == Int32(0)
        n = cnt[blob]
        base = offsets[blob]
        bg_val = bg[blob]

        acc = StaticArrays.MVector{27, Ta}(undef)
        for kk in 1:27
            acc[kk] = zero(Ta)
        end

        i = tid
        while i <= n
            gid = base + i
            fi = bg_val - Ta(v[gid])
            if fi >= Ta(1.0e-10)
                zi = log(fi)
                wi = Ta(w[gid]) * fi * fi
                a1 = Ta(A[gid, 1]); a2 = Ta(A[gid, 2]); a3 = Ta(A[gid, 3])
                a4 = Ta(A[gid, 4]); a5 = Ta(A[gid, 5]); a6 = Ta(A[gid, 6])
                acc[1] += wi * a1 * a1; acc[2] += wi * a1 * a2; acc[3] += wi * a1 * a3
                acc[4] += wi * a1 * a4; acc[5] += wi * a1 * a5; acc[6] += wi * a1 * a6
                acc[7] += wi * a2 * a2; acc[8] += wi * a2 * a3; acc[9] += wi * a2 * a4
                acc[10] += wi * a2 * a5; acc[11] += wi * a2 * a6; acc[12] += wi * a3 * a3
                acc[13] += wi * a3 * a4; acc[14] += wi * a3 * a5; acc[15] += wi * a3 * a6
                acc[16] += wi * a4 * a4; acc[17] += wi * a4 * a5; acc[18] += wi * a4 * a6
                acc[19] += wi * a5 * a5; acc[20] += wi * a5 * a6; acc[21] += wi * a6 * a6
                wz = wi * zi
                acc[22] += wz * a1; acc[23] += wz * a2; acc[24] += wz * a3
                acc[25] += wz * a4; acc[26] += wz * a5; acc[27] += wz * a6
            end
            i += nt
        end

        red = StaticArrays.MVector{27, Ta}(undef)
        for kk in 1:27
            smem[tid] = acc[kk]
            @synchronize()
            tree_sum!(@context, smem, tid, nt)
            red[kk] = smem[1]
            @synchronize()
        end

        if tid == Int32(1)
            m = Int32(np[blob])
            kidx = 0
            for jj in 1:6
                for jj2 in jj:6
                    kidx += 1
                    if jj <= m && jj2 <= m
                        AtWA[jj, jj2, blob] = red[kidx]
                        AtWA[jj2, jj, blob] = red[kidx]
                    end
                end
            end
            for jj in 1:6
                jj <= m && (AtWz[jj, 1, blob] = red[21 + jj])
            end
            for jj in (Int(m) + 1):6
                AtWA[jj, jj, blob] = one(Ta)
            end
        end
    end
end

# Apply solution + convergence/validity check — one thread per blob.
# Mirrors legacy: LU failure invalidates; !is_peak or aspect violation
# reverts to θ_old and stops; otherwise apply + rtol convergence check.
@kernel inbounds = true cpu = false unsafe_indices = true function _apply_check_ka!(
        theta, @Const(theta_old), @Const(AtWz), @Const(info), valid, converged,
        @Const(np), rtol, max_aspect_ratio_sq
    )
    blob = @index(Global, Linear)
    if blob <= size(theta, 2) && valid[blob] && converged[blob] == Int32(0)
        Ta = eltype(theta)
        if info[blob] != Int32(0)
            valid[blob] = false
        else
            m = np[blob]
            if !_shape_ok(AtWz[4, 1, blob], AtWz[5, 1, blob], AtWz[6, 1, blob], max_aspect_ratio_sq)
                for j in 1:6
                    theta[j, blob] = theta_old[j, blob]
                end
                converged[blob] = Int32(1)
            else
                for j in 1:Int(m)
                    theta[j, blob] = AtWz[j, 1, blob]
                end
                diff_sq = zero(Ta)
                norm_sq = zero(Ta)
                for j in 1:Int(m)
                    d = theta[j, blob] - theta_old[j, blob]
                    diff_sq += d * d
                    norm_sq += theta[j, blob] * theta[j, blob]
                end
                if sqrt(diff_sq) / (sqrt(norm_sq) + Ta(1.0e-300)) < rtol
                    converged[blob] = Int32(1)
                end
            end
        end
    end
end

# Fused per-iteration update: residuals → weighted scale ŝ² → Newton ν
# (5 steps, E-step weights from pin) → soft weights. The legacy thread-1
# `break` past the barrier is replaced by the `sflag` broadcast — identical
# stop conditions, identical ν, well-defined barriers on every backend.
# PAIRED COPY (2026-09 audit): the residual + soft-weight arithmetic lives in
# _residuals_ka!/_soft_weights_ka! (pre-loop init) AND re-inlined in
# _fused_ka! (the per-iteration kernel). Verified not-drifted; any change to
# the shared atoms' call shape must land in both. Deliberately NOT collapsed:
# adapt is nondeterministic on CUDA, so a de-inline cannot be certified
# bitwise — only frame-level — and this is the quality-critical path.
@kernel inbounds = true cpu = false unsafe_indices = true function _fused_ka!(
        r2, w, pin, s_hat_sq, nu,
        @Const(A), @Const(v), @Const(theta), @Const(bg), @Const(outlier_halfwidth),
        @Const(cnt), @Const(valid), @Const(offsets), spatial::Bool,
        @Const(converged), lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    blob = @index(Group, Linear)
    tid = Int32(@index(Local, Linear))
    Ta = eltype(theta)
    smem_a = @localmem eltype(theta) (256,)
    smem_b = @localmem eltype(theta) (256,)
    smem_c = @localmem eltype(theta) (256,)
    sflag = @localmem Int32 (1,)
    if valid[blob] && converged[blob] == Int32(0)
        n = cnt[blob]
        base = offsets[blob]
        bg_val = bg[blob]

        # --- Phase 1: residuals ---
        i = tid
        while i <= n
            gid = base + i
            z_pred = Ta(A[gid, 1]) * theta[1, blob] + Ta(A[gid, 2]) * theta[2, blob] +
                Ta(A[gid, 3]) * theta[3, blob] + Ta(A[gid, 4]) * theta[4, blob] +
                Ta(A[gid, 5]) * theta[5, blob] + Ta(A[gid, 6]) * theta[6, blob]
            r2[gid] = eltype(r2)(_adapt_r2(bg_val - Ta(v[gid]), z_pred))
            i += nt
        end
        @synchronize()

        # --- Phase 2: weighted scale ŝ² ---
        local_wr2 = zero(Ta)
        local_w = zero(Ta)
        i = tid
        while i <= n
            gid = base + i
            # Product in STORAGE precision, then widen — legacy parity: the
            # CUDA kernel multiplied w*r2 as stored (Float32 by default)
            # before accumulating in Ta. Widening the factors first changes
            # the rounding and drifts ŝ² off the legacy trajectory.
            local_wr2 += Ta(w[gid] * r2[gid])
            local_w += Ta(w[gid])
            i += nt
        end
        smem_a[tid] = local_wr2
        smem_b[tid] = local_w
        @synchronize()
        s = Int32(128)
        while s >= Int32(1)
            if tid <= s && tid + s <= nt
                smem_a[tid] += smem_a[tid + s]
                smem_b[tid] += smem_b[tid + s]
            end
            @synchronize()
            s >>= Int32(1)
        end
        if tid == Int32(1)
            s_hat_sq[blob] = max(smem_a[1] / max(smem_b[1], Ta(1.0e-30)), Ta(1.0e-30))
        end
        @synchronize()

        # --- Phase 3: Newton ν (5 steps, fixed E-step weights from pin) ---
        ν = nu[blob]
        s2 = max(s_hat_sq[blob], Ta(1.0e-30))

        iter = 1
        while iter <= 5
            ψ_half_νp1 = digamma_stirling((ν + Ta(1.0)) / Ta(2.0))
            ψ_half_ν = digamma_stirling(ν / Ta(2.0))
            ψ1_half_νp1 = trigamma_stirling((ν + Ta(1.0)) / Ta(2.0))
            ψ1_half_ν = trigamma_stirling(ν / Ta(2.0))
            inv_ν = Ta(1.0) / ν
            inv_ν2 = inv_ν * inv_ν

            local_dQ = zero(Ta)
            local_d2Q = zero(Ta)
            local_w = zero(Ta)
            i = tid
            while i <= n
                gid = base + i
                wi = Ta(pin[gid])
                if wi >= Ta(1.0e-15)
                    dq_i, d2q_data_i = _nu_point_terms(
                        Ta(r2[gid]) / (ν * s2), ν, inv_ν, inv_ν2, ψ_half_νp1, ψ_half_ν
                    )
                    local_dQ += wi * dq_i
                    local_w += wi
                    local_d2Q += wi * d2q_data_i
                end
                i += nt
            end
            smem_a[tid] = local_dQ
            smem_b[tid] = local_d2Q
            smem_c[tid] = local_w
            @synchronize()
            s = Int32(128)
            while s >= Int32(1)
                if tid <= s && tid + s <= nt
                    smem_a[tid] += smem_a[tid + s]
                    smem_b[tid] += smem_b[tid + s]
                    smem_c[tid] += smem_c[tid + s]
                end
                @synchronize()
                s >>= Int32(1)
            end
            if tid == Int32(1)
                ν_new, stop_b = _next_nu(
                    ν, smem_a[1], smem_b[1], smem_c[1], ψ1_half_νp1, ψ1_half_ν, inv_ν2
                )
                if ν_new != ν
                    ν = ν_new
                    nu[blob] = ν
                end
                sflag[1] = stop_b ? Int32(1) : Int32(0)
            end
            @synchronize()
            ν = nu[blob]
            sflag[1] == Int32(1) && break
            iter += 1
        end

        # --- Phase 4: soft weights + spatial (stores pin for next ν M-step) ---
        s2 = max(s_hat_sq[blob], Ta(1.0e-30))
        log_two_outlier_halfwidth = log(Ta(2.0) * outlier_halfwidth[blob])
        half_νp1 = (ν + Ta(1.0)) / Ta(2.0)
        log_C = _weight_logC(ν, s2, lgamma_f)
        log_pout = -log_two_outlier_halfwidth

        i = tid
        while i <= n
            gid = base + i
            p_in, wi = _posterior_weight(Ta(r2[gid]), ν, s2, half_νp1, log_C, log_pout)
            pin[gid] = p_in
            if spatial
                q = theta[4, blob] * Ta(A[gid, 4]) +
                    theta[5, blob] * Ta(A[gid, 5]) +
                    theta[6, blob] * Ta(A[gid, 6])
                wi *= exp(q)
            end
            w[gid] = wi
            i += nt
        end
    end
end

# =============================================================================
# Driver — any GPU backend
# =============================================================================

_ka_upload(device, x::AbstractArray{T}) where {T} = begin
    d = KA.allocate(device, T, size(x)...)
    copyto!(d, x)
    d
end

function _adapt_ka(
        img, sites::AbstractVector{S}, cfg::AdaptConfig,
        device::KA.Backend
    ) where {S <: Union{Disc, EllipticFrame}}
    device isa KA.CPU && return _adapt_cpu(img, sites, cfg)
    return _adapt_ka_gpu(img, sites, cfg, device)
end

function _adapt_ka_gpu(
        img, sites::AbstractVector{S}, cfg::AdaptConfig,
        device::KA.Backend
    ) where {S <: Union{Disc, EllipticFrame}}
    N = length(sites)
    N == 0 && return BlobDetection{Float64}[]

    # Accumulation/solve precision by backend capability: Float64 on
    # CUDA/CPU (identical to the legacy kernels), Float32 on Metal.
    Ta = KA.supports_float64(device) ? Float64 : Float32
    lgamma_f = _device_lgamma(device)

    cutoff = cfg.cutoff
    max_irls_iter = cfg.max_irls_iter
    spatial_weighting = cfg.spatial_weighting

    # The kernels promote per-pixel loads to Ta, so the image stays at its
    # native storage: a device-resident array is used AS IS (the pipeline
    # hands over Float32 — reading it directly is bit-identical to the
    # legacy Float64 staging, which only widened exact Float32 values); a
    # host image uploads once at Float32 (or Ta when it is already Float64
    # and the backend can hold it — the legacy precision).
    d_img = if KA.get_backend(img) == device
        img
    else
        hostf = gray_f32(img)
        Tp = eltype(hostf) <: Gray ? eltype(eltype(hostf)) : eltype(hostf)
        Tu = Tp == Float64 ? Ta : Float32
        _ka_upload(device, Tu.(Gray{Tu}.(hostf)))
    end
    H, W = size(d_img)
    sat = saturation_bounds(Float64)   # float regime — same (1e-6, 1-1e-6) as legacy

    # --- Host: per-blob metadata (identical to legacy) ---
    hw_vec = Vector{Int32}(undef, N)
    spacing_vec = Vector{Float64}(undef, N)
    cx_vec = Vector{Float64}(undef, N)
    cy_vec = Vector{Float64}(undef, N)
    σ_vec = Vector{Float64}(undef, N)
    valid = trues(N)
    max_scale_px = _max_scale_px(cfg, W, H)

    for i in 1:N
        cx, cy = Float64.(centroid(sites[i]))
        σ = characteristic_scale(sites[i])
        hw_i, spacing_i = _scan_grid(cutoff * cfg.scan_padding * σ)
        hw = Int32(hw_i)
        cx_vec[i] = cx; cy_vec[i] = cy; hw_vec[i] = hw
        spacing_vec[i] = spacing_i; σ_vec[i] = σ

        σ > max_scale_px && (valid[i] = false; continue)

        mr = cutoff * σ
        clip_x = max(0.0, mr - cx + 1.0, cx + mr - W)
        clip_y = max(0.0, mr - cy + 1.0, cy + mr - H)
        if (clip_x + clip_y) * mr / (π * mr^2) > cfg.max_clip_ratio
            valid[i] = false
        end
    end
    for i in 1:N
        valid[i] || (hw_vec[i] = Int32(0))
    end

    # --- Flat-packed buffer layout ---
    n_alloc = [valid[i] ? Int32((2 * hw_vec[i] + 1)^2) : Int32(0) for i in 1:N]
    offsets = Vector{Int32}(undef, N + 1)
    offsets[1] = Int32(0)
    for i in 1:N
        offsets[i + 1] = offsets[i] + n_alloc[i]
    end
    total_pixels = Int(offsets[N + 1])
    total_pixels == 0 && return BlobDetection{Float64}[]


    d_cx = _ka_upload(device, Ta.(cx_vec))
    d_cy = _ka_upload(device, Ta.(cy_vec))
    d_hw = _ka_upload(device, hw_vec)
    d_spacing = _ka_upload(device, Ta.(spacing_vec))
    d_valid = _ka_upload(device, collect(valid))
    d_offsets = _ka_upload(device, offsets)

    # Per-pixel staging at Float32 — the only value AdaptConfig.store ever
    # took before its 2026-09 deletion; kernels promote to Ta on every read.
    d_A = KA.zeros(device, Float32, total_pixels, 6)
    d_v = KA.zeros(device, Float32, total_pixels)
    d_f_obs = KA.zeros(device, Float32, total_pixels)
    d_cnt = KA.zeros(device, Int32, N)
    d_bg = KA.zeros(device, Ta, N)
    KA.synchronize(device)

    _scan_ka!(device, 256)(
        d_A, d_f_obs, d_v, d_cnt, d_bg,
        d_img, d_cx, d_cy, d_hw, d_spacing, d_valid, d_offsets,
        Int32(H), Int32(W), Ta(sat[1]), Ta(sat[2]), Ta(cfg.min_intensity);
        ndrange = 256 * N
    )
    KA.synchronize(device)

    # --- n_params per blob ---
    counts_h = Array(d_cnt)
    np_vec = zeros(Int32, N)
    for i in 1:N
        valid[i] || continue
        n = Int(counts_h[i])
        n < _MIN_POINTS_FIXED_CENTER && (valid[i] = false; continue)
        np_vec[i] = Int32(
            n < _MIN_POINTS_ISOTROPIC ? 2 :
                n < _MIN_POINTS_ANISOTROPIC ? 4 : 6
        )
    end
    d_np = _ka_upload(device, np_vec)
    d_valid_gpu = _ka_upload(device, collect(valid))

    # --- θ₀ seed (host, cheap); per-blob max(f_obs) reduced on device ---
    d_maxf = KA.zeros(device, Ta, N)
    _segmented_maxf_ka!(device, 256)(
        d_maxf, d_f_obs, d_cnt, d_valid_gpu, d_offsets; ndrange = 256 * N
    )
    KA.synchronize(device)
    maxf_h = Array(d_maxf)
    theta_h = zeros(Ta, 6, N)
    for i in 1:N
        valid[i] || continue
        max_f = Float64(maxf_h[i])
        z_max = max_f > 0 ? log(max_f) : 0.0
        c = coeffs(_site_seed6(sites[i], z_max))
        for j in 1:6
            theta_h[j, i] = Ta(c[j])
        end
    end
    d_theta = _ka_upload(device, theta_h)
    KA.synchronize(device)

    # --- IRLS buffers (NB no f_hat/r — write-only in the legacy kernels) ---
    d_r2 = KA.zeros(device, Float32, total_pixels)
    d_w = KA.allocate(device, Float32, total_pixels); fill!(d_w, one(Float32))
    d_pin = KA.allocate(device, Float32, total_pixels); fill!(d_pin, one(Float32))

    d_nu = KA.allocate(device, Ta, N); fill!(d_nu, Ta(4.0))
    d_s_hat_sq = KA.zeros(device, Ta, N)
    d_ohw = KA.zeros(device, Ta, N)

    d_AtWA = KA.zeros(device, Ta, 6, 6, N)
    d_AtWz = KA.zeros(device, Ta, 6, 1, N)
    d_pivot = KA.zeros(device, Int32, 6, N)
    d_info = KA.zeros(device, Int32, N)
    d_theta_old = KA.zeros(device, Ta, 6, N)
    d_converged = KA.zeros(device, Int32, N)
    KA.synchronize(device)

    # --- Initial residuals + fused sort/median/sweep + initial weights ---
    _residuals_ka!(device, 256)(
        d_r2, d_A, d_v, d_theta, d_bg, d_cnt, d_valid_gpu, d_offsets;
        ndrange = 256 * N
    )
    _sort_sweep_ka!(device, 256)(
        d_nu, d_s_hat_sq, d_ohw, d_r2, d_np, d_cnt, d_valid_gpu, d_offsets,
        lgamma_f; ndrange = 256 * N
    )
    _soft_weights_ka!(device, 256)(
        d_w, d_pin, d_r2, d_A, d_theta, d_nu, d_s_hat_sq, d_ohw,
        d_cnt, d_valid_gpu, d_offsets, spatial_weighting, lgamma_f;
        ndrange = 256 * N
    )
    KA.synchronize(device)

    # --- IRLS loop ---
    iters_done = 0
    for iter in 1:max_irls_iter
        iters_done = iter
        copyto!(d_theta_old, d_theta)

        d_AtWA .= zero(Ta)
        d_AtWz .= zero(Ta)
        _accumulate_ka!(device, 256)(
            d_AtWA, d_AtWz, d_A, d_v, d_w, d_bg,
            d_cnt, d_valid_gpu, d_offsets, d_np, d_converged;
            ndrange = 256 * N
        )

        _solve_6x6_batched!(
            device, d_AtWA, d_AtWz, d_pivot, d_info,
            d_valid_gpu, d_converged
        )

        _apply_check_ka!(device, 256)(
            d_theta, d_theta_old, d_AtWz, d_info, d_valid_gpu, d_converged,
            d_np, Ta(cfg.rtol), Ta(cfg.max_aspect_ratio^2);
            ndrange = 256 * cld(N, 256)
        )

        _fused_ka!(device, 256)(
            d_r2, d_w, d_pin, d_s_hat_sq, d_nu,
            d_A, d_v, d_theta, d_bg, d_ohw,
            d_cnt, d_valid_gpu, d_offsets, spatial_weighting,
            d_converged, lgamma_f;
            ndrange = 256 * N
        )

        # All-converged early exit (two tiny device reductions).
        n_active = Int(sum(d_valid_gpu .& (d_converged .== Int32(0))))
        n_active <= 0 && break
    end
    KA.synchronize(device)

    # --- Download + convert ---
    valid_gpu = Array(d_valid_gpu)
    for i in 1:N
        valid[i] = valid[i] && valid_gpu[i]
    end
    theta_out = Array(d_theta)
    frames = _finalize_frames(sites, valid, theta_out, σ_vec, cfg)
    return frames
end

# Shared finalize: θ → Paraboloid3D → EllipticFrame with the legacy gates.
function _finalize_frames(sites, valid, theta_out::AbstractMatrix, σ_vec, cfg::AdaptConfig)
    frames = BlobDetection{Float64}[]
    for i in eachindex(sites)
        valid[i] || continue
        θ = SVector{6, Float64}(Float64(theta_out[j, i]) for j in 1:6)
        p = Paraboloid3D(θ)
        is_peak(p) || continue
        Σ = moment_matrix(p)
        is_valid_shape(Symmetric(Σ)) || continue

        cx, cy = Float64.(centroid(sites[i]))
        σ = σ_vec[i]
        frame = EllipticFrame(Point2(cx, cy), p)
        sr = characteristic_scale(frame) / σ
        (sr > cfg.max_scale_ratio || sr < 1 / cfg.max_scale_ratio) && continue
        push!(frames, Detection(frame, (; id = i)))
    end
    return frames
end

# =============================================================================
# CPU implementation — same math, per-blob threaded host loops
# =============================================================================
#
# Deterministic (pixel order is the box scan order; the sweep sorts, so it
# is order-canonical anyway). Uses SpecialFunctions.loggamma. Per-pixel
# quantities round through Float32 (the deleted `AdaptConfig.store`'s only value) the GPU
# buffers use, so CPU and GPU agree to summation-order ulps.

function _adapt_cpu(
        img, sites::AbstractVector{S},
        cfg::AdaptConfig
    ) where {S <: Union{Disc, EllipticFrame}}
    N = length(sites)
    N == 0 && return BlobDetection{Float64}[]
    imgf = gray_f32(img)
    H, W = size(imgf)
    sat_lo, sat_hi = saturation_bounds(Float64)
    max_scale_px = _max_scale_px(cfg, W, H)

    thetas = zeros(Float64, 6, N)
    valid = fill(false, N)   # Vector{Bool}, NOT BitVector: parallel writes below
    σ_vec = zeros(Float64, N)
    Threads.@threads for i in 1:N
        ok = _adapt_cpu_one!(
            view(thetas, :, i), imgf, sites[i], cfg,
            H, W, sat_lo, sat_hi, max_scale_px
        )
        valid[i] = ok
        σ_vec[i] = characteristic_scale(sites[i])
    end
    return _finalize_frames(sites, valid, thetas, σ_vec, cfg)
end

# One blob: scan → seed → sweep → IRLS. Writes the fitted θ into `θ_out`,
# returns whether the blob survived the validity gates.
function _adapt_cpu_one!(
        θ_out, imgf, site, cfg::AdaptConfig,
        H, W, sat_lo, sat_hi, max_scale_px
    )
    cutoff = cfg.cutoff
    cx, cy = Float64.(centroid(site))
    σ = characteristic_scale(site)
    hw, spacing = _scan_grid(cutoff * cfg.scan_padding * σ)

    σ > max_scale_px && return false
    mr = cutoff * σ
    clip_x = max(0.0, mr - cx + 1.0, cx + mr - W)
    clip_y = max(0.0, mr - cy + 1.0, cy + mr - H)
    (clip_x + clip_y) * mr / (π * mr^2) > cfg.max_clip_ratio && return false

    # Canonical grid, anchored on the same integer pixel the pixel scan used
    # (`spacing == 1` ⇒ integer positions ⇒ `bilinear_sample_clamped` returns them exactly).
    ax = Float64(unsafe_trunc(Int32, floor(cx)))
    ay = Float64(unsafe_trunc(Int32, floor(cy)))
    Hi = Int32(H); Wi = Int32(W)
    _gx(g) = ax + (g - hw) * spacing
    _gy(g) = ay + (g - hw) * spacing

    # Stage 1: background = max unsaturated intensity.
    background = -Inf
    for gx in 0:(2hw), gy in 0:(2hw)
        ok, val = bilinear_sample_clamped(Float64, imgf, _gy(gy), _gx(gx), Hi, Wi)
        (ok && val >= sat_lo && val <= sat_hi && val > background) && (background = val)
    end
    background <= 0.0 && return false

    # Stage 2: collect valid pixels (design entries + intensities stored at
    # T like the GPU buffers, promoted on every read).
    min_int = cfg.min_intensity
    a2v = Float32[]; a3v = Float32[]; a4v = Float32[]; a5v = Float32[]; a6v = Float32[]
    vv = Float32[]; fv = Float32[]
    for gy in 0:(2hw), gx in 0:(2hw)
        xf = _gx(gx); yf = _gy(gy)
        ok, val = bilinear_sample_clamped(Float64, imgf, yf, xf, Hi, Wi)
        ok || continue
        (val < sat_lo || val > sat_hi) && continue
        f = background - val
        f <= min_int && continue
        # Image-pixel offsets regardless of `spacing` — no rescale downstream.
        dx = xf - 0.5 - cx
        dy = yf - 0.5 - cy
        push!(a2v, Float32(dx)); push!(a3v, Float32(dy))
        push!(a4v, Float32(dx * dx)); push!(a5v, Float32(2.0 * dx * dy)); push!(a6v, Float32(dy * dy))
        push!(vv, Float32(val)); push!(fv, Float32(f))
    end
    n = length(fv)
    n < _MIN_POINTS_FIXED_CENTER && return false
    np = n < _MIN_POINTS_ISOTROPIC ? 2 : n < _MIN_POINTS_ANISOTROPIC ? 4 : 6

    # θ₀ seed from the max inverted intensity.
    max_f = Float64(maximum(fv))
    z_max = max_f > 0 ? log(max_f) : 0.0
    θ = StaticArrays.MVector{6, Float64}(coeffs(_site_seed6(site, z_max))...)

    r2 = zeros(Float32, n)
    _cpu_residuals!(r2, θ, a2v, a3v, a4v, a5v, a6v, vv, background)

    # Sort + outlier_halfwidth median + marginal sweep (RSDR Jeffreys algebra).
    sorted = sort(Float64.(r2))
    outlier_halfwidth = max(sqrt(sorted[cld(n, 2)]), 1.0e-30)
    m = np + 1
    log_two_outlier_halfwidth = log(2.0 * outlier_halfwidth)
    RSS = 0.0
    best_S = -1.0e30
    best_k = m + 1
    for kk in 1:n
        RSS += sorted[kk]
        kk <= m && continue
        S = _sweep_score(kk - m, RSS, Float64(n - kk) * log_two_outlier_halfwidth, loggamma)
        if S > best_S
            best_S = S
            best_k = kk
        end
    end
    ν = max(Float64(best_k - m), 2.0)
    s2 = max(sum(@view(sorted[1:best_k])) / ν, 1.0e-30)

    w = ones(Float32, n)
    pin = ones(Float32, n)
    _cpu_weights!(w, pin, r2, θ, a4v, a5v, a6v, ν, s2, outlier_halfwidth, cfg.spatial_weighting)

    # IRLS loop.
    A6 = StaticArrays.MMatrix{6, 6, Float64}(undef)
    b6 = StaticArrays.MVector{6, Float64}(undef)
    θ_old = StaticArrays.MVector{6, Float64}(undef)
    max_aspect_sq = cfg.max_aspect_ratio^2
    for _ in 1:cfg.max_irls_iter
        θ_old .= θ

        # Accumulate A'WA / A'Wz with W = w·f² (Float64, pixel order).
        acc = zeros(Float64, 27)
        for i in 1:n
            fi = background - Float64(vv[i])
            fi < 1.0e-10 && continue
            zi = log(fi)
            wi = Float64(w[i]) * fi * fi
            a1 = 1.0; a2 = Float64(a2v[i]); a3 = Float64(a3v[i])
            a4 = Float64(a4v[i]); a5 = Float64(a5v[i]); a6 = Float64(a6v[i])
            acc[1] += wi * a1 * a1; acc[2] += wi * a1 * a2; acc[3] += wi * a1 * a3
            acc[4] += wi * a1 * a4; acc[5] += wi * a1 * a5; acc[6] += wi * a1 * a6
            acc[7] += wi * a2 * a2; acc[8] += wi * a2 * a3; acc[9] += wi * a2 * a4
            acc[10] += wi * a2 * a5; acc[11] += wi * a2 * a6; acc[12] += wi * a3 * a3
            acc[13] += wi * a3 * a4; acc[14] += wi * a3 * a5; acc[15] += wi * a3 * a6
            acc[16] += wi * a4 * a4; acc[17] += wi * a4 * a5; acc[18] += wi * a4 * a6
            acc[19] += wi * a5 * a5; acc[20] += wi * a5 * a6; acc[21] += wi * a6 * a6
            wz = wi * zi
            acc[22] += wz * a1; acc[23] += wz * a2; acc[24] += wz * a3
            acc[25] += wz * a4; acc[26] += wz * a5; acc[27] += wz * a6
        end
        fill!(A6, 0.0)
        fill!(b6, 0.0)
        kidx = 0
        for j in 1:6, j2 in j:6
            kidx += 1
            if j <= np && j2 <= np
                A6[j, j2] = acc[kidx]
                A6[j2, j] = acc[kidx]
            end
        end
        for j in 1:np
            b6[j] = acc[21 + j]
        end
        for j in (np + 1):6
            A6[j, j] = 1.0
        end

        info = lu_solve!(A6, b6)
        info != Int32(0) && return false

        # Apply + check (mirrors _apply_check_ka!).
        if !_shape_ok(b6[4], b6[5], b6[6], max_aspect_sq)
            θ .= θ_old
            break
        end
        for j in 1:np
            θ[j] = b6[j]
        end
        diff_sq = 0.0
        norm_sq = 0.0
        for j in 1:np
            d = θ[j] - θ_old[j]
            diff_sq += d * d
            norm_sq += θ[j] * θ[j]
        end
        # rtol convergence stops HERE, like the GPU path: the fused update is
        # skipped for converged blobs there, and nothing after this point
        # affects θ.
        sqrt(diff_sq) / (sqrt(norm_sq) + 1.0e-300) < cfg.rtol && break

        # Fused update: residuals → ŝ² → Newton ν → weights.
        _cpu_residuals!(r2, θ, a2v, a3v, a4v, a5v, a6v, vv, background)
        swr2 = 0.0
        sw = 0.0
        for i in 1:n
            swr2 += Float64(w[i]) * Float64(r2[i])
            sw += Float64(w[i])
        end
        s2 = max(swr2 / max(sw, 1.0e-30), 1.0e-30)
        for _ in 1:5
            ψp = digamma_stirling((ν + 1.0) / 2.0)
            ψ = digamma_stirling(ν / 2.0)
            ψ1p = trigamma_stirling((ν + 1.0) / 2.0)
            ψ1 = trigamma_stirling(ν / 2.0)
            inv_ν = 1.0 / ν
            inv_ν2 = inv_ν * inv_ν
            dQ = 0.0; d2Q_data = 0.0; sum_w = 0.0
            for i in 1:n
                wi = Float64(pin[i])
                wi < 1.0e-15 && continue
                dq_i, d2q_data_i = _nu_point_terms(Float64(r2[i]) / (ν * s2), ν, inv_ν, inv_ν2, ψp, ψ)
                dQ += wi * dq_i
                sum_w += wi
                d2Q_data += wi * d2q_data_i
            end
            ν_new, stop = _next_nu(ν, dQ, d2Q_data, sum_w, ψ1p, ψ1, inv_ν2)
            ν = ν_new
            stop && break
        end
        _cpu_weights!(w, pin, r2, θ, a4v, a5v, a6v, ν, s2, outlier_halfwidth, cfg.spatial_weighting)
    end

    θ_out .= θ
    return true
end

function _cpu_residuals!(r2, θ, a2v, a3v, a4v, a5v, a6v, vv, background)
    @inbounds for i in eachindex(r2)
        z_pred = θ[1] + Float64(a2v[i]) * θ[2] + Float64(a3v[i]) * θ[3] +
            Float64(a4v[i]) * θ[4] + Float64(a5v[i]) * θ[5] + Float64(a6v[i]) * θ[6]
        r2[i] = eltype(r2)(_adapt_r2(background - Float64(vv[i]), z_pred))
    end
    return r2
end

function _cpu_weights!(w, pin, r2, θ, a4v, a5v, a6v, ν, s2, outlier_halfwidth, spatial)
    T = eltype(w)
    s2 = max(s2, 1.0e-30)                      # kernel parity: the device path clamps on entry too
    half_νp1 = (ν + 1.0) / 2.0
    log_C = _weight_logC(ν, s2, loggamma)
    log_pout = -log(2.0 * outlier_halfwidth)
    @inbounds for i in eachindex(w)
        p_in, wi = _posterior_weight(Float64(r2[i]), ν, s2, half_νp1, log_C, log_pout)
        pin[i] = T(p_in)
        if spatial
            q = θ[4] * Float64(a4v[i]) + θ[5] * Float64(a5v[i]) + θ[6] * Float64(a6v[i])
            wi *= exp(q)
        end
        w[i] = T(wi)
    end
    return w
end
