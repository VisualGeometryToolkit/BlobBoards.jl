# =============================================================================
# Backend-generic blob adaptation — log-space posterior IRLS on any KA device
# =============================================================================
#
# Port of the CUDA implementation (formerly ext/adapt_cuda.jl) to pure
# KernelAbstractions, so `adapt` runs on CPU, CUDA, Metal, or ROCm.
#
# Design (each point audited against the legacy kernels, 2026-08-10):
#
# - ONE implementation on every backend: the per-blob kernels below run on
#   the CPU too (`KA.CPU()`), a blob one workgroup of `_adapt_groupsize`
#   work items — 256 on a GPU, one on the CPU, where the work item walks its
#   blob's pixels in scan order (see "The per-blob kernels"). The threaded
#   host twin of the same math (`_adapt_cpu`) that the CPU used to run is
#   deleted (2026-11, the 0.9 duplication audit).
#
# - Accumulation precision `Ta` dispatches on `KA.supports_float64(device)`:
#   Float64 on CUDA/CPU (arithmetic identical to the legacy kernels),
#   Float32 on Metal (which has no Float64 at all — a DISTINCT numeric
#   regime, never claimed as parity). Per-pixel storage stays
#   Float32 staging (the deleted `AdaptConfig.store`'s only value) exactly as before.
#
# - ONE lgamma on every backend: Core's `lgamma_lanczos` (device-safe; the
#   owner's ruling of 2026-11-11 retired SpecialFunctions' `loggamma` on the CPU
#   and CUDA's `__nv_lgamma`, which made the same kernel give different numbers
#   per backend).
#
# - ONE 6×6 normal-equation solve on every backend: the per-blob register LU
#   with partial pivoting (`_lu6_kernel!`; the same ruling retired CUDA's cuBLAS
#   getrf/getrs override).
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
#   ORDER is scheduler-dependent on a GPU — exactly as on legacy CUDA, whose
#   output already differs in final ulps between identical runs. Parity
#   testing is therefore frame-level with tolerances, never bitwise. On the
#   CPU (one work item per blob) the order is the scan order, run to run.
# =============================================================================

using KernelAbstractions: @kernel, @Const, @index, @groupsize,
    @localmem, @private, @synchronize, @uniform

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

# The clamped bilinear sample, the device-safe lgamma/digamma/trigamma series
# and the in-register 6×6 LU are VisualGeometryToolkitCore's since 0.21.1
# (hoisted verbatim from here — same op sequences), and so is the one upload
# to a device (`ka_upload`). Core's workgroup `tree_sum!` is not called: a
# barrier inside a helper does not run on KA's CPU backend, so the kernels
# spell their reductions inline (see "The per-blob kernels").
using VisualGeometryToolkitCore: bilinear_sample_clamped, lgamma_lanczos, digamma_stirling,
    trigamma_stirling, lu_solve!, ka_upload, on_device
# The E-step is Core's Student-t + uniform mixture, built in the kernel's float
# type with the kernels' lgamma (one per blob per iteration: ν and ŝ² change per
# blob); the apply-time shape gate is Core's closed-form 2×2 peak and aspect test.
using VisualGeometryToolkitCore: StudentTMixture, inlier_posterior, posterior_weight, is_peak, is_valid_shape
# The ν M-step is Core's: a point's Q-derivative terms and the safeguarded Newton
# update from the reduced sums (Newton only where Q is concave).
using VisualGeometryToolkitCore: nu_point_terms, next_nu_from_sums
# The sweep's per-partition score is VisualGeometryToolkitCore's one formula.
using VisualGeometryToolkitCore: IGPrior, partition_score
const _JEFFREYS = IGPrior()


# =============================================================================
# The per-blob 6×6 solve
# =============================================================================

# The per-blob 6×6 solve, on every backend: θ_new replaces `AtWz[:, 1, blob]`;
# `info[blob] ≠ 0` flags a singular system (the apply/check step invalidates
# those blobs). One work item per blob (invalid/converged blobs skipped: the
# apply step ignores them).
@kernel inbounds = true unsafe_indices = true function _lu6_kernel!(
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

# =============================================================================
# The per-blob kernels — one workgroup per blob, on every backend
# =============================================================================
#
# ONE implementation: the same kernels run on the CPU (`KA.CPU()`) and on any
# GPU backend. A blob is one workgroup of `_adapt_groupsize(backend)` work
# items: 256 on a GPU (the pixel loops strided over the group, group-shared
# reductions and a bitonic sort), ONE on the CPU, where the workgroup is a
# single work item that walks its blob's pixels in scan order and the
# reductions are the identity (KA's CPU backend spreads the blobs over the
# threads). The group size 1 makes the CPU path's pixel order, sums and sort
# sequential, as the threaded host loops it replaced had them.
#
# What KA's CPU transform requires of a kernel with barriers, and every kernel
# below keeps:
# - every `@synchronize` at the kernel body's top level or inside a loop whose
#   bounds are `@uniform` (the reduction strides, the sort's stages, the
#   ν-Newton steps) — never inside a branch on a work item's own value. A blob
#   that is invalid (or converged) runs the barriers with no work, and each
#   segment between barriers re-reads what it needs (`_blob_span`);
# - a `@uniform` loop counter is updated at the START of its loop body (the
#   CPU transform drops a `@uniform` statement left after the body's last
#   barrier);
# - a value a work item carries across a barrier is `@private` (the 27
#   normal-equation accumulators) or re-read from memory.
#
# The reductions guard `tid + s ≤ nt`, so a group smaller than the 256-slot
# shared buffers reads no slot nobody wrote.

"""
    _adapt_groupsize(backend) -> Int

Work items per blob of the adaptation kernels: 256 on a GPU backend, one on
the CPU (see "The per-blob kernels" in src/adapt_ka.jl).
"""
_adapt_groupsize(::KA.Backend) = 256
_adapt_groupsize(::KA.CPU) = 1

# A blob's pixel count and flat-buffer base; no pixels for a blob that is
# not live (invalid, or — where `converged` is given — converged).
@inline _blob_span(cnt, valid, offsets, blob) =
    (valid[blob] ? cnt[blob] : Int32(0), offsets[blob])
@inline _blob_span(cnt, valid, converged, offsets, blob) =
    (valid[blob] && converged[blob] == Int32(0) ? cnt[blob] : Int32(0), offsets[blob])

# The canonical scan grid of a blob: its anchor (the integer pixel the pixel
# scan used, so `st == 1` lands exactly on those pixels), its sample spacing,
# halfwidth and side, and the image position of grid point `i`.
@inline function _scan_layout(::Type{Ta}, cx, cy, hw, spacing, blob) where {Ta}
    bhw = hw[blob]
    side = Int32(2) * bhw + Int32(1)
    return floor(cx[blob]), floor(cy[blob]), Ta(spacing[blob]), bhw, side
end
@inline function _grid_point(::Type{Ta}, i, ax, ay, st, bhw, side) where {Ta}
    gx = (i - Int32(1)) % side
    gy = (i - Int32(1)) ÷ side
    return ax + Ta(gx - bhw) * st, ay + Ta(gy - bhw) * st
end

# Stage 1: per-blob background (max unsaturated intensity, shared-memory max
# reduction). Stage 2: collect valid pixels into the flat buffers, slots
# assigned by an atomic counter (order scheduler-dependent on a GPU, as on
# legacy; the scan order on the CPU).
@kernel inbounds = true unsafe_indices = true function _scan_ka!(
        A, f_obs, v, cnt, bg,
        @Const(img), @Const(cx), @Const(cy), @Const(hw), @Const(spacing), @Const(valid),
        @Const(offsets), H::Int32, W::Int32, sat_lo, sat_hi, min_int
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(bg)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    smem = @localmem eltype(bg) (256,)

    local_max = typemin(Ta)
    if valid[blob]
        ax, ay, st, bhw, side = _scan_layout(Ta, cx, cy, hw, spacing, blob)
        i = Int32(tid)
        while i <= side * side
            xf, yf = _grid_point(Ta, i, ax, ay, st, bhw, side)
            ok, val = bilinear_sample_clamped(Ta, img, yf, xf, H, W)
            if ok && val >= sat_lo && val <= sat_hi && val > local_max
                local_max = val
            end
            i += nt
        end
    end
    smem[tid] = local_max
    @synchronize()
    @uniform s = Int32(256)
    while s > Int32(1)
        @uniform s = s >> Int32(1)
        if tid <= s && tid + s <= nt
            smem[tid] = max(smem[tid], smem[tid + s])
        end
        @synchronize()
    end
    if tid == 1 && valid[blob]
        bg[blob] = smem[1]
        cnt[blob] = Int32(0)
    end
    @synchronize()

    background = smem[1]
    if valid[blob] && background > zero(Ta)
        ax, ay, st, bhw, side = _scan_layout(Ta, cx, cy, hw, spacing, blob)
        base = offsets[blob]
        bx = cx[blob]
        by = cy[blob]
        i = Int32(tid)
        while i <= side * side
            xf, yf = _grid_point(Ta, i, ax, ay, st, bhw, side)
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

# Per-blob max(f_obs) — strided shared-memory reduction, one group per blob.
@kernel inbounds = true unsafe_indices = true function _segmented_maxf_ka!(
        maxf, @Const(f_obs), @Const(cnt), @Const(valid), @Const(offsets)
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(maxf)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    smem = @localmem eltype(maxf) (256,)

    n, base = _blob_span(cnt, valid, offsets, blob)
    m = typemin(Ta)
    j = Int32(tid)
    while j <= n
        fv = Ta(f_obs[base + j])
        fv > m && (m = fv)
        j += nt
    end
    smem[tid] = m
    @synchronize()
    @uniform s = Int32(256)
    while s > Int32(1)
        @uniform s = s >> Int32(1)
        if tid <= s && tid + s <= nt
            smem[tid] = max(smem[tid], smem[tid + s])
        end
        @synchronize()
    end
    tid == 1 && (maxf[blob] = valid[blob] ? smem[1] : zero(Ta))
end

# =============================================================================
# Shared scalar core — the per-element algebra of the Student-t/uniform mixture
# =============================================================================
#
# The per-pixel and per-blob algebra the kernels below call, each piece once.
#
# EXPRESSION TREES ARE LOAD-BEARING. The device path is compared against the
# legacy CUDA trajectory — do not re-associate, do not introduce muladd/fma,
# do not "simplify". `Ta` is the accumulation type (Float64 on CPU/CUDA,
# Float32 on Metal); `lgamma_f` is the kernels' loggamma
# (Core's `lgamma_lanczos` on every backend).

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

# The paraboloid's log-intensity at flat pixel `gid`: z = A[gid, :]·θ[:, blob].
@inline _z_pred(::Type{Ta}, A, theta, gid, blob) where {Ta} =
    Ta(A[gid, 1]) * theta[1, blob] + Ta(A[gid, 2]) * theta[2, blob] +
    Ta(A[gid, 3]) * theta[3, blob] + Ta(A[gid, 4]) * theta[4, blob] +
    Ta(A[gid, 5]) * theta[5, blob] + Ta(A[gid, 6]) * theta[6, blob]

# The E-step of one pixel: its responsibility `pin[gid]` and IRLS weight
# `w[gid]` under Core's Student-t + uniform mixture `mix`, the weight times the
# spatial factor exp(θ₄a₄ + θ₅a₅ + θ₆a₆) when `spatial`.
@inline function _soft_weight!(w, pin, ::Type{Ta}, mix, r2, A, theta, gid, blob, spatial::Bool) where {Ta}
    r2i = Ta(r2[gid])
    p_in = inlier_posterior(mix, r2i)
    wi = posterior_weight(mix, r2i)
    pin[gid] = p_in
    if spatial
        q = theta[4, blob] * Ta(A[gid, 4]) +
            theta[5, blob] * Ta(A[gid, 5]) +
            theta[6, blob] * Ta(A[gid, 6])
        wi *= exp(q)
    end
    w[gid] = wi
    return nothing
end

# The E-step is Core's mixture at (ν, ŝ²) for a d_g = 1 residual, π = ½:
# `StudentTMixture{Ta}(ν, s2, log(2·outlier_halfwidth), 1; lgamma)`, whose
# `inlier_posterior` and `posterior_weight` are a pixel's responsibility and
# IRLS weight (the spatial factor is `_soft_weight!`'s — it reads the θ layout).
@inline _blob_mixture(::Type{Ta}, nu, s_hat_sq, outlier_halfwidth, blob, lgamma_f) where {Ta} =
    StudentTMixture{Ta}(nu[blob], max(s_hat_sq[blob], Ta(1.0e-30)), log(Ta(2.0) * outlier_halfwidth[blob]), 1; lgamma = lgamma_f)

# The sweep's score of a partition is VisualGeometryToolkitCore's one per-partition
# formula, `partition_score` (device-callable), at the Jeffreys prior (IG at
# α₀ = β₀ = 0): S = lgamma(j/2) − (j/2)·log 2π − (j/2)·log(RSS/2) − outlier_cost.
# The kernel supplies lgamma(j/2) — the backend's lgamma, its two closed forms at
# j = 1, 2 — and floors the RSS, as the score's callers do.
@inline function _half_dof_lgamma(jdof::Integer, ::Type{Ta}, lgamma_f::F) where {Ta, F}
    jdof == 1 && return Ta(0.5) * log(Ta(3.141592653589793))
    jdof == 2 && return zero(Ta)
    return lgamma_f(Ta(jdof) / Ta(2.0))
end

# The ν-Newton step's special functions at ν (the digammas Core's
# `nu_point_terms` reads, the trigammas its `next_nu_from_sums` reads).
@inline function _nu_terms(ν::Ta) where {Ta}
    ψ_half_νp1 = digamma_stirling((ν + Ta(1.0)) / Ta(2.0))
    ψ_half_ν = digamma_stirling(ν / Ta(2.0))
    ψ1_half_νp1 = trigamma_stirling((ν + Ta(1.0)) / Ta(2.0))
    ψ1_half_ν = trigamma_stirling(ν / Ta(2.0))
    return ψ_half_νp1, ψ_half_ν, ψ1_half_νp1, ψ1_half_ν
end

# The apply-time shape gate on (θ₄, θ₅, θ₆), the quadratic form
# C = [θ₄ θ₅; θ₅ θ₆]: the update is acceptable when C is a peak (`is_peak(C)`,
# a concave paraboloid) whose shape `-C` is within the aspect cap
# (`is_valid_shape`). Returns the blob state the refusal records (device
# side), or 0 when the update is acceptable.
@inline function _shape_gate_state(θ4::Ta, θ5::Ta, θ6::Ta, max_aspect_ratio) where {Ta}
    C = SMatrix{2, 2, Ta, 4}(θ4, θ5, θ5, θ6)
    is_peak(C) || return _ADAPT_NOT_PEAK
    is_valid_shape(-C; max_aspect_ratio) || return _ADAPT_ASPECT_RATIO
    return Int32(0)
end

# The gate's verdict by name (host side, and the tests): `:none` (acceptable),
# `:not_peak` or `:aspect_ratio`.
_shape_refusal(θ4, θ5, θ6, max_aspect_ratio) =
    _REJECTION[_shape_gate_state(promote(float(θ4), float(θ5), float(θ6))..., max_aspect_ratio)]

# The (row, column) of the normal matrix's upper triangle entry `kk` (1–21,
# row-major: (1,1), (1,2), …, (1,6), (2,2), …, (6,6)).
@inline function _upper_entry(kk::Integer)
    k = 0
    for jj in 1:6, jj2 in jj:6
        k += 1
        k == kk && return jj, jj2
    end
    return 6, 6
end

# Log-space whitened residuals r² = ((log(bg−v) − Aθ)·(bg−v))².
@kernel inbounds = true unsafe_indices = true function _residuals_ka!(
        r2, @Const(A), @Const(v), @Const(theta), @Const(bg),
        @Const(cnt), @Const(valid), @Const(offsets)
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(theta)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    n, base = _blob_span(cnt, valid, offsets, blob)
    i = Int32(tid)
    while i <= n
        gid = base + i
        r2[gid] = eltype(r2)(_adapt_r2(bg[blob] - Ta(v[gid]), _z_pred(Ta, A, theta, gid, blob)))
        i += nt
    end
end

# Per-blob bitonic sort of r² in shared memory → outlier_halfwidth = √median(r²) →
# marginal posterior sweep (RSDR's `partition_score` at the Jeffreys prior) → ν, ŝ².
@kernel inbounds = true unsafe_indices = true function _sort_sweep_ka!(
        nu, s_hat_sq, outlier_halfwidth, @Const(r2), @Const(np),
        @Const(cnt), @Const(valid), @Const(offsets), lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(nu)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    smem = @localmem eltype(nu) (4096,)

    n, base = _blob_span(cnt, valid, offsets, blob)
    i = Int32(tid)
    while i <= _SORT_PADDED
        smem[i] = (i <= n) ? Ta(r2[base + i]) : typemax(Ta)
        i += nt
    end
    @synchronize()

    @uniform k = Int32(1)
    while k < _SORT_PADDED
        @uniform k = k << Int32(1)
        @uniform j = k
        while j > Int32(1)
            @uniform j = j >> Int32(1)
            idx = Int32(tid) - Int32(1)
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
        end
    end

    if tid == 1 && valid[blob]
        n = cnt[blob]
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
                S = partition_score(
                    _JEFFREYS, _half_dof_lgamma(kk - m, Ta, lgamma_f), Ta(kk - m) / Ta(2.0),
                    max(RSS, Ta(1.0e-30)), Ta(n - kk) * log_two_outlier_halfwidth
                )
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

# Soft posterior weights + optional spatial factor (the initial weights; the
# per-iteration update is phase 4 of `_fused_ka!`, the same `_soft_weight!`).
@kernel inbounds = true unsafe_indices = true function _soft_weights_ka!(
        w, pin, @Const(r2), @Const(A), @Const(theta), @Const(nu),
        @Const(s_hat_sq), @Const(outlier_halfwidth),
        @Const(cnt), @Const(valid), @Const(offsets), spatial::Bool, lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(theta)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    n, base = _blob_span(cnt, valid, offsets, blob)
    if n > Int32(0)
        mix = _blob_mixture(Ta, nu, s_hat_sq, outlier_halfwidth, blob, lgamma_f)
        i = Int32(tid)
        while i <= n
            _soft_weight!(w, pin, Ta, mix, r2, A, theta, base + i, blob, spatial)
            i += nt
        end
    end
end

# 6×6 A'WA / A'Wz accumulate with W = w·f² — 27 work-item accumulators + the
# shared-memory tree reduction, np-masked write with identity padding on
# inactive params. Arithmetic mirrors legacy exactly.
@kernel inbounds = true unsafe_indices = true function _accumulate_ka!(
        AtWA, AtWz, @Const(A), @Const(v), @Const(w), @Const(bg),
        @Const(cnt), @Const(valid), @Const(offsets), @Const(np),
        @Const(converged)
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(AtWA)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    smem = @localmem eltype(AtWA) (256,)
    acc = @private eltype(AtWA) (27,)

    for kk in 1:27
        acc[kk] = zero(Ta)
    end
    n, base = _blob_span(cnt, valid, converged, offsets, blob)
    i = Int32(tid)
    while i <= n
        gid = base + i
        fi = bg[blob] - Ta(v[gid])
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

    # Each entry reduced over the group, then written by the first work item:
    # the upper triangle mirrored (rows and columns ≤ np only), then A'Wz.
    for kk in 1:27
        smem[tid] = acc[kk]
        @synchronize()
        @uniform s = Int32(256)
        while s > Int32(1)
            @uniform s = s >> Int32(1)
            if tid <= s && tid + s <= nt
                smem[tid] += smem[tid + s]
            end
            @synchronize()
        end
        if tid == 1 && valid[blob] && converged[blob] == Int32(0)
            m = Int(np[blob])
            if kk <= 21
                jj, jj2 = _upper_entry(kk)
                if jj <= m && jj2 <= m
                    AtWA[jj, jj2, blob] = smem[1]
                    AtWA[jj2, jj, blob] = smem[1]
                end
            elseif kk - 21 <= m
                AtWz[kk - 21, 1, blob] = smem[1]
            end
        end
        @synchronize()
    end

    if tid == 1 && valid[blob] && converged[blob] == Int32(0)
        for jj in (Int(np[blob]) + 1):6
            AtWA[jj, jj, blob] = one(Ta)
        end
    end
end

# A blob's IRLS state in `d_converged` (Int32): 0 live, `_ADAPT_CONVERGED`
# when the relative step fell below `rtol`, `_ADAPT_NOT_PEAK` /
# `_ADAPT_ASPECT_RATIO` when the shape gate refused the step (the blob keeps
# its previous iterate). Every kernel reads "live" as `== 0`, so each stop ends
# the blob's iterations. A blob still live after `max_irls_iter` was capped.
# The owner's §14 rulings for BlobBoards (2026-11-11): every returned frame
# records its outcome (`_blob_outcome`); a capped blob is UNCONVERGED, a
# refused one REJECTED, and neither is silently valid.
const _ADAPT_CONVERGED = Int32(1)
const _ADAPT_NOT_PEAK = Int32(2)
const _ADAPT_ASPECT_RATIO = Int32(3)
const _REJECTION = Dict(Int32(0) => :none, _ADAPT_NOT_PEAK => :not_peak, _ADAPT_ASPECT_RATIO => :aspect_ratio)

# The outcome record of one blob, in VisualGeometryToolkitCore's termination
# vocabulary (§14): converged → ReturnCode.Success / ParameterChange (rtol on
# the relative parameter step); capped → MaxIters / IterationCap; refused by
# the shape gate → Failure / ShapeGate, or InitialFailure when the very first
# step from the seed was refused; `rejection` names the gate that refused.
function _blob_outcome(state::Int32, initial::Bool)
    RC = VgtkCore.ReturnCode
    state == _ADAPT_CONVERGED && return (; converged = true, retcode = RC.Success, stop_rule = VgtkCore.ParameterChange, rejection = :none)
    state == Int32(0) && return (; converged = false, retcode = RC.MaxIters, stop_rule = VgtkCore.IterationCap, rejection = :none)
    return (; converged = false, retcode = initial ? RC.InitialFailure : RC.Failure, stop_rule = VgtkCore.ShapeGate, rejection = _REJECTION[state])
end

# Apply solution + convergence/validity check — one thread per blob.
# Mirrors legacy: LU failure invalidates; !is_peak or aspect violation
# reverts to θ_old and stops; otherwise apply + rtol convergence check.
@kernel inbounds = true unsafe_indices = true function _apply_check_ka!(
        theta, @Const(theta_old), @Const(AtWz), @Const(info), valid, converged,
        @Const(np), rtol, max_aspect_ratio
    )
    blob = @index(Global, Linear)
    if blob <= size(theta, 2) && valid[blob] && converged[blob] == Int32(0)
        Ta = eltype(theta)
        if info[blob] != Int32(0)
            valid[blob] = false
        else
            m = np[blob]
            refused = _shape_gate_state(AtWz[4, 1, blob], AtWz[5, 1, blob], AtWz[6, 1, blob], max_aspect_ratio)
            if refused != Int32(0)
                for j in 1:6
                    theta[j, blob] = theta_old[j, blob]
                end
                # Stopped, NOT converged: the shape gate refused the step (why).
                converged[blob] = refused
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
                    converged[blob] = _ADAPT_CONVERGED
                end
            end
        end
    end
end

# Fused per-iteration update: residuals → weighted scale ŝ² → Newton ν
# (at most 5 steps, E-step weights from pin) → soft weights. The legacy
# thread-1 `break` past the barrier is the `sflag` broadcast the ν loop's
# condition reads — identical stop conditions, identical ν, well-defined
# barriers on every backend. A blob that is not live sets the flag at once.
@kernel inbounds = true unsafe_indices = true function _fused_ka!(
        r2, w, pin, s_hat_sq, nu,
        @Const(A), @Const(v), @Const(theta), @Const(bg), @Const(outlier_halfwidth),
        @Const(cnt), @Const(valid), @Const(offsets), spatial::Bool,
        @Const(converged), lgamma_f
    )
    @uniform nt = Int32(@groupsize()[1])
    @uniform Ta = eltype(theta)
    blob = @index(Group, Linear)
    tid = @index(Local, Linear)
    smem_a = @localmem eltype(theta) (256,)
    smem_b = @localmem eltype(theta) (256,)
    smem_c = @localmem eltype(theta) (256,)
    sflag = @localmem Int32 (1,)

    # --- Phase 1: residuals ---
    n, base = _blob_span(cnt, valid, converged, offsets, blob)
    i = Int32(tid)
    while i <= n
        gid = base + i
        r2[gid] = eltype(r2)(_adapt_r2(bg[blob] - Ta(v[gid]), _z_pred(Ta, A, theta, gid, blob)))
        i += nt
    end
    @synchronize()

    # --- Phase 2: weighted scale ŝ² ---
    n, base = _blob_span(cnt, valid, converged, offsets, blob)
    local_wr2 = zero(Ta)
    local_w = zero(Ta)
    i = Int32(tid)
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
    @uniform s = Int32(256)
    while s > Int32(1)
        @uniform s = s >> Int32(1)
        if tid <= s && tid + s <= nt
            smem_a[tid] += smem_a[tid + s]
            smem_b[tid] += smem_b[tid + s]
        end
        @synchronize()
    end
    if tid == 1
        live = valid[blob] && converged[blob] == Int32(0)
        live && (s_hat_sq[blob] = max(smem_a[1] / max(smem_b[1], Ta(1.0e-30)), Ta(1.0e-30)))
        sflag[1] = live ? Int32(0) : Int32(1)
    end
    @synchronize()

    # --- Phase 3: Newton ν (≤ 5 steps, fixed E-step weights from pin) ---
    @uniform iter = Int32(0)
    while iter < Int32(5) && sflag[1] == Int32(0)
        @uniform iter = iter + Int32(1)
        n, base = _blob_span(cnt, valid, converged, offsets, blob)
        ν = nu[blob]
        s2 = max(s_hat_sq[blob], Ta(1.0e-30))
        ψ_half_νp1, ψ_half_ν, _, _ = _nu_terms(ν)
        local_dQ = zero(Ta)
        local_d2Q = zero(Ta)
        local_w = zero(Ta)
        i = Int32(tid)
        while i <= n
            gid = base + i
            wi = Ta(pin[gid])
            if wi >= Ta(1.0e-15)
                dq_i, d2q_data_i = nu_point_terms(Ta(r2[gid]) / (ν * s2), ν, ψ_half_νp1, ψ_half_ν)
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
        @uniform s = Int32(256)
        while s > Int32(1)
            @uniform s = s >> Int32(1)
            if tid <= s && tid + s <= nt
                smem_a[tid] += smem_a[tid + s]
                smem_b[tid] += smem_b[tid + s]
                smem_c[tid] += smem_c[tid + s]
            end
            @synchronize()
        end
        if tid == 1
            ν = nu[blob]
            _, _, ψ1_half_νp1, ψ1_half_ν = _nu_terms(ν)
            ν_new, stop_b = next_nu_from_sums(ν, smem_a[1], smem_b[1], smem_c[1], ψ1_half_νp1, ψ1_half_ν)
            ν_new != ν && (nu[blob] = ν_new)
            sflag[1] = stop_b ? Int32(1) : Int32(0)
        end
        @synchronize()
    end

    # --- Phase 4: soft weights + spatial (stores pin for next ν M-step) ---
    n, base = _blob_span(cnt, valid, converged, offsets, blob)
    if n > Int32(0)
        mix = _blob_mixture(Ta, nu, s_hat_sq, outlier_halfwidth, blob, lgamma_f)
        i = Int32(tid)
        while i <= n
            _soft_weight!(w, pin, Ta, mix, r2, A, theta, base + i, blob, spatial)
            i += nt
        end
    end
end

# =============================================================================
# Driver — every backend
# =============================================================================

function _adapt_ka(
        img, sites::AbstractVector{S}, cfg::AdaptConfig,
        device::KA.Backend
    ) where {S <: Union{Disc, EllipticFrame}}
    N = length(sites)
    N == 0 && return BlobDetection{Float64}[]

    # Accumulation/solve precision by backend capability: Float64 on
    # CUDA/CPU (identical to the legacy kernels), Float32 on Metal.
    Ta = KA.supports_float64(device) ? Float64 : Float32
    lgamma_f = lgamma_lanczos
    gs = _adapt_groupsize(device)

    cutoff = cfg.cutoff
    max_irls_iter = cfg.max_irls_iter
    spatial_weighting = cfg.spatial_weighting

    # The kernels promote per-pixel loads to Ta. The image is on `device`
    # through Core's one upload, on every backend: an array already there in
    # Float32 (the pipeline's) is used as it is; a Gray{Float64} image is read
    # at Ta (the legacy precision), anything else at Float32, an 8-bit or
    # 16-bit one converted on the device (`_device_image`).
    d_img = eltype(img) <: Gray{Float64} ? on_device(Ta, device, img) : _device_image(device, img)
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


    d_cx = ka_upload(Ta, device, cx_vec)
    d_cy = ka_upload(Ta, device, cy_vec)
    d_hw = ka_upload(Int32, device, hw_vec)
    d_spacing = ka_upload(Ta, device, spacing_vec)
    d_valid = ka_upload(Bool, device, collect(valid))
    d_offsets = ka_upload(Int32, device, offsets)

    # Per-pixel staging at Float32 — the only value AdaptConfig.store ever
    # took before its 2026-09 deletion; kernels promote to Ta on every read.
    d_A = KA.zeros(device, Float32, total_pixels, 6)
    d_v = KA.zeros(device, Float32, total_pixels)
    d_f_obs = KA.zeros(device, Float32, total_pixels)
    d_cnt = KA.zeros(device, Int32, N)
    d_bg = KA.zeros(device, Ta, N)
    KA.synchronize(device)

    _scan_ka!(device, gs)(
        d_A, d_f_obs, d_v, d_cnt, d_bg,
        d_img, d_cx, d_cy, d_hw, d_spacing, d_valid, d_offsets,
        Int32(H), Int32(W), Ta(sat[1]), Ta(sat[2]), Ta(cfg.min_intensity);
        ndrange = gs * N
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
    d_np = ka_upload(Int32, device, np_vec)
    d_valid_live = ka_upload(Bool, device, collect(valid))

    # --- θ₀ seed (host, cheap); per-blob max(f_obs) reduced on device ---
    d_maxf = KA.zeros(device, Ta, N)
    _segmented_maxf_ka!(device, gs)(
        d_maxf, d_f_obs, d_cnt, d_valid_live, d_offsets; ndrange = gs * N
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
    d_theta = ka_upload(Ta, device, theta_h)
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
    d_info = KA.zeros(device, Int32, N)
    d_theta_old = KA.zeros(device, Ta, 6, N)
    d_converged = KA.zeros(device, Int32, N)
    KA.synchronize(device)

    # --- Initial residuals + fused sort/median/sweep + initial weights ---
    _residuals_ka!(device, gs)(
        d_r2, d_A, d_v, d_theta, d_bg, d_cnt, d_valid_live, d_offsets;
        ndrange = gs * N
    )
    _sort_sweep_ka!(device, gs)(
        d_nu, d_s_hat_sq, d_ohw, d_r2, d_np, d_cnt, d_valid_live, d_offsets,
        lgamma_f; ndrange = gs * N
    )
    _soft_weights_ka!(device, gs)(
        d_w, d_pin, d_r2, d_A, d_theta, d_nu, d_s_hat_sq, d_ohw,
        d_cnt, d_valid_live, d_offsets, spatial_weighting, lgamma_f;
        ndrange = gs * N
    )
    KA.synchronize(device)

    # --- IRLS loop ---
    initial = falses(N)    # refused at the first step from the seed (InitialFailure)
    for iter in 1:max_irls_iter
        copyto!(d_theta_old, d_theta)

        d_AtWA .= zero(Ta)
        d_AtWz .= zero(Ta)
        _accumulate_ka!(device, gs)(
            d_AtWA, d_AtWz, d_A, d_v, d_w, d_bg,
            d_cnt, d_valid_live, d_offsets, d_np, d_converged;
            ndrange = gs * N
        )

        _lu6_kernel!(device, 256)(
            d_AtWA, d_AtWz, d_info, d_valid_live, d_converged;
            ndrange = 256 * cld(N, 256)
        )

        _apply_check_ka!(device, 256)(
            d_theta, d_theta_old, d_AtWz, d_info, d_valid_live, d_converged,
            d_np, Ta(cfg.rtol), Ta(cfg.max_aspect_ratio);
            ndrange = 256 * cld(N, 256)
        )

        _fused_ka!(device, gs)(
            d_r2, d_w, d_pin, d_s_hat_sq, d_nu,
            d_A, d_v, d_theta, d_bg, d_ohw,
            d_cnt, d_valid_live, d_offsets, spatial_weighting,
            d_converged, lgamma_f;
            ndrange = gs * N
        )

        # A refusal of the very first step is told apart (one N-Int32 download).
        iter == 1 && (initial .= Array(d_converged) .>= _ADAPT_NOT_PEAK)

        # All-converged early exit (two tiny device reductions).
        n_active = Int(sum(d_valid_live .& (d_converged .== Int32(0))))
        n_active <= 0 && break
    end
    KA.synchronize(device)

    # --- Download + convert ---
    valid_live = Array(d_valid_live)
    for i in 1:N
        valid[i] = valid[i] && valid_live[i]
    end
    theta_out = Array(d_theta)
    state = Array(d_converged)
    outcomes = [_blob_outcome(state[i], initial[i]) for i in 1:N]
    frames = _finalize_frames(sites, valid, outcomes, theta_out, σ_vec, cfg)
    n_capped = count(i -> valid[i] && state[i] == Int32(0), 1:N)
    n_not_peak = count(i -> valid[i] && state[i] == _ADAPT_NOT_PEAK, 1:N)
    n_aspect_ratio = count(i -> valid[i] && state[i] == _ADAPT_ASPECT_RATIO, 1:N)
    n_initial = count(i -> valid[i] && initial[i], 1:N)
    @debug "adapt: blob IRLS outcomes (before the frame gates)" n_blobs = N n_frames = length(frames) n_capped n_not_peak n_aspect_ratio n_initial max_irls_iter
    return frames
end

# Shared finalize: θ → Paraboloid3D → EllipticFrame with the legacy gates.
function _finalize_frames(sites, valid, outcomes, theta_out::AbstractMatrix, σ_vec, cfg::AdaptConfig)
    frames = BlobDetection{Float64}[]
    for i in eachindex(sites)
        valid[i] || continue
        θ = SVector{6, Float64}(Float64(theta_out[j, i]) for j in 1:6)
        p = Paraboloid3D(θ)
        is_peak(p) || continue
        Σ = shape_matrix(p)
        is_valid_shape(Symmetric(Σ)) || continue

        cx, cy = Float64.(centroid(sites[i]))
        σ = σ_vec[i]
        frame = EllipticFrame(Point2(cx, cy), p)
        sr = characteristic_scale(frame) / σ
        (sr > cfg.max_scale_ratio || sr < 1 / cfg.max_scale_ratio) && continue
        push!(frames, Detection(frame, (; id = i, outcomes[i]...)))
    end
    return frames
end
