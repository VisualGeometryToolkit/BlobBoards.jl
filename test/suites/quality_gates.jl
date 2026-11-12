# =============================================================================
# The stack's quality gates (VisualGeometryToolkitCore's `test_quality_gates`):
# tags, the dependency allow-list, public names across package boundaries and
# the contract generics. Every gate is in :report mode, the owner's rule for a
# stack package until it conforms: a finding is logged with @warn and recorded
# broken, never silenced, and a gate that finds nothing is an ordinary test.
# =============================================================================
using Test, Aqua, ExplicitImports, JET
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation
import VisualGeometryToolkitIO

VgtkCore.test_quality_gates(
    BlobBoards;
    # the count the gate's whole-package JET run finds today (2026-10-27): 5 are the vendored QREncode, which
    # suites/jet.jl ignores in its authoring run; the rest are unions in the package core. A ratchet: it may only move down.
    # (Julia 1.12 infers 15: the same unions plus three more in code this suite does not touch; the bound is per minor version)
    jet_bound = VERSION >= v"1.13" ? 12 : 15,
    stale_ignore = (:pdftocairo,),   # called inside Cmd interpolation, which the analysis does not see
    # the CUDA/cuDNN/ONNXRunTime runtime stack holds background tasks by design; the precompile-hang
    # risk this check probes is exercised by CI's precompilation of the package itself
    persistent_tasks = false,
    tag_rules = :report, tag_tables = :report, config_tags = :report,
    stack_dependencies = :report, public_names = :report, contract_generics = :report,
    # third_party_public stays :report. Its accepted findings, each with its reason:
    # - explicit imports: KernelAbstractions' `@context` (src/adapt_ka.jl) is not declared public there, but it is
    #   KernelAbstractions' API for the kernel context that Core's `tree_sum!(@context, smem, tid, nt)` takes, the
    #   stack's one workgroup sum (Core's numerics/ka_reduce.jl imports it the same way). Accepted 2026-11-12.
    # - qualified accesses: KernelAbstractions' `@atomic` and `zeros`, Base internals (as before).
)

# The extension points BlobBoards declares: the generics CaliBlobs extends (its `GeometryConfig` arm and its run
# configuration), and nothing else; the methods it adds to other stack packages' functions are all on their
# contracts (VisualGeometryToolkitIO's `migrate_schema`, src/migrate_boards.jl; VisualGeometryToolkitEstimation's
# `config_dict`, a run configuration's TOML table, src/config_toml.jl; VisualGeometryToolkitIO's `load` and `save`,
# a bundle's reader and a pattern's writer, src/io.jl — not yet among IO's CONTRACT_GENERICS: a hoist).
@testset "BlobBoards declares its extension points" begin
    @test BlobBoards.CONTRACT_GENERICS ==
        (BlobBoards.geometry_tag, BlobBoards.geometry_from_toml, BlobBoards.write_config)
    @test BlobBoards.config_dict === VisualGeometryToolkitEstimation.config_dict
    # `load`/`save` are VisualGeometryToolkitIO's; BlobBoards adds the methods of its own types
    @test BlobBoards.load === VisualGeometryToolkitIO.load && BlobBoards.save === VisualGeometryToolkitIO.save
    @test hasmethod(BlobBoards.load, Tuple{Type{BlobBoards.BoardBundle}, String})
    @test hasmethod(BlobBoards.save, Tuple{Matrix{Float64}, BlobBoards.PatternBundle})
    # IO lists `load`/`save` among its CONTRACT_GENERICS: BlobBoards' methods land on declared extension points
    cv = VgtkCore.contract_generic_violations(BlobBoards)
    @test isempty(cv)
end
