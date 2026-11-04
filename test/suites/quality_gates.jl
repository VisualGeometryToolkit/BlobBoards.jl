# =============================================================================
# The stack's quality gates (VisualGeometryToolkitCore's `test_quality_gates`):
# tags, the dependency allow-list, public names across package boundaries and
# the contract generics. Every gate is in :report mode, the owner's rule for a
# stack package until it conforms: a finding is logged with @warn and recorded
# broken, never silenced, and a gate that finds nothing is an ordinary test.
# =============================================================================
using Test, Aqua, ExplicitImports, JET
import VisualGeometryToolkitCore as VgtkCore

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
)

# The extension points BlobBoards declares: the generics CaliBlobs extends (its `GeometryConfig` arm and its run
# configuration), and nothing else; the methods it adds to other stack packages' functions are all on their
# contracts (VisualGeometryToolkitIO's `migrate_schema`, src/migrate_boards.jl).
@testset "BlobBoards declares its extension points" begin
    @test BlobBoards.CONTRACT_GENERICS ==
        (BlobBoards.geometry_tag, BlobBoards.geometry_from_toml, BlobBoards.config_dict, BlobBoards.write_config)
    @test isempty(VgtkCore.contract_generic_violations(BlobBoards))
end
