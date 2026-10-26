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
    # suites/jet.jl ignores; the rest are the unions that suite filters. A ratchet: it may only move down.
    jet_bound = 12,
    stale_ignore = (:pdftocairo,),   # called inside Cmd interpolation, which the analysis does not see (explicit_imports.jl)
    tag_rules = :report, tag_tables = :report, config_tags = :report,
    stack_dependencies = :report, public_names = :report, contract_generics = :report,
)
