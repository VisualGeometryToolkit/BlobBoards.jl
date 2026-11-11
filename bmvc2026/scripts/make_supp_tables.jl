#=
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
make_supp_tables.jl — the macros behind the supplement's two settings tables
(and the main paper's prose that quotes them), generated from data instead of
transcribed:

  board_params.tex   tab:supp:params — the evaluation gallery's pattern
                     parameters, read from the board JSONs the sweep detects
                     (`<dataset>/boards/blobboards/*.json`): boards per size
                     class, pattern extent, blobs per board (median and
                     range), and the shared PatternConfig values.
  run_settings.tex   tab:supp:settings — the operational parameters of the
                     BlobBoards run, read from the effective configuration it
                     wrote (`<run>/poses/bb/_config.toml`, the run found
                     through its config of record, src/runs.jl) through
                     `FindBoardsConfig`, plus the IMAS-25 tilt schedule the
                     empty `tilts` list stands for.

Values are formatted the way the tables print them (`5` px, `$10^{-6}$`), so
a macro and the literal it replaces render identically. Both files go to
plotsdir(); regenerate_paper.jl copies plots/*.tex into papers/figures/floorplan/.
Every input and output name is in the config of record,
configs/figures/supp_tables.toml (`--config=<path>` overrides); each run writes
its record (effective config, provenance, outputs) under plotsdir("records").

# reproduce: writes papers/figures/floorplan/board_params.tex
# reproduce: writes papers/figures/floorplan/run_settings.tex
# reproduce: reads configs/figures/supp_tables.toml

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/make_supp_tables.jl [--config=<path>]
Data: the inputs tier (boards/, the camera of record) and the BlobBoards run (its _config.toml).
=#

using DrWatson
@quickactivate
using DrWatson: projectdir

import JSON
import TOML
using Statistics: median
using Printf: @sprintf
using BlobBoards: FindBoardsConfig, config_dict
using VisualGeometryToolkitCore: CameraModel, parse_unit, ustrip, mm, dpi
import VisualGeometryToolkitIO
import VisualGeometryToolkitFeatures

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, figure_config, write_figure_records, figure_numbers, sweep_dir, camera_of_record
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using VisualGeometryToolkitEval.Reproduce: add_number!, write_numbers

const CFG, CFG_PATH = figure_config("supp_tables"; default = projectdir("configs", "figures", "supp_tables.toml"))
const DATASET = joinpath(data_root(; tiers = String.(CFG["experiment"]["tiers"])), "absolute_pose_experiments", CFG["experiment"]["dataset"])
const INPUTS = CFG["inputs"]
const RUN_CONFIG = joinpath(sweep_dir(INPUTS["run"]), "poses", "bb", "_config.toml")
# A board JSON's `{value, unit}` field as a quantity, its unit read through
# VisualGeometryToolkitCore's one guarded parse; lengths come out with `ustrip(mm, …)`.
_quantity(x) = Float64(x.value) * parse_unit(String(x.unit))

# A number macro of this driver (VisualGeometryToolkitEval's one writer), its body as typeset here.
_macro(nums, name, value) = add_number!(nums, name, string(value))

# ---- tab:supp:params ---------------------------------------------------------

# A board JSON's pattern extent in mm (pattern resolution at the print density).
function _pattern(path)
    j = JSON.parse(read(path, String))
    pc, bc = j.preamble.pattern_config, j.preamble.board_config
    density = _quantity(bc.print_density)
    extent_mm = round(Int, ustrip(mm, _quantity(pc.pattern_resolution.width) / density))
    return (;
        extent_mm, n_blobs = length(j.blobs.centers), dpi = ustrip(dpi, density),
        min_scale_mm = ustrip(mm, _quantity(pc.min_scale) / density),
        max_diameter_fraction = Float64(pc.max_diameter_fraction), cutoff = Float64(pc.cutoff),
        alpha = Float64(pc.alpha), scales_per_octave = Int(pc.scales_per_octave),
        polarity = String(pc.polarity),
    )
end

_num(x) = isinteger(x) ? string(Int(x)) : string(x)

# The evaluation gallery's boards, as `_pattern` reads them.
_gallery() = [_pattern(f) for f in readdir(joinpath(DATASET, INPUTS["boards_dir"]); join = true) if endswith(f, ".json")]

function write_board_params(nums)
    size_words = CFG["board_params"]["size_classes"]
    boards = _gallery()
    extents = sort(unique(b.extent_mm for b in boards))
    length(extents) == length(size_words) || error("expected $(length(size_words)) size classes, found extents $extents mm")
    _macro(nums, "bbGalleryBoards", length(boards))
    # the per-board range over the whole gallery (all sizes)
    _macro(nums, "bbBlobsMinAll", minimum(b.n_blobs for b in boards))
    _macro(nums, "bbBlobsMaxAll", maximum(b.n_blobs for b in boards))
    for (word, ext) in zip(size_words, extents)
        n = [b.n_blobs for b in boards if b.extent_mm == ext]
        _macro(nums, "bbBoards$word", length(n))
        _macro(nums, "bbExtentMM$word", ext)
        _macro(nums, "bbBlobsMed$word", _num(median(n)))
        _macro(nums, "bbBlobsMin$word", minimum(n))
        _macro(nums, "bbBlobsMax$word", maximum(n))
    end
    # The shared values must be shared: a gallery mixing parameter sets would
    # make one table column a lie.
    shared(f) = (v = unique(f.(boards)); length(v) == 1 ? only(v) : error("boards disagree on $f: $v"))
    _macro(nums, "bbPatMinScaleMM", _num(round(shared(b -> b.min_scale_mm); digits = 3)))
    _macro(nums, "bbPatMaxDiamFrac", _num(shared(b -> b.max_diameter_fraction)))
    _macro(nums, "bbPatCutoff", _num(shared(b -> b.cutoff)))
    _macro(nums, "bbPatAlpha", _num(shared(b -> b.alpha)))
    _macro(nums, "bbPatScalesPerOctave", shared(b -> b.scales_per_octave))
    _macro(nums, "bbPatDPI", _num(shared(b -> b.dpi)))
    _macro(nums, "bbPatPolarity", shared(b -> b.polarity))
    return nothing
end

# ---- tab:supp:settings -------------------------------------------------------

# The run's effective configuration (`_config.toml`, written by the sweep in the
# loader's own vocabulary).
_run_find_boards() = TOML.parsefile(RUN_CONFIG)["find_boards"]

function _camera()
    # the session's camera of record (calibration/<camera>.json, vgio/cameras/v3)
    return VgtkIO.load(CameraModel, camera_of_record(DATASET))
end

# 1e-6 → "10^{-6}", as the table prints a tolerance.
function _pow10(x)
    e = log10(x)
    isinteger(e) || error("$x is not a power of ten")
    return "10^{$(Int(e))}"
end

_thousands(n::Integer) = n < 10_000 ? string(n) : replace(string(n), r"(\d)(\d{3})$" => s"\1\\,\2")

function write_run_settings(nums)
    fb = _run_find_boards()
    cfg = FindBoardsConfig(fb; camera = _camera())     # the loader's key checking, on the run of record
    d = config_dict(cfg)
    det, ad, geo, ran = d["detect"], d["adapt"], d["geometry"], d["ransac"]
    isempty(det["tilts"]) || error("the run states an explicit tilt schedule; the table describes IMAS-25")
    tilts = VgtkFeatures.tilt_schedule(:imas25)    # what the empty `tilts` list stands for
    groups = [(t, count(x -> first(x) == t, tilts)) for t in sort(unique(first.(tilts))) if t != 1.0]
    length(groups) == 2 || error("unexpected IMAS schedule $groups")
    # the boards' cutoff (one across the gallery): the adaptation's scan window
    # and inner radius are stated in it
    board_cutoff = only(unique(b.cutoff for b in _gallery()))
    _macro(nums, "bbSetWarps", length(tilts))
    for (word, (t, n)) in zip(("One", "Two"), groups)
        _macro(nums, "bbSetTilt$word", @sprintf("%.2f", t))
        _macro(nums, "bbSetTiltAngles$word", n)
    end
    _macro(nums, "bbSetPeakThreshold", _num(det["peak_threshold"]))
    _macro(nums, "bbSetEdgeThreshold", @sprintf("%.1f", det["edge_threshold"]))
    _macro(nums, "bbSetNmsCutoff", @sprintf("%.1f", det["nms"]["cutoff"]))
    _macro(nums, "bbSetScanWindow", @sprintf("%.1f", ad["scan_padding"] * board_cutoff))
    _macro(nums, "bbSetMaxIrls", ad["max_irls_iter"])
    _macro(nums, "bbSetRtol", _pow10(ad["rtol"]))
    _macro(nums, "bbSetMaxScaleFraction", _num(ad["max_scale_fraction"]))
    _macro(nums, "bbSetMaxScaleRatio", _num(ad["max_scale_ratio"]))
    _macro(nums, "bbSetMaxClipRatio", _num(ad["max_clip_ratio"]))
    _macro(nums, "bbSetMaxAspectRatio", _num(ad["max_aspect_ratio"]))
    _macro(nums, "bbSetInnerRadius", @sprintf("%.1f", board_cutoff))
    _macro(nums, "bbSetMinSimilarity", _num(d["min_similarity"]))
    _macro(nums, "bbSetMinInliers", geo["min_inliers"])
    _macro(nums, "bbSetHalfwidthPx", _num(geo["outlier_halfwidth"]))
    _macro(nums, "bbSetRansacMin", _thousands(ran["min_trials"]))
    _macro(nums, "bbSetRansacMax", _thousands(ran["max_trials"]))
    _macro(nums, "bbSetConfidence", _num(ran["success_prob"]))
    _macro(nums, "bbSetAppearanceSupportPct", round(Int, 100 * d["min_appearance_support"]))
    _macro(nums, "bbSetVerifySimilarity", _num(d["verify_similarity"]))
    return nothing
end

function main()
    mkpath(plotsdir())
    outputs = String[]
    for (section, f) in (("board_params", write_board_params), ("run_settings", write_run_settings))
        out = plotsdir(CFG[section]["output"])
        nums = figure_numbers(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_supp_tables.jl")
        f(nums)
        write_numbers(out, nums)
        push!(outputs, out)
        println("$(basename(out)) -> $out")
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_supp_tables.jl", outputs,
        inputs = [joinpath(DATASET, INPUTS["boards_dir"]), RUN_CONFIG, camera_of_record(DATASET)])
    return nothing
end

run_stages("make_supp_tables", [Stage(:render, "board_params.tex + run_settings.tex", () -> false, main)])
