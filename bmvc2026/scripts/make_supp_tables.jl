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
plotsdir(); the copy_tex stage (scripts/copy_tex.jl) copies plots/*.tex into papers/figures/floorplan/.
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
using BlobBoards: FindBoardsConfig, config_dict
using VisualGeometryToolkitCore: BoundedCamera, ustrip, mm, dpi
import VisualGeometryToolkitIO
import VisualGeometryToolkitFeatures

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: input_path, figure_config, write_figure_records, figure_numbers, sweep_dir, _quantity   # a board JSON's `{value, unit}` field as a quantity
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using VisualGeometryToolkitEval.Reproduce: add_number!, write_numbers

const CFG, CFG_PATH = figure_config("supp_tables"; default = projectdir("configs", "figures", "supp_tables.toml"))
const BOARDS_DIR = input_path(CFG, "boards")
const CAMERA_JSON = input_path(CFG, "camera")
const RUN_CONFIG = joinpath(sweep_dir(CFG["runs"]["bb"]), "poses", "bb", "_config.toml")


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


# The evaluation gallery's boards, as `_pattern` reads them.
_gallery() = [_pattern(f) for f in readdir(BOARDS_DIR; join = true) if endswith(f, ".json")]

function write_board_params(nums)
    size_words = CFG["board_params"]["size_classes"]
    boards = _gallery()
    extents = sort(unique(b.extent_mm for b in boards))
    length(extents) == length(size_words) || error("expected $(length(size_words)) size classes, found extents $extents mm")
    add_number!(nums, "bbGalleryBoards", length(boards))
    # the per-board range over the whole gallery (all sizes)
    add_number!(nums, "bbBlobsMinAll", minimum(b.n_blobs for b in boards))
    add_number!(nums, "bbBlobsMaxAll", maximum(b.n_blobs for b in boards))
    for (word, ext) in zip(size_words, extents)
        n = [b.n_blobs for b in boards if b.extent_mm == ext]
        add_number!(nums, "bbBoards$word", length(n))
        add_number!(nums, "bbExtentMM$word", ext)
        add_number!(nums, "bbBlobsMed$word", median(n); sigdigits = 15)
        add_number!(nums, "bbBlobsMin$word", minimum(n))
        add_number!(nums, "bbBlobsMax$word", maximum(n))
    end
    # The shared values must be shared: a gallery mixing parameter sets would
    # make one table column a lie.
    shared(f) = (v = unique(f.(boards)); length(v) == 1 ? only(v) : error("boards disagree on $f: $v"))
    add_number!(nums, "bbPatMinScaleMM", shared(b -> b.min_scale_mm); digits = 3)
    add_number!(nums, "bbPatMaxDiamFrac", shared(b -> b.max_diameter_fraction); sigdigits = 15)
    add_number!(nums, "bbPatCutoff", shared(b -> b.cutoff); sigdigits = 15)
    add_number!(nums, "bbPatAlpha", shared(b -> b.alpha); sigdigits = 15)
    add_number!(nums, "bbPatScalesPerOctave", shared(b -> b.scales_per_octave))
    add_number!(nums, "bbPatDPI", shared(b -> b.dpi); sigdigits = 15)
    add_number!(nums, "bbPatPolarity", shared(b -> b.polarity))
    return nothing
end

# ---- tab:supp:settings -------------------------------------------------------

# The run's effective configuration (`_config.toml`, written by the sweep in the
# loader's own vocabulary).
_run_find_boards() = TOML.parsefile(RUN_CONFIG)["find_boards"]

function _camera()
    # the session's camera of record (calibration/<camera>.json, vgio/cameras/v3)
    return VgtkIO.load(BoundedCamera, CAMERA_JSON)
end

# 1e-6 → "10^{-6}", as the table prints a tolerance.
function _pow10(x)
    e = log10(x)
    isinteger(e) || error("$x is not a power of ten")
    return "10^{$(Int(e))}"
end

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
    add_number!(nums, "bbSetWarps", length(tilts))
    for (word, (t, n)) in zip(("One", "Two"), groups)
        add_number!(nums, "bbSetTilt$word", t; digits = 2)
        add_number!(nums, "bbSetTiltAngles$word", n)
    end
    add_number!(nums, "bbSetPeakThreshold", det["peak_threshold"]; sigdigits = 15)
    add_number!(nums, "bbSetEdgeThreshold", det["edge_threshold"]; digits = 1)
    add_number!(nums, "bbSetNmsCutoff", det["nms"]["cutoff"]; digits = 1)
    add_number!(nums, "bbSetScanWindow", ad["scan_padding"] * board_cutoff; digits = 1)
    add_number!(nums, "bbSetMaxIrls", ad["max_irls_iter"])
    add_number!(nums, "bbSetRtol", _pow10(ad["rtol"]))
    add_number!(nums, "bbSetMaxScaleFraction", ad["max_scale_fraction"]; sigdigits = 15)
    add_number!(nums, "bbSetMaxScaleRatio", ad["max_scale_ratio"]; sigdigits = 15)
    add_number!(nums, "bbSetMaxClipRatio", ad["max_clip_ratio"]; sigdigits = 15)
    add_number!(nums, "bbSetMaxAspectRatio", ad["max_aspect_ratio"]; sigdigits = 15)
    add_number!(nums, "bbSetInnerRadius", board_cutoff; digits = 1)
    add_number!(nums, "bbSetMinSimilarity", d["min_similarity"]; sigdigits = 15)
    add_number!(nums, "bbSetMinInliers", geo["min_inliers"])
    add_number!(nums, "bbSetHalfwidthPx", geo["outlier_halfwidth"]; sigdigits = 15)
    add_number!(nums, "bbSetRansacMin", ran["min_trials"]; group_from = 5)
    add_number!(nums, "bbSetRansacMax", ran["max_trials"]; group_from = 5)
    add_number!(nums, "bbSetConfidence", ran["success_prob"]; sigdigits = 15)
    add_number!(nums, "bbSetAppearanceSupportPct", round(Int, 100 * d["min_appearance_support"]))
    add_number!(nums, "bbSetVerifySimilarity", d["verify_similarity"]; sigdigits = 15)
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
        inputs = [BOARDS_DIR, RUN_CONFIG, CAMERA_JSON])
    return nothing
end

run_stages("make_supp_tables", [Stage(:render, "board_params.tex + run_settings.tex", () -> false, main)])
