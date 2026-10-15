#=
compare_flip_ecdfs.jl — BlobBoards-only ECDF comparison, without vs with
the mirror + symmetric-LRT flip disambiguation. NOT a paper figure: a
two-page review PDF (page 1: baseline poses/bb.noflip; page 2: shipped
poses/bb), same style as the paper's Fig. 6 panels (attempt-normalized,
log x, misses count as infinite error so curves saturate at the
detection rate; dashed guide at 0.5).

Writes plotsdir()/flip_ecdf_compare.{tex,pdf} (pdflatex).

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/compare_flip_ecdfs.jl

The runs are the evaluation's (configs/figures/evaluation.toml, src/runs.jl).
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
using DrWatson: plotsdir
using .BlobBoardsEval
using .BlobBoardsEval: rig_sequence, load_detected_poses, hand_eye,
    absolute_errors, eval_sources, load_config, poses_dir

SRC = eval_sources(load_config("figures/evaluation.toml"))
_stem(f) = splitext(basename(f))[1]

frames = rig_sequence(SRC.mocap, "bb")
cf = rig_sequence(SRC.mocap, "calibration_bb")   # with the static-camera recovery's frames
cd_ = load_detected_poses(poses_dir(SRC, "bb"; calibration = true))
# The ONE hand-eye authority (src/calibration.jl): cached, keyed on the
# frames and markers that determine the fit, and tagged with the commit.
# This was an independent refit — the project had FOUR of them.
he = hand_eye(cf, cd_; dataset = SRC.dataset, subset = "calibration_bb")
n_tracked = count(f -> f.T_cb !== nothing, frames)
N_ATTEMPTS = 30 * n_tracked

function errs(series)
    det = Dict(_stem(i) => p for (i, p) in load_detected_poses(poses_dir(SRC, series)))
    ae = absolute_errors(frames, det, he.X, he.Y)
    return (
        trans = sort([v.error_distance for v in values(ae)]),
        rot = sort([rad2deg(v.error_angle) for v in values(ae)]),
    )
end

variants = [
    ("without disambiguation (bb_noflip)", errs("bb_noflip")),
    ("with mirror + symmetric LRT (bb)", errs("bb")),
]

# ECDF panel: attempt-normalized, log x, style of the paper's Fig. 6
function panel(io, e, lo, hi, xticks, color)
    n = length(e)
    idx = unique(vcat(1, collect(1:max(1, n ÷ 250):n), n))
    coords = join(
        [
            "($(round(clamp(e[i], lo * 1.0001, hi * 0.9999); sigdigits = 5))," *
                "$(round(i / N_ATTEMPTS; sigdigits = 4)))" for i in idx
        ], " "
    )
    println(io, "\\begin{tikzpicture}")
    println(io, "\\begin{axis}[width=6.4cm, height=4.6cm, xmode=log, ymin=0, ymax=1,")
    println(io, "  xmin=$(lo), xmax=$(hi), ylabel={\$P(\\mathrm{error} < x)\$},")
    println(io, "  $(xticks),")
    println(io, "  ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,0.5,,1},")
    println(io, "  label style={font=\\scriptsize}, tick label style={font=\\scriptsize}]")
    println(io, "\\addplot[forget plot, black!35, dashed, line width=0.5pt] coordinates {($(lo),0.5) ($(hi),0.5)};")
    println(io, "\\addplot[const plot, draw=$(color), line width=1.0pt] coordinates {($(lo),0) $(coords) ($(hi),$(round(n / N_ATTEMPTS; sigdigits = 4)))};")
    println(io, "\\end{axis}")
    return println(io, "\\end{tikzpicture}")
end

io = IOBuffer()
println(io, "\\documentclass[11pt]{article}")
println(io, "\\usepackage[margin=2cm]{geometry}")
println(io, "\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}")
println(io, "\\usepackage{xcolor}\\definecolor{bbBlue}{HTML}{0072B2}")
println(io, "\\pagestyle{empty}")
println(io, "\\begin{document}")
for (k, (title, e)) in enumerate(variants)
    println(io, "\\begin{center}")
    println(io, "\\textbf{BlobBoards — $(title)}\\\\[1em]")
    println(io, "\\begin{tabular}{cc}")
    panel(io, e.trans, 0.5, 400.0, "xtick={1,10,100}, xticklabels={{1 mm},{10 mm},{100 mm}}", "bbBlue")
    println(io, "&")
    panel(
        io, e.rot, 0.02, 200.0,
        raw"xtick={0.1,1,10,100}, xticklabels={{0.1$^\circ$},{1$^\circ$},{10$^\circ$},{100$^\circ$}}", "bbBlue"
    )
    println(io, "\\\\")
    println(io, "(a) Translation error & (b) Rotation error \\\\")
    println(io, "\\end{tabular}")
    println(io, "\\end{center}")
    k < length(variants) && println(io, "\\clearpage")
end
println(io, "\\end{document}")

# plotsdir(): a figure is a derived product of THIS project, not an
# addition to the immutable dataset tree (and this write had no
# assert_primary_copy guard).
figdir = plotsdir(); mkpath(figdir)
texpath = joinpath(figdir, "flip_ecdf_compare.tex")
open(texpath, "w") do f
    write(f, take!(io))
end
run(Cmd(`pdflatex -interaction=nonstopmode flip_ecdf_compare.tex`; dir = figdir))
foreach(ext -> rm(joinpath(figdir, "flip_ecdf_compare." * ext); force = true), ("aux", "log"))
println("flip_ecdf_compare.pdf -> $figdir")
