# =============================================================================
# Package extensions: what each one needs, and the error a missing one raises
# =============================================================================
#
# GPU finding, the Lux descriptor backend, training, plotting and the Python
# interface live in package extensions (ext/), which load only once their
# trigger packages are installed and loaded. A function whose methods live in
# an extension has a fallback here that names the exact `pkg> add` line,
# instead of the MethodError a bare stub would raise. The README's "Install for
# your task" table states the Makie, training and CUDA lines, and
# test/suites/readme.jl checks that they agree and that every extension of the
# Project.toml has its entry. The Lux backend and PythonCall (the Julia side of
# the Python package, which a Python user gets from the clone's pixi
# environment instead) have no row. Included after every stub the table names.

# An extension's entry: the packages a reader installs and loads for it.
_extension_entry(what, packages...) = (
    install = "pkg> add " * join(packages, " "),
    load = "using " * join(packages, ", "),
    what = what,
)

"""
    EXTENSION_INSTALL

The install line and the load line of each package extension of BlobBoards,
keyed by the extension's name. A function whose methods live in an extension
names these lines when it is called without the extension loaded.
"""
const EXTENSION_INSTALL = (
    BlobBoardsMakieExt = _extension_entry(
        "a Makie backend (CairoMakie for files, GLMakie for windows)", "BlobBoards", "CairoMakie"),
    BlobBoardsPythonExt = _extension_entry(
        "PythonCall (the Julia side of the Python package)", "BlobBoards", "PythonCall"),
    BlobBoardsLuxExt = _extension_entry(
        "Lux, NNlib and cuDNN (the Lux descriptor backend)", "BlobBoards", "Lux", "NNlib", "cuDNN"),
    BlobBoardsTrainingExt = _extension_entry(
        "the training stack (Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote)",
        "BlobBoards", "Lux", "Optimisers", "Enzyme", "Reactant", "NNlib", "TensorBoardLogger", "Zygote"),
    BlobBoardsCUDAExt = _extension_entry(
        "CUDA and cuDNN (GPU detection and the ONNX CUDA provider)", "BlobBoards", "CUDA", "cuDNN"),
)

# ONNX Runtime's CUDA provider takes a CUDA 12 runtime (>= 12.8), older than
# CUDA.jl's default: an environment that finds boards on the GPU pins it once.
# The root Project.toml and the GPU workspace members carry the same pin.
const CUDA_RUNTIME_PIN = "using CUDA; CUDA.set_runtime_version!(v\"12.9\")"

_extension_loaded(extension::Symbol) = Base.get_extension(@__MODULE__, extension) !== nothing

# The message of a call to an extension's function without the extension.
function _missing_extension_message(f, extension::Symbol)
    e = EXTENSION_INSTALL[extension]
    return "BlobBoards.$(nameof(f)) needs $(e.what), and this session has not loaded it. " *
        "Install once with\n\n    $(e.install)\n\nand load with `$(e.load)`."
end

# The fallback method of a function whose methods live in `extension`: without
# the extension, an error naming the install line; with it loaded, the
# MethodError the call would have raised (the arguments fit no method of the
# extension).
function _require_extension(f, extension::Symbol, args)
    # The caller's world age: a call compiled before the extension loaded keeps
    # Julia's "method too new" hint.
    _extension_loaded(extension) && throw(MethodError(f, args, Base.tls_world_age()))
    return error(_missing_extension_message(f, extension))
end

"""
    EXTENSION_FUNCTIONS

Each function whose methods live in a package extension, with that extension's
name. Every one has the fallback `f(args...; kwargs...)` that raises the error
naming `EXTENSION_INSTALL`'s install line. `board` and `board!` are not
here: they forward to the extension's Makie recipe (src/authoring_stubs.jl).
"""
const EXTENSION_FUNCTIONS = (
    (plot_boards, :BlobBoardsMakieExt), (plot_board_poses, :BlobBoardsMakieExt),
    (board_overlay!, :BlobBoardsMakieExt), (inliers!, :BlobBoardsMakieExt),
    (board_panel!, :BlobBoardsMakieExt), (plot_comparison, :BlobBoardsMakieExt),
    (plot_tracks, :BlobBoardsMakieExt), (replay, :BlobBoardsMakieExt),
    (inspect_pipeline, :BlobBoardsMakieExt), (plot_annotations, :BlobBoardsMakieExt),
    (view_angle_figure, :BlobBoardsMakieExt), (save_view_angle_figure, :BlobBoardsMakieExt),
    (blob_pattern_python, :BlobBoardsPythonExt), (blob_board_python, :BlobBoardsPythonExt),
    (blobs_python, :BlobBoardsPythonExt), (detect_blobs_python, :BlobBoardsPythonExt),
    (find_boards_python, :BlobBoardsPythonExt), (load_board_python, :BlobBoardsPythonExt),
    (render_board_python, :BlobBoardsPythonExt), (track_boards_python, :BlobBoardsPythonExt),
    (LuxDescriptorNetwork, :BlobBoardsLuxExt), (train_descriptor, :BlobBoardsTrainingExt),
)

for (f, extension) in EXTENSION_FUNCTIONS
    @eval $(nameof(f))(args...; kwargs...) = _require_extension($f, $(QuoteNode(extension)), args)
end
