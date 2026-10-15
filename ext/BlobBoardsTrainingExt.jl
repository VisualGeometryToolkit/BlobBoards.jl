module BlobBoardsTrainingExt

# Trigger packages — extension only loads when ALL of these are present.
# Reactant + Enzyme are the maintainer-recommended path for big CNN
# training (see Lux's ResNet20 tutorial). The training loop pre-warms
# Reactant via @compile on the forward evaluation before entering
# single_train_step!, so AutoEnzyme reuses the cached HLO graph.
using Lux
using Optimisers
using Enzyme
using Reactant
using NNlib
using TensorBoardLogger
using Logging: with_logger
using Zygote
# Note: cuDNN is NOT a trigger here. It is loaded lazily by Lux when GPU
# operations actually run (and is an upstream weakdep of BlobBoards to make
# it Pkg.add-able by users opting into GPU training).

# Stdlib + lightweight deps already available in BlobBoards
using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra
using SHA: sha256
using JLD2: JLD2

using BlobBoards
using BlobBoards: AbstractDescriptorNetwork, load_annotations

# The model architecture (`BlobDescriptorNetwork`, ext/training/model.jl) is
# included HERE as well as in BlobBoardsLuxExt — one source file, compiled in
# both extension modules. It used to be reached through
# `Base.get_extension(BlobBoards, :BlobBoardsLuxExt)` behind an assertion that
# LuxExt "must load first", on the claim that this extension's trigger set is
# a superset of LuxExt's. It is not: LuxExt needs cuDNN and this one
# deliberately does not, so `Pkg.precompile` — which compiles each extension
# with exactly its own triggers — could NEVER compile this module (found
# 2026-09-07 when the SLURM training env was brought up to date; every env
# with the training stack had been failing this way silently). Training needs
# only the architecture, not the inference wrapper: a checkpoint trained here
# loads in LuxExt's `LuxDescriptorNetwork` because both compile the same
# Chain, and Lux parameters/state are keyed by layer name, not by module.
include("training/model.jl")

# =============================================================================
# Training pipeline (was src/training/Training.jl)
# =============================================================================
#
# Files moved from src/training/ to ext/training/. Code unchanged; only the
# enclosing module's load conditions changed (formerly always-loaded
# BlobBoards.Training submodule, now an extension that fires when the ML
# stack is present).

include("training/data.jl")
include("training/loss.jl")
include("training/eval.jl")
include("training/train.jl")

end  # module BlobBoardsTrainingExt
