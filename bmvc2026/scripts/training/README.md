# Training scripts — environment contract

Two experiments, each ONE config of record and ONE driver (no environment
variable configures either). The corpus is data of its own tier, written by its
one producer (Eval's `keyed_data_dir`); the descriptor is a result
(`keyed_cell_dir` under results_root()). Neither writes into another tier:

| experiment | config of record | driver | output |
|---|---|---|---|
| the training corpus (annotate + split) | `configs/training/annotations.toml` | `build_annotations.jl` | data tier `bb_descriptor_annotations`: `data_root()/bb_descriptor_annotations/annotations-<key>/{truth_images,splits}` (Eval's `keyed_data_dir`) |
| a descriptor trained on it | `configs/training/train_descriptor.toml` | `train_descriptor.jl` | `…/bmvc2026/descriptor-<key>/` |

Both are stages of `configs/reproduce.toml` (`annotations`, `train_descriptor`;
`default = false`: the paper's network of record is an INPUT, the bb_assets
tier, trained by the bootstrap chain the descriptor_training tiers publish).
Run them with `bash bmvc2026/scripts/reproduce/reproduce_results_dgx.sh
--only=annotations` (or `train_descriptor`), or call a driver directly.

The drivers run under the `bmvc2026` project, which declares everything the
ANNOTATION half needs (CUDA + cuDNN, FileIO/Images, Lux).

**The training LOOP's extension stack is deliberately NOT declared here.**
`BlobBoardsTrainingExt` triggers on Reactant + Enzyme + Optimisers + Zygote +
TensorBoardLogger, and Reactant cannot coexist with cuDNN in one environment
(it bundles its own — the reason inference and training are separate
extensions). `train_descriptor.jl` therefore needs those triggers on the load
path (the cluster's shared environment); without them it stops with
"BlobBoardsTrainingExt is not loaded" — that is the contract, not a bug.

`dataset_viewer.jl` (`--splits-dir=<run>/splits`) and `test_annotate_blobs.jl`
are interactive viewers (GLMakie, a display), not stages. Camera files are
canonical `vgio/cameras/v3`.
