# Training scripts — environment contract

These scripts run under the `bmvc2026` project (`@quickactivate` /
`--project=bmvc2026`), which declares everything the ANNOTATION half needs:
CUDA + cuDNN, Distributed/ClusterManagers (SLURM), FileIO/Images, and Lux.

**The training LOOP's extension stack is deliberately NOT declared here.**
`BlobBoardsTrainingExt` triggers on Reactant + Enzyme + Optimisers + Zygote +
TensorBoardLogger, and Reactant cannot coexist with cuDNN in one environment
(it bundles its own — the reason inference and training are separate
extensions). Training runs therefore load the trigger stack from the HOST
environment (the cluster's shared env; see `slurm/README.md`), not from this
project. A script that errors with "TrainingExt not loaded" is running in an
environment without those triggers — that is the contract, not a bug.

SLURM entry points live in `slurm/`; `build_annotations.jl` is the
annotation driver (see `build_annotations_pitfalls` notes for its failure
modes). Camera files are canonical `vgio/cameras/v3` — convert any RAID
copies of the dataset before pointing these scripts at them.
