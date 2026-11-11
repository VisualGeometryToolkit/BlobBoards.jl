#!/bin/bash
# Every result and generated asset of the paper on the desktop (one GPU, no Slurm): reproduce.sh with the site desktop.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=desktop "$@"
