#!/bin/bash
# Every result and generated asset of the paper on dgx01 (Slurm): reproduce.sh with the site dgx01. --dry-run, --only=…, --no-deps, status.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=dgx01 "$@"
