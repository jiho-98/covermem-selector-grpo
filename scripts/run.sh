#!/usr/bin/env bash
# Run INSIDE an existing allocation (Slurm) or on an authorized dedicated GPU.
# This script never invokes sbatch/scancel and never guesses a GPU number.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
: "${CUDA_VISIBLE_DEVICES:?Expose exactly one GPU authorized for this experiment}"
run_config="${1:-configs/A_loo.json}"
queue_dir="${2:-orchestration/$(basename "$run_config" .json)}"
.venv/bin/python scripts/queue.py init --config "$run_config" --root "$queue_dir"
exec .venv/bin/python supervision/supervisor.py --root "$queue_dir"
