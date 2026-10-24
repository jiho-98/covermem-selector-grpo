#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
recipe_root="$PWD"
export TMPDIR="$recipe_root/.cache/tmp"
export XDG_CACHE_HOME="$recipe_root/.cache"
export HF_HOME="$recipe_root/.cache/huggingface"
export TORCH_HOME="$recipe_root/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$recipe_root/.cache/inductor"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$recipe_root/vendor/verl:$recipe_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR"
if [[ "${1:-}" != "--train" ]]; then
  export CUDA_VISIBLE_DEVICES=''
  exec "$recipe_root/.venv/bin/python" "$recipe_root/train.py" "$@"
fi
shift
: "${SLURM_JOB_ID:?GPU training must be launched inside a Slurm allocation}"
: "${COVERMEM_NPROC:?Set the allocated GPU count (1 to 4) after hardware is selected}"
exec "$recipe_root/.venv/bin/python" -m torch.distributed.run --standalone \
  --nproc_per_node="$COVERMEM_NPROC" "$recipe_root/train.py" --train "$@"
