#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
recipe_root="$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$recipe_root:$recipe_root/vendor/verl"
unset COVERMEM_PERFORMANCE_PROFILE
export TMPDIR="$recipe_root/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$recipe_root/.cache" HF_HOME="$recipe_root/.cache/hf"
export TORCH_HOME="$recipe_root/.cache/torch" TORCHINDUCTOR_CACHE_DIR="$recipe_root/.cache/inductor"
mkdir -p "$TMPDIR"
stage="${1:?train or evaluate}";shift
if [[ "$stage" == train ]]; then
  export COVERMEM_NPROC="${1:?GPU count}";shift
  exec "$recipe_root/.venv/bin/python" -u "$recipe_root/pipeline.py" "$@"
elif [[ "$stage" == evaluate ]]; then
  exec "$recipe_root/.venv/bin/python" -u "$recipe_root/evaluate.py" "$@"
else
  exit 2
fi
