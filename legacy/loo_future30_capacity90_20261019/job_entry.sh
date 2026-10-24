#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kimjh/Covermem/loo_future30_capacity90_20261019
PY=/home/kimjh/Covermem/persistent_grpo_verl_loo_v1/.venv/bin/python
SOURCE=/home/kimjh/Covermem/persistent_grpo_verl_loo_v1
export PYTHONPATH="$ROOT:$SOURCE:$SOURCE/vendor/verl"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$ROOT/.cache/tmp" XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset PYTHONOPTIMIZE
case "$1" in
  smoke) exec "$PY" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$ROOT/smoke.py" ;;
  train)
    export COVERMEM_PERFORMANCE_PROFILE="$ROOT/smoke/runtime_smoke/PERFORMANCE_PROFILE.json"
    exec "$PY" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$ROOT/train_future.py" ;;
  evaluate) exec "$PY" -B -u "$ROOT/launch_evaluation.py" ;;
  analyze) exec "$PY" -B -u "$ROOT/analyze_run.py" ;;
  *) exit 2 ;;
esac
