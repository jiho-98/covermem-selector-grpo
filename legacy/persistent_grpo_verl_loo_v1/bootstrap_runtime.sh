#!/usr/bin/env bash
# A new isolated overlay; never pip-install into the existing inference/SFT env.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
recipe_root="$PWD"
core_python="${COVERMEM_CORE_PYTHON:-$recipe_root/../.venv/bin/python}"
peft_packages="${COVERMEM_PEFT_PACKAGES:-$recipe_root/../raw43500char_thought_teacher_sft_v4/runtime_packages}"
export TMPDIR="$recipe_root/.cache/tmp"
export PIP_CACHE_DIR="$recipe_root/.cache/pip"
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
if [[ -L .venv ]]; then
  echo 'Refusing to install through a shared venv symlink.' >&2
  exit 2
fi
if [[ ! -d vendor/verl/.git ]]; then
  git clone --depth 1 --branch v0.7.1 https://github.com/verl-project/verl.git vendor/verl
fi
[[ "$(git -C vendor/verl rev-parse HEAD)" == bec9ef74768dd201881cd4e54cd0385e87caae27 ]]
if [[ ! -x .venv/bin/python ]]; then "$core_python" -m venv --without-pip .venv; fi
"$core_python" - "$recipe_root" "$peft_packages" <<'PY'
from pathlib import Path
import sys, sysconfig
root=Path(sys.argv[1]); peft=Path(sys.argv[2]).resolve()
dest=root/'.venv/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/covermem_runtime.pth'
dest.write_text('\n'.join((sysconfig.get_path('purelib'),str(peft),str(root/'vendor/verl')))+'\n')
PY
.venv/bin/python -m pip install -r runtime-extra-requirements.txt
.venv/bin/python - <<'PY'
import verl
from verl.workers.actor.dp_actor import DataParallelPPOActor
print('Isolated verl runtime ready:', verl.__version__)
PY
