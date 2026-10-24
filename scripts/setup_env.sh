#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''
setup_python="${COVERMEM_PYTHON:-python3.10}"
"$setup_python" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
if [[ -L .venv ]]; then echo 'Refusing to install through a shared venv symlink' >&2; exit 2; fi
if [[ ! -x .venv/bin/python ]]; then "$setup_python" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
torch_index="${COVERMEM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
.venv/bin/python -m pip install torch==2.11.0 --index-url "$torch_index"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
mkdir -p validation
.venv/bin/python -m pip freeze > validation/installed.freeze.txt
.venv/bin/python -c 'from cmselect.backend import runtime_versions; import json; print(json.dumps(runtime_versions(),indent=2))'
echo 'Environment installed. No GPU initialized; next: bash scripts/download_model.sh'
