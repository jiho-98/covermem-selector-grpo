#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path('models/Qwen3-4B-Instruct-2507')
snapshot_download('Qwen/Qwen3-4B-Instruct-2507',
    revision='cdbee75f17c01a7cc42f958dc650907174af0554', local_dir=root,
    allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md'])
print('Base model downloaded:',root.resolve())
PY
