#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
check_python="${COVERMEM_TEST_PYTHON:-.venv/bin/python}"
"$check_python" -m compileall -q cmselect tests supervision
"$check_python" -m unittest discover -s tests -v
"$check_python" -m unittest discover -s supervision/tests -v
"$check_python" -m cmselect.preflight --config "${1:-configs/A_loo.json}" --assets-only
