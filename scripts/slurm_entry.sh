#!/usr/bin/env bash
# No hardcoded partition/node/time/GPU type: caller supplies server-specific sbatch options.
# The persistent supervisor stays as the outer allocation owner across stages.
set -euo pipefail
: "${SLURM_JOB_ID:?Invoke this entry inside a Slurm allocation}"
# Slurm copies this script into its spool; BASH_SOURCE no longer locates the package.
cd -- "${COVERMEM_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the package root}}"
[[ -f scripts/run.sh ]] || { echo 'COVERMEM_PACKAGE_ROOT must point to covermem_selector' >&2; exit 2; }
exec bash scripts/run.sh "${1:-configs/A_loo.json}" "${2:-orchestration/slurm_${SLURM_JOB_ID}}"
