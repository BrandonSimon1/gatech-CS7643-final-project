#!/bin/bash
#
# PACE environment setup for training jobs.
#
# This script loads the required modules and activates the project's
# conda/venv environment. It is sourced by the SLURM job scripts and
# can also be sourced manually for interactive work on a PACE node.
#
# Usage (interactive):
#   source setup_env.sh
#
# Usage (from another script):
#   source "$(dirname "$0")/../setup_env.sh"
#
# Edit the module versions and environment path below to match your
# PACE cluster setup.

# ── Modules ─────────────────────────────────────────────────────────────

module load anaconda3
module load cuda/11.6

# ── Python environment ──────────────────────────────────────────────────
# Resolve the repo root so this works whether sourced from the repo root,
# from a subdirectory, or from a SLURM job.

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    _REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    _REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _REPO_ROOT="$(pwd)"
fi

conda activate "${_REPO_ROOT}/venv2"

unset _REPO_ROOT

echo "Environment ready: $(python --version), CUDA $(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo 'N/A')"
