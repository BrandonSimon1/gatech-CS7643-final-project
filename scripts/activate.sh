#!/usr/bin/env bash
#
# Activate the project's Python virtual environment.
#
# Source this file (do not execute) so the activation persists in your shell:
#   source scripts/activate.sh
#
# If the venv does not exist, run scripts/setup_cloud_env.sh first.

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${_REPO_ROOT}/.venv/bin/activate" ]; then
    echo "ERROR: ${_REPO_ROOT}/.venv not found." >&2
    echo "Run: bash scripts/setup_cloud_env.sh" >&2
    unset _REPO_ROOT
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_REPO_ROOT}/.venv/bin/activate"

echo "Activated: $(python --version) at ${_REPO_ROOT}/.venv"
unset _REPO_ROOT
