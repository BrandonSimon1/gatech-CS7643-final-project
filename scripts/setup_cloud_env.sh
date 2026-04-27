#!/usr/bin/env bash
#
# One-time environment setup for a fresh cloud GPU node.
#
# Assumes: Ubuntu/Debian-style system with apt, an NVIDIA GPU + driver
# already installed, and Python 3.10 available (either system or pyenv).
#
# Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/setup_cloud_env.sh
#
# After this completes, activate the venv before training:
#   source scripts/activate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "════════════════════════════════════════════════════════════"
echo " Cloud node environment setup"
echo " Repo: ${REPO_ROOT}"
echo "════════════════════════════════════════════════════════════"

# ── 1. Submodule ────────────────────────────────────────────────────────

echo ""
echo "── [1/5] Initialize PAT submodule ──"
if [ ! -f PAT/train.py ]; then
    git submodule update --init --recursive
else
    echo "PAT submodule already initialized."
fi

# ── 2. Git LFS (for pretrained teacher weights) ─────────────────────────

echo ""
echo "── [2/5] Git LFS ──"
if ! command -v git-lfs >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing git-lfs via apt..."
        sudo apt-get update && sudo apt-get install -y git-lfs
    else
        echo "WARNING: git-lfs not found and apt-get unavailable."
        echo "Install it manually: https://git-lfs.com/"
    fi
fi
git lfs install
git lfs pull

# ── 3. Python venv ──────────────────────────────────────────────────────

echo ""
echo "── [3/5] Python virtual environment ──"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.10 first (system or pyenv)." >&2
    echo "  Hint: pyenv install 3.10.15  (.python-version pins this)" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    "${PYTHON_BIN}" -m venv .venv
    echo "Created .venv with $(${PYTHON_BIN} --version)"
else
    echo ".venv already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# ── 4. Pretrained symlink (PAT/pretrained -> ../pretrained) ─────────────

echo ""
echo "── [4/5] Pretrained symlink ──"
bash scripts/setup-pretrained-symlink.sh

# ── 5. Sanity check ─────────────────────────────────────────────────────

echo ""
echo "── [5/5] Sanity check ──"
python - <<'PY'
import torch
print(f"torch:        {torch.__version__}")
print(f"CUDA build:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_mem // (1024**2)} MB, sm_{p.major}{p.minor})")
PY

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Setup complete."
echo " Next: source scripts/activate.sh"
echo "════════════════════════════════════════════════════════════"
