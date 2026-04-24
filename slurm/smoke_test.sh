#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o slurm_outs/smoke_test_%j.out
#
# Quick smoke test: loads the PACE environment and prints driver, toolkit,
# PyTorch, and GPU info so you can verify everything lines up.
#
# Usage:
#   sbatch slurm/smoke_test.sh
#   # or with account/partition:
#   sbatch --account=gts-<PI> --partition=gpu slurm/smoke_test.sh

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-.}/slurm_outs"

echo "============================================================"
echo " Smoke Test"
echo " Job:  ${SLURM_JOB_NAME:-smoke_test} (${SLURM_JOB_ID})"
echo " Node: ${SLURMD_NODENAME}"
echo " Date: $(date)"
echo "============================================================"
echo ""

# ── Driver & toolkit ───────────────────────────────────────────────────
echo "── nvidia-smi ──"
nvidia-smi
echo ""

echo "── NVIDIA driver version ──"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
echo ""

echo "── nvcc (CUDA toolkit) ──"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
echo ""

# ── Load project environment ───────────────────────────────────────────
echo "── Loading project environment ──"
source "${SLURM_SUBMIT_DIR:-.}/setup_env.sh"
echo ""

# ── Python / PyTorch ───────────────────────────────────────────────────
echo "── Python ──"
which python
python --version
echo ""

echo "── PyTorch ──"
python -c "
import torch
print(f'torch version:       {torch.__version__}')
print(f'torch CUDA version:  {torch.version.cuda}')
print(f'cuDNN version:       {torch.backends.cudnn.version()}')
print(f'CUDA available:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count:           {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}  (compute {props.major}.{props.minor}, {props.total_mem // (1024**2)} MB)')
"
echo ""

# ── Quick GEMM sanity check ────────────────────────────────────────────
echo "── cuBLAS GEMM sanity check (FP32, 512x100) ──"
python -c "
import torch
device = torch.device('cuda')
# Mimics resnet18 final layer backward: 512 x 100 matmul
a = torch.randn(128, 512, device=device)
b = torch.randn(512, 100, device=device)
c = a @ b
loss = c.sum()
loss.backward()
print('FP32 GEMM (512x100): OK')
"
echo ""

echo "── cuBLAS GEMM sanity check (FP16, 512x100) ──"
python -c "
import torch
device = torch.device('cuda')
a = torch.randn(128, 512, device=device, dtype=torch.float16)
b = torch.randn(512, 100, device=device, dtype=torch.float16)
try:
    c = a @ b
    loss = c.float().sum()
    print('FP16 GEMM (512x100): OK')
except RuntimeError as e:
    print(f'FP16 GEMM (512x100): FAILED — {e}')
"
echo ""

# ── Key packages ───────────────────────────────────────────────────────
echo "── Key packages ──"
python -c "
import timm; print(f'timm:         {timm.__version__}')
import torchvision; print(f'torchvision:  {torchvision.__version__}')
import numpy; print(f'numpy:        {numpy.__version__}')
"
echo ""

echo "============================================================"
echo " Smoke test complete: $(date)"
echo "============================================================"
