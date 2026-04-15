#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o slurm_outs/%x_%j.out
#
# Required env vars (set via --export in submit_all.sh):
#   MODEL   student architecture name
#
# Usage:  sbatch --account=<acct> --partition=<part> \
#           --export=ALL,MODEL=resnet18 slurm/job.sh

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL is not set" >&2
    exit 1
fi

# ── Environment setup (edit these for your PACE environment) ────────────
# module load anaconda3
# module load cuda/11.7
# conda activate <your_env>

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_outs output

echo "============================================================"
echo " Job:     ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Node:    ${SLURMD_NODENAME}"
echo " GPU:     ${CUDA_VISIBLE_DEVICES:-N/A}"
echo " Model:   ${MODEL}"
echo " Started: $(date)"
echo "============================================================"

DATA_DIR="${DATA_DIR:-../PAT/data}"

python train_student.py \
    --model "${MODEL}" \
    --data-dir "${DATA_DIR}" \
    --output-dir ./output \
    --amp

echo ""
echo "Finished: $(date)"
