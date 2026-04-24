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
#   CONFIG         configs/cifar/cnn.yaml or configs/cifar/vit_mlp.yaml
#   MODEL          student architecture name
#   TEACHER        teacher architecture name
#   TEACHER_CKPT   path to teacher checkpoint
#   DISTILLER      ofa | fitnet | pat
#   EXTRA_FLAGS    (optional) e.g. "--pin-mem"
#
# Usage:  sbatch --account=<acct> --partition=<part> \
#           --export=ALL,CONFIG=...,MODEL=...,... slurm/job.sh
#
# Run this script from the repository root (where the PAT submodule lives).

set -euo pipefail

# ── Validate required vars ──────────────────────────────────────────────
for var in CONFIG MODEL TEACHER TEACHER_CKPT DISTILLER; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set" >&2
        exit 1
    fi
done

# ── Environment setup ───────────────────────────────────────────────────
source "${SLURM_SUBMIT_DIR:-.}/setup_env.sh"

# SLURM's -o path is relative to SLURM_SUBMIT_DIR (repo root); ensure it exists.
mkdir -p "${SLURM_SUBMIT_DIR:-.}/slurm_outs"

# All training paths (configs/, pretrained/, train.py) are relative to the
# PAT submodule, so cd in before launching training.
cd "${SLURM_SUBMIT_DIR:-.}/PAT"
mkdir -p output/cifar

echo "============================================================"
echo " Job:       ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Node:      ${SLURMD_NODENAME}"
echo " GPU:       ${CUDA_VISIBLE_DEVICES:-N/A}"
echo " Distiller: ${DISTILLER}"
echo " Teacher:   ${TEACHER}"
echo " Student:   ${MODEL}"
echo " Config:    ${CONFIG}"
echo " Started:   $(date)"
echo "============================================================"

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((29500 + RANDOM % 1000)) \
    train.py ./data \
    --dataset cifar100 \
    --num-classes 100 \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --teacher "${TEACHER}" \
    --teacher-pretrained "${TEACHER_CKPT}" \
    --amp --model-ema \
    --output ./output/cifar \
    --distiller "${DISTILLER}" \
    ${EXTRA_FLAGS:-}

echo ""
echo "Finished: $(date)"
