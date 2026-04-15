#!/bin/bash
#
# Submit all standalone student training jobs to SLURM (one job per model).
#
# Optionally set account and partition if your PACE setup requires them:
#   export SLURM_ACCOUNT="gts-<PI_username>"
#   export SLURM_PARTITION="gpu"
#
# Usage:
#   bash slurm/submit_all.sh                # submit all 5 students
#   bash slurm/submit_all.sh resnet18       # submit a single student

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCT="${SLURM_ACCOUNT:-}"
PART="${SLURM_PARTITION:-}"

ALL_STUDENTS=(
    resnet18
    mobilenetv2_100
    deit_tiny_patch16_224
    swin_pico_patch4_window7_224
    resmlp_12_224
)

submit() {
    local model="$1"
    echo "Submitting: standalone_${model}"
    sbatch \
        ${ACCT:+--account="${ACCT}"} \
        ${PART:+--partition="${PART}"} \
        --job-name="standalone_${model}" \
        --export="ALL,MODEL=${model}" \
        "${SCRIPT_DIR}/job.sh"
}

if [ $# -gt 0 ]; then
    for model in "$@"; do
        submit "$model"
    done
else
    for model in "${ALL_STUDENTS[@]}"; do
        submit "$model"
    done
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
