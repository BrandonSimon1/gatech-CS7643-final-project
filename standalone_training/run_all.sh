#!/bin/bash
#
# Train all student models on CIFAR-100 without distillation.
# This provides the standalone baselines for comparison with PAT.
#
# Usage:
#   bash run_all.sh              # train all students sequentially
#   bash run_all.sh resnet18     # train a single student
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../PAT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# ── All student models ──────────────────────────────────────────────────

CNN_STUDENTS=(
    "resnet18"
    "mobilenetv2_100"
)

VIT_STUDENTS=(
    "deit_tiny_patch16_224"
    "swin_pico_patch4_window7_224"
)

MLP_STUDENTS=(
    "resmlp_12_224"
)

ALL_STUDENTS=("${CNN_STUDENTS[@]}" "${VIT_STUDENTS[@]}" "${MLP_STUDENTS[@]}")

# ── Helper ──────────────────────────────────────────────────────────────

train_model() {
    local model="$1"
    echo "============================================================"
    echo " Training: ${model}"
    echo " Started:  $(date)"
    echo "============================================================"

    python "$SCRIPT_DIR/train_student.py" \
        --model "$model" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo " Finished: ${model} at $(date)"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────────

if [ $# -gt 0 ]; then
    # Train only the model(s) specified on the command line
    for model in "$@"; do
        train_model "$model"
    done
else
    # Train all students
    echo "Training all ${#ALL_STUDENTS[@]} student models standalone on CIFAR-100"
    echo ""
    for model in "${ALL_STUDENTS[@]}"; do
        train_model "$model"
    done
    echo "============================================================"
    echo " All models trained. Results in: ${OUTPUT_DIR}"
    echo "============================================================"
fi
