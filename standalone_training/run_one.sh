#!/bin/bash
#
# Run a single standalone student training locally (no SLURM).
#
# Run from the standalone_training/ directory.
#
# Usage:
#   bash run_one.sh --list                  # show all available models
#   bash run_one.sh resnet18                # train a specific model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../PAT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

ALL_MODELS=(
    resnet18
    mobilenetv2_100
    deit_tiny_patch16_224
    swin_pico_patch4_window7_224
    resmlp_12_224
)

# ── List mode ───────────────────────────────────────────────────────────

list_models() {
    echo "Available standalone student models:"
    echo ""
    for model in "${ALL_MODELS[@]}"; do
        echo "  ${model}"
    done
}

# ── Main ────────────────────────────────────────────────────────────────

if [ $# -eq 0 ] || [[ "$1" == "--list" ]] || [[ "$1" == "-l" ]]; then
    list_models
    exit 0
fi

MODEL="$1"

# Validate model name
valid=false
for m in "${ALL_MODELS[@]}"; do
    if [[ "$m" == "$MODEL" ]]; then
        valid=true
        break
    fi
done

if [ "$valid" = false ]; then
    echo "ERROR: Unknown model '${MODEL}'" >&2
    echo "Run 'bash run_one.sh --list' to see available models." >&2
    exit 1
fi

echo "============================================================"
echo " Training: ${MODEL}"
echo " Data:     ${DATA_DIR}"
echo " Output:   ${OUTPUT_DIR}"
echo " Started:  $(date)"
echo "============================================================"

python "$SCRIPT_DIR/train_student.py" \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --no-amp

echo ""
echo "Finished: ${MODEL} at $(date)"
