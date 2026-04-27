#!/bin/bash
#
# Run all PAT CIFAR-100 distillation jobs sequentially on a single node.
# Suitable for cloud GPU nodes where SLURM is not available.
#
# Usage:
#   bash run_all.sh                # run all 15 jobs
#   bash run_all.sh pat            # only PAT distiller jobs (12)
#   bash run_all.sh ofa            # only OFA distiller jobs (2)
#   bash run_all.sh fitnet         # only FitNet distiller jobs (1)
#   bash run_all.sh job1 job2 ...  # run a specific list of job names
#
# Run from the repository root (run_one.sh handles cd'ing into PAT/).
#
# Environment variables:
#   STOP_ON_ERROR=1   abort on first failing job (default: continue)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

# ── Job groups ──────────────────────────────────────────────────────────

OFA_JOBS=(
    ofa_swin-resnet18
    ofa_swin-resmlp12
)

FITNET_JOBS=(
    fitnet_swin-resnet18
)

PAT_JOBS=(
    pat_swin-resnet18
    pat_vit-resnet18
    pat_mixer-resnet18
    pat_swin-mobilenetv2
    pat_vit-mobilenetv2
    pat_mixer-mobilenetv2
    pat_convnext-deit_t
    pat_mixer-deit_t
    pat_convnext-swin_p
    pat_mixer-swin_p
    pat_convnext-resmlp12
    pat_swin-resmlp12
)

# ── Select jobs ─────────────────────────────────────────────────────────

JOBS=()

if [ $# -eq 0 ]; then
    JOBS=("${OFA_JOBS[@]}" "${FITNET_JOBS[@]}" "${PAT_JOBS[@]}")
elif [ $# -eq 1 ]; then
    case "$1" in
        all)    JOBS=("${OFA_JOBS[@]}" "${FITNET_JOBS[@]}" "${PAT_JOBS[@]}") ;;
        ofa)    JOBS=("${OFA_JOBS[@]}") ;;
        fitnet) JOBS=("${FITNET_JOBS[@]}") ;;
        pat)    JOBS=("${PAT_JOBS[@]}") ;;
        *)      JOBS=("$1") ;;
    esac
else
    JOBS=("$@")
fi

# ── Run ─────────────────────────────────────────────────────────────────

echo "Running ${#JOBS[@]} PAT distillation job(s) sequentially"
echo "Started: $(date)"
echo ""

FAILED=()
for job in "${JOBS[@]}"; do
    echo "════════════════════════════════════════════════════════════"
    echo " Job: ${job}"
    echo "════════════════════════════════════════════════════════════"

    if bash "${SCRIPT_DIR}/run_one.sh" "${job}"; then
        echo ""
        echo "✓ ${job} completed"
    else
        echo ""
        echo "✗ ${job} failed (exit $?)"
        FAILED+=("${job}")
        if [ "${STOP_ON_ERROR}" = "1" ]; then
            echo "STOP_ON_ERROR=1 — aborting."
            break
        fi
    fi
    echo ""
done

# ── Summary ─────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════"
echo " Finished: $(date)"
echo " Total: ${#JOBS[@]}, failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed jobs:"
    for j in "${FAILED[@]}"; do
        echo "   - ${j}"
    done
    exit 1
fi
echo "════════════════════════════════════════════════════════════"
