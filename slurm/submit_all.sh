#!/bin/bash
#
# Submit all PAT CIFAR-100 distillation jobs to SLURM (one job per run).
#
# Optionally set account and partition if your PACE setup requires them:
#   export SLURM_ACCOUNT="gts-<PI_username>"
#   export SLURM_PARTITION="gpu"
#
# Run from the repository root (so SLURM_SUBMIT_DIR is the repo root and
# job.sh can cd into ./PAT).
#
# Usage:
#   bash slurm/submit_all.sh            # submit all 15 jobs
#   bash slurm/submit_all.sh pat        # submit only PAT distiller jobs
#   bash slurm/submit_all.sh ofa        # submit only OFA distiller jobs
#   bash slurm/submit_all.sh fitnet     # submit only FitNet distiller jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCT="${SLURM_ACCOUNT:-}"
PART="${SLURM_PARTITION:-}"
FILTER="${1:-all}"

mkdir -p slurm_outs

TEACHERS_DIR="./pretrained/cifar_teachers"

submit() {
    local job_name="$1" distiller="$2" config="$3" model="$4" teacher="$5" teacher_ckpt="$6" extra="${7:-}"

    echo "Submitting: ${job_name}"
    sbatch \
        ${ACCT:+--account="${ACCT}"} \
        ${PART:+--partition="${PART}"} \
        --job-name="${job_name}" \
        --export="ALL,CONFIG=${config},MODEL=${model},TEACHER=${teacher},TEACHER_CKPT=${teacher_ckpt},DISTILLER=${distiller},EXTRA_FLAGS=${extra}" \
        "${SCRIPT_DIR}/job.sh"
}

# ── OFA ─────────────────────────────────────────────────────────────────

if [[ "$FILTER" == "all" || "$FILTER" == "ofa" ]]; then
    submit "ofa_swin-resnet18" ofa \
        configs/cifar/cnn.yaml resnet18 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth"

    submit "ofa_swin-resmlp12" ofa \
        configs/cifar/vit_mlp.yaml resmlp_12_224 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth"
fi

# ── FitNet ──────────────────────────────────────────────────────────────

if [[ "$FILTER" == "all" || "$FILTER" == "fitnet" ]]; then
    submit "fitnet_swin-resnet18" fitnet \
        configs/cifar/cnn.yaml resnet18 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth"
fi

# ── PAT: CNN students ──────────────────────────────────────────────────

if [[ "$FILTER" == "all" || "$FILTER" == "pat" ]]; then
    submit "pat_swin-resnet18" pat \
        configs/cifar/cnn.yaml resnet18 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_vit-resnet18" pat \
        configs/cifar/cnn.yaml resnet18 \
        vit_small_patch16_224 "${TEACHERS_DIR}/vit_small_patch16_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_mixer-resnet18" pat \
        configs/cifar/cnn.yaml resnet18 \
        mixer_b16_224 "${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_swin-mobilenetv2" pat \
        configs/cifar/cnn.yaml mobilenetv2_100 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_vit-mobilenetv2" pat \
        configs/cifar/cnn.yaml mobilenetv2_100 \
        vit_small_patch16_224 "${TEACHERS_DIR}/vit_small_patch16_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_mixer-mobilenetv2" pat \
        configs/cifar/cnn.yaml mobilenetv2_100 \
        mixer_b16_224 "${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" \
        "--pin-mem"

    # ── PAT: ViT students ──────────────────────────────────────────────

    submit "pat_convnext-deit_t" pat \
        configs/cifar/vit_mlp.yaml deit_tiny_patch16_224 \
        convnext_tiny "${TEACHERS_DIR}/convnext_tiny_cifar100.pth" \
        "--pin-mem"

    submit "pat_mixer-deit_t" pat \
        configs/cifar/vit_mlp.yaml deit_tiny_patch16_224 \
        mixer_b16_224 "${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" \
        "--pin-mem"

    submit "pat_convnext-swin_p" pat \
        configs/cifar/vit_mlp.yaml swin_pico_patch4_window7_224 \
        convnext_tiny "${TEACHERS_DIR}/convnext_tiny_cifar100.pth" \
        "--pin-mem"

    submit "pat_mixer-swin_p" pat \
        configs/cifar/vit_mlp.yaml swin_pico_patch4_window7_224 \
        mixer_b16_224 "${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" \
        "--pin-mem"

    # ── PAT: MLP students ──────────────────────────────────────────────

    submit "pat_convnext-resmlp12" pat \
        configs/cifar/vit_mlp.yaml resmlp_12_224 \
        convnext_tiny "${TEACHERS_DIR}/convnext_tiny_cifar100.pth" \
        "--pin-mem"

    submit "pat_swin-resmlp12" pat \
        configs/cifar/vit_mlp.yaml resmlp_12_224 \
        swin_tiny_patch4_window7_224 "${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" \
        "--pin-mem"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
