#!/bin/bash
#
# Run a single PAT distillation training locally (no SLURM).
#
# Run from the repository root.
#
# Usage:
#   bash run_one.sh --list                  # show all available job names
#   bash run_one.sh pat_swin-resnet18       # run a specific job

set -euo pipefail

TEACHERS_DIR="./pretrained/cifar_teachers"

# ── Job registry ────────────────────────────────────────────────────────
# Sets: DISTILLER, CONFIG, MODEL, TEACHER, TEACHER_CKPT, EXTRA_FLAGS

lookup_job() {
    EXTRA_FLAGS=""
    case "$1" in
        # OFA
        ofa_swin-resnet18)
            DISTILLER=ofa CONFIG=configs/cifar/cnn.yaml MODEL=resnet18
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" ;;
        ofa_swin-resmlp12)
            DISTILLER=ofa CONFIG=configs/cifar/vit_mlp.yaml MODEL=resmlp_12_224
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" ;;

        # FitNet
        fitnet_swin-resnet18)
            DISTILLER=fitnet CONFIG=configs/cifar/cnn.yaml MODEL=resnet18
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" ;;

        # PAT: CNN students
        pat_swin-resnet18)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=resnet18
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_vit-resnet18)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=resnet18
            TEACHER=vit_small_patch16_224 TEACHER_CKPT="${TEACHERS_DIR}/vit_small_patch16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_mixer-resnet18)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=resnet18
            TEACHER=mixer_b16_224 TEACHER_CKPT="${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_swin-mobilenetv2)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=mobilenetv2_100
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_vit-mobilenetv2)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=mobilenetv2_100
            TEACHER=vit_small_patch16_224 TEACHER_CKPT="${TEACHERS_DIR}/vit_small_patch16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_mixer-mobilenetv2)
            DISTILLER=pat CONFIG=configs/cifar/cnn.yaml MODEL=mobilenetv2_100
            TEACHER=mixer_b16_224 TEACHER_CKPT="${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;

        # PAT: ViT students
        pat_convnext-deit_t)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=deit_tiny_patch16_224
            TEACHER=convnext_tiny TEACHER_CKPT="${TEACHERS_DIR}/convnext_tiny_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_mixer-deit_t)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=deit_tiny_patch16_224
            TEACHER=mixer_b16_224 TEACHER_CKPT="${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_convnext-swin_p)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=swin_pico_patch4_window7_224
            TEACHER=convnext_tiny TEACHER_CKPT="${TEACHERS_DIR}/convnext_tiny_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_mixer-swin_p)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=swin_pico_patch4_window7_224
            TEACHER=mixer_b16_224 TEACHER_CKPT="${TEACHERS_DIR}/mixer_b16_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;

        # PAT: MLP students
        pat_convnext-resmlp12)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=resmlp_12_224
            TEACHER=convnext_tiny TEACHER_CKPT="${TEACHERS_DIR}/convnext_tiny_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;
        pat_swin-resmlp12)
            DISTILLER=pat CONFIG=configs/cifar/vit_mlp.yaml MODEL=resmlp_12_224
            TEACHER=swin_tiny_patch4_window7_224 TEACHER_CKPT="${TEACHERS_DIR}/swin_tiny_patch4_window7_224_cifar100.pth" EXTRA_FLAGS="--pin-mem" ;;

        *)  return 1 ;;
    esac
}

# ── List mode ───────────────────────────────────────────────────────────

ALL_JOBS="
ofa_swin-resnet18
ofa_swin-resmlp12
fitnet_swin-resnet18
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
"

list_jobs() {
    echo "Available PAT distillation jobs:"
    echo ""
    printf "  %-28s %-8s %-30s %s\n" "NAME" "METHOD" "STUDENT" "TEACHER"
    printf "  %-28s %-8s %-30s %s\n" "----" "------" "-------" "-------"
    for name in $ALL_JOBS; do
        lookup_job "$name"
        printf "  %-28s %-8s %-30s %s\n" "$name" "$DISTILLER" "$MODEL" "$TEACHER"
    done
}

# ── Main ────────────────────────────────────────────────────────────────

if [ $# -eq 0 ] || [[ "$1" == "--list" ]] || [[ "$1" == "-l" ]]; then
    list_jobs
    exit 0
fi

JOB_NAME="$1"

if ! lookup_job "$JOB_NAME"; then
    echo "ERROR: Unknown job name '${JOB_NAME}'" >&2
    echo "Run 'bash run_one.sh --list' to see available jobs." >&2
    exit 1
fi

echo "============================================================"
echo " Job:       ${JOB_NAME}"
echo " Distiller: ${DISTILLER}"
echo " Student:   ${MODEL}"
echo " Teacher:   ${TEACHER}"
echo " Config:    ${CONFIG}"
echo " Started:   $(date)"
echo "============================================================"

cd PAT

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
    ${EXTRA_FLAGS}

echo ""
echo "Finished: $(date)"
