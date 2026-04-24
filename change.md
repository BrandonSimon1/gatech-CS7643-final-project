# Change Log

## 2026-04-22 — Fix CUBLAS_STATUS_INVALID_VALUE with AMP on V100

**Problem:**
`RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx` during backward pass of standalone ResNet18 training with AMP (FP16) on V100 GPUs.

**Root cause:**
cuBLAS FP16 GEMMs require matrix dimensions to be multiples of 8. ResNet18's final FC layer is `512 × 100` — `num_classes=100` is not a multiple of 8, causing `cublasGemmEx` to reject the operation during the backward pass. V100 (compute capability 7.0) enforces this constraint strictly.

**Fix:**
Disabled AMP across all standalone training launch scripts and changed the default to off. FP32 training on V100 with small models like ResNet18 has negligible speed difference. The `--amp` flag remains available for Ampere+ GPUs (A100, etc.) where this constraint does not apply.

**Files changed:**

| File | Change |
|------|--------|
| `standalone_training/train_student.py` | `--amp` default changed from `True` to `False` |
| `standalone_training/run_one.sh` | `--amp` → `--no-amp` |
| `standalone_training/run_all.sh` | `--amp` → `--no-amp` |
| `standalone_training/slurm/job.sh` | `--amp` → `--no-amp` |
