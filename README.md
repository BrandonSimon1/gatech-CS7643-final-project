# CS 7643 Final Project

Reproducing "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation" (`original-paper.pdf`). The paper's code is the `PAT` submodule. We add standalone baseline training and run scripts for cloud GPU nodes (with SLURM scripts for PACE kept as a legacy option).

## Setup

### Quick start (cloud GPU node)

For a fresh Ubuntu/Debian-style cloud node with an NVIDIA GPU and Python 3.10 already available:

```bash
git clone --recursive <repo-url>
cd final-project
bash scripts/setup_cloud_env.sh    # one-time: submodule, LFS, .venv, deps, symlinks
source scripts/activate.sh         # activate the venv in any shell
```

`setup_cloud_env.sh` is idempotent — re-run it any time. It runs a CUDA sanity check at the end so you'll know immediately if the GPU is wired up.

### Docker

A `Dockerfile` is provided with CUDA 11.6 + cuDNN 8 (matches the PyTorch 1.13 cu116 wheels). Requires NVIDIA driver ≥ 510.47 on the host and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git submodule update --init --recursive   # PAT submodule
git lfs pull                              # teacher weights on host

docker build -t pat-distill .

# Mount data, weights, and outputs so they persist outside the container
docker run --gpus all --rm -it \
    -v "$PWD/PAT/data:/workspace/PAT/data" \
    -v "$PWD/pretrained:/workspace/pretrained" \
    -v "$PWD/PAT/output:/workspace/PAT/output" \
    -v "$PWD/standalone_training/output:/workspace/standalone_training/output" \
    pat-distill bash run_one.sh pat_swin-resnet18
```

### Manual setup

If you'd rather do each step yourself:

```bash
# Clone with submodule
git clone --recursive <repo-url>
# (or: git submodule update --init --recursive)

# Git LFS for pretrained teacher weights
sudo apt-get install git-lfs   # or: brew install git-lfs
git lfs install
git lfs pull

# Python 3.10.15 (use pyenv if not on system Python)
pyenv install 3.10.15          # .python-version pins this
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Symlink PAT/pretrained -> ../pretrained (for teacher checkpoints)
bash scripts/setup-pretrained-symlink.sh
```

## Training

Two pipelines, same hyperparameters, same five student models:

1. **PAT distillation** — teacher-student knowledge distillation (OFA, FitNet, PAT methods)
2. **Standalone baselines** — students trained alone on CIFAR-100 (cross-entropy only)

Comparing the two isolates the effect of distillation.

### Models

| Category | Student | Config | Optimizer | LR | Epochs |
|----------|---------|--------|-----------|------|--------|
| CNN | `resnet18` | `cnn.yaml` | SGD | 0.05 | 300 |
| CNN | `mobilenetv2_100` | `cnn.yaml` | SGD | 0.05 | 300 |
| ViT | `deit_tiny_patch16_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |
| ViT | `swin_pico_patch4_window7_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |
| MLP | `resmlp_12_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |

CNN preset: minimal augmentation. ViT/MLP preset: mixup, cutmix, RandAugment, random erasing, gradient clipping. Both use cosine LR with warmup, label smoothing (0.1), and model EMA. AMP is on by default for PAT distillation but disabled for standalone training (see [AMP note](#a-note-on-amp) below).

Teachers (distillation only): `swin_tiny_patch4_window7_224`, `vit_small_patch16_224`, `mixer_b16_224`, `convnext_tiny`. Checkpoints are in `pretrained/cifar_teachers/`.

### PAT Distillation (15 runs)

From the repo root:

```bash
bash run_one.sh --list             # list all 15 jobs
bash run_one.sh pat_swin-resnet18  # run one
bash run_all.sh                    # run all 15 sequentially
bash run_all.sh pat                # only PAT jobs (12)
bash run_all.sh ofa                # only OFA jobs (2)
bash run_all.sh fitnet             # only FitNet jobs (1)
bash run_all.sh pat_swin-resnet18 pat_vit-resnet18   # specific subset
```

Job names follow `<distiller>_<teacher>-<student>`. By default `run_all.sh` continues through failures and reports them at the end. Set `STOP_ON_ERROR=1` to abort on first failure.

For custom arguments, invoke PAT's `train.py` directly:

```bash
cd PAT
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 \
    train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml --model resnet18 \
    --teacher swin_tiny_patch4_window7_224 \
    --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat
```

**Output** — `PAT/output/cifar/<experiment>/`: `args.yaml`, `train.log`, `summary.csv`, `checkpoint/`, `ema<decay>_checkpoint/`, `train_events/` (TensorBoard).

### Standalone Baselines (5 runs)

```bash
cd standalone_training
bash run_one.sh --list             # list models
bash run_one.sh resnet18           # run one
bash run_all.sh                    # all 5 sequentially
bash run_all.sh resnet18 mobilenetv2_100   # specific subset
```

For custom arguments:

```bash
python train_student.py --model resnet18 --data-dir ../PAT/data --output-dir ./output
```

Supports `--epochs`, `--batch-size`, `--lr` overrides and `--resume <path>`. `run_all.sh` respects `DATA_DIR` and `OUTPUT_DIR` environment variables.

**Output** — `output/<model>_standalone/`: `config.json`, `log.txt`, `best.pth`, `last.pth`.

**Tests:**

```bash
python -m pytest test_train_student.py -v   # CPU only, no data download needed
```

### A note on AMP

Standalone training defaults to `--no-amp` (FP32). cuBLAS FP16 tensor ops require matrix dimensions to be multiples of 8, and CIFAR-100's 100 classes violates that constraint, causing `CUBLAS_STATUS_INVALID_VALUE` on V100s. The performance hit on these small models is negligible. Pass `--amp` if you're on Ampere+ (A100, etc.) where this isn't enforced.

## Running on PACE (SLURM) — legacy

We originally targeted Georgia Tech's PACE cluster, but queue wait times pushed us to cloud GPUs. The SLURM scripts still work and are useful when PACE is available.

### Project layout

```
run_one.sh, run_all.sh         # cloud-node PAT distillation
setup_env.sh                   # PACE environment (modules + conda)
scripts/
  setup_cloud_env.sh           # one-time cloud setup
  activate.sh                  # source to activate venv
  setup-pretrained-symlink.sh
slurm/
  job.sh                       # SLURM batch script for PAT distillation
  submit_all.sh                # submits all 15 PAT jobs
  smoke_test.sh                # 10-min env / cuBLAS sanity check
standalone_training/
  train_student.py             # standalone training code
  run_one.sh, run_all.sh       # cloud-node standalone runs
  test_train_student.py        # unit tests
  slurm/
    job.sh                     # SLURM batch script for standalone
    submit_all.sh              # submits all 5 standalone jobs
```

PAT SLURM scripts live at the repo root (not inside the submodule) so teammates can pull them.

### Environment setup

`setup_env.sh` loads modules and activates the conda env. The SLURM job scripts source it automatically.

```bash
# Defaults:
module load anaconda3
module load cuda/11.6
conda activate <repo_root>/venv2
```

For interactive work on a PACE compute node:

```bash
source setup_env.sh
bash run_one.sh pat_swin-resnet18
```

### Resources per job

| Resource | Value |
|----------|-------|
| Nodes | 1 |
| CPUs | 8 |
| GPU | 1× V100 |
| Memory | 32 GB |
| Walltime | 8 hours |

Edit `#SBATCH` directives in the `job.sh` files to change these.

### Submitting jobs

```bash
# Smoke test first (recommended) — 10 min, prints driver/CUDA/PyTorch versions and runs GEMM checks
sbatch slurm/smoke_test.sh

# PAT distillation (from repo root)
bash slurm/submit_all.sh            # all 15 jobs
bash slurm/submit_all.sh pat        # 12 PAT jobs only
bash slurm/submit_all.sh ofa        # 2 OFA jobs only
bash slurm/submit_all.sh fitnet     # 1 FitNet job only

# Standalone baselines (from standalone_training/)
cd standalone_training
bash slurm/submit_all.sh            # all 5 jobs
bash slurm/submit_all.sh resnet18   # single model
```

If PACE requires an account or partition:

```bash
export SLURM_ACCOUNT="gts-<PI_username>"
export SLURM_PARTITION="gpu"
```

### Monitoring jobs

```bash
squeue -u $USER                              # list jobs
scancel <job_id>                             # cancel one
tail -f slurm_outs/<jobname>_<jobid>.out     # follow output
```
