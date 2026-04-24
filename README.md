# CS 7643 Final Project

Reproducing "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation" (`original-paper.pdf`). The paper's code is the `PAT` submodule. We add standalone baseline training and SLURM automation for the Georgia Tech PACE cluster.

## Setup

### Clone

```bash
git clone --recursive <repo-url>   # pulls PAT submodule
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Git LFS

Pretrained teacher weights are stored with [Git LFS](https://git-lfs.com/):

```bash
brew install git-lfs   # or: sudo apt-get install git-lfs
git lfs install        # once per machine
git lfs pull           # if you cloned before installing LFS
```

### Python Environment

Requires Python 3.10.15. We recommend [pyenv](https://github.com/pyenv/pyenv):

```bash
pyenv install 3.10.15        # .python-version pins this automatically
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On PACE, the environment is different — `setup_env.sh` loads cluster modules and activates a conda environment (`venv2`). See [Running on PACE](#running-on-pace-slurm).

### Pretrained Weights

The PAT submodule expects weights at `PAT/pretrained/`. Run the setup script to create a symlink from the root `pretrained/` directory:

```bash
./scripts/setup-pretrained-symlink.sh
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

CNN preset: minimal augmentation. ViT/MLP preset: mixup, cutmix, autoaugment, random erasing, gradient clipping. Both use cosine LR with warmup, label smoothing (0.1), AMP, and model EMA.

Teachers (distillation only): `swin_tiny_patch4_window7_224`, `vit_small_patch16_224`, `mixer_b16_224`, `convnext_tiny`. Checkpoints are in `pretrained/cifar_teachers/`.

### PAT Distillation (15 runs)

```bash
# List all jobs
bash run_one.sh --list

# Run one (from repo root)
bash run_one.sh pat_swin-resnet18
```

Job names follow `<distiller>_<teacher>-<student>`. The script looks up the config, teacher checkpoint, and flags, then `cd`s into `PAT/` to run training.

For custom arguments, run PAT's `train.py` directly:

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

# List available models
bash run_one.sh --list

# Run one
bash run_one.sh resnet18

# Run all sequentially
bash run_all.sh

# Run specific subset
bash run_all.sh resnet18 mobilenetv2_100
```

For custom arguments:

```bash
python train_student.py --model resnet18 --data-dir ../PAT/data --output-dir ./output --amp
```

Supports `--epochs`, `--batch-size`, `--lr` overrides and `--resume <path>`. `run_all.sh` respects `DATA_DIR` and `OUTPUT_DIR` environment variables.

**Output** — `output/<model>_standalone/`: `config.json`, `log.txt`, `best.pth`, `last.pth`.

**Tests:**

```bash
python -m pytest test_train_student.py -v   # CPU only, no data download needed
```

## Running on PACE (SLURM)

Each training run is submitted as a separate SLURM job — they run in parallel and failures are isolated.

### Project Structure

```
setup_env.sh                   # shared PACE environment (modules + conda)
run_one.sh                     # run one PAT distillation job locally
slurm/
  job.sh                       # SLURM batch script for PAT distillation
  submit_all.sh                # submits all 15 PAT jobs
standalone_training/
  train_student.py             # standalone training code
  run_one.sh                   # run one standalone job locally
  run_all.sh                   # run all 5 standalone jobs locally
  test_train_student.py        # unit tests
  slurm/
    job.sh                     # SLURM batch script for standalone
    submit_all.sh              # submits all 5 standalone jobs
```

PAT SLURM scripts live at the repo root (not inside the submodule) so teammates can pull them.

### Environment Setup

All PACE configuration is in `setup_env.sh` (modules, conda). Both SLURM job scripts source it automatically. Edit this file to change module versions or the environment path.

```bash
# Current defaults:
module load anaconda3
module load cuda/11.6
conda activate <repo_root>/venv2
```

For interactive work on a compute node:

```bash
source setup_env.sh
bash run_one.sh pat_swin-resnet18
```

### Resources Per Job

| Resource | Value |
|----------|-------|
| Nodes | 1 |
| CPUs | 8 |
| GPU | 1x V100 |
| Memory | 32 GB |
| Walltime | 8 hours |

Edit `#SBATCH` directives in the `job.sh` files to change these.

### Submitting Jobs

```bash
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

### Monitoring Jobs

```bash
squeue -u $USER                              # list jobs
scancel <job_id>                             # cancel one
tail -f slurm_outs/<jobname>_<jobid>.out     # follow output
```
