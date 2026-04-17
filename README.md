# gatech-cs7643 final project

This is our final project, reproducing the paper at `original-paper.pdf`, "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation". The code for the paper is the `PAT` submodule.

## Cloning

Clone with `--recursive` to pull the PAT submodule automatically:

```bash
git clone --recursive <repo-url>
```

If you already cloned without `--recursive`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

## Git LFS

This repo uses [Git LFS](https://git-lfs.com/) to store large files (model weights, datasets, etc.).

### Install

```bash
# macOS
brew install git-lfs

# Debian/Ubuntu
sudo apt-get install git-lfs
```

Then initialize it once per machine:

```bash
git lfs install
```

### Fetching LFS objects

After installing LFS, a normal clone will fetch LFS-tracked files automatically:

```bash
git clone <repo-url>
```

If you cloned before installing LFS, pull the files with:

```bash
git lfs pull
```

**Note:** Fetching LFS objects (especially the pretrained model weights) may take a while depending on your connection speed. Be patient during the initial clone or `git lfs pull`.

### Tracking new file types

```bash
git lfs track "*.pt"
git lfs track "*.bin"
git add .gitattributes
```

Commit `.gitattributes` so the tracking rules are shared with collaborators. After that, `git add` / `git commit` / `git push` work as usual.

### Useful commands

```bash
git lfs ls-files   # list files tracked by LFS
git lfs status     # show LFS changes in the working tree
git lfs fetch      # download LFS objects for the current ref
```

## Environment Setup

This project uses Python 3.10.15. We recommend [pyenv](https://github.com/pyenv/pyenv) to manage Python versions and `virtualenv` to isolate dependencies.

### Install pyenv

```bash
# macOS
brew install pyenv

# Linux (via pyenv-installer)
curl https://pyenv.run | bash
```

After installing, add pyenv to your shell (follow the output from the installer, or see the [pyenv docs](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv)). Then restart your shell.

### Install the correct Python version

The `.python-version` file pins the project to Python 3.10.15. With pyenv configured:

```bash
pyenv install 3.10.15   # skip if already installed
```

pyenv will automatically activate 3.10.15 when you run `python` in this directory.

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Pretrained Weights Symlink

The pretrained model weights are stored in the root `pretrained/` directory (tracked via Git LFS). The PAT submodule expects them at `PAT/pretrained/`, so we use a symlink to bridge the two. Run the setup script to create it:

```bash
./scripts/setup-pretrained-symlink.sh
```

This creates the symlink `PAT/pretrained -> ../pretrained`. The script is idempotent — running it again when the symlink already exists is safe.

## Training

There are two training pipelines:

1. **PAT distillation** — teacher-student knowledge distillation from the paper (OFA, FitNet, and PAT methods)
2. **Standalone training** — students trained by themselves on CIFAR-100 as baselines (no teacher, no distillation)

Comparing the two measures how much each distillation method improves each student architecture.

### Student Models

Both pipelines train the same five student architectures. Each uses one of two hyperparameter presets that match the PAT paper's configs:

| Category | Model | Config | Optimizer | LR | Epochs |
|----------|-------|--------|-----------|------|--------|
| CNN | `resnet18` | `cnn.yaml` | SGD | 0.05 | 300 |
| CNN | `mobilenetv2_100` | `cnn.yaml` | SGD | 0.05 | 300 |
| ViT | `deit_tiny_patch16_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |
| ViT | `swin_pico_patch4_window7_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |
| MLP | `resmlp_12_224` | `vit_mlp.yaml` | AdamW | 5e-4 | 300 |

The CNN preset uses minimal augmentation (horizontal flip, random crop). The ViT/MLP preset adds mixup, cutmix, autoaugment, random erasing, color jitter, and gradient clipping. Both use cosine LR scheduling with warmup, label smoothing (0.1), AMP, and model EMA.

### Teacher Models (PAT Distillation Only)

The distillation runs pair students with the following pretrained teachers:

| Teacher | Pretrained Checkpoint |
|---------|-----------------------|
| `swin_tiny_patch4_window7_224` | `pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth` |
| `vit_small_patch16_224` | `pretrained/cifar_teachers/vit_small_patch16_224_cifar100.pth` |
| `mixer_b16_224` | `pretrained/cifar_teachers/mixer_b16_224_cifar100.pth` |
| `convnext_tiny` | `pretrained/cifar_teachers/convnext_tiny_cifar100.pth` |

These checkpoints are stored in Git LFS. Make sure you have run `git lfs pull` and `./scripts/setup-pretrained-symlink.sh` before training (see [Pretrained Weights Symlink](#pretrained-weights-symlink)).

---

### PAT Distillation Training

The PAT submodule trains student models using knowledge distillation from pretrained teachers. The full set of experiments is 15 training runs covering three distillation methods across various teacher-student pairs. The reference for all 15 commands is `PAT/command_cifar.sh`.

#### Running locally (bash)

All PAT training commands must be run from inside the `PAT/` directory, since `train.py` and all config/data paths are relative to it.

**Single run by name** (recommended — run from the repo root):

```bash
# List all 15 available jobs
bash run_one.sh --list

# Run a specific job
bash run_one.sh pat_swin-resnet18
```

Job names follow the pattern `<distiller>_<teacher>-<student>`. The script looks up the correct config, teacher checkpoint, and flags automatically, then `cd`s into `PAT/` to run training. Example output from `--list`:

```
NAME                         METHOD   STUDENT                        TEACHER
ofa_swin-resnet18            ofa      resnet18                       swin_tiny_patch4_window7_224
fitnet_swin-resnet18         fitnet   resnet18                       swin_tiny_patch4_window7_224
pat_swin-resnet18            pat      resnet18                       swin_tiny_patch4_window7_224
pat_vit-resnet18             pat      resnet18                       vit_small_patch16_224
...
```

**Single run manually** (if you need to customize arguments):

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

Key arguments:
- `--config` — hyperparameter preset (`configs/cifar/cnn.yaml` or `configs/cifar/vit_mlp.yaml`)
- `--model` — student architecture
- `--teacher` / `--teacher-pretrained` — teacher architecture and its checkpoint path
- `--distiller` — distillation method (`pat`, `ofa`, or `fitnet`)
- `--amp` — mixed-precision training
- `--model-ema` — exponential moving average of model weights
- `--pin-mem` — pin dataloader memory (used with the PAT distiller)

**Run all 15 commands sequentially** (not recommended — takes days on a single GPU):

```bash
cd PAT
bash command_cifar.sh
```

To swap the teacher or distiller, change the `--teacher`, `--teacher-pretrained`, and `--distiller` arguments. See `PAT/command_cifar.sh` for the full list of all 15 combinations.

Output (checkpoints, TensorBoard logs) is written to `PAT/output/cifar/`.

---

### Standalone Student Training (Baselines)

The `standalone_training/` directory trains each student model on CIFAR-100 with only cross-entropy loss — no teacher, no distillation. Hyperparameters are identical to the PAT configs so the only variable between the two pipelines is the presence or absence of distillation.

#### Running locally (bash)

**Single model by name** (recommended):

```bash
cd standalone_training

# List available models
bash run_one.sh --list

# Train a specific model
bash run_one.sh resnet18
```

The script validates the model name, sets up the correct data/output paths, and runs `train_student.py` with AMP enabled.

**Single model manually** (if you need to customize arguments):

```bash
cd standalone_training
python train_student.py --model resnet18 --data-dir ../PAT/data --output-dir ./output --amp
```

Key arguments:
- `--model` — one of: `resnet18`, `mobilenetv2_100`, `deit_tiny_patch16_224`, `swin_pico_patch4_window7_224`, `resmlp_12_224`
- `--data-dir` — path to the CIFAR-100 data root (will auto-download if missing)
- `--output-dir` — where to save checkpoints and logs
- `--amp` — mixed-precision training (enabled by default; disable with `--no-amp`)
- `--epochs`, `--batch-size`, `--lr` — optional overrides for the preset values
- `--resume <path>` — resume from a checkpoint

**All 5 students sequentially:**

```bash
cd standalone_training
bash run_all.sh
```

**Specific students only:**

```bash
cd standalone_training
bash run_all.sh resnet18 mobilenetv2_100
```

`run_all.sh` respects the `DATA_DIR` and `OUTPUT_DIR` environment variables:

```bash
DATA_DIR=/scratch/data OUTPUT_DIR=/scratch/output bash run_all.sh
```

#### Output

Each model writes to `output/<model>_standalone/`:
- `config.json` — full training configuration for reproducibility
- `log.txt` — per-epoch metrics (lr, train loss, val loss, val acc@1, val acc@5, EMA acc@1, best acc)
- `last.pth` — checkpoint from the most recent epoch
- `best.pth` — checkpoint with the highest validation accuracy (max of model and EMA)

#### Tests

```bash
cd standalone_training
python -m pytest test_train_student.py -v
```

The 33 unit tests verify model creation, config presets, transforms, optimizer/scheduler construction, accuracy computation, and smoke-test the training and evaluation loops on CPU with synthetic data. No GPU or CIFAR download needed.

---

## Running on PACE (SLURM)

Both training pipelines include SLURM scripts for the Georgia Tech PACE cluster. Each training run is submitted as a **separate SLURM job** (one model/configuration per job) rather than running all training sequentially in a single job. This means:
- Jobs run in parallel across the cluster if resources are available
- A failure in one run does not block the others
- Each job gets its own log file for easy debugging

### Directory Structure

```
setup_env.sh                   # shared PACE environment setup (modules + conda)
slurm/                         # PAT distillation SLURM scripts
  job.sh                       # batch script for one distillation run
  submit_all.sh                # submits all 15 distillation jobs

standalone_training/slurm/     # standalone baseline SLURM scripts
  job.sh                       # batch script for one student run
  submit_all.sh                # submits all 5 baseline jobs
```

The PAT SLURM scripts live at the **repo root** (in `slurm/`), not inside the PAT submodule, so changes stay in our repo and can be pulled by teammates. The scripts `cd` into `PAT/` at runtime before launching training.

### SLURM Resource Configuration

Both `slurm/job.sh` and `standalone_training/slurm/job.sh` request the same resources per job:

| Resource | Value | Notes |
|----------|-------|-------|
| Nodes | 1 | Single-node training |
| CPUs | 8 | Enough for dataloader workers |
| GPU | 1x V100 | Adjust type if your partition uses different GPUs |
| Memory | 32 GB | System RAM (not VRAM) |
| Walltime | 8 hours | 300 epochs on CIFAR-100; adjust if needed |

To change resource requests, edit the `#SBATCH` directives at the top of each `job.sh`.

### PACE Environment Setup

All PACE environment configuration (module loads, conda activation) is centralized in `setup_env.sh` at the repo root. Both SLURM job scripts source this file automatically, so you only need to edit it in one place.

The current defaults are:

```bash
module load anaconda3
module load cuda/11.6
conda activate <repo_root>/venv2
```

Edit `setup_env.sh` if your PACE setup uses different module versions or a different environment path. The exact module names depend on what is available on your PACE cluster (Phoenix, Hive, etc.). Run `module avail` on a login node to see available modules.

You can also source the script manually for interactive work on a PACE compute node:

```bash
# After ssh-ing into a compute node or starting an interactive session
source setup_env.sh

# Now you can run training directly
bash run_one.sh pat_swin-resnet18
cd standalone_training && bash run_one.sh resnet18
```

### Submitting PAT Distillation Jobs

Run all `slurm/` commands from the **repository root** (not from inside `PAT/`):

```bash
# Submit all 15 distillation jobs (OFA + FitNet + PAT)
bash slurm/submit_all.sh

# Submit only a specific distiller
bash slurm/submit_all.sh pat       # 12 PAT distiller jobs
bash slurm/submit_all.sh ofa       # 2 OFA distiller jobs
bash slurm/submit_all.sh fitnet    # 1 FitNet distiller job
```

Each job is named descriptively (e.g., `pat_swin-resnet18`, `ofa_swin-resmlp12`) so you can identify them in the queue. SLURM log files are written to `slurm_outs/<jobname>_<jobid>.out` at the repo root.

### Submitting Standalone Baseline Jobs

Run from inside `standalone_training/`:

```bash
cd standalone_training

# Submit all 5 student baseline jobs
bash slurm/submit_all.sh

# Submit specific students
bash slurm/submit_all.sh resnet18
bash slurm/submit_all.sh resnet18 deit_tiny_patch16_224
```

Jobs are named `standalone_<model>`. SLURM logs go to `standalone_training/slurm_outs/`.

### Optional SLURM Overrides

If your PACE setup requires a charge account or specific partition, set them as environment variables before submitting. These are **optional** — if unset, the flags are omitted from the `sbatch` call entirely:

```bash
# Set for both PAT and standalone submissions
export SLURM_ACCOUNT="gts-<PI_username>"
export SLURM_PARTITION="gpu"

# Then submit as usual
bash slurm/submit_all.sh
cd standalone_training && bash slurm/submit_all.sh
```

### Monitoring and Managing Jobs

```bash
squeue -u $USER                              # list your queued/running jobs
squeue -u $USER --format="%.10i %.30j %.8T"  # show job ID, name, and state
scancel <job_id>                             # cancel a specific job
scancel -u $USER                             # cancel all your jobs
tail -f slurm_outs/<jobname>_<jobid>.out     # follow live output from a job
```
