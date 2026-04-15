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

There are two training pipelines: **PAT distillation** (teacher-student knowledge distillation from the paper) and **standalone training** (students trained by themselves as baselines).

### PAT Distillation Training

The PAT submodule trains student models using knowledge distillation from pretrained teachers. The 15 training configurations cover three distillation methods (OFA, FitNet, PAT) across five student architectures and multiple teachers.

**Student models:**

| Category | Model | Config |
|----------|-------|--------|
| CNN | `resnet18` | `configs/cifar/cnn.yaml` |
| CNN | `mobilenetv2_100` | `configs/cifar/cnn.yaml` |
| ViT | `deit_tiny_patch16_224` | `configs/cifar/vit_mlp.yaml` |
| ViT | `swin_pico_patch4_window7_224` | `configs/cifar/vit_mlp.yaml` |
| MLP | `resmlp_12_224` | `configs/cifar/vit_mlp.yaml` |

**Run a single training locally:**

```bash
cd PAT
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 \
    train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml --model resnet18 \
    --teacher swin_tiny_patch4_window7_224 \
    --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --output ./output/cifar \
    --distiller pat
```

### Standalone Student Training (Baselines)

The `standalone_training/` directory contains code to train each student model on CIFAR-100 with only cross-entropy loss — no teacher, no distillation. This provides the baseline for measuring how much PAT improves each student.

Hyperparameters match the PAT configs exactly (same optimizer, LR schedule, augmentation, epochs) so the only variable is the presence or absence of distillation.

**Run a single student locally:**

```bash
cd standalone_training
python train_student.py --model resnet18 --data-dir ../PAT/data --output-dir ./output --amp
```

**Run all 5 students locally:**

```bash
cd standalone_training
bash run_all.sh
```

Available models: `resnet18`, `mobilenetv2_100`, `deit_tiny_patch16_224`, `swin_pico_patch4_window7_224`, `resmlp_12_224`.

**Tests:**

```bash
cd standalone_training
python -m pytest test_train_student.py -v
```

## Running on PACE (SLURM)

Both training pipelines include SLURM scripts for the Georgia Tech PACE cluster. Each training run is submitted as a separate job (one model per job) to avoid losing progress if a single run fails.

### SLURM Configuration

Both `PAT/slurm/job.sh` and `standalone_training/slurm/job.sh` request the following resources per job:

| Resource | Value |
|----------|-------|
| Nodes | 1 |
| CPUs | 8 |
| GPU | 1x V100 |
| Memory | 32 GB |
| Walltime | 8 hours |

### PACE Environment Setup

Before submitting, uncomment and edit the module/conda lines in each `slurm/job.sh` to match your PACE environment:

```bash
# module load anaconda3
# module load cuda/11.7
# conda activate <your_env>
```

### Submitting PAT Distillation Jobs

```bash
cd PAT

# Submit all 15 distillation jobs
bash slurm/submit_all.sh

# Submit only jobs for a specific distiller
bash slurm/submit_all.sh pat       # 12 PAT distiller jobs
bash slurm/submit_all.sh ofa       # 2 OFA jobs
bash slurm/submit_all.sh fitnet    # 1 FitNet job
```

### Submitting Standalone Baseline Jobs

```bash
cd standalone_training

# Submit all 5 student baseline jobs
bash slurm/submit_all.sh

# Submit a single student
bash slurm/submit_all.sh resnet18
```

### Optional SLURM Overrides

If your PACE setup requires an account or partition, set them before submitting:

```bash
export SLURM_ACCOUNT="gts-<PI_username>"
export SLURM_PARTITION="gpu"
bash slurm/submit_all.sh
```

These are only included in the `sbatch` call if set.

### Monitoring Jobs

```bash
squeue -u $USER                    # list your queued/running jobs
scancel <job_id>                   # cancel a specific job
tail -f slurm_outs/<jobname>_<id>.out  # follow live output
```
