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

pyenv will automatically activate 3.10.15 when you `cd` into this directory.

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
