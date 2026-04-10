# gatech-cs7643 final project

This is our final project, reproducing the paper at `original-paper.pdf`, "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation". The code for the paper is the `PAT` submodule.

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

### Cloning

After installing LFS, a normal clone will fetch LFS-tracked files automatically:

```bash
git clone <repo-url>
```

If you cloned before installing LFS, pull the files with:

```bash
git lfs pull
```

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
