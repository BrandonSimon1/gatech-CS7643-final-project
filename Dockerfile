# syntax=docker/dockerfile:1.6
#
# Self-contained training image for cloud GPU nodes.
#
# Base: pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel — already has Python +
# PyTorch 1.13 (cu116 build) + cuDNN 8 + CUDA 11.6 toolkit, so we just layer
# our extras on top. Requires NVIDIA driver >= 510.47 on the host and the
# NVIDIA Container Toolkit (run with `--gpus all`).
#
# Build (CI handles this; see .github/workflows/docker-build-push.yml):
#   git submodule update --init --recursive
#   git lfs pull
#   docker build -t pat-distill .
#
# Run — image is self-contained, no volume mounts required:
#   docker run --gpus all --rm -it pat-distill bash run_one.sh pat_swin-resnet18
#
# Or pull from ghcr:
#   docker run --gpus all --rm -it \
#       ghcr.io/brandonsimon1/gatech-cs7643-final-project:latest \
#       bash run_one.sh pat_swin-resnet18

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ── git, git-lfs, build tools ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz \
        | tar -xz -C /tmp \
    && /tmp/git-lfs-3.4.1/install.sh \
    && rm -rf /tmp/git-lfs-* \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python deps ─────────────────────────────────────────────────────────
# torch + torchvision are already pinned in the base image at the right
# CUDA 11.6 build, so pip just resolves the rest.
COPY requirements.txt ./
RUN pip install --upgrade pip wheel \
    && pip install -r requirements.txt

# ── Project code ────────────────────────────────────────────────────────
COPY . .

# Symlink PAT/pretrained -> ../pretrained so PAT finds the teacher checkpoints.
RUN bash scripts/setup-pretrained-symlink.sh || true

# Sanity check — fail the build early if torch's CUDA build doesn't match.
RUN python -c "import torch; assert torch.version.cuda == '11.6', f'expected CUDA 11.6, got {torch.version.cuda}'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}')"

CMD ["bash"]
