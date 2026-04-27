# syntax=docker/dockerfile:1.6
#
# Self-contained training image for cloud GPU nodes.
#
# Base: nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 — needed because PyTorch
# 1.13's cu116 wheels are built against CUDA 11.6 specifically. Python 3.10
# is installed via uv (Astral's package manager — downloads a prebuilt
# interpreter, no apt PPA gymnastics). Requires NVIDIA driver >= 510.47 on
# the host and the NVIDIA Container Toolkit (run with `--gpus all`).
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

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps (git, git-lfs, build tools, uv) ────────────────────────
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
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

# ── Python 3.10.15 + venv at /opt/venv ─────────────────────────────────
RUN uv python install 3.10.15 \
    && uv venv --python 3.10.15 /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# ── Python deps ─────────────────────────────────────────────────────────
# Install PyTorch + torchvision against the cu116 index explicitly
# (default PyPI wheels for torch==1.13 ship cu117).
COPY requirements.txt ./
RUN uv pip install --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
        torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
    && uv pip install --no-cache -r requirements.txt

WORKDIR /workspace

# ── Project code ────────────────────────────────────────────────────────
COPY . .

# Symlink PAT/pretrained -> ../pretrained so PAT finds the teacher checkpoints.
RUN bash scripts/setup-pretrained-symlink.sh || true

# Sanity check — fail the build early if torch's CUDA build doesn't match.
RUN python -c "import torch; assert torch.version.cuda == '11.6', f'expected CUDA 11.6, got {torch.version.cuda}'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}')"

CMD ["bash"]
