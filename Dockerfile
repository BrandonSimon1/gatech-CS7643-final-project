# syntax=docker/dockerfile:1.6
#
# CUDA 11.6 + cuDNN 8 on Ubuntu 20.04 — matches PyTorch 1.13 cu116 wheels.
# Requires NVIDIA driver >= 510.47 on the host and the NVIDIA Container Toolkit
# (run with `--gpus all`).
#
# Build:
#   git submodule update --init --recursive   # PAT submodule must be checked out
#   docker build -t pat-distill .
#
# Run (mount data and outputs so they persist):
#   docker run --gpus all --rm -it \
#     -v "$PWD/PAT/data:/workspace/PAT/data" \
#     -v "$PWD/pretrained:/workspace/pretrained" \
#     -v "$PWD/PAT/output:/workspace/PAT/output" \
#     -v "$PWD/standalone_training/output:/workspace/standalone_training/output" \
#     pat-distill bash run_one.sh pat_swin-resnet18

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.10 (deadsnakes PPA) + Git LFS ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3.10-distutils \
    && curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz \
        | tar -xz -C /tmp \
    && /tmp/git-lfs-3.4.1/install.sh \
    && rm -rf /tmp/git-lfs-* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && git lfs install --system \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python deps ─────────────────────────────────────────────────────────
# Install PyTorch + torchvision first against the cu116 index so we get
# the CUDA 11.6 build (default PyPI wheels for torch==1.13 ship cu117).
# The remaining requirements then resolve fine since torch is already pinned.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
        torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
    && python -m pip install -r requirements.txt

# ── Project code ────────────────────────────────────────────────────────
COPY . .

# Symlink PAT/pretrained -> ../pretrained so the PAT submodule finds
# teacher checkpoints (mount the host pretrained/ at /workspace/pretrained).
RUN bash scripts/setup-pretrained-symlink.sh || true

# Sanity check: fail the build early if torch can't see CUDA at runtime.
# (Skips the GPU check during build since the build host typically has none —
# this just verifies the cu116 build is present.)
RUN python -c "import torch; assert torch.version.cuda == '11.6', f'expected CUDA 11.6, got {torch.version.cuda}'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}')"

CMD ["bash"]
