#!/usr/bin/env bash
# Creates a symlink from PAT/pretrained -> ../pretrained so the PAT submodule
# can find the pretrained model weights stored (via Git LFS) in the root
# pretrained/ directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINK_PATH="$REPO_ROOT/PAT/pretrained"
TARGET="../pretrained"

if [ ! -d "$REPO_ROOT/PAT" ]; then
    echo "Error: PAT submodule not found. Initialize it first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

if [ -L "$LINK_PATH" ]; then
    echo "Symlink already exists: $LINK_PATH -> $(readlink "$LINK_PATH")"
    exit 0
fi

if [ -e "$LINK_PATH" ]; then
    echo "Error: $LINK_PATH already exists and is not a symlink."
    echo "Remove it manually if you want this script to create the symlink."
    exit 1
fi

ln -s "$TARGET" "$LINK_PATH"
echo "Created symlink: $LINK_PATH -> $TARGET"
