#!/bin/bash
set -euo pipefail

ROOT_DIR=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Build the environment image. Uses ${ROOT_DIR}/Dockerfile and ${ROOT_DIR}/lyra2.toml.
# Submodules must be initialized on the host first:
#   git submodule update --init --recursive
podman build --progress=plain -t lyra2:latest . 2>&1 | tee build.log

# Ensure proper Lustre striping for multi-node parallel reads
lfs setstripe \
    --component-end 4M --stripe-count 1 \
    --component-end 64M --stripe-count 4 \
    --component-end -1 --stripe-count 32 --stripe-size 4M \
    $SCRATCH/ce-images

# Remove the older `.sqsh`
rm -f $SCRATCH/ce-images/lyra2.sqsh

# A .sqsh file is a SquashFS image: a compressed, read-only filesystem.
# HPC nodes typically don't run Docker/Podman daemons.
# Enroot unpacks the sqsh into a lightweight user-namespace container at job launch.
enroot import -x mount -o $SCRATCH/ce-images/lyra2.sqsh podman://lyra2:latest

# Mount the sqsh as a FUSE filesystem so Pylance / pyright can resolve the
# container's site-packages from the login node (per-node mount, recreated
# each time this script runs).
MNT_DIR=/tmp/${USER}-ce-images/mnt/lyra2
fusermount -u "$MNT_DIR" 2>/dev/null || true
rm -rf "$MNT_DIR"
mkdir -p "$MNT_DIR"
squashfuse $SCRATCH/ce-images/lyra2.sqsh "$MNT_DIR"
echo "FUSE mount: $MNT_DIR"
