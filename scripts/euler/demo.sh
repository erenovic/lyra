#!/bin/bash
# Run the stock Lyra-2 sampling demo (src/external/lyra/run_sample.sh) on a Euler GPU node.
#
#   bash scripts/euler/lyra/demo.sh          # e.g. inside an alloc on eu-g7-004 (RTX Pro 6000, 96GB)
#
# ---- One-time venv setup (already applied to ~/memorykrea; kept here for reference) ----
# Lyra needs these packages on top of the training venv (versions from lyra's requirements.txt):
#   $HOME/memorykrea/bin/python -m pip install loguru==0.7.3 fvcore==0.1.5.post20221221 iopath==0.1.10 megatron-core==0.12.1 boto3==1.38.31 ffmpegcv peft==0.17.1
#   $HOME/memorykrea/bin/python -m pip install --no-deps "git+https://github.com/microsoft/MoGe.git"   # full install drags gradio (quota blowup + hub 1.x break)
# Stage 2 (vipe + DA3 recon) additionally needs:
#   $HOME/memorykrea/bin/python -m pip install rerun-sdk python-pycg OpenEXR==3.4.11 evo
#   $HOME/memorykrea/bin/python -m pip install --no-deps kornia kornia_rs   # --no-deps: don't let pip touch torch
#   $HOME/memorykrea/bin/python -m pip install numpy==1.26.4   # megatron-core/rerun bump numpy to 2.x; repin the uv.lock version
# vipe's droid.pth auto-download saves Drive's HTML viewer page (gdown without --fuzzy); fix with:
#   python -m gdown --fuzzy "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view" -O ~/.cache/torch/hub/droid_slam/droid.pth
# lyra also has NESTED submodules (depth_anything_3, vipe): git submodule update --init --recursive src/external/lyra
#
# transformer_engine must be source-built (prebuilt core wheel + compiled torch binding).
# TE 2.17 needs torch>=2.11 headers; 2.16.1 matches our torch 2.10.0+cu128. Build on a GPU node
# inside the `act memorykrea` env (~/.bashrc: modules cuda/12.8.0, cudnn, cmake, eth_proxy + venv);
# cudnn/nccl headers come from torch's bundled pip packages:
#   act memorykrea
#   SITE=$HOME/memorykrea/lib/python3.12/site-packages
#   export CUDNN_PATH=$SITE/nvidia/cudnn CPATH=$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include
#   export LIBRARY_PATH=$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib
#   export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" MAX_JOBS=6
#   python -m pip install "transformer_engine[pytorch]==2.16.1" --no-build-isolation
# ----------------------------------------------------------------------------------------

# set -uo pipefail
cd /cluster/scratch/ecetin/MemoryKrea || exit 1

# Env (HF_HOME cache, PYTHONPATH incl. src/external/lyra) + the memorykrea venv python first
# on PATH so run_sample.sh's bare `python` resolves to it.
# `act memorykrea` (bashrc function): module stack (cuda/12.8.0, cudnn, eth_proxy, ...) + venv
# activate. TE needs the cuda libs at import time, so this replaces any manual LD_LIBRARY_PATH.
# PS1 is set first to defeat bashrc's non-interactive early-return guard; nounset is relaxed
# while sourcing it (bashrc references unset vars like HISTCONTROL).
set +u
export PS1="${PS1:-demo$ }"
source ~/.bashrc
act memorykrea

# vipe (stage 2) JIT-builds its CUDA extension on first import and includes <eigen3/Eigen/*>.
# The eigen module prepends its include dirs to CPATH, but needs gcc/12.2.0 loaded first
# (loading both in one `module load` silently skips eigen). The gcc reload downgrades cuda
# to 12.1.1 as a side effect, so cuda/12.8.0 is re-loaded after. Module juggling must happen
# BEFORE the venv PATH prepend below or `python` stops being the memorykrea venv.
# The vipe build caches in ~/.cache/torch_extensions.
module load gcc/12.2.0
module load eigen
module load cuda/12.8.0
set -u

set -a; source .env; [ -f .env.secret ] && source .env.secret; set +a
export PATH="$HOME/memorykrea/bin:$PATH"

# NOTE: TE routes attention through the repo's editable flash-attn
# (src/external/xformers/third_party/flash-attention), which must be built with sm_120 for the
# RTX Pro 6000 (FLASH_ATTN_CUDA_ARCHS="80;120", see tmp/lyra_setup/rebuild_flash_attn.sh).

# Node-local caches (triton JIT etc.).
CACHE=/tmp/${USER}-lyracache; mkdir -p "$CACHE/triton"
export TMPDIR="$CACHE" TRITON_CACHE_DIR="$CACHE/triton"

# VIPE/DA3 checkpoints carry pickled configs; torch>=2.6 weights_only default rejects them
# (trusted NVIDIA release files, so restore the old torch.load behavior).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

LYRA=src/external/lyra

# Checkpoints live in the standard HF hub cache (.env HF_HOME=.cache/huggingface, relative to
# the repo root). Download only if the hub copy is missing (needs internet: login node, or
# eth_proxy on compute nodes via `act`); the run itself doesn't.
SNAP_CK=$(ls -d "$PWD"/.cache/huggingface/hub/models--nvidia--Lyra-2.0/snapshots/*/checkpoints 2>/dev/null | head -1)
if [ -z "$SNAP_CK" ]; then
  SNAP=$(realpath "$(hf download nvidia/Lyra-2.0 --include "checkpoints/*")")  # absolute, or the symlink breaks
  SNAP_CK="$SNAP/checkpoints"
fi

# Stage the checkpoints on the node-local NVMe (SLURM job scratch /scratch/tmp.<jobid>.<user>)
# so model load streams from local disk instead of Lustre. rsync -aL dereferences the hub
# snapshot's blob symlinks; reruns on the same alloc are a no-op. Falls back to the hub copy
# when there is no job scratch (e.g. login node). $LYRA/checkpoints is re-pointed every run,
# so a stale symlink from another node/alloc self-heals.
LOCAL=$(ls -d /scratch/tmp.*."$USER" 2>/dev/null | head -1)
if [ -n "$LOCAL" ]; then
  echo "staging checkpoints -> $LOCAL/lyra2_checkpoints (no-op if already there)"
  rsync -aL "$SNAP_CK/" "$LOCAL/lyra2_checkpoints/"
  ln -sfn "$LOCAL/lyra2_checkpoints" "$LYRA/checkpoints"
else
  ln -sfn "$SNAP_CK" "$LYRA/checkpoints"
fi

cd "$LYRA"
bash run_sample.sh
