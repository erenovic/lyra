#!/bin/bash
# Train Lyra 2.0 on Memory Maze (from-scratch small scale) on a Euler GPU node.
#
#   EXPERIMENT=lyra2_maze_smoke bash scripts/euler/lyra/train_maze.sh          # 65M pipeline smoke
#   EXPERIMENT=lyra2_maze_small GPUS=1 bash scripts/euler/lyra/train_maze.sh   # 240M main run
#   ... extra hydra overrides pass through: bash ... trainer.max_iter=100
#
# Env preamble mirrors scripts/euler/lyra/demo.sh (act env, module order, NVMe staging).
set -uo pipefail
cd /cluster/scratch/ecetin/MemoryKrea || exit 1

# act memorykrea (bashrc fn): module stack (cuda/12.8.0, cudnn, eth_proxy, ...) + venv.
# PS1 defeats bashrc's non-interactive guard; nounset relaxed while sourcing.
set +u
export PS1="${PS1:-train$ }"
source ~/.bashrc
act memorykrea
module load gcc/12.2.0
module load eigen
module load cuda/12.8.0
set -u

set -a; source .env; [ -f .env.secret ] && source .env.secret; set +a
export PATH="$HOME/memorykrea/bin:$PATH"

# Node-local caches; VIPE/DA3-style pickled-config checkpoints need old torch.load behavior.
CACHE=/tmp/${USER}-lyracache; mkdir -p "$CACHE/triton"
export TMPDIR="$CACHE" TRITON_CACHE_DIR="$CACHE/triton"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LYRA=src/external/lyra

# Checkpoints (frozen VAE/CLIP/T5 assets) from the HF hub cache; stage to node-local NVMe
# when a job scratch exists so the frozen encoders load from local disk.
SNAP_CK=$(ls -d "$PWD"/.cache/huggingface/hub/models--nvidia--Lyra-2.0/snapshots/*/checkpoints 2>/dev/null | head -1)
if [ -z "$SNAP_CK" ]; then
  SNAP=$(realpath "$(hf download nvidia/Lyra-2.0 --include "checkpoints/*")")
  SNAP_CK="$SNAP/checkpoints"
fi
LOCAL=$(ls -d /scratch/tmp.*."$USER" 2>/dev/null | head -1)
if [ -n "$LOCAL" ]; then
  echo "staging checkpoints -> $LOCAL/lyra2_checkpoints (no-op if already there)"
  rsync -aL "$SNAP_CK/" "$LOCAL/lyra2_checkpoints/"
  ln -sfn "$LOCAL/lyra2_checkpoints" "$LYRA/checkpoints"
else
  ln -sfn "$SNAP_CK" "$LYRA/checkpoints"
fi

export IMAGINAIRE_OUTPUT_ROOT=/cluster/scratch/ecetin/MemoryKrea/outputs/lyra2

# CWD must be the lyra root: VAE/CLIP paths are ./checkpoints/... relative.
cd "$LYRA"
torchrun --standalone --nproc_per_node="${GPUS:-1}" -m lyra_2.train -- -- "experiment=${EXPERIMENT:-lyra2_maze_small}" "$@"
