#!/bin/bash
#SBATCH --job-name=lyra2-recall
#SBATCH --output=logs/lyra2-recall-%j.out
#SBATCH --error=logs/lyra2-recall-%j.err
#SBATCH --partition=gpuhe.4h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=rtx_4090:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=eu-g6-027
#
# Reverse-recall probe (lyra_2/tasks/retrieval.py): prefill Lyra2's spatial + past-latent buffers
# with a reversal episode's GT forward leg, then AR-generate the turn+backward legs.
#
#   sbatch scripts/euler/run_retrieval_probe.sh --checkpoint outputs/.../checkpoints/iter_000009800
#   sbatch scripts/euler/run_retrieval_probe.sh --variant 1601 --num-scenes 2 --checkpoint ...
#
# Any extra args are forwarded to the module verbatim. eu-g6-027 is excluded: its GPU 1 is faulty.
# Runtime is ~35 chunks x --num-steps denoise passes per episode at the 401 variant.

set -uo pipefail
cd /cluster/scratch/ecetin/MemoryKrea/src/external/lyra || exit 1
mkdir -p logs

# `act memorykrea` (bashrc function) loads the module stack (cuda/12.8.0, cudnn, ...) and the venv;
# transformer_engine needs those cuda libs at import time. PS1 is set first to defeat bashrc's
# non-interactive early-return guard, and nounset is relaxed while sourcing it.
set +u
export PS1="${PS1:-probe$ }"
source ~/.bashrc
act memorykrea
set -u

# PYTHONPATH / HF cache; re-prepend the venv afterwards so a bare `python` cannot resolve to the
# project .venv (which lacks webdataset).
set -a
source /cluster/scratch/ecetin/MemoryKrea/.env
set +a
export PATH="$HOME/memorykrea/bin:$PATH"

CKPT_DEFAULT=outputs/lyra2_from_bidir/memorymaze/finetuned/checkpoints/iter_000009800

# Only supply the default checkpoint when the caller did not pass one.
case " $* " in
  *" --checkpoint "*) CKPT_ARG=() ;;
  *) CKPT_ARG=(--checkpoint "$CKPT_DEFAULT") ;;
esac

echo "[run_retrieval_probe] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
exec "$HOME/memorykrea/bin/python" -m lyra_2.tasks.retrieval "${CKPT_ARG[@]}" "$@"
