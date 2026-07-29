#!/usr/bin/env bash
# Lyra 2.0 Memory-Maze training: 8 nodes x 2 RTX-4090 = 16 GPUs.
#
# The whole recipe (net, batch_size, grad_accum_iter, lr, max_iter) lives in the registered
# experiment -- this script passes NO trainer overrides, so `EXPERIMENT=<name> sbatch ...` always
# gets that experiment's own settings. Outputs land in
#   <this repo>/outputs[/$OUT_DIR]/<job.project>/<job.group>/<job.name>
# so an EMPTY OUT_DIR lets the experiment's job block own the whole path.
# Resume is automatic: DCP reads .../checkpoints/latest_checkpoint.txt -- just resubmit.
#
#   sbatch scripts/euler/train_maze_16gpu.sh
#   EXPERIMENT=lyra2_maze_from_bidir sbatch scripts/euler/train_maze_16gpu.sh
# Env: EXPERIMENT, OUT_DIR, EXTRA_OVERRIDES (extra hydra args). Change the GPU layout with
# `sbatch --nodes=N --gpus-per-node=rtx_4090:M ...` -- the count is read back from SLURM.
#
# Requires two symlinks in the repo root (set up once, NOT by this script):
#   .env        -> ../../../.env          (HF_HOME is relative, so it resolves to ./.cache)
#   checkpoints -> .cache/huggingface/hub/models--nvidia--Lyra-2.0/snapshots/*/checkpoints
#
#SBATCH --job-name=lyra2-maze-16gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1                 # one torchrun launcher per node
#SBATCH --gpus-per-node=rtx_4090:2          # Euler: --gpus*, not --gres=gpu:. 8 x 2 = 16 GPUs;
#SBATCH --cpus-per-task=12                  # 2 procs/node x (main + num_workers=4) = 10 (< 12)
#SBATCH --mem-per-cpu=4G                    # 12 x 4G = 48G/node (24G per GPU proc)
#SBATCH --time=4-00:00:00
#SBATCH --time-min=1-00:00:00               # allow backfill; DCP auto-resume covers the rest
#SBATCH --account=es_schin
#SBATCH --output=/cluster/scratch/ecetin/MemoryKrea/src/external/lyra/logs/lyra2-maze-16gpu-%j.out
#SBATCH --error=/cluster/scratch/ecetin/MemoryKrea/src/external/lyra/logs/lyra2-maze-16gpu-%j.err

set -euo pipefail
LYRA=/cluster/scratch/ecetin/MemoryKrea/src/external/lyra
cd "$LYRA"
mkdir -p logs

# GPUs per node from the SLURM allocation (SLURM_GPUS_PER_NODE is "type:N" or "N") -- the
# #SBATCH --gpus-per-node line above is the single knob; nothing below hardcodes the count.
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE##*:}
EXPERIMENT=${EXPERIMENT:-lyra2_maze_small}
OUT_DIR=${OUT_DIR-lyra2_v2}   # no colon: an explicitly EMPTY OUT_DIR is honoured
head_ip=$(hostname --ip-address | awk '{print $1}')   # rendezvous host = the batch node
PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

echo "==== $(date '+%F %T') Lyra2 maze | ${SLURM_NNODES} x ${GPUS_PER_NODE} GPUs | head=${head_ip}:${PORT}"\
     "| experiment=${EXPERIMENT} | out=${LYRA}/outputs${OUT_DIR:+/$OUT_DIR} ===="

# 4090 nodes are Ethernet-connected; if cross-node NCCL hangs at startup, find the interface with
# `ip -o -4 addr` on a compute node and add NCCL_SOCKET_IFNAME=<iface> below.
srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none \
     --mem-per-cpu=4G --gpus-per-node="$SLURM_GPUS_PER_NODE" \
     bash -c '
  cd '"$LYRA"'
  # One at a time: cuda/12.8.0 only resolves after stack + python_cuda, and a multi-arg load aborts
  # the whole set on the first conflict. Without this transformer_engine cannot find cudart.
  module load eth_proxy; module load stack/2024-06; module load python_cuda/3.11.6
  module load cuda/12.8.0; module load cudnn/9.2.0.82-12; module load eigen
  source "$HOME"/memorykrea/bin/activate
  set -a; source .env; set +a

  export TMPDIR=/tmp/'"$USER"'-lyracache TRITON_CACHE_DIR=/tmp/'"$USER"'-lyracache/triton
  mkdir -p "$TRITON_CACHE_DIR"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # required to load a .pth init checkpoint
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
  export IMAGINAIRE_OUTPUT_ROOT='"$LYRA"'/outputs'"${OUT_DIR:+/$OUT_DIR}"'

  python -m torch.distributed.run \
    --nnodes='"$SLURM_NNODES"' --nproc-per-node='"$GPUS_PER_NODE"' \
    --node-rank=$SLURM_NODEID --master-addr='"$head_ip"' --master-port='"$PORT"' \
    -m lyra_2.train -- -- experiment='"$EXPERIMENT"' '"${EXTRA_OVERRIDES:-}"'
'
echo "==== $(date '+%F %T') training exited (code $?) ===="
