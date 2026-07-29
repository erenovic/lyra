#!/usr/bin/env bash
# Finetune the Stage-A bidirectional DiT inside Lyra 2.0.
#
# Picks the experiment + a fresh output dir, then hands off to train_maze_16gpu.sh (which carries
# the #SBATCH header). Run with bash, NOT sbatch.
#
#   bash scripts/euler/train_lyra_from_bidir.sh
#   SMOKE=1 bash scripts/euler/train_lyra_from_bidir.sh   # 50-iter pipeline check, own OUT_DIR
# Env: INIT_CKPT (converted .pth), OUT_DIR, EXPERIMENT.
#
# PREREQUISITE -- convert the Stage-A checkpoint, from the PARENT repo root (single line):
#   set -a; source .env; set +a; python -m src.utils.export_dit_to_lyra dit=outputs/dit_bidir_latentT101_L14D1024/checkpoints/latest.ema.pt out=src/external/lyra/outputs/lyra_from_bidir/converted.pth
#
# WHY a fresh OUT_DIR: checkpoints/latest_checkpoint.txt beats checkpoint.load_path unconditionally
# (imaginaire/checkpointer/dcp.py:704-712), so the transfer is consumed exactly once on the first
# submit and every requeue then auto-resumes this run's own weights. The smoke gets its own dir for
# the same reason -- otherwise its checkpoint would poison the real run.
#
# CHECK THE FIRST LAUNCH -- the .pth loader only logs, it never raises. Expect in the rank-0 log:
#   "Resuming ckpt <abs>.pth"      (a DIRECTORY path means the transfer was ignored)
#   _IncompatibleKeys(...)          unexpected_keys and incorrect_shapes MUST both be empty
#   "freeze re-applied: 89.29M / 280.61M params trainable"
# and NO "Training from scratch."

set -euo pipefail
LYRA=/cluster/scratch/ecetin/MemoryKrea/src/external/lyra
cd "$LYRA"
# Must exist BEFORE sbatch: SLURM resolves #SBATCH --output at submit time and will not create the
# directory, so a missing logs/ silently costs you the rank-0 log (the only place the checkpoint
# load is verifiable). The mkdir inside train_maze_16gpu.sh runs far too late.
mkdir -p "$LYRA/logs"

if [ "${SMOKE:-0}" = "1" ]; then
  export EXPERIMENT=${EXPERIMENT:-maze_small_smoke}
else
  export EXPERIMENT=${EXPERIMENT:-maze_small}
fi
# EMPTY on purpose: the experiment's job block owns the whole path, so runs land at
# outputs/lyra2_from_bidir/memorymaze/{finetuned,finetuned_smoke} instead of gaining an extra
# OUT_DIR level. train_maze_16gpu.sh uses ${OUT_DIR-...}, so an explicit empty value is honoured.
export OUT_DIR=${OUT_DIR-}

INIT_CKPT=${INIT_CKPT:-$LYRA/outputs/lyra2_from_bidir/converted.pth}
if [ ! -f "$INIT_CKPT" ]; then
  echo "==== $INIT_CKPT missing -- run src.utils.export_dit_to_lyra first (see header) ===="
  exit 1
fi

# The only thing that rides on the command line; everything else is in the registered experiment.
export EXTRA_OVERRIDES="checkpoint.load_path=${INIT_CKPT} ${EXTRA_OVERRIDES:-}"

echo "==== $(date '+%F %T') Lyra2 finetune from Stage-A DiT | experiment=${EXPERIMENT}"\
     "| init=${INIT_CKPT} | out_dir=${OUT_DIR} | smoke=${SMOKE:-0} ===="

exec sbatch "$LYRA/scripts/euler/train_maze_16gpu.sh"
