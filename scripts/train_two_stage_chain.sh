#!/usr/bin/env bash
# Two-stage reach, v2: train part 1 (smoother, settles), then part 2 behind it.
#
#   conda activate env_isaaclab
#   NUM_ENVS=128 ITERS=600 bash scripts/train_two_stage_chain.sh
#
# 1. AppleReachPregrasp-v2        -> logs/rsl_rl/apple_reach_pregrasp_v2/<run>
#    warm start: the v1 pregrasp model_599 (same obs / action layout)
# 2. (waits for the last checkpoint to be written)
# 3. AppleReachFinalChained-v1    -> logs/rsl_rl/apple_reach_final_chained/<run>
#    warm start: the v1 final model_300; the frozen part 1 is the NEWEST
#    checkpoint of step 1, picked up by the env itself.
# The starting checkpoints are COPIED into warmstart_* folders; the originals stay.
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_ENVS="${NUM_ENVS:-128}"
ITERS="${ITERS:-600}"
SEED="${SEED:-1}"
PRE=logs/rsl_rl/apple_reach_pregrasp_v2
FIN=logs/rsl_rl/apple_reach_final_chained

mkdir -p "$PRE/warmstart_v1" "$FIN/warmstart_final_v1"
cp -n logs/rsl_rl/apple_reach_pregrasp/2026-10-13_18-22-27/model_599.pt "$PRE/warmstart_v1/model_0.pt"
cp -n logs/rsl_rl/apple_reach_final/2026-10-13_18-51-28/model_300.pt "$FIN/warmstart_final_v1/model_0.pt"

echo "[chain] $(date '+%F %T') part 1: AppleReachPregrasp-v2, $NUM_ENVS envs, $ITERS iterations"
python scripts/train.py --task AppleReachPregrasp-v2 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" --seed "$SEED" --resume --load_run warmstart_v1

# let the runner finish writing the last checkpoint, then verify it
sleep 60
LAST=$(ls -d "$PRE"/20* | sort | tail -n1)
LASTCK=$(ls "$LAST"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -n1)
echo "[chain] part 1 last checkpoint: $LAST/model_${LASTCK}.pt"
if [ "$LASTCK" -lt $((ITERS - 1)) ]; then
    echo "[chain] part 1 did not reach iteration $((ITERS - 1)); not starting part 2" >&2
    exit 1
fi

echo "[chain] $(date '+%F %T') part 2: AppleReachFinalChained-v1, $NUM_ENVS envs, $ITERS iterations"
python scripts/train.py --task AppleReachFinalChained-v1 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" --seed "$SEED" --resume --load_run warmstart_final_v1
echo "[chain] $(date '+%F %T') done"
