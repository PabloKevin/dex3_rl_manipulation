#!/usr/bin/env bash
# Two-stage reach, part 1 v3 then part 2 chained v2 (trained behind part 1).
#
#   conda activate env_isaaclab
#   NUM_ENVS=256 ITERS_P1=3000 ITERS_P2=600 bash scripts/train_two_stage_chain_v3.sh
#
# 1. AppleReachPregrasp-v3         -> logs/rsl_rl/apple_reach_pregrasp_v3/<run>
#    FROM SCRATCH: no tactile input (39 inputs), so no 40-input weights fit
# 2. waits for the last checkpoint, checks it is complete
# 3. AppleReachFinalChained-v2     -> logs/rsl_rl/apple_reach_final_chained_v2/<run>
#    warm start: newest AppleReachFinalChained-v1 checkpoint; the frozen part 1
#    is the NEWEST checkpoint of step 1 (picked by the env)
# Starting checkpoints are COPIED into warmstart_* folders; originals stay.
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_ENVS="${NUM_ENVS:-128}"
ITERS_P1="${ITERS_P1:-3000}"
ITERS_P2="${ITERS_P2:-600}"
SEED="${SEED:-1}"

newest() {  # newest run's highest model_<n>.pt under $1 (warmstart_* skipped)
    local run; run=$(ls -d "$1"/20* 2>/dev/null | sort | tail -n1)
    [ -n "$run" ] || { echo "no run under $1" >&2; return 1; }
    ls "$run"/model_*.pt | awk -F'model_' '{print $NF+0, $0}' | sort -n | tail -n1 | cut -d' ' -f2
}
iter_of() { basename "$1" .pt | sed 's/model_//'; }

PRE=logs/rsl_rl/apple_reach_pregrasp_v3
FIN=logs/rsl_rl/apple_reach_final_chained_v2
mkdir -p "$PRE" "$FIN/warmstart_chained_v1"
cp -n "$(newest logs/rsl_rl/apple_reach_final_chained)" "$FIN/warmstart_chained_v1/model_0.pt"

echo "[chain] $(date '+%F %T') part 1: AppleReachPregrasp-v3, $NUM_ENVS envs, $ITERS_P1 iterations, from scratch"
python scripts/train.py --task AppleReachPregrasp-v3 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS_P1" --seed "$SEED"

sleep 60   # let the runner finish writing the last checkpoint
CK=$(newest "$PRE"); START=$(ls "$(dirname "$CK")"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | head -n1)
echo "[chain] part 1 last checkpoint: $CK"
if [ "$(iter_of "$CK")" -lt $((START + ITERS_P1 - 2)) ]; then
    echo "[chain] part 1 stopped early (iteration $(iter_of "$CK")); not starting part 2" >&2
    exit 1
fi

echo "[chain] $(date '+%F %T') part 2: AppleReachFinalChained-v2, $NUM_ENVS envs, $ITERS_P2 iterations"
python scripts/train.py --task AppleReachFinalChained-v2 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS_P2" --seed "$SEED" --resume --load_run warmstart_chained_v1
echo "[chain] $(date '+%F %T') done"
