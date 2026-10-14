#!/usr/bin/env bash
# Grasp after the single no-touch reach (AppleGraspSingle-v1).
#
#   conda activate env_isaaclab
#   bash scripts/train_grasp_single.sh
#   NUM_ENVS=1024 ITERS=1500 bash scripts/train_grasp_single.sh
#
# 1. exports the reach the grasp runs frozen in its reach phase:
#    REACH_CKPT (default: the newest model_<n>.pt of the newest run under
#    logs/rsl_rl/apple_reach_single_v1) -> envs/agents/best_weights/grasp_handoff_reach_single.pt
#    (frozen at this moment: a reach still training does not change it; run this
#    script again, with RESUME_RUN, to move the grasp onto a newer reach)
# 2. warm start (default): the spindle v3 grasp, it 1499 (pinned:
#    envs/agents/best_weights/grasp_spindle_v3_1499.pt) -- same 44 inputs / 14
#    outputs; it already brings the thumb in and holds under twist / pull.
#    FROM_SCRATCH=1: no warm start. RESUME_RUN=<run>: continue that grasp run.
# The iteration counter continues from the loaded checkpoint (1499 -> ...);
# ITERS counts the new iterations.
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-1500}"
SEED="${SEED:-1}"
REACH_ROOT=logs/rsl_rl/apple_reach_single_v1
GRASP_ROOT=logs/rsl_rl/apple_grasp_single_v1
REACH_OUT=envs/agents/best_weights/grasp_handoff_reach_single.pt

newest() {  # newest run's highest model_<n>.pt under $1 (warmstart_* skipped)
    local run; run=$(ls -d "$1"/20* 2>/dev/null | sort | tail -n1)
    [ -n "$run" ] || { echo "no run under $1" >&2; return 1; }
    ls "$run"/model_*.pt | awk -F'model_' '{print $NF+0, $0}' | sort -n | tail -n1 | cut -d' ' -f2
}

CK="${REACH_CKPT:-$(newest "$REACH_ROOT")}"
python scripts/export_policy_jit.py "$CK" --out "$REACH_OUT"
echo "$CK" > "${REACH_OUT%.pt}.txt"
echo "[grasp_single] reach phase: $CK"

RESUME=()
if [ -n "${RESUME_RUN:-}" ]; then
    RESUME=(--resume --load_run "$RESUME_RUN")
elif [ "${FROM_SCRATCH:-0}" != "1" ]; then
    mkdir -p "$GRASP_ROOT/warmstart_spindle_v3"
    cp -n envs/agents/best_weights/grasp_spindle_v3_1499.pt "$GRASP_ROOT/warmstart_spindle_v3/model_0.pt"
    RESUME=(--resume --load_run warmstart_spindle_v3)
fi

echo "[grasp_single] $(date '+%F %T') AppleGraspSingle-v1, $NUM_ENVS envs, $ITERS iterations ${RESUME[*]:-from scratch}"
python scripts/train.py --task AppleGraspSingle-v1 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" --seed "$SEED" "${RESUME[@]}"
