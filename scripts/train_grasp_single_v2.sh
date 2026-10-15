#!/usr/bin/env bash
# Grasp that finishes the approach by feel (AppleGraspSingle-v2,
# envs/apple_grasp_single_v2_env_cfg.py).
#
#   conda activate env_isaaclab
#   bash scripts/train_grasp_single_v2.sh
#   NUM_ENVS=1024 ITERS=1500 bash scripts/train_grasp_single_v2.sh
#
# 1. exports the reach the grasp runs frozen in its reach phase, as
#    train_grasp_single.sh does (REACH_CKPT, default: the newest single reach
#    checkpoint) -> envs/agents/best_weights/grasp_handoff_reach_single.pt;
#    SKIP_REACH_EXPORT=1 keeps the one already there;
# 2. warm start (default): the spindle v3 grasp, it 1499 (pinned
#    envs/agents/best_weights/grasp_spindle_v3_1499.pt), its 44 inputs kept
#    and the 21 new ones appended with ZERO weights
#    (scripts/warmstart_reach_tree.py) -- it starts out acting like v3.
#    FROM_SCRATCH=1: no warm start. RESUME_RUN=<run>: continue that run.
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-1500}"
SEED="${SEED:-1}"
REACH_ROOT=logs/rsl_rl/apple_reach_single_v1
GRASP_ROOT=logs/rsl_rl/apple_grasp_single_v2
REACH_OUT=envs/agents/best_weights/grasp_handoff_reach_single.pt
WARM=warmstart_spindle_v3

newest() {  # newest run's highest model_<n>.pt under $1 (warmstart_* skipped)
    local run; run=$(ls -d "$1"/20* 2>/dev/null | sort | tail -n1)
    [ -n "$run" ] || { echo "no run under $1" >&2; return 1; }
    ls "$run"/model_*.pt | awk -F'model_' '{print $NF+0, $0}' | sort -n | tail -n1 | cut -d' ' -f2
}

if [ "${SKIP_REACH_EXPORT:-0}" != "1" ]; then
    CK="${REACH_CKPT:-$(newest "$REACH_ROOT")}"
    python scripts/export_policy_jit.py "$CK" --out "$REACH_OUT"
    echo "$CK" > "${REACH_OUT%.pt}.txt"
fi
echo "[grasp_single_v2] reach phase: $(cat "${REACH_OUT%.pt}.txt" 2>/dev/null || echo "$REACH_OUT")"

RESUME=()
if [ -n "${RESUME_RUN:-}" ]; then
    RESUME=(--resume --load_run "$RESUME_RUN")
elif [ "${FROM_SCRATCH:-0}" != "1" ]; then
    if [ ! -f "$GRASP_ROOT/$WARM/model_0.pt" ]; then
        python scripts/warmstart_reach_tree.py --src envs/agents/best_weights/grasp_spindle_v3_1499.pt \
            --experiment apple_grasp_single_v2 --name "$WARM" --new_inputs 21 --new_mean 0 --new_var 0.5
    fi
    RESUME=(--resume --load_run "$WARM" --checkpoint model_0.pt)
fi

echo "[grasp_single_v2] $(date '+%F %T') AppleGraspSingle-v2, $NUM_ENVS envs, $ITERS iterations ${RESUME[*]:-from scratch}"
python scripts/train.py --task AppleGraspSingle-v2 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" --seed "$SEED" "${RESUME[@]}"
