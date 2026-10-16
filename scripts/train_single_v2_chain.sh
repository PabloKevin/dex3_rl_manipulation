#!/usr/bin/env bash
# Reach with the apple ENCLOSED in the hand, then the grasp that only closes,
# one after the other (envs/apple_reach_single_v2_env_cfg.py,
# envs/apple_grasp_single_v3_env_cfg.py).
#
#   conda activate env_isaaclab
#   nohup bash scripts/train_single_v2_chain.sh > logs/train_single_v2_chain.log 2>&1 &
#   tail -f logs/train_single_v2_chain.log          # the chain's own steps
#   tail -f logs/train_reach_single_v2.log          # step 1's training output
#   tail -f logs/train_grasp_single_v3.log          # step 4's training output
#
# 1. AppleReachSingle-v2           -> logs/rsl_rl/apple_reach_single_v2/<run>
#    warm start (same 40 inputs / 8 outputs): REACH_WARM, default the
#    AppleReachSingle-v1 checkpoint in git (2026-10-14_19-30-03/model_1899),
#    copied with fresh optimizer state and the exploration std reset to 0.5
# 2. checks the reach ran all its iterations (otherwise stops: no grasp on
#    top of a broken reach)
# 3. exports the newest reach checkpoint -> envs/agents/best_weights/
#    grasp_handoff_reach_single_v2.pt (+ .txt naming the checkpoint): the
#    frozen reach phase of step 4
# 4. AppleGraspSingle-v3           -> logs/rsl_rl/apple_grasp_single_v3/<run>
#    warm start: GRASP_WARM (a 65-input AppleGraspSingle-v2 checkpoint), else
#    the pinned spindle v3 grasp (44 inputs) with the 21 new ones appended
#    at zero weight, as scripts/train_grasp_single_v2.sh does
#
# Options (environment variables):
#   NUM_ENVS_REACH=512 ITERS_REACH=2000 NUM_ENVS_GRASP=1024 ITERS_GRASP=1500 SEED=1
#   REACH_WARM=<model_*.pt>   GRASP_WARM=<model_*.pt>
#   SKIP_REACH=1              steps 3-4 only, from the newest reach v2 run
#   TREE_SAMPLER_WORKERS=<n>  tree-sampling processes (default: half the cores, max 8)
# Each training runs under a guard: Isaac Sim sometimes hangs at shutdown,
# so it is killed if still alive 120 s after printing "Training time".
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_ENVS_REACH="${NUM_ENVS_REACH:-512}"
ITERS_REACH="${ITERS_REACH:-2000}"
NUM_ENVS_GRASP="${NUM_ENVS_GRASP:-1024}"
ITERS_GRASP="${ITERS_GRASP:-1500}"
SEED="${SEED:-1}"
REACH_WARM="${REACH_WARM:-logs/rsl_rl/apple_reach_single_v1/2026-10-14_19-30-03/model_1899.pt}"
GRASP_WARM="${GRASP_WARM:-}"
REACH_ROOT=logs/rsl_rl/apple_reach_single_v2
GRASP_ROOT=logs/rsl_rl/apple_grasp_single_v3
REACH_OUT=envs/agents/best_weights/grasp_handoff_reach_single_v2.pt
STATUS=logs/train_single_v2_chain.status
mkdir -p logs "$REACH_ROOT" "$GRASP_ROOT"

say() { echo "[chain] $(date '+%F %T') $*" | tee -a "$STATUS"; }

newest() {  # newest run's highest model_<n>.pt under $1 (warmstart_* skipped)
    local run; run=$(ls -d "$1"/20* 2>/dev/null | sort | tail -n1)
    [ -n "$run" ] || { echo "no run under $1" >&2; return 1; }
    ls "$run"/model_*.pt | awk -F'model_' '{print $NF+0, $0}' | sort -n | tail -n1 | cut -d' ' -f2
}
iter_of() { basename "$1" .pt | sed 's/model_//'; }

run_train() {  # run_train <log> <train.py args...>: train, kill Isaac if it hangs at shutdown
    local log=$1; shift
    : > "$log"
    setsid python scripts/train.py "$@" >> "$log" 2>&1 &
    local pid=$! t_done=""
    while kill -0 "$pid" 2>/dev/null; do
        if [ -z "$t_done" ] && grep -q "^Training time" "$log"; then t_done=$(date +%s); fi
        if [ -n "$t_done" ] && [ $(( $(date +%s) - t_done )) -gt 120 ]; then
            say "Isaac Sim still alive 120 s after 'Training time': killing it (checkpoints are saved)"
            kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
            break
        fi
        sleep 15
    done
    wait "$pid" 2>/dev/null || true
    grep -q "^Training time" "$log"   # the status of run_train: finished or not
}

check_complete() {  # check_complete <root> <iterations>: the newest run reached its last iteration
    local ck start
    ck=$(newest "$1")
    start=$(ls "$(dirname "$ck")"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | head -n1)
    if [ "$(iter_of "$ck")" -lt $((start + $2 - 2)) ]; then
        say "$1 stopped early (last checkpoint $ck)"
        return 1
    fi
    echo "$ck"
}

# ── 1-2. reach ─────────────────────────────────────────────
if [ "${SKIP_REACH:-0}" != "1" ]; then
    WARM=warmstart_$(basename "$(dirname "$REACH_WARM")")_$(iter_of "$REACH_WARM")
    if [ ! -f "$REACH_ROOT/$WARM/model_0.pt" ]; then
        python scripts/warmstart_reach_tree.py --src "$REACH_WARM" --experiment apple_reach_single_v2 \
            --name "$WARM" --new_inputs 0 --set_std "0:0.5,1:0.5,2:0.5,3:0.5,4:0.5,5:0.5,6:0.5,7:0.5"
    fi
    say "reach: AppleReachSingle-v2, $NUM_ENVS_REACH envs, $ITERS_REACH iterations, warm start $REACH_WARM"
    if ! run_train logs/train_reach_single_v2.log --task AppleReachSingle-v2 --headless \
            --num_envs "$NUM_ENVS_REACH" --max_iterations "$ITERS_REACH" --seed "$SEED" \
            --resume --load_run "$WARM" --checkpoint model_0.pt; then
        say "reach training did not finish (see logs/train_reach_single_v2.log); not starting the grasp"
        exit 1
    fi
    sleep 30
    check_complete "$REACH_ROOT" "$ITERS_REACH" > /dev/null || exit 1
fi

# ── 3. export the reach the grasp runs frozen ───────────────
CK=$(newest "$REACH_ROOT")
python scripts/export_policy_jit.py "$CK" --out "$REACH_OUT"
echo "$CK" > "${REACH_OUT%.pt}.txt"
say "reach phase for the grasp: $CK -> $REACH_OUT"

# ── 4. grasp ───────────────────────────────────────────────
if [ -n "$GRASP_WARM" ]; then
    GWARM=warmstart_$(basename "$(dirname "$GRASP_WARM")")_$(iter_of "$GRASP_WARM")
    [ -f "$GRASP_ROOT/$GWARM/model_0.pt" ] || python scripts/warmstart_reach_tree.py --src "$GRASP_WARM" \
        --experiment apple_grasp_single_v3 --name "$GWARM" --new_inputs 0
else
    GWARM=warmstart_spindle_v3
    [ -f "$GRASP_ROOT/$GWARM/model_0.pt" ] || python scripts/warmstart_reach_tree.py \
        --src envs/agents/best_weights/grasp_spindle_v3_1499.pt --experiment apple_grasp_single_v3 \
        --name "$GWARM" --new_inputs 21 --new_mean 0 --new_var 0.5
fi
say "grasp: AppleGraspSingle-v3, $NUM_ENVS_GRASP envs, $ITERS_GRASP iterations, warm start $GWARM"
if ! run_train logs/train_grasp_single_v3.log --task AppleGraspSingle-v3 --headless \
        --num_envs "$NUM_ENVS_GRASP" --max_iterations "$ITERS_GRASP" --seed "$SEED" \
        --resume --load_run "$GWARM" --checkpoint model_0.pt; then
    say "grasp training did not finish (see logs/train_grasp_single_v3.log)"
    exit 1
fi
say "done: reach $(cat "${REACH_OUT%.pt}.txt"), grasp $(newest "$GRASP_ROOT")"
