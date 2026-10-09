#!/usr/bin/env bash
# Train the G1 squat under a watchdog (same behaviour as the harvest project's watchdog):
#   - training in its own process group, stopped as a whole
#   - after "Training time" is printed: 90 s to exit by itself (Isaac Sim tends to hang at
#     shutdown), then SIGTERM, then SIGKILL — GPU/CPU are not left busy
#   - hard time limit (HARD_LIMIT_S, default 4 h) and a hang guard (log silent 20 min)
#   - every checkpoint (save_interval 100) stays on disk in g1_squat/logs/rsl_rl/g1_squat/
# Settings via env vars: ITERS (4000) NUM_ENVS (4096) HARD_LIMIT_S (14400) LOAD_RUN / CKPT (resume)
#
#   nohup g1_squat/train_watchdog.sh > /dev/null 2>&1 &
#   tail -f g1_squat/logs/train_squat.status
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
PY=/home/pablo_kevin/miniconda3/envs/env_isaaclab/bin/python
ITERS=${ITERS:-4000}
NUM_ENVS=${NUM_ENVS:-4096}
LOAD_RUN=${LOAD_RUN:-}
CKPT=${CKPT:-model_0.pt}
HARD_LIMIT_S=${HARD_LIMIT_S:-14400}
STALL_S=1200
GRACE_S=90
mkdir -p "$DIR/logs"
LOG=$DIR/logs/train_squat.log
STATUS=$DIR/logs/train_squat.status

say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

: > "$STATUS"
RESUME=()
[ -n "$LOAD_RUN" ] && RESUME=(--resume --load_run "$LOAD_RUN" --checkpoint "$CKPT")
say "starting G1-Squat-v0 for $ITERS iterations, $NUM_ENVS envs ${RESUME[*]}"
setsid "$PY" -u "$DIR/scripts/train.py" --task G1-Squat-v0 --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" "${RESUME[@]}" > "$LOG" 2>&1 &
PID=$!
PGID=$PID
START=$(date +%s)
DONE_AT=0
REASON="exited by itself"

stop_group() {
    say "stopping process group $PGID ($1)"
    kill -TERM -- -"$PGID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 -- -"$PGID" 2>/dev/null || return; sleep 1; done
    say "still alive after SIGTERM -> SIGKILL"
    kill -KILL -- -"$PGID" 2>/dev/null
    sleep 3
}

while kill -0 "$PID" 2>/dev/null; do
    sleep 30
    NOW=$(date +%s)
    if [ "$DONE_AT" -eq 0 ] && grep -q "Training time" "$LOG" 2>/dev/null; then
        DONE_AT=$NOW; say "training finished (Training time printed)"
    fi
    if [ "$DONE_AT" -gt 0 ] && [ $((NOW - DONE_AT)) -ge "$GRACE_S" ]; then
        REASON="hung at shutdown after finishing"; stop_group "$REASON"; break
    fi
    if [ $((NOW - START)) -ge "$HARD_LIMIT_S" ]; then
        REASON="hard time limit"; stop_group "$REASON"; break
    fi
    if [ $((NOW - $(stat -c %Y "$LOG"))) -ge "$STALL_S" ]; then
        REASON="log silent for ${STALL_S}s"; stop_group "$REASON"; break
    fi
    IT=$(grep -o "Learning iteration [0-9]*/[0-9]*" "$LOG" | tail -1)
    REW=$(grep "Mean reward:" "$LOG" | tail -1 | awk '{print $NF}')
    [ $((NOW % 600)) -lt 30 ] && say "${IT:-starting} mean reward ${REW:-?}"
done
grep -q "Traceback" "$LOG" && REASON="crashed (Traceback in log)"
RUN=$(ls -td "$DIR"/logs/rsl_rl/g1_squat/*/ 2>/dev/null | head -1)
say "done: $REASON; last $(grep -o 'Learning iteration [0-9]*/[0-9]*' "$LOG" | tail -1); run $RUN"
say "checkpoints: $(ls "$RUN" 2>/dev/null | grep -c model_) files"
