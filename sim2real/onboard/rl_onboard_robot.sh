#!/usr/bin/env bash
# Start / stop / status of the onboard reach runner on the Jetson (its own process; the arm daemon
# and the cameras are not touched). Idle, it sends nothing to the daemon.
#
#   ./rl_onboard_robot.sh start      [--net-if eth0]   detached; log /tmp/rl_onboard.log on the robot
#   ./rl_onboard_robot.sh stop                         SIGTERM (a running episode ends with "bye")
#   ./rl_onboard_robot.sh status
# Prints one line starting "ONBOARD: " (g1_launch.py reads it).
set -u
ACTION="${1:-status}"; shift || true
NETIF=eth0
[[ "${1:-}" == "--net-if" ]] && NETIF="${2:-eth0}"
ROBOT="unitree@192.168.123.164"; PASS="${ROBOT_PASS:-123}"
exec sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 \
  -o LogLevel=ERROR "$ROBOT" "ACTION=$ACTION NETIF=$NETIF PASS=$PASS bash -s" <<'ROBOT_EOF'
D=~/g1_arm_onboard/rl_onboard
PAT="python3 -u rl_runner.py"
pid=$(pgrep -f "$PAT" | head -1)
case "$ACTION" in
  start)
    [ -f "$D/rl_runner.py" ] || { echo "ONBOARD: FAILED not deployed (sim2real/onboard/deploy_rl_onboard.sh)"; exit 1; }
    if [ -n "$pid" ]; then echo "ONBOARD: REUSED pid $pid"; exit 0; fi
    cd "$D" && source ../onboard/env.sh >/dev/null 2>&1
    setsid nohup python3 -u rl_runner.py --net-if "$NETIF" > /tmp/rl_onboard.log 2>&1 < /dev/null &
    sleep 1.5
    pid=$(pgrep -f "$PAT" | head -1)
    if [ -z "$pid" ]; then echo "ONBOARD: FAILED (exited at once):"; tail -5 /tmp/rl_onboard.log; exit 1; fi
    # the daemon's parent runs on cores 0-6 at nice -10, its publisher child alone on core 7: same as the parent
    echo "$PASS" | sudo -S taskset -cp 0-6 "$pid" >/dev/null 2>&1
    echo "$PASS" | sudo -S renice -n -10 -p "$pid" >/dev/null 2>&1
    echo "ONBOARD: STARTED pid $pid"
    ;;
  stop)
    [ -z "$pid" ] && { echo "ONBOARD: NOT RUNNING"; exit 0; }
    kill -TERM $pid
    for i in $(seq 1 20); do pgrep -f "$PAT" >/dev/null || break; sleep 0.25; done
    pgrep -f "$PAT" >/dev/null && { kill -9 $(pgrep -f "$PAT"); echo "ONBOARD: KILLED"; exit 0; }
    echo "ONBOARD: STOPPED"
    ;;
  *)
    if [ -n "$pid" ]; then echo "ONBOARD: RUNNING pid $pid"; tail -3 /tmp/rl_onboard.log; else echo "ONBOARD: NOT RUNNING"; fi
    ;;
esac
ROBOT_EOF
