#!/usr/bin/env bash
# Start / stop / status of the onboard apple detector on the Jetson (its own process, low priority: cores 0-5,
# nice 10 -- the RL runner and the daemon keep theirs). It only READS the cameras; it never starts or stops them.
#
#   ./vision_onboard_robot.sh start      detached; log /tmp/vision_onboard.log on the robot
#   ./vision_onboard_robot.sh stop
#   ./vision_onboard_robot.sh status
# Prints one line starting "VISION: " (g1_launch.py reads it).
set -u
ACTION="${1:-status}"
ROBOT="unitree@192.168.123.164"; PASS="${ROBOT_PASS:-123}"
exec sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 \
  -o LogLevel=ERROR "$ROBOT" "ACTION=$ACTION PASS=$PASS bash -s" <<'ROBOT_EOF'
D=~/vision_onboard
PAT="python3 -u vision_onboard.py"
pid=$(pgrep -f "$PAT" | head -1)
case "$ACTION" in
  start)
    [ -f "$D/vision_onboard.py" ] || { echo "VISION: FAILED not deployed (sim2real/vision_onboard/deploy_vision_onboard.sh)"; exit 1; }
    [ -f "$D/models/rfdetr560_mixed.engine" ] || { echo "VISION: FAILED no engine (build_engine.py)"; exit 1; }
    if [ -n "$pid" ]; then echo "VISION: REUSED pid $pid"; exit 0; fi
    cd "$D"
    source /opt/ros/foxy/setup.bash; source ~/cyclonedds_ws/install/setup.bash 2>/dev/null
    export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=$HOME/ros2_cameras/cyclonedds_cameras.xml
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 setsid nohup taskset -c 0-5 nice -n 10 python3 -u vision_onboard.py \
        > /tmp/vision_onboard.log 2>&1 < /dev/null &
    sleep 4
    pid=$(pgrep -f "$PAT" | head -1)
    if [ -z "$pid" ]; then echo "VISION: FAILED (exited at once):"; tail -8 /tmp/vision_onboard.log; exit 1; fi
    echo "VISION: STARTED pid $pid"
    ;;
  stop)
    [ -z "$pid" ] && { echo "VISION: NOT RUNNING"; exit 0; }
    kill -INT $pid
    for i in $(seq 1 20); do pgrep -f "$PAT" >/dev/null || break; sleep 0.25; done
    pgrep -f "$PAT" >/dev/null && { kill -9 $(pgrep -f "$PAT"); echo "VISION: KILLED"; exit 0; }
    echo "VISION: STOPPED"
    ;;
  *)
    if [ -n "$pid" ]; then echo "VISION: RUNNING pid $pid"; tail -3 /tmp/vision_onboard.log; else echo "VISION: NOT RUNNING"; fi
    ;;
esac
ROBOT_EOF
