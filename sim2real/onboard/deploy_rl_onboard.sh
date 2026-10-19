#!/usr/bin/env bash
# Copy the onboard reach runner to the Jetson (~/g1_arm_onboard/rl_onboard) and check it THERE.
# Touches nothing else on the robot: not ~/g1_arm_onboard/onboard (v20), not ~/g1_arm_onboard/dual,
# no running process.
#
#   ./deploy_rl_onboard.sh [policy.npz ...]   copy the runner (+ these policies) and check
#   ./deploy_rl_onboard.sh --check            only the check on the robot
#
# Policies: sim2real/export_policy_npz.py --policy <policy.pt> --layout <layout> writes the .npz.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sim2real="$(dirname "$here")"
ROBOT="unitree@192.168.123.164"; PASS="${ROBOT_PASS:-123}"
DEST="g1_arm_onboard/rl_onboard"
URDF="$HOME/g1_fruits_picker/robot_model_urdf/g1_29dof_with_hand.urdf"
OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 -o LogLevel=ERROR)
SSH=(sshpass -p "$PASS" ssh "${OPTS[@]}" "$ROBOT")
SCP=(sshpass -p "$PASS" scp -q "${OPTS[@]}")

if [[ "${1:-}" != "--check" ]]; then
  for p in "$@"; do [[ -f "$p" && "$p" == *.npz ]] || { echo "not a policy .npz: $p" >&2; exit 2; }; done
  "${SSH[@]}" "mkdir -p ~/$DEST/policies"
  "${SCP[@]}" "$here/rl_runner.py" "$here/runner_config.py" "$here/np_policy.py" "$here/np_fk.py" \
              "$sim2real/reach_io.py" "$sim2real/arm_pose_io.py" "$URDF" "$ROBOT:~/$DEST/"
  for p in "$@"; do "${SCP[@]}" "$p" "$ROBOT:~/$DEST/policies/"; done
  echo "[deploy] copied to $ROBOT:~/$DEST (+ $# policies)"
fi

"${SSH[@]}" "cd ~/$DEST && source ../onboard/env.sh >/dev/null 2>&1; python3 -" <<'PY'
import glob, sys, time
ok = True
print("python", sys.version.split()[0])
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("MISSING numpy:", e); ok = False
try:
    import unitree_sdk2py, cyclonedds  # noqa: F401
    print("unitree_sdk2py + cyclonedds: ok")
except Exception as e:
    print("MISSING unitree_sdk2py / cyclonedds:", e); ok = False
sys.path.insert(0, ".")
import reach_io as R
from np_fk import NumpyKinematics
from np_policy import NumpyPolicy
U = "g1_29dof_with_hand.urdf"
k = NumpyKinematics(U, R)
t = time.perf_counter()
for _ in range(200):
    k.frames([0.0, 0.0, 0.0], R.ARM_DEFAULT)
print("numpy kinematics: %.2f ms per step" % ((time.perf_counter() - t) / 200 * 1e3))
try:
    pk = R.G1Kinematics(U)
    t = time.perf_counter()
    for _ in range(200):
        a = pk.frames([0.1, 0.0, 0.0], R.ARM_DEFAULT)
    b = k.frames([0.1, 0.0, 0.0], R.ARM_DEFAULT)
    print("pinocchio kinematics: %.2f ms per step, palm differs from numpy by %.1e m (the runner uses pinocchio)"
          % ((time.perf_counter() - t) / 200 * 1e3, float(np.max(np.abs(a.palm_ref - b.palm_ref)))))
except Exception as e:
    print("pinocchio not usable (%s): the runner uses the numpy kinematics" % e)
pols = sorted(glob.glob("policies/*.npz"))
for f in pols:
    p = NumpyPolicy(f)
    x = p.mean.copy()
    t = time.perf_counter()
    for _ in range(500):
        p(x)
    print("policy %s: %d in / %d out, layout %s, %.3f ms per step" % (f, p.n_in, p.n_out, p.layout,
                                                                      (time.perf_counter() - t) / 500 * 1e3))
if not pols:
    print("no policies in policies/ yet")
print("CHECK", "OK" if ok else "FAILED")
PY
