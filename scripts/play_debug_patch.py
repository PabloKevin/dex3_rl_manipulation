# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PATCH FOR  scripts/play.py
#
#  Two things you need before the new rewards can be trusted:
#    A) which local axis of right_hand_palm_link is the palm normal
#       -> sets PALM_NORMAL_AXIS in apple_grasp_env_cfg.py
#    B) the actual joint angles of the collapsed pose
#       -> fills in posture_box lower/upper
#  Plus a reset check, so you stop guessing whether the arm really returns
#  to base every episode.
#
#  Paste SECTION 1 right after   obs = env.get_observations()
#  Paste SECTION 2 inside the while loop, right after   obs, _, dones, _ = env.step(actions)
# ═══════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────
#  SECTION 1 — paste after:   obs = env.get_observations()
# ───────────────────────────────────────────────────────────────────────────

import torch as _t
from isaaclab.utils.math import matrix_from_quat as _mfq

_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

_robot = env.unwrapped.scene["robot"]
_apple = env.unwrapped.scene["apple"]
_jids = [_robot.data.joint_names.index(n) for n in _ARM_NAMES]
_palm_bid = _robot.data.body_names.index("right_hand_palm_link")


def _fmt(v):
    return [round(float(x), 3) for x in v]


# ── A) PALM AXIS PROBE ────────────────────────────────────────────────────
# Prints the three local axes of right_hand_palm_link expressed in world
# coordinates, for the CURRENT pose. The rows of matrix_from_quat's transpose
# are the local X/Y/Z axes in world frame.
#
# How to read it: put the arm in a pose where you KNOW the palm is facing up
# (use tune_arm.py / harvest_apple.py's REACH+CLOSE pose, or just look at the
# viewer). Whichever local axis has world-Z component closest to +1 is your
# palm normal. Then set, in apple_grasp_env_cfg.py:
#     PALM_NORMAL_AXIS = (1,0,0) or (0,1,0) or (0,0,1)   -- or the negative
#
# If none is close to +1 in a palm-up pose, take the one closest to -1 and
# negate it, e.g. PALM_NORMAL_AXIS = (0.0, -1.0, 0.0).
def _probe_palm_axes():
    q = _robot.data.body_quat_w[0:1, _palm_bid, :]
    R = _mfq(q)[0]                      # [3,3], columns are local axes in world
    print("\n─── PALM AXIS PROBE (local axis -> world vector) ───")
    for name, i in (("local X", 0), ("local Y", 1), ("local Z", 2)):
        ax = R[:, i]
        print(f"  {name} -> world {_fmt(ax)}    (world-Z component = {float(ax[2]): .3f})")
    print("  Pick the axis whose world-Z component is nearest +1 in a")
    print("  palm-up pose; negate it if it is nearest -1.")
    print("────────────────────────────────────────────────────\n")


_probe_palm_axes()

print("default arm q :", _fmt(_robot.data.default_joint_pos[0, _jids]))
print("apple world   :", _fmt(_apple.data.root_pos_w[0]))
print("palm  world   :", _fmt(_robot.data.body_pos_w[0, _palm_bid]))
print("SANITY: apple z should match TREE_HEIGHT - BRANCH_RADIUS - APPLE_RADIUS")
print("        from params/env.yaml of the run you are loading. If it does")
print("        NOT, the scene seed fix (BLOCK 2) has not taken effect and you")
print("        are evaluating on a tree the policy never saw.\n")

_step_i = 0


# ───────────────────────────────────────────────────────────────────────────
#  SECTION 2 — paste inside the loop, after:  obs, _, dones, _ = env.step(actions)
# ───────────────────────────────────────────────────────────────────────────

        _step_i += 1

        # ── B) BAD-POSE READOUT ──────────────────────────────────────────
        # Every 2 s, print the arm configuration and how far the palm is from
        # the grasp point. Run this on your current (bad) checkpoint and copy
        # the printed q into the posture_box lower/upper bounds: whichever
        # joint is sitting at an extreme in the collapsed pose is the one to
        # box in. Also tells you definitively which sign of shoulder_roll is
        # "inside" -- no convention guessing needed.
        if _step_i % 100 == 0:
            _q = _robot.data.joint_pos[0, _jids]
            _dq = _robot.data.joint_vel[0, _jids]
            _palm = _robot.data.body_pos_w[0, _palm_bid]
            _tgt = _apple.data.root_pos_w[0].clone()
            _tgt[2] -= 0.08
            _d = float(_t.norm(_tgt - _palm))

            _Rp = _mfq(_robot.data.body_quat_w[0:1, _palm_bid, :])[0]
            # change the column index below to match your PALM_NORMAL_AXIS
            _up = float(_Rp[:, 2][2])

            print(f"[{_step_i:5d}] d={_d*100:6.1f} cm  palm_up_cos={_up:+.2f}  "
                  f"|dq|={float(_t.norm(_dq)):5.2f}")
            print(f"         q  = {_fmt(_q)}")
            print(f"         dq = {_fmt(_dq)}")

        # ── C) RESET VERIFICATION ────────────────────────────────────────
        # Expect q ~ [0, 0, 0, 0.5, 0, 0, 0] +- the 0.20 rad jitter, and
        # |dq| < 0.05. Anything else means the problem is upstream in
        # default_joint_pos (e.g. a default that violates a soft limit and
        # gets silently clamped), not in the reset event.
        if dones.any():
            _q = _robot.data.joint_pos[0, _jids]
            _dq = _robot.data.joint_vel[0, _jids]
            print(f"  RESET -> q={_fmt(_q)}  |dq|={float(_t.norm(_dq)):.4f}")
