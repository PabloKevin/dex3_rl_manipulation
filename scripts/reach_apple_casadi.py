#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
CasADi IK reacher — classical nonlinear-optimization inverse kinematics,
NO neural network anywhere in the loop. Meant as a drop-in alternative to
BOTH the trained RL policy (scripts/train.py -> envs/apple_grasp_env_cfg.py)
and the cuRobo reachers (reach_apple_ik.py, reach_apple_curobo.py), driving
the running sim over the exact same DDS contract those already use.

Usage:
  1. Terminal 1 (Isaac Sim, unmodified — run WITHOUT --policy so run_env.py
     takes DDS targets instead of a loaded policy; --policy remains
     available whenever you want the RL policy instead, the two are
     mutually exclusive on run_env.py's own side):
       ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
  2. Terminal 2 (this script, a different conda env — no Isaac Lab import
     here at all, just casadi + torch + unitree_sdk2py):
       /home/pablo_kevin/miniconda3/envs/unitree_g1/bin/python scripts/reach_apple_casadi.py

Why this instead of reach_apple_ik.py (cuRobo IKSolver, solve-once +
smoothstep) or reach_apple_curobo.py (cuRobo MotionGen, full trajectory
optimization): the trained policy drives the arm with a fresh forward pass
EVERY 50 Hz control tick, producing a joint-position target that the env's
own RateLimitedJointPositionAction (envs/mdp/actions.py) smooths via its
velocity/acceleration caps. This script mirrors that architecture instead
of cuRobo's plan-then-execute style: it re-solves a small CasADi/IPOPT
nonlinear program every tick (warm-started from the previous solution, so
each solve is a handful of iterations from a near-optimal point, not a cold
start) and publishes the raw target directly — the IK solve stands in for
the neural net's forward pass, the env-side rate limiter still does the
smoothing exactly as it does for the RL policy.

Kinematics: analytic forward kinematics for the 7-DOF right-arm chain
(torso_link -> right_hand_palm_link), built directly from the exact joint
origins/axes in envs/assets/cuRobo/g1_29dof_with_hand_rev_1_0.urdf as
symbolic CasADi expressions -- no URDF parser, no cuRobo, no pinocchio.

Safety: reuses proximity_stop.QuinticStopController UNMODIFIED (same class
run_env.py uses for --policy runs) -- see the "Proximity stop" section
below for exactly how it's driven here.

DDS topics (same contract as run_env.py / reach_apple_ik.py):
  rt/arm_tune/targets  (publish)    {"arm": [7 floats], "hand": [7 floats]}
                                     RAW pre-scale actions -- run_env.py applies
                                     target = default_joint_pos + scale * raw_action
  rt/arm_tune/state    (subscribe)  {"torso_pos", "torso_quat", "arm_joint_pos",
                                     "apple_pos", "ee_pos", "ee_quat", "jacobian",
                                     "default_arm_pos", "arm_joint_lower", "arm_joint_upper"}

Install (only needed once, in whichever conda env you run this script from):
  pip install casadi
"""
import argparse
import json
import math
import sys
import time
import threading

import casadi as ca
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────
# 1. Quaternion / frame helpers — duplicated from reach_apple_ik.py (this
#    repo's convention: each DDS client script carries its own small copy
#    rather than importing a shared module).
# ─────────────────────────────────────────────────────────────────────────

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.stack([w, -x, -y, -z])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (3,) by quaternion q (4,), wxyz convention."""
    w = q[0]
    q_vec = q[1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w * t + torch.cross(q_vec, t, dim=-1)


def world_to_torso(pos_world: torch.Tensor, torso_pos: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    """Express a world-frame point in the torso_link local frame."""
    delta = pos_world - torso_pos
    return quat_rotate(quat_conjugate(torso_quat), delta)


def vec_world_to_torso(vec_world: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame DIRECTION (no translation) into the torso_link frame."""
    return quat_rotate(quat_conjugate(torso_quat), vec_world)


# ─────────────────────────────────────────────────────────────────────────
# 2. DDS setup — duplicated from reach_apple_ik.py (publisher on targets,
#    subscriber on state, same ChannelFactoryInitialize(1) domain).
# ─────────────────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import (
        ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    )
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(1)
    publisher = ChannelPublisher("rt/arm_tune/targets", String_)
    publisher.Init()
    DDS_OK = True
    print("[DDS] Publisher ready on rt/arm_tune/targets")
except Exception as e:
    print(f"[DDS] OFFLINE: {e}")
    DDS_OK = False

_state_lock = threading.Lock()
_latest_state = None  # dict: torso_pos, torso_quat, arm_joint_pos, apple_pos, ...


def _state_callback(msg):
    global _latest_state
    try:
        data = json.loads(msg.data)
        with _state_lock:
            _latest_state = data
    except Exception as e:
        print(f"[DDS] state parse error: {e}")


if DDS_OK:
    try:
        state_sub = ChannelSubscriber("rt/arm_tune/state", String_)
        state_sub.Init(_state_callback, 1)
        print("[DDS] Subscribed to rt/arm_tune/state")
    except Exception as e:
        print(f"[DDS] Could not subscribe to state: {e}")
        DDS_OK = False


def wait_for_state(timeout_s: float = 15.0) -> dict:
    """Block until run_env.py's first state message arrives."""
    print("Waiting for state from run_env.py (make sure it's running WITHOUT --policy,")
    print("past env.reset(), and both scripts use ChannelFactoryInitialize(1) on")
    print("the same network interface)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with _state_lock:
            if _latest_state is not None:
                return dict(_latest_state)
        time.sleep(0.1)
    raise RuntimeError(
        "No state received from run_env.py within timeout. Check: (1) run_env.py is "
        "running (without --policy) and past env.reset(), (2) it's publishing "
        "rt/arm_tune/state, (3) both scripts use ChannelFactoryInitialize(1) on the "
        "same network interface."
    )


def publish(arm_raw, hand_raw) -> None:
    if not DDS_OK:
        return
    try:
        publisher.Write(String_(data=json.dumps({"arm": list(arm_raw), "hand": list(hand_raw)})))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 3. Constants — must track envs/apple_grasp_env_cfg.py / the URDF. Kept as
#    a plain duplicated copy here (same fragility already accepted by
#    reach_apple_ik.py's ARM_DEFAULT_POSE/ARM_SCALE constants) rather than
#    importing the env cfg module, which pulls in Isaac Lab.
# ─────────────────────────────────────────────────────────────────────────

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# (origin_xyz, origin_rpy, axis) for each of the 7 revolute joints, torso_link
# -> right_hand_palm_link, then the final FIXED palm offset. Verified
# directly against envs/assets/cuRobo/g1_29dof_with_hand_rev_1_0.urdf:1002-1175.
_ARM_CHAIN = [
    # xyz,                              rpy,                                   axis
    ((0.0039563, -0.10021, 0.24778),   (-0.27931, 5.4949e-05, 1.9159e-04),   (0, 1, 0)),  # shoulder_pitch
    ((0.0,       -0.038,   -0.013831), (0.27925, 0.0, 0.0),                  (1, 0, 0)),  # shoulder_roll
    ((0.0,       -0.00624, -0.1032),   (0.0, 0.0, 0.0),                      (0, 0, 1)),  # shoulder_yaw
    ((0.015783,   0.0,     -0.080518), (0.0, 0.0, 0.0),                      (0, 1, 0)),  # elbow
    ((0.100,     -0.00188791, -0.010), (0.0, 0.0, 0.0),                      (1, 0, 0)),  # wrist_roll
    ((0.038,      0.0,      0.0),      (0.0, 0.0, 0.0),                      (0, 1, 0)),  # wrist_pitch
    ((0.046,      0.0,      0.0),      (0.0, 0.0, 0.0),                      (0, 0, 1)),  # wrist_yaw
]
_PALM_FIXED_OFFSET = (0.0415, -0.003, 0.0)  # right_hand_palm_joint (fixed), no rotation

# Same convention as envs/apple_grasp_env_cfg.py.
PALM_NORMAL_AXIS = (0.0, 1.0, 0.0)   # palm normal in the palm link's LOCAL frame
PALM_LOCAL_OFFSET = (0.08, 0.0, 0.0)  # grasp reference point in the palm link's LOCAL frame

# The real per-episode apple radius is jittered +-20% server-side and never
# published over DDS (same limitation reach_apple_ik.py's APPLE_RADIUS=0.04
# already has) -- a few-mm error here is well inside pos_tol=0.04.
APPLE_RADIUS_NOMINAL = 0.04
GRASP_OFFSET_Z_DEFAULT = 0.0 

# Must match apple_grasp_env_cfg.py's ActionsCfg.right_arm.scale.
ARM_SCALE = 0.5

HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]

# Fallback joint limits (URDF hard limits x0.95 soft factor, matching
# apple_grasp_env_cfg.py's soft_joint_pos_limit_factor) if run_env.py's
# published arm_joint_lower/upper are ever missing.
_HARD_LOWER = [-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144]
_HARD_UPPER = [2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144]
FALLBACK_LOWER = [v * 0.95 for v in _HARD_LOWER]
FALLBACK_UPPER = [v * 0.95 for v in _HARD_UPPER]

CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ


# ─────────────────────────────────────────────────────────────────────────
# 4. Analytic forward kinematics, built symbolically with CasADi.
# ─────────────────────────────────────────────────────────────────────────

def _rot_rpy(roll, pitch, yaw):
    """URDF <origin rpy="..."/> convention: fixed-axis roll(X) -> pitch(Y) ->
    yaw(Z), composed as R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = ca.DM([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = ca.DM([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = ca.DM([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_axis(axis, q):
    """Rotation by symbolic angle q about a unit basis axis (0/1,0/1,0/1)."""
    c, s = ca.cos(q), ca.sin(q)
    if axis == (1, 0, 0):
        return ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, c, -s),
            ca.horzcat(0, s, c),
        )
    if axis == (0, 1, 0):
        return ca.vertcat(
            ca.horzcat(c, 0, s),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-s, 0, c),
        )
    if axis == (0, 0, 1):
        return ca.vertcat(
            ca.horzcat(c, -s, 0),
            ca.horzcat(s, c, 0),
            ca.horzcat(0, 0, 1),
        )
    raise ValueError(f"Unsupported axis {axis} (only pure basis axes appear in this arm's URDF)")


def _fk_expr(q):
    """Build the FK expression graph from a symbolic 7-vector q. Generic over
    CasADi symbol type (SX or MX) since all ops used (cos/sin/matmul/vertcat)
    have both overloads -- this lets the Opti-based solver build FK directly
    on its own MX decision variable (no SX<->MX substitute() needed) while
    the self-test / logging Function below uses a plain SX symbol."""
    p = ca.DM.zeros(3, 1)
    R = ca.DM.eye(3)
    for i, (xyz, rpy, axis) in enumerate(_ARM_CHAIN):
        p = p + R @ ca.DM(xyz)
        R = R @ _rot_rpy(*rpy)
        R = R @ _rot_axis(axis, q[i])
    # Final fixed palm offset (no rotation).
    p = p + R @ ca.DM(_PALM_FIXED_OFFSET)
    return p, R


def build_fk():
    """Returns (q_sym, p_palm, R_palm): q_sym is a CasADi SX(7,1); p_palm/R_palm
    are symbolic expressions for the palm link's position/orientation in the
    torso_link frame, as functions of q_sym."""
    q = ca.SX.sym("q", 7)
    p, R = _fk_expr(q)
    return q, p, R


# ─────────────────────────────────────────────────────────────────────────
# 5. CasADi IK solver — one Opti problem built once, re-solved every tick
#    with fresh parameter values and a warm-started initial guess.
# ─────────────────────────────────────────────────────────────────────────

class CasadiIK:
    def __init__(self, w_pos: float, w_ori: float, w_reg: float):
        opti = ca.Opti()
        q_var = opti.variable(7)
        goal_p = opti.parameter(3)
        up_p = opti.parameter(3)
        q_prev_p = opti.parameter(7)
        q_lo_p = opti.parameter(7)
        q_hi_p = opti.parameter(7)

        # Build FK directly on the Opti MX decision variable -- no SX<->MX
        # substitute() needed (that mismatch is what a separate SX-built
        # graph would require, since opti.variable() is MX).
        p_palm, R_palm = _fk_expr(q_var)
        gp_expr = p_palm + R_palm @ ca.DM(PALM_LOCAL_OFFSET)
        n_expr = R_palm @ ca.DM(PALM_NORMAL_AXIS)

        pos_err = gp_expr - goal_p
        ori_term = 1.0 - ca.dot(n_expr, up_p)
        reg_err = q_var - q_prev_p

        cost = (
            w_pos * ca.sumsqr(pos_err)
            + w_ori * ori_term ** 2
            + w_reg * ca.sumsqr(reg_err)
        )
        opti.minimize(cost)
        opti.subject_to(opti.bounded(q_lo_p, q_var, q_hi_p))

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-6,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self.opti = opti
        self.q_var = q_var
        self.goal_p = goal_p
        self.up_p = up_p
        self.q_prev_p = q_prev_p
        self.q_lo_p = q_lo_p
        self.q_hi_p = q_hi_p
        # Plain numeric FK function, for logging / self-test (independent of
        # the Opti graph above -- built on its own SX symbol so it's callable
        # with plain numeric inputs without going through opti.solve()).
        q_sx, p_palm_sx, R_palm_sx = build_fk()
        gp_sx = p_palm_sx + R_palm_sx @ ca.DM(PALM_LOCAL_OFFSET)
        n_sx = R_palm_sx @ ca.DM(PALM_NORMAL_AXIS)
        self.fk_fn = ca.Function("fk", [q_sx], [gp_sx, n_sx])

    def solve(self, goal_local, up_local, q_prev, q_lower, q_upper):
        """All args plain python lists/floats. Returns a python list[7] (the
        solved q), falling back to q_prev unchanged if IPOPT fails so a
        transient solver hiccup never crashes the control loop."""
        self.opti.set_value(self.goal_p, goal_local)
        self.opti.set_value(self.up_p, up_local)
        self.opti.set_value(self.q_prev_p, q_prev)
        self.opti.set_value(self.q_lo_p, q_lower)
        self.opti.set_value(self.q_hi_p, q_upper)
        self.opti.set_initial(self.q_var, q_prev)
        try:
            sol = self.opti.solve()
            return [float(v) for v in sol.value(self.q_var)]
        except Exception as e:
            print(f"[CasadiIK] solve() failed ({e}); holding previous target.")
            return list(q_prev)


# ─────────────────────────────────────────────────────────────────────────
# 6. Main control loop
# ─────────────────────────────────────────────────────────────────────────

def _grasp_diag(state: dict, grasp_offset_z: float) -> dict:
    """Ground-truth reach-quality diagnostics computed directly from the
    published sim state (ee_pos/ee_quat/apple_pos) -- NOT from this script's
    own FK model -- so they're trustworthy even if the analytic FK chain
    somehow drifted from the real USD kinematics.

    Returns raw wrist-to-apple distance (kept for the periodic tick log and
    for continuity with older logs), the TRUE grasp-point distance (palm +
    PALM_LOCAL_OFFSET, vs apple - offset_z -- the actual metric
    reach_rewards.py/terminations_and_events.py score against, pos_tol=0.04,
    and, since the metric fix, the one the proximity stop consumes too), the
    palm-up orientation residual, and the raw (dx,dy,dz) world-frame gap for
    axis attribution.
    """
    ee_pos = torch.tensor(state["ee_pos"], dtype=torch.float32)
    ee_quat = torch.tensor(state["ee_quat"], dtype=torch.float32)  # wxyz
    apple_pos = torch.tensor(state["apple_pos"], dtype=torch.float32)

    delta = apple_pos - ee_pos
    raw_dist = float(torch.norm(delta))

    local_offset_world = quat_rotate(ee_quat, torch.tensor(PALM_LOCAL_OFFSET, dtype=torch.float32))
    grasp_point = ee_pos + local_offset_world
    target = apple_pos.clone()
    target[2] -= grasp_offset_z
    grasp_dist = float(torch.norm(target - grasp_point))

    normal_world = quat_rotate(ee_quat, torch.tensor(PALM_NORMAL_AXIS, dtype=torch.float32))
    theta_deg = math.degrees(math.acos(float(normal_world[2].clamp(-1.0, 1.0))))

    return {
        "raw_dist": raw_dist,
        "grasp_dist": grasp_dist,
        "theta_deg": theta_deg,
        "dxyz": tuple(float(v) for v in delta),
    }


def _selftest():
    """--selftest: solve IK for a known reachable point with no DDS/run_env
    involved, and confirm the FK of the solution lands within 1 cm."""
    print("[selftest] Building FK/IK...")
    ik = CasadiIK(w_pos=200.0, w_ori=20.0, w_reg=0.01)
    q_prev = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]  # robot's default right-arm pose
    # Within the arm's measured ~0.42 m physical reach from the shoulder
    # (see REACH_MAX's comment in apple_grasp_env_cfg.py) -- a point clearly
    # outside that envelope has no exact solution and isn't a useful test of
    # solver correctness, only of the reach envelope itself.
    goal = [0.25, -0.30, 0.05]    # a plausible in-front-of-shoulder point, torso frame
    up = [0.0, 0.0, 1.0]          # torso assumed world-aligned for this quick check
    q_sol = ik.solve(goal, up, q_prev, FALLBACK_LOWER, FALLBACK_UPPER)
    gp, n = ik.fk_fn(q_sol)
    gp = np.array(gp).flatten().tolist()
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(gp, goal)))
    print(f"[selftest] goal={goal}  solved grasp_point={[round(v, 4) for v in gp]}  "
          f"err={err * 1000:.2f} mm  q={[round(v, 3) for v in q_sol]}")
    print("[selftest] PASS" if err < 0.01 else "[selftest] FAIL (err >= 1cm)")


def main():
    parser = argparse.ArgumentParser(description="CasADi IK reacher for run_env.py")
    parser.add_argument("--w_pos", type=float, default=200.0, help="IK cost weight: grasp-point position error")
    parser.add_argument("--w_ori", type=float, default=20.0, help="IK cost weight: palm-up orientation error")
    parser.add_argument("--w_reg", type=float, default=0.01, help="IK cost weight: regularization toward previous q (branch selection / minimal motion)")
    parser.add_argument("--grasp_offset_z", type=float, default=GRASP_OFFSET_Z_DEFAULT,
                         help="Vertical offset (m) below the apple centre used as the IK goal (matches GRASP_OFFSET_Z in apple_grasp_env_cfg.py)")
    parser.add_argument("--solve_every", type=int, default=1,
                         help="Re-solve the IK NLP every N control ticks (default 1 = every tick / 50 Hz). "
                              "Raise this if IPOPT solves prove too slow to keep up with the 50 Hz publish loop.")
    parser.add_argument("--no_proximity_stop", action="store_true",
                         help="Disable the quintic proximity-stop safety controller (see proximity_stop.py).")
    parser.add_argument("--stop_dist", type=float, default=None, help="Proximity-stop trigger distance, palm to apple centre (m). Default: QuinticStopConfig.stop_dist.")
    parser.add_argument("--stop_dwell", type=float, default=None, help="Seconds within --stop_dist before the stop triggers. Default: QuinticStopConfig.dwell_time.")
    parser.add_argument("--stop_duration", type=float, default=None, help="Seconds the quintic deceleration takes. Default: QuinticStopConfig.stop_duration.")
    parser.add_argument("--stop_watchdog", type=float, default=5.0,
                         help="Seconds since episode start before an unconditional stop. Defaults to 5.0s here "
                              "(vs run_env.py --policy's 2.5s, QuinticStopConfig.watchdog_time) because the IK "
                              "controller's closed-loop convergence + IPOPT re-solves legitimately take longer "
                              "to settle near the apple than the trained policy's forward pass. Pass 0 or a "
                              "negative number to disable.")
    parser.add_argument("--selftest", action="store_true", help="Solve one IK problem with no DDS/run_env and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not DDS_OK:
        print("Cannot run: DDS unavailable.")
        return

    state = wait_for_state()
    print("Got initial state from run_env.py.")

    print("Building CasADi IK (analytic FK + IPOPT NLP)...")
    ik = CasadiIK(w_pos=args.w_pos, w_ori=args.w_ori, w_reg=args.w_reg)
    print("IK ready.")

    # ── Proximity-stop safety controller (see proximity_stop.py) — same
    # class run_env.py uses for --policy runs, driven the same way here. ──
    from proximity_stop import QuinticStopController, QuinticStopConfig
    stop_ctrl = None
    if not args.no_proximity_stop:
        _kwargs = {}
        if args.stop_dist is not None:
            _kwargs["stop_dist"] = args.stop_dist
        if args.stop_dwell is not None:
            _kwargs["dwell_time"] = args.stop_dwell
        if args.stop_duration is not None:
            _kwargs["stop_duration"] = args.stop_duration
        if args.stop_watchdog is not None:
            _kwargs["watchdog_time"] = args.stop_watchdog
        stop_cfg = QuinticStopConfig(**_kwargs)
        stop_ctrl = QuinticStopController(num_joints=7, cfg=stop_cfg, device="cpu")
        _wd = stop_cfg.watchdog_time
        _wd_str = f"{_wd:.2f}s" if (_wd is not None and _wd > 0.0) else "disabled"
        print(f"[ProximityStop] armed — stop_dist={stop_cfg.stop_dist*100:.1f}cm "
              f"(palm-ref to grasp target), "
              f"dwell={stop_cfg.dwell_time:.2f}s, stop_duration={stop_cfg.stop_duration:.2f}s, "
              f"resume_margin={stop_cfg.resume_margin*100:.1f}cm, watchdog={_wd_str} "
              f"(disable with --no_proximity_stop)")

    # ── Episode tracking (client-side reset detection, see plan docstring
    # above -- run_env.py's own reset happens in ITS terminal, this script
    # has no direct signal for it) ──
    q_prev = list(state["arm_joint_pos"])
    prev_apple_pos = list(state["apple_pos"])
    episode_start_t = time.time()
    q_prev_for_dq = torch.tensor(q_prev, dtype=torch.float32)
    t_prev_for_dq = episode_start_t

    q_target = list(q_prev)  # last commanded IK solution / stop-override target
    tick = 0
    prev_stop_state = "IDLE"

    print("\n" + "=" * 60)
    print("CASADI IK REACHER — driving the right arm via DDS")
    print("  Run  ~/IsaacLab/isaaclab.sh -p scripts/run_env.py  (no --policy)")
    print("  in another terminal if it isn't running already.")
    print("  Ctrl+C here to stop (publishes zero raw action first).")
    print("=" * 60 + "\n")

    try:
        while True:
            loop_t0 = time.perf_counter()
            with _state_lock:
                state = dict(_latest_state) if _latest_state is not None else state

            torso_pos = torch.tensor(state["torso_pos"], dtype=torch.float32)
            torso_quat = torch.tensor(state["torso_quat"], dtype=torch.float32)
            apple_pos = list(state["apple_pos"])
            arm_joint_pos = list(state["arm_joint_pos"])
            default_arm_pos = list(state["default_arm_pos"])
            q_lower = state.get("arm_joint_lower") or FALLBACK_LOWER
            q_upper = state.get("arm_joint_upper") or FALLBACK_UPPER

            now = time.time()

            # ── Episode-reset detection ──────────────────────────────────
            apple_moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(apple_pos, prev_apple_pos))) > 0.005
            near_default = math.sqrt(sum((a - b) ** 2 for a, b in zip(arm_joint_pos, default_arm_pos))) < 0.03
            we_were_elsewhere = math.sqrt(sum((a - b) ** 2 for a, b in zip(q_target, default_arm_pos))) > 0.05
            reset_detected = apple_moved or (near_default and we_were_elsewhere and tick > 0)
            if reset_detected:
                print("\n>>> New episode detected (apple moved / arm reset) — "
                      "re-solving IK, resetting proximity stop <<<\n")
                q_prev = list(arm_joint_pos)
                episode_start_t = now
                q_prev_for_dq = torch.tensor(arm_joint_pos, dtype=torch.float32)
                t_prev_for_dq = now
                if stop_ctrl is not None:
                    stop_ctrl.reset(t=0.0)
                prev_apple_pos = list(apple_pos)

            t_ep = now - episode_start_t

            # ── Finite-difference joint-velocity estimate (dq is NOT
            # published over DDS — run_env.py only computes it sim-side). ──
            q_now = torch.tensor(arm_joint_pos, dtype=torch.float32)
            dt_dq = max(now - t_prev_for_dq, 1e-3)
            dq_now = (q_now - q_prev_for_dq) / dt_dq
            q_prev_for_dq = q_now
            t_prev_for_dq = now

            # ── IK solve (every --solve_every ticks) ─────────────────────
            if tick % max(1, args.solve_every) == 0:
                goal_world = torch.tensor(apple_pos, dtype=torch.float32) - torch.tensor(
                    [0.0, 0.0, args.grasp_offset_z], dtype=torch.float32
                )
                goal_local = world_to_torso(goal_world, torso_pos, torso_quat).tolist()
                up_local = vec_world_to_torso(
                    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32), torso_quat
                ).tolist()
                q_target = ik.solve(goal_local, up_local, q_prev, q_lower, q_upper)
                q_prev = list(q_target)

            # ── Proximity stop — same distance metric run_env.py uses, which
            # is now the GRASP-POINT metric (palm reference point to the
            # target GRASP_OFFSET_Z below the apple centre), i.e. the one
            # reach_rewards.py scores against and the one QuinticStopConfig.
            # stop_dist is tuned for. It used to be raw ee_pos (palm-link
            # origin) to apple centre, which reads ~0.10 m at the very pose
            # this script is steering toward. ──
            q_cmd = q_target
            if stop_ctrl is not None:
                dist_now = _grasp_diag(state, args.grasp_offset_z)["grasp_dist"]
                # No contact sensing on this path (CasADi/IK, not the RL env
                # observation stack), so the contact trigger and contact
                # recall stay inactive here: proximity dwell + watchdog only.
                stop_ctrl.update_state(t_ep, dist_now, q_now, dq_now)
                overridden, q_cmd_t = stop_ctrl.get_command(t_ep, torch.tensor(q_target, dtype=torch.float32))
                if overridden:
                    q_cmd = [float(v) for v in q_cmd_t]
                    if tick % 100 == 0:
                        print(f"  [ProximityStop] {stop_ctrl.state} "
                              f"(trigger={stop_ctrl.last_trigger_reason}) — overriding IK target")
                # One-shot diagnostic the instant a stop actually triggers: how
                # far (m, joint-space rad) the arm still was from the IK
                # target at that moment -- if this is large, the stop fired
                # mid-approach (watchdog/dwell too tight), not because the IK
                # solution itself was wrong. Compare against the reach errors
                # this script logs -- if THIS gap correlates with your ~15cm
                # discrepancy, it's a timing issue, not a kinematics one.
                if stop_ctrl.state != "IDLE" and prev_stop_state == "IDLE":
                    joint_gap = float(torch.norm(torch.tensor(q_target, dtype=torch.float32) - q_now))
                    diag = _grasp_diag(state, args.grasp_offset_z)
                    print(f"\n>>> [ProximityStop] {stop_ctrl.last_trigger_reason} stop TRIGGERED at "
                          f"t_ep={t_ep:.2f}s — |q_target - q_now|={joint_gap:.3f} rad "
                          f"(0 = arm had fully converged)\n"
                          f"    raw wrist->apple   = {diag['raw_dist']*100:5.1f}cm  "
                          f"dxyz(cm)={tuple(round(v*100,1) for v in diag['dxyz'])}\n"
                          f"    TRUE grasp_point dist (matches reach_success pos_tol=4cm) "
                          f"= {diag['grasp_dist']*100:5.1f}cm\n"
                          f"    palm-up residual   = {diag['theta_deg']:5.1f} deg <<<\n")
                prev_stop_state = stop_ctrl.state

            # ── Publish ────────────────────────────────────────────────
            raw_arm = [(c - d) / ARM_SCALE for c, d in zip(q_cmd, default_arm_pos)]
            publish(raw_arm, HAND_OPEN_RAW)

            tick += 1
            if tick % 100 == 0:
                diag = _grasp_diag(state, args.grasp_offset_z)
                print(f"  tick={tick:6d}  d(ee,apple)={diag['raw_dist']*100:5.1f}cm  "
                      f"d(grasp_point,target)={diag['grasp_dist']*100:5.1f}cm  "
                      f"palm_up_err={diag['theta_deg']:4.1f}deg  q={[round(v,3) for v in q_cmd]}")

            elapsed = time.perf_counter() - loop_t0
            sleep_t = STEP_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nAborted — publishing zero raw action.")
        publish([0.0] * 7, [0.0] * 7)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        publish([0.0] * 7, [0.0] * 7)
        raise
