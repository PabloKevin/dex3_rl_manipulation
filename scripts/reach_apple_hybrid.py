#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Hybrid reacher — RL policy for the coarse approach, CasADi IK for the precise
finish, proximity_stop for the final smooth halt. NO Isaac Lab import here:
like reach_apple_casadi.py, this is a pure DDS client meant to run in a
DIFFERENT conda env (unitree_g1) than run_env.py's isaaclab_curobo env.

Why hybrid: the trained policy (scripts/train.py) is cheap (single forward
pass, 50 Hz) and generalizes across the apple's episode-to-episode jitter,
but it's only trained/rewarded for a coarse "get near the apple" reach --
right at the end it plateaus a few cm off, orientation included, exactly the
regime CasADi/IPOPT (reach_apple_casadi.py) is built for: an exact nonlinear
solve for palm position AND orientation, at the cost of more compute per
tick. So: let the cheap policy close most of the distance fast, then hand
off to the expensive-but-exact solver once close, then hand off to the
quintic proximity stop once converged. Using CasADi for the WHOLE reach
would waste its per-tick IPOPT cost on the easy far-away part of the motion,
where the policy is already good enough.

Why this can run as a pure DDS client (no Isaac Lab, no re-implementing the
env's observation logic): run_env.py now publishes the exact flat policy
observation vector (obs["policy"], 28 floats -- see its module docstring)
alongside the rest of rt/arm_tune/state. The exported policy.pt already has
the observation normalizer baked in (play.py's export_policy_as_jit call),
so torch.jit.load(policy.pt) + policy_module(obs_tensor) here reproduces
EXACTLY what run_env.py's own --policy path would compute in-process, given
the same obs. This script is the only thing driving the arm -- run it
WITHOUT --policy on run_env.py's side (same requirement as
reach_apple_casadi.py / reach_apple_ik.py), so run_env.py stays a passive
DDS-target follower the whole time and this script owns the RL->CasADi->stop
state machine entirely on the client side.

Usage:
  1. Terminal 1 (Isaac Sim, WITHOUT --policy -- --policy would make run_env.py
     drive the arm itself and ignore DDS targets, defeating this script):
       ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
  2. Terminal 2 (this script, unitree_g1 conda env -- needs torch, casadi,
     unitree_sdk2py; NOT Isaac Lab):
       /home/pablo_kevin/miniconda3/envs/unitree_g1/bin/python scripts/reach_apple_hybrid.py

DDS topics (same contract as run_env.py / reach_apple_casadi.py):
  rt/arm_tune/targets  (publish)    {"arm": [7 floats], "hand": [7 floats]}
                                     RAW pre-scale actions -- run_env.py applies
                                     target = default_joint_pos + scale * raw_action
  rt/arm_tune/state    (subscribe)  {"torso_pos", "torso_quat", "arm_joint_pos",
                                     "apple_pos", "ee_pos", "ee_quat", "jacobian",
                                     "default_arm_pos", "arm_joint_lower",
                                     "arm_joint_upper", "obs_policy"}

Install (only needed once, in whichever conda env you run this script from):
  pip install casadi
"""
import argparse
import json
import math
import os
import time
import threading

import casadi as ca
import torch

# ─────────────────────────────────────────────────────────────────────────
# 1. Quaternion / frame helpers — duplicated from reach_apple_casadi.py
#    (this repo's convention: each DDS client script carries its own small
#    copy rather than importing a shared module).
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
# 2. DDS setup — publisher on targets, subscriber on state.
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
_latest_state = None  # dict: torso_pos, torso_quat, arm_joint_pos, apple_pos, obs_policy, ...


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
#    a plain duplicated copy here, same as reach_apple_casadi.py.
# ─────────────────────────────────────────────────────────────────────────

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# (origin_xyz, origin_rpy, axis) for each of the 7 revolute joints, torso_link
# -> right_hand_palm_link, then the final FIXED palm offset. Same chain as
# reach_apple_casadi.py, verified against
# envs/assets/cuRobo/g1_29dof_with_hand_rev_1_0.urdf:1002-1175.
_ARM_CHAIN = [
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

APPLE_RADIUS = 0.04
GRASP_OFFSET_Z_DEFAULT = APPLE_RADIUS * 1.75

# Must match apple_grasp_env_cfg.py's ActionsCfg.right_arm.scale.
ARM_SCALE = 0.5

HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]

# Fallback joint limits (URDF hard limits x0.95 soft factor) if run_env.py's
# published arm_joint_lower/upper are ever missing.
_HARD_LOWER = [-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144]
_HARD_UPPER = [2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144]
FALLBACK_LOWER = [v * 0.95 for v in _HARD_LOWER]
FALLBACK_UPPER = [v * 0.95 for v in _HARD_UPPER]

CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ

# Expected flat length of the published obs_policy vector -- 7 (right_arm_pos)
# + 7 (right_arm_vel) + 3 (apple_pos_palm) + 4 (palm_orientation_torso) +
# 7 (last_action) + 1 (palm_touch_force), per envs/apple_grasp_env_cfg.py's
# ObservationsCfg.PolicyObs. Only used to fail loudly if run_env.py and this
# script's obs contract ever drift apart, instead of silently feeding the
# policy a malformed tensor. Bumped 28->29 when palm tactile sensing was
# added to the reach task -- only matches a policy trained AFTER that
# change; a pre-existing (still-preserved) 28-dim checkpoint will trip this
# guard if loaded here, which is the intended fail-loud behavior, not a bug.
OBS_POLICY_DIM = 29


# ─────────────────────────────────────────────────────────────────────────
# 4. Analytic forward kinematics + CasADi IK solver — duplicated verbatim
#    from reach_apple_casadi.py (used for the fine-reach phase only).
# ─────────────────────────────────────────────────────────────────────────

def _rot_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = ca.DM([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = ca.DM([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = ca.DM([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_axis(axis, q):
    c, s = ca.cos(q), ca.sin(q)
    if axis == (1, 0, 0):
        return ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, c, -s), ca.horzcat(0, s, c))
    if axis == (0, 1, 0):
        return ca.vertcat(ca.horzcat(c, 0, s), ca.horzcat(0, 1, 0), ca.horzcat(-s, 0, c))
    if axis == (0, 0, 1):
        return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
    raise ValueError(f"Unsupported axis {axis} (only pure basis axes appear in this arm's URDF)")


def _fk_expr(q):
    p = ca.DM.zeros(3, 1)
    R = ca.DM.eye(3)
    for i, (xyz, rpy, axis) in enumerate(_ARM_CHAIN):
        p = p + R @ ca.DM(xyz)
        R = R @ _rot_rpy(*rpy)
        R = R @ _rot_axis(axis, q[i])
    p = p + R @ ca.DM(_PALM_FIXED_OFFSET)
    return p, R


def build_fk():
    q = ca.SX.sym("q", 7)
    p, R = _fk_expr(q)
    return q, p, R


class CasadiIK:
    def __init__(self, w_pos: float, w_ori: float, w_reg: float):
        opti = ca.Opti()
        q_var = opti.variable(7)
        goal_p = opti.parameter(3)
        up_p = opti.parameter(3)
        q_prev_p = opti.parameter(7)
        q_lo_p = opti.parameter(7)
        q_hi_p = opti.parameter(7)

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
        q_sx, p_palm_sx, R_palm_sx = build_fk()
        gp_sx = p_palm_sx + R_palm_sx @ ca.DM(PALM_LOCAL_OFFSET)
        n_sx = R_palm_sx @ ca.DM(PALM_NORMAL_AXIS)
        self.fk_fn = ca.Function("fk", [q_sx], [gp_sx, n_sx])

    def solve(self, goal_local, up_local, q_prev, q_lower, q_upper):
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
# 5. Diagnostics — same ground-truth metric reach_apple_casadi.py logs.
# ─────────────────────────────────────────────────────────────────────────

def _grasp_diag(state: dict, grasp_offset_z: float) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────
# 6. Policy loading — mirrors run_env.py's _find_last_policy, duplicated
#    since this script has no Isaac Lab import to share it through.
# ─────────────────────────────────────────────────────────────────────────

def _find_last_policy() -> str:
    base_dir = "/home/pablo_kevin/dex3_rl_manipulation/logs/rsl_rl/apple_grasp_reach"
    latest_folder = max(os.listdir(base_dir))
    exported_path = os.path.join(base_dir, latest_folder, "exported", "policy.pt")
    if not os.path.isfile(exported_path):
        raise FileNotFoundError(
            f"No exported policy at {exported_path}. Run play.py against this "
            f"checkpoint first (it exports policy.pt as a side effect), e.g.:\n"
            f"  python scripts/play.py --task AppleGrasp-Play-v0 --num_envs 1"
        )
    return exported_path


# ─────────────────────────────────────────────────────────────────────────
# 7. Main control loop — RL phase -> CasADi phase -> proximity stop.
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hybrid RL+CasADi reacher for run_env.py")
    parser.add_argument("--policy", type=str, default="last",
                         help="Path to the JIT-exported policy.pt (e.g. from play.py), or "
                              "'last' (default) to auto-pick the newest run under "
                              "logs/rsl_rl/apple_grasp_reach/.")
    parser.add_argument("--switch_dist", type=float, default=0.10,
                         help="Raw wrist(ee)-to-apple distance (m) at which control hands "
                              "off from the RL policy to CasADi IK. Should be comfortably "
                              "above --stop_dist so CasADi still has room to refine "
                              "position AND orientation before proximity_stop engages.")
    parser.add_argument("--rl_max_time", type=float, default=2.5,
                         help="Seconds since episode start after which control hands off to "
                              "CasADi UNCONDITIONALLY, even if the policy never got within "
                              "--switch_dist -- CasADi's exact solve is the more reliable "
                              "closer, so this is a convergence fallback, not just a safety "
                              "one. Pass 0 or negative to disable (RL phase only ends via "
                              "--switch_dist).")
    parser.add_argument("--w_pos", type=float, default=200.0, help="CasADi cost weight: grasp-point position error")
    parser.add_argument("--w_ori", type=float, default=20.0, help="CasADi cost weight: palm-up orientation error")
    parser.add_argument("--w_reg", type=float, default=0.01, help="CasADi cost weight: regularization toward previous q")
    parser.add_argument("--grasp_offset_z", type=float, default=0.01,
                         help="Vertical offset (m) below the apple centre used as the CasADi goal "
                              "(matches GRASP_OFFSET_Z in apple_grasp_env_cfg.py)")
    parser.add_argument("--solve_every", type=int, default=1,
                         help="Re-solve the CasADi NLP every N control ticks during the CasADi "
                              "phase (default 1 = every tick / 50 Hz).")
    parser.add_argument("--no_proximity_stop", action="store_true",
                         help="Disable the quintic proximity-stop safety controller. It is only "
                              "ever armed during the CasADi phase (matches the pipeline: RL "
                              "approach -> CasADi fine reach -> proximity stop).")
    parser.add_argument("--stop_dist", type=float, default=0.03, help="Proximity-stop trigger distance, palm to apple centre (m). Default: QuinticStopConfig.stop_dist.")
    parser.add_argument("--stop_dwell", type=float, default=0.3, help="Seconds within --stop_dist before the stop triggers. Default: QuinticStopConfig.dwell_time.")
    parser.add_argument("--stop_duration", type=float, default=0.3, help="Seconds the quintic deceleration takes. Default: QuinticStopConfig.stop_duration.")
    parser.add_argument("--stop_watchdog", type=float, default=5.0,
                         help="Seconds since the CasADi phase started before an unconditional "
                              "proximity stop (independent of --rl_max_time, which only bounds "
                              "the RL phase). Pass 0 or negative to disable.")
    args = parser.parse_args()

    if not DDS_OK:
        print("Cannot run: DDS unavailable.")
        return

    policy_path = _find_last_policy() if args.policy == "last" else args.policy
    print(f"[Policy] Loading {policy_path} ...")
    policy_module = torch.jit.load(policy_path, map_location="cpu").eval()
    print("[Policy] Loaded — will replay it locally from published obs_policy.")

    state = wait_for_state()
    if state.get("obs_policy") is None:
        raise RuntimeError(
            "run_env.py's state message has no 'obs_policy' field -- this script needs the "
            "updated run_env.py that publishes it (see scripts/run_env.py's _publish_state). "
            "Restart run_env.py after pulling that change."
        )
    if len(state["obs_policy"]) != OBS_POLICY_DIM:
        print(f"[WARNING] obs_policy has {len(state['obs_policy'])} floats, expected "
              f"{OBS_POLICY_DIM} -- envs/apple_grasp_env_cfg.py's ObservationsCfg.PolicyObs "
              f"and this script's OBS_POLICY_DIM have drifted apart. Feeding it to the "
              f"policy anyway, but a shape mismatch will make torch.jit.load's forward pass "
              f"raise.")
    print("Got initial state from run_env.py.")

    print("Building CasADi IK (analytic FK + IPOPT NLP)...")
    ik = CasadiIK(w_pos=args.w_pos, w_ori=args.w_ori, w_reg=args.w_reg)
    print("IK ready.")

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
        print(f"[ProximityStop] armed (CasADi phase only) — stop_dist={stop_cfg.stop_dist*100:.1f}cm, "
              f"dwell={stop_cfg.dwell_time:.2f}s, stop_duration={stop_cfg.stop_duration:.2f}s, "
              f"resume_margin={stop_cfg.resume_margin*100:.1f}cm, watchdog={_wd_str} "
              f"(disable with --no_proximity_stop)")

    # ── Episode tracking (client-side reset detection -- run_env.py's own
    # reset happens in ITS terminal, this script has no direct signal) ──
    prev_apple_pos = list(state["apple_pos"])
    default_arm_pos = list(state["default_arm_pos"])
    episode_start_t = time.time()
    q_prev_for_dq = torch.tensor(state["arm_joint_pos"], dtype=torch.float32)
    t_prev_for_dq = episode_start_t

    q_target = list(state["arm_joint_pos"])  # last commanded IK target (CasADi phase only)
    q_last_cmd = list(state["arm_joint_pos"])  # last commanded joint target, EITHER phase — used
                                                 # for reset detection below (q_target alone would go
                                                 # stale during a long RL phase, since only the CasADi
                                                 # branch updates it)
    phase = "RL"
    tick = 0
    prev_stop_state = "IDLE"

    print("\n" + "=" * 60)
    print("HYBRID REACHER — RL policy -> CasADi IK -> proximity stop")
    print("  Run  ~/IsaacLab/isaaclab.sh -p scripts/run_env.py  (no --policy)")
    print("  in another terminal if it isn't running already.")
    print(f"  Phase switch: RL -> CasADi at dist < {args.switch_dist*100:.1f}cm "
          f"or after {args.rl_max_time:.1f}s")
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

            # ── Episode-reset detection (same heuristic reach_apple_casadi.py
            # uses: apple teleported, or arm snapped back near default while we
            # ourselves were commanding it elsewhere) ──────────────────────
            apple_moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(apple_pos, prev_apple_pos))) > 0.005
            near_default = math.sqrt(sum((a - b) ** 2 for a, b in zip(arm_joint_pos, default_arm_pos))) < 0.03
            we_were_elsewhere = math.sqrt(sum((a - b) ** 2 for a, b in zip(q_last_cmd, default_arm_pos))) > 0.05
            reset_detected = apple_moved or (near_default and we_were_elsewhere and tick > 0)
            if reset_detected:
                print("\n>>> New episode detected (apple moved / arm reset) — "
                      "back to RL phase, resetting proximity stop <<<\n")
                phase = "RL"
                q_target = list(arm_joint_pos)
                q_last_cmd = list(arm_joint_pos)
                episode_start_t = now
                q_prev_for_dq = torch.tensor(arm_joint_pos, dtype=torch.float32)
                t_prev_for_dq = now
                if stop_ctrl is not None:
                    stop_ctrl.reset(t=0.0)
                prev_stop_state = "IDLE"
                prev_apple_pos = list(apple_pos)

            t_ep = now - episode_start_t

            # ── Finite-difference joint-velocity estimate (dq is NOT
            # published over DDS — run_env.py only computes it sim-side). ──
            q_now = torch.tensor(arm_joint_pos, dtype=torch.float32)
            dt_dq = max(now - t_prev_for_dq, 1e-3)
            dq_now = (q_now - q_prev_for_dq) / dt_dq
            q_prev_for_dq = q_now
            t_prev_for_dq = now

            # ── Phase switch: RL -> CasADi ────────────────────────────────
            if phase == "RL":
                ee_pos = torch.tensor(state["ee_pos"], dtype=torch.float32)
                dist_now = float(torch.norm(torch.tensor(apple_pos, dtype=torch.float32) - ee_pos))
                watchdog_fired = args.rl_max_time > 0.0 and t_ep >= args.rl_max_time
                if dist_now < args.switch_dist or watchdog_fired:
                    phase = "CASADI"
                    q_target = list(arm_joint_pos)
                    reason = "watchdog" if watchdog_fired and dist_now >= args.switch_dist else "proximity"
                    print(f"\n>>> [Phase] RL -> CASADI at t_ep={t_ep:.2f}s "
                          f"(dist={dist_now*100:.1f}cm, trigger={reason}) <<<\n")

            # ── Drive the arm ──────────────────────────────────────────────
            if phase == "RL":
                obs_t = torch.tensor([state["obs_policy"]], dtype=torch.float32)
                with torch.inference_mode():
                    raw_action = policy_module(obs_t)[0]
                raw_arm = [float(v) for v in raw_action]
                q_cmd = [d + ARM_SCALE * r for d, r in zip(default_arm_pos, raw_arm)]  # for logging only
            else:  # CASADI
                if tick % max(1, args.solve_every) == 0:
                    goal_world = torch.tensor(apple_pos, dtype=torch.float32) - torch.tensor(
                        [0.0, 0.0, args.grasp_offset_z], dtype=torch.float32
                    )
                    goal_local = world_to_torso(goal_world, torso_pos, torso_quat).tolist()
                    up_local = vec_world_to_torso(
                        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32), torso_quat
                    ).tolist()
                    q_target = ik.solve(goal_local, up_local, q_target, q_lower, q_upper)

                q_cmd = q_target
                if stop_ctrl is not None:
                    ee_pos = torch.tensor(state["ee_pos"], dtype=torch.float32)
                    dist_now = float(torch.norm(torch.tensor(apple_pos, dtype=torch.float32) - ee_pos))
                    stop_ctrl.update_state(t_ep, dist_now, q_now, dq_now)
                    overridden, q_cmd_t = stop_ctrl.get_command(t_ep, torch.tensor(q_target, dtype=torch.float32))
                    if overridden:
                        q_cmd = [float(v) for v in q_cmd_t]
                        if tick % 100 == 0:
                            print(f"  [ProximityStop] {stop_ctrl.state} "
                                  f"(trigger={stop_ctrl.last_trigger_reason}) — overriding CasADi target")
                    if stop_ctrl.state != "IDLE" and prev_stop_state == "IDLE":
                        joint_gap = float(torch.norm(torch.tensor(q_target, dtype=torch.float32) - q_now))
                        diag = _grasp_diag(state, args.grasp_offset_z)
                        print(f"\n>>> [ProximityStop] {stop_ctrl.last_trigger_reason} stop TRIGGERED at "
                              f"t_ep={t_ep:.2f}s — |q_target - q_now|={joint_gap:.3f} rad\n"
                              f"    raw wrist->apple   = {diag['raw_dist']*100:5.1f}cm  "
                              f"dxyz(cm)={tuple(round(v*100,1) for v in diag['dxyz'])}\n"
                              f"    TRUE grasp_point dist (matches reach_success pos_tol=4cm) "
                              f"= {diag['grasp_dist']*100:5.1f}cm\n"
                              f"    palm-up residual   = {diag['theta_deg']:5.1f} deg <<<\n")
                    prev_stop_state = stop_ctrl.state

                raw_arm = [(c - d) / ARM_SCALE for c, d in zip(q_cmd, default_arm_pos)]

            q_last_cmd = list(q_cmd)

            # ── Publish ────────────────────────────────────────────────
            publish(raw_arm, HAND_OPEN_RAW)

            tick += 1
            if tick % 100 == 0:
                diag = _grasp_diag(state, args.grasp_offset_z)
                print(f"  [{phase:6s}] tick={tick:6d}  d(ee,apple)={diag['raw_dist']*100:5.1f}cm  "
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
