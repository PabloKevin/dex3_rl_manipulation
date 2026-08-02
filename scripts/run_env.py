#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 1 — View the Apple Grasp environment.
Usage:
# Basic viewer (holds sim, listens for targets, publishes state):
~/IsaacLab/isaaclab.sh -p scripts/run_env.py
# While sim is running, in a second terminal:
python3 scripts/reach_apple_curobo.py
# Or the IK-based reacher:
python3 scripts/reach_apple_ik.py
# Multiple envs:
~/IsaacLab/isaaclab.sh -p scripts/run_env.py --num_envs 4
# Print all robot link names:
~/IsaacLab/isaaclab.sh -p scripts/run_env.py --debug_links
# Print robot joint names:
~/IsaacLab/isaaclab.sh -p scripts/run_env.py --debug_joints

DDS topics:
rt/arm_tune/targets  (subscribe)  {"arm": [7 floats], "hand": [7 floats]}
RAW pre-scale actions — Isaac Lab applies
target = default_joint_pos + scale * raw_action

rt/arm_tune/state    (publish)    {"torso_pos": [x,y,z], "torso_quat": [w,x,y,z],
"arm_joint_pos": [7 floats] (absolute radians),
"apple_pos": [x,y,z],
"ee_pos": [x,y,z],
"jacobian": [21 floats] (3x7 pos-Jacobian, row-major),
"default_arm_pos": [7 floats],
"obs_policy": [28 floats] (flat AppleGrasp policy obs vector --
lets an external client replay policy.pt via
scripts/reach_apple_hybrid.py)}
All positions are WORLD frame, env-origin already
subtracted (irrelevant for single-env but kept for
correctness if you scale to --num_envs > 1 later —
note state is only ever published for env 0).
"""
import argparse
import sys
import os
import threading
import json

# ── Isaac Lab app must be set up BEFORE any omni/isaacsim imports ────────────
from isaaclab.app import AppLauncher

BASE_DIR =  "/home/pablo_kevin/dex3_rl_manipulation"

parser = argparse.ArgumentParser(description="Apple Grasp Environment Viewer")
parser.add_argument("--num_envs",       type=int,  default=1,    help="Number of parallel envs")
parser.add_argument("--debug_links",    action="store_true",      help="Print all robot link names and exit")
parser.add_argument("--debug_joints",   action="store_true",      help="Print all robot joint names and exit")
parser.add_argument("--random_actions", action="store_true",      help="Send random actions instead of zeros")
parser.add_argument("--policy",         type=str,  default=None,
                     help="Path to a JIT-exported policy (e.g. logs/rsl_rl/<exp>/<run>/exported/policy.pt "
                          "from play.py). If set, the loaded policy drives the arm every step instead of "
                          "DDS targets/zeros — reuses THIS script's already-running env, so you can 'r'+Enter "
                          "reset and re-run the policy repeatedly without relaunching Isaac Sim.")
parser.add_argument("--debug_pose",     action="store_true",
                     help="Print the palm's local-axis-to-world-Z probe once at start (to find "
                          "PALM_NORMAL_AXIS), plus a periodic joint-pose/distance readout and a "
                          "post-reset pose printout.")
parser.add_argument("--no_proximity_stop", action="store_true",
                     help="Disable the quintic proximity-stop safety controller (see "
                          "proximity_stop.py). Only relevant with --policy; ignored otherwise. "
                          "On by default: once the palm stays near the apple for --stop_dwell "
                          "seconds, control hands off from the RL policy to a smooth quintic "
                          "deceleration instead of letting the policy keep moving near the object.")
parser.add_argument("--stop_dist",     type=float, default=None,
                     help="Proximity-stop trigger distance, palm to apple centre (metres). "
                          "Defaults to QuinticStopConfig.stop_dist (proximity_stop.py) if not given.")
parser.add_argument("--stop_dwell",    type=float, default=None,
                     help="Seconds the palm must stay within --stop_dist before the stop triggers. "
                          "Defaults to QuinticStopConfig.dwell_time if not given.")
parser.add_argument("--stop_duration", type=float, default=None,
                     help="Seconds the quintic deceleration takes to bring the arm to rest. "
                          "Defaults to QuinticStopConfig.stop_duration if not given.")
parser.add_argument("--stop_watchdog", type=float, default=None,
                     help="Seconds since the current attempt started before the stop triggers "
                          "UNCONDITIONALLY, even if the palm never got near the apple. Pass 0 "
                          "or a negative number to disable. Defaults to QuinticStopConfig."
                          "watchdog_time if not given.")
parser.add_argument("--auto_reset_on_success", action="store_true",
                     help="By default this script suppresses the env's own reach_success "
                          "termination so it never auto-resets on its own — only 'r'+Enter "
                          "resets, so you can actually watch the arm sitting stopped at the "
                          "target. Pass this to opt back into an automatic reset once "
                          "reach_success fires, delayed by --auto_reset_delay seconds (NOT "
                          "instant like the default IsaacLab behaviour).")
parser.add_argument("--auto_reset_delay", type=float, default=2.0,
                     help="Seconds to wait after reach_success fires before auto-resetting. "
                          "Only used with --auto_reset_on_success.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch ─────────────────────────────────────
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, AppleGraspEnvCfg_PLAY
from proximity_stop import QuinticStopController, QuinticStopConfig

# Must match ActionsCfg.right_arm.joint_names order in apple_grasp_env_cfg.py
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
TORSO_LINK_NAME = "torso_link"
EE_LINK_NAME    = "right_hand_palm_link"


# ── Terminal Reset Listener ──────────────────────────────────────────────────
_reset_requested = False
_reset_lock = threading.Lock()

def _listen_terminal_reset():
    """Thread function to listen for 'r' input in the terminal without blocking sim."""
    global _reset_requested
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip().lower() == "r":
                with _reset_lock:
                    _reset_requested = True
        except Exception:
            break


# ── IK State Extractor ───────────────────────────────────────────────────────
class IKStateExtractor:
    """
    Safely extracts IK-relevant data from the running Articulation.
    Handles different Isaac Lab versions where the Jacobian attribute
    may be named 'jacobian_w', 'body_jacobian_w', or 'jacobian'.
    """

    def __init__(self, robot: Articulation, ee_body_idx: int, arm_joint_idx: list):
        self.robot = robot
        self.ee_body_idx = ee_body_idx
        self.arm_joint_idx = arm_joint_idx
        self._jac_attr = self._find_jacobian_attr()

    def _find_jacobian_attr(self) -> str | None:
        """Probe robot.data for the first Jacobian attribute that exists."""
        candidates = ["jacobian_w", "body_jacobian_w", "jacobian"]
        for attr in candidates:
            if hasattr(self.robot.data, attr):
                print(f"[IKStateExtractor] Found Jacobian attribute: '{attr}'")
                return attr
        # Last resort: try the raw PhysX view
        try:
            _ = self.robot.root_physx_view.get_jacobians()
            print("[IKStateExtractor] Using PhysX raw Jacobian fallback")
            return "__physx__"
        except Exception:
            pass
        print("[IKStateExtractor] WARNING: No Jacobian attribute found — "
              "IK client will receive zeros for the Jacobian.")
        return None

    def get_ik_state(self) -> dict:
        """Return a dict with ee_pos, default_arm_pos, and jacobian (3×7 flat)."""
        n_joints = len(self.arm_joint_idx)
        state = {}

        # 1. End-effector world position
        state["ee_pos"] = (
            self.robot.data.body_pos_w[0, self.ee_body_idx, :].tolist()
        )

        state["ee_quat"] = (self.robot.data.body_quat_w[0, self.ee_body_idx, :].tolist()) # [w, x, y, z]

        # 2. Default joint positions for the arm
        if hasattr(self.robot.data, "default_joint_pos"):
            state["default_arm_pos"] = (
                self.robot.data.default_joint_pos[0, self.arm_joint_idx].tolist()
            )
        else:
            state["default_arm_pos"] = [0.0] * n_joints

        # 3. Position Jacobian (3 × n_joints), flattened row-major
        if self._jac_attr is None:
            state["jacobian"] = [0.0] * (3 * n_joints)
        elif self._jac_attr == "__physx__":
            try:
                jacs = self.robot.root_physx_view.get_jacobians()
                # shape: (num_envs, num_bodies, 6, num_dofs)
                state["jacobian"] = (
                    jacs[0, self.ee_body_idx - 1, :3, self.arm_joint_idx]
                    .flatten()
                    .tolist()
                )
            except Exception:
                state["jacobian"] = [0.0] * (3 * n_joints)
        else:
            try:
                jac = getattr(self.robot.data, self._jac_attr)
                state["jacobian"] = (
                    jac[0, self.ee_body_idx - 1, :3, self.arm_joint_idx]
                    .flatten()
                    .tolist()
                )
            except Exception as e:
                print(f"[IKStateExtractor] Jacobian read error: {e}")
                state["jacobian"] = [0.0] * (3 * n_joints)

        # 4. Soft joint position limits for the arm (lower, upper), radians
        try:
            limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_idx, :]  # (n, 2)
            state["arm_joint_lower"] = limits[:, 0].tolist()
            state["arm_joint_upper"] = limits[:, 1].tolist()
        except Exception:
            state["arm_joint_lower"] = None
            state["arm_joint_upper"] = None

        return state


# ── DDS: subscriber for targets, publisher for state ─────────────────────────
_targets_lock = threading.Lock()
_arm_targets  = None   # None = hold default pose (zero actions)
_hand_targets = None
_dds_ok = False
_state_publisher = None

def _start_dds():
    """Start DDS subscriber (targets) + publisher (state). Non-fatal if unavailable."""
    global _dds_ok, _state_publisher
    try:
        from unitree_sdk2py.core.channel import (
            ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
        )
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        ChannelFactoryInitialize(1)   # channel 1 — same as reach_apple_curobo.py

        def _callback(msg: String_):
            global _arm_targets, _hand_targets
            try:
                data = json.loads(msg.data)
                with _targets_lock:
                    _arm_targets  = data.get("arm",  None)
                    _hand_targets = data.get("hand", None)
            except Exception as e:
                print(f"[DDS] target parse error: {e}")

        sub = ChannelSubscriber("rt/arm_tune/targets", String_)
        sub.Init(_callback, 1)

        _state_publisher = ChannelPublisher("rt/arm_tune/state", String_)
        _state_publisher.Init()

        _dds_ok = True
        print("[DDS] Subscribed to rt/arm_tune/targets, publishing rt/arm_tune/state — "
              "reach_apple_curobo.py / reach_apple_ik.py can now connect.")
    except Exception as e:
        print(f"[DDS] Unavailable ({e}). Running with zero actions only, no state published.")
        _dds_ok = False

def _get_actions(device, action_shape):
    """Return action tensor from latest DDS targets, or zeros if none received.

    Sized to match action_shape[-1]: 7 (arm-only, current reach-phase
    ActionsCfg) or 14 (arm+hand, if the Dex3 hand action is ever restored for
    grasp training) — rather than hardcoding 14, which broke the moment
    ActionsCfg dropped the hand action.
    """
    action_dim = action_shape[-1]
    with _targets_lock:
        arm = list(_arm_targets) if _arm_targets is not None else [0.0] * 7
        if action_dim == 7:
            combined = arm
        else:
            hand = list(_hand_targets) if _hand_targets is not None else [0.0] * 7
            combined = arm + hand
        return torch.tensor([combined], dtype=torch.float32, device=device)

def _publish_state(robot: Articulation, apple: RigidObject,
                   torso_idx: int, arm_joint_idx: list,
                   ik_extractor: IKStateExtractor, obs_policy: list | None = None):
    """Publish current torso pose (world), right-arm joint angles, apple position (world),
    IK-relevant data (ee_pos, jacobian, default_arm_pos), and the flat policy observation
    vector (obs_policy). The latter is what lets an external DDS client (e.g.
    reach_apple_hybrid.py) replay the SAME exported policy.pt outside this process: the
    exported JIT module already bakes in the observation normalizer, so feeding it this raw
    obs vector reproduces exactly what policy_module(obs["policy"]) would output in-process
    for --policy. Always published (even when this script isn't itself in --policy mode) so
    an external RL-driven client can consume it -- obs is computed by env.step()/reset()
    regardless of who supplies the actions."""
    if not _dds_ok or _state_publisher is None:
        return
    try:
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        torso_pos  = robot.data.body_pos_w[0, torso_idx, :].tolist()
        torso_quat = robot.data.body_quat_w[0, torso_idx, :].tolist()  # (w, x, y, z)
        arm_pos    = robot.data.joint_pos[0, arm_joint_idx].tolist()
        apple_pos  = apple.data.root_pos_w[0, :].tolist()

        # Merge IK-specific fields
        ik_state = ik_extractor.get_ik_state()

        payload = {
            "torso_pos": torso_pos,
            "torso_quat": torso_quat,
            "arm_joint_pos": arm_pos,
            "apple_pos": apple_pos,
            **ik_state,   # ee_pos, jacobian, default_arm_pos
            "obs_policy": obs_policy,
        }
        _state_publisher.Write(String_(data=json.dumps(payload)))
    except Exception as e:
        print(f"[DDS] state publish error: {e}")

def _find_last_policy():
    """Path to the latest run's JIT-exported policy — NOT the raw model_*.pt
    checkpoints (those are plain torch.save'd rsl_rl runner state, not a
    TorchScript archive; torch.jit.load on one fails with something like
    "PytorchStreamReader failed locating file constants.pkl"). The exported
    file only exists once play.py has been run against that run at least
    once (play.py is what calls export_policy_as_jit)."""
    base_dir = BASE_DIR + "/logs/rsl_rl/apple_grasp_reach"
    # Get the newest folder (alphabetical order works for YYYY-MM-DD_HH-MM-SS)
    latest_folder = max(os.listdir(base_dir))
    exported_path = os.path.join(base_dir, latest_folder, "exported", "policy.pt")
    if not os.path.isfile(exported_path):
        raise FileNotFoundError(
            f"No exported policy at {exported_path}. Run play.py against this "
            f"checkpoint first (it exports policy.pt as a side effect), e.g.:\n"
            f"  python scripts/play.py --task AppleGrasp-Play-v0 --num_envs 1"
        )
    return exported_path


def main():
    _start_dds()

    # Iniciar el hilo escuchador para el reset por terminal
    reset_thread = threading.Thread(target=_listen_terminal_reset, daemon=True)
    reset_thread.start()

    if args_cli.num_envs == 1:
        cfg = AppleGraspEnvCfg_PLAY()
    else:
        cfg = AppleGraspEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs

    env      = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
    # This script owns reset timing itself (manual 'r', or --auto_reset_on_success
    # below) instead of letting ManagerBasedRLEnv.step() instantly auto-reset the
    # moment reach_success fires — see terminations_and_events.reach_success.
    # Never set during training, so this has zero effect there.
    env._suppress_reach_success_termination = True
    obs, _   = env.reset()

    from envs.mdp.tree_utils import StemManager
    stem = StemManager(env)
    robot: Articulation = env.scene["robot"]

    if args_cli.debug_links:
        print("\n" + "="*60)
        print("ROBOT BODY (LINK) NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.body_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        env.close()
        return

    if args_cli.debug_joints:
        print("\n" + "="*60)
        print("ROBOT JOINT NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.joint_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        env.close()
        return

    # Resolve indices once, up front — used every step for state publishing.
    torso_idx     = robot.data.body_names.index(TORSO_LINK_NAME)
    arm_joint_idx = [robot.data.joint_names.index(n) for n in RIGHT_ARM_JOINT_NAMES]
    ee_body_idx   = robot.data.body_names.index(EE_LINK_NAME)

    # Build IK state extractor (probes for Jacobian attribute name)
    ik_extractor = IKStateExtractor(robot, ee_body_idx, arm_joint_idx)

    print("\n" + "="*60)
    print("APPLE GRASP VIEWER")
    print(f"  Physics dt : {cfg.sim.dt*1000:.1f} ms  ({1/cfg.sim.dt:.0f} Hz)")
    print(f"  Control dt : {cfg.sim.dt*cfg.decimation*1000:.1f} ms  "
          f"({1/(cfg.sim.dt*cfg.decimation):.0f} Hz)")
    print(f"  Action dim : {env.action_space.shape}")
    print(f"  Obs dim    : {obs['policy'].shape}")
    print("="*60)
    print("  Run  python3 scripts/reach_apple_curobo.py  (cuRobo planner)")
    print("   or  python3 scripts/reach_apple_ik.py       (Jacobian IK)")
    print("  in a second terminal to control joints.")
    print("  Without it the robot holds its default pose.")
    print("  Press 'r' + Enter in this terminal at any time to RESET.")
    print("="*60 + "\n")

    step  = 0
    apple: RigidObject = env.scene["apple"]
    try:
        sensor: ContactSensor = env.scene["contact_forces"]
    except Exception:
        sensor = None

    # ── Optional: drive the arm with a trained policy instead of DDS/zeros ──
    # Reuses THIS script's already-running env — no second gym.make(), so you
    # can 'r'+Enter reset and re-run the same loaded policy repeatedly without
    # relaunching Isaac Sim (unlike play.py, which owns its own env).
    policy_module = None
    if args_cli.policy:
        if args_cli.policy == "last":
            policy_path = _find_last_policy()
        elif args_cli.policy == "best":
            policy_path = BASE_DIR + "/envs/agents/best_weights/reach_apple.pt"
        else:
            policy_path = args_cli.policy
        policy_module = torch.jit.load(policy_path, map_location=env.device).eval()
        print(f"[Policy] Loaded {policy_path} — driving the arm every step.")

    palm_bid = robot.data.body_names.index("right_hand_palm_link")

    # ── Proximity-stop safety controller (see proximity_stop.py) ────────────
    # Only meaningful while a policy is actually driving the arm. Reads
    # ActionsCfg.right_arm's own scale/offset from the live action term
    # rather than hardcoding them, so it stays correct if those are retuned.
    stop_ctrl = None
    if policy_module is not None and not args_cli.no_proximity_stop:
        right_arm_term = env.action_manager.get_term("right_arm")
        _arm_scale = right_arm_term._scale
        _arm_offset = right_arm_term._offset
        # Only pass fields the user actually gave on the CLI -- omitting a
        # field (rather than passing None) is what lets QuinticStopConfig's
        # own dataclass defaults apply. Passing args_cli.stop_dist
        # unconditionally here was the previous bug: argparse always fills
        # in a default too, so it was ALWAYS passed explicitly, silently
        # shadowing any edit made to QuinticStopConfig's defaults in
        # proximity_stop.py -- editing that file had zero effect.
        _stop_kwargs = {}
        if args_cli.stop_dist is not None:
            _stop_kwargs["stop_dist"] = args_cli.stop_dist
        if args_cli.stop_dwell is not None:
            _stop_kwargs["dwell_time"] = args_cli.stop_dwell
        if args_cli.stop_duration is not None:
            _stop_kwargs["stop_duration"] = args_cli.stop_duration
        if args_cli.stop_watchdog is not None:
            _stop_kwargs["watchdog_time"] = args_cli.stop_watchdog
        stop_cfg = QuinticStopConfig(**_stop_kwargs)
        stop_ctrl = QuinticStopController(
            num_joints=len(arm_joint_idx),
            cfg=stop_cfg,
            device=env.device,
        )
        _wd = stop_cfg.watchdog_time
        _wd_str = f"{_wd:.2f}s" if (_wd is not None and _wd > 0.0) else "disabled"
        print(f"[ProximityStop] armed — stop_dist={stop_cfg.stop_dist*100:.1f}cm, "
              f"dwell={stop_cfg.dwell_time:.2f}s, stop_duration={stop_cfg.stop_duration:.2f}s, "
              f"resume_margin={stop_cfg.resume_margin*100:.1f}cm, watchdog={_wd_str} "
              f"(disable with --no_proximity_stop)")

        def _q_to_raw_action(q_cmd: torch.Tensor) -> torch.Tensor:
            """Inverse of JointAction.process_actions' affine transform
            (offset + scale*raw_action), so a direct joint-position target
            can be sent through the normal env.step(actions) path — this
            keeps the built-in RateLimitedJointPositionAction's own
            velocity/accel caps as an extra safety net under the quintic
            command too, rather than writing to the sim directly."""
            offset = _arm_offset[0] if isinstance(_arm_offset, torch.Tensor) else _arm_offset
            scale = _arm_scale if isinstance(_arm_scale, float) else _arm_scale[0]
            return (q_cmd - offset) / scale

    def _fmt(v):
        return [round(float(x), 3) for x in v]

    def _probe_palm_axes():
        """Local X/Y/Z of right_hand_palm_link expressed in world coords, for
        the CURRENT pose. Put the arm in a pose where you KNOW the palm faces
        up (viewer, or a tuned IK pose), then read which local axis has
        world-Z component closest to +1 — that's PALM_NORMAL_AXIS in
        apple_grasp_env_cfg.py. Closest to -1 means negate it."""
        from isaaclab.utils.math import matrix_from_quat
        q = robot.data.body_quat_w[0:1, palm_bid, :]
        R = matrix_from_quat(q)[0]
        print("\n─── PALM AXIS PROBE (local axis -> world vector) ───")
        for name, i in (("local X", 0), ("local Y", 1), ("local Z", 2)):
            ax = R[:, i]
            print(f"  {name} -> world {_fmt(ax)}    (world-Z component = {float(ax[2]): .3f})")
        print("  Pick the axis whose world-Z component is nearest +1 in a")
        print("  palm-up pose; negate it if it is nearest -1.")
        print("─────────────────────────────────────────────────────\n")

    if args_cli.debug_pose:
        _probe_palm_axes()

    global _reset_requested, _arm_targets, _hand_targets

    # Tracks how long env._reach_success[0] has been continuously True, for
    # --auto_reset_on_success's delay. None = not currently in a success
    # streak. Cleared on any reset (manual or delayed-auto) and the instant
    # _reach_success drops back False (e.g. the arm drifts back out of
    # tolerance) -- requires CONTINUOUS success for auto_reset_delay
    # seconds, not "was true at some point that long ago".
    _success_since = None

    while simulation_app.is_running():
        # Verificar si el usuario solicitó reset por terminal
        with torch.inference_mode(), _reset_lock:
            if _reset_requested:
                _reset_requested = False

                # 1. Vaciar objetivos DDS
                with _targets_lock:
                    _arm_targets = None
                    _hand_targets = None

                # 2. Reset de Gymnasium y tallo
                obs, _ = env.reset()
                stem.reset(env)

                # 3. Forzar posiciones y velocidades a CERO en el motor físico (PhysX)
                try:
                    robot.write_joint_state_to_sim(
                        robot.data.default_joint_pos,
                        torch.zeros_like(robot.data.default_joint_vel)
                    )
                    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                    robot.reset()
                except Exception as e:
                    print(f"[Reset Warning] {e}")

                # 4. PASO CLAVE: Simular 15 pasos con acciones CERO
                # Esto 'limpia' el buffer de los controladores PD e ignora el arrastre
                zero_actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
                for _ in range(15):
                    obs, _, _, _, _ = env.step(zero_actions)

                # 5. Volver a vaciar objetivos por si llegó algún paquete DDS durante los 15 pasos
                with _targets_lock:
                    _arm_targets = None
                    _hand_targets = None

                step = 0
                if stop_ctrl is not None:
                    stop_ctrl.reset(t=step * env.step_dt)
                _success_since = None
                print("\n>>> SIMULACIÓN Y ROBOT RESETEADOS A POSE INICIAL <<<\n")

                if args_cli.debug_pose:
                    _q  = robot.data.joint_pos[0, arm_joint_idx]
                    _dq = robot.data.joint_vel[0, arm_joint_idx]
                    print(f"  RESET -> q={_fmt(_q)}  |dq|={float(torch.norm(_dq)):.4f}")

        with torch.inference_mode():
            if policy_module is not None:
                actions = policy_module(obs["policy"])

                if stop_ctrl is not None:
                    _t = step * env.step_dt
                    _palm_now = robot.data.body_pos_w[0, palm_bid]
                    _dist_now = float(torch.norm(apple.data.root_pos_w[0] - _palm_now))
                    _q_now  = robot.data.joint_pos[0, arm_joint_idx]
                    _dq_now = robot.data.joint_vel[0, arm_joint_idx]
                    stop_ctrl.update_state(_t, _dist_now, _q_now, _dq_now)

                    # Scoped to env 0 only, matching the rest of this
                    # script's single-env-focused debug/publish logic — with
                    # --num_envs > 1 the override only ever applies to env 0's
                    # action row, other envs keep taking the raw policy output.
                    overridden, q_cmd = stop_ctrl.get_command(_t, _q_now)
                    if overridden:
                        actions[0, :len(arm_joint_idx)] = _q_to_raw_action(q_cmd)
                        if step % 100 == 0:
                            print(f"  [ProximityStop] {stop_ctrl.state} "
                                  f"(trigger={stop_ctrl.last_trigger_reason}) — overriding policy actions")
            elif args_cli.random_actions:
                raw     = env.action_space.sample()
                actions = torch.tensor(raw, device=env.device, dtype=torch.float32)
            else:
                actions = _get_actions(env.device, env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(actions)
            stem.update(env)

            _publish_state(robot, apple, torso_idx, arm_joint_idx, ik_extractor,
                           obs_policy=obs["policy"][0].tolist())

            step += 1
            if terminated.any() or truncated.any():
                stem.reset(env)
                # env.step() already teleported the terminated/truncated envs back to
                # their reset state internally — but without clearing the DDS targets
                # here too, the very next _get_actions() call re-applies whatever stale
                # target an external controller (reach_apple_curobo.py, the calibration
                # script, ...) is still streaming, snapping the arm right back toward
                # its pre-reset pose. Same fix as the manual 'r' reset path below.
                with _targets_lock:
                    _arm_targets = None
                    _hand_targets = None
                print("\n>>> Episode auto-reset (time_out/termination) — cleared stale DDS targets <<<\n")

            # reach_success's own termination is suppressed (env._suppress_
            # reach_success_termination, set above) so it never auto-resets
            # on its own — env._reach_success is still cached fresh every
            # step regardless, this is what --auto_reset_on_success watches
            # instead, with an actual delay rather than IsaacLab's instant
            # reset-inside-step() behaviour.
            _reach_success = getattr(env, "_reach_success", None)
            if _reach_success is not None and bool(_reach_success[0]):
                if _success_since is None:
                    _success_since = step * env.step_dt
                elif args_cli.auto_reset_on_success:
                    elapsed = step * env.step_dt - _success_since
                    if elapsed >= args_cli.auto_reset_delay:
                        with _reset_lock:
                            _reset_requested = True
                        print(f"\n>>> reach_success held {elapsed:.2f}s — "
                              f"auto-resetting (--auto_reset_on_success) <<<\n")
            else:
                _success_since = None

            if step % 100 == 0:
                apple_z = apple.data.root_pos_w[0, 2].item()
                rew_val = reward[0].item() if reward.numel() > 0 else 0.0
                print(f"  step={step:6d}  apple_z={apple_z:.3f}m  "
                      f"reward={rew_val:.4f}", end="")
                if sensor is not None:
                    total_f = sensor.data.net_forces_w[0].norm(dim=-1).sum().item()
                    print(f"  contact_N={total_f:.3f}", end="")
                if any(stem.is_broken()):
                    print("  STEM BROKEN 🍎", end="")
                print()

                if args_cli.debug_pose:
                    _q  = robot.data.joint_pos[0, arm_joint_idx]
                    _dq = robot.data.joint_vel[0, arm_joint_idx]
                    _palm = robot.data.body_pos_w[0, palm_bid]
                    _tgt  = apple.data.root_pos_w[0].clone()
                    _tgt[2] -= 0.08   # matches reach_rewards.py's offset_z
                    _d = float(torch.norm(_tgt - _palm))

                    from isaaclab.utils.math import matrix_from_quat
                    _Rp = matrix_from_quat(robot.data.body_quat_w[0:1, palm_bid, :])[0]
                    _up = [round(float(_Rp[:, i][2]), 2) for i in range(3)]  # world-Z of local X/Y/Z

                    print(f"           d={_d*100:6.1f} cm  local-axis world-Z=[X,Y,Z]={_up}  "
                          f"|dq|={float(torch.norm(_dq)):5.2f}")
                    print(f"           q  = {_fmt(_q)}")
                    print(f"           dq = {_fmt(_dq)}")

    env.close()

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"\n>>> CRASH: {e}")
    finally:
        simulation_app.close()