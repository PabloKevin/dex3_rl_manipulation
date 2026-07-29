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
"default_arm_pos": [7 floats]}
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

parser = argparse.ArgumentParser(description="Apple Grasp Environment Viewer")
parser.add_argument("--num_envs",       type=int,  default=1,    help="Number of parallel envs")
parser.add_argument("--debug_links",    action="store_true",      help="Print all robot link names and exit")
parser.add_argument("--debug_joints",   action="store_true",      help="Print all robot joint names and exit")
parser.add_argument("--random_actions", action="store_true",      help="Send random actions instead of zeros")
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
    """Return action tensor from latest DDS targets, or zeros if none received."""
    with _targets_lock:
        arm  = list(_arm_targets)  if _arm_targets  is not None else [0.0] * 7
        hand = list(_hand_targets) if _hand_targets is not None else [0.0] * 7
        combined = arm + hand
        return torch.tensor([combined], dtype=torch.float32, device=device)

def _publish_state(robot: Articulation, apple: RigidObject,
                   torso_idx: int, arm_joint_idx: list,
                   ik_extractor: IKStateExtractor):
    """Publish current torso pose (world), right-arm joint angles, apple position (world),
    and IK-relevant data (ee_pos, jacobian, default_arm_pos)."""
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
        }
        _state_publisher.Write(String_(data=json.dumps(payload)))
    except Exception as e:
        print(f"[DDS] state publish error: {e}")

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

    global _reset_requested, _arm_targets, _hand_targets

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
                print("\n>>> SIMULACIÓN Y ROBOT RESETEADOS A POSE INICIAL <<<\n")

        with torch.inference_mode():
            if args_cli.random_actions:
                raw     = env.action_space.sample()
                actions = torch.tensor(raw, device=env.device, dtype=torch.float32)
            else:
                actions = _get_actions(env.device, env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(actions)
            stem.update(env)

            _publish_state(robot, apple, torso_idx, arm_joint_idx, ik_extractor)

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