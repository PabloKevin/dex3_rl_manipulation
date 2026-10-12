# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
The reach policy's input/output, WITHOUT Isaac Lab: what the real robot
needs to run a reach policy exported by scripts/export_policy_jit.py (or
pinned in envs/agents/best_weights/).

numpy + torch + pinocchio only (the g1_dex3_sim2real env). Every constant
below is a copy of the training configuration; sim2real/validate_reach_io.py
checks them against the env cfgs and checks the observation, forward
kinematics and rate limiter against Isaac Lab, number by number. Run it
after changing a reach task, the robot model or this file.

Two policy layouts (picked from the policy's input size):
  "waist" (40 in / 8 out): AppleReachSpindle-v3 / AppleReachTreeWaist*-v1 --
      right arm (7) + waist yaw (1), envs/apple_reach_tree_waist_env_cfg.py
  "tree"  (37 in / 7 out): AppleReachTree-v1 -- right arm only

Observation, in the order the policy was trained on
(apple_grasp_env_cfg.ObservationsCfg.PolicyObs + the tree / waist terms):
  right arm q - default (7), right arm dq (7),
  apple centre - palm reference point (3, PELVIS axes: the sim's world frame,
      the pelvis is fixed upright there),
  palm orientation (4, w x y z) in the torso frame [waist: with the waist yaw
      taken out, waist_reach.palm_quat_torso_unyawed],
  previous raw arm action (7), palm contact force (1, N), clearance ring (8),
  [waist: waist yaw - default (1), waist yaw velocity (1), previous raw waist action (1)]

Actions: raw -> target = default + 0.5 * raw, clipped, then the rate
limiter of envs/mdp/actions.py (RateLimitedJointPositionAction), here
RateLimiter, with the deployment speed factor s in (0, 1]: velocity cap x s,
acceleration cap x s^2 (the trained trajectory shape played 1/s slower).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  Training constants (checked by validate_reach_io.py --constants)
# ─────────────────────────────────────────────
#: control period: sim.dt 0.005 x decimation 4
DT = 0.02
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
#: G1 29-dof motor index of each (rt/lowstate, rt/arm_sdk)
MOTOR_IDX = {"waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
             **{n: 22 + i for i, n in enumerate(RIGHT_ARM_JOINTS)}}
#: the training robot's default (home) pose: the reach starts here
ARM_DEFAULT = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
WAIST_DEFAULT = np.array([0.0, 0.0, 0.1])          # yaw, roll, pitch (roll/pitch locked in sim)
#: HOME = where every real run starts and where the arm returns: the default pose
#: with the shoulder 0.15 rad forward, out and rotated out -- inside the policy's
#: start distribution (default +-0.2 rad per joint, reset_arm_jitter) and the pose of
#: that box with the most room to the robot's own body (g1_fruits_picker body_collision,
#: Dex3 cup): elbow to torso +5.7 cm (default +1.9), legs +10.9 cm (default +7.2), also
#: with the waist +-0.1 rad. Reach from here vs from the default, AppleReachSpindle-v3,
#: 192 episodes each (sim2real/eval_start_pose.py): success 18% vs 14% (the strict
#: success gate), knocked off 2% vs 2%, median time to success 2.9 vs 3.4 s.
HOME_Q = ARM_DEFAULT + np.array([-0.15, -0.15, -0.15, 0.0, 0.0, 0.0, 0.0])
#: sim actuator of the right arm (ImplicitActuatorCfg: tau = kp (q* - q) - kd dq)
ARM_KP, ARM_KD = 100.0, 10.0
#: arm action term (apple_grasp_env_cfg.ActionsCfg.right_arm + v3 clip)
ARM_SCALE = 0.5
ARM_MAX_VELOCITY = 2.0
ARM_MAX_ACCELERATION = 8.0
#: AppleReachSpindle-v3 ARM_TARGET_CLIP (soft joint limits, absolute targets)
ARM_TARGET_LO = np.array([-2.945, -2.156, -2.487, -0.969, -1.874, -1.534, -1.534])
ARM_TARGET_HI = np.array([2.526, 1.492, 2.487, 2.016, 1.874, 1.534, 1.534])
#: older policies clip +-3 rad about the default instead
ARM_TARGET_CLIP_OLD = 3.0
#: waist action term (apple_reach_tree_waist_env_cfg.ActionsCfg.waist_yaw)
WAIST_SCALE = 0.5
WAIST_YAW_LIMIT = math.radians(45.0)
WAIST_MAX_VELOCITY = 1.0
WAIST_MAX_ACCELERATION = 4.0
#: grasp reference point and palm normal, in right_hand_palm_link's frame
PALM_LOCAL_OFFSET = np.array([0.08, 0.0, 0.0])
PALM_NORMAL_AXIS = np.array([0.0, 1.0, 0.0])
#: palm contact starts with the apple surface this far from the reference point; the
#: reach target presses 3 mm past it (GRASP_STANDOFF, ~1.2 N on the compliant apple)
PALM_TOUCH_HEIGHT = 0.016
GRASP_STANDOFF = 0.013
APPLE_CONTACT_STIFFNESS = 1000.0
TOUCH_FORCE_MIN, TOUCH_FORCE_MAX = 0.3, 6.0
RING_SECTORS = 8
#: the trees' apple radius range (the policy never sees it)
APPLE_RADIUS_RANGE = (0.032, 0.048)
#: Dex3 cup held during the spindle reach (apple_reach_spindle_env_cfg.REACH_HAND_PRESHAPE)
REACH_HAND_PRESHAPE = {
    "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.15, "right_hand_thumb_2_joint": -0.340,
    "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 1.396,
    "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 1.396,
}
#: rt/dex3/right order (g1_fruits_picker full_move2.DEX3_JOINT_NAMES)
DEX3_ORDER = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]

#: placement envelopes, relative to the right shoulder (right_shoulder_pitch_link) at
#: waist yaw 0, of the GRASP TARGET (the palm reference point at the touch)
ENVELOPE_WAIST = dict(reach_min=0.24, reach_max=0.44, az_center_deg=-25.0, az_span_deg=110.0,
                      dz_range=(-0.30, 0.28), min_forward=0.15, min_horizontal=0.18,
                      map=os.path.join(REPO, "envs", "assets", "reach_workspace_waist.npz"))
ENVELOPE_ARM = dict(reach_min=0.24, reach_max=0.34, az_center_deg=-45.0, az_span_deg=65.0,
                    dz_range=(-0.30, 0.28), min_forward=0.15, min_horizontal=0.18,
                    map=os.path.join(REPO, "envs", "assets", "reach_workspace.npz"))
#: stems point into the fruit wall within +-30 deg of +x: the palm comes in along +x
STEM_HEADING_MAX_DEG = 30.0

URDF_DEFAULT = os.path.expanduser("~/g1_fruits_picker/robot_model_urdf/g1_29dof_with_hand.urdf")
POLICY_DEFAULT = os.path.join(REPO, "envs", "agents", "best_weights", "grasp_handoff_reach_spindle.pt")

LAYOUTS = {40: "waist", 37: "tree"}


# ─────────────────────────────────────────────
#  Rotations (w, x, y, z), as Isaac Lab
# ─────────────────────────────────────────────

def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z), w >= 0 (Isaac Lab's quat_from_matrix convention)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ─────────────────────────────────────────────
#  Forward kinematics (pinocchio, fixed pelvis)
# ─────────────────────────────────────────────

def load_pinocchio():
    """The real pinocchio (PyPI 'pin'); g1_fruits_picker/apple_picker/ik_pin.load()
    explains the nose-plugin name clash it works around."""
    import importlib
    m = importlib.import_module("pinocchio")
    if hasattr(m, "buildModelFromUrdf"):
        return m
    import sys
    sys.path.insert(0, os.path.expanduser("~/g1_fruits_picker/apple_picker"))
    import ik_pin  # noqa: E402
    m = ik_pin.load()
    if m is None:
        raise ImportError("pinocchio (PyPI 'pin') not importable")
    return m


@dataclass
class Frames:
    """Poses in the PELVIS frame (= the training sim's world frame)."""
    palm_pos: np.ndarray        # right_hand_palm_link origin
    palm_R: np.ndarray
    palm_ref: np.ndarray        # grasp reference point (PALM_LOCAL_OFFSET)
    palm_normal: np.ndarray
    torso_pos: np.ndarray
    torso_R: np.ndarray
    shoulder_pos: np.ndarray    # right_shoulder_pitch_link
    camera_pos: np.ndarray      # d435_link
    camera_R: np.ndarray


class G1Kinematics:
    """Fixed-pelvis G1 model from the g1_fruits_picker URDF. Joints not given
    stay at 0 (the left arm and legs do not affect the right-arm frames)."""

    FRAMES = {"palm": "right_hand_palm_link", "torso": "torso_link",
              "shoulder": "right_shoulder_pitch_link", "camera": "d435_link"}

    def __init__(self, urdf: str = URDF_DEFAULT):
        pin = load_pinocchio()
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.idx_q = {self.model.names[j]: self.model.joints[j].idx_q for j in range(1, self.model.njoints)}
        self.idx_v = {self.model.names[j]: self.model.joints[j].idx_v for j in range(1, self.model.njoints)}
        self.fid = {k: self.model.getFrameId(v) for k, v in self.FRAMES.items()}
        for k, f in self.fid.items():
            if f >= self.model.nframes:
                raise KeyError(f"frame {self.FRAMES[k]} not in {urdf}")
        self.arm_q = [self.idx_q[n] for n in RIGHT_ARM_JOINTS]
        self.arm_v = [self.idx_v[n] for n in RIGHT_ARM_JOINTS]
        self.waist_q = [self.idx_q[n] for n in WAIST_JOINTS]
        self.waist_v = [self.idx_v[n] for n in WAIST_JOINTS]

    def set_q(self, waist3, arm7, named: dict | None = None) -> np.ndarray:
        q = self.q.copy()
        q[self.waist_q] = np.asarray(waist3, float)
        q[self.arm_q] = np.asarray(arm7, float)
        for n, v in (named or {}).items():
            if n in self.idx_q:
                q[self.idx_q[n]] = float(v)
        return q

    def frames(self, waist3, arm7, named: dict | None = None) -> Frames:
        pin = self.pin
        q = self.set_q(waist3, arm7, named)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        X = {k: self.data.oMf[f] for k, f in self.fid.items()}
        pR = np.array(X["palm"].rotation)
        pp = np.array(X["palm"].translation)
        return Frames(palm_pos=pp, palm_R=pR, palm_ref=pp + pR @ PALM_LOCAL_OFFSET,
                      palm_normal=pR @ PALM_NORMAL_AXIS,
                      torso_pos=np.array(X["torso"].translation), torso_R=np.array(X["torso"].rotation),
                      shoulder_pos=np.array(X["shoulder"].translation),
                      camera_pos=np.array(X["camera"].translation), camera_R=np.array(X["camera"].rotation))

    def gravity_torque(self, waist3, arm7, dq_arm=None, ddq_arm=None, named=None) -> np.ndarray:
        """[7] right-arm torque the motion itself needs (RNEA: M qdd + C dq + g), the
        model side of the safety monitor's external-torque residual."""
        pin = self.pin
        q = self.set_q(waist3, arm7, named)
        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)
        if dq_arm is not None:
            v[self.arm_v] = dq_arm
        if ddq_arm is not None:
            a[self.arm_v] = ddq_arm
        tau = pin.rnea(self.model, self.data, q, v, a)
        return np.asarray(tau)[self.arm_v]


def palm_quat_obs(fr: Frames, waist_yaw: float, unyawed: bool) -> np.ndarray:
    """The policy's palm-orientation input: palm relative to the torso, with the
    waist yaw taken out when `unyawed` (waist_reach.palm_quat_torso_unyawed:
    reference = Rz(-yaw) * torso, the pelvis being upright)."""
    R_ref = rot_z(-waist_yaw) @ fr.torso_R if unyawed else fr.torso_R
    return quat_from_matrix(R_ref.T @ fr.palm_R)


# ─────────────────────────────────────────────
#  Observation
# ─────────────────────────────────────────────

@dataclass
class ReachState:
    """What one control step needs (pelvis frame, radians)."""
    arm_q: np.ndarray           # [7] RIGHT_ARM_JOINTS order
    arm_dq: np.ndarray          # [7]
    waist_q: np.ndarray         # [3] yaw, roll, pitch
    waist_dq: np.ndarray        # [3]
    apple: np.ndarray           # [3] apple CENTRE
    touch_force: float = 0.0    # N, palm
    ring: np.ndarray | None = None   # [8] clearance ring (1 = free)


def build_obs(layout: str, kin: G1Kinematics, st: ReachState, prev_raw: np.ndarray,
              fr: Frames | None = None) -> tuple[np.ndarray, Frames]:
    """(obs, frames). prev_raw: [7] or [8] previous raw action (zeros at the start)."""
    fr = fr or kin.frames(st.waist_q, st.arm_q)
    ring = np.ones(RING_SECTORS) if st.ring is None else np.asarray(st.ring, float)
    parts = [
        st.arm_q - ARM_DEFAULT,
        st.arm_dq,
        st.apple - fr.palm_ref,
        palm_quat_obs(fr, float(st.waist_q[0]), unyawed=(layout == "waist")),
        np.asarray(prev_raw, float)[:7],
        [float(st.touch_force)],
        ring,
    ]
    if layout == "waist":
        parts += [[st.waist_q[0] - WAIST_DEFAULT[0]], [st.waist_dq[0]], [float(prev_raw[7])]]
    return np.concatenate([np.asarray(p, float).ravel() for p in parts]).astype(np.float32), fr


# ─────────────────────────────────────────────
#  Rate limiter (envs/mdp/actions.py, numpy)
# ─────────────────────────────────────────────

class RateLimiter:
    """RateLimitedJointPositionAction.process_actions for one env:
    desired = clip(offset + scale * raw); v = clip((desired - prev) / dt, +-vmax*s)
    [brake: v limited to what can still stop at desired]; v = clip(v, prev_v +- amax*s^2*dt);
    target = prev + v dt. Re-anchored to the MEASURED joints on the first two
    steps of an episode, as Isaac Lab does (episode_length_buf <= 1)."""

    ANCHOR_STEPS = 2

    def __init__(self, offset, scale, lo, hi, vmax, amax, dt=DT, brake=False, dtype=np.float64):
        f = self.f = np.dtype(dtype).type
        self.offset = np.asarray(offset, dtype)
        self.scale = f(scale)
        self.lo, self.hi = np.asarray(lo, dtype), np.asarray(hi, dtype)
        self.vmax, self.amax, self.dt, self.brake = f(vmax), f(amax), f(dt), bool(brake)
        self.prev = self.offset.copy()
        self.vel = np.zeros_like(self.offset)
        self.k = 0
        self.desired = self.offset.copy()

    def reset(self):
        self.k = 0

    def step(self, raw, q_meas, speed: float = 1.0) -> np.ndarray:
        f = self.f
        raw = np.asarray(raw, self.offset.dtype)
        desired = np.clip(raw * self.scale + self.offset, self.lo, self.hi)
        self.desired = desired
        if self.k < self.ANCHOR_STEPS:
            self.prev = np.asarray(q_meas, self.offset.dtype).copy()
            self.vel = np.zeros_like(self.prev)
        self.k += 1
        dt = self.dt
        speed = f(speed)
        vmax, amax = self.vmax * speed, self.amax * speed ** 2
        v = np.clip((desired - self.prev) / dt, -vmax, vmax)
        if self.brake:
            err = np.abs(desired - self.prev)
            v_stop = f(-0.5) * amax * dt + np.sqrt((f(0.5) * amax * dt) ** 2 + f(2.0) * amax * err)
            v = np.clip(v, -v_stop, v_stop)
        v = np.clip(v, self.vel - amax * dt, self.vel + amax * dt)
        self.prev = self.prev + v * dt
        self.vel = v
        return self.prev.copy()


def make_limiters(layout: str, clip: str = "v3", dtype=np.float64) -> dict:
    """float32 reproduces Isaac Lab bit for bit (validate_reach_io.py); the robot
    runs float64 -- the bang-bang limiter turns float32 rounding into ~1e-3 rad
    differences that persist, neither being more right than the other."""
    if clip == "v3":
        lo, hi = ARM_TARGET_LO, ARM_TARGET_HI
    else:
        lo, hi = ARM_DEFAULT - ARM_TARGET_CLIP_OLD, ARM_DEFAULT + ARM_TARGET_CLIP_OLD
    out = {"arm": RateLimiter(ARM_DEFAULT, ARM_SCALE, lo, hi, ARM_MAX_VELOCITY, ARM_MAX_ACCELERATION, dtype=dtype)}
    if layout == "waist":
        out["waist"] = RateLimiter([WAIST_DEFAULT[0]], WAIST_SCALE, [-WAIST_YAW_LIMIT], [WAIST_YAW_LIMIT],
                                   WAIST_MAX_VELOCITY, WAIST_MAX_ACCELERATION, brake=True, dtype=dtype)
    return out


# ─────────────────────────────────────────────
#  Policy
# ─────────────────────────────────────────────

class ReachPolicy:
    """An exported TorchScript reach policy (normalizer inside), CPU, batch 1."""

    def __init__(self, path: str = POLICY_DEFAULT):
        import torch
        self.torch = torch
        self.path = path
        self.net = torch.jit.load(path, map_location="cpu").eval()
        ws = [p for _, p in self.net.named_parameters() if p.dim() == 2]
        self.n_in, self.n_out = int(ws[0].shape[1]), int(ws[-1].shape[0])
        if self.n_in not in LAYOUTS:
            raise ValueError(f"{path}: {self.n_in} inputs -- not a reach layout {sorted(LAYOUTS)}")
        self.layout = LAYOUTS[self.n_in]
        want = 8 if self.layout == "waist" else 7
        if self.n_out != want:
            raise ValueError(f"{path}: {self.n_out} outputs, the {self.layout} layout has {want}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.as_tensor(np.asarray(obs, np.float32)).view(1, -1)
            return self.net(x)[0].numpy().astype(float)


# ─────────────────────────────────────────────
#  Virtual apple (no fruit in front of the robot)
# ─────────────────────────────────────────────

def virtual_touch_force(fr: Frames, apple: np.ndarray, radius: float) -> float:
    """Palm contact force [N] from a compliant apple (APPLE_CONTACT_STIFFNESS) at
    `apple` that is not really there: contact starts when its surface comes within
    PALM_TOUCH_HEIGHT of the palm reference point -- 1.2 N at the reach target
    (GRASP_STANDOFF), ~6 N one centimetre further. From ANY direction, as the
    training's palm sensor (palm_contact_force_any) reads any contact on the palm
    link: a virtual apple that only answered head-on let an off-angle palm run
    into it without the policy ever feeling a touch."""
    gap = float(np.linalg.norm(np.asarray(apple, float) - fr.palm_ref)) - radius
    return 0.0 if gap >= PALM_TOUCH_HEIGHT else APPLE_CONTACT_STIFFNESS * (PALM_TOUCH_HEIGHT - gap)


# ─────────────────────────────────────────────
#  Reachability (the training placement)
# ─────────────────────────────────────────────

def apple_from_grasp_target(target: np.ndarray, radius: float, approach_yaw: float = 0.0) -> np.ndarray:
    """Apple centre whose reach target (the palm reference point at the touch) is
    `target`: the palm comes in horizontally along `approach_yaw` (0 = +x, into
    the fruit wall), so the apple sits radius + GRASP_STANDOFF further on."""
    u = np.array([math.cos(approach_yaw), math.sin(approach_yaw), 0.0])
    return np.asarray(target, float) + (radius + GRASP_STANDOFF) * u


def grasp_target_from_apple(apple: np.ndarray, radius: float, approach_yaw: float = 0.0) -> np.ndarray:
    u = np.array([math.cos(approach_yaw), math.sin(approach_yaw), 0.0])
    return np.asarray(apple, float) - (radius + GRASP_STANDOFF) * u


#: the trunk the stems point at, from the right shoulder (AppleReachTreeWaist-v1 placement:
#: trunk_dist = REACH_WAIST_MAX + TRUNK_BEYOND_REACH along the envelope's centre azimuth)
TRUNK_FROM_SHOULDER = (0.44 + 0.21, -25.0)


def approach_yaw_default(apple_or_target: np.ndarray, shoulder: np.ndarray) -> float:
    """The palm's approach heading the training trees would give a fruit here: its
    stem points at the trunk (apple_tree.sample_tree heading_from), within
    +-STEM_HEADING_MAX_DEG of +x."""
    d, az = TRUNK_FROM_SHOULDER
    trunk = np.asarray(shoulder, float)[:2] + d * np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
    p = np.asarray(apple_or_target, float)[:2]
    h = math.atan2(trunk[1] - p[1], trunk[0] - p[0])
    lim = math.radians(STEM_HEADING_MAX_DEG)
    return max(-lim, min(lim, h))


def reach_check(target: np.ndarray, shoulder: np.ndarray, env: dict) -> dict:
    """Is the grasp target inside a training envelope and its probed safe-workspace
    map (envs/mdp/apple_tree.Envelope.contains, slack 0)? Positions in the pelvis
    frame; `shoulder` = right_shoulder_pitch_link at waist yaw 0."""
    rel = np.asarray(target, float) - np.asarray(shoulder, float)
    r, rxy = float(np.linalg.norm(rel)), float(np.linalg.norm(rel[:2]))
    az = math.degrees(math.atan2(rel[1], rel[0]))
    az_lo = env["az_center_deg"] - env["az_span_deg"] / 2
    az_hi = env["az_center_deg"] + env["az_span_deg"] / 2
    why = []
    if not env["reach_min"] <= r <= env["reach_max"]:
        why.append(f"distance {r:.3f} m outside {env['reach_min']:.2f}-{env['reach_max']:.2f}")
    if not env["dz_range"][0] <= rel[2] <= env["dz_range"][1]:
        why.append(f"height {rel[2]:+.3f} m outside {env['dz_range'][0]:+.2f}..{env['dz_range'][1]:+.2f}")
    if not az_lo <= az <= az_hi:
        why.append(f"azimuth {az:+.1f} deg outside {az_lo:+.1f}..{az_hi:+.1f}")
    if rel[0] < env["min_forward"]:
        why.append(f"forward {rel[0]:.3f} m < {env['min_forward']:.2f}")
    if rxy < env["min_horizontal"]:
        why.append(f"horizontal {rxy:.3f} m < {env['min_horizontal']:.2f}")
    in_map = None
    if env.get("map") and os.path.exists(env["map"]):
        m = np.load(env["map"])
        edges = (m["x_edges"], m["y_edges"], m["z_edges"])
        idx = [int(math.floor((v - e[0]) / (e[1] - e[0]))) for v, e in zip(rel, edges)]
        inside = all(0 <= i < n for i, n in zip(idx, m["feasible"].shape))
        in_map = bool(inside and m["feasible"][tuple(idx)])
        if not in_map:
            why.append("not a feasible cell of " + os.path.basename(env["map"]))
    return {"ok": not why, "why": why, "rel": rel, "r": r, "az_deg": az, "dz": float(rel[2]), "in_map": in_map}
