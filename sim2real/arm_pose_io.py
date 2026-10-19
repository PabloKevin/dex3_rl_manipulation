# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
The point-to-point arm policy's input/output (AppleArmPose-v1, envs/apple_arm_pose_env_cfg.py,
envs/mdp/arm_pose.py) WITHOUT Isaac Lab, and the waypoint plan of scripts/play_arm_pose_reach.py:
what the laptop toy (toy_arm_pose.py) and the onboard runner (onboard/rl_runner.py) run on the robot.
numpy only (+ reach_io's kinematics), Python 3.8. sim2real/validate_arm_pose_io.py checks it against
Isaac Lab number by number.

Goal = the GRASP POCKET (POCKET in the palm frame: where an enclosed apple's centre sits) at a point,
the palm normal (palm +y) along a direction, optionally the fingers (palm +x) along one; "via" goals
are flown through. Observation (35), in the env's term order:
  arm q - default (7), arm dq (7), goal - pocket (torso frame, 3), goal - right shoulder (torso frame, 3),
  goal palm normal in the palm frame (3), goal finger direction in the palm frame (3, zero when free),
  flags (finger counts, via) (2), previous raw action (7).
Action (7): reach_io's arm action (RateLimiter: 1.2 rad/s, 5 rad/s^2, braking, the soft-limit clip).

Plan (play_arm_pose_reach.make_plan), all with the palm normal along the horizontal line from the
torso to the apple:
  staging  at the apple goal's height, stage_radius from the torso horizontally (pulled in to stay
           min_stage_back short of the goal), outside the cluster: the hand rises / lowers there;
  apple    the pocket on the apple centre, moved apple_offset along the approach (negative = short).
Reached as in training (TerminationsCfg.goal_tracker): pocket within 2 cm, normal within 15 deg
(fingers 25 deg when they count), palm slower than 8 cm/s, for 0.2 s; a via goal at 3 cm / 25 deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import reach_io as R

#: envs/apple_grasp_grasp_env_cfg.GRASP_CENTER_LOCAL: the grasp pocket in the palm frame
POCKET = np.array([0.073, 0.057, 0.0])
PALM_NORMAL = np.array([0.0, 1.0, 0.0])
PALM_FINGER = np.array([1.0, 0.0, 0.0])
N_IN, N_OUT = 35, 7
#: the goal tracker of the env (track_goal params)
TRACK = dict(pos_tol=0.02, ang_tol_deg=15.0, finger_tol_deg=25.0, speed_tol=0.08, hold_time=0.2,
             via_pos_tol=0.03, via_ang_tol_deg=25.0)
#: play_arm_pose_reach.py defaults (stage_radius: the dashboard's field, 0.24 by default on the robot)
STAGE_RADIUS = 0.24
MIN_STAGE_BACK = 0.08
WAYPOINT_TIMEOUT_S = 4.0
#: the STAGING point is only there to make the approach (or the way back) better: no hold there. The next
#: goal comes once the pocket is within STAGE_ADVANCE_D and the palm normal within STAGE_ADVANCE_DEG
#: (robot runs 2026-10-19: the pocket was within 5 cm at 2.6-3.2 s with the palm still 125-134 deg off,
#: within 30 deg too at 6.5-6.7 s, then held to the 13 s timeout), or after STAGE_ADVANCE_S / speed
STAGE_ADVANCE_D = 0.05
STAGE_ADVANCE_DEG = 30.0
STAGE_ADVANCE_S = 2.4
#: the hand during the arm-pose policy: the reach's cup (the policy trained with it)
HAND_CUP = dict(R.REACH_HAND_PRESHAPE)


def use_constants() -> None:
    """reach_io's limiter / defaults for this policy: AppleReachSingle-v2's arm action (1.2 rad/s,
    5 rad/s^2, braking), the v3 soft-limit clip, waist pitch 0."""
    R.use_layout_constants("arm_pose")


@dataclass
class Goal:
    name: str
    target: np.ndarray          # pocket goal, pelvis frame
    normal: np.ndarray          # palm normal goal (unit)
    finger: np.ndarray          # finger direction goal (unit)
    finger_on: bool = False
    via: bool = False
    #: pass to the next goal once within advance_d (m) and advance_deg (palm normal), or after advance_s
    advance_d: float = 0.0
    advance_deg: float = 180.0
    advance_s: float = 0.0
    #: s before the next goal comes anyway (0 = the caller's waypoint timeout)
    timeout_s: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "target": np.asarray(self.target, float).tolist(),
                "normal": np.asarray(self.normal, float).tolist(), "finger": np.asarray(self.finger, float).tolist(),
                "finger_on": bool(self.finger_on), "via": bool(self.via), "advance_d": float(self.advance_d),
                "advance_deg": float(self.advance_deg), "advance_s": float(self.advance_s),
                "timeout_s": float(self.timeout_s)}

    @staticmethod
    def from_dict(d: dict) -> "Goal":
        return Goal(str(d["name"]), np.asarray(d["target"], float), np.asarray(d["normal"], float),
                    np.asarray(d["finger"], float), bool(d.get("finger_on", False)), bool(d.get("via", False)),
                    float(d.get("advance_d", 0.0)), float(d.get("advance_deg", 180.0)), float(d.get("advance_s", 0.0)),
                    float(d.get("timeout_s", 0.0)))


def hand_frame(fr: "R.Frames"):
    """(pocket, palm normal, finger direction) in the pelvis frame."""
    return fr.palm_pos + fr.palm_R @ POCKET, fr.palm_R @ PALM_NORMAL, fr.palm_R @ PALM_FINGER


def build_obs(q, dq, fr: "R.Frames", g: Goal, prev_raw) -> np.ndarray:
    pocket, _, _ = hand_frame(fr)
    Rt, Rp = fr.torso_R, fr.palm_R
    parts = [
        np.asarray(q, float) - R.ARM_DEFAULT,
        np.asarray(dq, float),
        Rt.T @ (g.target - pocket),
        Rt.T @ (g.target - fr.shoulder_pos),
        Rp.T @ g.normal,
        (Rp.T @ g.finger) * (1.0 if g.finger_on else 0.0),
        [1.0 if g.finger_on else 0.0, 1.0 if g.via else 0.0],
        np.asarray(prev_raw, float)[:7],
    ]
    return np.concatenate([np.asarray(p, float).ravel() for p in parts]).astype(np.float32)


def errors(fr: "R.Frames", g: Goal):
    """(pocket distance, normal angle, finger angle)."""
    pocket, normal, finger = hand_frame(fr)
    d = float(np.linalg.norm(g.target - pocket))
    ang = math.acos(float(np.clip(np.dot(g.normal, normal), -1.0, 1.0)))
    fang = math.acos(float(np.clip(np.dot(g.finger, finger), -1.0, 1.0)))
    return d, ang, fang


class Tracker:
    """track_goal for one arm: True on the step a goal is reached. Palm speed: the palm origin's motion per step."""

    def __init__(self, dt: float = R.DT, **kw):
        self.dt = float(dt)
        self.c = dict(TRACK, **kw)
        self.reset()

    def reset(self):
        self.hold, self.arrived, self.prev_palm = 0.0, False, None

    def step(self, fr: "R.Frames", g: Goal) -> bool:
        c = self.c
        d, ang, fang = errors(fr, g)
        v = 0.0 if self.prev_palm is None else float(np.linalg.norm(fr.palm_pos - self.prev_palm)) / self.dt
        self.prev_palm = fr.palm_pos.copy()
        fin_ok = (not g.finger_on) or fang < math.radians(c["finger_tol_deg"])
        if g.via:
            return d < c["via_pos_tol"] and ang < math.radians(c["via_ang_tol_deg"]) and fin_ok
        stop_ok = d < c["pos_tol"] and ang < math.radians(c["ang_tol_deg"]) and fin_ok and v < c["speed_tol"]
        self.hold = self.hold + self.dt if stop_ok else 0.0
        if self.hold >= c["hold_time"] and not self.arrived:
            self.arrived = True
            return True
        return False


def goal_event(g: Goal, trk: Tracker, fr: "R.Frames", t_wp: float, timeout_s: float) -> str:
    """This step's outcome for goal g, t_wp s after it was set: "" (go on), "passed" (an advance goal is
    close enough), "timer" (its advance time is up), "reached" (the trained stop test), "timeout"."""
    d, ang, _ = errors(fr, g)
    reached = trk.step(fr, g)
    if g.advance_d > 0.0 and d < g.advance_d and ang < math.radians(g.advance_deg):
        return "passed"
    if g.advance_s > 0.0 and t_wp >= g.advance_s:
        return "timer"
    if reached:
        return "reached"
    if t_wp >= (g.timeout_s or timeout_s):
        return "timeout"
    return ""


def set_stage_advance(g: Goal, speed: float, d: float = STAGE_ADVANCE_D, deg: float = STAGE_ADVANCE_DEG,
                      s: float = STAGE_ADVANCE_S) -> Goal:
    g.advance_d, g.advance_deg, g.advance_s = float(d), float(deg), float(s) / max(float(speed), 1e-3)
    return g


def make_return_plan(approach, home: Goal, speed: float):
    """Back the way it came: the approach's staging point (same advance rule), then home (a stop goal)."""
    st = approach[0]
    stage = set_stage_advance(Goal("staging back", st.target.copy(), st.normal.copy(), st.finger.copy(),
                                   st.finger_on, st.via), speed, st.advance_d or STAGE_ADVANCE_D,
                              st.advance_deg if st.advance_d else STAGE_ADVANCE_DEG,
                              st.advance_s * speed if st.advance_s else STAGE_ADVANCE_S)
    return [stage, Goal("home", home.target.copy(), home.normal.copy(), home.finger.copy(), True, False)]


def make_plan(fr: "R.Frames", apple: np.ndarray, stage_radius: float = STAGE_RADIUS,
              min_stage_back: float = MIN_STAGE_BACK, apple_offset: float = 0.0, stage_via: bool = False,
              fingers: str = "free"):
    """[Goal staging, Goal apple] (pelvis frame) and a one-line description."""
    _, _, finger_now = hand_frame(fr)
    axis = np.asarray(fr.torso_pos, float)
    u = np.asarray(apple, float) - axis
    u[2] = 0.0
    d_apple = float(np.linalg.norm(u))
    u = u / max(d_apple, 1e-6)
    goal = np.asarray(apple, float) + apple_offset * u
    d_goal = float(np.dot(goal - axis, u))
    r = float(stage_radius)
    note = ""
    if r > d_goal - min_stage_back:
        r = d_goal - min_stage_back
        note = f" (pulled in: the apple is only {d_goal:.2f} m out)"
    stage = axis + r * u
    stage[2] = goal[2]
    fon = fingers == "home"
    plan = [Goal("staging", stage, u.copy(), finger_now.copy(), fon, bool(stage_via)),
            Goal("apple", goal, u.copy(), finger_now.copy(), fon, False)]
    desc = (f"apple {d_apple:.2f} m from the torso axis, {float(goal[2] - axis[2]):+.2f} m vs the torso; "
            f"staging {r:.2f} m out{note}")
    return plan, desc


def home_goal(fr: "R.Frames") -> Goal:
    pocket, normal, finger = hand_frame(fr)
    return Goal("home", pocket.copy(), normal.copy(), finger.copy(), True, False)
