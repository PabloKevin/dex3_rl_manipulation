"""
g1_arm_env.py - Hybrid Constellation Reward
Combines exponential constellation reward with improvement shaping and residual position penalty.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

try:
    import sys as _sys
    import importlib as _importlib
    _cmeel_pin = (
        "/home/nahuel/.local/lib/python3.10/site-packages"
        "/cmeel.prefix/lib/python3.10/site-packages"
    )
    try:
        _sys.path.remove(_cmeel_pin)
    except ValueError:
        pass
    _sys.path.insert(0, _cmeel_pin)
    _importlib.invalidate_caches()
    _cached = _sys.modules.get("pinocchio")
    if _cached is not None and not hasattr(_cached, "buildModelFromUrdf"):
        del _sys.modules["pinocchio"]

    import pinocchio as pin
    if not hasattr(pin, "buildModelFromUrdf"):
        raise ImportError("Robotics Pinocchio not found.")
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False
    warnings.warn("pinocchio not found.", stacklevel=2)

from energy_model import EnergyModel, EnergyParams


class G1ArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config_path: str = "config.yaml",
        render_mode: Optional[str] = None,
        use_her: bool = False,
        target_curriculum: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.render_mode = render_mode
        self.use_her = use_her
        self.dt = float(self.cfg["episode"]["dt"])
        self.max_steps = int(self.cfg["episode"]["max_steps"])
        self.n_active = int(self.cfg["urdf"]["n_active"])

        jl = self.cfg["joint_limits"]
        self.q_lo = np.asarray(jl["lower"], dtype=np.float64)
        self.q_hi = np.asarray(jl["upper"], dtype=np.float64)
        self.q_range = self.q_hi - self.q_lo
        self.q_center = 0.5 * (self.q_lo + self.q_hi)
        self.dq_max = float(jl["max_velocity"])

        # Workspace-biased target sampling (apple-picker workspace bbox)
        ts = self.cfg.get("target_sampling", {})
        self._target_ws_enabled = bool(ts.get("enabled", False))
        wb = ts.get("workspace_bbox", {})
        margin = float(ts.get("margin", 0.0))
        wb_x = wb.get("x", [-np.inf, np.inf])
        wb_y = wb.get("y", [-np.inf, np.inf])
        wb_z = wb.get("z", [-np.inf, np.inf])
        self._target_ws_lo = np.array([wb_x[0], wb_y[0], wb_z[0]], dtype=np.float64) - margin
        self._target_ws_hi = np.array([wb_x[1], wb_y[1], wb_z[1]], dtype=np.float64) + margin
        self._target_ws_offset = np.asarray(ts.get("frame_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        self._target_ws_p = float(ts.get("p_workspace", 0.0))
        self._target_ws_max_tries = int(ts.get("max_tries", 50))

        self.pos_tol = float(self.cfg["goal"]["pos_tolerance"])
        self.ori_tol = float(self.cfg["goal"]["ori_tolerance"])

        rw = self.cfg["reward"]
        
        # Constellation reward parameters
        self.constellation_radius = float(rw.get("constellation_radius", 0.06))
        self.n_constellation_points = int(rw.get("constellation_points", 4))
        self.w_con = float(rw.get("w_con", 2.5))
        self.delta_con = float(rw.get("delta_con", 5.0))
        self.beta_residual = float(rw.get("beta_residual", 0.1))
        
        # Regularisation
        self.R_success = float(rw["R_success"])
        self.step_cost = float(rw["step_cost"])
        self.lambda_smooth = float(rw.get("lambda_smooth", 0.0))
        self.lambda_tau = float(rw.get("lambda_tau", 0.0))

        # Torque-limit penalty: penalises squared excess above per-joint actuator limits
        self.lambda_tau_limit = float(rw.get("lambda_tau_limit", 0.0))
        _tlim = rw.get("tau_limits", None)
        self.tau_limits = (np.array(_tlim, dtype=np.float64) if _tlim is not None
                           else np.full(self.n_active, np.inf))

        # Per-joint acceleration penalty (global)
        self.lambda_dq = float(rw.get("lambda_dq", 0.0))

        # Per-joint jerk penalty: penalises ddq_t - ddq_{t-1} = dq - 2*prev_dq + prev_prev_dq
        self.lambda_jerk = float(rw.get("lambda_jerk", 0.0))

        # Near-goal braking: penalises ||dq||^2 * (1 - dist/brake_dist)^2
        # to force deceleration as EE approaches the target
        self.lambda_brake = float(rw.get("lambda_brake", 0.0))
        self.brake_dist   = float(rw.get("brake_dist",   0.15))

        # Per-joint weights for acceleration penalties (lambda_smooth, lambda_dq, lambda_brake).
        # Falls back to legacy dq_weights key if acc_weights is not set.
        _acc_w = rw.get("acc_weights", rw.get("dq_weights", None))
        self.acc_weights = (np.array(_acc_w, dtype=np.float64) if _acc_w is not None
                            else np.ones(self.n_active, dtype=np.float64))

        # Per-joint weights for jerk penalty (lambda_jerk).
        # Falls back to acc_weights if jerk_weights is not set.
        _jerk_w = rw.get("jerk_weights", None)
        self.jerk_weights = (np.array(_jerk_w, dtype=np.float64) if _jerk_w is not None
                             else self.acc_weights.copy())

        # Hard acceleration constraint: max velocity change per step (rad/s). 0 = disabled.
        self.max_delta_dq = float(rw.get("max_delta_dq", 0.0))

        # Hold-to-succeed: EE must stay within neighbourhood for hold_steps consecutive steps.
        hold_cfg = self.cfg.get("hold", {})
        self.hold_time     = float(hold_cfg.get("time", 0.0))
        self.hold_pos_tol  = float(hold_cfg.get("pos_tolerance", self.pos_tol))
        self.hold_ori_tol  = float(hold_cfg.get("ori_tolerance", self.ori_tol))
        self._hold_steps_required = max(1, int(round(self.hold_time / self.dt)))
        self.R_hold_step   = float(rw.get("R_hold_step", 0.0))

        # Disabled terms
        self.terminal_failure_penalty = float(rw.get("terminal_failure_penalty", 0.0))

        # Electrical energy penalty: reward -= lambda_elec * P_elec each step
        self.lambda_elec = float(rw.get("lambda_elec", 0.0))

        # Cartesian alignment reward
        self.lambda_align = float(rw.get("lambda_align", 0.0))
        self.align_dist_threshold = float(rw.get("align_dist_threshold", self.pos_tol))

        # Joint excursion penalty: penalises weighted ||q||² toward homing pose (q=0)
        self.lambda_excursion = float(rw.get("lambda_excursion", 0.0))
        _exc_w = rw.get("excursion_weights", None)
        self.excursion_weights = (np.array(_exc_w, dtype=np.float64) if _exc_w is not None
                                  else np.ones(self.n_active, dtype=np.float64))

        # Orientation improvement shaping: explicit gradient on ori_dist reduction
        self.lambda_ori = float(rw.get("lambda_ori", 0.0))

        # IK pool config (loaded after Pinocchio model is built)
        ik_cfg = self.cfg.get("ik_pool", {})
        self._ik_pool_enabled = bool(ik_cfg.get("enabled", False))
        self._ik_pool_mix     = float(ik_cfg.get("mix", 1.0))
        self._ik_pool_path    = ik_cfg.get("path", "data/ik_pool_workspace.npz")
        self._ik_pool_data    = None   # populated below after model is built

        # Hard-case pool: jittered neighborhoods around previously-failed targets.
        # Checked before the base ik_pool — hardcase_mix fraction of episodes
        # sample from here, the rest fall through to the base pool/sampler.
        hc_cfg = self.cfg.get("hardcase_pool", {})
        self._hardcase_pool_enabled = bool(hc_cfg.get("enabled", False))
        self._hardcase_pool_mix     = float(hc_cfg.get("mix", 0.0))
        self._hardcase_pool_path    = hc_cfg.get("path", "data/ik_pool_hardcases.npz")
        self._hardcase_pool_data    = None

        # PD gains
        self.kp_pos = 15.0
        self.kd_damp = 0.0
        self.max_step = self.dq_max / self.kp_pos

        ctrl = self.cfg.get("control", {})
        self.kp = float(ctrl.get("kp", 400.0))
        self.kd = float(ctrl.get("kd", 12.0))

        self._build_pin_model()
        ep = EnergyParams.from_dict(self.cfg["energy_model"])
        self.energy_model = EnergyModel(ep)
        self._set_baseline_torques()
        """if self._ik_pool_enabled:
            self._load_ik_pool(self._ik_pool_path)
        if self._hardcase_pool_enabled:
            self._hardcase_pool_data = self._load_pool_file(self._hardcase_pool_path)"""

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_active,), dtype=np.float32)

        if self.use_her:
            core_dim = self.n_active * 2 + 1
            goal_dim = 3 + 9
            self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(-np.inf, np.inf, (core_dim,), np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
                desired_goal=spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
            ))
        else:
            obs_dim = self.n_active * 2 + 3 + 3 + 1
            self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self.q = np.zeros(self.n_active)
        self.dq = np.zeros(self.n_active)
        self._prev_dq      = np.zeros(self.n_active)
        self._prev_prev_dq = np.zeros(self.n_active)
        self._step_count = 0
        self._hold_counter = 0
        self._target_pos = np.zeros(3)
        self._target_rot = np.eye(3)
        self._prev_d_con = 0.0

        # Initialise constellation points
        self._init_constellation()

        curriculum = target_curriculum or {}
        self._curriculum_enabled = bool(curriculum.get("enabled", False))
        self._curriculum_start_scale = float(curriculum.get("start_scale", 1.0))
        self._curriculum_end_scale = float(curriculum.get("end_scale", 1.0))
        self._curriculum_progress = 0.0 if self._curriculum_enabled else 1.0

        self._ep_tau = []
        self._ep_dq = []
        self._ep_q = []

    def _init_constellation(self):
        angles = np.linspace(0, 2*np.pi, self.n_constellation_points, endpoint=False)
        self.constellation_points = self.constellation_radius * np.array([
            [np.cos(a), np.sin(a), 0.0] for a in angles
        ], dtype=np.float64)

    def _constellation_distance(self) -> float:
        pos_ee, rot_ee = self._fk(self.q)
        rot_ee_mat = rot_ee if isinstance(rot_ee, np.ndarray) else np.array(rot_ee)
        current_pts = pos_ee + self.constellation_points @ rot_ee_mat.T
        target_pts = self._target_pos + self.constellation_points @ self._target_rot.T
        diff = current_pts - target_pts
        return float(np.mean(np.sum(diff**2, axis=1)))

    def _constellation_reward(self) -> float:
        return np.exp(-self.w_con * self._constellation_distance())

    # ---------- Pinocchio methods (unchanged from working version) ----------
    def _build_pin_model(self) -> None:
        if not _PIN_AVAILABLE:
            raise RuntimeError("pinocchio required")
        urdf_cfg = self.cfg["urdf"]
        urdf_path = urdf_cfg["path"]
        arm_joints = urdf_cfg["arm_joint_names"]
        ee_frame = urdf_cfg["ee_frame"]

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        full_model = pin.buildModelFromUrdf(urdf_path)
        keep_ids = set()
        for name in arm_joints:
            jid = full_model.getJointId(name)
            if jid == full_model.njoints:
                raise ValueError(f"Joint '{name}' not found")
            keep_ids.add(jid)

        lock_ids = [jid for jid in range(1, full_model.njoints) if jid not in keep_ids]
        ref_q = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, ref_q)
        self.data = self.model.createData()

        if self.model.existFrame(ee_frame):
            self.ee_frame_id = self.model.getFrameId(ee_frame)
        else:
            self.ee_frame_id = self.model.getFrameId(self.model.frames[-1].name)
            warnings.warn(f"EE frame '{ee_frame}' not found, using {self.model.frames[-1].name}")

    def _set_baseline_torques(self) -> None:
        if not _PIN_AVAILABLE:
            return
        q0 = pin.neutral(self.model)
        tau_full = pin.computeGeneralizedGravity(self.model, self.data, q0)
        tau0_active = np.array(tau_full[: self.n_active])
        self.energy_model.set_baseline_torques(tau0_active)

    """def _load_pool_file(self, path: str) -> dict:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Target pool not found: {path}\n"
                f"Run:  python3 build_ik_pool.py --out {path}"
            )
        d = np.load(path)
        return {
            "q":   d["q"].astype(np.float64),          # (N, n_active)
            "pos": d["pos"].astype(np.float64),        # (N, 3)
            "rot": d["rot"].astype(np.float64),        # (N, 3, 3)
        }

    def _load_ik_pool(self, path: str) -> None:
        self._ik_pool_data = self._load_pool_file(path)"""

    @staticmethod
    def _sample_target_from_pool_data(pool: dict, rng) -> tuple:
        idx = int(rng.integers(len(pool["q"])))
        return pool["pos"][idx].copy(), pool["rot"][idx].copy(), pool["q"][idx].copy()

    def _sample_target_from_pool(self, rng) -> tuple:
        return self._sample_target_from_pool_data(self._ik_pool_data, rng)

    def _translational_jacobian(self) -> np.ndarray:
        """3 × n_active translational Jacobian of the EE frame, world-aligned."""
        q = self._full_q(self.q)
        pin.computeJointJacobians(self.model, self.data, q)
        J6 = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.array(J6[:3, : self.n_active])

    def _full_q(self, q_active):
        q = np.zeros(self.model.nq)
        q[: self.n_active] = q_active
        return q

    def _fk(self, q_active):
        q = self._full_q(q_active)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        tf = self.data.oMf[self.ee_frame_id]
        return np.array(tf.translation), np.array(tf.rotation)

    def _rnea_torques(self, q_active, dq_active, ddq_active):
        q = self._full_q(q_active)
        dq = np.concatenate([dq_active, np.zeros(self.model.nq - self.n_active)])
        ddq = np.concatenate([ddq_active, np.zeros(self.model.nq - self.n_active)])
        tau_full = pin.rnea(self.model, self.data, q, dq, ddq)
        return np.array(tau_full[: self.n_active])

    @staticmethod
    def _ori_error_rotvec(R_cur, R_tgt):
        R_err = R_tgt @ R_cur.T
        return pin.log3(R_err)

    def _ee_pos_error(self):
        pos, rot = self._fk(self.q)
        pos_err = self._target_pos - pos
        ori_err = self._ori_error_rotvec(rot, self._target_rot)
        pos_dist = float(np.linalg.norm(pos_err))
        ori_dist = float(np.linalg.norm(ori_err))
        return pos_err, pos_dist, ori_err, ori_dist

    def _norm_q(self):
        return 2.0 * (self.q - self.q_center) / self.q_range

    def _pose_vec(self, pos, rot):
        return np.concatenate([pos, rot.flatten()]).astype(np.float32)

    def _obs(self, pos_err, ori_err):
        q_norm = self._norm_q()
        dq_norm = self.dq / self.dq_max
        t_rem = 1.0 - self._step_count / self.max_steps
        if self.use_her:
            pos, rot = self._fk(self.q)
            return dict(
                observation=np.concatenate([q_norm, dq_norm, [t_rem]]).astype(np.float32),
                achieved_goal=self._pose_vec(pos, rot),
                desired_goal=self._pose_vec(self._target_pos, self._target_rot),
            )
        return np.concatenate([q_norm, dq_norm, pos_err / 0.5, ori_err / np.pi, [t_rem]]).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        pos_dist = np.linalg.norm(achieved_goal[..., :3] - desired_goal[..., :3], axis=-1)
        R_ach = achieved_goal[..., 3:].reshape(achieved_goal.shape[:-1] + (3, 3))
        R_des = desired_goal[..., 3:].reshape(desired_goal.shape[:-1] + (3, 3))
        R_err = np.matmul(R_des, R_ach.swapaxes(-1, -2))
        eye = np.eye(3)
        ori_dist = np.linalg.norm(R_err - eye, axis=(-2, -1)) / np.sqrt(2)
        success = (pos_dist < self.pos_tol) & (ori_dist < self.ori_tol)
        return success.astype(float)

    def _in_workspace(self, pos):
        p = np.asarray(pos, dtype=np.float64) - self._target_ws_offset
        return bool(np.all(p >= self._target_ws_lo) and np.all(p <= self._target_ws_hi))

    def _sample_target(self, rng):
        # Hard-case pool: jittered neighborhoods around previously-failed targets.
        # Checked first so hardcase_mix is an absolute fraction of all episodes.
        if (self._hardcase_pool_enabled
                and self._hardcase_pool_data is not None
                and rng.random() < self._hardcase_pool_mix):
            return self._sample_target_from_pool_data(self._hardcase_pool_data, rng)

        # IK-pool sampling: draw from precomputed workspace-IK solutions
        if (self._ik_pool_enabled
                and self._ik_pool_data is not None
                and rng.random() < self._ik_pool_mix):
            return self._sample_target_from_pool(rng)

        use_ws = self._target_ws_enabled and (rng.random() < self._target_ws_p)
        max_tries = self._target_ws_max_tries if use_ws else 1
        fallback = None
        tries = 0
        while True:
            if self._curriculum_enabled:
                scale = self._curriculum_start_scale + self._curriculum_progress * (self._curriculum_end_scale - self._curriculum_start_scale)
                scale = float(np.clip(scale, 0.0, 1.0))
                half_span = 0.5 * self.q_range * scale
                lo = np.maximum(self.q_lo, -half_span)
                hi = np.minimum(self.q_hi, half_span)
                q_rand = rng.uniform(lo, hi)
            else:
                q_rand = rng.uniform(self.q_lo, self.q_hi)
            pos, rot = self._fk(q_rand)
            if np.linalg.norm(pos) <= 0.15:
                continue
            if not use_ws or self._in_workspace(pos):
                return pos.copy(), rot.copy(), q_rand.copy()
            if fallback is None:
                fallback = (pos.copy(), rot.copy(), q_rand.copy())
            tries += 1
            if tries >= max_tries:
                return fallback

    def set_curriculum_progress(self, progress):
        self._curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def set_pos_tol(self, pos_tol: float, hold_pos_tol: float | None = None) -> None:
        """Tighten position tolerance at runtime (used by ToleranceCurriculumCallback)."""
        self.pos_tol = float(pos_tol)
        self.hold_pos_tol = float(hold_pos_tol) if hold_pos_tol is not None else self.pos_tol

    def set_ori_tol(self, ori_tol: float, hold_ori_tol: float | None = None) -> None:
        """Tighten orientation tolerance at runtime (used by ToleranceCurriculumCallback)."""
        self.ori_tol = float(ori_tol)
        self.hold_ori_tol = float(hold_ori_tol) if hold_ori_tol is not None else self.ori_tol

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed=self.np_random.integers(0, 2**31))

        init_scale = 0.3
        half_range  = 0.5 * self.q_range * init_scale
        q_lo_init   = np.maximum(self.q_lo, self.q_center - half_range)
        q_hi_init   = np.minimum(self.q_hi, self.q_center + half_range)
        for _ in range(50):
            q_candidate = rng.uniform(q_lo_init, q_hi_init).astype(np.float64)
            _init_pos, _ = self._fk(q_candidate)
            if np.linalg.norm(_init_pos) > 0.15 and _init_pos[0] >= -0.05:
                self.q = q_candidate
                break
        else:
            self.q = self.q_center.copy().astype(np.float64)
        self.dq = np.zeros(self.n_active)
        self._prev_dq      = np.zeros(self.n_active)
        self._prev_prev_dq = np.zeros(self.n_active)
        self._step_count = 0
        self._hold_counter = 0

        if options is not None and "target_pos" in options:
            self._target_pos = np.asarray(options["target_pos"])
            self._target_rot = np.asarray(options.get("target_rot", np.eye(3)))
            self._target_q = None
        else:
            self._target_pos, self._target_rot, self._target_q = self._sample_target(rng)

        # Initialise distances for improvement shaping
        self._prev_d_con = self._constellation_distance()
        _, _, _, _ori = self._ee_pos_error()
        self._prev_ori_dist = _ori

        self._ep_tau = []
        self._ep_dq = []
        self._ep_q = []

        pos_err, _, ori_err, _ = self._ee_pos_error()
        obs = self._obs(pos_err, ori_err)
        return obs, {"target_pos": self._target_pos.copy()}

    def _step_dynamics(self, action):
        """Apply action under the kinematic model. Updates self.q, self.dq, self._step_count.
        Override in subclasses to replace with physics simulation. Returns tau (n_active,)."""
        target_q = self.q + np.asarray(action, dtype=np.float64) * self.max_step
        target_q = np.clip(target_q, self.q_lo, self.q_hi)
        dq_desired = self.kp_pos * (target_q - self.q) - self.kd_damp * self.dq
        dq_cmd = np.clip(dq_desired, -self.dq_max, self.dq_max)
        if self.max_delta_dq > 0.0:
            dq_cmd = np.clip(dq_cmd, self.dq - self.max_delta_dq, self.dq + self.max_delta_dq)
        q_new = np.clip(self.q + dq_cmd * self.dt, self.q_lo, self.q_hi)
        dq_actual = (q_new - self.q) / self.dt
        ddq = (dq_actual - self.dq) / self.dt
        tau = self._rnea_torques(self.q, self.dq, ddq)
        self.q = q_new
        self.dq = dq_actual
        self._step_count += 1
        return tau

    def step(self, action):
        tau = self._step_dynamics(action)

        self._ep_q.append(self.q.copy())
        self._ep_dq.append(self.dq.copy())
        self._ep_tau.append(tau.copy())

        pos_err, pos_dist, ori_err, ori_dist = self._ee_pos_error()

        # Constellation distance
        d_con = self._constellation_distance()

        P_mech = np.sum(tau * self.dq)
        P_elec = self.energy_model.power(tau, self.dq)
        P_cost_mech = max(0.0, P_mech)
        P_cost_elec = max(0.0, P_elec)

        if self.use_her:
            success = (pos_dist < self.pos_tol) and (ori_dist < self.ori_tol)
            reward = 1.0 if success else 0.0
            terminated = success
            truncated = self._step_count >= self.max_steps and not success
        else:
            reward = 0.0

            # 1. Improvement shaping on constellation distance
            reward += self.delta_con * (self._prev_d_con - d_con)
            self._prev_d_con = d_con

            # 2. Residual linear position penalty (ensures gradient far away)
            reward -= self.beta_residual * pos_dist

            # 3. Exponential constellation reward (strong signal near goal)
            reward += self._constellation_reward()

            # 4. Cartesian alignment: reward EE velocity component toward target
            if self.lambda_align > 0.0 and pos_dist > self.align_dist_threshold:
                J_pos = self._translational_jacobian()   # (3, n_active)
                v_ee  = J_pos @ self.dq                  # (3,) world-frame EE vel
                reward += self.lambda_align * float(np.dot(v_ee, pos_err / pos_dist))

            # 5. Regularisation
            reward -= self.step_cost
            if self.lambda_elec > 0.0:
                reward -= self.lambda_elec * P_cost_elec
            ddq = self.dq - self._prev_dq  # per-joint acceleration (rad/s²·dt)
            if self._step_count > 1 and self.lambda_smooth > 0.0:
                reward -= self.lambda_smooth * float(np.dot(self.acc_weights, ddq ** 2))
            if self.lambda_tau > 0.0:
                reward -= self.lambda_tau * float(np.sum(tau ** 2))

            # 6. Per-joint weighted acceleration penalty (global)
            if self._step_count > 1:
                ddq_sq_w = float(np.dot(self.acc_weights, ddq ** 2))
                if self.lambda_dq > 0.0:
                    reward -= self.lambda_dq * ddq_sq_w

                # 7. Near-goal braking on acceleration: same penalty gated by distance
                if self.lambda_brake > 0.0 and pos_dist < self.brake_dist:
                    brake_frac = 1.0 - pos_dist / self.brake_dist
                    reward -= self.lambda_brake * ddq_sq_w * brake_frac ** 2

            # 8. Per-joint jerk penalty: rate of change of acceleration
            if self._step_count > 2 and self.lambda_jerk > 0.0:
                jerk = ddq - (self._prev_dq - self._prev_prev_dq)
                reward -= self.lambda_jerk * float(np.dot(self.jerk_weights, jerk ** 2))

            # 9. Joint excursion penalty: penalises deviation from homing pose (q=0)
            if self.lambda_excursion > 0.0:
                reward -= self.lambda_excursion * float(
                    np.dot(self.excursion_weights, self.q ** 2)
                )

            # 10. Orientation improvement shaping: direct gradient on ori_dist
            if self.lambda_ori > 0.0:
                reward += self.lambda_ori * (self._prev_ori_dist - ori_dist)
            self._prev_ori_dist = ori_dist

            # 11. Torque-limit penalty: quadratic penalty on excess above per-joint limits
            if self.lambda_tau_limit > 0.0:
                excess = np.maximum(0.0, np.abs(tau) - self.tau_limits)
                reward -= self.lambda_tau_limit * float(np.dot(excess, excess))

            # Hold-to-succeed: count consecutive steps within neighbourhood
            in_neighbourhood = (pos_dist < self.hold_pos_tol) and (ori_dist < self.hold_ori_tol)
            if in_neighbourhood:
                self._hold_counter += 1
                if self.R_hold_step > 0.0:
                    reward += self.R_hold_step
            else:
                self._hold_counter = 0

            success = self._hold_counter >= self._hold_steps_required
            if success:
                reward += self.R_success
                terminated = True
            else:
                terminated = False

            truncated = self._step_count >= self.max_steps and not success
            if truncated and self.terminal_failure_penalty != 0.0:
                reward -= self.terminal_failure_penalty

        self._prev_prev_dq = self._prev_dq.copy()
        self._prev_dq      = self.dq.copy()

        obs = self._obs(pos_err, ori_err)
        info = {
            "pos_dist": pos_dist,
            "ori_dist": ori_dist,
            "constellation_dist": d_con,
            "P_mech": P_mech,
            "P_elec": P_elec,
            "P_cost_mech": P_cost_mech,
            "P_cost_elec": P_cost_elec,
            "success": success,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["episode_energy"] = self.episode_energy
            info["episode_net_energy"] = self.episode_net_energy

        return obs, float(reward), terminated, truncated, info

    @property
    def episode_energy(self):
        if not self._ep_tau:
            return 0.0
        tau = np.array(self._ep_tau)
        dq = np.array(self._ep_dq)
        return self.energy_model.energy(tau, dq, self.dt)

    @property
    def episode_net_energy(self):
        if not self._ep_tau:
            return 0.0
        tau = np.array(self._ep_tau)
        dq = np.array(self._ep_dq)
        return self.energy_model.net_energy(tau, dq, self.dt)

    @property
    def episode_trajectory(self):
        return {
            "q": np.array(self._ep_q),
            "dq": np.array(self._ep_dq),
            "tau": np.array(self._ep_tau),
        }

    def render(self):
        if self.render_mode == "human":
            pos, _ = self._fk(self.q)
            dist = np.linalg.norm(self._target_pos - pos)
            d_con = self._constellation_distance()
            print(f"  step={self._step_count:3d} | pos_err={dist*100:.1f}cm | d_con={d_con:.4f}m² | E={self.episode_energy:.1f}J")

    def close(self):
        pass