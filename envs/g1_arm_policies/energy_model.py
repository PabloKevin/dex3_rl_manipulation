"""
energy_model.py
---------------
Identified per-joint electrical power model for the Unitree G1 left arm.

  P_net = Σ_i [ a_i·τ_i·q̇_i  +  b_i·Δτ²_i  +  c_i·|q̇_i|  +  d_i·q̇²_i ]
        + Σ_{i<j} e_ij·|q̇_i|·|q̇_j|

  Δτ²_i = τ²_i − τ²_{0,i}   (change in copper loss vs. static baseline at q=0)

Joint ordering (7 joints):
  0: shoulder_pitch  1: shoulder_roll  2: shoulder_yaw  3: elbow
  4: wrist_roll      5: wrist_pitch    6: wrist_yaw

The fitted model returns net power relative to the static baseline. For RL
costs, use the positive-cost helpers so negative net-power transients are not
turned into reward.
"""

from __future__ import annotations

import numpy as np
import yaml
from dataclasses import dataclass, field


# Joint ordering: shoulder_pitch(0), roll(1), yaw(2), elbow(3),
#                 wrist_roll(4), wrist_pitch(5), wrist_yaw(6)
_N = 7
_PAIRS = [                                          # C(7,2) = 21 pairs
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
    (1,2),(1,3),(1,4),(1,5),(1,6),
    (2,3),(2,4),(2,5),(2,6),
    (3,4),(3,5),(3,6),
    (4,5),(4,6),
    (5,6),
]


@dataclass
class EnergyParams:
    """Identified 7-joint model parameters."""
    a: np.ndarray = field(default_factory=lambda: np.ones(_N))
    b: np.ndarray = field(default_factory=lambda: np.array(
        [0.34253, 0.35198, 0.16880, 0.48991, 0.0, 0.0, 0.0]))   # wrists TBD
    c: np.ndarray = field(default_factory=lambda: np.zeros(_N))
    d: np.ndarray = field(default_factory=lambda: np.array(
        [0.84708, 0.27267, 0.07894, 0.12622, 0.0, 0.0, 0.0]))   # wrists TBD
    e: np.ndarray = field(default_factory=lambda: np.zeros(21))  # 21 pairs; fill from MATLAB
    tau0_sq: np.ndarray = field(default_factory=lambda: np.zeros(_N))

    def __post_init__(self) -> None:
        for k in ("a", "b", "c", "d", "e", "tau0_sq"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64))

    @classmethod
    def from_dict(cls, d: dict) -> "EnergyParams":
        return cls(
            a=np.asarray(d.get("a", np.ones(_N))),
            b=np.asarray(d["b"]),
            c=np.asarray(d.get("c", np.zeros(_N))),
            d=np.asarray(d["d"]),
            e=np.asarray(d.get("e", np.zeros(21))),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EnergyParams":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f)["energy_model"])


class EnergyModel:
    """Vectorised electrical power model with numpy backend."""

    PAIRS = _PAIRS

    def __init__(self, params: EnergyParams | None = None) -> None:
        self.p = params or EnergyParams()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_baseline_torques(self, tau0: np.ndarray) -> None:
        """Store τ²₀ from gravity compensation at q=0 (call once after URDF load)."""
        self.p.tau0_sq = np.asarray(tau0, dtype=np.float64) ** 2

    # ── Single-step power ─────────────────────────────────────────────────────

    def power(self, tau: np.ndarray, dq: np.ndarray) -> float:
        """
        Instantaneous net electrical power [W].

        Args:
            tau: active-joint torques [Nm], shape (7,)
            dq:  active-joint velocities [rad/s], shape (7,)
        Returns:
            P_net [W]  (positive = consuming more than baseline)
        """
        tau = np.asarray(tau, dtype=np.float64)
        dq  = np.asarray(dq,  dtype=np.float64)
        dtau2 = tau**2 - self.p.tau0_sq

        P = float(
            (self.p.a * tau * dq
             + self.p.b * dtau2
             + self.p.c * np.abs(dq)
             + self.p.d * dq**2).sum()
        )
        abs_dq = np.abs(dq)
        for k, (i, j) in enumerate(self.PAIRS):
            P += float(self.p.e[k]) * float(abs_dq[i]) * float(abs_dq[j])
        return P

    def power_cost(self, tau: np.ndarray, dq: np.ndarray) -> float:
        """Positive power cost [W] for RL rewards and energy accounting."""
        return max(0.0, self.power(tau, dq))

    # ── Batched power (trajectory) ────────────────────────────────────────────

    def power_batch(self, tau: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Vectorised power for a whole trajectory.

        Args:
            tau: shape (T, 7)
            dq:  shape (T, 7)
        Returns:
            P_net: shape (T,)
        """
        dtau2 = tau**2 - self.p.tau0_sq[None, :]
        P = (
            self.p.a[None, :] * tau * dq
            + self.p.b[None, :] * dtau2
            + self.p.c[None, :] * np.abs(dq)
            + self.p.d[None, :] * dq**2
        ).sum(axis=1)
        abs_dq = np.abs(dq)
        for k, (i, j) in enumerate(self.PAIRS):
            P += self.p.e[k] * abs_dq[:, i] * abs_dq[:, j]
        return P

    def power_cost_batch(self, tau: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Vectorised positive power cost [W]."""
        return np.maximum(self.power_batch(tau, dq), 0.0)

    def energy(self, tau: np.ndarray, dq: np.ndarray, dt: float) -> float:
        """Total positive energy cost [J] over a trajectory."""
        P = self.power_cost_batch(
            np.asarray(tau, dtype=np.float64),
            np.asarray(dq,  dtype=np.float64),
        )
        return float(np.trapz(P, dx=dt))

    def net_energy(self, tau: np.ndarray, dq: np.ndarray, dt: float) -> float:
        """Total net model energy [J], including negative transients."""
        P = self.power_batch(
            np.asarray(tau, dtype=np.float64),
            np.asarray(dq,  dtype=np.float64),
        )
        return float(np.trapz(P, dx=dt))
