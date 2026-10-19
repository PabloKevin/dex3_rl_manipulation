"""np_fk.py -- forward kinematics of the frames reach_io needs, in numpy, straight from the URDF.

reach_io.G1Kinematics uses pinocchio; this is the same computation for the four frames the reach
policy's observation needs (palm, torso, right shoulder, head camera), for a Jetson without
pinocchio. Fixed pelvis, chains of revolute / fixed joints: T = T_origin(xyz, rpy) . Rot(axis, q).
sim2real/onboard/check_np_fk.py compares it with pinocchio on random poses. Python 3.8, stdlib xml.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, q):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(q) * K + (1.0 - math.cos(q)) * (K @ K)


class UrdfChains:
    """frames(names, q_by_joint) -> {link: (R [3,3], p [3])} in the root link's frame."""

    def __init__(self, urdf: str, links):
        root = ET.parse(urdf).getroot()
        by_child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0]
            by_child[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "R0": _rpy(*rpy), "p0": np.array(xyz), "axis": axis}
        self.chains = {}
        for link in links:
            chain, l = [], link
            while l in by_child:
                chain.append(by_child[l])
                l = by_child[l]["parent"]
            self.chains[link] = list(reversed(chain))
            self.root = l

    def pose(self, link: str, q: dict):
        R, p = np.eye(3), np.zeros(3)
        for j in self.chains[link]:
            Rj = j["R0"]
            if j["type"] in ("revolute", "continuous"):
                Rj = Rj @ _axis_angle(j["axis"], float(q.get(j["name"], 0.0)))
            p = p + R @ j["p0"]
            R = R @ Rj
        return R, p


class NumpyKinematics:
    """Drop-in for reach_io.G1Kinematics.frames() (the frames the reach observation uses)."""

    def __init__(self, urdf: str, R):
        self.R = R
        self.names = dict(R.G1Kinematics.FRAMES)
        self.chains = UrdfChains(urdf, list(self.names.values()))

    def frames(self, waist3, arm7, named=None):
        R = self.R
        q = {n: float(v) for n, v in zip(R.WAIST_JOINTS, waist3)}
        q.update({n: float(v) for n, v in zip(R.RIGHT_ARM_JOINTS, arm7)})
        X = {k: self.chains.pose(v, q) for k, v in self.names.items()}
        pR, pp = X["palm"]
        return R.Frames(palm_pos=pp, palm_R=pR, palm_ref=pp + pR @ R.PALM_LOCAL_OFFSET,
                        palm_normal=pR @ R.PALM_NORMAL_AXIS, torso_pos=X["torso"][1], torso_R=X["torso"][0],
                        shoulder_pos=X["shoulder"][1], camera_pos=X["camera"][1], camera_R=X["camera"][0])
