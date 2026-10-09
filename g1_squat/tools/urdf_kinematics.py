"""Minimal numpy forward kinematics + centre of mass for a URDF tree.

Only used offline (design_squat_pose.py) — no Isaac Sim needed. Handles
revolute/continuous/fixed joints, which is all the G1 description uses.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def _rpy_to_rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _tf(rot, pos):
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def rot_y(angle):
    return _axis_angle([0, 1, 0], angle)


class UrdfKinematics:
    def __init__(self, urdf_path: str, root: str = "pelvis"):
        tree = ET.parse(urdf_path).getroot()
        self.root = root
        self.mass: dict[str, float] = {}
        self.com_local: dict[str, np.ndarray] = {}
        for link in tree.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            origin = inertial.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            self.mass[link.get("name")] = float(inertial.find("mass").get("value"))
            self.com_local[link.get("name")] = xyz
        self.joints = {}  # child link -> joint info
        self.limits: dict[str, tuple[float, float]] = {}
        for joint in tree.findall("joint"):
            if joint.get("type") == "floating":
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_el = joint.find("axis")
            axis = np.fromstring(axis_el.get("xyz"), sep=" ") if axis_el is not None else np.array([1.0, 0, 0])
            info = dict(
                name=joint.get("name"),
                type=joint.get("type"),
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                origin=_tf(_rpy_to_rot(rpy), xyz),
                axis=axis,
            )
            self.joints[info["child"]] = info
            limit = joint.find("limit")
            if limit is not None and info["type"] == "revolute":
                self.limits[info["name"]] = (float(limit.get("lower")), float(limit.get("upper")))
        # topological order from the root
        children: dict[str, list[str]] = {}
        for child, info in self.joints.items():
            children.setdefault(info["parent"], []).append(child)
        self.order = []
        stack = [root]
        while stack:
            link = stack.pop()
            self.order.append(link)
            stack.extend(children.get(link, []))
        self.total_mass = sum(self.mass.get(link, 0.0) for link in self.order)

    def fk(self, q: dict[str, float], root_tf: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """World transform of every link. Joints missing from `q` are at 0."""
        tfs = {self.root: np.eye(4) if root_tf is None else root_tf}
        for link in self.order[1:]:
            info = self.joints[link]
            t = tfs[info["parent"]] @ info["origin"]
            if info["type"] in ("revolute", "continuous"):
                t = t @ _tf(_axis_angle(info["axis"], q.get(info["name"], 0.0)), np.zeros(3))
            tfs[link] = t
        return tfs

    def com(self, tfs: dict[str, np.ndarray]) -> np.ndarray:
        acc = np.zeros(3)
        for link, m in self.mass.items():
            if link in tfs:
                acc += m * (tfs[link][:3, :3] @ self.com_local[link] + tfs[link][:3, 3])
        return acc / self.total_mass

    def subtree_links(self, link: str) -> list[str]:
        out, stack = [], [link]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(c for c, info in self.joints.items() if info["parent"] == cur)
        return out

    def gravity_torque(self, tfs, joint_child_link: str, axis_world: np.ndarray, support_side: str = "below"):
        """Static gravity torque about a joint, from the mass on the free side.

        For a standing robot the stance leg carries everything ABOVE the joint, so for leg
        joints pass the complement (everything not in the subtree below the joint) via
        `support_side="below"`. Returns the torque magnitude about `axis_world` (N*m) for
        the whole robot weight split equally over two legs is NOT applied here.
        """
        info = self.joints[joint_child_link]
        p_joint = tfs[joint_child_link][:3, 3]
        below = set(self.subtree_links(joint_child_link))
        links = [lk for lk in self.mass if lk in tfs and (lk not in below if support_side == "below" else lk in below)]
        torque = np.zeros(3)
        for lk in links:
            c = tfs[lk][:3, :3] @ self.com_local[lk] + tfs[lk][:3, 3]
            torque += np.cross(c - p_joint, np.array([0, 0, -9.81 * self.mass[lk]]))
        return float(torque @ axis_world)
