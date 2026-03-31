# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Task-specific MDP functions.

Import everything here so env_cfg can do:
    from . import mdp as task_mdp
    task_mdp.joint_pos_selected(...)
"""

from .observations import (
    joint_pos_selected,
    joint_vel_selected,
    joint_effort_selected,
    fingertip_contact_forces,
    fingertip_contact_normals,
    object_pos_world,
    object_pos_relative_to_body,
    object_vel_world,
)

from .rewards import (
    reward_reaching,
    reward_fingertip_contact,
    reward_grasp_quality,
    reward_lift,
    reward_hold,
    reward_place,
    penalty_joint_limits,
    penalty_action_rate,
)

from .terminations_and_events import (
    apple_dropped,
    apple_lifted_success,
    apple_placed_success,
    reset_apple_pose,
)
