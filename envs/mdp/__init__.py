# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Task-specific MDP functions.

Save as:  envs/mdp/__init__.py   (complete replacement)

Import everything here so env_cfg can do:
    from . import mdp as task_mdp
    task_mdp.joint_pos_selected(...)

Reach-phase reward terms live in reach_rewards.py and are imported as a
module rather than flattened into this namespace, so the env cfg can do:
    from .mdp import reach_rewards as rr
    rr.reward_reach_progress(...)
That keeps the new reach terms visibly separate from the grasp/lift/place
terms in rewards.py, which are unchanged.
"""

from . import reach_rewards

from .actions import RateLimitedJointPositionAction, RateLimitedJointPositionActionCfg

from .observations import (
    joint_pos_selected,
    joint_vel_selected,
    joint_effort_selected,
    fingertip_contact_forces,
    fingertip_contact_normals,
    object_pos_world,
    object_pos_relative_to_body,
    object_pos_in_body_frame,
    object_vel_world,
    body_quat_relative_to_body,
    tactile_slip,
    tactile_finger_balance,
    tactile_force_level,
    target_grip_level,
    palm_contact_force,
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
    reward_palm_below_apple,
    reward_palm_orientation,
)

from .terminations_and_events import (
    apple_dropped,
    apple_lifted_success,
    apple_placed_success,
    reach_success,
    reset_apple_pose,
    grasp_success,
    fruit_damaged,
    apple_fell_from_grip,
    hold_default_joint_targets,
)

from .self_collision import (
    body_tree_contact_depth, penalty_body_tree_contact, penalty_self_collision, penalty_tree_contact,
    self_collision_depth, tree_contact_depth,
)

from .domain_randomization import (
    randomize_apple_scale,
    randomize_stem_break_params,
    randomize_tree_and_apple,
    randomize_tree_geometry,
    randomize_tactile_gain,
    randomize_tactile_footprint,
    randomize_fruit_damage_force,
    randomize_closure_threshold,
)