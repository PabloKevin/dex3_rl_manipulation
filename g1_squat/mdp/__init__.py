"""MDP terms for the G1 squat task (Isaac Lab built-ins + squat-specific terms)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide  # noqa: F401

from .actions import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
