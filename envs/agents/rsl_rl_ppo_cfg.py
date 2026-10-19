# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
PPO (rsl_rl) runner config for AppleReach-v1 / AppleReach-Play-v1.

Save as:  envs/agents/rsl_rl_ppo_cfg.py   (complete replacement)

Changes from your previous version, with reasons:

  init_noise_std  1.0 -> 0.5
      With JointPositionAction(scale=1.0, use_default_offset=True) and no
      clip, a std of 1.0 meant the policy was sampling +-1 rad target jumps
      (+-3 rad in the tails) at 50 Hz. Now that scale=0.5 and clip=(-3,3) are
      set in ActionsCfg, 0.5 gives sane exploration without the flailing.

  entropy_coef    0.005 -> 0.002
      Lower entropy pressure means the converged policy keeps a tighter
      action distribution, which shows up directly as less residual jitter.
      Note this affects TRAINING only -- play.py uses
      runner.get_inference_policy(), which is the deterministic mean, so the
      buzzing you see at play time was never exploration noise. It is a
      closed-loop limit cycle between the policy and an underdamped PD, which
      is why the actuator damping change and the rate/jerk rewards are the
      real fix.

  num_steps_per_env  48 -> 64
      At 50 Hz, 48 steps is a 0.96 s rollout against a (now 5 s) episode.
      With 64 envs this gives 4096 samples/iteration and 1024 per minibatch,
      up from 1536/384 -- much less gradient noise per update.

  entropy_coef  0.002 -> 0.005 (modest bump, palm-touch retraining only)
      After adding the palm-touch goal (see apple_grasp_env_cfg.py's
      GRASP_OFFSET_Z/palm_touch_force/RewardsCfg.hold), an empirical trace
      of the resulting policy showed it making genuine contact once (at a
      violent 30-58 N, since fixed via penalty_excess_palm_force's max_excess
      cap and a stronger brake weight) and then retreating to a safe
      distance for the rest of the rollout, never approaching again. That's
      a learned-avoidance local optimum, not a training bug -- the
      DETERMINISTIC policy (play.py's inference mode) commits to hiding, so
      escaping it needs the STOCHASTIC training-time policy to occasionally
      re-sample a close approach despite the past bad experience. This is a
      smaller bump than the analogous fix needed for the grasp task's own
      "learned to do nothing" collapse (there: 0.002->0.01) since this is a
      narrower behavior gap (a converged approach policy needs to soften its
      final few cm, not discover contact from scratch) -- retune down once
      gentle contact is being found reliably.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AppleGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 100
    experiment_name = "apple_grasp_reach"
    run_name = ""
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AppleGraspGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for AppleGrasp-v1 / AppleGrasp-Play-v1.

    experiment_name MUST differ from "apple_grasp_reach" (it's
    "apple_grasp_grasp" here) — checkpoints for the two tasks must not land
    in the same logs/rsl_rl/<experiment_name>/ directory, since
    envs/mdp/grasp_autoplay.py's _find_last_reach_policy resolves the LATEST
    run under "apple_grasp_reach" specifically to drive the reach-autoplay
    hand-off, and would pick up a grasp-phase checkpoint by mistake if the
    two shared a directory.

    Wider hidden layers than the reach task's [128,128,64] — the grasp
    policy's observation (~65-dim: arm+hand pos/vel, apple/palm geometry,
    3 tactile features, commanded setpoint, 14-dim last-action) and action
    (14-dim) spaces are both substantially larger. Starting point, not a
    tuned value — retune from the training curve like every other number
    here.

    init_noise_std 0.5->0.8, entropy_coef 0.002->0.01: a first 500-iteration
    run at the reach task's original (already-converged-task-tuned) values
    collapsed to near-total inaction by iteration 500 (grasp_balance/hold/
    success never once nonzero, action_rate_weighted and posture_from_
    handoff shrinking toward zero, action noise std shrinking 0.43->0.25) --
    exploration was decaying faster than the policy ever discovered any
    grip-related reward to explore FOR. Raised alongside the reward-shaping
    fixes in grasp_rewards.py (finger_approach, partial-credit grasp_balance,
    contact-gated setpoint_tracking) so the policy has both more incentive
    and more room to actually explore hand-closing behavior before settling.
    Retune back down once grasping is reliably being discovered.
    """
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 100
    experiment_name = "apple_grasp_grasp"
    run_name = ""
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 -> 0.005: at 0.01 Policy/mean_noise_std kept CLIMBING (0.80 ->
        # 1.09 over 3000 iterations) once grasping was being found -- i.e.
        # 0.3+ rad of target noise per hand joint per step, which both
        # jittered the fingers and dropped latched grips. The collapse-to-
        # inaction this was raised against is no longer the failure mode.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class AppleGraspHarvestPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for AppleHarvest-v1 (own experiment folder,
    logs/rsl_rl/apple_grasp_harvest). 4 s episodes (200 steps), 7-dim delta
    arm actions. init_noise_std 0.5: with delta actions (0.025 rad/step per
    unit) the noise integrates into a slow random walk of the hand, which is
    all the exploration a held fruit tolerates. Starting values, retune from
    the curves."""
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 100
    experiment_name = "apple_grasp_harvest"
    run_name = ""
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AppleReachTreePPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachTree-v1: the v1 reach recipe, on the multi-apple tree. Own
    log folder (logs/rsl_rl/apple_reach_tree)."""
    experiment_name = "apple_reach_tree"


@configclass
class AppleReachTreeWaistPPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachTreeWaist-v1: the tree reach with the waist yaw. Own log
    folder (logs/rsl_rl/apple_reach_tree_waist)."""
    experiment_name = "apple_reach_tree_waist"


@configclass
class AppleReachTreeWaistFacePPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachTreeWaistFace-v1: the waist reach that faces the apple when
    it can. Own log folder (logs/rsl_rl/apple_reach_tree_waist_face)."""
    experiment_name = "apple_reach_tree_waist_face"


@configclass
class AppleReachSpindlePPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachSpindle-v1: the facing waist reach on the tall-spindle tree.
    Own log folder (logs/rsl_rl/apple_reach_spindle)."""
    experiment_name = "apple_reach_spindle"


@configclass
class AppleReachSpindleV2PPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachSpindle-v2: the spindle reach that keeps off the neighbours.
    Own log folder (logs/rsl_rl/apple_reach_spindle_v2)."""
    experiment_name = "apple_reach_spindle_v2"


@configclass
class AppleReachSpindleV3PPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachSpindle-v3: v2 + motor load (energy). Own log folder
    (logs/rsl_rl/apple_reach_spindle_v3)."""
    experiment_name = "apple_reach_spindle_v3"


@configclass
class AppleGraspTreePPORunnerCfg(AppleGraspGraspPPORunnerCfg):
    """AppleGraspTree-v1: the grasp recipe on the multi-apple tree with the
    hold-under-load stage. Own log folder (logs/rsl_rl/apple_grasp_tree)."""
    experiment_name = "apple_grasp_tree"


@configclass
class AppleGraspSpindlePPORunnerCfg(AppleGraspTreePPORunnerCfg):
    """AppleGraspSpindle-v1: the tree grasp after the waist reach, on the
    spindle tree. Own log folder (logs/rsl_rl/apple_grasp_spindle)."""
    experiment_name = "apple_grasp_spindle"


@configclass
class AppleGraspSpindleV2PPORunnerCfg(AppleGraspTreePPORunnerCfg):
    """AppleGraspSpindle-v2: stem-like twist load + harvest grip command.
    Own log folder (logs/rsl_rl/apple_grasp_spindle_v2)."""
    experiment_name = "apple_grasp_spindle_v2"


@configclass
class AppleArmGoalPPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleArmGoal-v1 (harvest-path goal reaching): the reach network size,
    512 envs x 32 steps per iteration. Own log folder (logs/rsl_rl/apple_arm_goal)."""
    experiment_name = "apple_arm_goal"
    num_steps_per_env = 32


@configclass
class AppleArmGoalWaistPPORunnerCfg(AppleArmGoalPPORunnerCfg):
    """AppleArmGoalWaist-v1: the harvest-path goal policy with the waist yaw.
    Own log folder (logs/rsl_rl/apple_arm_goal_waist)."""
    experiment_name = "apple_arm_goal_waist"


@configclass
class AppleGraspSpindleV3PPORunnerCfg(AppleGraspTreePPORunnerCfg):
    """AppleGraspSpindle-v3: v2 + the pull after an unbroken twist.
    Own log folder (logs/rsl_rl/apple_grasp_spindle_v3)."""
    experiment_name = "apple_grasp_spindle_v3"


@configclass
class AppleReachPregraspPPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachPregrasp-v1: two-stage reach, part 1 (home -> in front of the
    apple, fingers open). Own log folder (logs/rsl_rl/apple_reach_pregrasp)."""
    experiment_name = "apple_reach_pregrasp"
    save_interval = 25


@configclass
class AppleReachFinalPPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachFinal-v1: two-stage reach, part 2 (pregrasp -> gentle palm
    touch, cupped hand). Own log folder (logs/rsl_rl/apple_reach_final)."""
    experiment_name = "apple_reach_final"
    save_interval = 25


@configclass
class AppleReachPregraspV2PPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachPregrasp-v2: part 1 with the smoothness / settle rewards.
    Own log folder (logs/rsl_rl/apple_reach_pregrasp_v2); the chained part-2
    training picks its newest checkpoint from there."""
    experiment_name = "apple_reach_pregrasp_v2"
    max_iterations = 600
    save_interval = 25


@configclass
class AppleReachFinalChainedPPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachFinalChained-v1: part 2 trained behind the frozen part 1.
    Own log folder (logs/rsl_rl/apple_reach_final_chained)."""
    experiment_name = "apple_reach_final_chained"
    max_iterations = 600
    save_interval = 25


@configclass
class AppleReachPregraspV3PPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachPregrasp-v3: v2 + target push + rise near the body first.
    Own log folder (logs/rsl_rl/apple_reach_pregrasp_v3)."""
    experiment_name = "apple_reach_pregrasp_v3"
    max_iterations = 600
    save_interval = 25


@configclass
class AppleReachFinalChainedV2PPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachFinalChained-v2: tiered touch, behind the frozen v3 part 1.
    Own log folder (logs/rsl_rl/apple_reach_final_chained_v2)."""
    experiment_name = "apple_reach_final_chained_v2"
    max_iterations = 600
    save_interval = 25


@configclass
class AppleReachSinglePPORunnerCfg(AppleGraspPPORunnerCfg):
    """AppleReachSingle-v1: one reach policy, no touch input, palm forward,
    fingers vertical, scripted hand close. Own log folder
    (logs/rsl_rl/apple_reach_single_v1)."""
    experiment_name = "apple_reach_single_v1"
    max_iterations = 3000
    save_interval = 50


@configclass
class AppleGraspSinglePPORunnerCfg(AppleGraspTreePPORunnerCfg):
    """AppleGraspSingle-v1: the spindle v3 grasp after the single no-touch
    reach, arm free before the grip. Own log folder
    (logs/rsl_rl/apple_grasp_single_v1)."""
    experiment_name = "apple_grasp_single_v1"


@configclass
class AppleGraspSingleV2PPORunnerCfg(AppleGraspSinglePPORunnerCfg):
    """AppleGraspSingle-v2: the v1 grasp finishing the approach by feel
    (per-place tactile + memory + arm action inputs, realistic pads, new
    rewards). Own log folder (logs/rsl_rl/apple_grasp_single_v2)."""
    experiment_name = "apple_grasp_single_v2"


@configclass
class AppleReachSingleV2PPORunnerCfg(AppleReachSinglePPORunnerCfg):
    """AppleReachSingle-v2: the apple enclosed in the hand, tilted palm,
    distance hand trigger. Own log folder (logs/rsl_rl/apple_reach_single_v2)."""
    experiment_name = "apple_reach_single_v2"


@configclass
class AppleReachSingleTactilePPORunnerCfg(AppleReachSingleV2PPORunnerCfg):
    """AppleReachSingleTactile-v1: v2 with the palm touch input and the spindle
    reaches' touch rewards (a teacher candidate). Own log folder
    (logs/rsl_rl/apple_reach_single_tactile)."""
    experiment_name = "apple_reach_single_tactile"


@configclass
class AppleReachVisiblePPORunnerCfg(AppleReachSingleV2PPORunnerCfg):
    """AppleReachVisible-v1: apples the head camera sees, waist held at a
    fixed yaw, staging-point approach; trained from scratch. Own log folder
    (logs/rsl_rl/apple_reach_visible)."""
    experiment_name = "apple_reach_visible"


@configclass
class AppleReachVisibleV2PPORunnerCfg(AppleReachVisiblePPORunnerCfg):
    """AppleReachVisible-v2: v1 + camera tracking, wide starts, command delay.
    Same log folder as v1 (logs/rsl_rl/apple_reach_visible), so it warm-starts
    from a v1 run with --resume --load_run <v1 run>; tell the runs apart with
    --run_name v2."""
    run_name = "v2"


@configclass
class AppleGraspVisiblePPORunnerCfg(AppleGraspSingleV2PPORunnerCfg):
    """AppleGraspVisible-v1: the tactile grasp (single v2 layout) behind the
    frozen AppleReachVisible-v1 reach. Own log folder
    (logs/rsl_rl/apple_grasp_visible); warm start warmstart_spindle_v3/model_0.pt."""
    experiment_name = "apple_grasp_visible"


@configclass
class AppleGraspVisibleV2PPORunnerCfg(AppleGraspVisiblePPORunnerCfg):
    """AppleGraspVisible-v2: entropy 0.005 -> 0.002 (v1's noise std grew
    0.68 -> 0.88 while the rewards were flat). Same log folder as v1
    (logs/rsl_rl/apple_grasp_visible), so it starts from the same
    warmstart_spindle_v3/model_0.pt; tell the runs apart with --run_name v2."""
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AppleGraspSingleV3PPORunnerCfg(AppleGraspSingleV2PPORunnerCfg):
    """AppleGraspSingle-v3: grasp only, behind AppleReachSingle-v2. Own log
    folder (logs/rsl_rl/apple_grasp_single_v3)."""
    experiment_name = "apple_grasp_single_v3"


@configclass
class AppleArmPosePPORunnerCfg(AppleArmGoalPPORunnerCfg):
    """AppleArmPose-v1: point-to-point arm policy (no tree, no home), from
    scratch. Wider network than the reach (a general "where is the arm able
    to go without hitting the body" map), 1024 envs x 32 steps. Own log
    folder (logs/rsl_rl/apple_arm_pose)."""
    experiment_name = "apple_arm_pose"
    max_iterations = 4000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )


@configclass
class AppleArmPoseV2PPORunnerCfg(AppleArmPosePPORunnerCfg):
    """AppleArmPose-v2: v1 + the hold phase, fine-tuned from v1 (same inputs /
    outputs; warm start logs/rsl_rl/apple_arm_pose_v2/warmstart_v1/model_0.pt
    from v1's exported policy, scripts/warmstart_from_jit.py). Own log folder
    (logs/rsl_rl/apple_arm_pose_v2)."""
    experiment_name = "apple_arm_pose_v2"
    max_iterations = 1500
    # fine-tune: entropy 0.005 -> 0.001. Measured 2026-10-19: from v1's export
    # at std 0.3 / entropy 0.005, even the UNCHANGED v1 task fell apart in 200
    # its (goals reached 65% -> 18%, std growing 0.30 -> 0.36); warm start at
    # std 0.15 (warmstart_from_jit.py --std 0.15)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
