# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
PPO (rsl_rl) runner config for AppleGrasp-v0 / AppleGrasp-Play-v0.

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
class AppleGraspGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for AppleGraspGrasp-v0 / AppleGraspGrasp-Play-v0.

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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )