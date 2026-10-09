# G1 squat (separate from the apple-harvest project)

RL policy that takes the G1-29dof + Dex3 (floating base) from the velocity policy's stand
pose down into a deep flat-foot squat (to pick things up from the ground), holds it while the
arms move, and stands back up when the toggle is released — handing over to/from
unitree_rl_lab's velocity policy. Nothing here imports or modifies `envs/` or `scripts/`
(the apple-harvest code); it only reuses `unitree_rl_lab` (robot cfg, actuators) and Isaac Lab.

## Layout

```
g1_squat/
├── __init__.py            gym ids G1-Squat-v0 / G1-Squat-Play-v0
├── robot_cfg.py           G1 + Dex3 URDF, deployed PD gains, stand / squat / arm-rest poses
├── squat_env_cfg.py       scene, obs, actions, rewards, terminations, events
├── policies.py            Unitree v0 velocity policy loader (ONNX -> torch), squat actor loader
├── mdp/                   commands (toggle + arm motion), residual action, obs, rewards, terminations
├── agents/rsl_rl_ppo_cfg.py
├── scripts/
│   ├── train.py / play.py           (copies of the repo scripts, logs -> g1_squat/logs/)
│   ├── probe_reference.py           zero-action check of the reference (no policy)
│   └── switch_test.py               velocity -> squat -> hold (arms moving) -> stand -> velocity
├── tools/
│   ├── urdf_kinematics.py           numpy FK + CoM from the URDF
│   └── design_squat_pose.py         offline search of the balanced squat pose
└── train_watchdog.sh       training with time limit + clean Isaac shutdown
```

## Design

* **Command** — one toggle (`target` 0 stand / 1 squat). A phase `s` ramps toward it in 2 s;
  the policy sees `(target, s)`. On the robot this is one button + the same ramp in the FSM.
* **Action** — 15 joints (legs + waist), residual around a reference:
  `q_target = q_ref(s) + 0.25 * a`, `q_ref = lerp(stand, squat, smoothstep(s))`.
  At `s = 0` this is exactly the velocity policy's convention (default pose + 0.25 a).
* **Squat pose** (`tools/design_squat_pose.py`, FK of the Dex3 URDF): hip −2.32, knee 2.30,
  ankle −0.75, waist pitch 0.15 rad → pelvis 0.353 m, torso lean ≈ 53°, whole-body CoM over
  the middle of the feet. The palm grasp point (palm + 8 cm) reaches 10 cm above the ground at
  the toes with ~6 cm to spare; fingertips reach the ground. Deeper is limited by ankle
  dorsiflexion (−0.87 rad hard limit). The knees end up ~9 cm ahead of the toes, so the arms
  reach down outside the knees.
* **Arms** are never controlled by the policy. `ArmMotionCommand` holds them at the rest pose
  while going down/up; once the squat has settled it moves them between random poses (reaching
  forward/down/sideways, one or both arms) with optional hand loads (≤ 6 N per hand, kept
  through the stand-up = object picked). Amplitude/loads ramp in over the first ~1500 iterations.
* **Rewards** — pelvis-height and posture tracking of the reference, whole-body CoM over the
  centre of the feet, still stand pose when standing (hand-over), feet planted + flat + no slip,
  no sideways tilt, symmetry, smoothness, motors kept below 80 % of their torque limit,
  −200 on a fall. Falls = pelvis < 0.20 m, knee/torso/pelvis near the ground, or excessive tilt.
* **Robustness** — friction 0.4–1.0, torso mass −1..+3 kg and CoM ±3 cm, leg/waist PD gains
  ±10–15 %, pushes every 4–8 s, observation noise (as unitree_rl_lab).

## Things found while building it

* **PD gains**: the deployed velocity policy (`deploy/.../velocity/v0`) runs with waist roll/pitch
  kp 200 and arm kd 10, while unitree_rl_lab's current sim cfg has kp 40 / kd 1. With the sim cfg
  gains v0 falls when walking in Isaac; with the deployed gains it walks (0/16 falls, 4.8° peak
  tilt, Dex3 model). The squat uses the deployed gains, so nothing changes at the hand-over.
* **Arm rest pose**: Unitree's default arms (shoulder roll ±0.25) put the Dex3 thumbs on the
  hip-pitch links from stand to mid-squat. The squat holds shoulder roll ±0.40 (clear over the
  whole transition); observations still use Unitree's default pose as offset.
* **Waist pitch is the weak motor**: static torque 13.7 N·m in the squat, ~25 N·m (its limit)
  with both arms reaching far forward. The policy is penalised near motor limits so it should
  straighten the torso when the arms extend; the arm controller should still avoid long
  two-arm forward reaches with heavy objects while squatting.

## Run

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python   # from the repo root
$PY -u g1_squat/scripts/probe_reference.py --headless          # sanity check (no policy)
nohup g1_squat/train_watchdog.sh > /dev/null 2>&1 &             # full training (ITERS=4000 ~ 3.9 h)
tail -f g1_squat/logs/train_squat.status
$PY -u g1_squat/scripts/play.py --num_envs 16                   # watch + export ONNX/JIT
$PY -u g1_squat/scripts/switch_test.py --num_envs 16            # hand-over test (add --headless)
$PY -u g1_squat/scripts/switch_test.py --vel_only               # only the velocity policy
```

Use `python -u`: Isaac Sim hard-exits on close and loses buffered stdout.

## Status (2026-10-09)

Validation run (300 iterations, `logs/rsl_rl/g1_squat/2026-10-09_12-15-55`, ~18 min): 95 % of
training episodes reach the 20 s time-out, pelvis-height error 4.4 mm, ~2.7 squat cycles/episode.
Hand-over test (16 envs): arms still → 0/16 falls (pelvis 0.351 m, foot slip 0.3 cm, stand
pose within 0.06 rad at hand-back); full random arm motion + loads → 3/16 falls (this run only
reached ~10 % of the arm-motion curriculum). Full run (4000 it) not launched yet.

## Deploy (next step, not done yet)

A new FSM state in `unitree_rl_lab/deploy` (C++): button toggles `target`, the state ramps `s`
at 1/2 s⁻¹, observation = the same 6 terms × 5 history with `squat_command = (target, s)`
instead of `velocity_commands`, action → 15 leg/waist joints as `q_ref(s) + 0.25 a`, arms held
by their own controller (or at the rest pose). Transitions: Velocity → Squat at any time while
standing; Squat → Velocity only when `target = 0`, `s = 0` and the robot is still.
