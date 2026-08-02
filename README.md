# G1 Dex3 Apple Grasp

Isaac Lab 4.5 project: Unitree G1-29dof + Dex3 right hand, fixed base,  
picks up a red sphere ("apple") and lifts it. Trains with PPO, transfers to MuJoCo.

---

## Repo structure

```
dex3_rl_manipulation/
├── envs/
│   ├── __init__.py                 ← gymnasium task registration
│   ├── apple_grasp_env_cfg.py      ← scene + managers (MAIN CONFIG)
│   └── mdp/
│       ├── observations.py         ← custom obs functions
│       ├── rewards.py              ← legacy grasp/lift reward functions
│       ├── reach_rewards.py        ← reach-phase reward terms (current)
│       ├── actions.py              ← RateLimitedJointPositionAction
│       ├── domain_randomization.py ← per-episode tree + fruit placement
│       ├── eval_metrics.py         ← EvalProbe: per-episode metric recorder
│       └── terminations_and_events.py
├── scripts/
│   ├── run_env.py                  ← Phase 1: view the scene (+ proximity stop)
│   ├── move_arm.py                 ← Phase 1: scripted reach + grasp
│   ├── proximity_stop.py           ← quintic minimum-jerk arrival supervisor
│   ├── train.py / play.py          ← Phase 2: PPO training / playback
│   ├── evaluate_policy.py          ← batch evaluation → results/*.npz
│   ├── eval_utils.py               ← metric math + npz IO (numpy only)
│   ├── plot_results.py             ← paper figures + results macros
│   ├── plot_style.py               ← IEEE figure style
│   └── render_scene.py             ← scene stills for the paper
├── paper/                          ← IEEE write-up (Overleaf bundle)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Ubuntu 22.04 (distrobox or native)
- Isaac Lab 4.5 conda env active: `conda activate isaaclab`
- `unitree_sim_isaaclab` cloned and assets downloaded

### 2. Set the robot USD path

Either edit `apple_grasp_env_cfg.py` directly:

```python
G1_DEX3_USD_PATH = "/absolute/path/to/your/g1_29dof_dex3.usd"
```

Or set an environment variable (preferred):

```bash
export G1_DEX3_USD_PATH="/path/to/unitree_sim_isaaclab/usd/g1_29dof_dex3/g1.usd"
```

Use your preferred URDF (without hand cameras) — just convert it to USD if needed:

```bash
# Inside Isaac Lab env:
python -c "
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
cfg = UrdfConverterCfg(asset_path='/path/to/g1.urdf', usd_dir='/path/to/output/', fix_base=True)
UrdfConverter(cfg)
"
```

### 3. Clone this repo next to unitree_sim_isaaclab

```bash
cd ~/Code
# this repo is already here — just cd into it
cd g1_dex3_apple
```

---

## Phase 1 — Validate the environment

### Step 1: View the scene

```bash
python scripts/run_env.py
```

You should see: G1 robot standing (fixed), red sphere in front-right of the arm, ground, lights.

**If the robot doesn't appear:** check `G1_DEX3_USD_PATH`.  
**If the sphere position looks wrong:** edit `APPLE_INIT_POS` in `apple_grasp_env_cfg.py`.

### Step 2: Print joint and link names

```bash
# Robot joint names (for actuator patterns + action space):
python scripts/run_env.py --debug_joints

# Robot link (body) names (for contact sensor path + camera prim_path):
python scripts/run_env.py --debug_links
```

Then update these TODOs in `apple_grasp_env_cfg.py`:

| TODO | What to set |
|------|-------------|
| `G1_DEX3_USD_PATH` | Path to your USD file |
| `HEAD_CAMERA_LINK` | Head link name from `--debug_links` |
| `right_hand` actuator `joint_names_expr` | Dex3 joint pattern from `--debug_joints` |
| `contact_forces` `prim_path` regex | Finger link name pattern from `--debug_links` |
| `HAND_CLOSE` in `move_arm.py` | Values that fully close your hand |

### Step 3: Run scripted motion

```bash
python scripts/move_arm.py --slow --verbose
```

Watch the LIFT stage. Check the printed `apple_z`:
- **Increases by ~0.1 m** → grasp working ✓ → proceed to Phase 2
- **Stays flat** → hand not gripping → tune `HAND_CLOSE` targets or Dex3 `stiffness`
- **Apple falls** → fingers pushing apple away → adjust `ARM_REACH` targets

---

## Phase 2 — RL training (PPO via rsl_rl)

One-time setup (in the `isaaclab_curobo` conda env):
```bash
pip install rsl-rl-lib==3.0.1
```

```bash
# Train (32 envs, ~6h on RTX 4060). Logs land in logs/rsl_rl/apple_grasp_reach/<timestamp>/:
python scripts/train.py --task AppleGrasp-v0 --num_envs 32 --headless

# Watch training progress:
tensorboard --logdir logs/rsl_rl/apple_grasp_reach

# Watch the trained policy (auto-picks the latest run/checkpoint, or pass --checkpoint <path>):
python scripts/play.py --task AppleGrasp-Play-v0 --num_envs 1
```

PPO hyperparameters live in `envs/agents/rsl_rl_ppo_cfg.py`
(`AppleGraspPPORunnerCfg`); the reward/termination/domain-randomization setup
described above is unchanged for training — only `envs/__init__.py` gains an
`rsl_rl_cfg_entry_point` pointing at that config.

---

## Smooth motion: two mechanisms at different levels

Smoothness is enforced structurally rather than left to reward shaping, in two
places that do different jobs.

### 1. `RateLimitedJointPositionAction` — bounds motion *during* transit

`envs/mdp/actions.py`. A drop-in replacement for Isaac Lab's
`JointPositionActionCfg` that treats the policy's affine-mapped target as a
*desired* target and integrates a rate-limited command toward it, hard-capping
commanded velocity (`max_velocity`, default 2.0 rad/s) and commanded
acceleration (`max_acceleration`, default 15 rad/s²) every control step.

The bound holds for *any* policy output, including an untrained one, so
commanded jerk is analytically capped at `2·max_acceleration/dt` = 1500 rad/s³.
Measured over 727 evaluation episodes, commanded velocity and acceleration hit
exactly 2.000 rad/s and 15.000 rad/s² and never exceed them.

> **This is load-bearing, not cosmetic.** Re-running the same trained policy
> with the limits raised to 1e9 (`evaluate_policy.py --rate_limit off`) drops
> the success rate from 48.4% to **0.0%** and raises joint-jerk RMS by ~560×.
> The policy learns to emit targets that get *integrated*, not tracked
> directly; remove the integrator and you have changed the plant.

### 2. `proximity_stop.py` — brings the arm to rest *on arrival*

The rate limiter says nothing about what happens once the hand is there. A
policy trained with a dense proximity reward has no incentive to stop moving —
measured here, episodes that reach the tolerance leave it a median of **3
times** and spend only ~67% of their post-arrival steps inside it. That
residual jitter happens right next to an object the hand can damage, and in
deployment there is no episode boundary to hide it.

`QuinticStopController` is a supervisor layered over the trained policy:

| stage | behaviour |
|---|---|
| `IDLE` | policy drives the arm; palm-to-fruit distance is monitored |
| → `STOPPING` | armed once distance ≤ `stop_dist` (9 cm) held for `dwell_time` (0.6 s) |
| `STOPPING` | quintic minimum-jerk profile over `stop_duration` (0.4 s) brings the arm to rest |
| `HOLDING` | holds the final target |
| → `IDLE` | only after distance clears `stop_dist + resume_margin` (hysteresis) |

The quintic is solved for `q(0)=q₀, q̇(0)=v₀, q̈(0)=0` and
`q(T)=q₁, q̇(T)=q̈(T)=0`, with `q₀, v₀` measured at the trigger instant. The
stopping point is *derived* as `q₁ = q₀ + v₀T/2` — the least displacement
consistent with a jerk-limited stop in time `T` — rather than fixed in advance.

Two details worth keeping if you modify it:

- **Matching `q̇(0)=v₀` exactly is the whole point.** Simply zeroing the joint
  targets on arrival is a velocity discontinuity, i.e. precisely the jerk
  everything else here is built to avoid. `q̈(0)=0` is deliberately *not*
  matched: measured joint acceleration at the trigger is noisy and only affects
  the interior shape of the blend, not its endpoints.
- **Hysteresis applies on the way in, not just out.** Small residual motion
  near the target was ticking a hair above `stop_dist` every few steps, resetting
  the dwell timer, so `dwell_time` could never accumulate and the stop never
  fired. The arm-and-release band is `stop_dist` → `stop_dist + resume_margin`
  in both directions.

Threading: the controller's background thread never touches PhysX/Isaac Sim
tensors. It runs a state machine and a clock against samples the main loop
pushes via `update_state()`, and returns a command the main loop reads with
`get_command()` and applies through `env.step()` as usual.

```bash
# enabled by default when running a policy in the interactive viewer
python scripts/run_env.py --policy <path/to/policy.pt>

# tune or disable it
python scripts/run_env.py --policy <...> --stop_dist 0.12 --dwell_time 0.4
python scripts/run_env.py --policy <...> --no_proximity_stop
```

Scope: this is a deployment-time layer in the single-robot interactive loop. It
is **not** active during batched training or in `evaluate_policy.py`, and it
measures distance from the palm link origin, not the grasp reference point the
reward uses.

---

## Evaluation

```bash
# batch rollout of the deterministic policy (defaults to the newest checkpoint)
python scripts/evaluate_policy.py --headless --rate_limit on  --out results/eval_on.npz
python scripts/evaluate_policy.py --headless --rate_limit off --out results/eval_off.npz

# figures + LaTeX macros for paper/ (plain python, no Isaac Sim)
python scripts/plot_results.py
```

`evaluate_policy.py` prints a summary table and self-checks its own bookkeeping
(success ⇒ within tolerance, success ⇒ episode ≥ the 0.5 s hold, sampled azimuth
inside the configured span, rate limits respected). Metrics are recorded by
`EvalProbe` in `envs/mdp/eval_metrics.py`, an always-False *termination* term —
because by the time `env.step()` returns, terminated envs have already been
reset, so an outer-loop recorder would attribute the next episode's fruit
position to the one that just ended.

---

## Phase 3 — Add camera observations

Switch to the camera env:

```bash
python scripts/train.py --task AppleGrasp-Camera-v0 --num_envs 8
```

The head camera (128×128 RGB+Depth) is already wired up in `AppleGraspCameraEnvCfg`.  
You will need to add image encoding (CNN / ViT) to the policy network, and register
a new agent config / `rsl_rl_cfg_entry_point` for `AppleGrasp-Camera-v0` — see
`envs/agents/rsl_rl_ppo_cfg.py` for the pattern used by `AppleGrasp-v0`.

---

## Phase 4 — MuJoCo transfer

```bash
# Export trained policy to ONNX:
python scripts/export_policy.py --checkpoint logs/...

# Run in unitree_mujoco:
cd ~/Code/unitree_mujoco
python envs/apple_grasp_mujoco.py --policy ../g1_dex3_apple/exported/policy.onnx
```

---

## Key tuning knobs (apple_grasp_env_cfg.py)

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `APPLE_INIT_POS` | `(0.55, -0.25, 1.05)` | Apple start position |
| `scene.num_envs` | 32 | Parallel envs (lower if OOM) |
| `right_arm stiffness` | 40 | Arm joint stiffness (kp) |
| `right_hand stiffness` | 1.0 | Dex3 finger stiffness |
| `reward.reaching weight` | 1.0 | How strongly to reward proximity |
| `reward.lift weight` | 5.0 | How strongly to reward lifting |
| `decimation` | 4 | Physics steps per policy step |

---

## Troubleshooting

**`ModuleNotFoundError: envs`**  
→ Run scripts from the repo root: `cd g1_dex3_apple && python scripts/run_env.py`

**`UsdFileCfg: file not found`**  
→ Set `G1_DEX3_USD_PATH` correctly.

**Contact sensor fires but apple doesn't move**  
→ Dex3 stiffness too low or HAND_CLOSE targets too small. Increase both.

**OOM on RTX 4060**  
→ Reduce `scene.num_envs` to 16 or 8. Disable head camera (already off by default).

**Apple falls through ground**  
→ Increase `contact_offset` in apple's `CollisionPropertiesCfg`.
