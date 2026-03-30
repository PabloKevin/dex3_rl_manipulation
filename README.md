# G1 Dex3 Apple Grasp

Isaac Lab 4.5 project: Unitree G1-29dof + Dex3 right hand, fixed base,  
picks up a red sphere ("apple") and lifts it. Trains with PPO, transfers to MuJoCo.

---

## Repo structure

```
g1_dex3_apple/
├── envs/
│   ├── __init__.py                 ← gymnasium task registration
│   ├── apple_grasp_env_cfg.py      ← scene + managers (MAIN CONFIG)
│   └── mdp/
│       ├── observations.py         ← custom obs functions
│       ├── rewards.py              ← custom reward functions
│       └── terminations_and_events.py
├── scripts/
│   ├── run_env.py                  ← Phase 1: view the scene
│   └── move_arm.py                 ← Phase 1: scripted reach + grasp
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

## Phase 2 — RL training (PPO)

_Coming after Phase 1 validation._

```bash
# Train (32 envs, ~6h on RTX 4060):
python scripts/train.py --task AppleGrasp-v0 --num_envs 32

# Watch trained policy:
python scripts/play.py --checkpoint logs/ppo/.../model_XXXXX.pt
```

---

## Phase 3 — Add camera observations

Switch to the camera env:

```bash
python scripts/train.py --task AppleGrasp-Camera-v0 --num_envs 8
```

The head camera (128×128 RGB+Depth) is already wired up in `AppleGraspCameraEnvCfg`.  
You will need to add image encoding (CNN / ViT) to the policy network — see `config/ppo_apple_grasp.yaml`.

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
