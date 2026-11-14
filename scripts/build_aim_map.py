#!/usr/bin/env python3
"""Build the aim map (envs/mdp/aim.py): roll a frozen AppleArmPose-v5 policy out on random fruit goals, robot
only, from the harvest's READY pose at random waist yaws, and fit Q(goal relative to the shoulder in the torso
frame, palm heading, palm tilt, waist yaw) -> P(success).

A rollout: READY (+ jitter), ONE fruit goal for --seconds: goal position uniform in a box in front of the right
shoulder (torso frame), the palm normal = the horizontal torso-axis -> goal line turned by a random heading and
raised by a random tilt -- what the harvest commands. Recorded per rollout: reached (the policy's own stop test),
when, the final pocket error / normal error / palm speed, the deepest overlap with the body (sphere model) and the
deepest sweep through the fruit. Success = reached, still on it at the end, never in the body, fruit not swept.

    ~/miniconda3/envs/env_isaaclab/bin/python -u scripts/build_aim_map.py --headless \\
        --policy logs/rsl_rl/apple_arm_pose_v5/<run>/exported/policy.pt --batches 40
    -> envs/assets/aim_map_v5.pt (+ the raw rollouts next to it, .npz)
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--policy", required=True)
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--batches", type=int, default=40)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--out", default=None)
p.add_argument("--eval_only", action="store_true",
               help="only print the rollout statistics (no map written); a 35-input policy (AppleArmPose-v1/v2/v4) is "
                    "fed its own columns of the same goals, for a like-for-like comparison")
p.add_argument("--harvest_like", action="store_true",
               help="(--eval_only) goals as the harvest gives them: heading 0, tilt 0 (the palm along the torso -> "
                    "apple line), inside the tree placement's box")
p.add_argument("--fit_only", default=None, help="refit from an existing rollout .npz (no simulation)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = args.out or os.path.join(REPO, "envs", "assets", "aim_map_v5.pt")
BOX = dict(x=(0.08, 0.62), y=(-0.50, 0.32), z=(-0.52, 0.28), rmax=0.64)
HEADING_DEG, TILT_DEG = 60.0, 40.0
READY_Q = [-0.15, -0.15, -0.15, 0.5, 0.0, 0.0, 0.0]


def fit(data, out):
    import numpy as np
    import torch
    X = np.concatenate([data["rel"], data["heading"][:, None], data["tilt"][:, None], data["yaw"][:, None]], 1)
    ok = (data["arrived"] > 0) & (data["d_end"] < 0.03) & (data["ang_end"] < math.radians(20.0)) \
        & (data["overlap"] < 0.005) & (data["sweep"] < 0.03)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(ok, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    nv = max(1000, len(X) // 10)
    va, tr = perm[:nv], perm[nv:]
    mean, std = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    dims = [6, 128, 128, 64, 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(*layers).to(dev)
    Xn, yy = ((X - mean) / std).to(dev), y.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
    tr_d, va_d = tr.to(dev), va.to(dev)
    for it in range(6000):
        idx = tr_d[torch.randint(0, len(tr_d), (4096,), device=dev)]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xn[idx]).squeeze(-1), yy[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pv = torch.sigmoid(net(Xn[va_d]).squeeze(-1)).cpu()
    yv = y[va]
    acc = float(((pv > 0.5).float() == yv).float().mean())
    order = pv.argsort()
    ranks = torch.empty_like(pv); ranks[order] = torch.arange(1, len(pv) + 1, dtype=torch.float32)
    n1, n0 = float(yv.sum()), float((1 - yv).sum())
    auc = (float(ranks[yv > 0.5].sum()) - n1 * (n1 + 1) / 2) / max(n1 * n0, 1.0)
    brier = float(((pv - yv) ** 2).mean())
    import json
    CAMERA_D435I = json.loads(str(data["camera_json"]))
    torch.save({"dims": dims, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "mean": mean, "std": std, "shoulder_torso": torch.tensor(data["shoulder_torso"], dtype=torch.float32),
                "camera": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CAMERA_D435I.items()},
                "yaw_range_deg": (-45.0, 15.0), "policy": str(data.get("policy", "")), "n": len(X),
                "success_rate": float(y.mean()), "val_acc": acc, "val_auc": auc, "val_brier": brier,
                "features": ["x_sh", "y_sh", "z_sh", "heading", "tilt", "yaw"], "ready_q": READY_Q}, out)
    print(f"[aim] {len(X)} rollouts, success {float(y.mean()):.3f}; held-out acc {acc:.3f}, AUC {auc:.3f}, "
          f"Brier {brier:.3f} (trivial {float(yv.mean() * (1 - yv.mean())):.3f}) -> {out}", flush=True)


if args.fit_only:
    import numpy as np
    sys.path.insert(0, REPO)
    fit(dict(np.load(args.fit_only, allow_pickle=True)), OUT)
    sys.exit(0)

app = AppLauncher(args).app
sys.path.insert(0, REPO)
os.chdir(REPO)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_arm_pose_env_cfg import POCKET  # noqa: E402
from envs.apple_arm_pose_v5_env_cfg import AppleArmPoseV5EnvCfg  # noqa: E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.mdp import arm_pose as ap  # noqa: E402
from envs.mdp import arm_pose_fruit as apf  # noqa: E402
from envs.mdp import self_collision as scm  # noqa: E402

cfg = AppleArmPoseV5EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.episode_length_s = 1.0e5
cfg.events.advance_goals = None
cfg.events.hold_push = None
cfg.events.assign_fruit = None
cfg.terminations.collision = None
env = gym.make("AppleArmPose-v5", cfg=cfg).unwrapped
N, dev, dt = env.num_envs, env.device, env.step_dt
robot = env.scene["robot"]
pol = torch.jit.load(args.policy, map_location=dev).eval()
N_IN = next(q_ for q_ in pol.parameters() if q_.dim() == 2).shape[1]


def pol_in(o):
    # the v5 layout is v1's 35 with a third flag (col 28) and the waist yaw (col 36) added
    return o if N_IN == 37 else torch.cat([o[:, :28], o[:, 29:36]], dim=-1)


if args.harvest_like:
    HEADING_DEG, TILT_DEG = 0.0, 0.0
    BOX = dict(x=(0.20, 0.50), y=(-0.25, 0.15), z=(-0.42, 0.05), rmax=0.56)
arm = env.action_manager.get_term("right_arm")
arm_ids = list(arm._joint_ids)
arm_names = [robot.joint_names[j] for j in arm_ids]
ready = torch.tensor([READY_Q[RIGHT_ARM_JOINTS.index(n)] for n in arm_names], device=dev)
P = ap.pose_state(env)
F = apf.fruit_state(env)
sh_id = robot.data.body_names.index(ap.SHOULDER)
palm_id = robot.data.body_names.index(ap.PALM)
wj = robot.joint_names.index(ap.WAIST)
steps = int(round(args.seconds / dt))
rows = {k: [] for k in ("rel", "heading", "tilt", "yaw", "arrived", "t_arrive", "d_end", "ang_end", "v_end",
                        "overlap", "sweep")}
shoulder_torso = None
torch.manual_seed(args.seed)
with torch.inference_mode():
    for b in range(args.batches):
        env.reset()
        # the arm to READY (+ jitter), at rest; the waist yaw is the reset's (random in the library's range)
        jp = robot.data.joint_pos.clone()
        q0 = ready + 0.06 * (2 * torch.rand(N, len(arm_ids), device=dev) - 1)
        q0 = torch.maximum(torch.minimum(q0, arm._clip[:, :, 1]), arm._clip[:, :, 0])
        jp[:, arm_ids] = q0
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
        hold = (q0 - arm._offset) / arm._scale
        for _ in range(3):
            obs, *_ = env.step(hold)
        tp, tq = ap._torso(env)
        sh = robot.data.body_pos_w[:, sh_id]
        if shoulder_torso is None:
            shoulder_torso = quat_apply_inverse(tq[:1], (sh - tp)[:1])[0].cpu().numpy()
        rel = torch.zeros(N, 3, device=dev)
        todo = torch.ones(N, dtype=torch.bool, device=dev)
        for _ in range(20):
            cand = torch.stack([BOX[k][0] + (BOX[k][1] - BOX[k][0]) * torch.rand(N, device=dev) for k in "xyz"], -1)
            rel = torch.where(todo.unsqueeze(-1), cand, rel)
            todo = rel.norm(dim=-1) > BOX["rmax"]
            if not bool(todo.any()):
                break
        tgt = sh + quat_apply(tq, rel)
        u = tgt - tp
        u[:, 2] = 0.0
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        heading = torch.deg2rad(HEADING_DEG * (2 * torch.rand(N, device=dev) - 1))
        tilt = torch.deg2rad(TILT_DEG * (2 * torch.rand(N, device=dev) - 1))
        nrm = apf.approach_normal(u, heading, tilt)
        pk, _, fin = ap.hand_frame(env, POCKET)
        P.target[:], P.normal[:], P.finger[:] = tgt, nrm, fin
        P.finger_on[:] = False
        P.via[:] = False
        P.start[:] = pk
        P.pending[:] = False
        P.arrived[:] = False
        P.success[:] = False
        P.hold[:] = 0.0
        P.hold_left[:] = 1.0e9
        P.t_goal[:] = 0.0
        P.push[:] = 0.0
        F.on[:] = True
        F.tracked[:] = False
        F.r[:] = 0.04
        F.worst[:] = 0.0
        obs, *_ = env.step(hold)
        t_arr = torch.full((N,), float("nan"), device=dev)
        overlap = torch.zeros(N, device=dev)
        for k in range(steps):
            act = pol(pol_in(obs["policy"]))
            obs, *_ = env.step(act)
            new = P.arrived & torch.isnan(t_arr)
            t_arr[new] = (k + 1) * dt
            overlap = torch.maximum(overlap, scm.self_collision_depth(env, margin=0.0).amax(-1))
        d, ang, _, _ = ap.errors(env, POCKET)
        v = robot.data.body_lin_vel_w[:, palm_id].norm(dim=-1)
        yaw = robot.data.joint_pos[:, wj]
        for k, val in (("rel", rel), ("heading", heading), ("tilt", tilt), ("yaw", yaw), ("arrived", P.arrived.float()),
                       ("t_arrive", t_arr), ("d_end", d), ("ang_end", ang), ("v_end", v), ("overlap", overlap),
                       ("sweep", F.worst)):
            rows[k].append(val.cpu().numpy().copy())
        ok = P.arrived & (d < 0.03) & (overlap < 0.005) & (F.worst < 0.03)
        print(f"[aim] batch {b + 1}/{args.batches}: reached {float(P.arrived.float().mean()):.3f}, success "
              f"{float(ok.float().mean()):.3f}, body overlap {float((overlap >= 0.005).float().mean()):.3f}, fruit "
              f"swept {float((F.worst >= 0.03).float().mean()):.3f}, median t {float(torch.nanmedian(t_arr)):.2f} s",
              flush=True)
data = {k: np.concatenate(v) for k, v in rows.items()}
ok_all = (data["arrived"] > 0) & (data["d_end"] < 0.03) & (data["ang_end"] < math.radians(20.0)) \
    & (data["overlap"] < 0.005) & (data["sweep"] < 0.03)
print(f"[aim] TOTAL {len(ok_all)} rollouts ({'harvest-like' if args.harvest_like else 'wide'} goals): reached "
      f"{data['arrived'].mean():.3f}, success {ok_all.mean():.3f}, on the goal at the end (3 cm / 20 deg) "
      f"{((data['d_end'] < 0.03) & (data['ang_end'] < math.radians(20.0))).mean():.3f}, median end error "
      f"{np.median(data['d_end']) * 100:.1f} cm, end speed {np.median(data['v_end']) * 100:.1f} cm/s, body overlap "
      f"{(data['overlap'] >= 0.005).mean():.3f}, fruit swept {(data['sweep'] >= 0.03).mean():.3f}, median time "
      f"{np.nanmedian(data['t_arrive']):.2f} s", flush=True)
z = data["rel"][:, 2]
for lo_, hi_ in ((-0.55, -0.40), (-0.40, -0.30), (-0.30, -0.20), (-0.20, -0.10), (-0.10, 0.05), (0.05, 0.30)):
    m_ = (z >= lo_) & (z < hi_)
    if m_.sum() > 20:
        print(f"[aim]   z_sh {lo_:+.2f}..{hi_:+.2f}: n {int(m_.sum()):5d}, success {ok_all[m_].mean():.3f}, "
              f"reached {data['arrived'][m_].mean():.3f}, swept {(data['sweep'][m_] >= 0.03).mean():.3f}", flush=True)
if args.eval_only:
    os._exit(0)
data["shoulder_torso"] = shoulder_torso
data["policy"] = args.policy
import json  # noqa: E402
from envs.apple_reach_visible_env_cfg import CAMERA_D435I  # noqa: E402
data["camera_json"] = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in CAMERA_D435I.items()})
raw = os.path.splitext(OUT)[0] + "_rollouts.npz"
np.savez_compressed(raw, **data)
print(f"[aim] rollouts -> {raw}", flush=True)
fit(data, OUT)
os._exit(0)
