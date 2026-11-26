#!/usr/bin/env python3
"""EXPERIMENTO: EL MODELO ELIGE EL GIRO DE CINTURA (proyecto final de Ciencia de Datos e IA, UTEC 2026/2).

Copia de ml_project/recolectar_dataset.py (que NO se modifica). Al empezar cada árbol, el bosque aleatorio
entrenado en el notebook (modelos/bosque_exito.npz, exportado por exportar_bosque.py) evalúa los giros de
cintura posibles (-15..+15 grados, el rango de los datos) y, según --cintura:

  * mantener  : el robot se queda con el giro con que empezó (como en el dataset) = LÍNEA BASE;
  * modelo    : gira la cintura al giro de mayor probabilidad de éxito predicha;
  * aleatorio : gira a un giro al azar (control: ¿mejora el modelo o solo el hecho de girar?).

Solo cuentan los giros en que la cámara ve la manzana objetivo entera. El árbol queda QUIETO en el mundo y gira
el robot (como en el robot real): para cada giro candidato se recalculan las entradas (posición respecto al
hombro, píxel y profundidad en la cámara, vecinas visibles) sin mover nada. En los tres modos se guardan la
probabilidad predicha con el giro inicial y con el elegido, para comparar predicción y resultado.

    python ml_project/cosecha_cintura.py --headless --num_envs 48 --episodes 600 --seed 3 \
        --perceived_apple --apple_gap 0.06 --cintura modelo

--- documentación de recolectar_dataset.py ---

RECOLECCIÓN DEL DATASET (proyecto final de Ciencia de Datos e IA, UTEC 2026/2).

Copia de scripts/play_harvest_robot.py (el original NO se modifica) que corre la cosecha completa del robot en
simulación, episodio tras episodio, y guarda UNA FILA POR INTENTO en un CSV:

  * entradas (lo que el robot conoce ANTES de intentar): posición de la manzana objetivo respecto al hombro
    derecho en el marco del torso, píxel y profundidad en la cámara D435i, radio, yaw de la cintura, y los
    vecinos que la cámara VE ENTEROS (un vecino fuera del cuadro no cuenta como vecino). La distancia al tronco
    NO se guarda como entrada (en el robot real todavía no se estima bien);
  * etiquetas: exito (manzana depositada en la canasta) y etapa (exito / alcance / agarre / desprendimiento /
    retorno / transporte / deposito / tiempo) + el motivo en texto.

Cada env hace UN intento por árbol; al terminar (canasta o fallo) recibe un árbol nuevo (el robot vuelve a su
pose por defecto y la cintura toma un yaw nuevo). El CSV se escribe fila a fila (Ctrl-C no pierde lo ya
guardado) y, si el archivo ya existe, se agregan filas al final.

    ~/miniconda3/envs/env_isaaclab/bin/python -u ml_project/recolectar_dataset.py --headless \\
        --num_envs 16 --episodes 1500 --seed 0 --out ml_project/datos/cosecha_seed0.csv

--- documentación original (play_harvest_robot.py) ---

The ROBOT's autonomous harvest (sim2real/toy_harvest.py) in sim, many envs at once, recorded as a video.

Mirrors the real command

    g1_launch.sh toy --vision onboard -- harvest --policy apple_arm_pose/2026-10-19_13-31-55/exported/policy.pt
        --speed 0.3 (sim default 1.0) --target detection --onboard --trunk_side off --reactive off --grasp_mode policy
        --grip_n 20 --grip_stop_n 30

with the same policies and the same per-apple sequence (toy_harvest.harvest_one + deposit):

  READY      HOME (tucked) -> home_ready_via -> READY, scripted quintic joint moves (moves.speed_rad_s 0.6)
  APPROACH   the arm-pose policy (AppleArmPose-v1, the robot's export) at --speed through the robot's
             limiter (1.2 rad/s, 5 rad/s^2, braking, x speed): STAGING (0.24 m from the torso axis at the
             apple's height; left at 5 cm / 30 deg or 2.4 s / speed), then the APPLE (pocket on the
             perceived apple, the trained 2 cm / 15 deg / 0.2 s stop test, 4 s / speed timeout); the hand
             fully open (harvest_config reach_hand). The arm's path is recorded for the detach.
             More than 5 cm off at the end = no grasp (GRASP_MAX_ERR).
  PREGRASP   the hand to the pregrasp cup, cup_s 1.0
  GRASP      --grasp policy: the hand-only grasp policy (AppleGraspSpindle-v2 model 1499, the robot's
             default), FINGERS only, the arm held, grip command --grip_n, its hand limiter at --speed;
             done when two fingers > 0.5 N for 0.3 s and kept --grasp_hold_s, or --grasp_s without a grip;
             the fingers stop where they are above --grip_stop_n.
             --grasp scripted: the "traditional" close (each finger closes until it feels the fruit, then
             squeezes on), held CLOSE_HOLD_S -- the picker's close as on most successful robot runs.
  TWIST      45 deg about the palm normal through the pocket, hold 0.2 s, back (harvest_config twist)
  DETACH     the approach's recorded arm path played backwards for 8 cm of pocket travel at 0.03 m/s
  RETURN     arm-pose policy: staging back, READY; then the held check (the fruit in the pocket)
  HOME       READY -> via -> HOME
  DEPOSIT    the waist into [-15, 0] deg (only as far as needed), HOME -> deposit_start_q (scripted), the
             arm-pose policy to the drop (the basket point fixed to the ground, palm DOWN, fingers along the
             arm), the hand opens (1.5 s), the policy back to deposit start, HOME, the waist back.

The world is AppleGraspArmPose-v1's (the tree the head camera sees, a random fixed waist yaw per tree,
the perceived apple with the camera-ray bias, breakable stems on every fruit) + a basket under the drop.
A miss (no grasp / the stem held / nothing in the hand) tries the same apple again (up to --max_tries); a
fruit that came off the tree without being deposited gives that env a new tree once its arm is home.
The video ends when every env has deposited one apple (+ --tail_s), or at --max_s.

    ~/miniconda3/envs/env_isaaclab/bin/python -u scripts/play_harvest_robot.py --headless --num_envs 9 \\
        --video logs/videos/harvest_robot_sim.mp4
"""
import argparse
import csv
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--out", default="", help="CSV de salida (vacío: ml_project/datos_cintura/<cintura>_seed<semilla>.csv)")
p.add_argument("--cintura", choices=["mantener", "modelo", "aleatorio"], required=True,
               help="mantener = línea base; modelo = giro de mayor P(éxito) predicha; aleatorio = giro al azar")
p.add_argument("--modelo", default="ml_project/modelos/bosque_exito.npz", help="bosque exportado (exportar_bosque.py)")
p.add_argument("--giro_min_deg", type=float, default=-15.0)
p.add_argument("--giro_max_deg", type=float, default=15.0)
p.add_argument("--giro_paso_deg", type=float, default=2.5)
p.add_argument("--episodes", type=int, default=1500, help="intentos a guardar en esta corrida")
p.add_argument("--ep_max_s", type=float, default=120.0, help="tiempo simulado máximo por intento (etapa 'tiempo')")
p.add_argument("--detalle", action="store_true", help="imprime el detalle de cada env (como el script original)")
p.add_argument("--perfil", action="store_true",
               help="mide cuánto tarda cada parte del paso (ms por paso de control) y lo imprime cada 250 pasos")
p.add_argument("--con_extras", action="store_true",
               help="calcula recompensas, currículum y observaciones del entorno de entrenamiento (como el original); "
                    "por defecto se desactivan: la cosecha no los usa")
p.add_argument("--arm_policy", default=os.path.join(REPO, "logs/rsl_rl/apple_arm_pose/2026-10-19_13-31-55/exported/policy.pt"))
p.add_argument("--grasp", choices=["policy", "scripted"], default="scripted",
               help="scripted: the traditional close (most successful robot runs); policy: the robot command's "
                    "--grasp_mode policy (AppleGraspSpindle-v2 1499)")
p.add_argument("--grasp_policy", default=os.path.join(REPO, "envs/agents/best_weights/grasp_spindle_v2_1499.pt"))
p.add_argument("--speed", type=float, default=1.0,
               help="speed factor of the arm-pose policy limiter and the grasp limiter (the robot runs 0.3)")
p.add_argument("--grip_n", type=float, default=20.0)
p.add_argument("--grip_stop_n", type=float, default=30.0)
p.add_argument("--grasp_s", type=float, default=4.0)
p.add_argument("--grasp_hold_s", type=float, default=2.0)
p.add_argument("--close_speed", type=float, default=0.8, help="(scripted) finger closing speed, rad/s")
p.add_argument("--close_force", type=float, default=1.0, help="(scripted) contact force that stops a finger, N")
p.add_argument("--squeeze", type=float, default=1.4,
               help="(scripted) how far past full closure the finger targets may go, rad")
p.add_argument("--hold_n", type=float, default=25.0,
               help="(scripted) each finger group's contact force kept from the close until the release, N (the "
                    "picker's settle_and_hold: closes, then keeps every finger on its pressure)")
p.add_argument("--hand_kp", type=float, default=3.0,
               help="(scripted) hand joint stiffness, N*m/rad: the Dex3 SDK's grip mode (the sim's own 1.0 is its "
                    "rotate-mode-like default, which the grasp policy trained with)")
p.add_argument("--perceived_apple", action="store_true",
               help="aim at the perceived apple (the world's camera-ray bias); default: the true centre (perfect "
                    "perception: the fruit then sits between all three fingers)")
p.add_argument("--max_tries", type=int, default=3)
p.add_argument("--one_try", type=int, default=1, choices=[1],
               help="1: ONE harvest episode per env (deposit or miss), the video ends when every env is done")
p.add_argument("--hand_friction", type=float, nargs=2, default=[1.2, 1.0],
               help="hand static, dynamic friction (the world randomizes 0.6-1.0 / 0.45-0.8; the Dex3 pads are rubber)")
p.add_argument("--reaim", type=int, default=0,
               help="1: if the apple is > 2.5 cm from the pocket when the reach ends (pushed, or stopped short), one more "
                    "arm-pose leg (2.5 s) to where it is now (a fresh detection)")
p.add_argument("--reach_hand", choices=["cup", "open"], default="cup",
               help="hand during the reach: the pregrasp cup (default) or fully open (harvest_config reach_hand)")
p.add_argument("--stems", choices=["magnet", "tree"], default="tree",
               help="magnet: every fruit on the lab replica apples' magnet attachment (pregrasp.MAGNET_STEM_PARAMS: "
                    "1-3 N pull beyond the weight, 0.5-1.5 N sideways); tree: the world's real-stem limits (15-25 N "
                    "pull, twist breaks at 41-92 deg: the 45 deg twist rarely detaches, the pull slips out of the "
                    "sim grip)")
p.add_argument("--stem_break_pull", type=float, nargs=2, default=[6.0, 10.0],
               help="the TARGET apple's straight-pull break force range, N (the tree's own 15-25 N, kept for the "
                    "neighbours, is beyond the sim Dex3 grip: median 12 N measured; 0 0 = the tree's)")
p.add_argument("--target_bend", type=float, nargs=2, default=[0.15, 0.25],
               help="the TARGET's sideways-bend break moment once held, N*m (the pull is mostly across the stem)")
p.add_argument("--carry_speed", type=float, default=0.6, help="scripted moves with the apple in the hand, rad/s (robot 0.6)")
p.add_argument("--neighbour_bend", type=float, nargs=2, default=[0.5, 0.9],
               help="every stem's sideways-bend break moment, N*m (~7-13 N push at the fruit; the tree's own 0.25-0.45 "
                    "= 4-6 N let the reach knock neighbours off too easily)")
p.add_argument("--apples", type=int, nargs=2, default=[2, 5], help="apples per tree, min max (random per tree)")
p.add_argument("--apple_gap", type=float, default=0.10,
               help="neighbours kept only this far (surface to surface) from the apple to pick, m; every apple hangs "
                    "alone (no clusters; the world's own: clusters of 1-4 touching fruit, 6 cm between clusters, which "
                    "is also the sampler's limit for 5 apples in the reachable view)")
p.add_argument("--basket_xy", type=float, nargs=2, default=None,
               help="basket axis, env-local (default: under the robot's drop point, computed by --probe)")
p.add_argument("--probe", action="store_true", help="print the drop points and the basket position, then exit")
p.add_argument("--yaw_range", type=float, nargs=2, default=[-15.0, 15.0],
               help="each tree's fixed waist yaw range, deg (the world's -45..15: at the far right the trunk stands in "
                    "the basket)")
p.add_argument("--move_speed", type=float, default=1.2, help="scripted joint moves, rad/s (robot 0.6)")
p.add_argument("--twist_deg", type=float, default=0.0, help="twist before the pull, deg (robot 45; off: the sim's "
               "IK twist pulled apples out of the hand)")
p.add_argument("--pull_m", type=float, default=0.12, help="detach pull length, m (robot 0.08)")
p.add_argument("--pull_speed", type=float, default=0.08, help="detach pull speed, m/s (robot 0.03)")
p.add_argument("--close_hold_s", type=float, default=0.5, help="grip held this long after the close, s")
p.add_argument("--full_cycle", type=int, default=0,
               help="1: the robot's whole cycle (READY -> HOME -> waist -> deposit start, back HOME after the drop); "
                    "0: READY -> deposit start while the waist turns, the episode ends with the apple in the basket")
p.add_argument("--trunk_h", type=float, default=1.70, help="trunk height, m (the world's 2.0-2.5)")
p.add_argument("--lean_deg", type=float, nargs=2, default=[2.0, 7.0],
               help="lean of the upper trunk, deg (the world's 3-15)")
p.add_argument("--stand_h", type=float, default=0.42, help="height of the box under the basket, m (basket.py 0.58)")
p.add_argument("--open_s", type=float, default=0.8, help="the hand opens over this long at the basket (a snap-open "
               "flings the squeezed sim apple out)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--diag", action="store_true", help="per-state motion stats (joint speed, lag, palm detour) at the end")
p.add_argument("--max_s", type=float, default=1.0e9, help="simulated seconds at most")
p.add_argument("--tail_s", type=float, default=2.0, help="recorded after the last env finished")
p.add_argument("--video", type=str, default=None, help="mp4, timed in simulated time (works headless)")
p.add_argument("--video_fps", type=int, default=25)
p.add_argument("--video_res", type=int, nargs=2, default=[1920, 1080])
p.add_argument("--cam_eye", type=float, nargs=3, default=None, help="relative to the centre of the env grid")
p.add_argument("--cam_lookat", type=float, nargs=3, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if not args.out:
    args.out = f"ml_project/datos_cintura/{args.cintura}_seed{args.seed}.csv"
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, REPO)
os.chdir(REPO)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_arm_pose_env_cfg import POCKET  # noqa: E402
from envs.apple_grasp_arm_pose_env_cfg import AppleGraspArmPoseEnvCfg  # noqa: E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_grasp_tree_env_cfg import HandOnlyObservationsCfg  # noqa: E402
from envs.apple_reach_spindle_env_cfg import ARM_TARGET_CLIP, REACH_HAND_PRESHAPE  # noqa: E402
from envs.mdp import apple_tree as at  # noqa: E402
from envs.mdp import arm_pose as ap  # noqa: E402
from envs.mdp import basket as bk  # noqa: E402
from envs.mdp import grasp_autoplay as ga  # noqa: E402
from envs.mdp import harvest as hv  # noqa: E402

HC = json.load(open(os.path.join(REPO, "sim2real", "harvest_config.json")))
HOME_Q = HC["home_q"]
READY_Q = [-0.15, -0.15, -0.15, 0.5, 0.0, 0.0, 0.0]          # reach_io.HOME_Q (harvest_config ready_q null)
VIA_Q = HC["home_ready_via"]
DEP_Q = HC["deposit_start_q"]
MOVE_SPEED = args.move_speed
BASKET = HC["basket"]
TWIST = {**HC["twist"], "deg": args.twist_deg}
DETACH = {**HC["detach"], "pull_speed_m_s": args.pull_speed, "pull_m": args.pull_m}
REACH_HAND = HC["reach_hand"]
# arm_pose_io / toy_arm_pose / toy_harvest constants
STAGE_RADIUS, MIN_STAGE_BACK = 0.24, 0.08
STAGE_ADV_D, STAGE_ADV_DEG, STAGE_ADV_S = 0.05, 30.0, 2.4
WAYPOINT_TIMEOUT = 4.0 / args.speed
TRACK = dict(pos_tol=0.02, ang_tol_deg=15.0, finger_tol_deg=25.0, speed_tol=0.08, hold_time=0.2,
             via_pos_tol=0.03, via_ang_tol_deg=25.0)
GRASP_MAX_ERR = 0.05
CLOSE_HOLD_S, RELEASE_S = args.close_hold_s, 1.5
LEAVE = 0.06                    # fruit this far from the pocket = not in the hand
ARM_VMAX, ARM_AMAX = 1.2, 5.0   # the arm-pose policy's limiter (reach_io "arm_pose" layout), braking
WAIST_SPEED = math.radians(30.0)

# ── env ──
from envs import apple_grasp_tree_env_cfg as tree_cfg_mod  # noqa: E402

tree_cfg_mod.APPLES_PER_TREE = args.apples[1]                 # read by the scene / event cfgs below
cfg = AppleGraspArmPoseEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.episode_length_s = 1.0e5
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.terminations.next_apple = TerminationTermCfg(func=at.next_apple_requested)
cfg.events.release_stem_and_load = None                  # the training's twist / pull load
cfg.events.reset_arm_jitter = None
cfg.events.friction_hand.params.update(static_friction_range=(args.hand_friction[0],) * 2,
                                       dynamic_friction_range=(args.hand_friction[1],) * 2)
cfg.events.randomize_tree.params["yaw_range_deg"] = tuple(args.yaw_range)
cfg.events.randomize_tree_geometry.params["trunk_height_range"] = (args.trunk_h, args.trunk_h)
cfg.events.randomize_tree.params["spindle"] = {**cfg.events.randomize_tree.params["spindle"],
                                               "lean_deg": tuple(args.lean_deg)}
sp = dict(cfg.events.randomize_tree.params.get("stem_params") or {})
if args.stems == "magnet":
    from envs.mdp.pregrasp import MAGNET_STEM_PARAMS
    sp.update(MAGNET_STEM_PARAMS)
sp["break_bend"] = tuple(args.neighbour_bend)
cfg.events.randomize_tree.params.update(cluster_sizes=(1,), cluster_weights=(1.0,))      # no touching clusters
cfg.events.randomize_tree.params["stem_params"] = sp
cfg.events.randomize_tree.params["new_tree_each_reset"] = False
cfg.events.reset_reach_grasp_phase.params.update(
    stem_release_schedule=None, stem_release_time_range=None, release_on_latch=False, shaping_grace_s=None)
# this script drives the arm from the first step: the "hand-off" right away
cfg.events.update_reach_grasp_phase.params.update(handoff_mode="fixed", handoff_time_s=1e-3)
# the grasp policy's 44 inputs as their own group (the env's own policy group is the 65-input one)
cfg.observations.grasp = HandOnlyObservationsCfg.PolicyObs()
if not args.con_extras:
    # Nada de esto lo usa la cosecha (el brazo arma sus propias entradas con ga._reach_obs; los pedúnculos se
    # aplican en cada subpaso de física y el seguimiento de la cámara es un evento, no una recompensa):
    # recompensas y currículum (métricas) siempre; las observaciones solo con el agarre scripted (la política de
    # agarre lee obs["grasp"]).
    def _sin_terminos(grupo):
        if grupo is None:
            return 0
        n_ = 0
        for name in list(vars(grupo)):
            if not name.startswith("_") and getattr(grupo, name) is not None:
                setattr(grupo, name, None)
                n_ += 1
        return n_
    _nr = _sin_terminos(cfg.rewards)
    _nc = _sin_terminos(getattr(cfg, "curriculum", None))
    _no = _sin_terminos(cfg.observations) if args.grasp == "scripted" else 0
    print(f"[dataset] desactivados: {_nr} recompensas, {_nc} términos de currículum, {_no} grupos de observaciones",
          flush=True)
if args.basket_xy is not None:
    bk.CENTER_XY = tuple(args.basket_xy)
else:
    bk.CENTER_XY = (0.0, -0.60)                          # under the drop point (--probe: waist 0 and -15 both y -0.60)
# a shorter stand, the basket itself unchanged (its heights are module constants derived from the stand)
bk.STAND_SIZE = (bk.STAND_SIZE[0], bk.STAND_SIZE[1], args.stand_h)
bk.FLOOR_Z = bk.STAND_SIZE[2] + bk.FLOOR_THICKNESS
bk.RIM_Z = bk.FLOOR_Z + bk.DEPTH
if not args.probe:
    bk.add_to_scene(cfg.scene)
if args.grasp == "scripted":
    cfg.scene.robot.actuators["right_hand"].stiffness = args.hand_kp
if args.video:
    cfg.viewer.resolution = tuple(args.video_res)
env = gym.make("AppleGrasp-v1", cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
N, dev, dt = env.num_envs, env.device, env.step_dt
robot = env.scene["robot"]
apple = env.scene["apple"]
arm = env.action_manager.get_term("right_arm")
hand = env.action_manager.get_term("right_hand")
AD, HD = arm.action_dim, hand.action_dim
arm_ids = list(arm._joint_ids)
arm_names = [robot.joint_names[j] for j in arm_ids]
perm = [RIGHT_ARM_JOINTS.index(n) for n in arm_names]          # RIGHT_ARM_JOINTS order -> term order
hand_names = [robot.joint_names[j] for j in hand._joint_ids]
wj = robot.joint_names.index("waist_yaw_joint")
torso_id = robot.data.body_names.index("torso_link")
shoulder_id = robot.data.body_names.index("right_shoulder_pitch_link")
palm_id = robot.data.body_names.index("right_hand_palm_link")


def q_term(q):
    return torch.tensor([q[i] for i in perm], device=dev, dtype=torch.float32)


HOME, READY, VIA, DEPS = q_term(HOME_Q), q_term(READY_Q), q_term(VIA_Q), q_term(DEP_Q)
CLIP_LO = torch.tensor([ARM_TARGET_CLIP[n][0] for n in arm_names], device=dev)
CLIP_HI = torch.tensor([ARM_TARGET_CLIP[n][1] for n in arm_names], device=dev)
ARM_DEFAULT = arm._offset if isinstance(arm._offset, torch.Tensor) else robot.data.default_joint_pos[:, arm_ids]


def hand_q(d):
    base = robot.data.default_joint_pos[0, hand._joint_ids].clone()
    for k, n in enumerate(hand_names):
        if n in d:
            base[k] = float(d[n])
    return base


# the reach: fully open (reach_hand); the pregrasp: the policy's trained cup, or the picker's cup + reach thumb
H_REACH = hand_q({**REACH_HAND_PRESHAPE, **{f"right_hand_{k}_joint": v for k, v in REACH_HAND.items() if k != "cup_s" and not k.startswith("_")}})
H_CUP = hand_q(REACH_HAND_PRESHAPE) if args.grasp == "policy" else hand_q(
    {**REACH_HAND_PRESHAPE, **{f"right_hand_{k}_joint": v for k, v in REACH_HAND.items() if k.startswith("thumb")}})
H_OPEN = robot.data.default_joint_pos[0, hand._joint_ids].clone()
if args.reach_hand == "cup":
    H_REACH = H_CUP.clone()

arm_pol = torch.jit.load(args.arm_policy, map_location=dev).eval()
grasp_pol = torch.jit.load(args.grasp_policy, map_location=dev).eval() if args.grasp == "policy" else None
P = ap.pose_state(env)
ik = hv.DifferentialIK(robot, "right_hand_palm_link", arm_ids)
c_local = torch.tensor(POCKET, device=dev).expand(N, 3)
UP = torch.tensor([0.0, 0.0, 1.0], device=dev)


def T():
    return env._tree


# ── scripted grasp (the "traditional" close) ──
_lim = robot.data.soft_joint_pos_limits[0, hand._joint_ids]
_far = torch.where((_lim[:, 1] - H_OPEN).abs() > (_lim[:, 0] - H_OPEN).abs(), _lim[:, 1], _lim[:, 0])
_dir = torch.sign(_far - H_OPEN)
_goal = H_OPEN + 0.95 * (_far - H_OPEN)
SC_GROUPS = {"thumb": ["right_hand_thumb_1_joint", "right_hand_thumb_2_joint"],
             "index": ["right_hand_index_0_joint", "right_hand_index_1_joint"],
             "middle": ["right_hand_middle_0_joint", "right_hand_middle_1_joint"]}
SC_LINKS = {"thumb": ["right_hand_thumb_1_link", "right_hand_thumb_2_link"],
            "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
            "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"]}
_sensor = env.scene["contact_forces"]
_gmask = torch.zeros(3, HD, device=dev)
_glinks = []
for g, name in enumerate(SC_GROUPS):
    for j in SC_GROUPS[name]:
        _gmask[g, hand_names.index(j)] = 1.0
    _glinks.append([_sensor.body_names.index(l) for l in SC_LINKS[name]])


def finger_forces():
    f = _sensor.data.net_forces_w.norm(dim=-1)
    return torch.stack([f[:, ids].amax(-1) for ids in _glinks], dim=1)       # [N, 3]


# ── geometry ──
def frames():
    tp, tq = robot.data.body_pos_w[:, torso_id], robot.data.body_quat_w[:, torso_id]
    return tp, tq, robot.data.body_pos_w[:, shoulder_id]


def pocket_now():
    return ap.hand_frame(env, POCKET)


def fruit_to_pocket():
    return (apple.data.root_pos_w - pocket_now()[0]).norm(dim=-1)


# the basket point B: fixed to the ground, from the shoulder with the waist at 0 (computed once per env at
# the first reset: the robot's pelvis is fixed, so it is the same in env-local terms every time)
B_LOCAL = None


def basket_drop(i):
    """(drop point [3], palm normal, finger dir, short) for env i at its current waist (toy basket_goal)."""
    tp, tq, sh = frames()
    B = B_LOCAL + env.scene.env_origins[i]
    u = B - tp[i]
    u[2] = 0.0
    dB = float(u.norm())
    u = u / max(dB, 1e-9)
    drop = B.clone()
    if float((B - sh[i]).norm()) > BASKET["max_reach_m"]:
        lo, hi = 0.0, dB
        for _ in range(40):
            m = 0.5 * (lo + hi)
            pt = B - m * u
            lo, hi = (lo, m) if float((pt - sh[i]).norm()) <= BASKET["max_reach_m"] else (m, hi)
        drop = B - hi * u
    up = quat_apply(tq[i:i + 1], UP.view(1, 3))[0]
    f = drop - sh[i]
    f = f - (f * up).sum() * up
    f = f / f.norm().clamp(min=1e-9)
    return drop, -up, f, float((B - drop)[:2].norm())


# ── per-env state machine ──
(S_START, S_TO_READY, S_APPROACH, S_PREGRASP, S_GRASP, S_TWIST, S_DETACH, S_RETURN, S_CHECK, S_TO_HOME,
 S_WAIST_DEP, S_TO_DEP, S_DEPOSIT, S_RELEASE, S_BACK, S_DEP_HOME, S_WAIST_BACK, S_FAILHOME, S_DONE) = range(19)
NAMES = ["start", "to_ready", "approach", "pregrasp", "grasp", "twist", "detach", "return", "check", "to_home",
         "waist_dep", "to_dep", "deposit", "release", "back", "dep_home", "waist_back", "fail_home", "done"]
S_GIRO = len(NAMES)                     # la cintura gira al ángulo elegido antes del intento
NAMES.append("giro")

st = torch.full((N,), S_START, dtype=torch.long, device=dev)
t_st = torch.zeros(N, device=dev)
q_cmd = HOME.expand(N, AD).clone()                 # the arm target sent (term order)
lim_v = torch.zeros(N, AD, device=dev)
h_cmd = H_REACH.expand(N, HD).clone()
w_cmd = torch.zeros(N, device=dev)
w_ep = torch.zeros(N, device=dev)                  # this tree's waist yaw
prev_raw = torch.zeros(N, AD, device=dev)
grasp_prev = torch.zeros(N, HD, device=dev)
use_policy_hand = torch.zeros(N, dtype=torch.bool, device=dev)
chain = [None] * N           # joint-space quintic segments [(q0, q1, T)]
seg = [0] * N
legs = [None] * N            # arm-pose policy goals [(name, target, normal, finger, finger_on, via, advance)]
leg = [0] * N
t_leg = torch.zeros(N, device=dev)
path = [[] for _ in range(N)]
detach_path = [None] * N
tw = [None] * N
w_move = [None] * N
tries = [0] * N
deposited = [False] * N
outcome = [""] * N
log = [[] for _ in range(N)]
meas_bias = torch.zeros(N, 3, device=dev)
grip_since = torch.full((N,), float("nan"), device=dev)
held_since = torch.full((N,), float("nan"), device=dev)
sc_stopped = torch.zeros(N, 3, dtype=torch.bool, device=dev)
holding = torch.zeros(N, dtype=torch.bool, device=dev)       # scripted grip: force-regulated fingers
t_closed = torch.full((N,), float("nan"), device=dev)
twist_sign = torch.ones(N, device=dev)
ready_frame = [None] * N
apple_goal = [None] * N
n_deposits = 0
DIAG, SEG = {}, [None] * N
reaimed = [False] * N
REAIM_S = 2.5


def where(i):
    """The fruit in the palm frame minus the pocket (cm), its distance, the finger forces (N), stem state."""
    pq = robot.data.body_quat_w[i:i + 1, palm_id]
    rel = quat_apply_inverse(pq, (apple.data.root_pos_w[i] - robot.data.body_pos_w[i, palm_id]).view(1, 3))[0] - c_local[0]
    st_ = T().target_stem
    return (f"[fruit-pocket {np.round(rel.cpu().numpy() * 100, 1).tolist()} cm, |{float(rel.norm()) * 100:.1f}|; "
            f"fingers {np.round(finger_forces()[i].cpu().numpy(), 1).tolist()} N; stem "
            f"{'broken' if bool(st_.broken[i]) else ('on' if bool(st_.active[i]) else 'off')}]")


def note(i, msg):
    if args.detalle:
        print(f"  [env {i:2d} {float(sim_t):6.1f}s] {msg}", flush=True)


def go(i, s):
    st[i] = s
    t_st[i] = 0.0


def start_chain(i, points, speed=MOVE_SPEED, then=None):
    pts = [q_cmd[i].clone()] + list(points)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = float((b - a).abs().max())
        segs.append((a, b, max(0.5, 1.875 * d / speed, math.sqrt(5.77 * d / 1.5))))
    chain[i], seg[i] = segs, 0
    lim_v[i] = 0.0
    return segs


def chain_step(i):
    """-> True when the chain is done."""
    segs = chain[i]
    a, b, Tm = segs[seg[i]]
    f = float(hv.smoothstep(torch.tensor(float(t_st[i]) / Tm)))
    q_cmd[i] = a + (b - a) * f
    if float(t_st[i]) >= Tm:
        seg[i] += 1
        t_st[i] = 0.0
        if seg[i] >= len(segs):
            return True
    return False


def set_goal(i, g):
    name, tgt, nrm, fin, fon, via, adv = g
    P.target[i], P.normal[i], P.finger[i] = tgt, nrm, fin
    P.finger_on[i], P.via[i] = fon, via
    P.start[i] = pocket_now()[0][i]
    P.pending[i] = False
    P.hold[i] = 0.0
    P.success[i] = False
    P.arrived[i] = False
    t_leg[i] = 0.0


def start_legs(i, goals):
    legs[i], leg[i] = goals, 0
    lim_v[i] = 0.0
    set_goal(i, goals[0])


def aim_point(i):
    a = apple.data.root_pos_w[i]
    return a + meas_bias[i] if args.perceived_apple else a


def make_plan(i):
    """arm_pose_io.make_plan from READY: STAGING (advance rule), APPLE; and the way back."""
    tp, tq, sh = frames()
    a = aim_point(i)
    axis = tp[i]
    u = a - axis
    u[2] = 0.0
    u = u / u.norm().clamp(min=1e-6)
    d_goal = float(((a - axis) * u).sum())
    r = min(STAGE_RADIUS, d_goal - MIN_STAGE_BACK)
    stage = axis + r * u
    stage[2] = a[2]
    pk, nrm, fin = pocket_now()
    plan = [("staging", stage, u.clone(), fin[i].clone(), False, False, True),
            ("apple", a.clone(), u.clone(), fin[i].clone(), False, False, False)]
    ready_frame[i] = (pk[i].clone(), nrm[i].clone(), fin[i].clone())
    back = [("staging back", stage, u.clone(), fin[i].clone(), False, False, True)]
    if args.full_cycle:
        back.append(("ready", ready_frame[i][0], ready_frame[i][1], ready_frame[i][2], True, False, False))
    apple_goal[i] = a.clone()
    return plan, back


def legs_step(i, d, ang):
    """The arm-pose goal events (arm_pose_io.goal_event) -> None (going on), or "reached" / "timeout" of the LAST
    goal; an advance goal moves on to the next."""
    g = legs[i][leg[i]]
    adv = g[6]
    ev = ""
    if adv and float(d[i]) < STAGE_ADV_D and float(ang[i]) < math.radians(STAGE_ADV_DEG):
        ev = "passed"
    elif adv and float(t_leg[i]) >= STAGE_ADV_S / args.speed:
        ev = "timer"
    elif bool(P.arrived[i]):
        ev = "reached"
    elif float(t_leg[i]) >= WAYPOINT_TIMEOUT:
        ev = "timeout"
    if not ev:
        return None
    if leg[i] + 1 < len(legs[i]):
        leg[i] += 1
        set_goal(i, legs[i][leg[i]])
        return None
    return ev


def start_detach(i):
    """The approach's measured arm path backwards for pull_m of pocket travel (toy_harvest.replay_path)."""
    rec = path[i]
    qs = torch.stack([r[0] for r in rec]).flip(0)
    ps = torch.stack([r[1] for r in rec]).flip(0)
    dist = torch.cat([torch.zeros(1, device=dev), (ps[1:] - ps[:-1]).norm(dim=-1).cumsum(0)])
    k = int((dist < DETACH["pull_m"]).sum())
    k = max(2, min(k + 1, len(qs)))
    L = float(dist[k - 1]) if k - 1 < len(dist) else float(dist[-1])
    Tm = max(1.0, 1.875 * max(L, 0.02) / DETACH["pull_speed_m_s"])
    # ONE smooth joint move to the pose 8 cm back along the approach (replaying the recorded path replayed the
    # policy's jitter too: 1.6 rad/s joint speed on a 0.75 rad/s pull)
    detach_path[i] = (q_cmd[i].clone(), qs[k - 1].clone(), Tm)


def detach_step(i):
    q0, q1, Tm = detach_path[i]
    f = float(hv.smoothstep(torch.tensor(min(float(t_st[i]) / Tm, 1.0))))
    q_cmd[i] = q0 + (q1 - q0) * f
    return float(t_st[i]) >= Tm


def start_twist(i):
    pq = robot.data.body_quat_w[i, palm_id]
    pp = robot.data.body_pos_w[i, palm_id]
    axis = quat_apply(pq.view(1, 4), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
    pivot = pp + quat_apply(pq.view(1, 4), c_local[:1])[0]
    deg = float(TWIST["deg"])
    Tm = max(0.6, 1.875 * math.radians(deg) / float(TWIST["speed_rad_s"]))
    tw[i] = dict(p0=pp.clone(), q0=pq.clone(), axis=axis, pivot=pivot, rad=math.radians(deg), T=Tm,
                 hold=float(TWIST["hold_s"]), q_start=q_cmd[i].clone())


def twist_target(ids):
    """Desired palm poses for the twisting envs (out, hold, back)."""
    pos = robot.data.body_pos_w[:, palm_id].clone()
    quat = robot.data.body_quat_w[:, palm_id].clone()
    p0 = pos.clone(); q0 = quat.clone(); piv = pos.clone(); ax = torch.zeros(N, 3, device=dev); ang = torch.zeros(N, device=dev)
    ax[:, 1] = 1.0
    for i in ids:
        w = tw[i]
        t = float(t_st[i])
        if t < w["T"]:
            f = float(hv.smoothstep(torch.tensor(t / w["T"])))
        elif t < w["T"] + w["hold"]:
            f = 1.0
        else:
            f = 1.0 - float(hv.smoothstep(torch.tensor((t - w["T"] - w["hold"]) / w["T"])))
        p0[i], q0[i], piv[i], ax[i] = w["p0"], w["q0"], w["pivot"], w["axis"]
        ang[i] = float(twist_sign[i]) * w["rad"] * f
    zero = torch.zeros(N, device=dev)
    return hv.harvest_palm_pose(p0, q0, piv, ax, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(N, 3), zero, ang)


def start_waist(i, target):
    w0 = float(w_cmd[i])
    d = abs(target - w0)
    w_move[i] = (w0, target, max(0.6, 1.875 * d / WAIST_SPEED))


def waist_step(i):
    w0, w1, Tm = w_move[i]
    f = float(hv.smoothstep(torch.tensor(float(t_st[i]) / Tm)))
    w_cmd[i] = w0 + (w1 - w0) * f
    return float(t_st[i]) >= Tm + 0.3


# neighbours parked out of sight (2..5 apples per tree: the scene holds args.apples[1])
parked = torch.zeros(N, args.apples[1], dtype=torch.bool, device=dev)
PARK = torch.tensor([0.0, 0.0, -1.0], device=dev)


def choose_apples(i):
    """This tree's apple count: keep the target and a random n-1 of its neighbours, park the rest."""
    M = T().n
    n = int(torch.randint(args.apples[0], args.apples[1] + 1, (1,)))
    pos = torch.stack([o.data.root_pos_w[i] for o in T().objs])
    r = float(env._apple_radius[i]) if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) else 0.04
    gap = (pos[1:] - pos[0]).norm(dim=-1) - 2 * r                     # neighbour k+1 -> target, surface gap
    far = [k + 1 for k in torch.randperm(M - 1).tolist() if float(gap[k]) >= args.apple_gap]
    keep = far[:n - 1] or [int(gap.argmax()) + 1]
    parked[i] = False
    for k in range(1, M):
        if k not in keep:
            parked[i, k] = True
            T().stems[k].release(torch.tensor([i], device=dev))
    n = len(keep) + 1
    return n


def weaken_target(i):
    """Once the hand holds it: the target's pull limit within the sim grip's reach (until then it keeps the tree's
    15-25 N, so the reach does not knock it off)."""
    if args.stem_break_pull[1] > 0:
        lo, hi = args.stem_break_pull
        T().target_stem.f_break[i] = lo + (hi - lo) * float(torch.rand(1))
        lo, hi = args.target_bend
        T().target_stem.m_bend_break[i] = lo + (hi - lo) * float(torch.rand(1))


def park_apples():
    for k in range(1, T().n):
        ids = parked[:, k].nonzero(as_tuple=True)[0]
        if len(ids):
            o = T().objs[k]
            stt = o.data.default_root_state[ids].clone()
            stt[:, :3] = env.scene.env_origins[ids] + PARK + torch.tensor([0.0, 0.12 * k, 0.0], device=dev)
            stt[:, 7:] = 0.0
            o.write_root_state_to_sim(stt, env_ids=ids)


# ═════════════════════ dataset (proyecto ML) ═════════════════════
CSV_COLS = [
    # identificación
    "episodio", "semilla", "env", "t_inicio_s",
    # entradas (conocidas antes de intentar)
    "objetivo_visible", "yaw_cintura_deg",
    "x_hombro_m", "y_hombro_m", "z_hombro_m", "dist_hombro_m", "dist_horiz_m", "azimut_deg", "elevacion_deg",
    "u_px", "v_px", "prof_cam_m", "radio_manzana_m",
    "n_vecinos_vis", "dist_vecino_min_m", "vec_dx_m", "vec_dy_m", "vec_dz_m", "n_vecinos_vis_15cm",
    "n_vecinos_vis_debajo",
    # informativas (NO son entradas: el robot no las conoce)
    "n_manzanas_arbol", "n_vecinos_ocultos",
    # salidas / etiquetas
    "exito", "etapa", "estado_fallo", "motivo", "error_alcance_cm", "duracion_s",
    # experimento de la cintura
    "modo_cintura", "yaw_inicial_deg", "yaw_elegido_deg", "p_inicial", "p_elegido", "p_max", "yaw_max_deg",
    "n_giros_visibles",
]
ep_cint = [None] * N
ep_feat = [None] * N
ep_fail_state = [""] * N
ep_reason = [""] * N
ep_inb = [None] * N
ep_err = [float("nan")] * N
ep_t0 = [0.0] * N
ep_recorded = [True] * N           # True: nothing to write for this env until its next attempt starts


def _torso_yaw_frame(i, v):
    """World vector v [3] in env i's torso yaw frame (x forward, y left, z up)."""
    tq = robot.data.body_quat_w[i:i + 1, torso_id]
    return quat_apply_inverse(yaw_quat(tq), v.view(1, 3))[0]


def _pixel(i, p_w):
    """(u, v, depth) of world point p_w [3] in the head camera (D435i 424x240, the same model as the
    visibility test: tree turned back to waist yaw 0 where env._camera_view0 holds)."""
    from envs.mdp import visible_reach as vr
    view = env._camera_view0
    p0 = vr._unturn(env, torch.tensor([i], device=dev), p_w.view(1, 1, 3))[0, 0]
    pos = torch.tensor(view["pos"], dtype=torch.float32, device=dev)
    R = torch.tensor(view["R"], dtype=torch.float32, device=dev)
    pc = (p0 - pos) @ R
    z = float(pc[2])
    zz = max(z, 1e-6)
    return view["cx"] + view["fx"] * float(pc[0]) / zz, view["cy"] - view["fy"] * float(pc[1]) / zz, z


def episode_features(i, target_seen):
    """Las entradas del modelo para el intento que empieza en el env i (solo lo que el robot puede saber)."""
    from envs.mdp import visible_reach as vr
    ids = torch.tensor([i], device=dev)
    tp, tq, sh = frames()
    a = aim_point(i)                                       # la manzana tal como el robot la apunta
    rel = _torso_yaw_frame(i, a - sh[i])
    x, y, z = (float(c) for c in rel)
    dh = math.hypot(x, y)
    u, v, depth = _pixel(i, a)
    r = float(env._apple_radius[i]) if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) else 0.04
    # vecinos: los que siguen en el árbol (no estacionados) y que la cámara VE ENTEROS
    vis, hidden = [], 0
    for k in range(1, T().n):
        if bool(parked[i, k]):
            continue
        pk_ = T().objs[k].data.root_pos_w[i]
        if bool(vr._sees_yaw0(env, ids, pk_.view(1, 1, 3))[0, 0]):
            vis.append(_torso_yaw_frame(i, pk_ - a))
        else:
            hidden += 1
    f = dict(objetivo_visible=int(bool(target_seen)), yaw_cintura_deg=math.degrees(float(w_ep[i])),
             x_hombro_m=x, y_hombro_m=y, z_hombro_m=z, dist_hombro_m=math.sqrt(x * x + y * y + z * z),
             dist_horiz_m=dh, azimut_deg=math.degrees(math.atan2(y, x)), elevacion_deg=math.degrees(math.atan2(z, dh)),
             u_px=u, v_px=v, prof_cam_m=depth, radio_manzana_m=r,
             n_vecinos_vis=len(vis), n_manzanas_arbol=1 + len(vis) + hidden, n_vecinos_ocultos=hidden)
    if vis:
        d = [float(w.norm()) for w in vis]
        k = int(np.argmin(d))
        f.update(dist_vecino_min_m=d[k], vec_dx_m=float(vis[k][0]), vec_dy_m=float(vis[k][1]),
                 vec_dz_m=float(vis[k][2]), n_vecinos_vis_15cm=sum(dd < 0.15 for dd in d),
                 n_vecinos_vis_debajo=sum(float(w[2]) < -0.02 and math.hypot(float(w[0]), float(w[1])) < 0.15
                                          for w in vis))
    else:
        f.update(dist_vecino_min_m=float("nan"), vec_dx_m=float("nan"), vec_dy_m=float("nan"),
                 vec_dz_m=float("nan"), n_vecinos_vis_15cm=0, n_vecinos_vis_debajo=0)
    return f


STAGE = {"approach": "alcance", "pregrasp": "agarre", "grasp": "agarre", "twist": "desprendimiento",
         "detach": "desprendimiento", "return": "retorno", "check": "retorno", "to_home": "transporte",
         "waist_dep": "transporte", "to_dep": "transporte", "deposit": "transporte"}


def episode_row(i, timed_out=False):
    if timed_out:
        etapa, estado, motivo, ok = "tiempo", NAMES[int(st[i])], f"sin terminar en {args.ep_max_s:g} s", 0
    elif ep_inb[i] is True:
        etapa, estado, motivo, ok = "exito", "", "depositada en la canasta", 1
    elif ep_inb[i] is False:
        etapa, estado, motivo, ok = "deposito", "release", "soltada fuera de la canasta", 0
    else:
        estado = ep_fail_state[i] or NAMES[int(st[i])]
        etapa, motivo, ok = STAGE.get(estado, "otro"), ep_reason[i] or "?", 0
    row = dict(episodio=n_rows, semilla=args.seed, env=i, t_inicio_s=round(ep_t0[i], 2), **ep_feat[i],
               exito=ok, etapa=etapa, estado_fallo=estado, motivo=motivo,
               error_alcance_cm=round(ep_err[i] * 100, 2) if not math.isnan(ep_err[i]) else float("nan"),
               duracion_s=round(sim_t - ep_t0[i], 2), **ep_cint[i])
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}


out_path = os.path.abspath(os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
_new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
csv_f = open(out_path, "a", newline="")
csv_w = csv.DictWriter(csv_f, fieldnames=CSV_COLS)
if _new_file:
    csv_w.writeheader()
n_rows = 0
n_ok = 0
from collections import Counter as _Counter  # noqa: E402
stage_count = _Counter()
print(f"[dataset] escribiendo en {out_path} ({'nuevo' if _new_file else 'agregando filas'})", flush=True)


def record_and_next_tree(i, timed_out=False):
    """Guarda la fila del intento del env i y le pide un árbol nuevo."""
    global n_rows, n_ok
    if not ep_recorded[i] and ep_feat[i] is not None and n_rows < args.episodes:
        row = episode_row(i, timed_out)
        csv_w.writerow(row)
        csv_f.flush()
        n_rows += 1
        n_ok += row["exito"]
        stage_count[row["etapa"]] += 1
    ep_recorded[i] = True
    T().new_tree[i] = True
    T().next_request[i] = True
    go(i, S_START)


# ═════════════════════ experimento: elegir el giro de cintura ═════════════════════
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(REPO, "ml_project"))
from bosque_np import BosqueNP  # noqa: E402
from envs.mdp import visible_reach as vr  # noqa: E402

BOSQUE = BosqueNP(os.path.join(REPO, args.modelo) if not os.path.isabs(args.modelo) else args.modelo)
GIROS = np.arange(args.giro_min_deg, args.giro_max_deg + 1e-6, args.giro_paso_deg)
rng_giro = np.random.default_rng(args.seed + 1000)
n_check = 0


def _yaw_de(q):
    """Ángulo de guiñada (rad) de un cuaternión wxyz."""
    w, x, y, z = (float(c) for c in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _a_yaw0(i, p_w, psi):
    """Puntos del mundo [K, 3] -> marco del robot con la cintura en 0 (env-local), SI la cintura estuviera en
    psi (rad) y el árbol quieto: la misma transformación que usa la prueba de visibilidad (vr._unturn)."""
    old = env._fixed_waist_yaw[i].clone()
    env._fixed_waist_yaw[i] = float(psi)
    out = vr._unturn(env, torch.tensor([i], device=dev), p_w.view(1, -1, 3))[0]
    env._fixed_waist_yaw[i] = old
    return out


def features_at(i, psi):
    """(ve_objetivo, entradas) del intento del env i si la cintura girara a psi (rad), sin mover nada:
    replica episode_features() pero en el marco 'cintura en 0' (el hombro, el torso y la cámara son fijos ahí;
    lo que se mueve es el árbol, girado por -psi)."""
    tp, tq, sh = frames()
    psi_now = float(env._fixed_waist_yaw[i])
    sh0 = _a_yaw0(i, sh[i], psi_now)[0]
    th0 = _yaw_de(tq[i]) - float(robot.data.joint_pos[i, wj])       # yaw del torso con la cintura en 0
    c0, s0 = math.cos(-th0), math.sin(-th0)

    def al_torso(v):                                                  # vector del marco yaw-0 al del torso
        return torch.stack([c0 * v[0] - s0 * v[1], s0 * v[0] + c0 * v[1], v[2]])

    view = env._camera_view0
    pos = torch.tensor(view["pos"], dtype=torch.float32, device=dev)
    R = torch.tensor(view["R"], dtype=torch.float32, device=dev)
    a0 = _a_yaw0(i, aim_point(i), psi)[0]
    seen = bool(vr.sees_t(_a_yaw0(i, apple.data.root_pos_w[i], psi).view(1, 1, 3), pos, R, view)[0, 0])
    rel = al_torso(a0 - sh0)
    x, y, z = (float(c) for c in rel)
    dh = math.hypot(x, y)
    pc = (a0 - pos) @ R
    zc = max(float(pc[2]), 1e-6)
    u, v = view["cx"] + view["fx"] * float(pc[0]) / zc, view["cy"] - view["fy"] * float(pc[1]) / zc
    vis = []
    for k in range(1, T().n):
        if bool(parked[i, k]):
            continue
        pk0 = _a_yaw0(i, T().objs[k].data.root_pos_w[i], psi)[0]
        if bool(vr.sees_t(pk0.view(1, 1, 3), pos, R, view)[0, 0]):
            vis.append(al_torso(pk0 - a0))
    f = dict(yaw_cintura_deg=math.degrees(psi), x_hombro_m=x, y_hombro_m=y, z_hombro_m=z,
             dist_hombro_m=math.sqrt(x * x + y * y + z * z), dist_horiz_m=dh,
             azimut_deg=math.degrees(math.atan2(y, x)), elevacion_deg=math.degrees(math.atan2(z, dh)),
             u_px=u, v_px=v, prof_cam_m=float(pc[2]), n_vecinos_vis=len(vis))
    if vis:
        d = [float(w.norm()) for w in vis]
        k = int(np.argmin(d))
        f.update(dist_vecino_min_m=d[k], vec_dx_m=float(vis[k][0]), vec_dy_m=float(vis[k][1]),
                 vec_dz_m=float(vis[k][2]),
                 n_vecinos_vis_debajo=sum(float(w[2]) < -0.02 and math.hypot(float(w[0]), float(w[1])) < 0.15
                                          for w in vis))
    else:
        f.update(dist_vecino_min_m=float("nan"), vec_dx_m=float("nan"), vec_dy_m=float("nan"),
                 vec_dz_m=float("nan"), n_vecinos_vis_debajo=0)
    return seen, f


def elegir_giro(i):
    """Evalúa los giros candidatos con el bosque y devuelve (giro elegido en rad, info para el CSV)."""
    global n_check
    psi0 = float(env._fixed_waist_yaw[i])
    s0, f0 = features_at(i, psi0)
    p0 = float(BOSQUE.proba(BOSQUE.vector(f0))[0])
    if n_check < 3:                                   # control: con el giro actual = episode_features()
        n_check += 1
        ref = episode_features(i, s0)
        dif = max(abs(float(f0[k]) - float(ref[k])) for k in f0
                  if not (math.isnan(float(f0[k])) and math.isnan(float(ref[k]))))
        print(f"[cintura] control de entradas recalculadas (env {i}): diferencia máx. {dif:.2e}", flush=True)
    cand, probs = [], []
    for g in GIROS:
        psi = math.radians(float(g))
        seen, f = features_at(i, psi)
        if seen:
            cand.append(psi)
            probs.append(float(BOSQUE.proba(BOSQUE.vector(f))[0]))
    if cand:
        kmax = int(np.argmax(probs))
        psi_max, p_max = cand[kmax], probs[kmax]
    else:
        psi_max, p_max = psi0, p0
    if args.cintura == "modelo" and cand:
        psi_e, p_e = psi_max, p_max
    elif args.cintura == "aleatorio" and cand:
        k = int(rng_giro.integers(len(cand)))
        psi_e, p_e = cand[k], probs[k]
    else:
        psi_e, p_e = psi0, p0
    info = dict(modo_cintura=args.cintura, yaw_inicial_deg=math.degrees(psi0), yaw_elegido_deg=math.degrees(psi_e),
                p_inicial=p0, p_elegido=p_e, p_max=p_max, yaw_max_deg=math.degrees(psi_max),
                n_giros_visibles=len(cand))
    return psi_e, info


def begin_apple(i, new_tree):
    """At HOME, the hand open: (re)start the harvest of this env's apple."""
    if new_tree:
        note(i, f"new tree: {int((~parked[i, 1:]).sum()) + 1} apples, neighbours >= {args.apple_gap * 100:.0f} cm away")
        tries[i] = 0
        from envs.mdp import visible_reach as vr
        seen = bool(vr._sees_yaw0(env, torch.tensor([i], device=dev), apple.data.root_pos_w[i].view(1, 1, 3))[0, 0])
        note(i, f"target seen whole by the head camera at this waist: {seen}")
        w_ep[i] = env._fixed_waist_yaw[i]
        w_cmd[i] = w_ep[i]
        m = getattr(env, "_apple_meas", None)
        meas_bias[i] = (m[i] - apple.data.root_pos_w[i]) if m is not None else 0.0
        outcome[i] = ""
        deposited[i] = False
        ep_fail_state[i], ep_reason[i], ep_inb[i], ep_err[i] = "", "", None, float("nan")
        ep_t0[i] = sim_t
        ep_recorded[i] = False
        psi_e, ep_cint[i] = elegir_giro(i)
        if abs(psi_e - float(w_ep[i])) > math.radians(0.5):
            # el árbol queda quieto; gira el robot (la prueba de visibilidad y el seguimiento usan este giro)
            env._fixed_waist_yaw[i] = psi_e
            w_ep[i] = psi_e
            start_waist(i, psi_e)
            note(i, f"CINTURA ({args.cintura}): {ep_cint[i]['yaw_inicial_deg']:+.1f} -> {math.degrees(psi_e):+.1f} deg "
                    f"(P {ep_cint[i]['p_inicial']:.2f} -> {ep_cint[i]['p_elegido']:.2f})")
            go(i, S_GIRO)
            return
        ep_feat[i] = episode_features(i, seen)
    tries[i] += 1
    path[i] = []
    reaimed[i] = False
    use_policy_hand[i] = False
    holding[i] = False
    h_cmd[i] = H_REACH
    note(i, f"apple try {tries[i]} (waist {math.degrees(float(w_ep[i])):+.0f} deg): HOME -> READY")
    start_chain(i, [VIA, READY])
    go(i, S_TO_READY)


def fail(i, why):
    """No apple in the hand: open, back to READY by the policy if needed, HOME, then the next try / tree."""
    note(i, f"MISS: {why}")
    if not ep_fail_state[i]:
        ep_fail_state[i] = NAMES[int(st[i])]
        ep_reason[i] = why
    h_cmd[i] = H_OPEN
    use_policy_hand[i] = False
    holding[i] = False
    outcome[i] = why


def finish_fail(i):
    if args.one_try:
        note(i, "episode over (miss)")
        go(i, S_DONE)
        return
    on_tree = bool(T().target_stem.active[i]) and not bool(T().target_stem.broken[i])
    if on_tree and tries[i] < args.max_tries:
        begin_apple(i, new_tree=False)
    else:
        note(i, "the fruit is off the tree (or out of tries): a new tree")
        T().new_tree[i] = True
        T().next_request[i] = True
        go(i, S_START)


# ── video ──
video = None
if args.video:
    centre = env.scene.env_origins.mean(0).cpu().numpy()
    spread = float(env.scene.env_origins[:, 1].max() - env.scene.env_origins[:, 1].min())
    eye = args.cam_eye or (-2.2 - 0.3 * spread, -3.0 - 0.9 * spread, 2.4 + 0.45 * spread)
    look = args.cam_lookat or (0.4, -0.3, 0.6)
    env.viewport_camera_controller.update_view_location(
        eye=tuple(float(c + e) for c, e in zip(centre, eye)), lookat=tuple(float(c + l) for c, l in zip(centre, look)))
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    video = imageio.get_writer(args.video, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=8)
    video_every = max(1, round(1.0 / (args.video_fps * dt)))

torch.manual_seed(args.seed)
obs, _ = env.reset()

# ── perfil (--perfil): tiempo por sección, con la GPU sincronizada en cada marca ──
_perf, _perf_t = {}, [time.perf_counter()]


def _mark(name):
    if not args.perfil:
        return
    torch.cuda.synchronize()
    now = time.perf_counter()
    _perf[name] = _perf.get(name, 0.0) + now - _perf_t[0]
    _perf_t[0] = now


if args.perfil:
    _apply_orig = env._tree.apply

    def _apply_medido():                               # pedúnculos: corre en CADA subpaso de física
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _apply_orig()
        torch.cuda.synchronize()
        _perf["  (dentro de env.step) pedúnculos"] = _perf.get("  (dentro de env.step) pedúnculos", 0.0) + \
            time.perf_counter() - t0

    env._tree.apply = _apply_medido
w_cmd = env._fixed_waist_yaw.clone()
for _i in range(N):
    choose_apples(_i)
sim_t = 0.0
step = 0
t_all_done = None
wall0 = time.time()
grip_level = torch.full((N,), args.grip_n, device=dev)

while app.is_running():
    with torch.inference_mode():
        _mark("registro del dataset / progreso")
        # inputs for this step
        env._target_grip_level = grip_level.clone()
        phase_ok = env._grasp_phase_active.clone()
        env._stem_released[:] = True                       # the whole arm follows the commands
        d, ang, _, _ = ap.errors(env, POCKET)

        # B, once (waist 0 is not where the robots stand: computed from the shoulder rotated back)
        if B_LOCAL is None and bool(phase_ok.all()):
            tp, tq, sh = frames()
            w = robot.data.joint_pos[0, wj]
            # rotate the shoulder about the waist axis (vertical through the pelvis) back to waist 0
            c, s_ = math.cos(-float(w)), math.sin(-float(w))
            piv = robot.data.body_pos_w[0, robot.data.body_names.index("pelvis")]
            rel = sh[0] - piv
            sh0 = piv + torch.tensor([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1], rel[2]], device=dev)
            # torso frame at waist 0 = pelvis frame (fixed base, identity rotation): torso x/y/z = world
            B_LOCAL = sh0 + torch.tensor(BASKET["from_shoulder_m"], device=dev) - env.scene.env_origins[0]
            print(f"[harvest sim] basket point (env-local) {np.round(B_LOCAL.cpu().numpy(), 3).tolist()}", flush=True)
            if args.probe:
                for yd in (0.0, -15.0):
                    w_cmd[0] = math.radians(yd)
                    robot.set_joint_position_target(w_cmd.view(N, 1), joint_ids=[wj])
                    for _ in range(60):
                        env.step(torch.zeros(N, AD + HD, device=dev))
                    drop, _, _, short = basket_drop(0)
                    print(f"[probe] waist {yd:+.0f}: drop (env-local) "
                          f"{np.round((drop - env.scene.env_origins[0]).cpu().numpy(), 3).tolist()}, {short * 100:.1f} cm short",
                          flush=True)
                os._exit(0)

        # policy-leg arm targets for every env in a leg
        in_leg = torch.tensor([s in (S_APPROACH, S_RETURN, S_DEPOSIT, S_BACK) for s in st.tolist()], device=dev)
        if bool(in_leg.any()):
            ro = ga._reach_obs(env, arm._reach_arm_cfg, arm._palm_cfg, arm._apple_cfg, arm.cfg.palm_local_offset,
                               prev_raw, layout="arm_pose", waist={"pocket_local": POCKET})
            raw = arm_pol(ro)
            desired = torch.clamp(raw * 0.5 + ARM_DEFAULT, CLIP_LO, CLIP_HI)
            vmax, amax = ARM_VMAX * args.speed, ARM_AMAX * args.speed ** 2
            v = torch.clamp((desired - q_cmd) / dt, -vmax, vmax)
            err = (desired - q_cmd).abs()
            v_stop = -0.5 * amax * dt + torch.sqrt((0.5 * amax * dt) ** 2 + 2.0 * amax * err)
            v = torch.maximum(torch.minimum(v, v_stop), -v_stop)
            v = torch.clamp(v, lim_v - amax * dt, lim_v + amax * dt)
            m = in_leg.unsqueeze(-1)
            q_cmd = torch.where(m, q_cmd + v * dt, q_cmd)
            lim_v = torch.where(m, v, lim_v)
            prev_raw = torch.where(m, raw, prev_raw)

        # twist: IK on the commanded targets
        tw_ids = (st == S_TWIST).nonzero(as_tuple=True)[0].tolist()
        if tw_ids:
            pp, pq = twist_target(tw_ids)
            q0 = torch.stack([tw[i]["q_start"] if i in tw_ids else q_cmd[i] for i in range(N)])
            q_ik = ik.step(pp, pq, base=q_cmd, max_dq=0.03, null_target=q0, null_gain=0.2)
            for i in tw_ids:
                q_cmd[i] = q_ik[i]

        # actions
        a = torch.zeros(N, AD + HD, device=dev)
        a[:, :AD] = (q_cmd - arm._anchor) / arm._scale
        h_raw = (h_cmd - hand._grasp_offset) / hand._scale
        gp = use_policy_hand
        if bool(gp.any()):
            graw = grasp_pol(obs["grasp"])[:, AD:AD + HD]
            h_raw = torch.where(gp.unsqueeze(-1), graw, h_raw)
            grasp_prev = torch.where(gp.unsqueeze(-1), graw, grasp_prev)
        a[:, AD:] = h_raw
        hand.speed_scale[:] = torch.where(gp, torch.full_like(w_cmd, args.speed), torch.ones_like(w_cmd)).unsqueeze(-1)
        a = torch.where(phase_ok.unsqueeze(-1), a, torch.zeros_like(a))
        robot.set_joint_position_target(w_cmd.view(N, 1), joint_ids=[wj])
        if getattr(arm, "_waist_id", None) is not None:
            arm.waist_override[:] = w_cmd                  # else the arm term holds the waist where it was
        park_apples()
        if args.diag:
            st_prev = st.clone()
            palm_prev = robot.data.body_pos_w[:, palm_id].clone()

        _mark("política del brazo + acciones")
        obs, _, term, trunc, _ = env.step(a)
        _mark("env.step (física + gestores)")
        step += 1
        sim_t += dt
        t_st += dt
        t_leg += dt
        if video is not None and step % video_every == 0:
            frame = env.render()
            if frame is not None:
                video.append_data(frame)

        if args.diag:
            qd = robot.data.joint_vel[:, arm_ids].abs().amax(-1)
            lag = (robot.data.joint_pos[:, arm_ids] - q_cmd).abs().amax(-1)
            werr = (robot.data.joint_pos[:, wj] - w_cmd).abs()
            dp = (robot.data.body_pos_w[:, palm_id] - palm_prev).norm(dim=-1)
            for i in range(N):
                k = int(st_prev[i])
                D = DIAG.setdefault(NAMES[k], dict(qd=0.0, lag=0.0, werr=0.0, segs=[]))
                D["qd"] = max(D["qd"], float(qd[i])); D["lag"] = max(D["lag"], float(lag[i]))
                D["werr"] = max(D["werr"], float(werr[i]))
                if SEG[i] is None or SEG[i][0] != k:
                    if SEG[i] is not None:
                        DIAG[NAMES[SEG[i][0]]]["segs"].append((SEG[i][1], float((palm_prev[i] - SEG[i][2]).norm())))
                    SEG[i] = [k, 0.0, palm_prev[i].clone()]
                SEG[i][1] += float(dp[i])

        # resets (a new tree): wait for the hand-off, then start
        reset = (term | trunc)
        for i in reset.nonzero(as_tuple=True)[0].tolist():
            st[i] = S_START
            t_st[i] = 0.0
            q_cmd[i] = robot.data.joint_pos[i, arm_ids]
            h_cmd[i] = H_REACH
            use_policy_hand[i] = False
            holding[i] = False
            w_cmd[i] = env._fixed_waist_yaw[i]
            choose_apples(i)

        if args.diag and step % 5 == 0:
            for i in range(N):
                if int(st[i]) in (S_DETACH, S_RETURN, S_CHECK):
                    lg = legs[i][leg[i]][0] if int(st[i]) == S_RETURN else "-"
                    print(f"    trace env {i} {sim_t:5.2f} {NAMES[int(st[i])]:7s} {lg:12s} "
                          f"lag {float((robot.data.joint_pos[i, arm_ids] - q_cmd[i]).abs().max()):.2f} "
                          f"qd {float(robot.data.joint_vel[i, arm_ids].abs().max()):.2f} "
                          f"cmdv {float(lim_v[i].abs().max()):.2f} "
                          f"palm v {float(robot.data.body_lin_vel_w[i, palm_id].norm()):.2f} "
                          f"fruit {float(fruit_to_pocket()[i]) * 100:.1f}cm f {np.round(finger_forces()[i].cpu().numpy(), 0).tolist()} "
                          f"stem {'B' if bool(T().target_stem.broken[i]) else 'on'} "
                          f"q {np.round(robot.data.joint_pos[i, arm_ids].cpu().numpy(), 2).tolist()} "
                          f"cmd {np.round(q_cmd[i].cpu().numpy(), 2).tolist()}", flush=True)
        ap.track_goal(env, POCKET, **TRACK)
        d, ang, _, _ = ap.errors(env, POCKET)
        pk, nrm, fin = pocket_now()
        ff = finger_forces()
        if bool(holding.any()):
            lo_f, hi_f = 0.8 * args.hold_n, 1.3 * args.hold_n
            rate = torch.where(ff < lo_f, args.close_speed * dt,
                               torch.where(ff > hi_f, -0.3 * args.close_speed * dt, 0.0))      # [N, 3] per group
            lo_q = torch.minimum(H_OPEN, _goal + _dir * args.squeeze)
            hi_q = torch.maximum(H_OPEN, _goal + _dir * args.squeeze)
            h_new = torch.maximum(torch.minimum(h_cmd + (rate @ _gmask) * _dir, hi_q), lo_q)
            h_cmd = torch.where(holding.unsqueeze(-1), h_new, h_cmd)

        _mark("resets + seguimiento + agarre")
        carrying = (st == S_TO_HOME) | (st == S_WAIST_DEP) | (st == S_TO_DEP) | (st == S_DEPOSIT)
        for i in (carrying & (fruit_to_pocket() > LEAVE)).nonzero(as_tuple=True)[0].tolist():
            fail(i, f"dropped on the way to the basket ({NAMES[int(st[i])]})")
            start_chain(i, [HOME])
            go(i, S_DEP_HOME)
        st_l, t_l = st.tolist(), t_st.tolist()           # una sola copia GPU->CPU por paso (no 2 por env)
        for i in range(N):
            s = st_l[i]
            t = t_l[i]
            if s == S_START:
                if bool(env._grasp_phase_active[i]) and t >= 0.5:
                    q_cmd[i] = arm._prev_target[i].clone() if hasattr(arm, "_prev_target") else robot.data.joint_pos[i, arm_ids]
                    begin_apple(i, new_tree=True)
            elif s == S_GIRO:
                if waist_step(i):
                    # entradas reales (ya girado) para el CSV, y el intento empieza como siempre
                    seen = bool(vr._sees_yaw0(env, torch.tensor([i], device=dev),
                                              apple.data.root_pos_w[i].view(1, 1, 3))[0, 0])
                    ep_feat[i] = episode_features(i, seen)
                    begin_apple(i, new_tree=False)
                    tries[i] = 1
            elif s in (S_TO_READY, S_TO_HOME, S_TO_DEP, S_DEP_HOME, S_FAILHOME):
                w_ok = True
                if s == S_TO_DEP and w_move[i] is not None and len(w_move[i]) == 4:
                    w0, w1, Tw, tw0 = w_move[i]
                    tw0 += dt
                    w_move[i] = (w0, w1, Tw, tw0)
                    w_cmd[i] = w0 + (w1 - w0) * float(hv.smoothstep(torch.tensor(min(tw0 / Tw, 1.0))))
                    w_ok = tw0 >= Tw + 0.3
                if (seg[i] < len(chain[i]) and chain_step(i)) or (seg[i] >= len(chain[i]) and w_ok):
                    if not w_ok:
                        continue
                    if s == S_TO_READY:
                        plan, back = make_plan(i)
                        legs_back = back
                        start_legs(i, plan)
                        tw[i] = dict(back=legs_back)
                        note(i, "APPROACH: arm-pose policy, staging -> apple")
                        go(i, S_APPROACH)
                    elif s == S_TO_HOME:
                        tgt = min(max(float(w_ep[i]), math.radians(BASKET["waist_range_deg"][0])),
                                  math.radians(BASKET["waist_range_deg"][1]))
                        start_waist(i, tgt)
                        note(i, f"DEPOSIT: the waist to {math.degrees(tgt):+.0f} deg")
                        go(i, S_WAIST_DEP)
                    elif s == S_TO_DEP:
                        drop, n_, f_, short = basket_drop(i)
                        start_legs(i, [("basket", drop, n_, f_, True, False, False)])
                        dep_back = (pk[i].clone(), nrm[i].clone(), fin[i].clone())
                        tw[i] = dict(dep_back=dep_back)
                        note(i, f"DEPOSIT: arm-pose policy over the basket, palm down ({short * 100:.1f} cm short)")
                        go(i, S_DEPOSIT)
                    elif s == S_DEP_HOME:
                        start_waist(i, float(w_ep[i]))
                        go(i, S_WAIST_BACK)
                    elif s == S_FAILHOME:
                        finish_fail(i)
            elif s == S_APPROACH:
                path[i].append((q_cmd[i].clone(), pk[i].clone()))
                ev = legs_step(i, d, ang)
                if ev is not None and args.reaim and not reaimed[i] and bool(T().target_stem.active[i]) \
                        and not bool(T().target_stem.broken[i]) and float(fruit_to_pocket()[i]) > 0.025:
                    # the camera sees where the apple is NOW (pushed / the reach stopped short): one more policy leg
                    reaimed[i] = True
                    a_now = aim_point(i)
                    note(i, f"RE-AIM: the apple is {float(fruit_to_pocket()[i]) * 100:.1f} cm from the pocket ({ev})")
                    g = legs[i][-1]
                    start_legs(i, [("apple again", a_now.clone(), g[2], g[3], False, False, False)])
                    t_leg[i] = WAYPOINT_TIMEOUT - REAIM_S
                    apple_goal[i] = a_now.clone()
                    ev = None
                if ev is not None:
                    err = float((pk[i] - apple_goal[i]).norm())
                    ep_err[i] = err
                    back = tw[i]["back"]
                    tw[i] = dict(back=back)
                    if err > GRASP_MAX_ERR:
                        fail(i, f"the hand ended {err * 100:.1f} cm from the apple goal ({ev})")
                        start_legs(i, back)
                        go(i, S_RETURN)
                    else:
                        mv = apple.data.root_pos_w[i] - apple_goal[i]
                        u_ = P.normal[i]
                        side = torch.linalg.cross(UP, u_)
                        side = side / side.norm().clamp(min=1e-6)
                        note(i, f"  apple moved since the plan: along the approach {float((mv * u_).sum()) * 100:+.1f} cm, "
                                f"side {float((mv * side).sum()) * 100:+.1f}, up {float(mv[2]) * 100:+.1f}; "
                                f"pocket-goal {np.round(((pk[i] - apple_goal[i]) * 100).cpu().numpy(), 1).tolist()} cm")
                        note(i, f"apple goal {ev}, {err * 100:.1f} cm: PREGRASP {where(i)}")
                        h_cmd[i] = H_CUP
                        go(i, S_PREGRASP)
            elif s == S_PREGRASP:
                if t >= float(REACH_HAND.get("cup_s", 1.0)):
                    note(i, f"GRASP ({args.grasp}) after the cup {where(i)}")
                    grip_since[i] = float("nan")
                    held_since[i] = float("nan")
                    sc_stopped[i] = False
                    t_closed[i] = float("nan")
                    if args.grasp == "policy":
                        use_policy_hand[i] = True
                        grasp_prev[i] = 0.0
                    go(i, S_GRASP)
            elif s == S_GRASP:
                done = False
                if args.grasp == "policy":
                    lvl = obs["grasp"][i, 26:29]
                    if float(lvl.sum()) > args.grip_stop_n:
                        h_cmd[i] = robot.data.joint_pos[i, hand._joint_ids]
                        use_policy_hand[i] = False
                        note(i, f"grasp policy: pad force {float(lvl.sum()):.1f} N > {args.grip_stop_n:g}: fingers stop")
                        done = True
                    else:
                        grip = int((lvl > 0.5).sum()) >= 2
                        if grip:
                            if math.isnan(float(grip_since[i])):
                                grip_since[i] = t
                        else:
                            grip_since[i] = float("nan")
                        gripped = not math.isnan(float(grip_since[i])) and t - float(grip_since[i]) >= 0.3
                        if gripped and math.isnan(float(held_since[i])):
                            held_since[i] = t
                        if not gripped:
                            held_since[i] = float("nan")
                        if not math.isnan(float(held_since[i])) and t - float(held_since[i]) >= args.grasp_hold_s:
                            note(i, f"grasp policy: GRIP held (levels {np.round(lvl.cpu().numpy(), 1).tolist()} N)")
                            done = True
                        elif math.isnan(float(held_since[i])) and t >= args.grasp_s:
                            note(i, f"grasp policy: NO GRIP in {args.grasp_s:g} s (levels {np.round(lvl.cpu().numpy(), 1).tolist()} N)")
                            done = True
                    if done and use_policy_hand[i]:
                        # the fingers stay at their last command (the hand term's current target)
                        h_cmd[i] = hand._processed_actions[i].clone()
                        use_policy_hand[i] = False
                else:
                    if not bool(holding[i]):
                        h_cmd[i] = H_CUP
                        holding[i] = True                       # the fingers close / regulate (above)
                    full = torch.stack([(((h_cmd[i] - _goal) * _dir >= 0) | (_gmask[g] == 0)).all() for g in range(3)])
                    sc_stopped[i] |= (ff[i] >= 0.8 * args.hold_n) | full
                    if bool(sc_stopped[i].all()) and math.isnan(float(t_closed[i])):
                        t_closed[i] = t
                    if not math.isnan(float(t_closed[i])) and t - float(t_closed[i]) >= CLOSE_HOLD_S:
                        note(i, f"scripted close done {where(i)}")
                        weaken_target(i)
                        done = True
                if done:
                    if args.grasp == "policy":
                        weaken_target(i)
                    if float(fruit_to_pocket()[i]) > LEAVE:
                        fail(i, "nothing between the fingers after the close")
                        start_legs(i, tw[i]["back"])
                        go(i, S_RETURN)
                    else:
                        back = tw[i]["back"]
                        if TWIST["deg"] > 0:
                            start_twist(i)
                            tw[i]["back"] = back
                            note(i, f"TWIST {TWIST['deg']:.0f} deg about the palm normal")
                            go(i, S_TWIST)
                        else:
                            start_detach(i)
                            note(i, f"DETACH {where(i)}")
                            go(i, S_DETACH)
            elif s == S_TWIST:
                w = tw[i]
                if t >= 2 * w["T"] + w["hold"] + 0.1:
                    start_detach(i)
                    note(i, f"DETACH after the twist {where(i)}")
                    go(i, S_DETACH)
            elif s == S_DETACH:
                if abs(t - 0.5 * detach_path[i][2]) < 0.5 * dt:
                    note(i, f"  detach half way {where(i)}")
                if detach_step(i):
                    note(i, f"  detach end {where(i)}")
                    broke = bool(T().target_stem.broken[i])
                    held = float(fruit_to_pocket()[i]) < LEAVE
                    if not broke and held:
                        fail(i, "the stem held: the hand lets go")
                    elif not held:
                        fail(i, "the apple slipped out during the detach")
                    else:
                        note(i, "detached, in the hand: RETURN")
                        outcome[i] = ""
                    start_legs(i, tw[i]["back"])
                    go(i, S_RETURN)
            elif s == S_RETURN:
                ev = legs_step(i, d, ang)
                if ev is not None:
                    go(i, S_CHECK)
            elif s == S_CHECK:
                if t >= 0.5:
                    held = (not outcome[i]) and float(fruit_to_pocket()[i]) < LEAVE
                    if held:
                        if args.full_cycle:
                            note(i, "at READY: apple in the hand -> HOME")
                            start_chain(i, [VIA, HOME])
                            go(i, S_TO_HOME)
                        else:
                            tgt = min(max(float(w_ep[i]), math.radians(BASKET["waist_range_deg"][0])),
                                      math.radians(BASKET["waist_range_deg"][1]))
                            start_waist(i, tgt)
                            w_move[i] = (*w_move[i], 0.0)
                            note(i, f"at READY: apple in the hand -> deposit start, the waist to {math.degrees(tgt):+.0f} deg")
                            start_chain(i, [READY, VIA, DEPS], speed=args.carry_speed)
                            go(i, S_TO_DEP)
                    else:
                        if not outcome[i]:
                            fail(i, "nothing in the hand after the pull")
                        start_chain(i, [VIA, HOME] if args.full_cycle else [READY, VIA, HOME])
                        go(i, S_FAILHOME)
            elif s == S_WAIST_DEP:
                if waist_step(i):
                    start_chain(i, [DEPS])
                    go(i, S_TO_DEP)
            elif s == S_DEPOSIT:
                ev = legs_step(i, d, ang)
                if ev is None and float(d[i]) < 0.03 and float(t_leg[i]) > 0.5:
                    ev = "within 3 cm"
                if ev is not None:
                    note(i, f"over the basket ({ev}, {float(d[i]) * 100:.1f} cm): RELEASE")
                    holding[i] = False
                    tw[i]["h_closed"] = h_cmd[i].clone()
                    go(i, S_RELEASE)
            elif s == S_RELEASE:
                if abs(t / 0.4 - round(t / 0.4)) < 0.5 * dt / 0.4:
                    rel = apple.data.root_pos_w[i] - env.scene.env_origins[i]
                    note(i, f"  release t={t:.1f}: apple {np.round(rel.cpu().numpy(), 2).tolist()} "
                            f"v {float(apple.data.root_lin_vel_w[i].norm()):.2f} m/s, "
                            f"{float(fruit_to_pocket()[i]) * 100:.0f} cm from the pocket")
                f = float(hv.smoothstep(torch.tensor(min(t / args.open_s, 1.0))))
                h_cmd[i] = tw[i]["h_closed"] + (H_OPEN - tw[i]["h_closed"]) * f
                rad_ = env._apple_radius[i:i + 1] if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) \
                    else torch.full((1,), 0.04, device=dev)
                settled = (t >= args.open_s + 0.3 and float(apple.data.root_lin_vel_w[i].norm()) < 0.1 and
                           bool(bk.apple_in_basket(apple.data.root_pos_w[i:i + 1] - env.scene.env_origins[i:i + 1], rad_)[0]))
                if t >= args.open_s + RELEASE_S or settled:
                    rad = env._apple_radius[i:i + 1] if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) \
                        else torch.full((1,), 0.04, device=dev)
                    inb = bool(bk.apple_in_basket(apple.data.root_pos_w[i:i + 1] - env.scene.env_origins[i:i + 1], rad)[0])
                    rel = apple.data.root_pos_w[i] - env.scene.env_origins[i]
                    note(i, ("IN THE BASKET" if inb else "released, but NOT in the basket") +
                         f" (apple env-local {np.round(rel.cpu().numpy(), 2).tolist()}, "
                         f"{float(fruit_to_pocket()[i]) * 100:.0f} cm from the pocket)")
                    ep_inb[i] = inb
                    if inb and not deposited[i]:
                        deposited[i] = True
                        n_deposits += 1
                    if not args.full_cycle:
                        note(i, "episode over" + (" (DEPOSITED)" if inb else " (missed the basket)"))
                        go(i, S_DONE)
                        continue
                    db = tw[i]["dep_back"]
                    start_legs(i, [("deposit start", db[0], db[1], db[2], True, False, False)])
                    go(i, S_BACK)
            elif s == S_BACK:
                if legs_step(i, d, ang) is not None:
                    start_chain(i, [HOME])
                    go(i, S_DEP_HOME)
            elif s == S_WAIST_BACK:
                if waist_step(i):
                    if deposited[i]:
                        note(i, "DONE")
                        go(i, S_DONE)
                    else:
                        finish_fail(i)

        _mark("máquina de estados (bucle por env)")
        if step % 250 == 0:
            if args.perfil and _perf:
                tot = sum(v for k_, v in _perf.items() if not k_.startswith("  "))
                print(f"[perfil] {1000 * tot / 250:.1f} ms por paso de control (dt simulado {1000 * dt:.0f} ms):",
                      flush=True)
                for k_, v in sorted(_perf.items(), key=lambda kv: -kv[1]):
                    print(f"    {k_:38s} {1000 * v / 250:7.1f} ms/paso", flush=True)
                _perf.clear()
            from collections import Counter
            cnt = Counter(NAMES[int(x)] for x in st)
            rtf = 250 * dt / max(time.time() - wall0, 1e-6)
            wall0 = time.time()
            print(f"t={sim_t:7.1f}s  x{rtf:.2f} tiempo real  intentos {n_rows}/{args.episodes}  "
                  f"éxito {n_ok / max(n_rows, 1):.1%}  {dict(stage_count)}", flush=True)

        # dataset: cada intento terminado (o vencido) -> una fila y un árbol nuevo
        for i in range(N):
            if ep_recorded[i]:
                continue
            if int(st[i]) == S_DONE:
                record_and_next_tree(i)
            elif sim_t - ep_t0[i] > args.ep_max_s:
                note(i, "TIEMPO: intento sin terminar")
                record_and_next_tree(i, timed_out=True)
        if n_rows >= args.episodes or sim_t >= args.max_s:
            break

if video is not None:
    video.close()
    print(f"[harvest sim] video: {args.video} ({args.video_fps} fps, simulated time, {sim_t:.1f} s)", flush=True)
csv_f.close()
print(f"[dataset] {n_rows} intentos guardados en {out_path}; éxito {n_ok}/{max(n_rows, 1)} = "
      f"{n_ok / max(n_rows, 1):.1%}", flush=True)
for k_, v_ in stage_count.most_common():
    print(f"  {k_:16s} {v_:5d}  ({v_ / max(n_rows, 1):.1%})", flush=True)
if args.diag:
    print(f"[diag] waist term in the arm action: {getattr(arm, '_waist_id', None) is not None}")
    print(f"{'state':12s} {'max|qd|':>8s} {'max lag':>8s} {'waist err':>9s}  palm path/disp per segment (cm)")
    for k in NAMES:
        if k in DIAG:
            D = DIAG[k]
            segs = " ".join(f"{a*100:.0f}/{b*100:.0f}" for a, b in D["segs"][:8])
            print(f"{k:12s} {D['qd']:8.2f} {D['lag']:8.3f} {math.degrees(D['werr']):9.1f}  {segs}")
sys.stdout.flush()
os._exit(0)
