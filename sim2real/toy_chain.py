#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
SIM2REAL TOY TEST 3 -- the CHAINED two-stage reach on a REAL apple the camera sees:
part 1 (AppleReachPregrasp-v3) then part 2 (AppleReachFinalChained-v2).

This is toy_apple.py (the picker's perception -> a detected apple, the same robot
path, safety watch, E-STOP, home pose, return, logs: sim2real/toy_reach.py, subclassed,
nothing of it changed) with the policy replaced by the chain trained in
envs/chained_reach_env.py, stage by stage as in training:

  1  home, hand OPEN. The part-1 policy (39 inputs: no palm-touch term) drives the arm +
     waist until its own success criterion: the palm's grasp point 6 cm in front of the
     apple surface (within 3 cm), palm normal within 20 deg of horizontal and 45 deg of the
     approach, held 1.0 s  (envs/apple_reach_twostage_env_cfg.py PREGRASP_*).
  2  HAND-OFF: the hand goes to the wider cup (envs/mdp/pregrasp.FINAL_HAND_PRESHAPE: index
     / middle tips curled 60 deg), the arm waits --cup_wait_s for the fingers, the palm
     touch input is switched on (virtual, from the detected apple, as in toy_reach.py).
  3  The part-2 policy (40 inputs) drives to the gentle touch (palm gap at the touch
     distance within --reach_tol) -> hold -> back home along the way it came -> hand open.

The two policies are the raw rsl_rl checkpoints of training (model_<n>.pt): no export.
One rate limiter and one "previous action" run through both stages, as in the sim.

Terminals as toy_apple.py (humble-box + conda g1_dex3_sim2real; robot side only pane 1 of
launch_apple_picker.sh; live_locator + fruits_to_torso for the detections):

  cd ~/dex3_rl_manipulation
  python3 sim2real/toy_chain.py --mock   --speed 0.3 --mock_apples "0.33,-0.21,0.28" --policy1 P1.pt --policy2 P2.pt
  python3 sim2real/toy_chain.py --shadow --speed 0.3 --policy1 P1.pt --policy2 P2.pt   # nothing moves
  python3 sim2real/toy_chain.py          --speed 0.3 --policy1 P1.pt --policy2 P2.pt   # the test (at the robot, e-stop in hand)

--policy1 / --policy2 default to the newest checkpoint under logs/rsl_rl/apple_reach_pregrasp_v3
and logs/rsl_rl/apple_reach_final_chained_v2. Every toy_reach.py option applies
(python3 sim2real/toy_reach.py -h), defaults here: --target detection --waist follow --hand open.

What is NOT as in training, said plainly:
  - the clearance ring (neighbours around the target) is fed "all free" (toy_reach.py too);
  - the palm touch in part 2 is the virtual one computed from the detected apple;
  - the real hand closes slower than the sim's PD: --cup_wait_s pauses the arm for it (the sim
    has no pause; part 1 has settled, so the arm is still anyway);
  - if part 1 never reaches its criterion (--pregrasp_s / speed), the run STOPS and goes home;
    the sim hands off at the cap regardless (--handoff_on_timeout does the same here).
"""
import argparse
import glob
import importlib.util
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy_reach as T  # noqa: E402

R = T.R
REPO = T.REPO
C_RED, C_YEL, C_GRN, C_CYN, C_OFF = T.C_RED, T.C_YEL, T.C_GRN, T.C_CYN, T.C_OFF

P1_ROOT = os.path.join(REPO, "logs", "rsl_rl", "apple_reach_pregrasp_v3")
P2_ROOT = os.path.join(REPO, "logs", "rsl_rl", "apple_reach_final_chained_v2")
#: where the palm-touch input sits in the 40-input observation (reach_io.build_obs order:
#: arm q 7, arm dq 7, apple 3, palm quat 4, prev action 7, TOUCH, ring 8, waist 3)
TOUCH_COL = 28


# ─────────────────────────────────────────────
#  The sim's constants (read from the training files, not copied)
# ─────────────────────────────────────────────

def _number(path: str, name: str, default: float, degrees: bool = False) -> float:
    """`NAME = 0.03` or `NAME = math.radians(20.0)` from a source file (no Isaac Lab import)."""
    try:
        txt = open(path).read()
        m = re.search(rf"^{name}\s*=\s*(?:math\.radians\()?([0-9.]+)\)?", txt, re.M)
        if m:
            v = float(m.group(1))
            return math.radians(v) if "radians" in m.group(0) else v
    except OSError:
        pass
    print(f"{C_YEL}  could not read {name} from {path}: using {default}{C_OFF}")
    return default


def _load_pregrasp():
    spec = importlib.util.spec_from_file_location("g1_pregrasp", os.path.join(REPO, "envs", "mdp", "pregrasp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PG = _load_pregrasp()
_CFG2 = os.path.join(REPO, "envs", "apple_reach_twostage_env_cfg.py")
_CFG1 = os.path.join(REPO, "envs", "apple_grasp_env_cfg.py")
P1_STANDOFF = PG.PREGRASP_STANDOFF
P1_POS_TOL = _number(_CFG2, "PREGRASP_POS_TOL", 0.03)
P1_HOLD_S = _number(_CFG2, "PREGRASP_HOLD_TIME_S", 1.0)
P1_TILT_TOL = _number(_CFG1, "PALM_TILT_TOL_RAD", math.radians(20.0))
P1_STEM_TOL = _number(_CFG1, "PALM_STEM_TOL_RAD", math.radians(45.0))
P1_CAP_S = 6.0          # AppleReachFinalChainedEnvCfg.pregrasp_s (a class field there, not a module constant)
FINAL_S = _number(_CFG2, "FINAL_STAGE_S", 4.0)


# ─────────────────────────────────────────────
#  Policies: raw rsl_rl checkpoints
# ─────────────────────────────────────────────

class CheckpointPolicy:
    """The actor MLP + observation normaliser of an rsl_rl checkpoint (model_<n>.pt), as
    envs/chained_reach_env.FrozenActor, mean action, CPU, batch 1. A TorchScript export
    works too. Takes the 40-input observation reach_io.build_obs makes; a 39-input policy
    (no palm-touch term) gets the touch column removed."""

    EPS = 1e-2          # rsl_rl EmpiricalNormalization

    def __init__(self, path: str):
        import torch
        torch.set_num_threads(1)
        self.torch, self.path = torch, path
        self.net = None
        try:
            self.net = torch.jit.load(path, map_location="cpu").eval()
            ws = [p for _, p in self.net.named_parameters() if p.dim() == 2]
            self.n_in, self.n_out = int(ws[0].shape[1]), int(ws[-1].shape[0])
            self.mean = self.std = None
        except Exception:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            sd = blob["model_state_dict"]
            idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
            layers = []
            for n, i in enumerate(idx):
                w = sd[f"actor.{i}.weight"]
                layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
                if n < len(idx) - 1:
                    layers.append(torch.nn.ELU())
            self.net = torch.nn.Sequential(*layers)
            self.net.load_state_dict({f"{2 * n}.{t}": sd[f"actor.{i}.{t}"] for n, i in enumerate(idx)
                                      for t in ("weight", "bias")})
            self.net.eval()
            self.n_in, self.n_out = layers[0].in_features, layers[-1].out_features
            self.mean = sd["actor_obs_normalizer._mean"].clone().view(-1)
            self.std = sd["actor_obs_normalizer._std"].clone().view(-1)
        if self.n_in not in (39, 40):
            raise ValueError(f"{path}: {self.n_in} inputs -- not a reach policy (39 without the palm-touch term, 40 with)")
        if self.n_out != 8:
            raise ValueError(f"{path}: {self.n_out} outputs, the waist reach layout has 8")
        self.layout = "waist"

    def __call__(self, obs40: np.ndarray) -> np.ndarray:
        x = np.asarray(obs40, np.float32)
        if self.n_in == 39:
            x = np.delete(x, TOUCH_COL)
        with self.torch.no_grad():
            t = self.torch.as_tensor(x).view(1, -1)
            if self.mean is not None:
                t = (t - self.mean) / (self.std + self.EPS)
            return self.net(t)[0].numpy().astype(float)


def newest_checkpoint(root: str) -> str | None:
    """Highest model_<n>.pt of the newest dated run under root (warmstart_* skipped)."""
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and "warmstart" not in d)
    for run in reversed(runs):
        cks = glob.glob(os.path.join(run, "model_*.pt"))
        if cks:
            return max(cks, key=lambda p: int(os.path.basename(p)[6:-3]))
    return None


# ─────────────────────────────────────────────
#  The chained toy
# ─────────────────────────────────────────────

class ChainToy(T.Toy):
    def __init__(self, args):
        # Toy.__init__ builds ONE policy through R.ReachPolicy(args.policy): that is part 2
        R.ReachPolicy = CheckpointPolicy
        super().__init__(args)
        self.pol2 = self.pol
        self.pol1 = CheckpointPolicy(args.policy1)
        if self.pol2.n_in != 40:
            sys.exit(f"{args.policy2}: part 2 takes the palm-touch input (40 inputs), this one has {self.pol2.n_in}")
        self.stage = 1
        self._wrap_hand()

    # ---- the hand: one more shape on top of toy_reach's "cup" / "open" ----
    def _wrap_hand(self):
        io, orig = self.io, self.io.hand_set
        final = {n: PG.FINAL_HAND_PRESHAPE[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER if n != "thumb_0"}

        def hand_set(kind):
            if kind != "final_cup":
                return orig(kind)
            if hasattr(io, "hand_tgt"):                       # MockIO
                for i, n in enumerate(R.DEX3_ORDER):
                    if i:
                        io.hand_tgt[i] = final[n]
                return None
            if not io.commands or io.estop_why is not None:   # shadow, or stopped
                return None
            for i, n in enumerate(R.DEX3_ORDER):              # as PickerIO.hand_set's cup (thumb_0 is runtime-locked)
                if i:
                    io.app.set_hand_joint_deg("right", i, math.degrees(final[n]))
            return None

        io.hand_set = hand_set

    # ---- part 1's success criterion, as reach_neighbourhood with the pregrasp standoff ----
    def p1_state(self, fr):
        a = self.args
        seat = fr.palm_ref + (a.apple_radius + P1_STANDOFF) * fr.palm_normal
        d = float(np.linalg.norm(self.apple - seat))
        tilt = math.asin(float(np.clip(abs(fr.palm_normal[2]), 0.0, 1.0)))
        ah = np.array([math.cos(self.approach_yaw), math.sin(self.approach_yaw), 0.0])
        head = math.acos(float(np.clip(fr.palm_normal @ ah, -1.0, 1.0)))
        ok = d < P1_POS_TOL and tilt < P1_TILT_TOL and head < P1_STEM_TOL
        return ok, d, tilt, head

    def shadow_run(self):
        """Read-only: part 1 on the live state (its stage is the one a still arm is in)."""
        pol2, self.pol = self.pol, self.pol1
        try:
            return super().shadow_run()
        finally:
            self.pol = pol2

    def policy_loop(self):
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        L = R.make_limiters(self.layout)
        prev_raw = np.zeros(8)
        s = io.read()
        fr = self.frames(s)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=True)
        W.reset()
        self.path_cmd = []
        self.set_phase("PREGRASP")
        self.ctx["doing"] = "part 1 (pregrasp, hand open) driving the arm"
        self.stage = 1
        need = max(1, round(P1_HOLD_S / R.DT))
        hold_n = 0
        cap1 = a.pregrasp_s / a.speed
        t_hand = None                                    # when part 2 took over
        settled_since = reached_t = None
        stopped, outcome = False, None
        waist_sent, waist_sent_t = 0.0, -9.0
        level = sm.OK
        t_start = io.now()
        t_next = t_start
        t_print = -1.0
        prof = {k: 0.0 for k in ("read", "obs+policy", "watch", "send", "status+log")}
        self.prof = prof
        tw = T.time.perf_counter
        try:
            while True:
                t_tick = io.now()
                p0 = tw()
                if io.estop_why is not None:
                    stopped, outcome = True, f"E-STOP: {io.estop_why}"
                    self.halted = outcome
                    break
                s = io.read()
                p1 = tw()
                prof["read"] += p1 - p0
                t = t_tick - t_start
                fr = self.frames(s)
                gap = float(np.linalg.norm(self.apple - fr.palm_ref)) - a.apple_radius
                touch = 0.0
                if self.stage == 2 and a.touch == "virtual":
                    touch = R.virtual_touch_force(fr, self.apple, a.apple_radius)
                st = R.ReachState(arm_q=s.arm_q, arm_dq=s.arm_dq, waist_q=s.waist_q, waist_dq=s.waist_dq,
                                  apple=self.apple, touch_force=touch)
                obs, _ = R.build_obs(self.layout, self.kin, st, prev_raw, fr)
                raw = (self.pol1 if self.stage == 1 else self.pol2)(obs)
                slow = a.safety == "on" and level >= sm.SLOW
                speed = a.speed * (a.safety_slow if slow else 1.0)
                tgt = L["arm"].step(raw[:7], s.arm_q, speed)
                q_cmd = np.clip(tgt, self.lo, self.hi)
                hit = np.abs(q_cmd - tgt) > 1e-6
                if hit.any():
                    L["arm"].prev = q_cmd.copy()
                    L["arm"].vel[hit] = 0.0
                clipped = [R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)]
                w_cmd = float(np.clip(L["waist"].step(raw[7:8], s.waist_q[:1], speed)[0], -self.waist_lim, self.waist_lim))
                w_des = float(np.clip(L["waist"].desired[0], -self.waist_lim, self.waist_lim))
                now = io.now()
                if (a.waist == "follow" and not io.qref_waist and abs(w_des - waist_sent) > math.radians(2.0)
                        and now - waist_sent_t > 0.3):
                    dur = max(0.5, 1.875 * abs(w_des - float(s.waist_q[0])) / (R.WAIST_MAX_VELOCITY * speed))
                    io.waist_to(w_des, dur)
                    waist_sent, waist_sent_t = w_des, now
                self.ctx.update(gap=gap, touch=round(touch, 2), clipped=clipped, speed_now=speed, stage=self.stage)
                p2 = tw()
                prof["obs+policy"] += p2 - p1
                level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
                p3 = tw()
                prof["watch"] += p3 - p2
                self.ctx["level_name"] = sm.LEVEL_NAMES[level]
                self.ctx["warnings"] = [w for _, w in out][:4]
                W.print_new(out, t)
                if level >= sm.STOP and a.safety == "on":
                    stopped = True
                    outcome = "SAFETY STOP: " + "; ".join(w for l, w in out if l >= sm.STOP)
                    self.halted = outcome
                    break
                if out and any(l >= sm.STOP for l, _ in out) and a.safety != "on":
                    hard = [w for l, w in out if l >= sm.STOP and (w.startswith("lowstate") or "daemon" in w)]
                    if hard:
                        stopped, outcome = True, "STOP: " + "; ".join(hard)
                        self.halted = outcome
                        break
                stream_waist = a.waist == "follow" and io.qref_waist
                p4 = tw()
                io.send_arm(q_cmd, waist=w_cmd if stream_waist else None,
                            waist_dq=float(L["waist"].vel[0]) if stream_waist else 0.0)
                prof["send"] += tw() - p4
                p5 = tw()
                self.q_sent = q_cmd.copy()
                prev_raw = raw
                self.path_cmd.append(q_cmd.copy())
                err = gap - R.GRASP_STANDOFF
                tilt = math.degrees(math.asin(np.clip(fr.palm_normal[2], -1, 1)))
                head = math.degrees(math.atan2(fr.palm_normal[1], fr.palm_normal[0]) - self.approach_yaw)

                if self.stage == 1:
                    ok, d1, tl, hd = self.p1_state(fr)
                    hold_n = hold_n + 1 if ok else 0
                    if hold_n >= need or t >= cap1:
                        if hold_n < need and not a.handoff_on_timeout:
                            outcome = (f"PREGRASP NOT REACHED in {t:.1f} s: palm {d1 * 1000:.0f} mm from the pregrasp "
                                       f"point (tol {P1_POS_TOL * 1000:.0f}), tilt {math.degrees(tl):.0f} deg "
                                       f"(tol {math.degrees(P1_TILT_TOL):.0f}), heading {math.degrees(hd):.0f} deg "
                                       f"(tol {math.degrees(P1_STEM_TOL):.0f}) -- stopped, hand still open")
                            break
                        self.say(f"  PREGRASP {'reached' if hold_n >= need else 'TIMEOUT (handing off anyway)'} at "
                                 f"t={t:.2f} s: palm {d1 * 1000:.0f} mm from the pregrasp point, gap to the apple "
                                 f"surface {gap * 1000:.0f} mm -> hand to the cup, part 2", C_GRN)
                        io.hand_set("final_cup")
                        err_h = self.hold(self.q_sent, a.cup_wait_s)
                        if err_h:
                            stopped, outcome = True, err_h
                            self.halted = outcome
                            break
                        self.stage, t_hand = 2, io.now() - t_start
                        self.set_phase("POLICY")
                        self.ctx["doing"] = "part 2 (final approach, cupped hand) driving the arm"
                        t_next = io.now()               # the cup wait is not a late control tick
                        continue
                else:
                    still = float(np.max(np.abs(s.arm_dq))) < 0.08 * max(a.speed, 0.3) + 0.04
                    if a.after_reach == "freeze":
                        arrived = err <= a.reach_tol
                    else:
                        if abs(err) < a.reach_tol and still:
                            settled_since = settled_since if settled_since is not None else t
                        else:
                            settled_since = None
                        arrived = settled_since is not None and t - settled_since >= a.settle_s
                    if arrived and reached_t is None:
                        reached_t = t
                        self.reach_info = dict(t=t, t_handoff=t_hand, err_mm=err * 1000, tilt_deg=tilt,
                                               heading_deg=head, palm=fr.palm_ref.tolist(), q=s.arm_q.tolist())
                        self.say(f"  REACHED at t={t:.2f} s (part 2 since {t_hand:.2f} s): palm gap error "
                                 f"{err * 1000:+.1f} mm, palm normal {tilt:+.0f} deg from horizontal (trained: "
                                 f"within 20), {head:+.0f} deg from the approach (trained: within 45)", C_GRN)
                        if a.after_reach == "freeze":
                            outcome = f"REACHED in {t:.2f} s (gap error {err * 1000:+.1f} mm)"
                            break
                        self.set_phase("HOLD")
                        self.ctx["doing"] = "policy holding the touch"

                self.rows.append(dict(
                    t=t, stage=self.stage, obs=obs, raw=raw, q_cmd=q_cmd, q=s.arm_q, dq=s.arm_dq, tau=s.arm_tau,
                    waist=s.waist_q, waist_dq=s.waist_dq, waist_des=w_des, waist_cmd=w_cmd, palm=fr.palm_ref,
                    normal=fr.palm_normal, gap=gap, touch=touch, speed=speed, level=level,
                    clear=self.ctx.get("clear", np.nan), resid=self.ctx.get("resid", np.nan),
                    off_path=self.ctx.get("off_path", np.nan),
                    tactile=np.nan if s.tactile is None else s.tactile, age=s.age, compute=io.now() - t_tick))
                if t - t_print >= 0.5:
                    t_print = t
                    extra = (f"pregrasp d {self.p1_state(fr)[1] * 1000:4.0f} mm hold {hold_n}/{need}"
                             if self.stage == 1 else f"touch {touch:4.1f} N")
                    print(f"  [{self.stage}] t={t:5.2f}s gap {gap * 1000:+6.1f} mm  {extra}  palm v "
                          f"{W.palm_v:.2f} m/s  clear {self.ctx.get('clear', np.nan) * 100:+5.1f} cm  tilt {tilt:+4.0f} "
                          f"heading {head:+4.0f} deg  waist {math.degrees(s.waist_q[0]):+5.1f} "
                          f"(cmd {math.degrees(w_cmd):+5.1f})  [{sm.LEVEL_NAMES[level]}{' SLOW' if slow else ''}]",
                          flush=True)
                self.status(s)
                prof["status+log"] += tw() - p5
                if reached_t is not None and t - reached_t >= a.hold_s:
                    outcome = f"REACHED in {reached_t:.2f} s, held by the policy {a.hold_s:.1f} s " \
                              f"(gap error {err * 1000:+.1f} mm)"
                    break
                if self.stage == 2 and t - t_hand >= FINAL_S / a.speed:
                    outcome = (f"PART 2 TIME LIMIT {FINAL_S / a.speed:.0f} s (gap error {err * 1000:+.1f} mm)")
                    break
                if t >= a.time_limit:
                    outcome = f"TIME LIMIT {a.time_limit:.0f} s (gap error {err * 1000:+.1f} mm)"
                    break
                t_next += R.DT
                lag = io.now() - t_next
                if lag > R.DT:
                    W.print_new([(sm.WARN, f"control loop {lag * 1000:.0f} ms late")], t)
                    t_next = io.now()
                io.sleep_until(t_next)
        except KeyboardInterrupt:
            stopped, outcome = True, "OPERATOR STOP (Ctrl-C)"
            self.halted = outcome
        self.ctx["commanding"] = False
        col = C_GRN if outcome.startswith("REACHED") else C_RED if stopped else C_YEL
        self.say(f"  chain: {outcome}", col)
        return outcome, stopped


# ─────────────────────────────────────────────
#  Command line: toy_reach's, plus the chain's
# ─────────────────────────────────────────────

def parse_chain_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy1", default=None, help="part 1 checkpoint (default: newest under " + P1_ROOT + ")")
    p.add_argument("--policy2", default=None, help="part 2 checkpoint (default: newest under " + P2_ROOT + ")")
    p.add_argument("--pregrasp_s", type=float, default=P1_CAP_S,
                   help="s of part 1 allowed at the trained speed (the sim's stage-1 cap); / speed here")
    p.add_argument("--cup_wait_s", type=float, default=0.5,
                   help="s the arm is held while the fingers go to the part-2 cup (the real hand is slower than the sim's)")
    p.add_argument("--handoff_on_timeout", action="store_true",
                   help="hand off to part 2 when part 1 times out without its criterion (the sim does); default: stop")
    return p.parse_known_args(argv)


def main():
    mine, rest = parse_chain_args(sys.argv[1:])
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
    p1 = mine.policy1 or newest_checkpoint(P1_ROOT)
    p2 = mine.policy2 or newest_checkpoint(P2_ROOT)
    if not p1 or not p2:
        sys.exit(f"no checkpoint found ({'part 1' if not p1 else 'part 2'}): pass --policy1 / --policy2")
    for k, v in (("--target", ["detection"]), ("--waist", ["follow"]), ("--hand", ["open"])):
        if k not in rest:
            rest += [k] + v
    if "--policy" not in rest:
        rest += ["--policy", p2]
    sys.argv = [sys.argv[0]] + rest
    args = T.parse_args()
    args.policy1, args.policy2 = p1, p2
    args.pregrasp_s, args.cup_wait_s, args.handoff_on_timeout = mine.pregrasp_s, mine.cup_wait_s, mine.handoff_on_timeout
    if "--time_limit" not in rest:                       # both stages, at the commanded speed
        args.time_limit = (mine.pregrasp_s + FINAL_S) / args.speed
    if args.log_dir is None:
        mode = "mock" if args.mock else "shadow" if args.shadow else "robot"
        args.log_dir = os.path.join(REPO, "logs", "sim2real", "toy_chain",
                                    f"{T.time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}")
    print(f"{C_CYN}== chain: part 1 {p1}\n           part 2 {p2}{C_OFF}")
    toy = ChainToy(args)
    if args.real and not getattr(toy.io, "loopback", False):
        print(f"{C_RED}REAL ROBOT. Someone at the robot with the remote / emergency stop, the area "
              f"in front of the right arm clear.{C_OFF}")
    out = toy.run()
    sys.exit(0 if out and (out.startswith("REACHED") or out.startswith("shadow")) else 1)


if __name__ == "__main__":
    main()
