#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Calibrate the MAGNITUDE of the Dex3-1 right hand's pads, one phalanx at a time, against a
hanging (luggage) scale. Run tactile_calibrate.py first: it NAMES which elements of the 108
belong to which place; this script only fits counts per newton for each of those places and
writes them next to the names (gains_counts_per_N in the same json). Read-only like that one:
nothing is commanded, the hand only has to be powered and publishing rt/dex3/right/state.

  python3 sim2real/tactile_force_calibrate.py --net-if enp62s0
  python3 sim2real/tactile_force_calibrate.py --only index_dist thumb_dist --trials 4
  python3 sim2real/tactile_calibrate.py --check       # then: live N per place; hang the scale to compare
  python3 sim2real/tactile_force_gui.py --net-if enp62s0     # the same, in a window with the hand, lamps and buttons

What it does, for each phalanx in turn (thumb, middle, index; prox then dist)
  1. says which phalanx to load and shows the seven joint angles
  2. ENTER, then PREP seconds to get the strap on and the scale in your hand; the last
     ZERO seconds the pads are re-zeroed (strap slack, hold still), then
  3. PULL NOW: you pull and hold steady for PULL seconds while the script records the pads.
     You do not need to see the screen: three short beeps one second apart, then ONE LONG
     HIGH TONE = pull now; two short beeps = release and read the scale. The tones play on
     this laptop (paplay / pw-play / aplay), never on the robot; --no_sound turns them off.
  4. you type the scale's reading (kg by default; "450g", "4.4N", "1.2lb" also work)
  5. after the trials of the place: one gain, the median of the pulls' counts per newton
     (--fit ls: least squares through the origin), with every pull's error shown.

How to load a phalanx
  Strap or string looped round the phalanx, lying ON its pad; the scale hooks the loop. Pull
  straight, at 90 degrees to the finger, away from the back of the finger: the direction a
  fruit pushes. That presses the pad and pushes the finger toward its OPEN limit, which is
  where the load should end up. Pulling toward the palm side does not load the pad at all and
  would back-drive the joint. Keep the scale's body aligned with the pull.

How to load the palm
  Strap lying across the palm, its two ends passing round the sides of the hand to the
  scale; pull straight away from the back of the hand, at 90 degrees to the palm. There is
  no joint stop behind the palm: THE PULL GOES INTO THE WRIST AND ARM, so the arm must be
  held (the daemon's hold) or supported by you, never limp. The script reads the right arm's
  seven joints from rt/lowstate (read-only) and reports how far they moved during the pull.
  The strap stays where it is for all the pulls. It loads a strip, so only some of the 9
  elements answer; the others add nothing (their rise stays inside the dead zone) and the
  gain is the sensitivity of those that did, assumed the same for the rest. The summary lists
  which elements answered, so a palm element that never answered is visible.

Hand state
  Fingers fully open, left alone. No gains, no torque: the pad reads the contact force, which
  is the scale's tension whatever holds the finger, and a stiff hold only turns a standing
  position error into motor torque (index_0 latched out of control that way on the robot).
  The script only needs the finger still, and tells you if it moved (joint drift).

The scale
  A luggage scale reads in 10-50 g steps and is least accurate at the bottom of its range:
  check it first with a known mass (a 1 litre bottle of water is 1 kg) at the loads you will
  use. One pull per place by default, aimed at 0.30 kg (3 N). --trials 3 asks for 0.15, 0.30
  and 0.50 kg (1.5-5 N, the grasp's range) and shows their spread; the light ones are the least
  certain (--fit ls weighs the firm pulls most). --max_kg is a conservative warning level, not
  a rating of the Dex3 stop: check the hand's rated load before raising it.
"""
import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tactile_io as T  # noqa: E402

G = 9.80665
#: newtons per unit a typed reading may carry
UNITS = {"kg": G, "g": G / 1000.0, "n": 1.0, "lb": 0.45359237 * G, "lbs": 0.45359237 * G}
PLACES = list(T.LOCATIONS)
#: the aim of trial 1, 2, 3 ... [kg] (the last one repeats)
HINT_KG = (0.15, 0.30, 0.50)
#: right arm joints in rt/lowstate (full_move2.RIGHT_ARM_IDXS_HW)
RIGHT_ARM = range(22, 29)
#: rt/dex3/right wire order; thumb_0 rests at 30 degrees by design, the rest are open at 0
JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")

a = None
beep = None


class Beeper:
    """Cues to hear without looking at the terminal. Played on THIS laptop (paplay, pw-play or
    aplay), never on the robot's speaker; the terminal bell when there is no player.
    tick: short beep, three of them one second apart, the last one a second before GO.
    go: one long high tone = pull now.   end: two short beeps = release, read the scale."""
    TONES = {"tick": [(660, 0.12)], "go": [(1320, 0.7)], "end": [(660, 0.12), (0, 0.1), (660, 0.12)]}

    def __init__(self, enabled=True):
        self.enabled, self.player, self.files = enabled, None, {}
        if not enabled:
            return
        for cmd in (["paplay"], ["pw-play"], ["aplay", "-q"]):
            if shutil.which(cmd[0]):
                self.player = cmd
                break
        if self.player is None:
            return
        d = tempfile.mkdtemp(prefix="tactile_beeps_")
        atexit.register(shutil.rmtree, d, True)
        rate = 44100
        for name, parts in self.TONES.items():
            wav = []
            for hz, sec in parts:
                t = np.arange(int(rate * sec)) / rate
                x = 0.6 * np.sin(2 * np.pi * hz * t) if hz else np.zeros_like(t)
                ramp = min(len(x), int(0.01 * rate))
                x[:ramp] *= np.linspace(0, 1, ramp)
                x[len(x) - ramp:] *= np.linspace(1, 0, ramp)
                wav.append(x)
            path = os.path.join(d, name + ".wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((np.concatenate(wav) * 32767).astype("<i2").tobytes())
            self.files[name] = path

    def describe(self):
        if not self.enabled:
            return "sound off (--no_sound)"
        if self.player is None:
            return "no paplay / pw-play / aplay: the terminal bell only (it may be silent)"
        return f"tones through {self.player[0]} on this laptop"

    def play(self, name):
        if not self.enabled:
            return
        if self.player is not None:
            try:
                subprocess.Popen(self.player + [self.files[name]], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return
            except OSError:
                pass
        print("\a", end="", flush=True)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-if", default="enp62s0")
    ap.add_argument("--out", default=T.DEFAULT_CALIB, help="the calibration file to read and add the gains to")
    ap.add_argument("--topic", default="rt/dex3/right/state")
    ap.add_argument("--only", nargs="+", choices=PLACES, help="calibrate these places only")
    ap.add_argument("--trials", type=int, default=1, help="pulls per place (several go at increasing force)")
    ap.add_argument("--unit", default="kg", choices=sorted(UNITS), help="unit of a bare number typed as a reading")
    ap.add_argument("--prep_s", type=float, default=15.0, help="s between ENTER and the pull window")
    ap.add_argument("--zero_s", type=float, default=1.5, help="the last s of the prep, used to re-zero the pads")
    ap.add_argument("--pull_s", type=float, default=3.0, help="s of capture per pull")
    ap.add_argument("--fit", default="median", choices=("median", "ls"),
                    help="gain of a place: median of its pulls' counts/N, or least squares through the origin")
    ap.add_argument("--no_sound", action="store_true", help="no tones (they play on this laptop, not the robot)")
    ap.add_argument("--base_s", type=float, default=2.5)
    ap.add_argument("--dead_sigma", type=float, default=4.0,
                    help="sigma of each element's noise not counted, as the grasp does (tactile_calibrate --check)")
    ap.add_argument("--min_counts", type=float, default=400.0, help="a pull that raises the pads less is discarded")
    ap.add_argument("--max_kg", type=float, default=0.8, help="warn above this pull")
    ap.add_argument("--open_tol", type=float, default=0.15, help="rad from open the joints may sit before it warns")
    ap.add_argument("--drift_tol", type=float, default=0.05, help="rad a finger joint may move during a pull")
    ap.add_argument("--arm_tol", type=float, default=0.03, help="rad an arm joint may move during a palm pull")
    return ap


def parse(argv=None):
    return make_parser().parse_args(argv)


class HandReader:
    """Subscribe only: the pressure elements and the seven joint angles of the latest HandState_."""

    def __init__(self, net_if, topic):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
        ChannelFactoryInitialize(0, net_if)
        self.lock = threading.Lock()
        self.flat, self.q, self.t = None, np.full(7, np.nan), 0.0
        self.arm_q = np.full(len(RIGHT_ARM), np.nan)
        self.sub = ChannelSubscriber(topic, HandState_)
        self.sub.Init(self._on, 10)
        try:    # the palm pull goes into the arm: watch it. Optional, the hand does not need it
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            self.sub_arm = ChannelSubscriber("rt/lowstate", LowState_)
            self.sub_arm.Init(self._on_arm, 10)
        except Exception as e:  # noqa: BLE001
            print(f"   (no rt/lowstate: the arm cannot be watched during a palm pull: {e})")

    def _on_arm(self, msg):
        try:
            q = np.array([float(msg.motor_state[i].q) for i in RIGHT_ARM])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        with self.lock:
            self.arm_q = q

    def arm(self):
        with self.lock:
            return self.arm_q.copy()

    def _on(self, msg):
        f = T.flat_from_handstate(msg)
        try:
            q = np.array([float(msg.motor_state[i].q) for i in range(7)])
        except (AttributeError, IndexError, TypeError, ValueError):
            q = np.full(7, np.nan)
        with self.lock:
            self.flat, self.q, self.t = f, q, time.monotonic()

    def read(self):
        with self.lock:
            return self.flat, self.q.copy(), self.t

    def wait(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            f, _, t = self.read()
            if f is not None and time.monotonic() - t < 0.5:
                return True
            time.sleep(0.05)
        return False

    def collect(self, seconds, hz=50.0, on_sample=None):
        """[(t, pressure elements, joint angles)] for `seconds`, one per new message."""
        out, t_end = [], time.monotonic() + seconds
        last = -1.0
        while time.monotonic() < t_end:
            f, q, t = self.read()
            if f is not None and t != last:
                out.append((t, f.copy(), q))
                last = t
                if on_sample is not None:
                    on_sample(t, f)
            time.sleep(1.0 / hz / 2)
        return out


def ask(prompt, default=""):
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def parse_force(txt, unit):
    """Newtons from '0.45', '450g', '4.4 N', '1,2lb' (a bare number is in `unit`); None if not a reading."""
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-z]*)", txt.strip().lower().replace(",", "."))
    if not m or (m.group(2) and m.group(2) not in UNITS):
        return None
    return float(m.group(1)) * UNITS[m.group(2) or unit]


def counts_of(flat, zero, idx):
    """The place's summed rise over `zero` -- the quantity TactileCalib.force() divides by the gain."""
    d = np.nan_to_num(np.asarray(flat, float)[idx] - zero[idx], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.maximum(d, 0.0).sum())


def plateau(t, c):
    """(counts of the hold, its length [s], its spread): the samples within 20% of the 90th percentile."""
    peak = float(np.percentile(c, 90))
    if peak <= 0.0:
        return 0.0, 0.0, 0.0
    sel = c >= 0.8 * peak
    val = float(np.median(c[sel]))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return val, float(sel.sum() * dt), float(np.std(c[sel]) / max(val, 1e-9))


def fit_gain(counts, newtons, method="median"):
    """counts per newton: the median of the per-pull ratios, or least squares through the origin
    (the firm pulls weigh most); and the spread of the ratios relative to it."""
    c, f = np.asarray(counts, float), np.asarray(newtons, float)
    r = c / f
    g = float(np.median(r)) if method == "median" else float((c * f).sum() / (f * f).sum())
    return g, float((r.max() - r.min()) / g)


def hint_kg(k, total):
    """The aim of pull k of `total` [kg]: a single pull goes for the middle of the range."""
    return HINT_KG[1] if total == 1 else HINT_KG[min(k, len(HINT_KG) - 1)]


def analyse_pull(win, zero, z, idx, noise, q0, min_counts):
    """One pull window [(t, pressure elements, joint angles)] -> (counts of the hold, its length [s],
    its spread, finger-joint drift [rad], the elements of `idx` that answered). `zero` is the
    trial's resting pressure, z = zero + the dead zone."""
    t = np.array([w[0] for w in win])
    c = np.array([counts_of(w[1], z, idx) for w in win])
    drift = float(np.nanmax(np.abs(np.array([w[2] for w in win]) - q0)))
    val, dur, cv = plateau(t, c)
    resp = T.responders(T.window_rise([w[1] for w in win], zero), zero, noise, 8.0, min_counts)
    return val, dur, cv, drift, [i for i in idx if i in resp]


def run_trial(hand, loc, idx, base, noise, dead, k, total):
    """One pull. Returns a trial dict, or 'redo' / 's' (skip the place) / 'q' (stop and save)."""
    kg = hint_kg(k, total)
    _, q, _ = hand.read()
    print(f"\n   {loc}  pull {k + 1}/{total}: aim for about {kg:.2f} kg ({kg * G:.1f} N), then hold steady")
    if loc == "palm":
        print("   strap across the palm, at the same place every pull; pull straight away from the back of the "
              "hand. The pull goes into the wrist and arm: the arm must be held or supported, not limp.")
        if not np.isfinite(hand.arm()).all():
            print("   WARNING no arm state (rt/lowstate): the arm's movement will not be measured")
    print("   joints [deg]: " + "  ".join(f"{n} {np.degrees(v):+.0f}" for n, v in zip(JOINTS, q)))
    away = [JOINTS[j] for j in range(1, 7) if abs(q[j]) > a.open_tol]
    if away:
        print(f"   WARNING not at the open pose (0): {away} -- the pull would move the finger (and back-drive its "
              f"motor) instead of loading the stop; open it first")
    r = ask("   ENTER starts the countdown (s = skip this place, q = stop and save) ")
    if r in ("s", "q"):
        return r

    timers = [threading.Timer(a.prep_s - n, beep.play, ("tick",)) for n in (3, 2, 1) if a.prep_s >= n + 0.5]
    for tm in timers:
        tm.start()
    wait_s = max(a.prep_s - a.zero_s, 0.0)
    time.sleep(wait_s - int(wait_s))
    print(f"   {T.LOCATION_HELP[loc]}: strap on the pad, slack, scale in hand")
    for s in range(int(wait_s), 0, -1):
        print(f"   {s} ...", flush=True)
        time.sleep(1.0)
    print("   hold still, strap slack: zeroing the pads ...", flush=True)
    zwin = hand.collect(a.zero_s)
    if len(zwin) < 10:
        print(f"   only {len(zwin)} samples in {a.zero_s} s: the hand state is not streaming")
        for tm in timers:
            tm.cancel()
        return "redo"
    zero, _ = T.baseline([w[1] for w in zwin])
    pre = counts_of(zero, base + dead, idx)
    if pre > a.min_counts:
        print(f"   WARNING the pads were already {pre:.0f} counts above the untouched zero (strap resting hard, "
              f"or a hand on the finger): the pull is measured from there")

    z = zero + dead
    q0 = hand.read()[1]
    aq0 = hand.arm()
    arm_move = [0.0]
    tick = [0.0]

    def live(t, f):
        if np.isfinite(aq0).all():
            arm_move[0] = max(arm_move[0], float(np.nanmax(np.abs(hand.arm() - aq0))))
        if t - tick[0] >= 0.5:
            tick[0] = t
            print(f"      pads +{counts_of(f, z, idx):7.0f}", flush=True)

    beep.play("go")
    print(f"   PULL NOW -- ramp up, then hold steady and read the scale ({a.pull_s:g} s)", flush=True)
    win = hand.collect(a.pull_s, on_sample=live)
    beep.play("end")
    print("   release.", flush=True)
    if len(win) < 20:
        print(f"   only {len(win)} samples in {a.pull_s} s: the hand state is not streaming")
        return "redo"

    val, dur, cv, drift, answered = analyse_pull(win, zero, z, idx, noise, q0, a.min_counts)
    print(f"   pads: {val:.0f} counts, {len(answered)} of {len(idx)} elements answered, held {dur:.1f} s "
          f"(spread {cv * 100:.0f}%), finger joints moved {drift:.3f} rad"
          + (f", arm moved {arm_move[0]:.3f} rad" if loc == "palm" and np.isfinite(aq0).all() else ""))
    if loc == "palm" and np.isfinite(aq0).all() and arm_move[0] > a.arm_tol:
        print(f"   WARNING the arm moved {arm_move[0]:.3f} rad: it is not held stiffly enough, the strap's angle and "
              f"the palm's contact changed during the pull")
    if val < a.min_counts:
        print(f"   the pads barely answered (< {a.min_counts:.0f} counts): strap not on this pad, or too light")
    if dur < 1.0:
        print("   WARNING held less than 1 s: the scale's reading will not match this window -- a steadier pull")
    if drift > a.drift_tol:
        print(f"   WARNING the finger moved {drift:.3f} rad: the pad's contact changed during the pull")

    txt = ask(f"   scale reading [{a.unit}] (ENTER = discard and repeat, s = skip place, q = stop and save) ")
    if txt in ("s", "q"):
        return txt
    force = parse_force(txt, a.unit)
    if force is None or val < a.min_counts:
        print("   discarded")
        return "redo"
    if force < 0.5:
        print(f"   {force:.2f} N is below what this scale resolves: discarded")
        return "redo"
    if force > a.max_kg * G:
        print(f"   {force:.1f} N is above --max_kg ({a.max_kg * G:.1f} N): kept, but do not go higher")
    print(f"   {val:.0f} counts / {force:.2f} N = {val / force:.0f} counts per N")
    return {"place": loc, "force_N": force, "counts": val, "held_s": dur, "spread": cv, "elements": len(idx),
            "answered": answered, "joint_drift_rad": drift, "q_start": [float(x) for x in q0],
            "arm_drift_rad": arm_move[0] if np.isfinite(aq0).all() else None, "zero_above_rest_counts": pre}


def main(argv=None):
    global a, beep
    a = parse(argv)
    beep = Beeper(not a.no_sound)
    if a.zero_s > a.prep_s:
        sys.exit("--zero_s cannot be longer than --prep_s")
    if not os.path.exists(a.out):
        sys.exit(f"{a.out} not found: run tactile_calibrate.py first (it names the elements)")
    cal = T.TactileCalib.load(a.out)
    todo = [k for k in (a.only or PLACES) if cal.locations.get(k)]
    for k in (a.only or PLACES):
        if k not in todo:
            print(f"   {k}: no elements named in {a.out} -- redo it with tactile_calibrate.py --only {k}")
    if not todo:
        sys.exit("nothing to calibrate")

    hand = HandReader(a.net_if, a.topic)
    print(f"waiting for {a.topic} on {a.net_if} ...")
    if not hand.wait():
        sys.exit("no hand state: the hand is off, or the interface is wrong (--net-if), or DDS has not "
                 "discovered it yet (a fresh process can take 20-35 s: try again)")
    f0, _, _ = hand.read()
    n = len(f0)
    if cal.n_elements and n != cal.n_elements:
        sys.exit(f"the hand reports {n} elements, {a.out} was made with {cal.n_elements}")

    print(f"   sound: {beep.describe()}")
    if beep.enabled:
        beep.play("go")
        print("   you should hear one long high tone now (that is PULL NOW); if not, check the laptop volume")

    print("\n1. BASELINE: do not touch the hand.")
    ask("   ENTER when ready ")
    samples = [w[1] for w in hand.collect(a.base_s)]
    if len(samples) < 20:
        sys.exit(f"only {len(samples)} samples in {a.base_s} s: the hand state is not streaming")
    base, noise = T.baseline(samples)
    pres = T.present(base)
    dead = a.dead_sigma * np.nan_to_num(noise)
    print(f"   {int(pres.sum())} of {n} elements report at rest")

    print(f"\n2. PULLS. Strap on the pad, pull along the pad's normal (a fruit's push), fingers open and still; "
          f"{a.trials} pull{'s' if a.trials != 1 else ''} per place, {a.prep_s:g} s to get ready after each "
          f"ENTER.")
    new, trials = {}, []
    stop = False
    for i, loc in enumerate(todo):
        if stop:
            break
        idx = [e for e in cal.locations[loc] if e < n and pres[e]]
        if not idx:
            print(f"\n   {loc}: none of its elements reports at rest -- skipped")
            continue
        print(f"\n   [{i + 1}/{len(todo)}] {loc}: {len(idx)} elements {idx}")
        while True:
            got = []
            while len(got) < a.trials:
                r = run_trial(hand, loc, idx, base, noise, dead, len(got), a.trials)
                if r == "redo":
                    continue
                if r in ("s", "q"):
                    stop = stop or r == "q"
                    if r == "s":
                        got = []
                    break
                got.append(r)
            if len(got) < 1:
                break
            g, spread = fit_gain([t["counts"] for t in got], [t["force_N"] for t in got], a.fit)
            print(f"\n   {loc}:")
            for t in got:
                print(f"     {t['force_N']:5.2f} N  {t['counts']:7.0f} counts  {t['counts'] / t['force_N']:7.0f} "
                      f"counts/N  reads {t['counts'] / g:5.2f} N at the fit ({(t['counts'] / g / t['force_N'] - 1) * 100:+.0f}%)")
            ans = sorted({i for t in got for i in t["answered"]})
            if loc == "palm":
                print(f"   elements that answered in at least one pull: {len(ans)} of {len(idx)} {ans}; "
                      f"the rest are assumed as sensitive")
            was = cal.gains.get(loc, cal.gain)
            print(f"   gain {g:.0f} counts per N (was {was:.0f}; {len(got)} pull{'s' if len(got) != 1 else ''}, "
                  f"spread {spread * 100:.0f}%)"
                  + ("  -- WIDE: pads and scale do not agree linearly, or the pulls were not steady; repeat"
                     if len(got) > 1 and spread > 0.3 else "")
                  + ("  -- one pull: nothing to check it against (--trials 3 shows a spread)"
                     if len(got) == 1 else ""))
            if stop:
                new[loc] = g
                trials += got
                break
            r = ask("   ENTER = keep, r = redo this place ")
            if r == "r":
                continue
            new[loc] = g
            trials += got
            break

    if not new:
        print("\n3. nothing accepted, nothing saved")
        return
    cal.gains.update(new)
    cal.notes = (cal.notes + " + " if cal.notes else "") + "tactile_force_calibrate.py"
    cal.save(a.out)
    log = os.path.join(os.path.dirname(os.path.abspath(a.out)), time.strftime("tactile_force_%Y%m%d_%H%M%S.json"))
    with open(log, "w") as fh:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gains_counts_per_N": new, "trials": trials,
                   "dead_sigma": a.dead_sigma, "unit": a.unit}, fh, indent=1)
    print("\n3. gains [counts per N]:")
    for k in T.LOCATIONS:
        print(f"     {k:12s} " + (f"{cal.gains[k]:8.0f}" if k in cal.gains else f"{cal.gain:8.0f}  (scalar, not pulled)"))
    print(f"   saved {a.out}\n   raw pulls in {log}\n"
          f"   toy_grasp.py / toy_reach.py load it by themselves; tactile_calibrate.py --check shows N per place.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: nothing saved")
