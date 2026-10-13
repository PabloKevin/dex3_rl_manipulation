#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Calibrate and NAME the Dex3-1 right hand's pressure elements, for the grasp policy
(toy_grasp.py). Read-only: nothing is commanded, the hand only has to be powered and
publishing rt/dex3/right/state. No picker process is needed.

  python3 sim2real/tactile_calibrate.py --net-if enp62s0

What it does
  1. BASELINE   2.5 s with the hand untouched: the resting value and noise of every
                pressure element (the hand reports 9 entries x 12 elements).
  2. NAMING     for each place of the grasp policy's taxels (thumb x2, middle x2, index x2,
                palm) you press that place with a fingertip when it says GO; the elements
                that answer (above 8 sigma of their own noise) are recorded under that name.
                An element that answers in two places goes to the stronger one.
  3. FORCE      (optional) press a place onto a kitchen / postal scale and type the reading:
                counts per newton = the summed rise of that place's elements / force. Without
                it the sim's 3000 counts per N is kept. For a gain per phalanx against a
                hanging (luggage) scale use tactile_force_calibrate.py after this one.
  4. SAVE      logs/sim2real/tactile_calib.json -- toy_grasp.py loads it by itself.

  python3 sim2real/tactile_calibrate.py --check      # live view of the named places in N, Ctrl-C ends
  python3 sim2real/tactile_calibrate.py --only palm index_dist   # redo some places, keep the rest

Press firmly but not hard, with the hand held still (rest it on the table, palm up for the
fingers and thumb; palm down on a soft pad for the palm). Hold for the whole capture.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tactile_io as T  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--net-if", default="enp62s0")
ap.add_argument("--out", default=T.DEFAULT_CALIB)
ap.add_argument("--topic", default="rt/dex3/right/state")
ap.add_argument("--check", action="store_true", help="live view with the saved calibration")
ap.add_argument("--only", nargs="+", choices=T.LOCATIONS, help="redo these places, keep the others")
ap.add_argument("--base_s", type=float, default=2.5)
ap.add_argument("--press_s", type=float, default=4.0, help="s of capture per place")
ap.add_argument("--dead_sigma", type=float, default=4.0,
                help="--check: readings within this many sigma of the zero are not counted (as the toy does)")
ap.add_argument("--k_sigma", type=float, default=8.0)
ap.add_argument("--min_counts", type=float, default=400.0)
ap.add_argument("--force_trials", type=int, default=3, help="scale presses (0 = skip the force calibration)")
ap.add_argument("--yes", action="store_true", help="no prompts (testing against a stand-in hand only)")
ap.add_argument("--press_hook", default=None, help="TEST: file the script writes the place index to while capturing")
a = ap.parse_args()


class HandReader:
    def __init__(self, net_if, topic):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
        ChannelFactoryInitialize(0, net_if)
        self.lock = threading.Lock()
        self.flat, self.t = None, 0.0
        self.sub = ChannelSubscriber(topic, HandState_)
        self.sub.Init(self._on, 10)

    def _on(self, msg):
        f = T.flat_from_handstate(msg)
        with self.lock:
            self.flat, self.t = f, time.monotonic()

    def read(self):
        with self.lock:
            return self.flat, self.t

    def wait(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            f, t = self.read()
            if f is not None and time.monotonic() - t < 0.5:
                return True
            time.sleep(0.05)
        return False

    def collect(self, seconds, hz=50.0):
        out, t_end = [], time.monotonic() + seconds
        last = -1.0
        while time.monotonic() < t_end:
            f, t = self.read()
            if f is not None and t != last:
                out.append(f.copy())
                last = t
            time.sleep(1.0 / hz / 2)
        return out


def ask(prompt, default=""):
    if a.yes:
        print(f"{prompt} [--yes]")
        return default
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def show(resp, base, top=8):
    items = sorted(resp.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"[{i}] entry {i // 12}.{i % 12}: +{v:.0f}" for i, v in items) or "nothing answered"


def main():
    hand = HandReader(a.net_if, a.topic)
    print(f"waiting for {a.topic} on {a.net_if} ...")
    if not hand.wait():
        sys.exit("no hand state: the hand is off, or the interface is wrong (--net-if), or DDS has not "
                 "discovered it yet (a fresh process can take 20-35 s: try again)")
    f0, _ = hand.read()
    n = len(f0)
    print(f"hand state: {n} pressure elements ({n // 12} entries x 12)")

    if a.check:
        if not os.path.exists(a.out):
            sys.exit(f"{a.out} not found: run the calibration first")
        import select
        cal = T.TactileCalib.load(a.out)

        def zero():
            smp = hand.collect(1.5)
            b, nz = T.baseline(smp)
            return b, nz

        print("zeroing: do not touch the hand ...")
        base, noise = zero()
        dead = a.dead_sigma * np.nan_to_num(noise)
        per_place = ", ".join(f"{k} {v:g}" for k, v in cal.gains.items())
        print(f"checking with {a.out} (gain {cal.gain:g} counts/N"
              + (f"; per place: {per_place}" if per_place else "") + f"), zeroed now like the grasp does at "
              f"its start; ENTER re-zeroes, Ctrl-C ends.")
        try:
            while True:
                f, _ = hand.read()
                fr = cal.force(f, base + dead)
                print("  " + "  ".join(f"{k} {v:5.1f}" for k, v in fr.items()) + " N", flush=True)
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    sys.stdin.readline()
                    print("zeroing: do not touch ...")
                    base, noise = zero()
                    dead = a.dead_sigma * np.nan_to_num(noise)
        except KeyboardInterrupt:
            return
        return

    keep = {}
    old = None
    if a.only and os.path.exists(a.out):
        old = T.TactileCalib.load(a.out)
        keep = {k: v for k, v in old.locations.items() if k not in a.only}
    todo = a.only or T.LOCATIONS

    print("\n1. BASELINE: do not touch the hand.")
    ask("   ENTER when ready ")
    samples = hand.collect(a.base_s)
    if len(samples) < 20:
        sys.exit(f"only {len(samples)} samples in {a.base_s} s: the hand state is not streaming")
    base, noise = T.baseline(samples)
    pres = T.present(base)
    print(f"   {int(pres.sum())} of {n} elements report at rest (about {np.nanmedian(base[pres]):.0f} counts, "
          f"noise {np.nanmedian(noise[pres]):.0f} median / {np.nanmax(noise[pres]):.0f} max)")
    if pres.sum() == 0:
        sys.exit("no element reports above the 'absent' sentinel: the pads are not powered / not streaming")
    if old is not None and old.base is not None and len(old.base) == n:
        pass

    print("\n2. NAMING. For each place: ENTER, then press it when GO appears and hold the press.")
    found = {}
    for i, loc in enumerate(todo):
        while True:
            print(f"\n   [{i + 1}/{len(todo)}] {loc}: press {T.LOCATION_HELP[loc]}")
            r = ask("   ENTER starts the capture (s = skip this place, q = quit) ")
            if r == "q":
                sys.exit("aborted")
            if r == "s":
                break
            for c in (3, 2, 1):
                print(f"   {c} ...", flush=True)
                time.sleep(0.4)
            print(f"   GO -- press and hold ({a.press_s:.0f} s)", flush=True)
            if a.press_hook:
                with open(a.press_hook, "w") as fh:
                    fh.write(str(T.LOCATIONS.index(loc)))
            win = hand.collect(a.press_s)
            if a.press_hook:
                with open(a.press_hook, "w") as fh:
                    fh.write("-1")
            print("   release.", flush=True)
            rise = T.window_rise(win, base)
            resp = T.responders(rise, base, noise, a.k_sigma, a.min_counts)
            print(f"   answered: {show(resp, base)}  ({len(resp)} elements)")
            if not resp:
                r = ask("   nothing answered -- press harder / retry? ENTER = retry, s = skip ")
                if r == "s":
                    break
                continue
            r = ask("   ENTER = accept, r = redo ")
            if r == "r":
                continue
            found[loc] = resp
            break

    # an element that answered in two places belongs to the stronger one
    owner = {}
    for loc, resp in found.items():
        for i, v in resp.items():
            if i not in owner or v > owner[i][1]:
                owner[i] = (loc, v)
    shared = {i for loc, resp in found.items() for i in resp if owner[i][0] != loc}
    locs = dict(keep)
    for loc, resp in found.items():
        locs[loc] = sorted(i for i in resp if owner[i][0] == loc)
    print("\n   result:")
    for loc in T.LOCATIONS:
        e = locs.get(loc, [])
        print(f"     {loc:12s} {len(e):2d} elements  {e}" + ("" if e else "   <-- NOT FOUND"))
    if shared:
        print(f"   ({len(shared)} elements answered in more than one place and went to the stronger: "
              f"{sorted(shared)})")
    missing = [k for k in T.LOCATIONS if not locs.get(k)]
    if missing:
        print(f"   not found: {missing} -- the toy reads them as 0 N (or like the finger's other segment)")

    gain = old.gain if old is not None else T.DEFAULT_GAIN
    if a.force_trials > 0 and locs:
        print("\n3. FORCE (optional): press one place onto a scale and type what it reads.")
        loc = ask(f"   which place? {[k for k in T.LOCATIONS if locs.get(k)]} (ENTER = skip) ")
        if loc in locs and locs[loc]:
            gains = []
            for k in range(a.force_trials):
                ask(f"   trial {k + 1}/{a.force_trials}: ENTER, then press {loc} on the scale and hold steady ")
                for c in (3, 2, 1):
                    print(f"   {c} ...", flush=True)
                    time.sleep(0.4)
                print("   GO", flush=True)
                win = hand.collect(a.press_s)
                rise = T.window_rise(win, base)
                counts = float(sum(rise[i] for i in locs[loc]))
                txt = ask("   scale reading in N (grams x 0.00981; ENTER = discard) ")
                try:
                    force = float(txt)
                except ValueError:
                    continue
                if force > 0.2 and counts > 0:
                    gains.append(counts / force)
                    print(f"   {counts:.0f} counts / {force:.2f} N = {counts / force:.0f} counts per N")
            if gains:
                gain = float(np.median(gains))
                spread = (max(gains) - min(gains)) / max(gain, 1.0)
                print(f"   gain = {gain:.0f} counts per N (median of {len(gains)}, spread {spread * 100:.0f}%)"
                      + ("  -- WIDE: the readings do not agree, repeat with a steadier press" if spread > 0.4 else ""))
        else:
            print("   skipped: the sim's 3000 counts per N is kept")

    # per-place gains from tactile_force_calibrate.py survive for the places that were not redone
    gains = {k: v for k, v in old.gains.items() if k in keep} if old is not None else {}
    cal = T.TactileCalib(locs, gain, base, noise, n, notes="tactile_calibrate.py", gains=gains)
    cal.save(a.out)
    print(f"\n4. saved {a.out}\n   toy_grasp.py loads it by itself (--tactile_calib to point elsewhere); "
          f"--check shows the places live.")


main()
