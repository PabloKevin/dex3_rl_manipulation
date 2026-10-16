#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Fit the reach correction from recorded calibration trials (sim2real/dual/
calib_reach.py, via run_toy.py calib) and write the JSON that
run_toy.py single --calib applies (sim2real/reach_calib.py explains the model).

  python3 sim2real/fit_reach_calib.py                                   # all trials, default paths
  python3 sim2real/fit_reach_calib.py --trials logs/sim2real/calib_reach/trials.jsonl \\
      --policy_sha <12 hex>  --out logs/sim2real/reach_calib.json

Trials of other policies are skipped when --policy_sha is given (a correction belongs
to one policy). Prints every trial's error, the constant and linear fits with their
leave-one-out error, and the orientation errors (recorded, not corrected).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reach_calib as C  # noqa: E402

REPO = os.path.dirname(HERE)
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--trials", nargs="+", default=[os.path.join(REPO, "logs", "sim2real", "calib_reach", "trials.jsonl")])
p.add_argument("--out", default=os.path.join(REPO, "logs", "sim2real", "reach_calib.json"))
p.add_argument("--policy_sha", default=None, help="only trials of this policy (calib_reach.py records it)")
p.add_argument("--ridge", type=float, default=C.RIDGE)
args = p.parse_args()

trials = C.load_trials(args.trials)
if args.policy_sha:
    trials = [t for t in trials if t.get("policy_sha") == args.policy_sha]
shas = sorted({t.get("policy_sha") for t in trials})
if len(shas) > 1:
    print(f"WARNING: trials of {len(shas)} different policies ({shas}): pass --policy_sha to fit one")
if not trials:
    sys.exit("no trials")
print(f"{len(trials)} trials:")
for i, t in enumerate(trials):
    d = C.trial_delta(t)
    print(f"  #{i:2d} {t.get('time', '?')}  apple {np.round(t['apple_true'], 3).tolist()}  shift needed "
          f"{np.round(d * 100, 1).tolist()} cm (|{np.linalg.norm(d) * 100:.1f}|)  palm rotation off "
          f"{t.get('rot_err_deg', float('nan')):.1f} deg  waist {t.get('waist_end_deg', 0):+.1f} -> "
          f"{t.get('waist_ideal_deg', 0):+.1f} deg")
c = C.fit(trials, ridge=args.ridge)
print(f"\nconstant: {np.round(np.array(c['b']) * 100, 1).tolist()} cm   LOO RMS {c['loo_rms_m_const'] * 100:.1f} cm "
      f"(no correction: RMS {c['raw_rms_m'] * 100:.1f} cm)")
if "loo_rms_m_linear" in c:
    print(f"linear:   LOO RMS {c['loo_rms_m_linear'] * 100:.1f} cm -> using {c['model']}")
else:
    print(f"linear:   needs >= {C.MIN_TRIALS_LINEAR} trials -> using the constant")
rot = [t.get("rot_err_deg") for t in trials if t.get("rot_err_deg") is not None]
if rot:
    print(f"palm orientation (ideal vs end): median {np.median(rot):.1f} deg, max {np.max(rot):.1f} deg "
          "-- not corrected by the shift")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
import json  # noqa: E402
c["trials_files"] = args.trials
c["policy_sha"] = shas[0] if len(shas) == 1 else None
with open(args.out, "w") as f:
    json.dump(c, f, indent=1)
print(f"\nwrote {args.out}   apply with: run_toy.py single ... --calib {args.out}")
