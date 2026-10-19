# Onboard reach runner (the policy on the Jetson)

The single reach's 50 Hz policy loop runs **on the robot's Jetson**, as its own process beside the dual
arm daemon. Everything else of the toy (home path, detection, GO, return, hand, voice, dashboard)
stays on the laptop.

```
laptop: run_toy.py single ... --onboard              Jetson
  rt/rl_onboard/ctl   start / hb / stop / apple ──►  rl_runner.py (own process, py3.8, numpy only)
  ◄── rt/rl_onboard/state  phase, hand phase, ...        │ rt/arm_rl/ctl  rl_enter / hb / bye
  Dex3 hand ◄─ the laptop follows the hand phase          │ rt/arm_rl/cmd  arm + waist, 50 Hz
                                                         ▼
                                                    dual daemon (unchanged) ── rt/arm_sdk ──► motors
                                                         ▲ rt/lowstate (read on the robot, polled)
```

## Isolation from the original picker

- Separate folder on the robot (`~/g1_arm_onboard/rl_onboard`) and a separate process. The deploy
  copies nothing into `onboard/` (v20) or `dual/`, and starts or stops no other process.
- Idle, it sends **nothing** to the daemon. It only acts after a `start` from the laptop.
- It uses the daemon's existing RL protocol (`rt/arm_rl/*`), so v20 and the ORIGINAL mode are unchanged.
  It never writes `rt/arm_sdk`.
- `g1_launch.sh original` never starts it. `g1_launch.sh toy -- single ... --onboard` starts it and
  stops it at the end.

## Safety

The daemon's own guards stay in charge: E-STOP (dashboard, remote), contact "stuck", the 1.2 rad
jump freeze, joint limits, and the heartbeat watchdog. In addition, the runner ends the episode
(`bye`, and the daemon holds the arm at the last target with the same gains) when:

- the laptop goes silent for 0.5 s;
- `rt/lowstate` is older than 0.1 s;
- the daemon is no longer in RL with this runner as the owner;
- a joint speed or motor temperature limit is exceeded;
- a `stop` arrives;
- the process gets SIGTERM.

The hand-over needs the dashboard's ARM MODE switch on **RL**; the toy sets it. That way the daemon
keeps the arm at the same gains between the toy and the runner, so nothing jumps.

## Use

```bash
# 1. export the policy to numpy (checks numpy == TorchScript), named <experiment>_<run>_policy.npz
python3 sim2real/export_policy_npz.py --policy <run>/exported/policy.pt --layout single_tactile
# 2. copy to the robot and check it there (Python, numpy, pinocchio, SDK, timing)
sim2real/onboard/deploy_rl_onboard.sh sim2real/onboard/policies/<that>.npz
# 3. run (the launcher starts / stops the runner)
apple_picker/g1_launch.sh toy -- single --policy <run>/exported/policy.pt --speed 0.3 --target detection --onboard
# by hand: sim2real/onboard/rl_onboard_robot.sh start | stop | status   (robot log: /tmp/rl_onboard.log)
```

Each episode's per-step log (obs, raw, q_cmd, q, ...) is saved on the robot as `/tmp/rl_onboard/<id>.npz`
and copied into the toy's run folder as `onboard_<id>.npz`.

## Files

| File | Role |
|---|---|
| `rl_runner.py` | the process: DDS, episode loop (= `toy_reach_single.policy_loop`), stops, log |
| `np_policy.py` | the rsl_rl actor in numpy (normalizer, 3 x Linear + ELU, Linear) |
| `np_fk.py` | palm / torso / shoulder / camera frames from the URDF in numpy (used when pinocchio is missing) |
| `runner_config.py` | topics, timeouts, limits |
| `deploy_rl_onboard.sh` | copy + on-robot check |
| `rl_onboard_robot.sh` | start / stop / status over ssh |
| `check_np_fk.py` | numpy FK == pinocchio (laptop) |
| `tests/loopback_onboard_check.py` | fake motors + real dual daemon + runner on `lo`: hand-overs, 50 Hz, stops, and onboard == laptop numbers |
| `../export_policy_npz.py`, `../dual/onboard_link.py` | laptop side |

Not yet: the AppleReachVisible layout (37 inputs). It needs its observation in `reach_io.py`,
validated against Isaac like the other layouts.
