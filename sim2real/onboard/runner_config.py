"""runner_config.py -- constants of the onboard reach runner (rl_runner.py). Python 3.8."""

#: the runner's name towards the dual daemon (rt/arm_rl/ctl "src", the RL owner)
SRC = "rl_onboard"

# ---- DDS: the dual daemon's RL protocol (g1_fruits_picker/g1_dual/dual_config.py), unchanged ----
RL_CMD_TOPIC = "rt/arm_rl/cmd"
RL_CTL_TOPIC = "rt/arm_rl/ctl"
RL_STATE_TOPIC = "rt/arm_rl/state"
# ---- the runner's own topics (laptop <-> runner) ----
CTL_TOPIC = "rt/rl_onboard/ctl"       # String_ JSON from the laptop: start / stop / apple / hb / ping
STATE_TOPIC = "rt/rl_onboard/state"   # String_ JSON to the laptop

STATE_HZ_RUN = 25.0
STATE_HZ_IDLE = 2.0
DAEMON_HB_HZ = 10.0                   # explicit "hb" to the daemon while running (commands count too)

# ---- safety (the daemon guards contact / E-STOP / limits on its own) ----
#: no message from the laptop this long while running -> stop (the cable or the laptop is gone)
LAPTOP_TIMEOUT_S = 0.5
#: newest rt/lowstate older than this -> stop (toy_reach LightWatch: 100 ms)
LOWSTATE_STALE_S = 0.10
#: after rl_enter, the daemon must report this runner as the owner within this
ACQUIRE_TIMEOUT_S = 1.5
#: the daemon's state must keep saying RL + this owner; its state comes at 10 Hz
DAEMON_STATE_STALE_S = 0.6
TEMP_STOP_C = 95.0
POLL_S = 0.002
#: DDS discovery of this process: announce every SPDP_S, a lease of LEASE_S (rl_runner.main)
SPDP_S = 1.0
LEASE_S = 30.0

LOG_DIR = "/tmp/rl_onboard"
