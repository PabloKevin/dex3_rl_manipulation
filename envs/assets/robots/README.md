# Robot model

`g1_29dof_with_hand_rev_1_0.usd` is a stub: it references four layers by RELATIVE path,
`../../../../IsaacLab/configuration/ _base.usd` (and ` _physics`, ` _robot`, ` _sensor`; the
file names start with a space). From `envs/assets/robots/` that is `<parent of this repo>/IsaacLab/configuration/`
-- with the repo at `~/dex3_rl_manipulation`, `~/IsaacLab/configuration/`.

`isaaclab_configuration/` holds those four files (this laptop's trained-with copies). On a new
machine, once after cloning:

    bash scripts/install_robot_usd.sh

It copies them to `<parent of repo>/IsaacLab/configuration/` and never overwrites a file that is
already there. (Not the same files as `envs/assets/cuRobo/.../configuration/`: different export.)
