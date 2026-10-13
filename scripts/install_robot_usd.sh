#!/usr/bin/env bash
# Put the robot model layers where envs/assets/robots/g1_29dof_with_hand_rev_1_0.usd looks for them:
# <parent of this repo>/IsaacLab/configuration/  (see envs/assets/robots/README.md).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/envs/assets/robots/isaaclab_configuration"
DST="$(dirname "$REPO")/IsaacLab/configuration"
mkdir -p "$DST"
for f in "$SRC"/*; do
    b="$(basename "$f")"
    if [ -e "$DST/$b" ]; then
        if cmp -s "$f" "$DST/$b"; then echo "same     $DST/$b"; else echo "DIFFERS  $DST/$b  (left alone)"; fi
    else
        cp -p "$f" "$DST/$b"; echo "copied   $DST/$b"
    fi
done
