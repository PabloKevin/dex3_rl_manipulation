#!/bin/bash
# Recolección del dataset con la cadena v5 (ml_project/cosecha_v5.py), con watchdog:
#  - cada corrida se termina (grupo de procesos entero) cuando su CSV llega a las filas pedidas + 90 s de gracia
#  - límite duro de tiempo por corrida, y corte si el CSV no avanza en 15 min
#
#   bash ml_project/datos_v5/correr_v5.sh prueba          # 32 intentos, 16 envs, semilla 7
#   bash ml_project/datos_v5/correr_v5.sh datos N ENVS SEMILLAS...   # p. ej. datos 600 32 20 21 22
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/pablo_kevin/dex3_rl_manipulation
D=ml_project/datos_v5
L=$D/logs
mkdir -p $L
# el mismo mundo que el dataset original (manzana percibida, vecinas a >= 6 cm) ...
MUNDO="--headless --perceived_apple --apple_gap 0.06"
# ... y la mejor cadena v5 del bench (JTW60_KP3_E300, 78 % de los intentados), SIN omitir manzanas (skip_q 0):
# así el dataset tiene todos los casos y la Q del mapa queda como predictor de referencia
CADENA="--arm_policy logs/rsl_rl/apple_arm_pose_v5/2026-11-13_05-14-18/exported/policy.pt --arm_layout v5 \
 --stage none --detach_path monotone --track 1 --aim envs/assets/aim_map_v5.pt --skip_q 0 \
 --learned_grasp logs/rsl_rl/apple_grasp_arm_pose_v2/2026-11-13_13-42-31/exported/policy_300.pt \
 --learned_hold freeze_after_grasp --roll_down 1 --twist_away_deg 60 --twist_joint 1 --grip_kp 3"

filas() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

matar() {  # $1 = pgid
  kill -TERM -- -"$1" 2>/dev/null; sleep 10
  kill -KILL -- -"$1" 2>/dev/null; sleep 3
}

correr() {  # nombre csv n_episodios limite_s args...
  local nombre=$1 csv=$2 n=$3 lim=$4; shift 4
  echo "[$(date +%H:%M:%S)] INICIO $nombre -> $csv ($n intentos, límite $((lim/60)) min)"
  setsid python -u ml_project/cosecha_v5.py $MUNDO $CADENA --episodes "$n" --out "$csv" "$@" > "$L/$nombre.log" 2>&1 &
  local pid=$! t0=$SECONDS ultimo=-1 t_ultimo=$SECONDS lleno=""
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local f; f=$(filas "$csv")
    if [ "$f" != "$ultimo" ]; then ultimo=$f; t_ultimo=$SECONDS; fi
    if [ "$f" -ge "$n" ] && [ -z "$lleno" ]; then lleno=$SECONDS; echo "[$(date +%H:%M:%S)] $nombre: $f filas, esperando cierre"; fi
    if [ -n "$lleno" ] && [ $((SECONDS - lleno)) -ge 90 ]; then echo "[$(date +%H:%M:%S)] $nombre: no cerró solo, se termina"; matar $pid; break; fi
    if [ $((SECONDS - t0)) -ge "$lim" ]; then echo "[$(date +%H:%M:%S)] $nombre: LÍMITE DE TIEMPO, se termina ($f filas)"; matar $pid; break; fi
    if [ $((SECONDS - t_ultimo)) -ge 900 ] && [ $((SECONDS - t0)) -ge 900 ]; then echo "[$(date +%H:%M:%S)] $nombre: SIN AVANCE 15 min, se termina ($f filas)"; matar $pid; break; fi
  done
  wait $pid 2>/dev/null
  kill -KILL -- -"$pid" 2>/dev/null   # por si quedó algún hijo
  echo "[$(date +%H:%M:%S)] FIN $nombre: $(filas "$csv") filas en $(((SECONDS - t0)/60)) min"
}

if [ "$1" = "prueba" ]; then
  rm -f $D/prueba.csv
  correr prueba $D/prueba.csv 32 1500 --num_envs 16 --seed 7
  grep -n -A3 "Traceback" $L/prueba.log | head -30
  tail -12 $L/prueba.log
elif [ "$1" = "datos" ]; then
  n=$2 envs=$3; shift 3
  for s in "$@"; do
    correr seed$s $D/cosecha_v5_seed$s.csv "$n" 7200 --num_envs "$envs" --seed "$s"
  done
fi
echo "[$(date +%H:%M:%S)] TODO TERMINADO"
