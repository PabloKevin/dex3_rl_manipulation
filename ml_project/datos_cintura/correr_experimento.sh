#!/bin/bash
# Corre la prueba corta y las 3 corridas del experimento de cintura, una tras otra, con watchdog:
#  - cada corrida se termina (grupo de procesos entero) cuando su CSV llega a las filas pedidas + 90 s de gracia
#  - límite duro de tiempo por corrida, y corte si el CSV no avanza en 15 min
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/pablo_kevin/dex3_rl_manipulation
D=ml_project/datos_cintura
L=$D/logs
COMUN="--headless --perceived_apple --apple_gap 0.06"

filas() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

matar() {  # $1 = pgid
  kill -TERM -- -"$1" 2>/dev/null; sleep 10
  kill -KILL -- -"$1" 2>/dev/null; sleep 3
}

correr() {  # nombre csv n_episodios limite_s args...
  local nombre=$1 csv=$2 n=$3 lim=$4; shift 4
  echo "[$(date +%H:%M:%S)] INICIO $nombre -> $csv ($n intentos, límite $((lim/60)) min)"
  setsid python -u ml_project/cosecha_cintura.py $COMUN --episodes "$n" --out "$csv" "$@" > "$L/$nombre.log" 2>&1 &
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

rm -f $D/prueba.csv
correr prueba $D/prueba.csv 8 900 --num_envs 4 --seed 7 --cintura modelo --detalle
if ! grep -q "\[cintura\] control" $L/prueba.log || grep -q "Traceback" $L/prueba.log || [ "$(filas $D/prueba.csv)" -lt 8 ]; then
  echo "PRUEBA FALLIDA: no se lanzan las corridas"; grep -n -A3 "Traceback\|Error" $L/prueba.log | head -40; exit 1
fi
grep "\[cintura\] control" $L/prueba.log
for modo in modelo mantener aleatorio; do
  correr $modo $D/${modo}_seed3.csv 600 6600 --num_envs 48 --seed 3 --cintura $modo
done
echo "[$(date +%H:%M:%S)] TODO TERMINADO"
