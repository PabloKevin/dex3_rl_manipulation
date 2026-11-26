#!/usr/bin/env python3
"""Extrae de los registros del ROBOT REAL (cosecha autónoma, logs/sim2real/toy_reach_single/*_robot/) una fila
por manzana intentada, con las MISMAS entradas geométricas del dataset simulado y el resultado real.

Para cada manzana: la posición detectada por la cámara (marco de la pelvis, línea "next: apple [...] (pelvis)"),
el giro de cintura en ese momento (última línea "waist A -> B") y el resultado ("apple N: picked / missed").
Con la cinemática del G1 (pinocchio, el mismo URDF del robot) se calcula la posición respecto al hombro derecho
en el marco del torso, igual que en la simulación. Las vecinas y el píxel NO se registraron igual que en la
simulación: la verificación usa solo variables geométricas.

    conda activate g1_dex3_sim2real
    python ml_project/intentos_reales.py      # -> ml_project/datos_reales/intentos_reales.csv
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from sim2real.reach_io import G1Kinematics  # noqa: E402

LOGS = os.path.join(REPO, "logs", "sim2real", "toy_reach_single")
OUT = os.path.join(REPO, "ml_project", "datos_reales", "intentos_reales.csv")

RE_T = re.compile(r"^(\d\d:\d\d:\d\d) ")
RE_WAIST = re.compile(r"waist ([+-]?\d+(?:\.\d+)?) -> ([+-]?\d+(?:\.\d+)?) deg")
RE_NEXT = re.compile(r"next: apple \[([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\] \(pelvis\)")
RE_OUT = re.compile(r"apple (\d+): (picked|missed)(?: \((.*)\))?")

kin = G1Kinematics()
filas = []
for d in sorted(glob.glob(os.path.join(LOGS, "*_robot"))):
    bf = os.path.join(d, "brief.log")
    if not os.path.exists(bf):
        continue
    lines = open(bf, errors="replace").read().splitlines()
    if not any("ToyHarvest" in ln for ln in lines):
        continue
    meta = {}
    try:
        meta = json.load(open(os.path.join(d, "meta.json")))
    except Exception:
        pass
    a = meta.get("args", {})
    politica = os.path.basename(os.path.dirname(os.path.dirname(str(a.get("policy", ""))))) or "?"
    agarre = str(a.get("grasp_mode", "politica" if a.get("grasp") else "?"))
    waist = 0.0
    pend = None                      # el primer "next:" de la manzana en curso
    n_tries = 0
    for ln in lines:
        m = RE_WAIST.search(ln)
        if m:
            waist = float(m.group(2))
            continue
        m = RE_NEXT.search(ln)
        if m:
            n_tries += 1
            if pend is None:
                pend = dict(p=np.array([float(m.group(k)) for k in (1, 2, 3)]), waist=waist,
                            t=(RE_T.match(ln).group(1) if RE_T.match(ln) else ""))
            continue
        m = RE_OUT.search(ln)
        if m and pend is not None:
            fr = kin.frames([math.radians(pend["waist"]), 0.0, 0.0], [0.0] * 7)
            rel = fr.torso_R.T @ (pend["p"] - fr.shoulder_pos)
            # el marco del dataset es el del torso SOLO en guiñada (yaw): torso_R con la cintura en roll/pitch 0
            x, y, z = (float(c) for c in rel)
            dh = math.hypot(x, y)
            filas.append(dict(corrida=os.path.basename(d), hora=pend["t"], manzana=int(m.group(1)),
                              politica_brazo=politica, agarre=agarre, intentos=n_tries,
                              yaw_cintura_deg=pend["waist"], x_hombro_m=x, y_hombro_m=y, z_hombro_m=z,
                              dist_hombro_m=math.sqrt(x * x + y * y + z * z), dist_horiz_m=dh,
                              azimut_deg=math.degrees(math.atan2(y, x)), elevacion_deg=math.degrees(math.atan2(z, dh)),
                              exito=int(m.group(2) == "picked"), motivo=(m.group(3) or "").strip()))
            pend, n_tries = None, 0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    w.writeheader()
    for r in filas:
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
ok = sum(r["exito"] for r in filas)
print(f"{len(filas)} manzanas intentadas en {len({r['corrida'] for r in filas})} corridas, {ok} cosechadas "
      f"({ok / max(len(filas), 1):.1%}) -> {OUT}")
from collections import Counter  # noqa: E402
print("política del brazo:", Counter(r["politica_brazo"] for r in filas))
print("agarre:", Counter(r["agarre"] for r in filas))
print("intentos por manzana:", Counter(r["intentos"] for r in filas))
