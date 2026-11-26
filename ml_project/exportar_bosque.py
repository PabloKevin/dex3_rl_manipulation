#!/usr/bin/env python3
"""Entrena el bosque aleatorio binario del notebook (mismas entradas, mismos hiperparámetros elegidos por
validación cruzada, solo con las semillas de entrenamiento) y lo exporta a modelos/bosque_exito.npz para
usarlo dentro de la simulación (cosecha_cintura.py) sin scikit-learn.

    conda activate Control
    cd ml_project && python exportar_bosque.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bosque_np import BosqueNP

AQUI = os.path.dirname(os.path.abspath(__file__))
SEED = 42
SEMILLA_TEST = 1
ENTRADAS = ["yaw_cintura_deg", "x_hombro_m", "y_hombro_m", "z_hombro_m", "dist_hombro_m", "dist_horiz_m",
            "azimut_deg", "elevacion_deg", "u_px", "v_px", "prof_cam_m",
            "n_vecinos_vis", "dist_vecino_min_m", "vec_dx_m", "vec_dy_m", "vec_dz_m",
            "n_vecinos_vis_debajo", "hay_vecino_vis"]

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(AQUI, "datos", "*.csv")))], ignore_index=True)
df = df[df.objetivo_visible == 1].copy()
df["hay_vecino_vis"] = df.n_vecinos_vis.gt(0).astype(int)
df["dist_vecino_min_m"] = df.dist_vecino_min_m.fillna(0.5)
for c in ["vec_dx_m", "vec_dy_m", "vec_dz_m"]:
    df[c] = df[c].fillna(0.0)
tr, te = df[df.semilla != SEMILLA_TEST], df[df.semilla == SEMILLA_TEST]

modelo = Pipeline([("escala", StandardScaler()),
                   ("modelo", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                                     random_state=SEED, n_jobs=-1))])
modelo.fit(tr[ENTRADAS].values, tr.exito.values)
p_sk = modelo.predict_proba(te[ENTRADAS].values)[:, 1]
print(f"entrenamiento {len(tr)}, prueba {len(te)}: AUC prueba {roc_auc_score(te.exito, p_sk):.3f}")

esc, rf = modelo[0], modelo[-1]
izq, der, var, umbral, prob, raices = [], [], [], [], [], []
base = 0
for t in rf.estimators_:
    tt = t.tree_
    v = tt.value[:, 0, :]
    v = v / v.sum(axis=1, keepdims=True)
    k = list(rf.classes_).index(1)
    raices.append(base)
    izq.append(np.where(tt.children_left >= 0, tt.children_left + base, -1))
    der.append(np.where(tt.children_right >= 0, tt.children_right + base, -1))
    var.append(tt.feature)
    umbral.append(tt.threshold)
    prob.append(v[:, k])
    base += tt.node_count

os.makedirs(os.path.join(AQUI, "modelos"), exist_ok=True)
out = os.path.join(AQUI, "modelos", "bosque_exito.npz")
np.savez(out, entradas=np.array(ENTRADAS), media=esc.mean_, escala=esc.scale_,
         izq=np.concatenate(izq).astype(np.int64), der=np.concatenate(der).astype(np.int64),
         var=np.concatenate(var).astype(np.int64), umbral=np.concatenate(umbral), prob=np.concatenate(prob),
         raices=np.array(raices, dtype=np.int64))

b = BosqueNP(out)
p_np = b.proba(te[ENTRADAS].values)
print(f"exportado a {out} ({base} nodos, {len(raices)} árboles); "
      f"diferencia máx. numpy vs sklearn: {np.abs(p_np - p_sk).max():.2e}")
row = te.iloc[0].to_dict()
for c in ("dist_vecino_min_m", "vec_dx_m", "vec_dy_m", "vec_dz_m"):
    row[c] = np.nan if row["n_vecinos_vis"] == 0 else row[c]
assert abs(b.proba(b.vector(row))[0] - p_sk[0]) < 1e-9
assert np.abs(p_np - p_sk).max() < 1e-9
print("ok")
