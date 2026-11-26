#!/usr/bin/env python3
"""Sección 9 del notebook: el predictor reentrenado para la cadena v5.

La cadena v5 (alcance AppleArmPose-v5 + mapa de apuntado que elige cintura y dirección de la palma + agarre
aprendido + giro de muñeca) se diseñó DESPUÉS del primer modelo de predicción. Este script:

  1. compara el resultado de los intentos con el dataset original (cadena v1);
  2. entrena los mismos modelos del notebook sobre los datos v5 (semillas 20 y 21; prueba = semilla 22);
  3. los compara contra la Q del mapa de apuntado (la predicción de alcance que ya trae la cadena) y contra el
     bosque entrenado con la cadena anterior (sin reentrenar);
  4. evalúa la regla "intentar solo si P > umbral".

    conda activate Control
    cd ml_project && python analisis_v5.py
"""
import glob
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bosque_np import BosqueNP

warnings.filterwarnings("ignore", category=UserWarning)
AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
SEED = 42
SEMILLA_TEST_V5 = 22
ENTRADAS = ["yaw_cintura_deg", "x_hombro_m", "y_hombro_m", "z_hombro_m", "dist_hombro_m", "dist_horiz_m",
            "azimut_deg", "elevacion_deg", "u_px", "v_px", "prof_cam_m",
            "n_vecinos_vis", "dist_vecino_min_m", "vec_dx_m", "vec_dy_m", "vec_dz_m",
            "n_vecinos_vis_debajo", "hay_vecino_vis"]
ETAPAS = ["exito", "alcance", "agarre", "desprendimiento", "retorno", "transporte", "deposito", "tiempo", "otro"]


def preparar(df):
    df = df[df.objetivo_visible == 1].copy()
    df["hay_vecino_vis"] = df.n_vecinos_vis.gt(0).astype(int)
    df["dist_vecino_min_m"] = df.dist_vecino_min_m.fillna(0.5)
    for c in ["vec_dx_m", "vec_dy_m", "vec_dz_m"]:
        df[c] = df[c].fillna(0.0)
    return df


def leer(carpeta, patron="*.csv"):
    fs = [f for f in sorted(glob.glob(os.path.join(AQUI, carpeta, patron))) if "prueba" not in os.path.basename(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def pipe(m):
    return Pipeline([("escala", StandardScaler()), ("modelo", m)])


MODELOS = {
    "Trivial": (pipe(DummyClassifier(strategy="most_frequent")), {}),
    "Regresión logística": (pipe(LogisticRegression(max_iter=2000)), {"modelo__C": [0.01, 0.1, 1, 10, 100]}),
    "k-NN": (pipe(KNeighborsClassifier()),
             {"modelo__n_neighbors": [5, 11, 21, 41], "modelo__weights": ["uniform", "distance"]}),
    "SVM (RBF)": (pipe(SVC(kernel="rbf", probability=True, random_state=SEED)),
                  {"modelo__C": [0.3, 1, 3, 10], "modelo__gamma": ["scale", 0.03, 0.1]}),
    "Bosque aleatorio": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
                         {"modelo__max_depth": [4, 8, None], "modelo__min_samples_leaf": [1, 5, 20]}),
    "Gradient boosting": (pipe(GradientBoostingClassifier(random_state=SEED)),
                          {"modelo__n_estimators": [100, 300], "modelo__learning_rate": [0.03, 0.1],
                           "modelo__max_depth": [2, 3]}),
    "Red neuronal (MLP)": (pipe(MLPClassifier(max_iter=2000, early_stopping=True, validation_fraction=0.15,
                                              n_iter_no_change=30, random_state=SEED)),
                           {"modelo__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                            "modelo__alpha": [1e-2, 0.1, 1.0, 10.0]}),
}


def ic95(fn, y, p, B):
    v = [fn(y[b], p[b]) for b in B if len(np.unique(y[b])) == 2]
    return np.percentile(v, [2.5, 97.5])


def fila(nombre, y, p, B, auc_vc=np.nan):
    yhat = (p >= 0.5).astype(int)
    lo, hi = ic95(roc_auc_score, y, p, B) if len(np.unique(p)) > 1 else (0.5, 0.5)
    a_lo, a_hi = ic95(lambda yy, pp: accuracy_score(yy, pp >= 0.5), y, p, B)
    return {"modelo": nombre, "exactitud": accuracy_score(y, yhat), "IC95 exactitud": f"[{a_lo:.3f}, {a_hi:.3f}]",
            "exactitud balanceada": balanced_accuracy_score(y, yhat),
            "precisión": precision_score(y, yhat, zero_division=0),
            "exhaustividad": recall_score(y, yhat, zero_division=0), "F1": f1_score(y, yhat, zero_division=0),
            "AUC-ROC": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5, "IC95 AUC": f"[{lo:.3f}, {hi:.3f}]",
            "Brier": brier_score_loss(y, p), "AUC-VC (entren.)": auc_vc}


def correr(mostrar=print):
    v1 = preparar(leer("datos"))
    v5 = preparar(leer("datos_v5", "cosecha_v5_seed*.csv"))
    tr, te = v5[v5.semilla != SEMILLA_TEST_V5], v5[v5.semilla == SEMILLA_TEST_V5]
    mostrar(f"cadena v1: {len(v1)} intentos, éxito {v1.exito.mean():.1%}  |  cadena v5: {len(v5)} intentos "
            f"(entrenamiento {len(tr)}, prueba {len(te)}), éxito {v5.exito.mean():.1%}")

    # 1. etapas de fallo, v1 vs v5
    et = pd.DataFrame({"cadena v1": v1.etapa.value_counts(normalize=True),
                       "cadena v5": v5.etapa.value_counts(normalize=True)}).fillna(0)
    et = et.loc[[e for e in ETAPAS if e in et.index]]
    et.round(3).to_csv(os.path.join(FIG, "tabla_etapas_v1_v5.csv"))
    mostrar(et.round(3))

    # 2. modelos
    Xtr, ytr = tr[ENTRADAS].values, tr.exito.values
    Xte, yte = te[ENTRADAS].values, te.exito.values
    B = np.random.default_rng(SEED).integers(0, len(yte), size=(1000, len(yte)))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    filas, P, VC = [], {}, {}
    for nombre, (est, grid) in MODELOS.items():
        gs = GridSearchCV(est, grid or {"escala": [StandardScaler()]}, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        p = gs.predict_proba(Xte)[:, 1]
        P[nombre] = p
        VC[nombre] = gs.best_score_
        filas.append(fila(nombre, yte, p, B, gs.best_score_))
        mostrar(f"  {nombre:22s} AUC-VC {gs.best_score_:.3f}  prueba {filas[-1]['AUC-ROC']:.3f}  "
                f"{gs.best_params_ if grid else ''}")
    # referencias: la Q del mapa de apuntado, y el bosque de la cadena v1 sin reentrenar
    P["Q del mapa de apuntado"] = te.q_mapa.values
    filas.append(fila("Q del mapa de apuntado (referencia)", yte, P["Q del mapa de apuntado"], B))
    viejo = BosqueNP(os.path.join(AQUI, "modelos", "bosque_exito.npz"))
    P["Bosque de la cadena v1"] = viejo.proba(te[viejo.entradas].values)
    filas.append(fila("Bosque de la cadena v1, sin reentrenar", yte, P["Bosque de la cadena v1"], B))
    # bosque con la Q del mapa como entrada extra
    rf = MODELOS["Bosque aleatorio"][0]
    gs = GridSearchCV(rf, MODELOS["Bosque aleatorio"][1], scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(tr[ENTRADAS + ["q_mapa"]].values, ytr)
    P["Bosque + Q del mapa"] = gs.predict_proba(te[ENTRADAS + ["q_mapa"]].values)[:, 1]
    filas.append(fila("Bosque aleatorio + Q del mapa", yte, P["Bosque + Q del mapa"], B, gs.best_score_))
    tabla = pd.DataFrame(filas).set_index("modelo").round(3)
    tabla.to_csv(os.path.join(FIG, "tabla_binaria_v5.csv"))
    mostrar(tabla)

    # diferencia pareada: mejor modelo reentrenado (elegido por validación cruzada, no por la prueba) vs las
    # referencias, en ordenamiento (AUC) y en calibración (Brier; negativo = el reentrenado es mejor)
    mejor = max((n for n in MODELOS if n != "Trivial"), key=lambda n: VC[n])
    comp = []
    for ref in ["Q del mapa de apuntado", "Bosque de la cadena v1"]:
        ok = [b for b in B if len(np.unique(yte[b])) == 2]
        d = [roc_auc_score(yte[b], P[mejor][b]) - roc_auc_score(yte[b], P[ref][b]) for b in ok]
        e = [brier_score_loss(yte[b], P[mejor][b]) - brier_score_loss(yte[b], P[ref][b]) for b in ok]
        comp.append({"comparación": f"{mejor} − {ref}",
                     "diferencia AUC": roc_auc_score(yte, P[mejor]) - roc_auc_score(yte, P[ref]),
                     "IC95": f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                     "diferencia Brier": brier_score_loss(yte, P[mejor]) - brier_score_loss(yte, P[ref]),
                     "IC95 Brier": f"[{np.percentile(e, 2.5):+.3f}, {np.percentile(e, 97.5):+.3f}]"})
    comp = pd.DataFrame(comp).round(3)
    comp.to_csv(os.path.join(FIG, "tabla_v5_comparacion.csv"), index=False)
    mostrar(comp)

    # 3. regla de decisión con el mejor modelo
    reg = []
    for u in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]:
        m = P[mejor] >= u
        reg.append({"umbral": u, "intentados": m.mean(), "éxito de los intentados": yte[m].mean() if m.any() else np.nan,
                    "cosechas perdidas (de las posibles)": 1 - yte[m].sum() / max(yte.sum(), 1)})
    reg = pd.DataFrame(reg).round(3)
    reg.to_csv(os.path.join(FIG, "tabla_regla_v5.csv"), index=False)
    mostrar(reg)

    # figuras
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    et.plot.bar(ax=ax[0], rot=30, color=["#9aa5b1", "#2b6cb0"])
    ax[0].set_ylabel("proporción de intentos")
    ax[0].set_title("Resultado de los intentos: cadena v1 vs v5")
    for n, c in [(mejor, "#2b6cb0"), ("Q del mapa de apuntado", "#dd6b20"),
                 ("Bosque de la cadena v1", "#9aa5b1"), ("Regresión logística", "#38a169")]:
        f_, t_, _ = roc_curve(yte, P[n])
        ax[1].plot(f_, t_, color=c, label=f"{n} (AUC {roc_auc_score(yte, P[n]):.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set_xlabel("tasa de falsos positivos")
    ax[1].set_ylabel("tasa de verdaderos positivos")
    ax[1].set_title(f"Curva ROC, prueba cadena v5 (n = {len(yte)})")
    ax[1].legend(fontsize=8, loc="lower right")
    # calibración: éxito observado por decil de probabilidad predicha
    for n, c in [(mejor, "#2b6cb0"), ("Q del mapa de apuntado", "#dd6b20"), ("Bosque de la cadena v1", "#9aa5b1")]:
        q = pd.qcut(P[n], 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": P[n], "y": yte, "q": q}).groupby("q").mean()
        ax[2].plot(g.p, g.y, "o-", color=c, label=n)
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[2].set_xlabel("probabilidad predicha (media del decil)")
    ax[2].set_ylabel("éxito observado")
    ax[2].set_title("Calibración en la prueba de la cadena v5")
    ax[2].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cadena_v5.png"), dpi=200, bbox_inches="tight")
    return dict(v1=v1, v5=v5, tabla=tabla, comp=comp, regla=reg, etapas=et, mejor=mejor, P=P, yte=yte)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    correr()
