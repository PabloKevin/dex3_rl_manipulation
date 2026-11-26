"""Bosque aleatorio exportado de scikit-learn, evaluado solo con numpy.

El entorno de Isaac Lab no tiene scikit-learn: exportar_bosque.py guarda los árboles del modelo entrenado en
el notebook (escalado + nodos de cada árbol) en un .npz, y esta clase reproduce RandomForestClassifier.
predict_proba (promedio de la probabilidad de la hoja de cada árbol). exportar_bosque.py verifica que las
predicciones coinciden con las de scikit-learn.
"""
import numpy as np


class BosqueNP:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.entradas = [str(s) for s in z["entradas"]]
        self.media, self.escala = z["media"], z["escala"]
        self.izq, self.der = z["izq"], z["der"]                 # índices globales de los hijos (-1 = hoja)
        self.var, self.umbral = z["var"], z["umbral"]
        self.prob = z["prob"]                                    # P(éxito) de cada nodo
        self.raices = z["raices"]                                # nodo raíz de cada árbol

    def proba(self, X):
        """X [m, n_entradas] (en el orden de self.entradas) -> P(éxito) [m]."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xs = ((X - self.media) / self.escala).astype(np.float32).astype(np.float64)   # como sklearn (float32)
        m, nt = len(Xs), len(self.raices)
        nodo = np.broadcast_to(self.raices, (m, nt)).copy()
        filas = np.arange(m)[:, None]
        while True:
            hoja = self.izq[nodo] < 0
            if hoja.all():
                break
            f = np.maximum(self.var[nodo], 0)
            sig = np.where(Xs[filas, f] <= self.umbral[nodo], self.izq[nodo], self.der[nodo])
            nodo = np.where(hoja, nodo, sig)
        return self.prob[nodo].mean(axis=1)

    def vector(self, f):
        """dict de variables (como las filas del CSV) -> vector de entradas, con las mismas reglas del notebook
        para 'sin vecino visible' (distancia 0.5 m, desplazamientos 0, indicador aparte)."""
        g = dict(f)
        g["hay_vecino_vis"] = int(g.get("n_vecinos_vis", 0) > 0)
        if not np.isfinite(g.get("dist_vecino_min_m", np.nan)):
            g["dist_vecino_min_m"] = 0.5
        for c in ("vec_dx_m", "vec_dy_m", "vec_dz_m"):
            if not np.isfinite(g.get(c, np.nan)):
                g[c] = 0.0
        return np.array([float(g[c]) for c in self.entradas])
