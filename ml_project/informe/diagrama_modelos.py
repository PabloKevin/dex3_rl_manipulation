#!/usr/bin/env python3
"""Diagrama de la arquitectura de los modelos (entradas, preprocesamiento, modelos, salidas y uso) para el
informe: figuras/arquitectura_modelos.png.

    conda activate Control
    python ml_project/informe/diagrama_modelos.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figuras")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

AZUL, AZUL_C = "#2b6cb0", "#e8f0fa"
VERDE, VERDE_C = "#2f855a", "#e6f4ec"
NARANJA, NARANJA_C = "#c05621", "#fdf0e6"
GRIS, GRIS_C = "#4a5568", "#f2f3f5"
VIOLETA, VIOLETA_C = "#6b46c1", "#f0ebfa"

fig, ax = plt.subplots(figsize=(8.2, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 134)
ax.axis("off")


def caja(x, y, w, h, borde, fondo, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6",
                                fc=fondo, ec=borde, lw=lw))


def texto(x, y, s, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    ax.text(x, y, s, **kw)


def flecha(x0, y0, x1, y1, color=GRIS):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))


def etapa(y, n, nombre, color):
    ax.text(-0.5, y, f"{n}", ha="right", va="top", fontsize=13, fontweight="bold", color=color)
    ax.text(1.5, y, nombre, ha="left", va="top", fontsize=10.5, fontweight="bold", color=color)


# ── 1. fuentes ──
etapa(133, 1, "Antes del intento: lo que el robot ya sabe", GRIS)
caja(2, 120, 44, 9, GRIS, GRIS_C)
texto(4, 128, "Cámara RGB-D de la cabeza", fontweight="bold")
texto(4, 124.4, "detecta la manzana objetivo y las vecinas\n(píxel y profundidad de cada una)", fontsize=8.3)
caja(54, 120, 44, 9, GRIS, GRIS_C)
texto(56, 128, "Estado del robot", fontweight="bold")
texto(56, 124.4, "giro de la cintura y cinemática del brazo\n(posición del hombro derecho)", fontsize=8.3)
flecha(24, 119.6, 24, 115.4)
flecha(76, 119.6, 76, 115.4)

# ── 2. entradas ──
etapa(114, 2, "Entradas: vector x de 18 variables por manzana (datos tabulares)", AZUL)
caja(2, 77, 96, 32.5, AZUL, AZUL_C)
filas = [("Grupo", "Variables", "Tipo", "Unidad"),
         ("Posición respecto\nal hombro", "x, y, z; distancia 3D y horizontal;\nazimut y elevación", "7 continuas", "m, °"),
         ("Imagen", "columna u y fila v del centro;\nprofundidad medida", "3 continuas", "px, m"),
         ("Cintura", "giro de la cintura", "1 continua", "°"),
         ("Vecinas visibles", "cantidad; distancia a la más cercana\ny su desplazamiento dx, dy, dz;\n"
                              "cantidad debajo; ¿hay alguna?", "2 discretas, 4\ncontinuas, 1 binaria", "m, conteo")]
xs = [4, 25, 66, 85]
y = 108
for k, f in enumerate(filas):
    for x, s in zip(xs, f):
        texto(x, y, s, fontsize=8.3 if k else 8.6, fontweight="bold" if k == 0 else "normal",
              color=AZUL if k == 0 else "black")
    y -= 4.2 if k == 0 else (8.2 if k == 4 else 6.4)
ax.plot([3.5, 96.5], [104.0, 104.0], color=AZUL, lw=0.7)
flecha(50, 76.6, 50, 73.4)

# ── 3. preprocesamiento ──
etapa(72, 3, "Preprocesamiento (igual para todos los modelos)", AZUL)
caja(2, 59, 96, 9, AZUL, "white")
texto(4, 66.6, "• Sin vecina visible: distancia = 0,5 m, desplazamiento = 0 y el indicador «¿hay alguna?» = 0.",
      fontsize=8.3)
texto(4, 62.6, "• Estandarización: z = (x − μ) / σ, con la media μ y el desvío σ del conjunto de entrenamiento.",
      fontsize=8.3)
flecha(50, 58.6, 50, 55.4)

# ── 4. modelos ──
etapa(54, 4, "Modelo de clasificación supervisada (se comparan seis + uno trivial)", VERDE)
caja(2, 33, 96, 17, VERDE, VERDE_C)
mods = [("Regresión\nlogística", "frontera\nlineal"), ("k-NN", "vota con los\nk más parecidos"),
        ("SVM (RBF)", "frontera curva\nsuave"), ("Bosque\naleatorio", "promedio de\n300 árboles"),
        ("Gradient\nboosting", "árboles en serie\nque se corrigen"), ("Red\nneuronal", "MLP, capas\nde 32 y 16")]
for k, (n, d) in enumerate(mods):
    x = 3.5 + k * 15.8
    caja(x, 35.2, 13.6, 13.0, VERDE, "white", lw=1.0)
    texto(x + 6.8, 47.4, n, ha="center", fontweight="bold", fontsize=8.2)
    texto(x + 6.8, 41.6, d, ha="center", fontsize=7.4, color="#333")
flecha(28, 32.6, 28, 29.4)
flecha(72, 32.6, 72, 29.4)

# ── 5. salidas ──
etapa(28, 5, "Salidas", NARANJA)
caja(2, 13.5, 46, 10.8, NARANJA, NARANJA_C)
texto(4, 23.0, "Binaria: ¿llega a la canasta?", fontweight="bold")
texto(4, 19.4, "P(éxito), un número real entre 0 y 1.\nCon P ≥ 0,5 se predice «éxito».", fontsize=8.3)
caja(52, 13.5, 46, 10.8, NARANJA, NARANJA_C)
texto(54, 23.0, "Multiclase: ¿en qué etapa termina?", fontweight="bold")
texto(54, 19.4, "una de 6 clases: éxito, alcance, agarre,\ndesprendimiento, transporte u otro", fontsize=8.3)

# ── 6. uso ──
caja(2, 0.6, 96, 10.4, VIOLETA, VIOLETA_C)
ax.text(-0.5, 10.4, "6", ha="right", va="top", fontsize=13, fontweight="bold", color=VIOLETA)
texto(4, 10.0, "Cómo se interpreta y se usa", fontweight="bold", color=VIOLETA)
texto(4, 6.5, "• P es la tasa de éxito esperada: intentar solo si P ≥ umbral; ordenar las manzanas por P.\n"
              "• Elegir el giro de cintura con mayor P (se recalculan las entradas para cada giro posible).\n"
              "• La etapa predicha indica qué parte del sistema mejorar (orientó la nueva cadena de políticas).",
      fontsize=8.3)
flecha(25, 13.1, 25, 11.4, NARANJA)
flecha(75, 13.1, 75, 11.4, NARANJA)

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "arquitectura_modelos.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
print(out)
