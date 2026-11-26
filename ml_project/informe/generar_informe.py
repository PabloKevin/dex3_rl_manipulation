#!/usr/bin/env python3
"""Genera el informe del proyecto final (formato APA 7, en español, versión condensada) como .docx editable y lo
convierte a PDF con LibreOffice. Lee las tablas y figuras de analisis_cosecha.ipynb y analisis_v5.py (carpeta
figuras/), las corridas del experimento de cintura (datos_cintura/) y de la cadena v5 (datos_v5/). La Figura 1 la
genera informe/diagrama_modelos.py.

    <python con python-docx y pandas> ml_project/informe/generar_informe.py
"""
import glob
import math
import os
import subprocess

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

AQUI = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(AQUI)
FIG = os.path.join(ML, "figuras")
AUTOR = os.environ.get("INFORME_AUTOR", "[Nombre y apellido]")
FECHA = "26 de noviembre de 2026"
TITULO = ("Predicción del éxito de la cosecha robótica de manzanas a partir de la percepción previa al intento: "
          "comparación de modelos de aprendizaje automático entrenados en simulación")

doc = Document()

# ── estilo APA 7 (trabajo de estudiante) ──
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = st.paragraph_format
pf.line_spacing = 2.0
pf.space_before = Pt(0)
pf.space_after = Pt(0)
for s in doc.sections:
    s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Cm(2.54)


def numero_pagina(section):
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        r._r.append(e)


numero_pagina(doc.sections[0])


def par(texto="", sangria=True, centro=False, negrita=False, cursiva=False, espacio=2.0):
    p = doc.add_paragraph()
    if centro:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif sangria:
        p.paragraph_format.first_line_indent = Cm(1.27)
    p.paragraph_format.line_spacing = espacio
    _runs(p, texto, negrita, cursiva)
    return p


def _runs(p, texto, negrita=False, cursiva=False):
    """*cursiva* y **negrita** simples dentro del texto."""
    import re
    for trozo in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", texto):
        if not trozo:
            continue
        if trozo.startswith("**"):
            r = p.add_run(trozo[2:-2])
            r.bold = True
            r.italic = cursiva
        elif trozo.startswith("*"):
            r = p.add_run(trozo[1:-1])
            r.italic = True
            r.bold = negrita
        else:
            r = p.add_run(trozo)
            r.bold = negrita
            r.italic = cursiva


def titulo(texto, nivel):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    if nivel == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(texto)
        r.bold = True
    elif nivel == 2:
        r = p.add_run(texto)
        r.bold = True
    else:
        r = p.add_run(texto)
        r.bold = True
        r.italic = True
    return p


def lista(items):
    for it in items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        _runs(p, "•  " + it)


def _borde(celda, lado, val="single", sz=8):
    tcPr = celda._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    e = OxmlElement(f"w:{lado}")
    e.set(qn("w:val"), val)
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:color"), "000000")
    b.append(e)


N_TABLA = [0]
N_FIG = [0]


def tabla(titulo_t, filas, cabecera, nota=None, anchos=None, tam=10):
    N_TABLA[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(f"Tabla {N_TABLA[0]}").bold = True
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(titulo_t).italic = True
    t = doc.add_table(rows=1 + len(filas), cols=len(cabecera))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if anchos:
        t.autofit = False
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(Cm(anchos[j]).twips)))
        for j, col in enumerate(t.columns):
            col.width = Cm(anchos[j])
    for j, h in enumerate(cabecera):
        c = t.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.font.size = Pt(tam)
        r.bold = False
        _borde(c, "top")
        _borde(c, "bottom")
    for i, fila in enumerate(filas):
        for j, v in enumerate(fila):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            negr = isinstance(v, tuple)
            txt = v[0] if negr else v
            r = c.paragraphs[0].add_run(str(txt))
            r.font.size = Pt(tam)
            r.bold = negr
            if i == len(filas) - 1:
                _borde(c, "bottom")
    for row in t.rows:
        for j, c in enumerate(row.cells):
            for pp in c.paragraphs:
                pp.paragraph_format.line_spacing = 1.0
                pp.paragraph_format.first_line_indent = Cm(0)
                if j > 0:
                    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if anchos:
                c.width = Cm(anchos[j])
    if nota:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run("Nota. ")
        r.italic = True
        r.font.size = Pt(10)
        r = p.add_run(nota)
        r.font.size = Pt(10)
    doc.add_paragraph()


def figura(archivo, titulo_f, nota=None, ancho=15.5):
    ruta = os.path.join(FIG, archivo)
    if not os.path.exists(ruta):
        par(f"[Figura pendiente: {archivo}]", sangria=False)
        return None
    N_FIG[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(f"Figura {N_FIG[0]}").bold = True
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(titulo_f).italic = True
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = bool(nota)
    p.add_run().add_picture(ruta, width=Cm(ancho))
    if nota:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run("Nota. ")
        r.italic = True
        r.font.size = Pt(10)
        r = p.add_run(nota)
        r.font.size = Pt(10)
    doc.add_paragraph()
    return N_FIG[0]


def salto():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def f3(x):
    return f"{x:.3f}".replace(".", ",")


def sg(x):
    return f"{x * 100:+.1f}".replace(".", ",")


def fp(pv):
    return "p < 0,001" if pv < 0.001 else f"p = {pv:.3f}".replace(".", ",")


def pc(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


# ── datos de los resultados ──
tb = pd.read_csv(os.path.join(FIG, "tabla_binaria.csv"), index_col=0)
tm = pd.read_csv(os.path.join(FIG, "tabla_multiclase.csv"), index_col=0)
trn = pd.read_csv(os.path.join(FIG, "tabla_red_vs_bosque.csv"), index_col=0)
treal = pd.read_csv(os.path.join(FIG, "tabla_robot_real.csv"), index_col=0)
DATOS = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ML, "datos", "*.csv")))], ignore_index=True)
REAL = pd.read_csv(os.path.join(ML, "datos_reales", "intentos_reales.csv"))
n_total, n_vis = len(DATOS), int((DATOS.objetivo_visible == 1).sum())
VIS = DATOS[DATOS.objetivo_visible == 1]
n_tr = int((VIS.semilla != 1).sum())
n_te = int((VIS.semilla == 1).sum())
exito_tr = VIS[VIS.semilla != 1].exito.mean()
exito_te = VIS[VIS.semilla == 1].exito.mean()
etapas = VIS.etapa.value_counts()

# experimento de la cintura (si ya hay datos)
CINT = None
arch = [f for f in sorted(glob.glob(os.path.join(ML, "datos_cintura", "*.csv"))) if "prueba" not in os.path.basename(f)]
if arch:
    cz = pd.concat([pd.read_csv(f) for f in arch], ignore_index=True)
    rng = np.random.default_rng(42)
    CINT = {}
    for modo, g in cz.groupby("modo_cintura"):
        y = g.exito.values
        bs = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(4000)]
        CINT[modo] = dict(n=len(y), exito=y.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                          p_usado=g.p_elegido.mean(), p_ini=g.p_inicial.mean(),
                          giro=(abs(g.yaw_elegido_deg - g.yaw_inicial_deg) > 0.5).mean(),
                          giro_abs=abs(g.yaw_elegido_deg - g.yaw_inicial_deg).median(), y=y)

    def comp(a, b):
        ya, yb = CINT[a]["y"], CINT[b]["y"]
        n1, n2 = len(ya), len(yb)
        p1, p2 = ya.mean(), yb.mean()
        pp = (ya.sum() + yb.sum()) / (n1 + n2)
        z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        bs = [ya[rng.integers(0, n1, n1)].mean() - yb[rng.integers(0, n2, n2)].mean() for _ in range(4000)]
        return p1 - p2, np.percentile(bs, 2.5), np.percentile(bs, 97.5), pv

# cadena v5 (si ya hay datos y tablas de analisis_v5.py)
V5 = None
_t5 = os.path.join(FIG, "tabla_binaria_v5.csv")
_a5 = sorted(glob.glob(os.path.join(ML, "datos_v5", "cosecha_v5_seed*.csv")))
if _a5 and os.path.exists(_t5):
    _d5 = pd.concat([pd.read_csv(f) for f in _a5], ignore_index=True)
    _d5 = _d5[_d5.objetivo_visible == 1]
    V5 = dict(t=pd.read_csv(_t5, index_col=0), comp=pd.read_csv(os.path.join(FIG, "tabla_v5_comparacion.csv")),
              regla=pd.read_csv(os.path.join(FIG, "tabla_regla_v5.csv")),
              et=pd.read_csv(os.path.join(FIG, "tabla_etapas_v1_v5.csv"), index_col=0),
              n=len(_d5), n_tr=int((_d5.semilla != 22).sum()), n_te=int((_d5.semilla == 22).sum()),
              exito=_d5.exito.mean(), exito_te=_d5[_d5.semilla == 22].exito.mean(),
              semillas=sorted(_d5.semilla.unique()))
    _m = V5["t"].drop([i for i in V5["t"].index if i == "Trivial" or "referencia" in i or "sin reentrenar" in i
                       or "+ Q" in i])
    V5["mejor"] = _m["AUC-VC (entren.)"].idxmax()        # elegido por validación cruzada, como en analisis_v5.py

# ═════════════════════════ PORTADA ═════════════════════════
for _ in range(3):
    doc.add_paragraph()
par(TITULO, centro=True, negrita=True)
doc.add_paragraph()
for linea in (AUTOR,
              "Ingeniería Mecatrónica, Instituto Tecnológico Regional Suroeste, Universidad Tecnológica del Uruguay",
              "Ciencia de Datos e Inteligencia Artificial aplicados a la Mecatrónica",
              "Prof. Giovani Bolzan Cogo", FECHA):
    par(linea, centro=True)
salto()

mejor_b = tb.drop("Trivial")["AUC-ROC"].idxmax()
ic = lambda s: s.replace(".", ",").replace(", ", "; ")      # "[0.78, 0.85]" -> "[0,78; 0,85]"

# ═════════════════════════ RESUMEN ═════════════════════════
titulo("Resumen", 1)
res_txt = (
    "Un robot humanoide cosecha manzanas con un brazo y una mano de tres dedos, pero muchos intentos fallan. "
    "Este trabajo estudia si se puede predecir, antes de cada intento y solo con lo que el robot ya sabe "
    "(dónde está la manzana y qué manzanas vecinas ve su cámara), si la fruta llegará a la canasta y en qué "
    f"etapa fallará el intento. Se generó en simulación un conjunto de datos propio de {n_total} intentos, con 18 "
    "variables de entrada, y se compararon seis clasificadores supervisados de scikit-learn frente a un "
    f"clasificador trivial. El bosque aleatorio fue el mejor (AUC-ROC = {f3(tb.loc[mejor_b, 'AUC-ROC'])}); la "
    "etapa del fallo solo se anticipa bien cuando depende de la geometría. En 190 intentos del robot real, el "
    "modelo más simple (regresión logística) fue el que mejor conservó su desempeño. ")
if CINT and "modelo" in CINT and "mantener" in CINT:
    d, lo, hi, pv = comp("modelo", "mantener")
    res_txt += (f"Usado para elegir el giro de cintura antes de intentar, el modelo elevó el éxito simulado de "
                f"{pc(CINT['mantener']['exito'])} a {pc(CINT['modelo']['exito'])} ({fp(pv)}). ")
if V5:
    res_txt += (f"Con lo aprendido se diseñó una nueva versión de las políticas del robot, que cosecha el "
                f"{pc(V5['exito'])} de las manzanas; su predictor reentrenado logró AUC = "
                f"{f3(V5['t'].loc[V5['mejor'], 'AUC-ROC'])} con probabilidades bien calibradas.")
par(res_txt, sangria=False)
p = par("", sangria=True)
p.add_run("Palabras clave: ").italic = True
p.add_run("cosecha robótica, clasificación supervisada, bosque aleatorio, simulación")
salto()

# ═════════════════════════ INTRODUCCIÓN ═════════════════════════
par(TITULO, centro=True, negrita=True)
par("La cosecha de fruta requiere mucha mano de obra y automatizarla sigue siendo difícil: en una revisión de 50 "
    "proyectos de robots cosechadores, el éxito medio fue del 66 % (Bac et al., 2014). Este trabajo usa un sistema "
    "desarrollado en una tesis de grado: un robot humanoide Unitree G1 con una mano de tres dedos (Dex3) y una "
    "cámara de profundidad en la cabeza. Para cada manzana, la cámara la localiza, una red neuronal entrenada por "
    "refuerzo lleva el brazo hasta ella, la mano la agarra, el brazo tira para cortar el pedúnculo y la lleva a una "
    "canasta. En el robot real, solo un tercio de los intentos terminaba con la manzana en la canasta.")
par("Si el robot supiera de antemano qué intentos van a fallar, podría saltear manzanas imposibles, ordenar las "
    "demás y decidir mejor cómo posicionarse; y los desarrolladores sabrían qué etapa mejorar. El objetivo de este "
    "proyecto es construir ese predictor: un clasificador que, **antes de cada intento**, estime (a) la "
    "probabilidad de que la manzana llegue a la canasta y (b) la etapa en que fallaría. Como generar miles de "
    "intentos reales no es viable, los datos se generan en un simulador físico (Mittal et al., 2023), sabiendo que "
    "lo aprendido en simulación no siempre se transfiere a la realidad (Tobin et al., 2017). Por eso el modelo se "
    "contrasta, además, con registros del robot real.")

# ═════════════════════════ MÉTODO ═════════════════════════
titulo("Método", 1)
titulo("Datos", 2)
par(f"Los datos se generaron en el simulador Isaac Lab con el modelo del robot y las mismas políticas que usa el "
    "robot real (script recolectar_dataset.py). Cada intento recorre las etapas de alcance (la mano va a la "
    "manzana), agarre (la mano se cierra), desprendimiento (el brazo tira hasta romper el pedúnculo), retorno, "
    "transporte y depósito en la canasta. Cada intento se hace sobre un árbol nuevo, generado al azar con 2 a 5 "
    "manzanas, y se guarda como una fila: las entradas medidas antes de empezar y el resultado. Se hicieron tres "
    f"corridas independientes ({n_total} intentos): dos para entrenar ({n_tr} intentos útiles, éxito "
    f"{pc(exito_tr)}) y una para la prueba final ({n_te}, éxito {pc(exito_te)}), de modo que el modelo se evalúa "
    f"en árboles nunca vistos. Se descartaron {n_total - n_vis} intentos cuya manzana no se veía entera en la "
    "cámara, porque el robot real no los intentaría.")
titulo("Variables, tipo de problema y modelos", 2)
par("La Figura 1 resume la arquitectura completa. Las 18 entradas (variables independientes) son solo información "
    "disponible antes de intentar: la posición de la manzana respecto al hombro derecho, su posición y "
    "profundidad en la imagen, el giro de la cintura y las manzanas vecinas que la cámara ve. Son 15 continuas y 3 "
    "discretas. Las salidas (variables dependientes) son el éxito (binaria) y la etapa donde terminó el intento "
    "(categórica). Variables que solo existen en simulación o que se conocen después del intento no se usan, para "
    "no filtrar información del resultado. Como todas las filas tienen su resultado registrado, el problema es de "
    "**clasificación supervisada**; un enfoque no supervisado solo tendría sentido sin etiquetas.")
figura("arquitectura_modelos.png", "Arquitectura del predictor: entradas, preprocesamiento, modelos, salidas y uso",
       nota="Todos los modelos reciben el mismo vector de entradas estandarizado y se entrenan por separado para "
            "cada salida. P es la probabilidad de éxito predicha.", ancho=15)
par("Se compararon seis modelos de naturaleza distinta, todos de scikit-learn (Pedregosa et al., 2011), dentro de "
    "una tubería con estandarización de las entradas (Tabla 1), más un clasificador trivial que siempre predice la "
    "clase más frecuente como referencia mínima.")
MODS = [
    ("Regresión logística", "Línea de base lineal e interpretable", "C (regularización)"),
    ("k vecinos más cercanos", "El éxito depende de la zona del espacio de trabajo", "k, pesos"),
    ("SVM (núcleo RBF)", "Frontera no lineal suave", "C, γ"),
    ("Bosque aleatorio", "Interacciones entre variables; robusto (Breiman, 2001)", "profundidad, hojas"),
    ("Gradient boosting", "Muy preciso en datos tabulares", "árboles, tasa, profundidad"),
    ("Red neuronal (MLP)", "Aproximador no lineal general, con parada temprana", "capas, α (L2)"),
]
tabla("Modelos comparados", [list(m) for m in MODS], ["Modelo", "Por qué se eligió", "Hiperparámetros ajustados"],
      anchos=[4.2, 7.6, 4.5])
titulo("Validación y métricas", 2)
par("Los hiperparámetros se eligieron con búsqueda en grilla y validación cruzada de 5 particiones sobre el "
    "conjunto de entrenamiento; el de prueba se usó una sola vez. La métrica principal es el **AUC-ROC** "
    "(Fawcett, 2006): la probabilidad de que el modelo asigne mayor P a un intento exitoso que a uno fallido, es "
    "decir, qué tan bien ordena las manzanas (0,5 = azar, 1 = perfecto). Se reportan también la exactitud, la "
    "precisión, la exhaustividad y la F1 con umbral 0,5, y la puntuación de **Brier**, el error cuadrático medio de "
    "la probabilidad, que mide si P se puede leer como una tasa de éxito real (calibración; menor es mejor). Los "
    "intervalos de confianza del 95 % (IC 95 %) se obtuvieron por *bootstrap* del conjunto de prueba (Efron y "
    "Tibshirani, 1993), pareado cuando se comparan dos modelos. Para la etapa del fallo, cuyas clases están "
    "desbalanceadas, se usó la F1 macro (promedio de la F1 de cada clase).")
titulo("Tres pruebas adicionales", 2)
par("(1) **Robot real:** de los registros de 50 sesiones de cosecha autónoma se extrajeron "
    f"{len(REAL)} intentos (éxito {pc(REAL.exito.mean())}); como no se registraron las vecinas ni el píxel, se "
    "reentrenaron los modelos con las 7 variables geométricas y se evaluaron sobre esos intentos. (2) **Uso para "
    "decidir:** antes de cada intento, el bosque evaluó los giros de cintura de −15° a +15° y eligió el de mayor "
    "P; se comparó contra mantener el giro inicial y contra un giro al azar (600 intentos simulados cada uno). "
    "(3) **Nuevas políticas:** con lo aprendido se diseñó una cadena de políticas nueva (v5, descrita en "
    "Resultados); se generaron 1800 intentos con ella y se reentrenaron los modelos.")

# ═════════════════════════ RESULTADOS ═════════════════════════
titulo("Resultados", 1)
titulo("Análisis de las variables", 2)
par(f"De los intentos simulados, el {pc(etapas.get('exito', 0) / len(VIS))} terminó en la canasta; los fallos más "
    f"comunes fueron el agarre ({pc(etapas.get('agarre', 0) / len(VIS))}) y el desprendimiento "
    f"({pc(etapas.get('desprendimiento', 0) / len(VIS))}), seguidos del alcance "
    f"({pc(etapas.get('alcance', 0) / len(VIS))}) y el transporte ({pc(etapas.get('transporte', 0) / len(VIS))}). "
    "La variable que mejor separa éxitos de fracasos es la altura de la manzana respecto al hombro (Figura 2): por "
    "debajo de 0,42 m bajo el hombro ningún intento tuvo éxito. El mapa de correlaciones (Figura 3) muestra la "
    "misma relación (altura r = 0,44; distancia al hombro r = −0,40) y correlaciones fuertes entre entradas que "
    "describen la misma posición de formas distintas (por ejemplo, la altura y la fila del píxel); esto afecta a "
    "los modelos lineales pero no a los basados en árboles.")
figura("exito_vs_altura.png", "Tasa de éxito según la altura de la manzana respecto al hombro", ancho=12.5)
figura("correlaciones.png", "Correlaciones de Pearson entre las entradas y el éxito", ancho=12)

titulo("Predicción del éxito", 2)
filas_b = []
for m, r in tb.iterrows():
    fila = [m, f3(r["exactitud"]), f3(r["precisión"]), f3(r["exhaustividad"]), f3(r["F1"]),
            f"{f3(r['AUC-ROC'])} {ic(r['IC95 AUC'])}", f3(r["Brier"])]
    filas_b.append([(x,) for x in fila] if m == mejor_b else fila)
tabla("Desempeño en el conjunto de prueba (¿llega a la canasta?)", filas_b,
      ["Modelo", "Exactitud", "Precisión", "Exhaust.", "F1", "AUC-ROC [IC 95 %]", "Brier"],
      nota=f"{n_te} intentos de prueba. En negrita, el mejor modelo.", anchos=[3.6, 1.8, 1.8, 1.8, 1.5, 3.8, 1.5],
      tam=9)
par(f"Todos los modelos superaron al trivial (Tabla 2). El bosque aleatorio fue el mejor en AUC "
    f"({f3(tb.loc[mejor_b, 'AUC-ROC'])}), exactitud ({f3(tb.loc[mejor_b, 'exactitud'])} frente a "
    f"{f3(tb.loc['Trivial', 'exactitud'])} del trivial) y calibración; la regresión logística fue la más débil, "
    "señal de que la relación no es lineal; las curvas ROC están en la Figura 4. La importancia por permutación confirma que el bosque se apoya sobre "
    "todo en la distancia y la posición de la manzana respecto al hombro; las vecinas aportan poco. Como regla "
    "práctica, intentar solo las manzanas con P ≥ 0,5 sube el éxito de los intentos realizados del 42 % al 68 %, a "
    "cambio de intentar el 38 % de las manzanas.")
figura("roc.png", "Curvas ROC en el conjunto de prueba", ancho=10)
par("La red neuronal (dos capas ocultas de 32 y 16 neuronas) obtuvo AUC = "
    f"{f3(tb.loc['Red neuronal (MLP)', 'AUC-ROC'])}; la diferencia con el bosque es de +0,029 a su favor, IC 95 % "
    "[+0,011; +0,047], significativa. Su curva por época (Figura 5) muestra que la pérdida de entrenamiento "
    "sigue bajando mientras el AUC de prueba se estanca en torno a 0,79: entrenar más solo la sobreajusta, y la "
    "parada temprana la detuvo en la época 151. Que los árboles superen a la red en datos tabulares de este tamaño "
    "coincide con lo reportado en la literatura (Grinsztajn et al., 2022).")
figura("red_epocas.png", "Red neuronal: pérdida de entrenamiento y AUC por época", ancho=14)

titulo("Predicción de la etapa del fallo", 2)
mejor_m = tm.drop("Trivial")["F1 macro"].idxmax()
filas_m = []
for m, r in tm.iterrows():
    fila = [m, f3(r["exactitud"]), f3(r["exactitud balanceada"]), f3(r["F1 macro"])]
    filas_m.append([(x,) for x in fila] if m == mejor_m else fila)
tabla("Desempeño en la predicción de la etapa (conjunto de prueba)", filas_m,
      ["Modelo", "Exactitud", "Exactitud balanceada", "F1 macro"],
      nota="Seis clases: éxito, alcance, agarre, desprendimiento, transporte y «otro». El trivial predice siempre "
           "«éxito».", anchos=[4.5, 3, 4, 3])
par(f"El bosque también fue el mejor aquí (F1 macro = {f3(tm.loc[mejor_m, 'F1 macro'])}, Tabla 3), pero con "
    "resultados muy desiguales por clase: los fallos de alcance se predicen casi perfectamente (F1 = 0,93) y los "
    "de agarre razonablemente (0,67), porque dependen de la geometría; los de desprendimiento solo en parte (0,41), "
    "y las caídas durante el transporte casi nada (0,09), porque dependen de qué tan bien quedó agarrada la fruta, "
    "algo que no se conoce antes de intentar.")

titulo("Robot real", 2)
filas_r = []
mejor_r = treal["AUC robot real"].idxmax()
for m, r in treal.iterrows():
    fila = [m, f3(r["AUC simulación (prueba)"]), f"{f3(r['AUC robot real'])} {ic(r['IC95 real'])}"]
    filas_r.append([(x,) for x in fila] if m == mejor_r else fila)
tabla("Modelos geométricos: AUC en simulación y en el robot real", filas_r,
      ["Modelo", "AUC simulación", "AUC robot real [IC 95 %]"],
      nota=f"{len(REAL)} intentos reales, éxito {pc(REAL.exito.mean())}; 7 variables geométricas.", anchos=[5, 3.5, 5.5])
par("En el robot real el orden de los modelos se invierte (Tabla 4): la regresión logística, la más débil en "
    f"simulación, es la mejor (AUC = {f3(treal.loc['Regresión logística', 'AUC robot real'])}) y conserva su "
    "desempeño, mientras que los modelos flexibles caen a 0,56–0,68. Estos aprenden detalles propios del "
    "simulador que no se cumplen en la realidad; la frontera suave de la regresión logística capta solo la "
    "tendencia general (las manzanas bajas y lejanas fallan), que sí se transfiere. Todos sobreestiman el éxito "
    "real. Con 190 intentos de versiones mezcladas del sistema, esta conclusión es orientativa.")

titulo("Uso del modelo para decidir el giro de cintura", 2)
if CINT and all(k in CINT for k in ("modelo", "mantener")):
    filas_c = []
    for modo, nombre in (("mantener", "Mantener el giro inicial"), ("modelo", "Giro elegido por el modelo"),
                         ("aleatorio", "Giro al azar (control)")):
        if modo in CINT:
            c = CINT[modo]
            filas_c.append([nombre, c["n"], pc(c["exito"]), f"[{pc(c['lo'])}; {pc(c['hi'])}]", f3(c["p_usado"])])
    tabla("Éxito simulado según cómo se elige el giro de cintura", filas_c,
          ["Grupo", "Intentos", "Éxito", "IC 95 %", "P predicha media"], anchos=[5.2, 2, 2, 3.8, 3])
    d, lo, hi, pv = comp("modelo", "mantener")
    d2, _, _, pv2 = comp("aleatorio", "mantener")
    par(f"Elegir el giro con el modelo subió el éxito de {pc(CINT['mantener']['exito'])} a "
        f"{pc(CINT['modelo']['exito'])} ({sg(d)} puntos, IC 95 % [{sg(lo)}; {sg(hi)}], {fp(pv)}; Tabla 5), mientras "
        f"que girar al azar no mejoró significativamente ({sg(d2)} puntos, {fp(pv2)}): la mejora se debe a la "
        "elección del modelo y no al simple hecho de girar. Además, la probabilidad media predicha coincidió con "
        "el éxito observado en los tres grupos, es decir, el modelo anticipó cuánto iba a mejorar.")

if V5:
    titulo("Nuevas políticas (v5) y predictor reentrenado", 2)
    et5 = V5["et"]
    par("Los resultados anteriores mostraron que la mayoría de los fallos ocurrían en el agarre y el "
        "desprendimiento, y que la posición de la manzana respecto al brazo era decisiva. Con esa información se "
        "diseñó después una nueva cadena de políticas (v5): un mapa aprendido que elige el giro de cintura y la "
        "dirección de la mano para cada manzana, un alcance que evita empujar la fruta, y un agarre y un "
        "desprendimiento más firmes. El mapa trae su propia estimación de si el brazo llegará a la manzana, Q, que "
        "sirve como referencia. En el mismo mundo simulado, la cadena v5 cosechó el "
        f"{pc(V5['exito'])} de los intentos, frente al {pc(et5.loc['exito', 'cadena v1'])} de la original: los fallos "
        f"de agarre bajaron de {pc(et5.loc['agarre', 'cadena v1'])} a {pc(et5.loc['agarre', 'cadena v5'])} y los de "
        f"desprendimiento de {pc(et5.loc['desprendimiento', 'cadena v1'])} a "
        f"{pc(et5.loc['desprendimiento', 'cadena v5'])} (Figura 6, izquierda).")
    t5 = V5["t"]
    mejor5 = V5["mejor"]
    filas5 = []
    for i, nombre in ((mejor5, f"{mejor5} reentrenado"), ("Regresión logística", "Regresión logística reentrenada"),
                      ("Q del mapa de apuntado (referencia)", "Q del mapa (referencia)"),
                      ("Bosque de la cadena v1, sin reentrenar", "Bosque anterior, sin reentrenar"),
                      ("Trivial", "Trivial")):
        r = t5.loc[i]
        filas5.append([nombre, f"{f3(r['AUC-ROC'])} {ic(r['IC95 AUC'])}", f3(r["exactitud"]), f3(r["Brier"])])
    tabla("Predicción del éxito con las políticas v5 (conjunto de prueba)", filas5,
          ["Modelo", "AUC-ROC [IC 95 %]", "Exactitud", "Brier"],
          nota=f"{V5['n_te']} intentos de prueba; el resto de los modelos reentrenados quedó entre AUC 0,84 y 0,86.",
          anchos=[6, 4.5, 2.5, 2.2])
    c5 = V5["comp"].set_index("comparación")
    cq = c5.loc[f"{mejor5} − Q del mapa de apuntado"]
    cv1 = c5.loc[f"{mejor5} − Bosque de la cadena v1"]
    rg = V5["regla"].set_index("umbral")
    r5 = rg.loc[0.5]
    par(f"Reentrenado con los datos v5, el {mejor5.lower()} alcanzó AUC = {f3(t5.loc[mejor5, 'AUC-ROC'])} (Tabla 6). "
        "Ordena las manzanas igual que la Q del mapa y que el bosque anterior sin reentrenar (diferencias no "
        "significativas), porque con las nuevas políticas lo que más se puede anticipar es si el brazo alcanza la "
        "manzana. La ventaja está en la calibración (Figura 6, derecha): su Brier es significativamente menor que "
        f"el de la Q (IC 95 % de la diferencia {ic(cq['IC95 Brier'])}), que sobreestima el éxito, y que el del bosque "
        f"anterior ({ic(cv1['IC95 Brier'])}), que lo subestima porque aprendió con políticas que fallaban más. Es "
        "decir, cuando cambian las políticas el orden se conserva, pero las probabilidades deben reentrenarse. Como "
        f"regla, intentar solo con P ≥ 0,5 sube el éxito de los intentos de "
        f"{pc(rg.loc[0.0, 'éxito de los intentados'])} a {pc(r5['éxito de los intentados'])}, perdiendo el "
        f"{pc(r5['cosechas perdidas (de las posibles)'])} de las cosechas posibles.")
    figura("cadena_v5.png", "Políticas v5: resultado de los intentos, curvas ROC y calibración", ancho=16)

# ═════════════════════════ CONCLUSIONES ═════════════════════════
titulo("Conclusiones", 1)
par("Se puede predecir, antes de intentar y con información que el robot ya tiene, si una cosecha simulada "
    f"terminará bien: el bosque aleatorio ordena los intentos con AUC = {f3(tb.loc[mejor_b, 'AUC-ROC'])}, superó "
    "significativamente a la red neuronal, y lo predecible se explica sobre todo por la posición de la manzana "
    "respecto al hombro. La etapa del fallo solo se anticipa cuando depende de la geometría (alcance y agarre).")
par("El modelo fue útil más allá de predecir. Usado para decidir el giro de cintura, mejoró la cosecha simulada "
    "con probabilidades que coincidieron con lo observado; y el diagnóstico de dónde fallaban los intentos guió el "
    + (f"diseño de nuevas políticas que llevaron el éxito simulado de {pc(VIS.exito.mean())} a {pc(V5['exito'])}. "
       if V5 else "diseño de nuevas políticas. ") +
    "Dos lecciones prácticas: en el robot real conviene un modelo simple, porque los flexibles aprenden detalles "
    "del simulador; y cada vez que cambian las políticas, el predictor debe reentrenarse para que sus "
    "probabilidades sigan siendo confiables.")
titulo("Limitaciones y trabajo futuro", 2)
lista([
    "Los datos de entrenamiento son simulados; la física del agarre y del pedúnculo es aproximada.",
    "La prueba en el robot real es pequeña (190 intentos), mezcla versiones del sistema y no incluye las vecinas "
    "ni el píxel; el uso del giro de cintura y las políticas v5 solo se probaron en simulación.",
    "No se usó la distancia al tronco ni las oclusiones entre manzanas, que en un huerto real influyen.",
    "Próximos pasos: registrar en el robot real las mismas variables para reentrenar con datos reales, y sumar "
    "información del propio intento (fuerzas en los dedos) para anticipar las caídas durante el transporte.",
])

# ═════════════════════════ REFERENCIAS ═════════════════════════
salto()
titulo("Referencias", 1)
REFS = [
    "Bac, C. W., van Henten, E. J., Hemming, J., y Edan, Y. (2014). Harvesting robots for high-value crops: "
    "State-of-the-art review and challenges ahead. *Journal of Field Robotics, 31*(6), 888–911. "
    "https://doi.org/10.1002/rob.21525",
    "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Efron, B., y Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
    "Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. "
    "https://doi.org/10.1016/j.patrec.2005.10.010",
    "Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning "
    "on typical tabular data? *Advances in Neural Information Processing Systems, 35*, 507–520.",
    "Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J. L., Singh, R., Guo, Y., Mazhar, H., "
    "Mandlekar, A., Babich, B., State, G., Hutter, M., y Garg, A. (2023). Orbit: A unified simulation framework "
    "for interactive robot learning environments. *IEEE Robotics and Automation Letters, 8*(6), 3740–3747. "
    "https://doi.org/10.1109/LRA.2023.3270034",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, "
    "P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y "
    "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, "
    "2825–2830.",
    "Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., y Abbeel, P. (2017). Domain randomization for "
    "transferring deep neural networks from simulation to the real world. En *2017 IEEE/RSJ International "
    "Conference on Intelligent Robots and Systems (IROS)* (pp. 23–30). https://doi.org/10.1109/IROS.2017.8202133",
]
for ref in REFS:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)
    _runs(p, ref)

# ═════════════════════════ ANEXO ═════════════════════════
titulo("Anexo: código entregado (carpeta ml_project)", 2)
lista([
    "recolectar_dataset.py y cosecha_v5.py — simulación de la cosecha y registro de los datos (una fila por intento).",
    "analisis_cosecha.ipynb (y su versión .py) — análisis de variables, modelos, comparación y pruebas adicionales.",
    "analisis_v5.py — predictor reentrenado para las políticas v5.",
    "intentos_reales.py — extracción de los intentos del robot real a partir de sus registros.",
    "cosecha_cintura.py, exportar_bosque.py y bosque_np.py — experimento del giro de cintura.",
    "informe/diagrama_modelos.py — Figura 1.",
    "datos/, datos_reales/, datos_cintura/, datos_v5/ — conjuntos de datos (CSV).",
])

out = os.path.join(AQUI, "informe_proyecto_final.docx")
doc.save(out)
print("docx:", out)
# entorno limpio: las librerías de conda rompen LibreOffice (libreglo.so)
r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", AQUI, out],
                   capture_output=True, text=True, timeout=300,
                   env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin",
                        "LD_LIBRARY_PATH": "/usr/lib/libreoffice/program"})
print(r.stdout.strip() or r.stderr.strip())
