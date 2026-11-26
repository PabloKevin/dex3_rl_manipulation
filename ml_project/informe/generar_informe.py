#!/usr/bin/env python3
"""Genera el informe del proyecto final (formato APA 7, en español) como .docx editable y lo convierte a PDF
con LibreOffice. Lee las tablas y figuras que produce analisis_cosecha.ipynb (carpeta figuras/) y, si existen,
las corridas del experimento de cintura (datos_cintura/).

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

# ═════════════════════════ RESUMEN ═════════════════════════
titulo("Resumen", 1)
mejor_b = tb.drop("Trivial")["AUC-ROC"].idxmax()
res_txt = (
    "Un robot humanoide Unitree G1 con mano Dex3 cosecha manzanas mediante políticas de aprendizaje por refuerzo, "
    "pero muchos intentos fallan. Este trabajo estudia si es posible predecir, antes de intentar, si la manzana "
    "terminará en la canasta y en qué etapa fallará el intento, usando solo información que el robot conoce de "
    f"antemano. Se generó un conjunto de datos propio en simulación (Isaac Lab) con {n_total} intentos de cosecha "
    "completos sobre árboles aleatorios, con 18 variables de entrada (posición de la manzana respecto al hombro, "
    "píxel y profundidad en la cámara, y manzanas vecinas visibles). Se compararon seis modelos supervisados de "
    "scikit-learn (regresión logística, k vecinos más cercanos, máquina de vectores de soporte, bosque aleatorio, "
    "gradient boosting y una red neuronal multicapa) frente a un clasificador trivial, con hiperparámetros elegidos "
    "por validación cruzada y evaluación en una corrida de simulación independiente. "
    f"El bosque aleatorio obtuvo el mejor desempeño binario (AUC-ROC = {f3(tb.loc[mejor_b, 'AUC-ROC'])}, exactitud = "
    f"{f3(tb.loc[mejor_b, 'exactitud'])} frente a {f3(tb.loc['Trivial', 'exactitud'])} del trivial) y superó "
    "significativamente a la red neuronal. La etapa de fallo solo se predice bien cuando depende de la geometría "
    "(alcance y agarre). Al aplicar modelos geométricos entrenados en simulación a 190 intentos del robot real, la "
    f"regresión logística mantuvo su desempeño (AUC = {f3(treal.loc['Regresión logística', 'AUC robot real'])}) "
    "mientras que los modelos más flexibles perdieron capacidad, lo que evidencia la brecha entre simulación y "
    "realidad. ")
if CINT and "modelo" in CINT and "mantener" in CINT:
    d, lo, hi, pv = comp("modelo", "mantener")
    res_txt += (f"Finalmente, usar el modelo para elegir el giro de cintura antes de cada intento elevó la tasa de "
                f"éxito simulada de {pc(CINT['mantener']['exito'])} a {pc(CINT['modelo']['exito'])} "
                f"({sg(d)} puntos porcentuales, IC 95 % [{sg(lo)}; {sg(hi)}], {fp(pv)}), mientras que girar al azar "
                f"no produjo una mejora significativa.")
if V5:
    res_txt += (f" Con esa información se diseñó después una nueva cadena de políticas (v5) que, en el mismo mundo "
                f"simulado, eleva el éxito de {pc(VIS.exito.mean())} a {pc(V5['exito'])}; un predictor reentrenado "
                f"para ella alcanzó AUC = {f3(V5['t'].loc[V5['mejor'], 'AUC-ROC'])}, con probabilidades bien "
                "calibradas, mientras que el modelo de la cadena anterior conserva el orden pero subestima el éxito, "
                "por lo que el predictor debe reentrenarse con cada cambio de políticas.")
par(res_txt, sangria=False)
p = par("", sangria=True)
p.add_run("Palabras clave: ").italic = True
p.add_run("cosecha robótica, clasificación supervisada, bosque aleatorio, simulación, transferencia simulación-realidad")
salto()

# ═════════════════════════ INTRODUCCIÓN ═════════════════════════
par(TITULO, centro=True, negrita=True)
par("La cosecha de fruta es una de las tareas agrícolas que más mano de obra requiere, y su automatización es un "
    "problema abierto de la robótica. En una revisión de 50 proyectos de cosecha robótica, Bac et al. (2014) "
    "reportaron un éxito medio de cosecha del 66 % y tiempos de ciclo de unos 33 s por fruto, lejos de lo que "
    "exige la producción comercial. Los sistemas específicos para manzanas mejoraron estos valores (Silwal et al., "
    "2017), pero las revisiones recientes coinciden en que la variabilidad del entorno (oclusiones, frutos "
    "agrupados, ramas) sigue siendo la principal causa de fallos (Zhou et al., 2022).")
par("Este proyecto parte de un sistema real desarrollado en una tesis de grado: un robot humanoide Unitree G1 con "
    "mano de tres dedos Dex3 y una cámara RGB-D Intel RealSense D435i en la cabeza. La cámara detecta las "
    "manzanas; una política de aprendizaje por refuerzo (Sutton y Barto, 2018) lleva el brazo hasta la fruta; la "
    "mano la agarra, se desprende del pedúnculo tirando hacia atrás y se lleva a una canasta lateral. En las "
    "pruebas del robot real solo alrededor de un tercio de los intentos terminó con la manzana en la canasta. "
    "Saber de antemano qué intentos tienen pocas probabilidades de éxito permitiría ordenar las manzanas, evitar "
    "intentos inútiles y orientar el desarrollo hacia las etapas que más fallan.")
par("Predecir el resultado de una manipulación a partir de la percepción es un enfoque establecido: la síntesis "
    "de agarres basada en datos aprende qué configuraciones tienen éxito a partir de ejemplos (Bohg et al., 2014), "
    "y modelos de aprendizaje automático han predicho la calidad de un agarre a partir de la imagen (Lenz et al., "
    "2015). Como generar miles de intentos con el robot real no es viable, los datos se generan en simulación "
    "física, una práctica habitual en robótica (Mittal et al., 2023) que, sin embargo, enfrenta la brecha entre "
    "simulación y realidad (Tobin et al., 2017; Zhao et al., 2020). Para datos tabulares de tamaño moderado, los "
    "conjuntos de árboles suelen igualar o superar a las redes neuronales (Grinsztajn et al., 2022), lo que "
    "orienta la elección de modelos.")
par("El objetivo general es desarrollar una herramienta de clasificación que, antes de cada intento y con la "
    "información disponible para el robot, estime (a) si la manzana llegará a la canasta y (b) en qué etapa "
    "fallará el intento. Los objetivos específicos son: construir el conjunto de datos y caracterizar sus "
    "variables; definir el tipo de problema y elegir al menos tres modelos adecuados; compararlos con métricas "
    "justificadas e intervalos de confianza; verificar cualitativamente la transferencia al robot real con sus "
    "registros; y evaluar si el modelo puede usarse para tomar una decisión que mejore la cosecha (el giro de "
    "cintura previo al intento).")

# ═════════════════════════ MÉTODO ═════════════════════════
titulo("Método", 1)
titulo("Plataforma y secuencia de cosecha", 2)
par("Los datos se generaron en Isaac Lab (Mittal et al., 2023) con el modelo del robot G1 (pelvis fija, control "
    "a 50 Hz, física a 200 Hz) y con la misma secuencia y las mismas políticas que ejecuta el robot real "
    "(script ml_project/recolectar_dataset.py, copia del simulador de cosecha de la tesis). Cada intento sigue "
    "las etapas: alcance (la política lleva la mano a un punto de preparación y luego a la manzana), agarre "
    "(preforma de la mano y cierre controlado por fuerza), desprendimiento (giro y tirón de 8 cm hacia atrás), "
    "retorno a la pose de espera, transporte hasta la canasta y depósito. El pedúnculo se modela como una unión "
    "que se rompe al superar un límite de fuerza, y la posición de la manzana que usa el robot incluye el error de "
    "percepción de la cámara.")
titulo("Generación del conjunto de datos", 2)
par(f"Cada intento se realiza sobre un árbol nuevo generado al azar, con entre 2 y 5 manzanas separadas al "
    f"menos 6 cm, y el robot tiene un único intento por árbol. Se registró una fila por intento: {n_total} intentos "
    "en tres corridas independientes (semillas aleatorias 0, 1 y 2). Las corridas 0 y 2 forman el conjunto de "
    f"entrenamiento y la corrida 1 el de prueba, de modo que la evaluación se hace sobre árboles nunca vistos. Se "
    f"descartaron {n_total - n_vis} intentos cuya manzana objetivo no se veía entera en la cámara (el robot real no "
    f"los intentaría), quedando {n_tr} intentos de entrenamiento (éxito {pc(exito_tr)}) y {n_te} de prueba (éxito "
    f"{pc(exito_te)}). Hay respuesta real para cada observación (el simulador registra el resultado), por lo que "
    "el problema admite aprendizaje supervisado.")
titulo("Variables", 2)
par("Las entradas son únicamente lo que el robot conoce antes de actuar (Tabla 1). Por decisión de diseño se "
    "excluyó la distancia al tronco, que el robot real todavía no estima con fiabilidad, y solo cuentan como "
    "vecinas las manzanas que la cámara ve enteras. Dos variables registradas se eliminaron por no aportar "
    "información: el radio de la manzana (constante por entorno de simulación) y el número de vecinas a menos de "
    "15 cm (siempre cero por la separación mínima). Cuando no hay vecina visible, la distancia a la más cercana se "
    "fija en 0,5 m y su desplazamiento en cero, con un indicador binario aparte.")
VARS = [
    ("yaw_cintura_deg", "Giro de la cintura (°)", "continua"),
    ("x/y/z_hombro_m", "Posición de la manzana respecto al hombro derecho, marco del torso (m)", "continua"),
    ("dist_hombro_m, dist_horiz_m", "Distancia 3D y horizontal hombro-manzana (m)", "continua"),
    ("azimut_deg, elevacion_deg", "Ángulos de la manzana vistos desde el hombro (°)", "continua"),
    ("u_px, v_px", "Píxel del centro de la manzana en la imagen de 424 × 240", "continua"),
    ("prof_cam_m", "Profundidad medida por la cámara (m)", "continua"),
    ("n_vecinos_vis", "Manzanas vecinas vistas enteras por la cámara", "discreta"),
    ("dist_vecino_min_m", "Distancia a la vecina visible más cercana (m)", "continua"),
    ("vec_dx/dy/dz_m", "Posición de esa vecina respecto al objetivo (m)", "continua"),
    ("n_vecinos_vis_debajo", "Vecinas visibles debajo del objetivo", "discreta"),
    ("hay_vecino_vis", "1 si la cámara ve al menos una vecina", "discreta (binaria)"),
    ("exito", "1 = manzana depositada en la canasta", "discreta (binaria) — dependiente"),
    ("etapa", "Etapa donde terminó el intento (7 clases)", "discreta (categórica) — dependiente"),
]
tabla("Variables del conjunto de datos", [list(v) for v in VARS], ["Variable", "Descripción", "Tipo"],
      nota="Las 18 primeras (filas 1 a 11) son independientes (entradas); las dos últimas son las variables "
           "dependientes (salidas).", anchos=[4.2, 8.3, 3.8])
titulo("Tipo de problema", 2)
par("Ambas salidas son categóricas, por lo que el problema es de clasificación: binaria para el éxito y "
    "multiclase para la etapa de fallo. Como cada intento tiene su resultado registrado, se usa aprendizaje "
    "supervisado; los enfoques no supervisados o semisupervisados (Chapelle et al., 2006) no se justifican cuando "
    "todas las observaciones están etiquetadas.")
titulo("Modelos", 2)
par("Se eligieron modelos de naturaleza distinta, todos disponibles en scikit-learn (Pedregosa et al., 2011), "
    "siguiendo el flujo de trabajo habitual de preparación, ajuste y validación (Bishop, 2006; Géron, 2019; "
    "Igual y Seguí, 2017), dentro de una tubería con estandarización de las entradas (Tabla 2). Como referencia mínima se usó un "
    "clasificador trivial que siempre predice la clase más frecuente.")
MODS = [
    ("Regresión logística (L2)", "Línea de base lineal e interpretable; probabilidades bien calibradas",
     "C ∈ {0,01; 0,1; 1; 10; 100}"),
    ("k vecinos más cercanos", "El éxito depende de la zona del espacio de trabajo (Cover y Hart, 1967)",
     "k ∈ {5; 11; 21; 41}, pesos"),
    ("SVM, núcleo RBF", "Frontera no lineal suave; el espacio alcanzable del brazo es curvo (Cortes y Vapnik, 1995)",
     "C ∈ {0,3; 1; 3; 10}, γ"),
    ("Bosque aleatorio", "Interacciones entre variables y robustez a la colinealidad (Breiman, 2001)",
     "profundidad, hojas mínimas"),
    ("Gradient boosting", "Muy preciso en datos tabulares; permite ver el desempeño por iteración (Friedman, 2001)",
     "árboles, tasa, profundidad"),
    ("Red neuronal (MLP)", "Aproximador no lineal general; ReLU, Adam (Kingma y Ba, 2015), parada temprana",
     "capas ocultas, α (L2)"),
]
tabla("Modelos comparados y su justificación", [list(m) for m in MODS], ["Modelo", "Justificación", "Hiperparámetros"],
      anchos=[3.8, 8.0, 4.5])
titulo("Validación y métricas", 2)
par("Los hiperparámetros se eligieron con búsqueda en grilla y validación cruzada estratificada de 5 "
    "particiones, solo sobre el conjunto de entrenamiento; el conjunto de prueba se usó una única vez al final. "
    "Como la salida es categórica, no corresponden las métricas de regresión (MSE, RMSE, R²). Para la salida "
    "binaria se usaron: el área bajo la curva ROC (AUC-ROC; Fawcett, 2006), que mide la capacidad de ordenar los "
    "intentos de más a menos probables sin depender de un umbral y es la métrica principal, porque el uso previsto "
    "es priorizar manzanas; la exactitud, la precisión, la exhaustividad y la F1 con umbral 0,5; y la puntuación "
    "de Brier (Brier, 1950), que es el error cuadrático medio de las probabilidades y mide su calibración. Los "
    "intervalos de confianza del 95 % se obtuvieron por remuestreo bootstrap del conjunto de prueba (Efron y "
    "Tibshirani, 1993), y la comparación entre dos modelos se hizo con bootstrap pareado sobre los mismos intentos. "
    "Para la salida multiclase, desbalanceada, se usaron la F1 macro (promedio sin ponderar de la F1 de cada clase) "
    "y la exactitud balanceada, con pesos de clase balanceados en el entrenamiento; las clases con menos del 3 % de "
    "los intentos se agruparon como «otro».")
par("El desempeño según la cantidad de entrenamiento se estudió con curvas de aprendizaje y, para los modelos "
    "iterativos, con el AUC por iteración del gradient boosting y por época de la red neuronal. La importancia de "
    "las variables se midió por permutación en el conjunto de prueba.")
titulo("Verificación con el robot real", 2)
par("Los registros de las corridas de cosecha autónoma del robot real (octubre de 2026, misma política del brazo) "
    "se procesaron con ml_project/intentos_reales.py: para cada manzana intentada se tomó la posición detectada por "
    "la cámara, el giro de cintura y el resultado, y con la cinemática del robot (modelo URDF) se calcularon las "
    f"mismas variables geométricas. Se obtuvieron {len(REAL)} intentos de {REAL.corrida.nunique()} corridas (éxito "
    f"{pc(REAL.exito.mean())}). Como las vecinas y el píxel no se registraron igual que en la simulación, se "
    "reentrenaron los modelos con solo las siete variables geométricas y se evaluaron sobre los intentos reales.")
titulo("Experimento de decisión: giro de cintura", 2)
par("Para evaluar el uso del modelo en la toma de decisiones se programó una variante del simulador "
    "(ml_project/cosecha_cintura.py). Al comenzar cada árbol, el bosque aleatorio evalúa los giros de cintura "
    "entre −15° y +15° (el rango de los datos, cada 2,5°), recalculando las entradas para cada giro con el árbol "
    "quieto; solo se consideran los giros en que la cámara ve la manzana entera. Se compararon tres grupos de 600 "
    "intentos con la misma semilla: «mantener» (el giro inicial, como en el conjunto de datos; línea base), "
    "«modelo» (el giro de mayor probabilidad predicha) y «aleatorio» (un giro visible al azar, como control). La "
    "diferencia de tasas de éxito se evaluó con intervalo bootstrap y prueba z de dos proporciones.")

if V5:
    titulo("Cadena de políticas v5 y predictor reentrenado", 2)
    par("Los resultados del primer modelo (qué manzanas fallan y en qué etapa) orientaron el diseño posterior de "
        "una nueva cadena de políticas del robot, llamada v5: un alcance que tiene en cuenta la fruta para no "
        "empujarla, un mapa de apuntado que elige el giro de cintura (de −45° a +15°) y la dirección de la palma "
        "para cada manzana, el agarre aprendido con los dedos fijos una vez cerrados, un giro de muñeca de 60° y "
        "dedos más rígidos antes de tirar del pedúnculo. El mapa de apuntado es a su vez un modelo aprendido: una "
        "red que estima la probabilidad Q de que el alcance llegue a la manzana, ajustada con decenas de miles de "
        "alcances simulados; solo mira la etapa de alcance.")
    par("Con esa cadena se generó un segundo conjunto de datos (ml_project/cosecha_v5.py) con el mismo mundo "
        "simulado y las mismas 18 entradas, calculadas con la cintura ya girada al ángulo elegido; todas las "
        f"manzanas se intentan, sin descartar las de Q baja. Se obtuvieron {V5['n']} intentos válidos en tres "
        f"corridas (semillas {', '.join(str(s) for s in V5['semillas'])}): las dos primeras para entrenamiento "
        f"({V5['n_tr']}) y la última para prueba ({V5['n_te']}). Se reentrenaron los mismos modelos con la misma "
        "búsqueda de hiperparámetros y se compararon con dos referencias: la Q del mapa de apuntado y el bosque "
        "aleatorio de la cadena anterior aplicado sin reentrenar.")

# ═════════════════════════ RESULTADOS ═════════════════════════
titulo("Resultados", 1)
titulo("Análisis de las variables", 2)
filas_et = [[e, int(n), pc(n / len(VIS))] for e, n in etapas.items()]
tabla("Resultado de los intentos simulados (etapa en la que terminó)", filas_et, ["Etapa", "Intentos", "Proporción"],
      nota="«exito»: la manzana llegó a la canasta; el resto son etapas de fallo (alcance: la mano no llegó a menos "
           "de 5 cm; agarre: no quedó nada entre los dedos; desprendimiento: el pedúnculo resistió o la fruta se "
           "resbaló; retorno y transporte: cayó después de arrancarla; depósito: cayó fuera de la canasta).",
      anchos=[5, 3, 3])
par("Las clases están desbalanceadas: el éxito y los fallos de agarre y desprendimiento concentran el 84 % de los "
    "intentos, mientras que retorno y depósito no llegan al 3 %. Las distribuciones individuales (en el cuaderno) "
    "muestran que la mayoría de las entradas son continuas y aproximadamente unimodales, y que la altura de la "
    "manzana respecto al hombro es la que mejor separa éxitos de fracasos (Figura 1): por debajo de 0,42 m bajo el "
    "hombro ningún intento tuvo éxito, y las manzanas más altas se cosecharon en torno al 75 % de las veces.")
figura("exito_vs_altura.png", "Tasa de éxito según la altura de la manzana respecto al hombro",
       nota="Intentos simulados agrupados en ocho intervalos de igual ancho; n = intentos por intervalo.", ancho=15)
par("El mapa de correlaciones (Figura 2) confirma la relación con la altura (r = 0,44), la profundidad de la "
    "cámara (r = −0,42), la elevación (r = 0,40) y la distancia al hombro (r = −0,40). También muestra fuertes "
    "correlaciones entre entradas que describen la misma posición en sistemas distintos (por ejemplo, la altura y "
    "la fila del píxel), una colinealidad que afecta a los modelos lineales pero no a los basados en árboles.")
figura("correlaciones.png", "Mapa de correlaciones de Pearson entre las entradas y el éxito", ancho=13.5)

titulo("Clasificación binaria", 2)
filas_b = []
for m, r in tb.iterrows():
    fila = [m, f3(r["exactitud"]), f3(r["precisión"]), f3(r["exhaustividad"]), f3(r["F1"]),
            f"{f3(r['AUC-ROC'])} {r['IC95 AUC'].replace('.', ',').replace(', ', '; ')}", f3(r["Brier"])]
    if m == mejor_b:
        fila = [(x,) for x in fila]
    filas_b.append(fila)
tabla("Desempeño de los modelos en el conjunto de prueba (salida binaria)", filas_b,
      ["Modelo", "Exactitud", "Precisión", "Exhaust.", "F1", "AUC-ROC [IC 95 %]", "Brier"],
      nota=f"Conjunto de prueba: {n_te} intentos de una corrida independiente. En negrita, el mejor modelo por "
           "AUC-ROC. Brier: menor es mejor.", anchos=[3.6, 1.8, 1.8, 1.8, 1.5, 3.8, 1.5], tam=9)
par(f"Todos los modelos superaron claramente al clasificador trivial (Tabla 4). El bosque aleatorio obtuvo el mayor "
    f"AUC-ROC ({f3(tb.loc[mejor_b, 'AUC-ROC'])}), la mayor exactitud ({f3(tb.loc[mejor_b, 'exactitud'])}) y la "
    "mejor calibración (Brier = " + f3(tb.loc[mejor_b, "Brier"]) + "), seguido por el gradient boosting y la red "
    "neuronal. La regresión logística fue la más débil, lo que indica que la relación entre las entradas y el "
    "éxito no es lineal. Los intervalos de confianza de los mejores modelos se solapan, por lo que las diferencias "
    "pequeñas entre ellos no son concluyentes con 592 intentos de prueba. Las curvas ROC se muestran en la "
    "Figura 3.")
figura("roc.png", "Curvas ROC de los modelos en el conjunto de prueba", ancho=11)
par("Las curvas de aprendizaje (Figura 4) muestran que el bosque aleatorio ya está cerca de su techo con unos "
    "1000 intentos, mientras que el gradient boosting y la red neuronal siguen mejorando al aumentar los datos: "
    "más simulación podría acortar las diferencias. En el gradient boosting, el AUC de prueba se estabiliza tras "
    "unas 150 iteraciones mientras el de entrenamiento sigue subiendo, una señal de sobreajuste moderado.")
figura("curvas_aprendizaje.png", "Curvas de aprendizaje (AUC en validación cruzada según el tamaño del entrenamiento)",
       nota="Las bandas indican ± una desviación estándar entre las cinco particiones.", ancho=14)
par("La importancia por permutación del bosque aleatorio (Figura 5) coloca en primer lugar la distancia de la "
    "manzana al hombro, seguida de su posición lateral, la profundidad medida por la cámara y los ángulos. Las "
    "variables de las vecinas aportan poco, y el giro de cintura no aporta nada; esto último se explica porque, "
    "en la simulación, el árbol se genera girado junto con el torso, de modo que el giro no cambia la geometría "
    "relativa al brazo.")
figura("importancia.png", "Importancia de las variables por permutación (bosque aleatorio)",
       nota="Caída media del AUC-ROC de prueba al permutar cada variable (20 repeticiones); las barras de error "
            "son una desviación estándar.", ancho=12)
regla = pd.DataFrame({"u": np.linspace(0, 0.9, 19)})
par("Como uso práctico, se evaluó una regla de decisión: intentar solo las manzanas cuya probabilidad predicha "
    "supera un umbral (Figura 6). Con umbral 0,5, la tasa de éxito de los intentos realizados sube del 42 % al "
    "68 %, a cambio de intentar el 38 % de las manzanas; con umbral 0,3, el éxito es del 61 % intentando el 62 %. "
    "El umbral permite elegir el compromiso entre cosechar más fruta y no perder tiempo en intentos inviables.")
figura("regla_decision.png", "Regla de decisión: éxito de los intentos realizados y fracción de manzanas intentadas",
       nota="Conjunto de prueba, bosque aleatorio. La línea discontinua es la tasa de éxito sin regla (intentar todo).",
       ancho=13)

titulo("Red neuronal frente a bosque aleatorio", 2)
rf, rn = trn.loc["Bosque aleatorio"], trn.loc["Red neuronal (MLP)"]
par("La red seleccionada por validación cruzada tiene dos capas ocultas de 32 y 16 neuronas con α = 1,0 y se "
    "detuvo en la época 151 por parada temprana. Su AUC de prueba fue " + f3(tb.loc["Red neuronal (MLP)", "AUC-ROC"])
    + " frente a " + f3(tb.loc["Bosque aleatorio", "AUC-ROC"]) + " del bosque. El bootstrap pareado da una "
    "diferencia de +0,029 a favor del bosque, con IC 95 % [+0,011; +0,047]: la red fue igual o mejor en solo el "
    "0,1 % de los remuestreos, por lo que la diferencia es significativa. Además, la red es más sensible a la "
    "inicialización aleatoria (Tabla 5) y, por época (Figura 7), su AUC de prueba se estanca en torno a 0,79 "
    "mientras el de entrenamiento sigue subiendo, es decir, entrenar más solo la sobreajusta. Su ventaja es el "
    "costo: es mucho más pequeña y rápida, aunque 10 ms por predicción del bosque son despreciables frente a un "
    "intento de cosecha de unos 16 s. El resultado coincide con lo reportado para datos tabulares (Grinsztajn et "
    "al., 2022).")
tabla("Red neuronal frente a bosque aleatorio", [
    ["AUC-ROC con 5 semillas", rf["AUC prueba (5 semillas)"].replace(".", ","), rn["AUC prueba (5 semillas)"].replace(".", ",")],
    ["Tiempo de entrenamiento (s)", str(rf["entrenamiento [s]"]).replace(".", ","), str(rn["entrenamiento [s]"]).replace(".", ",")],
    ["Predicción de una manzana (ms)", str(rf["predicción de 1 manzana [ms]"]).replace(".", ","),
     str(rn["predicción de 1 manzana [ms]"]).replace(".", ",")],
    ["Tamaño", rf["tamaño"], rn["tamaño"]]],
    ["", "Bosque aleatorio", "Red neuronal"], anchos=[5.5, 5, 5])
figura("red_epocas.png", "Red neuronal: pérdida de entrenamiento y AUC por época",
       nota="La línea discontinua es el AUC de prueba del bosque aleatorio; la línea punteada, la época en que se "
            "detuvo la parada temprana.", ancho=15.5)

titulo("Clasificación multiclase: etapa del fallo", 2)
mejor_m = tm.drop("Trivial")["F1 macro"].idxmax()
filas_m = []
for m, r in tm.iterrows():
    fila = [m, f3(r["exactitud"]), f3(r["exactitud balanceada"]), f3(r["F1 macro"]), f3(r["F1 macro VC (entren.)"])]
    filas_m.append([(x,) for x in fila] if m == mejor_m else fila)
tabla("Desempeño en la predicción de la etapa del fallo (conjunto de prueba)", filas_m,
      ["Modelo", "Exactitud", "Exactitud balanceada", "F1 macro", "F1 macro (validación cruzada)"],
      nota="Seis clases: éxito, alcance, agarre, desprendimiento, transporte y «otro» (retorno + depósito). "
           "El trivial predice siempre «éxito».", anchos=[4, 2.6, 3.2, 2.4, 3.8])
par(f"El bosque aleatorio también fue el mejor en la salida multiclase (F1 macro = {f3(tm.loc[mejor_m, 'F1 macro'])} "
    f"frente a {f3(tm.loc['Trivial', 'F1 macro'])} del trivial). Por clase (Figura 8), el desempeño es muy desigual: "
    "los fallos de alcance se predicen casi perfectamente (F1 = 0,93), porque dependen de si la manzana está dentro "
    "del espacio alcanzable; los de agarre razonablemente (F1 = 0,67); los de desprendimiento solo en parte "
    "(F1 = 0,41); y las caídas durante el transporte o el retorno prácticamente no se pueden anticipar (F1 ≤ 0,09). "
    "Estas últimas dependen de la calidad del agarre logrado, que no se conoce antes de intentar.")
figura("confusion_multiclase.png", "Matriz de confusión normalizada por fila del bosque aleatorio (etapa del fallo)",
       ancho=11)

titulo("Verificación con el robot real", 2)
filas_r = []
mejor_r = treal["AUC robot real"].idxmax()
for m, r in treal.iterrows():
    fila = [m, f3(r["AUC simulación (prueba)"]),
            f"{f3(r['AUC robot real'])} {r['IC95 real'].replace('.', ',').replace(', ', '; ')}",
            f3(r["Brier real"]), f3(r["P media predicha"])]
    filas_r.append([(x,) for x in fila] if m == mejor_r else fila)
tabla("Modelos geométricos entrenados en simulación, evaluados en simulación y en el robot real", filas_r,
      ["Modelo", "AUC simulación", "AUC robot real [IC 95 %]", "Brier real", "P media predicha"],
      nota=f"Robot real: {len(REAL)} manzanas intentadas, éxito observado {pc(REAL.exito.mean())}. Solo las siete "
           "variables geométricas (posición respecto al hombro, distancias y ángulos).",
      anchos=[3.8, 2.6, 4.6, 2.3, 2.7], tam=9)
par("Todos los modelos ordenan los intentos reales mejor que el azar, pero el orden de los modelos se invierte "
    "respecto de la simulación (Tabla 7, Figura 9). La regresión logística, la más débil en simulación, es la "
    f"mejor en el robot real (AUC = {f3(treal.loc['Regresión logística', 'AUC robot real'])}) y mantiene su "
    "desempeño; los modelos más flexibles (SVM, bosque, boosting) caen a valores entre 0,56 y 0,64. Una "
    "interpretación es que estos modelos aprenden detalles propios del simulador (por ejemplo, el borde exacto "
    "del espacio alcanzable) que no se cumplen en el robot real, mientras que la frontera suave de la regresión "
    "logística captura la tendencia general —las manzanas bajas y lejanas fallan— que sí se transfiere. La "
    "calibración en el robot real es pobre: los modelos sobreestiman el éxito, coherente con que el robot real "
    "rinde peor que la simulación. Estos resultados son cualitativos: los intentos reales son pocos, mezclan "
    "versiones del agarre e incluyen giros de cintura de hasta −45°, fuera del rango simulado.")
figura("robot_real.png", "Transferencia al robot real: AUC por modelo y calibración del mejor modelo", ancho=15.5)

titulo("Uso del modelo para decidir el giro de cintura", 2)
if CINT and all(k in CINT for k in ("modelo", "mantener")):
    filas_c = []
    for modo in ("mantener", "modelo", "aleatorio"):
        if modo in CINT:
            c = CINT[modo]
            filas_c.append([modo, c["n"], pc(c["exito"]), f"[{pc(c['lo'])}; {pc(c['hi'])}]", f3(c["p_usado"]),
                            pc(c["giro"], 0)])
    tabla("Tasa de éxito simulada según cómo se elige el giro de cintura", filas_c,
          ["Grupo", "Intentos", "Éxito", "IC 95 %", "P predicha media", "Intentos con giro"],
          nota="Misma semilla y mismo procedimiento en los tres grupos; árboles independientes.",
          anchos=[2.8, 2, 2.2, 3.8, 3, 2.6])
    txt = []
    for a, b in (("modelo", "mantener"), ("modelo", "aleatorio"), ("aleatorio", "mantener")):
        if a in CINT and b in CINT:
            d, lo, hi, pv = comp(a, b)
            txt.append(f"«{a}» − «{b}»: {sg(d)} puntos, IC 95 % [{sg(lo)}; {sg(hi)}], {fp(pv)}")
    par("Las diferencias entre grupos fueron: " + "; ".join(txt) + ".")
    cm, cb, ca = CINT["modelo"], CINT["mantener"], CINT.get("aleatorio")
    par(f"Elegir el giro con el modelo aumentó la tasa de éxito de {pc(cb['exito'])} a {pc(cm['exito'])}, una mejora "
        f"relativa del {100 * (cm['exito'] / cb['exito'] - 1):.0f} %. El control aleatorio no difiere "
        "significativamente de la línea base y queda por debajo del modelo: la mejora proviene de la elección del "
        "modelo y no del simple hecho de girar. La probabilidad predicha media coincidió con el éxito observado en "
        f"los tres grupos ({f3(cb['p_usado'])} frente a {f3(cb['exito'])} en la línea base, {f3(cm['p_usado'])} "
        f"frente a {f3(cm['exito'])} con el modelo" + (f" y {f3(ca['p_usado'])} frente a {f3(ca['exito'])} con el "
        f"giro aleatorio" if ca else "") + "), es decir, el modelo anticipó correctamente cuánto iba a mejorar, aun "
        "evaluando configuraciones (cintura girada con el árbol quieto) que no aparecían como tales en los datos de "
        f"entrenamiento. El modelo giró la cintura en el {pc(cm['giro'], 0)} de los intentos, con una mediana de "
        f"{cm['giro_abs']:.0f}° (Figura 10), y en uno de cada cinco eligió el borde del rango (−15°), lo que sugiere "
        "que ampliar el rango podría mejorar más. La mejora se concentró en los fallos de agarre, que bajaron de 162 "
        "a 91. Como cada grupo usa árboles distintos, la comparación no es pareada, pero el intervalo de confianza "
        "excluye el cero con margen. El experimento es solo en simulación: no se probó en el robot real.")
    figura("cintura.png", "Éxito observado y probabilidad predicha según la elección del giro de cintura", ancho=15)
else:
    par("[Resultados pendientes: el experimento se está ejecutando.]")

if V5:
    titulo("Predictor reentrenado para la cadena v5", 2)
    et5 = V5["et"]
    par(f"La cadena v5 cosechó el {pc(V5['exito'])} de los intentos, frente al {pc(et5.loc['exito', 'cadena v1'])} de "
        "la cadena original en el mismo mundo simulado. Los fallos de agarre bajaron de "
        f"{pc(et5.loc['agarre', 'cadena v1'])} a {pc(et5.loc['agarre', 'cadena v5'])} y los de desprendimiento de "
        f"{pc(et5.loc['desprendimiento', 'cadena v1'])} a {pc(et5.loc['desprendimiento', 'cadena v5'])}, justamente "
        "las etapas que el primer modelo señalaba como más frecuentes.")
    t5 = V5["t"]
    orden = [i for i in t5.index if i != "Trivial"] + ["Trivial"]
    filas5 = [[i, f3(t5.loc[i, "AUC-ROC"]), t5.loc[i, "IC95 AUC"].replace(".", ",").replace(", ", "; "),
               f3(t5.loc[i, "exactitud"]), f3(t5.loc[i, "exactitud balanceada"]), f3(t5.loc[i, "F1"]),
               f3(t5.loc[i, "Brier"])] for i in orden]
    tabla("Desempeño en el conjunto de prueba de la cadena v5 (salida binaria)", filas5,
          ["Modelo", "AUC-ROC", "IC 95 % AUC", "Exactitud", "Exact. balanceada", "F1", "Brier"],
          nota="Las dos filas de referencia no se entrenaron con los datos v5: la Q del mapa de apuntado es la "
               "predicción de alcance que la cadena ya trae, y el bosque de la cadena v1 es el modelo de la "
               "sección anterior aplicado tal cual.", anchos=[5.2, 1.7, 2.9, 1.8, 1.9, 1.4, 1.4], tam=9)
    mejor5 = V5["mejor"]
    c5 = V5["comp"].set_index("comparación")
    cq = c5.loc[f"{mejor5} − Q del mapa de apuntado"]
    cv1 = c5.loc[f"{mejor5} − Bosque de la cadena v1"]
    tv, tq, tm = t5.loc["Bosque de la cadena v1, sin reentrenar"], t5.loc["Q del mapa de apuntado (referencia)"], t5.loc[mejor5]
    ic = lambda x: x.replace(".", ",").replace(", ", "; ")
    par(f"Todos los modelos reentrenados (Tabla {N_TABLA[0]}) quedaron entre AUC = {f3(t5.drop(['Trivial']).iloc[:6]['AUC-ROC'].min())} y "
        f"{f3(t5.drop(['Trivial']).iloc[:6]['AUC-ROC'].max())}, con intervalos que se superponen; el mejor en validación "
        f"cruzada fue el {mejor5.lower()} (AUC = {f3(tm['AUC-ROC'])}, exactitud {f3(tm['exactitud'])} frente a "
        f"{f3(t5.loc['Trivial', 'exactitud'])} del trivial). Las dos referencias ordenan los intentos igual de bien: "
        f"la Q del mapa de apuntado (AUC = {f3(tq['AUC-ROC'])}; diferencia {cq['diferencia AUC']:+.3f}".replace(".", ",") + f", "
        f"IC 95 % {ic(cq['IC95'])}) y el bosque de la cadena anterior sin reentrenar (AUC = {f3(tv['AUC-ROC'])}; "
        f"diferencia IC 95 % {ic(cv1['IC95'])}). Con las políticas v5 lo que se puede anticipar es, sobre todo, si "
        "el brazo alcanza la manzana, y eso ya lo captan la geometría y el mapa.")
    par(f"La diferencia está en la calibración (Figura {N_FIG[0] + 1}, panel derecho). El modelo reentrenado tiene la "
        f"menor puntuación de Brier ({f3(tm['Brier'])}), significativamente menor que la de la Q del mapa "
        f"({f3(tq['Brier'])}; diferencia IC 95 % {ic(cq['IC95 Brier'])}), que solo mide el alcance y sobreestima el "
        f"éxito, y que la del bosque anterior ({f3(tv['Brier'])}; IC 95 % {ic(cv1['IC95 Brier'])}), que lo subestima "
        "porque aprendió con una cadena que fallaba mucho más. Sus probabilidades, por lo tanto, se pueden leer "
        "directamente como tasas de éxito esperadas, que es lo que hace falta para fijar un umbral de decisión o "
        "estimar el rendimiento de una sesión de cosecha. Al cambiar las políticas cambia la relación entre las "
        "entradas y el resultado: el orden se conserva, pero las probabilidades deben reentrenarse.")
    rg = V5["regla"].set_index("umbral")
    filas_r5 = [[f"{u:.1f}".replace(".", ","), pc(r["intentados"], 0), pc(r["éxito de los intentados"]),
                 pc(r["cosechas perdidas (de las posibles)"])] for u, r in rg.iterrows()]
    tabla(f"Regla de decisión con {mejor5.lower()} en la cadena v5 (conjunto de prueba)", filas_r5,
          ["Umbral de P", "Manzanas intentadas", "Éxito de los intentos", "Cosechas posibles perdidas"],
          nota="Umbral 0: se intentan todas las manzanas (situación actual).", anchos=[3, 4, 4, 4.5])
    r5 = rg.loc[0.5]
    par(f"Como regla de decisión (Tabla {N_TABLA[0]}), intentar solo las manzanas con P ≥ 0,5 sube el éxito de los "
        f"intentos realizados de {pc(rg.loc[0.0, 'éxito de los intentados'])} a {pc(r5['éxito de los intentados'])}, "
        f"intentando el {pc(r5['intentados'], 0)} de las manzanas y perdiendo solo el "
        f"{pc(r5['cosechas perdidas (de las posibles)'])} de las cosechas posibles. Con la cadena v1 la misma regla "
        "descartaba la mayoría de las manzanas; con la v5, al ser mucho más alto el éxito de base, el modelo sirve "
        "sobre todo para descartar los casos claramente fuera de alcance y para estimar el rendimiento esperado.")
    figura("cadena_v5.png", "Cadena v5: resultado de los intentos, curvas ROC y calibración del predictor reentrenado",
           ancho=15.5)

# ═════════════════════════ DISCUSIÓN Y CONCLUSIONES ═════════════════════════
titulo("Discusión y conclusiones", 1)
par("Es posible predecir, antes de intentar y con información disponible para el robot, si una cosecha "
    "simulada terminará con la manzana en la canasta: el mejor modelo (bosque aleatorio) ordena los intentos "
    "con AUC-ROC = " + f3(tb.loc[mejor_b, "AUC-ROC"]) + " y, como regla de decisión, eleva el éxito de los intentos "
    "realizados del 42 % al 68 %. La geometría de la manzana respecto al hombro —sobre todo la altura y la "
    "distancia— explica la mayor parte de lo predecible; la presencia de vecinas visibles aporta poco.")
par("Los conjuntos de árboles fueron los mejores en simulación y superaron significativamente a la red neuronal, "
    "lo que coincide con la literatura sobre datos tabulares. Sin embargo, la verificación con el robot real "
    "muestra que el mejor modelo en simulación no es necesariamente el mejor en la realidad: la regresión "
    "logística transfirió mejor. Para un uso en el robot conviene preferir modelos simples o regularizados, o "
    "reentrenarlos con datos reales.")
if CINT and "modelo" in CINT and "mantener" in CINT:
    d, lo, hi, pv = comp("modelo", "mantener")
    par("El modelo no solo sirve para predecir: usado para decidir el giro de cintura antes de cada intento, "
        f"aumentó la tasa de éxito simulada en {sg(d)} puntos ({pc(CINT['mantener']['exito'])} → "
        f"{pc(CINT['modelo']['exito'])}), con probabilidades predichas que coincidieron con el éxito observado. "
        "Es una aplicación directa al robot, que ya puede girar la cintura, aunque falta validarla en el mundo real.")
if V5:
    par("El modelo también orientó el desarrollo: la cadena de políticas v5, diseñada después de identificar dónde "
        f"fallaban los intentos, pasó de {pc(VIS.exito.mean())} a {pc(V5['exito'])} de éxito en simulación. Para esa cadena "
        f"el predictor reentrenado alcanzó AUC = {f3(V5['t'].loc[V5['mejor'], 'AUC-ROC'])}, similar al del mapa de "
        "apuntado y al del modelo anterior, pero con probabilidades bien calibradas, a diferencia de ambos. El "
        "predictor está atado a las políticas con que se generaron los datos: el procedimiento (simular, "
        "registrar, reentrenar) debe repetirse con cada versión.")
par("La etapa del fallo solo es predecible cuando depende de la geometría (alcance y, en parte, agarre). Las "
    "caídas después de arrancar la manzana dependen de la calidad del agarre logrado y no pueden anticiparse "
    "con la información previa; para predecirlas haría falta información durante el intento (fuerzas en los "
    "dedos), lo que convierte el problema en otro distinto.")
titulo("Limitaciones", 2)
lista([
    "Los datos de entrenamiento son solo de simulación; la física del agarre y del pedúnculo es aproximada.",
    "La verificación real usa pocos intentos (190), con versiones del código mezcladas y sin las variables de "
    "vecinas ni de píxel; sus conclusiones son cualitativas.",
    "No se usó la distancia al tronco ni las oclusiones entre manzanas, que en un huerto real influyen.",
    "El modelo vale para las políticas actuales del robot: si se reentrena el brazo, hay que volver a generar datos.",
    "Las clases poco frecuentes (retorno, depósito) no tienen ejemplos suficientes para aprenderse.",
])
titulo("Trabajo futuro", 2)
lista([
    "Registrar en el robot real las mismas variables que en simulación (vecinas, píxel) para validar el modelo "
    "completo y reentrenarlo con datos reales.",
    "Ampliar el rango de giros de cintura en la simulación y usar el modelo para elegir, además, la orientación "
    "de la mano.",
    "Incorporar información del propio intento (fuerzas del agarre) para anticipar las caídas en el transporte.",
])

# ═════════════════════════ REFERENCIAS ═════════════════════════
salto()
titulo("Referencias", 1)
REFS = [
    "Bac, C. W., van Henten, E. J., Hemming, J., y Edan, Y. (2014). Harvesting robots for high-value crops: "
    "State-of-the-art review and challenges ahead. *Journal of Field Robotics, 31*(6), 888–911. "
    "https://doi.org/10.1002/rob.21525",
    "Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.",
    "Bohg, J., Morales, A., Asfour, T., y Kragic, D. (2014). Data-driven grasp synthesis—A survey. *IEEE "
    "Transactions on Robotics, 30*(2), 289–309. https://doi.org/10.1109/TRO.2013.2289018",
    "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, "
    "78*(1), 1–3.",
    "Chapelle, O., Schölkopf, B., y Zien, A. (2006). *Semi-supervised learning*. MIT Press.",
    "Cortes, C., y Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. "
    "https://doi.org/10.1007/BF00994018",
    "Cover, T., y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information "
    "Theory, 13*(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964",
    "Efron, B., y Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
    "Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. "
    "https://doi.org/10.1016/j.patrec.2005.10.010",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of "
    "Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2.ª ed.). O'Reilly Media.",
    "Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning "
    "on typical tabular data? *Advances in Neural Information Processing Systems, 35*, 507–520.",
    "Igual, L., y Seguí, S. (2017). *Introduction to data science: A Python approach to concepts, techniques and "
    "applications*. Springer.",
    "Kingma, D. P., y Ba, J. (2015). Adam: A method for stochastic optimization. En *Proceedings of the 3rd "
    "International Conference on Learning Representations (ICLR)*.",
    "Lenz, I., Lee, H., y Saxena, A. (2015). Deep learning for detecting robotic grasps. *The International "
    "Journal of Robotics Research, 34*(4–5), 705–724. https://doi.org/10.1177/0278364914549607",
    "Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J. L., Singh, R., Guo, Y., Mazhar, H., "
    "Mandlekar, A., Babich, B., State, G., Hutter, M., y Garg, A. (2023). Orbit: A unified simulation framework "
    "for interactive robot learning environments. *IEEE Robotics and Automation Letters, 8*(6), 3740–3747. "
    "https://doi.org/10.1109/LRA.2023.3270034",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, "
    "P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y "
    "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, "
    "2825–2830.",
    "Silwal, A., Davidson, J. R., Karkee, M., Mo, C., Zhang, Q., y Lewis, K. (2017). Design, integration, and "
    "field evaluation of a robotic apple harvester. *Journal of Field Robotics, 34*(6), 1140–1159. "
    "https://doi.org/10.1002/rob.21715",
    "Sutton, R. S., y Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). MIT Press.",
    "Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., y Abbeel, P. (2017). Domain randomization for "
    "transferring deep neural networks from simulation to the real world. En *2017 IEEE/RSJ International "
    "Conference on Intelligent Robots and Systems (IROS)* (pp. 23–30). https://doi.org/10.1109/IROS.2017.8202133",
    "Zhao, W., Queralta, J. P., y Westerlund, T. (2020). Sim-to-real transfer in deep reinforcement learning for "
    "robotics: A survey. En *2020 IEEE Symposium Series on Computational Intelligence (SSCI)* (pp. 737–744). "
    "https://doi.org/10.1109/SSCI47803.2020.9308468",
    "Zhou, H., Wang, X., Au, W., Kang, H., y Chen, C. (2022). Intelligent robots for fruit harvesting: Recent "
    "developments and future challenges. *Precision Agriculture, 23*(5), 1856–1907. "
    "https://doi.org/10.1007/s11119-022-09913-3",
]
for ref in REFS:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)
    _runs(p, ref)

# ═════════════════════════ ANEXO ═════════════════════════
salto()
titulo("Anexo: código fuente entregado", 1)
lista([
    "recolectar_dataset.py — simulación de la cosecha y registro del conjunto de datos (una fila por intento).",
    "analisis_cosecha.ipynb — análisis de variables, modelos, comparación, verificación real y experimento de cintura.",
    "intentos_reales.py — extracción de los intentos del robot real a partir de sus registros.",
    "exportar_bosque.py y bosque_np.py — exportación del bosque aleatorio para usarlo dentro de la simulación.",
    "cosecha_cintura.py — experimento de elección del giro de cintura con el modelo.",
    "cosecha_v5.py y analisis_v5.py — conjunto de datos de la cadena v5 y predictor reentrenado.",
    "datos/, datos_reales/, datos_cintura/, datos_v5/ — conjuntos de datos generados (CSV).",
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
