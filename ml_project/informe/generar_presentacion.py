#!/usr/bin/env python3
"""Presentación del proyecto final (16:9, .pptx editable; se importa en Google Slides). El guión de cada
diapositiva va en las notas del orador. Usa las figuras de ml_project/figuras/ y los números del informe.

    <python con python-pptx> ml_project/informe/generar_presentacion.py
"""
import os

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

AQUI = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(AQUI)
FIG = os.path.join(ML, "figuras")
AUTOR = os.environ.get("INFORME_AUTOR", "[Nombre y apellido]")
OUT = os.path.join(AQUI, "presentacion_proyecto_final.pptx")

# ── paleta: verde de hoja (dominante), rojo manzana (acento) ──
VERDE_OSC = RGBColor(0x1E, 0x3A, 0x2B)
VERDE = RGBColor(0x4E, 0x7D, 0x4A)
VERDE_CLARO = RGBColor(0xE9, 0xF1, 0xE6)
ROJO = RGBColor(0xC0, 0x39, 0x2B)
ROJO_CLARO = RGBColor(0xF8, 0xE3, 0xE0)
TINTA = RGBColor(0x1F, 0x2A, 0x24)
GRIS = RGBColor(0x5F, 0x6B, 0x66)
GRIS_BARRA = RGBColor(0xB4, 0xBD, 0xB8)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
TIT, CUERPO = "Cambria", "Calibri"

tb = pd.read_csv(os.path.join(FIG, "tabla_binaria.csv"), index_col=0)
tr = pd.read_csv(os.path.join(FIG, "tabla_robot_real.csv"), index_col=0)
t5 = pd.read_csv(os.path.join(FIG, "tabla_binaria_v5.csv"), index_col=0)
tc = pd.read_csv(os.path.join(FIG, "tabla_cintura.csv"), index_col=0)
c = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = 13.333, 7.5
VACIA = prs.slide_layouts[6]


def fondo(s, color):
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = color


def texto(s, x, y, w, h, runs, size=16, color=TINTA, font=CUERPO, bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, italic=False, nombre=None):
    """runs: str o lista de párrafos; cada párrafo str o lista de (texto, {opciones})."""
    tbx = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if nombre:
        tbx.name = nombre
    tf = tbx.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    pars = runs if isinstance(runs, list) else [runs]
    for k, p in enumerate(pars):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        para.alignment = align
        trozos = p if isinstance(p, list) else [(p, {})]
        for t, o in trozos:
            r = para.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
        para.space_after = Pt(o.get("after", 6) if trozos else 6)
    return tbx


def caja(s, x, y, w, h, color, borde=None, forma=MSO_SHAPE.ROUNDED_RECTANGLE, radio=0.08):
    sh = s.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if borde is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = borde
        sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    if forma == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radio
    return sh


def titulo(s, t, sub=None):
    """Título de diapositiva de contenido, siempre en la misma posición; el punto rojo es el motivo."""
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(0.62), Inches(0.26), Inches(0.26))
    d.fill.solid()
    d.fill.fore_color.rgb = ROJO
    d.line.fill.background()
    texto(s, 1.05, 0.42, 11.6, 0.7, t, size=32, color=VERDE_OSC, font=TIT, bold=True, nombre="Título")
    if sub:
        texto(s, 1.05, 1.12, 11.6, 0.45, sub, size=16, color=GRIS)


def imagen(s, archivo, x, y, w=None, h=None):
    ruta = os.path.join(FIG, archivo)
    return s.shapes.add_picture(ruta, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def nueva(notas, color=BLANCO):
    s = prs.slides.add_slide(VACIA)
    fondo(s, color)
    s.notes_slide.notes_text_frame.text = notas
    return s


def numero(s, x, y, n, color=ROJO, d=0.55, size=20):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    o.fill.solid()
    o.fill.fore_color.rgb = color
    o.line.fill.background()
    tf = o.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.name = Pt(size), True, CUERPO
    r.font.color.rgb = BLANCO
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return o


def estilo_grafico(ch, size=12):
    ch.font.name = CUERPO
    ch.font.size = Pt(size)
    ch.font.color.rgb = GRIS


def barras(s, x, y, w, h, cats, vals, colores, formato='0.000', tipo=XL_CHART_TYPE.BAR_CLUSTERED, vmax=1.0,
           eje=True):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("valor", vals)
    gf = s.shapes.add_chart(tipo, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    estilo_grafico(ch)
    ch.has_legend = False
    ch.has_title = False
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = formato, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(13), True, TINTA
    for k, col in enumerate(colores):
        pt = pl.series[0].points[k]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = vmax, 0
    va.has_major_gridlines = False
    va.visible = eje
    va.tick_labels.font.size = Pt(11)
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(13)
    ca.tick_labels.font.color.rgb = TINTA
    ca.format.line.color.rgb = GRIS_BARRA
    if tipo == XL_CHART_TYPE.BAR_CLUSTERED:
        ca.reverse_order = True
        va.visible = False
    return ch


# ═════════════════════════ 1. portada ═════════════════════════
s = nueva("Buenos días. Voy a presentar mi proyecto final: un modelo que predice, antes de que el robot intente "
          "cosechar una manzana, si ese intento va a terminar con la manzana en la canasta. Los datos salen de mi "
          "tesis, donde un robot humanoide cosecha manzanas, pero este proyecto se puede entender por sí solo: "
          "el foco es el problema de ciencia de datos.", VERDE_OSC)
o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), Inches(1.55), Inches(0.42), Inches(0.42))
o.fill.solid()
o.fill.fore_color.rgb = ROJO
o.line.fill.background()
texto(s, 0.9, 2.2, 11.5, 1.2, "¿Va a llegar a la canasta?", size=54, color=BLANCO, font=TIT, bold=True)
texto(s, 0.9, 3.45, 11.0, 1.0, "Predicción del éxito de la cosecha robótica de manzanas antes de cada intento",
      size=24, color=RGBColor(0xCF, 0xE0, 0xCB))
texto(s, 0.9, 5.3, 11.5, 1.4, [[(AUTOR, {"bold": True, "color": BLANCO})],
                               "Ciencia de Datos e IA aplicados a la Mecatrónica · UTEC ITR Suroeste · Prof. Giovani Bolzan Cogo",
                               "26 de noviembre de 2026"], size=16, color=RGBColor(0xCF, 0xE0, 0xCB))

# ═════════════════════════ 2. el problema ═════════════════════════
s = nueva("El sistema: un robot humanoide Unitree G1 con una mano de tres dedos y una cámara en la cabeza. Para "
          "cada manzana, la cámara la ubica, una red neuronal entrenada por refuerzo mueve el brazo hasta ella, la "
          "mano la agarra, el brazo tira para cortar el pedúnculo, vuelve y la deja en una canasta. El problema es "
          "que en el robot real solo un tercio de los intentos terminaba bien, y cualquiera de estas etapas puede "
          "fallar. La pregunta del proyecto es: con lo que el robot ya sabe antes de empezar, ¿podemos predecir si "
          "el intento va a salir bien, y en qué etapa fallaría?")
titulo(s, "El problema")
etapas = ["Alcance", "Agarre", "Despren-\ndimiento", "Retorno", "Transporte", "Canasta"]
x0, ancho, paso = 0.6, 1.98, 2.03
for k, e in enumerate(etapas):
    col = VERDE if k < 5 else ROJO
    caja(s, x0 + k * paso, 1.75, ancho, 0.95, col, forma=MSO_SHAPE.CHEVRON if k else MSO_SHAPE.PENTAGON)
    texto(s, x0 + k * paso + (0.08 if k == 0 else 0.3), 1.75, ancho - (0.3 if k == 0 else 0.5), 0.95, [[(l, {"after": 0})] for l in e.split("\n")],
          size=14 if k != 2 else 13, color=BLANCO, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
texto(s, 0.75, 2.85, 12, 0.4, "Cada etapa puede fallar: la mano no llega, no agarra, el pedúnculo no se corta, "
      "la manzana se cae…", size=16, color=GRIS, italic=True)
caja(s, 0.75, 3.75, 5.4, 3.1, VERDE_CLARO)
texto(s, 1.1, 3.95, 4.8, 1.4, "≈ 1 de 3", size=72, color=ROJO, font=TIT, bold=True)
texto(s, 1.1, 5.45, 4.8, 1.2, "intentos del robot real terminaban con la manzana en la canasta", size=18,
      color=TINTA)
caja(s, 6.6, 3.75, 6.0, 3.1, VERDE_OSC)
texto(s, 7.0, 4.05, 5.3, 2.6, [[("Pregunta", {"size": 16, "color": RGBColor(0xCF, 0xE0, 0xCB), "bold": True})],
                               [("Antes de intentar, con lo que el robot ya sabe, ¿se puede predecir…", {})],
                               [("① si la manzana llega a la canasta?", {"bold": True})],
                               [("② en qué etapa fallaría?", {"bold": True})]], size=20, color=BLANCO)

# ═════════════════════════ 3. los datos ═════════════════════════
s = nueva("Como no es posible hacer miles de intentos con el robot real, los datos se generan en el simulador "
          "Isaac Lab, con el mismo robot y las mismas políticas que usa el robot real. Cada intento es sobre un "
          "árbol nuevo generado al azar, y se guarda como una fila: 18 variables medidas antes de empezar y el "
          "resultado. Son 2604 intentos en tres corridas independientes: dos para entrenar y la tercera solo para "
          "la prueba final, así el modelo se evalúa en árboles que nunca vio. Descarté los intentos en que la "
          "cámara no veía la manzana entera, porque el robot real no los intentaría.")
titulo(s, "Los datos: un dataset propio, simulado")
stats = [("2 604", "intentos de cosecha simulados", "una fila por intento, árbol nuevo cada vez"),
         ("18", "entradas conocidas antes de intentar", "posición, imagen, cintura, vecinas"),
         ("2", "salidas a predecir", "¿llega a la canasta? · ¿en qué etapa falla?")]
for k, (n, a, b) in enumerate(stats):
    x = 0.75 + k * 4.05
    caja(s, x, 1.85, 3.75, 3.2, VERDE_CLARO)
    texto(s, x + 0.35, 2.05, 3.1, 1.3, n, size=66, color=VERDE_OSC, font=TIT, bold=True)
    texto(s, x + 0.35, 3.4, 3.1, 0.8, a, size=18, color=TINTA, bold=True)
    texto(s, x + 0.35, 4.25, 3.1, 0.7, b, size=14, color=GRIS)
caja(s, 0.75, 5.45, 7.95, 1.1, VERDE)
texto(s, 1.05, 5.45, 7.4, 1.1, [[("Entrenamiento  ", {"bold": True}), ("2 corridas · 1 971 intentos", {})]],
      size=18, color=BLANCO, anchor=MSO_ANCHOR.MIDDLE)
caja(s, 8.95, 5.45, 3.6, 1.1, ROJO)
texto(s, 9.25, 5.45, 3.1, 1.1, [[("Prueba  ", {"bold": True}), ("1 corrida · 592", {})]], size=18, color=BLANCO,
      anchor=MSO_ANCHOR.MIDDLE)
texto(s, 0.75, 6.7, 12, 0.4, "Simulador Isaac Lab, mismo robot y mismas políticas que el robot real", size=12,
      color=GRIS, italic=True)

# ═════════════════════════ 4. arquitectura ═════════════════════════
s = nueva("Este diagrama resume todo el sistema. Arriba, lo que el robot sabe: la cámara detecta la manzana y sus "
          "vecinas, y el robot conoce la posición de su hombro y el giro de su cintura. Con eso armamos un vector de "
          "18 variables por manzana: 15 continuas y 3 discretas. Todas pasan por el mismo preprocesamiento: se "
          "rellenan los casos sin vecina visible y se estandarizan. Después, cada uno de los seis modelos produce "
          "dos salidas: una probabilidad de éxito entre 0 y 1, y la etapa en que terminaría el intento. Abajo, "
          "cómo se usa: P se lee como tasa de éxito esperada, sirve para decidir si intentar, para ordenar manzanas "
          "y para elegir el giro de cintura. Es aprendizaje supervisado porque cada intento tiene su resultado "
          "registrado.")
titulo(s, "Arquitectura del predictor")
imagen(s, "arquitectura_modelos.png", 0.75, 1.3, h=6.0)
items = [("Entradas", "18 variables tabulares:\n15 continuas, 3 discretas"),
         ("Modelos", "6 clasificadores supervisados\n+ 1 trivial de referencia"),
         ("Salidas", "P(éxito) ∈ [0, 1]\netapa: 6 clases"),
         ("Uso", "intentar si P ≥ umbral,\nordenar, elegir la cintura")]
for k, (a, b) in enumerate(items):
    y = 1.45 + k * 1.45
    numero(s, 6.05, y + 0.12, k + 1, color=ROJO if k == 3 else VERDE)
    texto(s, 6.85, y, 5.8, 1.3, [[(a, {"bold": True, "size": 20, "color": VERDE_OSC})], b], size=16)

# ═════════════════════════ 5. variables ═════════════════════════
s = nueva("Antes de los modelos, el análisis de variables. A la izquierda, la tasa de éxito según la altura de la "
          "manzana respecto al hombro: es la variable que mejor separa. Por debajo de 42 centímetros bajo el "
          "hombro, ningún intento tuvo éxito; las manzanas altas se cosechan tres de cada cuatro veces. A la "
          "derecha, el mapa de correlaciones: confirma la altura y la distancia como lo más relacionado con el "
          "éxito, y muestra que varias entradas describen la misma posición de formas distintas, por ejemplo la "
          "altura y la fila del píxel. Eso afecta a los modelos lineales, pero no a los de árboles.")
titulo(s, "Lo que más pesa: la altura de la manzana")
imagen(s, "exito_vs_altura.png", 0.75, 1.45, w=7.4)
imagen(s, "correlaciones.png", 8.45, 1.35, h=4.3)
caja(s, 0.75, 5.1, 7.4, 1.6, ROJO_CLARO)
texto(s, 1.05, 5.1, 6.9, 1.6, [[("0 %", {"bold": True, "size": 40, "color": ROJO, "font": TIT}),
                                ("   de éxito con la manzana más de 0,42 m por debajo del hombro", {"size": 18})]],
      anchor=MSO_ANCHOR.MIDDLE)
texto(s, 8.45, 5.8, 4.3, 0.9, "Correlación con el éxito: altura r = 0,44 · distancia r = −0,40", size=14,
      color=GRIS)

# ═════════════════════════ 6. binaria ═════════════════════════
orden = ["Bosque aleatorio", "Gradient boosting", "Red neuronal (MLP)", "SVM (RBF)", "k-NN",
         "Regresión logística", "Trivial"]
s = nueva("Primera salida: ¿llega a la canasta? Comparé seis modelos de scikit-learn, con hiperparámetros "
          "elegidos por validación cruzada de cinco particiones y evaluados una sola vez en la corrida de prueba. "
          "La métrica principal es el AUC-ROC: la probabilidad de que el modelo le dé mayor P a un intento exitoso "
          "que a uno fallido, o sea, qué tan bien ordena las manzanas. 0,5 es azar y 1 es perfecto. El mejor fue el "
          "bosque aleatorio, con 0,814 e intervalo de confianza del 95 % entre 0,78 y 0,85, calculado por bootstrap. "
          "La regresión logística fue la más débil, lo que indica que la relación no es lineal. Como regla práctica, "
          "si el robot solo intenta las manzanas con P mayor a 0,5, el éxito de sus intentos sube del 42 al 68 %.")
titulo(s, "¿Llega a la canasta?  Gana el bosque aleatorio")
vals = [float(tb.loc[m, "AUC-ROC"]) for m in orden]
barras(s, 0.6, 1.4, 7.6, 5.6, [m.replace("Red neuronal (MLP)", "Red neuronal") for m in orden], vals,
       [ROJO] + [VERDE] * 5 + [GRIS_BARRA])
texto(s, 0.75, 6.95, 7.4, 0.4, "AUC-ROC en la corrida de prueba (592 intentos) · 0,5 = azar, 1 = perfecto",
      size=12, color=GRIS, italic=True)
caja(s, 8.65, 1.5, 4.0, 2.55, VERDE_CLARO)
texto(s, 8.95, 1.65, 3.5, 2.3, [[(c(tb.loc["Bosque aleatorio", "AUC-ROC"]), {"size": 60, "bold": True,
                                                                         "color": ROJO, "font": TIT})],
                               "AUC del bosque aleatorio",
                               [("IC 95 % [0,78; 0,85]", {"color": GRIS, "size": 14})]], size=17)
caja(s, 8.65, 4.3, 4.0, 2.55, VERDE_OSC)
texto(s, 8.95, 4.45, 3.5, 2.3, [[("42 % → 68 %", {"size": 36, "bold": True, "font": TIT})],
                               "éxito de los intentos si solo se intenta con P ≥ 0,5",
                               [("(se intenta el 38 % de las manzanas)", {"size": 13,
                                                                         "color": RGBColor(0xCF, 0xE0, 0xCB)})]],
      size=16, color=BLANCO)

# ═════════════════════════ 7. red vs bosque ═════════════════════════
s = nueva("Como pedía la consigna, analicé el entrenamiento por épocas con la red neuronal: dos capas ocultas de 32 "
          "y 16 neuronas. A la izquierda la pérdida de entrenamiento baja todo el tiempo; a la derecha el AUC de "
          "prueba se estanca en torno a 0,79 mientras el de entrenamiento sigue subiendo: entrenar más solo "
          "sobreajusta, y la parada temprana la detuvo en la época 151. La comparación pareada por bootstrap da "
          "0,029 de AUC a favor del bosque, con intervalo que no incluye el cero, así que la diferencia es "
          "significativa. Coincide con la literatura: en datos tabulares de este tamaño los árboles suelen ganarle "
          "a las redes.")
titulo(s, "Red neuronal vs. bosque: más épocas no ayudan")
imagen(s, "red_epocas.png", 0.75, 1.45, w=11.8)
caja(s, 0.75, 5.65, 5.7, 1.3, VERDE_CLARO)
texto(s, 1.05, 5.65, 5.2, 1.3, [[("+0,029", {"bold": True, "size": 34, "color": ROJO, "font": TIT}),
                                 ("  AUC a favor del bosque", {"size": 17})],
                                [("IC 95 % [+0,011; +0,047] · significativa", {"size": 14, "color": GRIS})]],
      anchor=MSO_ANCHOR.MIDDLE)
caja(s, 6.85, 5.65, 5.7, 1.3, VERDE_CLARO)
texto(s, 7.15, 5.65, 5.2, 1.3, [[("151", {"bold": True, "size": 34, "color": VERDE_OSC, "font": TIT}),
                                 ("  época de parada temprana", {"size": 17})],
                                [("después, solo sobreajuste", {"size": 14, "color": GRIS})]],
      anchor=MSO_ANCHOR.MIDDLE)

# ═════════════════════════ 8. multiclase ═════════════════════════
s = nueva("Segunda salida: ¿en qué etapa termina el intento? Son seis clases desbalanceadas, por eso uso la F1 por "
          "clase y la F1 macro, que las promedia sin ponderar. El mejor otra vez fue el bosque, con F1 macro 0,46 "
          "frente a 0,10 del trivial. Pero el resultado es muy desigual: los fallos de alcance se predicen casi "
          "perfecto y los de agarre bastante bien, porque dependen de la geometría. Las caídas durante el "
          "transporte prácticamente no se pueden anticipar: dependen de qué tan bien quedó agarrada la fruta, algo "
          "que no se sabe antes de intentar.")
titulo(s, "¿En qué etapa falla?  Solo se anticipa lo geométrico")
cl = ["Alcance", "Agarre", "Éxito", "Desprendimiento", "Transporte", "Otro"]
f1 = [0.93, 0.67, 0.66, 0.41, 0.09, 0.0]
barras(s, 0.6, 1.4, 7.6, 5.6, cl, f1, [VERDE, VERDE, GRIS_BARRA, ROJO, ROJO, ROJO], formato="0.00")
texto(s, 0.75, 6.95, 7.4, 0.4, "F1 por clase del bosque aleatorio (prueba) · F1 macro 0,46 vs 0,10 del trivial",
      size=12, color=GRIS, italic=True)
caja(s, 8.65, 1.5, 4.0, 2.55, VERDE_CLARO)
texto(s, 8.95, 1.7, 3.5, 2.2, [[("Se predice", {"bold": True, "size": 22, "color": VERDE_OSC})],
                              "lo que depende de dónde está la manzana: si el brazo llega y si la mano la agarra"],
      size=16)
caja(s, 8.65, 4.3, 4.0, 2.55, ROJO_CLARO)
texto(s, 8.95, 4.5, 3.5, 2.2, [[("No se predice", {"bold": True, "size": 22, "color": ROJO})],
                              "lo que depende de qué tan firme quedó el agarre, que se sabe recién durante el intento"],
      size=16)

# ═════════════════════════ 9. robot real ═════════════════════════
s = nueva("¿Y en el robot real? De los registros de 50 sesiones reales extraje 190 intentos, con 33 % de éxito. "
          "Como ahí no se registraron las vecinas ni el píxel, reentrené los modelos solo con las siete variables "
          "geométricas. El resultado se invierte: la regresión logística, la más simple y la más débil en "
          "simulación, es la mejor en el robot real, con AUC 0,755, igual que en simulación. Los modelos flexibles "
          "caen: aprenden detalles propios del simulador que no se cumplen en la realidad. La regresión logística "
          "solo capta la tendencia general, las manzanas bajas y lejanas fallan, y eso sí se transfiere. Son pocos "
          "intentos, así que lo tomo como orientativo.")
titulo(s, "En el robot real gana el modelo más simple")
mods = ["Regresión logística", "Red neuronal (MLP)", "k-NN", "Gradient boosting", "Bosque aleatorio", "SVM (RBF)"]
cd = CategoryChartData()
cd.categories = [m.replace("Red neuronal (MLP)", "Red neuronal").replace("Gradient boosting", "Boosting")
                 .replace("Regresión logística", "Reg. logística").replace("Bosque aleatorio", "Bosque") for m in mods]
cd.add_series("Simulación", [float(tr.loc[m, "AUC simulación (prueba)"]) for m in mods])
cd.add_series("Robot real", [float(tr.loc[m, "AUC robot real"]) for m in mods])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.4), Inches(8.0), Inches(5.6), cd).chart
estilo_grafico(ch)
ch.has_title = False
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(14)
pl = ch.plots[0]
pl.gap_width, pl.overlap = 70, -10
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
pl.data_labels.font.color.rgb = TINTA
for serie, col in zip(pl.series, [GRIS_BARRA, ROJO]):
    serie.format.fill.solid()
    serie.format.fill.fore_color.rgb = col
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0.4, 0.9
va.has_major_gridlines = False
va.visible = False
ch.category_axis.tick_labels.font.size = Pt(13)
ch.category_axis.tick_labels.font.color.rgb = TINTA
texto(s, 0.75, 6.95, 7.8, 0.4, "AUC-ROC con 7 variables geométricas · 190 intentos reales, éxito 32,6 %",
      size=12, color=GRIS, italic=True)
caja(s, 9.0, 1.5, 3.65, 2.55, VERDE_CLARO)
texto(s, 9.3, 1.65, 3.1, 2.3, [[("0,755", {"size": 56, "bold": True, "color": ROJO, "font": TIT})],
                              "AUC real de la regresión logística", [("IC 95 % [0,68; 0,83]", {"size": 14,
                                                                                               "color": GRIS})]],
      size=17)
caja(s, 9.0, 4.3, 3.65, 2.55, VERDE_OSC)
texto(s, 9.3, 4.5, 3.1, 2.2, [[("Lección", {"bold": True, "size": 20})],
                             "los modelos flexibles aprenden detalles del simulador; para el robot real conviene uno simple"],
      size=15, color=BLANCO)

# ═════════════════════════ 10. cintura ═════════════════════════
s = nueva("Un predictor también sirve para decidir. Antes de cada intento, el robot puede girar la cintura. Probé "
          "tres estrategias en simulación, 600 intentos cada una: mantener el giro inicial, elegir el giro con el "
          "modelo, que evalúa los giros de menos 15 a más 15 grados y se queda con el de mayor P, y como control, "
          "girar al azar. Con el modelo, el éxito sube de 31,3 a 42,5 %: 11 puntos, con valor p menor a 0,001. El "
          "valor p es la probabilidad de ver una diferencia así por puro azar si en realidad no hubiera mejora. "
          "Girar al azar no mejora significativamente, así que la mejora viene de la elección del modelo, no de "
          "girar. Y la probabilidad que predijo el modelo coincidió con el éxito observado en los tres grupos.")
titulo(s, "Usar el modelo para decidir el giro de cintura")
grupos = [("mantener", "Mantener el giro", VERDE_CLARO, VERDE_OSC), ("modelo", "Giro elegido por el modelo", ROJO, BLANCO),
          ("aleatorio", "Giro al azar (control)", VERDE_CLARO, VERDE_OSC)]
for k, (g, nombre, fc, tc_) in enumerate(grupos):
    x = 0.75 + k * 4.05
    caja(s, x, 1.65, 3.75, 3.6, fc)
    texto(s, x + 0.35, 1.85, 3.1, 0.6, nombre, size=18, color=tc_, bold=True)
    texto(s, x + 0.35, 2.55, 3.1, 1.3, f"{100 * tc.loc[g, 'éxito']:.1f} %".replace(".", ","), size=60, color=tc_,
          font=TIT, bold=True)
    texto(s, x + 0.35, 3.95, 3.1, 1.2, [f"de éxito en 600 intentos",
                                        f"P predicha media: {c(tc.loc[g, 'P predicha (giro usado)'], 2)}"],
          size=14, color=tc_)
caja(s, 0.75, 5.6, 11.85, 1.2, VERDE_OSC)
texto(s, 1.1, 5.6, 11.2, 1.2, [[("+11,2 puntos", {"bold": True, "size": 28, "font": TIT}),
                                ("   IC 95 % [+5,7; +16,7] · p < 0,001       al azar: +2,8 puntos, p = 0,30 (no "
                                 "significativo)", {"size": 16})]], color=BLANCO, anchor=MSO_ANCHOR.MIDDLE)

# ═════════════════════════ 11. políticas v5 ═════════════════════════
s = nueva("El modelo también orientó el desarrollo. Mostró que la mayoría de los fallos ocurrían en el agarre y el "
          "desprendimiento, y que la posición de la manzana respecto al brazo era decisiva. Con eso se diseñó una "
          "nueva cadena de políticas, la v5: un mapa aprendido que elige el giro de cintura y la dirección de la "
          "mano para cada manzana, un alcance que no empuja la fruta, y un agarre y un tirón más firmes. En el "
          "mismo mundo simulado, el éxito pasa de 37 a 68 %. A la izquierda se ve que bajan justo los fallos de "
          "agarre y de desprendimiento. Generé 1800 intentos nuevos con estas políticas para reentrenar el "
          "predictor.")
titulo(s, "Lo aprendido guió nuevas políticas del robot (v5)")
caja(s, 0.75, 1.5, 3.6, 1.55, VERDE_CLARO)
texto(s, 1.0, 1.5, 3.2, 1.55, [[("37 %", {"size": 44, "bold": True, "color": GRIS, "font": TIT})], "éxito antes"],
      size=15, anchor=MSO_ANCHOR.MIDDLE)
flecha = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.6), Inches(1.95), Inches(0.9), Inches(0.65))
flecha.fill.solid()
flecha.fill.fore_color.rgb = VERDE
flecha.line.fill.background()
caja(s, 5.75, 1.5, 3.6, 1.55, ROJO)
texto(s, 6.0, 1.5, 3.2, 1.55, [[("68 %", {"size": 44, "bold": True, "font": TIT})], "éxito con las políticas v5"],
      size=15, color=BLANCO, anchor=MSO_ANCHOR.MIDDLE)
texto(s, 9.65, 1.5, 3.0, 1.6, [[("Fallos de agarre", {"bold": True})], "27 % → 7 %",
                               [("Fallos de desprendimiento", {"bold": True})], "19 % → 10 %"], size=14,
      color=VERDE_OSC)
imagen(s, "cadena_v5.png", 0.75, 3.35, w=11.85)
texto(s, 0.75, 6.75, 11.8, 0.4, "Izquierda: resultado de los intentos con cada versión · centro: curvas ROC · "
      "derecha: calibración del predictor reentrenado", size=12, color=GRIS, italic=True)

# ═════════════════════════ 12. predictor v5 ═════════════════════════
filas5 = [("Bosque reentrenado", "Bosque aleatorio", ROJO),
          ("Q del mapa de las políticas", "Q del mapa de apuntado (referencia)", VERDE),
          ("Bosque anterior, sin reentrenar", "Bosque de la cadena v1, sin reentrenar", GRIS_BARRA)]
s = nueva("¿Sirve el predictor viejo con las políticas nuevas? Reentrené los modelos con los datos v5 y los comparé "
          "con dos referencias: la Q que ya trae el mapa de las políticas, que estima si el brazo va a llegar, y "
          "el bosque anterior sin reentrenar. Los tres ordenan igual de bien, AUC alrededor de 0,86, porque ahora "
          "lo predecible es sobre todo el alcance. La diferencia está en la calibración, medida con la puntuación "
          "de Brier, el error cuadrático de la probabilidad: menor es mejor. El reentrenado es significativamente "
          "mejor: la Q sobreestima el éxito y el bosque viejo lo subestima, porque aprendió con políticas que "
          "fallaban más. Conclusión: cuando cambian las políticas, el orden se conserva, pero hay que reentrenar "
          "para que P siga siendo una probabilidad confiable.")
titulo(s, "Con políticas nuevas, hay que reentrenar")
barras(s, 0.6, 1.4, 7.6, 4.6, [a for a, _, _ in filas5], [float(t5.loc[b, "Brier"]) for _, b, _ in filas5],
       [col for _, _, col in filas5], vmax=0.26)
texto(s, 0.75, 5.95, 7.4, 0.8, "Puntuación de Brier en la prueba v5 (586 intentos): error de la probabilidad, "
      "menor es mejor", size=12, color=GRIS, italic=True)
caja(s, 8.65, 1.5, 4.0, 2.55, VERDE_CLARO)
texto(s, 8.95, 1.65, 3.5, 2.3, [[("≈ 0,86", {"size": 54, "bold": True, "color": VERDE_OSC, "font": TIT})],
                               "AUC de los tres: ordenan igual",
                               [(f"bosque {c(t5.loc['Bosque aleatorio', 'AUC-ROC'])} · Q "
                                 f"{c(t5.loc['Q del mapa de apuntado (referencia)', 'AUC-ROC'])} · anterior "
                                 f"{c(t5.loc['Bosque de la cadena v1, sin reentrenar', 'AUC-ROC'])}",
                                 {"size": 13, "color": GRIS})]], size=16)
caja(s, 8.65, 4.3, 4.0, 2.55, VERDE_OSC)
texto(s, 8.95, 4.5, 3.5, 2.2, [[("Pero solo el reentrenado", {"bold": True, "size": 19})],
                              "da probabilidades que se pueden leer como tasa de éxito real"],
      size=16, color=BLANCO)

# ═════════════════════════ 13. conclusiones ═════════════════════════
s = nueva("Para cerrar. Primero, sí se puede predecir el éxito antes de intentar: el bosque aleatorio llega a AUC "
          "0,81, y lo que importa es la posición de la manzana respecto al hombro. Segundo, la etapa del fallo solo "
          "se anticipa cuando depende de la geometría. Tercero, el modelo sirvió para decidir: elegir la cintura "
          "con él sumó 11 puntos de éxito, y su diagnóstico orientó políticas nuevas que llevaron el éxito simulado "
          "al 68 %. Y cuarto, dos lecciones: en el robot real conviene un modelo simple, y el predictor se debe "
          "reentrenar cada vez que cambian las políticas. Las limitaciones principales: los datos son simulados y "
          "la prueba real es chica. El paso siguiente es registrar en el robot real las mismas variables para "
          "reentrenar con datos reales. Gracias, quedo atento a sus preguntas.", VERDE_OSC)
texto(s, 0.9, 0.55, 11.5, 0.8, "Conclusiones", size=36, color=BLANCO, font=TIT, bold=True)
concl = [("Se puede predecir", "AUC 0,81 antes de intentar; manda la posición respecto al hombro"),
         ("Solo lo geométrico", "la etapa se anticipa en alcance y agarre, no en las caídas"),
         ("Sirve para decidir", "+11 puntos eligiendo la cintura; guió políticas que llegan al 68 %"),
         ("Simple y reentrenado", "en el robot real gana lo simple; con políticas nuevas, reentrenar")]
for k, (a, b) in enumerate(concl):
    col, fila = k % 2, k // 2
    x, y = 0.9 + col * 6.0, 1.7 + fila * 2.05
    numero(s, x, y + 0.1, k + 1, color=ROJO, d=0.62, size=22)
    texto(s, x + 0.85, y, 4.9, 1.8, [[(a, {"bold": True, "size": 22, "color": BLANCO})], b], size=16,
          color=RGBColor(0xCF, 0xE0, 0xCB))
caja(s, 0.9, 5.95, 11.55, 0.95, RGBColor(0x2C, 0x4F, 0x3C))
texto(s, 1.2, 5.95, 11.0, 0.95, [[("Limitaciones: ", {"bold": True, "color": BLANCO}),
                                  ("datos simulados, prueba real chica (190 intentos).  ", {}),
                                  ("Próximo paso: ", {"bold": True, "color": BLANCO}),
                                  ("registrar las mismas variables en el robot real y reentrenar.", {})]],
      size=15, color=RGBColor(0xCF, 0xE0, 0xCB), anchor=MSO_ANCHOR.MIDDLE)

prs.save(OUT)
print(OUT, len(prs.slides), "diapositivas")
