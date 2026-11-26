# Predicción del éxito de la cosecha robótica de manzanas

Proyecto final de *Ciencia de Datos e IA aplicados a la Mecatrónica* (UTEC).

El proyecto predice, antes de que el robot intente cosechar una manzana, dos cosas:

- si la manzana llegará a la canasta (clasificación binaria);
- en qué etapa fallaría el intento (clasificación multiclase).

Cada intento se describe con 18 variables tabulares, tomadas de la cámara y del estado del robot. Se comparan seis modelos de scikit-learn y un modelo trivial de referencia.

## Cómo reproducir los resultados

Solo hace falta Python con las bibliotecas científicas habituales. No se necesita la simulación.

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook analisis_cosecha.ipynb      # ejecutar desde esta carpeta: Kernel → Restart & Run All
```

- El notebook entrega ya ejecutado, con todas sus salidas.
- Al volver a ejecutarlo se regeneran las figuras y tablas en `figuras/`.
- Se usó Python 3.12, numpy 2.1, pandas 2.3, scikit-learn 1.8 y matplotlib 3.9.
- Con otras versiones de scikit-learn, los números pueden variar en el tercer decimal.

## Contenido

### Análisis y modelos (se ejecutan sin la simulación)

| Archivo | Descripción |
|---|---|
| `analisis_cosecha.ipynb` | **Archivo principal.** Carga de datos, análisis exploratorio, partición, búsqueda de hiperparámetros con validación cruzada, evaluación con intervalos *bootstrap*, modelo multiclase, validación con el robot real, experimento del giro de cintura y la sección 8 (cadena v5). |
| `analisis_cosecha.py` | El mismo notebook exportado como script. |
| `analisis_v5.py` | Reentrena y evalúa el predictor para las políticas v5. El notebook lo llama desde la sección 8. |
| `bosque_np.py` | Reproduce en numpy un `RandomForestClassifier` exportado, para usarlo dentro de la simulación sin scikit-learn. |
| `exportar_bosque.py` | Entrena el bosque aleatorio y lo guarda en `modelos/bosque_exito.npz`. |

### Recolección de datos (referencia)

Estos scripts **no se ejecutan de forma aislada**. Necesitan el repositorio completo del robot: Isaac Lab, el modelo del robot Unitree G1 y las políticas de refuerzo entrenadas. Se incluyen para documentar cómo se generaron los CSV.

| Archivo | Descripción |
|---|---|
| `recolectar_dataset.py` | Corre la cosecha simulada con la primera cadena de políticas. Por cada intento guarda las 18 variables de entrada, el resultado y la etapa de fallo. |
| `cosecha_v5.py` | Hace lo mismo con la cadena de políticas v5. |
| `cosecha_cintura.py` | Experimento de decisión. Elige el giro de cintura con el modelo, al azar o sin girar. |
| `intentos_reales.py` | Extrae los intentos del robot real de sus registros. |
| `datos_v5/correr_v5.sh`, `datos_cintura/correr_experimento.sh` | Comandos con los que se lanzaron las simulaciones. |

### Base de datos

Todas las tablas son CSV con una fila por intento de cosecha. Los archivos de simulación tienen las mismas columnas:

- las 18 entradas;
- `exito` (0/1);
- `etapa` (la etapa donde terminó el intento);
- metadatos del episodio.

`intentos_reales.csv` tiene menos columnas, porque los registros del robot real guardan menos información.

| Archivo | Filas | Uso |
|---|---|---|
| `datos/cosecha_seed0.csv`, `seed1`, `seed2` | 388 + 600 + 1616 | Dataset principal (primera cadena de políticas). La semilla 1 es el conjunto de prueba. |
| `datos_v5/cosecha_v5_seed20.csv`, `21`, `22` | 600 × 3 | Dataset de las políticas v5. La semilla 22 es el conjunto de prueba. |
| `datos_cintura/{aleatorio,mantener,modelo}_seed3.csv` | 600 × 3 | Experimento del giro de cintura: tres estrategias sobre los mismos árboles. |
| `datos_reales/intentos_reales.csv` | 190 | Intentos del robot real, para validar la transferencia simulación → realidad. |
| `modelos/bosque_exito.npz` | — | Bosque aleatorio entrenado, usado en el experimento del giro de cintura. |

En el notebook se descartan las filas cuya manzana objetivo no queda entera dentro del cuadro de la cámara.

### Informe y presentación (opcional)

`informe/generar_informe.py`, `informe/diagrama_modelos.py` e `informe/generar_presentacion.py` generan el informe (.docx/.pdf), el diagrama de arquitectura y las diapositivas a partir de las tablas y figuras del notebook. Requieren además `python-docx` y `python-pptx`.
