# Análisis Comparativo: Imagen Proyectada (DMO) vs Imagen Percibida (UGC)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

## Objetivo

Este pipeline realiza un análisis de texto completo y genera un dashboard interactivo en formato de página web. Compara la **imagen de destino proyectada** por una Organización de Marketing de Destino (DMO) con la **imagen percibida** por los usuarios en fuentes digitales.

La comparación se apoya en cuatro preguntas concretas:

1. Qué temas comunica la DMO y qué temas aparecen de verdad en la conversación de los usuarios.
2. Dónde están las brechas: tópicos que sobran en la comunicación oficial y tópicos que faltan.
3. Cómo cambia la polaridad (positiva, neutra, negativa) según la fuente.
4. Cómo se habla de precio, valor y costo en cada corpus, con cuantificación del efecto y ejemplos trazables.

Todo el análisis es reproducible: semilla explícita, parámetros registrados y caché reutilizable para no recalcular embeddings ni sentimiento.

## Qué produce

- Un informe HTML principal con el dashboard completo.
- Secciones HTML independientes que se inyectan en el informe final.
- Tablas CSV con datos, métricas y resultados estadísticos (UTF-8 con BOM, sin índice).
- La tabla de datos de cada figura HTML exportada como CSV, para poder rehacer los gráficos.
- Un registro de tiempos y metadatos de ejecución en `results/comparison/base/run_metadata.csv`.

## Estructura del repositorio

```
ac_4_local/
├── code/                    # pipeline y módulos de métricas
│   ├── run_analysis.py      # punto de entrada del CLI
│   ├── utils.py             # paletas y utilidades de color
│   ├── metrics/             # funciones puras de comparación
│   ├── templates/           # plantillas HTML
│   └── resources/           # stopwords y anclas léxicas
├── data/                    # corpus de entrada (DMO y UGC)
├── docs/                    # CLI.md, ANCHORS.md, AUDIT.md, DEV_STATUS.md
├── results/                 # informe, secciones, tablas y caché
├── tests/                   # pruebas con fixtures sintéticos
├── environment.yml          # entorno conda con versiones fijadas
└── requirements.txt         # especificación explícita de paquetes
```

## Requisitos previos

- Conda (Miniconda o Anaconda), necesario para crear el entorno.
- Python 3.10.
- Entorno WORKING OFFLINE: los modelos de embeddings y sentimiento ya están descargados en la caché del entorno. El pipeline no ejecuta `nltk.download`, no instala paquetes nuevos y no descarga modelos de Hugging Face.

No hace falta GPU. La corrida de referencia se ejecuta en CPU reutilizando la caché.

## Instalación

Todos los comandos de instalación se ejecutan desde la raíz del repositorio (`~/ac_4_local`).

### Con `requirements.txt`

El archivo `requirements.txt` es una especificación explícita de Conda, con versiones y builds pinadas.

```bash
cd ~/ac_4_local
conda create --name nlp_report --file requirements.txt
conda activate nlp_report
python --version
```

Si el entorno ya existe y solo quieres sincronizarlo:

```bash
conda install --name nlp_report --file requirements.txt
```

### Con `environment.yml`

```bash
cd ~/ac_4_local
conda env create -f environment.yml
conda activate nlp_report
python --version
```

Para actualizar un entorno ya creado:

```bash
conda env update -f environment.yml --prune
```

La versión de referencia es Python 3.10.22. El archivo `environment.yml` define el entorno `nlp_report`.

## Datos de entrada

El pipeline necesita dos CSV:

- `data/df_dmo.csv` (o `data/processed/df_dmo.csv`): textos publicados por la DMO, es decir, imagen proyectada.
- `data/df_ugc.csv` (o `data/processed/df_ugc.csv`): textos de usuarios, es decir, imagen percibida. Se aceptan varios CSV de UGC repitiendo `--csv-ugc`.

La columna de texto por defecto es `text`. Si no existe, el lector intenta resolver alias habituales (`texto`, `comment` y similares). El idioma de trabajo es español (`--lang es`).

No se leen los CSV crudos de `data/raw`; se trabaja con los archivos procesados.

## Uso

El comando principal se ejecuta desde `code/`:

```bash
cd ~/ac_4_local/code

python run_analysis.py \
    --csv-dmo "../data/df_dmo.csv" \
    --csv-ugc "../data/df_ugc.csv" \
    --text-col text --lang es \
    --outlier-method IF --contamination auto \
    --min-partition 50 \
    --output-dir ../results \
    --model bertopic --palette ejecutiva --num-topicos automatico
```

La corrida imprime en consola un resumen por paso con separadores y títulos en español. Al terminar, abre el HTML principal de `results/` en un navegador. No necesita servidor: es un archivo autocontenido con Plotly embebido.

Si quieres usar los corpus procesados, cambia las dos primeras rutas por:

```bash
    --csv-dmo "../data/processed/df_dmo.csv" \
    --csv-ugc "../data/processed/df_ugc.csv" \
```

### Argumentos del comando principal

| Argumento | Valor del ejemplo | Qué hace |
|---|---|---|
| `--csv-dmo` | `../data/df_dmo.csv` | CSV del corpus DMO, obligatorio. Una sola fuente. |
| `--csv-ugc` | `../data/df_ugc.csv` | CSV del corpus UGC, obligatorio. Se puede repetir para añadir más fuentes. |
| `--text-col` | `text` | Nombre de la columna con el texto a analizar. |
| `--lang` | `es` | Idioma del corpus y de las reglas lingüísticas. |
| `--outlier-method` | `IF` | Detector de outliers: `IF` (Isolation Forest), `LOF` o `zscore`. |
| `--contamination` | `auto` | Proporción esperada de outliers. Acepta `auto` o un float entre 0 y 0.5. |
| `--min-partition` | `50` | Mínimo de comentarios por partición para modelar tópicos. Con menos filas se usa frecuencia de palabras. |
| `--output-dir` | `../results` | Carpeta de salida de HTML, CSV y caché. |
| `--model` | `bertopic` | Backend de tópicos: `bertopic`, `xidootopic` o `xidoo`. |
| `--palette` | `ejecutiva` | Paleta de colores de todos los gráficos HTML. |
| `--num-topicos` | `automatico` | Número de tópicos por partición de sentimiento: entero mayor o igual que 1 o `automatico`. Alias: `--num-topics`. |

### Opciones útiles

| Opción | Default | Uso |
|---|---|---|
| `--title` | `Imagen proyectada (DMO) vs imagen percibida (UGC)` | Título del informe HTML. |
| `--top-n` | `50` | Número de n-gramas mostrados en los reportes. |
| `--global-outliers` | detección por fuente | Cambia la detección de outliers a global. |
| `--embedding-model` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de sentence-transformers para embeddings. |
| `--sentiment-model` | modelo por defecto del módulo | Modelo HF de sentimiento. |
| `--sentiment-neutral-tau` | `0.15` | Banda de score considerada neutra. Reetiqueta la caché sin recalcular el modelo. |
| `--sent-max-length` | `512` | Máximo de tokens para sentimiento. Cambiarlo no invalida la caché existente. |
| `--cache-dir` | `<output-dir>/cache` | Carpeta de caché. |
| `--no-cache` | desactivado | Ignora la caché. Recalcula todo; útil solo cuando cambian los insumos. |
| `--force-steps` | `[]` | Recalcula pasos concretos: `load`, `embeddings`, `outliers`, `sentiment`, `topics` o `all`. |
| `--allow-recompute` | desactivado | Permite recalcular si la caché no encuentra un paso. |
| `--skip-comparison` | desactivado | Ejecuta solo el bloque base (pasos 0 a 7). |
| `--comparison-tasks` | todas | Selecciona tareas: `base`, `topics`, `lexical`, `thematic`, `price`, `semantic`, `affective`, `figures`, `report`. |
| `--seed` | `42` | Semilla de todos los procesos aleatorios. |
| `--bootstrap` | `2000` | Réplicas bootstrap para intervalos de confianza. |
| `--permutations` | `1000` | Permutaciones en pruebas estadísticas. |
| `--ngram-interactive` / `--no-ngram-interactive` | activado | N-gramas con comentarios de ejemplo en el hover. |
| `--umap-3d` / `--no-umap-3d` | activado | Scatter 3D de comentarios por tópico. Desactivar si no está `umap-learn`. |
| `--generate-wordclouds` / `--no-wordclouds` | activado | Genera las nubes de palabras en PNG. |
| `--verbose-jaccard` | desactivado | Imprime Jaccard auxiliar en consola. Nunca aparece en el HTML principal. |

La documentación completa del CLI, incluidos los aliases legacy, está en `docs/CLI.md`.

### Comandos de ejemplo

Corrida completa con varias fuentes UGC:

```bash
python run_analysis.py \
    --csv-dmo "../data/processed/df_dmo.csv" \
    --csv-ugc "../data/processed/df_ugc.csv" \
    --csv-ugc "../data/data_unified.csv" \
    --text-col text --lang es \
    --output-dir ../results \
    --palette naturaleza --num-topicos 8
```

Solo el bloque base, útil para validar la carga y la caché antes de seguir:

```bash
python run_analysis.py \
    --csv-dmo "../data/df_dmo.csv" \
    --csv-ugc "../data/df_ugc.csv" \
    --text-col text --lang es \
    --output-dir ../results \
    --skip-comparison
```

Solo tareas seleccionadas, reutilizando lo que ya esté cacheado:

```bash
python run_analysis.py \
    --csv-dmo "../data/df_dmo.csv" \
    --csv-ugc "../data/df_ugc.csv" \
    --text-col text --lang es \
    --output-dir ../results \
    --comparison-tasks lexical thematic price report
```

Corrida sin gráficos 3D ni nubes de palabras, por ejemplo en una máquina sin `umap-learn`:

```bash
python run_analysis.py \
    --csv-dmo "../data/df_dmo.csv" \
    --csv-ugc "../data/df_ugc.csv" \
    --text-col text --lang es \
    --output-dir ../results \
    --no-umap-3d --no-wordclouds
```

Recalcular solo embeddings y sentimiento:

```bash
python run_analysis.py \
    --csv-dmo "../data/df_dmo.csv" \
    --csv-ugc "../data/df_ugc.csv" \
    --text-col text --lang es \
    --output-dir ../results \
    --force-steps embeddings sentiment
```

## Salidas

Estructura orientativa de `results/`:

```
results/
├── informe_imagen_proyectada_dmo_vs_imagen_percibida_ugc_<fecha>.html
├── ngram_interactive.html
├── umap_section.html
├── topic_report_section.html
├── price_section.html
├── run_timings.csv
├── cache/
├── sentiment/          # conteos, histogramas y series por fuente
├── topics/             # selección de tópicos distribución y proyección 3D
├── outliers/           # outliers detectados y datos limpios
├── sentiment_validation/
└── comparison/
    ├── base/           # métricas base y run_metadata.csv
    ├── topics_joint/
    ├── lexical/
    ├── price/          # frecuencias, efectos, distribuciones, ejemplos y scatters
    └── report/         # informe final y metadatos de reporte
```

Los CSV se guardan con `encoding="utf-8-sig"` y `index=False`. Cada HTML lleva junto a sí un `_data.csv` con la tabla que alimenta la figura.

## Paletas de color

El sistema define cinco paletas. Cada una tiene tres variantes según el tipo de visión: `normal`, `protanopia` y `deuteranopia`. Las definiciones están en `code/utils.py`, en `COLOR_PALETTES`, y se convierten a colores de Plotly mediante `palette_to_plot_colors("ejecutiva")`.

Los roles de color son los mismos en todas las paletas:

| Rol | Uso en el informe |
|---|---|
| `texto` | Texto general. |
| `titulo` | Títulos de sección. |
| `subtitulo` | Encabezados secundarios. |
| `parrafo` | Cuerpo de texto. |
| `primario` | Serie principal y acento institucional. |
| `secundario` | Segunda serie y acentos gráficos. |
| `terciario` | Rellenos suaves y fondos secundarios. |
| `complementarios` | Tercera serie en adelante; el más cercano al verde y al rojo se usa para positivo y negativo. |
| `fondo` | Fondo del gráfico. |
| `borde` | Líneas de rejilla y bordes. |

Valores de la variante `normal`:

| Paleta | Texto | Título | Primario | Secundario | Terciario | Complementarios |
|---|---|---|---|---|---|---|
| `ejecutiva` | `#1A202C` | `#2D3748` | `#2B6CB0` | `#319795` | `#E2E8F0` | `#DD6B20`, `#38A169`, `#E53E3E` |
| `naturaleza` | `#264653` | `#2A9D8F` | `#2A9D8F` | `#E9C46A` | `#F4A261` | `#E76F51`, `#264653`, `#2B2D42` |
| `accesible` | `#00224E` | `#00224E` | `#184C7A` | `#7A94B0` | `#C4C4B0` | `#FFF3B3`, `#00224E`, `#1B3A5C` |
| `tecnologica` | `#1A202C` | `#553C9A` | `#6B46C1` | `#D53F8C` | `#ED8936` | `#38B2AC`, `#E53E3E`, `#ECC94B` |
| `hospitalidad` | `#2D3748` | `#C05621` | `#DD6B20` | `#2B6CB0` | `#ECC94B` | `#38A169`, `#2D3748`, `#744210` |

Ajustes de las variantes para daltonismo:

| Paleta | Protanopia | Deuteranopia |
|---|---|---|
| `ejecutiva` | Complementarios: `#D4A574`, `#38A169`, `#C53030` | Complementarios: `#DD6B20`, `#2F855A`, `#C53030` |
| `naturaleza` | Subtítulo `#9A7B1F`, secundario `#D4A853`, terciario `#E09959`, complementarios `#D65A31`, `#264653`, `#2B2D42` | Igual que `normal` |
| `accesible` | Igual que `normal` | Igual que `normal` |
| `tecnologica` | Secundario `#B83280`, complementarios `#38B2AC`, `#C53030`, `#ECC94B` | Igual que `normal` |
| `hospitalidad` | Igual que `normal` | Complementarios: `#2F855A`, `#2D3748`, `#744210` |

Todas mantienen `fondo` en `#FFFFFF`. Los bordes son `#CBD5E0` en `ejecutiva`, `#D3D3D3` en `naturaleza`, `#B0B0B0` en `accesible`, `#D6BCFA` en `tecnologica` y `#F6AD55` en `hospitalidad`.

## Secciones del informe HTML

El dashboard se organiza en estos bloques:

1. **Indicadores clave.** Volumen de comentarios por corpus y fuente, cobertura de tópicos y resumen de las brechas detectadas.
2. **Métricas descriptivas de polaridad.** Distribución de POS, NEU y NEG por fuente, con tablas de conteos y gráficos apilados.
3. **Análisis de N-gramas.** Uni, bi y trigramas interactivos, con comentarios de ejemplo al pasar el cursor.
4. **Tópicos retenidos.** Comparación DMO vs UGC de los tópicos que sobreviven a la selección.
   - Distribución semántica de comentarios (UMAP 3D).
   - Mapa temático DMO vs UGC.
   - Brechas DMO vs UGC.
   - Léxico distintivo de DMO y UGC.
5. **Precio, valor y costo.**
   - Frecuencia y tamaños del efecto.
   - Distribuciones por fuente y pruebas de diferencia.
   - Similitud semántica al concepto de precio.
   - Proximidad semántica (UMAP 3D).
   - Contexto temático por fuente.
   - Ejemplos de comentarios con menciones de precio, valor y costo.

La detección del concepto de precio es semántica por defecto. Las reglas léxicas e híbridas quedan como ablaciones. La calibración usa un placebo y un ancla documentada en `docs/ANCHORS.md`; no hay anotación humana, y esa limitación se declara en el informe.

## Capturas del informe

Espacio reservado para las imágenes del dashboard. Sustituir cada ruta por la captura real cuando se versionen las imágenes.

| Sección | Captura |
|---|---|
| Indicadores clave | `docs/img/informe_indicadores.png` |
| Polaridad POS / NEU / NEG | `docs/img/informe_polaridad.png` |
| N-gramas interactivos | `docs/img/informe_ngramas.png` |
| Mapa temático y brechas | `docs/img/informe_topicos.png` |
| Distribución semántica UMAP 3D | `docs/img/informe_umap.png` |
| Léxico distintivo | `docs/img/informe_lexico.png` |
| Precio, valor y costo | `docs/img/informe_precio.png` |

```html
<!--
<img src="docs/img/informe_indicadores.png" alt="Indicadores clave del informe">
<img src="docs/img/informe_polaridad.png" alt="Distribución de polaridad">
<img src="docs/img/informe_ngramas.png" alt="N-gramas interactivos">
<img src="docs/img/informe_topicos.png" alt="Mapa temático y brechas">
<img src="docs/img/informe_umap.png" alt="Distribución semántica UMAP 3D">
<img src="docs/img/informe_lexico.png" alt="Léxico distintivo">
<img src="docs/img/informe_precio.png" alt="Sección de precio, valor y costo">
-->
```

## Verificación

Desde la raíz del repositorio:

```bash
conda activate nlp_report
pytest -q
```

Las pruebas usan fixtures sintéticos y no ejecutan el pipeline completo. No lanzar GPU ni scripts SLURM para verificar cambios locales.

Para una comprobación rápida del CLI, usar los corpus de `data/` y una carpeta de salida independiente de `results/`.

## Solución de problemas

| Síntoma | Causa probable | Acción |
|---|---|---|
| `CondaNotFoundError` al activar el entorno | Conda no está en el `PATH` | Instalar Miniconda y reiniciar la terminal. |
| Falta el módulo `umap` | `umap-learn` no disponible | Usar `--no-umap-3d`. No instalar paquetes nuevos en el entorno offline. |
| El informe HTML no muestra una sección | La tarea correspondiente no se ejecutó | Añadirla a `--comparison-tasks` o quitar `--skip-comparison`. |
| Los resultados no cambian al modificar un parámetro | Caché reutilizada | Usar `--force-steps` con el paso afectado, o `--no-cache` de forma puntual. |
| Cambia `--sent-max-length` y el sentimiento queda igual | Ese parámetro no entra en la clave de caché | Usar `--force-steps sentiment` o `--no-cache`. |
| Pocas filas en DMO | Corpus pequeño | El pipeline usa Fisher y bootstrap cuando aplica; revisar los avisos de inestabilidad en el HTML. |

## Contacto

isabelrocha31415@gmail.com
