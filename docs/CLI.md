# CLI — `code/run_analysis.py`

Pipeline DMO (imagen proyectada) vs UGC (imagen percibida).

## Comando principal

```bash
cd code
python run_analysis.py \
    --csv-dmo "../data/processed/data_unified.csv" \
    --csv-ugc "../data/processed/data_unified.csv" \
    --text-col text \
    --lang es \
    --outlier-method IF \
    --contamination auto \
    --min-partition 50 \
    --output-dir ../results \
    --model bertopic \
    --palette ejecutiva
```

## Flags

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--csv-dmo` | str (req.) | — | CSV del corpus DMO (imagen proyectada). |
| `--csv-ugc` | str (req., repetible) | — | CSV(s) del corpus UGC (imagen percibida). |
| `--text-col` | str (req.) | — | Columna de texto. Si no existe, se resuelve por alias (`texto`, `comment`…). |
| `--lang` | str | `es` | Idioma. |
| `--outlier-method` | `IF`/`LOF`/`zscore` | `IF` | Detector de outliers. |
| `--contamination` | float o `auto` | `auto` | Proporción de outliers. |
| `--min-partition` | int | `50` | Mínimo de comentarios para modelar tópicos (si no, frecuencia de palabras). |
| `--output-dir` | str | `results` | Salida (artefactos + caché). |
| `--title` | str | `Imagen proyectada (DMO) vs imagen percibida (UGC)` | Título del informe HTML. |
| `--top-n` | int | `50` | Top n-gramas en reportes. |
| `--global-outliers` | flag | per-fuente | Detección global en vez de por fuente. |
| `--model` | `bertopic` / `bidootopic` / `bidoo` | `bertopic` | Backend de modelado de tópicos. |
| `--palette` | `ejecutiva`/`naturaleza`/`accesible`/`tecnologica`/`hospitalidad` | `ejecutiva` | Paleta de colores de todos los HTML. |
| `--embedding-model` | str | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de sentence-transformers. |
| `--sentiment-model` | str | modelo por defecto de `sentiment_analysis` | Modelo HF de sentimiento. |
| `--sentiment-neutral-tau` | float | `0.15` | Banda `|sentiment_score| < τ` para la clase NEUTRO. Reetiqueta la caché existente; no recalcula el modelo. |
| `--verbose-jaccard` | flag | OFF | Imprime Jaccard@100 auxiliar en consola al generar el informe HTML. Jaccard nunca aparece en el dashboard principal. |
| `--ngram-interactive` / `--no-ngram-interactive` | flag | ON | Uni/bi/trigramas interactivos: cada barra muestra hasta 3 comentarios de `text_norm` en el hover. Desactivar con `--no-ngram-interactive`. |
| `--umap-3d` / `--no-umap-3d` | flag | ON | Scatter 3D de comentarios por tópico sobre los embeddings cacheados. Requiere `umap-learn`; con `--no-umap-3d` se omite la sección. |
| `--generate-wordclouds` / `--no-wordclouds` | flag | ON | Genera las imágenes PNG de wordclouds. `--no-wordclouds` las omite. El flag positivo se conserva por compatibilidad. |
| `--stability-runs` | int | `3` | Corridas de estabilidad (antes 5; pasar 5 restaura el valor previo). |
| `--subsample` | float | `0.75` | Fracción por corrida (antes 0.8; pasar 0.8 restaura el valor previo). |
| `--force-parallel` | flag | — | Activa los pools del PASO 7 y del dispatcher (default serial: crossover 5k dio 0.58×). |
| `--sent-max-length` | int | `512` | Máximo de tokens del pipeline de sentimiento. No entra en la clave de caché: con caché existente, cambiar este valor no invalida el sentimiento guardado; usa `--no-cache` o `--force-steps sentiment all`. |
| `--cache-dir` | str | `<output-dir>/cache` | Carpeta de caché. |
| `--no-cache` | flag | — | Ignora la caché. |
| `--force-steps` | lista | `[]` | `load`, `embeddings`, `outliers`, `sentiment`, `topics`, `all`. |
| `--no-sentiment-validation` | flag | — | Omite el PASO 4.5 (validación de sentimiento). |
| `--gold-csv` | str | — | Opcional, OFF: CSV con etiquetas humanas (no se usa por defecto). |
| `--sentiment-ablation` | flag | OFF | Opcional: ablación léxico / TF-IDF / linear probe. |
| `--sentiment-finetune` | flag | OFF | Opcional: transformer afinado (costoso). |
| `--finetune-model` | str | `pysentimiento/robertuito-sentiment-analysis` | Modelo a afinar. |
| `--price-anchor-file` | str | `resources/anchors/precio_valor.json` | JSON de anchors (F4.2). Alias: `--price-anchors`. |
| `--price-tau-quantile` | float | `0.95` | Cuantil del placebo usado como τ. |
| `--price-placebo-docs` | int | `500` | Alias legacy de `--price-sample-size`. |
| `--price-mode` | `semantic`/`lexical`/`hybrid` | `semantic` | Detector primario SIEMPRE `semantic`; los otros dos solo son ablaciones. |
| `--price-center` / `--no-price-center` | flag | centrado | Score semántico centrado (`--no-price-center` = raw). |
| `--price-sample-size` | int | `500` | Docs aleatorios del corpus para calibrar τ por placebo (alias: `--price-validation-n`). |
| `--price-annotation-file` | str | — | Legacy aceptado pero IGNORADO: avisa `DeprecationWarning`; no se leen anotaciones. |
| `--price-top-k` | int | `5` | Top-k ejemplos por corpus/fuente (F4.4). |

## Tarea `price` (PASO 12, F4.2–F4.4)

`--comparison-tasks price` (tras `thematic`; PASO 12) ejecuta
`price_concept.run_price_task` → `run_price_core`:

1. `price_scores.csv`, `tau_calibration.csv`, `tau_sensitivity.csv` (F4.2).
2. Panel F4.4 en `<output-dir>/comparison/price/`: `mention_rate_by_source.csv`,
   `score_distribution_tests.csv`, `valence_conditioned.csv`,
   `price_context_logodds.csv`, `price_top_examples.csv`,
   `price_scatter_prototypes.html`/`_data.csv`, `price_scatter_valence.html`/`_data.csv`,
   `price_summary.html`, `price_density.csv`.
3. Validación F4.3 automática (sin intervención humana): el detector primario es
   `semantic` (coseno contra el centroide L2 del ancla sintética; `lexical` e
   `hybrid` solo son ablaciones). Escribe `price_validation_auto.csv`,
   `price_threshold_sensitivity.csv`, `price_anchor_jackknife.csv` y
   `price_validation_status.json` (`status="validada_automaticamente"`).
   Comprobaciones: κ `semantic_vs_lexical` (referencia débil, NO gold standard),
   FPR placebo, sensibilidad τ∈{p90,p95,p97.5} con `gap_sign_stable`, jackknife
   del ancla (200 réplicas, 20% de términos, seed 42) e IC bootstrap 95%.
   Solo docs normales: outliers excluidos. Todos los CSV llevan
   `anchors_version`, `anchors_sha256`, `seed`, `tau`, `detector_primario`,
   `n_normales` y `n_por_fuente`. Ver `docs/ANCHORS.md` (limitación: sin anotación
   humana).
4. Comparación F4.4b en el mismo subdir: `price_comparison.csv` (tasas DMO vs
   UGC_total y por fuente con CI bootstrap `--bootstrap`/`--seed`, χ² o Fisher + BH,
   Mann-Whitney + KS + Cliff's delta, banda neutra `|sentiment|<0.15`),
   `price_context_terms.csv` (top-20 co-ocurrencias con `tokenize` + log-odds
   Monroe, `--prior-strength`), `price_kpis.csv` (kpi,value,unit,scope; sin
   anotaciones), `price_examples_topk.csv` (top_k por fuente, texto ≤280 chars) y
   `price_scatter_valence.html`/`_data.csv` (x=proximidad, y=valencia,
   color=corpus, símbolo=fuente, línea τ). Metadata `price_comparison`.
   Caché propia idempotente: `<cache-dir>/price/manifest.json` guarda un
   fingerprint de (parámetros, scores); si no cambia y los outputs existen la
   segunda corrida no recalcula (`--no-cache` fuerza el recálculo).
5. PASO 12.8 (informe del concepto): `price_density.csv` (histograma normalizado
   por fuente), `price_distribution_tests.csv` (EMD/Wasserstein + JSD,
   histogramas normalizados DMO vs UGC), `price_alignment.csv` (media/mediana de
   coseno, coseno de centroides entre menciones, gaps),
   `price_valence_conditioned.csv`, `price_context_logodds.csv` (top-15),
   `price_topic_crosstab.csv`, `price_examples.csv` (top-5 global/DMO/UGC/plataforma
   con fuente visible), `price_comparison.csv` enriquecido con `test_used` y
   `test_reason`, `price_scatter_similarity.html` (todos los normales; x=coseno,
   y=sentiment_score; color+símbolo por fuente, 3 modos de color, línea τ),
   `price_umap_concept.html` (color continuo de coseno, concepto = baricentro
   k-NN etiquetado como aproximado), sus `_data.csv`, `price_findings.md` y
   `<output-dir>/price_section.html`. La tarea `report` inyecta esa sección en
   `{{PRICE_SECTION}}`. Si hay <10 menciones DMO se imprime WARNING y aparece
   el aviso de tasas inestables en el HTML.

Corrida real sobre el corpus completo (CPU, reutilizando caché):

```bash
cd ~/ac_4/code
CUDA_VISIBLE_DEVICES='' python run_analysis.py \
  --csv-dmo ../data/processed/df_dmo.csv \
  --csv-ugc ../data/processed/df_ugc.csv \
  --text-col text --output-dir ../results \
  --comparison-tasks base topics price report --seed 42 --price-sample-size 500
```

## Caché

Cada paso escribe en `<cache-dir>/<paso>/<clave>/`:

```
manifest.json          inputs, outputs, versión, timestamp
data.csv.gz            dataframe del paso
embeddings.npy         embeddings (BERT)
analysis.json          n-gramas y stats de outliers
topics_df.csv.gz       tópicos por sentimiento
```

La clave es un hash de los inputs (hash del CSV, flags, versión del paso),
así que cambiar un parámetro invalida solo ese paso y sus dependientes.
`--force-steps` recalcula los pasos indicados. El PASO 1 crea `text_norm` (stopwords/lemas
solo ahí; `text_col` queda natural) y mete huella módulo+`--lang`+params en la clave de
`embeddings_bert`: tras cambiar el preproc se recalculan embeddings y dependientes.
`metrics_lexical` y `price_concept` (términos de contexto) leen `text_norm`; wordclouds,
n-gramas, tópicos, hover y ejemplos conservan `text_col` natural.

## Validación de sentimiento (PASO 4.5)

Se escribe en `<output-dir>/sentiment_validation/`:

| Artefacto | Contenido |
|---|---|
| `sentiment_validation.json` | Métricas completas (JSON). |
| `sentiment_validation.html` | Reporte con tablas y matriz de confusión. |
| `confusion_matrix_weak_labels.csv` | Matriz de confusión (etiqueta débil). |
| `metrics_by_source.csv` | Accuracy / macro-F1 / MCC por fuente (domain shift). |
| `ablation.csv` | Tabla de ablación (si `--sentiment-ablation`). |

Etiquetas: rating externo 1–2 = negative, 3 = neutral (excluido), 4–5 = positive.
Además: Spearman y MAE del score contra el rating (automático); el gold set humano,
la ablación y el finetuning son opcionales y están OFF por defecto. Las predicciones
`neutral` se excluyen de esta validación binaria y se cuentan en
`predicciones_neutras_excluidas`.

## Polaridad POS / NEU / NEG

El etiquetado usa la banda `|sentiment_score| < --sentiment-neutral-tau` (τ=0.15)
sobre el score ya cacheado: `>τ` → positive, `<-τ` → negative, resto → neutral.
No se descarga ni se recalcula ningún modelo. Columnas producidas:
`sentiment_label` (`positive`/`neutral`/`negative`) y `sentiment_label_id` (1/0/-1).

La polaridad se reporta **solo como métrica descriptiva**: tabla POS/NEU/NEG por
corpus y fuente en `comparison/report/sentiment_descriptive.csv` y en la sección
"Métricas descriptivas de polaridad" del HTML. No se usa como prueba de la brecha
DMO vs UGC; las pruebas temáticas, de precio y los tópicos por sentimiento siguen
corriendo con sus artefactos habituales (ahora con la partición neutral añadida).

## Jaccard (auxiliar)

`metrics_lexical.csv` conserva `jaccard_freq`/`jaccard_tfidf` y `metrics_lexical.py`
sigue imprimiéndolos en consola. El generador del informe HTML no renderiza Jaccard
en el bloque principal; con `--verbose-jaccard` solo los imprime también en consola.

## Tareas y SLURM

`--comparison-tasks`: `base topics lexical thematic price report` (implementadas; `report`
genera `informe_final.html`) y `semantic affective figures` (PENDIENTE, sin fallar).
Pasos 15/17 = huecos; no existen `ias`/`qualitative`.

## N-gramas interactivos

`run_analysis.py` genera `results/ngram_interactive.html` antes de la tarea `report` y
lo inyecta en `informe_final.html` (marcador `{{NGRAM_SECTION}}`). Cada orden (uni/bi/tri)
es una única figura Plotly con dos botones ("Comentarios Normales" / "Comentarios Outliers")
que alternan las dos trazas. El hover de cada barra muestra frecuencia y hasta tres
comentarios de `text_norm` (120 chars, escapados para HTML), obtenidos de
`analysis_base.csv` y `outliers_base.csv`. La tokenización es `metrics_lexical.tokenize`.
Con `--no-ngram-interactive` se omite el bloque y el HTML muestra un placeholder.

## Tópicos y UMAP 3D en el informe

La tarea `topics` escribe `comparison/report/topics_summary.csv` con columnas
`topic_id,title,top5_words,all_keywords,coverage_pct,n_docs`. La cobertura es
`n_docs_asignados / total_comentarios_en_análisis * 100`; el título es el nombre
auto-generado del tópico (o la primera palabra si no hay nombre). El mismo bloque
con formato de caja se imprime en consola y se inyecta en `informe_final.html`.

El scatter `Distribución semántica de comentarios (UMAP 3D)` usa
`comparison/base/embeddings_clean.npy` (ya cacheados), `n_components=3` y
`random_state=--seed`. Un trace por tópico, leyenda visible y hover con tópico,
fuente, sentimiento y texto truncado a 200 caracteres. La tabla de puntos se
exporta a `comparison/report/umap_3d_points.csv`. Si falta `umap-learn`, el error
indica usar `--no-umap-3d`.

## Secciones del dashboard

`informe_final.html` incluye ocho secciones con IDs estables: `ejecutivo`,
`polaridad-descriptiva`, `lexico`, `ngramas`, `topicos`, `umap-3d`,
`pruebas-estadisticas` y `metadatos-run`. Las tareas sin artefactos (`semantic`,
`affective`, `figures`) dejan placeholders con ID propio.
Wrapper activo: `../slurm_script_CUDA.sh` (raíz; `_copy`/`_test` en `legacy/slurm/`).
Envío desde la raíz: `sbatch slurm_script_CUDA.sh`; hace `cd code/`, usa GPU y fija las cinco tareas en A/B.
