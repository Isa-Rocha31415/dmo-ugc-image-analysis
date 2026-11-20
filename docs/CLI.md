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
| `--top-n` | int | `50` | Top n-gramas en reportes. |
| `--global-outliers` | flag | per-fuente | Detección global en vez de por fuente. |
| `--model` | `bertopic` / `bidootopic` / `bidoo` | `bertopic` | Backend de modelado de tópicos. |
| `--palette` | `ejecutiva`/`naturaleza`/`accesible`/`tecnologica`/`hospitalidad` | `ejecutiva` | Paleta de colores de todos los HTML. |
| `--embedding-model` | str | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de sentence-transformers. |
| `--sentiment-model` | str | modelo por defecto de `sentiment_analysis` | Modelo HF de sentimiento. |
| `--cache-dir` | str | `<output-dir>/cache` | Carpeta de caché. |
| `--no-cache` | flag | — | Ignora la caché. |
| `--force-steps` | lista | `[]` | `load`, `embeddings`, `outliers`, `sentiment`, `topics`, `all`. |
| `--no-sentiment-validation` | flag | — | Omite el PASO 4.5 (validación de sentimiento). |
| `--gold-csv` | str | — | CSV con etiquetas humanas (texto + etiqueta). |
| `--sentiment-ablation` | flag | — | Ejecuta la ablación (léxico / TF-IDF / linear probe). |
| `--sentiment-finetune` | flag | — | Añade el transformer afinado a la ablación (costoso). |
| `--finetune-model` | str | `pysentimiento/robertuito-sentiment-analysis` | Modelo a afinar. |
| `--price-anchor-file` | str | `resources/anchors/precio_valor.json` | JSON de anchors (F4.2). Alias: `--price-anchors`. |
| `--price-tau-quantile` | float | `0.95` | Cuantil del placebo usado como τ. |
| `--price-placebo-docs` | int | `200` | Docs aleatorios en el placebo de τ. |
| `--price-mode` | `semantic`/`lexical`/`hybrid` | `semantic` | Regla del detector: `>=τ`, `lex>0`, o `>=τ`/`lex>=2`. |
| `--price-center` / `--no-price-center` | flag | centrado | Score semántico centrado (`--no-price-center` = raw). |
| `--price-sample-size` | int | `100` | Muestra estratificada de anotación (F4.3). Alias: `--price-validation-n`. |
| `--price-annotation-file` | str | `../annotations/precio_valor_labels.csv` | CSV de anotaciones F4.3. |
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
3. Validación F4.3: escribe `annotation_sample.csv`; si
   `annotations/precio_valor_labels.csv` está vacío imprime
   "validación pendiente de anotación" y NO falla (guía: `docs/ANNOTATION_GUIDE.md`).
   Con etiquetas añade `price_validation.csv`, `price_ablations.csv`,
   `confusion_matrix.csv`.
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
Además: Spearman y MAE del score contra el rating, y (opcional) un gold set humano.
