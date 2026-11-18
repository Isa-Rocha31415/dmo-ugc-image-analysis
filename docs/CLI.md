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
| `--price-anchors` | str | `code/resources/anchors/precio_valor.json` | JSON de anchors (F4.2). |
| `--price-tau-quantile` | float | `0.95` | Cuantil del placebo usado como τ. |
| `--price-placebo-docs` | int | `300` | Docs aleatorios en el placebo de τ. |
| `--price-mode` | `centered`/`raw` | `centered` | Modo de score del detector. |
| `--price-validation-n` | int | `100` | Muestra estratificada de anotación (F4.3). |
| `--price-top-k` | int | `10` | Top-k ejemplos por corpus/fuente (F4.4). |

## Tarea `price` (F4.2–F4.4)

`--comparison-tasks price` (incluida por defecto al final de la lista; las tareas
previas conservan su número de paso) ejecuta `run_price_core`:

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
`--force-steps` recalcula los pasos indicados.

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
