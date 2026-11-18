# AUDIT — code/main.py vs code/run_analysis.py

Fecha: 2026-11-16. Modo solo lectura: no se modificó ningún `.py`.
Alcance: lectura de `code/{main,run_analysis,df_unified,utils,cache,topic_modeling,xidoo_topic_modeler,sentiment_analysis,outlier_detector,visualization}.py`,
listado de `code/` y `code/__pycache__/`. No se leyó `data/raw/*.csv`.

## 1. Comparativa de entrypoints

| Aspecto | `code/main.py` (`run_pipeline` en `code/cache.py:1222`) | `code/run_analysis.py` (`main()` en `code/run_analysis.py:37`) |
|---|---|---|
| Pasos | load `cache.py:1268` → preprocess `1287` → embeddings `1309` → outliers `1328` → sentiment `1355` → topics `1381` → visuals `1420` | carga+limpieza `run_analysis.py:61,74-75` → embeddings BERT `82-85` → outliers+n-gramas `91-103` → viz outliers `113-125` → sentiment `132-152` → reporte sentimiento `155-170` → wordclouds `178-189` → **PASOS 7-8 + resumen summary muertos** (bloque triple-quoted `191-226`) |
| Tópicos | sí, por sentimiento (`topic_modeling.py:137`) | **no**: modelado de tópicos está comentado (`run_analysis.py:191-226`) |
| Brecha DMO vs percibido | sí (`cache.py:1033` → `visualization.py:691`) | no (`--dmo-label` es obligatorio pero solo se usa en código muerto `run_analysis.py:216`) |
| Caché / manifest / histórico | sí (`cache.py:247-1216`, `history.jsonl` `1481`, `latest_run.json` `1484`) | no |
| Embeddings | `embeddings_generator` **no existe** → TF-IDF 500d (`cache.py:410-427,489-492`) | BERT real vía `embeddings_comparison.obtener_embeddings_bert` (`run_analysis.py:82`) |
| Artefactos | `results/plots/07..18_*.html`, `df_clean_after_outliers.csv` `cache.py:1345`, `df_with_sentiment.csv` `1371`, `topics_by_sentiment.csv` `1399`, `topic_counts_by_source.csv` `1412` | `results/outliers/` `run_analysis.py:97`, `results/sentiment/` `160`, `results/sentiment/wordclouds/` `184`, `results/topics/` `202` |
| Flags | 23 (ver §2) | 11: `--csv --text-col --source-col --dmo-label --lang --outlier-method --contamination --min-partition --output-dir --top-n --global-outliers` (`run_analysis.py:39-53`) |
| Completitud | **más completo**: 7 pasos + caché + gap DMO/UGC + todos los flags del comando real | menos completo: sin caché, sin tópicos, sin gap; pero es el que produce los artefactos existentes |

**Canónico recomendado: `code/main.py`.** Evidencia: (a) es el único que acepta los 23 flags del comando real del usuario (`main.py:16-152`); (b) es el único con caché/manifest/historial (`cache.py:1443-1484`); (c) es el único que calcula la brecha DMO‑vs‑percibido (`cache.py:1033-1039` → `visualization.py:718-734`); (d) `run_analysis.py` perdió pasos 7‑8 en edición reciente (bloque `191-226`).

**Pero la evidencia de uso apunta al revés:** `code/slurm_script_CUDA.sh:42` ejecuta `python run_analysis.py`; en `results/` existen `outliers/`, `sentiment/`, `topics/`, `embeddings_comparison/` (salida de `run_analysis.py`) y **no** existen `results/plots/`, `results/cache/`, `history.jsonl` → `main.py` no se ha ejecutado en este estado del repo. Motivo del fallo: **NO VERIFICADO**.

## 2. Flags del entrypoint canónico (`code/main.py`)

| Flag | Tipo | Default | Consumido en | Efecto observable |
|---|---|---|---|---|
| `--csv` | str (req.) | — | `cache.py:1268` → `step_load` `284` | Carga el CSV unificado; hash de archivo define la clave de caché (`cache.py:86-103,255-262`) |
| `--text-col` | str (req.) | — | `cache.py:1290`, `1299` | Columna de texto a preprocesar; para sentimiento/tópicos manda `processed_text` si existe (`cache.py:193-206`) |
| `--source-col` | str (req.) | — | `cache.py:1277,1331,1358,1385` | Agrupa por fuente en todos los pasos |
| `--dmo-label` | str (req.) | — | `cache.py:1277-1278`, `1036` | Valor de `source_col` que significa DMO; el resto es UGC |
| `--lang` | str (req.) es/en/fr/pt | — | `cache.py:1291,1359` | Preprocesa en ese idioma; en sentimiento **no tiene efecto** (`sentiment_analysis.py:260` no lo usa) |
| `--title` | str (req.) | — | `cache.py:1452` | Solo se guarda en `history.jsonl`; no aparece en los gráficos |
| `--palette` | str (req.) | — | `cache.py:1335` → `outlier_detector.py:657` → `utils.get_palette` `utils.py:214` | Solo afecta los plots `06_*` de outliers (`visualization.py:55-72`); los plots del dashboard usan colores fijos/Plotly (`visualization.py:424-427`) |
| `--output-dir` / `-o` | str | `results` | `cache.py:1250-1254` | Plots en `results/plots/`, caché en `results/cache/`, CSVs en `results/` |
| `--outlier-method` | LOF/IF | `LOF` | `cache.py:1333` → `outlier_detector.py:101` | Elige la columna `outlier_IF`/`outlier_LOF`; **ambos detectores se calculan siempre** (`outlier_detector.py:90-91`) |
| `--contamination` | float | `0.05` | `cache.py:1334` → `outlier_detector.py:90-91` | Proporción de outliers de IF y LOF (global, no por fuente) |
| `--n-topics` | int | `None` | `cache.py:1233,1457` | **Sin efecto**: solo se registra en el histórico; nunca llega a `run_topic_modeling` (`cache.py:879-887`) |
| `--embedding-backend` | auto/generator/fallback | `auto` | `cache.py:1314` → `415-427` | `auto` resuelve a `tfidf_fallback` porque `embeddings_generator` no existe; `generator` lanza ImportError (`cache.py:419-423`) |
| `--cache-dir` | str | `None` | `cache.py:1254` | Raíz de la caché; por defecto `results/cache` |
| `--no-cache` | bool | `False` | `cache.py:1236` | `use_cache=False` → recalcula todo |
| `--force-steps` | lista | `[]` | `cache.py:1272,1294,1316,1340,1366,1392,1435` | Recalcula el paso indicado; **`comparison` es una choice sin paso real** (`main.py:87` vs lista de `cache.py`) |
| `--topic-backend` | xidoo/wordfreq | `xidoo` | `cache.py:1386` → `topic_modeling.py:151` | `!= "xidoo"` fuerza frecuencia de palabras; con `xidoo` y ≥`min_partition_size` usa XidooTopic (`topic_modeling.py:177`) |
| `--sentiment-backend` | transformers/auto | `transformers` | `cache.py:1359` → `sentiment_analysis.py:297` | Solo se acepta `transformers`/`auto`; cualquier otro valor lanza ValueError |
| `--sentiment-model` | str | `None` | `cache.py:1361` → `sentiment_analysis.py:294` | Modelo HF; por defecto `cardiffnlp/twitter-xlm-roberta-base-sentiment` (`sentiment_analysis.py:9`) |
| `--sentiment-threshold` | float | `0.0` | `cache.py:1360` → `sentiment_analysis.py:322-325` | `positive` si `score > threshold` (estricto); `run_analysis.py:26` usa `>=` → inconsistencia |
| `--min-partition-size` | int | `50` | `cache.py:1385` → `topic_modeling.py:151` | Si la partición positive/negative completa es menor, se usa frecuencia de palabras. **No particiona por fuente** (`topic_modeling.py:137`) |
| `--use-xidoo-outliers` | bool | `False` | `cache.py:1387` → `topic_modeling.py:178` → `xidoo_topic_modeler.py:134-137` | Intenta `xidootopic(outlier_detector=None)` vs constructor por defecto |
| `--umap-method` | umap/pca | `umap` | `cache.py:1430` → `visualization.py:266-291` | UMAP 2D con `random_state=42`; **si UMAP falla cae a PCA en silencio** (`visualization.py:280-282`) |
| `--max-plot-points` | int | `12000` | `cache.py:1431` → `visualization.py:329-333` | Submuestreo determinista (`np.random.default_rng(42)`) de los puntos UMAP |

## 3. Grafo de dependencias (aristas)

```
main.py            -> cache, utils
run_analysis.py    -> embeddings_comparison, outlier_ngram_analysis, outlier_visualizations,
                      sentiment_analysis, sentiment_report, sentiment_wordclouds,
                      topic_console_report   (este último solo en bloque muerto, línea 196)
cache.py           -> text_preprocessing, outlier_detector, sentiment_analysis,
                      topic_modeling, visualization, embeddings_generator (AUSENTE, opcional)
outlier_detector.py-> visualization
visualization.py   -> utils
topic_modeling.py  -> xidoo_topic_modeler
topic_console_report.py -> topic_modeling, visualization
df_unified.py      -> (ningún módulo local; script de datos)
utils.py           -> (ningún módulo local)
```

## 4. Código muerto

Módulos que existen pero **nadie importa desde el pipeline canónico** (`main.py`/`cache.py`):
`embeddings_comparison.py`, `outlier_ngram_analysis.py`, `outlier_visualizations.py`, `sentiment_report.py`, `sentiment_wordclouds.py`
(solo los usa `run_analysis.py:82,91,113,155,178`) y `topic_console_report.py` (solo `run_analysis.py:196`, dentro del bloque muerto `191-226`).

Scripts sueltos no importados por nadie: `analisis_pre.py`, `download_models.py`, `pruebas.py`, `report_generator.py` (0 bytes, vacío), `df_unified.py`
(este último sí es útil como paso de datos: produce `data/processed/data_unified.csv`, `df_unified.py:77-79`).

Funciones definidas y nunca invocadas: `visualization.py:119` (`graficar_distribucion_fuentes`), `visualization.py:770` (`plot_ngramas_outliers`),
`visualization.py:846` (`plot_sentiment_by_source_grouped`), `visualization.py:932` (`plot_embeddings_comparison` — **rota**: llama
`_reducir_umap_3d` en `951-952`, función que no existe en el repo), `topic_modeling.py:281` (`topic_keywords_to_dataframe`).
`visualization.py:1029` y `1165` solo se usan desde `topic_console_report.py:98` → cadena muerta.

Artefactos `__pycache__` obsoletos (NO son evidencia de módulos vivos):
`code/__pycache__/embeddings_generator.cpython-310.pyc` (217 B) **sin `.py` correspondiente** → `cache.py:410` siempre falla y el pipeline usa TF-IDF.
Los demás `.pyc` (`cache`, `sentiment_analysis`, `topic_modeling`, `visualization`, `utils`, `outlier_detector`, `text_preprocessing`,
`xidoo_topic_modeler`, `embeddings_comparison`, `outlier_ngram_analysis`, `outlier_visualizations`, `sentiment_report`, `sentiment_wordclouds`,
`topic_console_report`) son bytecode obsoleto de módulos que sí existen.

## 5. Dónde se decide DMO vs UGC

- Ingesta: la etiqueta DMO nace como valor de la columna `source` (`df_unified.py:42` → `"sectur"`; UGC en `df_unified.py:22,32,53`).
- Split principal (solo informativo): `cache.py:1277-1278` (`df_dmo = df[source==dmo_label]`, `df_ugc = resto`), impreso en `cache.py:1280` y **nunca reutilizado**.
- Decisión real: `visualization.py:718-719` (`dmo_df = tmp[source==dmo_label]`, `perceived_df = resto`) dentro de `plot_dmo_vs_perceived_topic_gap`, invocada en `cache.py:1033-1039` (gráfico `13_brecha_topics_dmo_vs_percibido.html`).
- `dmo_value` se pasa al detector de outliers (`cache.py:654`) pero **el parámetro nunca se usa** en `outlier_detector.py:73-129` (solo se firma en `outlier_detector.py:78`).
- Sentimiento/outliers/tópicos **no separan DMO de UGC**: son globales; solo se desagrega por `source` al resumir (`cache.py:1406-1413`).
- Solo en el entrypoint muerto: `embeddings_comparison.py:169` (`mask_dmo`).

## 6. Cinco riesgos para refactorizar

1. **Embeddings silenciosamente TF-IDF**: `embeddings_generator` no existe → `cache.py:410-427,489-492`; cualquier refactor que asuma BERT cambia todas las métricas sin error visible.
2. **Caché acoplada a rutas/flags**: la clave incluye hash del CSV (`cache.py:86-103`) y `parent_key` por paso; `--force-steps` solo fuerza un paso (`cache.py:82-83`) y cualquier cambio upstream invalida todos los hijos. La caché vive bajo `results/cache` (`cache.py:1254`) → borrar `results/` la destruye.
3. **DMO/UGC es igualdad de string sobre una columna**: `cache.py:1277` y `visualization.py:718`; pasar a dos corpus cambia 4 gráficos (`cache.py:954-1116`) y el conteo de particiones.
4. **Paso de outliers sin red de seguridad**: `spacy.load("es_core_news_sm")` (`visualization.py:24`) se ejecuta dentro de `analizar_ngramas_outliers` (`outlier_detector.py:119`), fuera de `_safe_plot` (`cache.py:233-241`) → si falta el modelo, cae todo el pipeline; además siempre se corren IF y LOF (`outlier_detector.py:90-91`) y el paso no está protegido por try/except.
5. **Resultados que cambian en silencio + falta de red de tests**: fallback Xidoo→frecuencia (`topic_modeling.py:217-228`), UMAP→PCA (`visualization.py:280-282`), TF-IDF en vez de embeddings, y umbrales `>` vs `>=` (`sentiment_analysis.py:323` vs `run_analysis.py:26`). En el repo **no existen `tests/` ni `requirements.txt`** (verificado por listado del root), así que no hay red que detecte regresiones.

## 7. Estrategia mínima para `--dmo dmo.csv --ugc ugc.csv` (propuesta, no implementada)

1. **CLI**: en `main.py` añadir `--dmo PATH` y `--ugc PATH` (repetición opcional para varios UGC), y dejar `--csv`/`--dmo-label` sin tocar. Regla: si `--dmo`/`--ugc` están presentes, `--csv` y `--dmo-label` se vuelve opcionales (mutua exclusividad validada con `parser.error`).
2. **Nuevo módulo puro** `code/io_corpus.py` con `build_unified_frame(dmo_csv, ugc_csv, text_col, source_col, dmo_label) -> pd.DataFrame`: concatena, aplica `df["source"] = dmo_label` a las filas DMO (precedente `df_unified.py:42`) y añade `df["corpus"] = "DMO"|"UGC"`; sin efectos secundarios ni `print`.
3. **Reutilizar `step_load` sin tocarlo**: `main.py` escribe el CSV unificado resultante en `output_dir/data_unified.csv` y llama a `run_pipeline(input_csv=ese_path, ...)` con `dmo_label` intacto. `step_load` (`cache.py:247`) sigue siendo el único punto de carga → cero cambios de caché y de contrato.
4. **Claves de caché**: el path del CSV fusionado entra en el hash (`cache.py:99`), por lo que las corridas nuevas no colisionan con las viejas; opcionalmente subir `CACHE_VERSION` (`cache.py:28`).
5. **Alineación de esquema**: normalizar nombres de columna de texto antes de concatenar (el CSV DMO traerá `texto`/DMO y el UGC `text`); documentar el mapeo esperado (`--text-col` se aplica al DMO, y `text` al UGC o `--ugc-text-col`).
6. **Tests mínimos** en `tests/` con fixtures de 5 filas (`tests/fixtures/`): (a) `--csv` sigue produciendo el mismo `source`/`dmo_label`; (b) `--dmo/--ugc` produce el mismo número de filas DMO+UGC y `corpus` correcto; (c) el CSV fusionado se puede releer con `pd.read_csv` y conserva el orden. Presupuesto: <200 líneas.

---

**Resumen**

- `code/main.py` es el entrypoint canónico (23 flags, 7 pasos, caché/manifest/histórico, gap DMO‑vs‑percibido); `code/run_analysis.py` es un entrypoint paralelo más pobre (pasos 7‑8 comentados en `191-226`, sin caché ni gap) y es el que usan los scripts SLURM (`code/slurm_script_CUDA.sh:42`) y cuyos artefactos son los que están en `results/`.
- `main.py` nunca se ha ejecutado en este estado (no existen `results/plots/`, `results/cache/` ni `history.jsonl`); causa: **NO VERIFICADO**.
- Los embeddings reales hoy son TF-IDF 500d: `embeddings_generator` solo existe como `.pyc` obsoleto y `cache.py:410` cae siempre al fallback.
- DMO vs UGC se decide por igualdad de string `source == dmo_label` en `cache.py:1277` y `visualization.py:718`; en outliers el parámetro DMO se pasa pero no se usa (`outlier_detector.py:78`).
- Antes de refactorizar: crear `tests/` + `requirements.txt`, blindar el paso de outliers (`spacy` en `visualization.py:24`) y decidir qué entrypoint se mueve a `legacy/`.
