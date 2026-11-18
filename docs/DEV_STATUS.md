# DEV STATUS

Estado de desarrollo del pipeline (una línea por hito, más reciente primero).

| Fecha | Hito | Estado | Notas |
|---|---|---|---|
| 2026-11-18 | F4.1 anchors precio/valor | ✅ | `code/resources/anchors/precio_valor.json` v1.0 + prompt (sha256 f11d9177…) + `docs/ANCHORS.md`. corpus_seen=false; revisión humana pendiente. |
| 2026-11-18 | F4.2 score semántico + τ placebo | ✅ | `code/price_concept.py` (funcs puras: load/valida anchors, lexical_hits con plegado de acentos, raw/centered cosine, compute_price_scores, calibrate_tau) + `run_price_core` (outputs en `results/comparison/price/`: price_scores, tau_calibration, tau_sensitivity). 14 tests en `tests/test_price_concept.py`. pytest requiere env `full_py`. |
| 2026-11-18 | F4.3 validación 100 docs | ✅ | `code/price_validation.py` (make_stratified_sample 5 bandas×corpus con peso `N_h/n_h`, load_annotations, cohen_kappa, classification_metrics, bootstrap_metrics_ci, ablation_predictions, run_price_validation) + `docs/ANNOTATION_GUIDE.md` + plantilla `annotations/precio_valor_labels.csv` (vacía → "validación pendiente de anotación", sin fallo). Con etiquetas: `price_validation.csv`, `price_ablations.csv`, `confusion_matrix.csv`. 10 tests en `tests/test_price_validation.py`. |
| 2026-11-18 | F4.4 panel precio/valor | ✅ | `code/price_concept.py::run_price_panel` (tareas `price` en `run_analysis.py`): en `results/comparison/price/` → `mention_rate_by_source.csv`, `score_distribution_tests.csv` (KS/W1/bootstrap/permutación), `valence_conditioned.csv`, `price_context_logodds.csv`, `price_top_examples.csv`, `price_scatter_prototypes.html`+`_data.csv` (Scattergl, tamaño=lex_hits, líneas τ), `price_scatter_valence.html`+`_data.csv`, `price_summary.html` + `price_density.csv`; consola con procedencia anchors, τ, tasas+CI, KS/W1, valencia, top-5 y frase interpretativa. Metadata `price_panel`/`price_validation` en run_metadata.csv. |
| 2026-11-18 | Huecos rúbrica CLI | ⏳ | Pendiente: `--source-col`, `--dmo-label`, `--title`, `--preprocess full`, `--sample`; README.md, requirements.txt, fixtures CSV. |
| — | Bloque I pasos 0–7 | ✅ | Estable, no tocar salvo threading de args con defaults idénticos. |
| — | Bloque II pasos 8–11 (base, topics, lexical, thematic) | ✅ | `run_analysis.py`; `run_comparison.py` obsoleto (NO usar). |
| — | Suite completa | ⚠️ | 2 fallos PREEXISTENTES no relacionados con F4: `test_joint_topic_model.py::test_parsimonia_elige_menor_k`, `test_topic_modeling_backends.py::test_backend_bertopic_produce_topicos`. |
