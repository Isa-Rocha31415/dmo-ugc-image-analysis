# DEV STATUS

Estado de desarrollo del pipeline (una línea por hito, más reciente primero).

| Fecha | Hito | Estado | Notas |
|---|---|---|---|
| 2026-11-18 | F4.1 anchors precio/valor | ✅ | `code/resources/anchors/precio_valor.json` v1.0 + prompt (sha256 f11d9177…) + `docs/ANCHORS.md`. corpus_seen=false; revisión humana pendiente. |
| 2026-11-18 | F4.2 score semántico + τ placebo | ✅ | `code/price_concept.py` (funcs puras: load/valida anchors, lexical_hits con plegado de acentos, raw/centered cosine, compute_price_scores, calibrate_tau) + `run_price_core` (outputs en `results/comparison/price/`: price_scores, tau_calibration, tau_sensitivity). 14 tests en `tests/test_price_concept.py`. pytest requiere env `full_py`. |
| 2026-11-18 | F4.3 validación 100 docs | ⏳ | Pendiente: plantilla gold CSV + métricas + degradación elegante. |
| 2026-11-18 | F4.4 panel precio/valor | ⏳ | Pendiente: `price_value_task.py`, CSVs + panel plotly en `results/comparison/price_value/`. |
| 2026-11-18 | Huecos rúbrica CLI | ⏳ | Pendiente: `--source-col`, `--dmo-label`, `--title`, `--preprocess full`, `--sample`; README.md, requirements.txt, fixtures CSV. |
| — | Bloque I pasos 0–7 | ✅ | Estable, no tocar salvo threading de args con defaults idénticos. |
| — | Bloque II pasos 8–11 (base, topics, lexical, thematic) | ✅ | `run_analysis.py`; `run_comparison.py` obsoleto (NO usar). |
| — | Suite completa | ⚠️ | 2 fallos PREEXISTENTES no relacionados con F4: `test_joint_topic_model.py::test_parsimonia_elige_menor_k`, `test_topic_modeling_backends.py::test_backend_bertopic_produce_topicos`. |
