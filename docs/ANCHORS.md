# ANCHORS — Recursos estáticos por concepto

## Qué son
`code/resources/anchors/<concept_id>.json` define, **antes de ver el corpus**, los
prototipos textuales, el ancla léxica y los controles negativos de cada concepto
temático analizado (p. ej. `precio_valor`). Son recursos versionados: el pipeline
los trata como parámetros de entrada, no como salidas de datos.

## Por qué estáticos
1. **Reproducibilidad**: el score semántico debe poder recalcularse idéntico en
   cualquier corrida; un anchor generado en caliente cambiaría entre ejecuciones.
2. **Auditoría**: cada JSON lleva `provenance` (generator, model, prompt_sha256,
   seed, created_utc, authors) → la trazabilidad es automática.
3. **Control de calidad**: la revisión humana (`provenance.review`) se hace sobre
   un artefacto estable, no sobre textos efímeros.

## Cómo se evita la fuga de información (leakage)
- Los prototipos se redactan con `corpus_seen=false`: se generan y se congelan
  **antes** de leer cualquier fila del corpus. No se copian ni parafeasean textos
  reales (ver `prompts/<concepto>_prompt.txt`).
- El `prompt_sha256` del prompt generador queda registrado en el propio JSON;
  cualquier cambio de prompt exige nueva versión del anchor.
- Controles negativos con especificidad comparable pero temática excluyente
  (precio vs. seguridad/clima/transporte/gastronomía) permiten calibrar τ
  (umbral) por placebo sin usar etiquetas del corpus.

## Consumo por el pipeline
- El pipeline **solo lee el JSON** (F4.2: `metrics_price_value.py`).
- Los prototipos y controles se embedean con el modelo local ya cacheado
  (misma instancia de sentence-transformers del PASO 2); nunca se descargan
  modelos ni se llaman APIs externas (entorno offline estricto).
- La ancla léxica (`terms_*`, `patterns`) se aplica con la tokenización regex ya
  existente; `patterns` son regex compatibles con `re` (case-insensitive).

## Validación automática (F4.3, sin anotación humana)
El detector primario es `semantic`: coseno entre el embedding del documento y el
centroide L2 del ancla sintética (media normalizada de los embeddings de sus
términos/frases). `lexical` e `hybrid` se conservan solo como ablación de
robustez; los scores de registro (reseña vs promocional) son una variable
secundaria de encuadre.

F4.3 se ejecuta sin intervención humana sobre scores ya calculados:
`kappa_semantic_vs_lexical` (κ de Cohen contra una referencia débil —
`reference_type="weak_lexical_reference_not_gold_standard"`; NO es exactitud ni
acuerdo inter-anotador), FPR sobre controles negativos y docs aleatorios,
sensibilidad de τ en {p90, p95, p97.5} con `gap_sign_stable`, jackknife del
ancla (200 réplicas quitando 20% de términos, seed 42) e IC bootstrap 95% de
tasas y cosenos por fuente (n_boot ≥ 1000 por defecto).

**Limitación declarada:** no hay anotación humana. El JSON registra
`provenance.review.status = "validacion_automatica"` y `review.method` con las
cinco comprobaciones. El flag CLI `--price-annotation-file` queda aceptado pero
ignorado con `DeprecationWarning`. `anchors_version` y `anchors_sha256` se
propagan a todos los CSV de `comparison/price/` para trazabilidad.

## Cómo revisar / editar
1. Cambia el JSON (o el prompt) y **sube `version`** (1.0 → 1.1). Nunca edites
   una versión publicada sin bump: la caché de scores clavea por `concept_id`
   + `version`.
2. Registra en `provenance.review` quién revisó, fecha y notas.
3. Si el prompt cambió, recalcula su `prompt_sha256`:
   ```bash
   python3 - <<'PY'
   import hashlib
   print(hashlib.sha256(open("code/resources/anchors/prompts/precio_valor_prompt.txt","rb").read()).hexdigest())
   PY
   ```
4. Valida el esquema mínimo:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   d = json.loads(Path("code/resources/anchors/precio_valor.json").read_text(encoding="utf-8"))
   assert d["concept_id"] == "precio_valor" and len(d["prototypes"]) >= 4
   assert len(d["negative_controls"]) >= 4 and d["provenance"]["corpus_seen"] is False
   print("✅ anchors OK")
   PY
   ```
5. No metas nombres de hoteles/ciudades/fechas del caso de estudio: el anchor
   debe ser reutilizable en otro destino.
