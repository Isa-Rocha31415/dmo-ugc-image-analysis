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
