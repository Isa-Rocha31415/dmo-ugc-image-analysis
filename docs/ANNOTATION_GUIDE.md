# Guía de anotación — Detector Precio/Valor/Costo (F4.3)

## Objetivo
Estimar la calidad del detector automático (`price_score >= τ`) contra juicio
humano sobre una muestra estratificada de 100 documentos
(`results/comparison/price/annotation_sample.csv`).

## Etiquetas (solo 3 opciones, en minúsculas)
| label | Significado |
|---|---|
| `precio` | El documento menciona precio/costo/valor de manera SUSTANTIVA: cuánto se pagó, si valió lo que costó, tarifas, ofertas, relacion calidad-precio. Contar también quejas de cobro ("nos cobraron de más"). |
| `no_precio` | No hay mención sustantiva de precio. Menciones incidentales ("el precio del tour es aparte" en un texto de otra tema dominante) también cuentan como `no_precio` si el asunto central es otro. |
| `ambiguo` | No puedes decidir. Se descarta del cómputo (no la uses de relleno). |

Reglas:
1. Etiqueta la INTENCIÓN dominante del documento completo, no una frase suelta.
2. Promocionales (voz DMO) que "venden" valor por precio → `precio`.
3. No consultes el `price_score` al etiquetar: evalúa solo el texto.
4. Una fila por (doc_id, anotador); para doble anotación repite el doc con otro
   `annotator` (p. ej. `isa` y `annotador2`).

## Formato
`annotations/precio_valor_labels.csv` — columnas exactas:
`doc_id, annotator, label, note` (note opcional). Codificación UTF-8.

Los `doc_id` provienen de `annotation_sample.csv` (incluye el texto). Puedes
copiar la plantilla vacía que el pipeline crea en `annotations/`.

## Qué hace el pipeline con esto
- Sin etiquetas: escribe la muestra + guía y termina sin fallar
  ("validación pendiente de anotación").
- Con etiquetas: consenso por mayoría (empate = descartado), precisión/recall/
  F1/MCC muestrales y ponderadas (peso w_h = N_h/n_h del diseño estratificado),
  CI bootstrap, matriz de confusión, κ de Cohen si hay ≥2 anotadores,
  y ablaciones (lexical-only, resena-only, promo-only, raw-max, centered-max,
  hybrid, tau_fijo_0.5) en `price_validation.csv` / `price_ablations.csv`.
