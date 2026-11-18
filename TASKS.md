
## F0.8 — Reorganización inicial
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F0.1 | Reorganización a paquetes por lotes + borrar ruido a legacy/ | Estructura limpia, imports resueltos |
| F0.2 | Imports resueltos, tests pasan, tree limpio | Tests verdes, sin warnings de import |
| F0.3 | Cada corrida escribe manifest con config, versiones, seeds, hashes | Manifest generado automáticamente |
| F0.5 | Pipeline mínimo end-to-end | Corre de inicio a fin sin errores |

## F1 — Pipeline mínimo end-to-end
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F1.1 | Cache de embeddings desacoplado (GPU produce .npy, CPU consume) | Segunda corrida no recalcula embeddings; corre en CPU |
| F1.2 | Espacio temático compartido + chunking DMO + proporciones por corpus | topics_df tiene corpus, topic_id, y %DMO/%UGC por tópico |
| F1.3 | Contrato de backends `bert` + `xidoo` + semántica documentada | Documentación clara de interfaces |
| F1.4 | Sentimiento: backends + score calibrado + desglose por fuente | Tabla de sentimiento por fuente y por tópico |
| F1.5 | Dashboard v1 desde manifest (5 figuras + tabla) | Un comando genera el HTML |

## F2 — Métricas de alineación (el corazón de la nota)
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F2.1 | L1 léxica: Jaccard, cobertura, log-odds con prior | metrics_lexical.json + nubes de log-odds |
| F2.2 | L2 temática: gaps en pp + bootstrap CI + χ²/Fisher + FDR | gaps_table.csv con GAP, CI, p-adj, veredicto |
| F2.3 | L3 semántica: centroides, EMD, kNN recall + centroides UGC-pos/UGC-neg | metrics_semantic.json |
| F2.4 | L4 afectiva + IAS compuesto + sensibilidad de pesos + test de permutación | KPI con CI y las 4 capas reportadas por separado |
| F2.5 | Figuras: heatmap fuente×tópico, mapa de cuadrantes, ranking de severidad | Panel de gestión en el dashboard |

## F3 — Evidencia
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F3.1 | Barrido de K + coherencia C_v/C_nPMI + diversidad + estabilidad (ARI) | Curva de selección de K con tabla justificada |
| F3.2 | Accuracy de sentimiento: plantilla gold + linear probe + ablación + ratings TripAdvisor como etiqueta débil | Matriz de confusión, macro-F1, κ, tabla de ablación |
| F3.3 | Comparación de backends: ARI/AMI/NMI + matching húngaro + ¿cambia el ranking de brechas? | Tabla de robustez bert vs xidoo |
| F3.4 | Fichas de tópico automáticas con documentos representativos | topic_cards.md con evidencia citada |

## F4 — Método propuesto (precio/valor)
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F4.1 | Prototipos LLM (dos registros: reseña y promocional) + ancla léxica, con trazabilidad | anchors/precio_valor.json + prompts y seeds guardados |
| F4.2 | Score de proximidad + umbral τ por distribución nula + placebo | τ justificado con percentil, no a ojo |
| F4.3 | Validación: precisión/recobrado vs 100 docs anotados + ablaciones | Tabla de validez del detector |
| F4.4 | Panel precio/valor: tasa de mención, KS/Wasserstein, valencia condicionada | Figura + ejemplos citados |

## F5 — Cierre
| ID | Objetivo | Terminado cuando |
|----|----------|------------------|
| F5.1 | Dashboard final + panel de evidencia + drill-down | HTML autocontenido navegable |
| F5.2 | Reproducibilidad: README, requirements pineado, un comando | Clon limpio reproduce todo |


## Checklist de progreso
- [ ] F0.1
- [ ] F0.2
- [ ] F0.3
- [ ] F0.5
- [ ] F1.1
- [ ] F1.2
- [ ] F1.3
- [ ] F1.4
- [ ] F1.5
- [ ] F2.1
- [ ] F2.2
- [ ] F2.3
- [ ] F2.4
- [ ] F2.5
- [ ] F3.1
- [ ] F3.2
- [ ] F3.3
- [ ] F3.4
- [ ] F4.1
- [ ] F4.2
- [ ] F4.3
- [ ] F4.4
- [ ] F5.1
- [ ] F5.2
- [ ] F5.3