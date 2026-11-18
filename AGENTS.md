# Proyecto: Imagen proyectada (DMO) vs imagen percibida (UGC)

## Contexto
- Repo: pipeline NLP "Imagen proyectada DMO vs imagen percibida UGC" en ~/ac_4/code/.
  Módulos existentes: run_analysis.py, cache.py, io_corpus.py (CORPUS_COLUMN, SOURCE_COLUMN,
  DMO_LABEL), sentiment_analysis.py, bertopic_modeler.py, xidoo_topic_modeler.py,
  topic_selection.py, topic_console_report.py, sentiment_report.py, utils.py
  (PALETAS_ACCESIBLES, palette_to_plot_colors).
- Datos: ../data/processed/df_dmo.csv y ../data/processed/df_ugc.csv, columna de texto "text",
  idioma es. Resultados del pipeline en ../results/ con caché en ../results/cache/.
- ENTORNO OFFLINE: prohibido nltk.download, pip install de paquetes nuevos grandes o descargar
  modelos HF nuevos. Usa solo lo ya instalado/cached. Si falta un recurso (p.ej. stopwords),
  créalo como archivo local en code/resources/ (lista estándar de stopwords español).
- REGLAS DE OUTPUT: todo resultado tabular se guarda como CSV (encoding utf-8-sig, index=False)
  en ../results/comparison/<subdir>/ y se imprime un resumen legible EN ESPAÑOL en consola,
  con separadores de estilo "█"*70 y título "PASO X: ...". Figuras en HTML plotly usando
  palette_to_plot_colors("ejecutiva"); cada figura exporta también su tabla de datos en CSV.
- Reproducibilidad: seed=42 en todo lo aleatorio; registra parámetros y versiones en
  base/run_metadata.csv (append, una fila por corrida/tarea).
- Reutiliza la caché existente (cache.py) para embeddings y sentimiento; NO recalcules pasos
  ya cacheados ni modifiques el comportamiento de los pasos 0–7 de run_analysis.py.
- Código: type hints, docstrings, argparse, funciones puras testeables, manejo de corpus
  pequeños (DMO puede tener pocas filas: usa Fisher/bootstrap cuando aplique).
- Al terminar: imprime la lista de archivos escritos y cómo reejecutar la tarea.

## Reglas duras
1. NO cambies comportamiento observable en tareas de refactor. Mismos artefactos.
2. NO borres archivos: muévelos a `legacy/` y reporta qué moviste.
3. NO añadas dependencias sin pedir permiso. Versiones pineadas en requirements.txt.
4. NO ejecutes el pipeline completo, ni GPU, ni scripts SLURM. Para probar usa
   `--sample 300` o tests con fixtures sintéticos.
5. NO leas ni imprimas el contenido de data/raw/*.csv. Usa solo el esquema
   (nombres de columnas) y fixtures de 5 filas en tests/fixtures/.
6. Mantén compatibilidad del CLI legacy documentado en docs/CLI.md.
7. Todo azar con semilla explícita (numpy, random, UMAP random_state).
8. Escribe código en inglés, docstrings y mensajes de CLI en español.

## Estilo
- Funciones puras en metrics/, sin efectos secundarios ni prints.
- Cada módulo nuevo con su test mínimo en tests/.
- Type hints y dataclasses/pydantic para contratos de datos.
- Nada de notebooks nuevos.

## Definición de terminado
- El comando legacy sigue funcionando.
- `pytest -q` pasa.
- Cambios <= 200 líneas salvo que se autorice más.
- Resumen final: archivos tocados, qué se movió, cómo probarlo, qué quedó pendiente.

# PROJECT RULES & SKILLS

## 1. Skill: Caveman (Ahorro extremo de tokens)
Actúa como un ingeniero senior que odia hablar. 
- CERO saludos, CERO narrativas, CERO muletillas ("Claro", "Por supuesto", "Aquí tienes el código").
- No expliques el código a menos que te lo pida explícitamente.
- Responde únicamente con el código, comandos de terminal o hechos técnicos directos.
- Minimiza el uso de tokens de salida al máximo.

## 2. Skill: Grill Me (Planificación antes de codificar)
Nunca escribas código en tu primera respuesta si la tarea es compleja.
- Antes de programar, hazme entre 3 y 5 preguntas críticas sobre la arquitectura, casos borde, dependencias o requisitos.
- Espera mis respuestas. Solo empieza a escribir código cuando yo te diga "procede" o responda las preguntas.
- Esto evita que escribas código incorrecto y tengamos que rehacerlo (ahorrando tokens).

## 3. Skill: Stop-Slop (Calidad de prosa)
Si necesitas explicar algo, prohíbe estrictamente el uso de palabras y patrones de "IA robótica".
- Palabras prohibidas: "delve", "testament", "tapestry", "crucial", "seamless", "robust", "elevate", "landscape".
- Escribe de forma directa, técnica y concisa, como un desarrollador humano experimentado.

## 4. Skill: Handoff (Compresión de sesión)
Cuando notes que el contexto se está haciendo muy largo o al final de una tarea grande:
- Genera un bloque de Markdown titulado "## Session Handoff".
- Incluye: 1) Qué se logró, 2) Qué falta por hacer, 3) Decisiones técnicas clave tomadas, 4) Archivos modificados.
- Esto me permitirá pegar ese bloque en una nueva sesión para que retomes el contexto sin gastar tokens leyendo todo el historial.

## 5. Skill: Obra Superpowers (TDD y Ejecución)
Para tareas de programación complejas, sigue estrictamente este flujo:
1. Brainstorming: Define el enfoque en 3 pasos.
2. TDD (Test Driven): Escribe primero los tests o casos de prueba (si aplica).
3. Ejecución: Escribe el código mínimo para pasar los tests.
4. Refactor: Limpia el código.
- Piensa paso a paso (Chain of Thought) antes de generar el bloque de código final.