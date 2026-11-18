# Análisis Comparativo: Imagen Proyectada (DMO) vs Imagen Percibida (UGC)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Objetivo

Pipeline de análisis NLP que compara la **imagen de destino proyectada** por una Organización de Marketing de Destino (DMO) con la **imagen percibida** por usuarios en fuentes digitales (Instagram, TripAdvisor, Reddit, etc.).

**Aporte diferencial:** Cuantificación de brechas temáticas, léxicas, de sentimiento y semánticas entre comunicación oficial y percepción real, con inferencia estadística robusta.

## 🚀 Instalación

### Requisitos

- Python 3.10+
- Conda con entorno `image_analysis` configurado
- Modelos HuggingFace cacheados localmente (ver `requirements.txt`)

### Setup

```bash
# Crear entorno (si no existe)
conda create -n image_analysis python=3.10

# Activar
conda activate image_analysis

# Instalar dependencias (offline, requiere caché local)
pip install -r requirements.txt --no-deps --no-build-isolation