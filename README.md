# Análisis Comparativo: Imagen Proyectada (DMO) vs Imagen Percibida (UGC)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Objetivo

Realizar un analisis de texto completo creando un dasboard interactivo en formato de pagina web dondese  compara la **imagen de destino proyectada** por una Organización de Marketing de Destino (DMO) con la **imagen percibida** por usuarios en fuentes digitales. 


## 🚀 Instalación
poner que debe tener conda para la creacion del entorno
### Con requiriments.txt
pon los pasos
### Con yaml
pon los pasos 


## Como se usa?
explicar como se usa 
el comando principal es 
    python run_analysis.py \
        --csv-dmo "../data/df_dmo.csv" \
        --csv-ugc "../data/df_ugc.csv" \
        --text-col text --lang es \
        --outlier-method IF --contamination auto \
        --min-partition 50 \
        --output-dir ../results \
        --model bertopic --palette ejecutiva --num-topicos automatico


### Explicando cada uno de los arguemntos del comando

### comandos de ejemplo

### 