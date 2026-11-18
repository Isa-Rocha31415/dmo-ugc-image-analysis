#!/bin/bash
#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0
#SBATCH --output=salida_gpu%J.out
#SBATCH --error=error_gpu%J.err

#Manda correo de inicio o si fallo la ejecucion (opciones: BEGIN, END, FAIL, REQUEUE, ALL)
###SBATCH --mail-type=END,FAIL
###SBATCH --mail-user=tucorreo@cimat.mx

# Ir al directorio desde donde se envió el job
cd $SLURM_SUBMIT_DIR

# Inicializar Conda
source /opt/anaconda3_titan/etc/profile.d/conda.sh
conda activate nlp_rodri

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1


export TOKENIZERS_PARALLELISM=false

# Información de diagnóstico
echo "Fecha y hora: $(date)"
echo "Nodo: $HOSTNAME"
echo "Usando Python en: $(which python)"
python --version
conda info --envs

# Ejecutar el script
# python test
python -c "import torch; print(torch.__version__); print('PyTorch GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sin GPU')"


python run_analysis.py --csv-dmo ../data/processed/df_dmo.csv \
  --csv-ugc ../data/processed/df_ugc.csv --text-col text \
  --output-dir ../results --comparison-tasks base topics thematic --bootstrap 300
    
echo "Fin: $(date)"
echo "Finalizó el script con código de salida $?"


hostname
nvidia-smi
