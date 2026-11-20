#!/usr/bin/env bash
# Ejecución LOCAL del pipeline (sin SLURM). Único entry point: run_analysis.py
set -uo pipefail

REPO="${REPO:-$HOME/ac_4_local}"
cd "$REPO/code" || { echo "❌ No existe $REPO/code"; exit 2; }
[[ -f run_analysis.py ]] || { echo "❌ Falta run_analysis.py"; exit 2; }

# conda LOCAL (autodetectado; nunca la ruta del cluster)
if [[ "${CONDA_DEFAULT_ENV:-}" != "nlp_rodri" ]]; then
  CB="$(conda info --base 2>/dev/null)"
  [[ -n "$CB" ]] && source "$CB/etc/profile.d/conda.sh"
  conda activate nlp_rodri || { echo "❌ No se pudo activar nlp_rodri"; exit 2; }
fi

# Offline estricto + logs en vivo
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Hilos: evita sobre-suscripción de BLAS/numba
NT="${NTHREADS:-$(nproc)}"; (( NT > 8 )) && NT=8
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT

LOG="$REPO/logs/local_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "══ Inicio $(date) | host $(hostname) | cwd $(pwd) | hilos $NT"
python -c "import torch;print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available())"

echo "██ PREFLIGHT 1: sintaxis"
python -m py_compile run_analysis.py text_preprocessing.py cache.py price_concept.py \
  joint_topic_model.py metrics_lexical.py metrics_thematic.py \
  comparison/common.py comparison/base_builder.py || { echo "❌ sintaxis"; exit 3; }

echo "██ PREFLIGHT 2: dependencias + modelos offline"
python - <<'PY' || exit 4
import importlib,sys
fail=[]
for m in ["numpy","pandas","scipy","sklearn","plotly","bertopic","hdbscan","umap","torch","transformers","sentence_transformers"]:
    try: importlib.import_module(m)
    except Exception as e: print("❌",m,e); fail.append(m)
import spacy; spacy.load("es_core_news_sm", disable=["parser","ner"])
from sentence_transformers import SentenceTransformer
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ dependencias y modelos locales OK")
sys.exit(1 if fail else 0)
PY

echo "██ PREFLIGHT 3: tests (sin los known-red)"
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  python -m pytest ../tests -q \
    -k "not test_backend_bertopic_produce_topicos and not test_parsimonia_elige_menor_k" \
    || { echo "❌ tests"; exit 5; }
fi

echo "██ CORRIDA REAL (una sola fase)"
python -u run_analysis.py \
  --csv-dmo ../data/processed/df_dmo.csv \
  --csv-ugc ../data/processed/df_ugc.csv \
  --text-col text --lang es \
  --outlier-method IF --contamination auto --min-partition 50 \
  --model bertopic --num-topicos automatico \
  --palette ejecutiva --seed 42 \
  --output-dir ../results \
  --comparison-tasks ${TASKS:-base topics lexical thematic price} \
  --bootstrap ${BOOTSTRAP:-2000} --permutations ${PERMUTATIONS:-1000} \
  --k-max ${KMAX:-30} --stability-runs ${STAB:-5}
rc=$?

echo "══ Fin $(date) | rc=$rc"
find ../results/comparison -type f -not -path "*_cache*" 2>/dev/null | sort
[[ $rc -eq 0 ]] && echo "✅ COMPLETADO" || echo "❌ ERROR rc=$rc (la caché conserva el progreso)"
exit $rc
