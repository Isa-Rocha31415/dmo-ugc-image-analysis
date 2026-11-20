#!/bin/bash
#SBATCH --job-name=dmo_ugc
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=salida_%x_%j.out
#SBATCH --error=error_%x_%j.err
### --gres=gpu:1        <-- descomenta SOLO si vas a procesar un dataset nuevo (sin caché)
###SBATCH --mail-type=END,FAIL
###SBATCH --mail-user=tucorreo@cimat.mx

# NO usamos 'set -e': capturamos códigos de salida manualmente para reportarlos bien.
set -uo pipefail

# ══════════════════════════════════════════════════════════════
# 0. ENTORNO
# ══════════════════════════════════════════════════════════════
source /opt/anaconda3_titan/etc/profile.d/conda.sh
conda activate nlp_rodri

# El cwd DEBE ser ~/ac_4/code: las rutas ../data y las claves de caché dependen de él.
cd "${SLURM_SUBMIT_DIR:-$HOME/ac_4/code}"
[[ -f run_analysis.py ]] || cd "$HOME/ac_4/code"
if [[ ! -f run_analysis.py ]]; then
  echo "❌ No se encuentra run_analysis.py. Envía el job desde ~/ac_4/code."
  exit 2
fi

# Offline estricto (regla del proyecto: prohibido descargar modelos/paquetes)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Hilos alineados con --cpus-per-task (evita sobre-suscripción de BLAS/numba)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "════════════════════════════════════════════════════════════"
echo " Fecha y hora : $(date)"
echo " Nodo         : $HOSTNAME"
echo " Job ID       : ${SLURM_JOB_ID:-local}"
echo " CPUs         : ${SLURM_CPUS_PER_TASK:-?}"
echo " CWD          : $(pwd)"
echo " Python       : $(which python)"
python --version
echo "════════════════════════════════════════════════════════════"
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sin GPU'))"
command -v nvidia-smi >/dev/null && nvidia-smi || true

# ══════════════════════════════════════════════════════════════
# 1. PREFLIGHT: sintaxis + imports + versiones  (FATAL si falla)
# ══════════════════════════════════════════════════════════════
echo ""
echo "██ PREFLIGHT 1/3: compilación sintáctica"
python -m py_compile \
  run_analysis.py joint_topic_model.py metrics_lexical.py metrics_thematic.py \
  comparison/common.py comparison/base_builder.py
rc=$?
if [[ $rc -ne 0 ]]; then echo "❌ SINTAXIS INVÁLIDA (rc=$rc). Aborto."; exit 3; fi
echo "✅ SINTAXIS OK"

echo ""
echo "██ PREFLIGHT 2/3: dependencias instaladas (sin descargas)"
python - <<'PY'
import importlib, sys
req = ["numpy","pandas","scipy","sklearn","plotly","bertopic","hdbscan","umap",
       "torch","transformers","sentence_transformers"]
fail = []
for m in req:
    try:
        mod = importlib.import_module(m)
        print(f"  ✅ {m:<22s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  ❌ {m:<22s} {type(e).__name__}: {e}")
        fail.append(m)
try:
    import plotly
    if int(plotly.__version__.split(".")[0]) < 5:
        print("  ❌ plotly>=5 requerido para add_vline(row,col) en curvas_k.html")
        fail.append("plotly>=5")
except Exception:
    pass
sys.exit(1 if fail else 0)
PY
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "❌ Faltan dependencias en el entorno nlp_rodri. Aborto (no se instala nada: offline)."
  exit 4
fi

python -c "import comparison.common, comparison.base_builder, joint_topic_model, metrics_lexical, metrics_thematic; print('✅ IMPORTS PROPIOS OK')"
rc=$?
if [[ $rc -ne 0 ]]; then echo "❌ Imports propios fallaron (rc=$rc). Aborto."; exit 4; fi

echo ""
echo "██ PREFLIGHT 3/3: tests unitarios"
if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
  echo "  ⏭️  SKIP_TESTS=1: tests omitidos"
elif python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest ../tests/test_joint_topic_model.py \
                   ../tests/test_metrics_thematic.py \
                   ../tests/test_metrics_lexical.py -q
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "❌ Tests fallaron (rc=$rc). Aborto antes de gastar horas GPU."; exit 5; fi
  echo "✅ TESTS OK"
else
  echo "  ⚠️ pytest no disponible; se omiten los tests"
fi

# ══════════════════════════════════════════════════════════════
# 2. ARGUMENTOS COMUNES (idénticos en ambas fases)
# ══════════════════════════════════════════════════════════════
COMMON_ARGS=(
  
)

# ══════════════════════════════════════════════════════════════
# 3. FASE A — SMOKE TEST (base + topics + thematic, B=300)
# ══════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████"
echo "  FASE A: validación rápida (base, topics, thematic | B=300)"
echo "████████████████████████████████████████████████████████████"
python -u run_analysis.py "${COMMON_ARGS[@]}" \
  --comparison-tasks base topics thematic \
  --bootstrap 300
rc_smoke=$?
echo ">>> FASE A terminó con código $rc_smoke"
if [[ $rc_smoke -ne 0 ]]; then
  echo "❌ La validación falló; NO se lanza la corrida completa."
  echo "Fin: $(date)"
  exit $rc_smoke
fi

# ══════════════════════════════════════════════════════════════
# 4. FASE B — CORRIDA COMPLETA (todas las tareas, B=2000)
#    RUN_FULL=0 para ejecutar solo la validación.
# ══════════════════════════════════════════════════════════════
rc_full=0
if [[ "${RUN_FULL:-1}" == "1" ]]; then
  echo ""
  echo "████████████████████████████████████████████████████████████"
  echo "  FASE B: análisis completo (todas las tareas | B=2000)"
  echo "████████████████████████████████████████████████████████████"
  python -u run_analysis.py "${COMMON_ARGS[@]}" \
    --bootstrap 2000 --permutations 1000
  rc_full=$?
  echo ">>> FASE B terminó con código $rc_full"
else
  echo "  ⏭️  RUN_FULL=0: solo se ejecutó la validación."
fi

# ══════════════════════════════════════════════════════════════
# 5. RESUMEN
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  RESUMEN"
echo "  Fase A (smoke) : $rc_smoke"
echo "  Fase B (full)  : $rc_full"
echo "  Fin            : $(date)"
echo "════════════════════════════════════════════════════════════"
echo "📁 Archivos generados en ../results/comparison/:"
find ../results/comparison -type f -not -path "*_cache*" | sort

if [[ $rc_smoke -ne 0 || $rc_full -ne 0 ]]; then
  echo "❌ JOB CON ERRORES"
  exit 1
fi
echo "✅ JOB COMPLETADO SIN ERRORES"
exit 0