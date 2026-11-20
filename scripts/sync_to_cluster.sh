#!/usr/bin/env bash
# Sincroniza código local → cluster. NO toca data/raw ni results/ del cluster.
# Uso: bash scripts/sync_to_cluster.sh --dry   (simular)
#      bash scripts/sync_to_cluster.sh         (real)
set -euo pipefail
SRC="$HOME/ac_4_local/"
DST="est_posgrado_isabel.rocha@148.207.185.31:/home/est_posgrado_isabel.rocha/ac_4/"
EXTRA=()
[[ "${1:-}" == "--dry" ]] && EXTRA+=(-n)
rsync -avz "${EXTRA[@]}" \
  --exclude='.git/' \
  --exclude='results/' \
  --exclude='results_smoke*/' \
  --exclude='data/raw/' \
  --exclude='data/final/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='logs/' \
  --exclude='*.out' --exclude='*.err' \
  -e 'ssh -p 2284' \
  "$SRC" "$DST"
