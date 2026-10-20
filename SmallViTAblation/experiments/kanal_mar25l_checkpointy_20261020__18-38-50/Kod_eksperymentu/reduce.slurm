#!/bin/bash
# Redukcja CPU po zakończeniu wszystkich 24 seedów.
set -euo pipefail
exp="$1"
python="$2"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
cd "$exp/Kod_eksperymentu"
"$python" -B reduce_kanal.py
