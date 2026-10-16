#!/bin/bash
#SBATCH --job-name=koop_lok
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_koopman_pulapki/logi/%x_%j.out
# Lokalny Koopman na pułapkach (kod/koopman_lokalny.py): okna przesuwne, krótki horyzont, wejścia z PACZKI.npz (zdjęcie w paczce,
# norma kroku AdamW, spadek straty, lr), gęsty zapis co krok (12 sond) i skala epok (ruch60, 20 zdjęć).
#   sbatch koop_lok.slurm <nazwa> [tylko_ruch60]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-13_koopman_pulapki"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
cd "$EXP/kod"
echo "koop_lok $NAZWA, węzeł $(hostname), start $(date)"
GESTY=("$POK/lr_5e-4_rp/gesty:300:2188" "$POK/dlugi_lr_5e-4_rp/gesty:12034:13128")
if [ "${2:-}" = tylko_ruch60 ]; then GESTY=(); fi
"$PY" -u -B koopman_lokalny.py --gesty "${GESTY[@]}" \
      --ruch60 "$B/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/ruch60_rp128_50tr_50val" --n_ruch60 20 --out "$EXP/wyniki/$NAZWA"
echo "KONIEC $NAZWA $(date)"
