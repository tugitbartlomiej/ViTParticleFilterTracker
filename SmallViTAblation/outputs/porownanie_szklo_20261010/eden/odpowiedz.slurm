#!/bin/bash
#SBATCH --job-name=odpowiedz
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Funkcja odpowiedzi G(r): pchnięcie jednego tokenu i odpowiedź pozostałych (kod/odpowiedz_tokenow.py), bieg 35 000 po epokach + kontrole.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
EXP35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
SCRATCH="/dev/shm/${USER}_odp_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/wyniki/odpowiedz"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$EXP35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" PYTHONPATH="$EXP35/deps"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "start $(date), węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
cp "$EXP/kod/odpowiedz_tokenow.py" "$EXP35/kod/odpowiedz_tokenow.py"
cd "$EXP35/kod"
WG="$EXP35/wyniki/pelny/wagi"
MOD="start=$WG/k00000.pt,ep1=$WG/k01094.pt,ep3=$WG/k03282.pt,ep10=$WG/k10940.pt,ep20=$WG/k21880.pt,ep40=$WG/k43760.pt,ep60=$WG/k65640.pt"
MOD="$MOD,bezpoz_start=$POK/bezpoz_lr_5e-4_rp/wagi/k00000.pt:bezpoz,bezpoz_ep3=$POK/bezpoz_lr_5e-4_rp/wagi/k03282.pt:bezpoz"
MOD="$MOD,z48_ep3=$POK/lr_5e-4_rp_z48_zostaw/wagi/k03282.pt,z49_ep3=$POK/lr_5e-4_rp_z49_zostaw/wagi/k03282.pt,lr2e-3_ep3=$POK/lr_2e-3_rp_zostaw/wagi/k03282.pt"
for p in $(echo "$MOD" | tr ',' '\n' | cut -d= -f2 | sed 's/:bezpoz//'); do [ -f "$p" ] || { echo "BRAK $p"; exit 2; }; done
"$PY" -u -B odpowiedz_tokenow.py --manifest "$EXP35/wejscia/MANIFEST_35000.jsonl" --zipy "$ZIPY" --out "$EXP/wyniki/odpowiedz" --modele "$MOD"
echo "KONIEC $(date)"
