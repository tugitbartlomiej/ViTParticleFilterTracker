#!/bin/bash
#SBATCH --job-name=pokretla
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Pokrętła szkła: ten sam przepis co bieg 35 000 (trening35000.py), zmienia się lr (temperatura szumu) albo liczba zdjęć
# (gęstość: zdjęć na parametr), gęsty obserwator na 24 sondach (co 1 krok do 2188, potem co 10), bez zapisu wag po drodze.
#   sbatch pokretla.slurm <nazwa> <lr> <epoki> <manifest> <sondy_plik|-> <sondy_n> [dodatkowe argumenty trening_pokretla.py]
#   sbatch pokretla.slurm lr_2.5e-4 2.5e-4 3 $E35/wejscia/MANIFEST_35000.jsonl $E35/wejscia/SONDY_100.jsonl 24
#   sbatch pokretla.slurm gestosc_1500 5e-4 70 $EXP/wejscia/MANIFEST_1500_z35k.jsonl - 24
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; LR=${2:?lr}; EPOKI=${3:?epoki}; MANIFEST=${4:?manifest}; SONDY=${5:--}; SN=${6:-24}; shift 6 || shift $#
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/gesty/BAZA_PCA32.npz" ] && { echo "ISTNIEJE $OUT/gesty - nie nadpisuję"; exit 4; }
for f in "$E35/deps/timm/__init__.py" "$E35/kod/trening.py" "$E35/kod/dane.py" "$E35/kod/obserwator_gesty.py" "$EXP/kod/trening_pokretla.py" \
         "$MANIFEST" "$ARCHIVE" "$ZIPY/TR_0001_S1_P03.zip" "$E35/wejscia/pretrain/valid/_annotations.coco.json"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH="/dev/shm/${USER}_pokretla_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/logi" "$OUT"
trap 'rm -rf -- "$SCRATCH"; du -sh "$OUT" 2>/dev/null; lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$E35/deps:$E35/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "pokrętła $NAZWA: lr $LR, epoki $EPOKI, manifest $(basename "$MANIFEST"), sondy $SONDY n=$SN, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"        # klatki walidacji (ten sam podział co bieg 35 000)
SARG=""; [ "$SONDY" != "-" ] && SARG="--sondy $SONDY"
cd "$EXP/kod"
"$PY" -u -B trening_pokretla.py --wyniki "$OUT" --zipy "$ZIPY" --manifest "$MANIFEST" --epoki "$EPOKI" --lr "$LR" \
      --obserwuj $SARG --sondy_n "$SN" --pelne 1 --bez_wag "$@"
echo "KONIEC $NAZWA $(date)"
