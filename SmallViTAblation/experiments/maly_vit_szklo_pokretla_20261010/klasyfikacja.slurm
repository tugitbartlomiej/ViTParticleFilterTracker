#!/bin/bash
#SBATCH --job-name=klas
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Drugi zbiór (Imagenette, klasyfikacja) pełny cykl: trening_klasyfikacja.py → pokretla_analiza.py → kasowanie surowych gesty.
#   sbatch [--job-name=laneK --dependency=singleton] [--export=ALL,ANALIZY=300:2188,ZOSTAW=0,WARSTWY=0] klasyfikacja.slurm \
#          <nazwa> <lr> <epoki> <sondy_n> [dodatkowe argumenty trening_klasyfikacja.py]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; LR=${2:?lr}; EPOKI=${3:?epoki}; SN=${4:-24}; shift 4 || shift $#
ANALIZY=${ANALIZY:-300:2188}; ZOSTAW=${ZOSTAW:-0}; WARSTWY=${WARSTWY:-0}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
DANE="$B/smallvit_data/imagenette/imagenette2-320"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/gesty/BAZA_PCA32.npz" ] && { echo "ISTNIEJE $OUT/gesty - nie nadpisuję"; exit 4; }
for f in "$E35/deps/timm/__init__.py" "$E35/kod/trening.py" "$E35/kod/model.py" "$E35/kod/obserwator_gesty.py" "$EXP/kod/trening_klasyfikacja.py" \
         "$EXP/kod/obserwator_rp.py" "$EXP/kod/model_cnn.py" "$EXP/kod/pokretla_analiza.py" "$DANE/train" "$DANE/val"; do
  [ -e "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH="/dev/shm/${USER}_klas_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/tmp" "$EXP/logi" "$OUT"
trap 'rm -rf -- "$SCRATCH"; du -sh "$OUT" 2>/dev/null; lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH" MALY_VIT_CACHE="$SCRATCH"
export PYTHONPATH="$E35/deps:$E35/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "klasyfikacja $NAZWA: lr $LR, epoki $EPOKI, sondy n=$SN, extra: $*, analizy $ANALIZY, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
cd "$EXP/kod"
"$PY" -u -B trening_klasyfikacja.py --wyniki "$OUT" --dane "$DANE" --epoki "$EPOKI" --lr "$LR" --obserwuj --sondy_n "$SN" --pelne 1 --rzut rp "$@"
echo "TRENING KONIEC $NAZWA $(date)"
ok=1
if [ "$ANALIZY" != "-" ]; then
  grep -q '"koniec"' "$OUT/dane/metadata.json" || { echo "bieg nieukończony - bez analizy"; ok=0; }
  if [ "$ok" = 1 ]; then
    for okno in ${ANALIZY//,/ }; do
      od=${okno%%:*}; do_=${okno#*:}
      if [ "$okno" = "300:2188" ]; then wy="$OUT/ANALIZA.json"; else wy="$OUT/ANALIZA_${od}_${do_}.json"; fi
      echo "=== analiza $NAZWA okno $od $do_ $(date)"
      "$PY" -u -B pokretla_analiza.py "$OUT" "$wy" "$od" "$do_" || { echo "ANALIZA $NAZWA $okno NIEUDANA"; ok=0; }
      if [ "$WARSTWY" = 1 ]; then
        for w in "$OUT"/gesty/w*/; do
          [ -d "$w" ] || continue
          wn=$(basename "$w"); wyw="${wy%.json}_${wn}.json"
          "$PY" -u -B pokretla_analiza.py "$OUT" "$wyw" "$od" "$do_" "gesty/$wn" || { echo "ANALIZA $NAZWA $okno $wn NIEUDANA"; ok=0; }
        done
      fi
    done
  fi
  if [ "$ok" = 1 ] && [ "$ZOSTAW" != 1 ]; then
    rm -f "$OUT"/gesty/pca32_*.npz "$OUT"/gesty/pelne768_*.npz "$OUT"/gesty/w*/pca32_*.npz
    echo "skasowano surowe gesty $NAZWA (analizy OK)"
  fi
fi
echo "KONIEC $NAZWA ok=$ok $(date)"
