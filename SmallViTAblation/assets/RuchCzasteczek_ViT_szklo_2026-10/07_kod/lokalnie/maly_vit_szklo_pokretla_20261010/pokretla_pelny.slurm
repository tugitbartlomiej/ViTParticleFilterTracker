#!/bin/bash
#SBATCH --job-name=pokretla
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Pokrętła szkła, pełny cykl: trening (trening_pokretla.py) → analiza okien (pokretla_analiza.py) → kasowanie surowych gesty (oszczędność kwoty;
# ANALIZA*.json, SONDY.json, BAZA_PCA32.npz, PACZKI.npz, dane/, pelny/stan_e*.pt zostają). Argumenty jak pokretla.slurm:
#   sbatch [--job-name=lane1 --dependency=singleton] [--export=ALL,ANALIZY=300:2188,ZOSTAW=0,WARSTWY=0] pokretla_pelny.slurm \
#          <nazwa> <lr> <epoki> <manifest> <sondy_plik|-> <sondy_n> [dodatkowe argumenty trening_pokretla.py]
#   ANALIZY: okna od:do oddzielone przecinkami ('-' = bez analizy, wtedy gesty zostają); okno 300:2188 → ANALIZA.json, inne → ANALIZA_<od>_<do>.json
#   WARSTWY=1: dodatkowo analiza każdego gesty/w<i> → ANALIZA[_od_do]_w<i>.json;  ZOSTAW=1: nie kasuj gesty po analizie
#   --job-name=laneK --dependency=singleton: najwyżej jeden bieg na tor naraz (ogranicza chwilowe zużycie kwoty).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; LR=${2:?lr}; EPOKI=${3:?epoki}; MANIFEST=${4:?manifest}; SONDY=${5:--}; SN=${6:-24}; shift 6 || shift $#
ANALIZY=${ANALIZY:-300:2188}; ZOSTAW=${ZOSTAW:-0}; WARSTWY=${WARSTWY:-0}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/gesty/BAZA_PCA32.npz" ] && { echo "ISTNIEJE $OUT/gesty - nie nadpisuję"; exit 4; }
for f in "$E35/deps/timm/__init__.py" "$E35/kod/trening.py" "$E35/kod/dane.py" "$E35/kod/obserwator_gesty.py" "$EXP/kod/trening_pokretla.py" \
         "$EXP/kod/obserwator_rp.py" "$EXP/kod/model_cnn.py" "$EXP/kod/pokretla_analiza.py" "$EXP/kod/dh.py" "$EXP/kod/odciski.py" \
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
echo "pokrętła $NAZWA: lr $LR, epoki $EPOKI, manifest $(basename "$MANIFEST"), sondy $SONDY n=$SN, extra: $*, analizy $ANALIZY, zostaw $ZOSTAW, warstwy $WARSTWY, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"        # klatki walidacji (ten sam podział co bieg 35 000)
SARG=""; [ "$SONDY" != "-" ] && SARG="--sondy $SONDY"
cd "$EXP/kod"
"$PY" -u -B trening_pokretla.py --wyniki "$OUT" --zipy "$ZIPY" --manifest "$MANIFEST" --epoki "$EPOKI" --lr "$LR" \
      --obserwuj $SARG --sondy_n "$SN" --pelne 1 --bez_wag "$@"
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
          echo "=== analiza $NAZWA okno $od $do_ warstwa $wn $(date)"
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
