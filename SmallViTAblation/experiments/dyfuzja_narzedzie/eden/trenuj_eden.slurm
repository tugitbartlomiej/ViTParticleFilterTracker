#!/bin/bash
#SBATCH --job-name=dyfuzja_nar
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie/logi/%x_%j.out
# Uczenie modelu dyfuzyjnego wycinków narzędzia (dyfuzja_narzedzie/PLAN.md). Wznawialne: każde zgłoszenie z tym samym
# trybem kontynuuje od ostatni.pt, aż do KROKI kroków. Przy SIGTERM (koniec czasu) skrypt tworzy STOP, Python zapisuje
# ostatni.pt i kończy.
#   próba (500 kroków, osobny katalog):   sbatch trenuj_eden.slurm 500 64 proba
#   fragment uczenia:                      sbatch trenuj_eden.slurm 100000 64
#   kolejny fragment po poprzednim:        sbatch --dependency=afterany:<JOB> trenuj_eden.slurm 100000 64
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
KROKI=${1:?liczba kroków}; BATCH=${2:-64}; TRYB=${3:-pelny}
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-29_dyfuzja_narzedzie"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$MY/wyniki/trening"; ARG=""
[ "$TRYB" = proba ] && { OUT="$MY/wyniki/proba"; ARG="--podglad-co 500 --zapis-co 500 --ostatni-co 250"; }
DANE="$MY/dane/filmy3"
for f in "$MY/kod/trenuj.py" "$MY/kod/model_dyfuzji.py" "$DANE/okna.bin" "$DANE/okna_indeks.npy" "$DANE/okna.json" "$DANE/SHA256SUMS"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
( cd "$DANE" && sha256sum -c SHA256SUMS ) || { echo "złe sumy danych"; exit 3; }
SCRATCH=$(mktemp -d "/dev/shm/dyfuzja-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/dyfuzja-*) rm -rf -- "$SCRATCH";; esac; }
zatrzymaj() { mkdir -p "$OUT"; touch "$OUT/STOP"; }
trap cleanup EXIT
trap zatrzymaj TERM INT
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$SCRATCH" TORCH_HOME="$SCRATCH/torch" MPLCONFIGDIR="$SCRATCH/mpl" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/GPU_${SLURM_JOB_ID}.txt"
echo "start $(date), węzeł $(hostname), GPU $(head -1 "$OUT/GPU_${SLURM_JOB_ID}.txt"), kroki $KROKI, batch $BATCH, tryb $TRYB"
cd "$MY/kod"
"$PY" -u -B trenuj.py --dane "$DANE" --out "$OUT" --kroki "$KROKI" --batch "$BATCH" --workery 6 $ARG &
child=$!
set +e
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done     # sygnał przerywa wait w trakcie zapisu
set -e
echo "koniec $(date), kod $rc"
exit "$rc"
