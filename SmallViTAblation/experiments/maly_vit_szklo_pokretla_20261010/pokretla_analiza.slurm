#!/bin/bash
#SBATCH --job-name=pokretla_an
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Analiza biegów z pokrętłami (odciski S0/S1, heterogeniczność, energia 768D) w zadanym gęstym oknie.
#   sbatch pokretla_analiza.slurm bieg[:od:do] ...     np. dlugi_lr_5e-4:12034:13128  okno60:65640:66734  lr_5e-4
#   bez argumentów: wszystkie ukończone biegi w oknie domyślnym 300–2188 -> wyniki/<bieg>/ANALIZA.json;
#   z oknem -> wyniki/<bieg>/ANALIZA_<od>_<do>.json
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/${USER}_mpl_$SLURM_JOB_ID
cd "$EXP/kod"
if [ $# -gt 0 ]; then BIEGI="$@"; else BIEGI=$(ls -d "$EXP"/wyniki/*/gesty/BAZA_PCA32.npz | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename | grep -v '^test$'); fi
for spec in $BIEGI; do
  b=${spec%%:*}; rest=${spec#"$b"}; rest=${rest#:}
  [ -f "$EXP/wyniki/$b/dane/metadata.json" ] && grep -q '"koniec"' "$EXP/wyniki/$b/dane/metadata.json" || { echo "pomijam $b (bieg nieukończony)"; continue; }
  if [ -n "$rest" ]; then od=${rest%%:*}; do_=${rest#*:}; out="$EXP/wyniki/$b/ANALIZA_${od}_${do_}.json"; args="$od $do_"; else out="$EXP/wyniki/$b/ANALIZA.json"; args=""; fi
  echo "=== $b okno ${args:-300 2188} $(date)"
  "$PY" -u -B pokretla_analiza.py "$EXP/wyniki/$b" "$out" $args || echo "ANALIZA $b NIEUDANA"
done
echo "KONIEC $(date)"
