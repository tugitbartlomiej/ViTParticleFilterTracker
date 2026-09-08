#!/bin/bash
# =============================================================================
# Zglasza LANCUCH zadan treningu DETR from-scratch na partycji `long`.
#
# Mechanizm: kazde kolejne ogniwo dostaje --dependency=afterany:<poprzednie>,
# wiec startuje dopiero gdy poprzednie sie zakonczy (ukonczenie / timeout / blad).
# Kazde ogniwo wznawia z ostatniego checkpointu (--resume_training w slurm).
# Ogniwo 1 startuje od zera (pusty katalog -> init COCO, start_epoch=0).
# Jesli trening juz ukonczony, ogniwo konczy sie natychmiast (GUARD w slurm).
#
# Uzycie:
#   ./submit_detr_chain.sh           # 3 ogniwa (domyslnie), v1, 0->400
#   ./submit_detr_chain.sh 2         # 2 ogniwa
#   VARIANT=v1 EPOCHS=400 ./submit_detr_chain.sh 3
#
# Zatrzymanie calego lancucha:  scancel <wszystkie wypisane jobid> (lub: scancel -n detr_20k_scratch)
# Podglad:                      squeue -u $USER -o '%.10i %.12P %.20j %.10T %.12M %R'
# Log ogniwa:                   ~/DETR/logs/detr_20k_scratch_<jobid>.log
# =============================================================================
set -euo pipefail

SLURM_FILE="${SLURM_FILE:-run_detr_20k_scratch_chain.slurm}"
N="${1:-3}"

if [[ ! -f "$SLURM_FILE" ]]; then
    echo "ERROR: nie znaleziono $SLURM_FILE (uruchom z katalogu, w ktorym lezy skrypt)"; exit 1
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "ERROR: liczba ogniw musi byc dodatnia (podano: '$N')"; exit 1
fi

echo "Zglaszam lancuch $N ogniw | slurm=$SLURM_FILE | VARIANT=${VARIANT:-v1} EPOCHS=${EPOCHS:-400}"
echo "-----------------------------------------------------------------"

JIDS=()
jid=$(sbatch --parsable "$SLURM_FILE")
JIDS+=("$jid")
echo "  ogniwo 1: job $jid  (start od zera / resume jesli sa checkpointy)"

for ((i=2; i<=N; i++)); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$SLURM_FILE")
    JIDS+=("$jid")
    echo "  ogniwo $i: job $jid  (afterany:${JIDS[$((i-2))]})"
done

echo "-----------------------------------------------------------------"
echo "Lancuch zgloszony. Job IDs: ${JIDS[*]}"
echo "Podglad:    squeue -u \$USER -o '%.10i %.12P %.20j %.10T %.12M %R'"
echo "Zatrzymanie: scancel ${JIDS[*]}"
