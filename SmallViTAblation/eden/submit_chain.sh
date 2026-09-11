#!/bin/bash
# Lancuch zadan SLURM przez --dependency=afterany (Agent 2, eden/).
# Potrzebny, bo calosc (~174 biegi) moze nie zmiescic sie w limicie 5 dni
# jednego zadania. Kazde ogniwo bierze kolejke tam, gdzie zostawila ja
# poprzednie (kolejka jest wznawialna, resztki requeue-stale).
#
# Uzycie:
#   bash submit_chain.sh [N_OGNIW]
# Przykladowe zmienne srodowiskowe (te same co run_node.slurm):
#   PROJECT_DIR QUEUE_DIR DATA_ARCHIVE RESULTS_DIR
#
# Ogniwa startuja po sobie NAWET gdy poprzednie sie wywali (afterany) —
# to celowe: ubicie limitem czasu to normalna sytuacja, nie awaria.

set -eu

N_LINKS="${1:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
SLURM="$SCRIPT_DIR/run_node.slurm"

mkdir -p "$PROJECT_DIR/slurm_logs"

PREV=""
for ((i = 1; i <= N_LINKS; i++)); do
    DEP=()
    if [[ -n "$PREV" ]]; then
        DEP=(--dependency="afterany:$PREV")
    fi
    JOB=$(sbatch --parsable "${DEP[@]}" \
        --export="ALL,PROJECT_DIR=$PROJECT_DIR,QUEUE_DIR=${QUEUE_DIR:-$PROJECT_DIR/eden/qdir},DATA_ARCHIVE=${DATA_ARCHIVE:-},RESULTS_DIR=${RESULTS_DIR:-}" \
        --job-name="abla_chain${i}" \
        "$SLURM")
    echo "[submit_chain] ogniwo $i/$N_LINKS -> jobid $JOB ${PREV:+(po $PREV)}"
    PREV="$JOB"
done

echo "[submit_chain] lancuch $N_LINKS zadan wyslany, ostatni: $PREV"
echo "[submit_chain] podglad: squeue -u \$USER ; status kolejki:"
echo "  python3.11 ${PROJECT_DIR}/eden/queue.py status ${QUEUE_DIR:-$PROJECT_DIR/eden/qdir}"
