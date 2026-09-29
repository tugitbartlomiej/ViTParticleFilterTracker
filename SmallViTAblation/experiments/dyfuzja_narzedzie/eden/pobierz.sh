#!/bin/bash
# Pobranie wyuczonego modelu dyfuzyjnego z Edenu (dyfuzja_narzedzie/PLAN.md) z kontrolą sum SHA256.
# Uruchomienie z Git Bash:  bash eden/pobierz.sh [trening|proba] [--wszystkie]
#   domyślnie: model_ema.pt, ostatni.pt, STATUS.json, LOG.jsonl, podglądy; --wszystkie: także ckpt_*.pt
set -euo pipefail
H=eden-cluster
TU=$(cd "$(dirname "$0")/.." && pwd)
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie
KAT=${1:-trening}; WSZ=${2:-}
D="$TU/wyniki/eden_$KAT"; mkdir -p "$D"
PLIKI="STATUS.json LOG.jsonl podglad_*.png GPU_*.txt ostatni.pt"
ssh "$H" "test -f $R/wyniki/$KAT/model_ema.pt" && PLIKI="$PLIKI model_ema.pt"
[ "$WSZ" = --wszystkie ] && PLIKI="$PLIKI ckpt_*.pt"
ssh "$H" "cd $R/wyniki/$KAT && sha256sum \$(ls $PLIKI 2>/dev/null | grep '\.pt\$') > SHA256SUMS_POBRANIE && cat STATUS.json | head -6"
ssh "$H" "cd $R/wyniki/$KAT && tar -cf - SHA256SUMS_POBRANIE \$(ls $PLIKI 2>/dev/null)" | tar -xf - -C "$D"
( cd "$D" && sha256sum -c SHA256SUMS_POBRANIE )
echo "Pobrano do $D"
