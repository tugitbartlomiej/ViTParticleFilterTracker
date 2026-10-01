#!/usr/bin/env bash
# Pobranie wyników kampanii 4.3 z Edenu: tylko pliki JSON i obrazy kotwic (bez wag i list wyszukiwania), do
# wyniki_eden43/. Uruchamiać z Git Bash: bash eden/pobierz43.sh
set -euo pipefail
MY=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie
TU="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$TU/wyniki_eden43"
ssh -o BatchMode=yes eden-cluster "cd $MY/wyniki && find . -path '*petla43_s*' \( -name '*.json' -o -name 'KOTWICE_I_OBRAZY.jpg' \) -not -path '*zapytania*' | tar -cf - -T -" 2>/dev/null \
  | tar -xf - -C "$TU/wyniki_eden43"
scp -o BatchMode=yes -q "eden-cluster:$MY/logi/petla43_*.out" "$TU/wyniki_eden43/" 2>/dev/null || true
find "$TU/wyniki_eden43" -name KONIEC.json | wc -l
