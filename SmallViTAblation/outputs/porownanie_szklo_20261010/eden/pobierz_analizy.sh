#!/bin/bash
# Pobiera wszystkie ANALIZA*.json i JADRO_*.json biegów z pokrętłami (Eden: 2026-10-10_szklo_pokretla/wyniki/<bieg>/) jednym strumieniem tar
# do eden/wyniki/pokretla/ANALIZA_<bieg>[_od_do][_wN].json i eden/wyniki/pokretla/JADRO_<bieg>_<stan>.json (nadpisuje: nowe pliki są nadzbiorem).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/wyniki
TMP="$(mktemp -d)"
ssh -o BatchMode=yes eden-cluster "cd '$EXP' && tar cf - */ANALIZA*.json */JADRO_*.json 2>/dev/null" | tar xf - -C "$TMP"
n=0
for f in "$TMP"/*/*.json; do
  bieg=$(basename "$(dirname "$f")"); p=$(basename "$f" .json)
  case "$p" in
    ANALIZA) cel="ANALIZA_${bieg}.json";;
    ANALIZA_*) cel="ANALIZA_${bieg}${p#ANALIZA}.json";;
    JADRO_*) cel="JADRO_${bieg}_${p#JADRO_}.json";;
    *) continue;;
  esac
  cp "$f" "$HERE/wyniki/pokretla/$cel"; n=$((n + 1))
done
rm -rf "$TMP"
echo "pobrano $n plików do $HERE/wyniki/pokretla"
