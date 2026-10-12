#!/bin/bash
# Pobiera gęsty zapis ruchu sond i sił (bieg „gesty”, zadanie 1826161) z Edena do wyniki/gesty, z kontrolą SHA-256.
# Pliki ze zgodną sumą są pomijane; kopia idzie do .part i dostaje nazwę po zgodności sumy. Nic nie zmienia na Edenie.
set -euo pipefail
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/gesty
L="$(cd "$(dirname "$0")" && pwd)/wyniki/gesty"
mkdir -p "$L/gesty" "$L/dane"
ssh -o BatchMode=yes eden-cluster "cd $R/gesty && sha256sum *" 2>/dev/null | grep -E '  [A-Za-z0-9_.]+$' > "$L/gesty/.zdalne.sha256"
echo "na Edenie: $(wc -l < "$L/gesty/.zdalne.sha256") plików"
while read -r h n; do
  if [ -f "$L/gesty/$n" ] && [ "$(sha256sum "$L/gesty/$n" | cut -d' ' -f1)" = "$h" ]; then continue; fi
  for proba in 1 2 3 4 5; do
    scp -q -o BatchMode=yes -o ConnectTimeout=30 "eden-cluster:$R/gesty/$n" "$L/gesty/$n.part" && break
    echo "ponawiam $n ($proba)"; sleep 20
  done
  [ "$(sha256sum "$L/gesty/$n.part" | cut -d' ' -f1)" = "$h" ] || { echo "SHA niezgodne: $n"; exit 3; }
  mv "$L/gesty/$n.part" "$L/gesty/$n"; echo "ok $n"
done < "$L/gesty/.zdalne.sha256"
scp -q -o BatchMode=yes "eden-cluster:$R/dane/*.json" "$L/dane/"
echo "gotowe: $(ls "$L/gesty" | grep -v '^\.' | wc -l) plików, $(du -sh "$L/gesty" | cut -f1)"
