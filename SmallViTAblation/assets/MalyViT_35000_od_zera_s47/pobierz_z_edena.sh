#!/bin/bash
# Wszystkie checkpointy biegu „35 000 klatek od zera, ziarno 47” z Edena do wagi/ (obok tego skryptu), z kontrolą
# SHA-256 względem sum policzonych na Edenie. Pliki ze zgodną sumą są pomijane; kopia idzie do .part i dostaje nazwę
# dopiero po zgodności sumy. Nic nie jest zapisywane ani kasowane po stronie Edena.
set -euo pipefail
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/pelny
L="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$L/wagi" "$L/dane"
ssh -o BatchMode=yes eden-cluster "cd $R/wagi && sha256sum k*.pt" 2>/dev/null | grep -E '  k[0-9]+\.pt$' > "$L/SHA256SUMS_EDEN.txt"
echo "na Edenie: $(wc -l < "$L/SHA256SUMS_EDEN.txt") plików wag"
while read -r h n; do
  if [ -f "$L/wagi/$n" ] && [ "$(sha256sum "$L/wagi/$n" | cut -d' ' -f1)" = "$h" ]; then continue; fi
  for proba in 1 2 3 4 5; do
    scp -q -o BatchMode=yes -o ConnectTimeout=30 "eden-cluster:$R/wagi/$n" "$L/wagi/$n.part" && break
    echo "ponawiam $n ($proba)"; sleep 20
  done
  [ "$(sha256sum "$L/wagi/$n.part" | cut -d' ' -f1)" = "$h" ] || { echo "SHA niezgodne: $n"; exit 3; }
  mv "$L/wagi/$n.part" "$L/wagi/$n"
done < "$L/SHA256SUMS_EDEN.txt"
scp -q -o BatchMode=yes "eden-cluster:$R/dane/*.json" "$L/dane/"
cd "$L/wagi" && sha256sum k*.pt | sed 's/ \*/  /' > "$L/SHA256SUMS_LOKALNE.txt"
diff <(sort -k2 "$L/SHA256SUMS_EDEN.txt") <(sort -k2 "$L/SHA256SUMS_LOKALNE.txt") && echo "ZGODNE: $(ls k*.pt | wc -l) plików, $(du -sh . | cut -f1)"
