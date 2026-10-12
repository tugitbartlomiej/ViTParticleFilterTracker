#!/bin/bash
# Pobiera wyniki biegu 35 000 klatek z Edena do assets/MalyViT_35000_od_zera_s47 (wagi/, dane/), z kontrolą SHA-256.
#   bash pobierz_z_edena.sh epoki     # k00000 + końce wszystkich epok (~12 MB na epokę) + dane/*.json
#   bash pobierz_z_edena.sh co100     # dodatkowo co 100 kroków (~250 plików, ~2,9 GB)
#   bash pobierz_z_edena.sh wszystko  # wszystkie zapisane wagi (~331 plików, ~3,9 GB)
# Pliki ze zgodnym SHA-256 są pomijane, kopia idzie do .part i dostaje nazwę dopiero po zgodności sumy.
# Nic nie jest zapisywane ani kasowane po stronie Edena.
set -euo pipefail
TRYB=${1:-epoki}
case "$TRYB" in epoki|co100|wszystko) ;; *) echo "tryb: epoki | co100 | wszystko"; exit 2;; esac
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/pelny
L="$(cd "$(dirname "$0")/../../assets/MalyViT_35000_od_zera_s47" && pwd)"
mkdir -p "$L/wagi" "$L/dane"
ssh -o BatchMode=yes eden-cluster "cd $R/wagi && sha256sum k*.pt" 2>/dev/null | grep -E '  k[0-9]+\.pt$' > "$L/wagi/.zdalne.sha256"
: > "$L/wagi/.lista"
while read -r h n; do
  k=$((10#${n:1:5}))
  if [ "$TRYB" = wszystko ] || [ $k -eq 0 ] || [ $((k % 1094)) -eq 0 ] || { [ "$TRYB" = co100 ] && [ $((k % 100)) -eq 0 ]; }; then
    echo "$h $n" >> "$L/wagi/.lista"
  fi
done < "$L/wagi/.zdalne.sha256"
echo "na Edenie: $(wc -l < "$L/wagi/.zdalne.sha256") plików wag, wybrane: $(wc -l < "$L/wagi/.lista")"
while read -r h n; do
  if [ -f "$L/wagi/$n" ] && [ "$(sha256sum "$L/wagi/$n" | cut -d' ' -f1)" = "$h" ]; then continue; fi
  scp -q -o BatchMode=yes "eden-cluster:$R/wagi/$n" "$L/wagi/$n.part"
  [ "$(sha256sum "$L/wagi/$n.part" | cut -d' ' -f1)" = "$h" ] || { echo "SHA niezgodne: $n"; exit 3; }
  mv "$L/wagi/$n.part" "$L/wagi/$n"; echo "ok $n"
done < "$L/wagi/.lista"
scp -q -o BatchMode=yes "eden-cluster:$R/dane/*.json" "$L/dane/"
echo "gotowe: $(ls "$L/wagi"/k*.pt | wc -l) plików wag w $L/wagi"
