#!/bin/bash
# Kopia z Edena wszystkiego, co w badaniu ruchu tokenów („szkło”) istnieje tylko na Edenie. Nic nie jest zmieniane
# ani kasowane po stronie Edena. Każda część idzie jednym strumieniem tar przez ssh (MSYS psuje listy ścieżek /home2
# w argumentach scp), potem kontrola SHA-256 z sum policzonych na Edenie (SHA256SUMS_EDEN_<część>.txt).
#   bash pobierz_z_edena.sh            # wszystkie części
#   bash pobierz_z_edena.sh pokretla   # jedna część
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R='~/smallvit_runs'
LOG="$HERE/pobieranie_z_edena.log"

czesc() {   # $1 nazwa, $2 zdalny katalog bazowy, $3 lista podkatalogów (w bazie), $4 lokalny katalog docelowy
  local nazwa="$1" baza="$2" co="$3" cel="$HERE/$4"
  mkdir -p "$cel"
  echo "[$(date +%T)] $nazwa: start ($baza: $co -> $4)" | tee -a "$LOG"
  ssh -o BatchMode=yes eden-cluster "cd $baza && find $co -type f -print0 | sort -z | xargs -0 sha256sum" 2>/dev/null \
    | grep -E '^[0-9a-f]{64}  ' > "$cel/SHA256SUMS_EDEN_$nazwa.txt"
  ssh -o BatchMode=yes eden-cluster "cd $baza && tar cf - $co" 2>/dev/null | tar xf - -C "$cel"
  ( cd "$cel" && sha256sum --quiet -c "SHA256SUMS_EDEN_$nazwa.txt" ) >> "$LOG" 2>&1 \
    && echo "[$(date +%T)] $nazwa: OK, $(wc -l < "$cel/SHA256SUMS_EDEN_$nazwa.txt") plików zgodnych z Edenem" | tee -a "$LOG" \
    || echo "[$(date +%T)] $nazwa: BŁĄD SUM, szczegóły w $LOG" | tee -a "$LOG"
}

T=${1:-wszystko}
[[ $T == wszystko || $T == pomiar60 ]] && czesc pomiar60 "$R/2026-10-10_szklo_koloidalne/wyniki" "pomiar_pelny60_3000tr_1000val" "03_pomiar_naturalny_60_epok_eden"
[[ $T == wszystko || $T == szklo_bd ]] && czesc szklo_bd "$R/2026-10-10_szklo_koloidalne" "wyniki/stala wyniki/stygnace wyniki/szybko_stala wyniki/szybko_stygnace wyniki/proba logi" "04_szklo_koloidalne_BD/eden_symulacje"
[[ $T == wszystko || $T == bieg35000 ]] && czesc bieg35000 "$R/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki" "gesty proba gesty_proba pelny/pelny pelny/dane" "01_bieg_35000_gesty_zapis/eden"
[[ $T == wszystko || $T == pokretla ]] && czesc pokretla "$R/2026-10-10_szklo_pokretla" "wyniki logi" "05_biegi_pokretla_eden"
[[ $T == wszystko || $T == ruch60 ]] && czesc ruch60 "$R/2026-10-10_szklo_koloidalne/wyniki" "\$(ls -d ruch60_* 2>/dev/null)" "03_pomiar_naturalny_60_epok_eden"
[[ $T == wszystko || $T == kod ]] && czesc kod "$R" "2026-10-10_szklo_pokretla/kod 2026-10-10_szklo_koloidalne/kod 2026-10-09_maly_vit_od_zera_35000_21epok/kod" "07_kod/eden"
echo "[$(date +%T)] koniec ($T)" | tee -a "$LOG"
