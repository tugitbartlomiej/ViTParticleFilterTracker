#!/bin/bash
# Wysyłka zbioru Cataract-LMM 3_Object_Tracking na Eden (prośba użytkownika 2026-09-30: „wrzuć filmy z Cataract-LMM
# na mój dysk”). Najpierw 3 archiwa filmów MP4 (5,6 GB), potem 170 archiwów z klatkami JPEG i obrysami (42 GB).
# Plik, który na Edenie ma już ten sam rozmiar, jest pomijany; archiwa wysłane wcześniej do dyfuzja_narzedzie/filmy_zip
# są dowiązane twardo (bez ponownej wysyłki). Każdy plik idzie pod nazwą tymczasową i dopiero po wysłaniu dostaje
# właściwą nazwę. Na końcu sprawdzenie SHA256 po obu stronach. Niczego nie usuwa i nie zgłasza zadań.
#   bash wyslij_cataract_lmm.sh   (log i sumy: F:/Studia/PhD_projekt/VIT/eden_archiwum/wysylka_cataract_lmm/)
set -uo pipefail
H=eden-cluster
SRC="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Dataset/Cataract-LMM/3_Object_Tracking"
DST=/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking
STARE=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie/filmy_zip
LOG="F:/Studia/PhD_projekt/VIT/eden_archiwum/wysylka_cataract_lmm"
ROWNOLEGLE=3
cicho() { grep -v "^\*\|SSH server\|Faculty\|Individuals\|excess\|all of\|recorded\|monitoring\|authorized\|this system\|advised\|activity\|such mon\|^$" || true; }
echo "start $(date)"
echo "kwota przed: $(ssh -o BatchMode=yes $H 'lfs quota -p 1299 /mnt/evafs | tail -1' 2>/dev/null | cicho)"
ssh -o BatchMode=yes $H "mkdir -p $DST/videos $DST/annotations" 2>/dev/null
scp -q -o BatchMode=yes "$SRC/README.md" "$H:$DST/README.md" 2>/dev/null
# dowiązania do archiwów wysłanych wcześniej (tylko przy zgodnym rozmiarze)
for f in TR_0001_S1_P03.zip TR_0003_S1_P03.zip TR_0006_S1_P03.zip; do
  r=$(stat -c %s "$SRC/annotations/$f")
  ssh -o BatchMode=yes $H "[ -f $DST/annotations/$f ] || { [ \"\$(stat -c %s $STARE/$f 2>/dev/null)\" = $r ] && ln $STARE/$f $DST/annotations/$f && echo dowiązane $f; }" 2>/dev/null | cicho
done
# lista rozmiarów po stronie Edenu, żeby pominąć gotowe pliki
ssh -o BatchMode=yes $H "cd $DST && find videos annotations -maxdepth 1 -type f -printf '%s %p\n'" 2>/dev/null | cicho > "$LOG/zdalne.txt"
wyslij() {
  local rel=$1 roz
  roz=$(stat -c %s "$SRC/$rel")
  if grep -qx "$roz $rel" "$LOG/zdalne.txt"; then echo "pominięty $rel"; return 0; fi
  local t0=$(date +%s)
  for proba in 1 2 3; do
    if scp -q -o BatchMode=yes "$SRC/$rel" "$H:$DST/${rel%/*}/.czesc_${rel##*/}" 2>/dev/null &&
       ssh -o BatchMode=yes $H "[ \$(stat -c %s $DST/${rel%/*}/.czesc_${rel##*/}) = $roz ] && mv $DST/${rel%/*}/.czesc_${rel##*/} $DST/$rel" 2>/dev/null; then
      local dt=$(( $(date +%s) - t0 + 1 ))
      echo "wysłany $rel $((roz / 1000000)) MB w $dt s ($((roz / dt / 1000000)) MB/s)"
      return 0
    fi
    echo "ponawiam $rel (próba $proba)"
  done
  echo "BŁĄD $rel"
}
export -f wyslij cicho; export H SRC DST LOG
( cd "$SRC" && ls videos/*.zip ) | xargs -P $ROWNOLEGLE -I{} bash -c 'wyslij "$@"' _ {}
( cd "$SRC" && ls annotations/*.zip ) | xargs -P $ROWNOLEGLE -I{} bash -c 'wyslij "$@"' _ {}
echo "wysyłka skończona $(date); liczę sumy SHA256"
( cd "$SRC" && sha256sum README.md videos/*.zip annotations/*.zip ) > "$LOG/SHA256SUMS"
scp -q -o BatchMode=yes "$LOG/SHA256SUMS" "$H:$DST/SHA256SUMS" 2>/dev/null
ssh -o BatchMode=yes $H "cd $DST && sha256sum -c --quiet SHA256SUMS && echo SUMY_ZGODNE; ls videos | wc -l; ls annotations | wc -l; lfs quota -p 1299 /mnt/evafs | tail -1" 2>/dev/null | cicho
echo "koniec $(date)"
