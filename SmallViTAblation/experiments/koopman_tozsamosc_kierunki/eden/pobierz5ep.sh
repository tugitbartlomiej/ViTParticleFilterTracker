#!/bin/bash
# Pobiera z Edena wyniki kampanii 5 epok (zadanie kont5ep): raporty, krzywe metrics.csv, logi SLURM oraz wagi
# z konca kazdej epoki (model_0000{103,206,309,412,515}.pth -> wagi/seed{S}_{ramie}_ep{k}.pth). Pomija ramiona bez
# REPORT.json (jeszcze licza sie) i pliki juz pobrane. Uruchamiac z katalogu eksperymentu.
set -u
REMOTE=smallvit_runs/2026-09-25_kontynuacja_5epok
DST=wyniki/dobor_klatek/trening5ep
mkdir -p "$DST/raporty" "$DST/wagi" "$DST/logi"
gotowe=$(timeout 90 ssh -o ConnectTimeout=25 -o BatchMode=yes eden-cluster "cd $REMOTE/results 2>/dev/null && ls -d */ 2>/dev/null | while read d; do [ -f \"\$d/REPORT.json\" ] && echo \"\${d%/}\"; done" 2>/dev/null)
for n in $gotowe; do
  if [ ! -f "$DST/raporty/$n/REPORT.json" ]; then
    timeout 120 ssh -o BatchMode=yes eden-cluster "cd $REMOTE/results && tar -cf - $n/REPORT.json $n/INPUTS.json $n/trening/VALIDATION.json $n/trening/STATUS.json $n/trening/metrics.csv" 2>/dev/null | tar -xf - -C "$DST/raporty" && echo "raport $n"
  fi
  k=1
  for s in 103 206 309 412 515; do
    f="$DST/wagi/${n}_ep${k}.pth"
    if [ ! -s "$f" ]; then
      timeout 600 scp -q -o BatchMode=yes "eden-cluster:$REMOTE/results/$n/trening/model_0000${s}.pth" "$f.part" && mv "$f.part" "$f" && echo "wagi $n ep$k"
    fi
    k=$((k + 1))
  done
done
timeout 60 scp -q -o BatchMode=yes "eden-cluster:$REMOTE/slurm-*.out" "$DST/logi/" 2>/dev/null
echo "POBRANO: ramion z raportem $(ls "$DST/raporty" | wc -l), plikow wag $(ls "$DST/wagi"/*.pth 2>/dev/null | wc -l)"
