#!/usr/bin/env bash
# Wysyłka pętli 4.3 na Eden (bez zgłaszania zadań): kod, plik wsadowy, pusty tekst SD; końce linii LF; sprawdzenie
# składni i importów na węźle logowania. Uruchamiać z Git Bash: bash eden/wyslij43.sh
set -euo pipefail
H=eden-cluster
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-10-01_petla43_sd_prowadzenie
V1=$B/smallvit_runs/2026-09-28_maly_vit_od_zera
SD=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
TU="$(cd "$(dirname "$0")/.." && pwd)"
SDM=/f/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele
cicho() { grep -v -e '^\*' -e 'monitor' -e 'law enforcement' -e 'authority' -e 'system personnel' -e 'Warsaw University' \
  -e 'Faculty of Mathematics' -e 'recorded' -e 'maintenance' -e 'evidence' -e '^$' || true; }
ssh -o BatchMode=yes $H "mkdir -p $MY/kod $MY/eden $MY/logi $MY/wyniki $MY/wejscia" 2>&1 | cicho
scp -o BatchMode=yes -q "$TU"/kod/*.py $H:$MY/kod/ 2>&1 | cicho
scp -o BatchMode=yes -q "$TU"/eden/petla43_eden.slurm $H:$MY/eden/ 2>&1 | cicho
scp -o BatchMode=yes -q "$SDM/pusty_tekst.pt" $H:$MY/wejscia/ 2>&1 | cicho
ssh -o BatchMode=yes $H "set -e; sed -i 's/\r\$//' $MY/kod/*.py $MY/eden/*.slurm
  cd $MY/kod && $PY -m py_compile *.py && echo 'składnia OK'
  PYTHONPATH=$V1/deps:$SD/deps $PY -c 'import diffusers, timm, torch; print(\"diffusers\", diffusers.__version__, \"timm\", timm.__version__, \"torch\", torch.__version__)'
  ls -la $MY/kod | wc -l; ls -la $MY/wejscia; sha256sum $MY/wejscia/pusty_tekst.pt" 2>&1 | cicho
sha256sum "$SDM/pusty_tekst.pt"
