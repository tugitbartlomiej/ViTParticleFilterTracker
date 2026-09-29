#!/bin/bash
# Wysyłka kodu, danych i wektorów tożsamości na Eden (dyfuzja_narzedzie/PLAN.md). NIE zgłasza żadnego zadania.
# Uruchomienie z Git Bash:  bash eden/wyslij.sh
set -euo pipefail
H=eden-cluster
TU=$(cd "$(dirname "$0")/.." && pwd)
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie
WEK="$TU/../maly_vit_tozsamosc_tla/wyniki/WEKTORY.npz"
D="$TU/dane/filmy3"
for f in "$D/okna.bin" "$D/okna_indeks.npy" "$D/okna.json" "$D/SHA256SUMS" "$D/MANIFEST.json" "$WEK"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
( cd "$D" && sha256sum -c SHA256SUMS )
echo "== kwota projektu przed wysyłką"
ssh "$H" "lfs quota -p 1299 /mnt/evafs | tail -3"
ssh "$H" "mkdir -p $R/kod $R/dane/filmy3 $R/eden $R/logi $R/wyniki $R/wejscia"
scp "$TU"/kod/{model_dyfuzji.py,trenuj.py,generuj.py,vit_model.py,przygotuj_filmy.py} "$H:$R/kod/"
scp "$TU"/eden/{trenuj_eden.slurm,generuj_eden.slurm} "$H:$R/eden/"
scp "$WEK" "$H:$R/wejscia/WEKTORY.npz"
scp "$D"/{okna_indeks.npy,okna.json,SHA256SUMS,MANIFEST.json,PODGLAD.jpg} "$H:$R/dane/filmy3/"
scp "$D/okna.bin" "$H:$R/dane/filmy3/"
ssh "$H" "sed -i 's/\r\$//' $R/eden/*.slurm $R/kod/*.py && cd $R/dane/filmy3 && sha256sum -c SHA256SUMS && ls -la $R/kod $R/eden $R/dane/filmy3"
echo "== kwota projektu po wysyłce"
ssh "$H" "lfs quota -p 1299 /mnt/evafs | tail -3"
echo "Gotowe. Zgłoszenie dopiero po zgodzie użytkownika:"
echo "  ssh $H 'cd $R/eden && sbatch trenuj_eden.slurm 500 64 proba'"
echo "  ssh $H 'cd $R/eden && sbatch trenuj_eden.slurm 100000 64'"
