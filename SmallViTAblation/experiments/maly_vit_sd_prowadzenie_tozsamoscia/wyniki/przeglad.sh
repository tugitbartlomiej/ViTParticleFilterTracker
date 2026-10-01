#!/usr/bin/env bash
# przegląd ustawień prowadzenia na 2 kotwicach (start z kotwicy)
cd "$(dirname "$0")/.."
U='F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_sd_prowadzenie_tozsamoscia\uruchom_lokalnie.cmd'
for cfg in "0.5 0.15 2" "0.5 0.3 3" "0.7 0.3 3" "0.7 0.5 4"; do
  set -- $cfg
  cmd //c "$U test_prowadzenia.py --kotwic 2 --sila-startu $1 --eta $2 --iteracje $3" >> wyniki/przeglad.log 2>> wyniki/przeglad.err
done
echo KONIEC >> wyniki/przeglad.log
