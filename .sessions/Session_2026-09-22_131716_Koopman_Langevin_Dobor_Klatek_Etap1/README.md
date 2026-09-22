# Session 2026-09-22 - Koopman i Langevin na latencie detektora narzedzi (projekt ViT), etap 1 doboru klatek

## Cel sesji
Przeniesc aparat diagnostyczny z projektu grantowego (rozszerzone rownanie Langevina, Koopman, czastki latentu)
na obrazy: latent detektora RF-DETR Nano na filmach z operacji zacmy (Cataract-LMM). Zmierzyc, czy latent
krystalizuje przestrzen stanow narzedzia, czy sygnal dynamiki poprawia dobor klatek do uczenia i co w Koopmanie
naprawde cos doklada. Praca merytoryczna zyje w repozytorium ViT, nie w projekcie grantowym.

## Typ
experiment

## Workerzy uzyci
Dwa podagenty literaturowe (Undermind plus Scite): (1) fizyka w latencie, Koopman na wideo i w JEPA, Koopman
i Langevin dla dynamiki uczenia, dobor danych, Mori-Zwanzig z danych; (2) autoenkodery Koopmana, Koopman
z nieobserwowanym wejsciem, Kalman w latencie, HAVOK. Wyniki wpisane do dokumentow 38 i 39 w repo ViT.

## Co zrobiono (skrot)
- Kopia zapasowa katalogu `DETR` z Edena uzupelniona o brakujace pliki (sumy kontrolne zgodne).
- Dwa zegary czastek latentu detektora: os budzetu (COCO, 500, 1500, 2500 klatek) i os epok (co 3 epoki).
  Polozenie, rozmiar i os narzedzia staja sie odczytywalne po uczeniu; predkosc nie z jednej klatki.
- Predkosc jako stan ukryty: odporny filtr Kalmana (rownanie Langevina z szumem pomiaru) daje korelacje 0,67.
- Etap 1 kontrolowanego doboru klatek: 9 ramion x 3 ziarna = 27 treningow na Edenie (A100 i H100).
  Reszta Langevina nie pomaga (placebo rowne prawdziwemu sygnalowi); pokrycie modow Koopmana najlepsze
  (+0,0068 AP nad pokryciem wygladu), ale jego wspolrzedne to tozsamosc operacji, nie stan narzedzia.
- Pilot "czego sieci brakuje": tylko 23 procent chybien malej sieci znika po 5x wiecej etykiet; wlasna
  pewnosc detektora wskazuje je lepiej (AUC 0,93) niz sygnaly z dynamiki (0,84 do 0,86).
- Widmo Koopmana: niezmienniki = tozsamosc operacji plus stan narzedzia; ich defekt poza proba spada
  z 0,19 przed uczeniem do 0,05 po uczeniu (krystalizacja bez etykiet). Liniowy operator dziala tylko
  przy 8 wymiarach (17 procent), model nieliniowy trzyma 21 procent.
- Pamiec: nowa regula "plan przed treningiem" (kampanie najpierw opisac, czekac na zatwierdzenie).

Pelny opis: `SESSION_SUMMARY.md`.
