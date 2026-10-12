# Raport: mały ViT od zera, tożsamość narzędzia, pętle FastGAN/SD, równanie ruchu latentu (28.09–1.10.2026)

Prefiksy: `EXP` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments`, `DOCS` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs`, DOK24 = `DOCS\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md` (wiersze 2383–4058).

Wspólne dla wycinka (DOK24 §4.6 i §5; `EXP\maly_vit_od_zera\PLAN.md`):
- **Sieć:** mały ViT, 2 926 493 parametry, 6 bloków, wymiar 192, wejście 480 × 320, 600 tokenów. Latent to 768 wyjść GELU warstwy ukrytej MLP ostatniego bloku, średnia po tokenach narzędzia (tło osobno).
- **Sondy i dane:** 192 obrazy 176 × 176 (12 wycinków × 16 wersji, 11 × 11 tokenów), 180 z narzędziem. Baza 1500 klatek, pula 9136 (log 4.3: 8495), 1 epoka = 47 kroków, paczka 32.
- **Walidacja a test:** każde mAP@0,5 to walidacja (próbka 3000 z 12 043 klatek 25 innych filmów). Zbioru testowego tu nie użyto („walidacja, nie test”, `EXP\maly_vit_petla_sd_rownanie\WYNIKI.md`). W eksperymentach równania „test” oznacza późniejsze kroki tego samego uczenia (zwykle 2000–2999).
- **Miara U:** umiejętność = 100·(1 − SSE/SS) wobec „braku zmiany”.
- **Zasięg DOK24:** kończy się 30.09 (F.1 dopisane 1.10). Wersje 4.0–4.3 opisują tylko `EXP\maly_vit_petla_sd_rownanie` i `DOCS\26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md`.

## A. Eksperymenty

**1. `maly_vit_od_zera` (pętla v1)** – `PLAN.md`, `PLAN_NASTEPNE_SPRAWDZENIE.md`
1. Czy sieć od zera ma ruchomy latent i czy klatki dobrane pod prognozę uczą szybciej niż losowe.
2. Lista neuronów „pełnego latentu” (min. 32) na 180 sondach, zapis co 10 kroków, bloki po 5, PCA do 8. Ziarna 42 (lokalnie) i 47–56 (Eden 1817935), 4500 kroków.
3. Koopman: Δy(t+1) = Σ_{j=0..3} A_j Δy(t−j). Langevin dodaje K·y(t) i stałą sondy. Dopasowanie tylko na historii. Cel FastGAN-a z Langevina, 3 obroty naprzód; obraz → DINOv2 top-90 → 134 klatki, reguła ±30.
4. Ramiona `losowe`, `bez`; „brak zmiany”, „trend”; kontrola target/current (wspólne top-90: 18,9 z 90).
5. Kalibracja Z0: 0,36 po 1500 krokach, 0,39–0,42 po 3500–6000, bramka ≥ 0,5 niespełniona. Langevin/„brak zmiany” w obrotach 5–9: 0,91; 0,99; 1,14; 1,05; 1,13. FastGAN bliżej celu w 39 % par. Reguła startu z analizy wstecznej niespełniona.
6. `koopman` minimalnie lepsze od losowego, bez dowodu udziału prognozy.
7. Zmierzone.

**2. `maly_vit_wektor_tozsamosci` (v2, tożsamość po 21 epokach)** – `PLAN.md`
1. Czy stan 768 neuronów i start po 21 epokach dają przewidywalną tożsamość.
2. 768 aktywacji × 180 sond; 10 ziaren (Eden 1817949); potem siła tożsamości, starty 1000–4250 co 250, horyzonty 50/250/500.
3. Modele jak w v1, horyzont 500 kroków, prognoza sprawdzana na następnym obrocie bez podglądania; potem na sile stałego wektora LDA.
4. `losowe`, `bez`, test offline na latentach v1, sieć nieuczona, wersja i wycinek wyłączony.
5. Prognoza wektora, Koopman: 0,989; 1,018; 1,006; 1,004; 1,003; 0,999 (0 z 6 obrotów). Siła, 500 kroków, ramię `bez`: Koopman 0,94 (85 % przypadków < 1), Langevin 0,90 (75 %); kierunek 64 % / 69 %.
6. Wektor stabilny, jego zmiana nieprzewidywalna; siła stałego wektora przewidywana lepiej niż „brak zmiany” o 6–10 %.
7. Zmierzone.

**3. `maly_vit_petla_sila_tozsamosci` (v3, iteracja 2)** – `ALGORYTM.md`, `PLAN.md`
1. Czy FastGAN sterowany o przewidzianą zmianę siły wybiera klatki lepsze niż losowe, kolejne i te same kotwice bez prognozy.
2. Siła 180 sond, wektor z kroku 1000; 10 ziaren (1818025), potem 20 (1818051).
3. Zwycięzca testu wstecznego (Koopman albo Langevin) prognozuje 500 kroków; 6 z 12 kotwic o największym |Δs|; inwersja i 300 kroków sterowania wejściem z.
4. `current`, `losowe`, `kolejne`, `bez`.
5. Prognoza w pętli v3: mediany 0,894–0,972. FastGAN: część zmiany 1,00, odporna 0,87–1,02. Reszta w sekcji C.
6. Sekcja C.
7. Zmierzone.

**4. `maly_vit_horyzont_prognozy`** – `WYNIKI.md`
1. Na ile kroków Koopman i Langevin, osobno, przewidują siłę tożsamości.
2. Historie siły z v3 i iteracji 2 (30 przebiegów, 5 ramion), H = 50–1000 kroków.
3. Modele pętli rekurencyjnie i „wprost” (osobne dopasowanie na każde H), tylko historia do startu.
4. „Brak zmiany”, przedłużony trend.
5. Iteracja 2, ramię `bez` (Koopman / Langevin / Koopman wprost / Langevin wprost): 500 kroków 14,1 / 15,9 / 15,4 / 17,3; 1000 kroków 13,3 / 8,3 / 18,2 / 18,5. Trend od −382 do −417.
6. Brak ostrej granicy: stały słaby sygnał 10–25 % do 1000 kroków.
7. Zmierzone (opisowo, bez kryterium z góry).

**5. `maly_vit_prognoza_tozsamosci_z_historii`** – `WYNIKI.md`
1. Jaki błąd na t+1, t+2, … mają równania Koopmana i Langevina z jądrem, pędem, przyspieszeniem.
2. Siła 180 sond co krok, nagrania s47 i s48; dopasowanie s47 1000–1999; 100 startów 2000–2495.
3. 15 wariantów liniowych ze stanu i historii.
4. Bezwładność; „znane klatki kroku t”; równanie na prawdziwych wagach (nie prognoza).
5. Najlepsze (Langevin + jądro 20 + pęd, wprost): 69 / 58 / 45 / 26 / 21 / 27 / 28 / 26 / 14 / 10 dla t+1, 2, 3, 5, 10, 20, 50, 100, 200, 500. Koopman z 20 opóźnieniami daje liczby identyczne z Langevinem z jądrem 20. Koopman DMD: 37 na t+1.
6. Z samej historii żaden model nie osiąga 90 %.
7. Zmierzone.

**6. `maly_vit_tozsamosc_jeden_krok`** – `WYNIKI.md`
1. Czy znając klatki kroku t, da się przewidzieć siłę w t+1 powyżej 85 %.
2. Latent i siła 180 sond, s47 i s48, test 2000–2998, współczynniki z s47.
3. Rozkład kroku AdamW przepuszczony przez sieć: S_paczka (gradient paczki t), S_pęd (pamięć do t−1), S_zanik.
4. Sam pęd; wersja z jedynkami bez dopasowania.
5. Latent / siła, s47 i s48: 0,90·S_pęd 66,4 / 65,4 i 64,4 / 63,8; S_paczka + S_pęd 95,4 / 95,2 i 95,5 / 95,6; z jedynkami 95,3 / 95,2 i 95,5 / 95,6.
6. 95 % na jeden krok przy znanej paczce t; bez niej 64–66 %.
7. Zmierzone (2 ziarna).

**7. `maly_vit_rownanie_latentu`** – `PLAN.md`, `WYNIKI_T0_T1.md`, `WYNIKI_T2.md`, `WYNIKI_EDEN.md`
1. Czy równanie Langevina z TFT ze sterowaniem danymi daje U > 80 na jeden krok.
2. 180 sond × 768, nagranie 3000 kroków, dopasowanie 1000–2000, test 2000–2999; lokalnie s47, Eden 10 ziaren (1818331).
3. T1: dryf do średnich ruchomych, prędkość, przyspieszenie, jądra pamięci, szum barwny. T2: dochodzi siła paczki S(t).
4. T1 jako podłoga; placebo (paczka z kroku −250); tanie jądro.
5. T0: wagi 97,8 % przyrostu, BatchNorm 0,7 %, ostatni blok 7,5 %. Eden, mediana (zakres): T1 59,1; T1 + paczka 85,4 (81,3–89,4); placebo 59,1; + pęd 94,9 (93,9–95,8); równanie proste 85,0 (80,5–89,2), a = 0,831, b = 0,981. Co 10 kroków T1 daje 17,6.
6. Cel spełniony w 10 z 10 ziaren, ale S(t) używa paczki przewidywanego kroku: wyjaśnienie, nie prognoza.
7. Zmierzone.

**8. `maly_vit_rownanie_macierze`** – `PLAN.md`, `WYNIKI.md`
1. Czy uczone macierze jak w TS przewidują krok bez przyszłych klatek.
2. 120 sond „fit”, 60 „holdout”; s47 i s48 osobno; uczenie 1000–1800, wybór 1800–2000, test 2000–2999; około 11 mln parametrów.
3. z(t+1) = z(t) + D + JP + W + N; W widzi paczki t−1…t−16.
4. `bez_W`, `placebo`, `bez_gradientu`, macierze 768 × 768, skalar bezwładności, pęd.
5. Latent (siła), s47 / s48: `pelne` 39,5 (17,0) / 40,8 (33,0); placebo 39,3 (15,4) / 40,3 (31,2); pełna macierz 52,0 (43,0) / 50,8 (47,5); 0,76·Δz 56,8 (56,6) / 56,3 (56,8); 0,94·S_pęd 66,8 (65,7) / 64,1 (63,8); wyjaśnienie 85,3 (85,7) / 87,8 (89,1).
6. Duże sieci przegrywają ze skalarem o 16–17 punktów; wymuszenie równe placebo.
7. Zmierzone; prognoza na 10 i 100 kroków niewykonana.

**9. `maly_vit_horyzont_rownania_krok`** – `PLAN.md`, `WYNIKI.md`
1. Na ile kroków równanie S_pęd + S_paczka trzyma 90 %.
2. Przyrost siły 180 sond, s47, 4 okna startu, H = 1–500.
3. A: co krok na prawdziwych wagach (nie prognoza). B: znane przyszłe klatki, wagi z t0. C1: sam pęd. C2: pęd i średni gradient zbioru.
4. A jako górna granica.
5. H = 1, 2, 3, 5: A 97, 98, 98, 98; B 98, 98, 88, 6; C1 62, 38, 20, −36; C2 73, 64, 56, −15. Ostatnie H z ≥ 90: A 150, B 2, C1 i C2 nigdy.
6. Przy zamrożonych wagach 90 % trzyma 2 kroki.
7. Zmierzone (1 ziarno).

**10. `maly_vit_petla_sd_rownanie` (4.0–4.3)** – `PLAN.md`, `WYNIKI.md`, `DOCS\26_...md`
1. Czy pętla ze Stable Diffusion uczy szybciej niż losowe i kolejne klatki i czy prognoza ma w tym udział.
2. Siła 180 sond; 4.3: 10 ziaren (Eden 1818736), 14 obrotów × 500 kroków.
3. Sekcja C.
4. Sekcja C.
5. Sekcja C.
6. Sekcja C.
7. 4.0 zatrzymana, 4.1 tylko diagnoza, 4.2 tylko próba kodu, 4.3 zmierzona.

**11. `maly_vit_sd_prowadzenie_tozsamoscia`** – brak plików .md; nagłówek `test_prowadzenia.py`, `wyniki\kadr.log`, `wyniki\proba1.log`
1. Czy obraz SD prowadzony latentem małego ViT pokrywa przewidziany latent w ≥ 90 %.
2. s47 po 1000 krokach, do 3 kotwic.
3. Cel z* = z_kotwicy + Δs·(sd ⊙ w), Δs z Langevina v3; gradient energii (s − s*)² + ‖(z − z*)/sd‖²/768 w krokach DDIM.
4. Ten sam obraz bez prowadzenia; cele ±1.
5. Ustawienie końcowe: tożsamość 93,5 % (bez prowadzenia 79,3), latent 91,9 % (82,8), oba ≥ 90 % w 2 z 3. Start z czystego szumu: 87,5 % / 61,9 %.
6. Próg osiągnięty w medianie na 3 kotwicach; w kampanii 4.3 mediany 88,5 % i 88,8 %.
7. Zmierzone (test lokalny).

**12. `dyfuzja_narzedzie`** – `PLAN.md`, DOK24 §5.4
1. Własny model dyfuzyjny wycinków narzędzia prowadzony małym ViT (FreeDoM).
2. UNet 22,3 mln parametrów, wycinki 176 × 176, 56 005 klatek z obrysem.
3. DDPM, T = 1000, 100 000 kroków × 64.
4. Nie znaleziono (ocena jakości niewykonana).
5. Próba 1818352: strata 0,97 → 0,05. Nasycenie kolorów 0,568 przy starcie DDIM od t = 999 wobec 0,163 w danych; poprawka niewprowadzona.
6. Model nauczony, jakość i przydatność w pętli niezmierzone.
7. Nierozstrzygnięte.

**13. `dyfuzja_sd_filmy`** – `PLAN.md`, `README.md`
1. Dostrojenie SD 1.5 na całych klatkach filmów.
2. 311 886 klatek 118 filmów `train`, 576 × 384, 7 kanałów wejścia (maski narzędzia, rogówki, źrenicy); 25 filmów `val` poza uczeniem.
3. Paczka 256, 20 000 kroków, 4 × H100 (1818496).
4. Nie znaleziono (`ocena_sd.py` niewykonane).
5. 7 h 05 min, strata 0,179 → 0,092, migawka `ema_020000`.
6. Wytrenowany bez błędów, jakość liczbowa niezmierzona.
7. Uczenie wykonane, ocena tylko w planie.

**14. `maly_vit_dekoder_latentu`** – `PLAN.md`
1. Rysowanie obrazu wprost z latentu, żeby zobaczyć różnicę „teraz” i „prognoza”.
2. Mediany z 4200 raportów FastGAN-a (v3 / iteracja 2).
3. Propozycja: dyfuzja warunkowana mapą 11 × 11 tokenów.
4. Kryteria etapu 2: część zmiany ≥ 0,8, zmiana w bok < 22–25.
5. |Δs| 0,55 / 0,66; zmiana w bok 22,3 / 25,1; błąd inwersji 0,104 / 0,105.
6. FastGAN trafia w siłę, ale zmienia resztę latentu około 40 razy mocniej.
7. Tylko plan.

**15. `maly_vit_tozsamosc_tla`** – `PLAN.md`
1. Czy pary „z narzędziem / bez” dają czystszy wektor i ile kierunków niesie tożsamość.
2. 10 sieci (krok 1000), 180 sond i 360 obrazów bez narzędzia, 1568 cząstek (tokenów) na sieć.
3. LDA Ledoit-Wolf (`stary`, `nowy`, `razem`); INLP.
4. Wycinki wyłączone.
5. Sekcja B.
6. Sekcja B.
7. Zmierzone.

## B. Tożsamość narzędzia w małym ViT

**Definicje w kolejnych wersjach**
- **v1, lista neuronów:** neuron „zapalony”, gdy aktywacja > średnia negatywów + 2 odchylenia; rdzeń = każda z 15 wersji na ≥ 75 % wycinków. Wyznaczana od nowa co obrót. Rdzeń: 0 w każdym ziarnie. Lista kurczy się z 80–88 do 47 (32 epoki) i 33 (96 epok); Jaccard ≈ 0,45 na 250 kroków.
- **v2, prosty wektor:** średnia narzędzia − średnia negatywów na 768 neuronach, liczona w chwili prognozy.
- **v3 i 4.x, `stary`:** LDA Ledoit-Wolf na średnich latentach sond; narzędzie 180 sond kontra tło 192 sond i obszar narzędzia w `empty_black`. Dopasowany raz w kroku 1000, stały do końca biegu. AUC 1,00 na wersji wyłączonej (żółte tło 0,93) i na wycinku wyłączonym; „rdzeń wektorowy” w 8 z 10 ziaren.
- **`nowy`:** narzędzie kontra te same tokeny bez narzędzia. Tylko pomiar, w pętli nieużyty.

**Rozdzielenie (`EXP\maly_vit_tozsamosc_tla\PLAN.md`, DOK24 §4.19; średnio 10 ziaren)**

| Miara | `stary` | `nowy` | `razem` |
|---|---|---|---|
| d′ wobec tego samego miejsca bez narzędzia | 2,0 | 16,6 | 15,3 |
| d′ wobec czarnego zdjęcia | 20,1 | 82,9 | 86,3 |
| d′ wobec dalekiego tła | 10,7 | 2,8 | 21,7 |
| AUC na wycinku wyłączonym | 0,967 | 0,997 | nie znaleziono |

Cosinus `stary`–`nowy`: 0,28.

**Ile kierunków**
- Cząstki, INLP: 1,3 kierunku z AUC ≥ 0,9 i 6,5 z AUC ≥ 0,75.
- Energia różnic: 50 % w 2,5 kierunku, 90 % w 38,4.
- Pierwszy kierunek cząstkowy ma cosinus 0,40 z `nowy` (około 66°).
- Na neuronach: 50 % wagi w 67–81, 90 % w 301–313 z 768.

**Stabilność**
- Między ziarnami: cosinus wektorów s47–s48 wynosi 0,085.
- W uczeniu, prosty wektor: cosinus z chwilą sprzed 250 kroków 0,72 (11 epok), 0,85 (16), 0,89 (32), 0,90 (64); z wektorem końcowym 0,20 / 0,41 / 0,59 / 0,75 / 0,83. Między obrotami v2: 0,84–0,86.
- Siła wzdłuż stałego wektora z 21 epok spada z 6,13 do 5,30 (`bez`) i 4,36 (`losowe`, `koopman`); Langevin k ≈ −0,16 na blok.

**Związek siły z mAP**
- Korelacji dla małego ViT nie znaleziono.
- Podano tylko: siła na końcu `tozsamosc` − `current` = +0,44, wyższa w 5 z 10 ziaren (`EXP\maly_vit_petla_sila_tozsamosci\wyniki_eden\DIAG_SILA.json`); przewidziane zmiany kotwic średnio −0,004.
- Dla RF-DETR Large (inna część linii): ρ = 0,07, p = 0,61, n = 53.

**Znane słabości**
- Sieć nieuczona przechodzi testy: 231–329 neuronów „pełnego latentu”; LDA 100 % poza żółtym tłem (97 %).
- `stary` mierzy w dużej części „miejsce i scenę”; diagnoza 4.1 dla wektora pętli: d′ 1,91, AUC 0,979 (`EXP\maly_vit_petla_sd_rownanie\wyniki\diagnoza.log`).
- Tło fałszywie zapala wektor w 0–41 % przypadków.
- Domknięcie neuronowe na czarnym i żółtym tle: 0,03 i 0,00.
- Sondy są poza rozkładem detektora: trafienie około 20 %, pewność na `empty_black` 0,24–0,39.
- Brak próby przyczynowej.
- `nowy` nie jest lepiej przewidywalny (Langevin 1,02 wobec 0,89).

## C. Pętle doboru klatek

| Wersja | Ramiona | Ziarna | Budżet | Kryteria z góry |
|---|---|---|---|---|
| v1 | `koopman`, `losowe`, `bez` | 11 | rozgrzewka 1500 kroków, 6 × 500, 804 klatki | pole w obrotach 4–9: `koopman` > obu w ≥ 4 z 5 |
| v2 | jak v1 | 10 | rozgrzewka 1000, 7 × 500, 938 klatek | ≥ 8 z 10; prognoza < 1 w ≥ 8 z 10 ziaren w ≥ 4 z 6 obrotów |
| v3 | `tozsamosc`, `current`, `losowe`, `kolejne`, `bez` | 10 | jak v2 (metoda 930 klatek) | > trzech odniesień w ≥ 8 z 10; rola prognozy: > `current` w ≥ 8 z 10 |
| v3 it. 2 | jak v3 | 20 | 14 × 500 do kroku 8000, lista 180, 1876 klatek (metoda 1708) | > `losowe` w ≥ 16 z 20 (pole i koniec); > `current` w ≥ 16 z 20 |
| 4.3 | `metoda`, `metoda_kl3`, `sd_bez_prognozy`, `losowe`, `kolejne`, `bez` | 10 | 14 × 500, równo 134 klatki na obrót | K1: > `losowe` w ≥ 70 % i p < 0,05; K2: > `kolejne` w ≥ 70 %; K3: > `sd_bez_prognozy` w ≥ 70 % |

**Wyniki (pole pod krzywą mAP@0,5, walidacja)**
- **v1:** 0,4426 / 0,4298 / 0,4044.
  - `koopman` > `losowe` w 9 z 11 (+0,013; Wilcoxon p = 0,005; znaków p = 0,065); `koopman` > `bez` w 11 z 11 (p = 0,001).
  - mAP końcowe tylko per ziarno (DOK24 §4.9).
- **v2:** 0,4432 / 0,4341 / 0,3999.
  - 8 z 10 (+0,009; p = 0,064; jedna wygrana o 0,00005).
  - Prognoza wektora: 0 z 6 obrotów.
  - Wobec v1: `koopman` +0,014, `losowe` +0,015.
- **v3:** pole 0,4432 / 0,4428 / 0,4327 / 0,4255 / 0,4030; koniec 0,5029 / 0,4913 / 0,4803 / 0,4608 / 0,4354.
  - Wobec `losowe`: +0,0105, 7 z 10 (p = 0,11).
  - Wobec `kolejne`: 10 z 10 (p = 0,002).
  - Wobec `current`: +0,0004, 5 z 10 (p = 0,92).
  - `current` > `losowe` w 9 z 10.
  - Kryterium główne i roli prognozy niespełnione; prognoza w pętli i FastGAN spełnione.
- **Iteracja 2:** koniec 0,545 / 0,548 / 0,536 / 0,534 / 0,456; pole 0,478 / 0,473 / 0,468 / 0,463 / 0,416.
  - Wobec `losowe`: pole 15 z 20 (+0,010; p = 0,006), koniec 11 (p = 0,41), krok 4500: 17.
  - Wobec `current`: 13 (p = 0,14), koniec 8 (−0,003).
  - Wobec `kolejne`: 18.
  - Kryteria niespełnione, poza FastGAN-em.
- **4.0:** cel zmieniony na spadek błędu detekcji; użytkownik zatrzymał.
- **4.1:** prognoza na 100 i 500 kroków równaniem Δs(t+1) = 0,83·Δs(t) + 0,98·F. Diagnoza: 1,5–50 razy gorsza od „braku zmiany”. Kampanii nie było.
- **4.2:** jeden krok S_pęd + S_paczka wybiera wersję SD. Tylko próba kodu (2 obroty × 40 kroków; równanie 97–99 %). Pełnego biegu nie znaleziono.
- **4.3** (Langevin z jądrem pamięci na 500 kroków; SD prowadzone do z*; DINOv2):

| Ramię | Pole | Koniec |
|---|---|---|
| `metoda` | 0,4883 | 0,566 |
| `metoda_kl3` | 0,4874 | 0,557 |
| `sd_bez_prognozy` | 0,4871 | 0,554 |
| `losowe` | 0,4720 | 0,544 |
| `kolejne` | 0,4646 | 0,519 |
| `bez` | 0,4219 | 0,451 |

  - `metoda` − `losowe`: +0,0163, 10 z 10, p = 0,002; na końcu +0,022, 7 z 10, p = 0,13.
  - `metoda` − `kolejne`: +0,0237, 10 z 10.
  - `metoda` − `sd_bez_prognozy`: +0,0012, 5 z 10, p = 1,0.
  - `metoda` − `metoda_kl3`: +0,0008, 6 z 10, p = 0,77.
  - `sd_bez_prognozy` − `losowe`: +0,0150, 9 z 10, p = 0,006.
  - K1 i K2 spełnione, K3 nie.
  - Trafność prognozy na 500 kroków: 28,7 % (Langevin z jądrem) wobec 16,2 % (v3).

**Co wywnioskowano o osobnych wkładach**
- **Prognoza:** udziału nie wykazano w żadnej wersji. W v1–v2 była równa „brakowi zmiany”; w v3, iteracji 2 i 4.3 ramię bez prognozy dało to samo. Trafniejsza prognoza nie zmieniła doboru.
- **Generator:** wkład nierozdzielony. Brak ramienia „wyszukanie z prawdziwej klatki kotwicy bez SD” (`WYNIKI.md` 4.3; DOK24 F.3 pkt 1).
- **Wyszukiwanie:** dokumenty przypisują zysk wyszukiwaniu prawdziwych klatek podobnych do wycinków narzędzia (w 4.3: „SD + wyszukanie”).
- **Inne:** kolejne klatki gorsze od losowych (v3: −0,007); każde dokładanie bije brak dokładania; przewaga nad losowymi maleje pod koniec uczenia.

## D. Sekcje E, F, G dokumentu 24

**E. „Co wiemy po dziesięciu dniach”**
- **E.1:** we wszystkich porównaniach, w których prognoza jest jedyną różnicą (9 pozycji z etapów 1–4), wynik mieści się w szumie ziaren albo jest ujemny. Zysk daje wyszukiwanie klatek podobnych do narzędzia. Na dostrajanym RF-DETR nie działało nic różnego od losowania (losowe 0,7131 najlepsze).
- **E.2, siedem mechanizmów:**
  1. Przewidziane przesunięcie jest małe (0,069 rozrzutu sond).
  2. Lista najbliższych klatek jest stabilna (54 z 60 wspólnych).
  3. Dostrajany kręgosłup prawie stoi (0,026 % w 6 epokach).
  4. Latent kręgosłupa nie wiąże się z postępem detekcji (ρ = 0,07).
  5. Generator nie zmienia jednej rzeczy (najwyżej 36 % drogi; zmiana w bok około 40 razy większa).
  6. Dobór celowany traci różnorodność (138–140 wobec 62–83 sąsiadów).
  7. Efekty są rzędu szumu ziaren (0,001–0,003 mAP).
- **E.3:**
  - Latent krystalizuje wcześnie.
  - Bezwładność 83 %; siła paczki ze współczynnikiem 0,98; wyjaśnienie 85 %, prognoza 57–67 %.
  - Proste równanie wygrywa z dużym (około 40 %; macierze 52–54 %).
  - Tożsamość trzeba mierzyć na parach (d′ 2,0 wobec 16,6); to wąska wiązka kierunków.
  - Siła przewidywalna słabo (6–10 %; 10–25 % do 1000 kroków).
  - Mały ViT jest dobrym polem doświadczalnym.
  - Wyszukiwanie przyspiesza uczenie (15 z 20, p = 0,006).
- **E.4:** tabela ewolucji prognozy od operatora Koopmana (etap 1) do równania z siłą paczki (etap 5).
- **E.5:** konieczne ramię różniące się tylko prognozą; placebo rozstrzyga o sterowaniu; wyjaśnienie to nie prognoza; kryteria przed wynikiem; zbiór testowy został użyty do diagnozy w etapie 2; lista błędów kodu.
- **E.6:** reguły pracy (plan przed kampanią; nie podmieniać metody użytkownika; bez przyszłych klatek w prognozie; higiena Edenu; raport po treningu).

**F. Stan i następne kroki**
- **F.1:** SD zakończone 30.09 o 17:56.
- **F.2, czeka na zgodę:** `ocena_sd.py` i prowadzenie FreeDoM; prognoza wielokrokowa T4 na wagach s48; usunięcie artefaktów próby SD; poprawka `siatka_krokow`; poprawka `SzumKolorowy` w kodzie TS.
- **F.3, pytania otwarte:**
  1. Czy prognoza może zmienić wybór klatek.
  2. Próba przyczynowa tożsamości.
  3. Równanie na wiele kroków i przy dokładaniu klatek.
  4. Jakość SD.
  5. Przeniesienie na RF-DETR.
  6. Zanik przewagi pod koniec uczenia.
  7. Nowy wektor w pętli.

**G.2. Sprzeczności oznaczone w dokumencie**
- **Etap 4:**
  - Kryterium v1: plan „spełnione w proporcji”, `ANALIZA.json` ma `false`.
  - „39 % par” przypisane raz v1, raz v2.
  - Z0: 0,39–0,42 / 0,40–0,42 / 0,354–0,423; plan 4–6 mln parametrów, zbudowano 2,93 mln.
  - Węzły (hopper wobec dgx-4).
  - Sprzątanie: 192,4 wobec 193,2 GB.
  - Wariant B FastGAN-a: 1000 wobec 400 kroków.
  - `current`: plan (sterowanie do obecnej siły) wobec kodu (obraz z inwersji).
  - Kierunek prognozy: 85 % poprawione na 69 %.
  - „Cały latent” wobec „tylko tożsamość” w słowach użytkownika.
  - Przypisanie commitu `fb899bde`.
- **Etap 5:**
  - Dwie nitki rozmowy i fork.
  - Cztery liczby koło 85 (85,8 i 85,4 lokalnie; 85,4 i 85,0 na Edenie).
  - Zaokrąglenia w `WYNIKI.md` horyzontu.
  - Jednostki kwoty (GB wobec GiB).
  - Podgląd SD: 32 klatki wobec 16 par.
  - 56 301 wobec 56 005 klatek z obrysem.
  - Los zadania 1818356.
  - Koder tekstu „nieładowany” wobec kodu.
  - 25 wobec 17–19 min na ziarno.
- **Etapy 1–3:** wiersze 3985–4022, poza tym wycinkiem.

**Rozbieżności zauważone przy odczycie (nieoznaczone w DOK24)**
- `DOCS\26` krok 8 podaje 7 obrotów i 938 klatek, a krok 10, `WYNIKI.md` i logi (k3–k16) podają 14 obrotów.
- Współczynnik przy S_pęd: 0,94 (`rownanie_macierze`), 0,90 (`tozsamosc_jeden_krok`), 0,917 (diagnoza 4.1).

## E. Porównania z szeregami czasowymi (TFT, projekt grantowy)

**Przeniesione**
- Postać z(t+1) = z(t) + D + JP + W + Z + N z `tft_langevin_latent_56` i `tft_langevin_czlon_zdarzen`.
- Miara „jak w TFT”.
- Jądra pamięci jako tłumione, obracające się filtry.
- Szum barwny z własnych reszt.
- Receptura uczenia macierzy: AdamW 1e-3, EMA 0,995, kara rozdzielności 0,05.

**Nazwane różnice**
- 72 600 serii wobec 120 sond (około 600 razy mniej, jeden przebieg).
- Dzień wobec kroku AdamW.
- Stan 192 wobec 768.
- Okno pamięci 56 dni wobec 32 kroków.
- TS widział „jutro” (plan promocji); tu paczki kroku t nie ma.

**Zachowanie**
- W TS bez wymuszenia wynik spadał o 30–60 punktów, a jądro pamięci dawało mniej niż punkt.
- Tu jądro daje około 4 punktów ponad bezwładność, siła paczki około 25 (+26 w medianie).
- Uczone wymuszenie z przeszłych paczek jest równe placebo.
- Przeuczenie: TS 94 na cząstkach uczących i 62 na nowych; tu około 89 / 50 / 40.
- Skalar 56–57 bije sieci (39–41) i macierze (52–54).

**Inne stwierdzenia**
- Koopman użytkownika na TFT miał kierunek rosnący o prawie 7 % na krok i rozjechał się po 3 dniach (DOK24 §5.2).
- Odpowiednik wymuszenia w ViT to zaplanowane dołożenia klatek; odpowiednik zdarzeń to początek obrotu.
- W `langevin_latent_detektora` pamięć 8 przyrostów poprawiała długi rollout o 11–13 %.
- Błąd w kodzie TS: `SzumKolorowy` widzi reszty od t−2 zamiast t−1; brakuje ich w biegach L1–L3; niepoprawione, wpływ niezmierzony.
- Pełnego równania TFT na horyzoncie siły tożsamości nie sprawdzano.

## Kluczowe źródła

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\PLAN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_wektor_tozsamosci\PLAN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sila_tozsamosci\PLAN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_tla\PLAN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_rownanie_latentu\WYNIKI_EDEN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_rownanie_macierze\WYNIKI.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sd_rownanie\WYNIKI.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sd_rownanie\wyniki_eden43\WYNIKI_EDEN43.json`
