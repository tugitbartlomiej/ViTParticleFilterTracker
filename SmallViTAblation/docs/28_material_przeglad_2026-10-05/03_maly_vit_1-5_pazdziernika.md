# Raport: prace 1–5 października 2026 (cząsteczki, J128, gęsta historia, EDMD, NNDMD, sufit, plany)

Tylko odczyt. Liczby w zapisie źródła (kropka albo przecinek jak w pliku). W katalogu `experiments` nie ma folderów nowszych niż 2026-10-03 07:34. Poza listą istnieje `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf` (2026-10-04), nieczytany.

## 0. PIPELINE od wiersza 327

`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\PIPELINE_TOZSAMOSCI_NARZEDZIA.md`

- **9.4–9.6:** zamrożenie protokołu przed runami; główna miara to sparowana różnica znormalizowanego pola pod krzywą mAP względem optimizer-step, F−B; jednostką replikacji jest cały seed-run; „10 seedów to cel planu”.
- **Sekcja 10 (dziesięć pytań):** 128 to „najmniejsza sprawdzona liczebność przekraczająca 90 procent na validation”; wynik 92,8274 procent, przedział [88,17%;96,99%]; EDMD to wspólna macierz 128×128 i bias 128.
- **Korekty z 2026-10-02:**
  - 12:32–12:36: ekstrakcja `LATENT_768.npy` (101,34,600,768) i decyzja „encoder 768 → 768, wyuczana macierz Koopmana 768 × 768, decoder 768 → 768”.
  - 14:12: wymóg 30/30 obrazów FIT z dokładnej bazy 1500 (było 25/30).
  - 14:36: 1500 obrazów, batch 32 obrazy, okno 10 przejść.
  - 15:12:03 (ostatni wpis): job 1819498 „PENDING (Resources), 0 kroków optymalizatora”.
- Późniejszych awarii, anulowania i wyniku Hoppera dokument nie zawiera.

## A. Eksperymenty

**A1. maly_vit_ruch_czastek_gesta_historia** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\WYNIKI.md`
1. Pytanie: czy w gęstej historii uczenia tokeny narzędzia ruszają się w kierunku odróżniającym je od tła.
2. Obiekt: pełne 768; 12 obrazów × 600 tokenów × 1001 stanów (kroki 0..1000 co 1), 22,1 GB; ziarno 47. Paczka 32, AdamW 5e-4, rozgrzewka lr 300 kroków. Wagi co 10 kroków, stan AdamW co 100. Tryb eval, więc ruch obejmuje BatchNorm.
3. Metoda: SVD kierunków ruchu po odjęciu dryfu tła, ROC AUC narzędzie/tło na 4 odłożonych filmach, kontrola losowymi kierunkami.
4. Prognoza: nie dotyczy.
5. Wyniki:
   - Względna zmiana tokenów na krok (mediana): 0,023 (kroki 1–10), 0,047 (10–100), 0,072 (100–300), 0,066 (300–1000).
   - AUC normy po odjęciu dryfu: 0,761 (co 1) / 0,740 (co 10); średni kierunek 0,519 / 0,522; 32 kierunki SVD/losowe 0,730 / 0,733.
   - Osi SVD do 90 % energii: 192 / 202.
6. Werdykt: tokeny narzędzia ruszają się „mocniej niż odległe tło, ale nie w swoistym kierunku”.
7. Status: zmierzone (Eden 1818732).

**A2. maly_vit_ruch_czastek_maska_1px** (2026-09-30) — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_maska_1px\WYNIKI.md`
1. Pytanie: czy maska od jednego piksela i osobne trajektorie tokenów ujawniają kierunki odróżniające narzędzie od tła.
2. Obiekt: pełne 768; [2, 12, 600, 768], kroki 60 i 120; 7–31 tokenów maski na obraz, mediana 18.5.
3. Metoda: SVD i AUC wobec losowych podprzestrzeni.
4. Prognoza: nie dotyczy.
5. Wyniki: AUC normy 0.5063; po odjęciu dryfu 0.5665; średni kierunek 0.6689; 8 kierunków 0.6071 wobec losowych 0.5606; 16 osi SVD do 90 procent energii.
6. Werdykt: słaby sygnał kierunkowy z jednego przedziału; „nie wyznacza prognozy Koopmana ani Langevina”.
7. Status: zmierzone (wstępnie).

**A3. maly_vit_tozsamosc_czasteczki_pary_400** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_czasteczki_pary_400\WYNIKI.md`
1. Pytanie: które osie reagują na usunięcie narzędzia (LaMa) i czy ich podmiana wpływa na detektor.
2. Obiekt: pełne 768; zamrożony k1000, ziarno 47; 400 par ze 116 filmów; 6729 tokenów maski × 2 i 3200 tokenów tła; pary 240/75/85 (filmy 69/23/24). To reakcja na zmianę obrazu, nie ruch w uczeniu.
3. Metoda: SVD niecentrowanego drugiego momentu różnic (rank 8, 60.53 procent energii; r50 = 5, r90 = 88) plus regresja logistyczna; LDA; interwencje na GELU z losowymi kierunkami o dopasowanej normie (768D i po fc2); sham.
4. Prognoza: nie dotyczy.
5. Wyniki:
   - SVD 8: AUROC 96.86%, zbilansowana trafność 88.61% [83.37, 93.30]; LDA 97.94% / 95.28%.
   - confidence×IoU: 0.36232 → 0.34303, CI zmiany [-0.02763, -0.01230].
   - AP50: 45.76% → 44.11%, CI [-6.24563, +4.11311] punktu.
   - Sham: delta L2 w tool ROI 0.25051 wobec 4.84873 po usunięciu narzędzia.
6. Werdykt: odczytywalny sygnał obecności i wpływ na confidence×IoU; „Brak rozstrzygnięcia AP50 i straty”.
7. Status: zmierzone.

**A4. maly_vit_test_orientacji_narzedzia_400** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_test_orientacji_narzedzia_400\WYNIKI.md`
1. Cel: zbiór 400 klatek train w różnych naturalnych kątach i ich wersji bez cystotomu.
2. Obiekt: 116 filmów, 12 przedziałów kąta osi po 15 stopni (33–34 zdjęcia); 800 obrazów; bez forwardów ViT.
3. Metoda: LaMa ONNX na CPU; maska edycji to dylatacja 5 pikseli (jedna para 10 po QC).
4. Prognoza: nie dotyczy.
5. Wyniki: „zero zmienionych pikseli poza zadeklarowanymi maskami edycji”; 10 277 poprawnych kandydatów spośród 10 637; ręcznie obejrzano 17 par.
6. Werdykt: zbiór gotowy; kąty naturalne, „nie kontrolowane rotacje”; „nie heldout detektora”.
7. Status: zmierzone (przygotowanie danych).

**A5. maly_vit_tozsamosc90_bez_pca_langevin** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc90_bez_pca_langevin\WYNIKI.md`
1. Pytanie: czy natywne kanały bez PCA dają ponad 90 procent rozpoznania obecności i czy GLE przewiduje ich ruch.
2. Obiekt: J128. Rozpoznanie: k500, średnia ROI, 400 par. Dynamika: sonda 2 (31 cząsteczek) i transfer na sondę 8 (7), 1001 stanów, ziarno 47, dt1 i dt10.
3. Metoda: ranking kanałów tylko na train plus logistyczna L2 (K=128, C=0.01). GLE lokalny/sprzężony, L0/1/2/4; wybrano lokalny L2 (ridge 1 dla dt1, 10 dla dt10).
4. Prognoza: fit 501..700, walidacja 700..749, refit do 799, test od 800, rollout autonomiczny. Podłogi: persystencja, stała prędkość, L0.
5. Wyniki:
   - Rozpoznanie: BA 92.83%, CI [88.17, 96.99]%.
   - GLE dt1: H1 skill 47.75%, znak 71.68%; H10 -4.15%; H50 -45.30%.
   - Stała prędkość H1: 55.92% / 78.67%.
   - GLE dt10, H10/20/50: 4.30 / 9.40 / 4.15%.
   - Sonda 8 (dt1): -1.41 / -262.36 / -265.27%.
6. Werdykt: próg 90 osiągnięty punktowo; „Nie wykazano jednego dobrego równania dla tych horyzontów”.
7. Status: zmierzone.

**A6. maly_vit_langevin_kierunki_tozsamosci** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_langevin_kierunki_tozsamosci\WYNIKI.md`
1. Pytanie: czy GLE przewiduje ruch cząsteczek w ośmiu kierunkach U500.
2. Obiekt: rzut 8D (U500 z 240 par train na k500); sonda 2, 31 tokenów; 1001 stanów; ziarno 47.
3. Metoda: wspólne b, A, K_l dopasowane ridge; wybrano L4, ridge 0.01 (dt1) i L1, ridge 1 (dt10).
4. Prognoza: jak A5; dodatkowo znak większości.
5. Wyniki:
   - H1: skill 60.24%, znak 78.61%, po filtrze 81.59%.
   - H10: 9.11%; H50: -5.03%.
   - Stała prędkość H1: 47.77%.
   - dt10: 16.12 / 20.12 / 13.28%.
   - Osiem tokenów tła, H1: 57.60%, znak 77.55%.
6. Werdykt: „Nie osiągnięto 80 procent ani redukcji MSE, ani znaku wszystkich zmian.”
7. Status: zmierzone.

**A7. maly_vit_langevin_wspolny_ruch** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_langevin_wspolny_ruch\WYNIKI.md`
1. Pytanie: czy sprzężenie ze średnim ruchem narzędzia i tła oraz kierunki PCA/PLS poza U500 poprawiają GLE.
2. Obiekt: cel 8D U500; sondy 2 i 8; ziarno 47.
3. Metoda: 10 rodzin, 450 konfiguracji walidacyjnych; wybrano localPLS8 L=2, ridge=1 (dt1) i coupledPLS4 L=0, ridge=10 (dt10).
4. Prognoza: jak A6; kontrole oldGLE i localU8.
5. Wyniki:
   - dt1: H1 49.68%, H10 14.88%, H50 0.18%; znak H1 75.87%.
   - Średni testowy skill H1/10/50: 21.58% wobec 21.44% starego modelu.
   - Sonda 8: 30.52 / -5.13 / -32.45%.
   - Udział energii wspólnego przyrostu: 82.10%.
6. Werdykt: „nie uzyskano ogólnie lepszego modelu ani skutecznego transferu długiej prognozy”.
7. Status: zmierzone.

**A8. maly_vit_identyfikacja_langevin_koopman** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_identyfikacja_langevin_koopman\WYNIKI.md` oraz `wyniki\krotki_horyzont\WYNIKI.md`, `wyniki\koopman_obserwator\WYNIKI.md`, `wyniki\joint_controlled\WYNIKI.md` w tym folderze
1. Pytanie: czy historyczne paczki i ich wpływy poprawiają prognozę siły tożsamości.
2. Obiekt: skalar LDA s (nie J128); 180 sond; nagrania s47 i s48 po 3001 stanów co 1 krok.
3. Metoda, cztery etapy:
   - 17 ramion H=10..500 (GLE: 20 opóźnionych przyrostów plus odchylenie od średniej kroczącej; deskryptory i wpływy paczek).
   - Krótkie horyzonty: lifting (x, x², x·v, v²) plus GLE20 plus wpływy.
   - EDMD i uczony lifting (AE, 18 epok) z filtrem Kalmana AR1.
   - Wspólny controlled Koopman-GLE.
4. Prognoza: stany do t, paczki do t−1. Podłogi: zero, dryf, większość znaku.
5. Wyniki:
   - H500: 22,65 wobec 22,64 procent (s48), znak 67.0; późny s47 -34.83.
   - H1: 80,98 procent znaków i 64,96 procent redukcji MSE; dawny Langevin 79,86 / 62,40.
   - H10: 65,07 / 22,13.
   - EDMD+KF, H1: −53,08 / 57,06.
   - Joint, H1: 64,12 / 80,92.
6. Werdykt: „Nie wykazano lepszego równania prognozy na 500 aktualizacji z historycznych paczek.”
7. Status: zmierzone; s48 eksploracyjne.

**A9. maly_vit_koopman_langevin_128_bez_pca** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_koopman_langevin_128_bez_pca\WYNIKI.md`
1. Pytanie: porównanie liniowego Koopmana z zamrożonym Langevinem na tych samych J128.
2. Obiekt: obraz 2 (31 cząsteczek) i 8 (7); ziarno 47.
3. Metoda: Koopman przyrostowy (K = I + A, czyli L0), bezpośredni (kara ku K = 0), Langevin z dwoma przyrostami; macierze 128×128.
4. Prognoza: fit do 700, walidacja do 799, test 800–950. Podłogi: brak zmiany, stała prędkość.
5. Wyniki (dt1, H1, znak / redukcja MSE):
   - Langevin 71.68% / 47.75%.
   - Koopman przyrostowy 50.65% / -0.43%.
   - Koopman bezpośredni 50.63% / -193.68%.
   - Stała prędkość 78.67% / 55.92%.
6. Werdykt: Koopman bez pamięci nie przewiduje; na H1 wygrywa stała prędkość.
7. Status: zmierzone.

**A10. maly_vit_prognoza_tozsamosci_500_bez_pca** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_prognoza_tozsamosci_500_bez_pca\WYNIKI.md`, `PODSUMOWANIE.md`, `LITERATURA_I_AUDYT.md`
1. Pytanie: czy którykolwiek model osiąga ponad 85 procent znaku skumulowanego przemieszczenia J128 do H500.
2. Obiekt: J128; 218 cząsteczek z 12 obrazów (fit 0–7, transfer 8–11); origin 500; ziarno 47.
3. Metoda: 99 konfiguracji (Koopman, przycięte widmo, GLE ze skończoną i wykładniczą pamięcią, dt1/dt10, Kalman, obserwator narzędzie+tło, OLS, wygaszana prędkość, powrót do średniej); 16 zamrożonych wariantów na teście.
4. Prognoza: dane 0..500, pełny autonomiczny przebieg do 1000. Podłogi: brak ruchu, ostatnia prędkość, znak większości.
5. Wyniki (transfer):
   - Wybrany `reversion_w200_t200`: znak 67.06% (H10), 56.52% (H500); MSEskill H500 -7.52%.
   - `reversion_w50_t50`, H500: 62.19%, 9.82%.
   - Ostatnia prędkość, H1: 75.73%.
   - Kalman, H500: -279377.46%.
6. Werdykt: „Nie osiągnięto celu ponad 85 procent.”
7. Status: zmierzone.

**A11. maly_vit_pykoopman_30_obrazow_128** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_pykoopman_30_obrazow_128\RAPORT_PYKOOPMAN.md`, `RAPORT_NNDMD_FULL768.md`, `AUDYT_NNDMD_ASTRA.md`
1. Pytanie: czy wspólny operator EDMD, a potem NNDMD, osiąga 85 procent na czterech holdoutach.
2. Obiekt: 34 obrazy (30 fit, holdout 8–11), 101 checkpointów co 10. EDMD: J128, tokeny narzędzia, 25100 par tokenowych. NNDMD: pełne 768, J128 po dekodowaniu.
3. Metoda:
   - EDMD: `pykoopman.Koopman(Identity,EDMD(svd_rank=-1,tlsq_rank=0))`, A 128×128 plus bias, lambda=0.01.
   - NNDMD: encoder 768→768→768 (SiLU), A=exp(G) 768×768, decoder liniowy; look_forward10, beta1, epoka 5.
4. Prognoza: origin 500, prognoza 500..1000 bez przyszłych obserwacji. Podłogi: brak zmiany, reversion.
5. Wyniki:
   - EDMD: znak 57.076 (H10), 57.369 (H500); najwyższy 61.226 przy H180; skill H500 -0.516026.
   - NNDMD, H500: znak 54.498, RMSE 0.448238 wobec 0.233894, skillMSE -2.672655; maksimum znaku 60,2754 przy H320.
   - Walidacja NNDMD: 0.237974 wobec 0.211199 braku zmiany.
6. Werdykt: oba gorsze od persystencji na wszystkich 50 horyzontach.
7. Status:
   - EDMD i NNDMD: zmierzone.
   - `nndmd_temporal90_10`: „aborted_before_first_completed_epoch” (wymóg 30/30 FIT z bazy 1500).
   - `BASE1500_CORRECTED`: dane 5 obrazów i plan.
   - Online RLS: odłożone.

**A12. maly_vit_dynamika_pelnych_czasteczek_checkpointy** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_dynamika_pelnych_czasteczek_checkpointy\WYNIKI_PILOT.md`
1. Pytanie: czy prognoza pełnych wektorów bije persystencję.
2. Obiekt: pełne 768; 218 tokenów masek z 12 obrazów; 101 stanów co 10; ziarno 47.
3. Metoda: PCA r=8 z przeszłości plus ridge AR(1); pełny ridge 768D; wariant z jednym opóźnieniem.
4. Prognoza: początki testu 600–900. Podłogi: persystencja, stała i wygaszana prędkość.
5. Wyniki:
   - Redukcja RMS względem persystencji: 3.62 / 3.81 / 5.06 / 4.31 procent (H 10/20/50/100).
   - Pełny ridge przy H=100 gorszy o 14.33 procent.
   - Stała prędkość: 7.7558 (stosunek do persystencji).
   - 8 osi opisuje 41.25 procent przeszłej wariancji; 63.35–70.88% przyszłego ruchu leży poza PCA.
6. Werdykt: „mały dodatni wynik predykcyjny jednego przebiegu”.
7. Status: zmierzone (pilot).

**A13. maly_vit_nndmd_1500_obrazow_768** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_nndmd_1500_obrazow_768\PLAN_DANYCH.md`, `AUDYT.md`
1. Pytanie: czy NNDMD na 1500 obrazach daje użyteczną prognozę 10 checkpointów.
2. Obiekt: 1500 × 600 × 768; fit cp0..90, test 90→100; COCO z bbox, „bez segmentacji”.
3. Metoda: nieliniowy encoder i decoder, A 768×768; rolling cache 11 checkpointów zamiast 279244800000 B.
4. Prognoza: planowane kontrole persistence, velocity/reversion, EDMD.
5. Wyniki: brak wyników prognozy; `BENCHMARK_NN.json` ma „benchmark_only”.
6. Werdykt: lokalny wariant zatrzymano około 14:40 i przeniesiono na Eden.
7. Status: plan i przygotowanie danych.

**A14. maly_vit_nndmd_1500_expandedstate_eden** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_nndmd_1500_expandedstate_eden\status.json`, `WYNIKI.md`, `run_multigpu\SELECTION.json`, `local_checkpoint_check_20261002\RAPORT_POSREDNI.md`
1. Pytanie: czy rozszerzony stan (przyrosty oraz statystyki wag, BN i AdamW) poprawia prognozę 90→100.
2. Obiekt: 1500 × 600 × 768; wejście 2544; walidacja 80→90.
3. Metoda: encoder 2544→768→768, A=exp(G), decoder 768→768→827; batch 32 obrazy; dwa A100.
4. Prognoza: encoder raz, bez przyszłych aux. Podłogi: persistence, kontynuacja przyrostu, średnia trzech stanów.
5. Wyniki:
   - 1819498: FAILED, mkdir „Permission denied”, 0 kroków.
   - 1819551: FAILED po 1 kroku, „dict() got multiple values for keyword argument 'origin'”.
   - 1819574: CANCELLED, „cancelled_by_user”, 12062 kroki, `final_test_completed: false`.
   - Walidacja, epoka 1: 0.14628801881827 wobec persistence 0.13378740272320994.
   - Pośredni checkpoint, jeden obraz, H10: 0.19047136 wobec 0.08160770; znak J128 50.367–54.273%.
6. Werdykt: wyniku końcowego nie ma; liczby pośrednie gorsze od persystencji.
7. Status: przebiegi nieudane i anulowany.

**A15. maly_vit_nndmd_lokalne_przyrosty_hopper** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_nndmd_lokalne_przyrosty_hopper\run1819918\RAPORT_KONCOWY.md`, `run1819918\RAPORT.md`
1. Pytanie: „czy wspólne uczenie lokalnej dynamiki przyrostów poprawia surowe RMSE względem persistence”.
2. Obiekt: 1500 × 600 × 768; fit cp50..90 („400 aktualizacji zrodla”); test 90→100.
3. Metoda: jak A14 plus prognoza zakotwiczona zhat_h = z0 + s·(D(A^h phi0) − D(phi0)); lrG 0.0001 (wybrane) i 0.001; 7379 kroków; cztery H200 i siedem MIG.
4. Prognoza: podłogi persistence, średnia historyczna, kontynuacja ostatniej zmiany.
5. Wyniki:
   - full768: 0.159935 wobec 0.139319 (14.80% większy błąd).
   - J128: 0.141612 wobec 0.122625 (15.48%).
   - MSE skill full768: od -0.086306 do -0.51512.
   - Sign all: 0.48797–0.54055 (full768), 0.47793–0.53481 (J128).
6. Werdykt: „Model przegrywa z persistence w każdym z dziesięciu horyzontów obu grup.”
7. Status: zmierzone, COMPLETED. Wcześniej nieudane próby infrastruktury: 1819903 („CUDA sees8, requested11 devices”) i 1819908 (NCCL 2.27.5, „Duplicate GPU”).

**A16. maly_vit_gesta_historia_paczka** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_gesta_historia_paczka\WYNIKI.md`
1. Pytanie: czy większa paczka zmniejsza losowe pchnięcia tak, by prognoza z historii przekroczyła 85 procent na 10 krokach.
2. Obiekt: pełne 768; (1001, 12, 60, 768); paczki 32, 128, 320; ziarno 47; 458 jednostek.
3. Metoda: jądro pamięci J = 5 plus ściąganie do średniej z 51 stanów, dopasowane wprost na horyzont; OU; sufit wyroczni.
4. Prognoza: historia do t; odniesienie bez zmian.
5. Wyniki (1 − SSE/SSE bez zmian, paczki 32 / 128 / 320):
   - h10: 0,20 / 0,24 / 0,24.
   - h1: 0,64 / 0,44 / 0,38.
   - acf 1: 0,71 / 0,50 / 0,31.
   - Krok względny: 0,078 / 0,083 / 0,082.
   - Sufit h10: 0,79 / 0,76 / 0,72.
6. Werdykt: „Teza z PLAN.md jest obalona.”
7. Status: zmierzone; ramię β1 niewykonane („kod gotowy”).

**A17. maly_vit_sufit_przewidywalnosci_100** — `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_sufit_przewidywalnosci_100\WYNIKI.md`, `WYNIKI_CO_JESZCZE.md`, `WYNIKI_PROGNOZA_900.md`, `WYNIKI_ROWNANIE_Z_PEDEM.md`
1. Pytanie: czy jakakolwiek prognoza na 100 kroków może zmniejszyć błąd „bez zmian” o 85 procent.
2. Obiekt: gęsta historia (ziarno 47); tokeny 768, średnie, rzut LDA. Dodatkowo 180 sond z ziaren 47 i 48 (równanie z pędem) oraz ziarna 47–56 co 10 kroków.
3. Metoda: wyrocznia (średnia ruchoma w=21/51), podłoga z drgań, model liniowy 8 cech, OU, jądro J=5 plus pęd i zanik wag.
4. Prognoza: stany do t; granice z przyszłymi paczkami to „diagnoza, nie metoda”.
5. Wyniki:
   - h=100: sufit 0,84–0,89 (w=21) i 0,69–0,80 (w=51); najlepsza prognoza przyczynowa 0,31–0,35 w MSE (0,17–0,19 w RMSE).
   - OU od kroku 900: 0,21–0,22.
   - NNDMD na tym odcinku: od -0,09 do -0,52.
   - Historia + pęd + zanik: h=1 0,68 (0,67); h=10 0,22 (0,23).
   - Z prawdziwą pełną siłą przyszłych kroków, h=10: 0,985 (0,983).
   - Wczesna faza, 10 ziaren: 0,11.
6. Werdykt: cel „jest poza zasięgiem prognozy z własnej historii latentu”.
7. Status: zmierzone.

## B. Cel 85 procent

- **Znak** (PIPELINE w. 13 i 167; A10; A11): „co najmniej 85 procent trafności znaku prognozowanego skumulowanego przemieszczenia”, wszystkie 128 kanałów, sign(0)=0, bez odsiewu. Horyzonty H1–500 od originu 500; w NNDMD-1500 tylko 10–100 aktualizacji. Najlepsze wartości:
  - Ostatnia prędkość, H1: 76,8822 procent (fit) i 75,7328 (transfer).
  - H500: 62.19% (transfer).
  - EDMD 61.226; NNDMD-34 60,2754; NNDMD-1500 0.54055.
  - J128 przy dt1 (A5): stała prędkość 78.67% (sonda 2) i 80.1753% (sonda 8, wg audytu); GLE 71.68%.
- **Wcześniejszy próg 80** (dokumenty z 1 października): plan żąda rozdzielenia „80 procent zgodności znaku”, zbilansowanej trafności i „80 procent redukcji MSE”. Najlepsze, wszystkie na H1 (jedna aktualizacja): skalar s 80,98 procent znaku i 64,96 redukcji MSE; rzut 8D 78.61% (81.59% po filtrze) i 60.24%.
- **Skill** (`maly_vit_sufit_przewidywalnosci_100\PLAN.md`, 3 października): „prognoza na 100 kroków uczenia lepsza o ponad 85 procent od prognozy „bez zmian””; „85 procent w RMSE to 97,75 procent w MSE”. Plan A16: 85 procent na 10 krokach. Najlepsze: h=100 0,31–0,35; h=10 0,24; h=1 0,68. Z przyszłymi siłami 0,985 na h=10. `WYNIKI_CO_JESZCZE.md` odsyła do docs/23: „85 procent na 1 krok” z siłą paczki.
- **Inne progi:** rozpoznanie ponad 90 (92.83%); „ponad 90 procent pokrycia J128” w SD jako „cel do operacjonalizacji”; cosine>0,85 jako „umowny wskaźnik”.

## C. Co wiemy o ruchu cząsteczek

- **Wielkość:** mediana względnej zmiany na krok 0,023–0,072 (A1); „Ruch nie wygasa do kroku 1000”. W kroku 1 lr wynosi 1,7e-6, a tokeny zmieniają się o 1 procent (BatchNorm).
- **Narzędzie a tło:** narzędzie rusza się mocniej (AUC 0,761), bez swoistego kierunku. Średnia po tokenach i losowy kierunek dają to samo co LDA (A17). Operator z A6 działa podobnie na tle (57.60% wobec 60.24%).
- **Ruch wspólny:** 82.10% energii przyrostów w 8D U500 (fit 501..700, A7); cosinus średnich narzędzia i tła 0.7385. Według `WYNIKI_CO_JESZCZE.md` w pełnych 768 udział ruchu wspólnego to 0,20–0,22: „80 procent ruchu jest własne dla tokenu”. Mediany korelacji reszt między tokenami: 0.558–0.896.
- **Autokorelacja przyrostów:** 0,62–0,88; „0,69 przy opóźnieniu 1, ujemna od opóźnienia 3, minimum −0,24 przy 4, zero od około 10”; suma do opóźnienia 100: 0,56. Trwałość dryfu na 100 kroków jest ujemna (-0,01 do -0,34). κ = 0,03–0,055, czyli połowa odchylenia zanika w 13–23 kroki.
- **Paczka:** 32→320 nie zmniejsza kroku (0,078 → 0,082); acf 1 spada 0,71 → 0,31; prognoza h10 0,20–0,24.
- **AdamW:** „Źródłem drgań jest optymalizator, nie dane”. Pęd pomaga tylko na 1 krok (0,60 → 0,68) i „gaśnie w około 10 kroków”.
- **Niskowymiarowość:** 192/202 osie do 90 % energii ruchu (A1); 8 osi PCA to 41.25 procent wariancji (A12); kąty główne U500/U1000: 27.69–83.95 stopnia, mean cos²=.531748.
- **Sufity:** jak w A17; w pierwszych 100 krokach 0,93–0,96; sufit w RMSE po kroku 100: 0,64–0,70; podłoga 0,45–0,60; D²/σ² 35,9 (kroki 0–100) wobec 4,6–6,8 później.

## D. Tożsamość narzędzia (J128)

- **Wybór:** ranking tylko na train, „abs(mean pairedΔ)/SD pairedΔ”, na 240 parach, k500, surowa średnia ROI ważona pokryciem. K z siatki 1…768, C z .01/.1/1/10. Walidacja wybiera najmniejsze K z BA>0.90: K 128, C 0.01, val 90.67%; K 64 daje 89.31. Bez PCA.
- **Trafność:** BA 92.83%, czułość 91.97%, swoistość 93.68%, AUROC 97.41%, nieważona 91.76%, CI [88.17, 96.99]% (2000 bootstrapów po filmach).
- **Wcześniejsze warianty:** U8 na k1000 88.61%; LDA 95.28%; U500 BA 90.00 procent (ważone obrazami). Osie top-5 na k500: 359, 594, 472, 512, 138; na k1000: 500, 472, 735, 512, 647.
- **Kontrole** (wykonane dla podprzestrzeni U8 na k1000, nie dla natywnych J128):
  - 24 sham z filmów testowych: brak zmiany decyzji binarnej, największe przesunięcie prawdopodobieństwa 0.27341; w 13/24 edycja i maska dzielą token.
  - Trafność per bin kąta 75.0–100.0 procent (2–10 par); leave-angle-out; 19/400 obrazów zmienia bin po resize.
  - Losowe kierunki o dopasowanej normie; noop.
- **Co krytykowały audyty Astry:**
  - Błąd formatu ramek XYWH/XYXY (P1, naprawiony i przeliczony); indeksy int16 hooka.
  - Różna norma po fc2 (8.566478 wobec około 4.5), stąd dodatkowa kontrola.
  - Dolna granica CI poniżej 90; „nie dowód minimalności” K; test wcześniej oglądany; rozpoznanie sprawdzone tylko na k500.
  - Filtr 81.59% to „warunkowa metryka retrospektywna”; kolizja nazwy `selected_epsilon`; rola sondy 2 błędnie nazwana „pierwszą odłożoną”.
  - Zmiana standaryzacji w A7 (dodano kontrolę localU8).
  - Rozbieżność SHA skryptu NNDMD (A-05, rozstrzygnięta snapshotem); liniowy decoder oznacza liniowy rollout po dekodowaniu (A-01).
  - Luka walidacji cache i rozbieżność plan/kod co do beta J128 (A13); błąd wznowienia z early stopping (A14); `LOCAL_RANK` jako indeks CUDA (A15).
- Audytu końcowych metryk A15 oraz audytów A16 i A17: nie znaleziono.

## E. Plany na przyszłość

Katalog: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\plan_na_przyszle_eksperymenty\`

1. **2026-10-01 14:06:21, poprawa równań Langevina i Koopmana.** Test, czy historia paczek i ich wpływów poprawia prognozę s na H=500 (ARX/Koopman z kontrolą, GLE, cztery wejścia; próg 5 procent względnej redukcji MSE; kampania 10 ziaren 57–66). Punkt wyjścia w planie: Langevin 74,2 procent znaku i 28,7 procent mediany redukcji. Wykonano lokalne piloty (A8), hipotezy nie potwierdzono. Kampanii i katalogu `maly_vit_prognoza_stan_i_historyczne_paczki` nie ma.
2. **14:35:31, pełna scena SD.** Retrieval całość do całości (MAE/MSE/SSIM, miary percepcyjne, pełnoobrazowy DINO, DETR-E bez znanego źródła); crop256 jako kontrola. Zależy od kodu pętli SD. Niewykonany.
3. **16:28:13, ruch cząsteczek a detekcja.** Rozdzielenie zmiany sieci od zmiany klatki, związek ruchu z detekcją, prognozy H 10–100. Zależy od 101 checkpointów. Wykonano pilot prognozy (A12). Związku ruchu z detekcją i sekwencji: nie znaleziono.
4. **16:46:21, podprzestrzeń tożsamości.** 400 par, podprzestrzeń, interwencje. Wykonane (A4, A3, A5). Inne narzędzia, drugi model inpaintingu, naturalne negatywy: niewykonane.
5. **2026-10-02 12:49:00, SD z czystego szumu.** Start z szumu VAE, energia L=λ_full·L768+λ_J·L_J128, akceptacja po końcowym RGB; krok 5 to DINO z cosine>0,85 i 0–100 klatek. Zależy od wiarygodnej prognozy. Niewykonany; wcześniejsza pętla używała `start=kk['obraz576']`.
6. **12:59:31, telemetria.** Kroki 6–7: dokładanie prawdziwych klatek, dzienniki selection/batch/checkpoint, walidacja przed i po rundzie. Zależy od kroku 5. „Plan dokumentacyjny”, niewykonany.
7. **13:02:43, test A–F.** Etap minimalny A/B/F, kontrast F−B, 10 sparowanych seedów. Zależy od działającego pipeline F. Niewykonany.
8. **13:06:44, luki.** Osiem hipotez ryzyka: artefakty LaMa, przyszły latent wobec bieżącego ViT, różnorodność retrieval, ruch a uczenie, prognoza jako „wąskie gardło”, osiągalność celu SD, DINO, uczciwe ramiona. „Nie nowy krok 8.”
9. **14:36:52, literatura.** Pięć modeli w kolejności testowania: bezpośredni MLP/TCN, historia AR/GRU, lokalny Koopman, Koopman z pamięcią MZ, model informowany optymalizatorem. Częściowo pokryte przez A14, A15, A17. MLP/TCN i GRU: nie znaleziono.

## F. Foldery research

- **research_koopman_wagi_i_czasteczki** (README, RAPORT, PROTOKOL_POROWNANIA, ZRODLA, `zrodla\` z PDF Dogra–Redman):
  - Najmocniejszy kandydat na zapamiętany artykuł: Dogra i Redman, NeurIPS 2020 (prognoza wag).
  - Tano i in.: wykonanie z DMD trwało „1,41 razy dłużej”. KMPNN: tysiące trajektorii małych sieci. PDT. Liu–Li 2026: AdamW z momentami.
  - Token jest obserwablą pełnego stanu S_t=(θ,m,v,b,…); nie wynika istnienie macierzy 768×768; kontrprzykład z przeciwnym m.
  - Protokół „prognoza wag+BN→forward” wobec tokenów: niewykonany.
  - Raport odnotowuje, że w pętli SD „dodatek prognozy nie wykazał przewagi nad SD prowadzonym do obecnego latentu”.
  - Zakres: 64 rekordy, osiem pełnych odczytów.
- **research_czasteczki_koncepty_biologiczne** (README, RAPORT, PROTOKOL_WALIDACJI, ZRODLA, `zrodla\`):
  - Analogie: komórki konceptu (Quiroga 2005), mieszana selektywność (Rigotti 2013), osie twarzy (Chang i Tsao 2017).
  - „Token ViT nie jest jednak biologicznym neuronem.”
  - Bliski prior art: pattern-CAV, CDLC, InG, MCD, FeatMap 2026, patching.
  - Możliwa nowość tylko w zweryfikowanej kombinacji.
  - LaMa jest deterministyczna.
  - Zakres: 134 rekordy, 18 pełnych odczytów.

## G. Sprzeczności i zastrzeżenia

- **Definicja celu się zmienia:** 80 (1 października), 85 jako znak (2 października), 85 jako skill (3 października).
- **Ruch wspólny:** 82.10% (A7, 8D) wobec około 20 procent (A17, 768D). Dokumenty nie zestawiają tych liczb.
- **Źródło losowości:** werdykt `WYNIKI_CO_JESZCZE.md` mówi „napędzany losowym składem paczek”, a A16: „Drgania nie pochodzą z losowania obrazów do paczki”. Punkt 2 tego pliku oznaczono „obalone”. `WYNIKI_ROWNANIE_Z_PEDEM.md`: „Cały brak przewidywalności to brak siły z przyszłych paczek”.
- **Sufit pierwszych 100 kroków:** 0,93–0,96 („fizycznie możliwy”) później nazwany „zawyżony” (0,11 na 10 ziarnach).
- **GLE a stała prędkość na H1:** w 8D wygrywa GLE (60.24% wobec 47.77%), w J128 stała prędkość (55.92% wobec 47.75%).
- **Nieaktualne statusy:**
  - A14 `README.md`/`WYNIKI.md`: „RUNNING”, podczas gdy `status.json` mówi CANCELLED.
  - A15 `submit_status.json`: RUNNING z 223 krokami, podczas gdy `WYNIKI.md` mówi COMPLETED.
  - PIPELINE kończy się na PENDING.
  - Tytuł PLAN_NNDMD mówi „128”, wykonano 768.
- **Zmiany po teście i błędy obliczeń:**
  - Definicję znaku w EDMD dodano po otwarciu testu (`AMENDMENT_METRYKA_ZNAKU.md`).
  - A17: „Wiersze OU w pierwszym biegu miały błąd znaku (κ = 0); poprawiony bieg trwa”; pierwszy bieg równania z pędem miał błędną aktualizację pędu.
  - `run1819918\RAPORT.md` ma uszkodzone kodowanie znaków.
- **Awarie:** jak w A11, A14, A15 (25/30 FIT, /raid, telemetria, anulowanie, enumeracja CUDA, NCCL).
- **Niedowiedzione według samych dokumentów:**
  - Tożsamość względem innych narzędzi i niezależność od LaMa.
  - Spadek AP50.
  - Białość reszt (Ljung–Box 4084 z 4096; 256/256).
  - Stabilność (promienie 1.001476, 1.005697, 1.033198).
  - Uogólnienie poza jedno ziarno 47 i wielokrotnie oglądany odcinek 800–1000 / 90–100 (wyniki „eksploracyjne”).
  - Wpływ β1 i SGD.
  - Niemożliwość 85 procent z informacją spoza historii latentu.
  - Modele nieliniowe w A17 („Nie sprawdzono”).
