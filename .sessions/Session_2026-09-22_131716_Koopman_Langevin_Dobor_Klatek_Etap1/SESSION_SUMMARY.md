# SESSION SUMMARY - Koopman i Langevin na latencie detektora narzedzi (ViT), etap 1 doboru klatek

Typ: experiment

## Cel sesji
Sprawdzic, czy aparat z projektu grantowego (czastki latentu, rozszerzone rownanie Langevina, operator Koopmana,
rozklad bledu na czesc usuwalna i zaskoczenie) opisuje latent detektora narzedzi chirurgicznych i czy pomaga
dobierac klatki filmu do uczenia. Odpowiedziec na zarzut recenzentow o za malo ablacji i ziaren.

## Kontekst
- Branch grantowy: `feature/koopman-tft-experiments` (bez zmian merytorycznych w tej sesji).
- Branch ViT: `codex/eden-nano-training-20260914` w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`,
  13 commitow od `37a684f8` do `b7f9438` (wszystkie z tej sesji).
- Stan Edena: 102 GB z 200 zajete; katalog roboczy `~/smallvit_runs/2026-09-21_koopman_langevin`
  (paczka 16 klipow, checkpointy, latenty, 27 wynikow etapu 1, kolejka `queue/stage1`). Nic nie biegnie.
- Powiazane sesje: `.sessions/Session_2026-09-20_211814_diagnoza_rownania_lean_czlon_zdarzen_optymalizacja_95`
  (rownanie i twierdzenia Lean, z ktorych ten aparat pochodzi).
- Rownolegla sesja (inny agent) dodala w repo ViT plan `37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md`;
  nie ruszany, moj dokument dostal numer 38.

## Wykonane kroki
1. Eden: sprawdzenie miejsca (198,9 GB, potem 87 GB po sprzataniu przez uzytkownika) i katalogu `DETR` (41 GB, dubel
   13 GB `Checkpoints/ablation` = `ablation_checkpoints`). Porownanie z kopia lokalna `Eden\EdenBackup\DETR`:
   4745 plikow zgodnych; pobrano 2 brakujace wagi (epoki 100 i 110), dluzszy log i plik zdarzen, sumy SHA256 zgodne.
   Stare krotsze wersje zachowane z przyrostkiem `.stara_kopia_2026-07-22`.
2. Przeglad literatury (podagent 1, Undermind + Scite, 21 prac w Scite): predkosc odczytywalna tylko z modeli
   WIDEO (Garrido 2025, Joseph 2026); Koopman na wideo uczony razem z siecia (Comas 2021, 2023); Koopman dla
   wag i dynamiki uczenia (Redman 2024, Kim 2026); RHO-loss (Mindermann 2022) jako pierwowzor rozdzialu
   usuwalne/nieusuwalne; Hu 2021 i Karamcheti 2021 o porazkach doboru wg straty; MEMnets (Liu 2025).
   Nowe: Koopman i GLE na latencie gotowej sieci sledzone miedzy checkpointami.
3. Kod diagnostyczny w `pipeline/koopman_langevin/`: `clip_package.py` (16 gestych klipow, 4800 klatek, bez
   testowych), `detector_latent.py` (trzy wektory na klatke: query, box_pool, global_pool, plus ramka i pewnosc),
   `epoch_dynamics.py` (sondy polozenia, rozmiaru, osi i predkosci z kontrola z samych ramek, Langevin z podloga
   z permutacji, wolne mody Koopmana kontra PCA, Procrustes i CKA miedzy checkpointami). Opcja trenera
   `--keep-epoch-checkpoints N`. Zadania Eden `run_detector_latent.slurm`, `run_epoch_series.slurm`.
4. Os budzetu (zadanie 1809131): przed uczeniem R2 polozenia okolo 0, po uczeniu 0,68 do 0,86; wolne mody
   Koopmana niosa polozenie 0,73 do 0,87 wobec PCA 0,04 do 0,45; predkosci nie czyta zadne ramie (sufit
   z torow GT 0,18 na kroku 3 klatek); przewidywalnosc kroku latentu SPADA z 31 do 21 procent (latent dziedziczy
   przewidywalnosc narzedzia).
5. Os epok (zadanie 1809133, wagi co 3 epoki): cala przebudowa latentu w pierwszych 3 epokach; AP walidacji
   rosnie do epoki 5 (0,687) i spada do 0,656 w epoce 29 (przeuczenie).
6. Predkosc jako stan ukryty: `latent_velocity.py` (korelacja 0,33 przy oknie 1, 0,62 przy 8) i `kalman_state.py`
   (model polozenie-predkosc-przyspieszenie z kolorowym wymuszeniem, strata Hubera, obciecie innowacji): z ramek
   korelacja 0,67, R2 0,44; z latentu 0,31; innowacje z kurtoza 25 do 132 (rzadkie szarpniecia). Innowacja wskazuje
   chybienia z AUC 0,60 do 0,68, pewnosc detektora 0,81 do 0,83. Podagent 2: Lusch 2018 (autoenkoder bez ograniczen
   nieinterpretowalny), Proctor 2018 (wymuszenie psuje mody), Becker 2019 RKN, HAVOK.
7. Warstwa predkosci: detektor gubi narzedzie w 14 procentach najszybszych 5 procent klatek wobec 3 procent
   w wolnych; wiecej klatek rownomiernych tego nie zmienia.
8. Uzytkownik przerwal uruchomienie generatora ramion: najpierw plan, potem zatwierdzenie. Zapisana pamiec
   `feedback-plan-przed-treningiem.md`. Plan zatwierdzony; stary test 12 biegow (1809135) skasowany.
9. Etap 1 (`selection_matrix.py`, `pool_confidence.py`, `build_stage1_queue.py`; zadania prep 1809161 i pieciu
   robotnikow 1809162 do 1809166 na dgx-2, dgx-4, hopper; dwa pierwsze zgloszenia prep padly na przeslanianiu
   modulu `select` przez `koopman_langevin/select.py`, poprawka flaga `-P`): 9 ramion z jednego L0 (500 klatek
   `uniform_time`) x 3 ziarna, budzet 1500, 27 treningow w 5 godzin, 0 bledow. AP najlepszej epoki: klatki po kolei
   0,667; losowanie 0,682; rowno w czasie 0,687; pokrycie wygladu 0,682; niepewnosc 0,683; Langevin z historia
   0,684; bez historii 0,686; placebo 0,684; pokrycie modow Koopmana 0,689. Zaden kontrast K1 nad progiem 0,004;
   K4 - B3 = +0,0068 (jedyny nad progiem). Chybienia w szybkim ruchu okolo 32 procent w kazdym ramieniu.
10. Diagnoza K1: kupil 2x wiecej klatek szybkiego ruchu niz losowanie (0,40 wobec 0,19), a chybien nie zmienil;
    B4 kupil 69 procent klatek z niepewnym L0 i tez nic. Liczy sie liczba etykiet i rozrzut po operacjach
    (B0: 58 filmow zamiast 118, strata 0,017).
11. Sekcja K4 (`koopman_cover_analysis.py`, `misses_by_speed.py` z bootstrapem po 25 operacjach): wspolrzedne
    Koopmana latentu calej klatki to tozsamosc operacji (72 procent wariancji miedzy filmami, 0 ze stanu narzedzia);
    K4 - B3 chybienia -1,42 pkt, przedzial [-2,73; -0,26]; K4 - B2 przedzial dotyka zera; kontrole wymiaru
    (PCA16, losowe rzuty) wybieraja inne klatki; szum ziarna WYBORU niezmierzony.
12. Pilot "czego sieci brakuje" (`koopman_defect_pilot.py`): usuwalne = 23 procent chybien L0; AUC usuwalnego bledu:
    pewnosc 0,928, dryf miedzy checkpointami 0,864, reszta Koopmana w wolnych modach 0,839, reszta Langevina 0,798,
    reszta minus ruch ramki 0,689; pewnosc + dryf 0,935; w gornych 10 procentach tylko 7 procent klatek usuwalnych.
13. Widmo Koopmana (`koopman_spectrum_axis.py`): niezmienniki 2 przed uczeniem, 4 do 5 po; wariancja miedzy filmami
    0,74 do 0,81; R2 ze stanu narzedzia 0,58 (epoka 2) i 0,29 (2500 klatek); defekt poza proba 0,19 -> 0,05.
    Mody obracajace: okres okolo 7 klatek, brak zwiazku z narzedziem. Liniowy operator: 17 procent przy 8 wymiarach
    i 8 opoznieniach, ujemny od 24 wymiarow; nieliniowy z historia 21 procent w kazdym wymiarze.
14. Dokumenty w repo ViT: `38_KRYSTALIZACJA_...` (sekcje 4a do 4g), `39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md`
    (sekcje 5a do 5e), `INDEKS.md`, katalogi dowodow `evidence_koopman_langevin_2026-09-21`,
    `evidence_selection_stage1_2026-09-21`. Wszystko zacommitowane (13 commitow).

## Pliki zmienione / utworzone
### W projekcie (Grant/Projekt)
- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_131716_koopman_langevin_vit_dobor_klatek_etap1\` - ten zapis.
- Brak zmian merytorycznych w projekcie grantowym; katalogi `experiments\tft_langevin_*` i `tft_deep_koopman_mz`
  tylko przegladane jako zrodlo rownania.

### Zewnetrzne (poza projektem)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\` - nowe:
  `clip_package.py`, `detector_latent.py`, `epoch_dynamics.py`, `latent_velocity.py`, `kalman_state.py`,
  `selection_matrix.py`, `pool_confidence.py`, `misses_by_speed.py`, `koopman_cover_analysis.py`,
  `koopman_defect_pilot.py`, `koopman_spectrum_axis.py`, `tests\test_epoch_dynamics.py`, `tests\test_kalman_state.py`;
  zmienione: `latent_langevin.py` (ramie bez historii).
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\train\train_rfdetr.py`,
  `run_with_validation.py` - opcja `--keep-epoch-checkpoints`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\` - `run_detector_latent.slurm`,
  `run_epoch_series.slurm`, `run_langevin_selection.slurm` (nieuzyty), `run_selection_stage1.slurm` (nieuzyty),
  `run_selection_stage1_prep.slurm`, `run_selection_stage1_worker.slurm`, `build_stage1_queue.py`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\` -
  `38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md`, `39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md`,
  `INDEKS.md`, `evidence_koopman_langevin_2026-09-21\`, `evidence_selection_stage1_2026-09-21\`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\curated\selection_matrix_stage1\` - listy
  klatek 9 ramion i pewnosc L0.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\detector_probe_clips\` (420 MB,
  niewersjonowane) i `results\koopman_langevin_epochs\` (latenty, raporty; niewersjonowane).
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup\DETR\` - uzupelniona kopia (4 pliki).
- Eden `~/smallvit_runs/2026-09-21_koopman_langevin/` - paczka, checkpointy, latenty, wyniki 27 biegow.
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-plan-przed-treningiem.md` - nowa pamiec.

## Otwarte problemy
- Przewaga K4 (+0,0068 AP nad pokryciem wygladu) ma jedno ziarno wyboru i jest najlepszym z 9 ramion; kontrole
  wymiaru (PCA16, losowe rzuty, rzut 24) nie byly trenowane; przyczyna zysku nieustalona.
- Kontrast K1 - K2 (historia) z gory bez mocy: 83 procent wspolnych klatek; potrzebna ocena rozniaca oba modele.
- Sygnaly z dynamiki latentu wskazuja usuwalny blad slabiej niz pewnosc detektora; dobor wg brakow sieci ma sufit
  23 procent i precyzje 7 procent.
- Szybki ruch: 32 procent chybien w kazdym ramieniu; nie jest to brak danych, tylko granica modelu jednoklatkowego.
- Latent przegrywa z wlasna ramka detektora w odczycie predkosci (blad polozenia 11 px wobec 3 px).
- Zastrzezenia planu 37 do kodu (skalowanie na calej puli, retrospektywna ocena, numer zapytania jako tozsamosc)
  nieusuniete; pomiary maja status pilota.
- Autoenkoder Koopmana nie byl uczony; czlon zdarzen z rownania grantowego nieprzeniesiony.

## Nastepne kroki
- Etap 2 (wymaga zatwierdzenia): B1, B2, B3, K4 plus kontrole PCA16, losowe rzuty, rzut 24 i K4 na latencie
  detektora; 5 ziaren treningu x 3 ziarna wyboru; budzety 750 i 1500; 12 epok zamiast 30.
- Autoenkoder Koopmana na czastkach (wymiary 8, 16, 32, 64) z zamrozona glowica polozenia; prog do pobicia:
  filtr Kalmana z ramek (korelacja 0,67) i model nieliniowy (21 procent).
- Koopman jako cel uczenia (spojnosc latentow sasiednich klatek pod jednym operatorem), bo tam lezy zapas.
- Miara krystalizacji co epoke: defekt niezmiennikow poza proba; zapisy wag gesciej w epokach 0 do 3.
- Przeniesc czlon zdarzen (wejscie i wyjscie narzedzia) i wlasna skale z rownania grantowego.

---

Uwaga o miejscu zapisu: ta sesja dotyczy pracy w repozytorium ViT i tutaj jest jej miejsce docelowe.
Kopia tego samego opisu lezy tez w archiwum projektu grantowego
(`F:\Studia\MCHTR\Grant\Projekt\.sessions\`), bo tam biegl zapis pierwotny.
