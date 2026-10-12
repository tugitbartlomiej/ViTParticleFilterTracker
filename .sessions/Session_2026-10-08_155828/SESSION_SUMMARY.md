# Session Summary: Session_2026-10-08_155828

## Metadata

- Date/time: 2026-10-08 15:58:28 Europe/Warsaw.
- Status: Completed.
- Type: Training / Benchmark / SSH.
- Workspace: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker`.
- Eden root: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256`.

## Objective

Użytkownik zlecił podmianę jednego obrazu w małej puli treningowej naturalnymi obrazami znalezionymi dla 11 generacji interpolujących od sztucznego do oryginalnego latentu m0011. Każda podmiana miała prowadzić do następnego checkpointa, a pomiar miał pokazać zmianę IoU i pewności detekcji.

## Design

- Start: identyczny checkpoint `MALY_VIT_K8000_s47_losowe.pth`, SHA256 `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`.
- Dziewięć gałęzi: kontrola plus osiem unikalnych obrazów. Trzy z 11 zapytań DINO wskazywały te same obrazy, więc nie liczono ich ponownie.
- Pula każdej gałęzi: 255 identycznych obrazów plus jeden zmieniany slot.
- Trening: 500 aktualizacji, batch 32, AdamW, learning rate 2e-5, weight decay 0.05, seed 20261008.
- Pomiar sondy m0011 co 25 aktualizacji: top confidence, IoU ramki o najwyższej pewności, najlepsze IoU spośród 10 ramek i odpowiadająca pewność.
- Pomiar ogólny: AP50 na 512 obrazach z 23 filmów rozłącznych z pulą treningową i filmami kandydatów.
- Najważniejsza estymata efektu: difference-in-differences, czyli przyrost wariantu minus przyrost kontroli. Sam dalszy trening silnie poprawia wyniki, a wykonanie BF16 na różnych GPU dawało małe różnice już w kroku zero.

## Execution

- Lokalny smoke 2 kroków przeszedł.
- Eden smoke: SLURM `1824689`, `COMPLETED 0:0`.
- Produkcyjna tablica: SLURM `1824697`, zadania 0-8, maksymalnie cztery równolegle; wszystkie `COMPLETED 0:0`.
- Agregacja: SLURM `1824702`, `COMPLETED 0:0`.
- Zweryfikowano SHA256 wszystkich dziewięciu checkpointów na Edenie.
- Pobrano raporty i logi; lokalny pakiet podsumowania ma SHA256 `3945c8cff7e306ddc9173aca1582d93ceb1f61fa0ebf9180b2e8d21cb7a6b8d9`.
- Pobrano kontrolę i najlepszy kompromis; oba checkpointy wczytano lokalnie z `strict=True`.

## Results

Kontrola po 500 krokach:

- best IoU: 0.3980 -> 0.8559, przyrost +0.4579;
- top confidence: 0.8449 -> 0.8587, przyrost +0.0138;
- guard AP50: 0.6645 -> 0.7192, przyrost +0.0547.

Efekt podmiany ponad kontrolę:

| wariant | obraz | IoU | confidence | guard AP50 |
|---|---|---:|---:|---:|
| 01 | TR_0126_S1_P03/1226 | +0.0129 | +0.0016 | -0.0031 |
| 02 | TR_0122_S1_P03/1480 | -0.0022 | -0.0098 | -0.0009 |
| 03 | TR_0122_S1_P03/1743 | -0.0022 | -0.0059 | -0.0095 |
| 04 | TR_0122_S1_P03/915 | +0.0015 | -0.0038 | -0.0028 |
| 05 | TR_0007_S1_P03/1165 | +0.0081 | +0.0000 | +0.0077 |
| 06 | TR_0007_S1_P03/1397 | -0.0312 | -0.0791 | -0.0029 |
| 07 | TR_0079_S1_P03/829 | -0.0022 | -0.0059 | -0.0020 |
| 08 | TR_0122_S1_P03/822 | +0.0007 | -0.0138 | -0.0024 |

Interpretacja:

- `TR_0007_S1_P03/1165` jest jedynym kandydatem dodatnim jednocześnie dla sondy IoU i guard AP50.
- `TR_0126_S1_P03/1226` najlepiej poprawia sondę, lecz wypada gorzej od kontroli na guard AP50. To celowana korzyść bez dowodu lepszej generalizacji.
- `TR_0007_S1_P03/1397` jest jednoznacznie niekorzystny dla sondy.
- Pozostałe efekty są niewielkie i przy jednym seedzie nierozstrzygające.

## Limitations and next step

- To jeden seed, jedna pozycja podmieniana i jedna pula 255 obrazów wspólnych.
- Różnice kilku tysięcznych mogą wynikać z wariancji numerycznej i losowości treningu.
- Najlepszy następny test to kontrola, wariant 05, wariant 01 i wariant 06 na co najmniej trzech seedach oraz dwóch lub trzech niezależnych pulach bazowych.
- Nie ma podstaw, aby wybierać obraz wyłącznie na podstawie cosine DINO; obraz o wysokim cosine mógł pomagać albo szkodzić.

## Follow-up: eight simultaneous replacements

Po teście pojedynczych obrazów wykonano drugie porównanie. Kontrola zawierała 256 oryginalnych obrazów. Wariant usunął deterministycznie osiem pozycji i wstawił wszystkie osiem unikalnych obrazów DINO. Obie gałęzie wykonały 500 aktualizacji kolejno na tym samym GPU od identycznego K8000.

- Eden job `1824755`, `COMPLETED 0:0`, czas 6 minut 8 sekund.
- Kontrola: IoU 0.8552, confidence 0.8578, guard AP50 0.7235.
- Osiem podmian: IoU 0.8644, confidence 0.8300, guard AP50 0.7182.
- Efekt ponad kontrolę: IoU +0.0091, confidence -0.0277, guard AP50 -0.0052.
- Najwyższe confidence x IoU: kontrola 0.7359 w kroku 425; wariant 0.7175 w kroku 500.

Wariant poprawił geometrię ramki na pojedynczej sondzie, lecz obniżył jej pewność oraz jakość na odłożonych filmach. Nie jest więc lepszą ogólną pulą treningową w tej konfiguracji.

## Follow-up: comparison of full natural latents

Porównano naturalny forward pełnego latentu `600 x 768` dla trzech modeli na tych samych wejściach: startowego K8000, kontroli po 500 aktualizacjach oraz wariantu po 500 aktualizacjach z ośmioma podmianami. Pomiar objął sondę m0011, osiem dodanych obrazów, osiem usuniętych obrazów i 512 obrazów z 23 odłożonych filmów. Nie zapisywano pełnych tablic latentów; zapisano wyłącznie zagregowane metryki i wartości per obraz.

- Na 512 odłożonych obrazach zwykły trening `K8000 -> kontrola` dał średni RMSE latentu 0.14016 i cosine 0.96819.
- Trening `K8000 -> wariant 8` dał średni RMSE 0.14355 i cosine 0.96706.
- Czysty efekt podmiany, `kontrola -> wariant 8`, dał średni RMSE 0.02451 i cosine 0.99892.
- Dodatkowy ruch wywołany podmianą miał średnio 18.16 procent normy ruchu kontrolnego.
- Na samych ośmiu dodanych obrazach efekt był 4.43 razy większy niż średnio na zbiorze odłożonym: RMSE 0.10865.
- Na sondzie m0011 efekt wyniósł RMSE 0.03723, około 93. percentyl rozkładu odłożonych obrazów. Wewnątrz ramki narzędzia RMSE wyniósł 0.03431, a poza ramką 0.03735, więc zmiana nie skoncentrowała się przestrzennie w obszarze narzędzia.
- Najsilniejsze 128 z 768 osi wybranych po pomiarze niosło 67.07 procent energii dodatkowej zmiany. Nie są to automatycznie wcześniejsze osie J128.

Wniosek: latent rzeczywiście się zmienił wskutek ośmiu podmian, ale większość całego ruchu pochodziła ze wspólnych 500 aktualizacji. Podmiana dodała mniejszy, odrębny składnik reprezentacji. Nie poprawił on generalizacji detektora: guard AP50 spadło względem kontroli o 0.0052.

## Files

- Kod: `SmallViTAblation/experiments/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/`.
- Główny raport: `SmallViTAblation/outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/results/ANALIZA.md`.
- Wykres: `SmallViTAblation/outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/results/EFEKT_PODMIANY.png`.
- Tabele: `EFEKT_PODMIANY.csv`, `EFEKT_11_KROKOW.csv`, `WYNIKI.md`, `COMPARISON.json`.
- Wybrane modele: `SmallViTAblation/outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/selected_checkpoints/`.
- Drugi eksperyment: `SmallViTAblation/experiments/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/`.
- Wyniki ośmiu podmian: `SmallViTAblation/outputs/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/results/`.
- Raport latentów: `SmallViTAblation/outputs/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/results/latent_comparison/LATENT_ANALIZA.md`.
- Dane latentów: `LATENT_COMPARISON.json`, `LATENT_PER_IMAGE.csv` i `LATENT_POROWNANIE.png` w tym samym katalogu.

## Follow-up: kierunek względem zgodnego sztucznego celu K8000

Stary cel m0011 pochodził z K1000, dlatego nie był zgodnym punktem odniesienia dla gałęzi startujących z K8000. Cel wyznaczono ponownie na dokładnym checkpointcie K8000 użytym w obu biegach. Sztuczna korekta poprawiła IoU z 0.05063 do 0.99720 i confidence z 0.84576 do 0.90505.

- Startowy K8000 miał RMSE 1.083592 do zgodnego celu.
- Kontrola po 500 aktualizacjach miała RMSE 1.121694.
- Wariant z ośmioma podmianami miał RMSE 1.121515.
- Podmiany dały względem kontroli poprawę RMSE tylko 0.000180.
- Cosine czystego ruchu podmiany z kierunkiem do celu wyniósł 0.02140.
- Rzut ruchu na kierunek celu odpowiadał 0.07088% pozostałej drogi.

Wniosek: względem kontroli pojawiła się minimalna dodatnia składowa w stronę sztucznego celu, lecz obie gałęzie po treningu były dalej od celu niż startowy K8000. Nie ma dowodu, że trening nauczył się kierunku sztucznego latentu.

- Raport: `SmallViTAblation/outputs/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/results/latent_comparison/compatible_k8000_target/DIRECTION_TO_COMPATIBLE_K8000_TARGET.md`.
- Dane: `DIRECTION_TO_COMPATIBLE_K8000_TARGET.json` i wykres `DIRECTION_TO_COMPATIBLE_K8000_TARGET.png` w tym samym katalogu.
