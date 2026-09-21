# 39. Kontrolowany eksperyment doboru klatek: czy sygnał Koopmana i Langevina pomaga uczyć detektor

Data: 2026-09-21. Status: **plan zatwierdzony przez autora 2026-09-21; etap 1 zgłoszony na Eden; wyników brak.**
Powiązania: plan [37](37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md) (ten dokument realizuje jego etap D
w wersji pilotażowej i rozszerza o ramiona B0, K3, K4), pilot [38](38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md),
uwagi recenzentów [04_REVIEWER_MAP](../04_REVIEWER_MAP.md).
Kod: `pipeline/koopman_langevin/selection_matrix.py`, `pool_confidence.py`; zadania
`eden/run_selection_stage1_prep.slurm`, `eden/run_selection_stage1_worker.slurm` (robotnik z jedną kartą),
`eden/run_selection_stage1.slurm` (wariant na jednym węźle z czterema kartami, nieużyty), `eden/build_stage1_queue.py`.

## 0. Czego ten eksperyment nie obiecuje

Żaden eksperyment nie da „stu procent". Ten jest ułożony tak, żeby wynik rozstrzygał w obie strony: potwierdzenie,
obalenie albo wynik nierozstrzygający, każdy z nazwanym powodem. Punkt wyjścia jest trudny: w kampanii statycznej
(dokument 35) wszystkie rozsądne selektory mieściły się w pół punktu AP od losowania przy budżetach 500 do 2500.

## 1. Na co odpowiada (uwagi recenzentów)

| Uwaga | Żądanie | Odpowiedź w tym planie |
|---|---|---|
| A6 | powtarzane losowanie przy tym samym budżecie | ramię B1, te same ziarna co pozostałe |
| A5, B4 | pełna macierz ablacji przy stałych warunkach treningu | dziewięć ramion, jedna receptura, jeden wspólny start |
| A9, B2 | powtórzenia i niepewność | etap 1: 3 ziarna; etap 2: 5 ziaren treningu x 3 ziarna wyboru; bootstrap po operacjach |
| A2, B1 | podział po operacjach | podział 117/25/25 grup operacji z kampanii statycznej; test otwierany raz |
| A13, A14 | czułość parametrów i alternatywne selektory | etap 3 dla zwycięskiego ramienia |

## 2. Wspólne warunki

- Start L0: `tracking_uniform_time_b500_s42` (500 klatek), identyczny dla wszystkich ramion; każde ramię dobiera 1000.
- Receptura kampanii statycznej: RF-DETR Nano, wagi COCO, wejście 384, 30 epok, profil `nano_static_pilot`,
  ten sam protokół `protocol_tracking_calibration.yaml` i to samo archiwum danych (sumy kontrolne w `*.inputs.sha256`).
- Ziarna treningu 42, 1337, 2024; ziarno wyboru 42.
- Wybór nie czyta etykiet. Modele Langevina dopasowane na nieoznaczonej puli train (transdukcyjnie, jawnie).
- Ocena na walidacji. Zbiór testowy nie jest dotykany w etapie 1.

## 3. Ramiona

| ID | Jak dobiera 1000 klatek | Po co |
|---|---|---|
| B0_consecutive | ciągi sąsiednich klatek od losowego miejsca filmu | punkt odniesienia autora: klatki po kolei |
| B1_random | losowo | żądanie recenzenta A6 |
| B2_uniform_time | równo w czasie | redundancja sąsiednich klatek |
| B3_coverage | pokrycie przestrzeni wyglądu DINOv2 | mocny selektor bez dynamiki |
| B4_uncertainty | B3 zmieszane pół na pół z `1 - pewność` detektora nauczonego na L0 | standard aktywnego uczenia |
| K1_langevin | B3 zmieszane pół na pół z resztą równania Langevina Z HISTORIĄ (8 kroków) | propozycja |
| K2_no_history | to samo, model widzi tylko bieżący latent, ta sama liczba cech nieliniowych | ablacja historii |
| K3_placebo | to samo, oceny K1 przetasowane między klatkami filmu | ablacja samego sygnału |
| K4_koopman_cover | pokrycie w 8 najwolniejszych funkcjach własnych Koopmana zamiast w wyglądzie | pokrycie przestrzeni stanów |

B4, K1, K2, K3 przechodzą przez tę samą funkcję `select_batch` z tymi samymi wagami; różni je tylko ocena pomocnicza,
zamieniona na rangę percentylową wewnątrz grupy operacji.

## 4. Kontrasty i kryteria zapisane PRZED wynikiem

| Kontrast | Pytanie | Próg |
|---|---|---|
| K1 - B3 | czy dynamika dodaje coś ponad wygląd | +0,004 AP |
| K1 - B4 | czy bije zwykłe aktywne uczenie | +0,004 AP |
| K1 - K2 | czy zasługa należy do historii | +0,004 AP |
| K1 - K3 | czy działa sygnał, a nie samo domieszanie drugiego kryterium | +0,004 AP |
| K4 - B3 | czy pokrycie przestrzeni Koopmana bije pokrycie wyglądu | +0,004 AP |
| każde - B0 | zysk nad klatkami po kolei | opisowo |

Miara główna: AP@[0.50:0.95] na walidacji z `last_epoch`; `best_val` osobno. Miara druga: udział chybień (IoU < 0,5)
w najszybszych 5 i 20 procentach klatek walidacji według prędkości z torów GT (dziś 14 procent wobec 3 procent w wolnych;
próg: spadek o 3 punkty procentowe). Miara trzecia: AP po każdej epoce, czyli szybkość uczenia (pole pod krzywą epok).

Efekt dodatni: oszacowanie punktowe nad progiem ORAZ 95-procentowy przedział sparowanej różnicy (bootstrap po całych
operacjach, 2000 replik, ziarno 20260921) powyżej zera, z korektą Holma dla czterech kontrastów K1. Sukces tezy
„Langevin pomaga" wymaga wszystkich czterech kontrastów K1; wynik częściowy nazwać po pokonanym ramieniu.
Przy trzech ziarnach etap 1 jest PILOTEM: rozstrzyga, czy warto iść do etapu 2, nie zamyka tezy.

## 5. Etapy

1. **Etap 1 (zgłoszony 2026-09-21):** przygotowanie na 1 karcie A100 (pewność detektora L0 na puli, budowa ramion,
   wspólna kolejka 27 biegów; zadanie 1809149), potem pięciu robotników po jednej karcie, którzy biorą biegi z tej samej
   kolejki: cztery na A100 (zadania 1809150 do 1809153) i jeden na H100 (1809154). Autor zniósł dla tego etapu zasadę
   jednego zadania naraz. Szacunek: około 8 godzin.
2. **Etap 2 (tylko gdy któryś kontrast K przekroczy próg punktowo):** B1, B2, B3, B4, K-zwycięzca, K2, K3;
   5 ziaren treningu x 3 ziarna wyboru; budżety 750, 1000, 1500, 2500 dla krzywej „ile etykiet trzeba";
   na końcu jednorazowa ocena na teście.
3. **Etap 3:** czułość zwycięskiego ramienia: waga domieszki 0,25 / 0,5 / 0,75, historia 2 / 4 / 8 kroków,
   latent DINOv2 kontra latent detektora (`detector_latent.py`), strata odporna na skoki.

## 5a. Stan po przygotowaniu (2026-09-21, zadanie 1809161, 12 minut na A100)

Pierwsze dwa zgłoszenia przygotowania (1809149, 1809155) padły po 5 sekundach: `koopman_langevin/select.py` przesłaniał
moduł `select` biblioteki standardowej, gdy `pool_confidence.py` szedł jako skrypt. Zadania zależne anulowały się same,
żaden trening nie ruszył. Poprawka: flaga `-P` interpretera. Trzecie zgłoszenie przeszło; pięciu robotników
(1809162 do 1809166; dgx-2, dgx-4, hopper) bierze biegi ze wspólnej kolejki 27 zadań.

Pula train do doboru ma 10 637 klatek. Część wspólna dobranych 1000 klatek z ramieniem K1:

| B0 | B1 | B2 | B3 | B4 | K2 | K3 | K4 |
|---|---|---|---|---|---|---|---|
| 0,11 | 0,11 | 0,10 | 0,46 | 0,33 | 0,83 | 0,26 | 0,17 |

Równanie z historią przewiduje 18,4 procent kroku latentu na puli, bez historii 4,6 procent. Mimo tej różnicy K1 i K2
wybierają w 83 procentach TE SAME klatki: oba modele uznają za trudne te same miejsca filmu. **Kontrast K1 - K2 ma więc
z góry małą moc** i prawie na pewno wyjdzie nierozstrzygający; zapisane przed wynikiem. Rozstrzygające pozostają
K1 - B3, K1 - B4 i K1 - K3. Żeby zbadać historię uczciwie, etap 3 potrzebuje oceny, która RÓŻNI oba modele
(na przykład zysk z historii na klatkę zamiast samej reszty).

## 6. Kontrargumenty

- Trzy ziarna nie wystarczą do twierdzeń; rozrzut między ziarnami w kampanii statycznej to około 0,002 AP.
- Reszta jest retrospektywna (używa bieżącej klatki); to dozwolone przy doborze z gotowej puli, ale nie jest prognozą.
- Duża reszta to w praktyce szybki ruch i wejście narzędzia; K1 może więc wygrać tylko w warstwie szybkiego ruchu.
- Część najszybszych klatek to błędy adnotacji; K1 może kupować szum etykiet (Hu i in. 2021).
- B0 jest słabym punktem odniesienia; wygrana tylko z nim niczego nie dowodzi.
- Pewność L0 pochodzi z jednego ziarna detektora; B4 może być przez to zaniżone.

## 7. Data gaps

- Brak linii bazowej RHO-loss (wymaga etykiet odłożonych; poza etapem 1).
- Brak wariantu z latentem detektora w K1 (etap 3).
- Identyfikacja pacjentów niepotwierdzona; jednostką resamplingu jest grupa operacji.
