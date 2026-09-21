# 39. Kontrolowany eksperyment doboru klatek: czy sygnał Koopmana i Langevina pomaga uczyć detektor

Data: 2026-09-21. Status: **etap 1 zakończony (27 z 27 biegów, 0 błędów). Reszta Langevina (K1) nie pomaga; pokrycie przestrzeni wolnych modów Koopmana (K4) wyszło najlepiej i przekroczyło próg względem pokrycia wyglądu. Brama do etapu 2 spełniona przez K4, do zatwierdzenia przez autora.**
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

## 5b. Wynik etapu 1 (stan 2026-09-21 19:45, 24 z 27 biegów po 4 h 48 min; K4 w toku)

Walidacja, 3 ziarna treningu na ramię, średnia +- odchylenie między ziarnami. Dowody:
`evidence_selection_stage1_2026-09-21/`. Druga miara: `pipeline/koopman_langevin/misses_by_speed.py`
na 11 940 klatkach walidacji z prędkością z torów GT (progi 4,6 / 9,3 / 19,7 px na 3 klatki).

| Ramię | AP najlepsza epoka | AP ostatnia epoka | średnie AP epok 0-5 | najlepsza epoka | chybienia, najszybsze 5 procent | chybienia, wolna połowa |
|---|---|---|---|---|---|---|
| B0_consecutive | 0,6672 +- 0,0030 | 0,6500 +- 0,0031 | 0,6120 | 16, 16, 15 | 35,96 procent | 6,94 |
| B1_random | 0,6822 +- 0,0011 | 0,6536 +- 0,0107 | 0,6251 | 6, 8, 8 | 33,00 | 6,18 |
| B2_uniform_time | **0,6866** +- 0,0042 | 0,6484 +- 0,0020 | 0,6261 | 7, 7, 7 | 31,77 | 6,17 |
| B3_coverage | 0,6817 +- 0,0013 | 0,6439 +- 0,0023 | 0,6246 | 6, 6, 7 | 32,21 | 7,03 |
| B4_uncertainty | 0,6826 +- 0,0005 | 0,6438 +- 0,0028 | 0,6270 | 8, 6, 9 | 32,33 | 6,36 |
| K1_langevin | 0,6844 +- 0,0020 | 0,6468 +- 0,0042 | 0,6240 | 8, 8, 9 | 31,99 | 6,46 |
| K2_no_history | 0,6859 +- 0,0038 | 0,6479 +- 0,0103 | 0,6243 | 7, 8, 9 | 31,21 | 6,23 |
| K3_placebo | 0,6838 +- 0,0005 | 0,6472 +- 0,0048 | 0,6251 | 8, 8, 8 | 32,16 | **5,64** |
| K4_koopman_cover | **0,6885** +- 0,0019 | 0,6510 | | 4, 6, 5 | **31,26** | **5,29** |

Kontrasty względem progów zapisanych w sekcji 4 (AP z najlepszej epoki):

| Kontrast | Różnica | Próg | Werdykt |
|---|---|---|---|
| K1 - B3 | +0,0027 | +0,004 | poniżej progu |
| K1 - B4 | +0,0018 | +0,004 | poniżej progu |
| K1 - K2 | -0,0015 | +0,004 | brak (zapowiedziana mała moc, 83 procent wspólnych klatek) |
| K1 - K3 (placebo) | +0,0006 | +0,004 | brak: przetasowany sygnał działa tak samo jak prawdziwy |
| chybienia w najszybszych 5 procentach, K1 - B3 | -0,2 punktu | -3 punkty | brak |
| każde - B0 | +0,015 do +0,019 AP; -3 do -5 punktów chybień w szybkim ruchu | | wyraźny efekt |

Odczyt:

1. **Teza „sygnał Langevina poprawia dobór klatek" nie znalazła potwierdzenia przy budżecie 1500 z puli 10 637.**
   Żaden kontrast K1 nie przekroczył progu, a placebo dorównuje prawdziwemu sygnałowi, więc drobna przewaga nad B3
   pochodzi z samego rozluźnienia pokrycia, nie z dynamiki. Brama do etapu 2 (którykolwiek kontrast K nad progiem
   punktowo) NIE jest spełniona; rozstrzygnie ostatecznie K4.
2. **Wszystkie rozsądne sposoby doboru mieszczą się w 0,005 AP**, przy rozrzucie między ziarnami 0,001 do 0,004.
   Najprostszy równy wybór w czasie jest najlepszy lub równy najlepszym. Zgodne z kampanią statyczną i z Mittal 2025.
3. **Jedyny wyraźny efekt: klatki po kolei są gorsze.** Tracą 0,015 do 0,019 AP, gubią narzędzie w szybkim ruchu
   o 3 do 5 punktów częściej i uczą się dwa razy wolniej (najlepsza epoka 16 zamiast 7, niższe AP w epokach 0-5).
   To odpowiada recenzentowi na pytanie, czy dobór w ogóle ma znaczenie: ma, ale wystarcza rozrzucenie klatek.
4. Wynik z ostatniej epoki jest niższy o około 0,035 AP niż z najlepszej i ma większy rozrzut: 30 epok to przeuczenie
   (dokument 38, sekcja 4f). Różnice między ramionami w ostatniej epoce są w całości w szumie.
5. Szybki ruch pozostaje problemem każdego ramienia: około 32 procent chybień w najszybszych 5 procentach klatek
   wobec 6 procent w wolnej połowie. Żaden dobór tego nie ruszył, więc to raczej granica modelu jednoklatkowego
   (rozmycie, wejście narzędzia, błędy adnotacji) niż brak danych.

Co z tego wynika dla dalszej pracy: sygnał Koopmana i Langevina ma na dziś wartość DIAGNOSTYCZNĄ (dokument 38:
położenie w wolnych modach, prędkość z filtru, grube ogony wymuszenia), nie selekcyjną. Miejsca, gdzie selekcja
może jeszcze coś dać, to bardzo małe budżety (100 do 500 etykiet) i pula z wieloma narzędziami; oba wymagają
osobnego zatwierdzenia i nie wynikają z tego etapu.

## 5c. Uzupełnienie po zakończeniu K4 i diagnoza, dlaczego K1 nie pomógł (2026-09-21 wieczór)

**K4.** Kontrast K4 - B3 = **+0,0068 AP** (próg +0,004): jedyny kontrast etapu 1 nad progiem. K4 ma też najmniej chybień
w wolnej połowie klatek (5,29 procent wobec 6,2 do 7,0) i w najszybszych 20 procentach (15,61), a najlepszą epokę osiąga
najwcześniej (4 do 6). Względem równego wyboru w czasie (B2, 0,6866 +- 0,0042) przewaga +0,0019 mieści się w szumie trzech
ziaren. Zastrzeżenia: jedno ziarno wyboru, trzy ziarna treningu, najlepsze z dziewięciu ramion (klątwa zwycięzcy), brak
kontroli wymiaru: K4 pokrywa przestrzeń 16 współrzędnych, więc zysk może pochodzić z samego NISKIEGO WYMIARU pokrycia,
a nie z tego, że to współrzędne Koopmana. Rozstrzyga to dopiero pokrycie w 16 głównych kierunkach PCA i w 16 losowych
rzutach tego samego latentu.

**Dlaczego K1 nie pomógł: trzy pomiary na tym, co ramiona faktycznie wybrały** (`curated/selection_matrix_stage1/`,
prędkość z torów GT puli train; progi 10,2 i 24,6 px na 3 klatki):

| Ramię | udział klatek z najszybszej piątej części | mediana prędkości | klatki wejścia i wyjścia narzędzia | klatki, gdzie L0 jest niepewny | filmy |
|---|---|---|---|---|---|
| B1_random | 0,19 | 4,8 | 0,04 | 0,21 | 118 |
| B2_uniform_time | 0,24 | 5,3 | 0,20 | 0,25 | 118 |
| B3_coverage | 0,34 | 6,4 | 0,19 | 0,39 | 117 |
| B4_uncertainty | 0,36 | 7,0 | 0,21 | 0,69 | 118 |
| K1_langevin | **0,40** | **8,0** | 0,09 | 0,32 | 118 |
| K2_no_history | 0,41 | 8,2 | 0,09 | 0,32 | 118 |
| K3_placebo | 0,25 | 5,5 | 0,05 | 0,27 | 117 |
| K4_koopman_cover | 0,25 | 5,3 | 0,09 | 0,24 | 116 |
| B0_consecutive | 0,22 | 5,3 | 0,04 | 0,17 | **58** |

1. **Sygnał zadziałał zgodnie z projektem.** K1 kupił dwa razy więcej klatek szybkiego ruchu niż losowanie (0,40 wobec 0,19).
   Równanie nie jest więc „za słabe"; wybrało to, w co celowało.
2. **Ale te klatki nie leczą błędu, w który celowały.** Dwa razy więcej przykładów szybkiego ruchu w treningu nie zmieniło
   chybień w szybkim ruchu na walidacji (około 32 procent w każdym ramieniu). To samo z niepewnością: B4 kupił 69 procent
   klatek, na których detektor L0 był niepewny, i nie zyskał nic. Błąd detektora w tych miejscach NIE jest błędem z braku
   przykładów. W języku twierdzenia o rozkładzie błędu: duża reszta to tutaj część nieusuwalna etykietami, nie usuwalna.
   Opóźnienia adnotacji w szybkim ruchu nie ma (przesunięcie predykcji wzdłuż ruchu: mediana -0,8 px); przy chybieniu
   detektor jest pewny siebie (wynik 0,80 wobec 0,89 przy trafieniu), więc trafia w coś innego niż oznaczone narzędzie.
3. **Liczy się liczba etykiet i rozrzut po operacjach, nie to, które klatki.** Krzywa z kampanii statycznej: 500 klatek 0,646,
   1500 klatek 0,682, 2500 klatek 0,693. Dołożenie tysiąca DOWOLNYCH rozrzuconych klatek daje +0,036, a wybór między nimi
   najwyżej +0,007. Ramię B0 pokryło 58 filmów zamiast 118 i straciło 0,017. Usuwalna część błędu leży w różnicach między
   operacjami i jest rozłożona równo, nie siedzi w zdarzeniach dynamicznych.

Wniosek: poprawianie równania Langevina albo głębszy autoenkoder Koopmana NIE zmieni wyniku K1, bo wąskim gardłem nie jest
jakość sygnału, tylko to, że wskazane klatki nie mają usuwalnego błędu. Dynamika pomaga tam, gdzie opisuje STANY do
pokrycia (K4), a nie ZASKOCZENIA do kupienia (K1).

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
