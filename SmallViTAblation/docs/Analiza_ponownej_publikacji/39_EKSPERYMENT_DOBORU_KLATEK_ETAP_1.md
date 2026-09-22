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

## 5d. Co w ramieniu Koopmana naprawdę działa i jaką przewagę daje (analiza bez treningu, 2026-09-21)

Kod: `pipeline/koopman_langevin/koopman_cover_analysis.py`; wynik `REPORT_koopman_cover.json`, `MISSES_PER_VIDEO_best_val.json`.
Odtworzony wybór K4 pokrywa się z wyborem z Edenu w 100 procentach, więc analiza dotyczy tego samego obiektu.

**Czym są współrzędne Koopmana latentu całej klatki.**

| Współrzędne (16) | wariancja MIĘDZY filmami | wyjaśnione stanem narzędzia (położenie, rozmiar) | wyjaśnione czasem w filmie | korelacja klatka do klatki |
|---|---|---|---|---|
| wolne mody Koopmana | **0,72** | -0,01 | 0,00 | 0,94 |
| 16 głównych kierunków PCA | 0,65 | 0,01 | 0,00 | 0,95 |
| 16 losowych rzutów | 0,58 | 0,00 | 0,00 | 0,83 |

To NIE jest stan narzędzia. Wolne mody latentu DINOv2 całej klatki to odcisk palca OPERACJI: oko, oświetlenie, ustawienie
mikroskopu; w obrębie jednego filmu prawie stoją. Stan narzędzia siedzi w latencie DETEKTORA z obszaru narzędzia
(dokument 38), nie tutaj. K4 rozkłada więc klatki po rodzajach scen, nie po stanach narzędzia.

**Przewaga w uczeniu, zmierzona na gotowych 27 biegach.**

| Porównanie | AP najlepszej epoki | odpowiednik w losowych etykietach (nachylenie 0,0107 na 1000) | chybienia, sparowany bootstrap po 25 operacjach walidacji |
|---|---|---|---|
| K4 - B3 pokrycie wyglądu | +0,0072 +- 0,0009 | około +670 | **-1,42 punktu, przedział [-2,73; -0,26]** |
| K4 - B1 losowanie | +0,0072 +- 0,0009 | około +670 | -0,72 punktu, przedział [-1,90; +0,06] |
| K4 - B2 równo w czasie | +0,0023 +- 0,0028 | około +210 | -0,69 punktu, przedział [-1,75; +0,06] |

Szybkość: K4 przekracza AP 0,675 w epoce 3 we wszystkich trzech ziarnach (pozostałe w epoce 3 do 5, klatki po kolei nigdy);
w epoce 3 ma 0,683 wobec 0,676 (B2), 0,674 (B1), 0,668 (B3). Średnie AP z epok 0 do 9: K4 0,6527, B2 0,6502, B1 0,6479, B3 0,6471.

**Czego NIE udało się wykazać.**

1. Że działa akurat składnik Koopmana. Opisy wyboru (równość po filmach, pokrycie położenia i rozmiaru narzędzia, czasu,
   promień pokrycia w przestrzeni wyglądu) są dla K4 praktycznie takie same jak dla losowania i równego wyboru w czasie;
   żaden nie wyróżnia K4. Pokrycie położenia narzędzia K4 ma wręcz najniższe (0,785), a najwyższe ma B4 (0,831), który nie wygrał.
2. Kontrole wymiaru wybierają INNE klatki niż K4 (część wspólna: 24-wymiarowy rzut 0,24, PCA16 0,21, losowe rzuty 0,13,
   przy około 0,10 dla czystego przypadku), więc nie wiadomo, czy uczyłyby tak samo dobrze. To wymaga treningów.
3. **Niezmierzony szum samego WYBORU.** Odchylenia w tabelach to ziarna TRENINGU przy jednym wyborze klatek. Dwa różne losowe
   wybory też różnią się o kilka tysięcznych (B1 i B2 o 0,0044, K3 i B3 o 0,0021). Przewaga K4 może w części być szczęśliwym
   losem jednego wyboru. Bez kilku ziaren wyboru nie da się tego rozdzielić.

Wniosek: przewaga K4 jest realna względem pokrycia wyglądu (przedział po operacjach poniżej zera), nierozstrzygnięta względem
równego wyboru w czasie, a jej przyczyna jest nieustalona. Etap 2 musi mieć kilka ziaren WYBORU dla każdego ramienia,
kontrole PCA16, losowe rzuty i rzut 24-wymiarowy, oraz wariant na latencie narzędzia z detektora.

## 5e. Czy Koopman na latencie SAMEJ SIECI wskazuje, czego jej brakuje (pilot bez treningu, 2026-09-22)

Pytanie autora: nie „które klatki pokryć", tylko „czego sieci brakuje do wytrenowania" — sygnał z dynamiki latentu
sieci w trakcie uczenia, który wskazuje klatki, po których cała sieć lepiej rozumie narzędzie.
Kod: `pipeline/koopman_langevin/koopman_defect_pilot.py`; wynik `REPORT_defect_pilot.json`.

Definicja mierzalna: brak, który da się usunąć treningiem = klatka, na której detektor z 500 etykiet chybia (IoU < 0,5),
a detektor z 2500 etykiet trafia. Na 3499 klatkach walidacji z torami: detektor L0 chybia 4,4 procent, duży 4,1 procent;
**tylko 23 procent chybień L0 znika po pięciokrotnie większej liczbie etykiet.** To sufit każdego doboru: 77 procent
błędów małej sieci nie jest brakiem danych.

Sygnały bez etykiet, liczone na checkpoincie L0, i ich zdolność wskazania USUWALNEGO błędu (AUC, 0,5 = los):

| Sygnał | chybienie L0 | usuwalne wśród chybionych | usuwalne wśród wszystkich | usuwalne w górnych 10 procentach |
|---|---|---|---|---|
| 1 - pewność detektora | 0,796 | 0,758 | **0,928** | 6,9 procent |
| reszta Langevina latentu | 0,772 | 0,591 | 0,798 | |
| reszta minus ruch własnej ramki („defekt") | 0,731 | 0,456 | 0,689 | |
| reszta Koopmana w wolnych modach | 0,768 | 0,672 | 0,839 | 5,2 |
| dryf cząstki między checkpointami (zegar treningu) | 0,774 | 0,710 | 0,864 | 5,7 |
| pewność + dryf (suma rang) | | | 0,935 | 7,1 |
| pewność + dryf + Koopman | | | 0,931 | 7,3 |

Ten sam dryf mierzony WEWNĄTRZ jednego biegu (epoki 23 do 29, trening b1500): AUC usuwalnego błędu 0,873, pewność 0,901,
suma rang 0,920. Dryf wczesny (epoki 2 do 5) 0,807.

Odczyt:

1. **Dynamika latentu sieci niesie informację o brakach** (AUC 0,84 do 0,87 wobec 0,5 dla losu), ale własna pewność
   detektora niesie jej więcej (0,93). Odjęcie ruchu ramki od reszty latentu POGARSZA sygnał: to, co pomaga, to właśnie
   część zgodna z ruchem narzędzia, nie „defekt reprezentacji" poza nim.
2. Jedyny sygnał z dynamiki, który coś DOKŁADA do pewności, to zegar treningu: klatki, których cząstka wciąż dryfuje
   między checkpointami (korelacja z pewnością tylko 0,28; suma rang 0,935 wobec 0,928; w górnych 10 procentach o 0,4 punktu
   więcej usuwalnych). Zysk jest mały i przy 35 usuwalnych klatkach mieści się w szumie.
3. Nawet najlepszy sygnał w górnych 10 procentach klatek trafia w usuwalny błąd tylko w 7 procentach. Dobór celujący
   w „braki sieci" kupuje więc głównie klatki, których błędu etykiety nie usuną (Hu i in. 2021, dokładnie ten mechanizm).
   To tłumaczy etap 1: B4 (pewność) i K1 (reszta) nie pomogły.

Wniosek dla konstrukcji: „Koopman wskazuje, czego sieci brakuje" da się zbudować jako sumę rang pewności, dryfu
między checkpointami i reszty w wolnych modach, liczoną na latencie DETEKTORA co rundę; ale sufit 23 procent i precyzja
7 procent mówią, że to poprawi detektor co najwyżej o ułamek tego, co daje dorzucenie rozrzuconych etykiet. Większy zapas
leży w samej sieci (32 procent chybień w szybkim ruchu niezależnie od doboru), więc kolejnym krokiem z sensem jest
Koopman jako CEL uczenia (spójność latentu sąsiednich klatek pod jednym operatorem liniowym), nie jako selektor.

## 5f. Ramię K5: pokrycie kodu tożsamości narzędzia (zbudowane i zmierzone bez treningu, 2026-09-22)

Skąd: sekcja 4j dokumentu 38 pokazała, że kod tożsamości (średnie współrzędne tokenów narzędzia w wolnych funkcjach
własnych operatora TRANSLACYJNEGO Koopmana, liczonego na tokenach samego detektora) jest semantyczny: nie da się go
zdekodować do pikseli, a wyszukiwanie klatek po „domknięciu" (odległości kodu od prototypu) szereguje po klipie.
Jedyna forma, w której pomysł „znajdź klatki, które dobrze kodują niezmienniki, i szukaj podobnych" daje ramię
doboru, to POKRYCIE stanów tożsamości w puli. Tu zbudowane i zmierzone; żaden trening nie został uruchomiony.

Wejście (zadania Eden 1809407 i 1809408: 3 min na H100 plus 13 s CPU). Rejestrator cząstek
(`patch_particles.py --frame-list`) na detektorze L0 (`ut_b0500_best.pth`) przeszedł wszystkie 10 637 klatek puli
treningowej (`tracking_static36`); `identity_code_pool.py` dopasował operator translacyjny na 612 682 parach tokenów
narzędzia i policzył kod (8 modów, |λ| od 0,64 do 0,32) oraz domknięcie każdej klatki. Kod ma 9 685 klatek; 952 klatki
są bez kodu, bo L0 nie ma tam pewnej ramki (score nie wyżej niż 0,5 albo mniej niż 4 tokeny w ramce). Plik:
`SmallViTAblation/results/koopman_langevin_epochs/KOD_TOZSAMOSCI_pool_b0500.npz` (439 KB, poza gitem). Strażnik
determinizmu rejestratora nadal nie przechodzi (0,131 jednostki współrzędnej): kod niesie szum numeryczny tego rzędu,
co przy odległościach pokrycia rzędu 1 nie przeszkadza, ale porównania klatka po klatce wyklucza.

Cztery ramiona (`pipeline/koopman_langevin/identity_cover_arm.py`; ten sam L0, budżet 1500, odstęp 5 klatek,
bez etykiet; pliki `curated/selection_matrix_stage1/matrix_K5*_b1500_sel42.json`, opis w `K5_MANIFEST.json`):

| Ramię | Reguła |
|---|---|
| K5_identity_cover | k-centrum euklidesowe w ustandaryzowanym kodzie (zawsze klatka najdalsza od już oznaczonych): rzadkie stany tożsamości najpierw |
| K5b_identity_cover_plus_unseen | jak K5, ale 20 procent dokupionego budżetu (200 klatek) równomiernie w czasie z klatek BEZ kodu |
| K5c_closure_near | dosłowny pomysł autora: klatki o kodzie NAJBLIŻSZYM prototypu (tożsamość zakodowana „poprawnie"); kwota na film proporcjonalna do jego udziału w puli, żeby nie wziąć jednego klipu |
| K5d_closure_far | lustro K5c: klatki o kodzie NAJDALSZYM od prototypu |

Pomiar, który sprawdza samo pokrycie (nie trening): odległość losowej klatki puli (2000 klatek) od najbliższej klatki
oznaczonej (L0 plus dokupione) w przestrzeni kodu, ta sama miara dla ramion etapu 1:

| Zbiór oznaczony | mediana | p90 | max | filmy | klatek bez kodu | mediana domknięcia dokupionych |
|---|---|---|---|---|---|---|
| tylko L0 (500) | 1,055 | 1,580 | 2,914 | | | |
| K5_identity_cover | 0,919 | 1,178 | 1,335 | 112 | 0 | 2,117 |
| K5b_identity_cover_plus_unseen | 0,943 | 1,220 | 1,408 | 114 | 200 | 2,145 |
| K5c_closure_near | 0,913 | 1,478 | 2,914 | 118 | 0 | 1,144 |
| K5d_closure_far | 0,938 | 1,307 | 2,206 | 118 | 0 | 2,301 |
| K4_koopman_cover | 0,859 | 1,313 | 2,443 | 116 | 100 | 1,666 |
| B3_coverage | 0,874 | 1,315 | 2,346 | 117 | 180 | 1,680 |
| B2_uniform_time | 0,868 | 1,285 | 2,483 | 118 | 117 | 1,681 |
| B1_random | 0,844 | 1,303 | 2,658 | 118 | 97 | 1,645 |
| K1_langevin | 0,865 | 1,324 | 2,485 | 118 | 142 | 1,641 |
| B4_uncertainty | 0,921 | 1,339 | 2,585 | 118 | 364 | 1,821 |

Mediana domknięcia w całej puli: 1,660. Wspólne klatki K5 z każdym ramieniem etapu 1: od 9 do 13 procent.

Odczyt:

1. **Żadne ramię etapu 1 nie różni się od losu w przestrzeni kodu tożsamości** (mediana 0,84 do 0,87, max 2,3 do 2,7).
   Etap 1 nie testował więc pokrycia stanów tożsamości narzędzia widzianych przez samą sieć; K5 jest nową osią, nie
   wariantem B3 ani K4.
2. **K5 zmniejsza największą lukę pokrycia o połowę** (max 2,9 dla samego L0 do 1,3) i najmocniej ściąga p90; medianę
   ma nieco gorszą od losu (0,92 wobec 0,84), bo k-centrum wydaje etykiety na rzadkie stany. Kupuje klatki o dużym
   domknięciu (mediana 2,12 wobec 1,66 w puli): nietypowe stany tożsamości, nie stany „dobrze zakodowane".
3. **K5c, czyli dosłownie „klatki, które kodują poprawnie", pokrywa typowe stany jak los, a rzadkich nie dotyka wcale**
   (max 2,914, tyle co sam L0). Jeśli w treningu wygra K5c, sieć uczy się z klatek, które już rozumie (przeciw
   hipotezie o brakach); jeśli K5, uczy się z rzadkich stanów tożsamości. To jest rozstrzygający kontrast tej rodziny.
4. B4 (niepewność) kupiło 364 klatki bez kodu (36 procent dokupionych) i ma najgorsze pokrycie: niepewność kupuje
   „nie widzę narzędzia", a nie „widzę je w nowym stanie".

Uwaga o metryce K4 z etapu 1 (znaleziona przy budowie K5): `select_batch(strategy='coverage')` liczy k-centrum
KOSINUSOWE i zakłada wektory jednostkowe; `selection_matrix.py` podał mu współrzędne Koopmana ustandaryzowane, ale
nie znormalizowane. K4 było więc regułą „kierunek najbardziej przeciwny do już wybranych" (z normą wzmacniającą
skrajne klatki), nie k-centrum w sensie odległości. Wynik K4 (0,6885) stoi, bo ramię jest deterministyczne i opisane
kodem, ale jego nazwa „pokrycie" jest na wyrost; kontrola K4 z k-centrum euklidesowym wchodzi do etapu 2. B3 tej wady
nie ma (cechy DINOv2 w pamięci podręcznej są jednostkowe). K5 używa własnego k-centrum euklidesowego (`kcentre`).

Sprawdzenie siatki i dopasowania tokenów (`sledzenie_czastek.py`, rysunki 05 i 06 w katalogu
`results/koopman_langevin_epochs/rysunki_k5`; prawdziwe ramki użyte WYŁĄCZNIE do oceny):

- **Przypisanie tokenu do miejsca obrazu jest poprawne.** Dziesięć tokenów najpodobniejszych do wzorca narzędzia
  leży w prawdziwej ramce w 35 procentach przypadków, przy 2 procentach dla losu (tyle wynosi udział pola ramki
  w kadrze) i 7 procentach przy siatce transponowanej. Transpozycja psuje wynik pięciokrotnie, więc kolejność
  wiersz-kolumna w `patch_particles.py` jest zgodna z obrazem.
- **Dopasowanie tokenów między klatkami po samych liczbach trzyma się narzędzia.** Token wybrany na narzędziu,
  szukany w kolejnych klatkach przez największe podobieństwo kosinusowe profilu, trafia w prawdziwą ramkę
  w 79 procentach na ośmiu klipach (97 procent na klipie z rysunku 05), przy 3 procentach dla losu. To odpowiednik
  śledzenia punktów na filmach DAVIS i pokazuje, że opis tokenu niesie tożsamość miejsca na narzędziu.
- **Odniesienie do eksperymentów z niedźwiedziem w projekcie grantowym.** Tam istnieją oba opisy. W linii DAVIS
  (`Research/DiagnosticTool/test/latki_lagrange.py`) cząstki zasiane na niedźwiedziu są niesione przepływem
  optycznym i po 80 klatkach 76,3 procent wciąż leży na zwierzęciu (94,4 procent przy trackerze CoTracker3),
  przy 7,0 procent dla kontroli z tła. Zysk dla dynamiki był jednak znikomy (R2 z 0,0762 na 0,0783), a na drugim
  filmie (Wilford) przepływ optyczny przegrał z cząstką nieruchomą (ICC 0,722 wobec 0,947) i został zastąpiony
  śledzeniem W PRZESTRZENI CECH: 40,8 procent cząstek zostaje na zwierzęciu przez cały odcinek, wobec 27,6 procent
  dla przepływu optycznego (teza D104). Test z rysunku 05 jest odpowiednikiem tego zwycięskiego wariantu: token
  szukany po podobieństwie cech trzyma się narzędzia w 79 procentach przy 3 procentach dla losu.

- **Sama cząstka się jednak nie porusza.** W konwencji projektu grantowego (opis Eulera, T15a) cząstka to stałe
  miejsce siatki, przez które narzędzie przepływa (rysunek 06). Ruch wchodzi do analizy przez parowanie tokenów,
  a nie przez wędrującą cząstkę.
- **Ograniczenie maski.** `maska_pred` to wnętrze ramki detektora, nie sylwetka narzędzia, więc przy cienkim
  narzędziu część tych tokenów opisuje tło wewnątrz ramki. Dotyczy to też par translacyjnych, z których liczony
  jest kod tożsamości.

Co naprawdę niosą te osiem liczb (`co_niesie_kod.py`, etykiety użyte WYŁĄCZNIE do oceny, predykcja grzbietowa
z podziałem po filmach, więc model nie może wygrać rozpoznaniem filmu; 9 142 klatki, 117 filmów):

| Wielkość przewidywana z kodu | Wyjaśniona zmienność z kodu | To samo z geometrii ramki |
|---|---|---|
| proporcja ramki prawdziwej (im bardziej wydłużona, tym bardziej narzędzie leży w płaszczyźnie kadru) | 85,8 procent | 3,0 |
| pole ramki prawdziwej (jak blisko/duże jest narzędzie) | 29,0 | 0,0 |
| zwrot: końcówka względem środka ramki, poziomo | 0,1 | 2,5 |
| zwrot: końcówka względem środka ramki, pionowo | 0,3 | 0,9 |
| położenie narzędzia w kadrze, poziomo | 10,0 | 100 |
| położenie narzędzia w kadrze, pionowo | 18,5 | 100 |

Udział zmienności kodu MIĘDZY filmami: 58 procent. Kod niesie więc przede wszystkim wygląd geometryczny narzędzia
(wydłużenie, czyli nachylenie względem płaszczyzny obrazu, oraz skalę) plus tożsamość operacji; nie niesie zwrotu
(gdzie jest końcówka) ani, w znacznym stopniu, położenia. Wcześniejsze sformułowanie „otwarte czy zamknięte" jest
nieuprawnione: w tym podzbiorze jest tylko jedna klasa narzędzia (cystotome, 22 680 klatek) i nie ma etykiety stanu
narzędzia, więc takiej hipotezy nie da się tu sprawdzić.

Czego to nie dowodzi: ani jednego treningu nie było. Lepsze pokrycie kodu tożsamości nie musi przekładać się na AP;
etap 1 pokazał, że pokrycie wyglądu (B3) nie pobiło równomiernego czasu. Wartość K5 to czysty test hipotezy „stany
tożsamości narzędzia widziane przez samą sieć są właściwą jednostką doboru", którego etap 1 nie zawierał.

Propozycja do etapu 2 (wymaga zatwierdzenia autora; nic nie zostało zgłoszone na Eden): K5, K5c i K5b jako trzy
ramiona przy tym samym L0 i budżecie 1500, po 3 ziarna treningu, 12 epok (najlepsza epoka w etapie 1 wypadała
około 7, a 30 epok przeuczało); kontrast główny K5 minus B2 (równomiernie w czasie), pomocniczy K5 minus K5c;
próg jak w etapie 1 (0,004 AP najlepszej epoki, chybienia z bootstrapem po operacjach). Koszt: 9 biegów po około
25 min na A100, w pięciu równoległych procesach około godziny.

## 6. Kontrargumenty

- Trzy ziarna nie wystarczą do twierdzeń; rozrzut między ziarnami w kampanii statycznej to około 0,002 AP.
- Reszta jest retrospektywna (używa bieżącej klatki); to dozwolone przy doborze z gotowej puli, ale nie jest prognozą.
- Duża reszta to w praktyce szybki ruch i wejście narzędzia; K1 może więc wygrać tylko w warstwie szybkiego ruchu.
- Część najszybszych klatek to błędy adnotacji; K1 może kupować szum etykiet (Hu i in. 2021).
- B0 jest słabym punktem odniesienia; wygrana tylko z nim niczego nie dowodzi.
- Pewność L0 pochodzi z jednego ziarna detektora; B4 może być przez to zaniżone.
- K4 z etapu 1 użyło k-centrum kosinusowego na nieznormalizowanych współrzędnych (sekcja 5f): jego wynik jest wynikiem
  reguły deterministycznej, nie „pokrycia" w sensie odległości; kontrola euklidesowa dopiero w etapie 2.

## 7. Data gaps

- Brak linii bazowej RHO-loss (wymaga etykiet odłożonych; poza etapem 1).
- Brak wariantu z latentem detektora w K1 (etap 3).
- Identyfikacja pacjentów niepotwierdzona; jednostką resamplingu jest grupa operacji.
