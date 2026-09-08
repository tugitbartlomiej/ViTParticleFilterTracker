# 04 — Mapa: zarzut recenzenta → eksperyment, który go zamyka

Źródła: `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\odpowiedziRecenzentów\`
— `Reviewer_Comments_to_Authors.pdf` (recenzent 1) i `Comments_to_the_Author.pdf`
(recenzent 2). **Obaj: Reject.** Recenzent 2 dopisał „do not encourage resubmit",
więc to jest nowe zgłoszenie w innym miejscu, nie poprawka.

Ta tabela jest listą kontrolną. Jeśli któryś wiersz zostanie pusty, artykuł
wraca z tym samym zarzutem.

---

## Recenzent 1

| # | Zarzut | Co go zamyka | Gdzie |
|---|---|---|---|
| 1 | losowy podział klatek w danych wideo | podział po filmie/pacjencie, jeden dla wszystkich biegów; weryfikacja, czy film = pacjent | `01_PROTOCOL` P1, `05_OPEN` O1 |
| 2 | jeden seed, brak dowodu powtarzalności | 3 ziarna dla macierzy, 5 dla głównej tabeli, 10 losowań dla baseline'u; średnia ± odchylenie | `01_PROTOCOL` P10 |
| 3 | wagi i parametry dobrane ręcznie, bez analizy czułości | zestaw `S-*`: perturbacje wag, wymiar PCA, wybór reprezentanta, algorytm pokrycia, współczynniki trudności | `02_MATRIX` §4 |
| 4 | ablacje trenowane innym protokołem niż pełny pipeline | pełny pipeline jest **wariantem w tej samej macierzy**; stała liczba epok, jedna reguła checkpointu | `01_PROTOCOL` P0, P5, P6 |
| 5 | inferencja z jednego zapytania DETR, nieuzasadniona | standardowe dekodowanie; jedno zapytanie tylko jako osobna obserwacja historyczna | `01_PROTOCOL` P7 |
| 6 | jeden arbitralny próg ufności, brak kalibracji | mAP z pełnej krzywej PR; próg operacyjny wybierany na walidacji; krzywa F1 od progu; test przeniesienia progu na obcą dziedzinę | `01_PROTOCOL` P8 |
| 7 | tylko YOLOv8L, brak nowoczesnego detektora | RF-DETR-Small jako model główny, D-FINE-S i bieżąca YOLO jako przeniesienie | `00_RESEARCH` §2, `02_MATRIX` §6 |
| 8 | porównanie architektur non-iso-resource | ta sama rozdzielczość, ten sam budżet epok, ten sam zbiór dla wszystkich architektur | `02_MATRIX` §6 |
| 9 | brak analizy NMS w YOLO | raportować ustawienia NMS i sprawdzić wrażliwość; przy jednej klasie to krótka analiza | `06_TODO` T14 |
| 10 | zbiór zewnętrzny oglądany zbyt wiele razy | dyscyplina „zero-look": wybór epoki i progu tylko na walidacji, zbiór zewnętrzny liczony raz, `external_access_ledger.jsonl` | `01_PROTOCOL` P9 |
| drobne | nowość to kombinacja, nie składniki | powiedzieć to wprost w artykule; macierz 15 kombinacji jest właśnie dowodem na wartość kombinacji | — |
| drobne | uzasadnienie rozmiaru 20 000 | sweep 2 500 / 5 000 / 10 000 / 20 000 | `02_MATRIX` §5 |
| drobne | 71,3% wybranych to kopie augmentowane | analiza składu augmentacji w wybranym zbiorze, bez GPU | `02_MATRIX` §7 |

## Recenzent 2

| # | Zarzut | Co go zamyka | Gdzie |
|---|---|---|---|
| 1 | dlaczego akurat te pięć składowych miary złożoności FastSAM | uzasadnienie merytoryczne + `S-*` na tej mierze; jeśli nie da się obronić, uprościć miarę | `02_MATRIX` §4, `05_OPEN` O4 |
| 2 | test wewnętrzny ma tylko 589 obrazów, brak rozmiaru walidacji, podział po klatkach | podział po filmie daje 7 filmów testowych zamiast losowych klatek; podać wszystkie trzy liczby w tabeli | `01_PROTOCOL` P1, `06_TODO` T2 |
| 3 | YOLOv8 nie reprezentuje rodziny | bieżąca YOLO w rozmiarze zbliżonym parametrami | `02_MATRIX` §6 |
| 4 | niespójne metryki między tabelami | jeden zestaw metryk wszędzie: mAP@0.5, mAP@[.5:.95], TP/FP/FN, P/R/F1 | `01_PROTOCOL` P8 |
| 5 | **ablacja niepełna — trzeba wszystkich kombinacji czterech cech** | macierz 15 kombinacji: pojedyncze, pary, trójki, komplet | `02_MATRIX` §2 |
| 6 | **baseline losowy niewystarczający; potrzebny losowy z klatek źródłowych** | `B-RANDAUG` na 10 losowaniach + `B-RANDSRC` z klatek przed augmentacją | `02_MATRIX` §3 |
| 7 | ablacja tylko na DETR, a teza dotyczy doboru danych ogólnie | poziom B powtarza najważniejsze konfiguracje na drugiej i trzeciej architekturze | `02_MATRIX` §6 |
| 8 | pochodzenie liczby 91 336 (22 837 × 4 = 91 348) | rekoncyliacja na danych i jedno zdanie w artykule | `02_MATRIX` §7, `05_OPEN` O3 |
| 9 | jedna instytucja, jeden seed | ziarna załatwione; jedna instytucja zostaje jako ograniczenie — napisać uczciwie, nie chować | `01_PROTOCOL` P10 |
| 10 | dlaczego izolacja efektu kuracji tylko na DETR | jak wiersz 7 | `02_MATRIX` §6 |
| 11–16 | rysunki, mapowanie faz pipeline'u na [Fig. 1] (rysunek pierwszy), wymiar PCA, alternatywy dla K-Means, tabela podziału danych, jasny opis co jest czym w sekcji ablacji | częściowo eksperymenty (`S-PCA`, `S-ALG`), częściowo redakcja | `02_MATRIX` §4, `06_TODO` T15 |
| 17 | interpretacja kliniczna przy niskim bezwzględnym mAP | oddzielić „lepsza generalizacja względna" od „użyteczne klinicznie"; napisać wprost | — |

---

## Trzy rzeczy, które recenzenci przeoczyli, a i tak trzeba naprawić

1. **Selekcja jechała po całej puli**, łącznie z klatkami z filmów testowych.
   To jest wyciek silniejszy niż ten, który recenzent 1 opisał w zarzucie 1.
2. **EL2N pochodzi z modelu wytrenowanego na całej puli przez 170 epok** — to nie
   jest EL2N w sensie oryginalnej definicji i domyka błędne koło.
3. **Wagi po wyłączeniu grupy były dobierane ręcznie**, więc „bez Fouriera"
   zmieniało też proporcje pozostałych cech. Dwie zmiany naraz w jednym wariancie.

Warto je naprawić i **opisać w artykule jako świadomą przebudowę protokołu**.
Recenzent, który zobaczy, że autor sam znalazł więcej niż on, ocenia inaczej.
