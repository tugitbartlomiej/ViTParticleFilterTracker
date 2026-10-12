# Ocena wklejonej interpretacji eksperymentów

11.10.2026. Przeczytano cały załącznik `Wklejony tekst.txt` oraz nieucięte sekcje 6b–6c źródłowego `outputs/porownanie_szklo_20261010/README.md`. Porównano twierdzenia z JSON-ami, kodem pomiarów i kontrolami. To ocena interpretacji, bez ponownego treningu ani modyfikacji historycznych wyników.

**Ocena:** wspólne aktualizacje wag są dobrze umotywowanym źródłem skorelowanego ruchu tokenów. Tekst przedstawia jednak jako rozstrzygnięte kilka hipotez, których eksperymenty nie izolują. Szczególnie za mocne są „test jądra zamyka temat”, „zamrożone, sztywniejące dołki”, „temperatura = długość kroku” oraz „mod zdjęcia pochodzi z uwagi”.

## Co można utrzymać

- Bliskie reprezentacje mają silnie skorelowane fluktuacje; po zastosowanym odjęciu pola obrazu korelacje par losowych są znacznie słabsze.
- Ranking ruchliwości utrzymuje się w badanych oknach i powtarza między wybranymi ziarnami. To obserwacja trwałej heterogeniczności w ograniczonym czasie.
- Perturbacje wspólnych wag odtwarzają część przestrzennej struktury ruchu. Efekt występuje w kilku architekturach i optymalizatorach.
- Nie ma podstaw, by obecne dane przedstawiać jako potwierdzenie fizycznego szkła koloidalnego.

## Co wymaga zmiany

| Twierdzenie | Ocena i właściwy zakres |
|---|---|
| „Cząstki w osobnych, zamrożonych i sztywniejących dołkach” | Model roboczy. Trwały ranking nie identyfikuje potencjału, siły przywracającej ani wzrostu sztywności. Taką mapę może tworzyć trwała wrażliwość tokenów na zmiany wspólnych wag. W kontroli OU-B sztywności są wylosowane raz, a zmniejsza się amplituda napędu: sama ta kontrola nie dowodzi sztywnienia. |
| „Test jądra zamyka temat” | Test jest wartościowy, ale retrospektywny. Checkpoint z końca okna porównuje się z wcześniejszymi 200 krokami. Spearman 0.85 i 0.96 dotyczy map amplitud, nie pełnych trajektorii, i nie oznacza 85–96% wyjaśnionej wariancji. Potrzebny test przyszłego okna i porównanie z prognozą wykorzystującą wcześniejszą mapę. |
| „Prawie cały ruch jest polem; ruch własny 10–15%” | Wartość dotyczy wybranych wczesnych warunków. Po 60. epoce ta sama miara wynosi 37.88% przy zapisie f16 i 36.83% przy f32. Ponadto energia sąsiedztwa i reszty nie tworzy rozłącznego podziału; wartości większe niż 100% nie są udziałami w zwykłym sensie. |
| „Pary nie rozbiegają się w ogóle” | Nie odpowiada zapisanym liczbom. Dla `lr_5e-4_rp` stosunek średnich kwadratów odległości wynosi 1.316, czyli wzrost o 31.6%; dla `klas_lr_5e-4` 2.250, czyli wzrost o 125%. To odpowiada wzrostowi odległości RMS o około 14.7% i 50.0%. Można mówić o braku obserwowanego reżimu Richardsona, nie o zerowym rozbieganiu. |
| „Szkło odpada jako mechanizm korelacji” | Kontrole osłabiają zgodność z konkretnym modelem BD i wzmacniają wyjaśnienie przez wspólne parametry. Zakres korelacji 0.004–0.24 nie jest uniwersalną granicą dla wszystkich szkieł. W kontroli BD czasy relaksacji wynoszą 2.24, 27.48, 435.64 klatki lub nie są osiągnięte. Stałe podokna 200 klatek i filtr 201 klatek obejmują bardzo różne skale relaksacji. Potrzebne porównania przy dopasowanych skalach czasu. |
| „Gradient clipping ustala krok około learning rate; paczka bez wpływu” | Kod używa SGD z momentum 0.9. Clipping ogranicza bieżący gradient, natomiast aktualizacja zależy też od akumulowanej historii; norma kroku nie jest przez to ustalona. Wskaźnik 0.394 → 0.305 między batch 16 i 128 zmienia się o około 22.5%. Zmienia się też liczba obejrzanych przykładów na tę samą liczbę kroków. Brak podstaw do deklaracji braku wpływu batch size. |
| „Mod zdjęcia pochodzi z uwagi” | Hipoteza zgodna z różnicą między ViT i CNN. Porównanie zmienia także kodowanie pozycji i liczbę parametrów. Nie izoluje przyczynowo attention. |
| „Zimne dno, gorąca góra” | Względna amplituda ruchu rośnie z głębokością, ale jej mianownik także się zmienia. Bezwzględne lokalne MSD przy odstępie 10 kroków wynosi około 0.730 w bloku 0 i 0.556 w bloku 5. Czas relaksacji nie maleje monotonicznie po wszystkich warstwach. Nie jest to pomiar termodynamicznej temperatury. |
| „Strata detekcji przypina tokeny, klasyfikacja ich nie przypina” | Obiecująca hipoteza, ale zmieniają się równocześnie dane, strata, głowa i rozdzielczość. Dodatkowo w tym samym długim biegu detekcji lokalne MSD przy odstępie 105 kroków rośnie o 82%, mimo spadku wskaźnika Lindemanna. Spadek ilorazu nie jest dowodem sztywnienia. |
| „f32 kontra f16: zapis f16 wystarcza” | Podobne są wskazane agregaty dwóch wznowionych biegów. To porównanie formatu zapisu rzutów i norm, przy aktywacjach obliczanych z autocast BF16. Nie wyklucza artefaktów obliczeń BF16. Najczystsza kontrola zapisu to dwie wersje tych samych aktywacji z tego samego przebiegu; kontrola obliczeń wymaga także forward FP32. |

## Dwie uwagi o fizycznych analogiach

Prawo Richardsona dotyczy średniego kwadratu odległości par w określonym reżimie skali i czasu. Brak wzrostu sześciennego sam w sobie nie jest ogólnym testem wykluczającym turbulencję. W tych danych nie ma dodatniego dowodu na turbulencję, więc nie warto budować na niej interpretacji. [Bourgoin, 2015](https://arxiv.org/abs/1410.0982).

W modelach cieczy szkłotwórczych struktura może predysponować konkretne cząstki do wolniejszego lub szybszego ruchu; istnieją także korelacje utrzymujące się dłużej niż czas relaksacji strukturalnej. Dlatego przeciwstawienie „trwała mapa = OU, przemijająca mapa = każde szkło” jest zbyt szerokie. [Dunleavy i in., 2015](https://www.nature.com/articles/ncomms7089).

Sformułowanie „tokeny nie oddziałują” także wymaga doprecyzowania. W ViT obliczenia tokenów są sprzężone przez self-attention, a aktualizacje przez wspólne parametry i stratę. Nie wykazano sił parowych między zmierzonymi wektorami GELU w latencie; to inne stwierdzenie niż całkowity brak sprzężeń.

## Proponowany wniosek do artykułu

„W badanych sieciach obserwujemy trwałe zróżnicowanie ruchliwości tokenów i przestrzennie skorelowane fluktuacje reprezentacji. Odpowiedź na perturbacje wspólnych parametrów odtwarza istotną część mapy ruchliwości oraz struktury korelacji. Wyniki wspierają wyjaśnienie przez wrażliwość reprezentacji na aktualizacje wag; analogia do procesu w lokalnych pułapkach pozostaje modelem efektywnym wymagającym osobnej walidacji.”

To teza mocniejsza dowodowo niż ogłoszenie identyfikacji nowej „substancji”. Dalsze rozstrzygnięcie wymaga przewidywania przyszłego ruchu, estymacji odpowiedzi przywracającej oraz interwencji mierzącej wpływ wybranej składowej ruchu na jakość sieci.

Liczby i ograniczenia uzupełniają `RAPORT.md`, `same_run_comparisons.json`, `kernel_audit.json` i `readout_nullspace.json` w tym samym katalogu. Źródłowe README i JSON-y pozostawiono bez zmian.
