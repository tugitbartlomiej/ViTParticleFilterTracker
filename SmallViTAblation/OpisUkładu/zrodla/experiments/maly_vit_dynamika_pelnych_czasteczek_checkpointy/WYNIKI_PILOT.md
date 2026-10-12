# Pilot prognozy pełnych cząsteczek

Data: 2026-10-01. Status: lokalny pilot zakończony. Jeden istniejący przebieg uczenia; bez nowego treningu ViT.

**Wynik:** PCA z 8 kierunkami i ridge AR(1), wybrane wyłącznie na wcześniejszej walidacji, zmniejszyły średni RMS błędu pełnych wektorów względem persystencji o **3.62 / 3.81 / 5.06 / 4.31 procent** dla horyzontów 10 / 20 / 50 / 100 aktualizacji. Pełny ridge 768D był przy H=100 gorszy o 14.33 procent. Dodanie jednego opóźnienia w testowanym wariancie nie poprawiło średniej względem PCA AR(1).

To mały dodatni wynik predykcyjny jednego przebiegu. Nie potwierdza tożsamości narzędzia ani ogólnej przewagi modeli z pamięcią. Wykres i tabela pokazują także wyniki negatywne.

## Aparat i poprawność danych

Analizowano 101 stanów co 10 aktualizacji, 12 stałych obrazów i 218 osobnych tokenów oryginalnych masek. Każdy token zachował wszystkie 768 współrzędnych. Gęsty plik 1001 stanów czytano przez mmap; wybrana tablica w RAM zajmuje 64.51 MiB.

Lokalne forwardy checkpointów 0 i 1000 porównano z zapisami Edenu. Model, obrazy i maski mają sprawdzone pochodzenie. Pomiar: ostatnia MLP po GELU, eval, float32, bez autocast i TF32. Wyniki kontroli:

| Krok | Względny błąd L2 lokalnie/Eden | Maks. błąd powtórzenia | BN bez zmian |
|---:|---:|---:|---|
| 0 | 8.68e-06 | 0 | True |
| 1000 | 6.31e-06 | 0 | True |

Szczegóły sprzętu i wersji: [AUDYT_ZGODNOSCI.json](AUDYT_ZGODNOSCI.json).

## Ocena poza oknem dopasowania

Ustawienia wybrano na początkach 400 i 500, z celami najdalej do kroku 600. Test obejmuje początki 600, 700, 800 i 900 oraz horyzonty 10, 20, 50, 100. Operator i PCA używają wyłącznie przeszłości danego początku; wszystkie roll-outy są autonomiczne. Po kroku 1000 nie ma danych.

Liczby poniżej to średni stosunek RMS błędu pełnego wektora do persystencji. **Mniej niż 1 oznacza mniejszy błąd niż persystencja; więcej niż 1 oznacza gorszy wynik.** Każdy obraz ma tę samą wagę, a tokeny w obrazie są ważone pokryciem maską.

| Model | H=10 | H=20 | H=50 | H=100 |
|---|---:|---:|---:|---:|
| Persystencja | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Stała prędkość | 1.5962 | 2.2565 | 4.2500 | 7.7558 |
| Wygaszana prędkość | 1.5962 | 2.1025 | 3.0496 | 3.7575 |
| Pełny ridge 768D | 0.9914 | 1.0034 | 1.0445 | 1.1433 |
| PCA + ridge AR(1) | 0.9638 | 0.9619 | 0.9494 | 0.9569 |
| PCA + ridge z pamięcią | 0.9926 | 0.9806 | 0.9560 | 0.9607 |

![Błędy pełnych wektorów i podłoga PCA](PROGNOZA_PILOT.png)

PCA AR(1) było lepsze od persystencji w 16 z 16 kombinacji początek/horyzont po zagregowaniu obrazów. Przy H=100 wynik był lepszy w 40 z 48 kombinacji obraz/początek prognozy; dla 1 obrazów średnia po początkach nie była lepsza. Nie jest to poprawa każdej cząsteczki ani każdej próby.

## Co pozostaje poza redukcją

Wybrane 8 osi opisywało średnio 41.25 procent ważonej przeszłej wariancji. Znaczna część przyszłego przemieszczenia pozostawała poza tą bazą; w rekonstrukcji stosowano dla niej persystencję, a błąd liczono nadal w pełnych 768D.

| Horyzont | Energia przyszłego ruchu poza PCA | Stosunek zwykłej średniej L2 do persystencji |
|---:|---:|---:|
| 10 | 63.35% | 0.9675 |
| 20 | 69.90% | 0.9691 |
| 50 | 68.83% | 0.9537 |
| 100 | 70.88% | 0.9611 |

Podłoga poza PCA jest oceną reprezentacji przy użyciu prawdziwych celów testowych; predyktor nie miał do niej dostępu. Nie dowodzi ona, że pozostały ruch jest fundamentalnie losowy lub nieprzewidywalny przez bogatszy model.

## Ustawienia wybrane przed testem

| Rodzina | Ranga PCA | Alpha ridge | Średni stosunek błędu na walidacji |
|---|---:|---:|---:|
| Pełny ridge 768D | pełne 768D | 10.0 | 1.0979 |
| PCA + ridge AR(1) | 8 | 0.1 | 0.9612 |
| PCA + ridge z pamięcią | 8 | 0.1 | 0.9740 |

Pełny model korzysta z dual ridge, bez identyfikowania dowolnej macierzy 768×768 z kilkudziesięciu próbek. PCA jest wspólną bazą 768×r dopasowaną z przeszłych odchyleń poszczególnych tokenów. Każdy token ma własny predyktor. To nie jest model jawnych oddziaływań między tokenami.

## Granice wyniku

- Test dotyczy prognozy ruchu w jednym biegu. Tokeny, obrazy i sąsiednie okna nie są niezależnymi replikacjami uczenia.
- Nie potwierdza to tożsamości konceptu narzędzia. PCA opisuje przeszłą zmienność; osobne badanie par z/bez narzędzia jest potrzebne do semantycznej interpretacji.
- Mniejszy błąd reprezentacji nie jest pomiarem poprawy mAP detektora. Takiego testu tutaj nie wykonywano.
- Brak zapisanego dziennika wykonanych minibatchy tego biegu. Nie użyto historii wejść z innych biegów, gradientów ani momentów AdamW.
- Zmiana tokenów obejmuje zmiany wag i buforów BN. Nie oddzielono tutaj ich przyczynowych wkładów.
- Historyczny podział obrazów fit/holdout nie jest w tym pilocie testem na nieznanych obrazach: każdy predyktor korzysta z dostępnej przeszłości swojego obrazu.
- Nie przeprowadzono testów istotności opartych na fałszywej niezależności 218 tokenów. Liczby i zakresy błędów są opisowe.

## Odtwarzanie i artefakty

Po zakończonym, sprawdzonym pobraniu uruchomić kolejno `pilot_zgodnosci.py`, `prognoza.py`, `raport.py` środowiskiem z torch, timm, NumPy, SciPy, psutil i matplotlib. Nie wymaga to treningu ani serwera Eden.

- [Plan](PLAN.md), [konfiguracja](CONFIG.json), [audyt metody](AUDYT_METODY.md).
- [Pełne metryki i indeksy wszystkich cząsteczek](METRYKI_PILOT.json).
- [Zamrożony wybór ustawień i siatka walidacyjna](WYBOR_WALIDACJA.json).

Czas prognozowania i dopasowań: 9.4 s. Szczyt pamięci procesu: 365899776 bajtów. Hash źródłowego pliku: `d148b9649154b1ac4957da2c7d39184921ad852b195fdefe0125e28b284c5576`.
