# Pilot doboru klatek: wynik detektora

Eksploracyjny pilot na jednym seedzie treningu (42), z selekcją seed 42. Pięć ramion trenowano przez 10 epok ze wspólnej inicjalizacji COCO. Każdy zbiór zawiera wspólne L0: 500 obrazów.

**Ustalona bramka praktyczna: true (spełniona).** Warunek: AP Koopman + Langevin minus AP błędu odczytu ≥ 0.004 oraz AP Koopman + Langevin minus AP placebo > 0. Nie wykonano testu istotności statystycznej. Wynik uzasadnia replikację z kolejnymi seedami.

## Niezależna ocena końcowa

Metryka główna: AP dla IoU 0.50:0.95, skala 0–1. Oceniono zwykłe wagi ostatniej epoki na pełnym val; każde ramię: 12043 obrazów. Zbioru test nie użyto. Wartości wczytano z ukończonego raportu pilota; generator nie uruchamia ponownej ewaluacji modeli.

| Ramię | Klatki treningowe | AP ostatniej epoki | AP50 | Czas treningu |
|---|---:|---:|---:|---:|
| Losowy 1500 | 1500 | 0.6555 | 0.8883 | 1088.81 s |
| Błąd odczytu | 1500 | 0.6460 | 0.8855 | 1059.53 s |
| Koopman + Langevin | 1500 | 0.6732 | 0.9046 | 1073.06 s |
| Placebo | 1500 | 0.6522 | 0.8949 | 1058.81 s |
| Losowy 3000 | 3000 | 0.6712 | 0.8907 | 1223.03 s |

| Kontrast końcowego AP | Różnica |
|---|---:|
| Koopman + Langevin minus Losowy 1500 | +0.0177 |
| Koopman + Langevin minus Błąd odczytu | +0.0272 |
| Koopman + Langevin minus Placebo | +0.0210 |
| Koopman + Langevin minus Losowy 3000 | +0.0019 |

## Krzywe uczenia i ekspozycje

![Uczenie detektora](WYKRES_UCZENIA.png)

Linie przedstawiają walidację wag EMA zapisaną podczas treningu (use_ema=true, eval_base_model=false). Romby pokazują niezależny wynik ostatnich zwykłych wag z tabeli powyżej. Te wartości mogą się różnić.

Ekspozycje to przybliżenie: ukończone epoki × liczba unikalnych klatek treningowych. Nie jest to pomiar kroków optymalizatora, czasu ani FLOP. Losowy 3000 ma dwukrotnie więcej nominalnych ekspozycji na epokę niż każde ramię 1500. Receptura ma rozgrzewkę learning rate przez dwie epoki, więc dla różnych budżetów obejmuje ona różną liczbę aktualizacji. Porównanie po ekspozycjach nie izoluje zatem w pełni wpływu liczby kroków i harmonogramu learning rate; kontrola 1500 wobec 1500 zachowuje oba.

Wspólny obserwowany zakres ekspozycji: 3000–15000 (szare pole). Średnie AP poniżej to pole pod krzywą podzielone przez szerokość tego zakresu; interpolacja liniowa między obserwacjami, bez ekstrapolacji.

| Ramię | Średnie AP we wspólnym zakresie ekspozycji |
|---|---:|
| Losowy 1500 | 0.6746 |
| Błąd odczytu | 0.6676 |
| Koopman + Langevin | 0.6816 |
| Placebo | 0.6749 |
| Losowy 3000 | 0.6708 |

## Co rzeczywiście wybrał selektor

Dobór jest statyczny: historię filmu wykorzystano przed treningiem detektora do zamrożenia zbioru. Selektor nie reagował na kolejne checkpointy ani kroki SGD. Cząstka to pojedynczy patch na stałej pozycji siatki obrazu; nie jest to potwierdzona trajektoria punktu materialnego narzędzia. Etykiety prostokątów uczestniczyły w rankingu z już oznaczonej puli.

Brakujące okna historii: 624. Kandydaci z dodatnim zyskiem: 4084.

| Ramię | Dodane klatki | Według oceny | Losowo | Losowe uzupełnienie rankingu | Odstęp żądany → końcowy | Poluzowanie |
|---|---:|---:|---:|---:|---:|---|
| Losowy 1500 | 1000 | 0 | 1000 | 0 | 5 → 5 | nie |
| Błąd odczytu | 1000 | 1000 | 0 | 0 | 5 → 5 | nie |
| Koopman + Langevin | 1000 | 1000 | 0 | 0 | 5 → 5 | nie |
| Placebo | 1000 | 1000 | 0 | 0 | 5 → 5 | nie |
| Losowy 3000 | 2500 | 0 | 2500 | 0 | 5 → 5 | nie |

Losowe uzupełnienie oznacza brak wystarczającej liczby dodatnich ocen przy ograniczeniu odstępu czasowego. Losowanie jako podstawowa strategia ramion losowych jest wykazane osobno i nie jest takim uzupełnieniem. Odstęp podano w klatkach filmu, z uwzględnieniem L0.

Udział wspólnych **dodanych** klatek: mianownikiem jest liczba klatek dodanych w ramieniu wiersza. Nie jest to indeks Jaccarda; tabela może być niesymetryczna, zwłaszcza przy 1500 wobec 3000.

| Ramię wiersza / ramię kolumny | Losowy 1500 | Błąd odczytu | Koopman + Langevin | Placebo | Losowy 3000 |
|---|---:|---:|---:|---:|---:|
| Losowy 1500 | 100.0% | 9.8% | 7.9% | 9.4% | 100.0% |
| Błąd odczytu | 9.8% | 100.0% | 50.9% | 43.8% | 25.3% |
| Koopman + Langevin | 7.9% | 50.9% | 100.0% | 41.8% | 23.2% |
| Placebo | 9.4% | 43.8% | 41.8% | 100.0% | 23.0% |
| Losowy 3000 | 40.0% | 10.1% | 9.3% | 9.2% | 100.0% |

## Koszt procedury

- Ekstrakcja tokenów wspólnego selektora: 475.58 s.
- Dopasowanie i selekcja: 78.40 s.
- Czasy treningów: tabela wyników; brak pola oznacza brak pomiaru, nie zero.

Koszt wspólnego detektora L0, transferu danych i pełnej procedury nie jest tu kompletnie zmierzony. Krótszy trening na 1500 klatkach sam w sobie nie dowodzi przyspieszenia całej procedury. Czasów równoległych biegów nie należy sumować jako czasu zegarowego.

## Diagnostyka patchy

Nie dostarczono DIAGNOSTYKA.json. Raport nie wyciąga wniosku o zmianie informacji zawartej w patchach między checkpointami.

## Źródła i powtarzalność

Opis procedury: [README.md](dokumentacja/README.md). Uzasadnienie teoretyczne: [LITERATURA.md](dokumentacja/LITERATURA.md). Są to lokalne kopie dokumentacji z chwili generowania raportu.

Generator sprawdza zgodność końcowych AP, kontrastów, bramki i całek z krzywych oraz zgodność liczności selekcji. Nie zastępuje audytu pierwotnych checkpointów i predykcji. Poniższe hashe identyfikują dokładne wejścia raportu.

- [pilot](<../WYNIKI_PILOT_Z_CZASEM.json>): SHA-256 `d68171c439bd08ad74f5c13f45c7e8978206dd4f8ad340e3d00e35ac68561ea7`.
- [selection](<../selection/BIEG_selekcja.json>): SHA-256 `48c3db83d997088ed69a6052c5bebd5e9c173accfd97648fd23958d9e1525321`.
- [raport.py](<../../raport.py>): SHA-256 `eb90003d0d98bef77edbad0ecf4ec12d2a2c809a2e4fcf864211b00b4152e230`.
- [README.md](<../../README.md>): SHA-256 `cfd5f287195a2f9679f52544ef8e8e1817e7493daf865b829b47782d571452ef`.
- [LITERATURA.md](<../../LITERATURA.md>): SHA-256 `35d556d3022b7187c2b265f8280b9ebe6d15b5a461172a00c31643ddc8f2b7c2`.
