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

![Odczyt patchy](WYKRES_PATCHY.png)

Odczyt ridge i standaryzację dopasowano osobno przy każdym checkpoincie wyłącznie na wspólnym L0 (500 obrazów). Ocena dotyczy stałej próby 192 obrazów val. Etykieta patcha pochodzi z prostokąta narzędzia; nie jest segmentacją ani etykietą tożsamości konkretnego narzędzia. Wzrost AUROC oznacza skuteczniejszy dopasowany odczyt liniowy tej słabej etykiety w użytej projekcji kanałów. Nie mierzy całej informacji reprezentacji ani nie dowodzi wzrostu jakości śledzenia.

| Ramię | Epoka | AUROC | AP patchy | Balanced BCE | CKA do poprzedniego checkpointu |
|---|---:|---:|---:|---:|---:|
| Losowy 1500 | 2 | 0.8914 | 0.4226 | 0.5420 | niedostępne |
| Losowy 1500 | 4 | 0.9026 | 0.4445 | 0.5315 | 0.9719 |
| Losowy 1500 | 6 | 0.9035 | 0.4504 | 0.5301 | 0.9824 |
| Losowy 1500 | 8 | 0.9027 | 0.4503 | 0.5324 | 0.9831 |
| Losowy 1500 | 10 | 0.9037 | 0.4547 | 0.5315 | 0.9851 |
| Błąd odczytu | 2 | 0.8900 | 0.4169 | 0.5447 | niedostępne |
| Błąd odczytu | 4 | 0.8986 | 0.4314 | 0.5364 | 0.9710 |
| Błąd odczytu | 6 | 0.9005 | 0.4412 | 0.5348 | 0.9902 |
| Błąd odczytu | 8 | 0.8994 | 0.4396 | 0.5363 | 0.9932 |
| Błąd odczytu | 10 | 0.9007 | 0.4398 | 0.5359 | 0.9920 |
| Koopman + Langevin | 2 | 0.8937 | 0.4255 | 0.5416 | niedostępne |
| Koopman + Langevin | 4 | 0.9102 | 0.4594 | 0.5270 | 0.9618 |
| Koopman + Langevin | 6 | 0.9103 | 0.4570 | 0.5267 | 0.9875 |
| Koopman + Langevin | 8 | 0.9152 | 0.4834 | 0.5238 | 0.9845 |
| Koopman + Langevin | 10 | 0.9154 | 0.4878 | 0.5234 | 0.9858 |
| Placebo | 2 | 0.8931 | 0.4254 | 0.5410 | niedostępne |
| Placebo | 4 | 0.9067 | 0.4514 | 0.5287 | 0.9614 |
| Placebo | 6 | 0.9072 | 0.4539 | 0.5284 | 0.9836 |
| Placebo | 8 | 0.9069 | 0.4638 | 0.5285 | 0.9930 |
| Placebo | 10 | 0.9088 | 0.4604 | 0.5281 | 0.9865 |
| Losowy 3000 | 2 | 0.9037 | 0.4414 | 0.5326 | niedostępne |
| Losowy 3000 | 4 | 0.9101 | 0.4564 | 0.5265 | 0.9704 |
| Losowy 3000 | 6 | 0.9105 | 0.4477 | 0.5280 | 0.9874 |
| Losowy 3000 | 8 | 0.9154 | 0.4654 | 0.5230 | 0.9853 |
| Losowy 3000 | 10 | 0.9134 | 0.4655 | 0.5270 | 0.9869 |

Balanced BCE używa sigmoidy wyniku ridge; nie jest oceną skalibrowanego prawdopodobieństwa. CKA opisuje wyłącznie zmianę reprezentacji tych samych patchy, bez odczytu znaczenia. Diagnostyka nie uczestniczyła w rankingu. Pięć checkpointów nie służy do dopasowania Langevina w osi postępu treningu.

## Źródła i powtarzalność

Opis procedury: [README.md](dokumentacja/README.md). Uzasadnienie teoretyczne: [LITERATURA.md](dokumentacja/LITERATURA.md). Są to lokalne kopie dokumentacji z chwili generowania raportu.

Generator sprawdza zgodność końcowych AP, kontrastów, bramki i całek z krzywych oraz zgodność liczności selekcji. Nie zastępuje audytu pierwotnych checkpointów i predykcji. Poniższe hashe identyfikują dokładne wejścia raportu.

- [pilot](<../WYNIKI_PILOT_Z_CZASEM.json>): SHA-256 `d68171c439bd08ad74f5c13f45c7e8978206dd4f8ad340e3d00e35ac68561ea7`.
- [selection](<../selection/BIEG_selekcja.json>): SHA-256 `48c3db83d997088ed69a6052c5bebd5e9c173accfd97648fd23958d9e1525321`.
- [diagnostics](<../DIAGNOSTYKA.json>): SHA-256 `cf0764802090141fe4d7201d5b826c55f2db747ee4f1c4ffa6399b44db92e995`.
- [raport.py](<../../raport.py>): SHA-256 `eb90003d0d98bef77edbad0ecf4ec12d2a2c809a2e4fcf864211b00b4152e230`.
- [README.md](<../../README.md>): SHA-256 `8e16a66c61ceffdc15b91e6e72f23ac41f723e78f9cf3b4d7f90f8ec8a3d5d8d`.
- [LITERATURA.md](<../../LITERATURA.md>): SHA-256 `35d556d3022b7187c2b265f8280b9ebe6d15b5a461172a00c31643ddc8f2b7c2`.
