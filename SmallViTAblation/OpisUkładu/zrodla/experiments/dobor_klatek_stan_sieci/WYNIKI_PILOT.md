# Wynik pilota z 22 września 2026

**W tym pojedynczym biegu dobór Koopman–GLE poprawił detektor uczony na 1500 klatkach.** Końcowy AP wzrósł o 0.01770 względem losowych 1500, o 0.02719 względem samego błędu odczytu i o 0.02097 względem placebo. Spełniono bramkę ustaloną przed treningiem. Nie jest to jeszcze potwierdzenie statystyczne ani dowód działania każdego członu równania osobno.

Pięć biegów RF-DETR Nano na Edenie, seed 42, 10 epok, wspólna inicjalizacja COCO. Ocena zwykłych wag ostatniej epoki na 12 043 obrazach walidacyjnych z 25 operacji. Zbiór test pozostał nietknięty. Każde ramię zawierało wspólne 500 obrazów L0.

| Dobór | Obrazy treningowe | Końcowy AP@[.50:.95] | Czas treningu z walidacjami |
|---|---:|---:|---:|
| Losowy | 1500 | 0.65547 | 18.15 min |
| Największy błąd odczytu patchy | 1500 | 0.64597 | 17.66 min |
| **Koopman + rozszerzony GLE** | **1500** | **0.67317** | **17.88 min** |
| Placebo | 1500 | 0.65220 | 17.65 min |
| Losowy, większy zbiór | 3000 | 0.67122 | 20.38 min |

Różnica względem 3000 wynosi tylko +0.00194 AP. Można powiedzieć, że w tym biegu połowa obrazów osiągnęła podobny końcowy wynik. Nie można jeszcze twierdzić, że metoda ogólnie zastępuje dwa razy większy zbiór. Wynik dotyczy ustalonej ostatniej epoki; większy zbiór osiągał wcześniej wyższe AP EMA niż mniejsze ramiona.

## Co połączyłem

Cząstka jest jednym patchem ViT. Każda klatka zachowuje siatkę 24×24 osobnych tokenów. Redukowana jest tylko liczba kanałów, z 384 do 24. Modeluje się historię tego samego miejsca siatki w filmie; narzędzie może przechodzić między patchami.

Koopman przewiduje rozwój reprezentacji przez operator liniowy w rozszerzonej przestrzeni cech. Adaptacja GLE z Grant dodaje dryf zależny od stanu i ruchu, pamięć z zanikiem i oscylacją, wpływ wcześniejszych metadanych, kotwicę ostatniego odczytu narzędzia oraz filtr wcześniejszych reszt. Zachowuje również własną skalę ruchu patcha i uczone wykładniki gamma. [Kod równania](token_model.py), [moduły członów](model.py), [pochodzenie i różnice względem TS](POCHODZENIE.json).

Pomocniczy odczyt ocenia ten sam patch jako obszar narzędzia albo tło. Klatka dostaje wysoką ocenę, gdy odczyt jej bieżących tokenów jest błędny, lecz tokeny przewidziane z historii zmniejszają ten błąd:

```text
score = max(0, błąd_odczytu_tokenów_bieżących − błąd_odczytu_tokenów_przewidzianych)
```

To hipoteza, że historia wskazuje brak możliwy do uzupełnienia przez uczenie. Nie jest matematyczną gwarancją najlepszego gradientu. Testem hipotezy były powyższe nowe treningi. Ranking używa etykiet kandydatów, więc jest selekcją z oznaczonego zbioru. Nie testuje oszczędzania kosztu anotacji.

## Tempo i koszt

Koopman–GLE miał wyższe AP EMA od losowych 1500 w każdej z 10 epok. Średnie AP EMA w wspólnym zakresie 3000–15000 nominalnych ekspozycji wyniosło 0.68155 wobec 0.67458 dla losowych 1500. Te krzywe dotyczą EMA; końcowa tabela dotyczy zwykłych wag. Przy porównaniu różnych budżetów rozgrzewka learning rate obejmuje różną liczbę aktualizacji, więc oś ekspozycji nie usuwa wszystkich różnic receptury.

Ekstrakcja tokenów zajęła 475.58 s, a dopasowanie i selekcja 78.40 s. Selekcja plus trening Koopmana–GLE to około 27.12 min, wobec 20.38 min samego treningu losowych 3000. Nie obejmuje to kosztu uzyskania wspólnego checkpointu L0 i przygotowania danych. **Ten pilot nie wykazał przyspieszenia całej procedury**, choć wykorzystał mniej obrazów w treningu końcowym i uzyskał lepszą krzywą niż losowy dobór tego samego budżetu.

## Checkpointy i granice wniosku

Zdiagnozowano wszystkie 25 checkpointów: po epokach 2, 4, 6, 8, 10 każdego ramienia. Odczyt ridge dopasowywano osobno na wspólnym L0, a oceniano na tych samych 192 obrazach walidacyjnych. Pomiar obejmował 110 592 tokeny na checkpoint. Diagnostyka nie wpływała na wybrany zbiór ani na treningi; 25 checkpointów nie stanowi 25 niezależnych powtórzeń eksperymentu.

| Ramię | AUROC patchy, epoka 2 | AUROC patchy, epoka 10 | AP patchy, epoka 10 |
|---|---:|---:|---:|
| Losowy 1500 | 0.89137 | 0.90374 | 0.45473 |
| Błąd odczytu | 0.89002 | 0.90066 | 0.43983 |
| Koopman–GLE | 0.89368 | 0.91544 | 0.48780 |
| Placebo | 0.89312 | 0.90883 | 0.46038 |
| Losowy 3000 | 0.90371 | 0.91342 | 0.46546 |

Koopman–GLE przewyższał losowe 1500 w AUROC we wszystkich pięciu checkpointach. Wynik dotyczy skuteczności tego konkretnego odczytu słabej etykiety prostokąta, nie całej informacji zawartej w reprezentacji. Etykieta rozróżnia obszar oznaczonego prostokąta od reszty obrazu; nie oznacza, że poza nim nigdy nie ma innego narzędzia.

[Wykres zmian odczytu patchy](wyniki/raport_koncowy/WYKRES_PATCHY.png) pokazuje pełną skalę i zbliżenie tych samych pomiarów. [Mapy na obrazach](wyniki/mapy_patchy/MAPY_PATCHY.png) pokazują epoki 2 i 10 dla losowania i Koopmana–GLE. Wybrano z góry pierwsze trzy posortowane identyfikatory próby walidacyjnej; wszystkie należą do jednego filmu. Mapy są ilustracją, nie reprezentatywnym testem ani segmentacją, a kolory są nieskalibrowanym odczytem ridge po sigmoidzie.

Wybór był jednorazowy, na podstawie reprezentacji detektora L0 i historii filmu. Nie wykonywano adaptacyjnej wymiany klatek między epokami ani dopasowania GLE do osi zmian wag. Kod ekstrakcji i rankingu przyjmuje dowolny checkpoint, lecz zamknięta pętla ponownego wyboru wymaga osobnego eksperymentu. Rozpoznawanie obszaru prostokąta nie jest dowodem tożsamości konkretnego narzędzia; jakości śledzenia nie mierzono.

Wzmocnienie dowodu wymaga sparowanych powtórzeń treningu, a następnie oceny na odłożonym teście. Ablacje Koopman-only, GLE-only oraz poszczególnych członów są potrzebne, aby przypisać zysk konkretnemu mechanizmowi. Tego pilot nie rozstrzyga.

[Wyniki maszynowe z czasami i hashami](wyniki/WYNIKI_PILOT_Z_CZASEM.json), [pełny raport z wykresami](wyniki/raport_koncowy/WYNIKI_PILOT.md), [diagnostyka 25 checkpointów](wyniki/DIAGNOSTYKA.json), [audyt wag i EMA](AUDYT_WAG.md), [plan sprzed wyników](PLAN.md), [uzasadnienie teoretyczne i literatura](LITERATURA.md).

Weryfikacja: 81 testów jednostkowych i syntetycznych przeszło; wszystkie pięć treningów i końcowych ocen zakończyło się poprawnie. Dodatkowo sprawdzono zgodność 34 pobranych artefaktów metadanych z hashami źródłowymi, wspólną inicjalizację, zwykłe wagi 25 checkpointów oraz równość wszystkich tensorów `last.pth` i ostatniego checkpointu każdego ramienia.
