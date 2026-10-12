# Wynik etapu selekcji tokenowej, 22 września 2026

Ekstrakcja i dopasowanie na Edenie zakończyły się poprawnie. Wynik dotyczy predykcji tokenów i utworzenia podzbiorów. Nie jest jeszcze wynikiem porównania treningów detektora.

## Aparat

57 073 klatki treningowe, 118 klipów, 117 grup operacji. Każdy obraz ma 576 osobnych tokenów ostatniej warstwy ViT. Stała projekcja 384 kanałów do 24 zachowuje indeks każdego patcha. Trzy foldy po operacjach, po 20 000 próbkowanych trajektorii patchy na fold. Dopasowano pełne D+JP+W+Z+N ze skalą własną patcha z wariantu optymalizacja 95. Nie używano walidacji do dopasowania.

## Prognoza poza dopasowanymi operacjami

MSE jest liczone w kanałach normalizowanych osobno z FIT danego foldu. Persistence oznacza skopiowanie ostatniego zaobserwowanego tokenu. Wartości pochodzą z zapisanych predykcji tych samych ocenianych patchy.

| Fold | Persistence | Koopman | Rozszerzone GLE | Średnia K+GLE |
|---|---:|---:|---:|---:|
|0|0.40985|0.36903|0.33297|0.33742|
|1|0.41401|0.37290|0.33696|0.34074|
|2|0.41395|0.37312|0.33975|0.34238|

Każdy z pięciu członów ma niezerowy RMS po dopasowaniu. Nie dowodzi to jego osobnej przydatności; taki wniosek wymaga ablacji. GLE ma mniejszy błąd niż persistence w każdym foldzie, ale lepsza predykcja latentu sama nie dowodzi użyteczności treningowej klatki.

## Zamrożone wybory

Pula ma 10 637 klatek, z czego 500 stanowi wspólne L0. Dla 624 klatek brakuje pełnej ośmiokrokowej historii; mają zerowy score dynamiki. W całej puli 4084 klatki mają dodatni zysk odczytu, a po odjęciu L0 pozostaje 3885 kandydatów z dodatnim score.

Wariant K+GLE wybrał pełne 1000 dodatkowych klatek według score, bez losowego uzupełnienia i bez relaksowania odstępu pięciu klatek. Z wyborem według samego błędu pokrywa się 509 z 1000 nowych klatek, z placebo 418, z losowaniem 79. Nie jest więc kopią samego błędu ani losowania.

Placebo zachowuje operację, przedział trudności i dostępność historii. Każde ramię ma 1500 obrazów łącznie ze wspólnym L0. Dodatkowe ramię random_large ma 3000, a random 1500 jest jego podzbiorem.

## Koszt i ograniczenia

Ekstrakcja tokenów: 475.58s. Dopasowanie trzech foldów wraz z rankingiem: 78.40s. Łącznie około 9.23min, bez wcześniejszego treningu L0, przygotowania danych i transferu. Koszt ten trzeba uwzględnić przy ocenie przyspieszenia całego procesu; losowy wybór nie potrzebuje modelu dynamiki.

Odczyt narzędzie/tło jest pomocniczym ridge, nie pełną głową detektora. Etykiety patchy pochodzą z prostokątów, nie z segmentacji. Wyznaczony zysk jest hipotezą przydatności uczenia na danej klatce. Pierwszy test tej hipotezy to pięć uruchomionych treningów i niezależny końcowy AP. Kryterium nie jest zmieniane po tych wynikach.

Surowe wyniki: `wyniki/selection/BIEG_selekcja.json`, `SCORES.npz`, `DIAGNOSTYKA_KLATEK.jsonl`, trzy zapisane modele i pięć wariantów wyboru. Hash cache i kodu znajduje się w raporcie selekcji.
