# Jak tańczyły cząsteczki w latencie K1000

Lokalny, interaktywny podgląd rzeczywistych zapisów z eksperymentu niezależnych optymalizacji 1500 obrazów. To ruch aktywacji przy zamrożonych wagach K1000, a nie ruch wywołany dalszym uczeniem wag.

## Otwieranie

Podczas tej sesji podgląd działa pod adresem http://127.0.0.1:8774/ . Dane, bibliotekę Plotly i stronę przechowujemy lokalnie. Nie wymaga połączenia z internetem po przygotowaniu danych.

Jeśli podgląd jest zamknięty, uruchom `URUCHOM_WIZUALIZACJE.ps1` z tego folderu. Nie otwieraj samego HTML przez `file://`: strona pobiera dane z lokalnego serwera.

## Trzy widoki

1. **Cały zbiór:** wszystkie 900 000 cząsteczek, bez próbkowania. Bezpośredni renderer WebGL 2 rysuje je jednym wywołaniem `drawArrays(POINTS)`. Zapisane pełne stany to kroki 0, 100 i 500; nie odtwarzamy brakujących kroków.
2. **Jeden obraz:** wszystkie 600 cząsteczek w tych trzech rzeczywistych stanach. Mapa 20 × 30 przedstawia normę ruchu w oryginalnej siatce patchy.
3. **Ruch co krok:** panel 16 obrazów, po 8 śledzonych cząsteczek w każdym; łącznie 128 trajektorii z 501 rzeczywistymi stanami. Można odtwarzać każdy krok lub przeskakiwać co 5 albo 10 zapisanych kroków. Ślad obejmuje ostatnie 40 rzeczywistych przejść.

Suwak wybiera krok optymalizacji, a wykres obok pokazuje rzeczywiste IoU i confidence w tym samym kroku. Wspólny timeline oznacza jednakowy numer aktualizacji niezależnych optymalizacji, nie jednoczesność w czasie rzeczywistym. Nie dopasowujemy osi czasu osobno do momentów sukcesu.

## Jak czytać wykres

- Każda cząsteczka zachowuje identyfikator obrazu i tokenu. Kliknięcie punktu w chmurze GPU lub wskazanie punktu w pozostałych widokach pokazuje oba identyfikatory.
- Jedna wspólna średnia i baza PCA obowiązuje wszystkie obrazy oraz kroki. Osie nie są dopasowywane ponownie podczas animacji.
- Rzut 3D pokazuje tylko część geometrii 768D. Procent wariancji dotyczy próbki dopasowania PCA i nie oznacza procentu poprawnej detekcji.
- W widokach zbioru i pojedynczego obrazu kolor przedstawia pełną normę L2 zmiany w 768D. W dokładnym panelu kolor oznacza tożsamość patcha, a długość ruchu pokazuje wykres słupkowy. Skale kolorów i zakresy osi pozostają stałe w czasie wybranego widoku.
- Przełącznik **przesunięcia względem startu** odejmuje początkową pozycję każdej cząsteczki. Ułatwia porównanie ruchu, ale wtedy wspólny punkt startu na wykresie nie oznacza jednakowych naturalnych reprezentacji.
- `after`, czyli wybrany przez oracle najlepszy odwiedzony stan, może mieć inny numer kroku dla każdego obrazu. Nie wstawiamy go jako wspólnego kroku 500.

## Wyniki i pliki

Statyczne wykresy PNG i PDF obejmują trzy zapisane stany całego zbioru, dokładne trajektorie jednego obrazu panelu oraz zbiorcze zmiany detekcji. Szczegóły źródeł, mapowania i transferu są w `README_DANYCH.md` i `data/`.

Tabela A i B odnosi się do wcześniejszych prób przeniesienia osobnych poprawek na wspólne reguły. Nie obejmuje nowego eksperymentu bezpośredniej optymalizacji wspólnej korekty per patch. Wyniki oracle korzystają z etykiet tych samych obrazów i nie mierzą generalizacji.

Źródła: `outputs/vit_k1000_1500_pelne_latenty_90_20261008` oraz odpowiadający mu zapis Edena. Kod odtworzenia podglądu: `experiments/vit_k1000_taniec_czasteczek_20261009`. Nie modyfikujemy modelu ani surowych trajektorii.

## Renderer GPU

Dane docierają jako binarne Float32: x, y, z oraz norma zmiany w pełnych 768 wymiarach. Kolejność punktów to obraz × 600 + patch. Każdy z trzech rzeczywistych stanów zajmuje 14,4 MB, a komplet buforów wierzchołków 43,2 MB. To nie obejmuje pamięci przeglądarki, buforów obrazu ani odczytu przy wybieraniu punktu.

Shader odejmuje pozycję początkową tylko w trybie przesunięć, obraca kamerę i nadaje kolor. Nie oblicza nowych latentów i nie interpoluje czasu. Zachowana jest jedna skala długości dla wszystkich trzech osi; ramka PCA i skala pozostają stałe w czasie. Gęste punkty mogą zasłaniać się na ekranie. Licznik oznacza liczbę przesłanych wierzchołków, nie liczbę rozróżnialnych pikseli.

Przeciągnięcie obraca, kółko myszy przybliża, a przełącznik automatycznego obrotu pokazuje zmierzoną częstotliwość rysowania. Renderer zatrzymuje ciągłe odświeżanie w tle oraz w innych widokach. Suwak widoczności zmienia przezroczystość punktów. Przy braku WebGL 2 komunikat jawnie zgłasza ograniczenie; pozostałe widoki nadal używają Plotly.

`WERYFIKACJA_SHADERA.json` potwierdza bitową zgodność pozycji i norm wszystkich 900 000 punktów we wszystkich trzech krokach z oryginalnym NPZ.

## Gęsty replay przygotowywany 9 października

Nowa rodzina `vit_k1000_replay900k_co5_20261009` ponownie liczy 500 rzeczywistych aktualizacji każdego z tych samych 1500 obrazów. Wagi K1000 pozostają zamrożone. Odtwarzane są aktywacje oraz optymalizator latentów, a nie dalszy trening sieci. Komplet 101 stanów co 5 kroków zostanie udostępniony po kontroli zgodności z historycznymi stanami 0, 100 i 500 oraz po weryfikacji transferu. Samo przygotowanie czy uruchomienie kodu nie oznacza dostępności kompletu.

Manifest `data/dense/manifest.json` jest publikowany dopiero po pobraniu i sprawdzeniu SHA256 wszystkich 101 plików. Do tego czasu podgląd zachowuje trzy stare pełne stany. Gęsty replay nie wykorzystuje interpolacji. Panel 128 dawnych trajektorii oraz widok jednego obrazu pozostają osobnymi zakresami obserwacji.

Zapis gęstej wizualizacji zajmie 1 454 400 000 bajtów. Renderer przechowuje w GPU stan początkowy i najwyżej trzy inne stany, do 57,6 MB buforów wierzchołków. Pozostałe stany czyta na żądanie. Stała baza PCA3 służy wyłącznie wyświetlaniu; analiza wzorców w PyDMD i deeptime ma własne dane oraz rozłączne filmy dopasowania i oceny.

Przebieg pobierania opisują `data/dense/FINALIZACJA.json` i `data/dense/TRANSFER.json`. Analiza temporalna jest odrębnym zadaniem; ukończony transfer wizualizacji nie oznacza ukończenia analizy ani potwierdzenia wspólnej reguły poprawiającej detekcję.
