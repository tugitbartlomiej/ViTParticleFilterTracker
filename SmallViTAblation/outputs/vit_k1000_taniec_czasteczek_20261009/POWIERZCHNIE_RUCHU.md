# Powierzchnie ruchu ViT

Wzorcem były wszystkie pięć HTML-i STUDNIE_3D_fav z Research/DiagnosticTool/koopman_m5/nagrania. Przejęto sposób prezentacji: półprzezroczysta powierzchnia, kontury, paleta granat–zieleń–piasek, niski aspekt osi Z i kolorowe trajektorie. Nie kopiowano danych TFT do ViT.

## Dane i sposoby prezentacji

Wybierz obraz w głównym selektorze, potem „Wczytaj trajektorie wybranego zdjęcia”. Powierzchnia obejmuje 600 cząsteczek tego obrazu × 101 rzeczywistych stanów 0–500 co 5 kroków. Można wyświetlać wybrany patch, 24, 60 lub 600 śladów, trzy płaszczyzny PCA i filtr rodziny kierunku ruchu. Krajobraz powstaje ze wszystkich 600 trajektorii niezależnie od liczby pokazanych śladów.

- Studnie gęstości: histogram 64×64, wygładzenie jądrem Gaussa o promieniu 3 komórek, U=−ln((gęstość+0.02)/(maksimum+0.02)). To mapa odwiedzin, nie fizyczny potencjał ani loss. Wysokości śladów są interpolowane na tej powierzchni; XY pozostaje rzeczywistym rzutem.
- Powierzchnia prędkości: lokalna wygładzona średnia normy przyrostu 3D PCA dzielona przez rzeczywisty odstęp kroków. Komórki bez obserwacji są puste. Nie jest to pełna prędkość 768D.
- Krok–patch–ruch: powierzchnia rzeczywistej normy przemieszczenia w pełnych 768D; osie X=krok, Y=numer patcha, Z=norma.

Kolor rodziny to dominująca oś końcowego przesunięcia w 3D PCA (±PC1/2/3), a szary oznacza normę do 10. percentyla obrazu. To jawna opisowa klasyfikacja kierunków; nie wynik DMD/Koopmana ani deeptime. Czerwone zdarzenia oznaczają zwrot ponad 90 stopni lub prędkość ponad trzykrotną medianę danej trajektorii. Nie należy ich utożsamiać z matematycznymi osobliwościami. Wysokie zdarzenia na początku mogą po prostu wynikać z szybkiej początkowej optymalizacji.

Przycisk „Pokaż bieżący krok i patch” synchronizuje wybrany patch i znacznik kroku. Ręczna zmiana suwaka kroku lub pola patcha również aktualizuje widok. Kliknięcie śladu wybiera jego patch w głównej wizualizacji i na zdjęciu. Po zmianie zdjęcia trzeba wczytać jego trajektorie ponownie; identyfikator analizowanego obrazu jest zawsze wyświetlany.

## Weryfikacja

Sprawdzono rzeczywisty obraz 1469, 101 stanów, 600 cząsteczek. Testy obliczeń obejmują prędkość na krok, ruch prostoliniowy, zwrot 180 stopni, zerowy ruch, skończone gęstości i puste komórki pola prędkości. Kod: terrain-inline.js, upgrade_terrain.py, test_terrain.cjs. Po odświeżaniu innych rozszerzeń HTML uruchom upgrade_terrain.py jako ostatni.

## Animacja powierzchni
Osobny suwak 0–500, odtwarzanie i pauza, restart oraz tempo 5/20/50/100 kroków na sekundę. Położenia cząsteczek są interpolowane liniowo między rzeczywistymi zapisami co 5 kroków. Mapa powierzchni i pełne ślady pozostają stałe; aktualizowane są tylko znaczniki położenia, bez ponownego budowania powierzchni. Podgląd dla jednego obrazu zawiera 600 cząsteczek i 101 zapisów projekcji PCA. Wysokość na powierzchni gęstości/prędkości jest interpolowana na mapie i nie jest czwartą współrzędną latentu. Testy: interpolacja pomiędzy zapisami, dokładne końce osi czasu, zachowanie wszystkich 600 tożsamości i wcześniejsze testy dynamiki.

## Maski narzędzia
Źródło: oryginalne archiwa Cataract-LMM/3_Object_Tracking/annotations/TR_*.zip, annotation_full.json. Dokładne dopasowanie nazwy filmu i numeru klatki do file_names, przy weryfikacji SHA256 oryginalnego zdjęcia. Wielokąty narzędzi z wyłączeniem Cornea i Pupil są skalowane do 480×320 i rasteryzowane; patch 16×16 należy do maski, jeśli obejmuje co najmniej jeden jej piksel. Obrys to brzeg maski względem czterech sąsiadów. Dostępne: wyróżnienie i filtr samych patchy maski w animacji powierzchni, białe znaczniki wybranych patchy w głównej chmurze, podgląd maski/zdjęcia po prawej, mozaika ośmiu kolejnych zdjęć z możliwością wyboru. Nie przypisuje to przyczynowej odpowiedzialności tokenowi. Brak maski nie uruchamia zastępowania jej bbox. Obsługiwane oryginalne poligony; inne formaty wyświetlają błąd. Dane masek/zdjęć odczytywane w RAM, bez wypakowywania na dysk. Wersja z nowym backendem działa na http://127.0.0.1:8776/; poprzedni serwer 8774 nie obsługuje nowego endpointu. Testy: 4 geometrii maski, 1 zestaw UI tożsamości i stanu bez maski, 3 rzeczywiste klatki z różnych filmów; sprawdzono 13 śladów narzędzia obrazu 0 podczas interpolowanej animacji i podgląd mozaiki.

## Wspólne studnie wszystkich 1500 zdjęć
W polu „Zakres studni” wybierz „Wszystkie 1500 zdjęć · 900 000 cząsteczek”. Każdy histogram obejmuje pełne 900 000 rzeczywistych położeń w jednym zapisie, bez próbkowania. Dostępne są trzy płaszczyzny wspólnej PCA oraz 101 zapisów: 0, 5, …, 500. Granice siatki 64×64, osie i skala wysokości są stałe dla danej płaszczyzny przez cały czas.

„W aktualnym kroku” animuje powierzchnię od 0 do 500 i pozwala zmieniać tempo. Między zapisami interpolowane są wysokości powierzchni logarytmicznej gęstości, nie obliczany nowy histogram położeń interpolowanych cząsteczek. „Cały ruch” jest stałą mapą odwiedzin obejmującą 90 900 000 zapisanych położeń (900 000 cząsteczek razy 101 stanów). Gęstość jest wygładzonym histogramem i nie stanowi funkcji błędu ani fizycznego potencjału. Maski pojedynczego zdjęcia nie filtrują globalnej mapy.

Artefakt data/global_density.npz zachowuje pełne zliczenia przed wygładzeniem i pochodzenie danych, w tym sumę manifestu źródłowego i bazę PCA. Kod global_terrain.py sprawdza sumy plików stanów oraz czy żaden punkt nie wypadł poza histogram. Testy test_global_density.py obejmują orientację osi, zachowanie liczności, wykrywanie utraconych punktów, wygładzenie oraz rzeczywisty pełny zbiór. API na porcie 8776 zweryfikowano dla pełnej populacji; wizualizację sprawdzono w przeglądarce z odtwarzaniem i mapą całego ruchu.

## Trajektorie wybranych zdjęć i całego zbioru

Panel „Trajektorie” ma dwa zakresy. „Wybrane zdjęcia” przyjmuje pojedyncze numery i zakresy, np. `12, 40-45`, maksymalnie 24 zdjęcia. Dla każdego z nich odczytywane są wszystkie 600 cząsteczek i 101 rzeczywistych stanów. Renderer pokazuje pełną drogę 0–500 oraz jaśniejszą część przebytą do aktualnego kroku. Binarny zapis zachowuje kolejność zdjęcie–patch–krok i nie próbkuje cząsteczek.

„Wszystkie 1500 zdjęć” rysuje za pomocą WebGL dokładne odcinki wszystkich 900 000 cząsteczek między kolejnymi rzeczywistymi zapisami. Ogon może obejmować 1, 3 albo 10 przedziałów; daje to odpowiednio 900 000, 2 700 000 albo 9 000 000 odcinków. Tryb globalny strumieniuje potrzebne stany i utrzymuje ograniczony cache GPU, zamiast ładować naraz 90,9 miliona położeń pełnej historii. Dostępne jest kolorowanie długością ruchu 768D, numerem zdjęcia i numerem patcha. Suwak, tempo i płaszczyzna PCA są wspólne z pozostałymi widokami.

Kod: `trajectory_data.py`, `trajectory-scope.js`, `trajectory-webgl.js` i `trajectory-inline.js`. Testy kontrolują rozwijanie zakresów zdjęć, limit, tożsamość obrazu i patcha w osi czasu, rozmiar binarny oraz wybór przedziału animacji. W przeglądarce zweryfikowano 1800 pełnych trajektorii trzech zdjęć oraz 2 700 000 odcinków wszystkich 900 000 cząsteczek przy ogonie trzech przedziałów.
