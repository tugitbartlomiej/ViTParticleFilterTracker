# Cykl filmów: cząstki ViT, tożsamość, Koopman i Langevin

Data: 2026-10-01. Zadanie: szczegółowo pokazać pipeline w animacjach podobnych do wcześniejszych filmów projektu Grant.

## Zawartość

Dwanaście rozdziałów, 51 ujęć, 18 minut i 12 sekund. Film zbiorczy, osobne rozdziały, polski lektor, napisy SRT i WebVTT, rozdziały w MP4, indeks HTML, pełny scenariusz i kod do ponownego renderowania.

1. Od pełnego obrazu do 600 tokenów.
2. Kod pozycji, uwaga i dwa rodzaje czasu.
3. Maska od jednego piksela i kontrola progu 25 procent.
4. Geometria latentu, rzut i podprzestrzeń.
5. Dawny odczyt LDA i różnica między informacją a przyczynowością.
6. Osobne trajektorie tokenów i ograniczenia dostępnych checkpointów.
7. SVD kierunków ruchu, podział po filmach i rzeczywiste wyniki.
8. Sto pomiarów kontra dwadzieścia średnich.
9. Dopasowanie Koopmana i rzeczywista prognoza jednej sondy bez uśredniania.
10. Wariant Langevina, pamięć, reszta i wybór modelu.
11. Wpływ paczki i nowsze równanie latentu, wyjaśnienie kontra prognoza.
12. FastGAN, dobór prawdziwych klatek, różnorodność i test poprawy treningu.

## Silnik i głos

Wcześniejsze animacje wskazane przez użytkownika używały **ManimGL 1.7.2**, nie ManimML. Użyto tej samej biblioteki z istniejącego środowiska:

`F:\Studia\MCHTR\Grant\Projekt\Research\Prezentacja\Inne\manim\Animation_codes\.venv-manimgl\Lib\site-packages`

Renderer uruchamia Python istniejącego środowiska RF-DETR 3.11, a wrapper dopisuje powyższą bibliotekę. Nie instalowano ani nie zmieniano pakietów. Import izolowanego dawnego środowiska działał bardzo wolno; wrapper wykorzystuje już dostępne zależności. Biblioteka renderująca pozostaje ManimGL.

OmniVoice nie odpowiadał na lokalnym porcie 8080. Narracja została wygenerowana lokalnym głosem **Microsoft Paulina Desktop, pl-PL**, do plików WAV. Nie uruchamiano odtwarzania przez głośniki ani nie zmieniano ustawień globalnej mowy użytkownika.

## Źródła i uczciwość wizualizacji

`ZRODLA.json` zawiera ścieżki i sumy kontrolne odczytanych plików. Treść scenariusza znajduje się w `SCENARIUSZ.md`, dane prezentacji w `assets/dane.json`.

Rzeczywiste materiały: pełne klatki Cataract-LMM, maski, macierze tokenów 2 × 12 × 600 × 768, osie SVD, wyniki pomiaru 60→120, historia skalarnego odczytu LDA seed47 bez dokładania klatek. Zdjęcia w animacji są wejściami do analizy; rysowane siatki są nakładką ilustracyjną, nie wejściem sieci.

Schematy uwagi, rozkłady ilustrujące LDA, strzałki sił i przepływy blokowe nie są wynikami pomiarów. Mają oznaczenie „SCHEMAT METODY”. Ruch pomiędzy dwoma zmierzonymi stanami tokenów jest wyłącznie interpolacją do wizualizacji; nie twierdzimy, że jest trajektorią między wszystkimi krokami.

Nowa gęsta kampania opisana w `maly_vit_ruch_czastek_gesta_historia` jest pokazana jako plan pomiaru. Nie sprawdzano w tej operacji stanu zadania na Edenie i nie podano fikcyjnych wyników tej kampanii.

### Dodatkowo obliczony przykład prognozy bez uśredniania

Scena w rozdziale 9 dopasowuje osobno do pierwszej sondy modele przyrostów z czterema opóźnieniami. Wejście: 100 nieuśrednionych odczytów w krokach 10..1000. Horyzont: kroki 1010..1500, bez podglądania przyszłych wartości. Kierunek LDA był już ustalony na końcu rozgrzewki. Model jest regresją z karą diagonalną 0,01, zgodną z konwencją starego kodu.

RMSE po 50 odczytach: Koopman 1,1276; Langevin 2,0911; brak zmiany 1,3193. To wynik jednej sondy i jednego przebiegu, bez strojenia na tym przyszłym odcinku. Nie jest porównaniem populacyjnym, prognozą 600 tokenów ani testem poprawy detektora. Współczynniki i pełne predykcje są zapisane w `assets/dane.json`.

## Odtwarzanie

Po zakończeniu montażu otwórz `index.html`. Film zbiorczy znajduje się w `videos/CALOSC_ViT_Tozsamosc_Koopman_Langevin.mp4`, a rozdziały w `videos/Rozdzial01.mp4` ... `Rozdzial12.mp4`.

## Reprodukcja

Pracuj w katalogu tej dokumentacji. Korzystaj z interpretera:

`F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe`

Kolejność: `scenariusz.py`, `przygotuj.py`, `narracja.ps1`, następnie `render.py sceny.py Rozdzial01 ... Rozdzial12 -w --hd` i `zloz_i_sprawdz.py`.

`narracja.ps1` zapisuje tekst obok każdego WAV i ponawia syntezę tylko po zmianie tekstu. Nie przekazuj `--fps` do tej wersji ManimGL: parser pozostawia wartość tekstową. Liczba 24 klatek na sekundę jest ustawiona w `custom_config.yml`.

`VIT_FILM_CHECK=1` skraca ujęcia i wyłącza głos, służy wyłącznie kontroli renderera. `VIT_FILM_SMOKE=1` renderuje tylko pierwsze ujęcie rozdziału. W finalnych filmach te zmienne są nieustawione.

## Weryfikacja

Próbny render obejmuje wszystkie 51 ujęć. Finalny skrypt montażu sprawdza liczbę scen, długość osi czasu, obecność głosu, Full HD i dekoduje całe strumienie filmów. Zapisuje `MANIFEST_FILMOW.json` i plansze kontrolne w `kontrola/`.

Nie zmieniono kodu modelu ani wcześniejszych eksperymentów. Nie uruchamiano treningu, nowych zadań klastra, instalacji ani commitów.
