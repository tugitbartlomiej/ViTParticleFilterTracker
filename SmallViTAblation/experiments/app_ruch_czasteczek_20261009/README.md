# Zakładka ruchu cząsteczek w aplikacji

Integracja wizualizacji całego zbioru z aplikacją Qt w `app/`, wykonana 2026-10-09.
Szczegółowa instrukcja znajduje się w `app/README.md`, w sekcji „Ruch zbioru”.

## Sprawdzenie działania

- 38 testów silnika: formaty danych, walidacja, integralność zapisów, zamrożone wagi,
  rzeczywiste krótkie obliczenie, przerwanie oraz kontrola zasobów.
- 10 testów GUI: pauza w interpolowanym stanie, odtwarzanie bez interpolacji,
  tożsamość obrazu i patcha, zachowanie konfiguracji, komunikat o innym checkpointcie
  oraz ochrona nowego zadania przed spóźnionym sygnałem zakończenia poprzedniego.
  Naturalne logi mają osobną serię; można wczytać ukończone stany częściowego pomiaru,
  a trwający pomiar uruchomiony przez własny helper nie jest dublowany.
- Testy `test_ruch_naturalny.py`: chronologia i pochodzenie wag, naturalny forward,
  brak zmian parametrów i buforów, ograniczenie pamięci, pełne publikowanie stanów,
  anulowanie, logi z rzeczywistymi krokami i usuwanie tymczasowego baseline.
  Osobna rzeczywista próba CPU porównuje wynik z niezależnym forwardem checkpointu.
- `smoke_gui.py`: pełne okno istniejącej aplikacji, rzeczywisty OpenGL na RTX 3070,
  wczytanie K1000, 900000 punktów z 1500 obrazów, odtwarzanie do kroku 500,
  przełączenie na jeden obraz i obliczenie dwóch nowych kroków dla jednego obrazu.
  Test sprawdza brak zmian wag oraz świeże odczytanie zapisanych wyników.
- `smoke_natural_gui.py`: import 101 checkpointów do K1000, 20 zapisów straty i 11 ocen
  mAP50; naturalny pomiar czterech obrazów w krokach 0 i 10, następnie odczyt i rysowanie
  w prawdziwym oknie aplikacji. Argument `--timeline` sprawdza gotowy pełny pomiar.
- `audyt_naturalnej_historii.py`: SHA256 wszystkich 101 wag względem zweryfikowanej
  kopii z Edena, zgodność końcowego modelu z K1000 w assets oraz tożsamość kolejności
  wszystkich 1500 obrazów względem treningowego COCO.
- `pomiar_naturalny.py`: pełny, niezależny od GUI pomiar 1500 obrazów przez 101 zapisanych
  modeli; raport postępu i zweryfikowany wynik w `NATURALNY_WYNIK_1500.json`.

Raport i zrzuty okna są w `outputs/app_ruch_czasteczek_20261009/`.
`SMOKE_GUI.json` zawiera faktycznie użyty renderer, SHA256 modelu, liczbę punktów,
wynik obliczenia i ścieżkę nowego pomiaru. Sprawdzone środowisko GUI to Python
`F:/Python/3.13/python.exe`, używany również przez istniejący launcher aplikacji.
Próba GUI wyłącza automatyczne uruchamianie dawnych obliczeń i zapis ustawień użytkownika.

## Zakres i ograniczenia

- Nie uruchomiono automatycznie nowej pełnej optymalizacji 1500 obrazów ani zadania na Edenie.
- Gotowe trajektorie zachowują pochodzenie K1000. Wybór innego checkpointu wymaga nowych obliczeń.
- Nowy silnik niezależnie koryguje pełne 600×768 aktywacji każdego obrazu i utrzymuje
  osiągnięty cel do końca wspólnej osi kroków. Historyczny zapis miał własną procedurę
  optymalizacji; nowa próba nie jest deklarowana jako jej identyczne powtórzenie.
- Do oglądania zapisywane są rzuty we wspólnej bazie PCA, prawdziwe normy ruchu 768D,
  metryki i provenance. Nie zapisuje się pełnych trajektorii 768D ani obrazów.
  Taki zapis nie zastępuje danych potrzebnych do analizy DMD/Koopmana w pełnej przestrzeni.
- Interpolacja dotyczy wyłącznie obrazu animacji; metryki dotyczą rzeczywistych stanów.
- Przerwany pomiar można oglądać, lecz nie można wznowić optymalizacji z samego rzutu 3D.
- Cel jest optymalizowany z etykietą obrazu. Poprawa tego pomiaru nie dowodzi generalizacji.
