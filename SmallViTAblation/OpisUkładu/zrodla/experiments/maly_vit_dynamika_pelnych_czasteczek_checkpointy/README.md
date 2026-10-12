# Dynamika pełnych cząsteczek z istniejącego biegu

Status 2026-10-01: pilot zakończony. Pełna historia jest pobrana i sprawdzona; nie wykonano nowego treningu ViT.

PCA r=8 z przeszłości i ridge AR(1) zmniejszyły średni błąd pełnych wektorów o około 3,6–5,1 procent względem persystencji. Pełny ridge 768D i wariant z jednym opóźnieniem nie dały lepszego wyniku niż PCA AR(1). To opisowy wynik jednego przebiegu, bez potwierdzenia semantycznej tożsamości narzędzia.

- [Wynik pilota, liczby i wykres](WYNIKI_PILOT.md).
- [Pełne metryki poszczególnych cząsteczek](METRYKI_PILOT.json).
- [Zgodność lokalnych forwardów z Edenem](AUDYT_ZGODNOSCI.json).
- [Przegląd matematyczny i kodu](AUDYT_METODY.md).

- [Plan, podłogi i kryteria obalenia](PLAN.md).
- [Konfiguracja przed oceną](CONFIG.json).
- [Deklaracja rodziny](WYNIKI.json).

Dane pozostają w `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny`. Własne skrypty i małe wyniki są w tym katalogu. Pełne tokeny czytamy przez mmap; nie utworzono kolejnej dużej kopii. Uruchomienie: `pilot_zgodnosci.py`, następnie `prognoza.py`, następnie `raport.py`, środowiskiem `F:/Python/3.13/python.exe`. Dane kończą się na kroku 1000.
