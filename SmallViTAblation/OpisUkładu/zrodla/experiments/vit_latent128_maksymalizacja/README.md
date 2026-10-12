# Maksymalizacja detekcji przez interwencję J128

Checkpoint1000, zamrożone wagi. Adam optymalizuje surowe600×128 przyrostów osobno dla obrazu. Zakres dokładnego GELU zachowany, bez wcześniejszego ograniczenia pooledSD. Szczegóły w PLAN.md.

Uruchomienie na lokalnej RTX:

```powershell
F:/Python/python.exe experiments/vit_latent128_maksymalizacja/test_core.py
F:/Python/python.exe experiments/vit_latent128_maksymalizacja/maximize.py --stage pilot
F:/Python/python.exe experiments/vit_latent128_maksymalizacja/maximize.py --stage full
```

`run.py` jest zachowaną kopią narzędzi pierwszego eksperymentu; tutaj importujemy wyłącznie funkcje pomocnicze. Punktem wejścia nowego eksperymentu jest `maximize.py`. Żadne parametry ViT nie są optymalizowane.

Wyniki: `outputs/vit_latent128_maksymalizacja`. `SELECTION.json`: wybór learning rate na12 obrazach. `RESULT.json`: pełne metryki, krzywe i pochodzenie. `TARGETS_FULL400.npz`: bazowe i docelowe128 współrzędnych wszystkich600 cząsteczek każdego z400 obrazów. `DETECTIONS_FULL400.npz`: pełne mapy głowy przed i po oraz kontrola losowa. Odtworzenie pełnego768D wymaga bazowego forward i podmiany tylko wskazanych kanałów.

Wszystkie400 etykiet używane do optymalizacji odpowiednich obrazów. To celowa analiza oracle zadanego zbioru, a nie test generalizacji czy trening wag. Dopuszczalny zakres funkcji aktywacji nie gwarantuje naturalnego obrazu realizującego znaleziony stan.
