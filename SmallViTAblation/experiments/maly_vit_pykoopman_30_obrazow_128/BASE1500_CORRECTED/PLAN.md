# Poprawka doboru: 30/30 fit z dokładnej bazy1500

Zapis przed doborem/ekstrakcją: 2026-10-02T14:10:43.463262+02:00.

W dawnym zbiorze25/30fit należało do dokładnej bazy1500. Wymieniamy pięć dawnychfit indeksów0,2,3,5,6 na pięć nowych rzeczywistych train z tej bazy i prawdziwymi maskami, deterministycznie seed202610021, różne filmy, poza wszystkimi filmami starego34 i holdoutami. Kolejność34slotów/fit0..7+12..33/holdout8..11 niezmienna. Holdout membership osobno: tylko8 należy do exactbase,9/10/11 poza; nie zmieniamyholdoutów.

Ścieżki starego zbioru/modeli pozostają nietknięte. Osobne OBRAZY.npy/MASKI.npz/MANIFEST i LATENT_NOWE5_768.npy(101,5,600,768), float32. Reszta pełnych trajektorii mapowana do starego LATENT_768.npy; bez duplikowania6GB. Manifest zawiera cache_sources perimage. Wszystkie nowe obrazymusząpixelmatch cache1500 resize, maski/coverage600 spójne. Zapis hashów plików/provenance i porównań.

GPU source SmallViT, strict checkpoint0..1000co10, eval/inference FP32 RGB/255 bez TF32/autocast, batch4, frozenBN. Wszystkie finite i pełne hash. Baseline checkpoint0/500/1000: powtórny forward5 w batch1 porównanie do batch4, tolerancja1e-5. Bez nowego treningu ViT/pobierania.

Aktualny plan modelu NN:30fit z exactbase1500,fit0..900 daje90przejść/image; przyszłe900..1000 to10przejść do oceny. Dobór/walidacja/refit/protokół NN prowadzi osobnyagent; ta poprawka dostarcza dane i nie zmienia sama jego modelu. Czteryholdouty zachowane, wcześniej widziane exploratory.
