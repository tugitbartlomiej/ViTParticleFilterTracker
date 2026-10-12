# Najbliższe zdjęcie w całym lokalnym zbiorzeTracking

Przeszukano461547 zdjęć ze170ZIP, bez subsamplingu. Podziały: {'train': 310257, 'val': 64167, 'test': 82370, 'unknown': 4753}.

Najlepszy obraz: TR_0064_S1_P03/TR_0064_S1_P03/TR_0064_S1_P03_001181.jpg (train). MSE768=14.211437; MSE128=84.913193; MSE640=0.071086; cosinus=0.165464. Pewność=89.74%.

Poprawa MSE768 względem minimum400=0.116%. Najlepszytrain: TR_0064_S1_P03/TR_0064_S1_P03/TR_0064_S1_P03_001181.jpg, MSE768=14.211437.

Wynik jest najbliższy według MSE; nie gwarantuje bliskiej rekonstrukcji sztucznego celu. Pełny rankingCSV zachowuje wynik każdego obrazu. Nie jest to pomiar mAP ani jakości na niewidzianych danych.


## Osiągalność sztucznego celu

Dodatkowa diagnostyka wag LayerNorm192 i FC1 wykazała konieczną granicę naturalnej aktywacji poGELU. Największa górna granica kanału, z zapasem0.01, wynosi10.246837, a sztuczny cel ma wartości do51.353161. Granice przekracza15658 współrzędnych w85 kanałach. Dokładne odtworzenie tego celu przez jakikolwiek obraz przy tym checkpointcie jest wykluczone. Konieczna dolna granica MSE768 wynosi6.260274.

To wyjaśnia, dlaczego większa pula i generowanieSD nie dały wiernej rekonstrukcji. Najlepsze zdjęcie jest minimum według ustalonego MSE, a nie realizacją sztucznego stanu. Kontrola pikseli poprzedniego zwycięzcy400 względem źródłaZIP potwierdziła identyczność. WalidacjaCSV potwierdziła461547 rekordów i pełny latent600×768.
