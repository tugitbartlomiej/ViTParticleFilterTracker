# SD K8000: kontrolowane ulepszenia

Zobacz PLAN.md. Wszystkie warianty startują z własnego najlepszego SD z kroku 7500. VAE i ViT są zamrożone. Nowy optymalizator jest zamierzony: najlepszy zapis bazowy nie zawiera stanu optymalizatora. Wejściem jest pełny latent 600 × 768. Filmy, klatki i latenty pozostają wyłącznie w RAM. Eksperyment zakończony 6 października pozostaje niezmieniony.

Uruchomienie: `deploy.py --deploy --smoke`. Po potwierdzeniu `SMOKE_OK` i zakończenia zadania z kodem 0: `deploy.py --submit`. Przydział obejmuje cztery GPU w partycji Hopper. Blokady zapobiegają duplikatom.

Kolejność: diagnostyka 50/100/200 kroków, leading/trailing; potem A_control, B_mixed, C_optim i D_roi, po maksymalnie 4000 nowych kroków każdy. Każdy wariant zapisuje osobne best, latest i metryki. `COMPARISON.json` zestawia wyniki.

Manualny `STOP` dotyczy katalogu porównania. `STOP_TIME` żąda zapisania checkpointa i kodu wyjścia 75. Recovery zachowuje wagi, optymalizator, EMA i RNG, lecz wznawia strumień danych z nowego epoch_offset; nie jest to dokładne odtworzenie kolejności próbek.

Monitoring obsługuje wyłącznie tę nową rodzinę. Nie wznawia zakończonych poprzednich eksperymentów.

Brak etykiety wyłącza ROI loss. Nie utożsamiać cosine z procentem zgodności, confidence z AP, ani IoU generacji względem ramki źródłowej z jakością wobec niezależnej etykiety wygenerowanego obrazu.
