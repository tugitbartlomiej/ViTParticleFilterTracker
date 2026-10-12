# Uruchomienie

`powershell -NoProfile -File experiments/vit_sd_szum_nowy_checkpoint/uruchom.ps1`

Proces ukryty czeka na READY_FOR_SD.json poprzedniego treningu, niezależnie sprawdza checkpoint i dopiero potem generuje jeden obraz z szumu. PROGRESS.json rozróżnia oczekiwanie, reload, ładowanie SD i denoising. OBRAZ_SD.png jest finalnym wynikiem 80 kroków; podglądy nie są dodatkowymi próbami. LATENT_VIT_OBRAZU_SD.npz zawiera pełny naturalny latent600x768 i niezmieniony sztuczny cel.
