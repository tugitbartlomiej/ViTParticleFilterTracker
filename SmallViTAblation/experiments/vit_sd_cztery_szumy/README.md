# Cztery szumy z prowadzeniem przez ViT

Autoryzowany przebieg w tle: cztery obrazy prowadzone i cztery kontrole z dopasowanymi ziarnami. Wyniki w `outputs/vit_sd_cztery_szumy`. Status bieżący w PROGRESS.json, start i PID w LAUNCH.json. Kod bazuje na lokalnych kopiach generatora SD i modelu z `vit_latent768_sd_od_szumu` oraz ograniczonej korekcie denoisingu z `vit_latent768_sd_od_najblizszego`.

Uruchomienie: `powershell -NoProfile -ExecutionPolicy Bypass -File experiments/vit_sd_cztery_szumy/uruchom.ps1`. Środowisko rfdetr_1_10_1, lokalne deps SD, offline i USE_TF=0. Skrypt odmawia nadpisania istniejącego biegu. Szczegóły porównania i ograniczenia w PLAN.md. Po zakończeniu RAPORT.md, RESULT.json i WYNIKI_SD.md powstają automatycznie. Oryginalny sztuczny cel jest dokładnie nieosiągalny; mierzymy przybliżenia.
