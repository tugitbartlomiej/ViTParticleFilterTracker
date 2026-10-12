# Wyuczenie latentu do85

Kontynuacja oryginalnego ViT od kroku2000 z Adamem aż agreement128>=85 po niezależnym odczycie. Confidence i IoU są wtórne. Nie zmieniamy targetu, metryki, architektury, czterech uczonych parametrów ani objective. Jeden źródłowy obraz i GT, bez generalizacji.

```powershell
& F:/Python/python.exe -B -u experiments/vit_wyuczenie_latentu_jedno_zdjecie_do85/trenuj.py
```

Trener sam uruchamia końcową niezależną weryfikację sprawdz.py. PROGRESS.json z PID/krokiem jest atomowy i niekrytyczny. Checkpointy best_live/latest i chunk co1000 umożliwiają odzyskanie wag i Adam; lock zapobiega duplikatom. RUN_CONFIG, BASELINE_CHECK, HISTORY, LR_EVENTS, latenty600x768, CSV128, audyty, overlay i RAPORT dokumentują wynik. PLAN ustala twardy limit50000 nowych aktualizacji oraz obsługę plateau. Poprzednie rodziny pozostają nietknięte.
