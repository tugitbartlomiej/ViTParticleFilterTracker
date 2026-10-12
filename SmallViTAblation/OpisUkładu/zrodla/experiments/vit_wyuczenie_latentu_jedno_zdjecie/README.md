# Wyuczenie latentu jednego zdjęcia

Test osiągalności zapisanego sztucznego GELU przez zmiany oryginalnych wag ViT. Jeden obraz źródłowy indeks29, TR_0154, klatka1926. Nie jest to ocena generalizacji. Obraz i etykieta GT są wykorzystane do optymalizacji.

Uruchomienie z katalogu SmallViTAblation:

```powershell
& F:/Python/python.exe -B -u experiments/vit_wyuczenie_latentu_jedno_zdjecie/trenuj.py
& F:/Python/python.exe -B -u experiments/vit_wyuczenie_latentu_jedno_zdjecie/sprawdz.py
```

`PLAN.md` zawiera progi i ograniczenia. `WYNIKI.json` deklaruje rodzinę badawczą lokalnie; współdzielony rejestr nie jest zapisywany ze względu na wyłączną własność dwóch katalogów. Wszystkie wyniki są w `outputs/vit_wyuczenie_latentu_jedno_zdjecie`. `PROGRESS.json` jest aktualizowany atomowo i zawiera rzeczywisty PID. `RUN_CONFIG.json`, pełna historia, checkpointy before/best/final z optimizerem, audyty wszystkich wag/BN, latenty600x768, CSV J128 i overlay umożliwiają kontrolę. Oryginalny model_snapshot jest skopiowany bez zmian; nie zawiera mechanizmu podmiany aktywacji w wykonanym forward. Hooki są wyłącznie odczytujące, a niezależna weryfikacja obejmuje forward bez hooków.

Stan końcowy i liczby: `WYNIKI_TRAIN.md` oraz `outputs/.../RAPORT.md`. Kod odmawia nadpisania zakończonego RESULT.json.
