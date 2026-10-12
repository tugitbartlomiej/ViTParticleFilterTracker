# Przestrzenne ustawianie J128

[Plan](PLAN.md): poszukiwanie wspólnej funkcji poprawiającej detekcję narzędzia przy zamrożonym ViT k1000. Cel AP50>0,90; 12 konfiguracji, augmentacje, destylacja training-only. Wynik końcowy na historycznie oglądanym benchmarku ma charakter eksploracyjny.

```powershell
F:/Python/python.exe experiments/vit_latent128_regulator_przestrzenny/szukaj.py --phase preflight
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/vit_latent128_regulator_przestrzenny/uruchom_w_tle.ps1
```

`outputs/vit_latent128_regulator_przestrzenny/PROGRESS.json` pokazuje postęp. `RUN.log` i `RUN.err` to logi, `PROCESS.json` identyfikuje proces. `model_selected.pt` przechowuje wspólne parametry; `RAPORT.md` i `RESULT.json` wynik. Inferencja własnego obrazu używa `inferencja.py`, bez etykiety i bez optymalizacji pod ten obraz.
