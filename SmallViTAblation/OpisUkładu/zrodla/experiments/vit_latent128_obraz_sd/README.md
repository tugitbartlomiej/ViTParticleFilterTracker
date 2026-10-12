# Obraz SD z zoptymalizowanego latentu128

Jedna próbka ze Stable Diffusion dostrojonego do Cataract-LMM, prowadzona celem z `vit_latent128_maksymalizacja/TARGETS_FULL400.npz`.

```powershell
$env:PYTHONPATH='F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
$env:PYTHONUTF8='1'
& F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe -u experiments/vit_latent128_obraz_sd/generuj.py
```

Wyniki: `outputs/vit_latent128_obraz_sd/OBRAZ_SD.png` oraz `RESULT.json`.
Model i pomocniczy kod wczytywania SD/masek skopiowano z istniejących eksperymentów; oryginałów nie zmieniono.
Szczegóły celu, granice interpretacji i kontrola: `PLAN.md`.
