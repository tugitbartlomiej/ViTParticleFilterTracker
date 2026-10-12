# Cztery paczki SD a sztuczny cel sondy29

Samodzielna rodzina badawcza, kontynuacja oryginalnego AdamW k1000 z oryginalnymi wagami SD. Zobacz PLAN.md. Dokładne cztery paczki32 z ukończonego częściowego retrievalu, losowe32 i nochange. Prognoza początkowej siły, potem niezależna walidacja100 aktualizacji każdego z pięciu ramion.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
F:/Python/python.exe -u experiments/vit_langevin_cztery_paczki_sd/worker.py --smoke
F:/Python/python.exe -u experiments/vit_langevin_cztery_paczki_sd/worker.py
```

Stan/PID/logi: outputs/vit_langevin_cztery_paczki_sd. Pierwsze liczby INITIAL_FORCE_SUMMARY.json; prognoza FORECAST.json/CSV i FORECAST_SEAL przed treningiem. ACTUAL_*.json, TRAJECTORY_*.npy, checkpointy k1100 z optymalizatorem, FORECAST_VS_ACTUAL.csv, wykres DISTANCE_FORECAST_ACTUAL.png i końcowy REPORT.md/RESULT.json. Brak nadpisania ukończonego wyniku i brak automatycznego wznowienia po błędzie. Nie interpretować100 powtórzeń paczki32 jako epok pełnego train.

Status: complete i niezależnie verified.500 aktualizacji zakończono; brak progów redukcji5 procent i wyższych. Źródło to wyłącznie oryginalny SD checkpoint. Końcowe wyniki w REPORT.md i FINAL_VERIFICATION.json.
