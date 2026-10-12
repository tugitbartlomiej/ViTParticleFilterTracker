# Planowanie kolejności paczek SD

Kontroler horyzontu1, ponownie liczona siła czterech paczek przed każdym krokiem. CEL: MSE J128 oryginalnego sztucznego after na identycznej sondzie29. Oryginalny k1000/AdamW; bez CAV i podmiany latentu. To oddzielny eksperyment od niezależnych powtórzeń czterech paczek.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
F:/Python/python.exe -u experiments/vit_langevin_planowanie_paczek_sd/worker.py --smoke
F:/Python/python.exe -u experiments/vit_langevin_planowanie_paczek_sd/worker.py --steps 500
```

Budżet najpierw100/politykę, potem do500 z zachowaniem pamięci. Polityki controlled/random_choice/balanced_cycle mają te same aktualizacje i te same cztery paczki. Wyniki: outputs/vit_langevin_planowanie_paczek_sd. PROGRESS/PID/JOB/logi, RESULT_100 oraz końcowy RESULT, REPORT, CONTROL_COMPARISON, naturalny BEST_RELOAD_VERIFICATION. W każdej polityce SCHEDULE, decyzje sealed, ACTUAL, TRAJECTORY i BEST_SEQUENCE. Najlepsze modele w checkpoints/BEST_*.pt, z optymalizatorem. Brak globalnej optymalności sekwencji, brak wymyślonych epok i czasów osiągnięcia nieprzekroczonych progów.

Zakończono i zweryfikowano500 kroków każdej z3 polityk. Kontroler nie wygrał w najlepszym stanie J128; losowy prefix450 najlepszy. Progi5/10/25% redukcji oraz85% zgodności wartości nieosiągnięte. Końcowe REPORT, FINAL_AUDIT, BEST_RELOAD_VERIFICATION i BEST_SEQUENCE są gotowe. Bez CAV.
