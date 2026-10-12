# Kontynuacja do 2000 aktualizacji

Wznowienie MODEL_T1_norm2_fc1_final.pt po kroku1000 z Adamem, stały learning rate3e-5. Ten sam obraz, sztuczny target, J128, norm2/fc1, objective i zamrożona reszta. Nie jest to restart ani ocena generalizacji.

```powershell
& F:/Python/python.exe -B -u experiments/vit_wyuczenie_latentu_jedno_zdjecie_2000/trenuj.py
& F:/Python/python.exe -B -u experiments/vit_wyuczenie_latentu_jedno_zdjecie_2000/sprawdz.py
```

PLAN.md deklaruje podłogę i kryterium; WYNIKI.json rejestruje rodzinę lokalnie w dozwolonym zakresie. Wyniki w outputs/vit_wyuczenie_latentu_jedno_zdjecie_2000: PROGRESS.json z PID i krokiem, RUN_CONFIG.json z hashes/stany Adam, HISTORY, checkpointy, latenty i CSV, audyty, overlay. WYNIKI_TRAIN.md i RAPORT.md opisują wynik. Źródłowe artefakty z kroku1000 oraz oryginalny NPZ/checkpoint pozostają bez zmian.
