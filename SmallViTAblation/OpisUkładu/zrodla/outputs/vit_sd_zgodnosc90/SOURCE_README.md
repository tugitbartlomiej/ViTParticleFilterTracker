# Uruchomienie

`F:/Python/python.exe experiments/vit_sd_zgodnosc90/run.py --audit-only` odtwarza certyfikat i porównanie zapisanych obrazów. `powershell -NoProfile -File experiments/vit_sd_zgodnosc90/uruchom.ps1` uruchamia ukryty proces z lokalnymi zależnościami SD i generuje dwa nowe obrazy (kontrola i prowadzenie, wspólne nowe ziarno). Wszystkie wyniki trafiają do outputs/vit_sd_zgodnosc90. PROGRESS.json i stdout.log pokazują stan procesu. Próg 90 jest wykluczony; generacja jest próbą najlepszego przybliżenia.
