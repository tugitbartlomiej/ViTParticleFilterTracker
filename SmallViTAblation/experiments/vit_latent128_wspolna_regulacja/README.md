# Wspólna regulacja J128

Zlecone 2026-10-05: program wyszukujący wspólne ustawienia wybranych kanałów zamrożonego ViT na RTX3070. [Protokół](PLAN.md) określa podziały, budżet, konfiguracje i wybór przed testem.

Uruchomienie z korzenia projektu:

```powershell
F:/Python/python.exe experiments/vit_latent128_wspolna_regulacja/szukaj.py --phase preflight
F:/Python/python.exe -u experiments/vit_latent128_wspolna_regulacja/szukaj.py --phase sweep
```

Uruchomienie niezależne od terminala, w tle bez okna:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/vit_latent128_wspolna_regulacja/uruchom_w_tle.ps1
```

Launcher zapisuje PID i czas startu w `PROCESS.json`, wyjście w `RUN.log`, a komunikaty diagnostyczne w `RUN.err`. Odmawia ponownego startu istniejącej kampanii, aby nie nadpisać jej wyników.

Dane są współdzielone tylko do odczytu z wejściami pierwszego eksperymentu J128. Nowe wyniki trafiają wyłącznie do `outputs/vit_latent128_wspolna_regulacja/`. `PROGRESS.json` i logi pokazują stan; `SELECTION.json` zamraża wybór przed testem; `RESULT.json` i `RAPORT.md` zawierają wynik. Skrypt zapisuje kod wykonany oraz hashe.

Nie wstawia tego samego obrazu ani wektora do wszystkich przykładów. Uczy wspólne parametry regulacji ich własnych reprezentacji. Adapter warunkowy jest osobnym ramieniem i raportowany oddzielnie od stałych skal i przesunięć.
