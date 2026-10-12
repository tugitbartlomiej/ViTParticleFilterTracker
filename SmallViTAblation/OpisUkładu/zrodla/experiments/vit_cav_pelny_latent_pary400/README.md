# Pełny latent CAV i TCAV

Badawczy eksperyment pełnych 600x768 aktywacji z 400 oryginałów i 400 LaMa. Szczegóły metody i ograniczeń: PLAN.md. Stan i PID: outputs/vit_cav_pelny_latent_pary400/PROGRESS.json. Końcowe liczby: METRICS.json i WYNIKI_TCAV.md.

Zakończono 2026-10-05: PID 15836, CUDA, 108.51 sekund obliczeń. Wyodrębniono wszystkie 800 obrazów. Test BA 0.758824, AUROC 0.858408; TCAV dodatni na 7/85 oryginałów. 30 bootstrap, 30 film-permutation, 30 full-random. Finite differences i niezależny audit.py przeszły. Pełny CAV obejmuje wszystkie 600 pozycji i 768 kanałów. Wagi modelu, BN, checkpoint, PDF i źródłowe obrazy niezmienione.

Osobny wykonawca został utworzony na wyraźną prośbę użytkownika, lecz jego narzędzia lokalne nie odpowiadały. Czat nadrzędny przejął fizyczną implementację i wykonanie; wykonawca zwrócił audyt metodologiczny bez zapisów. Nie działa drugi job tej rodziny.

Audyt artefaktów: `F:/Python/python.exe -B experiments/vit_cav_pelny_latent_pary400/audit.py`. Dodatkowe źródłowe sumy kontrolne, film-bootstrap AUROC i ograniczenia empirycznego null: outputs/vit_cav_pelny_latent_pary400/AUDIT.json. Kontrole nie potwierdzają istotności po korekcji dwóch porównań. Znak pochodnej detektora jest w większości ujemny mimo jakości rozdzielenia LaMa.

Uruchom z F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation:

```powershell
& 'F:/Python/python.exe' -B experiments/vit_cav_pelny_latent_pary400/run.py
```

Background: launch.ps1 zapisuje launcher PID oraz stdout/stderr we własnej rodzinie. run.py przy ponownym uruchomieniu odbudowuje cache, więc nie uruchamiaj jednocześnie drugiej kopii. Cache CPU ~1.5 GB; mały batch GPU, bounded CPU4. Nie ma treningu ViT.

Artefakty: LATENT_FULL.npy, CACHE_MANIFEST.json, SPLITS.json, pełna jednostkowa CAV_FULL_600x768.npy, COEF_FLAT.npy, TRAIN_MEAN.npy, CAV_METADATA.json, CAV_BANK_FULL.npy, GRADIENT_FULL.npy, TCAV_SENSITIVITIES.npy, PROBE_METRICS.json, TCAV_METRICS.json, FINITE_DIFFERENCE.json, VERIFICATION.json i mapy energii wszystkich współrzędnych.
