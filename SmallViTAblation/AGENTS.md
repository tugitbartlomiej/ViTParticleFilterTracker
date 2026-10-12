# Miejsce zapisu eksperymentów ViT

Na wyraźne polecenie użytkownika z2026-10-05 wszystkie nowe pliki dotyczące tego ViT, regulacji latentu i generowania obrazów SD zapisuj pod:

`F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`

Kod i plany eksperymentów: `experiments/`. Wyniki, obrazy, latenty, modele, wykresy, raporty i logi: `outputs/`. Nie używaj katalogu projektu Grant jako docelowego miejsca zapisu tych eksperymentów.

Wyjątek od reguły `experiments/` na wyraźne polecenie użytkownika z 2026-10-06: aplikacja GUI do edycji latentu 600x768 i generowania obrazów z wytrenowanego SD (checkpoint `assets/StableDiffusionModel/best_checkpoint.pt`) jest w `app/` (`app.py`, `silnik.py`, `uruchom.bat`). Jej wyniki nadal zapisuje do `outputs/latent_editor_sd_20261006/`.

Siedem dotychczasowych rodzin zostało skopiowanych z projektu Grant wraz z wynikami. Historyczne JSON-y, migawki kodu i sumy kontrolne pozostawiają oryginalne ścieżki pochodzenia. Manifest kopii: `outputs/przeniesienie_latent_sd_20261005/MANIFEST.json`.

Nie uruchamiaj ponownie zakończonych eksperymentów tylko w celu zmiany ścieżek. Nowe biegi zapisuj oddzielnie, aby zachować istniejące wyniki.
