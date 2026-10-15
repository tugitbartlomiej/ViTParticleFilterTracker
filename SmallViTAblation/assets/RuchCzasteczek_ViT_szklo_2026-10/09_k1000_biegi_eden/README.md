# 09 — wyniki produkcyjne biegów K1000 pobrane z Edena

Pobrane 2026-10-13, przy zwalnianiu kwoty projektu 1299 na Edenie (było ~191 z 200 GB).
Struktura katalogów jest taka sama jak na Edenie pod `~/smallvit_runs/`.
Każdy plik sprawdzono sumą SHA-256 (Eden kontra kopia lokalna) i dopiero potem usunięto go z Edena.

| katalog | pliki | rozmiar | sumy |
|---|---|---|---|
| `vit_k1000_1500_pelne_latenty_90_20261008/results/production` | 3024 | 11 GB | `SHA256SUMS_EDEN_pelne_latenty_90_production.txt` |
| `vit_k1000_replay900k_co5_20261009/results/production` | 3088 | 22 GB | `SHA256SUMS_EDEN_replay900k_production.txt` |
| `2026-10-08_k1000_podmiana150_sd/wyniki/ziarna` | 99 | 2,2 GB | `SHA256SUMS_EDEN_podmiana150_ziarna.txt` |

Kod, logi, provenance i pozostałe podkatalogi `results/` tych biegów zostały na Edenie.
W katalogu głównym każdego biegu na Edenie leży `USUNIETE_Z_EDENU_2026-10-13.txt` z listą usuniętych plików.
Raporty i podsumowania tych biegów są w `outputs/vit_k1000_1500_pelne_latenty_90_20261008/` i `outputs/vit_k1000_replay900k_co5_20261009/`.

Sprawdzenie kopii (Git Bash, z tego katalogu):

```bash
sha256sum -c SHA256SUMS_EDEN_pelne_latenty_90_production.txt --quiet
sha256sum -c SHA256SUMS_EDEN_replay900k_production.txt --quiet
```
