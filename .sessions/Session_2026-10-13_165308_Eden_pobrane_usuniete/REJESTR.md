# Rejestr: pobrane z Edena / usunięte na Edenie

Eden: `eden-cluster`, katalog domowy `/home2/faculty/bpiotrowski` (dalej `~`). Lokalnie: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker` (dalej `L:`).
Wspólny rejestr na Edenie (dopisują wszystkie sesje): `~/smallvit_runs/USUNIETE_Z_EDENU.txt`.

## Zasady
- Na Edenie kasuję katalog tylko wtedy, gdy wszystkie jego pliki mają lokalną kopię z tą samą sumą (MD5 albo SHA-256), policzoną na nowo
  po obu stronach tuż przed usunięciem (porównanie list przy `LC_ALL=C sort`, bo Eden i Git Bash sortują inaczej).
- Zostają na Edenie: zbiory danych do biegów (`smallvit_data/imagenette`, `smallvit_data/cataract_lmm`), kod, logi,
  `2026-10-13_deit_cifar10` (prowadzi sesja koloidy_CIFAR, katalog jeszcze rósł 61 → 64 GB).
- Lokalny dysk F: trzymam powyżej ~10 GB wolnego.

## Stan dysku Edena (`/mnt/evafs`, kwota 200 GB)
| Kiedy | Zajęte | Wolne |
|---|---|---|
| 2026-10-13 ~16:00 (przed porządkami) | 192 GB (96 %) | 8,1 GB |
| 2026-10-13 ~16:45 (po porządkach innej sesji) | 169 GB (85 %) | 32 GB |
| 2026-10-13 17:12 (po usunięciu `lr_5e-4_rp`) | 165 GB (83 %) | 36 GB |

## Pobrane z Edena
| Data | Z Edena | Lokalnie | Plików | Rozmiar | Kontrola |
|---|---|---|---|---|---|
| 2026-10-13 | `~/smallvit_data/imagenette/imagenette2-320/val` | `L:/SmallViTAblation/assets/Dane_Imagenette/imagenette2-320/val` | 3925 | ~100 MB | liczba plików; **na Edenie zostaje** (zbiór do biegów) |
| 2026-10-13 17:05 | `~/smallvit_runs/2026-10-10_szklo_pokretla/wyniki/lr_5e-4_rp` | `L:/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/lr_5e-4_rp` | 21 | 4,3 GB | MD5 21/21 (`MD5_lr_5e-4_rp_EDEN.txt` = `MD5_lr_5e-4_rp_LOKALNE.txt`) |

## Usunięte na Edenie
| Data | Kto | Ścieżka na Edenie (`~/smallvit_runs/…`) | Plików | Rozmiar | Lokalna kopia | Kontrola |
|---|---|---|---|---|---|---|
| 2026-10-13 ≤16:20 | inna sesja (bez wpisu w rejestrze Edena) | `2026-10-09_maly_vit_od_zera_35000_21epok/wyniki` (cały: `pelny/wagi` 806 ckpt 9,5 GB, `gesty`, `gesty_proba`) | ~1200 | ~13,6 GB | `pelny/wagi` → `L:/SmallViTAblation/assets/MalyViT_35000_od_zera_s47/wagi` (806, SHA-256 z pobrania 2026-10-09 zgodne); `gesty` → prawdopodobnie `assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis` (nie sprawdzałem) | nie moja; ponowne sumy niemożliwe, bo pliki już skasowane |
| 2026-10-13 16:29 | inna sesja | `2026-10-10_szklo_koloidalne/wyniki/wiazki_koloidy` | 4 | 3,5 GB | `L:/SmallViTAblation/outputs/porownanie_szklo_20261010/eden/wyniki/wiazki_koloidy` | MD5 zgodne (wg wpisu) |
| 2026-10-13 16:37 | inna sesja | `2026-10-10_szklo_pokretla/wyniki/dlugi_lr_5e-4_rp` | 26 | 7,7 GB | `L:/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/dlugi_lr_5e-4_rp` | MD5 zgodne (wg wpisu) |
| 2026-10-13 17:11 | ta sesja | `2026-10-10_szklo_pokretla/wyniki/lr_5e-4_rp` | 21 | 4,3 GB | `L:/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/lr_5e-4_rp` | MD5 21/21, oba spisy w tym folderze |

## Zostało na Edenie bez kopii lokalnej (duże)
| Ścieżka (`~/smallvit_runs/2026-10-10_szklo_pokretla/wyniki/…`) | Rozmiar |
|---|---|
| `klas_vit_od_zera_60ep_lr_5e-4` (ViT na Imagenette, 60 epok: `gesty` 12,1 GB + `wagi` 3,8 GB) | 15,2 GB |
| `bezpoz_lr_5e-4_rp`, `cnn_lr_5e-4_rp_zostaw`, `lr_2e-3_rp_zostaw`, `lr_5e-4_rp_z48_zostaw`, `lr_5e-4_rp_z49_zostaw` | 5 × 4,3 GB |
| `2026-10-13_deit_cifar10` (cały; nie ruszam) | 64 GB |
| `smallvit_data/cataract_lmm` (zbiór danych) | 48 GB |

Lokalnie 2026-10-13 17:12: F: 17 → ~13 GB wolne, E: 19 GB wolne — na resztę (~37 GB) nie ma miejsca.
