# Eksport ruch60_rp128_50tr_50val: trajektorie 100 zdjęć po 806 checkpointach w rzucie 128D (2026-10-11)

Opis części `ruch60_rp128_50tr_50val/` tego folderu (pobieranej z Edena przez `../pobierz_z_edena.sh ruch60`). Pochodzenie i format:

Eksport `outputs/porownanie_szklo_20261010/eden/ruch60_rp.py` (`ruch60_rp.slurm`, zadanie Slurm 1826867, hopper-2), katalog źródłowy
`~/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/ruch60_rp128_50tr_50val`. Zdjęcia: pierwsze 50 TRAIN i 50 VALID z tego samego
ciągu losowego co pomiar wyżej (ziarno 2026), czyli zdjęcia 0–49 i 3000–3049 pomiaru. Rzut 768 → 128: `torch.randn(768, 128)`
z generatorem o ziarnie 0 podzielone przez √128, identyczny z obserwatorem fali 3 pokręteł (`ANALIZA_*_rp`), więc współrzędne
są wprost porównywalne z oknami gęstymi.

| plik | kształt, typ | zawartość |
|---|---|---|
| `rp128.npy` | (806, 100, 600, 128) float16 | tokeny w rzucie 128D po każdym checkpoincie |
| `pelne768.npy` | (806, 4, 600, 768) float16 | pełny latent dla zdjęć 0, 1 (TRAIN) i 50, 51 (VALID) |
| `srednia768.npy` | (806, 100, 768) float32 | średnia tokenów zdjęcia w 768D (mod zdjęcia) |
| `normy768.npy` | (806, 100, 600, 2) float16 | \|z\| i \|z − z(krok 0)\| w 768D |
| `oceny.npy` | (806, 100, 2) float32 | pewność szczytu mapy i IoU jego ramki z GT |
| `R_rp128.npy` | (768, 128) float32 | macierz rzutu |
| `META.json` | | kroki, zdjęcia (`frame_id`, podział, wiersz), konfiguracja, sha checkpointu końcowego |

Latent = wyjście GELU w MLP ostatniego bloku MalyViT (dim 192, 6 bloków), 600 tokenów × 768, forward w trybie eval z bf16
autocast (jak obserwator gęsty i pomiar aplikacji). Wczytywanie bez ładowania całości: `np.load(..., mmap_mode='r')`.
