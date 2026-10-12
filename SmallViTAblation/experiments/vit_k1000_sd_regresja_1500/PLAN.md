# 1500 losowych startów regresji szumu SD w przestrzeni K1000: plan i warunek wstępny

Zlecenie użytkownika, 2026-10-08: „puść na edenie… 1500 różnych startów randomowych. Każdy start ma być potem regresją poprawiany”.

## Zakres

- Zdjęcie celu: 0055/4019 (j = 10389).
- Generator: stare SD uczone na latentach K1000 (`assets/StableDiffusionModel/best_checkpoint.pt`, ładowane przez `app/silnik.py`).
- Miara: ViT K1000 (`assets/MalyViT_K1000_SD_conditioning`, sha `3df42a6a…`), postęp wzdłuż δ = z*_K1000 − z0.
- Metoda: jak `experiments/sd_img2img_sterowanie_latentem/kod/optymalizacja_regresja_starty.py`:
  - 5 kroków regresji;
  - 16 par antytetycznych;
  - σ = 0,05, η0 = 1,0;
  - DDIM 20 kroków.
- Wykonanie na Edenie jako tablica zadań.

## Warunek wstępny: z* K1000 musi dawać sensowną detekcję

Wynik z 2026-10-08 nie spełnił warunku, więc generacji **nie uruchomiono**. Pliki są w `outputs/vit_k1000_sd_regresja_1500/z_cel/`, skrypt to `kod/z_cel_k1000_v2.py`.

| wariant | kroki | IoU | pewność | ‖Δz‖/‖z0‖ | poza (naturalne 1,7) |
|---|---:|---:|---:|---:|---:|
| K8000, etap 8 (odniesienie) | 300 | 1,00 | 0,81 | 2,85 | 19,8 |
| K1000, etap 8, pudło min–max puli (v1, poprzedni agent) | 300 | 0,70 | 0,055 | 2,83 | 19,7 |
| K1000, ta sama procedura, dłużej (v2) | 3000 | 0,70 | 0,055 | 2,84 | 19,8 |
| K1000, diagnostyka: samo pudło z wag | 3000 | 1,00 | 0,93 | 11,1 | 91,8 |

Wnioski:
- **Ta sama procedura nie dochodzi do detekcji.** Wydłużenie z 300 do 3000 kroków nic nie zmienia, bo optimum leży na granicy pudła min–max (11 procent współrzędnych na granicy). Strata dodatniej ogniskowej stoi na 2,58, a ramka powyżej progu 0,5 nie powstaje.
- **Bez naturalnego pudła detekcja się udaje,** ale latent zmienia się o 11 norm i jest 54 razy dalej od naturalnych tokenów niż prawdziwe zdjęcia. To prawie 5 razy dalej niż z* K8000.

Pliki:
- `zb_j10389_minmax_k1000_v2_3000.npy`: wersja zgodna z etapem 8;
- `zb_j10389_wagi_k1000_v2_3000.npy`: diagnostyka, nie jest celem.

## Szacunek kosztu generacji

Pomiar lokalny na RTX 3070: stare SD, 20 kroków DDIM, partia 16, z dekodowaniem VAE, daje 0,88 s na obraz.

Jeden start to 1 + 5 × 33 = 166 obrazów:
- lokalnie około 2,5 min;
- 1500 startów lokalnie około 62 h;
- na H100 szacunkowo 4–6 razy szybciej, czyli 10–15 h karty, w 12–15 zadaniach po około 1 h każde.

Czasu na Edenie jeszcze nie mierzono.

## Decyzja dla użytkownika

Generacja ruszy dopiero, gdy wiadomo, pod jaki cel ma iść. Możliwe cele:
1. z* K1000 z etapu 8: pudło naturalne, detekcja się nie udaje.
2. z* K1000 z samym pudłem wag: detekcja się udaje, ale latent jest skrajnie nienaturalny.
3. Inna definicja celu, na przykład tylko tokeny narzędzia.
