# dyfuzja_sd_filmy

Stable Diffusion 1.5 dostrajane na wszystkich klatkach 118 filmów treningowych Cataract-LMM (576×384, maski narzędzia,
rogówki i źrenicy jako warunek), na wielu kartach Edenu. Plan i różnice wobec `../dyfuzja_narzedzie`: `PLAN.md`.

| plik | rola |
|---|---|
| `kod/wspolne.py` | stałe, dekodowanie klatek, maski z wielokątów, pułapki danych |
| `kod/przygotuj_latenty.py` | klatki → latenty VAE i maski, jeden plik na film, wiele kart |
| `kod/trenuj_sd.py` | uczenie DDP (torchrun), EMA, zapis i wznowienie, podgląd, STOP |
| `eden/wyslij.sh` | kod, manifest, diffusers do `deps`, wagi SD 1.5 i VAE na Eden; bez zgłoszeń |
| `eden/przygotuj_sd.slurm` | przygotowanie latentów (`train` wszystkie klatki, `val` co 10.) |
| `eden/trenuj_sd.slurm` | uczenie na wszystkich kartach zadania, latenty w `/dev/shm`, STOP 15 min przed końcem |

## Stan (2026-09-30 02:30)

Sprawdzone lokalnie (RTX 3070):
- przygotowanie latentów: 2 filmy `train` co 20. klatka (575 klatek) i 2 filmy `val` co 50. (187), 10–13 klatek/s;
  odchylenie latentu 0,88–1,03 (skala SD ≈ 1); rekonstrukcje VAE wierne, kontury masek na narzędziu, rogówce
  i źrenicy (obraz kontrolny w notatkach sesji);
- uczenie małym UNet (3 mln parametrów, losowe wagi) 30 kroków: strata 1,11 → 0,72; zapis `ostatni.pt`, migawki EMA
  w formacie diffusers, podgląd; wznowienie od kroku 30 i 40; zatrzymanie plikiem STOP z zapisem i usunięciem STOP.

Nie sprawdzone lokalnie (8 GB karty nie mieści SD 1.5 z optymalizatorem): wagi SD 1.5, uczenie na kilku kartach (DDP).
To sprawdza zadanie próbne na Edenie (PLAN.md, krok 3).

Filmy na Edenie: `~/smallvit_data/cataract_lmm/3_Object_Tracking` (wysyłka `../../eden/wyslij_cataract_lmm.sh`).

## Eden (2026-09-30, zgoda użytkownika „tak, wysyłaj i trenuj”)

- `eden/wyslij.sh`: kod, diffusers 0.39.0 w `deps` (54 MB), wagi SD 1.5 fp16 (1,9 GB) i sd-vae-ft-mse (305 MB) w
  `~/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/modele`.
- **Limit konta `transformers_vsc` (wszyscy użytkownicy razem): 5 × H100, 17 × A100, 20 kart**, 600 procesorów. Ośmiu H100
  w jednym zadaniu nie da się dostać; na węzłach dgx wolne karty często nie mają wolnych procesorów.
- Próba 1818490/1818491 (przygotowanie 2 filmów `train` i 2 `val` co 10.): 73 klatki/s na jednej H100 (ograniczają
  procesory dekodujące JPEG), odchylenie latentu 0,88–1,03.
- Próba 1818492 (2 × H100, 16 na kartę × akumulacja 8 = 256, 200 kroków): **110 obrazów/s** (2,33 s/krok), 40 GB pamięci
  na karcie 0 (z EMA), strata 0,17 → 0,12, zapis `ostatni.pt` 13,75 GB w ~40 s; podgląd po 200 krokach: realistyczne
  klatki oka, narzędzie w miejscu obrysu (`wyniki/proba/podglad_000200.jpg`). A100 40 GB wymaga 8 na kartę.
- Zgłoszone: przygotowanie wszystkich klatek 118 filmów `train` (1818494, 4 × H100) i co 10. klatki 25 filmów `val`
  (1818495, 1 × H100); uczenie 1818496 (4 × H100, 32 na kartę × akumulacja 2 = 256, 20 000 kroków) po obu.
  Szacunek: ~220 obrazów/s, 5,1 mln przykładów ≈ 6,5 h.
