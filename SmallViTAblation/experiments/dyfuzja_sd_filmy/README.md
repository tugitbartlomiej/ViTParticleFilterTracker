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
