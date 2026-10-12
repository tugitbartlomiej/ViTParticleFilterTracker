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

## Wynik uczenia 1818496 (zakończone 2026-09-30 17:56)

- COMPLETED, kod wyjścia 0, 7 h 05 min; 20 000 kroków, 16 epok po 311 886 klatkach ze 118 filmów `train`;
  1,225 s/krok, ok. 209 obrazów/s, 57,1 GB na kartę; bez NaN i błędów w logu.
- Strata (średnia z ostatnich 500 kroków): 0,179 (krok 50) → 0,130 (2000) → 0,113 (5000) → 0,102 (10 000) →
  0,095 (15 000) → 0,092 (20 000). Spada coraz wolniej, ale jeszcze nie stoi.
- Podgląd `wyniki_eden_1818496/podglad_020000.jpg` (rzędy 1 i 3: klatka prawdziwa po VAE; rzędy 2 i 4: obraz z
  modelu z tych samych masek; kontury: czerwony narzędzie, zielony rogówka, niebieski źrenica): oko i źrenica w
  obrysach, narzędzie w większości przykładów wzdłuż czerwonego konturu; w kilku obrazach ziarnista, plamista
  tekstura; napis szpitala i daty rysowany jako nieczytelne znaki.
- Migawki EMA na Edenie: 5000, 10 000, 15 000, 20 000 (`modele`/katalog biegu) i `ostatni.pt` 13,75 GB. Lokalnie
  `F:\Studia\PhD_projekt\VIT\eden_archiwum\sd_modele\ema_020000` (sha256 safetensors
  `255fb1a5c207a9d81392d72baecbba6d88bdd79adb9fe43917778db0d6738eb6`, zgodne z Edenem), obok `ema_010000` i `ema_015000`.
- Log, konfiguracja i podglądy końca: `wyniki_eden_1818496/` (`LOG.jsonl`, `KONFIG.json`, `STATUS.json`,
  `podglad_018000.jpg`, `podglad_020000.jpg`).
- Nie zmierzone: jakość liczbowa (FD-DINOv2, kopiowanie klatek, zgodność narzędzia z maską) — `ocena_sd.py` czeka na zgodę.
