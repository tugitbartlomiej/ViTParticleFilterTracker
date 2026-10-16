# K1000: podmiana 150 klatek bazy na klatki znalezione przez SD i DINO, kontynuacja uczenia

Data: 2026-10-08. Polecenie użytkownika: „Weź właśnie te 150 zdjęć, podmień do tego zbioru uczącego, który był 1500
zdjęć … przeprowadź od tego checkpointa K1000 trening … zrób ramię porównawcze … do 5 checkpointów … zobaczymy, w którą
stronę latent pójdzie”, potem „jednak wrzucaj na eden!”.

## Ramiona

| tag | zbiór | po co |
|---|---|---|
| `C0` | oryginalna baza 1500 klatek K1000 | kontrola |
| `A150` | baza, w której 150 pozycji zastąpiono 150 klatkami TRAIN | ramię badane |
| `C0b` | powtórzenie C0 | poziom szumu obliczeń (w trybie deterministycznym oczekiwane 0) |

Nowe klatki: `outputs/sd_img2img_sterowanie_latentem/optymalizacja_regresja/PODOBNE_150_0088_1661` (DINO top-150 do
klatki 0088/1661, która była top-1 dla obrazu SD g19 z regresji szumu). Jedna z nich (0088/536, pozycja 19) jest
w bazie 1500, więc wypadła; w jej miejsce weszła następna z rankingu DINO bez bazy: 0088/1484 (pozycja 151).
143 ze 150 nowych klatek pochodzi z filmu 0088.
Usuwane pozycje: `default_rng(150).choice(1500, 150)` (83 filmy). Sondy: 64 pozycje z pozostałych (`default_rng(64)`).
Lista i hashe: `outputs/vit_k1000_podmiana150_sd/wejscia/PODMIANA.json`, `NOWE.json`, `CEL.json` (kod `kod/przygotuj.py`).

## Przepis (bez zmian względem gęstej historii, z której pochodzi K1000)

`kod/model.py`, `dane.py`, `trening.py` skopiowane bez zmian z `maly_vit_ruch_czastek_gesta_historia/kod`
(te same SHA256 co w jego SHA256SUMS). AdamW 5e-4, wd 0,05 (ndim >= 2) / 0, betas 0,9 / 0,999, rozgrzewka 300 kroków
(w kroku 1000 już minęła, lr stały), clip 1, bf16, odbicie 0,5, jasność/kontrast 0,8-1,2, paczka 32, kolejka permutacji.

Start: pełny stan `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny/k1000.pt`
(net + opt + krok 1000, SHA256 e7a7727c…), skopiowany do katalogu biegu. `bieg.py` sprawdza, że każdy tensor net jest
równy assetowi `MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt` (SHA256 3df42a6a…).

## Różnice, których nie da się uniknąć (jawnie)

1. **RNG kolejki i augmentacji od nowa:** `default_rng(47)` w kroku 1000. Stan RNG i kolejki paczek z kroku 1000
   oryginalnego biegu nie był zapisany, więc to nie jest wierna kontynuacja oryginalnej trajektorii, tylko nowa,
   wspólna dla ramion. Wszystkie ramiona dostają te same indeksy pozycji i te same augmentacje; w A150 na
   podmienionych pozycjach stoją inne obrazy.
2. **Epoka = 47 aktualizacji** (1504 ekspozycje, polityka kolejki pętli: dopełnianie nową permutacją), 5 epok =
   235 kroków, pomiary w K1000, K1047, K1094, K1141, K1188, K1235. Jedno ciągłe wywołanie `Uczen.kroki` na ramię.
3. **Tryb deterministyczny** (SDPA MATH, `use_deterministic_algorithms(warn_only)`, cudnn deterministic,
   `CUBLAS_WORKSPACE_CONFIG`). Oryginalny bieg był niedeterministyczny; bez tego dwie identyczne gałęzie rozchodzą się
   po kilku krokach o tyle, ile wynosi odpowiedź na obraz (pamięć projektu, 2026-10-08). C0b to sprawdza.
4. **Ramki nowych klatek** skalowane osobno w x i y do 480 × 320 (część klatek ma 720 × 576). Baza zostaje jak
   w oryginale (`dane.py`: stała skala 2/3), żeby C0 było identyczne z przepisem K1000.

## Pomiar (po każdej epoce i na starcie)

- Zdjęcie celu 0055/4019 (j = 10389 w `vit_dino_faiss_35000`, nie ma go w bazie): pełny latent 600 × 768
  (`bloki[-1].mlp.act`, eval, fp32 bez TF32), pewność i IoU najpewniejszej ramki.
- Grupy nowe150, usuniete150, sondy64: latent średni po tokenach i po tokenach ramki (2 × 768), pewność, IoU.
- mAP@0,5 na próbce 3000 klatek walidacji (`dane.walidacja`, jak oryginał).
- Kierunek (analiza lokalna): ruch latentu celu Δ_A − Δ_C, jego norma i kosinus z δ = z*_K1000 − z0_K1000, postęp
  ⟨z − z_K1000, δ⟩ / ||δ||². z*_K1000 policzony lokalnie tą samą procedurą co etap 8 `k8000_trening_trudne.py`
  (`kod/z_cel_k1000.py`, CPU): IoU 0 → 0,70, pewność 0,51 → 0,06, ||Δz|| / ||z|| = 2,83, „poza” 19,7 (prawdziwe 1,7).

## Czego to nie rozstrzygnie

- **Niezgodność przestrzeni:** 150 klatek znaleziono pod sztuczny latent K8000 (SD K8000, regresja szumu), a uczenie
  idzie od K1000 (polecenie użytkownika). Kierunek do z* mierzymy w przestrzeni K1000, gdzie ten wybór nie był
  optymalizowany.
- Jedno ziarno RNG kontynuacji; wniosek o mAP wymaga co najmniej 3 ziaren.
- 143 ze 150 klatek z jednego filmu; efekt może być efektem filmu 0088, nie „kierunku SD”. Brak ramienia z losowymi
  150 klatkami z puli (kontrola na samą podmianę) — do dodania, jeśli wynik A150 − C0 okaże się niezerowy.

## Uruchomienie (Eden)

```bash
cd ~/smallvit_runs/2026-10-08_k1000_podmiana150_sd
sbatch --export=NIL bieg_eden.slurm proba   # C0, 1 epoka -> wyniki/proba
sbatch --export=NIL bieg_eden.slurm pelny   # C0, A150, C0b po 5 epok -> wyniki/pelny
```
