# Mały ViT od zera na 35 000 klatkach: 21 epok, wagi po każdej epoce

Data: 2026-10-09. Polecenie użytkownika: „puść trening ViT na Edenie od zera na zbiorze 35000 zdjęć aż np. do 21 epok,
zapisuj każdą jedną epokę i ewentualne kroki. Pilnuj miejsca na Edenie”.

## Po co

Pomiar naturalny w aplikacji (zakładka „Ruch zbioru”, strona „Uczenie naturalne”) wczytuje kolejne checkpointy i
mierzy cząsteczki tych samych 1500 zdjęć. Model K8000 (`assets/MalyViT_K8000_s47_losowe`) ma tylko wagi końcowe.
Na Edenie bieg `2026-10-01_petla43_sd_prowadzenie/.../petla43b_s47/losowe` ma 116 KB i żadnych wag pośrednich
(sprawdzone 2026-10-09). Historii K8000 nie da się więc odtworzyć. Ten bieg wytwarza nową, dłuższą historię
uczenia od zera, zapisaną gęsto.

## Przepis

- Kod `model.py`, `dane.py`, `trening.py` skopiowany bez zmian z `../maly_vit_ruch_czastek_gesta_historia/kod`.
  To ten sam przepis co gęsta historia K1000: `trening.Uczen(47)`, `default_rng(47)`, paczka 32, AdamW 5e-4,
  wd 0,05, rozgrzewka lr 300 kroków, potem stały lr, bf16, odbicie 0,5 i jasność/kontrast ±20 %.
- Zmienia się tylko zbiór: `wejscia/MANIFEST_35000.jsonl`, kopia `outputs/vit_dino_faiss_35000/manifest.jsonl`.
  To 35 000 klatek TRAIN Cataract-LMM Object Tracking z 117 filmów, każda z ramką narzędzia.
- Klatki czytane prosto z archiwów zip na Edenie (`smallvit_data/cataract_lmm/3_Object_Tracking/annotations`,
  te same pliki co lokalnie, SHA-256 zgodne). SHA-256 każdej klatki jest sprawdzane z manifestem.
- Każda klatka jest rozciągana do 480 × 320, a ramki skalowane osobno w X i Y. Tak samo przygotowano
  `MANIFEST_1500` pomiaru: dla 200 wspólnych klatek ramki zgadzają się do 1e-5 piksela.
- Epoka to 1094 kroki, czyli 35 008 próbek. 21 epok daje 22 974 kroki. Koniec epoki e to krok 1094 · e,
  więc w aplikacji krok 1094 wybiera dokładnie końce epok.
- Walidacja mAP@0,5: ta sama próbka 3000 klatek co w biegach K1000, bez klatek obecnych w treningu.

## Co jest zapisywane (Eden, `~/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/pelny`)

| Plik | Zawartość | Rozmiar |
|---|---|---|
| `wagi/k00000.pt` … | `{'net', 'krok', 'epoka'}`: krok 0, co 10 do 1000, co 100 dalej, koniec każdej epoki | ~331 plików, ~3,9 GB |
| `pelny/stan_eEE.pt` | wagi, AdamW i stan losowania po epoce; zostają 2 ostatnie | ~70 MB |
| `dane/OCENY.json` | mAP@0,5: krok 0, co 1000 kroków, koniec każdej epoki | mały |
| `dane/LOG.json`, `dane/metadata.json` | strata co 50 kroków, opis biegu, sumy kodu i manifestu | mały |

Format `wagi/k<krok>.pt` to dokładnie to, czego oczekuje pomiar naturalny aplikacji. Krok kroku 1094 wybiera
same końce epok, krok 100 gęstą siatkę.

## Pilnowanie miejsca

- Przed startem kwota projektu miała 98,8 GB zajęte z 200 GB i 114 560 z 200 000 plików.
- Bieg odmawia startu przy wolnej kwocie poniżej 18 GB.
- Przed zapisem co 100 kroków i na końcu epoki sprawdza `lfs quota -p 1299`. Poniżej 12 GB pomija gęste kroki,
  a końce epok dalej zapisuje. Poniżej 4 GB zapisuje stan i kończy.
- Dane robocze w `/dev/shm` węzła, sprzątane przy końcu i przy błędzie. Klatki trenowania tylko w RAM.

## Uruchomienie

```bash
cd ~/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok
sbatch bieg_eden.slurm proba    # 512 klatek, 2 epoki -> wyniki/proba
sbatch bieg_eden.slurm pelny    # 35 000 klatek, 21 epok -> wyniki/pelny
sbatch bieg_eden.slurm wznow    # po przerwaniu: od ostatniego pelny/stan_e*.pt
```

Wznowienie odtwarza wagi, AdamW i generator, ale kolejka permutacji zaczyna się od nowa. Trajektoria po
wznowieniu nie jest więc bitowo tą samą trajektorią.

## Uwaga do pomiaru na 1500 zdjęciach

200 z 1500 zdjęć `MANIFEST_1500` należy do zbioru treningowego 35 000. Pomiar naturalny tych 200 zdjęć jest
pomiarem na danych uczących, pozostałych 1300 nie.

## Status

Zobacz `STATUS.md`.
