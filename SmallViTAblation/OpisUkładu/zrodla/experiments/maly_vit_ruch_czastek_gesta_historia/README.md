# Ruch tokenów narzędzia: gęsta historia z biegu na Edenie

Data: 2026-10-01. Plan pomiaru bez zmian: [../maly_vit_ruch_czastek_maska_1px/PLAN.md](../maly_vit_ruch_czastek_maska_1px/PLAN.md).
Polecenie użytkownika: „Możesz to wykonać jako test na Edenie. Zapisać co wejdzie”.

## Po co

Pomiar lokalny z 2026-09-30 ([../maly_vit_ruch_czastek_maska_1px/WYNIKI.md](../maly_vit_ruch_czastek_maska_1px/WYNIKI.md))
miał tylko dwa stany sieci (kroki 60 i 120), więc jeden przedział ruchu. Plan zakładał historię co 10 aktualizacji
do kroku 1000. Takich wag nie było, więc ten bieg je wytwarza: uczy sieć od zera jeszcze raz i zapisuje surowe
tokeny po każdej aktualizacji.

## Co się różni od planu

- **Nowy trening na klastrze.** Plan mówił „bez nowego treningu i bez klastra”; zmienione na wyraźne polecenie użytkownika.
- **Przepis uczenia bez zmian:** wspólna rozgrzewka pętli małego ViT (`trening.Uczen(47)`, `default_rng(47)`,
  baza 1500 klatek, paczka 32, AdamW 5e-4, rozgrzewka lr 300 kroków, bf16, odbicie i jasność). Kod `model.py`,
  `dane.py` skopiowany z `maly_vit_od_zera/eden/kod`, `trening.py` z `maly_vit_petla_sd_rownanie/kod` (ta sama
  pętla uczenia plus obserwator kroków, który nie zmienia uczenia ani losowania). Lokalne kroki 60 i 120 pochodziły
  z próby tego samego przepisu i ziarna, ale to inny przebieg na innym GPU: trajektorie nie są identyczne.
- **Gęściej niż w planie:** tokeny co 1 aktualizację (plan: co 10), kroki 0..1000, czyli 1001 stanów.
- **Zapis w float32**, nie float16 (o float16 prosił pierwotny plan pomiaru). Próba zmierzyła względną zmianę
  tokenów w jednym kroku: 1e-2 w kroku 1, do 5e-2 w kroku 20. Zaokrąglenie float16 dodałoby szum rzędu 1-2 procent
  tej zmiany, a więcej, jeśli ruch zwolni pod koniec. Kwota pozwala na float32, więc zapis jest dokładny.
- **Pomiar w trybie eval** jak w planie. Łodyga ma BatchNorm: jego statystyki biegnące zmieniają się w każdym kroku
  niezależnie od lr, więc ruch tokenów obejmuje też tę zmianę, nie tylko aktualizację wag. Próba to pokazuje:
  w kroku 1 lr wynosi 1,7e-6, a tokeny i tak zmieniają się o 1 procent.

## Co jest zapisane (Eden, `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny`)

| Plik | Zawartość | Rozmiar |
|---|---|---|
| `dane/LATENT_GESTY.npy` | (1001, 12, 600, 768) float32: krok, obraz, token, neuron | 22,1 GB |
| `wagi/k0000.pt` … `k1000.pt` | wagi sieci co 10 kroków (101 plików), do pomiaru na innych obrazach | ~1,2 GB |
| `pelny/k0000.pt` … | wagi + stan AdamW co 100 kroków, do wznowienia uczenia | ~0,4 GB |
| `dane/LATENT_SOND.npz` | dotychczasowa średnia telemetria 192 sond co 10 kroków (wariant uśredniony) | mały |
| `dane/OCENY.json`, `LOG.json`, `RUCH_KROKOW.json`, `metadata.json` | mAP@0,5 co 100 kroków, strata, względny ruch kroku, opis | mały |
| `analiza_co10/`, `analiza_co1/`, `analiza_kroki_60_120/` | kierunki ruchu i ocena poza dopasowaniem | mały |

Budżet: wolne było 47 GB kwoty projektu. Bieg zajmuje ~24 GB i zostawia ~23 GB zapasu innym zadaniom.
Wagi po każdym kroku dołożyłyby 11,7 GB; pominięte, żeby nie zejść do ~12 GB wolnego.
Lokalnie (dysk F: 6 GB wolnego) pobierane są tylko analizy, metadane i oceny; tokeny i wagi zostają na Edenie.

## Uruchomienie

```bash
cd ~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia
sbatch bieg_eden.slurm proba    # 20 kroków, sprawdzenie kodu -> wyniki/proba
sbatch bieg_eden.slurm pelny    # 1000 kroków -> wyniki/pelny
```

`bieg.py` odmawia nadpisania istniejącego `LATENT_GESTY.npy`. Analiza ruszy ponownie bez GPU:
`python analiza.py --dane wyniki/pelny/dane --wyniki <katalog> --co 10` (albo `--kroki 60,120`).

`analiza.py` to ta sama metoda co w pomiarze lokalnym. Sprawdzenie: na lokalnych danych (kroki 60 i 120) daje
identyczne liczby jak `../maly_vit_ruch_czastek_maska_1px/WYNIKI.md` (AUC 0,5063 / 0,5665 / 0,6689 / …).

## Status

Zakończone 2026-10-01: próba 1818729, bieg 1818732 (15 min, A100). Wynik: [WYNIKI.md](WYNIKI.md).
`kod/wykres_w_czasie.py` rysuje AUC w czasie i fazy z gotowego REPORT.json, bez tokenów.
