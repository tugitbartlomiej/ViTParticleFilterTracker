# Kompensacja: kontrola „oba zamrożone” i drugi układ (CIFAR-10: DeiT-Tiny i MałyViT)

**Data:** 2026-10-26, 19:02 (Europe/Warsaw). **Typ:** część K = 3 treningi od zera na Edenie z pomiarem; część C = pomiar na istniejących
checkpointach 6 biegów CIFAR-10 (bez treningu). **Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane przed uruchomieniem).

## Cel

Rodzic `mosty_fizyki_24seedy_ablacje_20261026__13-04-31` wykazał na 24 seedach replikowalny podział pracy (uwaga ściska i jest styczna,
MLP jest nienormalnym ścinaniem i rozdziela klasy, trening przenosi ściskanie z uwagi do MLP) oraz w ablacjach kompensację (zamrożony
MLP albo zamrożona uwaga nie psują AP, blok równoległy też nie). Dwie luki, których tamten pomiar nie zamyka:

- **K.** Nie wiadomo, ile detekcji daje sama łodyga z głową przy losowych, zamrożonych blokach. Bez tego „kompensacja” może być
  złudzeniem: AP 0,86 przy zamrożonym MLP może pochodzić nie z uwagi, lecz z łodygi. Wariant `frozen_both` to ten punkt odniesienia.
- **C.** Jeden zbiór i jedna architektura. Biegi CIFAR-10 z `experiments/deit_cifar10_20261013` (DeiT-Tiny 12 bloków z CLS; MałyViT na
  128 px; oba 192 wymiary, MLP 768) dają drugi zbiór i drugą architekturę bez nowego treningu, w tym kontrole bez weight decay i
  z losowymi etykietami, czyli odpowiedniki A5 i A6 rodzica.

## Rodzice i źródła

- `experiments/mosty_fizyki_24seedy_ablacje_20261026__13-04-31`: definicje miar A, B, C (kod `pomiar_mosty_eden.py` skopiowany bez
  zmian poza dodaniem wariantu `frozen_both` do listy), trener `trening_ablacje.py` (dodany wariant `frozen_both`), `model.py`
  (SHA256 0570c957…, identyczny z `deit_cifar10_20261013/kod/z_edena/model.py` użytym w biegach CIFAR), wyniki jako źródło predykcji.
- `experiments/deit_cifar10_20261013` (README tamże): biegi na Edenie `~/smallvit_runs/2026-10-13_deit_cifar10/wyniki/<bieg>/wagi/k<krok>.pt`
  ({'net', 'krok'}), 60 epok = 93 780 kroków, przepis jak s47 (paczka 32, AdamW 5e-4, wd 0,05, rozgrzewka 300, bf16); dokładność
  top-1 na końcu: DeiT 0,750 (z obcięciem) / 0,753 (bez), MałyViT 0,852 (s47) / 0,854 (s48) / 0,852 (wd0) / 0,051 (losowe etykiety).
  Model DeiT: `Kod_eksperymentu/model_deit_cifar.py` (kopia). Dane: `~/smallvit_runs/2026-10-13_deit_cifar10/dane/cifar10_uint8.npz`
  (SHA256 68e07d95…, 50 000 + 10 000 obrazów); pomiar na zbiorze testowym.
- Część K: dane treningu i checkpointy jak w rodzicu (manifest kampanii `kompensacja_funkcjonalna_dgx2_20261017`, archiwa zip).

## Procedura

**Część K** (`Kod_eksperymentu/ablacje_lodyga.slurm`, tablica 3 zadań, 1 × A100 + 3 CPU): `trening_ablacje.py --wariant frozen_both`
na seedach 3047, 4047, 5047 (uwaga, LN₁, MLP, LN₂ każdego bloku `requires_grad=False`; uczone: łodyga, norma końcowa, głowa; ten sam
przepis i siatka 12 checkpointów co w rodzicu), potem `pomiar_mosty_eden.py` na jego checkpointach (500 klatek TEST, 50 diagnostycznych).
Wyjście: `Wyniki/Dane/ablacje/frozen_both_s<seed>/{weights/, loss.jsonl, COMPLETE.json, pomiar/}`.

**Część C** (`Kod_eksperymentu/pomiar_cifar.slurm`, tablica 6 zadań, 1 × A100 + 3 CPU): `pomiar_mosty_cifar.py` na 14 checkpointach
(0, 300, 1000, 3000, 6000, 10 000, 16 000, 24 000, 33 000, 44 000, 55 000, 65 000, 80 000, 93 780) każdego biegu: 2000 obrazów testowych
(200 na klasę, ziarno 2026) dla B, C i dokładności; 100 obrazów (10 na klasę) × 16 losowych tokenów (+ CLS) × wszystkie bloki dla A.
Separacja klas liczona na poziomie obrazów (średnia tokenów obrazu i, w DeiT, CLS). Wyjście: `Wyniki/Dane/cifar/<bieg>/{wyniki.npz,
podsumowanie.json, WAGI_SHA256.txt, OCENY_treningu.json, COMPLETE.json}`.

**Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`.

## Metryki

Jak w rodzicu (docstringi `pomiar_mosty_eden.py` i `pomiar_mosty_cifar.py`); progi w `PREDYKCJE_PRZED_POMIAREM.md`. Dla CIFAR miary klasowe
per token z rodzica (sep_u per obraz, D_u_nn, cos_nt) są NaN, bo klasa jest cechą obrazu; zastępuje je `sep_u_mean`, `sep_x_mean`,
`sep_u_cls` i dokładność najbliższego centroidu. Skutek zamiany bloku dla wyjścia: poprawność top-1 i margines logitów zamiast IoU.

## Ograniczenia

- Część K: 3 seedy; test znaku 3/3; porównanie z base/frozen_mlp/frozen_attn rodzica tego samego seedu (ten sam kod i sprzęt).
- Część C: jeden seed DeiT (47, dwa warianty obcięcia), dwa seedy MałyViT; obraz 32 px, 64 tokeny (vs 600 w detekcji), 10 klas.
  Bieg `maly_s47_obciecie` ma wagi co 100 kroków, pozostałe co 1000 po kroku 1000; siatka 14 checkpointów jest wspólna.
- DeiT: CLS wchodzi do miar B (65 tokenów) i jest osobno flagowany w A; separacja na średniej tokenów obrazu nie jest odczytem DeiT (CLS).
- Pomiar CIFAR używa `timm` 1.0.24 i `torch` 2.10 z venv rodzica; biegi trenowano w innym venv, klucze state_dict zgodne (152 / 96).

## Odtwarzanie

```
python experiments/eden.py init kompensacja_lodyga_i_cifar_20261026__19-01-59
python experiments/eden.py push kompensacja_lodyga_i_cifar_20261026__19-01-59
python experiments/eden.py submit kompensacja_lodyga_i_cifar_20261026__19-01-59 --script Kod_eksperymentu/pomiar_cifar.slurm
python experiments/eden.py submit kompensacja_lodyga_i_cifar_20261026__19-01-59 --script Kod_eksperymentu/ablacje_lodyga.slurm
python experiments/eden.py watch kompensacja_lodyga_i_cifar_20261026__19-01-59 --pull-on-complete
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próby dymne: `--sbatch-arg=--array=0 --script-arg=proba` (CIFAR, 2 checkpointy, 20 obrazów) i `--script-arg=smoke` (frozen_both).

## Stan

Zobacz `Wyniki/Dane/Eden/STATUS_EDEN.txt` i koniec tego pliku (uzupełniany po pomiarze).

- 2026-10-26 19:13: próby dymne na dgx-2: CIFAR (1932993_0, s47_obciecie, 2 checkpointy, 20 obrazów, 70 s; DeiT wczytany `strict=True`, acc 0,70 na 20
  obrazach przy k93780) i frozen_both (1932995_0, smoke: 64 klatki, 4 kroki, 80 s). Wyniki prób: `Wyniki/Dane/proby_dymne/` i
  `Wyniki/Dane/ablacje_smoke/`; nie wchodzą do oceny. Lokalnie przed wysłaniem: test z losowymi wagami obu modeli (152 / 96 kluczy).
- 2026-10-26 19:13: zgłoszone pełne tablice: CIFAR 1932996 (zadania 0–5, 1 × A100 + 3 CPU, 6 h) i frozen_both 1932997 (0–2, 3 CPU, 6 h),
  partycja short; GPU zajęte innymi zadaniami, zadania czekają na zasoby.
