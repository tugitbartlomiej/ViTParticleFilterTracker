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
- 2026-10-26 20:53: część C ukończona (1932996_0…_5 COMPLETED; DeiT 1 h 32 min, MałyViT 42–47 min), wyniki pobrane, ocena (progi bez zmian):
  **C1 PASS 5/5** (‖N‖/‖J‖ bloku środkowego: k0 0,75–0,77 → koniec 0,92–0,93 w DeiT i MałymViT), **C1z PASS** (losowe etykiety 0,93),
  **C2z PASS** (losowe etykiety: ‖J‖ 3,26 → 0,41), **C6 MałyViT PASS 3/3** (Σ Δsep MLP 0,55–0,56 vs uwaga 0,27–0,36), **C7 PASS 6/6**,
  **C8 PASS**, **C9b PASS 5/5**. FAIL: C2 0/5 (‖J‖ bloku środkowego spada w treningu i nie wraca do poziomu k0: DeiT 2,09 → 0,90/0,92,
  MałyViT 3,26/3,34 → 1,41/2,10/1,91; minimum 0,1–0,7 przy 3000–10 000 krokach), C3 0/5 (w CIFAR ostatni blok jest najbardziej
  nienormalny, 0,95–0,96, odwrotnie niż w detekcji), C4a 4/6 (uwaga DeiT przy k0 prawie nie ściska, iloczyn 0,954), C4b 2/5 (MałyViT:
  uwaga przestaje ściskać 0,60–0,62 → 0,89–0,96, ale MLP ściska tylko w wd0; DeiT: MLP ściska 1,00 → 0,78/0,84, uwaga w biegu z obcięciem
  ściska mocniej), C5 4/5 (uwaga styczna 0,93–0,99 we wszystkich blokach poza blokiem 1 DeiT bez obcięcia, 0,89), C6 DeiT 0/2 (separacja
  średniej tokenów prawie nie rośnie, 0,64 → 0,79/0,92; separacja CLS rośnie 0,6 → 1,1 głównie w blokach 8–12), C9a 0/5 (szczyt defektu
  komutatora zawsze w bloku 1).
- Obserwacja wspólna z detekcją (opisowa): blok 1 jest krytyczny. Usunięcie MLP bloku 1 (tylko uwaga) obniża dokładność DeiT
  0,76 → 0,09–0,11, MałyViT 0,85 → 0,55–0,72 (w detekcji IoU 0,80 → 0,24); zamiana na równoległy kosztuje 0,05–0,11 tylko w bloku 1,
  w pozostałych blokach ≤ 0,01. ‖J‖ MLP bloku 1 przy końcu: DeiT 71–77, MałyViT 10–37 (inne bloki 0,8–3,6).
- Losowe etykiety w CIFAR (inaczej niż w detekcji): uwaga ściska silnie (iloczyn 0,16), ruch w blokach 5–6 przestaje być styczny
  (0,28–0,35); chmura tokenów zapada się.
- 2026-10-26 21:35: część K ukończona (1932997_0…_2 COMPLETED, 1 h 31 – 1 h 46; trening frozen_both ~50 min, pomiar ~35 min), wyniki pobrane, ocena:
  **K1 PASS 3/3** (AP@0,5 frozen_both 0,463 / 0,491 / 0,618 wobec base 0,888 / 0,885 / 0,864, frozen_mlp 0,866 / 0,857 / 0,884,
  frozen_attn 0,861 / 0,861 / 0,786: sama łodyga z głową daje 52–72 % detekcji, więc kompensacja w ablacjach rodzica jest prawdziwa),
  **K2 PASS 3/3** (zamrożone bloki nie budują separacji: Σ Δsep uwaga −0,004…+0,008, MLP −0,034…−0,047; separacja przez bloki lekko spada),
  **K3 FAIL 1/3** (separacja po łodydze 1,05 / 1,07 / 1,10 wobec base 0,93–0,94 i frozen_mlp 1,02–1,03: łodyga przejmuje część, ale próg
  1,10 osiąga tylko s5047), **K4 PASS 3/3** (iloczyn kontrakcji uwagi 0,62–0,66 → 1,00: uwaga o zamrożonych wagach staje się dokładnie
  identycznością, jak w frozen_attn), **K5 PASS 3/3** (‖N‖/‖J‖ bloku 4 = 0,76, poziom losowy).
- Wniosek części K: bez uczonych bloków AP spada o 0,27–0,42, czyli ok. 40 % detekcji pochodzi z bloków, ale z dowolnego ich
  podzbioru (uczona sama uwaga albo sam MLP daje tyle samo co oba). Separacja na sferze nie jest tym, co bloki wnoszą (K2);
  ich wkład nie jest widoczny w miarach geometrycznych rodzica.
