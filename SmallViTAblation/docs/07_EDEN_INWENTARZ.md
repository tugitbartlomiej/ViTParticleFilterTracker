# 07 — Inwentarz: co już policzyłem na Edenie pod podziałem po pacjencie

Sprawdzone 2026-09-08 w lokalnej kopii `Eden/EdenBackup/` (kopia z 2026-07-22,
źródła YOLO skasowane z klastra po weryfikacji 2026-07-23).

**Odpowiedź krótka: tak, podział po pacjencie był używany — ale tylko w części
biegów, i każdy z nich chodził na innej recepcie treningowej.**

---

## 1. Co istnieje pod podziałem po pacjencie (35/8/7, ziarno 42)

| Bieg | Model | Zbiór treningowy | Start | Budżet | Stan |
|---|---|---|---|---|---|
| `DETR/scratch_20k/v1` | DETR-R50, 100 zapytań | **kurowany 20k**, wariant v1 | od zera | 0 → 400 epok | checkpointy do **ep240**, jest `best_model.pth` — nie dojechał do celu |
| `DETR/fullpool_vs_scratch` | DETR-R50, 100 zapytań | **cały trening 71 316** (bez kuracji) | od zera | — | checkpointy ep50–ep90, jest `best_model.pth` |
| `DETR/fullpool_vs_scratch_q1` | DETR-R50, **1 zapytanie** | cały trening 71 316 | od zera | — | **anulowany na epoce 6** |
| `Yolo/…/train_out_scratch_90k_vs` | YOLOv8L | **cały trening 71 316** | od zera | 190 epok | **kompletny**, checkpointy co 10, `best.pt`, `last.pt` |
| `DETR/ablation/v1..v7` | DETR-R50 | 7 wariantów kuracji do 20k | **wznowienie z ep170** | 170 → 300, early stopping 30 | kompletne, to jest dawna [Tab. 5] (tabela piąta) |

Wynik YOLO na pełnym treningu pod podziałem po pacjencie, ostatnia epoka:
**mAP@0.5 = 0,982** i **mAP@[.5:.95] = 0,778** na walidacji własnego rozkładu.

## 2. Czego NIE ma

- **YOLO na kurowanym 20k pod podziałem po pacjencie.** Wszystkie biegi YOLO na
  20k (`train_out_ep170_lr0001`, `ep170_to_500`, `20k_finetune_short`) to
  dostrajanie od `epoch170.pt`, czyli od modelu z **losowego podziału klatek**.
  Brakuje więc pary „YOLO kurowany vs YOLO niekurowany" pod poprawnym podziałem.
- **DETR pełny pipeline w tej samej recepcie co warianty ablacji.** `scratch_20k/v1`
  ma dane v1, ale inną receptę (od zera, 400 epok) niż V2–V7 (wznowienie
  z ep170, 300 epok, early stopping).

## 3. Znalezisko, które trzeba zapisać: ablacja startuje z modelu z wyciekiem

`Eden/Scripts/AblationStudy/detr/run_ablation_variant.slurm`, linia 60:

```
EPOCH_170_CKPT=$HOME_DIR/DETR/Checkpoints/checkpoint_epoch_170.pth
```

i dalej, linia 149: *„Copy epoch 170 checkpoint as starting point"*.

To znaczy, że **każdy wariant ablacji V1–V7 zaczynał od modelu wytrenowanego
wcześniej przez 170 epok na pełnej puli z losowym podziałem klatek** — czyli
od modelu, który widział klatki z filmów testowych.

Konsekwencja: sam podział danych w ablacji był poprawny, ale **wyciek wszedł
tylnymi drzwiami, przez inicjalizację**. Żaden recenzent tego nie napisał, bo
tego nie było widać w tekście artykułu.

To samo dotyczy YOLO: `epoch170.pt` (md5 `4744db6d9c86ba99d7dcc85879d9614b`)
pochodzi z biegu `train_out_5gpu_hopper` z 2025-06-09, z **losowym podziałem
klatkowym** — 60 693 / 11 380 / 3 794. Stąd wynik mAP@0.5 = 0,995 w tamtym
biegu, który nigdy nie był realny.

**Wniosek dla nowego protokołu:** start wyłącznie z wag COCO, nigdy z żadnego
checkpointu ep170. To już jest zapisane w `configs/protocol.yaml` jako
`init: coco_pretrained` — teraz wiadomo, dlaczego to jest ważne.

## 4. Trzy różne recepty w biegach, które wyglądają na porównywalne

| Recepta | Kto | Start | Epoki | Stop |
|---|---|---|---|---|
| A | ablacja V1–V7 | ep170 (z wyciekiem) | 170 → 300 | early stopping, cierpliwość 30 |
| B | `scratch_20k/v1` | od zera | 0 → 400 (doszedł do 240) | budżet epok |
| C | `fullpool_vs_scratch`, YOLO 90k vs | od zera | ~90 / 190 | budżet epok |

Żadne dwie pozycje z tej tabeli nie są ze sobą porównywalne wprost. To jest
dokładnie ten confound, który zarzucił recenzent 1 w punkcie 4 — tylko szerszy,
bo dotyczy nie dwóch, a trzech recept.

## 5. Co z tego da się użyć w nowym artykule

**Bez ponownego liczenia:**

- YOLO na pełnym treningu pod podziałem po pacjencie (190 epok, kompletny) —
  jako punkt odniesienia „ile daje cały zbiór bez kuracji",
- DETR na pełnym treningu pod tym samym podziałem (ep50–90) — to samo dla
  drugiej rodziny,
- obserwacja z sekcji 3 jako **wyjaśnienie, dlaczego stare liczby były zawyżone** —
  to jest uczciwa narracja „znaleźliśmy u siebie więcej niż recenzenci".

**Do policzenia od nowa, bo recepty się nie zgadzają:** wszystko, co ma
porównywać warianty kuracji między sobą. Czyli cała macierz z `02_MATRIX.md`.

**Nie warto wskrzeszać:** wariantu z jednym zapytaniem (anulowany na epoce 6)
i dociągania `scratch_20k/v1` z epoki 240 do 400 — obie pozycje należą do starej
recepty i nowy protokół i tak liczy je od nowa na małym ViT.
