# Trening DETR i YOLO OD ZERA (epoch 0 -> 400) na zbiorze 20k

**Utworzono:** 2026-06-15
**Cel:** ablacja "czy etap pretreningu na 91k jest potrzebny?" — wytrenować DETR i YOLO
bezposrednio na kurowanym zbiorze 20k OD EPOKI 0 do ~400, zamiast standardowego
pipeline'u (pretrain 0-170 na ~91k -> fine-tune 170-... na 20k).

## Co to znaczy "od zera" (wazne rozroznienie)

Produkcyjny pipeline projektu:
- DETR base: epoki 0-170 na pelnym poolu ~91k -> potem fine-tune 170-435 na 20k
- YOLO base: epoki 0-190 na ~91k -> potem fine-tune od ep170 na 20k

Te skrypty robia COS INNEGO: **trening bezposrednio na 20k od epoki 0**, BEZ
posredniego pretreningu na 91k. Inicjalizacja wag pozostaje TAKA SAMA jak w base
(zeby izolowac tylko zmienna "posredni pretrening na 91k"):
- DETR: `facebook/detr-resnet-50` (COCO-pretrained DETR) — bez `--resume_training`
- YOLO: `yolov8l.pt` (COCO-pretrained) — bez `--pretrained_path`

Czyli "od zera na 20k" = od epoki 0, na samym 20k, z COCO-init (jak base), do 400 epok.
Jezeli chcesz TRULY-RANDOM init (bez COCO), zob. uwagi nizej.

## Zbior 20k = kurowany V1 FULL (produkcyjny)

Oba skrypty trenuja na TYM SAMYM kurowanym 20k co model produkcyjny:
- DETR: pool 91k (`datasets_20250606.tar.gz`) + anotacje wariantu v1 FULL
  (`~/DETR/ablation/v1/annotations_{train,val}.json`) — identyczna infra jak ablacje
- YOLO: `20kSelectedImages.tar` + `merged_20k_annotations_fixed.json` — identyczna
  infra jak produkcyjny YOLO 20k fine-tune

## Uruchomienie na Edenie

```bash
# skopiuj folder na Eden (np. do ~/DETR/ i ~/Yolo/)
# DETR:
cd ~/DETR
sbatch /sciezka/run_detr_20k_from_scratch.slurm          # domyslnie v1 (FULL 20k)

# YOLO (po zakonczeniu DETR lub rownolegle, osobny job):
cd ~/Yolo/20k_finetune
sbatch /sciezka/run_yolo_20k_from_scratch.slurm
```

Kolejnosc wg prosby: najpierw DETR, potem YOLO (mozna tez rownolegle — to osobne joby).

## Parametry (wybrane, udokumentowane)

| | DETR | YOLO |
|---|---|---|
| Entry | `detr_train_ablation.py` | `yolo_train_20k_finetune_v2.py` |
| Init | facebook/detr-resnet-50 (COCO) | yolov8l.pt (COCO) |
| Epochs | 0 -> 400 | 0 -> 400 |
| Partycja | hopper | hopper |
| GPU | 4 | 4 |
| Batch/GPU | 8 (eff. 32) | 32 |
| LR | 1e-4 / backbone 1e-5, cosine, warmup 5 | lr0=0.01, lrf=0.01 (SGD) |
| Patience | 60 | 100 |
| Save co | 10 epok | 10 epok |
| Czas (limit) | 4 dni | 3 dni |

LR dla DETR = wartosci z base-treningu (`run_detr_4gpu_hopper_resume140`: 1e-4/1e-5),
NIE zredukowane fine-tune'owe 5e-5. LR dla YOLO = standardowy full-training lr0=0.01
(NIE 0.0001, ktore bylo zredukowane SPECJALNIE pod fine-tune z modelu juz zbieznego).

## Wyniki

- DETR: checkpointy zachowywane co 10 epok + archiwizowane do
  `~/DETR/Checkpoints/scratch_20k/v1/` (NIE kasujemy posrednich — potrzebne do
  benchmarku trajektorii ep100/200/300/400, jak przy base DETR). Best w
  `~/DETR/scratch_20k_best/v1/`.
- YOLO: `~/Yolo/20k_finetune/train_out_scratch_20k/`, best w
  `~/Yolo/20k_finetune/best_scratch_20k/`, koncowe .pt w `final_scratch_20k/`.

## Jak porownac z obecnymi liczbami w artykule

Po treningu: ten sam protokol Q81 (DETR) / standardowy (YOLO) na external benchmark
(`local_eval_2026-06-11`). Headline porownanie:
- DETR-from-scratch-20k vs DETR-curated (91k->20k fine-tune, 26.04% FULL Q81)
- YOLO-from-scratch-20k vs YOLO-curated (91k->20k fine-tune, 34.17% FULL)
Jezeli from-scratch < pipeline -> potwierdza wartosc posredniego pretreningu na 91k.

## Modyfikacje (gdybys chcial inaczej)

- **Truly-random init DETR** (bez COCO): trzeba inicjalizowac `DetrForObjectDetection`
  z samego `DetrConfig` (nie `from_pretrained`) — wymaga drobnej zmiany w
  `detr_train_ablation.py`; obecny skrypt tego nie robi (COCO-init).
- **Truly-random init YOLO** (bez COCO): zmien w slurm `--model_size l` na sciezke do
  `yolov8l.yaml` przez modyfikacje wrappera (obecnie wrapper laduje `yolov8{size}.pt`).
- **Inny rozmiar YOLO** (m zamiast l): zmien `--model_size`.
- **Inny budzet epok**: zmienna `EPOCHS=` przed sbatch (DETR) / edytuj `--epochs` (YOLO).
- **Inny wariant 20k** (np. v5 RANDOM): `VARIANT=v5 sbatch run_detr_20k_from_scratch.slurm`.
