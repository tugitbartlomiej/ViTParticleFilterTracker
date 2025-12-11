# 🔍 DETR Training na klastrze EDEN — kompletna konfiguracja i wnioski (2025‑10‑31)

Ten dokument podsumowuje pełną konfigurację i historię treningu DETR na EDEN oraz wnioski z dotychczasowych benchmarków (YOLO vs DETR). Zawiera też checklistę naprawczą dla ewaluacji oraz rekomendacje kolejnych kroków.

---

## Historia treningu (SLURM)

1) Job 1190351 — start (2025‑10‑20)
- Node: dgx‑1; GPUs: 2×A100; RAM: 200 GB; limit: 5 dni
- Epoki: 0 → 125
- SLURM: `run_detr_2gpu_500epochs.slurm`

2) Job 1190356 — resume z epoki 125 (2025‑10‑20)
- Node: dgx‑1; GPUs: 2×A100; RAM: 200 GB
- Epoki: 125 → 140
- SLURM: `run_detr_2gpu_500epochs_with_archival.slurm`
- Nowość: system archiwizacji checkpointów co 10 min

3) Job 1190904 — resume z epoki 140 (2025‑10‑22)
- Node: dgx‑2; GPUs: 6×A100; RAM: 350 GB
- Epoki: 140 → 148 (CRASH — NaN values)
- Status: FAILED (exit code ‑6)

4) Bieżący Job — resume z epoki 140 na Hooper (2025‑10‑28+)
- Node: hooper; GPUs: 4×H100; RAM: 250 GB; partition: hopper
- Epoki: 140 → 160+
- SLURM: `run_detr_4gpu_hopper_resume140.slurm`
- Kluczowa zmiana: Mixed Precision (AMP) WYŁĄCZONY (fix dla NaN na ep. 148)

---

## Aktualna komenda treningu (rdzeń)

```
torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  detr_train_optimized.py \
  --images_dir "$IMAGES_DIR" \
  --annotations_path "$ANNOT_JSON" \
  --output_dir ./train_out_ddp_4gpu_hopper_resume140 \
  --checkpoint_dir ./ckpt_ddp_2gpu_dgx1_500ep \
  --best_model_dir ./best_ddp_4gpu_hopper_resume140 \
  --epochs 500 \
  --batch_size 16 \
  --num_workers 4 \
  --num_queries 100 \
  --save_interval 10 \
  --patience 50 \
  --augment \
  --model_checkpoint facebook/detr-resnet-50 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 1 \
  --resume_training
```
Uwaga: bez `--use_amp` → AMP OFF.

---

## Konfiguracja sprzętowa

| Parametr   | Wartość |
|------------|---------|
| Node       | hooper  |
| GPUs       | 4×H100  |
| Memory     | 250 GB  |
| Partition  | hopper  |
| Time limit | 5 dni   |

## Architektura modelu

| Parametr   | Wartość                   |
|------------|---------------------------|
| Base       | facebook/detr-resnet-50   |
| Backbone   | ResNet‑50                 |
| NumQueries | 100                       |
| Classes    | 1 (surgical_tool/tooltip) |

Uwaga: w danych historycznych nazwa klasy bywała „surgical_tool”. Docelowo ujednolicić na „tooltip” z `category_id=1` w całym pipeline (trening + ewaluacja).

## Hiperparametry treningu

| Parametr               | Wartość     | Opis                           |
|------------------------|-------------|--------------------------------|
| Epochs                 | 500 (target)| Obecnie ~160+                  |
| Start Epoch            | 140         | Resume                         |
| Batch / GPU            | 16          |                                |
| Effective Batch        | 64          | 4 GPUs × 16                    |
| LR (główny)            | 1e‑4        |                                |
| LR (backbone)          | 1e‑5        |                                |
| Weight Decay           | 1e‑4        |                                |
| Max Grad Norm          | 0.1         | Clipping                       |
| Grad Accumulation      | 1           | Brak akumulacji                |
| Augmentation           | Włączone    | Flip poziomy                   |

## Ustawienia optymalizacji

| Parametr            | Wartość   | Uzasadnienie                     |
|---------------------|-----------|----------------------------------|
| Mixed Precision     | OFF       | Fix dla NaN na ep. 148           |
| Torch Compile       | OFF       | SIGSEGV z DDP                    |
| TF32                | ON        | Optymalizacja A100/H100          |
| Fused AdamW         | ON        | Szybszy optimizer                |

## Dane i loader

| Parametr         | Wartość                              |
|------------------|--------------------------------------|
| Annotations JSON | augmented_coco_20250417_030014.json  |
| Train/Val Split  | 0.9 / 0.1 (w skrypcie treningowym)   |
| Num Workers      | 4                                    |
| Persistent       | True                                 |
| Pin Memory       | True                                 |
| Prefetch Factor  | 2                                    |

Uwaga: docelowo rekomendowana walidacja/test bez przecieku (split po wideo) i stały test wspólny dla YOLO/DETR.

## Checkpointing / DDP

| Parametr        | Wartość                                |
|-----------------|----------------------------------------|
| Save Interval   | co 10 epok                             |
| Patience        | 50 epok                                |
| Checkpoint Dir  | ./ckpt_ddp_2gpu_dgx1_500ep             |
| Archive Dir     | /mnt/evafs/.../DETR/Checkpoints        |
| Resume          | checkpoint_epoch_140.pth               |
| DDP Backend     | nccl (world_size=4, master_port=29500) |

---

## Benchmarki (YOLO vs DETR) — interpretacja

Ścieżka wyników: `YOLO_DETR_Benchmarks\\Benchmarks\\2025-10-31_18-30-53_CaDTD_YOLO_vs_DETR`.

Z logów:
- DETR ep. 100: mAP@0.5 ≈ 1.23%, mAP@[.5:.95] ≈ 0.34%, AR@100 ≈ 1.44%
- DETR ep. 160: mAP@0.5 ≈ 0.21%, mAP@[.5:.95] ≈ 0.03%, AR@100 ≈ 1.29%
- YOLO ep. 100: mAP@0.5 ≈ 1.25%, mAP@[.5:.95] ≈ 0.34%, AR@100 ≈ 0.42%

Absolutne wartości ~1% mAP są nienaturalnie niskie (wcześniejsze analizy dawały dziesiątki procent). To bardzo mocno wskazuje na problem ewaluacji, a nie realną jakość modeli.

Najczęstsze źródła błędów ewaluacji:
- Niezgodny `category_id` — GT ma np. 1 („tooltip”), a predykcje 0 lub inny ID.
- Format bbox — COCO wymaga `[x, y, width, height]` w pikselach; jeśli podano `[x1, y1, x2, y2]`, IoU ≈ 0 → mAP ≈ 0.
- Rozjazd `file_name`/zbioru — detekcje liczone na innym zestawie obrazów niż GT.
- Skala współrzędnych — wartości znormalizowane zamiast pikseli.

Dodatkowo: błąd w skrypcie (`Path.relative_to` – mieszane ścieżki absolutne/względne) nie psuje mAP, ale warto go poprawić, aby końcowe raporty generowały się bez crasha.

Wniosek: obecne wartości mAP nie nadają się do wnioskowania, „czy DETR jest lepszy od YOLO”. Najpierw trzeba naprawić ewaluację.

---

## Dlaczego (po poprawkach) DETR może wciąż ustępować YOLO
- Vanilla DETR (bez multi‑scale) bywa wrażliwy na małe/średnie obiekty i małą różnorodność danych; YOLO z FPN zwykle radzi sobie lepiej przy niewielkich zbiorach i w trybie RT.
- „Query specialization” (np. dominujący Query 81) → kruchość recallu; YOLO ma bardziej rozproszoną detekcję po siatce.
- Bez rozbudowy danych (więcej wideo/negatywy/anty‑duplikaty) przewaga YOLO jest typowa.

Aby DETR realnie konkurował:
- Rozważyć Deformable DETR / DINO (multi‑scale, denoising)
- Zwiększyć różnorodność danych; dodać negatywy
- Rozważyć TTA / box refinement (koszt FPS)

---

## Checklist naprawcza ewaluacji (priorytet)
1) Spójność kategorii
   - W predykcjach COCO ustaw `category_id` dokładnie tak jak w GT (rekomendacja: 1 i nazwa „tooltip”).
2) Bboxy
   - COCO format `[x, y, width, height]` w pikselach; bez normalizacji; zakresy w granicach obrazu.
3) Zbiór i nazwy plików
   - `images[].file_name` w GT musi odpowiadać dokładnie nazwom obrazów użytych do predykcji.
4) Test wspólny
   - Upewnij się, że ewaluacja YOLO i DETR idzie na tym samym teście (bez przecieku po wideo).
5) Poprawka skryptu
   - Napraw `relative_to` → użyj `Path.resolve()` lub trzymaj spójnie absolutne/względne ścieżki.

---

## Rekomendacje „co dalej”
- Szybkie sanity‑check predykcji (dla YOLO i DETR):
  - Otwórz 10 losowych obrazów, narysuj GT i pred, zweryfikuj wzrokowo IoU > 0.5
  - Sprawdź `category_id` i format bbox w plikach predykcji
- Powtórz ewaluację po poprawkach i dopiero wtedy porównaj mAP/AR
- Jeśli DETR nadal odstaje w mAP@[.5:.95] i AR@100 → rozważyć Deformable/DINO + rozbudowę danych
- Produkcyjnie (RT, ograniczenia pamięci): YOLO pozostaje bezpiecznym wyborem

---

## Ścieżki
- Benchmarks: `YOLO_DETR_Benchmarks\\Benchmarks\\2025-10-31_18-30-53_CaDTD_YOLO_vs_DETR`
- Skrypt treningu: `YOLO_DETR_Benchmarks\\scripts\\detr_train_optimized.py`
- SLURM (hooper): `YOLO_DETR_Benchmarks\\scripts\\run_detr_4gpu_hopper_resume140.slurm`
- Checkpointy (EDEN): `/mnt/evafs/faculty/home/bpiotrowski/DETR/ckpt_ddp_2gpu_dgx1_500ep/`

---

**Podsumowanie:** obecny stan treningu DETR na hooperze jest stabilny (AMP OFF), a pipeline checkpointów jest bezpieczny. Benchmarki wskazują na problem ewaluacyjny (mAP rzędu ~1%). Najpierw poprawić mapping kategorii, format bbox i zgodność zbioru; następnie powtórzyć porównanie YOLO vs DETR i rozważyć warianty multi‑scale dla DETR.
