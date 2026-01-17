# Raport: wpływ selekcji 20k (po epoce 170) na trening YOLO i DETR

## Cel
Odpowiedzieć na pytanie: **co zmieniło się po epoce 170** (przejście z puli ~90k na **wyselekcjonowane 20k**) oraz jaki był wpływ na dynamikę treningu **YOLO** i **DETR** (szczególnie DETR).

## Źródła (artefakty z repo)
- Selekcja 20k (raport i uzasadnienia):
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset1\selection_reasons.txt`
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset1\selection_reasons.json`
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset1\merged_20k_annotations.json`
- YOLO metryki:
  - Phase1 (~90k): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\exp\results.csv`
  - Fine-tune 20k (172–200): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\exp\results.csv`
- DETR fine-tune log + analiza sesji:
  - Log: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\detr_20k_finetune_pascal_1454332.log`
  - Raport sesji: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_sessions\2025-12-13_20kDataset_small_LR\detr_training_session_report.md`
- Analizy wag/checkpointów (wykresy + metryki “weight-space”):
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\compare_new\detr_vs_yolo_comparison.md`
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_training_analysis.md`

## Protokół faz (granica interpretacji)
- **Faza 1**: epoch `<= 170`, trening na dużej puli ~90k (raport deep-compare podaje `~91,260` obrazów).
- **Faza 2**: epoch `> 170`, trening/fine-tune na **20,000** obrazów (wyselekcjonowany subset).
- W praktyce oba fine-tune’y startują od **okolic epoch 172** (YOLO i DETR).

## Co robi selekcja 20k (jakie cechy zmienia w danych)
Z `selection_reasons.txt/json`:
- Wejście do selekcji: `~90,389` obrazów (po walidacji/filtracji względem surowej puli).
- Wyjście: dokładnie `20,000` obrazów, metoda `cluster`, strategia `centroid`, `20,000` klastrów.
- Statystyki “trudności”/złożoności praktycznie **bez przesunięcia średniej**:
  - EL2N mean (all → selected): `0.1361 → 0.1339`
  - SAM mean (all → selected): `0.4189 → 0.4163`
Wniosek: to nie jest “hard mining” (wybieranie najtrudniejszych), tylko **różnorodność + redukcja redundancji**.

Dodatkowo (z COCO jsonów):
- Zbiór 20k ma **dokładnie 1 bbox na obraz** (20k obrazów i 20k anotacji), podczas gdy w puli ~90k zdarzają się obrazy z 2–4 bbox.
- Rozkład rozmiarów bbox (area/width/height) jest bardzo podobny do dużej puli, ale bez najbardziej ekstremalnych outlierów (np. skrajnie małe / bardzo wydłużone).

## YOLO: co stało się po 170
### Metryki (mAP/PR)
Z `results.csv`:
- Phase1 @170 (na dużej puli): `mAP50-95 = 0.94078`
- Fine-tune 20k (172–200): peak `mAP50-95 = 0.96332` @189  
  - zysk vs @170: `+0.02254`
  - vs najlepszy wynik na phase1 do @200 (`0.94970`): `+0.01362`

Interpretacja:
- Fine-tune na 20k poprawia głównie `mAP50-95` (czyli “trudniejszą” część metryki zależną od IoU), przy bardzo wysokich i stabilnych precision/recall.

### Dynamika wag (z deep-compare)
Z `deep_training_analysis.md`:
- Median krok checkpoint→checkpoint (dW/||w||) spada po przejściu na 20k ~`4.84×` (model przechodzi w tryb “dostrajania”).

## DETR: co stało się po 170 (kluczowe)
### 1) To był osobny “fine-tune run” z resetem LR i schedulera
Z logu i raportu sesji:
- Detected resume: załadowany `final_checkpoint.pth` z epoch `172`.
- Reset LR: `1e-4 → 5e-5` (backbone: `5e-6`) + restart cosine schedulera na pozostałe `128` epok.
- Dane: `20,000` obrazów, 4×P100, batch `4/GPU` (efektywnie `16`).

Wniosek: po 170 nie było “ciągłej kontynuacji” identycznym reżimem — weszło **mniejsze LR + nowy harmonogram**, czyli klasyczny fine-tuning.

### 2) Zmieniła się skala aktualizacji wag (dużo mniejsze kroki)
Z `deep_training_analysis.md`:
- DETR: median dW/||w|| spada po przejściu na 20k ~`5.02×`.
- Drift 170→200 w przestrzeni wag: `||w200 - w170|| / ||w170|| = 0.0337` (cos `0.9994`).  
To oznacza, że faza 2 jest **lokalnym dostrajaniem**, a nie “przestawieniem” reprezentacji.

### 3) Najmocniej adaptuje się głowica klasyfikacji
Z `deep_training_analysis.md`:
- Największy drift w DETR podczas 170→200 ma grupa `cls_head` (dominująca zmiana względem encoder/decoder/backbone).

Interpretacja:
- Selekcja 20k + mniejsze LR powodują, że model “zgrywa” granice decyzyjne i kalibrację klasy/No‑Object przede wszystkim w głowicy, przy małych zmianach w backbone/transformerze.

### 4) Trend lossów po przełączeniu na 20k
Z checkpoint analizy i TensorBoard (raport sesji):
- W oknie 170–200 loss (z checkpointów) schodzi do okolic `~0.12` (np. najlepszy `0.1205` @195) i dalej poprawia się w długim biegu (best val `~0.0415` @291).
- W logu widać “twardszy” start (wyższe straty na pierwszych epokach po wznowieniu), co jest typowe dla przejścia na nową dystrybucję danych + reset LR/schedulera.

## Odpowiedź na pytanie „co się stało po 170 epoce?”
Po epoce 170 nastąpiło **przełączenie danych na wyselekcjonowane 20k** i wejście w tryb **fine-tune**:
- **YOLO**: zauważalny wzrost `mAP50-95` (szczególnie w okolicach epoki ~189), przy mniejszych krokach wag (dostrajanie).
- **DETR (najważniejsze)**: uruchomiono osobną sesję fine-tune z **resume ~E172 + reset LR + restart cosine**, a dynamika wag jednoznacznie wskazuje na **lokalną adaptację** — głównie w `cls_head`, bez dużego “driftu” całego modelu.

## Wykresy, które najlepiej to pokazują
- Deep porównanie: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_training_analysis.md`
- YOLO metryki (z zaznaczoną granicą faz): `...\deep_compare_new\deep_yolo_metrics.png`
- DETR fine-tune log (loss/grad/time): `...\deep_compare_new\deep_detr_finetune_log.png`
- Drift wag (170 jako baseline): `...\deep_compare_new\deep_weight_drift.png`
- Gdzie zmieniają się wagi (warstwy/grupy): `...\deep_compare_new\deep_group_drift.png`

