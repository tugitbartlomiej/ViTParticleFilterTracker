# Plan implementacji warstwy aktywnej kuracji

## Kontrakty

- `pool_manifest.jsonl`: jeden rekord na klatkę; wymagane `frame_id`, `split`,
  `group_id`, `video_id`, `frame_index`. Selekcja nie odczytuje adnotacji.
- cache latentów `.npz`: `frame_ids` oraz macierz `features` o identycznej
  kolejności.
- score `.jsonl`: `frame_id` oraz dowolne z `uncertainty`, `residual_energy`,
  `residual_mahalanobis`, `history_gain`.
- wariant wynikowy `.json`: dokładna, uporządkowana lista `frame_id`, budżet,
  seed, strategia, rodzice i hashe wejść.
- predykcje detektora: standardowy COCO JSON; wszystkie adaptery kończą na tym
  samym evaluatorze.

## Moduły

1. `pipeline/active_learning/acquisition.py` implementuje deterministyczny,
   dokładny budżet i wspólny interfejs sześciu strategii.
2. `pipeline/active_learning/simulate.py` buduje kumulatywne rundy od wspólnego
   L0 i zapisuje pełną linię pochodzenia.
3. `pipeline/active_learning/uncertainty.py` wyprowadza prosty, jawnie opisany
   score z predykcji COCO. Nie nazywa go BALD ani epistemic uncertainty.
4. `pipeline/latent_dynamics/gle.py` estymuje cross-fitowane modele Markowa i
   historii oraz mody kowariancji reszt.
5. `pipeline/latent_dynamics/diagnose.py` zapisuje score per przewidywana klatka.
6. `pipeline/latent_dynamics/synthetic_benchmark.py` sprawdza identyfikowalność
   diagnostyki na pięciu kontrolowanych generatorach.
7. `pipeline/transfer/train_transfer.py`, `pipeline/train/train_rfdetr.py` oraz
   `eden/runner.py` tworzą dispatcher adapterów YOLOv8n, RT-DETRv2-S i
   RF-DETR-Nano. Tryb dry-run nie wymaga GPU ani zewnętrznych repo.
8. `tools/build_active_manifest.py` generuje ograniczoną macierz smoke/pilot/
   confirm i odrzuca niezdefiniowane warianty.

## Polecenia lokalne

```powershell
py -3.11 -m SmallViTAblation.pipeline.latent_dynamics.synthetic_benchmark --out SmallViTAblation/results/synthetic
py -3.11 -m SmallViTAblation.pipeline.active_learning.simulate --help
py -3.11 SmallViTAblation/tools/build_active_manifest.py --variants-dir SmallViTAblation/variants/active --profile smoke --out SmallViTAblation/runs/smoke.jsonl
py -3.11 SmallViTAblation/pipeline/transfer/train_transfer.py --help
py -3.11 SmallViTAblation/pipeline/train/train_rfdetr.py --help
```

Na komputerze bez obrazów wykonujemy walidację konfiguracji, fixture'y i
dry-run. Pełny trening uruchamiamy na Edenie dopiero po weryfikacji liczności,
hasha splitu i dostępności wszystkich plików wskazanych przez manifest.
Generator macierzy zapisuje obok kolejki plik `*.lock.json` z hashami protokołu,
profilu i wariantów. Domyślnie paruje ziarno selekcji z tym samym ziarnem treningu,
zamiast tworzyć ich kosztowny iloczyn kartezjański. Manifest należy generować po
zmaterializowaniu danych w docelowym układzie katalogów Edena, ponieważ zawiera
absolutne ścieżki do wariantów.
