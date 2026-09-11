# Uruchomienie szybkich badań lokalnie i na Edenie

## Zakres lokalny

Lokalnie sprawdzamy kontrakty, deterministyczność, eksport COCO i komendy
treningowe. Nie interpretujemy dry-run jako wyniku modelu. Obecny lokalny
checkout Cataract-LMM nie zawiera kompletu obrazów ani archiwów trackingu, więc
pełny preflight danych i trening trzeba wykonać po ich dostarczeniu.

```powershell
py -3.11 -m compileall -q SmallViTAblation
py -3.11 -m pytest -q SmallViTAblation/pipeline/tests/test_active_curation.py
py -3.11 -m pytest -q SmallViTAblation/pipeline/eval/tests/test_evaluation_contract.py
py -3.11 SmallViTAblation/pipeline/train/tests/test_dry_run.py
py -3.11 SmallViTAblation/eden/tests/test_queue.py
```

Benchmark syntetyczny sprawdza, czy kontrast historii rozróżnia proces
Markowowski od procesu z ukrytą pamięcią. Nie dowodzi fizycznej interpretacji
latentu.

```powershell
py -3.11 -m SmallViTAblation.pipeline.latent_dynamics.synthetic_benchmark `
  --out SmallViTAblation/results/synthetic
```

## Przygotowanie Cataract-LMM

Tracking i instance segmentation prowadź jako osobne eksperymenty. Tracking
ma `cystotome_bbox` dla instrumentu oraz punkt `cystotome_tip`; pusta ramka w
CSV nie jest automatycznie prawdziwym negatywem.

```powershell
py -3.11 -m SmallViTAblation.pipeline.curation.run `
  --dataset-root Dataset/Cataract-LMM `
  --subset tracking --profile all --budget 500 `
  --candidate-labels all --no-export `
  --out SmallViTAblation/curated/tracking_s42_b500
```

Ten krok tworzy listę do anotacji i diagnostykę. Po sprawdzeniu pustych etykiet
uruchom eksport bez `--no-export`. Stały `split_manifest.json` należy następnie
przekazywać przez `--split-map` wszystkim kolejnym przebiegom.

## Szybka macierz statyczna

Każda strategia musi dostać ten sam split, budżet i seed. Warianty są
kumulatywne. Wspólny zbiór początkowy `L0` można wygenerować raz i przekazywać
pozostałym strategiom przez `--initial-selection`.

```powershell
py -3.11 -m SmallViTAblation.pipeline.active_learning.simulate `
  --pool SmallViTAblation/curated/tracking_s42_b500/pool_manifest.jsonl `
  --embeddings SmallViTAblation/cache/train_features.npz `
  --strategy coverage --budgets 500 1500 2500 --seed 42 `
  --out SmallViTAblation/variants/active

py -3.11 SmallViTAblation/tools/build_active_manifest.py `
  --variants-dir SmallViTAblation/variants/active `
  --profile pilot --out SmallViTAblation/runs/pilot.jsonl
```

Obok kolejki powstaje `pilot.jsonl.lock.json`. Zawiera hashe profilu,
protokołu, wariantów i stan Git. Jeżeli `small_vit_ablation_dirty` jest prawdą,
nie uruchamiaj kosztownej macierzy przed commitem.

Profil blokuje też rozdzielczość konkretnego modelu: smoke i pilot YOLOv8n
używają 384, confirm YOLOv8n 512, RT-DETRv2-S 640, a RF-DETR-Nano 384 pikseli.
Porównuj selektory przy tym samym modelu i profilu. Porównanie między
architekturami służy jedynie do sprawdzenia, czy efekt ma ten sam kierunek.
Manifest każdego biegu zapisuje rzeczywistą rozdzielczość i hash pliku profili.

## Rzeczywista runda active learning

`simulate.py` wykonuje wybór na podstawie przekazanego stanu. Aby eksperyment
był aktywny, po każdej rundzie trzeba ponownie wytrenować model i przeliczyć
predykcje, uncertainty oraz sygnał zadaniowy. Praktyczna runda wygląda tak:

1. Wytrenuj model na aktualnym wariancie `L_r`.
2. Uruchom inferencję aktualnego checkpointu na całej puli `U_train` i zapisz
   standardowe predykcje COCO.
3. Policz aktualne uncertainty.
4. Wyprowadź latent tego samego checkpointu albo jawnie zamrożonego enkodera i
   policz diagnostykę GLE.
5. Połącz score po `frame_id` i wybierz tylko następną porcję, podając bieżący
   wariant jako `--initial-selection`.

```powershell
py -3.11 -m SmallViTAblation.pipeline.active_learning.uncertainty `
  --coco pool_train.coco.json --predictions predictions_pool_r0.json `
  --out scores/uncertainty_r0.jsonl

py -3.11 -m SmallViTAblation.pipeline.latent_dynamics.diagnose `
  --pool pool_manifest.jsonl --features latent_r0.npz `
  --out scores/gle_r0 --seed 42

py -3.11 -m SmallViTAblation.pipeline.active_learning.merge_scores `
  --inputs scores/uncertainty_r0.jsonl scores/gle_r0/frame_scores.jsonl `
  --out scores/combined_r0.jsonl

py -3.11 -m SmallViTAblation.pipeline.active_learning.simulate `
  --pool pool_manifest.jsonl --embeddings latent_r0.npz `
  --scores scores/combined_r0.jsonl `
  --initial-selection variants/L0.json `
  --strategy coverage_history --budgets 500 1000 --seed 42 `
  --out variants/round_1
```

`combined_r0.jsonl` łączy pola po `frame_id`; nie może zawierać wyniku z val,
testu ani danych z przyszłej rundy. Powtórzenie jednego statycznego score'u dla
wszystkich budżetów jest kontrolą frozen-ranking, a nie pełnym active learning.

## Kolejka Eden

Manifest generuj na Edenie po rozpakowaniu danych, ponieważ zawiera absolutne
ścieżki do plików wariantów. Następnie inicjalizuj kolejkę i uruchom łańcuch.

```bash
python3.11 tools/build_active_manifest.py \
  --variants-dir "$PROJECT_DIR/variants/active" \
  --profile smoke --out "$PROJECT_DIR/runs/smoke.jsonl"

python3.11 eden/queue.py init "$PROJECT_DIR/eden/qdir_smoke" \
  --queue-file "$PROJECT_DIR/runs/smoke.jsonl"

QUEUE_DIR="$PROJECT_DIR/eden/qdir_smoke" \
DATA_ARCHIVE="$PROJECT_DIR/data/ablation_data.tar" \
RESULTS_DIR="$PROJECT_DIR/results/smoke" \
bash eden/submit_chain.sh 1
```

Po smoke uruchom `pilot`. Profil `confirm` ma sześć sparowanych seedów. Profil
`external_once` wymaga jawnego `--strategies` ze zwycięzcą zamrożonym przed
otwarciem zbioru zewnętrznego.

## Ewaluacja i niepewność

Najpierw wybierz próg operacyjny na walidacji i policz test tym samym progiem.
Agregacja seedów nie zastępuje niepewności między operacjami. Dla ustalonych
checkpointów wykonaj sparowany bootstrap grupowy na dokładnie tych samych
losowaniach operacji dla każdej metody.

```bash
python3.11 -m SmallViTAblation.pipeline.eval.aggregate \
  --results results/confirm --split test --metric map_50_95 \
  --baseline uniform_time --out analysis/confirm

python3.11 -m SmallViTAblation.pipeline.eval.group_bootstrap \
  --annotations annotations_test.json \
  --predictions uniform_time=pred_uniform.json proposed=pred_proposed.json \
  --baseline uniform_time --iterations 1000 --seed 20260910 \
  --out analysis/confirm/group_bootstrap.json
```

Przy małej liczbie operacji pokaż także wynik każdej operacji i analizę
leave-one-operation-out. Sześć seedów opisuje zmienność procedury treningowej;
bootstrap operacji opisuje niepewność kohorty testowej. Nie należy ich łączyć
w jedną liczbę bez wcześniej zdefiniowanego modelu hierarchicznego.
