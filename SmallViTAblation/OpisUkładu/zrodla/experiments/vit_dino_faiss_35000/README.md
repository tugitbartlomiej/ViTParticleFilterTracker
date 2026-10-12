# DINO i FAISS, pilot 35000 TRAIN

Specyfikacja wspólna z translatorem: DINO_SPEC.json. Forward FP32, normalizacja L2 FP32, 384 CLS; bez max-patch. Oficjalne wagi zweryfikowane przez pobranie i porównanie SHA256. Cache historyczny jest niezgodny i nie jest używany.

Uruchomienie i wznowienie:

```powershell
& F:/Python/python.exe F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_dino_faiss_35000/worker.py
```

Wyszukiwanie PNG/JPG po ukończeniu:

```powershell
& F:/Python/python.exe F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_dino_faiss_35000/QUERY.py obraz.png --topk 32
```

Wyniki w outputs/vit_dino_faiss_35000. PROGRESS.json, INVENTORY.json, COVERAGE.json, manifest.jsonl, CONFIG.json, PRECISION_QA.json, PILOT.json, chunks, features.npy, index.faiss, QA.json, METADATA.json, RAPORT.md i queries. Manifest i statystyki pokrycia powstają przed ekstrakcją. Lock OS zapobiega równoległemu uruchomieniu drugiego workera. Niedokończony chunk jest przeliczany, zakończone i zweryfikowane chunki pozostają.

PLAN.md opisuje kontrole i ograniczenia. To indeks pilotażowy TRAIN, bez nowego treningu, held-out oceny i bez twierdzenia o niezawodności translatora.

## Streaming dla większego translatora

`dane.py`: `train_archives()` wybiera wyłącznie groups[archive.stem] == 'train'. `iter_labeled_train(on_defect=callback)` zwraca pary `(record, PIL_RGB)` po kontroli dekodowania i GT. To ten sam parser i polityka co worker indeksu. `DINO_PREPROCESS` jest dokładnie transformacją indeksu. Import nie uruchamia indeksu ani CUDA. Dołóż katalog tego eksperymentu do sys.path, zanim zaimportujesz dane. Etykiety obejmują dodatnią ramkę narzędzia; same cornea/pupil nie kwalifikują klatki.

Źródło ZIP: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Dataset/Cataract-LMM/3_Object_Tracking/annotations`. Podział: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/feature_cache/eden_static36/data/tracking_static36/split_manifest.json`. Inwentaryzacja: 310257 obrazów TRAIN, 283577 kandydatów z poprawnym GT CSV; dopiero iterator sprawdza dekodowanie każdego kolejnego obrazu. Manifest indeksu jest wybranym pilotażem 35000, nie pełną pulą do streamingu.
