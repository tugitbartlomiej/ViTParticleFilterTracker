# Dane source1500 i bounded streaming

Stan: MANIFEST_1500.json verified, dokładnie 1500/1500 source RGB i ID. Źródła/cache/checkpointy są read-only. Pełnych 279GB nie zapisujemy. FrozenWindowReader z dane.py dostarcza wszystkie 1500 obrazy × 600 tokenów × 768 kanałów przez 11slotów mmap na dysku. Żadne dane wcześniejszych eksperymentów nie są usuwane.

API: begin_epoch(phase,epoch), ensure_window(origin,width=10), get_batch(image_indices,origin). Wynik(11,N,600,768)float32; origin jest indeksem cp,10 updates/cp. Cache checkpoint complete dopiero po pełnym 1500/finite/SHA, partial status nieprzyjmowany. 11 slotów około 30.4GB na dysku, batch 32 około649MB wRAM, microbatch źródła 4. Source ViT poensure CPUoffload; jedenproces współdzielącyGPU zNN.

Wszystkie 600 tokenów mają równą 1/600 wagę w obrębie obrazu. COCO zawiera tylko bboxXYWH; nie jest segmentacją. Model/protokół NN i run należy do osobnegoagenta. Test końcowy90→100 na tych samychknown1500, nieindependent imageheldout. Porównania score J128/presence/forecast różne.

Odtworzenie manifestu: F:/Python/python.exe przygotuj_manifest.py. Benchmark źródła: F:/Python/python.exe dane.py --benchmark. Pełnycache używany synchronicznie dopiero w uzgodnionym treningu NN; nie uruchamiać równoległego sourceprocessu.
