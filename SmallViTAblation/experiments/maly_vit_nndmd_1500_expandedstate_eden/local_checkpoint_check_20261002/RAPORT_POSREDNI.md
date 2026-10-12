# Lokalny test POŚREDNIEGO checkpointu — jeden obraz

Snapshot 85099613250068ba97637a3cd933b89f4f79b381083c7818019950d8b4fff5a5, phase final_refit, epoch 1, ostatni ukończony origin 34, total optimizer steps 11656. To nieukończony refit, diagnostyka bez strojenia.

Obraz exact1500 index 23, frame cataract_lmm/tracking/TR_0003_S1_P03/00004756, oldcache index 29. Wszystkie 600 cząsteczek i 768 współrzędnych. Wejście cp88..90, samodzielna prognoza cp91..100. Jeden checkpoint odpowiada 10 aktualizacjom źródłowego ViT. J128 odczytano po pełnym decoderze.

| H aktualizacji | RMSE full768 model | persistence | RMSE J128 model | persistence | znak J128 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.19047136 | 0.08160770 | 0.19936849 | 0.08011194 | 50.367% |
| 20 | 0.19876690 | 0.09951902 | 0.20786623 | 0.09542642 | 51.038% |
| 30 | 0.21329689 | 0.11999351 | 0.20387008 | 0.09799319 | 53.796% |
| 40 | 0.19187507 | 0.09870870 | 0.19239528 | 0.09577992 | 54.273% |
| 50 | 0.19974363 | 0.12263554 | 0.20231294 | 0.11231893 | 51.956% |
| 60 | 0.18252674 | 0.09588195 | 0.18447722 | 0.09424135 | 53.902% |
| 70 | 0.19148580 | 0.10355473 | 0.19663146 | 0.09342619 | 52.005% |
| 80 | 0.21120052 | 0.13495659 | 0.20905643 | 0.12559836 | 52.604% |
| 90 | 0.21303647 | 0.12989658 | 0.19754128 | 0.11750775 | 54.210% |
| 100 | 0.22598523 | 0.13371634 | 0.21209094 | 0.12003204 | 51.831% |

RMSE to pierwiastek z MSE agregowanego równomiernie po wszystkich cząsteczkach i kanałach, w surowych jednostkach latentu. Historyczna średnia i kontynuacja ostatniego przyrostu są zapisane w METRICS.json. Znak uwzględnia wszystkie współrzędne, sign(0)=0.

Strict load PASS, wszystkie tensory zgodne bitowo ze snapshotem i niezmienione po inferencji. Przyszłe auxiliary zatrute NaN nie zmieniły prognozy. Niezależny encode-once rollout phi_h=phi_(h-1) A^T jest identyczny. A finite; ||A−I||F/||I||F=0.080254681. Żadnych aktualizacji optymalizatora ani przyszłych wejść.

Runtime lokalny 2.1.2+cu121, PyKoopman 1.2.1, cuda; Eden używa Torch2.9.1+cu128. Różnice kerneli FP32 mogą powodować drobne różnice numeryczne. Prawdziwe cele pochodzą z wcześniej zweryfikowanego lokalnego cache, identyczne ID obrazu/piksele i hashe źródłowych checkpointów cp88..100 potwierdzono. SHA całego cache pochodzi z jego metadanych; nie przeliczano ponownie 6GB. Wynik dotyczy wyłącznie jednego obrazu, nie wszystkich1500 ani finalnego checkpointu.
