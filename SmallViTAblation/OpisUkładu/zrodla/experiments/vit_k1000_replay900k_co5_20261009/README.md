# vit_k1000_replay900k_co5_20261009

Izolowany replay historycznego oracle K1000 dla dokładnych 1500 klatek. Produkuje prawdziwe stany wszystkich 900 000 cząsteczek co 5 aktualizacji, pełny panel sterowań 768D oraz strumieniowe deskryptory ruchu pełnej populacji liczone w 768D.

W tym samym pojedynczym replay zapisuje wspólne współrzędne PCA32 dla całej populacji. Baza jest zamrożona przed smoke i dopasowana wyłącznie na historycznych stanach filmów train; PCA3 pozostaje wyłącznie wizualizacyjna.

Kod nie modyfikuje starego oracle, wspólnej korekty, viewer'a ani rodziny analizy PyDMD/deeptime. Publikacja manifestów jest atomowa i następuje po pełnym merge.

Zdalny root: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009`.

Docelowe manifesty:

- `results/merged/VISUALIZATION_MANIFEST.json`;
- `results/merged/TEMPORAL_PANEL_MANIFEST.json`;
- `results/merged/REPLAY_VALIDATION.json`.
- `results/merged/GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json`;
- `results/merged/ANALYTICAL_PCA32_MANIFEST.json`;
- `results/visualization_dense/manifest.json` — końcowa bramka 101 buforów main.
