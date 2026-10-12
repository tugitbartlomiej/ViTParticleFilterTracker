# Wyniki replay 900k co 5

Lokalny katalog zachowuje małe źródła provenance. Duże wyniki pozostają na Edenie pod `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/results`.

PCA3 jest wyłącznie wizualizacyjna. Panel 64 na 16 przechowuje pełne stany i sterowania 768D, a pliki per-image zawierają także globalne skalarne deskryptory ruchu obliczone w 768D dla wszystkich 900 000 cząsteczek.

Te same pliki zawierają `analysis_positions_pca32` we wspólnej bazie dopasowanej wyłącznie na filmach train oraz normy residuum w pełnych 768 wymiarach.

Końcowa bramka eksportu main na Edenie: `results/visualization_dense/manifest.json`.
