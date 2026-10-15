# CIFAR-10 (2026-10-13)

Drugi zbiór do eksperymentu `experiments/deit_cifar10_20261013` (DeiT-Tiny od zera, analiza laserów/pułapek/koloidów jak bieg s47).

- `train-00000-of-00001.parquet`, `test-00000-of-00001.parquet` — HuggingFace `uoft-cs/cifar10` (plain_text), PNG w parquet.
- `cifar10_uint8.npz` — z `experiments/deit_cifar10_20261013/kod/przygotuj_cifar10.py`: X_train (50000, 32, 32, 3) uint8, y_train,
  X_test (10000, 32, 32, 3), y_test, klasy (airplane … truck). SHA-256 w `cifar10_uint8.json`. Ta sama kopia jest na Edenie
  w `~/smallvit_runs/2026-10-13_deit_cifar10/dane/`.
- `cifar-10-python.tar.gz` — oryginał z https://www.cs.toronto.edu/~kriz/ (serwer bardzo wolny; jeśli plik jest niepełny, nie używać;
  pełny ma 170 498 071 bajtów, MD5 c58f30108f718f92721af3b95e74349a).

**Sprawdzenie (2026-10-13).** Oryginał pobrany w całości (MD5 zgodny z oficjalnym). Porównanie odcisków MD5 każdego obrazu:
train 50 000/50 000 i test 10 000/10 000 obrazów z tą samą etykietą co w npz z HuggingFace. Test ma tę samą kolejność co oryginał,
train ma inną kolejność obrazów (te same obrazy). Biegi na Edenie używają npz, więc liczą się na pełnym, oryginalnym CIFAR-10.
