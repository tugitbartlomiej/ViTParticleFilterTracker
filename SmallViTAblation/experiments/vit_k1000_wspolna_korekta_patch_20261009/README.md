# Wspólna korekta per patch K1000

Eksperyment uczy bezpośrednio jedną macierz `600x768` na naturalnych latentach
dokładnych 1500 bazowych obrazów. Nie korzysta z oracle delta jako celu ani
inicjalizacji i nie zmienia wag ViT.

Kod, SLURM i provenance należą wyłącznie do tej rodziny. Wyniki trafiają do
`outputs/vit_k1000_wspolna_korekta_patch_20261009/` oraz odpowiadającego rootu
na Edenie. Aktualny stan jest zapisany w `STATUS.md`.
