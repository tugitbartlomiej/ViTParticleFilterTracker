# Interwencja w 128 kanałów ViT

Plan: [PLAN.md](PLAN.md). Model zamrożony k0500, 400 oryginalnych obrazów, kanały dobrane we wcześniejszym eksperymencie. Trzy promienie zmiany: 0.25, 0.5, 1.0 SD. Dwa wyniki: optymalizacja każdego obrazu z jego etykietą i wspólna korekta z oceną na odłożonych filmach.

Lokalnie: `F:/Python/python.exe experiments/vit_optymalizacja_latentu128/test_core.py`, następnie `prepare.py`. Eden: `sbatch bieg.slurm smoke`, po pozytywnym smoke tablica `--array=0-2%3`. Wszystkie wyniki i logi w osobnym katalogu `~/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128`.

`TARGETS_ORACLE.npz`: indeksy obrazów, channels0, bazowe i docelowe wartości 128 kanałów wszystkich 600 tokenów oraz przyrosty. Reszta pełnego latentu jest niezmieniona i odtwarzalna z checkpointu/obrazu. `SHARED.npz`: wspólne przesunięcie. `SELECTION.json`: wybór epoki na walidacji. `RESULT.json`: metryki, kontrola losowa, ograniczenia. `RECORDS.json`: metryki każdego obrazu. `PREFLIGHT.json`: zgodność pełnej sieci i końcówki oraz hash wejść.

Optymalizacja wartości aktywacji nie dowodzi osiągalności przez zdjęcie. Oracle zna etykietę optymalizowanego obrazu; jego wynik nie jest jakością nowego detektora na niewidzianych danych. Transfer dotyczy tylko wspólnej korekty. Nie wykonujemy generowania obrazów ani uczenia wag ViT.
