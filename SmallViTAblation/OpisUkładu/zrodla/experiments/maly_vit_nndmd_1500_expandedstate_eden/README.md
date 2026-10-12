# NNDMD1500 expanded state na Edenie

Aktualny protokół: [PLAN_EDEN.md](PLAN_EDEN.md), bieżące testy i nowy wariant dwóch A100: [run_multigpu/README.md](run_multigpu/README.md). Źródłowe obrazy i checkpointy pozostają nietknięte. Lokalny poprzedni wariant zatrzymany; job 1819498 zakończył się błędem scratchu, a retry 1819551 błędem telemetrii po pierwszym rzeczywistym kroku. Jego logi i wykonane źródła zachowano w [failure_1819551](failure_1819551/FAILURE.json). Test modelu 1 vs 2 A100, job 1819570, zakończył się poprawnie; nie był pełnym fitem 1500.

- Dane: MANIFEST_1500.json, przygotuj_eden.py, dane.py (13-slotowy cache lub full101 na scratch).
- Historyczna telemetria: AUX_HISTORY.json / AUX_HISTORY.npz, wytworzone przez telemetria.py na źródłowych plikach Edenu.
- Provenance: EDEN_PROVENANCE.json; status faktycznego submit: [submit_status.json](submit_status.json).
- Uruchamianie: bieg_eden.slurm; model/train/config dostarcza właściciel NN.

Wyniki nie są jeszcze dostępne. Odtworzenie wymaga istniejącego source1500 archiwum,101checkpointów,11stanówAdamW oraz izolowanego runtime. Nie używamy przyszłych pomiarów jako wejścia prognozy.

Poprawka infrastruktury: [INFRA_FIX_1819498.md](INFRA_FIX_1819498.md). NN i konfiguracja bez zmian.
