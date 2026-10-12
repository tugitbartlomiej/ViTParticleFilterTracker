# Rozproszone uruchomienie na dwóch A100

Obowiązujący wariant: jeden node DGX, dwa pełne A100, globalny batch 32 obrazów (16/rank), tail 28 (14/rank), wszystkie 1500 obrazów i 600×768 współrzędnych. Model, config, straty i protokół czasowy pozostają takie same. Nie uruchomiono Hopper-2, MIG ani Pascal.

## Zweryfikowane testy

- Job 1819567: NCCL all-reduce i rzeczywisty serial optimizer step, COMPLETED exit 0. To wyłącznie preflight infrastruktury.
- Job 1819570 na dgx-4: właściwy model 1 vs 2 GPU, COMPLETED exit 0. `smoke_2gpu/` zawiera dokładny snapshot, hashe i wyniki obu testów (4 i 600 cząsteczek). Dane syntetyczne; zero kroków właściwego fitu 1500.
- Pełne 600 cząsteczek, batch 32: max różnica gradientu 3.73e-9, wag po Adam 7.53e-7, CUDA synchronized 1 GPU 1.480 s vs 2 GPU 1.118 s. Tail 28: gradient 1.02e-8, wagi 7.53e-7, 1.052 s vs 0.726 s. Dwa wczesne kroki nie stanowią warmed ETA epoki; nie zawierają ekstrakcji źródłowego ViT.
- CPU Gloo reader: rozłączne kompletne 37 wierszy testowego mmap, jeden writer nagłówka, atomowy complete, błędy jednego sharda propagowane do wszystkich ranków. PASS.
- `bash -n` i trzy testy selektora scratch PASS na Edenie.

## Pełny run

Osobny katalog zdalny: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu`. Źródła starego joba nie są nadpisywane. `EXECUTION_FREEZE.sha256` obejmuje kod, dane, plany i launcher. `bieg_eden_2gpu.slurm` wykonuje final-core GPU guard, równoległy source preflight 0..10, a driver przed fitami dodatkowo porównuje 1 vs 2 GPU na rzeczywistych źródłowych latentach batch 32 i tail 28. Modele proof są odrzucane i seed resetowany. Dopiero potem zaczyna się właściwy fit.

Cache wspólny dla node, poza home. Przy tmpfs zawsze rolling13 około 35.94 GB; full101 około 279 GB tylko na zweryfikowanym node-local dysku z odpowiednią pojemnością i prawem realnego zapisu. Rank 0 tworzy nagłówek i metadane, ranki zapisują rozłączne obrazy, complete/SHA są publikowane dopiero po wspólnym sukcesie. SourceViT jest odładowany na CPU przed krokiem NN.

Nieudany job 1819551 zachowano w `../failure_1819551`: źródła wykonane, provenance, pierwszy rzeczywisty krok i log błędu duplicate `origin`. Naprawa telemetrii nie zmieniła metody.
