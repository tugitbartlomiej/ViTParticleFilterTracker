# Kontynuacja: dwa A100 i zachowane awarie

Użytkownik autoryzował wznowienie rozszerzonego NNDMD1500 na dwóch pełnych A100 w jednym DGX. Globalny batch nadal 32 obrazy, tail 28; wszystkie 600 cząsteczek i pełne 768 współrzędnych. 4 GPU przestały być warunkiem; Hopper-2/MIG i Pascal zostały wykluczone. Nie uruchomiono lokalnego uczenia. Jeden właściciel Slurm submission: plan_poprawek.

Zachowano stare joby: 1819498 FAILED mkdir `/raid`, zero kroków; 1819551 FAILED duplicate `origin` w telemetrii po jednym rzeczywistym kroku. Lokalny folder `failure_1819551` ma wykonany snapshot, logi, GPU_TRAINING_SMOKE i provenance. Naprawa telemetrii nie zmieniła modelu ani strat.

1819567 COMPLETED: infrastruktura NCCL2 i serial real step. 1819570 COMPLETED na dgx-4: właściwy model 1 vs 2 A100 dla synthetic batch32 i tail28/full600/full768/H10. Zgodność gradientów/Adam/loss/prognoz/telemetrii/checkpointu PASS. Wagi maxabs7.53e-7; gradient3.73e-9 /1.02e-8. Dwie wczesne paczki CUDA-sync szybciej na2GPU (1.118vs1.480s,0.726vs1.052s), ale bez ETA pełnego uczenia/ekstrakcji.

Parallel reader: wszystkie ranki `ensure_window`, rank0 single-header/metadata/hash, disjoint obrazowe shardwrites mmap, cooperative error gates i atomic complete. Gloo tiny mmap proof PASS. Runtime remote CPU suite15PASS/1FAIL zidentyfikował brak mocka stream-capture w CPUfixture udającej CUDA; sam test poprawiony, targeted remote PASS. Produkcyjny model97f01e..., driver52db0c..., configc748... zachowane. Final18SHA remote PASS, audit Astra PASS.

Pełny nowy job **1819574** zgłoszony i potwierdzony RUNNING dgx-4. Remote: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu`; logi `slurm-1819574.out/.err`. Scratch rzeczywiście utworzony `/dev/shm/bpiotrowski_nndmd_expanded_1819574`, rolling13, grant128G. Launcher guard final-core synthetic600, parallel source0..10, driver actual dataset1vs2 batch32/tail28; resetseed przed właściwym fitem. Kroki proof liczone osobno. Na etapie tego zapisu właściwe kroki fitu i końcowe metryki nie były jeszcze potwierdzone. Root monitor ma przejąć dalsze śledzenie; nie submitować duplikatu.

Fit/timeprotocol pozostaje inner0..80, val80→90, max10/patience2, wybranaepoka refit0..90, test90→100 poFREEZE. Exact1500 source frames, equalimages/equaltokens; checkpointy rzeczywistego ViT z AdamWmomentami co100 i causal hold-last aux. To future-time na tych samych znanych obrazach, exploratory. Nie jest to niezależny heldout image ani gwarancja85procent.

Późniejsze potwierdzenie startu: parallel source preflight11cp PASS w68.54s, sourcecomparisonmaxabs2.15e-6. Actualdataset1vs2 guard PASS (maxgrad7.45e-9/9.78e-9, maxweights4.10e-7), modele odrzucone/resetseed. GPU_TRAINING_SMOKE potwierdził3 właściwe globalne kroki, trzeci0.645s. Po synchronizacji lokalny status już133actual_training_steps. Końcowych metryk jeszcze nie ma. Status/TRAINING/SMOKE/EXECUTION/NORMALIZATION/DATASET_PREFLIGHT pobrano lokalnie, monitorroot przejmuje1819574.
