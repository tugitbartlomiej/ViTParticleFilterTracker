# Lokalne przyrosty NNDMD na Hopper

Osobny eksperyment: exact1500, wszystkie600 cząsteczek, native768, expandedinput2544, learned768 i wspólny A768. PLAN_MODELU.md definiuje zatwierdzony zakres i loss. Właściciel danych/Slurm: plan_poprawek; model/driver/testy: pykoopman_obliczenia; niezależny audyt: Astra. Wcześniejsze dane i wyniki pozostają w poprzednich katalogach.

CPU: `python -m unittest test_expanded test_driver test_distributed -v`.

GPU proof: perrank launcher lub `torchrun --standalone --nproc_per_node=WORLD gpu_smoke.py --tokens600 --output GPU_PROOF`; deterministic synthetic fixture, optimizersteps i pełna ścieżka telemetry/checkpoint, zero actualFITsteps. Gdy Gloo: `EXPANDED_DISTRIBUTED_BACKEND=gloo`, transport tensorów przez CPU jawnie. Real source preflight następnie sprawdza actualdataset32/28 na origin50 i resetuje wszystkie modele/seedy przed fit.

Clusterfit po pozytywnym audycie/GPUguard: `python train.py --config CONFIG.json` w tym samym wieloprocesowym launcherze. Rank0 tworzy pliki; wszyscy uczestniczą w collective reader i gradientSUM. Pipeline inner2lrG→selection→freshfinalrefit→freeze→test. Resume automatycznie z własnych originowych checkpointów tego samego config/phase. Brak lokalnych treningówGPU.

METRICS_TEST.json i RAPORT.md powstają dopiero po rzeczywistym kompletnym teście; checkpoint/status/validation po drodze nie oznaczają zakończenia.


Aktualny dowód runtime: [PROOF_GPU_RUNTIME.json](PROOF_GPU_RUNTIME.json). Izolowany NCCL 2.32.3 oraz przypisanie per-rank UUID przeszły guard1819911; realny syntetyczny model1vs11/all600/batch32+28 przeszedł job1819912. Finalny source snapshot: [FINAL_SOURCE.sha256](FINAL_SOURCE.sha256), [SOURCE_HASHES.json](SOURCE_HASHES.json). Plan launchera: [PLAN_EDEN.md](PLAN_EDEN.md). Przed rzeczywistym fitem1500 obowiązkowe są reader/sourcepreflight oraz modelguard na rzeczywistym oknie50, z resetem modelu po testach. Wyniki prognozy nowego modelu nie są jeszcze znane.
