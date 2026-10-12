# Wdrożenie nowego modelu lokalnych przyrostów: Hopper-2

2026-10-02, Europe/Warsaw. Kontynuacja rozmowy: user anulował dawny1819574 i autoryzował nowy late-history/residual NNDMD oraz próbę11 urządzeń na Hopper-2. Nie jest to11 pełnych GPU:4 pełneH200+4MIG4g.71gb+3MIG3g.71gb. Nie użyto obcego zadania1819871.

Infra1819903: FAILED widocznośćCUDA8 zamiast11. Infra1819908: perUUID poprawne, staryNCCL2.27.5 DuplicateGPU dlaMIGtej samej karty. Izolowany job-onlyNCCL2.32.3+driverCUDA13.2+MNNVL0: infra1819911 PASS11/11/SUM66. Model1819912: PASS1vs11 na syntheticbatch32+28/all600, realAdam/telemetry/save; zero fit1500.

Jedyny pełny job1819918 przyjęty i RUNNING. Scratch własny `/dev/shm/bpiotrowski_nndmd_expanded_1819918`, cachefull101, grant384GB. Realmkdir/writeprobe; /raid niewritable pominięty. Runtime/sourcehashe22/22 i audytPASS. Powtórka CPU Gloo na wolnym loginie przerwana wyłącznie we własnychPID według decyzji roota: INTERRUPTED, niePASS;19 lokalnychCPU i exactGPUproof pozostały niezmienne.

Sourceprepare exact1500/101checkpointów PASS29.77s. Sourcepreflight11rank/all600×768 PASS48.97s, sourcecomparisonmaxabs2.62e-6. ActualdatasetGPUpreflight1vs11/32+28 PASS,2 kroki oddzielne. Model/seed zresetowane i realnyfit rozpoczęty: lokalny snapshot19:09:31UTC/21:09:31Warsaw,223 actualtrainingsteps,inner_LRG_0epoch1origin54. Warmedmedianstep0.162s; to nie ETA całości ani wynik walidacji.

Nowy eksperyment `SmallViTAblation/experiments/maly_vit_nndmd_lokalne_przyrosty_hopper`. Małe dowody w run1819918/, FINAL_SOURCE.sha256,SOURCE_HASHES.json,PROOF_GPU_RUNTIME.json,submit_status.json,PLAN_EDEN i WYNIKI. Remote `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper`, logslurm-1819918.out/.err. Monitor użytkownika pozostajeOFF; nie reaktywowano go. Nie składać drugiego pełnego joba, nie wznawiać anulowanego1819574. Finalny wynik prognozy nieznany.
