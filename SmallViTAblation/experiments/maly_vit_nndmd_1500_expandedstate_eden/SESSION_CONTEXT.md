# Kontynuacja NNDMD1500: Eden, rozszerzony stan

Ostatnie polecenie odwołuje lokalne GPU benchmarki/trening i autoryzuje uruchomienie rozszerzonego stanu na Edenie po audycie. Lokalny benchmark source zakończony, brak local1500training. Wszystkie wcześniejsze artefakty zachowane.

SSH działa, źródłowe dane i101checkpointów istnieją. Pełne AdamW11co100 istnieją i rzeczywiście zawierają exp_avg/exp_avg_sq. AUX_HISTORY101×59 complete. Źródłowe batch-ID/augmentacje/gradienty nie zapisane. Rolling13 lub full101 tylko scratch; home36GB nieprzeznaczone na cache.

Eksperyment: SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden. PLAN_EDEN.md opisuje bieżący protokół. Model owner pykoopman_obliczenia; submission owner plan_poprawek; auditor astra_koopman_wagi. Job jeszcze nie zgłoszony: czekamy code tests i audyt.

Sesja: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-10-02_145632


15:12:03: sbatch accepted1819498,12remote testsPASS, final16SHA PASS. Squeue PENDING Resources; optimizersteps0. Status: submit_status.json.


2026-10-02T15:37:28.823371+02:00: job 1819498 FAILED mkdir /raid permission,0optimizer steps. Infrastructure fixed and3testsPASS+independent rootPASS;18hashesPASS, NN/config/data unchanged. Retry1819551 accepted15:35:55,PENDINGResources,0steps; heartbeat root takesmonitor.
