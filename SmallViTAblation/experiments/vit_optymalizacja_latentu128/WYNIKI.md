# Wyniki pierwszego eksperymentu J128

2026-10-04. Eden: smoke1822385 i trzy ramiona1822386_0/1/2 zakończone COMPLETED, exit0. Raport1822387 także zakończony. Wszystkie artefakty pobrano i sprawdzono SHA256. Checkpoint500; 400 obrazów; promienie0.25/0.5/1.0 pooledSD.

Przy promieniu1.0 wybranym na walidacji AP50 dla85 obrazów części test wzrosło z0.226160 do0.276814 przy indywidualnej optymalizacji każdej klatki z jej etykietą. Wspólna korekta dała0.226157 i nie spełniła kryterium poprawy detekcji. Nie interpretować oracle jako testu generalizacji.

Kontrola fizycznego zakresu wykazała9.5204% współrzędnych target128 poniżej-0.17; minimum-0.527297, wobec minimum GELU około-0.169971. Zatem część celów nie jest osiągalna w tej warstwie z obrazu. `run_gelu.py` jest przygotowany i przeszedł lokalny smoke2 obrazów; NIE został zgłoszony na Eden przed poleceniem użytkownika zatrzymania. `gelu.slurm` to nieuruchomiony launcher.

Następne jawne zlecenie użytkownika dotyczy mocniejszej optymalizacji ostatniego checkpointu lokalnie, jako osobny eksperyment `vit_latent128_maksymalizacja`.

Raport i cele: `outputs/vit_optymalizacja_latentu128/eden/RAPORT.md`, `KANALY128.csv`, `r1.0/TARGETS_ORACLE.npz`. Definicja pełna w PLAN.md. Rejestr wyników zawiera zakończony pierwszy eksperyment; nie oznacza wykonania późniejszej próby GELU.
