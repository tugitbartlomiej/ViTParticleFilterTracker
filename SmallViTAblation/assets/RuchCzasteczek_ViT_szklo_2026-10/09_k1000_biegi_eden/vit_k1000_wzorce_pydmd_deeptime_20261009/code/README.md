# K1000 — rzeczywiste PyDMD/deeptime

Samodzielna rodzina CPU. Wyniki: `../../outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/`.

- `temporal_core.py`: właściwe API, skale train, pary, integracja kontrolna Adama.
- `analiza_temporalna.py`: stare128 lub nowy panel1024 pełne768 co1, fazy, prognozy1/5/20, VAMP i kontrola OOD.
- `analiza_populacji.py`: finalne manifesty Hubble, PCA32 co5, fit train i strumieniowe reżimy wszystkich900k, autonomiczne DMD5/20 oraz geometria pełnych768.
- `test_pipeline.py`: naukowe błędy parowania, leakage, sterowania, cech VAMP i utraty wymiarów.
- `submit_after_merge.py`: dwa joby CPU po rzeczywistym ID scalenia producenta, gate testów, snapshot produkcyjny i ochrona przed duplikatem.
- `podsumuj_smoke.py`: odczytowy audyt pobranych wyników, bez ponownego fit.

Odtwarzanie i komendy: [KONTRAKT_DANYCH.md](KONTRAKT_DANYCH.md) oraz [KOORDYNACJA.md](../../outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/KOORDYNACJA.md). Rzeczywiste smoke1825145 i1825150 zakończone; nie mylić z przygotowanymi lub oczekującymi biegami docelowymi.

Nie ma tu treningu/eksportu correctora, modyfikacji K1000, producenta replay lub wizualizacji. Brak bibliotecznej atrapy: PyDMD.DMDc oraz deeptime.decomposition.DMD/VAMP wykonują właściwy fit. Żaden tensor wizualizacji PCA3 nie jest wejściem analizy.
