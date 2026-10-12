# DMDc paczek — ukończony bieg

Główny raport: [full_v2/REPORT.md](full_v2/REPORT.md).
Główny wykres: [full_v2/KEY_RESULT.png](full_v2/KEY_RESULT.png).

- `full_v2/`: poprawnie ukończony i zweryfikowany bieg 300–1000, 3 sondy × 600 tokenów × 768 współrzędnych.
- `pilot_cached/`: poprawnie ukończony pilot 300–400.
- `source_eden/`: pobrane istniejące Z i S_nowa z jednego biegu H100 s47; nie powtarzano treningu.
- `full/`: zachowana pierwsza przerwana próba pełnego biegu; błąd końca tablicy sił wykryty przed obliczeniami i naprawiony w full_v2.
- `pilot/`: zachowany pilot z wolnym strided mmap; zatrzymany na rzecz szybszego czytania ciągłego do RAM.
- `ATTEMPT01_IMPORTS.json`: własny proces zatrzymany podczas wolnego importu nieużywanych rodzin PyDMD.
- `ACTUAL_BATCH_SEQUENCE.npz`: odtworzony skład paczek i augmentacje, zgodny z przepisem RNG47.

Wszystkie własne procesy obliczeń i audytu zakończone. Żaden cudzy proces nie był zatrzymywany.
