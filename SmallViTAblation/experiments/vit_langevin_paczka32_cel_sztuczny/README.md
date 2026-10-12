# Paczka32 do oryginalnego celu

Uruchomienie: `powershell -NoProfile -ExecutionPolicy Bypass -File experiments/vit_langevin_paczka32_cel_sztuczny/uruchom.ps1` z root SmallViTAblation. Ukryty proces; PID, atomowy PROGRESS.json, stdout.log i stderr.log w rodzinie outputs o tej samej nazwie. Nie uruchamiać drugiej kopii.

Plan i progi przed biegiem: PLAN.md. WYNIKI.json deklaruje rodzinę badawczą, istniejące narzędzie dopisało ją do outputs/rejestr/rodziny.json. Kod modelu i równania skopiowany bez zmian z istniejących eksperymentów do tego katalogu. `prognoza_krok_original_reference.py` jest wyłącznie zapisem źródła równania; worker używa rownanie_original.py bez jego zmiany.

Przygotowanie CPU: SELECTION.json, rzeczywiste zdjęcia, GROUNDTRUTH_selected32.json i GROUNDTRUTH_random32.json, surowe annotation.csv, ARCHIVE_AUDIT.json, SONDA_PROVENANCE.json. Checkpoint pełny pobierany z Edenu tylko do tej rodziny i sprawdzany SHA256 oraz tensor po tensorze. CHECKPOINT_VALIDATION.json i OPTIMIZER_PROVENANCE.json dokumentują stan. Brak tego stanu zatrzymuje bieg z BLOCKER.json.

GPU czeka na zakończenie vit_sd_cztery_szumy i wolne6500MiB. Potem FORECAST.json/csv i FORCE_*.npz zapisane przed prawdziwym treningiem, FIRST_STEP_VALIDATION.json, ACTUAL_*.json i latenty1/5/10/25/50. Kopie sieci uczone na całych wybranych32 lub stałych losowych32; oryginalna sonda jest zawsze ta sama. Bez podmiany latentu, bez zamrażania głowy, bez zmiany architektury. Brak augmentacji jawnie określony w PLAN.md.

Wynik końcowy: RESULT.json i REPORT.md; błąd: BLOCKER.json i REPORT.md. Null przy hitting_time oznacza brak trafienia w zadeklarowanym horyzoncie50. Prognoza stałej początkowej siły jest przybliżeniem, którego nie poprawia się przyszłymi gradientami. Progi10/25/50procent redukcji MSE mierzone osobno128/768. Epoka puli32 oznacza1batch, nie epokę pełnego zbioru. Transfer historycznego równania ze średnich tokenów na całą siatkę wymaga weryfikacji pierwszego kroku.
