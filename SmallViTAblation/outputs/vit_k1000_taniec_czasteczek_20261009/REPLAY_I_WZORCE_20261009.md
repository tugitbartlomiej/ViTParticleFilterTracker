# Gęsty replay i wzorce ruchu

Użytkownik zlecił ponowne obliczenie stanów wszystkich 1500 × 600 cząsteczek oraz użycie rzeczywistych bibliotek PyDMD z wejściem sterującym i deeptime. Wagi K1000 są zamrożone. Badamy ruch sztucznie optymalizowanych aktywacji, nie kolejne aktualizacje wag ViT.

## Realny stan wykonania

Kontrola SHA256 danych, checkpointa, manifestu oraz 1500 dawnych plików oracle przeszła. Zdalny snapshot ma 31 zgodnych hashy. GPU smoke 1825156 uruchomiono 9 października; stan na moment tego zapisu: RUNNING na NVIDIA H100 PCIe. Pełna produkcja i analizy są kolejnymi etapami. Sam smoke nie potwierdza kompletnego replay ani wspólnej reguły detekcji. Aktualne numery zadań trzeba odczytać z markerów, a nie stale opierać się na numerze smoke.

## Trzy oddzielne reprezentacje

### Aktualizacja wykonania, 2026-10-09 02:06 UTC

Smoke 1825156 zakończył się COMPLETED 0:0. Dla trzech sprawdzonych obrazów dawne stany 0/100/500 i telemetria zostały odtworzone dokładnie (maksymalny błąd 0); wagi i bufory ViT pozostały niezmienione. Pełny array 1825165 uruchomiono raz. Na moment kontroli zapisano 740 z 1500 obrazów; pierwsze cztery shardy kończyły pracę, kolejne cztery czekały na GPU. To stan pośredni, nie ukończenie replay.

Scalenie i eksport: job 1825166. Analiza panelu pełnych współrzędnych PyDMD/deeptime: 1825171. Analiza populacji 900 000 cząsteczek: 1825172. Oba zadania analityczne mają zależność afterok od scalenia. Lokalny collector pobierze raporty dopiero po COMPLETED 0:0, zgodnym provenance i weryfikacji SHA256 wszystkich plików. Finalizer animacji działa oddzielnie i publikuje gęstą oś dopiero po kompletnym transferze.

Osobny suplement populacyjnego PyDMD DMDc jest przygotowywany na tych samych danych PCA32, z przyczynowo dostępnymi w początku interwału normami gradientu i chwil Adama. Są to amplitudy, nie pełne kierunkowe wektory siły; nie zastępują szczegółowego panelu 768D. Numer zadania i wynik suplementu wymagają osobnego potwierdzenia. Brak jeszcze końcowych wyników nowych analiz.

### Aktualizacja wykonania, 2026-10-09 02:13 UTC

Zapisano 1050/1500 obrazów. Wszystkie cztery pozostałe shardy są RUNNING. Suplement DMDc populacji został zgłoszony jako 1825187, afterok:1825166, po 7 poprawnych testach; ma osobny nieruchomy snapshot, a kod zadań 1825171/1825172 nie został zmieniony. Collector PID51580 czeka na wszystkie trzy wyniki i weryfikuje sacct, provenance oraz SHA256. Hubble nadzoruje dokończenie replay/scalenia/transferu i sprawdzi rzeczywisty lokalny interfejs danych. Zeno kontynuuje analizę naukową trzech rzeczywistych wyników i przygotuje WNIOSKI_WZORCE.md. Są to zadania w toku, nie ukończone wyniki.

1. Wizualizacja wszystkich 900 000 cząsteczek: 101 rzeczywistych stanów, od 0 do 500 co 5. Ta sama stara baza PCA3. Bez interpolacji. Normy i kąty zmian pochodzą z pełnych 768 wymiarów.
2. Analiza całej populacji: oddzielna, stała PCA32 dopasowana wyłącznie na filmach train; projekcje wszystkich cząsteczek co 5, mierzone residua i energia ruchu zachowana przez projekcję. deeptime VAMP oraz DMD mają rozłączne filmy dopasowania i oceny. Przypisanie reżimu odbywa się dla całej populacji, a nie tylko dawnego panelu.
3. Panel szczegółowy: 64 obrazy × 16 tokenów, pełne 768 współrzędnych w każdym kroku 0..500 oraz gradient i obie chwile Adama przed aktualizacją. Zawiera target, naturalny top1 i konkurentów. Służy do PyDMD DMDc, porównań z deeptime i kontroli prognozy. Nie jest pełnym zapisem 768D wszystkich 900 000 cząsteczek.

## Kolejność i bramki

GPU smoke musi zakończyć się poprawnie i potwierdzić zgodność z historycznymi stanami 0, 100 i 500. Następnie wrapper producenta zgłasza jeden array ośmiu shardów oraz jedno scalenie z zależnością afterok. Konsumenci PyDMD/deeptime są zgłaszani raz przez własny wrapper dopiero po ustaleniu numeru scalenia, z zależnością afterok od tego scalenia.

Zdalny producent: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009`.

Zdalny odbiorca analizy: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009`.

Manifesty kompletności publikuje producent w `results/merged`. Eksport wizualizacji jest w `results/visualization_dense`. Lokalny finalizer czeka wyłącznie na zweryfikowany kompletny eksport; nie zgłasza treningów ani nie oblicza brakujących stanów. Dopiero po pobraniu i sprawdzeniu wszystkich hashy publikuje `data/dense/manifest.json`.

## Interpretacja

Osobno oceniamy fazę poprawiania detekcji oraz fazę ograniczania normy korekty. Rozdzielamy sukcesy i wyczerpany budżet, target i pozostałe tokeny. Prosta kontrakcja po sukcesie nie stanowi sama w sobie wspólnej reguły poprawiającej detekcję. Porównujemy błąd i kierunek prognozy z brakiem ruchu, stałym przyrostem i kontrakcją zależną od fazy.

Kontrola DMDc z analitycznym krokiem Adama ma charakter teacher-forced: zna stan i gradient w danym kroku. Wielokrokowy wynik z zapisanymi przyszłymi sterowaniami nie jest prognozą ze znajomości wyłącznie stanu początkowego. Grupowanie ruchów i dobra rekonstrukcja nie dowodzą jeszcze wzrostu IoU po zastosowaniu wspólnego korektora.

## Podgląd i powtarzalność

Adres podglądu: http://127.0.0.1:8774/ . Serwer i transfer działają jako osobne procesy. Do pełnej weryfikacji pozostają rzeczywiste dane producenta; prywatne syntetyczne testy sprawdziły kolejność obraz/token/krok, odmowę publikacji po uszkodzeniu oraz odrzucanie interpolacji i brakujących stanów.

Stan procesu lokalnego i numery zadań: `DENSE_PIPELINE_STATUS.json`, `data/dense/FINALIZACJA.json`, `data/dense/TRANSFER.json`. Końcowe wyniki naukowe i provenance znajdą się w `outputs/vit_k1000_wzorce_pydmd_deeptime_20261009`; transfer wizualizacji ma odrębny status.
