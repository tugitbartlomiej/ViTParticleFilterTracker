# Plan: jedna wspólna korekta per patch K1000

## Pytanie

Czy bezpośrednie joint uczenie jednej współdzielonej macierzy `delta[600,768]`
na naturalnych reprezentacjach dokładnych 1500 bazowych obrazów poprawia
treningowy mAP K1000? Jest to świadomie analiza in-sample. Nie stanowi testu
generalizacji.

## Wejścia i interwencja

- K1000 SHA-256: `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Corrected manifest 1500 SHA-256:
  `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`.
- GT skaluje każdą oś osobno do `480x320`, identycznie jak zweryfikowana
  poprzednia rodzina.
- Obrazy są czytane wprost z istniejącego TAR do RAM. Nie powstają wypakowane
  obrazy ani cache latentów na dysku.
- Punkt interwencji: post-GELU ostatniego MLP. Dla każdego obrazu zachowujemy
  jego naturalne `z0`; stosujemy `z' = clamp_min(z0 + delta, -0.16997121)` i
  oryginalny zamrożony tail.
- K1000 pracuje w `eval`, FP32, z wyłączonym TF32. Wagi i bufory są zamrożone.

## Optymalizacja

`delta` startuje dokładnie od zera. Jeden Adam aktualizuje wyłącznie tę samą
macierz `600x768`. Jeden numerowany pass oznacza dokładnie jeden gradient
średniej stałej straty po wszystkich 1500 obrazach i jedną aktualizację Adam.
Mikropaczki sumują `loss_i/1500`, więc każdy obraz ma wagę `1/1500`, również
gdy ostatnia mikropaczka jest mniejsza.

Stała strata każdego obrazu przez cały bieg zawiera: dodatni logit właściwego
tokenu, tłumienie pozostałych logitów, ranking top-1, Smooth L1 logarytmów
`ltrb`, GIoU ramki oraz `1e-6 * mean(delta^2)`. Nie ma przełączania celu po
sukcesie obrazu.

- Maksimum: 100 pełnych passów.
- LR Adam: `0.01` dla passów 1-40, `0.003` dla 41-70 i `0.001` dla 71-100.
- Pełna ewaluacja 1500 obrazów: pass 0 i co 5 passów.
- Ranking checkpointu: wyłącznie treningowy mAP50:95. Minimalna poprawa to
  `1e-5`. Delta zero pozostaje najlepsza, jeżeli uczenie jej nie przebije.
- Plateau: po co najmniej 30 passach, 6 kolejnych pełnych ewaluacji bez poprawy
  o `1e-5` kończy bieg. Limit 100 pozostaje twardy.
- Checkpoint wznowienia po każdym pełnym passie zawiera deltę, Adam, RNG,
  konfigurację, historię i najlepszą deltę.

## Podłoga i smoke

Podłogą jest dokładnie `delta=0` i pełny forward tego samego K1000. Pełna
podłoga musi odtworzyć AP50 `0.626887827062123` i mAP50:95
`0.24549623827449926` z dotychczasowego protokołu.

Smoke na rzeczywistych obrazach blokuje produkcję, jeżeli nie przejdzie
którykolwiek warunek: zgodność delta-zero z naturalnym forwardem, zgodność fast
tail z hook-forward, gradient wyłącznie do delty, dokładne ważenie średniej przy
nierównych mikropaczkach, niezmieniony hash wag i buforów, round-trip zapisu,
oraz spadek tej samej stałej straty po pierwszym kontrolowanym kroku Adam o LR
`0.01`. Raportuje także rzeczywistą pamięć i czas.

## Ewaluacja i artefakt

Każda pełna ewaluacja zapisuje AP50, AP75, mAP50:95, rozkłady top-1 confidence
i IoU, liczbę przypadków `both90`, relacje improved/unchanged/worse i Pareto,
trafność właściwego tokenu jako top-1, confidence właściwego tokenu oraz liczbę
false positives przy score co najmniej 0,5 i 0,9. Zapisuje też normę delty,
składniki straty, gradient i czas. Raport porównuje baseline, końcową deltę,
najlepszą deltę oraz historyczną Mean A. Mean A jest wyłącznie odniesieniem i
nie jest inicjalizacją ani celem treningu.

Po biegu osobny job ładuje samodzielny pakiet K1000 + najlepsza delta + kod +
config + provenance + hashe i ponownie ocenia wszystkie 1500 obrazów w świeżym
procesie. Selekcja i retest pozostają in-sample.

Historyczna Mean A mieszała korekty tokenu obiektu z korektami tła dla tego
samego indeksu patcha w innych obrazach; confidence spadło dla 1481 z 1500
obrazów. Historyczne B odtwarzało odkryte etykiety reżimu z trafnością tylko
36,06 procent na danych uczących. Są to hipotezy wyjaśniające porażki A/B, a
nie dowód, że bezpośrednie joint uczenie albo każdy conditional corrector musi
zawieść.

## Co obala hipotezę i czego wynik nie dowodzi

Brak wzrostu mAP50:95 względem `delta=0`, pogorszenie większości obrazów albo
utrzymanie baseline jako najlepszego checkpointu obala użyteczność tego
najprostszego wspólnego przesunięcia w zadanym budżecie. Nawet poprawa nie
dowodzi generalizacji na nowych filmach ani realizowalności korekty przez
wcześniejsze warstwy modelu.
