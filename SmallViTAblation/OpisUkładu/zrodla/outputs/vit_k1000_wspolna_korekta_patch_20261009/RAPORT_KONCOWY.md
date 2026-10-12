# Raport końcowy: wspólna korekta per patch K1000

## Werdykt

Bezpośrednie joint uczenie jednej wspólnej delty `600x768` poprawiło wynik
zbiorczy na tych samych 1500 obrazach. mAP50:95 wzrósł z `0.245496238` do
`0.298834875`, czyli o `0.053338637` bezwzględnie i o 21,73 procent względnie.
Najlepszym stanem był pass 100, ostatni w zadanym budżecie.

Korekta nie pomaga wszystkim. Confidence spadło dla 46 obrazów, IoU dla 451,
a oba wskaźniki jednocześnie pogorszyły się dla 17. Liczba false positives
przy score co najmniej 0,5 i IoU 0,5 wzrosła z 493 do 675. Wynik potwierdza
użyteczność jednej delty dla agregatu treningowego, ale nie potwierdza
uniwersalnej poprawy per image.

## Kontrakt eksperymentu

- Dokładny K1000 SHA-256:
  `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Dokładny corrected manifest 1500 SHA-256:
  `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`.
- FP32, TF32 wyłączone, tryb eval, wszystkie wagi i bufory zamrożone.
- Interwencja po GELU: `clamp_min(z0 + delta, -0.16997121)`; naturalne `z0`
  pozostaje zależne od obrazu.
- Delta startowała dokładnie od zera. Jeden pass oznacza średni gradient po
  wszystkich 1500 obrazach i jedną aktualizację Adam.
- Stały cel łączy dodatni logit właściwego tokenu, tłumienie 599 pozostałych
  logitów, ranking top-1, pozycję ramki, GIoU i małą karę normy delty.
- Selekcja checkpointu odbywała się według mAP50:95 na zbiorze uczącym.

## Kontrole przed produkcją

Smoke `1825059` zakończył się kodem zero i przeszedł 8 z 8 bramek. Ta sama
strata spadła po jednym kroku Adam z `41.6146163940` do `41.4689331056`.
Względna różnica gradientu między nierównymi mikropaczkami i pełną średnią
wyniosła `1.244628572e-7`; maksymalna różnica fast tail względem rzeczywistego
hook-forward wyniosła `1.430511475e-6`. Fresh-load delty w smoke był bitowo
zgodny, a wagi i bufory nie zmieniły się.

## Metryki

| wariant | AP50 | AP75 | mAP50:95 | both90 |
|---|---:|---:|---:|---:|
| baseline `delta=0` | 0.626887827 | 0.131096931 | 0.245496238 | 0 |
| historyczna Mean A | 0.631022894 | 0.115979549 | 0.239411249 | 0 |
| historyczne Regime B | 0.622991980 | 0.131139018 | 0.244481144 | 0 |
| joint delta, pass 100 | 0.673454775 | 0.221710540 | 0.298834875 | 8 |
| fresh-load joint delta | 0.673454775 | 0.221710540 | 0.298834875 | 8 |

Zmiany joint względem baseline:

- AP50: `+0.046566948`.
- AP75: `+0.090613609`.
- mAP50:95: `+0.053338637`.
- Właściwy token jako top-1: 846 do 889 obrazów.
- `both90`: 0 do 8 obrazów.
- Strata całego zbioru: `13.2242533773` do `11.6182832742`.
- False positives przy score co najmniej 0,5: 493 do 675; liczba obrazów z
  co najmniej jednym takim false positive: 458 do 576.
- True positives przy tym samym progu: 1114 do 1177.

Bilans per image:

| miara | improved | unchanged | worse |
|---|---:|---:|---:|
| confidence | 1454 | 0 | 46 |
| IoU | 992 | 57 | 451 |
| relacja Pareto | 1017 | 0 | 17 |

Dla relacji Pareto pozostałe 466 obrazów ma wynik mieszany: jedna miara
wzrosła, a druga spadła. Mediana zmiany confidence wynosi `+0.064183027`, a
mediana zmiany IoU `+0.020557612`. Najgorsza zmiana confidence to
`-0.306883484`, a IoU `-0.497956783`.

## Dlaczego historyczne A i B zawiodły

Kod per-image oracle nadaje dodatni cel jednemu tokenowi wskazanemu przez GT,
jednocześnie uśredniając karę ujemną po pozostałych 599 tokenach. Indeks
tokenu obiektu zmienia się między obrazami. Dla ustalonego indeksu patcha
większość indywidualnych optymalizacji traktuje ten token jako tło, a tylko
część jako obiekt. Mean A uśredniła te sprzeczne role przed zastosowaniem do
każdego obrazu. To najbardziej prawdopodobny mechanizm spadku confidence w
1481 z 1500 obrazów. Dodatkowo średnia niezależnych minimów nie musi być
minimum średniej straty, szczególnie po nieliniowym clampie i tailu modelu.

Regime B odkrywało etykiety przy użyciu naturalnego `z0` oraz kierunku i
magnitudy oracle delta. W inferencji etykietę przypisywał najbliższy centroid
wyłącznie z `z0`, pozycji tokenu i kontekstu obrazu. Trafność tego przypisania
wobec etykiet odkrytych z oracle wyniosła tylko `0.3605744444` już na zbiorze
uczącym. Następnie B stosowało uśrednioną korektę dla pary reżim-indeks
tokenu. Ten słaby interfejs między odkrywaniem i przypisaniem jest zgodny ze
spadkiem confidence dla 1490 z 1500 obrazów i niewielkim spadkiem mAP.

Te dane silnie wspierają opisane mechanizmy, ale nie są osobną ablacją
przyczynową. Nie dowodzą też, że każdy conditional corrector musi zawieść.
Powodzenie bezpośredniej delty joint pokazuje, że porażka średniej niezależnych
oracle nie była dowodem niemożliwości wspólnej korekty. Joint agreguje
gradient wspólnego celu przed każdą aktualizacją, zamiast uśredniać końcowe,
zależne od obrazu rozwiązania.

## Artefakt i odtwarzalność

- Smoke: `1825059`, `COMPLETED`, `0:0`, H100, `00:01:12`.
- Produkcja: `1825061`, `COMPLETED`, `0:0`, H100, `00:02:19`.
- Pakowanie i fresh-load: `1825062`, `COMPLETED`, `0:0`, H100, `00:00:23`.
- Wykonano 100 pełnych passów i dokładnie 100 aktualizacji Adam.
- Czas części optymalizacyjnej: `114.4878 s`; budowa cache wyłącznie w RAM/GPU:
  `13.7662 s`; peak reserved GPU: `3,913,285,632 B`.
- Delta ma RMS `0.349233419`, L2 `237.067505`, max abs `0.628483891` i 460800
  niezerowych współrzędnych.
- SHA-256 zawartości tensora delty:
  `9004495798a44ceaafca93f65943dc144166d66f824f83f134346c8bc4a8d777`.
- SHA-256 pliku `CORRECTION.pt`:
  `c46acd19a2247eead6209c700c910fdcfa42ecf66a434631c14e978d53171d71`.
- SHA-256 manifestu pakietu:
  `5aa4fdebbb53775fb543760b46319d0e30e2ead1e70b63846769382b0c89a35f`.

Osobny proces załadował samodzielny pakiet dwa razy, uzyskał bitową zgodność
pierwszej paczki i ponownie ocenił wszystkie 1500 obrazów. Różnica AP50, AP75
i mAP50:95 względem stanu best wyniosła dokładnie zero. Fingerprint wag i
buforów przed i po wynosił w obu przypadkach
`443ef243f956ead49a7b9119cf3d454d93251476115006d1416a88d45415246e`.

Pakiet znajduje się w `eden/results/package/`. Wszystkie osiem jego plików
przeszło lokalną kontrolę SHA-256. Rodzina nie zawiera obrazów ani cache
latentów na dysku. Zdalny root zajmuje `38,142,244 B`.

## Ograniczenie wnioskowania

Optymalizacja, wybór najlepszego passu i ocena używają tych samych 1500
obrazów. Jest to uczciwy test in-sample wspólnej interwencji, nie wynik
generalizacji na nowych filmach lub obrazach. Nie sprawdza również, czy delta
może zostać zrealizowana przez wcześniejsze warstwy modelu.
