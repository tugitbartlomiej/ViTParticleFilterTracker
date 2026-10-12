# Raport końcowy: K1000, 1500 pełnych latentów 600x768

## Werdykt

Bezpośrednia optymalizacja oracle zamrożonego latentu osiągnęła jednocześnie
confidence co najmniej 0,90 i IoU co najmniej 0,90 dla 1283 z 1500 obrazów
(85,53 procent). Pozostałe 217 przypadków uczciwie wyczerpało budżet 500
rzeczywistych aktualizacji.

Wspólne korekty nie przeniosły tego wyniku. Ani A, czyli średnia delta
600x768, ani B, czyli korekta przypisana z naturalnego `z0` i kontekstu, nie
osiągnęły obu progów dla żadnego obrazu. Obie obniżyły mAP50:95 względem
baseline, także w wewnętrznym holdoucie grupowanym po filmach. Nie pomagają
wszystkim i nie powinny być używane jako poprawa K1000 w obecnej postaci.

## Dokładne wejścia i wykonanie

- K1000 SHA-256:
  `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Manifest dokładnych 1500 klatek SHA-256:
  `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`.
- Pełny stan AdamW SHA-256:
  `e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6`.
- Archiwum źródłowe SHA-256:
  `5850b5c8dd4411159061e254e1f9b32e2b4b633036d3990049b553bea767d8b1`.
- Smoke `1824869`: `COMPLETED`, `exit 0:0`.
- Produkcja `1824873`: osiem niezależnych shardów H100, wszystkie
  `COMPLETED`, `exit 0:0`.
- Merge `1824877`, ruch `1824878`, reżimy `1824879` i DMDc `1824880`:
  `COMPLETED`, `exit 0:0`.
- Pierwsze evaluation `1824881` zatrzymało się przed oceną obrazów przez
  obowiązkowy odczyt metadanych opcjonalnego `timm`. Po teście regresji
  poprawiono tylko ten punkt. Evaluation `1824979` i final `1824980`
  zakończyły się `COMPLETED`, `exit 0:0`.

Wagi i bufory miały przed i po identyczny hash
`443ef243f956ead49a7b9119cf3d454d93251476115006d1416a88d45415246e`.
Wszystkie parametry modelu miały `requires_grad=False`. Ponowne wczytanie
latentów przeszło dla 1500/1500 przypadków. Nie zapisano żadnych nowych plików
obrazu.

## Wynik oracle

- Sukces obu progów: 1283/1500; wyczerpanie budżetu: 217/1500.
- Mediana pierwszego sukcesu: 58 kroków; 90. percentyl: 156; maksimum: 400.
- Mediana confidence: 0,770762 przed i 0,937905 po.
- Mediana IoU: 0,666063 przed i 0,923277 po.
- Confidence wzrosło dla 1389 obrazów, spadło dla 111.
- IoU wzrosło dla 1491 obrazów, spadło dla 9.
- Oba wskaźniki wzrosły jednocześnie dla 1380 obrazów; dla 120 wynik był
  mieszany. Nie było przypadku, w którym oba wskaźniki jednocześnie spadły.
- AP50: 0,975087; AP75: 0,940328; mAP50:95: 0,861131.

Pełne wspólne stany są rzeczywistymi stanami `k=0`, `k=100` i `k=500`, bez
terminalnego dopełniania. Pole `after` wskazuje najlepszy rzeczywiście
odwiedzony stan danego obrazu i nie jest wspólnym czasem.

## Wspólne reguły A i B

| metoda | AP50 | AP75 | mAP50:95 | oba progi 0,90 |
|---|---:|---:|---:|---:|
| baseline | 0,626888 | 0,131097 | 0,245496 | 0 |
| A mean, in-sample | 0,631023 | 0,115980 | 0,239411 | 0 |
| B regime, in-sample | 0,622992 | 0,131139 | 0,244481 | 0 |
| A mean, holdout po filmach | 0,630688 | 0,116778 | 0,239096 | 0 |
| B regime, holdout po filmach | 0,628032 | 0,127554 | 0,242897 | 0 |
| oracle per-image | 0,975087 | 0,940328 | 0,861131 | 1283 |

Dla A in-sample confidence poprawiło się dla 19 obrazów i pogorszyło dla
1481; IoU poprawiło się dla 627, nie zmieniło dla 60 i pogorszyło dla 813.
Tylko 4 obrazy miały poprawę Pareto obu wskaźników, a 858 pogorszenie Pareto.

Dla B in-sample confidence poprawiło się dla 10 obrazów i pogorszyło dla
1490; IoU poprawiło się dla 693, nie zmieniło dla 54 i pogorszyło dla 753.
Tylko 3 obrazy miały poprawę Pareto obu wskaźników, a 800 pogorszenie Pareto.

Pełne wyjątki i 1500 rekordów per-image znajdują się w
`eden_results/evaluation/`. Dopasowanie in-sample nie jest dowodem
generalizacji. Holdout po filmach jest tylko kontrolą wewnętrzną, a nie
zewnętrznym zbiorem testowym.

## Ruch, reżimy i DMDc

Analiza obejmuje 900 000 cząsteczek we wspólnej bazie 768D. Mediana cosinusa
między ruchem `k=0→100` i `k=100→500` wyniosła -1, a 768 954 cząsteczki miały
kierunki przeciwne. Mediana pełnego ruchu `k=0→500` była bliska zeru
(`1,69e-6`), natomiast 90. percentyl normy L2 wyniósł 105,58. Wskazuje to na
silne odwracanie i kasowanie ruchu dla większości cząsteczek oraz duży ruch
mniejszości; same średnie delty tracą tę strukturę.

Reżimy odkrywano z `z0` i oracle ruchu, ale inferencja B widziała wyłącznie
naturalne `z0`, pozycję tokenu i kontekst obrazu. Trafność odtworzenia
odkrytej etykiety reżimu wyniosła 36,06 procent na danych uczących; w trzech
modelach foldowych było to 34,91, 36,68 i 35,96 procent.

DMDc użył 128 rzeczywistych trajektorii, 64 000 przejść oraz surowego gradientu
i chwil Adama sprzed `optimizer.step`. Nie użyto przyszłej delty jako wejścia.
Na odłożonych całych trajektoriach RMSE wyniosło 0,080697 dla persistence,
0,166732 dla DMD i 0,166738 dla DMDc. DMDc nie poprawił DMD ani persistence i
pozostaje analizą opisową, nie regułą transferu.

## Pakiety i lokalizacja

Samodzielne pakiety zawierają pełny K1000, kod modelu, konfigurację,
provenance, korektę i manifest hashy:

- A: `eden_results/packages/model_mean`, korekta SHA-256
  `58050439532acc743a3b6581f937ba1dc38ad4d3d0322936c39a21ecfd6b70f2`.
- B: `eden_results/packages/model_regime`, korekta SHA-256
  `781944903dfb55587f53e2b6af15af479f1c5b3022d94115e373067883cf0fc3`.

Dwa niezależne fresh-load dały bitowo identyczne wyjścia dla obu pakietów.
Lokalne pakiety ponownie przeszły kontrolę wszystkich hashy i wczytanie na CPU.

Pełne latenty per-image pozostają na Edenie:
`/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/results/production`.
Zajmują 11 011 554 455 bajtów. Cały zdalny root zajmuje 11 474 808 890 bajtów;
po zakończeniu było 134 170 943 488 bajtów wolnego.

Lokalna kopia raportów, analiz i pakietów zawiera 1547 plików oraz
355 813 711 bajtów. Jej hash drzewa, zgodny z Edenem, to
`d179b873f883386aed86a23726f6da06ddcca1803c81d2a3477a91c8332e4131`.

## Ograniczenia interpretacji

Oracle działa bezpośrednio w punkcie post-GELU i jest górną granicą
zamrożonego ogona oraz głowy. Nie dowodzi, że znalezione stany są osiągalne z
obrazu przez wcześniejsze warstwy. Dla nowych celów pozostają potrzebne testy
realizowalności i analiza składowych w nullspace `fc2`.
