# Stan końcowy

- Eksperyment zakończony poprawnie na Edenie.
- Zdalne testy: 7/7, exit 0. Preflight deploy i production: `PASS`.
- Smoke `1825059`: `COMPLETED`, exit `0:0`, H100, czas `00:01:12`.
- Produkcja `1825061`: `COMPLETED`, exit `0:0`, H100, czas `00:02:19`.
- Fresh-load i pakowanie `1825062`: `COMPLETED`, exit `0:0`, H100,
  czas `00:00:23`.
- Wykonano 100 pełnych passów. Każdy pass objął dokładnie 1500 obrazów
  i jedną aktualizację Adam tej samej delty `600x768`.
- Najlepszy i końcowy stan pochodzi z passu 100.
- mAP50:95 wzrósł z `0.24549623827449926` do
  `0.2988348748525347`; AP50 z `0.626887827062123` do
  `0.6734547748214411`; AP75 z `0.13109693084564492` do
  `0.22171054002653443`.
- Confidence wzrosło dla 1454 obrazów i spadło dla 46. IoU wzrosło dla
  992, pozostało bez zmiany dla 57 i spadło dla 451. Jednoczesna poprawa
  co najmniej jednej metryki bez pogorszenia drugiej wystąpiła dla 1017;
  jednoczesne pogorszenie dla 17.
- Liczba `both90` wzrosła z 0 do 8. False positives przy score co najmniej
  0,5 wzrosły z 493 do 675.
- Wagi i bufory K1000 pozostały identyczne. Fresh-load wszystkich 1500
  obrazów odtworzył AP50, AP75 i mAP50:95 bez różnicy.
- Pakiet jest pod `outputs/vit_k1000_wspolna_korekta_patch_20261009/eden/results/package/`.
- Wynik jest świadomie in-sample; nie stanowi testu generalizacji.
