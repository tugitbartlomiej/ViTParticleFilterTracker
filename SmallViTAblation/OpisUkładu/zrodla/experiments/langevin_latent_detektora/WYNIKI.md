# Równanie Langevina dla latentu detektora: wynik pierwszego etapu (2026-09-26, noc)

Skrypt `langevin.py`, plan i bramka `PLAN.md`, liczby `wyniki/REPORT_S48.json`, `wyniki/REPORT_S96.json`.
Cząstka: skalar tożsamości narzędzia jednego obrazu sondy (72 obrazy, 48 fit / 24 holdout), krok = 5 aktualizacji,
bieg 1810662 (kroki 5..800). Miara: RMSE na obrazach holdout względem braku zmian (1,0 = tak źle jak nic nie
przewidywać). Odniesienie: zapisany wspólny operator L4 (Koopman na 4 ostatnich przyrostach).

## Wybór na kalibracji (przed testem)

W obu oknach kalibracja wybrała długą pamięć: 8 ostatnich przyrostów (40 aktualizacji). Członu położenia w dryfie
i wymuszenia harmonogramem (granica epoki, faza epoki, 1/krok) nie wybrała. Szum kolorowy AR(2) wybrała tylko w
oknie od kroku 485, gdzie prawie nic nie zmienia.

## Test na holdout

| okno | model | 1 krok | h = 10 | h = 20 | h = 40 | do końca |
|---|---|---|---|---|---|---|
| od kroku 245 (dryf trwa) | **Langevin: dryf Koopmana + pamięć 8** | 0,416 | 0,150 | **0,151** | **0,220** | **0,370** (h = 111) |
| | L4 zapisany | 0,407 | 0,152 | 0,171 | 0,254 | 0,417 |
| | sam dryf (bez pamięci) | 0,417 | 0,376 | 0,482 | 0,555 | 0,695 |
| od kroku 485 (dryf wygasł) | Langevin | 0,682 | 1,414 | 0,557 | 0,367 | 0,461 (h = 63) |
| | L4 zapisany | 0,682 | 1,232 | 0,454 | 0,435 | 0,578 |
| | sam dryf | 0,705 | 1,123 | 0,602 | 0,602 | 0,801 |

## Bramka (ustalona przed wynikiem): niespełniona

Od kroku 245 rollout h = 20 wynosi 0,886 × L4 (warunek ≤ 0,95 spełniony), ale 1 krok jest gorszy o 2,2 procent
(dopuszczalne 2). Od kroku 485 rollout h = 20 wynosi 1,23 × L4. Równanie nie zastępuje formalnie L4.

## Co z tego wynika

1. Ruch latentu niesie pamięć, nie dryf chwilowy. Bez pamięci błąd na 20 kroków rośnie z 0,15 do 0,48. Dłuższa
   pamięć (40 aktualizacji zamiast 20) poprawia długi rollout w fazie dryfu o 11–13 procent.
2. Od kroku 245 reszta jednokrokowa ma autokorelację 0,27 przy opóźnieniu 1: według twierdzenia o rozkładzie błędu
   w równaniu brakuje jeszcze członu (część usuwalna). AR na resztach go nie łapie na kalibracji.
3. Od kroku 485 reszty są białe (autokorelacja −0,07..−0,01): zostaje sama innowacja, czyli szum mini-batchy. Żaden
   człon nie przewidzi jej lepiej; krótkie horyzonty przegrywają z brakiem zmian (h = 10 powyżej 1).
4. Reszty od kroku 485 korelują z granicą epoki (−0,24), ale wymuszenie wybrane na kalibracji nie weszło. To kandydat
   na dopracowanie członu W (np. przetasowanie danych jako impuls z własnym zanikiem).
5. Sterowanie (skład danych) jest w tym biegu stałe. Etap 2 (PLAN.md) użyje 12 biegów kampanii 5 epok.
