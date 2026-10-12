# Wspólna identyfikacja Koopman-GLE z historią obrazowych wejść

Wykonano odrębny pilot CPU bez KF. Wszystkie bloki operatora, pamięci stanu oraz pamięci wejść dopasowano w jednym problemie ridge do całego przyrostu liftingu. To naprawia konstrukcję estymacji; nie zapewnia przewagi empirycznej. s48 było wcześniej oglądane i pozostaje eksploracyjne. Zapisana trajektoria s47 późno także nie jest nowym niezależnym biegiem.

## Wyprowadzenie i estymacja

Rozważmy wybrany lifting x oraz ukryty stan h, z historycznym sterowaniem u:

```
x[k+1] = A x[k] + B h[k] + D u[k] + noise_x[k]
h[k+1] = E h[k] + C x[k] + F u[k] + noise_h[k]
h[k] = E^k h[0] + sum(j<k) E^(k-1-j)(C x[j]+F u[j]+noise_h[j])
x[k+1] = A x[k] + D u[k]
           + sum(j<k) B E^(k-1-j) C x[j]
           + sum(j<k) B E^(k-1-j) F u[j]
           + B E^k h[0] + colored_residual[k]
```

To algebraiczna eliminacja przy przyjętej liniowej realizacji w liftingu. Jądra stanu i wejścia mają wspólne ukryte E, ale różne C/F. Nie dowodzi istnienia dokładnego skończonego liftingu dla ViT. Nieznane u[k] w chwili stanu x[k] pozostaje nieznane również po eliminacji. Model pełnego AdamW obejmowałby wagi, momenty, bufory i politykę paczek; obecna obserwabla jest tylko ustalonym skalarnym rzutem LDA.

Lean dowodzi siedmiu tożsamości skalarnych nad Int: rozdzielenia sum pamięci, eliminacji, zależności pamięci tylko od j<k oraz rozwinięcia przykładu. SymPy sprawdza macierzowy nieprzemienny przykład2x2 k0..3 i joint blockridge. [Zakres i założenia](../../matematyka/CONTROLLED_MEMORY.md). Dowód algebraiczny nie gwarantuje prognozy ani semantycznej tożsamości.

Pilot marginalny: ψ=(z,z²), gdzie z jest s minus historyczna średnia500, standaryzowane na train. Referencja okna zamrożona. Target to Δψ_H, regresory: bieżący ψ,20 opóźnień Δψ, obrazy wykonanych paczek w rozłącznych grupach ostatnia1, poprzednie4 i poprzednie15. Operator i oba jądra są estymowane JOINT. Obrazowy q[t] pochodzi z paczki źródłowej t−1; bez q[t+1:]. Dla opt dodatkowo osobne wykonane N/P. Intercept i normalizacja wyłącznie train. To projekcja marginalna K_H z historią, nie semigrupa ani eksperymentalnie zidentyfikowany mechanizm przyczynowy. Pamięć przyrostów jest jawną bazą temporalną; nie nadano jej statusu dokładnego kernelu MZ.

Osobny controlled_images fit używa historycznego u[k] wywołującego x[k+1], ale przy rollout zastępuje przyszłe wejścia średnią20 ostatnich wykonanych paczek. Wszystkie bloki one-step nadal estymowane wspólnie. Modeluje przyszłą politykę w bardzo uproszczony sposób; nie odczytuje przyszłych paczek. Addytywne wejście nie pokrywa wszystkich możliwych zależności stanu i obrazu; biliniowe/interakcyjne wejście pozostaje uzasadnioną przyszłą ablacją.

Protokół zapisany [przed estymacją](PLAN.md). Train do1800, walidacja1800..2000, refit do2500; końce wszystkich etykiet train/refit dostępne przed cut. Ridge0.001/0.1/10 i ramiona wybierane po MSE walidacji osobnoH; brak kary wiążącejH. PCA4 train-only opisuje rzeczywiste wykonane obrazy przez zapisane embeddingi paczki; nie sondy. To embeddingi osobno uczonych sieci, nie wspólny zamrożony encoder ani surowy piksel. Proste deskryptory pochodzą z tej samej sieci/LDA. Zgodność przestrzeni między biegami nie jest udowodniona.

## Wyniki wyboru na walidacji, s48

| H | Model | Redukcja MSE % | Znak % | Balanced % | MAE | RMSE | Pokrycie90 % |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | opt | 64.12 | 80.92 | 80.92 | 0.0683 | 0.0954 | 87.70 |
| 3 | opt | 40.38 | 72.84 | 72.89 | 0.2341 | 0.3217 | 88.70 |
| 5 | opt | 25.67 | 67.20 | 67.25 | 0.3833 | 0.5135 | 90.53 |
| 10 | memory | 21.86 | 65.19 | 65.19 | 0.5777 | 0.7608 | 89.89 |
| 20 | opt | 22.95 | 64.73 | 64.58 | 0.7330 | 0.9353 | 85.81 |
| 50 | opt | 29.74 | 68.17 | 67.95 | 0.7634 | 0.9810 | 87.18 |

80 procent osiągnięto wyłącznie dla znaku/balanced na H1:80,92. Nie osiągnięto80 redukcjiMSE. Maksymalny badany H spełniający próg znaku to1; zapis sond co10 oznacza inny H niż jedna aktualizacja. Nie stwierdzono monotoniczności ani potwierdzenia na świeżym biegu. Poprzedni direct H1 miał64,96 redukcjiMSE i80,98 znaku; nowyjoint H1 ma64,12 i80,92, zatem nie wykazano poprawy względem poprzedniego najlepszego wariantu.

Samo obrazowe sterowanie H1: pamięć62,31 redukcjiMSE i79,85 znaku; obrazy62,43 i79,77. Kontrolowany one-step z modelowanymi przyszłymi obrazami62,50 i79,93. Największy wzrost do64,12/80,92 pochodzi z ramienia z dodatkowymi historycznymi N/P, które nie są samymi cechami obrazów. Nie przypisujemy całego wyniku obrazowemu sterowaniu. Stary sequentialEDMD+images miał silne negatywy; w tym etapie poprawiono również lifting, podział i czas fit, więc różnica nie izoluje przyczynowo samej metodyJOINT.

## Późny s47 i identyfikowalność

| H | Redukcja MSE % | Znak % | Balanced % | Ridge | Ranga/kolumny | Uwarunkowanie |
|---|---:|---:|---:|---:|---|---:|
| 1 | 59.12 | 79.04 | 78.99 | 0.1 | 60/60 | 6.25e+01 |
| 3 | 36.83 | 71.31 | 71.28 | 0.1 | 60/60 | 6.25e+01 |
| 5 | 22.42 | 65.51 | 65.49 | 0.1 | 60/60 | 6.25e+01 |
| 10 | 14.89 | 61.39 | 61.58 | 0.1 | 42/42 | 6.73e+00 |
| 20 | 22.76 | 64.51 | 64.41 | 0.1 | 60/60 | 6.25e+01 |
| 50 | 24.86 | 66.06 | 65.88 | 0.001 | 60/60 | 6.25e+01 |

Ranga macierzy dotyczy zdefiniowanych regresorów przy danej normalizacji, nie obserwowalności pełnego ViT. Wielokrotne sondy współdzielą wejście i bieg; liczba wierszy nie jest liczbą niezależnych replik. Pełna ranga numeryczna nie dowodzi rozdziału mechanizmów, identyfikowalności semantycznej ani reakcji na interwencję. Ridge stabilizuje estymację, nie tworzy brakującego wzbudzenia. [PrzeglądAstry](../../PRZEGLAD_ASTRA_STEROWANIE.md) wyjaśnia także poprawną residualizacjęFWL/ridge.

## Diagnostyka reszt i granice

H1s48 wybranego opt: skośność−0,019, nadmiarowa kurtoza5,94, korelacja błędu² z |prognozą|0,224. ACF reszt dla początków oddalonych o5/10/25 aktualizacji:−0,036/−0,002/+0,006. Niska ACF kilku lagów nie dowodzi białości; duża kurtoza przeczy prostemu założeniu dokładnie gaussowskiego szumu. H>1 etykiety zachodzą na siebie, co samo może wprowadzać korelacje. Globalny przedział90 z walidacji pokrywa87,70 procent H1s48: kalibracja nie jest idealna. Reszty służą diagnozie niedomknięcia, bezFDT dla AdamW.

Wszystkie36 kontroli prefiksowych przechodzą, w tym rollout control bez źródłowych przyszłych wejść. Sprawdzenie wejścia przyszłego jest kodowe, nie domniemane z nazwy wariantu. Zachowano wszystkie ramiona i negatywy. Rozkład/ACF nie ustanawia skutecznego odzyskania ukrytego sygnału; świeży odłożony bieg i wspólny encoder obrazów pozostają potrzebne.

## Wszystkie ramiona s48

| H | Model | Redukcja MSE % | Znak % | Balanced % |
|---|---|---:|---:|---:|
| 1 | only | 0.42 | 51.61 | 51.61 |
| 1 | memory | 62.31 | 79.85 | 79.85 |
| 1 | images | 62.43 | 79.77 | 79.77 |
| 1 | simple | 61.63 | 79.34 | 79.34 |
| 1 | opt | 64.12 | 80.92 | 80.92 |
| 1 | controlled_images | 62.50 | 79.93 | 79.93 |
| 1 | zero | 0.00 | 0.00 | 0.00 |
| 1 | majority | -3948.97 | 49.73 | 49.73 |
| 3 | only | 3.84 | 55.96 | 55.96 |
| 3 | memory | 40.34 | 72.76 | 72.76 |
| 3 | images | 39.56 | 72.52 | 72.54 |
| 3 | simple | 39.51 | 72.62 | 72.58 |
| 3 | opt | 40.38 | 72.84 | 72.89 |
| 3 | controlled_images | 39.74 | 72.67 | 72.65 |
| 3 | zero | 0.00 | 0.00 | 0.00 |
| 3 | majority | -599.59 | 48.24 | 48.16 |
| 5 | only | 8.01 | 58.44 | 58.44 |
| 5 | memory | 25.78 | 67.39 | 67.39 |
| 5 | images | 25.48 | 67.27 | 67.29 |
| 5 | simple | 25.14 | 66.56 | 66.45 |
| 5 | opt | 25.67 | 67.20 | 67.25 |
| 5 | controlled_images | 25.60 | 67.28 | 67.30 |
| 5 | zero | 0.00 | 0.00 | 0.00 |
| 5 | majority | -305.38 | 47.16 | 47.10 |
| 10 | only | 16.95 | 63.20 | 63.20 |
| 10 | memory | 21.86 | 65.19 | 65.19 |
| 10 | images | 21.59 | 64.60 | 64.66 |
| 10 | simple | 21.57 | 64.80 | 64.84 |
| 10 | opt | 21.82 | 64.92 | 64.95 |
| 10 | controlled_images | 20.99 | 64.52 | 64.56 |
| 10 | zero | 0.00 | 0.00 | 0.00 |
| 10 | majority | -155.60 | 46.19 | 46.18 |
| 20 | only | 25.80 | 67.33 | 67.32 |
| 20 | memory | 28.86 | 68.21 | 68.20 |
| 20 | images | 22.02 | 64.05 | 63.89 |
| 20 | simple | 29.21 | 67.76 | 67.72 |
| 20 | opt | 22.95 | 64.73 | 64.58 |
| 20 | controlled_images | 26.75 | 67.25 | 67.18 |
| 20 | zero | 0.00 | 0.00 | 0.00 |
| 20 | majority | -103.06 | 46.29 | 46.31 |
| 50 | only | 30.25 | 68.45 | 68.46 |
| 50 | memory | 30.80 | 68.67 | 68.68 |
| 50 | images | 29.36 | 67.99 | 67.74 |
| 50 | simple | 26.03 | 66.62 | 66.87 |
| 50 | opt | 29.74 | 68.17 | 67.95 |
| 50 | controlled_images | 29.35 | 68.37 | 68.17 |
| 50 | zero | 0.00 | 0.00 | 0.00 |
| 50 | majority | -94.86 | 44.10 | 44.14 |

## Artefakty

METRYKI.json: wybór,val,test,diagnostyka,konfiguracje i stare porównania bezKF. PROGNOZY.npz: wszystkie przewidywania i cele. MODELE.npz: bloki operatorów i preprocessing. PROG80.json: maksymalne zmierzoneH. KRZYWE.png: porównanie horyzontów. [Kod](../../kod/joint_controlled.py). CPU25.54s, bez treninguViT, GPU, Edenu i commitów.

![Krzywe](KRZYWE.png)

## Zapisany EDMD bezKF na identycznych celach s48

| H | Model | Redukcja MSE % | Znak % |
|---|---|---:|---:|
| 1 | fixed_only | 0.34 | 51.52 |
| 1 | fixed_memory | 62.13 | 79.72 |
| 1 | learned_only | 0.30 | 51.45 |
| 3 | fixed_only | 3.56 | 55.69 |
| 3 | fixed_memory | 39.56 | 72.50 |
| 3 | learned_only | 3.41 | 55.68 |
| 5 | fixed_only | 7.60 | 58.22 |
| 5 | fixed_memory | 24.09 | 66.15 |
| 5 | learned_only | 7.35 | 58.37 |
| 10 | fixed_only | 16.24 | 62.89 |
| 10 | fixed_memory | 20.19 | 64.04 |
| 10 | learned_only | 15.93 | 63.03 |
| 20 | fixed_only | 24.81 | 66.83 |
| 20 | fixed_memory | 27.15 | 67.09 |
| 20 | learned_only | 24.70 | 67.06 |
| 50 | fixed_only | 27.73 | 67.12 |
| 50 | fixed_memory | 28.53 | 67.73 |
| 50 | learned_only | 28.32 | 67.40 |

To zachowane ramiona, nie nowy wybór po teście. Szczegóły wcześniejszego podziału train/drift są inne, dlatego nie izolujemy skutku jednej zmiany algorytmu.

Diagnostyka korelacji średniej reszty z ostatnimi wykonanymi obrazami (PCA4) i błędu po kwartylach amplitudy prognozy: DIAGNOSTYKA_INPUT.json. Korelacja nie identyfikuje przyczyny ani niezależności. [Pełne4 źródła i119 wyników DeepSearch](../../zrodla/LITERATURA_JOINT.md).
