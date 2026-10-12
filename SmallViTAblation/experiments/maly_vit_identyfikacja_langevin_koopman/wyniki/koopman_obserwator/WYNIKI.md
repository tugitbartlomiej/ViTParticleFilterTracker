# Pilot EDMD/learned lifting i obserwatora: wynik eksploracyjny

2026-10-01T15:19:57.093644+02:00

S48 wcześniej oglądano. Wybór wyłącznie na walidacji s47. Stan jest skalaremLDA z poolingu, nie trajektorią cząsteczek.

| H | Wybór walidacji | RedukcjaMSE s48 [%] | Znak [%] | Balanced [%] | Dawny Langevin MSE [%] | Wcześniejszy direct lifting+GLE MSE [%] |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed_opt_kf | -53.08 | 57.06 | 57.06 | 62.40 | 64.96 |
| 3 | fixed_opt | -29.90 | 56.24 | 55.94 | 40.26 | 41.73 |
| 5 | fixed_opt | -199.14 | 50.96 | 50.55 | 25.67 | 26.76 |
| 10 | fixed_memory | 20.19 | 64.04 | 64.11 | 21.79 | 22.13 |
| 20 | learned_only | 24.70 | 67.06 | 67.10 | 28.99 | 28.35 |
| 50 | learned_images | -39.92 | 54.56 | 55.30 | 31.13 | 30.12 |

Maksymalne zmierzone H z progiem80: {"sign_percent": null, "balanced_sign_percent": null, "skill_percent": null}. Brak monotoniczności lub gwarancji. Pełne ramiona, KF i wyniki s47 wMETRYKI.json.

EDMD fituje CAŁY lifting celu przy referencji m_t zamrożonej na czas okna. Learned lifting uczy encoder/decoder i dynamikę; potem osobny K_H. KażdyH to inny operator lokalny, nie iteracjaK1 ani dowód semigrupy. Sekwencyjny podział dryf/pamięć/sterowanie jest umowną projekcją; drugi etap widzi chronologiczne reszty dryfu poza jego treningiem. Nie identyfikuje mechanizmu przyczynowego. Obrazowe sterowanie pochodzi z rzeczywistych paczka_lat, pooled narzędzie/całość i PCA8 z treningu; N/P są oddzielnymi wpływami. Przyszłe wejście jest marginalizowane empirycznie przez historyczną estymację, nie znane.

KF aktualizuje ukrytą AR1-resztę indeksowaną KOŃCEM etykiety, wykorzystując e_H(t−H). Prognozuje rho^H eta_t. Przy H>5 okna testu nakładają się, a także błędy pomiarowe mogą być kolorowe; Gaussian KF jest aproksymacją drugich momentów. Parametry wariancji z dopasowanych reszt mogą być optymistyczne. EKF/UKF nie uruchamiano; wymagają konkretnej realizacji i oceny modelu pomiaru. Nie dowiedziono obserwowalności/sterowalności ViT.

MałyAE na CPU18epok; nie trenowano ViT, nie użytoGPU/Edenu, nie pobierano22GB. Wszystkie negatywy zachowano. Świeży przebieg po zamrożeniu modelu i miary jest wymagany do potwierdzenia.

## Pełne porównanie poza dopasowaniem

| H | Ramię | Redukcja MSE [%] | Znak [%] | Balanced [%] |
|---:|---|---:|---:|---:|
| 1 | fixed_only | 0.34 | 51.52 | 51.52 |
| 1 | fixed_only_kf | 53.95 | 77.46 | 77.46 |
| 1 | fixed_memory | 62.13 | 79.72 | 79.72 |
| 1 | fixed_memory_kf | 62.12 | 79.72 | 79.72 |
| 1 | fixed_images | -127.61 | 53.65 | 53.66 |
| 1 | fixed_images_kf | -118.64 | 54.07 | 54.07 |
| 1 | fixed_opt | -60.44 | 56.38 | 56.38 |
| 1 | fixed_opt_kf | -53.08 | 57.06 | 57.06 |
| 1 | learned_only | 0.30 | 51.45 | 51.45 |
| 1 | learned_only_kf | 53.93 | 77.46 | 77.46 |
| 1 | learned_memory | 62.09 | 79.72 | 79.72 |
| 1 | learned_memory_kf | 62.08 | 79.69 | 79.69 |
| 1 | learned_images | -130.57 | 53.57 | 53.57 |
| 1 | learned_images_kf | -121.36 | 53.98 | 53.99 |
| 1 | learned_opt | -62.67 | 56.26 | 56.27 |
| 1 | learned_opt_kf | -55.14 | 56.94 | 56.94 |
| 1 | zero | 0.00 | 0.00 | 0.00 |
| 1 | majority_past | -3948.97 | 49.73 | 49.73 |
| 3 | fixed_only | 3.56 | 55.69 | 55.76 |
| 3 | fixed_only_kf | -5.19 | 44.69 | 44.73 |
| 3 | fixed_memory | 39.56 | 72.50 | 72.46 |
| 3 | fixed_memory_kf | 36.87 | 72.21 | 72.20 |
| 3 | fixed_images | -81.86 | 52.82 | 52.48 |
| 3 | fixed_images_kf | 23.16 | 64.41 | 64.20 |
| 3 | fixed_opt | -29.90 | 56.24 | 55.94 |
| 3 | fixed_opt_kf | 30.63 | 68.29 | 68.13 |
| 3 | learned_only | 3.41 | 55.68 | 55.73 |
| 3 | learned_only_kf | -5.59 | 44.72 | 44.75 |
| 3 | learned_memory | 39.45 | 72.47 | 72.44 |
| 3 | learned_memory_kf | 36.67 | 72.20 | 72.18 |
| 3 | learned_images | -85.45 | 52.68 | 52.34 |
| 3 | learned_images_kf | 22.58 | 64.20 | 64.00 |
| 3 | learned_opt | -32.14 | 56.09 | 55.79 |
| 3 | learned_opt_kf | 30.20 | 68.13 | 67.97 |
| 3 | zero | 0.00 | 0.00 | 0.00 |
| 3 | majority_past | -599.59 | 48.24 | 48.16 |
| 5 | fixed_only | 7.60 | 58.22 | 58.28 |
| 5 | fixed_only_kf | -4.36 | 44.32 | 44.39 |
| 5 | fixed_memory | 24.09 | 66.15 | 66.08 |
| 5 | fixed_memory_kf | 19.24 | 65.79 | 65.75 |
| 5 | fixed_images | -271.79 | 50.58 | 50.17 |
| 5 | fixed_images_kf | -17.18 | 52.78 | 52.40 |
| 5 | fixed_opt | -199.14 | 50.96 | 50.55 |
| 5 | fixed_opt_kf | -6.96 | 54.80 | 54.45 |
| 5 | learned_only | 7.35 | 58.37 | 58.42 |
| 5 | learned_only_kf | -5.14 | 43.92 | 43.97 |
| 5 | learned_memory | 23.93 | 66.11 | 66.04 |
| 5 | learned_memory_kf | 18.81 | 65.68 | 65.64 |
| 5 | learned_images | -278.98 | 50.56 | 50.14 |
| 5 | learned_images_kf | -18.47 | 52.67 | 52.29 |
| 5 | learned_opt | -204.13 | 50.88 | 50.47 |
| 5 | learned_opt_kf | -7.97 | 54.72 | 54.37 |
| 5 | zero | 0.00 | 0.00 | 0.00 |
| 5 | majority_past | -305.38 | 47.16 | 47.10 |
| 10 | fixed_only | 16.24 | 62.89 | 62.86 |
| 10 | fixed_only_kf | 3.79 | 55.89 | 55.86 |
| 10 | fixed_memory | 20.19 | 64.04 | 64.11 |
| 10 | fixed_memory_kf | 13.36 | 62.53 | 62.59 |
| 10 | fixed_images | -1.35 | 56.16 | 56.35 |
| 10 | fixed_images_kf | 3.12 | 54.27 | 54.45 |
| 10 | fixed_opt | -0.24 | 56.47 | 56.65 |
| 10 | fixed_opt_kf | 4.25 | 54.87 | 55.05 |
| 10 | learned_only | 15.93 | 63.03 | 63.00 |
| 10 | learned_only_kf | 2.59 | 55.38 | 55.35 |
| 10 | learned_memory | 19.97 | 64.06 | 64.12 |
| 10 | learned_memory_kf | 12.64 | 62.37 | 62.42 |
| 10 | learned_images | -2.48 | 55.97 | 56.15 |
| 10 | learned_images_kf | 1.93 | 53.87 | 54.05 |
| 10 | learned_opt | -1.34 | 56.32 | 56.50 |
| 10 | learned_opt_kf | 3.09 | 54.47 | 54.65 |
| 10 | zero | 0.00 | 0.00 | 0.00 |
| 10 | majority_past | -155.60 | 46.19 | 46.18 |
| 20 | fixed_only | 24.81 | 66.83 | 66.88 |
| 20 | fixed_only_kf | 17.17 | 64.90 | 64.94 |
| 20 | fixed_memory | 27.15 | 67.09 | 67.00 |
| 20 | fixed_memory_kf | 22.47 | 66.05 | 65.98 |
| 20 | fixed_images | -424.55 | 50.48 | 50.15 |
| 20 | fixed_images_kf | -99.20 | 50.86 | 50.54 |
| 20 | fixed_opt | -368.88 | 50.55 | 50.23 |
| 20 | fixed_opt_kf | -84.15 | 51.17 | 50.85 |
| 20 | learned_only | 24.70 | 67.06 | 67.10 |
| 20 | learned_only_kf | 16.03 | 64.79 | 64.82 |
| 20 | learned_memory | 27.12 | 67.16 | 67.07 |
| 20 | learned_memory_kf | 21.82 | 66.00 | 65.93 |
| 20 | learned_images | -436.83 | 50.33 | 50.00 |
| 20 | learned_images_kf | -103.19 | 50.71 | 50.39 |
| 20 | learned_opt | -378.39 | 50.37 | 50.04 |
| 20 | learned_opt_kf | -87.39 | 50.98 | 50.66 |
| 20 | zero | 0.00 | 0.00 | 0.00 |
| 20 | majority_past | -103.06 | 46.29 | 46.31 |
| 50 | fixed_only | 27.73 | 67.12 | 67.31 |
| 50 | fixed_only_kf | 27.13 | 67.31 | 67.49 |
| 50 | fixed_memory | 28.53 | 67.73 | 67.46 |
| 50 | fixed_memory_kf | 28.11 | 67.25 | 67.01 |
| 50 | fixed_images | -45.80 | 54.06 | 54.80 |
| 50 | fixed_images_kf | -13.06 | 55.51 | 56.22 |
| 50 | fixed_opt | -61.26 | 53.24 | 54.00 |
| 50 | fixed_opt_kf | -21.32 | 54.57 | 55.30 |
| 50 | learned_only | 28.32 | 67.40 | 67.56 |
| 50 | learned_only_kf | 26.94 | 67.59 | 67.73 |
| 50 | learned_memory | 28.76 | 67.77 | 67.51 |
| 50 | learned_memory_kf | 27.84 | 67.41 | 67.18 |
| 50 | learned_images | -39.92 | 54.56 | 55.30 |
| 50 | learned_images_kf | -10.16 | 55.97 | 56.66 |
| 50 | learned_opt | -55.81 | 53.62 | 54.37 |
| 50 | learned_opt_kf | -18.64 | 55.06 | 55.77 |
| 50 | zero | 0.00 | 0.00 | 0.00 |
| 50 | majority_past | -94.86 | 44.10 | 44.14 |

![Krzywe porównania](KRZYWE.png)

PCA cech obrazowych: mediana normy w standaryzowanych8 współrzędnych dla treningu s47 wynosi2.73, a dla s48 7.40. To zmierzone przesunięcie rozkładu cech. Cechy pochodzą z oddzielnie uczonych sieci i zmieniają się także podczas ich uczenia; nie udowodniono wspólnej semantycznej zgodności tych współrzędnych między biegami. Nie przypisujemy samej korelacji diagnostycznej roli jedynej przyczyny pogorszenia. Przyszły test powinien porównać wspólny zamrożony encoder obrazów albo kalibrację dopasowania reprezentacji z dostępnej przeszłości, wraz z kontrolą mniej zmiennych deskryptorów obrazu.

KF jest filtrem pojedynczej ukrytej reszty dołączonej do deterministycznego predyktora pamięci. Nie jest pełnym filtrem wszystkich wag, momentów AdamW,600 cząsteczek ani wspólnej niepewności wszystkich opóźnionych stanów. Dla przyjętego AR1 pomiar eta jest bezpośredni; ta obserwowalność skalarnego modelu nie przechodzi na pełną sieć. EKF/UKF i pełna rozszerzona realizacja pozostają przyszłym etapem.

Wybrany H1 jest gorszy od wcześniejszego direct modelu; w tym EDMD/AE pilotażu żaden walidacyjnie wybrany H nie osiągnął80procent. Nie zastępujemy go wariantem wybranym po obejrzeniu s48. Wcześniejszy wynik80,98procent znaków naH1 pozostaje rezultatem direct lifting+GLE+historycznych wpływów, nie EDMD.

MODELE_WYBRANE.npz przechowuje odtworzone zamrożone operatory, PCA, współczynniki i parametry KF. Rekonstrukcja prognoz zgadza się z pierwotnym PROGNOZY.npz do1e-5; DIAGNOSTYKA.json zapisuje rzeczywiste błędy odtworzenia. AE_cut1400.pt i AE_cut2000.pt są małymi wagami encoder/decoder (CPU), nie checkpointami ViT.
