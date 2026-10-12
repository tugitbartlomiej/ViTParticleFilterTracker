# Pilot bliższego zapisu: wynik eksploracyjny

2026-10-01T14:58:56.737241+02:00 (Europe/Warsaw).

Stały skalar LDA, a nie pełny wektor768. Stan co1 aktualizację; H podano w aktualizacjach. S48 wcześniej obejrzano; wynik nie jest nowym niezależnym potwierdzeniem. Wybór wariantu przez MSE walidacji s47, bez przyszłych paczek.

| H | Wybrany model | RedukcjaMSE s48 [%] | Znak wszystkie [%] | Znak zbilansowany [%] | Większość z przeszłości [%] | Pokrycie90 [%] |
|---:|---|---:|---:|---:|---:|---:|
| 1 | koopman_lift_gle20_input | 64.96 | 80.98 | 80.98 | 49.73 | 88.2 |
| 3 | koopman_lift_gle20_input | 41.73 | 73.42 | 73.42 | 48.24 | 88.6 |
| 5 | koopman_lift_gle20_input | 26.76 | 67.67 | 67.66 | 47.16 | 90.1 |
| 10 | gle100_input | 22.13 | 65.07 | 65.07 | 46.19 | 90.1 |
| 20 | gle20_input_desc | 18.04 | 63.26 | 63.07 | 46.29 | 84.7 |
| 50 | old_langevin20 | 31.13 | 68.80 | 68.82 | 44.10 | 88.9 |

Wszystkie ramiona, MAE/RMSE, pomocnicza połowa dużych zmian, reszty i wynik s47: METRYKI.json. Progi80 są sprawdzone osobno w threshold80_s48. Cel80 nie jest gwarancją. Rzeczywista paczka powodująca cel jest niedostępna. Pchnięcia przesunięto do stanu po wykonaniu; u[t] odpowiada źródłowemu wierszowi t-1. Dryf lokalny/lifting x,x²,x*v,v² jest aproksymacją; 20 lub100 opóźnień przyrostów daje pamięć, a historyczne N/P i deskryptory osobne wejścia. Przyszłe wymuszenie jest uwzględniane przez bezpośrednią regresję warunkowej zmiany, nie przez podgląd przyszłych danych.

Cele H>5 nakładają się w diagnostyce co5, więc ich ACF nie dowodzi kolorowego szumu pojedynczego kroku. Przedziały kalibrowano historycznymi błędami walidacji; brak gwarancji IID. Dwa biegi i zależne sondy nie stanowią wieloziarnowego potwierdzenia. Znany kolejny batch z dawnych eksperymentów nie spełnia tego zadania. Nie badano semantycznej swoistości innych narzędzi/tła ani przyszłych zmian wszystkich składowych. Te etapy są zaplanowane w PLAN.md; wymagają zdefiniowanej podprzestrzeni i dodatkowych kontroli.

Nie uruchomiono nowego treningu, Edenu ani pobierania tokenów. WynikiH500 pozostały w poprzednich plikach.

## Jawne porównanie rodzin na s48

Samo lokalne przybliżenie Koopmana to bezpośredni model stanu x lub liftingu x,x²; nie jest dokładnym operatorem pełnej sieci. Dawny Langevin20 oznacza bazę20 przyrostów plus odchylenie od średniej500, bez wyrazu wolnego, ridge1e-6, ponownie dopasowaną na wspólnym podziale. H>1 to dawny wariant bezpośredni, nie rekurencyjny rollout. Wszystkie warianty mają ten sam cel i dostęp tylko do historii.

| H | Rodzina | Redukcja MSE [%] | Znak wszystkie [%] | Balanced [%] |
|---:|---|---:|---:|---:|
| 1 | zero | 0.00 | 0.00 | 0.00 |
| 1 | old_langevin20 | 62.40 | 79.86 | 79.86 |
| 1 | koopman_state | 0.38 | 51.65 | 51.65 |
| 1 | koopman_state_lift | 0.42 | 51.51 | 51.51 |
| 1 | koopman_lift_gle20_input | 64.96 | 80.98 | 80.98 |
| 3 | zero | 0.00 | 0.00 | 0.00 |
| 3 | old_langevin20 | 40.26 | 72.86 | 72.86 |
| 3 | koopman_state | 3.67 | 55.96 | 55.97 |
| 3 | koopman_state_lift | 3.80 | 56.04 | 56.03 |
| 3 | koopman_lift_gle20_input | 41.73 | 73.42 | 73.42 |
| 5 | zero | 0.00 | 0.00 | 0.00 |
| 5 | old_langevin20 | 25.67 | 67.53 | 67.53 |
| 5 | koopman_state | 7.83 | 58.44 | 58.44 |
| 5 | koopman_state_lift | 7.95 | 58.51 | 58.49 |
| 5 | koopman_lift_gle20_input | 26.76 | 67.67 | 67.66 |
| 10 | zero | 0.00 | 0.00 | 0.00 |
| 10 | old_langevin20 | 21.79 | 65.20 | 65.20 |
| 10 | koopman_state | 16.71 | 63.17 | 63.17 |
| 10 | koopman_state_lift | 16.89 | 63.12 | 63.13 |
| 10 | koopman_lift_gle20_input | 22.13 | 65.15 | 65.16 |
| 20 | zero | 0.00 | 0.00 | 0.00 |
| 20 | old_langevin20 | 28.99 | 68.18 | 68.18 |
| 20 | koopman_state | 25.71 | 67.32 | 67.32 |
| 20 | koopman_state_lift | 25.81 | 67.25 | 67.24 |
| 20 | koopman_lift_gle20_input | 28.35 | 67.57 | 67.56 |
| 50 | zero | 0.00 | 0.00 | 0.00 |
| 50 | old_langevin20 | 31.13 | 68.80 | 68.82 |
| 50 | koopman_state | 30.23 | 68.46 | 68.47 |
| 50 | koopman_state_lift | 30.25 | 68.45 | 68.46 |
| 50 | koopman_lift_gle20_input | 30.12 | 68.47 | 68.48 |

Na H1 walidacja wybrała Koopmanowski lifting + pamięćGLE20 + historyczne wpływy. Wynik s48:80,98procent znaków wszystkich sond i zbilansowanej trafności,64,96procent redukcjiMSE; kontrola większości49,73procent znaków. Późny s47:79,24procent znaków i59,50procent redukcjiMSE. Zatem80procent występuje w jednej mierze i jednym biegu eksploracyjnym, nie jako potwierdzona wieloziarnowa poprawa. Dla następnego zapisu sond co10 aktualizacji wynik znaku wynosi65,07procent; nie spełnia celu80. Sam Koopman bez pamięci nie otrzymuje z góry preferencji.

Do potwierdzenia potrzebny świeży odłożony przebieg po zamrożeniu modelu, miary celu80 i horyzontu konkretnego zapisu. Wielowymiarowa prognoza składowych tożsamości i kontrola semantyki pozostają osobnym planowanym etapem.
