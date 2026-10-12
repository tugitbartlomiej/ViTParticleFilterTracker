# Pilot bliższego zapisu: wynik eksploracyjny

2026-10-01T14:56:39.411150+02:00 (Europe/Warsaw).

Stały skalar LDA, a nie pełny wektor768. Stan co1 aktualizację; H podano w aktualizacjach. S48 wcześniej obejrzano; wynik nie jest nowym niezależnym potwierdzeniem. Wybór wariantu przez MSE walidacji s47, bez przyszłych paczek.

| H | Wybrany model | RedukcjaMSE s48 [%] | Znak wszystkie [%] | Znak zbilansowany [%] | Większość z przeszłości [%] | Pokrycie90 [%] |
|---:|---|---:|---:|---:|---:|---:|
| 1 | koopman_lift_gle20_input | 64.96 | 80.98 | 80.98 | 49.73 | 88.2 |
| 3 | koopman_lift_gle20_input | 41.73 | 73.42 | 73.42 | 48.24 | 88.6 |
| 5 | koopman_lift_gle20_input | 26.76 | 67.67 | 67.66 | 47.16 | 90.1 |
| 10 | gle100_input | 22.13 | 65.07 | 65.07 | 46.19 | 90.1 |
| 20 | gle20_input_desc | 18.04 | 63.26 | 63.07 | 46.29 | 84.7 |
| 50 | gle20_input_desc | 30.95 | 68.49 | 68.37 | 44.10 | 88.8 |

Wszystkie ramiona, MAE/RMSE, pomocnicza połowa dużych zmian, reszty i wynik s47: METRYKI.json. Progi80 są sprawdzone osobno w threshold80_s48. Cel80 nie jest gwarancją. Rzeczywista paczka powodująca cel jest niedostępna. Pchnięcia przesunięto do stanu po wykonaniu; u[t] odpowiada źródłowemu wierszowi t-1. Dryf lokalny/lifting x,x²,x*v,v² jest aproksymacją; 20 lub100 opóźnień przyrostów daje pamięć, a historyczne N/P i deskryptory osobne wejścia. Przyszłe wymuszenie jest uwzględniane przez bezpośrednią regresję warunkowej zmiany, nie przez podgląd przyszłych danych.

Cele H>5 nakładają się w diagnostyce co5, więc ich ACF nie dowodzi kolorowego szumu pojedynczego kroku. Przedziały kalibrowano historycznymi błędami walidacji; brak gwarancji IID. Dwa biegi i zależne sondy nie stanowią wieloziarnowego potwierdzenia. Znany kolejny batch z dawnych eksperymentów nie spełnia tego zadania. Nie badano semantycznej swoistości innych narzędzi/tła ani przyszłych zmian wszystkich składowych. Te etapy są zaplanowane w PLAN.md; wymagają zdefiniowanej podprzestrzeni i dodatkowych kontroli.

Nie uruchomiono nowego treningu, Edenu ani pobierania tokenów. WynikiH500 pozostały w poprzednich plikach.
