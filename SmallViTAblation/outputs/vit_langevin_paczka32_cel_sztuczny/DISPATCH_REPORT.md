# Uruchomiony proces w tle

AKTUALIZACJA KOŃCOWA: proces wznowiony jako PID37328 po pomiarze pamięci zakończył pracę normalnie. Status complete, 35.51sekundy od wznowienia, oba ramiona po50 rzeczywistych aktualizacji. Nie pozostaje aktywna kolejka. Szczegóły liczb w RESULT_SUMMARY.md i RESULT.json. Pierwotny opis poniżej jest historią przygotowania, nie bieżącym stanem. Próg6500 zastąpiono2816MiB po pełnym smoke32 (peak reserved1660MiB plus ponad1GiB zapasu), bez mikrobatche i bez zmiany BN. Żaden cudzy proces nie został zatrzymany.

Stan zweryfikowany 2026-10-05: worker PID44748 działa. PROGRESS.json: queued_gpu, SD complete, wolne3154MiB wobec wymaganych6500MiB. Proces sam sprawdza co20sekund; nie kończy cudzych zadań. PID launchera34636 nie jest PID właściwego workera; WORKER_IDENTITY.json i PID.txt zawierają44748.

Przygotowanie zakończone: 32 unikalne train z pełnego rankingu sortowanego po mse768/ordinal; pierwszy TR_0006_S1_P03/001193. Każde z32 ma prawdziwą ramkę narzędzia. Kontrola losowa32 z train: 28 z narzędziem, cztery obrazy negatywne z prawidłowymi pustymi celami detektora. Wszystkie64 fotografie pobrane z rzeczywistych ZIP, zachowane CRC, SHA256, surowe wiersze adnotacji i ramki w oryginalnych/przeskalowanych współrzędnych. Nie użyto predykcji detektora jako GT.

Sonda29 TR_0154/1926 jest oddzielnym oryginalnym zdjęciem, zweryfikowanym hashem ze SELECTION.json celu. Wybrane zdjęcie treningowe nie jest podstawiane zamiast sondy. after/before/J128 zachowane w oryginalnym LATENT_PELNY.npz.

Autentyczny checkpoint Eden k1000 pobrany do tej rodziny: SHA256 e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6, zgodny ze zdalnym. Klucze net,opt,krok; 89 stanów parametrów AdamW; wszystkie tensory net identyczne z MODEL_VIT_K1000.pt celu. Oryginalna kopia celu miała tylko net,krok; nie sfabrykowano pędu.

Oryginalna definicja siły: skopiowane rownanie.py/prognoza_krok.py z maly_vit_sd_prowadzenie_tozsamoscia. S_nowa i S_ped to naturalny odczyt tej samej sondy przy osobnych wirtualnych przesunięciach wag, bez podmiany aktywacji. Oryginalna prognoza delta=S_nowa+S_ped oraz dodatkowa historycznie dopasowana nowa0.9838,ped0.9844 z zapisanego TOZSAMOSC_JEDEN_KROK.json.

Po zwolnieniu GPU program najpierw zapisze FORCE_selected32/random32.npz i FORECAST.json/csv z założeniem stałej siły początkowej. Progi uprzednio10/25/50procent redukcji MSE128 i MSE768, cosinus128/768, horyzont50; brak trafienia = null, nie arbitralna liczba epok. Epoka puli32 to jedna aktualizacja, nie epoka pełnego zbioru.

Następnie dwie niezależne kopie autentycznego stanu przejdą50 rzeczywistych kroków: wybrane32 vs stałe losowe32, dodatkowo brak zmiany. Prawdziwa oryginalna strata CenterNet+GIoU, aktualizacja wszystkich parametrów i BN; jawny protokół bez augmentacji. FIRST_STEP_VALIDATION.json oddzieli błąd prognozy wag przy starych BN od pełnego treningowego kroku z nowymi BN. Latenty zapis1/5/10/25/50, rzeczywiste hitting_time z wszystkich50 kroków. Przyszłych gradientów nie używa się do poprawiania prognozy.

Nie ma jeszcze wyniku siły, prognozy ani treningu; jest działająca kolejka po pełnym przygotowaniu i weryfikacji optymalizatora. Transfer historycznego równania ze średnich narzędzia na pełne600tokenów wymaga pierwszej weryfikacji. Ograniczenia zamrożonego checkpointu nie dowodzą globalnej nieosiągalności po uczeniu wag. Błąd zapisze BLOCKER.json/REPORT.md. Syntax i git diff --check przeszły; cudze modyfikacje zachowane, bez commitów. Pierwszy błąd kodowania CSV poprawiony, log zachowany w attempt01_encoding_failure.
