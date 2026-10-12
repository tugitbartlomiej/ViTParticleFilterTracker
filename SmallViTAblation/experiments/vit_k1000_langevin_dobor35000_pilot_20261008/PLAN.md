# Czy dobór paczki przybliża naturalny latent do starego celu K1000?

Pierwszy etap jest próbą lokalną: 256 prawdziwych oznaczonych obrazów z manifestu35000, równomiernie między filmami, w32 paczkach po8. Film sondy wykluczony. Wszystkie paczki oceniane przy identycznych wagach i autentycznym stanie AdamW K1000. Oryginalny checkpoint pozostaje niezmieniony.

Prognoza klasyczna rozdziela gradient i pęd. Druga prognoza sumuje ich zmiany wag oraz decay przed obliczeniem latentu. Obie używają wyłącznie obecnego stanu. Prognozy zapisane przed rzeczywistymi krokami na kopii stanu. Gradienty i forward FP32. Rzeczywisty krok aktualizuje także BatchNorm. Zapis tylko identyfikatorów, kodu i metryk, bez klatek i latentów.

Mierzymy J128, pozostałe640 i pełne768, confidence, IoU względem oryginalnej etykiety, trafność rankingu i redukcję błędu prognozy względem braku ruchu. Losowa kontrola: oczekiwany wynik równomiernego losowania z tych samych32 paczek. Nie nazywamy tej kontroli niezależną walidacją.

Pilot może obalić użyteczność prognozy. Nie stanowi wyboru1500 zdjęć ze35000 ani dowodu globalnie najlepszej kolejności. Dalsze skanowanie wymaga najpierw potwierdzenia trafności i kosztu; po zmianie stanu modelu ranking traci aktualność.

## Etap rozszerzony

Po poprawnie zakończonym pilocie uruchomiono `pilot.py --full`. Obejmuje wszystkie34701 rekordów pozostałych po wykluczeniu filmu sondy, w4338 ustalonych paczkach. Wyniki trafiają do odrębnego podfolderu `skan_puli35000`. Skład batchy jest deterministyczny; to nie optymalizacja kombinatoryczna wszystkich możliwych paczek. Wynik1500 kandydatów to prefiks rankingu przy zamrożonymK1000, nie potwierdzona sekwencja wielokrokowa. Mierzymy także liczbę paczek rzeczywiście przybliżających do celu i nie zakładamy, że takie paczki istnieją.
