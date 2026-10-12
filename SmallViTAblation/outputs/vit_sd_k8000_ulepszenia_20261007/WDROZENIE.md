# Wdrożenie ulepszeń SD K8000

Użytkownik autoryzował wykonanie opisanych ulepszeń. Eksperyment jest odrębną rodziną; poprzedni zakończony bieg pozostaje niezmieniony.

## Wykonane

- Nowy sampler losuje jedną klatkę w każdym oknie 600 klatek i przeplata filmy w ograniczonej liczbie dekoderów RAM.
- Ewaluacja agreguje wyniki kolejno po seedach, klatkach i filmach. Zachowuje brak metryk jako brak danych, nie jako zero.
- Diagnoza odszumiania porównuje sześć konfiguracji. Generacje i latenty nie trafiają na dysk.
- Cztery warianty startują z własnego najlepszego modelu z kroku 7500, przy zamrożonym VAE i ViT K8000.
- Testy CPU: 49 testów zakończonych powodzeniem. Log: TESTY_CPU.log.
- Niezależny przegląd integracji wskazał cztery problemy. Poprawiono normalizację ROI po globalnej liczbie poprawnych etykiet, odbudowanie FINAL po przerwanym zapisie, obsługę STOP_TIME między wariantami oraz żądanie łagodnego zatrzymania zamiast natychmiastowego zabijania procesu nadrzędnego. Testy regresji obejmują normalizację, odtworzenie FINAL i sygnał SIGTERM oraz zatrzymanie między wariantami.
- Pierwszy smoke 1823542 potwierdził działanie diagnozy na GPU; zatrzymano go kontrolowanym plikiem STOP, ponieważ po przeglądzie trzeba powtórzyć próbę z poprawionymi źródłami. To nie wynik jakości ani ukończony trening.

## Uruchomienie pełnego porównania

- Smoke **1823543** zakończony: SLURM COMPLETED, exit 0:0. Wszystkie cztery warianty wykonały po dwie aktualizacje; zapisane best checkpointy zostały świeżo wczytane i zweryfikowane. W D_roi rzeczywiście wykonano niezerową stratę ROI (0,06337 w pierwszym kroku rank 0).
- SMOKE_OK potwierdza zgodność źródeł na Edenie, a dodatkowy lokalny pomiar potwierdził zgodność wszystkich 18 hashy plików kodu, dokumentacji i skryptu SLURM z przetestowaną wersją.
- Pełne porównanie zgłoszone raz: **1823544**, `sdimpk8`, **RUNNING** od **2026-10-07 04:51:54 +02:00** na Hopper, **4 × H100**, 24 CPU, 160 GB RAM.
- Uruchomiona sekwencja: diagnostyka DDIM, następnie A_control, B_mixed, C_optim, D_roi. Każdy wariant ma budżet maksymalnie 4000 nowych kroków i własny najlepszy model.
- Wyniki na Edenie: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie`.
- Godzinny monitoring nowej rodziny jest aktywny. Normalny postęp nie generuje powiadomień; awarie, blokady i zakończenie tak.
- Dowody: `URUCHOMIENIE_ZWERYFIKOWANE.json`, `SUBMIT_PRODUKCJA.log`, `smoke_1823543/SMOKE_OK.json`, `smoke_1823543/COMPARISON.json`, `smoke_1823543/SLURM.log`.

## Pozostałe kroki

1. Poczekać na wyniki diagnozy i porównywalnej walidacji czterech wariantów.
2. Porównać błąd pełnego latentu, ROI, działanie warunkowania i wygląd obrazów. Nie traktować wyboru minimum jednej metryki jako automatycznej oceny wizualnej.
3. Zweryfikować najlepszy model lokalnie i posprzątać zbędne zapisy po zabezpieczeniu modeli. Zachować recovery niedokończonych przebiegów.

Tylko rzeczywiste pomiary pozwalają stwierdzić poprawę. Sam test smoke jej nie potwierdza. Nowe modele nie były inicjalizowane cudzym wytrenowanym SD: wszystkie dziedziczą wagi własnego eksperymentu uczonego od zera.
