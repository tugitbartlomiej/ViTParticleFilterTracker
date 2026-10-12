# Wdrożenie lokalnych przyrostów na Hopper-2

Aktualizacja: 2026-10-02, Europe/Warsaw. Nowy model i jego protokół: [PLAN_MODELU.md](PLAN_MODELU.md). Dawny job1819574 jest CANCELLED i nie będzie wznawiany.

Zadanie żąda 4 pełnych H200, 4 MIG4g.71gb i3 MIG3g.71gb, na jednym Hopper-2. To11 urządzeń, a nie11 pełnych kart. Globalny batch32 i końcowy28 rozdziela się przez array_split, gradienty są sumowane z wagą globalnej liczby obrazów. Każda z1500 klatek zawiera600 cząsteczek i pełne768 współrzędnych. Wszystkie ranki używają przydzielonego UUID, widzą jedno urządzenie cuda0. Żadne obce zadanie ani niewidoczny MIG nie jest używany.

NCCL2.32.3 jest oddzielną biblioteką w katalogu runu, ładowaną wyłącznie przez LD_PRELOAD; sharedvenv/Torch pozostają niezmienione. Driver595.71.05/CUDA13.2. NCCL_MNNVL_ENABLE=0. Guard1819911 potwierdził rzeczywisty runtime23203 i SUM66 dla11 ranków. Modelguard1819912 porównał1GPU z11 dla batch32/tail28/all600: PASS, gradienty/Adam/rollout/telemetria/checkpoint. Dane tego testu były syntetyczne, nie fit1500.

Pełny job:24h,44CPU,384GB RAM. Scratch wybierany przez rzeczywiste mkdir/writeprobe we własnym katalogu. Full101 wymaga ponad350GB wolnego i w tmpfs co najmniej384GB przydzielonej pamięci; inaczej rolling13. Home/network nie służy jako cache. Właściciel cleanup sprawdza dokładny resolvedpath, nazwę własnego joba i marker owner. Cache zawiera wyłącznie stany żądane przez bieżącą fazę; pojemnośćfull101 nie oznacza odczytu przyszłego testu przed freeze.

Przed fitem przygotowanie weryfikuje exact1500 i101 checkpointów; wieloprocesowy readerpreflight na cp0..10 sprawdza wszystkie ranki i zgodność do źródła. Driver następnie przeprowadza guard na rzeczywistym oknie50: batch32/28, all600,1vs11, logging/checkpoint. Tylko PASS pozwala przejść do fitu; model/seed są resetowane. Kroki preflight oddzielone od actual_training_steps. Inner50..70, val80→90, refit50..80, freeze, test90→100. Nie ma futureaux/gradientów jako wejścia. AdamWmomenty są rzeczywiste tylko co100 aktualizacji, historycznie holdlast z age/mask.

Wynik jakości nie jest jeszcze znany. Uruchomienie jest eksperymentem, nie gwarancją trafności. Czasy syntetycznego guardu są zimnymi krokami i nie są ETA całego treningu.
