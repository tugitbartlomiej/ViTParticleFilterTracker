# Semantyka ruchu — pilotaż

21 checkpointów (lub zakres uruchomienia: 2), osiem obrazów dopasowania, cztery obrazy sprawdzenia sond. Wszystkie pochodzą z treningu detektora. Czas: 9.1 s.

IoU dotyczy ramki o najwyższej pewności dla pojedynczego narzędzia; nie jest mAP. Podprzestrzenie mają po osiem kierunków i mogą się nakładać.

- pozycja: średni udział ruchu 0.050; korelacja z ΔIoU 0.400; po usunięciu średniej dla każdego kroku 0.400; efekt cofnięcia IoU 0.0000; losowa kontrola o tej samej normie -0.0030.
- kolor: średni udział ruchu 0.038; korelacja z ΔIoU 0.400; po usunięciu średniej dla każdego kroku 0.400; efekt cofnięcia IoU 0.0002; losowa kontrola o tej samej normie -0.0030.
- proporcje_ksztaltu: średni udział ruchu 0.066; korelacja z ΔIoU 0.800; po usunięciu średniej dla każdego kroku 0.800; efekt cofnięcia IoU -0.0000; losowa kontrola o tej samej normie -0.0030.

Dodatni efekt cofnięcia oznacza, że usunięcie tej części ruchu pogarsza detekcję. Ujemny oznacza poprawę po ingerencji. Interpretację semantyczną dopuszcza dopiero selektywność sond, a nie ich nazwa. Korelacje są opisowe, nie wyliczamy naiwnych wartości istotności dla zależnych kroków.

Kształt oznacza tu zmianę proporcji całego kadru; tło też się zmienia. Kolor dotyczy maski narzędzia. Pozycja oznacza cyfrową translację, nie pełny ruch kamery 3D. Ruch jest mierzony na niezmienionym obrazie w pełnych 768 wymiarach, a nie w rzucie Fibonacciego.

Interwencja cofa część zmiany ostatniej gałęzi MLP, z bazą wyznaczoną na wcześniejszym checkpointcie. Nie jest odwróceniem uczenia całej sieci. Wartości mogą wyjść poza rozkład aktywacji. Jedna trajektoria treningu i cztery obrazy nie uzasadniają jeszcze twierdzenia publikacyjnego.

Pliki: MOTION.csv, PROBES.csv, DETECTION.csv, OVERLAP.csv, STATES_BASES.npz, RESULT.json, DIAGNOSTICS.png, PERTURBATIONS.png.