# Semantyka ruchu — pilotaż

21 checkpointów (lub zakres uruchomienia: 21), osiem obrazów dopasowania, cztery obrazy sprawdzenia sond. Wszystkie pochodzą z treningu detektora. Czas: 54.2 s.

IoU dotyczy ramki o najwyższej pewności dla pojedynczego narzędzia; nie jest mAP. Podprzestrzenie mają po osiem kierunków i mogą się nakładać.

- pozycja: średni udział ruchu 0.253; korelacja z ΔIoU -0.065; po usunięciu średniej dla każdego kroku -0.040; efekt cofnięcia IoU -0.0039; losowa kontrola o tej samej normie -0.0003.
- kolor: średni udział ruchu 0.248; korelacja z ΔIoU -0.092; po usunięciu średniej dla każdego kroku -0.075; efekt cofnięcia IoU -0.0038; losowa kontrola o tej samej normie -0.0003.
- proporcje_ksztaltu: średni udział ruchu 0.280; korelacja z ΔIoU -0.071; po usunięciu średniej dla każdego kroku 0.000; efekt cofnięcia IoU -0.0000; losowa kontrola o tej samej normie -0.0003.

Dodatni efekt cofnięcia oznacza, że usunięcie tej części ruchu pogarsza detekcję. Ujemny oznacza poprawę po ingerencji. Interpretację semantyczną dopuszcza dopiero selektywność sond, a nie ich nazwa. Korelacje są opisowe, nie wyliczamy naiwnych wartości istotności dla zależnych kroków.

Kształt oznacza tu zmianę proporcji całego kadru; tło też się zmienia. Kolor dotyczy maski narzędzia. Pozycja oznacza cyfrową translację, nie pełny ruch kamery 3D. Ruch jest mierzony na niezmienionym obrazie w pełnych 768 wymiarach, a nie w rzucie Fibonacciego.

Interwencja cofa część zmiany ostatniej gałęzi MLP, z bazą wyznaczoną na wcześniejszym checkpointcie. Nie jest odwróceniem uczenia całej sieci. Wartości mogą wyjść poza rozkład aktywacji. Jedna trajektoria treningu i cztery obrazy nie uzasadniają jeszcze twierdzenia publikacyjnego.

Pliki: MOTION.csv, PROBES.csv, DETECTION.csv, OVERLAP.csv, STATES_BASES.npz, RESULT.json, DIAGNOSTICS.png, PERTURBATIONS.png.
## Co wynika z danych

Sondy nie dają obecnie czystego podziału na semantyczne moduły. Średnie nakładanie się podprzestrzeni (suma kwadratów cosinusów kątów głównych / 8) wynosi 0.54 dla koloru i pozycji, 0.55 dla koloru i proporcji oraz 0.68 dla pozycji i proporcji. Nazwa sondy opisuje sposób jej wyznaczenia, nie udowodnione znaczenie każdego kierunku.

- pozycja: własna odpowiedź przechwycona w 0.498, najsilniejsza obca w 0.370. Efekt minus kontrola normy: średnia -0.00356, mediana 0.00000; bez kroku 500 średnia 0.00021; bez obrazu 11 0.00023.
- kolor: własna odpowiedź przechwycona w 0.378, najsilniejsza obca w 0.412. Efekt minus kontrola normy: średnia -0.00345, mediana 0.00000; bez kroku 500 średnia 0.00033; bez obrazu 11 0.00037.
- proporcje_ksztaltu: własna odpowiedź przechwycona w 0.378, najsilniejsza obca w 0.465. Efekt minus kontrola normy: średnia 0.00028, mediana 0.00000; bez kroku 500 średnia 0.00028; bez obrazu 11 0.00018.

Sonda pozycji jest częściowo selektywna; sondy koloru i proporcji nie przechwytują swojej perturbacji mocniej niż każdej obcej. Korelacje udziału ruchu z przyrostem IoU są bliskie zeru. Nie ma podstaw do wniosku, że poprawa detekcji wynika przede wszystkim z jednego z trzech czynników.

Największy efekt ingerencji pochodzi z pojedynczego obrazu 11 na kroku 500: IoU najwyżej punktowanej ramki zmienia się z 0.0000 na 0.3028. Względna norma edycji aktywacji: 0.16244. Ranking ramek jest nieciągły, dlatego nie należy utożsamiać tego skoku z globalną poprawą reprezentacji. Pełne ramki i pewności zapisano w INTERPRETATION.json.

Weryfikacja QA.json: zgodność rzutu z niezależnym obliczeniem z archiwum, ortonormalność baz, brak zmian koloru poza maską, dokładność cyfrowej translacji, identyczny wynik zerowej interwencji.

Dalszy eksperyment powinien użyć bardziej selektywnych sond (wyodrębnienie wspólnej odpowiedzi i model kontrastujący czynniki), izolowanej deformacji narzędzia z kontrolą artefaktów, kilku przebiegów treningu i nowych obrazów. Dopiero potem można przypisywać składowym ruchu znaczenie semantyczne.
