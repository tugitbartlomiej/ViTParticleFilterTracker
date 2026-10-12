# Ruch m0011 w stronę zgodnego sztucznego celu K8000

Sztuczny cel został wyznaczony ponownie z dokładnego checkpointa K8000 użytego jako początek obu gałęzi treningu. Korekta zmieniła wynik detekcji z IoU 0.050629 i confidence 0.845760 do IoU 0.997197 i confidence 0.905051.

- K8000 -> cel: RMSE 1.08359241.
- Kontrola po 500 aktualizacjach -> cel: RMSE 1.12169421.
- Wariant z 8 podmianami -> cel: RMSE 1.12151468.
- Czysty efekt podmiany, kontrola -> wariant: zmiana RMSE +0.00017953; cosine ruchu z kierunkiem do celu 0.02139676; zrealizowana część pozostałej drogi +0.07088%.

Wynik: względem kontroli wariant z ośmioma podmianami ma tylko minimalną składową ruchu w stronę celu. Obie wytrenowane gałęzie są jednak dalej od celu niż startowy K8000. Efekt podmiany jest prawie prostopadły do kierunku celu i realizuje około 0.07088% pozostałej drogi. Nie ma więc podstaw, aby twierdzić, że trening nauczył się kierunku sztucznego latentu. Jest to wynik jednej pary biegów i nie zastępuje powtórzeń na wielu seedach.
