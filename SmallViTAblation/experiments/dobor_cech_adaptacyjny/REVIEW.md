# Przegląd przed wykonaniem na Edenie

Przegląd obejmował moduły naukowe, adapter RF-DETR, dane, pętlę uczenia i skrypty Slurm. Niezależni recenzenci czytali kod; nie uruchamiali testów ani modelu. API sprawdzono w źródłach RF-DETR 1.10.1 z istniejącego środowiska na Edenie.

Poprawione ustalenia:

1. RF-DETR dla jednej klasy tworzy dwa kanały głowicy, włącznie z dodatkowym slotem. Guardy są zgodne z tą konwencją.
2. Słaby odczyt patchy wymaga jednego bbox; przygotowanie FIT/D odrzuca klatki puste i wieloobiektowe zamiast akceptować je i przerywać dopiero podczas GLE.
3. Hashowanie protokołu obejmuje również właściwe pliki COCO walidacji w podkatalogach.
4. Walidator wymaga rozłącznych identyfikatorów klatek między rolami, niezależnie od rozłączności grup. Zapobiega to nadpisaniu FIT przez D w mapie ID.
5. Raport warunków stosuje te same domknięcia przedziałów co diagnoza potrzeb.
6. Oprócz manifestów zapisujemy hashe archiwów obrazów. Marker testów sprawdza kod, konfigurację oraz dane wejściowe.
7. Dokumentacja opisuje sygnał GLE jako różnicę BCE odczytu tokenów, a nie potwierdzoną poprawę treningu. Rozstrzyganie remisów GLISTER jest powtarzalne przy tej samej kolejności kandydatów.
8. Kontrola rzeczywistych ID z loadera wykrywa niewłaściwe zastosowanie zmiany samplera. Brak poprawnej metryki walidacyjnej przerywa bieg.

Potwierdzone statycznie: rozłączność D względem dopasowania dynamiki, przyczynowe użycie historii, zachowanie pojedynczych patchy, cofanie wirtualnych wag i przywracanie RNG/trybów modelu, jeden nieprzerwany Trainer.fit, stałe budżety, bezpieczne rozpakowanie do katalogu zadania i trwałość wyników po usunięciu danych tymczasowych.

Pozostaje sprawdzenie wykonawcze: komplet testów oraz rzeczywisty trening dwóch epok na Edenie. Dopiero ich wynik potwierdzi działanie adaptera i całej pętli w runtime. Brak dowodu skuteczności naukowej przed kontrolowaną kampanią.
