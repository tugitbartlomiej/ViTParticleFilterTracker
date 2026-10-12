# GLE ruchu cząsteczek: kierunki U500

Zakończona eksploracyjna analiza stałego obrazu, 31 tokenów i 1001 stanów (0..1000), czyli 1000 aktualizacji. [Raport](WYNIKI.md), [podsumowanie roota](PODSUMOWANIE.md), [plan](PLAN.md), [audyt Astry](AUDYT_ASTRA.md). Kierunki500 wyznaczono wyłącznie train; U1000 tylko retrospektywnie.

Odtwarzanie z F:/Python/python.exe w tym katalogu: identyfikacja500.py, trajektorie.py, prognoza.py --stage validation, audyt, prognoza.py --stage test, raport.py. Nie nadpisuje źródeł. Małe NPZ zawierają parametry/projekcje/predykcje; źródłowe 22 GB używa mmap. Rola sondy 2 = fit/train, nie niezależny obraz. Brak logów minibatchy, bez nowego treningu i commitów.
