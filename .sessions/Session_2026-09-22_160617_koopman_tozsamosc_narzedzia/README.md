# Session 2026-09-22 16:06 - Wektor tozsamosci narzedzia: trzy drogi Koopmana, widmo i rozklad statyka-dynamika

## Cel sesji
Wydzielic w latencie detektora wektor, ktory niesie tozsamosc narzedzia, i oddzielic go od polozenia,
ruchu i odcisku nagrania. Sprawdzic trzy drogi (przesuniecie narzedzia, widmo czasu na wycinku z ramki,
deep Koopman), porownac je jednym metrem i rozstrzygnac zarzuty autora.

## Typ
experiment

## Workerzy uzyci
Jeden podagent przegladowy (Explore) do eksperymentow z niedzwiedziem DAVIS w projekcie grantowym. Reszta solo.

## Co zrobiono (skrot)
- Ramiona K5 na calej puli (10 637 klatek, zadania Eden 1809407 i 1809408); zadne ramie etapu 1 nie roznilo sie
  od losu w przestrzeni kodu tozsamosci, K5 zmniejsza najwieksza luke o polowe.
- Widmo bez symetryzacji: mody trwale, przemienne i obrotowe razem podnosza odroznienie narzedzia z 0,834 do 0,893.
- Niezmienniki wycinka klatka po klatce (pomysl autora): najtrwalszy mod 0,964 wobec podlogi 0,841 na potasowanym
  czasie; siatka 3 na 3 wystarcza, gestsza nic nie doklada.
- Deep Koopman: przegrywa z przewidywaniem "nic sie nie zmienia" i z wlasna podloga, ale znajduje te same kierunki
  co operator liniowy (kosinusy 0,98 i 0,94).
- Trzy drogi jednym metrem na puli: metoda z przesunieciem najczystsza (polozenie 4,1 procent), widmo czasu
  przecieka polozeniem (40,5), deep najslabszy. Zwykle osiem skladowych glownych nie jest gorsze.
- Odejmowanie (pomysl autora) dziala: reszta widma czasu po odjeciu przesuniecia to kod POLOZENIA (28,5 procent),
  bez tozsamosci.
- Teza autora potwierdzona: statyka filmu siedzi w modach o najwiekszej czesci rzeczywistej (64 procent wobec
  52 dla skladowych glownych i 30 dla losu), monotonicznie po Re.
- Z niezmiennikow NIE da sie odtworzyc ksztaltu narzedzia (R2 minus 0,17, gorzej niz srednia klatka).

Pelny opis: `SESSION_SUMMARY.md`.
