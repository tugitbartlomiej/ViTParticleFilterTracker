# Session 2026-09-22 - Kod tozsamosci narzedzia: ramiona K5, widmo bez symetryzacji, slad po przejsciu

## Cel sesji
Domknac pomysl autora "z latentu do klatek": policzyc kod tozsamosci narzedzia dla calej puli uczacej,
zbudowac z niego ramiona doboru klatek, a potem sprawdzic dwa zarzuty autora do dotychczasowego rachunku.
Pierwszy: w niezmiennikach Koopmana trzeba badac czesc rzeczywista (dodatnia i ujemna) oraz urojona.
Drugi: w eksperymentach z niedzwiedziem czastki sledzily obiekt, a w opisie Eulera lata miala trzymac slad
po przejsciu obiektu. Praca merytoryczna zyje w repozytorium ViT; tutaj tylko zapis procesu.

## Typ
experiment

## Workerzy uzyci
Jeden podagent przegladowy (Explore): odnalezienie i opis eksperymentow z niedzwiedziem DAVIS w projekcie
grantowym (opis Lagrange'a wobec Eulera, liczby wiernosci sledzenia). Reszta solo.

## Co zrobiono (skrot)
- Kod tozsamosci narzedzia policzony na Edenie dla wszystkich 10 637 klatek puli uczacej (zadania 1809407
  i 1809408, razem ponizej czterech minut); kolejka Edena pusta, nic tam juz nie liczy.
- Cztery ramiona doboru K5 zbudowane i zmierzone bez treningu: zadne ramie etapu 1 nie rozniło sie od losu
  w przestrzeni kodu tozsamosci, a K5 zmniejsza najwieksza luke pokrycia o polowe.
- Usterka znaleziona przy okazji: ramie K4 z etapu 1 dostalo wspolrzedne ustandaryzowane, ale nie jednostkowe,
  wiec liczylo cos innego niz pokrycie; zapisane jako zastrzezenie.
- Zmierzone, co kod naprawde niesie: wydluzenie prawdziwej ramki w 86 procentach, wielkosc w 29, zwrot
  koncowki w zerowym stopniu. Wczesniejsze slowa autorowi "otwarte czy zamkniete" wycofane.
- Widmo operatora translacyjnego BEZ symetryzacji (zarzut autora byl sluszny): same mody trwale daja
  odroznienie narzedzia 0,834, wszystkie trzy rodziny razem 0,893, a wielkosc ramki rosnie z zera do 30 procent.
- Szesc rysunkow objasniajacych na prawdziwych klatkach plus dwa sprawdzenia poprawnosci: siatka lezy tam,
  gdzie powinna (transpozycja psuje wynik pieciokrotnie), a dopasowanie tokenow miedzy klatkami trzyma sie
  narzedzia w 79 procentach przy 3 procentach dla losu.
- Slad po przejsciu narzedzia przez kwadrat: wychodzi 0,899, ale zapowiedz przed przyjsciem wychodzi 0,902,
  wiec to sasiedztwo, nie pamiec. Zgodne z ustaleniem U10 i U14 projektu grantowego.

Pelny opis: `SESSION_SUMMARY.md`.
