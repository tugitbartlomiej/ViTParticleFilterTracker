# Mały ViT: pętla Koopman/Langevin z FastGAN-em na sile tożsamości i kierunki tożsamości w przestrzeni cząstek

Data: 2026-09-28 (ok. 15:30) - 2026-09-29 (14:52). Kontynuacja sesji
`Session_2026-09-28_011148_Petla_Large_Koopman_i_Domykanie_Tozsamosci`.

## Cel sesji

Sprawdzić na małym ViT uczonym od zera, czy pętla „tożsamość narzędzia w latencie -> prognoza Koopmana i
Langevina -> obraz z FastGAN-a -> wyszukanie podobnych klatek -> dokładanie do uczenia” przyspiesza uczenie
detektora bardziej niż losowe i kolejne klatki. Na koniec wyznaczyć samą tożsamość narzędzia w latencie na parach
„to samo miejsce z narzędziem i bez” i policzyć, ile kierunków przestrzeni cząstek ją koduje.

## Typ

experiment

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Mały ViT od zera (2,93 mln parametrów), trzy wersje pętli na Edenie (10, 10 i 20 ziaren). W iteracji 2 wersji 3
  metoda uczy szybciej niż losowe klatki (15 z 20 ziaren, p = 0,006), ale na końcu ramiona się zrównują, a ramię
  bez prognozy jest prawie tak samo dobre: zysk daje wyszukiwanie klatek podobnych do narzędzia, nie sama prognoza.
- FastGAN sterowany wzdłuż wektora tożsamości realizuje przewidzianą zmianę siły (znak 12 z 12).
- Tożsamość narzędzia na parach z narzędziem i bez (zamalowane oko, czarne tło): stary wektor mierzył w dużej części
  miejsce i scenę (d' 2,0), nowy izoluje narzędzie (d' 16,6, AUC 0,997), ale nie jest lepiej przewidywalny.
- Obraz cząstkowy: tożsamość niesie 1-2 kierunki z AUC >= 0,9 i 5-8 z AUC >= 0,75; to wąska wiązka kierunków,
  nie jedna strzałka (kierunek cząstkowy i wektor `nowy` mają cosinus 0,40, oba rozdzielają równie dobrze).
- Eden posprzątany ze 193,2 do 52,2 GB.

Pełny opis: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
