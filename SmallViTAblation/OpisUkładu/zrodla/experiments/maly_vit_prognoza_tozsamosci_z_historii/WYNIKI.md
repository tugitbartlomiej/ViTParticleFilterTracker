# Prognoza siły tożsamości ze stanu i historii: Koopman, Langevin, pęd, przyspieszenie (2026-09-30)

Pytanie użytkownika: „w chwili t mamy stan i historię; pytanie, czy równanie Langevina z jądrem pamięci ma przewidywać
latent tożsamości narzędzia, czyli jak przewidzi t+1, t+2, t+3 itd., jaki jest błąd; sprawdzaj różne równania
Koopmana, Langevina, potem pomieszane z pędem, przyspieszeniem; rób wykres błędu”.

Skrypt `prognoza_z_historii.py` (opis modeli w nagłówku), wykres `wykres.py`. Wyniki `wyniki/PROGNOZA_Z_HISTORII.json`,
`wyniki/PROGNOZA_Z_HISTORII.png`, log `wyniki/prognoza.log`. Bez GPU, 225 s.

Dane: nagrania `../maly_vit_rownanie_latentu/nagranie_s47_sily` i `nagranie_s48_sily` (tylko odczyt), siła tożsamości
180 sond w każdym kroku. Dopasowanie na s47, kroki 1000-1999; test na s47 i s48 (s48 poza dopasowaniem), 100 startów
2000-2495 co 5 kroków. Umiejętność = 100 · (1 − SSE / SS) przyrostu s(t+H) − s(t) wobec „braku zmiany”.

## Umiejętność [%] (test s47 + s48)

| model | t+1 | t+2 | t+3 | t+5 | t+10 | t+20 | t+50 | t+100 | t+200 | t+500 |
|---|---|---|---|---|---|---|---|---|---|---|
| bezwładność v(t+1) = a·v(t) | 60 | 45 | 29 | 3 | −15 | −22 | −23 | −20 | −15 | −9 |
| Langevin OU (tarcie + sprężyna do poziomu) | 61 | 49 | 35 | 15 | 11 | 15 | 14 | 16 | 8 | 5 |
| Langevin + jądro pamięci 20 | 64 | 53 | 40 | 24 | 20 | 26 | 25 | 20 | 8 | 5 |
| Langevin + jądro pamięci 100 | 64 | 53 | 40 | 24 | 21 | 29 | 29 | 27 | 14 | 6 |
| Koopman: opóźnienia 20, każda sonda | 64 | 53 | 40 | 24 | 20 | 27 | 25 | 20 | 8 | 5 |
| Koopman DMD: 180 sond razem | 37 | 28 | 17 | 0 | −12 | −5 | 3 | 4 | −4 | −2 |
| Koopman DMD + pęd | 37 | 30 | 19 | −2 | −19 | −24 | −26 | −28 | −61 | −505 |
| Langevin OU + pęd | 67 | 53 | 34 | −2 | −40 | −52 | −13 | 16 | 10 | 5 |
| Langevin OU + pęd + przyspieszenie | 69 | 56 | 38 | 3 | −34 | −51 | −23 | 10 | 12 | 5 |
| Langevin + jądro 20 + pęd | 69 | 57 | 40 | 6 | −29 | −44 | −32 | −4 | 10 | 7 |
| Langevin + jądro 20 + pęd + przyspieszenie | 69 | 57 | 40 | 6 | −29 | −44 | −32 | −4 | 10 | 7 |
| **Langevin + jądro 20 + pęd, wprost** | **69** | **58** | **45** | **26** | **21** | **27** | **28** | **26** | **14** | **10** |
| Langevin + jądro 20, wprost (bez pędu) | 64 | 53 | 40 | 24 | 20 | 27 | 28 | 26 | 14 | 11 |
| znane klatki kroku t, dalej jądro 20 + pęd | 96 | 88 | 72 | 30 | −24 | −48 | −37 | −8 | 8 | 7 |
| równanie co krok na prawdziwych wagach (nie prognoza) | 96 | 98 | 98 | 98 | 98 | 97 | 95 | 92 | 88 | 76 |

## Werdykt

- Ze stanu i historii najlepsze równanie (Langevin z jądrem pamięci i pędem, dopasowane osobno na każde H) trafia
  69 % na t+1, 58 % na t+2, 45 % na t+3, około 25 % od t+5 do t+100 i 10 % na t+500. 90 % nie osiąga żaden model.
- Koopman z opóźnieniami i Langevin z jądrem pamięci dają identyczne liczby: to ta sama rodzina liniowych równań
  (opóźnienia Koopmana = jądro pamięci Langevina), zapisana inaczej.
- Koopman DMD na 180 sondach razem jest gorszy (37 % na t+1): wspólne mody sond nie niosą tu informacji o ruchu.
- Pęd pomaga na t+1 i t+2 (69 zamiast 64), ale składany krok po kroku psuje prognozę od t+5 do t+100. Przy osobnym
  dopasowaniu na każde H (wprost) już nie szkodzi.
- Przyspieszenie nic nie dodaje, gdy jest jądro pamięci (liczby identyczne); przy krótkim Langevinie daje 1-4 punkty.
- Znane klatki kroku t dają 96 % na t+1 i 88 % na t+2, ale dalej złożenie z modelem historii jest gorsze niż sam model
  historii. 90 % trzyma tylko t+1.
- Brakująca część to paczki przyszłych kroków: każda paczka pcha tożsamość losowo, a historia tego nie zawiera.
  Równanie na prawdziwych wagach (linia szara) pokazuje, że gdy te paczki są znane, 90 % trzyma do około 150 kroków.

## Czego to NIE dowodzi

- Dwa nagrania bez dokładania klatek, kroki 1000-3000; w pętli z dokładaniem klatek historia jest mniej stacjonarna.
- Współczynniki wspólne dla 180 sond; modele osobne dla sond nie były sprawdzane.
- Poziom równowagi m(t) = średnia z 500 kroków; inne okna nie były sprawdzane.
- Dotyczy siły tożsamości (jedna liczba na sondę), nie całego latentu 768.
