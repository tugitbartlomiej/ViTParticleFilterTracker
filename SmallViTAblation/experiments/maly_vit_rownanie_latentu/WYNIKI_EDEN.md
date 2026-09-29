# T2 na Edenie: równanie latentu ze sterowaniem danymi na 10 ziarnach (2026-09-29)

## Pytanie

Czy wynik z jednego lokalnego nagrania (ziarno 47: równanie z siłą paczki 85,8, placebo 60,2) powtarza się na
innych przebiegach uczenia. Kryterium z PLAN.md: umiejętność jednego kroku powyżej 80 na każdym ziarnie i placebo
równe wariantowi bez sterowania.

## Aparat

Eden, zadanie 1818331 (tablica 0–9, ziarna 47–56), `rownanie_eden.slurm`: `nagranie.py --sily` (3000 kroków,
siły od kroku 900) w `/dev/shm` węzła, potem `t2_sterowanie.py` (dopasowanie 1000–2000, test 2000–2999). Kod ten sam
co lokalnie, bez zmian metody. Węzły: 6 × H100 (hopper), 4 × A100 (dgx-1, dgx-2, dgx-4), 17–19 min na ziarno.
Pliki: `wyniki_eden/T2_s47..56.json`, zbiorczo `wyniki_eden/ZBIORCZO.json` (`zbierz_eden.py`). Nagrania skasowane
pułapką; sprawdzone po biegu: brak katalogów `rownlat-*` w `/dev/shm` na hopper, dgx-1, dgx-2, dgx-4.

## Wynik (umiejętność testowa, procent błędu kwadratowego przyrostu usunięty względem braku zmiany)

| ziarno | t1 | t1 + paczka | placebo | t1 + paczka + pęd | same siły | równanie proste | a | b | 10. percentyl kroków | najgorsza sonda |
|---|---|---|---|---|---|---|---|---|---|---|
| 47 | 58,3 | 84,0 | 58,3 | 94,6 | 94,6 | 83,6 | 0,828 | 0,983 | 76,4 | 68,1 |
| 48 | 62,3 | 88,8 | 62,3 | 95,6 | 95,7 | 88,5 | 0,837 | 0,980 | 83,1 | 79,1 |
| 49 | 63,4 | 89,4 | 63,4 | 95,8 | 95,8 | 89,2 | 0,846 | 0,988 | 84,5 | 77,5 |
| 50 | 61,1 | 87,8 | 61,1 | 95,4 | 95,5 | 87,5 | 0,830 | 0,980 | 81,8 | 74,4 |
| 51 | 55,5 | 81,3 | 55,5 | 93,9 | 93,9 | 80,5 | 0,817 | 0,979 | 71,6 | 61,3 |
| 52 | 58,0 | 83,6 | 57,9 | 94,4 | 94,4 | 83,2 | 0,833 | 0,983 | 74,7 | 62,8 |
| 53 | 59,8 | 86,7 | 59,8 | 95,1 | 95,2 | 86,4 | 0,838 | 0,982 | 80,4 | 75,6 |
| 54 | 58,4 | 83,4 | 58,4 | 94,6 | 94,7 | 82,8 | 0,816 | 0,980 | 75,2 | 66,2 |
| 55 | 61,7 | 88,0 | 61,7 | 95,3 | 95,3 | 87,7 | 0,842 | 0,990 | 82,0 | 79,9 |
| 56 | 57,9 | 83,0 | 57,9 | 94,4 | 94,5 | 82,4 | 0,821 | 0,977 | 75,2 | 68,2 |
| **mediana** | 59,1 | **85,4** | 59,1 | 94,9 | 94,9 | **85,0** | 0,831 | 0,981 | 78,4 | 71,3 |
| zakres | 55,5–63,4 | 81,3–89,4 | 55,5–63,4 | 93,9–95,8 | 93,9–95,9 | 80,5–89,2 | 0,816–0,846 | 0,977–0,990 | 71,6–84,5 | 61,3–79,9 |

Równanie proste, mediana współczynników z 10 ziaren:

    Δz_{t+1} = 0,83 · Δz_t + 0,98 · S_t + reszta

## Werdykt

- **Cel spełniony na każdym ziarnie.** Równanie z siłą paczki przekracza 80 w 10 z 10 ziaren (t1 + paczka 81,3–89,4,
  równanie proste z dwoma współczynnikami 80,5–89,2). Zysk ze sterowania danymi: +26 punktów w medianie.
- **Placebo równe wariantowi bez sterowania w każdym ziarnie** (największa różnica 0,01 punktu). Zysk nie jest
  przeciekiem ani nadmiernym dopasowaniem.
- **Współczynniki są stałe między ziarnami:** bezwładność a 0,82–0,85, siła paczki b 0,98–0,99. To ta sama fizyka
  w każdym przebiegu, a nie dopasowanie do jednego nagrania.
- Tanie jądro bez przeliczania sieci znów nie daje nic (t1 + jądro = t1 w każdym ziarnie).
- Pęd optymalizatora zamiast historii latentu podnosi wynik do 93,9–95,8 w każdym ziarnie.

## Zgodność z biegami pętli (Eden v2, ten sam start i te same paczki)

Ruch latentu narzędzia co 10 kroków, kroki 1000–2990: norma przyrostu 2,39 w nagraniu wobec 2,49 w zapisie pętli
(mediana 10 ziaren; nagranie 4 % spokojniejsze), korelacja kolejnych przyrostów −0,259 wobec −0,262. Statystyka ruchu
się zgadza, ale trajektorie nie są identyczne liczba w liczbę: ziarno 47 na Edenie (H100) daje 84,0, lokalnie (RTX 3070)
85,8. Różnice numeryczne kart rozchodzą się w trakcie uczenia, więc to inne, równie prawdopodobne przebiegi.

## Zastrzeżenia i czego to NIE dowodzi

- Jeden krok naprzód. Siła paczki S_t wymaga przejścia sieci w stanie sprzed kroku (gradient paczki i sondy w float32),
  więc nie daje wprost prognozy na 10 czy 500 kroków.
- Tylko ramię bez dokładania klatek i tylko kroki 1000–3000.
- Dziesięć ziaren to dziesięć przebiegów tego samego uczenia; inne zbiory i architektury niesprawdzone.

## Reguła użytkownika o przyszłych klatkach (dopisane 2026-09-29)

Użytkownik: „do równania nie mogą wchodzić przyszłe klatki, to jest niedozwolone w prognozie”. Siła S_t używa
paczki kroku t, czyli klatek, na których sieć dopiero się uczy. Wynik z S_t opisuje więc, co porusza latent,
a nie jest prognozą. Prognoza bez przyszłych klatek: `../maly_vit_rownanie_macierze` (pęd AdamW i paczki t−1
i wcześniejsze).
