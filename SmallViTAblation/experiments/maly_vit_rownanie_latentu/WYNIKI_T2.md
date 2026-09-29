# T2: równanie latentu ze sterowaniem danymi — cel 80 przekroczony (2026-09-29)

Nagranie `nagranie_s47_sily` (ziarno 47, lokalnie), dopasowanie na krokach 1000–2000, test na krokach 2000–2999.
Umiejętność = procent błędu kwadratowego przyrostu latentu (narzędzie, 180 sond × 768 liczb) usunięty względem braku
zmiany. Skrypt `t2_sterowanie.py`, liczby `wyniki/T2.json`.

## Wynik

| wariant | co widzi | umiejętność testowa |
|---|---|---|
| t1 (równanie z TFT bez sterowania) | historia latentu | 60,2 |
| t1 + siła paczki | + sterowanie danymi | **85,8** |
| t1 + siła paczki z kroku −250 (placebo, dopasowane od nowa) | + zła paczka | 60,2 |
| t1 + jądro bez przeliczania sieci | + podobieństwo sondy do obrazów paczki × gradient ostatniego bloku | 60,2 |
| t1 + siła paczki + pęd optymalizatora | + stan AdamW | 95,3 |
| same siły (paczka, pęd, zanik wag) | bez historii latentu | 95,4 |

Równanie proste, dwa współczynniki dopasowane na krokach 1000–2000:

    Δz_{t+1} = 0,834 · Δz_t + 0,985 · S_t + reszta          test: 85,4 (sama historia 0,758 · Δz_t: 56,5)

Mediana po krokach 86,3 (10. percentyl 78,7), mediana po sondach 87,5, najgorsza sonda 71,5.

S_t — siła nowej paczki: zmiana latentu sondy, gdy wagi przesunąć tylko o część kroku AdamW pochodzącą z gradientu
tej paczki, −lr/b1 · (1 − β1) · g_t / mianownik, w stanie sieci sprzed kroku (jedno dodatkowe przejście sond w float32).

## Odczytanie fizyczne

Latent sondy zachowuje się jak cząstka z bezwładnością popychana przez dane:

- pierwszy człon to bezwładność. Latent zachowuje 83 % prędkości z poprzedniego kroku, bo AdamW ma pęd β1 = 0,9,
  częściowo zjadany przez zmianę wrażliwości sieci i normowanie kroku;
- drugi człon to siła danych, wchodząca ze współczynnikiem ≈ 1, czyli wprost;
- zanik wag jest pomijalny (norma 0,0015 przy przyroście 0,37);
- zamiana historii latentu na zapamiętany pęd optymalizatora daje 95: resztę do 85 stanowi to, że pęd widziany przez
  latent (0,83 · Δz_t) jest tylko przybliżeniem pędu w wagach.

## Podłogi i dostęp do informacji

- Placebo spada dokładnie do poziomu bez sterowania (60,2), więc zysk nie jest przeciekiem ani nadmiernym dopasowaniem.
- Siła paczki używa tylko tego, co znane przed krokiem t: paczki, jej gradientu i stanu sieci w chwili t. Nie używa
  latentu po kroku ani zmiany wag.
- Tanie sterowanie (podobieństwo × gradient na ostatnim bloku) nic nie daje. Zgadza się to z rozkładem T0: sam ostatni
  blok odpowiada tylko za 7,5 % przyrostu, a ruch latentu idzie przez całą sieć.

## Zastrzeżenia i czego to NIE dowodzi

- Jedno ziarno i jedno nagranie lokalne. Statystyka ruchu zgadza się z Edenem (T0). Replikacja: 10 ziaren na Edenie,
  `rownanie_eden.slurm`.
- Równanie działa na jeden krok naprzód. Siła dla dalszych kroków wymaga przyszłych wag, więc nie daje wprost
  prognozy na 10 czy 500 kroków.
- Siła paczki wymaga przeliczenia sieci (gradient paczki i przejście sond), a nie jest zamkniętym wzorem.
- Ramię bez dokładania klatek; zachowanie przy zmianie zbioru nie jest sprawdzone.

## Reguła użytkownika o przyszłych klatkach (dopisane 2026-09-29)

Użytkownik: „do równania nie mogą wchodzić przyszłe klatki, to jest niedozwolone w prognozie”. Siła S_t używa
paczki kroku t, czyli klatek, na których sieć dopiero się uczy. Wynik z S_t opisuje więc, co porusza latent,
a nie jest prognozą. Prognoza bez przyszłych klatek: `../maly_vit_rownanie_macierze` (pęd AdamW i paczki t−1
i wcześniejsze).
