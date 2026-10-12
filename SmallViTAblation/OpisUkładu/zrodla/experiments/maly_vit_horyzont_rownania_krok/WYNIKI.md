# Wyniki: horyzont prognozy latentu tożsamości (2026-09-30, ziarno 47)

Plan i warianty: `PLAN.md`. Skrypt `horyzont.py`, wykres `wykres_horyzontu.py`. Wyniki `wyniki/HORYZONT_s47.json`,
`wyniki/horyzont_s47.npz`, wykres `wyniki/HORYZONT_s47.png`, log `wyniki/horyzont_s47.log`. 978 s na RTX 3070.

## Umiejętność prognozy przyrostu siły tożsamości (180 sond, 4 okna startu razem)

| H (kroków naprzód) | 1 | 2 | 3 | 5 | 10 | 30 | 100 | 150 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| A: równanie co krok na prawdziwych wagach (nie prognoza) | 97 | 98 | 98 | 98 | 98 | 95 | 94 | 93 | 86 | 82 |
| B: znane przyszłe klatki, wagi z t0 | 98 | 98 | 88 | 6 | −111 | −743 | −1968 | −3588 | −7646 | −16764 |
| B2: jak B, suma osobnych pchnięć | 98 | 96 | 83 | −4 | −231 | −5491 | −50641 | | | |
| C1: bez przyszłych klatek, sam pęd | 62 | 38 | 20 | −36 | −66 | −63 | 5 | 19 | −5 | 20 |
| C2: bez przyszłych klatek, pęd + średnia paczka | 73 | 64 | 56 | −15 | −106 | −701 | −1309 | −2240 | −5422 | −15609 |

Ostatnie H z umiejętnością co najmniej 90 procent: **A 150 kroków, B 2 kroki, B2 2 kroki, C1 i C2 nigdy** (nawet H = 1).
Cały latent (180 × 768) zachowuje się tak samo (`HORYZONT_s47.json`, klucz `caly_latent`).

## Werdykt

- Prawdziwa prognoza, czyli z wiedzą dostępną w chwili t0, trzyma 90 procent przez **2 kroki**, i to tylko wtedy, gdy
  znamy klatki tych kroków (B). Przy 3 krokach 88 procent, przy 5 krokach tyle co „brak zmiany”, dalej coraz gorzej.
- Bez przyszłych klatek (C1, C2) 90 procent nie ma nawet na jeden krok: 62-73 procent, bo nie znamy paczki kroku t.
- Równanie jednego kroku samo w sobie jest trafne: gdy liczyć je co krok na prawdziwych wagach i sumować (A), trzyma
  90 procent przez 150 kroków, a 82 procent po 500. Błąd nie narasta więc od samego sumowania kroków.
- Przyczyna rozpadu B i C2: wagi zamrożone w t0. W prawdziwym uczeniu każdy krok zmienia wagi, a zmienione wagi dają
  inne gradienty, które cofają część wcześniejszych pchnięć (siła powrotna krzywizny straty). Przy wagach z t0 tej siły
  nie ma, więc pchnięcia dodają się bez hamowania i prognoza po 100 krokach jest około 45 razy za duża (błąd/brak
  zmiany około 4,4 w B). Suma osobnych pchnięć (B2) jest jeszcze gorsza, bo dochodzi nieliniowość sieci.
- Sam wygasający pęd (C1) nie eksploduje, bo pęd gaśnie po około 10 krokach, ale też nic nie przewiduje (okolice zera).

## Wniosek dla pętli 4.2

Pętla używa równania do porównania kandydatów na jeden krok (wariant „znane klatki, H = 1”, 98 procent), więc jest
w obszarze, gdzie równanie działa. Dłuższej prognozy latentu tożsamości tym równaniem nie da się zrobić bez modelu siły
powrotnej: trzeba by albo liczyć gradienty przy przyszłych wagach (to jest po prostu uczenie na kopii sieci), albo dopisać
do równania człon, który cofa pchnięcia z czasem (jądro pamięci, dopasowane do historii).

## Porównanie z pomiarem z 2026-09-29

`../maly_vit_horyzont_prognozy/WYNIKI.md` (inna sesja): Koopman i Langevin dopasowani do historii siły tożsamości, bez
przyszłych klatek, dają 15-25 procent umiejętności od 50 do 1000 kroków. Też daleko od 90 procent, ale w przeciwieństwie
do C2 nie eksplodują, bo przewidują powrót do poziomu sondy. Na dłuższych horyzontach to wciąż najlepsza prognoza bez
przyszłych klatek.

## Czego to NIE dowodzi

- Jedno ziarno, jedna baza bez dokładania klatek, kroki 1000-3000; okna różnią się (pasma na wykresie).
- Wariant A zna przyszłe wagi, więc nie jest prognozą; pokazuje tylko, że samo sumowanie trafnych kroków nie psuje wyniku.
