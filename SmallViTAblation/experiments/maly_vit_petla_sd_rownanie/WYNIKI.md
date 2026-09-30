# Wyniki: pętla małego ViT ze Stable Diffusion i równaniem ruchu latentu

Plan i różnice wobec wersji 3: `PLAN.md`. Wszystko poniżej to sprawdzenie kodu i diagnoza, nie wynik pętli.

## Diagnoza prognozy wieloetapowej wersji 4.1 (2026-09-30, `diagnoza_prognozy.py`, `wyniki/diagnoza_s47/DIAGNOZA.json`)

- Wektor tożsamości wyznaczony dobrze: AUC 0,98 na odłożonych sondach (narzędzie kontra to samo miejsce bez narzędzia).
- Jeden krok równania trafny; prognoza na 100-500 kroków przy stałej średniej sile 1,5-50 razy gorsza od „braku zmiany”.
- Przyczyna: średnia siła paczek (około 0,0007 na krok) jest około 30 razy mniejsza od losowego rozrzutu sił kolejnych
  paczek, więc siła tożsamości na długim odcinku zachowuje się jak błądzenie losowe.
- Wniosek przyjęty przez użytkownika: „zróbmy przewidywanie na jeden krok do przodu” (wersja 4.2).

## Próba kodu wersji 4.2 (2026-09-30, `wyniki/proba_krok_s47`, log `wyniki/proba_krok_s47.log`)

Skala próby: rozgrzewka 2 × 60 kroków, 2 obroty × 40 kroków, 24 klatki na obrót, 2 kotwice × 4 wersje SD
(migawka `ema_015000`), 300 sekund na RTX 3070. Sprawdza tylko działanie kodu; mAP po 40 krokach nic nie mówi.

| ramię | równanie 1 krok, siła tożsamości (obrót 3 / 4) | sam pęd | przewidziane pchnięcie dodanych klatek (3 / 4) |
|---|---|---|---|
| `sd_prognoza` | 97,9 / 97,5 % | 48 / 30 % | +0,106 / +0,155 |
| `sd_bez_prognozy` | 97,1 / 98,8 % | 38 / 32 % | +0,074 / −0,097 |
| `losowe` | 98,9 / 99,1 % | 50 / 60 % | +0,022 / +0,067 |
| `kolejne` | 97,0 / 98,6 % | 30 / 49 % | +0,056 / −0,035 |
| `bez` | 99,2 / 98,0 % | 73 / 50 % | - |

- Równanie Δz(t+1) = S_pęd + S_paczka na prawdziwych krokach uczenia w pętli: 97-99 % dla siły tożsamości i 99 % dla
  całego latentu (10 kroków na obrót, wczesne uczenie). Na nagraniach z późniejszego uczenia było 95 %.
- Wybór wersji SD równaniem daje zestawy, które według równania mocniej podnoszą tożsamość niż losowe i kolejne klatki.
  To przewidywanie jednego kroku, nie zmierzony skutek całego obrotu.
- W próbie częściej wygrywały wersje z tym samym układem narzędzia (R2) niż z przestawionym (R1).

## Czego próba NIE dowodzi

- Nic o mAP: 40 kroków na obrót i jedno ziarno.
- Nic o tym, czy większe przewidziane pchnięcie jednego kroku przekłada się na siłę tożsamości po 500 krokach.
- Umiejętność 97-99 % zmierzona na wczesnym uczeniu (kroki około 120-200); pełne ziarno sprawdzi ją po rozgrzewce 1000 kroków.
