# Jeden krok siły tożsamości z paczką i pędem optymalizatora (2026-09-30)

## Pytanie użytkownika

„Warunek jest taki, że znamy historię klatek i latentu i znamy, jakie klatki tworzą latent w chwili t. Chcemy mieć
takie równanie, [...] żebym był w stanie powyżej 85 % przewidzieć, gdzie będzie cząsteczka w chwili t+1, [...] zwłaszcza
zmianę latentu, która odpowiada za tożsamość narzędzia.”

W `../maly_vit_rownanie_latentu` (T2, Eden 10 ziaren) wariant „same siły” dawał 94–96 % dla całego latentu, ale
nikt nie policzył go dla samej siły wektora tożsamości. Ten pomiar to uzupełnia.

## Aparat

Pomiar lokalny, bez uczenia sieci, na zapisanych nagraniach `../maly_vit_rownanie_latentu/nagranie_s47_sily`
(siły od kroku 900) i `nagranie_s48_sily` (siły od kroku 2000). Skrypt `tozsamosc_jeden_krok.py`, wynik
`wyniki/TOZSAMOSC_JEDEN_KROK.json`, log `wyniki/tozsamosc_jeden_krok.log`.

- Cząstka: latent sondy (768 liczb, średnia po tokenach narzędzia), 180 sond z narzędziem.
- Siły (jak w `docs/23_METODA_PRZEWIDYWANIA_LATENTU.md`): zmiana latentu sondy, gdy wagi przesunąć o jedną część
  kroku AdamW w stanie sprzed kroku. `S^paczka_t` z gradientu paczki kroku t (klatki, które tworzą latent t+1),
  `S^pęd_t` z pędu AdamW (pamięć paczek do t−1), `S^zanik_t` z zaniku wag.
- Tożsamość: rzut przyrostu, wystandaryzowanego negatywami, na kierunek LDA z kroku 1000 danego ziarna (`dane.py`
  z `../maly_vit_rownanie_macierze`).
- Współczynniki skalarne, wspólne dla wszystkich 768 liczb i sond, dopasowane na s47 w krokach 1000–1999.
  Test: s47 kroki 2000–2998 i s48 kroki 2001–2998 tymi samymi współczynnikami (s48 nie brało udziału w dopasowaniu).
- Miara: umiejętność = 100 · (1 − SSE / SS) przyrostu względem braku zmiany.

## Wynik (umiejętność testowa: cały latent / siła tożsamości)

| równanie | ziarno 47 | ziarno 48 |
|---|---|---|
| pęd: 0,90 · S^pęd (prognoza bez paczki kroku t) | 66,4 / 65,4 | 64,4 / 63,8 |
| bezwładność + paczka: 0,83 · Δz_t + 0,99 · S^paczka | 85,4 / 85,6 | 86,8 / 87,6 |
| **paczka + pęd: 0,98 · S^paczka + 0,98 · S^pęd** | **95,4 / 95,2** | **95,5 / 95,6** |
| paczka + pęd + zanik | 95,4 / 95,2 | 95,5 / 95,7 |
| bezwładność + paczka + pęd (bezwładność dostaje 0,04) | 95,4 / 95,2 | 95,5 / 95,6 |
| **paczka + pęd + zanik, wszystkie współczynniki = 1, bez dopasowania** | **95,3 / 95,2** | **95,5 / 95,6** |

## Werdykt

- Siła tożsamości w chwili t+1 daje się przewidzieć w 95 %, gdy znamy klatki kroku t i pęd optymalizatora
  (pamięć klatek wcześniejszych). Próg 85 % przekroczony o 10 punktów w obu ziarnach.
- Równanie nie wymaga dopasowania: Δz_{t+1} = S^pęd_t + S^paczka_t (+ S^zanik_t, pomijalny) z jedynkami daje to samo.
  To przybliżenie pierwszego rzędu samego kroku AdamW: każda część kroku przepuszczona przez sieć osobno i dodana.
- Bezwładność 0,83 · Δz_t była tylko zastępstwem pędu w wagach: gdy pęd jest w równaniu, bezwładność dostaje 0,04.
- Tożsamość zachowuje się tak samo jak cały latent (różnica poniżej 1 punktu).

## Czego to NIE dowodzi

- Jeden krok naprzód. Siły dla dalszych kroków wymagają przyszłych wag.
- Obie siły wymagają przejścia sieci: to nie jest zamknięty wzór, tylko rozkład kroku na części.
- Paczka kroku t to klatki, na których sieć się dopiero uczy. Zgodnie z warunkiem użytkownika z 2026-09-30 jest znana;
  jako prognoza bez tych klatek zostaje 64–66 % (sam pęd).
- Dwa ziarna lokalnie, ramię bez dokładania klatek, kroki 2000–3000. Cały latent na 10 ziarnach Edenu: 93,9–95,9 %
  (`../maly_vit_rownanie_latentu/WYNIKI_EDEN.md`), tożsamość tam niepoliczona.
- Brakujące 5 % to nieliniowość (części kroku nie sumują się dokładnie) i zaokrąglenie zapisu do float16;
  tego nie rozdzielono.
