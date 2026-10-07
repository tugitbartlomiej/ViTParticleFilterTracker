# Wyniki: liniowy opis ruchu każdej cząstki

Stan: 2026-10-07, etap 1 (lokalnie, gęsta historia 1001 kroków, ziarno 47). Etap 2 trwa na Edenie.
Liczby z `wyniki/MODY_CZASOWE.json`, `wyniki/STRUKTURA_SZUMU.json`, `wyniki/OSCYLATOR.json`; klatka próbna 3
(TR_0001_S1_P03 kl. 576, 17 patchy narzędzia), dalsze klatki dopisywane po przeliczeniu.

## 1. Wspólne mody czasowe (`mody_czasowe.py`)

Trajektorie wszystkich 600 cząstek x 768 kanałów jako suma k wspólnych modów czasowych (mody z kroków 0-700):

| k modów | wariancja położeń (0-700) | wariancja przyrostów (0-700) | R^2 tokenu na teście 700-1000 (narzędzie / tło) |
|---|---|---|---|
| 1 | 45,7 % | 18,7 % | 41 / 44 % |
| 5 | 75,1 % | 45,0 % | 56 / 62 % |
| 20 | 89,2 % | 68,5 % | 59 / 68 % |

Wniosek: położenia są nisko-wymiarowe (wolny, wspólny dryf), przyrosty z kroku na krok już nie.
Liniowy układ na modach a(t+1) = A a(t) + b lr(t) + c nie bije persystencji (jeden krok 1,01-1,18 x błąd
persystencji, jazda swobodna gorsza). Sam dryf w modach nie wystarcza do prognozy.

## 2. Struktura ruchu z kroku na krok (`struktura_szumu.py`)

| | rozgrzewka 1-300 | stały lr 300-1000 |
|---|---|---|
| cos kolejnych przyrostów (narzędzie / tło) | 0,74 / 0,75 | 0,69 / 0,72 |
| MSD(h=1 / 5 / 20 / 50 / 200), narzędzie | 0,56 / 6,5 / 20 / 29 / 64 | 0,74 / 7,5 / 17 / 20 / 28 |

Kolejne kroki są silnie dodatnio skorelowane (pęd AdamW, beta1 = 0,9), a przemieszczenie przy stałym lr
nasyca się po około 20-50 krokach. To obraz tłumionego oscylatora wokół wolno płynącego środka: dyskretny
Langevin z bezwładnością (AR(2)), zgodny z teorią SGD z momentem (Kunin i in. 2021) i z autokorelacją 0,69
zmierzoną wcześniej na sondach (docs/28).

## 3. Oscylator dla każdej cząstki osobno (`oscylator.py`)

v_(t+1) = beta_i v_t − kappa_i (z_t − m_t), z_(t+1) = z_t + v_(t+1), m_t = średnia ostatnich 50 kroków.
Dopasowanie 300-700, test 700-1000. Błąd jazdy swobodnej h kroków względem persystencji (mniej = lepiej):

| model | h = 1 | h = 5 | h = 20 | h = 50 |
|---|---|---|---|---|
| stała prędkość | 0,51 | 2,32 | 12,7 | 60,0 |
| sam pęd (beta_i) | 0,45 | 1,05 | 1,18 | 1,18 |
| **oscylator (beta_i, kappa_i)** | **0,42** | **0,91** | **0,86** | **0,81** |
| oscylator, tylko narzędzie | 0,44 | 0,92 | 0,75 | 0,71 |
| wspólny Koopman PCA-32 + lr | 0,69 | 1,13 | 1,30 | 1,54 |

Parametry (mediana po tokenach): beta = 0,75, kappa = 0,05 (narzędzie: 0,72 i 0,063). Oscylator na każdej
cząstce osobno to najlepszy model z samej historii: na 1 krok usuwa 58 % błędu persystencji, na 50 kroków
19 % (narzędzie 29 %). Wspólna macierz Koopmana dla wszystkich cząstek jest gorsza od persystencji, zgodnie
z wcześniejszymi wynikami projektu (EDMD, NNDMD).

## 4. Zgodność na trzech klatkach (2, 3, 6)

- Mody czasowe: 5 modów wyjaśnia 73-75 % wariancji położeń na każdej klatce, R^2 na teście 61-62 %.
- **Mody są wspólne dla klatek**: cosinusy kątów głównych między przestrzeniami 5 pierwszych modów klatek
  3-2, 3-6, 2-6 to 0,99 / 0,99 / 0,99 / 0,93-0,98 / 0,62-0,84. Pierwsze 3-4 mody to ten sam „zegar uczenia”
  dla różnych obrazów; cząstki różnią się tylko współczynnikami przy wspólnych modach.
- Korelacja kolejnych przyrostów 0,69-0,74 na każdej klatce i w obu oknach.
- Oscylator (całość, błąd / persystencja, h = 1 / 50): klatka 3: 0,42 / 0,81; klatka 2: 0,39 / 0,81;
  klatka 6: 0,41 / 0,78. Parametry: beta 0,72-0,77, kappa 0,04-0,05. Wspólny Koopman PCA-32 gorszy od
  persystencji na każdej klatce (h = 50: 1,44-1,74).

## Wniosek etapu 1

Ruch każdej cząstki da się zapisać liniowo w przestrzeni stanu (z, v) cząstki: bezwładność plus sprężyna do
wolnego środka. Ale z samej historii przewiduje to tylko część ruchu, bo każdy krok dostaje nowe pchnięcie
od paczki. Etap 2 mierzy to pchnięcie bezpośrednio, osobno dla każdej cząstki, i sprawdza, czy z nim model
liniowy wyjaśnia ruch każdej cząstki (jak 85 % na średniej w docs/23).

## Etap 2 i 3: siły na każdej cząstce (Eden 1823539, dopasowanie 1823541)

Bieg: ziarno 47, 1000 kroków, H100, 1202 s; klatki 2, 3, 6, wszystkie 600 tokenów w każdym kroku.
Dane na Edenie: `~/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane` (Z, S_nowa, S_ped, S_zan,
S_buf, S_grupy; około 17 GB). Lokalnie: `wyniki/eden_pelny/` (DOPASOWANIE.json, SKALARY.json, metadata.json, logi).

**Reszta liniowości** ||dz − (S_nowa + S_ped + S_zan + S_buf)|| / ||dz|| na krok (cała klatka): mediana 0,055,
90. centyl 0,107. Na pojedynczych tokenach R^2 sumy sił bez żadnych parametrów: 98,9-99,7 %.

Umiejętność na teście (kroki 700-1000, dopasowanie 300-700), mediana po tokenach, narzędzie / tło:

| model (na każdej cząstce) | klatka 2 | klatka 3 | klatka 6 |
|---|---|---|---|
| suma sił, bez parametrów | 99,7 / 99,7 | 99,3 / 99,6 | 98,9 / 99,5 |
| E1: dz_t = a dz_(t−1) + b S_nowa(t) | 97,6 / 98,0 | 96,5 / 97,3 | 96,1 / 97,0 |
| E3: + sprężyna, BatchNorm, zanik | 98,1 / 98,4 | 96,9 / 97,6 | 96,6 / 97,5 |
| **W: E3 z jednymi parametrami dla wszystkich cząstek** | **98,1 / 98,4** | **96,9 / 97,6** | **96,7 / 97,4** |
| P1: bez siły paczki (pęd + sprężyna), prognoza 1 krok | 59,5 / 59,7 | 53,7 / 58,1 | 54,7 / 56,6 |

Udziały przyczyn w ruchu cząstki (średnio <S_k, dz> / ||dz||^2): pęd AdamW 64-68 %, nowa paczka 31-35 %,
BatchNorm około 1 %, zanik wag 0 %.

Wspólne prawo ruchu (model W, te same liczby na trzech klatkach): a = 0,86-0,87, b = 0,985-0,99,
k (sprężyna) = 0,004-0,006, c (BatchNorm) = 0,43-0,55.

## Wniosek

Ruch każdej cząstki w uczeniu jest liniowy w przestrzeni stanu (z_i, v_i) z wejściem równym sile bieżącej paczki:

    v_i(t+1) = 0,87 v_i(t) + S_nowa,i(t)     (+ 0,5 S_buf,i, mała sprężyna 0,005 do średniej)
    z_i(t+1) = z_i(t) + v_i(t+1)

To jedno prawo z jednymi parametrami dla wszystkich cząstek opisuje 96-98 % ruchu każdej cząstki na krokach
testowych. Na średniej po tokenach docs/23 dawało 85 %. a = 0,87 to pęd AdamW (beta1 = 0,9), b = 1 to pełne
pchnięcie paczki. Z czego wynika ruch: w dwóch trzecich z pamięci poprzednich paczek (pęd), w jednej trzeciej
z bieżącej paczki; BatchNorm i zanik wag praktycznie nic.

Czego to nie rozstrzyga: siła paczki S_nowa,i(t) jest wejściem z zewnątrz. Bez niej prognoza jednego kroku to
54-60 %. Siła paczki zależy od tego, które obrazy trafią do paczki, a tym da się sterować. To jest punkt wyjścia
etapu 4 (ulepszenie uczenia): dobór paczek według przewidywanej siły na cząstki narzędzia.
Zastrzeżenia: 1 ziarno, 3 klatki, kroki 0-1000; parametry dopasowane na 300-700 przy stałym lr.
