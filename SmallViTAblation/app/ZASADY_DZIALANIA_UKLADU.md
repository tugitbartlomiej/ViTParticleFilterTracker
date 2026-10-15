# Zasady działania układu „lasery, pułapka i koloidy” (MałyViT, bieg s47) — wersja 2026-10-14

Ten plik ustala, co w obrazie „768 reflektorów oświetla 600 koloidów” jest **dokładną matematyką sieci**, co jest
**analogią**, a co jest **błędem**. Obowiązuje dla aplikacji (`app/`, zakładka „Lasery i koloidy”), filmów i opisów.

- **Skąd.** Połączenie i aktualizacja dwóch plików z `outputs/porownanie_szklo_20261010/film_krysztal/`: `ZASADY_LASER_KOLOID.md`
  (wersja z 2026-10-13, zasady Z1–Z24) i `NADAZANIE_PULAPKI_S47.md` (2026-10-12). Tamte pliki zostają jako archiwum, bo
  odwołują się do nich skrypty filmów; aktualna wersja jest tutaj.
- **Co nowego (2026-10-14).** Nowe znaczenie słowa „pułapka” (niżej); nowe zasady Z25–Z29: pułapka optyczna jako ogniskowa
  wiązek, temperatura bramek, nadążanie (dawny osobny plik), wyniki dla ViT-S/16 z ImageNet i to, co gdzie widać w aplikacji.
- **Pliki i liczby.** Ścieżki bez katalogu odnoszą się do `outputs/porownanie_szklo_20261010/film_krysztal/`. Liczby z biegu
  `assets/MalyViT_35000_od_zera_s47`: `przelicz_zasady_s47.py` → `PRZELICZENIE_ZASADY_S47.json`, `opis_ukladu/pomiary_opisu_s47.py`
  → `opis_ukladu/POMIARY_OPISU_S47.json`, `opis_ukladu/KONTROLA_WZOROW.txt`. Wyjątki są podane przy zasadzie.
- **Pełny opis z wyprowadzeniami:** `opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.pdf`. Mosty do fizyki z literaturą: `app/MOSTY_ML_FIZYKA_PULAPKI.md`.
- **Plan dalszych pomiarów** (jak w pęsecie optycznej, na Edenie): `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md`.

> **Słowo „pułapka” od 2026-10-14.** „Pułapka” oznacza **pułapkę optyczną z laserów**: pole sił wszystkich wiązek bloku
> wokół koloidu i jego ognisko (punkt, w którym pchnięcia się znoszą). Dawny „środek drgań” (średnie położenie koloidu
> z ±300 kroków uczenia, zielona poświata w aplikacji) nazywa się teraz **„średnia skoków koloidu”**. Jest liczona z ruchu
> koloidu i z przyszłych kroków, więc nie jest przyczyną ruchu. Dolina w krajobrazie straty to **„dolina straty”**.

## 1. Sieć, której dotyczą zasady

MałyViT (`experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`):

- łodyga: 4 konwolucje 3×3 z krokiem 2 (3 → 24 → 48 → 96 → 192 kanały; pierwsze trzy z BatchNorm i GELU), klatka 480×320
  → siatka 30×20 = **600 tokenów**, krok 16 px; do tokenów raz dodawane jest stałe kodowanie pozycji 2D sin-cos;
- **6 bloków** timm (pre-LN): `x = x + attn(LN1(x))`, potem `x = x + mlp(LN2(x))`; uwaga ma 3 głowy, MLP to 192 → 768 → 192
  z dokładnym GELU (erf); LayerNorm ma ε = 10⁻⁶;
- końcowa LayerNorm i głowa na każdym tokenie: 192 → 192 → GELU → 5 (logit środka narzędzia i odległości l, t, r, b);
- strata: ogniskowa strata CenterNet + 2 · (1 − GIoU); 2 926 493 parametry; AdamW lr 5·10⁻⁴, wd 0,05, paczka 32,
  rozgrzewka 300 kroków, obcięcie normy gradientu do 1;
- bieg s47: ziarno 47, 806 checkpointów do kroku 65 640 (60 epok po 1094 kroki). Przejście modelu odtwarza zapisane
  aktywacje z R² ≥ 0,9999.

## 2. Słownik: co jest czym

| Obraz fizyczny | Obiekt w sieci | Wzór |
|---|---|---|
| **koloid** (cząstka) | token j: stan w strumieniu rezydualnym | `x_j ∈ R^192`, j = 1…600 |
| czas ruchu koloidu | **głębokość**: przejście przez kolejne bloki | `x_j^(l+1) = x_j^(l) + uwaga + MLP` |
| **reflektor** (laser; 768 w bloku, 6 × 768 w sieci) | neuron c warstwy ukrytej MLP | celownik, wiązka i próg |
| **celownik** reflektora | wiersz c macierzy fc1 widziany przez LayerNorm | `s_c = P₁(γ ⊙ w_c)`, próg `b_c + w_cᵀβ` |
| **wiązka** reflektora (dawniej „igła”) | kolumna c macierzy fc2: kierunek i moc pchnięcia | `v_c ∈ R^192` |
| jasność wiązki (dioda j reflektora c) | aktywacja po GELU | `z_jc = GELU(w_c · LN(x_j) + b_c)` |
| pchnięcie koloidu w bloku | wyjście MLP | `Δx_j = Σ_c z_jc v_c + b₂` |
| **pułapka** (optyczna, z laserów) | pole wiązek wokół koloidu i jego zero | `F(x)`, Jakobian `J`, ognisko `F = 0` (Z15, Z25) |
| siły między koloidami | uwaga (3 głowy) | `f_jk = Σ_h A^h_jk W_O^h W_V^h LN₁(x_k)` (Z19) |
| jedyna prawdziwa sfera | LayerNorm przed MLP | sfera o promieniu √192 rozciągnięta przez γ (Z17) |
| ruch reflektorów | uczenie: AdamW zmienia `w_c`, `v_c`, `b_c` | `w_c(t)`, `v_c(t)` |
| moment siły na wiązce | gradient po `v_c` | `∂L/∂v_c = Σ_j z_jc δ_j` (Z22) |
| **plamka** | stan tokenu oglądany w czasie uczenia | `z_j(t) = f(x_j; θ_t)` (Z8) |
| **średnia skoków koloidu** | średnie położenie plamki z ±300 kroków uczenia | średnia krocząca, symetryczna (Z24) |
| **dolina straty** | dolina minimów w przestrzeni wag | (Z9) |

κ = cos(celownik, wiązka) tego samego reflektora: κ > 0 wzmacnia (pcha dalej w stronę, z której zapalił), κ < 0 działa jak
sprężyna, κ ≈ 0 pcha w bok i daje wir (Z23).

## 3. Zasady

**Z1. Koloid ma stan tylko wzdłuż głębokości.** Token jest prawdziwą cząstką w czasie płynącym przez bloki: jego położenie
w bloku l+1 zależy od położenia w bloku l (Geshkovski i Rigollet: tokeny jako układ oddziałujących cząstek, czas = głębokość).

**Z2. Reflektor ma dwa wektory.** Celownik decyduje, *kiedy* reflektor świeci; wiązka `v_c` decyduje, *w którą stronę* pcha.
To nie jest jeden promień od reflektora do koloidu. W filmach 1–6 kierunek pchnięcia był umownym kierunkiem `d_c` ze sfery
Fibonacciego.

**Z3. Jasność zależy od położenia samego koloidu.** `z_jc = GELU(w_c · LN(x_j) + b_c)`: reflektor świeci tym mocniej, im
lepiej koloid leży na linii celownika. To pole sił zależne od położenia, ale nie gradient żadnego potencjału: część wirowa
jest prawie tak duża jak potencjalna (Z15). Reflektor nie zna numeru koloidu; widzi tylko jego 192 liczby, a ten sam reflektor
działa równolegle na wszystkie 600 koloidów.

**Z4. Siły działają raz, przy przejściu przez blok.** Blok MLP to jeden krok: `Δx_j = Σ_c z_jc v_c + b₂`. Nie ma ciągłego
trzymania koloidu w pułapce ani lepkości (pre-LN `x + F(LN(x))` to jeden jawny krok Eulera, w pełni przetłumiony). Wcześniej
działa 5 bloków, każdy z własnymi 768 reflektorami i z uwagą. Koloid nigdy więc nie „spływa” do ogniska pułapki (Z25).

**Z5. Wiązka pcha tylko w jedną stronę, więc potrzebne są przeciwlasery.** GELU jest prawie nieujemne (minimum ≈ −0,17),
więc reflektor c pcha mocno tylko w stronę `+v_c`, a w stronę `−v_c` może ciągnąć najwyżej słabo. Mocne pchnięcie w stronę
`−v_c` wymaga innego reflektora c' z `v_c' ≈ −v_c`. Słabo ujemnych diod jest jednak dużo i nie wolno ich pomijać.

*Ile jest ujemnych diod* (próg 0,05; `opis_ukladu/POMIARY_OPISU_S47.json`, część B; mediana po tokenach zdjęcia TRAIN 0):

| epoka | 0 | 1 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| diody jasne (z > 0,05) | 45 % | 34 % | 28 % | 24 % | 24 % | 21 % | 19 % |
| diody słabo ujemne (z < −0,05) | 42 % | 53 % | 59 % | 53 % | 44 % | 42 % | 42 % |

Uczenie zapala coraz mniej reflektorów na token (rzadszy kod). Liczba „49 % na starcie, 79 % w 60. epoce” liczy wszystkie
diody z z < 0 bez progu (`opis_ukladu/KONTROLA_WZOROW.txt`); obie liczby są prawdziwe, ale nie wolno ich mieszać.

*Ile siły niosą ujemne diody.* Mediana po tokenach ‖Σ_c (z⁻_jc − z̄⁻_c) v_c‖ / ‖F_j − F̄‖, z⁻ = min(z, 0): 24 % na starcie,
18 % w 1. epoce, 9 % w 20., 13 % w 60. To stosunek długości wypadkowych, nie udział w sumie długości 768 sił (PDF pkt 3.6 opisuje
to nieściśle; sumę długości opisuje Z18).

*Rysowanie.* Jasności ze znakiem: ujemne diody na błękitno, w prawdziwej proporcji do dodatnich (od filmu 20 i w aplikacji).

*Przeciwlasery* (cos najbardziej przeciwnego partnera, wagi ostatniego bloku s47, `PRZELICZENIE_ZASADY_S47.json`, część P):

| epoka (krok) | reflektory z partnerem cos < −0,5 | mediana cos partnera | udział 3 składowych w kierunkach v_c | mediana \|v_c\| |
|---|---|---|---|---|
| 0 (0) | 0 % | −0,22 | 3 % | 0,29 |
| 1 (1094) | 0 % | −0,23 | 5 % | 0,29 |
| 5 (5470) | 1 % | −0,28 | 13 % | 0,33 |
| 10 (10 940) | 14 % | −0,37 | 18 % | 0,43 |
| 20 (21 880) | 36 % | −0,46 | 22 % | 0,60 |
| 40 (43 760) | 52 % | −0,51 | 23 % | 0,78 |
| 60 (65 640) | **53 %** | −0,51 | 22 % | 0,84 |

Na starcie przeciwlaserów nie ma; **tworzy je uczenie** od około 5. epoki, a wiązki rosną 2,9 raza. Pary są przybliżone
(najbardziej przeciwna ma cos −0,76). Bieg gęsty (inne wagi, ten sam przepis) daje prawie te same statystyki, choć same wagi
rozeszły się całkowicie: to cecha przepisu, a nie jednego biegu.

**Z6. Słaba siła zwrotna (zmierzona).** Mediana κ = cos(γ ⊙ w_c, v_c) spada z 0,00 na starcie do −0,05 w 5. epoce, −0,12
w 10., −0,17 w 20. i −0,18 w 40. i 60. Związek z Jakobianem jest dokładny: `tr J ≈ (1/σ) Σ_c GELU′(a_c) (γ ⊙ w_c)ᵀ v_c`,
więc ujemny cosinus daje ujemny ślad (kontrakcję objętości). Ślad/192 spada z −0,0006 do −0,016 w 60. epoce. Dla skończonego
kroku `x → x + F(x)` liczy się log|det(I + J)|: na kierunek od −0,0008 do −0,021, ujemny dla wszystkich 600 tokenów; objętość
otoczenia tokenu maleje około 1,2 raza na starcie i 55 razy w 60. epoce. Kontrakcja na kierunek słaba, w objętości duża.

**Z7. Między koloidami działa tylko uwaga.** Reflektory MLP działają na każdy koloid osobno. Bezpośrednie siły koloid–koloid
daje wyłącznie uwaga. W czasie uczenia „oddziaływanie” tokenów to sprzężenie przez wspólne wagi (jądro wpływu, empiryczne
NTK), a nie siła: około 85–90 % energii kroku to wspólne pole wag (test jądra, fala 4).

**Z8. W czasie uczenia cząstkami są reflektory, nie tokeny.** Uczenie przestawia w_c, v_c i b_c (mediana cos między 1. a 60.
epoką: celowniki 0,24, wiązki 0,14). Token nie ma własnego stanu w czasie uczenia: `z_j(t) = f(x_j; θ_t)`. To, co widać jako
ruch tokenu w czasie uczenia, to przesunięcie **plamki** (stanu tokenu na końcu toru przez głębokość). Pamięć ma całe uczenie,
wspólnie dla wszystkich tokenów: momenty AdamW (β₁ = 0,9 ≈ 10 kroków, β₂ = 0,999 ≈ 1000 kroków) i średnie BatchNorm łodygi;
norma gradientu jest w każdym kroku przycinana do 1. Szybkie drgania plamki w 84–98 % pochodzą z wcześniejszych bloków, wolny
dryf w 75 % z przestawiania reflektorów ostatniego bloku (Eden, `eden/wyniki/TEST_SFERY.json`, `wiazki_koloidy/WIAZKI.json`).

**Z9. Czego nie mówić.**
- Nie mówić, że koloid „nadąża za pułapką” w czasie uczenia ani odwrotnie. Pomiar (Z27): ruch koloidu i zmiana pola wiązek
  bloku 6 pchają wyjście bloku w tę samą stronę (wspólne zejście po gradiencie), a po odjęciu własnych drgań żadna strona nie
  reaguje na drugą z opóźnieniem. Jeden krok uczenia przestawia naraz pułapki i koloidy; nikt nikogo nie goni.
- Nie mówić, że średnia skoków koloidu ciągnie koloid. Jest liczona z jego ruchu i z kroków przyszłych (okno symetryczne
  ±300), więc „rusza” przed koloidem. Do pytań o przyczynę trzeba średniej tylko z przeszłości albo pola z wag (Z24, Z25).
- Nie mówić, że krok uczenia to temperatura. W biegach z pokrętłami τ_1e wynosi 13, 6 i 3 kroki dla lr 1,25e-4, 5e-4 i 2e-3
  (τ ∝ η^−0,5, nie η^−1); pomiar FDT pokazuje, że lr działa jak zegar (Z24).
- Nie mówić, że korelacje sąsiadów (0,6–0,85) to oddziaływania: wynikają ze wspólnego napędu przez wagi.
- Nie mówić, że koloid siedzi w ognisku pułapki. Zero pola wiązek w zasięgu 3 promieni chmury istnieje tylko w 7,7 %
  checkpointów (Z25); zwykle koloid jest niesiony prądem wiązek.

**Z10. Zasady rysowania (filmy od 7. i aplikacja).**
- Reflektor c stoi na sferze w punkcie −P·v̂_c i świeci do środka, więc pcha w stronę +P·v̂_c. P to stały rzut 192D → 3D, zawsze
  podany. Trzy składowe niosą tylko 3–23 % wariancji kierunków v_c: rzut jest ilustracją, a nie pomiarem.
- Jasność wiązki = z_jc; długość strzałki pchnięcia = z_jc · |P v_c|; strzałka wypadkowa = P · Δx_j.
- Celownik pokazywać osobno (to inny wektor niż wiązka).
- Drgania w czasie uczenia rysować jako drżenie wiązki wejściowej (z wcześniejszych bloków), a wędrówkę reflektorów jako
  przyczynę wolnego dryfu.
- Każda nowa animacja to nowy skrypt `film_lasery_N_<opis>.py` i nowe `.mp4`; starych nie przerabiać.
- Zawsze podać, która z trzech umów położenia obowiązuje (Z16).
- Jasności ze znakiem (Z5). Rysowany ruch musi być sumą rysowanych wektorów (Z20); zmiana skali kamery, obrót kamery i skok
  między krokami uczenia nie mogą wyglądać jak ruch.
- Obrót wiązek w czasie uczenia rysować strzałkami jak w filmie 23 (Z22).
- Pułapkę rysować jako pole wiązek z ogniskiem (Z25, okienko 11), a średnią skoków podpisywać tą nazwą, nigdy „pułapka”.

**Z11. Matryca reflektorów bloku: jak reflektory uczą się.** Jeden blok MLP to jedna matryca 768 reflektorów (jak reflektor
matrycowy w samochodzie). *Przełączanie* (które świecą i jak jasno) to aktywacje z_jc i dzieje się natychmiast przy każdym
tokenie; to nie jest uczenie. *Uczenie* to powolne przestawianie samych reflektorów. Między krokiem 0 a 65 640: wiązka v_c
tego samego reflektora ma medianę cos 0,11 (obrót o około 84°), celownik cos 0,20, długość wiązki rośnie 2,9 raza (celownika
1,7 raza); w oknach po 5 epok cos wiązki 0,77 → 0,86–0,91 → 0,93–0,97, więc ruch jest płynny i zwalnia.
*Matryca obraca się jako całość tylko w małej części* (jeden obrót R w 192D, Prokrustes; zawsze podawać definicję):
(a) surowe v_c, kolejne etapy: wspólny obrót wyjaśnia 10–13 % zmiany wobec 6–12 % zera; (b) kierunki v̂_c od kroku 0:
14 → 24 % wobec 11–12 % zera w epokach 1–60. Celowniki: 21 % wobec 12 % zera na całej drodze. To nie jest nadążanie za
strumieniem (Z27): wspólny obrót celowników zachodzi w kierunkach, których strumień prawie nie używa. Co najmniej trzy czwarte
zmiany to osobne przestawianie każdego reflektora, w tym tworzenie par przeciwnych (Z5).

**Z12–Z14. Sfera reflektorów (obrazy z filmów 10–12).** Reflektor to neuron, nie token (Z13): stoi za sferą i ma 600 diod,
po jednej na koloid; dioda j świeci z jasnością z_jc. Wspólne wagi wynikają z budowy: reflektor ma jedno ustawienie wiązki dla
wszystkich koloidów. Sfera działa telecentrycznie: po załamaniu wszystkie 600 wiązek reflektora c biegną równolegle w kierunku
v_c, każda do swojego koloidu (wiązki jednego reflektora różnią się tylko jasnością). W wariancie z filmu 12 i z aplikacji
reflektory stoją na sferze Fibonacciego (przypisanie węgierskie do kierunków końcowych; średni cos miejsca 0,9865 na wagach s47),
a kąt załamania (ważony jasnością) spada z 108° na starcie do 27° w 60. epoce; spadek wynika z doboru miejsc do kierunków
końcowych i nie jest odkryciem. Warunek z sieci: niezależne może być tylko celowanie wiązki w koloid; kierunek wiązki w sferze
musi być wspólny. Położenie liczone jak w filmach 8–12 (P·W₂(z_j − z̄)) to przesunięcie od bloku, a nie stan tokenu (Z16).

**Z15. Pole wiązek bloku wokół tokenu to siodło z wirem (dokładny Jakobian).** Wokół tokenu blok MLP działa jak pole sił
`F(x_j + δ) ≈ F(x_j) + J_j δ`, `J_j = W₂ diag(GELU′(a_j)) W₁ J_LN(x_j)`. Wzór jest dokładny (F zgadza się z wyjściem modelu do
5·10⁻⁶, J z autograd w float64 do 10⁻¹⁵). Mediany po 600 tokenach TRAIN 0, epoka 0 → 60 (część J przeliczenia):
- ujemnych wartości własnych części symetrycznej S jest stale około 51 %: połowa kierunków ściąga, połowa wypycha;
- symetryczność ‖S‖²/‖J‖² rośnie z 0,50 do 0,58, a wir ‖A‖/‖S‖ spada z 0,99 do 0,86 (macierz losowa: 0,503 i 0,995);
- najsilniejszy kierunek ściągający rośnie z −0,30 do −1,02, wypychający z 0,30 do 0,60; ‖J‖₂ z 0,47 do 1,39;
- mapa jednego kroku I + J: około połowy modów ma |1 + λ| > 1, promień spektralny 1,2–1,3: **mapa bloku nie ściska**;
- w 60. epoce (narzędzie / tło / token 316): wir 0,93 / 0,86 / 0,96; ‖J‖₂ 1,72 / 1,39 / 1,38.
Dla porównania: w pęsecie optycznej część wirowa to około 1–4 % sztywności (Pesce i in. 2009), a wyuczone autoenkodery są
symetryczne w 95–99 % (Im i in. 2016). W polach FFN dużych modeli językowych przybliżone punkty krytyczne to też głównie siodła
(Mudarisov i in. 2026). Pole wiązek to więc siodło z silnym wirem, które uczenie lekko ściąga; jego ognisko opisuje Z25.

**Z16. Trzy umowy położenia koloidu na ekranie (PDF pkt 3.2).** P ∈ R^(3×192), liczby zawsze w 192 wymiarach.
1. *Filmy 8–12 i 23:* p_j = s · P · W₂ (z_j − z̄): rzut przesunięcia od bloku, nie stanu tokenu.
2. *Aplikacja (dane wersji ≥ 2) i film 20:* p_j = s · P (x_j − x̄), rzut stanu tokenu tuż przed MLP bloku 6. Przesunięcie koloidu
   przez MLP to wtedy dokładnie Σ_c (z_jc − z̄_c) P v_c (Z20). P = 3 główne kierunki znormowanych v_c z ostatniego checkpointu,
   wiersze ortonormalne. Tę umowę używa też pułapka (Z25).
3. *Film 22:* rzut kierunku tokenu po LayerNorm bez γ i β (to, co czyta każdy blok, Z17); długość strumienia osobno.

**Z17. Jedyna prawdziwa sfera to LayerNorm.** `LN(x) = γ ⊙ (x − μ1)/σ + β`. LayerNorm rzutuje token na hiperpłaszczyznę ⟂ 1,
normuje na sferę o promieniu √192 i rozciąga przez γ (γ ∈ [0,20; 1,65] w 60. epoce, na starcie 1). Jakobian LN zeruje kierunek
1 i (do ε) kierunek radialny: reflektory widzą tylko kierunek tokenu, nie jego długość. Celownik w przestrzeni tokenu to
`s_c = γ ⊙ w_c` po odjęciu średniej, próg zapłonu `−(w_cᵀβ + b_c)`. Długość tokenu rośnie przez bloki (Z18), więc ta sama
siła obraca długi token o mniejszy kąt: długość działa jak bezwładność.

**Z18. Siły wiązek na koloidzie w większości się znoszą (PDF pkt 3.8).**

| epoka | 0 | 1 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| ‖Σ_c z_c v_c‖ / Σ_c \|z_c\| ‖v_c‖ (wypadkowa / suma długości) | 0,054 | 0,109 | 0,216 | 0,217 | 0,198 | 0,147 | 0,144 |
| ‖F(x_j)‖ / ‖x_j‖ | 0,19 | 0,13 | 0,30 | 0,30 | 0,35 | 0,28 | 0,23 |

Z sumy długości 768 sił zostaje 5–22 %; na starcie to poziom losowych kierunków (≈ 0,045). Norma ‖x_j‖ przed MLP bloku 6 rośnie
z 14,7 do 90,5, bo każdy blok dodaje swoje wyjście do strumienia; odległości między tokenami porównywać po LayerNorm (Z21).

**Z19. Uwaga to wiązki między koloidami; pełna sieć to łańcuch 6 stopni (dokładne).** Koloid j dostaje od koloidu k wektor
`f_jk = Σ_h A^h_jk W_O^h W_V^h LN₁(x_k)`, a jego przesunięcie w podkroku uwagi to Σ_k f_jk + b_O. Każdy blok to najpierw uwaga,
potem MLP z 768 własnymi reflektorami: jedno przejście ma 12 podkroków i 13 stanów (krok Eulera z rozszczepieniem
Lie–Trottera, Lu i in. 2019). Kontrola w filmie 20 i w aplikacji: odtworzone przesunięcia zgadzają się z modelem do 10⁻⁶.

**Z20. Rysowany ruch jest sumą rysowanych wektorów (dokładny rozkład względem środka chmury).**
- uwaga: `Δx_j − Δx̄ = Σ_{k,h} (A^h_jk − Ā^h_k) W_O^h W_V^h LN₁(x_k)`;
- MLP: `Δx_j − Δx̄ = Σ_c (z_jc − z̄_c) v_c`.
Wyrazy wolne b_O i b₂ znikają. W aplikacji suma rysowanych wektorów równa się rysowanemu ruchowi z błędem około 10⁻⁵.

**Z21. Ostatni blok rozsuwa tokeny wewnątrz grup (pomiar).** Mnożnik odległości widzianej przez następną warstwę
`r_kl = ‖LN̂(x_k + F_k) − LN̂(x_l + F_l)‖ / ‖LN̂(x_k) − LN̂(x_l)‖`, mediana (odsetek par przyciąganych):

| krok (epoka) | narzędzie–narzędzie | narzędzie–tło | tło–tło |
|---|---|---|---|
| 0 (0) | ×1,010 (14 %) | ×1,003 (41 %) | ×1,006 (34 %) |
| 1094 (1) | ×1,021 (27 %) | ×1,004 (41 %) | ×1,035 (20 %) |
| 10 940 (10) | ×1,021 (29 %) | ×1,053 (1,5 %) | ×1,082 (0,8 %) |
| 65 640 (60) | ×1,084 (26 %) | ×1,011 (25 %) | ×1,100 (0,3 %) |

W 60. epoce blok rozsuwa tło–tło o 10 %, narzędzie–narzędzie o 8 %, narzędzie–tło prawie wcale. Zdanie „blok rozsuwa narzędzie
mocniej niż tło” jest prawdziwe tylko względem tokenu 316.

**Z22. Gradient = obrót wiązek reflektorów (PDF rozdział 5).** W czasie głębokości gradientu nie ma. W czasie uczenia:
`∂L/∂v_c = Σ_j z_jc δ_j`; `g_jc = GELU′(a_jc) v_cᵀ δ_j`; `∂L/∂w_c = Σ_j g_jc LN₂(x̃_j)`; `∂L/∂b_c = Σ_j g_jc`
(δ_j = gradient straty po stanie tokenu za blokiem 6). Wiązkę ciągną poprawki −δ_j koloidów, na które reflektor świeci, ważone
jasnością. Celownik obraca się w stronę tokenów, którym pchnięcie pomogło (v_cᵀ δ_j < 0), i od tych, którym szkodzi; czynnik
GELU′ sprawia, że uczą go głównie koloidy w oknie cieplnym (Z26). AdamW: weight decay tylko skraca wiązkę (sprężyna nie obraca),
Adam zmienia kierunek kroku (osobno na współrzędną), obcięcie gradientu działało w 100 % paczek, krok po rozgrzewce stały.
*Pomiar* (mediana po 768 reflektorach, °/100 kroków):

| krok (epoka) | 0 | 300 | 1094 (1) | 21 880 (20) | 65 640 (60) |
|---|---|---|---|---|---|
| drganie (≥ 100 kroków) | 2,88 | 3,75 | 3,56 | 3,53 | 2,79 |
| dryf netto (≥ 2000 kroków) | 1,14 | 1,15 | 1,04 | 0,63 | 0,52 |

Drganie ma prawie stałą wielkość (stały krok), dryf netto zwalnia o połowę; wykładnik kąt ∝ Δ^α spada z 0,60 do około 0,37.
Wiązka obraca się od startu do 60. epoki o 84°, celownik o 78°. Nazwa „starzenie” to analogia, nie dowód.

**Z23. Wir jednego reflektora (dokładne algebraicznie).** Jeden reflektor wnosi do Jakobianu macierz rzędu 1 `M = v sᵀ`,
`v = GELU′(a_c) v_c`, `s = J_LNᵀ w_c`. Dla κ = cos(s, v): `‖A‖²/‖S‖² = (1 − κ²)/(1 + κ²)`, a niezerowe wartości własne S to
½(v·s ± |v||s|). Reflektor prostopadły (κ = 0) daje po równo siodła i wiru; przeciwny (κ = −1) czystą sprężynę; zgodny (κ = +1)
czyste wypychanie. Na starcie κ ≈ 0, więc suma ma poziom macierzy losowej (zmierzone 50 % i 0,993). W 60. epoce κ ≈ −0,18
przewiduje dla pojedynczych reflektorów wir 0,97; zmierzone 0,86. Pomiar z 2026-10-13 (Z26, część C): pojedyncze reflektory
tłumaczą około połowy spadku wiru w 60. epoce (przewidują 0,923 wobec zmierzonego 0,859), a w 1. epoce nic (0,993 wobec 0,936);
resztę dają wyrazy krzyżowe, czyli zgranie wielu reflektorów.

**Z24. Trzy rzeczy, które dawniej nazywano „pułapką”, i temperatura (2026-10-13, nazwy z 2026-10-14).**
- (a) **Pułapka** = pole wiązek F wokół tokenu, jego Jakobian (Z15, Z23) i ognisko (Z25). Czas to głębokość, szumu nie ma,
  temperatura 0. Siodło z wirem, nie studnia.
- (b) **Średnia skoków koloidu** = symetryczna średnia krocząca położenia plamki z ±300 kroków uczenia. Tak liczą aplikacja
  (zielona poświata) oraz `experiments/koopman_pulapki_20261013` i `experiments/fdt_temperatura_20261013`. Zawiera kroki
  przyszłe, więc „rusza” przed koloidem i nie może być przyczyną jego ruchu; ta sama średnia daje fałszywą „pamięć”
  (surogat R² 0,78 > dane 0,25) i fałszywą przewidywalność.
- (c) **Dolina straty**: pułapka w przestrzeni wag (drgania w poprzek doliny i dryf wzdłuż niej, Pashakhanloo i Koulakov 2023).

Wyniki FDT (`experiments/fdt_temperatura_20261013`, repliki z identycznymi paczkami i małym polem testowym):
- średnia skoków (b) spełnia relację Einsteina: T ≈ 3,6·10⁻⁴ (podprzestrzeń N) i 5,7·10⁻⁴ (V), stałe od kroku 300 do 7000,
  niezależne od lr;
- sama plamka nie ma jednej temperatury: T_eff spada mniej więcej jak 1/t, odpowiedź na stałą siłę rośnie bez nasycenia;
  jedyną sprężyną jest weight decay (przy wd = 1,0 odpowiedź się nasyca); lr działa jak zegar;
- Koopman na średnich skoków: lokalna wirowość = surogat; realny jest tylko obrót całej chmury (2–3 razy surogat); Koopman nie
  przewiduje lepiej niż „stoi”.
Nie mylić wiru pułapki (a) z obrotem chmury (b). Relacje FDT dla SGD (Yaida 2018) dotyczą parametrów, a nie reprezentacji.

**Z25. Pułapka optyczna: ognisko wiązek wokół koloidu (2026-10-14, `app/ogniskowa.py`, okienko 11).**
- *Definicja.* W przekroju 3D przez koloid rozpiętym rzutem P (Z16, umowa 2): `x(u) = x_j + Pᵀu`,
  `f(u) = P Σ_c (z_c(x(u)) − z̄_c) v_c` (pole wszystkich 768 wiązek bloku 6 względem środka chmury). `f(0)` to dokładny skok
  koloidu w bloku 6. **Ognisko** = punkt, w którym pchnięcia wszystkich wiązek się znoszą (f = 0; Newton od koloidu, do
  3 promieni chmury r99). Charakter ogniska z wartości własnych λ Jakobianu 3 × 3: Re λ < 0 ściąga, > 0 wypycha, Im λ ≠ 0 obraca.
- *Kontrola.* Jasności z przeliczenia zgadzają się z przejściem sieci do 4·10⁻⁶, położenia z cache aplikacji do 3·10⁻⁷.
- *Wynik* (s47, 806 checkpointów, token narzędzia 226, siatka 17³, zasięg 0,3 r99; cache obok historii w
  `app/cache_uczenia/lasery/eafb5a973be3d490/ogniskowa_t226_G17_z030.npz`): **zero pola w zasięgu 3 r99 istnieje tylko w 62
  z 806 checkpointów (7,7 %)**, głównie na początku uczenia, gdy skok koloidu jest mały (krok 0: ognisko 2,0 r99 od koloidu,
  skok 0,07 r99). Zwykle koloid jest **niesiony prawie jednorodnym prądem wiązek**: krok 28 800 — brak zera, skok 0,62 r99,
  liniowe oszacowanie zera 2,2 r99 dalej, a wokół koloidu siodło z wirem (ściąga −0,19, obraca 0,21). W 60. epoce ognisko
  pojawia się 2,8 r99 od koloidu (ściąga −0,16, obraca 0,07), a w samym koloidzie przekrój 3D ściąga we wszystkich trzech
  kierunkach (−0,48, −0,17, −0,15).
- *Ognisko naprawdę przesuwa koloid (kontrola 2026-10-14).* Suma wkładów 768 wiązek `Σ_c (z_jc − z̄_c) v_c` równa się prawdziwemu
  przesunięciu koloidu z przejścia sieci (stan po bloku 6 minus stan przed MLP, względem środka chmury, pełne 192D): błąd względny
  ≤ 1,3·10⁻⁶ we wszystkich 806 checkpointach (token 226). W podkrokach przejścia przez sieć (checkpoint 65 640) suma wkładów =
  ruch w podkroku: MLP bloku 1, 4 i 6 z błędem 5·10⁻⁷, 3·10⁻⁷ i 8·10⁻⁷. W podkrokach uwagi koloid przesuwają inne koloidy, nie
  lasery: 600 wkładów `Σ_h (A^h_jk − ā^h_k) u^h_k` = ruch z błędem 2·10⁻⁶. Skok to 16–68 % sumy długości wkładów (blok 6: 16 %,
  blok 4: 68 %, blok 1: 32 %): reszta pchnięć się znosi (Z18).
- *Kształt ogniska.* Tensor kierunków `T = Σ jas_c d_c d_cᵀ / Σ jas_c` (izotropowo 1/3 na oś): w blokach 1 i 6 prawie kulisty
  (0,42/0,34/0,24 i 0,45/0,31/0,24, wydłużenie ×1,1–1,2), w bloku 4 wydłużony (0,69/0,19/0,13, ×1,7) wzdłuż kierunku, z którego
  przychodzi najwięcej światła. W 3D rzucie to ilustracja; liczby z 3 z 192 wymiarów.
- *Żadna wiązka nie dominuje:* największy wkład ma 1–5 % sumy wkładów; pole i ognisko to dzieło setek wiązek naraz.
- *Ograniczenia.* Przekrój 3D to 3 z 192 kierunków (zero w 192D to osobne pytanie; Jakobian w 192D jest zawsze siodłem, Z15).
  Blok daje jeden krok (Z4), więc koloid nie spływa do ogniska; pułapka opisuje kształt pola, w którym koloid wykonuje skok.
  Na razie tylko blok 6 i oś uczenia (podkroki w głębokości: do zrobienia).

**Z26. Temperatura bramek: reflektory stygną (2026-10-13, `temperatura_pulapki/temperatura_pulapki_s47.py` →
`TEMPERATURA_PULAPKI_S47.json`).** GELU ≈ obsadzenie Fermiego: `Φ(a) ≈ 1/(1 + e^{−1,702a})`, kT₀ ≈ 0,59; temperatura reflektora
`T = 0,59 / sd_j(a)` (rozrzut pobudzenia po tokenach). Okno cieplne |a| < 1,28 to diody ani pewnie zapalone, ani zgaszone.
- *A (806 checkpointów, blok 6, TRAIN 0):* T 2,58 → 1,65 (1. ep.) → 1,08 (5.) → 0,89 (10.) → 0,76 (20.) → minimum 0,70 (24.)
  → 0,82 (60.). Okno cieplne 97 % → 45 %, zgaszone 1 % → 49 %, średnie pobudzenie 0 → −1,23; okno niesie 95 % → 64 % ‖J‖².
  Układ stygnie jak szkło dochodzące do równowagi z otoczeniem: gradient ochładza, weight decay grzeje.
- *B (termometr wirtualny J_T = E[J(x + δ)], δ = s·σ_j·ε, kontrola Steina):* kształt pola stały do s ≈ 0,3; przy s = 2
  (60. ep.) ujemne 51 → 54 %, wir 0,86 → 0,78, ‖J‖ 1,39 → 0,52. Siodło trwa w każdej temperaturze, nigdy studnia.
- *C:* pojedyncze reflektory tłumaczą około połowy spadku wiru (Z23).
- *Chłód ≠ dobra detekcja:* sieć zamarza, zanim nauczy się dobrze rozpoznawać (w ViT-S po 2. epoce bloki są zimne przy 11 %
  trafności). Temperatura mówi, że sieć podjęła decyzje (prawie zero-jedynkowy kod zapalonych reflektorów), a nie że dobre.
- Literatura (Undermind, 2 przeszukania 2026-10-13): nikt tak nie mierzy temperatury sieci; najbliżej samozorganizowane
  wyżarzanie, λ-GELU, wiry Browna (Sun i in. 2009) i non-Hermitian non-equipartition.

**Z27. Nadążanie: pole wiązek i koloid zmieniają się razem (dawny `NADAZANIE_PULAPKI_S47.md`, 2026-10-12;
`pomiar_nadazania_pulapki.py` → `NADAZANIE_PULAPKI_S47.json`).** Blok 6 s47. Rozkład zmiany wyjścia MLP między checkpointami:
ruch koloidu `Δ_in(t) = F(x(t+1); θ6(t)) − F(x(t); θ6(t))` i zmiana pułapki (pola wiązek) `Δ_W(t) = F(x(t+1); θ6(t+1)) − F(x(t+1); θ6(t))`.
C(τ) = ⟨cos(Δ_in(t), Δ_W(t+τ))⟩; a(1), c(1) = regresje z kontrolą własnej przeszłości (pułapka po koloidzie, koloid po pułapce).
Dane: 4 zdjęcia TRAIN (2400 tokenów), 747 checkpointów.

| faza | przedział | udział pułapki w energii zmiany | C(0) | C(+1) | C(−1) | zero C (sd) | a(1) | c(1) |
|---|---|---|---|---|---|---|---|---|
| kroki 0–1000 | 10 kroków | 13 % | +0,149 | −0,034 | −0,071 | 0,010 | −0,023 (−5,0 sd) | −0,077 (−2,8 sd) |
| epoki 1–20 | 100 kroków | 23 % | +0,089 | −0,047 | −0,044 | 0,011 | −0,017 (−2,5 sd) | −0,027 (−1,7 sd) |
| epoki 20–40 | 100 kroków | 22 % | +0,093 | −0,030 | −0,048 | 0,008 | −0,006 (−0,9 sd) | −0,026 (−1,5 sd) |
| epoki 40–60 | 100 kroków | 26 % | +0,057 | −0,013 | −0,031 | 0,009 | +0,006 (+1,0 sd) | −0,029 (−1,5 sd) |

- W tym samym przedziale oba ruchy pchają wyjście bloku w tę samą stronę (C(0) dodatnie, 6–15 sd): wspólne zejście po gradiencie.
  Gdyby pułapka goniła koloid, żeby utrzymać wyjście, C(0) byłoby ujemne.
- Przy opóźnieniu korelacje są ujemne w obie strony i podobne: każda część drga osobno tam i z powrotem (autokorelacja przy
  jednym przedziale −0,33 do −0,37 dla koloidu, −0,18 do −0,22 dla pułapki). Po kontroli własnej przeszłości żaden kierunek nie
  przekracza 3 sd po 1. epoce.
- Tylko w rozgrzewce (kroki 0–1000) słaba wzajemna kompensacja (pułapka odrabia 2,3 % ruchu koloidu w 10 krokach, koloid około
  8 % zmiany pułapki). 60–70 % współruchu to wspólny „wiatr” wszystkich tokenów.
- *Celowniki nie obracają się za strumieniem* (32 zdjęcia, okna 1-epokowe, baza 95 % wariancji strumienia): strumień obraca się
  co epokę jako całość (33–35 % zmiany wobec 0,1 % zera), ale ten obrót przyłożony do celowników działa jak obrót wokół losowych
  osi (np. −1,33 wobec −1,26 ± 0,04 w epokach 40–60), a generatory obrotów nie są zgodne (cos −0,026 przy zerze +0,001 ± 0,024).
  Wspólny obrót celowników (Z11) siedzi w kierunkach, których strumień prawie nie używa; wchłania go próg b_c.
- Ograniczenia: rozdzielczość = odstęp checkpointów (10, potem 100 kroków), więc szybsza reakcja wchodzi do C(0).

**Z28. Ten sam układ w ViT-S/16 na ImageNet (ViT Model Zoo, Falk i in. 2025; 2026-10-13).** Wagi przebiegu ziarna 42 po 1., 2.
i 90. epoce (`assets/ViTModelZoo_S16_ImageNet/`, walidacja na Imagenette: 76,3 % top-1 z 1000 klas po 90. epoce,
`experiments/vit_modelzoo_20261013/`):
- κ < 0 w blokach 1–11 (blok 0: κ > 0); na końcu najsilniej we wczesnych blokach (−0,26 w bloku 2), około 0 w blokach 9–11.
  To odwrotnie niż w modelach językowych (Gerstner i Schütze: neurony wzmacniające wcześnie, osłabiające późno).
- Przeciwlasery na brzegach sieci: 60 % w bloku 0, 19 % w bloku 11, 1–9 % w środku; wiązki rosną około 13 razy (wd 10⁻⁴).
- Temperatura bramek po 90. epoce: bloki 0–5 T 0,4–0,74, bloki 7–11 skrajnie zimne (T 0,03–0,12, zgaszone 88–95 %,
  średnie pobudzenie do −53): późne bloki to rzadkie przełączniki.
Brak checkpointu z kroku 0 w Model Zoo; pełna trajektoria 90 epok jeszcze niepobrana (miejsce na dysku).

**Z29. Co gdzie widać w aplikacji (zakładka „Lasery i koloidy”).**
- *Sfera:* 768 stałych reflektorów, wiązki (kolor = reflektor, błękit = ujemna jasność), koloidy (żółte), średnie skoków
  koloidów (zielona poświata; opcja „Pokazuj”), wektory i drogi; opcjonalnie pułapka (Z25).
- *Okienko 11 (zbliżenie):* ognisko JEDNEGO wybranego koloidu: 768 wiązek zbiega się w nim, każda z kierunku, w który pcha
  (wkład (z_jc − z̄_c)·P v_c), w kolorze reflektora, ze światłem wiązek gaussowskich (przewężenie w koloidzie) i osiami kształtu
  ogniska; czas: oś uczenia (blok 6 w każdym checkpoincie) albo podkroki (MLP: wiązki bloku; uwaga: 600 wkładów od innych koloidów);
  w opisie kontrola „suma wkładów = ruch”; wkłady (z_jc − z̄_c)·P v_c jako strzałki albo łańcuch głowa–ogon, którego koniec to skok (Z18, Z20). Opcjonalnie
  lokalne pole: mgła w kolorach wiązek (jasna w ognisku), strzałki pełnego pola albo kształtu
  pułapki (pole minus skok koloidu), osie siodła i wir, skok koloidu, ślady koloidu i ogniska, kamera śledząca pułapkę, koloid
  albo środek pola; odtwarzanie „najpierw lasery, potem koloid” (obie zmiany pochodzą z tego samego kroku uczenia).
- *Okienko „Przejście przez sieć” (podkroki):* zegar głębokości, 13 stanów, uwaga jako linie j–k, wiązki bloku (Z19, Z20).
- *Okienko 10:* obrót wiązek (Z22). Okienka 1–3: rzuty płaskie filmu 2 (centra studni = ta sama średnia skoków).

## 4. Wierność fizyczna w skrócie

| Element | Status |
|---|---|
| 768 sił działających na token w bloku MLP, jasność zależna od położenia | **dokładne** (wzór bloku) |
| przeciwlasery i różnica jasności wzdłuż osi | **dokładne** jako opis; pary przybliżone, tworzy je uczenie |
| siła zwrotna reflektora | **słaba, zmierzona** (κ = −0,18; ślad J/192 = −0,016 w 60. epoce) |
| pole wiązek bloku (Jakobian) | **dokładne** jako wzór; zmierzone: siodło z wirem, ‖A‖/‖S‖ = 0,99 → 0,86 |
| **pułapka = ognisko wiązek** (zero pola w przekroju 3D) | **dokładne** jako obliczenie; istnieje tylko w 7,7 % checkpointów (Z25) |
| mapa jednego kroku I + J | **pomiar**: ρ ≈ 1,2–1,3; mapa nie ściska (Z15) |
| wir jednego reflektora (1 − κ²)/(1 + κ²) | **dokładne**; pojedyncze reflektory tłumaczą ~połowę spadku wiru (Z23, Z26) |
| temperatura bramek (GELU jako Fermi) | **analogia ilościowa**: stygnięcie 2,6 → 0,7–0,8 zmierzone (Z26) |
| uwaga jako siły między koloidami, łańcuch 6 bloków | **dokładne** wzdłuż głębokości (Z19) |
| rozkład wycentrowany, suma rysowanych wektorów = ruch | **dokładne** (błąd ≈ 10⁻⁵, Z20) |
| LayerNorm jako sfera odczytu | **dokładne** (Z17) |
| reflektory jako ruchome cząstki w czasie uczenia | **dokładne** (to są parametry) |
| gradient = obrót wiązek | **dokładne** jako wzór; zmierzone drganie ≈ 3,5°/100 kroków, dryf 1,04 → 0,52 (Z22) |
| koloid nadążający za pułapką albo odwrotnie | **błędne**: wspólne zejście, brak opóźnionej reakcji (Z27) |
| średnia skoków jako przyczyna ruchu | **błędne**: zawiera kroki przyszłe (Z24) |
| krok uczenia = temperatura | **niepotwierdzone**; lr działa jak zegar (Z24) |
| temperatura średniej skoków | **pomiar FDT**: relacja Einsteina, T ≈ 3,6·10⁻⁴ / 5,7·10⁻⁴ (Z24) |
| temperatura samej plamki | **brak jednej temperatury**; sprężyną jest tylko weight decay (Z24) |
| położenie w 3D | **umowne**; trzy umowy (Z16) |

## 5. Otwarte testy

1. Test RPI i RPT (tasowanie patchy przy zachowanym kodowaniu pozycji), więcej ziaren bez kodowania pozycji (Mannes 2026).
2. Surogaty rzutu (błądzenie losowe, z pędem, OU) przez ten sam rzut 3D (Antognini i Sohl-Dickstein 2018).
3. Profile głębokości w każdym z 806 checkpointów (Isobe i in. 2026, Herty i Liu 2026).
4. Jądro lokalnej sprężystości (He i Su 2019) przy AdamW.
5. Prawdziwy czas relaksacji z autokorelacji wychylenia; skalowanie τ ∝ 1/(ηλ) przy AdamW (plan E1).
6. Geometria dryfu: macierz token–token, Prokrustes, dyfuzja kątowa (Qin i in. 2021).
7. Film 7: reflektory w punktach −P·v̂_c, przeciwlasery, osobno celowniki.
8. Zrobione (negatywne): wspólny obrót celowników nie jest nadążaniem za strumieniem (Z27).
9. Filmy 8–12 na wagach s47 (częściowo: film 23).
10. Zrobione: uwaga jako wiązki, łańcuch 6 bloków (Z19).
11. Pole wiązek i ognisko we wszystkich 6 blokach i w trybie podkroków; ognisko w pełnym 192D, nie tylko w przekroju 3D.
12. Hipoteza gęstej głowy: czy rozsuwanie tokenów narzędzia (Z21) koreluje z różnicą ich celów (l, t, r, b).
13. Przeciwlasery jako pary cech wejściowych (Shang i in. 2016, Elhage i in. 2022).
14. Wiązki poprawek δ_j jako obraz (wymaga gradientu na paczce, Z22).
15. Starzenie wiązek przy innym lr i z zapisem stanu AdamW (Z22).
16. Częściowo zrobione (Z26 C): ile spadku wiru tłumaczą pojedyncze reflektory; osobno udział par przeciwlaserów.
17. Krążący prąd w płaszczyźnie wiru (fontanna prawdopodobieństwa, Roichman i in. 2008): plan E3.
18. Pomiary jak w pęsecie optycznej na tokenach: kalibracja OU, FORMA kontra dryf dokładny, T_eff(ω), produkcja entropii,
    przełączanie reflektorów jak ucieczka Kramersa (plan `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md`, E1–E8).
19. Ognisko a detekcja: czy tokeny narzędzia i tła mają różne pułapki i różne „kody kreskowe” zapalonych reflektorów, i czy
    różnica rośnie z mAP i IoU.
20. Pełna trajektoria ViT-S z Model Zoo (90 epok, 2 ziarna) i DeiT-Tiny na CIFAR-10: czy Z5, Z15, Z25 i Z26 są uniwersalne.

## 6. Literatura

- Geshkovski i in. (2023), A mathematical perspective on Transformers — https://doi.org/10.48550/arXiv.2312.10794
- Rigollet (2025), The Mean-Field Dynamics of Transformers — https://doi.org/10.48550/arXiv.2512.01868
- Herty, Liu (2026), A Mean-Field Theory of Transformers: coupled data–parameter dynamics — https://arxiv.org/abs/2608.25055
- Isobe, Inoue, Imaizumi (2026), Training-Induced Escape from Token Clustering — https://doi.org/10.48550/arXiv.2605.07772
- Agazzi i in. (2026), Stochastic Scaling Limits and Synchronization by Noise in Deep Transformer Models — https://doi.org/10.48550/arXiv.2604.26898
- Pashakhanloo, Koulakov (2023), SGD-induced drift of representation in a two-layer neural network — https://doi.org/10.48550/arXiv.2302.02563
- He, Su (2019), The Local Elasticity of Neural Networks — https://arxiv.org/abs/1910.06943
- Qin i in. (2021), Coordinated drift of receptive fields during noisy representation learning — https://doi.org/10.1101/2021.08.30.458264
- Mannes (2026), Positional Encodings Anchor Spatial Structure in Vision Transformers — https://doi.org/10.48550/arXiv.2606.00124
- Antognini, Sohl-Dickstein (2018), PCA of high dimensional random walks with comparison to neural network training — https://arxiv.org/abs/1806.08805
- Malladi i in. (2022), On the SDEs and Scaling Rules for Adaptive Gradient Algorithms — https://doi.org/10.48550/arXiv.2205.10287
- Mei, Montanari, Nguyen (2018), A mean field view of the landscape of two-layer neural networks — https://doi.org/10.1073/pnas.1806579115
- Geva i in. (2021), Transformer Feed-Forward Layers Are Key-Value Memories — https://doi.org/10.18653/v1/2021.emnlp-main.446
- Lu i in. (2019), Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View — https://arxiv.org/abs/1906.02762
- Riechers (2024), Geometry and Dynamics of LayerNorm — https://doi.org/10.48550/arXiv.2405.04134
- Mudarisov, Burtsev, State (2026), Feed-Forward Steering in Transformer Residual Dynamics — https://arxiv.org/abs/2608.02071
- Im, Belghazi, Memisevic (2016), Conservativeness of Untied Auto-Encoders — https://doi.org/10.1609/aaai.v30i1.10268
- Pesce i in. (2009), Quantitative assessment of non-conservative radiation forces in an optical trap — https://doi.org/10.1209/0295-5075/86/38002
- Roichman i in. (2008), Influence of nonconservative optical forces on the dynamics of optically trapped colloidal spheres: the fountain of probability — https://doi.org/10.1103/PhysRevLett.101.128301
- Sun i in. (2009), Brownian vortexes — https://doi.org/10.1103/physreve.80.010401
- Kingma, Ba (2015), Adam: A Method for Stochastic Optimization — https://arxiv.org/abs/1412.6980
- Loshchilov, Hutter (2019), Decoupled Weight Decay Regularization — https://arxiv.org/abs/1711.05101
- Wang i in. (2022), Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis — https://doi.org/10.48550/arXiv.2203.05962
- Yaida (2018), Fluctuation-dissipation relations for stochastic gradient descent — https://arxiv.org/abs/1810.00004
- Shang i in. (2016), Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units — https://arxiv.org/abs/1603.05201
- Elhage i in. (2022), Toy Models of Superposition — https://doi.org/10.48550/arXiv.2209.10652
- Falk i in. (2025), A Model Zoo of Vision Transformers — https://doi.org/10.48550/arXiv.2504.10231
- Gieseler i in. (2021), Optical tweezers — from calibration to applications: a tutorial — https://doi.org/10.1364/aop.394888
- Pérez García i in. (2018), High-performance reconstruction of microscopic force fields from Brownian trajectories (FORMA) — https://doi.org/10.1038/s41467-018-07437-x

Mosty między tymi pracami a układem, z oceną, co jest dokładne, a co analogią: `app/MOSTY_ML_FIZYKA_PULAPKI.md`.
