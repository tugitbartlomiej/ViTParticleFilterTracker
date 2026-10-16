# Układ: koloidy, lasery i pułapki w małym Vision Transformerze

Stan: 15 października 2026. Ten plik zastępuje opis z 11 października (poprzednia wersja jest w historii git). Opisuje
MałyViT uczony od zera do detekcji narzędzia (bieg s47, 806 checkpointów) jako układ fizyczny: czym jest każdy element,
co zmierzono, co jest dokładną matematyką sieci, a co tylko obrazem. Rozstrzygające źródła:

- zasady i słownik: `app/ZASADY_DZIALANIA_UKLADU.md` (zasady Z1–Z29), wyprowadzenia:
  `outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.pdf`;
- pole wiązek w podkrokach: `app/pulapka_obliczenia.py` → `app/cache_uczenia/lasery/eafb5a973be3d490/pulapka_podkrokow_t226_G5.csv`;
- cel koloidu i pułapka uczenia: `experiments/pulapka_cel_koloidu_20261015/` → `outputs/pulapka_cel_koloidu_20261015/`;
- kod sieci: `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`.

Liczby z rozdziałów 4–6 dotyczą jednego zdjęcia i jednego tokenu (226, środek ramki narzędzia), chyba że podano inaczej.

## 1. Układ w jednym akapicie

Detektor oglądany jest jak układ cząstek w świetle. Jedno zdjęcie daje **600 tokenów**: to **koloidy**. W każdym z 6 bloków
sieci **768 neuronów MLP** to **reflektory**: każdy świeci na koloid wiązką o jasności zależnej od położenia koloidu i pcha go
w swoim stałym kierunku. **Uwaga** to siły, którymi koloidy działają na siebie nawzajem. Układ ma **dwie osie czasu**:
głębokość sieci (jedno przejście: 12 podkroków, 13 stanów) i uczenie (806 checkpointów, w których przestawiają się
reflektory). **Celem** jest odpowiedź głowy: z tokenu ma powstać 5 liczb (czy tu jest środek narzędzia i odległości do
krawędzi ramki). Najważniejsze wyniki: pole wiązek w jednym przejściu **nie trzyma** koloidu (działa jak układ soczewek, który
go przepycha), a w czasie uczenia koloid jest **uwięziony** jak cząstka Browna w pęsecie wokół wolno dryfującego środka i dąży do
celu, który częściowo leży w nieskończoności, więc zbliża się coraz wolniej i nigdy go nie osiąga.

## 2. Elementy układu

| Obraz fizyczny | Obiekt w sieci | Wzór |
|---|---|---|
| **koloid** (cząstka) | token j: stan w strumieniu rezydualnym | `x_j ∈ R^192`, j = 0…599, numer `30·wiersz + kolumna` |
| **reflektor** (laser) | neuron c warstwy ukrytej MLP bloku; 768 w bloku, 6 × 768 w sieci | para (celownik `w_c`, wiązka `v_c`) |
| **celownik** | wiersz c macierzy fc1, oglądany przez LayerNorm | decyduje, kiedy reflektor się zapala |
| **wiązka** | kolumna c macierzy fc2 | kierunek i moc pchnięcia `v_c ∈ R^192` |
| jasność wiązki | aktywacja po GELU | `z_jc = GELU(w_c · LN(x_j) + b_c)` |
| pchnięcie w bloku MLP | wyjście MLP | `Δx_j = Σ_c z_jc v_c + b₂` |
| siły między koloidami | uwaga, 3 głowy | `f_jk = Σ_h A^h_jk W_O^h W_V^h LN(x_k)` |
| **pułapka optyczna** | pole wszystkich wiązek bloku wokół koloidu | `F(x)`, jego Jakobian `J`, ognisko `F = 0` |
| sfera odczytu | LayerNorm przed MLP i przed głową | sfera o promieniu √192 rozciągnięta przez γ |
| ruch reflektorów | uczenie: AdamW zmienia `w_c`, `v_c`, `b_c` | `θ_t` |
| **plamka** | stan tokenu oglądany w czasie uczenia | `z_j(t) = f(x; θ_t)` |
| **głowa** | LN → Linear 192 → GELU → Linear 5 na każdym tokenie | `o_j = (logit, ln l, ln t, ln r, ln b)` |
| **cel** | wartości, których żąda strata dla tokenu j | mapa środków `heat_j`, odległości `ltrb_j` |

**Zmiana względem opisu z 11 października.** Wtedy „cząsteczką” był odczyt 768D po GELU bloku 6. To są **jasności 768 wiązek**
padających na koloid (jego „kod kreskowy” oświetlenia), a nie jego położenie. Położeniem koloidu jest 192-wymiarowy stan
strumienia rezydualnego. Dane 768D z wcześniejszych pomiarów (rozdział 9 i `02_DANE_I_TRAJEKTORIE.md`) pozostają ważne, ale
opisują oświetlenie, nie miejsce.

## 3. Dwie osie czasu, których nie wolno mieszać

**Głębokość (jedno przejście).** Obraz → łodyga CNN i kodowanie pozycji → stan 0. Potem 6 bloków, w każdym najpierw uwaga,
potem MLP: razem **12 podkroków** i **13 stanów** koloidu. Ruch w każdym podkroku jest **dokładnie** sumą wkładów (600 sił uwagi
albo 768 wiązek), z błędem rzędu 10⁻⁶–10⁻⁵ (Z19, Z20). To jest ruch w świetle przy ustalonych reflektorach.

**Uczenie (checkpointy).** Każdy krok AdamW zmienia wspólne wagi θ, czyli przestawia wszystkie reflektory naraz i zmienia głowę.
Stały obraz i stały token dają wtedy nowy stan: `z_j(t) = f(x; θ_t)`. Ruch plamki to różnica dwóch przejść, lokalnie
`Δz_j ≈ J_j Δθ`. Koloid nie ma własnej prędkości ani masy; porusza się, bo zmienia się sieć, która go oblicza.

Trzecia, osobna oś to klatki filmu (inny obraz wejściowy). Krok DDIM generatora SD to czwarta. Żadnej z nich nie należy łączyć
z dwiema powyższymi.

## 4. Pułapka optyczna w jednym przejściu: pole wiązek

Pole liczone jest tak: koloid próbny stawiany jest w punktach małej kuli wokół tokenu (promień = połowa odległości do najbliższego
innego tokenu w rzucie 3D), reszta tokenów stoi; sieć liczy pchnięcie w danym podkroku. W środku kuli pole równa się dokładnie
prawdziwemu skokowi tokenu. Pomiar: token 226, 806 checkpointów × 12 podkroków (`pulapka_podkrokow_t226_G5.csv`).

- **Pole w komórce jest liniowe** (R² 0,9994), więc opisuje je macierz `J`: część symetryczna (sztywność w trzech osiach),
  ślad (dywergencja: zbieganie albo rozbieganie) i część antysymetryczna (wir).
- **Zero pola nie leży w komórce ani razu** (0 z 806). Bliżej niż jeden skok tylko w 0,6 % podkroków. W przekroju 3D ognisko
  istnieje w 7,7 % checkpointów (Z25).
- **Kształt:** siodło w 75 % podkroków, rozpychanie we wszystkich osiach w 24 %, studnia w 1,2 %. Spirala (zespolone wartości
  własne) w 40 %: wyrzucająca 22 %, ściągająca 18 %.
- **MLP rozprasza:** dywergencja dodatnia praktycznie zawsze; najmocniej MLP bloku 1 (+2,2: otoczenie koloidu podwaja się w
  jednym podkroku). Uwaga prawie nie zmienia objętości, za to silnie obraca (wir/siodło 0,3–0,95). W MLP bloków 4–5 pod koniec
  uczenia wir ≈ siodło.
- **Całe przejście** (iloczyn 12 podkroków) rozpycha otoczenie koloidu ~3× (objętość ~9×); w uczeniu słabnie 18× → 6,6×.
  Nie działa jak pułapka Paula z obracającego się siodła.

**Czym to jest fizycznie.** Pole wiązek to matematycznie warstwa pamięci klucz–wartość (celownik = klucz, wiązka = wartość;
Geva i in. 2021), a transformer to układ cząstek w głębokości: MLP = dryf każdej cząstki osobno, uwaga = oddziaływanie
(Lu i in. 2019; Geshkovski i in. 2023). Pułapka wymaga siły zbiegającej się do ogniska (optyczne twierdzenie Earnshawa,
Ashkin i Gordon 1983) i „rdzenia”, w którym część potencjalna przeważa nad wirową (Huang i in. 2022). Tu ani jednego, ani
drugiego nie ma, a wiry Browna wymagają stabilnego punktu (Moyses i in. 2015). Pole bloku to więc **układ soczewek, który
przepycha cząstkę**, a nie pęseta, która ją trzyma.

## 5. Pułapka uczenia: ruch w czasie uczenia

W czasie uczenia koloid **jest** uwięziony, ale przez samo uczenie, nie przez pole jednego przejścia
(`PULAPKA_UCZENIA.md`, `WYNIKI.md`):

- **Proces Ornsteina–Uhlenbecka wokół dryfującego środka.** Kolejne przyrosty są ujemnie skorelowane (−0,415; model OU
  przewiduje −0,42): koloid drga wokół środka i do niego wraca.
- **Czas powrotu** 8–12 kroków na początku, ~30–50 kroków pod koniec (odstęp checkpointów 100 kroków, więc to tylko
  oszacowanie). Sztywność (1 / wariancja drgań) rośnie ~5×, środek przesuwa się ~30× wolniej.
- **Uwięzienie:** w późnym uczeniu przemieszczenie prawie nie rośnie z czasem (MSD ~ τ^0,13).
- **Zbliżanie do położenia końcowego jest potęgowe**, nie wykładnicze: odległość ~ t^−0,43 (R² 0,99 wobec 0,80).
  Współczynnik uczenia był stały (5·10⁻⁴ po rozgrzewce), więc to nie efekt gaszenia uczenia.
- **Odpowiedź bez opóźnienia:** ruch idzie w stronę spadku błędu od razu (cos +0,11–0,18, dodatni w ~75 % kroków) i ta
  zgodność znika po ≥ 500 krokach. Koloid nie „goni” pułapki z opóźnieniem ani odwrotnie (Z27).

**Czym to jest fizycznie.** Szum pochodzi z losowych paczek danych. Stochastyczny spadek gradientu blisko minimum zachowuje się
jak proces Ornsteina–Uhlenbecka (Mandt i in. 2017), a uczenie sieci zwalnia i na końcu dyfunduje po dnie krajobrazu błędu
(Baity-Jesi i in. 2019). Sprężyną wag jest weight decay (Z24). Najbliższy obraz: koloid w powoli przesuwanej pęsecie (van Zon
i Cohen 2003; Imparato i in. 2007), z tą różnicą, że cel też się przesuwa.

**Zastrzeżenie.** Gradient był obcinany do normy 1 przez cały bieg. Część uwięzienia może z tego wynikać; test: bieg bez
obcinania (DeiT na Edenie).

## 6. Cel: dokąd dąży koloid

**Gdzie leży.** Cel jest zapisany w stracie, w przestrzeni odpowiedzi głowy (5 liczb), nie w przestrzeni tokenu. Dla tokenu 226:
logit „tu jest środek” → **+∞** (p = 1) i odległości ramki **4,38; 0,25; 4,42; 1,17** kratki (krok 16 px), czyli
`o = (+∞; 1,476; −1,386; 1,485; 0,154)` (`WYJSCIE_CELU.md`).

- **Ramka (4 liczby): zwykły punkt.** Błąd odległości 1,41 → 0,25 już w kroku 500, potem drgania 0,04–0,2 wokół celu.
- **„Tu jest środek”: punkt w nieskończoności.** Logit −4,4 → 0 (krok 500) → +2,3 (krok 10 000) → +2,9 (koniec, p = 0,95);
  rośnie jak ~0,70 · ln t, a siła błędu na logit maleje jak ~t^−1,76. Im bliżej celu, tym słabsze pchnięcie, więc zbliżanie
  zwalnia bez końca, jak regresja logistyczna na danych rozdzielnych (Soudry i in. 2017). To tłumaczy prawo potęgowe z rozdziału 5.

**Dlaczego celu nie ma jako punktu w przestrzeni tokenu** (`CEL_W_LATENCIE.md`):

1. **Jednej odpowiedzi odpowiada cała dolina.** Głowa zamienia 192 liczby na 5; jej Jakobian ma rząd 5, więc tę samą odpowiedź
   daje 187-wymiarowy zbiór położeń. Wybór jednego punktu wymaga dodatkowego kryterium (np. „najbliżej koloidu”).
2. **Samym ruchem tokenu celu nie da się osiągnąć.** LayerNorm sprowadza token do stałej długości, więc przy ustalonej głowie
   logit ma sufit: +1,86 (krok 1100), +5,31 (10 100), +6,27 (28 444), +6,89 (koniec). Sufit podnosi się tylko dlatego, że rosną
   wagi głowy (|W1| 8,0 → 12,4, |W2| 1,3 → 2,2).
3. **Głowa zmienia się w każdym kroku**, więc przekład celu na latent obowiązuje tylko w jednym checkpoincie.

Najbliższy punkt latentu z ramką równą celowi i logitem o 1 wyższym leży ~6 jednostek od koloidu (10–20 % jego długości),
w 94–98 % w 5 kierunkach czytanych przez głowę. Koloid porusza się i drga 2–3× silniej w tych kierunkach niż w pozostałych 187,
bo siła błędu działa tylko przez nie (`DOLINA.md`).

**Kto zmniejsza błąd.** Uczenie przesuwa jednocześnie koloid i cel (głowę). Na początku uczenia sam ruch koloidu obniżyłby błąd
tokenu 226 o 3,1, sam ruch głowy o 4,2, oba razem o 4,6; pod koniec wkłady są równe. Oba ruchy w dużej mierze się dublują
(rozkład reprezentacja–odczyt; Chou i in. 2026). Pchnięcie MLP bloku 6 obniża błąd tokenu w 63–90 % checkpointów, pod koniec
o ~13 %. Na końcu koloid stoi przy logicie 2,9, choć sam ruch tokenu pozwalałby dojść do 6,9: hamują go wspólne wagi (inne
zdjęcia paczki, sąsiednie tokeny, które nie powinny mówić „środek”), weight decay i obcinanie gradientu.

## 7. Sieć, dane i uczenie

**Sieć.** Wejście 3 × 320 × 480 w [0, 1], normowanie średnimi i odchyleniami ImageNet. Łodyga: 4 konwolucje 3 × 3 z krokiem 2
(3 → 24 → 48 → 96 → 192; pierwsze trzy z BatchNorm i GELU), krok 16 px, siatka 20 × 30 = 600 tokenów po 192 liczby, do których
raz dodaje się stałe kodowanie pozycji 2D sin-cos (nieuczone). 6 bloków timm (pre-LN): `x = x + attn(LN1 x)`,
`x = x + mlp(LN2 x)`, 3 głowy uwagi z biasem qkv, MLP 192 → 768 → 192 z GELU (erf), ε LayerNorm 10⁻⁶. Końcowa LayerNorm i głowa
192 → 192 → GELU → 5; bias mapy środków startuje od −4,6, bias ramek od ln 3. Brak tokenu CLS. 2 926 493 parametry.

**Cel uczenia** (`model.cele`, `model.strata`). Gaussy wokół środka ramki na siatce, komórka środka = 1, szerokość ~ rozmiar
ramki (min. 0,7 komórki). Odległości l, t, r, b w kratkach, tylko w komórkach wewnątrz ramki z Gaussem > 0,3 (plus komórka
środka), ograniczone od dołu przez 0,05. Strata `L = L_heat + 2 · L_box`: ogniskowa CenterNet (dodatnie `(1 − p)²`, ujemne
`p² (1 − heat)⁴`, normowana liczbą środków) i `1 − GIoU` ważone mapą. Głowa zwraca logarytmy odległości. Dekodowanie: sigmoid,
max-pooling 3 × 3, do 10 ramek. mAP@0,5 to tu jedna klasa i jeden próg IoU.

**Dane.** Cataract-LMM Object Tracking: manifest 35 000 klatek TRAIN ze 117 filmów (film, klatka, archiwum ZIP, ramki,
SHA-256), obrazy rozciągane do 480 × 320 (INTER_AREA, bez zachowania proporcji). Walidacja 3000 klatek. Token zależy od całego
obrazu (uwaga), ale numer tokenu jest stały i pozwala śledzić jego stan.

**Uczenie biegu s47.** Ziarno 47, paczka 32, epoka = 1094 kroki. Augmentacje: odbicie poziome (0,5), jasność/kontrast 0,8–1,2.
bf16 na CUDA. AdamW lr 5·10⁻⁴ (β 0,9/0,999), weight decay 0,05 dla macierzy, rozgrzewka liniowa 300 kroków, potem **stały lr**
(bez wygaszania). Gradient obcinany do normy 1. Checkpointy: 0–1000 co 10 kroków, potem co 100, plus końce epok; razem 806 do
kroku 65 640 (60 epok). Bieg wznowiono po 21 epokach (krok 22 974, nowa permutacja). Wyniki: mAP@0,5 ≈ 0,449 po 1. epoce,
≈ 0,65 w epokach 21–26, 0,608 w epoce 60; strata treningowa dalej malała.

## 8. Trzy historie uczenia, których nie wolno sklejać

| Historia | Dane treningu | Czas | Zachowany pomiar |
|---|---|---|---|
| K1000 (1 października) | 1500 klatek | 1000 aktualizacji | 1001 stanów 12 sond; checkpointy co 10 kroków |
| 35k `pelny` = **s47** | 35 000 klatek, 117 filmów | 60 epok, 65 640 aktualizacji | 806 checkpointów; pełne 768D ośmiu sond w 75 stanach |
| 35k `gesty` | ta sama pula i przepis | 21 epok, 22 974 aktualizacje | 100 sond w PCA32, 8 sond w pełnych 768D, telemetria paczek |

K1000 nie jest początkiem biegu 60 epok, a `gesty` nie jest jego gęstą wersją: ten sam przepis i ziarno nie dają bitowo
tego samego przebiegu na GPU. Rozdziały 4–6 dotyczą biegu s47 i zdjęcia z historii `eafb5a973be3d490` (aplikacja,
zakładka „Lasery i koloidy”).

## 9. Wcześniejsze wątki w skrócie

- **Ruch tokenów i „szkło” (do 11 października).** Tokeny narzędzia ruszają się mocniej od tła (AUC 0,761), ale bez stałego
  kierunku specyficznego dla narzędzia. Audyty 768D: zależność od skali (MSD lag 1–5: α = 1,53; lag 10–100: 0,26), brak
  monotonnego „zamarzania” populacji; odjęcie dryfu i prosty OU wyjaśniają część odcisków przypisywanych szkłu. Wniosek:
  skorelowana, przestrzennie uporządkowana przebudowa aktywacji ze wspólnych zmian wag; podobieństwa do szkła bez dowodu przejścia
  szklistego (`outputs/audyt_szkla_768d_20261010/RAPORT.md`, `outputs/poszlaki_dynamiki_20261011/RAPORT.md`).
- **Równania ruchu.** Oscylator z wolnym środkiem i Koopman PCA32 nie przewidują dobrze przyszłości bez znajomości paczki; przy
  znanej paczce suma odpowiedzi składników aktualizacji wyjaśnia 98,9–99,7 % ruchu (pęd 64–68 %, paczka 31–35 %, BN ~1 %)
  (`experiments/maly_vit_liniowy_ruch_czastek/WYNIKI.md`).
- **Temperatura i reflektory** (Z22–Z26). GELU działa jak próg Fermiego: „temperatura bramek” stygnie 2,6 → 0,7–0,8; κ =
  cos(celownik, wiązka) daje słabą siłę zwrotną (−0,18), wir pojedynczych reflektorów tłumaczy ~połowę spadku wiru; wiązki
  obracają się ~3,5°/100 kroków.
- **Interwencje w latencie i SD.** Optymalizacja aktywacji przy zamrożonych wagach (oracle 1283/1500 obrazów z confidence i IoU
  ≥ 0,90) nie przenosi się na inne obrazy; wspólna delta 600 × 768 poprawiła mAP50:95 0,245 → 0,299 na tych samych obrazach
  kosztem fałszywych detekcji. Generator SD warunkowany latentem 600 × 768 (adapter + cross-attention) nie odtwarzał narzędzia
  z latentów sztucznych. To osobna gałąź: wagi stoją, a koloid jest przesuwany ręcznie (`app/silnik.py`, raporty
  `vit_k1000_*`, `vit_sd_pelny_latent_filmy_eden_20261006`).

## 10. Co jest faktem, co obrazem, co błędem

| Element | Status |
|---|---|
| 768 wiązek pchających koloid, jasność zależna od położenia | **dokładne** (wzór bloku MLP) |
| uwaga jako siły między koloidami, łańcuch 6 bloków | **dokładne** wzdłuż głębokości |
| suma wkładów = ruch w podkroku | **dokładne** (błąd 10⁻⁶–10⁻⁵) |
| pole wiązek, jego sztywność, wir i dywergencja | **dokładne** jako obliczenie; zmierzone dla tokenu 226 |
| pułapka trzymająca koloid w jednym przejściu | **błędne**: zero pola nigdy w komórce, pole rozpycha |
| koloid w czasie uczenia jako proces OU wokół dryfującego środka | **pomiar**, zgodny z modelem (−0,415 wobec −0,42) |
| cel jako punkt w przestrzeni tokenu | **błędne**: 187-wymiarowy zbiór, a p = 1 nieosiągalne przy stałej głowie |
| cel w przestrzeni wyjścia głowy | **dokładne** (z kodu straty) |
| koloid nadążający za pułapką z opóźnieniem | **błędne**: wspólne zejście bez opóźnienia (Z27) |
| temperatura, reflektory, soczewki, pęseta | **analogie**, użyteczne tylko z liczbami i kontrolami |
| położenie w 3D na ekranie | **umowne** (rzut 192 → 3) |

**Nowość.** Wyszukiwania literatury (Undermind, Scite, 12–15 października) nie znalazły pracy, która łączy pole wiązek MLP,
token-koloid i jego cel w czasie uczenia. Bliskie są: tokeny jako cząstki w głębokości (Geshkovski i in.), szkło i OU na
wagach (Baity-Jesi i in., Mandt i in.), rozkład reprezentacja–odczyt (Chou i in. 2026). Największe ryzyko wyprzedzenia:
Halperin, arXiv 2606.29679 (przykłady jako cząstki przez checkpointy, bez tokenów i bez pola wiązek).

## 11. Zasady dalszego liczenia

1. Wybierać jeden bieg; nie sklejać K1000, `gesty` i s47.
2. Rozróżniać oś głębokości i oś uczenia; podawać, w której liczony jest ruch.
3. Położenie koloidu to 192D strumienia rezydualnego; 768D po GELU to jasności wiązek. Nie zastępować ich PCA3, PCA32, RP128
   ani macierzami Grama.
4. Liczyć różnice w float32/float64, nie w float16 (siły tokenów tła są bliskie zera i w float16 tracą znaczenie).
5. Podawać populację, okno, lag, układ odniesienia i jednostkę kroku; „jeden krok” to jedna aktualizacja, nie odstęp 100 kroków.
6. Cel opisywać w przestrzeni wyjścia; do latentu przenosić tylko jako zbiór albo z jawnym kryterium wyboru punktu.
7. Oddzielać dopasowanie przeszłości, rekonstrukcję przy znanej paczce i prognozę nieznanej przyszłości.
8. Rzuty 3D sprawdzać na surogatach (błądzenie losowe przez ten sam rzut tworzy pozorną strukturę; Antognini i Sohl-Dickstein 2018).
9. Wnioski z jednego tokenu i jednego biegu powtarzać na innych tokenach, zdjęciach i biegach (DeiT, ViT-S na Edenie).
10. Zachowywać ścieżki, wersję kodu i dokładny punkt odczytu.

## 12. Gdzie co jest

| Co | Gdzie |
|---|---|
| surowe dane i trajektorie 768D | `02_DANE_I_TRAJEKTORIE.md`, skróty w `linki/` |
| katalog eksperymentów i scalone raporty | `03_KATALOG_EKSPERYMENTOW.md`, `04_RAPORTY_SCALONE.md`, `zrodla/` |
| zasady Z1–Z29 i mosty do fizyki | `app/ZASADY_DZIALANIA_UKLADU.md`, `app/MOSTY_ML_FIZYKA_PULAPKI.md`, `app/OGNISKA_FIZYKA_I_SIECI.md` |
| aplikacja (okienko 11: pułapka, podkroki, pole w komórce) | `app/`, zakładka „Lasery i koloidy”; przycisk „Policz pułapkę w podkrokach” |
| pomiary pola wiązek | `app/pulapka_obliczenia.py`, wynik `pulapka_podkrokow_t226_G5.npz/.csv` |
| cel koloidu i pułapka uczenia | `experiments/pulapka_cel_koloidu_20261015/`, `outputs/pulapka_cel_koloidu_20261015/` |
| literatura | `PDF/pulapki_optyczne/`, `literatura_szklo_tokeny_20261011/`; kolekcja Scite „ViT: pułapka optyczna koloidu (pole wiązek MLP, cel w uczeniu)”; Undermind, workspace ViT |
| plan pomiarów jak w pęsecie (Eden) | `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md` |

## 13. Otwarte pytania

- Czy uwięzienie w czasie uczenia zostaje bez obcinania gradientu i przy innym weight decay?
- Czy te same prawa (OU, potęgowe zbliżanie, rozpraszające MLP) zachodzą dla wszystkich 600 tokenów, innych zdjęć i innych
  sieci (DeiT-Tiny na CIFAR-10, ViT-S na ImageNet)?
- Co dokładnie zatrzymuje logit tokenu środka na 2,9 przy suficie 6,9: inne zdjęcia paczki, sąsiednie tokeny czy weight decay?
- Rzeczywisty czas relaksacji w późnym uczeniu (potrzebne gęstsze checkpointy niż co 100 kroków).
- Czy tokeny narzędzia i tła mają różne pułapki i różne „kody kreskowe” zapalonych reflektorów, i czy różnica rośnie z mAP.
