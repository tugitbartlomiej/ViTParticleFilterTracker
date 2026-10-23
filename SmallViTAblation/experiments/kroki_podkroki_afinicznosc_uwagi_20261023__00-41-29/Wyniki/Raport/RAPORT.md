# Kroki × podkroki: afiniczność uwagi, pole MLP i złożenie siodeł w małym ViT (bieg s47, klatka TRAIN 0)

Eksperyment `kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29`, raport z 2026-10-23. Pomiar na zapisanych przejściach przez sieć
(806 checkpointów × 12 podkroków, 192 wymiary) i na jakobianach autograd (7 cząsteczek × 7 checkpointów × 12 podkroków); bez treningu.
Wykresy: `../Wykresy/F1…F9` (PNG i SVG), dane: `../Dane/`, kod: `../../Kod_eksperymentu/` i `../Code_Interpretujacy_Wyniki/`.

## 1. Streszczenie

1. **Uwaga na starcie jest dokładnie translacją chmury** (wagi jednostajne: 570–590 efektywnych źródeł z 600, udział wspólnego
   przesunięcia 1,000, 95–98 % jakobianu w części barycentrycznej). W ułamku pierwszej epoki zamienia się w **gwiazdę**: każdy token
   słucha głównie 3 tokenów narzędzia (wzbogacenie 40× w bloku 4, 37× w bloku 5; narzędzie→narzędzie 60–75×), 0 % uwagi na sobie
   i na 8 sąsiadach siatki, średnia odległość 10–12 łat. Mechanizm przestaje być afiniczny (50–80 % jakobianu przez softmax), ale
   przesunięcie wygląda afinicznie: CV-R² względem własnego położenia 0,79–0,91 dla uwagi wobec 0,65–0,89 dla MLP (F2, F3, F8D).
2. **Role bloków powstają w uczeniu**: MLP 1 staje się rozpychaczem (×1,05 → ×2,0–2,3 rozrzutu, log|det| +5…+10), MLP 4 rośnie do końca
   uczenia (×1,12 → ×1,32, szczyt 1,40 w kroku 60 000), bloki 4–6 ściskają objętość (log|det| −4…−6). W 1. epoce wszystkie MLP pchają
   promieniowo na zewnątrz (cos 0,36–0,52), w 60. tylko bloki 1–3; bloki 4–6 sterują stycznie lub do środka (F1).
3. **Przejście przez głębokość: od ruchu prostego do zygzaka.** Wskaźnik balistyczny 12 podkroków 0,98 (start) → 2,88 (1 epoka)
   → 0,97 (60 epok); cos kolejnych MLP 0,55–0,63 → 0,05–0,28. Późno pojawia się korekta sąsiednich podkroków: cos(MLP b, uwaga b+1)
   −0,05…−0,14, cos(uwaga b, MLP b) −0,11…−0,16 w blokach 5–6 (F5).
4. **Front rozdziału narzędzie–tło przesuwa się w głąb**: w 1. epoce skok Fishera w bloku 3 (1,54 → 2,16), w 60. w bloku 4 (1,34 → 3,20),
   szczyt 3,49 w bloku 5, blok 6 obniża do 2,92; Fisher > 3 po MLP 4 dopiero od kroku 44 300 (F4). Uwaga nie wygładza cech po siatce,
   MLP tak (F6).
5. **Siodło wszędzie, złożenie siodeł nie łapie.** Każdy z 12 podkroków, w każdym checkpoincie i dla każdej cząsteczki, kurczy 50–53 %
   kierunków (siodło z wirem, ‖A‖/‖S‖ ≈ 1,0 uwaga, 0,8–0,95 MLP); iloczyn 12 jakobianów ma 47–54 % kierunków o σ < 1, a podprzestrzenie
   kurczące kolejnych bloków nakładają się jak losowe (0,52–0,55 wobec 0,51–0,53). Uczenie buduje mapę ściskającą: log σ_max iloczynu
   0,85 → 2,1–2,6, log σ_min −1,0 → −5,3…−5,6, log|det| całego przejścia −1,1 → −18,5 (narzędzie) i −6,2 (tło) (F7, F8).

## 2. Pytania

1. Co widać **tylko** na płaszczyźnie krok uczenia × podkrok przejścia?
2. Czy uwaga jest przekształceniem afinicznym chmury cząsteczek i czy to widać na cząsteczkach?
3. Jakie analogie fizyczne (pułapki optyczne, pola sił) pasują, a jakie nie?

Kontekst: zasady Z1–Z24 (`outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md`), pomiar
`KURCZENIE_UWAGA_MLP.json` (40 checkpointów × 4 zdjęcia) i eksperymenty z 2026-10-21/22 (digest w `Opis/`). Pomiar tutaj rozszerza
KURCZENIE na wszystkie 806 checkpointów i dokłada obserwable, których wcześniej nie było (afiniczność, spójność po głębokości,
front rozdziału, gładkość po siatce, jakobiany wszystkich podkroków).

## 3. Dane i metody

- Bieg `assets/MalyViT_35000_od_zera_s47` (ziarno 47), 806 checkpointów 0–65 640 kroków (60 epok po 1094 kroki; co 10 kroków do
  kroku 1000, potem co ok. 80). Klatka TRAIN 0 (TR_0050_S1_P03, klatka 1629), 600 tokenów: 3 narzędzie (udział pikseli obrysu ≥ 0,5),
  13 mieszanych, 584 tło. SHA256 wag i plików w `Kod_eksperymentu/pochodzenie.json`.
- Stany 13 podkroków z `cache_uczenia/lasery/7992f2317c64a2e3/przejscia/k*.npz`: `xn · rms` odtwarza dokładne wycentrowane stany
  192D, więc Δ_s = X_s − X_{s−1} to dokładne przesunięcie podkroku bez wyrazów wolnych (Z20); kontrola rozkładu ≤ 1,5·10⁻⁴.
- Obserwable (definicje w docstringu `pomiar_kroki_podkroki.py`): r_po/r_przed; |Δ|/r; cos(Δ, x); CV-R² afinicznego dopasowania Δ ~ M·x + c
  (2 fałdy po 300 tokenów, zero z przetasowania); R² w rzucie 3D; udział translacji (rzut 3D surowy); wskaźnik balistyczny
  |ΣΔ|²/Σ|Δ|²; cosinusy między podkrokami; uwaga: 1/ΣA², entropia, masa na sobie i 8 sąsiadach, odległość ważona uwagą, wzbogacenie
  na tokenach narzędzia; lasery: odsetek z > 0,05, (Σ|z|)²/Σz², udział top-10; Fisher narzędzie/tło i czystość kNN w 13 stanach;
  energia Dirichleta na siatce 4-sąsiedztwa / (2·wariancja chmury).
- Jakobiany (`jakobiany_podkrokow.py`): dla tokenów 223, 260, 224 (narzędzie), 226 (mieszana), 400, 434, 494 (tło) w checkpointach
  0, 1094, 5470, 10 940, 21 880, 43 760, 65 640: mapa własna J_s = ∂x_i^(s)/∂x_i^(s−1) (192 × 192, inne cząsteczki ustalone), dla uwagi
  także pełny jakobian ∂x_i^(s)/∂X^(s−1) (192 × 600 × 192) i jego wersja z zamrożonymi wagami A. Autograd float64; odtworzenie
  zapisanych stanów ≤ 4,4·10⁻⁷. Z map własnych: wartości osobliwe, część symetryczna, log|det|, iloczyny po głębokości, nakładanie
  podprzestrzeni kurczących kolejnych bloków (tr(P_b P_{b+1}) / min(k_b, k_{b+1}) wobec losowego max(k)/192).
- Literatura: trzy wyszukiwania Undermind (lista w `../Dane/literatura_undermind.md`), cztery pełne teksty czytane celowo.

## 4. Wyniki

### 4.1 Uwaga: od translacji do gwiazdy (F2B, F3, F8D)

| | krok 0 | 1 ep | 5 ep | 10 ep | 20 ep | 40 ep | 60 ep |
|---|---|---|---|---|---|---|---|
| efektywne źródła, blok 1 / 4 / 5 / 6 | 571 / 587 / 580 / 589 | 66 / 20 / 10 / 100 | 32 / 22 / 15 / 74 | 48 / 57 / 23 / 17 | 56 / 14 / 13 / 11 | 53 / 19 / 14 / 24 | 66 / 13 / 17 / 40 |
| entropia uwagi, blok 4 [nat] (jednostajna 6,40) | 6,39 | 3,41 | 3,88 | 4,04 | 3,25 | 3,35 | 3,07 |
| wzbogacenie tło → narzędzie, blok 4 / 5 | 1,0 / 1,1 | 43 / 30 | 47 / 25 | 41 / 37 | 38 / 33 | 41 / 33 | 44 / 37 |
| wzbogacenie narzędzie → narzędzie, blok 4 / 5 | 1,1 / 1,1 | 45 / 15 | 72 / 33 | 74 / 74 | 61 / 76 | 60 / 66 | 66 / 70 |
| udział translacji w ruchu podkroku uwagi, bloki 1–6 | 1,000 | 0,51–0,95 | 0,26–0,82 | 0,29–0,88 | 0,17–0,82 | 0,18–0,91 | 0,21–0,90 |

- Kolaps uwagi jest najszybszym zdarzeniem uczenia: blok 5 ma < 300 źródeł od kroku 120 i < 50 od kroku 310, blok 4 od 230 / 450,
  bloki 1–3 i 6 między krokiem 1000 a 11 300. Blok 6 schodzi do 3 źródeł w kroku 19 900 i potem rozszerza się do 40; blok 1 z 32 do
  66: najpierw rzadka, potem gęsta.
- Uwaga **nigdy nie jest lokalna**: masa na sobie 0,00, na 8 sąsiadach 0,00–0,01 (jednostajna dałaby 0,013), średnia odległość 10–12 łat
  (jednostajna 12,9). Zamiast tego wszyscy słuchają narzędzia: w blokach 4–5 cząsteczka tła kieruje ok. 20 % uwagi na 3 tokeny
  narzędzia (0,5 % chmury), cząsteczka narzędzia 30–37 %.
- Pełny jakobian uwagi (7 tokenów): efektywna liczba cząsteczek wpływających spada z 540–580 do 2–23 po 1 epoce. Udział części
  barycentrycznej (zamrożone A) wobec innych cząsteczek: 0,95–0,98 na starcie, 0,2–0,7 po uczeniu (F8D). Stosunek wpływu własnego
  położenia do wpływu wszystkich innych: narzędzie 0,1–1,4; tło w blokach 4–5 do 3,5–5,3, czyli dla tła uwaga działa jak pole zależne od
  własnego zapytania, a nie jak oddziaływanie par.

### 4.2 Afiniczność jako wygląd (F2A)

CV-R² afinicznego dopasowania przesunięcia do własnego położenia (192D; zero −2…−5):

| podkrok | krok 0 | 1 ep | 10 ep | 60 ep |
|---|---|---|---|---|
| uwaga 1 / 4 / 6 | 0,99 / 1,00 / 1,00 | 0,82 / 0,91 / 0,96 | 0,83 / 0,89 / 0,92 | 0,90 / 0,82 / 0,88 |
| MLP 1 / 4 / 6 | 0,82 / 1,00 / 1,00 | 0,78 / 0,92 / 0,96 | 0,82 / 0,93 / 0,96 | 0,65 / 0,89 / 0,88 |

Na starcie wszystko jest afiniczne (uwaga: translacja; MLP: GELU prawie liniowe przy małych pobudzeniach). W uczeniu to **MLP** staje się
nieliniowy (MLP 1, 2, 3, 5: 0,65–0,71 w 60. epoce), a uwaga zostaje afiniczna w wyglądzie (0,79–0,91), choć jej mechanizm (kto
kogo słucha) jest nieliniowy. Rzut 3D aplikacji tej różnicy nie oddaje (R² w 3D 0,2–0,9, bez wzoru).

### 4.3 MLP: rozpychacz i kompresory (F1)

| r_po/r_przed | krok 0 | 1 ep | 5 ep | 10 ep | 20 ep | 40 ep | 60 ep |
|---|---|---|---|---|---|---|---|
| MLP 1 | 1,05 | 1,31 | 1,69 | 2,00 | 2,12 | 2,12 | 2,04 |
| MLP 2 / 3 | 1,03 / 1,03 | 1,23 / 1,21 | 1,25 / 1,29 | 1,21 / 1,33 | 1,20 / 1,25 | 1,17 / 1,17 | 1,14 / 1,19 |
| MLP 4 | 1,02 | 1,12 | 1,12 | 1,12 | 1,19 | 1,27 | 1,32 |
| MLP 5 / 6 | 1,02 / 1,02 | 1,10 / 1,07 | 1,06 / 1,06 | 1,05 / 1,04 | 1,04 / 1,02 | 1,05 / 1,02 | 1,04 / 1,01 |
| uwaga 1–6 | 1,00 | 1,00–1,04 | 0,99–1,05 | 0,99–1,06 | 0,99–1,04 | 1,00–1,05 | 1,00–1,04 |

Składowa promieniowa (mediana cos(Δ, x)): w 1. epoce MLP 1–6: 0,36 / 0,41 / 0,46 / 0,49 / 0,44 / 0,52 (wszystko na zewnątrz);
w 60. epoce 0,36 / 0,19 / 0,26 / −0,02 / 0,04 / −0,05, uwaga 4–6: −0,07 / −0,13 / −0,09. MLP 1 jest też największym rozpychaczem
objętości (log|det| +5,1…+8,0 narzędzie, +5,8…+10,6 tło), a MLP 4, 5, 6 kompresorami (−4,3 / −5,5 / −5,7 narzędzie w 60. epoce).

### 4.4 Spójność po głębokości i korekta sąsiednich podkroków (F5)

| | krok 0 | 1 ep | 5 ep | 10 ep | 20 ep | 40 ep | 60 ep |
|---|---|---|---|---|---|---|---|
| wskaźnik balistyczny, 12 podkroków / same MLP | 0,98 / 0,97 | 2,88 / 2,97 | 1,62 / 1,94 | 1,42 / 1,63 | 1,14 / 1,32 | 1,02 / 1,20 | 0,97 / 1,12 |
| cos(MLP b, MLP b+1), zakres po b | −0,05…0,04 | 0,55–0,63 | 0,27–0,49 | 0,12–0,40 | 0,07–0,37 | 0,03–0,33 | 0,05–0,28 |
| cos(MLP b, uwaga b+1), b = 3 / 4 / 5 | 0,01 / 0,03 / 0,03 | −0,13 / 0,11 / 0,17 | −0,17 / −0,05 / 0,02 | −0,06 / −0,10 / −0,13 | −0,02 / −0,04 / −0,10 | −0,05 / −0,12 / −0,07 | −0,05 / −0,14 / −0,14 |
| cos(uwaga b, MLP b), b = 4 / 5 / 6 | −0,02 / 0,02 / −0,01 | −0,03 / 0,20 / 0,25 | 0,08 / −0,08 / −0,08 | 0,09 / 0,05 / −0,14 | 0,15 / −0,04 / −0,01 | 0,19 / −0,13 / −0,16 | 0,19 / −0,11 / −0,12 |

Spearman wskaźnika balistycznego z krokiem: −0,89. Podkroki uwagi są przez całe uczenie wzajemnie niezależne (wskaźnik 0,94–1,12).

### 4.5 Front rozdziału narzędzie–tło (F4)

| Fisher w stanie | wejście | MLP 1 | MLP 3 | uwaga 4 | MLP 4 | MLP 5 | MLP 6 |
|---|---|---|---|---|---|---|---|
| krok 0 | 0,62 | 0,63 | 0,62 | 0,62 | 0,59 | 0,57 | 0,57 |
| 1 ep | 0,86 | 0,91 | 2,16 | 2,24 | 2,36 | 2,79 | 2,49 |
| 10 ep | 0,66 | 1,20 | 1,92 | 1,88 | 2,59 | 2,42 | 2,22 |
| 60 ep | 0,83 | 1,07 | 1,33 | 1,34 | 3,20 | 3,49 | 2,92 |

Fisher > 1 po MLP 6 od kroku 80, po MLP 4 od kroku 110; > 3 po MLP 4 od kroku 44 300 (maksimum 3,37 na wyjściu w kroku 48 000).
Korelacja w czasie Fishera na wyjściu z liczbą źródeł uwagi bloku 4: −0,62, z pewnością detektora +0,77. Czystość kNN z 3 tokenami
narzędzia jest nieużyteczna (0,067 w prawie wszystkich stanach) i nie wchodzi do wniosków.

### 4.6 Gładkość po siatce i lasery (F6)

Energia Dirichleta / (2·wariancja): wejście 0,06 (krok 0) → 0,25 (1 ep) → 0,31 (60 ep); po przejściu 0,066 / 0,100 / 0,116.
W 60. epoce podkroki uwagi zmieniają ją o ≤ 0,013, MLP 1 o −0,127, MLP 2 o −0,014, MLP 4 o −0,026. Uczenie wyostrza wejście,
MLP (nie uwaga) wygładza po siatce, bo rozpycha chmurę wzdłuż kierunków wspólnych dla sąsiadów.
Lasery: odsetek z > 0,05 spada z 0,41–0,45 do 0,06 (blok 2) – 0,19 (blok 4); efektywna liczba laserów 340–360 → 123–148; udział 10
najjaśniejszych 0,07–0,08 → 0,18–0,21.

### 4.7 Jakobiany: siodło wszędzie, złożenie ściska (F7, F8A–C)

| | krok 0 | 1 ep | 5 ep | 10 ep | 20 ep | 40 ep | 60 ep |
|---|---|---|---|---|---|---|---|
| udział kurczących kierunków map własnych (zakres po 12 podkrokach) | 0,50–0,51 | 0,50–0,52 | 0,50–0,52 | 0,49–0,52 | 0,49–0,52 | 0,49–0,53 | 0,50–0,52 |
| udział σ < 1 iloczynu 12 podkroków, narzędzie / tło | 0,49 / 0,49 | 0,54 / 0,53 | 0,53 / 0,53 | 0,53 / 0,52 | 0,53 / 0,52 | 0,53 / 0,51 | 0,53 / 0,51 |
| log σ_max iloczynu, narzędzie / tło | 0,85 / 0,85 | 2,27 / 1,94 | 2,38 / 2,28 | 2,19 / 2,23 | 2,27 / 2,82 | 2,27 / 2,58 | 2,14 / 2,59 |
| log σ_min iloczynu, narzędzie / tło | −1,01 / −1,00 | −1,35 / −1,52 | −2,38 / −2,37 | −3,46 / −2,69 | −3,62 / −3,93 | −5,80 / −4,99 | −5,27 / −5,61 |
| log\|det\| całego przejścia, narzędzie / tło | −1,1 / −1,0 | +0,9 / +2,5 | +1,8 / +4,2 | −1,7 / +5,7 | −5,0 / +1,3 | −11,7 / −3,5 | −18,5 / −6,2 |
| nakładanie podprzestrzeni kurczących bloków b, b+1 (zakres) / losowe | 0,49–0,51 / 0,50–0,51 | 0,53–0,54 / 0,51–0,52 | 0,52–0,55 / 0,51–0,53 | 0,51–0,55 / 0,50–0,53 | 0,51–0,54 / 0,50–0,53 | 0,52–0,54 / 0,51–0,52 | 0,51–0,54 / 0,51–0,52 |

- Część antysymetryczna J − I: uwaga ‖A‖/‖S‖ 0,96–1,00 we wszystkich checkpointach (czysta nienormalność), MLP 1 spada do 0,79–0,84,
  MLP 6 do 0,84–0,86, pozostałe MLP 0,86–0,99. Odpowiedź na otwarty test 11 z ZASADY: siodło z wirem jest cechą każdego bloku i uwagi.
- Wykładniki po głębokości (log σ iloczynu / 12 podkroków): +0,07 / −0,08 na starcie → +0,18 / −0,44 w 60. epoce (narzędzie).
  Objętość całego przejścia maleje e⁻¹⁸ (narzędzie, −0,096 na kierunek), ale w kilku kierunkach rozciąga się ×8–13: hiperboliczna mapa
  ściskająca, a nie pułapka. Dla narzędzia kontrakcja objętości jest silniejsza niż dla tła (−18,5 wobec −6,2).
- Mapa własna uwagi jest słabsza (‖J − I‖_F 0,7–2,4 dla narzędzia; dla tła do 4,9 w bloku 5) niż MLP 1 (4,0–9,5); uwaga działa
  głównie przez innych (4.1).

### 4.8 Kolejność zdarzeń (F9)

1. Kroki 100–450: kolaps uwagi w blokach 5 i 4 (najszybsze zdarzenie), Fisher na wyjściu > 1 (krok 80).
2. Kroki 300–1100: wspólny kierunek wszystkich MLP (wskaźnik balistyczny do 2,9), inflacja promieniowa wszystkich bloków.
3. Epoki 1–10: MLP 1 dochodzi do ×2, pewność detektora do 0,9; wskaźnik balistyczny spada.
4. Epoki 10–60: blok 4 rośnie (rozrzut, lasery, uwaga na narzędziu), front rozdziału przesuwa się do bloku 4 i rośnie do Fishera 3,2–3,5,
   korekta sąsiednich podkroków staje się ujemna; blok 6 lekko obniża rozdział.

## 5. Interpretacja i analogie fizyczne

- **Afiniczność uwagi.** Przy ustalonych wagach A przesunięcie Δx_i = Σ_j A_ij W v_j + b jest liniowe w chmurze (kombinacja wypukła
  wektorów wartości) i to jest sens, w jakim „uwaga jest afiniczna” [Ges23, Cas25, Rig25]. Na starcie, gdy A ≈ 1/600, redukuje się do
  translacji (udział 1,000) i w aplikacji widać to jako wspólny ruch całej chmury w układzie surowym oraz bezruch w układach
  wycentrowanych. Po uczeniu afiniczny jest tylko *wygląd* (CV-R² 0,8–0,9), bo uwaga stała się gwiazdą wokół narzędzia; mechanizm
  jest nieliniowy w 50–80 %. To zgadza się z [Adl26]: wcześnie w uczeniu model jest nieodróżnialny od swojego pola średniego uwagi,
  a potem odchodzi od niego; i z przewidywaniem [Tia23] (uwaga najpierw rzadka, potem gęsta), które widać w blokach 1 i 6.
- **MLP jako pole, uwaga jako sprzężenie.** Rozkład na podkroki to rozszczepienie Lie–Trottera [Lu19e]: lokalny krok pola (MLP) i
  nielokalny krok oddziaływania (uwaga). Uwaga w tym ViT nie jest filtrem dolnoprzepustowym po siatce (0 % na sąsiadach, Dirichlet
  bez zmian), więc narracja „uwaga wygładza, FFN wyostrza” [Wan22as, Dov24] tu się nie stosuje; wygładza MLP, a uwaga przenosi
  kontekst z narzędzia do wszystkich. Składowa styczna/radialna FFN z [Mud26] ma tu wyraźny profil w czasie uczenia: najpierw
  wszystkie MLP są radialne, potem bloki 4–6 stają się styczne.
- **Korekta między podkrokami.** Ujemne cosinusy między MLP b a uwagą b+1 oraz uwagą b a MLP b (bloki 5–6) odpowiadają mechanizmowi
  korekty sąsiednich warstw z [Pat25d] (cos −0,15…−0,3 w LLM, powstaje w uczeniu); tu są słabsze (−0,05…−0,16) i także powstają
  dopiero po kilku epokach.
- **Pułapki.** Każdy podkrok to siodło z wirem (Z15 potwierdzone dla wszystkich 12). Siodło można ustabilizować dynamicznie, gdy obraca
  się szybciej niż jego krzywizna skupiająca (siodło obrotowe, pułapka Paula, optyczne siodło [Tan24n]: bez tłumienia |Ω| > ω_y), albo
  gdy sekwencja soczewek skupiających i rozpraszających jest uporządkowana (ogniskowanie naprzemienne [Cou58]). Tu kolejne siodła są
  obrócone losowo (nakładanie podprzestrzeni = losowe), a każde jest silne (σ_max MLP 1 ≈ e^1,5), więc złożenie zostaje siodłem;
  sieć przypomina linię sześciu przypadkowo ustawionych kwadrupoli, czyli układ rozbieżny w części kierunków i zbieżny w innych.
  To wyklucza „pułapkę” jako punkt stały przejścia (zgodnie z brakiem zer w 192D w eksperymencie odległych równowag) i tłumaczy, czemu
  tokeny są przenoszone prądem, a nie trzymane. W pęsecie optycznej wir to 1–4 % sztywności (Pesce 2009), tu 80–100 %; pola
  niekonserwatywne trzymają tylko z tłumieniem [Ng09], którego w czasie głębokości nie ma.
- **Mapa ściskająca zamiast pułapki.** Uczony wzrost rozciągnięcia i ściśnięcia iloczynu (wąskie gardło efektywnego rzędu w [Fer26d])
  to właściwa „fizyka” przejścia: cząstka wchodzi w kanał, który ją zgniata w ~100 kierunkach i rozciąga w kilku. Z6 (ujemny ślad)
  i Z15 (siodło) są dwoma widokami tej samej mapy.

## 6. Odniesienie do wcześniejszych ustaleń projektu

- KURCZENIE_UWAGA_MLP (40 checkpointów): uwaga ≈ 1, MLP 1 ×2,1, bloki 4–5 lekko kurczą — potwierdzone na 806 checkpointach, z czasem
  (MLP 4 rośnie do końca; MLP 6 w tym biegu nie kurczy).
- Digest eksperymentów 2026-10-21/22 (jeden seed 3047): wspólne wyjście uwagi 3 niesie 97 % normy i jego usunięcie kosztuje −0,18 IoU,
  a zostawienie samego wspólnego wyjścia −0,01. Tutaj (s47) translacja jest 0,2–0,9 ruchu podkroku uwagi po uczeniu, czyli uwaga
  pozostaje „w większości translacją”, ale translacją zależną od obrazu (gwiazda do narzędzia).
- Otwarty test 11 (ZASADY): zamknięty, siodło z wirem jest w każdym bloku. Hipoteza „silnego ogniskowania” (złożenie siodeł kurczy):
  odrzucona dla s47.

## 7. Literatura (klucze Undermind)

Ges23 Geshkovski i in. 2023, 10.52202/075280-2493 · Cas25 Castin i in. 2025, 10.48550/arXiv.2501.18322 · Rig25 Rigollet 2025,
10.48550/arXiv.2512.01868 · Iso26 Isobe i in. 2026, 10.48550/arXiv.2605.07772 · Tia23 Tian i in. 2023, 10.48550/arXiv.2310.00535 ·
Adl26 Adler i in. 2026, arXiv:2609.16382 · Mud26 Mudarisov i in. 2026, arXiv:2608.02071 · Pat25d Patrawala i in. 2026,
10.52202/085713-4643 · Lu19e Lu i in. 2019, arXiv:1906.02762 · Fer26d Fernando, Guitchounts 2026, 10.48550/arXiv.2605.14258 ·
Tan24n Tandeitnik i in. 2024, 10.1103/physrevapplied.22.044073 · Cou58 Courant, Snyder 1958, 10.1016/0003-4916(58)90012-5 ·
Kir15b Kirillov, Levi 2015, 10.1119/1.4933206 · Ng09 Ng, Lin, Chan 2009, 10.1103/PhysRevLett.104.103601 · Wan22as Wang i in. 2022,
10.48550/arXiv.2203.05962 · Dov24 Dovonon i in. 2024, 10.48550/arXiv.2401.04301. Pełna lista: `../Dane/literatura_undermind.md`.

## 8. Ograniczenia

- Jeden bieg (s47) i jedna klatka; 3 tokeny narzędzia (cienki cystotom), więc Fisher jest miarą orientacyjną, a kNN nieużyteczne.
- Jakobiany własne uwagi trzymają resztę chmury w miejscu; pełny jakobian uwagi tylko dla 7 tokenów i 7 checkpointów.
- CV-R² afiniczne: 193 parametry na współrzędną przy 300 tokenach uczących; zero z przetasowania (−2…−5) pokazuje, że wartości
  0,65–0,91 są daleko od przypadku, ale bezwzględne poziomy zależą od liczby tokenów.
- Udział translacji liczony w rzucie 3D aplikacji (surowe stany 192D nie są zapisane); w 192D translacja jest usunięta z definicji.
- Wnioski o „fizyce” są analogiami; żaden z pomiarów nie dowodzi własności innych biegów ani innych architektur.

## 9. Pliki

- `Kod_eksperymentu/pomiar_kroki_podkroki.py`, `jakobiany_podkrokow.py`, `pochodzenie.json`; `Opis/OPIS_EKSPERYMENTU.md`.
- `Wyniki/Dane/kroki_podkroki.npz`, `kroki_podkroki_tabele.json`, `jakobiany_podkrokow.npz`, `jakobiany_podkrokow.json`, `tabele_odczyt.txt`,
  `literatura_undermind.md`, logi.
- `Wyniki/Code_Interpretujacy_Wyniki/odczyt_tabel.py`, `wykresy.py`; `Wyniki/Wykresy/F1…F9.png/svg`; `Wyniki/Krótki_Wniosek/WNIOSEK.txt`.
