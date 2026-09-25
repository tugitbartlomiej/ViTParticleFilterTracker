# Czy Koopman przewiduje elementy latentu identyfikujące narzędzie? Wyniki (2026-09-25)

Stan: część 1 (73 checkpointy, kroki 5..365) i część 2 (160 checkpointów, kroki 5..800, sekcja 5) zakończone.
Pokaz FastGAN na przewidzianym latencie (sekcja 6) wykonany, wynik negatywny po stronie generatora.

## 1. Co jest „elementem identyfikującym narzędzie” i jak się zachowuje w trakcie uczenia

- Kierunek LDA narzędzie/tło w 384-wymiarowych tokenach rozdziela tokeny narzędzia od tła na obrazach holdout
  z AUC **0,995 już w kroku 5** (pierwszy checkpoint po starcie z COCO) i 0,998 w krokach 245 i 365.
  Tożsamość narzędzia nie powstaje w tym dostrajaniu; jest dziedziczona z wstępnie wyuczonego backbone
  i tylko się doostrza.
- Sam kierunek obraca się: cosinus kierunku z kroku t do kierunku z kroku 5 spada do ok. 0,75 i tam zostaje
  (obrót o ok. 40 stopni w pierwszych 100 aktualizacjach, potem stabilny).
- Kierunek LDA jest prawie prostopadły do głównych kierunków wariancji różnicy „token narzędzia − tło”
  (cos 0,02–0,10 z podprzestrzeniami PCA k = 1..32). Największa zmienność tej różnicy nie jest tym, co
  odróżnia narzędzie; to zgodne z Gur-Arim: kierunek „zadaniowy” jest wąski, reszta to bulk.

## 2. Przewidywalność podczas uczenia (K = 73, dopasowanie do kroku 245, test 24 przejść do kroku 365)

Miara: RMSE prognozy jednego kroku (5 aktualizacji) albo 20 kroków (100 aktualizacji) na obrazach holdout,
podzielone przez RMSE „brak zmian” (1,0 = tak źle jak nic nie przewidywać). Mniej = lepiej.

### 2a. Poziom tokenów narzędzia (1458 jednostek, każda ze swoim modelem)

| podprzestrzeń | alpha (v̂ = α v) | ostatni przyrost | pełna macierz na prędkości | Koopman na położeniu |
|---|---|---|---|---|
| tożsamość S_1 | 0,507 | 0,521 | 1,46 | 1,49 |
| losowa 1-wym. | 0,384 | 0,400 | 1,26 | 1,30 |
| tożsamość S_8 | 0,516 | 0,532 | 1,16 | 1,20 |
| losowa 8-wym. | 0,512 | 0,529 | 1,16 | 1,15 |
| tożsamość S_32 | 0,486 | 0,501 | 0,89 | 1,04 |
| dopełnienie (352 wym.) | 0,478 | 0,493 | 0,85 | – |
| pełne 384 | 0,479 | 0,494 | – | – |

Wniosek: na poziomie pojedynczych tokenów podprzestrzeń tożsamości **nie jest** bardziej przewidywalna niż
losowa ani niż reszta. Każdy model z macierzą per token przegrywa z brakiem zmian (47 par na 384 albo k wymiarów).
Prawo 1/t (hiperboliczne, τ = 10) nie jest lepsze od stałego α: 0,490 wobec 0,479.

### 2b. Poziom obrazu: skalar tożsamości a_i(t) = w_lda · (opis narzędzia − tło) — 72 trajektorie

| model | 1 krok | 20 kroków (100 aktualizacji) |
|---|---|---|
| **wspólny Koopman na prędkości, 4 opóźnienia (Hankel), 48 obrazów fit** | 0,272 | **0,171** |
| wspólny, 3 opóźnienia | 0,281 | 0,182 |
| wspólny, 1 opóźnienie | 0,272 | 0,238 |
| wspólny z prawem 1/t | 0,277 | 0,238 |
| alpha per obraz | 0,276 | 0,285 |
| hiperboliczne per obraz | 0,281 | 0,445 |
| ostatni przyrost | 0,284 | 0,617 |
| relaksacja afiniczna a_{t+1} = ρ a_t + c (Koopman z punktem stałym) | 0,657 | – |
| brak zmian | 1,0 | 1,0 |

To jest najlepszy operator: wspólna dla wszystkich obrazów macierz na czterech ostatnich przyrostach skalara
tożsamości. Na 100 aktualizacji naprzód ma błąd 17 procent braku zmian, ostatni przyrost 62 procent,
tłumienie α 28 procent. Przechodzi bramkę (≥ 5 procent lepiej od ostatniego przyrostu, ≥ 20 procent od braku zmian).

### 2c. Poziom obrazu: opis narzędzia w podprzestrzeni S_k (k wymiarów)

| stan | najlepszy model | 1 krok | 20 kroków | ostatni przyrost 20 kroków |
|---|---|---|---|---|
| S_1 | alpha | 0,418 | 0,222 | 0,419 |
| S_4 | alpha / wspólny L4 | 0,395 | 0,273 / 0,290 | 0,527 |
| S_8 | alpha | 0,399 | 0,294 | 0,588 |
| S_16 | alpha | 0,403 | 0,350 | 0,696 |
| S_32 | alpha | 0,394 | 0,365 | 0,730 |
| 8 losowych kierunków poza S_32 | alpha | 0,356 | 0,360 | 0,731 |
| klasyczny Koopman na położeniu, wspólny, S_1..S_32 | – | 3,5–8,7 | 2,0–2,5 | – |

Wniosek: im mniej wymiarów, tym lepiej; kilka wymiarów tożsamości przewiduje się na 100 aktualizacji z błędem
22–29 procent braku zmian. Ale losowe kierunki dają to samo co S_8: to nie tożsamość jest specjalna, tylko
uśrednienie po masce (mniej szumu) i mały wymiar. Koopman na samym położeniu jest kilkakrotnie gorszy od
„nic nie rób” we wszystkich wariantach.

## 3. Dlaczego nie wychodzi tak dobrze, jak się wydaje, że powinno

1. **Liniowość jest w prędkości, nie w położeniu.** Latent podczas uczenia jedzie jak rozpędzony wózek, który
   hamuje: kolejne przyrosty są prawie równoległe (cos 0,90), a ich długość maleje trzykrotnie. Operator na samym
   położeniu z_t musi mieć punkt stały (koniec uczenia), a ten leży poza oknem dopasowania: trajektoria nigdy nie
   wraca w to samo miejsce, więc każdy nowy stan jest ekstrapolacją. Stąd 3–28 razy gorzej od braku zmian.
   Dopiero stan [z, v] (opóźnienie) daje liniową, przewidywalną dynamikę; to samo mówią prace Koopmanowskie
   o treningu: Redman i in. używają 8–64 opóźnień, Aswani i in. stwierdzają, że bez opóźnień wektory osobliwe są
   szumem, z opóźnieniami wychodzą czyste sinusoidy.
2. **Stan tokenu nie jest domknięty.** O ruchu tokenu decyduje gradient liczony z całego zbioru i wszystkich wag
   (Marcotte i in.: dynamika reprezentacji jest przepływem gradientu z metryką zależną od całego stanu, bez
   zamkniętej liniowej postaci). W języku Mori–Zwanziga token to rzut dużego stanu; brakujący człon pamięci
   zastępują opóźnienia prędkości. Dlatego 4 opóźnienia na skalarze wygrywają, a per-token nic nie pomaga.
3. **Za mało par na macierz per jednostka.** 47 par nie wyznacza macierzy 384×384 ani nawet 8×8 z dobrą
   ekstrapolacją; regresja wybiera dowolne rozszerzenie. Wspólny operator (48 obrazów × 47 par = 2256 par) jest
   identyfikowalny, ale tylko w małym wymiarze (k ≤ 4); w S_32 znów przegrywa.
4. **Jednostka to jeden kierunek, a nie „pełna identyfikacja”.** Dla jednej klasy górna podprzestrzeń Hessjanu
   (Gur-Ari i in.: wymiar = liczba klas, zamrożona po ~100 krokach; Xu i in.: aktualizacje wag w 2K-wymiarowej
   podprzestrzeni, K = wymiar etykiet) jest jednowymiarowa. Skalar LDA jest tym jednym, gładkim, przewidywalnym
   kierunkiem; wszystko poza nim to bulk, którego uczenie prawie nie rusza w sposób systematyczny.
5. **Tłumienie jest bliżej wykładniczego niż 1/t** w oknie 245–365 (α ≈ 0,9 na 5 aktualizacji); prawo 1/t z toy
   modelu Gur-Ariego przewiduje za wolne hamowanie (0,445 wobec 0,285 na 20 krokach).

## 4. Literatura (odczyty pełnych PDF przez Undermind, workspace „ViT: detekcja końcówki narzędzia”)

| Praca | Co modeluje | Co bierzemy |
|---|---|---|
| Dogra, Redman 2020, Optimizing Neural Networks via Koopman Operator Theory, arXiv 2006.02361 | wagi MLP, ok. 2800 aktualizacji, prognoza 2 epok | Koopman na wagach, nie aktywacjach |
| Redman i in. 2024, Identifying Equivalent Training Dynamics, NeurIPS, DOI 10.52202/079017-0744 | wagi ostatniej warstwy, DMD-RRR, 8–64 opóźnień, widma; bez prognozy, bez momentu | opóźnienia są standardem |
| Manojlović i in. 2020, Applications of Koopman Mode Analysis to Neural Networks, arXiv 2006.11765 | wagi i strata co epokę; wartości własne → 1 gdy uczenie zwalnia | przy hamowaniu widmo zbiega do 1 |
| Tano i in. 2020, Accelerating Training with DMD, arXiv 2006.14371 | wagi, 14 zrzutów → ekstrapolacja 41 kroków; DMD filtruje szum Adama; późno potrzebne wygaszanie | horyzont ekstrapolacji i przestrzelenie |
| Gur-Ari, Roberts, Dyer 2018, Gradient Descent Happens in a Tiny Subspace, arXiv 1812.04754 | gradient w k-wymiarowej górnej podprzestrzeni Hessjanu (k = liczba klas), zamrożonej po ~100 krokach; w niej θ ~ log t | jeden kierunek zadaniowy + bulk |
| Xu i in. 2026, Emergent Low-Rank Training Dynamics in MLPs, arXiv 2602.06208 | aktualizacje wag w podprzestrzeni wymiaru 2K | mały wymiar stanu uczenia |
| Marcotte, Peyré, Gribonval 2025, Intrinsic training dynamics, arXiv 2508.07370 | dynamika reprezentacji = przepływ gradientu z metryką, nieliniowy, bez zamkniętej postaci | brak liniowego prawa na położeniu |
| Aswani, Jabari 2025, Koopman Autoencoders Learn Neural Representation Dynamics, arXiv 2505.12809 | dynamika po warstwach (nie po treningu) | nie dotyczy uczenia |
| Aswani i in. 2024, Representing NN Layers as Linear Operations via Koopman, arXiv 2409.01308 | po warstwach; bez opóźnień wektory osobliwe są szumem | opóźnienia linearyzują |
| Henry 2026, Geometric Evolution Maps, arXiv 2605.25848 | kierunek pojęcia po warstwach; po „przekazaniu” stabilny (cos 0,94) | mierzą, nie przewidują |
| Park i in. 2023, Concept Evolution in Deep Learning Training, DOI 10.1145/3583780.3614819 | pojęcia neuronów w 3–4 checkpointach, bez modelu dynamiki | brak prognozy w literaturze |

Sześć prac z audytu równoległej sesji (Lu–Tartakovsky, Tu, Williams, Dogra, Ouala, Haseli–Cortés) jest opisanych
w `koopman_gesta_historia/AUDYT_LITERATURY_6_KOOPMAN.md`. **Żadna znaleziona praca nie przewiduje Koopmanem
ewolucji kierunku tożsamości/pojęcia w trakcie uczenia**; najbliższe są prace na wagach z opóźnieniami.
Scite: brak not redakcyjnych (wycofań, korekt) dla sprawdzonych DOI.

## 5. Część 2: pełna historia 5..800 (K = 160)

Dane: 160 checkpointów (kroki 5..800 co 5 aktualizacji); brakujące 370..800 pobrane z Edena
(`koopman_gesta_historia/wyniki/run_1810662/pretrain`, weryfikacja SHA), tokeny wyciągnięte lokalnie tym samym kodem
co część 1 do `wyniki/tokeny_pelne` (`ekstrakcja.py`; 160 plików, 9,5 GB, poza gitem). Dwa okna: **wczesne** S = 48
(dopasowanie do kroku 245, test 111 przejść 245..795) i **późne** S = 96 (dopasowanie do kroku 485, test 63 przejść
485..795). Raporty: `wyniki/REPORT_K160_S{48,96}.json`, `REPORT_najlepszy_K160_S{48,96}.json`,
`ROLLOUT_DLUGI_K160_S{48,96}.json` (`rollout_dlugi.py`). Uwaga techniczna: pełne tokeny zajmują 10 GB RAM; `licz.py`
i `najlepszy.py` zwalniają je zaraz po wycięciu potrzebnych trajektorii, inaczej drugi proces obok nie ma pamięci.

### 5a. Tożsamość na pełnej historii

- AUC LDA na holdout: 0,995 (krok 5), 0,998 (krok 245), 0,998 (krok 800). Bez zmian wobec części 1.
- Kierunek LDA obraca się dalej, powoli: cos z krokiem 5 spada do 0,64 w kroku 800 (0,73 w kroku 365).
- Średni przyrost skalara tożsamości na krok: 0,0085 na starcie, ok. 0,001 w oknie 245–365, ok. 0 (±0,0008) po kroku
  485. **Dryf tożsamości kończy się około kroku 500.** Od kroku 245 do 800 średnia przesuwa się łącznie o 0,128
  rozrzutu między obrazami, od kroku 485 do 800 tylko o 0,028 rozrzutu.

### 5b. Skalar tożsamości, holdout (RMSE względem braku zmian)

| okno | model | 1 krok | 20 kroków |
|---|---|---|---|
| wczesne (fit do 245, test 245..795) | **wspólny L4** | **0,407** | **0,171** |
| | alpha per obraz | 0,421 | 0,285 |
| | ostatni przyrost | 0,433 | 0,617 |
| późne (fit do 485, test 485..795) | **wspólny L4** | **0,682** | **0,454** |
| | alpha per obraz | 0,712 | 0,538 |
| | ostatni przyrost | 0,733 | 0,996 |

Bramka (≥ 5 procent lepiej od ostatniego przyrostu, ≥ 20 procent od braku zmian) spełniona w obu oknach dla L4, w oknie
wczesnym na 1 krok ledwo (0,407 wobec progu 0,411): na długim horyzoncie testu przewaga nad ostatnim przyrostem topnieje,
bo późne przejścia to prawie sam szum. Widmo macierzy towarzyszącej L4: |λ| max 0,944 (wczesne) i 0,939 (późne), oba
operatory stabilne i tłumiące; suma wag opóźnień 0,92.

### 5c. Długi rollout od kroku 245 do 800 (111 kroków, bez prawdziwych stanów pośrednich)

| horyzont | krok | L4 | alpha | ostatni przyrost | błąd średniej L4 (rozrzuty) | błąd średniej alpha |
|---|---|---|---|---|---|---|
| 20 | 345 | 0,171 | 0,285 | 0,62 | 0,000 | 0,000 |
| 40 | 445 | 0,254 | 0,344 | 1,04 | 0,000 | −0,009 |
| 60 | 545 | 0,290 | 0,426 | 1,66 | +0,026 | +0,002 |
| 80 | 645 | 0,359 | 0,403 | 2,02 | +0,039 | −0,005 |
| 111 | 800 | 0,417 | 0,451 | 2,85 | +0,078 | −0,002 |

L4 przewiduje **per obraz** lepiej niż alpha na całym horyzoncie (0,42 wobec 0,45 braku zmian po 555 aktualizacjach),
ale **średnią** (przyrost ogólny) przestrzeliwuje po kroku ok. 500: przewiduje łączny dryf 0,21 rozrzutu wobec
rzeczywistego 0,13. Tłumienie alpha własne każdego obrazu trafia w średnią do końca (błąd ≤ 0,01 rozrzutu). Ostatni
przyrost rozjeżdża się 2,8 raza gorzej niż brak zmian. Rollout od kroku 485 (okno późne, 63 kroki): L4 0,45 po 20
krokach i 0,58 na końcu, alpha 0,45 na końcu; błąd średniej obu ≤ 0,04 rozrzutu, bo dryfu już nie ma.

### 5d. Poziom tokenów i opisów w oknie późnym

- Per token (1458 jednostek, pełne 384): alpha 0,767, ostatni przyrost 0,797, pełna macierz na prędkości 1,02 (gorsza
  od braku zmian); podprzestrzeń tożsamości nie lepsza od losowej (S_8: 0,763 wobec 0,781). Jak w części 1.
- Opis narzędzia w S_k, 20 kroków: S_1 alpha 0,426, S_4 L4 0,463, S_32 alpha 0,557, 8 losowych kierunków poza S_32
  alpha 0,451. Losowe kierunki nadal nie gorsze niż tożsamość.

### 5e. Wnioski części 2

1. Dryf tożsamości to zjawisko wczesne: hamuje wykładniczo (α ≈ 0,89 na 5 aktualizacji) i kończy się przy kroku
   ok. 500; potem skalar tożsamości fluktuuje bez kierunku.
2. Wspólny operator prędkości z 4 opóźnieniami pozostaje najlepszym predyktorem per obraz w obu oknach i spełnia
   bramkę, ale jego przewaga maleje, gdy dryf znika (0,17 → 0,45 na 20 kroków).
3. Do przewidywania **przyrostu ogólnego** (średniej po obrazach) na długi horyzont lepsze jest tłumienie alpha per
   obraz: trafia w średnią z błędem ≤ 0,01 rozrzutu do kroku 800, L4 przestrzeliwuje o 0,08.
4. Dla pokazu FastGAN (sekcja 6): przyrost ogólny od kroku 245 do końca uczenia to 0,13 rozrzutu, a od kroku 800
   dalej mieści się w szumie (≤ 0,03). Powtórzenie pokazu „od kroku 800 do 900” nie ma sensu: Koopman przewiduje tam
   brak zmian ogólnych.

## 6. Pokaz: przewidziany latent (Koopman) jako cel dla FastGAN (2026-09-25, `pokaz_fastgan.py`)

Cel pokazu, zgodnie z poleceniem: wziąć **ogólny** (nie per obraz) przyrost latentu przewidziany przez Koopmana
na kolejne kroki uczenia, dodać go do obecnego latentu, zamrozić taki cel i w pętli przez zamrożony detektor
wygenerować FastGAN-em obraz, którego latent trafia w ten cel.

Ustawienie: operatory `OPERATORY_K73_S48.npz`, zamrożony detektor `model_0000245.pth` (krok 245), horyzont 20
kroków (cel = krok 345, sprawdzalny na prawdziwym checkpoincie). Kod K = 5: skalar tożsamości
`w_lda · (narzędzie − tło)` oraz 4 kierunki S_4 średniej narzędzia. Przyrost ogólny: wspólny operator prędkości
(4 opóźnienia) zastosowany do średniej po obrazach historii prędkości, zsumowany po 20 krokach.
Kotwice: `TR_0006` (fit), `TR_0082` (fit), `TR_0134` (holdout). Ramiona pętli jak w `koopman_fastgan_petla`:
`target` (cel Koopmana), `current` (obecny kod, kontrola), `random_norm` (losowy cel o tej samej normie);
2 restarty × 40 kroków, bramki: błąd skalowany ≤ 0,25, IoU maski ≥ 0,5, L1 do kotwicy ≤ 0,25.
Wyniki: `wyniki/fastgan_K73_krok0245/REPORT.json`, obrazy `kotwica_XX/{target,current,random_norm}/best.png`.

| wielkość | wartość |
|---|---|
| przyrost ogólny w jednostkach rozrzutu między obrazami (5 wymiarów) | 0,063; −0,002; 0,021; 0,018; 0,009 |
| błąd prognozy średniej wobec prawdziwego kroku 345 (w rozrzutach) | 0,001; 0,011; −0,014; −0,006; 0,012 |
| RMSE prognozy per obraz wobec „brak zmian” | 0,885 |
| błąd skalowany ramienia `current` (czy pętla odtwarza obecny kod) | 0,67 / 0,63 / 0,61 (bramka 0,25) |
| błąd skalowany ramienia `target` | 2,40 / 0,94 / 1,47 |
| błąd skalowany ramienia `random_norm` | 0,47 / 0,57 / 0,47 |
| bramki zaliczone | 0 z 9 |

Wnioski:

1. **Część Koopmanowska działa**: średni (ogólny) przyrost trafia w prawdziwy krok 345 z błędem poniżej 0,015
   rozrzutu w każdym z 5 wymiarów. To potwierdza, że wspólny operator prędkości przewiduje dryf tożsamości.
2. **Przyrost ogólny jest mały**: 0,06 rozrzutu wzdłuż tożsamości, ≤ 0,02 w kontekście. Prognoza per obraz
   jest tylko 11 procent lepsza od „brak zmian” (0,885), bo indywidualne przyrosty obrazów są większe od
   wspólnego dryfu i mają różne znaki (`przyrost_wlasny_rzeczywisty` w raporcie).
3. **Pętla FastGAN nie ma rozdzielczości, by ten przyrost pokazać.** Ramię kontrolne `current` (cel = obecny kod
   kotwicy) kończy z błędem 0,6 rozrzutu, dwa i pół raza powyżej bramki; najlepszy krok optymalizacji wypada
   zwykle w kroku 0–5, potem strata rośnie (start z losowego z, 40 kroków, brak inwersji kotwicy). Skoro pętla
   nie trafia nawet w obecny kod, różnica 0,06 rozrzutu między `target` i `current` ginie w szumie
   optymalizacji; obrazy `target` i `current` różnią się z powodu innego przebiegu optymalizacji, nie z powodu
   przyrostu. Obrazy są rozmytymi próbkami FastGAN (oko, cienka linia narzędzia), nie odtworzeniem kotwicy.
4. Aby pokaz miał sens, najpierw trzeba podnieść rozdzielczość pętli generatora: start z inwersji kotwicy
   (z dopasowane do obecnego kodu), więcej kroków, dopiero potem przesunięcie o przyrost; wariant ze wzmocnionym
   przyrostem (np. ×10) albo z dłuższym horyzontem musi być jawnie oznaczony jako niebędący prognozą Koopmana.

### 6b. Naprawiona pętla: inwersja kotwicy, potem ramiona od z* (`pokaz_fastgan_inwersja.py`, `wyniki/fastgan_K73_krok0245_inwersja/`)

Krajobraz straty inwersji jest chaotyczny: ten sam start z lr 0,05 kończy na 0,67 albo 2,4 zależnie od celu
różniącego się o 0,06 rozrzutu, a lr 0,01 przez 300 kroków grzęźnie przy 1,8. Zadziałało dopiero: faza zgrubna
6 startów × 100 kroków (lr 0,05), doszlif 2 najlepszych × 200 kroków (lr 0,01), zwycięzca po pełnej stracie. Potem
4 ramiona od tego samego z* i z tym samym szumem warstw (Adam, lr 0,01, 100 kroków): cel Koopmana, obecny, losowy
przyrost tej samej długości, oraz jawnie oznaczony `target_x10_NIE_PROGNOZA` (przyrost ×10). Przegląd obrazów:
`PRZEGLAD.png` w katalogu wyników.

| kotwica | inwersja: błąd / IoU / L1 | cel Koopmana: błąd, cos zmiany, L1 RGB do „obecny” | ×10 (nie prognoza): błąd, cos, L1 RGB |
|---|---|---|---|
| TR_0006 (fit) | 0,21 / 0,67 / 0,31 | 0,213, cos 0,31, 0,052 | 0,277, cos 0,92, 0,078 |
| TR_0082 (fit) | 0,27 / 0,46 / 0,19 | 0,268, ramię nie ruszyło się z z* (L1 RGB 0) | 0,313, cos 0,84, 0,018 |
| TR_0134 (holdout) | 0,09 / 0,40 / 0,23 | 0,099, cos 0,10, 0,008 | 0,160, cos 0,94, 0,021 |

„cos zmiany” = cosinus między osiągniętą zmianą kodu (względem ramienia „obecny”) a zamierzonym przyrostem.

Wnioski: (a) inwersja działa: błąd kodu 0,09–0,27 rozrzutu (bramka kodu ≤ 0,25 zaliczona w 2 z 3), obrazy mają
narzędzie w miejscu i orientacji kotwicy; pełne bramki nadal nie przechodzą przez IoU maski (0,40–0,67) i L1 do
prawdziwego zdjęcia. (b) Dla przyrostu Koopmana ×1 ramiona albo nie ruszają się z z* (TR_0082), albo ruch jest
szumem optymalizacji (cos 0,10–0,31; losowy przyrost daje cos 0,72 na TR_0134): przyrost 0,06 rozrzutu leży
poniżej rozdzielczości pętli Adama nawet po inwersji. (c) Pętla umie iść w kierunku Koopmana, gdy ma po co: przy
×10 cos 0,84–0,94 i 79–111 procent zamierzonej długości; obrazy różnią się od „obecny” o L1 RGB 0,02–0,08,
głównie w okolicy narzędzia.

### 6c. Krok lokalny Gaussa–Newtona od z* (`pokaz_fastgan_gn.py`, `wyniki/fastgan_K73_krok0245_gn/`)

Zamiast Adama: w z* liczona jest pochodna kodu po wejściu generatora (macierz 5 × 256, autograd), a krok to
rozwiązanie Levenberga–Marquardta najmniejszej normy `dz = J^T (J J^T + λI)^{-1} r` z podwajaniem tłumienia, gdy
reszta nie maleje; 12 iteracji, każde ramię od tego samego z* i szumu warstw. Wynik jest deterministyczny i lokalny,
więc różnice między ramionami pochodzą wyłącznie z różnic celów.

| kotwica | reszta do „obecny” z* → po kroku | cel Koopmana: reszta, cos zmiany, udział długości, L1 RGB | ×10 (nie prognoza): cos, udział, L1 RGB |
|---|---|---|---|
| TR_0006 (fit) | 0,207 → 0,204 | 0,211, cos 0,84, 59 %, 0,0018 | cos 0,85, 48 %, 0,0075 |
| TR_0082 (fit) | 0,273 → 0,127 | 0,157, cos 0,86, 169 %, 0,0092 | cos 0,97, 83 %, 0,0090 |
| TR_0134 (holdout) | 0,093 → 0,075 | 0,076, cos 0,97, 87 %, 0,0013 | cos 0,93, 80 %, 0,0078 |

Wnioski końcowe pokazu:

1. **Mechanika działa.** Po inwersji i kroku lokalnym generator realizuje zmianę latentu w kierunku przewidzianym
   przez Koopmana z cos 0,84–0,97 i 59–87 procent zamierzonej długości (jeden przypadek przestrzelenia, gdy ramię
   „obecny” samo się przesunęło). Losowy przyrost tej samej długości jest realizowany równie dobrze (cos 0,37–0,89):
   generator umie iść w każdym z 5 kierunków kodu, więc test nie wyróżnia kierunku Koopmana, tylko potwierdza, że
   pętla go nie gubi.
2. **Obraz „przewidzianego latentu” jest nieodróżnialny od obrazu obecnego.** Różnica L1 RGB 0,001–0,009 (0,1–0,9
   procent zakresu jasności) leży poniżej kwantyzacji PNG. Nawet ×10 daje 0,008–0,009. To nie wada generatora, tylko
   fizyka: ogólny dryf tożsamości na 100 aktualizacji to 6 procent rozrzutu między obrazami, a do końca uczenia
   13 procent (sekcja 5c). Obraz, który „odpowiada latentowi po kolejnych krokach uczenia”, wygląda jak obraz
   obecny. Różnica wzmocniona ×30 (`PRZEGLAD.png` w katalogu `_gn`) skupia się na narzędziu i jego krawędziach,
   zwłaszcza dla TR_0082: przewidziany dryf tożsamości zmienia wygląd narzędzia, nie tła; przy ×10 zmiany rozlewają
   się także na tło.
3. Bramki pełne (kod ≤ 0,25, IoU ≥ 0,5, L1 ≤ 0,25) nie przechodzą w żadnym ramieniu; ogranicza je zgodność maski
   i kolorów z prawdziwym zdjęciem, nie kod.
4. Co dalej ma sens: nie powtarzanie pokazu od kroku 800 (dryf ≈ 0, sekcja 5e), lecz użycie tej samej pętli
   (inwersja + krok Gaussa–Newtona) do kierunków o dużej amplitudzie, np. różnic między obrazami albo między
   klasami wariantów sond, jeśli celem jest generowanie obrazów o zadanym latencie.
