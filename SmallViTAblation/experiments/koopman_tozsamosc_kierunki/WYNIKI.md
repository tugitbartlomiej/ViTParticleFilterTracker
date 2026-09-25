# Czy Koopman przewiduje elementy latentu identyfikujące narzędzie? Wyniki (2026-09-25)

Stan: część 1 (73 checkpointy, kroki 5..365) i część 2 (160 checkpointów, kroki 5..800, sekcja 5) zakończone.
Pokaz FastGAN na przewidzianym latencie (sekcja 6) wykonany: pętla działa mechanicznie, obrazy nie różnią się od
obecnego latentu. Dobór klatek i kontynuacja treningu (sekcja 7) zakończone: 134 dobrane klatki dają tyle samo,
co 134 losowe (+0,001 mAP@0,5, w rozrzucie ziaren); efekt lokalny na klatkach val podobnych do kotwic (7d) też
zerowy (dobrane −0,003, losowe +0,005, SD 0,007–0,009), a okolica kotwic jest dla detektora trudna (AP 0,55).

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

## 7. Dobór nowych klatek do obrazów przewidzianego latentu i kontynuacja treningu (2026-09-25, wieczór)

Zlecenie: znaleźć klatki, które nie weszły do treningu detektora, dopasowane do obrazów FastGAN przewidzianego
latentu; dołożyć je do zbioru 1500 klatek pretrainingu; trenować kolejną epokę; sprawdzić, czy detekcja się poprawia.

Uwaga o kotwicach pokazu: TR_0082 (klatka 1656) i TR_0134 (klatka 341) **były** w 1500 klatkach treningowych biegu
1810662 (`datasets/pretrain/train/_annotations.coco.json`); tylko TR_0006 (klatka 746) nie była. „Holdout” w
sekcjach 1–6 oznaczał wyłącznie wyłączenie z dopasowania operatora Koopmana.

### 7a. Wyszukiwanie (`wyniki/dobor_klatek/`)

Wyszukiwarka równoległej sesji (`koopman_fastgan_petla/wyszukiwanie.py`: DINOv2 ViT-S/14, średnia tokenów, FAISS,
indeks `local_1810448/run/index` nad 10 637 klatkami train z wycinkiem kwadratowym wokół ramki). Zapytania: obrazy
`target.png` (i kontrolnie `current.png`) z `fastgan_K73_krok0245_gn/kotwica_XX`. Wykluczone: 1500 klatek
pretrainingu i 3 kotwice; kwalifikujących się klatek 9 136. Ranking „najlepsza z każdego filmu, potem druga”.
Wyniki: `retrieval_kotwica_XX_{target,current}.json`, przegląd `KANDYDACI.png`, wybór `kandydaci_top6.json`.

| kotwica | top-3 (film/klatka, cos DINOv2) |
|---|---|
| TR_0006 | TR_0076/1816 0,752; TR_0036/611 0,747; TR_0055/5481 0,747 |
| TR_0082 | TR_0038/1591 0,746; TR_0018/9001 0,745; TR_0081/651 0,729 |
| TR_0134 | TR_0062/391 0,812; TR_0096/5506 0,807; TR_0148/356 0,803 |

Rankingi dla `target.png` i `current.png` są niemal identyczne (obrazy różnią się o 0,1 procent jasności), więc
dobór odzwierciedla obraz obecnego latentu kotwicy, nie przyrost Koopmana. Wszystkie wyniki przekraczają bramkę
kampanii równoległej sesji (cos ≥ 0,7). Do treningu wzięto top-6 z każdej kotwicy: **17 unikalnych klatek z 14
filmów**, żaden z filmów kotwic.

### 7b. Kontynuacja treningu na Edenie (`eden/kontynuacja.py`, `eden/kontynuacja.slurm`, zadanie 1814538)

- Start: wagi `model_0000245.pth` biegu 1810662 (ten sam checkpoint, na którym liczono prognozę Koopmana i
  pokaz), świeży optymalizator (checkpoint ma tylko wagi); receptura i kod `trening.py` równoległej sesji bez zmian
  (batch 4 × 4 akumulacje, LR 1,67e-5 stały, deterministyka, sonda co epokę).
- Jedna epoka = 100 aktualizacji = 1600 prezentacji (wszystkie klatki raz plus dopełnienie losowe).
- Ramiona: `bez_dodania` (1500), `koopman_fastgan` (1500 + 17 znalezionych), `losowe` (1500 + 17 losowych
  nieużytych klatek train, ziarno 42). Ziarna treningu 42, 43, 44 (tablica SLURM, hopper H100).
- Miara: walidacja na całym zbiorze val (12 043 klatki) po epoce: mAP@0,5, mAP@0,5:0,95, F1. Odniesienie z biegu
  1810662: koniec epoki 3 (krok 299) mAP@0,5 0,860, epoki 4 (krok 399) 0,877.
- Oczekiwanie a priori: 17 klatek to 1,1 procent zbioru; efekt, jeśli jest, będzie rzędu szumu między ziarnami.
  Dlatego trzy ziarna i dwa ramiona kontrolne.

Wyniki pilotu (zadanie 1814538, przerwane na polecenie użytkownika po czterech ramionach, żeby zwolnić GPU dla
kampanii z większą liczbą klatek; raporty w `wyniki/dobor_klatek/trening/raporty/`):

| ramię | ziarno 42 | ziarno 43 |
|---|---|---|
| bez_dodania (1500) | 0,8667 | 0,8688 |
| koopman_fastgan (1517) | 0,8667 | (przerwane) |
| losowe (1517) | 0,8663 | (przerwane) |

W ziarnie 42 wszystkie trzy ramiona mieszczą się w 0,0004 mAP@0,5, a sama zmiana ziarna w ramieniu bez dodania
przesuwa wynik o 0,002. Siedemnaście klatek to za mało, żeby cokolwiek zobaczyć; stąd kampania 7c.

### 7c. Kampania 134 klatek: top-50 na kotwicę (`eden/kontynuacja150.slurm`, zadanie 1814611, `wyniki/dobor_klatek/trening150/`)

Dobór: dla każdej z trzech kotwic 50 najbliższych klatek z wyszukiwarki (`retrieval50_kotwica_XX_target.json`,
ten sam indeks, wykluczenia i ranking co w 7a; cos DINOv2 od 0,66 do 0,81). Suma trzech list po usunięciu
powtórzeń (4, 7 i 8 klatek wspólnych między parami kotwic): **134 unikalne klatki z 84 filmów**, 8,9 procent
zbioru 1500. Kontrola losowa: 134 losowe nieużyte klatki train (ziarno 43, 74 filmy). Epoka = 103 aktualizacje
(`epoch_size` 1648, bo sampler wymaga rozmiaru epoki ≥ liczby klatek i podzielnego przez 16); to samo 103 dla
wszystkich ramion, także bez dodania. Reszta receptury jak w 7b. Ziarno 42 liczone na hopper, ziarna 43 i 44 na
dgx-1 (partycja short; przeniesione na życzenie użytkownika, bo hopper miał jedną wolną kartę). Każde ramię
13–14 minut; skrypt sprząta `/dev/shm` przez trap, po zadaniu nic nie zostało na węzłach. Raporty:
`trening150/raporty/seed{S}_{ramię}/{REPORT,INPUTS}.json`, logi SLURM w `trening150/logi/`, podsumowanie
`podsumuj_kontynuacje.py` → `PODSUMOWANIE_KONTYNUACJA.json`, rysunek `KONTYNUACJA.png`.

mAP@0,5 na całym val (12 043 klatki) po jednej epoce od kroku 245:

| ramię | ziarno 42 | ziarno 43 | ziarno 44 | średnia ± SD | mAP@0,5:0,95 (śr.) | F1 (śr.) |
|---|---|---|---|---|---|---|
| bez_dodania (1500) | 0,8678 | 0,8688 | 0,8694 | 0,8686 ± 0,0008 | 0,6175 | 0,8681 |
| koopman_fastgan (1634) | 0,8696 | 0,8695 | 0,8691 | 0,8694 ± 0,0002 | 0,6178 | 0,8697 |
| losowe (1634) | 0,8684 | 0,8698 | 0,8706 | 0,8696 ± 0,0011 | 0,6178 | 0,8689 |

Różnice sparowane (to samo ziarno) względem braku dodania, mAP@0,5: dobrane +0,0018, +0,0007, −0,0003
(średnio **+0,0007**); losowe +0,0006, +0,0010, +0,0012 (średnio **+0,0009**). mAP@0,5:0,95: obie +0,0003.
F1: dobrane +0,0016, losowe +0,0008. Dla skali: odniesienie z biegu 1810662 to 0,860 na końcu epoki 3 (krok 299)
i 0,877 na końcu epoki 4 (krok 399); jedna epoka od kroku 245 daje tu 0,869 w każdym ramieniu, czyli zwykły
postęp krzywej uczenia, a nie efekt doboru.

Werdykt:

1. Dołożenie 134 klatek **nie pogarsza** detekcji i podnosi mAP@0,5 o około 0,001, tyle samo dla klatek dobranych
   i dla losowych. Ta poprawa mieści się w rozrzucie ziaren (SD 0,0008–0,0011 przy trzech ziarnach), więc nie
   jest rozstrzygnięta; różnica dobrane − losowe (−0,0002 mAP@0,5, +0,0008 F1) tym bardziej nie.
2. Dobór przez Koopman → FastGAN → wyszukiwarkę **nie jest lepszy od losowego** w tej próbie. Nie jest to
   zaskoczenie po 7a: obrazy przewidzianego i obecnego latentu różnią się o 0,1 procent jasności, więc zapytania
   do wyszukiwarki niosły wyłącznie „podobieństwo do trzech kotwic”, a nie kierunek zmiany. Test sprawdza zatem,
   czy klatki podobne do trzech kotwic uczą lepiej niż losowe; przy 134 klatkach i jednej epoce nie uczą.
3. Czego ten test nie rozstrzyga: efektu lokalnego (czy detektor poprawił się na klatkach val podobnych do kotwic;
   miara globalna po 12 043 klatkach go nie widzi), efektu wielu epok ani większych porcji. Żeby rozdzielić
   różnice rzędu 0,001 przy SD 0,001, trzeba by około dziesięciu ziaren na ramię albo miary lokalnej.

### 7d. Efekt lokalny: klatki val podobne do kotwic (`efekt_lokalny.py`, `wyniki/dobor_klatek/efekt_lokalny/`)

Pytanie: może dobrane klatki pomagają tam, gdzie były celowane, a miara globalna po 12 043 klatkach tego nie widzi.
Procedura (lokalnie, RTX 3070, wagi `FINAL.pth` dziewięciu ramion pobrane z Edena do `trening150/wagi/`, poza gitem):

1. Podobieństwo: ten sam DINOv2 ViT-S/14, wycinek i preprocessing co w indeksie train (`wyszukiwanie.py` równoległej
   sesji, tylko odczyt), cechy wycinków wszystkich 12 043 adnotacji val, cosinus do zapytań `target.png` trzech
   kotwic; klatka dostaje maksimum po kotwicach (`PODOBIENSTWO_VAL.npz`). Mediana 0,58; 271 klatek ma cos ≥ 0,7;
   próg top-134 to 0,717. Zapytania `target` i `current` tej samej kotwicy mają cos 0,99–1,00, więc ranking val,
   jak w 7a, odzwierciedla obecny latent kotwicy.
2. Predykcje: każdy z dziesięciu zestawów wag (9 ramion + start `model_0000245.pth`) na wszystkich klatkach val,
   próg pewności 0,05, `PREDYKCJE_{nazwa}.npz`. AP COCO (101 punktów, IoU 0,5 i 0,5:0,95) liczone lokalnie
   z tych samych predykcji dla dowolnego podzbioru; F1 przy pewności 0,5. Kontrola implementacji: na całym val
   lokalne AP@0,5 0,8659 wobec 0,8686 z Edena (różnica 0,003 z odcięcia predykcji poniżej 0,05 i szczegółów
   COCO), identyczna procedura dla wszystkich ramion, więc różnice sparowane są wiarygodne.
3. Podzbiory według podobieństwa: top-134 (tyle, ile dołożono do treningu), cos ≥ 0,7 (271), 10 procent
   najbliższych (1 204), top-50 na kotwicę, dolna połowa (6 022, kontrola), decyle.

Najpierw fakt o samych klatkach: **okolica kotwic jest trudna**. AP@0,5 ramienia bez dodania: wszystkie 0,866,
10 procent najbliższych 0,828, cos ≥ 0,7 0,688, top-134 0,545; top-50 kotwicy TR_0134 (`kotwica_48`) tylko 0,383
(TR_0006 0,841, TR_0082 0,803). Profil po decylach podobieństwa: 0,83 w najbliższym decylu, 0,90–0,93 w środkowych,
0,65 w najdalszym (klatki niepodobne do żadnej kotwicy). Start (krok 245) na top-134 miał 0,526, więc epoka
podnosi ten podzbiór o 0,02 w każdym ramieniu.

Δ AP@0,5 względem braku dodania, średnia po trzech ziarnach ± SD różnic (`EFEKT_LOKALNY.json`, `EFEKT_LOKALNY.png`):

| podzbiór val (n) | dobrane − bez dodania | losowe − bez dodania |
|---|---|---|
| wszystkie (12 043) | +0,0003 ± 0,0003 | +0,0006 ± 0,0006 |
| dolna połowa (6 022) | +0,0002 ± 0,0006 | −0,0004 ± 0,0011 |
| 10 procent najbliższych (1 204) | −0,0020 ± 0,0039 | +0,0027 ± 0,0043 |
| cos ≥ 0,7 (271) | −0,0033 ± 0,0067 | +0,0035 ± 0,0061 |
| top-134 | −0,0028 ± 0,0085 | +0,0054 ± 0,0070 |
| suma top-50 kotwic (138) | −0,0067 ± 0,0113 | +0,0035 ± 0,0064 |
| kotwica TR_0006 top-50 | −0,0094 ± 0,0132 | +0,0009 ± 0,0011 |
| kotwica TR_0082 top-50 | −0,0117 ± 0,0063 | −0,0067 ± 0,0049 |
| kotwica TR_0134 top-50 | +0,0043 ± 0,0245 | +0,0139 ± 0,0168 |

Decyle: dobrane −0,002 ± 0,004 w najbliższym decylu i 0 do +0,002 w pozostałych; losowe +0,003 ± 0,004 w najbliższym
i 0 do +0,003 w pozostałych. mAP@0,5:0,95 i F1 nie zmieniają obrazu (top-134: dobrane −0,001 / −0,001, losowe
+0,004 / +0,002).

Werdykt lokalny:

1. **Nie ma efektu lokalnego.** Na klatkach val najbardziej podobnych do kotwic dobrane klatki dają −0,003 AP@0,5,
   losowe +0,005; oba mieszczą się w rozrzucie ziaren (SD 0,007–0,009 na 134 klatkach), a znak dobranych jest w
   trzech z czterech podzbiorów ujemny. Dodanie klatek podobnych do kotwic nie poprawiło detekcji w ich okolicy
   bardziej niż dodanie losowych.
2. Trudność okolicy kotwic (AP 0,38–0,55 dla TR_0134 i top-134) to jedyna rzecz, którą ten pomiar pokazał z zapasem:
   FastGAN i wyszukiwarka wskazały region, w którym detektor jest słaby, ale 134 klatki train z tego regionu przez
   jedną epokę go nie naprawiły. Żeby rozstrzygnąć efekt rzędu 0,003 na 134 klatkach, trzeba by kilkunastu ziaren
   albo kilkuset klatek val w tym regionie, których nie ma (271 z cos ≥ 0,7).
3. Razem z 7c: ani globalnie, ani lokalnie dobór Koopman → FastGAN → wyszukiwarka nie różni się od losowego doboru
   w tej próbie. Przyczyna po stronie wejścia pozostaje ta z 7a: obrazy przewidzianego latentu nie różnią się od
   obrazów obecnego, więc dobór nie niósł kierunku Koopmana.

### 7e. Czy dodane klatki były trudne? (`trudnosc_dodanych.py`, `dobor_trudnych.py`, `podglad_trudnych.py`)

Hipoteza użytkownika po 7d: skoro detektor jest słaby w okolicy kotwic, generator wskazał przypadki, które przy
dłuższym treningu wzmocnią detektor. Sprawdzenie na samych klatkach treningowych (`efekt_lokalny/TRUDNOSC_DODANYCH.json`):

| wagi | 134 dobrane (AP@0,5) | 134 losowe | 134 z widzianych 1500 |
|---|---|---|---|
| start, krok 245 | 0,857 | 0,878 | 0,868 |
| bez_dodania po epoce (klatek nie widział), 3 ziarna | 0,889–0,907 | 0,901–0,905 | 0,900–0,906 |
| koopman_fastgan po epoce (widział je raz) | 0,907–0,912 | 0,899–0,901 | 0,905–0,910 |

Dobrane klatki treningowe **nie były trudne**: przed treningiem tylko o 0,02 gorsze od losowych, a po epoce bez
oglądania ich detektor i tak ma na nich 0,90. Trudna jest okolica kotwic **w walidacji**, i to w kilku filmach:
134 najbliższe klatki val pochodzą z 10 filmów, 101 z czterech (TR_0100, TR_0002, TR_0060, TR_0056); film TR_0060
ma AP@0,5 0,31 w całości, TR_0147 0,49, TR_0002 0,73, reszta 0,74–1,00. Klatki treningowe podobne wyglądem do
kotwic nie przenoszą tego, co w tych filmach val jest trudne.

Kontrola „trudne przypadki” bez generatora (`trening5ep/selekcja_trudne.json`): 134 nieużyte klatki train,
na których detektor z kroku 245 najmniej trafia (trudność = 1 − max pewność × IoU; wybrane 0,945–1,000 przy
medianie puli 0,48), najwyżej 2 na film, 83 filmy (dobrane: 84). Wspólne z dobranymi 0, z losowymi 1. Podgląd
24 losowych (`TRUDNE_PODGLAD.png`): narzędzie wchodzące przy krawędzi kadru, malutki czubek, detektor zaznacza
inne narzędzie (pęsetę, rozwieracz) zamiast oznaczonego cystotomu; część to przypadki graniczne oznaczeń
(bardzo małe ramki), więc ramię może też nieść szum etykiet.
