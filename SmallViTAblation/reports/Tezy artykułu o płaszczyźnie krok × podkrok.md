# Tezy badawcze do jednego artykułu: płaszczyzna krok uczenia × podkrok przejścia w małym ViT

Notatka z 2026-10-23. Źródła liczb: eksperyment `experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29` (bieg s47, 806
checkpointów × 12 podkroków, klatka TRAIN 0, jakobiany 7 cząsteczek × 7 checkpointów) oraz wcześniejsze pomiary projektu
(ZASADY Z1–Z24, KURCZENIE_UWAGA_MLP, eksperymenty z 2026-10-21/22 na seedzie 3047). Wszędzie: jeden bieg = jedna liczba, nie uogólnienie.

## 0. Jedno zdanie

Jedno przejście tokenów przez wytrenowany ViT jest hiperbolicznym transportem cząstek (ściskanie w większości kierunków, rozciąganie w kilku),
a nie dynamiką pułapek; uczenie buduje ten transport w ustalonej kolejności (najpierw uwaga zamienia translację chmury w gwiazdę wokół
narzędzia, potem MLP rozdyma, potem bloki specjalizują się i zaczynają się nawzajem poprawiać) — i widać to wyłącznie, gdy jednocześnie
śledzi się głębokość i czas uczenia.

## 1. Tytuły robocze

- PL: *Translacja, gwiazda i mapa ściskająca: jak uczenie przebudowuje przejście tokenów przez głębokość małego ViT*
- EN: *Attention as a learned translation and the MLP as a squeezing field: the depth-by-training plane of a small vision transformer*
- EN (krótszy): *What a vision transformer does to its tokens, substep by substep, checkpoint by checkpoint*

## 2. Teza główna i cztery tezy składowe

**Teza główna (T0).** Rozkład przejścia na 12 podkroków (uwaga i MLP sześciu bloków) jest dokładny (błąd ≤ 10⁻⁴ w rzutach, ≤ 4·10⁻⁷ w
odtworzeniu stanów) i daje obserwable, których profil po głębokości zmienia się w uczeniu w sposób uporządkowany. Literatura ma albo
profil po głębokości w jednym modelu (Geshkovski 2023, Castin 2025, Isobe 2026), albo profil po checkpointach jednej warstwy (Varam
2025, Järve 2026); płaszczyzna krok × podkrok jest nowym obiektem pomiaru (wg podsumowania Undermind z 2026-10-23 nikt nie śledzi razem
kurczenia przez uwagę i MLP po warstwach i po checkpointach).

### T1. Uwaga jest wyuczoną translacją chmury plus sprzężeniem gwiazdowym do kilku tokenów; afiniczna dokładnie tylko na starcie

Treść. Przy ustalonych wagach A przesunięcie Δx_i = Σ_j A_ij W v_j + b jest liniowe w chmurze (kombinacja wypukła wektorów wartości).
Na starcie A jest jednostajne, więc uwaga to czysta translacja. Uczenie w ułamku epoki zamienia ją w gwiazdę: każdy token słucha głównie
tokenów narzędzia. Mechanizm przestaje być afiniczny (zależność wag od położeń), ale przesunięcie wygląda afinicznie, bo większość
chmury dostaje ten sam wektor „od narzędzia”.

Co już wynika z pomiarów (s47, TRAIN 0):
- udział wspólnego przesunięcia w ruchu podkroku uwagi 1,000 w kroku 0 (wszystkie 6 bloków), 0,2–0,9 po uczeniu;
- efektywne źródła 1/ΣA²: 570–590 z 600 na starcie → 10–66 po uczeniu; entropia 6,38 nat (maks. 6,40) → 3,1–4,7;
- kolaps najpierw w bloku 5 (< 300 źródeł od kroku 120, < 50 od 310) i 4 (230 / 450), bloki 1–3 i 6 między krokiem 1000 a 11 300;
  bloki 1 i 6 potem znów się rozszerzają (32 → 66; 3 → 40): „najpierw rzadka, potem gęsta” (zgodne z JoMA, Tian 2023);
- 0 % uwagi na sobie, 0–1 % na 8 sąsiadach siatki (jednostajna dałaby 1,3 %), średnia odległość 10–12 łat (jednostajna 12,9);
- wzbogacenie tło → narzędzie 40× (blok 4), 37× (blok 5); narzędzie → narzędzie 60–75×;
- pełny jakobian uwagi: część barycentryczna (zamrożone A) 95–98 % na starcie, 20–70 % po uczeniu; efektywna liczba cząsteczek
  wpływających 540–580 → 2–23;
- CV-R² afinicznego dopasowania Δ do własnego położenia (192D): uwaga 0,79–0,91, MLP 0,65–0,89 w 60. epoce (zero −2…−5);
- z seedu 3047 (eksperyment mechanizmów, 25 filmów, epoka 100): zostawienie samego wspólnego wyjścia uwagi 3 kosztuje −0,009 IoU, usunięcie
  wspólnego wyjścia −0,176; udział uwagi na tokenach ramki 14,7× — niezależne potwierdzenie, że translacja jest funkcjonalnie główną częścią.

Czego brakuje: replikacja na ≥ 3 seedach i ≥ 10 klatkach (zwłaszcza z większym narzędziem niż 3 łaty); sprawdzenie, czy gwiazda zawsze
jest na narzędziu (predykcja P2 niżej).

### T2. MLP to pole punktowe o strukturze siodła z wirem w każdym bloku; złożenie siodeł nie łapie, lecz ściska

Treść. Jakobian własny każdego z 12 podkroków (także uwagi, przy ustalonych innych tokenach) kurczy połowę kierunków i rozciąga połowę.
Iloczyn 12 jakobianów też jest siodłem, bo podprzestrzenie kurczące kolejnych bloków są obrócone losowo (brak ogniskowania
naprzemiennego). Uczenie zwiększa hiperboliczność: przejście staje się mapą ściskającą objętość przy silnym rozciąganiu w kilku
kierunkach. Słowo „pułapka” dotyczy obrazu w aplikacji, nie dynamiki przejścia.

Co już wynika z pomiarów (7 tokenów × 7 checkpointów, float64):
- udział kurczących kierunków części symetrycznej map własnych 0,50–0,53 w każdym podkroku, checkpoincie i tokenie; ‖A‖/‖S‖ ≈ 1,0 dla
  uwagi, 0,79–0,95 dla MLP (zamyka otwarty test 11 z ZASADY);
- iloczyn 12 jakobianów: udział σ < 1 wynosi 0,47–0,54; nakładanie podprzestrzeni kurczących kolejnych bloków 0,52–0,55 wobec losowego
  0,51–0,53;
- log σ_max iloczynu 0,85 (start) → 2,1–2,6; log σ_min −1,0 → −5,3…−5,6; wykładniki po głębokości +0,18 / −0,44 na podkrok;
- log|det| całego przejścia −1,1 → +1,8 (5 ep) → −18,5 (narzędzie, 60 ep) i −6,2 (tło); MLP 1 rozdyma (+5…+10), MLP 4–6 ściskają (−4…−6);
- rozrzut chmury: MLP 1 ×1,05 → ×2,0–2,3 (szczyt 2,29 w kroku 31 000), MLP 4 ×1,12 → ×1,32 (szczyt 1,40 w kroku 60 000), uwaga 0,99–1,06;
- wcześniejsze pomiary projektu: zero pola w komórce tokenu w 0,005 % par checkpoint × cząsteczka (mapa pułapek), 0 równowag w 192D
  (odległe równowagi, seed 3047), ślad jakobianu bloku 6 ujemny (Z6).

Analogie, które pasują i nie pasują: siodło obrotowe / pułapka Paula / optyczne siodło (Tandeitnik 2024) stabilizują tylko przy uporządkowanej
rotacji siodła szybszej niż jego krzywizna; ogniskowanie naprzemienne (Courant–Snyder 1958) wymaga uporządkowanej sekwencji soczewek.
Tu rotacja jest losowa, więc sieć jest linią przypadkowo ustawionych kwadrupoli: rozbieżna w części kierunków, zbieżna w innych.
Pasuje za to obraz „wąskiego gardła” iloczynu jakobianów (Fernando, Guitchounts 2026).

Czego brakuje: więcej tokenów (np. 60 losowych) i seedów; efektywny rząd iloczynu; sprawdzenie, czy kierunki rozciągane iloczynu pokrywają
się z osią narzędzie–tło (hipoteza: tak, bo kontrakcja objętości dla narzędzia −18,5 jest silniejsza niż dla tła −6,2).

### T3. Uczenie przebudowuje profil po głębokości w ustalonej kolejności, a rozdział narzędzie–tło przesuwa się w głąb długo po nasyceniu IoU

Treść. Cztery fazy: (i) kolaps uwagi w blokach 5 i 4 (kroki 100–450); (ii) spójna inflacja: wszystkie MLP pchają w tę samą stronę
(wskaźnik balistyczny 12 podkroków 0,98 → 2,88 w 1. epoce, cos kolejnych MLP 0,55–0,63); (iii) dekoherencja i specjalizacja: wskaźnik
balistyczny wraca do 0,97 (60 ep), bloki 1–3 zostają radialne (cos 0,19–0,36), bloki 4–6 stają się styczne lub do środka (−0,02…−0,13);
(iv) korekta sąsiednich podkroków (cos(MLP b, uwaga b+1) −0,05…−0,14, cos(uwaga b, MLP b) −0,11…−0,16 w blokach 5–6; w 1. epoce były
dodatnie) i front rozdziału: skok Fishera w bloku 3 w 1. epoce (1,54 → 2,16), w bloku 4 w 60. (1,34 → 3,20), szczyt 3,49 w bloku 5, blok 6
obniża do 2,92; Fisher > 3 po MLP 4 dopiero od kroku 44 300 (pewność detektora 0,9 od ok. 10. epoki).

Co już wynika: wszystkie liczby wyżej z 806 checkpointów; Spearman wskaźnika balistycznego z krokiem −0,89; Fishera na wyjściu z liczbą
źródeł uwagi bloku 4 −0,62, z pewnością +0,77; MLP 4 jako jedyny blok rośnie do końca (rozrzut, 19 % zapalonych laserów, uwaga na
narzędziu 40×). Z seedu 3047: MLP 3 i MLP 4 podnoszą separację o 43 % i 49 %, MLP 6 obniża o 11 % (epoka 200) — ten sam kształt frontu.

Odpowiedniki w literaturze: mechanizm korekty sąsiednich warstw (Patrawala 2026: cos −0,15…−0,3, powstaje w uczeniu) — tu słabszy i też
uczony; składowa styczna FFN (Mudarisov 2026) — tu z profilem w czasie uczenia; ucieczka z klastrowania w ostatnich warstwach (Isobe 2026)
— tu blok 6 nie ucieka, tylko lekko cofa rozdział.

Czego brakuje: 3 tokeny narzędzia to za mało na Fishera (klatki z większym narzędziem), zero przestrzenne dla Fishera po stanach,
replikacja faz na seedach 3047 (46 checkpointów 0–400 ep) i innych.

### T4. Uwaga w tym ViT nie jest filtrem dolnoprzepustowym po siatce; wygładza MLP

Treść. Narracja „uwaga wygładza, FFN wyostrza” (Wang 2022, Dovonon 2024) nie opisuje tego detektora: uwaga nie patrzy na sąsiadów,
a energia Dirichleta cech po siatce spada w podkrokach MLP, bo MLP rozpycha chmurę wzdłuż kierunków wspólnych sąsiadom.

Co już wynika: Dirichlet/(2·wariancja) na wejściu 0,06 (krok 0) → 0,31 (60 ep); po przejściu 0,066 → 0,116; w 60. epoce podkroki uwagi
zmieniają go o ≤ 0,013, MLP 1 o −0,127, MLP 4 o −0,026; masa uwagi na 8 sąsiadach 0,00–0,01 we wszystkich blokach i checkpointach.

Czego brakuje: inne klatki; rozróżnienie, ile wygładzania to wzrost wariancji chmury, a ile spadek różnic sąsiadów (oba człony osobno);
porównanie z ViT klasyfikacyjnym, gdzie efekt dolnoprzepustowy jest raportowany.

## 3. Czego z pomiarów NIE wolno jeszcze twierdzić

- Niczego o innych biegach, seedach, klatkach i architekturach (jeden bieg s47 na płaszczyźnie; seed 3047 tylko we wcześniejszych
  pomiarach, inne klatki, inne checkpointy).
- Że gwiazda uwagi jest *przyczyną* rozdziału (korelacja −0,62 i ablacja z innego seedu to nie dowód na s47).
- Że „pułapek nie ma w ogóle”: jakobiany własne liczono dla 7 tokenów przy zamrożonych innych; pełny jakobian 600 × 192 tylko dla uwagi.
- Liczb Fishera jako miary ogólnej (3 tokeny narzędzia; kNN nieużyteczne).
- Wartości CV-R² jako bezwzględnych (193 parametry na 300 tokenów); ważne są różnice uwaga/MLP i zero.
- Że w aplikacji „widać afiniczność”: rzut 3D ma R² 0,2–0,9 bez wzoru; afiniczność jest w 192D.

## 4. Predykcje do replikacji (falsyfikowalne, przed pomiarem)

- P1. W każdym seedzie liczba efektywnych źródeł uwagi w blokach 4–5 spada poniżej 50 zanim IoU VAL przekroczy 0,3 (s47: krok 310–450;
  IoU 0,31 w 0,27 ep ≈ krok 300 na seedzie 3047).
- P2. W ≥ 90 % klatek TEST tokeny o największym wzbogaceniu uwagi w blokach 4–5 leżą w obrysie narzędzia lub ≤ 1 łatę od niego.
- P3. Wskaźnik balistyczny 12 podkroków ma maksimum 2–3 między 0,5 a 2 epoką i spada poniżej 1,2 przed 20. epoką.
- P4. Udział σ < 1 iloczynu 12 jakobianów własnych mieści się w 0,45–0,55 dla każdego tokenu i checkpointu, a nakładanie podprzestrzeni
  kurczących kolejnych bloków nie przekracza losowego o więcej niż 0,03.
- P5. Od 20. epoki największy skok Fishera po stanach jest w bloku 4 (lub 3) w większości klatek, a blok 6 go nie podnosi.
- P6. Masa uwagi na 8 sąsiadach siatki < 2 × jednostajna (0,026) w każdym bloku i checkpoincie; Dirichlet spada w podkrokach MLP, nie uwagi.
- P7. log|det| całego przejścia jest ujemny od ok. 10. epoki i silniejszy dla tokenów narzędzia niż tła.

Negatywny wynik P4 (złożenie kurczy) byłby ciekawszy niż pozytywny: oznaczałby ogniskowanie naprzemienne i otwierał analogię Couranta–Snydera.

## 5. Szkielet artykułu

1. Wstęp: tokeny jako cząstki (Geshkovski, Castin, Rigollet), czas = głębokość (Lu 2019, Lie–Trotter), brak pomiarów na płaszczyźnie krok × podkrok.
2. Układ: MałyViT, 600 tokenów, 12 podkroków, dokładny rozkład sił (Z19–Z20), 806 checkpointów; obserwable w 192D; jakobiany autograd.
3. Wynik 1 (T1): translacja → gwiazda; F2, F3, F8D; tabela źródeł i wzbogacenia; CV-R².
4. Wynik 2 (T2): siodło wszędzie, iloczyn ściska; F7, F8A–C; analogie i ich granice.
5. Wynik 3 (T3): kolejność faz, front rozdziału, korekta; F1, F4, F5, F9.
6. Wynik 4 (T4): gładkość po siatce; F6.
7. Replikacja (P1–P7) na seedach i klatkach — do zrobienia przed wysłaniem.
8. Dyskusja: co jest fizyką (mapa ściskająca, gwiazda, linia kwadrupoli), co metaforą (lasery, koloidy), i dlaczego rzut 3D nie wystarcza.

Czasopisma do rozważenia (bez przesądzania): TMLR; Physical Review E / Machine Learning: Science and Technology (fizyka sieci);
warsztat ICLR/NeurIPS o dynamice uczenia. Wybór zależy od tego, czy akcent ma być na replikacji (TMLR) czy na analogiach (PRE/MLST).

## 6. Co trzeba policzyć przed pisaniem (minimalny zakres)

1. Płaszczyzna krok × podkrok dla seedu 3047 (46 checkpointów 0–400 ep) i 2–3 innych seedów z 24 istniejących historii; 10–20 klatek
   TEST z narzędziem ≥ 8 łat. Koszt: ok. 1,5 s na checkpoint i klatkę (CPU) po zapisaniu przejść; przejścia liczy się raz na GPU.
2. Jakobiany własne dla ~60 tokenów × 7 checkpointów × kilka klatek (ok. 2 min na 49 zestawów na RTX 3070).
3. Zero przestrzenne dla Fishera po stanach (maska narzędzia przesuwana po siatce, jak w R9).
4. Rozkład Dirichleta na człon sąsiadów i człon wariancji.
5. Test P2 na pełnym zbiorze TEST (wzbogacenie uwagi vs obrys).

## 7. Pliki

- Pomiary i wykresy: `experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29/` (RAPORT.md, F1–F9, WNIOSEK.txt, dane).
- Zasady i wcześniejsze liczby: `outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md`, `KURCZENIE_UWAGA_MLP.json`.
- Literatura: `.../Wyniki/Dane/literatura_undermind.md` (klucze Undermind, DOI).

## 8. Stan po replikacji na seedzie 3047 (2026-10-23; `experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41`)

46 checkpointów 0–400 ep, 10 klatek TEST z cystotomem po 20 łat (udział ≥ 0,25). Werdykt na progach zapisanych przed pomiarem: **2/7 PASS**
(P5, P6), 5 FAIL — z czego P2, P3, P4 o włos (0,886 vs 0,90; 1,203 vs 1,2; 0,036 vs 0,03), P1 i P7 to różnice czasu między seedami.

Co się powtórzyło (liczby = średnie po 10 klatkach):
- translacja na starcie (udział 0,999–1,000 w 192D; 573–590 źródeł) → gwiazda: 3 najwięksi odbiorcy uwagi w obrysie ∪ halo w 83–90 % par
  w blokach 3–6; udział tokenów narzędzia w pełnym jakobianie uwagi 0,56–0,77 (start 0,03); część barycentryczna 0,96–0,98 → 0,3–0,66;
- MLP 1 rozpychacz (×1,04 → ×1,97 w 22 ep → ×1,62 w 400 ep), **MLP 4 rośnie do końca** (×1,03 → ×1,21 (60 ep) → ×1,40 (100) → ×1,57 (400)),
  MLP 6 lekko kurczy od 100 ep (×0,98–0,99) i pcha do środka (cos −0,21);
- wskaźnik balistyczny 1,01 → 2,78 (k1000) → 1,02 (60 ep) → 0,90 (400 ep); cos kolejnych MLP 0,55 → 0,1, b5→6 ujemny (−0,13) w 400 ep;
- front Fishera: skok w m4 w 90 % par; Fisher m4 1,33 (1 ep) → 3,11 (60) → 3,66 (200) przy IoU płaskim od ~10 ep; blok 6 obniża;
- siodło w każdym podkroku (0,49–0,53), złożenie 0,48–0,56, nakładanie bliskie losowego (+0,01…+0,03, najwięcej w fazie spójnej 1–3 ep);
  przejście ściska coraz mocniej: log σ_max 0,87 → 2,85, log σ_min −0,96 → −6,5, Σ log|det| narzędzia −0,1 → −26,9 (400 ep), tła −18,8;
- uwaga na sąsiadach ≤ 0,018, wygładza MLP (Dirichlet −0,08…−0,20 vs −0,02…−0,03).

Co się różni: kolejność kolapsu bloków (3047: bloki 3, 1, 4 do 3 ep, blok 5 dopiero po 160 ep; s47: bloki 5 i 4 pierwsze); początek
kontrakcji objętości (3047: między 9 a 22 ep; s47: między 5 a 10 ep); MLP 2 staje się skrajnie nieliniowy (CV-R² 0,15 w 300–400 ep).
Wnioski do tez: T1, T2, T4 trzymają się; w T3 „kolejność faz” trzeba zapisać bez przypisywania bloków (kolaps uwagi w pierwszej epoce
w części bloków; spójna inflacja do ~3 ep; dekoherencja; front w bloku 4), a progi liczbowe przeskalować (wzbogacenie względem maksimum
1/udział narzędzia).

