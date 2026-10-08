# Od sztucznego latentu do zdjęć uczących: wyniki (2026-10-08)

Plan: `PLAN.md`. Kod: `kod/eksperyment.py` (etapy 1, 2a, 2b, 2c, 2d, 3, 4 i kontrola losowa), `kod/analiza.py`.
Wyniki: `outputs/latent_cel_do_obrazow/` (ETAP*.json, KONTROLA_LOSOWA.json, RYSUNEK.png, obrazy SD w `sd/`, wagi
ramion `model_ramie*_k1000.pt`). Czas: około 30 minut na RTX 3070 (etapy 1-3), plus 11 minut (etapy 4, 2d i kontrola).
Checkpoint k1000 gęstej historii (sha 3df42a6a), 24 obrazy manifestu J128 z 24 filmów. Przy k1000 te cele miały
IoU najpewniejszej ramki równe 0 (najsłabsze obrazy każdego filmu). Żaden z nich nie był w bazie uczenia.

## Najkrócej

1. Sztuczny latent osiąga cele detekcji. W 300 krokach IoU rośnie z 0 do 0,985, a pewność do 0,74 (średnio na 24
   obrazach). Wymaga to jednak korekty o normie około 8 razy większej niż sam latent obrazu (2000 wobec 240).
   Z normą 0,1 albo 0,3 latentu IoU zostaje 0.
2. Takiego latentu nie realizuje żadne prawdziwe zdjęcie. SD z latentu po korekcie rysuje rozmytą teksturę, w której
   detektor nie znajduje ramki celu (0 z 72 obrazów z IoU > 0,5). Z latentu bez korekty SD rysuje rozpoznawalne oko.
   Klatki znalezione przez konwerter są od celu tak samo daleko jak obraz wyjściowy.
3. Kierunek korekty da się jednak częściowo uzyskać prawdziwym uczeniem. Po 1000 krokach z 96 klatkami z konwertera
   w bazie latent celów przesuwa się w stronę korekty: kosinus wynosi 0,088. Dla 96 losowych klatek wynosi 0,035, a
   dla uczenia na samych celach 0,093. Klatki z konwertera wygrywają z losowymi w 21 na 24 celach.
4. Detekcji celów i walidacji to nie poprawia. IoU celów przy k1000: 0,195 (konwerter) wobec 0,169 (losowe) i 0,171
   (oryginał). mAP wal3000: 0,307 wobec 0,338 i 0,343. Konwerter zwraca klatki tylko z 7 filmów, więc baza traci
   różnorodność. Same cele w bazie podnoszą IoU celów do 0,367, a mAP walidacji bez zmian (0,348).

## Etap 1: korekta oracle i jej punkty pośrednie

Cel „detekcja” z zakładki Detekcja (pewność środka do 0,95, tłumienie tła, ranking, odległości ramki, GIoU). Tryb
„zakresy z checkpointu”, 768 kanałów, 600 tokenów, Adam 0,05.

| krok korekty | IoU najpewniejszej | pewność | ||Δz|| |
|---|---|---|---|
| 0 | 0,000 | 0,59 | 0 |
| 100 | 0,966 | 0,50 | 1497 |
| 200 | 0,981 | 0,68 | 1868 |
| 300 | 0,985 | 0,74 | 2019 |

Najpierw przesuwa się ramka: IoU dochodzi do 0,97 już po 100 krokach. Pewność rośnie wolniej. Część niewidoczna dla
fc2 wynosi 49 % (przy J128 było 72 %).

## Etap 2: zdjęcia dla punktów pośrednich

- Konwerter: z_w -> cechy DINO -> 20 najbliższych z 34 994 klatek puli. Pula ma 117 filmów, wszystkie z podziału
  TRAIN, i nie dzieli żadnego filmu z 25 filmami walidacji. Wynik:
  - najlepsza klatka po przeliczeniu prawdziwym latentem ma odległość od z_w równą 0,985-1,007 odległości obrazu celu;
  - 0 z 72 najlepszych klatek pochodzi z filmu celu;
  - dominuje jeden film (TR_0081);
  - kosinus ruchu latentu klatki do korekty wynosi średnio 0,108; dla losowych klatek 0,052, najlepsza z 20 losowych 0,124.
- SD z z_w (30 kroków DDIM): pewność detekcji 0,28-0,31, a z latentu bez korekty 0,42. Odległość latentu obrazu SD
  od z_w wynosi 1,003-1,004 odległości obrazu celu. Latent SD nie zbliża się do celu.
- Pchnięcie jednego kroku uczenia (990) na latent celu, 24 cele:

| dołożony obraz | cos(pchnięcie, korekta) | zmiana straty celu |
|---|---|---|
| sam obraz celu z etykietą | −0,061 | −0,184 |
| klatka z konwertera | −0,101 | −0,0052 |
| klatka z drogi SD -> konwerter | −0,088 | −0,0090 |
| obraz SD z etykietą celu | −0,067 | −0,0086 |
| losowa klatka puli | −0,041 | −0,0015 |

  - Żaden obraz nie pcha latentu w jednym kroku w stronę korekty, nawet sam cel.
  - Znalezione obrazy obniżają stratę celu 3-6 razy mocniej niż losowe: SD -> konwerter w 22 na 24 celach (p = 1e-4),
    obraz SD w 18 na 24 (p = 0,001), konwerter w 18 na 24 (p = 0,07).
  - W rozkładzie względem fc2 (etap 2d) kosinus zmiany wyjścia końcówki W2 S z W2 δ też jest ujemny
    (konwerter −0,155, losowe −0,077).

## Etap 3: uczenie z podmianą 96 zdjęć bazy

Przepis gęstej historii (Uczen(47), default_rng(47), 1000 kroków) w trybie deterministycznym, te same 96 pozycji bazy.

| ramię | IoU celów k500 / k1000 | średnie IoU 600-1000 | AP50 celów k1000 | mAP wal3000 k500 / k1000 | cos ruchu celów do korekty | filmy |
|---|---|---|---|---|---|---|
| A oryginał (powtórka) | 0,201 / 0,171 | 0,182 | 0,044 | 0,192 / 0,343 | - | - |
| E same cele (24 pozycje) | 0,308 / 0,367 | 0,351 | 0,138 | 0,236 / 0,348 | +0,093 | 24 cele |
| B konwerter | 0,169 / 0,195 | 0,199 | 0,090 | 0,187 / 0,307 | +0,088 | 7 |
| C losowe klatki | 0,250 / 0,169 | 0,180 | 0,081 | 0,196 / 0,338 | +0,035 | 72 |
| D SD -> konwerter | 0,189 / 0,187 | 0,188 | 0,112 | 0,201 / 0,290 | +0,015 | 26 |

- Oryginalny (niedeterministyczny) bieg miał mAP 0,242 / 0,308 przy krokach 500 / 1000. Powtórka deterministyczna
  daje 0,192 / 0,343. Pojedynczy bieg zmienia się o kilka punktów od samego trybu obliczeń. Różnice między ramionami
  w walidacji (0,29-0,35) są tego samego rzędu.
- Cele miały IoU 0 w oryginale, bo wybrałem je jako najgorsze. W powtórce A mają 0,17 (regresja do średniej). Dlatego
  porównanie dotyczy ramienia A, a nie zapisanych wag.
- Kosinus ruchu celów do korekty: B > C w 21 na 24 celach (Wilcoxon p = 4e-6), E > C w 22 na 24 (p = 2e-5). B jest
  dodatni na wszystkich 24 celach, C na 20, D na 14.

## Etap 4: mała korekta

Przy ||δ|| <= 0,1 albo 0,3 ||z|| (8 celów) korekta nie poprawia IoU (0 -> 0), a pewność spada. Obrazy SD z takiego
latentu trafiają ramkę celu tylko czasem (IoU 0,19-0,25 dla 3 z 8 przy 0,3). Kosinus ich latentu do korekty wynosi
0,04-0,11. Na tym checkpoincie i dla tych celów skuteczna korekta latentu nie jest mała.

## Czego to nie dowodzi

- Jedno ziarno uczenia na ramię i 24 cele. Różnice w mAP walidacji mieszczą się w zmienności między trybami obliczeń
  jednego biegu. Do wniosku o mAP potrzebne są co najmniej 3 ziarna na ramię.
- Kosinus ruchu latentu do korekty mierzy kierunek, nie jakość detekcji. Klatki z konwertera przesuwają latent
  w stronę korekty, ale detekcji celów prawie nie poprawiają.
- Konwerter i SD są uczone na latencie k1000. Inny checkpoint wymaga innego modelu warunkowego.
- Pula ma tylko klatki z ramką narzędzia z podziału TRAIN. Nie sprawdzono klatek bez narzędzia ani innych puli.

## Etap 3b: dodatkowe ziarno 48 (przerwane na prośbę użytkownika)

Ziarno 48, krok 1000 (IoU celów / AP50 celów / mAP wal3000): A 0,118 / 0,017 / 0,326; E 0,467 / 0,293 / 0,288;
B 0,152 / 0,027 / 0,325. Ramiona C i ziarno 49 nie zostały policzone. Same cele znowu wyraźnie poprawiają cele.
Konwerter znowu nie (0,152 wobec 0,118, przy jednym ziarnie w granicach szumu). Walidacja dla E spada z 0,326 do 0,288.

## Etapy 5-8: SD prowadzone i naturalny latent (2026-10-08, po uwagach użytkownika)

Kod: `kod/sd_prowadzone.py` (etapy 5, 7, 9), `kod/korekta_naturalna.py` (etap 6), `kod/korekta_zakresy_naturalne.py`
(etap 8), tabele `kod/podsumuj_sd.py`. Obrazy w `outputs/latent_cel_do_obrazow/sd_prowadzone/`.
Metoda SD jak u użytkownika (`maly_vit_sd_prowadzenie_tozsamoscia/kod/prowadzenie.py`, `vit_sd_zgodnosc90`):
- start od prawdziwego obrazu (img2img, siła 0,5, 30 kroków DDIM);
- w krokach 15-25: przewidywany x̂0 -> VAE -> ViT -> energia, 2 kroki gradientu po x̂0 (eta 0,15 · σ_t);
- 2 rundy: wynik rundy jest startem następnej.
Miara: α = część długości korekty zrealizowana przez latent obrazu SD. Detekcja oceniana etykietą celu
(img2img zachowuje układ). Próba: 4 cele, jedno ziarno.

Naturalność tokenu mierzę dwiema liczbami: „poza” to odległość od podprzestrzeni PCA-64 prawdziwych tokenów,
„maha” to odległość Mahalanobisa w tej podprzestrzeni. Prawdziwe tokeny: poza 1,7, maha 7,9.
Szerokość przedziału kanału: z checkpointu średnio 8,13, w danych 1,56 (min-max), 1,16 (0,1-99,9 %), 0,87 (1-99 %).

| korekta (8 celów, cel detekcja) | IoU | IoU >= 0,5 | pewność | poza | maha | ||Δz|| | zmiana W2·z |
|---|---|---|---|---|---|---|---|
| zakresy checkpointu (etap 1) | 0,985 | 8/8 | 0,74 | 86 | 122 | 2000 | 8,7 |
| zakres danych + margines 1,0 szerokości | 0,982 | 8/8 | 0,67 | 49 | 58 | 1406 | 7,1 |
| zakres danych + margines 0,5 | 0,951 | 8/8 | 0,46 | 37 | 49 | 1109 | 5,6 |
| zakres danych + margines 0,25 | 0,820 | 8/8 | 0,36 | 28 | 43 | 906 | 4,6 |
| zakres danych min-max | 0,652 | 5/8 | 0,25 | 20 | 37 | 691 | 3,4 |
| percentyle 1-99 | 0,453 | 4/8 | 0,19 | 11 | 21 | 419 | 2,2 |
| tylko PCA-256 prawdziwych tokenów, λ 0,1 | 0,677 | 7/8 | 0,25 | 2,0 | 14 | 278 | 1,4 |
| tylko PCA-64, λ 0 | 0,529 | 4/8 | 0,23 | 1,8 | 24 | 339 | 1,6 |

Wnioski z korekt:
- Detekcja rośnie z tym, jak daleko latent wychodzi poza naturalne wartości. Lokalizację (IoU) da się uzyskać
  w naturalnym rozkładzie tokenów (PCA-256: 7/8 celów, poza 2,0 jak prawdziwe obrazy). Wysoka pewność wymaga
  wyjścia poza zakres danych o około szerokość tego zakresu.
- Ograniczenie każdego kanału osobno (pudło) nie wystarcza do naturalności. Współrzędne są naturalne, ale ich
  kombinacje nie (poza 11-49). Ograniczenie w podprzestrzeni naturalnych kierunków daje naturalne tokeny przy
  podobnym IoU.
- Normalizacja w jądrze fc2 (etap 5: ta sama detekcja, bo W2 z bez zmian) obniża nienaturalność tylko o około 22 %.
  Sama zmiana widoczna dla detektora (W2·z) jest 6,7-8,8 razy większa niż W2·z obrazu.

| SD (4 cele, runda 2) | cos do celu | α | IoU | pewność |
|---|---|---|---|---|
| obraz wyjściowy | - | - | 0,000 | 0,63 |
| sztuczny cel: prowadzenie pełnym latentem (metoda użytkownika) | +0,140 | 0,005 | 0,000 | 0,63 |
| sztuczny cel: bez prowadzenia (kontrola) | −0,029 | −0,001 | 0,000 | 0,55 |
| sztuczny cel: prowadzenie częścią widoczną dla fc2 | +0,138 | 0,005 | 0,047 | 0,61 |
| sztuczny cel znormalizowany w jądrze fc2, prowadzenie fc2 | +0,161 | 0,012 | 0,051 | 0,75 |
| naturalny cel (PCA-256), warunek bez prowadzenia | +0,032 | 0,008 | 0,000 | 0,50 |
| naturalny cel, prowadzenie pełnym latentem | +0,260 | 0,092 | 0,077 | 0,60 |
| naturalny cel, prowadzenie częścią widoczną dla fc2 | +0,281 | 0,112 | 0,145 | 0,64 |

Wnioski z SD:
- Prowadzenie użytkownika działa kierunkowo. Latent obrazu SD idzie w stronę celu (+0,14 wobec −0,03 bez
  prowadzenia), a obraz zostaje naturalny. Przy sztucznym celu realizuje jednak tylko 0,5 % jego długości, bo cel
  leży 8 razy dalej niż naturalna zmienność.
- Przy naturalnym celu SD realizuje 9-11 % korekty, 20 razy więcej. Pojawia się detekcja: IoU 0,145 średnio,
  cel 11 ma 0,41, cel 369 0,17-0,19. Obrazy wyglądają jak prawdziwe klatki z drobnymi zmianami.
- Prowadzenie samą częścią widoczną dla fc2 jest trochę lepsze od pełnego latentu (α 0,112 wobec 0,092, IoU 0,145
  wobec 0,077). Nie trzeba odtwarzać części, której detektor nie widzi.
- Sam warunek SD bez prowadzenia prawie nie przesuwa latentu (α 0,008): kierunek daje prowadzenie, nie warunek.

Czego to nie dowodzi: 4 cele i 2 rundy, jedno ziarno SD. Obrazy SD nie trafiły jeszcze do uczenia. IoU obrazów SD
oceniane etykietą obrazu wyjściowego (img2img zachowuje układ, ale narzędzie mogło się przesunąć).

### Etap 9: SD z celu w zakresie danych z marginesem 0,25

Warunek i cel SD to latent z korekty w pudle [min − 0,25 szer, max + 0,25 szer] (IoU 0,82 w korekcie, poza 28).
Runda 2: z prowadzeniem pełnym latentem cos +0,194, α 0,033, IoU 0,032; z prowadzeniem fc2 cos +0,190, α 0,033,
IoU 0,032; sam warunek cos +0,062, α 0,010. Obraz zmienia latent mocniej (0,62 normy wobec 0,44 przy celu
naturalnym), ale realizuje 3 % celu zamiast 11 %. Im bliżej rozkładu prawdziwych tokenów leży cel, tym większą jego
część SD potrafi narysować.

## Najkrócej po etapach 5-9

1. Twoje prowadzenie SD działa kierunkowo i daje naturalne obrazy, ale przy sztucznym celu (8 razy poza naturalną
   zmiennością) realizuje 0,5 % korekty.
2. Detekcja w latencie rośnie, gdy latent wychodzi poza naturalne wartości. Lokalizację da się jednak uzyskać
   w naturalnym rozkładzie tokenów: korekta w podprzestrzeni PCA-256 prawdziwych tokenów daje IoU 0,68 (7 na 8 celów),
   a tokeny są tak naturalne jak w prawdziwych obrazach. Pewność wymaga wyjścia poza zakres danych.
3. Taki naturalny cel SD realizuje w 11 %, a dla dwóch z czterech celów pojawia się detekcja (IoU 0,17-0,41).
   Najlepiej działa prowadzenie samą częścią widoczną dla fc2.
4. Następny krok: więcej rund i celów dla celu naturalnego, potem obrazy SD (z etykietą obrazu wyjściowego) jako
   ramię uczenia w porównaniu z oryginałem, deterministycznie, co najmniej 3 ziarna.

## Etap 10: SD bezpośrednio z umiarkowanego sztucznego latentu i 10 kroków do oryginału (zlecenie użytkownika)

Kod `kod/sd_interpolacja.py`, obrazy `outputs/latent_cel_do_obrazow/sd_interpolacja/` (paski `PASEK_m<cel>_<wariant>.jpg`:
od lewej 100 % przyrostu, potem co 10 % do oryginału). Dla każdego tokenu przyrost δ = z_sztuczny − z_oryginał,
z_k = z_sztuczny − (k/10) δ. SD od czystego szumu (DDIM 40, to samo ziarno dla wszystkich k), bez img2img i prowadzenia.
Sztuczny latent z etapu 8: każdy kanał w zakresie danych (8 celów). Wariant z marginesem 0,25 przerwany po 2 celach
(użytkownik: wystarczy zdjęć).

| przyrost | IoU latentu (bez SD) | pewność latentu | IoU obrazu SD | pewność obrazu SD | cos(latent SD − oryginał, δ) |
|---|---|---|---|---|---|
| 100 % | 0,65 | 0,25 | 0,060 | 0,31 | +0,084 |
| 80 % | 0,52 | 0,22 | 0,048 | 0,34 | +0,106 |
| 60 % | 0,41 | 0,21 | 0,023 | 0,35 | +0,148 |
| 50 % | 0,34 | 0,22 | 0,023 | 0,39 | +0,156 |
| 40 % | 0,10 | 0,26 | 0,027 | 0,37 | +0,155 |
| 20 % | 0,00 | 0,43 | 0,049 | 0,54 | +0,151 |
| 0 % (oryginał) | 0,00 | 0,60 | 0,030 | 0,49 | +0,092 |

- Detekcja w samym latencie znika szybko: przy 40 % przyrostu IoU spada do 0,10, przy 20 % do 0.
  Potrzebny jest prawie cały przyrost.
- SD z dużym przyrostem (100-70 %) rysuje rozmytą teksturę bez struktury oka, mimo że każdy kanał jest w naturalnym
  zakresie. Scena wraca przy około 50-40 %, a przy 10-0 % obraz jest rozpoznawalny.
- Obrazy SD nie przenoszą detekcji z latentu (IoU 0,02-0,09 na wszystkich krokach). Kierunek ich latentu jest lekko
  zgodny z przyrostem (cos +0,08 do +0,16), najbardziej przy 60-20 %.
- SD od szumu nie odtwarza dokładnie zdjęcia nawet z oryginalnego latentu: latent obrazu jest o 0,7 normy od
  oryginału. Precyzyjne porównanie wymaga img2img od zdjęcia (jak w etapach 5-9).

## Etap 11: obrazy SD jako zapytania do wyszukania prawdziwych zdjęć (DINO)

Kod `kod/sd_szukanie_dino.py`, paski `outputs/latent_cel_do_obrazow/sd_szukanie/SZUKANIE_m<cel>.jpg` (górny rząd:
zdjęcie celu i 11 obrazów SD od 100 % przyrostu do oryginału; dolny: najbliższa klatka puli 35 000 TRAIN, DINOv2
CLS, te same przekształcenia co pula). 8 celów.

| przyrost | cos najbliższej klatki | najbliższa z filmu celu | wspólne z top 5 dla samego zdjęcia | cos(SD, zdjęcie celu) w DINO |
|---|---|---|---|---|
| 100 % | 0,36 | 0/8 | 0 | 0,16 |
| 70 % | 0,43 | 0/8 | 0 | 0,25 |
| 50 % | 0,53 | 0/8 | 0 | 0,31 |
| 30 % | 0,52 | 0/8 | 0 | 0,27 |
| 10 % | 0,51 | 0/8 | 0 | 0,29 |
| 0 % | 0,68 | 0/8 | 0 | 0,48 |

- Każdy krok przyrostu prowadzi do innych prawdziwych klatek: 4-7 różnych najbliższych klatek na cel, żadna nie
  pokrywa się z wynikami dla samego zdjęcia i żadna nie jest z filmu celu.
- Przy 100-70 % obrazy SD są teksturą, więc dopasowanie jest słabe (cos 0,36-0,43) i zwraca przypadkowe sceny.
  Od około 50 % dopasowanie rośnie (cos 0,5), a znalezione klatki mają podobny ogólny wygląd.
- Do sprawdzenia: czy znalezione klatki przesuwają latent celu w stronę przyrostu (pchnięcie jak w etapie 2c) i czy
  dołożone do uczenia poprawiają cel, z kontrolą losowych klatek.

## Etap 12: pchnięcie zdjęć znalezionych przez DINO z kontrolą losowych

Kod `kod/pchniecie_znalezionych.py`, wynik `outputs/latent_cel_do_obrazow/ETAP12_PCHNIECIE_ZNALEZIONYCH.json`.
Krok 990 -> 991, sondy = 8 celów. Na cel: 4-7 znalezionych zdjęć (najbliższe dla 11 obrazów SD, bez powtórzeń),
3 × tyle samo losowych z tej samej puli, sam cel z etykietą. δ = przyrost sztucznego latentu (etap 8, minmax).

| dołożone zdjęcie | cos(pchnięcie, δ) | cos z częścią δ widoczną dla fc2 | cos(W2 S, W2 δ) | zmiana straty celu | długość pchnięcia |
|---|---|---|---|---|---|
| sam cel z etykietą | −0,010 | −0,010 | −0,066 | −0,191 | 27,8 |
| znalezione z SD (DINO) | −0,050 | −0,044 | −0,094 | −0,0061 | 3,6 |
| losowe z puli | −0,046 | −0,033 | −0,082 | −0,0001 | 1,9 |

- Kierunek: znalezione zdjęcia nie pchają latentu w stronę przyrostu lepiej niż losowe (lepsze na 3 z 8 celów,
  p = 0,38; część widoczna dla fc2: 1 z 8). W jednym kroku nic nie pcha w stronę przyrostu, nawet sam cel.
- Strata celu: znalezione obniżają ją mocniej niż losowe (−0,0061 wobec −0,0001, lepsze na 6 z 8 celów, p = 0,15,
  nieistotne przy 8 celach) i pchają latent celu prawie 2 razy mocniej (3,6 wobec 1,9).
- Po kroku przyrostu: zdjęcia znalezione dla obrazów SD z 20-10 % przyrostu mają lekko dodatni kosinus (+0,02 do
  +0,05), dla 100-80 % ujemny (−0,11 do −0,15); po 1-8 zdjęć na krok, więc tylko wskazówka.

## Etapy 8, 10 i 11 od nowa na zgodnej parze ViT K8000 i SD K8000 (zlecenie użytkownika 2026-10-08)

Powód: użytkownik wskazał SD uczone na K8000 (`assets/SD_ViT_K8000_A_control_K10500`, teacher
`assets/MalyViT_K8000_s47_losowe`). Sprawdzenie: dotychczasowe SD (`assets/StableDiffusionModel`) było uczone na
`MODEL_VIT_K1000.pt` (sha 3df42a6a), identycznym z k1000 gęstej historii, więc etapy 10-11 były spójne; nowe liczenie
przenosi wszystko na K8000. Kod `kod/k8000_od_nowa.py` (Python 3.13, bo tylko tam jest diffusers), wyniki
`outputs/latent_cel_do_obrazow/k8000/` (ETAP8_K8000.json, ETAP10_K8000.json, ETAP11_K8000.json, `sd_interpolacja/`,
`sd_szukanie/`, PODGLAD_SD_K8000.jpg).

- Cele: 8 obrazów manifestu najtrudniejszych dla K8000 (181, 242, 102, 241, 224, 270, 369, 216; IoU 0); 4 wspólne
  z celami k1000. K8000 na manifeście: IoU średnio 0,61 (k1000: 0,47).
- Etap 8 (minmax danych K8000, szerokość kanału 2,2): IoU latentu 0 -> 0,86 (7 z 8 powyżej 0,5), pewność 0,68;
  ||Δz|| = 2,2 ||z||; odległość od podprzestrzeni naturalnych tokenów 18 (prawdziwe tokeny 2,6).
- Etap 10 (SD K8000 od szumu, DDIM 50, ziarno 0; cele 1-4: 11 kroków, cele 5-8: 6 kroków na prośbę użytkownika):
  latent obrazu SD z oryginalnego latentu ma cos 0,72 do warunku (jak walidacja modelu). Przy 100 % przyrostu cos do
  warunku 0,26, cos do przyrostu 0,11, odtworzony ułamek 4,1 %; ale już obraz z oryginału ma cos do przyrostu 0,06
  i ułamek 1,9 %, więc netto sztuczny przyrost dokłada ok. 2 punkty procentowe. Obrazy 100-70 % są prawie takie same
  dla wszystkich celów (różowa tkanka z poziomymi kreskami), oko wraca przy 50-30 %.
- Etap 11 (DINO, pula 35 000 TRAIN): najbliższe zdjęcie to dla 41 z 68 zapytań ta sama klatka 0010/2879, a łącznie
  tylko 8 różnych klatek na 68 zapytań; zero wspólnych z top-5 zapytania samym zdjęciem celu, zero z filmu celu.
  Wyszukiwanie oddaje styl obrazów SD K8000, nie kierunek przyrostu.

Wniosek: zgodna para K8000 przenosi kierunek sztucznego latentu do obrazu kilka razy lepiej niż para k1000
(4 % wobec 0,5 %), ale przy celu 7 razy dalej od naturalnych tokenów niż prawdziwe obrazy SD rysuje jeden wspólny
obraz zastępczy i wyszukiwanie zwraca prawie zawsze tę samą klatkę. Pchnięcia (etap 12) nie powtórzono: nie ma
lokalnie stanu AdamW uczenia K8000.
