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
