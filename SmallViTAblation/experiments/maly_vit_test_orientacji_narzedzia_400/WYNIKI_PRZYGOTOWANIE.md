# Przygotowanie400 rzeczywistych train do testu orientacji i par bez narzędzia

Zakończono 2026-10-01T16:45:35.250704+02:00. Gotowe400 oryginałów. Syntetyki0;400 par wPARY.json oczekuje na decyzję root o lokalnym modelu. Zbiór zajmuje około44.4MB; naF pozostaje23,91GB. Nie przeprowadzono forwardów, treningu ani nowych edycji obrazu.

## Pytanie i granica wyniku

Czy istnieje400 unikalnych prawdziwych train z dostarczoną segmentacją narzędzia, reprezentujących różne naturalne kąty? Tak:400 klatek z116 filmów,33–34 w każdym z12 przedziałów15stopni. Wszystkie wybrane maski mają eigenvalue ratio>=3; niepewne osie0. Zakres osi0,157–179,612stopni, modulo180. Osie nie mają zwrotu, więc nie są pełną orientacją końcówki0..360. Kąt PCA maski nie mierzy pozy3D. Źródłowy tip zachowany osobno.

To naturalne różnice obrazu, tła, narzędzia, oświetlenia i pozy. Nie są kontrolowanymi obrotami jednego narzędzia wokół środka na stałym tle. Po doprecyzowaniu użytkownika kontrolowane rotacje nie należą do obecnego ramienia; celem są400 oryginałów i odpowiadające400 syntetyków bez narzędzia.

## Źródła i selekcja

Kanoniczny lokalny tracking pool_manifest.jsonl:10637 wierszytrain z instrument bbox,118 filmów. W archiwach3_Object_Tracking odczytano cystotome_seg, cystotome_bbox i cystotome_tip. Klasą jestcystotome, nie wszystkie instrumenty. Po kontroli maski:10277 poprawnych kandydatów,10264 z osią PCA ratio>=3. Odrzucono220 zbyt małych/pustych masek i140 niezgodności maski z bbox>2piksele. Nie pobierano żadnych klatek wal/test i nie zastępowano maski prostokątem.

Dokładne mapowanie:film→archiwum→annotation.csv→frame_number→filename. CSV ma6cyfr indeksu w nazwie, manifest8cyfr; połączenie jest przez liczbowy frame_number i źródłową nazwęCSV. OryginalneJPEG zapisano byte-for-byte z archiwum. seed20261001; warstwowanie po kątach i filmach. Wszystkie400 dobrano z minimalnym rozstawem50klatek w tym samym filmie; nie trzeba było go zmniejszać. FPS nie jest zweryfikowany, zatem nie przeliczamy odstępu na sekundy. Grupy filmu nie są potwierdzonymi pacjentami. train oznacza przynależność do puli treningowej, nie gwarancję, że każda klatka była wcześniej podana konkretnej sieci.

## Liczebność kąta osi

| Przedział stopni modulo180 | Liczba |
|---|---:|
| 0–15 | 34 |
| 15–30 | 34 |
| 30–45 | 34 |
| 45–60 | 33 |
| 60–75 | 33 |
| 75–90 | 33 |
| 90–105 | 33 |
| 105–120 | 33 |
| 120–135 | 33 |
| 135–150 | 33 |
| 150–165 | 33 |
| 165–180 | 34 |

## Obrazy, maski, współrzędne i cząsteczki

Obrazy natywne720x480 i natywne maskiPNG. COCO zawiera źródłowe polygony, dokładny rasterbbox i source_bbox osobno. Manifest zachowuje film/frame/train/oryginalneźródło/CSVhash/IMAGEhash/maski/bbox/tip. Kąty wyznaczone z całej binarnej maski natywnej. Ośx w prawo, ośy w dół; atan2(e_y,e_x)mod180. Nie użyto wyniku modelu do wyboru.

Referencja480x320: maskaarea=INTER_AREA, binary=area>0. TOKEN_MASKI.npz ma400x600: coverage binary i coveragearea, tool>0, tool25>=0.25 i background po dylatacji17x17 maski480. grid20x30, patch16x16, token=30*row+col. binarycoverage jest konserwatywne i różni się od area coverage. Obraz można później przeskalować do480x320 identycznym preprocessingiem; nie zapisano transformacji w samych oryginałach.

W parze oryginał/syntetyk cząsteczka z_i ma ten sam indeks tokenu wspólnej siatki oraz wspólną bazę768 współrzędnych. Indeksy wybiera maska ORYGINAŁU i pozostają także w negatywie bez narzędzia. Nie usuwamy tych tokenów, nie zerujemy automatycznie makrowektorów/współrzędnych. Pooling/LDA są odrębnymi obserwablami. Brak nowych latentów na tym etapie.

## Syntetyki: stan i plan

PARY.json:400 par, każda wskazuje istniejący oryginał, oryginalną maskę i planowany PNGbez narzędzia; statuspending_model_decision. Katalogobrazy_bez_narzedzia jest pusty. Root bada LaMa naHuggingFace; nic nie zainstalowano, nie pobrano modelu i nie wykonano400 edycji.

Po decyzji zamrozić hashmodelu/version/padding/resize/seed/dilation. Na natywnej klatce zachować dokładnie zdekodowane piksele oryginału poza jawną maską interwencji. Maska interwencji i maska badanych tokenów są osobnymi rzeczami. Tło pod narzędziem będzie rekonstrukcją, nie znanym prawdziwym tłem. QC musi kontrolować pozostawione końcówki, odbicia/cienie, szwy oraz uszkodzenie struktur oka. Kontrola patchy tła i artefaktów edycji jest konieczna przed interpretacją różnicy jako swoistości narzędzia. Zamrożony model eval, te same600 pozycje i ten sam preprocessing obu obrazów; bez augmentacji mieszających korespondencję.

Odczytano12 dawnych paroriginal/empty_black bez żadnych zmian. Wszystkie tocrop256 i wszystkie zmieniają100procent pikseli poza maską oryginału. Nie są czystym inpaintingiem narzędzia na zachowanym tle i nie mogą zastąpić planowanych400negatywów. AUDYT_DAWNYCH_NEGATYWOW.json zapisuje pomiar; brak nowego forwardu.

## Weryfikacja i artefakty

Exactly400 unikalnychframe_id,400 unikalnychJPEGhashów i400 unikalnychhashów pikseli. Wszystkie mają splittrain wedługźródła, natywne dimensions zgodne i niepustą rasteryzację dokładnie zapisanych polygonów. IMAGE/mask hashzgodne, siatka400x600 skończona w[0,1], tool/background rozłączne. Syntetyki0, stan parjawny. Kontrola wykonana ponownie po dodaniu bbox480/tip480.

- obrazy_oryginalne/:400 JPEG.
- maski_natywne/:400 maskPNG720x480.
- maski_480x320/:400 maskPNG referencji sieci.
- manifest.json, annotations.coco.json, PARY.json, CONFIG.json, TOKEN_MASKI.npz.
- AUDYT_ZRODEL.json, AUDYT_DAWNYCH_NEGATYWOW.json, WYNIKI.json.
- PLANSZA_48.jpg:48 miniatur jako wizualizacja danych, nie zmodyfikowane źródła.
- przygotuj.py, raport.py: odtwarzalne przygotowanie i kontrola; przygotuj.py odmawia nadpisania istniejącegoJPEG.

![Podgląd48 klatek](PLANSZA_48.jpg)
