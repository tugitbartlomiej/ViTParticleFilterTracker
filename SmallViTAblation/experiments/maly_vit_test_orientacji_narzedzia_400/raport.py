"""Czy400 źródeł jest gotowe, a status400 syntetyków pozostaje jawny?
Niespójny manifest, niepełne pliki lub nieprawidłowa siatka obalą gotowość.
"""
import json
from pathlib import Path
import cv2,numpy as np
from przygotuj import HERE,verify,js,SEED

r=json.loads((HERE/'WYNIKI.json').read_text());rows=json.loads((HERE/'manifest.json').read_text())
for a in rows:
    mask=cv2.imread(str(HERE/a['mask_480_path']),0)>0;yy,xx=np.nonzero(mask)
    a['bbox_mask_480']=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
    a['bbox_source_480']=[float(x*(480/a['width'] if j%2==0 else 320/a['height'])) for j,x in enumerate(a['bbox_source_native'])]
    a['tip_source_480']=[float(a['tip_source_native'][0]*480/a['width']),float(a['tip_source_native'][1]*320/a['height'])] if a['tip_source_native'] else None
js(HERE/'manifest.json',rows);assert verify()['all_train']
binrows='\n'.join(f'| {i*15}–{(i+1)*15} | {r["angle_counts"].get(str(i),0)} |' for i in range(12))
config=dict(seed=SEED,count=400,source_split='train',source_category='cystotome',
    source_image_policy='unmodified native JPEG bytes',native_size_wh=[720,480],model_input_wh=[480,320],
    grid_hw=[20,30],patch_wh=[16,16],token_index='row-major:30*row+column',
    mask_policy='fillPoly rounded source polygons native; INTER_AREA to480x320; binary>0',
    comparison_indices='same original mask token indices for both members of each pair',
    original_count=400,synthetic_count=0,synthetic_status='pending root model decision',
    prospective_model='LaMa candidate provided by root, not downloaded or executed',
    inpaint_mask='native tool mask; any proposed dilation separately stored; no implicit change',
    synthetic_output='PNG at native720x480, exact decoded-original pixels outside intervention mask',
    inference_preprocessing='to be frozen after model choice; no original JPEG mutation',
    budget='small local dataset, no training/forward/Eden/commit')
js(HERE/'CONFIG.json',config)
report=f'''# Przygotowanie400 rzeczywistych train do testu orientacji i par bez narzędzia

Zakończono {r['created']}. Gotowe400 oryginałów. Syntetyki0;400 par wPARY.json oczekuje na decyzję root o lokalnym modelu. Zbiór zajmuje około{r['bytes_total']/1e6:.1f}MB; naF pozostaje23,91GB. Nie przeprowadzono forwardów, treningu ani nowych edycji obrazu.

## Pytanie i granica wyniku

Czy istnieje400 unikalnych prawdziwych train z dostarczoną segmentacją narzędzia, reprezentujących różne naturalne kąty? Tak:400 klatek z116 filmów,33–34 w każdym z12 przedziałów15stopni. Wszystkie wybrane maski mają eigenvalue ratio>=3; niepewne osie0. Zakres osi0,157–179,612stopni, modulo180. Osie nie mają zwrotu, więc nie są pełną orientacją końcówki0..360. Kąt PCA maski nie mierzy pozy3D. Źródłowy tip zachowany osobno.

To naturalne różnice obrazu, tła, narzędzia, oświetlenia i pozy. Nie są kontrolowanymi obrotami jednego narzędzia wokół środka na stałym tle. Po doprecyzowaniu użytkownika kontrolowane rotacje nie należą do obecnego ramienia; celem są400 oryginałów i odpowiadające400 syntetyków bez narzędzia.

## Źródła i selekcja

Kanoniczny lokalny tracking pool_manifest.jsonl:10637 wierszytrain z instrument bbox,118 filmów. W archiwach3_Object_Tracking odczytano cystotome_seg, cystotome_bbox i cystotome_tip. Klasą jestcystotome, nie wszystkie instrumenty. Po kontroli maski:10277 poprawnych kandydatów,10264 z osią PCA ratio>=3. Odrzucono220 zbyt małych/pustych masek i140 niezgodności maski z bbox>2piksele. Nie pobierano żadnych klatek wal/test i nie zastępowano maski prostokątem.

Dokładne mapowanie:film→archiwum→annotation.csv→frame_number→filename. CSV ma6cyfr indeksu w nazwie, manifest8cyfr; połączenie jest przez liczbowy frame_number i źródłową nazwęCSV. OryginalneJPEG zapisano byte-for-byte z archiwum. seed{SEED}; warstwowanie po kątach i filmach. Wszystkie400 dobrano z minimalnym rozstawem50klatek w tym samym filmie; nie trzeba było go zmniejszać. FPS nie jest zweryfikowany, zatem nie przeliczamy odstępu na sekundy. Grupy filmu nie są potwierdzonymi pacjentami. train oznacza przynależność do puli treningowej, nie gwarancję, że każda klatka była wcześniej podana konkretnej sieci.

## Liczebność kąta osi

| Przedział stopni modulo180 | Liczba |
|---|---:|
{binrows}

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
'''
(HERE/'WYNIKI.md').write_text(report,encoding='utf-8')
(HERE/'README.md').write_text('''# 400 naturalnych train: orientacja narzędzia i pary bez narzędzia

Gotowe400 unikalnych prawdziwych train z rzeczywistymi maskami cystotome,116 filmów,12 koszyków kąta osi33–34 zdjęcia. Syntetyki0;400 par pending do decyzji o modelu. [Raport](WYNIKI.md), [plan](PLAN.md), [konfiguracja](CONFIG.json), [manifest](manifest.json), [pary](PARY.json), [plansza48](PLANSZA_48.jpg).

Dane źródłowe są kopiami natywnychJPEG bez zmiany pikseli. Maski natywne i480x320; TOKEN_MASKI.npz zawiera400x600 udziałów/tokenów. W negatywie badamy TE SAME tokeny maski oryginału. Naturalne kąty nie są kontrolowanymi rotacjami ani orientacją3D.

Odtwarzanie w nowym pustym katalogu tego eksperymentu:

```powershell
& F:\\Python\\python.exe przygotuj.py
& F:\\Python\\python.exe audyt_dawnych_negatywow.py
& F:\\Python\\python.exe raport.py
```

Bez treningu, forwardu, Eden jobs lub commitów. Nie uruchamiać przygotuj.py ponownie na istniejącychJPEG; kontrolę można wykonać przez import przygotuj i verify(). Źródła pozostają tylko do odczytu.
''',encoding='utf-8')
print('Report ready, verified400')
