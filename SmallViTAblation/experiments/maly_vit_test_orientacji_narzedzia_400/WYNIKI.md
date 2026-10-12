# 400 par: prawdziwe train i lokalna LaMa bez maskowanego narzędzia

Ukończono: 2026-10-01T17:25:31.497890+02:00. Gotowe **800 zdjęć: 400 rzeczywistych train oraz 400 odpowiadających syntetyków**. Sprawdzono wszystkie pary po odczycie zapisanych PNG. Łącznie zero zmienionych pikseli poza zadeklarowanymi maskami edycji. Zbiór zajmuje około 241.0 MB, a model 208 MB znajduje się w cache HuggingFace.

## Źródła i orientacje

Wybrano 400 unikalnych klatek ze 116 filmów, po 33–34 zdjęcia w każdym z 12 przedziałów kąta osi po 15 stopni. Kąt PCA maski jest modulo 180, w zakresie 0,157–179,612 stopni. To oś bez zwrotu, odrębna od orientacji końcówki i pozy 3D. Wszystkie wybrane maski mają stosunek wartości własnych co najmniej 3. Rozstaw wybranych zdjęć tego samego filmu wynosi co najmniej 50 klatek; FPS nie jest potwierdzony. Ziarno wyboru: 20261001.

Źródło train: istniejący tracking pool_manifest.jsonl. Obrazy, cystotome_seg, bbox i tip pochodzą z archiwów Cataract-LMM/3_Object_Tracking. Po audycie dostępnych było 10 277 poprawnych kandydatów spośród 10 637 oznaczonych train. Dokładne powiązanie po filmie i frame_number CSV; nazwę obrazu bierze się z CSV. JPEG w rzeczywistych wymiarach źródłowych skopiowano byte-for-byte. Polygony, źródłowy bbox, bbox rasteryzacji, tip, kąty i checksums są w manifeście/COCO.

To naturalne kąty na różnych tłach, nie kontrolowane rotacje jednego narzędzia. Zbiór należy do train i służy testowi reprezentacji. Nie jest niezależnym heldoutem detektora ani dowodem znajomości tych klatek przez konkretny checkpoint. Filmy nie są potwierdzonymi identyfikatorami pacjentów. [Raport przygotowania](WYNIKI_PRZYGOTOWANIE.md) zachowuje wcześniejsze liczby; jego status pending został zastąpiony obecnym ukończonym etapem.

## Model, cache i wykonanie

[edgetools/lama](https://huggingface.co/edgetools/lama) jest mirror ONNX modelu [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX), wywodzącego się z [advimman/lama](https://github.com/advimman/lama). Sprawdzono model card i licencję Apache 2.0; kopie tekstów zachowano w model_lama/. Wagi pobrano jeden raz, następnie przeniesiono do standardowego cache zgodnie z doprecyzowaniem użytkownika. hf_hub_download zwraca ścieżkę cache; w eksperymencie nie ma kopii ONNX. HF_HOME ustawiono tylko w procesie.

- Revision: `d3f22ee032d32d9a68ac06dd3da40aba1f831f1d`.
- SHA256: `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`.
- Cache: `F:\HuggingFace\hub\models--edgetools--lama\snapshots\d3f22ee032d32d9a68ac06dd3da40aba1f831f1d\lama_fp32.onnx`.

Rzeczywiste IO sprawdzono: float32 image (batch,3,512,512), mask (batch,1,512,512), output (batch,3,512,512). Batch jest dynamiczny, rozmiar 512 stały; wykonanie używa batch 1. RGB CHW 0..1, maska 1 erase / 0 keep, surowe wyjście 0..255 potwierdzone w pilocie.

Istniejący ONNXruntime 1.19.2, CPU, cztery wątki. Nie instalowano zależności i nie zmieniano globalnego środowiska. Audyt GPU wykazał RTX 3070 z 8 GB, około 1,5 GB użytego VRAM i zerowe obciążenie obliczeniowe. GPU nie wykorzystano. Suma inferencji wyniosła 734.1 sekund, zwykle około 1,7–2 sekund na obraz. Jedyny forward to autoryzowana LaMa; bez badanego ViT/detektora, treningu, Eden jobs, commitów i imagegen.

## Geometria i maski

Maskę źródłową rasteryzowano z prawdziwej cystotome_seg, nie z bbox. Oddzielna maska edycji to jej eliptyczna dylatacja o promieniu 5 pikseli natywnych dla 399 par oraz 10 pikseli dla jednej jawnej poprawki QC. Przy 720×480 promień 5 odpowiada około 3,33 piksela w referencji 480×320; dla innych wymiarów skalę x/y zapisano osobno. Wyjątek promienia 10 opisano poniżej i w INPAINT_CONFIG.json. Każda maska edycji pokrywa oryginalną maskę. Oryginalne maski nie zostały zastąpione maskami edycji.

Preprocessing LaMa zachowuje proporcje każdej klatki, skaluje dłuższy bok do 512 i dopełnia kwadrat do 512×512. INTER_AREA dla obrazu, nearest dla maski, REFLECT_101 dla paddingu obrazu i zero dla maski. Dla dominującego 720×480 jest to 512×341 oraz padding top85/bottom86; rzeczywiste resized_wh i pad_tblr każdej pary są w QC. Z wyjścia usunięto padding, fill przeskalowano LINEAR do rzeczywistych wymiarów źródła, zaokrąglono/clipped do uint8 i wklejono wyłącznie w maskę edycji. Wynik zapisano bezstratnie jako PNG. Poza maską edycji pozostają dokładnie zdekodowane piksele oryginalnego JPEG.

TOKEN_MASKI.npz zachowuje referencję oryginału: coverage area po INTER_AREA do 480×320, konserwatywną binaryzację >0, siatkę 20×30 i patch 16×16. Indeks tokenu to 30×row+column. Referencja 480×320 jest bezpośrednim affine resize, więc dla 45 źródeł zmienia proporcje. Kąt wyboru był natywny; osobno zapisano angle_axis_480_deg. Bbox i tip mają prawidłowe skale x/y od rzeczywistego rozmiaru, nie uniwersalne 2/3. TOKEN_MASKI_EDYCJI.npz jest osobnym opisem zakresu interwencji. Oba człony pary bada się w **tych samych indeksach tokenów oryginalnej maski**, także gdy w syntetyku nie ma narzędzia. Nie zakładamy zerowania 768 współrzędnych; pooling i LDA są osobnymi obserwablami.

## Kontrola jakości i ograniczenia

Obejrzano sześć par pilota z różnymi orientacjami oraz dodatkową planszę 12 par (17 unikalnych par). W natywnym podglądzie TR_0022_S1_P03/00003266 znaleziono ciemną linię podobną do pozostawionego trzonu, choć jasny metal usunięto. Osobny pilot promieni 10 i 15 ograniczył ten ślad. Wybrano mniejszy z tych dwóch skutecznych promieni, 10, jako jawną poprawkę jednej pary; pozostałe 399 mają promień 5. Wersja pierwotna i obaj kandydaci są zachowani w pilot_poprawka/. To dobór maski na podstawie QC obrazu, nie wynik badanego ViT. Parametry, hash i rozszerzony zakres edycji zapisano; poza aktualną zadeklarowaną maską nadal jest zero zmian. Oryginalne tokeny i maska źródłowa tej pary pozostają takie same.

Mogą pozostać odbicia, cienie i artefakty wypełnienia. Automatycznie sprawdzono wszystkie 400 par; nie deklarujemy ręcznej certyfikacji każdego szczegółu 400 zdjęć w natywnej rozdzielczości. Kompletność 400 syntetyków oznacza wykonanie generacji i kontrolę geometrii, nie gwarancję idealnego usunięcia wszystkich wizualnych śladów narzędzia.

Tło pod narzędziem jest rekonstrukcją, nie odzyskanym prawdziwym tłem. Usuwany jest anotowany cystotome. Inne instrumenty poza jego maską pozostają i mogą wpływać na reprezentację. Maski źródłowe mogą pomijać odbicia lub fragmenty instrumentu. Te ograniczenia należy uwzględnić przed interpretacją zmiany latentu jako semantycznej swoistości.

Dawne 12 par empty_black to crop256, który zmienia 100 procent pikseli poza maską. Nie użyto ich zamiast obecnych 400 par. [Audyt dawnych negatywów](AUDYT_DAWNYCH_NEGATYWOW.json).

## Weryfikacja i pliki

- Dokładnie 400 oryginałów, 400 syntetyków, 800 obrazów i 400 par.
- 400 unikalnych frame_id, hashów JPEG oraz hashów pikseli; wszystkie należą do train.
- Native obraz/maska/wynik zgodne dla każdej pary: 355 par 720×480, 25 par 720×576, 17 par 640×432, 3 pary 540×540. Maski referencji 480×320 i coverage 400×600.
- Źródłowe hashe zachowane; wszystkie hashe syntetyków i masek edycji sprawdzone.
- Zero zmienionych pikseli poza maską edycji w każdej parze.
- Maski edycji są deklarowaną dylatacją i pokrywają źródłowe maski.
- Zachowano wszystkie indeksy tokenów oryginalnej maski; tool i background referencji są rozłączne.

[WERYFIKACJA_PAR.json](WERYFIKACJA_PAR.json), [WYNIKI.json](WYNIKI.json), [manifest.json](manifest.json), [COCO](annotations.coco.json), [PARY.json](PARY.json), [parametry inpaint](INPAINT_CONFIG.json), [QC każdej pary](INPAINT_QC.json).

Katalogi zbioru: obrazy_oryginalne/ (400 JPEG), obrazy_bez_narzedzia/ (400 PNG), maski_natywne/, maski_480x320/ i maski_edycji/ (po 400 PNG). Poza tym pilot_poprawka/ przechowuje dwa obrazy diagnostyczne i jedną archiwalną wersję; nie należą do 800 zdjęć zbioru. Model/cache/revision w model_lama/MODEL.json. Przygotowanie, inferencję i kontrolę można odtworzyć zapisanymi skryptami. Przygotowanie odmawia nadpisania JPEG, a inferencja przy wznowieniu sprawdza istniejące hashe i pomija gotowe pary. Końcowy protokół obejmuje lama_inpaint.py --all, osobny pilot_poprawka.py, zastosuj_poprawke_qc.py i sprawdz_pary.py. Nie zmieniać parametrów bez osobnego etapu QC.

Późniejsze forwardy/ruch między checkpointami są osobnym etapem [PLAN.md](PLAN.md). Nie uruchomiono zapisu trajektorii. Pełne 400×600×768 float32 ×101 stanów ×2 człony wymaga około 149 GB, dlatego wymaga osobnego budżetu i protokołu.

![12 par LaMa](PARY_LAMA_12.jpg)
