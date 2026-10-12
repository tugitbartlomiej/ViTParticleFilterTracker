# 400 prawdziwych train i 400 par bez narzędzia

Utworzono 2026-10-01 Europe/Warsaw przed przygotowaniem. Na teraz przygotować400 unikalnych rzeczywistych train z dostarczoną segmentacją cystotome; syntetyki400 mają oddzielny status pending do decyzji root o lokalnym modelu wymazywania. To naturalne kąty na różnych tłach, nie kontrolowane obroty jednego obiektu.

Źródło podziału: istniejący lokalny tracking pool_manifest.jsonl; wyłącznie split=train. Obrazy i maski z oryginalnych archiwów3_Object_Tracking, powiązanie przez dokładny film/frame_number/filename CSV. Oryginalne JPEG skopiować byte-for-byte. Zachować źródłowe polygons i bbox/tip oraz ich checksums. Nie zastępować maski bbox, nie używać wal/test, nie mutować źródeł.

Kąt głównej osi PCA binary mask w natywnych współrzędnych: atan2(eigenvector_y,eigenvector_x) modulo180, ośx w prawo, ośy w dół. To oś bez zwrotu; oznaczyć niepewność ratio eigenvalues<3, nie interpretować jako kąt końcówki. Tip od źródła osobno. Dobór400 warstwowo po12 kątach15stopni i filmach, deterministyczne seed20261001; preferować rozstaw50frames w danym filmie, w razie braku stopniowo20/5/0 i zapisać. Nie gwarantować400 równomiernych kątów, jeśli brak w danych. Preferować maski wydłużone.

Zachować natywne obrazy/maski; pochodna referencja maski480x320 ma INTER_AREA coverage oraz konserwatywną binaryzację>0 jak wcześniejszy test. Siatka20x30=600 tokenów, patch16x16, indeks row-major token=30*row+col. Coverage binary oraz coverage area oddzielnie; oryginalna maska wyznacza stałe indeksy w obu członach pary. Nie przeprowadzać forward/treningu. Wyłącznie przygotowanie i audyt.

Analiza późniejsza przy tym samym zamrożonym modelu i preprocessing: porównywać z_i oryginału i syntetyka w TYCH SAMYCH pozycjach tokenów maski oryginału, nawet gdy narzędzia w syntetyku nie ma. Kontrola patchy tła oddalonych od maski; komponenty768 traktować jako współrzędne, nie zakładać ich zerowania po usunięciu narzędzia. Zmiana reprezentacji może obejmować kontekst. Tożsamość semantyczna wymaga kontroli innych narzędzi i artefaktów.

Plan inpaint: lokalny kandydat LaMa z HuggingFace wskazany przez root, bez instalacji/pobierania ani realizacji400 do decyzji root. Obraz poza maską zachować dokładnie; jeśli dilatation mask do usunięcia krawędzi jest potrzebna, zapisać maskę interwencji oddzielnie od ground-truth maski cząsteczek. Nieznane tło pod narzędziem jest rekonstrukcją, nie prawdziwym odkrytym tłem. Zapisać model/version/hash, preprocessing/padding, seed, mask-dilation i metodę compositing oraz QC: pozostawione fragmenty, odbicia/cienie, szwy, uszkodzenie struktur oka. Ramię kontroli artefaktu edycji i wersje alternatywne należy wyznaczyć przed semantycznymi wnioskami.

Przygotowanie obali: mniej400 poprawnych/unikalnych train, brak maski lub niezgodny wymiar/membership/provenance. Nie uzupełniać innym splitem. Weryfikacja exactly400, unikalneidentyfikatory i bytes/pixels, kontrola masks/bboxes, źródła, histogramy kąta/filmów, kontaktowa plansza48 miniaturek. Dane małe, budżet dysku początkowo23,96GB wolnego. Żadnych Eden jobs, forwardów, treningu, commitów.


## Autoryzowane wykonanie LaMa, 2026-10-01 16:56:16 Europe/Warsaw

Użytkownik jawnie autoryzował pobranie LaMa z HuggingFace i wykonanie wszystkich400syntetyków przez lokalny model/programowąinferencję. HF_HOME ustalone wyłącznie w procesie: F:/HuggingFace; wagi wskazywane ze standardowego cache hub, nie pozostają w eksperymencie. Istniejący ONNXruntime1.19.2 CPU, batch1,4wątki; GPUaudyt wykonany, nie zmieniamy globalnych zależności i nie deklarujemyGPUinferencji. Model208MB, revision iSHA wMODEL.json. FaktyczneIO ma dynamicznybatch i fixed512x512.

Pilot6 rozstrzyga skalę wyjścia, geometrię i jakość usunięcia. Preprocessing zachowuje proporcje720x480→512x341, imageREFLECT_101 padtop85/bottom86, maskconstant0. Mask1erase/0keep, imageRGBCHW0..1, output0..255. Maskaedycji osobna: źródłowa maska natywna dylatowana elipsą radius5px=3,33px przy480x320. Wynikfillresize do720x480, PNGcomposite tylko wmasceedycji, poza nią dokładnie piksele zdekodowanegooryginału. Tokenymaskioryginału pozostają bez zmian. Po wizualnejkontroli pilota wykonaniecałych400, automatycznaQC każdej pary i plansza podglądowa. Rekonstrukcjatła syntetyczna; nie obiecywać braku wszystkich pozostałości narzędzia po samym teście liczbowym.

## Zakres późniejszego pomiaru reprezentacji

Zbiór400par służy do analizy reprezentacji na obrazach należących do train. Nie jest nowym niezależnym testem detektora. Najpierw wybrać jawny zamrożony checkpoint i stały preprocessing480x320, trybeval, bezaugmentacji. Dla każdej pary porównać pełne600 pozycji, a podgrupę narzędzia wyznacza oryginalnecoverage w obu członach. Ruch między stanami treningu mierzy się oddzielnie jako zmianęz_i tego samego obrazu i tokenu przy kolejnych checkpointach. Usunięcie narzędzia to interwencja obrazowa, nie samo przesunięcie czasu treningu.

Dla pełnej600x768 reprezentacji raportować normę i poszczególne współrzędne zmian, bez założenia ich wyzerowania po inpaint. Zmiany w tokenachmaski porównać z kontrolą tła poza maską/dylatacją oraz kontrolą artefaktówsyntetyku. Zależność kąta od reprezentacji może być skorelowana zfilmami/tłem/oświetleniem; ocena semantycznej swoistości wymaga podziału po filmach lub jawnej kontroli tychzmiennych, nie losowych tokenów uznanych za niezależnepróby. Nie nazywać116filmów116potwierdzonymi pacjentami.

Przed późniejszym zapisem trajektorii oszacować budżet:400x600x768float32 to około737MB na jeden stan i członpary;101stanów obu obrazów to około149GB. Nie uruchamiać takiego pełnego zapisu automatycznie przy24GBwolnego. Rozważyć agregaty wstrumieniu, wybranejednoznaczne podprzestrzenie lub mniejszy jawnie określony panel; wybór nie może zastąpić wszystkich600cząsteczek ich średnią bezoznaczenia. Obecnie żadnych takich forwardów ani zapisów nie wykonano.


Korekta metadanych po końcowym audycie: w400 występuje355 obrazów720x480,25 obrazów720x576,17 obrazów640x432 i3 obrazy540x540. Inpaint od początku używał rzeczywistych wymiarów i zachował je; nie potrzebował ponownej generacji. Usunięto błędne założenie jednego rozmiaru z finalnego verifiera. Bbox/tip480 mają skale480/w oraz320/h; kątyPCA natywne ireferencji480 zapisane osobno. LaMa zachowujeproporcje, a późniejsza referencjaViT480x320 jest affine resize, który zmienia proporcje45źródeł. Maskaedycji ma stały native radius5 i jawne odpowiedniki x/y wreferencji.

## Końcowy etap QC — 2026-10-01 17:24:29 Europe/Warsaw

Wygenerowano wszystkie 400 syntetyków. Kontrola geometryczna została wykonana dla wszystkich 400 par. W przeglądzie 17 unikalnych par wykryto ciemny ślad trzonu w TR_0022_S1_P03/00003266. Oddzielny pilot promienia 10 i 15 pokazał jego ograniczenie; zastosowano minimalny z przetestowanych skutecznych promieni 10 tylko w tej parze. Pozostałe 399 zachowują promień 5. To jawna poprawka QC, nie wynik badanego ViT ani strojenie miary latentu. Maski źródłowe i indeksy tokenów są stałe. Wersję pierwotną oraz dwóch kandydatów zachowano w pilot_poprawka. Ponowny audyt musi potwierdzić piksele poza aktualną maską i nowe hashe. Przegląd wizualny jest próbkowy; nie gwarantuje idealnego braku śladów narzędzia na wszystkich syntetykach. Żadne forwardy ViT/detektora nie zostały wykonane.
