# PyKoopman / EDMD: 30 obrazów fit i 4 zachowane holdouty, 128 natywnych kanałów

Utworzenie: 2026-10-02, Europe/Warsaw. Etap: dobór i ekstrakcja; nie dopasowujemy tu modeli dynamiki.

Zachowujemy dokładnie 12 dotychczasowych obrazów i ich kolejność. Indeksy 0..7 to poprzednie fit, 8..11 to te same cztery holdouty. Dodajemy 22 nowe obrazy fit z dokładnej bazy 1500 treningowych klatek użytej w source run. Łącznie 34 obrazy, 30 fit i 4 holdouty. Role dotyczą modelu dynamiki; wszystkie obrazy należą do dataset train detektora. Nie przypisujemy dawnym sondom członkostwa w bazie 1500, jeśli go nie mają.

Nowe obrazy dobieramy deterministycznie, na podstawie członkostwa w bazie, zweryfikowanych masek i orientacji obrazu; bez oglądania reprezentacji. Kandydaci z gotowego zbioru 400 prawdziwych train muszą należeć do bazy 1500. Wykluczamy wszystkie dotychczasowe frame-ID i filmy wszystkich 12 sond. Wybieramy po jednym nowym obrazie z filmu, z wyrównaniem liczebności przedziałów kąta osi maski. Geometryczny kąt maski nie jest PCA reprezentacji; kanały latentu nie są obracane ani redukowane. Zachowujemy prawdziwe maski i ich pochodzenie.

Model: dokładny source model.py, SmallViT dim192, 6 bloków, 3 głowy, MLP768. Reprezentacja cząsteczki to wyjście GELU bloki[-1].mlp.act, siatka 20×30, 600 tokenów w kolejności y*30+x. Każdy token zachowuje dokładnie J128 z poprzedniego wyboru na k500. Bez PCA, SVD reprezentacji, poolingowej zamiany cząsteczek ani nowego wyboru kanałów.

Pomiary dla każdego z 34 obrazów na wszystkich 101 lokalnych checkpointach k0000..k1000 co 10 aktualizacji. To 1001 stanów źródłowego biegu, lecz tylko 101 dostępnych checkpointów do ponownej ekstrakcji. 1000 aktualizacji z batch32 na bazie1500 to około 21.3 prezentacji datasetu, nie 21 oddzielnych modeli/epok prognozy. Nie odtwarzamy brakujących wag pomiędzy checkpointami.

Preprocessing: stałe RGB uint8 480×320, tensor float32 /255; normalizacja ImageNet wewnątrz source modelu. Dotychczasowe obrazy są dokładną kopią source obrazy.npy. Nowe obrazy pochodzą z cache baza1500.npy w porządku COCO. Weryfikujemy zgodność z odczytem rzeczywistych obrazów źródłowych. Eval, no_grad/inference_mode, FP32, bez TF32 i autocast, cudnn.benchmark=False. BatchNorm w eval wykorzystuje zapisane running buffers; ruch obejmuje zmianę wag i tych statystyk.

Kontrola: strict load wszystkich checkpointów, hash architektury/preprocess/J/manifestu/wag. Na 12 zachowanych sondach porównanie wyekstrahowanych J128 do LATENT_GESTY[step,probe,:,J] dla wszystkich 101 checkpointów. Dodatkowo pełne768 na k0/k500/k1000. Docelowa tolerancja maxabs1e-5; raportujemy rzeczywisty maxabs, relative L2 i zgodność, zamiast ukrywać różnicę sprzętu. Sprawdzamy skończoność, kształt, 34 unikalne frame-ID, niezmienność czterech holdoutów i brak aktualizacji BN podczas pomiaru.

Zapis: LATENT_J128.npy float32 [101,34,600,128], około1.055GB. MASKI.npz zachowuje tool/background/coverage/mask_input i tokenowe wagi, OBRAZY.npy oraz PNG wejściowe, MANIFEST.json i DANE.json z fit_indices=[0..7,12..33], holdout_indices=[8..11], checkpoint_steps=0..1000 co10. Nie zapisujemy pełnych768 wszystkich obrazów. Źródła tylko do odczytu. Brak nowego treningu, Eden jobs, pobierania ani commitów.

Przed dopasowaniem PyKoopman / EDMD root zapisze oddzielnie pełny protokół identyfikacji i walidacji. J wybrane na k500, więc wcześniejsze oceny w tej bazie są retrospektywne. Przy H500 od origin500 późniejszych rzeczywistych pomiarów nie wolno podawać modelowi. Cztery holdouty były już oglądane w dawnych badaniach, więc nowy wynik nie będzie nietkniętym niezależnym potwierdzeniem.
