# Pełny CAV z 400 par oryginał - LaMa

Klasa badawcza. Pytanie: czy pełne 460800 współrzędnych ostatniego GELU zamrożonego ViT zawiera liniowy kontrast oryginalnego narzędzia versus jego usunięcia i jak detektor reaguje na ten kierunek? Brak jakości testowej ponad kontrolami lub niewierność finite differences obala interpretację jako użytecznego kierunku detektora.

## Źródło metody

Przeczytano cały kim18d.pdf (10 stron) bez modyfikacji źródła. Kim et al., ICML 2018: §3.2 separator aktywacji i normalna hiperpłaszczyzny, §3.3 pochodna logitu wzdłuż jednostkowego CAV, §3.4 frakcja S>0, przypis 1 na stronie 4 nakazuje flatten w*h*c. §3.5 opisuje zwykle 500 dopasowań z losowymi negatywami oraz dwustronny test t względem 0.5 i korekcję Bonferroniego. §3.6 relative CAV. §4: sortowanie niezależnych obrazów, wizualizacja CAV, kontrole spurious concepts, dokładność sondy osobno od TCAV, kontrolowane caption/image datasets i odniesienie do znanego ground truth. Nie wykonujemy nowych DeepDream ani treningu kontrolowanych sieci, bo nie należą do zamówionego pomiaru.

## Aparat i dopasowanie

Wyłącznie oryginalny MODEL_VIT_K1000.pt, krok 1000. Strict state load, eval, requires_grad False, BN zamrożone, checkpoint i wszystkie parametry/bufory porównane przed/po. RGB, 480x320, INTER_AREA, /255, normalizacja wewnątrz modelu. Hook GELU ostatniego MLP: wszystkie 600 tokenów i wszystkie 768 kanałów. 800 forwardów, CPU float32 memmap 800x460800, 1474560000 bajtów plus nagłówek. Żadnego wykorzystania pooled features, masek, PCA ani J128.

PODZIAL.json jest autorytatywny: 69/23/24 rozłączne filmy, 240/75/85 par. Historyczne pole split w PARY lub manifest nie zastępuje tego podziału. Oba człony pary należą do tego samego filmu i splitu. Oryginał etykieta +1, LaMa -1. Cache manifest obejmuje hashe wszystkich obrazów oraz źródeł. Kolejność pair-major original, synthetic.

Separator ridge liniowy, bez featurewise scaling: średnia i globalny skalar s obliczone tylko z train. Cel sum_i(y_i - z_i dot beta)^2 + lambda ||beta||², z=(x-mu_train)/s, intercept w tych współrzędnych 0, ponieważ klasy są dokładnie zbilansowane. Dual alpha=(K+lambda I)^-1 y; raw coef=(X-mu)'alpha/s², intercept=-mu dot raw coef. Jest to adaptacja linear probe, nie SVM ani kernel nieliniowy. Lambda 1e-5..100 dobrana po val BA, potem AUROC, potem większa lambda. Wybór zamrożony przed oceną testu. Primal versus dual sprawdzony na wszystkich 800 próbkach. Unit raw CAV ma kształt 600x768 i normę Frobeniusa 1.

## Kontrole i równy dostęp do informacji

30 powtórzeń bootstrap filmów train, przy zamrożonym preprocessingu i lambda. 30 kontroli separatora z losowym znakiem etykiet dla całego filmu train: oryginał i LaMa zawsze przeciwne, zależności i bilans zachowane. 30 losowych kierunków w całej przestrzeni 460800D z normą 1, bez orientowania według heldout wyników. Wszystkie ramiona używają tego samego checkpointu, latentów, testu i definicji logitu. Kontrole widzą te same train i zamrożoną lambda, nie testowe etykiety do fittingu. BA i AUROC sondy raportowane osobno. Kontrola może wypaść powyżej badanego wyniku: to falsyfikacja przewagi nad tą kontrolą, a nie powód jej usunięcia.

## Ocena i niepewność

85 heldout oryginałów z 24 filmów. Wynik detektora to pre-sigmoid logit w ustalonej komórce centrum GT, zaokrąglonej zgodnie z model.cele. To adaptacja z klasyfikacji do dense detection. Autograd obejmuje pełną macierz GELU, iloczyn Frobeniusa z pełnym CAV. Żadnego ROI lift ani maskowania gradientu. Po ostatnim GELU operacje są token-local, więc naturalne zera innych tokenów są cechą architektury.

TCAV frakcja dodatnich S, średnia signed, mean |S|, mediana, zakres i zera. CI95 z 3000 bootstrapów całych filmów: primary equal-film oraz frame-weighted TCAV jako odpowiednik paper. BA także film bootstrap. Empiryczne null dla TCAV film-weighted: dwustronna odległość od .5, p=(1+liczba co najmniej tak skrajnych null)/(31), minimalna rozdzielczość 1/31. Nie traktujemy klatek jako niezależnych replikacji. Test t paper przeciwko .5 na bootstrap CAV tylko opisowy z invalid-inference flag; wspólne filmy i zmieniona kontrola nie uzasadniają oryginalnego testu istotności. Nie formułujemy niezależnych potwierdzających wniosków p<.05 z dwóch kontroli.

Finite differences na 3 z góry wybranych heldout obrazach (indeksy 0,42,84), główny, pierwszy permuted i random. Prawdziwy full GELU hook +/- epsilon w całym forwardzie, epsilon median(train latent norm)*[1e-5,1e-4,1e-3]. Bez doboru epsilon po teście. Main największe epsilon musi mieć absolute error < max(2e-5, .03|autograd|); pełne liczby ujawniają float32 cancellation, również kontroli. Niezgodność oznaczona warning, nie ukryta.

## Granice twierdzeń i zapisu

Wysoki wymiar i małe N; pozycja, tło, format JPEG/PNG i artefakty LaMa mogą rozdzielać przykłady. Controlled LaMa kontra random paper jest jawnie adaptowane. Test jest holdout dopasowania konceptu, nie nowy niezależny test ViT: obrazy mogły uczyć detektor. Wynik nie dowodzi wyłącznej tożsamości ani przyczynowej poprawy uczenia. Perturbacje mogą wychodzić poza manifold obrazów. J128 nie uczestniczy w żadnym obliczeniu, wykresy obejmują wszystkie kanały i pozycje.

Zapisy wyłącznie experiments/vit_cav_pelny_latent_pary400 i outputs/vit_cav_pelny_latent_pary400. WYNIKI.json i rodzinny REGISTRATION.json zastępują globalny rejestr poza dozwolonym zakresem. Bez tez i zapisów w Grant, commitów, modyfikacji PDF lub innych procesów. Background hidden Windows, JSON atomic z retry, CPU max4, pojedynczy obraz na GPU, limit GPU 12% i CPU fallback przy małej wolnej pamięci. Raport zawiera runtime, PID, model source i hashes.
