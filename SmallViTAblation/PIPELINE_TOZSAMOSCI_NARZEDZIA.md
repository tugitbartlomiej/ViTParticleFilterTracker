# Pipeline tożsamości narzędzia: identyfikacja, pomiar ruchu, prognoza, generacja SD i retrieval

Utworzono: 2026-10-02T12:23:39+02:00. Strefa czasu: Europe/Warsaw. Aktualizacja: 2026-10-02T13:02:43+02:00.

Dokument porządkuje siedem kroków ustalone w rozmowie z użytkownikiem. Opisuje wykonane pomiary i rzeczywisty zakres ich interpretacji. Pytania poniżej są wiernymi parafrazami rozmowy, a nie dosłownym transkryptem. Kolejne etapy można dopisać jako osobne, datowane sekcje po ich uzgodnieniu i sprawdzeniu.

## 1. Co jest już potwierdzone, a co jest celem

Pierwszy krok wyznaczył zestaw **128 oryginalnych kanałów**, których wspólny wzorzec pozwala rozpoznawać obecność narzędzia w znanym obszarze obrazu. Na odłożonych filmach uzyskano 92,8274 procent filmowo ważonej balanced accuracy. Użytkownik zaakceptował ten wynik jako podstawę dalszej analizy.

Drugi krok został wykonany: dla 34 stałych obrazów zapisano reprezentacje 600 cząsteczek na każdym z 101 checkpointów. Każda cząsteczka zachowuje te same 128 wybrane współrzędne. Dane pozwalają mierzyć zmianę reprezentacji podczas uczenia przy niezmienionym obrazie.

Trzeci krok dotyczy identyfikacji dynamiki i prognozy. **Celem użytkownika jest co najmniej 85 procent trafności znaku prognozowanego skumulowanego przemieszczenia**, a nie 85 procent trafności rozpoznania obecności. Dotychczasowe modele nie osiągnęły tego celu. Nowy przebieg z rzeczywistym pakietem PyKoopman został obliczony; w chwili zapisu dokumentu trwa jego raportowanie i audyt. Nie jest tu przedstawiany jako potwierdzony sukces prognozy.

## 2. Cząsteczki, makrowektory i współrzędne

Cząsteczka jest reprezentacją jednego przestrzennego tokenu obrazu. W mierzonym miejscu SmallViT ma pełny wektor

$$z_i(t)\in\mathbb R^{768}.$$

Dla wejścia o rozdzielczości 480 × 320 model daje siatkę 30 × 20, czyli **600 cząsteczek na obraz**. Nie są to paczki treningowe. Liczby 600 i 768 wynikają z konkretnej architektury i miejsca pomiaru.

Globalny układ latentny to wspólna przestrzeń współrzędnych wszystkich cząsteczek. Nie oznacza ich średniej. Makrowektory, w terminologii użytkownika, to wspólne wektory bazowe $e_j$. Mikrowektory cząsteczki to jej składowe $z_{ij}(t)e_j$. Implementacja przechowuje współrzędne $z_{ij}$, a nie osobny zbiór wielowymiarowych wektorów wewnątrz tokenu.

Po pierwszym kroku analizujemy podzbiór $J$ zawierający 128 oryginalnych kanałów:

$$x_i(t)=z_i(t)[J]\in\mathbb R^{128}.$$

Pełna reprezentacja sieci nadal ma 768 wymiarów. Zakończona ekstrakcja LATENT_768.npy uzupełnia dane dla neuronowego kroku trzeciego; nie zastępuje wcześniejszego LATENT_J128.npy. Wybór 128 współrzędnych nie zmienia tokenizacji, liczby cząsteczek ani sieci. Te same osie obowiązują dla wszystkich obrazów, cząsteczek i checkpointów. Nie wykonujemy PCA, autoenkodera ani obrotu reprezentacji w celu wyznaczenia tych kanałów. To ograniczenie dotyczy identyfikacji J i pomiaru natywnych współrzędnych. W kroku trzecim użytkownik później jawnie dopuścił neuronowy encoder–decoder NNDMD; nie zmienia on listy J służącej identyfikacji. Najnowsza korekta rozszerza wejście i decode NNDMD do pełnych 768 kanałów tokenu.

## 3. Krok pierwszy: nadzorowana identyfikacja kanałów związanych z obecnością narzędzia

### 3.1. Co porównujemy

Punktem wyjścia jest 400 rzeczywistych obrazów ze zbioru train z maskami narzędzia. Dla każdego obrazu przygotowano lokalnym modelem LaMa syntetyczną wersję z usuniętym narzędziem. Powstało 400 par, czyli 800 obrazów: oryginał z narzędziem oraz syntetyczny negatyw.

Model ViT jest zamrożony na checkpointcie **k0500**, odpowiadającym 500 wykonanym aktualizacjom treningu. Obie wersje obrazu przechodzą przez ten sam model w trybie eval. Reprezentację odczytujemy po GELU ostatniej MLP, z `bloki[-1].mlp.act`.

W obu obrazach bierzemy **dokładnie te same pozycje tokenów wyznaczone maską oryginalnego narzędzia**. W obrazie po usunięciu nie szukamy nowej maski. Zachowujemy indeksy badanych cząsteczek, choć ich piksele i reprezentacje zmieniły się po inpaintingu. Maska oryginalnego narzędzia i maska edycji LaMa są osobnymi artefaktami.

### 3.2. Z tokenów do przykładu klasyfikacyjnego

Nie identyfikowaliśmy oddzielnego klasyfikatora dla każdego patcha. Dla każdego obrazu liczymy jedną średnią 768-wymiarową reprezentację jego znanego ROI. Udział maski narzędzia w nominalnej komórce tokenu określa wagę coverage:

$$\alpha_i=\frac{c_i}{\sum_{k\in I}c_k},\qquad p^\pm=\sum_{i\in I}\alpha_i z_i^\pm\in\mathbb R^{768}.$$

$I$ to indeksy tokenów z dodatnim pokryciem oryginalnej maski, a znaki $+$ i $-$ oznaczają oryginał i wersję LaMa. Wagi oraz indeksy są identyczne po obu stronach pary. Pooling jest tu jawnym narzędziem identyfikacji na poziomie ROI obrazu; późniejsze trajektorie zachowują osobno każdą cząsteczkę.

Surowe średnie `pooled400` mają kształt `(400, 2, 768)`. Zapisano je przed jakąkolwiek projekcją wcześniejszego eksperymentu. Obecny dobór kanałów wykorzystuje wyłącznie te surowe współrzędne i ignoruje wcześniejsze U, wartości własne oraz scalery po U.

### 3.3. Jak wyznaczono 128 kanałów

Podział po filmach zawiera 69 filmów train, 23 validation i 24 test, odpowiadające 240, 75 i 85 parom. Obie strony każdej pary pozostają razem. Filmy mają równą masę, obrazy w obrębie filmu równą wagę, a obie klasy są równoważne.

Ranking powstaje wyłącznie na 240 parach train. Dla kanału $j$ liczymy sparowaną różnicę $\delta_{mj}=p^+_{mj}-p^-_{mj}$, jej ważoną średnią $\mu_j$ i odchylenie standardowe $\sigma_j$. Wynik rankingowy to

$$R_j=\frac{|\mu_j|}{\max(\sigma_j,10^{-12})}.$$

Duży wynik oznacza powtarzalną różnicę między obecnością a usunięciem narzędzia. Wartość bezwzględna pozwala uwzględnić zarówno kanały rosnące, jak i malejące. Nie oznacza, że dany kanał samodzielnie rozpoznaje narzędzie lub jest jego indywidualną przyczyną.

Dla pierwszych K kanałów rankingu dopasowujemy logistyczny probe L2. Standaryzacja każdego oryginalnego kanału pochodzi tylko z train; nie obraca osi. Przeszukano K = 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768 oraz C = 0,01; 0,1; 1; 10, czyli 48 dopasowań. Wybór odbył się na validation: najmniejsze **sprawdzone K** przekraczające 90 procent balanced accuracy, następnie ustalone rozstrzygnięcia remisu. To nie jest dowód minimalności spośród wszystkich możliwych podzbiorów 768 kanałów.

Wybrano **K = 128, C = 0,01**, przy stałym progu decyzji 0,5. Żadne mniejsze K z przeszukanej listy nie przekroczyło 90 procent na validation. Zestaw i model zamrożono przed pojedynczą oceną testową.

| Ocena rozpoznania na checkpointcie 500 | Wynik |
|---|---:|
| Validation: filmowo ważona balanced accuracy | 90,6703% |
| Test: filmowo ważona balanced accuracy | 92,8274% |
| Test: czułość | 91,9742% |
| Test: swoistość | 93,6806% |
| Test: AUROC | 0,9741 |
| Test: nieważona accuracy | 91,7647% |
| CI95 balanced accuracy, bootstrap całych filmów | [88,17%; 96,99%] |

Przy równych wagach obu klas filmowo ważona accuracy jest równoważna balanced accuracy. Jednostką rozpoznania jest **ROI obrazu**, nie pojedyncza cząsteczka. Punktowy wynik przekracza 90 procent, lecz dolna granica przedziału nie przekracza 90 procent. Nie ma gwarancji 100 procent ani gwarancji wyniku ponad 90 procent na nowych danych.

### 3.4. Co znaczy „kierunki tożsamości” i czego wynik nie dowodzi

Wybrane kanały **razem** rozróżniają obecność narzędzia i jego syntetyczne usunięcie w znanym ROI. Nie wszystkie muszą rosnąć przy obecności narzędzia, nie wszystkie zanikają po jego usunięciu i nie każdy samodzielnie wystarcza do klasyfikacji. Decyzję daje wspólny wzorzec ich wartości i dopasowane współczynniki probe.

Po usunięciu narzędzia nadal istnieje 600 tokenów. Zmienia się ich reprezentacja; cząsteczki nie znikają z architektury. Także wartość bliska zeru w danym kanale nie oznacza zaniku tokenu.

Maska ustala związek przestrzenny z narzędziem. Nie dowodzi wyłącznej semantycznej odpowiedzialności tokenu: wcześniejsze attention łączy informacje z całego obrazu. Reprezentacja tokenu narzędzia może zawierać kontekst tła. Wynik nie rozstrzyga odróżnienia tego narzędzia od każdego innego instrumentu ani tożsamości jego konkretnego fizycznego egzemplarza.

LaMa tworzy sztuczne tło, a nie odzyskuje prawdziwych zasłoniętych pikseli. Różnice mogą częściowo zawierać artefakty edycji. Obrazy pochodzą z train detektora; odłożenie filmów dotyczy dopasowania identyfikatora, a nie nowego niezależnego testu wytrenowanego ViT. Rozpoznanie na k0500 nie dowodzi utrzymania takiej samej trafności na każdym późniejszym checkpointcie.

## 4. Krok drugi: pomiar trajektorii tych samych cząsteczek

### 4.1. Co pozostaje stałe, a co się zmienia

Dla danego obrazu przechodzimy przez checkpointy wag **0, 10, 20, ..., 1000**. Za każdym razem podajemy dokładnie ten sam obraz i odczytujemy reprezentacje tych samych przestrzennych tokenów. Zestaw J pozostaje niezmieniony; nie identyfikujemy go ponownie na kolejnych checkpointach.

To pomiar zmiany reprezentacji podczas uczenia, **nie ruch narzędzia w filmie**. Jeden checkpoint opisuje cały model i służy do pomiaru wszystkich obrazów. Nie tworzymy odrębnego checkpointu dla każdego obrazu.

Zmieniają się wagi sieci oraz zapisane bieżące statystyki BatchNorm. Dlatego obserwowanej dynamiki eval nie przypisujemy wyłącznie gradientowi lub semantyce narzędzia. Eval zamraża zachowanie podczas pojedynczego pomiaru, lecz kolejne checkpointy zawierają różne stany sieci.

### 4.2. Dane, obrazy i liczba przejść

Zachowano 12 wcześniejszych sond: 8 fit oraz dokładnie te same 4 holdouty o indeksach 8–11. Dodano 22 obrazy fit z 22 nowych filmów dokładnej bazy 1500 klatek treningowych, z wykluczeniem filmów wszystkich wcześniejszych sond. Łącznie mamy **30 obrazów fit + 4 holdout = 34 obrazy**.

Historycznie tylko 3 z 8 starych obrazów fit należały do dokładnej bazy 1500; pozostałe zachowano zgodnie z poleceniem, ze źródłowego train poza tą bazą. Nie twierdzimy zatem, że wszystkie 34 obrazy są członkami bazy 1500.

| Zakres | Liczba |
|---|---:|
| Checkpointy na obraz | 101 stanów, od 0 do 1000 co 10 |
| Aktualizacje źródłowego treningu | 1000 |
| Obrazy fit / holdout | 30 / 4 |
| Zrzuty obraz–checkpoint dla fit | 30 × 101 = 3030 |
| Sąsiednie pary stanów na obraz w całej historii | 100 |
| Pary obraz–czas dla 30 fit w całej historii | 3000 |
| Checkpointy dostępne do originu 500 włącznie | 51 |
| Pary na obraz do originu 500 | 50 |
| Pary obraz–czas dla 30 fit do originu 500 | 1500 |

Każda para obraz–czas zawiera wiele par tokenów. Nie są one niezależnymi powtórzeniami treningu. 1000 aktualizacji z batch 32 przy bazie1500 odpowiada około 21,3 prezentacji całej bazy; nie jest to 21 checkpointów ani 101 aktualizacji.

### 4.3. Jak liczymy ruch

Dla tej samej cząsteczki $i$ zapisujemy $x_i(t)=z_i(t)[J]$. Przy sąsiednich zapisach ruch wynosi

$$\Delta x_i(t\to t+10)=x_i(t+10)-x_i(t).$$

To różnica przez **10 aktualizacji**. Podzielenie przez 10 daje średni przyrost na aktualizację, ale nie odtwarza nieobserwowanych dziewięciu stanów pośrednich. W tym zbiorze najbliższy bezpośrednio mierzony przyszły stan jest odległy o H = 10 aktualizacji.

Nie zmieniają się wspólne osie kanałów. Zmieniają się wartości współrzędnych cząsteczek na tych osiach. Dane pełnych trajektorii są przechowywane per obraz i per token; pooling nie zastępuje ich średnią. Średnie tła i coverage pozostają osobnymi opisami pomocniczymi.

### 4.4. Wykonanie i zgodność pomiaru

Pomiar używa dokładnej architektury źródłowego SmallViT: wymiar tokenu backbone 192, 6 bloków, 3 głowy attention, ostatnia MLP z wewnętrznym wymiarem 768. Wejście to stały RGB 480 × 320, wartości FP32 podzielone przez 255; normalizacja ImageNet odbywa się w modelu. Użyto eval, inference_mode, bez autocast i bez TF32, z lokalną RTX 3070.

Gotowy `LATENT_J128.npy` ma kształt **(101, 34, 600, 128)** i typ float32. Wszystkie wartości są skończone. Zachowano maski, coverage, indeksy, role obrazów oraz kolejność J.

Porównanie wszystkich 101 checkpointów na 12 wcześniejszych sondach dało maksymalną różnicę bezwzględną wybranych kanałów około 2,265 × 10⁻⁶. Pilot pełnych 768 na checkpointach 0/500/1000 dał maksymalną różnicę około 2,384 × 10⁻⁶ i względną normę błędu nie większą niż 4,922 × 10⁻⁷. Jest to poniżej zadanej tolerancji 10⁻⁵; nie deklarujemy bitowej identyczności.

Źródłowy bieg nie zapisuje kompletnej listy rzeczywistych batch-ID, augmentacji ani gradientów każdego kroku. Lokalne checkpointy nie dostarczają kompletnego stanu AdamW dla każdego przejścia. Brakujących wejść nie rekonstruujemy przez zgadywanie. Ta granica dotyczy późniejszej interpretacji modelu dynamiki.

## 5. Krok trzeci: benchmark EDMD oraz żądany neuronowy NNDMD

### 5.1. Czego EDMD dokonuje po pomiarze

Zrzuty reprezentacji i ich odejmowanie należą do kroku drugiego. EDMD otrzymuje następnie pary kolejnych stanów tego samego tokenu i obrazu. Dopasowuje **jeden wspólny model**

$$x_i(t+10)\approx A x_i(t)+b,\qquad A\in\mathbb R^{128\times128},\quad b\in\mathbb R^{128}.$$

Równoważnie opis przyrostu wynosi $\Delta x\approx(A-I)x+b$. Nie dopasowujemy 30 osobnych macierzy, aby następnie je uśrednić. Obrazy zachowują oddzielne trajektorie i dostarczają wiersze danych do wspólnej regresji. Model wspólny jest hipotezą sprawdzaną na odłożonych obrazach, a nie dowodem wspólnych sił między wszystkimi tokenami.

W wykonanym nowym eksperymencie rzeczywiście użyto `PyKoopman.Koopman`, obserwabli `Identity` i regresora `EDMD`. Jawna stała rozszerza stan do `[x,1]`, dając operator 129 × 129. Końcowe A i b są wyrażone w oryginalnych 128 kanałach. Pełna SVD użyta wewnętrznie przez solver EDMD jest algorytmem rozwiązania regresji bez redukcji rzędu; nie oznacza PCA lub wyboru obróconej reprezentacji tożsamości.

Wagi coverage nakłada się na obie strony pary wraz ze stałą. Każdy obraz ma taką samą łączną masę. Nie tworzy się przejść między końcem trajektorii jednego obrazu a początkiem drugiego.

### 5.2. Dopasowanie i prognoza są różnymi czynnościami

Wewnętrzny wybór używa 24 obrazów fit do dopasowania na prefiksie do 300 i 6 innych obrazów fit do walidacyjnej prognozy od 300 do 500. Po wyborze regularyzacji operator jest ponownie dopasowany na wszystkich 30 fit, wyłącznie z prefiksu0–500. Końcowe cztery holdouty nie uczestniczą w dopasowaniu dynamiki.

Prognoza od500 podaje modelowi obserwowany stan500 i dalej iteruje wyłącznie własne stany:

$$\widehat x(500+10(h+1))=A\widehat x(500+10h)+b.$$

Rzeczywiste stany510–1000 służą wyłącznie ocenie. Nie podajemy przyszłych obrazów, batchy, gradientów ani innowacji z przyszłych pomiarów. Wersja Kalmanowska lub obserwator również nie mogłaby otrzymywać przyszłych pomiarów podczas takiej prognozy.

J wybrano na k500, więc walidacja dynamiki300–500 jest retrospektywna względem wyboru współrzędnych. Cztery holdouty dynamiki były już oglądane we wcześniejszych badaniach i należą do train ViT. Nowy eksperyment jest eksploracyjny; nie jest niezależną replikacją na świeżym treningu.

### 5.3. Wąskie gardło i status wyników

Najważniejsze jest teraz uzyskanie prognozy, która na krótkich i długich horyzontach ma przewagę nad brakiem zmiany, ostatnim przyrostem/stałą prędkością oraz historyczną średnią lub reversion. Trzeba osobno oceniać **znak skumulowanego ruchu** i **wielkość błędu**. Duża zgodność znaków przy zbyt małej amplitudzie albo gorszym MSE nie wystarcza do uznania prognozy za użyteczną.

Roboczy cel 85 procent dotyczy porównania znaków składowych $\widehat x(t+H)-x(t)$ oraz $x(t+H)-x(t)$ na wszystkich 128 kanałach. Znak kolejnego mikroprzyrostu jest inną miarą. Odsiew według rzeczywistego przyszłego ruchu musi być oznaczony jako warunkowa diagnostyka po ocenie, a nie online pewność prognozy. Nowy protokół PyKoopman zachowuje pierwotne progi epsilon i jawny aneks porównywalności metryki wszystkich 128; nie ukrywamy zmiany definicji po teście.

**Wyniki wcześniejsze:** przeszukano 99 konfiguracji na walidacji, po czym oceniono 16 wcześniej zamrożonych wariantów. Żaden nie osiągnął 85 procent w grupach fit, transfer lub wszystkich 12 obrazów dla sprawdzonych horyzontów 1–500. Około76 procent na H1 dla kontroli stałej prędkości było wynikiem krótkiego horyzontu, a nie dowodem85 procent ani dobrej prognozy H500.

**Nowy przebieg:** obliczenia PyKoopman z 30 fit i 4 holdout zakończono 2026-10-02 o 12:13:17. Wybrano regularyzację 0,01 w wewnętrznej walidacji, zapisano wspólny operator z prefiksu 500 i wykonano otwartą ocenę holdoutów. Raportowanie i audyt są w toku w chwili sporządzenia tego dokumentu; cel 85 procent nie został osiągnięty według przekazanego statusu. Szczegółowe nowe liczby należy odczytywać z zatwierdzonego raportu eksperymentu, nie dopisywać jako wynik przed jego zamknięciem.

**Warunek sensowności gałęzi prognozującej:** dopełnianie reprezentacji lub prowadzenie kolejnych decyzji prognozowanym celem wymaga odpowiednio sprawdzonej prognozy. Słaba prognoza ogranicza tę część metody. Nie unieważnia to osobno zweryfikowanej korzyści SD/retrieval względem losowego wyboru w wariancie bez prognozowanego celu. Są to różne pytania i różne wyniki.

Skończenie wymiarowy affine EDMD jest empiryczną aproksymacją. Nie dowodzi dokładnej globalnej linearyzacji uczenia ViT ani zamknięcia dynamiki w 128 kanałach. Nie obiecujemy sukcesu kolejnej rodziny modeli.

### 5.4. Aktualizacja rozmowy: NNDMD jest drugim wariantem, nie inną nazwą EDMD

Podczas rozmowy doprecyzowano różnicę między liniowym EDMD a neuronowym NNDMD. Użytkownik potwierdził, że chce **rzeczywisty PyKoopman NNDMD z wyuczanym encoderem, decoderem i macierzą ewolucji**, a nie wyłącznie wcześniej wykonany liniowy EDMD. Zachowujemy EDMD jako wykonany benchmark; nowy wariant neuronowy rozpoczął rzeczywisty lokalny trening GPU i nie ma jeszcze wyniku końcowego.

**Korekta użytkownika z 2026-10-02:** encoder NNDMD otrzymuje pełne 768 współrzędnych każdej cząsteczki, a decoder odtwarza pełne 768. J128 pozostaje zestawem identyfikacji obecności i dodatkowym zakresem oceny prognozy. „Cały latent” oznacza pełną reprezentację każdego patcha, a nie cały wektor wag ViT. Pierwotnie zaproponowano learned state 128; użytkownik następnie zastąpił tę propozycję pełnym learned state 768 i operatorem 768 × 768. Nie jest to podzbiór J. Wewnętrzny model ma postać

$$h=E_\theta(z),\quad z\in\mathbb R^{768},\quad h\in\mathbb R^{768},\qquad \widehat h_{k+1}=A_{NN}\widehat h_k,\qquad \widehat z_k=D_\phi(\widehat h_k)\in\mathbb R^{768}.$$

Zgodnie z najnowszą decyzją $h$ ma wymiar 768, a wyuczona macierz $A_{NN}$ ma 768 × 768, lecz jej osie **nie są automatycznie oryginalnymi makroosiami tożsamości**. To współrzędne wyuczanej przestrzeni dynamiki. Do oceny trajektorii użytkownika trzeba dekodować z powrotem do natywnego z768, a potem jawnie oceniać również z[J]. Nie wolno podpisywać kolumn $A_{NN}$ jako indywidualnych oryginalnych kanałów J.

Użytkownik dopuszcza w tym wariancie autoenkoder; nie jest to PCA i nie zastępuje nadzorowanego wyboru kanałów z kroku pierwszego. Funkcja celu ma obejmować wielokrokowy błąd MSE/RMSE prognozy pełnych 768 natywnych współrzędnych po dekodowaniu. Ocena J128 pozostaje osobna od błędu pełnych 768. Ostatecznie użytkownik zatwierdził cel treningowy MSE pełnych 768 z dodatkową, większą wagą błędu na J128; współczynnik tej wagi należy wybrać wyłącznie wewnętrznie przed testem. Decoder odtwarza wszystkie 768 kanałów, nie tylko 128. Propozycja learned state 128 została zastąpiona pełnym learned state 768; wyuczone współrzędne nadal nie są automatycznie oryginalnymi osiami. Nie stosujemy PCA; wykonany EDMD na J128 pozostaje zachowanym benchmarkiem. Samo dobre odtworzenie wejścia przez autoenkoder nie dowodzi dobrej przyszłej prognozy.

Dobór modelu neuronowego musi korzystać wyłącznie z dozwolonych prefiksów i walidacji, z zachowaniem czterech dotychczasowych holdoutów. Po originie model używa własnej wyuczonej ewolucji i decode; rzeczywiste przyszłe stany nie korygują trajektorii. Konkretny plan treningu tego modelu dynamiki i jego końcowy raport pozostają osobnymi artefaktami. Nie opisujemy jego przygotowania jako ukończonego testu ani osiągnięcia 85 procent.

### 5.5. Dlaczego obecny model otrzymuje pełne 768, a nie tylko J128

Wybór J128 wykonano na **pośrednim checkpointcie 500**, w zadaniu wspólnego rozpoznania oryginału i wersji LaMa w znanym ROI. Jest to zweryfikowany wynik tego konkretnego klasyfikatora, nie pełna, uniwersalnie potwierdzona definicja tożsamości i nie dowód wyłączności tych kanałów. Warmup i etap uczenia nie czynią tej reprezentacji kompletnym stanem procesu. Dobry wynik rozpoznania nie oznacza, że wybrane współrzędne same wystarczą do autonomicznej prognozy.

Self-attention miesza informacje **pomiędzy tokenami**. Nie oznacza to osobnego attention dla każdego piksela. Cząsteczki wskazane maską narzędzia mogą kodować również kontekst i tło pochodzące z innych tokenów. Maskowanie w analizie wybiera miejsca odczytu, ale nie odcina wcześniejszej wymiany informacji w sieci. Podprzestrzeń współrzędnych J nie jest przez to dowodem przyczynowej ani semantycznej wyłączności.

Pozostałe 640 kanałów tej samej cząsteczki mogą zawierać informacje istotne dla późniejszej zmiany J128. To uzasadnia przyjęcie przez użytkownika pełnego stanu 768 jako **założenia projektowego** następnego wariantu. Nie jest to twierdzenie, że J128 nigdy nie da się sensownie modelować. O tym, czy poszerzenie rzeczywiście pomaga, rozstrzyga odłożona ocena jakości prognozy.

Poprzednia wersja ograniczała stan do J128 i nie odpowiada obecnemu założeniu pełnego stanu. Była uproszczoną bazą porównawczą; nie wykazano matematycznego błędu samego algorytmu EDMD. Jej operator i wyniki pozostają zachowane. Obecna metoda to **NNDMD / neural Koopman z encoderem–decoderem**, nawet jeżeli w rozmowie użytkownik skrótowo nazywa ją EDMD. Kolejny agent nie powinien zastępować tego zamówienia samą liniową regresją 128 kanałów.

Ostateczny przebieg to: pełny natywny wektor 768 → encoder 768 → wyuczony stan 768 → wyuczana macierz A768 × 768 → decoder 768 → odtworzony pełny wektor 768; **dopiero potem** odczyt J128 i metryki prognozy tych wybranych współrzędnych. Pełny błąd rekonstrukcji/prognozy 768 oraz dodatkowo ważony błąd J128 są celami uczenia, z wagą wybieraną wyłącznie w wewnętrznej walidacji. Żaden PCA nie wybiera lub nie redukuje kanałów w tym wariancie.

Także pełne 768 współrzędnych tokenu nie są całym stanem uczenia ViT. Nie obejmują wszystkich wag sieci, bieżących statystyk BatchNorm, pamięci optymalizatora ani niezapisanych wejść treningowych. Nie dowodzimy więc dokładnego domknięcia dynamiki, Markowowskości ani gwarancji trafności. Brak kompletnej telemetrii i ograniczenia odłożenia danych pozostają obowiązujące. Cel ponad 90 procent rozpoznania obecności z kroku pierwszego i cel 85 procent znaku przyszłego przemieszczenia z kroku trzeciego pozostają oddzielnymi wymaganiami.

## 6. Krok czwarty: SD od czystego szumu prowadzone prognozowanym stanem

**Wymaganie użytkownika:** inicjalizować latent VAE SD czystym losowym szumem, bez kodowania prawdziwego zdjęcia jako kotwicy img2img. Wybrany scheduler określa rozkład i skalę inicjalizacji. Seed i budżet restartów są jawne. Generator podczas denoisingu dekoduje estymatę czystego obrazu, odczytuje jego reprezentacje bieżącym zamrożonym ViT i stosuje prowadzenie zgodności z prognozowanym pełnym stanem 768, z dodatkowym naciskiem na J128.

Brak zdjęcia w inicjalizacji nie znaczy braku źródła celu. Prognoza powstała dla konkretnego obrazu i jego cząsteczek. Trzeba określić origin/horyzont, cel per token, geometrię i korespondencję indeksów. Ewentualne maski warunkujące istniejący SD mogą pozostawać znanym warunkiem przestrzennym; nie są obrazem startowym. Nie podajemy prawdziwych przyszłych obrazów.

Wagi ViT i bufory BatchNorm są zamrożone. Gradient potrzebny do prowadzenia płynie przez decoder VAE i ViT względem generowanego obrazu/latentu, ale nie aktualizuje sieci ViT. Energia obejmuje pełne 768 oraz J128, np. ważony standaryzowany MSE per token. Skalowanie pochodzi z historycznej kalibracji. „Coraz bardziej zgodny” jest celem prowadzenia, a nie matematyczną gwarancją monotoniczności ani konwergencji.

### 6.1. Nowe 90 procent nie jest wcześniejszą accuracy

Użytkownik chce końcowo ponad 90 procent „pokrycia” J128. To **cel do uzgodnienia i operacjonalizacji**, nie dotychczasowy wynik. Nie jest równoważny 90 procent rozpoznania obecności ani 85 procent znaków prognozy. Nie wolno arbitralnie podpisać cosine 0,9 jako 90 procent zgodności obrazów.

Przed testem definiujemy standaryzowany/znormalizowany błąd i RMSE oraz oddzielnie cosine, tolerancje małych wartości celu i dokładny sposób wyliczania ewentualnego procentowego wskaźnika. Kryterium akceptacji i próg należy uzgodnić, a następnie zamrozić. Jeżeli stosujemy udział składowych mieszczących się w tolerancji, muszą być jawne tolerancje, wagi i postępowanie dla małych norm; same ilorazy względem niemal zerowych wartości nie są stabilne.

Budżet obejmuje maksymalne kroki denoisingu, korekty wewnętrzne i restarty. Próbę niespełniającą kryterium po wyczerpaniu budżetu odrzucamy i raportujemy. Końcowy RGB musi ponownie przejść przez ścisły preprocessing i zamrożony ViT. Dopiero ten wynik jest podstawą acceptance. Pośrednia estymata mogła spełnić próg, ale dalszy denoising lub kwantyzacja mogły go pogorszyć; nie gwarantujemy natychmiastowego zatrzymania po pośrednim spełnieniu ani zawsze udanej generacji.

### 6.2. Co robił poprzedni kod

`prowadzenie.py` obsługuje `start=None` jako czysty szum, ale faktyczny przebieg `petla4.py` przekazywał `start=kk['obraz576']` i `sila_startu=0.5`. Używał 30 kroków schedulera, trzech korekt gradientowych, dwóch końcowych kroków bez prowadzenia i wagi latentu 1. Energia łączyła kwadrat różnicy rzutu LDA z MSE standaryzowanego pooled ROI768 z crop176. Finalne „pokrycie” raportowano po harmonogramie; nie było procedury acceptance ponad 90 procent. Zachowujemy tę implementację jako źródło i wcześniejszy wariant, a nowy plan określa zmieniony start, cel tokenowy i końcowe kryterium.

### 6.3. Aktualny status i następna granica

Actual PyKoopman NNDMD rozpoczął lokalny trening GPU pierwszego kandydata LF5/beta1: encoder 768 → learned 768, A768 × 768, decoder 768. Wynik końcowy nadal nie jest znany. Udane dopasowanie generacji do celu nie dowodzi poprawności samego celu prognozy.

Po generacji kolejnym planowanym krokiem jest retrieval prawdziwych klatek: **cały obraz wygenerowany do całych obrazów puli**. Krok piąty opisano poniżej; nie zastępujemy go retrieval cropu narzędzia.

Szczegółowy plan poprawki: [SD od czystego szumu i końcowa ocena J128](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-49-00_sd_z_czystego_szumu_prowadzenie_prognozowanym_latentem_768.md).

## 7. Krok piąty: ranking pełnych obrazów z niewykorzystanej puli train

Zapytaniem jest **cały końcowy, zdekodowany obraz RGB z SD**. Kandydatem jest **cała prawdziwa klatka** dostępnej puli train, która jeszcze nie weszła do uczenia bieżącego ViT. Ranking obejmuje cechy całego obrazu, nie crop lub ROI narzędzia. Wynik rankingu nie oznacza procentu identycznych pikseli ani procentowej zgodności sceny.

Przed wyszukiwaniem tworzymy jawny zbiór eligible: wyłącznie train, prawidłowe i unikalne frame-ID, po usunięciu klatek już użytych do uczenia ViT oraz duplikatów. Utrzymujemy rejestr `used_train_pool` i wybranych frame-ID; klatka wykorzystana później do uczenia przestaje być eligible. Nie wolno wyszukiwać w val/test. Jeżeli brak kompletnych logów wcześniejszego użycia, należy wskazać brak i uzupełnić rejestr ze sprawdzalnego źródła; nie udawać, że dowolny obraz train jest na pewno niewykorzystany.

Obowiązkowy rdzeń to pełnoobrazowe cechy obecnego DINO i ranking cosine; FAISS oraz dodatkowe metody są opcjonalne. Wybór konkretnej metody i preprocessingu trzeba zapisać przed porównaniem. SSIM, inne metryki percepcyjne oraz adaptacja wskazanego przez użytkownika DETR-E pozostają opcjami porównawczymi, bez cichego uznania ich za wykonany domyślny wariant. Dla DETR-E nadal potrzebna jest weryfikacja źródła i definicji; sama relacja o DINO/SAM/klasteryzacji nie zastępuje kodu.

Feature preprocessing zapytania i kandydatów jest zgodny i obejmuje pełną klatkę: jawny resize/letterbox, rozdzielczość, kolory i normalizacja, bez ukrytego cropu narzędzia. Informacje o narzędziu, tle i układzie sceny mogą wpływać na pełnoobrazowy ranking. Zachowujemy deterministyczną kolejność kandydatów i regułę remisów oraz zapisujemy metrykę, scores, identyfikator zapytania, identyfikatory wybranych obrazów i status eligibility.

Najnowsze doprecyzowanie użytkownika ustala roboczy cosine>0,85 i adaptacyjnie 0–100 przyjętych obrazów zamiast wymuszonego stałego budżetu. Historyczny wariant 134 pozostaje oddzielną kontrolą, a porównanie zmiennej liczebności musi uwzględniać zarówno dane, jak i liczbę aktualizacji; szczegóły poniżej.

Dalsze uczenie ma otrzymać **wybrane prawdziwe klatki**, nie obrazy SD. Obraz SD jest zapytaniem do rankingu. Kroki szósty i siódmy doprecyzowano następnie poniżej.

Dotychczasowy retrieval DINO wykorzystywał crop256, a prowadzenie ViT crop176; SD dekodował pełną klatkę. Nowe żądanie zmienia retrieval na pełny obraz do pełnego obrazu. Nie zakładamy, że sama zmiana wyszukiwania automatycznie zapewnia dobrą prognozę lub poprawę uczenia. Dokumentacja opisuje plan; nowy runtime retrieval nie został w tym zadaniu zaimplementowany.

### 7.1. Aktualna minimalna wersja: pełnoobrazowy DINO i ranking

**Najnowsza decyzja użytkownika zastępuje wcześniejszą propozycję wymaganej bazy/FAISS.** Obowiązkowa wersja robocza kroku 5 to pełny obraz SD → pełnoobrazowe cechy obecnego DINO → ranking pełnych rzeczywistych klatek eligible, niewykorzystanych train. Zapytanie i kandydaci używają zgodnego, jawnego preprocessingu pełnych obrazów. Nie stosujemy cropu narzędzia.

FAISS, trwała baza wcześniej wyliczonych cech, dodatkowe modele, geometria, orientacja, rozmycie, jakość, SSIM i wielokryterialny reranking są **na razie opcjonalne** ze względu na koszt implementacji i obliczeń. Nie są warunkiem wykonania minimalnego wariantu. Nie deklarujemy zainstalowanego FAISS ani zbudowanego indeksu. FAISS jest indeksem wektorowym, nie modelem tworzącym cechy.

Jeżeli później wdrożymy bazę, może obejmować precompute DINO wszystkich aktualnie dostępnych eligible train wraz z frame-ID, source/film, maską/anotacją, współrzędnymi i orientacją narzędzia, cechami jakości, wersją preprocessingu oraz used_flag. Informacja o rozmyciu/jakości nie jest automatyczną karą: trudne przykłady mogą pomagać w uczeniu. Dodatki muszą mieć osobną ocenę; nie stają się cichym domyślnym wymaganiem.

### 7.2. Roboczy próg cosine i adaptacyjna liczba 0–100

Aktualnie przyjmujemy **cosine > 0,85** między pełnoobrazowymi cechami DINO. Wynik można prezentować jako `score = 100 × cosine`, z progiem `score > 85`. Jest to **umowny wskaźnik podobieństwa cech**, nie procent identycznych pikseli, prawdopodobieństwo poprawności ani skalibrowana procentowa zgodność obrazów. Nie potrzebuje fikcyjnej kalibracji na taki procent. Roboczy próg 0,85 pozostaje do oceny; pokazujemy rzeczywiste scores zamiast deklarować empiryczny sukces 85 procent.

Liczba nowych klatek jest adaptacyjna: **od 0 do maksymalnie 100** na zadeklarowaną jednostkę wyboru; 100 jest roboczym limitem, a nie wymaganą liczebnością. Najpierw filtrujemy qualifying: wyłącznie poprawne, unikalne eligible train, niewykorzystane wcześniej, z cosine>0,85. Następnie wybieramy najwyżej 100 według scores i deterministycznej reguły remisów. Nie dopełniamy do 100 ani do historycznych 134. Jeżeli żaden kandydat nie przechodzi progu, zapisujemy zero przyjętych rekordów. Nie obniżamy po cichu progu.

Przy wielu zapytaniach w rundzie deduplikacja obejmuje także wspólną pulę query: ten sam frame-ID nie trafia do zbioru dwa razy. Sposób łączenia rankingów i zakres limitu 100, np. na rundę zamiast na zapytanie, musi zostać jawnie zapisany przed wykonaniem; nie mnożymy niejawnie budżetu przez liczbę zapytań. Ewentualny backoff jest odrębnym, wcześniej zadeklarowanym wariantem, nie zmianą podstawowej reguły 0 rekordów.

Logujemy query-ID, selected frame-ID, cosine/score, próg, liczby eligible, qualifying, unikalnych i wybranych, wykluczenia used/duplikatów oraz status braku wyników. `used_train_pool` jest aktualizowany zgodnie z rzeczywistym wykorzystaniem klatek w uczeniu. Kandydaci val/test są zabronieni. Obraz SD jest wyłącznie zapytaniem; uczenie dostaje wybrane **prawdziwe klatki**.

Historyczne stałe 134 pozostają oddzielnym benchmarkiem. Porównanie z nowym zmiennym doborem wymaga wyrównania budżetu danych/aktualizacji lub krzywych jakości względem zarówno liczby przykładów, jak i aktualizacji. Dostępna pula oznacza klatki rzeczywiście dostępne w danym momencie; opcjonalny indeks nie może zawierać niedostępnych przyszłych klatek, przyszłych latentów ani val/test.

To aktualizacja planu, **bez implementacji bazy lub runtime retrieval**. Chronologia jest jawna: wcześniejsza propozycja precompute/FAISS i kalibracji composite score została zastąpiona tym prostszym wariantem DINO/cosine. Dodatki pozostają możliwymi późniejszymi eksperymentami.

## 8. Krok szósty: uczenie na rozszerzonej puli prawdziwych klatek i checkpoint

Po rankingu dokładamy wybrane **prawdziwe obrazy train wraz z ich prawdziwymi anotacjami** do dotychczasowej puli. Obrazy SD są zapytaniami, nie nowymi przykładami uczącymi. Stare obrazy i ich anotacje zachowujemy; nowe mają unikalne frame-ID i sprawdzony split. Nie dodajemy val/test. Liczba nowych klatek pozostaje adaptacyjna 0–100 zgodnie z krokiem 5; brak spełniających próg oznacza 0, bez dopełniania.

Przed uczeniem rundy zapisujemy niezmienny snapshot puli: wszystkie frame-ID, pochodzenie film/klatka, źródłowy hash obrazu i anotacji, split, oznaczenie old/new oraz hash całego manifestu. Zapisujemy zdarzenie dodania każdego obrazu: run/arm-ID, round, epoch lub pozycję między epokami, global_optimizer_step, timestamp oraz selection/query-ID, score i metodę wyboru. Zachowujemy źródłowe współrzędne bbox/maski i ich konwencję; transformacje augmentacji mają osobny zapis.

**Dodany do puli nie znaczy użyty do uczenia.** Loader/sampler ma logować image-ID rzeczywiście podane w każdym batchu, ich kolejność i powtórzenia, przypisane do rundy, epoki i globalnego kroku. Dla gradient accumulation rozróżniamy microbatch-ID i optimizer-step; krok optymalizatora liczymy przy faktycznej aktualizacji. Dzięki temu można sprawdzić, która nowa klatka po raz pierwszy trafiła do batcha, ile razy została pokazana i przy którym checkpointcie jej wpływ mógł już wystąpić.

Uczenie prowadzi do checkpointu oznaczonego run/arm/round-ID i global_optimizer_step. Checkpoint zawiera wagi, bufory BatchNorm, identyfikator/hash snapshotu trainpool, hash konfiguracji i mapę kroków oraz, jeśli umożliwia to runtime, stan optymalizatora, schedulera i AMP/scalera. Replay wymaga także stanów RNG i informacji o samplerze; sam seed i lista dodanych klatek nie wystarczają do dokładnego odtworzenia. Wymagania telemetrii z kroku 7 obowiązują **już podczas kroku 6**, nie dopiero po treningu.

## 9. Krok siódmy: walidacja, telemetria pochodzenia i kontrole porównawcze

### 9.1. Co zapisywać, żeby odtworzyć chronologię klatek

Planowany zapis obejmuje snapshoty trainpool oraz append-only zdarzenia JSONL. Minimalny schemat oddziela `selection_events.jsonl` od `batch_samples.jsonl` i `checkpoint_events.jsonl`:

- **Selection/add:** run/arm-ID, source dataset/film/frame-ID, ścieżka i sourcehash obrazu/anotacji, split, old/new, selected_round, added_at, epoch/global_optimizer_step w chwili dodania, SD query-ID, metoda i wersja preprocessingu, cosine/score/prog, rank oraz status accepted/rejected/duplicate/used. Zapis obecności w puli nie jest zapisem ekspozycji na gradient.
- **Actual batch:** round, epoch, global_optimizer_step, microbatch/sample-order, rzeczywiste image-ID z powtórzeniami, seed samplera i dane augmentacji. Logować seed oraz parametry losowych cropów, resize, flipów i przekształceń koloru/geometrycznych, w miarę dostępności runtime; zachować powiązanie transformacji obrazu i anotacji. Sam frame-ID nie wyjaśnia, co rzeczywiście zobaczył model.
- **Checkpoint/replay:** checkpoint-ID/path/hash, run/arm/round, global_optimizer_step, zakres zakończonych batchów i hash puli, stan RNG Python/NumPy/Torch CPU/CUDA i workerów/samplera, optimizer/scheduler/scaler i konfiguracja środowiska, o ile są zapisywane. Niezapisane elementy jawnie oznaczyć; nie obiecywać bitowego replay bez kontroli deterministyczności.

Indeks kroków pozwala odtworzyć film i frame w chronologii źródła oraz chronologię wykorzystania w uczeniu. Nie utożsamiamy porządku klatek filmu z losową kolejnością samplerów. Sumujemy osobno prezentacje old/new, liczbę unikalnych klatek i wszystkie powtórzenia, per runda/epoka/krok i narastająco. Regułę `used_train_pool` aktualizujemy po rzeczywistym użyciu, a klatki już zarezerwowane/dodane oznaczamy dodatkowo, aby kolejna runda nie dodała ich ponownie.

Kontrole integralności sprawdzają: brak val/test w pool i batchach, każdy sampled-ID należy do snapshotu aktywnego przed batch-em, source/anotacje są zgodne, nowe klatki nie występują przed added_at, brak podwójnego dodania, wszystkie checkpointy mają jednoznaczny krok i powiązanie z puli. Ważna jest właśnie odpowiedź „która klatka, kiedy dodana, kiedy rzeczywiście użyta i ile razy”, nie tylko liczba wybranych obrazów.

### 9.2. Walidacja przed i po rundzie

Na stałym zbiorze validation, z tym samym preprocessingu i protokołem, oceniamy model **przed i po każdej rundzie**. Zapisujemy loss oraz mAP i jego ustalone warianty/progi IoU, liczbę przykładów, konfigurację decode/NMS i metryki per obraz oraz agregaty. Checkpoint-ID i globalny krok są częścią wyniku. Nie przypisujemy zmiany metryki do nowych danych bez kontroli ilości wykonanych aktualizacji.

Oddzielny końcowy test pozostaje niewykorzystywany do doboru klatek, progów, hiperparametrów i wyboru modelu. Obrazy validation/test nigdy nie są kandydatami retrieval ani danymi uczącymi. Wszystkie decyzje adaptacyjne i momenty odczytu validation trzeba logować, aby ograniczenie końcowego testu było sprawdzalne.

Datowany plan porównania: [standardowy ViT i pipeline](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_13-02-43_test_standardowego_vit_i_pipeline_doboru_klatek.md).

### 9.3. Test zwykłej sieci i pipeline: konkretne ramiona

Celem kroku 7 jest kontrolowany test, czy metoda poprawia uczenie detektora, a nie sama korelacja wysokiego score retrieval z wynikiem. Telemetria jest obowiązująca już w kroku 6. Każde ramię ma oddzielny run i checkpointy; nie wykonujemy kolejnych metod na tych samych już zmienionych wagach jako rzekomego porównania.

| Ramię | Dobór danych i znaczenie |
|---|---|
| A — standardowy ViT | Z góry ustalony losowy podzbiór train, dostępny od początku; zwykły shuffle i mini-batche, bez adaptacyjnego doboru. Docelowa liczba unikalnych obrazów dopasowana do budżetu F lub jawnie oznaczona jako kontekst przy innym budżecie. |
| B — progressive uniform random | W każdej rundzie losujemy bez zwracania z eligible train dokładnie n_r nowych klatek, gdzie n_r jest rzeczywistą zaakceptowaną liczbą F w sparowanym seedzie, 0–100. Nie dobieramy n_r do wyników B. |
| C — sequential | Dokładnie n_r kolejnych klatek według wcześniej ustalonego porządku filmów i frame-ID; przestrzegamy eligibility, deduplikacji i wykluczeń. Nie zmieniamy filmu/kolejności po wynikach. |
| D — direct real retrieval | Pełnoobrazowy DINO retrieval z rzeczywistego zapytania, bez SD i bez prognozy. Kotwice wybieramy regułą niezależną od prognozy, zamrożoną przed runem; n_r i budżet dopasowane do F. To brakująca wcześniej kontrola korzyści samego real retrieval. |
| E — SD current target, no forecast | Czysty szum SD i cel z obecnego stanu, bez prognozy. Dobór kotwic również niezależny od prognozowanego ruchu. Reszta prowadzenia/retrieval i n_r jak F. Historyczne „SD bez prognozy” z kotwicami wybieranymi za pomocą prognozy nie jest tą czystą ablacją. |
| F — pełny pipeline | NNDMD pełne 768 → prognozowany stan → SD z czystego szumu → pełnoobrazowy DINO ranking eligible train. Cosine>0,85, najwyżej 100, bez dopełniania. Wybrane prawdziwe obrazy uczą ViT. |

**Etap minimalny: A/B/F.** Najpierw odtwarzalny pilot porównawczy z uzgodnionym budżetem, później planowane 10 sparowanych seedów. Liczba 10 jest planem, nie liczbą wykonanych replik. Etap następny, gdy zasoby pozwolą: D i E jako kluczowe ablacje, następnie C. Nie uruchamiamy teraz wszystkich ramion ani nowego treningu ViT; zapisujemy protokół do wykonania.

Kontekstowo można dołączyć checkpoint bez dalszego uczenia/no-add, standardowe pretrained initialization oraz zwykłe uczenie na całym dostępnym train. Uczenie na całym train ma większy budżet etykiet/danych, a pretrained może korzystać z innej inicjalizacji; **nie są głównym równobudżetowym dowodem przewagi**. Nie mieszać takich liczb z A/B/F bez oznaczenia.

### 9.4. Warunki porównywalności i zamrożenie

W głównych ramionach zachowujemy tę samą architekturę, stan inicjalny i wspólny warmup checkpoint, sparowane seedy, AdamW/lr/scheduler, batch/gradient accumulation, augmentacje, kryteria detekcji, train-only pulę i niezmienny podział filmów. Te same seedy nie gwarantują identycznych augmentacji przy różnej kolejności ID; runtime zapisuje rzeczywiste parametry. Validation jest stałe, a final test odłożony po filmach.

Budżet optimizer-step jest wspólny. B/C/D/E dopasowują rzeczywistą liczebność nowych klatek F per runda, a logi pokazują prezentacje old/new oraz unikalne obrazy. Gdy F dodaje 0, kontrola nie dodaje nowych obrazów; wszystkie ramiona przestrzegają tej samej wcześniej ustalonej reguły aktualizacji rundy. Arm A ma stały zestaw od początku i inny harmonogram dostępności: to standardowa kontrola konwencjonalna, a B jest główną kontrolą efektu adaptacyjnej selekcji przy dopasowanej dostępności/liczebności.

Czas mierzymy end-to-end: trening ViT, NNDMD/prognoza, generacja SD, DINO/index/retrieval, walidacja i I/O. Oprócz optimizer-step raportujemy GPU-hours/wall-clock i nakład danych. Dla ustalonego wcześniej celu mAP mierzymy aktualizacje i czas do jego osiągnięcia; jeśli nieosiągnięty, zapisujemy censored/budget_exhausted, nie interpolujemy dowolnego sukcesu.

Przed runami zamrażamy budżet, liczbę rund/aktualizacji, harmonogram oceny, próg retrieval i SD acceptance, metryki, primary contrast F−B, reguły A i kotwic D/E, seedy oraz zasady awarii/zero-add. Nie usuwamy nieudanych runów. Final test odczytujemy raz po zamrożeniu protokołu i checkpoint-selection; nie używamy go do dalszego doboru metody.

### 9.5. Metryki, learning curves i niepewność

Główna miara to **sparowana różnica znormalizowanego pola pod krzywą mAP względem optimizer-step**, F−B, na wspólnym zakresie kroków i ustalonych punktach walidacji. AUC odnosi się tu do krzywej uczenia mAP, nie do AUROC klasyfikatora obecności. Konkretna definicja mAP, np. AP50 oraz dodatkowo AP50:95, decode/NMS i harmonogram punktów muszą być ustalone przed runem. Oprócz primary contrast pokazujemy F−A oraz późniejsze D/E/C bez wybierania najkorzystniejszego kontrastu po teście.

Jednostką replikacji jest **cały seed-run**. CI liczymy bootstrapem kompletnych sparowanych seedów, zachowując wspólne różnice ramion; tokeny, obrazy i rundy nie zastępują niezależnych treningów. Przy pilocie z małą liczbą seedów wynik jest opisowy. 10 seedów to cel planu; nie przedstawiamy niewykonanych replik ani przedziałów jako danych. Ewentualny bootstrap filmów oceny jest dodatkową niepewnością zbioru, nie zamiennikiem zmienności treningu.

Raportujemy final mAP, validation loss i training loss, precision/recall, IoU oraz FP/FN. Krzywe pokazujemy względem optimizer-step, liczby unikalnych obrazów, **łącznej liczby prezentacji** i runtime. Zmiana loss na zmieniającej się puli nie wystarcza do dowodu poprawy generalizacji; służy diagnostyce obok stałego validation.

Przed pomiarem ustalamy warstwy trudności validation: kąty/orientacje narzędzia, rozmycie, wielkość/położenie i sceny tła. Raportujemy wyniki, liczebności oraz FP/FN na tle; bez tworzenia warstw według tego, gdzie metoda akurat wygrywa. Monitorujemy rozkład wybranych klas/filmów/kątów i ekspozycje old/new: similarity sampling może zubożyć różnorodność. Nie gwarantujemy przewagi pipeline ani uzyskania zakładanego mAP.

### 9.6. Pierwotne źródła uzasadniające kontrole

Yoo i Kweon porównują dobór aktywny z random sampling, utrzymując te same hiperparametry modeli i raportując powtórzenia. To uzasadnia potrzebę silnej losowej kontroli, a nie przewagę naszego pipeline. Odczyt: artykuł HTML i fragmenty sekcji 4 porównań/obiektów detekcji; CVF PDF zwrócił 403, więc użyto autorskiego PDF i arXiv. [Learning Loss for Active Learning, CVPR 2019](https://arxiv.org/html/1905.03677v1), [autorski PDF](https://dgyoo.github.io/papers/cvpr19.pdf).

Emam i współautorzy pokazują, że dobór aktywny może nie przewyższać random, a imbalance wybranych klas jest istotnym problemem przy ImageNet. Uzasadnia to kontrolę rozkładu próbek i jakości na stałych warstwach; nie dowodzi takiego samego efektu dla naszej detekcji narzędzia. Odczyt: abstract, opis protokołu i sekcje 5.1.1–5.1.2. [Active Learning at the ImageNet Scale](https://arxiv.org/html/2111.12880v1).

Szczegółowy plan telemetrii i walidacji: [datowana poprawka](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-59-31_telemetria_uzytych_klatek_uczenie_i_walidacja_rund.md).

## 10. Pytania i doprecyzowane odpowiedzi

### 1. Za co odpowiada pierwszy krok?

Za wybór oryginalnych kanałów, których wspólny wzorzec rozpoznaje obecność narzędzia w znanym ROI na zamrożonym checkpointcie 500. Nie prognozuje przyszłego ruchu i nie nadaje każdej cząsteczce odrębnej tożsamości.

### 2. Jak wykonuje się ten krok?

Porównuje się 400 par oryginał–syntetyczne usunięcie. Obie wersje przechodzą przez ten sam zamrożony ViT, z tych samych indeksów oryginalnej maski oblicza się coverage-ważoną reprezentację 768D, a ranking i logistyczny probe dopasowuje się tylko na train. Validation wybiera konfigurację; test służy jednorazowej ocenie.

### 3. Na jakiej podstawie wyznaczono 128?

Według train-only rankingu powtarzalnej sparowanej różnicy oraz walidacji zestawów o określonej liczebności.128 było najmniejszą sprawdzoną liczebnością przekraczającą90 procent na validation. Nie było doborem 90 procent energii PCA ani dowodem, że nie istnieje inny mniejszy zestaw.

### 4. Czy cząsteczki się ustawiają albo zanikają po usunięciu narzędzia?

Zmieniają wartości reprezentacji, ale nie znikają z siatki tokenów. Poszczególne kanały mogą rosnąć lub maleć. Wspólny układ osi pozostaje ten sam; klasyfikuje wzorzec wielu wartości, nie wymaganie, aby wszystkie kanały zachowywały się jednakowo.

### 5. Czy użyto maski i dokładnie tych samych patchy?

Tak. Oryginalna maska określa indeksy i coverage użyte dla obu członów pary. Po usunięciu narzędzia mierzymy te same pozycje obrazu. Jest to identyfikacja pośredniczona przez średnią ROI, a nie osobne rozpoznanie każdego patcha. Sama maska nie eliminuje kontekstu przeniesionego przez attention.

### 6. Czy mamy pewność, że identyfikacja jest dobra?

Mamy konkretny wynik 92,8274 procent i przedział 95 procent [88,17%;96,99%] w opisanym zadaniu. Użytkownik zaakceptował go do dalszej pracy. Nie daje to 100 procent pewności, uniwersalnej tożsamości narzędzia ani pewności identycznej jakości na innych klasach, obrazach i checkpointach.

### 7. Na czym polega drugi krok?

Na pomiarze tych samych 128 współrzędnych tych samych cząsteczek tego samego obrazu pod kolejnymi stanami treningu ViT.34 obrazy przechodzą przez 101 checkpointów, a każda trajektoria jest zachowana osobno.

### 8. Czy zmieniają się parametry kanałów?

W trajektorii zmieniają się wartości współrzędnych tokenów, nie numery i definicje wybranych osi. Jednocześnie w źródłowym uczeniu zmieniają się parametry sieci i statystyki BatchNorm, co powoduje zmianę odczytanych reprezentacji. Nie wyznaczamy nowej bazy na każdym checkpointcie.

### 9. Z czego wynika różnica między kolejnymi stanami?

Z odczytów przy dwóch kolejnych checkpointach, dla identycznego obrazu i tokenu: stan po następnych 10 aktualizacjach minus stan bieżący. Nie jest to porównanie kolejnych klatek filmu ani bezpośredni pomiar wszystkich pojedynczych aktualizacji między zapisami.

### 10. Czy właśnie tutaj działa EDMD?

Odczyt reprezentacji i różnice wykonuje pipeline pomiarowy. EDMD działa później na parach stanów, dopasowując wspólną macierz 128 ×128 i bias 128. Dopiero wielokrotne zastosowanie zamrożonego operatora do jego własnych przewidywań jest prognozą. Późniejsze doprecyzowanie użytkownika dotyczy drugiego wariantu: NNDMD z encoderem–decoderem, opisanego w sekcji 5.4. Dwa rzeczywiście zmierzone stany i ich różnica same w sobie nie są prognozą.

## 11. Źródła i odtwarzalne artefakty

W poniższych plikach znajdują się definicje, pełna lista128 osi w obu konwencjach indeksowania, wyniki, maski i pochodzenie danych. Indeks1-based jest zawsze indeksem0-based powiększonym o1; nie są to dwie różne listy kanałów.

- [Wyniki wyboru kanałów i rozpoznania](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/WYNIKI.md).
- [Kod train-only rankingu i probe](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/wybor_kanalow.py).
- [Zamrożony wybór walidacyjny](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/WYBOR_WALIDACJA.json).
- [Jednorazowa ocena obecności na teście](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/OCENA_OBECNOSCI_TEST.json).
- [Pełna lista natywnych kanałów i parametrów probe](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/KANALY_MODEL.npz).
- [Dataset 400 par i ograniczenia inpaintingu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_test_orientacji_narzedzia_400/README.md).
- [Podział filmów identyfikacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/PODZIAL.json).
- [Kod surowego poolingu przed wcześniejszym U](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_langevin_kierunki_tozsamosci/identyfikacja500.py).
- [Manifest 34 obrazów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/MANIFEST.json).
- [Status i parametry gotowych danych](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE.json).
- [Weryfikacja zgodności ekstrakcji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_WERYFIKACJA.json).
- [Plan rozszerzenia do pełnych 768](<F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/PLAN_EKSTRAKCJA_768.md>).
- [Zakończona ekstrakcja pełnych 768: metadane](<F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_768.json>).
- [Końcowa weryfikacja pełnych 768](<F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_768_WERYFIKACJA.json>).
- [Schemat pełnych reprezentacji NNDMD](<F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_768_SCHEMA.md>).
- [Schemat danych](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_SCHEMA.md).
- [Kod ekstrakcji przez checkpointy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/ekstrakcja.py).
- [Protokół właściwego dopasowania PyKoopman](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/PROTOKOL_MODELU.md).
- [Wspólny operator z prefiksu 500](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/MODEL_PREFIX_500.npz).
- [Wcześniejszy przegląd 99 konfiguracji i 16 wariantów testu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/PODSUMOWANIE.md).

W dokumentacji należy zachować rozdzielenie: **rozpoznanie obecności**, **pomiar rzeczywistych trajektorii**, **identyfikacja operatora** oraz **ocena otwartej prognozy**. Wynik jednego z tych zadań nie zastępuje wyniku pozostałych.

### Korekta zakresu danych NNDMD — 2026-10-02T12:32:57+02:00

Uruchomiono ponowną ekstrakcję pełnych 768 dla niezmienionych 34 obrazów i 101 checkpointów. Docelowy plik LATENT_768.npy ma kształt (101,34,600,768), float32. Wcześniejsze dane 128 pozostają bez zmian. Status zakończenia i SHA będą zapisane w DANE_768.json po kontroli finite, zgodności ze źródłem i bitowej zgodności wycinka J z wcześniejszą ekstrakcją. To przygotowanie danych, nie wynik treningu NNDMD.

### Ostateczne doprecyzowanie wymiaru NNDMD — 2026-10-02T12:36:57.193638+02:00

Użytkownik zastąpił propozycję wyuczonego stanu 128 pełnym stanem 768: encoder 768 → 768, wyuczana macierz Koopmana 768 × 768, decoder 768 → 768. Nie wykonujemy redukcji wymiaru ani PCA. J128 służy końcowej ocenie wybranych współrzędnych; pełna funkcja celu zachowuje również dodatkową wagę J128 wybieraną wewnętrznie. Pełne dane są GOTOWE: LATENT_768.npy (101,34,600,768), wszystkie wartości finite, poprzedni J128 bez zmian. SHA256: `91a64e28612e2806cc203f8fc194b55cac4e7bbc8df7f5bbfee0ef1a377da140`. Zgodność source J128 maxabs 2,265e−6, pełne 768 maxabs 2,384e−6; wszystkie kontrole mieszczą się w 1e−5. To zakończenie pomiaru, nie wynik modelu neuronowego.

### Osobna notatka luk i możliwych poprawek — 2026-10-02T13:06:44+02:00

[Hipotezy, minimalne kontrole i rozszerzenia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_13-06-44_luki_i_poprawki_do_obecnego_pipeline.md) są oddzielną notatką do sprawdzenia, nie nowym krokiem 8 ani zmianą obowiązującej specyfikacji. Nie oznaczają uruchomienia dodatkowych testów/treningu.

### Korekta doboru i podział 90/10 NNDMD — 2026-10-02T14:12:41+02:00

Użytkownik wymaga obecnie, aby **wszystkie 30 obrazów FIT należały do dokładnych 1500 zdjęć źródłowego treningu ViT**. W poprzednim zestawie było 25/30 takich obrazów. Osobny dataset BASE1500_CORRECTED wymienia tylko dawne FIT sloty 0,2,3,5,6 na pięć nowych klatek z pięciu innych filmów tej dokładnej bazy, wykluczając wszystkie dotychczasowe filmy, w tym filmy holdout. Stare dane i wyniki pozostają zachowane.

Nowy plan NNDMD identyfikuje model na prefiksie 0–900: 91 checkpointów i 90 przejść na każdy z 30 fit. Przyszłe 900–1000 to 10 przejść do otwartej oceny, bez podawania prawdziwych przyszłych stanów. Szczegółowy wewnętrzny wybór NN i zamrożenie określa jego osobny protokół; dokumentacja nie deklaruje ukończonego nowego treningu lub sukcesu. Ten etap nie jest historycznym modelem prefix500.

Cztery dotychczasowe holdouty 8–11 pozostają z identycznymi frame-ID. Ich przynależność do exactbase1500 jest osobno sprawdzona: holdout 8 (TR_0134_S1_P03/00000341) należy do bazy; holdout 9 (TR_0132_S1_P03/00000916), 10 (TR_0080_S1_P03/00000686) i 11 (TR_0043_S1_P03/00001826) nie należą. Są to nadal odłożone obrazy dynamiki, wcześniej oglądane, nie cztery nowe nietknięte testy ani cztery obrazy exactbase.

Dane pięciu nowych obrazów zapisujemy osobno jako LATENT_NOWE5_768.npy (101,5,600,768). Pozostałe 29 slotów mapujemy do wcześniejszego LATENT_768.npy bez nadpisania lub kopiowania. MANIFEST zawiera dokładne frame-ID/hash, base1500_index i cache_sources. Logiczny dataset ma nadal (101,34,600,768), fit 30/holdout 4 i tę samą siatkę/tokeny 768.

### Aktualna autoryzacja exact1500 i batch 32 obrazów — 2026-10-02T14:36:50+02:00

Użytkownik wznowił obliczenia, rozszerzając modelowanie do wszystkich **1500 dokładnych obrazów źródłowego treningu ViT**, pełnych 600 tokenów każdego i 768 kanałów. Logical batch to 32 obrazy, nie 32 tokeny; memory microbatch/gradient accumulation zachowuje tę semantykę. Learned state i macierz oraz nieliniowy encoder/decoder mają 768 wymiarów.

Sliding window ma 10 przejść checkpointów, czyli 100 aktualizacji ViT. Fit checkpointy 0..90 oznacza optimizer-steps0..900, test 90→100 tofuture900→1000. To przyszłość tych samych znanych 1500 obrazów, nie holdout obrazowy. Stare 34/30 obrazy i częściowe 5 nie są obecnym datasetem. COCO source1500 ma bbox, bez segmentacji; primary lossall600 równomiernie, a J128 oceniamy po decode.

Źródła 1500 verified; pełny dump 279GB zastępuje ograniczony 11-checkpoint cache na dysku, wszystkie pełne dane dostarczane strumieniowo bez subsamplingu. Source ViT frozen, checkpointy/BN są tylko odczytywane. [Aktualny protokół danych](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_nndmd_1500_obrazow_768/PLAN_DANYCH.md) i [manifest exact1500](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_nndmd_1500_obrazow_768/MANIFEST_1500.json) określają provenance. Agent NN prowadzi osobny model/protokół; wyniku treningu 1500 jeszcze nie ma.


## Aktualizacja 2026-10-02: rozszerzony historyczny stan NNDMD na Edenie

Na aktualne polecenie użytkownika lokalny wariant i dalsze lokalne benchmarki zostały odłożone. Nowy eksperyment [maly_vit_nndmd_1500_expandedstate_eden](experiments/maly_vit_nndmd_1500_expandedstate_eden/PLAN_EDEN.md) obejmuje dokładne 1500 źródłowych obrazów ViT, pełne 600×768 tokenów, historyczne przyrosty latentu i podsumowania zmian wag/BN oraz faktycznie zapisanych momentów AdamW. Zapisy AdamW występują co 100 aktualizacji, dlatego pomiędzy nimi wejście ma ostatnią wcześniejszą obserwację i jej wiek; nie interpoluje z przyszłego checkpointu. Historyczne zmiany wag nie są gradientami. Pełny stan uczenia nadal nie jest obserwowany; przyszły aux jest prognozą modelu.

Batch ma 32 obrazy; okno ma 10 przejść checkpointowych. Inner fit 0..800, walidacja 800->900, refit 0..900, test 900->1000 na tych samych znanych obrazach. Uruchomienie jest autoryzowane na Edenie po audycie kodu; do chwili zapisu tej aktualizacji nie ma zgłoszonego joba ani nowych wyników. Rolę konkretnego statusu pełni submit_status.json w nowym eksperymencie.


2026-10-02, 15:12:03 Europe/Warsaw: Slurm przyjął job 1819498 rozszerzonego NNDMD 1500. 12 testów runtime i 16 hashów uploadu przeszły; audyt przed jobem PASS. Stan na zakończenie wdrożenia: PENDING (Resources), 0 kroków optymalizatora. Aktualny status w [submit_status.json](experiments/maly_vit_nndmd_1500_expandedstate_eden/submit_status.json). To potwierdzenie zgłoszenia, nie wynik ani rozpoczęty trening.
