# SD z czystego szumu: prowadzenie prognozowanym stanem cząsteczek 768D

Utworzono: 2026-10-02T12:49:00+02:00. Strefa: Europe/Warsaw. Status: wymagania i plan przyszłego eksperymentu; bez implementacji, generacji ani wyników w tej poprawce.

## Wymaganie użytkownika

Generacja ma zaczynać się od **czystego losowego szumu w przestrzeni VAE**, zgodnie z rozkładem i skalą wybranego schedulera SD. Nie kodujemy prawdziwego zdjęcia do latentu startowego i nie stosujemy img2img kotwiczonego w takiej klatce. Zapisujemy seed każdego startu/restartu.

Podczas iteracyjnego denoisingu dekodujemy estymatę czystego obrazu, podajemy ją do bieżącego zamrożonego ViT i prowadzimy generację w kierunku prognozowanego stanu. Porównanie obejmuje pełne 768 współrzędnych ustalonych cząsteczek, ze szczególnym naciskiem na J128 związane z rozpoznawaniem obecności. Sam cel pochodzi z modelu prognozy; nie powstaje z samego szumu.

Brak zdjęcia jako inicjalizacji nie usuwa potrzeby wskazania obserwowanej historii źródłowego obrazu, originu/horyzontu, celu dla każdej cząsteczki oraz zgodności pozycji tokenów. Model prognozy pracował na indywidualnych obrazach i ich cząsteczkach. Należy jawnie ustalić geometrię, siatkę, indeksy i ewentualne maski/warunki przestrzenne. Jeżeli użyjemy istniejącego SD warunkowanego maskami, maska nadal może być wejściem warunkującym; nie jest jednak zdjęciem img2img. Jej pochodzenie musi być znane w chwili prognozy, bez przyszłych prawdziwych obrazów.

## Stan wcześniejszej implementacji

`prowadzenie.py` już obsługuje `start=None`: wtedy rozpoczyna od losowego szumu. Faktycznie wykonana gałąź `petla4.py` przekazywała jednak `start=kk['obraz576']`, czyli kotwicę prawdziwej klatki. Parametry obejmowały `sila_startu=0.5`, 30 kroków schedulera, trzy wewnętrzne kroki gradientowe, dwa końcowe kroki bez prowadzenia i wagę latentu 1.

Stara energia była sumą `(s−s_cel)^2` na kierunku LDA oraz MSE standaryzowanej pełnej reprezentacji 768D, **po poolingu ROI z crop176**. Końcowe `pokrycie` było raportowanym wynikiem po stałym harmonogramie, a nie testem akceptacji ponad 90 procent. Nie utożsamiamy go z nowym wymaganiem dopasowania J128 każdej ustalonej cząsteczki. Stary kod jest dostępny; ta poprawka zmienia wymagany wariant wykonania i ocenę, nie twierdzi braku istniejącej dyfuzji.

## Plan prowadzenia i oceny

1. Zamrozić definicję celu: prognozowane pełne 768 per token, J128, zgodność indeksów i warunkowanie przestrzenne. Ocenić wcześniej wiarygodność prognozy; dopasowanie obrazu do błędnego celu nie naprawia modelu dynamiki.
2. Zamrozić bieżący checkpoint ViT, preprocessing, miejsce odczytu i skalowanie. Użyć eval oraz sprawdzenia niezmienności wag i buforów BatchNorm. W prowadzeniu gradient ma przepływać przez decode i ViT do obrazu/latentu dyfuzji, **bez aktualizacji parametrów ViT**.
3. Ustalić energię $L=\lambda_{full}L_{768}+\lambda_J L_{J128}$, np. coverage-ważony standaryzowany MSE per token. Skale pochodzą tylko z historycznej kalibracji; podłoga skali i tolerancja dla małych wartości celu są ustalone przed końcową oceną. Cosine jest odrębną diagnostyką kierunku, a nie zamiennikiem zgodności amplitudy.
4. Na każdym zadeklarowanym kroku prowadzenia obliczać estymatę czystego latentu SD, dekodować ją i mierzyć aktualne tokeny ViT. Gradient energii koryguje próbkę według jawnego harmonogramu siły prowadzenia. „Coraz bardziej zgodny” opisuje cel optymalizacji, **nie gwarantuje monotonicznego spadku błędu** w każdym kroku.
5. Po całym harmonogramie dekodować końcowy RGB, zapisać go w docelowej postaci i wykonać ścisły końcowy encode ViT tą samą drogą co pomiar referencyjny. Końcowa kwantyzacja, resize i końcowe denoising mogą zmienić wynik. Akceptacja dotyczy tego obrazu, nie wcześniejszej estymaty czystego obrazu.
6. Odrębnie zapisać nieudane próby. Budżet ma określać maksymalną liczbę kroków, wewnętrznych poprawek oraz restartów. Jeśli końcowe kryterium nie jest spełnione, wynik otrzymuje status rejected/budget_exhausted. Nie zapewniamy konwergencji ani obrazu spełniającego cel przy każdym seedzie.

## Co ma znaczyć „ponad 90 procent pokrycia J128”

To **nowy cel użytkownika do operacjonalizacji**. Nie oznacza 90 procent accuracy rozpoznania obecności z kroku pierwszego, 85 procent znaków prognozy ani zmierzonego sukcesu SD. Cosine 0,9 nie jest automatycznie90 procent zgodności obrazu lub pokrycia reprezentacji.

Przed testem trzeba uzgodnić standaryzowany/znormalizowany błąd, RMSE, cosine oraz ich progi i jawny sposób przeliczenia na ewentualny wskaźnik procentowy. Uwzględnić normy i wartości celu bliskie zeru: nie dzielić bez kontroli przez bardzo małe współrzędne. Ewentualna metryka udziału składowych w tolerancji wymaga ustalonej tolerancji bezwzględnej/względnej i wag. Definicja akceptacji ma zostać zamrożona przed oceną, zamiast arbitralnego uznania dowolnej metryki 0,9 za 90 procent.

Pośrednie spełnienie progu nie upoważnia do oznaczenia końcowego obrazu jako zaakceptowanego: kolejne kroki mogą odejść od celu. Reguła zatrzymania i końcowe potwierdzenie muszą być zgodne ze schedulerem i dotyczyć finalnego RGB.

## Status prognozy i dalsza granica

Actual PyKoopman NNDMD jest obecnie trenowany lokalnie na GPU: encoder pełne 768 → learned 768, A768 × 768, decoder pełne 768. Pierwszy kandydat ma LF5 i beta1. J128 jest oceniany po decode oraz dodatkowo ważony w loss. **Brak jeszcze końcowego wyniku NN**; poprawka nie deklaruje trafnej prognozy ani spełnionego celu SD.

Po zaakceptowanej generacji planowany krok piąty wyszuka prawdziwe klatki metodą **pełny obraz do pełnego obrazu**. Wymaganie retrieval rozszerzono poniżej; wcześniejszy plan pozostaje źródłem metod porównawczych.

## Źródła

- [Pipeline kroków 1–4](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md).
- [Generator i energia prowadzenia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_petla_sd_rownanie/kod/prowadzenie.py:58).
- [Losowy start i gałąź img2img](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_petla_sd_rownanie/kod/prowadzenie.py:95).
- [Faktyczne parametry wcześniejszego prowadzenia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_petla_sd_rownanie/kod/petla4.py:51).
- [Faktyczny start od kotwicy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_petla_sd_rownanie/kod/petla4.py:344).
- [Plan pełnoobrazowego retrieval](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-01_14-35-31_wyszukiwanie_pelnej_sceny_sd_w_pelnych_klatkach.md).

## Aktualizacja: krok piąty — 2026-10-02T12:50:12+02:00


Zapytaniem jest **cały końcowy, zdekodowany obraz RGB z SD**. Kandydatem jest **cała prawdziwa klatka** dostępnej puli train, która jeszcze nie weszła do uczenia bieżącego ViT. Ranking obejmuje cechy całego obrazu, nie crop lub ROI narzędzia. Wynik rankingu nie oznacza procentu identycznych pikseli ani procentowej zgodności sceny.

Przed wyszukiwaniem tworzymy jawny zbiór eligible: wyłącznie train, prawidłowe i unikalne frame-ID, po usunięciu klatek już użytych do uczenia ViT oraz duplikatów. Utrzymujemy rejestr `used_train_pool` i wybranych frame-ID; klatka wykorzystana później do uczenia przestaje być eligible. Nie wolno wyszukiwać w val/test. Jeżeli brak kompletnych logów wcześniejszego użycia, należy wskazać brak i uzupełnić rejestr ze sprawdzalnego źródła; nie udawać, że dowolny obraz train jest na pewno niewykorzystany.

Obowiązkowy rdzeń to pełnoobrazowe cechy obecnego DINO i ranking cosine; FAISS oraz dodatkowe metody są opcjonalne. Wybór konkretnej metody i preprocessingu trzeba zapisać przed porównaniem. SSIM, inne metryki percepcyjne oraz adaptacja wskazanego przez użytkownika DETR-E pozostają opcjami porównawczymi, bez cichego uznania ich za wykonany domyślny wariant. Dla DETR-E nadal potrzebna jest weryfikacja źródła i definicji; sama relacja o DINO/SAM/klasteryzacji nie zastępuje kodu.

Feature preprocessing zapytania i kandydatów jest zgodny i obejmuje pełną klatkę: jawny resize/letterbox, rozdzielczość, kolory i normalizacja, bez ukrytego cropu narzędzia. Informacje o narzędziu, tle i układzie sceny mogą wpływać na pełnoobrazowy ranking. Zachowujemy deterministyczną kolejność kandydatów i regułę remisów oraz zapisujemy metrykę, scores, identyfikator zapytania, identyfikatory wybranych obrazów i status eligibility.

Najnowsze doprecyzowanie użytkownika ustala roboczy cosine>0,85 i adaptacyjnie 0–100 przyjętych obrazów zamiast wymuszonego stałego budżetu. Historyczny wariant 134 pozostaje oddzielną kontrolą, a porównanie zmiennej liczebności musi uwzględniać zarówno dane, jak i liczbę aktualizacji; szczegóły poniżej.

Dalsze uczenie ma otrzymać **wybrane prawdziwe klatki**, nie obrazy SD. Obraz SD jest zapytaniem do rankingu. Nie rozwijamy tutaj nieuzgodnionego kroku szóstego.

Dotychczasowy retrieval DINO wykorzystywał crop256, a prowadzenie ViT crop176; SD dekodował pełną klatkę. Nowe żądanie zmienia retrieval na pełny obraz do pełnego obrazu. Nie zakładamy, że sama zmiana wyszukiwania automatycznie zapewnia dobrą prognozę lub poprawę uczenia. Dokumentacja opisuje plan; nowy runtime retrieval nie został w tym zadaniu zaimplementowany.


## Doprecyzowanie rankingu i adaptacyjnej liczebności — 2026-10-02T12:53:58+02:00

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


Ostatnia korekta obowiązującej wersji kroku5: 2026-10-02T12:55:28+02:00 (Europe/Warsaw).
