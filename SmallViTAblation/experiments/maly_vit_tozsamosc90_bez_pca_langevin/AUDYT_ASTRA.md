# Audyt Astra: rozpoznanie ponad 90 procent i GLE w natywnych kanałach

Data: 2026-10-02. Status: **FINAL — audyt zamknięty bez błędu blokującego. Zweryfikowano wybór kanałów, zamrożenie modeli, wyniki rozpoznania i GLE, reszty oraz końcową interpretację.**

Własność audytora obejmuje wyłącznie ten plik. Bez zmian kodu, planu lub parametrów wykonawcy; bez dodatkowych forwardów, dopasowań, treningów, pobierania i commitów. Audyt stosuje zasady dowodowe `audit-project-correctness`, z miejscem zapisu ograniczonym przez zlecenie. Nie modyfikuje osobnego rejestru projektu Grant.

Poprzedniki: [audyt kierunków U500](../maly_vit_langevin_kierunki_tozsamosci/AUDYT_ASTRA.md) i [audyt wspólnego ruchu](../maly_vit_langevin_wspolny_ruch/AUDYT_ASTRA.md). Nowy eksperyment zmienia przestrzeń celu na natywne kanały. Wyniki dawnych projekcji 8D nie są bezpośrednią bazą porównania nowego błędu K-wymiarowego.

## Przedmiot żądania i definicja wyniku

Użytkownik zdefiniował 90 procent jako wspólną trafność rozpoznania narzędzia, a następnie zakazał PCA w wyborze jego współrzędnych. W tym planie J jest podzbiorem indeksów oryginalnych osi e_j przestrzeni R768. Stan cząsteczki ma współrzędne z_i,J, bez zmiany bazy przez PCA/POD/SVD. Standaryzacja każdej osi i liniowy klasyfikator są dozwolonymi operacjami na tych współrzędnych; współczynniki dekodera nie stają się nową bazą mierzonego stanu. Macierze GLE mogą łączyć wybrane współrzędne w równaniu dynamiki, nie zastępując ich innym celem.

Główne 90 procent dotyczy rozpoznania oryginału versus wersji LaMa w znanym ROI obrazu. Jest to skuteczność klasyfikatora wspólnie korzystającego z J, nie trafność każdej pojedynczej osi, pojedynczego tokenu, swobodnego detektora lub prognozy ruchu. Pary syntetyczne mogą pozostawiać artefakty; rezultat nie dowodzi kompletnej semantycznej tożsamości ani przyczynowej odpowiedzialności kanałów.

## Sprawdzone pochodzenie surowych współrzędnych

Odczyt `IDENTYFIKACJA_K500.npz` poprzednika potwierdził pole `pooled400` o shape 400×2×768, dtype float32. W `maly_vit_langevin_kierunki_tozsamosci/identyfikacja500.py:23–25` powstaje ono jako średnia surowych wektorów z po ostatniej GELU MLP, ważona coverage oryginalnej maski, na identycznych indeksach tokenów w oryginale i wersji syntetycznej. Rzut na U występuje dopiero później, przy osobnym dawnym probe. Wczytanie samego `pooled400` nie wprowadza PCA ani rotacji.

Pola `probe_mu` i `probe_sd` w tym samym NPZ mają osiem wymiarów i dotyczą poprzedniego dekodera po U; nie nadają się do standaryzacji natywnych kanałów. Nowy ranking oraz nowe centra i skale muszą być wyznaczone na surowych train danych. Nie wykorzystywać poprzednich U500, eigenvalues ani pięciu osi jako gotowego wyboru nowych kanałów.

## Warunki audytu przed testem rozpoznania

1. Sprawdzić niezmienność źródła, zgodność indeksów oraz podział po filmach: 240/75/85 par i 69/23/24 filmów. Każda para pozostaje w jednym podziale, a filmy są rozłączne.
2. Ranking `abs(mean_train(delta))/SD_train(delta)` korzysta wyłącznie z 240 train par i równych wag filmów. Floor SD jest numeryczny i ustalony bez testu. Centra/skale dekodera pochodzą wyłącznie z train, a kolejność cech w train/val/test jest identyczna.
3. Etykiety mają odpowiadać kolejności oryginał/LaMa. W obrębie każdego filmu obie klasy otrzymują tę samą łączną wagę. Wtedy ważona accuracy jest dokładnie równa ważonej balanced accuracy: `0.5*(sensitivity+specificity)`. Nie utożsamiać jej ze zwykłą accuracy po wszystkich obrazach, jeżeli filmy mają różną liczbę par.
4. Próg 0.5 jest stały. K wybieramy jako najmniejsze **spośród z góry zadeklarowanej siatki** z val accuracy ściśle większą od0.90, następnie najlepszą val accuracy/AUROC i mniejsze C zgodnie z planem. Nie twierdzić, że znaleziono globalnie najmniejsze K spośród wszystkich liczb 1..768, jeżeli siatka ma tylko 12 wartości.
5. Zapisać kanały, ich kolejność, scaler, współczynniki, C i SHA przed pojedynczym nowym odczytem 85 test par. Klasyfikatora nie refitować na val. Nie poprawiać wyboru po wyniku testowym.
6. Twierdzenie o testowym przekroczeniu 90 procent wymaga testowej wartości >0.90. Sam val/train nie wystarcza. Punktowe przekroczenie i dolna granica CI>0.90 to różne twierdzenia. Bootstrap całych filmów ma zachować pary i obie klasy.

Te same testowe filmy i poprzednie wyniki były już oglądane. Algorytm może zachowywać ścisły podział train/val/test w tej próbie, lecz nie czyni to całej historii badawczej nietkniętym testem potwierdzającym. Plan poprawnie deklaruje eksploracyjny status.

## Warunki audytu dynamiki bez PCA

Kanały J są wybierane przez rozpoznawanie na checkpoint 500 i zamrożone przed dopasowaniem dynamiki. GLE otrzymuje pełny wektor tych K natywnych współrzędnych każdego tokenu, bez redukcji lub wyboru kanałów według ich późniejszej przewidywalności. Równanie i MSE wracają do fizycznej skali aktywacji.

Persystencja, ostatnia prędkość, lokalne L0 i warianty z pamięcią/sprzężeniem muszą przewidywać dokładnie to samo J, na tych samych originach, z tymi samymi wagami. Zmiana K uniemożliwia bezpośrednie przedstawienie różnicy MSE lub sign accuracy względem dawnych celów U8 jako poprawy tej samej prognozy. Ewentualna rekonstrukcja starego rzutu do J wymaga osobnego jawnego założenia o nieruchomym dopełnieniu.

Podział czasu zachowuje fit 501..700, validation z celami do799, refit operatorów do799 i test od800; dt10 odpowiednio końce 700/790. Dobór K z obrazów k500 nie może być później zmieniony na podstawie wyników prognozy. Wybór parametrów dynamiki ma korzystać tylko z validation. Transfer do sondy 8 nie dopasowuje operatora ani kanałów.

Wariant sprzężony musi zachować weryfikowane wcześniej domknięcie: średnia foreground jest funkcją prognozowanych tokenów; przyrosty foreground i kontekstu liczone z jednego snapshotu; przyszła historia pochodzi z własnych prognoz. To wspólne addytywne sprzężenie, nie identyfikacja indywidualnych sił. Prefix/poison oraz kontrolę skal należy potwierdzić ponownie w nowej ścieżce kodu.

## Bieżący stan

Przeczytano PLAN.md oraz kod i kształty źródłowego pooled 400. Przekazano wykonawcy rozróżnienie raw 768 od dawnych scalerów 8D oraz warunki wag, granicy 90 i rozdzielenia trafności obrazu od ruchu tokenów. Nie oceniono jeszcze nowego wyboru klasyfikatora, nowych macierzy lub wyniku testowego.

## Statyczny audyt selektora i podziału

`wybor_kanalow.py:20–25` przypisuje oryginałom etykietę 1, wersjom usuniętym 0, zachowując kolejność przy powielaniu wag. `partition` nadaje każdemu filmowi równą sumę wag, obrazom wewnątrz filmu równe udziały, a metryki dzielą łączną masę równo między klasy. Kod sprawdza numerycznie równość ważonej accuracy i BA. Sensitivity/specificity oraz accuracy nieważoną wskazano wykonawcy jako brakujące pola wymagane przez plan.

`wybor_kanalow.py:30–31` wyznacza ranking różnic parowych i scaler na train. Tie w rankingu rozstrzyga rosnący indeks kanału. Standardowe `sample_weight` w logistycznym fit mają sumę równą liczbie przykładów, więc klasy i filmy mają zadeklarowane względne wagi, bez przypadkowego dodatkowego osłabienia wszystkich próbek względem C. K/C są wybierane tylko z validation; próg pozostaje 0.5. Fizyczna postać dekodera w `:44` poprawnie odwraca diagonalną standaryzację, bez rotacji przestrzeni.

Źródłowe SHA splitu i manifestu porównano z `IDENTYFIKACJA_K500.json` zapisanym podczas pomiaru: obie wartości zgodne. Cały źródłowy NPZ ma SHA `457ac355f6cf48709cd0c6ce3f82651edae51706ff49fbfe5954b3ada8dc4298`. Odczyt metadanych potwierdził liczby train 240/69, val 75/23, test 85/24 oraz zerowe przecięcia wszystkich par zbiorów filmów. Nie odczytywano wyników klasyfikatora na testowych etykietach w tej kontroli podziału.

W `test()` bootstrap resampluje 24 wartości accuracy całych filmów. Każda z nich zachowuje oba człony wszystkich par danego filmu; jest to zgodne z film-weighted BA. Poproszono dodatkowo o sprawdzenie SHA splitu i manifestu przy przejściu do testu oraz ochronę przed przypadkowym powtórzeniem etapu testowego — samo zapisane w chwili freeze `test_evaluated=False` nie jest trwałą blokadą kolejnego uruchomienia.

Rozpoznanie dotyczy checkpoint 500. Prognoza dotyczy dalszych zmian wybranych przy nim kanałów. Bez ponownych parowych pomiarów na innych checkpointach nie wykazano, że to samo rozpoznanie utrzymuje 90 procent przez całą trajektorię 501..1000. To ograniczenie interpretacji, nie powód do dodatkowych forwardów w tym zleceniu.

## Zamrożenie wyboru przed testem rozpoznania

Odczytano `WYBOR_WALIDACJA.json` i `KANALY_MODEL.npz` przed pojawieniem się artefaktu testowego. Wybrano **K=128, C=0.01**; ważona BA validation wynosi **0.9067028985507246**, AUROC 0.9586680844360428. Wszystkie 48 dopasowań zakończyły się bez ostrzeżenia o braku zbieżności; wybrane potrzebowało 26 iteracji. Maksymalna BA przy K64 wynosi 0.8931159420289855, a przy wszystkich mniejszych K z siatki jest również poniżej 0.90. To najmniejsze skuteczne K w zadeklarowanej siatce, nie dowód minimalności spośród wszystkich wartości 1..768.

Potwierdzono 128 unikalnych indeksów, zakres 0-based 1..737, zgodność `channels1=channels0+1`, pełny ranking 768, scaler 768 i wektor współczynników 128. Macierz kanałów nie jest uczoną rotacją. Fizyczny dekoder i standaryzowany dekoder mają poprawne algebraiczne powiązanie; bezpośrednia kontrola ich zgodności pozostaje częścią testu.

SHA modelu przed testem: `5212a7bfff03d4d5a0756b37eae958ddc15a4f7fbd3935a9f8e1388b68b587a6`. Odczytany plik zgadza się z tym zapisem. `OCENA_OBECNOSCI_TEST.json` w chwili audytu nie istniał.

Po uwadze audytora `test()` sprawdza niezmienność splitu i manifestu oraz blokuje przypadkowe powtórzenie, gdy istnieje artefakt wyniku. `metrics` uzupełniono o sensitivity, specificity i nieważoną accuracy. Walidacja zdążyła uruchomić wcześniejszą definicję funkcji, dlatego jej historyczny grid zawiera BA/AUROC/ważoną macierz pomyłek bez tych dodatkowych pól. Ta zmiana raportowania nie modyfikuje wartości użytych do wyboru, J, C, progu lub współczynników. Nie wymaga ponownego fitu lub wyboru.

Wykonawca i root otrzymali jednoznaczne zamknięcie audytu przedtestowego przed odczytem testu. Na tym etapie potwierdzono jedynie przekroczenie 90 procent na walidacji; nie formułowano jeszcze twierdzenia o przekroczeniu progu na teście.

## Koszt i zakres planowanej dynamiki 128D

Po wyborze K128 root zapisał przed fitami dynamiki siatkę: lokalny/sprzężony × L0/1/2/4 × ridge 0.01/0.1/1/10, czyli 32 konfiguracje na dt i64 łącznie. Wszystkie 128 kanałów pozostają stanem i celem; pominięcie L8 jest jawnym ograniczeniem kosztu, nie redukcją współrzędnych. Nie używa się rekonstrukcji wcześniejszego U8 jako porównania.

Przy K128 i L4 pełny foreground design wariantu sprzężonego ma1920 regresorów, a operator kontekstu 1280. Tło ma tylko 200/20 czasowych wierszy pierwotnego fit dla dt1/dt10. Ridge definiuje dopasowany predyktor, lecz nie usuwa niedookreślenia dużej macierzy przez dane. Należy osobno raportować rangę i uwarunkowanie fg/bg oraz unikać fizycznej interpretacji indywidualnych współczynników.

Wykonawcy wskazano konieczność usunięcia dawnych ograniczeń do8 współrzędnych: wycinków rollout `:8`, progów `:8` i `range(8)` w diagnostyce reszt. W nowym etapie test musi sprawdzać cały wymiar 128, niezależnie od wcześniejszej implementacji q8. Ta ścieżka kodu pozostaje do osobnego audytu przed testem dynamiki.

## Zweryfikowany pojedynczy test rozpoznania

`OCENA_OBECNOSCI_TEST.json` powstał po zamknięciu audytu przedtestowego. Wybrany J i model mają ten sam SHA co przed testem i co bieżący `KANALY_MODEL.npz`. Nie zmieniono C, progu ani zestawu 128 kanałów.

| Miara na85 parach z24 filmów | Wartość |
|---|---:|
| Film-weighted balanced accuracy | 92.827381% |
| Film-weighted accuracy | 92.827381% |
| Nieważona accuracy 170 przykładów | 91.764706% |
| Film-weighted AUROC | 97.414510% |
| Sensitivity oryginału | 91.974206% |
| Specificity wersji usuniętej | 93.680556% |
| 95% CI BA, bootstrap całych filmów | [88.174479%,96.994048%] |

Punktowy warunek **BA>90% jest spełniony**. Dolna granica przedziału znajduje się poniżej 90%, więc nie wykazano statystycznej gwarancji przekroczenia progu w populacji filmów. Bootstrap ma2000 resamplowań jednostek filmowych i zachowuje komplet par obu klas. Formalna równość accuracy/BA wynika z równego ważenia obu klas; różnica względem nieważonej accuracy wynika z ważenia filmów.

Bezpośrednia zgodność logitów surowego i standaryzowanego dekodera ma maxabs 2.6645352591003757e-15. To potwierdza poprawne odwrócenie skali i fizycznych współczynników w zapisanym modelu. Kryterium 90 nie dotyczy pojedynczej cząsteczki, prognozy ani pełnej identyfikacji semantycznej; dotyczy wspólnego dekodera 128 natywnych współrzędnych średniej ROI na checkpoint 500.

## Odczyt przygotowania i modelu natywnego

`przygotuj_dynamike.py:7–17` sprawdza SHA wyboru i bezpośrednio odczytuje `latent[t,probe,token_ids][:,J]`, bez mnożenia przez U/V lub inną uczoną bazę. Średnia kontekstu jest liczona w tych samych J. `model_native.py:33,45,59` przechowuje progi, cały stan i cały wynik rollout w pełnym K; dawne ograniczenia do8 zostały usunięte.

Struktura dopasowania i synchronicznego sprzężenia pozostaje zgodna z poprzednim audytem, ale wymaga jeszcze kontroli nowego pipeline wyboru/testu. Wskazano wykonawcy, aby tablice s/bg konwertował do float64 przed odejmowaniem anchorów i przyrostów; źródłowy NPZ zachowuje dokładne float32 aktywacji, natomiast obliczenia regresji powinny używać spójnej precyzji.

## Audyt pipeline i zamrożenie dynamiki

`dynamika.py:35–55` stosuje dokładnie zapisaną siatkę 32 konfiguracji na dt, równe targety fit dla różnych L, pełne K128 oraz wybór z validation. Oddzielnie zachowuje najlepsze modele lokalne/sprzężone, kontrole L0 i wariant krótkiego horyzontu. Kanały nie są zmieniane przez wynik walidacji dynamiki. Dane s są konwertowane do float64 przed odejmowaniem; bg jest już float64.

`dynamika.py:64–89` ma ochronę przed powtórnym testem i porównuje SHA macierzy, danych oraz zamrożonego klasyfikatora. `:80` wymaga dokładnej zgodności kształtu prognozy z całym 128-wymiarowym celem. Progi i majority baseline są wspólne, pochodzą z finalnego fit lokalnego modelu głównej sondy; nie są dopasowywane na sondzie 8. Baselines persystencji, prędkości oraz L0 dotyczą tego samego J. Kontrasty są względem nowego lokalnego modelu 128D, a nie starego rzutu 8D. Wspólne originy co 10 są wybierane z tych samych prognoz dt1.

`DYNAMIKA_NATYWNE_J.npz` ma SHA `38942148bce344dc89d0309ed2efb10f2e4730986024eaf09acc387448b75559`, zgodny z `DANE_DYNAMIKI.json`. Bezpośredni odczyt potwierdził identyczne J jak w klasyfikatorze. Stany mają shape 1001×31×128 i1001×7×128; konteksty 1001×128. Dane surowe s zachowują float32 źródła, obliczenia dynamiki używają float64.

Przed testem odczytano `WYBOR_DYNAMIKA_DT1/DT10.json`, macierze i metadane. Wszystkie 32 konfiguracje w każdej dyskretyzacji dały skończony wynik validation. Żaden plik `METRYKI_DT*.json` jeszcze wtedy nie istniał. Wybory główny i krótkiego horyzontu są w tej próbie identyczne:

| dt | Wybrany model | Mean skill validation | Short skill validation | Targets refit |
|---:|---|---:|---:|---:|
|1|lokalny, L2, ridge 1|15.925907%|H1:45.743966%|299|
|10|lokalny, L2, ridge 10|5.494463%|H10:5.179586%|29|

Wybrany model w obu skalach nie korzysta z jawnego kontekstu/średniej, mimo że ramiona sprzężone zostały ocenione i zachowane. Macierz regresji B ma384×128; każdy z trzech bloków odpowiada Aᵀ, K0ᵀ lub K1ᵀ, według zapisanego layout. b i progi mają 128 składowych, anchor głównej sondy 31×128. Równanie lokalne w zakotwiczonym x jest zatem zgodne z pełnym natywnym stanem, bez wycinania ośmiu kanałów.

Metadane po reficie: dt1 rank 384/384, condition 41.2826, fizyczna/scalowana postać zgodna do2.85e-16; dt10 rank 384/384, condition 4.31320, zgodność 6.45e-16. Promienie operatora companion wynoszą odpowiednio 1.001475719 i1.005697177. Są większe od1; skończoność rollout na H<=50 nie oznacza asymptotycznej stabilności dowolnie długiej prognozy. Plan nie wymagał promienia<1 jako kryterium wyboru, więc nie jest to powód do zmiany zamrożonego modelu po selekcji.

SHA macierzy dt1 przed testem: `d7bbeeef7a9a8ebaed06498e5ecd697f066488a16d55c86392514fcefe15b741`.

SHA macierzy dt10 przed testem: `12d47ddafe09ff4a4580f33da0afe4ef7d28bb8eb06a4103e352d160b6f932cf`.

Bieżące pliki zgadzały się z zapisami wyboru. Wykonawcy i rootowi przekazano zamknięcie audytu przedtestowego przed jednorazowym testem dynamiki. Nie wprowadzano nowej siatki lub kryterium po odczycie walidacji.

## Zweryfikowany test GLE w całym J

Po wykonaniu testu hashe obu macierzy i danych nadal zgadzają się z zamrożeniem. `AUDYT_PREFIX_DT1/DT10.json` zawierają po24 poprawne przypadki, łącznie 48, dla sześciu modeli, dwóch sond i originów 800/950. Każdy potwierdza dokładną zgodność prognoz po usunięciu lub zatruciu przyszłości i jawnie obejmuje `all_K=128`. Nie wykryto przyszłego teacher forcing lub pominięcia części wybranych kanałów.

| Sonda | dt | H | Skill wybranego GLE | Znak wszystkich 128 składowych | Cosine przemieszczenia |
|---:|---:|---:|---:|---:|---:|
|2|1|1|47.7504%|71.6813%|0.66477|
|2|1|2|33.4519%|67.3848%|0.54551|
|2|1|5|7.5347%|60.5955%|0.33269|
|2|1|10|−4.1531%|58.7312%|0.27925|
|2|1|20|−7.1478%|59.7022%|0.30836|
|2|1|50|−45.3003%|57.1836%|0.24228|
|2|10|10|4.2963%|56.6718%|0.19579|
|2|10|20|9.3989%|58.7350%|0.28325|
|2|10|50|4.1462%|57.7347%|0.26830|
|8|1|1|−1.4093%|65.4089%|0.39686|
|8|1|10|−262.3568%|54.2473%|0.10007|
|8|1|50|−265.2654%|55.7879%|0.16677|
|8|10|10|−23.0052%|54.6761%|0.10190|
|8|10|20|−23.2255%|56.2194%|0.14470|
|8|10|50|−70.6320%|54.5823%|0.11139|

Skill jest redukcją MSE wobec persystencji, nie odsetkiem poprawnych predykcji. Przy H1 sondy 2 MSE GLE wynosi 0.00011510347 wobec 0.00022029536 persystencji. Wybrany GLE przegrywa jednak z prostą ostatnią prędkością: jej H1 skill 55.9193%, znak 78.6715%, wobec 47.7504% i 71.6813% GLE. Na sondzie 8 velocity także jest wyraźnie lepsza punktowo przy H1: skill 62.5324%, znak 80.1753%, podczas gdy GLE nie poprawia nawet persystencji. Nie wolno przypisywać wyniku velocity modelowi Langevina.

Ujemny skill −265.2654% transferu na H50 oznacza, że MSE jest 3.652654 razy większy od MSE persystencji; nie oznacza 265 procent błędnie sklasyfikowanych zdarzeń. Długi rollout obu sond jest słaby. H50 dt1 na sondzie 2 ma opisowy 95% CI skillu [−59.1046%, −34.5767%], w całości poniżej zera. H1 sondy 8 ma punktowo −1.4093%, lecz CI [−8.1237%, 3.2475%] obejmuje zero, więc nie ustanawia rozstrzygniętej przewagi żadnej strony na tym jednym kroku.

dt10 na sondzie 2 ma małe dodatnie wyniki. Opisowe CI skillu wynoszą H10 [1.5924%, 6.5456%], H20 [6.4582%, 11.9611%], H50 [−0.9082%, 7.8934%]. Ostatni obejmuje brak przewagi; przy zaledwie 16 nakładających się originach bootstrap ma około 3 efektywne długości bloków, a wynik pozostaje eksploracyjny. Transfer dt10 na sondzie 8 ma ujemne wartości dla wszystkich ocenionych horyzontów.

Kontrole są zdefiniowane na tym samym J. Lokalny L0 ma dla sondy 2 dt1/H1 skill −0.4335%, więc pamięć poprawia względem samego dryfu, lecz nie pokonuje velocity. Dla dt10/H10 lokalny L0 ma 7.3797%, a wybrany L2 tylko 4.2963%; nie wykazano jednolitej przewagi pamięci na każdym horyzoncie. Wariant sprzężony dt1 na sondzie 2 poprawia H50 względem wybranego lokalnego, ale nadal ma skill −21.4648%, za cenę słabszego H1. Nie podmieniono wyboru na ten wariant po teście.

Dla uczciwego porównania kroków, na tych samych 16 originach co 10, wybrany dt1 daje sondzie 2 H10/H20/H50 odpowiednio −1.0174/−8.6189/−45.3779%, wobec 4.2963/9.3989/4.1462% dt10. Na sondzie 8 oba pozostają gorsze od persystencji. Tych wartości nie należy zestawiać z dawnym 8D jako poprawy identycznego celu: współrzędne i wymiar zostały zmienione.

## Nowe reszty 128D i zakres równania Langevina

`reszty.py:11–24` wyznacza residuale wszystkich 128 współrzędnych wybranego modelu, osobno dla sond 2 i 8. Targety 801..1000 dają 200 obserwacji dt1; targety 810..1000 co 10 dają 20 obserwacji dt10. Każdy jednokrokowy residual używa dostępnej rzeczywistej historii do własnego originu i nie jest tym samym co autonomiczny błąd H50. Testy wykonano po wyborze parametrów, bez strojenia.

| dt | Sonda | Liczba szeregów token/kanał + średnia | Nominalne Ljung–Box p<0.05 | Nominalne Jarque–Bera p<0.05 | Udział wspólnej energii reszty |
|---:|---:|---:|---:|---:|---:|
|1|2|4096|4084|1949|65.6306%|
|1|8|1024|1024|370|68.4874%|
|10|2|4096|448|261|64.8904%|
|10|8|1024|185|43|66.0157%|

Nie ma zdegenerowanych serii w zapisanej diagnostyce. Tysiące szeregów nie są tysiącami niezależnych doświadczeń: dzielą obrazy i aktualizacje sieci. P-wartości są surowe, bez korekty wielokrotności i liczby dopasowanych parametrów, a dt10 ma tylko 20 obserwacji. Wyniki nie uzasadniają białego szumu gaussowskiego, dokładnego zamknięcia Mori–Zwanziga ani FDT. Czasowa białość i normalność pozostają odrębnymi własnościami.

Prognoza punktowa pomija przyszłą resztę; nie jest to empiryczny dowód jej zerowej średniej lub braku przewidywalnej struktury. Fakt rozpoznawania znanego ROI przez 128 kanałów z testową trafnością 92.8274% nie ustanawia równie dobrej przewidywalności ich ruchu w uczeniu.

## Zamknięcie audytu

Odczytano końcowe `WYNIKI.md`, `WYNIKI.json` i `PODSUMOWANIE.md`, w tym doprecyzowania transferu H50 oraz przedziału niepewności dt10. Ich główne liczby i wnioski są zgodne z zapisanymi metrykami. Osiem z ośmiu skrótów SHA256 kodu w raporcie zgadza się z plikami. Oba skróty `METRYKI_DT1.json` i `METRYKI_DT10.json` zgadzają się z `dynamic_metrics_sha`. Raport potwierdza zachowanie wszystkich 128 wybranych współrzędnych, brak PCA w stanie i rozmiar artefaktów około 72.46 MB, poniżej limitu 150 MB. Wcześniej sprawdzono zgodność zamrożonych modeli po teście i 48 z 48 przypadków prefix/poison.

**Werdykt naukowy:** w zadaniu znanego ROI przy checkpointcie 500 przekroczono próg 90 procent punktowo, z testowym wynikiem 92.8274%. Dolna granica przedziału po filmach wynosi 88.1745%, więc nie wykazano dolnego ograniczenia skuteczności ponad 90 procent. Wybrano natywne kanały, bez uczenia nowej bazy. Następny etap rzeczywiście prognozuje wszystkie te kanały, lecz nie daje podobnej trafności ruchu: GLE przegrywa z ostatnią prędkością przy H1, ma ujemny skill na dłuższych horyzontach dt1 i nie przenosi się dobrze na drugi obraz. Niewielki dodatni wynik dt10 na pierwszym obrazie nie odwraca tych ograniczeń.

Nie znaleziono nierozwiązanego błędu blokującego w audytowanym zakresie. Ograniczenia danych, historycznej ekspozycji testu, artefaktów LaMa, zależności tokenów, identyfikowalności macierzy i reszt zostały jawnie opisane. Audyt zamknięto bez rozszerzania siatki, powtórnego dopasowania lub wybierania zwycięzcy po teście. Zmieniono wyłącznie `AUDYT_ASTRA.md`.
