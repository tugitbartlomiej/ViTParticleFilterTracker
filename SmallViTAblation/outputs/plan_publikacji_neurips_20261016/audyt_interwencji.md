# Audyt tezy publikacyjnej i plan eksperymentów interwencyjnych na Edenie

16 października 2026. Plan na podstawie lokalnych raportów z 15 października; nie uruchomiono nowych obliczeń ani nie zmieniono aplikacji. Zakres: współdziałanie neuronów ostatniego MLP, geometria ramki i wybór zwycięskiej detekcji. To propozycja falsyfikowalnego badania, nie deklaracja nowości względem całej literatury ani gwarancja akceptacji na konferencji.

## Najmocniejsza kandydatura na główną tezę

**Hipoteza: organizacja podpisanych wkładów neuronów zależna od obrazu rozdziela stabilność współrzędnych ramki od stabilności rankingu kandydatów. Miara tego rozdzielenia, ustalona na oddzielnych danych, przewiduje szkodliwe przełączenia detekcji pod nowymi zaburzeniami lepiej niż sam margines, normy i standardowa analiza czułości.**

To hipoteza o wyuczonej, zależnej od kontekstu strukturze oraz jej praktycznej wartości. Samo dodawanie i znoszenie wektorów albo nieciągłość argmax nie wystarczą jako wkład naukowy. Dodatkowy wynik mechanistyczny powinien pokazać kontrolowane wywołanie i odwrócenie błędu na nowych obrazach, przy porównywalnej sile interwencji.

Mocniejsze sformułowanie „kompensujące zespoły stabilizują współrzędne, ale destabilizują wybór ramki” jest na razie **niepotwierdzoną wersją tej hipotezy**. Największe znane przełączenie 369→370 w kroku 55000 wywołała grupa **wzmacniająca**, nie grupa kompensująca ramkę. Nie wolno przedstawiać tego przykładu jako istniejącego dowodu głównej hipotezy.

Alternatywa, jeśli przewidywanie nie poprawi się względem mocnych baseline: zawęzić pracę do rzetelnego instrumentu diagnostycznego i benchmarku interwencji, z negatywnym wynikiem dotyczącym prostych fizycznych analogii. Nie podtrzymywać tezy o nowym mechanizmie za pomocą efektownych projekcji 3D.

## Co wiemy, a czego brakuje

| Obserwacja pilota | Co potwierdza | Czego jeszcze nie potwierdza |
| --- | --- | --- |
| Małe q dla par dobranych przez przeciwne `J_b u_c` | Zgodność skończonych ablacji z lokalnymi wpływami | Nowej wyuczonej organizacji; silny efekt istnieje już w inicjalizacji |
| Około 0,2% lokalnej reszty nieaddytywnej wybranych par | W tej skali głowa jest zwykle lokalnie bliska liniowej | Dużej nieliniowej synergii MLP |
| Globalna ablacja 32 neuronów zmieniła zwycięzcę i pogorszyła IoU | Istnienia możliwego przełączenia przy kumulacji małych wpływów | Częstości zjawiska, przewagi selektora ani związku z kompensującymi zespołami |
| Ujemniejszy bilans przeciwnych par w bbox niż w tle, także dla wcześniej wybranych par | Zależności rzeczywistych podpisanych wkładów od tokenu i obszaru obrazu | Specjalizacji wyłącznie w narzędziu, poprawy IoU, generalizacji na filmy |
| Słaby transfer zamrożonych par: q około 0,887 vs 0,898 kontroli | Ograniczenia lokalnego doboru | Istnienia stałych funkcjonalnych zespołów przenoszących się między obrazami |
| Brak silnego następstwa zmian po checkpointach | Braku potwierdzenia prostego schematu opóźnionej korekty | Braku wszystkich relacji czasowych; rozdzielczość wynosiła 100 kroków |

## Pytania i główne wyniki, ustalone przed testem

Rozdzielić trzy przestrzenie: residual, cztery współrzędne ramki dla **ustalonego tokenu**, oraz wyniki kandydatów decydujące o **wyborze tokenu**. Każda analiza ma raportować, w której z nich jest kompensacja.

Dla rozłącznych grup neuronów A i B, siły osłabienia α oraz aktualnego zwycięzcy w:

- `Δb_A`, `Δb_B`, `Δb_AB`: dokładne zmiany współrzędnych **ramki w**, bez ponownego wybierania zwycięzcy.
- `q_b = ||Δb_AB||/(||Δb_A||+||Δb_B||)`: kompensacja, z jawnie ustalonym dolnym progiem mianownika i raportowaniem odrzuconych ilorazów.
- `I_b = Δb_AB−Δb_A−Δb_B`: nieaddytywność głowy dla ustalonego tokenu. Nie utożsamiać jej z kompensacją.
- `m = s_w−max_{j≠w}s_j`: margines wyników kandydatów. Użyć rzeczywistego dekodera i jego reguł dopuszczania kandydatów; dla wielu obiektów oddzielić zmianę kandydata od zmiany przypisanego obiektu.
- `winner_changed`, zmiana IoU i znormalizowany błąd krawędzi: wynik po pełnym ponownym dekodowaniu.

**Główny eksperyment predykcyjny:** z czystego obrazu przewidzieć ryzyko szkodliwej zmiany detekcji pod zaburzeniem, którego konkretna realizacja nie jest udostępniona predyktorowi. Miary dodatkowe muszą poprawić przewidywanie ponad margines i odpowiednio silne baseline czułości. Zwykła rekonstrukcja skutku znanej interwencji z jej JVP pozostaje testem poprawności instrumentu.

**Główny eksperyment mechanistyczny:** częstość zdarzenia, w którym A i B osobno zachowują zwycięzcę, lecz razem przełączają go i pogarszają lokalizację. Dla przykładowej prerejestracji: model bazowy ma IoU≥0,5, obie osobne interwencje nie zmieniają zwycięzcy, wspólna go zmienia, a IoU spada o co najmniej 0,05. Te progi to propozycja projektowa do zamrożenia na zbiorze discovery; nie stroić ich na teście. Dodatkowo obowiązkowo podać rozkład ciągłego ΔIoU i wyniki bez ograniczenia do poprawnych wyjściowo detekcji.

Zmiana tokenu bez pogorszenia pokrycia nie jest automatycznie błędem. Wiele tokenów może proponować poprawną ramkę tego samego narzędzia.

## Oddzielić selekcję od sprawdzenia

Wybór par o przeciwnych `J_b u_c` i pomiar małego q w tym samym stanie jest bliski potwierdzeniu kryterium selekcji. Przy małej α wynika to z rozwinięcia Taylora. Nie przedstawiać go jako niezależnego odkrycia sieciowego „mechanizmu kompensacji”.

Potrzebne są dwa jawnie odrębne protokoły:

1. **Stałe zespoły:** wybrać identyfikatory na filmach discovery i ustalonym wcześniejszym checkpoincie; zamrozić przed późniejszymi checkpointami oraz nowymi filmami. To test przenoszenia zespołów. Nie porównywać tych samych numerów neuronów między seedami — neurony nie mają wspólnej tożsamości. Między seedami przenosi się reguła selekcji i hipoteza, nie numery.
2. **Adaptacyjny instrument:** algorytm może użyć czystego wejścia i gradientów bieżącego modelu, ale jego reguła, hiperparametry i kalibracja są zamrożone. Testuje się go na nowych filmach oraz rodzinach/severity zaburzeń. To test użyteczności metody, nie istnienia stałych zespołów. Żadnych GT ani wyników właściwej interwencji w cechach predyktora.

Równoważność informacji jest istotna: miara zbudowana wyłącznie z pełnych `J u_c` nie zawiera magicznie nowej informacji ponad pełny zestaw tych samych wektorów. Może być użytecznym, oszczędnym podsumowaniem. Przewaga nad pojedynczą normą gradientu, przy porażce względem pełnego przewidywania JVP danej interwencji, nie dowodzi nowego mechanizmu nieliniowego.

## Podział danych i powtórzenia

- Najpierw audyt manifestu treningowego: źródłowe nagranie, operacja/pacjent, klatka, etykieta i przynależność do splitu. Trzy obrazy pilota pozostają wyłącznie discovery.
- Train, discovery, calibration i final test muszą być rozłączne przynajmniej po filmie, a po pacjencie/operacji, jeśli identyfikatory na to pozwalają. Podział losowy po klatkach jest niewystarczający.
- Test główny nie może zawierać obrazów wcześniej użytych do uczenia detektora. Jeśli obecny model obejmował wszystkie proponowane filmy testowe, potrzebne są nowe biegi z poprawnym splitem.
- Próbkować klatki rzadko w czasie; stratygrafia na rozmiar narzędzia, położenie, okluzję i jakość bazowej detekcji. Ustalić wagi przed poznaniem wyników interwencji. Odróżnić reprezentatywny test od dodatkowego zbioru trudnych, małomarginesowych przypadków.
- Co najmniej trzy niezależne seedy jako etap minimalny; pięć lub więcej jako silniejszy test. Checkpointy tego samego seedu nie są niezależnymi powtórzeniami.
- Raportować efekty osobno per seed i per film. Przedziały niepewności resamplują filmy/pacjentów oraz seedy na właściwym poziomie, nie tokeny i pary jako niezależne obserwacje. Przy trzech seedach pokazać każdy wynik; wnioskowanie populacyjne po seedach będzie nadal słabe.

## Interwencje, które rozstrzygają hipotezę

### 1. Rozdzielenie stabilności współrzędnych i rankingu

Porównać selektory wybierające: kompensację w residual, kompensację współrzędnych, mały efekt współrzędnych przy dużym przewidywanym wpływie na margines, zgodność wpływów, normę/czułość i kontrolę losową. Dla każdego zestawu wykonać A, B i A∪B.

Primary: wyłączenie tych samych identyfikatorów we wszystkich tokenach, bez używania GT do wyboru miejsca działania. Secondary: interwencja w ustalonym tokenie zwycięskim; dodatkowo analiza z lokalizacją wyznaczoną przez GT oznaczona jako oracle, a nie metoda dostępna bez etykiet.

Zmieniać α i wielkość grupy. Porównywać nie tylko liczbę neuronów, lecz także sumę długości usuwanych wkładów, indywidualne wpływy na ramkę i na logit/margines oraz początkowy margines. Losowa grupa nie może być słabą kontrolą tylko dlatego, że składa się z prawie nieaktywnych neuronów.

Primary wynik: szkodliwe przełączenia na jednostkę kosztu interwencji i poprawność przewidywania zdarzenia. Secondary: q, I, rzeczywisty ΔIoU, zmiany wszystkich kandydatów, maskowanie dużych efektów osobnych. Raportować także brak przełączeń i wszystkie obrazy, bez selekcji największego skoku.

### 2. Kontrolowane zaburzenia znaczenia obrazu

Potrzebne są czyste/zmienione pary obrazów, z poprawną aktualizacją GT przy transformacji geometrycznej:

- zasłonięcie lub rozmycie fragmentu **narzędzia**;
- taka sama operacja na fragmencie **tła**, dopasowanym powierzchnią, położeniem i siłą zaburzenia;
- zmiana tła/kontekstu przy zachowaniu narzędzia;
- łagodne przesunięcie lub skalowanie obrazu z odpowiednio przekształconą ramką;
- kontrolne globalne zmiany jasności/kontrastu i brak zmiany.

Maskowanie całego bbox nie izoluje narzędzia. Użyć dostępnych masek segmentacyjnych albo ręcznie sprawdzić maski w niewielkiej prerejestrowanej próbce. Dodać identyczne artefakty brzegu maski w kontrolach, żeby nie pomylić semantyki z wykrywaniem sztucznego prostokąta. Rzeczywiste klatki o różnym stopniu zasłonięcia mogą służyć jako dodatkowe potwierdzenie, ale zawierają inne współzmieniające się czynniki.

Sprawdzić, czy na nowych filmach miary współdziałania reagują inaczej na usunięcie informacji o narzędziu niż na dopasowane zmiany tła. Ważniejsze od efektownego heatmapu jest przewidzenie pogorszenia lokalizacji oraz jego rozróżnienie od czystej zmiany pewności.

### 3. Podmiana aktywacji i odwrócenie efektu

W czystym i zaburzonym obrazie porównać aktywacje wybranych grup. W zaburzonym przebiegu podmienić aktywacje tych grup na wartości z czystego przebiegu, osobno A, B i A∪B; wykonać też kierunek odwrotny. Skorygować dopasowanie tokenów dla transformacji geometrycznej.

Porównać odzysk marginesu, zwycięzcy i IoU z grupami losowymi oraz wybranymi przez normy, gradienty i attribution, przy tej samej liczbie neuronów i zbliżonej normie poprawki. Taki „rescue” jest mocniejszy niż samo wyzerowanie, choć również może tworzyć stan spoza rozkładu. Podmiany między zgodnymi kontekstami i kontrola losowa muszą mieć podobną wielkość zmiany. Nie deklarować na tej podstawie przyczyn całego procesu uczenia; chodzi o mechanizm danego zamrożonego modelu.

## Obowiązkowe porównania

| Rodzina | Minimalny baseline |
| --- | --- |
| Decyzja detektora | Pewność, margines top1–top2, rozbieżność ramek top kandydatów, prosta regresja ryzyka z tych wielkości |
| Wielkość wkładu | `||u_c||`, suma/kwadrat norm grupy, energia aktywacji, znormalizowane budżety interwencji |
| Czułość | `||J_b u_c||`, wpływ na logit i margines; pełne pierwszorzędowe przewidywanie skutku **znanej** interwencji |
| Attribution | Gradient×activation dla tych samych wyjść; Integrated Gradients po neuronowej bramce przy jawnie zadanej bazie; ranking dokładnych efektów pojedynczych ablacji jako droższy baseline |
| Grupy | Suma attribution, dopasowane losowe grupy, kontrola po indywidualnych normach i początkowym marginesie; ten sam budżet wyszukiwania |
| Predykcja | Te same rodziny modeli predykcyjnych, rozmiary danych, strojenie wyłącznie calibration; ablation study cech współdziałania |

Dla znanej konkretnej interwencji można wyznaczyć próg przełączenia z liniowego modelu logitów/marginesów. Jeżeli ten prosty model wystarcza, raportować właśnie to; nie nazywać przekroczenia progu nową nieliniową synergią neuronów. Jeżeli potrzebna jest krzywizna, pokazać przyrost względem pełnego JVP oraz kontrolę drugiego rzędu lub dokładnego odczytu, a nie tylko względem pojedynczego gradientu.

## Inicjalizacja i modele zerowe

1. Ten sam instrument i selektor uruchomić na checkpoincie 0 każdego seedu. Porównywać rozkłady geometryczne, siłę efektów i dane o liniowości; w losowym detektorze IoU zwykle nie ma zakresu porównywalnego z wyuczonym, więc samo niższe IoU nie rozstrzyga mechanizmu.
2. Kontrola wyuczonych wag: permutacje aktywacji partnera wewnątrz tego samego obszaru i obrazu, z zachowaniem rozkładów; wielokrotne ustalone permutacje zamiast jednego przykładu. Osobno zachować strukturę czasową w kontrolach opóźnień.
3. Ograniczony bieg z permutowanymi etykietami jako diagnostyka „uczenie zadania vs samo dopasowywanie danych”. Mała kontrola nie daje pełnego porównania przy jednakowej jakości detektora; zaznaczyć ten brak.
4. Randomizacja głowy lub części wag to test zależności miary od zadaniowego odczytu, nie model o zachowanym działaniu. Nie traktować jako idealnie dopasowanego kontrolnego detektora.
5. Dokładna permutacja neuronów ukrytych, z odpowiednią permutacją wierszy W1, biasu i kolumn W2, ma zachować wyjście i wyniki przypisane do przestawionych identyfikatorów. To kontrola poprawności kodu i niezależności wniosku od numeracji neuronów.
6. Zachować kontrole numeryczne: pełny forward z hookiem vs szybkie wyłączenia, float64 blisko progów wyboru, dokładny dekoder, sprawdzenie GELU ze znakiem, jawne małe mianowniki.

Nie budować tezy na tym, że przeciwstawne kierunki istnieją; to spodziewany wynik selekcji w niskowymiarowym odczycie. Potrzebny jest **nadmiar efektu zależny od zadania lub kontekstu, jego transfer i funkcjonalna konsekwencja**.

## Kryteria falsyfikacji i decyzje po etapach

- Jeżeli kompensacja pojawia się równie łatwo w inicjalizacji i nie ma dodatkowego efektu semantycznego/transferu po dopasowaniu sił, odrzucić tezę o wyuczonym specjalnym układzie kompensującym.
- Jeżeli stałe pary nie przenoszą się na nowe filmy/checkpointy, nie nazywać ich trwałymi zespołami. Adaptacyjny instrument może nadal być przydatny, ale jest inną tezą.
- Jeżeli margines i pełne JVP wyjaśniają przełączenia równie dobrze, brak podstaw do tezy o nowej synergetycznej dynamice. Potencjalny wkład pozostaje w pomiarze/kompresji/benchmarku.
- Jeżeli cechy współdziałania nie poprawiają kalibracji i jakości predykcji na zablokowanym teście, nie promować ich jako wskaźnika jakości detekcji.
- Jeżeli wyniki znoszą się między seedami lub filmami, uogólnienie nie przechodzi; zwiększanie liczby tokenów nie naprawia tego problemu.
- Jeżeli „przełączenia” zwykle zachowują dobre IoU, wynik dotyczy redundancji kandydatów, nie niestabilności lokalizacji.
- Jeżeli podmiana aktywacji nie daje lepszego odzysku niż kontrole przy porównywalnej sile, nie ma potwierdzenia zadaniowej roli wybranego współdziałania.
- Jeżeli efekt zmian narzędzia jest taki sam po dopasowanych zmianach tła i artefaktach maskowania, hipoteza semantycznej specyficzności nie przechodzi.

W analizie predykcyjnej primary miarą może być log-loss/Brier i kalibracja, przy tej samej reprezentatywnej częstości zdarzeń; AUROC i AUPRC jako uzupełnienie. Dla rzadkich szkodliwych przełączeń podać bezwzględne liczby i przedziały klastrowe. Ustalić najmniejszy praktycznie ważny efekt na discovery, potem oszacować wymaganą liczbę filmów i zdarzeń. Podane niżej liczby są budżetem startowym, nie dowodem odpowiedniej mocy statystycznej.

## Budżet minimalny i rozszerzony

### Minimalny pakiet do decyzji go/no-go

- Jedna architektura, poprawny split, **3 niezależne treningi** i ich inicjalizacje. Dodatkowo **1 diagnostyczny trening z permutowanymi etykietami**, jeśli pozwala budżet; nie zastępuje replikacji głównej.
- Około **30 niezależnych filmów niewykorzystanych w uczeniu**, jeśli dostępne: 10 discovery, 5 calibration, 15 zablokowanego testu; około 40 rzadko wybranych klatek na film, razem 1200. Jeśli filmów jest mniej, raportować mniejszą niezależną próbę, nie zastępować jej gęstszymi klatkami.
- **20 z góry ustalonych checkpointów** na bieg do trajektorii i selekcji; dokładne interwencje przede wszystkim na **3 późnych, z góry wskazanych checkpointach**. Nie potrzeba od razu pełnego przekroju 806 checkpointów na każdym obrazie.
- Primary grupa **32 neurony, α=0,5**, A/B/A∪B, sześć rodzin selekcji oraz kilka dopasowanych kontroli losowych. Wybór tych wartości to propozycja do zamrożenia w discovery. Grupy 2 i 8 oraz α=0,1 i 1 jako badanie dawki na ustalonej mniejszej próbce, np. 10 procent testu, bez dobierania najciekawszych przypadków.
- Na reprezentatywnym, oznaczonym podzbiorze około **200 obrazów**: cztery rodziny zaburzeń, jedna bazowa siła, tool-vs-background i kontrola artefaktów; podmiana aktywacji na ustalonym podzbiorze, z pełnym raportowaniem nieudanych prób.
- Wynik tego etapu: wiarygodna decyzja, czy istnieje efekt większy od prostych baseline, oraz estymacja częstości zdarzeń potrzebna do zwiększenia próby. To nie automatycznie pełny pakiet pod dużą konferencję.

### Rozszerzony pakiet potwierdzający

- **5 seedów × 2 architektury ViT**, a przy twierdzeniu o ogólnym mechanizmie detekcji dodatkowo niezależny zbiór i/lub detektor z innym sposobem wyboru kandydatów. Dwa zbiory trenowane osobno w obu architekturach oznaczają **20 pełnych treningów głównych**, nie 10.
- Co najmniej około **50 niezależnych filmów na zbiór** i kilka tysięcy rzadko próbkowanych klatek, o ile dostępne; ostateczną wielkość ustalić według wariancji między filmami oraz częstości szkodliwych zdarzeń z etapu minimalnego.
- **40–60 checkpointów** na bieg; pełniejsza krzywa dawek/grup na stałej próbce; trzy poziomy zaburzeń, rodziny zaburzeń całkowicie wyłączone z kalibracji, większa próbka poprawnych masek narzędzia.
- Powtórzone label-null i pełne testy transferu reguły selekcji, mechanizmu oraz predykcji. Rozszerzenie na wcześniejszy blok wymaga ponownego liczenia dalszych bloków, a nie szybkiej samej głowy; zaplanować jako osobny droższy moduł.
- Osobny niezależny test zachowania detektora z innym dekoderem. Jeżeli efekt zależy wyłącznie od top1 konkretnej implementacji, odpowiednio zawęzić tytuł i twierdzenie.

### Jak planować wykonanie na Edenie

Bez pomiaru przepustowości i znajomości GPU nie podawać pozornie dokładnych GPU-godzin. Najpierw osobny mały benchmark przyszłego pipeline'u: czas forward, odczyt checkpointu, cache, dokładna interwencja głowy i pełny forward z hookiem. Obecnie żadnego benchmarku nie uruchomiono.

Koszt planować jako `liczba treningów × koszt biegu` oraz `seedy × checkpointy × obrazy × warianty interwencji × koszt odczytu`. Liczyć osobno tanie head-only i drogie perturbacje obrazu/full-forward. Dla ostatniego MLP można cache'ować wejście i odtwarzać wkłady oraz głowę; dla wcześniejszych bloków nie wolno używać tego skrótu bez przeliczenia reszty sieci.

Przykładowa wielkość zapisu float32 na obraz/checkpoint: stan 600×192 to około **0,46 MB**, a aktywacje 600×768 około **1,84 MB**. Nie zapisywać bezrefleksyjnie wszystkich aktywacji wszystkich obrazów i checkpointów. Zachować manifesty, wybrane stany potrzebne do odtworzenia oraz agregaty, a surowe duże tensory przechowywać warstwowo i dzielić po seedzie/checkpoincie. Różnice czasowe liczyć w float32; float16 dopuszczać do archiwum tylko po kontroli błędu.

Zadania na klastrze: trening per seed, ekstrakcja per checkpoint/shard filmów, head-only interwencje z cache, osobno perturbacje i pełne forward, na końcu agregacja CPU. Zapis atomowy wyników, manifest kodu i wag, brak nadpisywania pilota. Stopniować budżet: nie uruchamiać rozszerzonej macierzy treningów przed przejściem ustalonych kryteriów minimalnych.

## Kolejność prac

1. Audyt splitów i dostępnych etykiet; zamrożenie definicji zdarzenia, podziałów i protokołu.
2. Implementacja benchmarku interwencji z kontrolą pełnego forward i dekodera, bez rozbudowy GUI.
3. Minimalne biegi, inicjalizacje i baseline; zapis aktualizacji optymalizatora tylko jeśli późniejsza praca ma rzeczywiście dotyczyć przyczyn ruchu między checkpointami.
4. Test transferu stałych zespołów i ryzyka nowych zaburzeń. Wynik negatywny raportować przed dalszym rozszerzaniem hipotezy.
5. Potwierdzenie semantyczne i rescue na zablokowanych filmach; estymacja częstości i mocy.
6. Dopiero po udanym go/no-go: druga architektura/zbiór, więcej seedów i pełny pakiet publikacyjny.

Źródła lokalne: `outputs/wspoldzialanie_neuronow_20261015/RAPORT.md` oraz `outputs/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje/RAPORT.md`. Niniejsza notatka nie zastępuje przeglądu prior art dotyczącego neuron attribution, interakcji, robustness i zmian wyboru detekcji.
