# Ruch pojedynczych tokenów narzędzia podczas uczenia

Data: 2026-09-30. Status: plan i lokalny pomiar z istniejących wag, bez nowego treningu.
Autoryzacja: użytkownik poprosił o zapis planu i pomiar agentem w tle, z progiem choć jednego piksela maski.

## Pytanie

Czy trajektorie tokenów, których pola przecinają maskę cystotomu, wyznaczają kierunki zmiany reprezentacji odróżniające narzędzie od tła?
To test kandydatów na kierunki związane z narzędziem. Sam ruch, duża wariancja ani zgodność z maską nie dowodzą tożsamości semantycznej ani przyczynowego wykorzystania przez detektor.

## Co jest cząstką i co jest czasem

Pełna klatka ma wejście 480 × 320. Siatka 30 × 20 daje 600 tokenów. Każdy token ma 768 współrzędnych na wyjściu GELU MLP ostatniego bloku. Wszystkie 600 wektorów zachowujemy oddzielnie.
Ten sam obraz, bez augmentacji, jest analizowany w trybie eval przy kolejnych stanach wag jednego przebiegu uczenia. Czas oznacza numer aktualizacji optymalizatora, a nie numer klatki filmu.
Indeks tokena oznacza stałe miejsce siatki. Nie śledzimy tego samego fragmentu narzędzia przemieszczającego się w filmie.

## Dane i maski

Źródłem są pełne klatki odpowiadające oryginalnym sondom, ich adnotacje segmentacyjne i dostępne lokalne checkpointy tej samej sieci. Nie zastępujemy pełnej klatki wycinkiem 176 × 176.
Token jest związany z maską, gdy choć jeden piksel maski wejściowej wpada w jego pole 16 × 16. Zachowujemy także udział pokrycia, aby oddzielić tokeny brzegowe od mocno pokrytych.
Maskę źródłową przeskalowujemy z zachowaniem cienkich fragmentów: dodatni udział maski źródłowej daje dodatnią maskę na wejściu. Kryterium dotyczy maski po przeskalowaniu. Nie jest to twierdzenie, że pole 16 × 16 jest pełnym polem recepcyjnym tokena.
Kontrola: próg 25 procent na tych samych obrazach, checkpointach i tokenach.
Tło kontrolne: pola bez maski, oddalone od jej brzegu. Zapobiega to uznaniu tokenów mieszanych za czyste tło.

## Dostępność historii

Pożądana historia to stany co 10 aktualizacji do kroku 1000, bez średnich po pięć pomiarów. Nie mylimy pomiaru aktywacji z zapisanym checkpointem.
Istniejące historie średniej po tokenach nie pozwalają odzyskać pojedynczych tokenów. Jeśli nie ma gęstych wag ani surowych tokenów, wykorzystamy rzeczywiście dostępne stany i jawnie podamy odstępy. Nie odtwarzamy braków treningiem ani interpolacją.
Co najmniej dwa stany umożliwiają pomiar przemieszczenia. Dwa lub kilka stanów nie wystarczą do wiarygodnego wyznaczenia pamięci albo prognozy Koopmana.

## Pomiar i kierunki

Zapisujemy Z[t, obraz, token, cecha] oraz maski, numery kroków, identyfikatory obrazów i źródła wag. Bez uśredniania Z po tokenach lub czasie.
Przemieszczenie to różnica Z[t+1] − Z[t]. Gdy odstępy są nierówne, oddzielnie pokazujemy przemieszczenie oraz zmianę na jedną aktualizację wag.
W każdym obrazie uczącym i przedziale odejmujemy od zmian tokenów średnią zmianę czystego tła. Jest to kontrola wspólnego dryfu całej reprezentacji, nie nowe uśrednianie trajektorii narzędzia. Na obrazach odłożonych odejmujemy referencyjny dryf tła wyznaczony wyłącznie z obrazów uczących w tym samym przedziale. Nie używamy masek testowych do przekształcania cech; służą wyłącznie ocenie. Ten test dotyczy przenoszenia kierunków na inne obrazy w tym samym czasie, nie prognozy przyszłego czasu.
Na części obrazów wyznaczamy przez SVD główne kierunki niecentrowanych, skorygowanych zmian tokenów narzędzia. Zachowujemy wiele kierunków, ich energię i składowe w 768 wymiarach. Każdy obraz i przedział mają równy łączny udział w dopasowaniu.
Osobno zapisujemy kierunek średniego ruchu narzędzia względem tła, zachowujący znak. Osie SVD mają arbitralny znak; nie interpretujemy go jako aktywacji tożsamości.

## Ocena poza dopasowaniem

Podział po całych źródłowych filmach, nigdy po losowych tokenach. Kierunki dopasowane na filmach uczących oceniamy na odłożonych filmach.
Mierzymy energię rzutu zmian narzędzia i tła na znalezioną podprzestrzeń, udział pokrycia maski oraz separację rozkładów wyników (ROC AUC). Ranking używa wyłącznie wektora zmiany, etykiety służą do dopasowania i oceny.
Podłogi: norma całego ruchu, rzut na losową ortonormalną podprzestrzeń tego samego wymiaru i brak zmiany. Losowe kierunki zachowują dane i wymiar; maska 25 procent zmienia tylko definicję tokenów narzędzia. Nie zakładamy, że podłoga musi wypaść gorzej.
Ruch uznamy za nieswoisty, jeżeli kierunki wybrane na narzędziu nie rozdzielają jego zmian od zmian tła na odłożonych filmach albo nie przewyższają prostej normy ruchu.
Dobór liczby kierunków opisujemy jako eksploracyjny. Bez wybierania najlepszego wymiaru na teście i przedstawiania jego wyniku jako niezależnego potwierdzenia.

## Prognoza bez uśredniania

Jeżeli surowa historia będzie wystarczająco gęsta, następny etap dopasuje model zmian do nieuśrednionych punktów pojedynczego obrazu. Projekcje i współczynniki będą wyznaczane wyłącznie na przeszłości. Wariant surowy i uśredniony trzeba ocenić na wspólnych celach, horyzontach w aktualizacjach i porównywalnym czasie pamięci.
Bez dostępnej historii nie deklarujemy wykonania tego etapu. Pomiar kierunków ruchu i prognoza są osobnymi wynikami.

## Granice wniosku

To analiza zapisanych wag, nie poprawa treningu detektora. Nie identyfikuje automatycznie fizycznych cząstek, unikalnej tożsamości cystotomu ani przyczynowych neuronów. Do silniejszego wniosku potrzebne są inne narzędzia jako negatywy, zmiany tła i interwencje w aktywacje z kontrolą losowych kierunków.
Raport zachowuje także wyniki negatywne i braki danych. Bez kasowania źródeł, modyfikowania wcześniejszych eksperymentów, uruchamiania klastra ani commitowania.
