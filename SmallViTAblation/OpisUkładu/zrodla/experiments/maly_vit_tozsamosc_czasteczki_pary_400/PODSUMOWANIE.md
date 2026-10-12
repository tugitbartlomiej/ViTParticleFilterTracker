# Cząsteczki i makrowektory związane z obecnością narzędzia

Analiza z 1 października 2026 r. na 400 parach: oryginał z anotowanym narzędziem oraz jego wersja po usunięciu narzędzia przez LaMa.

**Wyznaczyliśmy podprzestrzeń, z której można odczytać obecność narzędzia, oraz cząsteczki, których zmiana wpływa na detekcję.** Są to dwa uzupełniające się wyniki. Sama duża zmiana cząsteczki nie przesądza o jej wpływie na decyzję detektora. W tym eksperymencie „tożsamość” oznacza operacyjnie obecność anotowanego cystotome względem jego syntetycznego usunięcia. Nie przeprowadzono jeszcze rozróżniania cystotome od innych rodzajów instrumentów.

Przeliczono wszystkie 800 obrazów przez ten sam zamrożony model: checkpoint po 1000 aktualizacji, ziarno 47, model uczony od zera według przepisu rozgrzewki. Z każdego obrazu uzyskano 600 tokenów po 768 współrzędnych. Zachowano pełne wektory obu członów pary dla wszystkich 6729 cząsteczek przecinających maskę narzędzia oraz dla 3200 kontrolnych cząsteczek tła. Udział pikseli narzędzia w patchu określał wagę jego udziału w analizie. Mierzymy tu reakcję zamrożonej sieci na zmianę obrazu; zmian między checkpointami ten pomiar nie obejmuje.

**Które makrowektory?** Poniżej osie o najbardziej powtarzalnej zmianie przy porównaniu oryginału z wersją LaMa. Ranking ustalono na części treningowej analizy; zgodność znaku sprawdzono na odłożonych obrazach. Numery osi w tej tabeli liczymy od 1. W plikach zapisano także odpowiadające im indeksy programistyczne od 0.

| Makrowektor | Odpowiedź przy obecności narzędzia względem LaMa | Zgodność znaku zmiany na obrazach testowych |
|---:|---|---:|
| 500 | większa wartość składowej | 96,5% |
| 472 | mniejsza wartość składowej | 96,5% |
| 735 | mniejsza wartość składowej | 97,6% |
| 512 | mniejsza wartość składowej | 96,5% |
| 647 | mniejsza wartość składowej | 92,9% |

Obecność narzędzia zwiększa jedne składowe, a zmniejsza inne. Ranking wskazuje powtarzalne odpowiedzi osi; nie dowodzi, że każda z nich samodzielnie odpowiada za rozpoznawanie narzędzia.

Wyznaczona podprzestrzeń ma **osiem kierunków będących kombinacjami wielu makrowektorów**. Nie jest to wybór ośmiu pojedynczych osi spośród 768. Osiem kierunków zachowuje 60,53% ważonej energii różnic na danych użytych do dopasowania. Ta liczba nie jest procentem „wydobytej tożsamości”.

**Czy sygnał działa na innych filmach?** Podział obejmował 240 par do dopasowania, 75 do wyboru ustawień i 85 do końcowej oceny, odpowiednio z 69, 23 i 24 rozłącznych filmów. Oba obrazy pary i wszystkie klatki danego filmu pozostawały w jednej części.

| Odczyt z reprezentacji | AUROC na części testowej | Zrównoważona trafność na części testowej |
|---|---:|---:|
| Osiem kierunków SVD i klasyfikator liniowy | 0,9686 | 88,61% |
| Regularizowana LDA wykorzystująca 768 współrzędnych | 0,9794 | 95,28% |

Przedział niepewności 95% dla trafności pierwszej metody wynosi 83,37–93,30%, przy losowaniu całych filmów. Wynik LDA jest lepszy w tym zadaniu odczytu obecności; nie należy przedstawiać podprzestrzeni ośmiowymiarowej jako najlepszego klasyfikatora. Jej zaletą badawczą jest zachowanie kilku kierunków zmian. Odczyt zna obszar narzędzia z anotacji. Wszystkie źródłowe zdjęcia pochodzą ze zbioru uczącego detektora, więc rozdzielenie filmów dotyczy nowej identyfikacji podprzestrzeni, a nie niezależnego testu wytrenowanego ViT.

**Które cząsteczki?** Ich numery zależą od położenia narzędzia na danym zdjęciu. Dla wszystkich 400 obrazów zapisano ranking reakcji, a na sześciu wybranych z góry przykładach testowych także wpływ pojedynczej cząsteczki na wynik detektora.

Na obrazie `TR_0053_S1_P03/00000881`:

- cząsteczka **280** najmocniej zmieniała się w wyznaczonej podprzestrzeni;
- cząsteczka **248** miała największy zmierzony wpływ na wskaźnik pewności detekcji pomnożonej przez IoU: jej pojedyncza podmiana zmniejszyła go o 0,02845;
- podmiana cząsteczki 280 nie zmieniła tego wskaźnika w tym przykładzie. Nie oznacza to, że nie zawierała informacji o narzędziu.

![Reakcja cząsteczek i ich wpływ na wynik detektora](HEATMAP_008.png)

Powyższa mapa dotyczy ostatniej warstwy MLP. Po niej głowica działa osobno dla komórek, dlatego wpływ na wykryty środek ramki może skupiać się w jednej cząsteczce. Informacja z innych miejsc obrazu mogła już wcześniej dotrzeć do niej przez attention. Dla dwóch przypadków sprawdzono, że szybkie liczenie wpływu pojedynczej komórki daje dokładnie taki sam wynik jak rzeczywista interwencja w tokenie sieci.

**Wpływ całej podprzestrzeni na detektor.** W maskowanych cząsteczkach przesunięto tylko składową należącą do ośmiu kierunków, od stanu oryginału w stronę stanu obrazu LaMa. Na 85 obrazach testowych średni wskaźnik confidence × IoU spadł z 0,36232 do 0,34303. Przedział 95% dla zmiany wynosi od −0,02763 do −0,01230. To potwierdza wpływ tej interwencji na badany wskaźnik.

AP50 spadło z 45,76% do 44,11%, ale przedział niepewności zmiany wynosi od −6,25 do +4,11 punktu procentowego. Ten pomiar **nie rozstrzyga**, czy interwencja obniża AP50 w sposób powtarzalny. Podobnie nie wykazano jednoznacznego wzrostu straty.

Porównanie z losowymi kierunkami obejmuje dwa sposoby wyrównania siły zmiany: w pełnym wektorze 768D oraz po projekcji fc2 do 192 wymiarów. W drugiej kontroli wykonano dodatkowe 255 rzeczywistych przejść przez sieć. Trzy losowe podprzestrzenie dawały confidence × IoU od 0,35525 do 0,35855, wobec 0,34303 dla wyznaczonej podprzestrzeni. Różnica względem średniej tych kontroli wynosi −0,01359, z przedziałem 95% od −0,01922 do −0,00842. Większy wpływ na ten wskaźnik utrzymał się zatem po wyrównaniu normy zmiany za fc2. **Nie uzyskano analogicznego rozstrzygnięcia dla AP50.** Wyniki kontroli zapisano w [DETEKCJA_FC2.json](DETEKCJA_FC2.json).

**Kontrola LaMa.** W dodatkowych 24 obrazach z 17 odłożonych filmów uzupełniono inny obszar, pozostawiając piksele narzędzia bez zmian. Średnia reakcja w podprzestrzeni na tokenach narzędzia wynosiła 0,159, a po usunięciu samego narzędzia na tych samych zdjęciach 3,562. Żadna z 24 edycji kontrolnych nie zmieniła binarnej decyzji odczytu, choć wartości jego prawdopodobieństwa mogły się zmieniać. Ten wynik osłabia wyjaśnienie, że mierzymy wyłącznie dowolną edycję LaMa. Nie usuwa wszystkich różnic wynikających z treści i miejsca uzupełniania.

Zapisane wyniki pozwalają wskazać kandydatów na kierunki obecności narzędzia, mierzyć odpowiedź każdej cząsteczki i wykonać kontrolowane interwencje. Rozróżnianie innych instrumentów, niezależność od metody usuwania oraz dynamika między checkpointami wymagają osobnych pomiarów.

Pełne liczby i procedura: [WYNIKI.md](WYNIKI.md). Ranking wszystkich osi: [MAKROWEKTORY.json](MAKROWEKTORY.json). Rankingi cząsteczek: [reakcja](CZASTECZKI_RANKING.json) i [wpływ na detektor](CZASTECZKI_WPLYW.json). Macierz ośmiu kierunków i parametry odczytu: [PODPRZESTRZEN.npz](PODPRZESTRZEN.npz). Niezależna weryfikacja: [AUDYT_ASTRA.md](AUDYT_ASTRA.md).
