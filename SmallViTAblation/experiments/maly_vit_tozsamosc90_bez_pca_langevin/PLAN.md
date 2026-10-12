# Rozpoznanie narzędzia ponad90% i Langevin bezPCA

Data: 2026-10-02. Nowe zlecenie i doprecyzowanie użytkownika: kierunki RAZEM mają rozpoznawać obecność narzędzia z trafnością ponad90%. Nie chodzi o90% energii. Użytkownik zabronił PCA na kierunkach tożsamości i nakazał obliczyć Langevina na tych współrzędnych bezPCA. Eksperyment badawczy, eksploracyjny, odrębny katalog według skilla eksperyment. Nie modyfikujemy dawnych wyników.

## Co znaczy bezPCA

Kierunki tej próby to wybrane ORYGINALNE osie e_j w przestrzeni768D. Dla każdej cząsteczki zachowujemy wektor jej współrzędnych z_j na wybranych osiach. Nie wyznaczamy PCA/POD, nie obracamy bazy przezSVD, nie używamy autoenkodera ani wcześniejszej macierzy U500. Skalowanie każdej współrzędnej osobno na treningu jest dozwolone, ale fizyczne równania i błędy wracają do oryginalnych jednostek. Dobór kanałów jest nadzorowany rozpoznaniem narzędzia.

Zmienia to definicję stanu wobec wcześniejszego rzutu na8 kierunków. Sygnał obecności mierzymy wspólnie w wybranych kanałach; nie zakładamy, że każdy pojedynczy kanał musi sam osiągać90%. Nie twierdzimy też, że znaleźliśmy jedyny lub pełny zbiór wszystkich semantycznie istotnych kierunków przestrzeni.

## Etap1: rozpoznanie na parach obrazów

- Źródło: 400 par oryginał/LaMa z maly_vit_test_orientacji_narzedzia_400, wcześniej zmierzonych przez zamrożony checkpoint500.
- W pliku IDENTYFIKACJA_K500.npz poprzedniego eksperymentu pooled400 ma shape400×2×768 i zawiera pełne współrzędne przed rzutem U. Odczytujemy tylko te surowe dane; U500 i wartości własne nie są wejściem.
- Zachowujemy wcześniejszy podział po filmach: train240par/69filmów, val75par/23filmy, test85par/24filmy. Sprawdzić SHA, indeksy i rozłączność filmów.
- Każdy obraz opisuje ważona pokryciem średnia tokenów ze ZNANEJ maski narzędzia. Klasyfikator odróżnia oryginał od syntetycznej wersji z usuniętym narzędziem. Wynik to rozpoznanie w znanym ROI, nie swobodna detekcja całego obrazu ani pomiar pojedynczego tokenu.
- Wagi filmów równe, obrazów w filmie równe, obie klasy równe. Główna trafność to film-weighted balanced accuracy; w tym parowym zrównoważonym zadaniu odpowiada ważonej accuracy. Próg prawdopodobieństwa0.5 stały. Pokazać też AUROC, sensitivity/specificity, macierz pomyłek oraz zwykłą nieważoną accuracy.
- Train-only ranking kanałów: bezwzględna ważona średnia różnicy parowej podzielona przez ważone SD tej różnicy. Lista K=1,2,4,8,16,32,64,128,256,384,512,768; dla każdego logistyczny klasyfikator L2,C=.01/.1/1/10. Rankingi, centra i skale wyłącznie z train.
- Selekcja: najmniejsze K spośród przebadanych, dla którego val accuracy jest STRICTLY>0.90; przy tymK wybrać największąval accuracy, potem AUROC, następnie mniejszeC. Jeśli taki zestaw nie istnieje, przed odczytem testu można jawnie zadeklarować drugą train-only ścieżkę nadzorowanąL1, nadal na oryginalnych kanałach. Jeśli żadna nie osiągnie celu, wybrać najlepszyval do diagnostyki i jasno oznaczyć niespełnienie90%.
- Kanały i klasyfikator zamrozić i zahashować przed testem. Nie refitować klasyfikatora na val, żeby zmiana nie dotyczyła tylko dopasowania progu. Jednorazowy test na24filmach. Wynik>90% na train/val nie wystarcza do twierdzenia o testowym rozpoznaniu. Nie poprawiać kanałów/C/progu po zobaczeniu testu.
- Raportować niepewność przez bootstrap CAŁYCH filmów, nie pojedynczych tokenów. Osobno punktowe przekroczenie90% i ewentualną dolną granicęCI>90%; nie utożsamiać tych twierdzeń.

Klasyfikator może wykorzystywać artefaktyLaMa i znaną maskę, więc trafność obecności nie dowodzi pełnej tożsamości konkretnego typu narzędzia ani przyczynowej odpowiedzialności osi. Stary podziałtest był już oglądany w poprzednich analizach; zachowujemy rozłączność w algorytmie, lecz nazywamy wynik eksploracyjnym.

## Etap2: równanie zmian wybranych współrzędnych

Nie czekamy na idealny wynik, aby policzyć żądaną dynamikę. Nawet gdy etap1 nie potwierdzi90%, dopasowujemy Langevina dla zamrożonego kandydata, oznaczając status progu uczciwie. Kanałów nie wybieramy według łatwości prognozowania przyszłego latentu.

Źródło: LATENT_GESTY.npy,1001stanów,seed47. Główna sonda2,31tokenów z maski, i transfer na sondę8,7tokenów. Stan x_i(t)=z_i,J(t)−z_i,J(500), gdzieJ to wybrane natywne kanały. Kontekst to średnia źródłowej maski tła w tych SAMYCH kanałach; może zawierać inne obiekty. Wagi tokenów coverage znormalizowane do sumy1. Wszystkie wybraneJ, bez dalszego PCA i bez redukcji wektora celu.

Wariant lokalny:

```math
x_{i,t+dt}=x_{i,t}+b+A x_{i,t}+\sum_{l=0}^{L-1}K_l\Delta x_{i,t-l dt}+\epsilon_{i,t+dt}.
```

Wariant sprzężony dodaje bieżącą średnią narzędzia f, średnią kontekstu g oraz ich historie przyrostów. Osobny operator prognozuje g. f zawsze wyliczamy z prognoz cząsteczek, nie jest osobnym niezależnie prognozowanym stanem. Wszyscy otrzymują przyrosty z tego samego bieżącego stanu i aktualizują się synchronicznie. Wspólne człony nie stanowią identyfikacji odmiennych oddziaływań pomiędzy każdą parą tokenów.

Nie podajemy przyszłych obrazów, gradientów, q, własnych przyszłych kanałów ani prawdziwych przyszłych średnich. Brak historii batch-ID/augmentacji/AdamW tego biegu; nie używamy ich z innych przebiegów. Ruch obejmuje zmiany wag i statystykBN.

## Protokół dynamiki, zamrożony przed testem

- dt1: fit target501..700; validation origins700..749, celeH1/10/50<=799. Refit operatorów do799 po wyborze. Test origins800..950 wspólny dlaH1/2/5/10/20/50.
- dt10: fit targets510..700co10; validation origins700..740co10,H10/50; refitdo790; test800..950co10,H10/20/50.
- Po poznaniu K na danych klasyfikacji, ale przed dopasowaniem dynamiki, zapisać konkretną siatkę i koszty. Startowy zamiar: dwa typy operatora (lokalny/sprzężony), L0/1/2/4/8 i ridge.01/.1/1/10; ograniczenia rachunkowe nie mogą usuwać wybranych współrzędnych z celu. Regularyzacja, diagonalny lub blokowy operator jest dozwolony, jeśli konieczny i zadeklarowany przed walidacją.
- Główne kryterium wyboru: równoważna średnia redukcji MSE względem persystencji naH1/10/50, analogicznieH10/50dla dt10. Osobny wybór dla najkrótszegoH. Równolegle raportować znak wszystkich składowych i normowanycosine wektora zmiany.
- Podłogi na DOKŁADNIEtymJ: persystencja, stała prędkość, równanieL0(bez jawnej pamięci). Nowe modeleporównujemy do tych samych prawdziwych kanałów, z identycznymi originami i wagami.
- Nie zestawiać MSE ani trafności nowego K z dawnym 8D tak, jakby cel był ten sam. W tym eksperymencie całkowicie pomijamy wcześniejszą bazę U8, również w rekonstrukcjach kontrolnych. Wszystkie porównywane modele dynamiki prognozują te same oryginalne kanały J.
- Transfer8 po wyborze bez refitu operatora lub doboru kanałów. Jedno ziarno, dwa obrazy, ocena eksploracyjna.
- Kontrole: prefix i poison całej przyszłości (także średnich), zgodność macierzy oryginalne/skalowane, skończoność, rank/condition/parametercount, wspólne originy co10 dla porównania dyskretyzacji. Resztyone-step analizować osobno od wielokrokowych błędów. Normalność i czasowa białość są odrębne; nie przyjmujemyFDT ani dokładnego zamknięciaMZ.

## Odpowiedzialność i zapis

### Zamrożona siatka dynamiki po wyborze kanałów, przed jej dopasowaniem

Zapis 2026-10-02, około 02:08 czasu lokalnego. Walidacja rozpoznania wybrała K = 128 oryginalnych kanałów oraz C = 0.01. Trafność walidacyjna ważona po filmach wynosi 90.6702898551 procent. Test rozpoznania pozostaje zadaniem niezależnego audytu; jego wynik nie zmienia J ani siatki dynamiki.

Ostateczna siatka: wariant lokalny i sprzężony, L = 0, 1, 2, 4 oraz ridge = 0.01, 0.1, 1, 10. Daje to 32 konfiguracje dla dt = 1 i 32 dla dt = 10. Wszystkie 128 wybranych kanałów pozostaje zarówno stanem każdej cząsteczki, jak i celem oceny. Używamy pełnych macierzy z regularyzacją, bez podziału kanałów na niezależne bloki. Pomijamy L = 8 ze względu na koszt estymacji szerokich operatorów i krótki dostępny odcinek czasowy. Kryteria wyboru i podział czasowy pozostają takie jak powyżej. Rozkład macierzy użyty wyłącznie do rozwiązania układu regresji nie może zmieniać bazy ani usuwać wymiarów stanu.

Próg rozpoznania jest oceniany dla checkpointu 500, z którego pochodzą pary obrazów. Nie stanowi gwarancji tej samej trafności klasyfikatora na każdym późniejszym checkpointcie. Prognozowanie zmian w kanałach wybranych na k500 jest osobnym zadaniem i ma osobne metryki.

Root: PLAN.md i PODSUMOWANIE.md. Wykonawca plan_poprawek: kod, deklaracjaWYNIKI.json, małe artefakty, raport/README/wykresy. Astra: AUDYT_ASTRA.md, niezależna kontrola przed testem rozpoznania i przed testem dynamiki.

Przed uruchomieniem zmierzyć dostępny RAM/dysk. Czytać pełny latent przezmmap, nie kopiować22GB. Zachować>1GBwolnego, nie kasować, nie pobierać, nie trenowaćViT, nie instalować, nie commitować. RuntimeF:/Python/python.exe, BLAS4. Zachować kanały0/1, współczynniki klasyfikatora, wszystkie valwyniki, równania w oryginalnych jednostkach, surowe metryki/CI, prognozy, hashe i wersje. Wynik negatywny jest wynikiem i musi być zapisany.
