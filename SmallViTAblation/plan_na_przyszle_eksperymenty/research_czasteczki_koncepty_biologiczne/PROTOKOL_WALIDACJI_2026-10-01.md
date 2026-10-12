# Protokół walidacji konceptu w pełnych tokenach

Data: 2026-10-01. Status: **propozycja; niczego tutaj nie wykonano**. Dokument uzupełnia research i nie zmienia równoległego planu przygotowania 400 par.

## A. Ustalenia przed oglądaniem wyników

Zamrozić checkpoint, hash wag i buforów BN, miejsce pomiaru w sieci, kod preprocessingu, rozdzielczość i kolejność tokenów. Model pracuje w eval, z zamrożonym BN i dropout wyłączonym. Powtórny identyczny forward ma służyć do określenia numerycznego poziomu szumu, nie do tworzenia niezależnych prób.

Zadanie semantyczne nazwać precyzyjnie: **obecność określonej kategorii narzędzia**, jej odróżnienie od innych instrumentów oraz odporność na pozę. Jeżeli zbiór zawiera tylko narzędzie i wygenerowany brak narzędzia, wolno oceniać obecność/sygnał edycji, ale nie pełną identyfikację kategorii ani fizycznego egzemplarza.

Podstawową obserwacją jest wektor tokenu 768D. Jeden obraz daje 600 obserwacji przestrzennych, ale statystycznie zależnych. Wnioskowanie grupować przede wszystkim po filmach/operacjach, nie po liczbie tokenów.

## B. Pary, maski i współrzędne

Dla obrazu \(a\) zachować osobno:

- oryginalny obraz \(x_a^+\);
- oryginalną maskę semantyczną \(M_a^{sem}\);
- maskę faktycznej edycji \(M_a^{edit}\), ewentualnie poszerzoną o ustalony margines;
- obraz po inpaintingu \(x_a^-\);
- identyfikator filmu, klatki, kategorii narzędzia oraz wersję i parametry edytora.

W naszej konwencji maska edycji równa 1 oznacza obszar podmieniany. Końcowy obraz składać jawnie:

\[
x_a^-=(1-M_a^{edit})\odot x_a^+
      +M_a^{edit}\odot G(x_a^+,M_a^{edit}).
\]

Konwencję masek biblioteki sprawdzić oddzielnie; nie zakładać jej na podstawie nazwy argumentu. Zapis bezstratny pozwala wymagać identycznych pikseli poza obszarem edycji. Przy miękkiej granicy do obszaru edycji należy cały niezerowy support alfy. Po resize/crop odnotować także pas wpływu interpolacji.

**Maska semantyczna wyznacza identyczny zbiór indeksów tokenów w obu obrazach.** Nie segmentować ponownie obrazu po usunięciu narzędzia w celu wyboru innego zbioru tokenów. Nie przenosić tokenu na „najbardziej podobny” token tła.

Dla footprintu wejściowego \(P_i\) tokenu:

\[
a_{ai}=\frac{|P_i\cap M_a^{sem}|}{|P_i|},\qquad
I_a(\tau)=\{i:a_{ai}>\tau\}.
\]

Footprint i kolejność przestrzenną trzeba odtworzyć z konkretnego preprocessingu i tokenizacji. Udział pikseli jest informacją o lokalizacji wejściowej. Nie jest udziałem czystego sygnału narzędzia w późnym tokenie: stem i attention mieszają kontekst. Nie dzielić automatycznie \(\Delta z\) przez \(a_{ai}\), aby otrzymać rzekomo „czysty token narzędzia”.

Proponowane ustawienie podstawowe: każdy dodatni udział, \(\tau=0\), i waga \(a_{ai}\). Z góry zadeklarować analizy czułości dla np. \(\tau=0{,}05, 0{,}25, 0{,}5\). Cienkie narzędzie może nie mieć ani jednego tokenu spełniającego wysoki próg; takie przypadki raportować, zamiast po cichu usuwać obrazy. Próg i wariant główny wybierać na części rozwojowej.

Wszystkie 600 tokenów nadal warto zachować, aby mierzyć przeniesienie wpływu na tło. **Identyczne piksele tła nie wymagają identycznych tokenów tła**, ponieważ późna reprezentacja ViT zależy od całej sceny.

## C. Kontrole edycji i swoistości

| Kontrola | Wykonanie | Jaką alternatywę sprawdza |
|---|---|---|
| Tożsamość pipeline | Identyczny obraz, wspólne operacje zapisu i preprocessingu; dodatkowo kontrola edytora z pustą maską, jeśli obsługiwana | Szum liczbowy, kodowanie obrazu, niezamierzona zmiana całości |
| **Sham tła** | Inpainting obszaru bez narzędzia, z maską dopasowaną do pola, kształtu, orientacji, położenia i charakteru tła | Czy odkryty kierunek jest odpowiedzią na samą edycję |
| Inny edytor / ustawienie | Alternatywny model inpaintingu; ustalone warianty marginesu maski | Zależność od konkretnego generatora lub obwódki |
| Seedy generatora | Tylko dla generatora, który rzeczywiście próbkuje różne wypełnienia | Wrażliwość na niepewność wygenerowanego tła |
| **Inne narzędzia** | Podobny metal, krawędzie, skala, kąty i lokalizacja; ich obecność i analogiczna edycja | Kategoria badanego narzędzia vs ogólna obecność instrumentu |
| Trudne negatywy | Odbłyski, nici, krawędzie, podobne struktury tkanki | Niskopoziomowe skróty |
| Naturalny brak narzędzia | Niezależne prawdziwe klatki bez narzędzia, dopasowane na poziomie filmu/kontekstu | Czy efekt przenosi się poza syntetyczne obrazy |
| Losowa inicjalizacja | Ten sam punkt odczytu i geometria w nietrenowanym backbone jako osobne porównanie | Ile separowalności wynika z wejścia i architektury |
| Kilka warstw | Niewielki wcześniej ustalony zestaw miejsc odczytu | Czy lokalizacja semantyki jest zgodna z przepływem obliczeń |

Standardowa LaMa nie dostarcza nowych próbek tła przez samą zmianę seeda. Nie raportować kilkunastu identycznych wyników jako replik generatywnych.

Jakość edycji oceniać bez znajomości znalezionego kierunku i wyniku detektora. Kryteria odrzucenia par ustalić wcześniej: pozostałości narzędzia, rażące uszkodzenia sceny, zmienione piksele poza zadanym supportem, błędna maska lub niedopasowanie klatki. Wszystkie odrzucenia i ich liczbę zachować. Nie wybierać tylko par, na których model już reaguje zgodnie z hipotezą.

Matched sham nie jest doskonałym odpowiednikiem nieznanego tła pod narzędziem. Różnica target–sham jest kontrolą diagnostyczną, a nie automatycznym identyfikatorem czystego efektu przyczynowego.

## D. Podział danych i brak przecieku

1. Dzielić **filmy/operacje**, zanim dopasuje się kierunki, normalizację, podprzestrzeń, progi i hiperparametry.
2. Wszystkie klatki jednego filmu, obie strony każdej pary i wszystkie warianty edycji pozostają w tym samym foldzie.
3. Część rozwojowa służy do fit, walidacyjna do ranku/liczby klastrów/progów, końcowa do zamrożonej oceny. Przy małej liczbie filmów zastosować grupowe cross-validation z wewnętrznym doborem parametrów.
4. Odłożenie obrazów od dopasowania konceptu nie oznacza, że backbone nie widział ich podczas treningu. Jeśli wszystkie 400 obrazów pochodzi z train backbone, oznaczyć to jawnie. Generalizacja całego modelu wymaga oddzielnych filmów nieużytych również do jego treningu.
5. Przed fit opisać zakresy kąta, pozycji, skali, jasności, filmu i narzędzia. Osobno testować nowe kombinacje oraz kąty nieobecne przy dopasowaniu.
6. Kąt w płaszczyźnie obrazu nie jest pełną orientacją 3D. Nie deklarować niezmienniczości na orientację przestrzenną wyłącznie z rotacji 2D lub augmentacji.
7. 400 par nie oznacza 400 niezależnych scen, a \(400\times600\) tokenów nie oznacza tylu niezależnych próbek. Raportować liczbę filmów, efekty per film i przedziały z bootstrapu/bloków na tym poziomie.

Analiza ponad jednym checkpointem jest osobnym rozszerzeniem. W pierwszym etapie zmieniamy wejście przy stałej sieci; nie mieszamy tego z ruchem tokenów powodowanym treningiem.

## E. Dopasowanie pełnych wektorów i porównania

Dane pierwotne:

\[
\Delta z_{ai}=z_{ai}^{+}-z_{ai}^{-}\in\mathbb R^{768}.
\]

Wagi nie powinny sprawiać, że pojedynczy duży obiekt lub film z wieloma podobnymi klatkami dominuje wynik. Przykładowa konwencja równej wagi filmów i obrazów w filmie:

\[
w_{ai}=\frac{1}{N_{\mathrm{film}}\,n_{f(a)}}
\frac{a_{ai}}{\sum_{j\in I_a}a_{aj}},
\quad i\in I_a.
\]

Jest to propozycja estymandu, którą należy zamrozić przed testem. Oddzielnie można raportować wynik ważony powierzchnią, lecz odpowiada on innemu pytaniu.

Porównać przynajmniej:

- brak wspólnej struktury: kierunki losowe / permutowane parowania jako kontrole;
- średnią ważoną pełnych różnic, jako prostą translację;
- H1: wspólny kierunek z różną siłą odpowiedzi;
- H2: podprzestrzeń ranku \(r\), z rzędem wybieranym na walidacji;
- H3: regularizowaną mapę \(z^+\approx Az^-+b\), jako odniesienie do FeatMap;
- separator CAV/LDA i kierunek pattern-CAV jako różne rodziny;
- pomocniczy model z informacją tylko o pozie, pokryciu maski, położeniu i normie jako kontrolę niskopoziomową.

Zachować surowe \(\Delta z\), ich normy i analizę kierunków po normalizacji. Mały wektor po normalizacji może amplifikować szum; próg małego ruchu ustalić na pomiarze powtarzalności/części rozwojowej.

Rozdzielić średni przyrost \(\mu_\Delta\) od kowariancji reszt. PCA po wycentrowaniu odpowiada na inne pytanie niż SVD niecentrowanej macierzy różnic. Utrzymywać pełne wektory, nawet jeśli raport pokazuje 2–3 składowe. T-SNE/UMAP i efektowny wykres nie są metryką potwierdzającą koncept.

Whitening, skalowanie, encoder/decoder i baza podprzestrzeni korzystają wyłącznie z fit. Edycje w znormalizowanej przestrzeni mapować z powrotem do oryginalnych współrzędnych, a dopasowanie norm kontroli opisać w tej samej metryce. Nie utożsamiać separatora z kierunkiem zapisu informacji.

## F. Metryki reprezentacji

1. Błąd odtwarzania **pełnych różnic** poza dopasowaniem, względem średniej translacji i modelu zerowego.
2. Cosine różnic i błąd norm; wynik również na małych ruchach, zamiast selekcji tylko największych zmian po teście.
3. Energia odłożonych różnic w podprzestrzeni:
   \[
   E_Q=\frac{\sum w_{ai}\|Q^\top\Delta z_{ai}\|^2}
              {\sum w_{ai}\|\Delta z_{ai}\|^2}.
   \]
   Wysoka wartość sama nie świadczy o semantyce. Porównać target, sham, inne narzędzia i tło.
4. Stabilność projektorów/kątów głównych między filmami, kątami i edytorami. Nie interpretować arbitralnego znaku wektora PCA.
5. Odłożona rozpoznawalność kategorii vs inne narzędzia i trudne negatywy; osobno lokalizacja względem prawdziwej maski.
6. Resztkowa przewidywalność typu edytora, położenia, kąta i filmu. Wykrywalność tych zmiennych nie przekreśla konceptu, ale wymaga sprawdzenia, czy to one tłumaczą deklarowaną generalizację.
7. Jeśli stosowane są INLP/RLACE/LEACE: zmiana dostępności informacji dla zdefiniowanego liniowego probe oraz osobny odłożony test nieliniowy. Nie ogłaszać pełnego usunięcia konceptu tylko po porażce jednego klasyfikatora.
8. Korekta wielokrotnych porównań lub jawne oznaczenie analiz eksploracyjnych dla wyboru warstw, progów, klastrów i kierunków.

## G. Interwencje: sprawdzenie, czy detektor wykorzystuje składową

Najpierw potwierdzić, że odczytywany tensor rzeczywiście leży na ścieżce do wyjścia głowicy. Podmiana kopii telemetrycznej lub tensoru po ostatnim użyciu nie jest interwencją w obliczenie. Wynik z ViT z głowicą korzystającą wyłącznie z CLS nie przenosi się automatycznie na detektor czytający tokeny przestrzenne.

Zdefiniować przed testem obserwowane wyjście: logit kategorii w ustalonym ROI, wynik detekcji przy stałej procedurze dopasowania predykcji do obiektu, lokalizacja oraz fałszywe alarmy na innych instrumentach/tle. Nie wybierać po interwencji najkorzystniejszego zapytania detektora.

Dla \(P_Q=QQ^\top\) i pełnego zarejestrowanego tensoru pary przeprowadzić:

\[
z_{ai}^{rescue}=z_{ai}^{-}+\alpha P_Q(z_{ai}^{+}-z_{ai}^{-}),
\]
\[
z_{ai}^{remove}=z_{ai}^{+}-\alpha P_Q(z_{ai}^{+}-z_{ai}^{-}).
\]

Interweniować na wcześniej wskazanych indeksach \(I_a\), zachowując pozostałe tokeny. Wprowadzić dodatnią kontrolę pełnego patchingu \(P_Q=I\), aby odróżnić nieskuteczną podprzestrzeń od niewłaściwego miejsca ingerencji.

To **diagnostyczna interwencja na znanej parze**: korzysta z obu reprezentacji. Nie jest jeszcze wdrażalnym odtwarzaniem brakującego obiektu z samego obrazu negatywnego. Transfer samego wyuczonego \(q\), \(Q\) lub mapy H3 do nowego obrazu stanowi osobny test bez dostępu do jego dodatniej pary.

Kontrole obowiązkowe:

- \(\alpha=0\), kilka ustalonych małych dawek i kierunek przeciwny;
- losowe kierunki/podprzestrzenie tego samego ranku, na tych samych tokenach, o **dopasowanej normie rzeczywistej zmiany aktywacji**;
- wcześniej ustalone ziarna losowych kontroli, bez wyboru najgorszej po teście;
- interwencje na tle o dopasowanej liczbie tokenów oraz podprzestrzeń uzyskana z sham;
- swoistość dla badanego narzędzia i brak porównywalnego załamania innych predykcji;
- diagnostyka wielkości/norm/stanu aktywacji oraz zakresów obserwowanych na danych; dopasowanie norm samo nie zapewnia pozostania na manifoldzie;
- zgodność efektu po alternatywnej edycji i na odłożonych filmach.

Spadek wyniku po ablacji może oznaczać uszkodzenie użytecznej informacji ogólnej. Rescue, dawka i swoistość wzmacniają interpretację, ale wynik nadal dotyczy określonej sieci, reprezentacji i rozkładu danych, nie uniwersalnej istoty konceptu.

## H. Co uznajemy za wsparcie, a co obala hipotezę

**Wstępne wsparcie dla sygnatury reprezentacyjnej:** wybrany wcześniej kierunek/subprzestrzeń przewiduje pełne różnice na nowych filmach i kątach oraz odróżnia badane narzędzie od kontrolnych edycji i innych instrumentów.

**Silniejsze wsparcie funkcjonalne:** selektywne usunięcie i przywrócenie składowej wpływa na odpowiednią detekcję wyraźniej niż norm-matched kontrola losowa, sham i interwencja na tle, bez globalnego uszkodzenia wyjść.

**Obalenie konkretnego wariantu:** efekt znika po zmianie edytora, po kontroli maski/odbłysków lub na nowych filmach; podprzestrzeń rozpoznaje edytor zamiast narzędzia; efekty ablacji są porównywalne dla losowych kierunków; wynik wymaga wyboru warstwy/kąta po obejrzeniu testu.

**Alternatywny wartościowy wynik:** H1 przegrywa, ale H2 lub H3 generalizuje. Oznacza to bogatszą, zależną od kontekstu strukturę, a nie niepowodzenie całego pytania o kodowanie.

Wynik rozwojowy na 400 wcześniej oglądanych obrazach należy tak nazwać. Pełne potwierdzenie wymaga zamrożenia metody przed niezależnym odłożonym sprawdzianem.

## I. Minimalne artefakty przyszłego wykonania

Manifest obrazów/masek/filmów i checkpointu; opis splitów i odrzuceń; para obrazów z mapą różnic pikseli; surowe pełne tokeny z identycznymi indeksami; wszystkie warianty edycji; wyuczone kierunki/projektory i parametry normalizacji; wyniki każdej kontroli per film/kąt; efekty interwencji wraz z rzeczywistymi normami i statystyką niepewności. Oddzielić wyniki potwierdzające od eksploracji.

Dokument zawiera projekt eksperymentu, a nie polecenie uruchomienia nowych treningów czy zadań klastra.

