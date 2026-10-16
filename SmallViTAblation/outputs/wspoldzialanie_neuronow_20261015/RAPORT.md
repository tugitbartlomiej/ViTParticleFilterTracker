# Współdziałanie neuronów ostatniego MLP a ramka detektora

Raport z rzeczywistych interwencji na checkpointach, 15 października 2026.

**Pomiar działa: można znaleźć neurony o dużych osobnych skutkach, które przy wspólnym wyłączeniu niemal się znoszą. Wpływ na ramkę zależy jednak od bieżącego obrazu, tokenu, checkpointu i głowy detektora. Samo istnienie kompensacji występuje już na początku biegu i nie dowodzi odkrycia mechanizmu wyuczonego podczas treningu.**

Najciekawszy zaobserwowany efekt funkcjonalny to przełączenie tokenu wybierającego zwycięską ramkę: dwie łagodne osobne interwencje wspólnie przekroczyły granicę wyboru detekcji. Jest to jeden przykład wybrany po analizie, a nie oszacowanie częstości w całym zbiorze.

## Co dokładnie policzono

Główna analiza: 34 checkpointy od 1000 do 65640, trzy zapisane obrazy, osiem tokenów najbliższych środkowi GT na obraz. Dla każdej wybranej pary wyłączałem pierwszy neuron, drugi neuron oraz oba razem. Siły wyłączenia: 10, 50 i 100 procent aktywacji. To 127,296 rekordów lokalnych i 15,912 rekordów globalnego wyłączenia tych samych identyfikatorów we wszystkich 600 tokenach. Osobne interwencje są zapisane w tych samych rekordach. Dodatkowo: 12,852 rekordów dla grup 8, 16 i 32 neuronów, kontrola checkpointu 0 oraz niezależne odtworzenie dużego skoku ramki.

Są to skorelowane pomiary jednego biegu treningowego, a nie tysiące niezależnych eksperymentów. Trzy obrazy nie pozwalają oszacować mAP ani uogólnić wyniku na wszystkie operacje i narzędzia. Obraz 3 został wyłączony z doboru zamrożonych par; **nie oznacza to, że był wyłączony z treningu modelu**. Pochodzenie filmowe i niezależność tych trzech obrazów nie zostały tu ustalone.

Badane miejsce to ostatni blok: 768 aktywacji ukrytego MLP działa na 192-wymiarowy strumień residual. Po nim pozostają LayerNorm i głowa działająca osobno na każdy token. Dlatego można dokładnie odtworzyć skutek wyłączenia bez ponownego liczenia wcześniejszych bloków. Aplikacja i checkpointy nie zostały zmienione.

## Co oznacza współdziałanie

Wkład neuronu do reprezentacji przed głową wynosi

`u_c = z_c W2[:,c]`.

Dla odczytu ramki `b(x)` w czterech współrzędnych xyxy, w pikselach, jego lokalny kierunek działania to

`a_c = J_b(x) u_c`.

W ten sposób „kierunek wiązki” w residual zostaje przeliczony na kierunek zmiany konkretnej ramki. Sam kąt między kolumnami W2 tego nie określa. Znaczenie mają także znak i siła aktywacji oraz Jacobian głowy w bieżącym stanie.

Rzeczywisty skutek ablacji liczę przez pełną funkcję głowy:

`D_c = b(x - alfa*u_c) - b(x)`

`D_cd = b(x - alfa*(u_c+u_d)) - b(x)`.

W raporcie `q = ||D_cd|| / (||D_c||+||D_d||)` mówi, jaka część sumy długości osobnych efektów pozostaje po ich połączeniu. Małe q oznacza kompensację. To **nie jest procent wyjaśnionej wariancji ani procent usuniętej energii**. q może przekroczyć 1 przez nieliniowość. Rozmiar zmiany ramki to RMS czterech współrzędnych, czyli norma tego wektora podzielona przez 2; nie jest to sam ruch środka ramki.

Osobno mierzę `I = D_cd-D_c-D_d`. Dopiero ta reszta oznacza nieaddytywność interwencji. Dwie zgodne strzałki, które po prostu się dodają, nie są jeszcze nieliniową synergią.

## Pary: rzeczywiste znoszenie, ale małe efekty bezwzględne

Pary kierunkowe wybierano spośród 64 największych wpływów `||a_c||`, z podobnymi normami obu członków. Porównanie losowe dopasowano do norm wpływów; nie jest to porównanie silnych neuronów z dowolnymi nieaktywnymi neuronami. Pary wskazuje Jacobian, a wyniki poniżej pochodzą z pełnych, skończonych ablacji.

Mediany dla kroków od 55000, pełnego wyłączenia i wszystkich 24 tokenów:

| Dobór par | Pozostała część efektów q | Zmiana ramki RMS px | Względna interakcja nieliniowa |
| --- | --- | --- | --- |
| Pary kompensujące ramkę | 0.0841 | 0.0093 | 0.0021 |
| Kontrola o podobnej sile | 0.7010 | 0.0733 | 0.0020 |
| Pary wzmacniające ramkę | 0.9994 | 0.1184 | 0.0024 |
| Kontrola wzmocnienia | 0.7331 | 0.0764 | 0.0022 |
| Przeciwne wkłady w residual | 0.2016 | 0.0277 | 0.0046 |
| Największa czułość ramki | 0.7465 | 0.2842 | 0.0158 |
| Największe wkłady w residual | 0.7882 | 0.2756 | 0.0249 |
| Pary losowe | 0.9538 | 0.0148 | 0.0001 |

W parach kompensujących pozostaje około **8,4 procent** sumy długości osobnych efektów, wobec około **70,1 procent** w kontroli o podobnej sile. Pary wzmacniające zachowują około **99,9 procent** sumy, wobec około **73,3 procent** w ich kontroli. Efekt jest obecny na wszystkich trzech obrazach: mediany q dla kompensacji to 0,069; 0,075; 0,103.

To potwierdza użyteczność kierunkowej mapy wpływów do przewidywania skutku interwencji. Nie jest niezależnym dowodem niezwykłej organizacji sieci: sam selektor celowo szuka skrajnych kątów w czterowymiarowej przestrzeni odczytu.

Przykład ustalony bez wybierania największego efektu: obraz 3, ostatni checkpoint, token najbliżej środka GT. Neuron **760** osobno zmienia ramkę o **0,1728 px RMS**, neuron **27** o **0,1617 px**, a oba razem tylko o **0,0279 px**. Indeksy neuronów są od zera. Druga para, 663 i 557, daje odpowiednio 0,0685; 0,0650; 0,1330 px. To bezpośrednie potwierdzenie dwóch przeciwnych sposobów sumowania wpływów.

![Ablacje par](przyklad_ablacji.png)

Dla par kierunkowych mediana nieliniowej reszty to około **0,21–0,24 procent** sumy osobnych efektów. Dziewięćdziesiąty percentyl jest blisko 1 procenta. Dla największych wkładów residual reszta bywa dużo większa: mediana 2,49 procent, percentyl 90 około 25,9 procent. Nie można więc uznać całej głowy za liniową; małe wybrane interwencje są po prostu zwykle dobrze przybliżane lokalnie.

Kontrola siły ablacji w float64 pokazuje dla par kompensujących medianę `||I||/2` około 0,0000025 px przy alfa=0,1 i 0,000258 px przy alfa=1. Wzrost bliski kwadratowemu jest zgodny z lokalną krzywizną odczytu. Nie wymaga interpretacji optycznej.

![Wyniki lokalne](wyniki_interwencji.png)

## Większe grupy i rzeczywiście wybierana ramka

Grupy są eksploracyjnym rozszerzeniem po stwierdzeniu małych efektów par. Tworzą je rozłączne pary z 128 najsilniejszych wpływów. „Wzmacniająca” oznacza zgodność wewnątrz par; algorytm nie wymusza jednego wspólnego kierunku wszystkich 32 neuronów. Połowa A zawiera po jednym członku każdej pary, połowa B pozostałych.

Poniżej globalne wyłączenie we wszystkich tokenach, ponowny wybór najpewniejszej ramki i mediany z 21 przypadków: siedem późnych checkpointów razy trzy obrazy. Mediany kolumn obliczane są osobno i nie muszą sumować się między kolumnami.

| Grupa 32 neuronów | Połowa A RMS px | Połowa B RMS px | Całość RMS px | Zmiana IoU |
| --- | --- | --- | --- | --- |
| Kompensujące | 0.1678 | 0.1549 | 0.0304 | 0.0001 |
| Kontrola kompensacji | 0.1812 | 0.1410 | 0.2609 | -0.0001 |
| Losowe | 0.0645 | 0.0594 | 0.0736 | -0.0003 |
| Wzmacniające | 0.1644 | 0.1413 | 0.3201 | -0.0007 |
| Największe wkłady | 0.2657 | 0.3038 | 0.5122 | -0.0011 |
| Największa czułość | 0.3288 | 0.3165 | 0.5549 | 0.0003 |

Zatem kompensacja może maskować wpływ również całej grupy. Jednak dobór według samej czułości lub normy wkładu powoduje większe przesunięcia. **Nie wykazano przewagi miary współdziałania jako rankingu „najważniejszych neuronów” ani sposobu poprawy detektora.** Zmiany IoU mają oba znaki. Wzmacnianie zmiany współrzędnych nie musi oznaczać poprawnego lokalizowania narzędzia.

![Grupy i detekcja](grupy_a_ramka.png)

## Wyjątek, który warto badać: przełączenie zwycięzcy

Obraz 3, krok 55000, grupa 32 neuronów wzmacniających. Ten przykład został wybrany **po analizie jako największe późne przesunięcie grupowe**, więc nie jest reprezentatywną medianą.

| Interwencja | Token wybierający ramkę | IoU z GT | Zmiana ramki RMS |
| --- | --- | --- | --- |
| Bez wyłączenia | 369 | 0,75997 | 0 px |
| Wyłączenie 16 neuronów A | 369 | 0,76435 | 0,166 px |
| Wyłączenie pozostałych 16 neuronów B | 369 | 0,76311 | 0,150 px |
| Wyłączenie wszystkich 32 | 370 | 0,70610 | **7,090 px** |

Po wyłączeniu obu połówek logit tokenu 369 spada z 0,17476 do 0,05930, a logit sąsiedniego tokenu 370 rośnie z −0,01310 do 0,07192. Zmiana rankingu wybiera inną ramkę. Obydwie pojedyncze interwencje pozostawiają token 369 na pierwszym miejscu.

Nie jest to dowód ogromnej interakcji wewnątrz MLP. Skok pojawia się w dyskretnym wyborze maksimum. Duże `I` obliczone na ostatecznej detekcji miesza krzywiznę głowy z przełączeniem zwycięzcy. Trzeba je mierzyć osobno. Niezależny pełny forward, oficjalny dekoder i ranking z głowy float64 potwierdzają przejście 369 → 370; margines po przełączeniu 0,01262 jest dużo większy niż błąd numeryczny około 0,0000015.

![Przełączenie ramki](przelaczenie_ramki.png)

## Co z uczeniem i stałymi zespołami neuronów

Na kroku 33000 zamrożono identyfikatory par, używając wyłącznie pierwszych dwóch obrazów. Kryterium uwzględniało jednocześnie 16 tokenów, zamiast osobnego doboru w każdym tokenie. Już na tych obrazach mediana lokalnego q par kompensujących wynosiła około 0,744–0,747, a więc dużo więcej niż po indywidualnym doborze lokalnym.

Na trzecim obrazie i późniejszych krokach 35000–65640 te same pary dają q **0.887**, wobec **0.898** w kontroli. Nie przenosi się silna lokalna kompensacja. Nie należy opisywać tego wyłącznie jako „rozpadu wyuczonego zespołu”: selekcja wspólna już od początku nie osiągała tak silnego lokalnego efektu. Możliwe źródła różnic to aktywacje, nierówne siły partnerów, zmiana kierunków wag i zmiana Jacobianu głowy; tutaj nie rozdzielono przyczyn ich zmian podczas treningu.

Kontrola startu tego samego biegu:

| Krok | q dobranych par kompensujących | Zmiana ramki RMS px |
| --- | --- | --- |
| 0 | 0.1187 | 0.0358 |
| 1000 | 0.1146 | 0.0111 |
| 33000 | 0.0903 | 0.0088 |
| 65640 | 0.0857 | 0.0101 |

**Silna kompensacja występuje już w checkpoincie 0.** W końcowym stanie można dobrać nieco silniej kompensujące pary, ale jedna inicjalizacja i trzy obrazy nie wystarczają do przypisania tej różnicy procesowi uczenia. Selekcja ekstremalnych kierunków w 4D sama daje silny efekt. Obserwacja kompensacji nie może więc stanowić sama w sobie głównej tezy publikacji.

![Stałe pary w czasie](transfer_w_czasie.png)

## Co to daje mikroskopowi

Wynik uzasadnia mapę **funkcjonalnych wpływów neuronów**: dla wybranego tokenu widzimy, które działania zmieniają lewą, górną, prawą i dolną krawędź oraz logit środka, jakie partnerzy mają kierunki i czy ich wspólna ablacja maskuje osobne efekty. To bardziej informacyjne niż sama długość strzałki w rzucie 3D.

Drugim użytecznym pomiarem jest **margines konkurencji tokenów**: różnica logitów pierwszego i drugiego kandydata, wraz z przewidywanym wpływem grup na ten margines. Tutaj umożliwia wyjaśnienie, dlaczego małe ciągłe zmiany wewnątrz modelu mogą prowadzić do wyraźnej zmiany ramki końcowej.

Najbardziej konkretna hipoteza publikacyjna po tym eksperymencie: *czy kompensujące się wpływy na współrzędne mogą jednocześnie destabilizować ranking kandydatów detekcji, oraz czy śledzenie tych dwóch wielkości w checkpointach pozwala przewidywać błędy lokalizacji?* Wymaga to rozłącznych filmów, wielu inicjalizacji, wcześniejszego ustalenia selektora i porównania z normami, gradientami, marginesem top1–top2 oraz losowymi kontrolami. Obecny wynik jest studium mechanizmu i sprawdzeniem instrumentu pomiarowego; nie gotowym dowodem nowego prawa uczenia.

## Kontrola jakości i odtwarzanie

- Maksymalny błąd rekonstrukcji residual: 3.81e-06; różnica zapisanych stanów i ponownego pełnego forward: 1.72e-05.
- Szybkie globalne ablacje sprawdzono hakiem aktywacji w pełnym forward: błąd dla par poniżej 0,000001, dla grup końcowych 1.91e-06 w surowym wyjściu.
- Centralna różnica float64 potwierdziła JVP, błąd względny 7.73e-11. Lokalne efekty i Jacobiany policzono w float64, globalny detektor w float32. Stany i wkłady źródłowe pochodzą z modelu float32.
- 2535 z 127296 lokalnych rekordów ma zbyt mały mianownik dla wiarygodnego ilorazu. Pozostają w zbiorze z wynikiem ramki i IoU; q i względne reszty są puste. Dotyczy to wyłącznie losowych albo zamrożonych rodzin, a nie adaptacyjnych par kierunkowych.
- 99.847 procent kontroli par mieści się w tolerancji 20 procent obu norm. Najgorsze dopasowanie ma 59.8 procent odchylenia. Po ograniczeniu późnej analizy do kontroli spełniających tolerancję: kompensacja q=0.0840 wobec 0.6999; wzmocnienie q=0.9994 wobec 0.7325. Główny wniosek pozostaje ten sam.
- W grupach dopasowanie jest słabsze: 5,56 procent grup kontrolnych ma co najmniej jeden neuron poza tolerancją 20 procent, maksimum 49,1 procent. To ogranicza precyzję porównań grup; wyniki połówki/całość pozostają interwencjami w tej samej grupie.
- Brak imputacji IoU, brak filtrowania źle działających checkpointów, brak nominalnych p-wartości liczonych po skorelowanych parach. Zakres od 55000 to opis późnych stanów; wszystkie checkpointy są w surowych CSV.

Skrypty w `experiments/wspoldzialanie_neuronow_20261015/`: uruchomić kolejno `pomiar.py`, `analiza.py`, `grupy.py`, `kontrola_startu.py`, `kontrola_przelaczenia.py`, `raport.py`, `weryfikacja.py`. Python lokalny z PyTorch, timm, numpy, pandas, scipy i matplotlib. Wyniki trafiają wyłącznie do tego katalogu outputs. Protokół i dodatki jawnie rozdzielają plan początkowy, poprawę precyzji oraz analizy dobrane po wyniku.

Surowe dane: `interwencje_lokalne.csv.gz`, `interwencje_globalne.csv.gz`, `interwencje_grupowe.csv.gz`; identyfikatory: `zamrozone_pary.json`, `grupy_neuronow.json`; kontrole: `kontrola_startu.csv`, `przelaczenie_weryfikacja.json`. `MANIFEST.json` zawiera hashe checkpointów głównego biegu, obrazów i kodu pomiaru. Końcowy manifest obejmuje pozostałe skrypty i wyniki. Źródłem liczb są własne obliczenia na tych plikach, nie cytaty z literatury.
