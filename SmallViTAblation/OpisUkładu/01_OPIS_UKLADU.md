# Układ cząsteczek w latencie małego Vision Transformera

Stan dokumentacji: 11 października 2026. Opracowanie kodu, zachowanych raportów, metadanych i danych lokalnych. Źródła są wskazane w tekście oraz w katalogu eksperymentów. Raporty historyczne opisują stan wiedzy w chwili ich powstania; poniżej uwzględniono korekty z audytów 10–11 października. Nie wykonano ponownie treningów. Pakiet końcowy zawiera istniejące pliki i odnośniki; rozpoczęte odtwarzanie aktywacji zatrzymano na polecenie użytkownika.

## 1. Czym dokładnie jest ten układ

Układ stanowią reprezentacje przestrzennych tokenów stałych obrazów obserwowane podczas zmiany modelu. Cząsteczka jest wektorem reprezentacji jednego tokenu jednego obrazu. Jej tożsamość w pomiarze wyznacza para: identyfikator obrazu i numer tokenu. Czas oznacza krok aktualizacji wag, a nie numer klatki filmu. Obraz sondy pozostaje ten sam. Zmienia się funkcja, która przekształca obraz w reprezentację.

Przy wejściu 480 × 320 pikseli model daje siatkę 30 × 20, czyli 600 cząsteczek na obraz. Token w wierszu y i kolumnie x ma numer `30*y+x`. Dla 1500 obrazów jest 900 000 cząsteczek; dla 35 000 byłoby 21 000 000. Nie jest więc prawdą, że każde z 1500 zdjęć zamieniano w jedną cząsteczkę. W części wcześniejszych analiz uśredniano tokeny narzędzia lub tła do jednego wektora na grupę i obraz. To statystyka populacji, inny obiekt niż pojedyncza cząsteczka.

Mierzony wektor ma 768 współrzędnych. Jest odczytem po GELU warstwy ukrytej MLP ostatniego bloku. Główna szerokość backbone wynosi 192; MLP rozszerza ją do 768 i następnie wraca do 192. Liczba 768 nie oznacza szerokości wszystkich reprezentacji w sieci ani liczby tokenów.

Globalny układ to wspólna przestrzeń współrzędnych wszystkich cząsteczek. Jego makrowektory są osiami e_j, j=0…767. Dla cząsteczki i współrzędna z_ij określa składową z_ij e_j, nazywaną mikrowektorem. Plik przechowuje liczby z_ij, nie osobne wielowymiarowe wektory wewnątrz tokenu. W zapisie matematycznym:

`z_i(t) = Σ_j z_ij(t) e_j`.

Porównanie tych samych indeksów kanałów w checkpointach jednego biegu daje stały układ implementacyjny. Nie oznacza to, że semantyka kanału jest niezmienna podczas uczenia. Kanał może zmieniać funkcję. Porównanie dwóch niezależnych treningów po numerze kanału nie zapewnia zgodności ich baz semantycznych.

## 2. Trzy historie, których nie wolno sklejać

| Historia | Dane treningu | Czas | Zachowany pomiar |
|---|---|---|---|
| K1000, gęsta historia z 1 października | baza 1500 klatek | 1000 aktualizacji | pierwotnie 1001 pełnych stanów 12 sond; checkpointy co 10 kroków |
| 35k, bieg `pelny` | 35 000 klatek TRAIN, 117 filmów | 60 epok, 65 640 aktualizacji | 806 checkpointów; zachowane pełne odczyty 768D ośmiu sond dla 75 wybranych stanów |
| 35k, osobny bieg `gesty` | ta sama pula i receptura | 21 epok, 22 974 aktualizacje | 100 sond w PCA32, 8 sond w pełnych 768D, paczki i telemetria |

„Model trenowany do 60 checkpointów” odpowiada w dokumentacji treningowi **do 60 epok**. Zapisano 60 końców epok plus stan początkowy i dodatkowe stany pośrednie: łącznie 806 checkpointów. K1000 nie jest pierwszym odcinkiem biegu 60 epok. Bieg `gesty` również nie jest jego gęstym początkiem. Ten sam seed i przepis nie gwarantują bitowej zgodności przebiegów na GPU.

Źródła: `app/historie_checkpointow.py`, `assets/MalyViT_35000_od_zera_s47/README.md`, `experiments/maly_vit_od_zera_35000_21epok_20261009/STATUS.md`, `experiments/maly_vit_ruch_czastek_gesta_historia/WYNIKI.md`.

## 3. Dane wejściowe i jednostka obserwacji

Bieg 60 epok korzysta z Cataract-LMM Object Tracking. Manifest 35 000 klatek zapisuje film, numer klatki, archiwum ZIP, nazwę elementu w archiwum, wymiary, ramki obiektów i SHA-256 bajtów obrazu. Kod czyta oryginalne obrazy z ZIP, sprawdza hash, dekoduje RGB i rozciąga obraz do 480 × 320 metodą INTER_AREA. Współrzędne ramek skaluje osobno w poziomie i pionie. Dla kwadratowych lub innych proporcji wejścia jest to deformacja do ustalonego prostokąta, nie zachowanie proporcji przez padding.

Wektor latentny jest zależny od całego obrazu. Początkowa konwolucyjna siatka wiąże token z obszarem obrazu, lecz po attention token otrzymuje informację z innych pozycji. Nie można interpretować go jako izolowanego fragmentu analizowanego bez kontekstu. Zwykły przestrzenny numer tokenu nadal jest stały i pozwala śledzić jego reprezentację.

Walidacja treningu obejmuje 3000 klatek, z usuwaniem identyfikatorów przecinających się z treningiem. Sam brak wspólnych klatek nie jest równoważny pełnej niezależności filmowej — jednostkę podziału należy sprawdzać w konkretnym manifeście. Oddzielny duży pomiar reprezentacji używa 3000 zdjęć TRAIN i 1000 VALID, razem 4000. Liczba obserwowanych sond, liczba danych treningowych i liczba danych walidacji to trzy różne wielkości.

Osiem sond pełnego pomiaru kontrolnego to wiersze puli: 144, 448, 922, 2741, 2786, 3200, 3438, 3698. Pochodzą z siedmiu filmów, należą do TRAIN i według audytu mają etykietę cystotome. Są pierwszą ósemką uporządkowanej listy 100 sond. Nie stanowią reprezentatywnej walidacji wszystkich narzędzi. Wnioski o ich ruchu odnoszą się do tej próbki.

Źródła: `kod/trening35000.py`, `wejscia/MANIFEST_35000.jsonl`, `wyniki/gesty/gesty/SONDY.json`, `outputs/audyt_szkla_768d_20261010/RAPORT.md`.

## 4. Architektura sieci, krok po kroku

Wejście ma format B × 3 × 320 × 480 i wartości od 0 do 1. Forward odejmuje średnie `[0.485,0.456,0.406]` i dzieli przez odchylenia `[0.229,0.224,0.225]`.

Łodyga wykonuje cztery konwolucje 3 × 3, stride 2, padding 1. Kanały: 3→24→48→96→192. Pierwsze trzy konwolucje nie mają biasu i są uzupełnione BatchNorm2d oraz GELU. Czwarta jest samą konwolucją. Łączny krok przestrzenny wynosi 16 pikseli. Siatka cech ma rozmiar 20 × 30.

Siatka jest spłaszczana do 600 tokenów po 192 wartości. Dodawane są stałe pozycje 2D sin-cos, wyznaczone z numerów wiersza i kolumny. Nie są uczonymi embeddingami pozycyjnymi. Dalej przechodzi sześć bloków `timm.models.vision_transformer.Block`: trzy głowy attention, qkv_bias=True, MLP ratio 4. Każdy MLP ma rozszerzenie 192→768→192. Mierzona cząsteczka pochodzi z GELU szóstego bloku, przed jego projekcją `fc2`.

Po blokach występuje końcowy LayerNorm i głowa Linear(192,192), GELU, Linear(192,5). Głowa przewiduje osobno dla każdego tokenu logit środka narzędzia oraz cztery logarytmiczne odległości do boków ramki. Ostatni bias mapy środków zaczyna od −4.6, co odpowiada małej początkowej pewności; bias ramek zaczyna od log(3). Nie ma tokenu CLS ani klasyfikacji wyłącznie przez średnią wszystkich tokenów w tym detektorze.

Hook rejestrowany na `bloki[-1].mlp.act` pobiera tensor B × 600 × 768. Sam odczyt nie zmienia aktywacji. Interwencja edytora, która podmienia tę aktywację, jest odrębną operacją.

Źródło rozstrzygające: `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`.

## 5. Cel uczenia i odczyt predykcji

Na siatce tworzy się mapę gaussowskich celów wokół środka każdej ramki. Najbliższa komórka środka dostaje wartość dokładnie 1. Szerokość Gaussa zależy od rozmiaru ramki, z minimum 0.7 komórki. Cele regresji l,t,r,b są w jednostkach kroku 16 pikseli. Dodatnie komórki regresji leżą wewnątrz ramki i mają wartość Gaussa większą od 0.3; wymuszana jest także komórka środka. Odległości celu są ograniczane od dołu przez 0.05.

Strata to `L = L_heat + 2*L_box`. L_heat jest ogniskową stratą mapy środków w stylu CenterNet: dodatnie środki mają czynnik `(1-p)^2`, ujemne miejsca `p^2*(1-heat)^4`. Normalizacja jest przez liczbę dodatnich centrów, co najmniej jeden. L_box jest średnią `1-GIoU` z wagami mapy dodatnich komórek. Sieć zwraca log-odległości; do ramek stosuje się exp.

Przy dekodowaniu sigmoid logitu daje pewność. Lokalny max-pooling 3 × 3 wybiera szczyty mapy. Wybieranych jest maksymalnie 10 ramek na obraz. mAP@0.5 w tym kodzie dotyczy jednej klasy, dopasowania predykcji do niewykorzystanych GT i interpolacji precyzji po 101 progach recall. Nie jest mAP uśrednionym po progach IoU 0.5…0.95. Pewność konkretnej predykcji nie jest AP ani IoU.

Duża pewność może towarzyszyć złej ramce. Dlatego eksperymenty korekty używały osobno confidence i IoU, czasem kryterium obu wartości co najmniej 0.90. Takie kryterium wykorzystuje ramkę referencyjną i nie jest metryką działania na nowym obrazie bez GT.

## 6. Jak uczono model 60 epok

Model jest inicjalizowany od zera z seedem 47. Numpy używa `default_rng(47)`. Paczka ma 32 przykłady. Kolejne permutacje zbioru dostarczają indeksy obrazów. Jedna umowna epoka trwa `ceil(35000/32)=1094` aktualizacje, czyli 35 008 pobrań przykładów. Nie należy traktować tego jako 35 008 różnych zdjęć.

Augmentacje to poziome odbicie z prawdopodobieństwem 0.5 i zmiany jasności/kontrastu współczynnikami z zakresu 0.8–1.2. Po odbiciu wybierane są zgodnie przekształcone cele. Trening używa autocast bfloat16 na CUDA. Strata w istotnych obliczeniach przechodzi na float32. Przed krokiem optymalizatora gradient jest obcinany do normy 1.

AdamW ma learning rate 0.0005, beta1=0.9, beta2=0.999. Parametry o co najmniej dwóch wymiarach mają weight decay 0.05; parametry jednowymiarowe i skalarne decay 0. Learning rate rośnie liniowo przez pierwszych 300 kroków: `lr=0.0005*min(1,(krok+1)/300)`. Potem pozostaje stały. Nie było harmonogramu cosinusowego ani wygaszania na końcu 60 epok.

Pierwszy odcinek, zadanie Eden 1826056, zakończył 21 epok. Zadanie 1826076 wznowiło go od `stan_e21.pt` do epoki 60, odtwarzając wagi, AdamW i generatory. Kolejka niewykorzystanej permutacji nie została zachowana i po wznowieniu zaczęła się od nowa. Punkt 22 974 jest więc udokumentowaną granicą organizacyjną przebiegu. Analizy starzenia nie powinny używać okien przecinających ją bez odpowiedniej kontroli.

Pliki `kNNNNN.pt` zapisują sieć, krok i informację o epoce. Nie każdy jest pełnym stanem optymalizatora. `stan_eXX.pt` służył wznowieniu i obejmował także AdamW i RNG. Istnienie 806 plików wag nie oznacza istnienia 806 stanów momentum.

Zapis wag: 0–1000 co 10 kroków, potem co 100, dodatkowo końce epok. Końce epok to 1094*e; metadane `epoka` nie zawsze są całkowitym numerem epoki, bo bywają liczone przez `krok*32/35000`. Dla identyfikacji stanu rozstrzygający jest krok.

## 7. Skąd bierze się ruch i co oznacza „siła”

Reprezentacje nie są niezależnymi parametrami optymalizowanymi przez AdamW w zwykłym treningu. Optymalizator aktualizuje wspólne wagi θ. Wszystkie sondy są potem ponownie przepuszczane przez nową sieć. Dla stałego obrazu i tokenu:

`z_i(t)=f_i(x; θ_t,b_t)`,

gdzie b_t to m.in. bufory BatchNorm. Dokładny ruch to różnica dwóch forwardów. Lokalne przybliżenie ma postać `Δz_i ≈ J_i Δθ + B_i Δb`. J_i jest wrażliwością konkretnego tokenu na wspólne parametry. Dlatego ta sama zmiana wag może przesuwać różne tokeny w różnych kierunkach i z różną amplitudą. Dwa podobne tokeny mogą odpowiadać podobnie, również gdy pochodzą z różnych obrazów.

Gradient paczki wpływa na pierwszy i drugi moment AdamW. Pierwszy moment gromadzi pamięć wcześniejszych gradientów. Drugi moment zmienia skalę kroku osobno dla parametrów. Aktualizacja schematycznie zawiera adaptacyjnie przeskalowany moment oraz osobny decay wag. To nie jest po prostu siła równa ujemnemu gradientowi położenia cząsteczki. Sam gradient dL/dz mierzy nacisk straty na aktywację, ale obserwowane Δz przechodzi przez aktualizację wszystkich wag.

BatchNorm występuje w łodydze. W treningu używa statystyk paczki i zmienia running_mean/running_var. W pomiarze eval używa zapisanych buforów. Zmiana reprezentacji eval może więc powstać zarówno przez aktualizację wag, jak i przez zmianę buforów. LayerNorm w blokach ma inne zasady: normalizuje reprezentację w bieżącym forwardzie, bez biegnących średnich paczek.

W eksperymentach „siły” oznaczały odpowiedzi na kontrolowane składniki aktualizacji lub skuteczne składniki równania ruchu. Siła nowej paczki, pamięć AdamW, decay i wpływ buforów mogą być mierzone przez forward przy częściowych zmianach stanu. Jest to operacyjny rozkład odpowiedzi sieci. Nie są to siły mechaniczne w niutonach, a latent nie jest położeniem w metrach.

Nie należy utożsamiać ruchu wspólnego z oddziaływaniem cząsteczek przez odległość w latencie. Attention rzeczywiście miesza tokeny wewnątrz jednego obrazu, lecz wspólny ruch między obrazami może wynikać ze współdzielonych parametrów. Dane nie ustanawiają uniwersalnego potencjału parowego U(|z_i-z_j|).

Źródła: `app/przyczyny_ruchu.py`, `app/rownanie_krokowe.py`, `experiments/maly_vit_liniowy_ruch_czastek/WYNIKI.md`, `outputs/poszlaki_dynamiki_20261011/RAPORT.md`.

## 8. Jak zapisano trajektorie

### 8.1. Pełne aktywacje

Tablica pełnych aktywacji zachowuje każdą z 768 współrzędnych każdego tokenu i każdego wybranego obrazu. Dla biegu 60 epok istnieją 75 plików kontrolnych, każdy z naturalną tablicą `fp32` o kształcie `(8,600,768)`. Osiem obrazów oznacza 4800 cząsteczek. Kolejne pliki wyznaczają stany tych samych tokenów; trzeba sortować je po liczbowym numerze kroku. W tych samych plikach są osobno odczyty bf16 i kontrola zamrożonego BN.

Kompletna robocza tablica 806 stanów ośmiu sond została usunięta przez historyczny skrypt po zbudowaniu macierzy Grama. Nie ma jej w końcowym pakiecie. Pliki kontrolne pozwalają liczyć na pełnych współrzędnych wybranych stanów, w tym końców wszystkich 60 epok. Brakujących kroków nie należy przedstawiać jako pomierzonych ani wypełniać interpolacją bez zmiany definicji analizy.

### 8.2. Gęsty bieg 21 epok

`obserwator_gesty.py` odczytywał sondy w eval. Zapis PCA32 i norm jest co krok do 2188, potem co 10. Pełne 768D dla 8 sond zapisano co krok do 1094, potem co 50. Pełny zapis ma 1533 stany, do 22950; trening kończy się w 22974. Brak końcowego pełnego stanu jest konsekwencją harmonogramu, nie powodem do dopisywania kopii ostatniego wektora.

Pliki `pelne768_NNN.npz` zawierają `kroki` i `latent` o układzie czas × obraz × token × kanał, float16. Pliki `pca32_NNN.npz` zawierają projekcję, pełnowymiarowe normy i kroki. `PACZKI.npz` zapisuje indeksy 32 przykładów, straty przed/po w eval, heat, box, normę aktualizacji wag, normę obciętego gradientu, lr i czas. Skład paczki bez zapisanej realizacji augmentacji i kompletnego stanu RNG/AdamW nie wystarcza do bitowego odtworzenia każdego kroku treningu.

### 8.3. Projekcje i pliki aplikacji

PCA32 zachowuje 32 kombinacje współrzędnych; PCA3 trzy. Norma 768D dołączona do rzutu nie umożliwia rekonstrukcji utraconych współrzędnych. Stała baza PCA z początku uczenia nie musi wyjaśniać równie dużej części ruchu później. 98.6% wariancji położeń w kroku 0 nie oznacza 98.6% wariancji wszystkich późniejszych przyrostów.

Plik q16 aplikacji przechowuje pozycje ekranowe XYZ i dodatkowe wartości według formatu producenta. W naturalnym pomiarze cztery uint16 na cząsteczkę kodują XYZ i normę; dekodowanie potrzebuje parametrów quant w manifeście. Nie jest to tensor 600×768. Istnieją też rzuty losowe RP128 i trajektorie względem klatki, wymagające osobnej identyfikacji.

Pliki `long_image*.npz` audytu to m.in. macierze Grama w czasie, centroidy i sąsiedztwa. Powstały z pełnych 768D, ale nie przechowują pełnych położeń wszystkich tokenów. Przydatne do odtworzenia zbiorczych MSD, nie do dowolnej analizy poszczególnych współrzędnych. Pliki `precision_controls/k*.npz` mają za to rzeczywiste aktywacje `fp32`, `frozen_bn`, `bf16` dla wybranych checkpointów.

## 9. Miary ruchu i układy odniesienia

Ruch surowy: `Δz_i=z_i(t+τ)-z_i(t)`. Układ względem obrazu odejmuje średnie przemieszczenie 600 tokenów danego zdjęcia. Układ względem lokalnej klatki odejmuje średni ruch wybranego sąsiedztwa. Graf sąsiadów może być ustalony w kroku 300, na początku okna albo inaczej — definicja musi być dołączona do wyniku. Sąsiedztwo w PCA32 nie jest automatycznie sąsiedztwem w pełnych 768D.

MSD jest średnią `||z_i(t+τ)-z_i(t)||²` po określonej populacji i początkach czasu. W audycie 768D nie dzielono przez 768. Wykładnik α opisuje nachylenie log(MSD) względem log(τ) w wybranym zakresie. α<1 sugeruje subdyfuzyjny wzrost w tym zakresie; nie ustanawia samodzielnie szkła. VACF opisuje korelację przyrostów; jej znak i wartość zależą od odstępu pomiarów i odjętego dryfu. Cosinus kolejnych przyrostów nie jest identyczny ze wszystkimi definicjami VACF.

Temperatura w aplikacji jest umowną miarą energii drgań po odjęciu pola kolektywnego w danym rzucie. Energia kinetyczna `1/2〈|Δx|²〉`, sprężyna OU, tempo powrotu κ, widmo i czas relaksacji są charakterystykami skutecznego modelu. Nie zmierzono temperatury termodynamicznej ani mas cząsteczek. DFT zakłada regularne odstępy; dla mieszanego harmonogramu checkpointów trzeba wybrać regularny fragment lub metodę obsługującą czas nieregularny.

„Klatka”, wskaźnik nazwany Lindemannem i `tau_1e` zależą od odległości sąsiadów i wybranych lagów. Spadek ilorazu ruch/odstęp nie dowodzi spadku bezwzględnego ruchu. Składowa sąsiedztwa i reszta nie są ortogonalne, więc sumowanie ich energii bez członu mieszanego nie daje rozłącznego podziału 100%.

## 10. Wyniki treningu i pierwszych analiz tokenów

Bieg 60 epok osiągnął mAP@0.5 około 0.449 po pierwszej epoce. W epokach 8–21 średnia była około 0.62. Epoka 21 dała około 0.6524, najlepszy koniec epoki 26 około 0.6533, a epoka 60 około 0.608. Strata treningowa nadal malała. Dłuższy trening przy stałym lr nie przyniósł trwałej poprawy walidacji. Pomysł wygaszania lr jest hipotezą następnego eksperymentu, nie już zmierzonym zyskiem.

W K1000 po około 100 krokach tokeny narzędzia ruszały się mocniej od odległego tła. AUC normy ruchu po odjęciu dryfu wyniosło 0.761 w gęstym pomiarze. Średni kierunek dał tylko 0.519 w całym biegu, choć we wczesnym fragmencie 20–100 był wyraźniejszy sygnał. Dla 32 kierunków wynik SVD 0.730 był porównywalny z losową bazą 0.733. Większa amplituda jest lepiej podparta niż stały kierunek specyficzny dla narzędzia.

Nie wolno z tego wywnioskować, że ruch jest tożsamością narzędzia. Krawędzie, refleksy i inne cechy obrazu mogą wpływać na ruchliwość. Pomiar obejmował jeden seed, 12 obrazów i 4 odłożone filmy do sprawdzenia kierunków. Tokeny i kolejne kroki nie są niezależnymi replikami.

Źródło: `experiments/maly_vit_ruch_czastek_gesta_historia/WYNIKI.md`.

## 11. Równania Langevina, pamięć i Koopman

Badano opis cząsteczki przez położenie i prędkość: `v(t+1)=βv(t)-κ(z(t)-m(t))+η(t+1)`, gdzie m jest wolnym środkiem, np. średnią ostatnich 50 stanów. Dopasowanie oscylatora na 300–700 i test 700–1000 dla klatki 3 dało błąd względem persystencji 0.42 na krok i 0.81 na horyzoncie 50. Dla klatek 2,3,6 wynik jednokrokowy był 0.39–0.42. Wspólny Koopman PCA32 z lr miał błąd 1.44–1.74 względem persystencji na 50 kroków.

Niski błąd rekonstrukcji położeń przez kilka modów nie zapewnia przewidywania przyrostów. Pięć modów czasowych wyjaśniało około 73–75% wariancji położeń, ale przyszły krok zależy również od nowej paczki. Koopman, Hankel-DMD, GLE z pamięcią i NNDMD są próbami domknięcia dynamiki z obserwowanej historii. Nie ustanowiono jednego operatora skutecznego dla dowolnego tokenu, obrazu i treningu.

Interwencyjny eksperyment sił na trzech klatkach wyjaśnił sumą odpowiedzi składników aktualizacji 98.9–99.7% ruchu tokenów. Model z wejściem bieżącej paczki i wspólnymi współczynnikami opisywał około 96–98% ruchu. Udziały projekcyjne raportowano jako pęd 64–68%, nowa paczka 31–35%, BN około 1%, decay blisko zera. Te wielkości dotyczą konkretnych warunków i definicji rozkładu. Przy znanej sile paczki jest to wyjaśnienie warunkowe; bez tej siły prognoza jednokrokowa miała około 54–60% R². Nie jest to 98% prognozowania nieznanej przyszłości z samej historii.

Źródła: `experiments/maly_vit_liniowy_ruch_czastek/WYNIKI.md`, `app/rownanie_czastki.py`. Pozostałe rodziny prognoz i ich oryginalne raporty są w katalogu eksperymentów.

## 12. Audyty szkła i korekty interpretacji

W pełnym 768D gęsty bieg wykazał zależność od skali: w oknie 300–1094 wykładnik surowego MSD dla lagów 1–5 wynosił 1.530, dla 10–100 tylko 0.259. Dodatnia korelacja małych kroków współistnieje z ograniczonym wzrostem przemieszczeń na dalszych lagach. Nie ma sprzeczności z ujemnymi korelacjami przyrostów pobranych co 10 kroków.

W biegu 60 epok wykładnik surowego MSD dla lagów 100–1000 wzrósł z 0.297 w oknie 1000–6000 do 0.368 w oknie 50000–65000. MSD względem klatki po 1000 kroków wynosiło około 2.681 dla wieku 1000, 7.480 dla 30000 i 6.631 dla 60000. Nie pokazuje monotonii „cała populacja zamarza”. Odjęcie dryfu i zastosowanie prostego OU z pamięcią plus dyfuzją wyjaśnia część odcisków przypisywanych szkłu.

Audyt z 11 października porównał wczesne i późne okna tych samych biegów w RP128. W detekcji energia pojedynczego kroku 768D spadła 0.29674→0.15720, podczas gdy lokalne MSD dla lagu 105 wzrosło 1.51584→2.76329. Skala sąsiedztwa wzrosła 2.31081→3.25843. Wskaźnik względny spadł, mimo wzrostu bezwzględnego skumulowanego ruchu. Klasyfikacja również miała spadek energii kroku, lecz wzrost lokalnego MSD. To rozdzielenie skal, nie dowód jednej temperatury układu.

Pojedyncze kroki z resetu checkpointu wskazują mechanizm wspólnych wag: mapy ruchliwości miały korelację Spearmana 0.8493 dla odpowiedzi resztowej w detekcji lr=0.0005 na końcu epoki 2. Sama perturbacja BN miała energię 0.86–2.72% odpowiedzi na paczki w trzech checkpointach. Nie są to rozłączne udziały przyczynowe. W długich przedziałach rola BN może być większa.

Kontrola `fc2` rozdzieliła ruch na jądro i przestrzeń wierszy projekcji 768→192. Jądro ma 576 wymiarów; dla izotropowego wektora punktem odniesienia jest 75% energii. Zmierzono około 59–67% w jądrze dla sprawdzonych par, czyli istotna część ruchu przechodzi przez fc2. To nie dowodzi automatycznie zmiany decyzji, bo dalej działają residual, normalizacja i głowa. Nie potwierdza też tezy, że reprezentacja porusza się wyłącznie w nieistotnych kierunkach.

Warianty CNN, SGD z momentum, paczek, lr, warstw i Imagenette dostarczają kontroli porównawczych. Nie izolują wszystkich czynników: zmieniają się architektura, nadzór, dane i tokenizacja. Nowszy audyt ma pierwszeństwo przed mocniejszymi werdyktami historycznego README porównania szkła. Poprawny wniosek: skorelowana, przestrzennie uporządkowana przebudowa aktywacji ze wspólnych zmian modelu; częściowe podobieństwa kinematyczne do szkła bez dowodu przejścia szklistego.

Źródła: `outputs/audyt_szkla_768d_20261010/RAPORT.md`, `WYNIKI_LICZBOWE.md`, `outputs/poszlaki_dynamiki_20261011/RAPORT.md`.

## 13. Interwencje w latencie i generowanie obrazów

Ta gałąź zaczyna się od zamrożonego checkpointu, a nie od naturalnego ruchu podczas uczenia. Modyfikuje aktywacje z po GELU, utrzymując wagi i bufory. Cel może podnieść confidence i dopasowanie ramki na znanym obrazie. Tak powstaje sztuczny latent, który nie musi być osiągalny przez zwykły forward żadnego realistycznego obrazu.

Oracle na 1500 obrazach osiągnął confidence i IoU co najmniej 0.90 dla 1283/1500 przypadków, 85.53%. Była to osobna optymalizacja każdego obrazu z wiedzą o celu. Średnia korekta i korekta warunkowa nie przeniosły sukcesów: oba progi dla 0 obrazów i pogorszenie mAP50:95. Późniejsza wspólna delta 600×768 uczona joint poprawiła mAP50:95 na tych samych 1500 obrazach z 0.245496 do 0.298835. IoU pogorszyło się dla 451 obrazów, oba wskaźniki dla 17, false positives wzrosły 493→675. Nie jest to uniwersalna poprawa ani walidacja niezależnej generalizacji.

SD jest osobnym generatorem dostrojonym do pełnego warunku 600×768 modelu K1000. Adapter pracuje per token 768→768→768 i ma uczone pozycje wierszy/kolumn. Domyślnie uczone są adapter i cross-attention attn2 UNet; reszta UNet i VAE pozostają zamrożone. To dostrojenie istniejącego SD, nie nauka całej dyfuzji od zera. Główna strata jest epsilon-MSE DDPM; pomocnicza gałąź porównuje latent i detekcję z odkodowanego x0.

Trening SD zakończył 20 000 aktualizacji. Best checkpoint jest z kroku 8500, val_film_mean_mse=0.1641106312, na 256 próbkach z 8 filmów. MSE odszumiania nie jest bezpośrednią miarą jakości rekonstrukcji ani poprawy detektora. Lokalny test naturalnego warunku uzyskał zgodność wartości J128 około 33.4%; sztuczny cel około 0.56%. Generacja nie odtworzyła narzędzia i położenia referencji; pierwotny sztuczny latent dawał niemal jednolity obraz. Nie udowodniono, że można skutecznie przełożyć dowolny zoptymalizowany wektor na obraz.

Edytor pozwala dodać wektor do wszystkich tokenów, jednego tokenu lub podmienić token; generuje DDIM od szumu albo img2img. Warunek ViT 600×768 jest inny niż przestrzenny latent VAE o czterech kanałach. W aplikacji VAE pracuje na obrazie 576×384, dając szum 4×48×72. Krok DDIM, aktualizacja ViT i przejście między klatkami filmu to trzy różne osie czasu. Latenty modelu 60 epok nie są automatycznie zgodne z generatorem uczonym na K1000.

Źródła: raporty `vit_k1000_1500_pelne_latenty_90_20261008`, `vit_k1000_wspolna_korekta_patch_20261009`, `vit_sd_pelny_latent_filmy_eden_20261006`, oraz `app/silnik.py`.

## 14. Rola aplikacji

Folder app integruje edycję latentu, naturalny odczyt detektora, podgląd trajektorii, analizę uczenia, przyczyn ruchu, attention, modeli krokowych, projekcji i porównania szkła. Moduły historii wybierają istniejące checkpointy. `ruch_naturalny.py` wykonuje inferencję i zapisuje oś wizualizacji. `ruch_zbioru.py` definiuje wspólne formaty. `analiza_uczenia.py` liczy charakterystyki historii, `przyczyny_ruchu.py` interwencje grup parametrów, `rownanie_czastki.py` modele z historii, `silnik.py` generowanie SD.

Filtry obszarów oka korzystają z segmentacji i progów pokrycia patcha. Wyniki zależą od definicji maski: dawny eksperyment „maska 1 px” i późniejsze pokrycie ≥25% narzędzia nie wyznaczają tej samej populacji. Kolor na ekranie, wygładzenie EMA i uśredniony ruch nie zastępują danych źródłowych. Wygładzanie zmienia przedstawiony tor i korelacje; obliczenia należy wykonywać na stanach surowych, chyba że protokół świadomie definiuje filtr.

## 15. Zasady dalszego liczenia

1. Zawsze wybierać jeden bieg i odczytywać rzeczywiste kroki. Nie scalać `gesty` z `pelny` ani K1000 z 35k.
2. Dla pełnego 768D używać aktywacji, nie PCA3, PCA32, RP128 ani samych macierzy Grama.
3. Liczyć różnice w float32 lub float64; nie odejmować i nie sumować kwadratów w float16.
4. Określić populację, maskę, lag, okno, układ odniesienia i graf sąsiadów. Zachować jednostkę kroku treningu.
5. Porównywać te same bezwzględne lagi w regularnych fragmentach. Nie nazywać różnicy dwóch zapisów „jednym krokiem”, jeśli dzieli je 100 aktualizacji.
6. Rozdzielić dopasowanie przeszłego ruchu, rekonstrukcję przy znanym wejściu i prognozę nieznanej przyszłości.
7. Resamplować na poziomie filmów lub treningów odpowiednim dla pytania. 4800 tokenów ośmiu obrazów nie daje 4800 niezależnych doświadczeń.
8. Oddzielić miary absolutne od względnych i uwzględnić człony mieszane przy podziale energii.
9. Nie przenosić metryk uczenia na jakość generacji SD. Nowa generacja potrzebuje własnych etykiet do oceny trafności detekcji.
10. Zachować checksumy, wersję kodu i dokładny punkt odczytu. Eksport inferencyjny nie odtwarza brakującego stanu AdamW.

## 16. Gdzie szukać pełnego materiału

Dokument `02_DANE_I_TRAJEKTORIE.md` jest instrukcją dostępu do najbardziej surowych tablic, z rozróżnieniem danych zachowanych, odtworzonych i brakujących. `03_KATALOG_EKSPERYMENTOW.md` porządkuje rodziny eksperymentów i ich raporty. Kopie raportów w `zrodla/` zachowują pełną treść historycznych wyników, w tym liczby rodzin, których nie wolno sprowadzać do jednej zbiorczej metryki. `04_RAPORTY_SCALONE.md` łączy ich treść w jeden przeszukiwalny plik, z jednoznaczną ścieżką pochodzenia każdej części. Katalog ten nie oznacza, że ponownie przeliczono wszystkie eksperymenty; oddziela syntezę kluczowych wyników od źródłowych zapisów pozostałych rodzin.
