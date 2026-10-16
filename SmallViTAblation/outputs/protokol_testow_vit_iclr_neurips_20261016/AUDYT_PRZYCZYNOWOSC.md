# Audyt przyczynowości: współdziałanie neuronów, patching i wybór detekcji

Data: 16 października 2026. **Protokół proponowanych badań, nie wyniki nowych eksperymentów.** Nie uruchomiono treningów, pomiarów modeli ani zadań na Edenie; aplikacja pozostaje niezmieniona. Zasady próbkowania i seedów nadrzędnego [PLAN.md](../../experiments/plan_publikacji_neurips_20261016/PLAN.md) mają pierwszeństwo przed wcześniejszymi wariantami budżetu z [audytu interwencji](../plan_publikacji_neurips_20261016/audyt_interwencji.md).

## 1. Twierdzenie, które rzeczywiście warto sprawdzić

**Kandydat na wynik:** istnieją przenoszalne, zależne od treści obrazu zespoły wkładów MLP, które stabilizują lokalizację przy zmianach nieistotnych dla zadania; ich wpływ na współrzędne ramki i na ranking kandydatów można rozdzielić interwencjami. Metoda wskazująca te zespoły ma przewidywać rzeczywiste skutki na nowych filmach i treningach lepiej niż porównywalne kosztowo normy, attribution oraz margines detektora.

To hipoteza. Obecne małe q po doborze przeciwnych gradientów nie dowodzi wyuczonej kompensacji. W pilocie duże przełączenie dotyczyło grupy wzmacniającej. Słaby transfer zamrożonych par i prawie addytywny lokalny odczyt są argumentami za rygorystyczną kontrolą, nie za zwiększaniem liczby efektownych przykładów.

Są trzy różne możliwe wyniki:

1. **Kompensacja liniowa:** duże osobne zmiany znoszą się w sumie; może być funkcjonalnie istotna bez nieliniowej synergii.
2. **Nieaddytywność obliczeń:** efekt wspólny różni się od sumy osobnych efektów przy ustalonej wielkości odczytu.
3. **Dyskretne przełączenie dekodera:** niewielkie, nawet idealnie liniowe zmiany logitów zmieniają wybranego kandydata.

Artykuł musi wskazać, który wynik otrzymano. Causal claim dotyczy interwencji w zamrożonym modelu i określonym rozkładzie obrazów. Wniosek o przyczynie powstawania mechanizmu podczas uczenia wymaga dodatkowo prospektywnych kontynuacji treningu z pełnym stanem optymalizatora, jak w E3 głównego planu.

## 2. Wspólne definicje, które należy zamrozić przed pomiarem

### 2.1. Miejsce i kierunek interwencji

W badanym modelu residual ma 192 wymiary, a warstwa ukryta MLP ma 768 neuronów. Dla tokenu p i bloku l zapisujemy:

- a = W1 LN(x) + b1, z = GELU(a);
- u_c = z_c v_c, gdzie v_c jest kolumną W2;
- MLP(x) = suma_c u_c + b2; wyjście residual = x + MLP(x).

Podpisane z może być ujemne. Przeciwny v_c nie przesądza o przeciwnym u_c. Bias, residual i dalsza normalizacja muszą być zachowane.

Podstawowa bramka grupy G: dla c należącego do G zastąpić z_c przez (1−α)z_c; pozostałe współrzędne są niezmienione. α=0 oznacza kontrolę, α=1 wyzerowanie. Nie zmieniać wag, nie wykonywać optymalizacji po interwencji. Patching: z_G(base) zastąpić z_G(donor), z dokładnym logowaniem tokenów, bloku i sposobu dopasowania dawcy. Cała reszta forward jest przeliczana z aktualnego stanu. Dla wcześniejszego bloku obejmuje to wszystkie późniejsze bloki.

Neurony w jednym fc1→GELU→fc2 nie mają połączenia c→d. Zmiana z_c po GELU nie może w tym samym forward wywołać zmiany z_d. Adaptacja downstream może zachodzić w późniejszych blokach, a zmiana po checkpointach należy do procesu treningu. Tych dwóch zegarów nie wolno łączyć.

### 2.2. Wyjścia i prawdziwy dekoder

Zapisywać pełne mapy logitów heat oraz regresji ltrb. Dekoder stosuje sigmoid, lokalne maksimum w oknie 3×3, maskę p·1[p=maxpool(p)], topk(10), a odległości ramki powstają przez exp(ltrb). Każda interwencja wymaga jego ponownego uruchomienia. Zera po maskowaniu, remisy i kolejność topk należy odtworzyć z implementacji, bez wymyślania własnego stabilnego tie-breaku.

Raportować osobno:

- ciągłe zmiany czterech krawędzi b_w dla tokenu w wybranego **przed** interwencją;
- logity i sigmoid confidence dla ustalonych tokenów; sigmoid nie jest skalibrowanym prawdopodobieństwem poprawnego bbox;
- margines raw top1–top2 oraz margines w zbiorze dopuszczonych lokalnych maksimów;
- maskę lokalnych maksimów, skład i kolejność top10, top1 po dekodowaniu;
- przypisanie predykcji do GT według zamrożonej reguły, IoU, błąd krawędzi, liczbę FP/FN i pełną jakość detektora.

Różnica dwóch logitów opisuje ranking kandydatów tylko przy ustalonym zbiorze ich dopuszczalności. Przy zmianie maski należy przeliczyć cały dekoder. Zmiana tokenu w obrębie tego samego obiektu nie jest automatycznie pomyłką; rozdzielić zmianę reprezentanta, zmianę przypisanego obiektu i utratę poprawnej detekcji. Oracle max-IoU w top10 jest diagnostyką, nie podstawową jakością wyjścia.

### 2.3. Miary współdziałania

Dla rozłącznych A i B mierzyć kontrolę, A, B i A∪B przy tej samej α. Dla ustalonej ciągłej wielkości y:

Δy_G = y_G − y_0;  
I_y = Δy_AB − Δy_A − Δy_B;  
q_y = ||Δy_AB|| / (||Δy_A|| + ||Δy_B||).

q opisuje kompensację; I opisuje nieaddytywność. Przy liniowym odczycie i przeciwnych zmianach q może być bliskie zeru, a I dokładnie zerowe. Nie liczyć I jako podstawowego wskaźnika na samym IoU lub indeksie zwycięzcy: ich nieliniowość może pochodzić z miary/dekodera, nie z badanej warstwy. Pokazać I w residual, logitach, ltrb i krawędziach, a IoU osobno.

Dla par dodatkowo C_cd = 2 u_c·u_d i q_u = ||u_c+u_d||/(||u_c||+||u_d||). Agregować wpływ z wagą faktycznej siły, nie liczyć mikroskopijnych aktywacji jak dużych zdarzeń. Minimalny mianownik i progi aktywności wybrać na discovery; wyniki ciągłe i odsetek wyłączeń przez próg podać zawsze. Nie nazywać sumy C_cd energią fizyczną.

Dla ratowania odpowiedzi primary stanowią surowe zmiany IoU, krawędzi i logitów. Recovery ratio R=(y_patch−y_corrupt)/(y_clean−y_corrupt) tylko secondary, z wyłączeniem jawnie małych mianowników i bez obcinania do [0,1]. Nieskuteczna korupcja jest wynikiem do raportu, nie powodem ukrytego usuwania obserwacji.

### 2.4. Jednostka niezależna i transfer

Pomiary są sparowane po obrazie, perturbacji i seedzie. Niezależne źródła zmienności to **film/zabieg/pacjent oraz niezależny trening**; klatki, tokeny, pary, dawcy i checkpointy są zagnieżdżone. W każdej pozycji poniżej „jednostka” oznacza ten układ, chyba że chodzi o deterministyczny test instrumentu. Dla tych samych filmów w wielu seedach jest to układ skrzyżowany film×seed, nie nowa niezależna populacja filmów. Raportować wyniki per seed i film oraz niepewność klastrową odpowiednią do układu. Przy trzech seedach pokazać każdy, nie ukrywać małej replikacji za tysiącami tokenów.

Dwie oddzielne tezy:

- **Stałe zespoły:** identyfikatory wybrane wcześniej na innych filmach, potem zamrożone w obrębie tego samego seedu.
- **Adaptacyjny instrument:** zamrożony algorytm doboru korzysta z bieżącego czystego wejścia. Testuje użyteczność reguły, a nie trwałość konkretnych neuronów.

Nie przenosić numerów neuronów pomiędzy seedami. GT dostępne do oceny i analiz oracle nie może przeciekać do predyktora jakości ani selektora reklamowanego jako dostępny bez etykiet.

## 3. Testy C01–C18

Wszystkie poniższe procedury są propozycjami naszego protokołu dla detektora. Cytowane prace uzasadniają problemy metodologiczne, a nie dowodzą, że opisywany mechanizm występuje w tym ViT.

### C01. Instrument, rzeczywisty graf i ujemne kontrole

**Hipoteza:** pomiary odtwarzają funkcję modelu, a pozorna wzajemna korekta nie pochodzi z hooka, cache ani dekodera.

**Interwencja i kontrola:** α=0; patchowanie aktywacji z tego samego obrazu i checkpointu; permutacja indeksów ukrytych z odpowiadającą permutacją W1, b1 i kolumn W2; pełny forward z hookiem wobec skrótu head-only wyłącznie tam, gdzie skrót jest matematycznie równoważny. Zmienić jeden z_c po GELU i potwierdzić niezmienność pozostałych z_d w tym samym MLP. Testy syntetyczne znanego sumatora, AND/OR i progowego dekodera odróżniają addytywność od dyskretnego wyboru. Dla dekodera wymusić sąsiadujące piki, remisy i przypadki mniej niż 10 dodatnich pików.

**Metryka:** maksymalny błąd residual/logitów/krawędzi, zgodność maski i top10, I=0 w sumatorze. Tolerancje zależne od precyzji ustalić wobec referencji; blisko progu ponowić obliczenie w wyższej precyzji i raportować nieokreślone remisy.

**Jednostka:** przypadek deterministycznego testu; brak testu istotności statystycznej.

**Falsyfikacja i minimum:** jakikolwiek niewyjaśniony błąd kontroli zatrzymuje C02–C18. Minimum: E0 z PLAN.md, 4 obrazy×3 checkpointy×6 bloków oraz małe sztuczne grafy. Literatura: rozróżnienie struktury AND/OR i sposobów patchingu [R2](https://arxiv.org/html/2404.15255v1); reszta jest kontrolą własnej implementacji.

### C02. Czy organizacja jest wyuczona, czy wynika z selekcji?

**Hipoteza:** po uczeniu występuje nadmiar funkcjonalnej kompensacji lub selektywności względem inicjalizacji, poza tym, co narzuca dobór przeciwnych wektorów.

**Interwencja i kontrola:** ten sam algorytm selekcji, liczba ocenionych grup i budżet normy w checkpointach 0 i późniejszych; A/B/AB przy tych samych α. Dwa protokoły: dobór tylko z geometrii W2 oraz dobór przez J_b u_c. Pary gradientowe wybrane na danych discovery oceniać na rozłącznych filmach; osobno lokalna selekcja na bieżącym obrazie jako baseline. Kontrole losowe dopasować do aktywności, norm pojedynczych wkładów i indywidualnej czułości odczytu; nie dopasowywać do kąta będącego przedmiotem testu. Osobna kontrola par o podobnym kącie testuje już efekt ponad geometrię. Permutacja aktywacji partnera wewnątrz obrazu/regionu niszczy współzależność; raportować jej nienaturalność.

**Metryka:** różnica transferowego q, C_cd i efektów na odczyt między stanami; nadmiar względem nulli, z bezwzględną siłą usuwanego sygnału. Nie porównywać tylko IoU losowego i wyuczonego modelu.

**Jednostka:** film×seed; checkpoint to pomiar powtarzany.

**Falsyfikacja i minimum:** brak transferowego nadmiaru albo efekt całkowicie wyjaśniony selekcją odrzuca tezę „wyuczone zespoły kompensujące”. Minimum: inicjalizacja, 33000 i stan końcowy; s47 discovery, dalsze seedy wg PLAN. Label-shuffle training jest rozszerzeniem, nie już wykonanym nullem. Literatura: rozdzielenie odkrywania i potwierdzania [R2](https://arxiv.org/html/2404.15255v1); brak uniwersalnej interpretacji ablacji [R4](https://arxiv.org/html/2407.08734v1).

### C03. Jednoczesna kompensacja czy różne konteksty?

**Hipoteza:** przeciwne kierunki W2 tworzą dla tego samego tokenu rzeczywiście silne przeciwne wkłady, ze specyfiką obszaru narzędzia ponad siłę aktywacji.

**Interwencja i kontrola:** rozdzielić cztery stany znaków z_c,z_d oraz nieaktywność według ustalonego progu. Wykonać A/B/AB na podpróbie współaktywnych i rozdzielnie aktywnych przypadków. Regiony: wnętrze bbox, granica, dalekie tło; do twierdzenia o narzędziu użyć masek. Dopasować rozmiar patcha/udział obiektu, położenie, normy wkładów; permutacje partnerów wewnątrz regionu i obrazu. Zachować oryginalne signed GELU, bez abs aktywacji przed wyznaczeniem wektora.

**Metryka:** ważony udział u_c·u_d<0, C_cd, q_u, q_b oraz Δlogit i ΔIoU. Odsetek „oba aktywne” podać obok siły, nie zamiast niej. Oddzielić wynik na parach wcześniej zamrożonych od końcowo wybranych retrospektywnie.

**Jednostka:** film×seed; nie token–para.

**Falsyfikacja i minimum:** efekt znikający po dopasowaniu aktywności lub występujący tylko dla wag, nie signed wkładów, nie potwierdza funkcjonalnej kompensacji. Minimum: 24 klatki pilota, 3 checkpointy, ostatni MLP, wszystkie tokeny dla tanich statystyk i ustalona próbka dokładnych interwencji. Literatura: ostrożność z utożsamianiem neuronu z cechą [R7](https://transformer-circuits.pub/2022/toy_model/); test jest własnym zastosowaniem.

### C04. Kompensacja kontra nieliniowa interakcja

**Hipoteza:** deklarowana synergia przeżywa usunięcie efektu zwykłego sumowania wektorów i nieliniowości miary wyniku.

**Interwencja i kontrola:** pełny układ 2×2: brak, A, B, AB; α=0,1; 0,5; 1. Primary: dwie rozłączne grupy po 16 neuronów, łącznie 32; pary i łączne grupy 8/16 jako zakres dodatkowy. Porównać dokładny forward z liniowym przybliżeniem całej głowy wokół stanu bazowego. Residual i bias identyczne; nie ponownie dobierać A/B po zobaczeniu efektów. Dla wcześniejszych bloków przeliczyć downstream.

**Metryka:** q_y, I_y i reszta Taylora dla logitów, ltrb i krawędzi ustalonego tokenu. Normalizację I podać obok wartości absolutnej. Przy regresji exp(ltrb) porównać interakcję przed i po exp: część nieliniowości jest znana analitycznie.

**Jednostka:** film×seed, grupy jako pomiary wewnętrzne.

**Falsyfikacja i minimum:** I mieszczące się w tolerancji/praktycznym progu, przy dobrym przewidywaniu liniowym, falsyfikuje silną synergię w tej skali; nie falsyfikuje liniowej kompensacji. Minimum: 24 obrazy×3 późne stany, ostatni MLP, z góry ustalone grupy i kontrole. Literatura: konieczność jawnego określenia mierzonego efektu [R1](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html), [R4](https://arxiv.org/html/2407.08734v1).

### C05. Gdzie dokładnie zmienia się zwycięzca?

**Hipoteza:** grupy mogą oszczędzać współrzędne ustalonej ramki, a jednocześnie zmieniać ranking z funkcjonalną szkodą.

**Interwencja i kontrola:** A/B/AB globalnie w wszystkich 600 tokenach, pełny dekoder. Kontrola diagnostyczna 1: zamrożona bazowa maska lokalnych maksimów; kontrola 2: odczyt bazowego w; kontrola 3: tylko ten sam token. Są to różne interwencje, nie zamienniki primary. Zapisywać los każdego z bazowych top10 i nowych kandydatów, w tym zmiany dopuszczalności. Porównania grup z dopasowaną sumą norm i indywidualnymi skutkami oraz bazowym marginesem.

**Metryka:** odsetek zmian maski, składu top10, rankingu, top1, przypisanego GT i szkodliwych zmian IoU; rozkład ΔIoU i Δkrawędzi w. Primary zdarzenie do zamrożenia na discovery: A i B osobno zachowują top1, AB go zmienia i obniża IoU o ustalony praktyczny próg. Nie używać tylko największego skoku ani warunkować testu wyłącznie na przełączeniach.

**Jednostka:** film×seed; denominator obejmuje wszystkie zaplanowane interwencje, dodatkowo raport podpopulacji poprawnych wyjściowo detekcji.

**Falsyfikacja i minimum:** zmiany tokenu zwykle zachowujące pokrycie nie dowodzą destabilizacji lokalizacji. Efekt tylko z maski ogranicza claim do dekodera. Minimum: jak C04; w confirmatory 250 odłożonych klatek. Literatura: metryka zmienia interpretację patchingu [R1](https://arxiv.org/abs/2309.16042); sam rozkład etapów dekodera wynika z naszego kodu.

### C06. Czy prosty model marginesu wyjaśnia przełączenie?

**Hipoteza:** użyteczny efekt nie sprowadza się do przekroczenia przewidywalnego progu przez sumę liniowych zmian logitów.

**Interwencja i kontrola:** dla zamrożonej grupy policzyć pełne JVP zmiany wszystkich heat logits i ltrb względem bramki. W zadanej siatce α uruchomić zarówno rzeczywisty model, jak i **ten sam pełny dekoder** na pierwszorzędowo przewidzianych mapach. To uwzględnia przewidywane wejście/wyjście z maski pików. Dodatkowo baseline samego marginesu, a na stałej podpróbie drugi rząd. Próg rzeczywisty wyznaczać z prerejestrowanej siatki; nie zakładać monotoniczności przełączeń i nie stosować bezwarunkowo bisekcji.

**Metryka:** błąd przewidzianej najwcześniejszej dawki zmieniającej top1, trafność zmian maski i rankingu, błąd ΔIoU i kalibracja zdarzenia. Przypadki bez przełączenia do α=1 raportować jako cenzurowane na tej siatce.

**Jednostka:** film×seed.

**Falsyfikacja i minimum:** jeżeli pełny model liniowy odtwarza zdarzenia, nie ma podstaw nazywać ich nową nieliniową synergią. Minimum: dane C04/C05 i dodatkowe α=0,25;0,75 na ustalonej podpróbie; żadnego nowego treningu. Literatura: zależność wniosków od odczytu [R1](https://arxiv.org/abs/2309.16042), walidowanie faithfulness zamiast samej zgodności list składników [R5](https://arxiv.org/html/2403.17806v1).

### C07. Warunkowa konieczność: noising

**Hipoteza:** zachowanie określonego sygnału w grupie G jest potrzebne dla poprawnej odpowiedzi w czystym kontekście.

**Interwencja i kontrola:** dla czystego I oraz kontrolowanie zmienionego I' podmienić z_G(I) na z_G(I') w forward I; osobno A/B/AB. Powtórzyć dla korupcji narzędzia i dopasowanej korupcji tła. Bazy pomocnicze: zero, średnia z discovery, naturalny dawca dopasowany kontekstem. Dopasować normę wprowadzonej poprawki przy porównywaniu selektorów; raportować niedopasowania. Reszta sieci liczy się normalnie.

**Metryka:** utrata IoU i marginesu, zmiana krawędzi w, szkoda po dekoderze, koszt interwencji. Primary raw outcome; nie nazywać każdego spadku confidence utratą lokalizacji.

**Jednostka:** film×seed, korupcje i dawcy zagnieżdżone.

**Falsyfikacja i minimum:** brak przewagi względem dopasowanych grup odrzuca specyficzną konieczność tej grupy. Brak szkody jednej grupy nie dowodzi braku udziału, bo może działać redundancja. Minimum: 24 pary obrazów, jeden zamrożony checkpoint i ostatni MLP; replikacja checkpointów/seedów później. Literatura: noising i denoising odpowiadają na różne pytania [R2](https://arxiv.org/html/2404.15255v1); wrażliwość na wartości zastępcze [R4](https://arxiv.org/html/2407.08734v1).

### C08. Warunkowa wystarczalność i rescue: denoising

**Hipoteza:** wybrana grupa przenosi informację wystarczającą do odzyskania określonej części zachowania w zaburzonym kontekście.

**Interwencja i kontrola:** w I' zastąpić z_G(I') przez z_G(I); A/B/AB, a także większa grupa jako kontrola dodatnia. Negative: patch I'→I', losowa grupa tej samej siły, dawca z niepasującym narzędziem, poprawny dawca z innym tłem. Po translacji stosować znane dopasowanie przestrzenne; nie porównywać bezpośrednio różnych patchy tylko dlatego, że mają identyczny indeks. Nie kopiować wyjściowej ramki ani całego końcowego residual jako rzekomego odkrycia małego mechanizmu.

**Metryka:** raw odzysk IoU, błędu krawędzi i marginesu; odzysk top1/top10; koszt i coverage przypadków, w których korupcja faktycznie zmieniała wynik. Osobno pełna populacja i z góry określona podpopulacja skutecznej korupcji. R secondary według sekcji 2.

**Jednostka:** film×seed.

**Falsyfikacja i minimum:** recovery równoważne losowej poprawce, brak transferu lub tylko nieadekwatny wzrost confidence nie potwierdza funkcjonalnego rescue. Wystarczalność jest warunkowa względem reszty I', a nie dowodem że G samodzielnie wykrywa obiekt. Minimum: te same pary co C07. Literatura: [R2](https://arxiv.org/html/2404.15255v1), kontrola steering≠faithfulness [R3](https://arxiv.org/html/2311.17030v2).

### C09. Wystarczalność obwodu, kompletność i redundancja

**Hipoteza:** proponowany obwód C obejmuje istotne ścieżki badanego zachowania, a nie tylko jedną z wielu wymiennych dróg.

**Interwencja i kontrola:** narysować przed testem graf obejmujący wybrane tokeny, MLP/attention i połączenia do głowy. W jednej procedurze resamplować składniki poza C, w drugiej w C; mierzyć zachowanie obu. W aktywnym C węzły **przeliczać z bieżących rodziców**, nie zamrażać do czystych cache. Dla dwóch kandydackich grup badać pojedyncze i wspólne usunięcie, następnie przywrócenie jednej przy usuniętej drugiej. Wyłączać po jednym elemencie C oraz kilka uprzednio wskazanych redundancji. Na discovery wolno zrewidować graf; na confirmatory nie.

**Metryka:** krzywa faithfulness–liczba składników dla stałej metody ablacji, kompletność względem efektów dodatkowych grup, koszt opisu i reszta predykcji grafu. Wyniki osobno dla zero/mean/resample; nie uśredniać rozbieżnych metod do jednego „prawdziwego” score.

**Jednostka:** film×seed; wariant grafu nie jest replikacją.

**Falsyfikacja i minimum:** mały obwód działa tylko przy kopiowaniu gotowych czystych wyjść, zależy od jednej arbitralnej bazy albo pozostawia duży niewyjaśniony efekt poza C — claim kompletnego obwodu upada. Minimum: 2 grupy i ich komplement w ostatnim MLP na 24 parach; graf wielowarstwowy tylko dla mocniejszego claimu. Literatura: [R4](https://arxiv.org/html/2407.08734v1), [R5](https://arxiv.org/html/2403.17806v1).

### C10. Kontrola iluzji subspace patching i stanów nietypowych

**Hipoteza:** skuteczna podmiana odtwarza normalnie używaną informację, a nie tworzy sztuczną drogę sterowania wyjściem.

**Interwencja i kontrola:** porównać grupy rzeczywistych neuronów z podprzestrzenią dopasowaną tylko na discovery; ta sama norma poprawki i wymiar. Podmiany naturalne między parami o kontrolowanej semantyce, neutralne tło i sztuczny kierunek sterujący traktować jako odrębne warunki. Sprawdzić, czy kierunek ma obserwowalny sygnał w naturalnych aktywacjach i czy dokładna ablacja tego sygnału zmienia zwykły przebieg. Dla znalezionego kierunku analizować składową poza zakresem naturalnej zmienności; nie nazywać jej „causally disconnected” tylko na podstawie małego lokalnego gradientu. Kontrolny mały graf z uśpioną równoległą drogą ma wykazać, że nasz zestaw testów odrzuca pozorną interpretację mimo skutecznego steeringu.

**Metryka:** skuteczność rescue versus szkoda po naturalnej ablacji, transfer między dawcami, odległość aktywacji od rozkładu discovery oraz zgodność przewidzianej ścieżki przy C12/C13. Odległość jest ostrzeżeniem, nie dowodem manifold membership.

**Jednostka:** film×seed; graf syntetyczny jest testem instrumentu.

**Falsyfikacja i minimum:** sterowanie istniejące tylko dla syntetycznej mieszanki, bez dowodów normalnego użycia sygnału, nie lokalizuje mechanizmu. Minimum: wszystkie C08 przypadki i dwie rodziny dawców; zaawansowana analiza subspace tylko jeśli taki claim trafi do pracy. Literatura: główny problem i kontrprzykład [R3](https://arxiv.org/html/2311.17030v2).

### C11. Lokalizacja informacji: neuron, token, sąsiedztwo, warstwa

**Hipoteza:** lokalne zespoły pozwalają wskazać rozmiar i miejsce sygnału potrzebnego detekcji, a efekt nie wynika wyłącznie z większego budżetu podmienionych aktywacji.

**Interwencja i kontrola:** porównać pojedynczy token, 3×3 sąsiedztwo, cały obszar maski i wszystkie tokeny; osobno subset neuronów oraz pełny residual. Zmierzyć krzywe względem liczby zmienionych skalarów i normy poprawki. Miejsca z GT są oracle; selektor bez GT ma własną ścieżkę testu. Patching czysty→neutralny obraz stanowi wariant porównawczy CAAP, natomiast główny test wykorzystuje dopasowane naturalne konteksty. Zmiana budżetu, warstwy i dawcy nie może być ukryta pod wspólną nazwą „patching”.

**Metryka:** odzysk jakości przy danym budżecie; selektywność narzędzie–tło; krzywa coverage–cost; porównanie z gradient×activation, IG po bramce i rankingiem dokładnych ablacji pojedynczych.

**Jednostka:** film×seed; przestrzenne pozycje są zagnieżdżone.

**Falsyfikacja i minimum:** przewaga znika przy dopasowaniu normy/liczby skalarów albo działa tylko patch całego residual — brak dowodu małego wyspecjalizowanego zespołu. Minimum: ostatni MLP i jeden wcześniejszy blok, 24 pary. Literatura: CAAP jest już pracą o patchingu reprezentacji ViT i sąsiedztw patchy [R10](https://arxiv.org/abs/2603.13652); nie przedstawiać samego patchingu ViT jako nowości.

### C12. Czy późniejszy blok naprawdę koryguje wcześniejszy?

**Hipoteza:** zmiana w wcześniejszym komponencie A wywołuje w późniejszym B odpowiedź przeciwdziałającą jej wpływowi na określone wyjście.

**Interwencja i kontrola:** dla A przed B cztery przebiegi: bazowy; ablacja A z normalną rekalkulacją downstream; ablacja A z B zablokowanym do bazowej wartości; tylko zmiana B na wartość powstałą po ablacji A przy reszcie bazowej. Porównać node-patching z jawnie wskazaną ścieżką A→B i A→residual→head; kontrolować pozostałe drogi w zdefiniowanym grafie. Zapisane wartości B przy blokowaniu są celową interwencją diagnostyczną, a nie „normalnym forward”. Osobno zamrozić skale statystyk normalizacji do wartości bazowych, zachowując affine parametry, aby odróżnić renormalizację od sygnału innego neuronu.

**Metryka:** różnica szkody po ablacji A z aktywnym versus zablokowanym B, signed projekcja poprawki B na zmianę logitów/krawędzi, odzysk po transplantacji B. Ocena na ciągłych odczytach i dekoderze.

**Jednostka:** film×seed.

**Falsyfikacja i minimum:** B nie odpowiada przeciwnie albo pozorna naprawa wyjaśnia się wyłącznie normalizacją — odrzucić claim neuronowej korekty lub zawęzić do normalizacji. Nie stosować jako dowodu c→d w tym samym MLP. Minimum: jedna wcześniej wybrana para kolejnych bloków na 24 obrazach; rozszerzenie do różnych głębokości dopiero po sygnale. Literatura: downstream self-repair opisano w LM [R8](https://arxiv.org/abs/2307.15771), w tym rolę normalizacji [R9](https://arxiv.org/abs/2402.15390); ich obecność w ViT jest tutaj testowana.

### C13. Causal scrubbing konkretnej hipotezy, nie dowolnej korelacji

**Hipoteza:** zależność „cecha narzędzia → wskazany sygnał → lokalizacja” jest wystarczającym opisem określonej części obliczeń.

**Interwencja i kontrola:** przed badaniem zdefiniować zmienne wyższego poziomu, np. położenie obiektu i typ zaburzenia, oraz mapowanie do konkretnych węzłów/ścieżek. Resamplować stany z przykładów zgodnych w hipotetycznie istotnej zmiennej i zmiennych w hipotetycznie nieistotnej; porównać z resamplingiem bez tego warunku i z warunkiem na złą zmienną. Na ścieżkach wymagających różnych dawców użyć poprawnego rozgałęzionego grafu obliczeń i jawnego schematu samplera; wspólny losowy dawca dla wszystkiego nie jest automatycznie causal scrubbing. Donor pool wyłącznie discovery; brak dopasowania dawcy musi być raportowany.

**Metryka:** rozbieżność od pełnego modelu i od przewidzianej interwencji semantycznej, zachowane IoU/logity, sukces na nowych kombinacjach położenie×tło. Nie oceniać tylko, czy output pozostaje „jakiś poprawny”.

**Jednostka:** film×seed; losowania dawcy to niepewność Monte Carlo wewnątrz jednostki.

**Falsyfikacja i minimum:** resampling dopuszczony przez hipotezę niszczy przewidywane zachowanie lub hipoteza trywialna z tym samym budżetem działa równie dobrze. Minimum: jedna jasno sformułowana hipoteza i mały graf na kontrolowanych parach; nie obiecywać pełnego scrubbingu całego ViT. Literatura: autorski raport metody, nie recenzowany artykuł konferencyjny [R6](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing).

### C14. Semantyka obrazu: narzędzie, tło i equivariance lokalizacji

**Hipoteza:** wskazana kompensacja chroni przed zmianami nieistotnymi, a przekazuje zmianę rzeczywistego położenia narzędzia.

**Interwencja i kontrola:** pary obrazów: maskowane rozmycie/zasłonięcie narzędzia versus tła o dopasowanej powierzchni i sile; zmiana jasności; translacja/skala z prawidłowo zmienionym GT. Użyć maski segmentacji lub ręcznie sprawdzonego fragmentu; bbox oznacza tylko region. Kontrolować brzegi maski, wygląd zasłony i przesunięcia patchowej siatki. Każdą perturbację wykonać z aktywną grupą oraz po jej ablacji, z dopasowaną grupą kontrolną. Ratowanie C08 stanowi dodatkowy trzeci warunek.

**Metryka:** różnica zmian błędu lokalizacji między grupą a kontrolą, osobno dla nuisance i zmiany położenia; błąd equivariance po odwrotnym odwzorowaniu transformacji, confidence/logity osobno. Interakcja (interwencja grupy × rodzaj perturbacji) primary, nie samo pogorszenie po zasłonięciu.

**Jednostka:** film×seed; rodziny perturbacji rozdzielone między development i test użyteczności.

**Falsyfikacja i minimum:** jednakowy efekt dla narzędzia i tła, dominacja artefaktów brzegowych lub tłumienie także potrzebnej reakcji na przesunięcie przeczy postulowanej funkcji. Minimum: 24 zweryfikowane obrazy, trzy rodziny perturbacji; docelowo 250 klatek testu z prerejestrowanym podzbiorem masek. Literatura: projekt jest naszym testem semantycznym; ograniczenia baz patchingu [R1](https://arxiv.org/abs/2309.16042) i ViT attribution [R10](https://arxiv.org/abs/2603.13652).

### C15. Geometria a IoU, confidence i margines: interwencja rozdzielająca

**Hipoteza:** orientacja mierzonej geometrii ma znaczenie funkcjonalne ponad rozmiar zmiany i aktualny margines.

**Interwencja i kontrola:** wykorzystać poprawnie zakotwiczoną interwencję lokalnego pola z audytu matematycznego: utrzymuje bazowy punkt i wartość F(x0), modyfikuje odpowiedź na δ, np. F̃(x0+δ)=F(x0+δ)−λ A(x0)δ dla jawnie zdefiniowanej części A. Porównać z korektą losową o tej samej normie oraz kontrolą zmiany samej skali. Osobno zastosować do δ indukowanych przez pary obrazów z C14 i do losowych δ jako testu matematycznego. Przeliczyć pełną resztę sieci; nie wnioskować o IoU wyłącznie z projekcji 3D. Konkretne S/A i pełna normalizacja według nadrzędnego protokołu matematycznego.

**Metryka:** osobne ΔIoU, błąd krawędzi w, Δlogit, Δmargines i zmiany maski/topk; regresja efektu względem normy, wyjściowej jakości i pełnego JVP. Dodać kontrolę, czy geometryczne cechy wnoszą wartość ponad czułość zadaniową.

**Jednostka:** film×seed; δ zagnieżdżone.

**Falsyfikacja i minimum:** równie silny efekt kontroli normy, brak transferu lub zmiana jedynie confidence nie potwierdza zadaniowej roli orientacji. Minimum: wspólna podpróba E2/C14, aby nie mnożyć badań. Literatura: pułapka utożsamiania steeringu z lokalizacją mechanizmu [R3](https://arxiv.org/html/2311.17030v2); sama modyfikacja pola jest propozycją naszego badania, nie przepisem z tej pracy.

### C16. Generalizacja przyczynowego efektu i użyteczność ponad baseline

**Hipoteza:** metoda przewiduje i wyjaśnia interwencyjne zachowanie poza danymi, na których została wybrana.

**Interwencja i kontrola:** zamrozić regułę, hiperparametry, pary w obrębie seedu, rodziny perturbacji i predyktor przed testem. Oceniać: inne filmy w tym samym checkpoincie; późniejsze checkpointy bez podglądania przyszłości; nowe seedy; odłożoną rodzinę zaburzeń. Dla predykcji ryzyka dostępny tylko czysty obraz/model i zadana klasa perturbacji, nie jej faktyczna realizacja ani GT. Baseline: confidence, margines, rozbieżność ramek topk, normy wkładów, gradient×activation, IG bramki, pełne informacje o czułości w porównywalnym budżecie. Dla znanej interwencji baseline pełnego JVP jest oddzielnym zadaniem rekonstrukcyjnym.

**Metryka:** primary sparowany przyrost właściwej funkcji oceny, np. Brier/log-loss dla ustalonego zdarzenia; calibration, AUPRC z częstością bazową, risk–coverage oraz koszt. Dla wyjaśniania również wielkość rzeczywistego efektu C07/C08, nie wyłącznie predykcja.

**Jednostka:** film×seed; test zablokowany.

**Falsyfikacja i minimum:** brak przyrostu ponad mocny baseline eliminuje claim wskaźnika jakości; transfer tylko algorytmu nie potwierdza trwałych par. Minimum: etapy i seedy nadrzędnego PLAN; na 24 klatkach wyłącznie go/no-go i szacunek precyzji. Literatura: różnica rankingu elementów i faithfulness mechanizmu [R5](https://arxiv.org/html/2403.17806v1).

### C17. Superposition i cechy: test opcjonalny wymagany tylko dla tego claimu

**Hipoteza:** zespoły kodują kontrolowalne cechy rozproszone, a pojedynczy neuron nie jest prostym detektorem jednego semantycznego atrybutu.

**Interwencja i kontrola:** mały kontrolowany benchmark obrazów z niezależnie zmienianymi cechami, np. położenie, kształt, kontrast i tło; część kombinacji odłożona. Porównać neurony i podprzestrzenie/ewentualne SAE o dopasowanym budżecie. Przy SAE utrzymać lub jawnie interweniować w resztę rekonstrukcji, zmierzyć jakość modelu po rekonstrukcji, porównać rank-matched PCA/random; cechy SAE nie są automatycznie prawdziwymi zmiennymi przyczynowymi. Podmieniać sygnał jednej cechy między przykładami, badając pożądaną zmianę i skutki uboczne na pozostałych.

**Metryka:** selektywność, compositional transfer, odzysk i necessity, leakage do innych cech, wpływ błędu rekonstrukcji na detekcję. Same rank, kąty W2, liczba neuronów ponad residual i liniowa dekodowalność są niewystarczające.

**Jednostka:** niezależne sceny/generatory i seedy; syntetyka nie zastępuje testu na filmach.

**Falsyfikacja i minimum:** tylko korelacja/decoding albo skuteczne steering bez przewidzianej selektywności nie potwierdza superposition cech. Minimum: pominąć ten claim, jeśli brak zasobów na kontrolowany benchmark; nie jest obowiązkową częścią wąskiej pracy o kompensacji. Literatura: znane cechy w modelach zabawkowych [R7](https://transformer-circuits.pub/2022/toy_model/), kontrola iluzji [R3](https://arxiv.org/html/2311.17030v2).

### C18. Konkurencyjne mechanizmy i próba obalenia wybranego wyjaśnienia

**Hipoteza:** proponowany mechanizm tłumaczy rozstrzygające interwencje lepiej niż proste alternatywy.

**Interwencja i kontrola:** przed confirmatory zamrozić co najmniej trzy konkurencyjne modele: M0 liniowe sumowanie plus dekoder; M1 kompensacja kierunków w odczycie bez specjalizacji semantycznej; M2 zadaniowo wyspecjalizowane zespoły z przewidzianą ścieżką. Wybrać na discovery warunki, w których ich przewidywania się różnią, np. dopasowana zmiana tła/narzędzia i noising/denoising z zablokowanym późniejszym komponentem. Oceniać całe prerejestrowane zestawy interwencji, nie tylko te z najbardziej różnymi prognozami po zobaczeniu wyniku testowego. Użyć tych samych danych i budżetu estymacji.

**Metryka:** błąd przewidywania dokładnych zmian ciągłych, maski i jakości, złożoność opisu oraz liczba kontrprzykładów. Wymagany zakres predykcji obejmuje również brak efektu i skutki uboczne.

**Jednostka:** film×seed; interwencje jako powtarzane próby tego samego modelu.

**Falsyfikacja i minimum:** M0/M1 równie dobre na odłożonych rozstrzygających interwencjach albo poprawne prognozy tylko po zmianie hipotezy — wycofać mocniejszy claim M2. Minimum: reuse C04–C16; bez dodatkowego dużego atlasu. Literatura: rygorystyczne testowanie jawnej hipotezy [R6](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), rozdzielenie steering i faithful explanation [R3](https://arxiv.org/html/2311.17030v2).

## 4. Kolejność i budżet, bez uruchamiania teraz

1. **Brama instrumentu:** C01 na E0; weryfikacja danych, identyfikacji filmu i historii użycia TEST. Nie wykonujemy kosztownego patchingu przed zgodnością pełnego forward.
2. **Pilot mechanizmu:** C02–C08 na 24 klatkach z 12 filmów VAL. C03 tanio na checkpointach E1, dokładne interwencje przede wszystkim 33000, 55000, 65640. C07–C12 zacząć od stanu końcowego i jednego wcześniejszego bloku dla self-repair. Te wybory zamrozić przed oglądaniem wyników.
3. **Weryfikacja wyjaśnienia:** C09–C14 na tych samych parach i ograniczonym grafie. C10 jest obowiązkowy przy subspace claim; C13 wymagany przy claimie jawnego obwodu, a nie tylko efektu grupy. C15 współdzieli dane z audytem geometrii.
4. **Rozwój:** atlas 250 klatek z 25 VAL; podział po filmach na selection/calibration/validation. Istniejący s47 i dwa dodatkowe kompletne seedy rozwojowe zgodnie z planem. Stare krótkie biegi wykorzystać tylko w granicach ich zgodności.
5. **Potwierdzenie:** zablokować kod, kryteria, listę analiz i praktyczny efekt; trzy nowe seedy oraz 250 klatek z 25 odłożonych filmów, jeśli audyt pozwala na niezależny TEST. Wynik pełnego detektora dodatkowo na pełnym zatwierdzonym TEST. C16/C18 rozstrzygają główne twierdzenie. Ta liczba seedów jest budżetem startowym, nie gwarancją mocy.
6. **Rozszerzenie:** większa liczba niezależnych filmów/seedów wynikająca z precyzji pilota; druga architektura i dekoder, dodatkowy zbiór; C17 tylko jeśli potrzebna teza o superposition. Kontynuacje treningu E3 wyłącznie do twierdzenia o przyczynie uczenia.

Minimalny sensowny pakiet publikacyjny o inferencji nie wymaga przeprowadzenia każdego opcjonalnego rozszerzenia. Wymaga C01–C08, odpowiedniej części C09–C14, C16 i C18 oraz jasnego zawężenia claimu. Artykuł o geometrii dodaje C15; o przyczynach uczenia — E3; o superposition — C17. Samo wykonanie listy testów nie gwarantuje nowości ani akceptacji konferencyjnej.

Proponowane kontrole losowe: stały zestaw 20 dopasowanych grup na obraz/warunek w pilocie; liczba może zostać zmieniona przed confirmatory według błędu Monte Carlo. Nie przedstawiać 20 kontroli jako 20 niezależnych filmów. Do poprawnego porównania zachować zarówno matching diagnostyczny, jak i rozkład wszystkich interwencji; brak dopasowania to osobny wynik.

Przy rejestracji primary ograniczyć rodzinę testów do kilku kontrastów, np. C14 semantyczna interakcja, C08 przewaga rescue oraz C16 przyrost predykcyjny. Wybrać je na discovery, podać minimalny praktyczny efekt i skorygować wielokrotność confirmatory. Pozostałe analizy są mechanistycznym wsparciem lub eksploracją. Nie wybierać najkorzystniejszego wyjścia spośród IoU, confidence, marginesu i krawędzi po poznaniu wyników.

## 5. Co może być wnioskiem, a co nim nie będzie

| Wynik | Uzasadniony wniosek | Nieuzasadnione rozszerzenie |
|---|---|---|
| Małe q i małe I, także przy inicjalizacji | Dobór przeciwstawnych wpływów, liniowa kompensacja | Specjalny wyuczony mechanizm lub synergia |
| Małe q przenosi się na filmy i zależy od semantyki; rescue działa | Zadaniowa rola zespołu w badanym kontekście i interwencji | Samodzielny obwód całej detekcji |
| Przełączenie odtwarzane przez pełny liniowy model + dekoder | Próg selekcji wyjaśnia skok ramki | Nieliniowa interakcja neuronów MLP |
| Wzrasta confidence, IoU nie | Zmiana wyniku heat/rankingu | Wzrost pewności poprawnego położenia |
| Rescue tylko z neutralnego tła lub nietypowej podprzestrzeni | Możliwość sterowania wyjściem | Wierna lokalizacja używanego mechanizmu |
| Ablacja wcześniejszego bloku wywołuje przeciwną odpowiedź późniejszego, potwierdzoną blokowaniem ścieżki | Warunkowa kompensacja downstream | Jeden równoległy neuron odpala drugiego w tym samym MLP |
| Stałe pary zawodzą, adaptacyjna reguła pomaga na teście | Użyteczny lokalny instrument | Trwałe zespoły neuronów wspólne dla obrazów i seedów |
| Brak wartości ponad margines i pełną czułość | Negatywny wynik o wystarczalności analogii/geometrii | Uniwersalny fizyczny wskaźnik jakości detekcji |

## 6. Literatura pierwotna i status weryfikacji

Nowe wyszukiwanie wykorzystało skill paper-search, SciSpace oraz Scite; identyfikację zweryfikowano na stronach autorów, arXiv lub oficjalnych materiałach konferencji. Zapis zapytań i wyników: [causal_zrodla.json](causal_zrodla.json). Scite nie udostępnił pełnego tekstu trzech głównych prac; użyto pierwotnego HTML. Brak pola editorialNotices w odpowiedzi nie jest potwierdzeniem braku korekt/wycofania. Wyniki wyszukiwania oraz fragmenty cytujących publikacji nie stanowią niezależnej replikacji ich wniosków.

**R1.** Zhang, F., & Nanda, N. (2024). *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. ICLR. [Oficjalna publikacja](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html). Zweryfikowano oficjalny abstrakt: wybór korupcji i miary zmienia wyniki patchingu. Nie traktować procedur LM jako gotowej walidacji dla detekcji.

**R2.** Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching*. Preprint. [Pełny HTML](https://arxiv.org/html/2404.15255v1). Czytano rozróżnienie discovery/confirmatory, noising/denoising, AND/OR oraz path patching. To podstawa rozdzielenia pytań, nie gwarancja identyfikacji mechanizmu.

**R3.** Makelov, A., Lange, G., & Nanda, N. (2023). *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*. Preprint. [Pełny HTML v2](https://arxiv.org/html/2311.17030v2). Badanie pokazuje, że skuteczne sterowanie subspace może nie odpowiadać normalnie używanej reprezentacji; dlatego C10/C18 nie kończą się na sukcesie rescue.

**R4.** Miller, J., Chughtai, B., & Saunders, W. (2024). *Transformer Circuit Faithfulness Metrics Are Not Robust*. COLM. [Oficjalny PDF](https://openreview.net/pdf?id=zSf8PJyQb2), [HTML](https://arxiv.org/html/2407.08734v1). Czytano definicje i porównania granularity/component/value/positions/direction/set oraz rekomendacje. W naszym protokole każdy shard musi zapisać te wybory.

**R5.** Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*. [Pełny HTML](https://arxiv.org/html/2403.17806v1). Czytano definicję odtwarzania obwodu i porównania faithfulness. Podobieństwo list neuronów nie zastępuje pomiaru zachowania obwodu, a jego aktywne węzły trzeba rzeczywiście przeliczać.

**R6.** Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., Nitishinskaya, J., Radhakrishnan, A., Shlegeris, B., & Thomas, N. (2022). *Causal Scrubbing: a method for rigorously testing interpretability hypotheses*. Autorski raport techniczny. [Tekst](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), [strona autora z metadanymi](https://chanlawrence.me/publication/chan-2022-causal/). Metoda określa warunkowe resamplingi dopuszczone przez hipotezę i sprawdza zachowanie modelu; nie jest nazwą dowolnego shuffle aktywacji.

**R7.** Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits. [Raport autorów](https://transformer-circuits.pub/2022/toy_model/), [arXiv](https://arxiv.org/abs/2209.10652). Znane cechy w modelach zabawkowych pozwalają badać ich współdzielone kodowanie. Samo sumowanie wkładów w naszym MLP nie spełnia tego kryterium.

**R8.** McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., & Legg, S. (2023). *The Hydra Effect: Emergent Self-repair in Language Model Computations*. [Preprint i abstrakt pierwotny](https://arxiv.org/abs/2307.15771). Zweryfikowano opis downstream compensation w LM; nie przenoszono go na checkpointową dynamikę uczenia.

**R9.** Rushing, C., & Nanda, N. (2024). *Explorations of Self-Repair in Language Models*. ICML, PMLR 235, 42836–42855. [Metadane i abstrakt pierwotny](https://arxiv.org/abs/2402.15390). Zweryfikowano rolę LayerNorm i zmienność self-repair w badanych LM. To uzasadnia kontrolę normalizacji w C12, nie zakładanie analogicznego efektu w ViT.

**R10.** Izadi, A., Banayeeanzade, M., Mirrokni, A., Hasani, H., Bagherian, M., Mehri, F., & Soleymani Baghshah, M. (2026). *Causal Attribution via Activation Patching*. Preprint. [arXiv](https://arxiv.org/abs/2603.13652). Zweryfikowano abstrakt pierwotny i fragmenty pełnego tekstu przez Scite: patchowanie reprezentacji ViT do neutralnego kontekstu i wykorzystanie sąsiedztw patchy. To bezpośrednie prior art dla samego pomysłu patchingu ViT; nasz potencjalny wkład dotyczy mechanizmu detekcji, transferu i kontroli.

## 7. Artefakty wymagane przy przyszłym wykonaniu

Dla każdego wariantu zachować manifest obrazu/filmu/GT, checkpoint i hash, seed, blok, tokeny i grupy neuronów, selektor oraz jego dane treningowe, dawkę, bazę/dawcę, normę interwencji, pełną definicję patchingu, wersję dekodera i informację czy analiza używa oracle GT. Zapisać wartości ciągłe i indeksy kandydatów przed/po, a także przyczyny braku dopasowanej kontroli i niewyznaczalnych ilorazów. Jeden wiersz wyniku nie powinien wymagać odtwarzania ukrytej heurystyki, aby ustalić co faktycznie zmieniono.

Raport główny ma zawierać wszystkie prerejestrowane kontrasty, wyniki zerowe, odsetek nieudanych korupcji, niepewność po filmach i seedach oraz rozróżnienie wniosków o inferencji od wniosków o uczeniu. Animacje cząsteczek i „reflektorów” mogą ilustrować wynik, ale nie zastępują tych pomiarów.

