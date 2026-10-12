# Z czego zbudować artykuł o dynamice tokenów ViT

Stan oceny: 2026-10-10. Dokument strategiczny, nie raport z nowych treningów. Podstawa: audyt pełnych 768 wymiarów, odczyt nowszego pilotażu pokręteł oraz celowane porównanie literatury przez Undermind i strony publikacji. Nie jest to wyczerpujący dowód pierwszeństwa.

## Rekomendacja

Najmocniejszy kierunek to **przyczynowe rozdzielenie ruchu reprezentacji od zmian istotnych dla zadania**. Obecne wyniki dają materiał do części empirycznej: ruch ma pamięć, zawracanie, ograniczony wzrost przemieszczeń oraz odmienne tendencje na poziomie zbiorowym i lokalnym. Potencjalnie większy wkład wymaga wykazania, co powoduje te zjawiska i czy wiedza o nich umożliwia trafną decyzję podczas uczenia.

Na dziś nie ma podstaw, by ogłosić odkrycie przejścia szklistego ani przełom. Jest obiecujący program badawczy i gotowa infrastruktura do jego sprawdzenia. Liczba wymiarów, zdjęć i epok zwiększa wiarygodność pomiaru, ale sama nie stanowi nowości naukowej.

Proponowany roboczy tytuł głównego artykułu:

**Multi-scale dynamics of vision-transformer tokens: separating collective drift, local rearrangements, and functional change**

Tytuł opisuje cel projektu. Człon o zmianie funkcjonalnej wymaga jeszcze wyników; nie należy przedstawiać go jako obecnie dowiedzionego rezultatu.

## Co już jest, a czego brakuje

| Element | Obecny status | Rola w artykule |
|---|---|---|
| Śledzenie stałych tokenów 768D | Zweryfikowane: 8 zdjęć, 1533 gęste stany, 806 checkpointów długiego biegu | Metoda i materiał badawczy |
| Pamięć kierunku i zawracanie | Potwierdzone po warmup 300; VACF +0,678 przy odcinkach 1-krokowych, −0,299 przy 10-krokowych | Charakterystyka wielu skal czasu |
| Różnica ruchu zbiorowego i lokalnego | Lokalny MSD przy stałym opóźnieniu rośnie 2,20 raza w gęstym biegu; część dużych przemieszczeń surowych zwalnia | Najciekawsza obecna obserwacja, wymagająca replikacji |
| Ograniczony ruch i szkło | Częściowe plateau istnieje; model OU z pamięcią i dyfuzją odtwarza wczesną krzywą | Kontrola alternatywnego wyjaśnienia, nie identyfikacja mechanizmu |
| Normalizacja | Wzrost późnego MSD kierunków około 47% spada do około 2,5% przy stałych statystykach BN | Silna wskazówka do interwencji, obecnie kontrola wrażliwości |
| Uogólnienie na populację | Audyt 768D obejmuje 8 zdjęć jednej klasy z 7 filmów; dostępne osobne, szersze pomiary PCA TRAIN/VALID | Szersze pomiary trzeba sprawdzić i wybrane powtórzyć w 768D |
| Znaczenie dla detekcji | Nieustalone dla rozdzielonych składowych ruchu | Główna luka do uzupełnienia |
| Sterowanie treningiem | Brak prospektywnego testu reguły opartej na dynamice tokenów | Potencjalnie największy wkład użytkowy |

Audyt: [RAPORT.md](../../outputs/audyt_szkla_768d_20261010/RAPORT.md).

## Najbliższa literatura i granica nowości

1. Pashakhanloo i Koulakov, *Stochastic Gradient Descent-Induced Drift of Representation in a Two-Layer Neural Network*, ICML 2023. Rozdzielają ograniczone fluktuacje i dyfuzję po rozmaitości minimów w liniowej sieci; stosują model OU i średnie kwadratowe przemieszczenie kątowe. Sama obserwacja ruchu przy stałej jakości ani połączenie fluktuacji i dyfuzji nie jest nowe. [Publikacja](https://proceedings.mlr.press/v202/pashakhanloo23a.html).
2. Halperin, *Learning as Observable Matrix Dynamics: Diffusive Relaxations versus Phase Transitions*, preprint 2026. Śledzi geometrię reprezentacji stałych wejść przez macierze odległości i diagnostyki widmowe. To bliska praca, którą trzeba porównać. W sprawdzonym pełnym tekście nie ma zestawu patch-token MSD, VACF, lokalnych przemieszczeń, aging oraz interwencji w momentum i BN. Ta różnica wyznacza możliwość wkładu, ale nie dowodzi pierwszeństwa wobec całej literatury. [Preprint](https://arxiv.org/abs/2606.29679).
3. Gigante i in., *Visualizing the PHATE of Neural Networks*, NeurIPS 2019. Wizualizacja ewolucji reprezentacji podczas treningu ma długą historię. Animacja chmury punktów i użycie języka cząsteczek nie wystarczą jako wkład. [Publikacja](https://arxiv.org/abs/1908.02831).
4. Winter i Janssen, *Glassy dynamics in deep neural networks: A structural comparison*. Podobieństwa i różnice wobec szkła oraz nieswoistość części objawów są już przedmiotem literatury. Wkładem może być ich test dla tokenów rzeczywistego detektora z kontrolami, a nie samo zdanie „plateau nie dowodzi szkła”. [Tekst](https://arxiv.org/abs/2405.13098).
5. NTK i TracIn: zależność zmiany funkcji od Jacobianów oraz wpływ próbek treningowych są istniejącymi koncepcjami. Nowa metoda musiałaby dodać zweryfikowane przewidywanie ruchu tokenów i jego skutków, nie tylko zapisać znany wzór. [NTK](https://arxiv.org/abs/1806.07572), [TracIn](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html).

Pełne teksty Halperina oraz Pashakhanloo i Koulakova sprawdzono przez Undermind. Pozostałe punkty opierają się na poprzednim audycie oraz stronach publikacji/abstraktach, zależnie od pracy. Nie przypisujemy wszystkim pracom pełnotekstowego przeglądu w tym etapie.

## Trzy możliwe poziomy wkładu

**A. Artykuł empiryczny: rozdzielenie skal dynamiki.** Pytanie: czy lokalna przebudowa trwa mimo spowolnienia ruchu zbiorowego i stabilizacji jakości? Potrzebne: odtworzenie zjawiska na innych inicjalizacjach, klasach, warstwach i modelu porównawczym; kontrola geometrii, BN i skali czasu. To najbliższa droga do spójnego artykułu. Ujemny wynik dotyczący szkła jest częścią interpretacji, nie jedyną tezą.

**B. Artykuł mechanistyczny: co porusza tokenami.** Pytanie: czy ruch wynika głównie z pamięci optymalizatora i normalizacji, czy pozostaje struktura wspólnych odpowiedzi na gradienty, która przewiduje lokalne przemieszczenia? Możliwy wkład: porównanie sąsiadów geometrycznych z zależnościami zmierzonymi przez kontrolowane aktualizacje; przewidywanie odpowiedzi na zmianę paczki lub parametrów treningu. Najbliższy sąsiad w latencie nie musi być najsilniejszym źródłem wspólnego ruchu. Sam Jacobian i NTK nie są nowym odkryciem.

**C. Artykuł o metodzie: diagnostyka prowadząca do lepszego treningu.** Pytanie: czy miara rozróżniająca drgania, ruch zbiorowy i przebudowę użyteczną dla zadania potrafi wybrać checkpoint albo zmianę learning rate lepiej niż standardowe reguły? To ma największy potencjał praktyczny. Obecnie jest hipotezą, nie wynikiem. Sukces musi wystąpić na nowych treningach, przy wliczonym koszcie obserwatora i strojenia reguły.

Nie zalecam rozbijania obecnego materiału na trzy małe publikacje. Najpierw jeden spójny artykuł A+B, a C tylko jeśli eksperyment decyzyjny wykaże korzyść.

## Eksperymenty rozstrzygające, w kolejności

### 1. Czy zmienia się geometria i wynik zadania, czy głównie opis współrzędnych?

Na istniejących checkpointach porównać ruch surowy, po kontroli norm i BN, geometrię odległości między tokenami oraz dopasowanie wspólnego obrotu/przesunięcia. Dopasowanie wykonywać na oddzielnym zbiorze kalibracyjnym i sprawdzać na innych zdjęciach. Nie dopasowywać niezależnego obrotu do każdego obrazu, bo może to usunąć interesujące zmiany. CKA może być dodatkowym odniesieniem, ale nie dowodem braku zmian funkcjonalnych.

Równolegle mierzyć zmiany predykcji: heatmap, pudełek, błędów i kalibracji. Rozdzielić tokeny narzędzia i tła na podstawie dostępnych etykiet, jawnie uwzględniając niedokładność przypisania patcha do obiektu. Kontrole ze stałą głową lub podmianą aktywacji interpretować z uwzględnieniem możliwego niedopasowania reprezentacji do głowy.

**Wynik pozytywny:** resztowa przebudowa przewiduje konkretne zmiany predykcji ponad prostą wielkość kroku, stratę i standardowe podobieństwo reprezentacji, również na odłożonych filmach/treningach. **Jeśli nie:** zawęzić artykuł do dynamiki geometrii; nie nazywać ruchu użytecznym uczeniem.

### 2. Test przyczynowy momentum, normalizacji i wymuszenia

Krótkie, sparowane kontynuacje od tego samego stanu modelu, pełnego stanu optymalizatora i generatorów losowych, z kontrolowaną kolejnością danych. Jeśli nie zapisano pełnego stanu, wykonać odtwarzalny pilot zamiast traktować restart optymalizatora jak identyczną kontynuację. Zmieniać osobno momentum, learning rate i skład/szum paczek. Warmup i okres przejściowy po interwencji jawnie wydzielić.

Porównywać zarówno kroki, liczbę ekspozycji danych, jak i zegar oparty na skumulowanej wielkości aktualizacji. Zmiana momentum może jednocześnie zmieniać amplitudę kroku, dlatego potrzebne są także kontrole jej skali. Zamrożenie BN w inferencji nie jest równoważne treningowi bez BN. AdamW z beta1=0 nadal nie jest zwykłym SGD.

Test Jacobianu zaczynać od przewidywania rzeczywistego przyrostu przez `Delta z_i ≈ J_i Delta theta`, z osobną kontrolą buforów normalizacji. To przybliżenie pierwszego rzędu, nie dokładne równanie skończonego kroku. Atrybucja do próbek musi uwzględniać pamięć i preconditioning AdamW, weight decay, clipping oraz sprzężenia przez BN. Warunkowej dekompozycji nie utożsamiać z niezależnymi siłami par ani z kontrfaktycznym usunięciem próbki.

**Wynik pozytywny:** mechanizm wyznaczony na części danych przewiduje zmianę czasu zawracania, plateau lub przemieszczeń na odłożonych interwencjach. **Jeśli nie:** zostaje opis empiryczny, bez twierdzenia o mechanizmie. W obu przypadkach wynik może być informacyjny.

### 3. Generalizacja i ewentualne sterowanie

Plan startowy: co najmniej 3 niezależne inicjalizacje głównego warunku, sondy obejmujące różne klasy i filmy, oddzielne TRAIN/VALID oraz druga architektura lub wyraźnie inny przepis treningu. To praktyczny początek replikacji, nie gwarancja mocy statystycznej. Dopasować zakres do zmienności i kosztu pilotażu. Tokenów i sąsiednich klatek nie traktować jak niezależnych powtórzeń.

Regułę wyboru checkpointu lub learning rate ustalić na treningach rozwojowych, a potem zamrozić. Porównać z walidacyjnym early stopping, ReduceLROnPlateau, harmonogramem ustalonym z góry, normą aktualizacji oraz miarą podobieństwa reprezentacji. Jeśli reguła korzysta tylko z nieetykietowanych sond, osobno porównać metody o tym samym dostępie do informacji. Raportować mAP, czas/GPU, koszt sond i koszt strojenia. Przykładowy, wcześniej ustalony cel użytkowy: redukcja kosztu o 20% przy pogorszeniu mAP nie większym niż 0,005; próg wymaga dostosowania do zastosowania i zmienności, nie jest prognozą wyniku.

## Jak wykorzystać istniejące pokrętła zamiast je powtarzać

Istnieje już sześć wyników w `outputs/porownanie_szklo_20261010/eden/wyniki/pokretla/` i tabela `POKRETLA.json`: cztery learning rate oraz podzbiory 1500 i 5000 zdjęć. Odczytana tabela pokazuje m.in. wzrost wskaźnika Lindemanna S1 z około 0,175 do 0,426 między lr 0,000125 i 0,001. To użyteczny pilot zależności od przepisu treningu, nie samoistny dowód szkła. W tej ocenie nie wykonano pełnego audytu kodu i danych pokręteł.

Przed użyciem jako dowodu mechanistycznego:

- Zweryfikować wybrane wyniki w pełnych 768D. README opisuje zapis PCA32 i brak checkpointów po drodze; przyszły pilot powinien zachować pełne wybrane sondy oraz stan potrzebny do kontynuacji.
- Utrzymać te same sondy przy porównywaniu liczby danych. Aktualny opis podzbiorów zmienia także dobór sond.
- Rozdzielić liczbę aktualizacji od liczby ekspozycji i stopnia nauczenia: 3, 21 i 70 epok przy podobnej liczbie kroków nie oznaczają tego samego etapu uczenia.
- Nie utożsamiać learning rate z temperaturą termodynamiczną. AdamW zmienia dryf i statystyki szumu, zależne też od stanu i danych.
- Nie utożsamiać liczby wyjść/liczby parametrów z fizyczną gęstością. Najpierw sprawdzić definicję niezależnych ograniczeń, korelacje etykiet i faktyczną interpolację. Sama wartość ilorazu nie ustanawia fazy zatartej ani niezatartej.
- Czasy dochodzące do końca dostępnego okna sprawdzić pod kątem cenzurowania; nie dopasowywać prawa relaksacji do nieosiągniętych progów jak do zwykłych pomiarów.

## Jedno zdanie, które artykuł powinien ostatecznie uzasadnić

Wersja możliwa po replikacji obecnych wyników: **Dynamika tokenów ma różne skale czasu i odmienne tendencje lokalne oraz zbiorowe, przez co jedna miara ruchliwości nie opisuje stabilizacji reprezentacji.**

Wersja ambitniejsza, dopiero do sprawdzenia: **Po oddzieleniu ruchu wywołanego optymalizatorem i normalizacją możemy przewidywać przebudowę reprezentacji istotną dla zadania i wykorzystać tę wiedzę do skuteczniejszego treningu.**

To drugie zdanie określa potencjalnie duży wkład. Aby go napisać jako rezultat, potrzebne są interwencje, przewidywanie na nowych danych i korzyść przekraczająca dobrze dobrane metody odniesienia.
