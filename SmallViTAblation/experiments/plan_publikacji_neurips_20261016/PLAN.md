# Plan badań do publikacji: lokalna geometria i funkcja tokenów ViT

Data: 16 października 2026. Status: plan badawczy; w ramach tego opracowania nie uruchomiono nowych treningów ani zadań na Edenie. Liczby proponowanych prób są budżetem startowym, nie wymaganiami NeurIPS ani gwarancją akceptacji. Istniejące wyniki są eksploracyjne.

## 1. Rekomendacja i główna teza do sprawdzenia

Najlepszy obecnie kierunek to **mikroskop funkcjonalnej geometrii ViT: jak lokalne przekształcenia tokenów i współdziałanie neuronów odpowiadają za wrażliwość lokalizacji obiektu**. Pole MLP jest dokładną funkcją modelu, a jego rozciąganie i składowa obrotowa są mierzalne. Wkładem naukowym musi być jednak nowe ustalenie o funkcji tej geometrii, sprawdzone interwencjami i na niezależnych danych.

Proponowane pytanie główne:

> Czy podczas uczenia geometria odpowiedzi attention i MLP organizuje się tak, że tłumi zmiany nieistotne dla detekcji, a przekazuje zmiany istotne dla lokalizacji; czy możemy wskazać i przyczynowo sprawdzić, które składowe za to odpowiadają?

To **hipoteza**, nie opis wyniku już uzyskanego. Nie zakładamy, że więcej wirowania pomaga, mniej wirowania pomaga ani że siodło jest zawsze złe. Potrzebna może być orientacja względem odczytu, nie sama norma składowej.

Roboczy tytuł: *A Functional Microscope of Vision Transformers: Local Geometry, Neuron Cooperation, and Localization Sensitivity*. Ostateczny tytuł dopasować do wyniku.

Najbardziej konkretna ścieżka zapasowa, jeśli geometria pola nie wniesie dostatecznie dużo: **rozdzielenie kompensacji wpływu na współrzędne ramki od destabilizacji wyboru zwycięskiej detekcji**. Nie łączyć na siłę obu historii. Na dziś duży skok ramki pochodzi z grupy wzmacniającej, a nie z grupy kompensującej.

Samo wykrycie siodła, niezerowej antysymetrii, przeciwnych neuronów, 187 kierunków jądra głowy lub ładnej animacji nie jest wystarczającą tezą artykułu. Zwiększenie liczby checkpointów nie zastępuje niezależnego potwierdzenia ani wykazania znaczenia dla zadania.

## 2. Stan dowodów: co już mamy

| Ustalenie | Co rzeczywiście potwierdzono | Co pozostaje do zrobienia |
|---|---|---|
| Rozkład działania MLP | Podpisane wkłady 768 neuronów odtwarzają aktualizację residual 192D | Ujednolicić instrument dla sześciu bloków i uwagi, z biasami i dokładnymi normalizacjami |
| Przeciwne reflektory | W s47 końcowo 408/768 neuronów ma partnera o cosinusie poniżej −0,5; 844 pary | Replikacja; znaczenie funkcjonalne; to nie odsetek znoszącej się energii |
| Kompensacja w odczycie ramki | Dobierane lokalnie pary: mediana q około 0,084; kontrola około 0,701 | Przewaga na niezależnym materiale; taka selekcja daje kompensację również przy inicjalizacji |
| Współaktywacja i etykiety | W bbox oba neurony aktywne w 28–33% obserwacji token–para; ujemny bilans par większy niż w tle | Więcej filmów i seedów, powtarzane permutacje, dopasowanie siły i udziału obiektu |
| Rzekoma korekta po czasie | Brak silnego sygnału na rozdzielczości 100 kroków; część efektu różnicowania występuje po przetasowaniu stanów | Nie przedstawiać jako dowodu sekwencyjnego odpalenia neuronów |
| Zmiana zwycięskiej ramki | Jedna zweryfikowana ablacja grupy daje 7,09 px skoku i spadek IoU około 0,054 | Zmierzyć częstość i przewidywalność na pełnej, z góry ustalonej populacji |
| Ruch a jakość | Brak uniwersalnej sygnatury wysokiego IoU i pewności; są słabe, zależne od obrazu związki | Prospektywna walidacja ponad confidence, marginesem i czasem treningu |
| Dolina odpowiedzi | Lokalny zbiór stałych pięciu wyjść głowy ma wymiar 187 przy rzędzie 5; policzono konkretny wycinek | Znaczenie semantyczne i osiągalność przez obraz; nie jest to potencjał MLP |
| Siodło i obrót | Istnieją pomiary lokalne i projekcje; wcześniejszy audyt wykazał ograniczenia interpretacji | Pełne wymiary, właściwy residual, pełne pochodne attention, testy skończonych perturbacji |

Źródła lokalne: [współdziałanie](../../outputs/wspoldzialanie_neuronow_20261015/RAPORT.md), [przeciwne pary](../../outputs/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje/RAPORT.md), [ruch i jakość](../../outputs/wspoldzialanie_neuronow_20261015/ruch_a_jakosc/RAPORT.md), [geometria wag](../../outputs/wspoldzialanie_neuronow_20261015/przeciwne_reflektory/RAPORT.md), [dolina](../../outputs/dolina_odpowiedzi_t226_20261015/OPIS.md), [audyt matematyczny](../../outputs/research_fizyczny_mikroskop_20261015/math_audit.md).

Notatki wykonane dla tego planu: [matematyka](../../outputs/plan_publikacji_neurips_20261016/audyt_matematyczny.md), [interwencje](../../outputs/plan_publikacji_neurips_20261016/audyt_interwencji.md), [Eden i dane](../../outputs/plan_publikacji_neurips_20261016/eden_i_dane.md). Zawierają warianty budżetu; kolejność i zakres obowiązujące dla proponowanej kampanii są w niniejszym PLAN.md.

Wiele głównych wyników funkcjonalnych pochodzi z jednego treningu i trzech obrazów. Setki checkpointów, 600 tokenów i tysiące par nie są niezależnymi replikacjami. Istnieją szersze starsze pomiary dynamiki; trzeba je wykorzystać po sprawdzeniu definicji, nie przenosić automatycznie ich liczebności do nowych twierdzeń o polu i detekcji.

Istnieją również krótkie biegi z48/z49 i skan sześciu MLP. Zacząć od ich inwentaryzacji i ponownego użycia zgodnych artefaktów. Nie powtarzać zakończonych eksperymentów dla samej zmiany katalogu. Ponowny pomiar jest uzasadniony, jeśli pierwotny stosował przybliżony Jacobian, odwracanie GELU, niewłaściwy epsilon lub niewystarczającą precyzję. Plan fizycznych E1–E8 z 14 października był oznaczony jako niewykonany — nie zaliczać go do wyników.

## 3. Co oznacza poziom NeurIPS i granica nowości

Oficjalne kryteria oceniają poprawność, jasność, znaczenie i oryginalność. Nowy wgląd w istniejące modele może być wkładem; nie ma obowiązku zbudowania nowej architektury ani pobicia SOTA w każdym artykule. W naszym przypadku potrzebny jest wynik, z którego inni badacze rzeczywiście skorzystają. [NeurIPS 2026: kryteria](https://neurips.cc/Conferences/2026/ReviewerGuidelines).

Termin pełnych zgłoszeń do głównego NeurIPS 2026 był 6 maja 2026. Plan z października dotyczy kolejnej odpowiedniej edycji lub innej konferencji. Nie zakładamy nieogłoszonych terminów. [Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers).

Najbliższe prace wymagające jawnego porównania:

| Praca | Co już obejmuje | Jakiego rozróżnienia potrzebujemy |
|---|---|---|
| Im, Belghazi, Memisevic, 2016 | Zachowawczość pól autoenkoderów i symetria Jacobianu | Sama antysymetria J i rozkład pola są znane |
| Geshkovski i in., NeurIPS 2023 | Tokeny jako oddziałujące cząstki i dynamika czystego attention | Rzeczywiste różne bloki MLP/attention, trening i funkcjonalne interwencje |
| Castin, Ablin, Peyré, ICML 2024 | Lokalna czułość attention i rola normalizacji | Wyjaśnić co nowego daje struktura kierunkowa poza normą/czułością |
| Tomihari i Karakida, NeurIPS 2025 | Jacobiany, oscylacyjne składowe i jakość w rekurencyjnym attention | Autorzy wskazują brak MLP, positional encoding i masking; nasz model ma różne bloki i zadanie detekcji |
| Tercan i Meisen, 2025, preprint | Ablacje składowych DETR/DDETR/DINO i skutki klasyfikacji/regresji | Same ablacje detektora nie są nowym pomysłem |
| He, Zhang, Yan, 2026 | Rozkład kierunkowy aktualizacji attention/MLP i interwencje | Nasz S/A nie jest rozkładem równoległym/poprzecznym do tokenu; trzeba porównać przydatność obu |
| Kapil i Upla, TGO-I, 2026, preprint | Obserwatorium widmowej geometrii ViT podczas uczenia | Sam atlas ranków i przebiegów treningu nie wystarczy |
| Szymkowiak, Pałubicki, Adamczewski, 2026, extended abstract warsztatowy | Odejście od lokalnej liniowości przy zmianie skali perturbacji | Krzywe błędu Taylora są kontrolą/odniesieniem, nie automatycznie naszą nowością |
| Kim i in., 2022, preprint | Zakrzywienie reprezentacji ViT a perturbacje i odporność | Krzywizna, obrót lokalnego pola i odporność wymagają odrębnych definicji |

Celowany przegląd z 16 października obejmował SciSpace, Undermind, Scite i strony pierwotne. Nie stanowi dowodu pierwszeństwa. Przed ustaleniem tytułu i claimów przeczytać pełne teksty najbliższych konkurentów, odtworzyć wykonalne baseline'y i zapisać tabelę „to samo / różnica / sprawdzone eksperymentalnie”. Bibliografia na końcu.

## 4. Matematyka do ustalenia przed dużymi obliczeniami

### 4.1 Dwa zegary i dwie przestrzenie

Dla ustalonego checkpointu i bloku, w residual x∈R^192:

`F_M(x)=W2 GELU(W1 LN(x)+b1)+b2`, `T_M(x)=x+F_M(x)`.

Wektor GELU z∈R^768 jest innym miejscem pomiaru. Wkład `u_c=z_c W2[:,c]` leży w 192D. Obraz ma 600 tokenów. Jeden reflektor to jeden neuron, jego 600 wiązek odpowiada 600 różnym aktywacjom dla tokenów.

Dla pełnego stanu X∈R^(600×192) blok ma kolejność:

`X_a=X+F_A(X)`, `X_out=X_a+F_M(X_a)`.

Sześć bloków oznacza 12 różnych map. Z kolei w czasie treningu `x_t=f(I;θ_t,b_t)`; b obejmuje bufory BN. Nie ma tożsamości `x_(t+1)−x_t=F_M(x_t)`.

### 4.2 Rozkład lokalny i poprawne znaczenie siodła

`J=D F_M`, `S=(J+Jᵀ)/2`, `A=(J−Jᵀ)/2`.

S i A są rozkładem lokalnej pochodnej w określonej metryce. Mieszane znaki wartości własnych S nazwać **rozciąganiem i ściskaniem lokalnym**. Siodło punktu równowagi wymaga jeszcze znalezienia punktu F(x*)=0 i analizy pełnej dynamiki. Niezerowe A nie dowodzi zamkniętych orbit ani wiru Browna. Rozkład punktowy S/A nie jest globalnym rozkładem Helmholtza.

Dla rzeczywistego kroku residual liczymy `M=I+J`. W lokalnym przybliżeniu:

`||Mδ||²−||δ||² = 2 δᵀSδ + ||Jδ||²`.

Drugi składnik jest istotny przy kroku równym 1. Ujemny kierunek S sam nie gwarantuje skrócenia skończonego kroku. Liczyć wartości singularne M, nie wyciągać wniosku o kurczeniu tylko ze śladu J albo znaku S. W wielu blokach potrzebny jest iloczyn pełnych pochodnych po rzeczywistej ścieżce; iloczyn projekcji 3×3 nie jest tym samym.

### 4.3 Normalizacja i wybór współrzędnych

Pre-LN daje dokładnie `J·1=0`, a residual `M·1=1`. Kierunek dodania stałej do wszystkich współrzędnych tokenu jest neutralny. Nie ma ścisłego przyciągania całego 192D przez taki MLP. Przy eps>0 niezmienniczość skali jest tylko przybliżona.

Do głównej analizy użyć także `Q=I−11ᵀ/192`, `Jbar=QJQ` ograniczonego do 191-wymiarowej podprzestrzeni `1⊥`, oraz jego Sbar/Abar. Ten wariant usuwa dokładną symetrię przesunięcia. Zachować równolegle surowy pomiar, aby określić, ile efektu było związane z normalizacją.

Nie mylić Q, które centruje współrzędne **wewnątrz tokenu**, z odejmowaniem średniej **po wszystkich tokenach** w wizualizacji. Do obliczeń widmowych można użyć ortonormalnej bazy U tej podprzestrzeni i macierzy `UᵀJU` o rozmiarze 191×191. Neutralny kierunek radialny nie jest dokładną symetrią przy eps>0.

Wspólna ortogonalna zmiana bazy zachowuje odpowiednie normy i widma; dowolne skalowanie osi już nie. Sprawdzić permutacje, ortogonalny przepis współrzędnych dla zmierzonej funkcji oraz jawnie deklarowaną normalizację. Nie zakładać, że dowolny obrót wag jest symetrią architektury z LayerNorm. Nie wyrównywać każdego obrazu osobno; ewentualne wyrównanie checkpointów dopasować na osobnych sondach.

### 4.4 Co trzeba wykazać analitycznie

- [ ] Dokładną rekonstrukcję residual przez wkłady, z biasami, LayerScale i trybem eval.
- [ ] Dokładny Jacobian MLP z pochodną prawdziwego LayerNorm i GELU.
- [ ] Regułę skończonego wzmocnienia przez I+J oraz regułę kompozycji 12 map.
- [ ] Warunki i granice lokalnej interpretacji S/A, w tym kontrprzykład ścinania z A≠0 bez zamkniętych orbit.
- [ ] Zachowanie symetrii LN przez proponowane interwencje; brak efektu kontrolnej interwencji w punkcie bazowym.
- [ ] Granicę błędu lokalnego przybliżenia przy założeniu ograniczonej zmienności J, z empirycznie sprawdzonym zakresem promieni.

Są to podstawy poprawności metody, przeważnie standardowa matematyka. Nie reklamować ich jako nowych twierdzeń bez rzeczywiście nowego rezultatu.

## 5. Dane, rozdziały i jednostka niezależności

Audyt lokalnych manifestów potwierdza 117 filmów TRAIN, 25 VAL i 25 TEST. Wybrane 35 000 klatek treningowych pochodzi z TRAIN; dotychczasowe 3000 klatek walidacji obejmuje 25 filmów VAL. Manifest nie zapewnia patient-disjointness. Nie traktować tego automatycznie jako udowodnionego przecieku; zweryfikować identyfikatory pacjentów/zabiegów i pochodzenie klatek.

- [ ] Sprawdzić wszystkie trzy dotychczasowe ilustracje: train/val/test, film, timestamp, czy były wybierane na podstawie wyniku.
- [ ] Zweryfikować brak duplikatów i sąsiednich klatek tego samego zabiegu między podziałami oraz historię wcześniejszego użycia TEST.
- [ ] Gdy TEST był już używany do strojenia, wydzielić nowy odłożony zestaw lub zewnętrzny benchmark; nie przemianowywać go na ślepy test.
- [ ] VAL używać do rozwoju; zamrozić specyfikację przed TEST. W obrębie VAL stosować podziały po filmach do doboru i sprawdzania selektorów.
- [ ] Start pilota: 24 klatki z 12 filmów VAL, po dwie odległe w czasie; wybór z manifestu przed oglądaniem metryk.
- [ ] Atlas rozwojowy: 250 klatek, po 10 z każdego z 25 filmów VAL. Test atlasu: analogiczne 250 z 25 odłożonych filmów, jeśli przejdą audyt. Nie utożsamiać tej próbki z pełnym testem mAP.
- [ ] Raportować wynik detektora na pełnym zatwierdzonym zbiorze TEST. Przy ograniczeniu kosztu użyć z góry utrwalonej próbki i jawnie określić zakres.

Podział tokenów: środki wewnątrz bbox, granica bbox, dalekie tło; uwzględnić udział przecięcia patcha z bbox i cienkie narzędzia. Bbox nie jest maską segmentacji. Token po attention zawiera kontekst. Do głównej tezy „specyficzne dla narzędzia” dołączyć istniejące maski albo mały ręcznie zweryfikowany zbiór masek; jeśli ich nie ma, ograniczyć twierdzenie do regionu bbox.

Nie traktować położenia patcha jako etykiety znaczenia konkretnego neuronu. Mierzyć reakcje na treść obrazu i kontrolowane zmiany, a nie tylko na pozycję w siatce.

## 6. Kolejność eksperymentów na Edenie

### E0 — audyt instrumentu i benchmark kosztu; obowiązkowy początek

Wejście: istniejący s47, checkpointy 0, 33000, 65640; 4 obrazy; 12 deterministycznie wybranych tokenów na obraz; wszystkie sześć bloków. Bez nowego treningu.

- [ ] Odtworzyć 13 stanów residual: wejście i wynik każdego attention/MLP. Sprawdzić zgodność hooków, rekonstrukcji oraz końcowej głowy i dekodera.
- [ ] Zweryfikować analityczny J MLP przez autodiff i centralne różnice w float64 na kopii modułu; przejście rzeczywistego modelu sprawdzić w float32.
- [ ] Dla attention rozróżnić pole query-only z zamrożonym K/V, pełny blok `∂F_i/∂x_i` i wpływy `∂F_i/∂x_j`. Pełny JVP musi uwzględniać Q, K, V, softmax, LN i wszystkie tokeny.
- [ ] Testy: pole gradientowe, czysty obrót, ścinanie, znane liniowe residual oraz zachowanie przesunięcia LN. Te testy sprawdzają instrument, nie uczony mechanizm.
- [ ] Porównać JVP/VJP w oryginalnej implementacji uwagi i w wersji jawnej, gdy fused kernel nie wspiera pochodnych.
- [ ] Zmierzyć czas forward, JVP, VJP, pełnego 192×192 J, SVD i interwencji; peak VRAM/RAM, przepustowość i zapis. Wykonać rozgrzewkę i co najmniej trzy powtórzenia pomiaru czasu.

Brama E0: błędy rekonstrukcji i testów muszą być znacznie mniejsze od analizowanego efektu; początkowo cel względny 1e−5 float32 i 1e−7 dla dobrze uwarunkowanych testów pochodnej float64. Używać także tolerancji bezwzględnej przy wartościach bliskich zeru i sprawdzać zakres kroku różnicowego. Jeśli tolerancja nie przechodzi, diagnozować, nie dostrajać jej do oczekiwanego wyniku.

Wynik: `E0_audyt.json`, raport kosztu i decyzja o liczbie równoległych zadań. Obecny opis aplikacji nie zastępuje tego testu.

### E1 — atlas pełnych wymiarów zamiast pojedynczego obrazka pola

Pilot: 7 checkpointów `[0,1000,5000,15000,33000,55000,65640]`, 24 obrazy, 6 bloków. Dla wszystkich 600 tokenów zapisać tanie miary i wkłady; pełny J MLP dla 12 tokenów na obraz, wybranych przed pomiarem. To 12 096 token–blok–checkpoint do pełnego Jacobianu w pilocie.

Pomiary: normy F/J/S/A, udział obrotowy `||A||F²/(||S||F²+||A||F²)`, widmo S, singular values I+J, analogiczny wariant Q, signed cancellation, rząd efektywny, geometria wpływów na odczyt. Dla uwagi stosować JVP/VJP całego stanu i wybrane bloki sprzężeń; nie tworzyć pełnej macierzy 115200×115200.

Dla skali perturbacji `ε/||Qx||` zbadać `[1e−4,3e−4,1e−3,3e−3,1e−2,3e−2]`, o ile stan ma dostateczną normę. Kierunki: losowe, wybrane własne Sbar, singularne I+Jbar oraz rzeczywiste różnice z perturbacji obrazów. Ustalić, przy jakiej skali lokalny model przestaje być trafny. Porównać z inicjalizacją i dokładnie tak samo próbkowanymi kontrolami.

Wersja rozszerzona: 250 klatek, 12 checkpointów ustalonych przed analizą, 6 bloków, 24 tokeny na obraz do J MLP. Pozostałe tokeny przez tanie agregaty. Pełne skany wszystkich 600 tokenów tylko na małej, reprezentatywnej podpróbie. Nie potrzebujemy mapować całej nieograniczonej przestrzeni latentnej.

Proponowana siatka 12 stanów s47: `[0,300,1000,3000,6000,10000,16000,24000,33000,44000,55000,65640]`; przed uruchomieniem potwierdzić dostępność w manifeście. W nowych biegach zapisać te same liczby aktualizacji oraz metadane liczby ekspozycji danych. Brak konkretnego stanu rozwiązać jawną regułą najbliższego dostępnego checkpointu, bez doboru według jakości.

Brama E1: istotna interpretacja nie może znikać po przejściu z rzutu 3D do pełnego wymiaru, po właściwym residual lub kontroli LN. Jeżeli znika, opisać ograniczenie narzędzia i nie budować dalszej tezy na obrazie wiru.

### E2 — czy geometria ma określoną funkcję dla ramki?

To najważniejszy eksperyment. Zamrożony checkpoint, obraz i pełny stan tokenów. Zaburzenie podajemy przed wybranym MLP, a następnie liczymy wszystkie pozostałe bloki i prawdziwy dekoder.

Zamrożony punkt bazowy x0, `δ=x−x0`. Dla wariantu z kontrolą symetrii LN:

`F_minus_A(x)=F(x)−α Abar(x0) Qδ`.

Analogicznie testować usunięcie Sbar albo jego dodatniej/ujemnej części. `α∈{0,0.5,1}`; jedna interwencja w jednym bloku na raz. W punkcie x0 wszystkie warianty są identyczne. To celowa zmiana **odpowiedzi na perturbację**, nie usunięcie globalnego wiru ani poprawa niezmienionego obrazu. Nie wykonywać tylko próby δ=0, bo wynik zerowy jest wtedy wymuszony konstrukcją.

Kontrole: losowa orientacja operatora w `1⊥` o tym samym widmie/normie; kontrola o tej samej normie faktycznie zastosowanej korekty; równoległa/poprzeczna składowa względem tokenu; kierunki singularne; normmatched perturbacje. Usunięcie A zmienia też normę całego J, więc porównanie tylko „A włączone/wyłączone” jest niewystarczające.

Naturalne zaburzenia: małe translacje obrazu z przeliczeniem GT, skala z przeliczeniem GT, jasność/kontrast bez zmiany GT, rozmycie oraz zasłonięcie wybranego obszaru. Lokalne edycje tła/narzędzia wymagają maski lub jawnego ograniczenia do bbox; kontrolować krawędzie edycji. Nie zakładać, że każda transformacja jest semantycznie neutralna. Kierunki z rzeczywistych obrazów testują osiągalne zmiany; przypadkowe δ są testem matematycznym poza manifoldem danych.

Przy naturalnej zmianie obrazu zmienia się wiele tokenów. Testować zarówno interwencję pojedynczego tokenu, jak i maskowanej grupy przy pełnym, zmienionym stanie reszty. Dla wcześniejszych bloków nie wolno użyć skrótu ostatniej głowy bez ponownego przejścia downstream.

Mierzyć: błąd czterech krawędzi w pikselach i względem rozmiaru obiektu, IoU, zmianę logitu, zachowanie poprawnej detekcji, częstość switcha top1 i wynik pełnego dekodera. Pokazać krzywe po ε, nie tylko dogodny punkt. Osobno stabilność tej samej ramki i skok przy zmianie zwycięzcy.

Jeżeli `B=D(head_downstream)` przy bazowym stanie po MLP, liniowa prognoza odpowiedzi to `B(I+J)δ`. Rozłożyć ją na `Bδ`, `BSδ`, `BAδ`, zachowując człony krzyżowe norm. To pokaże orientację geometrii względem zadania. **Pełny JVP zawiera tę samą informację; S/A nie może wygrać dzięki rzekomo dodatkowej informacji.** Sens porównania to interpretowalność, stabilność selektora lub korzystny koszt przybliżenia.

Powyższy zapis używa pełnego J/S/A. W analizie ilorazowej konsekwentnie stosować `BU`, `UᵀJU` i `Uᵀδ`; nie mieszać surowego S/A z ilorazowym. Dla wczesnego tokenu B jest pochodną całej pozostałej sieci względem tego tokenu w pełnym kontekście, nie prostym przyłożeniem ostatniej głowy do wczesnego stanu.

Główny kontrast potwierdzający: różnica błędu lokalizacji dla interwencji ukierunkowanej względem korekty losowej o dopasowanej sile, sparowana po obrazie, skali i seedzie. Kierunek hipotezy i praktyczny minimalny efekt wybrać na pilocie i zamrozić. Bez hipotezy kierunkowej użyć testu dwustronnego i opisywać zmianę, a nie automatycznie korzyść. Osobno pokazać zachowanie poprawnej odpowiedzi na translację i odporność na zmianę jasności.

Brama E2: efekt kierunkowy powtarza się na odłożonych filmach i naturalnych perturbacjach, przekracza normmatched sham i błąd numeryczny, ma zakres skali i nie sprowadza się do dowolnego uszkodzenia sieci. Jeżeli znika poza arbitralnymi δ, ograniczyć claim do lokalnej funkcji, nie do odporności wizji.

### E3 — co zmienia się podczas treningu: pole czy punkt pomiaru?

Dla par checkpointów policzyć cztery kombinacje `(θ_t,x_t)`, `(θ_t,x_(t+1))`, `(θ_(t+1),x_t)`, `(θ_(t+1),x_(t+1))`. Dla F oraz J rozdzielić zmianę wag, zmianę wejścia i resztę interakcyjną. Normy części nie muszą się sumować. Mieszane kombinacje mogą być poza rozkładem; ich funkcją jest lokalna dekompozycja, nie nowy działający model.

Osobno badać faktyczny ruch reprezentacji między checkpointami przez `Dθ f Δθ + Db f Δb` z kontrolą reszty. Używać rzeczywistych różnic wag i buforów; dla identyfikacji optymalizatora potrzebny pełny stan AdamW, losowania danych i RNG. Finalnych wag s47 nie traktować jako zapisu całej historii optymalizatora.

Krótkie sparowane kontynuacje od kompletnego stanu: kontrola, zmiana learning rate, zmiana pamięci optymalizatora, kontrola BN — po jednej zmianie. Dopasować lub raportować normy aktualizacji. Nie utożsamiać beta1=0 w AdamW z SGD. Zachować wspólną kolejność minibatchy i oddzielić stan przejściowy po interwencji.

Budżet początkowy: 3 punkty treningu × 3 kompletne stany seedów × 3 warunki × 300 kroków = 8100 kroków kontynuacji. Konkretne trzy warunki wybrać z pilota i zamrozić przed replikacją; nie testować wszystkich kombinacji bez hipotezy. Ten etap potrzebny do tezy o przyczynie ruchu w uczeniu; nie jest warunkiem samego artykułu o inferencji.

### E4 — współdziałanie neuronów i przełączenia detekcji

Rozszerzyć istniejące ablacje, nie powtarzać ich na tych samych trzech zdjęciach. Pary i grupy 2/8/16/32 wybierać na części rozwojowej; sprawdzać na innych filmach, checkpointach i seedach. Między seedami porównywać algorytm wyboru lub jawnie wyrównane podprzestrzenie, nie numer neuronu 760.

Dwie osobne hipotezy:

1. Kompensacja podpisanych wkładów jest zależna od treści obrazu i zmienia się przez trening w sposób wykraczający poza geometrię inicjalizacji, normy i aktywność marginalną.
2. Grupy o małym wpływie na współrzędne danej ramki mogą mieć duży wpływ na margines konkurencji kandydatów. Obecny przykład dużego switcha jeszcze tego nie dowodzi.

Kontrole: initialization, random labels na małym budżecie, normy u, gradient×activation, `J_b u`, gradient marginesu, pojedyncze ablacje, losowe grupy dopasowane do siły, permutacje aktywacji w obrębie grup przestrzennych. Używać wielu permutacji; obecna jedna permutacja była kontrolą opisową.

Do wyzerowań A/B/AB dodać test odwrócenia błędu: w zaburzonym obrazie przywrócić aktywacje wybranej grupy z czystego przebiegu i sprawdzić odzysk marginesu oraz IoU, z kontrolami dopasowanymi do liczby neuronów i normy poprawki. Wykonać też kierunek odwrotny. Dla translacji jawnie wyrównać tokeny; patching może tworzyć stan spoza rozkładu i sam nie dowodzi przyczyny całego treningu.

Dla winnera i runner-up mierzyć `m=logit_top1−logit_top2` oraz przewidywaną zmianę m. Sprawdzić wszystkich kandydatów mogących przejąć maksimum, bo po perturbacji top2 nie zawsze wygrywa. Dla danego tokenu raportować wynik fixed-token niezależnie od wyniku finalnego dekodera. Nie nazywać skoku argmax nieliniową synergią MLP.

Brama E4: przy zamrożonym selektorze wynik utrzymuje się poza danymi selekcji albo metoda przewiduje odpowiedzi/failures lepiej niż silny baseline o tym samym dostępie do informacji. Samo odszukanie przeciwnych wektorów w 4D i późniejsze małe q nie spełnia tej bramy.

### E5 — czy mikroskop pomaga przewidywać błędne ramki?

Zadanie użytkowe ustalić przed TEST: np. ryzyko `IoU<0,5` lub spadku IoU o co najmniej 0,1 przy zadanej perturbacji. GT służy do etykietowania i oceny; predyktor wdrożeniowy nie dostaje GT ani tokenów wskazanych na jego podstawie.

Baseline'y: confidence, margines top1–top2, rozbieżność ramek top-k, normy tokenów i aktualizacji, czułość gradientowa/JVP, czas treningu; opcjonalnie zgodność pod augmentacjami przy równym budżecie obliczeń. Porównanie nested: baseline, baseline+geometria, baseline+współdziałanie, baseline+oba.

Miary: AUPRC z częstością zdarzeń, AUROC pomocniczo, Brier/calibration, risk–coverage oraz koszt GPU na obraz. Prognoza zmian między checkpointami używa tylko przeszłości: bez średniej ±300, PCA dopasowanej na przyszłości, końcowego położenia celu i finalnych par wybieranych retrospektywnie.

Wstępny praktyczny cel do prerejestracji po pilocie: wzrost AUPRC o co najmniej 0,03 ponad najlepszy tani baseline **albo** redukcja ryzyka przy tej samej coverage o co najmniej 10% względnie. To propozycje progów użyteczności, nie przewidywany wynik. Wybrać jedną miarę główną i próg przed testem; nie ogłaszać sukcesu, wybierając później korzystniejszą alternatywę.

Ta ścieżka nie jest obowiązkowa, jeśli uzyskamy silne nowe wyjaśnienie mechanistyczne. Jeśli jednak reklamujemy „wykrywanie wysokiego IoU z ruchu”, bez takiej oceny twierdzenie pozostaje niepotwierdzone.

### E6 — replikacja i zakres uogólnienia

Minimalny etap rozwojowy: wykorzystać s47 jako discovery oraz dwa dodatkowe niezależne pełne biegi, jeśli porównywalne kompletne checkpointy jeszcze nie istnieją. Po zamrożeniu metody wykonać trzy nowe biegi potwierdzające. Docelowo pięć nowych seedów pozwala oddzielić dwa rozwojowe od trzech potwierdzających; liczbę zwiększyć, jeżeli wariancja pilota nie daje rozstrzygającej precyzji. To plan praktyczny, nie magiczna liczba wymaganych seedów.

Rozszerzenie po przejściu E2 lub E4: druga konfiguracja ViT o wyraźnie innej szerokości/głębokości oraz drugi zbiór z etykietami lokalizacji. Preferowany publiczny benchmark detekcji, np. odpowiednio zdefiniowane podzadanie COCO; najpierw ocenić zgodność wieloobiektowości z aktualną głową. Nie porównywać naiwnie single-object top1 z pełnym wieloobiektowym detektorem. Alternatywą jest odrębny detektor transformerowy i adaptacja miernika do jego odczytu. Konkretny benchmark i preprocessing zamrozić przed wynikami.

Imagenette może potwierdzić geometrię na innym zadaniu, ale nie zastąpi replikacji twierdzenia o IoU. CNN może być kontrolą mechanizmu attention, nie obowiązkowym dodatkowym artykułem. Nie uruchamiać pełnej siatki architektury×dane×optymalizatory bez wyniku z pilota.

## 7. Statystyka, która pozwoli obronić twierdzenia

- [ ] Jednostki replikacji: niezależny trening i niezależny film/zabieg; tokeny, klatki i checkpointy to pomiary zagnieżdżone.
- [ ] Raportować sparowane efekty na tych samych obrazach i perturbacjach oraz osobne wyniki każdego seeda. Używać hierarchicznego bootstrapu/modele mieszane odpowiednie do układu, z klastrowaniem po filmie i seedzie.
- [ ] Dla czasu stosować bloki lub bootstrap całych trajektorii i jawne odstępy kroków. Pokazać wariant bez detrendingu, z kontrolą czasu i z przyrostami; detrending nie jest dowodem przyczynowości.
- [ ] Oddzielić trzy główne testy: geometria→odpowiedź na perturbację, organizacja w uczeniu, użyteczność prognozy. Kontrolować wielokrotne porównania warstw i miar; wyniki dodatkowe oznaczać eksploracyjnie.
- [ ] Podawać efekty i przedziały, nie samo p. Brak istotności nie oznacza równoważności; dla tezy „A nie ma znaczenia” potrzebny wcześniej ustalony margines równoważności i dostateczna moc.
- [ ] Estymować precyzję i minimalny wykrywalny efekt z pilota. Gdy przedziały za szerokie, zwiększać liczbę filmów/seedów, a nie mnożyć sąsiednie klatki.
- [ ] Ustalić zasady brakujących tokenów, pustych bbox, małych mianowników q i niedopasowanych kontroli przed testem. Nie usuwać źle działających checkpointów.

## 8. Budżet i organizacja Edena

Najpierw E0. Historyczny s47 na H100: 21 epok około 728 s, kontynuacja 39 epok około 1374 s; osobny gęsty obserwator 21 epok około 1222 s. Są to **czasy historyczne innych obciążeń**. Nie wolno z nich obiecać czasu dla Jacobianów i wszystkich sześciu bloków. Aktualna dostępność GPU, kwota i wolny dysk nie były sprawdzane na żywo w tym planie.

| Etap | Zakres startowy | Warunek uruchomienia następnego |
|---|---|---|
| 0 | Audyt danych, E0 i benchmark | Rekonstrukcja i pochodne zgodne |
| 1 | s47, E1 pilot + E2 pilot na 24 nowych klatkach VAL | Pełnowymiarowy, funkcjonalny efekt poza kontrolami |
| 2 | E3/E4 na istniejących wagach i atlas 250 klatek | Zamrożona główna hipoteza i metoda |
| 3 | Dwa nowe biegi rozwojowe, następnie trzy potwierdzające | Precyzja efektów i brak zależności od pojedynczego seeda |
| 4 | E5 i odłożony test | Niezależna ocena claimów, bez dostrajania |
| 5 | Drugi model / drugi zbiór | Rozszerzenie zakresu tylko dla utrzymanego wyniku |

Formuła budżetu z E0: `czas ≈ liczba_forward*t_forward + liczba_JVP*t_JVP + liczba_VJP*t_VJP + liczba_J_MLP*t_J_MLP + trening + IO`. Raportować GPU-godziny i ścienny czas osobno. Dla attention liczyć ograniczoną liczbę JVP/VJP i losowych sond; np. zacząć od 16 kierunków, sprawdzić zbieżność 8/16/32 na podzbiorze. Sondy pełnego stanu i pojedynczych tokenów odpowiadają różnym pytaniom.

Koszt surowych tensorów, bez narzutów i kompresji:

- Jeden pełny Jacobian attention 115200×115200 w float32: **53,1 GB**, przed pamięcią autograd. Nie materializować.
- Pełne J MLP dla 12 checkpointów × 250 obrazów × 6 bloków × 24 tokeny: około **63,7 GB na seed**. Liczyć strumieniowo i przechowywać głównie statystyki, wybrane operatory i identyfikatory pozwalające odtworzyć wynik.
- Trzynaście stanów residual dla 12 checkpointów × 250 obrazów × 600 tokenów × 192 współrzędne w float32: około **18,0 GB na seed**. Zwykle odtwarzać z checkpointów partiami; pełne stany zachować dla niewielkiego zestawu reprodukcyjnego.

Zakresy nowego eksperymentu lokalnie: `experiments/plan_publikacji_neurips_20261016/` (plan) oraz przyszłe `experiments/publikacja_geometria_vit/` (kod). Wyniki w przyszłym `outputs/publikacja_geometria_vit/`. Na Edenie nowy, odrębny katalog biegu; wykorzystać istniejący wzorzec `~/smallvit_runs/`, ale sprawdzić konfigurację i uprawnienia przed wysłaniem.

Każdy shard: seed, checkpoint, zbiór, zakres obrazów, blok, wersja miernika; zapis atomowy, manifest SHA256, log środowiska, identyfikatory danych, parametry próbkowania. Wznawianie pomija wyłącznie kompletne, zgodne shardy. Liczyć analizy na Edenie, lokalnie pobierać raporty i wybrane artefakty. Nie usuwać historycznych danych w celu zrobienia miejsca.

Przyszły kod dzielić na: `audit_forward.py`, `audit_derivatives.py`, `scan_fields.py`, `intervene_response.py`, `cooperation_and_switches.py`, `decompose_training_change.py`, `evaluate_failure_prediction.py`, `aggregate_results.py` oraz konfiguracje z utrwalonymi podziałami. To planowane pliki, jeszcze nie gotowe komendy.

## 9. Co będzie wystarczającym pakietem dowodów

**Do głównej tezy o mechanizmie:** poprawny instrument + spójny efekt E2 na perturbacjach obrazów + replikacja treningów + dowód, że efekt zależy od wyuczonej orientacji, a nie tylko normy czy arbitralnej projekcji + jasne odróżnienie od bliskich prac.

**Do tezy o uczeniu mechanizmu:** dodatkowo E3, porównanie inicjalizacji i kontroli zadania oraz prospektywny test na nowych treningach. Atlas checkpointów sam pokazuje zmianę, nie przyczynę.

**Do tezy o praktycznej diagnostyce:** E5 z przewagą nad silnymi baseline'ami i policzonym narzutem. Nie trzeba jednocześnie ulepszać architektury i treningu.

**Do tezy o optycznej/Brownowskiej pułapce:** potrzebna osobna identyfikacja dynamiki stochastycznej, dryfu, pamięci i dyfuzji oraz odpowiedzi na wymuszenie. Plan z 14 października zawiera propozycje takich badań, lecz nie mogą one automatycznie zastąpić powyższych testów. Nie są priorytetem tego artykułu.

Jeśli po E2/E4 wszystkie efekty wyjaśnia zwykła lokalna liniowość, margines argmax i normy, wstrzymać duże treningi. Mamy wtedy użyteczny instrument i wynik ograniczający interpretację, ale nie dowód nowego mechanizmu. Nie obiecywać NeurIPS na podstawie samej liczby eksperymentów.

## 10. Plan artykułu i materiał reprodukcyjny

1. Jedno pytanie i jedna główna odpowiedź, z zakresem twierdzenia.
2. Definicja 12 map, dwóch zegarów, 192D/768D i mierzonego obiektu.
3. Walidacja instrumentu i kontrola normalizacji/projekcji.
4. Główny wynik interwencji z przedziałami i normmatched kontrolami.
5. Rozwój w treningu oraz replikacja na seedach/filmach.
6. Związek z ramką i rankingiem kandydatów; ewentualna diagnostyka.
7. Granice, wyniki ujemne, koszt i dostępność materiałów.

Główne figury: schemat prawdziwych obliczeń; atlas warstwa×trening; krzywe interwencji po skali; wpływ na krawędzie i ranking; transfer między seedami/danymi; opcjonalnie risk–coverage. Film i GUI jako demonstracja metody, nie jedyny dowód.

- [ ] Udostępnić kod odtworzenia liczb, środowisko, konfiguracje, seedy, manifesty i przykładowe zgodne checkpointy.
- [ ] Ująć negatywne wyniki: słaby związek samego ruchu z IoU, kompensacja przy inicjalizacji, brak silnego opóźnienia na obecnej rozdzielczości, brak dotychczasowej przewagi rankingu grup.
- [ ] Sprawdzić licencje i zasady dostępu do danych chirurgicznych; jeśli nie można ich udostępnić, dostarczyć publiczną replikację i skrypt przygotowania danych dostępnych uprawnionym badaczom.
- [ ] Wypełnić checklistę aktualnej docelowej konferencji: zakres claimów, odtwarzalność, niepewność, zasoby, ograniczenia i wpływ zastosowania. [NeurIPS handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).

Kierunek NeurIPS/ICLR/ICML oceniać po głównym wyniku mechanistycznym; CVPR/ECCV jest naturalnym kandydatem, jeśli wkład okaże się przede wszystkim o lokalizacji i detektorach. Przy węższym, wstępnym wyniku rozważyć odpowiedni warsztat, a nie rozszerzać claimów. To ocena dopasowania tematycznego, nie prognoza decyzji recenzentów.

## 11. Pierwsze pięć zadań do wykonania

- [ ] **1. Zamrozić definicje i audyt danych.** Zidentyfikować punkty pomiaru, seedy, filmy i dostępne pełne stany; rozstrzygnąć historię TEST.
- [ ] **2. Zbudować i sprawdzić E0.** Pełny residual, pełne pochodne attention, prawdziwe LN, testy znanych pól i benchmark Edena.
- [ ] **3. E1+E2 na s47 i 24 nowych klatkach VAL.** Odpowiedzieć, czy orientacja S/A ma skutek funkcjonalny poza zmianą normy i projekcją.
- [ ] **4. E4 na niezależnych filmach.** Sprawdzić przenoszenie kompensacji i wyjaśnienie switchów ponad margines/JVP.
- [ ] **5. Decyzja po pilocie.** Wybrać główną historię, ustalić progi i prerejestrację; dopiero wtedy uruchomić serię nowych seedów i zewnętrzną replikację.

## Bibliografia i podstawa przeglądu

- Im, D. J., Belghazi, M. I., & Memisevic, R. (2016). *Conservativeness of Untied Auto-Encoders*. AAAI. [DOI](https://doi.org/10.1609/aaai.v30i1.10268). Scite zwrócił fragment: „an auto-encoder defines a conservative vector field if and only if its Jacobian is symmetric for all x ∈ K”; warunki dziedziny trzeba zachować z twierdzenia. W odpowiedzi brak pola editorialNotices, więc nie traktujemy tego jako certyfikatu braku korekt.
- Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2023). *The emergence of clusters in self-attention dynamics*. NeurIPS. [Publikacja](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html).
- Castin, V., Ablin, P., & Peyré, G. (2024). *How Smooth Is Attention?* ICML. [Publikacja](https://proceedings.mlr.press/v235/castin24a.html).
- Tomihari, A., & Karakida, R. (2025). *Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians*. NeurIPS 2025 według wersji autorskiej. [Pełny tekst](https://arxiv.org/html/2505.19458v4). Sprawdzono abstrakt i ograniczenia modelu w pełnym HTML.
- Tercan, H., & Meisen, T. (2025). *Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations*. Preprint. [ArXiv](https://arxiv.org/abs/2507.21723). W tej aktualizacji sprawdzono abstrakt.
- He, S., Zhang, H., & Yan, S. (2026). *Disentangling Representation Evolution in Transformers through Directional Decomposition*. Findings of EMNLP 2026 według strony autorskiej. [ArXiv](https://arxiv.org/abs/2609.15975). Sprawdzono metadane i abstrakt; potrzebne pełne porównanie.
- Kapil, K., & Upla, K. P. (2026). *Transformer Geometry Observatory TGO-I: Spectral Geometry Observatory*. Preprint. [ArXiv](https://arxiv.org/abs/2606.19249). Metadane i abstrakt.
- Szymkowiak, J., Pałubicki, W., & Adamczewski, K. (2026). *Beyond Local Linearity: Scale-Resolved Geometry of Learned Image Encoders*. Extended abstract, NeurReps 2026 według strony autorskiej. [ArXiv](https://arxiv.org/abs/2609.39115). Metadane i abstrakt; nie utożsamiamy warsztatu z głównym NeurIPS.
- Kim, J., Park, J., Kim, S., & Lee, J. (2022). *Curved Representation Space of Vision Transformers*. Preprint. [ArXiv](https://arxiv.org/abs/2210.05742). Metadane i abstrakt.
- Adebayo, J., i in. (2018). *Sanity Checks for Saliency Maps*. NeurIPS. [Publikacja](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). Uzasadnienie kontroli modeli i etykiet; nie dowód, że nasze pole jest saliency mapą.

Wcześniejszy szerszy przegląd: [synteza z 15 października](../../outputs/research_fizyczny_mikroskop_20261015/SYNTEZA_I_PROPOZYCJA_PUBLIKACJI.md). Wcześniejsza strategia dynamiki: [plan z 10 października](../strategia_artykulu_20261010/PLAN_ARTYKULU.md). Nowy plan koncentruje się na znaczeniu funkcjonalnym pola; nie uruchamia automatycznie wszystkich starszych propozycji fizycznych.
