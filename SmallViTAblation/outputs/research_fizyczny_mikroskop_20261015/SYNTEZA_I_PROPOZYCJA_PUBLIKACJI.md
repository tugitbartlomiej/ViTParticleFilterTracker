# Czym jest mikroskop SmallViT i gdzie jest zagadnienie publikacyjne

Stan wiedzy i audytu: 15 października 2026. Synteza Undermind, SciSpace, Scite oraz odczytu lokalnego kodu i wyników. Autorstwo ocen i proponowanych hipotez: analiza w tej sesji. Liczby eksperymentalne przytoczono z wcześniejszych pomiarów użytkownika; nie wykonano nowego treningu ani ponownego pomiaru checkpointów.

**Ocena:** układ można rygorystycznie badać metodami fizyki i dynamiki. Najtrafniejsza nazwa to **wirtualny mikroskop transportu reprezentacji w uczącym się Vision Transformerze**. W głębokości sieci obserwuje się dyskretny układ sprzężonych stanów tokenów, sterowany polami uwagi i MLP. Podczas uczenia obserwuje się zmianę tych pól i reprezentacji pod wpływem wspólnego procesu optymalizacji. Optyczne reflektory są wizualnym kodowaniem rzeczywistych składników obliczeń; nie ustanawiają samodzielnie praw optyki, temperatury ani pułapki Browna.

Najsilniejszy obecnie kierunek artykułu: **jak uczenie organizuje przeciwstawne wkłady neuronów w transport reprezentacji użyteczny dla zadania**. Kluczowe jest przejście od obserwowanej geometrii do przewidywanej i sprawdzonej odpowiedzi na interwencję. Sam film 3D, signed-GELU decomposition albo nazwanie tokenów cząstkami nie wystarcza już jako nowość.

## 1. Zakres wykonanego researchu

- Undermind: 3 nowe, ukończone deep searches, 8 dodatkowych wyszukiwań semantycznych, odczyt wyników 4 wcześniejszych deep searches, weryfikacja metadanych oraz 3 wywołania odczytu PDF, po 4 prace. Narzędzie czytające PDF odpowiadało na konkretne pytania do pełnych tekstów 12 prac.
- Nowe deep searches zwróciły odpowiednio 140, 112 i 200 pozycji rankingowych: łącznie 452 wpisy, **nie 452 unikalne przeczytane artykuły**.
- SciSpace: 12 zapytań tematycznych; 120 zwróconych rekordów, 113 identyfikatorów wyników. Cztery wywołania ekstrakcji kolumn, 8 ekstrakcji. Osobno sprawdzano źródła pierwotne i status publikacji.
- Scite: 27 kwerend tematycznych oraz 8 kwerend o informacje redakcyjne. 18 wywołań odczytu: 13 stron tekstu z 11 DOI, 3 odpowiedzi tylko z abstraktem i 2 bez tekstu. Była to lektura fragmentów, nie kompletnych 11 artykułów. Dodatkowo graf cytowań i eksport bibliografii. Łącznie 57 wywołań Scite.
- Łącznie 50 wyszukiwań tematycznych, licząc 3 deep searches jako trzy wyszukiwania; kwerendy redakcyjne i odczyty są osobno.
- Audyt kodu obejmował m.in. definicję modelu, rekonstrukcję MLP i uwagi, pola próbne 3D, pomiar głowy, analizę OU oraz zapisany harmonogram checkpointów. Sprawdzono także lokalny runtime LayerNorm.
- To przegląd ukierunkowany, nie gwarancja kompletności literatury. Wyniki wyszukiwarki, syntezy narzędzi i abstrakty nie były traktowane jako dowody równoważne pełnemu tekstowi.

Raporty składowe:

- [Audyt matematyczny i kodowy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_fizyczny_mikroskop_20261015/math_audit.md).
- [SciSpace: najbliższe wcześniejsze prace](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_fizyczny_mikroskop_20261015/scispace/RAPORT_SCISPACE.md).
- [Scite: fizyka, identyfikacja, bibliografia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_fizyczny_mikroskop_20261015/scite/SCITE_RAPORT.md).
- [Undermind: wyniki i odczyty](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_fizyczny_mikroskop_20261015/undermind/RESULTS_AND_READINGS.json).
- [Undermind: dodatkowe cztery pełne teksty](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_fizyczny_mikroskop_20261015/undermind/FULLTEXT_FOLLOWUP.json).

Nowe wyszukiwania Undermind:

1. [Physical realizability and coordinate validity of a transformer optical microscope](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Physical%20realizability%20and%20coordinate%20validity%20of%20a%20transformer%20optical%20microscope).
2. [Causal emergence of coordinated neuron force fields during transformer learning](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Causal%20emergence%20of%20coordinated%20neuron%20force%20fields%20during%20transformer%20learning).
3. [Identifying stochastic confinement from neural representation trajectories](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Identifying%20stochastic%20confinement%20from%20neural%20representation%20trajectories).

## 2. Dokładna definicja badanego układu

Bieg s47: obraz 480×320, łodyga konwolucyjna, siatka 20×30, 600 tokenów, sześć bloków pre-LN, trzy głowy uwagi, 806 checkpointów. W strumieniu rezydualnym token ma 192 współrzędne. W MLP powstaje 768 aktywacji GELU, następnie rzutowanych z powrotem do 192 wymiarów.

To dwie poprawne, ale różne reprezentacje:

| Obiekt | Wymiar | Znaczenie |
|---|---:|---|
| x_j | 192 | Stan tokenu, do którego dodawana jest aktualizacja MLP |
| z_j | 768 | Wektor aktywacji ukrytej warstwy MLP; wcześniejszy badany latent |
| v_c | 192 | Kierunek wyjściowego wkładu neuronu c |
| X | 600×192 | Wspólny stan wszystkich tokenów przy analizie uwagi |

W terminologii użytkownika współrzędne konkretnego stanu wyznaczają jego mikrowektory względem wybranych makrowektorów, czyli osi danej przestrzeni. Nie należy jednak utożsamiać 768 kolumn wyjściowych v_c z bazą 192D: jest ich więcej niż wymiarów i nie są niezależną bazą. Globalny układ współrzędnych nie jest średnią tokenów.

Dla ustalonego checkpointu i bloku:

\[
z_{jc}=\operatorname{GELU}(w_c^\top\operatorname{LN}(x_j)+b_c),\quad
u_{jc}=z_{jc}v_c,\quad
F_j=\sum_{c=1}^{768}u_{jc}+b_2,\quad x_j^+=x_j+F_j.
\]

To ścisła algebra sieci. „Celownik” w_c ustala zależność aktywacji od stanu, a „wiązka” v_c ustala kierunek wkładu. Neuron ma tę samą parę wag dla wszystkich tokenów, ale różne aktywacje.

GELU ma znak. Jego ujemna wartość oznacza ujemny współczynnik wektora, nie ujemne fizyczne natężenie światła. Natężenie, faza, polaryzacja, długość fali, polaryzowalność cząstki i transfer pędu nie zostały wyprowadzone z tej architektury.

Uwaga sprzęga stany tokenów. Jej wkłady są na ogół skierowane i nie spełniają zasady akcji–reakcji. Pełny stan ma 115200 współrzędnych. MLP działa lokalnie na każdy token; uwaga zależy od pozostałych tokenów. Taki opis ma precedensy w matematyce transformerów [Lu i in.](https://arxiv.org/abs/1906.02762), [Geshkovski i in.](https://doi.org/10.1090/bull/1863).

### Dwa zegary

**Głębokość:** 12 różnych map, po jednej uwadze i jednym MLP na blok. Token rzeczywiście przechodzi od stanu do kolejnego stanu. Nie przebywa dowolnie długo w jednym zamrożonym polu.

**Uczenie:** dla stałego obrazu I odczyt jest x_j(t)=f_j(I;θ_t,b_t), gdzie b_t zawiera bufory BatchNorm. Różnica między checkpointami jest zmianą wyjścia przeliczanej sieci. Pamięć procesu obejmuje także momenty AdamW i sposób losowania danych. Trajektoria tego odczytu jest poprawnym przedmiotem badania; nie wynika z niej autonomiczne równanie ruchu pojedynczego tokenu.

W pierwszym rzędzie:

\[
\Delta x_j\approx D_\theta f_j\,\Delta\theta+D_b f_j\,\Delta b.
\]

Stąd wspólny napęd i korelacje tokenów. 600 tokenów jednego obrazu nie daje 600 niezależnych realizacji procesu uczenia. Równanie pola MLP z zegara głębokości nie jest automatycznie równaniem dryfu tokenu między checkpointami.

## 3. Najbliższe mosty do fizyki

| Most | Co można przenieść | Czego jeszcze nie wykazano |
|---|---|---|
| Koloid w pułapce sterowanej sprzężeniem zwrotnym | Pomiar stanu, obliczenie żądanego działania, przestawienie aktuatora | Realizacji dowolnego pola 192D przez fizyczne lasery |
| Rekonstrukcja lokalnego pola sił, FORMA | Jacobian, odpowiedź na perturbację, rozciąganie i część obrotowa | Tożsamości aktywacji z siłą w niutonach |
| Niepotencjalna dynamika, układy aktywne | Odpowiedź niesymetryczna, brak wspólnego potencjału, stabilność właściwej mapy | Fizycznej energii, naprężenia lub bilansu pracy |
| Wielowymiarowy OU i uogólnione równanie Langevina | Modele dryfu, fluktuacji, pamięci i odpowiedzi | Markowowskości tokenu, równowagi cieplnej, zmierzonej temperatury |
| Badania szkła | MSD, zależność od wieku, rozkłady czasów relaksacji jako hipotezy | Fazy szklistej wynikającej z podobieństwa filmu lub plateau MSD |

**Najbliższy układ laboratoryjny:** Kumar i Bechhoefer pokazują koloid, którego położenie jest mierzone, a środek optycznej pułapki przesuwany tak, by uzyskać zaprogramowany potencjał. Ich demonstracje obejmują potencjały harmoniczne i dwustudniowe; nie pokazują dowolnego niekonserwatywnego pola wielowymiarowego. To most strukturalny: stan → reguła → działanie. Rozszerzenie tego języka na Twoje pole jest propozycją modelowania, nie wynikiem ich eksperymentu. [Nanoscale virtual potentials using optical tweezers, APL2018](https://doi.org/10.1063/1.5055580).

**Najbliższa metoda pomiarowa:** FORMA rekonstruuje lokalne pola, także niekonserwatywne. W SmallViT można bezpośrednio odpytać funkcję sieci w zadanym stanie, więc nie trzeba wszystkiego odtwarzać ze ścieżki cząstki. To mocna właściwość Twojego mikroskopu. [Pérez García i in., Nature Communications2018](https://doi.org/10.1038/s41467-018-07437-x).

**Pułapka nie wymaga uniwersalnego progu „rotacja mniejsza od sztywności”.** Niektóre niekonserwatywne układy mogą stabilizować siodła, inne destabilizują stabilną pułapkę. Wynik zależy od pełnej dynamiki. Nie wolno używać samego stosunku norm części Jacobianu jako rozstrzygającego kryterium. [Li, Cao, Ng2024](https://doi.org/10.1038/s41467-024-46058-5).

## 4. Co jest poprawne w analizie ruchu i „wiru”

Dla pola MLP:

\[
J_F=W_2\operatorname{diag}(\operatorname{GELU}'(a))W_1D\operatorname{LN}(x),
\quad S=\frac{J_F+J_F^\top}{2},\quad A=\frac{J_F-J_F^\top}{2}.
\]

To dokładne narzędzia lokalne. S opisuje lokalną zmianę długości dla liniowej dynamiki ciągłej w zadanej metryce euklidesowej, A lokalną część obrotową. W 3D wektor z elementów A odpowiada połowie rotacji pola, zależnie od przyjętej konwencji.

Niezerowe A wyklucza zwykłe euklidesowe pole gradientowe w danym otoczeniu, lecz nie dowodzi zamkniętych orbit. Pole ścinające F(x,y)=(ky,0) ma niezerowe A, mimo prostoliniowych trajektorii. Nie jest to też globalny rozkład Helmholtza ani automatycznie prąd prawdopodobieństwa.

Dla rzeczywistego kroku residual liczy się M=I+J_F:

- stabilność liniowa hipotetycznie iterowanego punktu stałego: promień spektralny M poniżej 1;
- maksymalne wzmocnienie perturbacji w jednym kroku: największa wartość singularna M;
- mnożnik lokalnej objętości: |det M|;
- przepływ przez wiele podkroków: właściwy iloczyn Jacobianów w kolejności sieci.

Kurczenie objętości może współistnieć z rozszerzaniem części kierunków. Dodatkowo pre-LN daje J_F·1=0 i M·1=1. Sam MLP zachowuje neutralny kierunek przesunięcia wszystkich współrzędnych tokenu. Bez usunięcia tej symetrii nie jest ściśle przyciągającą pułapką w całej przestrzeni.

Rozkład S+A zależy od metryki. Jest spójny pod wspólną ortogonalną zmianą bazy, ale dowolne przeskalowanie osi zmienia jego interpretację. Rzut do 3D może ukryć składowe pola. P F=0 nie oznacza F=0; punkt stały rzutu nie musi być punktem stałym pełnego układu.

PCA dopasowana do samej trajektorii może wytwarzać regularnie wyglądające krzywe także dla prostych procesów losowych. Nie znaczy to, że każdy film 3D jest artefaktem; potrzebne są kontrole z tą samą procedurą projekcji. [Antognini, Sohl-Dickstein, NeurIPS2018](https://arxiv.org/abs/1806.08805).

Wielowymiarowy OU pokazuje dodatkową granicę pojęcia „wir”. Przy dx=-Bx dt+sqrt(2D)dW stacjonarna kowariancja C spełnia BC+CBᵀ=2D, a równowaga szczegółowa wymaga BD=DBᵀ. Antysymetria B w zwykłej metryce nie wystarcza, gdy D jest anizotropowe. Nie wolno stosować tych kryteriów do A pola MLP, jakby to był już zmierzony dryf procesu uczenia. [Godrèche, Luck2018](https://doi.org/10.1088/1751-8121/aaf190).

## 5. Co już jest ciekawe w Twoich wynikach

Źródło liczb: [zasady Z5, Z6, Z15, Z18 i Z20](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app/ZASADY_DZIALANIA_UKLADU.md), zapisane pomiary w katalogu film_krysztal/opis_ukladu. Ich przytoczenie nie oznacza nowej niezależnej replikacji.

### A. Powstaje uporządkowana geometria przeciwnych kierunków

W ostatnim MLP udział reflektorów mających co najmniej jednego partnera z cos(v_c,v_d)<-0,5 rośnie od 0 procent na starcie do 53 procent na końcu. To **odsetek reflektorów z partnerem**, nie 53 procent wszystkich par. Mediana normy wiązki rośnie z 0,29 do 0,84. W dokumentacji znajduje się też porównanie z drugim biegiem, ale nie jest to jeszcze szeroka replikacja wieloseedowa.

To obserwacja uczenia geometrii wag. Nie dowodzi sama, że przeciwne neurony jednocześnie działają na ten sam token ani że mają funkcję hamującą.

### B. Przeciwne kierunki nie oznaczają monotonicznego wzrostu znoszenia

Wskaźnik

\[
q=\frac{\|\sum_c z_cv_c\|}{\sum_c |z_c|\|v_c\|}
\]

wynosi około 0,054 na starcie, 0,216–0,217 w epokach5–10 i 0,144 na końcu. Większe q oznacza **większą wypadkową względem sumy długości**, czyli mniej względnego znoszenia. Jeżeli nazwać znoszenie 1-q, wynosi ono około 95, 78 i 86 procent. Nie są to udziały energii.

Zatem dane nie mówią po prostu „uczenie tworzy coraz większe znoszenie”. Mówią: zmienia się geometria dostępnych kierunków oraz sposób ich używania, a wypadkowa ma niemonotoniczną historię. To lepsze pytanie naukowe. Silne znoszenie występuje już dla wielu losowych wektorów, więc potrzebny jest model zerowy zachowujący normy, anizotropię i aktywacje.

### C. Duża część obrotowa istnieje już przed uczeniem

W dotychczasowych pomiarach stosunek norm A/S około0,99 na starcie jest zbliżony do losowej macierzy; pod koniec wynosi około0,86–0,96 w zależności od tokenu. Obecność A nie jest dowodem narodzin funkcjonalnego wiru wskutek uczenia. Interesujące może być **ukierunkowanie, selekcja albo osłabianie** lokalnej rotacji oraz jej związek z odpowiedzią głowy.

### D. Można powiązać obraz z dokładnymi wkładami

Rysowany skok w głębokości może być sumą rysowanych, poprawnie wycentrowanych wkładów. To odróżnia narzędzie od animacji układu punktów bez związku z operacjami modelu. Trzeba jednak rozróżnić w filmach rzut stanu x, rzut aktualizacji W₂z i rzut kierunku po LN; w historii projektu używano wszystkich tych umów.

### E. Ruch i wynik zadania mają różne geometrie

Głowa czyta pięć liczb z192 współrzędnych. Przy lokalnym rzędzie5 istnieje187-wymiarowa przestrzeń styczna zmian niewidocznych dla wyjścia w pierwszym rzędzie. Duży ruch może mało zmieniać predykcję, a mały ruch w czułym kierunku może zmienić ją znacznie. To dobrze pasuje do badania transportu użytecznego dla zadania. Nie oznacza globalnej płaskiej doliny ani stałych187 kierunków przez cały trening.

## 6. Korekty konieczne przed użyciem obecnego opisu jako podstawy artykułu

| Obecne mocne stwierdzenie | Wynik audytu | Wersja obronna |
|---|---|---|
| „W czasie uczenia koloid jest OU w pułapce” | Dwustronny detrending, różne statystyki w porównaniu, nieregularne odstępy | Opisowe fluktuacje wokół oszacowanego trendu; OU jest hipotezą do falsyfikacji |
| „Cel logitu leży w +∞” | Focal loss clampuje sigmoid do [1e-4,1-1e-4] | Dodatni składnik wykonywanej straty ma plateau od logitu ln9999≈9,21024 |
| „Maksymalny osiągalny logit to6,89” | Pojedynczy Adam i miękka kara, z błędem ramki | Znaleziony kandydat o podanym logicie i błędzie ograniczeń; brak certyfikatu maksimum |
| „Pole uwagi wokół tokenu daje pełny Jacobian self-attention” | Zmieniane tylko query; key/value i średnia są zamrożone | Warunkowe pole odbiorcy przy zamrożonym kontekście |
| „Iloczyn lokalnych macierzy3D mierzy objętość całej sieci” | Projekcja i kompozycja nie komutują | Objętość modelu z projekcją po każdym kroku; pełna propagacja wymaga JVP przez sieć |
| „Duży wir wyklucza pułapkę” | Brak uniwersalnego progu A/S; stabilność zależy od mapy | Testować pełną odpowiedź i właściwą dyskretną dynamikę |
| „1/wariancja to sztywność” | Nieznana dyfuzja, anizotropia, brak równowagi i jednostek | Odwrotność wariancji; sztywność dopiero w zidentyfikowanym modelu |
| „Pomiary są dokładnie float32 i zgodne z głową” | Bufory po drodze float16; epsilon pomiaru1e-6 vs lokalny runtime1e-5 | Oszacować błąd kwantyzacji i zgodność funkcji przed subtelnymi wnioskami |

OU wymaga osobnego podkreślenia. Odjęcie średniej korzystającej z przyszłości nie odkrywa niezależnie centrum fizycznej pułapki. Nawet zwykły random walk po takim filtrowaniu daje pozornie ograniczone reszty. Niezależne pozycje wokół trendu mają korelację sąsiednich przyrostów -1/2. Wartość około-0,415 nie identyfikuje więc siły przywracającej.

W kodzie phi wyznaczono z reszt, lecz porównano ze średnią cosinusów innych, surowych przyrostów. Maska „co100” zostawia w późnym szeregu także57 odstępów po200. Oszacowanie relaksacji29–52 kroki z takich zapisów jest silnie zależne od modelu i wymaga niepewności. Literatura zna błędy estymacji przy rzadkim próbkowaniu [Pérez-García i in.2023](https://doi.org/10.1364/BOE.495468).

Model OU bywa uzasadnionym przybliżeniem SGD pod konkretnymi założeniami. To nie przenosi się automatycznie na nieliniowy odczyt tokenu w AdamW. [Mandt, Hoffman, Blei2017](https://jmlr.org/papers/v18/17-214.html). Pamięć po usunięciu zmiennych ma naturalny język w formalizmie Mori–Zwanziga [Venturi, Li2023](https://doi.org/10.1007/s40687-023-00390-2).

Powyższe nie dowodzi, że wszystkie dotychczasowe pomiary są błędne. Rozdziela dokładną algebrę, wyniki numeryczne konkretnego protokołu i silniejsze interpretacje, których ten protokół jeszcze nie uzasadnia. Nie zmieniono źródłowego opisu ani danych.

## 7. Najbliższe prace i realna granica nowości

| Praca i status | Co już obejmuje | Co pozostaje odrębne w proponowanej pracy |
|---|---|---|
| [Geva i in., EMNLP2021](https://aclanthology.org/2021.emnlp-main.446/) | MLP jako klucze i wartości | Twoje wielokrotne pomiary w trakcie uczenia detektora |
| [Geva i in., EMNLP2022](https://aclanthology.org/2022.emnlp-main.3/) | Signed neuron sub-updates, ujemne GELU, wkłady wzmacniające i przeciwne | Geneza geometrii i przewidywanie interwencji w tokenach obrazowych |
| [Tomihari, Karakida, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cca569b38ec3612904e625dad7a743dc-Abstract-Conference.html) | Jacobiany, stabilność i dynamika rekurencyjnej uwagi bez wymogu energii | Różne bloki z MLP oraz zegar uczenia konkretnego ViT |
| [Jacobs i in., Block-Recurrent Dynamics in ViTs](https://arxiv.org/abs/2512.19941), PDF oznaczony ICLR2026 | Dynamika przez głębokość, kierunkowa zbieżność, perturbacje, koherencja patchy, DMD | Wkłady pojedynczych neuronów i ich organizacja przez setki checkpointów |
| [Mudarisov i in., Feed-Forward Steering](https://arxiv.org/abs/2608.02071), preprint2026 | Pola FFN, kierunki radialne/styczne, lokalne krytyczne kierunki, interwencje w LLM | Detekcja obrazu, uczenie grup wkładów oraz powiązanie z odpowiedzią zadania |
| [Fernando, Guitchounts, Dynamics of Transformer Residual Stream](https://arxiv.org/abs/2605.14258), preprint2026 | Jacobiany, nienormalność i spektra LLM, także kilka etapów uczenia | Dokładny poziom neuron–token w ViT i interwencje w rzeczywistym forward, nie tylko stosie średnich Jacobianów |
| [Halperin, Learning as Observable Matrix Dynamics](https://arxiv.org/abs/2606.29679), preprint2026 | Reprezentacje stałych przykładów jako poruszające się cząstki podczas uczenia | Przestrzenne tokeny w obrazie, rozkład mechanizmu na neurony i uwagę |
| [Chowdhury i in., Hedonic Neurons](https://arxiv.org/abs/2509.23684), czytany preprint2025 | Koalicje MLP, współaktywacje i nieaddytywność ablacji w LLM | Ewolucja koalicji przez trening oraz wektorowe znoszenie w ViT |
| [Berend i in., ResLRP](https://arxiv.org/abs/2609.17152), preprint2026 | Znoszenie gałęzi residual w ViT i niestabilność atrybucji | Znoszenie wewnątrz sumy neuronów MLP; skutki dla forward i uczenia |
| [Rushing, Nanda, ICML2024](https://arxiv.org/abs/2402.15390) | Samonaprawa i kompensacja po ablacji | Konieczna kontrola interpretacji efektów interwencji w Twoim układzie |

Doprecyzowania wynikające z pełnych tekstów:

- Jacobs i in. mierzą m.in. uśrednione dynamiki grup tokenów oraz odpowiedź na perturbację. Ich DMD nie jest pełnym Jacobianem pola pojedynczego neuronu.
- Mudarisov i in. analizują pola i sterowanie kierunkiem residual streamu w modelach językowych. Sama nazwa „pole MLP sterujące tokenem” jest przez to bliskim wcześniejszym pomysłem.
- Fernando i Guitchounts rozdzielają spektrum i nienormalną strukturę; ich opisana interwencja Schura na stosie macierzy nie jest dowodem zmiany jakości zadania w zmodyfikowanym forward.
- Halperin używa stałych przykładów jako cząstek podczas treningu. Nie należy twierdzić, że checkpointowy „mikroskop uczenia” jako ogólna idea nie miał precedensu.
- Hedonic Neurons mierzy koalicje i interwencje, ale „śledzenie” oznacza głównie sąsiednie warstwy wytrenowanego LLM, nie historię powstawania koalicji przez checkpointy.
- ResLRP bada przede wszystkim znoszenie dwóch gałęzi w dodawaniu residual. Jego interwencje zmieniają regułę propagacji relewancji, nie są ablacją Twoich768 ukrytych wkładów w forward.

Te różnice uzasadniają poszukiwanie konkretnego nowego wyniku. Nie dowodzą automatycznie pierwszeństwa całej proponowanej kombinacji. Statusy preprintów są jawne; metadane agregatorów bywały niepoprawne.

## 8. Proponowane pytanie artykułu

**Czy uczenie organizuje przeciwstawne, zależne od tokenu wkłady MLP tak, że ich współdziałanie przewiduje i przyczynowo kontroluje zmianę odpowiedzi detektora?**

Roboczy tytuł:

**Learning to Steer: Training-Time Organization of Neuron Contributions in Vision Transformers**

Polski odpowiednik: **Jak Vision Transformer uczy się sterować reprezentacją: powstawanie współdziałania neuronów obserwowane mikroskopem obliczeń.**

Mikroskop jest metodą i narzędziem. Głównym wynikiem powinna być odpowiedź na powyższe pytanie. Optyka może organizować ilustracje i objaśniać metodę, ale zdanie streszczenia powinno mówić, co dokładnie zmierzono i jak wykazano funkcję tego mechanizmu.

### Hipoteza główna

W trakcie uczenia powstają zespoły neuronów o zależnej od rodzaju tokenu konfiguracji kierunków i aktywacji. Ich geometria oraz lokalna odpowiedź pozwalają przewidzieć, które interwencje zmienią logit środka i geometrię ramki. Przeciwne kierunki mogą zapewniać elastyczność sterowania, nawet jeśli jednoczesne znoszenie nie rośnie monotonicznie.

To propozycja do sprawdzenia. Obecny histogram partnerów i wskaźnik q jeszcze jej nie potwierdzają.

### Alternatywne wyjaśnienia

- Antypary są skutkiem anizotropowego, małowymiarowego rozkładu wag.
- Przeciwne neurony są używane w innych kontekstach, więc nie tworzą współdziałającej pary na danym tokenie.
- Wpływ ablacji wynika jedynie z normy usuniętego wkładu.
- Zmiany obserwowane w czasie są głównie wspólnym napędem wcześniejszych bloków lub normalizacją.
- Różnice narzędzie–tło wynikają z obrazu, położenia lub doboru kilku tokenów.
- Film uwidacznia tylko kierunki wybrane z końca treningu.

### Kryterium falsyfikacji

Jeżeli po dopasowaniu norm, aktywności, anizotropii i warstwy pary/grupy wskazane przez mikroskop nie przewidują skutków lepiej niż losowe grupy, teza o odkryciu funkcjonalnej organizacji nie jest podparta. Pozostaje wynik opisowy i narzędzie. Jeżeli efekt jest odtwarzalny i przewiduje interwencje na nowych obrazach, dostajemy wynik mechanistyczny.

## 9. Najkrótszy program eksperymentalny prowadzący do mocnej pracy

To plan kolejnych eksperymentów, nie lista wykonanych badań. W pierwszej kolejności można użyć zapisanych checkpointów. Nowe biegi należy zapisywać osobno.

1. **Kalibracja instrumentu.** Zgodność oryginalnego forward z rekonstrukcją, jawne epsilon LN, precyzja odczytu, biasy, centrowanie. Pełny attention JVP osobno od query-only. Perturbacje należy przepuszczać przez całą sieć bez projekcji po każdym kroku. Film powinien mieć stałą bazę i jawną skalę; dodatkowo wyniki pełnowymiarowe.

2. **Mapa powstawania organizacji.** Dla wybranych checkpointów, wszystkich bloków i wielu obrazów mierzyć kierunki wag, q, współaktywacje, parowe iloczyny skalarnych rzeczywistych wkładów oraz ich zależność od logitu/ramki. Rozdzielić geometrię wag od sposobu używania wag.

   Tożsamość pomocnicza:
   \[
   \|\sum_c u_c\|^2=\sum_c\|u_c\|^2+2\sum_{c<d}u_c^\top u_d.
   \]
   Pozwala wskazać, jakie pary faktycznie wzmacniają albo zmniejszają wypadkową dla danego tokenu. Bias raportować osobno.

3. **Modele zerowe zachowujące istotne właściwości.** Inicjalizacja; permutacja aktywacji między odpowiednio dobranymi tokenami; losowe grupy dopasowane do norm, aktywności i warstwy; surogaty zachowujące anizotropię i widmo geometrii. Zwykła izotropowa losowa macierz nie wystarcza jako jedyna kontrola wytrenowanej macierzy.

4. **Interwencja pojedyncza i wspólna.** Wyłączać albo zastępować średnią wkłady neuronu c, d i pary c,d; mierzyć dokładny forward, lokalną aktualizację, logit, ramkę i jakość detekcji. Porównywać wspólny efekt z sumą pojedynczych oraz z kontrolami o dobranej normie usuniętego wektora. Nieliniowość głowy i dalsza kompensacja wymagają oddzielnych odczytów lokalnych i końcowych.

5. **Przewidywanie przed interwencją.** Na części obrazów wyznaczyć regułę wskazywania ważnych par/grup. Na nieużytych obrazach przewidzieć kierunek i wielkość zmiany wyjścia, a potem ją zmierzyć. Porównać z prostym rankingiem aktywacji, normą wkładu i gradientem. To test użyteczności nowego narzędzia.

6. **Replikacja.** Kilka niezależnych seedów, rozdzielenie klatek według filmów i co najmniej druga konfiguracja ViT. Tokeny i kolejne klatki są skorelowane; nie traktować ich jako niezależnych prób w przedziałach ufności. Podawać niepewność na poziomie obrazu/filmu i biegu.

7. **OU jako oddzielna gałąź.** Najpierw przetestować obecny pipeline na random walk, niezależnym szumie wokół trendu, OU i procesach z pamięcią przy dokładnie tym samym harmonogramie zapisów. Następnie dopiero krótkie, oddzielne biegi ze znacznie gęstszym odczytem i replikami losowania batchy. Nie uzależniać głównego artykułu o mechanizmie MLP od potwierdzenia pułapki.

Mierniki lokalnej cyrkulacji i rotacji mogą wejść do testu przewidywania jako cechy. Nie trzeba na początku dowodzić produkcji entropii, fazy szklistej ani realizować fizycznego układu laserowego, żeby uzyskać wartościowy wynik.

## 10. Proponowana konstrukcja wyników publikacji

- Rysunek1: definicja obu zegarów i dokładne powiązanie reflektor–wkład–przesunięcie.
- Rysunek2: ewolucja geometrii przeciwnych kierunków i aktywnego znoszenia, z modelami zerowymi.
- Rysunek3: zależność mechanizmu od tokenów obiektu/tła i od kierunków czytanych przez głowę.
- Rysunek4: przewidywanie i rzeczywisty efekt interwencji pojedynczych oraz grupowych.
- Rysunek5: replikacja, zakres poprawności lokalnej linearyzacji oraz różnica między pełnym wymiarem a filmem3D.
- Materiały dodatkowe: precyzja pomiaru, szczegóły projekcji, wyniki negatywne, osobna falsyfikacja OU, aplikacja i filmy.

Wynik negatywny także może być ważny: np. antypary nie współdziałają jednocześnie, ale tworzą zależny od kontekstu zestaw przeciwnych działań; albo pozorna studnia 3D nie przewiduje stabilności pełnego przejścia. Trzeba wtedy przepisać tezę zgodnie z pomiarem, zamiast bronić pierwszej metafory.

## 11. Ocena końcowa

Masz instrument pozwalający zobaczyć nie tylko, że reprezentacja się zmienia, lecz także z jakich składników powstaje jej konkretny ruch. To jest obronny rdzeń „mikroskopu uczenia”.

Najbardziej obiecujące pytanie nie brzmi obecnie „czy sieć jest koloidem w laserze”, tylko **jak w trakcie uczenia powstaje organizacja wektorowych działań neuronów i czy ta organizacja wyjaśnia działanie detektora**. Fizyka dostarcza do tego precyzyjnych metod: odpowiedzi na perturbację, identyfikacji pola, stabilności, oddzielenia dryfu od fluktuacji i kontroli ukrytych zmiennych.

Na dziś można mówić o dokładnym mikroskopie obliczeń i interesujących obserwacjach geometrii. Tezy o fizycznej pułapce OU, temperaturze, celu w nieskończoności i globalnej ekspansji wymagają opisanych korekt. Najbliższą drogą do publikacji jest jedna sprawdzona zależność przyczynowa, wsparta tym narzędziem, kontrolami i replikacją.

