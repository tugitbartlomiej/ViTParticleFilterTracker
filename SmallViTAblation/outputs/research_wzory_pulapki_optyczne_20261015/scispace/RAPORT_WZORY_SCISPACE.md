# Wzory z pułapek optycznych i fizyki koloidów dla SmallViT: audyt SciSpace

Data: 2026-10-15. Autor: agent badawczy SciSpace. Zakres: literatura, definicje pomiarów, propozycje falsyfikowalnych doświadczeń; żadnych eksperymentów ani zmian kodu nie wykonano.

## Najważniejszy wynik

Można zbudować rygorystyczny „mikroskop” wokół lokalnej cyrkulacji, podatności między tokenami i propagacji zaburzeń do zadania. Większość samych narzędzi ma jednak wcześniejsze zastosowania w ML. Nie ma podstaw do twierdzenia, że przeniesienie pojęć niekonserwatywności, stochastic area, Cauchy–Green, nienormalności lub obserwowalności do sieci jest pierwsze.

Najbardziej wartościowa hipoteza dla tego modelu brzmi: **w trakcie uczenia detektora zmienia się nie tylko geometria reprezentacji, ale też to, które kierunki i interakcje między tokenami rzeczywiście sterują detekcją i lokalizacją**. Hipoteza wymaga przewidywań perturbacyjnych poza próbą i porównań z prostszymi miarami. Skojarzenie z fizyką pomaga wybrać pomiar; nie nadaje latentowi jednostek siły, energii czy temperatury.

W tym raporcie siedem kart zawiera wzory źródłowe lub standardowe oraz wyraźnie oznaczoną własną adaptację do SmallViT. Kolejność wdrożenia: karta 6 wraz z 5, następnie 1–2. Karta 3 jest najbardziej wymagająca statystycznie.

## Audyt wyszukiwania

- 15 różnych pytań w SciSpace, po 10 rekordów: 150 rekordów, 139 różnych identyfikatorów SciSpace.
- Identyfikator SciSpace nie jest tożsamy z publikacją: wersje arXiv i wydawnicze bywają osobnymi rekordami. Liczba 139 nie oznacza 139 niezależnych badań.
- 0 wywołań add_column w tej rundzie. Wcześniejszy audyt wykazał, że automatyczne kolumny nie zastępują sprawdzenia pełnego tekstu.
- 7 zapisanych pakietów weryfikacji internetowej, zawierających wiele otwarć stron, PDF-ów, wyszukań i odczytów fragmentów.
- Surowe wyniki: query_01.json–query_15.json; surowa weryfikacja: web_1.json–web_7.json; pytania i rekordy: search_manifest.json.
- Daty i etykiety SciSpace bywają błędne. Przykładowo rekordy arXiv bywają oznaczone „Journal Article”, a data indeksowania zastępuje datę publikacji. Status poniżej pochodzi z weryfikacji źródła pierwotnego.

Pytania objęły: niekonserwatywne siły optyczne; Brownian vortex; odd elasticity; nienormalność i pseudospektra; FTLE i Cauchy–Green; Gramians; stochastic area; nierównowagowe SGD; odd response w ML; logit versus bbox; podobieństwo dynamiczne; FORMA i próbkowanie; oryginalną teorię odd elasticity; sterowanie sieciami feedforward; DSA i wpływ wejścia.

## Słownik i dwa zegary

Niech x_i^(l)(θ_t) ∈ R^192 oznacza stan rezydualny i-tej cząsteczki, czyli tokenu, w miejscu l i przy checkpointcie t. Jest 600 tokenów, więc pełny stan X ma 115200 współrzędnych. Wewnętrzna aktywacja MLP ma 768 współrzędnych. Nie utożsamiamy 768 aktywacji GELU z wymiarem stanu rezydualnego 192.

**Zegar głębokości:** przy zamrożonym θ mapa bloku T_l prowadzi X_l do X_(l+1). To skończony ciąg różnych map, a nie wielokrotne uruchamianie jednego autonomicznego układu.

**Zegar optymalizatora:** przy stałym obrazie śledzimy X_l(θ_t). Zmieniają się wspólne wagi, stany AdamW i ewentualne bufory normalizacji. Sam token zwykle nie jest stanem Markowa. 806 checkpointów daje jedną obserwowaną historię; 600 tokenów w tej historii nie jest 600 niezależnymi powtórzeniami procesu.

Dla MLP przy zamrożonych wagach:

F(x) = W_2 GELU(W_1 LN(x) + b_1) + b_2,

J_F(x) = W_2 diag[GELU'(W_1 LN(x)+b_1)] W_1 J_LN(x).

Należy uwzględniać J_LN, epsilon i affine LayerNorm. Dla mapy rezydualnej Jacobian wynosi A = I + J_F. Dla pełnego bloku attention+MLP trzeba stosować regułę łańcuchową w rzeczywistej kolejności operacji.

Źródłowe pola sił z pułapek fizycznych są mierzone w przestrzeni położeń. Tutaj F jest obliczeniowym przyrostem stanu. Używana metryka, punkt pomiaru i współrzędne muszą być jawne. Rotacja ortogonalna zachowuje wiele norm, ale dowolna zmiana skali osi zmienia euclidean curl, pola i normy.

## Karta 1. Praca po małej pętli: czy lokalne pole MLP jest gradientowe?

**Źródło fizyczne.** Pesce i in., Quantitative assessment of non-conservative radiation forces in an optical trap, EPL 86, 38002 (2009), DOI [10.1209/0295-5075/86/38002](https://doi.org/10.1209/0295-5075/86/38002), [pełny tekst autora](https://arxiv.org/pdf/0902.4178). Przeczytano pełny PDF, szczególnie analizę różnicy korelacji i niekonserwatywności. FORMA, Nat. Commun. 9, 5166 (2018), DOI [10.1038/s41467-018-07437-x](https://doi.org/10.1038/s41467-018-07437-x), wprost rozdziela lokalny Jacobian na część symetryczną i antysymetryczną; przeczytano [PDF](https://arxiv.org/pdf/1808.05468), równania 5–8. Poniższa pętla i znak są standardowym rachunkiem różniczkowym, zapisanym we własnej notacji.

Niech u,v będą ortonormalnymi kierunkami w R^192, a Γ małą dodatnio zorientowaną pętlą w płaszczyźnie (u,v), przy stałym checkpointcie i wszystkich pozostałych tokenach:

W_Γ = ∮_Γ F(x) · dx,

W_Γ / area(Γ) → v^T J_F u − u^T J_F v.

Przy Ω=(J_F−J_F^T)/2 wynik wynosi −2u^TΩv. Dla prostokąta znak odpowiada przejściu najpierw wzdłuż +u, następnie +v. Wielkość W ma umowne jednostki współrzędnej latentu do kwadratu na krok mapy; nie jest energią w dżulach.

**Własna adaptacja do SmallViT.** Obliczyć pole MLP na małej siatce wokół faktycznego tokenu, a następnie całkę po pętli oraz Jacobianowy limit. Porównać płaszczyzny wybrane przez wyjście modelu z losowymi płaszczyznami. Jest to badanie statycznego pola zamrożonego modelu, a nie dowód, że trajektoria treningowa krąży pod jego wpływem.

**Prognoza dla detekcji i bbox.** Sama duża wartość W nie przewiduje jakości zadania. Hipoteza: podczas nabywania lokalizacji wzrasta lub reorganizuje się cyrkulacja w płaszczyznach, których zaburzenia są widoczne dla bbox, silniej niż w płaszczyznach niewidocznych dla głowicy. Alternatywa: cyrkulacja rośnie wszędzie i nie daje informacji ponad normę Jacobianu.

**Interwencja i kontrole.** Oszacować Ω lokalnie, wstrzyknąć małe zaburzenia ±εu, ±εv, porównać przewidywane i rzeczywiste odpowiedzi przez resztę sieci. Do przyczynowego testu roli Ω można w lokalnym obszarze zastosować poprawkę F_corr(x)=F(x)−Ω(x−x_0), która zachowuje F(x_0), ale zmienia liniową rotację; jest to sztuczna lokalna interwencja, nie usunięcie istniejącego modułu. Porównać z poprawką losową o tej samej normie i z usunięciem części symetrycznej. Wykonać serię ε, odwrócić orientację pętli, sprawdzić skalowanie z polem. Nie wolno po każdej perturbacji wybierać innej korzystnej płaszczyzny.

**Nowość.** Matematyka nie jest nowa; niekonserwatywność sieci nie jest nową ideą. Kandydatem na wkład jest falsyfikowalne powiązanie tego pomiaru z rozdzieleniem detekcji i lokalizacji w konkretnym ViT. Uwaga z równoległego audytu: Li i in. (2026), Activation-Space Order-Swap Geometry: A Site-Asymmetry Audit, [arXiv:2608.25315](https://arxiv.org/abs/2608.25315), bada zamianę kolejności interwencji między miejscami. To nie jest ta sama wielkość co ∮F·dx. Przy doświadczeniu opartym na zamianie kolejności trzeba jednak oddzielić asymetrię miejsc i samą krzywiznę. Weryfikacja tej pracy: agent główny, pełny tekst, a nie sam wynik SciSpace.

## Karta 2. Wzajemność odpowiedzi między tokenami: analogia do odd elasticity

**Źródło.** Scheibner i in., Odd elasticity, Nature Physics 16, 475–480 (2020), DOI [10.1038/s41567-020-0795-y](https://doi.org/10.1038/s41567-020-0795-y). Przeczytano [pełny PDF autora](https://purehost.bath.ac.uk/ws/portalfiles/portal/205056987/OddElasticity_MainText.pdf), zwłaszcza równanie siły pary i pracę po cyklu odkształcenia. W fizyce odd modulus spełnia C^o_ijmn=−C^o_mnij i może dawać niezerową pracę w zamkniętym cyklu strain. Nie należy nazywać Jacobianu ViT modułem sprężystości bez definicji strain i stress.

Dla pełnego przyrostu bloku F_i(X) zdefiniujmy bloki Jacobianu:

J_ij = ∂F_i / ∂x_j,   R_ij = J_ij − J_ji^T.

Dla kierunków u w tokenie i oraz v w tokenie j:

r_ij(u,v) = u^T J_ij v − v^T J_ji u.

Jest to **nieodwzajemniona podatność**: porównanie wpływu j na i z odpowiednią odwróconą perturbacją. W stałej metryce euclidean pełne pole gradientowe miałoby symetryczny Jacobian. Wynik zależy od jawnie przyjętej metryki; przy innej metryce warunek odpowiednio się zmienia.

**Własna adaptacja.** Same attention weights A_ij ≠ A_ji nie wystarczają. Trzeba uwzględnić pochodne Q/K, softmax, projekcje V/O, normalizację i residual. MLP działający osobno na tokenach ma w tym miejscu zerowe bloki między różnymi tokenami; attention i późniejsze warstwy mogą je wytworzyć. Pełny J ma wymiar 115200×115200, ale do pomiaru r wystarczą iloczyny Jacobian–wektor i odpowiednie odczyty.

**Prognoza zadaniowa.** Hipoteza: w miarę uczenia bbox wpływ tokenów na granicy narzędzia na reprezentację/wyjście wybranego tokenu detekcyjnego staje się kierunkowo inny niż wpływ odwrotny. Miara powinna przewidywać wynik jednostronnej perturbacji także po dopasowaniu odległości przestrzennej, normy aktywacji i siły attention. Nie wynika z teorii, że asymetria musi monotonicznie rosnąć.

**Interwencja i kontrola.** Zaburzyć parę tokenów w obu kierunkach z tą samą normą i zmierzyć lokalną odpowiedź partnera oraz ostateczny logit/bbox. Porównać obiekt–obiekt, obiekt–tło i tło–tło z dopasowaniem odległości. Losować pary i kierunki na oddzielnej próbie. Dodatkowo użyć czułości do głowicy z karty 6, bo R_ij może być duże w kierunkach niewidocznych zadaniowo.

**Nowość.** Odd elasticity w materii aktywnej jest ustalonym polem. Znaleziona praca Emergence of odd elasticity in a microswimmer using deep reinforcement learning, Phys. Rev. Research 6, 033016 (2024), DOI [10.1103/PhysRevResearch.6.033016](https://doi.org/10.1103/PhysRevResearch.6.033016), dotyczy fizycznego obiektu uczonego RL, nie dowodu odd elasticity ukrytych reprezentacji. Nie znaleziono w tej rundzie potwierdzonego identycznego protokołu dla tokenów detektora; to luka w wynikach wyszukiwania, nie dowód pierwszeństwa.

## Karta 3. Stochastic area i asymetria opóźnionych korelacji: czy ruch w treningu ma kierunek czasu?

**Źródła.** Strang, A Theoretical Review of Area Production Rates as Test Statistics for Detecting Nonequilibrium Dynamics in Ornstein–Uhlenbeck Processes, Axioms 13(12), 820 (2024), DOI [10.3390/axioms13120820](https://doi.org/10.3390/axioms13120820), [pełny HTML autora](https://arxiv.org/html/2411.07613v1), równania 4, 7–9 oraz część o wybielaniu szumu. Pesce 2009 stosuje różnicę korelacji krzyżowych w optycznej pułapce. **Bezpośrednie wcześniejsze zastosowanie w ML:** Adhikari i in., Machine learning in and out of equilibrium, [arXiv:2306.03521](https://arxiv.org/html/2306.03521v1), preprint 2023; przeczytano pełny HTML, szczególnie §2 i równania 6–7 o stochastic area i produkcji entropii w SGD.

Przy naszej jawnej konwencji OU:

dz = Bz dt + sqrt(2D) dW,   BC + CB^T + 2D = 0,

gdzie B ma stabilne wartości własne, D jest macierzą dyfuzji, a C stacjonarną kowariancją. Zorientowane pole na jednostkę czasu:

A_hat = (1/(2T)) Σ_k [z_k z_(k+1)^T − z_(k+1) z_k^T],

E[A_hat] → (CB^T − BC)/2.

Dla K(τ)=E[z_t z_(t+τ)^T] ta sama granica wynosi:

lim_(τ→0) [K(τ)−K(τ)^T]/(2τ).

To zapis w naszej notacji i konwencji znaku. Zmiana definicji kolejności korelacji zmienia znak. Dla stacjonarnego OU o zmiennych parzystych przy odwróceniu czasu detailed balance oznacza BC=CB^T. Poza tym modelem wniosek wymaga dodatkowych założeń.

**Własna adaptacja do zegara optymalizatora.** Mierzyć niskowymiarowe, z góry wybrane projekcje x_i(θ_t), po oddzieleniu trendu. Najciekawsza projekcja zadaniowa to para kierunków logit–bbox lub dwa kierunki bbox. Głowica też uczy się w czasie, więc trzeba porównać odczyt bieżącą głowicą z odczytem ustaloną głowicą referencyjną, jasno wskazując ich różne znaczenie. Nie mieszać kroku checkpointu z jednym krokiem AdamW.

**Prognoza.** Hipoteza: czasowe uporządkowanie zmian cech lokalizacji i klasyfikacji, mierzone A_hat, przewiduje opóźnienie poprawy bbox względem logitu. Alternatywa: sygnał znika po usunięciu learning-rate schedule, zmian head i trendu, a więc był wspólnym wymuszeniem. Sama niezerowa empiryczna macierz nie jest dowodem prądu stacjonarnego.

**Interwencja i kontrola.** Dla wiarygodnej przyczynowości potrzebne byłyby nowe kontrolowane powtórzenia treningu lub wznowienia z identycznego checkpointu, z kontrolą minibatchy i stanów optymalizatora. W tym raporcie ich nie wykonano. Na istniejących danych: blokowy bootstrap, przedziały niepewności, test po odwróceniu czasu, korekta wielokrotnego wyboru płaszczyzn i nulowy model odwracalny dopasowany do kowariancji, autokorelacji i próbkowania. Permutacja pojedynczych checkpointów niszczy autokorelację i jest zbyt łatwym nullem.

**FORMA jako warunkowa procedura identyfikacji.** W lokalnym oknie można dopasować Δz/Δt=a+Bz+η i D z reszt, lecz zmienny krok wymaga odpowiednich wag. Dla rzadkiego próbkowania lepiej dopasować dyskretnie z_(k+1)=A_Δ z_k+ε_k, z:

A_Δ=e^(BΔ),   Q_Δ=∫_0^Δ e^(Bs) 2D e^(B^T s) ds.

Przybliżenie B≈(A_Δ−I)/Δ może być obciążone; logarytm macierzowy może być niejednoznaczny. Nie wolno z checkpointów co 100 kroków deklarować zidentyfikowanego czasu relaksacji kilku kroków bez dodatkowych danych. Ukryte momenty AdamW i niestacjonarność mogą unieważnić model OU. W forward pole jest dostępne z modelu: nie trzeba rekonstruować go metodą FORMA z trajektorii.

**Nowość.** Stochastic area w uczeniu ma bezpośredni prior Adhikari 2023. Nowym wynikiem może być konkretna, odtworzona zależność między prądami obserwowanej reprezentacji a rozdzieleniem zadań, po odrzuceniu wspólnego wymuszenia i efektu head. Nie należy nazywać A_hat produkcją entropii ani temperaturą.

## Karta 4. Nienormalność i pseudospektrum: stabilne wartości własne mogą ukrywać wzrost zaburzenia

**Źródło ML.** Kerg i in., Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics, NeurIPS 2019. [Strona konferencji](https://proceedings.neurips.cc/paper_files/paper/2019/hash/9d7099d87947faa8d07a272dd6954b80-Abstract.html), [pełny PDF](https://papers.nips.cc/paper_files/paper/2019/file/9d7099d87947faa8d07a272dd6954b80-Paper.pdf), przeczytano pełny tekst. Praca celowo wykorzystuje nienormalność i przejściową dynamikę w RNN. Wcześniejsze neuronalne sieci losowe: Non-normal amplification in random balanced neuronal networks, Phys. Rev. E 86, 011909 (2012), DOI [10.1103/PhysRevE.86.011909](https://doi.org/10.1103/PhysRevE.86.011909); w tej rundzie abstrakt/metadane.

Dla Jacobianu mapy A=DT_l standardowe diagnostyki:

g_1 = σ_max(A),

ν(A) = ||A^T A−AA^T||_F / ||A||_F^2,

Λ_ε(A) = {z ∈ C : σ_min(zI−A) < ε}.

ν to wybrana tu normalizacja standardowego komutatora; wartość wymaga stałej metryki. Jeśli jeden układ jest wielokrotnie iterowany, ρ(A)<1 nie wyklucza ||A^k||>1 przez skończony czas. Własne wartości nie opisują całego przejściowego wzmacniania.

**Własna adaptacja.** SmallViT ma różne bloki. A_l^k nie jest jego rzeczywistym forward. Pseudospektrum pojedynczego A_l jest lokalną diagnostyką i hipotetyczną odpowiedzią powtarzanej mapy; rzeczywistą propagację wyznacza iloczyn różnych A_l z karty 5. Używać mapy residual A, a nie wartości własnych samego przyrostu J_F, jeśli pytanie dotyczy wzrostu zaburzenia po bloku.

**Prognoza.** Hipoteza: w pewnych checkpointach mała perturbacja reprezentacji daje duże chwilowe przemieszczenie, ale niekoniecznie zmianę detekcji. Po dodaniu czułości głowicy tylko część tych kierunków będzie przewidywać niestabilny bbox. Porównanie σ_max(A) i ρ(A) może ujawnić różnicę pomijaną przez same eigenvalues.

**Interwencja i kontrola.** Zaburzyć stan w prawym wektorze singular odpowiadającym największemu σ, następnie przeprowadzić prawdziwy forward. Porównać z eigenvector, kierunkiem losowym i kierunkiem o największej czułości bbox, przy identycznej normie i kilku ε. Oddzielić wzrost normy stanu od wzrostu błędu zadania.

**Nowość.** Nienormalne wzmacnianie w sieciach neuronowych nie jest nowe. Potencjalnym wynikiem jest dodatkowa zdolność predykcyjna dla bbox w tym modelu oraz przyczynowa walidacja względem tańszych baseline'ów.

## Karta 5. Tensor Cauchy’ego–Greena i FTLE: rzeczywiste odkształcenie przez kolejne bloki

**Bezpośredni prior ML.** Storm, Linander, Bec, Gustavsson i Mehlig, Finite-Time Lyapunov Exponents of Deep Neural Networks, Physical Review Letters 132, 057301 (2024), DOI [10.1103/PhysRevLett.132.057301](https://doi.org/10.1103/PhysRevLett.132.057301). Przeczytano [pełny PDF wydawcy](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.132.057301), szczególnie definicję Jacobianowego iloczynu, singular values i Cauchy–Green na początku pracy. To wyklucza deklarację pierwszego zastosowania tych miar do deep learningu.

Dla zamrożonego checkpointu:

M_(b←a) = A_(b−1) A_(b−2) ... A_a,

C_(b←a) = M_(b←a)^T M_(b←a),

λ_FT(a,b) = log σ_max(M_(b←a)) / (b−a).

Wektor własny C o największej wartości daje kierunek maksymalnego lokalnego rozciągnięcia. „Czas” jest liczbą wybranych kroków architektury; należy jawnie określić, czy krok oznacza cały blok, attention czy MLP. W sześciu blokach nie wynika z tego asymptotyczny wykładnik ani chaos.

**Własna adaptacja.** Stan pełny ma 600×192 współrzędne. Iloczyn samych diagonalnych bloków Jacobianu 192×192 ignoruje ścieżki perturbacji przez inne tokeny i powrotu. Można zbadać interwencję w jednym tokenie przez embedding E_i: M E_i, a następnie odpowiedni readout. To uwzględnia wszystkie tokeny po interwencji, choć wejście zaburzenia jest 192D. JVP/VJP pozwalają unikać materializacji pełnej macierzy.

**Prognoza.** Hipoteza: w uczeniu pojawiają się kierunki, które są wzmacniane przez wczesne bloki i następnie przekształcane w kierunki lokalizacji widoczne głowicy. Alternatywa: maksymalnie rozciągany kierunek jest niewidoczny dla zadania. Dodatni λ_FT nie oznacza automatycznie lepszej lub gorszej detekcji.

**Interwencja i kontrola.** Wprowadzić ±εv w tokenie w warstwie a, zmierzyć stan w b i końcowy logit/bbox, sprawdzić stosunek błędu nieliniowego do ε. Porównać z końcowym task Jacobianem z karty 6 oraz z normą przyrostu MLP. Zachować wybraną bazę między checkpointami albo jawnie wyrównać reprezentacje, nie przeliczać niezależnej PCA i traktować osi jako tych samych.

**Nowość.** Wzór i interpretacja FTLE w DNN są opublikowanym priorem. Możliwa nowość dotyczy eksperymentalnego związku przepływu perturbacji z lokalizacją narzędzia, nie samego tensora.

## Karta 6. Obserwowalność przez zadanie i minimalna interwencja: co naprawdę dociera do logitu i bbox?

**Źródła i ograniczenia prioru.** Sterowalność i obserwowalność sieci feedforward pojawiają się już w rozdziale The Use of State Space Control Theory for Analysing Feedforward Neural Networks (1998), DOI [10.1007/978-1-4471-1523-6_2](https://doi.org/10.1007/978-1-4471-1523-6_2); tu zweryfikowano abstrakt/metadane. Jihoon Moon, From Black-Box to White-Box: Control-Theoretic Neural Network Interpretability, [arXiv:2511.12852](https://arxiv.org/html/2511.12852v1), preprint 2025, przeczytano pełny HTML. Ta praca pokazuje małe przykłady i konstrukcję ze stosem stanów oraz A=0; nie należy przenosić jej bezpośrednio jako dynamiki przez rzeczywiste kolejne bloki.

Z równoległego audytu: Hazelden i in., Disentangling Computation in Multi-Task Neural Networks with the Green's Operator, [arXiv:2609.40292](https://arxiv.org/abs/2609.40292), NeurReps 2026 workshop według weryfikacji agenta głównego; przeczytał on pełny tekst. Praca wykorzystuje sekwencyjne Jacobiany w wielozadaniowym RNN i porównuje inicjalizację z wytrenowaną siecią. To bardzo bliski prior dla „propagacja + zadanie”, choć nie dotyczy ViT/bbox.

**Standardowa linearyzacja i własna adaptacja.** Niech y=(s,b_1,b_2,b_3,b_4) oznacza wybrany logit oraz cztery jawnie zdefiniowane współrzędne bbox, a g rzeczywisty head wraz z dekodowaniem:

H_a = Dg(X_L) M_(L←a).

Dla perturbacji tylko tokenu i używamy H_a E_i. Dalsze wzory można liczyć w tej 192D przestrzeni wejściowej. Niech h_s będzie wierszem logitu, H_b czterema wierszami bbox:

W_s = h_s^T h_s,

W_b = H_b^T R_b H_b,

δy ≈ H_a δX.

R_b jest zadeklarowaną metryką błędu bbox, np. uwzględniającą skalę obrazu. W_s i W_b to lokalne macierze obserwowalności końcowego wyjścia / pullback metryki wyjścia. Klasyczny Gramian obserwacji wielu chwil wymaga sumy takich składników; przy jednym odczycie nie należy udawać, że obserwowano pełną historię.

Kierunek selektywny dla bbox można definiować przez uogólniony iloraz:

q(v) = v^T W_b v / [v^T W_s v + λ ||v||²].

Minimalna norma perturbacji dla osiągalnego małego celu δy* w nieograniczonym modelu liniowym:

δX* = H_a^T (H_a H_a^T)^† δy*.

Dla interwencji w określone neurony, głowy lub tokeny z operatorami B_k:

T_k = Dg M_(L←k) B_k,   K = Σ_k T_k T_k^T.

Minimalny koszt sumy kwadratów sterowań wynosi δy*^T K^† δy*, jeżeli cel należy do obrazu operatora sterowania. To standardowa teoria liniowa, nie fizyczna energia.

**Prognoza.** Hipoteza: trening wyodrębnia kierunki, które poprawiają lokalizację przy małej zmianie logitu, lub odwrotnie. Testować można przewidywanie podpisanej zmiany konkretnej współrzędnej bbox i logitu, a nie tylko korelację z normą latentów. Head może odczytywać silnie tylko niewielki podzbiór kierunków mimo dużego ruchu całego tokenu.

**Interwencja i kontrola.** Wyznaczyć v na obrazach treningowych protokołu, sprawdzić na innych obrazach i checkpointach. Wprowadzać ±εv i porównywać predykcję HδX z rzeczywistym wynikiem. Baseline: losowy kierunek tej samej normy, kierunek maksymalnego singular value M, gradient logitu i gradient bbox. Raportować odrębnie logit, prawdopodobieństwo, surowe wyjścia bbox i dekodowane piksele. Przy porównywaniu checkpointów potrzebny jest wariant z zamrożonym readoutem oraz wariant z bieżącym head.

**Uwaga o selekcji detekcji.** Jeżeli końcowa decyzja obejmuje argmax, threshold, top-k lub NMS, lokalne pochodne mogą nie opisywać zmiany wybranego obiektu. Trzeba ustalić analizowany kandydat przed perturbacją i osobno zmierzyć skoki dyskretnej selekcji.

**Nowość.** Sam task Jacobian, pseudoodwrotność i Gramians nie są nowe; 2026 Green operator dodatkowo zbliża prior do analizy wielozadaniowej. Wkład musi polegać na konkretnym odkryciu i skutecznej interwencji w detektorze.

## Karta 7. Bezwymiarowe mapy reżimów i podobieństwo dynamiki

**Źródła.** Strang 2024, pełny tekst jak wyżej, analizuje wybielanie szumu i rozdzielenie rotacyjnej/symetrycznej części OU. Ostrow i in., Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis, NeurIPS 2023, [strona konferencji](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html), [arXiv:2306.10168](https://arxiv.org/abs/2306.10168): przeczytano abstrakt i metadane, nie pełny PDF w tej rundzie. Huang i in., InputDSA: Demixing then Comparing Recurrent and Externally Driven Dynamics, [arXiv:2510.25943](https://arxiv.org/abs/2510.25943), preprint 2025: abstrakt. InputDSA odróżnia operator wewnętrzny od wpływu wejścia; to istotne ostrzeżenie dla obserwowanych trajektorii.

**Własna adaptacja, nie nowa liczba podobieństwa z fizyki.** Jeśli lokalny model OU przeszedł walidację i D jest dodatnio określone, wybielić szum:

S = D^(−1/2),   B_tilde = S B S^(−1),

χ = ||skew(B_tilde)||_F / [||sym(B_tilde)||_F + ε_num],

τ_relax = 1 / [−max Re eig(B)],   q_sample = Δt / τ_relax.

χ porównuje rotację z częścią symetryczną po określonym wybieleniu; q_sample mówi, czy próbkowanie rozdziela skalę relaksacji. Nie nazywamy ich liczbą Reynoldsa ani temperaturą. Jeśli B nie jest stabilne, τ_relax w tej definicji nie istnieje. Jeśli D jest osobliwe, potrzebna jest jawna redukcja do dobrze oszacowanej podprzestrzeni i analiza wrażliwości na próg.

Dla forward, bez modelu OU, bezpieczniejsze bezwymiarowe współrzędne mapy reżimów to np. ||skew J||/||sym J||, σ_max(M) oraz względna czułość bbox/logit z karty 6, przy tej samej metryce i tych samych skalach. To opisowe współrzędne, nie twierdzenie o uniwersalności.

**Prognoza.** Hipoteza: checkpointy o podobnej walidacyjnej jakości mogą znajdować się w innych reżimach wrażliwości bbox; grupa wyznaczona miarami dynamiki przewiduje odpowiedź na kontrolowane perturbacje lepiej niż epoka albo odległość CKA. Alternatywa: po dopasowaniu learning rate i normy aktywacji grupowanie nie wnosi informacji.

**Interwencja i kontrola.** Porównać te same interwencje w modelach/checkpointach dopasowanych geometrią, lecz różniących się miarami dynamicznymi. Odróżnić wspólne wymuszenie treningowe od wewnętrznej dynamiki; jeśli używa się DSA na historii optymalizatora, należy uwzględnić harmonogram LR i inne obserwowalne wejścia oraz uczciwie opisać brak stanów AdamW. Nie dopasowywać dowolnych transformacji, które poprawiają podobieństwo kosztem zniszczenia interpretacji osi.

**Nowość.** Porównanie sieci „po dynamice, a nie tylko geometrii” jest już celem DSA. Nowe może być odkrycie konkretnego reżimu detektora i przewidywana interwencja, nie samo hasło ani wykres χ.

## Trzy hipotezy, które można obalić

1. **Cyrkulacja selektywna dla lokalizacji.** Po dopasowaniu normy Jacobianu i aktywacji, lokalna cyrkulacja w ustalonych płaszczyznach widocznych dla bbox przewiduje podpisaną odpowiedź lokalizacji na perturbację lepiej niż cyrkulacja w płaszczyznach losowych. Obalenie: brak dodatkowej predykcji poza normami i task Jacobianem albo zanik wyniku na innych obrazach. Kontrola przyczynowa: lokalne usunięcie Ω versus poprawka losowa tej samej normy.

2. **Wzmacnianie stanu rozdziela się od wpływu na zadanie.** Kierunki największego rozciągania Cauchy–Green nie są konsekwentnie kierunkami największej zmiany bbox; wraz z uczeniem zmienia się ich pokrycie. Obalenie: prosta norma singular całkowicie przewiduje wpływ na bbox, a readout nie dodaje informacji. Interwencja: porównanie v_stretch, v_bbox i kierunku losowego.

3. **Kierunkowa wymiana informacji o granicy obiektu.** Nieodwzajemniona podatność między tokenami granicy narzędzia a tokenami detekcyjnymi przewiduje asymetrię wyników jednostronnych perturbacji ponad attention weights, odległość i normy. Obalenie: brak trafności poza próbą lub podobny efekt w dopasowanych parach tła. Nie zakładamy z góry, że większa asymetria oznacza lepszy model.

Każda hipoteza dotyczy związku i mechanizmu w detektorze, a nie „prawa fizyki dla wszystkich transformerów”. Jeden obraz i jedno seed s47 wystarczą do demonstracji pomiaru, ale nie do mocnej generalizacji o uczeniu tool detection.

## Co mówić o publikacyjnej nowości

Uzasadnione: „Adaptujemy ilościowe diagnostyki lokalnej cyrkulacji i nieodwzajemnionej odpowiedzi do tokenów ViT, zestawiamy je z rzeczywistym transportem perturbacji do dwóch wyjść zadaniowych i testujemy przewidywania interwencyjnie w trakcie uczenia.”

Nieuzasadnione bez dalszego dowodu: „Pierwszy fizyczny opis transformera”, „pierwsze FTLE w deep learningu”, „pierwsze prądy nierównowagowe w uczeniu”, „pierwsza obserwowalność sieci”, „rzeczywista sprężystość i energia tokenów”.

Najsilniejszym rezultatem byłaby odpowiedź na konkretne pytanie: dlaczego dany checkpoint już rozpoznaje narzędzie, ale jeszcze źle wyznacza jego granice, oraz czy mała, przewidziana perturbacja poprawia lokalizację bez psucia detekcji. Wtedy fizyczne narzędzia wyjaśniają mierzalne zachowanie modelu.

## Tabela dowodów i statusów

| Praca | Status zweryfikowany | Co odczytano | Znaczenie |
|---|---|---|---|
| Pesce i in. 2009, DOI 10.1209/0295-5075/86/38002 | EPL, artykuł | Pełny PDF autora | Niekonserwatywne siły, różnica korelacji |
| Pérez García i in. 2018, DOI 10.1038/s41467-018-07437-x | Nature Communications, artykuł | Pełny PDF autora | FORMA, lokalny Jacobian, rozdział części |
| Scheibner i in. 2020, DOI 10.1038/s41567-020-0795-y | Nature Physics, artykuł | Pełny PDF autora | Odd modulus, praca cykliczna |
| Strang 2024, DOI 10.3390/axioms13120820 | Axioms, artykuł | Pełny HTML arXiv; metadane wydawcy | Area production, OU, testy, wybielanie |
| Adhikari i in. 2023, arXiv:2306.03521 | Zweryfikowany preprint; nie potwierdzono późniejszej publikacji | Pełny HTML | Bezpośredni prior stochastic area w SGD |
| Kerg i in. 2019, arXiv:1905.12080 | NeurIPS, artykuł konferencyjny | Pełny PDF konferencji | Bezpośredni prior nienormalności w RNN |
| Storm i in. 2024, DOI 10.1103/PhysRevLett.132.057301 | PRL, artykuł | Pełny PDF wydawcy | Bezpośredni prior FTLE/Cauchy–Green w DNN |
| Moon 2025, arXiv:2511.12852 | Preprint | Pełny HTML | Małe przykłady Gramians; nie modelować błędnie jako pełny depth flow |
| State Space Control Theory... 1998, DOI 10.1007/978-1-4471-1523-6_2 | Rozdział książki | Abstrakt/metadane | Historyczny prior control theory w feedforward NN |
| Ostrow i in. 2023, arXiv:2306.10168 | NeurIPS, artykuł konferencyjny | Abstrakt i strona konferencji | DSA, prior porównań dynamiki |
| Huang i in. 2025, arXiv:2510.25943 | Preprint | Abstrakt | InputDSA, wpływ wspólnego wymuszenia |
| Li i in. 2026, arXiv:2608.25315 | Preprint według agenta głównego | Pełny tekst odczytany przez agenta głównego | Kontrole asymetrii miejsc dla order swap |
| Hazelden i in. 2026, arXiv:2609.40292 | NeurReps 2026 workshop według agenta głównego | Pełny tekst odczytany przez agenta głównego | Bardzo bliski prior Green operator + wielozadaniowość |

Weryfikacje Li/Hazelden są świadomie przypisane agentowi głównemu; ich surowe źródła znajdują się w głównym pakiecie badań, nie w plikach web_1–7 tego podraportu. Brak identycznego wyniku w 15 zapytaniach SciSpace nie dowodzi braku wcześniejszej pracy. Wszystkie prognozy dla SmallViT w kartach są propozycjami badań, a nie zaobserwowanymi wynikami.

