# Scite: fizyczne odpowiedniki mikroskopu SmallViT
Data: 2026-10-15. Zakres: literatura o polach sił, pułapkach, dynamice stochastycznej, próbkowaniu i redukcji wymiaru. Nie uruchamiano nowych eksperymentów ani nie zmieniano kodu modelu.

## Wniosek

Najmocniejszym uzasadnieniem fizycznym jest **wirtualny mikroskop lokalnego pola transportu reprezentacji**, korzystający z metod badania niekonserwatywnych pól sił, stabilności i identyfikacji procesów stochastycznych. To może być samodzielny temat publikacji, jeżeli pomiary przewidują skutki interwencji i wynik detektora. Obecne dane nie uzasadniają jeszcze tezy, że tokeny podczas uczenia są rzeczywistym odpowiednikiem niezależnych cząstek Browna w pułapkach optycznych.

Najbliższy odpowiednik dla przepływu w głębokości to sterowany, na ogół niekonserwatywny układ dyskretny. Dla trajektorii checkpointów jest nim nieliniowa obserwacja wspólnego procesu optymalizacji, która może wykazywać dryf, pamięć, skorelowany szum i lokalne przybliżenie OU. Rozróżnienie tych dwóch zegarów jest warunkiem poprawnej interpretacji. Wniosek ten jest syntezą własną na podstawie konstrukcji modelu i źródeł [1–9].

## Co sprawdzono i jakie są ograniczenia źródeł

- 27 wywołań `search_literature`: 139 trafień wraz z powtórzeniami, 120 unikalnych rekordów.
- 18 wywołań `read_fulltext`: 13 stron rzeczywistego pełnego tekstu z 11 prac lub wersji, 3 odpowiedzi z samym abstraktem i 2 bez czytelnego tekstu.
- Jeden graf cytowań: 80 krawędzi, 83 węzły, limit osiągnięty. Pokrycie dla FORMA było dobre, dla Moyses i in. (2015) niskie; nie wyciągano wniosków z braku krawędzi.
- Osiem dodatkowych zapytań o retraction/correction/erratum/concern dla 23 DOI. Żadne nie zwróciło rekordu. Metadane prac zwykle pomijały pole `editorialNotices`. To oznacza **brak znalezionej informacji w indeksie Scite**, a nie gwarancję, że żadna informacja redakcyjna nie istnieje.
- Surowe odpowiedzi, parametry wyszukiwania, strony pełnych tekstów i indeks odczytów znajdują się obok tego raportu.
- Wszystkie cytaty poniżej pochodzą z rzeczywiście zwróconych tekstów. Cytat z abstraktu wyraźnie oznaczono. Brak pełnego tekstu nie został przedstawiony jako jego lektura.
- Eksport Scite pomija Jana Gieselera jako pierwszego autora tutorialu [1]. Autorstwo poprawiono według strony zespołu autorskiego: https://softmatterlab.org/2021/03/09/aop_2021_13_1_74_241/. Eksport surowy zachowano bez zmian. Rok artykułu Reshniak i Webster to 2021 w cytowaniu czasopisma, mimo publikacji online 31 grudnia 2020.

Wyszukiwania szerokie Scite dawały również odległe tematy, dlatego liczby trafień nie są liczbą prac popierających analogię. To przegląd ukierunkowany, nie kompletna metaanaliza ani twierdzenie o wyczerpaniu literatury.

## 1. Co jest dokładną matematyką modelu

Przy ustalonych wagach blok MLP wyznacza pole
`F(x)=sum_c GELU(w_c^T LN(x)+b_c) v_c+b_2`
i jednorazową aktualizację `T(x)=x+F(x)`. Rozkład na 768 podpisanych wkładów neuronów jest dokładny. Dla podkroku uwagi analogicznie otrzymuje się sumę wkładów innych tokenów, z zależnymi od stanu współczynnikami.

W tym modelu pozycja tokenu jest 192D, a 768 liczb po GELU jest zestawem aktywacji ukrytej warstwy. Ta korekta wymiaru jest istotna dla interpretacji jako cząstki. Kontekst lokalny: `OpisUkładu/01_OPIS_UKLADU.md`, sześć bloków pre-LN, siatka 600 tokenów.

Wyrażenie `GELU(a)` może być ujemne. W fizyce optycznej natężenie światła nie jest ujemne. Istnieją ujemne siły optyczne i siły ciągnące, ale wynikają z transferu pędu i struktury pola, a nie z ujemnego natężenia [1, 18]. Dlatego „reflektor” jest dobrym symbolem wkładu sterowanego stanem; „jasność lasera” wymaga pokazania znaku, np. dwóch kolorów oraz osobnej wielkości `|z_c|`. Nie należy obcinać ujemnych wkładów w równaniu.

Pojedynczy neuron nie musi wytwarzać potencjału. Bez LayerNorm jego lokalny Jacobian ma postać `phi'(w^T x+b) v w^T`; jest symetryczny tylko w szczególnych konfiguracjach, np. przy współliniowości w i v. Z LayerNorm pojawia się dodatkowy Jacobian normowania. Zatem brak globalnej energii potencjalnej jest naturalny dla tej architektury, a nie błędem modelu.

## 2. Dlaczego analogia do optyki jest użyteczna, ale nie literalna

Tutorial optycznych pęset [1] opisuje cząstkę utrzymywaną w pobliżu stabilnego położenia przez rzeczywiste siły gradientowe i rozpraszania, przy znanej lepkości i temperaturze. FORMA [2] odtwarza lokalny Jacobian pola z przemieszczeń cząstki i osobno identyfikuje części konserwatywne oraz niekonserwatywne. Mikroskop SmallViT ma tu przewagę metodologiczną: pole można wywołać bezpośrednio w zadanych punktach, więc jego wartość i pochodne nie muszą być zgadywane z samej trajektorii.

**Dowód tekstowy z FORMA:** „it probes simultaneously the conservative and non-conservative components of the force field” [2, pełny tekst, Results/Introduction]. Ta sama praca dopuszcza charakteryzowanie także niestabilnych punktów równowagi. Dlatego narzędzie nie musi wykazywać istnienia pułapki, aby jego pomiary były wartościowe.

Należy rozróżniać:
- lokalną dekompozycję liniową `J=S+A`, gdzie `S=(J+J^T)/2`, `A=(J-J^T)/2`;
- globalną dekompozycję Helmholtza pola, wymagającą określenia dziedziny, warunków brzegowych i sposobu rozwiązania;
- prawdopodobieństwowy prąd stanu stacjonarnego, wymagający trajektorii procesu stochastycznego oraz oszacowania gęstości i dyfuzji.

Te trzy obiekty nie są zamienne. Duże A jest dowodem lokalnej niesymetrii odpowiedzi, lecz samo nie jest pomiarem wiru Browna, produkcji entropii ani pracy mechanicznej.

## 3. Stabilność: najważniejsza korekta interpretacji pola

Dla hipotetycznej dynamiki ciągłej `dx/dt=F(x)` stabilność liniowa punktu `F(x*)=0` zależy od rzeczywistych części wartości własnych J. Dla rzeczywistego kroku resztowego `x_next=x+F(x)` zależy od wartości własnych `I+J`; asymptotyczna stabilność liniowa wymaga `rho(I+J)<1`. Literatura o residual networks i dyskretyzacji Eulera podkreśla właśnie odmienność stabilności przepływu oraz jego dyskretyzacji [10].

Własne wyprowadzenie dla lokalnej perturbacji delta:
`||delta_next||^2-||delta||^2 = delta^T(2S+J^T J)delta`.
Ujemny kierunek S nie musi więc kurczyć się po pełnym kroku, jeżeli składnik drugiego rzędu `J^T J` jest istotny. Największa wartość singularna `I+J` mierzy maksymalne lokalne wzmocnienie w jednym kroku.

Ujemne `log|det(I+J)|` oznacza kurczenie objętości lokalnej, nie kurczenie wszystkich kierunków. Przykład własny: macierz diag(2,0.2) ma wyznacznik 0.4, ale jeden kierunek rośnie dwukrotnie. Zatem opis „pełne 192D kurczy objętość, a promień spektralny przekracza 1” jest spójny i nie dowodzi pułapki.

Dodatkowo pre-LN daje dokładną niezmienniczość względem dodania stałej do wszystkich współrzędnych pojedynczego tokenu: `LN(x+a*1)=LN(x)`. Dla samego MLP `J*1=0`, więc `(I+J)*1=1`. Ścisła kontrakcja całej przestrzeni w tym podkroku jest wykluczona przez kierunek neutralny. Analiza stabilności może sensownie dotyczyć ilorazu przez tę symetrię albo podprzestrzeni obserwowalnej. Wniosek pochodzi bezpośrednio z równania LayerNorm; nie wymaga nowego eksperymentu.

Brak zera pola w lokalnej komórce oznacza brak znalezionego lokalnego stałego punktu w przeszukanym obszarze. Nie wyklucza stałego punktu poza nim, cyklu, ograniczenia w obszarze większym, pułapki ruchomej ani stabilizacji przez sekwencję zmiennych bloków. Dla 12 różnych podkroków należy analizować iloczyn właściwych Jacobianów w kolejności przejścia. Wielokrotne iterowanie jednego bloku byłoby osobnym, sztucznym eksperymentem, ponieważ oryginalna sieć wykonuje go raz.

## 4. Wiry Browna i pułapki niekonserwatywne

Moyses i in. [3] piszą: „Brownian vortexes arise in time-independent force fields that include at least one point of stable equilibrium.” Ich definicja wymaga szumu i stacjonarnego prądu prawdopodobieństwa. Sun i in. [4] opisują to samo rozróżnienie między równowagą mechaniczną bez szumu a stanem poza równowagą z szumem. Deterministyczne przejście przez sześć różnych bloków nie spełnia tych warunków automatycznie.

Bardzo ważny nowy punkt: Li, Cao i Ng [5] wykazują, że niesymetryczne siły mogą destabilizować pułapkę, ale również stabilizować pewne siodła. Cytat: „a particle in an originally unstable saddle potential can be stabilized by an appropriate nonconservative force”. Dlatego proporcja norm A/S około 0.86 nie jest uniwersalnym kryterium „brak pułapki”. Potrzebne są pełna dynamika, tłumienie lub mapa dyskretna i warunki stabilności. Wynik tej pracy nie oznacza, że takie zjawisko już zaszło w SmallViT.

Twierdzenie Ashkina i Gordona [6] mówi konkretnie o małej cząstce dielektrycznej utrzymywanej wyłącznie siłą rozpraszania światła. Nie jest ogólnym zakazem stabilnych punktów dowolnego neuronowego pola wektorowego. Cytowanie go jako fizycznego kontekstu jest poprawne; zastosowanie jako rozstrzygający dowód braku pułapki w ViT wymagałoby wykazania jego założeń, których obecna architektura nie spełnia.

Szerszym językiem niż optyka jest niesymetryczna odpowiedź układów aktywnych. Odd elasticity [17] pokazuje znaczenie antysymetrycznej odpowiedzi i zależności pracy od drogi. To inspiracja do mierzenia obiegów lokalnego pola, nie uzasadnienie nazwania tokenów ciałem sprężystym: brak tu zdefiniowanych naprężeń, odkształceń i rzeczywistej energii.

## 5. Uczenie: OU jest hipotezą modelu efektywnego

Mandt, Hoffman i Blei [7] uzasadniają przybliżenie SGD procesem OU przy konkretnych założeniach: w przybliżeniu gaussowskim szumie gradientowym, lokalnie stałej jego kowariancji, kontrolowanym błędzie dyskretyzacji i lokalnie kwadratowym koszcie. Nie jest to twierdzenie, że każde uczenie AdamW, każda warstwa i każda trajektoria tokenu są OU.

W SmallViT stan odczytywanego tokenu jest `x_j(t)=f_j(image;theta_t)`. Nawet gdyby pełny stan optymalizatora był dobrze opisany procesem Markowa, jego nieliniowy odczyt nie musi być procesem Markowa. AdamW zawiera pamięć momentów, więc sam theta nie opisuje całego stanu algorytmu. Projekcja do 3D lub PCA dodatkowo usuwa zmienne; formalizm Mori–Zwanziga wskazuje, skąd powstają wtedy człony pamięci i efektywny szum [8, 9].

W przybliżeniu pierwszego rzędu `delta x_j = G_j delta theta`. Wtedy
`Cov(delta x_j,delta x_k) ≈ G_j Cov(delta theta) G_k^T`.
Oznacza to wspólne, skorelowane wymuszenie tokenów. Liczba 600 nie daje 600 niezależnych replik procesu. Jeżeli stosuje się przybliżenie SDE i nieliniową zmianę obserwabli, dochodzi także człon dryfu Itô zależny od drugich pochodnych. „Szum niezależnie kopie każdy koloid” nie jest opisem wynikającym z implementacji.

Weight decay jest prostym składnikiem aktualizacji wag, ale nie staje się automatycznie sprężyną `-k(x-x*)` w reprezentacji. Wpływ przechodzi przez G_j, normowania i wspólnie zmieniającą się głowę.

## 6. Dlaczego antykorelacja i MSD nie rozstrzygają o pułapce

**Kontrola matematyczna, bez nowego eksperymentu:** jeżeli pozycje `X_n` są niezależnymi próbkami o wariancji V, to
`corr(X_(n+1)-X_n, X_(n+2)-X_(n+1))=-1/2`.
Wspólny punkt w dwóch kolejnych przyrostach wystarcza do tak silnej antykorelacji.

Dla stacjonarnego OU próbkowanego co delta, którego dyskretna korelacja wynosi `a=exp(-delta/tau)`, dokładnie
`corr(delta X_n,delta X_(n+1))=-(1-a)/2`.
Obserwacja około -0.415 jest zgodna z pewnymi modelami OU, ale nie identyfikuje sama siły przywracającej. Potrzebna jest jednoczesna zgodność korelacji pozycji, warunkowego dryfu, wariancji innowacji i rozkładów przejścia. Nie należy przeliczać jednej zbiorczej wartości z nieregularnych odstępów na pojedynczy czas relaksacji.

Praca Owensa [11], dostępna tutaj tylko jako abstrakt, wskazuje że odejmowanie średniej ruchomej lub dopasowanego trendu może wprowadzać „spurious features in the Fourier spectrum, correlation function, and power spectrum”. To bezpośrednio dotyczy obecnego detrendingu. Istnieje komentarz Kennedy'ego z 1980 r. do tej pracy (DOI 10.1029/JA085iA01p00219), lecz narzędzie nie udostępniło tekstu; nie przypisano mu treści ani stanowiska.

Plateau MSD występuje również dla niezależnych próbek ze stacjonarnego rozkładu: `MSD(lag)=2V` dla dodatnich lagów. Wprowadzone filtrem ograniczenie zmienności może więc wyglądać jak uwięzienie. Nie jest to dowód, że wynik użytkownika jest artefaktem; jest to konkretny model zerowy, który musi zostać odrzucony.

## 7. Próbkowanie, sztywność i temperatura

FORMA [2] zaleca częstość próbkowania znacznie większą od charakterystycznej częstości pułapki. Pérez-García i in. [12] pokazują systematyczne obciążenia standardowych estymatorów przy niskiej częstości i podają skorygowane wzory. Cytat Smart Citation z tej pracy o prostym FORMA: „giving rise to wrong estimates”. Lade [13; tylko abstrakt w odczycie] niezależnie analizuje wpływ skończonego odstępu na współczynniki Kramersa–Moyala.

Jeśli checkpoint jest co 100 kroków, a deklarowany powrót trwa 8–50 kroków, dane nie rozdzielają bezpośrednio tego powrotu. Parametry można czasem estymować z dokładnego modelu przejścia, lecz przy zanikającej korelacji problem staje się słabo identyfikowalny. Potrzebne są przedziały ufności, analiza profilu likelihood i test odzyskiwania parametrów dla identycznego harmonogramu checkpointów oraz filtra.

Relacja `k∝1/Var(x)` wymaga stałej, znanej skali szumu i odpowiedniego modelu równowagowego. W OU wielowymiarowym kowariancja spełnia równanie Lapunowa; sama wariancja nie oddziela dryfu od dyfuzji. Li i in. [5] dodatkowo pokazują zawodność prostej ekwipartycji w układach niesymetrycznych.

Stały learning rate nie oznacza stałej temperatury efektywnej. Feng i Tu [14] mierzą kierunkowo zależny i zmienny w przestrzeni wag szum SGD. Chaudhari i Soatto [15] oraz Zhang i in. [16] analizują konsekwencje anizotropii i roli wielkości batcha. Przeniesienie tych modeli na AdamW wymaga osobnej weryfikacji preconditionera, pamięci momentów, clippingu, danych i reprezentacji. Na razie należy pisać „amplituda i kowariancja fluktuacji”, a nie mierzone stopnie temperatury.

## 8. Pomiary, które zrobiłyby z analogii testowalny program

1. **Kalibracja pola w pełnym wymiarze.** Zachować aktualny bieg; na zapisanych checkpointach porównać autodiff J z perturbacjami skończonymi dla kilku promieni. Raportować 192D, przekrój 3D i zmianę bazy osobno. Sprawdzić wszystkie istotne podkroki, nie jeden token.
2. **Stabilność właściwej mapy.** Podawać singular values i spektrum `I+J`, `log|det(I+J)|`, kierunki neutralne LN i propagację perturbacji przez pełne 12 podkroków. Nie nazywać lokalnego trace J zmianą skończonej objętości.
3. **Walidacja interwencjami.** Perturbować stan w kierunkach ekspansji, kontrakcji i niezerowej lokalnej cyrkulacji; sprawdzić, czy przewidziane wzmocnienie i wpływ na detekcję występują na nieużytych obrazach. Porównać z perturbacjami losowymi o tej samej normie i prostymi miarami czułości.
4. **Test falsyfikacji OU na istniejących danych.** Przepuścić przez identyczny detrending i harmonogram próbkowania: niezależny szum wokół trendu, random walk, OU z różnymi tau, ARMA oraz proces z pamięcią. To proponowane kontrole, nie wykonane eksperymenty.
5. **Nowe krótkie biegi gęste, dopiero jako osobny eksperyment.** Zapis co krok w kilku oknach, osobne losowania batchy od wspólnego checkpointu, porównanie ze ścieżką pełnego batcha; obserwować także momenty AdamW. Daje to replikacje do rozdzielenia dryfu i szumu.
6. **Model przejścia bez wymuszania efektu filtrem.** Dopasować latentny wolny trend i innowacje wspólnie, uwzględniając rzeczywiste odstępy czasu; porównać przewidywanie poza próbką z OU, VAR/ARMA i modelem pamięci. Sprawdzać białość, gaussowskość, heteroscedastyczność i zależność od historii.
7. **Sterowane testy mechanizmu.** W oddzielnych nowych biegach zmieniać clipping, weight decay, batch size i learning rate; dla uczciwego porównania nie przypisywać zmiany wariancji pojedynczemu czynnikowi bez kowariancji gradientów. Co najmniej kilka ziaren i klas tokenów.
8. **Prądy prawdopodobieństwa dopiero po kalibracji modelu.** Oszacować warunkowy dryf, dyfuzję i rozkład w oknie z uzasadnioną lokalną stacjonarnością. Frishman i Ronceray [19] dają narzędzia i granice informacyjne identyfikacji sił z zaszumionych trajektorii. Ehrich [20] ostrzega, że częściowa obserwacja może ukrywać nieodwracalność. Brak wiru na PCA3 nie jest dowodem równowagi całego modelu.

Do odczytu „pracy” z zamkniętej pętli w latencie można zacząć od bezwymiarowej całki `integral F·dx` w wybranym, jawnie zadanym iloczynie skalarnym. Jest to miara niekonserwatywności reprezentacji, nie energia w dżulach. Należy zbadać wrażliwość na skalowanie współrzędnych, ponieważ geometria euklidesowa latentu nie jest samoczynnie fizyczną metryką.

## 9. Najbardziej obronny temat publikacji

**Propozycja:** *A virtual force microscope for residual vision transformers: separating depthwise transport from optimizer-driven representation fluctuations.*

Rdzeń nowości powinien zawierać: wierną dekompozycję wkładów; zgodność przewidywań lokalnego Jacobianu z kontrolowaną perturbacją; ocenę informacji traconej w 3D; rozdzielenie dwóch zegarów; falsyfikowalne modele fluktuacji uczenia. Wynik negatywny „ładnie wyglądająca pułapka w projekcji nie ma stabilnego odpowiednika w pełnej mapie” jest potencjalnie wartościowy, jeśli będzie systematyczny i przewiduje zachowanie modelu.

Optyczne pęsety mogą stanowić inspirację pomiarową i język wizualizacji. Mocniejsze tytuły o „fizycznych pułapkach”, „temperaturze ViT” lub „szkle” wymagałyby dodatkowego rozdzielenia sił, szumu, pamięci i skali czasu. Sama obecność trajektorii i rozkładu S+A nie jest nową teorią fizyczną.

## References

[1] Gieseler, J., Gomez-Solano, J. R., Magazzù, A., et al. (2021). Optical tweezers — from calibration to applications: a tutorial. *Advances in Optics and Photonics, 13*(1), 74–241. https://doi.org/10.1364/AOP.394888. Pełny tekst, pierwsza strona 8000 znaków; autorstwo skorygowane według strony autorów.

[2] Pérez García, L., Donlucas Pérez, J., Volpe, G., Arzola, A. V., & Volpe, G. (2018). High-performance reconstruction of microscopic force fields from Brownian trajectories. *Nature Communications, 9*. https://doi.org/10.1038/s41467-018-07437-x. Pełny tekst oraz graf cytowań.

[3] Moyses, H. W., Bauer, R. O., Grosberg, A. Y., & Grier, D. G. (2015). Perturbative theory for Brownian vortexes. *Physical Review E, 91*, 062144. https://doi.org/10.1103/PhysRevE.91.062144. Pełny tekst.

[4] Sun, B., Lin, J., Darby, E., Grosberg, A. Y., & Grier, D. G. (2009). Brownian vortexes. *Physical Review E, 80*, 010401. https://doi.org/10.1103/PhysRevE.80.010401. Metadane i abstrakt; definicję sprawdzono też w [3].

[5] Li, X., Cao, Y., & Ng, J. (2024). Non-Hermitian non-equipartition theory for trapped particles. *Nature Communications*. https://doi.org/10.1038/s41467-024-46058-5. Dwie strony pełnego tekstu, 16 000 znaków.

[6] Ashkin, A., & Gordon, J. P. (1983). Stability of radiation-pressure particle traps: an optical Earnshaw theorem. *Optics Letters, 8*, 511. https://doi.org/10.1364/OL.8.000511. Metadane i abstrakt.

[7] Mandt, S., Hoffman, M. D., & Blei, D. M. (2017). Stochastic gradient descent as approximate Bayesian inference. *Journal of Machine Learning Research, 18*. Wersja sprawdzona: https://arxiv.org/abs/1704.04289. Dwie strony pełnego tekstu i wyszukiwanie założeń. Oficjalna publikacja: https://jmlr.org/papers/v18/17-214.html.

[8] Venturi, D., & Li, X. (2023). The Mori–Zwanzig formulation of deep learning. *Research in the Mathematical Sciences*. https://doi.org/10.1007/s40687-023-00390-2. Abstrakt wersji czasopismowej oraz pełny tekst preprintu https://arxiv.org/abs/2209.05544.

[9] Lin, Y. T., Tian, Y., Pérez, D., & Livescu, D. (2023). Regression-based projection for learning Mori–Zwanzig operators. *SIAM Journal on Applied Dynamical Systems*. https://doi.org/10.1137/22M1506146. Metadane i abstrakt.

[10] Reshniak, V., & Webster, C. G. (2021). Robust learning with implicit residual networks. *Machine Learning and Knowledge Extraction, 3*(1), 34–55. https://doi.org/10.3390/make3010003. Fragmenty pełnego tekstu; rok bibliograficzny sprawdzony u wydawcy.

[11] Owens, A. J. (1978). On detrending and smoothing random data. *Journal of Geophysical Research, 83*, 221. https://doi.org/10.1029/JA083iA01p00221. Wyłącznie abstrakt oraz graf cytowań.

[12] Pérez-García, L., Selin, M., Ciarlo, A., et al. (2023). Optimal calibration of optical tweezers with arbitrary integration time and sampling frequencies: a general framework [Invited]. *Biomedical Optics Express*. https://doi.org/10.1364/BOE.495468. Pełny tekst i Smart Citation o błędach prostego FORMA.

[13] Lade, S. J. (2009). Finite sampling interval effects in Kramers–Moyal analysis. *Physics Letters A*. https://doi.org/10.1016/j.physleta.2009.08.029. Wyłącznie abstrakt.

[14] Feng, Y., & Tu, Y. (2021). The inverse variance–flatness relation in stochastic gradient descent is critical for finding flat minima. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.2015617118. Fragmenty pełnego tekstu.

[15] Chaudhari, P., & Soatto, S. (2018). Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks. *Information Theory and Applications Workshop*. https://doi.org/10.1109/ITA.2018.8503224. Pełny tekst; wnioski o cyklach zależą od przyjętych założeń, nie przeniesiono ich automatycznie na AdamW.

[16] Zhang, Y., Saxe, A., Advani, M., & Lee, A. A. (2018). Energy–entropy competition and the effectiveness of stochastic gradient descent in machine learning. *Molecular Physics*. https://doi.org/10.1080/00268976.2018.1483535. Fragmenty pełnego tekstu.

[17] Scheibner, C., Souslov, A., Banerjee, D., et al. (2020). Odd elasticity. *Nature Physics*. https://doi.org/10.1038/s41567-020-0795-y. Pełny tekst.

[18] Sukhov, S., & Dogariu, A. (2017). Non-conservative optical forces. *Reports on Progress in Physics*. https://doi.org/10.1088/1361-6633/aa834e. Metadane i abstrakt; kontekst sił niesprowadzalnych do gradientu natężenia.

[19] Frishman, A., & Ronceray, P. (2020). Learning force fields from stochastic trajectories. *Physical Review X, 10*, 021009. https://doi.org/10.1103/PhysRevX.10.021009. Pełny tekst.

[20] Ehrich, J. (2021). Tightest bound on hidden entropy production from partially observed dynamics. *Journal of Statistical Mechanics: Theory and Experiment*. https://doi.org/10.1088/1742-5468/ac150e. Pełny tekst.

Dodatkowy zweryfikowany punkt wyjścia: Pesce, G., Volpe, G., De Luca, A. C., Rusciano, G., & Volpe, G. (2009). Quantitative assessment of non-conservative radiation forces in an optical trap. *EPL, 86*, 38002. https://doi.org/10.1209/0295-5075/86/38002. Jest odpowiedni jako historyczny przykład ilościowej oceny niekonserwatywności, lecz szczegółowy argument raportu oparto na dostępnych pełnych tekstach [2, 3, 5, 12, 19].

