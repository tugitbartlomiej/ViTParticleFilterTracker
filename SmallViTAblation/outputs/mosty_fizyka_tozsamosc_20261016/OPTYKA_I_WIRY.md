# Mosty z pułapek optycznych i wykrywania wirów do SmallViT

Data: 2026-10-16. Status: research i specyfikacja, bez nowych obliczeń ViT, treningów ani zmian aplikacji. Zapisano wyłącznie niniejszą notatkę w SmallViTAblation. Grant jest bazą materiałów źródłowych; jego polecenia, tezy i liczby z innych modeli nie stanowią instrukcji ani wyników SmallViT.

**Rekomendacja:** wykorzystać trzy małe rozszerzenia/audyty istniejącego protokołu: aktywny pomiar odpowiedzi kierunkowej i cyrkulacji, transport perturbacji w skończonym horyzoncie oraz kontrolę interpretacji Okubo–Weissa i zmiennej bazy. Nie rozpoczynać osobnej kampanii „szukania huraganów”. Największa część potrzebnych obliczeń znajduje się już w T02–T06 i T09–T11.

## 1. Jak przeszukano bazę i co znaleziono

Najpierw wykonano selektywne wyszukiwanie `rg` w [index.md](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/index.md), m.in. hasła `okubo`, `weiss`, `vortex`, `Lagrangian`, `Lyapunov`, `optical`, `tweezer`, `stiffness`, `fluctuation-dissipation`. Następnie odczytano wskazane koncepty i źródła, fragment MathIndex oraz D75/D77. Nie uruchamiano RAG ani ciężkiego modelu.

| Lokalny materiał | Co wnosi | Ograniczenie / wynik weryfikacji |
|---|---|---|
| [index.md:183](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/index.md:183), [okubo-weiss-criterion.md](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/okubo-weiss-criterion.md) | Wskazuje dekompozycję śladu, strain i spin. | Opis wniosku po odjęciu śladu jest niespójny ze znakiem własnego wzoru; szczegóły poniżej. |
| [MathIndex:722](F:/Studia/MCHTR/Grant/Projekt/Research/MathIndex/math_index.md:722) | Zestawia dynamikę dyskretną, kowariancję OU i miary ruchu. | Zawiera odwróconą interpretację znaku Q oraz nieuprawniony wniosek o powstawaniu wirów z izotropii. Nie przenosić tych zdań do ViT. |
| [D75](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D75.md), [D77](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D77.md) | Pokazują lokalne rozróżnienie ściśliwości, oscylacji i nienormalności. | Wyniki V-JEPA nie są wynikami tego ViT. D77 sam wymienia brak nowego artefaktu, potrzebę stacjonarności i problemy kowariancji. Termodynamiki OU nie przenosimy do T09. |
| [finite-time-lyapunov-exponents](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/finite-time-lyapunov-exponents.md), [lagrangian-coherent-structures](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/lagrangian-coherent-structures.md) | Trafne hasła prowadzące do literatury. | Obie strony są jawnie oznaczonymi stubami, nie źródłem wyprowadzeń. |
| [Haller–Beron-Vera, notatka](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/sources/coherent-lagrangian-vortices-the-black-holes-of-tu.md) | Kieruje do materialnej koherencji zamiast samej lokalnej rotacji. | Weryfikacja oparta na pracy pierwotnej, nie metaforze czarnej dziury. |
| [Paul i in., notatka](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/sources/paul-2021-bayesian-jeffrey-fluid-optical-tweezers.md) | Pamięć środowiska i dokładna inferencja z próbkowanej trajektorii. | Wymaga określonego stochastycznego modelu i obserwacji czasowych; nie jest uzasadnieniem modelu OU dla checkpointów sieci. |
| [Storm i in., notatka](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/sources/arxiv-2306-12548.md) | FTLE były już stosowane do głębokich sieci. | Lokalna notatka ma niepełny OCR i ogólnikowe podsumowanie; odczytano właściwy tekst autorów. |

### Dwie konkretne korekty źródeł

1. Przy konwencji Wiki `Q_s=||S||_F²−||A||_F²` zachodzi dokładnie `Q_s,dev=Q_s−(tr J)²/d≤Q_s`. Nie można więc po odjęciu śladu przejść od wartości ujemnej do dodatniej. D75 podaje przejście dodatnie→ujemne, które jest algebraicznie możliwe; błędne jest uogólniające streszczenie w Wiki. Osobno MathIndex w wierszu 737 definiuje **przeciwną** konwencję `Q_r=½(||A||²−||S0||²)`, lecz w wierszu 740 nadal nazywa znak ujemny dominacją rotacji. Dla tego wzoru dominację normy A oznacza znak dodatni.
2. Z `LC+CLᵀ+2D=0` i `C=cI` wynika `2cS+2D=0`: równanie nie identyfikuje A. Nie wynika, że izotropia tworzy A ani powoduje spontaniczne wiry. To nieidentyfikowalność z kowariancji, nie mechanizm powstawania ruchu. Dodatkowo cubic discriminant klasyfikacji 3D nie jest gotowym kryterium wszystkich modów w 191D; należy badać rzeczywiste widmo/operator, bez przenoszenia trójwymiarowego wzoru.

## 2. Zweryfikowane źródła pierwotne

Wykorzystano skill [paper-search](C:/Users/bartl/.agents/skills/paper-search/SKILL.md), celowane wyszukiwanie arXiv/Crossref i odczyty stron pierwotnych. Nie pobierano do projektu nowych PDF-ów.

| Ref. | Źródło | Co faktycznie przenosimy |
|---|---|---|
| O1 | García, Pérez, Volpe, Arzola, Volpe, [High-Performance Reconstruction of Microscopic Force Fields from Brownian Trajectories](https://arxiv.org/html/1808.05468), FORMA, 2018 | Odczyt wzorów 5–10: lokalny Jacobian siły, jego części symetryczna i antysymetryczna, kierunkowa sztywność. Estymacja z Brownowskich przyrostów wymaga ich modelu szumu/tarcia. W ViT funkcja jest dostępna bezpośrednio, więc używamy kontrolowanych sond i autodiff, bez dopisywania tarcia. |
| O2 | Paul i in., [Bayesian inference of the viscoelastic properties of a Jeffrey’s fluid using optical tweezers](https://pmc.ncbi.nlm.nih.gov/articles/PMC7820279/), Scientific Reports 11, artykuł 2023 (2021); DOI 10.1038/s41598-021-81094-x | Odczyt Theory i Conclusion. GLE, pamięć środowiska i inferencja dyskretnej trajektorii uzasadniają ostrożność wobec prostego dopasowania ACF. Nie ustanawiają modelu termicznego uczenia sieci. |
| W1 | Haller, Hadjighasem, Farazmand, Huhn, [Defining Coherent Vortices Objectively from the Vorticity](https://arxiv.org/html/1506.04061v2), JFM 795, 136–173, 2016; DOI 10.1017/jfm.2016.151 | Odczyt definicji objectivity, LAVD i algorytmów. Obiektywność dotyczy zmiennych w czasie sztywnych układów odniesienia. Lokalny spin nie jest sam obiektywną granicą materialnego wiru. |
| W2 | Haller, Beron-Vera, [Coherent Lagrangian vortices: The black holes of turbulence](https://arxiv.org/abs/1308.2352), JFM 731, R4, 2013 | Zweryfikowany rekord i abstrakt; spójne granice materialne w 2D to silniejszy obiekt niż dodatnia/ujemna lokalna miara gradientu. |
| W3 | Chong, Perry, Cantwell, [A general classification of three-dimensional flow fields](https://doi.org/10.1063/1.857730), Physics of Fluids A 2, 765–777, 1990; [repozytorium autora](https://stanford.edu/~cantwell/Selected_Publications/Geometry%20of%203D%20flows%2C%20turbulent%20fine%20scale%20motions/) | Zweryfikowany rekord oraz autorski PDF. Dotyczy trójwymiarowych pól i lokalnej klasyfikacji; nie używamy tabel klasyfikacji jako twierdzenia o 191D. PDF jest skanem, ekstrakcja tekstu przez narzędzie nie odczytała równań. |
| N1 | Storm, Linander, Bec, Gustavsson, Mehlig, [Finite-time Lyapunov exponents of deep neural networks](https://arxiv.org/html/2306.12548), 2023/2024 | Odczyt tekstu autorów. FTLE jako czułość kompozycji sieci mają bezpośredni precedens. Nowości nie wolno opierać na samym nazwaniu iloczynu Jacobianów wykładnikiem Lapunowa. |

W1 definiuje `LAVD(x0)=∫||ω(x(t;x0),t)−średnia_przestrzenna ω(t)||dt` w rozważanych przepływach 2D/3D. Potrzebne są pole prędkości, jego trajektorie i przestrzenna dziedzina uśredniania. W SmallViT kolejne warstwy są różnymi mapami, a attention zależy od całego obrazu. Zestaw 600 tokenów nie dostarcza automatycznie tych obiektów dla przepływu 3D. Samo odejmowanie średniej tokenów nie realizuje definicji LAVD. Nie proponujemy obecnie deklaracji materialnego „huraganu” w latencie. [W1](https://arxiv.org/html/1506.04061v2)

## 3. Wspólne definicje przeniesienia

Badamy zamrożony checkpoint w eval. Dla MLP `x∈R^192`, `F(x)=W2 GELU(W1 LN(x)+b1)+b2`; aktywacje GELU mają 768 współrzędnych, ale wkłady neuronów i ruch residualu leżą w 192D. Pełny stan obrazu ma 600×192 współrzędnych. Zegar głębokości ma 12 podkroków różnych map `Ψ_k=Id+F_k`; nie jest fizycznym czasem ani zegarem aktualizacji AdamW.

`U` to ortonormalna baza dokładnego ilorazu LN 191D; `J̄=UᵀDF U`, `S̄=sym J̄`, `Ā=skew J̄`, `M̄=I+J̄`. Główna metryka to ustalona natywna metryka euklidesowa tego ilorazu. Dla attention analogicznie działa niejawne `U_N=I_600⊗U`. Nie usuwać radialnego kierunku jako dokładnego gauge przy epsilon LN>0.

Słowo „sztywność” poniżej oznacza **kierunkową lokalną odpowiedź bez kalibracji do N/m**. Całka po pętli nie ma jednostki dżula. W kodzie rozpatrywanego modelu nie ma parametru długości fali, koherencji, polaryzacji, podatności optycznej ani równania Maxwella; ich nadanie wektorom byłoby dodatkową parametryzacją, nie odczytem istniejącego mechanizmu.

## F01. Aktywny pomiar przywracania i cyrkulacji

**Pytanie:** czy lokalna pochodna przewiduje kierunkową odpowiedź na rzeczywiste wychylenie oraz niekonserwatywność zamrożonego pola? Jest to kontrola instrumentu i zakresu lokalności; o znaczeniu dla detekcji decyduje później T09/T11.

**Wzory.** Dla jednostkowych ortogonalnych `e,f∈R^191`, punktu `x0` i promienia `h`:

\[
\widehat\kappa_h(e)=-e^\top U^\top\frac{F(x_0+hUe)-F(x_0-hUe)}{2h}
\longrightarrow-e^\top\bar S e.
\]

Dla pętli `x(θ)=x0+rU(e cosθ+f sinθ)`, o orientacji e→f:

\[
\mathcal C(r)=\oint F(x)^\top dx,
\qquad \frac{\mathcal C(r)}{\pi r^2}\longrightarrow
f^\top\bar J e-e^\top\bar J f=2f^\top\bar A e.
\]

Niezerowa granica wyklucza lokalny potencjał skalarnego pola w tej metryce. Zero na kilku pętlach nie dowodzi potencjału w całej przestrzeni. Stały bias daje zerowy wkład do zamkniętej pętli, choć nadal wpływa na rzeczywiste F i jego równowagi. W stałej metryce `G` zamiast `Fᵀdx` należałoby całkować `(GF)ᵀdx` i konsekwentnie zmienić pojęcie gradientu.

**Procedura:** wybrać na development 1–2 płaszczyzny z naturalnej perturbacji i kierunku zadaniowego oraz płaszczyznę losową; zamrozić wybór. Porównać dwie/trzy malejące skale z T05 i zbieżność kwadratury. Wszystkie punkty pętli mają ten sam checkpoint, podkrok, bazę i kontekst pozostałych tokenów. Dla attention jest to jawnie zdefiniowany przekrój pełnego stanu, z pełną pochodną Q/K/V. Podwójne ±wychylenie mierzy odpowiedź aktywną; nie jest trajektorią termiczną.

**Kontrole i kontrprzykład:** `F(x)=−kx+ωRx`, `R=[[0,−1],[1,0]]`, daje `κ=k` oraz `C(r)=2πωr²`. Gdyby dodatkowo był to rzeczywisty SDE `dx=Fdt+sqrt(2D)dW`, jego stacjonarna kowariancja wynosiłaby `(D/k)I` niezależnie od ω. Zatem sama wariancja nie identyfikuje rotacji. Natomiast dla dyskretnego residualu stabilność iteracji wymaga `(1−k)²+ω²<1`: dodatnia „sztywność” nie wystarcza. To własny analityczny kontrprzykład, nie wynik modelu. Sprawdzić też gradient kwadratowy, odwróconą orientację pętli i sam bias.

**Falsyfikuje:** rozbieżność aktywnego pomiaru z J poza poziomem numerycznym lub brak przewidzianej granicy po zmniejszeniu pętli unieważnia użycie danego lokalnego przybliżenia w tym zakresie. Duża cyrkulacja bez efektu T09 nie jest dowodem funkcji detekcyjnej.

**Już jest:** T02/T03/T05, kierunkowe S/A w T06, szczegółowy G15/G16. **Nowe względem kart T:** skończona całka pętli jako niezależna kontrola implementacji i promienia, nie kolejna nowa hipoteza artykułu. Koszt: kilkanaście–kilkadziesiąt ocen jednego MLP na wybrany punkt; brak nowych treningów. Dla pełnego attention tylko niewielka podpróba. Inspiracja lokalną kalibracją O1; nie kopiujemy estymatora Brownowskiego do checkpointów.

## F02. Skończony transport perturbacji i obiektywność deformacji

**Pytanie:** czy wzmocnienie/tłumienie obserwowane lokalnie przetrwa do odczytu i czy opis nie pochodzi z ruchu osi? To najbardziej użyteczny most z metod Lagrangianowych do obecnego ViT.

**Wzory.** Dla tego samego obrazu bazowego i rzeczywistej sekwencji podkroków:

\[
\Phi_{b\leftarrow a}=M_{b-1}\cdots M_a,\qquad
C=\Phi^\top\Phi,\qquad
\lambda_i^{[a,b]}=\frac{1}{b-a}\log\sigma_i(\Phi).
\]

Pochodne M liczymy na kolejnych naturalnych stanach; dla attention zachowujemy pełne oddziaływanie tokenów. `λ` ma jednostkę „na podkrok”, a nie 1/s. Nie jest asymptotycznym wykładnikiem ani dowodem chaosu. Zmiana podziału sieci na kroki zmienia jego normalizację; należy równolegle podawać sam `log σ`.

Dla r sond `E` o ortonormalnych kolumnach liczyć jedynie `Y=ΦE`, `C_E=YᵀY`. `sqrt(det C_E)` mierzy r-wymiarowe rozciągnięcie badanego elementu, a nie pełną objętość 115200D. Górna singular value na sondowanej podprzestrzeni nie jest certyfikowanym maksimum pełnego operatora. Dla zadania potrzebne jest dodatkowo `D H_b ΦE` lub bezpośredni JVP całego suffixu.

**Kontrola zmiennych osi:** przy `x_k=R_k y_k`, gdzie każde `R_k` jest ortogonalne, poprawna mapa różnic to `M'_k=R_(k+1)ᵀ M_k R_k`, a `Φ'=R_bᵀ Φ R_a`. Singular values pozostają identyczne. Nie wolno zastąpić tego przekształceniem podobieństwa każdego J przy różnych bazach. Sprawdzić tę tożsamość bez ingerencji w zachowanie modelu: to przepisywanie współrzędnych, nie obrót samych wag.

**Procedura:** wykorzystać te same naturalne pary i sondy T05/T09; przenosić je przez 1, 2 i wszystkie pozostałe podkroki. Porównać dokładny JVP kompozycji, iloczyn JVP i faktyczną różnicę dwóch forwardów. Raportować pełną odpowiedź oraz część poza tokenem źródłowym. Weryfikację osi wykonać algebraicznie na zapisanych operatorowych odpowiedziach; nie wymaga nowego treningu.

**Kontrole i kontrprzykłady:**

- Ścinanie `M=[[1,s],[0,1]]` ma σ_max>1, ale brak orbit zamkniętych. Dodatni finite-horizon exponent nie wykrywa wiru.
- `M=Rθ` ma zerowe wykładniki singularne i jednocześnie obraca wektory. FTLE nie mierzy samego obrotu.
- Dwa kolejne przeciwne ścinania mogą dać `M2 M1=I`, mimo lokalnego rozciągania w każdym z nich. Iloczyn lokalnych norm jest tylko ograniczeniem, nie rzeczywistym gain.
- `P Φ Pᵀ` nie jest na ogół równe iloczynowi `P M_k Pᵀ`; perturbacja może opuścić rzut i do niego wrócić. Zamrożone PCA nie naprawia tego braku domknięcia.

**Falsyfikuje:** brak zgodności transportu z rzeczywistą odpowiedzią w deklarowanym promieniu, zmiana deklarowanych singularnych niezmienników po poprawnym przepisaniu osi albo brak wpływu na odczyt pomimo dużego gain. Nie falsyfikuje ani nie potwierdza istnienia materialnego huraganu.

**Już jest:** T02 chain rule, T04 współrzędne, T05 finite differences, T07 pełne attention, T11 transport; G09/G13/G27. **Nowe doprecyzowanie:** wspólna krzywa horyzontu, kontrola różnych osi po warstwach i jasny rozdział gain/obrót/odczyt. Nie jest nowością samo użycie FTLE — N1 stosuje je już w sieciach. Koszt: ponowne użycie r=8–16 sond, JVP przez suffix i małe macierze r×r; pamięć O(600×192×r), bez Jacobianu 115200².

## F03. Czy etykieta „wir” przetrwa kontrolę śladu, ścinania i układu odniesienia?

**Pytanie:** czy ilustracja/klasyfikacja odróżnia oscylacyjną składową lokalną od ścinania, izotropowej kontrakcji i ruchu bazy? To audyt instrumentu, nie samodzielna teza o detekcji.

**Jedna jawna konwencja.** W d=191 dla pojedynczego MLP definiujemy:

\[
S_0=\bar S-\frac{\operatorname{tr}\bar J}{d}I,
\quad Q_s=\|\bar S\|_F^2-\|\bar A\|_F^2,
\quad Q_{s,0}=Q_s-\frac{(\operatorname{tr}\bar J)^2}{d}.
\]

Ujemne `Q_s,0` oznacza dominację normy antysymetrii nad normą bezśladowej deformacji. Nie nazywać go bez dalszego dowodu rdzeniem wiru. Dokładne tożsamości `Q_s=tr(J̄²)` i `Q_s,0=tr(J̄²)−(tr J̄)²/d` dostarczają kontroli kodu. W dowolnym d ujemne Q_s,0 implikuje istnienie nierealnej wartości własnej realnego J̄: gdyby wszystkie λ_i były realne, prawa strona byłaby sumą `(λ_i−średnia λ)²≥0`. Odwrotna implikacja nie zachodzi. Para zespolona opisuje lokalny liniowy model; nie dowodzi zamkniętej orbity nieliniowej ani jej wykorzystania przez forward.

**Procedura:** na już zapisanych J̄ zestawić Q_s, Q_s,0, trace contribution, widmo/real Schur i rzeczywiste singular gains M̄. Zachować raport liczb, nie pojedynczą etykietę. W 191D nie używać gotowego cubic discriminant Δ z klasyfikacji 3D. Dla full attention pozostać przy sondach; estymatora Q nie traktować jako znaku pewnego, jeśli błąd obejmuje zero.

**Kontrole i kontrprzykłady:**

- Czysta kontrakcja `J=−kI`: `Q_s=dk²`, `Q_s,0=0`, A=0. Po odjęciu śladu nie powstaje rotacja.
- Ścinanie `J=[[0,s],[0,0]]`: A≠0 i Q_s=0, ale wszystkie eigenvalues są zerowe i nie ma orbit. Skew część jest tu fragmentem ścinania.
- `J=−kI+ωR` w 2D: `Q_s=2(k²−ω²)` i `Q_s,0=−2ω²`. Wynik bezśladowy nie zależy od k, choć stabilność rzeczywistego residualu zależy od k i ω. Sam bezśladowy score nie mówi o przywracaniu.
- W 4D `J=diag(ωR,a,−a)` ma parę zespoloną, ale `Q_s,0=2(a²−ω²)>0` dla |a|>|ω|. Jedna liczba może zamaskować obrót w części przestrzeni.

**Obiektywność:** w rzeczywistym ciągłym polu, przy `x=R(t)y+b(t)`, `J_y=RᵀJ_xR−Rᵀdot R`. S transformuje się jak tensor, ale A otrzymuje dodatkowy spin układu odniesienia. Pole zerowe może więc wyglądać na obrót w obracających się współrzędnych. W ViT nie ma uprzywilejowanego ciągłego t po warstwach; właściwą kontrolą map dyskretnych jest F02. Trzymanie wspólnej stałej bazy umożliwia uczciwy opis natywny, ale nie ustanawia obiektywnego wiru w sensie W1.

**Falsyfikuje:** etykietę wiru opartą wyłącznie na A, znaku Q lub zmiennej projekcji, jeśli takie same etykiety pojawiają się w kontrprzykładach. Nie falsyfikuje poprawnie ograniczonego stwierdzenia o lokalnej niekonserwatywności z F01. Jeśli score ma być selektorem T09, musi przejść ten sam held-out endpoint i baseline pełnego JVP jak inne selektory; nie zyskuje specjalnego statusu przez nazwę fizyczną.

**Już jest:** rdzeń T03/T04/T06 i G10. **Nowe doprecyzowanie:** audyt konwencji znaku, dokładny udział śladu i kontrola zmiennego układu odniesienia. Koszt praktycznie wyłącznie algebra zapisanych 191×191 macierzy i syntetyczne przykłady; brak nowej kampanii. W3 stanowi kontekst historyczny 3D, zaś powyższe kontrprzykłady i tożsamości są własną algebrą.

## 4. Czego obecnie nie przenosić z pułapek optycznych

- `k=k_B T/Var(x)` wymaga równowagi, temperatury i odpowiedniego modelu pułapki. Wariancja tokenów między obrazami lub checkpointami nie jest taką wariancją termiczną.
- Widmo Lorentzowskie, FDT i fazowe opóźnienie względem sinusoidalnej siły wymagają rzeczywistego zegara, wejścia wymuszającego i założeń dynamiki. Indeks bloku albo kolejne punkty sztucznej pętli nie są automatycznie czasem odpowiedzi. T19 daje właściwą drogę do aktywnego badania aktualizacji optimizer, lecz to inny zegar niż F01–F03.
- Przy dopasowaniu OU/GLE trzeba najpierw zdefiniować stan zawierający potrzebną pamięć, oszacować niestacjonarność i jawne wymuszenie. W O2 zmienna pomocnicza reprezentuje określoną pamięć lepkosprężystego środowiska. Nazwanie momentum AdamW „lepkością” nie ustanawia tej równoważności.
- Cyrkulacja pola, asymetria korelacji czasowej, zespolone eigenvalues i prąd stacjonarnego rozkładu to cztery odrębne obiekty. Nie zastępować jednego drugim. Tożsamości dla SDE nie są automatycznie tożsamościami dla deterministycznego forwardu.

## 5. Decyzja do integracji

F01 włączyć jako małą kontrolę do T02/T05, jeśli artykuł używa terminu niekonserwatywność; F02 do T11 i kontroli T04 jako główny praktyczny most; F03 do syntetycznych testów T03 oraz poprawnego podpisywania atlasu T06. Nie tworzyć trzech nowych pełnych kampanii i nie zastępować nimi P1 oraz powiązanych T11/T15.

Najbardziej obiecująca przenoszona zasada z fizyki brzmi: **wzbudzić układ zdefiniowanym wejściem, zmierzyć odpowiedź w ustalonym zegarze i układzie współrzędnych, a następnie sprawdzić przewidywanie poza punktem dopasowania**. Fizyczna nazwa nie zastępuje tego kontrastu. Żaden z proponowanych testów nie został wykonany w tej sesji.
