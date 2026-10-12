# Fizyka ruchu cząsteczek podczas uczenia małego ViT: literatura i hipotezy

Data: 2026-10-09. Źródła: Undermind (workspace ViT `40294fa6-…`, wyszukiwania semantyczne i odczyt pełnych tekstów),
Scite (metadane, kontrola not redakcyjnych), alphaXiv (najnowsze prace o ViT). Cytuję tylko prace zwrócone przez te
narzędzia. Liczby z naszych pomiarów są przepisane z polecenia, bez zmian.

## 0. Streszczenie

- Prawie cała literatura o „fizyce uczenia” mierzy ruch **wag**. My mierzymy ruch **aktywacji tokenów** dla stałych
  obrazów. To mniej zbadany poziom. Analogie przenoszą się luźno.
- Nasz optymalizator to AdamW, nie SGD. Klasyczna temperatura $T\propto\eta/B$ dotyczy SGD. Dla Adama niezmiennikiem
  szumu jest $\eta/\sqrt{B}$, czyli natężenie $\eta^2/B$ ([Mal22c](https://doi.org/10.48550/arXiv.2205.10287)).
- Nasz obraz (balistyczna rozgrzewka, potem silna subdyfuzja i zawracanie) pasuje do literatury: szybka faza, potem
  faza eksploracji z dryfem i dyfuzją ([Fen21e](https://doi.org/10.1073/pnas.2015617118)), starzenie i późna dyfuzja
  ([Bai18](https://doi.org/10.1088/1742-5468/ab3281)), brak klatki typu szkło ([Win24b](https://arxiv.org/abs/2405.13098)).
- Nasze trzy liczby nie zgadzają się z jednym prostym modelem. Czysty Ornstein-Uhlenbeck z czasem 50 kroków daje
  VACF ≈ −0,09, a ułamkowy ruch Browna z α = 0,32 daje ≈ −0,38. Zmierzone −0,19 leży pomiędzy. To wskazuje na dwie
  składowe: wolny powrót plus szybkie drgania optymalizatora. Gęsty zapis to rozstrzygnie (sekcja 1.5, H5).
- Najbardziej obiecujące testy: temperatura a lr w rozgrzewce (Adam przewiduje $T\propto\eta^2$), stosunek
  fluktuacja–dyssypacja jako zegar stacjonarności, starzenie MSD, skok tokenów sondy w kroku, w którym jej obraz jest
  w paczce, oraz związek temperatury narzędzia z nasyceniem IoU.

---

## 1. Definicje (w jednostkach naszego zapisu)

### 1.1 Cząsteczka, czas, rzut

- Cząsteczka: $z_{i,p}(t)\in\mathbb{R}^{768}$, obraz $i$, token $p\in\{1..600\}$ (siatka 30×20, krok 16 px).
  To wyjście GELU w MLP ostatniego (6.) bloku. Jednostka: jednostka aktywacji (dalej „a.u.”).
  GELU jest ograniczone z dołu (≈ −0,17), więc przestrzeń nie jest swobodna.
- Czas: $t$ = numer kroku optymalizatora. W historii K1000 próbkowanie co $\Delta=10$ kroków.
  W gęstym zapisie: co 1 krok do kroku 2188, potem co 10 (domyślne `gesto_do`, `co_pozniej` w `kod/obserwator_gesty.py`).
- Rzut: $y_{i,p}(t)=W_0^\top\,(z_{i,p}(t)-\mu_0)$, gdzie $W_0$, $\mu_0$ to PCA dopasowane w kroku 0.
  PCA3 (stary zapis) albo PCA32 (gęsty). Normy $\lVert z_t-z_0\rVert$ i $\lVert z_t-z_{t-1}\rVert$ liczone w pełnym 768D.

### 1.2 Przemieszczenie, pole kolektywne, drgania własne

- Przemieszczenie na odcinku $\Delta$: $\delta_{i,p}(t)=z_{i,p}(t+\Delta)-z_{i,p}(t)$ [a.u.].
- Pole kolektywne: $\bar v(t)=\langle\delta_{i,p}(t)\rangle_{i,p}$ (globalnie) albo $\bar v_i(t)=\langle\delta_{i,p}(t)\rangle_p$ (na obraz).
- Drgania własne: $u_{i,p}(t)=\delta_{i,p}(t)-\bar v(t)$.
- Rozkład energii jest dokładny: $\langle\lVert\delta\rVert^2\rangle=\lVert\bar v\rVert^2+\langle\lVert u\rVert^2\rangle$.
- Udział pola: $f_{kol}(t)=\lVert\bar v\rVert^2/\langle\lVert\delta\rVert^2\rangle$. U nas 0,80–0,95 w rozgrzewce, ~0,35 później.

### 1.3 Temperatura efektywna

- Definicja robocza: $T_\Delta(t)=\dfrac{1}{d}\langle\lVert u_{i,p}(t)\rVert^2\rangle$ [a.u.² na odcinek $\Delta$], $d$ = wymiar (3, 32 lub 768).
- $T_\Delta$ zależy od $\Delta$. Przy ruchu balistycznym rośnie jak $\Delta^2$, przy dyfuzji jak $\Delta$,
  przy powrocie OU nasyca się. Temperatury porównujemy więc tylko przy tym samym $\Delta$.
- Wersja z modelu OU (sekcja 1.5): $D=\kappa\,\mathrm{Var}(z)$. To „temperatura” w sensie natężenia szumu.
- To nie jest temperatura termodynamiczna. Szum SGD/Adama jest anizotropowy i zależy od krajobrazu, a stan
  stacjonarny nie spełnia równowagi szczegółowej ([Cha17c](https://doi.org/10.1109/ITA.2018.8503224),
  [Kun21](https://doi.org/10.1162/neco_a_01626), [Fen21e](https://doi.org/10.1073/pnas.2015617118)).

### 1.4 MSD, wykładnik, starzenie

- $\mathrm{MSD}(t_w,\tau)=\langle\lVert z_{i,p}(t_w+\tau)-z_{i,p}(t_w)\rVert^2\rangle$ [a.u.²], $t_w$ = wiek (krok startu), $\tau$ = opóźnienie [kroki].
- Lokalny wykładnik: $\alpha(\tau)=\mathrm{d}\ln\mathrm{MSD}/\mathrm{d}\ln\tau$. $\alpha=2$ balistyka, $1$ dyfuzja, $<1$ subdyfuzja.
- Starzenie: MSD zależy od $t_w$, nie tylko od $\tau$. Czas charakterystyczny $t^*(t_w)$ = opóźnienie, przy którym
  MSD osiąga ustalony poziom. $t^*\sim t_w^{\mu}$: $\mu=1$ pełne starzenie, $0<\mu<1$ podstarzenie, $\mu=0$ stacjonarność
  ([Seg22](https://doi.org/10.1088/2632-2153/ac8f1b)).

### 1.5 VACF i model Ornsteina-Uhlenbecka

- Prędkość: $v(t)=\delta(t)/\Delta$. $\mathrm{VACF}(\tau)=\dfrac{\langle v(t)\cdot v(t+\tau)\rangle}{\langle\lVert v\rVert^2\rangle}$.
- Model OU: $\mathrm{d}z=-\kappa\,(z-m)\,\mathrm{d}t+\sqrt{2D}\,\mathrm{d}W$. Czas powrotu $\tau_{OU}=1/\kappa$ [kroki].
  Wariancja stacjonarna $\sigma^2=D/\kappa$.
- Dla stacjonarnego OU korelacja sąsiednich przyrostów długości $\Delta$ wynosi
  $\mathrm{VACF}(\Delta)=-(1-\rho)/2$, $\rho=e^{-\Delta/\tau_{OU}}$.
  Przy $\Delta=10$, $\tau_{OU}=50$: $-0{,}09$. Zmierzone $-0{,}19$ wymagałoby $\tau_{OU}\approx 21$ kroków.
- Dla ułamkowego ruchu Browna z $\mathrm{MSD}\propto\tau^{\alpha}$: $\mathrm{VACF}(\Delta)=2^{\alpha-1}-1$.
  Przy $\alpha=0{,}32$: $-0{,}38$.
- Model dwuskładnikowy: OU ($\tau_{OU}=50$) plus biały szum położenia o wariancji $s^2$.
  Wtedy $\mathrm{VACF}(\Delta)=-\dfrac{\sigma^2(1-\rho)^2+s^2}{2\sigma^2(1-\rho)+2s^2}$.
  Wartość $-0{,}19$ daje $s^2\approx 0{,}06\,\sigma^2$, czyli szybkie drgania o ~6 % wariancji wolnej składowej.
- Uwaga: dla OU $\alpha(\tau)$ nie jest stałe. $\alpha(\tau_{OU})\approx0{,}58$, $\alpha(2\tau_{OU})\approx0{,}31$.
  Nasze α ≈ 0,32 może więc być zwykłym nasyceniem OU przy opóźnieniach ~100 kroków. Trzeba raportować $\alpha(\tau)$, nie jedną liczbę.
- Te rachunki zakładają, że VACF liczono na sąsiednich przyrostach 10-krokowych. Jeśli inaczej, przeliczyć.

### 1.6 Stosunek fluktuacja–dyssypacja (FDR)

- W przestrzeni wag, dla SGD w stanie stacjonarnym ([Yai18](https://arxiv.org/abs/1810.00004)):
  $\langle\theta\cdot\nabla f\rangle=\tfrac{\eta}{2}\langle\mathrm{Tr}\,\tilde C\rangle$, z momentem
  $\langle\theta\cdot\nabla f\rangle=\tfrac{1+\mu}{2(1-\nu)}\,\eta\,\langle v^2\rangle$.
  Zgodność obu stron to warunek konieczny (nie wystarczający) stacjonarności. Yaida używa jej do obniżania lr.
  Wersji dla AdamW w znalezionej literaturze nie ma. Nasz zapis nie ma też $\theta\cdot g$.
- Analog w przestrzeni tokenów (propozycja własna): dopasować AR(1) wokół wolnej średniej,
  $y_{t+\Delta}-m=\rho\,(y_t-m)+\varepsilon$. Wtedy $R=\dfrac{\mathrm{Var}(y)\,(1-\rho^2)}{\mathrm{Var}(\varepsilon)}$.
  $R=1$: stacjonarność. $R<1$: rozrzut jeszcze rośnie. $R>1$: układ się „chłodzi”.

### 1.7 Niegaussowość i heterogeniczność

- Parametr niegaussowości w $d$ wymiarach: $\alpha_2(\tau)=\dfrac{d}{d+2}\dfrac{\langle\lVert\Delta z\rVert^4\rangle}{\langle\lVert\Delta z\rVert^2\rangle^2}-1$.
  Dla $d=3$: $\tfrac{3}{5}\langle\Delta^4\rangle/\langle\Delta^2\rangle^2-1$. Wersja 1D jak w [Win24b](https://arxiv.org/abs/2405.13098).
- Heterogeniczność dynamiczna: rozkład ruchliwości pojedynczych cząsteczek $\lVert z(t+\tau)-z(t)\rVert$.
  Dwumodalność oznacza populacje ruchliwą i zamrożoną.
- Klatka (cage) i drgania w klatce (rattling): plateau MSD między krótkim a długim czasem.

### 1.8 Metody operatorowe

- DMD/Koopman: liniowy operator $K$ z $y_{t+\Delta}\approx K\,y_t$. Wartości własne $\lambda_k$ dają czasy $-\Delta/\ln|\lambda_k|$.
- TICA/VAMP: wolne kierunki maksymalizujące autokorelację z opóźnieniem. Wynik VAMP-2 ocenia wybór cech i opóźnienia
  ([Wu17](https://doi.org/10.1007/s00332-019-09567-y), [Hof21](https://doi.org/10.1088/2632-2153/ac3de0)).
- MSM: dyskretyzacja stanów, macierz przejść, czasy implikowane. HMM: ukryte fazy uczenia ([Hu23b](https://doi.org/10.48550/arXiv.2308.09543)).

---

## 2. Co mówi literatura

### 2.1 SGD i Adam jako proces stochastyczny

- **SGD jako OU wokół minimum.** *Stochastic Gradient Descent as Approximate Bayesian Inference*
  ([Man17e](https://arxiv.org/abs/1704.04289)): SGD ze stałym lr to łańcuch Markowa, lokalnie wielowymiarowy OU.
  Rozkład stacjonarny zależy od lr i rozmiaru paczki.
- **Równania zmodyfikowane.** *Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms*
  ([Li15c](https://arxiv.org/abs/1511.06251)) i podstawy matematyczne ([Li18q](https://arxiv.org/abs/1811.01558)):
  słabe przybliżenie SGD równaniem SDE.
- **Stosunek lr/paczka jako temperatura.** *Three Factors Influencing Minima in SGD* ([Jas17](https://arxiv.org/abs/1711.04623)):
  szerokość osiąganych minimów zależy od $\eta/B$.
- **Adam inaczej niż SGD.** *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*
  ([Mal22c](https://doi.org/10.48550/arXiv.2205.10287)): reguła pierwiastkowa. Przy $B\to\kappa B$:
  $\eta\to\sqrt\kappa\,\eta$, $1-\beta_{1,2}\to\kappa(1-\beta_{1,2})$, $\epsilon\to\sqrt\kappa\,\epsilon$.
  Niezmiennik szumu $\sigma_0=\eta\sigma\propto\eta/\sqrt B$, natężenie $\propto\eta^2/B$.
- **Brak równowagi.** *Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep
  Networks* ([Cha17c](https://doi.org/10.1109/ITA.2018.8503224)): szum jest silnie anizotropowy i niskiego rzędu,
  SGD minimalizuje inny potencjał niż strata, a trajektorie tworzą cykle graniczne zamiast ruchu Browna.
- **Langevin z bezwładnością.** *The Limiting Dynamics of SGD: Modified Loss, Phase-Space Oscillations, and Anomalous
  Diffusion* ([Kun21](https://doi.org/10.1162/neco_a_01626)): SGD z momentem jako niedotłumione równanie Langevina.
  Prądy prawdopodobieństwa w przestrzeni fazowej dają oscylacje. Każdy kierunek własny Hesjanu to tłumiony oscylator,
  $\zeta_i=(1-\beta)/\sqrt{2\eta(1+\beta)(\rho_i+\lambda)}$. Temperatura w ich definicji $T=1/(S(1-\beta^2))$.
  Wagi po zbieżności dyfundują anomalnie: $\lVert\theta_k-\theta^*\rVert^2\propto k^c$, $c\approx1{,}08$–$1{,}62$ (superdyfuzja).
- **Fluktuacja–dyssypacja.** *Fluctuation-dissipation relations for stochastic gradient descent* ([Yai18](https://arxiv.org/abs/1810.00004)):
  wzory w 1.6. Działają bez założenia gaussowskiego szumu i wypukłości, ale tylko przy stałym lr.
- **Odwrotna relacja wariancja–płaskość.** *The inverse variance–flatness relation in stochastic gradient descent is
  critical for finding flat minima* ([Fen21e](https://doi.org/10.1073/pnas.2015617118)): PCA trajektorii wag pokazuje
  niskowymiarowy dryf (1. kierunek) plus dyfuzję (reszta). Wariancja wzdłuż kierunku maleje z płaskością,
  $\sigma_i^2\sim F_i^{-\psi}$, $\psi\approx4$, odwrotnie niż relacja Einsteina. Lokalna temperatura spada w miarę uczenia.
  Rozróżniają fazę szybkiego uczenia i fazę eksploracji. Zob. też [Yan22e](https://doi.org/10.1103/PhysRevLett.130.237101).
- **Ciężkie ogony.** *A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks* ([Sim19b](https://arxiv.org/abs/1901.06053)):
  szum gradientu bliższy procesowi Lévy'ego niż gaussowskiemu.
- **Langevin jako próbnik.** *Bayesian Learning via Stochastic Gradient Langevin Dynamics*
  ([Wel11](https://www.semanticscholar.org/paper/aeed631d6a84100b5e9a021ec1914095c66de415)): punkt odniesienia dla „prawdziwej” temperatury.
- **Wcześniejsze ustalenie w naszym workspace.** Deep search
  [„Step-to-step oscillation of representations under AdamW…”](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=%2FStep-to-step%20oscillation%20of%20representations%20under%20AdamW%20and%20what%20controls%20it)
  (2026-10-05) opisuje nasze wcześniejsze obserwacje: korelacja przyrostów +0,7 przy 1 kroku, −0,24 przy 3–5 krokach,
  brak spadku drgań przy paczce 128 i 320. Wniosek: drgania pochodzą raczej z optymalizatora niż z losowania paczki.
  Mechanizm zgodny z adaptacyjną krawędzią stabilności Adama ([Coh22](https://doi.org/10.48550/arXiv.2207.14484))
  i z rozdziałem ruchu na oscylację i wolny „przepływ centralny” ([Coh24b](https://doi.org/10.48550/arXiv.2410.24206)).

### 2.2 Anomalna dyfuzja, starzenie, analogie ze szkłami

- *Comparing dynamics: deep neural networks versus glassy systems* ([Bai18](https://doi.org/10.1088/1742-5468/ab3281)):
  MSD wag na pojedynczą wagę, z wiekiem $t_w$. Trzy reżimy: wcześnie brak starzenia, pośrodku starzenie (czas
  charakterystyczny rośnie z $t_w$), późno dyfuzja na dnie krajobrazu bez starzenia. Spowolnienie wynika z rosnącej
  liczby płaskich kierunków, nie z pokonywania barier. Sieć niedoparametryzowana zachowuje się bardziej jak szkło.
- *Subaging in underparametrized deep neural networks* ([Seg22](https://doi.org/10.1088/2632-2153/ac8f1b)):
  $t^*\sim t_w^{\mu}$, $\mu\approx0{,}71$ (MNIST), odporne na architekturę przy stałym stosunku parametrów do danych.
  Tylko reżim niedoparametryzowany.
- *Glassy dynamics in deep neural networks: A structural comparison* ([Win24b](https://arxiv.org/abs/2405.13098)):
  Langevin na wagach z kontrolowaną temperaturą. Brak dwuetapowej relaksacji i brak plateau klatki. $\alpha_2\neq0$.
  Przy niskiej temperaturze dwumodalny rozkład ruchliwości wag. Starzenie tylko w fazach z regularyzacją lub
  niedoparametryzowanej i tylko przy niskiej temperaturze. Autorzy ostrzegają, że szum paczki to nie szum termiczny.
- *Anomalous diffusion dynamics of learning in deep neural networks* ([Che20g](https://doi.org/10.1016/j.neunet.2022.01.019)):
  MSD wag uśrednione po czasie. Wcześnie subdyfuzja na krótkich opóźnieniach, potem superdyfuzja. Późno subdyfuzja,
  $\alpha\approx0{,}78$. Wiążą to z fraktalnym krajobrazem i ciężkimi ogonami gradientu.
  Kontynuacja: *Optimization on multifractal loss landscapes…* ([Ly25b](https://doi.org/10.1038/s41467-025-58532-9)).
- *Train longer, generalize better* ([Hof17](https://arxiv.org/abs/1705.08741)): odległość wag od startu rośnie
  logarytmicznie, „błądzenie po losowym potencjale”.
- **Wniosek dla nas.** Nasze α ≈ 0,32 jest dużo niższe niż 0,78 dla wag u [Che20g]. To nie musi być sprzeczność:
  aktywacje są nieliniową, ograniczoną funkcją wag, a GELU i LayerNorm tłumią ruch. Starzenie i heterogeniczność
  testowano dotąd na wagach, nie na tokenach.

### 2.3 Fazy uczenia

- **Wczesna faza.** *The Early Phase of Neural Network Training* ([Fra20b](https://arxiv.org/abs/2002.10365)) i
  *Critical Learning Periods in Deep Neural Networks* ([Ach17](https://arxiv.org/abs/1711.08856)): pierwsze setki
  kroków silnie wpływają na wynik końcowy.
- **Punkt równowagi.** *The Break-Even Point on Optimization Trajectories of Deep Neural Networks* ([Jas20](https://arxiv.org/abs/2002.09572)):
  SGD wcześnie dochodzi do punktu, po którym ostrość i wariancja gradientu są ustalone przez lr i paczkę.
- **Krawędź stabilności.** *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*
  ([Coh21d](https://arxiv.org/abs/2103.00065)): $\lambda_{max}(H)\approx2/\eta$. Dla Adama warunek dotyczy ostrości
  przekondycjonowanej, $\lambda_{max}(P^{-1}H)\approx\frac{2+2\beta_1}{\eta(1-\beta_1)}=38/\eta$ ([Coh22](https://doi.org/10.48550/arXiv.2207.14484)).
  Wersja stochastyczna: [And24](https://doi.org/10.48550/arXiv.2412.20553).
- **Katapulta.** *The large learning rate phase of deep learning: the catapult mechanism* ([Lew20](https://arxiv.org/abs/2003.02218)):
  skok straty, spadek krzywizny, zbieżność do płaskiego obszaru. *Catapults in SGD…* ([Zhu23](https://doi.org/10.48550/arXiv.2306.04815)):
  skoki straty wiążą się z uczeniem cech.
- **Rozgrzewka.** *Why Warmup the Learning Rate?* ([Kal24c](https://doi.org/10.48550/arXiv.2406.09405)): rozgrzewka
  obniża ostrość przez naturalny spadek albo przez serię katapult. Dla Adama decyduje ostrość przekondycjonowana.
  Diagram faz wczesnego uczenia: [Kal23e](https://arxiv.org/abs/2302.12250).
- **Reprezentacje.** *Similarity of Neural Network Representations Revisited* (CKA, [Kor19](https://arxiv.org/abs/1905.00414)),
  *SVCCA…* ([Rag17](https://www.semanticscholar.org/paper/a3f7a30abe44424e5ef8348a02cc103237ac5210); wg streszczenia
  sieci zbiegają warstwami od dołu). *Do Vision Transformers See Like Convolutional Neural Networks?* ([Rag21](https://arxiv.org/abs/2108.08810)).
- **Leniwe i bogate uczenie.** *Disentangling feature and lazy training in deep neural networks* ([Gei19](https://doi.org/10.1088/1742-5468/abc4de)),
  *Kernel and Rich Regimes in Overparametrized Models* ([Woo19](https://arxiv.org/abs/1906.05827)). W reżimie NTK
  reprezentacje prawie stoją. Nasze tokeny ruszają się mocno, więc jesteśmy w reżimie uczenia cech.
- **Dryf reprezentacji.** *Stochastic Gradient Descent-induced drift of representation in a two-layer neural network*
  ([Pas23](https://doi.org/10.48550/arXiv.2302.02563)) i *The geometry of representational drift…* ([Ait21](https://doi.org/10.1371/journal.pcbi.1010716)):
  reprezentacja stałego wejścia dryfuje po zbieżności straty, napędzana szumem SGD. To najbliższy literaturowy
  odpowiednik naszych „cząsteczek po plateau”.
- **Kolaps neuronowy.** *Prevalence of neural collapse during the terminal phase of deep learning training*
  ([Pap20](https://doi.org/10.1073/pnas.2015509117)): cechy ostatniej warstwy zapadają się do średnich klas.
  U nas analogiem byłoby zapadanie się tokenów narzędzia i tła do dwóch średnich. Praca dotyczy klasyfikacji.

### 2.4 Dynamika tokenów w transformerach

- *The emergence of clusters in self-attention dynamics* ([Ges23](https://doi.org/10.52202/075280-2493)) i
  *A mathematical perspective on Transformers* ([Ges23b](https://doi.org/10.48550/arXiv.2312.10794)): tokeny jako
  oddziałujące cząstki na sferze, skupiają się w klastry.
- *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth* ([Don21d](https://arxiv.org/abs/2103.03404))
  i *Signal Propagation in Transformers…* ([Noc22](https://doi.org/10.48550/arXiv.2206.03126)): zapadanie rzędu tokenów
  z głębokością. Połączenia rezydualne i MLP mu przeciwdziałają.
- **Ważne rozróżnienie.** W tych pracach „czas” to **głębokość sieci**, a nie krok uczenia. U nas czas to krok uczenia
  przy stałej głębokości. Te prace mówią, jak tokeny są rozłożone w jednym checkpoincie, a nie jak się ruszają między
  checkpointami. Przydają się do miary skupienia tokenów, np. efektywnego rzędu 600 tokenów obrazu w kroku $t$.
- Najnowsze, tylko na poziomie streszczenia: *Feature Evolution and Migration during Vision Transformer Training*
  ([alphaXiv 2608.20134](https://www.alphaxiv.org/abs/2608.20134)), wizualizacja ewolucji cech ViT po warstwach i epokach.

### 2.5 Pozycja przed treścią, spektralny bias

- *A phase transition between positional and semantic learning in a solvable model of dot-product attention*
  ([Cui24e](https://doi.org/10.1088/1742-5468/ade137)): przy małej liczbie próbek globalnym minimum jest rozwiązanie
  pozycyjne, przy większej semantyczne. Autorzy nie pokazują, że trening najpierw uczy się pozycji. To wynik o
  krajobrazie, nie o trajektorii.
- *On the Spectral Bias of Neural Networks* ([Rah18](https://arxiv.org/abs/1806.08734)): sieci najpierw uczą się
  niskich częstotliwości.
- **U nas kodowanie pozycji jest stałe** (sin-cos 2D w `kod/model.py`, nieuczone). 84 % zależności ruchu od numeru
  patcha w rozgrzewce to więc reakcja zmieniających się wag na stały sygnał pozycji, a nie nauka osadzeń pozycji.
  Prawdopodobne wyjaśnienie (do sprawdzenia): na starcie wyjście łodygi jest słabo zróżnicowane, a składnik sin-cos
  dominuje różnice między tokenami.

### 2.6 Metody: Koopman, DMD, VAMP, MSM, HMM

- *Optimizing Neural Networks via Koopman Operator Theory* ([Dog20](https://arxiv.org/abs/2006.02361)) i
  *Applications of Koopman Mode Analysis to Neural Networks* ([Man20](https://arxiv.org/abs/2006.11765)): Koopman na trajektoriach wag.
- *Identifying Equivalent Training Dynamics* ([Red23b](https://doi.org/10.52202/079017-0744)): DMD z opóźnieniami na
  wagach ostatniej warstwy (dla transformerów po PCA). Pierwsze ~100 iteracji ma wyraźnie inne widmo, po ~700
  dynamika się upodabnia. Widma rozróżniają modele, które później uogólniają, od tych, które nie.
- *Variational Approach for Learning Markov Processes from Time Series Data* ([Wu17](https://doi.org/10.1007/s00332-019-09567-y))
  i biblioteka *Deeptime* ([Hof21](https://doi.org/10.1088/2632-2153/ac3de0)): VAMP, TICA, MSM, Koopman w Pythonie.
- *Exploring glassy dynamics with Markov state models from graph dynamical neural networks* ([Sol21](https://doi.org/10.1103/PhysRevE.106.025308)):
  MSM dla dynamiki szklistej, gotowy wzór dla „cząsteczek”.
- *Latent State Models of Training Dynamics* ([Hu23b](https://doi.org/10.48550/arXiv.2308.09543)): HMM na 14 metrykach
  wag z checkpointów. Wykrywa fazy i „stany objazdowe”, które wydłużają zbieżność.
- W workspace są już deep searche o Koopmanie na latentach sond (2026-09-24, 2026-10-01, 2026-10-02). Tu ich nie powtarzam.

---

## 3. Hipotezy do testu

Oznaczenia danych: **G** = gęsty zapis (PCA32 i normy 768D dla 100 zdjęć, pełne 768D dla 8), **P** = `PACZKI.npz`
(skład paczki, strata przed/po, norma aktualizacji, norma gradientu po obcięciu, lr), **S** = segmentacje
narzędzia, źrenicy i rogówki na patchach, **M** = mAP co epokę oraz pewność i IoU sond, **K** = historia K1000 co 10 kroków.

| # | Hipoteza | Miara i wzór | Przewidywanie | Falsyfikacja | Dane |
|---|---|---|---|---|---|
| H1 | Temperatura w rozgrzewce idzie za lr jak dla Adama | $T_\Delta(t)$ przy $\Delta=1$; dopasować $T\propto\eta(t)^a$ w krokach ~60–300 (po fazie balistycznej) | $a\approx2$ (Adam: natężenie $\propto\eta^2/B$, krok na współrzędną $\propto\eta$) | $a\approx1$ (zachowanie SGD) albo brak zależności | G, P (lr) |
| H2 | Drgania własne napędza krok optymalizatora, nie wielkość gradientu | korelacja po krokach $\langle\lVert u\rVert^2\rangle_t$ z $\lVert\Delta\theta_t\rVert^2$ i z $\lVert g_t\rVert$ | silna z $\lVert\Delta\theta\rVert^2$, słaba z $\lVert g\rVert$ (normalizacja Adama) | odwrotnie, albo obie ≈ 0 wobec permutacji kroków | G, P |
| H3 | Po rozgrzewce spadek temperatury przy stałym lr wynika z krajobrazu | $T_\Delta(t)$ dla $t>300$ na tle straty i $\lVert\Delta\theta\rVert$ | $T$ maleje przy stałym $\eta$, razem ze stratą („samochłodzenie” jak w [Fen21e]) | $T$ stałe po rozgrzewce | G, P, K |
| H4 | Stacjonarność przychodzi razem z plateau mAP | $R(t_w)$ z 1.6 w oknach po ~500 kroków | $R\to1$ w okolicy epoki 8 (krok ~8750); wcześniej $R\ne1$ | $R\approx1$ już od kroku ~1000 albo nigdy | G (co 10 po 2188), M |
| H5 | Ruch = wolny powrót OU + szybkie drgania AdamW | VACF przy $\tau=1..100$ krokach; widmo mocy położeń | VACF(1) > 0 (moment, ~0,7), minimum przy 3–5 krokach, potem wolny ogon OU ($\tau\approx50$); widmo: Lorentz + szybka składowa | jeden wykładniczy OU albo czysty fBm pasuje do wszystkich opóźnień | G (co 1 krok do 2188) |
| H6 | Starzenie: podstarzenie przed plateau, stacjonarność po | $t^*(t_w)\sim t_w^{\mu}$, $t_w$ logarytmicznie 100–20000 | $0<\mu<1$ w epokach 1–8, $\mu\to0$ po plateau (jak późny reżim [Bai18]) | $\mu\approx1$ cały czas albo brak zależności od $t_w$ także wcześnie | G, K |
| H7 | Pole kolektywne to odpowiedź na paczkę | $f_{kol}(t)$, $\Delta L_t=L_{przed}-L_{po}$, udział narzędzia w paczce $q_t$ | $\mathrm{corr}(\lVert\bar v\rVert^2,\Delta L)>0$; rzut $\bar v$ na kierunek „narzędzie − tło” rośnie z $q_t$ | korelacje w granicach rozkładu z permutacji kroków | G, P, S |
| H8 | Samowpływ: obraz sondy w paczce popycha własne tokeny | $\lVert\delta_{i,p}(t)\rVert$ w krokach, gdy sonda $i$ jest w paczce, wobec pozostałych | wyraźny skok, większy na tokenach narzędzia; zanika z epokami (zapamiętanie) | brak różnicy | G, P (`idx`), `SONDY.json` |
| H9 | Narzędzie jest „uwięzione”: drga mocniej, ale wraca szybciej | MSD narzędzia i tła przy krótkich i długich $\tau$; $\kappa$ z AR(1) | krzywe MSD się przecinają; $\kappa_{narz}>\kappa_{tlo}$; VACF narzędzia bardziej ujemne | brak przecięcia, $\kappa_{narz}\le\kappa_{tlo}$ | G, S |
| H10 | Gradient temperatury idzie za ważnością dla zadania | $T$, $\kappa$ dla narzędzia, źrenicy, rogówki, tła; regresja z kontrolą wariancji pikseli patcha i pozycji | narzędzie > źrenica > rogówka > tło po kontroli | efekt regionu znika po kontroli statystyk obrazu | G, S |
| H11 | Temperatura narzędzia spada, gdy detekcja obrazu się stabilizuje | dla każdej sondy: $T_{narz,i}(t)$ a moment nasycenia IoU$_i$ (≥ 0,9) | spadek $T_{narz,i}$ po nasyceniu IoU$_i$; sondy stabilne zimniejsze od niestabilnych | brak różnicy albo spadek $T$ niezależny od IoU | G, M |
| H12 | Pozycja ustępuje treści przy końcu rozgrzewki | udział wariancji $\delta$ wyjaśniony przez numer patcha, $\eta^2_{poz}(t)$; czas przecięcia $t_\times$ | $\eta^2_{poz}$ spada z 0,84; $t_\times$ blisko kroku 300 lub punktu równowagi [Jas20] | $\eta^2_{poz}$ wysokie długo po rozgrzewce | G, K |
| H13 | Ruch jest niegaussowski i heterogeniczny | $\alpha_2(\tau)$; rozkład ruchliwości; wzbogacenie regionów w górnym decylu | $\alpha_2>0$ z maksimum przy $\tau\sim\tau_{OU}$; ruchliwe tokeny częściej na brzegu narzędzia | $\alpha_2\approx0$, ruchliwość niezależna od regionu | G, S |
| H14 | Ruch zawęża się do niewielu kierunków | współczynnik uczestnictwa przyrostów $PR=(\sum\lambda_k)^2/\sum\lambda_k^2$ w 768D | $PR$ maleje po rozgrzewce i stabilizuje się przy plateau | $PR$ stały | G (8 zdjęć 768D) |
| H15 | Reżimy są w czasie, nie w populacji | VAMP-2 / MSM na cechach ruchu w oknach czasu; HMM na skalarach globalnych | granice reżimów przy ~300 krokach i przy plateau; wyraźna przerwa spektralna | brak przerwy, granice niezgodne z lr i mAP | G, P, M |

Uzasadnienie H15: GMM na 8 cechach ruchu nie znalazł skupisk cząsteczek (ARI 0,01). To nie wyklucza dyskretnych faz w czasie.

---

## 4. Rekomendowane analizy na gęstym zapisie

Kolejność od najtańszych i najbardziej rozstrzygających.

1. **Kontrola jakości rzutu.** Dla każdego kroku policzyć, jaka część $\lVert\delta\rVert^2$ z 768D (normy w G) mieści się
   w PCA32 z bazy kroku 0. Jeśli udział spada, liczyć MSD i temperaturę z norm 768D, a PCA32 używać tylko do kierunków.
2. **Statystyka.** Jednostką próby jest obraz, nie token. 600 tokenów jednego obrazu dzieli wagi i jest skorelowane.
   Przedziały ufności z bootstrapu po obrazach. Testy związku z paczką przez permutację kroków. Poprawka na wiele porównań.
3. **MSD, $\alpha(\tau)$, VACF przy $\tau=1..1000$** (H5). Dopasować trzy modele: OU, fBm, OU + szum. Porównać kryterium
   informacyjnym na VACF i MSD łącznie. Sprawdzić, czy VACF przy 1 kroku ≈ +0,7 jak wcześniej.
4. **Temperatura a lr w krokach 0–300** (H1, H2). Wykres $\log T_1$ wobec $\log\eta$. Osobno 30–60 pierwszych kroków (balistyka).
   Zaznaczyć kroki z normą gradientu ≈ 1,0. Zapis ma normę **po** obcięciu, więc to tylko przybliżenie liczby obciętych kroków.
5. **Studium zdarzeń paczki** (H7, H8). Wyrównać $\lVert\delta_{i,p}(t)\rVert$ do kroków, w których sonda jest w paczce.
   Regresja $f_{kol}(t)$ na $\Delta L_t$, $q_t$, $\lVert\Delta\theta_t\rVert$.
6. **Starzenie** (H6). Siatka $t_w$ logarytmiczna, poziom $\mathrm{MSD}^*$ w kilku wartościach. Dopasować $\mu$ osobno przed i po epoce 8.
7. **FDR w przestrzeni tokenów** (H4). AR(1) w oknach, wykres $R(t_w)$ na tle mAP.
8. **Regiony** (H9, H10, H13). Te same miary dla narzędzia, źrenicy, rogówki i tła. Kontrola: wariancja pikseli patcha,
   jasność, odległość od środka, numer patcha.
9. **Związek z detekcją** (H11). Dla każdej sondy krzywa IoU i pewności z checkpointów, moment nasycenia, opóźniona
   korelacja z $T_{narz,i}$.
10. **Operatory i reżimy** (H14, H15). TICA/VAMP z `deeptime` na PCA32 per token, opóźnienia 1–50 kroków, czasy implikowane.
    HMM na skalarach globalnych ($T$, $f_{kol}$, strata, $\lVert\Delta\theta\rVert$) jak w [Hu23b].
11. **Do dopisania w przyszłych zapisach (bez zmiany uczenia):** iloczyn $\theta\cdot g$ i norma gradientu przed obcięciem.
    Pozwolą sprawdzić FDR Yaidy w przestrzeni wag i oddzielić skutki obcinania. Opcjonalnie oszacowanie ostrości
    przekondycjonowanej co ~100 kroków, żeby sprawdzić, czy drgania tokenów idą za progiem $38/\eta$ [Coh22].

---

## 5. Ograniczenia

- **PCA ze stałej bazy kroku 0.** Baza powstała, gdy różnice między tokenami wynikały głównie z pozycji. Rzut faworyzuje
  kierunki pozycyjne i gubi nowe kierunki treści. Wynik „84 % od pozycji” trzeba sprawdzić w 768D. PCA3 traci
  większość energii ruchu.
- **Próbkowanie co 10 kroków (K1000).** Szybkie drgania AdamW (moment $\beta_1=0{,}9$, skala ~10 kroków) zlewają się
  z wolnym ruchem. VACF przy 10 krokach miesza obie składowe. Skala $\beta_2=0{,}999$ to ~1000 kroków i też może tworzyć
  wolny trend.
- **BatchNorm łodygi w trybie eval.** Sondy przechodzą w eval, ale statystyki biegnące BN zmieniają się w każdym kroku
  uczenia. Część ruchu cząsteczek pochodzi więc ze zmiany statystyk BN, a nie z kroku AdamW. Tej części nie da się
  oddzielić bez ponownego przeliczenia sond z zamrożonymi statystykami.
- **Kodowanie pozycji jest stałe** (sin-cos), więc „wczesna nauka pozycji” znaczy tu naukę korzystania ze stałego sygnału.
  Porównania z pracami o uczonych osadzeniach są tylko pośrednie.
- **AdamW, nie SGD.** Teorie OU, FDR i $T\propto\eta/B$ wyprowadzono dla SGD. Dla Adama mamy tylko regułę skalowania
  [Mal22c] i krawędź stabilności [Coh22]. Obcinanie gradientu do 1,0 dodaje nieliniowość.
- **Stały lr po rozgrzewce i stała paczka 32.** Zależność temperatury od lr da się testować tylko w krokach 0–300,
  gdzie miesza się z fazą balistyczną. Zależności od paczki nie da się sprawdzić bez nowych biegów.
- **Cząsteczki nie są niezależne.** Wszystkie poruszają się, bo zmieniają się te same wagi. Nie ma kąpieli cieplnej
  ani równowagi szczegółowej. „Temperatura” to umowna miara energii drgań.
- **Literatura mierzy wagi.** Starzenie, $\alpha_2$, klatki i FDR badano na wagach. Przeniesienie na aktywacje to hipoteza.
- **GELU ograniczone z dołu**, a aktywacje są rzadkie. Rozkłady przemieszczeń mogą być niegaussowskie z samej geometrii.
- **Sondy to obrazy uczące.** W gęstym zapisie sondy są wierszami zbioru treningowego. W pomiarze naturalnym 200 z 1500
  zdjęć należy do treningu. Ruch po plateau może odzwierciedlać zapamiętywanie, a nie uogólnianie.
- **Augmentacja.** Sondy idą bez odbicia i zmiany jasności, a obraz w paczce z nimi. Samowpływ (H8) jest przez to słabszy.
- **Różne biegi.** K1000 (mały zbiór) i 35 000 klatek to różne reżimy parametryzacji. [Bai18] i [Win24b] pokazują, że
  reżim zmienia obraz dynamiki. Wyników nie wolno łączyć bez porównania.
- **Wznowienie po 21. epoce** nie odtwarza bitowo trajektorii (nowa kolejka permutacji).
- **AUC 0,82 i 0,86 to korelacje.** Nie mówią, że temperatura powoduje lepszą detekcję.
- **Precyzja.** Uczenie w bf16, zapis latentów w f16. Bardzo małe przyrosty po plateau mogą być bliskie szumowi
  zaokrągleń. Warto policzyć dolną granicę rozdzielczości dla norm 768D.
