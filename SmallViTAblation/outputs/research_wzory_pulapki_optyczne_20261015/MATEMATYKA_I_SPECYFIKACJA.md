# Wzory z fizyki pułapek optycznych: matematyka i specyfikacja dla SmallViT

Data: 2026-10-15. Status: **research i specyfikacja, bez implementacji i treningu**. Punktem odniesienia jest kod SmallViT oraz wcześniejszy audyt: outputs/research_fizyczny_mikroskop_20261015/math_audit.md. Poniższe protokoły są proponowanym przeniesieniem metod pomiarowych. Nie są opisem nowych wykonanych eksperymentów.

## 1. Co warto przenieść

Najbardziej użyteczny transfer z pęset optycznych obejmuje **lokalną odpowiedź na kontrolowane przesunięcie, cyrkulację pola, anizotropię odpowiedzi i staranną identyfikację szumu**. W sieci pole bloku jest dostępne bezpośrednio, więc jego pochodne można policzyć dokładniej niż odtwarzać je ze swobodnej trajektorii. Związek z zadaniem trzeba następnie sprawdzić na rzeczywistym wyjściu detektora.

Rekonstrukcja FORMA jest przykładem fizycznego pomiaru lokalnego pola i jego składników, obejmującego również niestabilne otoczenia. W SmallViT przenosimy pytanie o lokalną odpowiedź, lecz zamiast estymować nieznaną siłę z ruchu Browna, różniczkujemy znaną mapę sieci. [Pérez García i in., 2018](https://www.nature.com/articles/s41467-018-07437-x).

| Metoda | Zegar | Obiekt | Najważniejszy wynik |
|---|---|---|---|
| M1. Podpisane wkłady neuronów i czułość zadania | jeden podkrok | 768 wektorów MLP | które wkłady zmieniają detekcję, a które się znoszą |
| M2. Cyrkulacja po zamkniętej pętli | zamrożony checkpoint | lokalne pole resztowe | lokalna niekonserwatywność w jawnej metryce |
| M3. Ruch radialny i styczny | podkrok; osobno uczenie | geometria przed LayerNorm | zmiana kierunku odczytu kontra zmiana skali i niewidoczne przesunięcie |
| M4. Kierunkowa sztywność i kontrakcja mapy | jeden podkrok | J oraz I+J | różnica między ujemnym gradientem lokalnym a stabilizacją rzeczywistego kroku |
| M5. Odpowiedź impulsowa i Cauchy–Green | pozostała głębokość sieci | pełna propagacja perturbacji | co jest wzmacniane, tłumione i rozpraszane między tokenami |
| M6. Obserwowalność względem zadania | od miejsca interwencji do głowy | Jacobian końcowego odczytu | które zaburzenia są widoczne dla detektora |
| M7. OU z rzeczywistymi odstępami i wspólnym wymuszeniem | aktualizacje treningu | proces odczytu sondy | czy po kontrolach pozostaje statystyczna relaksacja i jaka jest jej skala |
| M8. Pary interwencji i niesymetria wpływu | podkrok i jego dalszy ciąg | dwa neurony albo dwa tokeny | współdziałanie w detekcji oraz kierunkowość oddziaływania |

Najpierw warto rozważyć M1, M4, M5 i M6, bo łączą ścisłą matematykę istniejącego modelu z detekcją. M2 i M3 wyjaśniają geometrię. M7 wymaga najwięcej założeń o obserwacji i stanie ukrytym. M8 daje test przyczynowy wybranych hipotez z pozostałych metod.

## 2. Wspólny kontrakt matematyczny

### 2.1. Stan, wkład i czas

Dla jednego obrazu:
- N=600 tokenów;
- stan strumienia rezydualnego x_j należy do R^192;
- z_j należy do R^768 i zawiera podpisane aktywacje po GELU;
- X należy do R^(600×192), czyli po wektoryzacji do R^115200;
- \(\Psi_k(X)=X+F_k(X)\) to jeden z12 podkroków;
- \(J_k=D_XF_k\), \(M_k=D_X\Psi_k=I+J_k\).

Aktywacje768D pozostają pełnoprawną reprezentacją tokenu. Nie są jednak stanem, do którego dodawany jest wektor wyjściowy MLP. Wszystkie poniższe interwencje określają dokładnie, którą z tych dwóch przestrzeni dotyczą.

Zegar głębokości jest indeksem podkroku. Zegar uczenia jest numerem aktualizacji AdamW. Numer zapisanej klatki historii nie jest automatycznie numerem aktualizacji. Wzorów obu zegarów nie łączymy w jeden iloczyn Jacobianów.

### 2.2. Dokładna symetria pre-LN: kluczowa kontrola ujemna

Dla \(g=(c_1\mathbf1,\ldots,c_N\mathbf1)\), gdzie każdy c_j jest dowolnym skalarem:

\[
LN(x_j+c_j\mathbf1)=LN(x_j).
\]

W zamrożonym badanym modelu, po wstrzyknięciu perturbacji do strumienia przed blokami:

\[
F_k(X+g)=F_k(X),\quad \Psi_k(X+g)=\Psi_k(X)+g.
\]

Wynika stąd
\[
J_kg=0,\qquad M_kg=g.
\]

Ostateczna LayerNorm usuwa ten sam składnik, zatem końcowe wyjście detektora O spełnia

\[
O(X+g)=O(X),\qquad D O\,g=0.
\]

To dokładna symetria także przy epsilon>0, a nie przybliżona własność dla dużych norm. Dotyczy do600 niezależnych kierunków, po jednym na token. **Ścisła asymptotyczna pułapka w całej przestrzeni strumienia rezydualnego jest więc niemożliwa** dla powtarzania zamrożonych map pre-LN: istnieją neutralne przesunięcia, a ewentualne zera nie są izolowane w tych kierunkach.

Właściwy obiekt geometryczny może być ilorazem przez tę symetrię. Niech
\[
C=I_d-\mathbf1\mathbf1^\top/d,\quad Q=I_N\otimes C.
\]
Reprezentantem klasy jest QX. Dynamika ilorazowa \(\bar\Psi(QX)=Q\Psi(X)\) jest dobrze zdefiniowana, bo wybór przesunięcia g nie zmienia jej wyniku. Nie wolno przy tej okazji usuwać kierunku radialnego jako dokładnej symetrii: przy epsilon>0 nie jest on dokładnie neutralny, a resztowe dodawanie zmienia geometrię między blokami.

Praktyczny test przyszłej implementacji: wstrzyknięcie c_j·1 musi zachować rzeczywiste wyjście detektora do tolerancji numerycznej; JVP musi potwierdzić Jg=0 i Mg=g. To kontrola znacznie mocniejsza niż wizualna zgodność na wykresie3D.

### 2.3. Metryka i jednostki

F jest aktualizacją reprezentacji na podkrok, nie siłą w niutonach. Iloczyn F·dx ma jednostkę kwadratu jednostki reprezentacji na umowny podkrok. Po stałym skalowaniu \(\hat x=x/s,\hat F=F/s\) można raportować liczbę bezwymiarową, lecz nie wolno oznaczać jej jako energii w dżulach.

Wyniki euklidesowe są niezmiennicze względem wspólnego ortogonalnego obrotu współrzędnych; nie są automatycznie niezmiennicze względem dowolnej anizotropowej zmiany skali. Jeżeli x'=Ax, to zachowanie tej samej geometrii wymaga transformacji metryki:
\[
G'=A^{-\top}GA^{-1}.
\]
Wtedy \(dx^\top Gdx\) oraz \(F^\top Gdx\) pozostają takie same. Przestawienie osi z pozostawieniem wszędzie G=I oznacza wybór innego pomiaru.

Dla każdej metody trzeba zapisać: punkt interwencji, jednostkę kroku, sposób centrowania, metrykę, skalę s i to, czy są one stałe podczas różniczkowania. Rzut3D jest jedynie ograniczeniem obserwacji. Nie zastępuje pełnego stanu.

### 2.4. Warunki obliczeniowe

Dla proponowanych pomiarów:
- zamrożone wagi i bufory, tryb eval;
- użycie rzeczywistych modułów net.norm i bloków, bez ręcznego podstawiania epsilon;
- brak float16 w stanach, gradientach i różnicach; float32 jako minimum, float64 do małych kontroli;
- w uwadze ponowne przeliczenie całego qkv, jeżeli interwencja dotyczy rzeczywistego tokenu;
- kwantyfikacja błędu przy \(\varepsilon,\varepsilon/2,\varepsilon/4\), z rozróżnieniem zakresu liniowego i szumu numerycznego;
- osobne wyniki raw, ilorazowe Q i projekcyjne3D;
- bez materializacji pełnej macierzy115200×115200; użycie JVP, VJP oraz małej liczby kierunków próbnych.

Zamrożenie parametrów do uczenia nie oznacza wyłączenia różniczkowania względem wstrzykniętego stanu. Otaczające całe obliczenie no_grad uniemożliwiłoby wymagane pomiary.

## 3. M1 — Podpisane wkłady neuronów i ich znaczenie dla detekcji

**Wzór.** Dla MLP:
\[
z_{jc}=\operatorname{GELU}(w_c^\top LN(x_j)+b_{1c}),\qquad
u_{jc}=z_{jc}v_c,\qquad F_{M,j}=\sum_cu_{jc}+b_2.
\]
Wskaźnik znoszenia wkładów:
\[
\chi_j=\frac{\|\sum_cu_{jc}\|}{\sum_c\|u_{jc}\|}.
\]
Niski chi oznacza znoszenie wektorów. Nie oznacza małej wrażliwości zadania.

Dla skalarnego odczytu g zależnego od końcowej odpowiedzi detektora, przy interwencji \(z_{jc}\mapsto\alpha z_{jc}\):
\[
\left.\frac{\partial g}{\partial\alpha}\right|_{\alpha=1}
=a_j^\top u_{jc},\qquad
a_j=\nabla_{x_j^+}g.
\]

**Dokładny obiekt i zegar.** Jeden neuron c i token j w ustalonym MLP, na ustalonym obrazie i checkpoincie. Interwencja jest w aktywacji po GELU, przed fc2. Zegar to pojedynczy podkrok i rzeczywisty dalszy ciąg sieci. Wyzerowanie neuronu w jednym tokenie i wyzerowanie kanału we wszystkich600 tokenach to różne interwencje.

**Minimalny algorytm.**
1. Zachować wejście MLP i z, policzyć u oraz bias osobno.
2. Jednym przejściem wstecznym uzyskać a dla wybranego g.
3. Policzyć wszystkie768 iloczynów a·u bez768 osobnych uruchomień.
4. Dla uprzednio wybranych neuronów wykonać skończone interwencje, np. alpha=0 oraz alpha=1±epsilon, a następnie pełne dekodowanie.

**Kontrole.** Suma u+b2 musi odtworzyć rzeczywistą aktualizację. Porównać z losowymi kierunkami o tej samej normie, neuronami o podobnej aktywacji i wkładem biasu. Znak GELU zachować. Używać u surowych do interwencji; \((z-\bar z)v\) jest wkładem względem ruchomego środka chmury i nie odpowiada automatycznie lokalnej ablacji.

**Wniosek o detekcji.** Można wskazać neurony, których wzmocnienie lub osłabienie poprawia określony odczyt. Sam wzrost logitu jest niewystarczający: równolegle należy ocenić ramkę, fałszywe detekcje i wynik całego zbioru. Gradient jest lokalną prognozą; wpływ ablacji potwierdza skończona interwencja.

**Granica transferu.** To dokładny rozkład architektury i kontrolowana odpowiedź na „wiązkę”. Nie opisuje intensywności światła, polaryzacji ani koherencji optycznej.

**Który reflektor przywraca, a który obraca.** Rozkład można pogłębić poza samo znoszenie wektorów. Dla \(\sigma'_c=\operatorname{GELU}'(a_c)\), \(D_L=D LN(x)\), jednostkowych e,f:
\[
J_c=\sigma'_c v_cw_c^\top D_L,\qquad
\kappa_c(e)=-\sigma'_c(e^\top v_c)(w_c^\top D_L e),
\]
\[
\Gamma_c(e,f)=\sigma'_c[(f^\top v_c)(w_c^\top D_L e)
-(e^\top v_c)(w_c^\top D_L f)].
\]
Zachodzi dokładnie \(\kappa(e)=\sum_c\kappa_c(e)\) i \(\Gamma(e,f)=\sum_c\Gamma_c(e,f)\). Bias fc2 nie wnosi pochodnej. Dla wszystkich768 neuronów wystarczą produkty macierzy W1 z D_L e oraz D_L f i iloczyny v_c z e,f; nie potrzeba768 Jacobianów. To bezpośrednia mapa roli każdego reflektora: przywracanie i lokalny obrót w jawnie wybranych kierunkach. Znaki GELU' są istotne; sama jasność z_c nie określa tej roli.

## 4. M2 — Lokalna cyrkulacja po zamkniętej pętli

Badania sił w rzeczywistych pęsetach pokazują, że pole optyczne może mieć składnik niekonserwatywny. Sam pomiar takiego składnika i obserwacja stacjonarnego prądu cząstek są jednak różnymi pomiarami. [Wu i in., 2009](https://arxiv.org/abs/0903.2590), [Roichman i in., 2008](https://arxiv.org/abs/0804.0730).

**Wzór.** Dla małego prostokąta o bokach a,b w ortonormalnych kierunkach e_i,e_j, zorientowanego najpierw wzdłuż e_i, potem e_j:
\[
\Gamma_\square=\oint_\square F(x)^\top dx,
\]
\[
\frac{\Gamma_\square}{ab}\ \longrightarrow\
e_j^\top J e_i-e_i^\top J e_j
=2e_j^\top A e_i,\qquad A=(J-J^\top)/2.
\]
Dla metryki stałej G całkujemy \(F^\top Gdx\), a pochodną porównujemy z \(GJ\), nie z samym J.

**Dokładny obiekt i zegar.** Pole jednego podkroku w zamrożonym checkpoincie. Dla MLP przesuwamy token j w pełnych192 wymiarach wzdłuż pętli, pozostali stoją. Dla self-attention każda próba ponownie oblicza qkv całej konfiguracji. Jeżeli badane jest dotychczasowe query-only field, nazwa i zakres wyniku muszą to jawnie zaznaczać.

**Minimalny algorytm.**
1. Wybrać dwa kierunki w przestrzeni Q; mogą to być kierunki przypadkowe, czułości głowy lub lokalnej największej antysymetrii.
2. Próbkować każdą krawędź pętli, użyć kwadratury trapezowej lub Simpsona.
3. Porównać \(\Gamma/(ab)\) z dwiema pochodnymi kierunkowymi JVP.
4. Zmniejszać pętlę oraz zwiększać liczbę punktów kwadratury osobno.

Nie potrzeba pełnego J. Dwie pochodne kierunkowe wystarczą do lokalnej granicy.

**Kontrole.** Odwrócenie orientacji zmienia znak; dodanie stałego biasu lub zamrożonego dryfu nie zmienia całki po zamkniętej pętli. Pole gradientowe kwadratowego potencjału musi dać zero; proste pole ścinania może dać wynik niezerowy mimo braku orbit kołowych. Porównać różne płaszczyzny i rzut3D z pełnym osadzeniem pętli. Zero w jednej płaszczyźnie nie dowodzi globalnej zachowawczości.

**Wniosek o detekcji.** Sama Gamma mówi o geometrii pola. Aby powiązać ją z zadaniem, badać po tych samych punktach pętli końcowe logity i ramki oraz wpływ małych perturbacji wybranych kierunków. Korelacja Gamma z jakością detekcji wymaga kontroli normy pola, warstwy i etapu uczenia.

**Jednostki.** To cyrkulacja aktualizacji w przestrzeni reprezentacji. Po normalizacji stałą skalą może być bezwymiarowym wskaźnikiem niekonserwatywności. Nie jest pracą w dżulach ani produkcją entropii. Domknięta pętla jest protokołem interwencji, nie samorzutną trajektorią tokenu.

**Krótki kontrprzykład analityczny.** Niech w2D \(F(x)=-kx+\omega R x\), gdzie \(R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\), k>0, a szum SDE ma amplitudę \(\sqrt{2D}\). Kowariancja stacjonarna wynosi \(\Sigma=(D/k)I\) niezależnie od omega, lecz cyrkulacja po okręgu o promieniu r, przeciwnie do ruchu wskazówek zegara, wynosi \(2\pi\omega r^2\). Sama kowariancja nie wykryje obrotu. Hipotetyczna iteracja resztowa ma dodatkowy, odrębny warunek stabilności \((1-k)^2+\omega^2<1\). Wzory wyprowadzono algebraicznie; nie uruchamiano symulacji. Wszystkie punkty pętli muszą używać tego samego checkpointu, bloku, kontekstu i metryki.

## 5. M3 — Ruch radialny i styczny w geometrii pre-LN

**Wzór.** Niech \(r=Cx,\rho=\|r\|,n=r/\rho\). Dla małego przesunięcia delta:
\[
\delta\mu=\mathbf1^\top\delta/d,\quad
\delta\rho=n^\top\delta,\quad
\delta n=\frac{(C-nn^\top)\delta}{\rho}.
\]
Rozkład:
\[
\delta=\delta\mu\,\mathbf1+(n^\top\delta)n+(C-nn^\top)\delta.
\]
Trzy składniki to niewidoczne przesunięcie średniej, zmiana promienia oraz ruch styczny.

Dla \(s^2=\rho^2/d+\epsilon\):
\[
D LN(x)=\Gamma\left(C/s-rr^\top/(d s^3)\right).
\]
W szczególności:
\[
D LN(x)n=\Gamma n\,\epsilon/s^3,\qquad
D LN(x)e=\Gamma e/s\quad(e\perp\mathbf1,n).
\]

**Dokładny obiekt i zegar.** Surowy stan192D przed konkretną LayerNorm. Dla głębokości delta jest prawdziwym skokiem od podkroku. Dla uczenia delta jest różnicą tego samego stanu tego samego obrazu między rzeczywistymi checkpointami; wyniki obu zegarów raportujemy oddzielnie.

**Minimalny algorytm.**
1. Wyznaczyć mu,r,rho,n bez PCA.
2. Rozłożyć rzeczywisty skok albo kontrolowaną perturbację.
3. Policzyć zmianę kierunku dokładnie z n(x+delta), a następnie porównać z liniowym wzorem.
4. Dla kierunków radialnego i stycznych policzyć rzeczywiste JVP końcowego odczytu.

Całość lokalnej geometrii ma koszt liniowy w192; koszt powiązania z zadaniem to kilka JVP dalszego ciągu sieci.

**Kontrole.** Perturbacja c·1 nie zmienia LN. Kierunek radialny musi wykazywać skończoną zależność od epsilon; nie wolno z góry przyjąć, że odpowiedź jest dokładnie zerowa. Wykluczyć lub osobno raportować rho bliskie zeru. Sprawdzać normy w jednostkach x oraz po LN; wybór zależy od pytania. Kąt w3D nie zastępuje kąta w pełnych192 wymiarach.

**Wniosek o detekcji.** Można sprawdzić, czy poprawę odczytu niesie głównie obrót reprezentacji czy zmiana jej skali oraz ile pozornego „ruchu” przypada na dokładnie niewidoczną symetrię. Wzrost normy nie jest automatycznie wzrostem informacji, a tłumienie zmian kąta przez dużą rho nie jest dowodem masy lub bezwładności mechanicznej.

## 6. M4 — Kierunkowa sztywność a stabilność skończonego kroku

W pęsetach lokalną odpowiedź często opisuje się macierzą gradientu siły; metody takie jak FORMA rekonstruują ją z trajektorii. Tutaj macierz jest pochodną znanego podkroku, więc należy wykorzystać tę przewagę, zachowując różnicę między siłą ciągłą a skończonym krokiem sieci. [Pérez García i in., 2018](https://www.nature.com/articles/s41467-018-07437-x).

**Wzór.** Dla jednostkowego kierunku e:
\[
\kappa_e=-e^\top S e,\quad S=(J+J^\top)/2.
\]
Dodatnie kappa oznacza różniczkowy składnik przywracający w tym kierunku dla interpretacji ciągłej. Dla rzeczywistej mapy:
\[
q_e=\|(I+J)e\|^2-1
=2e^\top Se+\|Je\|^2.
\]
q_e<0 oznacza lokalne zmniejszenie długości tej perturbacji w jednym kroku. Dodatnie kappa samo nie wystarcza.

W ilorazie przez symetrię pre-LN, dla e=Qe:
\[
\bar M=Q(I+J)Q,\quad
\bar q_e=\|\bar M e\|^2-1
=2e^\top J e+\|QJe\|^2.
\]

**Dokładny obiekt i zegar.** Jeden rzeczywisty podkrok, przy bazowej konfiguracji wszystkich tokenów. Dla MLP J jest blokowo diagonalny po tokenach; dla uwagi należy uwzględnić odpowiedź wszystkich tokenów na wstrzyknięcie. To lokalny pomiar różnic między dwiema trajektoriami, nie dowód osiągania zera pola.

**Minimalny algorytm.**
1. Obliczyć Je przez JVP dla wybranych e.
2. Wyznaczyć kappa i q bez diagonalizacji.
3. Do znalezienia skrajnych kierunków S użyć operatora \(Sv=(Jv+J^\top v)/2\), realizowanego przez JVP+VJP; do największej czułości mapy użyć \(\bar M^\top\bar M\).
4. Porównać prognozowaną zmianę odległości z pełnym przejściem dwóch stanów X±epsilon e.

**Kontrole.** Dla kierunków g symetrii pre-LN trzeba dostać kappa=0, q=0, Mg=g. Używać kilku amplitud i kontrolować nienormalność. Przykład J=-3I ma dodatnią kappa, lecz I+J=-2I i ekspansję: konieczne jest kryterium dyskretne. Przy wielokrotnej iteracji jednego bloku \(\rho(I+J)<1\) opisuje lokalną stabilność asymptotyczną, natomiast \(\sigma_{\max}(I+J)<1\) daje kontrakcję wszystkich kierunków w jednym kroku. W rzeczywistym ViT bloki są różne i używane pojedynczo.

**Wniosek o detekcji.** Sprawdzić osobno q dla kierunków istotnych dla głowy i niewidocznych lokalnie. Tłumienie kierunków szumu przy zachowaniu kierunków zadania jest użyteczniejsze niż ogólne twierdzenie „chmura się kurczy”. Sama kontrakcja może niszczyć sygnał potrzebny do detekcji.

## 7. M5 — Pełna odpowiedź impulsowa i tensor Cauchy’ego–Greena

Tensor Cauchy’ego–Greena mierzy deformację przez pełne odwzorowanie trajektorii. Jego zastosowanie do sieci jest przeniesieniem narzędzia z geometrii dynamiki, nie twierdzeniem, że transformer jest płynem lub pęsetą. [Farazmand i Haller, 2012, wzór2](https://www.georgehaller.com/reprints/compvariLCS.pdf).

**Wzór.** Dla perturbacji w podkroku l:
\[
\delta X_{l+h}\approx\Phi_{l,h}\delta X_l,\quad
\Phi_{l,h}=M_{l+h-1}\cdots M_l.
\]
\[
C_{l,h}=\Phi_{l,h}^\top G_{\rm out}\Phi_{l,h}.
\]
Przy G_in=I wzmocnienia długości są pierwiastkami z wartości własnych C. Przy innej metryce wejścia rozwiązujemy
\[
C v=\lambda G_{\rm in}v.
\]
Przyrost logarytmiczny \(\log\sqrt\lambda/h\) ma jednostkę „na podkrok głębokości”; nie jest wykładnikiem na sekundę lub aktualizację treningu.

**Dokładny obiekt i zegar.** Mały impuls dodany do jednego tokenu albo wybranej kombinacji tokenów przed konkretnym podkrokiem. Wagi zamrożone, a pełna konfiguracja przechodzi przez rzeczywistą pozostałą sieć, wraz z propagacją przez uwagę.

**Minimalny algorytm.**
1. Wybrać r małych kierunków, np. r=6–16, w macierzy E o rozmiarze115200×r, z \(E^\top G_{\rm in}E=I\).
2. Przenosić każdą kolumnę przez JVP kolejnych rzeczywistych podkroków: \(T_{k+1}=M_kT_k\).
3. Zapisywać rozkład normy perturbacji po600 tokenach.
4. Policzyć małą macierz \(C_E=T^\top G_{\rm out}T\), r×r.

Względna objętość obrazu tego r-wymiarowego elementu wynosi \(\sqrt{\det C_E}\). To objętość **wybranej perturbacji**, a nie całego otoczenia115200D. Pełny rzut końcowy można raportować osobno.

**Kontrole.** Bezpośredni forward stanów X±epsilon E_i musi zgadzać się z JVP. Wstrzyknięta symetria g powinna pozostać g i zniknąć w ilorazie oraz w końcowej głowie. Nie mnożyć macierzy \(PM_kP^\top\): taki iloczyn usuwa po każdym kroku drogi wyjścia poza3D i powrotu. Skala i baza porównania między checkpointami muszą być jawne; PCA dobierane od nowa może obracać obserwowany obiekt.

**Wniosek o detekcji.** Metoda pokazuje, czy informacja o zaburzeniu pozostaje przy tokenie, rozchodzi się na inne tokeny i wzmacnia w określonych kierunkach. Zestawiona z M6 odróżnia duże deformacje nieistotne dla zadania od małych, lecz silnie wpływających na ramkę lub klasyfikację.

## 8. M6 — Gramian czułości zadania i ograniczona obserwowalność

Gramian obserwowalności sumuje wpływ perturbacji stanu na określone pomiary w czasie. To pojęcie teorii sterowania; zastosowanie do ViT jest poprawne tylko po wskazaniu, jakie odczyty rzeczywiście występują. [Åström i Murray, materiał B.2 do Feedback Systems](https://www.cds.caltech.edu/~murray/books/AM08/pdf/fbs-exercises_04Apr20.pdf).

**Wzór ogólny.** Przy pomiarach \(\delta y_h=H_h\Phi_{l,h}\delta X_l\):
\[
W_o=\sum_{h\in\mathcal M}\Phi_{l,h}^\top H_h^\top R_hH_h\Phi_{l,h},
\quad
\delta X_l^\top W_o\delta X_l
=\sum_h\|\delta y_h\|_{R_h}^2.
\]

**Istotne ograniczenie badanego kodu.** SmallViT ma jedną wytrenowaną głowę na końcu. Bez dodatkowych jawnie wirtualnych odczytów zbiór pomiarów zawiera tylko koniec:
\[
J_{\rm task}=B_{\rm final}\Phi_{l,H},\qquad
W_{\rm task}=J_{\rm task}^\top R J_{\rm task},
\]
gdzie B_final jest Jacobianem rzeczywistej końcowej LN i głowy dla wszystkich600 tokenów.

Sumowanie tej samej głowy przyłożonej pośrednim warstwom tworzyłoby dodatkowy protokół diagnostyczny. Nie jest automatycznie obserwowalnością działającego detektora.

**Dokładny obiekt i zegar.** Perturbacja strumienia w wybranej warstwie, odczyt końcowych logitów i czterech logarytmów odległości lub jawnie zdefiniowanych odczytów po dekodowaniu. Zegar to pozostała głębokość, nie trening.

**Minimalny algorytm.**
1. Ustalić dodatnio półokreśloną metrykę R wyjścia: np. wagi klasyfikacji i współrzędnych ramki, ustalone wcześniej na zbiorze kalibracyjnym.
2. Dla r kierunków E obliczyć \(Y=J_{\rm task}E\) przez JVP całego dalszego ciągu.
3. Wyznaczyć \(W_E=Y^\top RY\), małą macierz r×r.
4. Jeśli potrzebne są kierunki maksymalnej czułości, realizować działanie Wv przez JVP, mnożenie przez R i VJP, bez materializowania W.

R=I miesza jednostki logitu i logarytmów odległości umownie, co trzeba podać. Hessian GIoU lub całej straty nie musi być dodatnio półokreślony, więc nie jest automatycznie metryką. Focal loss nie daje automatycznie interpretacji probabilistycznego Fisher information.

**Kontrole i gauge.**
- \(J_{\rm task}g=0\) dla dokładnych symetrii pre-LN.
- Przy odczycie pojedynczego końcowego tokenu B ma rząd najwyżej5; jego lokalne jądro nie jest globalną płaską doliną.
- Przy perturbacji wcześniejszego tokenu uwaga może przekazać wpływ wielu tokenom końcowym, więc ograniczenie „tylko5 widocznych kierunków” nie obowiązuje dla całego końcowego odczytu.
- „Niewidoczny” oznacza pierwszorzędowo niewidoczny; test skończonej perturbacji sprawdza wpływ drugiego rzędu.
- Wynik iloczynu kwadratowego jest niezmienniczy przy spójnej zmianie współrzędnych i metryki. Same wartości własne W wymagają określonej metryki wejścia.

**Wniosek o detekcji.** To bezpośredni pomiar podatności detektora na zmianę reprezentacji. Pozwala pytać, czy „wir” albo „sztywność” dotyczą kierunków odczytywanych przez zadanie. Duże W oznacza dużą czułość, nie wysoką jakość. Jakość i odporność trzeba ocenić przez rzeczywiste metryki oraz kontrolowane perturbacje w obu znakach.

## 9. M7 — OU z dokładnymi przejściami, nieregularnym czasem i wspólnym wymuszeniem

Fizyka pęset dostarcza wzoru na relaksację i narzędzi identyfikacji sztywności, ale wymaga zgodności modelu obserwacji i czasu próbkowania. Wpływ skończonej częstotliwości oraz czasu integracji był analizowany bezpośrednio w kalibracji pęset. Nie należy utożsamiać czasu ekspozycji kamery z checkpointem sieci; przenosimy obowiązek jawnego modelowania obserwacji. [Pérez-García i in., 2023](https://arxiv.org/abs/2305.07245).

**Wzór.** Dla liniowego modelu warunkowego w wybranych współrzędnych:
\[
dy=(Ay+b(t)+U c(t))dt+L\,dW_t.
\]
Przy stałych A,L i stałym wymuszeniu b_n+Uc_n w odstępie \(\Delta_n=t_{n+1}-t_n\):
\[
y_{n+1}=\Phi_ny_n+\int_0^{\Delta_n}e^{As}(b_n+Uc_n)\,ds+\eta_n,
\]
\[
\Phi_n=e^{A\Delta_n},\quad
\eta_n\sim N(0,Q_n),\quad
Q_n=\int_0^{\Delta_n}e^{As}LL^\top e^{A^\top s}ds.
\]
Dla skalarnego OU \(dy=-\lambda(y-m)dt+\sqrt{2D}dW\):
\[
E[y_{n+1}|y_n]=m+e^{-\lambda\Delta_n}(y_n-m),
\]
\[
\operatorname{Var}(y_{n+1}|y_n)=\frac D\lambda(1-e^{-2\lambda\Delta_n}).
\]
Granica lambda→0 daje wariancję2D Delta i proces bez przywracania.

**Dokładny obiekt i zegar.** Ta sama sonda obrazu i ten sam punkt warstwy śledzony przez aktualizacje treningu. y to jawnie ustalone współrzędne w192D lub niskowymiarowa stała projekcja, np. kierunki zadania i kontrolne. t jest numerem aktualizacji AdamW. W badanym zapisie odstępy trzeba odczytywać z kroki, a nie zakładać100.

**Wspólne wymuszenie i pamięć.** Naturalny stan uczącego się modelu obejmuje theta, bufory BN, momenty AdamW, paczkę i augmentacje. c(t) może zawierać dostępne obserwacje wspólnego wymuszenia: projekcje aktualizacji wag, zmiany BN, lr, clipping i statystyki optymalizatora. Jeżeli ich nie zapisano, model tokenu jest modelem z ukrytymi zmiennymi. Brak pamięci w samym y nie jest zagwarantowany.

Usunięcie globalnej średniej tokenów albo pierwszych PCA nie dowodzi usunięcia wspólnego napędu: może również usunąć sygnał zadania. Każdy taki zabieg musi być dopasowany na danych uczących model statystyczny i sprawdzony na oddzielnym oknie czasowym. Wersje raw oraz warunkowe powinny być raportowane obok siebie.

**Minimalny algorytm.**
1. Wybrać stałe współrzędne o wymiarze pozwalającym estymować parametry z dostępnych danych; nie dopasowywać swobodnej macierzy192×192 do krótkiej trajektorii bez regularizacji i walidacji.
2. Dopasować przejścia z rzeczywistym Delta przez likelihood. Centrum stałe w oknie albo latentne centrum z osobnym równaniem stanu; nie wyznaczać go dwustronną średnią przyszłych obserwacji.
3. Porównać model stabilny, model z dryfem bez przywracania, model z kolorowym szumem lub dodatkowym stanem pamięci.
4. Walidować prognozę na przyszłych fragmentach; badać innowacje standaryzowane przez Q_n.
5. Surogaty muszą przejść dokładnie to samo próbkowanie, kwantyzację i filtrowanie co dane.

To dopasowanie statystyczne do istniejących odczytów, bez potrzeby trenowania ViT. Wykonanie nie jest częścią obecnego zadania.

**Kontrole.** Nie narzucać stabilnego A jako jedynej możliwości, gdy celem jest test istnienia relaksacji. Sprawdzić autokorelację i zależność innowacji od historii, wspólnych zmiennych i fazy uczenia. Nie porównywać wzoru na korelację kowariancyjną OU z uśrednionym cosinusem innych przyrostów. Przy tau mniejszym od odstępu checkpointów raportować szeroką niepewność i słabą identyfikowalność.

Wcześniejszy audyt wykazał, że wyśrodkowana średnia może tworzyć dodatnią autokorelację reszty nawet dla błądzenia losowego. Ten sam filtr musi więc wystąpić w kontroli ujemnej, jeśli pozostaje w jakiejkolwiek wizualizacji.

**Wniosek o detekcji.** Po walidacji można porównać czas i amplitudę zmian w kierunkach zadania z kierunkami kontrolnymi oraz sprawdzić, czy przewidują pogorszenie detekcji na danych walidacyjnych. Samo OU nie dowodzi, że szum jest termiczny, że istnieje energia ani że1/var jest fizyczną sztywnością.

Do badania relacji odpowiedź–fluktuacje potrzebny byłby osobny protokół małego, kontrolowanego wymuszenia w stanie uczącego się układu, z zachowaniem optimizer/BN i identycznym losowaniem. Nie można zastąpić go korelacją nieinterweniowanej trajektorii. Taki protokół wymagałby późniejszej zgody na nowe obliczenia.

**Mały atlas testów czasowych.**

- Kowariancja Allana dla sąsiednich średnich blokowych o jednakowej długości tau:
  \[
  A(\tau)=\tfrac12 E[(\bar y_{m+1}^{(\tau)}-\bar y_m^{(\tau)})
  (\bar y_{m+1}^{(\tau)}-\bar y_m^{(\tau)})^\top].
  \]
  Stosować osobno do logitu ustalonego tokenu, jego dekodowanej ramki oraz stałej projekcji stanu. Przy nieregularnych checkpointach bloki muszą mieć tę samą długość w aktualizacjach; średnia po równej liczbie zapisów nie zapewnia tego warunku. Ujawnić ważenie czasem i ewentualną interpolację. Średnie służą tu opisowi skal czasowych, nie są postulowanym centrum pułapki. Dla końcowej ramki wybranej przez top-k trzeba zachować identyfikację obiektu i oznaczać zmiany wyboru lub brak detekcji: skok rankingu nie jest automatycznie dyfuzją tego samego tokenu.
- Asymetria opóźnionej kowariancji \(K(\tau)-K(\tau)^\top\), gdzie \(K(\tau)=E[y(t+\tau)y(t)^\top]\), mierzy uporządkowaną w czasie zależność. Dla stacjonarnego liniowego procesu \(K(\tau)=e^{A\tau}\Sigma\); antysymetria zależy więc również od Sigma i tau. Nie jest równa \((J-J^\top)/2\) deterministycznego pola MLP. W zapisie \(A=-B\) dokładny propagator OU ma postać \(\exp(-B\Delta)\), a kowariancja innowacji pozostaje całką macierzową podaną wyżej.
- Parowana odpowiedź optymalizatora: sklonować ten sam stan początkowy theta, momenty AdamW, BN, scheduler i RNG; zastosować mały impuls +h/−h i użyć identycznej przyszłej sekwencji paczek oraz augmentacji. Po impulsie oba układy ewoluują zgodnie ze swoimi równaniami, włącznie z naturalną aktualizacją BN i momentów. Różnica odczytów podzielona przez2h estymuje odpowiedź; powtórzenia z różnym wspólnym losowaniem oceniają niepewność.
- Dla porównania typu FDT wymuszenie musi być jawnie sprzężone z obserwablą B, np. przez \(L_h=L-hB(\theta)\) i wynikający z tego składnik gradientowy. Ręczne przesunięcie tokenu przy zamrożonych wagach jest innym eksperymentem. Nawet poprawne sprzężenie nie daje automatycznie równowagowego FDT dla AdamW z clippingiem i pamięcią; najpierw trzeba ustalić dynamikę, mobilność, szum i warunki stacjonarności. Bez tego raportować odpowiedź i korelację oddzielnie, bez temperatury w kelwinach.

## 10. M8 — Dwie interwencje: współdziałanie, znoszenie i wpływ skierowany

**Wariant A: dwa neurony.** W ustalonym MLP wybieramy dwa zamrożone kierunki bazowe u_a,u_b. Końcowe wyjście po wstrzyknięciu jest Y(alpha,beta). Dla skalarnego g:
\[
I_{ab}=g(\alpha,\beta)-g(\alpha,0)-g(0,\beta)+g(0,0).
\]
W lokalnej granicy:
\[
I_{ab}\approx\alpha\beta\,u_a^\top\nabla^2g\,u_b.
\]
Geometria znoszenia w podkroku zależy od u_a·u_b, natomiast I_ab mierzy nieliniowe współdziałanie dalszej sieci. Te dwie rzeczy nie są równoważne. Dla liniowego końcowego odczytu I_ab=0, nawet gdy u_a=-u_b.

**Wariant B: dwa tokeny w uwadze.** Niech
\[
K_{ba}=\partial F_b/\partial x_a.
\]
Dla kierunków e_a,e_b miara niesymetrii wzajemnego wpływu to
\[
\mathcal R_{ab}=e_b^\top K_{ba}e_a-e_a^\top K_{ab}e_b.
\]
To antysymetryczny element pełnego Jacobianu w płaszczyźnie dwóch tokenów. Nie jest testem trzeciej zasady Newtona obowiązującej sieć; mierzy, w jakim stopniu sieć różni się od modelu wzajemnej interakcji gradientowej w wybranej metryce.

**Dokładny obiekt i zegar.** Interwencje w tym samym checkpoincie i tym samym miejscu sieci, następnie pełny dalszy forward. Dla pary neuronów trzeba określić, czy dotyczą jednego czy wszystkich tokenów. Dla pary tokenów aktualizujemy całe qkv. Jeżeli zmieniamy dwa różne bloki, kolejność jest częścią protokołu i nie należy porównywać jej z dodawaniem dwóch sił w tym samym punkcie.

**Minimalny algorytm.**
1. Wybrać pary na zbiorze odkrywczym: przeciwne, zgodne i losowe, z dopasowaną normą oraz aktywnością.
2. Wykonać cztery warunki bazowy/A/B/A+B, dla obu znaków małej amplitudy.
3. Zmierzyć stan bezpośrednio po interwencji i pełną detekcję końcową.
4. Dla wariantu tokenowego policzyć dwa JVP i odczytać wskazane komponenty odpowiedzi.
5. Oddzielić test par wcześniej wybranych od ocen na innych obrazach i checkpointach.

**Kontrole.** Dla dwóch dodawanych perturbacji w tym samym miejscu stan bezpośredni musi sumować się dokładnie; interakcja może pojawić się dopiero w nieliniowym dalszym odczycie. Losowe pary o dopasowanej normie chronią przed wyjaśnieniem samą wielkością zmiany. Wstrzyknięcia gauge c·1 nie powinny zmieniać detekcji. Wysoki efekt na jednym zdjęciu dobranym z GT nie dowodzi generalizacji.

**Wniosek o detekcji.** Można wykazać, że określone pary neuronów lub tokenów współdziałają w lokalizacji narzędzia, tłumieniu fałszywych alarmów albo utrzymaniu ramki. To mocniejszy argument niż samo zauważenie przeciwnych wiązek na wizualizacji. „Interferencja” w opisie powinna wtedy znaczyć nieliniową interakcję wpływów albo znoszenie wektorów, z jednoznacznym określeniem którego.

## 11. Co w tym kodzie jest optyką, a czego w nim nie ma

Odczyt model.py oraz app/pole_podkrokow.py, pulapka_obliczenia.py, ogniskowa.py i lasery_dane.py potwierdza:

| Pojęcie optyczne | Obecność w obliczeniach modelu |
|---|---|
| Rzeczywiste współczynniki aktywacji z | tak; GELU, także wartości ujemne |
| Suma podpisanych wektorów wkładów | tak |
| Lokalny Jacobian, dywergencja, część antysymetryczna | tak jako obiekty matematyczne |
| Faza elektromagnetyczna wiązki | brak |
| Amplituda zespolonego pola E | brak |
| Interferencja \(|\sum_c E_c|^2\) | brak |
| Długość fali, częstotliwość optyczna, polaryzacja | brak |
| Pole elektryczne i magnetyczne, równania Maxwella | brak |
| Współczynnik załamania koloidu i ośrodka | brak |
| Tensor naprężeń Maxwella lub przeniesienie pędu fotonów | brak |
| Kalibracja niutonów, dżuli, kelwinów | brak |

W pole_podkrokow.py słowo faza oznacza przełączenie uwaga/MLP. Typ complex w pulapka_obliczenia.py:55 i ogniskowa.py:91–92 służy zapisowi wartości własnych rzeczywistej macierzy Jacobianu. Nie jest amplitudą fali. Kluczowe obliczenie MLP jest liniową sumą \(W_2z+b_2\), a nie kwadratem modułu sumy pól.

Rzeczywiste pęsety mogą wymagać modelowania fal, rozpraszania i właściwości ośrodka. Przegląd metod kalibracyjnych i zastosowań optycznych nie ustanawia takiej fizyki w dowolnym polu wektorowym. [Gieseler i in., 2020](https://arxiv.org/abs/2004.05246).

Przenoszalny jest zatem **język i protokół pomiaru odpowiedzi pola**, po dopasowaniu definicji. Dodanie do wizualizacji faz laserów lub prążków interferencyjnych nie odpowiadałoby mechanizmowi obecnego modelu.

## 12. Minimalne dane wynikowe przyszłej implementacji

Dla każdej obserwacji należałoby zapisać:

1. Identyfikator obrazu, podział danych, checkpoint, rzeczywisty krok uczenia, miejsce interwencji i zegar.
2. Definicję stanu, centrowania, projekcji, gauge, metryki i skali; hashe wag i kodu.
3. Kierunek i amplitudę perturbacji w pełnej przestrzeni; dla neuronów również bazowe z i v.
4. Wynik lokalny i końcowy: stan, logity, ramki, wybrane metryki zadania.
5. Prognozę liniową, pomiar skończony i ich rozbieżność; wynik testu symetrii c·1.
6. Wersje bibliotek, rzeczywiste epsilon LN, dtype oraz ustawienia eval/BN.
7. Dla modeli stochastycznych: rzeczywiste Delta, definicję centrum, zbiór fit/validation, hipotezę napędu, innowacje i niepewność parametrów.

To specyfikacja, nie polecenie uruchomienia. Nowe narzędzia, jeśli zostaną później zamówione, powinny korzystać ze wspólnego interfejsu do pełnego forward/JVP/VJP i interwencji, żeby wszystkie metody używały tego samego modelu i punktu odniesienia.

## 13. Granice wnioskowania o jakości detektora

Wszystkie8 metod powinny kończyć się rozróżnieniem trzech poziomów dowodu:

- **Tożsamość lub geometria:** równanie i jego lokalna kontrola numeryczna.
- **Wpływ w modelu:** ustalona interwencja zmienia odpowiedź detektora zgodnie z prognozą.
- **Znaczenie dla zadania:** zależność utrzymuje się na niezależnych obrazach i nie wynika wyłącznie z normy, rozdziału narzędzie/tło, checkpointu lub trafności bazowej.

Wzrost pewności pojedynczego tokenu nie zastępuje jakości detekcji. GIoU, IoU i AP mierzą różne rzeczy; AP i wybór top-k są niegładkie, więc ich zmianę należy odczytać po skończonej interwencji, nie wyprowadzać wyłącznie z gradientu logitu.

„Pułapka wzmacnia detekcję” można powiedzieć dopiero po wskazaniu mierzalnej własności mapy i wykazaniu jej wpływu na zadanie. Nie wynika to z samego użycia nazw „sztywność”, „wir”, „temperatura” lub „laser”.

## 14. Źródła kodowe i zakres weryfikacji

Wykorzystano wcześniejszy audyt i ponownie odczytano wybrane fragmenty:
- experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py: konstrukcja pre-LN i głowy, położenie hooka768D, clamp straty;
- app/lasery_dane.py:367–453: pełny rozkład podkroków i centrowania;
- app/pole_podkrokow.py:60–80: pole MLP i query-only field uwagi;
- app/pulapka_obliczenia.py:54–69: dopasowanie J3 i kryterium I+J;
- app/ogniskowa.py: lokalne pole i zespolone wartości własne;
- experiments/pulapka_cel_koloidu_20261015/dolina.py: definicja lokalnej przestrzeni odczytu;
- wcześniejszy audyt pomiar.py, analiza.py i pulapka_uczenia.py: epsilon, kwantyzacja, nieregularne odstępy i filtrowanie OU.

Źródła z fizyki i teorii dynamiki przywołano przy konkretnych elementach. Formuły dla gauge pre-LN, obserwowalności końcowej głowy oraz ograniczenia złożenia rzutów są własnym wyprowadzeniem z badanego kodu; nie przypisano ich publikacjom o pęsetach.

Nie zmieniono kodu, nie uruchomiono inferencji modelu, treningu ani eksperymentu fizycznego. Wykonano odczyt plików i wyszukiwanie źródeł; zapisano wyłącznie ten raport.

