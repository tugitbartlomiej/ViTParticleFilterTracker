# Audyt matematyczny mikroskopu SmallViT

Data: 2026-10-15. Zakres: odczyt kodu i istniejących wyników, analityczne kontrprzykłady oraz odczyt metadanych zapisanego pomiaru. Nie zmieniono źródeł ani nie uruchomiono treningu. Audyt nie jest przeglądem literatury.

Najmocniejszy, obronny rdzeń pracy to **dokładny rozkład aktualizacji strumienia rezydualnego na podpisane wkłady neuronów MLP i tokenów uwagi, połączony z analizą lokalnej geometrii oraz śledzeniem tych obiektów podczas uczenia**. Obecne dowody nie wystarczają do utożsamienia tego układu z fizyczną pęsetą, wykazania procesu OU ani wyprowadzenia z tej analogii prawa relaksacji.

## 1. Najważniejsze ustalenia

| Priorytet | Ustalenie z kodu | Skutek dla publikacji |
|---|---|---|
| P1 | Implementacja focal loss clampuje prawdopodobieństwo. Dla dodatniego tokenu płaskie minimum zaczyna się od logitu około 9,21024. | „Cel straty leży dokładnie w +∞” jest fałszywe dla zaimplementowanej funkcji. Należy oddzielić idealną stratę bez clamp od wykonywanego kodu. |
| P1 | Pole uwagi zmienia tylko query; key/value wszystkich tokenów, w tym badanego, pozostają stałe. Średnia odniesienia także pozostaje stała. | Jest to warunkowe pole testowego odbiorcy. Zgodność wartości w środku nie potwierdza pełnego Jacobianu self-attention. |
| P1 | Pomiary podkroków zapisują lokalne macierze 3×3. Iloczyn takich macierzy nie jest rzutem pełnego iloczynu Jacobianów. | Nie można z nich wyprowadzić objętości całego przejścia ani wykluczyć mechanizmu stabilizacji wszystkich 600 tokenów. |
| P1 | Dopasowanie OU odejmuje dwustronną średnią, miesza różne statystyki i traktuje nieregularne odstępy jako stałe. | „Uwięzienie OU” i fizyczna sztywność nie są zidentyfikowane. Obecny wynik jest opisową zgodnością wybranych statystyk. |
| P1 | „Najwyższy logit” i „najbliższy punkt” to rezultaty pojedynczej optymalizacji Adam z miękkimi karami. | Są to znalezione kandydaty; nie certyfikowane maksimum, minimum odległości ani dokładne spełnienie celu ramki. |
| P1/P2 | Pomiar używa ręcznego LayerNorm z epsilon 1e-6 oraz buforów float16, mimo innej konstrukcji oryginalnej głowy i deklaracji precyzji. | Wymaga kontroli zgodności analitycznej głowy z rzeczywistą i odrębnej oceny błędu kwantyzacji przed podawaniem precyzyjnych efektów. |
| P2 | Antysymetryczna część Jacobianu jest lokalną częścią obrotową; sama nie dowodzi wirowych orbit ani prądów prawdopodobieństwa. | Terminy „wir Browna”, „produkcja entropii” i „temperatura” wymagają dodatkowych, właściwych im pomiarów. |
| P2 | Rząd głowy równy 5 daje lokalną rozmaitość poziomicową o wymiarze 187, nie globalną płaską dolinę. | Kierunki jądra Jacobianu są niewidoczne tylko w pierwszym rzędzie przy danym punkcie i checkpoincie. |
| P2 | Ruch tokenu podczas uczenia wynika z aktualizacji wspólnych parametrów, stanu optymalizatora i buforów BatchNorm. | Model autonomiczny pojedynczego tokenu nie wynika z architektury. |

P1 oznacza problem mogący zmienić główny wniosek; P2 oznacza potrzebne doprecyzowanie interpretacji lub zakresu twierdzenia.

## 2. Ścisły model przejścia przez sieć

Niech N=600, d=192, m=768, H=3. Dla ustalonego obrazu i checkpointu X=(x_1,...,x_N) jest pełnym stanem tokenów, x_j należy do R^192. Łodyga CNN i kodowanie pozycji wyznaczają X^(0). Każdy blok realizuje dwa różne odwzorowania:

\[
\widetilde X^{(b)}=X^{(b)}+F_A^{(b)}(X^{(b)}),\qquad
X^{(b+1)}=\widetilde X^{(b)}+F_M^{(b)}(\widetilde X^{(b)}).
\]

To 12 dyskretnych podkroków, każdy z własnymi wagami, a nie powtarzanie jednej autonomicznej siły. Zapis „jawny Euler z krokiem 1” jest poprawną reprezentacją algebraiczną. Nie stanowi dowodu istnienia fizycznego czasu, mobilności, lepkości ani granicy ODE. Podobnie kompozycja dwóch resztowych map ma formę rozszczepienia, ale nie jest automatycznie kompozycją dokładnych przepływów Liego–Trottera.

### MLP: dokładny rozkład na podpisane wkłady

\[
a_{jc}=w_c^\top L(x_j)+b_{1c},\qquad
z_{jc}=\operatorname{GELU}(a_{jc})=a_{jc}\Phi(a_{jc}),
\]
\[
F_{M,j}(X)=W_2z_j+b_2=\sum_{c=1}^{768}z_{jc}v_c+b_2,\quad v_c=W_2[:,c].
\]

To równanie jest ścisłe dla badanego bloku z tożsamościowym LayerScale i bez aktywnego dropout. Istniejący kod rekonstrukcji uwzględnia te warunki: app/lasery_dane.py:367–411.

Wektor z_j w R^768 jest **reprezentacją aktywacji** tokenu. Można zgodnie z wcześniejszą terminologią użytkownika badać jego 768 współrzędnych w przestrzeni latentu. Nie jest on jednak tym samym stanem co x_j w R^192, do którego dodawana jest suma wkładów W_2z_j. Po przyjęciu obrazu „koloid + reflektory” z_j pełni rolę 768 podpisanych współczynników wiązek. Nie wolno bez jawnej zmiany definicji używać tych dwóch przestrzeni zamiennie.

GELU przyjmuje wartości ujemne, a po odjęciu średniej znak może zmienić się również dla dodatniej aktywacji. „Natężenie światła” nie jest dosłowną wielkością fizyczną. Znak i normę każdego wkładu trzeba zachować. Wektory wiązek nie tworzą na ogół ortogonalnej bazy; są 768 kierunkami w 192 wymiarach.

### Uwaga: wyrazy wolne i brak mechaniki parowej

Przy y_j=L_1(x_j) dla głowy h:

\[
q_j^h=W_Q^hy_j+b_Q^h,\quad k_j^h=W_K^hy_j+b_K^h,\quad
v_j^h=W_V^hy_j+b_V^h,
\]
\[
A_{jk}^h=\frac{\exp((q_j^h)^\top k_k^h/\sqrt{d_h})}
{\sum_r\exp((q_j^h)^\top k_r^h/\sqrt{d_h})},
\qquad
F_{A,j}(X)=\sum_{h,k}A_{jk}^hW_O^hv_k^h+b_O.
\]

Dokumentowane surowe równanie w Z19 pomija b_V przy pozostawieniu tylko b_O. Ponieważ qkv_bias=True (model.py:41), trzeba albo użyć powyższego v_k, albo zastąpić wyraz wolny przez:

\[
b_{\mathrm{eff}}=b_O+\sum_h W_O^h b_V^h.
\]

Kod app/lasery_dane.py:399–404 bierze v z pełnego qkv, więc implementacja rekonstrukcji ten składnik zawiera.

Są to oddziaływania skierowane i zależne od wszystkich tokenów przez normalizację softmax. Nie muszą spełniać f_jk=-f_kj, zależeć tylko od x_j-x_k ani zachowywać środka masy. „Siły między tokenami” jest nazwą matematycznych wkładów, nie dowodem mechaniki Newtona.

### Dokładne centrowanie chmury

Dla \(\bar F=N^{-1}\sum_j F_j\):

\[
F_{M,j}-\bar F_M=\sum_c(z_{jc}-\bar z_c)v_c,
\]
\[
F_{A,j}-\bar F_A=\sum_{h,k}(A_{jk}^h-\bar A_k^h)W_O^hW_V^hy_k.
\]

Bias MLP, output bias uwagi oraz bias V znikają po centrowaniu. Ostatni składnik znika, bo \(\sum_k(A_{jk}^h-\bar A_k^h)=0\). Dlatego zapis centrowany Z20 jest poprawny mimo niepełnego surowego zapisu Z19.

Dla stałego rzutu P o ortonormalnych wierszach i stałej skali s w obrębie przejścia:

\[
p_j=P(x_j-\bar x)/s,\qquad
p_j^+-p_j=P(F_j-\bar F)/s.
\]

To jest bardzo mocna własność narzędzia: suma rysowanych wkładów odtwarza rysowany skok. Implementacja używa jednej skali s_ref na checkpoint; w app/lasery_dane.py:427–428 jest to średnia norm wycentrowanych tokenów po 13 stanach, nie matematyczny RMS. Nazwa zmiennej rms nie zmienia definicji.

## 3. Dokładny LayerNorm i Jacobian MLP

Niech \(C=I-\mathbf1\mathbf1^\top/d\), \(r=Cx\), \(s=\sqrt{r^\top r/d+\epsilon}\), \(\Gamma=\operatorname{diag}(\gamma)\). Wtedy

\[
L(x)=\Gamma r/s+\beta,\qquad
D L(x)=\Gamma\left(\frac{C}{s}-\frac{rr^\top}{d s^3}\right),
\]
\[
J_M(x)=W_2\operatorname{diag}(\operatorname{GELU}'(a))W_1D L(x).
\]

Dokładny wkład jednego neuronu ma rząd co najwyżej jeden:

\[
J_c=v_c\,\operatorname{GELU}'(a_c)\,w_c^\top D L(x).
\]

Zapis z Z15 jest poprawny. Natomiast przybliżenie śladu z Z6, polegające na zastąpieniu D L przez \(\Gamma/s\), pomija projekcję centrowania i składnik radialny. Dokładnie:

\[
\operatorname{tr}J_M=\sum_c \operatorname{GELU}'(a_c)\,w_c^\top D L(x)v_c.
\]

Sam znak cosinusa \(\cos(\gamma\odot w_c,v_c)\) nie przesądza o znaku całego śladu. Dochodzą lokalne projekcje LayerNorm, wagi GELU', których znak może być ujemny, oraz sumowanie wkładów.

Przy epsilon>0:

\[
\|r/s\|^2=d\,\frac{\operatorname{var}(x)}
{\operatorname{var}(x)+\epsilon}<d.
\]

Jest to wnętrze ograniczonej powierzchnią sfery dziedziny w hiperpłaszczyźnie, nie dokładna sfera. Dokładnie zerowany jest kierunek przesunięcia \(\mathbf1\). Kierunek radialny jest jedynie silnie tłumiony:

\[
D L(x)\,r=\Gamma r\,\epsilon/s^3.
\]

Opis „sfera” jest dobrym przybliżeniem, kiedy wariancja jest dużo większa od epsilon. Nie powinien być oznaczany jako bezwarunkowo dokładny.

## 4. Co naprawdę mierzy pole 3D

### Zamrożone odniesienie MLP

Kod app/pole_podkrokow.py:60–65 oblicza

\[
f_{\mathrm{fr}}(u)=s^{-1}P\left[
F_M(x_j+sP^\top u)-\bar F_M(X_{\mathrm{base}})
\right].
\]

Stąd \(f_{\mathrm{fr}}(0)\) jest dokładnym wycentrowanym skokiem bazowym, a
\(D f_{\mathrm{fr}}(0)=P J_M(x_j)P^\top\).

Jest to poprawnie zdefiniowane pole próbne z zamrożoną średnią. Jeśli naprawdę przesuwamy jeden token, a następnie ponownie centrujemy całą chmurę, średnia siła też się zmienia. Dla MLP pochodna siły j względem surowego przesunięcia tego tokenu zawiera czynnik \(1-1/N\). Dla uwagi zmieniają się także siły wszystkich pozostałych odbiorców. Nie jest to ta sama operacja co przyjęta w polu próbnym.

### Uwaga: pole tylko po query

App/pole_podkrokow.py:73–80 liczy bazowe K,V,A i \(\bar A\); następnie dla próbnych Xp wykorzystuje tylko nowe q. Nawet k_j i v_j pozostają bazowe. Dokładny zapis badanego pola to:

\[
f_{Q}(u)=s^{-1}P\sum_{h,k}
[A^h(q_j(x_j+sP^\top u),K_{\mathrm{base}})_k-\bar A_k^h]
W_O^h v_{k,\mathrm{base}}^h.
\]

Dla prawdziwej self-attention zmiana x_j zmienia q_j, k_j, v_j oraz poprzez k_j uwagę wszystkich innych tokenów. Do pełnego Jacobianu potrzebne są te pochodne i pochodna średniej. Zgodność f_Q(0) ze skokiem nie testuje ich.

Nazwa zalecana w metodzie: **warunkowe pole odpowiedzi odbiorcy przy zamrożonym kontekście**, z jawną definicją kontekstu. Jest to przydatny pomiar; zakres jego twierdzeń musi odpowiadać interwencji.

### Rzut i kompozycja nie komutują

Pełny stan uwagi ma Nd=115200 wymiarów. Dla pełnych map podkroków \(\Psi_k\), ich Jacobianów \(M_k=D\Psi_k\), pełna czułość przejścia jest:

\[
D\Psi=M_{12}M_{11}\cdots M_1.
\]

Jeśli nawet pominiemy sprzężenia między tokenami, na ogół

\[
P(M_{12}\cdots M_1)P^\top
\ne (PM_{12}P^\top)\cdots(PM_1P^\top).
\]

Prawa strona po każdym kroku usuwa składowe poza trzema wybranymi kierunkami. Lewa zachowuje ścieżki wyjścia z podprzestrzeni i powrotu do niej. Równość wymaga szczególnych warunków niezmienniczości podprzestrzeni; nie wynikają one z kodu.

Kontrprzykład: P=[1,0], \(M_1=\begin{pmatrix}1&0\\a&1\end{pmatrix}\), \(M_2=\begin{pmatrix}1&b\\0&1\end{pmatrix}\). Rzut pełnego iloczynu to 1+ab, iloczyn rzutów to 1.

Dla perturbacji 3D osadzonej pełną macierzą E właściwe miary to:

- objętość końcowego obrazu tej trójwymiarowej perturbacji w pełnej przestrzeni: \(\sqrt{\det(E^\top D\Psi^\top D\Psi E)}\);
- objętość widoczna po końcowym rzucie: \(|\det(PD\Psi E)|\);
- objętość pełnego otoczenia: \(|\det D\Psi|\), w pełnym wymiarze.

To różne pytania. Implementacja app/pulapka_obliczenia.py:54–69 zapisuje tylko J3×3 uzyskane z pola próbnego. W badanych plikach nie ma propagacji pełnej perturbacji przez wszystkie podkroki. Zdanie w OpisUkładu/01_OPIS_UKLADU.md:77 o objętości całego przejścia około 9 razy nie ma przez to statusu pomiaru pełnej objętości, jeżeli otrzymano je przez iloczyn zapisanych I+J3.

Nie ma sprzeczności między kontrakcją objętości w 192D i ekspansją w wybranych trzech kierunkach. Przykład w 4D: macierz mapy diag(1.1,1.1,1.1,0.5) zwiększa objętość pierwszych trzech kierunków 1.331 razy, lecz pełną objętość mnoży przez 0.6655. Z6 i nowy opis 3D muszą mieć podaną przestrzeń, zamiast wzajemnie się zastępować.

### Stabilność i zero pola

Dla hipotetycznej autonomicznej ODE \(\dot u=f(u)\) lokalna stabilność zera wymaga Re eigen(J)<0. Dla iteracji \(u_{n+1}=u_n+f(u_n)\) wymaga \(\rho(I+J)<1\). App/pulapka_obliczenia.py:69 używa poprawnego kryterium dyskretnego dla zlinearyzowanej mapy. App/ogniskowa.py w opisie klasyfikuje ognisko po znaku Re eigen(J), czyli według kryterium ciągłego.

Nawet \(\rho(I+J)<1\) nie gwarantuje spadku odległości w każdym pojedynczym kroku dla macierzy nienormalnej. Do kontrakcji euklidesowej każdego kierunku potrzebne jest \(\sigma_{\max}(I+J)<1\). W rzeczywistej sieci blok nie jest wielokrotnie iterowany, więc „pułapka trzyma” pozostaje interpretacją kontrfaktycznej iteracji, nie obserwowaną własnością jednego użycia bloku.

Dla skończonego kroku objętość zmienia się przez \(|\det(I+J)|\); sam ślad jest przybliżeniem log-determinantu tylko dla małego J. Przykład J=diag(-0.9,1) ma dodatni ślad 0.1, lecz mnożnik objętości 0.2.

Rozwiązanie \(-J^{-1}f(0)\) w pulapka_obliczenia.py:65–68 jest zerem lokalnego przybliżenia afinicznego. Wysokie R² wewnątrz kuli nie uzasadnia ekstrapolacji do odległego zera. Newton startujący z jednego punktu w ogniskowa.py może znaleźć zero, ale brak zbieżności nie dowodzi nieistnienia wszystkich zer w komórce. Zalecane brzmienie wyniku ujemnego: „nie znaleziono zera zastosowanym algorytmem w badanej domenie”.

## 5. Część antysymetryczna: co wolno nazywać wirem

\[
S=(J+J^\top)/2,\qquad A=(J-J^\top)/2.
\]

S opisuje lokalne rozciąganie w liniowej ODE, a A jej lokalną część obrotową. W 3D zapisany wektor
\(\omega=(A_{32},A_{13},A_{21})\) jest **połową curl f**, nie pełną wirowością curl. Warto podać tę konwencję przy porównaniach z optyką.

Niezerowe A nie zapewnia spiralnej ani zamkniętej trajektorii. Dla pola ścinającego f(x,y)=(ky,0) część antysymetryczna jest niezerowa, ale trajektorie są prostymi. Z kolei wartości własne zespolone nie wystarczają do stabilności dyskretnego podkroku.

„Część potencjalna” S jest właściwa lokalnie w liniowym przybliżeniu; nie jest globalnym rozkładem Helmholtza nieliniowego pola. Niezerowa antysymetria wyklucza lokalne przedstawienie całego pola jako gradientu potencjału w zwykłej metryce euklidesowej, ale nie dostarcza potencjału i prądu stacjonarnego.

Dla zidentyfikowanego procesu OU
\(dy=B y\,dt+\sqrt{2D}\,dW\)
z kowariancją stacjonarną C zachodzi

\[
BC+CB^\top+2D=0,\qquad
j(y)=(B+DC^{-1})y\,p(y).
\]

Prąd zależy od B, D i C. Antysymetria samego B nie wystarcza. Kontrprzykład:
\(D=\operatorname{diag}(1,2)\), \(C=\begin{pmatrix}2&1\\1&2\end{pmatrix}\),
\(B=-DC^{-1}=\begin{pmatrix}-2/3&1/3\\2/3&-4/3\end{pmatrix}\).
B jest niesymetryczne, lecz j=0. Dlatego „wir Browna” wymaga oszacowania prądów i procesu stochastycznego, nie tylko A pola MLP.

## 6. Ruch podczas uczenia i problem identyfikacji OU

### Właściwy stan układu uczącego

Niech \(\theta_t\) zawiera parametry, \(b_t\) bufory BatchNorm, \(m_t,v_t\) momenty AdamW, a \(\xi_t\) paczkę i augmentacje. Dla stałej sondy I:

\[
x_{j,t}=f_j(I;\theta_t,b_t),\quad
\Delta x_{j,t}=D_\theta f_j\,\Delta\theta_t+D_b f_j\,\Delta b_t+O(\|\Delta(\theta,b)\|^2).
\]

Dla treningu z clippingiem:

\[
g_t=\nabla_\theta L_{\xi_t},\quad
\tilde g_t=c_tg_t,\ c_t=\min(1,1/\|g_t\|),
\]
\[
m_{t+1}=\beta_1m_t+(1-\beta_1)\tilde g_t,\quad
v_{t+1}=\beta_2v_t+(1-\beta_2)\tilde g_t^{\odot2},
\]
\[
\theta_{t+1}=(I-\eta_t\Lambda)\theta_t-
\eta_t\,\hat m_{t+1}/(\sqrt{\hat v_{t+1}}+\epsilon_{\rm Adam}).
\]

Lambda jest diagonalnym wyborem weight decay: kod trening.py:27–30 nakłada go na parametry o ndim>=2, a nie na wszystkie parametry. BatchNorm aktualizuje bufory w trybie train, a sondy odczytywane są w eval. Nawet gdy \(\theta\) się nie zmienia, zmiana buforów może zmienić odczyt.

Przy zwykłym SGD i odpowiednich uproszczeniach pojawia się jądro sprzężenia tokenów:
\[
\Delta x_j\approx-\eta\sum_k
(D_\theta f_j)(D_\theta f_k)^\top\nabla_{x_k}L.
\]
Przy AdamW dochodzi adaptacyjna geometria i pamięć. Obserwowany ruch nie musi być równoległy do lokalnego \(-\nabla_{x_j}L\), bo inne przykłady i tokeny korzystają ze wspólnych wag.

Weight decay daje dokładny składnik kurczący wybrane parametry, ale nie dowodzi „jedynej sprężyny” reprezentacji. Krzywizna straty i optymalizator mogą dawać siłę przywracającą nawet przy weight decay=0. W reprezentacji wkład decay to w pierwszym rzędzie \(-\eta D_\theta f_j\Lambda\theta\), a nie ogólnie \(-\eta\lambda x_j\).

### Dlaczego obecny test nie identyfikuje OU

1. **Dwustronne filtrowanie.** pulapka_uczenia.py:23–28 wyznacza środek z przeszłości i przyszłości. Reszta jest wyjściem filtra, nie niezależnie zmierzoną odległością od zewnętrznego centrum.
2. **Analityczny kontrprzykład.** Niech X_t będzie zwykłym random walk z niezależnymi przyrostami i nieograniczonym MSD. Dla wyśrodkowanej średniej długości w=2h+1 reszta y_t=X_t-MA_w(X)_t ma współczynnik lag-1
   \[
   \phi_{\rm filt}=\frac{h-2}{h+1}.
   \]
   Zatem w=11 daje 0.5, a w=21 daje 0.72727, mimo braku siły przywracającej w X. Są to dokładne ilorazy kowariancji filtra przy białych przyrostach, z pominięciem brzegów.
3. **Różne statystyki.** phi jest regresją znormalizowanych reszt (wiersze 38–42), ale c1 jest średnią cosinusów surowych przyrostów X (43–44). Wzór OU \(\operatorname{Corr}(\Delta y_t,\Delta y_{t+1})=-(1-\phi)/2\) dotyczy stacjonarnej kowariancji procesu AR(1), nie ogólnie średniego cosinusa tych innych wektorów.
4. **Niejednakowe odstępy.** Maska sprawdza, czy odstęp po danym checkpoincie wynosi 100, a następnie liczy różnice między wybranymi checkpointami. Odczyt istniejącego pomiar.npz wykazał w późnym szeregu 530 odstępów po 100 i 57 odstępów po 200 kroków. Użycie jednego dt=100 w regresji jest więc nieścisłe.
5. **Wrażliwość na okno już w wynikach.** Dla tokenu226 okno11 daje phi=0.032, tau≈29 i przewidywaną korelację -0.484; okno21 daje phi=0.146, tau≈52 i -0.427. Obserwowany cosinus wynosi -0.415 w obu. Zgodność „-0.415 wobec -0.42” wybiera drugi wariant.
6. **Słaba identyfikacja szybkiej relaksacji.** Tau=30–50 przy próbkowaniu co100 jest oszacowaniem procesu szybszego od odstępu pomiarowego. Nie jest wykluczone matematycznie, ale ma małą identyfikowalność i wymaga niepewności estymacji oraz poprawnego modelu obserwacji.
7. **Nieznana temperatura i anizotropia.** 1/var nie jest fizyczną sztywnością bez modelu równowagowego, kalibracji szumu i jednostek. Przy izotropowej termicznej pęsecie C=kBT K^{-1}; dla ogólnego OU obowiązuje równanie Lapunowa, a nie skalarna ekwipartycja.

Dodatkowo X_t=m_t+epsilon_t z niezależnym szumem obserwacyjnym daje ujemną korelację kolejnych przyrostów bliską -1/2, bez dowodu mechanicznej sprężyny. Zależność MSD z wykładnikiem mniejszym od1 na jednym oknie jest zgodna z ograniczeniem ruchu, lecz nie identyfikuje jednoznacznie OU ani fazy szklistej.

Analiza.py:71–79 stosuje przesunięcia indeksów checkpointów mimo wpisów końców epok. „Lag=5” nie jest wszędzie dokładnie 500 aktualizacjami. Do analizy fizycznego czasu potrzebny jest dobór par po rzeczywistych krokach, nie samych indeksach.

## 7. Cel wyjściowy, skończony próg clamp i geometria głowy

### Cel wykonywanej funkcji straty

Model.py:101 i pomiar.py:40 używają
\[
p_c=\operatorname{clip}(\sigma(a),\epsilon_p,1-\epsilon_p),\quad\epsilon_p=10^{-4}.
\]

Dla dodatniego tokenu:
\[
\ell_+(a)=-(1-p_c)^2\log p_c.
\]

W arytmetyce rzeczywistej dla
\[
a\ge \log\frac{1-\epsilon_p}{\epsilon_p}
=\log9999\approx9.210240367
\]
strata jest stała, około \(1.00005\cdot10^{-12}\), a gradient przez clamp wynosi zero powyżej progu. Dla tokenu ujemnego analogiczny próg wynosi -9.21024. Szczegóły zaokrąglenia float32 mogą minimalnie przesunąć granicę.

Dlatego:
- p=1 jako idealna, niesklampowana odpowiedź sigmoid wymaga +infinity;
- minimalny dodatni składnik **wykonywanej straty** osiąga płaskie plateau przy skończonym logicie;
- to, czy zamrożona głowa może dosięgnąć tego plateau z prawidłową ramką, pozostaje osobnym problemem wykonalności.

Sama idealna focal loss ma asymptotykę \(\ell_+\sim e^{-3a}\), a nie zwykłego logistic loss \(e^{-a}\). Przeniesienie asymptotycznych wniosków regresji logistycznej na ten model wymaga uwzględnienia focal loss, AdamW, weight decay, clippingu oraz współuczonej głowy i reprezentacji. Zaobserwowany wzrost logitu jak log t nie dowodzi, że odpowiada za konkretny potęgowy zanik odległości tokenu od ostatniego checkpointu.

Dla dodatniego tokenu pełny zbiór niezależnych minimów w przestrzeni wyjścia, przed ograniczeniami osiągalności przez sieć, ma postać
\[
\{(a,\log l^*,\log t^*,\log r^*,\log b^*):a\ge a_{\rm clamp}\}.
\]
To półprosta, nie pojedynczy punkt w nieskończoności. Dla heat<1 cel klasyfikacyjny tej CenterNet loss to p przy dolnym progu, a nie p=heat; heat osłabia karę ujemną.

### Ograniczona głowa: istnienie ograniczenia a pomiar maksimum

Przy zamrożonych skończonych wagach LayerNorm ma ograniczony obraz, a głowa jest ciągła, więc logit ma skończone supremum. To jest ścisły argument. Dla
\(H(x)=W_h^{(2)}\operatorname{GELU}(W_h^{(1)}L(x)+b_h^{(1)})+b_h^{(2)}\)
proste, konserwatywne ograniczenie brzmi:
\[
H_0(x)\le b_{h,0}^{(2)}+\|w_{h,0}^{(2)}\|_2
[\|W_h^{(1)}\|_2(\|\gamma\|_\infty\sqrt d+\|\beta\|_2)
+\|b_h^{(1)}\|_2],
\]
bo \(|\operatorname{GELU}(a)|\le|a|\).

Cel_w_latencie.py:51–55 minimalizuje z jednego startu
\[
-H_0(x)+100\|\exp H_{1:}(x)-d^*\|^2.
\]
To nie jest dowód maksimum H_0 przy dokładnej ramce. W zapisanym wyniku nawet ramka różni się od celu; np. krok1100 ma t=0.27 zamiast0.25. Wartości 1.86,5.31,6.27,6.89 to znalezione logity z określonym błędem ramki. Nie dowodzą ani wielkości sufitu, ani nieosiągalności logitu9.21.

Część B (wiersze58–63) również rozwiązuje problem z miękkimi karami. Przykład: w kroku1100 odległość t wynosi0.36 zamiast0.25. Zatem „najbliższy punkt z ramką równą celowi” nie jest spełniony przez zapisany kandydat, niezależnie od problemu globalnej optymalności.

### Wymiar187: poprawna wersja lokalna

Dla \(H:\mathbb R^{192}\to\mathbb R^5\), jeżeli rank DH(x0)=5, to lokalnie poziomica H(x)=H(x0) jest gładką rozmaitością wymiaru187. Jej przestrzeń styczna to ker DH(x0). Nie wynika z tego:
- że cała poziomica jest afiniczną podprzestrzenią;
- że przesunięcie o dowolną skończoną długość w ker DH zachowuje wyjście;
- że poziomica innego celu jest niepusta lub ma wszędzie ten sam rząd;
- że 187 kierunków pozostaje tymi samymi podczas uczenia.

Dla prawdziwego celu z clamp i dokładną ramką mamy cztery równości oraz nierówność logitu. Wnętrze wykonalnego plateau ma generically wymiar188, a jego regularna granica wymiar187. Wymiar187 z Jacobianu pięciu wyjść dotyczy innego pytania: zachowania dokładnie ustalonego logitu i ramki.

Gradient lokalnej straty końcowego tokenu leży w row(DH) w danym punkcie. Całkowity ruch tokenu przy aktualizacji wspólnych wag nie jest przez to ograniczony do tej podprzestrzeni.

## 8. Zgodność pomiaru z modelem i precyzja

### Epsilon LayerNorm

Oryginalny model.py:42 tworzy net.norm=nn.LayerNorm(dim), bez zmiany epsilon. Zainstalowany kod PyTorch w F:/Python/Lib/site-packages/torch/nn/modules/normalization.py:180 ma domyślną wartość 1e-5. Lokalne timm Block używa norm_layer=nn.LayerNorm (vision_transformer.py:136,140,153), więc również nie narzuca opisanej w dokumentacji wartości1e-6. Odczyt utworzonych obiektów w lokalnym runtime potwierdził: torch2.6.0+cu124, timm0.9.16, norm1.eps=norm2.eps=1e-5, domyślna końcowa LayerNorm.eps=1e-5; ls1, ls2 i q_norm to Identity. Nie wykonano forward ani uczenia w tej kontroli.

Tymczasem pomiar.py:51 i86 rekonstruuje głowę przy epsilon1e-6. Oznacza to inną funkcję od oryginalnej głowy w lokalnej implementacji. Skala różnicy może być mała przy dużej wariancji tokenów, ale „dokładna zgodność” wymaga pomiaru. Checkpoint state_dict nie przechowuje epsilon. Nie stwierdzono tutaj wersji biblioteki faktycznie użytej do treningu na Edenie, dlatego oryginalny runtime trzeba dokumentować oddzielnie. App/ogniskowa.py pobiera epsilon z obiektu norm2, co jest poprawnym sposobem.

### Kwantyzacja przed różnicowaniem

Pomiar.py:57 tworzy XF i GF jako float16, a:84 zaokrągla do nich stany i gradienty. Dopiero:103–110 konwertuje je z powrotem na float32. Odczyt istniejącego pomiar.npz wykazał X_wyb.shape=(806,8,192), dtype=float32 oraz G_wyb.dtype=float32, lecz pochodzenie tych liczb z float16 pozostaje w kodzie.

Konwersja do float32 przed odejmowaniem ogranicza błąd arytmetyki odejmowania, lecz nie odzyskuje utraconych bitów wejścia. Może tworzyć podłogę szumu w małych przyrostach i zaburzać słabe gradienty. Nie wyznaczono tutaj wielkości tego efektu względem pełnego ponownego odczytu checkpointów, więc nie należy twierdzić, że unieważnia wszystkie liczby. Należy natomiast poprawić deklarację precyzji i oszacować wpływ przed interpretacją subtelnych efektów.

### Dodatkowy szczegół gradientu przed ostatnim MLP

Pomiar.py:74–81 oblicza gp jako gradient pełnego złożenia \(L(H(X_p+F_M(X_p))))\) względem Xp. Jest to gradient po przejściu przez Jacobian ostatniego MLP, a nie gradient \(L(H(X_p))\) oceniony na stanie przed MLP. Cosinus z push powinien być opisany dokładnie tą definicją. Osobno liczone dL_push jest poprawną skończoną różnicą straty głowy między Xp i Xp+push.

## 9. Co można obecnie napisać jako twierdzenie i co wymaga kontroli

**Obecnie obronne:**

1. Architektura pozwala dokładnie rozłożyć resztową aktualizację MLP na768 podpisanych wkładów oraz aktualizację uwagi na wkłady źródeł, z jawnymi biasami.
2. Przy wspólnym rzucie i poprawnym centrowaniu suma rysowanych wkładów jest równa rysowanemu przemieszczeniu.
3. Jacobian MLP ma podaną wyżej dokładną faktoryzację; jego symetryczna i antysymetryczna część opisują lokalną geometrię.
4. Zbadane pole3D jest jawnie określoną warunkową interwencją, którą można porównywać między checkpointami.
5. Obserwowany ruch przy stałym obrazie jest wynikiem ewolucji wspólnego modelu. Można mierzyć jego statystyki i ich zmianę z czasem.
6. Głowa ograniczona przez LayerNorm ma skończone ograniczenie logitu przy stałych wagach; poziomice jej pełnego wyjścia są lokalnie187-wymiarowe przy rzędzie5.

**Kontrole, które rozstrzygają obecne luki bez zmiany pytania badawczego:**

- Oddzielić pełny Jacobian od query-only; potwierdzić Jacobian pełnego przejścia przez JVP propagowane przez wszystkie podkroki. Macierzy115200×115200 nie trzeba materializować, by śledzić kilka perturbacji.
- Raportować pełny lub osadzony przyrost objętości osobno od obrazu po rzucie; porównywać stały rzut, kilka niezależnych rzutów i geometrię w pełnym wymiarze.
- Dla OU użyć rzeczywistych czasów checkpointów, tego samego obserwabla w estymacji i teście, osobnego okna walidacyjnego, kontrolowanego modelu centrum i surogatów przechodzących identyczny filtr. Sprawdzić białość innowacji, kowariancję wielolagową oraz stabilność wyników na oknach. Nazwać 1/var odwrotnością wariancji do chwili identyfikacji temperatury.
- Dla maksimum/minimum latentu podawać osiągnięte błędy ograniczeń, wiele startów i niezależny warunek optymalności. Certyfikowany globalny „sufit” wymaga osobnego ograniczenia górnego, nie tylko dobrego kandydata.
- Zweryfikować epsilon rzeczywistego modelu i analitycznej głowy oraz wpływ kwantyzacji; potem oceniać, które istniejące liczby potrzebują ponownego pomiaru.
- Rozstrzygnięcia „pułapka”, „wiry Browna”, „temperatura”, „prawo relaksacji spowodowane celem w nieskończoności” traktować jako hipotezy mechanistyczne, nie etykiety wynikające z samych wykresów.

## 10. Źródła kodowe i wykonane kontrole

Główny katalog:
F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation

Odczytano:
- OpisUkładu/01_OPIS_UKLADU.md
- app/ZASADY_DZIALANIA_UKLADU.md
- app/lasery_dane.py, app/pole_podkrokow.py, app/pulapka_obliczenia.py, app/ogniskowa.py
- experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py i trening.py
- experiments/pulapka_cel_koloidu_20261015/pulapka_uczenia.py, analiza.py, cel_w_latencie.py, pomiar.py
- outputs/pulapka_cel_koloidu_20261015/PULAPKA_UCZENIA.md, CEL_W_LATENCIE.md i pomiar.npz
- lokalne pliki implementacji timm Block oraz torch LayerNorm wskazane w sekcji8.

Kontrole bez treningu:
- dokładny próg clamp: ln9999=9.21024036697585;
- wartość dodatniej focal loss na plateau: 1.0000500033334732e-12;
- analityczne phi reszty random walk: 0.5 dla okna11, 0.7272727272727273 dla okna21;
- rzeczywiste odstępy późnego szeregu wybranego przez kod: {100:530,200:57};
- kształt i dtype zapisanego pomiaru opisane w sekcji8.

Nie zmieniono dokumentacji źródłowej, aplikacji, modeli ani wyników eksperymentów. Niniejszy plik jest nowym raportem audytu.

