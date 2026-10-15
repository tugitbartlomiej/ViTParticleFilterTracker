# Aneks: niezależny audyt matematyczny analogii reflektorów i tokenów ViT

Data: 2026-10-12. Zakres: zasady Z1–Z15, opis układu optycznego, matematyka reflektorów oraz zgodność ich równań z definicją badanego modelu.

Rdzeń analogii jest poprawny: podwarstwa MLP rozkłada przyrost tokenu na sumę 768 składników `z_c v_c` oraz bias. Dokumenty oznaczają jednak jako „dokładne” także wnioski, które nie wynikają z tego rozkładu. Najważniejsze korekty dotyczą konieczności przeciwlaserów, wnioskowania ze znaku cosinusa, znaczenia „siodła” oraz rozróżnienia Jakobianu pola od Jakobianu rzeczywistego kroku.

Nie zmieniono plików źródłowych ani nie uruchomiono istniejących eksperymentów. Audyt opiera się na czytaniu kodu, wyprowadzeniach algebraicznych i małych obliczeniach wartości GELU. Liczby dotyczące checkpointów przytoczone ze źródeł nie zostały tu ponownie zmierzone. Weryfikacja literatury stanowi osobny zakres głównego raportu.

## 1. Źródła i oznaczenia

- **Z**: [ZASADY_LASER_KOLOID.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md).
- **O**: [OPIS_UKLADU_OPTYCZNEGO_VIT.tex](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex).
- **M**: [matematyka_reflektorow.tex](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex).
- **Kod**: [model.py](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py).

W całym aneksie wektory zapisujemy jako kolumny. Liczba tokenów wynosi `N=600`, szerokość strumienia rezydualnego `d=192`, szerokość ukryta MLP `m=768`. Symbol `x_j` w analizie samego MLP oznacza jego stan wejściowy po podwarstwie uwagi, ale przed LayerNorm poprzedzającą MLP. W analizie całego bloku stan po uwadze oznaczamy osobno przez `x̃_j`.

## 2. Dokładny szkielet obliczeń

### 2.1. Trzy reprezentacje tego samego tokenu

\[
x_j\in\mathbb R^{192},\qquad
z_j=g(W_1\operatorname{LN}_2(x_j)+b_1)\in\mathbb R^{768},\qquad
F_j=W_2z_j+b_2\in\mathbb R^{192}.
\]

`x_j` jest stanem strumienia rezydualnego, `z_j` aktywacją ukrytą po GELU, a `F_j` przyrostem dodawanym przez MLP. Są to różne obiekty i nie należy utożsamiać ich położeń, odległości ani trajektorii.

[Hook modelu zapisuje latent po GELU, przed fc2](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py:51). Reprezentacja użytkownika w przestrzeni 768D jest zatem prawidłowo zdefiniowanym położeniem tokenu w konkretnym miejscu sieci. Nie trzeba rezerwować słowa „cząsteczka” wyłącznie dla `x192`; trzeba jawnie nazywać wybraną reprezentację.

Zgodnie ze słownikiem użytkownika makrowektory latentu 768D są wspólnymi osiami `e_c`, a mikrowektory cząsteczki mają postać `z_jc e_c`. Kolumny `v_c` macierzy `W₂` należą do innej przestrzeni, 192D. Są kierunkami zapisu MLP, nie bazą latentu 768D. Ponieważ 768 kolumn leży w przestrzeni 192D, nie tworzą one liniowo niezależnej bazy.

### 2.2. Wielogłowa uwaga i MLP

Model ma [trzy głowy oraz `qkv_bias=True`](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py:38). W modelu bez aktywnego dropout, dla głowy `h=1,2,3`:

\[
u_j=\operatorname{LN}_1(x_j),\qquad
q_j^h=W_Q^hu_j+b_Q^h,\qquad
k_j^h=W_K^hu_j+b_K^h,\qquad
r_j^h=W_V^hu_j+b_V^h,
\]

\[
A_{jk}^h=
\frac{\exp((q_j^h)^\top k_k^h/\sqrt{64})}
{\sum_n\exp((q_j^h)^\top k_n^h/\sqrt{64})},
\]

\[
H_j(X)=b_O+\sum_{h=1}^{3}W_O^h\sum_k A_{jk}^h r_k^h,
\qquad \widetilde x_j=x_j+H_j(X),
\]

\[
z_j=g(W_1\operatorname{LN}_2(\widetilde x_j)+b_1),\qquad
x_j^+=\widetilde x_j+W_2z_j+b_2.
\]

`W_O^h` oznacza odpowiedni blok kolumn projekcji wyjściowej. Jeden softmax `A_jk` i pojedynczy iloczyn `W_O W_V` mogą opisywać schemat jednej głowy, ale nie pełną wielogłową uwagę tego modelu.

### 2.3. LayerNorm i dokładny Jakobian MLP

Niech

\[
C=I-\frac{\mathbf1\mathbf1^\top}{d},\quad
u=Cx,\quad \sigma^2=\frac{\|u\|^2}{d}+\epsilon,\quad
D_\gamma=\operatorname{diag}(\gamma).
\]

Wtedy

\[
\operatorname{LN}(x)=D_\gamma\frac{u}{\sigma}+\beta,
\qquad
J_{\operatorname{LN}}(x)=D_\gamma
\left(\frac C\sigma-\frac{uu^\top}{d\sigma^3}\right).
\]

Dla neuronu `c`, `w_cᵀ` jest wierszem `W₁`, a `v_c` kolumną `W₂`:

\[
a_c=w_c^\top\operatorname{LN}(x)+b_c,\quad
z_c=g(a_c),\quad F(x)=\sum_c z_cv_c+b_2.
\]

\[
J(x)=\frac{\partial F}{\partial x}
=W_2\operatorname{diag}(g'(a))W_1J_{\operatorname{LN}}(x).
\]

Ten wzór jest poprawnym rdzeniem Z15. Jest Jakobianem przyrostu `F`, nie całego kroku `x↦x+F(x)`. Jakobianem kroku jest `I+J`.

## 3. Przeciwlasery: użyteczny układ, lecz nie konieczność

**Miejsca wymagające korekty:** [Z5](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:47), [O:212–215](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:212), [O:421–422](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:421), [M:71](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:71).

Fałszywe są twierdzenia, że silne pchnięcie w stronę `−v_c` wymaga kolumny `v_c′≈−v_c`, albo że ruch w obie strony osi wymaga przeciwpary. Dodatnia kombinacja kilku innych kierunków może dać dokładnie kierunek przeciwny.

Kontrprzykład: cztery kierunki wierzchołków foremnego czworościanu:

\[
v_1=(1,1,1),\quad v_2=(1,-1,-1),\quad
v_3=(-1,1,-1),\quad v_4=(-1,-1,1).
\]

Każda para ma cosinus `−1/3`, zatem żadna nie spełnia nawet przyjętego progu `cos<−0,5`. Jednak

\[
v_2+v_3+v_4=-v_1.
\]

Dodatnie aktywacje trzech neuronów dają dokładnie przeciwne pchnięcie. Analogicznie regularny simpleks w `d` wymiarach pozwala dodatnio generować wszystkie kierunki przy `d+1` wektorach i cosinusie par `−1/d`. Ograniczenia konkretnej sieci mogą ograniczać dostępne aktywacje, ale nie przywracają postulowanej ogólnej konieczności przeciwpar.

Dla dokładnie antyrównoległej pary `v_c=αe`, `v_c′=−βe` poprawny wzór brzmi

\[
z_cv_c+z_{c'}v_{c'}=(αz_c-βz_{c'})e.
\]

Równe aktywacje znoszą siły tylko wtedy, gdy długości wektorów są równe. Przy przybliżonej antyrównoległości pozostaje również składowa poprzeczna. Dla dowolnej osi jednostkowej `e` projekcja całej wypadkowej wynosi

\[
e^\top F=\sum_c z_c(e^\top v_c)+e^\top b_2.
\]

**Proponowane brzmienie:** „Antyrównoległe kierunki umożliwiają prostą, dwuneuronową kontrolę obu zwrotów osi. Nie są konieczne: przeciwny przyrost może powstać przez sumę wielu składników, ujemne aktywacje oraz bias.”

Wynik 53 procent opisuje odsetek neuronów mających przynajmniej jeden kierunek o cosinusie poniżej progu. Nie opisuje 53 procent rozłącznych par. Wybór partnera przez minimum cosinusa może być wielokrotny i niewzajemny. Sam pomiar kolumn `W₂` nie dowodzi przeciwstawnej selektywności wejściowej ani funkcjonalnego sterowania osią. Wzrost ujemnych cosinusów należy także porównać z modelem zerowym zachowującym wyuczoną anizotropię kierunków; nie tylko z izotropowymi wektorami w 192D.

## 4. GELU: znak i niemonotoniczność

**Miejsca wymagające korekty:** [Z3](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:43), [Z6](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:63), [O:175–176](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:175), [M:65](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:65).

Dla dokładnego GELU:

\[
g(a)=a\Phi(a),\qquad
g'(a)=\Phi(a)+a\phi(a),\qquad
g''(a)=(2-a^2)\phi(a).
\]

Małe obliczenie z użyciem `math.erf` i bisekcji dało:

\[
a_{\min}=-0{,}7517915247,\qquad
g(a_{\min})=-0{,}1699712075,
\]

\[
g'(-1)=-0{,}0833154706,\qquad
\min g'\approx-0{,}1289041452,\quad
\max g'\approx1{,}1289041452.
\]

Na części ujemnej osi wzrost pobudzenia zmniejsza aktywację. Stwierdzenie „im lepiej cząstka leży na linii celowania, tym mocniej świeci laser” wymaga ograniczenia do monotonicznej części odpowiedzi. „Próg” jest opisem punktu zmiany znaku pobudzenia i odpowiedzi, nie twardym przełącznikiem neuronu.

**Proponowane brzmienie:** „Aktywacja jest podpisanym współczynnikiem odpowiedzi neuronu. Jej znak określa zwrot składnika `z_cv_c`, a moduł wraz z `‖v_c‖` określa jego długość. GELU nie jest wszędzie monotoniczne.”

Jasność w wizualizacji może kodować `|z_c|`, a znak można pokazać kolorem i zwrotem strzałki. Obcięcie `z<0` do zera zmienia przedstawiane pole. Ograniczenie pojedynczego ujemnego składnika wynosi `0,16997‖v_c‖`; nie gwarantuje małego wpływu względem wypadkowej wielu składników.

Wskaźnik 9–24 procent z kontroli obcięcia jest stosunkiem norm ujemnej składowej i całej wypadkowej po centrowaniu. Takich stosunków nie należy interpretować jako addytywnych udziałów procentowych siły: sumujące się wektory mogą się znosić.

## 5. Ujemny cosinus nie implikuje ujemnego śladu

**Miejsca wymagające zasadniczej korekty:** [Z6](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:63), [O:218–226](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:218), [tabela O:281](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:281), [przypis O:532–535](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:532).

Niech `s_c=D_γw_c`. Rzeczywisty lokalny gradient pobudzenia w przestrzeni strumienia rezydualnego wynosi

\[
g_c(x):=\nabla_xa_c
=\left(\frac C\sigma-\frac{uu^\top}{d\sigma^3}\right)s_c.
\]

Nie jest nim samo `s_c=γ⊙w_c`. Ten ostatni wektor jest kierunkiem odczytu przed centrowaniem i zależnym od tokenu normowaniem.

Dokładnie:

\[
J=\sum_c g'(a_c)v_cg_c(x)^\top,
\qquad
\operatorname{tr}J=\sum_c g'(a_c)v_c^\top g_c(x),
\]

czyli

\[
\operatorname{tr}J=
\sum_c g'(a_c)\left[
\frac{s_c^\top Cv_c}{\sigma}
-\frac{(s_c^\top u)(u^\top v_c)}{d\sigma^3}\right].
\]

Przybliżenie przez `σ⁻¹Σg′(a_c)s_cᵀv_c` usuwa projekcję centrowania i człon radialny. Dokumenty nie podają pomiaru błędu tego przybliżenia. Ponadto `g′` może być ujemne, a mediana cosinusów nie określa znaku sumy ważonej.

Kontrprzykład nawet przy dodatnim `g′`: niech `e_r=u/‖u‖`, a `e_t` będzie jednostkowym kierunkiem prostopadłym do `u` i `1`. Wybierz

\[
s=e_r+e_t,\qquad v=-2e_r+e_t.
\]

Wtedy `sᵀv=−1`, lecz

\[
v^\top\left(\frac C\sigma-\frac{uu^\top}{d\sigma^3}\right)s
=\frac1\sigma-\frac{2\epsilon}{\sigma^3}>0
\]

przy `σ²>2ε`. Ujemny cosinus może dać dodatni wkład do śladu. Inny mechanizm odwrócenia znaku pojawia się tam, gdzie pochodna GELU jest ujemna.

**Proponowane brzmienie:** „Ujemna mediana cosinusów opisuje zmianę geometrii wag. Zmierzony ujemny ślad potwierdza lokalnie ujemną dywergencję pola. Powiązanie obu obserwacji wymaga oceny dokładnych składników `g′(a_c)v_cᵀg_c(x)`.”

Pomiar ujemnego śladu pozostaje wartościowy. Nie jest natomiast ilościowo wyjaśniony samą medianą cosinusów.

## 6. Geometria LayerNorm i neutralne kierunki

**Miejsca wymagające doprecyzowania:** [O:152–160](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:152), [O:509–512](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:509), [M:79–85](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:79).

Dla `q=Cx/σ`:

\[
\|q\|^2=
\frac{d\|Cx\|^2}{\|Cx\|^2+d\epsilon}<d.
\]

Przy dodatnim epsilon obraz przed `γ,β` wypełnia otwartą kulę w hiperpłaszczyźnie `1⊥`, nie powierzchnię sfery. Gdy wariancja jest duża względem epsilon, punkty leżą bardzo blisko sfery promienia `√d`. W granicy `ε=0`, dla niezerowej wariancji, powierzchnia ma wymiar `d−2`, czyli 190 przy `d=192`. Po odwracalnym diagonalnym przeskalowaniu i przesunięciu jest to elipsoidalna powierzchnia w odpowiedniej afinicznej podprzestrzeni. Dla `ε>0` otrzymujemy jej wnętrze w tej podprzestrzeni.

**Proponowane brzmienie:** „LayerNorm dokładnie usuwa średnią, a przy wariancji znacznie większej od epsilon prawie usuwa informację o długości. Odczyt można wtedy przybliżyć przez punkt na sferze w wycentrowanej podprzestrzeni, następnie przeskalowany i przesunięty przez wyuczone parametry.”

Istotny skutek dokładnej niezmienniczości względem przesunięcia:

\[
\operatorname{LN}(x+c\mathbf1)=\operatorname{LN}(x),\qquad
F(x+c\mathbf1)=F(x),
\]

\[
J\mathbf1=0,\qquad (I+J)\mathbf1=\mathbf1.
\]

MLP z pre-LN nie może zatem być ścisłą kontrakcją w całej `R^192`, niezależnie od wag. Jeśli `F(x*)=0`, to także `F(x*+c1)=0`; taki punkt równowagi nie jest izolowany. Przy `ε→0` dochodzi neutralność radialna, podczas gdy dla rzeczywistego epsilon:

\[
J_{\operatorname{LN}}u=D_\gamma\frac{\epsilon u}{\sigma^3}.
\]

Wpływ radialny jest wtedy mały względnie do wpływu stycznego, jeśli `ε/σ²` jest małe, ale nie jest dokładnie zerowy.

## 7. Jakobian nie dowodzi istnienia siodła ani pułapki

**Miejsca wymagające zasadniczej korekty:** [Z15](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:104), [O:293–296](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:293), [O:336–338](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:336), [M:121–123](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:121).

Linearyzacja

\[
F(x+\delta)=F(x)+J(x)\delta+O(\|\delta\|^2)
\]

jest poprawna. Aby mówić o równowadze, trzeba jednak wykazać `F(x*)=0`. Dokumenty analizują stany tokenów, dla których raportują niezerowe przyrosty, i nie wyznaczają równowag.

Kontrprzykład:

\[
F(x,y)=(1,x),\qquad
J=\begin{pmatrix}0&0\\1&0\end{pmatrix}.
\]

Pole nie ma żadnego punktu równowagi. Mimo to jego część symetryczna ma wartości własne `±1/2`, a część antysymetryczna jest niezerowa. Mieszane znaki `S` oraz `A≠0` nie wystarczają więc do nazwania punktu „siodłem z wirem”.

Nawet w punkcie równowagi mieszane znaki `S` nie klasyfikują stabilności widma `J`. Dla

\[
J=\begin{pmatrix}-1&10\\0&-1\end{pmatrix}
\]

część symetryczna ma wartości własne `−6,4`, lecz obie wartości własne `J` wynoszą `−1`. Ciągły układ liniowy jest asymptotycznie stabilny mimo możliwego przejściowego wzrostu zaburzeń.

**Proponowane brzmienie:** „W badanych położeniach tokenów pole ma lokalne kierunki chwilowego zbliżania i oddalania zaburzeń oraz silną część antysymetryczną. Nie wyznaczono punktu równowagi; określenie ‘siodło z wirem’ jest metaforą lokalnego odkształcenia.”

Słowo „pułapka” może pozostać nazwą ilustracji lokalnej odpowiedzi, lecz powinno mieć status analogii. Nie należy przenosić klasyfikacji punktów krytycznych z innej pracy na stany tokenów, które nie zostały sprawdzone jako punkty krytyczne.

## 8. Część symetryczna, wir i rzeczywisty skończony krok

**Miejsca wymagające korekty:** [O:312–316](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:312), [O:319–349](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:319), [M:97–102](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:97).

Niech

\[
S=\frac{J+J^\top}{2},\qquad A=\frac{J-J^\top}{2}.
\]

W pomocniczym przepływie ciągłym `δ̇=Jδ`:

\[
\frac{d}{dt}\|\delta\|^2=2\delta^\top S\delta.
\]

Wartości własne `S` opisują chwilowy wzrost lub spadek długości w jego ortogonalnych kierunkach własnych. Liczba ujemnych wartości własnych nie jest odsetkiem wszystkich losowych kierunków, które się skracają, ani liczbą kierunków skracanych w skończonym kroku sieci.

Dla rzeczywistego kroku `T(x)=x+F(x)`:

\[
\delta^+=(I+J)\delta+O(\|\delta\|^2),
\]

\[
\|\delta^+\|^2-\|\delta\|^2
=\delta^\top(2S+J^\top J)\delta+O(\|\delta\|^3).
\]

Należy zatem rozdzielić cztery miary:

| Miara | Co opisuje |
|---|---|
| Wartości własne `S` | Chwilową zmianę normy w pomocniczym przepływie ciągłym |
| Wartości singularne `I+J` | Rozciąganie i ściskanie ortogonalnych kierunków wejściowych w jednym rzeczywistym kroku |
| `|1+λ(J)|`, `ρ(I+J)` | Mody i asymptotyczny wzrost powtarzanej, zamrożonej mapy liniowej |
| `|det(I+J)|` | Lokalny mnożnik objętości w jednym kroku |

Przy nienormalnym `J` wartości własne nie opisują w pełni zmian długości. Na przykład podana wyżej macierz z diagonalą `−1` daje `I+J=[[0,10],[0,0]]`: wszystkie wartości własne mapy są zerowe, ale największa wartość singularna wynosi 10.

„A tylko obraca, nie oddala” jest dokładne dla `δ̇=Aδ`, lecz nie dla `δ⁺=(I+A)δ`:

\[
\|(I+A)\delta\|^2=\|\delta\|^2+\|A\delta\|^2.
\]

Dla `A=[[0,−1],[1,0]]` krok Eulera zwiększa każdą długość o `√2`. Zachowujący długość obrót daje `exp(A)`, nie `I+A`.

Również ujemny ślad nie gwarantuje zmniejszenia objętości skończonego kroku. Dla skalarnego `J=−3` ślad jest ujemny, lecz `I+J=−2` rozszerza długość dwukrotnie.

**Dobra korekta już obecna w źródłach:** [O:344–346](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:344) oraz [Z:65](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:65) podają `log|det(I+J)|`. Mnożnik dotyczy nieskończenie małego otoczenia w przestrzeni 192D. Nie dowodzi zmniejszenia każdej odległości ani zagęszczenia rzeczywistej chmury tokenów. Z zaokrąglonych wartości `−0,0008` i `−0,021` na wymiar wynikają odwrotne mnożniki około 1,166 i 56,37; opisy „około 1,2” i „około 55” są zgodne z takim zaokrągleniem.

## 9. Zachowawczość pola: uwzględnienie LayerNorm

**Miejsca wymagające korekty:** [M:100–101](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:100), [O:550–552](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:550).

Dla gładkiego pola w obszarze otwartym i jednospójnym symetria Jakobianu w całym obszarze jest warunkiem koniecznym i wystarczającym istnienia potencjału. W badanym bloku warunek brzmi

\[
W_2D_{g'}W_1J_{\operatorname{LN}}
=J_{\operatorname{LN}}^\top W_1^\top D_{g'}W_2^\top.
\]

Warunek dla dwuwarstwowej sieci bez LN nie przenosi się automatycznie na jej złożenie z LN. `v_c∝w_c` wystarcza dla odpowiedniej postaci MLP bez LN, lecz iloczyn dwóch macierzy symetrycznych po dodaniu LN może nie być symetryczny, jeśli macierze nie komutują. Również `v_c∝γ⊙w_c` nie usuwa zależności `J_LN(x)` od tokenu.

Brak proporcjonalności wag nie dowodzi niezachowawczości. To niespełnienie jednego warunku wystarczającego, nie negacja warunku koniecznego. W tych pomiarach niezachowawczość uzasadnia niezerowy `A`.

`S` jest częścią potencjalną lokalnego pola liniowego. Nie stanowi automatycznie globalnego składnika potencjalnego nieliniowego pola; punktowe rozkładanie Jakobianu nie jest pełnym rozkładem pola na część potencjalną i wirową. W konwencji fizycznej `F=−Kδ` symetryczną sztywnością jest `−S`, nie `S`.

Stosunek norm `‖A‖/‖S‖` opisuje lokalne pochodne w wybranej metryce współrzędnych. Jest niezmienny przy ortogonalnej zmianie bazy, lecz może zmieniać się przy anizotropowym przeskalowaniu współrzędnych. Nie należy traktować go bez dodatkowych założeń jako uniwersalnego procentu siły niezachowawczej.

## 10. Uwaga, łodyga i analogia dyfuzji

**Miejsca wymagające korekty:** [O:73–74](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:73), [M:46–53](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:46), [Z7](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:67).

Równania uwagi w obu dokumentach zawierają jeden softmax i pomijają biasy. Powinny być zastąpione wzorem z sekcji 2.2 albo wyraźnie podpisane jako schemat jednej głowy. Uruchomienie rzeczywistego modelu i uzyskanie zgodnych aktywacji nie weryfikuje uproszczonego równania, którego skrypt nie implementuje osobno.

Uwaga jest mechanizmem zależności reprezentacji tokenu od innych tokenów. Określenie „siły między cząstkami” jest użyteczną analogią, ale oddziaływania są ogólnie asymetryczne, niecentralne i nie spełniają zasady akcji i reakcji. Dodatnie, sumujące się do jedności wagi softmax dotyczą mieszania wartości w obrębie głowy. Projekcje wartości i wyjścia oraz połączenie rezydualne nie gwarantują dyfuzji ani zmniejszania wszystkich odległości.

„Tylko uwaga miesza tokeny” jest trafne po ograniczeniu do sześciu bloków transformera. [Łodyga konwolucyjna](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py:33) miesza obszary obrazu, a BatchNorm w trybie treningowym sprzęga przykłady i pozycje przez wspólne statystyki. Cztery konwolucje `3×3, stride2` dają krok siatki 16 pikseli, lecz nominalne pole recepcyjne 31 pikseli. Tokeny nie odpowiadają niezależnym, niepokrywającym się patchom 16×16.

Cały blok jest dokładnie złożeniem dwóch map rezydualnych. Interpretacja jako rozszczepiony krok Eulera jest dopuszczalna po zdefiniowaniu pomocniczego pola i kroku czasu. Nie jest to jednak dokładny przepływ ciągły ani jeden zwykły krok Eulera sumy pól: drugi przyrost jest liczony w stanie zmienionym przez pierwszy. Kolejne bloki mają różne parametry, więc powtarzanie zamrożonego pola ostatniej warstwy jest dodatkową ilustracją.

## 11. Dwa czasy i mechanizm uczenia

**Miejsca wymagające doprecyzowania:** [Z1](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:39), [Z8–Z9](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:69), [O:120–134](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:120).

Rozróżnienie głębokości i kroków uczenia jest trafne. Token nie ma w czasie uczenia osobnego równania integrującego siły w przestrzeni aktywacji. Dla ustalonego obrazu jego reprezentacja jest ponownie obliczana:

\[
z_{I,j}^{(\ell)}(t)=f_{\ell,j}(I;\theta_t,b_t).
\]

`b_t` obejmuje bufory modelu, zwłaszcza średnie i wariancje BatchNorm. Odczyt musi mieć określony tryb, np. `eval`. Argumentem jest cały obraz `I`, nie tylko izolowany patch, ponieważ wcześniejsze konwolucje i uwaga wpływają na token.

Pierwszy rząd zmiany przy ustalonym obrazie:

\[
\Delta z_j=G_{\theta,j}\Delta\theta+G_{b,j}\Delta b
+O(\|\Delta(\theta,b)\|^2).
\]

Zmiana obrazu lub augmentacji dodaje kolejny składnik. Wspólny napęd przez parametry nie implikuje standardowej dynamiki NTK bez uwzględnienia optymalizatora, preconditionera, pędu, clippingu i buforów.

**Proponowane brzmienie:** „Trajektoria reprezentacji w czasie uczenia jest obserwacją dynamiki parametrów i buforów modelu. Nie ma dodatkowej bezwładności tokenu pomiędzy zapisanymi stanami sieci.”

Zakaz mówienia o jakimkolwiek opóźnieniu jest zbyt szeroki. Szeregi reprezentacji mogą mieć korelacje i opóźnienia odziedziczone po AdamW, danych i BatchNorm. Nie dowodzi to jednak osobnej lepkości tokenu. Traktowanie neuronów jako cząstek parametrycznych jest użyteczną reprezentacją, nie tożsamością fizyczną ani automatycznym spełnieniem założeń teorii pola średniego przez skończony ViT z AdamW.

„Uczenie jest powolne, aktywacje szybkie” opisuje dwa etapy obliczeń. Ścisła separacja skal czasowych wymaga pomiaru. Duży cosinus między checkpointami oddalonymi o pięć epok nie wyklucza szybkich wychyleń i powrotów pomiędzy nimi.

### 11.1. Co faktycznie przestawia neuron

Jeśli `δ_j=∂L/∂F_j` i `u_j=LN(x_j)`, to

\[
\frac{\partial L}{\partial v_c}=\sum_j z_{jc}\delta_j,
\]

\[
\frac{\partial L}{\partial w_c}
=\sum_j g'(a_{jc})(v_c^\top\delta_j)u_j,
\qquad
\frac{\partial L}{\partial b_c}
=\sum_j g'(a_{jc})(v_c^\top\delta_j),
\]

\[
\frac{\partial L}{\partial b_2}=\sum_j\delta_j.
\]

Suma obejmuje tokeny i przykłady w paczce. To bezpośrednio pokazuje współdzielenie wag: jeden neuron jest aktualizowany na podstawie wielu tokenów naraz.

Schemat AdamW, z gradientem po ewentualnym clippingu `ĝ_t`, można zapisać jako

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)\widehat g_t,\qquad
r_t=\beta_2r_{t-1}+(1-\beta_2)\widehat g_t^{\odot2},
\]

\[
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat r_t}+\epsilon_{\rm Adam}},
\]

gdzie daszki nad momentami oznaczają korektę biasu, a rozpad wag należy stosować zgodnie z grupą parametrów. Bufory BatchNorm mają własną aktualizację. Nie jest to zwykłe przesuwanie reflektora w kierunku wypadkowej jego promieni.

### 11.2. Które wagi bezpośrednio zmieniają zapisany latent

Hook zapisuje `z768` przed `fc2`. Przy ustalonym wejściu, `W₁`, biasie i LN zmiana bieżącego `W₂` nie zmienia tego `z`. Zmienia `F192` oraz gradienty wpływające na przyszłe uczenie. Opis przyczyn dryfu powinien wskazywać konkretną obserwablę oraz grupę parametrów. „Wcześniejsze warstwy” względem hooka mogą obejmować także uwagę w ostatnim bloku.

## 12. Pozycje i strzałki w filmie muszą opisywać ten sam układ współrzędnych

**Miejsca wymagające korekty:** [Z10](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:77), [Z12](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md:91), [O:147–150](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:147).

Pozycja

\[
p_j=PW_2(z_j-\bar z)=P(F_j-\bar F)
\]

przedstawia wycentrowany przyrost, nie stan tokenu. Gdy `W₂` zmienia się w czasie, jej ruch miesza zmianę latentu i zmianę dekodera. Wysoka korelacja odległości tej reprezentacji i `x` nie czyni obu reprezentacji tożsamymi.

Jeśli pokazujemy `p=Px`, rzeczywista strzałka kroku wynosi `PF(x)`. Jeśli pokazujemy `p=P LN(x)`, strzałka powinna wynosić

\[
\Delta p=P[\operatorname{LN}(x+F(x))-\operatorname{LN}(x)],
\]

a nie `PF(x)`. Centrowanie po tokenach, zmienna skala, zmienne PCA lub inne transformacje obrazu również trzeba uwzględnić w tej zgodności.

`−P v̂_c` nie leży ogólnie na sferze 3D: rzut zmienia długość. Dla sfery promienia `R` poprawny punkt to

\[
-R\frac{Pv_c}{\|Pv_c\|},
\]

jeśli projekcja jest niezerowa. Długość strzałki składnika musi nadal uwzględniać `|z_c|‖Pv_c‖`, a zwrot jego znak. Przeciwne wektory pozostają przeciwne pod liniowym rzutem, ale ogólne kąty i bliskość do antyrównoległości mogą się istotnie zmienić.

Grupowanie 460 800 składników według tokenów lub neuronów jest algebraicznie dopuszczalne. Współdzielone wagi trzeba pokazać jako identyczne parametry neuronów dla wszystkich tokenów. Geometria szkła, źródeł i galw pozostaje ilustracją programowalnego przekierowania; z równań sieci nie wynika dosłowna realizowalność układu zwykłą jednorodną szklaną sferą.

## 13. Korekty interpretacji metryk

### 13.1. Odległości po standaryzacji a rzeczywista LayerNorm

[O:361–363](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:361) nazywa odległość bez `γ,β` odległością widzianą przez następną warstwę. [Skrypt rzeczywiście pomija te parametry](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/pomiary_opisu_s47.py:152).

`β` znosi się przy różnicy dwóch reprezentacji. Wspólne, ale anizotropowe `γ` nie znosi się: zmienia metrykę. Wynik opisuje odległości po standaryzacji, a nie dokładną metrykę rzeczywistej następnej LN. Nie ma też obowiązku zastępowania wszystkich surowych odległości taką metryką; wybór zależy od pytania badawczego i powinien być jawny.

Skrócenie odległości dwóch tokenów pod wspólną mapą MLP nie jest dowodem ich wzajemnego przyciągania. MLP działa osobno na każdy token. Lepiej mówić o zbliżaniu lub oddalaniu reprezentacji pod działaniem tej samej mapy.

### 13.2. Udział trzech składowych

[przelicz_zasady_s47.py:89](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/przelicz_zasady_s47.py:89) oraz [pomiary_opisu_s47.py:80](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/pomiary_opisu_s47.py:80) używają SVD niecentrowanego zbioru jednostkowych kierunków. Podawane 3–23 procent jest udziałem energii drugiego momentu, nie wariancji centrowanego PCA. Należy zmienić nazwę miary albo, w nowym pomiarze, jawnie wycentrować dane.

### 13.3. Prokrustes

[przelicz_zasady_s47.py:107–109](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/przelicz_zasady_s47.py:107) używa `R=UVᵀ` bez sprawdzenia `det R=+1`. To dopasowanie ortogonalne, potencjalnie z odbiciem. Jeśli ma być dosłownie obrotem, trzeba sprawdzić determinantę albo wymusić `SO(d)`. Udział redukcji błędu przez dopasowanie nie jest sam w sobie przyczynowym udziałem wspólnego obrotu w mechanizmie uczenia.

### 13.4. Losowy poziom odniesienia

[O:339–340](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex:339) opisuje wartości `0,504` i `0,993` jako dokładnie poziom `0,503` i `0,995`. Są one bliskie wybranemu poziomowi odniesienia, nie identyczne. Iloraz oczekiwanych kwadratów norm nie jest ogólnie oczekiwanym ilorazem norm. Rzeczywisty losowy Jakobian ma ponadto strukturę iloczynu wag, bramek i LN, a nie niezależne wpisy.

### 13.5. Dokładność obliczeń i zakres wniosków

[Główne przeliczenie używało `epsilon=1e−5`](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/przelicz_zasady_s47.py:155), podczas gdy dokument podaje dla LN bloku `1e−6`; późniejsze skrypty kontroli pobierają epsilon z modułu. Mały wpływ może uzasadniać zachowanie zaokrąglonych wartości, ale nie identyczność tych dwóch obliczeń. Zgodność wzoru z autogradem na próbie tokenów weryfikuje implementację pochodnej, nie wszystkie interpretacje fizyczne dokumentu.

Wyniki dla ostatniego bloku i jednego obrazu nie dowodzą własności wszystkich warstw, obrazów, ziaren czy architektur. Dokumenty częściowo zaznaczają ten zakres; należy go utrzymać także w streszczeniach i podpisach filmów.

## 14. Temperatura, dyfuzja i odwracalność

**Miejsca wymagające ostrożniejszego brzmienia:** [M:142–152](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:142), [M:163](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/matematyka_reflektorow/matematyka_reflektorow.tex:163).

Oszacowana macierz dyfuzji `D` nie jest sama temperaturą. Identyfikacja z temperaturą wymaga określenia ruchliwości, relacji fluktuacja–dyssypacja, metryki i modelu szumu. AdamW oraz bufory i niestacjonarność uczenia utrudniają bezpośredni model Langevina w reprezentacji tokenu.

Dla procesu OU

\[
dX=B X\,dt+\sqrt{2D}\,dW
\]

stacjonarna kowariancja spełnia

\[
B\Sigma+\Sigma B^\top+2D=0.
\]

Warunek równowagi szczegółowej dla takiego procesu to

\[
B\Sigma=\Sigma B^\top=-D.
\]

Nie jest nim sama symetria `B`. Zachowawczy dryf w metryce euklidesowej nie gwarantuje równowagi przy dowolnym anizotropowym szumie. Odwrotnie, niesymetryczne `B=−DK` z symetrycznym `K` może opisywać równowagę w odpowiedniej metryce ruchliwości.

Niezerowa asymetria opóźnionych korelacji, przy spełnionych założeniach stacjonarności i właściwej transformacji odwrócenia czasu, jest testem nieodwracalności obserwowanych zmiennych. Nie jest samą produkcją entropii ani uniwersalną ilościową miarą prądu. W niestacjonarnych trajektoriach uczenia asymetria może mieć również źródło w dryfie rozkładu.

Zatem FORMA, OU, ekwipartycja i DCCF są propozycjami modeli lub diagnostyk do osobnej walidacji. Nie powinny być przedstawiane jako już ustalona termodynamika tokenów. Zastrzeżenia obecne w dłuższym dokumencie są tu istotnym atutem i należy zachować je w skrótach.

## 15. Ocena zasad Z1–Z15

| Zasada | Co można zachować | Co poprawić |
|---|---|---|
| Z1 | Rozróżnienie głębokości i kroków uczenia; stan rezydualny przechodzi przez bloki. | „Prawdziwa cząstka” i „stan tylko w głębokości” zastąpić opisem rekurencji w głębokości i obserwacji parametrów w czasie uczenia. |
| Z2 | Oddzielne wektory odczytu `w_c` i zapisu `v_c`; wspólne wagi. | Dodać biasy, LN i rozróżnienie kierunku odczytu od lokalnego gradientu pobudzenia. |
| Z3 | Zależność aktywacji od własnego stanu tokenu, poprawny wzór. | GELU nie jest globalnie monotoniczne; niezachowawczość wynika ze zmierzonego `A≠0`, nie z samej metafory. |
| Z4 | Jeden przyrost na podwarstwę, osobne parametry sześciu bloków. | Przepływ ciągły i powtarzanie ostatniego pola opisywać jako ilustracje. |
| Z5 | Pomiar częstszych ujemnych cosinusów wyjściowych kierunków. | Przeciwpary nie są konieczne; uwzględnić normy, znaki i brak dowodu funkcjonalnych par. |
| Z6 | Dokładny wzór na ślad oraz sam pomiar dywergencji. | Usunąć implikację z mediany cosinusów; uwzględnić pełny `J_LN` i znaki `GELU′`. |
| Z7 | W blokach uwaga miesza tokeny, a MLP działa osobno. | Zawęzić względem łodygi i treningowego BatchNorm; NTK nie opisuje automatycznie całego AdamW. |
| Z8 | Brak osobnego integratora tokenu między checkpointami. | Uwzględnić bufory, ustalony obraz, konkretną obserwablę i parametry przed hookiem; „neurony jako cząstki” to reprezentacja. |
| Z9 | Ostrożność wobec temperatury i utożsamiania korelacji z oddziaływaniem. | „Brak jakiegokolwiek opóźnienia” i „jedyna pułapka” są zbyt kategoryczne; odróżnić pamięć modelu od własnej bezwładności tokenu. |
| Z10 | Jawny rzut oraz osobne pokazywanie odczytu i zapisu. | Normalizować punkty na sferze, pokazać znak GELU, zachować zgodność strzałek z transformacją pozycji. |
| Z11 | Wspólna matryca neuronów, aktualizowana w uczeniu. | Zmiany między snapshotami nie dowodzą płynności; Prokrustes jest ortogonalnym dopasowaniem, nie automatycznie obrotem ani analizą przyczynową. |
| Z12 | Grupowanie składników według tokenów jest algebraicznie dopuszczalne. | `W₂z` jest przyrostem; dla pozycji po LN nie wolno dodawać surowej strzałki `PF`. |
| Z13 | Grupowanie według neuronów dobrze pokazuje współdzielone parametry. | Ujemne aktywacje odwracają zwrot składnika; wspólny `v_c` nie oznacza zawsze wspólnego zwrotu fizycznej strzałki. |
| Z14 | Schemat programowalnego przekierowania wiązek i jawnie umowna geometria. | Nie wywodzić realnej szklanej optyki z równań sieci; zmiany kierunków, aktywacji i odczytu są różnymi elementami. |
| Z15 | Dokładny Jakobian MLP, jego rozkład oraz lokalne pomiary. | Oddzielić odpowiedź pola, skończony krok i równowagę; użyć wartości singularnych kroku do rozciągania, nie nazywać badanego punktu dowiedzionym siodłem. |

## 16. Gotowy rdzeń poprawionego opisu

„Każda cząsteczka oznacza reprezentację jednego tokenu obrazu w wybranym miejscu sieci. W strumieniu rezydualnym ma ona 192 współrzędne, a zapisywany latent po GELU ma 768 współrzędnych. Jeden neuron MLP odczytuje stan tokenu przez LayerNorm i nadaje mu podpisany współczynnik aktywacji. Ta aktywacja skaluje wspólny dla wszystkich tokenów wektor zapisu neuronu. Suma 768 takich składników i biasu jest przyrostem dodawanym w jednym przejściu przez podwarstwę MLP.

Podczas uczenia zmieniają się parametry odczytu, zapisu i pozostałych części modelu. Reprezentacja tego samego tokenu tego samego obrazu jest za każdym razem ponownie obliczana. Jej trajektoria odzwierciedla historię całego procesu uczenia, ale token nie ma dodatkowego integratora sił ani własnej lepkości. Wspólne wagi są aktualizowane na podstawie sygnałów błędu wielu tokenów naraz.

Wokół aktualnego stanu tokenu Jakobian opisuje reakcję przyrostu na małe zaburzenia. Badane pole ma kierunki chwilowego zbliżania i oddalania oraz część antysymetryczną. Nie wyznaczono równowagi, dlatego ‘pułapka’ i ‘siodło z wirem’ pozostają nazwami ilustracji. Rzeczywiste rozciąganie w jednym kroku opisują wartości singularne `I+J`, a zmianę lokalnej objętości jego wyznacznik. Przeciwne kierunki neuronów mogą ułatwiać dwustronną kontrolę przyrostu, ale nie są do niej matematycznie konieczne.”
