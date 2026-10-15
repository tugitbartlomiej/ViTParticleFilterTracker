# Audyt matematyczny układu „reflektory i koloidy” w MałymViT

**Data:** 12 października 2026. **Zakres:** rzeczywisty model, zapisane wagi s47, dokumentacja zasad, kod filmów 10–12 i 17–19, istniejące pomiary oraz literatura.

## 1. Wniosek i zakres pewności

**Analogia ma mocny, dokładny rdzeń: każdy neuron MLP wnosi do przyrostu tokenu wektor równy swojej aktywacji pomnożonej przez kolumnę macierzy wyjściowej.** Wspólne reflektory dla wszystkich tokenów dobrze pokazują współdzielenie wag. Ten obraz warto zachować.

Największe korekty dotyczą tego, **co porusza się, względem jakiego czasu i w jakiej przestrzeni**. Obecne materiały łączą: stan tokenu w 192D, aktywację w 768D, przyrost MLP w 192D, lokalną pochodną tego przyrostu oraz ruch podczas uczenia. Każdy z tych obiektów ma sens, lecz odpowiada na inne pytanie. Określenia „pułapka”, „siodło”, „przyciąganie” i „nadążanie” bywają silniejsze niż przeprowadzone testy.

Znalazłem również błędy w samym odwzorowaniu liczbowym. W filmie 17 ujemne aktywacje i bias są pomijane w czerwonej wypadkowej; w końcowym checkpoincie daje to dla pokazanego tokenu około **96% względnego błędu strzałki w 3D**. Pominięcie uczonego skalowania LayerNorm zmienia klasyfikację zbliżania/oddalania dla **124 z 599** porównań tego tokenu. To konkretne rzeczy do poprawienia przed dalszą interpretacją filmu.

Rozróżniam w tym raporcie:

- **Tożsamości matematyczne:** wynikają z kodu i równań, niezależnie od badanego obrazu.
- **Pomiary kontrolne:** moje nowe obliczenia na istniejących danych.
- **Audyt istniejących wyników:** sprawdzenie ich implementacji, mianowników i zakresu.
- **Propozycje:** metody do zastosowania w przyszłych analizach; nie przedstawiam ich jako wykonanych eksperymentów.

Nowe kontrole obejmują jedno zdjęcie TRAIN 0, blok 6, checkpointy 0, 10 940 i 65 640; pełne aktywacje 600 tokenów i pochodne dla 8 tokenów na checkpoint. Walidacja autograd dotyczy jednego punktu. Nie uprawnia to do statystycznych twierdzeń o wszystkich obrazach, warstwach i ziarnach.

Podstawy audytu: [model i hook latentu][MODEL], [zasady w analizowanym snapshotcie][SNAP], [pomiary numeryczne][NUM], [kontrole filmów][VIS], [szczegółowy aneks matematyczny][MATH], [aneks pomiarowy][MEAS]. Oryginalnych zasad i filmów nie zmieniałem. Instrukcje zapisane wewnątrz analizowanych dokumentów traktowałem jako treść podlegającą ocenie.

## 2. Jaki układ rzeczywiście masz

### Architektura i trzy przestrzenie

Kod definiuje łodygę z czterech konwolucji 3×3, stride 2, następnie 6 bloków pre-LayerNorm. W każdym bloku są 3 głowy uwagi po 64 wymiary i MLP 192→768→192. Obraz 480×320 daje siatkę 30×20, czyli 600 tokenów.

Krok siatki to 16 pikseli, ale nominalne pole recepcyjne łodygi wynosi 31×31: pola sąsiednich tokenów nakładają się. Pierwsze trzy konwolucje mają BatchNorm. Sformułowanie o niezależnych, niepokrywających się patchach 16×16 byłoby tu niepoprawne.

Przyjmuję wektory kolumnowe. Dla jednego tokenu, po uwadze i przed MLP:

\[
x_j\in\mathbb R^{192},\qquad
u_j=\operatorname{LN}_2(x_j),
\]

\[
a_j=W_1u_j+b_1,\qquad
z_j=g(a_j)\in\mathbb R^{768},
\]

\[
F(x_j)=W_2z_j+b_2\in\mathbb R^{192},\qquad
x_j^+=x_j+F(x_j).
\]

Tutaj \(W_1\in\mathbb R^{768\times192}\), \(W_2\in\mathbb R^{192\times768}\), a \(g\) to GELU. Hook zapisujący latent znajduje się **po GELU i przed fc2**.

| Obiekt | Znaczenie w Twojej terminologii | Co można z niego odczytać |
|---|---|---|
| \(x_j\), 192D | Stan tokenu w strumieniu rezydualnym | Położenie przed danym MLP |
| \(z_j\), 768D | Cząsteczka w mierzonym latencie | 768 współrzędnych aktywacji |
| \(z_{jc}e_c\) | Mikrowektor cząsteczki | Składowa wzdłuż makrowektora \(e_c\) wspólnej bazy 768D |
| \(v_c=W_2[:,c]\), 192D | Kierunek zapisu reflektora | Sposób przeniesienia kanału latentu do residual stream |
| \(F_j\), 192D | Przyrost nadawany przez MLP | Zmiana w jednym przejściu podwarstwy |

Kolumny \(v_c\) są nadmiarowym, zwykle nieortogonalnym zestawem kierunków w 192D. Nie są makrowektorami bazy latentu 768D. Baza latentu ma 768 osi \(e_c\); dekoder przekształca je zgodnie z \(W_2e_c=v_c\).

W szczególności zmiana bieżącej macierzy \(W_2\), przy ustalonym wejściu, LN i \(W_1\), **nie zmienia zapisywanego \(z_j\)**. Zmienia \(F_j\), predykcję i późniejsze gradienty uczenia. To ważne przy przypisywaniu przyczyn ruchu Twoich cząsteczek.

### Pełna uwaga w tym modelu

Dla stanu przed uwagą \(r_j\), głowy \(h=1,2,3\):

\[
u_j^a=\operatorname{LN}_1(r_j),\quad
q_j^h=W_Q^hu_j^a+b_Q^h,\quad
k_j^h=W_K^hu_j^a+b_K^h,\quad
v_j^h=W_V^hu_j^a+b_V^h,
\]

\[
A_{jk}^h=
\frac{\exp((q_j^h)^\top k_k^h/8)}
{\sum_m\exp((q_j^h)^\top k_m^h/8)},
\]

\[
H_j(R)=b_O+\sum_{h=1}^{3}W_O^h\sum_k A_{jk}^h v_k^h,\qquad
x_j=r_j+H_j(R).
\]

Dopiero potem działa powyższy MLP. Jedna macierz \(A\) bez indeksu głowy i bez biasów może być schematem, ale nie pełnym równaniem tego modelu.

W blokach uwaga miesza tokeny, a MLP stosuje wspólną funkcję osobno do każdego. Jednak strzałki uwagi nie są siłami centralnymi ani parami akcji i reakcji. Zwykle \(A_{jk}\ne A_{kj}\), a wiadomość nie wskazuje wzdłuż \(r_k-r_j\).

Ponadto macierz uwagi nie jest pełną macierzą wpływu. Dla perturbacji:

\[
\delta H_j
=\sum_h W_O^h\sum_k
\left[(\delta A_{jk}^h)v_k^h+A_{jk}^h\delta v_k^h\right].
\]

Pokazanie samych \(A_{jk}\) pomija zmianę samych wag uwagi. Do badania wpływu jednego tokenu na drugi należy używać pochodnych albo kontrolowanych perturbacji.

## 3. Dokładny matematyczny odpowiednik reflektora

Niech \(w_c^\top\) będzie wierszem \(W_1\), a \(v_c\) kolumną \(W_2\). Wówczas:

\[
a_{jc}=w_c^\top\operatorname{LN}(x_j)+b_{1c},
\quad z_{jc}=g(a_{jc}),
\quad
F(x_j)=\sum_{c=1}^{768}z_{jc}v_c+b_2.
\]

Każdy reflektor ma więc:

1. wspólną dla tokenów regułę odczytu \(w_c,b_{1c}\);
2. odpowiedź zależną od tokenu \(z_{jc}\);
3. wspólny kierunek i skalę zapisu \(v_c\).

To odpowiada idei odczytu i zapisu w analizach FFN jako pamięci klucz–wartość. Semantyczne wyniki dla modeli językowych nie przenoszą się automatycznie na Twoje obrazy, ale rozkład algebraiczny jest ten sam: [Geva i in., *Transformer Feed-Forward Layers Are Key-Value Memories*](https://doi.org/10.18653/v1/2021.emnlp-main.446).

### Podpisana aktywacja

Dla GELU:

\[
g(a)=a\Phi(a),\qquad g'(a)=\Phi(a)+a\phi(a).
\]

GELU jest ujemne dla ujemnego argumentu. Ma minimum około −0,16997 przy \(a\approx-0,75179\), a część jego pochodnych jest ujemna. Nie wolno zatem utożsamiać \(z\) z nieujemnym natężeniem światła ani zakładać, że zwiększanie pobudzenia zawsze rozjaśnia reflektor.

Dokładna długość wkładu wynosi \(|z_{jc}|\|v_c\|\); znak decyduje o jego zwrocie. W filmie można zastosować dwa oznaczenia zwrotu:

\[
z=z^+-z^-,\quad
z^+=\max(z,0),\quad z^-=\max(-z,0),
\]

\[
zv=z^+v+z^-(-v).
\]

Oba „kanały światła” są wtedy nieujemne, lecz kodują przeciwnie skierowane wektory. Jest to reprezentacja graficzna jednego neuronu, nie dowód istnienia drugiego neuronu w sieci. Bias \(b_2\) wymaga dodatkowej wspólnej strzałki.

W moim końcowym forwardzie **79,04% wszystkich aktywacji 600×768 było ujemnych**, a 61,00% pochodnych GELU było ujemnych. Liczba ujemnych aktywacji nie jest miarą ich udziału w normie sumy. Obcięcie ich zmieniało całe pole \(F\) o 20,65% w relatywnej normie Frobeniusa; po wycentrowaniu pola błąd wynosił 12,65%. Dla konkretnego tokenu i projekcji 3D błąd może być znacznie większy. [Dane kontrolne][NUM]

### Przeciwpary są możliwością, nie koniecznością

Z5 zbyt mocno stwierdza konieczność przeciwlasera. Kontrprzykład:

\[
v_1=(1,1,1),\quad v_2=(1,-1,-1),\quad
v_3=(-1,1,-1),\quad v_4=(-1,-1,1).
\]

Każda para ma cosinus −1/3, a mimo to:

\[
v_2+v_3+v_4=-v_1.
\]

Przeciwne pchnięcie może powstać przez sumę wielu kierunków. Dla prawdziwej przeciwpary \(v_c=\alpha e,\ v_d=-\beta e\) suma wynosi \((\alpha z_c-\beta z_d)e\). Równe aktywacje znoszą się tylko przy równych długościach kierunków.

Statystyka „53% neuronów ma przeciwny kierunek poniżej progu” może być poprawną obserwacją geometrii wag. Nie oznacza 53% rozłącznych par ani dowodu funkcjonalnego sterowania obiema stronami osi.

## 4. LayerNorm zmienia znaczenie „celownika” i geometrii

Niech \(d=192\), \(C=I-\mathbf1\mathbf1^\top/d\), \(c=Cx\), \(s=\sqrt{\|c\|^2/d+\epsilon}\). Wtedy:

\[
\operatorname{LN}(x)=D_\gamma c/s+\beta,
\]

\[
L(x):=D_x\operatorname{LN}(x)
=D_\gamma\left(\frac Cs-\frac{cc^\top}{d s^3}\right).
\]

Rzeczywisty lokalny kierunek czułości neuronu to:

\[
g_c(x)=\nabla_xa_c=L(x)^\top w_c.
\]

Samo \(\gamma\odot w_c\) jest tylko częścią tej konstrukcji. Kierunek czułości zależy również od położenia tokenu, jego średniej i normy. Dwa tokeny mogą mieć różne lokalne „celowniki” tego samego neuronu mimo współdzielonych parametrów.

LayerNorm dokładnie usuwa średnią. Dla \(\epsilon>0\) jej wyjścia przed \(\gamma,\beta\) spełniają:

\[
\mathbf1^\top(c/s)=0,\qquad
\|c/s\|^2=\frac{d\|c\|^2}{\|c\|^2+d\epsilon}<d.
\]

Obraz mieści się we wnętrzu kuli w podprzestrzeni \(\mathbf1^\perp\); przy typowej wariancji dużo większej od epsilon leży blisko jej sferycznego brzegu. Dokładna sfera pojawia się w granicy \(\epsilon\to0\). Wymiar jej powierzchni dla \(d=192\) wynosi 190. Uczone \(\gamma,\beta\) dają następnie transformację afiniczną. Tę geometrię szczegółowo opisuje [Riechers, *Geometry and Dynamics of LayerNorm*](https://doi.org/10.48550/arXiv.2405.04134).

W filmach należy jawnie wybrać współrzędne:

- Gdy punkt jest w \(p=Px\), strzałką jednego MLP jest \(PF(x)\).
- Gdy punkt jest w \(p=P\operatorname{LN}(x)\), strzałką jest
  \[
  P[\operatorname{LN}(x+F(x))-\operatorname{LN}(x)].
  \]
- Gdy punkt przedstawia \(P_z z\), strzałka ruchu podczas uczenia to \(P_z(z_{t+\Delta t}-z_t)\).

Dodawanie surowego \(PF\) do pozycji po nieliniowej LN miesza układy współrzędnych. Użycie \(\gamma\) ma znaczenie także dla odległości: \(\beta\) znika w różnicach, lecz anizotropowe \(\gamma\) nie znika.

## 5. Co naprawdę mówi Jakobian: odkształcenie, krok, objętość i pułapka

Dokładna pochodna pola MLP:

\[
J(x)=D_xF(x)=W_2D_{g'(a)}W_1L(x)
=\sum_c g'(a_c)v_cg_c(x)^\top.
\]

W mojej kontroli wzór na \(F\) zgadzał się z forwardem float32 z błędem względnym poniżej \(4,7\cdot10^{-7}\). Jakobian analityczny w końcowym punkcie tokenu 316 zgadzał się z autograd float64 z błędem \(6,8\cdot10^{-16}\). To potwierdza równanie; nie potwierdza automatycznie każdej interpretacji tego równania. [Kontrola i kod][NUM]

### Cosinus wag nie określa znaku śladu

Dokładnie:

\[
\operatorname{tr}J=\sum_c g'(a_c)v_c^\top g_c(x).
\]

Po podstawieniu \(s_c=D_\gamma w_c\):

\[
\operatorname{tr}J
=\sum_c g'(a_c)
\left[
\frac{s_c^\top Cv_c}{s}
-\frac{(s_c^\top c)(c^\top v_c)}{d s^3}
\right].
\]

Ujemna mediana \(\cos(s_c,v_c)\) nie ustala znaku tej ważonej sumy. Pochodne GELU mogą być ujemne, a centrowanie i składowa radialna zmieniają iloczyn skalarny. Dlatego wynik „cosinus wag spada” i wynik „ślad jest ujemny” należy raportować jako dwie obserwacje, dopóki nie zostaną powiązane dokładnym bilansem powyższych składników.

### Cztery różne kryteria

Rozłóżmy \(J=S+A\), gdzie \(S=(J+J^\top)/2\), \(A=(J-J^\top)/2\).

| Badane pytanie | Właściwa wielkość |
|---|---|
| Chwilowa zmiana długości w pomocniczym przepływie \(\dot\delta=J\delta\) | \(\delta^\top S\delta\) |
| Rozciąganie i ściskanie w jednym rzeczywistym kroku MLP | Wartości singularne \(I+J\) |
| Mnożnik objętości nieskończenie małego otoczenia w jednym kroku | \(|\det(I+J)|\) |
| Asymptotyczna stabilność powtarzanej zamrożonej mapy liniowej | Promień spektralny \(\rho(I+J)\) |

Dla rzeczywistego kroku:

\[
\delta^+=(I+J)\delta+O(\|\delta\|^2),
\]

\[
\|\delta^+\|^2-\|\delta\|^2
=\delta^\top(2S+J^\top J)\delta+O(\|\delta\|^3).
\]

Sama ujemna wartość własna \(S\) nie gwarantuje zatem skrócenia w skończonym kroku. Również \(A\) zachowuje normę w ciągłym przepływie, lecz jawny krok Eulera daje:

\[
\|(I+hA)\delta\|^2
=\|\delta\|^2+h^2\|A\delta\|^2.
\]

To ważne dla filmu symulującego wirowanie małych sond.

### Wynik kontrolny dla Twojego tokenu 316

Wszystkie liczby dotyczą tego samego zdjęcia, wejścia MLP bloku 6 i właściwego \(\epsilon=10^{-6}\).

| Krok | Najmniejsza wartość singularna \(I+J\) | Największa | Odsetek wartości singularnych >1 | \(|\det(I+J)|\) |
|---:|---:|---:|---:|---:|
| 0 | 0,7182 | 1,3066 | 52,08% | 0,9027 |
| 10 940 | 0,5780 | 1,3863 | 48,96% | 0,7549 |
| 65 640 | 0,2759 | 1,8721 | 49,48% | 0,1960 |

W końcowym punkcie mała objętość jest pomniejszana około 5,1 raza, a najbardziej rozciągany kierunek wydłuża się około 1,87 raza. Obie rzeczy zachodzą jednocześnie. U wszystkich 8 sprawdzonych końcowych tokenów odsetek rozciąganych kierunków wynosił około 49–50%.

Nie jest to sprzeczność: moduł wyznacznika jest iloczynem wszystkich wartości singularnych. Silne ściskanie części kierunków może przeważyć objętościowo nad rozszerzaniem pozostałych. To lokalna własność mapy w 192D, nie pomiar gęstnienia całej chmury ani dowód stabilnej pułapki.

Dodatkowo pre-LN wymusza:

\[
F(x+c\mathbf1)=F(x),\qquad J\mathbf1=0,\qquad
(I+J)\mathbf1=\mathbf1.
\]

Istnieje dokładnie neutralny kierunek przesunięcia średniej; ścisła kontrakcja całej przestrzeni jest niemożliwa. Kontrola numeryczna \(J\mathbf1\) dała resztę rzędu \(10^{-16}\).

### Dlaczego „siodło z wirem” jest za mocnym wnioskiem

Aby klasyfikować punkt jako równowagę, trzeba najpierw wykazać \(F(x_*)=0\). Mieszane znaki \(S\) w aktualnym położeniu tokenu tego nie zapewniają. Przykład \(F(x,y)=(1,x)\) nie ma żadnej równowagi, a jego \(S\) ma wartości własne ±1/2.

W Twoim końcowym punkcie 316 norma \(F\) wynosi około 8,60, a norma \(x\) około 52,72. Nie mamy podstaw traktować tego położenia jako znalezionego zera pola. Poprawny opis brzmi: **„lokalne pole z kierunkami chwilowego ściskania i rozciągania oraz częścią antysymetryczną”**.

Jeśli chcesz badać prawdziwe pułapki, potrzebny jest osobny test istnienia zera, jego reszty, stabilności i zakresu przyciągania. Trzeba też usunąć neutralną swobodę LN albo jawnie pracować w przestrzeni ilorazowej. Sześć różnych bloków nie powtarza jednego zamrożonego pola, więc stabilność takiego pomocniczego układu nie jest automatycznie stabilnością pełnej sieci.

### Potencjał i wir

Na obszarze jednospójnym gładkie pole ma potencjał wtedy i tylko wtedy, gdy jego Jakobian jest wszędzie symetryczny. Dla tego modelu należy sprawdzać cały iloczyn \(W_2D_{g'}W_1L(x)\). Samo związanie \(W_2=W_1^\top\) nie wystarcza po dołożeniu LN.

To warunek omówiony dla autoenkoderów w [Im i in., *Conservativeness of Untied Auto-Encoders*](https://doi.org/10.1609/aaai.v30i1.10268), zastosowany tutaj do rzeczywistej złożonej funkcji. Podawane w tej pracy 95–99% dotyczą m.in. iloczynu wag, a nie wyłącznie pełnego Jakobianu; bez ujednolicenia macierzy i normy nie należy porównywać ich wprost z Twoim wskaźnikiem Frobeniusa.

Lokalny rozkład \(S+A\) nie jest globalnym rozkładem Helmholtza nieliniowego pola. W konwencji siły \(F=-K\delta\) sztywność odpowiada \(-S\), nie \(S\).

## 6. Co w obecnych filmach wymaga poprawki

| Miejsce | Stan faktyczny | Wierniejszy zapis |
|---|---|---|
| Filmy 10–12 | Aktywacje s47 połączone z fc2 biegu gęstego; ponadto \(z\) obcięte do nieujemnych | Jeden bieg i checkpoint dla wszystkich składników |
| Film 17, czerwona suma | \(PW_2\max(z,0)\), bez biasu | \(P(W_2z+b_2)\); pełne \(F\) już jest w cache |
| Filmy 17–19, pozycje | \(s_tP(x_j-x_{316})\), przed LN | Podpis zgodny z tą definicją albo jawnie nowa reprezentacja po LN |
| Barwy odległości | Standaryzacja bez uczonego \(\gamma\) | Jawna metryka; dla „następnej LN” użyć rzeczywistego \(\gamma\) |
| Ruch na ekranie | Ruchomy punkt odniesienia, zmienny zoom, obrót kamery | Stały środek, skala i kamera w widoku pomiarowym |
| Prawy panel 18/19 | Symulowane sondy, sztuczny szum, resety, znormalizowane pole 3D | Jawny podpis symulacji albo rzeczywisty pojedynczy krok perturbacji |
| Podpis „e1 cofa” | Oś wybrana z końca; na starcie jej odpowiedź jest dodatnia | „Końcowa oś e1” i bieżąca wartość odpowiedzi |
| Podpis „dokładny Jakobian” | W cache wejścia float16 i epsilon \(10^{-5}\) | Odczyt epsilon z modelu i określenie dokładności danych |

Dwa szczególnie ważne pomiary filmu 17:

- Końcowa czerwona strzałka ma błąd 95,89% względem właściwego \(PF\), a cosinus obu strzałek wynosi 0,2945. Jest to wynik dla tokenu 316 i użytej w filmie projekcji, nie średni błąd wszystkich tokenów.
- Uwzględnienie rzeczywistego końcowego \(\gamma\) zmienia znak testu odległości dla 124/599 tokenów. Mediana stosunku odległości zmienia się z 1,02049 na 1,00796. Test nadal dotyczy przejścia MLP przy zamrożonych wagach.

Pełna suma z cache zgadza się z bieżącym \(W_2z+b_2\) w granicach około 0,02%; błąd czerwonej strzałki nie jest więc skutkiem samego float16. [Dane i definicje testów][VIS]; [kod czerwonej sumy][FILM17].

### Zamknięty układ 3D nie odtwarza pełnej dynamiki

Dla ortonormalnej bazy \(Q\in\mathbb R^{192\times3}\) można pokazać \(J_3=Q^\top JQ\). Jednak to nie wystarcza, by całe trajektorie pozostawały w tej przestrzeni:

\[
\dot y=Q^\top JQ\,y+Q^\top J(I-QQ^\top)\delta,
\]

a perturbacja startująca w \(Q\) generuje składową \((I-QQ^\top)JQ\,y\) poza nim.

Zmierzony stosunek norm

\[
\ell_Q=\frac{\|(I-QQ^\top)JQ\|_F}{\|JQ\|_F}
\]

wynosi około **98,85% na starcie i 34,40% na końcu**. To stosunek norm, nie procent energii. Wymuszenie ciągłego pozostawania sond w 3D usuwa znaczną część odpowiedzi. Taki panel można uczciwie nazwać rzutowanym lokalnym modelem ilustracyjnym, ale nie pełnym zbliżeniem dynamiki 192D.

Stałe \(Q\) i \(P\) są dobrym wyborem do porównywania checkpointów. Problemem jest zmiana środka, skali i znaczenia współrzędnych. Procent energii zachowanej trzeba liczyć dla **faktycznie użytej bazy**, a nie dla najlepszych trzech składowych dopasowanych osobno w każdym czasie.

Sama szklana sfera realizuje w kodzie geometryczne łączenie odcinków; nie ma modelu współczynnika załamania ani prawa Snelliusa. Najtrafniejszy podpis to „schemat programowalnego przekierowania wiązek”. Dosłowna realizacja optyczna wymagałaby osobnego projektu fizycznego.

## 7. Rzeczywista dynamika podczas nauki

Najlepszym opisem uczenia jest dyskretny układ ze stanem rozszerzonym:

\[
S_t=(\theta_t,m_t,v_t,B_t^{BN},k_t),
\qquad
S_{t+1}=\mathcal T(S_t;\xi_t).
\]

\(\theta\) obejmuje wszystkie uczone parametry; \(m,v\) momenty AdamW; \(B^{BN}\) bufory BatchNorm; \(k\) liczniki i harmonogram. \(\xi_t\) oznacza batch, augmentację i inne losowe wymuszenia. Do dokładnego odtworzenia całej sekwencji dochodzą stan generatorów i kolejność próbkowania.

Dla stałego obrazu odczytywanego w eval:

\[
z_{I,j,t}=\mathcal O_j(I;\theta_t,B_t^{BN}),
\]

\[
\Delta z_j=
D_\theta\mathcal O_j\,\Delta\theta+
D_B\mathcal O_j\,\Delta B+
O(\|(\Delta\theta,\Delta B)\|^2).
\]

Cząsteczka nie ma tu dodatkowego integratora ani niezależnej prędkości. Jednak jej obserwowana trajektoria może mieć pamięć i opóźnienia odziedziczone po całym układzie. Kategoryczny zakaz jakiegokolwiek opóźnienia byłby zbyt mocny.

W treningu źródłowym gradient jest najpierw wspólnie przycinany do normy 1. AdamW aktualizuje:

\[
m_{t+1}=\beta_1m_t+(1-\beta_1)\bar g_t,\qquad
v_{t+1}=\beta_2v_t+(1-\beta_2)\bar g_t^{\odot2},
\]

\[
\theta_{t+1}=(I-\eta_t\Lambda)\theta_t
-\eta_t\frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}}+\epsilon_{\rm Adam}}.
\]

Tutaj \(\beta_1=0,9,\ \beta_2=0,999\); weight decay 0,05 dotyczy macierzy, a nie wektorów i biasów. Występuje warmup 300 kroków do lr \(5\cdot10^{-4}\), batch 32 oraz augmentacje. BatchNorm ma dodatkowe bufory o momentum 0,1. [Kod treningu][TRAIN]

**Wniosek dla obrazu:** reflektory powinny być sterowane wspólnym układem uczenia z pamięcią. Tokeny pokazują odczyt tej zmieniającej się sieci. Dodawanie im niezależnego białego szumu w przestrzeni 3D nie odtwarza szumu treningu.

W pierwszym przybliżeniu, dla wspólnego losowego przyrostu stanu o kowariancji \(\Sigma_S\):

\[
\operatorname{Cov}(\Delta z_j,\Delta z_k)
\approx G_j\Sigma_SG_k^\top,
\qquad G_j=D_S\mathcal O_j.
\]

Szum obserwowany przez różne cząsteczki jest więc współdzielony, anizotropowy i zależny od stanu. Nie ma uzasadnienia dla jednego skalaru „temperatura = learning rate”. Modele SDE optymalizatorów adaptacyjnych także wymagają jawnych momentów i założeń skalowania: [Malladi i in., *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*](https://doi.org/10.48550/arXiv.2205.10287).

### Co naprawdę przestawia pojedynczy reflektor

Niech \(\delta_j=\partial L/\partial F_j\). Wówczas dla danego bloku:

\[
\frac{\partial L}{\partial v_c}=\sum_j z_{jc}\delta_j,
\]

\[
\frac{\partial L}{\partial w_c}
=\sum_j g'(a_{jc})(v_c^\top\delta_j)u_j,
\qquad
\frac{\partial L}{\partial b_{1c}}
=\sum_j g'(a_{jc})(v_c^\top\delta_j).
\]

Suma obejmuje tokeny i obrazy w batchu, z uwzględnieniem sposobu agregacji straty w \(\delta_j\). Potem gradient przechodzi przez clipping i AdamW.

To daje wierniejszą animację uczenia: **sygnał błędu z wielu tokenów wspólnie zmienia odczyt i zapis każdego reflektora**. Reflektor nie śledzi mechanicznie jednej cząsteczki. Jeśli wyświetla się gradient, należy odróżnić go od rzeczywistego przyrostu parametru po AdamW.

## 8. Co wynika z pomiarów dryfu i „nadążania”

### Dlaczego „75% dryfu od reflektorów” nie jest udowodnione

W istniejącym rozkładzie dla \(z=G(u,W,b)\) porównano zmianę wejścia i zmianę fc1 względem starego stanu:

\[
D=z_B-z_A,\quad
A=G(u_A,W_B,b_B)-z_A,\quad
B=G(u_B,W_A,b_A)-z_A,
\]

\[
I=D-A-B.
\]

Raportowane 74,58% to \(E(A)/(E(A)+E(B))\), gdzie \(E\) oznacza średnią kwadratu normy po tokenach i obrazach. Dokładny bilans jest inny:

\[
E(D)=E(A)+E(B)+E(I)+2\langle A,B\rangle
+2\langle A,I\rangle+2\langle B,I\rangle.
\]

W istniejących danych między epokami 1 i 60:

| Wielkość | Wartość |
|---|---:|
| Energia obserwowanej zmiany \(E(D)\) | 198,8499 |
| Podmiana fc1 \(E(A)\) | 242,9453 |
| Podmiana wejścia \(E(B)\) | 82,7924 |
| Reszta interakcyjna \(E(I)\) | 247,7178 |
| Suma składników krzyżowych | −374,6056 |

Reszta jest większa od obserwowanej energii. Iloraz dwóch wybranych energii nie jest procentowym wyjaśnieniem całego dryfu. [Oryginalny pomiar][WIAZKI]; [audyt mianowników][MEAS]

Podobnie 84–98% jest dopełnieniem takiego ilorazu dla krótkich przyrostów. Na początku odstęp wynosi 10 kroków, później 100; nie wyodrębniono pasma „szybkich drgań”. Wejście \(u\) obejmuje również norm2 i uwagę ostatniego bloku, więc nie oznacza wyłącznie poprzednich pięciu bloków.

### Dokładniejszy bilans kontrfaktyczny

Dla dwóch grup stanu \(\phi\) i wejścia \(u\) można uśrednić oba porządki podmian:

\[
D_\phi=\frac12[
G(\phi_1,u_0)-G(\phi_0,u_0)
+G(\phi_1,u_1)-G(\phi_0,u_1)],
\]

\[
D_u=\frac12[
G(\phi_0,u_1)-G(\phi_0,u_0)
+G(\phi_1,u_1)-G(\phi_1,u_0)].
\]

Wtedy dokładnie \(\Delta z=D_\phi+D_u\). To dwugrupowy podział Shapleya; nowy pomiar wykorzystuje już podobną korektę.

Nadal normy kwadratowe nie sumują się bez składnika krzyżowego. Jeśli koniecznie potrzebne są addytywne udziały, można raportować:

\[
\alpha_\phi=\frac{\langle\Delta z,D_\phi\rangle}{\|\Delta z\|^2},
\quad
\alpha_u=\frac{\langle\Delta z,D_u\rangle}{\|\Delta z\|^2},
\quad \alpha_\phi+\alpha_u=1.
\]

Wartości mogą być ujemne albo większe od 1. To przypisanie zgodnie z wybraną regułą podziału, nie zidentyfikowane udziały przyczynowe. Przy dalekich checkpointach hybrydowe modele mogą dodatkowo działać poza swoim typowym zakresem.

### Pełny nowy test nie potwierdza jednostronnego podążania

W trakcie audytu został ukończony niezależnie uruchomiony pomiar [NADAZANIE_PULAPKI_S47.json][LAG]. Włączyłem jego pełny wynik: 746 przedziałów do kroku 65 600, odróżniając go od wcześniejszego testu skróconego.

Dla późnej fazy, kroki 43 700–65 600, średni cosinus dla lagu +1 wynosi −0,01288, a dla −1: −0,03083; jedno przesunięcie to 100 kroków. W środkowych fazach sygnał pojawia się w obu kierunkach czasowych. Nie jest to czysty wzór jednostronnego nadążania.

Również zastosowanie obrotu strumienia do czujników w epokach 40–60 daje ujemny wskaźnik dopasowania: −1,330 dla okien jednej epoki i −0,643 dla pięciu epok w bazie 95%. Ujemny wynik oznacza pogorszenie względem pozostawienia czujników bez tej transformacji w użytej definicji błędu. Nie należy nazywać go ujemnym „procentem obrotu”.

Te wyniki nie wykluczają adaptacji neuronów do danych. Ograniczają konkretne hipotezy: jednostronny lag w użytych obserwablach i ten sam pełny obrót strumienia oraz czujników. Nie rozstrzygają skali 3–13 kroków przy próbkowaniu co 100. Potrzebne są także zera zachowujące autokorelację, kalibracja przeszukiwania wielu lagów i kontrola wspólnych przyczyn.

## 9. Najbardziej użyteczne poprawki z literatury

Przejrzałem istniejące wyszukiwania Undermind i zleciłem odczyt pełnych tekstów ośmiu prac. Zachowałem pytania i wyniki odczytu w [zapisie literatury][LIT]. Poniżej rozdzielam wnioski prac od proponowanego zastosowania u Ciebie; prace z 2026 roku są preprintami.

| Praca | Co rzeczywiście wnosi | Zastosowanie i granica przeniesienia |
|---|---|---|
| [Lu i in., *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View* (2019)](https://arxiv.org/abs/1906.02762) | Interpretacja przechodzenia przez warstwy jako dyskretyzacji układu cząstek i rozdzielenia uwagi/FFN | Uzasadnia oś głębokości. Nie zamienia checkpointów treningowych w kroki tej samej ODE |
| [Riechers, *Geometry and Dynamics of LayerNorm* (2024)](https://doi.org/10.48550/arXiv.2405.04134) | Dokładna geometria projekcji, skalowania i transformacji afinicznej LN | Poprawić sferę, wymiar i metrykę; używać rzeczywistych gamma i epsilon |
| [Mudarisov i in., *Feed-Forward Steering in Transformer Residual Dynamics* (2026)](https://arxiv.org/abs/2608.02071) | Rozdział radialnej i stycznej części pola oraz równowagi kierunków | Dobry nowy panel do wizualizacji. Wyniki eksperymentalne dotyczą LLM; wymagają osobnego pomiaru w Twoim ViT |
| [Herty, Liu, *A Mean-Field Theory of Transformers: Well-Posedness of the Coupled Data–Parameter Dynamics and Global Convergence of Training* (2026)](https://arxiv.org/abs/2608.25055) | Dwa sprzężone opisy: ewolucja tokenów po głębokości i parametrów podczas uczenia | Mocne uzasadnienie rozdzielenia dwóch czasów. Twierdzenia graniczne nie dowodzą dynamiki skończonego ViT z 6 blokami, 3 głowami i AdamW |
| [Malladi i in., *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms* (2022)](https://doi.org/10.48550/arXiv.2205.10287) | Stanowe, adaptacyjne modele szumu optymalizatora | Porzucić identyfikację jednego lr z temperaturą; nie przenosić wyników bez uwzględnienia AdamW, clippingu i BN |
| [Liu, Li, *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* (2026)](https://arxiv.org/abs/2608.19762) | Wpływ perturbacji gradientu przez pamięć momentów i propagację rozszerzonego stanu | Zastąpić sam lead–lag pomiarem odpowiedzi kontrolowanej. Odczyt reprezentacji oraz dodanie BN i clippingu są proponowanym tu rozszerzeniem |
| [Antognini, Sohl-Dickstein, *PCA of high dimensional random walks with comparison to neural network training* (2018)](https://arxiv.org/abs/1806.08805) | Gładkie, niskowymiarowe obrazy PCA mogą powstawać z błądzenia losowego, pędu i OU | Przepuścić dopasowane procesy kontrolne przez identyczny pipeline filmu; nie wnioskować o potencjale z kształtu 3D |
| [Im i in., *Conservativeness of Untied Auto-Encoders* (2015/2016)](https://doi.org/10.1609/aaai.v30i1.10268) | Warunek symetryczności pełnego Jakobianu pola | Testować rzeczywiste \(J\), nie sam związek wag; porównania procentów dopiero po ujednoliceniu miary |

### Użyteczny nowy rozkład: zmiana długości i zmiana kierunku

Dla \(x=ru\), \(r=\|x\|>0\), \(u=x/r\), w pomocniczym przepływie:

\[
\dot r=u^\top F,\qquad
\dot u=\frac{(I-uu^\top)F}{r}.
\]

W Twoim skończonym kroku można policzyć dokładnie:

\[
r^+=\|x+F(x)\|,\qquad
u^+=\frac{x+F(x)}{\|x+F(x)\|}.
\]

Pokazywałbym osobno przyrost normy i kąt między \(u,u^+\). To czytelniejsze od jednego pojęcia „siły pułapki”. Dla obrazu po LN należy analogicznie zastosować tę analizę do wybranego centrowania i jasno określić wpływ gamma.

Równowaga kierunku \( (I-uu^\top)F=0 \) oznacza radialność pola, nie \(F=0\). Nie wolno mieszać tych dwóch typów równowag. Praca Mudarisova i in. zawiera przybliżone kandydatury kierunkowe; nie stanowią one dowodu znalezienia dokładnych pułapek w Twojej sieci.

### Uwaga i MLP działają sekwencyjnie

Dla pełnego stanu wszystkich tokenów:

\[
T(X)=X+H(X)+F(X+H(X)),
\]

\[
D T(X)=[I+D F(X+H(X))][I+D H(X)].
\]

\(DF\) jest blokowo diagonalny po tokenach, lecz \(DH\) je sprzęga. MLP mierzone w ostatnim bloku nie wystarcza do opisania całej propagacji.

Można zmierzyć defekt \(F(X+H(X))-F(X)\), aby ocenić, jak bardzo uwaga zmienia późniejszą odpowiedź MLP. Przy sztucznym małym kroku \(\tau\) różnica między sekwencyjną a równoległą aktualizacją zaczyna się od \(\tau^2 DF\,H\). Dopiero różnica obu kolejności wiąże się z komutatorem \(DF\,H-DH\,F\). Te dwa defekty nie są zamienne.

### Jak sprawdzić wpływ uczenia przyczynowo

Lokalna perturbacja gradientu \(\xi\) w kroku \(t\) zapisuje się w parametrach i momentach:

\[
\delta S_{t+1}\approx B_t\xi,\qquad
\delta z_{t+h}\approx
G_{t+h}A_{t+h-1}\cdots A_{t+1}B_t\xi,
\]

gdzie \(A_s=D_S\mathcal T_s\), a \(G=D_S\mathcal O\). To adaptacja podejścia input–state–output do Twojej obserwabli.

Krótka para kontrolowanych biegów powinna startować z tego samego pełnego stanu i mieć te same następne batche oraz augmentacje. Należy jasno zdefiniować perturbację, los momentów i buforów BN. Clipping należy uwzględnić w pochodnej przejścia; na jego granicy potrzebna jest ostrożność.

Same pliki wag z kluczami net/krok/epoka nie zawierają momentów AdamW i stanu próbkowania. Pozwalają odtworzyć reprezentację, lecz nie wystarczają do wiernego odtworzenia kroku optymalizatora. To propozycja przyszłego krótkiego eksperymentu, nie wykonany w tym audycie test.

## 10. Jak zbudowałbym wierniejszy układ

### Widok A: rzeczywiste uczenie

Położenie cząsteczki:

\[
p_j(t)=P_z[z_j(t)-z_{\rm ref}],
\]

gdzie \(P_z\), \(z_{\rm ref}\), skala i kamera są stałe w wybranym porównaniu. Można obok pokazać residual stream z osobnym \(P_x\). Strzałki oznaczają rzeczywistą zmianę między checkpointami; podpis podaje odstęp kroków i ewentualną interpolację.

Dodatkowy widok współporuszający się z tokenem 316 jest użyteczny, o ile jest oznaczony jako względny. Nie powinien zastępować widoku pozwalającego zobaczyć ruch tego tokenu.

### Widok B: jedno przejście przez sieć

Zamrożony checkpoint. Pokazujemy kolejno uwagę i MLP w sześciu blokach. Dla MLP pozycja zmienia się przez pełną sumę podpisanych wkładów i biasu. Oś czasu oznacza głębokość, a nie krok treningu.

Reflektory mogą pozostać na sferze, ale ich fizyczne ustawienie jest częścią interfejsu. Wkłady liczymy w pełnej przestrzeni i dopiero potem rzutujemy. Wspólna suma wyświetlanych strzałek musi odtwarzać wyświetlany pełny przyrost przed ozdobnym skalowaniem jasności.

### Widok C: lokalna odpowiedź

Pokazujemy małą kulę perturbacji oraz jej obraz przez \(I+J\): elipsoidę ściskania i rozciągania. Podajemy największą i najmniejszą wartość singularną, logarytm mnożnika objętości oraz stratę rzutu.

Dla ograniczonej liczby sond można bezpośrednio policzyć \(T(x+\delta)-T(x)\) w pełnym 192D i dopiero wyświetlić 3D. W ten sposób unikamy założenia, że rzutowana przestrzeń zamyka dynamikę. Dobór wielkości perturbacji należy sprawdzić przez błąd linearyzacji.

Jeśli zostaje animacja ciągłego pola, powinna mieć osobny zegar, jawny model szumu i podpis „symulacja lokalna”. Dla czystej części antysymetrycznej można użyć \(\exp(hA)\), która zachowuje normę; nie należy traktować tego jako rzeczywistego kroku residual \(I+J\).

## 11. Kolejność poprawek i testów

1. **Zgodność danych i sumy.** Jeden manifest biegu, checkpointu, obrazu, tokenu, dtype i miejsca hooka; zachowanie signed GELU; bias; wykorzystanie pełnego F w filmie 17. To największy bezpośredni zysk.
2. **Zgodność współrzędnych i podpisów.** Jawne x192/z768/F192, właściwa LN, rzeczywiste gamma i epsilon, stała skala pomiarowa. Dopisać, że barwy opisują zmianę odległości, a nie dowód wzajemnej siły.
3. **Zgodność kryteriów dynamiki.** Dodać singular values \(I+J\), objętość i kontrolę skończonej perturbacji. Zastąpić twierdzenie o siodle opisem zmierzonej lokalnej deformacji.
4. **Bilans uczenia.** Oddzielić parametry i BN, uporządkować podział grup, stosować dokładny bilans wektorowy z interakcjami. Pokazać rzeczywisty przyrost parametru obok gradientu.
5. **Kontrole statystyczne.** Dopasowane błądzenia losowe/OU/pęd przez ten sam renderer; różne obrazy i fazy; bootstrap na poziomie obrazów/filmów i bloków czasu zamiast uznawania 600 tokenów za niezależne próby.
6. **Dopiero potem silniejsze hipotezy.** Rzeczywiste równowagi, metastabilność, odpowiedzi impulsowe i ewentualny efektywny opis stochastyczny. Do opóźnień kilku kroków potrzebny jest zapis o porównywalnej rozdzielczości.

Nie proponuję zmieniania architektury tylko po to, by pasowała do pułapki optycznej. Najpierw warto dopasować opis i pomiar do sieci, którą rzeczywiście wytrenowałeś.

## 12. Proponowane brzmienie trzech najważniejszych zasad

**Z5:** „Przeciwstawne kierunki kolumn fc2 umożliwiają prostą kontrolę obu zwrotów osi. Nie są konieczne: przeciwny przyrost może powstać z wielu innych składników, ujemnych aktywacji i biasu. Aktywność pary należy ważyć długościami obu kolumn.”

**Z6:** „Zmiana cosinusów pomiędzy odczytem i zapisem opisuje geometrię wag. Lokalną dywergencję określa dokładny ślad Jakobianu z pochodną GELU i LayerNorm. Ujemna dywergencja nie gwarantuje ściskania wszystkich kierunków ani zmniejszenia objętości skończonego kroku.”

**Z15:** „W badanych stanach MLP ma lokalną odpowiedź zawierającą kierunki chwilowego ściskania, rozciągania i część antysymetryczną. Nie wyznaczono punktów równowagi. Rzeczywisty krok residual oceniamy przez wartości singularne i wyznacznik I+J; trajektoria podczas uczenia jest osobną obserwacją zmieniających się parametrów oraz buforów.”

Pełny przegląd wszystkich Z1–Z15, dodatkowe kontrprzykłady i miejsca w plikach znajdują się w [aneksie matematycznym][MATH].

## Pliki audytu i odtwarzalność

- [Kontrole równań, GELU, singular values, wyznacznika i rzutu][NUM] — wraz z SHA256 modelu, wag i wybranych źródeł.
- [Skrypt kontroli numerycznych][NUMCODE] — trzy forwardy CPU; brak treningu.
- [Kontrole renderowanej strzałki, metryki i pola 3D][VIS].
- [Skrypt kontroli filmów][VISCODE] — odczyt istniejących cache i checkpointów.
- [Audyt istniejących pomiarów][MEAS] oraz [utrwalone wyniki i hashe źródeł][MEASJSON].
- [Aneks matematyczny][MATH].
- [Snapshot analizowanych zasad][SNAP].
- [Zapis odczytów literatury][LIT].

Błędy względne macierzy i całych pól w kontrolach są liczone normą Frobeniusa; wektory normą euklidesową. Wartości singularne i norma operatorowa są oznaczone osobno. Procenty ze stosunków norm nie są automatycznie udziałami energii.

Źródła w folderze były równolegle aktualizowane w trakcie analizy, w szczególności ukończono pełny pomiar nadążania. Snapshot i hashe określają, do której wersji odnosi się audyt. Nowe pliki zapisano oddzielnie; nie uruchamiano ponownie zakończonego treningu ani generowania filmów.

[MODEL]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py:38
[TRAIN]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/trening.py:26
[SNAP]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/ZASADY_LASER_KOLOID_snapshot.md
[NUM]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/numerical_spotchecks.json
[NUMCODE]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/audyt_reflektory_20261012/audit_spotchecks.py
[VIS]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/visual_spotchecks.json
[VISCODE]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/audyt_reflektory_20261012/visual_spotchecks.py
[MATH]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/math_audit_report.md
[MEAS]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/measurement_audit_report.md
[MEASJSON]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/measurement_audit_existing_results.json
[LIT]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_reflektory_20261012/literature_evidence.json
[FILM17]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/film_lasery_17_pulapka_ognisko.py:281
[WIAZKI]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/eden/wyniki/wiazki_koloidy/WIAZKI.json
[LAG]: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/NADAZANIE_PULAPKI_S47.json

