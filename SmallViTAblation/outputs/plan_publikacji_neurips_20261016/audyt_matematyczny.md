# Rygor matematyczny i falsyfikowalny plan: geometria transportu w SmallViT

Data: 2026-10-16. Notatka planistyczna na podstawie odczytu kodu i istniejących raportów. Nie wykonano nowego przejścia modelu, treningu ani obliczeń checkpointów; nie zmieniono aplikacji. Poniższe protokoły są propozycją przyszłych pomiarów, a nie wynikami. Nie przyjmuję szczególnych wymagań żadnej konferencji.

## 1. Co rzeczywiście jest dostępne i co wymaga nowego pomiaru

Nie ma katalogu `outputs/pomiary_pulapki_fizyka_20261014`. Istnieje `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md`, jawnie oznaczony „PLAN, nic nie uruchomione”. Nie wolno przedstawiać jego E1–E8 jako wykonanych eksperymentów.

Sprawdzone źródła lokalne:

- `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py` i `trening.py`: rzeczywista architektura oraz AdamW, clipping i odczyty sond.
- `app/pole_podkrokow.py`: próbne pola 3D, w tym uwaga przy zamrożonych K i V.
- `outputs/porownanie_szklo_20261010/film_krysztal/pulapka_jakobian.py` oraz `PULAPKA_JAKOBIAN.json`: starsza rekonstrukcja wejścia z aktywacji GELU, przyjęcie skali σ=1 i idealizowanego rzutu na 190 wymiarów. To nie jest dokładny Jacobian rzeczywistego residualu.
- `outputs/porownanie_szklo_20261010/film_krysztal/temperatura_pulapki/temperatura_pulapki_s47.py`: pełniejszy wzór Jacobianu, lecz rekonstrukcja stosuje EPS=1e−6; część stanów pochodzi z cache float16. Audyt z 15 października potwierdził w lokalnym oryginalnym modelu eps=1e−5. State dict nie przechowuje epsilon, więc środowisko treningowe należy udokumentować osobno.
- `outputs/research_fizyczny_mikroskop_20261015/math_audit.md` oraz `SYNTEZA_I_PROPOZYCJA_PUBLIKACJI.md`: wcześniejsze rozstrzygnięcia i źródła.
- `outputs/dolina_odpowiedzi_t226_20261015/OPIS.md` oraz kod w `experiments/dolina_odpowiedzi_t226_20261015/`: lokalna poziomica zamrożonej głowy, odróżniona poprawnie od potencjału MLP. Wymiar 187 wynika lokalnie z rzędu pięć, nie oznacza globalnej płaszczyzny ani osiągalności przez sieć.

Najmocniejsza teza do sprawdzenia: **uczenie organizuje lokalne kierunki deformacji i obracania perturbacji w sposób przewidujący odpowiedź zadaniową; ukierunkowane interwencje zmieniają tę odpowiedź bardziej niż dobrane kontrole**. Samo istnienie niesymetrycznego Jacobianu, ujemnych wartości własnych S albo przeciwstawnych wektorów nie jest wystarczającym odkryciem.

## 2. Obiekt i zegar: definicja przed interpretacją

Przy zamrożonym checkpointcie θ i trybie eval: N=600 tokenów, d=192 współrzędne residualu, m=768 aktywacji ukrytych MLP. Pełny stan X należy do R^(N×d). Wartości z po GELU są odrębną reprezentacją 768D; aktualizacja dodawana do x jest 192D.

Dwanaście podkroków głębokości to różne mapy

\[
X_{k+1}=\Psi_{k,\theta}(X_k)=X_k+F_{k,\theta}(X_k),\qquad k=0,\ldots,11.
\]

Nie zakłada się ani autonomicznego ODE, ani wielokrotnego stosowania tego samego bloku. Dla MLP pola tokenów są rozdzielone; dla attention sprzężone. Funkcja pola w danym miejscu pozostaje określona także poza stanami naturalnie wytwarzanymi przez sieć, ale nie oznacza to, że jej całe otoczenie jest osiągalne z obrazów.

Zegar uczenia jest inny: dla stałego obrazu I, x_t=f(I;θ_t,b_t), gdzie b_t obejmuje bufory BN. Pierwszy rząd to Δx=D_θf Δθ+D_bf Δb. Stan mechanizmu uczenia obejmuje ponadto momenty AdamW, harmonogram i strumień danych. J pola głębokości nie jest Jacobianem tego procesu.

W planie z 14 października należy osłabić określenie „dokładny dryf”: krok AdamW na średnim gradiencie nie jest ogólnie średnią kroków AdamW, a f(θ+EΔθ) nie jest E f(θ+Δθ). Warunkowy dryf reprezentacji definiuje się jako oczekiwanie przyrostu po losowaniu paczki przy tym samym pełnym stanie optymalizatora; skończona liczba replik daje estymator, nie dokładne oczekiwanie. Korelacja wiru głębokości i asymetrii ruchu między checkpointami nie dowodzi związku przyczynowego.

## 3. Minimalne twierdzenia algebraiczne i ich granice

### T1. Dokładne wkłady neuronów i Jacobian

\[
F(x)=W_2\phi(W_1L(x)+b_1)+b_2=\sum_c u_c(x)+b_2,
\quad u_c=\phi(a_c)v_c,
\]
\[
J=\sum_c\phi'(a_c)v_cw_c^\top DL(x).
\]

Dowód: liniowość drugiej warstwy i reguła łańcuchowa. Bias b₂ zmienia punkt zerowy pola, choć nie występuje w J. Nie wolno zastąpić podpisanych aktywacji ich wartościami bezwzględnymi. Kwadrat normy sumy zawiera dokładne wyrazy krzyżowe; normy pojedynczych składników nie sumują się do „energii” pola.

Przy C=I−11ᵀ/d, r=Cx, s²=rᵀr/d+ε i Γ=diag(γ):

\[
DL=\Gamma\left(C/s-rr^\top/(ds^3)\right),\quad DL\,1=0,
\quad DL\,r=\Gamma r\,\epsilon/s^3.
\]

Radialny kierunek jest tylko przybliżenie neutralny; nie należy usuwać go jak dokładnego gauge. Kryterium poprawności przyszłego pomiaru: zgodność F i JVP z oryginalnymi modułami i ich epsilon; kontrola różnicą centralną na kilku malejących amplitudach, z wykazaniem zakresu ponad błędem precyzji.

### T2. Dokładny iloraz symetrii pre-LN

Dla przesunięć g=(c₁1,…,c_N1), niezależnych dla tokenów:

\[
F_k(X+g)=F_k(X),\qquad \Psi_k(X+g)=\Psi_k(X)+g.
\]

Dowód: tokenowa LN usuwa każde c_j; attention otrzymuje identyczne Q,K,V; residual przenosi g do następnego podkroku. Końcowa LN usuwa g także przed odczytem głowy. Zatem J_k g=0, (I+J_k)g=g, a pochodna końcowego wyjścia anihiluje g. To do 600 dokładnych kierunków neutralnych, nie tylko jeden globalny.

Wniosek: nie ma pełnowymiarowej ścisłej kontrakcji ani izolowanego asymptotycznie przyciągającego punktu stałego całej mapy pre-LN. Nie wyklucza to kontrakcji użytecznych kierunków lub przyciągania w ilorazie.

Niech U ma ortonormalne kolumny rozpinające 1⊥, a Q=UUᵀ=C. Dokładny stan ilorazowy pojedynczego tokenu ma 191 wymiarów. Pole ilorazowe to UᵀF(Uq), a jego Jacobian J̄=UᵀJU. Dla pełnego obrazu używa się U_N=I_N⊗U. **Centrowanie współrzędnych wewnątrz tokenu usuwa gauge; odejmowanie średniej po tokenach jest inną operacją.**

### T3. S i A opisują lokalną deformację, nie automatycznie siodło i wir

W ustalonej metryce euklidesowej:

\[
S=(J+J^\top)/2,\quad A=(J-J^\top)/2.
\]

Dla hipotetycznej liniowej dynamiki ciągłej dδ/dt=Jδ zachodzi d||δ||²/dt=2δᵀSδ. Dla małej dodatnio zorientowanej pętli w płaszczyźnie ortonormalnych e,f:

\[
\oint F\cdot dx=\text{pole pętli}\,(f^\top Je-e^\top Jf)+o(\text{pola}).
\]

To lokalna niekonserwatywność w ustalonej metryce, w jednostkach przyjętych współrzędnych, bez uzasadnienia dla dżuli. A≠0 nie dowodzi orbit zamkniętych, oscylacji tokenu, ani stacjonarnego prądu prawdopodobieństwa.

S nieokreślone oznacza kierunki lokalnego ściskania i rozciągania w tym kryterium. **Siodło dynamiczne wymaga punktu równowagi i właściwego widma dynamiki.** Kontrprzykład: J=[[-1,k],[0,-1]] przy k>2 ma nieokreślone S, lecz I+J jest nilpotentne i hipotetyczna iteracja liniowa jest asymptotycznie stabilna. Silny wzrost przejściowy współistnieje z tą stabilnością.

Dokładny rozkład ról neuronów dla jednostkowych e,f:

\[
\kappa_c(e)=-\phi'(a_c)(e^\top v_c)(w_c^\top DL e),
\]
\[
\Gamma_c(e,f)=\phi'(a_c)\left[(f^\top v_c)(w_c^\top DL e)-(e^\top v_c)(w_c^\top DL f)\right].
\]

Sumy dają −eᵀSe oraz fᵀJe−eᵀJf. W ilorazie podstawia się kierunki Ue,Uf. To lepsza podstawa do wyboru interwencji niż sama para przeciwległych v_c.

### T4. Rzeczywisty residual wymaga I+J i iloczynów

\[
M=I+J,\qquad M^\top M-I=2S+J^\top J.
\]

Dla małej perturbacji prawdziwej nieliniowej mapy:

\[
\|\Psi(x+\delta)-\Psi(x)\|^2-\|\delta\|^2
=\delta^\top(2S+J^\top J)\delta+o(\|\delta\|^2).
\]

Ujemne eᵀSe nie wystarcza do kontrakcji kroku. Należy mierzyć wartości singularne M, kierunkowe wzmocnienie i, gdy potrzebne, log|det M|. Ślad J jest dywergencją pola, nie dokładną zmianą objętości residualu. ρ(M)<1 daje stabilność liniowej iteracji i wystarczający warunek lokalnej asymptotycznej stabilności nieliniowej przy punkcie stałym; przypadek ρ=1 wymaga osobnej analizy. Bez punktu stałego jest to wyłącznie diagnoza lokalnego operatora.

Transport do dalszej warstwy opisuje Φ=M_{b−1}…M_a, oceniany na rzeczywistej trajektorii. Nie wolno zastąpić go iloczynem stałych rzutów 3D: PΦPᵀ zwykle różni się od iloczynu PM_kPᵀ. Dla ortonormalnego E zawierającego r kierunków pełna zmiana r-wymiarowej objętości wynosi sqrt(det(EᵀΦᵀΦE)); rzutowana objętość jest innym obserwablem. Ekspansja w trzech kierunkach jest zgodna z kontrakcją objętości w pełnych 192 wymiarach.

### T5. Pełna uwaga i kontekst

\[
F_j(X)=\sum_{h,k}A_{jk}^{h}(X)W_O^h v_k^h(X)+b_O,
\quad A_{jk}^{h}=\operatorname{softmax}_k(q_j^\top k_k/\sqrt{d_h}).
\]

Q,K,V obejmują biasy i rzeczywistą LN. Pełne DF działa na 600×192 współrzędnych, zawiera pochodne przez queries, keys i values oraz wpływ na inne tokeny. Sam rozmiar macierzy attention 600×600 nie jest rozmiarem Jacobianu stanu.

Pole próbne w `app/pole_podkrokow.py` zmienia tylko query wybranego słuchacza przy zamrożonych K,V i średniej odniesienia. Jego wartość w centrum odtwarza wyświetlany skok, lecz pochodna jest warunkowa. Nie zastępuje pełnego DF ani nawet pełnej własnej pochodnej j→j, ponieważ zmiana własnego key/value też ma znaczenie. Nowy protokół powinien liczyć wszystkie kanały JVP pełnego podkroku, a pomiar query-only zachować jako jawnie nazwaną ablację.

### T6. Współrzędne, równowagi i poziomice głowy

Przy stałej liniowej zmianie x′=Tx: J′=TJT⁻¹. Widmo jest niezmiennicze, ale euklidesowe S,A, normy i kąty nie są ogólnie niezmiennicze. Przy wybranej stałej metryce G>0 odpowiedni adjoint to J†=G⁻¹JᵀG, a po zmianie współrzędnych G′=T⁻ᵀGT⁻¹. To relacja współrzędnych, nie twierdzenie, że każda T jest symetrią architektury z GELU i LN. Metrykę trzeba zamrozić przed oceną; PCA lub whitening dopasowywane osobno do każdego checkpointu mogą zmienić wynik. Rozkłady kanałowe należy agregować w sposób odporny na permutację neuronów albo jawnie dopasowywać neurony między niezależnymi biegami.

Prawdziwy punkt stały pojedynczego residualu wymaga F(x*)=0. Zero pola wycentrowanego względem chmury oznacza ruch razem z odniesieniem, a zero rzutu nie oznacza zera pełnego pola. Równowaga ilorazowa wymaga UᵀF(x*)=0 i może zawierać niewidoczny dla głowy dryf gauge. Nie wolno extrapolowanego z lokalnej prostej zera uznać za znalezioną równowagę bez reszty pełnej funkcji i kontroli otoczenia.

Dla końcowej głowy H:R^192→R^5, rank DH=5 daje lokalną poziomicę wymiaru 187 i styczną ker DH. Po usunięciu dokładnego gauge odpowiada jej 186 wymiarów w ilorazie. Nie utożsamiamy jej z atraktorem. Macierz BᵀRB, B=DH, jest dodatnio półokreśloną czułością zadaniową, nie dodatnio określoną metryką całego latentu. Dla wcześniejszego stanu używa się BΦ i pełnego kontekstu, nie samej końcowej głowy przyłożonej do wcześniejszego tokenu.

## 4. Minimalny program przyczynowy

### H1. Organizacja wyuczona, a nie uniwersalna geometria losowej macierzy

Hipoteza: ukierunkowanie Γ i κ względem kierunków istotnych dla końcowego wyjścia zmienia się podczas uczenia ponad to, co wyjaśniają normy i anizotropia wag/aktywacji. Miary: rozkład J̄, 2S̄+J̄ᵀJ̄, Γ, κ, wzmocnienie ΦE oraz czułość BΦE na poprawny logit i ramkę. Surowa wartość ||A||/||S|| jest opisowa.

Kontrole: inicjalizacja tej samej architektury, istniejące niezależne seedy, permutacja przypisań neuron–aktywacja lub dopasowane losowanie kierunków z zachowaniem norm i jawnie określonej anizotropii. Takie permutacje są modelami zerowymi, nie muszą reprezentować poprawnie wytrenowanych sieci. Wybór tokenów, warstw i kierunków na części odkrywczej; ocena przewidywań na innych obrazach/sekwencjach. Jednostką niepewności jest co najmniej obraz/sekwencja i seed, nie 600 skorelowanych tokenów traktowanych jako niezależne próby.

Falsyfikuje: efekty znikające po dopasowaniu kontroli albo brak przenoszenia przewidywań na wydzielone dane. Wynik dodatni opisuje organizację, jeszcze nie jej konieczność.

### H2. Lokalna część obrotowa ma przewidywalny wpływ na odpowiedź perturbacyjną

W punkcie bazowym x₀ obliczyć J̄₀, Ā₀. Dla gładkiego okna χ, r>0 i siły α:

\[
F_\alpha(x)=F(x)-\alpha\,\chi(\|U^\top(x-x_0)\|/r)\,U\bar A_0U^\top(x-x_0),\quad\chi(0)=1.
\]

Dokładnie F_α(x₀)=F(x₀), a pochodna ilorazowa w centrum to S̄₀+(1−α)Ā₀. Symetria LN pozostaje zachowana. **Usunięcie surowego A bez ilorazu byłoby błędem:** J1=0 nie implikuje A1=0; zwykle A1=−Jᵀ1/2. Taka operacja mogłaby wytworzyć sztuczną czułość na gauge.

Testować α=0 i wartości po obu stronach, kilka małych amplitud perturbacji ±δ, oraz kontrfaktyczne odpowiedzi wszystkich dalszych bloków. Przewidywana pierwszorzędowa różnica wyjścia to −α D R(Ψ(x₀)) U Ā₀Uᵀδ, gdzie R oznacza rzeczywistą resztę sieci. W nieliniowym teście sprawdzić resztę oraz zależność od amplitudy. Niezmieniona odpowiedź przy δ=0 jest wymagana przez konstrukcję; ten test nie może wykazać spadku zwykłej accuracy na nieporuszonym punkcie.

Kontrole: operator losowy antysymetryczny w tym samym ilorazie, dopasowany rangą i normą Frobeniusa; operator symetryczny o dopasowanej sile; losowanie kierunków w ortogonalnych podprzestrzeniach; zmiana znaku/dawki. Raportować także wynikowe normy korekty i wzmocnienie residualu, ponieważ A wpływa na JᵀJ. Sam sham o równej normie nie dowodzi specyficzności „wirowej” wobec wszystkich zmian czułości.

Interpretacja dodatnia: **przyczynowy wpływ lokalnego operatora obrotowego na odpowiedź badanego obliczenia w określonym otoczeniu**. To interwencja na aktywacjach/operatorze, nie realizacja globalnego usunięcia curl za pomocą wag i nie dowód istnienia wiru stochastycznego.

### H3. Mieszana geometria deformacji wspiera określone funkcje

Analogicznie usuwać wybrane dodatnie lub ujemne mody S̄, zachowując F(x₀), i mierzyć odpowiedź końcową. Dobierać kierunki według skończonego residualu M̄ᵀM̄−I, gdy hipoteza dotyczy kontrakcji kroku. Nie nazywać modyfikacji dodatnich modów S „usunięciem niestabilności” bez tego sprawdzenia.

Test funkcjonalny powinien rozdzielać perturbacje zmieniające cel od zakłóceń, na które detektor ma być odporny. Przy kontrolowanej translacji obrazu poprawna ramka powinna zmienić się zgodnie z translacją; przy zakłóceniu niezmieniającym GT pożądana jest odporność. Miary: błąd zgodności przewidzianej i zmierzonej zmiany, błąd ramki względem właściwego GT, logit/score i końcowa miara detekcji. Kwadrat zmiany bbox nie jest sam z siebie błędem: zmiana może naprawiać wcześniejszy błąd. Ranking/top-k jest niesmooth, dlatego oddzielić od lokalnego testu stałego tokenu.

### H4. Lokalna mapa ma znaczenie dla stanów osiągalnych z danych

Porównać dwa jawne protokoły: (a) perturbacje pojedynczego tokenu w otaczającej przestrzeni, (b) pełne wspólne zmiany wszystkich tokenów wywołane małymi kontrolowanymi zmianami wejściowego obrazu. Dla (b) kierunki E można otrzymać z pochodnej obraz→stan albo z symetrycznych par zmienionych obrazów. To styczne do zadanej rodziny interwencji wejściowych, nie dowód zidentyfikowania całej „rozmaitości danych”. PCA dostarcza podprzestrzeni wariancji, nie dowodu osiągalności.

Wyznaczyć zakres amplitud, gdzie błąd linearyzacji i przewidywania końcowego pozostaje mały; większe perturbacje raportować jako odrębne testy nieliniowe. Ukierunkowane maskowanie neuronów według sum Γ/κ może następnie sprawdzić rolę rzeczywistych składników modelu, z kontrolą liczby neuronów, norm, aktywacji i dawki. Maskowanie zmienia jednocześnie F, S i A; samo nie izoluje roli obrotu. Potrzebne są oba rodzaje interwencji, jeśli celem jest powiązanie lokalnego mechanizmu z rzeczywistymi neuronami.

## 5. Minimalny koszt i kolejność przyszłej pracy

1. **Tożsamość pomiaru:** mały audyt oryginalnego forwardu i JVP z rzeczywistym epsilon, biasami i eval BN. Stanów residualu nie odtwarzać przez wieloznaczną odwrotność GELU. Nowy pomiar pełnej precyzji może być konieczny dla wiarygodności, ale nie został tutaj uruchomiony.
2. **MLP:** lokalny J ma 192² wpisów, około 147 kB w float32 na token; można liczyć analitycznie z kosztem O(md²+d³), lub zastosować JVP. Cały obraz to około 88,5 MB macierzy na jeden MLP, więc przetwarzać partiami i zapisywać miary, nie wszystkie Jacobiany wszystkich checkpointów. Pełne widmo wymaga dodatkowo kosztu O(d³) na wybrany token.
3. **Attention i dalszy transport:** pełny J miałby 115200² wpisów, około 53 GB samego float32, bez grafu i gradientów. Nie materializować. Propagować r wektorów JVP przez rzeczywiste podkroki; dla r=16 same wektory zajmują około 7,4 MB. Koszt jest r-krotnością odpowiednich operacji różniczkowania, z narzutem zależnym od backendu; nie podaję niezmierzonego czasu GPU. Standardowe attention ma koszt O(N²d+Nd²), a MLP O(Ndm).
4. **Odczyt zadaniowy:** dla pięciu wyjść ustalonego końcowego tokenu wystarczy pięć VJP przez pozostałą sieć albo r JVP dla wybranych kierunków. Nie budować pełnego Gramianu. Dla pojedynczego końcowego tokenu rank DH≤5; dla wcześniejszego tokenu oddziałującego na wiele końcowych wyjść rząd może być większy.
5. **Interwencje:** mała wydzielona próbka punktów i warstw, naturalne pary wejść, dawki oraz shamy; koszt rośnie z liczbą par, dawek i przebiegów przez dalszą część sieci. Dopiero po wykazaniu efektu i zakresu liniowości rozszerzać na obrazy/seedy. Zadania checkpointowe/OU pozostawić osobnym badaniem, jeżeli nie są potrzebne do głównej tezy.

Minimalny pakiet dowodowy obejmuje: zgodność algebra–autodiff, test gauge, kontrolę reszty linearyzacji, rzeczywisty iloczyn pełnych pochodnych, przewidywanie odpowiedzi poza próbką wyboru oraz interwencję z kontrolą dawki i sham. Żaden z tych punktów nie wymaga tezy, że ViT realizuje Maxwella, ma temperaturę termodynamiczną albo dąży do jednej globalnej pułapki.
