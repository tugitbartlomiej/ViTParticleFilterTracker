# Wzory z fizyki pułapek optycznych dla mikroskopu SmallViT

Data: 15 października 2026. Nowy research na prośbę użytkownika. Zakres: literatura, sprawdzenie wzorów i propozycja pomiarów. Nie implementowano nowych pomiarów w aplikacji i nie uruchamiano treningu.

**Wniosek:** najbardziej użyteczne są aktywne pomiary odpowiedzi, tensor lokalnego pola, cyrkulacja po pętli i kierunkowe sprzężenia między tokenami. Można je stosować do znanej funkcji sieci bez zakładania termicznego koloidu. Metody analizy drgań są kolejnym poziomem: wymagają poprawnego zegara, kontroli pamięci i próbkowania. Termodynamiczne temperatury i energie potrzebują dodatkowo zidentyfikowanego modelu dynamiki.

**Najsilniejsze pytanie badawcze:** czy uczenie organizuje pola MLP i uwagi tak, aby ograniczać zakłócenia odpowiedzi o narzędziu i jego ramce, mimo dalszego ruchu reprezentacji w innych kierunkach? Uzupełnienie: które neurony i tokeny odpowiadają za to sterowanie oraz czy potrafimy przewidzieć skutki ich zaburzenia?

## 1. Co jest i co nie jest „światłem” w tym modelu

W obecnym kodzie dokładnie zachodzi:

\[
F(x)=\sum_{c=1}^{768}u_c(x)+b_2,\qquad
u_c(x)=\operatorname{GELU}(w_c^\top L(x)+b_c)v_c,\qquad
x^+=x+F(x).
\]

L oznacza rzeczywisty LayerNorm. Stan x ma 192 współrzędne, a 768 współczynników GELU jest inną reprezentacją tokenu. Obraz daje 600 tokenów. MLP wyznacza pole indywidualne; uwaga sprzęga tokeny. Każdy blok ma własne wagi.

W fizycznej optyce, w przybliżeniu dipolowym i konwencji zespolonych amplitud z czasem exp(-iωt), występuje natomiast:

\[
\langle F_i\rangle=\frac12\operatorname{Re}
\left[\alpha E_j\,\partial_iE_j^*\right].
\]

To równanie zawiera rzeczywiste pole elektromagnetyczne i polaryzowalność cząstki. Jego część gradientowa ma postać Re(α)∇|E|²/4 w przyjętej konwencji. Pozostałe składniki zależą m.in. od fazy i rozpraszania. Nie należy mieszać konwencji jednostek dla α.

Źródło: Chaumet i Nieto-Vesperinas, *Optics Letters*2000, równania 7–9; [DOI](https://doi.org/10.1364/OL.25.001065), [preprint autorów](https://arxiv.org/abs/physics/0305043).

W sprawdzonym modelu nie ma fazy optycznej, długości fali, zespolonej amplitudy, polaryzowalności ani równań Maxwella. Dodawanie podpisanych u_c nie jest jeszcze interferencją koherentnych pól |ΣE_c|². Nazwa „laser” nie dostarcza tych dodatkowych własności. Ujemny współczynnik GELU jest poprawny jako współczynnik działania wektora.

Można natomiast korzystać z fizyki jako **języka pomiaru znanego pola działania**. Programowalne pułapki ze sprzężeniem zwrotnym pokazują, jak zmierzone położenie steruje obliczanym działaniem aktuatora [Kumar i Bechhoefer2018](https://doi.org/10.1063/1.5055580). Jest to dobry most do mikroskopu obliczeń, przy jawnie zadanych współrzędnych, metryce i zegarze.

## 2. Minimalny słownik i dwa zegary

**Głębokość:** X_{l+1}=T_l(X_l), gdzie X obejmuje wszystkie 600 tokenów, a T_l jest rzeczywistym podkrokiem sieci. Dla mapy residual T_l=Id+F_l. L=12 podkroków. Tu badamy jednorazowy transport perturbacji przez różne mapy.

**Uczenie:** x_j(t)=f_j(I;θ_t,b_t). Stały obraz jest odczytywany w kolejnych stanach modelu. Stan procesu uczącego obejmuje θ, momenty AdamW, BatchNorm, harmonogram i źródło losowości. Jednostką czasu jest rzeczywisty krok optymalizatora. Wzory temporalne stosujemy do rzeczywistych odstępów, nie do samego indeksu checkpointu.

**Odczyt zadania:** zalecany wektor y_j=(a_j,b_left,b_top,b_right,b_bottom), gdzie a jest surowym logitem, a b zdekodowanymi krawędziami w pikselach lub normalizacji obrazu. Raportować logit i geometrię ramki osobno. Odczyt jednej ustalonej komórki i wynik całego detektora po wyborze kandydatów są różnymi obserwablami. Argmax/NMS mogą być niegładkie; pochodne dotyczą gładkiego odczytu przed nimi, a końcową detekcję sprawdzamy forward.

Nie porównywać bez normalizacji jednostek logitu z pikselami. Wprowadzić jawną macierz wag R, np. przez ustalone tolerancje błędu ramki. Nie dopasowywać skali osobno do każdego checkpointu w sposób usuwający badany efekt.

## 3. Atlas metod i kolejność wdrożenia

| Nr | Pomiar | Co może odsłonić | Dane i priorytet |
|---|---|---|---|
| 1 | Lokalny tensor odpowiedzi i kierunkowe przywracanie | W jakich kierunkach blok przeciwdziała zaburzeniu | Zapisane wagi, aktywne sondy; pierwszy etap |
| 2 | Cyrkulacja pola po zamkniętej pętli | Działanie niesprowadzalne lokalnie do potencjału | Zapisane wagi; pierwszy etap |
| 3 | Wkład każdego neuronu do przywracania i cyrkulacji | Funkcje pojedynczych reflektorów i ich zespołów | Zapisane wagi i aktywacje; pierwszy etap |
| 4 | Odwrócone sondowanie token–token | Kierunkowe wykorzystanie obiektu i kontekstu | Pełna uwaga i JVP; pierwszy etap |
| 5 | Transport małej chmury perturbacji | Rozciąganie, obrót i przejściowe wzmocnienie | Pełna propagacja, bez rzutowania po drodze |
| 6 | Odpowiedź głowy na perturbację | Stabilność informacji o środku i ramce | Rzeczywista głowa i decode; pierwszy etap |
| 7 | MSD, korelacje i dokładny model OU | Jak zmienia się skala fluktuacji; test modelu pułapki | Istniejące trajektorie do opisu; ograniczenia czasowe |
| 8 | Wariancja/kowariancja Allana | Skala uśredniania, drgania odpowiedzi, dryf | Surowe trajektorie i rzeczywiste czasy |
| 9 | Antysymetria korelacji i pole powierzchni trajektorii | Uporządkowanie kolejności zmian, strzałka czasu | Gęste dane i kontrola niestacjonarności |
| 10 | Warunkowy dryf, dyfuzja i test pamięci | Czy stan tokenu wystarcza do modelowania następnego kroku | Wiele przejść, najlepiej replikacje |
| 11 | Odpowiedź na impuls w procesie uczenia | Czas i znak odroczonego wpływu aktualizacji | Osobne sparowane biegi z pełnym stanem |
| 12 | Fluktuacje wobec odpowiedzi, FDT/Harada–Sasa | Czy prosty model fluktuacji wyjaśnia aktywną odpowiedź | Osobne eksperymenty; termodynamika warunkowa |

„Pierwszy etap” oznacza, że nowe pomiary można oprzeć na istniejących checkpointach bez powtarzania treningu. Nie oznacza, że opisane wyniki już obliczono.

## 4. Metoda 1 — tensor pola, czyli w którą stronę blok przywraca

**Fizyka:** wokół położenia odniesienia F(x_0+δ)≈F(x_0)+Jδ. W liniowej pułapce F≈-Kδ. Metody rekonstrukcji pola, np. FORMA, identyfikują także odpowiedź niekonserwatywną [Pérez García i in.2018](https://doi.org/10.1038/s41467-018-07437-x).

**Adaptacja własna do sieci:**

\[
J=DF(x_0),\quad S=(J+J^\top)/2,\quad
\kappa(e)=-e^\top S e,\quad \|e\|=1.
\]

κ(e)>0 oznacza kierunkowe przeciwdziałanie zmianie w lokalnym polu różnicowym. Jest to wielkość w jednostkach reprezentacji na jednostkę reprezentacji, nie N/m. Odjęcie F(x_0) usuwa dryf bazowy; nie dowodzi, że x_0 jest równowagą.

**Wykonanie:** sondy ±εe, porównanie centralnej różnicy z autodiff, kilka ε i punktów odniesienia. Kierunki: gradient logitu, gradienty krawędzi ramki, kierunki dominujące S i kontrolne losowe. Dla uwagi przeliczać Q,K,V wszystkich zależnych tokenów.

**Co może powiedzieć o zadaniu:** czy blok zaczyna przeciwdziałać perturbacjom wpływającym na lokalizację krawędzi, a inaczej traktuje perturbacje niewidoczne dla głowy.

**Konieczna poprawka dyskretna:** dla liniowego przybliżenia δ⁺=Mδ zachodzi dokładnie:

\[
M=I+J,\qquad
\|\delta^+\|^2-\|\delta\|^2
=\delta^\top(2S+J^\top J)\delta.
\]

W rzeczywistej nieliniowej mapie do prawej strony dochodzi reszta o(||δ||²). Dodatnia κ nie gwarantuje kontrakcji skończonego kroku. Dla hipotetycznej powtarzanej iteracji liniowej asymptotyczna stabilność zachodzi wtedy i tylko wtedy, gdy ρ(M)<1; jednokrokowa kontrakcja wszystkich kierunków wymaga σ_max(M)<1. Dla gładkiej mapy nieliniowej ρ(M)<1 jest warunkiem wystarczającym lokalnej asymptotycznej stabilności punktu stałego, a ρ(M)=1 nie rozstrzyga jej bez dalszej analizy. Prawdziwa sieć wykonuje dany blok raz.

**Kontrola:** pre-LN ma dokładny neutralny kierunek 1. Usunąć go z analizy stabilności lub raportować jawnie. Nie przedstawiać ścisłego uwięzienia całej przestrzeni jako możliwego w tym samym sensie co fizyczna pułapka 3D.

## 5. Metoda 2 — cyrkulacja po pętli, czyli pomiar „skręcania”

**Fizyka:** pole gradientowe ma zerową całkę po zamkniętej pętli w odpowiedniej dziedzinie. Niezerowa cyrkulacja jest sposobem wykrycia niekonserwatywności; takie pola mierzy się w pęsetach [Wu i in.2009](https://doi.org/10.1103/PhysRevLett.103.108101), [Pesce i in.2009](https://doi.org/10.1209/0295-5075/86/38002).

**Adaptacja:** w ustalonym checkpoincie, bloku, kontekście i metryce wybieramy ortonormalne kierunki e,f i małą pętlę C wokół x_0:

\[
W_C=\oint_C F(x)^\top dx,\qquad
\frac{W_C}{\mathcal A_C}\longrightarrow
f^\top Je-e^\top Jf=2f^\top A e,\quad A=(J-J^\top)/2.
\]

Orientacja dodatnia biegnie najpierw wzdłuż e, potem f. W_C ma jednostki kwadratu współrzędnej latentu; przy bezwymiarowej reprezentacji jest bezwymiarowe. Nie nazywać wyniku energią w dżulach.

**Protokół:** koło lub kwadrat w pełnych 192D, coraz mniejszy promień, coraz gęstsza kwadratura, odwrócenie orientacji, sprawdzenie zgodności z J. Pętla jest zadanym eksperymentem próbnym, nie twierdzeniem, że token sam nią krąży.

**Co może odsłonić:** czy lokalne działania wiążą np. zmianę logitu z korektą szerokości ramki; czy kierunek tego sprzężenia zmienia się podczas uczenia; czy zespoły neuronów redukują lub wzmacniają cyrkulację w płaszczyznach ważnych dla zadania.

**Granice:** niezerowa cyrkulacja nie dowodzi stabilnej orbity ani termodynamicznego prądu. Pętla w jednym zamrożonym polu różni się od dodawania perturbacji w kolejnych warstwach. Te drugie mają efekty zależne od miejsca aplikacji [Li2026](https://arxiv.org/abs/2608.25315).

Dla stałej metryki G używamy ∮(GF)ᵀdx i odpowiednio GJ. Przy zmianie bazy metrykę trzeba transformować razem z polem. Dla metryki zależnej od x dochodzą pochodne G; nie używać bezrefleksyjnie poprzedniego wzoru.

## 6. Metoda 3 — który reflektor przywraca, a który obraca

To szczególnie dobra adaptacja do Twojego dokładnego rozkładu MLP. Nie jest nowym prawem fizycznym; jest rozłożeniem znanych miar pola na rzeczywiste składniki modelu.

Dla a_c=w_cᵀL(x)+b_c:

\[
J_c=\operatorname{GELU}'(a_c)\,v_cw_c^\top DL(x),\qquad J=\sum_cJ_c.
\]

Dostajemy dokładne wkłady:

\[
\kappa_c(e)=
-\operatorname{GELU}'(a_c)(e^\top v_c)(w_c^\top DL(x)e),
\]

\[
\Gamma_c(e,f)=\operatorname{GELU}'(a_c)
\left[(f^\top v_c)(w_c^\top DL(x)e)
-(e^\top v_c)(w_c^\top DL(x)f)\right].
\]

\[
\kappa(e)=\sum_c\kappa_c(e),\qquad
\Gamma(e,f)=\sum_c\Gamma_c(e,f),\qquad
W_C=\sum_c\oint_Cu_c^\top dx.
\]

Stały bias nie daje cyrkulacji po domkniętej pętli. GELU' może mieć znak ujemny; nie zastępować go samą aktywacją ani cosinusem wag.

**Co zobaczymy:** neuron może wnosić duże przesunięcie, ale małą odpowiedź różniczkową, albo odwrotnie. Można odróżnić „pcha token” od „zmienia sposób reagowania tokenu na zaburzenie”. Można wykryć grupy o dużych, przeciwnych Γ_c, mimo małej sumy, lub o selektywnym κ_c w kierunkach ramki.

**Test funkcji:** grupy wybieramy na zbiorze rozwojowym, potem zmieniamy ich wkłady w forward i sprawdzamy ramkę oraz logit na nowych obrazach. Porównania: ranking normy u_c, aktywacji, gradientu i losowe grupy o dobranej normie interwencji. Sama dokładność sumy jest kalibracją narzędzia, nie odkryciem.

**Prior:** rozkład podpisanych neuronowych wkładów jest znany [Geva2022](https://aclanthology.org/2022.emnlp-main.3/); koalicje i kompensacje także mają literaturę. Potencjalny wynik to zmiana ich konkretnych ról i przyczynowego znaczenia w uczącym się detektorze.

## 7. Metoda 4 — czy tło steruje narzędziem tak samo jak narzędzie tłem

**Fizyka:** optycznie sprzężone cząstki mogą mieć niesymetryczne odpowiedzi wzajemne. Reisenbauer i in. sterują częścią wzajemną i przeciwwzajemną sprzężenia przez rzeczywiste parametry optyczne [Nature Physics2024](https://doi.org/10.1038/s41567-024-02589-8).

**Adaptacja do jednego podkroku uwagi:**

\[
J_{ij}=\frac{\partial F_i}{\partial x_j},\qquad
N_{ij}=J_{ij}-J_{ji}^\top.
\]

Dla zadanych kierunków e u tokenu j i f u tokenu i:

\[
n_{ij}(e,f)=f^\top J_{ij}e-e^\top J_{ji}f.
\]

Liczymy dwie centralne różnice lub JVP. Nie trzeba materializować 115200×115200. Przy potencjale całego układu i tej samej metryce odpowiednie bloki Hessianu byłyby transpozycjami.

**Pytanie o detekcję:** które fragmenty tła wpływają na reprezentację narzędzia, jak silna jest droga odwrotna i jak ten układ zmienia się z checkpointu na checkpoint. Osobno mierzymy końcową zmianę ramki po perturbacji tokenu źródłowego.

**Kontrole:** odległość na siatce, pozycja, normy, widoczność obiektu, losowe pary i seed. Przeliczać wszystkie zależności Q,K,V. Macierz attention weight sama nie jest macierzą odpowiedzi.

**Granice:** dla jednego bloku mamy lokalną odpowiedź oddziaływania. Dla całego stosu J_ij obejmuje drogi pośrednie; nazwać to odpowiedzią całkowitą. N_ij≠0 nie jest osobnym dowodem złamania fizycznej zasady zachowania pędu przez sieć.

## 8. Metoda 5 — deformacja małej chmury i przejściowe wzmocnienie

**Fizyka/dynamika:** mała kula zaburzeń pod wpływem liniowej mapy staje się elipsoidą. Jej osie opisują wartości singularne, a tensor Cauchy'ego–Greena opisuje zmianę długości.

\[
M_l=DT_l(X_l),\quad
\Phi_{b:a}=M_{b-1}\cdots M_a,\quad
C_{b:a}=\Phi_{b:a}^\top\Phi_{b:a}.
\]

\[
g(e)=\|\Phi e\|/\|e\|,\qquad
\lambda_i^{(b:a)}=\frac{\log\sigma_i(\Phi)}{b-a}.
\]

Mianownik to liczba podkroków, nie sekundy. Przy różnej metryce wejścia/wyjścia należy ją jawnie uwzględnić.

**Co mierzyć:** końcowe wzmocnienie i maksimum po prefiksach, max_b||Φ_{b:a}e||. Perturbacja może najpierw rosnąć, a potem maleć; samo końcowe spektrum tego nie pokaże. Dla mapy nienormalnej wartości własne też nie wystarczają.

**Co to może powiedzieć:** w których etapach sieć wzmacnia sygnał krawędzi narzędzia, a gdzie ogranicza zakłócenia. Do rozróżnienia sygnału i zakłócenia potrzebne są kontrolowane kierunki oraz odczyt zadania, nie sama wielkość rozciągania.

**Wykonanie:** JVP/VJP lub propagate ±ε przez rzeczywiste wszystkie tokeny. Nie rzutować po każdym kroku. Końcowy rzut 3D służy prezentacji; pełny pomiar zachowuje kierunki niewidoczne w filmie.

**Nowość:** FTLE w sieciach już istnieją [Storm i in., PRL2024](https://doi.org/10.1103/PhysRevLett.132.057301). Globalne operatory odpowiedzi i ich zmiana w trakcie uczenia również [Hazelden2026](https://arxiv.org/abs/2609.40292). Te wzory są metodą bazową, nie samodzielnym nowym odkryciem.

## 9. Metoda 6 — podatność ramki i logitu, najważniejszy odczyt mikroskopu

Dla rzeczywistej końcowej głowy i decode:

\[
B=D_{X_L}y(X_L),\qquad
H_a=B\Phi_{L:a},\qquad
\delta y\approx H_a\delta X_a.
\]

Jeżeli perturbujemy tylko wybrane 192 współrzędne tokenu j, osadzamy je macierzą E_j:

\[
H_{a,j}=B\Phi_{L:a}E_j.
\]

Rozdzielamy wiersz h_logit i cztery wiersze H_box. Macierze:

\[
W_{\rm logit}=h_{\rm logit}^\top h_{\rm logit},\qquad
W_{\rm box}=H_{\rm box}^\top R_{\rm box}H_{\rm box}
\]

opisują odpowiednio kwadrat lokalnej zmiany logitu i ważony kwadrat zmiany ramki dla kierunku perturbacji. Nie są automatycznie zmianą błędu względem anotacji b_*: ta ostatnia wynosi 2(b−b_*)ᵀR_box δb + δbᵀR_box δb. Mała podatność może również utrwalać błędną ramkę, dlatego odczyt trzeba zestawić z poprawnością detekcji.

**Znaczenie:** dwa kierunki o podobnym ruchu w latencie mogą dać zupełnie inne skutki dla detektora. To pozwala operacyjnie określić, co sieć chroni, co wykorzystuje i jakie zmiany są dla wyjścia obojętne.

**Szczególnie ciekawy test:** czy po treningu rośnie różnica między reakcją na naturalne zmiany obrazu ważne dla krawędzi narzędzia a reakcją na zakłócenia tła o porównywalnej skali. Sama niska podatność może oznaczać także ignorowanie potrzebnego sygnału; oceniać razem z poprawnością detekcji.

**Konieczne rozróżnienie:** latentny wektor niewidoczny dla głowy w pierwszym rzędzie nie jest automatycznie semantycznie nieistotny dla innych obrazów ani skończonych perturbacji.

W tym modelu jest jedna właściwa głowa na końcu. Suma odczytów hipotetycznych głów przy każdej warstwie byłaby innym eksperymentem. Pomiar H ma standardową podstawę w teorii odpowiedzi i obserwowalności; dokładna zgodność δy≈HδX w granicy ε→0 jest testem implementacji. Wynik naukowy musi dotyczyć struktury H, jej genezy i funkcjonalnej roli składników.

Prior obejmuje [Moon2025](https://arxiv.org/abs/2511.12852) i [Hazelden2026](https://arxiv.org/abs/2609.40292); zastosowania i definicje nie są identyczne z proponowanym odczytem detektora.

## 10. Metoda 7 — drgania, MSD i OU jako testowany model

**Podstawowy wzór fizyczny** dla jednego kierunku termicznej pułapki:

\[
\gamma\dot q=-k(q-q_*)+\sqrt{2\gamma k_BT}\,\xi(t).
\]

Definiując λ=k/γ oraz D=k_BT/γ, dla stanu stacjonarnego:

\[
C(\tau)=\frac D\lambda e^{-\lambda|\tau|},\qquad
MSD(\tau)=\frac{2D}{\lambda}(1-e^{-\lambda|\tau|}),
\]

\[
S_q(\omega)=\frac{2D}{\lambda^2+\omega^2}.
\]

S jest widmem dwustronnym z kowariancją odtwarzaną przez całkę S(ω)exp(-iωτ)dω/(2π). Nie mieszać z jednostronnym PSD w hercach.

**Dokładne przejście dla nierównych odstępów:**

\[
q_{n+1}-q_*=
e^{-\lambda\Delta t_n}(q_n-q_*)+\eta_n,\quad
\operatorname{Var}\eta_n=
\frac D\lambda(1-e^{-2\lambda\Delta t_n}).
\]

Wielowymiarowo, dq=-B(q-μ)dt+sqrt(2D)dW:

\[
A_\Delta=e^{-B\Delta},\qquad
Q_\Delta=2\int_0^\Delta e^{-Bs}D e^{-B^\top s}\,ds.
\]

**Co można zrobić teraz:** opisać surowe MSD/logit/bbox według rzeczywistych lagów i faz treningu. Zestawić OU z dryfem, szumem obserwacyjnym i modelem pamięci; oceniać przewidywanie następnych checkpointów oraz przedziały niepewności.

**Co dałby pozytywny wynik:** lokalnie uzasadnione czasy utrzymywania zmian reprezentacji i ich zależność od zadania. Sama zgodność jednej krzywej nie identyfikuje siły, temperatury ani pełnego procesu.

Źródła metrologiczne: [Lansdorp i Saleh2012](https://doi.org/10.1063/1.3687431), [Pérez-García i in.2023](https://doi.org/10.1364/BOE.495468). Dla rzadkiego próbkowania potrzebny jest model obserwacji; gęste i rzadkie części biegu analizować osobno.

## 11. Metoda 8 — wariancja Allana: jak długo uśredniać i co naprawdę drga

Dla średniej obserwabli w oknie czasu τ:

\[
\sigma_A^2(\tau)=\frac12
\mathbb E[(\bar y_{k+1,\tau}-\bar y_{k,\tau})^2].
\]

Dla wektora można mierzyć macierz:

\[
\Sigma_A(\tau)=\frac12
\mathbb E[(\bar y_{k+1}-\bar y_k)(\bar y_{k+1}-\bar y_k)^\top].
\]

W implementacji użyć jawnie średnich z sąsiednich okien. Dla nieregularnych checkpointów nie przypisywać automatycznie takiej samej wagi każdemu zapisowi jako próbce czasu. Analizować regularne segmenty albo jawny model całkowania/obserwacji i jego błąd.

**Co to daje w Twoim zadaniu:** osobne krzywe logitu, środka, szerokości, wysokości i krawędzi pokażą, czy detektor wcześniej stabilizuje obecność narzędzia niż jego rozmiar; na jakich skalach uśrednianie redukuje drgania, a kiedy zaczyna mieszać różne fazy uczenia.

**Ważna interpretacja:** stacjonarny OU ma charakterystyczne **maksimum** krzywej Allana około 1,89/λ. Minimum może powstać przy konkurencji szumu i dryfu; nie jest uniwersalną miarą relaksacji pułapki. Dla niezależnego szumu próbek o wariancji s² średnia z m próbek daje wkład s²/m. Dla dryfu y=vt wkład wynosi v²τ²/2; to własne proste wyprowadzenia kontrolne.

Źródło definicji i modelu pułapki: [Lansdorp i Saleh2012](https://doi.org/10.1063/1.3687431), równania 12–17. Pomiar jest użyteczny jako diagnostyka nawet przy odrzuceniu OU.

## 12. Metoda 9 — czy fluktuacje mają uporządkowany kierunek obiegu

Dla lokalnie stacjonarnych, wycentrowanych obserwabli y:

\[
C(\tau)=\mathbb E[y(t+\tau)y(t)^\top],\qquad
\mathcal A(\tau)=C(\tau)-C(\tau)^\top.
\]

Przykład: składowa logit–szerokość ramki może pokazać, czy ich zmiany mają uporządkowaną kolejność. Może to być skutkiem uczenia, pamięci optymalizatora lub wspólnego sterowania; sama asymetria nie identyfikuje przyczyny.

Dyskretne zorientowane pole powierzchni w wybranej płaszczyźnie:

\[
\mathcal S_{ab}=
\frac12\sum_n(y_{a,n}y_{b,n+1}-y_{b,n}y_{a,n+1}).
\]

Podzielić przez rzeczywisty czas obserwacji i raportować konwencję znaku. Dla otwartej trajektorii wynik zależy od wyboru początku współrzędnych i domknięcia; trzeba ustalić centrum, kontrolować trend oraz efekty brzegowe. Sekwencja próbek w punktach 3D nie pokazuje obiegu między rzadkimi checkpointami.

**Kontrole:** odwrócenie czasu zmienia znak; permutacja czasu; dopasowany model dryfu i wspólnego napędu; stała baza; bootstrap blokowy lub niezależne biegi; wiele lagów bez wybierania najbardziej efektownego.

Fizyczny pomiar asymetrii korelacji: [Pesce i in.2009](https://doi.org/10.1209/0295-5075/86/38002). **Prior ML istnieje:** [Adhikari i in.2023](https://arxiv.org/abs/2306.03521) badają m.in. stochastic area i prądy w przestrzeni wag SGD. Nie można ogłosić pierwszego przeniesienia tej idei do uczenia. Odrębny problem stanowi odczyt tokenów i detekcji przy AdamW.

## 13. Metoda 10 — dryf, dyfuzja i pytanie, czy token ma ukrytą pamięć

Dla założonej dyfuzji Markowa:

\[
b(q)=\lim_{\Delta\to0}\frac{\mathbb E[\Delta q\mid q]}{\Delta},\qquad
D(q)=\lim_{\Delta\to0}
\frac{\operatorname{Cov}[\Delta q\mid q]}{2\Delta}.
\]

Przy skończonym odstępie są to estymatory warunkowych przyrostów, nie automatycznie współczynniki generatora. 806 checkpointów jednego niestacjonarnego przebiegu nie wystarcza do nieparametrycznej rekonstrukcji 192D b i D. Zacząć od małego zestawu jawnych obserwabli i modeli o kontrolowanej złożoności.

**Test pamięci:** dla jednorodnego w czasie procesu Markowa propagatory spełniają:

\[
P_{\tau_1+\tau_2}(q_2\mid q_0)=
\int P_{\tau_2}(q_2\mid q_1)P_{\tau_1}(q_1\mid q_0)dq_1.
\]

Odtwarzalne naruszenie przy spełnionych założeniach wskazuje pamięć obserwabli. Spełnienie tej równości nie dowodzi Markowowskości. Niestacjonarność i błąd estymacji też powodują rozbieżności. Praktyczny dodatkowy test: czy historia i mierniki momentów AdamW poprawiają prognozę poza próbką w porównaniu z samym obecnym q.

**Co to daje:** rozdzielenie pozornego „drgania w studni” od pamięci sterownika i wspólnego napędu; wskazanie, jakie zmienne trzeba jeszcze rejestrować.

Źródło testów propagatorów: [Lapolla i Godec, PRR2021](https://doi.org/10.1103/PhysRevResearch.3.L022018). Ich praca dotyczy m.in. DNA w pęsecie oraz dynamiki molekularnej; zastosowanie do tokenów jest proponowaną adaptacją.

## 14. Metoda 11 — impuls w uczeniu: co faktycznie wraca i jak długo

To najbliższy odpowiednik aktywnego eksperymentu dynamicznego, jeśli pytamy o ruch **między checkpointami**.

Z tego samego pełnego stanu s_t=(θ,m,v,b,licznik) tworzymy dwie gałęzie. W jednym kroku dodajemy +εr lub -εr do jawnie wybranej wielkości, np. gradientu. Dalsze minibatchy, augmentacje i losowość w parze są identyczne. Definicja perturbacji musi mówić, czy jest przed clippingiem czy po nim.

\[
R_{y,r}(t,\tau)=
\frac{y^+_{t+\tau}-y^-_{t+\tau}}{2\epsilon}.
\]

**Co mierzyć:** zmianę logitu, krawędzi ramki i reprezentacji, jej znak, maksimum, czas maksimum i utrzymywanie się. Utrata wpływu perturbacji nie musi oznaczać powrotu do nieruchomego punktu; porównujemy do równolegle uczącej się kontroli.

**Co można odkryć:** czy impuls pozornie poprawiający ramkę natychmiast szkodzi jej po kilkudziesięciu krokach; czy opóźnienie wynika z pamięci optymalizatora; czy wybrane grupy parametrów kontrolują inny horyzont odpowiedzi.

**Prior bardzo bliski:** [Liu i Li2026](https://arxiv.org/abs/2608.19762) analizują sparowane trajektorie AdamW ze wspólną przyszłością i lokalną perturbacją gradientu. To preprint, nie potwierdzona publikacja czasopismowa. Nie opisuje ViT/bbox ani pełnego uwzględnienia BatchNorm i clippingu. Idea sparowanych gałęzi nie jest nowa.

Nowe gałęzie są oddzielnymi eksperymentami; nie nadpisują zakończonego biegu. Stary zapis wag bez momentów nie pozwala wiernie odtworzyć oryginalnego stanu AdamW.

Sinusoidalny protokół optyczny [Tolić-Nørrelykke i in.2006](https://doi.org/10.1063/1.2356852) inspiruje również badanie odpowiedzi częstotliwościowej, ale musi istnieć rzeczywisty powtarzany zegar: np. sterowany proces uczenia. Sześć różnych bloków nie daje samo z siebie stacjonarnego widma podatności.

**Rozwinięcie: wymuszenie okresowe.** W fizycznym liniowym modelu γ dq/dt = −kq + u(t), przy konwencji u(t)=Re[u_ω exp(−iωt)], podatność wynosi χ_q(ω)=1/(k−iγω). Określa amplitudę i opóźnienie odpowiedzi. Dla lokalnie stałego, zdyskretyzowanego modelu pełnego stanu uczenia δs_(n+1)=Aδs_n+B_in u_n oraz δy_n=Hδs_n analogiem jest:

\[
\chi_y(\Omega)=H(e^{-i\Omega}I-A)^{-1}B_{\rm in}.
\]

Ω jest częstością w radianach na krok optymalizatora. Założenia obejmują stabilność, mały sygnał, stałość współczynników w badanym oknie i brak bezpośredniego składnika wejście–wyjście; jeżeli taki składnik występuje, dodaje się go do χ. W rzeczywistym niestacjonarnym treningu właściwsza może być odpowiedź dwuczasowa R(t,τ). Pomiar odpowiada na pytanie, przy jakim tempie wymuszenia ramka nadąża za zmianą i jak opóźniona jest pewność detekcji. Szczegóły protokołu są w podraporcie Scite.

## 15. Metoda 12 — fluktuacje wobec odpowiedzi i granica termodynamiki

W fizycznym stanie równowagowym fluktuacje i odpowiedź na sprzężone wymuszenie są powiązane. Dla prostego nadtłumionego procesu równanie Harady–Sasy ma postać:

\[
\langle\dot Q\rangle=
\gamma\left[
\bar v^2+\int_{-\infty}^{\infty}
\left(\widetilde C_{vv}(\omega)
-2k_BT\,\operatorname{Re}\widetilde R_{vF}(\omega)\right)
\frac{d\omega}{2\pi}
\right].
\]

C dotyczy wycentrowanych fluktuacji prędkości, R odpowiedzi prędkości na małą dodatkową siłę, γ tarcia, a T temperatury kąpieli. Nie pomijać składnika średniego dryfu. Źródło: [Harada i Sasa, PRL2005](https://doi.org/10.1103/PhysRevLett.95.130602), równania 1–6.

**Co można przenieść:** pytanie, czy aktywna odpowiedź z metody 11 jest przewidywalna z biernych fluktuacji przy określonym modelu i tej samej dynamice.

**Czego nie wolno zrobić:** połączyć szumu checkpointów z odpowiedzią tokenu podczas forward, dobrać stałą „temperaturę” i nazwać resztę ciepłem. To dwa zegary i dwa różne doświadczenia. Potrzebne są właściwa perturbacja, model szumu, mobilność i zgodność jednostek.

Bardzo bliski fizyczny przykład: [Goerlich i in., PRE2026](https://doi.org/10.1103/kldw-kftz), [czytany preprint](https://arxiv.org/abs/2601.05198). Cząstka jest napędzana kolorowym szumem. Sama pozycja nie wystarcza do zwykłej relacji fluktuacja–odpowiedź; potrzebny jest wspólny stan pozycji i napędu oraz odpowiednia obserwabla sprzężona. To dobry kierunek myślenia o ukrytej pamięci AdamW, bez utożsamiania tych procesów.

**Prior ML:** [Yaida, ICLR2019](https://arxiv.org/abs/1810.00004) wyprowadza stacjonarne relacje fluktuacyjne dla uczenia; [Adhikari2023](https://arxiv.org/abs/2306.03521) bada stacjonarne prądy i entropię w modelach SGD. FDT nie jest niewykorzystanym pomysłem w świecie sieci.

Na pierwszym etapie raportować funkcję odpowiedzi i kowariancję oddzielnie. Test zgodności z modelem jest wartościowy nawet wtedy, gdy model termiczny zostaje odrzucony.

## 16. Prosty przykład pokazujący, po co tyle różnych pomiarów

To wyprowadzenie analityczne, nie nowy wynik SmallViT. Dla dwuwymiarowego modelu:

\[
dq=(-kI+\omega R)q\,dt+\sqrt{2D}\,dW,\qquad
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad k>0,
\]

stacjonarna kowariancja wynosi C=(D/k)I niezależnie od ω. Sama chmura punktów i jej wariancja nie rozróżniają modelu bez obiegu od modelu z obiegiem.

Tymczasem dla koła o promieniu r:

\[
\oint F\cdot dq=2\pi\omega r^2.
\]

Cyrkulacja i korelacje opóźnione niosą informację, której brak w wariancji. Gdy zaś ten sam algebraiczny F użyjemy jako kroku residual z krokiem 1, stabilność hipotetycznej wielokrotnej iteracji wymaga:

\[
(1-k)^2+\omega^2<1.
\]

Stabilność ciągła k>0 nie wystarcza. Ten przykład dobrze tłumaczy, dlaczego „mała chmura”, „wir” i „pułapka” są różnymi pytaniami pomiarowymi.

## 17. Co może stać się wynikiem publikacji

Nie proponuję tezy „odkryliśmy zastosowanie znanego wzoru do sieci”. Najbliższa obronna hipoteza:

**W trakcie uczenia detektora powstaje selektywna organizacja odpowiedzi: określone grupy neuronów i kierunkowe oddziaływania tokenów utrzymują informację potrzebną do lokalizacji narzędzia, podczas gdy pozostałe zmiany reprezentacji mogą być duże.**

Przewidywania do testowania:

1. Charakterystyki κ_c i Γ_c zależą od roli tokenu i kierunku odczytu, a ich organizacja zmienia się podczas uczenia.
2. Grupy wskazane przez te pomiary przewidują rzeczywiste skutki interwencji dla ramki lepiej niż proste kontrole normy i aktywacji.
3. Niesymetria odpowiedzi token–token ujawnia wykorzystywany kontekst; zmiana tej drogi pogarsza detekcję zgodnie z przewidywaniem.
4. Spadek fluktuacji ramki może współistnieć z utrzymującym się ruchem w latencie. Związek ten daje się wyjaśnić zmianą odczytu i transportu, a nie tylko globalnym zmniejszeniem norm.

Nie są to obecnie potwierdzone wyniki. Falsyfikacja: efekty znikają po kontrolach, nie generalizują między filmami/seedami albo nie przewidują interwencji ponad standardowy gradient/atrybucję. Wtedy instrument pozostaje użyteczny, lecz hipotezę mechanizmu trzeba zmienić.

**Najpierw wdrożyłbym:** metodę 6 jako odczyt zadania, metody 1–3 jako rozkład pola i metodę 4 jako pomiar kontekstu. Metodę 8 można równolegle zastosować opisowo do istniejących trajektorii. Eksperymenty 11–12 zostawiłbym jako osobny, kosztowniejszy etap.

## 18. Minimalny protokół pierwszej implementacji

To specyfikacja przyszłej pracy, nie wykonana implementacja.

- Wybrać kilka checkpointów obejmujących inicjalizację, wzrost jakości i koniec; kilka bloków oraz obrazy z niezależnych filmów.
- Zdefiniować tokeny obiektu, granicy i tła oraz odczyt gładkiej głowy. Wyborów nie dopasowywać po zobaczeniu najbardziej atrakcyjnych wyników.
- Odczytać float32/float64 z rzeczywistym epsilon LN. Nie używać kwantyzowanego bufora jako wzorca bardzo małych perturbacji.
- Wybrać kierunki ze względu na zadanie i kontrolne losowe. Stosować tę samą skalę normy perturbacji i kilka jej amplitud.
- Sprawdzić JVP centralnymi różnicami, sumę neuronowych κ_c/Γ_c, znak pętli, zbieżność kwadratury i neutralny kierunek LN.
- Propagować pełny stan przez uwagę. Przekroje 3D wyświetlać dopiero po pomiarze.
- Osobno raportować zmianę logitu, czterech krawędzi, IoU i jakości całej detekcji. Pojedynczy token nie zastępuje ewaluacji detektora.
- Przy interwencjach używać kontroli pojedynczych, wspólnych i normowo dopasowanych. Zmienione wejście do dalszych warstw może wywołać kompensację.
- Oceniać prognozy na nieużytych obrazach. Niepewność liczyć z uwzględnieniem wspólnych filmów i biegów, nie traktować 600 tokenów jako niezależnych replik.

### Dodatkowy eksperyment: zbudować jawną wirtualną pułapkę

Jeśli celem jest również sterowanie reprezentacją, można zdefiniować własny potencjał zadania i kontroler:

\[
U(X)=\tfrac12(y(X)-y_*)^\top R(y(X)-y_*)
+\tfrac\lambda2\|X-X_0\|^2,\qquad
X_{n+1}=X_n-\eta_c\nabla_X U(X_n).
\]

To nowy eksperyment z aktywną kontrolą latentu. Można badać osiągalność małej korekty krawędzi, wpływ gain i opóźnienia oraz koszt uboczny dla logitu i pozostałych krawędzi. Nie jest to równanie istniejącego forward ani dowód, że naturalny trening minimalizuje U. Dla linearyzacji δy≈HδX regularizowane rozwiązanie lokalne ma postać:

\[
\delta X=H^\top(HH^\top+\lambda I)^{-1}\delta y_*.
\]

Wzór zakłada jawnie znormalizowany odczyt i metrykę euklidesową; dla innych wag należy odpowiednio zmienić problem najmniejszych kwadratów. Cel musi leżeć w osiągalnym zakresie, a skutki skończonej zmiany należy sprawdzić przez pełny forward.

Most fizyczny: [Jun i Bechhoefer 2012](https://doi.org/10.1103/PhysRevE.86.061106) pokazują, że nawet prosta wirtualna pułapka zmienia stabilność pod wpływem opóźnienia. Dla x_(n+1)=(1−α)x_n+ξ_n warunek wynosi 0<α<2, a dla x_(n+1)=x_n−αx_(n−1)+ξ_n zawęża się do 0<α<1. Te zakresy dotyczą tych konkretnych skalarnych modeli; nie są gotowym zakresem learning rate dla ViT. Są dobrym uzasadnieniem pomiaru pełnej odpowiedzi i pamięci zamiast samej wariancji.

## 19. Granice nowości i status źródeł

| Metoda | Najbliższy znaleziony wcześniejszy zakres | Uczciwy zakres proponowanej adaptacji |
|---|---|---|
| Pole neuronów i jego pochodne | Geva 2022: neuronowe wkłady do predykcji; standardowy rachunek Jacobianów | Rozkład reakcji konkretnych neuronów względem celu detekcji w trakcie uczenia |
| FTLE/deformacja | Storm2024; Jacobiany transformerów | Pełna propagacja tokenów, ramka i interwencje |
| Operator odpowiedzi | Hazelden2026; Moon2025 | Odczyt rzeczywistej głowy ViT i źródła neuron/token |
| Kolejność interwencji/pętle | Li2026 pokazuje artefakty zależne od miejsca | Całka w jednym zamrożonym polu i jawne kontrole interwencji |
| Prądy i stochastic area | Adhikari2023 w wagach SGD | Obserwable reprezentacji i wyjścia przy AdamW, po kontroli dryfu |
| FDT | Yaida2019 i dalsze prace | Test modelu dla konkretnych obserwabli, nie pierwsze użycie FDT w ML |
| Pamięć impulsu AdamW | Liu i Li2026 | Stan z BatchNorm/clippingiem i wynik detekcji |

Artykuły optyczne o FORMA, kalibracji i niekonserwatywności są publikacjami czasopismowymi. Storm ma rok publikacji 2024 mimo rekordu preprintu 2023. Reisenbauer ma publikację 2024 mimo etykiety 2023 w agregatorze. Yaida to przyjęta wersja ICLR 2019, nie tylko preprint 2018. Goerlich: czytano preprint, a publikację PRE 114, 014138 (2026) potwierdzają metadane bibliograficzne. Li 2608.25315 i Liu 2608.19762 są czytanymi preprintami. Hazelden 2609.40292 ma deklarację przyjęcia na workshop NeurReps 2026; nie przedstawiać jako głównej konferencji NeurIPS.

Odnalezienie bliskiego wzoru nie dowodzi kompletności przeglądu, a brak trafienia nie dowodzi nowości. Wyciągi narzędzi były sprawdzane względem definicji i źródeł; surowe wyniki zachowano, także z błędnymi latami i DOI.

## 20. Zapis researchu i źródła

Undermind: 3 nowe ukończone deep searches, 12 dodatkowych zapytań semantycznych, ukierunkowany odczyt 12 pełnych PDF i weryfikacja metadanych. Nowe rankingi zawierają 155+195+96=446 wpisów, z możliwymi powtórzeniami; nie jest to liczba przeczytanych prac.

SciSpace: 15 nowych zapytań, 150 zwróconych rekordów i 139 różnych identyfikatorów; 7 pakietów weryfikacji źródeł pierwotnych. Scite: 17 zapytań merytorycznych oraz 4 kontrole uwag redakcyjnych, 78 trafień i 56 unikalnych rekordów. W Scite wykonano także 15 odczytów: 13 stron rzeczywistego tekstu z 10 DOI oraz 2 odpowiedzi zawierające wyłącznie abstrakty; ponadto eksport bibliografii i dwa sprawdzenia cytowań. Łącznie było 39 wywołań Scite. Liczb między usługami nie należy sumować jako unikalnych publikacji.

Ta runda obejmuje zatem 3 nowe deep searches oraz 44 dodatkowe kwerendy merytoryczne i 4 kontrole redakcyjne, a także ukierunkowane odczyty i weryfikację równań. Brak indeksowanego erratum nie jest gwarancją poprawności artykułu; status źródła i zakres przeczytanego tekstu zapisano w podraportach.

- [Metrologia pułapki jako próba transportu tokenów](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Optical%20trap%20metrology%20as%20a%20mechanistic%20probe%20of%20learned%20token%20transport).
- [Pola niekonserwatywne i współdziałanie neuronów](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Nonconservative%20optical%20forces%20and%20geometric%20diagnostics%20of%20neuron%20cooperation).
- [Fluktuacje, odpowiedź i pamięć podczas uczenia](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Fluctuation%20response%20and%20memory%20measurements%20for%20optimizer%20driven%20latent%20trajectories).

Pozostałe raporty i ślad źródłowy znajdują się w tym samym katalogu:

- [Matematyka i specyfikacja](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/MATEMATYKA_I_SPECYFIKACJA.md).
- [Raport SciSpace: 7 kart pomiarów i ocena nowości](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/scispace/RAPORT_WZORY_SCISPACE.md).
- [Raport Scite: 8 protokołów metrologicznych](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/scite/ATLAS_SCITE_WZORY.md).
- [BibTeX Scite](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/scite/bibliography.bib).
- [Surowe odczyty i wyszukiwania Undermind](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/undermind/RESEARCH_I_ODCZYTY.json).
- [Uzupełniające odczyty o odpowiedzi i pamięci](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/undermind/FULLTEXT_ODPOWIEDZ_PAMIEC.json).

Wybrane źródła pierwotne, uzupełniające linki przy wzorach:

1. Chaumet, P.C., Nieto-Vesperinas, M. (2000). Time-averaged total force on a dipolar sphere in an electromagnetic field. Optics Letters25,1065–1067. DOI10.1364/OL.25.001065.
2. Tolić-Nørrelykke i in. (2006). Calibration of optical tweezers with positional detection in the back focal plane. Review of Scientific Instruments77,103101. DOI10.1063/1.2356852.
3. Lansdorp, B.M., Saleh, O.A. (2012). Power spectrum and Allan variance methods for calibrating single-molecule video-tracking instruments. Review of Scientific Instruments83,025115. DOI10.1063/1.3687431.
4. Pérez García i in. (2018). High-performance reconstruction of microscopic force fields from Brownian trajectories. Nature Communications9. DOI10.1038/s41467-018-07437-x.
5. Wu, P., Huang, R., Tischer, C., Jonas, A., Florin, E.-L. (2009). Direct measurement of the nonconservative force field generated by optical tweezers. Physical Review Letters103,108101. DOI10.1103/PhysRevLett.103.108101. Agregator Undermind przypisał inny DOI, więc użyto identyfikatora PRL.
6. Pesce, G., Volpe, G., De Luca, A.C., Rusciano, G., Volpe, G. (2009). Quantitative assessment of non-conservative radiation forces in an optical trap. EPL86,38002. DOI10.1209/0295-5075/86/38002.
7. Reisenbauer i in. (2024). Non-Hermitian dynamics and non-reciprocity of optically coupled nanoparticles. Nature Physics20,1629–1635. DOI10.1038/s41567-024-02589-8.
8. Harada, T., Sasa, S. (2005). Equality connecting energy dissipation with a violation of the fluctuation-response relation. PRL95,130602. DOI10.1103/PhysRevLett.95.130602.
9. Lapolla, A., Godec, A. (2021). Toolbox for quantifying memory in dynamics along reaction coordinates. PRR3,L022018. DOI10.1103/PhysRevResearch.3.L022018.
10. Goerlich, R., Tartar, A., Roichman, Y., Sokolov, I.M. (2026). Fluctuation-response relation for a nonequilibrium system with resolved Markovian embedding. PRE114,014138. DOI10.1103/kldw-kftz. Czytany pełny tekst: arXiv2601.05198v1.
11. Storm, L., Linander, H., Bec, J., Gustavsson, K., Mehlig, B. (2024). Finite-Time Lyapunov Exponents of Deep Neural Networks. PRL132,057301. DOI10.1103/PhysRevLett.132.057301.
12. Yaida, S. (2019). Fluctuation-dissipation relations for stochastic gradient descent. ICLR2019. arXiv1810.00004.
13. Adhikari, S. i in. (2023). Machine learning in and out of equilibrium. Czytany preprint arXiv2306.03521.
14. Li, A.P. (2026). Activation-Space Order-Swap Geometry: A Site-Asymmetry Audit. Preprint arXiv2608.25315.
15. Hazelden, J. (2026). Disentangling Computation in Multi-Task Neural Networks with the Green's Operator. arXiv2609.40292; przyjęty na NeurReps2026.
16. Liu, K., Li, S. (2026). Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW. Preprint arXiv2608.19762.

