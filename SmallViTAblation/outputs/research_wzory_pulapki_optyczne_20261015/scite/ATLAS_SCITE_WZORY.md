# Osiem protokołów z fizyki pułapek do mikroskopu SmallViT
Stan: 2026-10-15. Przegląd Scite dotyczący konkretnych wzorów i pomiarów. Proponowane pomiary nie zostały uruchomione. Kod aplikacji i modele pozostają bez zmian.

## Najważniejsza propozycja

W pierwszej kolejności wdrażać **aktywny pomiar wpływu zaburzenia na bbox**, **tensor odpowiedzi bloku**, **Allan variance wyników** oraz **wspólny test OU/ACF/MSD/PSD z poprawnym próbkowaniem**. Dają mierzalną wiedzę o detekcji narzędzia bez wprowadzania temperatury, energii ani rzeczywistej siły optycznej. Wymuszenia sinusoidalne podczas uczenia, feedback, FDT i stochastic force inference są drugim etapem: potrzebują osobnych kontrolowanych biegów i gęstych trajektorii.

| Metoda | Co uzyskujemy | Gdzie działa | Dostępność |
|---|---|---|---|
| 1. Tensor pola i aktywna kalibracja | Kierunki lokalnej odpowiedzi i jej niesymetrii | Ustalony checkpoint, podkrok | Możliwe na zapisanych modelach |
| 2. Impuls i propagacja do bbox | Przyczynowa czułość krawędzi i logitu | Głębokość, pełne forward | Możliwe na zapisanych modelach |
| 3. Susceptibility sinusoidalna | Pasmo odpowiedzi, opóźnienia, pamięć | Uczenie lub jawny simulator feedback | Nowy kontrolowany eksperyment |
| 4. OU + ACF/MSD/PSD + likelihood | Relaksacja i szum rozdzielone od trendu | Uczenie w lokalnym oknie | Częściowo teraz; gęstość może być niewystarczająca |
| 5. Allan variance | Skala fluktuacji, dryfu i użytecznego uśredniania | Surowe checkpointy i wyniki | Możliwe na istniejących danych |
| 6. Wirtualna pułapka feedback | Test sterowalności i zakresu stabilnej korekcji bbox | Osobna interwencja w latent lub trening | Nowy eksperyment |
| 7. FDT / Harada–Sasa | Test zgodności fluktuacji z kontrolowaną odpowiedzią | Jeden, jasno zdefiniowany proces | Dopiero po kalibracji wejścia i szumu |
| 8. SFI + model pamięci | Dryf, dyfuzja, prądy oraz błędy modelu Markowa | Gęste trajektorie uczenia | Nowy eksperyment; mały wymiar obserwabli |

## Wspólne konwencje

**Dwa zegary.** `l=0,...,11` oznacza podkroki forward przy zamrożonych wagach. `n` albo `t` oznacza rzeczywiste aktualizacje optymalizatora. Kolejność 806 checkpointów nie jest jednostajną osią czasu; należy używać numerów aktualizacji.

**Pełny stan.** `X_l∈R^(600×192)`. Lokalny token `x_j∈R^192`. 768 aktywacji GELU nie jest wektorem położenia tego tokenu. Uwaga sprzęga tokeny, więc pełna propagacja zaburzenia obejmuje wszystkie tokeny, nawet gdy zaburzono tylko jeden.

**Odczyt związany z zadaniem.** Używamy raw head `o=(a,ell_L,ell_T,ell_R,ell_B)`, gdzie a jest logitem. Dla ustalonego tokenu, jego kotwicy `(c_x,c_y)` w pikselach i kroku siatki `s=16`, dekodowane krawędzie mają postać
`b=(c_x-s exp ell_L, c_y-s exp ell_T, c_x+s exp ell_R, c_y+s exp ell_B)`.
Kotwicę należy wziąć dokładnie z dekodera modelu. Definiujemy `q=(b_L,b_T,b_R,b_B,a)` oraz `H_l=∂q/∂X_l`. Cztery pierwsze wyjścia mają jednostkę pikseli; logit jest bezwymiarowy. Przed wspólnym SVD trzeba wybrać jawną skalę, np. krawędzie podzielone przez szerokość/wysokość GT i logit osobno.

**Ocena detekcji.** Gradient dotyczy ustalonego tokenu i gładkiego odczytu przed progowaniem/max-pooling/NMS. Zmiana zwycięskiego tokenu i mAP wymagają osobnego rzeczywistego dekodowania. Mierzyć również błąd środka, szerokości, wysokości i IoU. Nie wnioskować o mAP tylko z lokalnej pochodnej.

**Jednostki.** W latencie nie ma automatycznie metrów, niutonów, kelwinów ani dżuli. Współczynniki opisujemy jako „latent na podkrok”, „piksele na jednostkę perturbacji”, „latent² na aktualizację”, „aktualizacje”. Wszystkie użycia optycznych wzorów poniżej są mapowaniem matematycznym z podanymi założeniami.

**Notacja spektralna.** W ciągłym czasie stosujemy `tilde f(ω)=∫f(t)exp(iωt)dt`, a pobudzenie sinusoidalne ma postać `u(t)=Re[u_ω exp(-iωt)]`. Dwustronne PSD `S_x(ω)=∫C_x(t)exp(iωt)dt`, więc `Var(x)=(1/2π)∫S_x(ω)dω`. Dla dyskretnych danych `Ω=ωΔ∈[-π,π]` i `Var(x)=(1/2π)∫S_d(Ω)dΩ`. Tej konwencji nie mieszać z jednostronnym PSD w Hz.

Wzory optyczne zapisano w jednolitej notacji, nie jako dosłowne kopie typografii artykułów. Numery równań dotyczą wskazanych wersji źródeł; adaptacje do ViT oraz propozycje zakresów danych oznaczono jako własne.

## 1. Tensor lokalnego pola: aktywna kalibracja inspirowana FORMA

**Źródło.** Pérez García i in. (2018), sekcja Results: uogólnienie 2D, równania (5)–(10). Metoda FORMA dopasowuje liniowe pole i rozdziela części symetryczną oraz antysymetryczną [R1]. Dowód tekstowy: „the stiffnesses along these axes correspond to the eigenvalues with a minus sign”.

**Wzór optyczny.** Dla ruchu przeciążonego tłumieniem:
`dx = (F_0+J_F x)dt/γ + sqrt(2D)dW`.
Przy polu harmonicznym `F=-Kx`, zatem `J_F=-K`. Część `K_s=-sym(J_F)` opisuje lokalną odpowiedź zachowawczą; `skew(J_F)` opisuje lokalną niesymetrię. W 2D rotację liniową można parametryzować `Ω_rot=(J_21-J_12)/(2γ)`. To nie jest automatycznie prąd prawdopodobieństwa.

**Adaptacja dokładna.** Dla bloku `T(x)=x+F(x)`:
`F(x_0+δ)≈F_0+Jδ`,
`Jv≈[F(x_0+εv)-F(x_0-εv)]/(2ε)`.
Lokalnie dopasowując odpytywane pole, z kolumnami macierzy V równymi δ_r i kolumnami Y równymi `F(x_0+δ_r)-F_0`, mamy `J_hat=YV^T(VV^T)^-1` przy pełnym rzędzie. Intercept należy zachować, jeśli `F_0≠0`; nie dopasowywać sztucznej pułapki ze środkiem w bieżącym tokenie.

Dla prawdziwego kroku używać `M=I+J` i
`||Mδ||²-||δ||²=δ^T(2 sym J+J^T J)δ`.
„Sztywność” `-sym J` jest tu umowną macierzą lokalnej odpowiedzi w jednostce na podkrok; nie siłą na metr.

**Protokół.** Kierunki v: prawoskrętne wektory singularne H_l dla bbox/logitu, kierunki ekspansji M, kierunki kontrolne losowe i kierunki LN. Użyć kilku ε i sprawdzić zbieżność różnic centralnych. Pytanie badawcze: czy lokalna ekspansja występuje w kierunkach zmieniających bbox, czy w kierunkach nieczytanych przez głowę?

**Minimum danych.** Jeden checkpoint i obraz wystarczają do deterministycznej kalibracji punktowej. Pełne odtworzenie d-wymiarowej macierzy z odpytań potrzebuje d niezależnych kierunków; różnice centralne to 2d wywołań plus baseline. Autodiff/JVP redukuje koszt wybranych kierunków. Uogólnienie wymaga wielu obrazów, tokenów narzędzia i tła oraz checkpointów.

**Pułapka identyfikacyjna / prior ML.** 192D pole pojedynczego tokenu z resztą zamrożoną jest lokalnym eksperymentem interwencyjnym, nie pełnym Jacobianem uwagi. W ML analogiem są Jacobian sensitivity, local linearization i adversarial directions. Wkład publikacji powinien być walidowanym protokołem i związkiem z bbox, nie odkryciem samego Jacobianu.

## 2. Odpowiedź impulsowa i propagacja perturbacji do bbox

**Źródło.** Toyabe i in. (2007), równanie (2) definiuje odpowiedź na małą siłę; eksperyment w Fig. 3–4 mierzy relaksację po skoku środka pułapki [R2]. Dla skalarnej pułapki `γ dx/dt=-kx+u(t)`:
`R_x(t)=γ^-1 exp(-t/τ)1_(t≥0)`, `τ=γ/k`;
dla skoku `u(t)=u_0 1_(t≥0)`:
`<x(t)>=u_0/k [1-exp(-t/τ)]`.

**Adaptacja do głębokości — własne wyprowadzenie.** Jeśli perturbacja δ zostaje dodana do pełnego stanu X_l PRZED zastosowaniem podkroku l:
`δq≈Dq(X_L) M_(L-1)...M_l B_l δ`.
`M_r=DT_r(X_r)`; `B_l` umieszcza perturbację w wybranym tokenie/podprzestrzeni. Jest to odpowiedź skończonej, nieautonomicznej sekwencji bloków, nie wykładnicza relaksacja w czasie fizycznym. Cały iloczyn najlepiej mierzyć JVP lub dwoma pełnymi forward; nie materializować macierzy 115200×115200.

**Protokół.** Dla każdej krawędzi i logitu mierzyć
`r_l(v;ε)=[q(X_l+εB_lv)-q(X_l-εB_lv)]/(2ε)`.
Porównać z liniową prognozą. Zarejestrować odpowiedź na każdym dalszym podkroku, aby ustalić, który blok tłumi, obraca lub wzmacnia błąd. Test uszkodzenia i naprawy: zaburzenie pogarszające bbox, potem przeciwne zaburzenie w dalszym bloku o tej samej normie.

**Co mówi o zadaniu.** Ujawnia, czy sieć aktywnie koryguje błędy lokalizacji narzędzia; które tokeny przenoszą informację o lewej/prawej krawędzi; czy logit pozostaje stabilny mimo niestabilnej ramki. Ocenia także lokalną sterowalność: macierz R ma zwykle znacznie niższy rząd niż latent.

**Jednostki i minimum.** Piksele/jednostkę latentu i logit/jednostkę latentu. Jedna para ±ε jest pojedynczym pomiarem kierunkowym; kilka amplitud i niezależnych obrazów testuje liniowość i przenoszenie. Normalizować kierunki przed porównaniem.

**Pułapka / prior ML.** Poke podczas odczytu checkpointu nie zmienia kolejnych wag i nie jest odpowiedzią procesu uczenia. Prior ML: activation patching, causal tracing i Jacobian attribution. Wartość dodana to ilościowa kalibracja amplitudy, pełna sekwencja bloków i przewidywanie błędów bbox.

## 3. Susceptibility sinusoidalna: amplituda, faza i pamięć

**Źródła.** Paul, Kundu i Banerjee (2019), sekcja Theory, równania (1)–(4), oraz Kumar i in. (2021), Theoretical background i modulacja wieloczęstotliwościowa [R3, R4]. Wersja z pamięcią:
`m xddot(t)+∫_0^∞Γ(s)xdot(t-s)ds+kx(t)=u(t)+ξ(t)`.
Dla naszej konwencji:
`χ_x(ω)=1/[k-mω²-iω Γ_tilde(ω)]`.
Bez bezwładności i pamięci `χ_x=(k-iωγ)^-1`.
Amplituda wynosi `|u_ω||χ_x|`; faza przy wybranej konwencji wynika z argumentu χ, a odpowiedź w zwykłej osi czasu jest opóźniona względem wymuszenia.

**Adaptacja do uczenia.** Najpierw określić rzeczywiste wejście u_n: np. jawny, podpisany dodatek do aktualizacji wag w znanym kierunku albo perturbację celu pomocniczego. Pełnym stanem jest `s_n=(θ_n,m_n,v_n,...)`, a nie sam token. Lokalny model:
`δs_(n+1)=Aδs_n+B_in u_n+η_n`, `δq_n=Hδs_n`.
Przy sinusoidzie `exp(-iΩn)`:
`χ_q(Ω)=H(e^(-iΩ)I-A)^-1 B_in`.
To własne przepisanie liniowego modelu stanu; nie zakłada kąpieli termicznej.

**Estymator.** Po usunięciu transjentu, dla wejścia o znanych częstotliwościach i replikowanych kontrolach można szacować `χ_hat=S_qu/S_uu`, z orientacją iloczynu sprzężonego wybraną tak, by `q=χu`. Sprawdzać koherencję `|S_qu|²/(S_qq S_uu)`, zależność odpowiedzi od amplitudy i powtórzeń. Sygnał wieloczęstotliwościowy musi ograniczać maksymalną łączną amplitudę.

**Bbox i dane.** Otrzymujemy częstotliwości, na których ramka silnie reaguje na zmianę celu lub update, oraz opóźnienie tej reakcji. To rozróżnia szybkie ustawianie geometrii bbox od wolnej zmiany pewności. Potrzebne zapis co aktualizację, baseline, wspólne losowania batchy w parach ±u i kilka niezależnych par. Zalecenie projektowe, nie uniwersalny próg: kilkanaście próbek na najkrótszy okres i wiele okresów w oknie krótszym od znaczącego dryfu parametrów.

**Pułapka / prior ML.** Sześć różnych bloków nie jest stacjonarnym szeregiem dla FFT. Widmo po głębokości ma inne znaczenie. Prior ML: linear system identification, control, temporal influence; specyficzna nowość musi dotyczyć identyfikowalnego wejścia i odczytu detekcji.

## 4. OU, ACF, MSD i PSD jako jeden wspólny test, z dokładnym próbkowaniem

**Źródła.** Pérez-García i in. (2023), sekcja 2 „Problems due finite sampling frequency and integration time”, równanie Langevina (1), sekcje ACF/MSD/PSD/FORMA; Paul i in. (2021), Theory, transition density i autoregressive representation [R5, R6].

**Wzory.** Dla `dx=-λ(x-μ)dt+sqrt(2D)dW`, `λ>0`:
`V=D/λ`,
`C(τ)=V exp(-λ|τ|)`,
`MSD(τ)=2V[1-exp(-λτ)]`,
`S_x(ω)=2D/(λ²+ω²)`.
Nie są to cztery niezależne dowody: dla tego modelu zawierają tę samą informację drugiego rzędu.

Dla odstępów `Δ_n=t_(n+1)-t_n`:
`x_(n+1)|x_n ~ N( μ+a_n(x_n-μ), Q_n )`,
`a_n=exp(-λΔ_n)`,
`Q_n=V(1-a_n²)`.
Minimalizować sumę `0.5[log Q_n+e_n²/Q_n]` z prawdziwymi odstępami. Dla stałego Δ dokładne dyskretne PSD:
`S_d(Ω)=Q/[1+a²-2a cos Ω]`.
Dla macierzy B w `dy=Bydt+sqrt(2D)dW`:
`A_Δ=exp(BΔ)`,
`Q_Δ=∫_0^Δ exp(Bs)2D exp(B^T s)ds`,
`BΣ+ΣB^T+2D=0`.
To rozdziela dynamikę od wariancji; nie zakłada `K=Σ^-1`.

**Protokół / bbox.** Dopasować kilka jednowymiarowych odczytów: logit, błąd każdej krawędzi, zadaniowe kierunki latentu. Wolny trend estymować wspólnie jako latentny stan lub zewnętrzny model; porównać z trendem+IID i ARMA. Prognozować rozkład zmiany bbox po 10, 100 i 500 aktualizacjach, a następnie sprawdzić pokrycie przedziałów na nieużytych fragmentach.

**Minimum i jednostki.** λ: 1/aktualizację, D: piksel²/aktualizację lub latent²/aktualizację, τ=1/λ. Nie ma uniwersalnego minimalnego N. Trzeba obserwować relaksację oraz wiele jej czasów; gdy `a≈0`, λ jest słabo identyfikowalne niezależnie od jakości dopasowania histogramu. Przy d wymiarach liczba parametrów rośnie jak d², więc 192D wymaga struktury lub wielu replik.

**Pułapki / prior ML.** Filtr zmienia model obserwacji. Dla średniej prostokątnej o szerokości w transmisja to `sinc(ωw/2)`; PSD mnoży się przez kwadrat modułu i aliasuje po próbkowaniu. Nie stosować tych samych wzorów do wygładzonych i surowych danych. Antykorelacja przyrostów nie wystarcza: IID daje −1/2. Prior ML: local linear state-space, predictive uncertainty, dynamika SGD; przewaga polega na wspólnej walidacji i kontroli czasu.

## 5. Allan variance wyników: kiedy uśrednianie pomaga, a kiedy ukrywa dryf

**Źródło.** Czerwinski, Richardson i Oddershede (2009), sekcja „Allan Variance”, Definition, równanie (2); Andersson i in. (2011), abstrakt o optymalnym czasie kalibracji [R7, R8]. Cytat: „half the averaged squared mean of neighboring intervals”.

Dla równomiernego próbkowania i `τ=mΔ`:
`bar q_k=(1/m)sum_(i=0)^(m-1) q_(km+i)`,
`σ_A²(τ)=0.5 E[(bar q_(k+1)-bar q_k)²]`.
Wielowymiarowa wersja własna:
`C_A(τ)=0.5 E[Δbar q Δbar q^T]`.
Jej diagonalne elementy mierzą stabilność poszczególnych krawędzi/logitu, a elementy poza diagonalą współdryf. Wersja overlapping wykorzystuje wszystkie przesunięcia okna, lecz powstałe składniki nie są niezależnymi replikami.

**Protokół.** Liczyć na surowych wynikach stałych obrazów i stałych tokenów. Dla nierównego czasu tworzyć okna o równej długości w aktualizacjach z jawnie określoną wagą czasową; nie interpolować bez modelu i traktować siatki jako prawdziwych obserwacji. Porównać wczesne, środkowe i późne uczenie oraz narzędzie/tło. Wskazać minimum i przedziały ufności albo jawnie zgłosić brak minimum.

**Wiedza o bbox.** Skala, na której maleją fluktuacje, ale nie dominuje jeszcze zmiana reprezentacji, może służyć do wyboru okna EMA/ensemblingu oraz długości okna do lokalnej identyfikacji. Współdryf obu krawędzi oznacza zmianę środka; przeciwne zmiany — szerokości. Można więc oddzielić drganie pewności od przesuwania ramki.

**Ważne rozróżnienie.** Minimum czasu uśredniania dotyczy mieszanki szumu i dryfu; idealny OU nie daje takiego minimum. Z całkowania jego kowariancji wynika (wyprowadzenie własne) `σ_A²(τ)=V[2a-3+4exp(-a)-exp(-2a)]/a²`, gdzie `a=λτ`. Ta krzywa ma **maksimum** przy `a≈1.89`, po czym maleje. Nie mylić maksimum idealnego OU z minimum pomiaru zawierającego wolny dryf.

**Minimum i jednostki.** Minimum definicyjne to dwie sąsiednie średnie; nie wystarcza ono do wiarygodności. Praktyczne zalecenie: raportować duże τ tylko z wieloma parami okien i bootstrapem blokowym; przykładowy ostrożny zakres `τ≤T/20` daje około 20 niepokrywających się okien. Wynik w piksel² lub logit², Allan deviation w pikselach lub logitach.

**Pułapka / prior ML.** Minimum nie dowodzi pułapki i nie daje temperatury. Zmiana wariancji Allana może wynikać z dryfu głowy. Prior ML: analizy stabilności checkpointów, EMA i SWA; funkcja jest metrologiczna, a nie nowy fizyczny mechanizm.

## 6. Wirtualna pułapka jako kontrolowany test sterowalności bbox

**Źródła.** Jun i Bechhoefer (2012), II.B, równania (15)–(19); Kumar i Bechhoefer (2018), równania (1a)–(1c), Fig. 3; Gavrilov, Jun i Bechhoefer (2014), real-time calibration [R9–R11].

**Wzory wzorcowe.** Bez opóźnienia:
`x_(n+1)=(1-α)x_n+ξ_n`, `Var ξ=2DΔ`,
`Var x=2DΔ/[α(2-α)]`, stabilnie dla `0<α<2`.
Przy jednym kroku opóźnienia:
`x_(n+1)=x_n-αx_(n-1)+ξ_n`,
`Var x=2DΔ(1+α)/[α(1-α)(2+α)]`, stabilnie dla `0<α<1`.
`α=kΔ/γ`. W skończonym kroku nawet ten prosty feedback nie spełnia naiwnie `Var x=k_BT/k`.

**Adaptacja do ViT: interwencja, nie opis istniejącego treningu.** Wybrać checkpoint, token oraz osiągalny skończony cel `q*`. Pomocniczy koszt:
`U(X)=0.5(q(X)-q*)^T W(q(X)-q*)+0.5 λ||X-X_0||²`.
Korekta w latencie:
`X_next=X-η_c ∇_X U(X)`,
`∇_X U=H^T W(q-q*)+λ(X-X_0)`.
Wielokrotne takie kroki tworzą nowy kontroler latentu. Nie wolno prezentować ich jako naturalnej trajektorii forward/AdamW. W treningu trzeba zdefiniować osobny dodatkowy koszt i porównać odgałęzienia z identycznego checkpointu.

**Bbox.** Test pyta: jak duża i w jakich kierunkach korekta reprezentacji jest potrzebna, aby skorygować lewą krawędź bez zmiany pozostałych? Czy obniża to błąd dla innych obrazów, czy tylko nadpisuje wynik jednego? Przy liniowym H rozwiązanie minimalnej normy:
`δX*=H^T(HH^T+λI)^-1δq*`.
To jest regularizowana korekta odczytu, nie globalny cel uczenia.

**Minimum.** Pełne stany przed/po, zadany gain i opóźnienie, log kontrolera, małe i większe cele, baseline bez korekcji. Dla mapowania stabilności sweep gain oraz delay. Jednostki U są umowne po określeniu W; nie nazywać ich dżulami. Logit celu musi być skończony.

**Pułapka / prior ML.** Punkty poza manifoldem treningowym, saturacja LayerNorm i sigmoid, przeciek GT oraz optymalizacja tylko jednej ramki. Prior ML: activation steering, feature optimization, control i regularized inverse problems. Dowodem przydatności będzie generalizacja korekty i ograniczenie kosztów ubocznych.

## 7. FDT i Harada–Sasa: test warunkowy, bez przypisywania AdamW temperatury

**Źródła.** Harada i Sasa (2005), główna równość; Toyabe i in. (2007), równania (1)–(2) i definicja wymuszenia [R12, R2]. Oryginalny Harada–Sasa odczytano tylko jako abstrakt; równania potwierdzono w pełnym tekście Toyabe.

**Wzór fizyczny.** Dla `γxdot=F+u+ξ`, `<ξ(t)ξ(s)>=2γk_BTδ(t-s)`, stacjonarnie:
`S_x(ω)=2k_BT Im χ_x(ω)/ω` w równowadze.
Dla prędkości `R_v(ω)=-iωχ_x(ω)`,
`S_v(ω)=2k_BT Re R_v(ω)`.
Z centralnym kowariancyjnym PSD prędkości:
`<J>=γ{vbar²+(1/2π)∫[S_v(ω)-2k_BT Re R_v(ω)]dω}`.
Toyabe zapisuje wersję z pełną korelacją; jej wkład DC zawiera średnią prędkość. Nie dodawać vbar² dwukrotnie.

**Wymuszenie musi być sprzężone.** Ogólna relacja equilibrium dla obserwabli A oraz perturbacji energii `U_h=U_0-h(t)B_obs` wiąże odpowiedź A na h z korelacją A i B_obs. B_obs jest obserwablą sprzężoną z h; nie jest dowolnym kierunkiem tokenu. Przykładowy skalar x musi otrzymywać właśnie siłę sprzężoną do x.

**Co wolno w ViT.** Najpierw zmierzyć empiryczną odpowiedź `R_(q←u)` na dokładnie zdefiniowane u działające na pełny proces optymalizatora oraz spontaniczne fluktuacje TEGO SAMEGO procesu. Sprawdzić, czy prosta lokalna równowagowa hipoteza liniowa potrafi wyjaśnić oba pomiary. Poke latentu w eval działa na inną mapę niż szum batchy i nie dostarcza odpowiedzi dla training-time FDT.

**Wiedza o bbox.** Niezgodność pasywnej niepewności i aktywnej podatności pokaże, że małe wahania bbox nie znaczą dużej odporności na zmianę danych/aktualizacji. To użyteczny wynik jako rozbieżność response–fluctuation. Wyrażenie fizyczne nie daje w obecnym modelu mocy cieplnej w watach.

**Dane i pułapki.** Potrzebne: zdefiniowane mobilność/sprzężenie, gęste spontaniczne trajektorie, wiele odpowiedzi ±u, stacjonarne okno i kontrola całego pasma częstotliwości. Rzadkie checkpointy nie kontrolują wysokoczęstotliwościowego wkładu całki. Brak równowagowej kąpieli, anizotropia, pamięć AdamW i ukryte stopnie swobody ograniczają interpretację. Prior ML: fluctuation–response i noisy optimization już istnieją; samo nazwanie rozbieżności „dyssypacją” nie stanowi odkrycia.

## 8. Stochastic force inference i test pamięci zamiast założenia OU

**Źródła.** Frishman i Ronceray (2020), sekcja „Stochastic force inference”, równania (4)–(6), oraz część o niejednorodnej dyfuzji, równania (9)–(10) [R13]. Paul i in. (2021) pokazują powrót do opisu Markowa po dodaniu zmiennej ukrytej dla modelu z pamięcią [R6]. Dowód tekstowy SFI: „the information in the trajectory should exceed the number of inferred parameters”.

**Model.** `dy=b(y)dt+sqrt(2D(y))dW` dla małego zestawu obserwabli zadania y. Przy stałym D i empirycznie ortonormalnych funkcjach `c_α`, lokalny estymator Itô:
`b_hat_(iα)=(1/T)sum_n Δy_(n,i)c_α(y_n)`,
`b_hat_i(y)=sum_α b_hat_(iα)c_α(y)`.
Dla nieortonormalnej bazy trzeba uwzględnić macierz Grama. Ocena dyfuzji w granicy małego Δ:
`D_hat(y)≈E[(Δy-b_hatΔ)(Δy-b_hatΔ)^T | y]/(2Δ)`.
Dokładne poprawki na szum pomiaru i niejednorodność dyfuzji wymagają wariantów ze źródła; prostego wzoru nie przedstawiać jako odpornego na dowolne błędy.

Przy stałym D stacjonarny prąd:
`j(y)=b(y)p(y)-D∇p(y)`,
`v(y)=j(y)/p(y)`.
Dla zmiennego D konwencji Itô:
`j_i=b_i p-∂_j(D_ij p)`.
Jeżeli spełniony jest model dyfuzyjny, czasowo nieodwracalna miara ma postać
`dot S=∫j^T D^-1 j/p dy`
w jednostce 1/czas; utożsamienie z termodynamicznym ciepłem wymaga dodatkowych fizycznych założeń.

**Test pamięci.** Porównać predykcję b(y_n) z modelem
`y_(n+1)=sum_(k=0)^p A_k y_(n-k)+B_in u_n+ε_n`.
Skończona suma pamięci jest kontrolowaną alternatywą; liczbę lagów wybierać poza próbką, nie na podstawie samego dopasowania. Oceniać autokorelację innowacji, warunkową zależność od przeszłości, predykcję wielokrokową i kalibrację niepewności. Zmienne pomocnicze mogą reprezentować ukryte momenty optymalizatora, ale ich interpretacja wymaga weryfikacji.

**BBox.** W y umieścić błąd środka, log szerokości, log wysokości i logit lub kilka kierunków H. Pole dryfu może pokazać, czy uczenie systematycznie poprawia geometrię ramki, czy ją nadmiernie koryguje, i czy zmiany pewności poprzedzają korektę kształtu. Mierzyć też korelacje tokenów: wszystkie korzystają ze wspólnych wag.

**Minimum / prior ML.** Gęste, długie lokalnie stacjonarne trajektorie i niezależne replikacje, mała baza o pełnym rzędzie oraz liczba efektywnych obserwacji znacznie większa od liczby współczynników. Praca SFI wiąże błąd z informacją trajektorii, nie tylko liczbą klatek. 192D z pełną bazą kwadratową jest nieadekwatne dla 806 checkpointów. Prior ML: SINDy/stochastic identification, state-space i Mori–Zwanzig; nowość musi wynikać z walidacji w task-aligned observables i przewidywania efektów interwencji.

## Wspólny plan falsyfikacji i oceny jakości

To propozycja dalszego eksperymentu, nie wykonane badanie:
1. Zarezerwować obrazy i trajektorie do oceny poza próbką, zachować grupowanie według filmu; wiele tokenów z jednej ramki nie jest niezależnym zbiorem.
2. Dla metod 1–2 testować zgodność JVP z faktyczną odpowiedzią w całej sieci i zakres liniowości; baseline: losowe kierunki o równej normie, same gradienty głowy i normy aktywacji.
3. Dla metod 4–5 pokazać surowe szeregi, harmonogram próbkowania i wyniki identycznych analiz dla trend+IID, random walk, znanego OU oraz procesu z pamięcią.
4. Dla 3, 6–8 uruchamiać oddzielne odgałęzienia wspólnego checkpointu. Replikować batch schedule i augmentacje w parach interwencja/kontrola; osobno powtórzyć na niezależnych seedach.
5. Nie dopasowywać wszystkich narzędzi do jednego tokenu 226. Równolegle sprawdzić środek narzędzia, jego krawędzie, trudne negatywy i tło.
6. Oddzielić latent, głowę i dekoder: sprawdzać także kontrfaktyczny odczyt z zamrożoną głową, aby poprawy outputu nie przypisać wyłącznie ruchowi latentu.

## References

[R1] Pérez García, L., Donlucas Pérez, J., Volpe, G., Arzola, A. V., & Volpe, G. (2018). High-performance reconstruction of microscopic force fields from Brownian trajectories. *Nature Communications*. https://doi.org/10.1038/s41467-018-07437-x. Nowy odczyt: Results, znaki 8000–16000.

[R2] Toyabe, S., Jiang, H.-R., Nakamura, T., et al. (2007). Experimental test of a new equality: Measuring heat dissipation in an optically driven colloidal system. *Physical Review E, 75*, 011122. https://doi.org/10.1103/PhysRevE.75.011122. Pełny tekst Scite oraz równania potwierdzone w https://arxiv.org/pdf/cond-mat/0608415, s. 1–2.

[R3] Paul, S., Kundu, A., & Banerjee, A. (2019). Single-shot phase-sensitive wideband active microrheology of viscoelastic fluids using pulse-scanned optical tweezers. *Journal of Physics: Condensed Matter*. https://doi.org/10.1088/1361-648X/ab32f3. Pełny tekst, Introduction/Theory.

[R4] Kumar, R., Vitali, V., Wiedemann, T., et al. (2021). Multi-frequency passive and active microrheology with optical tweezers. *Scientific Reports*. https://doi.org/10.1038/s41598-021-93130-x. Pełny tekst, Theoretical background.

[R5] Pérez-García, L., Selin, M., Ciarlo, A., et al. (2023). Optimal calibration of optical tweezers with arbitrary integration time and sampling frequencies: a general framework [Invited]. *Biomedical Optics Express*. https://doi.org/10.1364/BOE.495468. Pełny tekst, sekcja 2.

[R6] Paul, S., Narinder, N., Banerjee, A., et al. (2021). Bayesian inference of the viscoelastic properties of a Jeffrey’s fluid using optical tweezers. *Scientific Reports*. https://doi.org/10.1038/s41598-021-81094-x. Pełny tekst, Theory. Metadane narzędzia podają autora jako „N Narinder”; sprawdzić pełną pisownię przy finalnej bibliografii manuskryptu.

[R7] Czerwinski, F., Richardson, A. C., & Oddershede, L. B. (2009). Quantifying noise in optical tweezers by Allan variance. *Optics Express, 17*, 13255. https://doi.org/10.1364/OE.17.013255. Dwie strony tekstu Scite, sekcja Allan Variance.

[R8] Andersson, M., Czerwinski, F., & Oddershede, L. B. (2011). Optimizing active and passive calibration of optical tweezers. *Journal of Optics, 13*, 044020. https://doi.org/10.1088/2040-8978/13/4/044020. Wyłącznie abstrakt; nie przypisano niedostępnych wzorów.

[R9] Jun, Y., & Bechhoefer, J. (2012). Virtual potentials for feedback traps. *Physical Review E, 86*, 061106. https://doi.org/10.1103/PhysRevE.86.061106. Pełny tekst Scite i równania (15)–(19) potwierdzone w https://arxiv.org/pdf/1212.4588, s. 3–4.

[R10] Kumar, A., & Bechhoefer, J. (2018). Nanoscale virtual potentials using optical tweezers. *Applied Physics Letters*. https://doi.org/10.1063/1.5055580. Dwie strony pełnego tekstu; równania feedback potwierdzone w https://arxiv.org/pdf/1809.02896, s. 2–3.

[R11] Gavrilov, M., Jun, Y., & Bechhoefer, J. (2014). Real-time calibration of a feedback trap. *Review of Scientific Instruments*. https://doi.org/10.1063/1.4894383. Fragmenty pełnego tekstu z wyszukiwania.

[R12] Harada, T., & Sasa, S.-i. (2005). Equality connecting energy dissipation with a violation of the fluctuation-response relation. *Physical Review Letters, 95*, 130602. https://doi.org/10.1103/PhysRevLett.95.130602. Odczyt zwrócił tylko abstrakt; wzory sprawdzone przez [R2].

[R13] Frishman, A., & Ronceray, P. (2020). Learning force fields from stochastic trajectories. *Physical Review X, 10*, 021009. https://doi.org/10.1103/PhysRevX.10.021009. Pełny tekst, Stochastic force inference, równania (4)–(6).

## Status źródeł i reprodukowalność wyszukiwania

Wykonano 17 nowych kwerend merytorycznych Scite (w tym celowany zbiorczy odczyt metadanych), 4 kwerendy redakcyjne i 15 wywołań read_fulltext. 13 odpowiedzi read_fulltext zawiera rzeczywisty tekst, 2 wyłącznie abstrakty. 13 stron tekstu pochodzi z 10 unikalnych DOI; dwukrotnie odczytano kolejne strony Allan variance, feedback 2012 i feedback 2018.

Przeszukano aktywną kalibrację, response functions, Allan variance, Harada–Sasa, virtual feedback, stiffness tensors, finite sampling, SFI, memory kernels, spectra z aktywnym wymuszeniem, efekty delay oraz nieregularnie próbkowane OU. Wszystkie odpowiedzi zapisano w tym katalogu. Część szerokich wyszukiwań zwracała odległe tematy; nie są one dowodami dla metod atlasu.

Dla 13 DOI z bibliografii sprawdzono filtry retraction/correction/erratum/concern: zero trafień. Brak informacji w indeksie nie stanowi gwarancji braku uwag redakcyjnych. Surowe metadane zwykle nie zwracały pola editorialNotices. Eksport bibliografii znajduje się w bibliography.bib; zachowano też JSON eksportu.

Źródła Scite często usuwają równania z tekstu ciągłego. Dlatego tam, gdzie było to istotne dla znaków, mianowników i konwencji PSD, odczytano również jawnie dostępne wersje arXiv. Nie traktowano pustego miejsca po równaniu jako zweryfikowanego wzoru.

