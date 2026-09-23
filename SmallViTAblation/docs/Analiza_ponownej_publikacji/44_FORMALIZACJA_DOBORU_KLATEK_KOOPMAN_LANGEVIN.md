# 44. Dobór klatek przez przewidywaną poprawę detektora: sformułowanie i dowody

Data: 2026-09-23. Autor opracowania: Codex, na polecenie autora projektu.
Status: wyprowadzenia matematyczne z jawnymi założeniami; kontrola algebraiczna SymPy oraz częściowa formalizacja sprawdzona w Lean. Bez nowych treningów. Nie jest to dowód przewagi obecnej implementacji RF-DETR. Dokładny zakres weryfikacji Lean jest w sekcji 14.

**Główny wynik:** można zagwarantować poprawę po wybranej partii klatek, jeżeli przewidywana poprawa straty przewyższa ograniczenie błędu tego przewidywania. Można również zagwarantować przewagę nad inną partią, porównując dolne ograniczenie poprawy pierwszej z górnym ograniczeniem poprawy drugiej. Koopman i Langevin mogą dostarczać przybliżeń potrzebnych do takiego obliczenia. Samo widmo, pokrycie modów ani małe jądro pamięci nie wystarczają do gwarancji.

Dokument koryguje wskazane w sekcji 2 twierdzenia dokumentów 42 i 43. Historyczne pomiary pozostają pomiarami; ich interpretacje nie stają się założeniami dowodów.

## 1. Podstawa lokalna i zakres istniejących dowodów

Wykorzystane pliki projektu ViT:

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\43_KRYTERIUM_DOMKNIECIA_TOZSAMOSCI_KOOPMAN_LANGEVIN.md`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_klatek_stan_sieci\token_wybierz.py` oraz `WYNIKI_PILOT.md` z tego katalogu.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_cech_adaptacyjny\glister.py`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\zajetosc_tozsamosci.py` oraz `wyniki\REPORT_maly_budzet.json`.

Wykorzystane pliki matematyczne projektu Grant:

- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\MoriZwanzig.lean`: `projekcja_rodzi_pamiec`, `reszta_pelna_eq_schur`, `historia_nie_szkodzi`, `mz_zero_iff_warunkowa`, `kontrprzyklad_D59`.
- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\MzLogdetMacierz.lean`: `matrix_mz_zero_iff_conditional_covariance_zero`.
- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\OuKoopman.lean`: rekurencje AR(1), własność półgrupy, gramiany i stacjonarna wariancja.

Odczytano treść tych dowodów. Nie uruchamiano ponownie ich kompilacji. Twierdzenia Lean powyżej są algebraiczne: same nie dowodzą istnienia procesu o zadanych kowariancjach, gaussowskości danych, poprawności etykiet ani związku z AP detektora. Interpretacja probabilistyczna wymaga dodatkowych założeń. Komentarz przy twierdzeniu nie ma statusu formalnie sprawdzonej tezy.

Zewnętrzny punkt odniesienia: [GLISTER, AAAI 2021](https://cdn.aaai.org/ojs/16988/16988-13-20482-1-2-20210518.pdf) formułuje wybór podzbioru przez jakość na zbiorze sterującym; jego twierdzenie 2 także wymaga zgodności kierunków gradientów i ograniczenia kroku. Związek aproksymacji Koopmana z pamięcią opisuje [Lin i współautorzy, Data-driven learning for the Mori–Zwanzig formalism](https://arxiv.org/abs/2101.05873). Poniższe dowody zapisano jawnie, zamiast przenosić gwarancje tych prac na RF-DETR.

## 2. Korekty, które są konieczne przed formalizacją

1. **Zerowe jądro pamięci nie jest równoważne pełnemu opisowi stanu.** Może pozostać wymuszenie od ukrytego stanu. Kontrprzykład jest w sekcji 6 i w lokalnym `MoriZwanzig.lean`.
2. **Trwałość nie jest tożsamością narzędzia.** Stałe tło i stała funkcja też mają wartość własną równą jedności. Trzeba sprawdzić informację o zadaniu na odrębnych filmach.
3. **Część rzeczywista nie ma uprzywilejowanego znaczenia semantycznego.** Faza zespolonego wektora własnego jest dowolna. Zmiana fazy miesza części rzeczywistą i urojoną bez zmiany operatora.
4. **Kwadrat i moduł nie są zawsze niezmiennikami.** Dla rzeczywistego modu kwadrat skaluje się przez kwadrat wartości własnej. Moduł zespolonego modu skaluje się przez moduł wartości własnej.
5. **Biała reszta względem czasu nie dowodzi nieusuwalności błędu detekcji.** Reszta może zależeć od pominiętych cech obrazu lub etykiety, mimo zerowej autokorelacji.
6. **Związek fluktuacji i dyssypacji nie rozstrzyga sam, czy zabrakło danych.** Wymaga określonej projekcji, miary i dynamiki ortogonalnej. Dowolne reszty regresji nie muszą spełniać tej samej tożsamości.
7. **Brak pokrycia latentu nie jest jeszcze brakiem wiedzy potrzebnej do wykrywania.** Potrzebny jest związek odległości w tym latencie z błędem albo gradientem zadania.
8. **R² odczytu proporcji prostokąta nie oznacza procentu kompletności przestrzeni stanu.** Nie dowodzi też odczytu kąta przestrzennego, zwrotu końcówki ani tożsamości konkretnego narzędzia.

Nie wnioskujemy również, że liniowo niedekodowalna cecha nie może być odtworzona nieliniowo, że obserwowane poprawki błędów są uniwersalnym sufitem uczenia ani że obrót bazy między checkpointami sam dowodzi utraty informacji.

## 3. Zadanie, dane, czas i koszt

### 3.1. Rozdzielenie danych po operacjach

Niech \(v\) oznacza operację, \(t\) czas filmu, \(X_{v,t}\) obraz, a \(Y_{v,t}\) zbiór poprawnych klas i prostokątów. Pulę kandydatów oznaczamy przez \(U\). Dopuszczamy dobór pojedynczych klatek lub grup klatek. Wszystkie zbiory oddzielamy po operacjach:

- \(U\): dane do treningu i wyboru kandydatów;
- \(D\): dane sterujące wyborem, zawierające poprawne oznaczenia;
- \(E\): dane do końcowej oceny procedury, niewykorzystywane w selekcji;
- opcjonalna odrębna próba do kalibracji błędów przybliżeń.

Zbiór \(D\) staje się częścią procedury uczenia, gdy używamy go wielokrotnie do wyboru. Nie jest wtedy niezależnym końcowym sprawdzianem. Jeżeli czytamy etykiety kandydatów w \(U\), badamy selekcję z oznaczonego zbioru. Oszczędność anotacji wymaga innego protokołu.

Definiujemy ryzyko na nowych operacjach:

\[
R(\theta)=\mathbb E_{v\sim P_{\rm oper}}\left[\mathbb E_{t\sim P(t\mid v)}
\ell(f_\theta(X_{v,t}),Y_{v,t})\right]. \tag{1}
\]

Wagi operacji i rozkład czasu trzeba ustalić z góry. Średnia po wszystkich klatkach nadaje dłuższym filmom większą wagę niż średnia po operacjach. \(\ell\) jest stratą zadania, np. ustaloną sumą strat klasyfikacji i lokalizacji. AP jest osobną miarą; spadek \(\ell\) bez dodatkowego twierdzenia nie gwarantuje wzrostu AP.

### 3.2. Dwa różne zegary

Indeks \(t\) dotyczy kolejnych obrazów przy **zamrożonych** wagach detektora. Indeks \(k\) dotyczy kolejnych aktualizacji wag podczas uczenia. Operator Koopmana dopasowany po \(t\) nie jest automatycznie modelem zmian wag po \(k\).

Pełny stan treningu to \(s_k=(\theta_k,o_k)\), gdzie \(o_k\) obejmuje pamięć optymalizatora, harmonogram i potrzebne bufory. Dla partii \(S_k\) i losowości augmentacji \(\xi_k\):

\[
s_{k+1}=\mathcal A(s_k,S_k,\xi_k),\qquad
S_k=\pi_k(\mathcal I_k), \tag{2}
\]

gdzie \(\mathcal I_k\) zawiera tylko informacje dozwolone przez protokół. To jest ścisły sens sterowania doborem klatek: decyzją sterującą jest \(S_k\), a stanem sterowanego układu jest stan treningu. Ruch kamery jest innym wejściem, w modelu filmu.

### 3.3. Dwa cele optymalizacyjne

Przy budżecie czasu \(T\):

\[
\min_\pi\;\mathbb E R(\theta_{N_\pi(T)}),\qquad
N_\pi(T)=\max\{n:C_{\rm init}+\sum_{k<n}(C_{\rm select,k}+C_{\rm train,k}+C_{\rm monitor,k})\le T\}. \tag{3}
\]

Przy ustalonej tolerancji błędu \(r_0\) można zamiast tego minimalizować koszt dojścia do \(R(\theta_k)\le r_0\). Do kosztu początkowego należą checkpoint pomocniczy L0, ekstrakcja tokenów i dopasowanie modeli pomocniczych. Mniejszy zbiór końcowego treningu nie usuwa wykorzystania wcześniejszych oznaczeń przy budowie L0.

Dla jednorazowego wyboru klasyczny zapis dwupoziomowy ma postać
\(\min_{S:|S|=b}V(\theta_K(S))\), gdzie \(V\) jest stratą na \(D\), a \(\theta_K(S)\) wynikiem dokładnie określonych \(K\) kroków treningu. Użycie rzeczywistego algorytmu \(\mathcal A\), zamiast niejednoznacznego globalnego \(\arg\min\) straty sieci, uwzględnia inicjalizację i optymalizator.

## 4. Co mierzy Koopman

### 4.1. Obserwable i dopasowanie operatora

Dla zamrożonego \(\theta\) token \(z_{v,t,p}\) opisuje miejsce \(p\) siatki obrazu, ale jego cechy mogą zależeć od szerszego kontekstu. Token nie jest automatycznie śledzonym punktem metalu. Prostokąt wycięcia zawiera też tło. Z tokenów budujemy obserwable, czyli mierzalne funkcje danych:

\[
\psi_{v,t}=\Psi_\omega(X_{v,t},z_{v,t},b_{v,t})\in\mathbb R^d. \tag{4}
\]

\(b\) musi mieć określone pochodzenie: prawdziwe oznaczenie lub predykcja. \(\Psi_\omega\) może być PCA, zbiorem funkcji nieliniowych albo uczonym enkoderem deep Koopman. Wymiar jest parametrem aproksymacji, nie liczbą stopni swobody narzędzia.

Dla wierszowych próbek \(\psi_t^\top\) dopasowanie EDMD ma postać

\[
\widehat K=\arg\min_K\sum_{(v,t)\in F}
\|\psi_{v,t+\Delta}^{\top}-\psi_{v,t}^{\top}K\|^2+\lambda_{\rm reg}\|K\|_F^2. \tag{5}
\]

Przy \(\widehat K u_j=\lambda_j u_j\), \(\phi_j(x)=\psi(x)^\top u_j\). Empiryczne residuum wynosi
\(r_{j,t}=\phi_j(X_{t+\Delta})-\lambda_j\phi_j(X_t)\).
Skalę normalizujemy na zbiorze dopasowania, a residua oceniamy na odrębnych filmach. Trzeba raportować wariancję, błąd oraz kontrolę stałej funkcji, żeby nie nagradzać zaniku reprezentacji.

Dla pełnego stanu Markowa \(H_t\) operator populacyjny to
\((\mathcal K_\Delta f)(h)=\mathbb E[f(H_{t+\Delta})\mid H_t=h]\).
Gdy dysponujemy jedynie niepełnym \(z_t\), regresja przyszłości na \(z_t\) nadal istnieje, ale kolejne regresje nie muszą tworzyć półgrupy. Przy nieregularnych odstępach klatek stosujemy stałe \(\Delta\), model zależny od \(\Delta\) albo jawny generator czasu. Nie interpretujemy jednego operatora z pomieszanych odstępów jako jednego fizycznego kroku.

Ruch kamery można warunkować wejściem \(u_t\), np. \(\widehat\psi_{t+1}=K\psi_t+B u_t+\sum_j u_{t,j}K_j\psi_t\). Należy również uwzględnić pominięty ruch narzędzia i jakość estymacji kamery. Odjęcie oszacowanego ruchu nie dowodzi usunięcia całego wpływu kamery. Dopasowanie z danych obserwacyjnych nie dowodzi reakcji na dowolną interwencję.

### 4.1a. Jawny opis przez cząstki

W tym projekcie podstawowym zapisem klatki jest pole cząstek

\[
\mathcal Z_{v,t}=(z_{v,t,p})_{p\in\mathcal P},\qquad z_{v,t,p}\in\mathbb R^d,
\]

gdzie \(\mathcal P\) jest zbiorem miejsc siatki, a każda cząstka zachowuje własny wektor cech. Wymiar kanałów można zmniejszyć wspólnym, zamrożonym przekształceniem bez usuwania indeksu \(p\). Redukcja kanałów i uśrednienie cząstek to różne operacje. W eksperymencie `dobor_klatek_stan_sieci` funkcja wczytująca wręcz odrzuca zapis z wcześniejszym uśrednianiem przestrzennym.

Wariant rejestratora `patch_particles.py` dodatkowo zapisuje profil względem końcowej normalizacji warstwy, według teorii DiagnosticTool. Pilot `dobor_klatek_stan_sieci` używa ostatnich surowych tokenów ViT. To dwie konwencje cech cząstki; ich współrzędnych i błędów nie wolno mieszać bez przeliczenia.

Ogólny model liniowy z pamięcią dla całego pola może mieć postać

\[
z_{t+1,p}=\sum_q A_{pq}z_{t,q}
+\sum_{\ell=1}^{L_m}\sum_q M_{\ell,pq}z_{t-\ell,q}
+B_pu_t+\xi_{t,p}.
\]

Zapis dopuszcza wpływ innych cząstek. Nie zakłada ich niezależności ani fizycznego oddziaływania punktów metalu. W nieliniowym opisie analogicznie modelujemy obserwable \(\psi_p\), a błąd skończonego przybliżenia wchodzi do ograniczeń z sekcji 8.5. Konkretne uproszczenie tego równania w kodzie jest osobnym wyborem modelu.

Możemy zachować całe pole albo zbudować wybrany opis klatki z wagami \(w_p\), np. \(a=\sum_p w_p z_p\). Dla tych samych wag i przy \(\|z_p-\widetilde z_p\|\le e_p\):

\[
\left\|\sum_p w_p z_p-\sum_p w_p\widetilde z_p\right\|
\le\sum_p|w_p|e_p.
\]

To wynika z liniowości sumy i nierówności trójkąta. Dostarcza błędu opisu klatki do dalszego oszacowania gradientu. Nie dowodzi, że średnia zachowuje informację o położeniu. Dla detekcji można zachować wszystkie tokeny albo dołączyć informację przestrzenną. Jeżeli zmieniają się także wagi lub maska, potrzebny jest dodatkowy składnik błędu wynikający z ich zmiany.

Indeks \(p\) oznacza tutaj stałe miejsce obrazu. Aby analizować cząstkę podążającą za narzędziem, trzeba osobno określić trajektorię \(p_t\), sposób odczytu cech między komórkami i błąd śledzenia. Same równania Koopmana nie ustanawiają tej korespondencji. Pozostaje więc dokładnie pomysł opisu latentu przez cząstki, ale bez nieudowodnionego utożsamiania cząstki siatki z punktem obiektu.

### 4.2. Twierdzenie T1: dokładne znaczenie trwałości

Z definicji residuum:

\[
\phi_{t+1}-\phi_t=(\lambda-1)\phi_t+r_t,
\qquad
\|\phi_{t+1}-\phi_t\|_{L^2}
\le |\lambda-1|\|\phi_t\|_{L^2}+\|r_t\|_{L^2}. \tag{6}
\]

**Dowód:** odejmujemy \(\phi_t\) i stosujemy nierówność trójkąta. To dowodzi małej zmiany obserwabli, kiedy oba składniki są małe. Nie dowodzi, czego obserwabla dotyczy.

Przez iteracyjne podstawienie otrzymujemy również

\[
\phi_{t+h}-\lambda^h\phi_t
=\sum_{j=0}^{h-1}\lambda^{h-1-j}r_{t+j}. \tag{7}
\]

W deterministycznym przypadku z zerowym residuum \(\lambda=1\) oznacza niezmienność na trajektorii. W stochastycznym przypadku równanie \(\mathcal K\phi=\phi\) oznacza niezmienność **warunkowej średniej**; realizacje mogą się zmieniać. Trzeba kontrolować także wariancję innowacji. Mały błąd jednokrokowy nie daje automatycznie małego błędu długoterminowego, szczególnie przy \(|\lambda|>1\).

Dla dokładnej deterministycznej funkcji własnej:

\[
|\phi_{t+1}|=|\lambda||\phi_t|,\qquad
(\phi_{t+1})^2=\lambda^2\phi_t^2\quad\text{dla rzeczywistych }\lambda,\phi. \tag{8}
\]

Zatem kwadrat jest niezmienny dla \(\lambda=\pm1\), a moduł dla \(|\lambda|=1\), poza przypadkiem zerowej funkcji. Dla \(\lambda=-1/2\) kwadrat maleje czterokrotnie. W procesie losowym dodatkowa wariancja uniemożliwia ogólne przenoszenie (8) na warunkowe średnie.

Jeżeli \(\phi=a+ib\), to \(e^{i\alpha}\phi\) jest tą samą funkcją własną z inną fazą:
\(\operatorname{Re}(e^{i\alpha}\phi)=a\cos\alpha-b\sin\alpha\).
Badamy więc całą płaszczyznę \(\operatorname{span}(a,b)\), nie uprzywilejowujemy osi rzeczywistej. Dla pary sprzężonej wystarczy jeden wektor i jego części Re/Im. Dwie części Re z pary sprzężonej mogą być tą samą osią. Znak wektora rzeczywistego też jest dowolny; znak wartości własnej to odrębna własność dynamiki.

Symetryzacja problemu własnego wymusza rzeczywiste widmo. Nie jest dowodem braku obrotów w danych. Sam wybór wektorów własnych dopasowanego \(\widehat K\), a następnie sprawdzanie ich niezmienniczości względem tego samego \(\widehat K\), jest tautologią algebraiczną. Potrzebna jest ocena funkcji i ich następców na nowych trajektoriach.

### 4.2a. Niezmienność względem ruchu a lokalizacja

Jeżeli \(T_g\) jest określonym przekształceniem obrazu, niezmiennik spełnia \(\psi(T_gx)=\psi(x)\). Cecha zmieniająca się zgodnie z ruchem spełnia natomiast \(p(T_gx)=\rho(g)p(x)\), gdzie \(\rho(g)\) opisuje odpowiednią zmianę współrzędnych. To dwa różne wymagania.

**Lemat:** jeżeli cały opis detektora jest niezmienny względem przesunięcia, a jego wyjście zależy tylko od tego opisu, nie może on poprawnie lokalizować wszystkich przesuniętych kopii obiektu. Dowód: identyczny opis daje identyczne wyjście głowicy, natomiast poprawny prostokąt po niezerowym przesunięciu jest inny. Zatem trwałe cechy mogą wspierać rozpoznanie, ale detektor potrzebuje również współrzędnych pozwalających odtworzyć położenie.

Parowanie tokenów o tej samej względnej pozycji w dwóch prostokątach nie ustanawia samo działania grupy \(T_g\): może zmienić się perspektywa, zawartość tła lub fragment metalu. Wymaga to kontroli korespondencji. Także rozkład na tożsamość, położenie i tło nie wynika automatycznie z diagonalizacji operatora.

### 4.3. Deep Koopman

Możliwy cel uczenia opisu to

\[
\min_{\omega,K,D}\sum_t\|\Psi_\omega(x_{t+1})-K\Psi_\omega(x_t)\|^2
+\beta\|D(\Psi_\omega(x_t))-a(x_t)\|^2+\gamma\mathcal R(\Psi_\omega). \tag{9}
\]

Tu \(a(x)\) jest jawnie określonym celem rekonstrukcji lub zadania, a \(\mathcal R\) ograniczeniem degeneracji. Sama pierwsza suma ma rozwiązanie \(\Psi\equiv0\). Rekonstrukcja całego obrazu może premiować tło. Deep Koopman powiększa klasę możliwych obserwabli, lecz nie usuwa potrzeby sprawdzenia ich związku z narzędziem i gradientem zadania.

## 5. Co znaczy, że modelowi czegoś brakuje

### 5.1. Deficyt informacji o zadaniu

Dla zamrożonej reprezentacji \(Z=F_\theta(X)\) definiujemy

\[
\Delta_{\rm zad}(Z)=\inf_h\mathbb E\ell(h(Z),Y)-\inf_f\mathbb E\ell(f(X),Y)\ge0. \tag{10}
\]

Infima są po wszystkich dopuszczalnych mierzalnych predyktorach o skończonym ryzyku; predyktory z \(Z\) muszą być zawarte w klasie predyktorów z \(X\). Nierówność wynika z tego zawierania. Przy klasach ograniczonych, np. wyłącznie liniowych, wynik zależy również od ograniczenia odczytu i trzeba zmienić nazwę mierzonej wielkości.

**Twierdzenie T2.** Dla straty kwadratowej i \(Y\in L^2\):

\[
\Delta_{\rm zad}(Z)=\mathbb E\|\mathbb E[Y\mid X]-\mathbb E[Y\mid Z]\|^2. \tag{11}
\]

**Dowód:** optymalnymi predyktorami są warunkowe średnie. Rozkładamy
\(Y-\mathbb E[Y\mid Z]=(Y-\mathbb E[Y\mid X])+(\mathbb E[Y\mid X]-\mathbb E[Y\mid Z])\).
Iloczyn mieszany ma zerową średnią, bo pierwszy składnik jest ortogonalny do funkcji \(X\). Po podniesieniu normy do kwadratu odejmujemy ryzyko optymalnego predyktora z \(X\).

Dla skończonego zbioru klas i log loss, przy wszystkich rozkładach predykcyjnych i skończonych entropiach, analogiczną luką jest \(H(Y\mid Z)-H(Y\mid X)=I(Y;X\mid Z)\). Nie utożsamiamy tej tożsamości z całym RF-DETR, który ma także lokalizację i dopasowanie obiektów.

Do \(\Delta_{\rm zad}\) dochodzą ograniczenia architektury odczytu, estymacji ze skończonej próby i optymalizacji. Z samej dynamiki \(Z\) nie znamy żadnej z tych wielkości. „Domknięcie tożsamości” zastępujemy tu określonym pytaniem o informację potrzebną do oznaczonego zadania.

Założenie \(Z=F_\theta(X)\) jest istotne. Wycięcie prawdziwego prostokąta z użyciem \(Y\) daje opis uprzywilejowany informacją z oznaczenia. Taki pomiar jest dopuszczalny w diagnostyce i selekcji oznaczonych danych, ale nie dowodzi, że detektor sam znalazł narzędzie. Nie podstawiamy \(Z=F(X,Y)\) do (10) jako funkcji samego obrazu.

Także poprawa odczytu dzięki historii nie oznacza, że model pojedynczej klatki może ją całą przyswoić. Jeżeli historia zawiera informację o \(Y\), której nie ma w \(X\), minimalne ryzyko modelu korzystającego tylko z \(X\) nadal ogranicza drugi składnik definicji (10). Przykładem jest końcówka całkowicie zasłonięta na bieżącej klatce, widoczna wcześniej. Sam dobór klatek nie dostarczy modelowi w czasie użycia brakującego wejścia.

### 5.2. Dlaczego średnia tokenów może tracić zwrot

Jeżeli dwa przypadki różnią się wyłącznie permutacją tego samego zestawu wektorów tokenów, ich średnia jest identyczna. To dokładny fakt algebraiczny. Nie dowodzi on, że średnia rzeczywistych tokenów nigdy nie zawiera zwrotu, ponieważ tokeny mogą już kodować pozycję i kontekst.

Momenty pozycyjne \(\sum_p w_p z_p\), miękka maska i odczyt geometrii są hipotezami sposobu zmniejszenia (10). Ich skuteczność trzeba zmierzyć przy oznaczeniach definiujących tę geometrię. Nie wynika ona z samego dodania kolejnej współrzędnej.

## 6. Pamięć, wymuszenie i szum: dokładne równanie

### 6.0. Dokładna tożsamość operatorowa, także dla układu nieliniowego

Niech \(\mathcal K\) będzie ograniczonym operatorem Koopmana na wybranej przestrzeni funkcji, a \(P\) ograniczoną projekcją na rozpiętość obserwabli. Oznaczmy \(Q=I-P\). Dla każdego \(n\ge0\):

\[
\mathcal K^{n+1}
=\sum_{j=0}^{n}\mathcal K^{n-j}P\mathcal K(Q\mathcal K)^j
+(Q\mathcal K)^{n+1}.
\]

**Dowód:** podstawiamy \(\mathcal K=P\mathcal K+Q\mathcal K\) do ostatniego nierozłożonego czynnika od lewej przed ciągiem \((Q\mathcal K)^j\). Każde podstawienie wyodrębnia następny składnik sumy; po ostatnim pozostaje \((Q\mathcal K)^{n+1}\). Równoważnie jest to indukcja bez zamieniania kolejności operatorów.

Dla wektora liniowo niezależnych obserwabli \(g\) tworzących bazę obrazu \(P\), zapiszmy \(P\mathcal K(Q\mathcal K)^j g=\Omega_j g\). W przypadku deterministycznej ewolucji otrzymujemy po obliczeniu funkcji w stanie początkowym:

\[
g(H_{n+1})=\Omega_0g(H_n)+\sum_{j=1}^n\Omega_jg(H_{n-j})+W_n(H_0),
\qquad W_n=(Q\mathcal K)^{n+1}g.
\]

To dokładny dyskretny rozkład z pamięcią dla ustalonej projekcji. Jeżeli \(P\) jest ortogonalna w zadanym iloczynie skalarnym, \(PW_n=0\). Nie oznacza to niezależności kolejnych \(W_n\) ani ich niezależności od wszystkich późniejszych obserwacji. Dla stochastycznego operatora warunkowej średniej tożsamość nadal dotyczy funkcji operatorowych, ale \(\mathcal K^ng(H_0)\) jest średnią warunkową, a nie każdą realizacją \(g(H_n)\). Opis realizacji wymaga także innowacji procesu lub rozszerzenia stanu o losowe wejścia.

Domknięcie przestrzeni obserwabli ma ścisły warunek \(Q\mathcal KP=0\). Wtedy dla \(g\in\operatorname{Ran}P\) znikają zarówno pamięć, jak i \(W_n\). Odwrotnie, zniknięcie \(W_0=Q\mathcal Kg\) dla wszystkich takich \(g\) daje ten warunek. **Samo** zniknięcie \(P\mathcal K(Q\mathcal K)^jP\) dla \(j\ge1\) go nie daje: projekcja \(P\) może ukryć niezerową dynamikę w dopełnieniu.

Skończeniewymiarowy EDMD, dobór opóźnień i model reszt w naszym kodzie są estymatorami wybranych składników; dokładność powyższej tożsamości nie dowodzi dokładności tych estymatorów. Uczenie reprezentacji zmienia również projekcję i jej reszty. Poniższy przykład blokowy pokazuje wszystkie składniki jawnie, bez odwołania do nieobserwowalnego operatora.

### 6.1. Twierdzenie T3: eliminacja nieobserwowanych współrzędnych

Rozpatrujemy jawny układ liniowy z obserwowanym \(c_t\) i ukrytym \(h_t\):

\[
\begin{bmatrix}c_{t+1}\\h_{t+1}\end{bmatrix}
=\begin{bmatrix}A&B\\C&D_h\end{bmatrix}
\begin{bmatrix}c_t\\h_t\end{bmatrix}. \tag{12}
\]

Dla każdego \(t\ge0\) zachodzi

\[
c_{t+1}=A c_t+\sum_{j=0}^{t-1}B D_h^j C\,c_{t-1-j}+B D_h^t h_0. \tag{13}
\]

**Dowód:** indukcja daje \(h_t=D_h^t h_0+\sum_{j=0}^{t-1}D_h^j Cc_{t-1-j}\). Podstawiamy do pierwszego wiersza (12). Dla \(t=0\) suma jest pusta. Macierze \(B D_h^j C\) tworzą jądro pamięci. Ostatni składnik jest wymuszeniem od początkowego stanu ukrytego; nie musi być niezależnym białym szumem.

Warunek \(B=0\) wystarcza, aby zniknęły pamięć i wymuszenie. Jest też konieczny dla autonomii \(c_{t+1}=A c_t\) dla **wszystkich** \(c_0,h_0\): wystarczy w (12) wziąć \(c_0=0\) i dowolne \(h_0\).

Zerowanie samych \(BD_h^j C\) nie wymusza \(B=0\). Kontrprzykład:

\[
A=1/2,\quad B=1,\quad C=0,\quad D_h=1/3.
\qquad c_{t+1}=\tfrac12c_t+3^{-t}h_0. \tag{14}
\]

Całe jądro jest zerowe, lecz \(c_0\) nie wyznacza \(c_1\). Ten sam obserwowany stan z różnymi \(h_0\) ma różne przyszłości. Stwierdzenie z dokumentu 43 „jądra znikają wtedy i tylko wtedy, gdy pierwszy blok jest zerowy” jest więc fałszywe bez dodatkowych założeń.

Trzeba też rozróżnić niezmienniczość podprzestrzeni **wektorów stanu** i **funkcji obserwowanych**. W (12) podprzestrzeń stanów \(h=0\) jest niezmiennicza, gdy \(C=0\). Funkcje zależne liniowo tylko od \(c\) są domknięte pod działaniem operatora Koopmana, gdy \(B=0\). Zamiana tych warunków gubi transpozycję operatora.

Jeżeli \(\|D_h\|\le\rho<1\) i \(\|c_s\|\le M\), pominięcie składników pamięci o indeksach \(j\ge L_m\) daje ograniczenie

\[
\left\|\sum_{j=L_m}^{t-1}BD_h^j Cc_{t-1-j}\right\|
\le \|B\|\|C\|M\frac{\rho^{L_m}}{1-\rho}. \tag{15}
\]

To wynika z podmultiplikatywności normy i szeregu geometrycznego. Sama stabilność wartości własnych nie oznacza \(\|D_h\|<1\) w dowolnej normie; przy układzie nienormalnym potrzebne jest odpowiednie ograniczenie \(\|D_h^j\|\). Dodatkowy szum wejściowy w (12) tworzy dodatkową sumę wymuszeń, której nie wolno pominąć.

### 6.2. Kontrprzykład lokalny i jego zakres

W `MoriZwanzig.lean` współczynnik dodatkowego opóźnienia zeruje się dla kowariancji
\(\gamma_0=1,\gamma_1=r,\gamma_2=r^2\), lecz przy \(\gamma_3=0\) reszty regresji mają kowariancję \(r^4\) na drugim opóźnieniu. Formalny plik dowodzi tych tożsamości dla liczb rzeczywistych.

Aby otrzymać poprawny przykład stacjonarny, można wybrać \(r=1/4\) oraz \(\gamma_j=0\) dla \(|j|>2\). Wielomian widmowy jest dodatni:
\(1+2r\cos\omega+2r^2\cos2\omega\ge1-2r-2r^2=3/8\).
Zatem ciąg jest dodatnio określony i może być kowariancją stacjonarnego procesu gaussowskiego. To uzupełnia interpretację probabilistyczną; nie przypisujemy dowodowi Lean założeń, których nie zawiera.

### 6.3. Twierdzenie T4: ile naprawdę daje historia

Niech \(Q\in L^2\) będzie **zdefiniowanym celem predykcji**. Niech \(\mathcal G\subseteq\mathcal H\) opisują dostępną informację: bieżący stan i bieżący stan wraz z historią. Wtedy

\[
\mathbb E\|Q-\mathbb E[Q\mid\mathcal G]\|^2
=\mathbb E\|Q-\mathbb E[Q\mid\mathcal H]\|^2
+\underbrace{\mathbb E\|\mathbb E[Q\mid\mathcal H]-\mathbb E[Q\mid\mathcal G]\|^2}_{\Delta_{\rm hist}(Q)\ge0}. \tag{16}
\]

**Dowód:** rozkładamy błąd na innowację względem \(\mathcal H\) i różnicę warunkowych średnich. Iloczyn mieszany znika po warunkowaniu na \(\mathcal H\).

Jeżeli \(Q=Z_{t+1}\), to (16) dotyczy przewidywania latentu. Jeżeli \(Q=Y_t\), dotyczy informacji o oznaczeniu. Jeżeli \(Q\) jest gradientem detektora, dotyczy przewidywania aktualizacji treningu. Duża pierwsza luka nie dowodzi dużej drugiej ani trzeciej. Równość zerowa mówi o średniej warunkowej, nie o całym rozkładzie i nie o własności Markowa dla dowolnego horyzontu.

Dla regresji liniowej, po usunięciu bieżącego stanu, niech \(S\) będzie kowariancją reszt celu, \(P\) kowariancją reszt historii, a \(B_c\) kowariancją krzyżową. Przy dodatniej określoności \(P,S,S-B_cP^{-1}B_c^\top\):

\[
\Sigma_{Q\mid Z,H}^{\rm lin}=S-B_cP^{-1}B_c^\top,\qquad
M_Q^{\rm lin}=\tfrac12\log\frac{\det S}{\det(S-B_cP^{-1}B_c^\top)}. \tag{17}
\]

Lokalny `MzLogdetMacierz.lean` dowodzi \(M_Q^{\rm lin}=0\iff B_c=0\) przy tych założeniach. Przy wspólnej gaussowskości jest to także warunkowa niezależność odpowiednich wektorów. Bez gaussowskości to brak dodatkowej **liniowej** przewidywalności. Nie dowodzi braku zależności nieliniowej.

W estymacji z danych historia może pogorszyć wynik przez przeuczenie, mimo nieujemności populacyjnej (16). Potrzebne są odrębne filmy i kontrola złożoności. Norma współczynników jądra zależy także od skali i bazy współrzędnych; nie porównujemy jej między checkpointami bez wspólnej normalizacji. Wygodniejsza interpretacyjnie jest poprawa przewidywania tego samego celu na odłożonych danych.

## 7. Od klatki do zmiany błędu detektora

### 7.1. Twierdzenie T5: rzeczywista poprawa po aktualizacji

Niech \(V\) będzie różniczkowalną stratą docelową, np. na \(D\). Zakładamy, że jej gradient jest \(L\)-Lipschitz na wszystkich odcinkach rozważanych aktualizacji. Dla \(\eta>0\), aktualizacji \(\theta^+=\theta-\eta d\) i \(g=\nabla V(\theta)\) oznaczamy poprawę przez \(G(d)=V(\theta)-V(\theta^+)\). Wtedy

\[
\left|G(d)-\eta\langle g,d\rangle\right|
\le\frac{L\eta^2}{2}\|d\|^2. \tag{18}
\]

**Dowód:** całkujemy gradient po odcinku:
\(V(\theta-\eta d)-V(\theta)=-\eta\int_0^1\langle\nabla V(\theta-s\eta d),d\rangle ds\).
Odejmujemy \(-\eta\langle g,d\rangle\), stosujemy Cauchy'ego–Schwarza i ograniczenie Lipschitza. Całka z \(s\) daje czynnik \(1/2\).

W szczególności

\[
\eta\langle g,d\rangle-\tfrac12L\eta^2\|d\|^2>0
\quad\Longrightarrow\quad V(\theta^+)<V(\theta). \tag{19}
\]

Wielkość \(d\) jest **faktycznie zastosowanym kierunkiem**. Dla SGD to gradient partii. Dla AdamW obejmuje pamięć momentów, normalizację i weight decay. Można zawsze zdefiniować \(d=(\theta-\theta^+)/\eta\) z wirtualnego rzeczywistego kroku. Nie wolno automatycznie podstawiać gradientu SGD za kierunek AdamW.

W RF-DETR gładkość może zawodzić przy zmianie przyporządkowania obiektów, punktach niegładkich i operacjach progowych. Wtedy (18) ma zastosowanie tylko tam, gdzie założenie zachodzi, lub do jawnie wygładzonego celu. Alternatywą jest bezpośrednie obliczenie zmiany \(V\) po wirtualnym kroku i cofnięcie całego stanu. To ocena tego kroku na \(D\), nie dowód generalizacji.

### 7.2. Twierdzenie T6: gwarancja z niedokładnych przewidywań

Załóżmy, że umiemy ograniczyć błędy:

\[
\|g-\widehat g\|\le\varepsilon,\qquad
\|d_S-\widehat d_S\|\le\delta_S. \tag{20}
\]

Definiujemy

\[
\widehat a_S=\eta\langle\widehat g,\widehat d_S\rangle,\qquad
E_S=\eta\left[\varepsilon\|\widehat d_S\|+(\|\widehat g\|+\varepsilon)\delta_S\right]
+\frac{L\eta^2}{2}(\|\widehat d_S\|+\delta_S)^2. \tag{21}
\]

Wtedy

\[
G(d_S)\in[\widehat a_S-E_S,\widehat a_S+E_S]. \tag{22}
\]

**Dowód:** rozpisujemy
\(\langle g,d_S\rangle-\langle\widehat g,\widehat d_S\rangle
=\langle g-\widehat g,\widehat d_S\rangle+\langle g,d_S-\widehat d_S\rangle\).
Z (20) jego moduł jest nie większy niż \(\varepsilon\|\widehat d_S\|+(\|\widehat g\|+\varepsilon)\delta_S\). Ponadto \(\|d_S\|\le\|\widehat d_S\|+\delta_S\). Stosujemy (18).

**Wniosek o poprawie:** wystarczy \(\widehat a_S-E_S>0\).

**Wniosek o przewadze nad partią B:** przy tym samym stanie treningu i kroku wystarczy

\[
\boxed{\widehat a_S-E_S>\widehat a_B+E_B.} \tag{23}
\]

Wówczas \(V(\theta-\eta d_S)<V(\theta-\eta d_B)\). Nie wystarczy porównać samych dolnych ograniczeń. Dla losowej partii \(B\) można porównać dolne ograniczenie \(S\) z wartością oczekiwaną górnego ograniczenia \(B\), jeżeli ograniczenia są ważne dla wszystkich realizacji i istnieją odpowiednie momenty. Daje to przewagę nad oczekiwanym wynikiem jednego losowego kroku.

Jeżeli \(g\) jest dokładnie policzonym gradientem \(D\), błąd \(\varepsilon\) wynosi zero dla celu \(V_D\). Zastąpienie go gradientem ryzyka populacyjnego wymaga osobnego ograniczenia błędu generalizacji gradientu. RMSE modelu przewidującego gradient nie jest automatycznie jednostajnym \(\delta_S\). Wybieranie maksimum spośród wielu kandydatów wymaga gwarancji jednoczesnej albo jawnego uwzględnienia błędu selekcji.

Wniosek (23) dotyczy wspólnego punktu startowego. Przewaga lokalna nad losowaniem nie dowodzi przewagi dwóch całych treningów, których wagi po pierwszym kroku są już różne.

### 7.3. Gwarancja wyboru najlepszej przewidywanej partii

Jeśli (22) zachodzi jednocześnie dla rodziny \(\mathcal S\), \(E_S\le E_{\max}\), a \(\widehat S\) maksymalizuje \(\widehat a_S\), to

\[
G(d_{\widehat S})\ge\max_{S\in\mathcal S}G(d_S)-2E_{\max}. \tag{24}
\]

**Dowód:** dla rzeczywiście najlepszej \(S^*\),
\(G(d_{\widehat S})\ge\widehat a_{\widehat S}-E_{\max}\ge\widehat a_{S^*}-E_{\max}\ge G(d_{S^*})-2E_{\max}\).
Jeżeli algorytm tylko przybliża maksimum z błędem \(\zeta\), prawa strona zmniejsza się dodatkowo o \(\zeta\). Ograniczenie dotyczy tylko przeszukanej rodziny; odsianie dobrej klatki wcześniej może usunąć ją z \(\mathcal S\).

To jest konkretny cel teorii: zmniejszać koszt wyznaczenia \(\widehat a_S\), kontrolując jednocześnie \(E_S\) i utratę kandydatów. Stabilny mod może być użyteczny, jeżeli pomaga w tym celu.

## 8. Gdzie dokładnie wchodzą Koopman i Langevin

### 8.1. Zmienne do przewidywania korzyści

Dla kandydatki \(i\) tworzymy opis

\[
\chi_i=(\psi_i,\widehat\psi_i^{\rm hist},\psi_i-\widehat\psi_i^{\rm hist},
\text{geometria, wejścia ruchu, jakość predykcji, niepewność}). \tag{25}
\]

Model \(q_\omega(\chi_i,Y_i,s_k)\) przewiduje gradient \(g_i\), kierunek aktualizacji albo bezpośrednio poprawę \(G\). Jeżeli korzysta z \(Y_i\), jest to dobór oznaczonych danych. Przy braku \(Y_i\) błąd przypuszczalnej etykiety staje się dodatkowym źródłem \(\delta\). Walidacja modelu \(q\) musi rozdzielać filmy i stany treningu, dla których chce się twierdzić, że działa.

Koopman daje przewidywalną część opisu, pamięć dostarcza dodatkowego kontekstu, a reszty wskazują niepewność przewidywania. Żaden z tych składników sam nie mówi, jaki będzie znak \(\langle g,d_S\rangle\). Potrzebny jest związek z zadaniem, opisany przez \(q\) i ograniczenie jego błędu.

### 8.2. Twierdzenie T7: jawny związek błędu cech z błędem gradientu

Dla zamrożonych cech rozpatrzmy liniowy odczyt \(W\) i stratę \(\ell(W;z,y)=\frac12\|Wz-y\|^2\). Wtedy
\(g_W(z,y)=(Wz-y)z^\top\).
Załóżmy \(\|W\|_2\le M\), \(\|z\|,\|\widetilde z\|\le R_z\), \(\|y\|\le Y_{\max}\) i \(\|z-\widetilde z\|\le e\). Zachodzi

\[
\|g_W(z,y)-g_W(\widetilde z,y)\|_F
\le (2MR_z+Y_{\max})e. \tag{26}
\]

**Dowód:** różnicę zapisujemy jako
\(W(z-\widetilde z)z^\top+(W\widetilde z-y)(z-\widetilde z)^\top\).
Norma Frobeniusa iloczynu zewnętrznego jest iloczynem norm wektorów. Stosujemy nierówność trójkąta i podane ograniczenia.

Jeżeli model Koopmana z pamięcią przewiduje \(\widetilde z_i\) z ograniczonym błędem \(e_i\), średni gradient wybranej partii ma błąd nie większy niż
\(\delta_S\le(2MR_z+Y_{\max})|S|^{-1}\sum_{i\in S}e_i\).
To konkretny most do (21), dla **tego odczytu, tych etykiet i kroku SGD**. Przy aproksymowaniu etykiety przez \(\widetilde y\) dochodzi składnik \(R_z\|y-\widetilde y\|\), po zastosowaniu (26) i osobnej różnicy po etykiecie.

Mniejsze błędy \(e_i\) zwężają przedział (22). Nie oznacza to, że kandydaci z największym residuum są najlepsi: duże \(e_i\) zwiększa niepewność gwarancji. Kandydaci trudni mogą wymagać dokładnego obliczenia gradientu zamiast zaufania przybliżeniu.

Dla całego detektora potrzebne są dodatkowo ograniczenia zmian pochodnych enkodera i strat, dopasowania obiektów oraz przekształcenia gradientu przez optymalizator. Twierdzenia (26) nie przypisujemy obecnemu RF-DETR bez tych warunków.

### 8.3. Twierdzenie T8: kiedy pokrycie pomaga

Niech klatka \(i\) ma opis \(\chi_i\) oraz gradient \(g_i\). Załóżmy **zadaniową** regularność
\(\|g_i-g_j\|\le L_g d(\chi_i,\chi_j)\).
Przypiszmy każdą klatkę do reprezentanta \(r(i)\in S\) i nadajmy reprezentantowi wagę \(w_s=|\{i:r(i)=s\}|/|U|\). Wówczas

\[
\left\|\frac1{|U|}\sum_{i\in U}g_i-\sum_{s\in S}w_sg_s\right\|
\le\frac{L_g}{|U|}\sum_{i\in U}d(\chi_i,\chi_{r(i)}). \tag{27}
\]

**Dowód:** ważona suma reprezentantów jest równa \(|U|^{-1}\sum_i g_{r(i)}\). Odejmujemy sumy, stosujemy nierówność trójkąta i założenie.

Takie pokrycie przybliża gradient pełnej puli. Nie dowodzi przewagi nad nim ani nad losowaniem na nowych operacjach. Pokrycie samych trwałych modów nie zapewnia założenia o gradientach: może pomijać pozycję, rozmiar i etykietę. Brak wag \(w_s\) zmienia proporcje danych, nawet przy idealnym pokryciu.

Kontrprzykład wag: dziewięć punktów z gradientem \(+1\) i jeden z \(-1\). Pełna średnia to \(0{,}8\). Dwa idealne reprezentanty bez wag dają średnią zero. Wagi \(0{,}9\) i \(0{,}1\) odtwarzają ją dokładnie.

### 8.4. Twierdzenie T9: samo widmo nie wyznacza użyteczności klatki

Weźmy zadanie skalarne z \(V(w)=\frac12(w+1)^2\) i \(w=0\). Dwie kandydatki mają jednakowy, idealnie stały opis Koopmana \(z=1\), zerową pamięć i zerowy błąd przewidywania. Ich straty treningowe to odpowiednio \(\ell_+(w)=\frac12(w+1)^2\) i \(\ell_-(w)=\frac12(w-1)^2\). Gradienty wynoszą \(+1\) i \(-1\).

Dla \(0<\eta<2\) pierwszy krok poprawia \(V\) o \(\eta-\eta^2/2\). Drugi pogarsza ją o \(\eta+\eta^2/2\). Wszystkie podane statystyki dynamiczne są identyczne.

**Wniosek:** żadna reguła korzystająca wyłącznie z tych statystyk nie może mieć gwarancji poprawnego wyboru dla wszystkich dopuszczalnych zadań. Etykiety lub sprawdzony związek opisu z gradientem rozwiązują nierozróżnialność. Kontrprzykład nie dowodzi bezużyteczności Koopmana; wskazuje brakującą przesłankę.

Podobnie biały składnik latentu może nieść całe zadanie: niech \(Y_t\) będzie niezależnym znakiem losowym, a \(Z_t=Y_t\). Reszta przewidywania z przeszłości jest biała i nieprzewidywalna, ale bieżąca etykieta jest odczytywana bezbłędnie z \(Z_t\). Usunięcie „szumu” usunęłoby informację potrzebną do zadania.

### 8.5. Wniosek T7a: pełny łańcuch od równania pamięci do poprawy

Pokażmy jawnie, skąd może pochodzić \(e_i\) w (26). W liniowym modelu (12) oznaczmy \(K_j=BD_h^jC\), \(f_t=BD_h^t h_0\). Predyktor z ograniczoną historią jest równy

\[
\widetilde c_{t+1}=\widehat A c_t+
\sum_{j=0}^{\min(L_m,t)-1}\widehat K_j c_{t-1-j}+\widehat f_t. \tag{36}
\]

Zakładamy \(\|c_s\|\le M_c\), \(\|A-\widehat A\|\le a_e\), \(\|K_j-\widehat K_j\|\le k_{e,j}\), \(\|f_t-\widehat f_t\|\le f_{e,t}\), \(\|D_h\|\le\rho<1\). Z odejmowania (13) i (36) oraz (15):

\[
\|c_{t+1}-\widetilde c_{t+1}\|
\le M_c\left(a_e+\sum_{j=0}^{\min(L_m,t)-1}k_{e,j}\right)
+\|B\|\|C\|M_c\frac{\rho^{L_m}}{1-\rho}+f_{e,t}
=: e_{c,t+1}. \tag{37}
\]

Gdy \(t\le L_m\), ogon jest pusty i można zastąpić jego ograniczenie zerem. Jeżeli prawdziwa dynamika zawiera dodatkową resztę modelu, jej ograniczenie trzeba dodać do (37). Przy szumie o nieograniczonym nośniku potrzebne jest zdarzenie probabilistyczne z kontrolą wszystkich używanych chwil; nie istnieje automatycznie skończona deterministyczna granica \(f_{e,t}\).

Niech cechy używane przez odczyt spełniają \(z_i=D_z c_i+r_i\), \(\|r_i\|\le r_{e,i}\), gdzie \(D_z\) jest ustaloną mapą rekonstrukcji. Wtedy dla \(\widetilde z_i=D_z\widetilde c_i\)

\[
e_i:=\|D_z\|e_{c,i}+r_{e,i}
\quad\Longrightarrow\quad \|z_i-\widetilde z_i\|\le e_i. \tag{38}
\]

Przy założeniach T7 i kroku SGD liniowego odczytu, z prawdziwymi etykietami,

\[
\delta_S\le (2MR_z+Y_{\max})\frac1{|S|}\sum_{i\in S}e_i. \tag{39}
\]

Podstawienie (39) do (21) daje (22), a spełnienie (23) gwarantuje przewagę wybranej partii nad odniesieniem. Każdy krok tego łańcucha wynika z pokazanej nierówności; nie zakładamy z góry, że wybrana partia jest lepsza. Warunek może okazać się niespełniony lub zbyt zachowawczy.

To również wyjaśnia wybór wymiaru i długości pamięci. Dłuższa pamięć zmniejsza ograniczenie ogona, ale zwiększa liczbę estymowanych współczynników i może zwiększyć sumę \(k_{e,j}\). Więcej modów może zmniejszyć \(r_{e,i}\), lecz pogorszyć estymację operatora. Optymalizujemy całkowity błąd przewidywania użyteczności i koszt, nie samą trwałość modów. Cechy pominięte jako „szybkie” mogą dominować składnik \(r_{e,i}\), jeśli są ważne dla lokalizacji.

## 9. Wybór grup klatek i wiele kroków uczenia

### 9.1. Dlaczego grupa nie jest sumą korzyści pojedynczych klatek

Przy lokalnie kwadratowej stracie \(V\), macierzy drugich pochodnych \(H\) i SGD po średnim gradiencie partii \(\bar g_S\), dokładna poprawa jest równa

\[
G(S)=\eta\langle g,\bar g_S\rangle
-\frac{\eta^2}{2|S|^2}\sum_{i,j\in S}g_i^\top H g_j. \tag{28}
\]

Dla ogólnej dwukrotnie różniczkowalnej straty to przybliżenie z resztą, którą trzeba ograniczyć. W pierwszym składniku klatki się sumują. W drugim oddziałują przez podobieństwo kierunków i krzywiznę straty. Dlatego ranking najlepszych pojedynczych klatek nie musi dać najlepszej grupy. Przy założeniu \(H\preceq LI\) wystarcza dolne ograniczenie z normą średniego gradientu, znane z (18).

Sekwencyjne przeliczanie gradientu na \(D\), jak w lokalnym `glister.py`, przybliża zmieniającą się użyteczność grupy. Nie daje samo z siebie gwarancji optymalności zachłannego wyboru. Nie zakładamy submodularności straty RF-DETR. Kary za podobieństwo czasowe i limity udziału filmów definiują rodzinę \(\mathcal S\); gwarancja (24) odnosi się do tej ograniczonej rodziny.

### 9.2. Pełny zapis jako problem sterowania

Dla skończonej liczby etapów, znanej dynamiki treningu i stanu zawierającego całą potrzebną informację, definiujemy wartość końcową \(J_K(s)=R(\theta)\). Rekurencja Bellmana jest następująca:

\[
J_k(s)=\min_{S\in\mathcal S(s)}
\mathbb E_{\xi}\left[J_{k+1}(\mathcal A(s,S,\xi))\right]. \tag{29}
\]

Pozostały budżet kosztu można dołączyć do \(s\) i zmniejszać po każdej decyzji. (29) dokładnie formalizuje wybór klatek dla końcowego wyniku. Rozwiązanie jest zwykle obliczeniowo niedostępne. Jednokrokowy ranking z (21) jest przybliżeniem decyzji, a nie rozwiązaniem (29). Jeżeli \(R\) zastępujemy \(V_D\), optymalizujemy wynik na danych sterujących i osobno potrzebujemy generalizacji.

### 9.3. Twierdzenie T10: warunkowe tempo zbieżności

Niech w obszarze całej trajektorii obowiązują gładkość z T5 oraz nierówność Polyaka–Łojasiewicza

\[
\|\nabla V(\theta)\|^2\ge2\mu(V(\theta)-V_*),\quad \mu>0. \tag{30}
\]

Ta nierówność mówi, że przy dużej luce do minimum gradient nie może być dowolnie mały. Dodatkowo załóżmy, że dobór dostarcza kierunki
\(d_k=g_k+e_k\), \(\|e_k\|\le\alpha\|g_k\|\), \(0\le\alpha<1\).
Jeżeli

\[
0<\eta\le\frac{1-\alpha}{L(1+\alpha)^2},\qquad
q=1-\mu\eta(1-\alpha)\in(0,1), \tag{31}
\]

to

\[
V(\theta_k)-V_*\le q^k\bigl(V(\theta_0)-V_*\bigr). \tag{32}
\]

**Dowód:** \(\langle g,d\rangle\ge(1-\alpha)\|g\|^2\), a \(\|d\|\le(1+\alpha)\|g\|\). Z (18) poprawa wynosi co najmniej
\([\eta(1-\alpha)-L\eta^2(1+\alpha)^2/2]\|g\|^2\).
Warunek (31) ogranicza ją od dołu przez \(\eta(1-\alpha)\|g\|^2/2\). Stosujemy (30), przenosimy wyrazy i iterujemy. Zakładamy pozostawanie trajektorii w obszarze, w którym wszystkie warunki obowiązują.

Liczba kroków wystarczająca dla dodatniej tolerancji \(\tau<V(\theta_0)-V_*\) wynosi

\[
K_\tau=\left\lceil\frac{\log((V(\theta_0)-V_*)/\tau)}{-\log q}\right\rceil. \tag{33}
\]

To pełne warunkowe twierdzenie o tempie. Dla RF-DETR nie wykazano dotąd (30), stałej \(L\), jednostajnego błędu względnego \(\alpha\) ani utrzymania tych własności na trajektorii. W pobliżu zera gradientu założenie względne jest szczególnie mocne. Nie przedstawiamy (32) jako zmierzonej gwarancji obecnego treningu.

### 9.4. Warunek oszczędności czasu

Górne ograniczenie kosztu metody do tolerancji wynosi
\(T_{\rm metoda}^{\rm up}=C_{\rm init}+K_\tau(C_{\rm select}^{\rm up}+C_{\rm train}^{\rm up}+C_{\rm monitor}^{\rm up})\), jeżeli istnieją takie jednostajne ograniczenia.
Ścisła matematyczna przewaga czasowa nad inną procedurą wynika z
\(T_{\rm metoda}^{\rm up}<T_{\rm baza}^{\rm low}\), gdzie drugie wyrażenie jest **dolnym** ograniczeniem czasu odniesienia do tej samej tolerancji i przy tym samym celu. Porównanie dwóch górnych ograniczeń tego nie dowodzi.

W eksperymencie porównujemy faktyczny czas do z góry ustalonej jakości, na sparowanych inicjalizacjach i przy jednakowym rozliczeniu przygotowania. Przy wielu treningach można amortyzować koszt ekstrakcji, ale trzeba jawnie podać liczbę użyć i koszt odświeżania reprezentacji.

## 10. Generalizacja na osobne filmy

Twierdzenia T5–T10 są analityczne i warunkowe względem wybranego \(V\). Aby przenieść wynik na nowe operacje, potrzebujemy oddzielnej kontroli statystycznej.

**Twierdzenie T11.** Niech \(m\) gotowych detektorów/procedur będzie ustalonych niezależnie od zbioru oceny \(E\). Niech \(E\) zawiera \(n\) niezależnych operacji z rozkładu docelowego. Niech wynik straty jednej operacji należy do \([0,B_\ell]\). Z prawdopodobieństwem co najmniej \(1-\delta\), dla wszystkich tych modeli jednocześnie:

\[
|\widehat R_E(\theta)-R(\theta)|
\le b_n=B_\ell\sqrt{\frac{\log(2m/\delta)}{2n}}. \tag{34}
\]

**Dowód:** dla każdego ustalonego modelu stosujemy nierówność Hoeffdinga do niezależnych strat operacji, następnie sumujemy prawdopodobieństwa \(m\) niepowodzeń. Różnica empirycznych ryzyk większa niż \(2b_n\) daje odpowiednio skierowaną różnicę ryzyk populacyjnych na tym zdarzeniu.

Nie traktujemy tysięcy sąsiednich klatek jako tysięcy niezależnych operacji. Dla straty nieograniczonej trzeba założyć kontrolę ogonów albo jawnie analizować obciętą stratę; obcięcie zmienia cel. Dla AP liczonego globalnie po wszystkich detekcjach nie podstawiamy automatycznie (34); średnia AP po operacjach jest inną metryką niż wspólny AP zbioru.

Jeżeli dobieramy klatki i modele adaptacyjnie na tym samym \(E\), założenie niezależności procedur od \(E\) przestaje zachodzić. Potrzebny jest nowy odłożony zbiór lub mocniejszy wynik o złożoności całej klasy/stabilności algorytmu. Podobnie strojenie reguły zatrzymania po obejrzeniu całych krzywych nie jest prospektywną oceną tej reguły.

## 11. Ile brakuje do zakończenia uczenia

Nie ma jednej liczby wynikającej z widma, która mierzy kompletność modelu. Rozdzielamy:

\[
\Delta_{\rm zad}(Z),\quad
\Delta_{\rm hist}(Y),\quad
\Delta_{\rm hist}(Z_{t+1}),\quad
\Delta_{\rm pula}(s)=\max\left(0,\sup_{S\in\mathcal S}[V(\theta)-V(\theta^+(s,S))]\right). \tag{35}
\]

Pierwsza to utrata informacji o zadaniu przez reprezentację. Druga to dodatkowa użyteczność historii dla zadania. Trzecia to użyteczność historii dla dynamiki latentu. Czwarta to najlepsza możliwa poprawa jednym dopuszczonym krokiem z obecnej puli.

Jeżeli (22) obowiązuje jednocześnie dla **wszystkich** dopuszczonych partii i \(\sup_S(\widehat a_S+E_S)\le\tau\), to \(\Delta_{\rm pula}(s)\le\max(0,\tau)\). To warunkowy certyfikat nasycenia **jednokrokowego, dla tej puli, receptury i straty**. Nie dowodzi, że inne dane, sekwencja kroków, większy model albo inna reprezentacja nie pomogą. Maksimum po krótkiej liście kandydatów nie ogranicza maksimum po całej puli bez dowodu poprawności odsiewu.

Spadek pamięci, stabilizacja bazy i nasycenie pokrycia pozostają pomocniczymi pomiarami. Także odwrotna sytuacja jest możliwa: dobry detektor nie musi dobrze przewidywać przyszłego tła ani całej dynamiki filmu.

## 12. Co wynika dla dotychczasowych implementacji

| Element lokalny | Co faktycznie wylicza | Czego brakuje do odpowiedniego twierdzenia |
|---|---|---|
| K5/K6 i komórki modów | Różnorodność opisu w wybranych współrzędnych | Regularności gradientów w tej metryce i odpowiednich wag z (27); semantyki osi |
| `token_wybierz.py` | Dodatnia różnica błędu pomocniczego odczytu bieżących i przewidzianych tokenów | Ograniczenia błędu przewidywania rzeczywistej zmiany straty detektora |
| `glister.py` | Iloczyn gradientów wybranej głowicy i danych sterujących, odświeżany po wirtualnym SGD | Kontroli krzywizny i błędów, zgodności z rzeczywistym AdamW, wpływu pozostałych parametrów |
| Jądro pamięci i residua | Braki konkretnego modelu przewidywania konkretnego celu | Związku z zadaniem; rozdzielenia błędu estymacji, modelu i innowacji |
| Maksimum trwałości po epokach | Retrospektywnego kandydata na moment zatrzymania | Prospektywnego protokołu i oceny całej reguły na nowych treningach |

W `zajetosc_tozsamosci.py` branie części rzeczywistych wielu wektorów własnych wymaga kontroli par sprzężonych. Dwie takie osie mogą być liniowo zależne. Test udziału wariancji wewnątrz filmu nie identyfikuje sam tożsamości narzędzia. Tego kodu nie zmieniano w ramach niniejszej formalizacji.

Zapisane wyniki są zgodne z powyższymi granicami. W pilocie `dobor_klatek_stan_sieci` pojedynczy bieg Koopman–GLE osiągnął AP 0,67317 wobec 0,65547 dla losowania tego samego budżetu. Nie rozstrzyga to mechanizmu ani powtarzalności. Selekcja i trening razem trwały około 27,12 min, a trening losowego większego zbioru około 20,38 min; nie wykazano oszczędności całej procedury.

W `REPORT_maly_budzet.json` średnia najlepszego wyniku K6 wynosi około 0,62959, a losowania około 0,63193. To nie wspiera twierdzenia o przewadze samego zapełnienia komórek. Te dwa eksperymenty mają różne reguły wyboru i protokoły; nie należy ich scalać w jedno oszacowanie efektu.

## 13. Wynikająca z dowodów procedura

To specyfikacja matematyczna następnej implementacji. Nie uruchomiono jej treningu.

1. Ustalić stratę docelową, rozkład wag operacji, budżet i rzeczywisty optymalizator. Rozdzielić \(U,D,E\) po operacjach.
2. Zamrozić checkpoint podczas pomiaru po czasie filmu. Zapisać prawdziwe odstępy klatek i pochodzenie prostokątów. Wektor obserwabli powinien zachować także informacje potrzebne do lokalizacji.
3. Dopasować modele Koopmana i pamięci na \(U\), z oceną na odrębnych częściach \(U\). Ocenić przewidywanie gradientu lub poprawy straty, nie tylko następnego latentu. Porównać dokładność i koszt z prostym modelem bez dynamiki.
4. Użyć opisu dynamicznego do taniego wstępnego rankingu. Zachować próbę eksploracyjną spoza rankingu i kontrolować straty spowodowane odsiewem. Gwarancję ograniczyć do faktycznie ocenionej rodziny, jeśli odsiew nie ma własnego ograniczenia błędu.
5. Dla krótkiej listy grup obliczyć dokładniejsze gradienty albo rzeczywiste wirtualne kroki. Odtwarzać wagi, stan optymalizatora, bufory i stan losowości. Przy kroku stochastycznym oceniać ustaloną realizację lub odpowiednio kontrolowaną średnią.
6. Wybierać przez dolne ograniczenie poprawy z (22). Do twierdzenia o przewadze nad losowaniem wymagać (23), a nie samego dodatniego wyniku rankingu. Gdy ograniczenia są zbyt szerokie, wynik jest nierozstrzygnięty; dokładne obliczenie lub losowanie jest jawną decyzją procedury.
7. Po zmianie wag sprawdzić ważność opisów i oszacowań błędów. Odświeżanie modeli pomocniczych ma koszt i nie może korzystać z końcowego \(E\).
8. Oceniać całą procedurę i regułę zatrzymania na sparowanych powtórzeniach. Rozliczyć przygotowanie, selekcję, trening i monitorowanie. Potwierdzać AP osobno od analizowanej straty.

Matematyczny cel dla Koopmana–Langevina jest zatem konkretny: **przy mniejszym koszcie zachować wystarczająco dokładne przewidywanie użyteczności grup klatek, aby spełnić (23) albo kontrolować stratę względem najlepszego wyboru przez (24).** To mocniejsze i sprawdzalne zadanie niż samo wyszukiwanie trwałych modów.

## 14. Artefakty kontroli i status dowodu

Skrypt: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\sprawdz_wyprowadzenia.py`.

Raport: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\WERYFIKACJA.json`.

Sprawdzane są tożsamości symboliczne, dokładne kontrprzykłady oraz przykłady numeryczne spełniające założenia nierówności. Testowanie skończonej liczby przykładów nie zastępuje dowodów. Nie zweryfikowano empirycznie założeń T6–T11 dla całego RF-DETR.

### 14.1. Dodatkowa kontrola w Lean

Źródło dowodów: `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\ViTFrameSelection.lean`.

Powtórzenie kontroli: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\sprawdz_lean.ps1`.

Raport z hashem źródła i audytem aksjomatów: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\LEAN_WERYFIKACJA.json`.

W Lean 4.28.0 z lokalnym Mathlib sprawdzono następujące twierdzenia:

| Nazwa w Lean | Zakres |
|---|---|
| `alignment_error` | Błąd iloczynu gradientu i kierunku, przy ograniczonych błędach obu wektorów |
| `direction_norm`, `direction_norm_sq` | Kontrola normy rzeczywistego kierunku i jej kwadratu |
| `gain_error` | Przedział poprawy z T6, **przy założonym ograniczeniu reszty Taylora**; parametr `curvature` odpowiada \(L\eta^2/2\) |
| `certified_improvement` | Dodatnie dolne ograniczenie poprawy implikuje mniejszą stratę |
| `certified_comparison` | Rozdzielenie dolnej granicy wyboru i górnej granicy odniesienia implikuje przewagę wyboru |
| `selection_regret` | Strata względem dowolnego kandydata o nie większej przewidzianej korzyści jest ograniczona przez podwojony błąd |
| `particle_error` | Ograniczenie błędu ważonego opisu klatki na podstawie błędów osobnych cząstek, przy tych samych wagach |
| `particle_gradient_error` | Przeniesienie tego ograniczenia przez mapę gradientu o **założonej, znanej wrażliwości** |

Plik nie zawiera `sorry` ani `admit`. Wszystkie zależności aksjomatyczne wypisane przez Lean ograniczają się do standardowych `propext`, `Classical.choice`, `Quot.sound`. Nie dodano aksjomatu o korzyści z klatek ani o poprawności Koopmana.

Nie przeniesiono jeszcze do Lean dowodu analitycznego T5 z gładkości, jawnej stałej T7 dla liniowego odczytu, całego równania MZ, zbieżności T10 ani generalizacji T11. `gain_error` przyjmuje ograniczenie reszty jako hipotezę, a `particle_gradient_error` przyjmuje ograniczenie wrażliwości mapy. Nie wolno przedstawiać tej częściowej formalizacji jako maszynowego dowodu całego dokumentu.

Oddzielone poziomy wniosku:

- **Udowodnione w podanym modelu matematycznym:** T1–T11 i ich wskazane wnioski, przy jawnych założeniach.
- **Sprawdzone obliczeniowo:** pozycje raportu `WERYFIKACJA.json`, z podanym zakresem.
- **Sprawdzone formalnie w Lean:** twierdzenia z sekcji 14.1, z hipotezami widocznymi w ich sygnaturach.
- **Zmierzone wcześniej:** wyniki z sekcji 12, zgodnie z lokalnymi raportami.
- **Nadal hipoteza badawcza:** że Koopman–Langevin przewiduje korzyść z klatek dostatecznie tanio i dokładnie, by poprawić AP RF-DETR i czas całej procedury względem mocnych bazowych metod.
