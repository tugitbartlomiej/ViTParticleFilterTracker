# Reszta rozszerzonego Langevina, niepewność Kalmana i dobór klatek

Data: 2026-09-23. Kontynuacja [formalizacji 44](44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md).

**Wniosek:** reszta może dostarczyć informacji o brakach przewidywania, a Kalman pozwala odnieść błąd do przewidywanej niepewności. Ich połączenie nie daje jeszcze gwarancji przydatności klatki do treningu detektora. Poniżej wyprowadzamy warunki pozwalające połączyć te wielkości z przedziałem poprawy straty z dokumentu 44.

Status: wyprowadzenie, audyt istniejącego kodu i sprawdzenia algebraiczne na układach kontrolnych. Nie uruchomiono treningu ani nowego filtra na filmach. Nowych twierdzeń z tego dokumentu nie sprawdzano w Lean; wcześniejszy rdzeń z dokumentu 44 pozostaje sprawdzony w opisanym tam zakresie.

## 1. Punkt wyjścia w istniejącym projekcie

Odczytano następujące źródła lokalne:

- `SmallViTAblation/experiments/dobor_klatek_stan_sieci/model.py`: `ExtendedLangevin`, `ColoredResidual`, `EventTerm`.
- `SmallViTAblation/experiments/dobor_klatek_stan_sieci/token_model.py`: rozszerzenie z własną skalą tokenów.
- `SmallViTAblation/pipeline/koopman_langevin/kalman_state.py`: filtr położenia, prędkości i przyspieszenia oraz zapisane raporty jego oceny.
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_krok_po_kroku\README.md`: kolejność przewidź–zaobserwuj–zaktualizuj oraz dwie reszty.
- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\DiagnostykaInnowacji.lean`: rozkład błędu przez rzut ortogonalny; odczyt bez modyfikacji i bez ponownej kompilacji tego pliku.

Ścieżki względne w tym dokumencie odnoszą się do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`.

Standardowy krok Kalmana i jego kowariancje opisują algorytmy w [opracowaniu Särkkä i współautorów, rozdział 5](https://users.aalto.fi/~ave/publications/SarkkaTamminenVehtariLampinen_B36.pdf). Przy nieliniowej dynamice filtrowanie wymaga przybliżeń; rozróżnienie to zachowujemy zgodnie z [Bayesian Filtering and Smoothing, rozdziały 6–7](https://www.cambridge.org/core/books/bayesian-filtering-and-smoothing/F88740E8D25010CF3119A5CA379FA37A). Pozostałe wnioski o doborze klatek wyprowadzamy tutaj, a nie przypisujemy tym źródłom.

## 2. Cząstki pozostają osobnymi tokenami

Niech \(z_{t,p}\in\mathbb R^d\) oznacza cechy tokenu \(p\) na klatce \(t\), przy zamrożonych wagach detektora. Pole cząstek to \(Z_t=(z_{t,1},\ldots,z_{t,P})\). Redukcja liczby kanałów nie usuwa indeksu cząstki. Cząstka na stałej siatce nie jest automatycznie tym samym fragmentem narzędzia w kolejnych klatkach. Śledzenie wymaga jawnej korespondencji.

Można diagnozować osobne tokeny lub wybrane współrzędne całego pola. W obu przypadkach zachowujemy pochodzenie, pozycję i czas obserwacji. Tokeny współdzielą obraz i uwagę transformera, więc ich błędów nie wolno z góry uznać za niezależne.

Nie zakładamy, że pojedynczy mod oznacza tożsamość, metal, ruch albo tło. Tę interpretację trzeba sprawdzić względem oznaczeń i kontroli. Także cząstka wewnątrz ramki może opisywać tkankę.

## 3. Dwie reszty i znaczenie pozostałych członów

Pomijając indeks cząstki, zapis aktualnej implementacji jest następujący:

\[
\widehat z^{\,0}_{t+1}=z_t+D_t+JP_t+W_t+Z_t^{\rm zd},\qquad
r^0_{t+1}=z_{t+1}-\widehat z^{\,0}_{t+1},
\]
\[
\widehat N_{t+1}=\mathcal N(r^0_t,r^0_{t-1},\ldots),\qquad
\widehat z_{t+1}=\widehat z^{\,0}_{t+1}+\widehat N_{t+1},\qquad
e_{t+1}=r^0_{t+1}-\widehat N_{t+1}. \tag{1}
\]

W `model.py` nazwy zapisów to `base_residual`, `terms["N"]`, `residual`. Człon \(\widehat N\) jest **deterministyczną prognozą z dawnych reszt**, a nie nowym losowym szumem. Końcowe \(e\) jest dopiero kandydatem na innowację, czyli część nieprzewidywalną z ustalonej dostępnej informacji.

| Człon | Co reprezentuje implementacja | Co można sprawdzać |
|---|---|---|
| \(D\) | Zależność przyrostu od bieżących cech, prędkości i przyspieszenia | Czy prosty opis lokalnego ruchu wystarcza |
| \(JP\) | Wyuczone wykorzystanie historii | Czy historia zmniejsza błąd poza filmami dopasowania |
| \(W\) | Wpływ dostępnych metadanych i wejść | Czy pomiar ruchu kamery wyjaśnia część zmiany |
| \(Z^{\rm zd}\) | Korekta oparta na kotwicy i zmianie pewności | Czy pomaga przy utracie i odzyskaniu obserwacji |
| \(\widehat N\) | Przewidywalna część wcześniejszych reszt | Czy po innych członach została struktura czasowa |
| \(e\) | Błąd po wszystkich poprawkach | Czy pozostaje systematyczny błąd istotny dla zadania |

Ruch kamery może stanowić wejście \(W\), ale samo nazwanie członu wymuszeniem nie mierzy kamery. Nieobserwowany ruch ręki może nadal pozostać w reszcie. W obecnym kodzie `observed` pochodzi z pewności detektora, nie z niezależnego pomiaru fizycznej obecności narzędzia.

Duża norma \(JP\) albo \(\widehat N\) oznacza dużą **już zastosowaną** poprawkę. Nie mierzy automatycznie tego, czego modelowi jeszcze brakuje. Gdy człony są elastyczne i skorelowane, mogą się wzajemnie zastępować. Ich fizycznej interpretacji nie da się dowieść z samego rozkładu sumy.

W trybie prognostycznym wszystkie człony prognozy muszą być mierzalne względem historii \(\mathcal H_t\), dostępnej przed docelową obserwacją. Tryb warunkowy z wejściami z chwili docelowej ma inną informację i wymaga osobnej oceny. Historyczne reszty trzeba liczyć przy tych samych zamrożonych parametrach albo jawnie modelować ich zmianę.

## 4. Która część reszty rzeczywiście daje się poprawić

Niech \(m_t=\widehat z_{t+1}\), \(m_t^*=\mathbb E[z_{t+1}\mid\mathcal H_t]\), \(b_t=m_t^*-m_t\). Wtedy

\[
e_{t+1}=b_t+\varepsilon_{t+1},\qquad
\mathbb E[\varepsilon_{t+1}\mid\mathcal H_t]=0.
\]

**R1.** Przy skończonych drugich momentach:

\[
\mathbb E[\|e_{t+1}\|^2\mid\mathcal H_t]
=\|b_t\|^2+\operatorname{tr}\Sigma_t,
\quad \Sigma_t=\operatorname{Cov}(\varepsilon_{t+1}\mid\mathcal H_t). \tag{2}
\]

Dowód: rozwinąć kwadrat; warunkowa średnia wyrazu \(2b_t^\top\varepsilon_{t+1}\) jest zerowa. Pierwszy składnik jest błędem warunkowej średniej. Drugi pozostaje nieprzewidywalny **dla tego celu i tej historii**. Nie nazywamy go nieusuwalnym błędem detektora.

**R2.** Dla poprawki \(c_t\) wyznaczonej wyłącznie z \(\mathcal H_t\):

\[
\mathbb E[\|e_{t+1}\|^2-\|e_{t+1}-c_t\|^2\mid\mathcal H_t]
=2b_t^\top c_t-\|c_t\|^2
=\|b_t\|^2-\|b_t-c_t\|^2. \tag{3}
\]

Dowód: ten sam rozwój kwadratu. Poprawka pomaga średnio dokładnie wtedy, gdy lepiej przybliża brakującą warunkową średnią niż poprawka zerowa. To ścisły sposób oceny, czy kolejny człon równania wyjaśnia przewidywalny błąd. Dopasowanie i ocena poprawki muszą dotyczyć odrębnych filmów.

### Pułapka wybierania największych zaobserwowanych reszt

Dla dwóch prognoz \(h,r\) oraz \(y=m^*+\varepsilon\) punktowo:

\[
\|y-h\|^2-\|y-r\|^2
=\|m^*-h\|^2-\|m^*-r\|^2+2\varepsilon^\top(r-h). \tag{4}
\]

Ostatni składnik znika po odpowiednim uśrednieniu warunkowym, a nie na każdej klatce. Przykład: \(m^*=h=0,r=1,\varepsilon\in\{-1,1\}\) z równymi szansami. Poprawka do prognozy \(1\) pogarsza średni błąd o \(1\). Po wybraniu tylko przypadków \(\varepsilon=1\) pozornie poprawia go o \(1\).

To ogranicza interpretację lokalnego twierdzenia `roznica_strat_bez_zaskoczenia`: w przestrzeni \(L^2\) norma dotyczy uśrednionego błędu. Wybór grupy według właśnie zaobserwowanej reszty może zmienić rozkład innowacji. Twierdzenia o podgrupach wyznaczonych z historii nie uzasadniają automatycznie takiego wyboru.

Równanie (3) dotyczy kwadratowego błędu prognozy. Nie stosujemy go bez nowego dowodu do różnicy BCE pomocniczego odczytu ani do AP detektora. Brak autokorelacji liniowej reszt także nie dowodzi \(b_t=0\): mogą pozostać zależności nieliniowe od historii.

## 5. Jak połączyć Langevina z Kalmanem bez liczenia tego samego dwa razy

Proponujemy stan \(s_t\), zawierający wybrane cechy cząstek oraz zmienne potrzebne do reprezentacji pamięci i barwnej reszty. Przy skończonym oknie można dołączyć historię; przy jądrze wykładniczym można użyć stanów rekurencyjnych. Ogólna pamięć Mori–Zwanziga nie musi mieć dokładnej realizacji skończenie wymiarowej. Jej obcięcie jest dodatkowym błędem modelu.

Model:

\[
s_{t+1}=F_\psi(s_t,u_t)+\xi_t,\qquad y_t=H_t s_t+v_t. \tag{5}
\]

\(F_\psi\) zawiera sumę członów rozszerzonego Langevina. Kalman aktualizuje oszacowanie stanu i jego niepewność po nadejściu nowej obserwacji. \(Q_t\) opisuje kowariancję nieprzewidzianego zakłócenia procesu \(\xi_t\), a \(R_t\) kowariancję błędu obserwacji \(v_t\).

To wymaga **zdefiniowania obserwowanego stanu**. Enkoder z zamrożonymi wagami może być deterministyczny. Wówczas \(R_t\) nie opisuje losowości obliczenia tokenu, lecz jego błąd jako pomiaru postulowanego ukrytego stanu. Bez takiego uzasadnienia uczciwiej estymować warunkowy rozkład reszty tokenu, bez nazywania go dokładnym filtrem stanu fizycznego.

Nie dodajemy niezależnie drugiego predyktora tych samych dawnych reszt, jeśli \(\widehat N\) już je uwzględnia. Nie uznajemy też ramki detektora i cech z tego samego obrazu za dwa niezależne pomiary. Potrzebne są wspólne kowariancje albo jawne przybliżenie wraz z oceną kalibracji.

Dla modelu liniowego, poprawnych parametrów, niezależnych białych szumów i gaussowskiego stanu początkowego standardowy filtr jest dokładny. Dla nieliniowego \(F_\psi\) poniższy zapis z \(A_t=J F_\psi(\widehat s_t^+)\) jest lokalnym przybliżeniem EKF:

\[
\widehat s_{t+1}^-=F_\psi(\widehat s_t^+,u_t),\quad
P_{t+1}^-=A_tP_t^+A_t^\top+Q_t,
\]
\[
\nu_{t+1}=y_{t+1}-H_{t+1}\widehat s_{t+1}^-,\quad
S_{t+1}=H_{t+1}P_{t+1}^-H_{t+1}^\top+R_{t+1},
\]
\[
K_{t+1}=P_{t+1}^-H_{t+1}^\top S_{t+1}^{-1},\quad
\widehat s_{t+1}^+=\widehat s_{t+1}^-+K_{t+1}\nu_{t+1},
\]
\[
P_{t+1}^+=(I-KH)P_{t+1}^-(I-KH)^\top+KRK^\top. \tag{6}
\]

Ostatni wiersz to postać Josepha; pominięte indeksy odnoszą się do tej samej chwili. W prognozach na nieregularnie próbkowanej puli przejście i \(Q\) muszą uwzględniać rzeczywisty odstęp czasu. Innowację zapisujemy **przed aktualizacją**. Reszta po korekcie jest mechanicznie zmniejszona i odpowiada innemu pytaniu.

## 6. Wielkości, których nie wolno utożsamiać

\[
\underbrace{\|\nu_t\|^2}_{\text{wielkość błędu}},\qquad
\underbrace{\operatorname{tr}P_t^-}_{\text{niepewność stanu}},\qquad
\underbrace{\nu_t^\top S_t^{-1}\nu_t}_{\text{błąd względem oczekiwanej skali}}. \tag{7}
\]

Duży błąd przy dużej przewidzianej zmienności może być zwyczajny. Mały błąd przy bardzo małej przewidzianej zmienności może być nieoczekiwany. Żaden przypadek sam nie rozstrzyga, czy dodatkowy trening poprawi detektor.

Przy dokładnym modelu liniowym i gaussowskim oraz dodatnio określonym \(S_t\), ostatnia wielkość ma warunkowo rozkład \(\chi^2_q\), gdzie \(q\) to wymiar obserwacji. Dowód: przemnożenie innowacji przez \(S_t^{-1/2}\) daje standardowy wektor normalny. Nie przenosimy tego rozkładu na dowolny EKF, obcinanie innowacji czy ciężkie ogony. Modele odporne wymagają własnej aktualizacji i oceny rozkładu; przykład stanowi [Roth i współautorzy, filtrowanie z rozkładem Studenta](https://arxiv.org/abs/1703.02428).

Kalibracja oznacza sprawdzenie, czy deklarowane zakresy rzeczywiście obejmują błędy z deklarowaną częstością na osobnych filmach. Pewność klasyfikatora nie jest automatycznie wariancją pomiaru. Same obserwacje innowacji nie zawsze pozwalają jednoznacznie oddzielić \(Q\), \(R\) i błąd modelu.

### Bezpośrednie połączenie reszty i niepewności

**R3.** Dla reszty \(\nu=b+\varepsilon\), o warunkowej średniej \(b\) i warunkowej kowariancji \(\Sigma\), oraz dodatnio określonej macierzy odniesienia \(S\), znanej z historii:

\[
\mathbb E[\nu^\top S^{-1}\nu\mid\mathcal H]
=\operatorname{tr}(S^{-1}\Sigma)+b^\top S^{-1}b. \tag{7a}
\]

Dowód: rozwinąć formę kwadratową, wyzerować średni wyraz mieszany i użyć \(\mathbb E[\varepsilon^\top A\varepsilon\mid\mathcal H]=\operatorname{tr}(A\Sigma)\). Nie potrzeba założenia gaussowskiego, wystarczą skończone drugie momenty.

Jeżeli \(S=\Sigma\), to dla \(q\)-wymiarowej obserwacji:

\[
\mathbb E[\nu^\top S^{-1}\nu\mid\mathcal H]-q=b^\top S^{-1}b\ge0. \tag{7b}
\]

Zatem **średnia nadwyżka znormalizowanego błędu ponad poziom wynikający z wymiaru mierzy błąd przewidywania warunkowej średniej**, ale tylko przy poprawnej kowariancji. Gdy \(S\) jest zaniżone, nadwyżka może pochodzić wyłącznie ze złej kalibracji. Odejmowanie \(q\) od wyniku pojedynczej klatki nie usuwa losowej realizacji szumu. Przed użyciem nadwyżki do wyboru trzeba oszacować ją dla porównywalnych sytuacji, poza danymi dopasowania, i powiązać z zadaniem detektora.

## 7. Co można dowieść o zmniejszeniu niepewności

**K1.** Przy założeniach dokładnego filtra i odwracalnym \(S\):

\[
P^- -P^+=P^-H^\top S^{-1}HP^-\succeq0. \tag{8}
\]

Dowód: podstawienie \(K=P^-H^\top S^{-1}\) do (6) daje równość. Dla dowolnego \(a\) forma kwadratowa prawej strony wynosi \(\|S^{-1/2}HP^-a\|^2\ge0\).

Dla ustalonej liniowej wielkości zadaniowej \(Ts\), zmniejszenie oczekiwanego kwadratowego błędu estymacji wynosi

\[
u_{\rm stan}=\operatorname{tr}\big[T(P^- -P^+)T^\top\big]\ge0. \tag{9}
\]

Dowód: błąd estymacji średnią warunkową ma średni kwadrat równy śladowi kowariancji; zastosować (8) i liniowość śladu. Przy nieliniowej głowicy jej Jacobian daje jedynie lokalne przybliżenie \(T\). Zmniejszenie niepewności dotyczy **oszacowania stanu po pomiarze**, nie zmiany wag detektora po treningu.

To rozróżnienie ma znaczenie dla cząstek. Jeżeli pierwsza współrzędna opisuje tło, druga wielkość zadaniową, \(P=\operatorname{diag}(100,1)\), a pomiar dotyczy wyłącznie tła, ogólna niepewność silnie maleje. Niepewność zadania \(T=(0,1)\) nie maleje wcale. Suma niepewności wszystkich tokenów może więc promować klatki trudne dla tła.

**K2.** Dla ustalonych \(A_t,H_t,Q_t,R_t,P_0\) oraz tego samego harmonogramu obserwacji ciąg \(P_t,S_t\) nie zależy od wartości obrazów/pomiarów. Dowód indukcyjny: rekurencja (6) dla kowariancji nie zawiera \(y_t\). Różne innowacje nie zmieniają tego faktu.

## 8. Co faktycznie robi obecny filtr w projekcie

`kalman_state.py` filtruje stan \((p,v,a)\) osobno na każdej osi obrazu. Nie jest jeszcze filtrem całego pola tokenów. Każdy dopasowany zestaw parametrów daje wspólny ciąg wariancji wszystkim ocenianym sekwencjom o tym samym harmonogramie. To dokładnie ograniczenie K2. Różnice wartości innowacji nadal mogą odróżniać klatki.

Kod obcina innowację przy aktualizacji średniej, ale kowariancję aktualizuje jak w standardowym filtrze. Dopasowuje kryterium Hubera. Dlatego zwracane \(P,S\) są **nominalnymi** macierzami użytego modelu, a nie dowiedzioną kowariancją posterioru tej odpornej procedury. Przy starcie wykorzystuje pierwszy pomiar jako położenie początkowe; pierwsza innowacja nie jest niezależnym sprawdzianem prognozy. Wygładzanie wykorzystuje przyszłe obserwacje i trzeba je oddzielać od prognozy przyczynowej.

Zapisane wcześniej wyniki AUC rozpoznawania nietrafionej ramki na podstawie sygnału:

| Checkpoint | Innowacja z odczytu latentu | Innowacja z ramki detektora | Niepewność klasyfikatora, czyli minus confidence |
|---|---:|---:|---:|
| `ut_b0500_best` | 0,6745 | 0,6780 | 0,8133 |
| `ut_b2500_best` | 0,6781 | 0,6031 | 0,8268 |

Źródła: `SmallViTAblation/docs/Analiza_ponownej_publikacji/evidence_koopman_langevin_2026-09-21/REPORT_kalman_ut_b0500_best.json` i `REPORT_kalman_ut_b2500_best.json`. To odczyt istniejących raportów, bez nowego pomiaru. AUC mierzy porządkowanie klatek z błędną ramką, a nie korzyść z ich użycia do treningu. Raportowane ciężkie ogony i pozostała autokorelacja dodatkowo ograniczają interpretację gaussowską.

Poprawiono opis w kodzie, który wcześniej utożsamiał białe reszty z kompletnością modelu, kryterium Hubera z gaussowskim maximum likelihood oraz oznaczenia wyłącznie z oceną. Odczyt położenia z latentu jest nadzorowany. Nie zmieniono obliczeń ani wyników tych eksperymentów.

## 9. Niepewność stanu a niepewność parametrów

Niech \(\Theta\) oznacza niepewne parametry modelu, a \(\mathcal H\) dostępną informację. Z prawa całkowitej kowariancji:

\[
\operatorname{Cov}(y\mid\mathcal H)=
\mathbb E_\Theta[\operatorname{Cov}(y\mid\mathcal H,\Theta)\mid\mathcal H]
+\operatorname{Cov}_\Theta(\mathbb E[y\mid\mathcal H,\Theta]\mid\mathcal H). \tag{10}
\]

Dowód: zapisać \(y-\mathbb E[y\mid\mathcal H]\) jako sumę odchylenia od średniej przy ustalonym \(\Theta\) i odchylenia tej średniej; mieszane składniki mają zerową średnią warunkową.

Filtr z ustalonymi parametrami nie oblicza automatycznie drugiego składnika. Rozrzut różnych modeli dynamiki może go przybliżać, ale nie jest od razu rozrzutem wag detektora. Trzeba jawnie określić, których parametrów dotyczy \(\Theta\). Uczenie detektora i estymacja bieżącego stanu są różnymi działaniami.

## 10. Przejście od niepewności cząstek do przedziału poprawy

Nie podstawiamy \(S\) z (6) w miejsce błędu cech bez dodatkowego modelu: \(S\) opisuje innowację obserwacji, a potrzebujemy błędu tej reprezentacji, której używa przybliżony gradient.

**K3.** Załóżmy, że dla błędu reprezentacji \(a=z-\widetilde z\), przy informacji \(\mathcal G\) dostępnej w chwili wyboru, mamy poprawne ograniczenie

\[
\mathbb E[\|a\|^2\mid\mathcal G]\le M_{\mathcal G}.
\]

Dla \(0<\alpha<1\), z warunkowej nierówności Markowa:

\[
\Pr\left(\|a\|>\sqrt{M_{\mathcal G}/\alpha}\mid\mathcal G\right)\le\alpha. \tag{11}
\]

Jeżeli znane są warunkowa średnia błędu \(b\) i kowariancja \(P\), można wziąć \(M=\|b\|^2+\operatorname{tr}P\). Pominięcie obciążenia \(b\) zaniża zakres. Dla dokładnego gaussowskiego błędu w wymiarze \(d\) mocniejsze ograniczenie to

\[
\|a\|\le\|b\|+\sqrt{\lambda_{\max}(P)\,\chi^2_{d,1-\alpha}}
\quad\text{z prawdopodobieństwem co najmniej }1-\alpha. \tag{12}
\]

Dowód: zapisać \(a=b+P^{1/2}w\), zastosować nierówność trójkąta i \(\|P^{1/2}w\|\le\sqrt{\lambda_{\max}(P)}\|w\|\). Macierz dopasowana z małej próby bez kontroli jej błędu nie spełnia automatycznie założeń (11) ani (12).

Dla opisu klatki \(q=\sum_p w_pz_p\), z tymi samymi wagami dla obu reprezentacji, na jednoczesnym zdarzeniu ograniczeń \(\|a_p\|\le e_p\):

\[
\|q-\widetilde q\|\le\sum_p|w_p|e_p. \tag{13}
\]

Suma prawdopodobieństw przekroczeń wystarcza bez niezależności cząstek. Jeżeli wagi same zmieniają się wraz z reprezentacją, dochodzi składnik błędu wag. Alternatywnie, dla znanych wspólnych kowariancji i ustalonych warunkowo wag:

\[
P_q=\sum_{p,r}w_pw_r\operatorname{Cov}(a_p,a_r\mid\mathcal G). \tag{14}
\]

Pominięcie składników \(p\ne r\) wymaga uzasadnienia. Wspólny błąd wszystkich tokenów nie znika przez uśrednienie dużej liczby tokenów.

Jeżeli mapa od reprezentacji do rzeczywistego kierunku aktualizacji ma znaną stałą wrażliwości \(C\), uzyskujemy

\[
\|d-\widehat d\|\le\delta=C\sum_p|w_p|e_p. \tag{15}
\]

Dla samego gradientu analogiczna nierówność daje błąd gradientu; przejście do AdamW wymaga uwzględnienia jego stanu i wrażliwości. Cząstki ani Kalman nie usuwają tego wymagania.

Z twierdzenia o przedziale poprawy w dokumencie 44, dla \(\theta^+=\theta-\eta d\), \(g=\nabla V(\theta)\), \(\|g-\widehat g\|\le\epsilon\), \(\eta\ge0\), lokalnej stałej gładkości \(L\):

\[
V(\theta)-V(\theta^+)\ge\eta\langle\widehat g,\widehat d\rangle-E,
\]
\[
E=\eta\left[\epsilon\|\widehat d\|+(\|\widehat g\|+\epsilon)\delta\right]
+\frac{L\eta^2}{2}(\|\widehat d\|+\delta)^2. \tag{16}
\]

Jeżeli wszystkie potrzebne ograniczenia zachodzą z łącznym prawdopodobieństwem co najmniej \(1-\alpha\), taki sam status ma (16). To konkretny wkład poprawnie skalibrowanej niepewności: **określa szerokość przedziału przewidywanej poprawy**, zamiast być kolejnym arbitralnie dodanym wynikiem rankingu.

Przy wyborze najlepszego z wielu kandydatów wymagamy ograniczeń jednoczesnych albo gwarancji warunkowych względem całej informacji użytej przy wyborze. Pokrycie marginalne dla losowej klatki nie gwarantuje pokrycia dla maksimum rankingu. Przewaga nad losowaniem wymaga rozdzielenia dolnej granicy wybranego wariantu i górnej granicy wariantu odniesienia, jak w dokumencie 44.

## 11. Jak diagnozować poszczególne człony

Proponowana ocena na osobnych filmach obejmuje trzy różne pytania:

1. Czy wyłączenie członu przy zamrożeniu innych pogarsza prognozę? To mierzy użyteczność w istniejącym dopasowaniu.
2. Czy ponowne dopasowanie modelu bez tego członu pogarsza prognozę? To ocenia potrzebę tego składnika wobec możliwości zastąpienia go innymi.
3. Czy jego poprawka przewiduje błąd zadania lub gradient detektora? Dopiero to wiąże go z doborem klatek.

Nie utożsamiamy tych ocen. Porównanie modeli o różnej pojemności musi uwzględniać koszt oraz dopasowanie. Uczenie modelu reszty i ocenianie go na tych samych klatkach zawyża przewidywalną część błędu.

Ruch kamery i zmiany tła mogą zwiększać reszty wielu cząstek jednocześnie. Pomiar kamery powinien być kontrolowany przez wariant bez tego wejścia i z wejściem potasowanym. Usunięcie ruchu z reprezentacji wymaga zachowania informacji o położeniu potrzebnej detekcji. Gdy celem jest pojedyncza klatka, nie każdą informację dostępną dopiero z historii da się przenieść do jej enkodera.

## 12. Specyfikacja dalszej metody i granica wniosku

Zachowujemy osobne cząstki, porównujemy ich prognozy przed obserwacją i zapisujemy \(r^0,e,\nu\) wraz z właściwą skalą niepewności. Uczymy się na osobnych operacjach, które reszty dają się przewidzieć i które odnoszą się do błędu detektora. Są to dane wejściowe do przybliżenia kierunku aktualizacji lub korzyści z grupy klatek.

Następnie kontrolujemy błąd tego przybliżenia, uwzględniając niepewność stanu, parametrów i kalibracji. Wybór uzasadniamy przez (16) albo przez dokładniejsze wirtualne kroki dla krótkiej listy. Zachowujemy klatki eksploracyjne, aby błąd obecnego modelu nie wykluczył całych obszarów puli. Koszt selekcji wliczamy do czasu uczenia.

Wysoka reszta, duża niepewność Kalmana, duża poprawka pamięci i trwały mod pozostają różnymi diagnozami. **Możliwy do dowiedzenia związek prowadzi od kontrolowanego błędu reprezentacji do kontrolowanego błędu prognozy aktualizacji, a następnie do dolnego ograniczenia poprawy straty.** Nie prowadzi bezpośrednio od największej reszty do najlepszej klatki.

## 13. Weryfikacja

Skrypt: `SmallViTAblation/analysis/formalizacja_doboru_klatek/sprawdz_reszte_kalmana.py`.

Raport: `SmallViTAblation/analysis/formalizacja_doboru_klatek/RESZTA_KALMAN_WERYFIKACJA.json`.

Sprawdzenia obejmują tożsamości (1)–(4) i (7a), skalarną aktualizację Kalmana, zgodność postaci Josepha, zmniejszenie kowariancji, niezależność nominalnej wariancji istniejącej implementacji od obserwacji, kontrprzykład tła, korelacje cząstek, obciążenie błędu i przejście do przedziału poprawy na kontrolowanych układach liniowych. Nie stanowią pomiaru skuteczności doboru klatek ani dowodu poprawnej kalibracji na narzędziach chirurgicznych.
