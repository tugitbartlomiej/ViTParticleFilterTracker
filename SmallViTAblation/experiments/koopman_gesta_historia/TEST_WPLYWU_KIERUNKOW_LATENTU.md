# Test wpływu kierunków latentu na wykrywanie narzędzia

Data: 2026-09-24. Status: protokół badawczy i hipoteza do sprawdzenia. Nie wykonano opisanych interwencji. Dokument nie potwierdza poprawy detektora ani pełnego domknięcia tożsamości. Nie zmienia treningu na Edenie.

## 1. Pytanie i hipoteza

Czy kierunki rozpoznane na podstawie masek narzędzia i kontrolowanych zmian obrazu mają selektywny wpływ na decyzje RF-DETR? Czy przyrost przewidziany z historii uczenia przez Koopmana zmienia te decyzje w stronę mniejszego błędu?

Hipoteza H1: interwencje w kierunkach kandydujących do opisu narzędzia zmieniają błędy obecności i lokalizacji silniej i bardziej selektywnie niż interwencje kontrolne o takim samym rozmiarze.

Hipoteza H2: przyrost przewidziany przez Koopmana zmniejsza błąd zamrożonego detektora na odłożonych obrazach częściej lub silniej niż przyrost zerowy, przeciwny i losowy o tej samej normie. H1 nie implikuje H2. Potwierdzenie H2 w interwencji nie dowodzi korzyści z dotrenowania na znalezionej klatce.

Hipoteza H3, na przyszłość: model z pamięcią i znanym sterowaniem lepiej przewiduje funkcjonalnie istotne przyrosty niż model bez pamięci. Langevin nie jest teraz uruchamiany na mocy tego dokumentu.

## 2. Pełny stan i miejsce interwencji

Dla obrazu x i checkpointu k zapisujemy Z_k(x) w R^(576 x 384). Jeden token/cząstka to z_p w R^384. Kierunek v w R^384 to kombinacja kanałów, a nie jedna z 576 cząstek. Pełny kierunek V może mieć rozmiar 576 x 384.

Nie stosujemy PCA, redukcji liczby kanałów ani zastępowania stanu średnią. Badanie jednego kierunku jest kontrolowaną zmianą pełnego tensora, a nie jego kompresją. Reszta tensora pozostaje w sieci.

Zapisujemy dokładny moduł, kształt tensora, normalizację, kolejność pozycji i wersję wag. Sieć ma postać:

    D_k(x) = H_k(Z_k(x), R_k(x)).

R oznacza wszystkie dodatkowe wejścia dalszej części sieci, jeśli istnieją: inne poziomy cech, pozycje, maski techniczne i drogi omijające miejsce interwencji. Trzeba je zidentyfikować w kodzie, a nie zakładać ich obecność lub nieobecność. Podczas testu Z zmieniamy tylko wskazany tensor, natomiast H, R i wszystkie wagi są zamrożone. Causalność oznacza wpływ interwencji w tym grafie obliczeń, nie automatycznie przyczynę w świecie fizycznym.

Maskę narzędzia sprowadzamy do siatki jako udział pikseli narzędzia m_p w [0,1]. Osobno raportujemy wnętrze, brzeg i tło. Token wewnątrz maski może zawierać informacje spoza niej. Maska służy do wyboru miejsca interwencji i oceny; nie dodajemy jej jako nowego kanału wejściowego do detektora. To analiza z adnotacjami, nie metoda detekcji bez dostępu do prawdy.

## 3. Dane i identyfikacja kandydatów

Podział po filmach lub źródłach narzędzia: dopasowanie kierunków, wewnętrzny dobór ustawień, końcowa ocena. Wszystkie warianty tego samego wycinka muszą pozostać w tej samej grupie. 256 kolorów jednego wycinka nie stanowi 256 niezależnych obiektów. Przy zbyt małej liczbie niezależnych filmów wynik pozostaje pilotem.

Wykorzystujemy pary: to samo narzędzie na różnych tłach; ten sam kadr z narzędziem i poprawnie przygotowana kontrola bez niego; znany obrót; skala; światło. Nie utożsamiamy transformacji 2D z pełnym obrotem sztywnego narzędzia w 3D. Artefakty wycinania kontrolujemy przez identyczny sposób kompozycji obu klas. Do twierdzenia o rozróżnianiu konkretnego typu narzędzia potrzebne są także inne narzędzia, a nie wyłącznie tło.

Na pełnych 384 kanałach dopasowujemy regularizowane odczyty obecności, kąta i innych oznaczonych własności. Oddzielnie sprawdzamy naturalne osie kanałów względem maski. Kandydatów nie wyznaczamy na końcowym zbiorze oceny. Rzetelnie raportujemy zarówno dodatnie, jak i ujemne odpowiedzi oraz kierunki bez efektu. Nie zakładamy limitu ośmiu kierunków ani tego, że jeden kierunek wyczerpuje pojęcie. Skala kierunku jest ustalana na danych fit i zamrożona; normalizacja nie usuwa wymiarów.

## 4. Interwencje, które wykonujemy osobno

### A. Zmiana pojedynczej cząstki lub kanału

Dla wybranej pozycji p i jednostkowego kierunku v:

    Z'(x) = Z(x) + alpha * e_p v^T.

Dla pojedynczego kanału j przyjmujemy v=e_j. Porównujemy dodatnie i ujemne alpha. Przy interwencji na wszystkich pozycjach narzędzia używamy W_p=m_p v, a przy kontroli tła odpowiednio dopasowanego przestrzennego wsparcia poza narzędziem. Wyrównujemy normę Frobeniusa i liczbę zmienianych pozycji. Jest to inny test niż pojedynczy token i otrzymuje osobny identyfikator.

### B. Podmiana wartości kierunku z pary obrazów

Dla obrazu bazowego b i obrazu źródłowego s przy tym samym checkpoincie:

    z'_p(b) = z_p(b) + v v^T [z_p(s) - z_p(b)].

Pozostałe współrzędne zachowujemy. Dla zmiany samego tła piksele narzędzia i pozycje mogą być identyczne. Dla zmiany pozycji lub obrotu potrzebna jest jawna mapa odpowiadających sobie miejsc; jej interpolacja wnosi błąd, który mierzymy osobną kontrolą. Hybrydowy latent nie musi odpowiadać żadnemu prawdziwemu obrazowi nawet przy podmianie z rzeczywistych aktywacji.

Testujemy oba kierunki podmiany. Oczekiwany skutek określamy przed obliczeniem: zmiana tła powinna zachować obecność i ramkę; zmiana pozycji powinna przemieszczać ramkę zgodnie z transformacją. Podmiana części latentu nie ma z góry gwarantowanego prawdziwego odpowiednika. Nie przypisujemy jej dowolnej etykiety, aby uzyskać korzystny wynik.

### C. Wygaszenie składowej i kontrola uszkodzenia

Wygaszenie względem średniej fit mu:

    z'_p = z_p - v v^T(z_p - mu).

To osobny eksperyment usuwania informacji, nie preprocessing całej analizy. Obowiązkowo porównujemy z usunięciem losowego kierunku o takim samym wsparciu i podobnej energii edycji. Brak efektu nie dowodzi nieistotności: informacja może być powielona, nieliniowa lub płynąć inną drogą. Spadek jakości nie dowodzi wyłączności narzędzia: mogły ucierpieć inne cechy. LEACE jest możliwym późniejszym wariantem diagnostycznym, nie zastępuje obecnie pełnego stanu ani nie jest automatycznie wdrażany.

### D. Przyrost Koopmana

Przyrost delta_hat = Z_hat_(k+h)(x)-Z_k(x) jest wyznaczany bez przyszłych danych oceny. Podajemy Z_k+delta_hat do zamrożonej dalszej części checkpointu k. Horyzont podstawowy: 100 aktualizacji. Zachowujemy pełny tensor.

Mierzymy także zmianę tylko w wybranym kierunku jako analizę wpływu, ale nie zastępujemy nią pełnej prognozy. Przy niestabilnym lub niefinitywnym przyroście zgłaszamy nieudaną prognozę. Nie dobieramy znaku ani skali z końcowych etykiet. Skalowane warianty są osobną analizą z ustawieniami zatwierdzonymi na wcześniejszych danych.

## 5. Miary wpływu i poprawy

Definiujemy stratę dla zamrożonego checkpointu:

    L_k(Z;x,y) = loss_det(H_k(Z,R_k(x)), y).
    effect(V,alpha;x) = L_k(Z+alpha V;x,y) - L_k(Z;x,y).

Wynik ujemny oznacza poprawę tej straty przy tej interwencji. Osobno podajemy klasyfikację, błąd ramki, pominięcia i fałszywe alarmy. Zbiorcze AP/mAP liczymy na zbiorze, nie przypisujemy „AP pojedynczej cząstce”. Próg pewności do miar dyskretnych zamrażamy wcześniej.

W detektorze opartym na zbiorze zapytań trzeba kontrolować przyporządkowanie zapytań do obiektów. Dla gradientu używamy określonego, zamrożonego lokalnie przyporządkowania i różniczkowalnej straty pomocniczej. Dla końcowej oceny ponownie wykonujemy normalną ocenę detekcji, dopuszczając zmianę zapytań. Raportujemy rozbieżność obu wyników. Nie stosujemy gradientu przez dyskretne progi ani nie dowodzimy poprawy mAP samą pochodną logitu.

Wrażliwość kierunkowa:

    s_V(x) = <grad_Z L_k(Z;x,y), V>_F.

Porównujemy ją z rzeczywistymi różnicami skończonymi dla kilku małych dodatnich i ujemnych amplitud. Wzrost logitu narzędzia bez poprawy lokalizacji lub z większą liczbą fałszywych alarmów nie jest uznawany za poprawę detekcji.

## 6. Twierdzenie warunkowe i dowód

Niech L będzie różniczkowalna na odcinku od Z do Z+delta, a jej gradient będzie beta-Lipschitz na tym odcinku w normie Frobeniusa. Wtedy:

    L(Z+delta)-L(Z) <= <grad L(Z),delta> + beta/2 * ||delta||_F^2.

Dowód: różnica strat jest całką od 0 do 1 z <grad L(Z+t delta),delta>. Odejmując <grad L(Z),delta>, stosując nierówność Cauchy'ego-Schwarza i ograniczenie Lipschitza, dostajemy całkę beta*t*||delta||², równą beta*||delta||²/2.

Zatem ujemna prawa strona jest wystarczającym warunkiem zmniejszenia L. Dla przyrostu alpha V, przy ||V||=1 i s_V<0, wystarcza 0<alpha<-2*s_V/beta. Bez certyfikowanego beta sam znak pochodnej daje lokalną wskazówkę, nie gwarancję dla wybranej dużej edycji.

Jeżeli prognoza delta_hat przybliża rzeczywisty przyrost delta z błędem ||delta-delta_hat||<=epsilon, to pod tymi samymi założeniami dla rzeczywistego odcinka:

    L(Z+delta)-L(Z)
      <= <grad L(Z),delta_hat> + ||grad L(Z)||*epsilon
         + beta/2 * (||delta_hat||+epsilon)^2.

Wynika to z poprzedniej nierówności, Cauchy'ego-Schwarza i nierówności trójkąta. Warunek ujemności całej granicy łączy jakość prognozy z korzystnym wpływem na zamrożony odczyt. Empiryczny błąd na kilku checkpointach nie jest automatycznie gwarantowaną granicą epsilon dla nowego obrazu. Ograniczenie krzywizny oszacowane numerycznie również nie jest certyfikatem.

Dowód dotyczy określonej straty i zamrożonej dalszej sieci, nie całego przyszłego treningu, nie globalnej optymalności, nie mAP i nie pełnej tożsamości semantycznej. Nie został jeszcze sformalizowany w Lean.

## 7. Oddzielenie rozwoju latentu od zmiany reszty detektora

Między checkpointami zmienia się także H oraz ewentualne R. Dla odłożonej przyszłości porównujemy obserwacyjnie:

    L_k(Z_k), L_k(Z_(k+h)), L_(k+h)(Z_k), L_(k+h)(Z_(k+h)).

W każdym wywołaniu zachowujemy jawnie określone dodatkowe wejścia R, a gdy różnią się między checkpointami, raportujemy je jako dodatkowy czynnik. Krzyżowe podmiany wymagają zgodnych warstw, normalizacji i współrzędnych; mogą być nienaturalne. Służą diagnozie i nie są zwykłym wykonaniem detektora. Wyniku z przyszłych wag nie używamy do wyboru kierunków przed prognozą.

Poprawa samego latentu pod zamrożonym H_k jest innym stwierdzeniem niż poprawa całego modelu. Możliwa jest współadaptacja: lepszy późniejszy detektor może mieć reprezentację gorzej czytaną przez stary H_k.

## 8. Kontrole i werdykt

- Brak edycji i ponowne wstawienie identycznego tensora muszą odtwarzać wynik; restore po każdej próbie.
- Losowy kierunek: ten sam obszar, liczba edytowanych współrzędnych i norma. Minimum kilka ustalonych seedów; budżet zapisać przed oceną.
- Przeciwny kierunek oraz równoważna interwencja w tle; nie zakładać, że tło nie ma przyczynowego wpływu.
- Podmiana bez zmiany semantycznej i pary o znanym efekcie wejściowym.
- Kontrola dopasowania kierunków z pomieszanymi etykietami; permutacje na poziomie źródeł, nie dowolnych zależnych tokenów.
- Wariancja, norma, NaN/Inf, wyjście poza rozkład aktywacji i zgodność znaku pochodnej z małą edycją.
- Raportować osobno warunki kąta, światła i tła. Bootstrap grupami filmów; uwzględnić wielokrotne testowanie wielu kierunków. Kandydatów i amplitud nie wybierać na końcowych wynikach.

Dla kierunku zapisujemy odrębne statusy: informacja odczytywalna; lokalna wrażliwość; zmierzony wpływ interwencji; korzystny wpływ; prognozowany korzystny wpływ potwierdzony później. Nie nadajemy automatycznie statusu „tożsamość wyłącznie” ani „domknięty”. Brak rozstrzygnięcia może oznaczać małą moc testu, a nie brak roli.

Ocena domknięcia wymaga dodatkowo zadanego zakresu warunków i tolerancji błędu, dostatecznej poprawności oraz testu informacji w pozostałych współrzędnych. Nie ustalamy progów po zobaczeniu wyniku.

## 9. Powiązanie z Langevinem i resztą

Dla osi czasu uczenia dopuszczamy późniejszy model Z_(k+1)=A Z_k+b+sum_j K_j Z_(k-j)+B u_k+epsilon_(k+1). Sterowanie opisuje dane, augmentacje i ustawienia kolejnych aktualizacji, jeśli są dostępne przed prognozą. Przy niestacjonarnym treningu nie przenosimy bez sprawdzenia założeń stacjonarnego układu fizycznego. Nawet pełny tensor sond nie stanowi pełnego stanu wag, optymalizatora i generatora liczb losowych.

Resztę można oceniać zarówno normą, jak i wrażliwością straty: <grad L,epsilon>. Mały błąd latentu w czułym kierunku może mieć duży wpływ na detekcję. To proponowana diagnoza, nie twierdzenie z prac MZ o semantyce narzędzia. Biała reszta nie musi być gaussowska, niezależna ani mała. Brak wykrytej zależności nie dowodzi jej nieistnienia.

## 10. Literatura: dziesięć pełnych odczytów przez Undermind

Wykonano kolejne wyszukiwania semantyczne, identyfikację konkretnych prac oraz zbiorczy read_pdfs dla 11 pozycji. Dziesięć zwróciło odczyt PDF; Rag17 (SVCCA) zgłosiło brak PDF i nie jest wliczane do dziesięciu. To synteza odczytów przez narzędzie; jego interpretacje sprawdzano pod kątem spójności, nie traktowano jako gotowych twierdzeń o RF-DETR. Daty konferencji odróżniono od dat preprintów.

| Klucz, autorzy, rok | Tytuł i źródło | Użycie w protokole; ograniczenie |
|---|---|---|
| Kim17, Kim i in., ICML 2018 | [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors](https://arxiv.org/abs/1711.11279) | §3.2–3.5: kierunek pojęcia i pochodna logitu. Mierzy lokalną wrażliwość; nie dowodzi poprawy wykrywania. |
| Bau17, Bau i in., CVPR 2017 | [Network Dissection: Quantifying Interpretability of Deep Visual Representations](https://doi.org/10.1109/CVPR.2017.354) | §2.2: zgodność map aktywacji z maskami; §3: zależność interpretacji od bazy. Zgodność przestrzenna nie jest przyczynową koniecznością. |
| Yeh19, Yeh i in., NeurIPS 2020 | [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969) | §3–4: wystarczalność zbioru pojęć i wkład poszczególnych pojęć. Odtworzenie decyzji przez dodatkowy odczyt nie dowodzi mechanizmu oryginalnej sieci. |
| Bel23, Belrose i in., 2023 | [LEACE: Perfect linear concept erasure in closed form](https://doi.org/10.48550/arXiv.2306.03819) | §3–4: formalna definicja usuwania informacji względem klasy predyktorów i strat; §5.3: interwencje. Nie usuwa gwarantowanie informacji dostępnej nieliniowo. |
| Gei21, Geiger i in., NeurIPS 2021 | [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997) | §3, §5: podmiana aktywacji i zgodność z modelem przyczynowym. Trafność na skończonej próbie nie dowodzi zgodności dla wszystkich obrazów. |
| Gei21b, Geiger i in., ICML 2022 | [Inducing Causal Structure for Interpretable Neural Networks](https://arxiv.org/abs/2112.00826) | §3 i dodatek A: zerowa odpowiednia strata na wszystkich wymaganych parach daje warunkową relację abstrakcji. U nas nie uruchamiamy IIT ani nowego uczenia wag. |
| Dob25, Dobrzeniecka, Fokkens, Sommerauer, ACL Findings 2025 | [Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE](https://doi.org/10.18653/v1/2025.findings-acl.674) | §2–3: kontrola przypadkowego uszkodzenia przy usuwaniu informacji. Wyniki językowe, wymagają przeniesienia i walidacji w detektorze. |
| Lin21b, Lin, Tian, Anghel, Livescu, 2021 | [Data-Driven Learning for the Mori–Zwanzig Formalism: A Generalization of the Koopman Learning Framework](https://doi.org/10.1137/21M1401759) | §3–5: opis obserwabli przez człon bieżący, pamięć i resztę. Dotyczy dynamiki; nie certyfikuje tożsamości narzędzia. |
| Kor19, Kornblith i in., ICML 2019 | [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) | §2 i miara CKA: ostrożność przy porównaniach reprezentacji o większym wymiarze niż liczba przykładów. Podobieństwo nie oznacza poprawności. Można liczyć macierze Grama bez PCA stanu. |
| Gei23, Geiger, Wu, Potts, Icard, Goodman, preprint 2023 / CLeaR 2024 | [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://doi.org/10.48550/arXiv.2303.02536) | §3: podmiany w rozproszonych kierunkach, mierzenie zgodności interwencji. Nie trzeba utożsamiać pojęcia z jednym neuronem. U nas nie wdrażamy ogromnej macierzy obrotu ani kompresji. |

Uwagi audytowe do odczytów: 576 x 384 oznacza 576 tokenów i 221184 liczb, nie 221184 tokenów. W LEACE „brak liniowej informacji” odnosi się do formalnie określonej rodziny predyktorów i wypukłych strat, nie do nieograniczonej wiedzy modelu. W MZ brak domknięcia dynamiki nie jest synonimem braku semantycznego. Twierdzenia o dokładnej abstrakcji wymagają odpowiedniego zakresu wszystkich interwencji, nie tylko stu procent trafności w skończonej próbie.

## 11. Artefakty przyszłego wykonania i granica zakresu

Zapis: manifest podziałów i par, pełne identyfikatory checkpointów i warstwy, kierunki oraz skale fit, normy i amplitudy edycji, wyniki przed/po, kontrolne seedy, błąd prognozy, tabelę wyników według warunków, przedziały niepewności i osobny raport wyników negatywnych. Przed pełnym testem konieczne są testy poprawności podmiany na małym sztucznym układzie, zerowej edycji oraz zgodności gradientu z różnicą skończoną.

Dokument dodaje zaplanowany test badawczy. Nie stanowi informacji, że hook do interwencji został zaimplementowany lub uruchomiony. Nie uruchamia dodatkowej kampanii ani nie opóźnia automatycznie wcześniej zleconego pobrania i pokazu FastGAN. Koszt liczby kierunków, par i checkpointów trzeba określić przed wykonaniem; nie obiecywać wyczerpującej analizy wszystkich kombinacji.
