# Prognoza 128 natywnych kanałów do 500 aktualizacji: literatura i audyt protokołu

Data: 2026-10-02. Autor: Astra, niezależny research i audyt. Status: **FINAL — literatura, protokół, kod oraz wyniki zamrożonego testu zweryfikowane; cel 85 procent nie został osiągnięty**. Poprawność obliczeń jest oddzielona od negatywnego wyniku celu prognostycznego. Chronologię doprecyzowań i weryfikacji opisano poniżej. Własność autora obejmuje tylko ten plik. Autor nie uruchamiał fitów, forwardów, treningu ViT ani nowych pobrań danych; niezależne przeliczenia poniżej służyły wyłącznie kontroli już zapisanych wyników.

## Rekomendacja

Najbardziej uzasadniona pierwsza próba to **wygaszany trend lub lokalny model stanu poziom–prędkość, następnie GLE z kilkoma czasami pamięci i stabilna regresja w oryginalnych 128 kanałach**. Należy porównać je w autonomicznych prognozach z identycznej historii. Literatura nie pozwala obiecać 85 procent ani wyznaczyć takiego horyzontu z samej nazwy modelu. Obserwator może poprawić oszacowanie stanu na początku; po odcięciu pomiarów nie ma korekt, które usuwałyby przyszłe błędy.

W poprzednim eksperymencie 92.83 procent oznaczało rozpoznanie oryginału versus LaMa w znanym ROI przy k500. Nie jest to wynik prognozy dynamiki. Nowy cel, potwierdzony przez roota, to **znak skumulowanego przemieszczenia każdej ze 128 współrzędnych**, bez odrzucania małych ruchów. Dodatni skill MSE jest osobnym warunkiem użyteczności przewidywanej wielkości.

## Jak przeprowadzono research

Zastosowano [paper-search SKILL.md](C:/Users/bartl/.agents/skills/paper-search/SKILL.md). Wykonano wyszukiwanie CLI `paper-search` w arXiv/Crossref; jego sześć wyników początkowych zawierało mało trafne trafienia i nie stanowi podstawy rekomendacji. Użyto internetowego wyszukiwania i odczytu źródeł pierwotnych oraz callable Undermind, a nie tylko wyszukania metadanych.

Przejrzano istniejące wyszukiwanie o wielokrokowym prognozowaniu latentu z 2026-09-29. Dotyczyło skalarnego rzutu i bloków czasu, więc nowe ograniczenie 128 natywnych osi i prefiksu do 500 uzasadniało jeden dodatkowy, celowany deep search. [Nowy Undermind Deep Search](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Long%20horizon%20native%20coordinate%20latent%20forecasting%20with%20short%20history%20and%20no%20future%20measurements) ukończono 2026-10-02; wyniki obejmują 175 kandydatów. To ranking z abstraktów i metadanych, nie 175 przeczytanych artykułów. Osobno zlecono pełnotekstowy odczyt Lei, Lin i Mamakoukas oraz Uy; kluczowe fragmenty dodatkowo czytano w HTML/PDF autorów. Narzędzie nie udostępniało ustawienia budżetu/liczby kredytów ani informacji o koszcie; uruchomiono tylko jedno wyszukiwanie w ramach istniejącego upoważnienia, bez zakupu usługi.

## Źródła pierwotne i rzeczywisty zakres odczytu

| Źródło | Dowód istotny dla projektu | Zakres odczytu i ograniczenie |
|---|---|---|
| R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems* (1960), [DOI 10.1115/1.3662552](https://doi.org/10.1115/1.3662552), [PDF kopii akademickiej](https://people.duke.edu/~hpgavin/SystemID/References/Kalman-JBE-1960.pdf) | Rozróżnia filtrację, wygładzanie i predykcję warunkowaną obserwacjami. | Odczyt początkowych sekcji PDF: *Optimal Estimates*, *Orthogonal Projections*, definicja rozkładu warunkowego i twierdzenie 1-a; część kolejnych odczytów PDF zwracała timeout. Nie przypisujemy tej pracy weryfikacji konkretnego modelu latentu. |
| E. S. Gardner Jr., E. McKenzie, *Forecasting Trends in Time Series* (1985), [DOI 10.1287/mnsc.31.10.1237](https://doi.org/10.1287/mnsc.31.10.1237) | Pierwotny model wygaszania trendu. | Metadane i abstrakt wydawcy; pełny tekst tej wersji nie był dostępny przez Undermind. Równania sprawdzono w późniejszej pracy autorów poniżej. |
| E. McKenzie, E. S. Gardner Jr., *Damped trend exponential smoothing: A modelling viewpoint* (2010), [DOI 10.1016/j.ijforecast.2009.07.001](https://doi.org/10.1016/j.ijforecast.2009.07.001), [PDF autorów](https://www.bauer.uh.edu/gardner/Damped-trend-Modelling.pdf) | Tłumiony trend jako model stanu; parametr tłumienia opisuje trwałość nachylenia. | Pełny tekst preprintu, sekcja 2, równania (5)–(8), oraz wnioski. Strony PDF 5–8; nie mylić z numeracją stron wersji czasopisma. |
| M. O. Williams, I. G. Kevrekidis, C. W. Rowley, *A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition* (2015), [arXiv 1408.4408](https://arxiv.org/abs/1408.4408) | EDMD to regresja w zadanym słowniku obserwabli; skończony słownik daje przybliżenie. | Abstrakt i źródło pełnotekstowe HTML. Natywne kanały mogą być słownikiem, ale ich zamknięcia względem rzeczywistej dynamiki nie wykazano. |
| G. Mamakoukas, I. Abraham, T. D. Murphey, *Learning Stable Models for Prediction and Control*, [DOI 10.1109/TRO.2022.3228130](https://doi.org/10.1109/TRO.2022.3228130), [arXiv 2005.04291](https://arxiv.org/abs/2005.04291) | Dopasowywanie stabilnej macierzy z uwzględnieniem błędu na danych; stabilność nie wystarcza do kontroli przejściowego wzrostu. | Pełnotekstowe `read_pdfs` oraz bezpośredni HTML: III-A, IV-A, równania (14), (27)–(30), V-A. Klucz Undermind `Mam20` odzwierciedla rok preprintu, nie rok finalnej publikacji. Nie przenosimy ich przykładów na gwarancję H500. |
| S. Dahdah, J. R. Forbes, *System Norm Regularization Methods for Koopman Operator Approximation*, [DOI 10.1098/rspa.2022.0162](https://doi.org/10.1098/rspa.2022.0162), [arXiv 2110.09658](https://arxiv.org/abs/2110.09658) | Rozdziela ridge, ograniczenia stabilności i regularyzację wzmocnienia systemu. | Pełny HTML, sekcje 4–5, równania (27)–(33). Część problemów jest niekonweksem BMI. Sekcja 6 używa redukcji wymiaru, której tutaj nie przejmujemy. Autorzy ujawniają pominięty rozbieżny fragment; w naszym teście takich fragmentów nie wolno usuwać. |
| H. Lei, N. A. Baker, X. Li, *Data-driven parameterization of the generalized Langevin equation* (2016), [DOI 10.1073/pnas.1609587113](https://doi.org/10.1073/pnas.1609587113) | Wymierne jądro pamięci można realizować przez pomocnicze zmienne Markowa. | Pełnotekstowe `read_pdfs`; równania (11)–(18), części *Rational Approximations*, *Extended Dynamics*. Wykorzystujemy konstrukcję pamięci; nie zakładamy równowagi termodynamicznej ani FDT dla uczenia ViT. |
| Y. T. Lin, Y. Tian, M. Anghel, D. Livescu, *Data-Driven Learning for the Mori–Zwanzig Formalism: A Generalization of the Koopman Learning Framework* (2021), [DOI 10.1137/21M1401759](https://doi.org/10.1137/21M1401759), [arXiv 2101.05873](https://arxiv.org/abs/2101.05873) | Rozdziela operator Markowa, pamięć i nierozwiązany składnik; odróżnia projekcję Moriego od regresji po osadzonych opóźnieniach. | Pełnotekstowe `read_pdfs` oraz HTML; sekcje 3.7, 5, 6.2, 7; równania (41)–(43), (52). Założenia analizy i eksperymentów nie są automatycznie spełnione przez jeden niestacjonarny bieg uczenia. |
| D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, *Next generation reservoir computing* (2021), [DOI 10.1038/s41467-021-25801-2](https://doi.org/10.1038/s41467-021-25801-2) | NVAR: opóźnione współrzędne i jawne nieliniowe cechy, z liniowo uczonym wyjściem i autonomicznym podawaniem prognoz na wejście. | Pełny HTML wydawcy, równania (3), (5), (9), (11), opis treningu i testu. Przykłady o małym wymiarze nie dowodzą, że pełny kwadratowy słownik 128 kanałów jest oszczędny. |
| W. I. T. Uy, D. Hartmann, B. Peherstorfer, *Operator inference with roll outs for learning reduced models from scarce and low-quality data*, [arXiv 2212.01418](https://arxiv.org/abs/2212.01418) | Strata wielokrokowych autonomicznych rolloutów może ograniczać błędy niewidoczne w stracie jednego kroku. | Pełny HTML: sekcje 2–3, równania (8)–(10); zakończony dodatkowy odczyt `read_pdfs`. Praca używa zredukowanych współrzędnych; tutaj zapożyczamy tylko zasadę celu rollout, zachowując natywne kanały. |
| M. Marcellino, J. H. Stock, M. W. Watson, *A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series* (2006), [DOI 10.1016/j.jeconom.2005.07.020](https://doi.org/10.1016/j.jeconom.2005.07.020), [PDF autorów](https://swh.princeton.edu/~mwatson/papers/Marcellino_Stock_Watson_FinalVersion_hstep.pdf) | Bezpośrednia prognoza horyzontowa i iteracja mają różny kompromis obciążenia i wariancji. | Odczyt abstraktu i wprowadzenia pełnego PDF. Nie ma uniwersalnej przewagi prognozy bezpośredniej; jej wyniki makroekonomiczne nie ustanawiają prognozowalności latentu. |

Literatura wspiera **konstrukcję kandydatów i kontrole**, a nie wynik procentowy. Dodatkowe trafienia Undermind dotyczące identyfikacji podprzestrzeni i modeli z autoenkoderem pozostają źródłami kontekstu; nie są propozycją obejścia zakazu PCA.

## Sześć konkretnych kandydatów bez redukcji kanałów

Poniższe równania to **proponowane adaptacje**, nie dosłowne modele zweryfikowane w artykułach dla tego ViT. Nie stanowią jeszcze zamrożonej siatki implementacji. Niech `z_i(t)` oznacza pełny wektor 128 wybranych współrzędnych cząsteczki, a `v_i(t)=z_i(t)−z_i(t−1)`. Makrowektory pozostają oryginalnymi osiami; zmieniają się współrzędne i mikrowektory cząsteczek.

| Kandydat | Konkretna konstrukcja | Dlaczego pasuje i co może zawieść |
|---|---|---|
| 1. Wygaszany trend zakotwiczony w pomiarze | `zhat(t+H)=z(t)+vhat(t) sum_{h=1..H} phi^h`; `vhat` z historycznego okna lub jednostronnego EMA, `0<phi<=1`. | Mało parametrów; ogranicza katastrofalne przedłużanie ostatniej prędkości. Trend może zmienić znak po originie. To obowiązkowy mocny punkt odniesienia, a nie tylko wariant dekoracyjny. |
| 2. Kalman poziom–prędkość w każdej oryginalnej osi | `ell(t+1)=ell(t)+phi v(t)+eta_l`, `v(t+1)=phi v(t)+eta_v`, `y(t)=ell(t)+nu`. Wspólne ustawienia tłumienia/ilorazów wariancji, osobne stany dla tokenów i kanałów. | Estymuje wygładzoną prędkość ze wszystkich pomiarów prefiksu. Po originie tylko predykcja. Warianty R i Q są założeniem statystycznym, nie fizycznym pomiarem szumu sensora w deterministycznym odczycie aktywacji. |
| 3. Kanałowa relaksacja do równowagi | `zhat_j(t+1)=mu_j+rho_j(zhat_j(t)−mu_j)`, `0<=rho_j<=1`; równowaga i współczynniki wyłącznie z prefiksu, z silnym shrinkage. | Bardzo tani model nasycania. Nie zna przyszłego optimum treningu. Błędne `mu` może dać stały błędny znak przez całe H500. Nie wolno przyjąć końcowego z1000 jako równowagi. |
| 4. GLE z 2–4 pamięciami wykładniczymi | `m_r(t)=a_r m_r(t−1)+(1−a_r)v(t)`, `a_r=exp(−1/tau_r)`; `vhat(t+1)=b+A xhat(t)+sum_r K_r mhat_r(t)`, `zhat(t+1)=zhat(t)+vhat(t+1)`. | Zachowuje 128 współrzędnych; pomocnicze stany przechowują historię. Pierwszy wariant diagonalny po kanałach ogranicza liczbę parametrów; pełne macierze wymagają silnego ridge. Czasy pamięci ustala/wybiera wyłącznie walidacja prefiksu. Stabilność samych `a_r` nie gwarantuje stabilności całego połączonego układu. |
| 5. Stabilny natywny Koopman/DMD | `zhat(t+1)=c+F zhat(t)` lub stabilny model prędkości; wszystkie 128 osi w wejściu i wyjściu. Dopasowanie ogranicza wzrost operatora, a intercept pozostaje osobny. | Może usunąć eksplozję poprzednich operatorów. Zbyt silna kontrakcja zgasi rzeczywisty trend. Operację przycięcia widma po zwykłym ridge trzeba nazwać stabilizacją heurystyczną, nie odtworzeniem DISKO. Nie narzucać kontrakcji sztucznej stałej `1` w afinicznym rozszerzeniu. |
| 6. Mała natywna NVAR | `vhat(t+1)=W psi(z(t),v(t),v(t−s))`, z oryginalnymi kanałami, kilkoma opóźnieniami i uprzednio wybranymi funkcjami współrzędnych. | Pozwala sprawdzić nieliniowość bez PCA i autoenkodera. Pełne iloczyny wszystkich opóźnionych osi szybko dają dziesiątki tysięcy cech; zalecana początkowo struktura kanałowa lub bardzo mały słownik. Prognozowane kanały i liczba celów nadal wynoszą 128. Nieliniowe cechy mogą wybuchać poza zakresem prefiksu. |

Priorytet wykonania: 1–4, następnie 5; kandydat 6 jest rezerwą, jeśli został wpisany do siatki przed testem. Model bezpośredni dla każdego H jest alternatywnym sposobem dopasowania, nie dodatkowym obowiązkowym siódmym modelem. Przy prefiksie 0..500 i dodatniej długości historii nie ma niezależnego zestawu treningowych par dla bezpośredniego H500. Współdzielona parametryczna funkcja horyzontu może ekstrapolować do 500, ale nie jest tym samym co nauczenie H500 na wielu przykładach.

Stan pomocniczy pozycji/prędkości/pamięci nie zastępuje obserwowanego stanu nową wyuczoną bazą. Wprowadzanie niskiej rangi, SVD lub stanów podprzestrzennych nie jest tu potrzebne. Numeryczne rozwiązywanie układu liniowego i kontrola widma macierzy nie oznaczają zmiany współrzędnych celu.

## Co obserwator może zrobić bez przyszłych pomiarów

Dla liniowego modelu pomocniczego:

```text
s_(t+1) = F s_t + c + w_t
y_t     = C s_t + v_t
shat_(t|t) = shat_(t|t-1) + K_t (y_t - C shat_(t|t-1))
```

Po odcięciu obserwacji w `t0` poprawna prognoza średniej i kowariancji ma postać:

```text
shat_(t0+H|t0) = F^H shat_(t0|t0) + sum_(h=0..H-1) F^h c
P_(t0+H|t0)    = F^H P_(t0|t0) (F^H)^T
                  + sum_(h=0..H-1) F^h Q (F^h)^T
```

Wzór zakłada modelowe zerowe średnie przyszłych innowacji i ustalone parametry. Nie obejmuje automatycznie błędu estymacji F/Q ani zmiany reżimu uczenia. Dla obserwatora Luenbergera po zniknięciu pomiarów nie ma członu `L(y−C shat)`; zamknięta macierz błędu `F−LC` nie jest wtedy operatorem otwartej prognozy. Stabilny observer ze stale dostarczanymi pomiarami nie stanowi dowodu skuteczności autonomicznego H500.

Jeżeli używa się modelu kolorowej reszty `e(t+1)=R e(t)+xi(t)`, dostępny jest tylko warunkowy wkład `R^H ehat(t0)` i propagacja niepewności. Rzeczywiste przyszłe residuale są zabronione. Samo losowanie dodatkowego szumu nie poprawia optymalnej prognozy średniej pod stratą kwadratową; służy modelowaniu rozkładu. Losowania nie są niezależnymi przyszłymi realizacjami prawdziwego treningu.

W filtrze należy jawnie ustalić punkt zakotwiczenia. Prognoza surowego odczytu `C shat(t0+H)` i prognoza przemieszczenia dodana do rzeczywistego `z(t0)`, czyli `z(t0)+C[shat(t0+H)−shat(t0)]`, są różnymi modelami, jeśli filtr nie przechodzi dokładnie przez pomiar. Drugi wariant może być rozsądny dla tego zadania, ale nie jest automatycznie optymalnym predyktorem Kalmana z założonym modelem szumu pomiarowego.

## Protokół: co można uczciwie zbadać

1. **Niezmienny J i dane.** Zachować SHA modelu wyboru128 kanałów i kolejność J0/J1. Celami są raw współrzędne w `DYNAMIKA_NATYWNE_J.npz` albo dokładnie te same indeksy z mmap gęstego latentu. Bez nowych kierunków, wyboru łatwych tokenów, maskowania małych celów i bez importowania sterowań z innego biegu. To ruch reprezentacji stałego obrazu wskutek aktualizacji sieci i BN.
2. **Origin 500 jest legalny dla ustalonych J500.** Wszystkie parametry i stany prognozy mogą korzystać z danych 0..500. Wewnętrzne originy np. 250 i 300 z H1/10/50/100/200 są retrospektywnym wyborem metody w prefiksie dostępnym w 500. Centra, skale, równowagi, residuale i filtry muszą jednak korzystać wyłącznie z właściwego prefiksu do danego originu. Kotwica z500 w walidacji300 byłaby przeciekiem poziomu tokenu.
3. **Nie ukrywać warm-upu.** Historia opóźnień i inicjalizacja EMA muszą być dostępne w każdej walidacji. Wszystkie modele porównuje się na tych samych startach i celach. Jeśli wprowadzimy krótkie okno lokalnego treningu, jego długość również należy zamrozić. Obcięcie pierwszej fazy uczenia może być hipotezą, nie decyzją po wyniku1000.
4. **Wybór wielohoryzontowy.** Nie wystarczy zminimalizować one-step residual. Zapisać skończoną siatkę i równą wagę zadeklarowanych H. Regułę selekcji trzeba ustalić przed fitami: przykładowo najpierw dopuszczalność dodatniego średniego skillu MSE, następnie znak, remisy przez MSE i prostotę. Jawnie rozróżnić bramkę średnią od dodatniego skillu na każdym H. Jeśli nikt nie przechodzi warunków, można wskazać najlepszy dostępny, ale nie ogłosić osiągnięcia celu.
5. **Zamrożenie i rollout.** Po wyborze refit<=500, zapis konfiguracji, SHA i prognozy przed oceną501..1000. Pełny rollout bez teacher forcing; przyszłe fg/bg, EMA i lags powstają tylko z prognoz. Interpolacja między prawdziwymi checkpointami, globalne wygładzanie oraz filter/smoother uruchomiony na1001 stanach są zabronione jako wejście.
6. **Próba i transfer.** Ustalić obraz lub obrazy wyboru przed fitami; transfer na sondę8 lub inne obrazy oceniać po zamrożeniu. Wykorzystanie historii innego obrazu do inicjalizacji obserwatora jest dozwolone tylko jako jawny sposób inicjalizacji, nie ponowny dobór hiperparametrów. Wiele obrazów z tego samego treningu nie daje wielu niezależnych seedów sieci.
7. **H500 i niepewność.** Od origin 500 istnieje jeden terminalny cel 1000. Nie da się z niego uzyskać przedziału po niezależnych originach czasu. Nie bootstrapować 128 kanałów lub 31 tokenów jako niezależnych prób. Krótsze H można badać na dodatkowych późniejszych originach, lecz ich lokalne inicjalizacje nie mogą zastępować jednego autonomicznego500-krokowego przebiegu. To osobny rolling-origin eksperyment.
8. **Historia była już oglądana.** Nowy kod może być przyczynowy informacyjnie, ale kolejne rodziny wybierane po wcześniejszych porażkach korzystają z wiedzy badacza o tej trajektorii. Status: eksploracyjny wynik na znanym biegu. Potwierdzenie85 procent wymaga później niezależnej trajektorii lub dalszego zapisu; obecny research nie autoryzuje nowego treningu.

## Metryki i pułapka 85 procent

Dla `d_i,H=z_i(t0+H)−z_i(t0)` i `dhat_i,H=zhat_i(t0+H)−z_i(t0)`, kanałów J i wag tokenów `w_i=coverage_i/sum coverage`:

```text
S(H) = sum_i w_i (1/128) sum_j 1[sign(dhat_i,H,j) = sign(d_i,H,j)]
MSE(H) = sum_i w_i ||dhat_i,H - d_i,H||^2 / 128
skill(H) = 1 - MSE_model(H) / MSE_persistence(H)
```

Przy wielu obrazach najpierw liczyć miary wewnątrz obrazu, potem ich średnią z równymi wagami. Zdefiniować dokładne zera symetrycznie (`sign(0)=0`); niefinitywna prognoza jest niepowodzeniem, nie powodem usunięcia wiersza. Gdy mianownik MSE persystencji jest zerowy, raportować brak zdefiniowanego skillu, nie sztuczne100 procent. Progi near-zero mogą być tylko dodatkową diagnostyką z prefiksu i z podanym coverage; główne85 procent obejmuje wszystkie 128 współrzędnych.

Minimalne punkty odniesienia: persystencja, ostatnia prędkość, średni przyrost z historycznego okna, tłumiony trend, kanałowa relaksacja. Dla samego znaku także historyczny znak większości lub znak średniego historycznego przemieszczenia na danym H; oszacowanie tylko z prefiksu, bez wyznaczania większości z testu. Wysoki udział jednego znaku należy ujawnić per kanał i zbiorczo.

Znak jest nieczuły na amplitudę. Prognoza `z(t0)+epsilon*sgn` może mieć dobry wynik znaku i znikomą wartość metryczną, a nawet nieznacznie dodatni skill, gdy epsilon jest bardzo małe. Dlatego dodatni MSEskill jest konieczną, lecz słabą bramką: obowiązkowo porównać też wielkość z najlepszym prostym baseline, normę prognozowanego/prawdziwego przemieszczenia, cosinus i poziom błędu. Nie podbijać znaku przez arbitralny epsilon po teście.

Raportować krzywe S(H) i skill(H) dla całego 1..500, tabelę ustalonych H oraz osobno `Hmax`, dla którego warunek spełniono punktowo. Zdanie „działa do H” wymaga podania, czy warunek zachodzi na całym ocenionym zakresie, czy tylko na jednym końcu. Wysoki znak skumulowanego ruchu nie dowodzi odtworzenia każdego z 500 jednokrokowych przyrostów; tę drugą trafność można policzyć osobno bez zmiany celu głównego.

## Kontrola stabilności i ograniczenia interpretacji

Ridge nie gwarantuje stabilności. Dla modelu z pamięcią kontrola dotyczy całej macierzy rozszerzonego stanu, nie tylko dryfu A lub każdego jądra z osobna. Sprawdzić propagację błędu na H<=500, skończoność trajektorii oraz wzmocnienie potęg operatora; operator nienormalny może silnie wzmacniać błędy przejściowo pomimo rho<1. Dla modelu poziom–prędkość neutralny poziom z wartością własną1 nie musi być błędem; dla kontrakcyjnego modelu poziomu trzeba uzasadnić równowagę. Nie zakładać, że trening ma już stały punkt w badanym okresie.

Duża liczba tokenów nie usuwa problemu identyfikowalności. Wszystkie są pobudzane tymi samymi aktualizacjami wag, a powtórzone cechy globalne nie tworzą nowych niezależnych zdarzeń. Udział wspólnego ruchu może pomóc regularizacji, ale nie dowodzi sił między cząsteczkami. Macierze A/K są parametrami predyktora. Dopasowanie joint ridge po lagach nie jest automatycznie odzyskaniem dokładnych operatorów Mori–Zwanziga ani dowodem FDT. Ruch minibatch/AdamW/BN pozostaje częściowo ukryty; observer nie odzyska jego przyszłej realizacji z samej historii.

## Odczyt bieżącego PLAN.md

Sprawdzono wersję ogólną planu istniejącą 2026-10-02 podczas researchu, przed szczegółową siatką. Zgodne są: cel znaku skumulowanego przemieszczenia, wszystkie 128 kanałów, brak PCA, origin 500, zakaz kotwicy500 we wcześniejszych foldach, open-loop obserwatora, ujawnienie wcześniejszej ekspozycji testu i odrębność H500 od prognozy kolejnych przyrostów. **Nie stwierdzono blokera w tej części planu.** Przed obliczeniami pozostają do zapisania: konkretne foldy, zbiór obrazów, siatka modeli, reguła wyboru i dokładne kryterium uznania celu za osiągnięty. Ten odczyt nie potwierdza jeszcze poprawności przyszłego kodu ani osiągnięcia 85 procent.

## Audyt doprecyzowanej specyfikacji przed fitami

Root następnie przekazał konkretny podział i regułę wyboru. Poniższa ocena dotyczy tej specyfikacji, nie nieistniejących jeszcze wyników:

| Etap | Dane do estymacji parametrów | Stan początkowy prognozy | Oceniane horyzonty |
|---|---|---|---|
| Walidacja A | Sondy 0–5, koniec 250 | Sondy 6–7, origin 300 | 1, 10, 50, 100, 200 |
| Walidacja B | Sondy 0–5, koniec 350 | Sondy 6–7, origin 400 | 1, 10, 50, 100 |
| Refit i test | Sondy 0–7, koniec 500 | Origin 500; sondy 0–7 i transfer 8–11 osobno | 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500 |

Zasada wyboru: dopuszczalny kandydat ma średni skill MSE większy od 0 i średni skill dla H co najmniej 100 nieujemny. Ranking maksymalizuje minimum po H ze średniej trafności znaku, następnie średni znak, średni skill i ustalony identyfikator konfiguracji. Jeśli nikt nie przechodzi bramki, ranking diagnostyczny pozostaje jawnie niepowodzeniem kryterium użyteczności. Przed testem zamraża się zwycięzcę, wariant najlepszego MSE i najlepszy wariant każdej rodziny. NVAR nie wchodzi do tego pilota; pozostaje udokumentowaną opcją literaturową.

**Werdykt: konstrukcja jest logicznie spójna.** Warunek85 procent jest sprawdzany po zamrożeniu osobno dla każdego H, a nie narzucany przez dalsze strojenie. Doprecyzowania przekazane rootowi przed fitami:

1. W foldzie A centra, skale i operator pochodzą wyłącznie z danych do 250, ale stan filtra, opóźnień i EMA na obrazie walidacyjnym wolno odtworzyć z jego rzeczywistej historii do 300. To stan początkowy, nie fit parametrów na walidacji. Analogicznie działa fold B. Nie można wyliczać tych samych statystyk z całego prefiksu do 500.
2. Przy nierównej liczbie foldów najpierw trzeba otrzymać średni wynik na każdym H, a potem równoważnie agregować H. W przeciwnym razie H200 jest automatycznie niedoważony względem horyzontów obecnych w obu foldach. H200 nadal opiera się tylko na jednym foldzie. Sposób agregacji MSE przed ilorazem skill musi być jawny i jednakowy dla modeli.
3. Podany Kalman estymuje prędkość z prefiksu, a po originie przewiduje ruch od rzeczywistego położenia. Jest to dozwolony, jawnie zakotwiczony wariant obserwatora. Nie wolno nazywać go identycznym z pełną prognozą odszumionego poziomu modelu stanu. R, F i Q są zamrożonymi ustawieniami, bez przyszłych korekt.
4. Stabilizowany Koopman jest tylko lokalnym L0 z przycięciem promieni wartości własnych do 0.995 lub1. Jest to heurystyczna stabilizacja w pełnych 128 kanałach, nie implementacja DISKO ani redukcja bazy. Dla progu 1 nie wynika stabilność asymptotyczna; bloki Jordana mogą powodować wzrost potęg. Także próg 0.995 wymaga kontroli wzmocnienia przejściowego.
5. Po zmianie F należy z góry określić sposób ustawiania interceptu. Rekomendacja: ponownie obliczyć `c=mean_train(Y)−F mean_train(X)` na tych samych danych treningowych. Alternatywa może być legalna, lecz nie powinna przypadkowo przestawiać równowagi przez pozostawienie interceptu starej macierzy bez wyjaśnienia.
6. W wariancie sprzężonym wspólne wymuszenie dodane wszystkim tokenom znika w ich odchyleniach od coverage-średniej. Przy stałych wagach i wspólnych interceptach można osobno sprawdzać operator odchyleń oraz pełny blok średniaFG–bg. Dla GLE/EMA do obu trzeba dodać odpowiednie pamięci; sprawdzenie samego A nie wystarcza. Ten rozkład nie jest PCA ani dowodem oddziaływań fizycznych.

Nie ma literaturowej podstawy do ogłoszenia sukcesu przed obliczeniami. Priorytetem jest uzyskanie rzetelnej krzywej horyzontu użyteczności i porównania z trendem, nawet gdy żaden model nie przekroczy 85 procent.

## Audyt kodu przed walidacją, 2026-10-02

Zakres: statyczny odczyt `modele_liniowe.py`, `modele_obserwatora.py` oraz szczegółowego `PLAN.md`. Audytor nie uruchamiał dopasowania ani prognoz. W odczytanej wersji siatka zawierała 32 modele liniowe i 35 obserwatorów; późniejszą propozycję dodania kroku 10 opisano oddzielnie poniżej. **Nie znaleziono blokującego błędu indeksowania ani ujawnienia przyszłych obserwacji w tych dwóch modułach.** To nie zastępuje kontroli wyboru, metryk i zapisanych wyników.

- `modele_liniowe.py:55` ogranicza dane do końca fitu przed centrowaniem i budowaniem cech. Stan `q−q(0)` wykorzystuje dostępną kotwicę początkową, również w wewnętrznych foldach. Nie używa przyszłego poziomu z kroku 500. Wspólne cele zaczynają się od indeksu 5, więc model z czterema przyrostami ma komplet historii bez przesunięcia celu.
- `modele_liniowe.py:46` dopasowuje regresję z równymi wagami obrazów i coverage wewnątrz obrazu. Przejście ze standaryzacji do natywnych macierzy oraz interceptu jest algebraicznie poprawne; kod zawiera sprawdzenie równoważności predykcji. Obecność wielu tokenów nie zwiększa liczby niezależnych aktualizacji wag.
- `modele_liniowe.py:20` i `modele_liniowe.py:86` używają tego samego indeksowania EMA. Pamięć na originie pochodzi z przeszłości, a potem aktualizuje się wyłącznie przewidzianymi przyrostami. Dryf FG i przyrost tła są obliczane przed zmianą obu stanów, co daje aktualizację synchroniczną. Średnia FG jest każdorazowo wyliczana z bieżących prognoz tokenów.
- `modele_liniowe.py:77` przycina wartości własne macierzy przejścia `F=I+Bᵀ`, a następnie przelicza intercept, zachowując średni przyrost treningowy przy średniej cech treningowych. Po uwadze audytora dodano rzeczywisty promień spektralny zrekonstruowanej macierzy i uwarunkowanie wektorów własnych. Sam zadany próg promienia nie kontroluje nienormalnego wzmocnienia przejściowego; próg 1 nie gwarantuje stabilności asymptotycznej.
- `modele_obserwatora.py:72` realizuje filtr poziomu i prędkości z macierzą `F=[[1,rho],[0,rho]]`. Suma przyszłego przemieszczenia odpowiada `sum(h=1..H) rho^h v(origin)`. Jawne zakotwiczenie prognozy w rzeczywiście obserwowanej pozycji originu jest zgodne z planem. Nie jest to pełna prognoza od odszumionego poziomu filtra.
- `modele_obserwatora.py:98` odcina przyszły sufiks przed filtrowaniem i regresją trendu. W obserwatorze sprzężonym współczynnik odpowiedzi na tło pochodzi wyłącznie z fitu; po originie zarówno ruch tła, jak i składowa resztowa są prognozowane. Nie wykryto użycia przyszłego tła jako znanego sterowania. Zamieszczone w module kontrole prefiksu, przesunięcia i stałej trajektorii są użytecznymi kontrolami implementacji, lecz nie dowodzą trafności naukowej modelu.

## Proponowane rozszerzenie o krok 10 przed nową walidacją

Root zgłosił rozszerzenie siatki o te same 32 konfiguracje liniowe liczone na próbkach co 10 aktualizacji. Daje to 99 konfiguracji łącznie. Uzasadnieniem jest wcześniejszy, jawnie już znany wynik eksperymentu natywnego, a nie wynik nowej walidacji. Rozszerzenie jest metodologicznie dopuszczalne po zapisaniu rewizji planu i zamrożeniu całej siatki przed odczytem nowych wyników. Nie wolno przedstawiać tej decyzji jako niezależnej od całej wcześniejszej historii badań.

Krótkie H1, H2 i H5 mogą być otrzymane przez liniową interpolację od obserwowanej pozycji originu do **przewidzianej** pozycji H10. To nie wymaga przyszłego pomiaru. Narzuca jednak wspólny znak tych trzech prognoz przemieszczenia i nie identyfikuje zmian wewnątrz dziesięciu aktualizacji. Metryki muszą nadal używać rzeczywistych celów H1, H2 i H5 z pełnej historii.

Jednostki parametrów muszą być jawne. Niezmieniony bank półtrwania 2/10/50 kroków modelu oznacza odpowiednio 20/100/500 aktualizacji dla kroku 10. Podobnie rho=0.995 jest wtedy ograniczeniem na dziesięć aktualizacji. `FIRST_TARGET=5` oznacza pierwszy cel w kroku 50 zamiast 5, więc różnią się również początki zbiorów przejść treningowych. Jest to dopuszczalne porównanie metod o różnych rozdzielczościach, lecz nie izolowany test wpływu samego próbkowania przy identycznym zbiorze przejść. Wszystkie modele zachowują te same fizyczne originy, oceniane H i regułę wyboru.

## Werdykt przed testem H500

**Audyt przedtestowy zakończony pozytywnie.** Przejrzano dodatkowo `badanie.py` i `metryki.py`, a następnie `WYBOR_PRZED_TESTEM.json` oraz zapisane macierze i metadane. W chwili kontroli `OCENA_H500.json` jeszcze nie istniał, a wybór miał `test_evaluated=false`. Audytor niezależnie przeliczył agregację i ranking z zapisanych wyników poszczególnych foldów, bez ponownego dopasowania modeli i bez odczytywania przyszłych celów testu.

Walidacja objęła 99 konfiguracji; wszystkie miały skończone prognozy w ocenianym zakresie. Tylko trzy przeszły bramkę MSE i wszystkie należały do rodziny powrotu do historycznej średniej. Wybrany `reversion_w200_t200` uzyskał minimum trafności znaku po H równe 56,3447 procent, średni znak 61,3722 procent i średni skill MSE 8,0155 procent. Wybrany osobno najlepszy MSE, `reversion_w50_t50`, uzyskał średni skill 16,3545 procent oraz minimum znaku 55,5774 procent. Trzeci dopuszczalny wariant to `reversion_w50_t200`. Nie są to wyniki testu H500 ani osiągnięcie celu 85 procent. Reguła minimum znaku może legalnie wybrać wariant o mniejszym średnim skillu.

Przeszły 32 zapisane porównania pełnego prefiksu, uciętego prefiksu i sufiksu podmienionego na NaN: po dwie sondy dla 16 zamrożonych modeli. Kształty macierzy zachowują 128 wyjść; w modelach z pamięcią większy wymiar wejścia oznacza dodatkowe natywne stany historyczne, nie redukcję. Zgodne były SHA256 plików Python, danych, macierzy i metadanych. Dla dwóch wybranych stabilizowanych operatorów rzeczywiste promienie wynoszą około 0,995 i 1,000; uwarunkowania wektorów własnych około 358,81 i 239,81. Z tych liczb nie wynika brak wzmocnienia przejściowego.

Wykryto i jawnie rozwiązano rozbieżność hasha planu. Wersja na początku procesu walidacji miała SHA256 `756ef3ff0772f2dadf1f76b319e84299f141fbcd71a7ff79776dbcf8f43c94c5`, a wersja przed testem `af9b209576a365adeedc4e437e68103254937f15d38dc2fa35e85f3be33d40c0`. Jedyną zmianą było dopisanie zdania precyzującego, że rho dotyczy kroku modelu, czyli dziesięciu aktualizacji w dt10. Nie zmieniono siatki, foldów, rankingu ani wykonanych fitów. Zachowano dokładną wersję początkową w `PLAN_SNAPSHOT_START.md`, oba hashe i treść zmiany w `ANEKS_ZAMROZENIA_PLAN.json`; hash snapshotu został niezależnie sprawdzony. Początkowa blokada testu nie została pominięta bez śladu.

Metryki stosują równe wagi obrazów i coverage wewnątrz obrazu. Raportowany średni skill jest **średnią względnych redukcji MSE poszczególnych obrazów/foldów**, a nie ilorazem ich średnich MSE. H200 ma dwa rekordy obrazów z jednego folda; pozostałe H mają cztery rekordy z dwóch foldów. Agregacja po H daje im równe znaczenie. Znak obejmuje wszystkie kanały bez odsiewu; cosinus wyłącza tokeny o zerowej normie i ujawnia liczebność ważnych tokenów. Interpolacja dt10 nie używa prawdziwych przyszłych endpointów. Po tym audycie dopuszczono wyłącznie ocenę już zamrożonych modeli, bez rozszerzania siatki.

## Końcowa weryfikacja i wynik H500

Test zapisano 2026-10-02 po audycie przedtestowym. Sprawdzono `OCENA_H500.json`, końcowe `WYNIKI.md`, `CO_OPISUJE_MODEL.md` oraz kod i wynik diagnostyki operatorów. **Nie znaleziono blokującego błędu obliczeń ani podstawy do ogłoszenia osiągnięcia 85 procent.** Walidacja dotyczyła 99 konfiguracji, natomiast w końcowym teście oceniono 16 wcześniej zamrożonych wariantów. Żaden z tych 16 wariantów nie osiągnął progu w żadnym H od 1 do 500 w agregacji grupy fit, transfer ani wszystkich obrazów. Nie jest to twierdzenie o każdej pojedynczej cząsteczce ani dowód niemożliwości dla wszystkich potencjalnych metod.

| Model zamrożony przed testem | Grupa | Znak H500 | Średni skill MSE H500 | Norma prognozy / rzeczywista H500 |
|---|---|---:|---:|---:|
| Główny wybór: powrót, okno 200, czas 200 | Fit 0–7 | 57,1094% | +0,7408% | 0,5108 |
| Główny wybór: powrót, okno 200, czas 200 | Transfer 8–11 | 56,5201% | −7,5170% | 0,5235 |
| Osobny najlepszy MSE: powrót, okno 50, czas 50 | Fit 0–7 | 63,1993% | +18,3241% | 0,4402 |
| Osobny najlepszy MSE: powrót, okno 50, czas 50 | Transfer 8–11 | 62,1930% | +9,8235% | 0,4637 |

Drugi wariant poprawia wielkość błędu na tym jednym długim horyzoncie, ale nadal niedoszacowuje normę ruchu i nie osiąga celu znaku. Był wcześniej zamrożonym porównaniem, a nie nowym zwycięzcą wybranym po teście. Jego dodatni skill nie potwierdza odtworzenia kolejnych przyrostów ani fizycznego mechanizmu ruchu.

Dla kontrastu ostatnia prędkość na H1 daje znak 76,8822 procent i skill 55,0526 procent na grupie fit, a na transferze 75,7328 procent i 45,7229 procent. Dalsza ekstrapolacja tego trendu silnie zawyża amplitudę. To pokazuje, że dobry wynik jednego kroku nie przenosi się automatycznie na H500. Również filtr Kalmana bez kolejnych pomiarów nie naprawił długiego horyzontu; nie jest to negatywny wynik dla zadania filtracji z dostępnymi pomiarami, którego tutaj nie oceniano.

Niezależne kontrole audytora:

1. Ponowne zagregowanie zapisanych wyników per obraz do 3 grup, dla 16 modeli, 500 horyzontów i 5 miar dało 120000 zgodnych porównań, maksymalna różnica 0.
2. Dla obu modeli powrotu odtworzono bez fitu zamknięty wzór z surowego prefiksu do 500: `zhat(H)=z(500)+(1−exp(−H/tau))*(mean(z(501−W)..z(500))−z(500))`. Sprawdzono 12 obrazów, H1/H50/H500 i 4 miary: 288 porównań, maksymalna różnica 4,44×10⁻¹⁶. Nie użyto celu do wyznaczenia średniej ani prognozy.
3. Hash wyboru wskazany przez `OCENA_H500.json` był zgodny z bieżącym zamrożonym wyborem. Hashe macierzy i metadanych pozostały zgodne. Nie stwierdzono zmiany modeli po teście.
4. Macierze companion w diagnostyce operatorów zawierają pełne poziomy i pamięci, a dla EMA również aktualizację śladów przewidzianym przyrostem. Wyłączono stały intercept przy analizie operatora jednorodnego. Przycięty model dt10 ma promień około 0,995, lecz normę jednego kroku około 2,064, więc nie jest kontrakcją w tej normie. Losowo próbkowane wzmocnienia są poprawnie oznaczone jako dolne granice, nie dokładne normy potęg. Normy rozszerzonych stanów zależą także od skali współrzędnych pamięci i nie są bezpośrednią miarą błędu położenia cząsteczki.

Końcowa interpretacja jest ograniczona do tego zbioru kandydatów i jednej znanej trajektorii uczenia. H500 ma jeden start, a obrazy i tokeny współdzielą aktualizacje wag. Nie ma tu niezależnego przedziału ufności po startach ani dowodu stabilnej identyfikacji sił, szumu białego, dokładnego GLE czy pełnej tożsamości narzędzia. Oryginalne osie pozostają stałe; prognozowane są ich współrzędne, nie obroty makrowektorów.
