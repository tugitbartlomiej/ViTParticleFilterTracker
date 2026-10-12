# Sześć prac o Koopmanie: audyt metody i wykonane poprawki
Data: 2026-09-24. Zakres: istniejące lokalne checkpointy RF-DETR, bez nowego treningu detektora lub generatora.

## Wynik
Pełne odczyty sześciu publikacji przez Undermind oraz kontrola kluczowych wzorów w źródłach pozwoliły poprawić konstrukcję testu i sprawdzić model z pamięcią. Nie znaleźliśmy twierdzenia gwarantującego, że surowy latent RF-DETR będzie dobrze prognozowany liniowo.

Wykonany wariant zachowuje wszystkie 576×384 współrzędne i dodaje poprzedni przyrost. Na odłożonych grupach lepiej przewiduje pojedynczy checkpoint niż poprzednia macierz, lecz nie pokonuje prostego powtórzenia ostatniej zmiany. Przy stu aktualizacjach jest gorszy także od braku zmian. Nie promujemy go do zwycięskiego modelu i nie zmieniamy historycznej bramki.

## Wybór i odczyt źródeł
[Wyszukiwanie Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Koopman%20forecasting%20detector%20representations%20across%20training%20checkpoints) zakończyło się listą 164 prac. Wybrano sześć najwyżej sklasyfikowanych pozycji z dostępnym PDF. Pominięte w pierwszej dziesiątce Sik25, Lu20c i Sha24b nie miały PDF w Undermindzie. Relevance jest oceną wyszukiwarki, nie oceną prawdziwości wyników.

Każdy pełny PDF odczytał osobny agent Undermind: metody, twierdzenia, eksperymenty i dostępne dodatki. Pytanie obejmowało dokładny kształt naszego stanu, budowę par, małą liczbę próbek, podziały grup i czasu oraz zakaz PCA. Odczyty Dog20 i Has22 dodatkowo sprawdzono ponownie z powodu błędów w pierwszych odpowiedziach. Poniżej są wnioski po weryfikacji, a nie bezkrytyczna kopia sugestii wyszukiwarki.

| Praca | Wynik relevance | Odczytany zakres | Znaczenie dla nas |
|---|---:|---|---|
| [Lu i Tartakovsky, Extended Dynamic Mode Decomposition for Inhomogeneous Problems, JCP 2021](https://doi.org/10.1016/j.jcp.2021.110550) [Lu20e] | 1,380 | 34 strony wersji arXiv 2020 z suplementem | Przyrost i wyraz wolny |
| [Tu i in., On Dynamic Mode Decomposition: Theory and Applications, JCD 2014](https://doi.org/10.3934/jcd.2014.1.391) [Tu13] | 1,350 | 30 stron preprintu, dodatek A | Pary, rząd, pamięć |
| [Williams, Kevrekidis i Rowley, A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition, JNS 2015](https://doi.org/10.1007/s00332-015-9258-5) [Wil14] | 1,250 | 34 strony preprintu, dodatek A | Funkcje obserwacji i domknięcie |
| [Dogra i Redman, Optimizing Neural Networks via Koopman Operator Theory, 2020](https://arxiv.org/abs/2006.02361) [Dog20] | 1,130 | 18 stron z suplementem | Rzeczywisty zakres eksperymentów na wagach |
| [Ouala i in., Extending the extended dynamic mode decomposition with latent observables: the latent EDMD framework, MLST 2023](https://doi.org/10.1088/2632-2153/acccd6) [Oua23] | 1,090 | 20 stron, dodatki A–C | Dodatkowy stan przy częściowej obserwacji |
| [Haseli i Cortés, Temporal Forward–Backward Consistency, Not Residual Error, Measures the Prediction Accuracy of Extended Dynamic Mode Decomposition, L-CSS 2023](https://doi.org/10.1109/LCSYS.2022.3214476) [Has22] | 1,080 | 7 stron preprintu 2022 z dodatkiem A | Ograniczenia samego błędu dopasowania |

Daty w kluczach odnoszą się często do preprintów. Tu13 ukazał się w Journal of Computational Dynamics; nazwa czasopisma w metadanych Undermind była błędna.

## Co dokładnie bierzemy z każdej pracy

### Lu i Tartakovsky
Sekcje 2.3–2.4, równania 2.14–2.19: model ma postać y=x+Bx+b. To uzasadnia istniejącą poprawkę do uczenia przyrostu. Autorzy rozdzielają odtwarzanie, ekstrapolację i uogólnianie. W sekcji 4.2 opisują pogorszenie przy zaszumionych danych, więc dodatkowy wyraz wolny i residual nie zapewniają przewagi w każdym zadaniu. Ich xDMD nie jest tym samym co nieliniowy słownik EDMD Williamsa.

**Nasza adaptacja:** regularyzujemy poprawkę do znanej dynamiki bazowej; nie wymazujemy automatycznie części wejścia poza rozpiętością danych. Dalej stosujemy ridge, ale dobieramy go na walidacji wewnętrznej. Nie kopiujemy obcinania SVD ani wyników PDE jako gwarancji dla sieci.

### Tu i współautorzy
Definicja 2 i twierdzenie 2: dokładne dopasowanie AX=Y zależy od zgodności zależności liniowych danych. Przy niezależnych kolumnach X warunek może być trywialnie spełniony, także dla próbek nieliniowego procesu. Sekcja 3.2.2 pokazuje przypadek oscylacji, w którym opóźniona obserwacja usuwa problem niewystarczającego stanu. Sekcja o wielu trajektoriach łączy poprawnie sparowane obserwacje, bez sztucznego przejścia między końcem jednej a początkiem drugiej.

**Nasza adaptacja:** każda para pozostaje tym samym obrazem. Dodajemy historię całego stanu. Błędu dopasowania nie nazywamy dowodem liniowości.

### Williams, Kevrekidis i Rowley
Sekcja 2.2: EDMD dopasowuje działanie na słowniku funkcji obserwacji. Sekcja 2.3 omawia granicę dużej liczby odpowiednio próbkowanych danych i projekcję Galerkina. Sekcja 2.4 wiąże słownik współrzędnych z DMD. To nie jest twierdzenie, że dowolny skończony latent sam stanowi niezmienniczą przestrzeń Koopmana.

**Nasza adaptacja:** oddzielamy pełny tensor sieci od kompletnego stanu procesu uczenia. Kolejne cechy nieliniowe można dołączyć do pełnego stanu, lecz trzeba sprawdzać ich dynamikę i prognozę. Samo nazwanie regresji „deep Koopman” nie zmienia założeń.

### Dogra i Redman
Sekcja 3 i suplement S10: operator dzielono na bloki Node/Quasi-Node, w MNIST pierwsza warstwa miała q=157. Dane wag zbierano od początku epoki 3 do końca epoki 5, przy 938 aktualizacjach na epokę; to około 2814 aktualizacji. Autorzy nie podają wprost liczby zapisanych par po uwzględnieniu indeksowania granic. Prognoza obejmowała następne dwie epoki.

**Nasza adaptacja:** porównujemy wymiar względem liczby próbek, a nie samą liczbę epok. To doświadczenia na wagach małych sieci, nie dowód dla aktywacji ViT. Podział na niezależne bloki byłby dodatkowym założeniem o sprzężeniach; nie wprowadziliśmy go po cichu zamiast pełnego operatora.

### Ouala i współautorzy
Sekcja 3.2, równania 22–23 i 32–33: do obserwowanych wielkości dołącza się latentne obserwable, optymalizuje ich przebieg i liniową dynamikę. Prognoza wymaga wyznaczenia stanu ukrytego z dostępnej historii. To nie jest zwykłe użycie gotowego enkodera obrazu. Praca rozważa też szum i estymację stanu.

**Nasza adaptacja:** pamięć jest uzasadnionym kandydatem do uzupełnienia obserwacji. Nasz jawny poprzedni przyrost jest prostszym modelem, nie replikacją LEDMD ani rozszerzonego Langevina. Nie narzucamy jednostkowego widma z eksperymentów oscylacyjnych procesowi uczenia detektora.

### Haseli i Cortés
Założenie II.1 wymaga pełnego rzędu kolumn macierzy słownika. Twierdzenia V.1 i V.3 wiążą indeks zgodności z błędem na danych przy tym założeniu. Sekcja III pokazuje zależność zwykłych residuali od wyboru bazy.

**Nasza adaptacja:** przy 48 lub 192 próbkach i setkach tysięcy współrzędnych założenie nie zachodzi. Nie używamy tego twierdzenia jako certyfikatu domknięcia. Wskaźnik różnicy projektorów próbek może być zerowy przez interpolację; nie jest wtedy równoważny oryginalnemu indeksowi. Testujemy ten kontrprzykład numerycznie.

## Diagnoza naszej konstrukcji
Poniższe rozumowanie jest naszym audytem, nie twierdzeniem jednej z prac.

1. **Pary są poprawne:** z(i,t)→z(i,t+5) dla identycznych pikseli. Wiele obrazów powiększa zbiór par, ale nie daje niezależnych treningów wag. Obrazy z tej samej grupy i ich augmentacje są zależne.
2. **Pełny latent nie oznacza pełnego stanu uczenia.** Z(i,t)=f(x_i;theta_t). Następny latent zależy także od aktualizacji wag, stanu optymalizatora i danych. Dwa stany sieci mogą mieć podobne z, lecz inne kolejne zmiany.
3. **Wspólna macierz dla obrazów jest hipotezą empiryczną.** Nie ma zakazu teoretycznego jej dopasowania ani gwarancji istnienia dokładnej wspólnej mapy. Lokalnie zmiana reprezentacji zależy od jakobianu względem wag dla danego obrazu. Nie uśredniamy osobno dowolnie wybranych niedookreślonych macierzy; dopasowujemy jedną wspólną mapę do zgodnych par.
4. **Macierz per checkpoint zna oba końce odcinka na grupach fit.** Mały residual mówi o dopasowaniu tego odcinka. Dopiero użycie poprzedniej macierzy na nieznanym odcinku bada prognozę w czasie.
5. **Brak PCA nie usuwa niedookreślenia.** Wycentrowane 48 próbek ogranicza rząd danych do 47. Regularizacja wybiera jedno z możliwych rozszerzeń operatora. Wartości własne równe jeden, wymuszone przez prior identyczności w niezbadanych kierunkach, nie dowodzą tożsamości narzędzia.
6. **Silniejsze tłumienie poprawki może pomagać.** Nie wolno utożsamiać zachowania surowych współrzędnych z zakazem regularyzacji niepewnego prawa ich zmiany. Walidacja wewnętrzna wybrała tutaj 0,1 spośród 0,00001, 0,001 i 0,1. Zachowaliśmy wszystkie liczby stanu, nie obcinaliśmy kierunków.

## Zaimplementowany wariant
Pliki: koopman_pamiec_full.py, audyt_literatura6.py, test_literatura6.py.
Specyfikację LITERATURA6_PROTOCOL.json zapisano przed uruchomieniem analizy.

Niech v_t=z_t−z_(t−1). Rozszerzony stan [z_t,v_t] zachowuje informację o obu pełnych tensorach. Model przewiduje:
```text
a_hat = mean_a + ((z-mu_z)/s_z) U^T C + ((v-mu_v)/s_v) V^T C
v_next = v + a_hat
z_next = z + v_next

U = (X-mu_z)/s_z
V = (Vel-mu_v)/s_v
R = Y-X-Vel-mean_a
G = U U^T + V V^T
C = solve(G + lambda I, R)
```
s_z i s_v są dodatnimi skalami RMS bloków wyznaczonymi na fit. To odwracalne skalowanie, bez projekcji. Macierz 442368×442368 nie jest materializowana; jej działanie i rollout liczymy dokładnie przez czynniki dualne, bez obcięcia rzędu. Nie jest to trening deep Koopmana.

W każdej prognozie używamy czterech zakończonych odcinków i 48 obrazów fit, czyli 192 par. Osiem początków prognozy: kroki 210,230,250,270,290,310,330,350. Kalibracja używa początków 105,145,185, sześciu grup do dopasowania i dwóch innych wewnętrznych grup do wyboru ridge. Cztery pierwotne grupy holdout nie uczestniczą w wyborze. Dla dłuższych horyzontów pozostają tylko początki z dostępnym przyszłym checkpointem. Podczas rollout nie podstawiamy prawdziwego przyszłego latentu.

Oprócz błędu pełnego stanu zapisujemy błędy na tokenach ważonych maską, kąty i długości przyrostów, błędy dla każdego obrazu i odcinka, rząd Gram, uwarunkowanie, efektywną liczbę stopni swobody oraz residual rozwiązania układu. Wszystkie 73 pliki tensorów przeszły SHA256 względem zapisanych dowodów ekstrakcji.

## Wyniki
Zakończony wcześniejszy bieg: 73 checkpointy, 72 obrazy, 5184 par. Na wspólnych odcinkach K_t miał RMSE fit 0,0000391 i holdout 0,0116170. Przewidywanie kolejną macierzą z poprzedniego odcinka: 0,0117613 na holdout; ostatni przyrost: 0,0043095. Nie porównujemy tych liczb wprost z poniższą tabelą, która używa późniejszych, wspólnych początków prognoz.

Nowe porównanie, RMSE na holdout; mniejsze lepsze:

| Horyzont w aktualizacjach | Poprzednia macierz | Pełny stan z pamięcią | Ostatni przyrost | Brak zmian |
|---:|---:|---:|---:|---:|
| 5, jeden checkpoint | 0,0089230 | 0,0045273 | **0,0042506** | 0,0094380 |
| 20, cztery checkpointy | 0,0363908 | 0,0246363 | **0,0217568** | 0,0382765 |
| 100, dwadzieścia checkpointów | 0,1720721 | 0,2873054 | **0,1382725** | 0,1725821 |

Dla jednego checkpointu wariant z pamięcią zmniejsza błąd względem poprzedniej macierzy o około 49 procent, ale ma około 6,5 procent większy błąd niż ostatni przyrost. Na tokenach narzędzia odpowiednie RMSE wynoszą 0,0058353 i 0,0055865; wniosek się nie odwraca.

**Nie przypisujemy poprawy samej pamięci ani nauczonej macierzy:** jednocześnie zmieniliśmy okno dopasowania z 48 do 192 par, dobór regularyzacji ze stałej na walidację wewnętrzną i dynamikę bazową z zachowania stanu na zachowanie ostatniego przyrostu. Kontrola korzystająca z historii obrazu bez uczenia nadal wygrywa. Na 100 aktualizacjach model z pamięcią ma znacznie większy błąd. Parametry dobierano pod jeden krok; nie zmienialiśmy ich po obejrzeniu dalszych wyników.

W obrębie każdego wiersza wszystkie metody mają te same początki prognozy. Liczby początków wynoszą odpowiednio 8, 7 i 3; nie należy zatem przypisywać całej różnicy między wierszami samemu horyzontowi. Dodatkowo policzono wszystkie horyzonty dla tych samych kroków 210, 230 i 250: RMSE modelu z pamięcią wynosi 0,0048506 / 0,0240714 / 0,2873054, a ostatniego przyrostu 0,0045515 / 0,0197751 / 0,1382725. Wniosek pozostaje taki sam. Dokładne zestawienie: wyniki/audyt_literatura6/MATCHED_HORIZONS.json. Wykres per checkpoint: [PROGNOZY.png](wyniki/audyt_literatura6/PROGNOZY.png).

W modelach końcowych rząd Gram wynosi 191 przy 192 parach, a residual rozwiązania liniowego jest około 10^−14. Mały residual solvera potwierdza poprawność rozwiązania zadanej regresji, nie poprawność założeń dynamiki.

Wyniki: wyniki/audyt_literatura6/REPORT.json i RAPORT.md; per obraz ERRORS_PER_IMAGE.csv; podziały CONFIG.json; dobór CALIBRATION.json; diagnostyka DIAGNOSTICS.json; macierze OPERATOR_DELAY_*.npz i listy par PAIRS_*.json. Odtworzenie U,V korzysta z tensorów źródłowych wskazanych przez te listy oraz zapisanych średnich i skal.

## Testy i ograniczenia
Przeszło sześć testów unittest, w tym siedem kontroli wewnętrznych modelu: dualna regresja względem jawnej, rollout 20 kroków, zerowy horyzont, rozróżnienie ukrytych prędkości, stała prędkość, chronologia i nieregularna siatka czasu. Dodatkowe kontrprzykłady wykazują, że interpolacja nie dowodzi liniowości oraz że nie wolno stosować twierdzenia Haseli–Cortés bez jego warunku rzędu.

Niezależny przegląd kodu nie wskazał błędu wzoru regresji, rollout ani podziałów grup i czasu. Zgłoszone ograniczenia porównania opisano powyżej. Skrypt weryfikacja_literatura6.py odtworzył zapisany operator kroku 210 z pliku i tensorów źródłowych, a następnie niezależnie odtworzył zapisany błąd jednego obrazu holdout (zgodność względna 10^−10). Dowód: VERIFICATION.json. Kopie kodu są w source_snapshot/, a DEPENDENCIES_SHA256.json obejmuje także implementację kontroli, metryki, maski i konfigurację źródłową. Ten dodatkowy manifest zapisano po obliczeniach; nie przedstawiamy go jako wcześniejszej prerejestracji.

To analiza diagnostyczna, ponieważ historyczny holdout był już oglądany. Cztery grupy źródłowe i ich warianty nie są niezależnymi filmami. Nie wykazano kompletności tożsamości narzędzia ani wpływu na jakość detekcji. Nie zmieniono wag RF-DETR/FastGAN. Pobranie całości z Edena i pokaz FastGAN pozostają osobnymi niedokończonymi etapami.

Następny uzasadniony test to dobór już zaimplementowanego modelu korekty ostatniego przyrostu pod prognozę wielokrokową wewnątrz fit albo jawne sterowanie zapisanym przebiegiem optymalizatora. Deep Koopman może dołączyć funkcje do pełnego stanu, ale musi pokonać ostatni przyrost na nowych danych; nie ma powodu traktować go z góry jako rozwiązania. Tych kolejnych testów nie uruchamiano w tym audycie.

## Kontrola błędów odczytu literatury
- Pierwszy odczyt Dog20 błędnie wyliczył 1876 kroków dopasowania. Zweryfikowany suplement mówi o trzech pełnych epokach, około 2814 aktualizacjach; 1876 odpowiada dwóm epokom prognozy.
- Sugestia „żadna wspólna mapa aktywacji różnych obrazów nie może istnieć” nie wynika z pracy. Odrzucona; sprawdzamy to empirycznie.
- Proponowany w pierwszym odczycie Has22 wzór ridge na zgodność i arbitralny próg 0,2 nie pochodzą z twierdzenia i nie zostały użyte.
- Przy pełnym rzędzie wierszy i N<D różnica projektorów w przestrzeni próbek wynosi zero, ale pierwotna macierz I−K_F K_B ma jednostkowe wartości własne poza zbadanymi kierunkami. Nie można użyć równości z twierdzenia V.3 wbrew jego założeniu.
- Dla niedookreślonej regresji A=X^+Y oraz I+X^+(Y−X) są zgodne na próbkach, ale różnią się poza ich rozpiętością o I−X^+X. Stwierdzenie o globalnej równoważności nie jest poprawne bez dodatkowych założeń. Odpowiedni test jest w kodzie.
