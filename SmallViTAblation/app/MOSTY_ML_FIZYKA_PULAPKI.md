# Mosty między światem uczenia maszynowego a fizyczną pułapką optyczną koloidu

Stan na 2026-10-14. Model MałyViT, bieg s47. Nazwy obowiązujące od 2026-10-14.

## Wstęp

Ten plik zbiera w jednym miejscu wszystkie połączenia między obrazem „laserów i koloidu” w aplikacji a znaną literaturą.
Przy każdym moście mówimy, czy jest to tożsamość matematyczna, przybliżenie, czy tylko analogia, i gdzie się łamie.
Słownik pojęć jest krótki.
**Koloid** to token `j` jednego zdjęcia: jego stan w strumieniu rezydualnym (192 liczby) tuż przed MLP bloku 6, liczony względem środka chmury 600 tokenów.
**Reflektor** (laser) to neuron `c` warstwy MLP. Ma **celownik** `s_c = P₁(γ⊙w_c)` i **wiązkę** `v_c` (kolumna `fc2`), a jego jasność dla tokenu to `z_jc = GELU(s_c·x_j/σ_j + b_c + w_c·β)`.
**Pułapka** to od dziś pułapka optyczna z laserów: **ogniskowa** to punkt, w którym pole wiązek `f(u)` się zeruje.
**Średnia skoków koloidu** to średnie położenie koloidu z ±300 kroków uczenia; dawniej mylnie nazywana pułapką.
Mamy dwa zegary: **głębokość** (6 bloków, jedno przejście, deterministyczne) i **czas uczenia** (koloid przesuwa się, bo zmieniają się wagi, a zdjęcie jest stałe).

Numery w nawiasach kwadratowych, na przykład [9], odsyłają do listy literatury na końcu, gdzie są DOI albo numery arXiv.

## Tabela główna

| # | most | obiekt w sieci (wzór) | odpowiednik fizyczny | status | gdzie analogia się łamie | prace |
|---|------|-----------------------|----------------------|--------|--------------------------|-------|
| 1 | blok rezydualny jako krok Eulera | `x_{l+1} = x_l + F_l(x_l)` | krok Eulera równania ruchu `dx/dt = F(x,t)` | dokładne jako zapis; przybliżenie jako ciągłe równanie | tylko 6 kroków, wagi różne w każdym bloku, brak gładkości w głębokości | [1] [2] [3] [4] |
| 2 | tokeny jako układ cząstek | `x ← x + Attn(LN₁x)`, potem `x ← x + MLP(LN₂x)` | podział Lie–Trottera: oddziaływanie cząstek, potem pole zewnętrzne | dokładne jako podział kroku; analogia jako „siły” | w czasie uczenia brak sił między koloidami, sprzęga je tylko wspólny gradient | [5] [6] [7] [8] |
| 3 | reflektor = neuron = para klucz–wartość | `MLP(x) = Σ_c z_c v_c + b₂` | wiązka lasera o mocy `z_c` i kierunku `v_c` | dokładne (tożsamość algebraiczna) | jasność ma znak (GELU < 0), światło nie ma; wiązka jest ta sama dla wszystkich tokenów | [9] [10] |
| 4 | κ i prawo pojedynczego lasera | `κ = cos(s_c, v_c)`, `‖A‖²/‖S‖² = (1−κ²)/(1+κ²)` dla `v_c s_cᵀ` | lustro, które wzmacnia (κ>0), sprężyna (κ<0), pchnięcie w bok (κ≈0) | dokładne dla członu rzędu 1; przybliżenie dla całego bloku | Jakobian bloku to suma 768 członów i rzutu LayerNorm, więc nie dziedziczy prawa | [11] [12] |
| 5 | LayerNorm jako sfera | `LN(x) = γ⊙P₁x/σ + β`, `σ = ‖P₁x‖/√d` | celownik widzi tylko kierunek; długość tokenu działa jak bezwładność | dokładne dla kierunku; tylko analogia dla bezwładności | brak drugiego rzędu w czasie, więc „masa” nie daje pędu | [13] [14] [15] [16] |
| 6 | ogniskowa jako punkt równowagi | `f(u) = P3 Σ_c (z_c(x(u)) − z̄_c) v_c`, ogniskowa: `f = 0`, charakter z wartości własnych `∂f/∂u` | ognisko pęsety optycznej, punkt równowagi siły | dokładne jako definicja; przybliżenie, bo przekrój 3D | zero istnieje tylko w 7,7 % checkpointów; w 192D zawsze siodło, nigdy studnia | [17] [18] [19] |
| 7 | rozkład pola na część potencjalną i wirową | `J = S + A`, `S = (J+Jᵀ)/2`, `A = (J−Jᵀ)/2` | rozkład siły na gradient potencjału i część cyrkulacyjną (rozkład Ao) | dokładne lokalnie | to rozkład liniowy w punkcie, nie globalny potencjał | [20] [21] [18] [22] |
| 8 | siły niezachowawcze w pęsecie | antysymetryczna część Jakobianu pola wiązek | wir Browna, fontanna prawdopodobieństwa | tylko analogia (ta sama algebra dryfu) | w głębokości nie ma szumu, więc nie ma prądu prawdopodobieństwa | [23] [24] [25] [26] [27] [28] |
| 9 | bramka GELU jako obsadzenie Fermiego | `Φ(a) ≈ 1/(1+e^{−1,702a})`, `T_laser = 0,59/sd(a)` | rozkład Fermiego–Diraca, krzywa Boltzmanna kanału jonowego | przybliżenie (Φ a logistyczna); analogia (temperatura) | brak kąpieli cieplnej; „temperatura” to szerokość bramki względem rozrzutu wejść | [29] [30] [31] [32] [33] |
| 10 | termometr J_T | `J_T = E[∇F(x+δ)] = E[F(x+δ)δᵀ]/T`, `δ ~ N(0, T·I)` | pomiar siły przez fluktuacje cieplne (mikroskop siły fotonowej) | dokładne (lemat Steina); analogia (temperatura) | dla pola liniowego J_T nie zależy od T; mierzy nieliniowość, nie ciepło | [34] [35] [36] [22] [24] [27] |
| 11 | SGD jako dynamika Langevina | `θ ← θ − η ĝ`, szum paczki o skali ok. `η/B` | cząstka Browna w potencjale straty, temperatura ∝ lr/paczka | przybliżenie | szum anizotropowy, niskiego rzędu, nie biały; AdamW i obcięcie gradientu; prądy nierównowagowe | [37] [38] [39] [40] [41] [42] [43] |
| 12 | relacja Einsteina i FDR | wariancja skoków kontra odpowiedź na siłę | relacja Einsteina, twierdzenie fluktuacyjno-dyssypacyjne, Harada–Sasa | przybliżenie dla średniej skoków; łamie się dla koloidu | średnia skoków nie jest przyczyną ruchu; dla koloidu relacja nie zachodzi | [44] [45] [46] |
| 13 | weight decay jako sprężyna | `θ ← θ − η·λ·θ` (AdamW, λ = 0,05) | harmoniczna sprężyna do zera w przestrzeni wag | dokładne w wagach; przybliżenie w tokenach | w przestrzeni koloidu działa tylko przez Jakobian `∂x/∂θ` | [47] [37] |
| 14 | wspólne wagi jako ośrodek sprzęgający | `δx_i = J_i δθ`, `J_i = ∂x_i/∂θ` | sprzężenie hydrodynamiczne, wspólne wymuszenie (efekt Morana) | dokładne do pierwszego rzędu | „ośrodek” nie ma własnej dynamiki; sprzężenie to jedno wspólne wymuszenie, nie oddziaływanie par | [48] [49] [50] [51] [52] [53] |
| 15 | pomiary z pęsety przeniesione na tokeny | trajektorie koloidu, pole i Jakobian liczone dokładnie | ekwipartycja, widmo mocy, FORMA, przeciągana pułapka, Harada–Sasa | metodyka przenośna; interpretacja tylko analogia | estymatory zakładają ruch Browna w stałej pułapce, czego tu nie ma | [54] [55] [56] [18] [57] [58] |

## Mosty jeden po drugim

### Most 1. Blok rezydualny jako krok Eulera

Każdy blok dodaje poprawkę do tokenu: `x_{l+1} = x_l + F_l(x_l)`.
To jest dosłownie krok metody Eulera z krokiem równym jeden, co zauważyli E [1] oraz Haber i Ruthotto [2], a Chen i współautorzy zrobili z tego sieci ODE [3].
Dlatego w aplikacji głębokość można rysować jako czas, a token jako punkt, który się przemieszcza.
Granica jest jasna u Sandera, Ablina i Peyré [4]: sieć rezydualna przybliża ciągłe równanie tylko wtedy, gdy funkcje bloków zmieniają się gładko z głębokością.
U nas jest 6 bloków o niezależnych wagach, więc „ciągła trajektoria w głębokości” to tylko interpolacja do rysunku.
Zapis jest dokładny, ciągły ruch nie jest.

### Most 2. Tokeny jako układ wielu cząstek

W bloku pre-LN z timm najpierw działa uwaga, potem MLP.
Lu i współautorzy [5] pokazali, że to jest schemat podziału Lie–Trottera dla układu wielu cząstek: uwaga to człon oddziaływania (dyfuzja między tokenami), a MLP to pole działające na każdą cząstkę osobno (konwekcja).
Geshkovski, Letrouit, Polyanskiy i Rigollet [6] [7] badali tokeny jako cząstki na sferze, które z głębokością skupiają się w klastry; nowsza praca Mudarisova i współautorów [8] dokłada MLP jako lokalne „pole sterujące” i pokazuje, że liczy się jego składowa styczna do sfery.
Nasz obraz laserów zajmuje się właśnie tym drugim członem: polem MLP bloku 6.
Łamie się to w czasie uczenia. W tym zegarze koloidy nie przyciągają się ani nie odpychają, a ich wspólny ruch wynika ze wspólnych wag (most 14).
W literaturze cząstek tokenowych zegarem jest głębokość przy stałych wagach, a u nas głównie czas uczenia.

### Most 3. Reflektor jako neuron, czyli para klucz–wartość

Wyjście MLP to suma `Σ_c z_jc v_c + b₂`, więc każdy neuron dokłada wektor `v_c` przemnożony przez swoją jasność.
To jest tożsamość, a Geva i współautorzy [9] opisali ją jako pamięć klucz–wartość: wiersz pierwszej warstwy to klucz, kolumna drugiej to wartość.
Nasz celownik to ten sam klucz, ale z wbudowanym wzmocnieniem LayerNorm i bez średniej: `s_c = P₁(γ⊙w_c)`, a próg to `b_c + w_c·β`.
W teorii pola średniego Mei, Montanari i Nguyen [10] cząstkami są właśnie neurony, czyli pary `(w_c, v_c)`; u nas te same pary to reflektory, a cząstkami są tokeny.
Analogia łamie się na znaku: GELU daje też małe wartości ujemne, a natężenie światła nie może być ujemne.
Druga różnica: wiązka `v_c` jest identyczna dla wszystkich 600 tokenów, jak w układzie telecentrycznym, czego zwykły laser nie zapewnia.

### Most 4. Współczynnik κ i prawo pojedynczego lasera

Jeden laser wnosi do Jakobianu człon rzędu 1, proporcjonalny do `v_c s_cᵀ`.
Jego część symetryczna i antysymetryczna spełniają dokładnie `‖A‖²/‖S‖² = (1−κ²)/(1+κ²)`, gdzie `κ = cos(s_c, v_c)`.
Wynika to z dwóch faktów: norma Frobeniusa `v sᵀ` wynosi jeden dla wektorów jednostkowych, a iloczyn `v sᵀ` z `s vᵀ` wynosi `κ²`.
Przy κ dodatnim laser wzmacnia kierunek, który widzi, przy ujemnym działa jak sprężyna, a przy κ bliskim zera pcha w bok i tworzy wir.
Sam cosinus między wejściem i wyjściem neuronu jest znany: Gerstner i Schütze [11] nazywają takie neurony wzbogacającymi i zubażającymi, a później osłabiającymi [12], i widzą zubażanie częściej w późnych warstwach.
Prawo dla stosunku części wirowej do potencjalnej oraz odczytanie κ jako sztywności i wiru nie pojawiło się w znalezionej literaturze.
Granica: prawo jest dokładne tylko dla jednego członu, a Jakobian bloku to suma 768 takich członów, przemnożonych przez pochodną GELU i rzut LayerNorm.

### Most 5. LayerNorm jako sfera i długość jako bezwładność

LayerNorm usuwa składową wzdłuż wektora jedynek, dzieli przez długość reszty i skaluje przez pierwiastek z wymiaru; Gupta i współautorzy [13] opisują to geometrycznie wprost.
Riechers [14] pokazuje, że wyniki leżą na elipsoidzie w hiperpłaszczyźnie, a Brody i współautorzy [15] badają rolę tego rzutu w uwadze.
Stąd wynika dokładnie, że celownik widzi tylko kierunek tokenu: `s_c·x_j/σ_j` nie zmienia się, gdy token wydłużymy.
To samo pchnięcie obraca długi token o mniejszy kąt niż krótki, i to nazywamy bezwładnością.
Tuomi [16] pokazuje w dużych modelach językowych, że progi bramek MLP są w praktyce sprzężeniem z kierunkiem odniesienia niesionym w strumieniu; nasz próg `b_c + w_c·β` ma podobną budowę, ale tego nie badaliśmy.
Bezwładność jest tylko analogią: w modelu nie ma prędkości ani drugiej pochodnej, więc długość nie daje pędu.

### Most 6. Ogniskowa jako punkt równowagi pułapki

Pułapka optyczna Ashkina [17] trzyma cząstkę w punkcie, gdzie suma sił znika, a jej sztywność to pochodna siły.
U nas ogniskowa to punkt, w którym pole wiązek `f(u)` w przekroju 3D wokół koloidu jest zerowe, a charakter tego punktu daje znak wartości własnych Jakobianu.
Tak samo pracuje algorytm FORMA [18], który z ruchów Browna znajduje stabilne i niestabilne punkty równowagi.
Tomihari i Karakida [19] pokazują, że analiza Jakobianu dobrze opisuje dynamikę uwagi bez funkcji energii; my robimy to samo dla pola MLP.
Wynik na s47 jest wyraźny: zero pola w zasięgu 3 promieni chmury istnieje tylko w 7,7 % z 806 checkpointów, głównie na początku uczenia.
Zwykle koloid niesie prawie jednorodny prąd wiązek, który przesuwa go o 0,3 do 0,7 promienia r99, a wokół niego pole jest siodłem z wirem, na przykład ściąga z siłą −0,19 i obraca z siłą 0,21.
W pełnej przestrzeni 192 wymiarów Jakobian jest zawsze siodłem z wirem, nigdy studnią, a udział wiru maleje z uczeniem.
Tu analogia pęka najmocniej: prawdziwa pęseta jest studnią, a nasze „ognisko” zwykle nie istnieje.
Dodatkowo zero w przekroju 3D nie musi być zerem w 192 wymiarach.

### Most 7. Rozkład pola na część potencjalną i wirową

Każdy Jakobian można rozłożyć na część symetryczną `S` i antysymetryczną `A`.
Pole jest gradientem potencjału tylko wtedy, gdy Jakobian jest symetryczny wszędzie; Im i współautorzy [20] pokazali, że nieskrępowane sieci typu autoenkoder zwykle tego nie spełniają.
Kwon, Ao i Thouless [21] dają właściwszy rozkład przy obecności szumu: siłę liniową dzielą na część z równowagą szczegółową i część, która krąży po powierzchniach stałej funkcji kosztu.
W pęsetach ten sam podział mierzą FORMA [18] i metoda Volpe i współautorów [22].
U nas rozkład jest dokładny w punkcie, ale nie daje globalnego potencjału.
Przy szumie anizotropowym (most 11) podział Ao nie pokrywa się z prostym podziałem `S` i `A`, więc liczby z aplikacji opisują geometrię pola, a nie prądy.

### Most 8. Siły niezachowawcze w pęsecie optycznej

Pęseta optyczna nie jest idealnie zachowawcza, bo ciśnienie promieniowania dokłada część wirową.
Roichman i współautorzy [23] zobaczyli to jako „fontannę prawdopodobieństwa”, Sun i współautorzy [24] jako wir Browna, a Moyses i współautorzy [25] dali teorię zaburzeń.
Pesce i współautorzy [26] zmierzyli tę część ilościowo, a Saberi i Gittes [27] pokazali, że moc rozpraszana przez wir rośnie proporcjonalnie do temperatury.
Li, Cao i Ng [28] pokazali, że siła niehermitowska może nawet zamienić siodło w stan stabilny.
Nasze pole ma tę samą algebrę: liniowy dryf z częścią antysymetryczną.
To jednak tylko analogia, bo w głębokości nie ma szumu, więc nie ma rozkładu stacjonarnego ani prądu prawdopodobieństwa.
Wir pojawia się w sensie fizycznym dopiero, gdy dodamy szum sztucznie (most 10), a nawet wtedy nasze siodło nie staje się studnią.

### Most 9. Bramka GELU jako obsadzenie Fermiego i temperatura bramek

GELU to `a·Φ(a)`, a Hendrycks i Gimpel [29] podają przybliżenie `Φ(a) ≈ 1/(1+e^{−1,702a})`.
Ta funkcja logistyczna ma kształt rozkładu Fermiego–Diraca, więc szerokość bramki to temperatura `1/1,702 ≈ 0,59` w jednostkach preaktywacji.
Dzieląc przez rozrzut preaktywacji neuronu dostajemy `T_laser = 0,59/sd(a)`: mówi, jaka część tokenów leży w miękkim przejściu, a jaka jest całkiem włączona lub wyłączona.
Logistyczne bramki z temperaturą są znane z maszyn Boltzmanna [30], gdzie neuron włącza się losowo z takim prawdopodobieństwem.
Pérez-Corral i współautorzy [31] uczą twardość bramki GELU, a Emmert-Streib [32] widział samoistne „stygnięcie” sieci w czasie uczenia, ale w innym modelu.
W biofizyce ta sama krzywa opisuje otwieranie kanałów jonowych [33].
Nasz pomiar: reflektory stygną z 2,6 do 0,7–0,8 w czasie uczenia, a w ViT-S/16 z Model Zoo późne bloki są skrajnie zimne, T od 0,03 do 0,12.
Przybliżenie Φ funkcją logistyczną jest dobre, ale „temperatura” to tylko analogia: nie ma kąpieli cieplnej, a bramka jest deterministyczna.

### Most 10. Termometr J_T

Dodajemy do koloidu szum gaussowski o wariancji `T` i liczymy średni Jakobian `J_T`.
Lemat Steina daje dokładnie `J_T = E[F(x+δ)δᵀ]/T`, więc wystarczą same wartości pola, bez pochodnych.
To narzędzie jest znane w uczeniu maszynowym: Torop i współautorzy [34] liczą tak wygładzone pochodne, Wright i współautorzy [35] propagują kowariancję przez sieć, a Pretorius i współautorzy [36] badają sieci z szumem.
W fizyce odpowiada temu mikroskop siły fotonowej, w którym ruch cieplny próbkuje pole wokół punktu [22], a wir Browna zależy od temperatury w nieoczywisty sposób [24] [27].
Nasz wynik: siodło trwa w każdej temperaturze, a wir słabnie, gdy grzejemy.
Ważna granica: dla pola liniowego `J_T` nie zależy od `T`, więc zmiana z temperaturą mierzy nieliniowość pola, a nie ciepło.

### Most 11. SGD jako dynamika Langevina i jego prądy nierównowagowe

Mandt, Hoffman i Blei [37] opisali SGD ze stałym krokiem jako proces Ornsteina–Uhlenbecka, a Jastrzębski i współautorzy [38] pokazali, że skalę szumu ustala stosunek kroku uczenia do rozmiaru paczki.
Mignacco i Urbani [39] oraz Ziyin i współautorzy [40] liczą efektywny szum dokładniej, a Ignashin i współautorzy [41] pokazują, że przy skończonym kroku SGD nie jest ruchem Browna.
Chaudhari i Soatto [42] udowodnili, że szum SGD jest bardzo anizotropowy, niskiego rzędu, i daje zamknięte pętle zamiast równowagi; Kunin i współautorzy [43] widzą oscylacje w przestrzeni fazowej i anomalną dyfuzję.
U nas klatka drgań tokenu skaluje się z długością kroku optymalizatora jak temperatura, co zgadza się z tym obrazem.
To jednak przybliżenie. Uczymy AdamW z obcięciem gradientu do normy 1, więc szum nie jest biały ani izotropowy.
Wir w polu wiązek (zegar głębokości) i pętle SGD (zegar uczenia w przestrzeni wag) to różne zjawiska, choć oba są nierównowagowe.

### Most 12. Relacja Einsteina i relacje fluktuacyjno-dyssypacyjne

Yaida [44] wyprowadził dokładne relacje fluktuacyjno-dyssypacyjne dla stanu stacjonarnego SGD, w przestrzeni wag, i zauważył, że krok uczenia razy ślad kowariancji szumu gra rolę temperatury.
Harada i Sasa [45] pokazali, że naruszenie takiej relacji mierzy rozpraszanie energii, a Seifert [46] zbiera całą termodynamikę stochastyczną.
Nasz pomiar w zegarze uczenia: dla średniej skoków koloidu relacja Einsteina jest spełniona z temperaturą około 3,6·10⁻⁴, a dla samego koloidu nie.
Średnią skoków liczymy z ruchu koloidu i z przyszłych kroków, więc nie jest przyczyną ruchu; spełniona relacja mówi o statystyce uśrednienia, nie o pułapce.
Nie ma też nadążania: nikt nikogo nie goni, a korelacja przesunięć w zerowym opóźnieniu jest dodatnia, co pasuje do wspólnego zejścia.

### Most 13. Weight decay jako sprężyna

AdamW Loshchilova i Huttera [47] odejmuje w każdym kroku `η·λ·θ`, czyli dokłada liniową siłę ciągnącą wagi do zera.
W przestrzeni wag to dokładnie sprężyna harmoniczna, a z szumem SGD daje proces Ornsteina–Uhlenbecka jak w [37].
W naszych pomiarach to jedyna sprężyna w czasie uczenia.
W przestrzeni koloidu działa jednak tylko pośrednio, przez Jakobian tokenu względem wag, więc tam jest przybliżeniem pierwszego rzędu.
LayerNorm dodatkowo usuwa wpływ skali części wag, więc sprężyna w wagach nie musi dawać sprężyny w położeniu tokenu.

### Most 14. Wspólne wagi jako ośrodek sprzęgający koloidy

Przesunięcie tokenu w jednym kroku to do pierwszego rzędu `δx_i = J_i δθ`, gdzie `δθ` jest jedno dla wszystkich.
To jest język jądra stycznego NTK Jacota i współautorów [48]; He i Su [49] nazwali to lokalną sprężystością sieci, a Zhang, Wang i Su [50] zbudowali z tego równania stochastyczne.
W fizyce odpowiada temu wspólne wymuszenie, które synchronizuje niezależne cząstki [51], i w ekologii nazywa się efektem Morana.
Pashakhanloo i Koulakov [52] liczą dryf reprezentacji pod szumem SGD w małej sieci liniowej, co jest najbliższym teoretycznym odpowiednikiem naszego ruchu koloidu.
Nasz pomiar: 85 do 90 procent energii kroku to wspólne pole, sąsiednie tokeny korelują od 0,6 do 0,85, a różnice ruchliwości między tokenami są zamrożone.
W pęsetach podobne korelacje sąsiadów daje sprzężenie hydrodynamiczne [53], ale tam ośrodek ma własną dynamikę, a u nas „ośrodkiem” jest jeden wspólny krok wag.

### Most 15. Pomiary z pęsety przeniesione na tokeny

Fizycy kalibrują pęsetę przez ekwipartycję, widmo mocy z częstością odcięcia i dopasowanie trajektorii; dobrym przewodnikiem jest samouczek Gieselera i współautorów [54], praca Berg-Sørensen i Flyvbjerga [55] oraz ogólna rama Pérez-García i współautorów [56].
FORMA [18] odtwarza pole siły z przesunięć i odróżnia część zachowawczą od wirowej.
Przeciągana pułapka ma teorię i pomiary pracy oraz ciepła [57] [58], z czasem relaksacji, który decyduje, czy cząstka nadąża.
U nas mamy coś, czego fizyk nie ma: pole i Jakobian znamy dokładnie, więc każdy estymator można sprawdzić z prawdą.
Mamy też coś gorszego: brak niezależnego szumu cieplnego, brak stałej pułapki i brak nadążania.
Dlatego metody są przenośne jako narzędzia, ale wynik „sztywność” albo „temperatura” trzeba czytać jako opis statystyki, nie jako wielkość termodynamiczną.

## Gdzie analogia pęka

Pierwsze pęknięcie to brak równowagi termodynamicznej. Uczenie nigdy nie dochodzi do stanu stacjonarnego, krok uczenia i stan AdamW zmieniają się, a model wychodzi z ciągle innej „pułapki”.
Drugie to szum. Szum paczki jest anizotropowy, niskiego rzędu i nie jest biały [42] [41], a do tego jest przycinany przez obcięcie gradientu, więc pojęcie jednej temperatury jest przybliżeniem.
Trzecie to głębokość. W zegarze głębokości mamy tylko 6 kroków bez relaksacji i bez szumu, więc nie ma czego uśredniać w czasie i nie ma rozkładu stacjonarnego.
Czwarte to sama pułapka. Zero pola wiązek istnieje rzadko, tylko w 7,7 % checkpointów, a w pełnych 192 wymiarach Jakobian jest siodłem z wirem, nigdy studnią.
Piąte to średnia skoków. Liczymy ją z ruchu koloidu i z przyszłych kroków, więc nie może być przyczyną tego ruchu, nawet jeśli spełnia relację Einsteina.
Szóste to niezależność cząstek. Koloidy nie są niezależne, bo wszystkie przesuwa jeden krok wspólnych wag; korelacje sąsiadów i zamrożona heterogeniczność wynikają z tego wspólnego wymuszenia, a nie z oddziaływań par.
Siódme to rzut. Rysunki w aplikacji to przekroje 3D, a Antognini i Sohl-Dickstein [59] ostrzegają, że rzuty losowych błądzeń wyglądają na uporządkowane nawet wtedy, gdy nic w nich nie ma.
Wreszcie klasa materiału: sieci porównywano ze szkłami [60] [61], ale nasze tokeny zachowują się raczej jak kryształ przypięty kodowaniem pozycji [62], z drganiami i wspólnym przesuwem, a nie jak szkło.

## Co jest nowe u nas względem literatury

Uczciwie: wiele elementów jest znanych.
Opis MLP jako pamięci klucz–wartość jest znany [9], cosinus wejście–wyjście neuronu też [11] [12], a tokeny jako cząstki w głębokości opisano dokładnie [5] [6] [7] [8].
Znane są także termodynamiczne obrazy SGD w przestrzeni wag [37] [42] [44], dryf reprezentacji w małych sieciach [52] i porównania ze szkłami [60] [61].

Nowe u nas, według przeszukań z 2026-10-12 do 2026-10-14, są następujące rzeczy.
Pierwsza to śledzenie tokenów jednego zdjęcia w czasie uczenia, przez 806 checkpointów, jako cząstek z pełną diagnostyką miękkiej materii; przeszukanie o wcześniejszych pracach nie znalazło takiego połączenia.
Najbliżej jest Halperin [63], który rysuje uczenie jako ruchomą chmurę cząstek z macierzy odległości, ale bez pól sił, Jakobianów i pomiarów z pęsety.
Druga to zastosowanie do transformera wizyjnego z detekcją, a nie do modeli językowych, w których powstała większość prac o neuronach.
Trzecia to temperatura bramek `T_laser` z szerokości GELU i jej stygnięcie w czasie uczenia; najbliższe są twardość bramki [31] i „temperatura neuronowa” w innym modelu [32], ale nie ta definicja.
Czwarta to termometr `J_T` z lematu Steina, użyty do odróżnienia siodła od wiru; narzędzie matematyczne jest znane [34], ale nie jego odczyt jako termometr pola.
Piąta to pułapka zdefiniowana jako ogniskowa wiązek i wynik, że prawdziwej studni prawie nigdy nie ma.
Szósta to prawo pojedynczego lasera `(1−κ²)/(1+κ²)` i odczytanie κ jako przełącznika między wzmacnianiem, sprężyną i wirem.
Siódma to przeniesienie zestawu pomiarów z pęsety (kalibracja, FORMA, relacja Einsteina, test nadążania) na tokeny, przy znanym dokładnie polu.
Wszystkie te punkty są „nowe” tylko w tym sensie, że przeszukania ich nie znalazły; nie jest to dowód, że nikt ich nie zrobił.

## Literatura

Numeracja jak w tekście. DOI w postaci linku; dla prac bez DOI podany jest arXiv.

1. Weinan E. *A Proposal on Machine Learning via Dynamical Systems.* 2017. https://doi.org/10.1007/s40304-017-0103-z
2. E. Haber, L. Ruthotto. *Stable architectures for deep neural networks.* 2017. https://doi.org/10.1088/1361-6420/aa9a90
3. R. T. Q. Chen i in. *Neural Ordinary Differential Equations.* 2018. https://arxiv.org/abs/1806.07366
4. M. E. Sander, P. Ablin, G. Peyré. *Do Residual Neural Networks discretize Neural Ordinary Differential Equations?* 2022. https://doi.org/10.48550/arXiv.2205.14612
5. Y. Lu, Z. Li, D. He i in. *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View.* 2019. https://doi.org/10.48550/arXiv.1906.02762
6. B. Geshkovski, C. Letrouit, Y. Polyanskiy, P. Rigollet. *The emergence of clusters in self-attention dynamics.* 2023. https://doi.org/10.52202/075280-2493
7. B. Geshkovski, C. Letrouit, Y. Polyanskiy, P. Rigollet. *A mathematical perspective on Transformers.* 2023. https://doi.org/10.48550/arXiv.2312.10794
8. T. Mudarisov, M. Burtsev, R. State. *Feed-Forward Steering in Transformer Residual Dynamics.* 2026. https://arxiv.org/abs/2608.02071
9. M. Geva, R. Schuster, J. Berant, O. Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* 2021. https://doi.org/10.18653/v1/2021.emnlp-main.446
10. S. Mei, A. Montanari, P.-M. Nguyen. *A mean field view of the landscape of two-layer neural networks.* 2018. https://doi.org/10.1073/pnas.1806579115
11. S. Gerstner, H. Schütze. *Understanding Gated Neurons in Transformers from Their Input-Output Functionality.* 2025. https://doi.org/10.48550/arXiv.2505.17936
12. S. Gerstner, H. AlQuabeh, K. Inui, H. Schütze. *Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence.* 2026. https://arxiv.org/abs/2609.18612
13. A. Gupta, A. Ozdemir, G. Anumanchipalli. *Geometric Interpretation of Layer Normalization and a Comparative Analysis with RMSNorm.* 2024. https://arxiv.org/abs/2409.12951
14. P. M. Riechers. *Geometry and Dynamics of LayerNorm.* 2024. https://doi.org/10.48550/arXiv.2405.04134
15. S. Brody, U. Alon, E. Yahav. *On the Expressivity Role of LayerNorm in Transformers' Attention.* 2023. https://doi.org/10.48550/arXiv.2305.02582
16. O. Tuomi. *Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction.* 2026. https://arxiv.org/abs/2609.31956
17. A. Ashkin, J. M. Dziedzic, J. E. Bjorkholm, S. Chu. *Observation of a single-beam gradient force optical trap for dielectric particles.* 1986. https://doi.org/10.1364/OL.11.000288
18. L. Pérez García, J. Donlucas Pérez, G. Volpe i in. *High-performance reconstruction of microscopic force fields from Brownian trajectories.* 2018. https://doi.org/10.1038/s41467-018-07437-x
19. A. Tomihari, R. Karakida. *Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians.* 2025. https://doi.org/10.48550/arXiv.2505.19458
20. D. J. Im, M. I. Belghazi, R. Memisevic. *Conservativeness of Untied Auto-Encoders.* 2016. https://doi.org/10.1609/aaai.v30i1.10268
21. C. Kwon, P. Ao, D. J. Thouless. *Structure of stochastic dynamics near fixed points.* 2005. https://doi.org/10.1073/pnas.0506347102
22. G. Volpe, G. Volpe, D. Petrov. *Brownian motion in a nonhomogeneous force field and photonic force microscope.* 2007. https://doi.org/10.1103/PhysRevE.76.061118
23. Y. Roichman, B. Sun i in., D. G. Grier. *Influence of nonconservative optical forces on the dynamics of optically trapped colloidal spheres: the fountain of probability.* 2008. https://doi.org/10.1103/PhysRevLett.101.128301
24. B. Sun, J. Lin i in., D. G. Grier. *Brownian vortexes.* 2009. https://doi.org/10.1103/PhysRevE.80.010401
25. H. W. Moyses, R. O. Bauer i in., D. G. Grier. *Perturbative theory for Brownian vortexes.* 2015. https://doi.org/10.1103/PhysRevE.91.062144
26. G. Pesce, G. Volpe i in. *Quantitative assessment of non-conservative radiation forces in an optical trap.* 2009. https://doi.org/10.1209/0295-5075/86/38002
27. I. Saberi, F. Gittes. *Nonconservative forcing and diffusion in refractive optical traps.* 2011. https://doi.org/10.1364/JOSAB.28.002369
28. X. Li, Y. Cao, J. Ng. *Non-Hermitian non-equipartition theory for trapped particles.* 2024. https://doi.org/10.1038/s41467-024-46058-5
29. D. Hendrycks, K. Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. https://arxiv.org/abs/1606.08415
30. D. H. Ackley, G. E. Hinton, T. J. Sejnowski. *A Learning Algorithm for Boltzmann Machines.* 1985. https://doi.org/10.1207/s15516709cog0901_7
31. C. Pérez-Corral, A. Fernández-Hernández i in. *λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks.* 2026. https://doi.org/10.48550/arXiv.2603.21991
32. F. Emmert-Streib. *Self-organized annealing in laterally inhibited neural networks shows power law decay.* 2004. https://arxiv.org/abs/cond-mat/0401633
33. S. Chowdhury, B. Chanda. *Estimating the voltage-dependent free energy change of ion channels using the median voltage for activation.* 2012. https://doi.org/10.1085/jgp.201110722
34. M. Torop, A. Masoomi, D. Hill i in. *SmoothHess: ReLU Network Feature Interactions via Stein's Lemma.* 2023. https://doi.org/10.48550/arXiv.2311.00858
35. O. Wright, Y. Nakahira, J. M. F. Moura. *An Analytic Solution to Covariance Propagation in Neural Networks.* 2024. https://doi.org/10.48550/arXiv.2403.16163
36. A. Pretorius i in. *Critical initialisation for deep signal propagation in noisy rectifier neural networks.* 2018. https://arxiv.org/abs/1811.00293
37. S. Mandt, M. D. Hoffman, D. M. Blei. *Stochastic Gradient Descent as Approximate Bayesian Inference.* 2017. https://doi.org/10.48550/arXiv.1704.04289
38. S. Jastrzębski i in., A. Storkey. *Three Factors Influencing Minima in SGD.* 2017. https://arxiv.org/abs/1711.04623
39. F. Mignacco, P. Urbani. *The effective noise of stochastic gradient descent.* 2022. https://doi.org/10.1088/1742-5468/ac841d
40. L. Ziyin i in., M. Ueda. *Strength of Minibatch Noise in SGD.* 2021. https://arxiv.org/abs/2102.05375
41. I. Ignashin, A. Radovskaya i in., A. Beznosikov. *Why SGD is not Brownian Motion: A New Perspective on Stochastic Dynamics.* 2026. https://doi.org/10.48550/arXiv.2605.22644
42. P. Chaudhari, S. Soatto. *Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks.* 2018. https://doi.org/10.1109/ITA.2018.8503224
43. D. Kunin, J. Sagastuy-Breña i in., D. Yamins. *The Limiting Dynamics of SGD: Modified Loss, Phase-Space Oscillations, and Anomalous Diffusion.* 2023. https://doi.org/10.1162/neco_a_01626
44. S. Yaida. *Fluctuation-dissipation relations for stochastic gradient descent.* 2018. https://doi.org/10.48550/arXiv.1810.00004
45. T. Harada, S. Sasa. *Equality connecting energy dissipation with a violation of the fluctuation-response relation.* 2005. https://doi.org/10.1103/PhysRevLett.95.130602
46. U. Seifert. *Stochastic thermodynamics, fluctuation theorems and molecular machines.* 2012. https://doi.org/10.1088/0034-4885/75/12/126001
47. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization* (wersja arXiv: *Fixing Weight Decay Regularization in Adam*). 2017. https://arxiv.org/abs/1711.05101
48. A. Jacot, F. Gabriel, C. Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* 2018. https://doi.org/10.48550/arXiv.1806.07572
49. H. He, W. Su. *The Local Elasticity of Neural Networks.* 2019. https://arxiv.org/abs/1910.06943
50. J. Zhang, H. Wang, W. Su. *Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations.* 2021. https://arxiv.org/abs/2110.05960
51. S. Lise, A. Maritan, M. Swift. *Langevin equations coupled through correlated noises.* 1999. https://doi.org/10.1088/0305-4470/32/28/305
52. F. Pashakhanloo, A. A. Koulakov. *Stochastic Gradient Descent-induced drift of representation in a two-layer neural network.* 2023. https://doi.org/10.48550/arXiv.2302.02563
53. J.-C. Meiners, S. R. Quake. *Direct Measurement of Hydrodynamic Cross Correlations between Two Particles in an External Potential.* 1999. https://doi.org/10.1103/PhysRevLett.82.2211
54. J. Gieseler i in., G. Volpe. *Optical tweezers — from calibration to applications: a tutorial.* 2021. https://doi.org/10.1364/AOP.394888
55. K. Berg-Sørensen, H. Flyvbjerg. *Power spectrum analysis for optical tweezers.* 2004. https://doi.org/10.1063/1.1645654
56. L. Pérez-García, M. Selin, A. Ciarlo i in. *Optimal calibration of optical tweezers with arbitrary integration time and sampling frequencies: a general framework.* 2023. https://doi.org/10.1364/BOE.495468
57. R. van Zon, E. G. D. Cohen. *Stationary and transient work-fluctuation theorems for a dragged Brownian particle.* 2003. https://doi.org/10.1103/PhysRevE.67.046102
58. A. Imparato, L. Peliti, G. Pesce i in. *Work and heat probability distribution of an optically driven Brownian particle: theory and experiments.* 2007. https://doi.org/10.1103/PhysRevE.76.050101
59. J. Antognini, J. Sohl-Dickstein. *PCA of high dimensional random walks with comparison to neural network training.* 2018. https://arxiv.org/abs/1806.08805
60. M. Baity-Jesi, C. Cammarota i in., G. Biroli. *Comparing dynamics: deep neural networks versus glassy systems.* 2019. https://doi.org/10.1088/1742-5468/ab3281
61. M. Kerr Winter, L. M. C. Janssen. *Glassy dynamics in deep neural networks: A structural comparison.* 2025. https://doi.org/10.1103/PhysRevResearch.7.023010
62. M. Mannes. *Positional Encodings Anchor Spatial Structure in Vision Transformers: A Geometric Perspective on Robustness.* 2026. https://doi.org/10.48550/arXiv.2606.00124
63. I. Halperin. *Learning as Observable Matrix Dynamics: Diffusive Relaxations versus Phase Transitions.* 2026. https://doi.org/10.48550/arXiv.2606.29679

## Skąd ta literatura

Prace pochodzą z workspace'u Undermind ViT (`40294fa6-c9a3-4469-9a61-d3eddeaee893`): z istniejących głębokich przeszukań o pęsecie i MLP, o obrazie koloidu, o temperaturze w sieciach, o sztywności zależnej od temperatury, o pomiarach w pęsetach, o wcześniejszych pracach o trajektoriach tokenów, o szkłach, o klasie dynamicznej i o krysztale, oraz z kilku wyszukiwań semantycznych (SGD i Langevin, prądy nierównowagowe SGD, sieci ODE, AdamW, NTK, cosinus wejście–wyjście neuronu).
Metadane i brak wycofań sprawdzono w Scite dla większości prac z DOI; szczegóły są w raporcie sesji.
