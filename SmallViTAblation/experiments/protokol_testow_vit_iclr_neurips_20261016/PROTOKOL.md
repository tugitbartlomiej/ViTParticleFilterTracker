# Protokół mechanistycznego badania ViT pod ICLR / NeurIPS

**Wersja 1.2 po drugiej recenzji, 16 października 2026. Status: przygotowany plan testów, bez nowych pomiarów modelu i bez wysyłania zadań na Eden.** ICLR i NeurIPS są konferencjami. Ten protokół określa standard dowodu dla naszej tezy; nie jest oficjalną checklistą liczby eksperymentów ani obietnicą akceptacji.

## 1. Co chcemy wykazać

Główne pytanie: **czy wyuczona organizacja lokalnych przekształceń tokenów selektywnie przenosi informację potrzebną do lokalizacji obiektu, a ogranicza wpływ zmian nieistotnych dla zadania?**

W języku mikroskopu: nie wystarczy zobaczyć, że przy cząsteczce występuje rozciąganie albo obrót. Trzeba wskazać, jaki składnik obliczenia odpowiada za określoną reakcję ramki, przewidzieć skutki jego zmiany, wykonać interwencję i sprawdzić przewidywanie na nowych obrazach i treningach.

Hipotezy z rozdzielonym zakresem:

- **H1, główna:** orientacja wybranych lokalnych operatorów ma specyficzną funkcję dla lokalizacji, której nie wyjaśnia sama norma zaburzenia, czułość ani selekcja detekcji.
- **Poziom dowodu H1:** sam P1 testuje kierunkową podatność, a mocniejszy wniosek mechanistyczny wymaga prospektywnego przewidywania i związanych z tym samym procesem T11/T15. Nie zakładamy, że rozkład S/A wnosi informację ponad pełną pochodną.
- **H2, uczenie:** funkcjonalna organizacja zmienia się przez trening i przekracza efekt inicjalizacji oraz kontroli niezwiązanych z zadaniem. Atlas pokazuje rozwój; wyjaśnienie przyczyny kroku wymaga T19.
- **H3, szlak obliczenia:** możemy wskazać, przez które attention/MLP informacja dociera do głowy, z uwzględnieniem redundancji i kompensacji.
- **H4, warunkowa:** podpisane wkłady neuronów tworzą powtarzalną współpracę albo korektę między blokami istotną dla ramki.
- **H5, opcjonalna:** pomiary pomagają prognozować błąd detekcji bez GT ponad mocne baseline'y.

Nie zakładamy z góry, że silniejszy obrót jest lepszy, słabszy obrót jest lepszy, siodło jest złe lub końcowy token musi się zatrzymać. Nie wymagamy wszystkich pięciu hipotez w jednym artykule. Główna historia ma zostać wybrana i zamrożona po discovery, zanim obejrzymy confirmation.

## 2. Pliki i sposób korzystania

**Tryb pracy ustalony z użytkownikiem:** kosztowne obliczenia i treningi wykonujemy na Edenie, następnie pobieramy biegi do lokalnego oglądania i interpretacji. Każdy test ma wyeksportować surowe wyniki, metadane oraz zaplanowane dane do mikroskopu; zwykła analiza zapisanych wyników ma działać bez ponownej inferencji. Obowiązują pakiety analysis/viewer/replay oraz próba pobrania i lokalnego odtworzenia już w E0, opisane w EDEN_I_STATYSTYKA.md, sekcja 0.

| Dokument | Zawartość |
|---|---|
| [TESTY.md](TESTY.md) | 30 kart T00–T29: hipoteza, procedura, kontrole, metryki, falsyfikacja, zakres i planowany wynik |
| [TESTY.json](TESTY.json) | Ten sam katalog w formie maszynowej; wszystkie statusy planned_not_executed |
| [EDEN_I_STATYSTYKA.md](EDEN_I_STATYSTYKA.md) | Etapy, macierz danych i seedów, bramki, koszt i reguły analizy |
| [KAMPANIA.json](KAMPANIA.json) | Kontrakt przyszłego wykonania; null oznacza wymagane ustalenie, nie bezpieczny domyślny parametr |
| [LITERATURA.md](LITERATURA.md) | Zweryfikowane źródła, poziom przeczytania i co każde zmienia w projekcie testu |
| [Druga recenzja przedpublikacyjna](../../outputs/protokol_testow_vit_iclr_neurips_20261016/RECENZJA_PRZEDPUBLIKACYJNA.md) | Sześć poprawek, ograniczony rdzeń obliczeń i bramka oceny rzeczywistych wyników |

Szczegółowe załączniki: [geometria G01–G28](../../outputs/protokol_testow_vit_iclr_neurips_20261016/AUDYT_GEOMETRIA.md), [przyczynowość C](../../outputs/protokol_testow_vit_iclr_neurips_20261016/AUDYT_PRZYCZYNOWOSC.md), [dane i statystyka](../../outputs/protokol_testow_vit_iclr_neurips_20261016/AUDYT_STATYSTYKA_EDEN.md). Są rozwinięciem kart T, nie kolejną niezależną kampanią. Przy różnicy przykładowych progów, znaku kontrastu lub wariantów pierwszeństwo mają niniejszy protokół i zamrożona wersja KAMPANIA.json.

Rozwijamy [plan z wcześniejszej analizy](../plan_publikacji_neurips_20261016/PLAN.md), zachowując historyczne wyniki. Nie zmieniono aplikacji. W tym zadaniu przygotowano specyfikacje; **nie powstał jeszcze zweryfikowany runner uruchamiający cały katalog**.

## 3. Rzeczywisty układ, który mierzymy

Sprawdzony kod: [model.py](../maly_vit_od_zera_35000_21epok_20261009/kod/model.py).

- Hybrydowa łodyga konwolucyjna, z BN w części warstw, stałe pozycje 2D sin-cos.
- Sześć bloków, domyślnie po trzy głowy attention; w każdym najpierw attention, potem MLP.
- Dla wejścia 480×320: 600 przestrzennych tokenów. Stan residualu ma **192 współrzędne**; ukryte aktywacje MLP po GELU mają **768 współrzędnych**. To różne miejsca w grafie.
- Jeden reflektor reprezentuje neuron MLP; jego wkład na token j to podpisane u_jc=z_jc v_c w 192D. Wszystkie reflektory danego MLP korzystają z tego samego wejścia tokenu, a ich wkłady sumują się. Nie odpalają się kolejno c=1,…,768.
- Token po attention zawiera kontekst innych miejsc. Siatka patchy nie jest segmentacją treści tokenu; konwolucyjny stem ma zachodzące pola recepcyjne.
- Głowa zwraca logit środka i cztery log-odległości do krawędzi. Strata jest sumą focal heat loss i dwukrotnej straty GIoU.
- Rzeczywisty dekoder używa sigmoid, maski lokalnych maksimów 3×3, top-k (domyślnie 10) i exp(ltrb)×16. Analiza top1 w aplikacji nie zastępuje audytu całego dekodera.

Przy obecnym eval i tożsamościowym LayerScale:

\[
F_M(x)=W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(x)+b_1)+b_2,\qquad
\Psi_M(x)=x+F_M(x).
\]

Dla całego obrazu:

\[
X_a=X+F_A(X),\qquad X_{\rm out}=X_a+F_M(X_a).
\]

Sześć bloków to 12 różnych map. Drugim zegarem jest trening: \(x_t=f(I;\theta_t,b_t)\), gdzie b obejmuje bufory BN. Pole MLP nie jest równaniem ruchu między checkpointami.

## 4. Matematyka obowiązująca wszystkie testy

### 4.1 Pole i skończony krok

\[
J=DF,\qquad S=(J+J^\top)/2,\qquad A=(J-J^\top)/2,\qquad M=I+J.
\]

Dla małej różnicy \(\delta\) porównujemy odpowiedzi dwóch bliskich wejść:

\[
\Psi(x+\delta)-\Psi(x)\approx M\delta,\qquad
\|M\delta\|^2-\|\delta\|^2=2\delta^\top S\delta+\|J\delta\|^2.
\]

Dlatego znaki S nie wystarczają do stwierdzenia kontrakcji rzeczywistego kroku. Raportujemy singular values M i odpowiedź całego suffixu. Mieszane znaki S nazywamy lokalnym rozciąganiem/ściskaniem. Dynamiczne siodło wymaga punktu równowagi i odpowiedniej analizy stabilności; A≠0 nie oznacza zamkniętej orbity. Lokalne S/A nie są globalnym rozkładem Helmholtza.

### 4.2 LayerNorm i iloraz

\[
Q=I-\mathbf1\mathbf1^\top/192,\qquad
U^\top U=I_{191},\quad UU^\top=Q,\qquad \bar J=U^\top J U.
\]

J1=0 jest dokładną symetrią pre-LN, a M1=1. Kierunek radialny nie jest dokładnie neutralny przy eps>0. Analizujemy 192D oraz iloraz 191D, zachowując prawdziwy epsilon i affine parametry LN. Centrowanie współrzędnych wewnątrz tokenu to inna operacja niż odejmowanie średniej po tokenach używane w GUI.

Metryka główna to natywna metryka euklidesowa residualu ograniczona do ilorazu LN. S/A zależą od metryki: przy innej dodatnio określonej G adjoint ma postać G⁻¹JᵀG. Kontrola obrotów ortogonalnych nie uzasadnia niezależności od dowolnej zmiany parametryzacji; zakres wniosku jest jawnie ograniczony do zamrożonej metryki.

Pełny stan ma 600×192 współrzędnych. Attention sprzęga tokeny: jego pochodne mierzymy przez JVP/VJP i wybrane bloki J_ij. Nie zastępujemy pełnej pochodnej mapą query-only z zamrożonym K/V.

### 4.3 Zegar treningu

Cztery rogi dla dowolnego miernika g(w,x):

\[
g_{ab}=g(w_{t+a},x_{t+b}),\quad a,b\in\{0,1\},
\]
\[
D_w=\tfrac12[(g_{10}-g_{00})+(g_{11}-g_{01})],\quad
D_x=\tfrac12[(g_{01}-g_{00})+(g_{11}-g_{10})].
\]

\(D_w+D_x=g_{11}-g_{00}\). Dodatkowo raportujemy interakcję \(g_{11}-g_{10}-g_{01}+g_{00}\). To dokładny rozkład zależności od wag i punktu; sam nie dowodzi mechanizmu AdamW. Dla ruchu treningowego T19 sprawdza \(D_\theta f\,\Delta\theta+D_bf\,\Delta b\) względem faktycznego ruchu, z pełnym stanem optymalizatora i resztą Taylora.

## 5. Główny eksperyment przyczynowy — T09

### Co zmieniamy

Dla czystego obrazu zamrażamy stan x0 oraz \(\bar J_0\). Z obrazu zmienionego zgodnie z T10 otrzymujemy x. W wybranym MLP stosujemy:

\[
F^{B}_{\alpha}(x)=F(x)-\alpha\,U B_0 U^\top(x-x_0),\qquad
\alpha\in\{0,0.5,1\}.
\]

B to \(\bar A_0\), \(\bar S_0\), dodatnia lub ujemna część \(\bar S_0\), wybrana w discovery. Zostawiamy niezmienione wagi pozostałych bloków i uruchamiamy cały rzeczywisty downstream.

**Token interwencji w podstawowym T09 to j=i0**, czyli kandydat wybrany na oryginalnym clean forward. Indeks jest stały we wszystkich ramionach danego seed×checkpoint×obrazu/obiektu, lecz może zmieniać się między modelami i checkpointami. Regionalna próbka atlasu nie jest automatycznie populacją T09. Interwencje j≠i0 mają osobny kontrast: w ostatnim MLP dają strukturalnie zerową zmianę współrzędnych i0, bo dalsza LN i głowa są per-token; mogą jednak zmienić ranking końcowego dekodera. Wcześniejsze bloki mają downstream attention, więc ich wpływ między tokenami wymaga pomiaru T07/T11.

Interwencja zachowuje wartość F w x0 i symetrię przesunięcia LN. **Nie mierzy poprawy niezmienionego obrazu:** przy δ=0 efekt jest zerowy konstrukcyjnie. Nie jest globalnym „wyłączeniem wiru”, tylko lokalną zmianą reakcji. Surowego A192 nie odejmujemy bez kontroli, ponieważ J1=0 nie implikuje A1=0.

Wariant podstawowy ogranicza próbę do zwalidowanego promienia T05. Jeśli potrzebna jest funkcja odcięcia χ, definiujemy jej promień przed confirmation, a χ=1 w testowanym obszarze; nie wnioskujemy, że jej pochodna poza tym obszarem nadal równa się −αB.

### Co porównujemy

1. Oryginalny forward.
2. Interwencja ukierunkowana.
3. Sham \(B_{\rm sham}=O B O^\top\), gdzie O jest wylosowanym obrotem w 191D. Zachowuje właściwości widmowe B i jego symetrię/antysymetrię.
4. Oddzielny sham dopasowany do **normy faktycznej poprawki** \(\|B\bar\delta\|\). Zgodna norma operatora nie zapewnia zgodnej dawki na konkretnym wejściu.
5. Baseline pełnego JVP i porównania directional/coupling określone w T26.

**Główna kontrola dla P1 to sham dopasowany normą faktycznej poprawki z punktu 4.** Sham zachowujący widmo z punktu 3 raportujemy oddzielnie; nie mieszamy obu rodzajów w jednej średniej.

Sham dopasowany per δ jest kontrolą skończonej dawki; nie przedstawiamy go jako pochodnej tego samego liniowego pola dla wszystkich wejść. Małe mianowniki i niedopasowane kontrole oznaczamy jawnie. Nie przeskalowujemy arbitralnie prawie zerowych poprawek do wielkich wartości.

Start: 8 orientacji sham; sprawdzamy stabilność Monte Carlo na 8/16/32 na podpróbie discovery. Shams nie są nowymi niezależnymi treningami. W confirmation zamrażamy ich liczbę, ziarna i sposób agregacji.

### Co uznajemy za wynik

Dla kandydata i0 wybranego przez oryginalny clean przebieg, z ustalonym przypisaniem do docelowego GT, definiujemy:

\[
e_{\rm edge}(b,g)=
\frac{\sqrt{\frac14\sum_{k=1}^{4}(b_k-g_k)^2}}
{\max(\mathrm{diag}(g),d_{\min})}.
\]

d_min to z góry ustalony próg rozmiaru w pikselach wynikający z rozdzielczości adnotacji; nie wybieramy go według efektu. Pixel RMS raportujemy obok. Dla transformacji geometrycznej używamy przekształconego GT.

**Kontrola osiągalności ramki:** dodatnie exp(ltrb) wymuszają, że bbox zawiera kotwicę ((col+0.5)×16,(row+0.5)×16). Główne P1 wybiera rodzinę nuisance: jasność albo kontrast, zamrożoną po discovery. W teście geometrycznym P3 stały kandydat jest poprawnym porównaniem tylko w prerejestrowanym zakresie, w którym kotwica pozostaje wewnątrz obu GT z ustalonym marginesem. Reguła kwalifikacji opiera się na clean kandydacie i GT, przed wynikami interwencji. Alternatywą jest odrębny endpoint z deterministycznie przekształconą kotwicą, bez wyboru według IoU i z kontrolą kwantyzacji siatki. Przypadki poza wsparciem pozostają w pełnej ocenie dekodera; raportujemy ich liczbę. Nie przypisujemy tego ograniczenia głowy polu MLP.

**P1:** średnia sparowana różnica \(D=e_{\rm target}-\mathrm{mean}(e_{\rm sham})\), gdzie sham oznacza kontrolę dopasowaną normą faktycznej poprawki; najpierw po ustalonych perturbacjach/klatkach, potem równo po filmach i seedach. Dodatnie D oznacza większy błąd po interwencji ukierunkowanej niż po sham. To podstawowa konwencja znaku; pomocnicza notatka statystyczna pokazuje także odwrotny znak jako „korzyść”.

Przed confirmation zamrażamy operator, warstwę lub agregację warstw, etap treningu, rodzinę perturbacji, α i praktyczny minimalny efekt. Jeżeli discovery wskazuje konkretną rolę, przewidywany znak zapisujemy przed testem; w innym przypadku testujemy dwustronnie i nie dopisujemy korzystnego kierunku po wynikach.

**P2, jeżeli twierdzimy „wyuczone”:** różnica P1 między z góry wybranym późnym etapem a inicjalizacją. **P3, jeżeli twierdzimy „selektywne”:** wcześniej określony kontrast odpowiedzi na zmianę geometrii obiektu i na zakłócenie, z odpowiednią ekwiwariancją GT. Claim wymagający obu rodzin jest koniunkcją, nie sukcesem w wygodniejszej rodzinie. Opcjonalny endpoint prognozowania jakości z T25 nazywamy Q1; nie utożsamiamy go z P3 selektywności, mimo roboczego nazewnictwa w pomocniczym audycie.

Fixed-candidate jest główną ciągłą miarą mechanizmu regresji. Równolegle T16 ocenia IoU i rzeczywisty wybór detekcji. Bez potwierdzenia na finalnym dekoderze nie piszemy, że mechanizm poprawia końcową detekcję. Regułę przypisania obiektu w obrazach wieloobiektowych zamrażamy w manifeście przed predykcjami; nie wybieramy GT dającego najlepszy IoU po każdej interwencji. Brak detekcji nie jest powodem usunięcia przykładu.

P2 dodatkowo wymaga kontroli tej samej deterministycznie wybranej kotwicy obiektu i tej samej populacji obrazów między init/late i seedami, z jawną regułą osiągalności oraz pokrycia. Wybór z GT jest analizą mechanistyczną z etykietami. Bez tego ograniczamy wniosek do zmiany modelu łącznie z wyborem kandydata, zamiast przypisywać całość zmianie reprezentacji tego samego tokenu.

### Co musi rozstrzygnąć cały pakiet mechanistyczny

Dla q=−αUBUᵀδ pierwszy rząd efektu na wyjściu to Cq, gdzie C jest pochodną suffixu. Dopasowanie ||q|| nie dopasowuje Cq, dlatego dodatni P1 może wynikać ze zwykłej kierunkowej czułości. Przed confirmation zamrażamy kartę mechanizmu M1: postulowaną operację, komponent, podpisane przewidywania dla określonych krawędzi/odczytów, kontrast zadaniowy, warunek małego efektu, zakres i minimalny efekt. Prognozy zapisujemy przed obliczeniem odpowiedzi interwencyjnej. M0 obejmuje pełny lokalny JVP i rzeczywisty dekoder; wartość M1 oceniamy jako prostsze, sprawdzalne wyjaśnienie, nie jako nową informację ponad J.

T11/T15 muszą testować ten sam komponent i proces co T09. Cofnięcie dokładnie odjętej poprawki oraz przywrócenie całego późnego stanu są tylko kontrolami dodatnimi. Główny rescue używa uprzednio wybranej ograniczonej ścieżki/komponentu i donorów rozdzielających postulowaną zmienną od kontekstu. Resztę modelu przeliczamy. Manifest zamraża selektory, budżety, dawców i ich dopasowanie do nowego seeda; przegląd filmów confirmation nie służy do wyboru komponentów głównej analizy.

Dla zamrożonego znaku s i praktycznego progu δ* oceniamy CI dla sP1: dolna granica powyżej δ* wspiera deklarowaną wielkość; górna poniżej δ* odrzuca ją w tej skali; pozostały przypadek jest nierozstrzygający. CI musi odpowiadać zadeklarowanej rodzinie hipotez. Dwustronny efekt bez prognozy kierunku pozostaje wynikiem podatności, nie samodzielnym dowodem konkretnej funkcji mechanizmu.

## 6. Cztery rodzaje „korygowania”, których nie wolno mieszać

| Zjawisko | Test | Co wynik pozwala powiedzieć |
|---|---|---|
| Wspólna suma neuronów w jednym MLP | T12–T13 | Czy podpisane wkłady jednocześnie wzmacniają/kompensują się i jak wpływają na odczyt |
| Reakcja następnego bloku na zmianę poprzedniego | T14 | Czy istnieje adaptacyjna korekta w jednym forwardzie |
| Self-repair po ablacji | T11/T15 | Czy inne komponenty przejmują funkcję po zmianie; brak efektu ablacji nie dowodzi zbędności |
| Zmiana reprezentacji podczas uczenia | T18–T19 | Jak aktualizacja parametrów i buforów zmienia pole i sondowany stan |

W testach A/B/AB definiujemy Δy_G=y_G−y_0 oraz dwie odrębne miary: q=||Δy_AB||/(||Δy_A||+||Δy_B||) opisuje kompensację, a I=Δy_AB−Δy_A−Δy_B opisuje nieaddytywność. Małe q może wystąpić przy I=0, więc nie dowodzi nieliniowej synergii. Odczyt y musi być ustaloną ciągłą wielkością, np. krawędziami stałego kandydata; progi małego mianownika zamrażamy wcześniej. Nie traktujemy samego IoU lub indeksu zwycięzcy jako podstawowej miary interakcji wewnątrz MLP.

Dla T14 nie wystarczy ujemny cosinus dwóch różnic stanów. Interweniujemy dodając αa do rzeczywistego wejścia kolejnego bloku i mierzymy \(b(x+\alpha a)-b(x)\), z kontrolą kierunków losowych o tej samej normie. To jest bezpośrednia odpowiedź na pytanie, czy kolejny blok „odkręca” wcześniejszy ruch. Korelacja opóźniona między checkpointami odpowiada innemu pytaniu. Osobny przebieg z zamrożonymi na bazowym forward średnimi i odchyleniami LN, przy zachowanym affine, pozwala oddzielić renormalizację od późniejszej korekty neuronalnej. To diagnostyczna interwencja w normalizację, nie równoważny wariant oryginalnego modelu.

## 7. Najmocniejsze porównania

Obowiązkowe dla naszej tezy: pełny JVP do wyjścia, normy F/J i singular gains I+J, gradient×activation / wpływ na bbox, gradient marginesu, dopasowane losowe grupy i kierunki, inicjalizacja, silne kontrole perturbacji.

Z literatury dokładamy:

- **Aubry i in., ICLR 2025:** coupling Jacobianów, także ViT podczas treningu. Odtworzyć definicję eq. 9 z arXiv v4 dla całych bloków: J_block=(I+J_M)(I+J_A)−I. Zamrozić K, p i obsługę niejednoznacznych baz SVD. Warianty samych podkroków i ilorazu oznaczamy jako rozszerzenia; starsza wersja artykułu ma inną miarę.
- **Tomihari i Karakida, NeurIPS 2025:** Jacobiany i składowe dynamiki; rozdzielić rekurencyjne attention od sześciu różnych bloków.
- **Patrawala i in., NeurIPS 2025:** adaptacyjna korekta następnej warstwy i jej emergence. Samo znalezienie ujemnych kierunków nie stanowi nowej tezy.
- **Patching:** Zhang/Nanda, Heimersheim/Nanda, Makelov oraz Miller. Rescue i zmiana predykcji wymagają kontroli OOD, redundancji i sposobu mierzenia faithfulness.
- **Prisma i CAAP:** istnieją narzędzia mechanistycznej analizy vision i patchowanie reprezentacji ViT. Nie deklarować pierwszego „mikroskopu ViT” bez bardzo precyzyjnego porównania.

Rozkład S/A nie zawiera informacji, której nie ma pełny J. Wkład może polegać na sprawdzalnym, prostszym wyjaśnieniu określonego działania, a nie na obietnicy pokonania dokładnej pochodnej przez jej rozkład.

## 8. Pakiety wykonania i warunki zatrzymania

**Pakiet A — poprawność:** T00–T05, T10, T28. Nie przechodzimy dalej przy błędnym grafie, pochodnych, nieprawidłowym GT lub niejasnym splicie.

**Pakiet B — tani pilot mechanizmu:** T06–T09, T11, T15–T16, T20, T26. Bazowo s47 i 24 nowe klatki z 12 VAL, zgodnie z EDEN_I_STATYSTYKA.md. T20 zaczyna od inicjalizacji i kontrolnej randomizacji; nowe null-treningi po bramce.

**Pakiet C — wybór i zamrożenie:** dwa nowe discovery seedy, atlas, T27. T12–T14 i T17 tylko jeśli wspierają wybraną główną historię. T18–T19 gdy claim dotyczy wyjaśnienia uczenia, a nie tylko działania wyuczonego modelu.

**Pakiet D — confirmation:** trzy nowe seedy jako startowy budżet, zamknięte filmy TEST po audycie i ustalona statystyka. Liczbę seedów może zmienić analiza precyzji na discovery przed odczytem confirmation.

**Pakiet E — zakres i publikacja:** T22/T23 potrzebne dla szerokiego claimu o ViT i detekcji, T24 dla odporności, T25 dla diagnostyki. T29 kończy weryfikację odtwarzalności.

Po recenzji: pełna siatka atlasu jest budżetem maksymalnym, nie warunkiem każdego pilota i confirmation. Mały T09 może użyć 216 źródłowych J dla i0 (24 obrazy×3 checkpointy×3 bloki); dodatkowe kontrole i suffix liczymy osobno. Po discovery kosztowny rdzeń confirmation obejmuje zamrożone bloki i init/late. Całą siatkę głębokość×czas wykonujemy wtedy, gdy wymaga jej claim. Pięć nowych treningów podstawowych nie obejmuje warunkowych random-label/loss ablations T20; te wymagają osobnej liczby seedów, długości okna i wyceny.

Bramka po pilocie:

- Efekt jest artefaktem projekcji/normalizacji → naprawiamy miernik i ograniczamy interpretację.
- Efekt znika względem właściwych sham → nie uruchamiamy dużej kampanii „żeby wyszło”; zmieniamy pytanie na discovery.
- Działa wyłącznie wybór top1 → historia o geometrii lokalizacji wymaga zawężenia do rankingu i selekcji.
- Działa w jednym obrazie/seedzie → brak potwierdzenia; ustalamy precyzję i replikację.
- Działa, ma przewidzianą dawkę i rescue oraz transfer → kandydat na mechanistyczny wynik artykułu.

## 9. Co trafi do artykułu

| Figura / tabela | Dowód |
|---|---|
| F1: graf 12 map i dwa zegary | Definicje i dokładność instrumentu; 192D/768D, bez dosłownej fizyki światła |
| F2: atlas warstwa×checkpoint | Co zmienia się przez trening, z inicjalizacją i niezależnymi seedami |
| F3: główny kontrast target/sham | Odpowiedź ramki po dawce i skali perturbacji, uncertainty i kontrola normy |
| F4: ślad downstream i rescue | Które obliczenia pośredniczą w efekcie oraz co przywraca wynik |
| F5: fixed-candidate / peak mask / ranking | Rozdzielenie regresji i nieciągłości dekodera |
| F6: nowe seedy / drugi model / drugi zbiór | Zakres uogólnienia; osobne wyniki i niepowodzenia |
| Tabela porównań | Pełny JVP, coupling, TLCM, gradienty, random controls oraz koszt |

Film i interfejs wspierają komunikację. Liczby muszą powstawać z deterministycznego pipeline'u także bez GUI.

Istniejące wyniki ujemne pozostają częścią historii: kompensacja występuje już przy inicjalizacji, transfer wybranych par był słaby, sam ruch nie daje uniwersalnej sygnatury wysokiego IoU, a duży znany skok ramki wynikał z przełączenia zwycięzcy. Nie można ich usuwać, bo utrudniają narrację.

## 10. Co jest gotowe, a co jeszcze trzeba wykonać

Gotowe w tym opracowaniu: 30 kart testów, wyprowadzenia i kontrole, bibliografia z weryfikacją, plan danych/statystyki, kontrakt kampanii i zależności.

Do wykonania po tym planie: implementacja i audyt runnerów, wybór/hashowanie rzeczywistych próbek, E0 benchmark na dostępnej konfiguracji Edena, pilot i analiza precyzji, zamrożenie protokołu, następnie confirmation. Nie uznajemy istniejącej aplikacji ani tego dokumentu za zaliczenie T01–T29.

Pierwsza implementacja powinna obejmować wyłącznie T00–T05/T10/T28 i mały T09. To najszybciej rozstrzygnie, czy warto ponosić koszt pełnej kampanii.

Przed wysłaniem artykułu T29 obejmuje ponowną recenzję pobranych rzeczywistych wyników według drugiej recenzji przedpublikacyjnej. Obecny audyt planu nie zastępuje tego etapu.

