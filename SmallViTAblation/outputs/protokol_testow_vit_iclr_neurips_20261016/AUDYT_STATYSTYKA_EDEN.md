# Protokół statystyki, danych i wykonania Eden: audyt do integracji

16 października 2026. Dokument nadrzędny: `experiments/plan_publikacji_neurips_20261016/PLAN.md` z dzisiejszego planu mechanizmu ViT. Audyt infrastruktury: `outputs/plan_publikacji_neurips_20261016/eden_i_dane.md`. Starszy główny `PLAN.md` z września dotyczy RF-DETR i innego korpusu; jego liczebności/podział 50 pacjentów **nie obowiązują** w tej kampanii.

Status: projekt protokołu, bez nowych treningów, pomiarów modeli, jobów Eden i zmian aplikacji. Sprawdzono pliki oraz źródła zewnętrzne. Wykonano jedynie arytmetyczną ilustrację MDE z założonych wariancji, wyraźnie odróżnioną od empirycznej analizy mocy. Jedynym nowym plikiem tego zadania jest niniejsza notatka.

## 1. Co rzeczywiście wymagają źródła konferencyjne

Aktualna strona NeurIPS Paper Checklist wymaga dołączenia checklisty i pyta m.in. o odtwarzalność, dane/hiperparametry, definicje przedziałów lub innych miar niepewności i ich źródeł oraz zasoby obliczeniowe. Trzeba rozróżniać SD, SE i CI oraz podawać metodę wyznaczenia przedziałów; koszt obejmuje także istotne próby i nieudane biegi. **Strona nie narzuca „trzech” ani „pięciu” seedów, liczby zbiorów czy mocy 80%.** Nasze liczebności są decyzją projektową. [Oficjalna checklista NeurIPS](https://nips.cc/public/guides/PaperChecklist).

ICLR 2027 zachęca do kodu oraz Reproducibility Statement wskazującego miejsce specyfikacji przetwarzania danych, założeń i reprodukcji. Nie jest to reguła ustalająca N eksperymentów. Opublikowany termin pełnych zgłoszeń tej edycji to 25 września 2026; na dzień audytu jest już po terminie. Projektować standard dowodów, nie obiecywać zgłoszenia do zamkniętej edycji ani wymyślonych terminów kolejnej. [Oficjalne Author Guidelines ICLR 2027](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Weryfikację przeprowadzono przez skill `paper-search` (CLI arXiv/Crossref) oraz strony pierwotne. Wyniki jednego zapytania CLI były nietrafne; nie wykorzystano ich jako dowodu. Istotne źródła metodologiczne i granice zastosowania:

| Źródło pierwotne | Zastosowanie tutaj |
|---|---|
| [Bouthillier i in., Accounting for Variance in ML Benchmarks, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html) | Jawne źródła wariancji: inicjalizacja, dane, tuning. Wspólne ustawienia i sparowane porównania; więcej tokenów nie usuwa wariancji treningu. |
| [Owen i Eckles, Bootstrapping data arrays of arbitrary order, AOAS 2012](https://arxiv.org/abs/1106.2125) | Niezależne wagi poziomów czynników dla danych skrzyżowanych. Pigeonhole bootstrap może być konserwatywny przy odpowiednich założeniach; nie daje automatycznej gwarancji poprawnego CI przy trzech seedach. |
| [Agarwal i in., Statistical Precipice, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Raportowanie niepewności, rozkładów i wyników per-run zamiast pojedynczego rekordu. To praca o RL: nie przenosimy bez sprawdzenia jej bootstrapu zadanie×run na skrzyżowanie seed×film. |
| [Künsch, Jackknife and bootstrap for general stationary observations, Annals of Statistics 1989](https://doi.org/10.1214/aos/1176347265) | Bloki zachowujące zależność szeregu. Trening jest niestacjonarny, więc nie traktować bootstrapu dowolnych checkpointów jako uzasadnionego tylko przez cytowanie tej pracy. |
| [Goeman i Solari, Sequential rejection principle, Annals of Statistics 2010](https://arxiv.org/abs/1211.3313) | Jawna rodzina hipotez, Holm/gatekeeping; warunkiem są poprawne testy składowe. Korekta wielokrotności nie naprawia pseudoreplikacji. |
| [Demšar, Statistical Comparisons of Classifiers, JMLR 2006](https://jmlr.org/papers/v7/demsar06a.html) | Nie przenosić testów dla wielu niezależnych datasetów na sześć bloków jednego ViT ani na skorelowane tokeny. |

## 2. Macierz discovery / confirmation — zgodna z PLAN.md

Budżet nadrzędny: **istniejący s47 + dwa pełne biegi discovery + trzy NOWE biegi confirmation**. Nie sumować ich w jedną pięcioseedową próbę potwierdzającą. Przykładowe identyfikatory nowych seedów do rejestru: discovery 1047/2047, confirmation 3047/4047/5047; są propozycją, wymagają kontroli kolizji z istniejącymi biegami. Numery same w sobie nie świadczą o niezależności; trzeba odtworzyć osobno RNG modelu, batchy i augmentacji.

| Etap | Seedy / dane | Checkpointy / instrument | Uprawnienie do wyboru metody |
|---|---|---|---|
| E0 poprawność i benchmark | s47, 4 obrazy | `[0,33000,65640]`, 6 bloków, 12 tokenów/obraz | Wyłącznie debug; bez claimów |
| E1/E2 pilot | s47, 24 klatki z 12 filmów VAL, po 2 odległe czasowo | `[0,1000,5000,15000,33000,55000,65640]`, 6 bloków, 12 tokenów do pełnego J; agregaty wszystkich 600 | Można wybrać hipotezę, skale i selektor |
| Discovery atlas | s47 + 2 nowe discovery; 250 VAL, po 10 z 25 filmów | 12 stanów poniżej, 6 bloków, 24 tokeny do J; attention osobno | Można stroić przez CV po filmach, raportować optymizm |
| Lock | brak nowych danych | Hash konfiguracji, endpointy, tolerancje, MC/power, zamrożony selektor i dekoder | Koniec wyboru na podstawie confirmation |
| Confirmation in-domain | 3 NOWE seedy; 250 klatek z 25 TEST, po 10/film; pełny zatwierdzony TEST do metryk detektora | Ta sama siatka i instrument; dodatkowo VAL tylko do jawnie przewidzianej oceny odtwarzalności, nie strojenia | Test zablokowanego protokołu; nie wybierać najlepszego bloku/ε |
| Generalization gate | nowa konfiguracja ViT + drugi detection dataset po E2/E4 | Osobny zamrożony manifest i benchmark kosztu | Osobny potwierdzający eksperyment; żadnego cofania wyników do selektora testowanego wcześniej |

12 stanów: `[0,300,1000,3000,6000,10000,16000,24000,33000,44000,55000,65640]`. Sprawdzić dostępność w manifestach przed startem. Awaryjna reguła: najbliższy stan według liczby aktualizacji, remis → wcześniejszy, maksymalna tolerancja 50 kroków, poza nią status `missing_checkpoint` i decyzja przed analizą. Nowe biegi mają zapisywać dokładne kroki; najlepiej nie potrzebować zastępstwa. Numer epoki nie zastępuje kroku, a przy zmianie datasetu porównywać również liczbę ekspozycji danych i budżet aktualizacji.

E0 daje 864 lokalizacje token–blok–checkpoint; pilot 12 096; atlas **432 000** na seed. Są to liczby pomiarów operatora, **nie N niezależnych obserwacji statystycznych**. Resampling dotyczy najwyższych właściwych jednostek.

### Filmy, pacjenci, klatki, tokeny

Potwierdzony podział Cataract-LMM: 117 TRAIN / 25 VAL / 25 TEST po grupach operacji. Patient-disjointness nie została zweryfikowana. Jeżeli audyt ujawni wiele filmów tego samego pacjenta, najwyższą jednostką resamplingu i rozdziału jest pacjent, z filmami wewnątrz. Jeśli ID pacjenta brak, deklarować uogólnienie do filmów/grup operacji, z zastrzeżeniem możliwej zależności pacjentów; nie udawać ustalonej liczby pacjentów.

24 klatki pilota: wybór z manifestu przed inferencją, stratyfikacja po etapie/narzędziu, rozdzielenie w czasie. 250 klatek atlasu: 10 przedziałów czasu na film z wyborem klatki ze środkowej części każdego przedziału przy użyciu stałego RNG; obsługa krótkich filmów i brakujących adnotacji zapisana przed wyborem. Nie wybierać „najładniejszych” albo najlepiej rozpoznanych klatek. Hashy zdjęć, sekwencji i duplikatów nie zastępuje sam różny numer klatki.

Tokeny: bbox interior / boundary / far background, z jawnym polem overlap patch–bbox. Budżet 24 może oznaczać 8 na kategorię, jeśli istnieje osiem; przy niedoborze wybrać wszystkie dostępne bez powielania, a braków nie kompensować niejawnie tokenami innej kategorii. Ustalić wagi grup, nie pozwolić by duży bbox zwiększał wagę filmu. Bbox nie jest segmentacją narzędzia. E5 bez GT przy wyborze cech/tokenów: używanie GT do tego etapu jest oracle leakage.

TEST wymaga rejestru wcześniejszych dostępów. Jeśli był używany do strojenia obecnej tezy, nowy seed nie naprawia przecieku danych. Potrzebny nowy odłożony materiał lub jawne ograniczenie do ponownej oceny na znanym zbiorze.

## 3. Estymandy, endpointy i wielokrotność

Wybrać najpóźniej na `Lock`, czy główna historia to E2 geometria odpowiedzi, czy ścieżka E4 kompensacja/ranking. **Nie wolno po nieudanym E2 nazwać E4 potwierdzeniem na tych samych odłożonych wynikach.** Discovery może uzasadnić wybór jednej historii przed testem.

Propozycja dla głównej E2 (konkretna, do zatwierdzenia po pilocie):

1. `P1_localization`: sparowana różnica błędu lokalizacji między zablokowaną interwencją ukierunkowaną a średnią kontroli o dopasowanej normie zastosowanej poprawki. Błąd: RMS czterech krawędzi / przekątna przekształconej GT, z równoległym raportem pikselowym i IoU. Znak `error_sham−error_target` oznacza korzyść. Jeśli pilot nie uzasadni kierunku korzyści, użyć testu dwustronnego zmiany i odpowiedniego języka.
2. `P2_learning`: sparowany kontrast różnic, czyli zmiana P1 między inicjalizacją a uprzednio wybranym późnym etapem. To test organizacji podczas treningu, nie dowód przyczyny treningu; mocniejszy claim wymaga sparowanych kontynuacji E3. Inicjalizacja nie może wejść też do średniej późnego etapu.
3. `P3_prediction` **tylko jeśli artykuł reklamuje diagnostykę jakości**: przyrost AUPRC dla z góry wybranego ryzyka ponad zablokowany najmocniejszy baseline. Próg ryzyka np. IoU<0,5; dane ewaluacyjne muszą zawierać zdarzenia obu klas. ΔAUPRC≥0,03 z PLAN jest propozycją praktycznego celu do wyboru przed testem, nie liczbą potwierdzoną pilotem ani normą konferencji. Nie zastępować po wynikach AUPRC korzystniejszym risk–coverage.

Każdy P jest jedną liczbą per seed×film po zamrożonym uśrednieniu klatek, perturbacji, tokenów i warstw. Wybrać jeden operator, α i zbiór warstw do P1 na discovery; default do dyskusji: α=1, równe wagi 6 bloków i późnych stanów `[44000,55000,65640]`. Jeżeli efekt ma przeciwny znak w blokach, można przed Lock ustalić hipotezę zależną od warstwy, ale wtedy każda potwierdzająca hipoteza zwiększa rodzinę. Wszystkie pozostałe α/ε/bloki/metryki pokazywać jako krzywe i analizy dodatkowe, a nie wybierać rekord testowy.

Naturalne perturbacje: osobne ustalone rodziny nuisance (jasność/kontrast) i task-relevant (translacja z nową GT). Skale ustalić z pilota w zakresach poprawnej interpretacji i niezmienionej widoczności obiektu. P1 może mieć jedną rodzinę główną; druga jest wcześniej określoną kontrolą specyficzności. Jeśli claim wymaga obu, uwzględnić oba endpointy lub zdefiniować test koniunkcji — nie ignorować drugiej połowy twierdzenia.

Przy M endpointach głównych: Holm na poprawnie skalibrowanych testach dwustronnych, FWER 0,05, albo prostsze równoczesne przedziały Bonferroniego o poziomie `1−0,05/M` dla każdego endpointu. Dla M=3 to 98,33% marginalnie. Kierunek jednostronny tylko po rzeczywistej prerejestracji, nie po obejrzeniu confirmation. Te progi są naszym wyborem, nie wymogiem venue. Przy CI bootstrapowym pokrycie sprawdzić symulacyjnie; samo 10 000 resampli nie zapewnia poprawności.

Nie odliczać „brak istotności” jako równoważności. Teza o braku istotnego wpływu wymaga przed Lock marginesu równoważności i odpowiednio precyzyjnego przedziału. Oddzielić błąd numeryczny, statystyczny i praktyczną wielkość efektu. Równoważność samego odczytu przy δ=0 jest kontrolą konstrukcji, a nie hipotezą o danych.

**Nie gubić nieudanych detekcji.** Dla pełnego dekodera ustalić dopasowanie obiektów niezależnie od korzystnego wyniku interwencji. Brak detekcji → odpowiednio brak recall/IoU=0, nie usunięcie klatki. Dla P1 trzeba zdefiniować odczyt krawędzi przy braku detekcji; w obecnej gęstej głowie można zachować fixed-token lub z góry ustalone top1 bez threshold, a pełny decoder oceniać obok. Mieszanie tych dwóch wersji w jednym wyniku jest błędem. Rzadkie switche raportować z mianownikiem i rozkładem po filmach; zero switchy nie daje dowodu odporności.

## 4. Bootstrap skrzyżowany seed×film

Seed nie zawiera własnej niezależnej populacji filmów: te same filmy oceniamy każdym modelem. Poprawny schemat zachowuje wspólny efekt trudnego filmu dla wszystkich seedów i wspólny efekt seeda dla wszystkich filmów.

1. Zbudować sparowany kontrast `d[s,v,f,t,b,p]` ukierunkowana−kontrola na tych samych klatkach, checkpointach, blokach, perturbacjach i losowaniach sham. Dla sham użyć wielu z góry zarejestrowanych losowań; ich liczbę dobrać do MC-error, nie wliczać do N treningów.
2. Zredukować wewnętrzne osie zgodnie z estimandem do `D[s,v]`, z równymi wagami filmów. Podstawowa średnia: `mean_s mean_v D[s,v]`. Standardowe dataset mAP raportować oddzielnie: średnia AP po filmach nie jest tym samym co globalne AP.
3. W każdej replice losować **niezależnie** indeksy seedów i indeksy filmów z powtórzeniami. Zastosować wspólny iloczyn wag `w_seed[s]×w_video[v]` do obu ramion oraz całych trajektorii. Nie losować filmów oddzielnie w każdym seedzie.
4. Jeśli losujemy także klatki, ich indeksy losować w obrębie filmu, a ten sam wybór stosować we wszystkich seedach/ramionach. Dla sąsiednich klatek wykorzystać przedziały czasu lub bloki; dla deterministycznie wybranych 10 klatek podstawowy estimand może być warunkowy względem tej próbki i nie wymagać dodatkowego bootstrapu klatek.
5. Jeżeli znane są pacjenty, losować pacjenta, potem filmy w nim, zachowując skrzyżowanie pacjent×seed. Nie udawać tej hierarchii przy brakujących ID.
6. Dla AUPRC/AP ponownie obliczyć metrykę z ważonych predykcji w replice, zamiast bootstrapować pojedynczy już uśredniony wynik; zachować regułę dopasowania i wagi. Przy nieobecności klasy w replice jawnie raportować odsetek niezdefiniowanych replik; nie wstawiać arbitralnie zera.

Rekomendowany start obliczeniowy CI: 10 000 replik, stałe ziarno statystyczne i osobna kontrola błędu Monte Carlo. To niewielki koszt względem GPU, ale nie naprawia małej liczby klastrów. Przy **trzech confirmation seedach** bootstrap populacji treningów jest bardzo słabo podparty. Podawać surowe wyniki każdego seeda, leave-one-seed-out i warunkowe CI po filmach oraz odrębne CI skrzyżowane. Przy niezadowalającym pokryciu w symulacji nie używać etykiety „potwierdzone dla losowych treningów”; albo zwiększyć wcześniej ustalone N, albo ograniczyć wniosek do trzech modeli.

Dla orientacji: dokładny dwustronny sign-flip na trzech niezależnych średnich seedów ma najmniejszą osiągalną wartość p równą 2/8=0,25. Nie jest to zakaz użycia innych uzasadnionych modeli, ale ostrzeżenie, że 25 filmów i 600 tokenów nie staje się 18 750 niezależnymi seedami.

## 5. Moc i precyzja: czego pilot musi dostarczyć

Nie ma jeszcze variance components dla nowych interwencji sześciu bloków, więc **nie można uczciwie wyliczyć docelowej mocy z trzech starych zdjęć**. Początkowe 24 klatki/12 filmów służą implementacji i wstępnej zmienności po filmach; s47 sam nie pozwala oszacować wariancji seedów. Dwa dodatkowe discovery seedy dają tylko wstępny, niestabilny szacunek tej wariancji.

Model planistyczny sparowanego efektu po uśrednieniu zamrożonego czasu/warstw/perturbacji:

`d_svf = δ + a_s + b_v + c_sv + e_svf`,

`Var(mean d) ≈ σ²_seed/S + σ²_video/V + σ²_seed×video/(S V) + σ²_residual/(S V F)`.

To przybliżenie dla zbilansowanego, odpowiednio zdefiniowanego endpointu; korelacja klatek wymaga zmiany ostatniego członu lub symulacji trajektorii. Jeżeli zwiększamy F, nie zmniejszamy składnika σ²_seed/S.

### Wykonana ilustracja arytmetyczna, nie wynik pilota

Założono sztucznie jednostkową wariancję pojedynczego sparowanego efektu. W scenariuszu film-dominant komponenty wynoszą `(0,04;0,36;0,20;0,40)`, w seed-dominant `(0,36;0,16;0,20;0,28)`. Dla rodzin trzech testów dwustronnych, znanych wariancji i normalnego przybliżenia MDE dla mocy 80% wynosi około `3,236×SE`. Wyliczono:

| Nowe seedy S | Filmy V | Klatki F/film | MDE film-dominant | MDE seed-dominant |
|---:|---:|---:|---:|---:|
| 3 | 25 | 10 | 0,569 | 1,164 |
| 5 | 25 | 10 | 0,505 | 0,916 |
| 8 | 25 | 10 | 0,464 | 0,742 |
| 3 | 25 | 20 | 0,567 | 1,163 |

Jednostką jest SD hipotetycznego pojedynczego efektu, **nie piksel ani IoU**. To optimistyczny oracle ze znanymi wariancjami; estymacja z trzech seedów może być gorsza. Tabela wyjaśnia, dlaczego podwojenie sąsiednich klatek może prawie nic nie dać. Nie zastępuje symulacji na danych discovery.

### Plan symulacji przed Lock, jeszcze niewykonany

- Siatka: confirmation S∈{3,5,8}, V∈{12,25}; F∈{2,10}. V>25 wyłącznie jeśli rzeczywiście pozyskamy nowe niezależne filmy, nie poprzez rozcięcie filmu na fragmenty.
- Warianty δ: 0; ±0,5δ_min; ±δ_min; ±2δ_min. δ_min określić w jednostkach wybranego endpointu na podstawie praktycznego celu i rozdzielczości pomiaru, nie na podstawie najlepszego uzyskanego efektu. Dla P3 rozważyć prerejestrowane 0,03 AUPRC; dla P1 nie wymyślać przeliczenia piksel–IoU bez sprawdzenia wielkości obiektów.
- Generować skrzyżowane efekty seed/film i seed×film ze współczynnikami z discovery; powtórzyć dla dolnej/górnej wiarygodnej wariancji seedów. Dodać skośne/heavy-tail błędy, różne wielkości bbox, heteroskedastyczność i obserwowaną zależność czasową. Dla switchy symulować binomial/logistic mixed effects z rzeczywistą częstością; dla AUPRC symulować wspólnie score i etykietę, nie traktować AUPRC jak zwykłej średniej.
- Co najmniej 2000 niezależnych symulowanych kampanii na kluczowy scenariusz; ta liczba daje MCSE około 0,005 dla oszacowania błędu typu I 0,05 i około 0,009 przy mocy 0,8. Wewnątrz stosować dokładnie planowaną procedurę inferencji i wielokrotności; najpierw tańsza liczba replik bootstrap, potem sprawdzenie granicznych scenariuszy na docelowej liczbie.
- Raport: pokrycie CI, type-I/FWER przy δ=0, moc przy δ_min, medianowa szerokość CI, bias i awarie. Bez wyników tej symulacji nie deklarować mocy 80%.
- Wybrać najmniejszą dostępną konfigurację spełniającą zamrożony cel mocy/precyzji i budżet. Jeśli żaden wariant nie spełnia celu, zmniejszyć zakres claimu lub jawnie zakończyć jako nierozstrzygający, a nie zwiększać liczbę tokenów na potrzeby p-value.
- Decyzję o liczbie confirmation seedów podjąć **przed** obejrzeniem ich wyników. Ewentualna adaptacja ma być wcześniej zaplanowaną, zaślepioną reestymacją wariancji lub poprawną procedurą sekwencyjną; nie dokładać seedów „aż wyjdzie”.

## 6. Czas, negative controls i nieobserwowane przypadki

Główna inferencja zachowuje całe trajektorie w obrębie seed×film. Siatka 12 checkpointów jest nieregularna: do pochodnych używać rzeczywistego Δstep. Na podstawie 12 stanów nie estymować precyzyjnego widma drgań ani chwilowej dynamiki. Gęsty podzbiór z krokiem 100 może służyć dodatkowej analizie czasu, ale nie zwiększa N treningów.

Raportować efekty w z góry ustalonych etapach oraz surowe przebiegi. Detrending spline (np. 4/6/10 węzłów) i przyrosty są kontrolą opisową. Bootstrap bloków tylko w sensownie lokalnie stacjonarnych oknach, z długościami wybranymi w discovery; nie mieszać początkowego warmupu z końcem treningu. Dla krzywych z wieloma punktami zastosować jednoczesne pasmo lub jasno oznaczyć punktowe CI.

E5 wyłącznie przyczynowy czasowo: żadnych średnich ±300, PCA dopasowanego na pełnej historii, finalnych par retroaktywnie wybieranych do wczesnego kroku ani końcowego położenia jako celu. Transformacje/kalibrację i hiperparametry fitować tylko na discovery filmach i do dozwolonego kroku. W CV czasowym purge/embargo co najmniej długość historii cech + horyzont targetu; różne filmy i różne seedy nie rozwiązują automatycznie leakage przyszłości.

| Kontrola | Co odróżnia | Warunek uczciwości |
|---|---|---|
| α=0 oraz δ=0 | błąd implementacji | Identyczny odczyt w tolerancji; nie zaliczać jako dowodu mechanizmu |
| Inicjalizacja | wyuczenie vs geometria losowa | Ten sam instrument, nie inny próg aktywności dobrany po wyniku |
| Normmatched sham w 191D | orientacja vs wielkość poprawki | Zachować Q/LN; dopasować rzeczywistą normę korekty, nie tylko Frobenius operatora |
| Gradient/JVP/normy i parallel–transverse | wartość ponad silnym baseline | Ten sam dostęp do informacji i budżet; pełny JVP nie zawiera mniej informacji niż S/A |
| Permutacja aktywacji | wspólna struktura vs marginesy | Przetasowanie w zadanych klasach przestrzennych/checkpointach, wiele losowań; nie udawać interwencji fizycznej |
| Random labels | organizacja zadaniowa vs optymalizacja | Osobny mały budżet, sprawdzić stopień dopasowania i normy; niedouczony model nie jest doskonałą kontrolą każdego claimu |
| Nuisance vs translacja | tłumienie szumu vs tłumienie użytecznej odpowiedzi | GT aktualizowana, brak obcięć obiektu w podstawowym teście |
| A/B/AB i rescue/reverse patch | addytywność, kompensacja, ranking | Wspólna populacja, fixed-token osobno od switcha argmax, kontrola off-manifold |

Brak normmatched kandydata: flagować i raportować coverage; główny wynik warunkowy na uprzednio określonej populacji dopasowalnej. Nie stosować bez etykiety starego fallback „najbliższy, choć poza tolerancją”. Ustalić próg małych mianowników q z numeryki na discovery, pokazać wrażliwość; zero nie jest tożsame z brakującym pomiarem. Nie usuwać gorszych checkpointów, NaN ani OOM bez rejestru przyczyny. Brak pełnego instrumentu w seedzie może wymagać uzupełnienia całego sharda, nie selektywnej utraty ciężkich przypadków.

## 7. Mapowanie istniejących skryptów — co można ponownie wykorzystać

| Istniejący plik | Rzeczywista funkcja / użycie w protokole | Brakujący zakres |
|---|---|---|
| `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/{model.py,trening35000.py,dane.py}` | Referencyjny model, dane, trening s47 | Seed w trening35000 stały; nowy rejestr seedów, dokładny restart batch/RNG i rozszerzona telemetria wymagają adaptacji |
| `experiments/maly_vit_szklo_pokretla_20261010/kod/trening_pokretla.py` | Parametryzacja `--ziarno`, lr i kontroli | Nie jest gotową implementacją nowego protokołu E0–E6; historyczna retencja danych musi być zmieniona przed użyciem |
| `.../kod/obserwator_rp.py` | Rzut RP i MLP sześciu warstw | Nie pełne J, attention, Q191 ani zamrożony downstream interwencji |
| `experiments/optyka_hipoteza_20261015/ekstrakcja.py` | Float32 last-MLP residual/aktywizacje i walidacja sum | Trzy obrazy i ostatni blok, nie atlas niezależnych filmów |
| `experiments/wspoldzialanie_neuronow_20261015/pomiar.py` | Dobór par, Jacobian głowy, pojedyncze/wspólne ablacje | Last-block skrót nie działa we wcześniejszych blokach; trzeba pełny suffix, nowe dane/manifesty |
| `.../grupy.py` | Grupy8/16/32, normy i porównania z kontrolami | Dobór lokalnie na testowanym stanie bywa opisem konstrukcyjnym; confirmation musi zamrozić algorytm i dostęp do informacji |
| `.../kontrola_przelaczenia.py`, `.../weryfikacja.py` | Niezależny pełny forward i kontrole znanego switcha | Rozszerzenie na pełną populację; nie liczyć pojedynczej ilustracji jako częstości |
| `.../przeciwne_aktywacje*.py` | Współaktywacja, energie par i kontrole czasu | Głównie s47/ostatniMLP, badanie opisowe; potrzebne nowe filmy/seedy i jawne permutacje |
| `.../ruch_a_jakosc.py`, `.../ruch_a_jakosc_etykiety.py` | Dopasowanie czasowe, spline, bbox/tło, wskazane leakage | Retrospektywne 3 obrazy; nie model predykcyjny E5 |
| `eden/queue.py`, `eden/runner.py`, `eden/run_node.slurm` | Istniejący wzorzec kolejki i workerów | Kontrakty dotyczą innej kampanii; brak dowodu kompatybilności z nowymi operatorami. Nie uruchamiać z obecnym manifestem bez implementacji |
| `experiments/maly_vit_od_zera_35000_21epok_20261009/bieg_eden.slurm` | Wzorzec środowiska, lokalnych danych, cleanup | Hardcoded ścieżki/tryby; nie gotowy launcher confirmation |
| `experiments/deit_cifar10_20261013/kod/{model_deit_cifar.py,deit_cifar.slurm}` | Drugi model dla klasyfikacji | Nie dowód generalizacji bbox; wymaga osobnego adaptera pomiaru |

Nie istnieje zweryfikowany, pojedynczy launcher wszystkich zaproponowanych testów. Powyższa tabela rozróżnia zasób od gotowej implementacji; nie podaję fikcyjnych komend, które miałyby już działać.

## 8. Manifest maszynowy — kontrakt do implementacji

Docelowo `campaign.yaml` + JSONL rekordów klatek, seedów, checkpointów, shardów i surowych wyników. Poniższy zapis jest **schematem propozycji**, a `null` oznacza decyzję niezamrożoną, nie wartość domyślną. Walidator ma odmawiać confirmation, gdy pola wymagane pozostają null.

```yaml
schema_version: 1
campaign_id: vit_mechanism_20261016
status: draft
authoritative_plan: experiments/plan_publikacji_neurips_20261016/PLAN.md
lock:
  timestamp: null
  config_sha256: null
  code_commit: null
  source_snapshot_sha256: null
  analysis_sha256: null
splits:
  source_manifest_sha256: null
  train_operation_groups: 117
  val_operation_groups: 25
  test_operation_groups: 25
  patient_disjointness_verified: false
  patient_mapping_sha256: null
  test_access_audit_passed: null
  access_ledger: null
roles:
  exploratory_existing: [47]
  discovery_proposed: [1047, 2047]
  confirmation_proposed: [3047, 4047, 5047]
sampling:
  pilot_frames: 24
  pilot_val_films: 12
  atlas_frames_per_film: 10
  atlas_exact_mlp_tokens_per_frame: 24
  checkpoint_steps: [0,300,1000,3000,6000,10000,16000,24000,33000,44000,55000,65640]
  checkpoint_fallback_max_steps: 50
  frame_manifest_sha256: null
  token_manifest_sha256: null
  rng_frame_selection: null
measurement:
  residual_dim: 192
  quotient_dim: 191
  quotient_basis_sha256: null
  blocks: [0,1,2,3,4,5]
  branches: [attention, mlp]
  layernorm_eps_from_runtime: true
  forward_mode: eval
  intervention_downstream: full_suffix
  derivative_dtype: null
  numerical_tolerances: null
interventions:
  primary_operator: null
  primary_blocks: null
  primary_checkpoint_weights: null
  primary_alpha: null
  perturbation_manifest_sha256: null
  sham_method: null
  sham_draws: null
  matching_tolerance: null
endpoints:
  primary_ids: null
  primary_definitions_sha256: null
  practical_effect_thresholds: null
  failure_and_missingness_policy_sha256: null
statistics:
  top_units: [training_seed, operation_group]
  crossed_resampling: true
  paired_arms_and_trajectories: true
  target_seed_population: null
  bootstrap_draws: 10000
  familywise_alpha: 0.05
  multiplicity_method: null
  power_simulation_report_sha256: null
  planned_confirmation_seed_count: 3
  optional_stopping_rule: none
execution:
  eden_live_preflight_sha256: null
  benchmark_report_sha256: null
  quota_bytes_budget: null
  max_concurrent_gpu_shards: null
  retention_policy_sha256: null
```

Każdy `frame_record`: dataset/version/split, operation_group, patient_id lub null, film_id, frame/time, SHA256 źródłowych bajtów, GT i jego pochodzenie, przekształcenie rozmiaru, wagi próbkowania. Każdy `checkpoint_record`: seed/rola, steps/epoch/exposures, model/optimizer/buffers/RNG/sampler hashes, pełny versus tylko inference state, historia wznowień.

Klucz wyniku: campaign+seed+checkpoint+film+frame+block+branch+token/group+perturbation+scale+operator+sham_draw. Pola: baseline/target/control output, fixed-token i final decoder oddzielnie, GT reference, metric definitions hash, applied correction norm, matching error, numerical residual, status/reason. Zapisać baseline także dla nieudanego wariantu. Rejestr shardów: klucz, inputs hashes, version, expected_count, row_count, status, attempt, checkpoint restart, byte_count, SHA256, wall/GPU time, peak RAM/VRAM, hostname/GPU/software i failure reason; bez sekretów.

## 9. Eden DAG i benchmark kosztu

Graf logiczny, **nie gotowe komendy ani wysłane jobs**:

```text
A inventory + split audit + read-only preflight
  → B validate manifests/instrument/QA E0 + cost benchmark
  → C discovery pilot E1/E2 on s47
  → D discovery atlas + 2 discovery trainings + E3/E4 optional
  → E power/calibration, choose hypothesis, lock all manifests
  → F 3 confirmation trainings (or N fixed by E before outcomes)
  → G exact checkpoint validation + instrument shards
  → H confirmation interventions + negative-control shards
  → I completeness/reduction/per-seed tables + locked statistics
  → J one-time TEST report + release audit
  → K optional second architecture/detection dataset, separate lock
```

TEST w G/H dostępny tylko po Lock i audycie uprawnień protokołu; J to raport, nie pierwsze obliczenie na teście. Pipeline ma zapisać wszystkie dostępy od chwili przygotowania G. Zadania zależne uruchamiać dopiero po pomyślnym zakończeniu i walidacji artefaktów poprzednika, nie tylko kodzie exit0. E3 kontynuacje wymagają pełnych stanów optimizer/RNG/data queue; inference-only checkpoint nie spełnia dependency.

Sharding podstawowy: jeden seed×checkpoint×zakres klatek×wersja instrumentu; bloki przetwarzać w tym samym shardzie, gdy pozwala pamięć i można współdzielić forward. Przy pełnych Jacobianach rozdzielić bloczki, zachowując identyczne baseline i input hashes. Nie serializować całej macierzy attention ani wszystkich J domyślnie. Przechowywać agregaty i niewielki pakiet operatorów fp32 do odtworzenia; wybór pakietu przed wynikami.

Atomowy zapis do nowego pliku tymczasowego, checksum i liczba wierszy, dopiero potem rename i `complete`. Restart pomija wyłącznie shardy ze zgodnym kompletnym manifestem. Nie uznaje istnienia pustego pliku za wynik. Idempotentne klucze zapobiegają podwójnemu ważeniu po retry. Przerwanie treningu zapisuje model, AdamW, BN buffers, RNG CPU/CUDA/NumPy, aktualną permutację i pozycję dataloadera; test restartu porównuje następną paczkę i krok. Przed partial-results reduction kontrolować brakujące shardy; nie filtrować ich cicho.

E0 mierzy oddzielnie: odczyt wag/danych, baseline forward, pełny MLP J 191D/192D, attention JVP/VJP, wartości singularne, jedno full-suffix rerun, rescue i serializację. Logować wall time i GPU time, warmup, peak allocated/reserved VRAM, MaxRSS, bajty oraz rodzaj GPU. Benchmark co najmniej na wczesnym i późnym bloku oraz reprezentatywnych obrazach, ponieważ koszt full suffix zależy od bloku.

Koszt planować z empirycznych jednostek:

`T_total = T_train + N_ckpt×T_load + Σ(N_operator×T_operator + N_suffix×T_suffix + N_image×T_forward) + T_IO + T_reduce`.

Podawać zakres według czasów medianowych i wolniejszych prób, plus narzut walidacji/nieudanych prób, nie optymistyczne mnożenie czasu jednego forwarda. Historyczne 728+1374 s s47 to wyłącznie referencja treningu, nie koszt Jacobianów. Dotychczasowe 806 wag ~9,45GB na seed. Atlas pełnych J float32 według PLAN ~63,7GB/seed, 13 residuali ~18,0GB/seed; zalecane strumieniowanie. Bieżąca kwota/GPU/RAM nadal nieznane — nie można na ich podstawie zadeklarować równoległości ani deadline.

Domyślnie jeden mały GPU worker na shard; zwiększać concurrency dopiero po benchmarku i limicie wspólnej kwoty. Cleanup wyłącznie własnego, zweryfikowanego scratch; historyczne dane zachować. Retencja checkpointów tego badania jest częścią protokołu, więc stara zasada „zostaw tylko best” tu nie wystarcza.

## 10. Uogólnienie — co jest zasobem, a co propozycją

Istnieją modele/skrypty MałyViT/CIFAR, DeiT-Tiny/CIFAR i Imagenette. Pomogą sprawdzić ogólność geometrycznej części, lecz klasyfikacja nie potwierdza wniosku o lokalizacji. CNN może kontrolować potrzebę attention, nie zastępuje drugiego detektora.

Zgodnie z PLAN kandydatem na drugi detection dataset jest **COCO**, jednak w tym audycie nie potwierdzono lokalnej kompletnej kopii ani gotowego adaptera. COCO definiuje benchmark wieloobiektowy i własną ewaluację; sprawdzić oficjalne dane, licencje oraz wymagania przed planem pobrania. [COCO Detection 2017](https://cocodataset.org/dataset/detection-2017.htm), [oficjalna ewaluacja](https://cocodataset.org/dataset/detection-eval.htm), [publikacja zbioru](https://arxiv.org/abs/1405.0312).

Dwie dopuszczalne ścieżki, wybór przed wynikami: (a) jawnie nazwane single-object podzadanie z regułą wyboru obiektu niezależną od modelu i pełnym opisem ograniczenia; nie reklamować jako pełny COCO; (b) pełny detektor transformerowy ze standardowym dopasowaniem wielu obiektów i adapterem instrumentu. Dla current dense top1 nie wolno po prostu wybrać GT o największym IoU po perturbacji i nazwać tego oceną całego obrazu.

Druga architektura: parametryzacja szerokości/głębokości istnieje w `MalyViT`, ale nowy wariant wymaga treningu i ponownej kalibracji instrumentu/Q; wymiar ilorazu wynosi D−1, nie zawsze 191. Zmiana architektury i zbioru naraz miesza dwa czynniki. Minimalny most: A1/D1, A2/D1, A1/D2 po przejściu głównej bramki; pełne 2×2 dopiero jeśli budżet i pytanie to uzasadniają. Dla pretrained modeli audyt danych pretrainingu i możliwej znajomości testu jest osobnym źródłem zależności.

## 11. Warunki gotowości do confirmation

- Instrument E0 poprawny i pełny suffix, Q191 zgodny z runtime LN, tolerancje i formaty określone.
- Plan ról seedów i filmów, dostęp TEST i patient-unit opisane; wszystkie hashe manifestów zamrożone.
- Jedna główna historia, kierunek i praktyczny próg efektu określone; rodzina endpointów i obsługa braków zatwierdzone.
- Symulacyjna kalibracja inferencji/mocy albo jawny zakres wnioskowania warunkowego na małej liczbie seedów.
- Brak przyszłości i GT w deployable E5, silne baseline'y przy porównywalnym koszcie.
- Bieżący benchmark kosztu i kwota zweryfikowane, restart i completeness gate działają.
- Log nieudanych prób, dane per-seed/per-film oraz konfiguracja umożliwiają odtworzenie liczb bez GUI.

Do tego momentu wszystko jest discovery, także bardzo wyraźny wynik na s47. Spełnienie tych warunków podnosi wiarygodność dowodów; nie jest gwarancją akceptacji konferencyjnej.
