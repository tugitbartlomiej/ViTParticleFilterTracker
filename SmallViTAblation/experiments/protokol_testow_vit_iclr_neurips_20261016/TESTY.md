# Katalog 30 testów ViT: T00–T29

Data: 16 października 2026. **Status wszystkich pozycji: zaplanowane, niewykonane w tym zadaniu.** Istniejące pomiary są materiałem discovery. Katalog jest specyfikacją naukową, nie zestawem gotowych skryptów. Czytać łącznie z [PROTOKOL.md](PROTOKOL.md), [EDEN_I_STATYSTYKA.md](EDEN_I_STATYSTYKA.md) i [literaturą](LITERATURA.md).

Każdy test ma hipotezę lub warunek poprawności, procedurę, kontrole, metrykę oraz wynik obalający twierdzenie albo ograniczający interpretację. Brak efektu przy małej precyzji oznacza nierozstrzygnięcie. `depends_on` określa dostępność instrumentu; niezależnie każda faza confirmation wymaga wcześniejszego zamrożenia T27. „Obowiązkowy” oznacza rdzeń wybranego przez nas protokołu, nie oficjalny wymóg konferencji.

| ID | Test | Kiedy potrzebny |
|---|---|---|
| [T00](#t00) | Pochodzenie danych i niezależność oceny | obowiązkowy |
| [T01](#t01) | Tożsamość forward i dekodera | obowiązkowy |
| [T02](#t02) | Pochodne MLP i pełnego attention | obowiązkowy |
| [T03](#t03) | Pola o znanym zachowaniu i kontrprzykłady | obowiązkowy |
| [T04](#t04) | LayerNorm, współrzędne i projekcja | obowiązkowy |
| [T05](#t05) | Zakres lokalnego przybliżenia | obowiązkowy |
| [T06](#t06) | Atlas geometrii w sześciu blokach | obowiązkowy |
| [T07](#t07) | Attention jako sprzężenie całego obrazu | obowiązkowy |
| [T08](#t08) | Kierunki istotne dla zadania i zakłócenia | obowiązkowy |
| [T09](#t09) | Przyczynowa zmiana geometrii pola | obowiązkowy |
| [T10](#t10) | Semantyka regionów i realistyczność zmian | obowiązkowy |
| [T11](#t11) | Szlak wpływu przez kolejne bloki | obowiązkowy |
| [T12](#t12) | Podpisane wkłady i współaktywacja neuronów | dla_tezy_współdziałanie |
| [T13](#t13) | Ablacje pojedyncze, parowe i grupowe | dla_tezy_współdziałanie |
| [T14](#t14) | Czy następny blok koryguje poprzedni | dla_tezy_korekta |
| [T15](#t15) | Przywracanie poprawnego wyniku przez podmianę aktywacji | obowiązkowy |
| [T16](#t16) | Rozdzielenie regresji ramki, maksimów i rankingu | obowiązkowy |
| [T17](#t17) | Transfer zamrożonych selektorów | dla_tezy_współdziałanie |
| [T18](#t18) | Skąd zmienia się pole między checkpointami | dla_tezy_uczenie |
| [T19](#t19) | Rzeczywisty krok optymalizatora i ruch tokenu | dla_tezy_uczenie |
| [T20](#t20) | Czy efekt jest wyuczony i zależny od zadania | obowiązkowy |
| [T21](#t21) | Niezależne powtórzenia treningu | obowiązkowy |
| [T22](#t22) | Druga architektura | dla_tezy_ogólne_ViT |
| [T23](#t23) | Drugi zbiór z lokalizacją obiektu | dla_tezy_ogólna_detekcja |
| [T24](#t24) | Nowe typy zaburzeń i przypadki trudne | dla_tezy_odporność |
| [T25](#t25) | Prognozowanie jakości bez dostępu do GT | opcjonalny_użytkowy |
| [T26](#t26) | Silne porównania i granica nowości | obowiązkowy |
| [T27](#t27) | Zamrożenie analizy i wnioskowanie statystyczne | obowiązkowy |
| [T28](#t28) | Koszt, precyzja i deterministyczność | obowiązkowy |
| [T29](#t29) | Odtwarzalność i niezależny przegląd wyniku | obowiązkowy |

Wersja 1.2 po drugiej recenzji: minimalny rdzeń confirmation dobieramy po discovery; pełna siatka atlasu jest zależna od claimu. Wszystkie statusy nadal oznaczają plan, nie wynik.

<a id="t00"></a>

## T00. Pochodzenie danych i niezależność oceny

**Zakres:** obowiązkowy. **Zależności:** brak. **Literatura:** [R14](LITERATURA.md#r14), [R15](LITERATURA.md#r15).

- **Hipoteza / warunek:** Ocena dotyczy nowych filmów/zabiegów, a dane i etykiety są zgodne z rzeczywistym wejściem.
- **Procedura:** Zbudować manifest obrazu, filmu, zabiegu/pacjenta jeśli dostępny, czasu, bbox, splitu i hash. Zweryfikować 117/25/25 filmów, historię użycia TEST, duplikaty exact/perceptual i klatki sąsiednie. Ustalić rozdział discovery/confirmation przed odczytem metryk.
- **Kontrole:** Kontrola nazw plików nie wystarcza: porównać zawartość i źródłowe operacje. Trzy dotychczasowe ilustracje zostają discovery. Brak identyfikatorów pacjenta oznaczyć unknown.
- **Metryki:** Liczba niezależnych filmów/zabiegów; duplikaty i konflikty etykiet; pokrycie metadanych.
- **Jednostka:** film/zabieg/pacjent, zależnie od potwierdzonych metadanych.
- **Falsyfikacja / ograniczenie:** Przeciek lub nieudokumentowane wcześniejsze strojenie TEST uniemożliwia nazywanie go ślepym testem; wymaga nowego splitu lub ograniczenia twierdzenia.
- **Zakres startowy i koszt:** CPU; wszystkie manifesty; ręczny przegląd wylosowanej próbki i konfliktów.
- **Planowany wynik:** `dataset_audit.json`.

<a id="t01"></a>

## T01. Tożsamość forward i dekodera

**Zakres:** obowiązkowy. **Zależności:** T00. **Literatura:** [R13](LITERATURA.md#r13), [R22](LITERATURA.md#r22).

- **Hipoteza / warunek:** Mikroskop mierzy rzeczywisty forward konkretnego checkpointu.
- **Procedura:** Rejestrować wejście residualu i 12 wyjść attention/MLP, wejście LN, 768 aktywacji GELU i wyjście w 192D. Odtworzyć F_M z podpisanych wkładów wraz z biasem. Porównać pięć surowych wyjść, sigmoid, maskę maksimów 3×3, top-k i ramki exp(ltrb) z modelem źródłowym.
- **Kontrole:** Hooki off/on, powtórzony eval, batch1/batchN, CPU/GPU na podpróbie; nie włączać treningowego BN/dropout.
- **Metryki:** Maxabs i względny błąd stanów, raw outputs, ramek; identyczność indeksów z osobną obsługą remisów.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Błąd większy od tolerancji lub badanego efektu zatrzymuje analizę; różnic dekodera nie wolno przypisać polu.
- **Zakres startowy i koszt:** E0: 3 checkpointy × 4 obrazy × 6 bloków; mały benchmark GPU.
- **Planowany wynik:** `forward_audit.json`.

<a id="t02"></a>

## T02. Pochodne MLP i pełnego attention

**Zakres:** obowiązkowy. **Zależności:** T01. **Literatura:** [R01](LITERATURA.md#r01), [R02](LITERATURA.md#r02), [R22](LITERATURA.md#r22).

- **Hipoteza / warunek:** Analityczny Jacobian i pochodne liczone bez materializacji macierzy zgadzają się z funkcją.
- **Procedura:** Porównać J_MLP z autodiff float64 i centralnymi różnicami w szeregu kroków. Attention: pełne Q/K/V/LN/softmax/proj; sprawdzić JVP, VJP i iloczyn 12 pochodnych względem end-to-end JVP.
- **Kontrole:** Tożsamość adjoint <v,Ju>=<J^Tv,u>; jawny vs fused attention; osobno frozenKV i pełna pochodna; fallback ma odtwarzać forward.
- **Metryki:** Błędy Jv/finite differences, adjoint, chain rule; zakres stabilnych epsilon.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Zgodność tylko query-only nie waliduje pełnej uwagi; brak plateau błędu oznacza niewiarygodny instrument.
- **Zakres startowy i koszt:** E0; J192×192 dla 12 tokenów na obraz; attention przez 16 sond, kontrola zbieżności na 8/16/32.
- **Planowany wynik:** `derivative_audit.json`.

<a id="t03"></a>

## T03. Pola o znanym zachowaniu i kontrprzykłady

**Zakres:** obowiązkowy. **Zależności:** T02. **Literatura:** [R01](LITERATURA.md#r01), [R02](LITERATURA.md#r02), [R21](LITERATURA.md#r21).

- **Hipoteza / warunek:** Mierniki rozpoznają zdefiniowane obiekty, bez nadinterpretacji wiru i siodła.
- **Procedura:** Sprawdzić gradient funkcji kwadratowej, obrót, ścinanie, macierz nienormalną, mieszany układ liniowy i residual o kroku 1. Porównać J, S, A, I+J oraz dokładne przesunięcia.
- **Kontrole:** Shear z A≠0 bez orbit; F(x0)≠0 mimo mieszanego S; ujemne S przy kroku dyskretnym bez kontrakcji.
- **Metryki:** Zgodność dokładnych wartości, singular values i przyrostu normy; błędy nazw klasyfikacji.
- **Jednostka:** znana funkcja syntetyczna; brak wnioskowania populacyjnego.
- **Falsyfikacja / ograniczenie:** Jeżeli implementacja utożsamia A≠0 z orbitą albo S<0 z kontrakcją każdego kroku, trzeba poprawić interpretację.
- **Zakres startowy i koszt:** CPU; syntetyczne macierze 2D/192D; test poprawności metody, nie dowód uczenia.
- **Planowany wynik:** `known_fields.json`.

<a id="t04"></a>

## T04. LayerNorm, współrzędne i projekcja

**Zakres:** obowiązkowy. **Zależności:** T02, T03. **Literatura:** [R02](LITERATURA.md#r02), [R12](LITERATURA.md#r12), [R21](LITERATURA.md#r21), [R22](LITERATURA.md#r22).

- **Hipoteza / warunek:** Wniosek nie zależy od arbitralnej projekcji ani dokładnej symetrii LN.
- **Procedura:** Sprawdzić F(x+c1)=F(x) oraz J1=0; zbudować Q i bazę U191. Raportować surowy J oraz U^TJU. Sprawdzić ortogonalną zmianę współrzędnych mierzonej funkcji i PCA 3D zamrożoną na discovery; odrębnie kontrolować centrowanie po tokenach.
- **Kontrole:** Permutacje i obroty współrzędnych z poprawnym przepisaniem całej funkcji, nie dowolna rotacja samych wag LN. Przy eps>0 kierunek radialny nie jest dokładnie zerowy. Metryka główna: natywna euklidesowa residualu na ilorazie LN. Nie deklarować niezależności S/A od dowolnej metryki lub nieortogonalnej parametryzacji.
- **Metryki:** Stabilność widm i norm pod ortogonalną zmianą; część efektu zachowana w 3D; błędy symetrii LN.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Efekt obecny wyłącznie po projekcji lub znikający po poprawnej kontroli LN nie wspiera pełnowymiarowej tezy.
- **Zakres startowy i koszt:** E0+pilot; koszt mały względem Jacobianów.
- **Planowany wynik:** `coordinates_audit.json`.

<a id="t05"></a>

## T05. Zakres lokalnego przybliżenia

**Zakres:** obowiązkowy. **Zależności:** T02, T04. **Literatura:** [R01](LITERATURA.md#r01), [R02](LITERATURA.md#r02).

- **Hipoteza / warunek:** Lokalny operator trafnie przewiduje odpowiedź w zadeklarowanej skali.
- **Procedura:** Zbadać względne promienie 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 normy Qx; kierunki losowe, własne S, singularne I+J oraz zmiany wywołane edycją obrazu. Porównać F i downstream z ich JVP.
- **Kontrole:** Oba znaki perturbacji; inicjalizacja; te same promienie i normy. Przypadki z Qx bliskim zeru analizować oddzielnie, ze skalą odniesienia ustaloną przed testem.
- **Metryki:** Błąd Taylora vs promień, kosinus i względny błąd predykcji wyjść; brak gwarancji monotoniczności.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeśli realistyczne zaburzenia wykraczają poza zwalidowany zakres, ograniczyć claim lokalny lub użyć rzeczywistych finite differences.
- **Zakres startowy i koszt:** Pilot: 7 checkpointów × 24 obrazy; 12 tokenów × 6 MLP. Liczbę kierunków ustalić po benchmarku E0.
- **Planowany wynik:** `locality_curves.parquet`.

<a id="t06"></a>

## T06. Atlas geometrii w sześciu blokach

**Zakres:** obowiązkowy. **Zależności:** T04, T05. **Literatura:** [R01](LITERATURA.md#r01), [R03](LITERATURA.md#r03).

- **Hipoteza / warunek:** Struktura pól ma powtarzalny przebieg po głębokości i treningu.
- **Procedura:** Policzyć F, J, S, A, udział A, wartości singularne I+J i effective rank w 192D oraz ilorazie LN. Tanie agregaty dla wszystkich 600 tokenów, dokładne J na ustalonej próbce. Mapować wynik względem checkpointu, bloku i regionu.
- **Kontrole:** Inicjalizacja, wcześniej wybrane tokeny, losowe projekcje, stała metryka; kontrola norm i etapu treningu. Nie traktować setek checkpointów jako niezależnych replik.
- **Metryki:** Rozkłady i efekty seed×film; spektra, transient gain, nie tylko średnia.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak powtarzalności albo pełne wyjaśnienie normą/czasem ogranicza wniosek do opisu tej trajektorii.
- **Zakres startowy i koszt:** Pilot: 12 096 Jacobianów; atlas: 12 checkpointów × 250 klatek × 6 bloków × 24 tokeny na seed. Agregować strumieniowo.
- **Planowany wynik:** `geometry_atlas.parquet`.

<a id="t07"></a>

## T07. Attention jako sprzężenie całego obrazu

**Zakres:** obowiązkowy. **Zależności:** T02, T05. **Literatura:** [R02](LITERATURA.md#r02), [R03](LITERATURA.md#r03).

- **Hipoteza / warunek:** Za odpowiedź odpowiadają określone sprzężenia między tokenami, a nie wyłącznie lokalny MLP.
- **Procedura:** Porównać pełny attention JVP, bloki J_ii/J_ij i wariant z zamrożonym K/V; sondy pojedynczego tokenu i grup obiektu/tła. Obliczyć osobno attention, MLP i rzeczywistą kompozycję na tej samej ścieżce.
- **Kontrole:** Dopasowane normą perturbacje, background/foreground matched po odległości, dropout off; uwaga wagowa jako słaby baseline, nie siła przyciągania.
- **Metryki:** Energia odpowiedzi tokenu źródłowego i pozostałych; wpływ na logit/krawędzie przez pełny downstream; błąd uproszczenia query-only.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeżeli frozenKV zmienia obraz efektu, nie wolno przedstawiać go jako pomiaru pełnego attention.
- **Zakres startowy i koszt:** 16 sond pełnego stanu; kontrola zbieżności 8/16/32; wybrane pary zamiast macierzy 115200×115200.
- **Planowany wynik:** `attention_coupling.parquet`.

<a id="t08"></a>

## T08. Kierunki istotne dla zadania i zakłócenia

**Zakres:** obowiązkowy. **Zależności:** T05, T07, T10. **Literatura:** [R01](LITERATURA.md#r01), [R03](LITERATURA.md#r03), [R12](LITERATURA.md#r12).

- **Hipoteza / warunek:** Wyuczona orientacja przekazuje pożądaną zmianę lokalizacji i ogranicza niepożądane zmiany od zakłóceń.
- **Procedura:** Utworzyć pary obrazów z kontrolowaną translacją/skalą i przekształconym GT oraz jasnością/kontrastem z niezmienionym GT. Przenieść rzeczywiste δX przez pozostałe bloki. Zmierzyć odstępstwo od oczekiwanej ekwiwariancji ramki, nie tylko normę δ. Dla stałego kandydata geometryczny GT musi nadal zawierać jego kotwicę; alternatywą jest odrębny odczyt z deterministycznie przekształconą kotwicą.
- **Kontrole:** Dopasowanie norm zmian latentu, stały kandydat i pełny dekoder oceniane oddzielnie, inicjalizacja, orientacja losowa, pełny JVP. Kierunki wybrać bez TEST. Zakres i regułę osiągalności GT ustalić przed interwencjami; przypadki poza wsparciem zachować w ocenie pełnego dekodera.
- **Metryki:** Normalizowany błąd krawędzi, IoU, alignment predicted/observed Δbbox; gain w kierunkach nuisance/task.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Podobny efekt przy inicjalizacji lub po losowaniu orientacji nie wspiera wyuczonej roli orientacji. Brak transferu ogranicza wniosek do badanej rodziny. Błędu wynikającego z nieosiągalności GT przez stałą kotwicę nie przypisywać polu.
- **Zakres startowy i koszt:** Pilot: 24 obrazy; 4 rodziny × 3 poziomy oraz clean, następnie odłożone filmy.
- **Planowany wynik:** `task_alignment.parquet`.

<a id="t09"></a>

## T09. Przyczynowa zmiana geometrii pola

**Zakres:** obowiązkowy. **Zależności:** T05, T08, T26. **Literatura:** [R01](LITERATURA.md#r01), [R06](LITERATURA.md#r06), [R07](LITERATURA.md#r07).

- **Hipoteza / warunek:** Wybrana składowa i jej orientacja mają specyficzny wpływ na lokalizację ponad siłę zaburzenia.
- **Procedura:** Na czystym x0 zamrozić Jbar/Sbar/Abar; w zmienionym obrazie stosować F'(x)=F(x)−α U B U^T(x−x0), α∈{0,0.5,1}. B=Abar, Sbar lub jej dodatnia/ujemna część, wybrane na discovery. Jedno MLP na raz, pełny downstream; następnie ograniczona replikacja na grupie tokenów. Główną rodzinę nuisance zamrozić; geometria ma osobny endpoint z kontrolą kotwicy. Podstawowy token interwencji j=i0, ustalony z clean kandydata danego seed×checkpoint×obrazu/obiektu. Prognozę efektu zapisać przed odpowiedzią interwencyjną; kandydat jest stały tylko w obrębie tych ramion.
- **Kontrole:** Primary sham dopasowany normą faktycznej poprawki Bδ. Oddzielnie sham OBO^T zachowujący strukturę i widmo; nie mieszać ich wyników. Zapisane RNG, zerowe δ, pełny JVP i kontrola kierunku. Normowanie per δ jest kontrolą dawki, nie jednym liniowym operatorem dla całego otoczenia. W ostatnim MLP j≠i0 jest kontrolą strukturalnego zera współrzędnych i0, choć ranking może się zmienić. Porównać Δy z Cq pełnego suffixu; równa norma q nie wyrównuje czułości Cq.
- **Metryki:** P1: sparowana różnica normalizowanego błędu krawędzi target−średnia norm-matched sham, dla stałego clean kandydata; dawka i CI. IoU, maska maksimów i ranking końcowego dekodera równolegle.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak różnicy względem dopasowanego sham lub efekt wyłącznie dla skrajnych stanów poza zwalidowanym zakresem nie wspiera lokalnego mechanizmu. Znak, próg, operator i warstwę zamrozić przed confirmation. Efekt na clean jest zerowy konstrukcyjnie. Sam P1 pokazuje kierunkową podatność. Claim mechanistyczny wymaga zamrożonych przewidywań i zgodnych T11/T15 dla tego samego procesu. P2 ma dodatkową kontrolę wspólnej kotwicy GT przez porównywane modele/stany.
- **Zakres startowy i koszt:** Pilot: 24 obrazy × 3 checkpointy × bloki 1/3/6, jeden i0: 216 źródłowych J. Nie obejmuje kosztu suffixu, sham, walidacji i dodatkowych kotwic. Start 8 orientacji sham, kontrola zbieżności 8/16/32; potem zamrożony zakres i odłożona warstwa.
- **Planowany wynik:** `field_interventions.parquet`.

<a id="t10"></a>

## T10. Semantyka regionów i realistyczność zmian

**Zakres:** obowiązkowy. **Zależności:** T00, T01. **Literatura:** [R06](LITERATURA.md#r06), [R10](LITERATURA.md#r10).

- **Hipoteza / warunek:** Różnice obiekt/tło dotyczą treści obrazu, a nie wyłącznie pozycji lub artefaktu edycji.
- **Procedura:** Regiony określić przez overlap patch–bbox, granicę i dalekie tło; uwzględnić konwolucyjny stem z zachodzącymi polami recepcyjnymi. Lokalne edycje tylko z maską lub ręcznym sprawdzeniem; zamiana tła/narzędzia z naturalnymi dawcami jako osobny test.
- **Kontrole:** Dopasowanie powierzchni, położenia i jasności, sham compositing, stałe GT tylko wtedy, gdy pozostaje poprawne. Kontrola niedokładnych masek i osiągalności GT przez kotwicę w testach translacji.
- **Metryki:** Poprawność adnotacji, udział obiektu, efekt treści wobec pozycji, jakość edycji; odległości aktywacji jako przybliżenie nietypowości stanu.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Sam bbox nie dowodzi semantyki neuronu; nierealistyczne edycje lub maski naruszające obiekt nie uzasadniają tezy o ignorowaniu nieistotnego zakłócenia.
- **Zakres startowy i koszt:** CPU i ręczny audyt oraz GPU; obowiązkowo 24 obrazy pilota. Podpróbka masek wybierana przed wynikami.
- **Planowany wynik:** `perturbation_audit.json`.

<a id="t11"></a>

## T11. Szlak wpływu przez kolejne bloki

**Zakres:** obowiązkowy. **Zależności:** T07, T09. **Literatura:** [R03](LITERATURA.md#r03), [R04](LITERATURA.md#r04), [R07](LITERATURA.md#r07), [R16](LITERATURA.md#r16), [R17](LITERATURA.md#r17), [R20](LITERATURA.md#r20), [R23](LITERATURA.md#r23).

- **Hipoteza / warunek:** Efekt badanego MLP przechodzi przewidywalnym szlakiem attention/MLP do odczytu.
- **Procedura:** Zapisać zmianę wszystkich późniejszych residuali po pojedynczej interwencji. Porównać end-to-end JVP, rzeczywisty forward i lokalne odczyty. Wybrane późniejsze stany przywracać z bazowego przebiegu według zamrożonego planu mediatorów. Zamrozić manifest selektora mediatorów, budżet i regułę dopasowania w nowym seedzie. Testować ten sam proces i przewidziany kierunek odpowiedzi co T09.
- **Kontrole:** Sham i patching w obu kierunkach; pojedynczy mediator i grupy; przywracanie wybranych tokenów wobec reszty. Nie sumować efektów pośrednich tak, jakby były addytywne. Kontrole redundancji, normalizacji i nietypowych stanów po podmianie. Przywrócenie całego późnego stanu jest kontrolą dodatnią, nie dowodem małej ścieżki.
- **Metryki:** Wzmocnienie, tłumienie i zmiana kierunku δX; zmiana efektu wyjściowego po przywróceniu stanu; interakcja między mediatorami.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeśli różne szlaki dają ten sam rescue lub występuje self-repair, ograniczyć wniosek do udziału szlaku, nie unikalnej konieczności.
- **Zakres startowy i koszt:** Wybrane przypadki pilota oraz wcześniej określona próbka odłożona; wszystkie 12 punktów pomiaru.
- **Planowany wynik:** `downstream_paths.parquet`.

<a id="t12"></a>

## T12. Podpisane wkłady i współaktywacja neuronów

**Zakres:** dla_tezy_współdziałanie. **Zależności:** T01, T06, T10. **Literatura:** [R05](LITERATURA.md#r05), [R09](LITERATURA.md#r09).

- **Hipoteza / warunek:** Przeciwne kierunki mają funkcjonalne współdziałanie zależne od treści ponad marginalne aktywacje.
- **Procedura:** Liczyć u_c=z_c v_c, podpisane iloczyny skalarne wkładów par, współaktywację i q w residualu oraz odczycie. Jawny próg aktywności i analiza wrażliwości. Oddzielić przeciwne kierunki W2 od przeciwnych aktywnych wkładów u.
- **Kontrole:** Inicjalizacja, losowe etykiety, wielokrotne permutacje aktywacji wewnątrz dopasowanego regionu/obrazu/checkpointu; losowe pary dopasowane normą i aktywnością. Selekcja wyłącznie na discovery.
- **Metryki:** Iloczyn skalarny wkładów, udział znoszenia, q, częstość par i ich wpływ funkcjonalny z CI po filmach/seedach.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Efekt odtwarzany przy inicjalizacji lub znikający po kontroli norm nie dowodzi wyuczonej kooperacji. Współaktywacja sama nie dowodzi kompensacji.
- **Zakres startowy i koszt:** Tanie agregaty 600×768; ograniczona liczba par, bez zapisu wszystkich tensorów 600×768×768.
- **Planowany wynik:** `signed_cooperation.parquet`.

<a id="t13"></a>

## T13. Ablacje pojedyncze, parowe i grupowe

**Zakres:** dla_tezy_współdziałanie. **Zależności:** T12, T16. **Literatura:** [R05](LITERATURA.md#r05), [R07](LITERATURA.md#r07), [R08](LITERATURA.md#r08), [R16](LITERATURA.md#r16).

- **Hipoteza / warunek:** Wybrane grupy mają specyficzne wspólne działanie, którego nie wyjaśnia selekcja ani dekoder.
- **Procedura:** Porównać baseline, A wyłączone, B wyłączone i oba wyłączone; grupy 2/8/16/32 na tych samych wejściach. Zerować wkład po GELU, zachować bias W2. Pełny downstream dla wcześniejszych MLP. Liczyć wspólną odpowiedź oraz rAB=ΔyAB−ΔyA−ΔyB.
- **Kontrole:** Losowe grupy dopasowane normą, suma efektów pojedynczych ablacji i pełny JVP, inicjalizacja; ciągłe wyjścia stałego kandydata oddzielnie od top1 i maski maksimów.
- **Metryki:** q oraz bezwzględny efekt w pikselach/normalizowanym błędzie; reszta nieliniowa rAB; IoU, logit, margines i wielkość grupy.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Małe q po selekcji na tym samym odczycie nie jest wynikiem potwierdzającym; skok argmax nie dowodzi synergii MLP.
- **Zakres startowy i koszt:** Pilot stanów i obrazów zgodny z T09; zwiększać zakres po sprawdzeniu transferu T17.
- **Planowany wynik:** `group_ablation.parquet`.

<a id="t14"></a>

## T14. Czy następny blok koryguje poprzedni

**Zakres:** dla_tezy_korekta. **Zależności:** T02, T07, T26. **Literatura:** [R04](LITERATURA.md#r04), [R08](LITERATURA.md#r08), [R19](LITERATURA.md#r19), [R22](LITERATURA.md#r22).

- **Hipoteza / warunek:** Kolejny blok adaptacyjnie odpowiada przeciwnie na zmianę rzeczywistego wkładu poprzednika.
- **Procedura:** W jednym forwardzie zmieniać x_next=x_prev+a na x_next+αa; start α∈{0,±0.05,±0.1,±0.25}. Mierzyć b(x_next+αa)−b(x_next) i projekcję na a. Rozszerzenie do ±1 tylko jako test dużej dawki. Oddzielnie attention→MLP oraz MLP→attention/cały następny blok.
- **Kontrole:** Losowe kierunki o tej samej normie, kierunki ortogonalne i pochodzące z niesąsiednich bloków, inicjalizacja, rzeczywiste wkłady zapisane hookami. Osobny przebieg z LN o średniej/odchyleniu zamrożonych na nieperturbowanym forward i zachowanym affine: to kontrola diagnostyczna, nie oryginalny model. Wspólny odejmowany stan i shuffle czasu jako kontrole pozornego opóźnienia.
- **Metryki:** Nachylenie a·Δb/(α||a||²) dla α≠0, krzywa odpowiedzi i skutek na wyjściu; późniejsza korekta oddzielona od jednoczesnego znoszenia w jednym MLP.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Ujemny cosinus bez adaptacyjnej odpowiedzi nie potwierdza korekty. Efekt wyjaśniony renormalizacją zawęża tezę do LN. Wynik między blokami nie mówi o opóźnieniu między checkpointami.
- **Zakres startowy i koszt:** 5 przejść między 6 blokami i 6 przejść attention→MLP; pilot 7 checkpointów × 24 klatki, małe α.
- **Planowany wynik:** `cross_depth_correction.parquet`.

<a id="t15"></a>

## T15. Przywracanie poprawnego wyniku przez podmianę aktywacji

**Zakres:** obowiązkowy. **Zależności:** T10, T11. **Literatura:** [R06](LITERATURA.md#r06), [R07](LITERATURA.md#r07), [R08](LITERATURA.md#r08), [R10](LITERATURA.md#r10), [R16](LITERATURA.md#r16), [R17](LITERATURA.md#r17), [R19](LITERATURA.md#r19), [R23](LITERATURA.md#r23).

- **Hipoteza / warunek:** Zidentyfikowany komponent przenosi informację potrzebną do obserwowanego zachowania.
- **Procedura:** Zapisać clean i corrupted przebiegi. W obu kierunkach podmienić wybrane neurony/tokeny/podprzestrzenie, przeliczyć resztę sieci. Używać donorów naturalnych i dopasowanych; dla translacji wyrównać siatkę. Testować zachowanie tylko proponowanego obwodu oraz usunięcie go i podzbiory. Zamrozić selector komponentu i donorów oraz budżet podmiany. Rozdzielić postulowaną zmienną od kontekstu przez zwalidowane pary donorów; przewidzieć odzysk wybranej odpowiedzi i brak zmiany odpowiedzi kontrolnej.
- **Kontrole:** Zero/mean/resample baseline, sham, dopasowane normą random directions, identyczny donor, błędny donor; dormant pathway i redundancja OR. Rozdzielić sterowanie wynikiem od odtworzenia normalnego obliczenia. Cofnięcie dokładnie własnej poprawki i przywrócenie całego stanu służą tylko kontroli dodatniej; główny rescue musi obejmować ograniczony wcześniej określony komponent.
- **Metryki:** Bezwzględny odzysk E_edge/IoU i marginesu, noising i denoising, recovery ratio tylko przy ustalonym dolnym progu mianownika; wielkość minimalnego zestawu.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Rescue jedynie przy dużym sztucznym patchu albo dowolnym donorze nie potwierdza specyficznego mechanizmu; brak efektu pojedynczej ablacji nie dowodzi nieistotności przy redundancji.
- **Zakres startowy i koszt:** Naturalne pary z pilota i niezależne filmy; pełne forwardy, nie sam ostatni head.
- **Planowany wynik:** `activation_rescue.parquet`.

<a id="t16"></a>

## T16. Rozdzielenie regresji ramki, maksimów i rankingu

**Zakres:** obowiązkowy. **Zależności:** T01, T05. **Literatura:** [R05](LITERATURA.md#r05), [R07](LITERATURA.md#r07).

- **Hipoteza / warunek:** Efekty przypisywane polu nie są wyłącznie nieciągłością selekcji detekcji.
- **Procedura:** Dla każdego wariantu zapisać wszystkie 600 logitów, raw ltrb, peak mask 3×3, top-k IDs i bbox. Porównać stałego kandydata z clean, tego samego przypisanego obiektu oraz rzeczywisty dekoder. Monitorować wejście/wyjście kandydatów z peak mask i zmianę zwycięzcy.
- **Kontrole:** Gradient/JVP wszystkich konkurentów, nie tylko początkowego top2; tie handling i przypadki no-detection; nie wybierać ex post najlepszego kandydata według GT.
- **Metryki:** Ciągły błąd krawędzi, margin crossing, peak change, winner switch, IoU finalnej ramki, częstość switchy szkodliwych/korzystnych/neutralnych.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeśli duży efekt w całości tłumaczy margines plus selekcja, claim dotyczy podatności dekodera, nie nieliniowej synergii neuronów.
- **Zakres startowy i koszt:** Tani przegląd wszystkich 600; pełne interwencje dla z góry określonej populacji.
- **Planowany wynik:** `decoder_events.parquet`.

<a id="t17"></a>

## T17. Transfer zamrożonych selektorów

**Zakres:** dla_tezy_współdziałanie. **Zależności:** T12, T13, T15, T26. **Literatura:** [R06](LITERATURA.md#r06), [R09](LITERATURA.md#r09), [R17](LITERATURA.md#r17), [R20](LITERATURA.md#r20).

- **Hipoteza / warunek:** Sposób wyboru ważnych par/kierunków działa poza próbką, na której go opracowano.
- **Procedura:** Zamrozić algorytm, progi i budżet grup na discovery. Zastosować do nowych filmów, checkpointów i seedów. Osobno badać identyczne pary w jednym treningu oraz przenoszenie reguły wyboru między treningami.
- **Kontrole:** Identyczny dostęp do danych i gradientów dla baseline; frozen i adaptive selectors raportowane osobno; bez przenoszenia surowego numeru neuronu między seedami.
- **Metryki:** Held-out efekt interwencji, precision of harmful switch prediction, przewaga nad margin/pełny JVP/norm ranking.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Efekt wyłącznie w danych selekcji obala generalizację selektora; odtworzenie go po ponownym doborze na TEST nie jest transferem.
- **Zakres startowy i koszt:** Discovery cross-validation po filmach; końcowa ocena nowych seedów×filmów.
- **Planowany wynik:** `selector_transfer.parquet`.

<a id="t18"></a>

## T18. Skąd zmienia się pole między checkpointami

**Zakres:** dla_tezy_uczenie. **Zależności:** T01, T04, T06. **Literatura:** [R01](LITERATURA.md#r01), [R03](LITERATURA.md#r03).

- **Hipoteza / warunek:** Można oddzielić zmianę mierzonej funkcji od zmiany miejsca, w którym ją badamy.
- **Procedura:** Dla t,t+1 wyliczyć cztery kombinacje F_t(x_t),F_t(x_t+1),F_t+1(x_t),F_t+1(x_t+1), analogicznie metrykę geometryczną. Użyć symetrycznego podziału wkładu zmiany wag i wejścia, oraz jawnej interakcji. Oddzielić bufory BN od parametrów.
- **Kontrole:** Obie kolejności podstawienia, stała baza, kontrola OOD hybrydowego wejścia; gęste i rzadkie odstępy cp.
- **Metryki:** Dokładna rekonstrukcja różnicy, udziały wejścia/wag/interakcji, brak narzuconej dodatniości udziałów.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Rozkład czterech rogów opisuje zależność algebraiczną; bez odtworzenia optymalizatora nie wyjaśnia przyczyny kroku treningowego.
- **Zakres startowy i koszt:** Istniejące wagi; 6 wybranych sąsiedztw cp×24 obrazy; następnie nowy gęsty log.
- **Planowany wynik:** `checkpoint_factorial.parquet`.

<a id="t19"></a>

## T19. Rzeczywisty krok optymalizatora i ruch tokenu

**Zakres:** dla_tezy_uczenie. **Zależności:** T18, T28. **Literatura:** [R11](LITERATURA.md#r11).

- **Hipoteza / warunek:** Ruch między checkpointami wynika ilościowo ze zmian parametrów i buforów.
- **Procedura:** W nowym krótkim biegu zapisać optimizer, scheduler, RNG, sampler, batch i BN przed/po krokach. Obliczyć faktyczne Δθ oraz Dθf Δθ i wkład buforów. Odtworzyć ten sam krok i krótkie kontynuacje z kontrolowaną zmianą batcha lub składnika loss.
- **Kontrole:** Paired forks z pełnego tego samego stanu; identyczna kolejność danych dla porównań; weight decay i momentum AdamW osobno; mały krok vs Taylor remainder.
- **Metryki:** Reconstruction error token motion, wkład parametrów/BN, cosine i norm predicted/actual; paired effect na sondach.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeśli brak stanów optymalizatora w starych cp, nie odtwarzać fikcyjnej historii AdamW. TracIn/gradient dot product nie jest dokładnym ruchem AdamW.
- **Zakres startowy i koszt:** Nowe krótkie okna, np.3 etapy×100 kroków, ostateczny koszt po E0; nie pełna kampania od razu.
- **Planowany wynik:** `optimizer_replay.parquet`.

<a id="t20"></a>

## T20. Czy efekt jest wyuczony i zależny od zadania

**Zakres:** obowiązkowy. **Zależności:** T06, T09, T15. **Literatura:** [R03](LITERATURA.md#r03), [R04](LITERATURA.md#r04), [R09](LITERATURA.md#r09).

- **Hipoteza / warunek:** Efekt funkcjonalny przekracza geometrię inicjalizacji i uczenie niezwiązane z właściwym zadaniem.
- **Procedura:** Porównać init, wytrenowany model, stopniową randomizację warstw i ograniczony trening z losowymi etykietami. Random bbox przypisać stałą permutacją między obrazami, zachowując dopuszczalne rozmiary; dodatkowo pomocnicze paired loss ablations heat/box, jeśli główna teza wymaga.
- **Kontrole:** Takie same budżety kroków i preprocessing; raportować dopasowanie train loss osobno; losowe etykiety nie są równoważnym rozkładem zadania.
- **Metryki:** Geometria i docelowy efekt interwencji, nie tylko accuracy; interakcja trained-vs-init × target-vs-sham.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Ten sam funkcjonalny efekt przy inicjalizacji lub po randomizacji nie potwierdza mechanizmu powstałego przez uczenie zadania. Bez odpowiedniej kontroli zadania ograniczyć claim do zmiany względem inicjalizacji. Krótki null nie kontroluje pełnego końcowego treningu.
- **Zakres startowy i koszt:** Inicjalizacja bez treningu. Random-label/loss ablations po bramce pilota, w osobnym budżecie i manifeście liczby seedów oraz pełnych lub dopasowanych krótkich okien. Pięć podstawowych nowych treningów nie zawiera tych biegów.
- **Planowany wynik:** `learning_nulls.parquet`.

<a id="t21"></a>

## T21. Niezależne powtórzenia treningu

**Zakres:** obowiązkowy. **Zależności:** T27, T28. **Literatura:** [R14](LITERATURA.md#r14), [R15](LITERATURA.md#r15), [R18](LITERATURA.md#r18).

- **Hipoteza / warunek:** Efekt nie jest osobliwością pojedynczego treningu.
- **Procedura:** Użyć s47 tylko discovery; dwa nowe pełne treningi rozwojowe, potem zamrozić protokół i wykonać trzy nowe confirmation. Rozdzielić seed inicjalizacji, danych i perturbacji; zapisywać pełny stan.
- **Kontrole:** Zgodny przepis treningu i liczba ekspozycji, bez wznowienia z resetem samplera; historyczne z48/z49 tylko gdy zgodne i kompletne. Reguła clean-winner jest identyczna, lecz jego indeks może się różnić między seedami; nie utożsamiać tego z tą samą kotwicą. Dla P2 wspólna kotwica GT i populacja obrazów jako oddzielny pomiar mechanistyczny.
- **Metryki:** Efekt każdego seeda, CI i wrażliwość leave-one-seed-out; ustalona precyzja głównego kontrastu.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak spójności lub przedziały zbyt szerokie oznaczają nierozstrzygnięcie; więcej klatek nie zastąpi treningów.
- **Zakres startowy i koszt:** 5 nowych pełnych biegów jako startowy budżet, nie wymaganie konferencji; liczba końcowa według precyzji przed testem.
- **Planowany wynik:** `seed_replication.parquet`.

<a id="t22"></a>

## T22. Druga architektura

**Zakres:** dla_tezy_ogólne_ViT. **Zależności:** T09, T15, T21. **Literatura:** [R03](LITERATURA.md#r03), [R13](LITERATURA.md#r13).

- **Hipoteza / warunek:** Mechanizm przenosi się poza szczegóły s47.
- **Procedura:** Po potwierdzeniu głównego efektu wybrać drugą szerokość/głębokość ViT albo odrębny transformerowy detektor. Przenieść definicję metryki i interwencji, nie numery neuronów. Przy odrębnej głowie jawnie dostosować readout.
- **Kontrole:** Zgodny cel detekcji i jawny budżet; CNN/MLP-mixer tylko gdy claim wymaga specyficzności attention. Nowego modelu nie stroić na końcowym TEST.
- **Metryki:** Ten sam znormalizowany kontrast, zakres perturbacji, effect size/CI i koszt; każda architektura osobno.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak transferu zawęża claim do rodziny architektury; sama druga liczba parametrów nie dowodzi ogólności wszystkich ViT.
- **Zakres startowy i koszt:** Najpierw mały pilot drugiego modelu, potem niezależne seedy według T27; koszt benchmarkowany.
- **Planowany wynik:** `architecture_transfer.parquet`.

<a id="t23"></a>

## T23. Drugi zbiór z lokalizacją obiektu

**Zakres:** dla_tezy_ogólna_detekcja. **Zależności:** T00, T09, T21. **Literatura:** [R05](LITERATURA.md#r05), [R10](LITERATURA.md#r10).

- **Hipoteza / warunek:** Mechanizm nie zależy wyłącznie od obrazów chirurgii zaćmy.
- **Procedura:** Wybrać publiczny zbiór detekcji i zgodny odczyt przed wynikami. Dla COCO określić wieloobiektowe matching i metryki albo jawne single-object podzadanie z regułą wyboru niezależną od predykcji. Udokumentować licencję i split.
- **Kontrole:** Nie zastępować replikacji IoU klasyfikacją CIFAR/Imagenette. Kontrola rozmiaru obiektu, rozdzielczości i tokenizacji.
- **Metryki:** E_edge, IoU i standardowa metryka zbioru, replikacja target-vs-sham/rescue; oddzielne wyniki domen.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Transfer wyłącznie na klasyfikację potwierdza co najwyżej geometrię, nie mechanizm lokalizacji.
- **Zakres startowy i koszt:** Dataset/adaptor obecnie do wyboru; najpierw zgodność zadania i budżet, potem replikacja.
- **Planowany wynik:** `dataset_transfer.parquet`.

<a id="t24"></a>

## T24. Nowe typy zaburzeń i przypadki trudne

**Zakres:** dla_tezy_odporność. **Zależności:** T10, T15, T21. **Literatura:** [R02](LITERATURA.md#r02), [R07](LITERATURA.md#r07).

- **Hipoteza / warunek:** Wyjaśnienie działa poza perturbacjami użytymi do jego wyboru.
- **Procedura:** Zamrozić interwencję na discovery jasność/translacja; odłożyć rodziny blur/kontrast/occlusion lub odwrotnie. Stratyfikować rozmiar i pozycję narzędzia, odblaski, zasłonięcia i błędne clean detekcje.
- **Kontrole:** Severity ustalone przed TEST, audit GT po zmianach, te same grupy i shams; naturalne trudne obrazy oddzielnie od syntetycznych.
- **Metryki:** Kontrast target-vs-sham i rescue per rodzina; degradation curves; liczność i niepewność rzadkich grup.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Efekt ograniczony do jednej rodziny nie uzasadnia ogólnej odporności; brak mocy w rzadkich grupach raportować.
- **Zakres startowy i koszt:** Nowe perturbacje na zamkniętych filmach, bez nowego treningu jeśli wystarczy eval.
- **Planowany wynik:** `perturbation_transfer.parquet`.

<a id="t25"></a>

## T25. Prognozowanie jakości bez dostępu do GT

**Zakres:** opcjonalny_użytkowy. **Zależności:** T16, T17, T21, T26. **Literatura:** [R14](LITERATURA.md#r14).

- **Hipoteza / warunek:** Mikroskop dostarcza praktycznej informacji o ryzyku błędnej ramki ponad dostępne metryki.
- **Procedura:** Wybrać jedno zdarzenie, np. IoU<0.5, i jedną miarę główną przed TEST. Uczyć predyktor na discovery; porównać confidence/margin/norm/JVP/augment consistency z dodatkiem geometrii i kooperacji.
- **Kontrole:** Bez GT w cechach i wyborze tokenów; bez przyszłych cp, finalnych par, PCA przyszłości i okna±300; baseline o tym samym dostępie do danych i compute.
- **Metryki:** AUPRC z prevalence, risk-coverage, Brier/calibration i GPUczas; AUROC pomocniczo.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak held-out przewagi przy uczciwym koszcie odrzuca użytkowy claim; nie odrzuca automatycznie mechanistycznego odkrycia.
- **Zakres startowy i koszt:** Opcjonalna gałąź po głównym wyniku; nie konieczna do artykułu o mechanizmie.
- **Planowany wynik:** `quality_prediction.parquet`.

<a id="t26"></a>

## T26. Silne porównania i granica nowości

**Zakres:** obowiązkowy. **Zależności:** T02, T06. **Literatura:** [R01](LITERATURA.md#r01), [R02](LITERATURA.md#r02), [R03](LITERATURA.md#r03), [R04](LITERATURA.md#r04), [R10](LITERATURA.md#r10), [R12](LITERATURA.md#r12), [R13](LITERATURA.md#r13), [R17](LITERATURA.md#r17).

- **Hipoteza / warunek:** Nowa interpretacja wnosi coś ponad znane miary i dokładny lokalny model.
- **Procedura:** Implementować pełny JVP do wyjścia, normy J/F, wartości singularne I+J, gradient×activation, czułość marginesu, coupling Aubry v4 oraz odpowiedź korekcyjną Patrawala. Coupling podstawowy dotyczy całego bloku J_block=(I+J_M)(I+J_A)−I; warianty podkroków/ilorazu to rozszerzenia. Zamrozić K, p i obsługę zdegenerowanych podprzestrzeni SVD. CKA jako opis podobieństwa. CAAP/Prisma dla właściwego porównania atrybucji/infrastruktury. Zdefiniować kartę M1 z prospektywną podpisaną prognozą, zakresem i warunkiem małego efektu. M0 obejmuje lokalne Cq i rzeczywisty decoder. Wartość M1 oceniać jako sprawdzalne prostsze wyjaśnienie, nie przewagę informacyjną nad J.
- **Kontrole:** Ten sam split, normy, liczba wybranych komponentów, cel i budżet; wstępna replikacja baseline na znanym przykładzie; kod autorów wersjonowany.
- **Metryki:** Przewidywanie interwencji, prostota/reprodukowalność wyjaśnienia, efekt ponad dopasowane normą null; nie oczekiwać dodatkowej informacji ponad pełne J.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Jeżeli wynik redukuje się do znanej coupling/TLCM/linear sensitivity, ograniczyć nowość do zweryfikowanego rozszerzenia albo zmienić pytanie.
- **Zakres startowy i koszt:** Mały benchmark baseline przed dużą kampanią; koszt osobno dla każdej metody.
- **Planowany wynik:** `baseline_comparison.parquet`.

<a id="t27"></a>

## T27. Zamrożenie analizy i wnioskowanie statystyczne

**Zakres:** obowiązkowy. **Zależności:** T00. **Literatura:** [R14](LITERATURA.md#r14), [R15](LITERATURA.md#r15), [R18](LITERATURA.md#r18).

- **Hipoteza / warunek:** Wniosek nie zależy od selekcji najładniejszej warstwy, seeda lub miary.
- **Procedura:** Po discovery zamrozić komponent, kierunek hipotezy, główny endpoint, praktyczny próg, wykluczenia, agregację, liczbę seedów i brakujące dane. Agregować pary po filmie; resampling seed×film z zachowaniem całych sparowanych trajektorii. Parametry estymatora i plan mocy sprawdzić symulacją na pilocie. Lock także dla reguły tokenu, mediatorów, donorów, budżetów, odłożonej warstwy, wspólnej kotwicy P2 i karty przewidywań. Zablokować prognozy przed odczytem interwencji.
- **Kontrole:** Holm dla kilku równorzędnych primary tests lub jeden primary kontrast; FDR opisowo dla atlasu. Raportować każdy seed i leave-one-out. Nie liczyć tokenów/cp jako niezależnego N. Dla prerejestrowanego znaku s i praktycznego δ*: dolna granica właściwego CI dla sP1 > δ* wspiera wielkość; górna < δ* ją odrzuca; pozostały wynik jest nierozstrzygający. Oglądanie confirmation nie zmienia selektorów.
- **Metryki:** Wielkość efektu, 95% CI, simulation coverage/minimal detectable effect; equivalence margin dla negatywnego twierdzenia.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak significance nie dowodzi zera. Za mała liczba seedów wymaga szerszych CI lub jawnie wąskiego claimu; nie wolno zmieniać endpointu po TEST.
- **Zakres startowy i koszt:** CPU; planowanie przed confirmation, bootstrap agregatów po zakończeniu.
- **Planowany wynik:** `preregistration_and_statistics.json`.

<a id="t28"></a>

## T28. Koszt, precyzja i deterministyczność

**Zakres:** obowiązkowy. **Zależności:** T01, T02. **Literatura:** [R14](LITERATURA.md#r14), [R15](LITERATURA.md#r15).

- **Hipoteza / warunek:** Eksperyment jest wykonalny, a efekt przewyższa błąd numeryczny.
- **Procedura:** Zmierz forward/JVP/VJP/J192/SVD/IO po rozgrzewce,co najmniej 3 powtórzenia; peak VRAM/RAM, dysk, GPUh. Porównaj float32/float64 referencyjnie, TF32/autocast kontrolowane, wersje torch/timm/CUDA utrwalone.
- **Kontrole:** Ponowny shard z tym samym seedem; drugi backend podpróba; tolerancje bezwzględne i względne; diagnoza rozbieżności zamiast luzowania pod wynik.
- **Metryki:** Runtime distribution, numerical error, resource manifests, accounting plan vs actual.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Efekt wielkości szumu numerycznego nie jest wynikiem; brak zasobów prowadzi do strumieniowania/próbkowania, nie fikcyjnej pełnej macierzy.
- **Zakres startowy i koszt:** E0; brak materializacji pełnego attention115200²; plan kosztu z pomiaru.
- **Planowany wynik:** `resource_benchmark.json`.

<a id="t29"></a>

## T29. Odtwarzalność i niezależny przegląd wyniku

**Zakres:** obowiązkowy. **Zależności:** T21, T26, T27, T28. **Literatura:** [R14](LITERATURA.md#r14), [R15](LITERATURA.md#r15).

- **Hipoteza / warunek:** Osoba niezwiązana z analizą potrafi odtworzyć główną liczbę i ograniczenia claimu.
- **Procedura:** Z nowego środowiska odtworzyć mini-zestaw liczb z publicznego/możliwego do udostępnienia przykładu, sprawdzić kod→tabela→figura. Zachować manifesty, hashy, seed, environment, checkpoint provenance, dostępność danych, negatywne wyniki. Po pobraniu rzeczywistych wyników wykonać ponowną recenzję według RECENZJA_PRZEDPUBLIKACYJNA.md: przewidywania, kontrola czułości/dekodera, nietautologiczny rescue, transfer, wyniki ujemne i zgodność zakresu artykułu z dowodami.
- **Kontrole:** Review osoby/agenta niepiszącego miernika; wyniki raw vs agregaty; atomowe shardy i kontrola kompletności; anonimizacja pakietu zgłoszenia.
- **Metryki:** Zgodność głównego kontrastu/CI, brak brakujących shardów, pełna ścieżka źródłowa każdej figury.
- **Jednostka:** niezależny trening i film/zabieg; tokeny, obrazy, checkpointy i shams są pomiarami powtarzanymi.
- **Falsyfikacja / ograniczenie:** Brak odtwarzalnego głównego wyniku lub nieudokumentowane wykluczenia blokują gotowość do zgłoszenia.
- **Zakres startowy i koszt:** Mały CPU/GPU reproduction subset; żadnej obietnicy akceptacji konferencji.
- **Planowany wynik:** `reproduction_report.json`.
