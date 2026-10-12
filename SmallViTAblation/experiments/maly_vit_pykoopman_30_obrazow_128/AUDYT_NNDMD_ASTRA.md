# Audyt NNDMD: pełne 768 współrzędnych

Status: **FINAL — audyt wykonania i metryk zakończony**. Audyt przedtestowy: 2026-10-02, 13:09–13:17; odczyt końcowych wyników dopiero po zamrożeniu, 13:19–13:25, Europe/Warsaw. Audytor: Astra. Własność audytora obejmuje wyłącznie ten plik. Bez zmian kodu wykonawcy, treningów i użycia GPU.

Nie znalazłem błędu w indeksowaniu czasu, normalizacji, zasadniczej funkcji straty ani autonomicznej propagacji. Początkowo wykryta rozbieżność SHA skryptu została **rozstrzygnięta przed testem** przez odtworzenie dokładnego snapshotu i niezależne porównanie AST. Historia ustalenia A-05 pozostaje poniżej.

**Wynik predykcyjny jest negatywny:** na H500 znak wszystkich J128 wynosi 54,4975 procent, a MSE jest 3,6727 razy większy niż dla persystencji. Żaden z 50 ocenianych horyzontów nie osiąga 85 procent znaku ani dodatniego MSEskill, zarówno w J128, jak i full768. Poprawność obliczeń nie oznacza skuteczności modelu.

Stosuję procedurę skilla `audit-project-correctness` w granicach jawnie zleconej własności. W tym repozytorium nie istnieje `Research/Audyty/dashboard.md`; nie tworzę rejestru innego projektu ani dodatkowych plików AU. Ten NNDMD ma pierwszy niezależny audyt. Wcześniejsze audyty [benchmarku 99 modeli](../maly_vit_prognoza_tozsamosci_500_bez_pca/LITERATURA_I_AUDYT.md) oraz [natywnych kanałów](../maly_vit_tozsamosc90_bez_pca_langevin/AUDYT_ASTRA.md) nie weryfikowały tej architektury.

## Dowody i zgodność protokołu

Odczytano [PLAN_NNDMD.md](PLAN_NNDMD.md), [nndmd_fit.py](nndmd_fit.py), rzeczywisty zainstalowany `pykoopman/regression/_nndmd.py` w wersji 1.2.1, [test_nndmd_fit.py](test_nndmd_fit.py), źródłowe funkcje metryk oraz artefakty niżej. Nie otwierano przyszłych wyników końcowego holdoutu przed wyborem modelu.

| Element | Zweryfikowany zakres |
|---|---|
| Dane | [DANE_768_WERYFIKACJA.json](DANE_768_WERYFIKACJA.json): `[101,34,600,768]`, float32, 6 329 548 928 bajtów, skończoność i zgodność wcześniejszego J128. Audytor odczytał manifest; nie powtarzał całego forwardu ani hashowania 6,3 GB. |
| Reprezentacja | `nndmd_fit.py:225–242`: encoder 768→768→768 z SiLU; decoder liniowy 768→768; operator uczony 768×768. Rzeczywiste `NNDMD.fit(list_of_trajectories)`; brak PCA lub wstępnej rotacji/redukcji. J128 jest dodatkową oceną i składnikiem straty po dekodowaniu. |
| Tożsamość trajektorii | `:22–46`: jedna trajektoria to ten sam token tego samego obrazu. Wewnętrzny fit ma obrazy 0–7 i 12–27, 406 trajektorii po 31 stanów, do aktualizacji 300. Walidacja: obrazy 28–33, origin 300, cele do 500. Holdout: 8–11. |
| Ważenie | `:49–73`: prawdopodobieństwo okna jest wagą coverage tokenu podzieloną przez liczbę okien; masa obrazu wynosi 1/24. Z manifestu niezależnie przeliczono maksymalny błąd tej masy: 1,39×10⁻¹⁷. Liczby losowań/epokę dla LF5/10/20: 10556/8526/4466. |
| Normalizacja | `:35–40,60–65`: wspólny skalar 0,5973187081613646 dla fitu wewnętrznego, wyliczony z ważonych momentów jego prefiksu. Wszystkie kanały dzielone przez ten sam skalar; średnia używana do transformacji wynosi zero. Brak statystyk z przyszłego sufiksu lub obrazów walidacyjnych. |
| Okna | Kod biblioteki `:973–984`: przy look_back=1 i długości T powstaje T−LF pełnych okien. Dla obecnych 31/51 stanów i LF≤20 długości są jednakowe. Ostatni cel okna pozostaje w prefiksie. Prawdziwe stany późniejsze od początku okna są dozwolonymi celami treningu, jeżeli nie przekraczają 300 lub 500 odpowiedniej fazy. |
| Wybór | `nndmd_fit.py:176–201,309–328`: walidacja tylko po epokach 5/10/15/20; kryterium to średnia pięciu RMSE w oryginalnych jednostkach decoded J128 dla H10/20/50/100/200. Każde RMSE to pierwiastek MSE po coverage i równych wagach sześciu obrazów. Full768 jest raportowane osobno. |
| Refit | `:330–351`: nowa inicjalizacja z ustalonym seedem, wszystkie 30 fit, prefiks do 500, wybrana liczba epok i LF/beta. Więcej okien oznacza więcej kroków optymalizatora niż w tej samej liczbie epok wewnętrznego fitu; to jawna reguła protokołu, nie dokładne odtworzenie budżetu update'ów. |
| Prognoza | `:76–90`: encoder używany raz na originie; następnie mnożenia przez A w learned state i dekodowanie. Żadnego wstrzykiwania przyszłego pomiaru ani ponownego kodowania prognozowanego z w pętli. |

## Funkcja straty i adaptery

Niech `x=z/s`, `phi=E(x)`, `D(phi)=C phi` i `A=exp(G)`. Implementacja minimalizuje sumę trzech typów MSE: rekonstrukcji początku, rekonstrukcji prawdziwych przyszłych stanów wewnątrz prefiksu oraz przewidywania tych stanów przez autonomiczny rollout. Do każdego rodzaju dodaje beta razy analogiczny MSE w decoded J128. Kanały J128 są zawarte również w składniku full768. Dla beta=1 jednostkowy błąd pojedynczego kanału J dostaje wagę `1/768+1/128`, czyli siedem razy wagę kanału spoza J. To zamierzone dodatkowe premiowanie podzbioru, nie siedmiokrotne powielenie tokenów.

Wektorowa maska `:93–102` ma ten sam licznik i dzielnik co pakietowa `MaskedMSELoss`. Przy obecnych pełnych oknach średnia jest po wszystkich LF×D elementach każdej próbki. Próbowanie z replacement daje zamierzony ważony cel w wartości oczekiwanej. Poszczególne minibatch'e ani epoki nie muszą mieć dokładnie równych mas obrazów. Dla hipotetycznych trajektorii o różnych krótkich długościach dzielnik maski dodatkowo ważyłby długość; takie przypadki nie występują w tym biegu.

Dowody wykonawcy: [NNDMD_VECTORIZATION_SMOKE.log](NNDMD_VECTORIZATION_SMOKE.log) zawiera cztery testy. Porównanie beta0 obejmuje wartość i gradienty oryginalnego oraz zaadaptowanego treningu; beta1 sprawdza algebraiczną sumę full+J; pozostałe testy obejmują prefix-poison i wagi okien. [NNDMD_PILOT_MASK_EQUIVALENCE.json](NNDMD_PILOT_MASK_EQUIVALENCE.json) podaje dokładnie zerową różnicę walidacji slow/fast po 5/10/15 epokach. Audytor odczytał kod i wyniki tych testów, nie uruchamiał ich ponownie. Test gradientów dotyczy małego przykładu CPU, a zgodność pełnego pilota wspiera zakres faktycznego wykonania.

`matrix_exp` jest liczony raz na minibatch, a wspólny tensor A używany w całym rollout. Autograd sumuje pochodne wszystkich użyć; współdzielenie tensora nie odcina gradientu. Pakietowy Adam pozostaje Adamem z lr=0,001. Test [NNDMD_SMOKE.json](NNDMD_SMOKE.json) potwierdza na małym układzie orientację operatora i zgodność `package.predict(n)` z encode-once rollout, maxabs=0; nie dowodzi zbieżności do prawdziwego generatora.

## Istotne ograniczenia matematyczne

**A-01 — P1, ograniczenie strukturalne, potwierdzone.** Kwadratowy liniowy decoder nie gwarantuje nieliniowej dynamiki po dekodowaniu. Jeżeli C jest odwracalna, zachodzi dokładnie

`zhat_h = s C A^h E(z0/s) = (C A C^{-1})^h R(z0)`, gdzie `R(z)=s C E(z/s)`.

Nieliniowość znajduje się wtedy w inicjalizującej rekonstrukcji R. Przy idealnej rekonstrukcji R=id decoded rollout jest liniowy w oryginalnych kanałach. Jeśli C jest osobliwa, wyjścia pozostają w liniowym obrazie C; istnieją potencjalnie nieobserwowalne kierunki learned state. Równość liczby wymiarów 768 nie dowodzi injektywności encodera, odwracalności decoderu ani zachowania informacji. Nie zmienia to poprawności oceny predykcyjnej modelu, ale ogranicza interpretację jako dokładnej nieliniowej linearyzacji Koopmana.

**A-02 — P2, ograniczenie interpretacji, potwierdzone.** Błąd rekonstrukcji początku nie może zniknąć z głównej metryki przez odjęcie zrekonstruowanego originu. Kod poprawnie porównuje `zhat_h−z0` z `z_true_h−z0`; zatem MSE ruchu jest równe MSE stanu, a błąd rekonstrukcji wpływa na krótki horyzont. H0 RMSE rekonstrukcji należy raportować obok błędu ruchu. Nie wolno wyciągać wniosku o autonomicznej dynamice tylko z niskiego reconstruction loss.

**A-03 — P2, ograniczenie identyfikowalności, potwierdzone.** `dt=1` w bibliotece to jeden checkpoint, czyli 10 aktualizacji ViT. G jest parametrem generatora w tej jednostce, A=exp(G), a H500 oznacza A^50. Formalne G/10 lub exp(G/10) definiuje wybraną interpolację modelu; dane co 10 nie identyfikują jednoznacznie wszystkich ruchów pomiędzy checkpointami. Standardowy generator nie ma narzuconej stabilności. Również learned osie mogą być zmienione przez podobieństwo S; nie są oryginalnymi makrowektorami e_j. Normy A i reszt learned state zależą od tej reprezentacji.

**A-04 — P2, ograniczenie dowodowe, potwierdzone.** J128 zostało wybrane z pomiaru na k500. Wewnętrzna walidacja od k300 pozostaje retrospektywna w ustalonym J500, choć finalny fit i origin500 nie używają przyszłości. Cztery holdouty i jedna historia treningowa były już wcześniej oglądane. Wynik będzie eksploracyjny. 600 tokenów obrazu oraz ich współrzędne nie stanowią niezależnych replik; nie ma podstaw do populacyjnego CI z niezależnego bootstrapu kanałów.

Porównanie J128 prognozy nowego NNDMD ze statycznym EDMD ma wspólny cel i obrazy, lecz odmienne wejście: nowy model widzi pełne768, dawny tylko J128. Jest to porównanie osiąganych prognoz, a nie izolowany dowód wpływu nieliniowego encodera. Dodatkowo loss treningu premiuje J, a wybór optymalizuje jego RMSE; wynik full768 jest oceną dodatkową, nie kryterium wyboru. Różna liczba okien na epokę przy różnych LF oznacza również różne budżety kroków optymalizatora w czterech ustalonych kandydatach.

## Integralność wykonania i kwestia otwarta

**A-05 — P1, potwierdzona rozbieżność wersji; wyjaśnienie w toku.** W odczycie 13:12 SHA bieżącego `nndmd_fit.py` różniło się od `NNDMD_RUNTIME.json.script_sha256` zapisanego 12:52:14: `d0e2115335c5d7b65718e8056ab5f2a5c6915acf79652ab04404c4392a7c2945`. SHA PLAN i zainstalowanego źródła biblioteki zgadzały się. Wykonawca zgłosił dopisanie późniejszej ewaluacji, użycia zapisanego A w inference oraz korekt docstringów. Końcowe `sha256(__file__)` identyfikuje plik na dysku, a nie automatycznie kod wcześniej załadowany do procesu. Kryterium zamknięcia: dokładny snapshot zgodny ze startowym SHA, udokumentowany diff i rozdzielenie wersji treningu/ewaluacji w zamrożeniu. Samo zapewnienie o braku zmiany treningu nie zamyka tej kwestii.

Historyczny tytuł PLAN mówi „operator128”, lecz jawna korekta w wierszu 7 oraz implementacja określają 768. Tytuł nie jest dowodem trenowania 128; pozostawienie pliku w spokoju w trakcie biegu chroni zapisany hash. Ostateczny raport powinien tę historię wyjaśnić.

**Aktualizacja A-05, 13:17 — naprawa potwierdzona, bez zmiany treningu.** Audytor niezależnie obliczył SHA pliku [NNDMD_TRAIN_EXECUTED_SNAPSHOT.py](NNDMD_TRAIN_EXECUTED_SNAPSHOT.py): jest dokładnie równy startowemu d0e211…2945. Niezależne porównanie AST całego programu po usunięciu docstringów i funkcji `evaluate` wykazało równość. Odczytany [diff](NNDMD_TRAIN_VS_EVALUATION.diff) obejmuje trzy docstringi oraz osobny zapis J128 wyciętego z tej samej prognozy full768. Pierwsze wyjaśnienie wykonawcy z pamięci było szersze; dokładny snapshot pokazuje, że `evaluate` i fixed saved A już istniały na starcie. [AMENDMENT_NNDMD_SOURCE.md](AMENDMENT_NNDMD_SOURCE.md) i [NNDMD_AUDYT_ZRODLA.json](NNDMD_AUDYT_ZRODLA.json), zapisane przed testem, rozdzielają hash wykonywanego treningu i późniejszej ewaluacji. Nie zmieniono straty, samplera, wyboru ani refitu; restart nie jest potrzebny.

**Niezależna kontrola wyboru, 13:17.** Z czterech historii po epokach 5/10/15/20 ponownie wyznaczono minimum ustalonego kryterium. Najlepsze epoki LF5_B1/LF10_B1/LF20_B1/LF10_B0 to odpowiednio 15/5/5/10. Ich średnie walidacyjne J128 RMSE: 0,262227340249199 / 0,23797352815801376 / 0,26911049192720954 / 0,241094196026045. Wszystkie zapisane epoki i wartości są zgodne, różnica liczbowo zero. Zwycięzca [WYBOR_NNDMD.json](WYBOR_NNDMD.json): LF10, beta1, epoka5. Jest wybrany według RMSE, nie znaku ani końcowego holdoutu. Refit ma zatem trwać pięć epok od nowej inicjalizacji; nie jest kontynuacją zwycięskiego checkpointu.

## Zaplanowany zakres oceny końcowej — zapis przed testem

Po zapisaniu wyboru i freeze: niezależnie przeliczyć z zapisanych prognoz MSE, RMSE i znak wszystkich współrzędnych (bez filtra) po coverage, następnie równych wagach czterech obrazów. Sprawdzić tożsamość `J128 = full768[...,J]`, indeks H10..500, rekonstrukcję i stosunek MSE do persystencji. Porównać mały replay zapisanego operatora/encodera/decoderu z prognozą bez treningu, na CPU. Nie wybierać innego modelu po wyniku ani usuwać ujemnych wyników. Cel 85 procent znaku i użyteczność MSE pozostają dwiema osobnymi ocenami.

Na przedtestowym cutoffie nie było niezależnego końcowego werdyktu liczbowego. Brak blokującego błędu w matematyce implementacji nie oznaczał, że model osiągnie wymagany wynik.

## Ocena końcowa po zamrożeniu

Zamrożenie [ZAMROZENIE_NNDMD.json](ZAMROZENIE_NNDMD.json) nastąpiło o 13:17:21.215663. Niezależnie potwierdzono SHA checkpointu, PLAN i źródła późniejszej ewaluacji. Refit ma 502 trajektorie po 51 stanów, sumę wag 1,000000000000001, historię dokładnie epok 1–5 i wspólny skalar 0,6078782777854828. Wszystkie trzy wymiary wynoszą 768. Wybrano LF10/beta1 przed końcowym testem. Zgodność `package.predict` i encode-once rollout w zapisanym freeze ma maxabs=0.

Audytor odczytał ukończone [WYNIKI_NNDMD.json](WYNIKI_NNDMD.json), [NNDMD_METRYKI_HOLDOUT.json](NNDMD_METRYKI_HOLDOUT.json) oraz prognozy full768 i J128. Przy użyciu osobnych obliczeń NumPy, bez importowania funkcji agregacji wykonawcy, ponownie policzył 2400 skalarów: MSE stanu/ruchu/persystencji, znak wszystkich współrzędnych, RMSE i MSEskill, per obraz i po równych wagach obrazów, dla obu reprezentacji i wszystkich 50 horyzontów. Największa różnica wynosi 8,88×10⁻¹⁶ (MSEskill); dla znaku J128 różnica wynosi dokładnie zero. Wycinek J128 jest bitowo zgodny z full768 dla wszystkich zapisanych 50×4×600 cząsteczek.

| H, aktualizacje | Znak wszystkich J128 | RMSE J128 | RMSE persystencji J128 | MSEskill J128 |
|---:|---:|---:|---:|---:|
| 10 | 56,2995% | 0,138555 | 0,120948 | −31,2336% |
| 50 | 57,1395% | 0,195706 | 0,163220 | −43,7683% |
| 100 | 53,4512% | 0,212525 | 0,164972 | −65,9596% |
| 200 | 57,2253% | 0,234563 | 0,192491 | −48,4903% |
| 500 | 54,4975% | 0,448238 | 0,233894 | −267,2655% |

Wśród wszystkich horyzontów maksymalny znak J128 wynosi 60,2754 procent przy H320, a najlepszy MSEskill pozostaje ujemny: −18,4868 procent przy H130. To opis zakresu ukończonego testu, nie nowy wybór modelu. Full768 na H500: znak 54,5284 procent, RMSE 0,440500 wobec 0,250804 persystencji, MSEskill −208,4771 procent. Jego największy znak w całej siatce to 59,0209 procent, a największy MSEskill to −20,7112 procent. Wyniku 85 procent nie uzyskano w żadnej z tych ocen.

**Niezależny replay CPU:** z checkpointu pobrano bezpośrednio wagi encodera, decoderu i A. Ręcznie wykonano Linear→SiLU→Linear, następnie 50 kolejnych mnożeń przez zapisane A i dekodowanie, dla wszystkich 60 tokenów tool czterech holdoutów (7/20/20/13). Bez importowania funkcji `learned_rollout` i bez GPU. Największa różnica względem zapisanej prognozy GPU float32: 1,049041748×10⁻⁵; dla H10: 1,1921×10⁻⁶, H100: 2,5332×10⁻⁶. Jest to numeryczna zgodność CPU/GPU w float32, nie zgodność bitowa. Macierz A z checkpointu i osobnego NPY jest bitowo identyczna.

RMSE rekonstrukcji H0 odtworzone na CPU: full768 0,07741899176, J128 0,05871465581, zgodne z raportem odpowiednio 0,07741899165 i 0,05871465762. Błąd rekonstrukcji jest zmierzonym składnikiem zachowania modelu; **nie jest ścisłym dolnym ograniczeniem przyszłego błędu**, ponieważ późniejszy ruch może część błędów skompensować.

## Odczyt diagnostyki i ostateczny werdykt

[NNDMD_AUDYT_OPERATORA.json](NNDMD_AUDYT_OPERATORA.json) oraz jego kod podają rho(A)=1,0331977, ||A||₂=1,1936923 i ||A^50||₂=10,6771155. Są zgodne z brakiem gwarancji stabilności standardowego generatora, lecz nie dowodzą samodzielnie przyczyny każdego błędu prognozy. Decoder C ma s_min≈2,42×10⁻⁵ i kondycję≈168250. Rząd 767 oznacza rząd **przy jawnej tolerancji float32**, nie dowód matematycznej osobliwości. Ilustruje problem słabej identyfikowalności kierunków z punktu A-01.

Reszta domknięcia `A E(z_t/s)−E(z_(t+10)/s)` została policzona przez wykonawcę po teście, bez strojenia. Jej RMSE względem persystencji learned state wynosi 0,8362 na prefiksie fit i 1,2298 na przyszłym holdoucie. To diagnostyka w learned coordinates, odrębna od native J128 i od autonomicznego rollout; użycie prawdziwego przyszłego stanu do tej oceny po teście nie jest teacher forcing prognozy. Audytor sprawdził definicję i artefakt, nie powtarzał pełnej diagnostyki operatora.

Odczytano [RAPORT_NNDMD_FULL768.md](RAPORT_NNDMD_FULL768.md) i jego generator. Interpretacja odróżnia pełne/native/learned768, rozpoznawanie J128 od dynamiki, nienaruszony prefiks od wcześniej znanego testu oraz rekonstrukcję od przewidywania. Wewnętrzna walidacja także nie pokonała persystencji: wybrany średni J128 RMSE 0,237974 wobec 0,211199. Nie należy przedstawiać samego wyboru najlepszego NN jako przewagi nad baseline.

**Werdykt:** wykonano rzeczywisty NNDMD z encoderem, operatorem i decoderem; nie znalazłem blokującego błędu obliczeniowego w zweryfikowanym zakresie. Jedna konkretna rozbieżność identyfikacji kodu została zamknięta dowodem przed testem. Cztery opisane ograniczenia interpretacji pozostają. Finalny model nie realizuje celu predykcyjnego i jest gorszy od persystencji w każdej ocenianej wartości H10..500. Ten ujemny wynik dotyczy jawnie ograniczonej architektury, siatki i przebiegu; nie obala wszystkich metod Koopmana ani nie uzasadnia zmiany modelu po obejrzeniu testu. Nie wykonano nowych treningów, zmian hiperparametrów ani nowych przebiegów testowych w ramach audytu.
