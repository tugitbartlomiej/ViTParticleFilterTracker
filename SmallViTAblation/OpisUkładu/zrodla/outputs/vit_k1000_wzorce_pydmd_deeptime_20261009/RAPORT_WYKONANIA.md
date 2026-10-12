# Rzeczywiste PyDMD/deeptime: wykonany smoke i przygotowanie populacyjnej analizy K1000

Data: 2026-10-09. Wykonawca: **gpt-6-astra, effort xhigh**, zgodnie z metadanymi tej sesji. Projekt: SmallViTAblation. Wszystkie zapisy tego etapu są w rodzinie `vit_k1000_wzorce_pydmd_deeptime_20261009`; producent replay i wizualizacja pozostają własnością innych wykonawców.

## Wniosek na obecnym etapie

**Biblioteki zostały rzeczywiście wykonane**, nie tylko zaimportowane lub wpisane do planu. PyDMD DMDc z kontrolą, deeptime DMD oraz deeptime VAMP przeszły syntetyczne testy i analizę 128 rzeczywistych trajektorii. Job1825145 zakończył się COMPLETED. Job1825150 zakończył się COMPLETED po usunięciu algebraicznej pułapki w cechach VAMP. Żadne parametry DMD/DMDc nie zostały zmienione po zobaczeniu testu.

Wynik jest mieszany i nie usprawiedliwia zapewnienia o skutecznym correctorze. Autonomiczny deeptime DMD poprawia prognozę ruchu względem persistence na odłożonych filmach starego panelu. DMDc z uczoną macierzą sterowania **bardzo mocno zawodzi w fazie fit** na jednym odłożonym filmie, mimo dobrego wyniku walidacyjnego. KnownB daje dobry wynik jako kontrola integracji Adama, lecz korzysta z oracle wejść i nie jest odkrytym rozwiązaniem bez GT.

To jeszcze nie wynik całej populacji. Stary panel128 nie ma ani jednego target tokenu; jedyna porażka oracle znalazła się w train. Przygotowany zakres docelowy ma dwa osobne biegi: panel1024 pełne768 co1 oraz wszystkie900000 cząsteczek w analitycznej PCA32 co5 z niezależnymi deskryptorami pełnych768. Zgłoszenia i statusy znajdują się w [STATUS.md](STATUS.md); wykonanie producenta nie jest tutaj deklarowane przed jego potwierdzeniem.

## Dowody wykonania i odtwarzalność

| Element | Dowód |
|---|---|
| Job1825145, 128 trajektorii, 64000 przejść | [PROVENANCE v1](smoke128/PROVENANCE.json), [wyniki v1](smoke128/WYNIKI.md) |
| Job1825150, wersja bez algebraicznego nakładania cech | [log](smoke_1825150.out), [PROVENANCE v2](smoke128_v2/PROVENANCE.json), [wyniki v2](smoke128_v2/WYNIKI.md) |
| Rzeczywiste wywołania i kondycja operatorów | [MODEL_DIAGNOSTICS](smoke128_v2/MODEL_DIAGNOSTICS.json), [snapshot wykonanego kodu](smoke128_v2/code_snapshot/analiza_temporalna.py), [adapter API](smoke128_v2/code_snapshot/temporal_core.py) |
| Wersje prywatnego środowiska | [ENVIRONMENT_FREEZE](ENVIRONMENT_FREEZE.txt) |
| Hash plików wejściowych i split filmów | [INPUTS](smoke128_v2/INPUTS.json) |
| Parametry baz/skalerów i macierze operatorów | [BASIS](smoke128_v2/BASIS.json), [BASIS_SCALERS](smoke128_v2/BASIS_SCALERS.npz), [OPERATORS](smoke128_v2/OPERATORS.npz) |
| Testy wykonanej wersji smoke | [11 testów PASS](TESTS_SMOKE_V2.txt) |
| Zgodność snapshotów z hashami zapisanego biegu | [AUDIT_SMOKE](smoke128_v2/AUDIT_SMOKE.json) |

Prywatne środowisko Edena: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv`. Nie nadpisuje środowisk replay ani innych agentów. Python3.11; PyDMD2025.8.1, deeptime0.4.5, NumPy2.2.6, SciPy1.15.3, scikit-learn1.6.1. Hash rzeczywiście użytego `pydmd/dmdc.py`: `12d7935cf2192f400f87e8f2dabc6096749463d0851d0ded14434d57ba40dafe`; `deeptime/decomposition/_vamp.py`: `4872c21c506e97d02f43e9490c810954b942f1f70fb039b4128ccd3c635fcd5b`. Pełne hashe własnych skryptów w PROVENANCE; zapisane snapshoty zostały z nimi porównane.

Pierwszy job CPU trwał31s, w tym same obliczenia ok.19.77s; zarejestrowany MaxRSS433368KiB. Drugi trwał22s. Krótkie próbkowanie SLURM nie daje wiarygodnego szczytu RAM drugiego biegu, więc nie wyciągam wniosku z jego małego raportowanego MaxRSS. Podczas analizy surowe768D są czytane po jednym obrazie.

## Co dokładnie dopasowano

1. Całe filmy podzielono deterministycznie: 9train, 3validation, 4test. Lista utrwalona przed fit. To inny split niż w historycznym NumPy DMDc, dlatego wyników obu biegów nie należy odejmować jako kontrolowanej ablacjii biblioteki.
2. Wspólna PCA16 jest dopasowana do przemieszczeń `z_t-z0` z pełnych768D wyłącznie train. Próbka16200 stanów zachowuje97.6498% wariancji swojej próbki; nie jest to gwarancja zachowania energii wszystkich przyrostów ani całej populacji. Błędy prognoz są liczone po podniesieniu ruchu do pełnych768D, a residuum stanu początkowego pozostaje stałe.
3. Skalery stanu i bloków sterowania są dopasowane tylko train. Małe wartości m2 nie są porzucane z powodu arbitralnego progu skali. Modele są osobne dla fit i min_norm: odpowiednio9488 i26512 par treningowych. Nigdy nie zestawia się końca obrazu z początkiem następnego.
4. PyDMD learnedB ma stan16 i51 wejść: projekcje g/m1/m2 po16, learning rate, zegar oraz stała1. Macierz regresji `[X;U]` ma67 kolumn. Są to **teacher-forced oracle controls**: przyszła sekwencja gradientów/momentów jest brana z zapisanego oracle. Stan przewidywany jest propagowany bez resetowania do prawdy w środku okna.
5. PyDMD knownB ma `B=I` w przestrzeni skalowanych przemieszczeń. Wejście powstaje przez analityczny krok Adama i clamp w pełnych768D, a dopiero potem projekcję. Nie stosujemy pierwiastkowania do projected m2. Zgodność tak policzonego kroku z rzeczywistym zapisem: maksymalny błąd bezwzględny9.4998e-7. To kontrola wyrównania danych i integracji, a nie skuteczny predictor dostępny bez GT.
6. deeptime DMD otrzymuje jawne pary i stałą obserwablę do wyrazu wolnego. Nie dostaje gradientów. Rozkład faz jest nadal oracle: główne okna mieszczą się w jednej fazie. Wynik „autonomiczny” nie oznacza jeszcze deployable od naturalnego z0, ponieważ sam wybór fazy i historyczny stan startowy nie są rozwiązanym problemem inference.
7. deeptime VAMP w wersji v2 analizuje przyrost z poprzedniego rzeczywistego kroku, z train-only skalowaniem. Początek trajektorii i pierwsza para przecinająca zmianę fazy nie trafiają do fit VAMP. Następnie kmeans8 w cechach VAMP jest dopasowany tylko train; test jest tylko przypisywany.

Publiczne API PyDMD DMDc przyjmuje macierz snapshotów i lag. Wiele niezależnych par pakujemy jako `[X,Y]` i ustawiamy techniczny lag równy liczbie par; wewnętrzne wycięcie biblioteki daje dokładnie X oraz Y. Fizyczny lag pozostaje1. Macierze A/B są potem używane jawnie; nie korzystamy z `reconstructed_data` ani zegara PyDMD. Test znanego dwuwymiarowego układu z kontrolą sprawdza odzyskanie A i B do1e-9, zarówno learnedB, jak i knownB. Nie zastąpiono biblioteki własnym ridge NumPy. [Dokumentacja i źródło PyDMD](https://pydmd.github.io/PyDMD/_modules/pydmd/dmdc.html), [API deeptime DMD](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.DMD.html).

## Wyniki starego panelu bez targetów

RMSE współrzędnych w pełnych768D na czterech filmach test; h oznacza liczbę rzeczywistych aktualizacji. Maksymalnie80 równomiernie wybranych początków na token/fazę/h; wszystkie metody porównano na tych samych oknach. Dane wszystkich metod i walidacji są w [SUMMARY_METRICS](smoke128_v2/SUMMARY_METRICS.json).

| Faza | h | Persistence | Stały przyrost | Fazowy skurcz | deeptime DMD | PyDMD learnedB | PyDMD knownB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit |1|0.02479245|0.01833339|0.02332853|0.01527846|**1.6482962**|0.01226743|
| fit |5|0.12051527|0.08711076|0.11332967|0.07233281|**8.3091882**|0.05690320|
| fit |20|0.44471269|0.30421115|0.41925997|0.25720975|**35.21277**|0.19237309|
| min_norm |1|0.02034921|0.02007660|0.01619442|0.01591834|0.01799553|0.01058197|
| min_norm |5|0.10107582|0.09964841|0.07277076|0.07181513|0.05502565|0.04899226|
| min_norm |20|0.27218486|0.26864937|**0.13946465**|0.14635785|0.11319682|0.11107804|

Prosty fazowy skurcz jest lepszy od autonomicznego DMD na h20 min_norm. Samo użycie biblioteki lub operatora Koopmana nie daje przewagi nad kompetentnym baseline. LearnedB poprawia persistence na h1 min_norm, ale przegrywa tam z fazowym skurczem i autonomicznym DMD; na h5/h20 poprawia także te dwa wyniki. Nie należy mieszać faz w jednej liczbie i na jej podstawie ogłaszać sukcesu lub porażki „DMD”.

KnownB korzysta z prawie dokładnego analitycznego kroku, obejmującego stan i sterowanie oracle. Jego przewaga jest spodziewana; pozostały błąd jest głównie skutkiem redukcji i niewprowadzania ruchu poza bazą. Nie jest dowodem, że reguła potrafi sama wyznaczyć właściwy ruch od naturalnego obrazu.

### Dlaczego learnedB zawiodło na teście

99.99525% sumy kwadratów błędu test-fit h1 pochodzi z obrazu300, filmu `TR_0029_S1_P03`. RMSE tego filmu wynosi3.0836066; pozostałe trzy filmy dają około0.0131–0.0147. Wejścia obrazu300 w skali train mają max199.7930 i99percentyl96.0106 w fit. Maksymalna wartość wśród train wynosi36.6634. To potwierdzony problem ekstrapolacji wejść. [Rozbicia per film](smoke128_v2/METRICS_BREAKDOWN.json), [OOD kontroli](smoke128_v2/CONTROL_SCALE_DIAGNOSTICS.json), [udział błędu](smoke128_v2/AUDIT_SMOKE.json).

Macierz fit ma rangę67/67 przy progu względnym1e-8 i condition około4971.21; promień spektralny A=1.0067358. Formalna pełna ranga nie zapewnia stabilnej generalizacji poza zakres kontroli. W min_norm ranga wynosi51/67, czyli wejścia i stan nie są niezależnie identyfikowalne we wszystkich kierunkach; zachowana podprzestrzeń ma condition78.38. Regresja obcina małe wartości singular według progu ustalonego przed wynikiem testu. Żadnego nowego progu nie dobrano, aby naprawić obraz300.

Kauzalna interpretacja A/B ma jeszcze osobne ograniczenie: gradient jest wynikiem feedbacku zależnego od stanu, GT i obrazu. Bogaty zbiór takich przejść nie jest automatycznie eksperymentem z niezależnym wymuszeniem. Mechanika Adama jest nieliniowa względem m2 i clamp, z pamięcią i zmianą celu. Pełna ranga liniowego designu nie dowodzi identyfikacji osobnych „sił”. [Pierwotne DMDc](https://arxiv.org/abs/1409.6358), [Adam](https://arxiv.org/abs/1412.6980).

### Kontrola pułapki VAMP

Pierwsza wersja używała cech `[pozycja_t, poprzedni_przyrost_t]`. W kolejnej próbce `pozycja_t = pozycja_t+1 - przyrost_t+1`. Zatem oba końce pary zawierały dokładnie tę samą liniową obserwablę. Otrzymane16 wartości singular≈1 nie były dowodem16 odkrytych wolnych reżimów. V1 została zachowana jako jawna diagnoza.

V2 używa wyłącznie poprzedniego przyrostu. Największe wartości singular wynoszą0.9999335 w fit i0.9984176 w min_norm; nie ma już algebraicznej paczki16 dokładnych jedynek. Nadal nie oznacza to metastabilności ani praw fizycznych: optimizer generuje silnie skorelowany ruch. W pełnych768D średni cos kolejnych aktywnych kroków to0.9972310 w fit i0.9712336 w min_norm; odwrócenia znaku cos występują odpowiednio0% i0.3519%. Próg aktywności to norma obu kroków>1e-4. Wyniki dotyczą wyłącznie badanego panelu i faz; nie opisują targetów ani wszystkich900k. [Zmiany kierunku](smoke128_v2/DIRECTION_CHANGES.json), [VAMP API](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html).

Zapisano także [zajętość](smoke128_v2/REGIME_OCCUPANCY.csv), [przejścia reżimów](smoke128_v2/REGIME_TRANSITIONS.json), [normy i spójność kierunków pełnych768D](smoke128_v2/REGIME_MOTION.json) oraz [okna wyrównane do pierwszego sukcesu](smoke128_v2/SUCCESS_EVENT_ALIGNED.json). Liczby reżimów nie są wspólne między osobno dopasowanymi fazami; reżim0 fit nie oznacza reżimu0 min_norm. Średnie nieaktywnego ruchu nie służą do normalizacji kierunku.

## Docelowe dwa zakresy: panel i populacja

Kontrakt jest zapisany w [KONTRAKT_DANYCH.md](../../experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/KONTRAKT_DANYCH.md), a komunikat dla koordynacji w [KOORDYNACJA.md](KOORDYNACJA.md).

**Panel1024:** 64obrazy×16tokenów, pełne768 i g/m1/m2/LR/faza dla każdej z500 aktualizacji. Adapter korzysta z rzeczywistych `control_*`, role mask i podziału producenta40/12/12filmów. Ma wykonać ten sam prerejestrowany zestaw prognoz1/5/20, z rozbiciem sukces/porażka, target/inne, maską top1/konkurenta i filmem. Raportuje OOD, rangę i stabilność; nie przepisuje parametrów ze starego testu.

**Populacja900k:** rzeczywisty nowy kod Hubble zapisuje `analysis_positions_pca32` o kształcie101×600×32 per obraz, stałą train-only bazę, residual norms stanów/przyrostów oraz normy/dot/cos ruchu policzone przed PCA w768D. Odbiornik używa finalnych manifestów `GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json`, `ANALYTICAL_PCA32_MANIFEST.json`, `REPLAY_INDEX.jsonl` i walidacji producenta. Trzeba potwierdzić kompletność i hashe przed uznaniem tego zakresu za wykonany.

Fit populacyjnego VAMP/kmeans i autonomicznego deeptime DMD używa maksymalnie160 stratyfikowanych obrazów train, do24 interwałów i24 przestrzennych tokenów plus role. Podział64 filmów panelu jest zachowany; nieobecne filmy dostają deterministyczny podział hash60/20/20 przed fit. Wszystkie filmy użyte do PCA producenta muszą być train również u odbiorcy. Następnie przetwarzane są strumieniowo wszystkie1500×600×100 interwałów; zapis etykiet uint8 zajmuje najwyżej90MB przed kompresją. Mieszane interwały faz dostają255 i są wyłączone z głównego fit; mianownik wykluczeń jest jawny.

Regimy populacyjne opisują przyrosty PCA32, a ich zajętość, skala i zmiany kierunku są konfrontowane z deskryptorami pełnych768D. To wspólny układ kierunków, jednak część ruchu poza bazą nadal istnieje. Raport zawiera udział energii przyrostów zachowany przez PCA, osobno dla faz/roli/splitu. Nie przenosi się panelowych reżimów z innej bazy16D i lag1 bez transformacji oraz walidacji.

Autonomiczny DMD populacji prognozuje kolejne przyrosty co5. Porównania: persistence, ostatni przyrost, średni przyrost train i fazowy skurcz prędkości. Dla h5 błąd768 można odtworzyć z błędu w ortonormalnej PCA32 i normy residualnego przyrostu. Dla h20 same normy residuów nie identyfikują ich wzajemnych kątów; raportuje się błąd32 i dolną/górną granicę błędu768 z nierówności trójkąta. Nie podaje się nieznanej dokładnej wartości. Prognoza rzeczywistego h1 pozostaje zweryfikowana tylko panelemco1.

Populacyjne normy gradientu/momentów nie są wektorami sterowania. Bez populacyjnych g/m/v we właściwych chwilach nie deklarujemy populacyjnego DMDc. Mean/last gradient w danym interwale i jego docelowe przemieszczenie są outcome, nie legalnymi przyszłymi wejściami deploy. Panelowy knownB pozostaje oddzielną kontrolą teacher-forced.

## Co to mówi o correctorze i dawnym A/B

Szczegółowe rozliczenie historycznego A/B pozostaje w [wcześniejszym audycie](../vit_k1000_analiza_rezimow_astra_20261009/RAPORT.md). Tam potwierdzono: NumPy DMDc rzeczywiście wykonano, lecz nie zasilało eksportu B; B było statycznym grupowaniem końcowych ruchów i przypisaniem z naturalnych cech. Nowy etap usuwa brak rzeczywistego użycia bibliotek i bada czas, ale sam forecast latentów jeszcze nie poprawia mAP.

Aby otrzymać corrector bez GT, etykieta fazy, target, przyszły gradient i oracle przemieszczenie nie mogą być wymagane przy inference. Można użyć ich jako nadzoru treningowego, a następnie nauczyć z naturalnego z0 i kontekstu obrazu reguły przewidującej reżim, kierunek oraz amplitudę. Należy osobno porównać stałą deltę, regułę warunkowaną obrazem, destylację wzorców temporalnych oraz bezpośrednio optymalizowaną wspólną deltę Hubble, na identycznych filmach odłożonych. To przyszły test detekcji; w tym etapie nie wytrenowano correctora i nie zmieniono modelu.

Najważniejszy pozytywny wynik jest ograniczony, ale konkretny: w panelu istnieje przewidywalna struktura ruchu i różnica między fazami, którą prosty operator potrafi częściowo wykorzystać. Najważniejszy negatywny wynik jest równie konkretny: liniowy learnedB z oracle kontrolami nadal może silnie ekstrapolować i zawieść na nowym filmie. Cel użytkownika jest sensowny jako program badawczy; nie jest potwierdzony przez sam import biblioteki, ładny wykres PCA ani dobry knownB.
