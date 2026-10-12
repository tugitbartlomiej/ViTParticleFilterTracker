# Protokół identyfikacji PyKoopman, zapisany przed końcowym testem

Data: 2026-10-02. Protokół dotyczy tylko nowych artefaktów tego katalogu. Źródłowy trening i wcześniejsze modele pozostają bez zmian.

## Stan i model

Stan cząsteczki to 128 oryginalnych kanałów z KANALY_J128.npz, zachowanych w ustalonej kolejności. Nie stosujemy PCA, autoenkodera, obrotu reprezentacji, selekcji dodatkowych kanałów ani redukcji rzędu. Model wspólny dla wszystkich obrazów i tokenów narzędzia ma postać z_next=A*z+b. Raportujemy macierz A o wymiarze128×128, bias128 oraz operator rozszerzony129×129 dla [z,1]. Jeden krok operatora odpowiada10 aktualizacjom treningu ViT.

PyKoopman1.2.1, Identity oraz EDMD(svd_rank=-1,tlsq_rank=0) rzeczywiście wykonują dopasowanie. Wewnętrzna pełna SVD jest algorytmem rozwiązania regresji; końcowy operator rekonstruujemy do oryginalnych kanałów jako U*state_matrix*U.T. Nie interpretujemy wewnętrznych współrzędnych SVD jako nowych kanałów. Niewielkie numeryczne odchylenie ostatniego wiersza od [0,...,0,1] dokumentujemy; prognoza affine zawsze używa jawnej stałej1.

## Pary, wagi i ograniczenia API

Tworzymy tylko pary tego samego tokenu tego samego obrazu pomiędzy sąsiednimi checkpointami. Nie łączymy końca obrazu z początkiem innego i nie uśredniamy osobnych operatorów. Używamy wyłącznie tokenów maski tool z dodatnią coverage. Waga tokenu to coverage/sum(coverage obrazu), waga czasu1/liczba_par, waga obrazu1/liczba_obrazów. Każdy obraz ma zatem jednakową łączną wagę.

API EDMD.fit(x,y,dt) nie ma sample_weight ani ridge. Po jawnym rozszerzeniu obu par do [z,1] mnożymy CAŁE wiersze obu stron przez sqrt(w). Identity nie dodaje wtedy nowej stałej; także istniejąca stała jest skalowana. To dokładnie ważona norma Frobeniusa. Nigdy nie dodajemy kolumny samych jedynek po ważeniu.

Siatka lambda=[0,1e-6,1e-4,1e-2]. Dodatnie lambda to jawne sztuczne pary [sqrt(alpha)*I,0]→[sqrt(alpha)*I,0], dopisane do danych dla EDMD. Dają karę alpha*||A-I||_F², bez kary biasu i bez kary ostatniego wiersza. alpha=lambda*ważona_średnia(||z||²)/128, przy sumie wag danych1. Jest to regularyzacja wokół braku zmiany, wykonana przez regresor pakietu na rozszerzonych parach, nie zastąpienie go własnym solverem. Odrębnie zapisujemy liczbę rzeczywistych i sztucznych par.

## Podział i wybór bez końcowych holdoutów

30fit=indeksy0..7,12..33; cztery końcowe holdouty8..11. Wewnętrzny podział przed obliczeniami: pierwsze24fit do identyfikacji, ostatnie6fit(indeksy28..33) do walidacji. W siatce dopasowanie tylko checkpointy0..300, czyli30przejść obraz-czas na każdy z24obrazów. Dla sześciu obrazów walidacyjnych podajemy wyłącznie origin300 i prognozujemy open-loop do500. Rzeczywiste stany310..500 służą tylko metrykom.

Wybór: najmniejsza średnia po20horyzontach10..200 z ilorazu MSE_model/MSE_brak_zmiany, liczona z równymi wagami sześciu obrazów. Remis rozstrzyga mniejsze lambda. Nie dobieramy modelu według wyników końcowych czterech holdoutów.

Po wyborze refit na wszystkich30fit wyłącznie0..500:50przejść obraz-czas/obraz,1500przejść obraz-czas łącznie. Operator jest zamrożony PRZED odczytem przyszłych stanów czterech holdoutów. Open-loop500..1000 używa tylko stanu500 i operatora. Test czterech holdoutów obejmuje50horyzontów10..500. Wyniki30fit raportujemy oddzielnie jako diagnostykę. Żadna przyszła obserwacja nie poprawia prognozy ani modelu. Dodatkowy operator na0..1000 może być wyłącznie opisowy:100przejść/obraz,3000przejść obraz-czas; nie używamy go w prognozie od500.

## Metryki ustalone z góry

Skumulowany ruch to delta_true=z(origin+h)-z(origin) i delta_pred=forecast(origin+h)-z(origin). Główna miara kierunku: ważona zgodność znaku składowych, pomijając wyłącznie |delta_true|<=1e-8. Zapisujemy masę aktywnych składowych i dokładną definicję zera; przewidywane zero nie zalicza się jako trafienie dla ruchu niezerowego. Dodatkowo próg ruchu1e-3 jako z góry ustalona diagnostyka, bez dobierania progu z testu. Raportujemy bazę większościowego znaku wyuczoną ze skumulowanego ruchu0..500 na30fit.

MSE stanu, MSE ruchu, MSE braku zmiany, skill=1-MSE_model/MSE_brak_zmiany, cosine ruchu per token oraz względny błąd normy ruchu liczymy coverage z jednakową wagą obrazów. Zera cosine mają jawny brak wartości i udział zdefiniowanych tokenów. Wyniki agregowane oraz dla każdego obrazu i horyzontu zapisujemy maszynowo. 85procent zgodności znaku jest celem, nie gwarancją; horyzont kończy się na pierwszym nieprzerwanym przedziale spełniającym cel odH10.

## Kontrole i granice interpretacji

Przed danymi rzeczywistymi testujemy znany affine układ, ważoną regresję versus niezależne ważone least-squares, skalowanie stałej, pełny wymiar, orientację operatora i wielokrokową prognozę. Zapisujemy wersje środowiska, hash skryptu/protokołu/danych, źródła API i wynik pip check.

J128 zostało wybrane na500, więc walidacja300..500 jest retrospektywna względem wyboru kanałów. Wszystkie obrazy były w dataset train ViT; holdouty dotyczą wyłącznie identyfikacji dynamiki i były wcześniej oglądane w starszych badaniach. 22nowefit należą do dokładnej bazy1500, tylko3z8starychfit do tej bazy, reszta pochodzi z train poza nią. Wspólna trajektoria wag, powtarzane tokeny i checkpointy nie stanowią niezależnych replik eksperymentalnych. Nie wnioskujemy o prognozie pomiędzy checkpointami ani o niezależnym potwierdzeniu na nowych treningach ViT.

Źródła pierwotne: [API EDMD v1.2.1](https://raw.githubusercontent.com/dynamicslab/pykoopman/v1.2.1/src/pykoopman/regression/_edmd.py), [oficjalny przykład par X,Y](https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmd_with_rbf.html), [release](https://github.com/dynamicslab/pykoopman/releases/tag/v1.2.1), [metadane wersji](https://pypi.org/project/pykoopman/1.2.1/).
