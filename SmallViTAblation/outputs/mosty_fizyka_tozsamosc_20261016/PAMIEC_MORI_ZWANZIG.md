# Pamięć ruchu tokenów: most Mori–Zwanziga, bez importowania dosłownej fizyki

Data: 2026-10-16. Status: audyt źródeł i propozycja maksymalnie trzech testów. Nie uruchomiono modeli, treningów ani zadań Edena. Nie zmieniono Grant ani aplikacji. Nowe testy są propozycją do istniejącego protokołu, a nie uzyskanym wynikiem.

## Najważniejsza odpowiedź

**Tak, model z pamięcią może być przydatny, jeżeli historia ruchu tokenu przewiduje jego następny ruch lepiej niż obecny token, etap treningu, learning rate i odpowiednio silny model bez pamięci.** Nie wiemy jeszcze, czy tak jest w tym ViT. Ewentualny zysk oznacza informację o stanie pominiętym przez mikroskop; nie dowodzi bezwładności koloidu, fizycznej temperatury ani działania światła. Najciekawsza dalsza obserwacja to zanik tego zysku po dodaniu odpowiednich aktualnych zmiennych: kontekstu pozostałych tokenów lub stanu optymalizatora.

Wcześniejszy [raport ruchu a jakości](../wspoldzialanie_neuronow_20261015/ruch_a_jakosc/RAPORT.md) nie znalazł jednoznacznej sygnatury wysokiego IoU i pewności w pojedynczej historii trzech obrazów. Jego `vib` i węzły korzystają z przyszłości: okna ±300 kroków i projekcji całej trajektorii. Nie mogą być wejściami do uczciwego testu przewidywania. Nowe pytanie dotyczy domknięcia dynamiki, a dopiero wtórnie przydatności do przewidywania jakości ramki.

## Co rzeczywiście sprawdzono w lokalnej bazie

Najpierw selektywnie przeszukano `F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/index.md` przez `rg`: Mori, Zwanzig, Langevin, pamięć, closure, nonstationary, SGD. Następnie odczytano wybrane noty z `concepts/`, `sources/`, `MathIndex/math_index.md` oraz `Tezy/lektury/Langevin - Teoria Mori-Zwanziga.md`.

Istotne lokalne tropy:

- `sources/venturi-li-2023-mori-zwanzig-deep-learning.md`: pamięć przy redukcji obserwowanych zmiennych w propagacji po warstwach.
- `sources/meyer-voigtmann-schilling-2017-nonstationary-gle.md`: jądro zależne od dwóch czasów, istotne przy niestacjonarności.
- `sources/arxiv-2101-05873-mori-zwanzig-koopman.md` oraz `sources/arxiv-1908-07725.md`: uczenie operatorów z danych i związek z modelami autoregresyjnymi.
- `sources/mandt-hoffman-blei-2017-sgd-approximate-bayesian-inference.md`: warunkowy most SGD–OU.
- `sources/tepper-dalton-netz-2024-memory-kernel-discretized-data.md`: ograniczenia estymacji pamięci z rzadkich zapisów.

**Nie przenoszę twierdzeń z projektu Grant na ten ViT.** Liczby o TS-JEPA/M5, rzekomym odrzuceniu OU i białym residualu w lokalnej syntezie są wynikami opisanymi w innym projekcie; nie audytowano tutaj ich danych. Nie są dowodem pamięci ruchu tokenów tego modelu. W `concepts/mori-zwanzig-projection.md` zapis z dodatkowym P przed dynamiką ortogonalną oraz końcowe równanie z całym generatorem jako „członem markowskim” nie nadają się do bezpośredniego kopiowania. Bez definicji projektora, generatora i dziedziny można wyzerować rzekomy szum albo podwójnie policzyć dynamikę.

Wykorzystano skill paper-search. Pierwsze zapytanie CLI o Venturiego i Li zwróciło nietrafne prace; odrzucono te trafienia. Weryfikacja kluczowych twierdzeń odbyła się bezpośrednio w źródłach pierwotnych podanych niżej, a nie na podstawie etykiet `confidence` lokalnego wiki.

## Trzy różne osie, trzy różne stany

| Oś | Co jest stałe, co się zmienia | Co ukrywa pojedynczy token |
|---|---|---|
| Krok optymalizatora k | Ten sam obraz i patch; zmieniają się wagi, bufory i optymalizator | Wagi, AdamW m/v, BN, sampler, bieżące i przyszłe minibatch/augmentacje |
| Podkrok sieci l | Jeden checkpoint i obraz; token przechodzi przez attention i MLP | Inne tokeny oraz niewidoczne współrzędne; identyfikator warstwy jest konieczny |
| Czas filmu τ | Zamrożony model; zmienia się obraz sceny | Ruch narzędzia, kamery, zasłonięcie; ten sam patch nie oznacza tego samego fragmentu narzędzia |

Dla pełnego stanu treningu można zapisać `S[k+1]=G_k(S[k], ξ[k])`; `S` obejmuje także stan optymalizatora, bufory i stan losowania danych. Dla ustalonego obrazu `z_i,l[k]=f_i,l(image; θ[k], buffers[k])`. Nawet znając `z_i,l[k]`, zwykle nie znamy całego `S[k]`. Nie wolno utożsamiać residualu prognozy tokenu z szumem minibatch: zawiera również błąd modelu, pominięte zmienne i niestacjonarność.

W obrębie forwardu pełny tensor tokenów i wagi danego bloku wyznaczają kolejny tensor deterministycznie w eval. Sam token przed attention nie wystarcza, ale pełny token przed punktowym MLP wraz z wagami MLP już wyznacza jego aktualizację. Pamięć otrzymana z wcześniejszych bloków może więc być zastępczym opisem pominiętego kontekstu, a nie nowym mechanizmem fizycznym.

Sześć bloków nie jest sześcioma równymi odcinkami fizycznego czasu. Kolejne zdjęcia wideo nie mogą zostać doklejone do osi checkpointów jako dalszy ciąg tej samej trajektorii. Gdy później bada się czas filmu, trzeba osobno ustalić śledzenie obiektu lub pozostać przy jawnej współrzędnej Eulera, czyli stałym patchu obrazu.

## Bezpieczny most matematyczny

Najprostszy dokładny przykład pokazuje, skąd bierze się pamięć. Załóżmy liniowy podział pełnego stanu na obserwowane z oraz ukryte h:

\[
z_{n+1}=Az_n+Bh_n,\qquad h_{n+1}=Cz_n+Dh_n.
\]

Po dokładnym wyeliminowaniu h:

\[
z_{n+1}=Az_n+\sum_{j=0}^{n-1}BD^{n-1-j}Cz_j+BD^nh_0.
\]

To algebraiczna tożsamość dla takiego modelu, nie dopasowany opis ViT. Pamięć powstaje wskutek pominięcia h. Przy operatorach zależnych od kroku zamiast potęg D pojawiają się uporządkowane iloczyny `D[n−1]…D[j+1]`, a jądro zależy od n i j osobno. Stabilność D daje warunki zanikania w tym przykładzie; sam rysunek trajektorii takich warunków nie zapewnia.

W formalizmie MZ rozdziela się ewolucję obserwowalnych na część rozwiązaną, pamięć i dynamikę ortogonalną. Projektor i miara określają ten podział. Liniowa projekcja Moriego i projekcja warunkowa Zwanziga nie są tym samym; wyuczony VAR/NARX nie staje się przez nazwę dokładnie zidentyfikowanym operatorem MZ. [Lin, Tian, Anghel i Livescu, 2021](https://arxiv.org/html/2101.05873v3) uczą część markowską i pamięć z szeregów ograniczonego zestawu obserwowalnych; dowód użyteczności w ich modelu fizycznym nie jest dowodem dla ViT. [Lin i Lu, 2021](https://arxiv.org/abs/1908.07725) wyjaśniają związek redukcji z historią obserwacji i modelami NARMAX.

[Meyer, Voigtmann i Schilling, 2017](https://arxiv.org/html/1706.00658v5) pokazują, że w niestacjonarnym opisie potrzebne jest generalnie `K(t,s)`, a nie wyłącznie `K(t−s)`. Ich relacja podobna do FDT nie nadaje automatycznie termodynamicznej interpretacji dowolnym residualom. Dla treningu z harmonogramem LR nie narzucamy zatem jednego stacjonarnego jądra, dodatniego tarcia ani relacji `szum = temperatura × tarcie`.

[Venturi i Li, 2023](https://link.springer.com/article/10.1007/s40687-023-00390-2) zastosowali już MZ do sieci głębokich i wyprowadzili wystarczające warunki zaniku pamięci po warstwach. To ogranicza nowość: samo nazwanie tokenów układem MZ nie jest odkryciem. Ich warunki kontrakcji nie są automatycznie spełnione przez ten ViT i nie są twierdzeniem o osi treningu.

[Mandt, Hoffman i Blei, 2017, sekcja 3.2](https://jmlr.org/papers/volume18/17-214/17-214.pdf) opierają przybliżenie SGD przez OU na założeniach o szumie gradientu, lokalnej kowariancji, małym błędzie dyskretyzacji i kwadratowym otoczeniu minimum, przy stałym learning rate. Nie wystarcza to do uznania AdamW, całego niestacjonarnego treningu albo tokenów za termiczną dyfuzję. Parametry i reprezentacje to różne zmienne.

## Test M1 — czy historia przewiduje ruch ponad obecny stan i zegar?

**Pytanie główne:** czy pominięta historia niesie replikowalną informację o następnej zmianie tokenu? To pierwszy i najtańszy test; pozostałe uruchamiać dopiero po nim lub razem z już zaplanowanymi pomiarami.

Obserwujemy stały obraz, patch i miejsce sieci. Pierwotny endpoint to prognoza `Δz[k,h]=z[k+h]−z[k]` w residualu 192D; iloraz LN 191D jako osobno zadeklarowana kontrola, nie zamiana podstawowego celu po wyniku. Horyzont startowy h=100 aktualizacji. Historia na siatce co 100 kroków: kandydaci 1, 2, 4, 8 poprzednich obserwacji, wybór wyłącznie discovery. To projekt startowy, nie stwierdzona długość pamięci.

Porównać w tych samych podziałach:

1. Persistence i liniową ekstrapolację z ostatniego przyrostu. Ta druga już korzysta z historii i chroni przed ogłoszeniem zwykłego trendu odkryciem MZ.
2. Silny model bez historii: obecne z, identyfikator warstwy i patcha, k, ekspozycja danych, LR i jawne zmiany harmonogramu. Nieliniowa funkcja aktualnego stanu oraz gładki trend czasu; pojemność i strojenie porównywalne z modelem pamięci.
3. Ten sam model plus opóźnione przyrosty albo opóźnione residuale prognozy. Przykładowa parametryzacja: `Δz_hat = f(z[k], c[k]) + Σ_l K_l(c[k]) r[k−l]`, gdzie `c` jest znanym zegarem/wymuszeniem. Macierze niskiego rzędu i regularizacja, nie swobodna macierz dla każdej chwili. `r` musi być błędem wcześniejszej uczciwej prognozy, obliczonym z danych dostępnych w danej chwili, bez dopasowania modelu na całej przyszłości.

Primary: sparowana różnica MSE prognozy wektorowej, uśredniana najpierw w ustalonej populacji tokenów i obrazów, potem równo po filmach i seedach. Dodatnia różnica `MSE_bez_pamięci−MSE_pamięć` wspiera przydatność historii. Względna poprawa wyłącznie z zamrożonym progiem małego mianownika. Kosinus przyrostów, residual ACF, błędy rolloutu i jakość ramki są secondary; białe residuale nie dowodzą Markowowskości.

**Kontrole ujemne i niestacjonarność:**

- Model samego zegara/LR oraz nieliniowy model bieżącego stanu. Jeśli przewaga historii znika, wcześniejszy wynik był skutkiem zbyt słabego baseline.
- Przypisanie historii innego tokenu/obrazu przy podobnym obecnym stanie i tym samym przedziale treningu. Osobno historia innego seeda. Raportować jako placebo, nie automatycznie poprawny test permutacyjny: wymienność trajektorii nie jest zagwarantowana.
- Null syntetyczny: proces markowski z dryfem i wariancją zmiennymi w czasie, ta sama siatka i pipeline. Kontrola dodatnia: ukryty stabilny liniowy stan z powyższego przykładu. Są to przyszłe małe testy estymatora, bez uruchamiania ich w tym audycie.
- Zmiana kroku zapisu na dostępnych danych i małe gęste okna. Model nie może wskazywać tej samej „stałej pamięci” niezależnie od jednostek. Nie interpolować brakujących checkpointów w głównej analizie.

**Niezależność i walidacja:** s47 i dwa planowane nowe discovery służą wyborowi cech/lagów/regularizacji. Nowe confirmation są odłożone według zamrożonego protokołu. Całe filmy pozostają w jednym split; sąsiednie, nakładające się okna nie trafiają losowo do train/test. Użyć chronologicznego fit→score i przerwy eliminującej współdzielone cele; konserwatywnie długość największego okna plus h. Wynik końcowy ma crossed seed×video CI z zachowaniem całych trajektorii, a nie N=tokeny×checkpointy.

**Problem baz między seedami:** nie stosować jednego wyuczonego regresora surowych współrzędnych seeda47 do innych seedów bez uzasadnionego wyrównania. Najprostszy uczciwy wariant to zamrożony algorytm estymacji dopasowany osobno w każdym biegu na ustalonym wcześniejszym prefiksie i obrazach VAL, a oceniany na późniejszym sufiksie i odłożonych filmach. Reguła prefiksu, cechy, hiperparametry i aktualizacje muszą być zamrożone przed confirmation. To replikacja prequential estymatora, nie całkowicie zamrożony predictor na nieznanych seedach. Alternatywą są z góry zdefiniowane obserwable niewrażliwe na ortogonalny obrót; ta wersja ma inny endpoint i nie zastępuje bez ujawnienia prognozy całego wektora.

**Warunek interpretacji:** wynik dodatni mówi o predykcyjnym niedomknięciu wybranego aktualnego opisu. Nie dowodzi dosłownego jądra siły. Wynik ujemny ogranicza użyteczność danego zakresu historii, próbkowania i klasy estymatora; nie dowodzi braku wszelkiej pamięci.

## Test M2 — czy pamięć pojedynczej cząsteczki zastępuje aktualne otoczenie?

**Pytanie:** jeśli historia pomaga, czy faktycznie niesie informację o pozostałych tokenach? Test wykorzystuje zapis 13 stanów forwardu i attention już przewidziany w kampanii; nie wymaga nowego treningu ViT.

W ustalonym checkpointcie porównać prognozę następnego podkroku: obecny token; obecny token plus jego wcześniejsze stany w tym forwardzie; obecny token plus aktualny kontekst attention; obecny token plus kontekst i historia. Kontekst powinien mieć zamrożoną definicję: np. zapisany rzeczywisty wkład attention z bieżącego pełnego tensora albo mały zestaw jego składowych. Pierwszy wariant stanowi mocną, częściowo konstrukcyjną kontrolę domknięcia, nie konkurencyjny samodzielny predyktor.

Wagi i identyfikator podkroku są warunkiem każdego porównania; nie łączyć różnych bloków tak, jakby obowiązywała ta sama funkcja przejścia. Modele dopasowywać osobno lub jawnie warunkować na operatorze. Film jest jednostką walidacji; 12 przejść w pojedynczym forwardzie to krótki łańcuch, nie szereg do niezależnej estymacji długiego stacjonarnego jądra.

Primary kontrast tego testu: spadek przewagi historii po dodaniu aktualnego kontekstu. Interpretacja: „historia zastępuje pominiętą informację o otoczeniu”, jeśli efekt replikowalnie maleje. Pełny tensor 600×192 i rzeczywisty operator tworzą kontrolę górną: przewidują kolejny tensor bez historii. Punktowe MLP w eval jest kontrolą ujemną względem historii po podaniu pełnego aktualnego tokenu i wag. Historia pomocna tylko przy 3D, a nie przy 192D, wskazuje przede wszystkim na utratę informacji w projekcji.

Kontrole: błędny kontekst dobrany z innego obrazu tego samego checkpointu, losowa projekcja o tej samej liczbie wymiarów, pojemnościowo dopasowane baseline. Donor nie może być wybierany według jakości jego przyszłej predykcji. Zmniejszenie błędu przez dodanie prawdziwego wyniku attention jest oczekiwane konstrukcyjnie; wartością badawczą jest wskazanie, które ograniczone obserwable wystarczają do zastąpienia historii, oraz ich związek z odczytem ramki.

To test w osi głębokości. Nie utożsamia się jego pamięci z pamięcią M1 w osi treningu.

## Test M3 — ile pozornej pamięci tokenu przenosi optymalizator?

**Pytanie:** czy informacja historyczna M1 jest częściowo zastępczym pomiarem momentów AdamW, buforów i sekwencji danych? To gałąź warunkowa, wspólna z T19; nie uzasadnia dodatkowej pełnej kampanii treningowej przed wynikiem M1.

Zaplanować krótkie, gęsto zapisane okna w ustalonych fazach treningu. Punkt startowy: trzy fazy × 100 kroków, do zatwierdzenia po benchmarku i po wyborze testowanej skali pamięci. Dla skali dłuższej niż okno potrzebne są dłuższe okna; nie deklarować identyfikacji jądra długiego na podstawie 100 kroków.

Z tego samego pełnego stanu wykonać sparowane kontynuacje: identyczny replay jako kontrola numeryczna; zmiana kolejności minibatch przy tych samych wagach i momentach; ewentualnie jawna interwencja w m przy niezmienionym v, schedulerze i batchach. Ta ostatnia wytwarza stan sztuczny i mierzy wrażliwość na pamięć optymalizatora, nie naturalny efekt historii. Nie nazywać resetu m/v równoważną zmianą temperatury. Żadnej z tych gałęzi teraz nie uruchomiono.

Zestawić rzeczywisty `Δz` z liniową odpowiedzią `D_θ f · Δθ`, resztą nieliniową oraz oddzielnym wpływem zmiany BN. Dodawać do baseline M1 ustalone bieżące podsumowania optymalizatora. Rzeczywisty przyszły `Δθ` i batch są dozwolone tylko w analizie wyjaśniającej po fakcie, nie jako wejście do prognozy dostępnej przed krokiem. Primary tego testu: replikowalna redukcja błędu domknięcia przez zarejestrowany stan lub wpływ sparowanej interwencji na następny ruch; wybrać jedną wersję przed biegiem.

W pełnym deterministycznym replay identyczne wagi, bufory, optimizer, scheduler, RNG i sampler muszą dać zgodne dalsze wyniki w ustalonej tolerancji. Gdy różni się przeszłość, ale cały aktualny stan i przyszłe wejścia są rzeczywiście identyczne, nie powinien pozostać dodatkowy wpływ przeszłości. Odchylenie wskazuje na niepełny zapis stanu albo niedeterminizm, a nie odkrycie nowej siły.

Powtórzenia gałęzi z jednego stanu są replikami warunkowymi, a nie nowymi niezależnymi treningami. Wnioskowanie o ogólności wymaga powtórzenia na niezależnych seedach, z nadrzędnym klastrowaniem po seedzie i punkcie startowym.

## Co konkretnie zapisać na Edenie

Używać istniejących pakietów analysis/viewer/replay; dodać do manifestu zakres tej gałęzi, nie tworzyć konkurencyjnego systemu eksportu.

| Zapis | Po co i jaki zakres |
|---|---|
| Pełne residuale 192D stałej próbki tokenów | M1: te same obrazy, patche, podkroki, jednostki i gęsta siatka; same xyz nie wystarczą do prognozy pełnego ruchu |
| Pozycje 600 tokenów oraz 13 stanów forwardu | Viewer; M2 na małej wybranej próbce także pełne 600×192 albo dokładnie zdefiniowane bieżące wkłady attention |
| Rzeczywiste step, liczba ekspozycji, LR, scheduler, optimizer type/hyperparameters, BN status | Rozdzielenie historii od niestacjonarnego wymuszenia; jawny zakres przed/po kroku i przed/po forwardzie |
| Wszystkie wymagane checkpointy i bufory | Pochodzenie reprezentacji, odtworzenie dodatkowych ciężkich pomiarów na Edenie |
| Pełny optimizer/RNG/sampler/batch/augmentacja w krótkich oknach M3 | Dokładny replay i sparowane forks; seed liczbowy bez stanu samplera nie wystarcza |
| Manifest `seed/video/patient_if_known/frame/token/block/substep/step`, region GT i transformacje | Parowanie i powtórzenia; region bbox jest etykietą lokalizacji, nie segmentacją |
| Bazy projekcji, centrowanie, skalowanie i zakres danych użytych do fit | Ochrona przed pozornym ruchem z obrotu PCA i przed użyciem przyszłości |
| Prognozy każdego baseline, prawdziwe cele, history masks, status braków, fold i rolling origin | Lokalne CPU odtwarza MSE, CI, wykresy i kontrole bez forwardu GPU |
| Wersjonowane modele domknięcia i pełny config wyboru lagów/ranków | Odtworzenie oszacowania; jawne distinction fit/calibration/scoring dla każdego seeda |

Dla M1 gęsty zapis co 100 kroków jest rozsądnym startem kompatybilnym z dotychczasowym cache, lecz **nie rozstrzyga o pamięci poniżej 100 aktualizacji**. Dodatkowe małe okna co jeden krok są potrzebne do takiego pytania. Dwanaście punktów atlasu nie wystarcza do estymacji bogatego jądra pamięci. [Tepper, Dalton i Netz, 2024](https://arxiv.org/abs/2401.09249) pokazują zależność rekonstrukcji od dyskretyzacji w ich układzie molekularnym; ich metoda i zakresy skuteczności nie przenoszą się automatycznie na niestacjonarny trening.

Dla wielu seedów zachować osobne współrzędne i jawny protokół porównania; nie dopasowywać wspólnego Prokrustesa do całej przyszłości. Cache obserwatora treningu nie może zmieniać RNG, optimizer ani BN. Błędy odczytu, brak checkpointu i zera to różne statusy. Schemat i readback sprawdzić w E0. Nie potrzeba pełnych Jacobianów do podstawowego M1; koszt dominujący to z góry ograniczony zapis reprezentacji i forward sond. Czas i transfer dopiero zmierzyć, nie szacować z parametrów nieznanej partycji Edena.

## Wniosek publikacyjny i bramki

Najbardziej defensywny potencjalny wkład: **ilościowe określenie, kiedy obecne położenie tokenu nie wystarcza do przewidywania jego ruchu oraz które ukryte informacje usuwają tę lukę, z rozdzieleniem osi uczenia i głębokości.** Powiązanie tego wyniku z lokalizacją narzędzia jest dodatkowym testem, nie konsekwencją samego MZ.

Najpierw M1 na historycznych danych wyłącznie jako discovery i test pipeline. Następnie powtórzenie w zaplanowanych nowych treningach, bez tworzenia dodatkowych seedów tylko dla atrakcyjnego wykresu. M2 korzysta z już zaplanowanych stanów forwardu. M3 dopiero po sygnale M1 lub jako mała część istniejącego T19. Liczbę seedów i filmów zachować z głównego protokołu oraz jego analizy precyzji; pojedynczy seed i tysiące tokenów nie wystarczają do wniosku populacyjnego.

Przed confirmation zamrozić: oś czasu, obserwowany stan, target i horyzont, listę lagów, regułę kalibracji nowego seeda, klasy baseline, skalowanie, sposób modelowania zegara, regiony, missingness, primary kontrast i agregację. M1 jest jedynym primary tej gałęzi. M2/M3 mechanistyczne secondary; jeśli mają tworzyć równorzędne potwierdzające twierdzenia, trzeba je włączyć do rodziny wielokrotnych testów przed wynikami.

Jeśli historia nie wygrywa z silnym bieżącym stanem i zegarem, nie budować na siłę GLE. Jeśli wygrywa jedynie w 3D, wynikiem jest ograniczenie wizualizacji. Jeśli wygrywa w 192D i efekt znika po dodaniu stanu optymalizatora lub kontekstu, otrzymujemy konkretną identyfikację brakujących informacji w mikroskopie. Jeśli pozostaje, mamy uzasadnienie dla empirycznego modelu pamięci, ale nadal nie dowód termicznej pułapki optycznej.
