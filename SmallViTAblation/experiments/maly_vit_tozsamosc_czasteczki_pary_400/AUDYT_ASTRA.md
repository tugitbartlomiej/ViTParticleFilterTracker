# Audyt Astra: identyfikacja z 400 par

Data: 2026-10-01. Zakres właścicielski audytora: wyłącznie ten plik; pozostałe dane, kod i wyniki czytane bez zmian. Bez GPU, nowych dopasowań, treningu, dużych plików i commitów.

**Status: audyt techniczny i liczbowy zakończony. Znaleziony błąd XYWH/XYXY został poprawiony i wyniki przeliczone. Sprawdzono również ostatnie ramię z normą dopasowaną po fc2 i końcowe PODSUMOWANIE.md. Nie ma otwartego błędu krytycznego w sprawdzonym zakresie. Wniosek dotyczy odczytywalnego kontrastu oryginał–LaMa i wpływu konkretnej interwencji na confidence×IoU, a nie rozstrzygniętej tożsamości semantycznej ani spadku AP50.**

## Co jest badane

Plan operacyjnie identyfikuje obecność anotowanego cystotome w znanym ROI przy porównaniu obrazu oryginalnego i jego edycji LaMa. To nie jest automatycznie identyfikacja kategorii względem innych instrumentów, fizycznego egzemplarza ani czysty efekt fizycznego odsłonięcia tła. Cząsteczka jest pełnym wektorem tokenu 768D; kierunek podprzestrzeni może mieszać wiele wspólnych osi e_j.

400 par pozostaje w źródłowym split=train. Nowy podział po filmach sprawdza generalizację dopasowywanego odczytu/kierunków pomiędzy tymi filmami; nie gwarantuje filmu nieznanego trenowanemu detektorowi. Numery filmów nie są potwierdzonymi identyfikatorami pacjentów.

## Przegląd protokołu przed pomiarem

Przeczytano plan źródłowy `plan_na_przyszle_eksperymenty/2026-10-01_16-46-21_szukanie_podprzestrzeni_tozsamosci_narzedzia_w_latencie.md`, nowy [PLAN](PLAN.md), źródłowy model i przygotowanie kontroli sham.

Poprawne założenia planu:

- Jedno zamrożone k1000, ta sama warstwa po GELU, eval i niezmienny preprocessing pary.
- Te same indeksy oryginalnej maski w obu członach; maska edycji jest odrębna. Pokrycie komórki maską nie jest procentem „czystej informacji o narzędziu” w późnym tokenie.
- Podział wszystkich członów par i klatek filmu razem, przed dopasowaniem bazy, standaryzacji, probe i hiperparametrów.
- Wagi dopasowania: film, obraz, następnie udział maski. Niecentrowany drugi moment różnic zachowuje średni kierunek.
- Pełne tokeny maski zapisane osobno; pooling jest jawnie ograniczony do określonego klasyfikatora.
- Sparowana podmiana latentów jest eksperymentalną interwencją z dostępem do obu obrazów, a nie samodzielnym identyfikatorem działającym bez kontrfaktycznej pary.
- Brak przyszłych wag/gradientów i brak użycia testu do doboru kierunku lub znaku.

## Uwagi wymagające sprawdzenia w kodzie

1. **Norma perturbacji losowej ma być dopasowana per token.** Sama zgodność normy całego obrazu dopuszcza inny rozkład energii pomiędzy tokenami, co może uprzywilejować kierunek trafiający w komórkę odpowiedzialną za środek ramki. Worker potwierdził przyjęcie per-token matching dla trzech ustalonych losowych podprzestrzeni tego samego r. Sprawdzić zerowe mianowniki, liczbę zmienianych tokenów oraz rzeczywiste normy.
2. **Leave-angle-bin-out dotyczy całej uczonej transformacji.** Należy ponownie dopasować bazę i standaryzację na train bez tego binu, nie tylko regresję logistyczną. Ranga dobrana na pierwotnym val obejmującym wszystkie kąty oznacza częściową ekspozycję na kąt; wynik pozostaje diagnostyką, nie ścisłym benchmarkiem całkowicie nieznanej orientacji.
3. **Ważenie metryk i jednostka niepewności to osobne wybory.** Bootstrap po filmach kontroluje grupowanie replikacji, ale nie zamienia automatycznie średniej po wszystkich klatkach w estymator o równych wagach filmów. Raport musi nazwać obie decyzje.
4. **Hook ma rzeczywiście zmieniać obliczenia dalej w sieci.** Oczekiwany tensor to B×600×768. Kontrola noop, brak zmian poza ROI, pełny patching i porównanie wyjścia są potrzebne przed interpretacją efektów.

## Istotna geometria konkretnego detektora

Źródłowy `model.py` tworzy ostatnią MLP w bloku timm i odczyt po jej GELU. Następnie są projekcja fc2 768→192, residual, końcowa LayerNorm i gęsta głowica. Po tej MLP nie ma kolejnej warstwy attention. Głowica przewiduje środek ramki i jej rozmiar, a nie segmentację wszystkich pikseli instrumentu.

W konsekwencji interwencja w tokenie ostatniej MLP zmienia bezpośrednio jego komórkę mapy wyjściowej. Komórka środka bbox lub szczytu detektora może leżeć poza cienką maską narzędzia. Brak efektu interwencji w pozostałych tokenach nie dowodzi wtedy braku kodowania konceptu. Worker przyjął diagnostykę center/peak-in-ROI oraz pełną podmianę jako kontrolę zasięgu.

Projekcja fc2 ma jądro o wymiarze co najmniej 576. Ta sama norma perturbacji w 768D nie gwarantuje tej samej normy sygnału po fc2. Zapis `||W2 δ||` dla ramion pomaga odróżnić geometrię toru obliczeń od semantycznej swoistości. Norm-matched random jest wartościowym punktem odniesienia, ale sam nie usuwa problemu nienaturalnych aktywacji i różnic w funkcjonalnym wzmocnieniu.

## Kontrole sham

Przeczytano [prepare_sham.py](controls_sham/prepare_sham.py). Skrypt przesuwa rzeczywistą maskę edycji bez zmiany pola i bez obcięcia, wyklucza przecięcie z poszerzoną maską cystotome, zachowuje jego piksele i stosuje ten sam mechanizm LaMa/compositing.

Kontrola powinna obejmować **ROI edytowanego tła** oraz oryginalne ROI narzędzia. Sam drugi odczyt sprawdzałby przede wszystkim oddalony wpływ kontekstu, a nie lokalną odpowiedź na artefakt wypełnienia. Plan przewiduje oba.

Pozostałe ograniczenia: inna pozycja i treść edytowanego obszaru; skrypt nie wyklucza wszystkich innych instrumentów; wybór 24 obrazów według kąta nie jest automatycznie ograniczony do testowych filmów probe. Połączenie z podziałem filmu powinno być jawne. Nie należy deklarować czystego tła ani niezależnego heldoutu bez odpowiedniej weryfikacji.

## Granice poziomu dowodu

- Separacja oryginału od LaMa w znanym ROI: dowód odczytywalności kontrastu obrazowego.
- Transfer między odłożonymi filmami i kątami: dowód określonej generalizacji tego kontrastu, z ograniczeniami danych i podziału.
- Selektywny skutek podmiany większy niż odpowiednie kontrole: dowód wpływu danej perturbacji na detektor w badanych warunkach.
- Tożsamość cystotome względem innych instrumentów, brak zależności od edytora i niezmienniczość semantyczna: wymagają dodatkowych kontroli. Nie wynikają automatycznie z poprzednich trzech punktów.

## Implementacja i wyniki: wykonane sprawdzenia

Audytor odczytał kod `dane.py`, `pomiar.py`, `identyfikacja.py`, `interwencje.py`, `model_snapshot.py` i `metryki_source.py`, a następnie niezależnym skryptem CPU sprawdził niewielki zapis `TOKENY_PARY.npz`. Nie wykonywał forwardów ani dopasowania i nie otwierał tensora 22 GB.

- Zapis zawiera dokładnie `(6729, 2, 768)` skończonych liczb float32. Wszystkie indeksy tokenów i pokrycia zgadzają się z oryginalnym `TOKEN_MASKI.npz`; 400 niepustych przedziałów offsets rozdziela obrazy. SHA256 zapisu zgadza się z `POMIAR.json`.
- Podział faktyczny: 69/23/24 filmów i 240/75/85 obrazów train/val/test. Żaden identyfikator filmu nie przechodzi pomiędzy częściami. Oba człony pary są przechowywane obok siebie.
- Ponowne obliczenie coverage-weighted poolingu w float64 daje maksymalną różnicę `5.27e-7` względem zapisu float32. Podprzestrzeń ma kształt `(768, 8)`; maksymalny błąd ortogonalności float32 `3.58e-7`.
- `identyfikacja.py:12–21,53–64` dopasowuje niescentrowany drugi moment delty wyłącznie na train, a rangę/C/próg wybiera na validation. Kod `:115–123` ponownie dopasowuje bazę i probe w diagnostyce leave-angle-bin-out oraz jawnie oznacza wcześniejszą ekspozycję hiperparametrów przez podstawowy val.
- W głównych miarach i bootstrapie obecności zachowano równe wagi filmów. Standaryzacja probe (`identyfikacja.py:30`) jest nieważoną średnią i odchyleniem po obrazach train; wagi filmowe są stosowane w regresji logistycznej. Nie stanowi to przecieku, ale nie należy opisywać każdego etapu transformacji jako ważonego po filmach.
- `POMIAR.json` podaje niezmienne bufory, zerową różnicę powtórzonego forwardu i 800 obrazów. Jest to kontrola wykonawcy, nie niezależny forward audytora. Runtime tego pomiaru to Python3.11.8/PyTorch2.6.0+cu124/timm0.9.16; samo identyczne SHA modelu nie oznacza bitowej zgodności z innym runtime Eden.

Wynik identyfikacji: wybrane r=8, C=1, test AUROC ważone po filmach `0.968594`, balanced accuracy `0.886111`. LDA daje odpowiednio `0.979377` i `0.952778`; ten wynik nie wykazuje przewagi ośmiowymiarowego odczytu nad LDA. Energia pierwszych ośmiu kierunków train wynosi `0.605318`, r50=5, r90=88. To energia różnic odpowiedzi na edycję, nie liczba konceptów.

Kąty `identyfikacja.py:116,120` pochodzą z natywnego `angle_axis_deg`. Dla 19/400 obrazów bin15° różni się od binu `angle_axis_480_deg` po resize. Wnioski kątowe dotyczą obecnie orientacji natywnej anotacji, a nie dokładnego kąta wejścia sieci.

## Wykryte błędy i stan naprawy

**P1 naprawiony: format ramek w interwencjach.** Pierwsza odczytana wersja `interwencje.py:70` przekazywała `bbox_source_480` bezpośrednio do `quality`. Manifest zapisuje `[x,y,width,height]`, zaś `cele`, `iou` i wyliczenie center oczekują `[x0,y0,x1,y1]`. Przykład source_index0: `[236,142,132,24.6667]`. Wykonawca poprawił konwersję w aktualnym `interwencje.py:76–78` i ponownie wykonał interwencje. Błąd unieważniałby metryki detekcji, centra i ranking funkcjonalny; nie dotyczył zapisanych tokenów, bazy i klasyfikacji par. Pierwszy bieg przerwał się przed zapisem wyników na indeksach int16; poprawiono również indeksy hooka na `torch.long` (`:20`).

Zgłoszono także brak jawnego UTF-8 w odczytach JSON (`dane.py`, `pomiar.py`, `interwencje.py`), istotny na lokalnym Windows. Wykonawca poprawił odczyty; odczyt aktualnego kodu potwierdza zmianę. Nie stwierdzono wpływu na numeryczne wyniki identyfikacji.

## Pojedyncze cząsteczki i surowa mapa wyjściowa

`model_snapshot.py:63–65` oraz punktowa końcówka ostatniej MLP uzasadniają podmianę pojedynczej komórki surowej mapy `(20,30,5)` zamiast osobnego pełnego forwardu dla każdego tokenu. Oba wyjścia muszą pochodzić z tego samego obrazu, a interwencja w danym tokenie musi być identyczna. Następnie trzeba ponowić całe `dekoduj`: lokalny max-pooling3×3 i globalne topk (`model_snapshot.py:121–128`) mogą zmienić selekcję innych komórek. Kod `interwencje.py:112–126` robi podmianę przed dekodowaniem. Dwa rzeczywiste hooki (source_index8/token247 oraz source_index80/token253) dały maksymalny błąd mapy równy0.

Ranking normy projekcji delta nie jest rankingiem odpowiedzialności. Ranking zmiany metryki po podmianie pojedynczego tokenu jest bardziej funkcjonalny, ale dotyczy konkretnej sparowanej perturbacji, checkpointu, obrazu i metryki, a nie uniwersalnej konieczności danego tokenu.

## Aktualny zakres kontroli sham

Pierwszy `controls_sham/MANIFEST.json` zawiera 24 kontrole: 18 train, 3 val i 3 test według niezależnego `PODZIAL.json`. Cały zbiór to postfit diagnostyka techniczna; nie jest 24-próbkowym heldoutem. Osobny `controls_sham/heldout/MANIFEST.json` zawiera24 unikalne obrazy z17 testowych filmów. Zbiory są analizowane oddzielnie; nie wolno dwukrotnie liczyć tych samych source_index w łączonej agregacji. Ograniczenie dotyczące możliwych innych instrumentów w przesuniętej masce nadal obowiązuje.

W heldout sham nie zmieniła się żadna decyzja binarna, ale największa bezwzględna zmiana prawdopodobieństwa wyniosła `0.273408`. Średnia `p_original-p_sham` jest ujemna: po obrazach `−0.011880`, po filmach `−0.016458`. Nie wolno przedstawiać zerowej liczby przejść przez próg jako braku reakcji klasyfikatora. W13/24 przypadków rozłączne na poziomie pikseli maski edycji i narzędzia zajmują co najmniej jeden wspólny token16px. Reakcja toolROI nie jest wtedy wyłącznie odległym skutkiem kontekstu.

## Niezależna kontrola głównych interwencji

Audytor ponownie obliczył confidence×IoU z zapisanych surowych map w NumPy, używając konwersji XYWH→XYXY, max-pool3×3 i ponownej selekcji top10. Dla595 obrazów-ramion maksymalna różnica względem zapisu wyniosła `6.04e-7`. Wszystkie ważone średnie loss/heat/box/q/max_confidence z per-image records zgadzają się z `DETEKCJA.json` do `1e-10`. Noop jest bitowo identyczny z bazą. Maksymalna zmiana surowej mapy poza toolROI wynosi0.

GT-center leży w toolROI dla80/85 obrazów, peak bazowego detektora dla60/85. Problem braku zasięgu istnieje dla części przypadków, ale nie dotyczy większości centrów. Dopasowanie losowej normy w768D jest per token (`interwencje.py:27–39`), z maksymalnym błędem `9.54e-7` i bez degeneracji mianownika.

Główne wyniki: q spada z `0.362321` do `0.343028`; bootstrap95% zmiany `[-0.027628,-0.012299]`. AP50 spada z `0.457589` do `0.441056`, ale przedział zmiany `[-0.062456,+0.041131]` obejmuje0. Przedział zmiany straty także obejmuje0. Nie należy ogłaszać pewnego spadku AP50 ani straty.

Przy tej samej średniej normie ukrytej `15.494743` norma po fc2 wynosi dla znalezionej podprzestrzeni `8.566478`, a dla random11/22/33 odpowiednio `4.523630/4.499544/4.720566`. Sam pierwszy test nie rozdziela większego wzmocnienia fc2 od orientacji zmiany. Dlatego wykonano ostatnią osobną kontrolę opisaną poniżej.

## Końcowa kontrola dopasowania normy po fc2

`interwencje.py:29–37` wylicza dla każdego tokenu normę `δ W2ᵀ`, skaluje losową perturbację do normy perturbacji znalezionej podprzestrzeni i zapisuje błąd dopasowania. Bias projekcji się kasuje w różnicy. `kontrola_fc2.py:8–23` używa tych samych danych, macierzy U i trzech nasion losowych, bez dopasowania parametrów. Główne ramiona pozostają zachowane. Potwierdzono zgodność SHA256 U z zapisem przed tą kontrolą.

Wykonano255 dodatkowych forwardów. Maksymalny błąd normy per token po fc2 to `4.77e-7`, bez zerowych mianowników. Średnia norma po fc2 wynosi we wszystkich czterech porównywanych ramionach `8.566478`. Normy ukryte random11/22/33 zwiększają się do `29.431786/29.621183/28.164090`; nie są jednocześnie dopasowane do ukrytej normy identity `15.494743`. Obie rodziny kontroli należy czytać oddzielnie.

Audytor ponownie obliczył q z255 nowych surowych map, ze zmianą względem zapisu najwyżej `5.20e-7`. Surowa mapa poza ROI pozostaje niezmieniona. Wszystkie ważone agregaty z records oraz filmowy bootstrap różnicy odtworzono; kontrola95% jest zgodna do `1e-12`. Różnica kolejności rzutowania współrzędnych GT do float32 między skryptami daje najwyżej `3.06e-5` piksela i nie zmienia żadnego indeksu GT-center; nie ma materialnego wpływu na sprawdzony q.

Wyniki q: identity `0.343028`; fc2random11 `0.356068`, fc2random22 `0.358546`, fc2random33 `0.355252`. Identity minus średnia trzech kontroli to `−0.013594391`, bootstrap95% `[-0.019216225,-0.008421261]`. Zatem większy wpływ wybranej perturbacji na q utrzymuje się po wyrównaniu normy za fc2. To nie jest analogiczny wynik dla AP50: dla tych ramion AP50 wynosi odpowiednio `0.441056` oraz `0.417700/0.452290/0.419422`; nie wolno uogólniać przewagi w q na całą jakość detekcji.

Dopasowanie normy po fc2 usuwa konkretną różnicę amplitudy na wejściu dalszego toru; nie wyrównuje jego orientacji względem residual, LayerNorm i głowy ani naturalności zmienionych aktywacji. Wynik wzmacnia wniosek o zależnym od kierunku wpływie tej interwencji. Nie izoluje sam przez się tożsamości instrumentu od wszystkich cech obrazu i edycji.

## Ocena końcowego opisu

Przeczytano `PODSUMOWANIE.md`: wyniki osi, testowego odczytu, przykładu cząsteczek280/248, interwencji, obu rodzin kontroli losowych oraz sham są zgodne z artefaktami. Tekst poprawnie rozróżnia kontrast oryginał–LaMa, wpływ na q i nierozstrzygnięty AP50; zaznacza brak porównania z innymi instrumentami i train membership źródłowych obrazów. Obejrzano `HEATMAP_008.png`; osobne mapy reakcji i wpływu odpowiadają tym definicjom.

Zakres pozytywnego wniosku: w jednym zamrożonym modelu można odczytać sparowaną obecność narzędzia z reprezentacji znanego ROI oraz wskazać kierunki i konkretne tokeny, których kontrolowana podmiana zmienia confidence×IoU. Nie ma podstaw do deklaracji jednej wyłącznej osi tożsamości, tożsamości względem innych narzędzi, biologicznej homologii neuronów, poprawy detektora lub przewidywania ruchu podczas treningu. Wyniki są eksploracyjne, z 24 testowymi filmami i zależnymi tokenami; bootstrap warunkuje na wybranym modelu i trzech kontrolach, nie obejmuje wszystkich możliwych treningów lub losowych podprzestrzeni.
