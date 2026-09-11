# 26. Swift Sampling: analiza krytyczna i zastosowanie do kuracji CATARACT-LMM

Data: 2026-09-11. Artykuł: **Dahye Kim i in., „Swift Sampling: Selecting Temporal Surprises via Taylor Series”, arXiv:2605.22678v1, 21 maja 2026**. [Rekord autorów w arXiv](https://arxiv.org/abs/2605.22678v1). Stan: preprint; nie przypisuję mu potwierdzonej akceptacji konferencyjnej. Analiza dotyczy dostarczonej wersji v1 i osobno publicznego kodu autorów.

## 1. Wniosek dla naszej pracy

**Swift jest obowiązkowym bliskim punktem odniesienia dla pomysłu „wybieraj klatki, których cech nie da się przewidzieć”.** Ta ogólna idea, zastosowanie Taylora i ranking reszt nie mogą stanowić naszej głównej deklaracji nowości. Artykuł dostarcza argumentu za sprawdzeniem takiego sygnału; nie dowodzi, że wybrane w ten sposób obrazy poprawiają trening detektora narzędzi.

Najbardziej obiecujące pytanie dla nas brzmi: **czy przy tym samym budżecie adnotacji i treningu rozpoznanie rodzaju błędu predykcji oraz ochrona pokrycia danych dają lepszy podzbiór niż sama wielkość zaskoczenia?** Dopiero odpowiedź mierzona na nowych operacjach pozwoli uzasadnić wkład. Nie zakładamy dużej poprawy przed pomiarem.

Praktyczna decyzja: zachować prosty wariant czasowy i wariant pokrycia, dodać jawnie zdefiniowaną adaptację Swift jako konkurenta, a diagnostykę historii traktować jako hipotezę. Pełny formalizm Mori–Zwanziga pozostawić osobnemu kierunkowi z [dokumentu 25](25_DIAGNOSTYKA_LATENTU_I_AKTYWNA_KURACJA.md). Samo nazwanie reszty „pamięcią” nie tworzy tego powiązania.

## 2. Materiał i granice lektury

Podstawą jest **cały tekst główny dostarczonego OCR**, jego tabele i bibliografia, a nie streszczenie wyszukiwarki:

- [OCR v1](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/Ocr/raw/2605.22678v1/auto/2605.22678v1.md).
- [Oryginalny PDF v1](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2605.22678v1.pdf): 14 stron; strony 1–9 zawierają pracę, 10–14 bibliografię. Wizualnie porównano strony 4–9, obejmujące równania 1–7 i tabele 1–8.
- [Repozytorium autorów](https://github.com/kim-dahye/SwiftSampling/tree/14729b73659c14fbc1c0c804d065249a599efe20), commit `14729b73659c14fbc1c0c804d065249a599efe20`: odczytano cały `peak_helper.py`, `main.py`, skrypt uruchomieniowy i licencję oraz część wrappera odpowiedzialną za kodowanie i wybór klatek. Nie uruchamiano repozytorium.
- Nasz kod `pipeline/curation/{data,features,dynamics,selection,run}.py`, z naciskiem na przepływ puli, cechy, przypisanie score i wybór; punkt odniesienia: commit `e1b67e6e`.
- Kontekst: [23 — grant](23_POLACZENIE_Z_GRANTEM_DYNAMIKA_LATENTU.md), [24 — małe detektory i Eden](24_SZYBKIE_ABLACJE_I_OPTYMALIZACJA_EDEN.md), [25 — diagnostyka](25_DIAGNOSTYKA_LATENTU_I_AKTYWNA_KURACJA.md) oraz [mapa recenzji](../notatki/MAPA_UWAG_RECENZENTOW.md).

**W PDF v1 nie ma dodatku**, mimo odsyłaczy do Appendix przy implementacji, adaptacji DySeg, analizie warstw i przykładach niepowodzeń. Brak dodatku w tej wersji ogranicza reprodukcję; nie jest dowodem, że autorzy nie mają tych materiałów. OCR nie zgubił końcowego dodatku obecnego w tym lokalnym PDF.

Książki z bibliografii zweryfikowano bibliograficznie na stronach autorów/wydawcy; nie deklaruję przeczytania ich w całości. Wyprowadzenia poniżej są własnym rachunkiem na podstawie podanych równań. Nie wykonano nowego OCR, testów, treningów ani uruchomienia pipeline’u. Ten dokument aktualizuje ocenę literatury; nie zmienia kodu selektora.

## 3. Co artykuł rzeczywiście bada

Swift wybiera **wejście dla już wytrenowanego modelu Video-LLM podczas inferencji**. Typowy protokół to 128 kandydatów pobranych równomiernie z jednego filmu, następnie 32 wybrane klatki. Dwa badane backbone’y to LLaVA-OneVision i LLaVA-Video. Wyniki obejmują Video-MME, LongVideoBench i MLVU, kompresję tokenów oraz ocenę opisów w TempCompass. „Ponad dziesięć zadań” obejmuje kategorie pytań, a nie kilkanaście niezależnych badań treningu modeli. Źródło: OCR, sekcje 4.1–4.5.

| Element | Swift w pracy | Nasze badanie |
|---|---|---|
| Cel wyboru | Trafna odpowiedź na pytanie o konkretny film | Generalizacja detektora po treningu na podzbiorze |
| Jednostka budżetu | Liczba wejściowych klatek jednego filmu | Unikalne obrazy/adnotacje z wielu operacji |
| Koszt modelu po wyborze | Jedno przetworzenie wybranych klatek i generowanie odpowiedzi | Wiele aktualizacji wag detektora |
| Reprezentacja | Średnie projekcji key wczesnej warstwy enkodera VLM | Obecnie końcowe cechy DINOv2, CLS i max po patchach |
| Adnotacje bbox | Nie są przedmiotem pracy | Podstawowy target i istotny koszt |
| Wynik | Głównie accuracy VQA | COCO AP, recall, błędy lokalizacji i koszt |
| Bezpośredni dowód transferu do CATARACT-LMM | Brak | Wymaga nowych eksperymentów |

Równe K w obu zastosowaniach nie oznacza tego samego problemu statystycznego. Klatka potrzebna do odpowiedzi o pojedynczym zdarzeniu może być nieprzydatna do nauczenia stabilnej lokalizacji. Powolne, podobne obrazy nadal mogą reprezentować ważne położenia małego narzędzia albo dostarczać koniecznych negatywów.

## 4. Rekonstrukcja algorytmu

Niech `f_t` oznacza wektor cech t-tego **kandydata** w regularnej siatce czasowej. Indeks nie musi oznaczać sąsiedniej klatki surowego nagrania. Autorzy aproksymują pochodne wstecznymi różnicami, podstawiają je do obciętego szeregu Taylora i przewidują `f_t` z wcześniejszych obserwacji. Dla kroku jednostkowego:

\[
\widehat f_t^{(N)}=\sum_{m=0}^{N}c_m^{(N)}f_{t-1-m},\qquad
c_m^{(N)}=(-1)^m\sum_{j=m}^{N}\frac{\binom{j}{m}}{j!}.
\]

Współczynniki wynikające z równań 4–5:

| N | Predykcja | Liczba poprzednich kandydatów |
|---|---|---:|
| 0 | `f_(t-1)` | 1 |
| 1 | `2 f_(t-1) − f_(t-2)` | 2 |
| 2 | `(5/2) f_(t-1) − 2 f_(t-2) + (1/2) f_(t-3)` | 3 |
| 3, domyślnie | `(8/3) f_(t-1) − (5/2) f_(t-2) + f_(t-3) − (1/6) f_(t-4)` | 4 |

Zatem rząd 3 potrzebuje **czterech wcześniejszych klatek i ocenianej piątej**. Nie utożsamiać rzędu, liczby lagów oraz całkowitej długości okna. `N=0` daje zwykłą różnicę kolejnych cech; `N=1` daje resztę drugiej różnicy, a nie zwykły frame difference.

W równaniu 6 score wynosi `r_t = ||f_t − f̂_t||₂`. Następnie:

1. Wyszukaj maksima większe od obu bezpośrednich sąsiadów score.
2. Jeżeli maksimów jest co najmniej K, wybierz K o najwyższej wartości.
3. Jeżeli jest ich mniej, uzupełnij wybór najwyższymi wartościami spośród pozostałych klatek.
4. Zwróć klatki w kolejności czasowej.

Predyktor korzysta z krótkiej historii; sam wybór wymaga także porównania z następnym score i globalnego rankingu. To selekcja offline. Nie należy opisywać jej jako gotowego algorytmu strumieniowego bez dodatkowej reguły opóźnienia i zarządzania budżetem.

## 5. Co zmienia kod autorów

Sprawdzenie [konkretnej wersji `peak_helper.py`](https://github.com/kim-dahye/SwiftSampling/blob/14729b73659c14fbc1c0c804d065249a599efe20/swift_sampling/peak_helper.py) ujawniło różnice ważne dla reprodukcji:

| Element | PDF v1 | Publiczna implementacja |
|---|---|---|
| Skala reszty | Norma błędu, równanie 6 | Norma błędu podzielona przez `||f_t||₂ + 1e-8` |
| Początek sekwencji | Brak pełnej specyfikacji | Niższe rzędy dla krótkiej historii; część początkowych wyników nadpisana przewidywaniem z przyszłych klatek |
| Maksima | Dwie ścisłe nierówności | `scipy.signal.find_peaks`; ma również regułę płaskich szczytów |
| T ≤ K | Brak szczegółowej procedury | Zwrócenie wszystkich indeksów |

To odczyt statyczny, nie wynik porównania wyjść programu. Rozbieżność nie oznacza automatycznie złych wyników artykułu. Oznacza konieczność podania, czy reprodukujemy wzory, czy wskazany commit. [Dokumentacja SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) potwierdza obsługę plateau; wywołanie autorów nie ustawia `distance` ani `prominence`.

Dla dostatecznie długiej sekwencji i N=3 początkowa ekstrapolacja z przyszłości używa współczynników `4, −6, 4, −1`. Jest to inna konstrukcja niż wsteczne współczynniki `8/3, −5/2, 1, −1/6`. Pozostałe klatki mają score o odmiennej polityce brzegowej. Przy offline jest to dopuszczalny projekt, lecz nie spełnia interpretacji „każdy score zależy wyłącznie od przeszłości”.

**Kontrakt naszego baseline’u:** `Swift-code/DINO` oznacza adaptację reguł wskazanego commitu do zdefiniowanych cech DINO, z zachowaną polityką początku i wyboru maksimów. `Taylor-abs/coverage` oznacza score z równania 6 we wspólnym selektorze pokrycia. Żaden z tych wariantów nie jest reprodukcją wyników VQA z artykułu. Pełna reprodukcja VQA wymaga też oryginalnego enkodera, preprocessingu i benchmarków.

## 6. Głębsza ocena matematyczna

### 6.1 Rząd Taylora nie jest automatycznie rzędem dokładności predyktora

**Własne wyprowadzenie:** pierwsza pochodna użyta w pracy to przybliżenie wsteczne pierwszego rzędu. Po pomnożeniu przez krok h wnosi błąd rzędu h². Dodanie dalszych składników z podobnie aproksymowanymi pochodnymi nie daje automatycznie błędu `O(h^(N+1))` dla całego predyktora.

Przykład algebraiczny bez uruchamiania eksperymentu: `x(t)=t²`, punkt bieżący `t₀=0`, krok h. Wartości przeszłe to `0, h², 4h², 9h²`. Prawdziwa następna wartość wynosi `h²`. Predyktor N=2 daje `0 − 2h² + 2h² = 0`; predyktor N=3 również daje zero. Reszta wynosi `h²` pomimo idealnie gładkiego ruchu o stałym przyspieszeniu.

To **nie unieważnia selektora**, który może używać takiej reszty jako skutecznej heurystyki. Ogranicza jednak interpretację, że wyższy rząd całkowicie odejmuje przewidywalną dynamikę wielomianową. W pracy należy mówić o błędzie konkretnego dyskretnego predyktora. Dokładna ekstrapolacja wielomianowa z tych samych czterech poprzedników miałaby współczynniki `4, −6, 4, −1`, więc stanowi inny baseline, a nie kosmetyczną poprawkę Swift.

Również zapis nieskończonego szeregu jako równego funkcji wymaga więcej niż samej gładkości. Dla naszego uzasadnienia wystarczy skończone rozwinięcie z resztą i jawną aproksymacją pochodnych. Nie potrzebujemy deklaracji analityczności cech wideo. Kontekst metod różnicowych: [książka LeVeque’a, strona autora](https://faculty.washington.edu/rjl/fdmbook/).

### 6.2 Wysoki score może oznaczać szum

Dla ilustracyjnego modelu jednej współrzędnej `f_t = s_t + ε_t`, gdzie błędy ε są niezależne i mają wariancję σ², sam składnik szumu w reszcie N=3 ma wariancję:

\[
\operatorname{Var}(e_t)=\left[1+(8/3)^2+(5/2)^2+1+(1/6)^2\right]\sigma^2
=\frac{277}{18}\sigma^2\approx15{,}39\sigma^2.
\]

Dla zwykłej różnicy kolejnych klatek wynosi ona `2σ²`. To własny rachunek dla uproszczonego modelu; nie pomiar szumu DINO ani charakterystyka rzeczywistego CATARACT-LMM. Pokazuje, dlaczego zwiększanie rzędu nie musi pomagać. Dzielenie przez normę obserwacji nie usuwa automatycznie tego problemu.

W chirurgii źródłem dużej reszty mogą być odbłysk, zmiana ekspozycji, gwałtowny ruch kamery, kompresja, zasłonięcie lub artefakt oznaczeń ekranowych. Część to istotne trudne przypadki, część nie. Nie odrzucać ich wszystkich progiem ostrości; mierzyć strukturę wybranych przykładów i wpływ na detektor.

### 6.3 „Surprise” jest warunkową interpretacją, nie miarą użyteczności etykiety

Równanie 7 jest zgodne z ujemnym logarytmem gęstości przy **zadanej średniej predykcyjnej, gaussowskim błędzie i wspólnej izotropowej wariancji**. To założenie modelowe, które autorzy sami nazywają idealizacją. Dla cech ciągłych mówimy o gęstości; jej wartość zależy od reprezentacji i skali.

Nie wynika z niego, że:

- duże `r_t` maksymalizuje informację o klasie lub położeniu narzędzia;
- duże `r_t` rozróżnia brak wiedzy modelu od nieusuwalnego szumu;
- dodanie etykiety bbox zmniejszy błąd dynamiki zamrożonego DINO;
- `history_gain` jest estymatorem warunkowej informacji wzajemnej;
- mała reszta dowodzi dostateczności stanu dla detekcji.

Wersja względna z kodu nie dziedziczy bez dodatkowego uzasadnienia równania 7. Przy zmiennej kowariancji gaussowski log-score zawiera również składnik log-wyznacznika. Nie wystarczy zastąpić wariancji normą bieżącego wektora i zignorować pozostałych składników. Gdy wszystkie oceniane wektory mają tę samą normę, dzielenie przez nią nie zmienia rankingu; normalizowanie wektorów **przed** predykcją zmienia jednak samą trajektorię.

Tło pojęciowe: [Cover i Thomas, Elements of Information Theory](https://doi.org/10.1002/047174882X), [Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). To podręczniki definicji i modeli probabilistycznych, nie dowody skuteczności tego selektora.

### 6.4 Lokalne maksima nie zapewniają pokrycia filmu

Po znalezieniu maksimów autorzy nadal sortują je **globalnie według wysokości**. Przykładowa sekwencja score `0,100,0,99,0,98,0,0,1,0` i K=3 prowadzi do wyboru trzech wysokich pików we wczesnym fragmencie. Późny pik zostaje pominięty. To algebraiczny kontrprzykład dla gwarancji pokrycia, nie uruchomiony test.

Maksima ograniczają wybieranie sąsiadów na pojedynczym zboczu, ale nie zapewniają reprezentacji każdego odcinka, fazy, operacji ani stanu narzędzia. Przy niedoborze maksimów uzupełnianie globalnym rankingiem może ponownie wybrać bliskie klatki. Nasze kotwice czasowe i przydziały grupowe mają więc sens jako jawne ograniczenia, lecz ich wartość trzeba wykazać w ablacji.

### 6.5 Czas i geometria cech są częścią metody

Przy jednakowym odstępie h między kandydatami oraz horyzoncie równym h czynniki h znoszą się we współczynnikach predykcji. Nie znaczy to, że wynik jest niezależny od częstotliwości próbkowania: zmiana h daje inne obserwacje i inną skalę zdarzeń. 128 kandydatów z filmu półgodzinnego to zupełnie inna skala niż klatka co pięć indeksów z gęstego klipu chirurgicznego.

Nie stosować tych współczynników do nieregularnej listy po odrzuceniu klatek bez uwzględnienia czasu. Można zachować regularną trajektorię do obliczeń i dopiero potem ograniczyć pulę możliwych adnotacji. Alternatywa z dopasowaniem wielomianu do rzeczywistych timestampów jest odrębną metodą.

Norma reszty nie jest niezmiennicza na dowolne skalowanie kanałów ani nieliniową zmianę reprezentacji. Porównanie surowych średnich reszt pomiędzy key i hidden features nie wystarcza do wykazania lepszej przewidywalności: najpierw trzeba kontrolować skalę. Podobnie max po patchach może przełączać dominujący patch pomiędzy klatkami i tworzyć zmiany deskryptora bez odpowiadającego im nowego zdarzenia. To możliwy mechanizm do zbadania, nie stwierdzony wynik naszego zbioru.

## 7. Wyniki: co jest mocne, a czego nie przenosić do abstraktu

Poniższe liczby odczytano z dostarczonych tabel i porównano z PDF. Różnice obliczono z pokazanych, zaokrąglonych wartości. Są to **wyniki autorów**, nie nasze wyniki.

| Warunek | Uniform | Swift | Różnica accuracy, punkty procentowe |
|---|---:|---:|---:|
| OneVision, Video-MME overall, K=32 | 58,3 | 59,0 | +0,7 |
| OneVision, LVB overall, K=32 | 55,3 | 57,9 | +2,6 |
| OneVision, MLVU overall, K=32 | 64,7 | 65,6 | +0,9 |
| LLaVA-Video, Video-MME overall, K=32 | 61,4 | 61,9 | +0,5 |
| LLaVA-Video, LVB overall, K=32 | 56,8 | 58,6 | +1,8 |
| LLaVA-Video, MLVU overall, K=32 | 64,2 | 67,2 | +3,0 |
| OneVision, MLVU ≥30 minut, K=4 | 45,8 | 58,3 | +12,5 |
| OneVision, MLVU overall, K=4 | 54,4 | 56,7 | +2,3 |

Źródło: tabele 1 i 8b, PDF s. 6 i 9. Nagłówkowe **+12,5 punktu** dotyczy długiego podzbioru MLVU i budżetu czterech klatek, nie całego benchmarku ani AP. Nie można z niego wyprowadzić oczekiwanego wzrostu naszego detektora.

Mocne strony: dwa backbone’y w tym samym zadaniu, kilka zbiorów, małe budżety, prosty score oraz ablacje rzędu, warstwy i agregacji przestrzennej. Wyniki uzasadniają sensowny, stosunkowo tani baseline.

Ograniczenia istotne dla naszej argumentacji:

- **Nie wygrywa wszędzie.** W tabeli 2 MaxInfo ma 66,5 na MLVU overall wobec 65,6 Swift. W części dłuższych podzbiorów Cosine Uniqueness też ma wyższe wyniki. Nie opisywać pracy jako bezwarunkowo najlepszej metody.
- **N=3 nie jest uniwersalnym optimum.** W tabeli 8a N=6 daje MLVU 66,0 wobec 65,6 dla N=3; rząd 3 jest kompromisem. W tabeli 6 siatka 4×4 ma na MLVU 65,8 wobec 65,6 dla global mean. Tego wyboru nie przenosić automatycznie na małe narzędzia.
- **Zadania też tracą.** Tabela 5: Attribute Perception 74,8→71,2; Temporal Perception 63,6→60,0; Counting 37,7→35,4. Object Reasoning pozostaje 55,1. Tabela 4: Attribute Change 38,52→34,44. Selekcja zdarzeń nie zastępuje pokrycia.
- **Opis kompresji zawiera niespójność liczbową.** Sekcja 4.3 nazywa +1,6 przy r=0,15 największym zyskiem, ale tabela 3 podaje 62,1→64,5, czyli +2,4, przy r=0,10. Na ≥30 min przy r=0,15 jest spadek 62,5→56,2, a przy r=0,10 spadek 64,6→62,5. Poprawa overall nie oznacza poprawy każdej podgrupy. To widać również w oryginalnym PDF, nie tylko OCR.
- Drobne delty w tabeli 5 nie zawsze są dokładną różnicą zaokrąglonych komórek, np. 53,7→57,5 opisano jako +3,9. Bez danych źródłowych możliwe jest zaokrąglanie; nie przedstawiać takich rozbieżności jako dowodu nierzetelności.
- W odczytanej v1 brak przedziałów ufności, surowych par odpowiedzi, liczebności każdej podgrupy w tabelach oraz pełnego protokołu odłożenia danych do wyboru N/warstwy. Nie da się z samych tabel zweryfikować istotności małych różnic. Zagnieżdżone podzbiory długości nie są niezależnymi replikacjami.
- Captioning oceniono pośrednio przez odpowiedź modelu oceniającego na pytanie na podstawie opisu; nie jest to bezpośrednia ocena kompletności opisu przez ekspertów. Autorzy wskazują wersję GPT-4o, co pomaga identyfikowalności, ale nie znosi ograniczeń takiej miary.

## 8. Jak poprawnie rozumieć „30×” i koszt na Edenie

W tabelach koszt inferencji jest znormalizowany: uniform 1,00; Cosine Uniqueness 1,60; Swift 1,02. Własny rachunek:

\[
\frac{1{,}60-1}{1{,}02-1}=30,
\qquad
\frac{1{,}60}{1{,}02}\approx1{,}57.
\]

Pierwsza liczba opisuje stosunek **narzutów FLOPs**, druga stosunek **całkowitych FLOPs**. Żadna nie jest zmierzonym przyspieszeniem zegarowym naszego pipeline’u. Również 1,02 to nie wzrost kosztu treningu o dwa procent.

W odczytanej ścieżce wrappera autorów wszystkie T klatek przechodzą wczesne obliczenia, a wybrane K przechodzą następnie pełny ViT. Model wykonuje więc ponownie wczesną część dla K, nie kontynuuje jej z zachowanego hidden state. Własny model kosztu tej ścieżki to w przybliżeniu `T·C_early + K·C_full + C_LLM + C_select`; pamięć i transfery mogą zmienić czas rzeczywisty.

W naszym przypadku DINO jest osobnym ekstraktorem, a trenowany YOLO lub RT-DETR nie ma automatycznie zgodnego enkodera wczesnej warstwy. Raportować:

`czas przygotowania = odczyt/dekodowanie + inicjalizacja enkodera + ekstrakcja cech + fit diagnostyki + wybór + eksport`.

Do tego osobno dodać adnotowanie, trening i ocenę. Pokazać koszt pierwszego użycia oraz koszt ponownego wyboru ze współdzielonego cache. Nie porównywać nowej metody z ciepłego cache z konkurentem liczonym od zera. `features.py` rozpoczyna obecnie swój licznik po konstrukcji enkodera; sama wartość `features.wall_seconds` nie obejmuje całego przygotowania. Potrzebny jest także całkowity czas procesu i jawne etapy.

Najtańszy eksperyment z już gotowymi cechami to policzenie kilku score na tym samym cache. Eksperyment z tanimi cechami wczesnej warstwy jest osobną ablacją **reprezentacji i kosztu**, nie dowodem przewagi samego Taylora.

## 9. Co ta lektura ujawnia w obecnym pipeline’ie

To ocena kodu po refaktorze `e1b67e6e`; nie jest dowodem wykonania ani poprawności programu. Trzy profile są prototypem. `simple` zawiera preferencję ostrości wewnątrz części przedziałów, więc **nie jest czystym uniform baseline’em**. `balanced` łączy kotwice i pokrycie; `dynamics` moduluje pokrycie diagnostyką. Żaden nie implementuje obecnie oryginalnego Swift.

| Priorytet | Obserwacja w kodzie | Znaczenie i zalecenie |
|---|---|---|
| P0 | `dynamics.py`: `anchor = windows[:, 0]`, target z ostatniej kolumny, zapis błędu pod indeksem anchor | Model przewiduje `z_(t+1)` z kontekstu kończącego się na t, lecz wynik służy do wyboru obrazu t. Swift ocenia zaobserwowane `z_t`. Dla kuracji zaskakującej klatki przypisywać błąd do targetu; dla wyboru kontekstu zachować osobny, jawny tryb. |
| P0 | `run.py` ogranicza train do `label_status == provided` przed ekstrakcją i tworzeniem okien | Nieznane etykiety tworzą luki w trajektorii. Rozdzielić regularną pulę obserwowaną U od puli obrazów dopuszczonych do eksportu/adnotowania A. Okna liczyć na U, wybór ograniczać do A. |
| P0 | Ten sam `features.py` dostarcza CLS+max patch do pokrycia i dynamiki | Wybór reprezentacji wymaga jawnej ablacji. Zarejestrować typ/warstwę/pooling/normalizację; rozważyć osobny deskryptor ruchu i semantyki. Nie nazywać obecnego deskryptora średnią key z wczesnej warstwy. |
| P0 | Brak implementacji Taylor i wyboru maksimów w nowych profilach | Potrzebne osobne, nazwane warianty konkurencyjne przed deklaracją przewagi nad Swift. Nie podmieniać po cichu `dynamics` na Taylor. |
| P1 | `history_gain = error(wide_present) − error(history)` | To kontrast błędów modeli o kontrolowanej liczbie wejść, nie estymata jądra pamięci. Sama równa liczba cech nie wyrównuje elastyczności różnych map ani jakości ich strojenia. |
| P1 | Rankingi dynamiczne są liczone globalnie; standaryzacja targetu zależy od foldu | Jednostki błędu mogą różnić się między foldami i grupami. Przeanalizować kalibrację score na danych treningowych spoza ocenianej grupy oraz wpływ rankingów lokalnych. Nie stroić jej na teście. |
| P1 | Okna nieobsługiwane dostają brak score; później brak priorytetu | Klatki brzegowe i fragmenty z lukami mogą być systematycznie mniej promowane. Raportować ważność okna i odsetek takich klatek w puli oraz wyborze. |
| P1 | `selection.py` mierzy pokrycie w zrzutowanej przestrzeni, którą selektor optymalizuje | Lepsze pokrycie tego samego embeddingu jest oczekiwane. Dodać diagnostykę na innym zamrożonym deskryptorze i właściwe AP; inaczej ocena faworyzuje własny cel. |
| P1 | Czas cech nie zawiera inicjalizacji enkodera | Rejestrować koszt zimnego startu, odczytu, GPU i eksportu oraz koszt ciepłego cache. Nie opierać porównania kosztów na jednym częściowym liczniku. |

Odnośniki do sprawdzonej implementacji: [dynamics.py](../../pipeline/curation/dynamics.py), [run.py](../../pipeline/curation/run.py), [features.py](../../pipeline/curation/features.py), [selection.py](../../pipeline/curation/selection.py).

Przypisanie score do t w obecnym kodzie może być celowe, jeżeli pytamy o informatywny **kontekst przed zdarzeniem**. Problemem jest jego niezadeklarowane utożsamienie ze score klatki zdarzenia. Offline można używać przyszłych obrazów z train; nie jest to samo w sobie przeciek do testu. Trzeba jednak zachować tę samą definicję dla wszystkich porównywanych score i osobno zadeklarować ewentualny tryb online.

Dla `history_gain` pojedynczy dodatni wynik nie dowodzi lokalnej pamięci: oba modele mogą mylić się losowo, a różnica strat ma dużą wariancję. Najpierw oceniać kontrast na odłożonych **grupach treningowych** i jego powtarzalność; dopiero potem badać, czy użycie go jako rankingu pomaga detektorowi. Nie wykorzystywać finalnych val/test do uczenia predyktora, PCA czy kalibracji score.

## 10. Adaptacja do CATARACT-LMM

### 10.1 Gęste klipy `3_Object_Tracking`

To najbardziej naturalny materiał do pierwszego badania dynamiki. Zachować kolejność `frame_number` oraz granice klipów. Wspólną pulę wejściową ustalić przed selekcją. Jeśli zbieramy cechy co pięć klatek, score dotyczy tej siatki, a nie każdej surowej klatki.

W tym podzbiorze P03 jest kodem fazy Capsulorhexis, **nie identyfikatorem pacjenta**. Lokalny audyt grupuje powiązane klipy w konserwatywne grupy operacji; nie potwierdza tożsamości pacjentów. Zapisać grupowanie i duplikaty w manifeście. Wnioski określać jako generalizację między grupami operacji, dopóki identyfikatory pacjentów nie są zweryfikowane. Szczegóły wcześniejszego audytu: [dokument 17](../17_PODZIAL_PER_PACJENT_CATARACT_LMM.md).

Rozdzielić dwie sytuacje:

- **Retrospektywna kuracja istniejących etykiet:** wybieramy tylko obrazy z użyteczną, zweryfikowaną adnotacją; ograniczenie dostępności jest wspólne dla wszystkich selektorów i ujawnione.
- **Akwizycja nowych adnotacji:** scoring może obejmować również obrazy bez znanego bbox; po wyborze wymagają oznaczenia. Ich brak bbox w CSV nie oznacza automatycznie poprawnego negatywu.

Proponowana separacja U/A pozwala korzystać z nieoznaczonych obrazów treningowych jako kontekstu ruchu, bez eksportowania ich jako fałszywych negatywów. Domyślne `provided` w obecnym importerze jest ostrożne wobec etykiet, ale może ograniczać badanie do nietypowo dodatniej puli. Opisać liczbę wykluczonych obrazów i wpływ ograniczenia na wykrywanie wejścia/wyjścia narzędzia.

### 10.2 Rzadkie obrazy `2_Instance_Segmentation`

Warto użyć ich do detekcji wielu narzędzi i badania pokrycia. Nie wolno jednak sklejać ich w sztuczny regularny film. Do Swift lub diagnostyki historii potrzebujemy odpowiedniego otoczenia z nagrania źródłowego i wiarygodnej mapy czasu. Bez tego podstawowym wariantem pozostaje selekcja obrazowa.

Nie łączyć automatycznie tego podzbioru z trackingiem: identyfikatory nie zapewniają kompletnej mapy wspólnych operacji. Cross-subset transfer nie jest wiarygodnym niezależnym testem, dopóki nakładanie operacji nie zostanie wykluczone. Zachować surowe identyfikatory SE/global ID, pochodzenie bbox oraz manifest mapowania klas.

### 10.3 Obiekt, jakość i cel uczenia

Wstępny eksperyment powinien rozdzielać pełne obrazy, stałą siatkę patchy oraz ewentualną reprezentację obiektu. Siatka 4×4 czy ROI to **propozycje ablacji**, nie wnioski z chirurgicznych wyników Swift. Zbyt mocne uśrednianie może zgubić mały instrument, a pooling lokalny może zwiększać wrażliwość na odbłyski.

ROI z prawdziwego bbox/maski ujawnia etykietę, jeżeli celem jest wybór obrazów jeszcze nieoznaczonych. Taki wariant wolno pokazać jako oracle albo kurację już oznaczonego zbioru, ze wskazaniem kosztu. Uczciwy wariant akwizycyjny może korzystać z zamrożonego modelu pomocniczego uczonego wyłącznie na dozwolonym train; jego trening i błędy muszą wejść do rachunku. Samo dołożenie SAM nie usuwa problemu uwagi skupionej na niewłaściwym obiekcie.

BBox narzędzia i punkt końcówki to różne targety. Nowy eksporter detekcyjny nie tworzy ground truth końcówki z centrum bbox. Jeżeli publikacja nadal ma mówić o końcówkach, potrzebna jest osobna adnotacja i model/metryka punktowa. Przewaga w AP bbox nie potwierdza poprawy lokalizacji końcówki.

## 11. Trzy poziomy ulepszenia

**Poziom prosty — kontrolowany wybór czasowy.** Osobny czysty uniform, losowanie unikalnych źródeł i istniejący `simple` jako wariant z jakością. Wspólne grupy, K i dostępność adnotacji. Ten poziom jest potrzebny nawet wtedy, gdy score Taylora wygląda atrakcyjnie wizualnie.

**Poziom średni — dwa sygnały i kontrola pokrycia.** Jeden zamrożony cache DINO do pierwszej ablacji. Policzyć persistence, Taylor oraz ridge na tej samej trajektorii, wyrównać targety czasowe. Porównać oryginalną regułę maksimów z naszymi kotwicami i pokryciem. Rozdzielić `representation`, `score`, `allocation`, `selection_rule`, `boundary_policy` w manifeście; nazwa profilu sama nie wystarcza do reprodukcji.

**Poziom badawczy — diagnoza przed wyborem.** Porównać predyktor z bieżącego stanu, mocniejszy predyktor bieżącego stanu i predyktor z historią. Wykorzystać kontrasty do wyboru danych, a w osobnym badaniu do decyzji o dołożeniu kontekstu lub obserwacji. Przewaga musi wystąpić wobec Taylora oraz porównywalnie elastycznego modelu bez historii. Sam wzrost zdolności predykcyjnej po dodaniu wejść nie identyfikuje przyczyny.

Nie dodawać od razu całej kombinacji DINO/Fourier/SAM/MZ/JEPA. Bez rozdzielenia czynników trudno wyjaśnić recenzentom, co poprawiło wynik i za jaki koszt. Na ten moment najlepszy kandydat na wkład to **sprawdzona zasada doboru danych z kontrolą nieistotnych zaskoczeń i pokrycia**, a nie nowa nazwa złożonego pipeline’u.

## 12. Mały program ablacji rozstrzygający rolę Swift

Poniższy program jest **osobną, skupioną propozycją**, a nie automatycznym dodatkiem do 51 treningów z dokumentu 24. Nie obejmuje pełnej historycznej ablacji DESF; nie zastępuje jej, jeśli utrzymamy pierwotny zakres artykułu. Nie został uruchomiony.

Wspólny punkt startowy: tracking, zamrożony split grupowy, K=2500 tylko jeśli wykonalne po audycie, natywne bbox, YOLOv8n, czyste publiczne wagi początkowe, jednakowa liczba rzeczywistych aktualizacji optymalizatora. Konkretne rozdzielczość/batch/LR przyjąć z kalibracji z dokumentu 24, nie z FPS inferencji w literaturze. Nie używać checkpointu ep170 wybranego w starej procedurze.

| Wiersz | Warunek | Pytanie |
|---|---|---|
| S0 | Random z unikalnych źródeł w ramach wspólnych przydziałów grupowych | Czy metoda przewyższa losowanie przy tym samym składzie grup? |
| S1 | Czysty uniform w czasie, te same przydziały | Czy wygrywa z prostą redukcją korelacji? |
| S2 | Kotwice czasowe + pokrycie DINO | Czy sam score wnosi coś ponad bazowy selektor? |
| S3 | `Swift-code/DINO`: reszta względna i reguła maksimów autorów | Jak wypada bezpośredni konkurent po jawnej adaptacji? |
| S4 | Kotwice + pokrycie + ten sam score Swift co S3 | Ile daje reguła wyboru zamiast samego score? |
| S5 | Kotwice + pokrycie + residual ridge, przypisany do targetu | Czy uczony predyktor daje przewagę nad stałym Taylor? |
| S6 | Kotwice + pokrycie + `history_gain`, przypisany do targetu | Czy kontrast modeli wnosi coś ponad sam błąd? |

S0–S6 używają tej samej puli, cech wejściowych i zewnętrznego podziału K między grupy/klipy. Przy S3 wskazać adaptację: Swift nie definiuje pierwotnie podziału budżetu na wiele operacji. S2/S4/S5/S6 mają identyczne kotwice, odstępy i wagę modulacji; S3 zachowuje regułę maksimów bez dopisania naszego minimalnego odstępu. Różnica S3–S4 dotyczy więc całej reguły wyboru. Efekt samego score najlepiej izoluje S2–S4–S5–S6.

Brzegi i nieregularne okna wymagają wspólnej jawnej polityki. W analizie mechanizmu można porównywać score na wspólnym zbiorze ważnych targetów, przy zachowaniu tej samej obsługi brzegów w selekcji. Dla wierności S3 zachować osobno wynik z oryginalną polityką kodu. Każdy dodatkowy trening takiej czułości doliczyć; nie ukrywać go pod tym samym wierszem.

Etapy i rzeczywisty rachunek:

1. **Pilot:** 7 warunków × jedno ziarno × krótki wspólny budżet = **7 treningów**. Development służy do oceny wykonalności i skali efektu, nie do ogłoszenia zwycięzcy.
2. **Potwierdzenie:** te same 7 warunków × trzy sparowane ziarna × dłuższy budżet = **21 treningów**. Zachować S6 także przy niekorzystnym pilocie. Nie zaliczać krótkich biegów do tej liczby.
3. **Transfer:** S0, S1, S3 i uprzednio zadeklarowana metoda badana, np. S6, × trzy ziarna × RT-DETRv2-S/r18 = **12 treningów**. Te same manifesty wybranych obrazów; własna recepta RT-DETR zamrożona przed oceną porównania.

Łącznie **40 treningów docelowych**, w tym siedem krótkich. Kalibracja, ewentualny proxy, kolejne K, pełna pula, kontrola DESF, dodatkowy konkurent obiektowy i sensitivity nie są w tej liczbie. Dodatkowy większy model przy tych samych czterech warunkach i trzech seedach to następne **12**. RT-DETRv2-S sam w sobie jest drugim małym detektorem, nie potwierdzeniem dla wszystkich dużych modeli.

Nie trzeba rozpoczynać od 40 biegów. Najpierw przygotować zidentyfikowane poprawki i manifesty, potem siedem krótkich porównań. Jeżeli wynik nie uzasadnia kosztu, można zakończyć badanie z negatywnym wnioskiem. Nie należy wtedy przedstawiać przerwanego planu jako potwierdzenia skuteczności. Ziarno wyboru i ziarno treningu zapisywać osobno; trzy powtórzenia jednego manifestu mierzą zmienność treningu, a nie losowania danych.

Kontrole score, które można najpierw obejrzeć bez dodatkowego treningu detektora: persistence vs N=1/2/3, absolute vs relative, rozkład norm cech, overlap wybranych klatek, odsetek brzegów, piki w jednym epizodzie, zmiana wyniku po ograniczeniu ekspozycji/znaczników ekranowych. Wizualna atrakcyjność selekcji nie jest kryterium ostatecznego wyboru metody.

## 13. Co mierzyć, żeby odpowiedzieć naukowo

### Wynik docelowy

Główna metryka: **COCO AP@[0.50:0.95]**, z podaną skalą 0–1 albo 0–100. Pomocniczo AP50, AP75, recall/AR przy zadanym limicie detekcji i wyniki klas, jeśli zadanie jest wieloklasowe. AP ma obejmować wszystkie obrazy zbioru oceny i wszystkie właściwe GT, także obrazy bez predykcji. Nie wybierać dogodnego slotu query.

Dla małych obiektów raportować jawne definicje obszaru i rozdzielczość; nie przenosić bez komentarza progów area między różnymi przeskalowaniami obrazu. Punkt końcówki oceniać wyłącznie względem prawdziwej adnotacji punktowej, z błędem w pikselach lub po jawnej normalizacji. Milimetry wymagają kalibracji.

Próg dla precision/recall/F1 wybierać na walidacji i zamrozić. AP liczyć z rankingu przy ujawnionym wstępnym cutoff. Postprocessing YOLO i RT-DETR opisać zgodnie z ich implementacjami; nie dopisywać automatycznie NMS do RT-DETR. Szczegółowy protokół istnieje w [dokumencie 22](../22_AUDYT_LITERATURY_I_PLAN_PUBLIKACJI.md).

### Mechanizm i koszty

| Obszar | Pomiar | Pułapka interpretacyjna |
|---|---|---|
| Redundancja | Exact duplicates, odstępy czasowe, podobieństwo do najbliższej wybranej klatki | Sama odległość cech nie mierzy użyteczności do detekcji |
| Pokrycie | Reprezentowane operacje/klipy, udział zajętych przedziałów czasu, największe pominięte odcinki | Kwoty gwarantują część wyniku konstrukcyjnie; pokazać go jako własność, nie niezależny sukces |
| Informacja o obiekcie | Skala bbox, widoczność, położenie, trudne przypadki, brak narzędzia | Analiza GT po wyborze jest dopuszczalna diagnostycznie; użycie jej do rankingu zmienia założenia akwizycyjne |
| Zaskoczenie nieistotne | Powiązanie score z blur, saturacją, ruchem tła i nakładką czasu | Korelacja nie rozstrzyga, czy przypadek należy usuwać |
| Diagnostyka | Błędy predykcji poza grupą uczącą, kontrasty historii, ważność okien, niepewność po grupach | Średnie błędy różnych skal i foldów nie są automatycznie porównywalne |
| Budżet | Unikalne obrazy, bboxy do sprawdzenia, czas oznaczania i korekty | Równe K nie gwarantuje równego kosztu adnotacji wielu obiektów |
| Obliczenia | Zimny i ciepły cache, GPU-godziny, czas ścienny, szczyt VRAM, optimizer updates | Równe epoki, równe kroki i równy czas odpowiadają na inne pytania |

Dane do oceny końcowej pozostają niekuratorowane według badanego score. Można użyć ustalonej wcześniej siatki oceny, ale wtedy wniosek dotyczy tej siatki. Nie wybierać testowych klatek tylko dlatego, że mają duże zaskoczenie. Wynik na pełnej lub wcześniej ustalonej trajektorii pokaże także pominięte przez selektor spokojne odcinki.

### Niepewność i kryteria decyzji

Raportować rozrzut pomiędzy seedami oddzielnie od niepewności między operacjami. Dla różnicy metod stosować sparowane losowanie całych grup operacji i ponowne liczenie AP z połączonych predykcji/GT danej próby. Średnia AP policzonych osobno dla operacji jest inną miarą niż globalne AP; można ją dodać pomocniczo, z inną nazwą. Nie bootstrapować pojedynczych silnie skorelowanych klatek jako niezależnych przypadków.

Główny kontrast dla hipotezy diagnostyki można ustalić jako **S6 minus S4** przy stałym K i liczbie kroków. S4 minus S2 bada przydatność samego Taylor score, a S4 minus S3 zmianę reguły wyboru. Porównania z S0/S1 pozostają koniecznym kontekstem praktycznym. Ustalić przed finalnym testem minimalny użyteczny efekt i akceptowany koszt; nie wpisywać arbitralnego p-value jako oczekiwanego wyniku. Przy wielu potwierdzanych kontrastach zadeklarować sposób uwzględnienia wielokrotności.

Trzy seedy są początkiem, nie gwarancją mocy. Jeżeli przedział różnicy obejmuje efekty dodatnie i ujemne o praktycznym znaczeniu, wniosek pozostaje nierozstrzygnięty. Zwiększenie liczby niezależnych operacji może dać więcej niż kolejne treningi na tej samej małej grupie testowej.

## 14. Nowość, Mori–Zwanzig i uwagi recenzentów

Swift nie zamyka tematu naszej publikacji, ale zawęża obronę nowości. Sam residual, sam dobór w latencie, dołożenie lokalnych maksimów czy podmiana enkodera na DINO są adaptacjami znanych elementów. Task-sensitive selection jest także wskazanym przez autorów kierunkiem przyszłych prac. Nie należy nazywać dowolnego dodania ROI odkryciem zupełnie nowego problemu.

Potencjalny wkład wymagający dowodów:

1. Rozróżnienie błędu wyjaśnianego przez bogatszą funkcję, historię i zmianę obserwacji oraz użycie tego rozróżnienia w decyzji o danych.
2. Zmierzone ograniczenie wybierania nieistotnych zmian przy zachowaniu trudnych, ważnych przykładów i pokrycia.
3. Przewaga w detekcji przy określonym budżecie etykiet i pełnym koszcie obliczeniowym, z przeniesieniem tych samych podzbiorów między rodzinami detektorów.

To **kandydaci na wkład empiryczny**, nie potwierdzona nowość względem całej literatury. Przed silną deklaracją potrzebne są także konkurenty uwzględniające obiekty i korelację, omówione w dokumencie 22: CSOD/AdaDeDup oraz COWAL i LRTD z jawnym zakresem ich adaptacji. Wygrana tylko z uniform i Swift nie wystarczy do twierdzenia o najlepszej kuracji danych wideo.

Związek z Mori–Zwanzigiem ma sens na poziomie pytania, jak ograniczona obserwacja i pominięte zmienne prowadzą do zależności od historii. Swift nie identyfikuje jądra pamięci, nie estymuje operatora projekcji i nie dowodzi niedomknięcia. Nasz `history_gain` też nie daje takiego dowodu: porównuje konkretne modele przy skończonych danych. Potwierdzenie, że historia poprawia predykcję latentu, nadal nie dowodzi poprawy detekcji ani tego, że potrzebne jest więcej etykiet.

| Uwaga recenzentów | Co wnosi obecna analiza | Jaki dowód pozostaje potrzebny |
|---|---|---|
| R2-m1, R1-14: nowość i alternatywna selekcja | Wskazany bliski konkurent, rozdzielone score i reguła wyboru | Porównanie z silnym selektorem obiektowym/temporalnym i wspólnym budżetem |
| R1-2, R2-1: przeciek między podobnymi klatkami | Utrzymany split przed selekcją, zastrzeżenie operation vs patient | Zweryfikowane manifesty źródeł i niezależności grup |
| R1-4, R2-5/6: metryki i slot query | AP z całego zbioru, osobna metryka końcówki, threshold na val | Predykcje, wersja evaluatora i wyniki dla wszystkich wierszy |
| R1-9, R2-2: stabilność | Sparowane powtórzenia, grupowa niepewność | Surowe wyniki seedów i grup; bez dopisywania niezmierzonych SD |
| R2-7/8: modele i koszt | Najpierw mały YOLO, potem RT-DETR; oddzielony narzut selekcji | Zmierzony czas treningu i identyczny protokół wewnątrz architektury |
| R1-5/7, R2-4: historyczna ablacja | Nowy mechanizm opisany osobno od DESF | Zachowanie albo jawna zmiana zakresu pierwotnej tezy; Swift nie zastępuje ablacji starych bloków |

Bezpieczna teza robocza do poprawionego manuskryptu: „Badamy, czy połączenie pokrycia danych z diagnostyką przewidywalności cech poprawia dobór obrazów do treningu detektorów narzędzi chirurgicznych względem próbkowania czasowego i selekcji na podstawie zaskoczenia, przy kontrolowanym budżecie”. Czas teraźniejszy opisuje pytanie. Zdanie o wykazanej przewadze można napisać dopiero po uzyskaniu wyników.

## 15. Źródła do odtworzenia analizy

| Źródło | Dokładny odnośnik i zakres wykorzystania |
|---|---|
| Swift v1 | [arXiv](https://arxiv.org/abs/2605.22678v1), [PDF](https://arxiv.org/pdf/2605.22678v1). Szczegółowa analiza oparta na lokalnym OCR/PDF z sekcji 2. |
| Strona autorów | [Projekt](https://kim-dahye.github.io/swift-sampling/), [publikacje pierwszej autorki](https://kim-dahye.github.io/publications/). Potwierdzenie tożsamości i oficjalnego repozytorium. |
| Kod wyboru | [peak_helper.py, przypięty commit](https://github.com/kim-dahye/SwiftSampling/blob/14729b73659c14fbc1c0c804d065249a599efe20/swift_sampling/peak_helper.py). Odczyt całości; żadnego wykonania. |
| Koszt i integracja | [wrapper](https://github.com/kim-dahye/SwiftSampling/blob/14729b73659c14fbc1c0c804d065249a599efe20/swift_sampling/llava_arch.py), [skrypt ewaluacji](https://github.com/kim-dahye/SwiftSampling/blob/14729b73659c14fbc1c0c804d065249a599efe20/scripts/eval_swift.sh). Odczyt odpowiedniej ścieżki kodowania/wyboru i parametrów. |
| Maksima | [SciPy find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). Definicja pików i plateau, parametry odstępu. |
| Metody różnicowe | [LeVeque, 2007](https://faculty.washington.edu/rjl/fdmbook/). Strona autora książki; nie pełna lektura książki. |
| Teoria informacji | [Cover i Thomas, wydanie 2, 2006](https://doi.org/10.1002/047174882X). Zweryfikowana pozycja bibliograficzna; nie dowód transferu do bbox. |
| Model probabilistyczny | [Bishop, 2006 — strona autora/wydawcy instytucjonalnego](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), [dostępny PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). Link do dalszej lektury; bez deklaracji pełnego odczytu. |

[Rejestr źródeł, hashy i zakresu pracy](evidence_swift_2026-09-11/zrodla.json) identyfikuje wykorzystane pliki. [Lista do pobieracza](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/swift-sampling-analiza-szczegolowa-i-podstawy-matematyczne-2026-09-11.txt) zawiera dwa nowe adresy: PDF Bishopa i rekord książki Covera/Thomasa. Swift pominięto jako obecny już we wcześniejszej liście; bezpośredni PDF zachowano powyżej i w [wejściowej liście](evidence_swift_2026-09-11/linki_wejsciowe.txt). Nie zastępowano ani nie przenoszono dostarczonego OCR/PDF.
