# Protokół: prognoza wag a prognoza cząsteczek

Data: 2026-10-01. Status: plan przyszłego badania, **niewykonany**. Dotyczy historii treningu na stałych obrazach. Nie zastępuje istniejących planów ani ich wyników.

## Pytanie i zadanie główne

Czy przy tym samym odcięciu czasu i tym samym horyzoncie lepiej przewidywać pełne tokeny bezpośrednio, czy prognozować wagi z buforami BN i wykonać forward sondy?

Dla chwili \(t\) oraz \(H\) aktualizacji celem jest rzeczywisty tensor \(Z_{t+H}\) zapisany przy **niezmienionej bazowej trajektorii uczenia**. Predyktor nie zmienia sieci, nie wybiera nowych paczek i nie otrzymuje przyszłych gradientów. To test prognozy. Późniejsze wstrzykiwanie przewidzianych wag lub sterowanie doborem klatek byłoby odrębnym eksperymentem interwencyjnym.

## Dostępne dane i pierwszy dopuszczalny horyzont

| Artefakt | Rozdzielczość | Co wystarcza sprawdzić |
|---|---:|---|
| Pełne tokeny 12 obrazów, 600 × 768 | co 1 aktualizację, 0..1000 | Bezpośrednia prognoza H=1,3,5,10… |
| Wagi i bufory BN, cały state_dict | co 10 aktualizacji, 0..1000 | Porównanie wag→forward i tokenów na wspólnej siatce H=10,20,50,100 |
| Pełny stan z AdamW | co 100 aktualizacji | Diagnostyka momentów; tylko 11 stanów, za mało do wiarygodnej bogatej identyfikacji dynamiki |

Weryfikacja schematu zapisu: `../../experiments/maly_vit_ruch_czastek_gesta_historia/kod/bieg.py`, linie 103–109 i domyślne interwały 121–122. Przed wykonaniem sprawdzić klucze i znaczniki czasu **rzeczywistych** checkpointów oraz hashe obrazów i preprocesowania. Nie zakładać, że plik nazwany „wagi” zawiera momenty AdamW. Nie interpolować przyszłych momentów do chwili odcięcia.

Duże pliki pozostają na Edenie. Ten plan nie zleca ich pobierania ani nowego zadania klastra. Obliczenia można wykonać później przy danych albo na świadomie wybranym małym eksporcie, po ustaleniu budżetu.

## Ramiona

1. **P0, persystencja:** \(\widehat Z_{t+H}=Z_t\).
2. **P1, dryf:** ostatni przyrost lub średnia przyrostów z wyłącznie wcześniejszego okna; wybór okna na walidacji.
3. **Z-DMD:** regularizowany model przyrostu pełnego tokenu we wspólnej bazie, np. \(\widehat Z_{t+d}=Z_t+A\psi(Z_t)\). Kontrola affine/identity-prior.
4. **Z-delay:** to samo z opóźnieniami/pamięcią; dopiero potem ustalony lifting nieliniowy. Dobór pamięci na walidacji.
5. **W-DMD→Z:** DMD historii wag warstw/grup; osobna jawna prognoza buforów BN; następnie deterministyczny eval forward tych samych sond.
6. **W-drift→Z:** ekstrapolacja parametrów i BN bez Koopmana. Sprawdza, czy zysk W-DMD pochodzi z modelowania dynamiki, a nie z samego forward.
7. **W-DMD + stałe BN:** ablacją pozostawić BN z chwili \(t\). Pozwala zmierzyć błąd wynikający z braku prognozy buforów.
8. **Z + historia stanu/sterowania:** historyczne opisy faktycznie wykonanych paczek, już zmierzone pchnięcia i dostępne podsumowania momentów. Oddzielnie raportować dodatkową informację.
9. **Learned lifting/autoenkoder:** osobny etap po sprawdzeniu powyższych kontroli. Encoder, decoder i macierz ewolucji uczą się wyłącznie na prefiksie. Raportować rekonstrukcję i prognozę osobno. Jedna trajektoria nie uzasadnia skali meta-treningu KMPNN na tysiącach trajektorii.

Momentów nie potrzeba do samego odczytu tokenu z prognozowanych \((\theta,b)\). Są jednak potencjalnie ważne przy modelowaniu ich przyszłej dynamiki. W-DMD jest celowo modelem obserwowanej historii wag, a nie deklaracją odtworzenia pełnego stanu AdamW.

**Jednostka modelu Z musi być jawna.** Podstawowy wariant stosuje wspólną regularizowaną mapę afiniczną przyrostu w 768 wymiarach do każdego tokenu; fit może łączyć pary tokenów o tym samym indeksie w kolejnych chwilach z ośmiu sond rozwojowych. Wspólność mapy jest sprawdzaną hipotezą. Nie wolno zamienić tego po ocenie na osobny model dla każdego tokenu albo jeden model spłaszczonego obrazu. Te dwa warianty mogą być wcześniej zdefiniowanymi ablacjami, ze wskazaniem wymiaru i ograniczenia rangi przez liczbę migawek. W Z-delay wejściem są opóźnienia tej samej cząsteczki. Cztery odłożone sondy mogą dostarczyć własnej historii do wejścia zamrożonego predyktora; dopasowanie jego parametrów do tej historii jest oddzielnym wariantem adaptacji, a nie zerowym transferem.

**Bufory BN wymagają osobnej obsługi.** Prognozować running mean i running variance; wariancja musi pozostać nieujemna, np. przez jawnie zdefiniowaną transformację logarytmiczną z dodatnim progiem dobranym na przeszłości. Licznik num_batches_tracked nie jest ciągłą wagą do dopasowania DMD. Stałe, typy całkowite, kształty tensorów i nazwy kluczy zachować zgodnie z modelem. Każdą projekcję do poprawnego zakresu i nieudany forward raportować; nie odrzucać po cichu niestabilnych prognoz.

## Równość czasu nie oznacza równości informacji

Dwa raporty:

- **Porównanie podstawowe:** wszystkie rodziny dostają historię o tej samej długości mierzonej w aktualizacjach i tej samej siatce co 10. Różnią się przestrzenią obserwacji; trzeba podać wymiary i koszt.
- **Porównanie praktyczne:** każda rodzina korzysta z dostępnych informacji do \(t\), np. tokeny co 1. Poprawa tutaj może wynikać z gęstszej historii. Nie przypisywać jej samemu Koopmanowi lub reprezentacji.

Zaprojektować dodatkową ablację wspólnych historycznych cech optymalizatora, tak aby ich użycie nie było ukrytym przywilejem jednej metody. Nie nazywać historii samych tokenów i pełnej historii parametrów „tym samym wejściem”.

Kompresję parametrów, rangi SVD, normalizację i ewentualną podprzestrzeń tokenów dopasowywać tylko na przeszłości. Model z kompresją nadal ma zwracać pełne 768 współrzędnych do metryki głównej; oddzielnie policzyć dolną granicę błędu rekonstrukcji. Gdy celem staje się tylko podprzestrzeń, nazwać to innym zadaniem.

## Podział czasu i sond

Proponowany **eksploracyjny** podział istniejącego biegu:

- Dane do identyfikacji i wszystkie etykiety treningowe kończą się najpóźniej w kroku 500.
- Walidacja: początki 500 i 600, horyzonty 10,20,50,100; wszystkie cele najpóźniej 700. Zamrozić rodzinę i hiperparametry po tej walidacji.
- Refit może użyć etykiet do 700.
- Ocena: początki 700,800,900, cele nie dalej niż 1000; te same początki dla wszystkich ramion. Dodatkowe gęste początki są tylko diagnostyką zależnych okien.
- Późniejsze aktualizacje parametrów predyktora dopuszczalne wyłącznie w osobnym prequential wariancie: etykieta pary \(r,r+H\) staje się dostępna dopiero, gdy \(r+H\le t\).

Ten pojedynczy bieg i jego raporty były już oglądane. Ten podział **nie odzyskuje** statusu niezależnego testu potwierdzającego. Na H=100 ma tylko trzy niepokrywające się odcinki oceny. Wiarygodny wniosek o poprawie wymaga późniejszych, świeżych ziaren po zamrożeniu protokołu.

Zachować istniejący podział 8 obrazów/filmów do rozwoju i 4 odłożonych (indeksy 2,5,6,10). Rozdzielić dwie kwestie: ekstrapolację po czasie na znanych sondach i transfer do nowego obrazu. Nie traktować 600 tokenów jednego obrazu jak 600 niezależnych biegów.

Między niezależnymi inicjalizacjami osie cech i symetrie parametrów mogą się różnić. Nie uśredniać bezrefleksyjnie współrzędnych wag ani tokenów z różnych ziaren. W podstawowym porównaniu prowadzić modele per bieg i agregować błędy. Każde uczenie wspólnej bazy/alignment między biegami musi korzystać wyłącznie z ich dopuszczalnej przeszłości.

## Szczelność przyczynowa

Przy oznaczeniu \(B_t\) jako paczki prowadzącej ze stanu \(t\) do \(t+1\), predyktor w \(t\) dostaje co najwyżej \(B_{t-1}\), jej historyczny opis i skutek. To dotyczy także gradientów, pchnięć, straty po aktualizacji i nowych statystyk BN.

Obowiązkowe kontrole:

- Prognoza z uciętego prefiksu jest identyczna jak z pełnego pliku po usunięciu dostępu do sufiksu.
- Podmiana całego sufiksu stanów \(t+1,\ldots\) i paczek \(t,\ldots\) nie zmienia prognozy w \(t\).
- Normalizacja, wybrana ranga, encoder, decoder, kierunki tożsamości i próg „dużej zmiany” nie korzystają z oceny przyszłości.
- Nie obliczać gradientu na paczce powodującej prognozowany cel. Nie używać późniejszego rzeczywistego checkpointu podczas rolloutu.
- Nie stosować smoothera wykorzystującego przyszłe pomiary. Filtr/obserwator może korygować stan tylko pomiarami dostępnymi do \(t\); wewnątrz rollout przyszłe pomiary są niedostępne.
- Nie kalibrować BN prognozowanych wag przyszłymi obrazami. Kalibracja tylko na dostępnej przeszłej puli jest osobnym wariantem z odnotowanym kosztem.
- Rzeczywiste przyszłe wagi, bufory i gradienty mogą służyć wyłącznie do **offline diagnostyki błędu**, jawnie oznaczonej oracle.

## Metryki: przyrost jako główny cel

\[
D_{t,H}=Z_{t+H}-Z_t,\qquad
\widehat D_{t,H}=\widehat Z_{t+H}-Z_t.
\]

Raportować:

1. MSE/RMSE pełnych przyrostów oraz umiejętność względem persystencji:
\[
\mathrm{Skill}_H=1-\frac{\sum\|\widehat D-D\|_F^2}{\sum\|D\|_F^2}.
\]
Mianownik musi być liczony na tych samych próbach; przy zerowym ruchu podać przypadek osobno.
2. Cosine kierunku **przyrostu**, błąd jego normy i odsetek zmienionych współrzędnych. Cosine samych końcowych \(Z\) może być wysoki nawet dla nieużytecznej persystencji.
3. Zgodność znaków i balanced accuracy z uprzednio ustalonym progiem dla praktycznie zerowych zmian; wyniki wszystkich składowych, osobno mniejszych i większych ruchów. Nie wybierać największej połowy według przyszłości jako wyniku głównego.
4. Błędy per obraz, token narzędzia/tła, faza uczenia i horyzont. Uśrednić najpierw w obrębie obrazu/biegu. Zapisywać także rozkłady, aby łatwe tło nie przykryło błędu narzędzia.
5. Dla W: błąd przyrostów wag osobno dla warstw, biasów, normalizacji, oraz błąd BN. Najważniejsza pozostaje wynikowa prognoza \(Z\).
6. Gdy są prognozy probabilistyczne: pokrycie i szerokość przedziałów na odłożonych danych. Gaussowskość, autokorelacja i heteroskedastyczność reszt są odrębnymi cechami.
7. Czas, pamięć i transfery: pozyskanie historii, fit, encoding, SVD, rollout i forward sond. Nie porównywać tylko taniego mnożenia już dopasowanej macierzy z pełnym treningiem.
8. Jeżeli później celem jest korzyść dla detekcji, porównać mAP/przyspieszenie uczenia i dobór klatek w osobnym, równobudżetowym eksperymencie. Trafne wektory nie implikują zysku zadaniowego.

Próg „80 procent” wymaga osobnej definicji. 80 procent poprawnych znaków, 80 procent redukcji MSE i 80 procent pokrycia przedziału nie oznaczają tego samego.

## Diagnostyki rozstrzygające mechanizm

**A. BN.** Na istniejących migawkach wykonać cztery forwardy \(h(\theta_t,b_t)\), \(h(\theta_{t+H},b_t)\), \(h(\theta_t,b_{t+H})\), \(h(\theta_{t+H},b_{t+H})\). Dają dokładne dekompozycje zależne od kolejności oraz interakcję. To oracle po fakcie, nie prognoza.

**B. Przenoszenie błędu wag.** Zestawić błąd prognozowanej aktualizacji wag z błędem wynikowych tokenów. Jeśli to technicznie wykonalne w budżecie, obliczyć JVP na prognozowanym błędzie lub jego próbkowanych kierunkach; nie materializować ogromnego Jacobianu. Sprawdzić, kiedy mały błąd wag jest wzmacniany.

**C. Brakujący stan.** Porównać Z-DMD z Z-delay, a następnie z historycznymi cechami AdamW/BN. Poprawa na odłożonym czasie wspiera hipotezę brakującego stanu, ale nie dowodzi pełnej obserwowalności. Badać też reszty zależne od dostępnej przeszłości.

**D. Wspólna dynamika sond.** W-DMD→Z wymusza, aby wszystkie sondy wynikały z jednego prognozowanego modelu. Niezależne prognozy tokenów mogą być wzajemnie niespójne z jakąkolwiek siecią. Porównać model wspólny/strukturalny i per-token, bez założenia, że wspólność zawsze wygrywa.

**E. Rozkład przyszłości.** Jeżeli powstanie zespół prognoz parametrów, porównać średnią forwardów ze forwardem średnich parametrów. Jest to test nieliniowości przenoszenia niepewności.

## Kryteria wyniku i falsyfikacji

- **Wsparcie dla bezpośredniej prognozy cząsteczek:** zamrożony Z-delay/lifting wygrywa poza dopasowaniem z P0, P1 i porównywalną rodziną bez liftingu; poprawa utrzymuje się w pełnym celu i na świeżych biegach.
- **Wsparcie dla drogi przez parametry:** W→Z przewyższa bezpośrednie prognozy przy uczciwie opisanej informacji/koszcie, także po uwzględnieniu BN.
- **Wsparcie dla hybrydy:** historyczne podsumowania parametrów lub optymalizatora poprawiają Z w porównaniu z tym samym modelem bez nich.
- **Obalenie badanego wariantu:** wynik znika po teście prefiksu, nie bije persystencji/dryfu, zależy od przyszłej paczki, tylko wybranych po fakcie obrazów, rekonstrukcji treningowej lub bardzo krótkiego horyzontu niewystarczającego dla zastosowania.
- **Wynik nierozstrzygający:** przewaga tylko na trzech odcinkach jednego już obejrzanego biegu. Nie zamieniać go w deklarację uniwersalnej skuteczności Koopmana.
- **Oddzielna hipoteza semantyczna:** nawet perfekcyjna prognoza ruchu nie dowodzi tożsamości narzędzia. Potrzeba kontroli innych narzędzi, tła, krawędzi/odbłysków i kontrolowanych zmian obrazu przy stałej sieci.

## Artefakty przyszłego wykonania

Konfiguracja z odcięciami i indeksem źródeł, manifest checkpointów i obrazów, model wybrany na walidacji, prognozy każdego ramienia, wszystkie metryki i niepowodzenia, koszt, test prefiksu, raport faz i horyzontów. Pliki przyszłego biegu mają powstać w nowym katalogu wyników, z zachowaniem tej datowanej wersji protokołu.
