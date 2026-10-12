# Audyt Astra: Langevin w zamrożonych kierunkach obecności narzędzia

Data: 2026-10-02. Własność audytora obejmuje wyłącznie ten plik. Bez zmian kodu, dodatkowych dopasowań, forwardów GPU, pobierania danych, treningu i commitów.

**Status: FINAL. Audyt matematyki, kodu, zapisanych wyników i końcowej interpretacji zakończony bez błędu blokującego. Wybory validation zapisano przed testem; zgodność SHA i 36 kontroli prefix/poison potwierdzono po teście. Wszystkie forwardy, dopasowania i obliczenia eksperymentalne wykonał worker; audytor sprawdził matematykę, kod, zapisane wyniki i pochodzenie plików. Zakres dowodowy pozostaje eksploracyjny: jedna uprzednio oglądana trajektoria jednego obrazu, prognoza ośmiowymiarowej projekcji.**

## Przestrzeń i to, co rzeczywiście się porusza

Cząsteczka to pełny wektor tokenu z_i(t) w R^768, mierzony po ostatniej GELU MLP dla tego samego obrazu i tego samego indeksu tokenu. Jednostka czasu odpowiada jednej zakończonej aktualizacji wag. Zmiana reprezentacji uwzględnia również zapisany stan BN. Obserwacja w trybie eval nie jest ponownym treningiem BN.

Wspólne makrowektory e_j pozostają osiami zapisanych współrzędnych. Zmieniają się z_ij(t) oraz mikrowektory z_ij(t)e_j. Jeżeli U500 wyznaczymy raz i zamrozimy, także jego osiem mieszanych kierunków pozostaje nieruchomych. Analiza q_i(t)=U500^T z_i(t) mierzy ruch współrzędnych w tej bazie, a nie obrót osi. Osobne pięć osi natywnych to wybór pięciu e_j, nie kolejne pięć składowych SVD.

Podprzestrzeń pochodzi z kontrastu oryginał–LaMa w znanym ROI. Nazwa „kierunki tożsamości” oznacza tutaj kandydatów związanych z tą operacyjną obecnością; wcześniejsze badanie nie rozstrzygnęło swoistości względem innych instrumentów.

Sonda pozostaje wybrana przez root jako index0=2, `cataract_lmm/tracking/TR_0057_S1_P03/00001431`. Poprawna rola źródłowa to `fit`, split=`train`. Początkowe określenie „pierwsza odłożona” było błędne i zostało sprostowane przed ekstrakcją. Nie należy przypisywać użytkownikowi wyboru tego konkretnego indeksu. Odłożenie w nowym badaniu dotyczy przedziału czasu.

## Granica czasowa i brak przyszłości

U1000 i osie wybrane przy końcu trajektorii mogą służyć jedynie retrospektywnemu opisowi ruchu. Nie stanowią uczciwej reprezentacji prognostycznej dla czasu poniżej1000.

Dla głównego badania U500 i wybór pięciu osi mają zależeć wyłącznie od checkpointu500 i240 par z filmów train analizy400. Ranga8 zostaje przyjęta przed nową prognozą; nie jest na nowo wybierana z wyników trajektorii. Baza, kolejność i znaki pozostają stałe.

Cały bieg był wcześniej oglądany, a ranga8 ma historię poprzedniej analizy. Jest to eksploracyjna rekonstrukcja prognozy algorytmu korzystającego z prefiksu. Nie jest to nietknięty test potwierdzający lub formalna prerejestracja. U500 usuwa konkretny przeciek przyszłej macierzy obserwacji U1000, nie usuwa historii wcześniejszego wyboru hipotez przez badaczy.

Ustalony harmonogram:

- Fit: cele501–700; obserwacje i pamięć wyłącznie sprzed odpowiadającego celu.
- Validation: originy700–749, H1/10/50, maksymalny cel799. Wybór L i ridge z jednego z góry określonego agregatu tych horyzontów.
- Refit po wyborze: cele nie późniejsze niż799; parametry zamrożone przed testem.
- Test główny: wszystkie warianty na tych samych originach800–950, H1/2/5/10/20/50. Maksymalny cel1000.
- H100: osobna diagnostyka, originy800–900; nie dopisywać nieistniejącej przyszłości po1000.

PLAN przewiduje również osobny model dt10: cele fit510–700 co10, validation700–740 co10, wybór po H10/50, refit najpóźniej790 oraz test800–950 co10. Dla pierwszego celu510 i L8 potrzebny jest stan420. Porównanie z dt1 musi używać także jego wyników na tych samych originach co10; H1/2/5 nie istnieją w tym modelu. Dziesięć aktualizacji pozostaje dziesięcioma aktualizacjami, a nie „jednym krokiem uczenia”.

Oryginalny stan w originie jest dostępny. Dalsze kroki jednego rolloutu muszą używać wyłącznie własnych przewidywań i zapamiętanej przeszłości. Kolejny origin może korzystać z faktycznie zaobserwowanych wcześniej stanów; nie oznacza to teacher forcing wewnątrz poprzedniego rolloutu.

## Równanie i indeksowanie pamięci

Dla L>=0 analizowane równanie ma postać:

```text
delta x_i(t+1) = b + A x_i(t)
                 + sum[l=0..L-1] K_l delta x_i(t-l) + epsilon_i(t+1)
x_i(t+1) = x_i(t) + delta x_i(t+1)
delta x_i(t) = x_i(t) - x_i(t-1)
```

L=0 jest modelem bez jawnej pamięci przyrostów. Dla celu tau wektor poziomu pochodzi z tau−1, a przyrost o indeksie l z tau−1−l. Dla pierwszego celu501 i L8 potrzebne są stany do492 włącznie. Ta historia była dostępna w chwili500; wolno przeliczyć jej współrzędne w zamrożonej wtedy bazie. Wszystkie L powinny jednak wykorzystywać te same cele501–700, zamiast zyskiwać lub tracić inne próbki treningowe przez przypadkowe przesunięcie indeksów.

Dla L>=1 jest to algebraicznie afiniczny VAR(L+1):

```text
x(t+1) = b + (I+A+K_0)x(t)
         + sum[j=1..L-1] (K_j-K_(j-1))x(t-j)
         - K_(L-1)x(t-L) + epsilon(t+1).
```

Joint ridge ogranicza arbitralność estymacji, ale silnie skorelowane poziomy i przyrosty mogą pozostawiać podział na A i K wrażliwy na karę i skalowanie. Lepszy forecast nie dowodzi sam w sobie identyfikacji fizycznych sił, ścisłej redukcji Mori–Zwanziga ani dokładnej liniowości Koopmana. Jest to skończony model predykcyjny na określonych obserwablach i ich historii.

Przed testem trzeba jasno określić, czy macierze są wspólne dla tokenów, czy dopasowane osobno. Przy wspólnych macierzach należy uzasadnić sposób obsługi różnych średnich położeń tokenów i wag coverage. Tokeny z jednego obrazu nie są niezależnymi powtórzeniami trajektorii.

PLAN:17–31 rozstrzyga tę decyzję: wspólne b,A,K, stały token-specific anchor q_i(500), skala poziomu i przyrostu wyłącznie z fit, jednakowe wagi celów czasowych i coverage tokenów. Regresja jest joint, z niekaranym interceptem i karą lambda razy średnia przekątna ważonego Gram. PLAN:35–37 ustala średni skill po horyzontach jako kryterium walidacji, a nie wybierany po wyniku agregat. Dodatkowy model wybrany dla H1/H10 jest z góry nazwanym wariantem; nie zastępuje po teście głównego wyboru.

Macierze8×8 mieszają współrzędne wewnątrz reprezentacji jednej cząsteczki. Wspólny estymator dla wielu tokenów nie jest automatycznie operatorem ich wzajemnych oddziaływań: w równaniu tokenu i nie ma jawnych stanów innych tokenów j. Współzależność reszt również nie identyfikuje kierunku przyczynowego pomiędzy cząsteczkami.

## Skale, środki i zapis operatorów

Środki, skale, progi małego ruchu i wszelkie reguły normalizacji muszą wynikać z części fit. W dopuszczonym reficie można je ponownie wyznaczyć wyłącznie z prefiksu kończącego się przed testem; należy zapisać obie wersje i konsekwentnie przeliczyć operatory. Nie wolno cicho normalizować całej trajektorii1001 stanów.

Jeżeli x=S^(-1)(q−mu_i), macierze zapisane dla x nie są bezpośrednio macierzami w surowych współrzędnych q. Dla kolumn A_q=S A_x S^(-1), K_q=S K_x S^(-1), zaś po rozwinięciu wyraz wolny w q zależy także od mu_i. Trzeba zachować S, mu_i i konwencję orientacji macierzy. W implementacji wierszowej X@B blok B jest transponowany względem macierzy mnożącej wektor kolumnowy w powyższym równaniu.

Reguła kary ridge i skalowania poziomu względem przyrostów powinna być jawna. Same identyczne liczby alpha nie zapewniają tej samej siły regularyzacji, jeżeli zmienia się normalizacja Gram, liczba cech lub sumaryczna masa wag.

## Metryki, reszta i niepewność

Główny MSE i skill względem persystencji muszą używać rzeczywistych błędów wektorowych, jawnych wag tokenów i tego samego zbioru originów. Kosinus dotyczy dwóch przemieszczeń: x_hat(t+h)−x(t) oraz x(t+h)−x(t), a nie stanów bez odjęcia originu. Dla zerowego przewidywanego ruchu persystencji kosinus jest nieokreślony; należy jawnie podać tę sytuację i liczebność ważnych przypadków. MSE pozostaje liczone także wtedy. Trafność znaku wymaga jawnego progu i mianownika.

ACF i Ljung–Box dotyczą korelacji reszt. Nie rozstrzygają gaussowskości. Diagnostykę należy wykonać na ciągłej sekwencji jednokrokowych reszt, osobno dla tokenów/współrzędnych lub jawnie zdefiniowanego agregatu; nie sklejać tokenów w jedną sztuczną serię ani nie mieszać błędów wielu H. P-wartości po estymacji i selekcji modelu, przy zależnych tokenach i jednym biegu, mają ograniczoną interpretację.

Bez rzeczywistych logów minibatchy, pełnego stanu AdamW i innych ukrytych stanów sieci residual obejmuje także pominięte wymuszenia, błąd modelu i niestacjonarność. Nie jest automatycznie siłą Browna. Jeżeli raportowany będzie współczynnik dyfuzji, trzeba podać konwencję; dla przyrostu sqrt(2D) dW jego kowariancja wynosi2D delta_t. Sama kowariancja błędów predykcji nie ustanawia modelu SDE.

Bootstrap ma losować wspólne bloki originów dla całego wektora wyników, wszystkich tokenów i metod. Blok co najmniej50 ogranicza część zależności wynikającą z nakładania H50, ale151 originów daje bardzo mało efektywnych bloków. CI są opisowe dla tej trajektorii i przyjętej procedury. H100 wymaga odrębnej obsługi dłuższej zależności lub pozostawienia bez CI. Zabronione jest traktowanie tysięcy tokenów jako tysięcy niezależnych eksperymentów.

## Zakres kontroli wykonania

- Odczyt PLAN/CONFIG oraz kodu przed testem, w tym agregatu wyboru L/ridge i dokładnych granic refitu.
- Potwierdzenie pochodzenia U500 i osi, mmap właściwego biegu, dt1 oraz braku domieszki sterowań z innych eksperymentów.
- Indeksowanie joint regresji i autoregresyjnego rolloutu; zgodność zapisanych A/K z równaniem i transformacjami.
- Jawne błędy lub eksplozje prognoz bez usuwania niewygodnych originów.
- Odczyt końcowych liczb, testów zgodności i ograniczeń raportu.

## Dotychczasowa weryfikacja kodu i źródła

`identyfikacja500.py:15–30` używa wyłącznie240 indeksów train do drugiego momentu różnic i orientacji U. `:31` wybiera top5 osi również wyłącznie na train. Pomiar400 par w `:20–28` nie zasila estymatora danymi val/test. Kod ładuje checkpoint500 przez `dane.py:28–35`, z kontrolą kroku, strict state_dict, eval, zamrożeniem parametrów i wyłączeniem TF32. Bufory BN są sprawdzane po pomiarze. Nie wykonywano tu niezależnego forwardu przez audytora.

Diagnostyczny probe500 ma z góry ustalone C=1 i próg0.5. `identyfikacja500.py:35–37` oblicza AUROC/BA bez wag filmowych, czyli po obrazach. Trzeba to nazwać jawnie: wcześniejszy główny raport400 używał miar film-weighted i progu wybranego na val. Same liczby nie są bezpośrednio porównaniem identycznych protokołów.

Odczyt `maly_vit_ruch_czastek_gesta_historia/kod/bieg.py` potwierdza zapis wag z buforami, metadanych, sond i dziennika wysokiego poziomu. Nie ma w tych zapisach jawnego per-step szeregu wykonanych identyfikatorów obrazów/augmentacji/gradientów. Zapisane nasiona i konfiguracja nie są obserwowanym szeregiem sterowań. Nie wolno wstawić w to miejsce dziennika innego biegu.

## Porównywalność retrospektywnego U500 i U1000

Sprawdzono obie definicje estymatora: poprzedni `maly_vit_tozsamosc_czasteczki_pary_400/identyfikacja.py:12–21` oraz nowy `identyfikacja500.py:15–31`. Oba liczą NIEcentrowany drugi moment pełnych różnic tokenów `C=Σ w delta delta^T`, dla tych samych240 obrazów z69 filmów train i tych samych wag film→obraz→coverage. Pierwszy moment służy orientacji znaku kolumn U; nie stanowi oddzielnej dodatkowej macierzy dodawanej do C. SHA manifestu jest identyczne; SHA obecnego PARY.json odpowiada zapisowi poprzedniego POMIAR.json. Kod modelu ma ten sam SHA. Zmienny jest checkpoint.

Mała różnica numeryczna: stara metoda najpierw odejmowała tokeny float32, po czym rzutowała deltę do float64; nowa rzutuje tokeny do float64 przed odejmowaniem. To nie zmienia definicji estymatora. U1000 zapisano w float32; QR w porównaniu poprawnie usuwa niewielki błąd jego ortonormalności.

`porownanie_podprzestrzeni/porownaj.py:38–43` poprawnie wyznacza cosinusy kątów głównych jako wartości osobliwe Q500^T Q1000 i normę różnicy projektorów jako sqrt(2r−2Σ cos²). Test `:81–84` sprawdza niezmienniczość po niezależnych obrotach baz. Podane kąty27.69/30.56/31.41/35.54/42.72/44.10/58.60/83.95 stopnia i średnie cos²=0.531748 opisują geometrię dwóch oszacowanych podprzestrzeni. Nie są procentem zachowanej tożsamości, dokładnością prognozy ani obrotem natywnych makrowektorów e_j. Główny forecast nadal korzysta wyłącznie z zamrożonego U500.

Kalibracja500 zakończyła się według zapisu wykonawcy z niezmienionymi buforami. Top5 osi1-based:359,594,472,512,138. Diagnostyczny test obrazowy probe500: AUROC0.955294, BA0.90 przy stałym progu0.5; to nie jest podmiana głównego celu predykcji ruchu na klasyfikację obecności.

## Audyt kodu przed uruchomieniem testu

`trajektorie.py:7–23` wybiera jawnie właściwy frame_id i source-role, otwiera gęstą historię jako mmap, a następnie projektuje tylko31 tokenów maski i8 stałych kontroli tła. U1000 zapisuje oddzielnie (`:24–25`). `ZGODNOSC_K500.json` potwierdza wykonany przez workera forward tej sondy względem wiersza500 historii: relative L2 `4.30e-7`, max abs `1.79e-6`, zgodny kształt600×768. Audytor nie wykonywał własnego forwardu.

`gle.py:5–13` prawidłowo buduje pary celu i poziomu oraz pamięć przyrostów. Wagi spłaszczonych wierszy mają kolejność czas→token zgodną z reshape; ich suma wynosi1. Intercept jest centrowany ważoną średnią i nie jest karany. `:15–20` właściwie przelicza b,A,K ze skali poziomu i przyrostu do współrzędnych surowych oraz sprawdza zgodność dwóch postaci predykcji. Transpozycje odpowiadają równaniu kolumnowemu.

`gle.py:22–27` buduje poprawny companion dla rozszerzonego stanu `[x_t,delta x_t,...]`, nie bezpośrednio dla `[x_t,x_(t−1),...]`; promień spektralny dotyczy części jednorodnej. `:34–51` korzysta tylko z historii do originu i przesuwa pamięć o własny przewidziany przyrost. Nie znaleziono teacher forcing wewnątrz rolloutu.

`prognoza.py:30–45` wybiera ustawienia wyłącznie na validation i zapisuje modele po reficie oraz SHA przed testem. dt1: wybór główny L4/ridge0.01, średni walidacyjny skill po H1/10/50 około0.2492; wariant wybrany dla H1 ma L2/ridge0.01. dt10: główny L1/ridge1, średni skill po H10/50 około0.0460. Pole artefaktu `best_H1` przy dt10 oznacza faktycznie wybór dla H10; raport musi zachować właściwą jednostkę czasu.

`prognoza.py:67–89` zachowuje wspólne originy dla wariantów, oddzielne H100 i porównanie dt1/dt10 na originach co10. Transfer na tło zmienia tylko anchor na `q_bg[500]`; nie dopasowuje operatora ani skal na kontroli. Testy `:77–81` przewidują dokładną zgodność prognozy pełnego wejścia, prefiksu i wejścia z zatrutą przyszłością.

`gle.py:54–70` liczy MSE rzeczywistych wektorów w surowym q, skill jako stosunek zagregowanych MSE, kosinus przemieszczeń z jawnym mianownikiem oraz osobne znaki z progiem fit. Persystencja ma nieokreślony kosinus. `:72–81` losuje całe bloki originów; to samo ziarno i rozmiar dla metod dają sparowane resamplowanie. Podczas audytu wskazano, że parametr dt tej funkcji oznacza odstęp originów, a nie zawsze dt modelu. Obecny pipeline jest poprawny: główne originy mają ten sam odstęp co dt, zaś osobne porównanie dt1 na originach co10 nie raportuje bootstrap CI. Ewentualne przyszłe dodanie takiego CI wymaga odstępu10.

`prognoza.py:48–63` zachowuje ciągły czas w residual diagnostics, nie skleja tokenów i oddziela ACF/Ljung–Box od skew/kurtosis/Jarque–Bera. Surowe p-wartości są jawnie opisane jako eksploracyjne i bez korekty wielokrotności. dt10 ma tylko20 residual observations; zwłaszcza jego testy asymptotyczne mają małą wiarygodność.

Nie wykryto przed testem błędu wymagającego zmiany fit lub algorytmu prognozowania. Wykonawca otrzymał ten wynik przed uruchomieniem etapu testowego. Po audycie zmienił nazwę argumentu bootstrapu na jawne `origin_stride`; nie zmieniło to głównego estymatora ani siatki testowej.

## Sprawdzone wyniki testu

Audytor porównał bieżące SHA plików U500, trajektorii i macierzy obu dt z zapisami zamrożonej walidacji: wszystkie są zgodne. `AUDYT_PREFIX_DT1.json` oraz `AUDYT_PREFIX_DT10.json` zawierają po18 poprawnych przypadków, obejmujących9 modeli/baseline'ów i originy800/950. Prognozy pozostają dokładnie identyczne po usunięciu lub zatruciu przyszłego sufiksu. Nie wykazano użycia przyszłości w fit/rollout.

Główny model dt1 ma L4, ridge0.01, pełny numeryczny rank40/40, condition po regularyzacji352.619, promień companion0.980263 oraz błąd zgodności surowej i skalowanej postaci `4.45e-16`. Dla dt10: L1, ridge1, rank16/16, condition4.134, promień0.949201 i błąd `3.34e-16`. Promień poniżej1 opisuje jednorodną część danego modelu; nie ustanawia stacjonarności prawdziwego uczenia ani identyfikowalności fizycznych sił.

MSE dotyczy wyłącznie ośmiu współrzędnych w zamrożonym U500. Skill oznacza `1−MSE_model/MSE_persistence` dla tego samego zbioru originów, nie odsetek poprawnie przewidzianych cząsteczek i nie jakość całego wektora768D.

| dt | H w aktualizacjach | Skill głównego modelu | Cosine przemieszczenia8D | Trafność znaku wszystkich składowych | Trafność po progu |
|---:|---:|---:|---:|---:|---:|
|1|1|60.238%|0.74024|78.611%|81.594%|
|1|2|44.277%|0.62317|73.833%|75.192%|
|1|5|12.341%|0.38310|63.871%|64.238%|
|1|10|9.114%|0.36734|65.342%|65.739%|
|1|20|9.397%|0.41279|66.619%|66.912%|
|1|50|−5.029%|0.39313|63.840%|64.076%|
|10|10|16.118%|0.33640|63.030%|64.300%|
|10|20|20.124%|0.39602|65.250%|66.218%|
|10|50|13.276%|0.38091|63.776%|64.973%|

H1 dt1: MSE `0.002959884` wobec persystencji `0.007444041`. Wszystkie4681 wektorów token–origin mają określony kosinus. Znak ocenia37448 składowych, a z góry ustalony próg zachowuje33417, czyli około89.24% nieważonej liczby porównań. Sama accuracy jest ważona coverage. Wynik81.59% jest warunkową trafnością znaku składowych; nie oznacza81.59% prawidłowych całych wektorów i nie stanowi bezwarunkowego osiągnięcia80%. Dla wszystkich zmian wynik wynosi78.61%.

Opisowy bootstrap H1 dt1: skill `[56.448%,64.464%]`, znaki wszystkich składowych `[77.493%,79.717%]`, znaki po progu `[80.200%,82.979%]`, cosine `[0.69855,0.77854]`. Blok ma50 aktualizacji, a151 originów daje tylko około3 takich długości. Dolnego końca warunkowego CI powyżej80% nie wolno przedstawiać jako gwarancji generalizacji na inne obrazy, biegi lub fazy treningu.

Znaczenie pamięci jest zależne od skali. Przy dt1/H1 model L0 ma skill−0.308%, a constant velocity47.770%, wobec60.238% głównego L4. Przy dt10 model L0 uzyskuje15.753/19.079/10.447% dla H10/20/50, czyli jest blisko wybranego L1. Nie wykazano jednakowo dużej korzyści pamięci w każdym dt i H. Osobny wariant wybrany na validation dla H1 (L2) ma H1 skill60.592% i znaki78.516%; nie zastępuje po teście głównego modelu.

Dla H50 wynik dt1 jest punktowo gorszy od persystencji, a jego opisowe CI skillu `[-26.071%,11.714%]` obejmuje0. Dodatni wynik H50 dt10 także ma CI obejmujące0: `[-7.958%,29.937%]`. Nie należy ogłaszać rozstrzygniętej poprawy na tym horyzoncie. H20 dt1 ma dodatnią estymatę punktową, lecz CI `[-7.226%,23.838%]` również obejmuje0.

Uczciwe porównanie dt1/dt10 na wspólnych16 originach co10 daje dla dt1 skille H10/20/50 około12.27/5.90/−4.25%, wobec16.12/20.12/13.28% dt10. To wynik opisowy; nie testowano wszystkich możliwych dyskretyzacji ani nie wybierano nowego wariantu po teście.

H100 pozostaje osobną diagnostyką bez CI: dt1 skill13.045% na101 originach, dt10 skill26.811% na11 originach. Zmienia się zbiór originów względem H<=50 i między dt. Nie wolno z tych dwóch liczb bezpośrednio orzekać przewagi dt10 lub „odzyskania” dokładności wraz z horyzontem.

## Reszty i zakres interpretacji Langevina

Diagnostyka dt1 zawiera200 kolejnych residual observations, dt10 tylko20. Dla ważonej wspólnej reszty dt1 nominalne p Ljung–Box są poniżej0.05 w6/8 współrzędnych; osobny Jarque–Bera ma małe p w jednej współrzędnej. Wyniki nie pozwalają uznać reszt za udowodniony biały szum gaussowski. Brak odrzucenia normalności w pozostałych współrzędnych nie dowodzi normalności. Mała próba dt10 szczególnie ogranicza testy asymptotyczne.

Średnie korelacje reszt pomiędzy różnymi tokenami w tej samej współrzędnej wynoszą przy dt1 około0.489–0.890. To mocna empiryczna współzależność, zgodna z oddziaływaniem wspólnych aktualizacji sieci na wiele tokenów. Nie identyfikuje parowych sił między cząsteczkami. Korelację przestrzenną należy odróżniać od czasowej bieli: wielowymiarowy biały szum także może mieć niediagonalną kowariancję.

W badaniu dopasowano empiryczny model przyrostu z dryfem i pamięcią oraz zmierzono reszty; forecast jest prognozą warunkowego przyrostu bez losowania przyszłego szumu. Nie zidentyfikowano niezależnie źródła wymuszenia batch/AdamW/BN, współczynnika dyfuzji fizycznej, FDT ani pełnego generatora stochastycznego. Nie ma w tym wyniku nowego treningu ViT lub dowodu poprawy detektora.

## Końcowa zgodność artefaktów i tekstu

Odczytano końcowy `WYNIKI.md` wygenerowany 2026-10-02 o 01:05:46 oraz aktualne `PODSUMOWANIE.md`. Ich zasadnicze liczby, równanie i rozróżnienie MSE, kosinusa oraz trafności znaku zgadzają się z artefaktami. Końcowy odczyt NPZ, bez ponownego dopasowania, potwierdził:

| Plik i pole | Kształt | Znaczenie |
|---|---:|---|
| MACIERZE_DT1.npz / selected_b | 8 | Wyraz wolny w zakotwiczonym x |
| MACIERZE_DT1.npz / selected_A | 8×8 | Korekta dryfu, nie pełny operator I+A |
| MACIERZE_DT1.npz / selected_K | 4×8×8 | Cztery macierze opóźnionych przyrostów |
| MACIERZE_DT1.npz / selected_anchor | 31×8 | Oddzielny q_i(500) każdej cząsteczki |
| MACIERZE_DT10.npz / selected_K | 1×8×8 | Jedna macierz przyrostu co 10 aktualizacji |
| MACIERZE_DT1/DT10.npz / selected_epsilon | 8 | Progi pomijania bardzo małych zmian w metryce znaku |

Pole `selected_epsilon` nie przechowuje realizacji losowej reszty z równania GLE. Ta kolizja nazw wymaga odróżnienia progu metryki od epsilon_i(t) w zapisie dynamiki. Odtworzenie ruchu w mierzonej podprzestrzeni ma postać `U500 @ (q_hat - q_now)`; nie wyznacza ruchu w pozostałych 760 ortogonalnych kierunkach.

Prognoza punktowa przyjmuje przyszłą resztę równą zeru. Nie wynika z tego, że empiryczna średnia odłożonych reszt wynosi zero. Zapis podsumowania poprawnie rozróżnia te dwa twierdzenia. Podobnie próg nearzero pochodzi z fit, ale to, czy rzeczywista przyszła zmiana przekracza próg, wiadomo dopiero po poznaniu celu. Wynik 81.59% jest warunkową metryką retrospektywną i nie jest dostępnym online poziomem pewności.

`FILTR_NEARZERO.json` potwierdza dla H1 zachowanie 33417/37448 składowych, czyli 89.24% nieważonej liczby i 89.00024% sumy wag coverage. Ten drugi odsetek jest zgodny z ważeniem samej trafności. Nie jest to usuwanie całych tokenów ani identyfikacja pewnych wektorów.

Sprawdzono również transfer do ośmiu cząsteczek tła: H1 skill 57.60012%, cosine 0.71731, znak wszystkich składowych 77.54553%. Zmieniono tylko tokenowy anchor, zachowując operator. Wynik wspiera ostrożne stwierdzenie, że przewidywalność obejmuje także tło; nie wykazuje wyłącznej dynamiki semantycznej narzędzia. Nie ustala też przyczynowego udziału samego BN.

Do autora raportu przekazano wyłącznie doprecyzowania prozy: źródło zawiera 1001 stanów od k=0 do k=1000, czyli 1000 wykonanych aktualizacji; rozdzielenie obu znaczeń epsilon; jawny charakter filtra rzeczywistej zmiany. Przekazano też uwagę, że diagnostyczne H100 używa 101 originów dt1 i 11 originów dt10, co uniemożliwia bezpośrednie porównanie tych dwóch liczb jako wyniku na tym samym zbiorze początków. Żadna uwaga końcowa nie wymagała zmiany fit, ponownego testu ani wyboru nowego wariantu.

Końcowa ocena: implementacja realizuje zadeklarowaną, autonomiczną prognozę ze wspólnym dryfem i pamięcią przy zachowaniu dostępności czasowej danych. Najmocniejszy wynik dotyczy jednej aktualizacji: 60.24% redukcji MSE i 78.61% trafności znaku wszystkich ośmiowymiarowych składowych. Wniosek o 80% dla całego wektora, białym szumie Langevina, fizycznych siłach między tokenami lub poprawie uczenia detektora pozostawałby nieuprawniony. Aktualne podsumowanie tych wniosków nie stawia.
