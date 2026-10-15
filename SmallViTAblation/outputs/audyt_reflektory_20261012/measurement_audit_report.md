# Audyt pomiarów ruchu reflektorów i cząsteczek

Data: 2026-10-12. Audyt istniejącego kodu i wyników; bez treningu i bez ponawiania zakończonych eksperymentów. Ścieżki poniżej są względne wobec SmallViTAblation. Wyniki ponownego odczytu JSON-ów: `measurement_audit_existing_results.json`; skrypt: `experiments/audyt_reflektory_20261012/measurement_audit_existing_results.py`.

## Główne ustalenia

1. Liczba 74,58% jest poprawnie obliczonym ilorazem dwóch energii wariantów kontrfaktycznych, ale nie udziałem w rzeczywistym wolnym dryfie. Między epokami 1 i 60 interakcja pominięta w mianowniku jest większa od obserwowanej energii zmiany.
2. Zakres 84–98% ma podobny mianownik. Dotyczy przyrostów odległych o 10 kroków na początku i 100 kroków później; nie jest pomiarem pasma szybkich drgań ani rozdzieleniem czasu 3–13 kroków.
3. W starym rozkładzie wejście `u` zawiera ostatnią LayerNorm i uwagę ostatniego bloku. Nie można utożsamiać jego zmiany wyłącznie z wcześniejszymi pięcioma blokami. „Lasery” tego rozkładu to tylko fc1 i jej bias, a nie pełny MLP z fc2.
4. Pełny wynik `NADAZANIE_PULAPKI_S47.json` został udostępniony podczas audytu przez niezależnie uruchomiony pomiar. Obejmuje 746 przedziałów do kroku 65 600 i P2 do epoki 60. Nie potwierdza jednostronnego nadążania: wczesne i środkowe antykorelacje występują w obu kierunkach, a obrót strumienia źle przewiduje zmiany czujników także późno. Osobny `_test.json` obejmuje tylko początek i wymaga odróżnienia od pełnego wyniku.
5. AdamW i BatchNorm mają pamięć. Ich czasy wygaszania nie stanowią wyjaśnienia zmierzonego czasu zaniku funkcji pośredniego rozpraszania bez osobnego testu.

## 1. Mianowniki starego rozkładu

Źródło: `outputs/porownanie_szklo_20261010/eden/wiazki_koloidy.py:118–128`.

Przy `z(u,W,b)=GELU(Wu+b)` skrypt liczy:

\[
D=z(u_B,W_B,b_B)-z(u_A,W_A,b_A),
\]
\[
A=z(u_A,W_B,b_B)-z(u_A,W_A,b_A),\quad
B=z(u_B,W_A,b_A)-z(u_A,W_A,b_A),\quad I=D-A-B.
\]

`udzial_wiazek = E(A)/(E(A)+E(B))`, gdzie `E(V)` to średnia kwadratu normy 768D po tokenach i zdjęciach. Mianownik nie zawiera `E(I)` ani iloczynów krzyżowych. Dokładna tożsamość brzmi `E(D)=E(A)+E(B)+E(I)+2<A,B>+2<A,I>+2<B,I>`.

W `WIAZKI.json`, `cala_droga_e1_e60`:

| Wielkość | Wartość |
|---|---:|
| `E(D)` | 198,849945 |
| `E(A)` | 242,945343 |
| `E(B)` | 82,792366 |
| `E(I)` | 247,717834 |
| `E(A)/(E(A)+E(B))` | 0,745831 |
| `E(A)/E(D)` | 1,221752 |
| `E(I)/E(D)` | 1,245753 |
| suma składników krzyżowych | −374,605598 |

To nie jest wykazany udział 75% w całkowitym ruchu. Jest to porównanie odpowiedzi sieci na dwa osobne podmienienia stanu. Dla tak odległych checkpointów model z krzyżowo podmienionymi elementami może być poza rozkładem, do którego dostosowało się uczenie.

`TEST_SFERY.json` nie dostarcza niezależnego potwierdzenia: `test_sfery_laserow.py:46–52` kopiuje i zaokrągla ten sam iloraz z WIAZKI.

Krótkie przyrosty: średni iloraz `E(A)/(E(A)+E(B))` wynosi 0,02415 na starcie (10 kroków), potem 0,11184–0,16301 dla odstępu 100 kroków. Dopełnienie wynosi 83,70–97,58%. Reszta `E(I)/E(D)` jest wtedy niewielka: około 0,002–0,018. To wspiera ostrożne zdanie „w tym rozkładzie krótkich przyrostów zmiana wejścia daje znacznie większą normę niż podmiana fc1”. Nie dowodzi pochodzenia 84–98% rzeczywistych drgań. Etapy uśredniają ilorazy po maksymalnie 30 parach checkpointów (`wiazki_koloidy.py:139–158`), a nie liczą ilorazu sum energii całego etapu.

„Wolny dryf 75%” jest różnicą końców epok 1 i 60 (`:163–166`), bez filtru częstotliwości, odjęcia drgań lub estymacji prędkości dryfu. Dane obejmują 100 zdjęć, 50 TRAIN i 50 VALID, po 600 tokenów; tokeny nie stanowią niezależnych replik. Wybór to pierwsze elementy posortowanej podpróby (`:55–62`), niekoniecznie reprezentatywny rozkład filmów.

## 2. Co faktycznie rozdziela WIAZKI

`wiazki_koloidy.py:68–70` przechwytuje wejście fc1, czyli stan **po norm2** bloku 6. Podział jest następujący:

- `W,b`: tylko macierz fc1 i jej bias ostatniego bloku.
- `u`: łodyga z buforami BN, poprzednie pięć bloków, uwaga bloku 6, rezyduum przed jego MLP i affine LayerNorm norm2 tego samego bloku.
- `z`: latent po GELU w 768D; macierz fc2 nie uczestniczy w jego wzorze.

Dlatego zdania „75% od przestawiania laserów ostatniego bloku” i „84–98% od wcześniejszych warstw” nie mają identycznego podziału parametrów jak model reflektora zawierający zarówno fc1, jak i fc2. Nowy P1 naprawia część tej niespójności: `theta()` w `pomiar_nadazania_pulapki.py:134` obejmuje fc1, fc2, oba biasy oraz parametry norm2; wejście jest przechwycone przed norm2. Nadal uwaga ostatniego bloku należy do strony wejścia.

## 3. Stan nowego pomiaru opóźnienia

**Pełny wynik, ostateczny stan audytu:** `NADAZANIE_PULAPKI_S47.json` ma `test=false`; 746 przedziałów, maksymalny błąd dokładnego rozkładu 5,45e-17. Fazami są 100 przyrostów po 10 kroków i 209/218/219 przyrostów po 100 kroków. Podpisy epok 1–20 itd. są przybliżone: rzeczywiste granice kroków są jawne.

| Faza | C(+1) | C(−1) | zero średnia ± sd | a(+1) | c(+1) |
|---|---:|---:|---:|---:|---:|
| 0–1000 | −0,03427 | −0,07071 | −0,00057 ± 0,00989 | −0,02294 | −0,07655 |
| 1000–21900 | −0,04667 | −0,04354 | −0,00134 ± 0,01051 | −0,01730 | −0,02681 |
| 21900–43700 | −0,03006 | −0,04804 | 0,00028 ± 0,00781 | −0,00587 | −0,02596 |
| 43700–65600 | −0,01288 | −0,03083 | −0,00063 ± 0,00947 | 0,00576 | −0,02903 |

Przy własnym progu 3 sd dla C, w fazach 1–20 i 20–40 **oba** kierunki tau=±1 wychodzą poza zero. W fazie 40–60 tylko tau=−1 wychodzi nieznacznie poza ten próg (3,19 sd), a kierunek +1 nie. Żaden przypadek nie daje czystej sekwencji „najpierw ruch wejścia, potem jego wyłączne naśladowanie przez parametry”. Wersja Shapleya zachowuje ten wzór. Regresje a(+1) w fazach po starcie nie przekraczają 3 sd ich własnych zer. To nie potwierdza P1; przy dosłownym kryterium skryptu wcześniejsze dwukierunkowe wyniki należą do kategorii niejednoznacznej. Ograniczenia metodologiczne wymienione niżej obowiązują także pełny wynik.

**Osobny snapshot próbny (nie mylić z pełnym):**

Plik `NADAZANIE_PULAPKI_S47_test.json` zawierał `test=true`; został usunięty przez zewnętrzną sesję w trakcie audytu. Jego SHA i podsumowanie zachowano w naszym JSON audytu. Gałąź TEST (`pomiar_nadazania_pulapki.py:176–177`, `:337–338`) ogranicza P1 do 0–300 co 10 oraz 1000–6000 co 100 i P2 do epok 0–6. Etykiety w JSON pozostają „0–1000”, „epoki 1–20”, „epoki 0–20”, więc same nazwy faz zawyżają faktyczny zakres. P1 ma 30 i 50 analizowanych przyrostów; dodatkowy 81. przedział 300–1000 nie wchodzi do faz o stałym kroku.

Po stronie rachunkowej test jest dobry: błąd tożsamości `Delta_in+Delta_W=Delta_F` wynosi maksymalnie 6,46e-17 przed konwersją do float16; ręczny wzór MLP zgadza się z forwardem na poziomie 4,68e-7 względnie. Te kontrole sprawdzają implementację, nie hipotezę przyczynową.

W faktycznym oknie 1000–6000:

- `C(+1)=-0,05969`, `C(-1)=-0,03727`, `C(0)=0,09859`; jedno przesunięcie oznacza 100 kroków optymalizatora.
- Zero dla C: średnia −0,00162, odchylenie 0,01606.
- Współczynnik regresji „pułapka za koloidem” dla +1: −0,03262; zero −0,0007 ± 0,012.
- Kierunek odwrotny: 0,01098; zero −0,0011 ± 0,02341.
- Wariant Shapleya: C(+1)=−0,05675, C(−1)=−0,02581.

To wąska obserwacja antykorelacji, a nie wykazanie nadążania. P1 operuje przyrostami funkcji wyjścia, nie położeniem środka rzeczywistej pułapki. Ujemny znak może oznaczać kompensację, ale także wspólną odpowiedź na batch/stratę lub wynik niestacjonarności.

Ograniczenia metody (`pomiar_nadazania_pulapki.py:218–259`):

- C to średni cosinus wektorów, bez odjęcia średniej czasowej; nie jest wycentrowaną korelacją Pearsona. Regresja jest wspólną regresją skalarnego współczynnika bez interceptu, z jednym własnym stanem jako kontrolą, a nie pełnym modelem historii.
- Null regresji jest liczony tylko dla tau=1, a badane są tau=1…10. Dla startowego testu istnieje tylko dziewięć dopuszczalnych przesunięć null.
- Przesunięcia kołowe i permutacje czasu zakładają wymienność/stacjonarność. Rozgrzewka i zmiana dynamiki uczenia naruszają ją. Odjęcie trendu stosuje się tylko do skalarnej log-prędkości, nie do C/regresji.
- Przeszukiwanie lagów, kierunków, faz i obserwabli nie ma kalibracji maksimum ani poprawki wielokrotnej. Kryterium „ponad 3 sd dla dowolnego tau” nie jest globalnym testem na tym poziomie.
- Wspólny batch, strata, clipping, AdamW i BN są niewidocznymi wspólnymi przyczynami. Istotność kierunkowa nie identyfikuje przyczynowości.
- Rozdzielczość 100 kroków po starcie nie pozwala ustalić opóźnienia 3–13 kroków. Nazwa „τ>=1” oznacza lag checkpointów, nie kroków optymalizatora.
- Dane zapisuje się do float16 po policzeniu małych różnic; wymaga kontroli błędu kwantyzacji przed wnioskowaniem o subtelnych efektach późnego uczenia.

Nowy `energia_W` również ma mianownik `||Delta_in||²+||Delta_W||²`, nie `||Delta_F||²`. Shapley zapewnia dokładny podział wektorowej zmiany, ale kwadraty norm składników nadal nie są addytywne, jeśli składniki są skorelowane.

## 4. Nowy P2 i obracanie czujników

W pełnym wyniku dla epok 40–60:

| Okno / baza | Własny obrót czujników | Obrót strumienia na czujnikach | Zero losowe osie | Cosinus generatorów |
|---|---:|---:|---:|---:|
| 1 epoka / 95% | 0,08082 | −1,33022 | −1,26232 | −0,02638 |
| 1 epoka / 99% | 0,17603 | −1,73681 | −1,59836 | −0,02219 |
| 5 epok / 95% | 0,06991 | −0,64274 | −0,61010 | −0,01764 |
| 5 epok / 99% | 0,15317 | −0,82107 | −0,76978 | −0,01649 |

To mocny negatywny wynik dla konkretnego modelu „ten sam pełny obrót strumienia przenosi czujniki”. Udział jest wszędzie ujemny i nie osiąga wymaganego dodatniego progu. Nie należy z tego wnioskować braku jakiejkolwiek odpowiedzi parametrów na dane: testuje się wybrany wspólny obrót, nie wszystkie formy adaptacji. Późne cosinusy z opóźnieniem są małe i bliskie zerom; cztery okna pięcioepokowe dają tylko trzy pary sąsiadujących okien i sześć par zerowych.

Osobny wynik próbny dla sześciu okien po jednej epoce (faktycznie 0–6), baza 95%, miał 47 wymiarów. Obrót strumienia wyjaśniał 0,37474 jego własnej zmiany, lecz zastosowany do czujników dawał udział −2,97568. Losowe osie dawały −3,00104; cosinus generatorów wynosił −0,01153. Ten test wstępny również nie wspierał wspólnego obrotu, ale sam nie obejmował późnych epok.

Komentarz w nagłówku, że składowe czujników poza bazą 95% wariancji dają jedynie stałą wchłanianą przez bias, jest niepoprawny: reszta PCA ma 5% wariancji, a więc nie jest przestrzenią zerową. Stały wkład dotyczy dokładnie zerowej zmienności danych, nie umownego progu 95%. Mała wariancja nie gwarantuje małego wpływu po przemnożeniu przez uczone czujniki. Trzeba mierzyć błąd rekonstrukcji preaktywacji/logitów, a nie tylko wariancję wejścia.

R_x opisuje obrót najlepiej dopasowany do określonych danych i wybranej bazy. Ujemne dopasowanie nie wyklucza lokalnej odpowiedzi gradientowej czujników ani mniejszego obrotu w podobnych kierunkach. Dodatni wynik też nie wykazywałby sam w sobie przyczynowości.

### Dodatkowe nadinterpretacje w nowym opisie wyników

W trakcie audytu pojawił się też `outputs/porownanie_szklo_20261010/film_krysztal/NADAZANIE_PULAPKI_S47.md`. Jego zasadniczy ostrożny wynik P1 i negatywny wynik konkretnego testu P2 są zgodne z liczbami. Niektóre rozwinięcia nie wynikają jednak z zastosowanego estymatora:

- Dodatnie C(0) nie dowodzi „wspólnego zejścia po gradiencie”. Gradient dotyczy straty w parametrach, a mierzony cosinus dotyczy zmian wyjścia MLP. Można stwierdzić dodatnią zgodność kierunków, a mechanizm przedstawić jako hipotezę.
- Ujemna autokorelacja przyrostów nie musi znaczyć odwracania ruchu cząstki w pułapce. Różnicowanie stacjonarnego poziomu samo może tworzyć ujemną autokorelację kolejnych przyrostów. Należy modelować także poziomy sygnału i ich widmo.
- Sumy a(tau) i c(tau) z osobnych regresji predykcji przyszłego przyrostu nie są oszacowaniem procentu „odrobionego” przez 1000 kroków. Horyzonty odnoszą się do zależnych danych i nie tworzą funkcji odpowiedzi impulsowej. Do takiej interpretacji potrzebny jest poprawnie wyspecyfikowany układ dynamiczny/interwencja i jego odpowiedź.
- a(7) nie ma własnego null w pliku; zastosowanie sd z a(1) nie kalibruje testu a(7). Pojedynczy wynik przy wielu testach nie daje mocnego dowodu, ale nie można z samych 60 porównań rozstrzygać, że konkretnie jest szumem.
- Iloraz C po tasowaniu tokenów do C przed tasowaniem nie rozdziela wariancji na „60–70% wspólnego wiatru” i resztę „własną tokenu”. To spadek średniego cosinusa po jednej permutacji, bez addytywnego rozkładu wspólnego i indywidualnego składnika.
- Kwadrat średniego cosinusa generatorów nie jest automatycznie procentem wyjaśnionej wariancji ruchu reprezentacji. Dopasowanie projekcji pojedynczego generatora to inna wielkość niż wariancja zmian tokenów lub czujników.
- Różnica między procentem dopasowania obrotu w 192D, w 99% PCA i w 95% PCA nie dowodzi, że „większość obrotu siedzi w nieużywanych kierunkach”: zmieniają się zarazem dane, optymalny obrót i mianownik. Należy pokazać addytywne energie zmian w dopełnieniach przestrzeni oraz wpływ na rzeczywiste preaktywacje.

## 5. Dokładny stan uczenia i obserwacja tokenu

Minimalny użyteczny pełny opis to stochastyczny układ dyskretny z pamięcią optymalizatora i buforami:

\[
S_t=(\theta_t,m_t,v_t,\mu^{BN}_t,(\sigma^2)^{BN}_t,k_t,Q_t,R_t),
\qquad S_{t+1}=\mathcal T(S_t;\mathcal D).
\]

`theta` zawiera **wszystkie** parametry łodygi, sześciu bloków, norm i głowy; `m,v` wszystkie momenty AdamW; BN wszystkie running_mean/running_var i liczniki; k także liczniki optymalizatora i harmonogram lr; Q pozostałą kolejkę permutacji; R stan generatorów losowych i augmentacji. Dla stochastycznego modelu zamiast RNG można jawnie wprowadzić batch i augmentację jako wymuszenie `xi_t`, ale kolejność próbkowania bez zwracania nadal ma pamięć. Bitowa reprodukcja wymaga dodatkowo zgodnego środowiska obliczeń.

Kod treningu: `assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/trening.py`:

- linie 26–33: AdamW, beta1=0,9, beta2=0,999; weight decay 0,05 dla parametrów o ndim>=2, zero dla wektorów/biasów;
- linie 44–67: lr rośnie liniowo przez 300 kroków do 5e-4; batch 32, kolejka permutacji, odbicie i jasność/kontrast;
- linie 74–82: forward w bf16, backward, wspólne przycięcie normy gradientu do 1, krok optymalizatora;
- linie 97–115: odczyt sond w eval.

Nieprzycięty gradient `g_t = grad L_train(theta_t; batch_t, aug_t)` staje się `gbar_t = c_t*g_t`, gdzie w implementacji `c_t=min(1,1/(||g_t||+1e-6))`. AdamW aktualizuje momenty wykładniczo, stosuje bias correction i dzielenie współrzędnych przez `sqrt(vhat)+eps`, a macierze wag dodatkowo mnoży przez `1-lr*wd`. To nie jest izotropowe prawo ruchu overdamped Langevina w latentach.

W train BN używa statystyk bieżącej paczki i aktualizuje bufor z momentum=0,1: `running <- 0,9*running+0,1*batch_stat` (wariancja bufora wykorzystuje estymator z poprawką). W eval używa buforów. Odczyt ustalonego obrazu powinien więc być zapisany:

\[
z_{i,j,t}=\operatorname{Obs}_{j}(I_i;\theta_t,B^{BN}_t).
\]

Obserwacja dotyczy całego obrazu I_i; przez uwagę token j zależy od pozostałych tokenów. Sam zapis `f(x_j;theta_t)` może to ukrywać. Przy ustalonych theta i BN, w eval, token nie ma niezależnej pamięci ani własnego losowego napędu. Nie znaczy to, że czasowy sygnał z nie może wykazywać opóźnienia: jest obserwacją układu z pamięcią. Opóźnienia i metastabilność mogą być emergentnymi cechami tej obserwacji.

Potwierdzono na utworzonym obiekcie modelu: trzy BN mają momentum=0,1, eps=1e-5; normy w blokach eps=1e-6, końcowa norm eps=1e-5; wszystkie Dropout p=0. W checkpointcie k01094.pt są tylko klucze `net,krok,epoka`, BN ma num_batches_tracked=1094. Ten typ pliku pozwala odtworzyć obserwację, ale nie pełny krok AdamW: brak momentów optymalizatora, kolejki i RNG. `trening35000.py:199–222` rozróżnia pliki samych wag od pełnego stanu; `:191` jawnie uprzedza, że kolejka permutacji przy wznowieniu zaczyna się od nowa.

Czas e-folding beta1 i BN to `-1/log(0,9)=9,49` kroków, beta2 `-1/log(0,999)=999,50`. To czasy filtrów przy ustalonym pobudzeniu, nie zmierzone czasy relaksacji całej sieci. Wartości 10/1000 w Z8 są przybliżeniami tej pamięci. Przypisanie jej zmierzonego tau wymaga pomiaru; tau z F_s nie jest automatycznie stałą autokorelacji wychylenia.

## 6. Zalecane nowe pomiary

1. **Pełny bilans ruchu:** cztery kombinacje wejścia i parametrów, interakcja, oba porządki, Shapley, wszystkie iloczyny krzyżowe; raport norm i przypisania energii `a_in=<Delta_in,Delta_total>/||Delta_total||²`, `a_W=<Delta_W,Delta_total>/||Delta_total||²`. Te ostatnie sumują się do 1, ale mogą być ujemne i zależą od reguły podziału; nie nazywać ich udziałami przyczynowymi.
2. **Jednolity podział stanu:** osobno łodyga i BN, wcześniejsze bloki, uwaga ostatniego bloku, norm2, fc1 i fc2. Najpierw diagnoza podmianami w istniejących checkpointach, z kontrolą wyników poza rozkładem. Zmierzyć zarówno z768, stan residual192, jak i wyjście/głowę.
3. **Pamięć BN:** odczytać ten sam obraz przy zamrożonych theta i zmienianych wyłącznie buforach BN; porównać do zmiany theta z zamrożonym BN. To rozdzieli część obserwacyjnego ruchu pochodzącą z buforów.
4. **Rzeczywisty lag:** wykorzystać istniejący zapis co krok, jeśli zawiera potrzebny stan. Siatka co 100 kroków służy do efektów co najmniej tej skali, nie tau3–13. Późne fazy, niezależne zdjęcia i ziarna, bootstrap bloków czasowych/filmów, null zachowujący autokorelację i lokalną niestacjonarność, kalibracja maksymalnego wyniku po lagach.
5. **Przyczynowość (dopiero osobny, autoryzowany przyszły eksperyment):** krótkie kontrolowane rozgałęzienia z tego samego pełnego stanu i tymi samymi batchami/augmentacjami; perturbacja wybranej grupy, pomiar odpowiedzi reszty, oddzielne utrzymanie/reset momentów i BN. Zamrożenie grupy zmienia także clipping, więc to trzeba świadomie kontrolować. Sam lead-lag nie zastępuje takiej interwencji.
6. **Ruch wspólnej bazy:** sprawdzić dopasowanie Prokrustesa na wydzielonych zdjęciach, przewidywanie preaktywacji po obrocie oraz resztę poza PCA. Dopasować również wielkość obrotu; zbadać wrażliwość na 95%, 99% i pełną regularizowaną przestrzeń. Nie utożsamiać dowolnego obrotu reprezentacji z fizycznym obrotem makrowektorów.

## 7. Granice pewności pozostałych liczb

`PRZELICZENIE_ZASADY_S47.json` obejmuje statystyki wag wybranego ostatniego bloku i Jakobian dla 600 tokenów jednego zdjęcia TRAIN 0 w siedmiu checkpointach. `POMIARY_OPISU_S47.json` analogicznie dotyczy TRAIN 0; C/D zawierają cztery checkpointy. Nie są to wyniki dla wszystkich zdjęć, wszystkich bloków ani powtórzeń z różnymi ziarnami.

Pochodzenie WIAZKI jest dobrze sprawdzone przez porównanie fc1 z s47 w kilku punktach (`przelicz_zasady_s47.py:185–200`), a nowe forwardy odtwarzają zapisane latenty z R2 około 0,99995. Nie należy z tych kontroli wywodzić, że wszystkie drobne przyrosty są tak samo dokładne: błąd względem dużego sygnału może być istotny względem różnicy dwóch sygnałów.

Skrypt `przelicz_zasady_s47.py:155` i starszy `pomiary_opisu_s47.py:125` nadal używają eps=1e-5 w ręcznym LN, mimo że model ma 1e-6. Nowy pomiar opóźnienia pobiera rzeczywiste eps. Jeśli inne pliki kontrolne wykazują nieistotny wpływ błędu, można zachować stare liczby jako przybliżone; nie nazywać starego skryptu dosłownie dokładną implementacją aktualnej normy.
