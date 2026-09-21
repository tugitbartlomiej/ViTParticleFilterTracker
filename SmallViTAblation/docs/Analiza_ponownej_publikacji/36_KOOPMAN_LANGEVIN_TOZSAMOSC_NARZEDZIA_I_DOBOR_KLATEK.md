# 36. Koopman i rozszerzone równanie Langevina: tożsamość narzędzia, stan narzędzia i dobór klatek

Data: 2026-09-20. Status: **projekt plus kod plus pierwsze pomiary lokalne; brak wyniku na detektorze.**
Rozwija dokumenty [25](25_DIAGNOSTYKA_LATENTU_I_AKTYWNA_KURACJA.md) i [27](27_PROTOKOL_SZYBKICH_BADAN_ACTIVE_CURATION_GLE.md).
Kod: `pipeline/koopman_langevin/`. Wyniki: `results/koopman_langevin_smoke/`. Wybrane klatki: `curated/koopman_langevin/`.

## 0. Czego ten dokument NIE twierdzi

Nie ma tu dowodu, że równanie Langevina poprawia detektor narzędzia. Takiego dowodu nie da się dać z góry;
rozstrzyga go wyłącznie trening detektora przy równym budżecie klatek. Dotychczasowa kampania statyczna
(`35_KAMPANIA_STATIC36_EDEN.md`, 36 biegów) pokazuje skalę możliwego efektu: wszystkie selektory mieszczą się
w około pół punktu mAP od losowania (budżet 2500: losowanie 0,6925, `uniform_time` 0,6899, pokrycie DINO 0,6971,
hybryda 0,6983; rozrzut między ziarnami około 0,002). Nowy selektor trzeba oceniać na tym tle.

## 1. Co zostało zbudowane

| Moduł | Co robi | Etykiety |
|---|---|---|
| `tokens.py` | cechy DINOv2 dla każdego kawałka obrazu (siatka 16 x 16), stała losowa projekcja bez dopasowania | nie czyta; kopiuje je obok tylko dla walidacji |
| `koopman_tool.py` | rozkład Koopmana (DMD) okna filmu na część wolną (scena) i resztę (to, co się rusza); mapa ruchu, położenie, rozrzut i opis wyglądu obiektu; trzy proste kontrole w tym samym przebiegu | tylko w funkcji `validate` |
| `track_dynamics.py` | Koopman (EDMD) i rozszerzone równanie Langevina na torze narzędzia `[cx, cy, log w, log h]`; niezmiennik punktu obrotu; wyniki na klatkę | tor z GT służy do pomiaru fizyki ruchu; docelowo tor z detektora |
| `latent_langevin.py` | rozszerzone równanie Langevina na latencie całej klatki; wynik części usuwalnej zależny od już wybranych klatek | nie czyta |
| `select.py` | wybór klatek rundami; pliki wariantów w formacie istniejącego trenera | nie czyta do rankingu |
| `synthetic_selection_benchmark.py` | kontrolowany test mechanizmu na danych o znanej prawdzie | nie dotyczy |
| `tests/test_koopman_langevin.py` | 5 testów na CPU; razem z istniejącymi 24 testami projektu przechodzą | |

Jedyna zmiana w istniejącym kodzie: `pipeline/active_learning/acquisition.py` dostał strategię
`coverage_reducible` (pole `reducible_gain`), 5 linii; 18 testów tego modułu przechodzi.

## 2. Równanie

Postać ta sama co w projekcie grantowym, bez wymuszenia zewnętrznego, bo film z operacji nie ma wejścia znanego
z góry:

    z_{t+1} = z_t + D(z_t, v_t, a_t) + JP(z_{t-1}, z_{t-2}, ...) + N(dawne reszty) + e_{t+1}

`D` to dryf z bieżącego stanu, prędkości i przyspieszenia; `JP` to jądro pamięci po wcześniejszych krokach;
`N` to przewidywalna część dawnych reszt; `e` to reszta. Wszystkie człony to regresje grzbietowe dopasowane na
INNYCH operacjach niż oceniana klatka (podział po grupach operacji), siła regularyzacji wybierana wewnątrz grup
dopasowania.

Twierdzenie porządkujące (dowiedzione w Lean 4 w repozytorium grantu,
`F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\DiagnostykaInnowacji.lean`): błąd dowolnego
predyktora z historii to zaskoczenie (nieusuwalne) plus część usuwalna. Wielkość reszty ich nie rozróżnia.

**Poprawka znaleziona w tej pracy.** Twierdzenie mówi, że różnica OCZEKIWANYCH błędów dwóch predyktorów nie zawiera
zaskoczenia. Dla pojedynczej klatki to nie wystarcza:

    |z-h1|^2 - |z-h2|^2 = |m-h1|^2 - |m-h2|^2 + 2 <e, h2-h1>

Ostatni człon niesie zaskoczenie tej jednej klatki. Pierwsza wersja wyniku (różnica błędów) kupowała w teście
syntetycznym 44 procent klatek szumowych. Wersja obowiązująca nie używa zrealizowanej następnej klatki wcale:

    reducible_gain(i) = |h_known(i) - h_pool(i)|^2 / (err_pool(i) + mediana err_pool)

`h_pool` to model dopasowany na całej puli (na innych operacjach), `h_known` to ten sam model dopasowany tylko na
przejściach w sąsiedztwie już wybranych klatek. Wynik zmienia się co rundę, więc jest kryterium aktywnym.

## 3. Wyniki lokalne (RTX 3070, CPU)

### 3.1. Koopman na cechach obrazu NIE znajduje narzędzia lepiej niż zwykła różnica klatek

8 filmów, 3653 klatki, krok 5 klatek. Trafienie szczytu mapy w ramkę narzędzia (GT tylko do oceny):

| metoda | trafienie | błąd położenia (mediana, piksele) | korelacja prędkości x / y |
|---|---|---|---|
| los | 0,110 | | |
| Koopman (DMD), część bez modów wolnych | 0,353 | 120 | 0,02 / 0,03 |
| mediana w czasie | 0,364 | 119 | 0,02 / 0,07 |
| średnia w czasie | 0,358 | 120 | 0,02 / 0,06 |
| różnica klatek | 0,398 | 122 | 0,02 / 0,07 |

Wniosek: rozkład Koopmana trafia w narzędzie trzy razy częściej niż los, ale nie lepiej niż trywialne kontrole,
a prędkości narzędzia z mapy ruchu nie da się odczytać (korelacja około zera, także po wygładzeniu i centroidzie
wokół szczytu: błąd 135 pikseli). Rusza się nie tylko narzędzie (oko, drugie narzędzie, odblaski), a siatka
16 x 16 ma oczko około 45 pikseli przy typowym ruchu 10 pikseli na krok. **Stan narzędzia trzeba brać z ramek
detektora, nie z samych cech obrazu.**

### 3.2. Fizyka ruchu narzędzia z torów GT (170 filmów, 167 grup operacji)

| krok (klatki) | okna | stała prędkość | dryf | + pamięć | + człon szumu | autokorelacja reszty |
|---|---|---|---|---|---|---|
| 1 | 425 552 | -168,8 | 20,8 | 22,5 | 22,4 | -0,001 |
| 5 | 82 675 | -169,9 | 25,9 | 28,1 | 27,8 | -0,003 |
| 15 | 25 969 | -180,7 | 28,5 | 29,3 | 28,8 | 0,000 |

Liczby to procent przewidzianego ruchu względem założenia "narzędzie stoi", na operacjach niewidzianych przy
dopasowaniu. Wnioski: równanie przewiduje 22 do 29 procent następnego kroku narzędzia; pamięć dodaje 1 do 2
punktów; człon szumu nic (reszta jest już biała); założenie stałej prędkości jest dużo gorsze niż brak ruchu,
bo kroki ramki są przeciwskorelowane (drganie ramek i ręki). Ruch narzędzia to w około trzech czwartych zaskoczenie.

Widmo Koopmana na torze: 4 mody wolne (moduł powyżej 0,98) leżą na położeniu (72 procent ładunku) i rozmiarze
(24 procent), na prędkości 0,5 procent; mody szybko gasnące leżą na prędkości (74 procent). Czyli wielkości
zachowywane to "gdzie jest i jak duże", a stan szybko zapominany to prędkość. To wynik zgodny z intuicją autora,
ale skromny: położenie jest trwałe z natury.

Niezmiennik punktu obrotu (oś narzędzia celuje w nacięcie): odległość osi od punktu obrotu 20 pikseli przy
ruchu końcówki 47 pikseli, stosunek 0,45. To słaby niezmiennik w tej postaci, bo oś wyznaczam z narożnika ramki.
Do sprawdzenia z osią z wielokąta `cystotome_seg`.

### 3.3. Równanie Langevina na latencie klatki

Pula kampanii Static36: 57 073 klatki uczące, 55 683 okna, 117 grup. Umiejętność na niewidzianych operacjach
15,7 procent (regularyzacja wybrana 0,1 w każdym foldzie). Na 8 filmach umiejętność była ujemna (-5; pomiar sprzed automatycznego
doboru regularyzacji), więc do tego równania trzeba dziesiątek operacji.

**Co równanie uznaje za zaskoczenie w prawdziwych klatkach (8 filmów, GT tylko do oceny).** Korelacja rang
zaskoczenia z prędkością narzędzia 0,39; w górnych 10 procentach zaskoczenia narzędzie rusza się 17,6 piksela na
krok wobec 8,2 w reszcie; pojawienie albo zniknięcie narzędzia jest tam 28 razy częstsze (2,8 wobec 0,1 procent).
**To nie jest szum etykiet, tylko szybki ruch i wejście narzędzia, czyli rzeczy, których detektor może potrzebować.**

### 3.4. Test mechanizmu na danych syntetycznych (10 ziaren, budżet 240 z 5760 klatek)

| selektor | błąd modelu zadania na klatkach uczalnych | udział klatek szumowych | lepszy od losowania (ziarna z 10) |
|---|---|---|---|
| losowanie | 2,04 | 0,12 | |
| `uniform_time` | 1,22 | 0,11 | 8 |
| pokrycie | 2,41 | 0,18 | 3 |
| pokrycie + wielkość reszty | 3,29 | 0,74 | 2 |
| pokrycie + część usuwalna | 0,90 | 0,13 | 10 |
| `uniform_time` z wetem zaskoczenia | 0,80 | 0,06 | 10 |
| pokrycie + część usuwalna z wetem | 0,79 | 0,09 | 10 |

Wnioski: dobór po wielkości reszty szkodzi (kupuje 74 procent szumu), wersja usuwalna usuwa tę szkodę i wygrywa
z `uniform_time`, weto zaskoczenia pomaga najbardziej. Żaden selektor oparty na ruchu nie znajduje rzadkiego
reżimu, który różni się wyglądem, a nie ruchem (udział rzadkich 0,01; błąd na rzadkich rośnie).

**Założenie, którego ten test nie sprawdza:** że klatki o nieprzewidywalnym ruchu mają etykiety niewarte zakupu.
Pomiar 3.3 sugeruje, że w filmie z operacji to założenie NIE zachodzi. Dlatego weto i dobór po zaskoczeniu idą
do eksperymentu jako dwa przeciwne ramiona, a nie jako rozstrzygnięta rekomendacja.

### 3.5. Co faktycznie wybrały selektory na prawdziwej puli (audyt po fakcie, GT tylko do opisu)

Wspólne pierwsze 500 klatek to `tracking_uniform_time_b500_s42` z kampanii; poniżej kolejne 1000 klatek.

| selektor | mediana prędkości narzędzia | udział szybkich (górne 10 procent puli) | grup | najwięcej z jednej grupy |
|---|---|---|---|---|
| losowanie (kampania, 1500) | 7,5 | 0,12 | 117 | 37 |
| `uniform_time` (kampania) | 7,8 | 0,12 | 117 | 27 |
| pokrycie DINO (kampania) | 11,5 | 0,22 | 117 | 27 |
| `langevin_reducible` | 11,1 | 0,22 | 115 | 54 |
| `langevin_surprise` | 15,3 | 0,33 | 113 | 27 |
| `langevin_veto_coverage` | 10,1 | 0,15 | 116 | 14 |
| `langevin_veto_uniform` | 9,6 | 0,14 | 117 | 12 |

Uwaga: samo pokrycie DINO już kupuje dwa razy więcej szybkich klatek niż losowanie, a dało najwyżej pół punktu mAP.

## 4. Eksperyment rozstrzygający (przygotowany, nieuruchomiony)

Pytanie: czy przy budżecie 1500 klatek któryś z czterech wariantów Langevina daje wyższe mAP na walidacji niż
`uniform_time` i pokrycie DINO z kampanii Static36, przy identycznej recepturze RF-DETR Nano (30 epok, wejście 384,
ziarna treningu 42, 1337, 2024).

- Warianty: `curated/koopman_langevin/tracking_langevin_*_b1500_sel42.json`; pierwsze 500 klatek identyczne
  z `tracking_uniform_time_b500_s42`, więc różni się tylko dobór kolejnego tysiąca.
- Punkt odniesienia (już policzony, budżet 1500): losowanie 0,6818; `uniform_time` 0,6832; pokrycie 0,6838;
  hybryda 0,6821; rozrzut ziaren do 0,002.
- Kryterium zapisane przed biegiem: wariant uznaję za lepszy, gdy wszystkie trzy ziarna są powyżej średniej
  `uniform_time` i średnia przewaga przekracza 0,004 (dwa rozrzuty ziaren). Każdy inny wynik to remis albo porażka.
- Koszt: 4 warianty x 3 ziarna = 12 biegów, około 1,3 godziny H100 na bieg.

**Blokada.** Katalog kampanii leży na tym samym systemie plików co reszta Edenu: wolne 1,2 GB przy limicie 200 GB.
Jeden bieg zapisuje około 240 MB (`best.pth` i `last.pth`), więc 12 biegów się nie zmieści. 8,2 GB wyników
kampanii Static36 na Edenie ma kopię lokalną z manifestem (`results/eden_static36_2026-09-15/DOWNLOAD_MANIFEST.json`).
Niczego nie usuwałem; decyzja o zwolnieniu miejsca należy do autora.

## 5. Etap drugi: tor z detektora (projekt)

Pomiar 3.1 pokazał, że stan narzędzia trzeba brać z detektora. Po każdej rundzie aktywnej pętli detektor przechodzi
po nieoznaczonej puli; jego ramki tworzą tor. `track_dynamics.langevin` daje wtedy na klatkę: przewidywane położenie
z historii toru, resztę, część usuwalną oraz naruszenie niezmiennika punktu obrotu. Kandydaci do anotacji to klatki,
w których detektor łamie fizykę toru w sposób przewidywalny z historii (błąd detektora), a nie klatki, w których
narzędzie naprawdę skoczyło (zaskoczenie). Wymaga predykcji RF-DETR na puli, czyli Edenu; lokalnie `rfdetr` nie jest
zainstalowany.

## 6. Literatura użyta w tej pracy

- Mindermann i in. 2022, Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt,
  arXiv:2206.07137 (wynik usuwalny zamiast wielkości straty).
- Hu i in. 2021, When does loss-based prioritization fail?, arXiv:2107.07741 (dobór po stracie zawodzi przy szumie).
- Billings, Zhu 1994, Nonlinear model validation using correlation tests, doi:10.1080/00207179408921513 (białość reszty).
- Yang i in. 2021, Delving into Deep Imbalanced Regression, arXiv:2102.09554.
- Montero-Manso, Hyndman 2021, doi:10.1016/j.ijforecast.2021.03.004 (modele wspólne zyskują na danych).
- Rozdzielenie tła i obiektu przez DMD (Grosek, Kutz 2014) przywołane z pamięci; do zweryfikowania w bazie przed cytowaniem.
- Precedensy z dokumentu 25 (LRTD, COWAL, aktywne uczenie z Koopmanem) ograniczają nowość; nie sprawdzałem ich ponownie.

## 7. Counter-arguments

- Jeśli zaskoczenie to w praktyce szybki ruch narzędzia, a pokrycie DINO już go kupuje i daje pół punktu, to
  selektor Langevina może nie dać nic ponad pokrycie. To jest najbardziej prawdopodobny wynik.
- Latent całej klatki opisuje scenę, nie narzędzie; przewidywalność 15,7 procent może dotyczyć oka i oświetlenia.
- Widmo Koopmana na torze pokazuje rzecz prawie oczywistą (położenie trwałe, prędkość ulotna); nie jest dowodem
  "tożsamości narzędzia". W zbiorze trackingowym jest jedno narzędzie, więc tożsamości między narzędziami nie da się tu zmierzyć.
- Tory GT zawierają drganie anotacji; 22 procent przewidywalności może być zaniżone albo zawyżone przez sposób anotacji.

## Data gaps

- Brak jakiegokolwiek treningu detektora na nowych wariantach (blokada miejsca na Edenie).
- Brak toru z detektora; etap drugi jest projektem.
- Niezmiennik punktu obrotu liczony z narożnika ramki, nie z osi segmentacji.
- Test syntetyczny opiera się na założeniu, które pomiar 3.3 podważa.
- Jedno ziarno selekcji (42); powtórzenia selekcji dopiero po pierwszym efekcie.
