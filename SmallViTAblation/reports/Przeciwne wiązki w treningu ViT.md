# Znak GELU, a nie wagi, tworzy przeciwne wiązki

## Najważniejszy wniosek

Przeciwne wiązki w naszym małym ViT są rzadkim produktem ubocznym znaku GELU, a nie obwodem wzajemnego znoszenia. W seedzie 3047 pary neuronów MLP, których wkłady do strumienia rezydualnego na tym samym tokenie tworzą kąt ≥150°, nie istnieją przy inicjalizacji (maks. 110°). Pojawiają się w epoce 9, dochodzą do 25 par w epoce 40 i osiągają maksimum 30 par (spośród ok. 1,77 mln par w 6 blokach) w epokach 80–110, a do epoki 200 spadają do 15. Żadna nie przekracza 173°, a skład par się wymienia: z 24 par z epoki 60 do epoki 200 przetrwało 5. Każda taka para łączy niemal równoległe kolumny W2 (≤30°) z przeciwnymi znakami GELU na danym tokenie, bo żadna para kolumn W2 nie jest antyrównoległa (maks. 142°). Pary pojawiają się dopiero po głównym zysku IoU: 84% końcowego VAL IoU model osiąga w epoce 2,7, zanim istnieje jakakolwiek para. W 24-seedowych kampaniach z inną, geometryczną definicją (cos W2 < −0,5) zmiany przeciwstawności nie korelują ze zmianami IoU. Najbardziej skrajna para (neurony 370 i 391 w bloku 6) to dwie kopie powstałe po epoce 100. Ich wkłady dodają się na 94% tokenów i częściowo znoszą tylko na 6%, a wyłączenie pary zmienia IoU o ok. 10⁻⁵. Pozostałe pary T150 to **częściowe kopie**: mają prawie równoległe W2 (≤30°) i podobne wejścia (mediana cos W1 0,75–0,81 w epokach 60–200), ale przesunięte progi, więc na tokenach między progami jeden neuron jest dodatni, a drugi ujemny. **Spadek liczby par po epoce 110 to rozchodzenie się tych częściowych kopii, a nie ich scalanie.** Wszystkie 20 par z epoki 110, które zniknęły do epoki 200, ma kąt W2 > 30° (mediana 28° → 40°), a ich wejścia się oddaliły (cos 0,77 → 0,66). W pełną kopię (cos W1 > 0,9) scaliła się tylko jedna para, 370–391. Literatura z Undermind i Scite nie przeczy żadnej obserwacji: ruch wag po plateau metryki, późne kopie neuronów, ujemny punkt pracy GELU i dryf pewności są dobrze udokumentowane. Nie znaleziono jednak pracy, która śledzi przeciwne pary neuronów per token przez cały trening ani rozkłada przeciwieństwo na geometrię W2 i znak aktywacji. To jest rdzeń nowości. Artykuł może bronić nowej miary i nowego mechanizmu geometrycznego, ale nie funkcji przeciwnych wiązek w detekcji. Przed wysłaniem trzeba zamknąć trzy luki:
- wszystkie wyniki tokenowe opierają się na jednym seedzie i 50 klatkach;
- brakuje modelu zerowego dla liczby par;
- pełna kopia jest tylko jedna (para 370–391, wybrana post hoc); w całej sieci par z cos W1 > 0,9 i kątem W2 < 30° jest 0–3 w każdej epoce.

Konwencje raportu:
- Skróty ścieżek ([P200], [ATLAS] itd.) są rozwinięte w tabeli na końcu.
- „1 seed" oznacza wyłącznie seed 3047. Statystyki tokenowe dodatkowo pochodzą tylko z 50 stałych klatek DIAG50 (30 000 tokenów na blok).
- „24 seedy" oznacza inną definicję przeciwieństwa, epoki 0–60 i głównie ostatni MLP.
- Prace z 2026 r. bez recenzji są oznaczone jako preprinty.
- „Wyliczenie własne" oznacza liczbę policzoną przy pisaniu raportu z danych w notatkach. Nie występuje ona w plikach eksperymentów.

## Dwie definicje przeciwieństwa, których nie wolno mieszać

Wiązka to wkład jednego neuronu ukrytego c na jednym tokenie t: u[t,c] = GELU(z[t,c])·W2[:,c] ∈ ℝ¹⁹². Siła wiązki to |GELU(z)|·‖W2[:,c]‖, a ujemne GELU odwraca jej kierunek ([ATLAS] `Opis\OPIS_EKSPERYMENTU.txt`). W projekcie funkcjonują co najmniej dwie różne „przeciwności":

- **Definicja T150 (tokenowa).** Dwie wiązki z tego samego bloku na tym samym tokenie tworzą kąt ze znakiem ≥150°. Para liczy się raz na blok i checkpoint, jeśli osiągnie próg na co najmniej jednym z 30 000 tokenów. To jest obiekt eksperymentów 0–200 i istnieje tylko dla seeda 3047.
- **Definicja G120 (geometryczna W2).** Kolumny W2 dwóch neuronów mają cos < −0,5 (kąt >120°), niezależnie od aktywacji. Na niej opierają się starsze kampanie 24-seedowe (epoki 0–60, głównie ostatni MLP).

Obok nich występują jeszcze pary funkcjonalne F (przeciwne efekty JVP na krawędziach ramki) i przestarzała definicja sferyczna 3D.

**Zbiory T150 i G120 są rozłączne.** Każda para T150 w epokach 60–200 ma kąt W2 ≤29,98°, więc nigdy nie jest antyrównoległa w W2 (wyliczone z `CANDIDATE_PAIRS_60_200.csv`). Z 1 327 par funkcjonalnych tylko 70 spełnia G120 ([AUDYT] `RAPORT_PL.md`). Każdy wynik poniżej ma podaną definicję i bazę danych:

| Wynik | Definicja | Seedy | Epoki | Próbka |
|---|---|---|---|---|
| Liczba par ≥150°, kąt maksymalny, wymiana tożsamości | T150 | **1 (3047)** | 0–200, 26 checkpointów | wszystkie 294 528 par na blok; 50 klatek TEST (DIAG50) |
| Mechanizm: W2 ≤30° i przeciwne znaki | T150 + kąty W2 | **1** | 0–200 | wynika z wag; niezależne od klatek |
| Para 370–391: historia wag, tokeny, ablacja | T150 | **1, jedna para** | 0–200 | 50 klatek (tokeny); 500 VAL + 500 TEST i 300 par kontrolnych (ablacja) |
| IoU i pewność | — | **1** | 0–200 | 500 VAL (34 puste) + 500 TEST (28 pustych); bootstrap po filmach |
| Wzrost G120, moment pojawienia się względem IoU, korelacje | G120 | **24** | 0–60 | ostatni MLP |
| Pilot kanałów Martinellego | G120 | **3** (3047, 4047, 5047) | 0–60 | wagi: wszystkie pary; tokeny: 10 + 10 klatek |
| Kompensacja funkcjonalna, pruning, przywracanie | F, G120 | **24** | koniec (epoka 60) | 500 VAL + 500 TEST |
| Wygaszanie wiązek, rozkłady kątów | wszystkie wiązki | **1** | 0–50 | 50 klatek |

W notatkach i wcześniejszych materiałach są dwa skróty myślowe, których nie należy przenosić do artykułu:
- **Definicja przeciwnych wiązek.** Notatka literaturowa (`literature_dynamics_calibration_ablation.md`) proponuje definiować przeciwne wiązki jako „neurony o antyrównoległych wektorach wag wyjściowych". To jest błędne dla T150, bo pary T150 mają równoległe W2. Poprawna definicja jest w Twierdzeniu 2.
- **91% tokenów w ogonie GELU.** Notatka o GELU uogólnia ten wynik na „neurony bloku 6", a dotyczy on tylko pary 370–391. Dla populacji ostatniego MLP (24 seedy, epoka 60) obie aktywacje są ujemne w 84,83% przypadków dla par kontrolnych i w 88,39% dla par funkcjonalnych ([FUN24] `RAPORT_WYNIKOW_24SEEDY.md`).

Sześć obserwacji ocenionych w notatce literaturowej z 2026-10-20 ([LIT]) dotyczy definicji G120 i F, nie T150.

## Obserwacje i werdykty literatury

Werdykty oparto na trzech źródłach:
- głębokie wyszukiwania Undermind (workspace 40294fa6…, kilkaset prac, w tym 4 nowe wyszukiwania z 2026-10-20/21);
- pełne teksty ok. 50 prac;
- statystyki Smart Citation ze Scite.

Scite nie pokazuje sprzeciwu wobec prac-kotwic:

| Praca | Wspierające | Kontrastujące | Wspominające | Uwagi |
|---|---|---|---|---|
| Guo 2017 | 8 | 0 | 577 | — |
| Morcos 2018 | 2 | 0 | 55 | — |
| McGrath 2023 | 1 | 0 | 47 | — |
| Fukumizu & Amari 2000 | 2 | 0 | — | 147 zdań cytowań łącznie |
| Minderer 2021 | 1 | 2 | 26 | jedno kontrastujące zdanie pochodzi z pracy o kalibracji w fotopułapkach |
| Martinelli 2025 | 0 | 0 | 1 | jedyne zdanie: Marchetti 2026 |

Zbiorczy werdykt dla każdej obserwacji:

| Obs. | Co zaobserwowano | Definicja i dane | Werdykt literatury | Najbliższe prace |
|---|---|---|---|---|
| A | Cykl życia: 0 par → 2 (ep. 9) → 25 (ep. 40) → 30 (ep. 80–110) → 15 (ep. 200); maks. 173°; wymiana tożsamości | T150; 1 seed | **Prawdopodobnie nowe** na poziomie neuronów; analogie tylko warstwowe i w SNN | Patrawala 2025; Zhu 2026; Tuomi 2026 (preprint) |
| B | Każda para ≥150°: W2 ≤30° i przeciwne znaki GELU; kąt W2 nigdy >142° | T150; 1 seed; z wag | **Składowe potwierdzone, konfiguracja nowa**; kontrast ze wszystkimi opublikowanymi mechanizmami opartymi na przeciwnych wagach wyjściowych | Gerstner 2025; Shvetsov 2026 (preprint); Gurnee 2024; Martinelli 2025; Mahan 2021 |
| C | Przeciwieństwo pojawia się po głównym zysku IoU; brak korelacji po detrendzie | T150 (1 seed) + G120 (24 seedy) | **Zgodne** z teorią ruchu wag po plateau; sam pomiar prawdopodobnie nowy | Kunin; Zhang 2022; Kosson 2024 |
| D | Plateau IoU od ok. ep. 40; pewność rośnie, także na pustych klatkach | 1 seed | Plateau **typowe**; dryf pewności **potwierdzony**; trajektoria fałszywej pewności na pustych klatkach **bez precedensu** | Guo 2017; Mukhoti 2020; Wei 2022; Oksuz 2023; Xie 2026 (preprint) |
| E | Para 370–391 staje się parą kopii w ep. 100–150 | 1 para, 1 seed | **Jakościowo potwierdzone** (stochastic collapse), szczegóły nowe; teza „WD woli dwie połówki" **niepoparta** | Chen 2023; Ziyin; Doimo; Savarese 2019; Xie & Li 2024 |
| F | Znoszenie lokalne (6% tokenów), dodawanie na 94%; ablacja: ΔIoU ok. 10⁻⁵, efekty addytywne | 1 para | Wynik zerowy **zgodny**; metodyka ablacji **niepełna** wobec obecnych standardów | Zhou 2018; Pochinkov 2024; McGrath 2023; Rushing & Nanda 2024 |
| G | G120 dotyczy ok. 55% neuronów ostatniego MLP, ale to nie kanały Martinellego | G120; 24 + 3 seedy | **Zgodne** z przewidywaniem autorów dla regularyzacji; nie występuje nawet „wejście" w kanał | Martinelli 2025; Liang 2019 |
| H | Geometria W2 ≠ funkcja; kompensacja istnieje, ale wyłączanie par nie poprawia IoU | G120, F; 24 seedy | **Częściowo antycypowane** | Casper 2021; Meyes 2019; Ahmad 2026 (preprint) |
| I | „Wygaszanie" wiązek to utrata względnej ważności, nie zera | wszystkie wiązki; 1 seed, ep. 0–50 | **Zgodne** (GELU daje małe ujemne wyjścia, a nie zera) | Shvetsov 2026 (preprint); Li 2023 |

**Żadna obserwacja nie jest sprzeczna z literaturą.** Sprzeczne z literaturą albo przez nią niepoparte są natomiast cztery interpretacje z notatek i raportów:
1. „Kara L2 woli dwie zgodne połówki" jako mechanizm kopii (sekcja o parze 370–391).
2. Skala czasowa równowagi Kossona jako wyjaśnienie okna epok 80–150. Po korekcie batch = 32 skala ta wskazuje ok. 37 epok.
3. Kanały Martinellego jako mechanizm przeciwieństwa.
4. Rachunek „antyrównoległej pary w ogonie GELU" zastosowany do pary 370–391, która ma W2 równoległe.

### Przeciwne wiązki pojawiają się w epoce 9, szczytują przy 30 parach i wymieniają skład

W seedzie 3047 żadna para wiązek nie osiąga 150° od kroku 0 do 6000 (epoka 5,49). Największy kąt rośnie od 110,23° przy inicjalizacji do 145,00° w epoce 2,74 i 147,70° w epoce 5,49. Pierwsze dwie pary pojawiają się w kroku 10 000 (epoka 9,14, bloki 2 i 5). Potem liczba par zsumowana po 6 blokach:
- rośnie do 25 w epokach 40–50;
- osiąga **maksimum 30 par w epokach 80, 90 i 110**;
- spada do 21 w epoce 120 i do **15 w epokach 180–200**.

Źródła: [P200] `Wyniki\Dane\polaczone_0_200\SUMA_BLOKOW_0_200.csv` i [P200] `Wyniki\Raport\RAPORT.html` §3.1.

Progów skrajnych par prawie nie ma:
- **Progu 175° nie osiąga żadna para w żadnym z 26 checkpointów.**
- ≥170° osiąga jedna para, i to dopiero od epoki 170 (170,41°); w epoce 200 ma 173,00°.
- Par ≥160° jest zawsze od 0 do 4.

Bloki zachowują się różnie:
- Blok 6 ma najwięcej par od epoki 21,94, z wyjątkiem epok 160 i 170.
- Blok 3 nie ma żadnej pary (maks. 148,29°).
- Blok 2 ma jedną parę (87, 112) od epoki 9,14 do 70.
- Blok 1 spada z 8 par do 1 po epoce 90.

W obrębie seeda liczba jest wyczerpująca i nie zależy od wyboru klatek: w każdym checkpoincie liczba zaobserwowanych par równa się górnemu ograniczeniu wyliczonemu z samych wag. Potwierdza to niezależnie skrypt rzutowania (sekcja metodyczna): z samych kierunków W2 daje on w epoce 60 dokładnie 7/1/0/1/6/9 par na blok, a w epoce 200 dokładnie 1/0/0/2/5/7, czyli tyle, ile wynosi liczba par T150. Słabości są dwie: to jeden przebieg bez przedziałów ufności, a trajektoria nie jest gładka (30 → 21 między epokami 110 i 120, potem znów 25 w epoce 140).

Przeciwne wiązki są skrajnie rzadkie i coraz rzadziej występują jako dwie aktywne wiązki naraz. W szczycie (epoka 110) wszystkich wystąpień para-token ≥150° jest 162 725, a w epoce 200 jest ich 86 808. To ok. 1,6·10⁻⁶ wszystkich kombinacji para × token w 6 blokach (wyliczenie własne: 6 × 294 528 × 30 000).

Gdy obie wiązki muszą przekraczać 5% najsilniejszej wiązki na tokenie, liczba wystąpień:
- ma szczyt 21 384 już w epoce 21,94;
- wynosi jeszcze 17 934 w epoce 110;
- spada do 2 084 w epoce 140 i 4 249 w epoce 200.

Źródło: [P200] `Wyniki\Dane\polaczone_0_200\COUNTS_0_200.csv`. Te liczby zależą od wyboru 50 klatek, więc są wstępne.

Skład par się wymienia. Przez 26 checkpointów przewijają się **73 różne tożsamości** (blok, neuron, neuron). Do epoki 200 przetrwały:
- 3 z 18 par z epoki 30;
- 5 z 24 par z epoki 60;
- 10 z 30 par z epoki 110.

Najdłużej żyjące pary leżą w bloku 5; para 81–108 jest obecna w 21 z 26 checkpointów (te liczby wyliczono z plików `CANDIDATE_PAIRS*.csv` i `INTERMEDIATE_PAIRS.csv`). W definicji G120 na 24 seedach tożsamości są równie ruchliwe: między krokami 55 000 i 65 640 średnio 299,13 par wchodzi w przeciwstawność, a 340,25 z niej wychodzi ([PRUNE] `analysis\DYNAMICS_INTERPRETATION_PL.md`).

**Werdykt literatury: przebieg nie jest sprzeczny z literaturą, ale nie ma w niej precedensu na poziomie neuronów.** Siedem głębokich wyszukiwań Undermind nie znalazło pracy, która mierzy per token przeciwieństwo wkładów pojedynczych neuronów MLP przez cały trening transformera. Jedno z nich wprost uznało pomiar sumowania i znoszenia wektorów neuronów na tokenie, zwłaszcza w ViT, za lukę ([Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/How%20individual%20MLP%20neuron%20contributions%20add%20up%20and%20cancel%20on%20a%20single%20token%20in%20transformers%20and%20how%20this%20evolves%20in%20training)).

Najbliższe analogie dotyczą innych poziomów:
- **Warstwy w LLM-ach.** Kosinus per token między wkładami sąsiednich warstw wynosi ok. −0,2 (do −0,4) w 5 z 7 rodzin modeli. W OLMo-7B wzorca nie ma po ok. 5% treningu, a pojawia się po ok. 21% ([Patrawala et al., NeurIPS 2025](https://doi.org/10.52202/085713-4643)).
- **Sieci impulsowe.** Stosunek hamowania push–pull rośnie w treningu z 1,06 do 2,0 ([Zhu et al. 2026](https://doi.org/10.1007/s00422-025-01030-4)).
- **Kształt „narastanie–szczyt–częściowy spadek".** Ma precedens w ujemnym punkcie pracy bramek GELU w Pythii, który narasta do szczytu w kroku 8000 i potem częściowo się relaksuje ([Tuomi 2026, preprint](https://arxiv.org/abs/2609.31956)). Ta praca nie śledzi jednak par.

Wymiana tożsamości jest zgodna z dwoma znanymi zjawiskami:
- ciągła rotacja wag przy stałym ηλ ([Kosson et al., ICML 2024](https://arxiv.org/abs/2305.17212));
- anomalna dyfuzja parametrów po zbieżności, w której kwadrat przemieszczenia rośnie jak k^1,075 ([Kunin et al., Neural Computation](https://doi.org/10.1162/neco_a_01626)).

Zastrzeżenie: przegląd opierał się na abstraktach i nie obejmował blogów (LessWrong, Alignment Forum).

### Każda para ≥150° łączy równoległe kolumny W2 z przeciwnymi znakami GELU

Mechanizm wynika z prostej algebry. Jeśli kolumny W2 dwóch neuronów tworzą kąt α, to ich wiązki na tokenie tworzą kąt α przy zgodnych znakach GELU albo 180° − α przy przeciwnych. Żadna para kolumn W2 nigdy nie osiąga 150°. Największy kąt W2 we wszystkich blokach wynosi:

| Epoka | 0 | 9,14 | 50,29 | 200 |
|---|---|---|---|---|
| Maks. kąt W2 | 110,23° | 132,56° | 140,42° | 142,31° |

**Para T150 wymaga więc α ≤ 30° i przeciwnych znaków.** Sprawdzono to bezpośrednio:
- Wszystkie 339 wierszy kandydatów ≥150° z epok 60–200 mają α ≤ 29,98°.
- 15 par z epoki 200 ma α od 7,00° (para 370–391) do 29,89°.
- Dla każdej z nich liczba tokenów ≥150° równa się liczbie tokenów z przeciwnymi znakami i niezerowymi wiązkami.

Raport podsumowuje to tak: „Wszystkie przeciwne wiązki biorą się z prawie równoległych W2 i przeciwnych znaków GELU" ([P200] `RAPORT.html` §3.1). W seedzie 3047 wynik jest mocny, bo wynika z wag w każdym checkpoincie. Na innych seedach dla epok 60–200 go nie sprawdzono.

Warunkiem mechanizmu jest ujemny ogon GELU, a ten jest w naszym modelu typowy dla transformerów.
- **Parametry GELU (obliczone, nie z pracy).** Minimum GELU wynosi ok. −0,170 przy z ≈ −0,752.
- **Para 370–391.** Przy efektywnych biasach b̂ = −1,47 i −1,11 GELU daje ok. −0,10 i −0,15. Te wartości są orientacyjne, bo b̂ = b1 + W1·β (przesunięcie LayerNorm wliczone w bias) nie musi być typową preaktywacją.
- **Populacja ostatniego MLP (24 seedy, epoka 60).** Obie aktywacje pary są ujemne w 84,83% przypadków dla par kontrolnych, a znaki przeciwne występują w 14,19% ([FUN24] `RAPORT_WYNIKOW_24SEEDY.md`).

Literatura potwierdza każdą składową:
- Ułamek ujemnych preaktywacji GELU wynosi 70,5% w MaxViT-Tiny i 89,3% w GPT-nano ([Shvetsov et al. 2026, preprint](https://arxiv.org/abs/2605.17659)).
- Średnia preaktywacji w BERT szybko staje się ujemna w treningu, także z GELU ([Li et al., ICLR 2023](https://arxiv.org/abs/2210.06313)).
- Mediana spoczynkowej preaktywacji bramek GELU w Phi-2 wynosi od −1,115 do −1,568 ([Tuomi 2026, preprint](https://arxiv.org/abs/2609.31956)).
- Reguła znaku (ujemna aktywacja pisze w kierunku −w_out) została opisana dla neuronów bramkowanych ([Gerstner & Schütze 2025](https://arxiv.org/abs/2505.17936); [Gerstner et al. 2026, preprint](https://arxiv.org/abs/2609.18612)).

**Nowa jest konfiguracja: równoległe W2, a przeciwieństwo bierze się tylko ze znaku aktywacji.** Wszystkie opublikowane mechanizmy przeciwieństwa opierają się na przeciwnych wagach wyjściowych:
- antypodalna para neuronów entropii w GPT-2 ma kosinus wag wyjściowych −0,886 ([Gurnee et al., TMLR 2024](https://arxiv.org/abs/2401.12181));
- kanały Martinellego wymagają a_i·a_j < 0 ([Martinelli et al. 2025](https://arxiv.org/abs/2506.14951));
- konstrukcja różnic skończonych używa wag wyjściowych +n i −n ([Mahan et al. 2021](https://doi.org/10.1016/j.neunet.2021.01.007));
- antypodalne pary w toy models of superposition leżą w przestrzeni osadzeń, nie w wagach wyjściowych MLP ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)).

Nasz mechanizm jest strukturalnie inny i możliwy tylko przy aktywacji z niezerowym ujemnym ogonem.

Interpretacja własna, dotąd nieopublikowana: para czyta prawie ten sam kierunek wejścia przy dwóch różnych progach. Działa więc jak dwuwęzłowe rozwinięcie jednej funkcji grzbietowej (ridge function). Poniżej obu progów obie wiązki są małe i ujemne, a powyżej obu dodatnie i sumują się. „Przeciwieństwo" to tylko okno między progami.

Jeden szczegół wymaga sprawdzenia, zanim ta interpretacja trafi do tekstu. Na tokenach z przeciwnymi znakami ujemny jest prawie zawsze neuron 391 (1 718 z 1 789 tokenów), choć to on ma mniej ujemny b̂. Efektywny próg trzeba więc mierzyć w jednostkach rzutu wejścia (b̂/‖w1‖), a nie samym b̂.

Recenzent zauważy też, że przy równoległych W2 kąt 173° jest prawie tautologią, bo możliwe są tylko kąty α albo 180° − α. Dlatego to twierdzenie trzeba przedstawić jako dekompozycję (geometria W2 kontra znak) i dodać model zerowy.

Nie należy twierdzić, że ujemny ogon GELU jest bezużyteczny. W LLM-ach zerowanie ujemnych preaktywacji w 1% neuronów na warstwę mniej więcej podwaja perplexity ([Kong et al., ICLR 2026](https://arxiv.org/abs/2509.24198)).

### Przeciwieństwo przychodzi po zysku IoU, a pewność rośnie dalej

**Seed 3047, definicja T150.** Pary pojawiają się, gdy model ma już większość swojej jakości:

| Epoka | VAL IoU | Udział końcowego VAL IoU (0,7654) | Pary T150 |
|---|---|---|---|
| 2,74 | 0,6420 | 84% | 0 (maks. kąt 145°) |
| 9,14 | 0,7207 | 94% | 2 (pierwsze) |
| 21,94 | 0,7452 | 97% | 13 |

Od epoki 40 do 200 IoU waha się w wąskim paśmie: VAL 0,748–0,775, TEST 0,736–0,762. Przedziały bootstrapowe po filmach nakładają się we wszystkich epokach; w epoce 200 VAL wynosi 0,765 [0,730; 0,796] ([P200] `RAPORT.html` §3.2). Para ≥170° pojawia się w epoce 170 bez żadnej zmiany IoU.

**24 seedy, definicja G120.** Najmocniejsze, wieloseedowe wsparcie tej tezy pochodzi z definicji G120:
- We wszystkich 24 seedach największy przyrost IoU i score przypada na kroki 0–300.
- W krokach 0 i 300 żaden neuron nie ma partnera G120.
- Największy przyrost liczby neuronów z przeciwnikiem przypada na kroki 10 000–16 000 (20/24 seedów) albo 16 000–24 000 (4/24).
- Po kontroli seeda i przedziału wszystkie sześć zaplanowanych korelacji (pierwsze różnice lub tempa; opóźnienia −1, 0, +1) mają przedziały ufności obejmujące zero.
- Przy opóźnieniu 0: r = 0,0704 [−0,0400; 0,1613] dla różnic i r = 0,0101 [−0,0187; 0,0380] dla temp.
- W 108 analizach 8 przedziałów wyklucza zero, bez korekty na wielokrotność.

Źródła: [PRUNE] `RAPORT_KONCOWY_PL.md` i [PRUNE] `analysis\DYNAMICS_INTERPRETATION_PL.md`. Na T150 przenosi się to tylko przez analogię.

**Literatura mówi, że plateau metryki nie jest punktem stacjonarnym.** Po zbieżności parametry dyfundują anomalnie ([Kunin et al.](https://doi.org/10.1162/neco_a_01626)), a wagi nie zbiegają do punktów stacjonarnych, choć strata się stabilizuje ([Zhang et al., ICML 2022](https://arxiv.org/abs/2110.06256)).

Pod AdamW z weight decay wagi dochodzą do równowagi rotacyjnej ze stałą aktualizacją kątową sqrt(2ηλ(1−β1)/(1+β1)) na krok ([Kosson et al., ICML 2024](https://arxiv.org/abs/2305.17212)). Dla nas daje to ok. 1,6·10⁻³ rad na krok. Ta wartość nie zależy od batcha.

**Korekta skali czasowej:**
- Trening używał batcha 32, a nie 64–128.
- Dla ηλ = 5·10⁻⁴ × 0,05 = 2,5·10⁻⁵ skala równowagi 1/(ηλ) wynosi ok. 40 000 kroków.
- To ≈ 40 000·32/35 000 ≈ **37 epok**.

Ta skala zbiega się więc z plateau IoU około epoki 40 i z końcem wzrostu liczby par T150 (25 par w epoce 40,23). Nie zbiega się z oknem epok 80–150. **To przybliżone, niezweryfikowane oszacowanie.** Wzory Kossona zakładają wagi niezmiennicze względem skali, a fc1 i fc2 w blokach pre-LN są takie tylko częściowo. Dla wag, które nie są niezmiennicze, nie istnieje też wzór zamknięty dla AdamW.

Jeśli oszacowanie jest trafne, to szczyt, spadek i powstanie kopii przypadają ok. 2,7–4,1 skali czasowej po starcie (epoka 100 to 109 375 kroków, epoka 150 to 164 063). Normy powinny wtedy być już blisko równowagi. Te zjawiska lepiej więc opisuje dynamika stanu ustalonego (stała rotacja plus szum) niż przejściowe dochodzenie do równowagi.

Jest jeden wyjątek: norma W2 neuronu 370 rośnie 3,3× między epokami 100 i 200, co nie pasuje do równowagi wag niezmienniczych. Pasuje natomiast do przypadku z odśrodkową składową gradientu, który Kosson opisuje dla wag zależnych od skali. To hipoteza do sprawdzenia przez logowanie norm.

Recenzent może zapytać, czy plateau przy stałym LR nie ukrywa postępu. Dłuższa faza dużego kroku daje lepszy wynik testowy po obniżeniu LR, choć strata już stała ([Andriushchenko et al., ICML 2023](https://doi.org/10.48550/arXiv.2210.05337); [Chen et al. 2023](https://doi.org/10.1088/1742-5468/ad6429)).

**Pewność rośnie dalej, mimo że IoU stoi.** Między epokami 50 i 200 średni score:

| Klatki | Zbiór | Epoka 50 | Epoka 200 |
|---|---|---|---|
| Z narzędziem | VAL | 0,863 | 0,899 |
| Z narzędziem | TEST | 0,862 | 0,887 |
| Puste | VAL (n = 34) | 0,629 | 0,675 |
| Puste | TEST (n = 28) | 0,541 | 0,579 |

Trend na pustych klatkach nie jest monotoniczny: na VAL wynosi 0,722 w epoce 110 i 0,624 w epoce 120.

Literatura dobrze dokumentuje wzrost pewności po nasyceniu trafności:
- **Guo et al.** W ResNet-110 na CIFAR-100 testowe NLL rośnie, choć błąd spada z 29% do 27% ([Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)).
- **Mukhoti et al.** Entropia predykcji spada także dla błędnych przykładów, a ECE rośnie po epoce 150 ([Mukhoti et al., NeurIPS 2020](https://arxiv.org/abs/2002.09437)).
- **Wei et al.** Norma logitów rośnie także na danych OOD, z ok. 6 do 10–11 ([Wei et al., ICML 2022](https://doi.org/10.48550/arXiv.2205.09310)).
- **Minderer et al.** Dłużej trenowane modele są trafniejsze i gorzej skalibrowane ([Minderer et al., NeurIPS 2021](https://arxiv.org/abs/2106.07998)).
- **Oksuz et al. (jedyny precedens detektorowy).** Faster R-CNN trenowany 36 zamiast 12 epok ma AP 25,5 zamiast 24,8, ale mECE 18,7 zamiast 16,4 ([Oksuz et al., CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.00894)).
- **Xie et al.** Minimum ECE przypada wcześnie, a potem ECE rośnie, także w ViT-B/32 ([Xie et al. 2026, preprint](https://arxiv.org/abs/2610.12048)).

**Nie znaleziono pracy, która śledzi epoka po epoce fałszywą pewność detektora na klatkach bez obiektu.** To jest drobny, ale realny wkład.

Pewność na pustych klatkach to sigmoid maksymalnego logitu spośród 600 tokenów, czyli statystyka ekstremalna. Różnica między klatkami z narzędziem i pustymi zmienia się na VAL niewiele, z 0,234 do 0,224 (wyliczenie własne). Sugeruje to dryf skali logitów, a nie utratę rozróżniania, co trzeba wykazać przez AUROC obecności i skalowanie temperaturą. Score nie jest skalibrowany, a ECE nie mierzono. Tego wyniku nie należy przyczynowo łączyć z przeciwnymi wiązkami.

### Najbardziej przeciwna para to dwie kopie, nie antagoniści

Para 370–391 z bloku 6 jest jedyną parą ≥170°. Jako para T150 występuje dopiero od epoki 140. Jej historię wag zmierzono w całości ([P200] `Wyniki\Dane\znoszenie_370_391\PARA_WAGI_0_200.csv`):

| | Do epoki 100 | Epoka 150 | Epoka 200 |
|---|---|---|---|
| Kąt W2 | 72–93° (niezależne neurony) | 15,02° | 7,00° |
| Kosinus W1 (po wliczeniu LayerNorm) | ≤ 0,31 | 0,903 | 0,964 |
| Normy W2 (370 / 391) | 0,647 / 0,869 (ep. 100) | — | 2,138 / 1,822 (3,3× i 2,1×) |
| b̂ (370 / 391) | −0,815 / −0,628 (ep. 100) | — | −1,473 / −1,106 |

Neurony zbliżają się do siebie między epokami 100 i 150. W epoce 200:
- obie wiązki leżą razem w ujemnym ogonie GELU na **91% tokenów** (27 150 z 29 896);
- korelacja aktywacji wynosi 0,80, wobec 0,03 w epoce 60.

**Znoszenie jest prawdziwe, ale lokalne i małe** ([P200] `RAPORT.html` §3.3). κ to indeks znoszenia energii (2 = wiązki równe i zgodne, 1 = niezależne, 0 = pełne znoszenie):

| Tokeny | Udział | Kąt | κ | Udział pary w wyjściu MLP (mediana) |
|---|---|---|---|---|
| Zgodne znaki | 94% (28 107) | 7° | 1,80 | 1,6% |
| Przeciwne znaki | 6% (1 789) | 173° | 0,54 | 0,56% |
| Przeciwne znaki, obie wiązki aktywne | 269 tokenów | 173° | 0,19 | 0,32% |

Tokeny z przeciwnymi znakami rzadko leżą w ramce narzędzia: 1,3% wobec średnio 4,3%.

**Korekta interpretacji ablacji.** Para wyłączana w ablacji (A1 w notatkach) i para scalająca się (D3) to te same neurony 370 i 391. Ich kolumny W2 są prawie **równoległe** (7°), nie antyrównoległe. Gdy oba neurony siedzą w ujemnym ogonie GELU, oba piszą wzdłuż −Ŵ2, więc ich stałe offsety się **dodają**. Zerowanie obu usuwa sumę tych offsetów, orientacyjnie 0,10–0,15·‖W2‖ na neuron przy b̂ jako punkcie pracy. Spekulatywny rachunek z notatki literaturowej o „antyrównoległej parze w ogonie GELU", której offsety miałyby się znosić, tej pary nie dotyczy.

Dla dwóch równoległych kopii przewidywanie jest proste: efekt łączny ≈ suma efektów pojedynczych, więc współczynnik interakcji ≈ 1 + b/a (a ≥ b to efekty pojedyncze). Przy medianie stosunku sił wiązek 0,66 daje to 1,66. **Zmierzony współczynnik wynosi 1,66 (1,658).** To zgodność, nie dowód (wniosek własny), bo obie wielkości liczy się inaczej. Mediany dla kontroli wynoszą 1,07 (losowe), 1,24 (podobna siła) i 1,29 (najbardziej równoległe W2); para leży na 97., 99. i 86. percentylu tych grup. Współczynnik spadał poniżej 1 tylko w epokach 60, 80 i 90, zanim neurony stały się kopiami. „Najbardziej przeciwna" para jest więc funkcjonalnie jedną z najbardziej zgodnych.

Efekt detekcyjny jest znikomy. Ablacja obu neuronów zmienia IoU:
- na VAL o −1,6·10⁻⁵ [−5,4; +2,2]·10⁻⁵;
- na TEST o +3,1·10⁻⁵ [−2,1; +10,7]·10⁻⁵.

Maksymalna zmiana IoU na pojedynczej klatce wynosi 0,0080, a token-zwycięzca zmienia się w 1 z 1000 klatek. Na tle 300 par kontrolnych para leży na 20.–34. percentylu na VAL i na 75.–77. na TEST. Pojedynczo neurony 370 i 391 zajmują 253. i 341. miejsce z 768 pod względem wpływu na logity ([P200] `Wyniki\Dane\znoszenie_370_391\`).

Jedno drobne odstępstwo: dla samego neuronu 370 przedział ufności na VAL wyklucza zero, ale efekt jest rzędu 10⁻⁵.

**Literatura potwierdza wzór, choć nie te szczegóły.** Najlepszym precedensem jest stochastic collapse ([Chen, Kunin, Yamamura & Ganguli, NeurIPS 2023 / JSTAT 2024](https://doi.org/10.1088/1742-5468/ad6429)): szum SGD przyciąga dynamikę do zbiorów identycznych neuronów (te same wagi wejściowe, bias i wagi wyjściowe). W tej pracy:
- VGG-16 i ResNet-18 trenowano z GELU i wd 5·10⁻⁴;
- skupiska bliskich kopii pojawiają się po 10⁵–10⁶ krokach;
- dłuższa faza dużego LR daje mniej niezależnych neuronów, nawet gdy strata i trafność już stoją.

Inne prace wspierają ten obraz:
- Przy dostatecznie silnym weight decay i symetrii permutacyjnej punkt środkowy (dwa identyczne neurony) ma niższą stratę niż dwa różne neurony ([Ziyin, ICML](https://arxiv.org/abs/2309.16932)).
- Klony pojawiają się tylko w sieciach regularyzowanych i dopiero po zerowym błędzie treningowym ([Doimo et al., J. Stat. Mech. 2023](https://doi.org/10.1088/1742-5468/aceb4f)).
- W gradient flow podprzestrzeni symetrii nie da się osiągnąć w skończonym czasie, więc oczekiwane są bliskie kopie, nie dokładne ([Şimşek et al., ICML 2021](https://arxiv.org/abs/2105.12221)).

Jeden szczegół kontrastuje z literaturą: u nas normy rosną w trakcie scalania. W preprincie o kondensacji napędzanej weight decay kolejność jest odwrotna, czyli najpierw kompresja amplitud, potem kondensacja kątowa ([Xu et al. 2026, preprint](https://arxiv.org/abs/2610.04533)).

Interpretacja z raportu P200, że „kara L2 woli dwie zgodne połówki", jest w połowie niepoparta:
- Dla ReLU koszt L2 po przeskalowaniu równa się normie ℓ1 wag wyjściowych. Podział neuronu na dwie zgodne połówki jest więc kosztowo neutralny, a znoszenie się przeciwnych wag kosztuje ([Savarese et al., COLT 2019](https://arxiv.org/abs/1902.05040)).
- AdamW nie jest L2. Jego ukryty bias to ograniczenie ℓ∞ rzędu 1/λ = 20 ([Xie & Li, ICML 2024](https://arxiv.org/abs/2404.04454)), daleko powyżej naszych wag.

Bezpieczne sformułowanie brzmi: weight decay karze kosztowne znoszenie i przy symetrii permutacyjnej faworyzuje identyczne jednostki. Szum przy stałym LR jest równie dobrym kandydatem na przyczynę.

Ablacja zerowa jest akceptowana jako obserwacja, ale nie jako dowód nieistotności:
- Pojedyncze jednostki rzadko zmieniają metrykę zbiorczą. W AlexNet trafność spada z 50,6% do 50,4%, choć dla pojedynczych klas nawet o 50% ([Zhou et al. 2018](https://arxiv.org/abs/1806.02891)).
- W ViT ablacje zero, mean i peak nie różnią się istotnie ([Pochinkov et al. 2024](https://doi.org/10.48550/arXiv.2408.17322)).
- Dalsze warstwy kompensują, ze współczynnikiem 0,69 w Chinchilli ([McGrath et al. 2023](https://doi.org/10.48550/arXiv.2307.15771)).
- Skalowanie LayerNorm odpowiada za ok. 30% bezpośredniego efektu głowicy ([Rushing & Nanda 2024](https://doi.org/10.48550/arXiv.2402.15390)).
- Wartość zastępcza może „oszukiwać" dalsze warstwy ([Li & Janson 2024](https://doi.org/10.48550/arXiv.2409.09951)).

W tej literaturze „redundantne" oznacza zapasowe jednostki typu OR, które dają efekt nadaddytywny. U nas efekty są addytywne, więc właściwy termin to „addytywne wkłady równoległych kopii".

### Pary T150 to częściowe kopie, a późny spadek to ich rozchodzenie się

*Dopisane przez koordynatora 2026-10-21 po napisaniu raportu. Pomiar z wag checkpointów epok 60–200, seed 3047. Skrypty: [P200] `Wyniki\Code_Interpretujacy_Wyniki\kopie_neuronow.py`, `wejscia_par_t150.py`, `los_par_t150.py`; dane: `Wyniki\Dane\pary_epoki_60_200\WEJSCIA_PAR_T150.csv`.*

**Pełnych kopii w sieci prawie nie ma.** Pełną kopią nazywamy parę z cos W1 > 0,9 (W1 złożone z wzmocnieniem LayerNorm i wycentrowane) i kątem W2 < 30°. Wśród ok. 1,77 mln par w 6 blokach takich par jest:

| Epoka | Pełne kopie (b1–b6) | Pary „kanałowe” (cos W1 > 0,9, cos W2 < −0,5) |
|---|---|---|
| 60 | 3 (wszystkie w bloku 1) | 0 |
| 70–90 | 2, 2, 1 (blok 1) | 0 |
| 100–140 | 0 | 0 |
| 150–200 | 1 (tylko 370–391, blok 6) | 0 |

**Pary T150 to jednak częściowe kopie.** W każdej epoce 60–200 mediana cos W1 par ≥150° wynosi 0,75–0,81, a niemal wszystkie mają cos W1 > 0,5 (np. 14 z 15 w epoce 200). Przy kącie W2 ≤ 30° oznacza to dwa neurony patrzące na podobny wzorzec, piszące w podobnym kierunku, ale z przesuniętymi progami. Na tokenach, których projekcja leży między tymi progami, jeden neuron jest w dodatniej, a drugi w ujemnej części GELU, i wtedy wiązki są przeciwne. To ilościowo potwierdza dla wszystkich par opis „szczeliny między dwoma progami na prawie tym samym kierunku wejścia”. Kontrast z pilotem Mar25l nie jest sprzecznością: tam wejścia par przeciwnych były losowe (cos ≈ 0,07), ale to była definicja G120 (antyrównoległe W2), czyli inna populacja par.

Wyjątkiem jest jedna para z cos W1 ≈ −0,67, obecna od epoki 80. Ma przeciwne wejścia i równoległe wyjścia, czyli konstrukcję typu |x| = ReLU(x) + ReLU(−x) z Toy Models of Superposition. To pojedynczy przypadek, wart wzmianki, ale nie uogólnienia.

**Co się stało z parami po szczycie.** Spośród 30 par z epoki 110 do epoki 200 przetrwało 10, zniknęło 20, a 5 jest nowych:

| Grupa | cos W1 (ep. 110 → 200) | kąt W2 (ep. 110 → 200) | \|Δb̂\| (ep. 110 → 200) | kąt W2 > 30° w ep. 200 |
|---|---|---|---|---|
| przetrwały (10) | 0,80 → 0,78 | 23,4° → 26,7° | 0,13 → 0,14 | 0/10 |
| zniknęły (20) | 0,77 → 0,66 | 28,1° → 40,3° | 0,05 → 0,17 | **20/20** |

Pary znikają więc dlatego, że ich wyjścia się rozchodzą (kąt W2 przekracza 30°, a wtedy kąt wiązek nie może osiągnąć 150°), a wejścia i progi się różnicują. Jest to przeciwne do hipotezy „pary stają się kopiami”. Zastrzeżenia:
- część par w epoce 110 leżała blisko granicy 30° (25–29,6°), więc próg wzmacnia efekt, choć mediana przesunięcia (12°) nie jest marginalna;
- to jeden seed.

**Bezpieczna wersja tezy:** w późnym treningu częściowo zduplikowane neurony przeważnie się różnicują. Tylko pojedyncza para scala się w pełną kopię i to ona daje najostrzejszy kąt.

### Przeciwstawność W2 jest powszechna, ale nie jest kanałem ani mechanizmem detekcji

**24 seedy, definicja G120.** Przy inicjalizacji żaden neuron ostatniego MLP nie ma partnera z cos W2 < −0,5. Średnia liczba neuronów z partnerem rośnie tak:

| Krok | 0 i 300 | 10 000 | 16 000 | 65 640 (epoka 60) |
|---|---|---|---|---|
| Neurony z partnerem | 0 | 92,17 | 210,50 | 424,04 z 768 (55,21%) |

Na końcu odpowiada to 739,83 parom, czyli 0,251% wszystkich par ([FUN24] `RAPORT_WYNIKOW_24SEEDY.md`; [PRUNE] `RAPORT_KONCOWY_PL.md`). Liczba silnie zależy od progu. W przebiegu gęstym (806 checkpointów) przy końcu treningu wynosi:
- 87,37% neuronów przy cos < −0,4;
- 53,125% przy cos < −0,5;
- 17,188% przy cos < −0,6.

Seed tego przebiegu nie jest podany; prawdopodobnie to seed 47 ([WSP] `przeciwne_reflektory\RAPORT.md`). Przeciwstawność G120 skupia się w ostatnim bloku. W seedzie 3047 jest tam 655 par, wobec 39/6/32/56/201 w blokach 1–5. Dla kontrastu cała populacja T150 w epoce 60 to 24 pary.

**Pilot kanałów Martinellego (3 seedy, epoka 60, ostatni blok) odrzucił hipotezę kanałów w 3/3 seedach.** Wyniki zestawia tabela:

| Wielkość | Pary przeciwne | Pary losowe | Kryterium kanału |
|---|---|---|---|
| Mediana podobieństwa wejść c_in_eff | 0,067–0,078 | 0,059–0,078 | ≥ 0,9 |
| Pary bliskie kanałowi (c_in_eff > 0,9, c_out < −0,5) | 0 | — | — |
| Mediana κ (TEST) | 0,975–1,000 | — | ≤ 0,3 |
| Błąd przybliżenia GLU | 0,61–0,66 | 0,68–0,73 | ≤ 0,2 |
| Największa norma parametrów pary | ok. 3 | — | dosłownie > 10³ |

κ bliskie 1 oznacza, że wkłady się nie znoszą. Pary podobne na wejściu (c_in_eff > 0,5) są głównie zgodne, nie przeciwne: w seedzie 3047 zgodnych jest 441, przeciwnych 17. Źródło: [MAR] `Wyniki\Dane\eden\redukcja_pilot\HYPOTHESES_AND_SUMMARY.json` i `POPULATION.csv`.

Ten wynik zgadza się z oczekiwaniem samych autorów. Martinelli i wsp. piszą, że regularyzacja, podobnie jak skończony krok, zatrzymuje dynamikę wewnątrz kanału, a pomiar kanałów w transformerach nazywają pracą w toku ([Martinelli et al. 2025](https://arxiv.org/abs/2506.14951)). Z regularyzacją wszystkich parametrów rozbieganie się norm do nieskończoności jest i tak wykluczone, choć Liang i wsp. dowodzą tego dla specjalnych kar ([Liang et al., SIAM J. Optim.](https://doi.org/10.1137/19m1299074)).

Informacyjne jest to, że brakuje nawet **wejścia** w kanał, czyli zbiegających się wejść przy przeciwnych wyjściach. Martinelli i wsp. podają, że SGD i Adam łatwo do niego docierają. Uczciwe zastrzeżenia:
- ich kanały zdefiniowano dla skalarnego wyjścia, bez weight decay i dla gradient flow;
- uogólnienie a_i·a_j < 0 na antyrównoległe kolumny W2 jest nasze;
- pilot ma flagę `smoke`, 10 + 10 klatek i okno 0–60, więc nie obejmuje późnych kopii.

**Geometria W2 jest w dużej mierze oderwana od funkcji:**
- Tylko 70 z 1 327 par funkcjonalnych spełnia G120, a 298 ma nieujemny kosinus W2.
- Śledzono te same 16 szybko zmieniających się par od kroku 0 do 65 640:

  | Wielkość | Krok 0 | Krok 65 640 |
  |---|---|---|
  | Kosinus W2 | 0,028 | −0,560 |
  | Współaktywacja | 86,17% | 17,73% |
  | Indeks kompensacji q | 0,797 | 0,820 |

  Wagi stają się bardziej przeciwne, a kompensacja się nie wzmacnia ([PRUNE] `analysis\DYNAMICS_INTERPRETATION_PL.md`).
- Kompensacja funkcjonalna istnieje: P1 = 0,160 [0,132; 0,190], dodatnie w 24/24 seedach ([FUN24]).
- Usuwanie całych rodzin par nie poprawia IoU, a obniża score w 24/24 seedach ([PRUNE] `RAPORT_KONCOWY_PL.md`):

  | Rodzina | ΔIoU [95% CI] | Δscore |
  |---|---|---|
  | Funkcjonalna | −0,000015 [−0,00028; 0,00026] | −0,0152 |
  | Geometryczna | −0,00087 [−0,0033; 0,0019] | −0,0755 |

- Pojedyncza para funkcjonalna stanowi ok. 2% normy wyjścia MLP (mediana 1,899%; [REL] `WNIOSKI_PL.md`).

Te wyniki są częściowo antycypowane:
- Redundantne i usuwalne jednostki mnożą się szybciej niż szerokość sieci ([Casper et al., AAAI 2021](https://doi.org/10.1609/aaai.v35i8.16853)).
- Ablacje parami ujawniają efekty niewidoczne pojedynczo ([Meyes et al. 2019](https://arxiv.org/abs/1901.08644)).
- Mały efekt netto nie oznacza nieistotnego składnika ([Ahmad et al. 2026, preprint](https://arxiv.org/abs/2610.02173)).

Zastrzeżenie: reguła wyboru par funkcjonalnych częściowo gwarantuje przeciwne efekty na krawędziach.

### Wygaszanie wiązek to koncentracja, nie zerowanie

W seedzie 3047 między krokiem 0 i 55 000 (50 klatek) rośnie koncentracja wiązek. Przykład dla bloku 4:

| Wielkość (na token, blok 4) | Krok 0 | Krok 55 000 |
|---|---|---|
| Wiązki powyżej 5% maksimum na tokenie | 588,7 | 64,4 |
| N_eff (efektywna liczba wiązek) | 353,2 | 87,1 |
| Wiązki o sile powyżej 0,05 (próg absolutny) | 264,9 | 429,2 |
| Suma sił wiązek | 53,28 | 123,71 |

Na krok 55 000 dokładnych zer jest tylko 0,01–3,31 na token. Wiązki tracą więc względną ważność, a nie znikają ([ATLAS] `Wyniki\Raport\RAPORT.html` §1).

Rozkład kątów między trwałymi wiązkami rozszerza się w obu kierunkach. W bloku 4 średnia odległość od 90° rośnie z 3,31° do 8,79°, a udział par-tokenów ≥150° pozostaje znikomy: 3,57·10⁻⁶.

Literatura jest z tym zgodna. Z GELU dryf w stronę ujemnych preaktywacji daje małe ujemne wyjścia zamiast zer ([Shvetsov et al. 2026, preprint](https://arxiv.org/abs/2605.17659)). Rzadkość pojawia się wcześnie w treningu ([Li et al., ICLR 2023](https://arxiv.org/abs/2210.06313)).

Nie należy pisać, że „głębsze warstwy są rzadsze". U Li i wsp. najrzadsze są warstwy środkowe, a pierwsze i ostatnie są gęstsze. Atlasu nie liczono po epoce 50.

## Metodyka: rzut do 3D fabrykuje przeciwieństwo na poziomie przypadku

Kąty między wiązkami trzeba liczyć w pełnej 192-wymiarowej przestrzeni strumienia rezydualnego. Pokazuje to skrypt koordynatora ([SKRYPT]), który przy pisaniu raportu uruchomiono ponownie na checkpointach z epok 60 i 200 ([K200] `Wyniki\Dane\checkpoints\k065640.pt` i `k218750.pt`). Liczby poniżej pochodzą z tego uruchomienia.

**W 192-D niewiele par może dać kąt wiązek ≥150°.** Liczono pary kierunków W2, dla których max(α, 180° − α) ≥ 150°. Na blok jest ich **0–9**, a największy taki kąt wynosi 141,9–173,0°. Te liczby pokrywają się z liczbą par T150 w każdym bloku.

**Po rzucie do 3D takich par jest kilkadziesiąt tysięcy na blok.** Kąt zmienia się o ponad 30° dla 42,6–47,5% par.

| Rzut do 3D | Zachowana energia wektorów W2 | Pary „mogące być przeciwne" (z 294 528) |
|---|---|---|
| Najlepszy widok PCA | 8,4–22,1% | 39 721–45 755 (13,5–15,5%) |
| Losowe rzuty | — | 39 929–41 893 (13,6–14,2%) |

To praktycznie **poziom czysto geometrycznego przypadku**. Dla losowych kierunków w 3D P(α ≤ 30° lub α ≥ 150°) = 1 − cos 30° ≈ 13,4%. Wynik jest taki sam w epoce 60 i 200, więc liczba z rzutu 3D nie niesie żadnej informacji o treningu.

| Blok | Ep. 60: pary w 192-D (maks.) | Ep. 60: PCA 3D (energia) | Ep. 60: losowy 3D | Ep. 200: pary w 192-D (maks.) | Ep. 200: PCA 3D (energia) | Ep. 200: losowy 3D |
|---|---|---|---|---|---|---|
| 1 | 7 (157,8°) | 39 833 (10,2%) | 40 006 | 1 (150,1°) | 39 721 (8,4%) | 39 929 |
| 2 | 1 (152,9°) | 41 173 (14,5%) | 40 551 | 0 (141,9°) | 42 391 (12,5%) | 40 427 |
| 3 | 0 (144,5°) | 41 471 (14,8%) | 40 608 | 0 (148,3°) | 41 515 (12,5%) | 40 429 |
| 4 | 1 (156,9°) | 43 088 (17,3%) | 40 991 | 2 (159,4°) | 45 755 (14,7%) | 40 324 |
| 5 | 6 (164,2°) | 42 318 (20,8%) | 41 287 | 5 (158,4°) | 41 803 (18,0%) | 41 344 |
| 6 | 9 (154,4°) | 40 959 (22,1%) | 41 893 | 7 (173,0°) | 41 917 (19,8%) | 41 467 |

Rzut liniowy P zachowuje jednak dwie rzeczy:
- **sumę sił:** rzut wypadkowego pchnięcia równa się sumie rzutów pchnięć, P(Σu) = ΣPu;
- **dokładne znoszenie:** jeśli u_a = −u_b, to Pu_a = −Pu_b.

Widok 3D nadaje się więc do pokazania efektu netto w wybranej podprzestrzeni, np. w kierunkach czytanych przez głowicę. Nie nadaje się do mierzenia ani liczenia przeciwieństwa, bo implikacja odwrotna nie zachodzi: Pu_a ≈ −Pu_b nie oznacza, że u_a ≈ −u_b.

To wyjaśnia wcześniejszy pilot sferyczny. Wybrano w nim 19 575 par „przeciwnych w 3D" (seed 3047, epoka 60), czyli 6,65% z 294 528. To prawie dokładnie jednostronny poziom przypadku (1 − cos 30°)/2 ≈ 6,7% (wyliczenie własne). Żadna z tych par nie osiąga w 192-D kąta ze znakiem 150° ([SFERA] `RAPORT_PL.md`). Z 479 248 par wybranych w 3D w 24 modelach tylko 10 828 spełnia G120 w 192-D, a 78 327 ma nieujemny kosinus. **Wyniki sferyczne 3D należy wycofać jako miarę przeciwieństwa.**

Proponowane zdanie do sekcji Metody:

> **EN:** "All angles are computed in the full 192-dimensional residual space. Projecting the 768 output directions of a block to 3D—either onto the top three principal components, which retain only 8–22% of their energy, or onto random subspaces—marks 13.5–15.5% of all 294,528 pairs as potentially opposed. This is the chance level for random 3D directions (1 − cos 30° ≈ 13.4%), identical at epochs 60 and 200, whereas in 192-D at most nine pairs per block can be. Linear projection preserves sums and exact cancellation, so 3D views are used only to visualize net effects, never to count opposition."

Skrypt skopiowano do projektu: [P200] `Wyniki\Code_Interpretujacy_Wyniki\rzut_3d_vs_192d.py`.

**Zbiorowa równowaga wszystkich 768 sił, 192-D vs rzut P3 z aplikacji** (dopisane przez koordynatora; [P200] `Wyniki\Code_Interpretujacy_Wyniki\rownowaga_192d_vs_3d.py`, dane `Wyniki\Dane\rownowaga_192d_vs_3d\`; 4 klatki TEST z 4 filmów, 2400 tokenów). Miara to R = ‖Σu‖ / Σ‖u‖, czyli 1 przy siłach zgodnych i blisko 0 przy zbiorowej równowadze. P3 to rzut aplikacji (`app\lasery_dane.py`): 3 główne kierunki znormowanych kolumn fc2 z epoki 200.
- **W 192-D (blok 6, epoka 200)** mediana R = 0,174, a dla sił o losowych kierunkach 0,083. Wypadkowa jest krótka, ale siły są ok. 2× bardziej zgodne niż losowe. Nie ma więc zbiorowego mechanizmu równowagi; krótka wypadkowa to naturalny skutek wielu sił w wielu kierunkach.
- **W 3D** mediana R = 0,199. Na tokenach narzędzia R wynosi 0,29–0,42 wobec 0,16–0,25 w 192-D, więc 3D zawyża zgodność 1,5–1,9×. W epoce 60 błąd ma przeciwny znak (na 77% tokenów 3D pokazuje więcej równowagi), a w bloku 4 R wynosi 0,58 w 3D wobec 0,33 w 192-D. Zniekształcenie ma więc zmienny znak i nie da się go poprawić stałym współczynnikiem.
- **P3 dobrze pokazuje samą wypadkową.** Widać w nim 47% długości wypadkowej bloku 6 (mediana) i 68–83% na tokenach narzędzia, wobec ok. 12% dla losowego rzutu 3D. Pojedyncze siły widać w ok. 41% sumy ich długości.
- **Wniosek metodyczny:** wypadkowa w 3D jest wiarygodna, zbiorowa równowaga w 3D jest zniekształcona o 20–90% i ma zmienny znak, a równowaga par w 3D to poziom przypadku.

## Co można napisać w artykule: sześć twierdzeń o różnej pewności

Najmocniejsza wersja artykułu to praca o **geometrii wkładów neuronów w trakcie treningu**, z minimalnymi i negatywnymi twierdzeniami funkcjonalnymi.

Pewność oceniono w trzech stopniach:
- **wysoka:** wynik wynika z samych wag albo pochodzi z 24 seedów z przedziałami ufności;
- **średnia:** wyczerpujący pomiar w jednym seedzie albo jednomyślny pilot;
- **niska:** jedna para wybrana post hoc albo mniej niż 50 obserwacji.

### Twierdzenie 1: cykl życia przeciwnych wiązek

**Pewność: średnia (1 seed, 26 checkpointów, liczba wyczerpująca i niezależna od klatek).** Przeciwne wiązki są nieobecne przy inicjalizacji, pojawiają się w epoce 9, rosną do ok. 25 par w epoce 40, szczytują przy 30 parach w epokach 80–110 i spadają do 15. Ich skład się wymienia. W literaturze brak precedensu na poziomie neuronów. Najbliższe analogie to korekta między warstwami w LLM-ach (Patrawala 2025) i hamowanie push–pull w SNN (Zhu 2026). Bez replikacji na innych seedach to twierdzenie musi mieć dopisek „in one training run".

> **EN:** "Pairs of MLP neurons whose signed same-token contributions to the residual stream differ by at least 150° are absent at initialization (largest angle 110°), first appear at epoch 9, plateau at about 25 pairs from epoch 40, peak at 30 pairs (out of 1.77 million pairs across six blocks) at epochs 80–110 and decline to 15 by epoch 200, with substantial turnover of pair identities; no pair ever exceeds 173° (one training run)."

### Twierdzenie 2: przeciwieństwo pochodzi ze znaku aktywacji, nie z geometrii wag

**Pewność: wysoka w obrębie seeda 3047 (wynika z wag w każdym checkpoincie); średnia jako uogólnienie.** To centralny i najbardziej nowy wynik. Kontrastuje ze wszystkimi opublikowanymi mechanizmami (Gurnee 2024, Martinelli 2025, Mahan 2021), a jego składowe mają oparcie w literaturze (Gerstner 2025, Hendrycks & Gimpel 2016, Shvetsov 2026 jako preprint). W tym miejscu należy podać formalną definicję, zamiast mylącej definicji przez antyrównoległe wagi.

> **EN (definicja):** "We call two hidden neurons c and d of the same MLP an opposite-beam pair on token t if their signed contributions GELU(z_{t,c})·W2[:,c] and GELU(z_{t,d})·W2[:,d] to the residual stream form an angle of at least 150°."
>
> **EN (twierdzenie):** "None of these pairs has anti-parallel output weights—the largest angle between any two fc2 columns never exceeds 142°—so every pair separated by ≥150° combines nearly parallel output weights (≤30°) with opposite GELU signs on that token: in this ViT, extreme opposition is a property of activation sign, not of output-weight geometry."

### Twierdzenie 3: przeciwieństwo pojawia się po głównym zysku detekcji i nie śledzi IoU

**Pewność: wysoka dla definicji G120 (24 seedy, zaplanowane korelacje po detrendzie); średnia dla T150 (1 seed).** W tekście trzeba wyraźnie zaznaczyć, że dwie połowy zdania używają różnych definicji. Należy pisać o „plateau metryki przy stałym LR", nie o „zbieżności", i cytować Kunina, Zhanga i Kossona dla tezy, że plateau nie jest punktem stacjonarnym.

> **EN:** "Opposition emerges after most of the detection gain: 84% of the final validation IoU is reached by epoch 2.7, before any opposite-beam pair exists, and across 24 seeds, changes in output-weight opposition (cosine < −0.5) are uncorrelated with changes in IoU after controlling for seed and training interval (r = 0.07, 95% CI [−0.04, 0.16])."

### Twierdzenie 4: pary T150 to częściowe kopie; po szczycie przeważnie się różnicują, a scala się tylko najbardziej przeciwna

**Pewność: średnia dla „częściowe kopie” i „różnicowanie po szczycie” (wszystkie pary, z samych wag, 1 seed); niska–średnia dla scalania (jedna para, wybrana post hoc, ale z pełną historią wag w 26 checkpointach).** Historie wag pozostałych par T150 nie pokazały scalania. 20 z 30 par z epoki 110 rozeszło się (kąt W2 28° → 40°, cos W1 0,77 → 0,66), a w pełną kopię scaliła się tylko para 370–391. Scalanie pozycjonujemy jako studium przypadku zgodne ze stochastic collapse (Chen 2023) i z faworyzowaniem identycznych jednostek przez weight decay (Ziyin). Nie twierdzimy, że przyczyną jest weight decay.

> **EN (częściowe kopie i różnicowanie):** "Opposite-beam pairs are partial duplicates: their output weights are within 30° and their LayerNorm-folded input weights are similar (median cosine 0.75–0.81 at every epoch from 60 to 200), but their thresholds differ, so tokens falling between the two thresholds activate one neuron positively and the other negatively. After the peak, most of these pairs differentiate rather than merge: all 20 pairs lost between epochs 110 and 200 drifted apart (median output-weight angle 28° → 40°, input cosine 0.77 → 0.66), and only one pair merged into a near-exact copy (one training run)."

> **EN:** "The most opposed pair at epoch 200 is not an antagonist pair but a pair of near-copies formed late in training: between epochs 100 and 150 its input weights align (cosine 0.31 → 0.90) and its output weights converge (72° → 15°; 7° at epoch 200) while both units move into GELU's negative regime, so their contributions add on 94% of tokens and partially cancel on only 6%."

### Twierdzenie 5: wyłączenie pary nie zmienia detekcji, a jej efekty się sumują

**Pewność: średnia dla „brak wpływu na IoU" (1000 klatek, 300 par kontrolnych, bootstrap po filmach); niska dla interpretacji (tylko ablacja zerowa, tylko efekt całkowity).** Unikamy słowa „redundant". Jeśli przed wysłaniem nie powstanie pakiet ablacyjny (mean/mode, efekt bezpośredni, Δ_ij), należy dopisać zastrzeżenie o efekcie całkowitym z cytatem McGrath 2023.

> **EN:** "Zero-ablating this pair changes IoU by about 10⁻⁵ (95% CIs include zero), within the distribution of 300 matched control pairs, and its effects on the detection logit are additive, as expected for near-parallel copies rather than for cancelling units."

### Twierdzenie 6: przeciwstawność wag jest powszechna, ale nie jest kanałem ani mechanizmem funkcji

**Pewność: wysoka dla częstości G120 i rozdźwięku geometrii z funkcją (24 seedy); średnia dla odrzucenia kanałów (pilot na 3 seedach).** Sformułowanie musi być uczciwe wobec Martinellego: testujemy, czy kanały wyjaśniają przeciwieństwo w regularyzowanym transformerze, a nie twierdzimy, że ich teoria jest błędna.

> **EN:** "Output-weight opposition (cosine < −0.5) is common—about 55% of last-block neurons acquire such a partner by epoch 60 across 24 seeds—but in a three-seed pilot it does not reflect the flat-channel mechanism of Martinelli et al. (2025): opposed pairs have input weights no more similar than random pairs (median effective cosine 0.078 vs. 0.078 in seed 3047), none meets the channel criteria, and their contributions do not cancel."

### Twierdzenia drugorzędne

Do artykułu nadają się trzy twierdzenia drugorzędne:
- **Dryf pewności przy stałym IoU. Pewność: niska–średnia** (1 seed, 34 i 28 pustych klatek, score nieskalibrowany). Przydatny jako kontekst, ale nie łączyć go przyczynowo z przeciwnymi wiązkami. Proponowane zdanie: "Under a constant learning rate, validation IoU plateaued from about epoch 40, while winner-token confidence kept rising on frames with and without an instrument, consistent with continued logit-scale growth reported for classifiers (Guo et al. 2017; Wei et al. 2022) and with longer detector schedules (Oksuz et al. 2023)."
- **Wygaszanie jako koncentracja. Pewność: średnia** (1 seed, epoki 0–50).
- **Artefakt rzutu 3D. Pewność: wysoka** (czysta geometria, sprawdzone na dwóch checkpointach).

Deklaracja nowości powinna być wąska i zawierać „to our knowledge":

> **EN:** "To our knowledge, no prior work tracks, across the full training of a transformer, the per-token directional opposition between the residual-stream contributions of individual MLP neurons, nor decomposes it into output-weight geometry versus activation sign."

### Czego nie pisać

- Nie przedstawiać przeciwnych wiązek jako obwodu znoszenia typu erasure lub suppression.
- Nie twierdzić, że ujemny ogon GELU jest funkcjonalnie bez znaczenia (Kong 2026).
- Nie definiować przeciwnych wiązek przez antyrównoległe wagi wyjściowe.
- Nie pisać „redundant" o efektach addytywnych.
- Nie pisać, że weight decay „woli dwie połówki".
- Nie pisać, że głębsze warstwy są rzadsze.
- Nie mówić o „zbieżności" przy plateau metryki.
- Nie łączyć skali czasowej Kossona z oknem epok 80–150.
- Nie używać liczb ze sfery 3D.
- Nie pisać „first ever".

## Ryzyka recenzenckie

| # | Zarzut recenzenta | Dlaczego jest groźny | Odpowiedź lub kontrola |
|---|---|---|---|
| 1 | Jeden seed i 50 klatek dla wszystkich wyników T150 | Według Chen 2023 kopie zależą od LR i szumu, więc tożsamości par mogą się różnić między przebiegami | Policzyć T150 na istniejących checkpointach 24 seedów (epoki 0–60); dodać 2–3 przebiegi lub kontynuacje z perturbacją do epoki 200; raportować statystyki, nie tożsamości |
| 2 | Kąt 173° to tautologia | Przy równoległych W2 możliwe są tylko kąty α i 180° − α | Przedstawić wynik jako dekompozycję (cel Twierdzenia 2) i dodać model zerowy |
| 3 | Brak modelu zerowego | 15–30 par na 1,77 mln może być wynikiem oczekiwanym przy danych rozkładach preaktywacji | Tasowanie partnerów; pary o dopasowanym α i dopasowanych rozkładach preaktywacji; losowe rotacje W2 z zachowanym spektrum |
| 4 | Mieszanie definicji przeciwieństwa | G120, T150, F i sfera 3D dają liczby od 0–30 do 479 248 | Jedna formalna definicja w Metodach; G120 jako osobny wynik z osobną nazwą; usunąć wyniki 3D |
| 5 | Ablacja zerowa niczego nie dowodzi | Efekt całkowity może być maskowany kompensacją (McGrath, Rushing & Nanda); wartość zastępcza może „oszukiwać" sieć (Li & Janson); współczynnik interakcji jest niestandardowy | Ablacje zero, mean i mode; efekt bezpośredni; zamrożony LayerNorm; Δ_ij = δ_pair − δ_i − δ_j z CI na tle 300 kontroli; metryki ciągłe (logit, krawędzie ramki) |
| 6 | Para wybrana post hoc | Najbardziej skrajny przypadek może być nietypowy | Regułę wyboru ustalić z góry; pokazać rozkład dla wszystkich par T150 i wszystkich par podobnych do kopii |
| 7 | Weight decay jako przyczyna kopii | AdamW to nie L2 (Xie & Li 2024); teza o „dwóch połówkach" jest niepoparta (Savarese 2019); szum to alternatywa | Przebieg z wd = 0 i przebieg z innym LR od epoki 60; ostrożne sformułowanie |
| 8 | Plateau to artefakt stałego LR | Andriushchenko 2023 i Chen 2023 pokazują ukryty postęp przy dużym kroku | Sonda spadku LR z epok 40, 100 i 200 |
| 9 | Pewność na pustych klatkach | n = 34 i 28; score nieskalibrowany; maksimum logitów to statystyka ekstremalna | AUROC obecności narzędzia; skalowanie temperaturą; LaACE₀ i D-ECE; sparowany bootstrap; test trendu |
| 10 | Prace mogące wyprzedzać nasz wynik | Kwon & Kim 2005 „anti-pairing", Guo et al. 2017 „opposite singularity" i Fukumizu & Amari 2000 nie zostały przeczytane | Przeczytać przed wysłaniem ([Kwon & Kim](https://doi.org/10.1088/0305-4470/38/25/003); [Guo et al.](https://doi.org/10.1016/j.neucom.2017.12.018); [Fukumizu & Amari](https://doi.org/10.1016/S0893-6080(00)00009-5)) |
| 11 | Niesprawiedliwy test kanałów Martinellego | Ich kanały zdefiniowano dla skalarnego wyjścia, bez weight decay i dla gradient flow | Sformułować jako test w regularyzowanym transformerze; zaznaczyć, że uogólnienie warunku na W2 jest nasze |
| 12 | Ujemny ogon GELU jest funkcjonalny | Kong et al. (ICLR 2026) pokazują to w LLM-ach | Nie twierdzić inaczej; zrobić test zerowania z < 0 i odwrócenia znaku |
| 13 | Zbiór TEST był już oglądany | VAL służył do wyboru checkpointu, a TEST pojawiał się w poprzednich kampaniach | Świeży zbiór testowy albo jawne zastrzeżenie |
| 14 | Niejasna definicja „efektywnego biasu" | Tuomi 2026 (preprint) pokazuje, że punkt spoczynku niesie głównie w·E[LN(x)], 48–56× więcej niż b | Raportować b, b̂ i średnią preaktywację; podawać próg w jednostkach rzutu |

## Brakujące eksperymenty (priorytet)

Kontynuacja z epoki 60 do 200 (153 110 kroków) trwała na Edenie 4 718 s ([K200]). Dodatkowe przebiegi są więc tanie.

| Priorytet | Eksperyment | Co rozstrzyga | Dane i koszt |
|---|---|---|---|
| **P1** | Liczenie T150 (`count_opposite_pairs.py`) na checkpointach 24 seedów z epok 0–60 | Czy pojawienie się pierwszych par po epoce 9 i wzrost do ok. 25 par się replikują | Checkpointy i kod już istnieją; tani |
| **P1** | 2–3 dodatkowe przebiegi do epoki 200: nowe seedy albo kontynuacje z perturbacją z epoki 60 | Czy szczyt w epokach 80–110, spadek i kopie są deterministyczne, czy szumowe | Ok. 1,3 h na Edenie na kontynuację |
| **P1** | Model zerowy dla liczby par T150 | Czy 15–30 par to więcej, niż wynika z przypadku | Z istniejących wag i aktywacji |
| ~~P1~~ **zrobione częściowo** | Historie W1, kąta W2 i b̂ dla par T150 z epoki 110 (epoki 110 vs 200) | **Rozstrzygnięte przeciw hipotezie:** spadek po epoce 110 to rozchodzenie się par (20/20 zniknęłych ma kąt W2 > 30°), nie scalanie. Zostaje: pełne trajektorie wszystkich 73 tożsamości i to samo na innych seedach | [P200] `los_par_t150.py`, `wejscia_par_t150.py` |
| **P1** | Pakiet ablacyjny dla pary 370–391 i wszystkich par T150: zero, mean i mode; efekt bezpośredni; zamrożony LN; Δ_ij z CI; metryki ciągłe; granica równoważności (TOST) | Czy wynik zerowy wytrzymuje standardy z literatury | 500 VAL + 500 TEST |
| **P1** | Przeczytać Kwon & Kim 2005, Guo et al. 2017 i Fukumizu & Amari 2000 | Ryzyko, że ktoś już opisał ten mechanizm | Lektura |
| **P2** | wd = 0 oraz inny LR (kontynuacje z epoki 60) | Czy kopie i spadek napędza weight decay, czy szum | 2 przebiegi |
| **P2** | Sonda spadku LR z epok 40, 100 i 200 (5–10 epok, kosinus do zera) | Czy plateau ukrywa postęp i czy późna reorganizacja ma znaczenie funkcjonalne | 3 krótkie przebiegi |
| **P2** | Pakiet kalibracji: LaACE₀, D-ECE, AUROC obecności, temperatura, sparowany bootstrap, test trendu | Czy dryf pewności to dryf skali, czy utrata rozróżniania | Istniejące predykcje |
| **P2** | Analiza epok 200–400 ([K400]) po zakończeniu treningu | Czy kąt pary 370–391 dąży do 180° i czy liczba par spada dalej | Trening trwa |
| **P2** | Logowanie norm i aktualizacji kątowych | Weryfikacja skali ok. 37 epok i ok. 1,6·10⁻³ rad na krok (Kosson) | Z checkpointów lub nowy log |
| **P3** | Zerowanie ujemnego ogona GELU i odwrócenie znaku (para, blok 6, losowe neurony) | Odpowiedź na Kong 2026 | Tani |
| **P3** | Test scalania: zastąpić parę jednym neuronem (zsumowane W2, uśrednione W1, jeden z biasów) | Czy przesunięcie progów ma znaczenie | Tani |
| **P3** | Pełny test kanałów na 24 seedach, także w oknie epok 100–200 | Domknięcie odrzucenia kanałów | Kampania przygotowana, niezlecona |
| **P3** | Statystyka trajektorii: bootstrap wystąpień po filmach, test trendu, korekta na wielokrotność | Przedziały ufności dla Twierdzenia 1 | Tani |
| **P3** | Kontynuacja w FP32 z epoki 100 | Czy bf16 wpływa na późny wzrost logitów | 1 przebieg |
| **P3** | Atlas siły wiązek, N_eff i rozkładu kątów po epoce 50 | Czy koncentracja trwa | Istniejący kod |

## Rysunki: co już istnieje i czego brakuje

Wszystkie rysunki [ATLAS] i [P200] istnieją także jako SVG i PDF. Pełny atlas to [ATLAS] `Wyniki\Wykresy\ATLAS_WYKRESOW.pdf`.

| Proponowany rysunek | Plik(i) | Co pokazuje | Miejsce w artykule |
|---|---|---|---|
| Rys. 1. Cykl życia przeciwnych wiązek na tle IoU i pewności | [P200] `Wyniki\Wykresy\01_pary_przeciwne_0_200_progi.png`, `02_pary_przeciwne_bloki.png`, `08_iou_i_pewnosc_0_200.png` | Liczba par T150 według progów i bloków oraz IoU i pewność na wspólnej osi epok | Główny; panele zestawić we wspólnej osi |
| Rys. 2. Mechanizm: W2 i znaki | [ATLAS] `Wyniki\Wykresy\16_wagi_i_znaki.png`, `17_rzadkie_katy.png`; [P200] `Wyniki\Wykresy\03_maksymalny_kat_bloki.png` | Kąty W2 nigdy >142°; przeciwieństwo ze znaku GELU; maksymalny kąt na blok | Główny; brakuje wykresu α względem kąta wiązki dla 339 kandydatów (do zrobienia z `CANDIDATE_PAIRS_60_200.csv`) |
| Rys. 3. Para 370–391 staje się kopiami | [P200] `Wyniki\Wykresy\05_para_370_391_historia.png`, `06_para_370_391_tokeny_epoka200.png` | Historia wag 0–200; podział tokenów (94% dodawania, 6% znoszenia) | Główny |
| Rys. 4. Ablacja na tle 300 kontroli | [P200] `Wyniki\Wykresy\07_wylaczenie_pary_kontrola.png` | ΔIoU i współczynnik interakcji na tle kontroli | Główny lub suplement |
| Rys. 5. Moment pojawienia się względem IoU (24 seedy) | [PRUNE] `figures_20261019\trajectories_gt_anchor.png`, `correlations_gt_anchor_difference.png` | Najpierw zysk IoU, potem wzrost G120; korelacje po detrendzie | Główny; w podpisie zaznaczyć definicję G120 |
| Wystąpienia na tokenach | [P200] `Wyniki\Wykresy\04_wystapienia_par_tokeny.png` | Spadek wystąpień przy obu wiązkach aktywnych po epoce 110 | Suplement |
| IoU i pewność od epoki 40 | [P200] `Wyniki\Wykresy\09_iou_i_pewnosc_od_40_epoki.png` | Plateau IoU i rosnąca pewność | Suplement |
| Epoki 0–60 w szczegółach | [ATLAS] `Wyniki\Wykresy\PARY_PRZECIWNE_PRZEBIEG_EPOKI_BLOKI.png`, `PARY_PRZECIWNE_0_55000_65640.png` | Pojawienie się par T150 | Suplement |
| Wzrost G120 | [S3] `geometria_ostatni_mlp.png`, `geometria_szesc_blokow.png`; [PRUNE] `figures_20261019\trajectories_clean_winner.png` | 0 → ok. 55% neuronów z partnerem | Suplement |
| Geometria a funkcja | [PRUNE] `figures_20261019\group_retrospective_fast16_clean_winner.png`; [WSP] `przeciwne_aktywacje\wspolaktywacja_w_czasie.png`; [FUN24] `figures\directions_confidence_24seeds.png` | Spadek współaktywacji z 86% do 18% przy rosnącej przeciwstawności | Suplement |
| Kompensacja i pruning | [FUN24] `figures\functional_compensation_24seeds.png`; [PRUNE] `figures_20261019\pruning_functional_clean_winner.png`, `contrasts_clean_winner.png`; [RESTORE] `figures\poziomy_clean_winner.png` | Kompensacja istnieje, ale wyłączanie par nie poprawia IoU | Suplement |
| Wygaszanie jako koncentracja | [ATLAS] `Wyniki\Wykresy\01_podsumowanie.png`, `02_aktywnosc_wzgledna.png`, `03_aktywnosc_absolutna.png`, `04_efektywna_liczba.png`, `07_zera.png` | Mniej wiązek ważnych względnie, więcej absolutnie; brak zer | Suplement |
| Rozkłady kątów | [ATLAS] `Wyniki\Wykresy\11_rozklad_katow.png`, `13_rozklad_zmian_katow.png` | Rozszerzanie się rozkładu w obu kierunkach | Suplement |
| Przebieg gęsty | [WSP] `przeciwne_reflektory\przebieg_w_czasie.png`, `korelacje_po_trendzie.png` | 806 checkpointów, G120 | Suplement; seed niepodany, prawdopodobnie 47 |
| **Do zrobienia:** rzut 3D a 192-D | [SKRYPT] | Histogram liczby par z rzutu 3D na tle poziomu przypadku 13,4% wobec 0–9 par w 192-D | Suplement (Metody) |
| **Do zrobienia:** pilot kanałów | brak rysunku; tabele w [MAR] `Wyniki\Dane\eden\redukcja_pilot\` | Rozkład c_in_eff dla par przeciwnych i losowych | Suplement |
| **Do zrobienia:** model zerowy | — | Obserwowana liczba par T150 na tle rozkładu zerowego | Główny, jeśli powstanie |
| **Nie używać** | [RESTORE] `validation_plot\synthetic_figures\*`; rysunki sfery 3D ([SFERA]) jako miara przeciwieństwa | Dane syntetyczne; artefakt rzutu | — |

## Wnioski końcowe

Najważniejsza zmiana w rozumieniu: w transformerze z GELU przeciwieństwo wkładów neuronów ma dwa niezależne źródła, geometrię wag i znak aktywacji, i mierzą one różne zjawiska. Przeciwstawność wag (G120):
- jest powszechna;
- rośnie wcześnie;
- jest oderwana od funkcji.

Skrajne przeciwieństwo na tokenie (T150):
- jest rzadkie;
- pochodzi wyłącznie ze znaku aktywacji nad prawie równoległymi wagami;
- szczytuje późno.

Pytania „czy neurony się znoszą" nie da się więc rozstrzygnąć ani z samych wag, ani z widoku 3D, który daje poziom przypadku. Potrzebne są wkłady ze znakiem per token w pełnym wymiarze. Co ważniejsze, pary T150 nie są antagonistami, tylko częściowymi kopiami: podobne wejścia, równoległe wyjścia, przesunięte progi. Okno przeciwnych znaków to szczelina między dwoma progami na prawie tym samym kierunku wejścia. Przeciwne wiązki przestają być motywem funkcjonalnym i stają się znacznikiem późnej reorganizacji duplikatów. Ta reorganizacja trwa długo po zatrzymaniu się metryki zadania i przeważnie polega na **różnicowaniu** częściowych kopii (20 z 30 par po szczycie rozeszło się), a tylko wyjątkowo na scaleniu w pełną kopię (para 370–391).

Hipoteza z notatek, że spadek po epoce 110 to pary stające się kopiami, została sprawdzona na wagach i **odrzucona**: pary przed zniknięciem mają malejący, a nie rosnący kosinus W1 (0,77 → 0,66). Pozostają falsyfikowalne przewidywania na inne przebiegi:
- na innych seedach powinny powtarzać się liczby par i proporcja „różnicowanie vs scalanie”, a nie tożsamości;
- przebieg z wd = 0 albo z niższym LR powinien pokazać, czy różnicowanie i rzadkie scalanie napędza kara, czy szum.

Najtańszy rozstrzygający krok to dwa ruchy z danymi, które już istnieją albo kosztują godzinę obliczeń: istniejący kod liczenia par na checkpointach 24 seedów i dwie krótkie kontynuacje z epoki 60. Twierdzenia funkcjonalne powinny pozostać skromne i negatywne. Wartość pracy leży w nowej miarze i w dekompozycji przeciwieństwa, nie w tezie, że przeciwne wiązki czemuś służą.

## Skróty ścieżek wewnętrznych

| Skrót | Ścieżka bezwzględna |
|---|---|
| [ATLAS] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_wiazkiprzeciwstawne_20261020__18-10-00` |
| [P200] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\pary_przeciwne_do_200epok_20261020__23-21-30` |
| [MAR] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\kanal_mar25l_checkpointy_20261020__18-38-50` |
| [K200] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_200epok_20261020__19-06-42` |
| [K400] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_400epok_20261020__23-50-57` |
| [FUN24] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\kompensacja_funkcjonalna_dgx2_20261017` |
| [PRUNE] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\kompensacja_pruning_dynamics_20261018` |
| [RESTORE] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\kompensacja_restore_all_20261018` |
| [REL] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\kompensacja_wklad_wzgledny_20261019` |
| [S3] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\przeciwstawne_eden_20261017_3seeds` |
| [SFERA] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\wiazki_sfera_abcd_20261019` |
| [AUDYT] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\opposition_sphere_audit_20261019` |
| [WSP] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\wspoldzialanie_neuronow_20261015` |
| [LIT] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\plan_na_przyszle_eksperymenty\2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md` |
| [SKRYPT] | `C:\Users\bartl\AppData\Local\Temp\claude\W--Studia-PhD-projekt-VIT-ViTParticleFilterTracker\adc24cc7-c1cf-463b-b253-5ed9d9c2dbaa\scratchpad\rzut_3d_vs_192d.py` |
