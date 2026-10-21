# Mikroskop latentny widzi przepływ, nie pułapki

## Najważniejsze wnioski

Chmura 600 tokenów-koloidów bloku 6 przypomina nie szkło koloidalne, lecz zbiór rozróżnialnych cząstek-znaczników przypiętych do własnych wejść, które po relaksacji trwającej ok. 30 epok (prędkość spada ok. 100×) fluktuują w nierównowagowym stanie stacjonarnym AdamW przy stałym LR, są rozdzielane na narzędzie i tło przez pole zależne od etykiety (jak w neural collapse) i dalej się upraszczają (PR 10,6 → 3,2) [1 seed]. Szkła nie potwierdza ani nie wyklucza żaden dotychczasowy pomiar — rozstrzygną to krzywa antykorelacji klatki c(Δt), α₂ z funkcją van Hove'a, dwuczasowy test starzenia, χ₄ ze zrywaniem wiązań i test „temperatury" na gałęziach lr/batch — ale prostemu starzeniu już przeczą stała ruchliwość od epoki 30 i spadek trwałości sąsiedztwa przy stałym odstępie 10 epok z 0,775 do 0,712, czyli lekko przyspieszająca przebudowa. W polu z aplikacji nie ma prawdziwych pułapek: przy koloidzie |f| jest 12–17× większe od połowy komórki, Newton bez ograniczeń zbiega tylko dla 15–16% tokenów, a znalezione zero leży ok. 37 komórek (2–3,3 promienia chmury) dalej, więc „pułapka" to lokalna topologia silnego przepływu w układzie poruszającym się razem z koloidem. Ta topologia ma wyraźną historię (siodła przy inicjalizacji, odpychanie w kroku 300, przyciąganie w epokach 3–70), a od epoki 80 rozwarstwia gatunki: wokół narzędzia najczęstszą klasą pozostaje zbieżność (43–72%), wokół tła dominują siodła (78–95% od epoki 120), co potwierdza niezależna od przekroju dywergencja pełnego jakobianu (ok. −0,024 wobec −0,013 do −0,018 na wymiar), choć po korekcie na przypadek klasa pojedynczej cząstki jest równie słabo trwała u obu gatunków (κ ≈ 0,15–0,33). Pełny jakobian 192-D ma 87–93% zespolonych wartości własnych i ok. 45% z Re > 0, czyli poziom losowy (inicjalizacja 93% i 50%, macierz Ginibre'a 94%), a znaleziskiem jest tylko przesunięcie: trening czyni pole nieco bardziej symetrycznym (stosunek rotacji do odkształcenia 1,00 → 0,85–0,90) i ściskającym. Publikowalne są niezmienniki niezależne od przekroju z modelami zerowymi, mechanizm na poziomie lasera (część obrotowa pola to rozbieżność kierunku, z którego neuron „czyta", i kierunku, w który „pcha", formalnie jak ciśnienie promieniowania w pęsecie optycznej) oraz widok 3-D jako wizualizacja opisowa z kontrolami — nie „wiry wokół koloidów". Lasery kolimują się ze 154 do 31 efektywnych kierunków do epoki 30 i powoli rozchodzą do 37, ich norma rośnie z 0,29 do 0,92 (blisko skali równowagi AdamW, ok. 0,98), Gini mocy stoi na 0,17, ok. 30% specjalizuje się w tle, a połowę hamulca rozdziału daje 10–19 laserów; „stały obrót 2,2°/epokę" to w rzeczywistości stacjonarna dyfuzja rotacyjna z czasem dekorelacji ok. 150 epok, a kąt po 10 epokach (22°) jest 2,3× większy niż dla nieskorelowanego spaceru o kroku Kossona (9,7°). Zmiany liczby ciemnych laserów na plateau korelują ze zmianami IoU (r = −0,69 VAL, −0,79 TEST; 15 zmian, 72 testy), ale wynik TEST leży dokładnie na suficie szumu √0,62 ≈ 0,79, więc to hipoteza, nie wynik. Mikroskop może pomóc przy okluzji jako źródło informacji o wiarygodności pomiaru dla filtra cząsteczkowego, jeśli potwierdzi się hipoteza, że zasłonięte tokeny narzędzia tracą cechy narzędzia (położenie wobec chmury tła, ściskającą dywergencję, aktywność laserów narzędzia) wcześniej, niż spada pewność głowicy; trzeba to sprawdzić na sparowanych bliźniakach okluzyjnych i ok. 200 ręcznie oznaczonych klatkach, a wynik wprowadzić do filtra jako szum pomiaru, bramkę i stan widoczności. Nie znaleziono żadnego zbioru katarakty z potwierdzonymi etykietami okluzji, więc etykiety trzeba przygotować samodzielnie.

**Konwencje.**
- **Zakres danych.** Pomiary własne pochodzą z jednego przebiegu (seed 3047) i dotyczą wyłącznie bloku 6. Użyto 12 klatek TEST z 12 różnych filmów oraz 26 checkpointów od kroku 0 do epoki 200. Siatka jest nieregularna do epoki 60, potem checkpointy są co 10 epok.
- **Oznaczenie [1 seed].** Wszystkie takie wyniki trzeba przed publikacją powtórzyć.
- **Ścieżki.** Skróty w nawiasach kwadratowych rozwija tabela na końcu.
- **„Wyliczenie własne".** To liczba policzona przy pisaniu raportu z plików danych eksperymentu, której nie ma w plikach wynikowych.
- **Preprinty.** Prace z lat 2025–2026 bez recenzji są oznaczone jako preprinty.
- **Przeciwne wiązki.** Tło dotyczące przeciwnych wiązek jest w [PW] i nie jest tu powtarzane.

## Chmura to znaczniki w stanie stacjonarnym, a szkło pozostaje niesprawdzoną hipotezą

### Co mierzymy

Chmura to 600 stanów tokenów jednej klatki tuż przed MLP bloku 6, z odjętą średnią klatki, w pełnych 192 wymiarach. Ruch to zmiana położenia tej samej klatki między checkpointami, czyli ruch w czasie treningu, a nie w filmie ([KOL] `Opis\OPIS_EKSPERYMENTU.txt`).

Pierwsze 30 epok to relaksacja po „wstrząsie" inicjalizacji. Prędkość tła po usunięciu sztywnego obrotu (Procrustes) spada z 2,5 R/epokę w pierwszych 300 krokach do 0,035 R/epokę między epokami 22 i 30. Potem stoi w paśmie 0,028–0,035 R/epokę aż do epoki 200 ([KOL] `Wyniki\Dane\RUCH_MIEDZY_CHECKPOINTAMI.csv`). W tym samym czasie:
- tokeny narzędzia opuszczają chmurę tła: poza jej promieniem 95% jest ich 92% w epoce 22 i 95–99% od epoki 40;
- odległość środków obu gatunków rośnie z 0,74 R do 1,8–2,0 R;
- współczynnik uczestnictwa (PR) spada z 10,6 do 5,0 w epoce 40 i dalej do 3,15 w epoce 200, długo po plateau IoU (VAL 0,748–0,775 od epoki 40, [PW]) ([KOL] `METRYKI_CHECKPOINTY.csv`).

Przemieszczenia na plateau są silnie subdyfuzyjne. MSD tła rośnie tylko z 0,114 do 0,195 R² przy odstępach 10–140 epok (wykładnik ≈ 0,20). MSD narzędzia rośnie z 0,117 do 0,267 R² (wykładnik ≈ 0,31–0,34) ([KOL] `MSD_60_200.csv`). MSD przy odstępie 10 epok to już 44–58% wartości przy 140 epokach. **Większość ruchu dzieje się więc w czasie krótszym niż 10 epok**, a siatka checkpointów nie rozdziela krótkiego reżimu. Właśnie tam leżałaby ewentualna „klatka" o promieniu ok. 0,34 R (wyliczenie własne: √0,114).

| Epoka | PR chmury | Rozdział środków / R | Narzędzie poza 95% tła | Prędkość tła (R/epokę, odcinek kończący się w tej epoce) | Trwałość 10 NN (ten sam odcinek) |
|---|---|---|---|---|---|
| 0 | 10,6 | 0,74 | 0% | — | — |
| 0,27 (krok 300) | 6,3 | 0,91 | 33% | 2,50 | 0,58 |
| 2,7 | 5,8 | 1,08 | 62% | 0,21 | 0,76 |
| 9,1 | 6,1 | 1,31 | 82% | 0,080 | 0,81 |
| 30 | 5,4 | 1,50 | 92% | 0,035 | 0,81 |
| 40 | 5,0 | 1,58 | 95% | 0,029 | 0,80 |
| 60 | 4,6 | 1,73 | 96% | 0,030 | 0,79 |
| 100 | 3,7 | 1,82 | 96% | 0,029 | 0,75 |
| 150 | 3,4 | 1,88 | 96% | 0,032 | 0,73 |
| 200 | 3,15 | 1,82 | 96% | 0,035 | 0,71 |

Źródło: [KOL] `METRYKI_CHECKPOINTY.csv` i `RUCH_MIEDZY_CHECKPOINTAMI.csv` [1 seed, 12 klatek]. Odcinki do epoki 30 mają różne długości (0,27–8,2 epoki), od epoki 30 trwają ok. 10 epok.

### Najbliższy opis fizyczny: znaczniki w NESS z polem zależnym od etykiety

Dwie cechy odróżniają tę chmurę od zawiesiny koloidów.

Po pierwsze, **cząstki są rozróżnialne i przypięte do wejścia**. Położenie tokenu to deterministyczna funkcja jego stałej treści obrazu i wag. „Ruch zbiorowy" pochodzi więc ze wspólnej aktualizacji wag, a nie z sił między cząstkami.

Po drugie, **mierzone pole MLP działa na każdy token osobno**: to siła jednociałowa. Klatki w sensie sąsiadów blokujących ruch nie mogą więc powstać w tym polu. Oddziaływania między tokenami wnoszą tylko atencja i wspólne wagi.

Najbliższy ilościowy opis to znaczniki w nierównowagowym stanie stacjonarnym (NESS) SGD/AdamW przy stałym LR:
- stacjonarna kowariancja wag skaluje się jak η/S, a ruch po zbieżności jest anomalną dyfuzją z niezerowymi prądami fazowymi ([Kunin i in. 2021](https://doi.org/10.1162/neco_a_01626));
- detailed balance jest złamany, a stan stacjonarny nie jest rozkładem Boltzmanna funkcji straty ([Adhikari i in. 2023](https://doi.org/10.48550/arXiv.2306.03521));
- na to nakłada się ograniczone, wracające do średniej drganie w poprzek rozmaitości rozwiązań i dyfuzja obrotowa wzdłuż niej ([Pashakhanloo i Koulakov 2023](https://doi.org/10.48550/arXiv.2302.02563)).

Ostatnia z tych prac przewiduje też, że rzadszy bodziec dryfuje szybciej. To zgadza się z większą ruchliwością rzadszych tokenów narzędzia (wykładnik 0,31–0,34 wobec 0,20).

Rozdział narzędzia od tła wygląda jak neural collapse przeniesiony na tokeny wewnątrz obrazu. NC1 maleje o rzędy wielkości i spada dalej po osiągnięciu „terminal phase" ([Papyan i in. 2020](https://doi.org/10.1073/pnas.2015509117)). Ta definicja dotyczy jednak ostatniej warstwy i klas obrazów. Nawet praca o NC w ViT nie analizuje tokenów obiektu wobec tła ([IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3602733)), co jest luką do zagospodarowania.

Dalsze upraszczanie chmury przy stałym LR ma dwa udokumentowane mechanizmy:
- **stochastic collapse**: szum SGD przyciąga do zbiorów niezmienniczych (neurony znikające lub identyczne), także po plateau straty i dokładności ([Chen i in. 2023](https://doi.org/10.1088/1742-5468/ad6429));
- **niski rząd z weight decay**, przy czym twierdzenie ogranicza rząd, ale nie przewiduje jego spadku bez końca ([Galanti i in. 2022](https://arxiv.org/abs/2206.05794)).

Teoria tokenów jako oddziałujących cząstek na sferze ([Geshkovski i in. 2023](https://doi.org/10.48550/arXiv.2312.10794)) opisuje ewolucję **w głębokości** w modelach bez MLP. Nie wolno jej cytować jako przewidującej grupowanie w czasie treningu.

### Szkło to hipoteza, a literatura DNN raczej jej nie wspiera

Szkło koloidalne wymagałoby czterech sygnatur, których jeszcze nie policzono:
- piku α₂ w czasie ucieczki z klatki;
- grubych, wykładniczych ogonów funkcji van Hove'a;
- antykorelacji przemieszczeń, która najpierw dochodzi do ok. 0,5, a potem maleje ([Weeks i Weitz 2002](https://doi.org/10.1016/s0301-0104(02)00667-5); [Kob i Andersen 1995](https://doi.org/10.1103/PhysRevE.51.4626));
- piku χ₄ i zależności od wieku ([Berthier i Biroli 2011](https://doi.org/10.1103/RevModPhys.83.587)).

Literatura o sieciach w przestrzeni wag raczej temu przeczy:
- sieci trenowane dynamiką Langevina nie wykazują klatek (abstrakt) ([Kerr Winter i Janssen 2025](https://doi.org/10.1103/physrevresearch.7.023010));
- przeparametryzowane SGD starzeje się tylko, dopóki strata maleje, a potem jest stacjonarne i dyfuzyjne ([Baity-Jesi i in. 2018](https://doi.org/10.1088/1742-5468/ab3281)).

Wykładniki MSD wag w literaturze:

| Zjawisko | Wartość | Źródło |
|---|---|---|
| Subdyfuzja późnego treningu, ResNet-14 | α ≈ 0,78 | [Chen i in. 2022](https://doi.org/10.1016/j.neunet.2022.01.019) |
| Subdyfuzja, RNN | α ≈ 0,7 | [Hertz i Tyrcha 2025](https://doi.org/10.1103/physreve.111.055307) |
| Superdyfuzja po zbieżności | c ≈ 1,07–1,6 | [Kunin i in. 2021](https://doi.org/10.1162/neco_a_01626) |
| Subaging, MNIST | μ = 0,709 ± 0,017 | [Herrera Segura i in. 2022](https://doi.org/10.1088/2632-2153/ac8f1b) |

Nasze 0,20–0,34 jest niższe niż każda z tych wartości. Możliwe są trzy niewykluczające się wyjaśnienia:
1. Procrustes i normowanie przez R usuwają dyfuzyjne mody obrotu i skali.
2. Większość dyfuzji wag biegnie płaskimi kierunkami, niewidocznymi w funkcji.
3. Prawdziwe uwięzienie.

Prac o wagach nie wolno więc cytować jako *poparcia* dla klatek tokenów, tylko jako kontekst.

**Dwa istniejące wyniki już przeczą prostemu starzeniu.**

Pierwszy to stała ruchliwość. W starzejącym się szkle koloidalnym ruchliwość przy stałym odstępie maleje jak t_w^(−a), z a = 0,05–0,5 ([Courtland i Weeks 2003](https://doi.org/10.1088/0953-8984/15/1/349)). U nas prędkość przy odstępie 10 epok jest stała od epoki 30 do 200, czyli a ≈ 0.

Drugi to trwałość sąsiedztwa. Trwałość 10 najbliższych sąsiadów, liczona przy stałym odstępie ok. 10 epok, spada:
- z 0,80 (odcinki 30–40 i 40–50)
- przez 0,775 (60–70)
- do 0,712 (190–200).

Przy wcześniejszych, krótszych i nieregularnych odstępach wynosiła 0,80–0,82. **Przebudowa sąsiedztw lekko przyspiesza z wiekiem, czyli odwrotnie niż przy starzeniu.**

Prawdopodobne wyjaśnienie to zagęszczenie chmury przez spadek wymiaru (oszacowanie własne, zgrubne). Dla 600 punktów w d_eff wymiarach typowa odległość do najbliższego sąsiada skaluje się jak 600^(−1/d_eff). Spadek PR z 4,6 do 3,2 skraca ją ok. 1,8× przy tym samym przemieszczeniu, więc sąsiedzi zamieniają się częściej. Hipotezę można sprawdzić od ręki: liczyć trwałość w jednostkach lokalnej odległości NN albo w stałej liczbie składowych głównych.

### Analogie testowalne i metafory

| Analogia | Status | Co pasuje | Co nie pasuje | Test na naszych danych |
|---|---|---|---|---|
| NESS stałego LR, „temperatura" ∝ η/B ([Kunin 2021](https://doi.org/10.1162/neco_a_01626); [Adhikari 2023](https://doi.org/10.48550/arXiv.2306.03521)) | **testowalna** | Relaksacja ok. 30 epok, potem stała ruchliwość; skala Kossona ok. 37 epok | Pod Adamem temperatura nie równa się η/B; szum anizotropowy | Gałęzie lr×{½, 2} i B×{½, 2}: wysokość plateau MSD ∝ η/B, czas relaksacji w krokach ∝ 1/η |
| Znaczniki w harmonicznych klatkach plus dryf obrotowy ([Pashakhanloo 2023](https://doi.org/10.48550/arXiv.2302.02563)) | **testowalna** | Subdyfuzja po Procrustesie; rzadszy gatunek ruchliwszy | Klatka sama dryfuje, bo PR dalej spada | c(Δt) rośnie monotonicznie do 0,5; α₂ ≈ 0; bez Procrustesa liniowa część MSD |
| Segregacja jak neural collapse ([Papyan 2020](https://doi.org/10.1073/pnas.2015509117)) | **testowalna** | 95–99% narzędzia poza chmurą tła; trwa po plateau | Hamulec (−9 do −14% rozdziału od epoki 80) nie pasuje do monotonicznego NC | Tr(Σ_W Σ_B⁺) w skali log z załamaniem przy epoce 80; to samo w blokach 1–6 |
| Upraszczanie przez szum ([Chen 2023](https://doi.org/10.1088/1742-5468/ad6429); [Galanti 2022](https://arxiv.org/abs/2206.05794)) | **testowalna** | PR 10,6 → 3,2; kolimacja laserów | Rząd z WD nie maleje bez końca | Sonda spadku LR spowalnia spadek PR; przebieg z wd = 0 go osłabia |
| Szkło koloidalne z rozdziałem faz ([Weeks i Weitz 2002](https://doi.org/10.1016/s0301-0104(02)00667-5)) | **hipoteza, niesprawdzona** | Subdyfuzja; mniejszościowy gatunek ruchliwszy (jak cząstki B w mieszaninie Koba–Andersena) | Brak starzenia ruchliwości; siła MLP jednociałowa; DNN bez klatek ([Kerr Winter 2025](https://doi.org/10.1103/physrevresearch.7.023010)) | D1–D4 poniżej |
| Tokeny jako cząstki na sferze ([Geshkovski 2023](https://doi.org/10.48550/arXiv.2312.10794)) | testowalna **tylko w głębokości** | Grupowanie, stany metastabilne | Czas to głębokość; model bez MLP | Histogramy kosinusów i PR w blokach 1 → 6 przy stałym checkpoincie |
| Materia aktywna (wspólny „wiatr" 0,58 R) | metafora z jednym testem | Trwałe źródła siły, które powoli się obracają | Wiatr to jednorodna część kopnięcia w klatce, która nie zmienia struktury względnej | Znak autokorelacji przyrostów przy odstępach 10–30 epok (dodatni dla materii aktywnej, ujemny dla klatek) |
| Tworzenie pasów ruchu (lane formation, [Vissers 2011](https://doi.org/10.1039/C0SM01343A)) | metafora z jednym testem | Napęd zależny od gatunku | Pasy tworzą się równolegle do pola przez zderzenia poprzeczne; u nas rozdział biegnie wzdłuż osi | Ruchliwość narzędzia w poprzek osi rozdziału: zwiększona przed rozdziałem, stłumiona po nim |
| Jamming ([Liu i Nagel 2010](https://doi.org/10.1146/annurev-conmatphys-070909-104045)) | metafora | — | Dotyczy przestrzeni wag (więzy kontra parametry) | Brak testu na tokenach |
| Wiązanie optyczne i pułapki optyczne ([Dholakia i Zemánek 2010](https://doi.org/10.1103/RevModPhys.82.1767)) | metafora | Lasery jako źródła kierunków siły | Pułapki optyczne są przywracające; wiązanie optyczne to oddziaływanie między cząstkami, a siła MLP jest jednociałowa | Brak; ilościowo przenosi się tylko forma „natężenie × kierunek" (sekcja o polu) |
| Wir Browna ([Sun i in. 2009](https://doi.org/10.1103/PhysRevE.80.010401)) | metafora, **nie używać** | — | Wymaga szumu termicznego i prądu prawdopodobieństwa; pole ViT jest deterministyczne | Tylko po dodaniu modelu szumu i pomiarze prądów |

### Diagnostyki, które rozstrzygną obraz chmury

Wszystkie poniższe diagnostyki liczy się osobno dla tła i narzędzia, z Procrustesem i bez, oraz we współrzędnych względnych wobec klatki (przemieszczenie minus średnie przemieszczenie k sąsiadów). Przedziały ufności pochodzą z bootstrapu po 12 klatkach.

D1–D4 wymagają tylko zapisanych położeń. To ok. 0,25 GB dla 46 checkpointów po dołączeniu 20 nowych z [K400] i minuty CPU. Weeks i Weitz podają związek, z którego wynikają przewidywania dla D1: dla stacjonarnych przyrostów lokalny wykładnik γ ≈ 1 + log₂(1 − c). Stąd c ≈ 0,43 dla tła (γ = 0,20) i c ≈ 0,37 dla narzędzia (γ = 0,34) ([Weeks i Weitz 2002](https://doi.org/10.1016/s0301-0104(02)00667-5)).

| Diagnostyka | Co liczyć | Szkło przewiduje | NESS/OU przewiduje | Dane |
|---|---|---|---|---|
| D1. Antykorelacja klatki | c(Δt) = −⟨Δr(t,Δt)·Δr(t+Δt,Δt)⟩ / ⟨\|Δr\|²⟩ i odpowiedź warunkowa ⟨x₁₂⟩(r₀₁) | c dochodzi do ok. 0,5 i maleje do 0 przy ucieczce | c rośnie monotonicznie do 0,5 i zostaje | Istniejące; siatka co 10 epok |
| D2. Niegaussowskość | α₂ na osiach PCA; G_s w skali log; kontrola: przemieszczenia normowane per token | Pik α₂ przy Δt*, który przetrwa normowanie | α₂ ≈ 0 albo wyjaśnione statyczną niejednorodnością | Istniejące |
| D3. Starzenie dwuczasowe | MSD(t_w, Δt) dla t_w = 40–200; t* ∝ t_w^μ ([Herrera Segura 2022](https://doi.org/10.1088/2632-2153/ac8f1b)) | μ ≈ 0,5–1 albo a ≈ 0,05–0,5 | Zależność tylko od Δt (μ ≈ 0) | Potrzebne checkpointy do epoki 400 |
| D4. Niejednorodność dynamiczna | χ₄(Δt) z funkcji nakładania; zrywanie wiązań z histerezą A₁ < A₂ ([Yamamoto i Onuki 1998](https://doi.org/10.1103/PhysRevE.58.3515)); trwałość rankingu ruchliwości | Pik χ₄ przy τ_α; ranking ruchliwości się rozpada | Płaskie χ₄; ranking stały (statyczna niejednorodność) | Istniejące |
| D5. Test „temperatury" | Gałęzie (η, B) ∈ {(η/2, B), (2η, B), (η, 2B), (2η, 2B)} z epok 100 i 200, po 15–20 epok | Rozmiar klatki stały, czas ucieczki silnie zależny od T | Plateau MSD ∝ η/B; (2η, 2B) ≈ (η, B) ([Adhikari 2023](https://doi.org/10.48550/arXiv.2306.03521)) | 8 krótkich treningów |

Reguła werdyktu:
- c(Δt) rośnie do 0,5 i α₂ ≈ 0 → harmoniczne uwięzienie bez ucieczki;
- c(Δt) ma pik i maleje, α₂ ma pik, który przetrwa normowanie per token, a χ₄ ma pik → zachowanie szkliste;
- c ≈ 0 przy subdyfuzyjnym MSD → subdyfuzję tworzy zwalniający dryf albo artefakt wyrównania.

Do czasu tych pomiarów w artykule wolno pisać o „caging" tylko w sensie operacyjnym (podliniowe MSD), a słowa „glass-like" nie należy używać.

## Zero pola leży kilka promieni dalej, a topologia rozwarstwia gatunki

### Jak liczono

Pole jest dokładnie takie jak w aplikacji ([APP] `ogniskowa.py`):

f(u) = P3 Σ_c (z_c(x_j + P3ᵀu) − z̄_c) v_c

- z_c to wyjście GELU lasera c, a z̄_c to jego średnia w klatce;
- v_c to kolumna fc2;
- P3 to trzy główne kierunki znormowanych kolumn fc2 z checkpointu epoki 200, ten sam dla wszystkich 26 checkpointów.

Klasa koloidu wynika z wartości własnych jakobianu 3×3 M = P3 J P3ᵀ w punkcie koloidu:
- „przyciąga", gdy wszystkie Re λ < 0;
- „odpycha", gdy wszystkie Re λ > 0;
- „siodło" przy mieszanych znakach;
- „wir", gdy występuje para zespolona; to nakładka na klasę znaku.

Kontrole to 10 losowych przekrojów 3-D na koloid i pełny jakobian 192-D J = W2·diag(GELU′)·W1·∂LN/∂x. Próbka liczy do 30 koloidów narzędzia i 30 tła na klatkę. Ponieważ część klatek ma mniej tokenów narzędzia (8–72), łącznie jest **256 koloidów narzędzia i 360 tła** w 12 klatkach i 26 checkpointach [1 seed] ([KOL] `Kod_eksperymentu\pola_pulapek.py`, `Wyniki\Dane\POLA_PULAPEK.csv`, `POLA_PULAPEK_CZASTKI.npz`).

### Pułapka z aplikacji nie jest równowagą

Diagnostykę zera uruchomiono na wszystkich 7200 tokenach 12 klatek w epokach 0, 60 i 200 ([KOL] `Kod_eksperymentu\pulapki_newton_diag.py`). Użyto Newtona bez ograniczenia komórki, z tłumieniem do jednej komórki na krok. Wyniki:
- długość pola w samym koloidzie wynosi 3× połowę komórki w epoce 0 i 12–17× w epokach 60–200;
- Newton zbiega dla 0,6% tokenów w epoce 0 i 15–16% w epokach 60 i 200;
- znalezione zera leżą w medianie ok. 37 komórek, czyli 2–3,3 promienia chmury, od koloidu;
- zero leży w komórce w 0% przypadków; Newton ograniczony do komórki w `pola_pulapek.py` znalazł zero raz na 616 × 26 prób.

Zero leży więc **poza chmurą**. Nie jest stanem, do którego koloid mógłby się zbliżyć. Aplikacja w takim przypadku pokazuje minimum |f| w komórce ([APP] `ogniskowa.py`, opis modułu), a ten tryb zastępczy działa praktycznie zawsze.

To, co aplikacja klasyfikuje, jest jednak prawomocnym obiektem: to **topologia gradientu prędkości w układzie współporuszającym się z koloidem**. Po odjęciu f(0) macierz M opisuje, jak sąsiednie punkty poruszają się względem koloidu. Schemat Chonga, Perry'ego i Cantwella dotyczy dowolnego gładkiego pola 3-D ([Chong i in. 1990](https://doi.org/10.1063/1.857730)). W turbulencji stosuje się go do tensora gradientu prędkości w każdym punkcie przepływu, a nie tylko w punktach krytycznych ([Bechlars i Sandberg 2017](https://doi.org/10.1017/jfm.2017.40)). Najbliższy metodologiczny bliźniak w transformerach opisuje swoje etykiety zer pola rzutowanego jako "approximate-zero, local-spectrum diagnostics" ([Mudarisov i in. 2026, preprint](https://arxiv.org/abs/2608.02071)).

Druga poprawka dotyczy czasu. Blok rezydualny stosuje mapę x ↦ x + F(x) tylko raz, więc język przepływu to linearyzacja jednego kroku. Stabilność lokalna mapy wymaga |1 + λ| < 1, a nie tylko Re λ < 0.

### Topologia wokół koloidów przechodzi cztery fazy

| Epoka | Narzędzie P3: przyciąga / siodło / odpycha | Narzędzie: wir | Tło P3: przyciąga / siodło / odpycha | Tło: wir |
|---|---|---|---|---|
| 0 | 0,01 / 0,99 / 0,00 | 0,23 | 0,06 / 0,94 / 0,00 | 0,37 |
| 0,27 (krok 300) | 0,00 / 0,40 / **0,61** | 0,67 | 0,00 / 0,79 / 0,21 | 0,59 |
| 2,7 | 0,38 / 0,62 / 0,00 | 0,34 | 0,54 / 0,46 / 0,00 | 0,43 |
| 9,1 | 0,50 / 0,50 / 0,00 | 0,40 | 0,29 / 0,71 / 0,00 | 0,38 |
| 30 | 0,61 / 0,39 / 0,01 | 0,29 | 0,53 / 0,48 / 0,00 | 0,50 |
| 50 | 0,63 / 0,37 / 0,00 | 0,30 | **0,81** / 0,19 / 0,00 | 0,28 |
| 70 | 0,57 / 0,43 / 0,01 | 0,22 | 0,69 / 0,31 / 0,00 | 0,21 |
| 80 | 0,52 / 0,47 / 0,00 | 0,20 | 0,31 / **0,69** / 0,00 | 0,11 |
| 120 | 0,63 / 0,36 / 0,01 | 0,20 | 0,14 / **0,86** / 0,00 | 0,02 |
| 160 | 0,46 / 0,54 / 0,00 | 0,17 | 0,05 / **0,95** / 0,00 | 0,03 |
| 200 | 0,61 / 0,38 / 0,01 | 0,25 | 0,08 / **0,92** / 0,00 | 0,05 |

Źródło: [KOL] `POLA_PULAPEK.csv` [1 seed, 256 + 360 koloidów].

**Faza 1, inicjalizacja.** Prawie wszystkie koloidy siedzą w siodłach (99% narzędzia, 94% tła). Losowy przekrój i pełny jakobian dają w tym punkcie poziom losowy (niżej).

**Faza 2, „eksplozja" w kroku 300.** Ten krok to dokładnie koniec rozgrzewki LR (300 kroków, [K200] `Kod_eksperymentu\train_continue.py`). W rzucie P3 61% koloidów narzędzia i 21% tła odpycha we wszystkich trzech kierunkach, a udział wirów skacze do 59–67%. Chmura rośnie w tym czasie 2,6× (R 5,9 → 15,3), PR spada z 10,6 do 6,3, a jedna trzecia tokenów narzędzia już opuszcza chmurę tła ([KOL] `METRYKI_CHECKPOINTY.csv`).

**Faza 3, epoki 3–70.** Przeważa zbieżność: 50–65% narzędzia i do 82% tła w epokach 40–50.

**Faza 4, od epoki 80.** Gatunki się rozwarstwiają. Wokół narzędzia przyciąganie pozostaje najczęstszą klasą (43–72%, większość w 11 z 13 checkpointów; siodła 27–55%, wiry 15–25%). Tło przechodzi w siodła: 69% w epoce 80 i 78–95% w epokach 120–200 (w 7 z 9 checkpointów powyżej 85%), a wiry tła spadają do 2–13%.

Przejście tła w siodła zbiega się w czasie z dwoma zjawiskami. Pierwsze to start hamulca: rozdział po kopnięciu MLP maleje o 9–14% od epoki 80 ([KOL] `KOPNIECIE_A_ROZDZIAL.csv`). Drugie to wzrost liczby laserów tła ze 152 do 185 w epoce 80 i dalej do 192–250 ([KOL] `LASERY_PODSUMOWANIE.csv`). Tę zbieżność trzeba dopiero sprawdzić przyczynowo. Najtańszy test polega na sprawdzeniu, czy kierunek rozszerzania siodła tła pokrywa się z osią rozdziału, wzdłuż której działa hamulec.

### Część kontrastu przetrwa poza przekrojem P3, a część nie

| Epoka | Losowe przekroje: wir (N / T) | Losowe przekroje: siodło (N / T) | 192-D: zespolone (N / T) | 192-D: Re > 0 (N / T) | 192-D: ‖A‖/‖S‖ (N / T) | 192-D: tr J / 192 (N / T) |
|---|---|---|---|---|---|---|
| 0 | 0,55 / 0,59 | 0,68 / 0,75 | 0,93 / 0,93 | 0,50 / 0,50 | 1,00 / 1,00 | 0,000 / 0,000 |
| 0,27 | 0,55 / 0,62 | 0,69 / 0,66 | 0,93 / 0,93 | 0,49 / 0,49 | 0,99 / 0,99 | −0,001 / −0,001 |
| 9,1 | 0,56 / 0,56 | 0,73 / 0,79 | 0,90 / 0,90 | 0,47 / 0,47 | 0,92 / 0,91 | −0,006 / −0,005 |
| 40 | 0,52 / 0,57 | 0,56 / 0,64 | 0,88 / 0,88 | 0,46 / 0,45 | 0,86 / 0,89 | −0,016 / −0,016 |
| 80 | 0,55 / 0,62 | 0,52 / 0,58 | 0,88 / 0,87 | 0,45 / 0,44 | 0,87 / 0,90 | −0,019 / −0,017 |
| 120 | 0,47 / 0,48 | 0,55 / 0,60 | 0,87 / 0,88 | 0,45 / 0,44 | 0,86 / 0,89 | −0,022 / −0,018 |
| 160 | 0,46 / 0,50 | 0,57 / 0,74 | 0,88 / 0,88 | 0,45 / 0,44 | 0,88 / 0,88 | −0,023 / −0,014 |
| 200 | 0,47 / 0,47 | 0,57 / 0,77 | 0,88 / 0,87 | 0,45 / 0,45 | 0,86 / 0,89 | −0,024 / −0,013 |

N = narzędzie, T = tło; mediany dla wielkości 192-D ([KOL] `POLA_PULAPEK.csv`) [1 seed].

**Siodła tła są częściowo odporne na wybór przekroju.** W losowych przekrojach siodła tła stanowią 0,72–0,77 wobec 0,53–0,59 dla narzędzia (epoki 160–200). Zbieżność we wszystkich trzech kierunkach widać więc wokół narzędzia w ok. 41–47% losowych przekrojów, a wokół tła w ok. 23–28%. Losowa macierz 3×3 daje tu 10,5% (Monte Carlo własne, 200 000 macierzy Ginibre'a).

Kontrast potwierdza też dywergencja, czyli tr J/192, wielkość niezależna od przekroju. Do epoki 80 jest taka sama u obu gatunków (−0,016 wobec −0,016 w epoce 40). Od epok 110–130 narzędzie jest ok. 1,5× bardziej ściskające: średnio −0,024 wobec −0,016 w epokach 120–200 (wyliczenie własne z `POLA_PULAPEK.csv`). P3 wyostrza ten kontrast: w epoce 200 siodła stanowią 92% tła wobec 38% narzędzia. Ponieważ P3 nie jest losowy, jego liczby trzeba kalibrować przekrojami losowymi i przekrojami Schura.

**Zanik wirów wokół tła to własność przekroju P3, a nie pola.** W losowych przekrojach wiry tła stanowią 0,44–0,50. Pełny stosunek rotacji do odkształcenia dla tła (0,87–0,90) jest nawet nieco wyższy niż dla narzędzia (0,85–0,88).

Ta sama lekcja wynikała z [PW] dla par wiązek. Rzut do 3D zachowuje sumy i dokładne znoszenie, ale fabrykuje statystyki na poziomie przypadku, więc liczby z przekroju bez modelu zerowego nie są pomiarem.

### Trwałość klasy pojedynczej cząstki jest słaba i podobna u obu gatunków

Surowa trwałość klasy (znak plus wir) między kolejnymi checkpointami różni się mocno między gatunkami. Dla tła rośnie po epoce 120 do 0,74–0,85 („zamrożone siodła"). Dla narzędzia zostaje na poziomie 0,46–0,57 („migoczące pułapki") ([KOL] `Wyniki\Code_Interpretujacy_Wyniki\pokaz_pola_pulapek.py`).

Ta różnica wynika jednak głównie z rozkładu klas. Jeśli 78–95% tła to siodła, to dwa niezależne losowania dadzą tę samą klasę z prawdopodobieństwem 0,63–0,79. Dla narzędzia, o bardziej równym rozkładzie klas, to prawdopodobieństwo wynosi 0,32–0,38. Po korekcie na przypadek kappa Cohena w epokach 120–200 wynosi 0,15–0,29 dla tła i 0,19–0,33 dla narzędzia (wyliczenie własne z `POLA_PULAPEK_CZASTKI.npz`). W epokach 10–120 κ wynosi 0,22–0,38 dla narzędzia i 0,05–0,32 dla tła.

**Stała jest statystyka populacji („tło to siodła"), a nie charakter pola wokół konkretnej cząstki**, który co 10 epok w dużej mierze się przetasowuje. Sformułowania „zamrożone siodła" i „migoczące pułapki" trzeba wycofać.

### Co certyfikuje przekrój 3-D, a co tylko pełne pole

Dla niesymetrycznego J wartości własne kompresji M = P3 J P3ᵀ to wartości Ritza. Gwarantowane jest tylko to, że leżą wewnątrz zakresu numerycznego W(J), a ich części rzeczywiste leżą między λ_min i λ_max części symetrycznej S ([Loisel 2018](https://doi.org/10.1137/17M1148608); [Carden i Embree 2012](https://doi.org/10.1137/120872693)). Z tego wynikają dwie granice interpretacji:
- **siodło w przekroju** dowodzi tylko, że S jest nieokreślona; nie dowodzi, że J ma wartość własną z Re > 0;
- **wir w przekroju** dowodzi tylko, że część antysymetryczna ma wagę w tej płaszczyźnie.

Przekrój jest wierny dokładnie wtedy, gdy P3 rozpina podprzestrzeń niezmienniczą J, np. wektory Schura. Miarą wierności jest reszta niezmienniczości η_inv = ‖(I − P3ᵀP3) J P3ᵀ‖ / ‖J P3ᵀ‖.

Małe przekroje mają też wbudowane obciążenie w stronę odkształcenia. Dla macierzy o niezależnych elementach stosunek energii ‖S‖²/‖A‖² wynosi (n+1)/(n−1): 2 dla n = 3 i 1,01 dla n = 192 (wyliczenie własne; zgodne z wyprowadzeniem dla losowych ramek Haara w [NOT] `vortex_saddle_traps.md`). Stąd losowa macierz 3×3 ma 65% par zespolonych i 79% siodeł (Monte Carlo własne).

Nasze losowe przekroje przy inicjalizacji dają 55–59% wirów i 68–75% siodeł, czyli blisko tego poziomu. Po treningu dają 41–47% wirów. Trening przesuwa więc lokalne pole w stronę mniejszej rotacji i większej zbieżności, co zgadza się z pełnym jakobianem.

### Pełny jakobian: zespolone wartości własne to poziom zerowy

Wartości przy inicjalizacji pokrywają się z losowymi modelami odniesienia:
- **Zespolone wartości własne:** 93% wobec 94% dla macierzy Ginibre'a 192×192. Asymptotyka √(2n/π) daje ok. 11 rzeczywistych wartości własnych ([Edelman i in. 1994](https://doi.org/10.1090/S0894-0347-1994-1231689-0)); Monte Carlo własne daje 94,1%.
- **Udział Re > 0:** 50% wobec 50%.
- **Stosunek rotacji do odkształcenia ‖(J−Jᵀ)/2‖ / ‖(J+Jᵀ)/2‖:** 1,00 wobec 0,995 dla macierzy o niezależnych elementach (wyliczenie własne).

**Same zespolone wartości własne i mieszane znaki nie są więc znaleziskiem.** To samo dotyczy liczby 98% zespolonych w blokach LLM ([Fernando i Guitchounts 2026, preprint](https://doi.org/10.48550/arXiv.2605.14258)). Dla d = 4096 macierz Ginibre'a daje ok. 98,8% (wyliczenie własne). Nie jest to dowód na rotację „zainstalowaną" w LLM.

Po treningu (od epoki 40) udział zespolonych spada do 87–88%, czyli z ok. 13 do ok. 23 rzeczywistych wartości własnych na ok. 190 niezerowych. Udział Re > 0 spada do 44–46%, a stosunek rotacji do odkształcenia do 0,85–0,90 (w epokach 15–30 tło jeszcze 0,91–0,92). Udział energii części symetrycznej rośnie więc z 0,50 do 0,56–0,58 (wyliczenie własne: 1/(1 + ρ²)).

Porównania z literaturą:
- **Kierunek zmiany** zgadza się z autoenkoderami, w których symetryczność jakobianu rośnie w treningu ([Im i in. 2016](https://doi.org/10.1609/aaai.v30i1.10268)).
- **Skala zmiany** leży na dolnym końcu przepływu tokenów w Pythii: udział symetryczny 0,53–0,84 wobec modelu zerowego 0,50–0,54, ale w 16-wymiarowej przestrzeni PCA ([Quemy 2026, preprint](https://arxiv.org/abs/2608.29706)).
- **Fernando i Guitchounts** raportują odwrotny kierunek, ale ich jakobian obejmuje cały blok z tożsamością rezydualną. Tożsamość czyni go przy inicjalizacji prawie symetrycznym, a nasz J to sam podblok MLP, więc liczby nie są porównywalne.

### Każdy laser to formalnie ciśnienie promieniowania

Laser c wnosi do pola siłę (z_c(x) − z̄_c) v_c, czyli „natężenie" razy stały „kierunek propagacji". To ta sama forma co człon ciśnienia promieniowania f₁ I(r) ẑ w pęsecie optycznej ([Roichman i in. 2008](https://doi.org/10.1103/PhysRevLett.101.128301)). Krętość optyczna pochodzi z gradientu natężenia prostopadłego do kierunku propagacji ([Wu i in. 2009](https://doi.org/10.1103/PhysRevLett.103.108101)).

Jakobian lasera ma rząd 1: J_c = g_c v_c w̃_cᵀ, gdzie g_c = GELU′ i w̃_c = (∂LN/∂x)ᵀ w_c. Z tego wynikają trzy własności (wyprowadzenie własne w [NOT] `vortex_saddle_traps.md`, do weryfikacji numerycznej):
- część antysymmetryczna lasera rośnie jak |sin θ_c|, gdzie θ_c to kąt między kierunkiem „czytania" w̃_c i „pchania" v_c;
- dywergencja tr J = Σ_c g_c ‖v_c‖‖w̃_c‖ cos θ_c, więc ujemna i rosnąca dywergencja oznacza, że aktywne lasery są coraz bardziej antyrównoległe;
- lasery w ujemnym zakresie GELU wnoszą wkład z odwróconym znakiem.

Pole jest konserwatywne wtedy i tylko wtedy, gdy zsumowany jakobian jest symetryczny przy każdym stanie tokenu. Ten warunek dla pól w kształcie MLP podali [Im i in. 2016](https://doi.org/10.1609/aaai.v30i1.10268).

Optyka podpowiada też skalę porównawczą:
- w zwykłej pęsecie jednowiązkowej stosunek rotacji do sztywności wynosi tylko 0,7–3,8% ([Pesce i in. 2009](https://doi.org/10.1209/0295-5075/86/38002));
- nasze 0,85–0,90 przypomina raczej wiązki wirowe i duże skupiska „optical matter", w których przy N > 50 większość modów jest zespolona ([Li i in. 2021](https://doi.org/10.1038/s41467-021-26732-8)).

Słownik optyki ma jednak granice. „Wir Browna" wymaga szumu termicznego i prądu prawdopodobieństwa, a nie samej deterministycznej części solenoidalnej ([Sun i in. 2009](https://doi.org/10.1103/PhysRevE.80.010401)). Tego terminu nie należy używać.

## Lasery kolimują się w 30 epok, a potem dyfundują po sferze

Wszystkie liczby w tabeli są z [KOL] `METRYKI_CHECKPOINTY.csv`, `LASERY_PODSUMOWANIE.csv`, `KOPNIECIE_A_ROZDZIAL.csv` i `KORELACJE_IOU_TEMPERATURA.csv` [1 seed, 12 klatek].

| Cecha laserów (blok 6, 768 neuronów) | Wartość [1 seed] | Odczytanie |
|---|---|---|
| Efektywna liczba kierunków wyjścia | 154 → 31,7 (ep. 30) → min. 30,8 (ep. 40) → 36,8 (ep. 200) | Kolimacja, potem powolne ponowne rozchodzenie się |
| Średnia norma kolumny fc2 („moc") | 0,29 → 0,71 (ep. 30) → 0,82 (ep. 60) → 0,92 (ep. 200), nadal rośnie | Blisko skali równowagi AdamW (niżej) |
| Nierówność mocy (Gini) | 0,02 → 0,18–0,19 (ep. 9–15) → 0,16–0,17 do ep. 200 | Ustalona hierarchia mocy |
| Nierówność użycia (Gini) | 0,56–0,61 od ep. 15 | Użycie bardziej skupione niż moc |
| Kąt kierunku po 10 epokach | 21,4–23,4° od ep. 50 (2,14–2,33°/epokę) | Stacjonarny w czasie; to nie stała prędkość kątowa (niżej) |
| Korelacja obrotu z użyciem (Spearman) | −0,33 do −0,44 od ep. 22; −0,40 do −0,44 od ep. 70 | Bardziej używane lasery obracają się wolniej |
| Lasery selektywne dla tła (s < −0,5) | 0 → 144 (ep. 30) → 192–250 (ep. 110–200), 221 w ep. 200 | Ok. 25–33% laserów specjalizuje się w tle |
| Lasery selektywne dla narzędzia (s > 0,5) | Szczyt 62 (ep. 22) i 61 (ep. 30), potem 15–24 (ep. 160–200) | Specjalizacja przesuwa się z narzędzia na tło po plateau IoU |
| Elita top-50 według użycia | Co 10 epok zostaje 56–76% | Wymiana ok. ⅓ elity na dekadę epok |
| Lasery ciemne (nigdy > 5% najsilniejszego lasera na tokenie, 12 klatek) | 43–79 od ep. 22; rzadkie (< 0,1% tokenów) 65–99 | Niestała grupa „migających" laserów |
| Średni udział tokenów z dodatnim GELU | 0,51 → 0,19–0,25 od ep. 9 | Typowy laser przez 75–80% czasu pracuje w ujemnym ogonie |
| Udział siły z ujemnego GELU | 0,28–0,36 od ep. 2,7 | Ok. 30% siły to „ciągnięcie" |
| Wypadkowa / suma długości sił na cząstce | 0,15–0,19 | Silne wzajemne znoszenie (w 192-D zgodność sił jednak ok. 2× większa niż dla losowych kierunków, [PW]) |
| Hamulec: pchnięcie tła wzdłuż osi rozdziału | +0,22–0,35 R od ep. 80; rozdział po kopnięciu −9 do −14% (typowo −11 do −13%) | Tło pchane w stronę narzędzia |
| Wielkość hamulca, liczba laserów na jego połowę | ok. 7× wzrost (−0,05 → −0,34 do −0,43); połowę daje 10–19 laserów | Hamulec skupiony w kilkunastu laserach |
| Wspólny „wiatr" a część indywidualna kopnięcia | 0,50–0,66 R (0,58 w ep. 200) wobec 0,29–0,37 R | Wiatr przesuwa całą chmurę, nie zmienia struktury względnej |

### Obrót to stacjonarna dyfuzja rotacyjna, a nie stała prędkość kątowa

Liczba „2,2°/epokę" to średni kąt kierunku lasera między checkpointami odległymi o 10 epok, podzielony przez 10 ([KOL] `Kod_eksperymentu\dynamika_koloidow.py`). Ta wartość jest rzeczywiście stała od epoki 50 do 200 (2,14–2,33°/epokę). Gdy jednak policzyć kąt dla wszystkich odstępów na plateau z zapisanych kierunków, kąt nie rośnie liniowo (wyliczenie własne z [KOL] `LASERY_26_CHECKPOINTOW.npz`, epoki 60–200):

| Odstęp | Średni kąt | ⟨cos⟩ | Nieskorelowany spacer o kroku Kossona | Pomiar / spacer |
|---|---|---|---|---|
| 10 epok | 21,9° | 0,915 | 9,7° | 2,3× |
| 20 epok | 28,4° | 0,861 | 13,7° | 2,1× |
| 40 epok | 37,9° | 0,767 | 19,4° | 2,0× |
| 80 epok | 51,8° | 0,599 | 27,5° | 1,9× |
| 140 epok | 66,2° | 0,395 | 36,4° | 1,8× |

Kąt rośnie jak odstęp^0,4. ⟨cos⟩ dobrze opisuje 0,98·exp(−Δt/τ) z τ ≈ 150 epok. To podpis **dyfuzji rotacyjnej po sferze**, z nasyceniem kąta przy dużych odstępach i małą szybką składową (ok. 2%), a nie sztywnego obrotu ze stałą prędkością.

Liczba kroków na epokę (35 000/32 = 1093,75) i skala równowagi ok. 37 epok w notatkach są poprawne. Porównanie „3,1°/epokę przewidywane wobec 2,2°/epokę zmierzone" zestawiało jednak spacer losowy po jednej epoce ze zmierzonym kątem po 10 epokach podzielonym przez 10.

Przy zgodnym odstępie wynik jest inny. Krok Kossona przy AdamW to η̂ = √(2ηλ(1−β₁)/(1+β₁)) ≈ 1,62·10⁻³ rad, przy η = 5·10⁻⁴, λ = 0,05, β₁ = 0,9 ([Kosson i in. 2023](https://arxiv.org/abs/2305.17212); hiperparametry z [K200] `Opis\OPIS_EKSPERYMENTU.txt`). Nieskorelowany spacer tymi krokami daje 9,7° po 10 epokach i τ ≈ 700 epok (wyliczenie własne). Zmierzony kąt jest 2,3× większy, a dekorelacja ok. 4,6× szybsza.

**Wniosek z notatek o „lekko antyskorelowanych aktualizacjach" należy więc wycofać.** Kolejne aktualizacje są raczej dodatnio skorelowane w oknie kilku epok albo efektywny krok kątowy kolumn fc2 jest ok. 2× większy niż u Kossona. To drugie jest możliwe, bo kolumny fc2 nie są niezmiennicze względem skali, a założenie Kossona tego wymaga. Rozstrzygnie to odcinek treningu z zapisem co 10 kroków.

### Skala Kossona pasuje do normy, ale nie do jednorodności

Dla wektora o C = 192 elementach równowaga AdamW przewiduje normę ≈ √(ηC/(2λ)) ≈ 0,98 i czas dojścia ok. 1/(ηλ) = 40 000 kroków ≈ 37 epok ([Kosson i in. 2023](https://arxiv.org/abs/2305.17212); liczby to wyliczenie własne). Zmierzona średnia norma 0,92 w epoce 200 to 94% tej wartości. Koniec „ochładzania" chmury i kolimacji (epoki 30–40) zgadza się z tą skalą czasu co do rzędu wielkości.

Dwie rzeczy odbiegają od prostego obrazu:
- **Norma nadal rośnie.** W epokach 60–200 wzrosła z 0,82 do 0,92, choć czas e-krotny kwadratu normy to tylko ok. 18 epok.
- **Lasery nie są jednorodne.** Gini mocy stoi na 0,16–0,17, a bardziej używane lasery obracają się wolniej (ρ ≈ −0,40 do −0,44 od epoki 70). Kosson przewiduje ten sam krok kątowy dla wszystkich wektorów niezmienniczych względem skali i zastrzega, że wektory wrażliwe na skalę mogą obracać się nierówno.

W artykule zgodność ze skalą Kossona należy więc podać jako zgodność rzędu wielkości, a niejednorodność jako osobny, mierzalny wynik.

### Specjalizacja, hamulec i ujemne GELU

Kolimacja ze 154 do 31 kierunków to przejaw przyciągania do zbiorów „identycznych neuronów" przy dużym, stałym LR ([Chen i in. 2023](https://doi.org/10.1088/1742-5468/ad6429)). Test operacyjny to udział niezależnych neuronów według kryteriów tej pracy (znikający: normy < 10% maksimum warstwy; identyczne: znormalizowana odległość < 10%).

Ponowne rozchodzenie się do 37 kierunków po epoce 40 zgadza się z wynikiem z [PW]. Po epoce 110 częściowe kopie przeważnie się różnicują: 20 z 30 par T150 rozeszło się, a scaliła się tylko jedna para.

Specjalizacja przesuwa się po plateau IoU z narzędzia na tło. Laserów narzędzia jest najwięcej w epokach 22–30 (61–62), potem spadają do 15–24. Laserów tła przybywa aż do 192–250 (221 w epoce 200).

Hamulec rośnie ok. 7× i jest skupiony: połowę daje 10–19 laserów. Tło jest pchane wzdłuż osi rozdziału w stronę narzędzia o 0,22–0,35 R, a narzędzie lekko w stronę tła (−0,01 do −0,02 R).

Ok. 30% siły na cząstce pochodzi z laserów w ujemnym zakresie GELU. To zgodne z [PW]: skrajne przeciwieństwo wiązek wynika ze znaku GELU nad prawie równoległymi kolumnami W2, a nie z geometrii wag.

### Ciemne lasery a IoU: wynik na samym suficie szumu

Wstępne korelacje policzono na plateau, w epokach 40–200. Użyto pierwszych różnic (15 zmian), korelacji Spearmana z p permutacyjnym (10 000 permutacji) i 72 testów. IoU pochodzi z 500 VAL + 500 TEST, czyli z innych klatek niż 12 klatek analizy ([KOL] `KORELACJE_IOU_TEMPERATURA.csv`).

| Metryka | Zmiany IoU VAL | Zmiany IoU TEST |
|---|---|---|
| Liczba ciemnych laserów | r = −0,69 (p = 0,004) | r = −0,79 (p = 0,0005) |
| Liczba laserów narzędzia | r = +0,61 (p = 0,015) | r = +0,68 (p = 0,006) |
| „Temperatura" cząstek | brak związku (\|r\| ≤ 0,17) | brak związku |
| Obrót laserów | brak związku (0,19 i 0,34, n.s.) | brak związku |

Część innych metryk daje istotny wynik tylko na jednym zbiorze. Przykłady: wymiar wiązki daje −0,63 na VAL i −0,12 na TEST, rozdział daje 0,11 na VAL i 0,57 na TEST. To pokazuje poziom szumu.

Trzy zastrzeżenia są tu decydujące:
- **Sufit szumu.** Zmiany IoU na VAL i TEST korelują ze sobą tylko na r = 0,62. Przy takiej rzetelności korelacja dowolnej miary z jednym zbiorem nie powinna przekraczać √0,62 ≈ 0,79 (wyliczenie własne). Wynik z TEST leży dokładnie na suficie, co przy 15 punktach jest podejrzanie dobre.
- **Wielokrotność testów.** Przy 72 testach poprawka Bonferroniego (0,0007) zostawia tylko wynik z TEST.
- **Mała próba.** Status „ciemny" liczono na 12 klatkach.

Migające lasery nie są specjalistami narzędzia (0%). Są rzadkie, raczej związane z tłem i aktywne na ok. 0,75% tokenów ([KOL] `Wyniki\Code_Interpretujacy_Wyniki\migajace_lasery.py`).

Uczciwe brzmienie to hipoteza: checkpointy, w których więcej rzadkich laserów tła gaśnie na tych 12 klatkach, wykrywają nieco gorzej. Hipotezę trzeba zarejestrować przed replikacją na nowych seedach i większej liczbie klatek.

## Mikroskop może ważyć pomiary filtra, ale etykiety okluzji trzeba wytworzyć

### W kataraktach nie ma etykiet okluzji

Nie znaleziono publicznego zbioru katarakty z jawnymi etykietami okluzji lub widoczności narzędzia:
- Cataract-LMM (podzbiór `3_Object_Tracking`: 170 klipów kapsuloreksji, 469 118 klatek, maski, ramki, trwałe ID, punkty końcówki, CC-BY 4.0) nie ma flagi widoczności. Przy anotacji przerwy z okluzją mostkowano filtrem Kalmana ([Ahmadi i in. 2026](https://doi.org/10.1038/s41597-026-07464-0)).
- Cataract-1K nie ma etykiet okluzji, a klatki z silnym rozmyciem wykluczono ([Ghamsarian i in. 2024](https://doi.org/10.1038/s41597-024-03193-4)).
- CATARACTS etykietuje tylko „narzędzie w użyciu" ([Al Hajj i in. 2019](https://doi.org/10.1016/j.media.2018.11.008)).
- Dla CaDIS przeczytano tylko abstrakt, więc brak etykiet nie jest w pełni potwierdzony ([Grammatikopoulou i in. 2021](https://doi.org/10.1016/j.media.2021.102053)).

Jawne etykiety istnieją tylko w laparoskopii i robotyce:

| Zbiór | Domena | Rodzaj etykiety | Liczby | Źródło |
|---|---|---|---|---|
| CholecTrack20 | laparoskopia | Etykieta „occlusion" wśród 8 wyzwań wizualnych | 20 filmów, 35 tys. klatek | [Nwoye i in. 2025](https://doi.org/10.1109/CVPR52734.2025.00836) |
| AIS | robotyka | Amodalne maski narzędzi | 1455 z 7084 instancji zasłoniętych, średnio 20,5% | [Shi i in. 2024](https://doi.org/10.1007/s11220-025-00609-2) |
| SegSTRONG-C | robotyka | Sparowane nagrania czyste i zakłócone | — | [Ding i in. 2024](https://doi.org/10.48550/arXiv.2407.11906) |
| ClipAssistNet | laparoskopia | Widoczność końcówki klipsownicy | Niezgodność anotatorów 18% klatek | [Aspart i in. 2021](https://doi.org/10.1007/s11548-021-02441-x) |

Skala problemu jest znana ze zbiorów innych niż katarakta:
- **SegSTRONG-C:** krew obniża Dice czterech modeli z 0,894 do 0,645 ([Ding i in. 2024](https://doi.org/10.48550/arXiv.2407.11906)).
- **ViT na prawdziwej okluzji:** spadek 13–14 pp przy ≤ 50% zasłonięcia i 29–32 pp powyżej 50%, podobnie jak w CNN ([Kassaw i in. 2024](https://doi.org/10.48550/arXiv.2409.10775)).
- **CholecTrack20:** głowice typu CenterNet i FCOS, takie jak nasza, mają na klatkach z okluzją AP 51,7 i 40,8 wobec 82,1 dla YOLOv7, bez podanej wartości dla klatek normalnych ([Nwoye i in. 2025](https://doi.org/10.1109/CVPR52734.2025.00836)).
- **Cataract-LMM:** przesunięcie domeny między ośrodkami (mAP 70,45 → 47,12) bywa większe niż efekt okluzji ([Ahmadi i in. 2026](https://doi.org/10.1038/s41597-026-07464-0)).

Najmocniejszą znaną poprawą jest **augmentacja okluzją w treningu**:
- Dice na zakłóconym teście 0,512 → 0,735 ([Shao 2025](https://doi.org/10.54254/2755-2721/2025.ld30738));
- błąd końcówki 73,5 → 29,5 mm, a z rekurencją 25,9 mm ([Agethen i in. 2025](https://doi.org/10.1007/s11548-025-03436-8)).

Śledzenie świadome niepewności daje umiarkowane zyski:
- przełączenia ID −19% do −55% i HOTA +0,5 do +1,7 ([Lee i in. 2024](https://doi.org/10.1109/ICRA57147.2024.10610458));
- głowica widoczności PermaTrack: HOTA +5,0 i +8,2 ([Tokmakov i in. 2021](https://doi.org/10.1109/ICCV48922.2021.01068)).

Metody mikroskopowe trzeba więc porównywać z augmentacją okluzją, a nie z brakiem augmentacji.

### Co mikroskop przewiduje dla zasłoniętej końcówki

Nasze pomiary dają trzy punkty zaczepienia:
- tokeny narzędzia tworzą osobny klaster (95–99% poza chmurą tła);
- wokół narzędzia pole jest ściskające, a wokół tła siodłowe (od epoki 80–120, także w dywergencji niezależnej od przekroju);
- hamulec pcha tło w stronę narzędzia, a pewność głowicy rośnie także na pustych klatkach ([PW]).

Ostatni punkt przypomina znany mechanizm błędu. Przy silnie zasłoniętym obiekcie cechy kontekstu wywołują fałszywe detekcje, a pomaga konkurencja „obiekt kontra okluder" w każdym położeniu ([Wang i in. 2020](https://doi.org/10.1109/cvpr42600.2020.01266)). W tym ujęciu ciężka okluzja dała +41,2 AP, za cenę −5,4 AP na obiektach niezasłoniętych.

Dla zasłoniętej końcówki wynikają z tego trzy przewidywania. Wszystkie są hipotezami:
1. Zakryte tokeny przesuną się w stronę chmury tła, czyli ujemny log-iloraz wiarygodności narzędzie/tło.
2. Stracą ściskającą dywergencję narzędzia na rzecz wartości typowych dla tła.
3. Stracą aktywność laserów narzędzia.

Tokeny na granicy okludera powinny być mieszane, bo cienka końcówka w patchu 16 px siatki 20×30 tworzy jeden token po części narzędzia, po części tkanki.

Brak równowag zmienia jedną propozycję z notatek. „Sztywność pułapki jako odwrotność wariancji pomiaru" nie ma sensu, skoro pułapki nie ma. Lepiej użyć dywergencji tr J/192 albo wzmocnienia kierunkowego ‖(I + J)δ‖/‖δ‖ wzdłuż rzeczywistego zaburzenia δ = r(zasłonięty) − r(czysty). Obie wielkości mówią, czy blok tłumi zaburzenie wywołane okluzją.

Odległości trzeba liczyć na znormowanych resztach, z kowariancją wspólną i shrinkage. Sama norma cech nie wskazuje okluzji, a normowanie przed odległością Mahalanobisa poprawia wykrywanie OOD w 41 z 44 modeli ([Mueller i Hein 2025](https://doi.org/10.48550/arXiv.2505.18032)).

| Wielkość mikroskopu | Definicja operacyjna | (a) Detektor okluzji | (b) Wejście filtra cząsteczkowego | (c) Sygnał treningowy |
|---|---|---|---|---|
| Położenie wobec klastrów | LLR_i = log N(r̂_i; μ_N, Σ) − log N(r̂_i; μ_T, Σ), r̂ znormowane L2, Σ wspólna z shrinkage, dopasowane na rozłącznym podziale | Udział tokenów ramki z LLR < 0 | Temperatura wiarygodności β_t albo szum R_t | Strata spójności: token bliźniaka zasłoniętego w stronę czystego |
| Promień chmury tła | Znormowany promień Mahalanobisa w chmurze tła | Tokeny ramki wracają do promienia 95% | Bramka: odrzucić pomiar z tokenami wewnątrz chmury tła | Kara za tokeny tła poza promieniem na pustych klatkach |
| Dryf w głąb sieci | d_rel(l) = 2 d_N(l) / (d_N(l) + d_T(l)) dla tokenów maski, w blokach 1–6 ([Ernst i in. 2019](https://doi.org/10.1007/978-3-030-30508-6_24)) | d_rel > 1 w ostatnich blokach | d_rel bloku 6 jako niezawodność | Nagroda za spadek d_rel w głąb na bliźniakach |
| Hamulec | Rzut kopnięcia MLP na (μ_N − μ_T)/‖·‖ | Hamulec na tle pustych klatek = podejrzenie fałszywej detekcji | Zwiększyć R_t przy dużym hamulcu w miejscu detekcji | Kara za hamulec na pustych klatkach |
| Lasery narzędzia | Suma aktywności ustalonego zbioru laserów narzędzia (selekcja na rozłącznych danych) | Krzywa dawka–odpowiedź wobec ułamka zasłonięcia; czy spada przed pikiem mapy cieplnej | Niska aktywność jako cecha niepewności | Wzmocnienie lub ablacja (test przyczynowy) |
| Lokalne pole | tr J/192; wzmocnienie kierunkowe wzdłuż δ; udział zbieżności w losowych przekrojach | Zakryte tokeny tracą ściskanie narzędzia | Silne ściskanie = stabilniejszy pomiar | Porównanie przed i po augmentacji |
| Głowica (punkt odniesienia) | Pik mapy cieplnej, IoU, ECE | Detektor odniesienia | R_t(pewność) | — |

### Plan eksperymentu

**Krok 0, dane.**
- **Bliźniaki syntetyczne.** 300–500 klatek z maskami lub ramkami narzędzia, z własnych danych, Cataract-1K albo Cataract-LMM. Dla każdej klatki powstają bliźniaki okluzyjne:
  - okludery: łaty tęczówki, soczewki i tkanki wycięte z tej samej klatki, półprzezroczyste czerwone elipsy krwi (α 0,2–0,6) i białe odblaski ([Shao 2025](https://doi.org/10.54254/2755-2721/2025.ld30738));
  - położenie: na końcówce albo na trzonie;
  - zasłonięta część maski: 0, 10–30, 30–50 i 50–70%;
  - co najmniej jeden rozmiar mniejszy niż patch.
- **Zbiór prawdziwy.** Ok. 200 klatek z etykietą widoczna / częściowo / zasłonięta / niepewna, od 2 anotatorów, z kappa Cohena. Kandydatów najłatwiej znaleźć w Cataract-LMM, bo trajektorie z punktem końcówki mostkowane Kalmanem wskazują przerwy.
- **Puste klatki.** Zbiór klatek bez narzędzia.

**Krok 1, opis.** Na zamrożonych modelach z epok 40, 100, 200 (później 300 i 400) i co najmniej 3 seedach liczy się wszystkie wielkości z tabeli. Liczy się je osobno dla tokenów zakrytych, granicznych i widocznego trzonu oraz dla tła na klatkach z narzędziem i na pustych.

**Krok 2, detektor okluzji.**
- Miara: AUROC i AUPRC dla „zasłonięta lub częściowo" wobec „widoczna", najpierw na etykietach syntetycznych, potem na prawdziwych.
- Odniesienia: pewność głowicy, wariancja zespołu 3 seedów albo MC dropout, rozrzut atencji ([Ruppel i in. 2022](https://doi.org/10.48550/arXiv.2210.14391)).
- Kryterium sukcesu: przewaga ≥ 0,05 AUROC nad pewnością na zbiorze prawdziwym, z przedziałem bootstrapowym wykluczającym 0.

**Krok 3, filtr cząsteczkowy.** Odtworzenie sekwencji z epizodami okluzji i porównanie pięciu wariantów:
1. stały R;
2. R z pewności głowicy;
3. R lub β_t z wyniku mikroskopu;
4. filtr ze stanem widoczności v ∈ {widoczna, zasłonięta} na każdej cząstce, z macierzą przejść i p(v | wynik mikroskopu); hipotezy zasłonięte propaguje model ruchu i nie giną przy braku pomiaru;
5. wyrocznia widoczności.

Metryki: RMSE końcówki w czasie i po okluzji, odsetek utraconych torów, opóźnienie ponownego uchwycenia, przełączenia ID i fałszywe narodziny torów na pustych klatkach.

**Krok 4, interwencje na zamrożonym modelu.**
- Wzmocnienie laserów narzędzia (×1,5, ×2) albo ich ablacja na bliźniakach, z kontrolami: losowe lasery w tej samej liczbie i ciemne lasery w tej samej liczbie.
- Usunięcie hamulca: odjęcie składowej kopnięcia wzdłuż μ_N − μ_T na tokenach tła.

Miary to recall i IoU na bliźniakach oraz fałszywe detekcje i pewność na pustych klatkach. Jeśli usunięcie hamulca obniża pewność na pustych klatkach i jednocześnie recall zasłoniętego narzędzia, hamulec działa także jako mechanizm „dopełniania" amodalnego.

**Krok 5, trening.** Krótki harmonogram na tych samych seedach, w pięciu ramionach:
1. baza;
2. augmentacja okluzją tkanką i krwią;
3. augmentacja plus strata spójności bliźniaków na tokenach bloku 6;
4. augmentacja plus głowica widoczności per token;
5. opcjonalnie kara za hamulec na pustych klatkach.

Po treningu mikroskop uruchamia się ponownie. Przewidywanie: augmentacja zmniejsza d_rel zakrytych tokenów, poszerza ich margines wobec chmury tła i przywraca liczbę laserów narzędzia.

**Kontrole w każdym kroku.**
- Okluder tej samej wielkości na tle z dala od narzędzia.
- Neutralna szara łata, która oddziela usunięcie informacji od dystraktora podobnego do tkanki.
- Globalne zakłócenie o tej samej sile (rozmycie, jasność).
- Selekcja laserów i klastrów na danych rozłącznych z ewaluacją.
- Testy sparowane (Wilcoxon) po klatkach i modele mieszane po seedach.
- Stały podział według ośrodka lub filmu.

Realistyczne oczekiwanie to zysk rzędu kilku do kilkudziesięciu procent w utratach torów i przełączeniach ID, największy w epizodach okluzji. Tak wynika z literatury o filtrze Kalmana; obietnica dużych zysków zostanie zauważona przez recenzentów. Wielkości mikroskopu muszą pokazać wartość predykcyjną **ponad** zwykłymi odległościami i pewnością. Inaczej recenzent uzna lasery i pułapki za ozdobnik.

## Co nadaje się do publikacji

Najmocniejsza wersja artykułu to praca o **geometrii i dynamice reprezentacji tokenów w czasie treningu**, z niezmiennikami niezależnymi od przekroju i z modelami zerowymi. Metafory fizyczne zostają w dyskusji. Wszystkie twierdzenia poniżej mają pewność średnią lub niską, dopóki nie powstaną 2–3 seedy. Każde ma dopisek "one training run".

**T1. Chmura: relaksacja, stan stacjonarny, segregacja i dalsze upraszczanie bez starzenia** (pewność średnia; 1 seed, 12 klatek). Brzmienie musi mówić o uwięzieniu operacyjnym, a nie o szkle.

> **EN:** "Tracking the 600 patch-token states of a frame through training, we observe a fast relaxation (≈30 epochs, a ~100-fold drop in speed) followed by stationary, strongly subdiffusive motion at constant learning rate (MSD ∝ Δt^0.20 for background and ≈Δt^0.3 for instrument tokens), label-driven segregation of instrument from background tokens (95–99% outside the background's 95% radius from epoch 40), and a dimensional collapse that continues long after the IoU plateau (participation ratio 10.6 → 3.2). Neighbourhood persistence at a fixed 10-epoch lag decreases from 0.78 to 0.71, so restructuring slightly accelerates rather than slowing down as in aging (one training run)."

**T2. Pole MLP nie ma równowag przy stanach tokenów** (pewność średnia–wysoka dla przekroju P3; wszystkie 7200 tokenów, 3 epoki). To wynik metodyczny, ważny dla każdego, kto wizualizuje „pułapki" w przestrzeni latentnej.

> **EN:** "The local force field of the block-6 MLP has no equilibria near token states: in the visualized 3-D slice its magnitude at a token exceeds the token's cell half-size 12–17-fold, an unconstrained damped Newton search converges for only 15–16% of tokens, and the zeros it finds lie a median of 2–3.3 cloud radii away. We therefore classify the local velocity-gradient topology in each token's co-moving frame rather than 'traps'."

**T3. Topologia lokalna rozwarstwia się według gatunku tokenu** (pewność średnia; 1 seed, 256 + 360 koloidów). Liczby z P3 muszą iść razem z losowymi przekrojami i dywergencją. Zanik wirów tła nie może być twierdzeniem, bo jest artefaktem przekroju.

> **EN:** "From epoch ≈80 the local topology splits by token type: around instrument tokens contraction remains the most frequent class of the projected field (43–72% of tokens, a majority at 11 of 13 checkpoints), whereas around background tokens it becomes saddle-like (78–95% from epoch 120). The slice-independent divergence of the full 192-D Jacobian confirms the contrast (median −0.024 vs. −0.013 to −0.018 per dimension), but the class of an individual token is only weakly persistent between checkpoints for both types (Cohen's κ 0.15–0.33; one training run)."

**T4. Pełny jakobian odbiega od poziomu losowego tylko umiarkowanie; mechanizm leży w pojedynczych laserach** (pewność wysoka dla porównania z modelami zerowymi przy inicjalizacji; mechanizm do zmierzenia). Drugie zdanie to zdanie do Metod, uczciwie wyprowadzone i przypisane Im i in.

> **EN:** "Complex eigenvalues (87–93%) and a near-even split of eigenvalue signs (≈45% with positive real part) are the random-matrix baseline in 192 dimensions—our initialization gives 93% and 50%, a Ginibre matrix 94%—so we report only departures from it: training makes the MLP Jacobian slightly more symmetric (rotation-to-strain ratio ‖A‖/‖S‖ from 1.00 to 0.85–0.90) and contracting."
>
> **EN (Metody):** "Each hidden unit contributes a rank-one Jacobian term g_c v_c w̃_cᵀ, formally identical to a radiation-pressure force (an intensity acting along a fixed direction); its rotational part scales with |sin∠(v_c, w̃_c)| and its contribution to the divergence with cos∠(v_c, w̃_c), and the field is conservative if and only if the summed Jacobian is symmetric (Im et al., 2016)."

**T5. Lasery: kolimacja, potem stacjonarna dyfuzja rotacyjna przy normach bliskich równowadze AdamW** (pewność średnia; kierunki z wag, 1 seed). Nie wolno pisać o „stałej prędkości obrotu".

> **EN:** "The output directions of the 768 MLP units first collimate (from 154 to 31 effective directions by epoch 30) and then undergo stationary rotational diffusion: the mean angle between a unit's output directions grows from 22° at a 10-epoch lag to 66° at 140 epochs, and their cosine decays approximately as 0.98·exp(−Δt/150 epochs). Output norms approach the AdamW equilibrium scale (0.92 vs. ≈0.98 predicted), power inequality stays constant (Gini 0.16–0.17), and more heavily used units rotate more slowly (Spearman ρ ≈ −0.40 to −0.44; one training run)."

**T6. Hamulec robi kilkanaście laserów, a specjalizacja przesuwa się na tło** (pewność średnia; 1 seed). Bez twierdzeń przyczynowych o roli hamulca w detekcji, dopóki nie powstanie Krok 4.

> **EN:** "A small set of units implements a 'brake' that pushes background tokens towards the instrument cluster: from epoch 80 it reduces instrument–background separation by 9–14% (typically 11–13%), and half of it is produced by 10–19 of 768 units, while background-selective units grow to 25–33% of the layer and instrument-selective units fall from ≈60 to 15–24 (one training run)."

Deklaracja nowości powinna być wąska i zawierać „to our knowledge". Najbliższe prace to:
- [Fernando i Guitchounts 2026, preprint](https://doi.org/10.48550/arXiv.2605.14258);
- [Quemy 2026, preprint](https://arxiv.org/abs/2608.29706);
- [Mudarisov i in. 2026, preprint](https://arxiv.org/abs/2608.02071);
- [Tomihari i Karakida 2025, preprint](https://doi.org/10.48550/arXiv.2505.19458);
- [Aubry i in. 2024](https://arxiv.org/abs/2407.07810): ViT o d = 192, tylko SVD.

Żadna z nich nie śledzi tokenów obiektu wobec tła w czasie treningu ani nie analizuje samego MLP ViT z modelami zerowymi.

> **EN:** "To our knowledge, no prior work tracks the local Jacobian topology of a vision transformer's MLP around instrument and background tokens throughout training against random-matrix and initialization baselines."

**Czego nie pisać:**
- „szkło", „glass-like" ani „aging" bez D1–D4;
- „pułapka", „uwięzione tokeny" ani „wir Browna";
- „98% zespolonych wartości własnych" jako znalezisko;
- „zamrożone siodła tła wobec migoczących pułapek narzędzia" na podstawie surowej trwałości;
- „lasery obracają się ze stałą prędkością 2,2°/epokę";
- „aktualizacje są antyskorelowane";
- korelację ciemnych laserów z IoU jako wynik;
- jakiekolwiek twierdzenie o okluzji bez danych z Kroków 1–3.

## Ryzyka recenzenckie

| # | Zarzut | Dlaczego groźny | Odpowiedź lub kontrola |
|---|---|---|---|
| 1 | Jeden seed, 12 klatek, 616 koloidów | Fazy topologii, kolimacja i hamulec mogą zależeć od przebiegu | 2–3 dodatkowe przebiegi do epoki 200; bootstrap po klatkach; raportować statystyki, nie tożsamości |
| 2 | Artefakt przekroju | P3 jest stały z epoki 200 i zbudowany tylko z W2; wartości Ritza nie są wartościami własnymi J | Losowe przekroje (≥ 1000), przekroje Schura i SVD, η_inv, niezmienniki 192-D; wycofać zanik wirów tła |
| 3 | „Pułapka" nie jest równowagą, a blok nie jest przepływem | Zero leży poza chmurą; blok to jeden krok | Termin „local velocity-gradient topology"; kryterium mapy \|1+λ\| < 1 |
| 4 | Ogólność zespolonych wartości własnych | Są typowe dla losowych macierzy | Modele zerowe: inicjalizacja, Ginibre z tym samym rzutem LN, tasowanie par w–v; raportować przesunięcia |
| 5 | Kontrast trwałości klas | Wynika z rozkładu klas | κ zamiast surowej zgodności |
| 6 | Artefakt wyrównania w MSD | Procrustes usuwa dyfuzję obrotową, którą przewiduje teoria dryfu | MSD bez wyrównania, z wyrównaniem i względem klatki |
| 7 | Statyczna a dynamiczna niejednorodność | Cząstki są przypięte do różnych wejść, więc α₂ > 0 może być statyczne | Normowanie per token; trwałość rankingu ruchliwości |
| 8 | „Temperatura" to nie temperatura | Szum Adama jest anizotropowy, a detailed balance złamany ([Adhikari 2023](https://doi.org/10.48550/arXiv.2306.03521)) | Test gałęzi lr/batch (D5); termin „effective noise level" |
| 9 | Stosowalność Kossona | Kolumny fc2 nie są niezmiennicze względem skali; norma nadal rośnie | Odcinek krokowy: kąt na krok, autokorelacja aktualizacji; zgodność tylko co do rzędu wielkości |
| 10 | Korelacja ciemnych laserów z IoU | 15 punktów, 72 testy, wynik na suficie szumu | Rejestracja hipotezy; replikacja na seedach i większej liczbie klatek |
| 11 | Słownictwo fizyczne bez fizyki | Lasery, koloidy i pułapki mogą wyglądać na ozdobnik | Tabela analogii i metafor; metafory tylko w dyskusji; zostawić dokładną zgodność „natężenie × kierunek" |
| 12 | Nowość wobec preprintów 2025–2026 | Fernando, Quemy, Mudarisov i Tomihari analizują pokrewne obiekty | Cytować; zawęzić nowość do MLP ViT, tokenów obiektu wobec tła, czasu treningu i modeli zerowych |
| 13 | „So what?" | Brak związku z funkcją | Eksperyment okluzyjny (Kroki 1–4) i interwencje symetryzujące J |
| 14 | Okluzja: syntetyczna a prawdziwa, niejednoznaczne etykiety, kołowość, przesunięcie domeny | Rankingi z danych syntetycznych zgadzają się z prawdziwymi tylko w przybliżeniu ([Kassaw 2024](https://doi.org/10.48550/arXiv.2409.10775)); 18% niezgodności anotatorów | Zbiór prawdziwy z κ, klasa „niepewna", rozłączne podziały, stały podział ośrodków |
| 15 | Numeryka | Wartości własne macierzy nienormalnych są wrażliwe; LN daje dwie zerowe wartości własne | Uwarunkowanie, porównanie z różnicami skończonymi, margines hiperboliczności min\|λ(M)\|/‖M‖ |

## Następne pomiary (priorytet)

| Priorytet | Pomiar | Co rozstrzyga | Dane i koszt |
|---|---|---|---|
| **P1** | Modele zerowe dla J: tasowanie par (w̃_c, v_c), Ginibre z tym samym rzutem LN, inicjalizacja. Rozkład cos θ_c na laser, rozkład dywergencji na lasery (z ujemnym GELU) w czasie | Czy zbieżność i siodła to efekt uczenia; mechanizm T4 | Istniejące checkpointy; minuty GPU |
| **P1** | Kontrole przekroju: ≥ 1000 losowych przekrojów na koloid, przekroje Schura, η_inv, margines hiperboliczności; zgodność kierunku rozszerzania siodła tła z osią rozdziału | Co z P3 jest wierne; czy siodło tła to hamulec | Istniejące; `pola_pulapek.py` do rozszerzenia |
| **P1** | D1 i D2 na zapisanych położeniach; trwałość sąsiedztwa w jednostkach lokalnej odległości NN | Klatki harmoniczne czy szkliste; czy przyspieszona przebudowa to zagęszczenie przez spadek PR | Istniejące; minuty CPU |
| **P1** | 2–3 nowe seedy lub kontynuacje z perturbacją do epoki 200; powtórzenie `pola_pulapek.py`, laserów i korelacji ciemnych laserów według zarejestrowanego planu | Replikacja T1–T6 i hipotezy o IoU | Ok. 1,3 h na Edenie na kontynuację ([PW]) |
| **P1** | Pilot okluzji na zamrożonym modelu (Kroki 0–2 na bliźniakach syntetycznych) | Czy LLR, dywergencja i lasery narzędzia wyprzedzają pewność głowicy | 300–500 klatek; bez treningu |
| **P2** | Odcinek krokowy z epok 100 i 200 (zapis co 10 kroków przez ok. 1 epokę) | Kąt na krok i autokorelacja aktualizacji (T5); krótki reżim MSD i promień klatki | 2 krótkie przebiegi |
| **P2** | Gałęzie lr/batch (D5) z epok 100 i 200 | Czy chmura to NESS o „temperaturze" ∝ η/B | 8 × 15–20 epok |
| **P2** | D3 i D4 na siatce do epoki 400 ([K400]) | Starzenie i niejednorodność dynamiczna | Trening do 400 już trwa |
| **P2** | Zbiór ok. 200 prawdziwych klatek z widocznością (2 anotatorów, κ) i odtworzenie sekwencji w filtrze (Krok 3) | Wartość mikroskopu dla śledzenia | Anotacja i kod filtra |
| **P3** | Interwencje przyczynowe (Krok 4): lasery narzędzia, usunięcie hamulca, symetryzacja J (odjęcie A) | Funkcjonalna rola hamulca i rotacji | Tanie na zamrożonym modelu |
| **P3** | Ramiona treningowe (Krok 5); sonda spadku LR z epok 40, 100 i 200; przebieg z wd = 0 | Czy mikroskop nadaje się na cel treningu; czy upraszczanie napędza szum, czy WD | Kilka krótkich treningów |
| **P3** | Model szumu (dropout, augmentacja) i pomiar prądów prawdopodobieństwa | Jedyna droga do prawomocnego „wiru Browna" | Tylko jeśli potrzebny |

## Skróty ścieżek

| Skrót | Ścieżka bezwzględna |
|---|---|
| [KOL] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koloidy_wiazki_dynamika_0_200_20261021__01-48-10` (dane w `Wyniki\Dane`, kod w `Kod_eksperymentu`) |
| [APP] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\ogniskowa.py` |
| [K200] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_200epok_20261020__19-06-42` |
| [K400] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_400epok_20261020__23-50-57` |
| [P200] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\pary_przeciwne_do_200epok_20261020__23-21-30` |
| [PW] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\reports\Przeciwne wiązki w treningu ViT.md` |
| [NOT] | `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\research_notes\Fizyka chmury koloidów i okluzja\` |
