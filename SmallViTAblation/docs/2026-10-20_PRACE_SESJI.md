# Prace sesji 2026-10-20

Dokumentacja sesji z 2026-10-20. Zawiera: stan po ostatniej zapisanej sesji (2026-10-15), wyniki eksperymentów z 17–19.10 (bez zapisanej sesji), przegląd literatury, zasady zapisu eksperymentów oraz plan eksperymentu, który **nie został jeszcze wykonany**.

Konwencja: ścieżki są względem `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\` (skrót `SmallViTAblation\...`). W tej sesji nie uruchamiano żadnych eksperymentów ani poleceń ssh; wszystkie liczby pochodzą z przeczytanych plików, chyba że przy punkcie napisano inaczej. Dane o dysku Eden są w osobnym pliku: `SmallViTAblation\docs\2026-10-20_EDEN_MIEJSCE_NA_DYSKU.md`.

---

## 1. Stan po ostatniej zapisanej sesji

Źródło: `.sessions\Session_2026-10-15_172242_Pulapka_Cel_Koloidu_Start_Aplikacji\SESSION_SUMMARY.md` (sesja 2026-10-15, 10:55–17:22, status: Paused).

Najważniejsze wyniki:

1. Start aplikacji `Edytor latentu.exe` skrócony z 14 s do 0,2 s: karty budowane leniwie, torch importowany dopiero w funkcjach (commit `e02afa2a`).
2. Dodano F11 w całej aplikacji, pasek okienek z przyciskami ⛶/❐/✕ i zapis układu okienek w `ustawienia.json` (commity `e02afa2a`, `53ae1f76`).
3. Opcja „płynne przejście uwaga ↔ MLP” obok osobnych podkroków (commit `7177373b`) oraz pole w kuli komórki w podkrokach (`pole_podkrokow.py`, commit `c6f324e5`).
4. Osobny commit „reflektory stoją”: usunięto przesuwanie reflektorów między blokami; zbiór miejsc i kolorów jest identyczny w każdym bloku.
5. Pułapka wiązek dla tokenu 226 (806 checkpointów × 12 podkroków): pole liniowe w komórce, zero pola nigdy w komórce; siodło 75 %, rozpychanie we wszystkich osiach 24 %, studnia 1,2 %. MLP prawie zawsze rozprasza (blok 1 najmocniej, dywergencja +2,2). Iloczyn 12 podkroków rozpycha objętość ok. 3× (18× → 6,6×).
6. Pułapka uczenia: proces Ornsteina–Uhlenbecka wokół dryfującego środka; korelacja przyrostów −0,415 (OU −0,42); czas relaksacji τ 8–12 → ok. 30–50 kroków; dojście potęgowe t^−0,43.
7. Cel w uczeniu: błąd tokenu spadł z 4,7 do 0,048; ruch w uczeniu wzdłuż −∇L słabo zgodny (cos +0,1–0,18).
8. Cel w przestrzeni wyjścia: ramka ltrb osiągnięta po ok. 500 krokach; logit „środek” rośnie ok. 0,70·ln t. W latencie p = 1 nieosiągalne samym ruchem tokenu (sufit logitu +6,89 na końcu).
9. Hipoteza „koloid wędruje po dnie doliny” obalona: ruch i drgania silniejsze w kierunkach czytanych przez głowę.
10. Literatura: brak pracy łączącej pole wiązek MLP, koloid w tokenie i cel w uczeniu (ryzyko: Halperin 2606.29679). Kolekcja Scite (27 DOI) i zbiór PDF w `SmallViTAblation\PDF` (pobrano 8 z 18).
11. Problemy: 10 z 18 PDF nieudanych (paywall); strażnik worktree blokował złożone polecenia git.

Otwarte kroki z pliku (Next Steps):

- Pomiar pułapki uczenia dla wszystkich 600 tokenów i innego biegu (DeiT na Edenie), w tym bieg bez obcinania gradientu.
- Gęstsze checkpointy w późnym uczeniu, aby rozdzielić τ < 100 kroków.
- Ustalenie, co zatrzymuje logit na ok. 2,9 przy suficie 6,9 (rozkład siły na inne zdjęcia, weight decay, obcinanie).
- Ręczne pobranie 10 nieudanych PDF z `SmallViTAblation\PDF\nieudane\nieudane_linki.html`.
- Ewentualny przycisk pomiaru celu koloidu w aplikacji.

---

## 2. Eksperymenty 17–19.10 (bez zapisanej sesji)

### 2.1 Kompensacja, pruning i dynamika uczenia (18.10, raport 19.10)

Źródło: `SmallViTAblation\outputs\kompensacja_pruning_dynamics_20261018\RAPORT_KONCOWY_PL.md`

- Zakres: 24 istniejące modele (seedy 3047 + 1000·i), 12 checkpointów (0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640), ostatni MLP (768 neuronów). Ablacja zeruje aktywacje GELU dla wszystkich 600 tokenów. Ranking na 500 klatkach VAL (25 filmów), pomiar na 500 klatkach TEST (472 z GT, 28 bez GT).
- Główny wynik: wyłączenie całego zbioru par functional obniżyło score w **24/24 seedach**. ΔIoU = −0,0000149 (95 % CI [−0,000278; +0,000264]), Δscore = −0,01523 (CI [−0,01772; −0,01293]).
- Rodziny geometric (cos W2 < −0,5) i fast: ΔIoU ok. −0,0008 (CI obejmują zero), Δscore −0,0755 i −0,0729 (CI poniżej zera), w 24/24 seedach.
- Kontrola: wyłączenie zbioru functional obniżyło score o 0,01019 bardziej niż kontrolne wyłączenie tej samej liczby neuronów (CI różnicy [−0,01184; −0,00858]).
- Score to sigmoid logitu, nie skalibrowane prawdopodobieństwo. Raport nie mierzy AP ani kalibracji.
- Liczba neuronów z przeciwnikiem (cos W2 < −0,5): 0 w krokach 0 i 300 → 92,17 (krok 10000) → 210,50 (krok 16000) → 424,04 (koniec treningu).
- Największy przyrost IoU i score nastąpił w przedziale 0–300 we wszystkich 24 seedach. Największy przyrost liczby przeciwników: przedział 10000–16000 (20/24 seedów). Przeciwstawność narasta więc głównie po głównym skoku jakości.
- Korelacje zmian IoU z przyrostem przeciwstawności po kontroli seeda i przedziału: wszystkie zaplanowane warianty z CI obejmującym zero (dla lagu 0, różnice: r = 0,0704, CI [−0,0400; +0,1613]).
- Mechanika: pierwsza para functional ma średni cos rzeczywistych zmian krawędzi ok. −0,753 (poniżej −0,5 w 24/24 seedach), a wyłączenie obu neuronów razem obniża score w 24/24. Oba neurony wspierają więc logit, choć kompensują krawędzie ramki.
- Stałe 16 par (32 neurony): ΔIoU +0,0000704, Δscore −0,0042926 (24/24); kontrola: +0,0000446 i −0,0018380. Koaktywacja na panelu 50 obrazów spada z ok. 86,17 % do 17,73 %.
- Ograniczenia z raportu: badany jest ostatni MLP jednej rodziny modeli; zbiór TEST był już wcześniej oglądany; 24 seedy wzmacniają powtarzalność tylko w tym ustawieniu. Nie wykazano, że przeciwstawność jest uniwersalnym mechanizmem ani że usuwanie kompensujących się neuronów poprawia detektor.
- Weryfikacja pobrania: archiwum 8 781 762 010 B, wynik PASS, 23 615 plików manifestu, 24 seedy, 12 checkpointów, 600 shardów końcowych, 7200 dynamicznych, 288 odwołań do wag.

### 2.2 Względny wkład pary reflektorów w MLP (19.10)

Źródło: `SmallViTAblation\outputs\kompensacja_wklad_wzgledny_20261019\WNIOSKI_PL.md`

- Pomiar lokalny, bez treningu, na 24 końcowych checkpointach i panelu 50 obrazów z 25 filmów: 1200 odtworzeń modelu, nie 1200 niezależnych obrazów.
- Pierwsza para functional przy clean_winner: wypadkowa dwóch wiązek to **2,043 %** normy pełnego przesunięcia ostatniego MLP z biasem (średnie modeli 1,380–3,098 %).
- Rozkład 1200 obserwacji: mediana 1,899 %, P5–P95 0,592–3,557 %, maksimum 20,521 %. Ponad 5 % w 16 obserwacjach, ponad 10 % w 5.
- Wyłączenie pary zmienia kierunek pełnego przesunięcia MLP średnio o 1,096°; długość spada do 99,384 % pierwotnej (ok. 0,616 %).
- Dwa różne mianowniki (nie wolno ich mieszać): 2,132 % względem sumy wektorowej 768 wiązek bez biasu, 0,309 % względem sumy 768 indywidualnych norm.
- Dla wszystkich 1327 par functional średni udział pojedynczej pary: 1,682 % (pary się nakładają, więc nie sumuje się ich do całej grupy). Dla pierwszych 16 par: 1,935 %.
- Znoszenie w latencie: stosunek normy wypadkowej do sumy norm dwóch wkładów dla pierwszej pary wyniósł 0,706.
- Koaktywacja, cos poniżej −0,5 i siły w stosunku co najwyżej 1,25 wystąpiły łącznie tylko w 17 obserwacjach (3 modele, 14 kombinacji film × model). To kategoria post hoc o małym wsparciu.

### 2.3 Wiązki sfery A–D: audyt i pilot (19.10)

Źródło: `SmallViTAblation\outputs\wiazki_sfera_abcd_20261019\RAPORT_PL.md`

- Stan: audyt geometrii 24 modeli i pilot dwóch obrazów zakończone i zweryfikowane. **Pełnej kampanii 24 × 500 nie zgłoszono** z powodu zmierzonego kosztu zapisu.
- Próg 150° jest operacjonalizacją autora. Kierunki promieni: d = normalize(P3V), P3 z SVD wycentrowanych kolumn W2 ostatniego bloku.
- Katalog przy 150°: 479 248 par łącznie w 24 modelach (19 435–20 753 na model); każdy z 768 reflektorów ma partnera. Warianty C = clean, D_A = A, D_B = B są jawnymi aliasami.
- Z 479 248 par tylko 10 828 ma cos wag poniżej −0,5 w pełnej przestrzeni 192D; 78 327 ma cos nieujemny. Przeciwieństwo w projekcji 3D nie oznacza przeciwieństwa w 192D.
- Pilot (seed 3047, krok 65640, 600 tokenów, 19 575 par, 347 rozłącznych par B), dwa obrazy z tego samego filmu, jeden z GT i jeden bez:
  - w klatce z GT 62,37 % ważnych par-tokenów ma przeciwieństwo ≥150° w 3D (w pustej 67,35 %); w pełnym 192D żadna para tego nie osiąga;
  - po zgaszeniu 347 par score clean winner w klatce z GT zmienił się z 0,982918 na 0,973197, a IoU z 0,8106723 na 0,8106718; w pustej klatce score wzrósł z 0,432046 do 0,451213 (IoU bez GT pozostaje null).
- Koszt na Edenie: H100 PCIe, 1 GPU, job pilota ok. 2 min 1 s (wall), szczyt pamięci GPU ok. 205 MB, MaxRSS ok. 1,3 GB. Zapis ok. 495 MB na obraz. Odczyt, zapis i metryki: 60,66 s (z GT), 47,84 s (bez GT).
- Ekstrapolacja zapisu surowego formatu: 24 × 500 TEST ok. 6,057 TB (5641 GiB), 203,8 h seryjnie, 25,5 h przy 8 równoległych GPU. 24 × 50 ok. 605,6 GB (564 GiB), 20,6 h seryjnie, 2,58 h przy 8 GPU.
- Kontrole: maksymalny błąd raw wobec algebraicznego odczytu 5,843e−6 (tolerancja 2e−4); testy jednostkowe 11/11 PASS; 48 rzeczywistych hooków B bez niezgodności zwycięzcy.
- Joby: smoke 1887984 → pilot 1887985 → reduce 1887986 (COMPLETED, exit 0:0), dodatkowy 1887988 (13 s). Eksport 1 015 797 760 B, 1243 pliki zweryfikowane. Pierwsza próba zgłoszenia v1 nie utworzyła jobów z powodu braku konta projektu; v2 używa `transformers_vsc`.
- Ograniczenia z raportu: jeden model, dwa obrazy z jednego filmu. Wyniki są opisowe i eksploracyjne.

### 2.4 Audyt zakresu lokalnego gaszenia wiązek (19.10)

Źródło: `SmallViTAblation\outputs\opposition_sphere_audit_20261019\AUDYT_ZAKRESU_WIAZEK_PL.md`

- Odpowiedź na pytanie o zakres: **tak**, w kampanii functional 24 seedów wykonano lokalne A/B/AB (zerowanie wiązki A, B i obu). Działo się to jednak na panelu diagnostycznym 50 obrazów na model i checkpoint. Główne testy P1/P2 używały globalnego wyłączenia reflektorów na 500 obrazach TEST.
- Nie wykonano pełnej koniunkcji wymagań: pary nie były dobierane wg aktualnych podpisanych kierunków wewnątrz sfery przy danej cząsteczce ani wg kąta 3D. Rodzina functional wybrana na VAL przez JVP bounding boxa. Rodzina weight_opposed wymagała W2 cos < −0,5 w checkpoincie selekcji.
- Audyt próby: 96 rekordów (24 modele × 4 checkpointy, pierwsza klatka diagnostyczna pierwszego filmu); 45 020 lokalnych warunków `method=zero, alpha=1`, w tym 6144 pary functional_cancel i 2036 par weight_opposed.
- Zgodność local/global: maksymalna różnica pięciu wyjść stałego tokenu 5,68·10⁻¹⁴. Norma perturbacji całej równa normie jej jedynego tokenu (różnica 8,88·10⁻¹⁶).
- Zwycięzca nie jest ten sam: w 3188 wynikach A/B/AB lokalne i globalne wyłączenie wybrały inny token. Interpretacji lokalnej nie wolno więc przenosić na score, bounding box i IoU ponownie wybranego zwycięzcy.
- Pilot 3 seedów (47, 1047, 2047): 117, 116 i 116 lokalnych warunków; przykłady podpisanego latent_cos od −0,70 do −0,54.
- Historycznie ok. 53 % reflektorów miało przeciwnika wg progu pełnego W2. To opis geometrii wag, nie dowód, że wszystkie takie wiązki były jednocześnie aktywne.
- Nie uruchomiono nowego eksperymentu.

### 2.5 Compact trial wiązek sfery (19.10): status

Źródło: `SmallViTAblation\experiments\wiazki_sfera_compact_trial_20261019\PLAN.md`

- Cel: zwarta reprezentacja wszystkich 19 575 par × 600 tokenów × 347 rund dla seeda 3047, kroku 65640, obrazu `TR_0008_S1_P03/1` (z GT). Bez nowego forwardu ViT i bez GPU przy konwersji. Checkpointów nie usuwano.
- Zapis w planie (ledger): CPU-only job na Edenie 1887991, COMPLETED 0:0; dane 31 755 318 B, archiwum transferu 31 979 520 B; testy: 176 przypadków replay na Edenie i 128 dodatkowych lokalnie, wszystkie PASS; manifest 18 plików, 617 plików źródłowych, SHA checkpointu zgodne.
- **Uwaga: raport wskazany w planie nie istnieje lokalnie.** Plan odwołuje się do `outputs\wiazki_sfera_compact_trial_20261019\RAPORT_PL.md` i `EDEN_STATUS.json`. W `SmallViTAblation\outputs\` nie ma katalogu `wiazki_sfera_compact*`. Przed cytowaniem wyniku trzeba go pobrać z Edenu lub zweryfikować.

---

## 3. Przegląd literatury: przeciwstawne neurony w trakcie uczenia (Undermind + Scite)

Źródło: `SmallViTAblation\plan_na_przyszle_eksperymenty\2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md`

Wyszukiwania: workspace Undermind „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”, id `40294fa6-c9a3-4469-9a61-d3eddeaee893` (id podane przez użytkownika; plik podaje nazwę workspace). Trzy nowe deep searche z 2026-10-20:

- „Emergence of opposing antagonistic neurons during neural network training tracked across checkpoints” (69 prac),
- „Measuring synergy redundancy and compensation between neurons through training with interventions” (132 prace),
- „Prior art check for specific measured findings on opposing MLP neuron pairs in a ViT detector across training” (62 prace; cel zawierał nasze liczby z kampanii pruning, wkład względny i sfera).

Pełne teksty czytane dla: Mar25l, Che23an, Ahm26c, Ran23c, Hut25, Pat25d, Gon26h, Sam26b, Zhu26l, Urb26b, Saf26. Pozostałe prace znane z abstraktu i metadanych.

### Odpowiedź wprost

Nikt nie zmierzył dokładnie tego, co my: przeciwstawności neuronów ostatniego MLP śledzonej przez checkpointy i seedy razem z interwencjami A/B/AB oraz rozdzieleniem IoU od score. Są jednak bliskie precedensy:

1. **Mar25l** — „Flat Channels to Infinity in Neural Loss Landscapes” (2025), DOI 10.48550/arXiv.2506.14951. Gradient prowadzi pary neuronów, których wagi wejściowe się zbiegają (cos → 1), a wyjściowe rozbiegają z przeciwnymi znakami. Duże wkłady prawie się znoszą, a suma działa jak gated linear unit. Zjawisko późnego uczenia, z bardzo wolnym spadkiem straty. Badano 4420 sieci, 50 inicjalizacji na konfigurację; kanały w 20–64 % biegów. Kryteria: norma parametrów > 1e3, cos-odległość wag wejściowych < 1e-3, > 90 % aktualizacji w podprzestrzeni kanału.
   - **Kwo05b** — „Anti-pairing in learning of a neural network with redundant hidden units” (J. Phys. A, 2005), DOI 10.1088/0305-4470/38/25/003. Abstrakt niedostępny w Undermind.
2. **Pat25d** — „LLM Layers Immediately Correct Each Other” (2026), DOI 10.52202/085713-4643. Cos wkładów sąsiednich warstw do tego samego tokenu ok. −0,2 (do −0,4). W OLMo wzorzec pojawia się ok. 21 % treningu i narasta. Brak zestawienia ze stratą.
3. **Zhu26l** — push-pull w sieciach spikujących (Biol. Cybern., 2026), DOI 10.1007/s00422-025-01030-4. Stosunek hamowania across/within rośnie z 1,06 do 2,0. Brak analizy, czy wzrost poprzedza jakość, czy po niej następuje.
4. **Che23an** — „Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition” (2023), DOI 10.48550/arXiv.2310.06301. Plateau straty i nagłe spadki przy przejściach geometrii; antypodalnych par osobno nie liczono.

Uwaga do zestawu: plik wymienia cztery kategorie precedensów: (1) Mar25l i Kwo05b, (2) Pat25d, (3) Zhu26l, (4) Che23an. Zestaw z polecenia (Mar25l, Kwo05b, Pat25d, Zhu26l) pokrywa te same prace bez Che23an.

Wniosek z pliku: nasz wynik „przeciwstawność wag rośnie głównie po głównym skoku jakości i nie koreluje z przyrostami IoU po kontroli trendu” nie ma precedensu (ocena Undermind: prawdopodobnie nowe). Mar25l daje gotową hipotezę mechanistyczną: późne pary to kanały/GLU powstające z symetrii gradientu, a nie „przeszkadzające sobie lasery”.

### Ocena sześciu naszych obserwacji

Źródło: ta sama tabela w pliku literatury.

| Obserwacja | Ocena | Najbliższe prace |
|---|---|---|
| 1. Wzrost liczby neuronów z antyrównoległym partnerem 0 → 424/768, wymiana partnerów pod koniec | częściowo antycypowane | Mar25l, Kwo05b; kondensacja Xu25ad, Zho21h (raczej grupy podobne niż przeciwne) |
| 2. Opozycja po głównym zysku jakości, brak korelacji po kontroli trendu | prawdopodobnie nowe | brak; metodycznie Ran23c, Saf26 (korelacja zmian metryk) |
| 3. Geometria wag ≠ funkcjonalne znoszenie | częściowo antycypowane | Lan22 (upstream vs downstream), Fak24, Ahm26c (z_r przewiduje wielkość, nie znak) |
| 4. Przeciwne skutki na krawędzie, zgodne wspieranie logitu | antycypowane na poziomie zadania, nowe na poziomie par | Wu19d, Zha23bf, Hut25 (DETR: MHCA psuje mgIoU, F1 stabilne) |
| 5. Usuwanie par nie poprawia detektora | częściowo antycypowane | Cas19e (frivolous units), Yu22f (pruning wag, które się znoszą), Mor18c |
| 6. Opozycja rośnie, a koaktywacja spada (86 % → 18 %) | częściowo antycypowane | Mar25l, Pat25d, Mcg23, Rus24b, Ahm26c |

### Osiem pomysłów na eksperymenty

Źródło: sekcja „Co z tych prac warto przenieść do naszych eksperymentów” w pliku literatury.

1. Test kanału Mar25l na istniejących 288 checkpointach (bez treningu). Szczegóły w sekcji 5 niniejszego dokumentu.
2. Dawka zamiast zera (Ahm26c): skalowanie z_A współczynnikiem λ ∈ [−1, 1], λ = 0 jako średnia, nie zero.
3. Interwencja w czasie uczenia (Ran23c): regularyzator karzący antyrównoległe kolumny W2, włączany od kroku 0, 300 lub 10000. Wymaga nowych biegów na Edenie.
4. Zastąpienie przez rówieśników (Saf26): rekonstrukcja aktywacji wyłączonego neuronu z top-k neuronów tej samej warstwy.
5. Interakcja i zapasowe neurony (Gon26h, Fak21, Kei04c): I_uv = δz_uv − δz_u − δz_v oraz skan po wyłączeniu zbioru S.
6. Kontrole zerowe: liczba antyrównoległych kolumn w sieci losowej o tym samym spektrum norm lub po losowej rotacji; wpływ weight decay; porównanie przeciwstawności bloków jak w Pat25d.
7. Czy przedział 10000–16000 to przejście fazowe? Estymacja LLC (Che23an, biblioteka devinterp) na checkpointach wokół tego przedziału.
8. Ablacje ostrożne (Hut25, Poc24, Mil24c, Li24bi): raportować mgIoU/IoU i score osobno; porównać zero-, mean- i resample-ablation.

### Zasady statystyczne (do protokołu)

Źródło: ta sama sekcja „Statystyka (do protokołu)”.

- Bootstrap na poziomie seeda z zachowaniem zależności checkpointów w seedzie.
- Selekcja na VAL, ocena na nowym, nieoglądanym TEST (obecny TEST był oglądany).
- Korelacje geometrii z jakością: pierwsze różnice, zaplanowane lagi, korekta na wielokrotność; nie wybierać najlepszego lagu.

### Kluczowe prace (DOI)

Źródło: sekcja „Prace kluczowe (DOI)” w pliku literatury.

| Praca | DOI / arXiv |
|---|---|
| Mar25l — Flat Channels to Infinity in Neural Loss Landscapes | 10.48550/arXiv.2506.14951 |
| Kwo05b — Anti-pairing in learning of a neural network with redundant hidden units | 10.1088/0305-4470/38/25/003 |
| Pat25d — LLM Layers Immediately Correct Each Other | 10.52202/085713-4643 |
| Zhu26l — Task success in trained SNN … cross-stimulus-modulated inhibition | 10.1007/s00422-025-01030-4 |
| Che23an — Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition | 10.48550/arXiv.2310.06301 |
| Ahm26c — Every Ablation Is a Dose: Counterweights and the Semblance of Self-Repair | arXiv 2610.02173 |
| Ran23c — On the special role of class-selective neurons in early training | 10.48550/arXiv.2305.17409 |
| Saf26 — Task Relevance Is Not Local Replaceability | 10.48550/arXiv.2605.07086 |
| Gon26h — Conditional Co-Ablation | 10.48550/arXiv.2607.01940 |
| Hut25 — Detection Transformers Under the Knife | 10.48550/arXiv.2507.21723 |
| Lan22 — Clustering units in neural networks: upstream vs downstream information | 10.48550/arXiv.2203.11815 |
| Yu22f — The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another | 10.48550/arXiv.2203.04466 |
| Cas19e — Frivolous Units | 10.1609/aaai.v35i8.16853 |
| Mor18c — On the importance of single directions for generalization | arXiv 1803.06959 |
| Fak21 — Systematic perturbation of an ANN (MSA) | 10.1371/journal.pcbi.1010250 |
| Urb26b — A Brain-like Synergistic Core in LLMs | 10.48550/arXiv.2601.06851 |
| Sam26b — Hidden not Deleted: How Networks Suppress Entangled Features | arXiv 2609.27593 |
| Min23b — Early Neuron Alignment in Two-layer ReLU Networks | 10.48550/arXiv.2307.12851 |
| Elh22 — Toy Models of Superposition | 10.48550/arXiv.2209.10652 |
| Mcg23 / Rus24b — Hydra effect / self-repair | 10.48550/arXiv.2307.15771, 10.48550/arXiv.2402.15390 |

### Wniosek metodyczny

Źródło: wniosek sformułowany przez użytkownika w tej sesji; nie wynika z przeczytanych plików.

Undermind lepiej przyjmuje surowe wyniki (cel do 30 000 znaków, agenci sami układają zapytania). Scite służy do weryfikacji cytowań i fragmentów pełnych tekstów.

---

## 4. Zasady eksperymentów od 2026-10-20

Źródła: `SmallViTAblation\AGENTS.md`, `SmallViTAblation\experiments\AGENTS.md`, `SmallViTAblation\CLAUDE.md`, `SmallViTAblation\experiments\experiments_index.md`, `SmallViTAblation\experiments\CLAUDE.md`.

Zasady wprowadzone poleceniem użytkownika z 2026-10-20 (obowiązują też Claude Code):

1. **Folder:** `experiments/<nazwa>_YYYYMMDD__HH-MM-SS/` (czas Europe/Warsaw). Jeśli użytkownik poda dokładną nazwę, używa się jej.
2. **Podfoldery (dokładna pisownia):**
   - `Kod_eksperymentu/` — kod, skrypty uruchomieniowe i konfiguracja;
   - `Opis/` — obowiązkowy opis: cel, hipotezy, model i checkpoint, dane, procedura, metryki, ograniczenia, sposób odtworzenia (np. `OPIS_EKSPERYMENTU.txt`);
   - `Wyniki/` — wszystkie wyniki.
3. **Nic luzem:** kodu ani opisu nie zapisuje się bezpośrednio w katalogu eksperymentu.
4. **Podfoldery `Wyniki/`:**
   - `Code_Interpretujacy_Wyniki/` — kod analizy, interpretacji i wizualizacji;
   - `Dane/` — pomiary, tabele, logi, manifesty, surowe artefakty (modele, checkpointy, obrazy i latenty można grupować tutaj);
   - `Wykresy/`, `Raport/`, `Krótki_Wniosek/`.
5. **Indeks:** nowy eksperyment od razu trafia do `experiments/experiments_index.md` z pełną nazwą, odnośnikiem i krótkim celem. Kontynuacja aktualizuje istniejący wpis, bez duplikatu. Wpisów innych agentów nie usuwa się.
6. **Wykresy, raport i krótki wniosek:** przed ich przygotowaniem trzeba zapytać użytkownika, które chce (tak/nie, osobno lub łącznie). Wyraźne zamówienie lub zgoda w tej samej sesji oznacza zatwierdzenie i nie wymaga ponownego pytania. Brak odpowiedzi nie blokuje obliczeń ani zapisu danych. Foldery mogą zostać puste.
7. **Eden:** ten sam układ folderów. Historycznych eksperymentów nie przenosi się i nie uruchamia ponownie tylko w celu porządkowania.
8. **Baza lokalna:** `W:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation` (wskazana 2026-10-20 zamiast niedostępnego F:). Nie używać katalogu projektu Grant jako miejsca zapisu.
9. **Zastąpienie starego podziału:** zasady zastępują podział `experiments/` i `outputs/` z 2026-10-05. Wyjątek: aplikacja GUI w `app/` nadal zapisuje do `outputs/latent_editor_sd_20261006/`.

Przykład podany przez użytkownika: `experiments/vit_wiazkiprzeciwstawne_20261020__18-10-00/`.

Stan indeksu na dziś: `experiments_index.md` ma jeden wpis, `vit_wiazkiprzeciwstawne_20261020__18-10-00` (cel: zmiana siły pojedynczych wiązek MLP w sześciu blokach ViT podczas treningu oraz atlas wykresów). Katalog istnieje (sprawdzone 2026-10-20 ok. 18:30: `Kod_eksperymentu/`, `Opis/`, `Wyniki/Dane/` z pomiarami; plik `Wyniki/PODSUMOWANIE.png` leży poza `Wyniki/Wykresy/`). Po 18:38 dopisano drugi wpis: `kanal_mar25l_checkpointy_20261020__18-38-50`. Indeks podaje też, że wcześniejsze eksperymenty nie zostały jeszcze zinwentaryzowane.

---

## 5. Eksperyment w przygotowaniu: test kanału Mar25l na checkpointach (Eden)

**Status: w przygotowaniu, NIE wykonano.** Katalog: `experiments/kanal_mar25l_checkpointy_20261020__18-38-50/` (wpis w indeksie dodany); kod w budowie. Brak uruchomień na Edenie w chwili zapisu tego dokumentu.

Źródła planu: polecenie użytkownika z 2026-10-20, pomysł nr 1 w `SmallViTAblation\plan_na_przyszle_eksperymenty\2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md`. Dane wejściowe sprawdzone w kodzie (podane przy każdym punkcie).

### Cel i hipotezy

- Sprawdzić, czy pary neuronów ostatniego MLP o antyrównoległych kolumnach wyjściowych W2 (cos < −0,5) są „kanałami do nieskończoności” z pracy Mar25l (arXiv 2506.14951, „Flat Channels to Infinity in Neural Loss Landscapes”). Autorzy zweryfikowani w Undermind: Flavio Martinelli, …, Johanni Brea (2025).
- Kryteria kanału z literatury: wagi wejściowe pary zbiegają się (cos → 1), wagi wyjściowe rozbiegają z przeciwnym znakiem, duże wkłady prawie się znoszą. Suma pary działa jak gated linear unit:

  s·[g(z_i) + g(z_j)] + m·[g(z_i) − g(z_j)] ≈ 2s·g(z̄) + m·(z_i − z_j)·g′(z̄),

  gdzie s = (a_i + a_j)/2, m = (a_i − a_j)/2, z̄ = (z_i + z_j)/2. Przybliżenie ma błąd rzędu s·Δ² i m·Δ³ (Δ = z_i − z_j). Błąd przybliżenia zaplanowano jako pomiar.
- **H1 (kanał)** kontra **H0 (antyrównoległość wyjść bez zbieżności wejść)**.
- Wcześniejsze dane sugerują raczej H0: koaktywacja par geometrycznych na panelu 50 wynosi 22,57 % (`kompensacja_pruning_dynamics_20261018\RAPORT_KONCOWY_PL.md`), a znoszenie dla pierwszej pary functional wynosi 0,706 (`kompensacja_wklad_wzgledny_20261019\WNIOSKI_PL.md`). Rozstrzygnąć ma to pomiar.

### Dane (sprawdzone w kodzie)

- **Seedy:** 24, 3047 + 1000·i dla i = 0…23 (`experiments\kompensacja_funkcjonalna_dgx2_20261017\prepare_packet.py:66`, `PLAN.md:13`).
- **Checkpointy:** 12, kroki 0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640 (`train_replicate.py:30`). Razem 288 plików wag.
- **Nazwa pliku wagi:** `replications/s<seed>/weights/k<step>.pt` z krokiem zapisanym na 5 cyfrach, czyli np. `k00300.pt`, `k65640.pt` (`run_seed.py:42`, `export_results.py:106`). Plan pisze `k<step>.pt`; w skryptach przyjęto `k{step:05d}.pt`.
- **Lokalizacja wag na Edenie** (podana przez użytkownika, nie zweryfikowana w repozytorium): `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017/replications/`. Według poprzedniego punktu ścieżka kończy się na `replications/s<seed>/weights/`.
- **Panel TEST:** 50 klatek, pozycje 5 i 15 w każdym z 25 filmów (`prepare_packet.py:59`, `kompensacja_pruning_dynamics_20261018\stage_packet.py:73`, gdzie `expected_images=500`, `expected_films=25`). Panel VAL analogicznie, według planu użytkownika.
- **Katalogi par:** functional, geometric i fast z `kompensacja_pruning_dynamics_20261018` (definicje i liczby w `RAPORT_KONCOWY_PL.md`: functional 18–125 par na seed, geometric 527–961, fast 488–877).

### Model i trening (sprawdzone w kodzie)

- **Architektura** (`experiments\kompensacja_funkcjonalna_dgx2_20261017\model.py`): dim 192, 6 bloków `timm.Block` (3 głowy, MLP 4.0 = 768 ukrytych, qkv_bias), końcowy LayerNorm i gęsta głowa (Linear–GELU–Linear do 5 wyjść). Plan użytkownika podaje tylko „dim 192, 6 bloków timm, MLP 768 GELU, pre-LN”. W pliku nie ma wprost zapisu pre-LN; struktura `timm.Block` jest pre-norm według biblioteki timm (nie sprawdzono w kodzie timm). Pominięto w planie:
  - **Łodyga konwolucyjna:** 4 konwolucje 3×3 z krokiem 2 (3 z BN+GELU), krok sieci 16 px. Dla obrazu 480×320 siatka 30×20 = 600 tokenów.
  - **Pozycje:** stałe sin-cos 2D.
  - **Głowa:** gęsta, na każdy token logit środka i odległości (l, t, r, b).
- **Latent do pomiaru:** aktywacje po GELU w `bloki[-1].mlp.act`, 768 na token (`model.py:53`).
- **Optymalizacja** (`train_replicate.py:29–31`, `train_replicate.py:65`, `run_seed.py:109`): batch 32, AdamW, weight decay 0,05 tylko na parametrach o ≥2 wymiarach (norma i bias bez decay), clip normy gradientu 1,0, autocast bf16, 65 640 kroków, 35 000 wierszy TRAIN. Trening nie czyta VAL ani TEST.
- **Rozgrzewka lr, której plan nie podaje:** `lr = 5e-4 · min(1, (step+1)/300)` (`train_replicate.py:98`). Lr jest liniowo rozgrzewany przez 300 kroków, a potem stały.
- **Ważne dla kryteriów:** weight decay ogranicza rozbieganie norm. Literalne kryteria Mar25l (norma > 1e3, cos-odległość wag wejściowych < 1e-3, > 90 % aktualizacji w podprzestrzeni kanału) prawdopodobnie nie będą spełnione. Test musi być stopniowany, jak zaznacza plan.

### Pomiary (plan)

- Cos wag wejściowych: surowy (z biasem), efektywny po złożeniu afinicznego LayerNorm i wycentrowaniu, oraz korelacja preaktywacji na tokenach.
- Normy ‖a_i‖, ‖m‖, ‖s‖, ‖Δw‖ i iloczyn ‖m‖·‖Δw‖.
- Frakcja aktualizacji między checkpointami w kierunkach kanału.
- Stosunek znoszenia ‖u_i + u_j‖ / (‖u_i‖ + ‖u_j‖), gdzie u_j = z_j·v_j.
- Błąd przybliżenia GLU i wpływ zastąpienia pary jej przybliżeniem na wyjście głowy (ostatni blok).
- Rozkład łączny (cos wejścia, cos wyjścia) wszystkich 294 528 par w każdym bloku.
- Zbiory kontrolne: pary losowe oraz „bliźniaki” (najwyższy cos wejścia).

### Wykonanie (plan)

- Slurm na Edenie, partycja `hopper` (H100), konto `transformers_vsc`.
- Kolejność: smoke, potem tablica 24 seedów, potem redukcja na CPU.
- Wyniki w katalogu grupy `/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation/experiments/<nazwa>/`, nie w home, który jest prawie pełny (dane w `2026-10-20_EDEN_MIEJSCE_NA_DYSKU.md`).
- Środowisko Python kampanii (podane przez użytkownika, nie zweryfikowane w repozytorium): `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python` (torch 2.10.0+cu126, timm 1.0.24, numpy 2.2.6).

### Ryzyka do sprawdzenia przed uruchomieniem

1. **Limit GPU na koncie.** Plan zakłada „max 8 GPU”. W `SmallViTAblation\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md:3679` zapisano (sprawdzone 30.09 poleceniem `sacctmgr`): konto `transformers_vsc` ma łącznie **5 × H100** dla wszystkich użytkowników, a partycja `hopper` ma 8 × H100 z limitem 5 dni. Dokument stwierdza, że ośmiu H100 w jednym zadaniu nie da się uzyskać. Dane mogą być nieaktualne, więc przed zgłoszeniem trzeba sprawdzić limity ponownie. Kampania funkcjonalna używała 8 workerów na węźle `dgx-2` (`PLAN.md:13`), co jest innym zasobem.
2. **Rozmiar wag.** Rozmiar pojedynczego checkpointu nie został zmierzony w tej sesji. Przed uruchomieniem trzeba sprawdzić go na Edenie (`ls -l` na katalogu wag), bo wagi leżą w home, który ma ok. 16 G wolnych.
3. **Plan pomija** rozgrzewkę lr (300 kroków) oraz łodygę konwolucyjną i głowę. Trzeba je uwzględnić w opisie eksperymentu.
4. **Nazwa wag** w planie (`k<step>.pt`) powinna być `k{step:05d}.pt`, np. `k00300.pt`.
5. **Katalog eksperymentu** trzeba utworzyć wg zasad z sekcji 4 i dopisać do `experiments_index.md` od razu po utworzeniu.
6. **Zapis.** Wyniki mają być agregatami, nie surowym formatem. Surowy format 24 × 50 zajmuje ok. 564 GiB, a w katalogu grupy jest 477 G wolnych (`2026-10-20_EDEN_MIEJSCE_NA_DYSKU.md`).
