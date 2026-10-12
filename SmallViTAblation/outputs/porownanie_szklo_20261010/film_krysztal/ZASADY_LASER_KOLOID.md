# Zasady modelu „lasery i koloidy” dla MałegoViT (2026-10-12)

Ten plik ustala, co w obrazie „sfera laserów oświetla 600 koloidów” jest **dokładną matematyką sieci**, co jest **analogią**, a co jest **błędem**.
Obowiązuje dla kolejnych filmów w tym folderze (od filmu 7). Diagram architektury: `diagram_maly_vit/diagram_maly_vit.pdf`.
Pomiar przeciwlaserów: `sprawdz_przeciwlasery.py` → `PRZECIWLASERY.json`. Obrót reflektora: `sprawdz_obrot_reflektora.py` → `OBROT_REFLEKTORA.json`.

## 1. Sieć, której dotyczą zasady

MałyViT (`assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py`):

- łodyga: 4 konwolucje 3×3 z krokiem 2 (3 → 24 → 48 → 96 → 192 kanały), klatka 480×320 → siatka 30×20 = **600 tokenów**, krok 16 px;
- do tokenów raz dodawane jest stałe kodowanie pozycji 2D sin-cos (192 wymiary);
- **6 bloków** timm (pre-LN): `x = x + attn(LN1(x))`, potem `x = x + mlp(LN2(x))`; uwaga ma 3 głowy, MLP to 192 → 768 → 192 z GELU;
- końcowa LayerNorm i głowa gęsta na każdym tokenie: 192 → 192 → GELU → 5 (logit „tu jest środek narzędzia” i odległości l, t, r, b do krawędzi ramki);
- strata: ogniskowa strata CenterNet na mapie środków + 2 · (1 − GIoU); około 2,9 mln parametrów; AdamW.

Korekta do rozmowy z 2026-10-12: przed ostatnim blokiem jest **5** bloków, a nie 11.

## 2. Słownik: co jest czym

| Obraz fizyczny | Obiekt w sieci | Wzór |
|---|---|---|
| cząstka (koloid) | token j, stan w strumieniu rezydualnym | x_j ∈ R^192, j = 1…600 |
| czas ruchu cząstki | **głębokość**: przejście przez kolejne bloki | x_j^(l+1) = x_j^(l) + uwaga + MLP |
| laser c (w każdym bloku 768, razem 6 × 768) | neuron c warstwy ukrytej MLP | para wektorów (w_c, v_c) i próg b_c |
| celownik lasera | wiersz c macierzy fc1 | w_c ∈ R^192 |
| natężenie światła na cząstce | aktywacja po GELU („latent”, 768 na token) | z_jc = GELU(w_c · LN(x_j) + b_c) |
| kierunek pchnięcia lasera | kolumna c macierzy fc2 | v_c ∈ R^192 |
| siła wypadkowa w bloku | wyjście MLP | Δx_j = Σ_c z_jc v_c + b₂ |
| siły między cząstkami | uwaga (3 głowy) | Δx_j = Σ_k A_jk (V x_k) |
| ruch laserów | uczenie: AdamW zmienia w_c, v_c, b_c | w_c(t), v_c(t) |

## 3. Zasady

**Z1. Cząstka ma stan tylko wzdłuż głębokości.** Token jest prawdziwą cząstką w czasie płynącym przez bloki: jego położenie w bloku l+1 zależy od położenia w bloku l. To ujęcie z prac Geshkovskiego i Rigolleta (tokeny jako układ oddziałujących cząstek, czas = głębokość).

**Z2. Laser ma dwa wektory.** Celownik w_c decyduje, *kiedy* laser świeci; kierunek v_c decyduje, *w którą stronę* pcha. To nie jest jeden promień od lasera do cząstki. W filmach 1–6 kierunek pchnięcia był umownym kierunkiem d_c ze sfery Fibonacciego. Wzór aplikacji p = Σ_c (z_c − μ_c) d_c ma postać fc2, tylko z kierunkami Fibonacciego zamiast prawdziwych v_c.

**Z3. Natężenie zależy od położenia samej cząstki.** z_jc = GELU(w_c · LN(x_j) + b_c): laser świeci tym mocniej, im lepiej cząstka leży na jego linii celowania. To pole sił zależne od położenia, podobne do siły gradientowej w pęsecie optycznej.

**Z4. Siły działają raz, przy przejściu przez blok.** Blok MLP to jeden krok: Δx_j = Σ_c z_jc v_c + b₂. Nie ma ciągłego trzymania cząstki w pułapce ani lepkości. Ostatni blok to tylko ostatnia „soczewka”; wcześniej działa 5 bloków, każdy z własnymi 768 laserami i z uwagą.

**Z5. Laser pcha tylko w jedną stronę, więc potrzebne są przeciwlasery.** GELU jest prawie nieujemne (minimum ≈ −0,17), więc laser c pcha praktycznie tylko w stronę +v_c. Pchnięcie w stronę −v_c wymaga innego lasera c' z v_c' ≈ −v_c. Wypadkowa wzdłuż takiej osi to **różnica natężeń** pary przeciwnych laserów: jeśli laser z lewej świeci mocniej niż jego przeciwnik z prawej, cząstka przesuwa się w prawo; przy równych natężeniach pchnięcia się znoszą. Zmierzone na wagach ostatniego bloku (`PRZECIWLASERY.json`, cos najbardziej przeciwnego partnera):

| krok (epoka) | lasery z partnerem cos < −0,5 | mediana cos partnera | udział 3 składowych w kierunkach v_c | mediana \|v_c\| |
|---|---|---|---|---|
| 0 (0) | 0 % | −0,22 | 3 % | 0,29 |
| 5300 (4,8) | 2 % | −0,28 | 13 % | 0,34 |
| 11 000 (10,1) | 17 % | −0,38 | 18 % | 0,44 |
| 22 974 (21,0) | **38 %** | −0,47 | 22 % | 0,64 |

Na starcie kierunki są losowe i przeciwlaserów nie ma. **Uczenie je tworzy**: od około 5. epoki rośnie liczba par przeciwnych laserów, a lasery stają się silniejsze. Pary są przybliżone (żadna nie ma cos < −0,8). Nie ma wymogu równowagi: blok daje jedno pchnięcie, a cząstka nie musi spoczywać tam, gdzie siły się znoszą.

**Z6. Słaba siła zwrotna (hipoteza).** Mediana cos(w_c, v_c) spada z 0,00 na starcie do −0,16 w 21. epoce. Laser coraz częściej pcha cząstkę z powrotem, przeciw kierunkowi, z którego ją „widzi”: gdy cząstka przesunie się wzdłuż w_c, laser się rozjaśnia i odpycha ją wstecz. To zalążek pułapki wzdłuż głębokości. Zastrzeżenie: w_c działa na LN(x), a v_c dodaje się do x, więc porównanie kierunków jest przybliżone (LayerNorm skaluje osie).

**Z7. Między cząstkami działa tylko uwaga.** Lasery MLP działają na każdą cząstkę osobno. Bezpośrednie siły cząstka–cząstka daje wyłącznie uwaga (3 głowy). W czasie uczenia „oddziaływanie” tokenów to sprzężenie przez wspólne wagi (jądro wpływu, empiryczne NTK), a nie siła.

**Z8. W czasie uczenia cząstkami są lasery, nie tokeny.** Uczenie przestawia w_c, v_c i b_c (obrót laserów: cos 0,24 między 1. a 60. epoką). Token nie ma własnego stanu w czasie uczenia: z_j(t) = f(x_j; θ_t). To, co widać jako ruch tokenu w czasie uczenia, to przesunięcie **plamki ogniskowej** na końcu toru przez głębokość. Pomiary: szybkie drgania w 84–98 % pochodzą z wcześniejszych warstw (wiązka wejściowa wchodzi w ostatnią soczewkę w innym miejscu); wolny dryf w 75 % z przestawiania laserów ostatniego bloku. Sprzężony obraz obu czasów: Herty i Liu 2026.

**Z9. Czego nie mówić.**
- Nie mówić, że koloid „nadąża za pułapką” w czasie uczenia: z_j(t) = f(x_j; θ_t) w każdym kroku, więc nie ma opóźnienia, lepkości ani własnego szumu cząstki.
- Nie mówić, że krok uczenia to temperatura. W biegach z pokrętłami (RP128, okno 12034–13128) τ_1e wynosi 13, 6 i 3 kroki dla lr 1,25e-4, 5e-4 i 2e-3, czyli τ ∝ η^−0,5 zamiast η^−1, a rozmiar klatki nie rośnie monotonicznie. Prosty model koloidu w dołku z SGD nie zgadza się ilościowo; przy AdamW potrzebne jest osobne równanie (Malladi i in. 2022). Zastrzeżenie: τ_1e to czas zaniku F_s, nie autokorelacja wychylenia.
- Nie mówić, że korelacje sąsiadów (0,6–0,85) to oddziaływania: wynikają ze wspólnego napędu przez wagi. Synchronizację trzeba wykazać maleniem odległości, nie korelacją przyrostów (Agazzi i in. 2026).
- Pułapka w czasie uczenia istnieje tylko w krajobrazie straty: drgania w poprzek doliny minimów i dryf wzdłuż niej (Pashakhanloo i Koulakov 2023), a nie przy każdym tokenie osobno.

**Z10. Zasady rysowania w filmach (od filmu 7).**
- Laser c stoi na sferze w punkcie −P·v̂_c i świeci do środka, więc pcha cząstki w stronę +P·v̂_c. Przeciwlasery same lądują po przeciwnych stronach sfery. P to stały rzut 192D → 3D, zawsze podany w filmie. Trzy główne składowe niosą tylko 3–22 % wariancji kierunków v_c, więc rzut jest ilustracją, a nie pomiarem.
- Jasność wiązki = z_jc; długość strzałki pchnięcia = z_jc · |P v_c|; strzałka wypadkowa = P · Δx_j.
- Celownik w_c pokazywać osobno (np. stożek lub kolor), bo to inny wektor niż kierunek pchnięcia.
- Kierunki Fibonacciego d_c tylko jako umowny układ ze starszych filmów, z podpisem, że to wybór, a nie pomiar.
- Drgania w czasie uczenia rysować jako drżenie wiązki wejściowej (z wcześniejszych bloków), a wędrówkę laserów jako przyczynę wolnego dryfu.
- Każda nowa animacja to nowy skrypt `film_lasery_N_<opis>.py` i nowe `.mp4`; starych nie przerabiać.

**Z11. Reflektor matrycowy: jak lasery uczą się.** Jeden blok MLP to jeden reflektor z 768 lusterkami (6 reflektorów, po jednym na blok, z uwagą między nimi). Lusterko c ma celownik w_c, kierunek wyrzutu v_c i próg b_c.
- *Przełączanie lusterek dla danej cząstki* (które świecą i jak jasno) to aktywacje z_jc. Dzieje się natychmiast przy każdym tokenie, jak w reflektorze matrycowym reagującym na scenę. To nie jest uczenie.
- *Uczenie to powolne przestawianie samych lusterek.* Między krokiem 0 a 22 974 (`OBROT_REFLEKTORA.json`): kierunek wyrzutu v_c tego samego lasera ma medianę cos 0,31 (laser przesuwa się po sferze o około 70°), celownik w_c cos 0,45, a długość v_c rośnie 2,2 raza. W oknach po kilka tysięcy kroków cos wynosi 0,80–0,95, więc ruch jest płynny.
- *Reflektor nie obraca się jako całość.* Jeden wspólny obrót 192D wyjaśnia 10–14 % zmiany, a zero (ta sama zmiana przetasowana między laserami) 7–12 %. Nadwyżka to 1–3 punkty procentowe. Prawie cała zmiana to osobne przestawianie każdego lusterka, w tym tworzenie par przeciwnych (Z5).
- *Wcześniejsze bloki nie obracają ostatniego reflektora.* To poprzednie reflektory w łańcuchu: przesuwają cząstkę, zanim dotrze do ostatniego. W czasie uczenia dają 84–98 % szybkich drgań (Z8).

**Z12. 600 reflektorów za szklaną sferą (obraz „lentykularny”, 2026-10-12).** Każdy token ma swój reflektor: piksel z 768 kanałami kierunkowymi zamiast 3 kanałów RGB. Wiązka kanału c wychodzi z reflektora j pod swoim kątem, trafia w szklaną sferę, załamuje się i dochodzi do cząsteczki j; wektor siły tego kanału to z_jc · v_c. Żeby obraz był zgodny z siecią:
1. *Reflektory są identyczne* (wagi MLP wspólne dla tokenów): kanał c ma to samo v_c (ustawienie galw) i ten sam celownik w_c we wszystkich 600 reflektorach; różnią się tylko jasności z_jc. W czasie uczenia galwa kanału c obracają się naraz we wszystkich reflektorach. Jak ekran lentykularny: jedna optyka, różne obrazy dla różnych odbiorców.
2. *Sfera działa telecentrycznie*: po załamaniu wiązki kanału c dochodzą do każdej cząsteczki równolegle, w kierunku v_c, niezależnie od położenia reflektora. Punkt wejścia wiązki c → j to przecięcie sfery z prostą przez cząsteczkę j o kierunku v_c.
3. *Reflektor j oświetla tylko cząsteczkę j*: MLP nie miesza tokenów. Mieszanie to uwaga, osobny element (można go pokazać jako kontrolowany „przeciek” j → k z wagą A_jk).
4. *Reflektor ma czujnik swojej cząsteczki*: jasność z_jc = GELU(w_c · LN(x_j) + b_c) zależy od bieżącego położenia cząsteczki (jak śledzenie oczu w ekranie lentykularnym).
5. *To nie jest RGB*: 768 kierunków w przestrzeni 192D jest nadmiarowe i nieortogonalne (pary przeciwne, Z5).
Jeden taki stopień = jeden blok; pełna sieć to 6 stopni z uwagą między nimi. Film 10 (`film_lasery_10_reflektory_sfera.py`): położenie cząsteczki = P·W₂·(z_j − z̄), czyli rzut sumy jej 768 wektorów, więc wiązki i położenia są spójne.

**Z13. Poprawka do Z12: 768 reflektorów po 600 diod (2026-10-12).** Reflektor to neuron (kanał), nie token. Reflektor c stoi za sferą w punkcie −RR · d_c i ma 600 diod, po jednej na cząsteczkę; dioda j świeci z jasnością z_jc. Wspólne wagi wynikają wtedy z budowy: reflektor ma jedno ustawienie (kierunek d_c) dla wszystkich cząsteczek, a w czasie uczenia obraca się (wędruje po sferze). Po załamaniu 600 wiązek reflektora biegnie równolegle w kierunku d_c, każda do swojej cząsteczki. Film 10 ma te same 460 800 wiązek (te same kierunki i jasności), ale grupuje je po tokenach; obowiązuje układ z filmu 11 (`film_lasery_11_reflektory_kanalow.py`). 600 diod reflektora c na siatce obrazu to mapa aktywacji neuronu c.

**Z14. Stałe reflektory, wiązki sterowane galwami, sfera uczona (2026-10-12, film 12).** Wariant równoważny Z13, lepszy do prezentacji: reflektory stoją na stałe na sferze Fibonacciego (układ z aplikacji), każda z 600 wiązek reflektora c ma własne galwo, które celuje w swoją cząsteczkę (szybkie śledzenie, bez znaczenia dla siły). Sfera działa jak soczewka gradientowa: wiązka biegnie w niej po łuku i dochodzi do cząsteczki w kierunku d_c, wspólnym dla wszystkich 600 wiązek reflektora. Warunek z sieci: galwa mogą ruszać się niezależnie tylko w celowaniu; kierunek wiązki na końcu musi być wspólny (inaczej każdy token miałby własne wagi). W czasie uczenia reflektory stoją, a zmienia się to, jak sfera zagina wiązki: w niej „siedzą” wagi v_c. Łuk w filmie to krzywa Hermite'a (ilustracja), nie ścisły ślad promienia w ośrodku gradientowym.

## 4. Wierność fizyczna w skrócie

| Element | Status |
|---|---|
| 768 sił działających na token w bloku MLP, natężenie zależne od położenia | **dokładne** (wzór bloku) |
| przeciwlasery i różnica natężeń wzdłuż osi | **dokładne** jako opis; pary przybliżone, tworzy je uczenie |
| siła zwrotna lasera | **słaba hipoteza** (cos w_c, v_c = −0,16) |
| uwaga jako siły między cząstkami | **dokładne** wzdłuż głębokości |
| lasery jako ruchome cząstki w czasie uczenia | **dokładne** (to są parametry) |
| koloid nadążający za pułapką w czasie uczenia | **błędne** (brak własnego stanu tokenu) |
| krok uczenia = temperatura | **niepotwierdzone**, ilościowo się nie zgadza |
| położenie w 3D (rzut Fibonacciego lub P) | **umowne** |

## 5. Otwarte testy (z lektury 10 prac, 2026-10-12)

1. Test RPI (tasowanie patchy przy zachowanym kodowaniu pozycji) i kontrola RPT, miara SSDC, krzywe po warstwach, więcej ziaren bez kodowania pozycji (Mannes 2026).
2. Surogaty rzutu: błądzenie losowe, z pędem (β1 = 0,9) i OU z dopasowaną kowariancją, przepuszczone przez ten sam rzut 3D; kilka zestawów kierunków (Antognini i Sohl-Dickstein 2018).
3. Profile głębokości (rozrzut, efektywny wymiar, podobieństwa cosinusowe, energia oddziaływania po 6 blokach) w każdym z 806 checkpointów: jak uczenie przebudowuje przepływ tokenów przez głębokość (Isobe i in. 2026, Herty i Liu 2026).
4. Jądro lokalnej sprężystości: przesunięcie tokenu i po kroku z jednego źródła k, w funkcji odległości na siatce i treści (He i Su 2019); przy AdamW mierzyć faktyczne przesunięcia.
5. Prawdziwy czas relaksacji z autokorelacji wychylenia od środka klatki oraz krzywizna straty; sprawdzić skalowanie z lr przy AdamW.
6. Geometria dryfu: stabilność macierzy token–token, Prokrustes (obrót wobec odkształcenia), dyfuzja kątowa (Qin i in. 2021).
7. Film 7: lasery w punktach −P·v̂_c, przeciwlasery, osobno celowniki w_c.

## 6. Literatura

- Geshkovski i in. (2023), A mathematical perspective on Transformers — https://doi.org/10.48550/arXiv.2312.10794
- Rigollet (2025), The Mean-Field Dynamics of Transformers — https://doi.org/10.48550/arXiv.2512.01868
- Herty, Liu (2026), A Mean-Field Theory of Transformers: coupled data–parameter dynamics — https://arxiv.org/abs/2608.25055
- Isobe, Inoue, Imaizumi (2026), Training-Induced Escape from Token Clustering — https://doi.org/10.48550/arXiv.2605.07772
- Agazzi i in. (2026), Stochastic Scaling Limits and Synchronization by Noise in Deep Transformer Models — https://doi.org/10.48550/arXiv.2604.26898
- Pashakhanloo, Koulakov (2023), SGD-induced drift of representation in a two-layer neural network — https://doi.org/10.48550/arXiv.2302.02563
- He, Su (2019), The Local Elasticity of Neural Networks — https://arxiv.org/abs/1910.06943
- Qin i in. (2021), Coordinated drift of receptive fields during noisy representation learning — https://doi.org/10.1101/2021.08.30.458264
- Mannes (2026), Positional Encodings Anchor Spatial Structure in Vision Transformers — https://doi.org/10.48550/arXiv.2606.00124
- Antognini, Sohl-Dickstein (2018), PCA of high dimensional random walks with comparison to neural network training — https://arxiv.org/abs/1806.08805
- Malladi i in. (2022), On the SDEs and Scaling Rules for Adaptive Gradient Algorithms — https://doi.org/10.48550/arXiv.2205.10287
- Mei, Montanari, Nguyen (2018), A mean field view of the landscape of two-layer neural networks — https://doi.org/10.1073/pnas.1806579115
- Geva i in. (2021), Transformer Feed-Forward Layers Are Key-Value Memories (klucz = w_c, wartość = v_c; niesprawdzone w tej sesji przez Undermind)
