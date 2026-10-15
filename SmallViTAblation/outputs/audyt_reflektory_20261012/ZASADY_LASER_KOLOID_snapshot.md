# Zasady modelu „lasery i koloidy” dla MałegoViT (2026-10-12, liczby przeliczone na biegu s47)

Ten plik ustala, co w obrazie „sfera laserów oświetla 600 koloidów” jest **dokładną matematyką sieci**, co jest **analogią**, a co jest **błędem**.
Obowiązuje dla kolejnych filmów w tym folderze (od filmu 7). Diagram architektury: `diagram_maly_vit/diagram_maly_vit.pdf`.
Pełny opis punkt po punkcie, z przypisami i literaturą: `opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.pdf`.

Wszystkie liczby: `przelicz_zasady_s47.py` → `PRZELICZENIE_ZASADY_S47.json`, bieg `assets/MalyViT_35000_od_zera_s47`, z którego pochodzą zapisane aktywacje.
Wcześniejsze `PRZECIWLASERY.json`, `OBROT_REFLEKTORA.json` i `PULAPKA_JAKOBIAN.json` liczono na wagach innego biegu (`01_bieg_35000_gesty_zapis`, ten sam przepis i ziarno). Zostają bez zmian jako zapis tamtego biegu.

## 1. Sieć, której dotyczą zasady

MałyViT (`assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py`):

- łodyga: 4 konwolucje 3×3 z krokiem 2 (3 → 24 → 48 → 96 → 192 kanały; pierwsze trzy z BatchNorm i GELU, ostatnia bez), klatka 480×320 → siatka 30×20 = **600 tokenów**, krok 16 px;
- do tokenów raz dodawane jest stałe kodowanie pozycji 2D sin-cos (192 wymiary);
- **6 bloków** timm (pre-LN): `x = x + attn(LN1(x))`, potem `x = x + mlp(LN2(x))`; uwaga ma 3 głowy, MLP to 192 → 768 → 192 z dokładnym GELU (erf); LayerNorm w blokach ma ε = 10⁻⁶ (wcześniejsze skrypty przyjmowały 10⁻⁵; wpływ na liczby poniżej 10⁻⁵ względnie);
- końcowa LayerNorm i głowa gęsta na każdym tokenie: 192 → 192 → GELU → 5 (logit „tu jest środek narzędzia” i odległości l, t, r, b do krawędzi ramki);
- strata: ogniskowa strata CenterNet na mapie środków + 2 · (1 − GIoU); 2 926 493 parametry; AdamW;
- bieg: `assets/MalyViT_35000_od_zera_s47` (ziarno 47, 806 checkpointów do kroku 65 640, czyli 60 epok po 1094 kroki). Przejście modelu z tymi wagami odtwarza zapisane aktywacje z R² ≥ 0,9999.

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

**Z3. Natężenie zależy od położenia samej cząstki.** z_jc = GELU(w_c · LN(x_j) + b_c): laser świeci tym mocniej, im lepiej cząstka leży na jego linii celowania. To pole sił zależne od położenia. W odróżnieniu od siły gradientowej w pęsecie optycznej nie jest to gradient żadnego potencjału: część wirowa pola jest prawie tak duża jak część potencjalna (Z15).

**Z4. Siły działają raz, przy przejściu przez blok.** Blok MLP to jeden krok: Δx_j = Σ_c z_jc v_c + b₂. Nie ma ciągłego trzymania cząstki w pułapce ani lepkości. Ostatni blok to tylko ostatnia „soczewka”; wcześniej działa 5 bloków, każdy z własnymi 768 laserami i z uwagą.

**Z5. Laser pcha tylko w jedną stronę, więc potrzebne są przeciwlasery.** GELU jest prawie nieujemne (minimum ≈ −0,17), więc laser c pcha mocno tylko w stronę +v_c, a w stronę −v_c może ciągnąć najwyżej słabo. Mocne pchnięcie w stronę −v_c wymaga innego lasera c' z v_c' ≈ −v_c. Słabo ujemnych diod jest jednak dużo (49 % na starcie, 79 % w 60. epoce) i razem dają 9–24 % siły bloku względem średniej po tokenach, więc w rachunkach nie wolno ich pomijać (w filmach 10–12 są obcięte do zera). Wypadkowa wzdłuż takiej osi to **różnica natężeń** pary przeciwnych laserów: jeśli laser z lewej świeci mocniej niż jego przeciwnik z prawej, cząstka przesuwa się w prawo; przy równych natężeniach pchnięcia się znoszą. Zmierzone na wagach ostatniego bloku biegu s47 (`PRZELICZENIE_ZASADY_S47.json`, część P; cos najbardziej przeciwnego partnera):

| epoka (krok) | lasery z partnerem cos < −0,5 | mediana cos partnera | udział 3 składowych w kierunkach v_c | mediana \|v_c\| |
|---|---|---|---|---|
| 0 (0) | 0 % | −0,22 | 3 % | 0,29 |
| 1 (1094) | 0 % | −0,23 | 5 % | 0,29 |
| 5 (5470) | 1 % | −0,28 | 13 % | 0,33 |
| 10 (10 940) | 14 % | −0,37 | 18 % | 0,43 |
| 20 (21 880) | 36 % | −0,46 | 22 % | 0,60 |
| 40 (43 760) | 52 % | −0,51 | 23 % | 0,78 |
| 60 (65 640) | **53 %** | −0,51 | 22 % | 0,84 |

Na starcie kierunki są losowe i przeciwlaserów nie ma. **Uczenie je tworzy**: od około 5. epoki rośnie liczba par przeciwnych laserów, a lasery stają się silniejsze (|v_c| rośnie 2,9 raza do 60. epoki). Po 40. epoce liczba par prawie się nie zmienia. Pary są przybliżone: żadna nie ma cos < −0,8, a najbardziej przeciwna para w 60. epoce ma cos −0,76. Nie ma wymogu równowagi: blok daje jedno pchnięcie, a cząstka nie musi spoczywać tam, gdzie siły się znoszą.

Porównanie z biegiem gęstym, z którego pochodziła wcześniejsza wersja tej tabeli: w 21. epoce przeciwlasery to 38 % wobec 37 % w s47, mediana cos partnera −0,47 w obu, udział 3 składowych 22 % w obu, |v_c| 0,64 wobec 0,62. Same wagi obu biegów rozeszły się całkowicie (odległość względna 1,0), a statystyki laserów są prawie takie same.

**Z6. Słaba siła zwrotna (zmierzona).** Laser „widzi” cząstkę przez LayerNorm, więc w przestrzeni strumienia rezydualnego jego kierunek celowania to s_c = γ ⊙ w_c (γ to skala LayerNorm). Mediana cos(γ ⊙ w_c, v_c) spada z 0,00 na starcie do −0,05 w 5. epoce, −0,12 w 10., −0,17 w 20. i −0,18 w 40. i 60. epoce. Bez poprawki na γ wychodzi prawie to samo (−0,18 w 60. epoce). Laser coraz częściej pcha cząstkę z powrotem, przeciw kierunkowi, z którego ją widzi: gdy cząstka przesunie się wzdłuż s_c, laser się rozjaśnia i odpycha ją wstecz.
Związek jest dokładny: ślad Jakobianu bloku to tr J = Σ_c GELU′(a_c) · w_c^T J_LN v_c ≈ (1/σ) · Σ_c GELU′(a_c) · (γ ⊙ w_c)^T v_c, gdzie σ to odchylenie standardowe tokenu w LayerNorm. Ujemny cosinus daje więc ujemny ślad, czyli kontrakcję objętości. Zmierzony ślad (mediana po 600 tokenach, podzielony przez 192) spada z −0,0006 na starcie do −0,005 w 10. epoce i −0,016 w 60. epoce. To słaba kontrakcja: ściąga tylko około 51 % kierunków, a wir jest prawie tak silny jak część potencjalna (Z15).
Ślad opisuje przybliżenie ciągłe, a blok robi jeden skończony krok x → x + F(x). Dla tego kroku liczy się log|det(I + J)|: na kierunek wynosi od −0,0008 na starcie do −0,021 w 60. epoce i jest ujemny dla wszystkich 600 tokenów. Jedno przejście przez MLP ostatniego bloku zmniejsza więc objętość małego otoczenia tokenu około 1,2 raza na starcie i około 55 razy w 60. epoce (192 wymiary). Kontrakcja na kierunek jest słaba, w całej objętości duża.

**Z7. Między cząstkami działa tylko uwaga.** Lasery MLP działają na każdą cząstkę osobno. Bezpośrednie siły cząstka–cząstka daje wyłącznie uwaga (3 głowy). W czasie uczenia „oddziaływanie” tokenów to sprzężenie przez wspólne wagi (jądro wpływu, empiryczne NTK), a nie siła.

**Z8. W czasie uczenia cząstkami są lasery, nie tokeny.** Uczenie przestawia w_c, v_c i b_c. Obrót między 1. a 60. epoką: mediana cos celowników w_c wynosi 0,24, a kierunków pchnięcia v_c 0,14. Token nie ma własnego stanu w czasie uczenia: z_j(t) = f(x_j; θ_t). To, co widać jako ruch tokenu w czasie uczenia, to przesunięcie **plamki ogniskowej** na końcu toru przez głębokość. Pamięć ma za to całe uczenie, wspólnie dla wszystkich tokenów: momenty AdamW (β₁ = 0,9, czyli około 10 kroków; β₂ = 0,999, czyli około 1000 kroków) i średnie kroczące BatchNorm w łodydze (około 10 kroków; latent zapisywano w trybie eval, więc z tymi średnimi). Do tego norma gradientu jest w każdym kroku przycinana do 1 (`trening.py:80`). Ta wspólna pamięć może dawać opóźnienia w ruchu tokenów rzędu zmierzonego τ (3–13 kroków). Pomiary: szybkie drgania w 84–98 % pochodzą z wcześniejszych warstw (wiązka wejściowa wchodzi w ostatnią soczewkę w innym miejscu); wolny dryf w 75 % z przestawiania laserów ostatniego bloku. Źródło: Eden, `eden/wyniki/TEST_SFERY.json` i `wiazki_koloidy/WIAZKI.json`, 100 zdjęć. To ten sam bieg s47: zapisane tam wagi fc1 różnią się od wag s47 o 0,02–0,04 % (precyzja float16), a od biegu gęstego o 40–106 % (poza krokiem 0, gdzie oba biegi mają te same wagi; część W przeliczenia). Sprzężony obraz obu czasów: Herty i Liu 2026.

**Z9. Czego nie mówić.**
- Nie mówić, że koloid „nadąża za pułapką” w czasie uczenia: z_j(t) = f(x_j; θ_t) w każdym kroku, więc nie ma opóźnienia, lepkości ani własnego szumu cząstki. Odwrotny obraz, w którym pułapka (wagi ostatniego bloku) powoli nadąża za całą chmurą koloidów, jest hipotezą w trakcie pomiaru (`NADAZANIE_PULAPKI_S47`).
- Nie mówić, że krok uczenia to temperatura. W biegach z pokrętłami (RP128, okno 12034–13128) τ_1e wynosi 13, 6 i 3 kroki dla lr 1,25e-4, 5e-4 i 2e-3, czyli τ ∝ η^−0,5 zamiast η^−1, a rozmiar klatki nie rośnie monotonicznie. Prosty model koloidu w dołku z SGD nie zgadza się ilościowo; przy AdamW potrzebne jest osobne równanie (Malladi i in. 2022). Zastrzeżenie: τ_1e to czas zaniku F_s, nie autokorelacja wychylenia.
- Nie mówić, że korelacje sąsiadów (0,6–0,85) to oddziaływania: wynikają ze wspólnego napędu przez wagi. Synchronizację trzeba wykazać maleniem odległości, nie korelacją przyrostów (Agazzi i in. 2026).
- Pułapka w czasie uczenia istnieje tylko w krajobrazie straty: drgania w poprzek doliny minimów i dryf wzdłuż niej (Pashakhanloo i Koulakov 2023), a nie przy każdym tokenie osobno.

**Z10. Zasady rysowania w filmach (od filmu 7).**
- Laser c stoi na sferze w punkcie −P·v̂_c i świeci do środka, więc pcha cząstki w stronę +P·v̂_c. Przeciwlasery same lądują po przeciwnych stronach sfery. P to stały rzut 192D → 3D, zawsze podany w filmie. Trzy główne składowe niosą tylko 3–23 % wariancji kierunków v_c (3 % na starcie, 22–23 % od 20. epoki), więc rzut jest ilustracją, a nie pomiarem.
- Jasność wiązki = z_jc; długość strzałki pchnięcia = z_jc · |P v_c|; strzałka wypadkowa = P · Δx_j.
- Celownik w_c pokazywać osobno (np. stożek lub kolor), bo to inny wektor niż kierunek pchnięcia.
- Kierunki Fibonacciego d_c tylko jako umowny układ ze starszych filmów, z podpisem, że to wybór, a nie pomiar.
- Drgania w czasie uczenia rysować jako drżenie wiązki wejściowej (z wcześniejszych bloków), a wędrówkę laserów jako przyczynę wolnego dryfu.
- Każda nowa animacja to nowy skrypt `film_lasery_N_<opis>.py` i nowe `.mp4`; starych nie przerabiać.

**Z11. Matryca laserów bloku: jak lasery uczą się.** Jeden blok MLP to jedna matryca 768 laserów (6 matryc, po jednej na blok, z uwagą między nimi), jak reflektor matrycowy w samochodzie. Od Z13 słowo „reflektor” oznacza jeden laser, czyli jeden neuron. Laser c ma celownik w_c, kierunek wyrzutu v_c i próg b_c.
- *Przełączanie laserów dla danej cząstki* (które świecą i jak jasno) to aktywacje z_jc. Dzieje się natychmiast przy każdym tokenie, jak w reflektorze matrycowym reagującym na scenę. To nie jest uczenie.
- *Uczenie to powolne przestawianie samych laserów.* Między krokiem 0 a 65 640 (60. epoka; `PRZELICZENIE_ZASADY_S47.json`, część O): kierunek wyrzutu v_c tego samego lasera ma medianę cos 0,11 (laser przesuwa się po sferze o około 84°), celownik w_c cos 0,20, a długość v_c rośnie 2,9 raza (w_c 1,7 raza). Do 21. epoki (krok 22 974): v_c cos 0,32 (około 71°), w_c cos 0,45, długość v_c ×2,1. W oknach po 5 epok cos kierunku v_c wynosi 0,77 w pierwszych 5 epokach, 0,86–0,91 w epokach 5–15 i 0,93–0,97 później, więc ruch jest płynny i z czasem zwalnia.
- *Matryca obraca się jako całość tylko w małej części.* Jeden wspólny obrót 192D (Prokrustes na surowych wektorach) wyjaśnia 10–13 % zmiany v_c między kolejnymi etapami uczenia, a zero (ta sama zmiana przetasowana między laserami) 6–12 %. Nadwyżka to 1–3 punkty procentowe do 21. epoki i około 5 punktów w epokach 21–60. Po stronie celowników w_c wspólny obrót jest wyraźniejszy: na całej drodze 0 → 60 wyjaśnia 21 % zmiany wobec 12 % zera, a na kierunkach jednostkowych w późnych oknach 5-epokowych 24 % wobec 10 %. Co najmniej trzy czwarte zmiany to jednak osobne przestawianie każdego lasera, w tym tworzenie par przeciwnych (Z5).
- *Wcześniejsze bloki nie obracają ostatniej matrycy.* To poprzednie matryce w łańcuchu: przesuwają cząstkę, zanim dotrze do ostatniej. W czasie uczenia dają 84–98 % szybkich drgań (Z8).

**Z12. 600 reflektorów za szklaną sferą (obraz „lentykularny”, 2026-10-12).** Każdy token ma swój reflektor: piksel z 768 kanałami kierunkowymi zamiast 3 kanałów RGB. Wiązka kanału c wychodzi z reflektora j pod swoim kątem, trafia w szklaną sferę, załamuje się i dochodzi do cząsteczki j; wektor siły tego kanału to z_jc · v_c. Żeby obraz był zgodny z siecią:
1. *Reflektory są identyczne* (wagi MLP wspólne dla tokenów): kanał c ma to samo v_c (ustawienie galw) i ten sam celownik w_c we wszystkich 600 reflektorach; różnią się tylko jasności z_jc. W czasie uczenia galwa kanału c obracają się naraz we wszystkich reflektorach. Jak ekran lentykularny: jedna optyka, różne obrazy dla różnych odbiorców.
2. *Sfera działa telecentrycznie*: po załamaniu wiązki kanału c dochodzą do każdej cząsteczki równolegle, w kierunku v_c, niezależnie od położenia reflektora. Punkt wejścia wiązki c → j to przecięcie sfery z prostą przez cząsteczkę j o kierunku v_c.
3. *Reflektor j oświetla tylko cząsteczkę j*: MLP nie miesza tokenów. Mieszanie to uwaga, osobny element (można go pokazać jako kontrolowany „przeciek” j → k z wagą A_jk).
4. *Reflektor ma czujnik swojej cząsteczki*: jasność z_jc = GELU(w_c · LN(x_j) + b_c) zależy od bieżącego położenia cząsteczki (jak śledzenie oczu w ekranie lentykularnym).
5. *To nie jest RGB*: 768 kierunków w przestrzeni 192D jest nadmiarowe i nieortogonalne (pary przeciwne, Z5).
Jeden taki stopień = jeden blok; pełna sieć to 6 stopni z uwagą między nimi. Film 10 (`film_lasery_10_reflektory_sfera.py`): położenie cząsteczki = P·W₂·(z_j − z̄), czyli rzut sumy jej 768 wektorów, więc wiązki i położenia są spójne.
Uwaga do intuicji: tak liczone położenie (filmy 10–12) to przesunięcie, które nadaje blok (F_j − F̄), a nie stan tokenu x_j. Oba obrazy są podobne (korelacja odległości między tokenami 0,94 na starcie i 0,86 w 60. epoce), ale do śledzenia ruchu tokenu właściwym położeniem jest x_j po LayerNorm (tak liczą filmy 17–19), a suma wiązek to strzałka przesunięcia.

**Z13. Poprawka do Z12: 768 reflektorów po 600 diod (2026-10-12).** Reflektor to neuron (kanał), nie token. Reflektor c stoi za sferą w punkcie −RR · d_c i ma 600 diod, po jednej na cząsteczkę; dioda j świeci z jasnością z_jc. Wspólne wagi wynikają wtedy z budowy: reflektor ma jedno ustawienie (kierunek d_c) dla wszystkich cząsteczek, a w czasie uczenia obraca się (wędruje po sferze). Po załamaniu 600 wiązek reflektora biegnie równolegle w kierunku d_c, każda do swojej cząsteczki. Film 10 ma te same 460 800 wiązek (te same kierunki i jasności), ale grupuje je po tokenach; obowiązuje układ z filmu 11 (`film_lasery_11_reflektory_kanalow.py`). 600 diod reflektora c na siatce obrazu to mapa aktywacji neuronu c.

**Z14. Stałe reflektory, wiązki sterowane galwami, sfera uczona (2026-10-12, film 12).** Wariant równoważny Z13, lepszy do prezentacji: reflektory stoją na stałe na sferze Fibonacciego (układ z aplikacji), każda z 600 wiązek reflektora c ma własne galwo, które celuje w swoją cząsteczkę (szybkie śledzenie, bez znaczenia dla siły). Sfera załamuje wiązkę raz, na swojej powierzchni, w punkcie E_jc = p_j − s · d_c; w środku wiązka biegnie prosto do cząsteczki w kierunku d_c, wspólnym dla wszystkich 600 wiązek reflektora. Galwo celuje w E_jc. Miejsca reflektorów: 768 punktów spirali Fibonacciego, kanał c dostaje miejsce najbliżej za swoim kierunkiem −d_c w ostatnim kroku filmu (przypisanie węgierskie), żeby załamanie było małe. Film 12 liczono na wagach biegu gęstego (kroki 0–22 974): średni cos miejsca 0,975, a ważony jasnością średni kąt załamania wynosi około 107° na starcie, 88° w kroku 1400 i 33° w kroku 22 974. Na wagach s47, z przypisaniem do kierunków z 60. epoki: średni cos 0,99, kąt 108° na starcie, 68° w kroku 22 974 i 27° w 60. epoce (część F przeliczenia). Spadek kąta wynika z tego, że miejsca dobrano do kierunków końcowych, i nie jest odkryciem. Warunek z sieci: galwa mogą ruszać się niezależnie tylko w celowaniu; kierunek wiązki w sferze musi być wspólny (inaczej każdy token miałby własne wagi). W czasie uczenia reflektory stoją, punkty i kąty załamania się zmieniają: w sferze „siedzą” wagi v_c.

**Z15. „Pułapka” bloku to siodło z wirem (dokładny Jakobian, bieg s47).** Wokół tokenu blok MLP działa jak pole sił F(x_j + δ) ≈ F(x_j) + J_j δ, gdzie J_j = W₂ diag(GELU′(a_j)) W₁ J_LN(x_j). Wzór jest dokładny: przejście modelu odtwarza zapisane aktywacje z R² ≥ 0,99993, F ze wzoru zgadza się z wyjściem modelu z błędem względnym ≤ 5·10⁻⁶, a J ze wzoru zgadza się z Jakobianem policzonym automatycznie przez PyTorch (autograd, float64) z błędem względnym ≤ 10⁻¹⁵. Mediany po 600 tokenach zdjęcia TRAIN 0 między epoką 0 a 60 (część J przeliczenia):
- ujemnych wartości własnych części symetrycznej S jest stale około 51 %: połowa kierunków ściąga, połowa wypycha;
- symetryczność ‖S‖²/‖J‖² rośnie z 0,50 do 0,58 (macierz losowa: 0,503), a wir ‖A‖/‖S‖ spada z 0,99 do 0,86 (macierz losowa: 0,995);
- najsilniejszy kierunek ściągający rośnie z −0,30 do −1,02, a najsilniejszy wypychający z 0,30 do 0,60; strona ściągająca rośnie więc mocniej i stąd ujemny ślad (Z6);
- norma ‖J‖₂ rośnie z 0,47 do 1,39, czyli około 3 razy (szczyt 1,64 w 20. epoce); dodatnia część rzeczywista wartości własnych J zostaje przez całe uczenie.

Dla porównania: w pęsecie optycznej część wirowa to około 1–4 % sztywności, a wyuczone autoenkodery są symetryczne w 95–99 %. Słowo „pułapka” jest więc tylko skrótem: to siodło z silnym wirem, które uczenie lekko ściąga. Pełna tabela: `matematyka_reflektorow/matematyka_reflektorow.pdf`, rozdział 4.

## 4. Wierność fizyczna w skrócie

| Element | Status |
|---|---|
| 768 sił działających na token w bloku MLP, natężenie zależne od położenia | **dokładne** (wzór bloku) |
| przeciwlasery i różnica natężeń wzdłuż osi | **dokładne** jako opis; pary przybliżone, tworzy je uczenie |
| siła zwrotna lasera | **słaba, zmierzona** (cos(γ ⊙ w_c, v_c) = −0,18; ślad J / 192 = −0,016 w 60. epoce) |
| „pułapka” bloku MLP (Jakobian pola) | **dokładne** jako wzór; zmierzona: siodło z wirem, ‖A‖/‖S‖ = 0,99 → 0,86 |
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
8. Czy rosnący pod koniec uczenia wspólny obrót celowników w_c (Z11) to nadążanie za obrotem całego strumienia rezydualnego, wywołanym przez wcześniejsze bloki: porównać z obrotem Prokrustesa stanów x_j między tymi samymi checkpointami.
9. Nowe wersje filmów 8–12 na wagach s47 (jako nowe pliki; stare filmy zostają bez zmian).

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
