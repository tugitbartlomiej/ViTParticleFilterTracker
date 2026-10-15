> **Archiwum (2026-10-14).** Aktualna, zaktualizowana wersja zasad (razem z treścią `NADAZANIE_PULAPKI_S47.md`) jest w `app/ZASADY_DZIALANIA_UKLADU.md`. Od 2026-10-14 słowo „pułapka” oznacza pułapkę optyczną z laserów (pole i ognisko wiązek), a dawny „środek drgań” (średnia ±300 kroków) nazywa się „średnia skoków koloidu”. Ten plik zostaje bez zmian treści, bo odwołują się do niego skrypty filmów.

# Zasady modelu „lasery i koloidy” dla MałegoViT (2026-10-12, aktualizacja 2026-10-13; liczby przeliczone na biegu s47)

Ten plik ustala, co w obrazie „sfera laserów oświetla 600 koloidów” jest **dokładną matematyką sieci**, co jest **analogią**, a co jest **błędem**.
Obowiązuje dla kolejnych filmów w tym folderze (od filmu 7). Diagram architektury: `diagram_maly_vit/diagram_maly_vit.pdf`.
Pełny opis punkt po punkcie, z przypisami i literaturą: `opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.pdf` (źródło `.tex`, wersja z 2026-10-13: filmy 20–23, aplikacja, dokładny rozkład względem środka chmury, ujemne jasności, rozdział 5 o gradiencie). Ten plik to skrót zasad; wyprowadzenia i przypisy są w PDF.
Aktualizacja 2026-10-13: poprawione Z5, Z11, Z14, Z15; nowe Z16–Z24; uzupełnione tabele, otwarte testy i literatura.

Wszystkie liczby: `przelicz_zasady_s47.py` → `PRZELICZENIE_ZASADY_S47.json` oraz `opis_ukladu/pomiary_opisu_s47.py` → `opis_ukladu/POMIARY_OPISU_S47.json` i `opis_ukladu/KONTROLA_WZOROW.txt`, bieg `assets/MalyViT_35000_od_zera_s47`, z którego pochodzą zapisane aktywacje. Wyjątek: wyniki FDT i Koopmana w Z24 pochodzą z osobnych eksperymentów z 2026-10-13.
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
| siły między cząstkami | uwaga (3 głowy) | Δx_j = Σ_k A_jk (V x_k); dokładnie f_jk = Σ_h A^h_jk W_O^h W_V^h LN₁(x_k) (Z19) |
| wewnętrzna sfera (jedyna prawdziwa) | LayerNorm przed MLP | sfera o promieniu √192 rozciągnięta przez γ (Z17) |
| ruch laserów | uczenie: AdamW zmienia w_c, v_c, b_c | w_c(t), v_c(t) |
| moment siły na igle reflektora | gradient po v_c | ∂L/∂v_c = Σ_j z_jc δ_j (Z22) |

## 3. Zasady

**Z1. Cząstka ma stan tylko wzdłuż głębokości.** Token jest prawdziwą cząstką w czasie płynącym przez bloki: jego położenie w bloku l+1 zależy od położenia w bloku l. To ujęcie z prac Geshkovskiego i Rigolleta (tokeny jako układ oddziałujących cząstek, czas = głębokość).

**Z2. Laser ma dwa wektory.** Celownik w_c decyduje, *kiedy* laser świeci; kierunek v_c decyduje, *w którą stronę* pcha. To nie jest jeden promień od lasera do cząstki. W filmach 1–6 kierunek pchnięcia był umownym kierunkiem d_c ze sfery Fibonacciego. Wzór aplikacji p = Σ_c (z_c − μ_c) d_c ma postać fc2, tylko z kierunkami Fibonacciego zamiast prawdziwych v_c.

**Z3. Natężenie zależy od położenia samej cząstki.** z_jc = GELU(w_c · LN(x_j) + b_c): laser świeci tym mocniej, im lepiej cząstka leży na jego linii celowania. To pole sił zależne od położenia. W odróżnieniu od siły gradientowej w pęsecie optycznej nie jest to gradient żadnego potencjału: część wirowa pola jest prawie tak duża jak część potencjalna (Z15).

**Z4. Siły działają raz, przy przejściu przez blok.** Blok MLP to jeden krok: Δx_j = Σ_c z_jc v_c + b₂. Nie ma ciągłego trzymania cząstki w pułapce ani lepkości. Ostatni blok to tylko ostatnia „soczewka”; wcześniej działa 5 bloków, każdy z własnymi 768 laserami i z uwagą.

**Z5. Laser pcha tylko w jedną stronę, więc potrzebne są przeciwlasery.** GELU jest prawie nieujemne (minimum ≈ −0,17), więc laser c pcha mocno tylko w stronę +v_c, a w stronę −v_c może ciągnąć najwyżej słabo. Mocne pchnięcie w stronę −v_c wymaga innego lasera c' z v_c' ≈ −v_c. Słabo ujemnych diod jest jednak dużo i w rachunkach nie wolno ich pomijać.

*Ile jest ujemnych diod.* Z progiem 0,05 (`opis_ukladu/POMIARY_OPISU_S47.json`, część B; PDF pkt 3.6; mediana po tokenach zdjęcia TRAIN 0):

| epoka | 0 | 1 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| diody jasne (z > 0,05) | 45 % | 34 % | 28 % | 24 % | 24 % | 21 % | 19 % |
| diody słabo ujemne (z < −0,05) | 42 % | 53 % | 59 % | 53 % | 44 % | 42 % | 42 % |

Uczenie zapala coraz mniej laserów na token (rzadszy kod), a słabo ujemnych diod jest stale 42–59 %. Wcześniejsza liczba „49 % na starcie, 79 % w 60. epoce” liczy wszystkie diody z z < 0, bez progu, jako odsetek wszystkich 600 × 768 diod (`opis_ukladu/KONTROLA_WZOROW.txt`, skrypt `kontrola_wzorow.py`; 60 % w 1. epoce, 74 % w 20.). Obie liczby są prawdziwe, ale nie wolno ich mieszać: różnica (np. 79 % wobec 42 % w 60. epoce) to prawie ciemne diody z −0,05 < z < 0, których w uczeniu przybywa. Liczby z progiem to mediana po tokenach, liczby bez progu to odsetek z wszystkich diod; przy tak małym rozrzucie między tokenami obie miary prawie się pokrywają.

*Ile siły niosą.* Liczba 9–24 % pochodzi z tego samego pliku kontrolnego. To mediana po tokenach stosunku ‖Σ_c (z⁻_jc − z̄⁻_c) v_c‖ / ‖F_j − F̄‖, gdzie z⁻ = min(z, 0): długość wkładu ujemnych diod w siłę bloku względem średniej po tokenach, podzielona przez długość tej siły. Wynosi 24 % na starcie, 18 % w 1. epoce, 9 % w 20. i 13 % w 60. To zarazem błąd, jaki daje obcięcie z < 0 do zera (w rzucie 3D 10–21 %). PDF (pkt 3.6) nazywa tę liczbę „9–24 % sumy długości sił”. To nieścisłe: to stosunek długości wypadkowych, a nie udział w sumie długości 768 sił (sumę długości opisuje Z18).

*Rysowanie.* Filmy 8–12 obcinały jasności do z ≥ 0 (i rysowały, i liczyły z nich położenia); aplikacja robiła to do 13.10. Od filmu 20, w filmie 23 i w aplikacji ujemne jasności są rysowane na błękitno, w prawdziwej proporcji do dodatnich. Ujemna dioda nie jest ciemna: ten sam laser ciągnie wtedy token słabo w stronę −v_c.

Wypadkowa wzdłuż takiej osi to **różnica natężeń** pary przeciwnych laserów: jeśli laser z lewej świeci mocniej niż jego przeciwnik z prawej, cząstka przesuwa się w prawo; przy równych natężeniach pchnięcia się znoszą. Zmierzone na wagach ostatniego bloku biegu s47 (`PRZELICZENIE_ZASADY_S47.json`, część P; cos najbardziej przeciwnego partnera):

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
- Nie mówić, że koloid „nadąża za pułapką” w czasie uczenia: z_j(t) = f(x_j; θ_t) w każdym kroku, więc nie ma opóźnienia, lepkości ani własnego szumu cząstki. Odwrotny obraz, w którym pułapka nadąża za koloidami, też się nie potwierdził (`NADAZANIE_PULAPKI_S47.md`). W tym samym przedziale ruch koloidu i zmiana pułapki pchają wyjście bloku w tę samą stronę, bo to wspólne zejście po gradiencie. Po odjęciu własnych drgań żadna z nich nie reaguje na drugą z opóźnieniem 100 kroków. Tylko w pierwszych 1000 krokach widać słabą wzajemną kompensację (2–8 %). Właściwy obraz: jeden krok uczenia przestawia naraz pułapki i koloidy, a nikt nikogo nie goni.
- Nie mówić, że krok uczenia to temperatura. W biegach z pokrętłami (RP128, okno 12034–13128) τ_1e wynosi 13, 6 i 3 kroki dla lr 1,25e-4, 5e-4 i 2e-3, czyli τ ∝ η^−0,5 zamiast η^−1, a rozmiar klatki nie rośnie monotonicznie. Prosty model koloidu w dołku z SGD nie zgadza się ilościowo; przy AdamW potrzebne jest osobne równanie (Malladi i in. 2022). Zastrzeżenie: τ_1e to czas zaniku F_s, nie autokorelacja wychylenia. Pomiar FDT z 2026-10-13 (Z24) to potwierdza z innej strony: lr działa jak zegar, a nie jak temperatura; stałą temperaturę ma tylko środek drgań, a nie sama plamka.
- Nie mówić, że korelacje sąsiadów (0,6–0,85) to oddziaływania: wynikają ze wspólnego napędu przez wagi. Synchronizację trzeba wykazać maleniem odległości, nie korelacją przyrostów (Agazzi i in. 2026).
- Pułapka w czasie uczenia istnieje tylko w krajobrazie straty: drgania w poprzek doliny minimów i dryf wzdłuż niej (Pashakhanloo i Koulakov 2023), a nie przy każdym tokenie osobno. „Środek drgań” z aplikacji (Z24, znaczenie b) to średnia krocząca położenia, a nie siła działająca na token. Słowo „pułapka” ma trzy znaczenia i trzeba mówić, o które chodzi (Z24).

**Z10. Zasady rysowania w filmach (od filmu 7).**
- Laser c stoi na sferze w punkcie −P·v̂_c i świeci do środka, więc pcha cząstki w stronę +P·v̂_c. Przeciwlasery same lądują po przeciwnych stronach sfery. P to stały rzut 192D → 3D, zawsze podany w filmie. Trzy główne składowe niosą tylko 3–23 % wariancji kierunków v_c (3 % na starcie, 22–23 % od 20. epoki), więc rzut jest ilustracją, a nie pomiarem.
- Jasność wiązki = z_jc; długość strzałki pchnięcia = z_jc · |P v_c|; strzałka wypadkowa = P · Δx_j.
- Celownik w_c pokazywać osobno (np. stożek lub kolor), bo to inny wektor niż kierunek pchnięcia.
- Kierunki Fibonacciego d_c tylko jako umowny układ ze starszych filmów, z podpisem, że to wybór, a nie pomiar.
- Drgania w czasie uczenia rysować jako drżenie wiązki wejściowej (z wcześniejszych bloków), a wędrówkę laserów jako przyczynę wolnego dryfu.
- Każda nowa animacja to nowy skrypt `film_lasery_N_<opis>.py` i nowe `.mp4`; starych nie przerabiać.
- Zawsze podać, która z trzech umów położenia obowiązuje (Z16).
- Jasności rysować ze znakiem: ujemne diody na błękitno, w prawdziwej proporcji do dodatnich (od filmu 20, film 23, aplikacja). Nie obcinać do z ≥ 0 (Z5).
- Rysowany ruch musi być sumą rysowanych wektorów. Przy położeniach względem środka chmury rysować wycentrowane A_jk − Ā_k i z_jc − z̄_c, bez wyrazów wolnych (Z20). Cząsteczka rusza się tylko wtedy, gdy świecą wiązki podkroku; zmiana skali kamery, obrót kamery i skok między krokami uczenia nie mogą wyglądać jak ruch (film 22).
- Obrót igieł w czasie uczenia rysować strzałkami jak w filmie 23 (Z22): kierunek strzałki jest tylko orientacyjny (rzut 3D), długość pochodzi z pełnych 192 wymiarów.

**Z11. Matryca laserów bloku: jak lasery uczą się.** Jeden blok MLP to jedna matryca 768 laserów (6 matryc, po jednej na blok, z uwagą między nimi), jak reflektor matrycowy w samochodzie. Od Z13 słowo „reflektor” oznacza jeden laser, czyli jeden neuron. Laser c ma celownik w_c, kierunek wyrzutu v_c i próg b_c.
- *Przełączanie laserów dla danej cząstki* (które świecą i jak jasno) to aktywacje z_jc. Dzieje się natychmiast przy każdym tokenie, jak w reflektorze matrycowym reagującym na scenę. To nie jest uczenie.
- *Uczenie to powolne przestawianie samych laserów.* Między krokiem 0 a 65 640 (60. epoka; `PRZELICZENIE_ZASADY_S47.json`, część O): kierunek wyrzutu v_c tego samego lasera ma medianę cos 0,11 (laser przesuwa się po sferze o około 84°), celownik w_c cos 0,20, a długość v_c rośnie 2,9 raza (w_c 1,7 raza). Do 21. epoki (krok 22 974): v_c cos 0,32 (około 71°), w_c cos 0,45, długość v_c ×2,1. W oknach po 5 epok cos kierunku v_c wynosi 0,77 w pierwszych 5 epokach, 0,86–0,91 w epokach 5–15 i 0,93–0,97 później, więc ruch jest płynny i z czasem zwalnia.
- *Matryca obraca się jako całość tylko w małej części.* Miarą jest jeden wspólny obrót R w 192D (Prokrustes): udział = 1 − ‖V(t) − R·V(t₀)‖² / ‖V(t) − V(t₀)‖². Zero to ta sama macierz zmian z kolumnami przetasowanymi między laserami. Liczba zależy od definicji, więc zawsze trzeba podać, którą się liczy. Obie poniższe są prawdziwe:
  - (a) *surowe wektory v_c, kolejne etapy uczenia* (0 → 1400 → 5300 → 11 000 → 22 974 → 43 760 → 65 640; `PRZELICZENIE_ZASADY_S47.json`, część O): wspólny obrót wyjaśnia 10–13 % zmiany, zero 6–12 %. Nadwyżka to 1–3 punkty procentowe do 21. epoki i około 5 punktów w epokach 21–60.
  - (b) *kierunki jednostkowe v̂_c, skumulowane od kroku 0* (PDF pkt 3.14; `opis_ukladu/POMIARY_OPISU_S47.json`, część A_wagi): 14 (12), 14 (12), 15 (12), 17 (11), 21 (11) i 24 (11) % dla epok 1, 5, 10, 20, 40 i 60; w nawiasie zero. Nadwyżka rośnie do około 13 punktów w 60. epoce.
  - Cała droga 0 → 60 (część O): kierunki v̂_c 24,3 % wobec 11,5 % zera; kierunki ŵ_c 25,8 % wobec 11,7 %; surowe w_c 21,1 % wobec 11,6 %; surowe v_c 13,2 % wobec 8,8 %.
  Różnica między (a) i (b) wynika z dwóch rzeczy: (a) mierzy krótkie etapy, (b) obrót skumulowany od startu; w (a) do zmiany wlicza się też wzrost długości wektorów, którego obrót nie wyjaśnia. Po stronie celowników w_c wspólny obrót jest wyraźniejszy: na całej drodze 0 → 60 wyjaśnia 21 % zmiany surowych wektorów wobec 12 % zera, a na kierunkach jednostkowych w późnych oknach 5-epokowych 24 % wobec 10 % (PDF podaje dla okien po 40. epoce 22–24 % wobec 9–10 %). Nie jest to nadążanie za strumieniem (`NADAZANIE_PULAPKI_S47.md`): strumień rezydualny obraca się co epokę jako całość (około jednej trzeciej zmiany), ale jego obrót przyłożony do celowników działa tak źle jak obrót wokół losowych osi. Wspólny obrót celowników zachodzi głównie w kierunkach, których strumień prawie nie używa, a tę część wchłania próg b_c. Co najmniej trzy czwarte zmiany to jednak osobne przestawianie każdego lasera, w tym tworzenie par przeciwnych (Z5).
- *Wcześniejsze bloki nie obracają ostatniej matrycy.* To poprzednie matryce w łańcuchu: przesuwają cząstkę, zanim dotrze do ostatniej. W czasie uczenia dają 84–98 % szybkich drgań (Z8).

**Z12. 600 reflektorów za szklaną sferą (obraz „lentykularny”, 2026-10-12).** Każdy token ma swój reflektor: piksel z 768 kanałami kierunkowymi zamiast 3 kanałów RGB. Wiązka kanału c wychodzi z reflektora j pod swoim kątem, trafia w szklaną sferę, załamuje się i dochodzi do cząsteczki j; wektor siły tego kanału to z_jc · v_c. Żeby obraz był zgodny z siecią:
1. *Reflektory są identyczne* (wagi MLP wspólne dla tokenów): kanał c ma to samo v_c (ustawienie galw) i ten sam celownik w_c we wszystkich 600 reflektorach; różnią się tylko jasności z_jc. W czasie uczenia galwa kanału c obracają się naraz we wszystkich reflektorach. Jak ekran lentykularny: jedna optyka, różne obrazy dla różnych odbiorców.
2. *Sfera działa telecentrycznie*: po załamaniu wiązki kanału c dochodzą do każdej cząsteczki równolegle, w kierunku v_c, niezależnie od położenia reflektora. Punkt wejścia wiązki c → j to przecięcie sfery z prostą przez cząsteczkę j o kierunku v_c.
3. *Reflektor j oświetla tylko cząsteczkę j*: MLP nie miesza tokenów. Mieszanie to uwaga, osobny element (można go pokazać jako kontrolowany „przeciek” j → k z wagą A_jk).
4. *Reflektor ma czujnik swojej cząsteczki*: jasność z_jc = GELU(w_c · LN(x_j) + b_c) zależy od bieżącego położenia cząsteczki (jak śledzenie oczu w ekranie lentykularnym).
5. *To nie jest RGB*: 768 kierunków w przestrzeni 192D jest nadmiarowe i nieortogonalne (pary przeciwne, Z5).
Jeden taki stopień = jeden blok; pełna sieć to 6 stopni z uwagą między nimi. Film 10 (`film_lasery_10_reflektory_sfera.py`): położenie cząsteczki = P·W₂·(z_j − z̄), czyli rzut sumy jej 768 wektorów, więc wiązki i położenia są spójne.
Uwaga do intuicji: tak liczone położenie (filmy 8–12 i 23; Z16, umowa 1) to przesunięcie, które nadaje blok (F_j − F̄), a nie stan tokenu x_j. Oba obrazy są podobne (korelacja odległości między tokenami 0,94 na starcie i 0,86 w 60. epoce; `opis_ukladu/KONTROLA_WZOROW.txt`), ale do śledzenia ruchu tokenu właściwym położeniem jest x_j po LayerNorm (tak liczą filmy 17–19 i film 22), a suma wiązek to strzałka przesunięcia.

**Z13. Poprawka do Z12: 768 reflektorów po 600 diod (2026-10-12).** Reflektor to neuron (kanał), nie token. Reflektor c stoi za sferą w punkcie −RR · d_c i ma 600 diod, po jednej na cząsteczkę; dioda j świeci z jasnością z_jc. Wspólne wagi wynikają wtedy z budowy: reflektor ma jedno ustawienie (kierunek d_c) dla wszystkich cząsteczek, a w czasie uczenia obraca się (wędruje po sferze). Po załamaniu 600 wiązek reflektora biegnie równolegle w kierunku d_c, każda do swojej cząsteczki. Film 10 ma te same 460 800 wiązek (te same kierunki i jasności), ale grupuje je po tokenach; obowiązuje układ z filmu 11 (`film_lasery_11_reflektory_kanalow.py`). 600 diod reflektora c na siatce obrazu to mapa aktywacji neuronu c.

**Z14. Stałe reflektory, wiązki sterowane galwami, sfera uczona (2026-10-12, film 12).** Wariant równoważny Z13, lepszy do prezentacji: reflektory stoją na stałe na sferze Fibonacciego (układ z aplikacji), każda z 600 wiązek reflektora c ma własne galwo, które celuje w swoją cząsteczkę (szybkie śledzenie, bez znaczenia dla siły). Sfera załamuje wiązkę raz, na swojej powierzchni, w punkcie E_jc = p_j − s · d_c; w środku wiązka biegnie prosto do cząsteczki w kierunku d_c, wspólnym dla wszystkich 600 wiązek reflektora. Galwo celuje w E_jc. Miejsca reflektorów: 768 punktów spirali Fibonacciego, kanał c dostaje miejsce najbliżej za swoim kierunkiem −d_c w ostatnim kroku filmu (przypisanie węgierskie), żeby załamanie było małe. Film 12 liczono na wagach biegu gęstego (kroki 0–22 974): średni cos miejsca 0,975, a ważony jasnością średni kąt załamania wynosi około 107° na starcie, 88° w kroku 1400 i 33° w kroku 22 974. Na wagach s47, z przypisaniem do kierunków z 60. epoki: średni cos miejsca 0,99 (dokładnie 0,9865; przypis „sfera” w PDF podaje 0,98, to błąd zaokrąglenia), kąt 108° na starcie, 68° w kroku 22 974 i 27° w 60. epoce (część F przeliczenia). Spadek kąta wynika z tego, że miejsca dobrano do kierunków końcowych, i nie jest odkryciem. Warunek z sieci: galwa mogą ruszać się niezależnie tylko w celowaniu; kierunek wiązki w sferze musi być wspólny (inaczej każdy token miałby własne wagi). W czasie uczenia reflektory stoją, punkty i kąty załamania się zmieniają: w sferze „siedzą” wagi v_c.

**Z15. „Pułapka” bloku to siodło z wirem (dokładny Jakobian, bieg s47).** Wokół tokenu blok MLP działa jak pole sił F(x_j + δ) ≈ F(x_j) + J_j δ, gdzie J_j = W₂ diag(GELU′(a_j)) W₁ J_LN(x_j). Wzór jest dokładny: przejście modelu odtwarza zapisane aktywacje z R² ≥ 0,99993, F ze wzoru zgadza się z wyjściem modelu z błędem względnym ≤ 5·10⁻⁶, a J ze wzoru zgadza się z Jakobianem policzonym automatycznie przez PyTorch (autograd, float64) z błędem względnym ≤ 10⁻¹⁵. Mediany po 600 tokenach zdjęcia TRAIN 0 między epoką 0 a 60 (część J przeliczenia):
- ujemnych wartości własnych części symetrycznej S jest stale około 51 %: połowa kierunków ściąga, połowa wypycha;
- symetryczność ‖S‖²/‖J‖² rośnie z 0,50 do 0,58 (macierz losowa: 0,503), a wir ‖A‖/‖S‖ spada z 0,99 do 0,86 (macierz losowa: 0,995);
- najsilniejszy kierunek ściągający rośnie z −0,30 do −1,02, a najsilniejszy wypychający z 0,30 do 0,60; strona ściągająca rośnie więc mocniej i stąd ujemny ślad (Z6);
- norma ‖J‖₂ rośnie z 0,47 do 1,39, czyli około 3 razy (szczyt 1,64 w 20. epoce); dodatnia część rzeczywista wartości własnych J zostaje przez całe uczenie;
- mapa jednego kroku bloku I + J (PDF tabela 4.4): około połowy modów ma |1 + λ| > 1 (51 % na starcie, 46 % w 60. epoce), a promień spektralny ρ(I + J) wynosi 1,19 na starcie, 1,14 w 1. epoce, 1,33 w 10. i 1,25 w 60., czyli około 1,2–1,3. **Mapa bloku nie ściska** (PDF pkt 4.5, punkt 5): sam MLP ostatniego bloku nie jest pułapką w sensie dyskretnej dynamiki; wzmacnia część różnic, inne tłumi. Objętość maleje (Z6), ale nie każdy kierunek;
- w 60. epoce osobno (narzędzie, 20 tokenów / tło, 580 / token 316): wir ‖A‖/‖S‖ 0,93 / 0,86 / 0,96; ‖J‖₂ 1,72 / 1,39 / 1,38; ślad/192 −0,0139 / −0,0163 / −0,0072. Wir słabnie najbardziej w tle; token 316 zostaje blisko poziomu losowego.

Dla porównania: w pęsecie optycznej część wirowa to około 1–4 % sztywności (Pesce i in. 2009), a wyuczone autoenkodery są symetryczne w 95–99 % (Im i in. 2016). W analizie pól FFN dużych modeli językowych większość przybliżonych punktów krytycznych to też siodła (Mudarisov i in. 2026). Słowo „pułapka” jest więc tylko skrótem: to siodło z silnym wirem, które uczenie lekko ściąga. Pełna tabela: `matematyka_reflektorow/matematyka_reflektorow.pdf`, rozdział 4, oraz PDF opisu, pkt 4.4. Wzór na wir jednego lasera: Z23.

**Z16. Trzy umowy położenia cząsteczki na ekranie (PDF pkt 3.2).** We wszystkich P ∈ R^(3×192) to rzut liniowy, a liczby liczy się w 192 wymiarach. Przy każdym filmie trzeba podać, która umowa obowiązuje.
1. *Filmy 8–12 i 23:* p_j = s · P · W₂ (z_j − z̄), P = trzy główne składowe kierunków v̂_c z 60. epoki, z̄ = średnia po tokenach. To rzut przesunięcia od bloku (sumy sił bez b₂), a nie stanu tokenu (Z12). Filmy 8–12 liczyły go z jasności obciętych do z ≥ 0, film 23 z jasności ze znakiem.
2. *Aplikacja (dane wersji ≥ 2) i film 20:* p_j = s · P (x_j − x̄), rzut stanu tokenu tuż przed MLP bloku 6. Przesunięcie cząsteczki przez MLP to wtedy dokładnie Σ_c (z_jc − z̄_c) P v_c, czyli suma wiązek względem środka chmury (Z20).
3. *Film 22:* p_j = s · P (LN̂(x_j) − średnia LN̂ po tokenach), gdzie LN̂ to LayerNorm bez γ i β, liczony dla każdego tokenu osobno. To kierunek tokenu, czyli dokładnie to, co czyta każdy blok (Z17); długość strumienia pokazuje osobny licznik ‖x‖. W pierwszym rzędzie zmiana LN̂ wywołana przesunięciem Δx to część P₁Δx prostopadła do kierunku tokenu, podzielona przez σ.
Trzy składowe niosą tylko 3–22 % wariancji kierunków v̂_c, więc obraz 3D jest ilustracją.

**Z17. Jedyna prawdziwa sfera to LayerNorm (PDF pkt 3.3 i 3.5).** LN(x) = γ ⊙ (x − μ1)/σ + β, σ = √(‖x − μ1‖²/192 + ε), ε = 10⁻⁶. LayerNorm rzutuje token na hiperpłaszczyznę prostopadłą do 1, normuje go na sferę o promieniu √192 ≈ 13,9 i rozciąga przez γ do hiperelipsoidy (Riechers 2024). Jakobian LN zeruje kierunek 1 dokładnie, a kierunek radialny z dokładnością do ε. Reflektory widzą więc tylko kierunek tokenu na tej sferze, a nie jego długość. Anizotropia elipsoidy rośnie w uczeniu: γ ∈ [1; 1] na starcie, γ ∈ [0,20; 1,65] (mediana 1,19) w 60. epoce. Rozpisując LN, pobudzenie neuronu to a_jc = (γ ⊙ w_c)ᵀ(x_j − μ_j 1)/σ_j + (w_cᵀβ + b_c). Rzeczywisty celownik w przestrzeni tokenu to więc czujnik s_c = γ ⊙ w_c (po odjęciu średniej), a próg zapłonu to −(w_cᵀβ + b_c). Tej sfery filmy nie rysują jako sfery (film 22 pokazuje tylko położenia po LN̂); szklane sfery z filmów 4–12 są umowne.

**Z18. Siły w ognisku w większości się znoszą (PDF pkt 3.8).** Mediany po tokenach zdjęcia TRAIN 0 (`POMIARY_OPISU_S47.json`, część B):

| epoka | 0 | 1 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| ‖Σ_c z_c v_c‖ / Σ_c \|z_c\| ‖v_c‖ (wypadkowa / suma długości) | 0,054 | 0,109 | 0,216 | 0,217 | 0,198 | 0,147 | 0,144 |
| ‖F(x_j)‖ / ‖x_j‖ | 0,19 | 0,13 | 0,30 | 0,30 | 0,35 | 0,28 | 0,23 |

Z sumy długości 768 sił zostaje jako wypadkowa tylko 5–22 %. Na starcie to poziom sumy losowych kierunków (dla 768 losowych wektorów około 0,045). W 5.–10. epoce siły układają się spójniej, a późniejszy spadek do 0,14 zbiega się z przyrostem przeciwlaserów (Z5). Dla tokenu 316: 5,6 % w kroku 0, 12,8 % w 1. epoce i 16,8 % w 60. Mediana normy ‖x_j‖ tuż przed MLP bloku 6 rośnie z 14,7 (krok 0) do 90,5 (60. epoka), bo każdy blok dodaje swoje wyjście do strumienia rezydualnego, a LayerNorm normuje tylko wejście podwarstwy. Odległości między tokenami trzeba więc porównywać po LayerNorm (Z21).

**Z19. Uwaga to wiązki między cząsteczkami; pełna sieć to łańcuch 6 stopni (PDF pkt 3.15–3.16, dokładne).** Cząsteczka j dostaje od cząsteczki k wektor f_jk = Σ_h A^h_jk W_O^h W_V^h LN₁(x_k) (3 głowy), a jej przesunięcie w podkroku uwagi to Σ_k f_jk + b_O. To jedyne prawdziwe oddziaływanie cząstka–cząstka w sieci (Z7); w ViT uwaga działa też jak filtr dolnoprzepustowy, który wygładza różnice między tokenami (Wang i in. 2022). Każdy blok to najpierw uwaga, potem MLP z 768 własnymi reflektorami, więc jedno przejście ma 12 podkroków i 13 stanów strumienia rezydualnego (w obrazie Lu i in. 2019: krok Eulera z rozszczepieniem Lie–Trottera). Pokazują to filmy 20–22 (linie j–k o jasności A_jk, dwie najsilniejsze na głowę) i aplikacja (schemat sieci, przycisk podkroków, okienko „Przejście przez sieć” z liniami uwagi i wektorami f_jk wybranej cząsteczki). W filmie 21 każdy podkrok „świeci, przesuwa, gaśnie”; w filmie 22 cząsteczki ruszają się tylko wtedy, gdy świecą wiązki podkroku (położenia z Z16 punkt 3, jedna skala kamery na krok uczenia, kamera stoi w czasie przejścia). Kontrola w filmie 20: przesunięcie w podkroku uwagi odtworzone z A i przesunięcie w podkroku MLP odtworzone z z zgadzają się z przejściem modelu z błędem względnym ≤ 10⁻⁶.

**Z20. Rysowany ruch jest sumą rysowanych wektorów: dokładny rozkład względem środka chmury (PDF pkt 3.17).** Ekran pokazuje położenia względem średniej po tokenach, więc wiązki trzeba wycentrować tak samo. Odejmując średnią po j:
- uwaga: Δx_j − Δx̄ = Σ_{k,h} (A^h_jk − Ā^h_k) W_O^h W_V^h LN₁(x_k), gdzie Ā^h_k = (1/N) Σ_j A^h_jk;
- MLP: Δx_j − Δx̄ = Σ_c (z_jc − z̄_c) v_c, gdzie z̄_c = (1/N) Σ_j z_jc.
Wyrazy wolne b_O i b₂ znikają. To tożsamość algebraiczna. Aplikacja ma trzy układy: surowy (z wyrazami wolnymi), względem środka i względem środka w jednej skali na checkpoint (to nie jest LayerNorm: jedna liczba dla wszystkich tokenów i 13 stanów). W każdym suma rysowanych wektorów równa się rysowanemu ruchowi z błędem względnym około 10⁻⁵ (bieg s47, kroki 0, 10 940 i 65 640). Przed poprawką (aplikacja przed 13.10) przy wycentrowanych położeniach rysowano surowe A_jk, z_jc i wyrazy wolne: błąd wynosił 0,3–3,8, a w uwadze na starcie do 176. Dawny układ „jak LayerNorm” dzielił każdy stan przez jego własną średnią normę, przez co w podkrokach 2–137 % pokazanego ruchu było samą zmianą skali.

**Z21. Ostatni blok rozsuwa tokeny wewnątrz grup (PDF pkt 4.7, pomiar).** Dla pary tokenów (k, l) mnożnik odległości widzianej przez następną warstwę to r_kl = ‖LN̂(x_k + F_k) − LN̂(x_l + F_l)‖ / ‖LN̂(x_k) − LN̂(x_l)‖; r < 1 to przyciąganie, r > 1 odpychanie. Mediana mnożnika po wszystkich parach, w nawiasie odsetek par przyciąganych (`POMIARY_OPISU_S47.json`, część D):

| krok (epoka) | narzędzie–narzędzie (190 par) | narzędzie–tło (11 600) | tło–tło (167 910) |
|---|---|---|---|
| 0 (0) | ×1,010 (14 %) | ×1,003 (41 %) | ×1,006 (34 %) |
| 1094 (1) | ×1,021 (27 %) | ×1,004 (41 %) | ×1,035 (20 %) |
| 10 940 (10) | ×1,021 (29 %) | ×1,053 (1,5 %) | ×1,082 (0,8 %) |
| 65 640 (60) | ×1,084 (26 %) | ×1,011 (25 %) | ×1,100 (0,3 %) |

W 60. epoce blok rozsuwa tło–tło o 10 % i narzędzie–narzędzie o 8 %, a odległości narzędzie–tło prawie nie zmienia (o 1 %; w 10. epoce jeszcze o 5 %). Względem samego tokenu 316: tokeny narzędzia ×1,098 (żaden nie jest przyciągany), tło ×1,020 (5 % przyciąganych). Zdanie „blok rozsuwa narzędzie mocniej niż tło” jest prawdziwe tylko względem tokenu 316, a nie dla wszystkich par. Wynik zgadza się z obrazem, w którym FFN przywraca zróżnicowanie tokenów zmniejszane przez uwagę (Wang i in. 2022, Mudarisov i in. 2026); wkładu uwagi nie mierzono.

**Z22. Gradient = obrót igieł reflektorów (PDF rozdział 5).** W czasie głębokości gradientu nie ma: siły liczy przejście w przód. W czasie uczenia gradient obraca i wydłuża igły v_c i przestawia celowniki w_c.
- *Poprawka z głowy.* δ_j = ∂L/∂x_j⁺ to gradient straty po stanie tokenu za blokiem 6: „w którą stronę przesunąć ten token, żeby strata spadła”. Wstecz przez blok 6 przechodzi jako (I + J_j)ᵀ δ_j, dalej przez uwagę do wszystkich tokenów i do bloków 5, 4, …, 1.
- *Moment siły na pokrętłach reflektora (dokładne; suma po 32 × 600 cząsteczkach paczki, z czynnikiem 1/32 średniej):* ∂L/∂v_c = Σ_j z_jc δ_j; g_jc = ∂L/∂a_jc = GELU′(a_jc) v_cᵀ δ_j; ∂L/∂w_c = Σ_j g_jc LN₂(x̃_j); ∂L/∂b_c = Σ_j g_jc. Igłę v_c ciągną poprawki −δ_j cząsteczek, na które reflektor świeci, ważone jasnością diody (ujemna dioda ciągnie w przeciwną stronę). Celownik w_c obraca się w stronę tokenów, którym pchnięcie wzdłuż v_c by pomogło (v_cᵀ δ_j < 0), i od tokenów, którym szkodzi. Zdjęcie TRAIN 0 to tylko jeden z 32 obrazów paczki.
- *Od gradientu do kroku: AdamW* (`trening.py`; Kingma i Ba 2015, Loshchilov i Hutter 2019). Obcięcie normy całego gradientu do 1, potem AdamW z β = (0,9; 0,999), krokiem η_k = 5·10⁻⁴ · min(1, (k + 1)/300) (rozgrzewka 300 kroków, potem stały) i rozpadem wag λ = 0,05 tylko dla tensorów o wymiarze ≥ 2 (wagi fc1, fc2, uwagi, konwolucji; bez biasów i parametrów LayerNorm). Skutki dla igieł:
  1. *Sprężyna nie obraca.* Rozpad wag mnoży całą kolumnę v_c przez 1 − ηλ = 1 − 2,5·10⁻⁵: skraca igłę, ale nie zmienia jej kierunku. Obrót pochodzi wyłącznie z członu gradientowego.
  2. *Adam zmienia kierunek.* Dzielenie przez √ŝ jest osobne dla każdej współrzędnej, więc krok nie jest równoległy do −∂L/∂v_c. Pierwszy moment uśrednia około 10 kroków (bezwładność), drugi około 1000 (hamulec).
  3. *Obcięcie działało zawsze.* Norma gradientu przekraczała 1 w 100 % paczek od kroku 0 do 60. epoki (mediana 9,8 na starcie, 2,4 w 60. epoce; `obciecie_gradientu/NORMA_GRADIENTU.md`). To prawie stały mnożnik, który Adam znosi, bo normuje skalę.
  4. *Stały krok.* Po rozgrzewce η się nie zmienia, a znormowany krok Adama ma na każdej współrzędnej rząd η. Długość kroku sama z siebie nie maleje.
- *Czego nie da się zmierzyć.* Checkpointy biegu s47 zawierają tylko wagi (`net`), bez stanu AdamW. Dlatego mierzy się rzeczywisty obrót między checkpointami, a nie sam gradient ani m̂.
- *Pomiar (PDF tabela 5.4).* Kąt między v̂_c(t) a v̂_c(t + Δ) w pełnych 192 wymiarach, mediana po 768 laserach, w stopniach na 100 kroków. „Drganie” z checkpointów odległych o ≥ 100 kroków, „dryf netto” z odległych o ≥ 2000 kroków (kąt podzielony przez liczbę setek kroków):

| krok (epoka) | 0 | 300 (0,3) | 1094 (1) | 21 880 (20) | 65 640 (60) |
|---|---|---|---|---|---|
| drganie (≥ 100 kroków) | 2,88 | 3,75 | 3,56 | 3,53 | 2,79 |
| dryf netto (≥ 2000 kroków) | 1,14 | 1,15 | 1,04 | 0,63 | 0,52 |

- *Wykładnik i obrót całkowity.* Wykładnik α w zależności kąt ∝ Δ^α (z Δ = 300 i 3000): 0,60 od kroku 300, 0,45 od 5. epoki, 0,36 od 20., 0,38 od 40. Mediana obrotu igły od kroku 0 do 60. epoki to 84° (zgodnie z cos 0,11 w Z11), z czego 19 % w pierwszej epoce; celowniki w_c obracają się o 78°. Kontrola danych: wyjście MLP = Σ_c z_c v_c + b₂ z błędem względnym 5·10⁻⁸.
- *Odczyt.* Krótki ruch to głównie drganie: przy samym dryfie igła obróciłaby się w 100 krokach o 0,5–1°, a obraca się o około 4°, bo kolejne paczki ciągną ją w różne strony. Drganie ma prawie stałą wielkość, 2,8–3,8° na 100 kroków przez całe uczenie, zgodnie ze stałym krokiem. Dryf netto zwalnia o połowę między 1. a 60. epoką, a α spada do około 0,37, czyli poniżej 0,5: wolniej niż zwykłe błądzenie losowe. Nazwa „starzenie” (jak w szkle) to analogia, nie dowód. Pierwsza wersja filmu 23 zakładała, że „igły z czasem zwalniają”; zwalnia tylko dryf netto.
- *Rysowanie.* Film 23 (`film_lasery_23_obrot_reflektorow.py` → `lasery_23_obrot_reflektorow.mp4`) i aplikacja (dane wersji 3, `lasery_dane.obroty`, shader sfery w trybie 11, okienko 10 z medianą drgania i dryfu w czasie). Igła to odcinek od reflektora w kierunku d_c. Strzałka obrotu to część zmiany d_c prostopadła do igły (rzut 3D, tylko orientacyjny: 3 składowe niosą 22 % wariancji v̂_c); jej długość to prędkość kątowa w 192 wymiarach w skali logarytmicznej. Gruba strzałka w skali plasma to dryf netto, cienka biała kreska to drganie.

**Z23. Wir jednego lasera (wyprowadzenie 2026-10-13, dokładne algebraicznie).** Jeden laser wnosi do Jakobianu pułapki (Z15) macierz rzędu 1: M = v sᵀ, gdzie v = GELU′(a_c) v_c, a s = J_LNᵀ w_c (w przybliżeniu γ ⊙ w_c / σ po odjęciu średniej, Z17). Niech κ = cos(s, v). Wtedy:
- ‖S‖_F² = ½(|v|²|s|² + (v·s)²) oraz ‖A‖_F² = ½(|v|²|s|² − (v·s)²), więc ‖A‖²/‖S‖² = (1 − κ²)/(1 + κ²);
- niezerowe wartości własne części symetrycznej S to ½(v·s ± |v||s|).
Wnioski:
- *Laser prostopadły (κ = 0)* daje po równo siodła i wiru: jedna wartość własna S dodatnia, jedna ujemna (±½|v||s|), a ‖A‖ = ‖S‖.
- *Laser przeciwny (κ = −1)* daje czystą sprężynę zwrotną: A = 0 i jedna wartość własna −|v||s|. Laser zgodny (κ = +1) daje czyste wypychanie.
- *Start uczenia.* Kierunki są losowe w 192D, więc κ ≈ 0 (rozrzut około 1/√192 ≈ 0,07). Każdy laser to prawie czyste siodło z wirem, a suma 768 takich wkładów ma poziom macierzy losowej: siodło 50/50 i ‖A‖/‖S‖ ≈ 0,995 (zmierzone 50 % i 0,993).
- *60. epoka.* Mediana κ ≈ −0,18 (Z6). Dla jednego lasera daje to ‖A‖/‖S‖ = √((1 − 0,0324)/(1 + 0,0324)) ≈ 0,97. Zmierzone 0,86 jest wyraźnie niższe. Reszta musi pochodzić ze zgrania wielu laserów, czyli z wyrazów krzyżowych między ich macierzami (na przykład par przeciwlaserów, Z5). To hipoteza, niezmierzona (otwarty test 16).

**Z24. Trzy znaczenia słów „pułapka” i „temperatura” (2026-10-13).** Słowa „pułapka” nie wolno używać bez doprecyzowania, o który obiekt chodzi:
- (a) *Pułapka laserowa bloku* = pole sił F i jego Jakobian J wokół tokenu (Z15, Z23). Czas to głębokość, szumu nie ma, więc temperatura wynosi 0. To siodło z wirem, a nie studnia.
- (b) *Środek drgań* = średnia krocząca położenia plamki z ±300 kroków uczenia. Tak liczy aplikacja oraz eksperymenty `experiments/koopman_pulapki_20261013` i `experiments/fdt_temperatura_20261013`. Obiekt to plamka ogniskowa w czasie uczenia (Z8), a nie koloid z własnym stanem.
- (c) *Dolina w krajobrazie straty* (Z9): pułapka w przestrzeni wag.

Wyniki FDT (eksperyment `experiments/fdt_temperatura_20261013`; wynik z innej sesji, liczby nieprzeliczane w tym pliku; repliki uczenia z identycznymi paczkami i małym polem testowym na sondach):
- Środek drgań (b) spełnia relację Einsteina (fluktuacja–odpowiedź): T ≈ 3,6·10⁻⁴ (podprzestrzeń N) i 5,7·10⁻⁴ (podprzestrzeń V), stałe od kroku 300 do 7000 i niezależne od lr.
- Sama plamka (latent z_j, 768 wymiarów) nie ma jednej temperatury: T_eff spada mniej więcej jak 1/t, a odpowiedź na stałą siłę rośnie bez nasycenia. Jedyną sprężyną jest weight decay: przy wd = 1,0 odpowiedź się nasyca, a T jest stałe. lr działa jak zegar, nie jak temperatura. Kierunki widoczne dla fc2 są 2–3 razy „gorętsze” od pozostałych.
- Koopman na środkach drgań: lokalna wirowość ich ruchu nie odróżnia się od surogatu, więc jest artefaktem; realny jest tylko obrót całej chmury, 2–3 razy większy niż w surogacie. Koopman nie przewiduje ruchu lepiej niż model „stoi”.
Nie mylić wiru z (a), czyli wiru Jakobianu bloku w czasie głębokości, z obrotem chmury z (b), czyli ruchem w czasie uczenia. Relacje fluktuacja–dyssypacja dla SGD (Yaida 2018) dotyczą parametrów, a nie reprezentacji; ten pomiar jest więc nową, jeszcze niezweryfikowaną metodą.

## 4. Wierność fizyczna w skrócie

| Element | Status |
|---|---|
| 768 sił działających na token w bloku MLP, natężenie zależne od położenia | **dokładne** (wzór bloku) |
| przeciwlasery i różnica natężeń wzdłuż osi | **dokładne** jako opis; pary przybliżone, tworzy je uczenie |
| siła zwrotna lasera | **słaba, zmierzona** (cos(γ ⊙ w_c, v_c) = −0,18; ślad J / 192 = −0,016 w 60. epoce) |
| „pułapka” bloku MLP (Jakobian pola) | **dokładne** jako wzór; zmierzona: siodło z wirem, ‖A‖/‖S‖ = 0,99 → 0,86 |
| mapa jednego kroku I + J | **pomiar**: ρ ≈ 1,2–1,3, 46–51 % modów rośnie; mapa nie ściska (Z15) |
| wir jednego lasera, ‖A‖²/‖S‖² = (1 − κ²)/(1 + κ²) | **dokładne** algebraicznie; tłumaczenie spadku do 0,86 przez zgranie laserów to **hipoteza** (Z23) |
| uwaga jako siły między cząstkami | **dokładne** wzdłuż głębokości |
| uwaga jako wiązki między cząsteczkami, łańcuch 6 bloków (filmy 20–22) | **dokładne**; kontrola z przejściem modelu ≤ 10⁻⁶ (Z19) |
| rozkład wycentrowany (A_jk − Ā_k, z_jc − z̄_c, bez wyrazów wolnych) | **dokładne**; suma rysowanych wektorów = rysowany ruch z błędem ≈ 10⁻⁵ (Z20) |
| LayerNorm jako sfera odczytu | **dokładne**; γ ∈ [0,20; 1,65] w 60. epoce (Z17) |
| lasery jako ruchome cząstki w czasie uczenia | **dokładne** (to są parametry) |
| gradient = obrót igieł reflektorów | **dokładne** jako wzór; **pomiar**: drganie ≈ 3,5°/100 kroków, dryf netto 1,04 → 0,52°/100 kroków (Z22) |
| koloid nadążający za pułapką w czasie uczenia | **błędne** (brak własnego stanu tokenu) |
| krok uczenia = temperatura | **niepotwierdzone**, ilościowo się nie zgadza; FDT: lr działa jak zegar (Z24) |
| temperatura środka drgań (średnia ±300 kroków) | **pomiar FDT** (inna sesja): relacja Einsteina, T ≈ 3,6·10⁻⁴ / 5,7·10⁻⁴, stałe od kroku 300 do 7000 (Z24) |
| temperatura samej plamki (latent z_j) | **brak jednej temperatury**: T_eff ∝ ~1/t, sprężyną jest tylko weight decay (Z24) |
| położenie w 3D (rzut Fibonacciego lub P) | **umowne**; trzy umowy (Z16) |

## 5. Otwarte testy (z lektury 10 prac, 2026-10-12; aktualizacja 2026-10-13 z PDF, rozdział 7)

1. Test RPI (tasowanie patchy przy zachowanym kodowaniu pozycji) i kontrola RPT, miara SSDC, krzywe po warstwach, więcej ziaren bez kodowania pozycji (Mannes 2026).
2. Surogaty rzutu: błądzenie losowe, z pędem (β1 = 0,9) i OU z dopasowaną kowariancją, przepuszczone przez ten sam rzut 3D; kilka zestawów kierunków (Antognini i Sohl-Dickstein 2018).
3. Profile głębokości (rozrzut, efektywny wymiar, podobieństwa cosinusowe, energia oddziaływania po 6 blokach) w każdym z 806 checkpointów: jak uczenie przebudowuje przepływ tokenów przez głębokość (Isobe i in. 2026, Herty i Liu 2026).
4. Jądro lokalnej sprężystości: przesunięcie tokenu i po kroku z jednego źródła k, w funkcji odległości na siatce i treści (He i Su 2019); przy AdamW mierzyć faktyczne przesunięcia.
5. Prawdziwy czas relaksacji z autokorelacji wychylenia od środka klatki oraz krzywizna straty; sprawdzić skalowanie τ ∝ 1/(ηλ) przy AdamW. Częściowo dotyka tego FDT z 2026-10-13 (Z24: lr działa jak zegar, sprężyną jest weight decay), ale sam czas relaksacji nie jest zmierzony.
6. Geometria dryfu: stabilność macierzy token–token, Prokrustes (obrót wobec odkształcenia), dyfuzja kątowa (Qin i in. 2021).
7. Film 7: lasery w punktach −P·v̂_c, przeciwlasery, osobno celowniki w_c.
8. Zrobione, wynik negatywny (`NADAZANIE_PULAPKI_S47.md`): wspólny obrót celowników w_c (Z11) nie jest nadążaniem za obrotem strumienia rezydualnego.
9. Nowe wersje filmów 8–12 na wagach s47 (jako nowe pliki; stare filmy zostają bez zmian). Częściowo zrobione: film 23 to geometria filmu 12 na wagach s47, z jasnościami ze znakiem.
10. Zrobione: uwaga jako wiązki między cząsteczkami i łańcuch 6 bloków w jednym obrazie (filmy 20–22, aplikacja; Z19).
11. Profile pułapki po wszystkich 6 blokach w 806 checkpointach: czy siodło z wirem to cecha ostatniego bloku, czy każdego.
12. Hipoteza gęstej głowy: czy przyrosty odległości między tokenami narzędzia (Z21) korelują z różnicami ich celów (l, t, r, b).
13. Przeciwlasery jako pary cech wejściowych: porównać selektywność w_c i w_c' oraz wzory aktywacji pary; czy to pary cech, czy tylko geometria wyjścia (Shang i in. 2016, Elhage i in. 2022 w przypisie „pary” PDF).
14. Wiązki poprawek δ_j (gradient wstecz od głowy przez 6 bloków) jako obraz; wymaga liczenia gradientu na paczce (Z22).
15. Starzenie igieł: α i dryf netto w biegach z innym lr (`eden/wyniki/pokretla`); do tego krótki bieg z zapisem stanu AdamW, żeby porównać kierunek kroku z kierunkiem gradientu (Z22).
16. Ile obniżenia wiru 0,99 → 0,86 tłumaczą κ pojedynczych laserów (Z23): policzyć wir z samych wkładów diagonalnych (Σ_c ‖A_c‖² wobec Σ_c ‖S_c‖², bez wyrazów krzyżowych) i porównać z pełnym J; osobno udział par przeciwlaserów.
17. Czy szum uczenia daje krążący prąd w płaszczyźnie najsilniejszego wiru A (odpowiednik „fontanny prawdopodobieństwa”, Roichman i in. 2008; różnica korelacji krzyżowych jak u Pesce i in. 2009): rzutować drgania środka drgań (Z24 b) na tę płaszczyznę i porównać średni moment obrotowy z surogatem.

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
- Geva i in. (2021), Transformer Feed-Forward Layers Are Key-Value Memories (klucz = w_c, wartość = v_c; niesprawdzone w tej sesji przez Undermind) — https://doi.org/10.18653/v1/2021.emnlp-main.446

Dopisane 2026-10-13 (z bibliografii `opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.tex`, do zasad Z15–Z24):
- Lu i in. (2019), Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View — https://arxiv.org/abs/1906.02762
- Riechers (2024), Geometry and Dynamics of LayerNorm — https://doi.org/10.48550/arXiv.2405.04134
- Mudarisov, Burtsev, State (2026), Feed-Forward Steering in Transformer Residual Dynamics — https://arxiv.org/abs/2608.02071
- Im, Belghazi, Memisevic (2016), Conservativeness of Untied Auto-Encoders — https://doi.org/10.1609/aaai.v30i1.10268
- Pesce i in. (2009), Quantitative assessment of non-conservative radiation forces in an optical trap — https://doi.org/10.1209/0295-5075/86/38002
- Roichman i in. (2008), Influence of nonconservative optical forces on the dynamics of optically trapped colloidal spheres: the fountain of probability — https://doi.org/10.1103/PhysRevLett.101.128301
- Kingma, Ba (2015), Adam: A Method for Stochastic Optimization — https://arxiv.org/abs/1412.6980
- Loshchilov, Hutter (2019), Decoupled Weight Decay Regularization — https://arxiv.org/abs/1711.05101
- Wang i in. (2022), Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis — https://doi.org/10.48550/arXiv.2203.05962
- Yaida (2018), Fluctuation-dissipation relations for stochastic gradient descent — https://arxiv.org/abs/1810.00004
- Shang i in. (2016), Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units — https://arxiv.org/abs/1603.05201
- Elhage i in. (2022), Toy Models of Superposition — https://doi.org/10.48550/arXiv.2209.10652
