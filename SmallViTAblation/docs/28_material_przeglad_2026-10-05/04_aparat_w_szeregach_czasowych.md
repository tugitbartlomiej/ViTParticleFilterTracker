# Aparat Koopmana–Langevina na latencie TFT: ekstrakt faktów (tylko odczyt)

Przeczytano wszystkie wskazane pliki; niczego nie zmieniono ani nie uruchomiono. Pięć rzeczy trzeba wiedzieć przed lekturą:

- **Wycofana teza.** Syntezy z 2026-09-03 opierają część wniosków na kryterium Pawuli 978 („ruch skokowy, żadne jądro pamięci nie pomoże”). Teza D117, która to niosła, ma od 2026-09-15 status WYCOFANA: pomiar był tylko w rzucie 2D, a równanie z dryfem, pamięcią 28 dni i wejściami przewiduje w 192D 72,75 procent.
- **Brak dokumentów wyników.** Dla `tft_wlasna_pamiec_ze_stanu` i `tft_korekta_pamiec_roznicowa` w katalogach są tylko README i deklaracja `WYNIKI.json`; wyników nie znaleziono.
- **Rejestr tez kończy się na 2026-09-15.** Ostatnie tezy to D120 i D121. Tez o równaniu L1/L2/L3, schemacie diagnostyki, doborze danych ani o detektorze wizyjnym nie znaleziono. Tezy ze słowem ViT (D66, D84, G13) dotyczą enkodera V-JEPA.
- **Zegar w dokumentach pomostowych.** Dokumenty 23 i 25 opisują zegar czasu filmu przy zamrożonym enkoderze, nie zegar uczenia. Zegar uczenia opisuje część C schematu z 2026-09-18, jako projekt bez pomiaru.
- **Źródła spoza listy.** Dla kontekstu doczytałem `tft_destylacja_rekurencji`, `tft_langevin_pamiec_ukryta`, `tft_droga_do_95`, `czastki_i_pole\TEORIA.md`, plik Lean, dwa podsumowania sesji i notatki pamięci.

## A. Aparat

**Obiekty.**
- **Cząstka:** jedna seria sklep–produkt Favority. Pełny zrzut ma 174 685 serii, 85 klatek (2017-04-25 do 2017-07-18) i 192 wymiary.
- **Położenie `z_t`:** latent enkodera TFT dla okna 56 dni kończącego się w dniu `t`. Jest to ostatnia pozycja historii po uwadze z maską przyczynową, więc `z_t` nie zawiera planu promocji ani sprzedaży jutra.
- **Zegar:** czas danych, krok to jeden dzień, wagi sieci zamrożone. Schemat dopuszcza drugi zegar: `z_i(τ) = f_{θ_τ}(x_i)`, gdzie `τ` to krok uczenia.
- **Stan pochodny:** prędkość `v_t = z_t − z_{t−1}` i przyspieszenie `a_t = v_t − v_{t−1}`.
- **Wejścia `u_t`:** kalendarz, plan promocji, nowe obserwacje dnia. W Koopmanie K4 to 74 kanały dnia wchodzącego do okna; w równaniu Langevina 24 kanały wybrane regułą a priori.
- **Historia `H_t = (z_0..z_t, u_0..u_{t+1|t})`:** tylko to, co znane przy wydaniu prognozy.

**Koopman z wejściem.** Enkoder `g = φ(z) ∈ R^r` i dekoder `ψ` to sieci MLP (razem około 800 tys. parametrów). Krok ma postać `g_{t+1} = K g_t + B u_t`. Z pamięcią Mori–Zwanziga:

`g_{t+1} = K g_t + B u_t + Σ_{l=1..L} M_l g_{t−l} + r_{t+1}`

Schemat nazywa to dyskretnym uogólnionym równaniem Langevina i zaznacza, że dla skończonego zestawu współrzędnych „rozkład jest tożsamością, nie założeniem”.

**Równanie Langevina w `z`** (z `tft_langevin_krok_po_kroku\README.md`):

```
z^_{t+1|t} = z_t + D_t + JP_t + W_t + N_t
D_t  = s ⊙ f_D(ẑ_t, v̂_t, â_t)                                  dryf
JP_t = s ⊙ W_o Σ_{l=0}^{55} A_h(t,t−l) V n_{t−l},  n=[ẑ, v̂]     jądro pamięci 56 dni
       A_h ∝ ρ_h^l · (φ(q_t)·φ(k_{t−l}) z obrotem pary kanałów o ω_h·l)
W_t  = s ⊙ f_W(u_{t+1|t}, Σ_{l=0}^{55} K_W(l) x̂_{t−l}, ẑ_t)     wymuszenie
N_t  = s ⊙ U Σ_{l=1}^{55} Σ_m g_m(l) V_m (r0_{t+1−l}/s),  g_m(l)=ρ_m^{l−1} cos(ω_m(l−1))
r0_{t+1} = z_{t+1} − (z_t + D_t + JP_t + W_t)    reszta bazowa (czyta ją N)
e_{t+1}  = z_{t+1} − z^_{t+1|t}                  reszta końcowa (oceniana)
```

- `N` to „szum kolorowy”, czyli przewidywalna część dawnych reszt.
- `f_D` i `f_W` to sieci dwuwarstwowe (1024 i 1536 neuronów); `JP` i `N` to jawne wzory.
- Człony nie dzielą parametrów; `s` to odchylenie kroku z danych uczących.

**Człon zdarzenia `Z`** (z `tft_langevin_czlon_zdarzen\README.md`). Kotwica `a_t` to latent z ostatniej klatki z obecnym wierszem sprzedaży, a `c_t = z_t − a_t`.
- Tryb warunkowy: `Z = [wiersz wraca jutro]·(−A ⊙ c_t) + [wiersz znika jutro]·s ⊙ f_Z(...)`.
- Tryb prognostyczny: `Z = [brak wiersza dziś]·π(g_t)·(−A ⊙ c_t)`.
- Dodatkowo „odczepy reszty”: `N += a1 ⊙ r0_t + a7 ⊙ r0_{t−6}`.

**Dopasowanie współczynników.**
- **Koopman:** kary za odtworzenie, liniowość, prognozę, największy błąd i zanik wag; w K4 dochodzi kara wprost za krok latentu. Epokę wybiera się po walidacji odsuniętej w czasie (klatki 49..55), a wybór zapisuje w `SELEKCJA*.json` przed testem.
- **Pamięć:** w K2 rachunkiem z korelacji (rekurencja Lina, nic nie jest trenowane). W K5 najmniejszymi kwadratami na resztach przy zamrożonych `K`, `B`, z progiem z surogatu Markowa. W K6 trenowana razem z autoenkoderem.
- **Langevin:** cztery człony uczone razem jednym błędem kroku. Normalizacja pochodzi tylko z dni uczenia, uczenie jest wsadowe z maskami przyczynowymi. Podział jest rozłączny po cząstkach i czasie: 72 600 cząstek uczących (cele 20.06..04.07), 9074 wyboru (05..10.07), 9074 testu (11..18.07).

**Poziomy i tryby.**

| Nazwa | Co wie o jutrze | Status |
|---|---|---|
| L1 | plan promocji, kalendarz, święta | uczciwy |
| L3 (`K_prognoza`) | pełne kanały do dziś; jutro: plan promocji, kalendarz, święta; sprzedaż zastąpiona prognozą TFT wydaną w `t`; reszta nieznanych kanałów z dnia `t` | uczciwy („prognostyczny”) |
| L2 (`K_warunek`) | cały dzień jutra, w tym sprzedaż, transakcje, brak wiersza | warunkowy: „reakcja sieci na znany dzień, nie prognoza” |
| K1–K3 | Koopman bez wejść | autonomiczny |
| K4–K6 | wszystkie 74 kanały dnia | warunkowy |
| K7 `jawne` | promocja, kalendarz, święta | uczciwy |
| „Stare równania” R0, T1c, KF, T3 | prawdziwy latent z 28 opóźnieniami | „widziały sprzedaż jutra przez odpowiedzi nauczyciela” |

**Reguła rozdziału** (z `WYNIKI_L2.md`). Są dwa uczciwe zadania: (a) równanie ruchu przy znanym wejściu dnia, (b) prognoza o dzień z informacji dostępnej dziś. Plik mówi: „Błędem byłoby nazwać równanie z jutrzejszą sprzedażą «prognozą»”. Schemat dodaje, że porównywanie trybu warunkowego z prognostycznym „jest błędem kategorii”.

Protokół ma kolejność: uczenie, przewidywanie (zapis z podpisem przed obserwacją), obserwacja i ocena, aktualizacja historii. Na walidacji i teście wagi i normalizacja są zamrożone. Testy akceptacyjne dały: wsad kontra krok 0,0; podmiana przyszłości 0,0; wpływ planu promocji jutra 0,024.

## B. Wyniki w szeregach czasowych

**Miara.** Umiejętność = 100 (1 − SSE/SS), gdzie SS to błąd prognozy „brak ruchu”. Wszystko to jeden krok (jeden dzień) z prawdziwego stanu. „Stary test” to 320 cząstek z panelu, „nowy test” to 9074 cząstki nieużyte w uczeniu; oba na dniach 11..18 lipca.

| Model | Stary test | Nowy test |
|---|---|---|
| L1 | 64,88 | 38,59 |
| L3_uczciwe / L3_uczciwe_szeroki | 78,70 / **79,80** | 47,40 / 48,28 |
| L2_dzien / L2_dzien_szeroki | 87,47 / **90,36** [86,9; 92,7] | 60,07 / 57,52 [43,1; 83,3] |
| K_prognoza_apriori / K_warunek_apriori | 79,0 / 89,8 | 49,1 / 61,2 |
| ridge (L1; L3) | 36,33; 40,13 | 21,50; 24,21 |
| stare równania R0 / KF / T3 | 72,11 / 75,18 / 74,78 | – |

Koopman liczono na 320 seriach przez 28 dni (8960 przejść):

| Model | TEST |
|---|---|
| K4_r64_wej_duze (pełne wejście) | 86,16 |
| K4_r16 bez wejść | 57,48 |
| K4_r16 z wejściami innej serii (placebo) | 52,09 |
| K7 jawne, r = 16 / r = 64 | 57,45 / 54,53 |
| DMDc_r64 / DMDc_192 / ARc7_192 (podłogi liniowe) | 45,02 / 49,27 / 61,28 |

- **Uczciwa prognoza Koopmana:** K7 mówi, że to „około 57 procent, a nie 86”; kanały znane z góry „nie dodają prawie nic ponad sam stan”.
- **Bieg swobodny:** dla K4 r = 64 po 3 dniach 73,46, po 7 dniach −42,98. Po 7 dniach wszystkie ramiona są gorsze od braku ruchu; największy moduł `K` to 1,068.
- **Sufit bez jutra:** modele L2 z prognozą TFT zamiast sprzedaży jutra dają 66,02–67,06 (stary test) i 39,28–42,97 (nowy). Plik podaje: „Około 23 punktów różnicy do 90 to nieprzewidywalna część jutrzejszej sprzedaży”. Luka L3→L2 wynosi 10..12 punktów.
- **Zaskoczenie a długość kroku:** Spearman 0,08 (placebo 0,05). Zaskoczenie zmienia głównie kierunek kroku, nie długość.

**Pułapki miary** (z `WYNIKI_L2.md` i `WYNIKI_DIAGNOZA_SLABYCH.md`).
- Mediana per seria jest w obu testach prawie ta sama (L3: 76,5 stary, 78,3 nowy), mimo wyniku łącznego 79,8 wobec 48,3.
- Jedna seria (id 69372: krok 270 razy mediana, umiejętność 28,8) niesie 37 procent mianownika nowego testu.
- 1 procent cząstek niesie 73–85 procent mianownika.
- Stary test leży na 94. percentylu losowych 320-podzbiorów nowego (mediana 70,0; przedział 29,9..80,0).
- Równanie nie zapamiętuje cząstek: mediana nowych równa się medianie uczących na tych samych dniach (L2: 93,6 wobec 93,7).
- Przesunięcie w czasie kosztuje 3–6 punktów mediany.
- Wniosek zapisany w pliku: raportować medianę per seria i wynik bez 1 procenta największych; „320 serii to za mało na wynik łączny”.

**Co dało jądro pamięci.**
- **L1:** `JP` wnosi 1,3 punktu na teście i 13 na dniach uczenia („pamięć przenosi się słabo na nowe dni”).
- **L3:** wyłączenie `JP` zabiera 5,3 punktu na nowym teście (47,40→42,13) i 1,0 na starym. W wersji szerokiej 3,6 i 0,4.
- **L2 szerokie:** `JP` nic nie wnosi (bez `JP` 58,38 wobec 57,52).
- **Wymuszenie niesie większość wyniku:** bez `W` spadek o 30 do 60 punktów.
- **Koopman K5:** pamięć „mierzalna, ale nie przewiduje”. Normy 0,08–0,18 przy progu 0,007, a każda długość pamięci obniża walidację (82,29 bez pamięci, 79,88–82,01 z pamięcią).
- **K6:** zysk 0,81 punktu, poniżej progu 1 punktu („pamięć nie spełnia kryterium”). Pamięć stabilizuje bieg 7-dniowy (−45,4 → −30,1 / −24,0).
- **Teza D120:** pełne macierze pamięci biją bank tłumionych oscylatorów na teście o 13,5 procent mimo równej walidacji.

**Co dał człon zdarzeń** (z `tft_langevin_czlon_zdarzen\WYNIKI.md`).
- Diagnoza wstępna: 49 procent błędu 1 procenta największych cząstek to dni znikania i powrotu wiersza sprzedaży (4,5 procent komórek).
- Z członem `Z` i odczepami: powrót wiersza u dużych cząstek 42,9→50,5 oraz 39,9→57,1; próg 60 nie został osiągnięty.
- Autokorelacja reszty po dniu spadła z −0,16 do +0,09.
- Wynik łączny na dwóch ziarnach: +5,5 i −4,6. Plik podsumowuje: „zysku w wyniku łącznym nie da się stwierdzić, bo rozrzut między biegami (około 5 punktów) jest większy od efektu”.
- W trybie prognostycznym `Z` szkodzi: 40,7 wobec 44,2 bez `Z`.
- „Biała reszta nie oznacza dużego zysku”: autokorelacja −0,15 daje najwyżej około 2 procent energii reszty.

**Optymalizacja pod 95 (tryb warunkowy).** 18 biegów w trzech rundach; cel nieosiągnięty. Najlepsze pojedyncze biegi: stary test 94,0, nowy 69,9. Pułap: stary 92–94, nowy 66–70, mediana per cząstka 90–93. Pomogły tylko więcej dni uczenia i więcej dużych cząstek. Luka w czasie (około 42 punkty) jest większa niż między cząstkami (około 18); jej przyczyna pozostaje nieznana.

**„Destylacja 95 procent”** (z `tft_destylacja_rekurencji\WYNIKI_TEST.md`).
- Uczeń E3_N56_duze ma 95,106 procent na teście, przy T1c 75,337, T1c+R1 76,638 i podłodze R0 72,750.
- Mechanizm to „kotwica różnicowa”: krok liczony jako różnica dwóch własnych położeń ucznia. Ta sama uwaga daje 79,9 od prawdziwego `z_t` i 95,6 jako różnica.
- Zastrzeżenie z pliku: „To nie jest zwarte jawne równanie. Uczeń ma 696 tys. parametrów […] kopiuje LSTM, GRN i normy nauczyciela”.
- Ocenę zapisano w notatce pamięci: użytkownik 2026-09-16 uznał to za „przepisany TFT, a nie równanie ruchu”. Celem jest jawne równanie z czterema osobnymi, wyłączalnymi członami; wynik destylacji ma służyć „tylko jako sufit informacji”.

**Pętla aktywna** (z `WYNIKI_PETLA.md`). Wagi serii to energia reszty podzielona przez medianę, obcięta do [1, 20].

| Ramię | Nowy test łącznie | Mediana per seria | Bez 1 procenta |
|---|---|---|---|
| L3 bez wag | 48,28 | 78,34 | 72,94 |
| A_aktywne | 48,46 | **65,82** | 69,23 |
| A_losowe (te same wagi przetasowane) | 47,35 | 78,46 | 71,61 |

Koszyk skrajnego ruchu zyskał 1 punkt (45,5→46,5), a zwykłe serie straciły 8 do 19 punktów. Wniosek z pliku: „Selektor powinien wybierać po PRZYCZYNIE błędu (grupa wejść), nie po wielkości błędu”. Przykład przyczyny: koniec promocji, czyli 3 procent komórek, 22 procent energii reszty, umiejętność 24,5.

**Inne wyniki negatywne.**
- `tft_langevin_pamiec_ukryta`: pamięć doklejona do zamrożonego T1c poprawia walidację i psuje test (74,262 wobec 75,337).
- `tft_rownanie_od_zera_pamiec`: jawne równanie z samych wejść daje 38,453 procent (placebo 0,946; czyste jądro liniowe −6,920).
- Synteza z 2026-09-04: pięć konfiguracji rdzenia Koopmana w TFT nie pobiło kontroli (MAE 2,1003 wobec 2,0895).
- Synteza SIGReg z 2026-09-03: kary dynamiczne miały bilans ujemny (MASE 0,5613 z karą wobec 0,4998 bez; bez członu wymuszenia 0,667).

## C. Schemat diagnostyki (2026-09-18)

**Twierdzenie.** Niech `m_t = E[z_{t+1} | H_t]` i innowacja `e* = z_{t+1} − m_t`. Dla dowolnego predyktora `h(H_t)`:

`E|z_{t+1} − h|² = E|e*|² (nieusuwalne) + E|m_t − h|² (usuwalne)`

Wnioski:
1. Sufit umiejętności to `S* = 1 − E|e*|²/E|Δz|²`; podnosi go tylko bogatsza historia.
2. Innowacja jest biała i nieskorelowana z przeszłymi wejściami. Autokorelacja reszty albo jej korelacja z wejściami oznacza brak członu (testy Billingsa i Zhu).
3. „Reszta modelu = zaskoczenie + brak w modelu. Sama wielkość reszty ich nie rozdziela.”

**Rozkład trójdzielny dla modelu liniowego.** Błąd = zaskoczenie + luka nieliniowości + błąd dopasowania. Luka nieliniowości mierzy jakość podniesienia Koopmana. Pomiar w projekcie: 45–49 procent (krok liniowy w surowym latencie), 86 (po podniesieniu), 90 (jawne równanie warunkowe); liczby „nie są ściśle porównywalne”.

**Mierniki do liczenia zawsze.** Umiejętność (łącznie, mediana per cząstka, bez 1 procenta), luka trybów, białość reszty, widmo `K`, długość pamięci wobec surogatu, udział członów.

**Reguła doboru danych.** Celować w część usuwalną estymatorem RHO-loss: `wynik_i = strata_modelu(i) − strata_modelu_odniesienia(i)`. Model odniesienia to mały model uczony na osobnych danych. Dobór po samej stracie zawodzi przy szumie (Hu i in. 2021).

**Jedna funkcja celu.** `J = E|g_{t+1} − K g_t − B u_t − Σ M_l g_{t−l}|² + α E|ψ(g_t) − z_t|² + λ R(g)`. Bez odwracalności albo członu przeciw kolapsowi minimum jest trywialne.

**Dowody Lean** (plik `DiagnostykaInnowacji.lean`, bez `sorry`, aksjomaty `propext`, `Classical.choice`, `Quot.sound`). Nagłówek wymienia 13 twierdzeń:
- `rozklad_bledu`, `sufit`, `umiejetnosc_sufit`, `kryterium_bialosci`, `blad_usuwalny_zero_iff`;
- `bogatsza_historia` (tryb warunkowy ma sufit nie niższy niż prognostyczny);
- `roznica_strat_bez_zaskoczenia` (podstawa doboru danych);
- `rozklad_bledu_na_podgrupie` (podgrupa musi być wyznaczona przez historię);
- `reszta_prostopadla_do_czlonow` (równanie Langevina jako tożsamość z równań normalnych);
- `rozklad_trojdzielny`, `luka_klasy_zero_iff`, `kolaps_zeruje_strate`, `rzut_istnieje`.

W pliku są też lematy pomocnicze `blad_usuwalny` i `rzut_wiezowy`. Dowód jest w abstrakcyjnej przestrzeni z iloczynem skalarnym; pomostu do `L²` i warunkowej wartości oczekiwanej nie ma. Decyle ruchu używane w diagnozie wyznacza przyszłość, więc twierdzenie o podgrupie ich nie obejmuje.

**Przeniesienie na zegar uczenia (część C schematu).** Cząstka to latent ustalonego przykładu sondy śledzony po krokach `τ`; wejście `u_τ` to skład partii i ustawienia uczenia. Schemat podaje trzy różnice:
1. Układ jest niestacjonarny, więc `K` dopasowuje się w oknie przesuwnym i śledzi widmo w funkcji `τ`. Wczesny sygnał kolapsu to moduły schodzące do zera w wielu kierunkach.
2. Pomiar musi być tani: stały zbiór sond rzędu tysięcy przykładów i zrzut co ustaloną liczbę kroków.
3. Dobór działa na zbiorze uczącym, nie na sondach. Wynik liczy się stratą predyktora JEPA, a modelem odniesienia jest mała kopia uczona na odłożonej części danych.

Zastrzeżenia ze schematu: „Brak jakiegokolwiek pomiaru na zegarze uczenia (część C jest projektem, nie wynikiem)”; RHO-loss nie ma potwierdzenia dla regresji latentu ani JEPA.

## D. Niezmienniki tożsamości

**Co znaleziono.**
- **D116:** 92,26 procent wariancji latentu to „adres” (średnia serii po czasie), 0,14 procent wspólny dzień, 7,60 procent interakcja. Po zdjęciu adresu promień spektralny spada z 0,9953 do 0,9388.
- **Tożsamość bez identyfikatora:** przez usterkę `item_nbr` model nigdy nie widział kodu produktu i odtworzył tożsamość z zachowania. Z adresu 192D log średniej sprzedaży wychodzi z R² 0,91.
- **D118:** „studnie” to mapa tożsamości, nie stany metastabilne (Spearman inwariancji z głębokością −0,784). Liczba dołków jest pokrętłem siatki (8 do 1154).
- **D115:** moduł wartości własnej nie rozdziela adresu od dynamiki. Czołowe mody leżą na podłodze z ponownego dopasowania na przetasowanym czasie; adres niosą mody szybkie.
- **Koopman:** w K1 sześć modów rzeczywistych (moduł 0,98–0,998) to 63 procent zmienności współrzędnych przy zerowym wkładzie w ruch.
- **Zastrzeżenie z dokumentu 23:** plik wynikowy D116 podaje 85 stanów, a 0,9226 dotyczy łącznego odjęcia dnia i serii (seria około 92,1255 procent, dzień 0,1384, reszta 7,7361).

**Jak użyto.**
- **Kotwica różnicowa:** stały błąd adresu znosi się w różnicy położeń.
- **Zdejmowanie adresu przed pomiarem dynamiki:** „każdy pomiar dynamiki na tym latencie mierzy głównie tożsamość, dopóki się jej jawnie nie zdejmie”.
- **Kody tożsamości w równaniu:** równe placebo (63,74 wobec 63,72, D120).
- **Położenie względne zamiast bezwzględnego:** stary test spadł o 6 punktów, więc położenie bezwzględne niesie potrzebną informację.
- **Propozycje z 2026-09-08, bez treningu:** adres jako zachowany kontekst prognozy, jako warunek odpowiedzi promocyjnej, oraz oddzielenie adresu od celu dynamiki.
- **Propozycja z 2026-09-03:** SIGReg na innowacji zamiast na latencie, żeby nie walczył z tożsamością.

## E. Reżimy

**Uczenie kontra inferencja.** Źródłem jest postulat T37 w `czastki_i_pole\TEORIA.md` (rodzina V-JEPA, 2026-08-12), „ustalone przez użytkownika, nie wywnioskowane z pomiaru”.
- **Uczenie:** działają predyktor i SIGReg, wagi się zmieniają, istnieje zegar kroków uczenia.
- **Inferencja:** enkoder zamrożony, bez predyktora i SIGRegu. Ruch jest bierny: cząstka zmienia się tylko dlatego, że zmienia się wejście.
- **Dlaczego to ważne:** liczb z inferencji nie wolno przypisywać siłom straty. Różnica tego samego pomiaru w obu reżimach to „jedyny uczciwy sposób przypisania czegokolwiek SIGRegowi albo predyktorowi”.
- **Egzekwowanie:** klucz `rezim` jest wymagany w `czastki_i_pole\WYNIKI.json`.
- **T37d:** trzeci podział to wejście z rozkładu uczenia albo spoza niego.
- **T37e:** trajektoria w inferencji jest układem wymuszonym `z(t+1) = F(z(t), u(t))`, nie autonomicznym.

**Pokrewne rozróżnienia.**
- Synteza „Równania dla latentu” wymienia trzy osie, których nie wolno mylić: dane (cząstka to seria), warstwy (cząstka to token), trening (cząstka to neuron).
- Cała linia TFT z września leży na zegarze danych przy zamrożonych wagach.
- W protokole równania są trzy fazy: uczenie, przewidywanie, obserwacja. Reszta z `t+1` nie poprawia wstecz prognozy z `t`.

## F. Most do obrazów (dokumenty 23 i 25)

Oba dokumenty to propozycje z 2026-09-11 bez wyników eksperymentalnych.

**Dokument 23.**
- **Rola grantu:** dostarcza metody mierzenia redundancji i zmian w czasie. Najpierw ma służyć jako selektor danych przed treningiem zwykłego detektora.
- **Mapowanie:** `z(v,t)` to zamrożone cechy DINO klatki lub klipu nagrania `v`; nowy JEPA nie jest trenowany. Zegar to czas PTS w sekundach. Dokument zastrzega: „Indeks warstwy transformera i numer epoki treningu to inne osie i nie służą tutaj do estymacji dynamiki filmu”.
- **Predyktory:** P0 `ẑ(t) = z(t−1)` (persistence), P1 ridge `A z(t−1) + b`, Pp z krótką historią.
- **Score:** `q(t) = ||r(t)||²/d`, czyli reszta predyktora.
- **Wybór klatek:** pokrycie cech plus reszta, z udziałem reszt `alpha = 0,25` jako propozycją startową.
- **Planowane ablacje:** pokrycie kontra pokrycie plus różnica cech; różnica cech kontra reszta ridge; poprawna kolejność kontra permutacja czasu wewnątrz filmu.
- **Ostrzeżenia:**
  - losowy błysk w tle daje dużą resztę bez informacji o narzędziu, a wolno poruszająca się końcówka małą;
  - `A` to aproksymacja liniowa, nie operator fizyczny;
  - udział wariancji to nie udział informacji;
  - „Niski MSE latentu nie oznacza lepszego AP”;
  - progi z D59 nie są uzasadnione dla chirurgii;
  - `miary.py` wymaga poprawek (granice segmentów, centrowanie celu);
  - Swift Sampling to bliski odpowiednik samego score.

**Dokument 25.**
- **Kierunek:** diagnoza, wybór działania (anotacja, nowe nagranie, dłuższy kontekst, dodatkowa obserwacja), niezależny pomiar skutku.
- **Predyktory P0–P4:** persistence, ridge, nieliniowy stan bieżący, historia, dodatkowy kanał.
- **Miara przewagi historii:** `Delta_hist = mean(||y − P2(z)||² − ||y − P3(z, historia)||²)`, liczona poza próbą.
- **Rozdzielenie:** przy zamrożonym enkoderze i predyktorze dopasowanym do całej puli ujawnienie nowych bboxów nie zmienia ani `z`, ani danych predyktora. Score jest wtedy statyczną kuracją.
- **Plan:**
  - etap A: syntetyki na CPU, pięć znanych przyczyn błędu;
  - etap B: YOLOv8n, strategie B0–B5 (losowo, równomiernie, pokrycie, pokrycie plus niepewność, plus reszta P2, plus przewaga historii), 13 treningów w pilocie i 39 z trzema ziarnami;
  - etap C: 12 treningów.
- **Ostrzeżenia:** regresja wieloopóźnieniowa to „model inspirowany pamięcią”, nie estymator jądra Mori–Zwanziga. Jeśli B5 przegra z B4, nie ma podstaw twierdzić, że diagnoza pamięci uzasadnia selekcję.

**INDEKS.md.**
- Decyzja autora z 2026-09-14: „najpierw statyczna kuracja na Nano, bez Langevina i Koopmana”.
- Opis dokumentu 36 (nieczytanego): „DMD nie bije różnicy klatek; ruch narzędzia przewidywalny w 22 do 29 procent”.
- Opis dokumentu 38 (nieczytanego): pomiar „na dwóch zegarach (klatki i epoki)”.

## G. Lekcje metodologiczne zapisane w plikach

**Przeciek przyszłości.**
- Wejścia jutra dzielić na znane z góry i nieznane, a tryb nazywać przy każdym wyniku.
- Kanały wolniejsze niż odcinek uczenia (miesiąc, rok, ropa) służą do rozpoznawania dat: decyl 1 spadł do −120, ridge do −111 procent. Wykluczać je regułą a priori; walidacja leżąca tuż za uczeniem ich nie wyłapuje.
- Normalizację liczyć tylko z dni uczenia.
- Robić testy przyczynowości i prefiksu; taki test znalazł błąd w `model.py`.

**Podłogi i kontrole.**
- Podłogą miary jest brak ruchu; obok niej ridge, liniowy rzut tego samego wymiaru, AR, model bez wejść.
- Placebo: wejścia innej serii, przetasowane wagi lub kody, sieć o losowych wagach.
- N21: „Podłoga musi zachowywać tę własność danych, której nie badamy. Tasowanie nie zachowuje żadnej.” Przykład: 505 dni wobec 21 z tasowania i 276 z AR(1).

**Wybór i ocena.**
- Kryteria i wybór zapisywać przed testem.
- Walidację odsuwać w czasie; w linii TFT trzykrotnie walidacja i test wybrały co innego.
- Epokę wybierać po mierze odpornej; rozrzut ziaren (około 5 punktów) przewyższał efekty.
- Plik członu zdarzeń zaleca co najmniej 5 ziaren.
- Odnotowywać, że test nie jest nietknięty.

**Raportowanie.**
- Podawać wynik łączny, medianę per cząstka, wynik bez 1 procenta największych, decyle i bootstrap po grupach.
- Każdy plik wyników ma sekcję „Czego to NIE dowodzi”.

**Czego nie twierdzić.**
- Że tryb warunkowy to prognoza.
- Że wysoka umiejętność jednokrokowa oznacza dobry model dynamiki (bieg swobodny się rozbiega).
- Że kopia sieci to jawne równanie.
- Że wynik z rzutu 2D przenosi się na pełny wymiar (D117 wycofana).
- Że minimum gęstości to potencjał albo bariera.
- Że spadek błędu latentu oznacza poprawę zadania końcowego.

**Dobór danych.** Celować w przyczynę błędu, nie w jego wielkość. Selektor ma używać wielkości znanych z historii.

**Kary w uczeniu** (synteza SIGReg).
- Człon dynamiczny bez jawnego wejścia wypycha wymuszenie z latentu.
- Parametry równania muszą być w optymalizatorze (błąd z 4 sierpnia).
- Skalę `sigma` trzeba przypiąć.

**Pomiar przed modelem** (synteza z 2026-09-04). Sufity liczone offline przewidziały wynik biegu i unieważniły trzy własne propozycje przed uruchomieniem.

## Źródła kluczowe

- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\schemat-diagnostyki-latentu-koopman-langevin-2026-09-18.md`
- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\czym_jest_ten_uklad_synteza_koncowa_2026-09-03.md`
- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\sigreg_koopman_langevin_jak_polaczyc_2026-09-03.md`
- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\rownania_dla_latentu_czego_szukac_2026-09-03.md`
- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\ile_dynamiki_jest_w_dziennej_sprzedazy_2026-09-04.md`
- `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\wykorzystanie-niezmiennikow-tozsamosci-tft-2026-09-08.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56\WYNIKI_L1.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56\WYNIKI_L2.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_krok_po_kroku\README.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_krok_po_kroku\WYNIKI_DIAGNOZA_SLABYCH.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_krok_po_kroku\WYNIKI_DIAGNOZA_DUZYCH.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_deep_koopman_mz\PLAN.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_deep_koopman_mz\WYNIKI_K1_K2.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_deep_koopman_mz\WYNIKI_K4_K5.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_deep_koopman_mz\WYNIKI_K6.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_deep_koopman_mz\WYNIKI_K7.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_petla_aktywna\WYNIKI_DIAGNOZA.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_petla_aktywna\WYNIKI_PETLA.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_czlon_zdarzen\README.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_czlon_zdarzen\WYNIKI.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_optymalizacja_95\WYNIKI_RUNDA1.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_optymalizacja_95\WYNIKI_RUNDA2_3.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_rownanie_od_zera_pamiec\ROWNANIE.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_rownanie_od_zera_pamiec\WYNIKI_TEST.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_destylacja_rekurencji\WYNIKI_TEST.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_pamiec_ukryta\WYNIKI_TEST.md`
- `F:\Studia\MCHTR\Grant\Projekt\experiments\czastki_i_pole\TEORIA.md` (część VIII, T37)
- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\DiagnostykaInnowacji.lean`
- `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\build_diagnostyka_innowacji.log`
- `F:\Studia\MCHTR\Grant\Projekt\Research\Tezy\tezy\` — pliki `D107.md`, `D115.md`, `D116.md`, `D117.md`, `D118.md`, `D119.md`, `D120.md`, `D121.md`, `D25.md`, `D37.md`, `D59.md`, `F23.md`, `N21.md`
- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-20_211814_diagnoza_rownania_lean_czlon_zdarzen_optymalizacja_95\SESSION_SUMMARY.md`
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\cel-rownanie-langevina-nie-kopia-sieci.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\23_POLACZENIE_Z_GRANTEM_DYNAMIKA_LATENTU.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\25_DIAGNOSTYKA_LATENTU_I_AKTYWNA_KURACJA.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\INDEKS.md`
