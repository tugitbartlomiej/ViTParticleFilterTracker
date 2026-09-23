# 43. Kryterium domknięcia tożsamości obiektu w latencie: Koopman, Mori-Zwanzig, Langevin

Data: 2026-09-23. Status: **synteza literatury (jedenaście prac przeczytanych w pełnym tekście) i formalny pipeline.
Bez nowych treningów.** Odpowiada na pytanie autora: „przeczytaj te prace i wyciągnij z nich jeden wniosek, co by
wskazywało, że tożsamość w latencie się domyka dla danego obiektu, i jak to mierzyć Koopmanem i Langevinem, tak
żeby było sformalizowane".

Rozwija dokument [42](42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md), który podał trzy nierównoważne
znaczenia domknięcia (grupowe, wymiarowe, Markowa) i miarę `Δ_pamięci`. Ten dokument dokłada to, czego tam brakowało:
twierdzenie wiążące dwa z tych znaczeń w jedno, gotowe do policzenia miary z literatury, i test spójności, który
odróżnia „brakuje danych" od „brakuje zmiennej". Wyniki pomiarowe z eksperymentu
`experiments/krystalizacja_latentu` są cytowane tam, gdzie kryterium było już częściowo policzone.

## 0. Jeden wniosek

> **Tożsamość obiektu jest domknięta w latencie dokładnie wtedy, gdy podprzestrzeń współrzędnych trwałych jest
> niezmiennicza względem kroku czasu. Ten sam fakt ma dwie równoważne miary: geometryczną (wyciek podprzestrzeni
> po jednym kroku, liczony kątami głównymi) i dynamiczną (zanik jądra pamięci w równaniu Moriego-Zwanziga
> dla tych współrzędnych). Domknięcie jest osiągnięte dla danego zbioru danych, gdy obie miary spadają poniżej progu,
> podprzestrzeń przestaje się obracać przy dokładaniu klatek, a jej wymiar przestaje rosnąć. Szum, który zostaje,
> musi spełniać związek fluktuacji i dyssypacji, inaczej brakuje zmiennej, nie danych.**

Każde zdanie tego wniosku ma źródło w literaturze i miarę do policzenia. Poniżej rozpisane po kolei.

## 1. Skąd bierze się równoważność (Lin21b, Uy21)

Niech `g(t) ∈ R^M` będzie wektorem obserwabli, u nas współrzędnych latentu klatki `t` wewnątrz jednego filmu.
Dokładna dyskretna równość Moriego-Zwanziga dla dowolnego wyboru obserwabli (Lin21b, tw. 2, równ. 41):

    g(k+1) = Ω(0) g(k) + Σ_{l=1..k} Ω(l) g(k−l) + W(k)

- `Ω(0) = C(Δ) C(0)^{-1}` to operator Markowa i **jest identyczny z operatorem EDMD**, czyli naszym `K`
  (Lin21b, sekcje 3.7, 4, 5). Nic nowego nie trzeba estymować, żeby mieć pierwszy człon.
- `Ω(l)`, `l ≥ 1`, to jądra pamięci, liczone rekurencyjnie z korelacji dwuczasowych (Lin21b, równ. 52):

      Ω(k) = ( C((k+1)Δ) − Σ_{l=0..k−1} Ω(l) C((k−l)Δ) ) C(0)^{-1}

- `W(k)` to dynamika ortogonalna, czyli szum, z definicji prostopadła do rozpiętości obserwabli.

Kluczowe zdanie, które czyni z tego kryterium (Lin21b, sekcja 3.2): *„pamięć i wymuszenie zewnętrzne istnieją tylko
dlatego, że mamy niekompletny zestaw obserwabli; gdy dynamika jest domknięta w M, oba człony są zerem"*. Uy21 daje
to samo w postaci jawnego wzoru (równ. 3.8): jądra pamięci

    E_l = Qᵀ A Q⊥ ((Q⊥)ᵀ A Q⊥)^{l−1} (Q⊥)ᵀ A Q

**znikają tożsamościowo wtedy i tylko wtedy, gdy `Qᵀ A Q⊥ = 0`, czyli gdy rozpiętość `Q` jest podprzestrzenią
niezmienniczą operatora `A`.**

To jest równoważność, której brakowało w [42]: domknięcie Markowa (znikające jądro pamięci, sekcja 3c tamtego
dokumentu) i domknięcie geometryczne (niezmienniczość podprzestrzeni) to **jeden warunek**. Można go mierzyć dwiema
niezależnymi drogami i sprawdzać, czy się zgadzają.

## 2. Co jest podprzestrzenią tożsamości (Ber23, nasz odsiew sceny)

Ber23 pokazuje, że w operatorze `K` na latencie sekwencji **czynniki stałe w czasie odpowiadają wartościom własnym
równym jedności** (ich obserwacja kluczowa, sekcja 4): jeżeli `λ_1 = 1`, to składnik `z̄_j^1 φ_1` w rozwinięciu
`z_jᵀ K^r = Σ_i z̄_j^i λ_i^r φ_iᵀ` nie zależy od `r`. Ber23 ten podział **wymusza** stratą (Eq. 6 do 8: `k_s`
wartości własnych ciągniętych do `1 + 0i`, reszta do kuli o promieniu `ε`), a potem **weryfikuje** z etykietami przez
podmianę współrzędnych stałych między sekwencjami i klasyfikator (90,6 procent przeniesienia koloru włosów, 4,46
procent EER identyfikacji mówcy na `Z_stat` wobec 26,8 na `Z_dyn`).

U nas nikt niczego nie wymuszał, a wartości własne bliskie jedności wyszły same z gotowego detektora. Definicja
podprzestrzeni tożsamości musi więc mieć jeszcze jeden warunek, którego Ber23 nie potrzebował, bo jego zbiory nie
miały tła:

    V_toż = span{ φ_i : |λ_i − 1| < δ  ∧  w_i ≥ w_min }

gdzie `w_i` to udział zmienności współrzędnej `φ_i` WEWNĄTRZ filmu (odchylenie od średniej filmu) w jej zmienności
całkowitej. Mod trwały, ale stały przez cały film, opisuje oko, światło i kadr, a nie narzędzie; tożsamość narzędzia
musi się w filmie zmieniać, bo narzędzia się wymienia. Zmierzone na puli (`zajetosc_tozsamosci.py`): z dziesięciu
modów o `Re λ > 0,85` pięć ma `w_i` od 27 do 35 procent (scena), pięć ma od 42 do 53 procent (tożsamość).

Ostrzeżenie z Ber23 (dodatek B.5), które dotyczy nas wprost: *które* współrzędne stałe niosą *którą* cechę, nie da
się ustalić z góry, a wymiar podprzestrzeni pojedynczej cechy zmieniał się między partiami danych (1, 2 albo 3).
Dlatego kryterium domknięcia musi dotyczyć całej `V_toż`, a nie pojedynczych osi.

## 3. Miara geometryczna: wyciek podprzestrzeni (Con26, Pan20)

Con26 definiuje niezmienniczość podprzestrzeni `V` przez **kąty główne** między `V` a jej obrazem `KV`:

    θ_1 = min { arccos |⟨u, v⟩| : u ∈ V, v ∈ KV, ‖u‖ = ‖v‖ = 1 },  dalej rekurencyjnie na dopełnieniach

i dowodzi (lemat 3.2), że `sin θ_j` jest dokładnie normą tego, co z `v_j ∈ KV` nie mieści się z powrotem w `V`:

    ‖P_V v_j − v_j‖ = sin θ_j

Dla podprzestrzeni jednowymiarowej `span{g}` to zwykłe względne residuum funkcji własnej,
`inf_λ ‖K g − λ g‖ / (|λ| ‖g‖)`. Jednokrokowy błąd rzutu dowolnego `g ∈ V` jest ograniczony przez
`Σ_i |g_i| sin θ_i` (sekcja 3.2), a wielokrokowy przez sumę geometryczną tych wycieków (tw. 4.1).

**Definicja miary:**

    wyciek(V) = max_j sin θ_j(V, KV)      (albo średnia ważona udziałem wariancji)

Liczy się z danych bez żadnego domknięcia operatora: macierze Grama `G = ΨᵀWΨ`, `A = ΨᵀWΨ'`, `L = Ψ'ᵀWΨ'`
z par migawek, kompresja na `V`, ortonormalizacja przez macierz łączną `J_V` i SVD (Con26, algorytm 3.1).
Próg: Con26 znajduje go z załamania krzywej błędu jednokrokowego względem `θ` (dopasowanie odcinkowo-liniowe
z jednym punktem załamania); w ich układzie testowym wypadł przy `θ ≈ 0,01`. **Próg trzeba wyznaczyć z naszych
danych tą samą procedurą, nie przepisać.**

Pan20 daje wersję wielokrokową, a posteriori, na trajektorii walidacyjnej (równ. 3.1 do 3.2):

    Q_i = max_t |φ_i(x(t)) − e^{λ_i t} φ_i(x(0))| / ‖φ_i‖

i selekcję modów przez ElasticNet wielozadaniowy z odcięciem na kolanie krzywej rekonstrukcji. To jest kontrola
dla `wyciek(V)`: mod, który przecieka po jednym kroku, musi mieć duże `Q_i` po wielu krokach.

## 4. Miara dynamiczna: jądro pamięci współrzędnych tożsamości (Lin21b, Gup23c, Uy21)

Rzutujemy latent na `V_toż` i liczymy równanie z sekcji 1 **tylko dla tych współrzędnych**:

    c(k+1) = K_V c(k) + Σ_{l=1..L} Ω_V(l) c(k−l) + W_V(k),   c = P_V z

**Definicja miary:**

    pamięć(V) = Σ_{l=1..L} ‖Ω_V(l)‖_F / ‖K_V‖_F

z jądrem liczonym rekurencją Lin21b (równ. 52) albo, równoważnie, stopniowym OpInf Uy21 (równ. 3.16, 3.20:
kolejne `Ω(l)` dopasowywane do reszty po odjęciu poprzednich). Gup23c pokazuje na latencie autoenkodera, że tam,
gdzie operator liniowy nie wystarcza, norma członu pamięci `‖ξ‖` skacze (przejście w śladzie za cylindrem, rys. 5),
a gdy liczba obserwabli rośnie, wkład pamięci maleje do zera (rys. 3: `r ∈ {2, 8, 16}`, błędy Koopman-AE i MZ-AE
zbiegają do tej samej wartości). To jest dokładnie zachowanie, którego oczekujemy od domykającej się tożsamości:
**przy dokładaniu klatek `pamięć(V)` ma maleć, a przy dokładaniu osi do `V` ma maleć jeszcze szybciej.**

Długość `L`: Uy21 nie ma testu, wybiera z zaniku norm `‖E_l‖` (rys. na s. 15). Lin21b obserwuje zanik jądra w
skończonym czasie i proponuje wielkość albo skalę czasową jądra jako cel optymalizacji obserwabli (sekcja 7).
U nas grantowe ustalenie U14 dało głębokość jednego kroku na cząstkach Eulera; `L = 3` wystarczy jako pierwszy
pomiar, z kontrolą przy `L = 6`.

Równoważność z sekcji 1 mówi: `wyciek(V) → 0 ⟺ pamięć(V) → 0`. **Jeżeli jedno spada, a drugie nie, coś jest
źle policzone** (za mało par, zły próg, scena wciąż w `V`). To jest pierwszy test wewnętrzny pipeline'u.

## 5. Test spójności: fluktuacja i dyssypacja (Lin21b, równ. 89)

Dyskretny związek fluktuacji i dyssypacji:

    Ω(k) = − ⟨W(k), W(0)ᵀ⟩ C(−Δ)^{-1},   k ≥ 1

wiąże jądro pamięci z autokorelacją szumu. Lin21b używa go jako sprawdzianu numerycznego (rys. 4, 5): jądro
wyciągnięte wprost ma się zgadzać z jądrem policzonym z korelacji reszt. Dla nas ma to znaczenie diagnostyczne,
którego w [42] nie było:

- **jądro i autokorelacja reszt się zgadzają, oba małe:** tożsamość domknięta, szum jest nieusuwalny;
- **jądro małe, ale reszty skorelowane w czasie:** rekurencja niedoszacowała pamięci, za krótkie `L` albo za mało
  par;
- **jądro duże, reszty białe:** sprzeczność z równ. 89, czyli model jest źle wyspecyfikowany, w latencie brakuje
  zmiennej, której żadna pamięć nie zastąpi. To jest przypadek „brakuje zmiennej, nie danych" i wtedy dokładanie
  klatek nie pomoże.

Trzeci przypadek jest dokładnie tym, o co pytał autor przy obrocie w głąb: skrócenie narzędzia przy obrocie wokół osi
patrzenia to brakująca zmienna, nie brakujące dane.

## 6. Domknięcie względem danych: stabilność i nasycenie (Kap25, She20, Ans19, Rec19)

Powyższe miary są liczone dla ustalonego zbioru klatek. Domknięcie „dla danego obiektu" znaczy: dokładanie klatek
niczego już nie zmienia. Dwie miary:

**Stabilność podprzestrzeni** (Kap25, równ. 2.14 i 3.9; sprawdzone w pełnym tekście, bez żadnych wartości
własnych): dla baz ortonormalnych `X`, `Y` podprzestrzeni policzonych z `n` i `n + Δn` klatek

    D̂(X, Y) = sqrt( 1 − ‖Xᵀ Y‖_F² / max(p, q) ) ∈ [0, 1]

zero to te same kierunki, jeden to prostopadłe. Kap25 stawia kolejny pomiar w środku między dwoma ustawieniami,
dla których `D̂` jest największe, bo tam podprzestrzeń zmienia się najszybciej, więc tam brakuje danych. U nas
„ustawieniem" jest liczba oznaczonych klatek albo checkpoint. Zmierzone dotąd (`krystalizacja.py`, CCA wolnych
modów między checkpointami): 0,84 do 0,98, bez zbieżności do jedności, czyli **podprzestrzeń wciąż się obraca**.

**Nasycenie wymiaru** (She20, równ. 3: `s_l = dim E_l^k / dim Z_l`, liczba kierunków potrzebnych do progu `δ`
wariancji; Ans19, Rec19: wymiar wewnętrzny najpierw spada, potem rośnie). Dla `V_toż`:

    dim(V_toż)(n) = |{ i : |λ_i − 1| < δ ∧ w_i ≥ w_min }|   przy n klatkach

Domknięcie wymaga plateau tej liczby względem `n`. Zmierzone dotąd: liczba modów o `Re λ > 0,9` w bloku `query`
spada z 8 do 6 w trakcie przeuczania, a uczestnictwo kierunków rośnie z 8,6 do 16,4; to jest rozlewanie się, nie
nasycenie. Najlepsza epoka wypada tam, gdzie trwałość modów ma szczyt (`kiedy_przestac.py`, trzy ziarna, koszt
0,004 mAP wobec 0,030 przy treningu do końca).

## 7. Weryfikacja z etykietami (Ber23, jedyne miejsce, gdzie etykiety wolno użyć)

Ber23 weryfikuje tożsamość podmianą współrzędnych stałych i klasyfikatorem. U nas nie ma dekodera, więc podmiana
nie ma obrazu; odpowiednikiem jest **odczyt liniowy cechy fizycznej z `V_toż`** z podziałem po filmach:

    R²_toż(cecha) = R²( cecha ~ P_V z ),   uczenie i ocena na różnych filmach

Zmierzone: wydłużenie narzędzia 85,7 procent z ośmiu modów trwałych, położenie 4,1 procent (kod przesunięciowy)
albo 44 procent (kod czasowy); różnica między nimi to właśnie to, co [42] nazwało domknięciem grupowym.
Domknięcie z etykietami znaczy: `R²_toż` przestaje rosnąć z `n`, przy `R²` położenia bliskim zera.

## 8. Od domknięcia do doboru klatek (Lin21b sekcja 7, Kim26, Uy21b)

Lin21b proponuje użyć wyciągniętej dynamiki ortogonalnej `F(t)` „analogicznie do procesu Grama-Schmidta" do
znalezienia obserwabli, których brakuje: dominujący kierunek `F(t)` to kierunek, który trzeba dołożyć do zestawu.
Przeniesione na dobór klatek daje to regułę, której etap 1 nie testował:

    kowariancja reszt   Σ_W = ⟨W(k) W(k)ᵀ⟩  na puli, po rzucie na V_toż i jej dopełnienie
    kierunek braku      e_1 = wektor własny Σ_W o największej wartości własnej
    ocena klatki        s(t) = |⟨W(t), e_1⟩|  ·  1[ W skorelowane z historią w otoczeniu t ]

Klatki z dużą resztą **wzdłuż kierunku braku** i **ze strukturą** (reszta przewidywalna z historii, sekcja 5) niosą
brakującą zmienną. Klatki z dużą resztą białą to szum i nie warto ich oznaczać. To odróżnia tę regułę od Swift
Sampling (Kim26): tam zaskoczenie to norma reszty po ekstrapolacji Taylora trzeciego rzędu na cechach warstwy
pierwszej, bez pytania, czy reszta ma kierunek i strukturę, a wybór to lokalne maksima `r_t`. Kim26 jest właściwą
podłogą porównawczą dla naszej reguły, bo jest najbliższy i tani (narzut 1,02 razy).

Uy21b daje wersję dla identyfikacji operatora przy szumie: wybieraj wiersz danych, który najbardziej podnosi
najmniejszą wartość osobliwą macierzy danych, `argmax (ψ_Mᵀ d_+)²`. To jest kryterium D-optymalne z pilota
w [42] (sekcja 5, reguła B), który już pokazał, że identyfikowanie operatora nie jest tym samym, co uczenie
detektora. Zostawiamy je jako kontrolę, nie jako regułę główną.

## 9. Pipeline

Wejście: wstępny model `M_0` (u nas: B2 na 1500 klatkach), pula `P` bez etykiet, budżet `B`.

1. **Kod klatki.** Dla każdej klatki puli: wycinek z ramki PRZEWIDZIANEJ (bez etykiet), opis na siatce 3 × 3
   z tokenów, wybielenie do `d = 48`. Wyjście `z_t`.
2. **Operator.** Pary kolejnych klatek tego samego filmu (odstęp ≤ 6), EDMD → `K`, `λ_i`, `φ_i`.
3. **Podprzestrzeń tożsamości.** `V_toż` z sekcji 2 (`δ`, `w_min` do ustalenia z załamania krzywej, start:
   `Re λ > 0,85`, `w ≥ 0,35`). Zapisać `dim V_toż`.
4. **Wyciek.** `wyciek(V_toż)` kątami głównymi (sekcja 3), próg z załamania krzywej błędu jednokrokowego.
5. **Pamięć.** `pamięć(V_toż)` z rekurencji Lin21b przy `L = 3` i `L = 6` (sekcja 4). Test wewnętrzny:
   kierunek zmian obu miar musi się zgadzać.
6. **Spójność.** Związek fluktuacji i dyssypacji (sekcja 5): rozstrzyga między „brak danych" a „brak zmiennej".
   Przy „brak zmiennej" pipeline zatrzymuje się i zgłasza kierunek braku; dobór klatek nic nie da.
7. **Stabilność i nasycenie.** Powtórzyć kroki 2 do 5 dla podzbiorów rosnących: `n ∈ {250, 500, 1000, 1500, ...}`.
   `D̂(V_n, V_{n+Δn})` i `dim V_toż(n)`. Domknięcie: `D̂ < ε_D` i plateau wymiaru.
8. **Weryfikacja.** `R²_toż(wydłużenie)` vs `n` z podziałem po filmach (sekcja 7); wolno użyć etykiet tylko tu.
9. **Dobór.** Dopóki nie ma domknięcia: (a) zajętość `V_toż` na kratce, po jednej klatce z każdej zajętej komórki,
   od najliczniejszych (`nasycenie_tozsamosci.py`: 250 klatek daje 99 procent komórek, równomierny czas potrzebuje
   2000); (b) gdy zajętość jest już pełna, reguła kierunku braku z sekcji 8. Kontrole przy tym samym budżecie:
   równomierny czas, los, Swift Sampling.
10. **Zatrzymanie treningu.** Szczyt trwałości modów bez etykiet (`kiedy_przestac.py`).
11. **Werdykt.** mAP na walidacji przy tym samym budżecie, trzy ziarna, próg 0,006 (dwa odchylenia ziarna).

Kroki 1 do 3 i 9a są policzone. Kroki 4 do 8 to nowy kod: `wyciek_tozsamosci.py` (kąty główne),
`pamiec_tozsamosci.py` (rekurencja Lin21b plus test FDT), `stabilnosc_tozsamosci.py` (D̂ i wymiar względem `n`).
Krok 11 dla 9a właśnie liczy się na Edenie (dziewięć biegów, `results_maly_budzet`).

## 10. Co już wiemy i co to zmienia

| miara | źródło | stan u nas | wniosek |
|---|---|---|---|
| mody z `Re λ` blisko 1 | Ber23, nasze | 10 modów `> 0,85`, 5 scena, 5 tożsamość | podprzestrzeń istnieje bez wymuszania |
| wyciek `V_toż` | Con26 | nie liczony | nowy kod |
| pamięć `V_toż` | Lin21b, Uy21 | U14: 1 krok, 5 do 6 procent na cząstkach Eulera; tu nie liczone na `V_toż` | nowy kod |
| spójność FDT | Lin21b | nie liczona | nowy kod, rozstrzyga o obrocie w głąb |
| stabilność | Kap25 | CCA 0,84 do 0,98, bez zbieżności | tożsamość NIE jest domknięta |
| nasycenie wymiaru | She20 | uczestnictwo rośnie 8,6 → 16,4 po szczycie | rozlewanie, nie nasycenie |
| odczyt z etykietami | Ber23 | wydłużenie 85,7, położenie 4,1 | tożsamość oddzielona od położenia |
| zajętość | nasze | 250 klatek po kwadratach = 99 procent komórek | test na Edenie w toku |

Najważniejsza zmiana względem [42]: **domknięcie Markowa i domknięcie geometryczne to jeden warunek** (sekcja 1),
więc `Δ_pamięci` z [42] i `wyciek` z Con26 muszą dawać ten sam werdykt; rozbieżność to błąd pomiaru. Drugi wniosek:
związek fluktuacji i dyssypacji daje test, którego [42] nie miał, i to on decyduje, czy obrót w głąb (skrócenie
narzędzia) jest do nadrobienia klatkami.

## 11. Kontrargumenty

- **Podprzestrzeń trwała może być trwała z nudów.** Mod o `λ ≈ 1` może opisywać wielkość, która nie zmienia się, bo
  jest nieinformatywna (stała). Odsiew sceny `w_i ≥ w_min` częściowo to łapie; Pan20 przypomina, że stałe funkcje
  własne trzeba usuwać jawnie.
- **Wyciek zależy od słownika.** Con26 liczy kąty w przestrzeni rozpiętej przez słownik `Ψ`; u nas słownik to
  wybielony opis wycinka. Wyciek mały w tym słowniku nie znaczy mały w latencie pełnym. Kontrola: policzyć przy
  `d = 24` i `d = 96`.
- **Pamięć znika też wtedy, gdy ruch to ruch kamery.** [42] sekcja 3c: przy surowym latencie deficyt pamięci był
  mały, bo mierzony ruch był ruchem kamery. Dlatego wszystkie miary liczymy na `V_toż` i na jej dopełnieniu osobno,
  nigdy na surowym latencie.
- **Równoważność z sekcji 1 jest ścisła dla dynamiki liniowej.** Uy21 dowodzi jej dla `A` liniowego; dla naszego
  latentu `K` jest przybliżeniem EDMD, więc „znikanie" znaczy „poniżej progu szacowanego z par", nie zero.
- **Ber23 nie odkrywa tożsamości, tylko ją wymusza.** Nasza sytuacja jest inna (bez wymuszenia), co jest mocniejsze
  jako obserwacja, ale słabsze jako gwarancja: nie ma straty, która pilnowałaby, żeby `V_toż` nie mieszała się
  z pozą przy dalszym treningu. Krok 7 (stabilność) jest jedynym zabezpieczeniem.
- **Zajętość może być pełna, a detektor i tak nie zyskać.** K4 (pokrycie modów trwałych) w etapie 1 wygrało o 0,0068,
  na granicy szumu. Biegi na 250 klatkach rozstrzygną, czy zajętość `V_toż` jest lepsza niż pokrycie odległościowe.

## 12. Data gaps

- Wyciek, pamięć i FDT na `V_toż` nie są policzone; to trzy skrypty do napisania.
- Brak eksportu latentu dla podzbiorów `n < 1500` (krok 7 wymaga przeliczenia `V_toż` z rosnących podzbiorów
  klatek puli, co da się zrobić z nagrania `CZASTKI_pool_b0500` bez nowego treningu, bo operator liczy się z par
  klatek, nie z modelu).
- Swift Sampling nie jest zaimplementowany jako ramię kontrolne.
- Test zbioru testowego (25 filmów) nigdy nie był uruchamiany; wszystkie werdykty są na walidacji.

## Bibliografia (DOI albo link, wszystkie przeczytane w pełnym tekście przez Undermind)

- Ber23: Berman, Naiman, Azencot, *Multifactor Sequential Disentanglement via Structured Koopman Autoencoders*,
  ICLR 2023. https://doi.org/10.48550/arXiv.2303.17264
- Mas21: Comas i in., *Self-Supervised Decomposition, Disentanglement and Prediction of Video Sequences while
  Interpreting Dynamics: A Koopman Perspective*, 2021. https://arxiv.org/abs/2110.00547
- Con26: Conradie i in., *Trustworthy Koopman Operator Learning: Invariance Diagnostics and Error Bounds*, 2026.
  https://doi.org/10.48550/arXiv.2603.15091
- Pan20: Pan, Arnold-Medabalimi, Duraisamy, *Sparsity-promoting algorithms for the discovery of informative
  Koopman-invariant subspaces*, JFM 2021. https://doi.org/10.1017/jfm.2021.271
- Lin21b: Lin, Tian, Anghel, Livescu, *Data-Driven Learning for the Mori-Zwanzig Formalism: A Generalization of the
  Koopman Learning Framework*, SIAM J. Appl. Dyn. Syst. 2021. https://doi.org/10.1137/21m1401759
- Gup23c: Gupta i in., *Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder*, 2023.
  https://doi.org/10.48550/arXiv.2310.10745
- Uy21: Uy, Peherstorfer, *Operator Inference of Non-Markovian Terms for Learning Reduced Models from Partially
  Observed State Trajectories*, J. Sci. Comput. 2021. https://doi.org/10.1007/s10915-021-01580-2
- Uy21b: Uy i in., *Active operator inference for learning low-dimensional dynamical-system models from noisy data*,
  SIAM J. Sci. Comput. 2021. https://doi.org/10.1137/21m1439729
- Kap25: Kapadia, Benner, Feng, *Subspace-Distance-Enabled Active Learning for Efficient Data-Driven Model Reduction
  of Parametric Dynamical Systems*, 2025. https://doi.org/10.48550/arXiv.2505.00460
- She20: Shenk i in., *Feature Space Saturation during Training*, BMVC 2021. https://doi.org/10.5244/c.35.28
- Kim26: Kim i in., *Swift Sampling: Selecting Temporal Surprises via Taylor Series*, 2026.
  https://doi.org/10.48550/arXiv.2605.22678
- Ans19: Ansuini i in., *Intrinsic dimension of data representations in deep neural networks*, NeurIPS 2019.
  https://arxiv.org/abs/1905.12784 (streszczenie)
- Rec19: Recanatesi i in., *Dimensionality compression and expansion in Deep Neural Networks*, 2019.
  https://arxiv.org/abs/1906.00443 (streszczenie)
- Col19: Coleman i in., *Selection via Proxy: Efficient Data Selection for Deep Learning*, ICLR 2020.
  https://arxiv.org/abs/1906.11829 (streszczenie)
- Eva23: Evans i in., *Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual
  Understanding*, 2023. https://doi.org/10.48550/arXiv.2312.05328 (streszczenie)
