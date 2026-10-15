# Pomiary jak w pułapce optycznej: plan eksperymentów na Edenie (2026-10-14)

Status: **PLAN, nic nie uruchomione.** Polecenie użytkownika 2026-10-14: „zapisz to koniecznie jako kolejne eksperymenty do
przeprowadzenia na Edenie, opisz go dokładnie”. Plan przenosi na tokeny MałegoViT (s47) standardowe pomiary, które fizycy robią
na koloidzie w pęsecie optycznej, i dodaje jeden test, którego w laboratorium zrobić się nie da: porównanie pola sił odtworzonego
z ruchu z polem znanym dokładnie (mamy wagi, stan AdamW i Jakobian).

Literatura: Undermind, workspace ViT, głębokie przeszukanie „Measurement toolkit for colloidal particles in optical traps
observables calibration and diagnostics” (2026-10-14, 228 prac) + Scite (fragmenty pełnych tekstów, żadna z użytych prac nie ma
wycofania ani korekty). Lista DOI na końcu.

---

## 0. Słownik: co jest czym w tych pomiarach

| fizyka | u nas (czas UCZENIA, nie głębokości) |
|---|---|
| koloid | token i sondy (zdjęcia z puli uczącej, obserwowanego w trybie eval): latent `z_i(t)` = GELU MLP ostatniego bloku, 768D, zapisywany w losowym rzucie 128D (rp128) albo pełny |
| czas | krok uczenia `t` (Δt = 1 krok); obserwator zapisuje co 1 krok w gęstym oknie |
| pułapka | **znaczenie (b) z ZASAD Z24**: centrum drgań, wokół którego token się trzęsie; NIE pole laserów J (to jest zegar głębokości, T = 0) |
| szum termiczny | szum paczki (minibatch): różnica kroku na paczce i kroku na średnim gradiencie |
| siła zewnętrzna (do odpowiedzi) | dodatkowy człon straty `−h·ε·z_i` (protokół z `experiments/fdt_temperatura_20261013`) |
| tarcie / ruchliwość | odpowiedź przemieszczenia na stałą siłę w krótkim czasie, `χ(t) ≈ t/γ` |

Most do pola laserów: w E3 porównujemy wir w ruchu (czas uczenia) z udziałem wiru w Jakobianie głębokości tego samego tokenu
(`film_krysztal/temperatura_pulapki`, zasady Z15, Z23). To jest pytanie, czy siodło z wirem z laserów rządzi ruchem w uczeniu.

## 1. Co już jest zrobione (nie powtarzać)

- Relacja Einsteina / FDT na pułapce (średnia ±300 kroków) spełniona, na koloidzie nie; jedyna sprężyna to weight decay
  (`experiments/fdt_temperatura_20261013`, pamięć `langevin-koopman-pulapki-2026-10-13`).
- MSD i wykładnik α, retencja sąsiadów, korelacja krzyżowa par z C(0) > 0 (wspólny napęd, nie hydrodynamika), test jądra pola
  (`jadro_pola.py`), odpowiedź tokenów na pchnięcie po łodydze w głębokości (`eden/odpowiedz_tokenow.py`).
- Temperatura bramek (Fermi) i termometr J_T w głębokości (`film_krysztal/temperatura_pulapki`).

Ten plan dokłada: kalibrację sztywności pełnym zestawem metod (E1), rekonstrukcję pola z ruchu kontra pole dokładne (E2),
prądy i wir w ruchu (E3), temperaturę zależną od częstotliwości i rozpraszanie (E4), produkcję entropii (E5), przeskoki
laserów jak ucieczki Kramersa (E6), sprzężenie par po odjęciu napędu (E7), mikroreologię (E8).

---

## 2. Dane

### 2a. Co już leży na Edenie (2026-10-14, `~/smallvit_runs/2026-10-10_szklo_pokretla/wyniki/`)

| bieg | porcje `gesty` (surowe) | stany `pelny/` | do czego |
|---|---|---|---|
| `okno60_rp` | 3 pliki (porcje skasowane) | `stan_e60.pt`, `stan_e61.pt` (s47, krok 65 640 i 66 734) | **start G1** |
| `lr_5e-4_rp_z48_zostaw`, `lr_5e-4_rp_z49_zostaw` | 10 | e02, e03 | ziarna 48 i 49 do E1, E3, E5, E7 |
| `lr_2e-3_rp_zostaw` | 10 | e02, e03 | gorąco (lr 2e-3) do E1, E3, E5 |
| `cnn_lr_5e-4_rp_zostaw` | 10 | e02, e03 | kontrola architektury (bez uwagi) |
| `bezpoz_lr_5e-4_rp` | 10 | e02, e03 | kontrola bez kodowania pozycji |
| `klas_vit_od_zera_60ep_lr_5e-4` | 15 | e59, e60 | inne zadanie (klasyfikacja Imagenette, brak samostygnięcia) |
| `lr_5e-4_rp`, `dlugi_lr_5e-4_rp` | **tylko lokalnie** (`assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/`), na Edenie usunięte 2026-10-13 | — | analizy lokalne albo regeneracja (G2) |

Biegi „zostaw” zapisują co 1 krok do kroku 2188, potem co 10 (`gesto_do 2188`, `co_pozniej 10`), 12 sond = 7200 tokenów, rp128.
Do E2, E4, E6, E8 brakuje w nich stanów AdamW po drodze i pre-aktywacji laserów, więc potrzebne są nowe biegi (2b).

### 2b. Nowe biegi do wygenerowania

Wspólne: `trening_pokretla.py` z `experiments/maly_vit_szklo_pokretla_20261010/kod` (na Edenie `2026-10-10_szklo_pokretla/kod`),
`--rzut rp --rzut_dim 128 --zapis f16` (f16 = f32 w miarach, fala 4), `--sondy` z `SONDY_100.jsonl` biegu 35 000, 12 sond,
nowy katalog `~/smallvit_runs/2026-10-14_pomiary_pulapki/wyniki/<bieg>`.

| id | co | jak | gęste okno | czas GPU | miejsce |
|---|---|---|---|---|---|
| **G1** | s47 zimny: jedna epoka po 60. | wznowienie od `okno60_rp/pelny/stan_e60.pt` jak `okno60_rp` (`--wznow --epoki 61`, `gesto_do 70000`), + K1 stany co 25 kroków, + K2 pre-aktywacje bloku 6 dla 2 sond | kroki 65 640–66 734 (1094) co 1 | ~2 min (okno60_rp: 57 s) | ~6 GB |
| **G2** | s47 gorący: start uczenia | od zera, ziarno 47, lr 5e-4, 2 epoki, `ZOSTAW=1`, + K1 (co 25 kroków do 2188), + K2 | kroki 0–2188 co 1 | ~3 min + 5 min danych | ~8 GB |
| **G3a / G3b** | seria temperatur | jak G2, `--lr 1.25e-4` i `--lr 2e-3` | 0–2188 co 1 | 2 × ~8 min | 2 × ~8 GB |
| G4 (opcja) | dłuższe okno zimne | G1 przez 3 epoki (61–63), bez K2 | 3282 kroki | ~5 min | ~6 GB |

Budżet: Eden ma 35 GB wolnego (2026-10-14). G1 + G2 + G3 ≈ 30 GB, więc **kolejno**: wygenerować, policzyć analizy, zapisać JSON,
pobrać; surowe porcje usuwać dopiero po pobraniu i zgodnym MD5 (zasada użytkownika z 2026-10-13, rejestr
`~/smallvit_runs/USUNIETE_Z_EDENU.txt`) albo za zgodą użytkownika. Lokalnie F: ma ~13 GB, więc pobierać tylko wyniki analiz
i jedną kopię surową na raz.

Uwaga do wznowienia G1: w `okno60_rp` kolejka permutacji danych po wznowieniu zaczęła się od nowa (README biegu 35 000), więc
kroki G1 nie są bitowo tymi samymi krokami co bieg 806 checkpointów. Dla pomiarów to bez znaczenia (stan wag i AdamW jest ten sam).

### 2c. Zmiany w kodzie (przed G1–G3)

- **K1 `--stany_co N` (+ `--stany_od`, `--stany_do`)** w `trening_pokretla.py`: zapis pełnego stanu (wagi, AdamW m i v, generatory)
  co N kroków w gęstym oknie do `pelny/okno/stan_k<krok>.pt`. Potrzebne do dokładnego dryfu i dyfuzji (E2) i do replik (E4, E8).
  Rozmiar: ~35 MB na stan (2,9 mln param. × 3 × f32); 1094/25 ≈ 44 stany ≈ 1,5 GB. Pułapka znana: `Optimizer.load_state_dict`
  nie kopiuje tensorów, repliki muszą dostać `copy.deepcopy` (pamięć langevin-koopman); bufory BatchNorm łodygi odtwarzać przez
  `net.load_state_dict` (pamięć fala 4).
- **K2 `--preakt blok,n_sond,n_tok`** w `obserwator_rp.py`: zapis pre-aktywacji `a = fc1(LN(x))` (przed GELU, 768 laserów) wybranego
  bloku dla pierwszych n sond i n_tok tokenów, f16, co krok w gęstym oknie. Z samego GELU się ich nie odzyska (GELU nie jest
  odwracalne poniżej −0,75). Rozmiar dla 2 sond × 600 tokenów × 768 × 1094 kroków × 2 B ≈ 2 GB.
- **K3 protokoły pola w `experiments/fdt_temperatura_20261013/kod/fdt_odpowiedz.py`**: obecnie „pole włączone w t_w” (skok). Dodać
  `--protokol impuls` (pole tylko w jednym kroku t_w → funkcja odpowiedzi R(t) wprost) i `--protokol sinus --okresy 10,30,100,300`
  (h(t) = h0·sin(2πt/P) → zespolona podatność χ(ω), E8). Zachować bliźniaka h = 0 (poziom niedeterminizmu GPU musi być ≪ odpowiedzi).
- **K4 skrypty analiz** (nowe, w `kod/` tego eksperymentu): `ou_kalibracja.py` (E1, E5a), `forma_pole.py` (E2), `prady_wir.py` (E3),
  `fdt_czestotliwosc.py` (E4, E8), `entropia.py` (E5), `kramers_lasery.py` (E6); E7 jako nowa funkcja w `outputs/porownanie_szklo_20261010/dh.py`.
  Każdy skrypt: wejście = katalog biegu, wyjście = jeden JSON + mały npz; liczy na Edenie, lokalnie tylko rysunki.

---

## 3. Eksperymenty

Wspólne oznaczenia: `x_i(t) = z_i(t) − z̄_i` (położenie tokenu względem średniej w oknie albo średniej kroczącej, zawsze w
blokach NIEnakładających się, bo średnia krocząca sama tworzy pamięć — pamięć langevin-koopman); `C = ⟨x xᵀ⟩`;
`C(τ) = ⟨x(t+τ) x(t)ᵀ⟩`; rzut na k pierwszych kierunków PCA przemieszczeń danego tokenu (k = 2, 4, 8), bo pełne 128D na token
to za dużo parametrów na 1094 kroki.

### E1. Kalibracja sztywności pułapki pełnym zestawem metod

**Pytanie.** Jaka jest sztywność pułapki tokenu i czy różne metody kalibracji dają to samo? Zgodność metod to test, czy ruch
tokenu jest procesem Ornsteina–Uhlenbecka (OU), czyli czy opis „koloid w harmonicznej pułapce” w ogóle pasuje.

**Odpowiednik fizyczny.** Kalibracja pęsety optycznej: ekwipartycja, widmo mocy z częstotliwością załamania, wariancja Allana,
wnioskowanie bayesowskie z dokładnej wiarygodności OU [Gie20, Ber04b, Lan12, Ber16]; wielowymiarowy OU [Sin17c].

**Obserwable.**
1. Ekwipartycja: `K/T = C⁻¹` (tensor; sama fluktuacja daje tylko stosunek sztywności do temperatury).
2. Widmo mocy w kierunkach głównych: dopasowanie Lorentza `S(f) = D / (π²(f_c² + f²))` metodą największej wiarygodności,
   nie najmniejszych kwadratów (obciążenie amplitudy [Nrr09]); `κ/γ = 2π f_c`.
3. Wariancja Allana `σ²(τ)` i jej dopasowanie do wzoru OU [Lan12].
4. Wielowymiarowy OU z kroków: `x(t+1) = Φ x(t) + η`, `Φ = exp(−A)`, `A = −log Φ`, dyfuzja z równania Lapunowa
   `A C + C Aᵀ = 2D`. Rozkład `A = A_s + A_a` (część symetryczna = sprężyna, antysymetryczna = obrót).
5. Wpływ próbkowania (aliasing, rozmycie [Won06, Per23c]): te same estymatory na danych co 1, 10 i 100 kroków (podpróbkowanie
   gęstego okna). Ile przekłamuje zapis co 100 kroków, którego używaliśmy w ruch60?

**Dane.** G1 (zimno), G2 (gorąco), G3a/b (seria lr), istniejące z48/z49_zostaw, cnn_zostaw, bezpoz.

**Kontrole.** Surogat OU o tych samych C i A (symulacja) przez te same estymatory: obciążenie metod. Tasowanie kroków w czasie:
widmo płaskie, f_c znika.

**Wynik.** Tabela per bieg i faza: mediana f_c, `κ/T` (ekwipartycja), zgodność metod (stosunek), udział `‖A_a‖/‖A‖`,
obciążenie przez próbkowanie. **Tak, OU pasuje**: metody zgodne w granicach ~10–20 % jak w pęsecie [Ber16]. **Nie pasuje**: widmo
nie jest Lorentzem (np. potęgowe, jak `k^−2,7` fononów), wtedy opis „harmoniczna pułapka + biały szum” odpada i trzeba pamięci (pęd AdamW).

**Zasoby.** CPU, minuty na bieg. `ou_kalibracja.py`.

### E2. Pole sił z ruchu kontra pole dokładne (test niemożliwy w laboratorium)

**Pytanie.** Fizyk musi odtwarzać pole sił z samych trajektorii. My znamy dokładny dryf i dyfuzję w każdym zapisanym stanie.
Ile z prawdziwego dryfu tokenu da się wyjaśnić jako funkcję **samego położenia tokenu** (obraz „pułapka”), a ile zależy od
reszty sieci (wspólne wagi, inne tokeny)? To bezpośredni test, czy obraz koloidu w pułapce jest adekwatny.

**Odpowiednik fizyczny.** FORMA: estymator największej wiarygodności pola sił z przemieszczeń, z rozkładem na część zachowawczą
i niezachowawczą oraz punkty stabilne i niestabilne [Gar18c]; pełna mapa pola i anharmoniczność [Jah11, Vol07].

**Obserwable.**
1. FORMA na przemieszczeniach tokenu: dryf `F̂(x)` jako funkcja liniowa (potem lokalnie liniowa w komórkach) położenia; dyfuzja `D̂`.
2. Dryf dokładny w stanie z K1: `d_i(θ_t) = z_i(θ_t + Δθ_śr) − z_i(θ_t)`, gdzie `Δθ_śr` to krok AdamW (ze stanu m, v) na
   średnim gradiencie (rodzina `srednia` z `jadro_pola.py`); dyfuzja dokładna `D_i` = kowariancja `z_i(θ+Δθ_paczka) − z_i(θ+Δθ_śr)`
   po 40 losowych paczkach (rodzina `szum`).
3. Porównanie: R² i cos między `F̂(x_i(t))` a `d_i(θ_t)`; `D̂` kontra `D_i` (stosunek śladów, kąt między kierunkami głównymi);
   część antysymetryczna Jakobianu `∂F̂/∂x` kontra ta sama z dokładnego dryfu (różnice skończone po położeniu niedostępne, więc
   porównujemy `A_a` z E1 z rotacją dokładnego dryfu wzdłuż trajektorii).
4. Punkty stałe `F̂ = 0`: czy leżą przy centrum drgań (średnia w bloku), czy pułapka „ucieka” (pamięć: od ~10 000 kroków globalne
   kurczenie, mapa ujść nietrwała).

**Dane.** G1 i G2 (potrzebne K1). Średni gradient: pełna epoka (1094 paczek) albo podzbiór 200 paczek z oszacowaniem błędu
(różnica dwóch rozłącznych podzbiorów).

**Kontrole.** FORMA na surogacie OU (górna granica R²); FORMA na trajektorii z potasowanymi przyrostami (dolna granica).

**Wynik.** Liczba „ile ruchu to pułapka”: R² dryfu wyjaśniony położeniem tokenu. Wysokie (> 0,5): obraz pułapki trzyma się
i można go używać ilościowo. Niskie (< 0,2): ruch tokenu to w większości wspólny napęd wag (zgodne z testem jądra: 85–90 %
energii kroku to pole), a „pułapka” jest tylko średnią. Oba wyniki publikowalne; drugi zamyka analogię uczciwie.

**Zasoby.** GPU. Średni gradient na stan ~1–2 min na H200 (forward + backward na 35 000 klatkach), 44 stany ≈ 1–1,5 h; ograniczyć
do 20 stanów albo podzbioru paczek. `forma_pole.py` + rodziny `srednia`/`szum` z `jadro_pola.py`.

### E3. Wir w ruchu: prądy prawdopodobieństwa i asymetria korelacji

**Pytanie.** Czy token w czasie uczenia krąży (wir Browna), czy tylko drga? Czy wir w ruchu ma związek z wirem w Jakobianie
laserów tego samego tokenu?

**Odpowiednik fizyczny.** Wir Browna: krążący prąd prawdopodobieństwa w niezachowawczym polu [Brownian vortexes, doi
10.1103/physreve.80.010401]; „fontanna prawdopodobieństwa” [Roi08]; ilościowa ocena sił niezachowawczych [Pes09b, Wu09b];
wykrywanie momentu siły z asymetrii korelacji [Vol06]; prądy w przestrzeni fazowej [Battle 2016, doi 10.1126/science.aac8167];
prądy w zawiesinach napędzanych optycznie [Tha23c].

**Obserwable.**
1. Asymetria korelacji krzyżowej w płaszczyźnie (x, y) dwóch pierwszych kierunków PCA tokenu:
   `Δ_xy(τ) = ⟨x(t+τ) y(t)⟩ − ⟨y(t+τ) x(t)⟩`; w równowadze ≡ 0.
2. Tempo pola stochastycznego (pole zakreślone na krok): `Ȧ = ½⟨x ẏ − y ẋ⟩`, znak = kierunek obrotu.
3. Mapa prądu: średnia prędkość w komórkach siatki 2D (gruboziarniste prądy jak w Battle 2016), cyrkulacja po pętli.
4. Most do laserów: dla tych samych tokenów udział wiru w Jakobianie głębokości bloku 6 (`‖J_a‖²/‖J‖²`, skrypt temperatury
   pułapki s47) i udział obrotu w ruchu (`‖A_a‖/‖A‖` z E1, `Ȧ` z pkt 2). Korelacja Spearmana po tokenach.
5. Zależność od temperatury: G3a / G2 / G3b. W fizyce kierunek wiru Browna może się odwrócić ze zmianą temperatury
   (Moy15, Sun09 z poprzedniego przeszukania); czy u nas Ȧ zmienia znak albo rośnie z lr?

**Kontrole.** Odwrócenie czasu (Δ_xy zmienia znak, Ȧ → −Ȧ; na surogacie z odwróceniem czasu w danych z równowagi Δ ≈ 0).
Surogat OU z symetrycznym A (równowaga) przez te same estymatory: poziom zerowy. Rzut: prąd może zniknąć w rzucie, którego nie
widać w pełnym stanie i odwrotnie [doi 10.1038/s41467-019-11051-w], więc liczyć w kilku płaszczyznach (PC1–2, PC1–3, losowe).

**Wynik.** Tak: `Ȧ` istotnie ≠ 0 dla większości tokenów (test znaków na surogacie) i korelacja z wirem laserów > 0,3.
Wtedy siodło z wirem z laserów naprawdę rządzi ruchem w uczeniu. Nie: krążenie na poziomie surogatu; wtedy wir laserów jest cechą
pola w głębokości, której ruch w uczeniu nie dziedziczy (pamięć: lokalna wirowość = surogat w Koopmanie pułapek, obrót całej
chmury 2–3× surogat, więc spodziewamy się obrotu zbiorowego, nie lokalnego; E3 to rozstrzyga na danych co 1 krok).

**Zasoby.** CPU. `prady_wir.py`. Dane: wszystkie biegi z porcjami (2a) + G1–G3.

### E4. Temperatura zależna od częstotliwości i rozpraszanie (Harada–Sasa)

**Pytanie.** Czy jest jedna temperatura na wszystkich skalach czasu, czy szybkie drgania i wolny dryf mają różne temperatury?
Ile „energii” rozprasza ruch tokenu?

**Odpowiednik fizyczny.** Uogólnione relacje fluktuacyjno-dyssypacyjne w stanie stacjonarnym nierównowagowym [Meh10, Gom09];
równość Harady–Sasy: rozpraszanie = całka z naruszenia FDT [Har05]; nieinwazyjny pomiar rozpraszania [Lan12b]; FDT w starzejącym
się szkle koloidalnym [doi 10.1103/physrevlett.98.108302].

**Obserwable.**
1. Funkcja odpowiedzi `R(t)` z protokołu impuls (K3) i podatność `χ(t)` ze skoku (już istnieje).
2. Widmo fluktuacji `C̃(ω)` z bliźniaka h = 0 (ta sama paczka, bez pola).
3. Temperatura efektywna `T_eff(ω) = ω C̃(ω) / (2 χ''(ω))`: stała = jedna temperatura; zależna od ω = kilka.
4. Tarcie `γ` z krótkoczasowej odpowiedzi na stałą siłę (`χ(t) ≈ t/γ`), potem Harada–Sasa:
   `rozpraszanie = γ { ⟨v⟩² + ∫ [C̃_v(ω) − 2T R̃'_v(ω)] dω/2π }` (prędkości = przyrosty na krok). Założenie: dynamika
   przetłumiona z jednym γ; sprawdzić je w danych (czy `χ(t)` jest liniowe dla małych t).

**Dane.** Repliki od stanów K1 z G1 (zimno) i G2 (gorąco, krok ~1000): pola ±h0, ±3h0, impuls i skok, bliźniak h = 0,
replika z innym strumieniem paczek. 12 sond, zapis liczb na token jak w `fdt_odpowiedz.py`.

**Kontrole.** Bliźniak h = 0 (niedeterminizm ≪ odpowiedź; próba 1 z 2026-10-13 pokazała, że bez `use_deterministic_algorithms`
niedeterminizm ≈ odpowiedź). Liniowość: χ z 3h0 wobec h0.

**Wynik.** Krzywa `T_eff(ω)`. Przewidywanie z 2026-10-13 (koloid nie spełnia Einsteina, pułapka spełnia): `T_eff` duża przy
niskich ω (wolny dryf, starzenie), zbiega do T pułapki (~3,6·10⁻⁴) przy wysokich ω. Harada–Sasa daje jedną liczbę na fazę uczenia.

**Zasoby.** GPU, jak `fdt3.slurm` (repliki równolegle na jednej karcie), ~1–2 h na stan. `fdt_odpowiedz.py` + K3 + `fdt_czestotliwosc.py`.

### E5. Produkcja entropii i złamana równowaga szczegółowa

**Pytanie.** Jak bardzo nieodwracalny jest ruch tokenów? Czy nieodwracalność maleje, gdy sieć stygnie?

**Odpowiednik fizyczny.** Entropia wzdłuż trajektorii i twierdzenia fluktuacyjne [Sei05, Sei12, Cil17, Mar25j]; termodynamika
w nieharmonicznym potencjale [Bli05]; Landauer [Ber15h]; złamana równowaga szczegółowa z prądów [Battle 2016]; estymatory z samych
trajektorii: krótkoczasowy z relacji niepewności termodynamicznej (TUR, np. doi 10.48550/arxiv.2204.09283) i neuronowy NEEP
(doi 10.48550/arxiv.2003.04166).

**Obserwable.**
1. Liniowy (z E1): prąd nieodwracalny `Ω = A − D C⁻¹` (zero ⇔ równowaga szczegółowa) i tempo produkcji entropii
   `σ = tr(Ωᵀ D⁻¹ Ω C)` (w jednostkach k_B na krok).
2. TUR: dolna granica `σ ≥ 2⟨J⟩² / Var(J)` dla prądu J = pole stochastyczne z E3 i dla prądów w kilku płaszczyznach.
3. NEEP: sieć szacująca σ z par (x_t, x_{t+1}); trenować na połowie tokenów, testować na drugiej.
4. Zależność od fazy (gorąco G2 kontra zimno G1), od lr (G3), od ziarna (z48, z49), od architektury (CNN) i od pozycji (bezpoz).

**Kontrole.** Odwrócona w czasie trajektoria równowagowego surogatu (σ musi wyjść ≈ 0: poziom obciążenia estymatora).
Estymatory 1–3 muszą się zgadzać co do rzędu wielkości.

**Wynik.** σ dla każdej fazy i każdego pokrętła. Przewidywanie: σ > 0 wszędzie, większe w gorącej fazie i przy większym lr.
Jeśli σ ≈ 0 w zimnej fazie, ruch w późnym uczeniu jest w praktyce drganiem równowagowym wokół wolno pełzającej pułapki.

**Zasoby.** CPU (1–2) i GPU krótko (3). `entropia.py`.

### E6. Przeskoki laserów jak ucieczka Kramersa

**Pytanie.** Gdy token przechodzi przez płaszczyznę celownika lasera c, laser się przełącza (zapala albo gaśnie). Czy te
przełączenia zachowują się jak termicznie aktywowane przeskoki przez barierę, i jak ich częstość zależy od temperatury?

**Odpowiednik fizyczny.** Termicznie aktywowane przejścia w bistabilnej pułapce [Mcc99]; dynamika ścieżek przejścia [Zij19b];
rezonans stochastyczny i twierdzenie fluktuacyjne w podwójnej studni (doi 10.1209/0295-5075/81/50005).

**Obserwable.** Z pre-aktywacji K2 (`a_ci(t)`, blok 6):
1. Rozkład czasów przebywania w stanie „zapalony” (`a > 0`) i „zgaszony”; wykładniczy = proces Markowa z jedną częstością.
2. Częstość przełączeń `k_ci` kontra wzór Rice'a dla procesu gaussowskiego `ν = (1/2π)·√(−ρ''(0))·exp(−ā²/(2 s²))`
   (ā i s to średnia i odchylenie a_ci w oknie; ρ autokorelacja). Wykres Arrheniusa: `log k` od `ā²/(2s²)`.
3. Związek z temperaturą bramek: okno cieplne |a| < 1,28 (Fermi, s47) kontra częstość przełączeń; zimno (G1) kontra gorąco (G2), seria lr.
4. Ścieżki przejścia: czas trwania przejścia przez okno cieplne; czy przełączenie jednego lasera pociąga przełączenia innych (lawiny)?

**Kontrole.** Surogat gaussowski o tej samej autokorelacji (wzór Rice'a dokładny). Tasowanie w czasie.

**Wynik.** Jeśli częstości trzymają się Rice'a/Arrheniusa: przełączanie laserów to „chemia” z barierą, a stygnięcie bramek
działa jak zamrażanie reakcji. Odchyłki w górę (lawiny, przełączenia skorelowane) wskazują na zbiorowe przestawianie kodu
kreskowego tokenu.

**Zasoby.** CPU, pamięć: 2 GB pre-aktywacji na bieg. `kramers_lasery.py`.

### E7. Sprzężenie par po odjęciu wspólnego napędu (niski priorytet)

**Pytanie.** Wiadomo, że sąsiednie tokeny ruszają się razem (C(0) 0,63–0,84), bo mają wspólny napęd. Czy po odjęciu napędu
zostaje sprzężenie z opóźnieniem jak hydrodynamiczne?

**Odpowiednik fizyczny.** Korelacje krzyżowe dwóch cząstek w pułapkach: przy sprzężeniu hydrodynamicznym C(0) = 0 i ujemne
minimum z opóźnieniem [Mei99]; spektroskopia korelacji krzyżowej (doi 10.1098/rsta.2000.0805).

**Obserwable.** Reszty przemieszczeń po regresji na wspólne składowe (średnia zdjęcia, średnia globalna, pierwsze składowe przyrostu
wag; rada z przeszukania „Diagnosing the dynamical class of trapped particles with common drive”, 2026-10-11). Korelacja krzyżowa
reszt par sąsiadów i par losowych w funkcji opóźnienia τ; położenie i głębokość minimum.

**Wynik.** Minimum z opóźnieniem = sprzężenie przenoszone przez wagi z czasem odpowiedzi (odpowiednik hydrodynamiki). Brak = czysty
wspólny napęd. Uwaga: `eden/wspolny_naped_wagi.py` ma niewyrównany czasowo przyrost wag (dTh[idx[:-1]]); napisać od nowa.

### E8. Mikroreologia: sprężystość i lepkość „ośrodka wag”

**Pytanie.** Czy ośrodek, w którym pływa token, reaguje jak ciecz lepka, ciało sprężyste czy coś pośredniego, i czy to się zmienia
z wiekiem uczenia?

**Odpowiednik fizyczny.** Mikroreologia pasywna (z fluktuacji) i aktywna (z wymuszenia) [Tas12, Bra07]; zgodność obu metod jako
test FDT w starzejącym się szkle koloidalnym (doi 10.1103/physrevlett.98.108302).

**Obserwable.**
1. Pasywna: z MSD przez uogólnione Stokesa–Einsteina `G*(ω)`, potem `G'(ω)` (sprężystość) i `G''(ω)` (lepkość).
2. Aktywna: repliki z polem sinusoidalnym (K3) o okresach 10, 30, 100 i 300 kroków, zespolona podatność `χ(ω)`, `G* ∝ 1/χ(ω)`.
3. Porównanie 1 i 2: zgodne = FDT w dziedzinie częstotliwości (łączy się z E4).

**Wynik.** Częstość przecięcia `G' = G''`: skala czasu, powyżej której token „widzi” ośrodek jako sprężysty. Przesunięcie tej
częstości między G2 a G1 = ilościowa miara starzenia.

**Zasoby.** GPU jak E4 (4 okresy × 2 znaki pola × 2 stany).

---

## 4. Kolejność, zależności, budżet

| etap | co | zależy od | GPU | dysk Eden |
|---|---|---|---|---|
| 0 | K1, K2, K3 + próba (`--proba`, 512 klatek) + bliźniak h = 0 | — | 10 min | < 1 GB |
| 1 | G1, potem E1, E3, E5 na G1 i na istniejących biegach „zostaw” | 0 | 2 min | 6 GB |
| 2 | E2 na G1 (dryf dokładny) | 1 | ~1 h | +1 GB |
| 3 | G2, G3a, G3b; E1, E3, E5, E6 na nich | 0 | 30 min | 3 × 8 GB, kolejno |
| 4 | E4 i E8 (repliki) na stanach G1 i G2 | K3, 1, 3 | 4–8 h | < 2 GB (liczby na token) |
| 5 | E7, rysunki, raport `WYNIKI.md`, wpis do `ZASADY_LASER_KOLOID.md` (nowa zasada) | 1–4 | — | — |

Najpierw etap 1: jest najtańszy, korzysta z istniejących danych i odpowiada na pytanie, czy w ogóle jest sens iść dalej
(jeśli E1 pokaże, że ruch nie jest OU nawet w przybliżeniu, E4 i E8 trzeba zaprojektować z pamięcią).

## 5. Zasady Edena (z pamięci projektu)

- Katalog: `~/smallvit_runs/2026-10-14_pomiary_pulapki/{kod,logi,wyniki}`; kod kopiować z `2026-10-10_szklo_pokretla/kod`
  i `2026-10-13_fdt_temperatura` plus nowe skrypty.
- `sbatch --partition=hopper-2 --gres=gpu:1` (H200 MIG; gdy zajęte: `--partition=hopper-2,hopper,short`).
- Zmiennych NIE przekazywać przez `sbatch --export` (zadanie ginie po 2 s); tylko przez środowisko (`ZMIENNA=... sbatch ...`).
- Pliki `.slurm` z Windows tylko z końcami linii LF (CRLF → „DOS line breaks”).
- venv i PYTHONPATH jak w biegach pokręteł (`~/venvs/...`, deps biegu 35 000; timm 0.9.16); TF32 wyłączone w różnicach skończonych,
  `use_deterministic_algorithms(True)` w replikach, fused SDPA timm wyłączone.
- Pobieranie: strumień `ssh eden-cluster 'tar cf - ...' | tar xf -`, nie listy scp. Usuwanie na Edenie tylko po pobraniu i zgodnym
  MD5 każdego pliku (`md5sum -c`), wpis do `~/smallvit_runs/USUNIETE_Z_EDENU.txt` i do rejestru sesji.
- Wyniki lokalnie: `SmallViTAblation/experiments/pomiary_pulapki_fizyka_20261014/wyniki/` (JSON, małe npz); duże surowe dane
  do `assets/` z README.

## 6. Literatura (DOI)

Undermind cite keys z workspace ViT (`40294fa6-c9a3-4469-9a61-d3eddeaee893`), przeszukanie z 2026-10-14:

| klucz | praca | DOI |
|---|---|---|
| Gie20 | Optical tweezers — from calibration to applications: a tutorial | 10.1364/aop.394888 |
| Ber04b | Power spectrum analysis for optical tweezers | 10.1063/1.1645654 |
| Per23c | Optimal calibration of optical tweezers with arbitrary integration time and sampling frequencies | 10.1364/BOE.495468 |
| Lan12 | Power spectrum and Allan variance methods for calibrating single-molecule video-tracking instruments | 10.1063/1.3687431 |
| Ber16 | Fast Bayesian inference of optical trap stiffness and particle diffusion | 10.1038/srep41638 |
| Sin17c | Fast Bayesian inference of the multivariate Ornstein-Uhlenbeck process | (Undermind, bez DOI w wyniku) |
| Won06 | The effect of integration time on fluctuation measurements: calibrating an optical trap in the presence of motion blur | 10.1364/OE.14.012517 |
| Nrr09 | Power spectrum analysis with least-squares fitting: amplitude bias and its elimination | (Undermind) |
| Luk07 | Motion of a colloidal particle in an optical trap | 10.1103/PHYSREVE.76.011112 |
| Gar18c | High-performance reconstruction of microscopic force fields from Brownian trajectories (FORMA) | 10.1038/s41467-018-07437-x |
| Jah11 | Measuring the complete force field of an optical trap | 10.1364/OL.36.001260 |
| Vol07 | Brownian motion in a nonhomogeneous force field and photonic force microscope | 10.1103/PhysRevE.76.061118 |
| Roi08 | Influence of nonconservative optical forces …: the fountain of probability | 10.1103/PhysRevLett.101.128301 |
| Pes09b | Quantitative assessment of non-conservative radiation forces in an optical trap | 10.1209/0295-5075/86/38002 |
| Wu09b | Direct measurement of the nonconservative force field generated by optical tweezers | 10.1016/j.bpj.2008.12.1430 |
| Vol06 | Torque detection using Brownian fluctuations | 10.1103/PHYSREVLETT.97.210603 |
| Tha23c | Nonequilibrium probability currents in optically-driven colloidal suspensions | 10.21468/SciPostPhys.17.4.096 |
| Har05 | Equality connecting energy dissipation with a violation of the fluctuation-response relation | 10.1103/PhysRevLett.95.130602 |
| Meh10 | Experimental accessibility of generalized fluctuation-dissipation relations for NESS | 10.1103/PhysRevE.82.032401 |
| Gom09 | Experimental verification of a modified fluctuation-dissipation relation … NESS | 10.1103/PhysRevLett.103.040601 |
| Lan12b | Noninvasive measurement of dissipation in colloidal systems | 10.1103/PhysRevE.86.030401 |
| Sei05 | Entropy production along a stochastic trajectory and an integral fluctuation theorem | 10.1103/PhysRevLett.95.040602 |
| Sei12 | Stochastic thermodynamics, fluctuation theorems and molecular machines | 10.1088/0034-4885/75/12/126001 |
| Cil17 | Experiments in Stochastic Thermodynamics: Short History and Perspectives | 10.1103/PHYSREVX.7.021051 |
| Mar25j | Fluctuation theorems with optical tweezers: theory and practice | 10.1140/epjp/s13360-025-07181-4 |
| Bli05 | Thermodynamics of a colloidal particle in a time-dependent nonharmonic potential | 10.1103/PhysRevLett.96.070603 |
| Ber15h | Information and thermodynamics: experimental verification of Landauer's erasure principle | 10.1088/1742-5468/2015/06/P06015 |
| Mcc99 | Thermally activated transitions in a bistable three-dimensional optical trap | 10.1038/45492 |
| Zij19b | Transition path dynamics of a dielectric particle in a bistable optical trap | 10.1103/PhysRevLett.125.146001 |
| Mei99 | Direct measurement of hydrodynamic cross correlations between two particles in an external potential | 10.1103/PHYSREVLETT.82.2211 |
| Tas12 | Microrheology with optical tweezers: data analysis | 10.1088/1367-2630/14/11/115032 |
| Bra07 | Passive and active microrheology with optical tweezers | (Undermind) |

Ze Scite (bez klucza Undermind): Brownian vortexes, 10.1103/physreve.80.010401; Broken detailed balance at mesoscopic scales in
active biological systems, 10.1126/science.aac8167; Inferring broken detailed balance in the absence of observable currents,
10.1038/s41467-019-11051-w; Fluctuation-dissipation theorem in an aging colloidal glass, 10.1103/physrevlett.98.108302;
Measurement of the hydrodynamic forces between two polymer-coated spheres, 10.1098/rsta.2000.0805; Work and dissipation fluctuations
near the stochastic resonance of a colloidal particle, 10.1209/0295-5075/81/50005; Inferring entropy production in anharmonic
Brownian gyrators, 10.48550/arxiv.2204.09283; Learning entropy production via neural networks, 10.48550/arxiv.2003.04166.
