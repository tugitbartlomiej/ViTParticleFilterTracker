# Predykcje zapisane przed pomiarem (2026-10-27, 02:49, Europe/Warsaw)

Zapisane przed napisaniem kodu pomiaru, przed testem lokalnym i przed uruchomieniem jakiegokolwiek zadania tego eksperymentu na Edenie.
Progi nie będą zmieniane po pomiarze; uwagi po fakcie będą oznaczone „(po fakcie)”. Miary zdefiniowane niżej i w docstringu
`Kod_eksperymentu/znak_normy_eden.py`; ocena w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` (napisana przed odczytem wyników).

## Pytanie

Rodzic `odbiorca_zapisu_mlp_uwaga_20261027__01-55-25` (wstępnie, 4 z 5 biegów): kierunek zapisu wzmacnianego modu MLP u₁ nie ma
wyróżnionego odbiorcy (uwaga, MLP i głowa czytają go 1,7–4× losowe), ale 20–26 % jego energii (detekcja; CIFAR 8 %, losowe 0,7 %) leży
w jądrze LayerNorm następnego bloku, span{1, x̂}. To jądro ma **dwa różne kierunki**:

- **średnia** 1̂ = 1/√C: dodanie stałej do tokenu nie zmienia niczego, co czyta jakakolwiek dalsza część sieci (każdy odbiorca jest za LN,
  także głowa za `net.norm`) — kanał „kosza”;
- **radialny** ê = P x / ‖P x‖ (P = centrowanie): zmienia skalę ρ = ‖P x‖ tokenu. LN ją usuwa, ale każda późniejsza aktualizacja Δ obraca
  token o kąt ∝ ‖Δ‖/ρ, więc ρ jest **bezwładnością** tokenu dla całej reszty sieci — kanał „regulacji wzmocnienia”.

Pytania: (1) który z dwóch kierunków niesie te 20 %; (2) jaki jest **znak**: gdy token jest pchnięty w kierunku odczytu (bardziej
„narzędzie” / bardziej poprawna klasa), czy MLP zwiększa, czy zmniejsza jego normę; (3) czy kanał normy **wzmacnia** (zatrzask, dodatnie
sprzężenie) czy **osłabia** (stabilizator, ujemne sprzężenie) skutek tego pchnięcia dla wyjścia sieci; (4) czy różni się na narzędziu i tle.

## Miary (blok b = 0…NB−1, stan x_a = wejście MLP bloku b, x_c = x_a + MLP(x_a) = stan po bloku b)

- J = jakobian pola MLP w x_a (jak w rodzicach), para osobliwa (u₁, v₁, σ₁) przez eigh JᵀJ (jak w rodzicu). ê = ê(x_c), 1̂ = 1/√C.
- **Energia:** E_rad(d) = ⟨d, ê⟩², E_mean(d) = ⟨d, 1̂⟩² dla d ∈ {u₁, losowy}; losowe 1/192 = 0,0052 każde. Wszystkie wybrane tokeny.
- **Odczyt:** r_a = ∂y/∂x_a, r_c = ∂y/∂x_c (autograd przez resztę sieci), y = wyjście tokenu: detekcja — logit ciepła tego tokenu;
  CIFAR — margines logit[y] − max logit[inne]. 8 tokenów na klatkę; detekcja: do 4 tokenów narzędzia, reszta tło.
- **Orientacja:** dla kierunku pchnięcia d ∈ {v₁, r̂_a, losowy} znak s = sign⟨d, r_a⟩, tak by pchnięcie +s·d zwiększało y.
- **Znak liniowy:** c_rad(d) = s·⟨ê, J d⟩ / ‖J d‖ (cosinus odpowiedzi MLP z kierunkiem radialnym; > 0: norma rośnie przy pchnięciu
  w stronę odczytu); η(d) = s·⟨ê, J d⟩ · ρ(x_a)/ρ(x_c) (elastyczność normy).
- **Rozkład skutku (liniowo):** F_rad^lin(d) = s·⟨r_c, ê⟩⟨ê, J d⟩ / (s·⟨r_a, d⟩), F_perp^lin(d) = s·⟨r_c, P_⊥ J d⟩ / (s·⟨r_a, d⟩),
  P_⊥ = rzut prostopadły do {ê, 1̂}: udział kanału normy i kanału kierunku w całej liniowej zmianie y.
- **Rozkład skutku (nieliniowo, główny):** pchnięcie x_a += δ·skala·s·d jednego tokenu (δ ∈ {0,03; 0,1}, skala = mediana normy tokenu
  w klatce), Δm = MLP(x_a+) − MLP(x_a), rozkład Δm = a·ê + μ·1̂ + Δm_⊥; warianty stanu tokenu po bloku b: pełny, bez a·ê (bez_rad),
  bez μ·1̂ (bez_mean), bez Δm_⊥ (bez_perp), bez Δm (bez_mlp); każdy przez resztę sieci (ręcznie, float64), Δy = y − y₀.
  F_X(d, δ) = mediana_tokenów(Δy_pełny − Δy_bez_X) / mediana_tokenów|Δy_pełny|, mediana po blokach. F_rad < 0: kanał normy przeciwdziała
  pchnięciu (stabilizator); F_rad > 0: wzmacnia je (zatrzask).
- **Opisowo:** rzeczywista aktualizacja MLP: cos(m₀, ê(x_a)), wzrost normy przez MLP ρ(x_c)/ρ(x_a) − 1 i przez uwagę; profil ρ po blokach;
  wszystko osobno narzędzie / tło.

Statystyka M[·]: **mediana po blokach z median po tokenach i klatkach**, na ostatnim checkpoincie (koniec), obok k0 i 3000.

## Biegi i próg

5 biegów jak w rodzicu: detekcja s3047 / s4047 / s5047 (k0, 3000, 65 640), MałyViT CIFAR s47 i DeiT CIFAR s47 (k0, 3000, 93 780).
PASS: warunek w ≥ 4/5 biegów (5/5 „silnie”, p = 0,031; 4/5 „słabo”, p = 0,19). Testy tylko detekcji: 3/3. Kontrole K: 5/5.

## Predykcje

| nr | predykcja | warunek liczbowy (koniec, M[·], δ = 0,03 gdzie nieliniowo) | próg |
|---|---|---|---|
| R1 | jądro LN niesie norma, nie średnia | M[E_rad(u₁)] > M[E_mean(u₁)] | ≥ 4/5 |
| R2 | udział radialny jest nieprzypadkowy i wyuczony | M[E_rad(u₁)] > 3 × M[E_rad(los)] **i** > wartość przy k0 | ≥ 4/5 |
| Z1 | znak jest spójny | M[c_rad(v₁)] ma ten sam znak w biegach **i** w każdym z nich > 60 % tokenów ma ten znak | ≥ 4/5 |
| Z2 | pchnięcie w stronę odczytu zwiększa normę (hipoteza kierunkowa) | M[c_rad(r̂_a)] > 0 | ≥ 4/5 |
| S1 | kanał normy stabilizuje (ujemne sprzężenie, odpowiednik hamowania) | M[F_rad(r̂_a)] < 0 **i** M[F_rad(v₁)] < 0 | ≥ 4/5 |
| S2 | kanał normy jest funkcjonalnie istotny | M[\|F_rad(r̂_a)\|] / (M[\|F_rad(r̂_a)\|] + M[\|F_perp(r̂_a)\|]) > 0,1 | ≥ 4/5 |
| T1 | na narzędziu kanał normy jest słabszy niż na tle (MLP prawie wyłączony) | M[\|η(r̂_a)\|]: narzędzie < tło | 3/3 |
| K1 | zgodność z rodzicem | M[E_rad(u₁) + E_mean(u₁)] w ± 0,05 od Enull_u1 rodzica dla tego biegu | 5/5 |
| K2 | niezmienniczość względem średniej (kontrola kodu) | max \|⟨r_c, 1̂⟩\| / ‖r_c‖ < 10⁻⁶ **i** max\|F_mean\| < 10⁻⁶ | 5/5 |
| K3 | zgodność liniowego i nieliniowego rozkładu | znak M[F_rad^lin(r̂_a)] = znak M[F_rad(r̂_a), δ = 0,03] | ≥ 4/5 |

Uzasadnienie kierunku Z2 i S1 (zapisane z góry, może być błędne): na tle MLP pracuje, na narzędziu jest prawie wyłączony (rodzic
balanced_amplification, BA5b); wyjście sieci jest odporne na wzmocnienie przejściowe (rodzic wzmocnienie_przejsciowe, ΔIoU ≤ 0,013).
Spójne z tym jest, że pchnięcie tła w stronę „narzędzia” zwiększa normę tokenu, czyli jego bezwładność, i tym samym osłabia dalsze
przesunięcia — normalizacja dzieląca jako odpowiednik stabilizacji hamowaniem w korze (Murphy & Miller 2009; Carandini & Heeger 2012).

## Reguła rozstrzygania (zapisana z góry)

- **Kanał średniej („kosz”):** R1 FAIL (E_mean ≥ E_rad). Wtedy wcześniejsza interpretacja „MLP reguluje bezwładność tokenu” jest
  błędna dla tej części zapisu; F_rad i Z oceniane opisowo.
- **Stabilizator (normalizacja dzieląca):** R1 ∧ S1 ∧ S2 PASS. Zdanie do artykułu: MLP wzmacnia sygnał w kierunku odczytu i jednocześnie
  zwiększa normę tokenu, co osłabia skutek tego samego sygnału dla wyjścia — wzmocnienie stabilizowane normalizacją.
- **Zatrzask (dodatnie sprzężenie):** R1 ∧ S2 PASS, F_rad > 0 w ≥ 4/5. Zdanie: kanał normy utrwala decyzję tokenu.
- **Kanał nieistotny:** R1 PASS, S2 FAIL — norma się zmienia, ale nie wpływa na wyjście w sposób znaczący.
- Detekcja i CIFAR mogą się różnić (rodzice: inny typ odczytu); wtedy wniosek osobno, bez uogólniania.
