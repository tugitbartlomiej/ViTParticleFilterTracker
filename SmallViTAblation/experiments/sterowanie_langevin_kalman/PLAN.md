# Sterowanie uczeniem przez dobór danych: Koopman + Langevin + Kalman + FastGAN na stanie dekodera

Data: 2026-09-28. Status: **G1 i G2 niespełnione lokalnie na Nano (sekcja „Wynik bramek”)**; nic nie zgłoszone na Edenie.

Polecenie użytkownika: połączyć równanie Koopmana i równanie Langevina z poprzedniej pracy (dryf, pamięć, wymuszenie,
sterowanie, szum; na błędzie liczony Kalman i odszumianie, teraz także FastGAN) w jedną metodę doboru klatek, która
przyspiesza uczenie detektora — „to musi zadziałać”.

## Co wiemy przed projektem (pomiary, bez nowego GPU)

Z `../petla_od_poczatku_large/DIAGNOZA_PORAZKI.md` (commit 5acea33):
- latent kręgosłupa DINOv2 prawie się nie uczy (wagi +0,026 % w 6 epokach, lr_encoder 1,67e-6) i nie wiąże się
  z mAP między pętlami (ρ = 0,07);
- prognoza Koopmana z historii kręgosłupa jest gorsza niż „brak zmiany” w 12/12 przypadków (przestrzela wielkość,
  kierunek trafia z korelacją 0,81–0,94) — brak członu hamowania;
- wybór wokół 3 kotwic skupia klatki (cos 0,66–0,68 vs 0,62, 2× więcej sąsiednich klatek) i przegrywa z losowym.

Z `../dekoder_sondy_large` (`odczyt_dekodera.py`, `analiza_dekodera.py`, ten commit) — stan **dekodera** na 72 sondach,
wagi k00..k06 pętli koopman, 3 ziarna:
- jakość wykrycia sondy q = max(pewność × IoU) po ostatniej warstwie dekodera rośnie przez wszystkie 6 obrotów
  (0,24 → 0,37 → 0,44–0,47), kręgosłup staje po k02; związek z mAP w czasie (wycentrowane w ziarnie): dekoder
  ρ = 0,86, kręgosłup ρ = 0,41;
- stan dekodera ciągle się rusza (0,35–0,44 rozrzutu sond na obrót także w k03–k06);
- **niedomknięta tożsamość jest widoczna w detekcji**: jakość wg wariantu w k06 — oryginał 0,57, jasne narzędzie
  0,65, skrócone 0,58, obrót 90° 0,40, żółte tło 0,30, czarne tło 0,18. Sieć nie rozpoznaje tego samego narzędzia
  na zmienionym tle i po obrocie; luka maleje z uczeniem, więc da się ją domykać;
- stan ukryty wybranego zapytania nie jest niezmienniczy (R ≈ 1,5; koduje też położenie i kształt ramki), więc
  latentem nie jest sam wektor zapytania, tylko jakość wykrycia sond.

## Metoda: pętla sterowania (obserwator, model, sterownik, aktuator)

**Stan x_t** (co 5 aktualizacji): jakość wykrycia q każdej sondy po warstwach dekodera 1–3 (72 × 3) oraz IoU
propozycji przed dekoderem; do równania wchodzi zredukowany wektor: średnia q per wariant (6) i per obraz (12).
Tożsamość domknięta = q równe we wszystkich wariantach jednego obrazu i wysokie (luka wariantów → 0).

**Model (równanie Langevina z dryfem Koopmana):**

    Δx_t = −K (x_t − x*) + Σ_{j=1..m} B_j Δx_{t−j} + W f_t + C u_t + ε_t

- −K (x_t − x*): dryf Koopmana **z hamowaniem** do punktu stałego x* (OU); brak tego członu był przyczyną
  przestrzelenia w obecnej pętli;
- Σ B_j Δx_{t−j}: pamięć (w `langevin_latent_detektora` 8 przyrostów poprawiało długi rollout o 11–13 %);
- W f_t: wymuszenie harmonogramem (granica epoki, reset optymalizatora na starcie obrotu — reszty korelowały z granicą);
- C u_t: **sterowanie danymi** — u_t to skład dodanych klatek; C mierzone wprost (niżej);
- ε_t: szum mini-paczek.

**Obserwator (Kalman):** odczyt y_t = x_t + ν_t jest zaszumiony (SGD). Filtr Kalmana na tym modelu daje odszumiony
stan i jego niepewność; innowacja (odczyt − prognoza) mówi, czego model nie tłumaczy. Dryf i pamięć dopasowujemy do
trajektorii odszumionej, nie surowej (surowy szum zawyżał wielkość prognozy). Kowariancje kalibrujemy na
rzeczywistych błędach prognoz (warunek z `fastgan_narzedzie/PIPELINE_LATENT_KOOPMAN_FASTGAN.md`, sekcja 7).

**Sterownik:** co obrót wybieramy klatki u, które minimalizują przewidywaną lukę po H krokach:
J = Σ_warianty (x* − x̂_{t+H})², z priorytetem wariantów, które model przewiduje jako niedomknięte bez pomocy
(prognoza bez sterowania). Ograniczenia: różnorodność (jedna klatka na skupisko DINOv2, wykluczenie sąsiadów ±30
klatek, limit na film).

**Macierz sterowania C (wpływ klatki na stan):** pierwszy rząd — krok uczenia na klatce z przesuwa stan o
ΔX ≈ −η ⟨∇θ x, ∇θ L_det(z)⟩ (zgodność gradientu detekcji klatki z gradientem jakości sond), liczone na gradientach
dekodera i projektora (tam jest uczenie). Sprawdzane pomiarem prawdziwego kroku (bramka G3).

**Aktuator (FastGAN + wyszukiwanie):** FastGAN nie celuje już w przewidziany latent kręgosłupa, tylko tworzy
obrazy narzędzia w warunkach luki (inne tło, obrót), których wpływ C·u na stan jest największy; zapytanie do puli
prawdziwych klatek idzie po wpływie C (nie tylko po wyglądzie DINOv2), z ograniczeniem różnorodności. Wariant
tańszy do porównania: FastGAN jako generator narzędzia do wklejania na różne tła (copy-paste).

## Bramki (ustalone przed wynikiem; każda porażka = stop i raport)

- **G1 — stan przewiduje zadanie (bez uczenia, ~15 min GPU):** odczyt dekodera na istniejących wagach wszystkich
  pętli (koopman, losowe, bez; 3 ziarna × 6 obrotów, na Edenie). Warunek: między pętlami w tym samym (ziarno,
  obrót) średnia q koreluje z mAP@0,5, ρ ≥ 0,4 i p < 0,05 (kręgosłup: 0,07).
- **G2 — model bije „brak zmiany” (~4 h GPU):** jeden bieg `bez` 4 epoki od k00 z odczytem dekodera co 5 kroków
  (dopisek w `record()` w `trening.py`, 72 sondy ≈ kilka s na odczyt); dopasowanie Langevin + Kalman na początku
  trajektorii, test prognozy w przód. Warunek: błąd na 1 obrót naprzód ≤ 0,8 × „brak zmiany” i lepszy od
  samego dryfu i od Koopmana bez hamowania.
- **G3 — sterowanie jest przewidywalne (~1 h GPU):** 48 kandydatów; przewidziana zmiana stanu z C vs zmierzona po
  prawdziwym kroku na klatce (przywracanie wag). Warunek: Spearman ≥ 0,5 dla luki wariantów.
- **G4 — pętla zamknięta (kampania, ~45 h GPU):** 5 ziaren × 4 obroty × 3 ramiona: `sterowanie`, `losowe z
  wykluczeniem sąsiadów`, `augmentacja` (obrót, tło/kolor); ramię `bez` z tej samej receptury. Kryterium: pole pod
  krzywą mAP@0,5 `sterowanie` > oba punkty odniesienia w ≥ 4 z 5 ziaren i średnio ≥ +0,003; wtórnie szybsze
  domknięcie luki wariantów.

## Receptura i miejsce

Receptura jak w `petla_od_poczatku_large` (porównywalność), plus odczyt dekodera. Zmiana lr_encoder tylko jako osobne
ramię, jeśli G1 wskaże, że cała poprawa jest w dekoderze i tak ma zostać. Kwota Edenu: G4 ≈ 60 zadań × 0,59 GB ≈ 35 GB
— wymaga zgody na usunięcie starych katalogów (`2026-09-26_petla_uczenia` 6,9 GB, `2026-09-25_kontynuacja_*`)
albo wag pośrednich obrotów po ich wykorzystaniu.

## Wynik bramek G1 i G2 policzonych lokalnie na wagach Nano (2026-09-28)

Na prośbę użytkownika najpierw lokalnie, bez Edenu. Skrypty: `../dekoder_sondy_large/odczyt_dekodera_nano.py`,
`../dekoder_sondy_large/bramki_lokalne.py`; wynik: `../dekoder_sondy_large/wyniki/BRAMKI_LOKALNE.json`.
Nano ma 2 warstwy dekodera; stan = q po ostatniej.

**G1 — niespełniona.** Między ramionami w tym samym (ziarno, epoka):
- kampania 5 epok (4 ramiona × 3 ziarna × 5 epok, 60 wag): ρ = −0,07 (p = 0,61); żaden inny odczyt (średnia warstw,
  IoU propozycji, pojedyncze warianty, najsłabszy wariant) nie przekracza |ρ| = 0,14;
- pilot (3 pętle × 8 obrotów, ziarno 42): ρ = −0,51 (p = 0,01), **znak odwrotny**; najmocniej dla sond na
  czarnym i żółtym tle (ρ −0,69 i −0,73). Ramiona z dodanymi klatkami lepiej wykrywają narzędzie na zmienionym tle
  (q tła: bez 0,257, koopman 0,277, losowe 0,301), a mAP@0,5 val jest w nich minimalnie niższy (0,8989 / 0,8981 /
  0,8973).
- Skala: rozrzut mAP@0,5 między ramionami w grupie to 0,0009–0,0011 — ramiona prawie się nie różnią na val.

Wniosek: sondy mierzą rozpoznanie narzędzia w zmienionych warunkach (wycinek 256 × 256 z filmów treningowych; narzędzie wklejone na jednolite czarne i żółte tło, obrót o 90°), a val tego nie sprawdza: to 12 043 całe kadry operacji zaćmy z 25 innych filmów, zawsze z okiem w tle (`../dekoder_sondy_large/wyniki/SONDY_ARKUSZ.png`, `WAL_PROBKA.png`). Na czarnym tle detektor zwykle trafia ramką, ale z pewnością ~0,1, więc q jest niskie.
Domknięcie tożsamości i mAP@0,5 val to dwa różne cele; ρ = 0,86 na Large (w czasie, jedna pętla) był wspólnym
wzrostem obu wielkości z uczeniem, nie dowodem, że stan dekodera rozróżnia sposoby doboru danych.

**G2 — niespełniona.** Bieg gęsty 1810662 (wagi co 5 aktualizacji, kroki 5–800): q rośnie od 0 (nowa głowica
jednoklasowa) do ~0,40 w kroku ~400, potem stoi i drga (0,39–0,44). Dopasowanie na krokach ≤ 400, test h = 20
zapisów (100 aktualizacji): Koopman na przyrostach 1,00 × brak zmiany, OU 1,77 ×, Langevin z pamięcią i wymuszeniem
1,64 ×, z Kalmanem 1,64 × (model nauczony na fazie wzrostu przestrzela plateau). Kontrola po wyniku, dopasowanie na
samym plateau (kroki 300–600, test > 600): sondy 0,97–1,00 ×; średnia sond: hamowanie + pamięć 0,92 ×, sama średnia
plateau 0,82 × — nadal powyżej progu 0,8. Na plateau stan nie ma ruchu do przewidzenia; najlepsza prognoza to
uśrednienie szumu.

**Decyzja:** G3 i G4 w tej postaci nie idą na Eden. Żeby pętla miała sens, miara sukcesu musi obejmować to, co
sondy mierzą: zbiór walidacyjny z narzędziem na innym tle i po obrocie (albo sondy wzięte z klatek val), a stan
modelu trzeba czytać w fazie, w której się rusza (początek dostrajania, nie plateau).

## Uczciwie

Nie da się zagwarantować, że zadziała. Bramki są ustawione tak, żeby każdy słaby punkt wyszedł tanio i wcześnie:
G1 i G3 kosztują razem ~1,5 h GPU, zanim wydamy 45 h na kampanię.
