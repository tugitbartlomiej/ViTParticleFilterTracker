# Sterowanie uczeniem przez dobór danych: Koopman + Langevin + Kalman + FastGAN na stanie dekodera

Data: 2026-09-28. Status: **projekt, czeka na zatwierdzenie** (nic nie zgłoszone na Edenie).

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

## Uczciwie

Nie da się zagwarantować, że zadziała. Bramki są ustawione tak, żeby każdy słaby punkt wyszedł tanio i wcześnie:
G1 i G3 kosztują razem ~1,5 h GPU, zanim wydamy 45 h na kampanię.
