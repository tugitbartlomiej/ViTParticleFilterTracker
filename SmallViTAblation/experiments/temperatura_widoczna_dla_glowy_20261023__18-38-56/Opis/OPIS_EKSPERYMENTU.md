# Temperatura widoczna i niewidoczna dla głowy: czy sieć uczy się dalej w kierunkach, których strata nie widzi

**Data:** 2026-10-23, 18:38 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach, bez treningu, lokalnie (GPU do przejść w przód, jakobiany na CPU w float64).
**Rodzic:** `plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41` (moduł `Kod_eksperymentu/wspolne.py`, klatki TEST `Wyniki/Dane/klatki.json`,
46 checkpointów seedu 3047).

## Cel
Test hipotezy „sieć uczy się dalej w kierunkach, których strata nie widzi”. „Temperaturą” nazywamy drganie położeń tokenów (koloidów) tej samej klatki
między sąsiednimi checkpointami, które gaśnie po wytrenowaniu. Rozkładamy przesunięcie tokenu w stanie końcowym (po MLP 6, X[12], 192 wymiary) na składową
WIDOCZNĄ dla głowy (podprzestrzeń wierszy jakobianu ścieżki LayerNorm → Linear 192→192 → GELU → Linear 192→5 względem stanu tokenu, ≤ 5 wymiarów)
i NIEWIDOCZNĄ (reszta, ≥ 187 wymiarów).

## Predykcja przed pomiarem (zapisana przed uruchomieniem pomiaru)
1. Temperatura widoczna (energia przesunięcia w podprzestrzeni wierszy jakobianu głowy) gaśnie razem z nasyceniem IoU, czyli do ok. 60–100 epok.
2. Temperatura niewidoczna gaśnie wolniej albo wcale do 400 epok.
3. Udział energii widocznej w przesunięciu (u_vis) spada w czasie uczenia poniżej udziału na starcie.
Progi werdyktu: (a) E_vis na 1000 kroków spada o ≥ 10× między oknem 1–10 ep a oknem 200–400 ep; (b) E_inv spada mniej niż E_vis, stosunek spadków ≥ 3;
(c) u_vis w 200–400 ep jest niższy niż w 0–10 ep; porównanie z zerem losowym (5/192 ≈ 0,026 w izotropii i udział w losowej 5-wymiarowej podprzestrzeni):
powyżej zera losowego znaczy, że uczenie wciąż głównie rusza w widocznych kierunkach.

## Hipotezy
H1. u_vis na starcie jest wyraźnie ponad zerem losowym (uczenie na początku popycha tokeny tam, gdzie głowa patrzy) i spada z czasem.
H2. E_vis i E_inv gasną w różnym tempie: widoczna szybciej (nasycenie strat), niewidoczna wolniej (dryf w jądrze głowy, napędzany szumem SGD i AdamW).
H3. Wynik nie jest artefaktem nierównych odstępów między checkpointami ani wzrostu rozmiaru chmury (kontrola: energie na 1000 kroków i podzielone przez kwadrat promienia chmury).

## Model i dane
MałyViT 35 000 od zera, seed 3047, 46 checkpointów (0, 300, 1000, 3000, 6000, 10 000, 16 000, 24 000, 33 000, 44 000, 55 000, 65 640, potem co 10 epok do
437 500; epoka = 1093,75 kroku): `experiments/skae_replikacja_mapy_vit3047_20261022__17-14-27/Wyniki/Dane/Wejscia/wagi`. Kod modelu:
`experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`. Droga od X[12] do wyjścia: `norm` (LayerNorm 192) → `glowa` = Linear(192,192) → GELU →
Linear(192,5), bez reshape do siatki (reshape do 20 × 30 × 5 dotyczy tylko układu wyjścia, token po tokenie funkcja jest ta sama), więc
`glowa_tokenu(x) = glowa(norm(x))`, R^192 → R^5. 10 klatek TEST z `klatki.json`; tokeny: narzędzie = udział obrysu ≥ 0,25, tło = udział == 0
(łaty o udziale pośrednim wchodzą tylko do grupy „wszystkie”), 600 tokenów na klatkę.

## Procedura
`Kod_eksperymentu/temperatura_glowy.py Wyniki/Dane`:
1. Dla każdego checkpointu k i klatki: X_k = X[12] (600 × 192, float64, CPU), IoU i pewność z przejścia w przód (GPU, float32), jakobian
   H_{k,i} = ∂glowa_tokenu/∂x_i (5 × 192) dla każdego z 600 tokenów (`torch.func.vmap(jacrev)`, float64, wagi głowy z checkpointu k). Baza ortonormalna
   V_i = wiersze Vh z SVD macierzy H (wartości osobliwe < 1e-8 · max odrzucone), wymiar efektywny r_i. Kontrola odtworzenia głowy: błąd względny
   `glowa_tokenu` na pełnym X[12] wobec (i) ścieżki modelu w float32 na GPU (`net.glowa(net.norm(t))`) oraz (ii) pełnego `net(x)` na obrazie.
2. Przesunięcie d_i = X_{k+1,i} − X_{k,i} (45 par). d_vis = V_i V_iᵀ d_i (baza z checkpointu k), d_inv = d_i − d_vis. Na parę i klatkę mediana po tokenach osobno
   dla narzędzia, tła i wszystkich: E_tot = ‖d‖², E_vis, E_inv, u_vis = E_vis/E_tot (mediana ilorazów po tokenach). Zero 1: 5/192 = 0,026. Zero 2: udział energii w losowej
   5-wymiarowej podprzestrzeni (jedna baza na token z QR macierzy gaussowskiej, seed 0, ta sama dla wszystkich checkpointów i klatek). Normalizacja czasu:
   E/Δkrok · 1000. Normalizacja przestrzeni: dodatkowo podzielone przez R²_k = średnia ‖x_i − x̄‖² chmury 600 tokenów w checkpoincie k.
3. Drganie bez dryfu (druga różnica): j_i = X_{k,i} − (X_{k−1,i} + X_{k+1,i})/2, rozkład z bazą z checkpointu k; liczone dla wszystkich 44 trójek, ale w tabelach
   tylko trójki o praktycznie równych odstępach (maska `rowny_odstep`: różnica odstępów < 1 %; w oknach tabel to środki 70–390 ep, odstępy 10 923 i 10 938 kroków; maska obejmuje też środek 44 000, który poza oknami tabel); pozostałe oznaczone maską.
4. Kontrola linearyzacji: ‖glowa_k(X_{k+1,i}) − glowa_k(X_{k,i})‖ / ‖H_{k,i} d_i‖, mediana po tokenach (głowa z checkpointu k po obu stronach).
5. Okna epok 0–1, 1–10, 10–30, 30–60, 60–100, 100–200, 200–300, 300–400 (para przypisana do okna przez epokę środka pary, [a, b); trójka przez epokę środka;
   IoU: średnia po klatkach w lewym końcu pary). Tabele: mediana po parach × klatkach z E_vis/1000 kroków, E_inv/1000 kroków (też / R²), u_vis, zero losowe,
   IoU, osobno narzędzie/tło/wszystkie.

## Metryki
E_tot, E_vis, E_inv (mediany po tokenach), u_vis, u_rand, E_rand, wymiar efektywny podprzestrzeni widocznej, stosunek zmiany wyjścia głowy do przewidywania liniowego,
błąd odtworzenia głowy, IoU, pewność, promień chmury.

## Ograniczenia
Rozkład wykorzystuje głowę z checkpointu k, a tokeny w k+1 są w przestrzeni, w której głowa k+1 patrzy trochę inaczej; „widoczne” znaczy widoczne dla głowy z początku
pary (jakobian lokalny, nie globalny). Energie na 1000 kroków dla przesunięć (nie dla zmiennej o wariancji liniowej w czasie) zakładają skalowanie liniowe energii z Δkrok:
przy ruchu balistycznym energia rośnie jak Δ², przy szumie jak Δ, więc to nie jest jedyna słuszna normalizacja (stąd osobno drganie bez dryfu na równych odstępach).
Jeżeli linearyzacja jest daleka od 1, rozkład na „widoczne” traci sens ilościowy. 10 klatek TEST, jeden seed (3047), tokeny narzędzia to ok. 20 na klatkę.
Bez wykresów, raportu i wniosku (nie zamówione).

## Odtwarzanie
`cd Kod_eksperymentu; PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe temperatura_glowy.py ../Wyniki/Dane` (ok. 30 s, GPU + CPU float64), potem
`cd ../Wyniki/Code_Interpretujacy_Wyniki; PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe odczyt_temperatura.py ../Dane` (zapisuje `tabele_odczyt.txt`).
Pochodzenie: `Kod_eksperymentu/pochodzenie.json`.

## Stan (2026-10-23, 18:47): pomiar ukończony, `Wyniki/Dane/temperatura_glowy.npz`, `temperatura_glowy_tabele.json`, `tabele_odczyt.txt`, `log.txt`

- Odtworzenie głowy: błąd względny `glowa_tokenu` (float64, vmap) wobec ścieżki modelu i pełnego `net(x)` ≤ 8,9e-8 (średnio 6,7e-8) na wszystkich 460 parach
  (checkpoint, klatka); to poziom float32, próg 1e-4 spełniony z zapasem. Wymiar efektywny podprzestrzeni widocznej: 5 dla każdego z 276 000 tokenów
  (jakobian ma zawsze rząd 5; wartości osobliwe mediany: k0 0,22–0,15, k65640 0,28–0,03, k437500 0,13–0,04).
- Linearyzacja (‖Δ wyjścia głowy‖ / ‖H d‖, mediana): 0,64–0,74 w oknie 0–1 ep, 0,93 w 1–10, 0,99–1,05 w 10–200, 0,87–1,01 w 200–400 (tło w oknie
  200–300: 0,87): zawsze w przedziale 0,5–2, więc rozkład jest ilościowo sensowny; w pierwszej epoce linearyzacja przeszacowuje zmianę wyjścia o ok. 1,4×.
- Zero losowe: mediana udziału w losowej 5-wymiarowej podprzestrzeni 0,023 (średnia 5/192 = 0,026; mediana ilorazu jest niższa od średniej).
- E_vis i E_inv na 1000 kroków (mediany), okno 1–10 ep → 200–400 ep: narzędzie E_vis 16,5 → 2,41 (spadek 6,8×), E_inv 169 → 22,6 (7,5×);
  tło E_vis 17,6 → 6,5 (2,7×), E_inv 233 → 66,6 (3,5×); wszystkie 17,4 → 6,3 (2,8×) i 228 → 65,8 (3,5×). Po podzieleniu przez R² chmury: narzędzie 9,9× i 9,4×,
  tło 3,7× i 5,0×.
  Od ok. 30 ep obie składowe są płaskie (narzędzie E_vis 2,9 → 2,4, E_inv 27 → 22; tło E_vis 5,6 → 6,5, E_inv 66 → 72, w jednostkach R² tło E_inv rośnie 0,035 → 0,049):
  temperatura nie gaśnie do zera, tylko ustala się na stałym poziomie, a nasycenie IoU (0,5–0,56) następuje ok. 10–30 ep.
- Udział u_vis: start (0–10 ep) 0,145 narzędzie / 0,088 tło / 0,090 wszystkie; 200–400 ep: 0,087 / 0,086 / 0,086; zero losowe 0,023 (3,7× niższe). Czyli u_vis spada tylko na
  narzędziu (0,20 w 0–1 ep → 0,09), na tle prawie nie; w całym przebiegu jest 3–9× ponad zerem losowym: przesunięcie jest wzbogacone w kierunki widoczne dla głowy,
  ale 91 % energii jest w kierunkach niewidocznych (przy 5 z 192 wymiarów).
- Drganie bez dryfu (druga różnica, równe odstępy): J_vis i J_inv praktycznie stałe od 60 do 400 ep (wszystkie: J_vis 60,4 → 48,2, J_inv 474 → 508; u_vis 0,114 → 0,089;
  w jednostkach R²: J_vis 0,037 → 0,035, J_inv 0,28 → 0,37, tło rośnie, narzędzie 0,136 → 0,133).
- Werdykt predykcji: (1)/(a) nie potwierdzona: E_vis spada tylko 6,8× (narzędzie; 9,9× w jednostkach R², tuż poniżej 10×) i 2,7–2,8× na tle i wszystkich, a po ok. 30 ep jest płaski, nie gaśnie
  „do 60–100 ep”, tylko wcześniej i do poziomu niezerowego. (2)/(b) nie potwierdzona: E_inv nie spada wolniej niż E_vis; stosunek spadków E_vis/E_inv = 0,91 (narzędzie), 0,77 (tło), 0,80 (wszystkie),
  próg ≥ 3 niespełniony; obie składowe stygną razem i razem się ustalają. (3)/(c) częściowo: u_vis w 200–400 ep jest niższy niż w 0–10 ep (0,087 vs 0,145 narzędzie; 0,086 vs 0,090 wszystkie, na tle różnica w granicach rozrzutu
  między parami), ale znacznie powyżej zera losowego (0,023): uczenie nadal głównie wzbogaca ruch w kierunkach widocznych dla głowy. Hipoteza „sieć uczy się dalej w kierunkach, których strata nie widzi” w tej postaci
  (niewidoczne gasną wolniej) nie ma poparcia w tych danych; poparcie ma tylko trywialna wersja: większość energii (≈ 91 %) i całe pozostałe drganie leży w kierunkach niewidocznych.
- Odstępstwa i ograniczenia pomiaru: 10 klatek, jeden seed; mediana po tokenach narzędzia to 20 tokenów na klatkę; grupa „tło” to 560–578 tokenów; energie na 1000 kroków przy ruchu o charakterze szumu nie skalują się
  liniowo z Δkrok, więc wartości w oknach 0–10 ep (odstępy 300–3000 kroków) i późnych (10 938 kroków) nie są w pełni porównywalne; porównanie późnych okien między sobą jest czyste.
