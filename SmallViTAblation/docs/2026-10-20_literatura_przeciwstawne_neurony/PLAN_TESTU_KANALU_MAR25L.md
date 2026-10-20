# Plan testu kanału Mar25l na istniejących checkpointach ViT

> **SZKIC — eksperyment jeszcze nie utworzony ani nie uruchomiony (stan 2026-10-20 ok. 18:40)**

Źródła: notatka `../../plan_na_przyszle_eksperymenty/2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md` (pkt 1 i 3 listy działań) oraz `PRACE_PELNE_TEKSTY.md` (sekcja Mar25l).

## 1. Cel

Sprawdzić na istniejących 288 checkpointach, czy pary neuronów MLP z antyrównoległymi wagami wyjściowymi (cos kolumn fc2 < −0,5) są „kanałami” w sensie Mar25l, tj. czy mają prawie identyczne wejścia i działają jak GLU. Bez nowych treningów.

Checkpointy: 24 seedy o numerach 3047 + 1000·i (i = 0..23) × 12 kroków: 0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640.

## 2. Hipotezy

- **H1 (kanał):** cos wejść pary przeciwstawnej jest bliski 1 i silnie wzbogacony; ρ jest małe; błąd GLU jest mały; ‖m‖ rośnie, a ‖Δ‖ maleje.
- **H0 (brak kanału):** przeciwstawność wyjść bez zbieżności wejść. Jest to oczekiwane, bo wcześniej koaktywacja par geometrycznych wynosiła tylko ok. 23%, a ρ ≈ 0,71.

Wynik negatywny (H0) też jest wartościowy dla artykułu.

## 3. Model i trening (fakty do kontroli)

- MalyViT: łodyga konwolucyjna, 6 bloków timm (pre-LN: x + attn(norm1 x), x + mlp(norm2 x)), dim 192, 3 głowy, MLP 768 GELU, ls2 = Identity.
- fc1: 768×192 (wiersze to wejściowe wagi w_i); fc2: 192×768 (kolumny to wyjściowe wagi a_i ∈ R^192).
- 600 tokenów (siatka 20×30, krok 16 px); głowa gęsta (logit środka oraz l, t, r, b).
- Trening: AdamW, lr 5e-4, warmup 300, weight decay 0,05 na wagach ≥ 2D (w tym fc1/fc2), clip 1,0, batch 32, bf16.

## 4. Uwaga o weight decay

Weight decay ogranicza rozbieganie norm do nieskończoności, więc dosłowne kryterium Mar25l (norma parametrów > 1e3) prawdopodobnie nie zostanie spełnione. Test jest stopniowany: obok kryteriów dosłownych liczymy miary ciągłe (pkt 6 i 7).

## 5. Przeniesienie z Mar25l (adaptacja jawna)

Mar25l zakłada skalarne wyjście a_i. Tutaj a_i jest kolumną fc2 w R^192, więc przyjmujemy:

- s = (a_i + a_j)/2 i m = (a_i − a_j)/2 są wektorami;
- przeciwstawność mierzymy przez cos(a_i, a_j) < −0,5, a nie przez znak skalara;
- Δ = w_i − w_j, gdzie w_i to wiersz fc1;
- podprzestrzeń kanału do liczenia frakcji aktualizacji ustalamy przed pomiarem i zapisujemy w kodzie. Proponowany wybór: rzut na współrzędne parametrów pary (w_i, w_j, a_i, a_j). Wymaga zatwierdzenia przed uruchomieniem.

## 6. Pomiary (w każdym checkpoincie i każdym bloku)

### 6.1 Geometria wszystkich 294 528 par

- Liczba par w bloku: C(768, 2) = 294 528.
- cos wyjść: c_out = cos(a_i, a_j).
- cos wejść c_in w trzech wersjach:
  - (a) surowe wiersze fc1 z biasem;
  - (b) efektywne wejścia po złożeniu affine LayerNorm: ŵ = w ⊙ γ oraz b̂ = w·β + b; ŵ jest wycentrowane względem wektora jedynek (ŵ − mean(ŵ)·1, bo LayerNorm odejmuje średnią), b̂ zapisywane osobno;
  - (c) korelacja preaktywacji z_i i z_j na tokenach.
- Histogram 2D (c_in, c_out).
- Wzbogacenie P(c_in > τ | c_out < −0,5) / P(c_in > τ) dla progów τ ustalanych na VAL.

### 6.2 Normy i dynamika pary

- s, m, Δ jak w pkt 5.
- Czy ‖m‖ rośnie w czasie? Czy ‖Δ‖ maleje?
- Iloczyn ‖m‖·‖Δ‖ między checkpointami.
- Frakcja aktualizacji pary między sąsiednimi checkpointami: ‖Δθ_proj‖/‖Δθ‖ w podprzestrzeni kanału (pkt 5), gdzie Δθ to zmiana parametrów (w_i, w_j, a_i, a_j) między checkpointami.

### 6.3 Funkcja na danych

Dane: 50 klatek TEST (pozycje 5 i 15 w każdym z 25 filmów) oraz analogiczne 50 klatek VAL.

Definicje proponowane (do zapisania w kodzie):

- u_i = a_i · GELU(z_i): wkład neuronu i do strumienia resztkowego dla danego tokenu.
- ρ = ‖u_i + u_j‖ / (‖u_i‖ + ‖u_j‖). Kanał przewiduje ρ ≪ 1; dla par funkcjonalnych zmierzono wcześniej ok. 0,706.
- Względny błąd przybliżenia GLU: porównanie g(z_i) − g(z_j) z (z_i − z_j)·g'(z̄), gdzie g = GELU, z̄ = (z_i + z_j)/2, a błąd to ‖[g(z_i) − g(z_j)] − (z_i − z_j)·g'(z̄)‖ / ‖g(z_i) − g(z_j)‖.
- Rozkład z_i − z_j.
- Dla ostatniego bloku: interwencja polegająca na zamianie wkładu pary na przybliżenie GLU i pomiar zmiany położenia krawędzi (l, t, r, b) i logitu w pikselach.

### 6.4 Zbiory par do porównania

- **opposed_now:** c_out < −0,5 w bieżącym checkpoincie.
- **opposed_final:** c_out < −0,5 w końcowym checkpoincie, śledzone wstecz.
- **functional:** katalog par z kampanii `kompensacja_pruning_dynamics_20261018`.
- **twins:** top-500 par o najwyższym c_in.
- **losowe pary kontrolne.**

## 7. Kryteria dosłowne Mar25l obok miar stopniowanych

Dla każdego zbioru podajemy liczbę par spełniających każde kryterium dosłowne, obok miar stopniowanych:

| Kryterium | Dosłowne (Mar25l, App. B.3) | Stopniowane (ten plan) |
|---|---|---|
| Norma parametrów | > 1e3 | trajektorie ‖m‖ i ‖Δ‖ (pkt 6.2) |
| Zbieżność wejść | 1 − cos(w_i, w_j) < 1e-3 | c_in w wersjach a/b/c, wzbogacenie (pkt 6.1) |
| Aktualizacje w podprzestrzeni kanału | > 90% | frakcja ‖Δθ_proj‖/‖Δθ‖ jako rozkład (pkt 6.2) |
| Działanie jak GLU | brak kryterium dosłownego | ρ i błąd GLU (pkt 6.3) |

## 8. Kontrola jakości

- Liczba par z c_out < −0,5 w każdym checkpoincie musi zgadzać się dokładnie z wcześniejszym `GEOMETRY.json`.
- Smoke test na jednym seedzie i jednym checkpoincie przed uruchomieniem tablicy.
- Sumy SHA wag zgodne z `WEIGHTS_MANIFEST.json`.

## 9. Wykonanie (plan, nie uruchomiony)

- Klaster: Eden, partycja hopper (H100), konto transformers_vsc.
- Tablica 24 seedów, maksymalnie 8 równolegle. Kolejność: smoke, tablica, redukcja CPU.
- Wagi źródłowe: `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017/replications/s{seed}/weights/k{step:05d}.pt` (sumy SHA w `WEIGHTS_MANIFEST.json`).
- Środowisko: `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python` (torch 2.10, timm 1.0.24, numpy 2.2.6).
- Wyniki zgodnie z zasadami projektu: `experiments/<nazwa>_YYYYMMDD__HH-MM-SS/` z podfolderami `Kod_eksperymentu/`, `Opis/`, `Wyniki/`. Ten sam układ na Edenie w `/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation/experiments/`.
- Po utworzeniu folderu od razu wpis w `experiments/experiments_index.md` (pełna nazwa, odnośnik, cel).
- W `Wyniki/` podfoldery: `Code_Interpretujacy_Wyniki/`, `Dane/`, `Wykresy/`, `Raport/`, `Krótki_Wniosek/`.
- Wykresy, raport i krótki wniosek przygotowujemy dopiero po zapytaniu użytkownika o zakres (`AGENTS.md`). Pomiar i zapis danych nie wymagają dodatkowej zgody.

## 10. Ograniczenia

- Brak nowych treningów: wnioski dotyczą 12 zapisanych kroków, a nie ciągłej dynamiki.
- Dane TEST były już oglądane w poprzednich kampaniach (notatka źródłowa). Progi ustalamy na VAL, a wyniki na TEST traktujemy jako wstępne.
- Lagi między checkpointami zaplanowane z góry; korekta na wielokrotność porównań; nie wybieramy najlepszego lagu.

## 11. Statystyka (do protokołu)

- Bootstrap na poziomie seeda, z zachowaniem zależności checkpointów w seedzie. Obrazy nie są niezależne między seedami.
- Selekcja na VAL, ocena na TEST.
