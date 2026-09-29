# 23 — Metoda przewidywania latentu w uczeniu małego ViT (stan 2026-09-30)

Równanie Langevina dla jednego kroku uczenia: latent sondy porusza się jak cząstka z bezwładnością, popychana przez
dane. Zapisane na prośbę użytkownika („zapisz tę metodę przewidywania latentu”). Pomiary, z których wynika:
`experiments/maly_vit_rownanie_latentu` (WYNIKI_T0_T1.md, WYNIKI_T2.md, WYNIKI_EDEN.md) i
`experiments/maly_vit_rownanie_macierze` (WYNIKI.md).

## Co przewidujemy

- **Latent** z sondy: wyjście GELU w MLP ostatniego bloku małego ViT (768 liczb), średnia po tokenach narzędzia.
  Sondy: 192 stałe obrazy (12 wycinków × 16 wersji), 180 z narzędziem.
- **Krok**: jedna aktualizacja AdamW (lr 5e-4, β1 0,9, β2 0,999, paczka 32 obrazy). Przewidujemy przyrost
  Δz_{t+1} = z_{t+1} − z_t, gdzie z_t to latent po t aktualizacjach.
- **Siła wektora tożsamości**: rzut przyrostu, wystandaryzowanego negatywami z kroku 1000, na kierunek LDA
  narzędzie/tło (`experiments/maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py`). Każde ziarno ma własny kierunek
  (dwa ziarna: cosinus 0,085), więc równanie działa w układzie danej sieci.
- **Miara**: umiejętność = 100 · (1 − SSE / SS), czyli jaki procent błędu kwadratowego „braku zmiany” usuwa równanie.

## Dwa tryby (reguła użytkownika, 2026-09-29 i 2026-09-30)

- **Wyjaśnienie**: latent z chwili t+1 tłumaczony paczką t, która go stworzyła. Dozwolone („tłumaczenie aktualnego
  latentu tymi klatkami, które stworzyły ten latent w tej chwili, to ok”).
- **Prognoza z wyprzedzeniem**: tylko stany do chwili t, paczki do t−1 i stan optymalizatora. Klatki z kroków,
  których jeszcze nie było, są zabronione („do równania nie mogą wchodzić przyszłe klatki”).

## Równania

Wyjaśnienie (bezwładność plus siła paczki, która tworzy krok):

    Δz_{t+1} = a · Δz_t + b · S_t + szum          a ≈ 0,83,  b ≈ 0,98

Prognoza z wyprzedzeniem (pęd optymalizatora przepuszczony przez sieć):

    Δz_{t+1} = c · S^pęd_t + szum                 c ≈ 0,94   (bezwładność dostaje wtedy −0,04)

Prognoza bez przeliczania sieci (sama bezwładność):

    Δz_{t+1} = a₀ · Δz_t + szum                   a₀ ≈ 0,76

Współczynniki to zwykłe najmniejsze kwadraty na krokach uczenia (1000–1800 albo 1000–2000), wspólne dla wszystkich
768 współrzędnych i wszystkich sond. Są prawie takie same w każdym ziarnie (a 0,816–0,846, b 0,977–0,990 na 10
ziarnach), więc można je przenosić między przebiegami.

## Jak liczyć siły

Obie siły to zmiana latentu sond, gdy wagi przesunąć tylko o jedną część kroku AdamW, w stanie sieci sprzed kroku:

    S_t      = z(θ_t + d_nowa) − z(θ_t),   d_nowa = −lr / (1 − β1^k) · (1 − β1) · g_t     / mianownik
    S^pęd_t  = z(θ_t + d_pęd)  − z(θ_t),   d_pęd  = −lr / (1 − β1^k) · β1 · m_{t−1}       / mianownik
    mianownik = sqrt(β2 · v_{t−1} + (1 − β2) · g_t²) / sqrt(1 − β2^k) + eps

g_t to gradient paczki t po obcięciu normy do 1, m_{t−1} i v_{t−1} to stan AdamW sprzed kroku, k numer kroku.
Procedura w pętli uczenia, po `loss.backward()` i obcięciu gradientu, przed `opt.step()`:

1. policzyć części kroku (`kierunki()` w `experiments/maly_vit_rownanie_latentu/nagranie.py`);
2. przepuścić sondy przez sieć z wagami θ_t oraz θ_t + d w float32, w trybie eval, z tymi samymi buforami BatchNorm
   (`z_fp32()` w tym samym pliku, przez `torch.func.functional_call`, bez kopiowania sieci);
3. różnica latentów narzędzia to siła.

Koszt: dwa albo trzy dodatkowe przejścia 192 sond na krok. Nagranie 3000 kroków z siłami od kroku 900 trwa 16 min
na RTX 3070, około 6 min na H100 i 7 min na A100 (logi `wyniki_eden/logi/` w maly_vit_rownanie_latentu).

Uwaga o czystości prognozy: mianownik AdamW zawiera gradient bieżącej paczki z wagą 1 − β2 = 0,001. To jedyny ślad
paczki t w S^pęd; przeciek pomijalny, ale nazwany. Czysta wersja wymaga mianownika z samego v_{t−1}.

## Dokładność (test na krokach 2000–2999, poza dopasowaniem)

| tryb | równanie | latent | siła tożsamości | skąd |
|---|---|---|---|---|
| wyjaśnienie | 0,83 Δz_t + 0,98 S_t | 85,0 mediana 10 ziaren (80,5–89,2) | 85,7 (s47), 89,1 (s48) | Eden 1818331; lokalnie s47, s48 |
| prognoza | 0,94 S^pęd_t | 64,1–66,8 | 63,8–65,7 | lokalnie s47, s48 |
| prognoza bez sieci | 0,76 Δz_t | 56,3–56,8 | 56,6–56,8 | lokalnie s47, s48 |
| prognoza bez sieci | jądra pamięci z TFT na każdej współrzędnej (T1) | 59,1 mediana 10 ziaren | — | Eden 1818331 |

Sondy, których równanie nie widziało przy dopasowaniu (nowe narzędzia), dają to samo z dokładnością do 1–2 punktów.
Placebo (siła paczki sprzed 250 kroków, dopasowane od nowa) spada dokładnie do poziomu bez sterowania w każdym
z 10 ziaren, więc zysk z siły nie jest przeciekiem.

## Co sprawdzono i nie działa

- Duże uczone sieci jak w TFT (dryf, jądro pamięci, wymuszenie, szum kolorowy; 11 mln parametrów): 40 procent.
  Uczą się na pamięć okresu dopasowania (89 procent na nim).
- Wymuszenie z przeszłych paczek (uwaga sondy na obrazy paczek t−1..t−16): nic nie dodaje, równe placebo.
- Pełne macierze liniowe 768 × 768: 52–54 procent, gorzej niż jedna liczba bezwładności.
- Tanie jądro bez przeliczania sieci (podobieństwo sondy do obrazów × gradient ostatniego bloku): nic nie dodaje.
  Ostatni blok niesie tylko 7,5 procent ruchu; latent zmienia się przez wagi całej sieci (97,8 procent).
- Statystyki pikseli paczki i bufory BatchNorm (0,7 procent ruchu): nic nie dodają.

## Granice

- Jeden krok AdamW naprzód. Na 10 krokach sama historia latentu daje około 17 procent
  (`experiments/maly_vit_horyzont_prognozy`); prognoza wielu kroków z siłą pędu nie była sprawdzana.
- Tylko ramię bez dokładania klatek, kroki 1000–3000, jedna architektura i jeden zbiór.
- Siły wymagają dostępu do sieci i optymalizatora w trakcie uczenia; z samych zapisów latentu co 10 kroków ich nie
  odtworzymy.
