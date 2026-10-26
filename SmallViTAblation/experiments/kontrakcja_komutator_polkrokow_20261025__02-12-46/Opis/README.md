# kontrakcja_komutator_polkrokow_20261025__02-12-46

Lokalnie, bez treningu i bez Edena. Rodzic: [schur_qr_jakobiany_mlp_20261024__10-03-40](../../schur_qr_jakobiany_mlp_20261024__10-03-40/).

## Cel

Dwa „mosty” między przejściem MałegoViT przez 12 półkroków strumienia rezydualnego a fizyką układów cząstek:

- **Most 2: kontrakcja i porządek na sferze.** Tokeny po LayerNorm bez parametrów afinicznych leżą na sferze; dynamika
  uwagi jest bliska modelom synchronizacji/konsensusu na sferze (Criscitiello i in. 2024; Markdahl 2016), a pytanie, czy
  sieć klastruje się do jednego punktu, czy z klastrowania „ucieka” (Isobe i in. 2026), sprowadza się do tego, które półkroki
  ściskają chmurę, a które ją rozciągają, i jak zachowuje się parametr porządku. Mierzone w każdym z 13 stanów / 12 półkroków.
- **Most 3: defekt komutatora uwaga–MLP** (Mudarisov i in. 2026, „Feed-Forward Steering in Transformer Residual Dynamics”):
  na ile kolejność „najpierw uwaga, potem MLP” w bloku ma znaczenie w porównaniu z odwróconą i z addytywną (równoległą), i
  jaki to ma skutek dla wyjścia detekcji.

Tylko definicje, tabele liczb i uwagi techniczne; bez wykresów, raportu i wniosków (foldery `Wyniki/Wykresy`, `Wyniki/Raport`,
`Wyniki/Krótki_Wniosek` celowo puste).

## Model, checkpointy, dane

- Model: MałyViT (`laser_app.compute.history.modul_modelu()`, `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`):
  łodyga CNN, 6 bloków timm z pre-LayerNorm (ls = Identity), d = 192, uwaga 3 głowy, MLP 192 → 768 → 192 z GELU, 600 tokenów
  (zdjęcie 320 × 480 → siatka 20 × 30), `net.norm` (LayerNorm z γ, β) i głowa gęsta (5 wyjść na token, [0] = logit ciepła).
- Historia: `LaserLatentSpaceApp/cache_uczenia/lasery/90f58d4f8b263d8b` (`History(katalog)`; **K = 806** checkpointów, kroki 0–65640,
  1094 kroki/epokę), wagi `assets/MalyViT_35000_od_zera_s47/wagi/k<krok>.pt`. SHA256 6 checkpointów w `Kod_eksperymentu/pochodzenie.json`
  (zgodne z rodzicem), wszystkich 806 w `Wyniki/Dane/wagi_sha256_wszystkie.csv` (z tych samych bajtów, które wczytano).
- 6 checkpointów pełnego pomiaru: jak w rodzicu, `np.linspace(0, K−1, 6)` po indeksie historii → kroki 0, 6564, 21300, 36100,
  50800, 65640 (epoki 0, 6, 19.5, 33, 46.4, 60).
- Tokeny narzędzia: `History.narz` po domyślnym `ustaw_etykiete()` („obrys TR_0127_S1_P03 kl. 1 (cystotome), dotknięte: 17 łat”):
  17 tokenów narzędzia, 583 tła. GT: `h.gt` = [239.33, 88.0, 376.67, 136.67] (xyxy w pikselach 480 × 320).
- Aplikacji nie zmieniano (import `laser_app` tylko do odczytu; `settings.json` nietknięty).

## Procedura

1. Własne przejście w przód (`pomiar_mosty.py: przejscie`), wzorzec z `pomiar_schur_qr.py`: łodyga + `sincos_2d` → stan 0; dla bloku b:
   A(t) = t + ls1(attn(norm1 t)) → stan 2b+1, M(t) = t + ls2(mlp(norm2 t)) → stan 2b+2. Stan 12 = wyjście bloku 5 (surowy strumień);
   osobno stan „glowa” = net.norm(X_12) (rzeczywiste wejście głowy, LayerNorm z γ, β; klucze `*_glowa`). Przejście w float32 na GPU
   (jak aplikacja), wszystkie miary w float64.
2. Kontrola: dla 6 checkpointów porównanie z `hist.podkroki(..., lodyga=False)`: głowa na 13 stanach, mapa ciepła, ramka szczytu
   i pewność — maks. różnica 0.0 (identyczne). Dla wszystkich 806 checkpointów pewność i IoU równe `h.pew`, `h.iou` z historii
   (maks. różnica 0 i 2·10⁻⁷).
3. Most 2 (`skalary`, `polkroki`): 6 checkpointów pełny zestaw; wszystkie 806 checkpointów — skalary punktów 1–3 i dodatkowo
   ranga efektywna (punkt 4) dla 13 stanów + „glowa”, oraz pewność / IoU / maks. ciepło wyjścia.
4. Most 3 (`komutator`): 6 checkpointów × 6 bloków × 6 wariantów, reszta sieci (bloki b+1..5, norm, głowa) przepuszczana po każdym.

## Definicje miar

Oznaczenia: X_s ∈ ℝ^{600×192} chmura surowa w stanie s; u_i = (x_i − mean_c x_i)/‖x_i − mean_c x_i‖ (średnia po 192 współrzędnych
tokenu = LayerNorm bez γ, β, bez ε). „narz.” = 17 tokenów narzędzia, „tło” = 583. Półkrok s → s+1: uwaga gdy s parzyste, MLP gdy nieparzyste.

| nazwa (klucz w plikach) | definicja |
|---|---|
| D_{X,u}_{wszystkie, narzedzie, tlo, narz_tlo} | średnia odległość euklidesowa par: wszystkie pary i ≠ j; pary narz.–narz.; tło–tło; wszystkie pary narz.–tło (na u odległość cięciwy, D ∈ [0, 2]). `torch.cdist` w trybie dokładnym (bez mnożenia macierzy) |
| iloraz_D_* | D_{s+1}/D_s; < 1 kontrakcja, > 1 rozciąganie. Iloczyn ilorazów 6 półkroków uwagi × iloczyn 6 MLP = D_12/D_0 |
| r_{wszystkie, narzedzie, tlo} | parametr porządku Kuramoto r = ‖mean_i u_i‖ ∈ [0, 1] (1 = wszystkie tokeny w jednym punkcie sfery) |
| cos_srednich_narz_tlo | cos kąta między mean_{narz} u i mean_{tło} u („r między klasami”) |
| sep_{X,u} | ‖μ_n − μ_t‖ / sqrt(½(V_n + V_t)), V_k = mean_{i∈k} ‖y_i − μ_k‖² (ślad kowariancji klasy, wszystkie 192 kierunki) |
| sep_os_{X,u} | jak sep, ale wariancje tylko wzdłuż osi w = (μ_n − μ_t)/‖μ_n − μ_t‖ (separacja Fishera na osi średnich) |
| odl_srednich_*, war_wewn_* | licznik ‖μ_n − μ_t‖ i mianownik ½(V_n + V_t) separacji |
| zmiana_sep_*, zmiana_r_* | wartość w s+1 minus w s |
| PR_{X,u}_{wszystkie, narzedzie, tlo} | ranga efektywna (participation ratio) (Σλ)²/Σλ² wartości własnych kowariancji chmury (192 × 192, `numpy.linalg.eigvalsh`, CPU); dla narzędzia ≤ 16 (17 punktów) |
| norma_srednia_X | średnia ‖x_i‖ (skala surowego strumienia) |
| norma_dx | ‖Δx_i‖, Δx = X_{s+1} − X_s (per token) |
| udz_srednia, udz_radialna, udz_styczna | rozkład ‖Δx‖² na trzy ortogonalne części względem stanu wejściowego: wzdłuż wektora 1 (C·mean_c(Δx)², niewidoczna dla LayerNorm), radialną (u_i·Δx)² (zmienia ‖x − mean_c x‖, czyli tylko skalę przed LN) i styczną do sfery (reszta; jedyna, która przesuwa u_i). Per token (mediany w tabelach) |
| udz_*_suma | ten sam rozkład sumowany po tokenach: Σ_i część / Σ_i ‖Δx_i‖² (ważony normą) |
| radialna_wzgl | (u_i·Δx)/‖x_i − mean_c x_i‖: względna zmiana normy części środkowanej |
| kat_u_deg | kąt (°) między u_i(s) a u_i(s+1): faktyczny ruch tokenu po sferze |
| knn_zachowane_{u,X} | odsetek k = 5 najbliższych sąsiadów tokenu (wśród wszystkich 600, na u; dla porównania na X) zachowanych po półkroku. W tabelach średnia po tokenach (mediana ma ziarno 0,2) |
| S, R, P (most 3) | X = stan 2b; S = M(A(X)) (prawdziwa kolejność, = stan 2b+2, zgodność 0.0), R = A(M(X)) (odwrócona), P = X + attn(LN₁X) + mlp(LN₂X) (równoległa, addytywna) |
| SR_do_SX, SP_do_SX | per token ‖S − R‖/‖S − X‖ i ‖S − P‖/‖S − X‖; „globalnie” = to samo w normie Frobeniusa całej chmury |
| cos_SX_PX, cos_SX_RX | per token cos kąta między (S − X) a (P − X), (R − X) |
| warianty | S_prawdziwa, R_odwrocona, P_rownolegla, tylko_uwaga (X → A(X), MLP bloku b pominięte), tylko_MLP (X → M(X), uwaga pominięta), bez_bloku (X → X; dodatkowy punkt odniesienia); każdy przepuszczony przez bloki b+1..5, norm, głowę |
| max_cieplo, pewnosc, iou, korelacja_mapy_z_S | maks. sigmoid(logit ciepła) po 600 tokenach; pewność szczytu z `M.dekoduj(o, k=1)`; IoU ramki szczytu z GT (`hist.iou`, jak `podkroki`/`policz`); współczynnik Pearsona mapy sigmoid(ciepło) (600) wariantu z mapą S; dodatkowo `maks_roznica_mapy_od_S`, `argmax_mapy` |

## Pliki wyników (`Wyniki/Dane/`)

- `wszystkie_checkpointy.npz`: `kroki`, `epoki` (806,), dla każdej miary tablica **(806 checkpointów × 13 stanów)** (`D_u_wszystkie`, `r_tlo`, `sep_u`, `PR_X_narzedzie`, …), ta sama miara w stanie „glowa” jako `<miara>_glowa` (806,), `max_cieplo`, `pewnosc`, `iou` (806,), `wagi_sha256`.
- `wszystkie_checkpointy_skalary.csv`: to samo w formacie długim (krok, epoka, stan 0–12 / glowa, miary).
- `mosty_k<krok>.npz` (6 plików): `stan_<miara>` (14,) = 13 stanów + glowa; `tok_<miara>` (12, 600) per półkrok i token; `polkrok_<miara>` (12,) sumy; `komutator_<miara>` (6, 600) per blok i token; `narz`, `heat_S`.
- `mosty_6_checkpointow.json`: wszystko dla 6 checkpointów (stany, 12 półkroków z ilorazami i percentylami 10/50/90 po tokenach dla wszystkie/narzędzie/tło, komutator z wariantami, kontrola z `podkroki`, czasy, SHA256).
- `komutator.csv`: checkpoint × blok × wariant (wyjście i mediany ilorazów).
- `tabele.md` (z `Wyniki/Code_Interpretujacy_Wyniki/tabele_markdown.py`): wszystkie tabele, także 12 półkroków dla pozostałych 5 checkpointów.
- `log_pomiaru.txt`, `log_wszystkie.json`, `wagi_sha256_wszystkie.csv`.

## Odtworzenie

```
cd Kod_eksperymentu
W:/Python/3.11/python.exe pomiar_mosty.py --szesc --wszystkie      # ok. 6,5 min (RTX 3070)
cd ../Wyniki/Code_Interpretujacy_Wyniki
W:/Python/3.11/python.exe tabele_markdown.py
```

## Uwagi techniczne

- Historia ma 806 checkpointów (nie ok. 486): gęsto co 10 kroków na początku, potem rzadziej; „wszystkie” = 806.
- Stan 12 to surowe wyjście bloku 5; rzeczywiste wejście głowy (net.norm z γ, β) jest osobno jako „glowa” (w ilorazach 12 półkroków nie występuje).
- Na sferze u nie ma ε LayerNorm; normy ‖x − mean_c x‖ są rzędu 10–80, więc różnica z LN(ε = 1e-6) bez γ, β jest pomijalna.
- Składowa wzdłuż 1 przesunięć jest bardzo mała (Σ-udział ≤ 0,01), więc ruch rozkłada się praktycznie na radialny i styczny.
- k0 (wagi losowe): IoU = 0 we wszystkich wariantach i maks. ciepło 0,014, więc skutek komutatora dla wyjścia w k0 jest nieinformatywny (korelacje map są liczone, ale na mapie prawie stałej).
- IoU ramki szczytu jest nieciągłe: przy zmianie wariantu szczyt może przeskoczyć na inny token (stąd warianty z IoU > IoU S, np. R w bloku 0 dla k65640). Pełniejszy obraz daje razem maks. ciepło, korelacja map i `argmax_mapy`.
- Dla 17 tokenów narzędzia percentyle i średnie są z 17 wartości; sąsiedzi kNN liczeni wśród wszystkich 600 tokenów.
- Czasy: 6 checkpointów 1,2–1,6 s każdy, łącznie z kontrolą `podkroki` (0,5–0,7 s); 806 checkpointów 370 s (0,33–0,57 s/checkpoint, w tym SHA256 i odczyt 11,7 MB wag).
- Pomiar wykonano raz, bez błędów. Pliki próbne usunięte.

## Tabele liczb

Poniżej wybrane tabele z `Wyniki/Dane/tabele.md` (bez interpretacji).

## Most 2

### k65640: 12 półkroków (stosunki D_(s+1)/D_s na u; r, separacja, PR w stanie s+1)

| # | s→s+1 | półkrok | D_u wsz. | D_u narz. | D_u tło | D_u n–t | D_X wsz. | r | r narz. | r tło | cos(μn,μt) | sep_u | Δsep_u | sep_X | PR_u | PR_X | med‖Δx‖ narz. | med‖Δx‖ tło | udz. styczny narz. (med) | udz. styczny tło (med) | udz. styczny Σ | udz. radialny Σ | udz. wzdłuż 1 Σ | med Δ‖c‖/‖c‖ | kąt u narz. [°] | kąt u tło [°] | kNN zach. narz. | kNN zach. tło |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0→1 | uwaga b0 | 0.928 | 0.952 | 0.927 | 0.945 | 1.026 | 0.657 | 0.768 | 0.659 | 0.735 | 0.758 | +0.024 | 0.737 | 12.60 | 11.71 | 6.13 | 5.90 | 0.984 | 0.992 | 0.987 | 0.013 | 0.001 | +0.001 | 29.3 | 24.8 | 0.953 | 0.936 |
| 2 | 1→2 | MLP b0 | 1.066 | 0.911 | 1.065 | 1.079 | 2.101 | 0.593 | 0.805 | 0.597 | 0.549 | 0.980 | +0.221 | 0.966 | 9.88 | 9.45 | 17.51 | 18.48 | 0.742 | 0.825 | 0.836 | 0.163 | 0.002 | +0.539 | 35.2 | 36.9 | 0.694 | 0.768 |
| 3 | 2→3 | uwaga b1 | 0.958 | 0.947 | 0.956 | 0.987 | 1.033 | 0.636 | 0.827 | 0.642 | 0.534 | 1.080 | +0.101 | 1.078 | 9.98 | 9.73 | 10.68 | 8.96 | 0.988 | 0.990 | 0.985 | 0.015 | 0.000 | +0.032 | 21.1 | 17.2 | 0.918 | 0.951 |
| 4 | 3→4 | MLP b1 | 0.950 | 0.926 | 0.950 | 0.959 | 1.136 | 0.679 | 0.854 | 0.684 | 0.583 | 1.134 | +0.054 | 1.115 | 9.00 | 8.40 | 8.01 | 10.34 | 0.833 | 0.890 | 0.874 | 0.123 | 0.003 | +0.113 | 12.0 | 16.0 | 0.918 | 0.918 |
| 5 | 4→5 | uwaga b2 | 0.957 | 0.973 | 0.956 | 0.973 | 1.009 | 0.710 | 0.862 | 0.715 | 0.607 | 1.166 | +0.032 | 1.151 | 8.73 | 8.17 | 9.56 | 8.13 | 0.999 | 0.978 | 0.966 | 0.033 | 0.000 | +0.030 | 16.1 | 12.5 | 0.988 | 0.974 |
| 6 | 5→6 | MLP b2 | 0.941 | 0.981 | 0.937 | 1.013 | 1.192 | 0.746 | 0.868 | 0.754 | 0.547 | 1.336 | +0.171 | 1.298 | 7.49 | 6.87 | 6.68 | 17.52 | 0.829 | 0.826 | 0.846 | 0.153 | 0.001 | +0.167 | 9.6 | 19.0 | 0.918 | 0.904 |
| 7 | 6→7 | uwaga b3 | 0.979 | 0.974 | 0.979 | 0.971 | 0.992 | 0.759 | 0.875 | 0.766 | 0.588 | 1.320 | -0.017 | 1.289 | 7.91 | 7.32 | 12.34 | 11.16 | 0.999 | 0.978 | 0.971 | 0.028 | 0.001 | -0.012 | 18.1 | 13.0 | 0.976 | 0.924 |
| 8 | 7→8 | MLP b3 | 0.859 | 1.093 | 0.844 | 1.062 | 1.173 | 0.823 | 0.845 | 0.836 | 0.447 | 1.632 | +0.312 | 1.696 | 7.53 | 6.77 | 10.05 | 30.91 | 0.890 | 0.865 | 0.871 | 0.128 | 0.000 | +0.220 | 12.7 | 24.9 | 0.847 | 0.877 |
| 9 | 8→9 | uwaga b4 | 0.985 | 1.013 | 0.985 | 0.981 | 1.006 | 0.829 | 0.841 | 0.841 | 0.483 | 1.582 | -0.050 | 1.666 | 7.52 | 6.82 | 14.04 | 13.12 | 0.974 | 0.992 | 0.986 | 0.014 | 0.000 | +0.005 | 17.9 | 11.2 | 0.988 | 0.968 |
| 10 | 9→10 | MLP b4 | 0.944 | 1.087 | 0.941 | 0.970 | 1.078 | 0.849 | 0.814 | 0.861 | 0.546 | 1.463 | -0.119 | 1.533 | 7.96 | 7.14 | 25.30 | 19.56 | 0.946 | 0.889 | 0.896 | 0.104 | 0.000 | +0.094 | 26.2 | 13.2 | 0.753 | 0.926 |
| 11 | 10→11 | uwaga b5 | 1.059 | 0.987 | 1.062 | 1.030 | 0.985 | 0.830 | 0.820 | 0.842 | 0.501 | 1.492 | +0.029 | 1.541 | 8.00 | 7.15 | 11.50 | 17.02 | 0.958 | 0.816 | 0.818 | 0.180 | 0.002 | -0.092 | 12.8 | 12.0 | 0.929 | 0.975 |
| 12 | 11→12 | MLP b5 | 1.059 | 1.038 | 1.064 | 1.010 | 1.012 | 0.808 | 0.806 | 0.820 | 0.505 | 1.388 | -0.104 | 1.436 | 8.44 | 7.52 | 15.99 | 14.40 | 0.885 | 0.896 | 0.883 | 0.115 | 0.003 | -0.062 | 15.5 | 10.6 | 0.871 | 0.948 |

### k65640: stany (0–12; „glowa” = net.norm(X_12))

| stan | D_u | D_u narz. | D_u tło | D_u n–t | D_X | śr. ‖x‖ | r | r narz. | r tło | cos(μn,μt) | sep_u | sep_os_u | sep_X | sep_os_X | PR_u | PR_u narz. | PR_u tło | PR_X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.131 | 0.956 | 1.129 | 1.172 | 14.75 | 14.10 | 0.584 | 0.740 | 0.586 | 0.683 | 0.734 | 2.301 | 0.718 | 2.238 | 11.85 | 5.36 | 11.80 | 11.63 |
| 1 | 1.050 | 0.911 | 1.046 | 1.108 | 15.14 | 15.30 | 0.657 | 0.768 | 0.659 | 0.735 | 0.758 | 2.458 | 0.737 | 2.387 | 12.60 | 5.31 | 12.54 | 11.71 |
| 2 | 1.119 | 0.830 | 1.114 | 1.195 | 31.82 | 28.67 | 0.593 | 0.805 | 0.597 | 0.549 | 0.980 | 2.837 | 0.966 | 2.697 | 9.88 | 3.53 | 9.96 | 9.45 |
| 3 | 1.071 | 0.786 | 1.065 | 1.180 | 32.88 | 30.90 | 0.636 | 0.827 | 0.642 | 0.534 | 1.080 | 3.123 | 1.078 | 3.051 | 9.98 | 3.33 | 10.11 | 9.73 |
| 4 | 1.018 | 0.728 | 1.012 | 1.131 | 37.35 | 36.49 | 0.679 | 0.854 | 0.684 | 0.583 | 1.134 | 3.225 | 1.115 | 3.027 | 9.00 | 3.45 | 9.07 | 8.40 |
| 5 | 0.975 | 0.708 | 0.968 | 1.101 | 37.68 | 38.45 | 0.710 | 0.862 | 0.715 | 0.607 | 1.166 | 3.218 | 1.151 | 2.993 | 8.73 | 3.49 | 8.83 | 8.17 |
| 6 | 0.918 | 0.694 | 0.906 | 1.115 | 44.92 | 48.35 | 0.746 | 0.868 | 0.754 | 0.547 | 1.336 | 3.443 | 1.298 | 3.121 | 7.49 | 3.64 | 7.60 | 6.87 |
| 7 | 0.898 | 0.676 | 0.888 | 1.083 | 44.58 | 49.18 | 0.759 | 0.875 | 0.766 | 0.588 | 1.320 | 3.440 | 1.289 | 3.140 | 7.91 | 3.47 | 8.04 | 7.32 |
| 8 | 0.772 | 0.739 | 0.750 | 1.150 | 52.28 | 66.75 | 0.823 | 0.845 | 0.836 | 0.447 | 1.632 | 3.680 | 1.696 | 3.587 | 7.53 | 3.02 | 8.18 | 6.77 |
| 9 | 0.760 | 0.749 | 0.739 | 1.128 | 52.59 | 68.18 | 0.829 | 0.841 | 0.841 | 0.483 | 1.582 | 3.537 | 1.666 | 3.543 | 7.52 | 3.08 | 8.10 | 6.82 |
| 10 | 0.717 | 0.815 | 0.695 | 1.094 | 56.70 | 77.91 | 0.849 | 0.814 | 0.861 | 0.546 | 1.463 | 3.721 | 1.533 | 3.686 | 7.96 | 3.42 | 8.32 | 7.14 |
| 11 | 0.760 | 0.804 | 0.738 | 1.127 | 55.84 | 72.64 | 0.830 | 0.820 | 0.842 | 0.501 | 1.492 | 3.772 | 1.541 | 3.665 | 8.00 | 3.60 | 8.38 | 7.15 |
| 12 | 0.804 | 0.835 | 0.785 | 1.138 | 56.53 | 69.61 | 0.808 | 0.806 | 0.820 | 0.505 | 1.388 | 3.625 | 1.436 | 3.475 | 8.44 | 3.86 | 8.65 | 7.52 |
| glowa | 0.795 | 0.792 | 0.775 | 1.143 | 8.96 | 11.23 | 0.812 | 0.825 | 0.824 | 0.477 | 1.488 | 3.602 | 1.491 | 3.529 | 7.20 | 3.41 | 7.42 | 7.13 |

### Iloczyny stosunków D_u po 6 półkrokach uwagi i 6 MLP (= wkład do D_u,12/D_u,0), 6 checkpointów

| checkpoint | Π uwaga wsz. | Π MLP wsz. | Π uwaga narz. | Π MLP narz. | Π uwaga tło | Π MLP tło | uwaga ściska | MLP ściska | Σ Δsep_u uwaga | Σ Δsep_u MLP | sep_u 0 | sep_u 12 | r_0 | r_12 | cos_12 | IoU | maks. ciepło |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k0 | 0.689 | 0.938 | 0.701 | 0.935 | 0.689 | 0.938 | 6/6 | 5/6 | +0.010 | +0.006 | 0.840 | 0.855 | 0.772 | 0.912 | 0.952 | 0.000 | 0.014 |
| k6564 | 0.793 | 0.991 | 0.844 | 1.029 | 0.790 | 0.981 | 5/6 | 4/6 | +0.072 | +0.387 | 0.798 | 1.256 | 0.526 | 0.732 | 0.449 | 0.819 | 0.741 |
| k21300 | 0.786 | 0.907 | 0.851 | 1.035 | 0.782 | 0.890 | 6/6 | 4/6 | +0.131 | +0.550 | 0.743 | 1.423 | 0.532 | 0.788 | 0.420 | 0.861 | 0.825 |
| k36100 | 0.843 | 0.844 | 0.862 | 1.004 | 0.840 | 0.825 | 5/6 | 4/6 | +0.136 | +0.615 | 0.764 | 1.515 | 0.564 | 0.799 | 0.410 | 0.859 | 0.911 |
| k50800 | 0.846 | 0.833 | 0.865 | 0.956 | 0.841 | 0.816 | 5/6 | 4/6 | +0.261 | +0.538 | 0.773 | 1.573 | 0.572 | 0.807 | 0.417 | 0.845 | 0.703 |
| k65640 | 0.869 | 0.818 | 0.855 | 1.021 | 0.868 | 0.801 | 5/6 | 4/6 | +0.119 | +0.535 | 0.734 | 1.388 | 0.584 | 0.808 | 0.505 | 0.841 | 0.743 |

Pozostałe 5 checkpointów: tabele 12 półkroków w `Wyniki/Dane/tabele.md`.

### Wszystkie checkpointy (K = 806): wybrane epoki (najbliższy checkpoint)

| krok | epoka | r_12 | r_12 narz. | r_12 tło | cos_12 | sep_u,12 | sep_X,12 | sep_u glowa | D_u,12/D_u,0 | Π uwaga | Π MLP | D_u narz. 12/0 | PR_u,12 | IoU | maks. ciepło |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.912 | 0.966 | 0.911 | 0.952 | 0.855 | 0.848 | 0.855 | 0.646 | 0.689 | 0.938 | 0.656 | 11.35 | 0.000 | 0.014 |
| 50 | 0.05 | 0.870 | 0.919 | 0.871 | 0.881 | 0.984 | 1.049 | 0.984 | 0.669 | 0.735 | 0.910 | 0.689 | 5.68 | 0.000 | 0.196 |
| 110 | 0.10 | 0.801 | 0.847 | 0.806 | 0.755 | 1.029 | 1.055 | 1.030 | 0.767 | 0.823 | 0.931 | 0.819 | 5.49 | 0.202 | 0.310 |
| 270 | 0.25 | 0.828 | 0.818 | 0.834 | 0.757 | 1.024 | 1.020 | 1.025 | 0.655 | 0.791 | 0.828 | 0.785 | 5.95 | 0.496 | 0.450 |
| 550 | 0.50 | 0.813 | 0.750 | 0.821 | 0.699 | 0.991 | 1.028 | 0.993 | 0.668 | 0.770 | 0.868 | 0.903 | 5.93 | 0.655 | 0.578 |
| 1094 | 1.00 | 0.819 | 0.800 | 0.829 | 0.569 | 1.306 | 1.363 | 1.308 | 0.677 | 0.759 | 0.892 | 0.833 | 5.51 | 0.555 | 0.460 |
| 2188 | 2.00 | 0.816 | 0.810 | 0.825 | 0.614 | 1.248 | 1.323 | 1.257 | 0.661 | 0.717 | 0.922 | 0.873 | 5.65 | 0.729 | 0.575 |
| 3282 | 3.00 | 0.807 | 0.749 | 0.819 | 0.522 | 1.239 | 1.331 | 1.251 | 0.682 | 0.728 | 0.937 | 0.907 | 5.52 | 0.831 | 0.677 |
| 5470 | 5.00 | 0.737 | 0.748 | 0.751 | 0.334 | 1.306 | 1.340 | 1.325 | 0.774 | 0.801 | 0.967 | 0.941 | 6.52 | 0.804 | 0.767 |
| 6564 | 6.00 | 0.732 | 0.791 | 0.743 | 0.449 | 1.256 | 1.263 | 1.276 | 0.786 | 0.793 | 0.991 | 0.868 | 6.37 | 0.819 | 0.741 |
| 10940 | 10.00 | 0.793 | 0.792 | 0.808 | 0.337 | 1.535 | 1.626 | 1.568 | 0.691 | 0.780 | 0.886 | 0.887 | 5.47 | 0.855 | 0.709 |
| 16410 | 15.00 | 0.780 | 0.826 | 0.794 | 0.375 | 1.544 | 1.560 | 1.592 | 0.723 | 0.780 | 0.927 | 0.809 | 7.40 | 0.873 | 0.703 |
| 21880 | 20.00 | 0.755 | 0.806 | 0.769 | 0.338 | 1.470 | 1.496 | 1.526 | 0.760 | 0.840 | 0.905 | 0.852 | 7.32 | 0.856 | 0.726 |
| 32820 | 30.00 | 0.782 | 0.841 | 0.793 | 0.485 | 1.440 | 1.439 | 1.508 | 0.746 | 0.841 | 0.888 | 0.816 | 8.04 | 0.880 | 0.677 |
| 43760 | 40.00 | 0.791 | 0.798 | 0.803 | 0.453 | 1.397 | 1.424 | 1.487 | 0.733 | 0.854 | 0.858 | 0.912 | 7.94 | 0.913 | 0.849 |
| 54700 | 50.00 | 0.809 | 0.819 | 0.822 | 0.448 | 1.510 | 1.548 | 1.619 | 0.705 | 0.874 | 0.806 | 0.863 | 7.20 | 0.881 | 0.858 |
| 65640 | 60.00 | 0.808 | 0.806 | 0.820 | 0.505 | 1.388 | 1.436 | 1.488 | 0.711 | 0.869 | 0.818 | 0.873 | 8.44 | 0.841 | 0.743 |

Udział checkpointów (wszystkie 806), w których półkrok ściska D_u (stosunek < 1), i mediana stosunku:

| półkrok | 0→1 (A0) | 1→2 (M0) | 2→3 (A1) | 3→4 (M1) | 4→5 (A2) | 5→6 (M2) | 6→7 (A3) | 7→8 (M3) | 8→9 (A4) | 9→10 (M4) | 10→11 (A5) | 11→12 (M5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| odsetek < 1 | 1.00 | 0.06 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.96 | 0.95 | 0.20 | 0.05 |
| mediana stosunku | 0.927 | 1.053 | 0.960 | 0.954 | 0.957 | 0.934 | 0.969 | 0.907 | 0.981 | 0.955 | 1.022 | 1.084 |
| mediana (epoka ≥ 6) | 0.928 | 1.054 | 0.961 | 0.953 | 0.959 | 0.938 | 0.971 | 0.894 | 0.984 | 0.949 | 1.026 | 1.087 |

## Most 3

### k65640: komutator na blok (S: IoU 0.841, maks. ciepło 0.743); warianty: IoU / maks. ciepło / korelacja mapy z S

| blok | ‖S−R‖/‖S−X‖ | narz. | tło | ‖S−P‖/‖S−X‖ | narz. | tło | cos(S−X,P−X) | narz. | tło | ‖S−R‖_F/‖S−X‖_F | ‖S−P‖_F/‖S−X‖_F | R odwrócona | P równoległa | tylko uwaga | tylko MLP | bez bloku |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.411 | 0.405 | 0.411 | 0.305 | 0.264 | 0.305 | 0.955 | 0.970 | 0.955 | 0.406 | 0.316 | 0.914 / 0.684 / 0.975 | 0.877 / 0.749 / 0.993 | 0.000 / 0.747 / 0.144 | 0.904 / 0.709 / 0.896 | 0.000 / 0.782 / 0.138 |
| 1 | 0.344 | 0.349 | 0.344 | 0.292 | 0.236 | 0.293 | 0.962 | 0.975 | 0.962 | 0.347 | 0.310 | 0.830 / 0.833 / 0.996 | 0.847 / 0.818 / 0.998 | 0.837 / 0.879 / 0.970 | 0.745 / 0.787 / 0.934 | 0.736 / 0.669 / 0.798 |
| 2 | 0.311 | 0.336 | 0.310 | 0.270 | 0.255 | 0.274 | 0.967 | 0.970 | 0.967 | 0.315 | 0.289 | 0.842 / 0.751 / 0.984 | 0.835 / 0.782 / 0.984 | 0.866 / 0.763 / 0.959 | 0.853 / 0.818 / 0.941 | 0.886 / 0.856 / 0.981 |
| 3 | 0.497 | 1.051 | 0.492 | 0.407 | 1.010 | 0.397 | 0.967 | 0.793 | 0.968 | 0.507 | 0.432 | 0.790 / 0.726 / 0.635 | 0.732 / 0.778 / 0.601 | 0.884 / 0.595 / 0.767 | 0.846 / 0.622 / 0.689 | 0.559 / 0.516 / 0.433 |
| 4 | 0.395 | 0.735 | 0.394 | 0.331 | 0.681 | 0.329 | 0.955 | 0.832 | 0.956 | 0.473 | 0.410 | 0.745 / 0.924 / 0.680 | 0.816 / 0.786 / 0.723 | 0.853 / 0.446 / 0.910 | 0.738 / 0.941 / 0.622 | 0.685 / 0.603 / 0.810 |
| 5 | 0.547 | 0.528 | 0.547 | 0.373 | 0.408 | 0.371 | 0.966 | 0.952 | 0.967 | 0.529 | 0.418 | 0.821 / 0.681 / 0.970 | 0.821 / 0.733 / 0.976 | 0.841 / 0.619 / 0.987 | 0.845 / 0.619 / 0.918 | 0.859 / 0.460 / 0.955 |

### Pozostałe checkpointy (mediany; IoU ramki szczytu z GT)

| checkpoint | blok | ‖S−R‖/‖S−X‖ | ‖S−P‖/‖S−X‖ | cos(S−X,P−X) | IoU S | IoU R | IoU P | IoU tylko uwaga | IoU tylko MLP | IoU bez bloku |
|---|---|---|---|---|---|---|---|---|---|---|
| k0 | 0 | 0.366 | 0.255 | 0.968 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k0 | 1 | 0.335 | 0.238 | 0.971 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k0 | 2 | 0.286 | 0.198 | 0.980 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k0 | 3 | 0.248 | 0.198 | 0.980 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k0 | 4 | 0.268 | 0.185 | 0.983 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k0 | 5 | 0.279 | 0.187 | 0.983 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| k6564 | 0 | 0.558 | 0.326 | 0.952 | 0.819 | 0.753 | 0.805 | 0.501 | 0.764 | 0.138 |
| k6564 | 1 | 0.381 | 0.270 | 0.976 | 0.819 | 0.785 | 0.810 | 0.744 | 0.731 | 0.566 |
| k6564 | 2 | 0.227 | 0.129 | 0.994 | 0.819 | 0.829 | 0.824 | 0.842 | 0.832 | 0.838 |
| k6564 | 3 | 0.472 | 0.314 | 0.959 | 0.819 | 0.678 | 0.736 | 0.764 | 0.817 | 0.707 |
| k6564 | 4 | 0.457 | 0.208 | 0.984 | 0.819 | 0.696 | 0.721 | 0.761 | 0.651 | 0.693 |
| k6564 | 5 | 0.259 | 0.143 | 0.996 | 0.819 | 0.811 | 0.824 | 0.793 | 0.845 | 0.856 |
| k21300 | 0 | 0.494 | 0.350 | 0.943 | 0.861 | 0.886 | 0.900 | 0.000 | 0.920 | 0.000 |
| k21300 | 1 | 0.309 | 0.240 | 0.975 | 0.861 | 0.857 | 0.874 | 0.900 | 0.850 | 0.856 |
| k21300 | 2 | 0.270 | 0.213 | 0.982 | 0.861 | 0.885 | 0.876 | 0.864 | 0.896 | 0.916 |
| k21300 | 3 | 0.635 | 0.531 | 0.926 | 0.861 | 0.819 | 0.818 | 0.795 | 0.533 | 0.643 |
| k21300 | 4 | 0.594 | 0.400 | 0.930 | 0.861 | 0.700 | 0.765 | 0.857 | 0.632 | 0.664 |
| k21300 | 5 | 0.380 | 0.217 | 0.978 | 0.861 | 0.842 | 0.872 | 0.887 | 0.833 | 0.834 |
| k36100 | 0 | 0.433 | 0.318 | 0.951 | 0.859 | 0.870 | 0.850 | 0.002 | 0.868 | 0.000 |
| k36100 | 1 | 0.337 | 0.282 | 0.964 | 0.859 | 0.851 | 0.852 | 0.829 | 0.803 | 0.728 |
| k36100 | 2 | 0.290 | 0.223 | 0.980 | 0.859 | 0.883 | 0.861 | 0.888 | 0.894 | 0.923 |
| k36100 | 3 | 0.541 | 0.474 | 0.945 | 0.859 | 0.737 | 0.723 | 0.878 | 0.752 | 0.746 |
| k36100 | 4 | 0.440 | 0.274 | 0.966 | 0.859 | 0.620 | 0.674 | 0.777 | 0.587 | 0.650 |
| k36100 | 5 | 0.407 | 0.212 | 0.979 | 0.859 | 0.860 | 0.853 | 0.855 | 0.789 | 0.792 |
| k50800 | 0 | 0.397 | 0.302 | 0.955 | 0.845 | 0.767 | 0.807 | 0.000 | 0.799 | 0.000 |
| k50800 | 1 | 0.375 | 0.303 | 0.959 | 0.845 | 0.829 | 0.837 | 0.864 | 0.762 | 0.720 |
| k50800 | 2 | 0.320 | 0.292 | 0.961 | 0.845 | 0.830 | 0.828 | 0.824 | 0.802 | 0.782 |
| k50800 | 3 | 0.545 | 0.487 | 0.959 | 0.845 | 0.720 | 0.732 | 0.708 | 0.665 | 0.672 |
| k50800 | 4 | 0.543 | 0.318 | 0.961 | 0.845 | 0.606 | 0.690 | 0.729 | 0.609 | 0.620 |
| k50800 | 5 | 0.394 | 0.267 | 0.971 | 0.845 | 0.834 | 0.813 | 0.837 | 0.870 | 0.884 |
| k65640 | 0 | 0.411 | 0.305 | 0.955 | 0.841 | 0.914 | 0.877 | 0.000 | 0.904 | 0.000 |
| k65640 | 1 | 0.344 | 0.292 | 0.962 | 0.841 | 0.830 | 0.847 | 0.837 | 0.745 | 0.736 |
| k65640 | 2 | 0.311 | 0.270 | 0.967 | 0.841 | 0.842 | 0.835 | 0.866 | 0.853 | 0.886 |
| k65640 | 3 | 0.497 | 0.407 | 0.967 | 0.841 | 0.790 | 0.732 | 0.884 | 0.846 | 0.559 |
| k65640 | 4 | 0.395 | 0.331 | 0.955 | 0.841 | 0.745 | 0.816 | 0.853 | 0.738 | 0.685 |
| k65640 | 5 | 0.547 | 0.373 | 0.966 | 0.841 | 0.821 | 0.821 | 0.841 | 0.845 | 0.859 |
