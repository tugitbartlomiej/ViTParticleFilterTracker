# Przejściowe wzmocnienie i pseudospektrum pola MLP: co wynika z nienormalności

**Data:** 2026-10-26, 21:12 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach (bez treningu) na trzech układach.
**Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane przed uruchomieniem).

## Cel

Rodzice wykazały, że trening czyni pole MLP nienormalnym (‖N‖/‖J‖ 0,76 → 0,92) w detekcji na 24 seedach i na CIFAR-10 w MałymViT
i DeiT-Tiny, także bez weight decay i przy losowych etykietach. Nie wiadomo, czy to ma skutek dla działania sieci. Nienormalność
w teorii (Trefethen–Embree) oznacza, że ‖Aᵏ‖ może przejściowo rosnąć ponad ρ(A)ᵏ, a zaburzenia są wzmacniane bardziej, niż mówi widmo.
Ten eksperyment mierzy to wprost: (W) liniowo na jakobianach i (F) nieliniowo na sieci przez zaburzanie tokenów.

## Rodzice i źródła

- `experiments/mosty_fizyki_24seedy_ablacje_20261026__13-04-31`: kod pomiaru detekcji (`pomiar_mosty_eden.py`: klatki diagnostyczne,
  maski, jakobiany), `model.py`; checkpointy kampanii `kompensacja_funkcjonalna_dgx2_20261017` seedów 3047, 4047, 5047 (k0, 3000, 65 640).
- `experiments/kompensacja_lodyga_i_cifar_20261026__19-01-59`: kod pomiaru CIFAR (`pomiar_mosty_cifar.py`, `model_deit_cifar.py`);
  checkpointy biegów `maly_s47_obciecie` i `s47_obciecie` z `2026-10-13_deit_cifar10` (k0, 3000, 93 780).
- Teoria: Trefethen & Embree, *Spectra and Pseudospectra* (2005): stała Kreissa, abscysa numeryczna, wzrost przejściowy;
  Henrici (1962): odchylenie od normalności.

## Procedura

`Kod_eksperymentu/wzmocnienie.slurm`, tablica 5 zadań (1 × A100 + 4 CPU, partycja short): `wzmocnienie_eden.py --uklad {detekcja,
cifar_maly, cifar_deit}`. Klatki: detekcja 50 klatek diagnostycznych TEST (tokeny: wszystkie narzędzia + 20 losowych tła), CIFAR 50 obrazów
testowych (5 na klasę, 16 losowych tokenów obrazu). Dla każdego bloku i tokenu: J jak w rodzicu, A = I + J.

- **W (pula procesów, numpy):** λ(J); ρ = max|1+λ|; ‖Aᵏ‖₂ dla k = 1…6 i G_k = ‖Aᵏ‖₂/ρᵏ; ‖A‖_F i √Σ|1+λ|²; abscysa numeryczna
  ω = λ_max((J+Jᵀ)/2) i max Re λ; dolne oszacowanie stałej Kreissa K = max (|z|−ρ)/σ_min(zI−A) na siatce |z| = ρ(1+ε),
  ε ∈ {0,02; 0,05; 0,1; 0,2; 0,4}, 12 kątów; ‖N‖/‖J‖ Henriciego. Złożenie po blokach dla tego samego tokenu: ‖A_NB⋯A_1‖₂ wobec Π ρ_b.
- **F (GPU, nieliniowo):** zaburzenie wybranych tokenów przed półkrokiem MLP o δ·‖x_i‖ (δ ∈ {0,01; 0,03; 0,1}) w kierunku losowym
  albo najgorszym (prawy wektor osobliwy σ_max(A_i)); wzmocnienie ‖Δx_out‖/‖Δx_in‖ w samym półkroku wobec przewidywań liniowych
  (losowy: ‖A‖_F/√192 nienormalne i √(Σ|1+λ|²/192) normalne; najgorszy: σ_max wobec ρ); skutek dla wyjścia po reszcie sieci
  (detekcja: ΔIoU ramki szczytu, korelacja mapy ciepła; CIFAR: Δmargines logitów).

Wyjście: `Wyniki/Dane/pomiar/<bieg>/{wyniki.npz, podsumowanie.json, WAGI_SHA256.txt, COMPLETE.json}`.
**Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`.

## Ograniczenia

- Stała Kreissa z siatki 60 punktów to dolne oszacowanie; pseudospektrum nie jest rysowane w całości (koszt), tylko próbkowane.
- Zaburzenia dotyczą wybranych tokenów jednej klatki, nie całej chmury; uwaga między tokenami nie jest zaburzana bezpośrednio.
- 3 checkpointy na bieg (k0, wczesny, koniec); trend w treningu jest zgrubny.
- 5 biegów (3 detekcja, 1 MałyViT CIFAR, 1 DeiT): test znaku 5/5 jest ostry, ale mało biegów.

## Odtwarzanie

```
python experiments/eden.py init wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11
python experiments/eden.py push wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11
python experiments/eden.py submit wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11 --script Kod_eksperymentu/wzmocnienie.slurm
python experiments/eden.py watch wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11 --pull-on-complete
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próba dymna: `--sbatch-arg=--array=0,4 --script-arg=proba` (2 klatki / obrazy, 2 checkpointy).

## Stan

Zobacz `Wyniki/Dane/Eden/STATUS_EDEN.txt` i koniec tego pliku.

- 2026-10-26 21:28–21:33: próby dymne 1933042_0 (detekcja s3047) i 1933042_4 (DeiT): 2 klatki / obrazy, 2 checkpointy, 30–45 s; wyniki
  w `Wyniki/Dane/proby_dymne/`, nie wchodzą do oceny. Pełne zadania: 1933046 (CIFAR, 16 CPU na dgx-3), 1933047 (detekcja s3047, 3 CPU),
  1933048 (detekcja s4047/s5047, 16 CPU na dgx-3) anulowane jako ReqNodeNotAvail i zgłoszone ponownie jako 1933050 (8 CPU, bez węzła).
  Skrypt SLURM przekazuje `--cpus $SLURM_CPUS_PER_TASK` (poprawka przed pełnymi zadaniami).
- 2026-10-26 21:58: biegi CIFAR ukończone (DeiT 18 min, MałyViT 10 min), detekcja w toku. **Błąd w części F** (wykryty na wynikach CIFAR: Δ wyjścia
  nie zależało od δ): zaburzenie było przykładane do stanu 2b (wejście bloku) zamiast 2b+1 (po uwadze bloku b), więc ścieżka zaburzona
  pomijała uwagę bloku b, a Δ wyjścia mierzyło głównie tę różnicę. Jakobian J był liczony w poprawnym punkcie (wejście MLP), więc część W
  jest poprawna. Poprawka: `stany[2 * b + 1, i]`; tryb `--bez_W` / `tylkoF` przelicza samą część F (GPU, kilka minut na bieg) do
  `wyniki_F.npz`, a ocena czyta F z tego pliku, gdy istnieje. Progi predykcji bez zmian. Część F z pierwszego biegu nie wchodzi do oceny.
- Uwaga po fakcie do F1: przewidywanie dla kierunku losowego ‖A‖_F/√192 jest bliskie 1 (pole MLP jest małe względem identyczności
  w 192 wymiarach: 1,00–1,03), więc warunek pred_nn / pred_norm > 1,15 nie może zajść przy tej definicji; F1 będzie oceniona
  według zapisanego progu (FAIL), a różnica nienormalne / normalne jest widoczna tylko w najgorszym kierunku (F2, F3).
- 2026-10-26 22:13: po poprawce: zadanie detekcji s3047 (1933047, 3 CPU, 1/3 checkpointu po 36 min) anulowane; kod zaktualizowany na Edenie
  (`push --update-code`); s3047 zgłoszone ponownie na węźle CPU sr-3 (1933064, 24 CPU, bez GPU, torch na CPU; W i F liczone na nowo
  poprawionym kodem), przeliczenie samej części F dla 4 gotowych biegów (1933065_1…_4, tryb tylkoF, 1 CPU + A100 na dgx-2).
- Wstępne wyniki W (poprawne, przed oceną końcową) dla 4 biegów, blok środkowy, k0 → koniec: G₁ detekcja s4047 1,11 → 2,59,
  s5047 1,12 → 1,66; MałyViT CIFAR 1,12 → 1,30; DeiT CIFAR 1,07 → 1,14. G₆: 1,6 → 5,3 / 1,8 → 3,7 / 1,7 → 2,5 / 1,5 → 1,8.
  Kreiss: 2,3 → 5,1 / 2,5 → 4,0 / 2,0 → 3,3 / 2,1 → 3,0. Efekt w detekcji wyraźnie silniejszy niż w klasyfikacji CIFAR. Blok 1 ma
  największe G₁ (1,6–2,4) i promień widmowy ρ 2,6–13 (inne bloki 1,1–1,6). Opisowo (detekcja, blok 4, koniec): tokeny tła mają
  większe wzmocnienie niż tokeny narzędzia (s4047: G₁ 2,76 vs 1,44, Kreiss 5,65 vs 3,74; s5047: 1,74 vs 1,37) — pole wokół narzędzia
  jest „spokojniejsze”, co koresponduje z nieudaną P3 rodzica (tr J) mierzoną innym wskaźnikiem.
- 2026-10-26 22:30: komplet 5/5 (s3047 na sr-3 w 16 min: W na 24 CPU, F na CPU; przeliczenia F 1933065_1…_4 po 6–8 min). Ocena końcowa
  (`Wyniki/Dane/OCENA_PREDYKCJI.txt`, progi bez zmian): **W3 PASS 5/5** (stała Kreissa bloku środkowego 2,0–2,5 → 3,0–5,1),
  **W4 PASS 5/5** (luka abscysy numerycznej rośnie 2–10×), **F2 PASS 5/5** (zmierzone nieliniowo wzmocnienie w najgorszym kierunku =
  σ_max co do 1 %: 3,21/3,22, 4,47/4,51, 2,06/2,07, 1,47/1,48, 1,21/1,21), **F3 PASS 5/5** (najgorszy kierunek zmienia wyjście
  2,6–7× bardziej niż losowy). FAIL: W1 3/5 (G₁ > 1,5 tylko w detekcji: 2,15 / 2,59 / 1,66; CIFAR 1,30 / 1,14, przy k0 1,07–1,13),
  W2 4/5 (G₆ ≥ 2 poza DeiT 1,81; detekcja 5,1 / 5,3 / 3,7), W5 2/5 (złożenie bloków wzmacnia bardziej niż pojedynczy blok tylko w CIFAR;
  w detekcji G_P 0,97–1,21 < mediana G₁ 1,65–1,76, bo kierunki najgorsze kolejnych bloków nie są zgodne), W6 4/5 (G₁(k0) DeiT 1,07 < 1,1),
  F1 0/5 (źle skalibrowana: wzmocnienie losowego kierunku 1,00–1,05, różnica nienormalne/normalne 0–7 %), F4 0/5 (wyjście odporne we
  wszystkich blokach: ΔIoU 0,006–0,011 przy δ = 0,1 losowo; żaden blok nie dominuje), F5 4/5 (poza s5047).
- Interpretacja: nienormalność ma skutek dynamiczny potwierdzony nieliniowo (F2, F3) i rosnący w treningu (W3, W4), silny w detekcji
  (wzmocnienie jednego półkroku 2,2–4,5 wobec widmowego 1,2–1,6), słaby w DeiT na CIFAR (1,2 wobec 1,06). Skutek dla wyjścia jest mały:
  δ = 0,03 w najgorszym kierunku na ~37 tokenach zmienia IoU o 0,005–0,013 (CIFAR: margines o 0,07–0,11), 3–7× więcej niż losowo.
  Mechanizm istnieje, konsekwencja dla wyniku jest ograniczona; wyjście sieci jest odporne.
