# Koopman i rozszerzone GLE do doboru klatek

Cząstka jest pojedynczym tokenem patcha. Eksperyment sprawdza, czy klatki wskazane przez zadaniowy zysk z dynamiki tych tokenów uczą detektor lepiej przy mniejszym zbiorze. Wynikiem jest AP detektora i jego krzywa uczenia, nie samo dopasowanie równania.

Aktualny stan i identyfikatory Edena: `WYNIKI.json`. Ustalone przed treningiem kryteria: `PLAN.md`. Uzasadnienie i źródła: `LITERATURA.md`. Ten pilot wybiera stały zbiór przed treningiem i diagnozuje późniejsze checkpointy. Nie twierdzi jeszcze, że selektor reaguje na każdy krok SGD.

## Co jest połączone

```text
L0: 500 obrazów -> bieżący detektor -> tokeny 24x24 patchy każdej klatki
    -> dla każdego patcha: historia filmu -> Koopman + rozszerzone GLE
    -> odczyt narzędzie/tło z tokenu obserwowanego i przewidzianego
    -> dodatnia różnica błędów -> ranking całych klatek
    -> trening całego detektora na wybranych 1500 obrazach
    -> AP i checkpointy 2/4/6/8/10 -> diagnostyka tych samych patchy
```

Token pozostaje osobnym tokenem przez ekstrakcję i równanie. Projekcja Gaussa redukuje 384 kanały do 24, nigdy 576 patchy do jednego wektora. Etykiety prostokątów są używane do oceny rankingu i odczytu pomocniczego. Jest to dobór z już oznaczonej puli, bez obietnicy oszczędzenia niepozyskanych etykiet.

```text
z_hat_next = z + sigma^gamma_D * D
                 + sigma^gamma_JP * JP
                 + sigma^gamma_W * W + Z_anchor + N
```

Skale globalne pochodzą z FIT. Sigma jest własną, przyczynową skalą ruchu patcha. Z jest oczekiwanym powrotem przy braku pewnego odczytu narzędzia. Nie zna zdarzenia w przyszłej klatce. N przewiduje skorelowaną część wcześniejszych reszt, nie losuje szumu.

Źródła portu, zachowane bez zmian:

- `F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_latent_56/model.py`: D, JP, W, N.
- `F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_czlon_zdarzen/model_z.py`: kotwica zdarzeń Z.
- `F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_optymalizacja_95/model_s.py`: własna skala i gamma.
- `F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_krok_po_kroku/README.md`: rozdzielenie FIT, prognozy, obserwacji i aktualizacji historii.

Adaptacje są jawne: 24 zamiast 192 wymiarów, 8 zamiast 56 kroków, metadane obrazu zamiast sprzedaży i kalendarza, tryb przyczynowy zamiast znajomości wejść jutra. Te zmiany wymagają nowej walidacji; wynik TS nie przenosi się automatycznie na obrazy.

## Pliki i wykonanie

- `token_ekstrakcja.py`: hook przed projektorem RF-DETR, jeden wektor na patch, cache z hashami.
- `token_model.py`, `model.py`: pełniejsza struktura równania i skala własna.
- `token_dopasowanie.py`: rozłączne operacje, FIT i odczyt L0, predykcje pojedynczych patchy.
- `token_wybierz.py`, `selekcja.py`: pięć zamrożonych wyborów, kontrola samego błędu i placebo.
- `uruchom.py`, `trening.py`: kolejka pięciu treningów, wspólne COCO i receptura, checkpointy co dwie epoki.
- `diagnostyka_checkpointow.py`: porównanie reprezentacji tych samych obrazów i patchy między checkpointami.
- `run_extract.slurm`, `run_prep.slurm`, `run_pilot.slurm`: wykonanie etapów na Edenie.

Polecenia uruchamiać z katalogu zawierającego `SmallViTAblation` w środowisku RF-DETR1.10.1. Dane i duże artefakty są poza kodem eksperymentu.

```bash
python -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_ekstrakcja \
  --data-root CONTEXT_TRAIN --checkpoint L0.pth --out CACHE/L0_tokens
python -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_wybierz \
  --pool DATA/pool_manifest.jsonl --cache CACHE/L0_tokens \
  --initial-selection DATA/variants/tracking_uniform_time_b500_s42.json --out SELECTION
python -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.uruchom \
  --project-dir SmallViTAblation --variants-dir SELECTION/variants --output-root PILOT \
  --base-protocol SmallViTAblation/configs/protocol_tracking_calibration.yaml --epochs 10
python SmallViTAblation/eden/queue.py init PILOT/queue --queue-file PILOT/queue.jsonl
python -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.uruchom --output-root PILOT --report
```

Nie używać listy zmiennych `sbatch --export=ALL,ROOT=...` na tej konfiguracji Edena: uruchamia mechanizm pobierania środowiska użytkownika, który zawiódł. Ustawiać zmienne poleceniem `env ROOT=... PY=... sbatch --export=ALL ...`. [Dokumentacja Slurm](https://slurm.schedmd.com/sbatch.html).

Testy:

```powershell
py -3.11 -m pytest SmallViTAblation/experiments/dobor_klatek_stan_sieci/tests -q
```

Brak poprawy AP względem samego błędu i placebo jest wynikiem negatywnym dla tej procedury, nawet gdy GLE dobrze przewiduje latent. Jeden seed nie stanowi dowodu przewagi statystycznej. Dobór 1500 wobec 3000 trzeba oceniać razem z liczbą ekspozycji i kosztem samego selektora.
