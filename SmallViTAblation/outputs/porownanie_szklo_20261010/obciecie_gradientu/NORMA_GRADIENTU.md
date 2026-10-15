# Obcięcie gradientu w MałymViT: jak często działało i czy psuło „temperaturę” (2026-10-12)

Pomiar: `pomiar_normy_gradientu.py` → `NORMA_GRADIENTU.json` (log: `pomiar_normy_gradientu.log`).
W każdym biegu po `L.backward()` jest `torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)`:
`07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/trening.py:80`. Biegi z pokrętłami, także SGD, wołają ten sam `Uczen.kroki`.

Jak mierzone:
- Warunki kroku uczenia są odtworzone: 3072 losowe klatki z 35 000 (SHA-256 zgodne z manifestem), 480 × 320, odbicie 0,5, jasność i kontrast ±20 %, autocast bf16, `M.strata`, BatchNorm łodygi w trybie train.
- Liczona jest norma całego gradientu **przed** obcięciem: 64 paczki na checkpoint dla AdamW i 48 dla SGD. Paczka ma rozmiar jak w biegu (16–128); każda zmieściła się w GPU w całości.
- Niezależna kontrola z prawdziwej historii uczenia: w AdamW Σ v̂ to średni kwadrat normy *obciętego* gradientu z ok. 1000 ostatnich kroków. Wartość 1 znaczy, że obcięcie działało w praktyce w każdym kroku.
- Stany biegów z pokrętłami pobrano z Edena (36 plików, SHA-256 zgodne) do `assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/` (`SHA256_POBRANE_2026-10-12.txt`).
- Stan `01_bieg_35000_gesty_zapis/eden/pelny/pelny/stan_e59.pt` i `stan_e60.pt` ma wagi identyczne z biegiem s47.

## Werdykt

1. **Obcięcie było aktywne prawie w każdym kroku — TAK.**
   - Bieg s47 (AdamW 5e-4): norma przekraczała 1 w 100 % paczek od kroku 0 do 60. epoki. Mediana normy wynosi od 9,8 na starcie do 2,4 w 60. epoce; nawet 10. percentyl nie schodzi poniżej 1,9.
   - Historia Adama potwierdza to niezależnie: średni kwadrat obciętej normy to 1,000 przed 59. i 60. epoką oraz w oknie 12 034–13 128 wszystkich trzech biegów z pokrętłami.
   - SGD: 100 % przy lr 0,03 i 0,1 (każda paczka), 40–88 % przy lr 0,3.

2. **Czy tłumaczy brak temperatury?**
   - **AdamW (τ_1e ∝ η^−0,5): NIE.**
     - Obcięcie to prawie stały mnożnik, który Adam znosi, bo normuje skalę gradientu. Zmienność mnożnika jest mała i taka sama przy każdym lr (współczynnik zmienności 0,16–0,21).
     - Zmierzona ze stanu optymalizatora długość kroku Adama rośnie dokładnie liniowo z lr: lr^1,03 (0,042 / 0,17 / 0,73 dla lr 1,25e-4 / 5e-4 / 2e-3).
     - „Temperatura” jako długość kroku jest więc ∝ lr, tak jak bez obcięcia, i η^−0,5 ma inne źródło.
   - **SGD (brak prawa „temperatura ∝ lr/paczka”): TAK.**
     - Obcięty gradient ma normę 1, więc SGD z obcięciem działa jak znormalizowany SGD z pędem. Długość kroku to lr · ‖bufor pędu‖ ≈ (1,5–2,1) · lr, niezależnie od paczki.
     - Przy lr 0,1 krok wynosi 0,19–0,21 przy paczce 16, 0,18 przy 32 i 64 oraz 0,15–0,16 przy 128. Ośmiokrotna zmiana paczki zmienia krok najwyżej o około jedną trzecią (0,21 wobec 0,15), a przy zwykłym SGD szum spadłby √8 ≈ 2,8 raza.
     - To potwierdza przypuszczenie z README (sekcja 6c): „obcięcie ustala normę kroku ≈ lr niezależnie od szumu paczki”.

## Bieg s47 (AdamW 5e-4, paczka 32)

| krok | epoka | obcięte | mediana ‖g‖ (p10–p90) | średni mnożnik min(1, 1/‖g‖) | zmienność mnożnika (cv) | strata |
|---|---|---|---|---|---|---|
| 0 | 0 | 100 % | 9,77 (9,54–9,95) | 0,10 | 0,02 | 5,66 |
| 100 | 0,09 | 100 % | 6,21 (4,99–7,76) | 0,16 | 0,19 | 2,78 |
| 1000 | 0,91 | 100 % | 7,54 (6,45–9,29) | 0,13 | 0,13 | 1,18 |
| 1094 | 1 | 100 % | 6,21 (5,30–7,41) | 0,16 | 0,16 | 1,13 |
| 5470 | 5 | 100 % | 3,78 (3,06–4,80) | 0,27 | 0,17 | 0,58 |
| 10 940 | 10 | 100 % | 3,62 (2,82–4,55) | 0,28 | 0,20 | 0,47 |
| 12 034 | 11 | 100 % | 4,53 (3,79–5,54) | 0,22 | 0,17 | 0,46 |
| 13 128 | 12 | 100 % | 3,14 (2,51–4,01) | 0,32 | 0,19 | 0,42 |
| 21 880 | 20 | 100 % | 3,47 (2,78–4,27) | 0,30 | 0,18 | 0,37 |
| 43 760 | 40 | 100 % | 2,96 (2,27–4,01) | 0,35 | 0,21 | 0,28 |
| 65 640 | 60 | 100 % | 2,41 (1,94–3,13) | 0,42 | 0,21 | 0,25 |

Pozostałe kroki (10, 50, 300, 500) są w JSON; wszędzie obcięte jest 100 %.

Kto robi normę gradientu: w 60. epoce 75 % ‖g‖² pochodzi z głowy, 14 % z łodygi, a 6 bloków razem daje ok. 11 %. Na starcie głowa ma 41 %, a łodyga i każdy blok po 7–10 %; w 1. epoce łodyga ma 50 %. Wspólny mnożnik obcięcia ustawia więc głównie głowa.

## Biegi z pokrętłami, AdamW, okno pomiaru τ (stan po 11. i 12. epoce)

| lr | obcięte | mediana ‖g‖ | średni mnożnik | cv mnożnika | średni kwadrat obciętej normy z historii Adama | długość kroku Adama lr·‖m̂/(√v̂+ε)‖ |
|---|---|---|---|---|---|---|
| 1,25e-4 | 100 % / 100 % | 5,03 / 5,59 | 0,19 / 0,18 | 0,21 / 0,17 | 1,000 / 1,000 | 0,043 / 0,042 |
| 5e-4 | 100 % / 100 % | 4,48 / 3,29 | 0,23 / 0,30 | 0,17 / 0,19 | 1,000 / 1,000 | 0,177 / 0,160 |
| 2e-3 | 100 % / 100 % | 1,89 / 2,28 | 0,54 / 0,44 | 0,17 / 0,16 | 1,000 / 1,000 | 0,718 / 0,744 |

Wykładniki: długość kroku Adama ∝ lr^1,03, a surowa norma gradientu ∝ lr^−0,34.

## Biegi SGD z pędem 0,9 (dwie ostatnie epoki)

| lr | paczka | obcięte | mediana ‖g‖ | średni mnożnik | ‖bufor pędu‖ | długość kroku lr·‖bufor‖ |
|---|---|---|---|---|---|---|
| 0,03 | 32 | 100 % / 100 % | 5,03 / 4,30 | 0,20 / 0,23 | 1,88 / 1,87 | 0,056 / 0,056 |
| 0,1 | 32 | 100 % / 100 % | 2,69 / 1,73 | 0,37 / 0,59 | 1,75 / 1,79 | 0,175 / 0,179 |
| 0,3 | 32 | 88 % / 40 % | 1,21 / 0,95 | 0,83 / 0,96 | 1,67 / 1,56 | 0,50 / 0,47 |
| 0,1 | 16 | 100 % / 100 % | 2,26 / 2,14 | 0,45 / 0,48 | 1,91 / 2,06 | 0,19 / 0,21 |
| 0,1 | 64 | 100 % / 100 % | 1,97 / 1,65 | 0,50 / 0,61 | 1,83 / 1,83 | 0,18 / 0,18 |
| 0,1 | 128 | 100 % / 100 % | 1,96 / 1,43 | 0,52 / 0,70 | 1,54 / 1,62 | 0,15 / 0,16 |

Bufor pędu z wektorów o normie 1 miałby normę 2,29 przy kierunkach niezależnych i 10 przy zgodnych. Zmierzone 1,5–2,1 znaczy, że kolejne obcięte gradienty są prawie niezależne, a nawet lekko przeciwne (oscylacja w dolinie).

## Co dalej wynika (hipoteza, nie sprawdzana tutaj)

W oknie pomiaru τ biegi o różnym lr nie siedzą w tym samym dołku: surowa norma gradientu maleje z lr (5,3 / 3,9 / 2,1, czyli ∝ lr^−0,34), a strata jest różna (0,42 / 0,42 / 0,51). Porównanie τ między lr zakłada jeden dołek w różnych temperaturach. Jeśli większy lr prowadzi do innych, np. płaskich, rejonów krajobrazu, wykładnik τ może się różnić od −1 bez udziału obcięcia.

## Zastrzeżenia

- Paczki są losowane na nowo z puli 3072 klatek: mają ten sam rozkład co w uczeniu, ale nie są to te same paczki. Historia Adama (Σ v̂) jest niezależnym potwierdzeniem z prawdziwych kroków.
- Dla SGD to końcówki krótkich biegów (2–9 epok), a nie okno 12 034–13 128.
