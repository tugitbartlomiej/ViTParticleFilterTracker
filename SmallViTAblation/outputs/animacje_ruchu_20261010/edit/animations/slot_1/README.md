# Slot 1: Trzy reżimy ruchu cząsteczki, MSD(τ)

`render.mp4`: 1920×1080, 30 kl./s, H.264 (libx264, High), yuv420p, CRF 18, 30,5 s (915 klatek), bez dźwięku.

## Co pokazuje
Krzywe MSD(τ) tokenów ViT („cząsteczek” latentu) w skali log-log, osobno dla czterech obszarów oka
(tło, rogówka, źrenica, narzędzie). Krzywe rysują się od τ = 1 do 1500 kroków uczenia, a licznik
po prawej pokazuje bieżące τ, MSD i lokalny wykładnik α(τ) = d log MSD / d log τ. Potem markery
przejeżdżają kolejno do trzech reżimów i podświetlają je wraz z przerywaną linią nachylenia:
- balistyczny: α ≈ 1,8 (τ = 1–3, marker przy τ = 2),
- klatka: α ≈ 0,1 (τ = 30–100, marker przy τ = 80),
- ucieczka dyfuzyjna: α ≈ 1 (τ ≈ 1000, marker przy τ = 988).

Na końcu pojawia się panel VACF (+0,73 po 1 kroku, ≈ −0,1 po 10 krokach; średnie z czterech obszarów
z `PODSUMOWANIE.json`) i wniosek: dynamika szklista.

| czas [s] | faza |
|---|---|
| 0–3,6 | tytuł i zdanie o MSD, potem przejazd do nagłówka |
| 3,3–12,3 | osie, licznik α(τ), rysowanie krzywych |
| 14,0–23,6 | trzy reżimy, każdy z pauzą ≥ 1,6 s |
| 23,6–30,5 | panel VACF, wniosek, ostatnia klatka trzymana 2 s |

## Dane (tylko odczyt, bez liczenia z trajektorii)
`../../dane/trajektorie_wlasnosci.npz`: `taus` (32 wartości 1–1500), `msd_stac` (32×4), `alfa_lok` (32×4),
`lags` i `vacf` (34×4); `../../dane/PODSUMOWANIE.json`: `vacf_lag1_po_rozgrzewce`, `vacf_lag10_po_rozgrzewce`.
Między punktami danych krzywe interpolowane są liniowo w log-log, a punkty danych zaznaczono kropkami.
Na brzegach, gdzie `alfa_lok` ma NaN (τ = 1 i τ = 1500), licznik pokazuje najbliższą zdefiniowaną wartość
(τ = 2 albo τ = 1217).

## Odtworzenie
```
F:\Python\3.13\python.exe script.py                    # render.mp4 + verify_00..03.png (0/33/66/100 %)
F:\Python\3.13\python.exe script.py --podglad 16.5,30  # pojedyncze klatki PNG
F:\Python\3.13\python.exe script.py --audyt 16.5,30.4  # kontrola nachodzenia tekstu na tekst i krzywe
```
Wymaga matplotlib 3.10, numpy, ffmpeg w PATH i czcionki Segoe UI (`C:\Windows\Fonts\segoeui.ttf`, `seguisb.ttf`).
Jeśli czcionka się nie wczyta, skrypt kończy się błędem, bez cichego zastępstwa. Render trwa około 2 minut.
