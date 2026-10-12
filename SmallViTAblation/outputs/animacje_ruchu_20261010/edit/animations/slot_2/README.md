# Slot 2: „Zegar uczenia: co ruch cząsteczek mówi o treningu ViT”

`render.mp4`: 1920×1080, 30 kl./s, H.264 (libx264, CRF 18, yuv420p), ok. 39 s, bez dźwięku.
Pulpit z seriami czasowymi w krokach uczenia i kursorem czasu. Wszystko jest policzone z ruchu 60 000 cząsteczek oraz z logu paczek.

## Dane (tylko odczyt, nic nie jest liczone z surowych trajektorii)
- `../../dane/trajektorie_wlasnosci.npz`: `kroki`, `energia`, `pole`, `temp_ob`, `cos_ob`, `lr`, `strata_po`, `krok_paczki`
- `../../dane/PODSUMOWANIE.json`: liczby do napisów (`efekt_paczki.stosunek` = 1,11, `pole_kolektywne_udzial`, `temperatura_pca32`)
- `SmallViTAblation/assets/MalyViT_35000_od_zera_s47/dane/OCENY.json`: mAP@0,5 po epokach (wpisy z `epoka` ≠ null, krok ≤ 23 000). Pochodzi z **innego biegu** tego samego przepisu, dlatego jest tylko tłem informacyjnym.

Wygładzanie: centrowana mediana ruchoma. Dla udziału pola, temperatury i cos to 15 zapisów w części gęstej i 9 zapisów w części co 10 kroków. Dla straty to 50 kroków. Surowy udział pola i surowa strata są narysowane blado pod liniami wygładzonymi.

## Przebieg
| czas | co widać |
|---|---|
| 0–3 s | tytuł i zdanie, że wszystko pochodzi z ruchu cząsteczek i logu paczek |
| 3–5 s | panele A (lr, udział pola), B (temperatura obszarów, log Y) i C (cos przyrostów) wjeżdżają kolejno |
| 5–20 s | kursor 0 → 2188 z przystankami na 150, 300, 600 i 1000. Pasma faz i ich opisy pojawiają się, gdy kursor je mija. Przy kroku 1000 jest dymek „narzędzie 2,4 × cieplejsze od tła” |
| 20–24 s | panele się zwężają, wjeżdża panel D (strata i mAP@0,5), oś X płynnie przechodzi do 0–23 000, a linia przerywana oznacza zmianę zapisu przy kroku 2188 |
| 25–31 s | kursor 2188 → 22 970, potem znaczniki plateau mAP (~8. epoka) i podpis wniosku |
| 31–39 s | karta 4 wniosków (pojawiają się po kolei); ostatnia klatka stoi 1,9 s |

## Uwagi interpretacyjne (świadome odstępstwa od szkicu zadania)
- **Skok przy kroku 2188.** Od tego miejsca zapis jest co 10 kroków, więc Δx obejmuje 10 kroków. Temperatura rośnie wtedy około 30 ×, a cos przyrostów spada do ≈ −0,1…0. To zmiana zapisu, nie fizyka, i animacja mówi to wprost (linia przerywana z notatką). Spadki widać tylko wewnątrz każdej z dwóch części.
- **Udział pola ≤ 25 %** nie pojawia się zaraz po kroku 300. Przy kroku 300 wynosi 47 %, przy 1000 i 2000 wynosi 22–26 %, a w części co 10 kroków 13–20 %. Teksty podają te liczby.
- **Szczyt temperatury** przypada na kroki ~240–290, czyli koniec „przejścia”. W fazie 300–600 temperatura pozostaje wysoka, a od ~600 maleje (ok. 4 × do kroku 2188).
- **„Narzędzie cieplejsze od tła”** (PCA32): ok. 2,4 × w krokach 600–1000 i 1,9 × do kroku 2188, ale pod koniec biegu już tylko ~5 %. Źrenica jest 1,4–2,1 × cieplejsza od tła przez cały czas po rozgrzewce. W rozgrzewce obszary są praktycznie równe. Uwaga: w pełnych 768D (`temp768_ob`) narzędzie w części co 10 kroków jest *chłodniejsze* od tła. Animacja pokazuje wyłącznie PCA32.
- Rozgrzewka: udział pola 80–88 % (dane), cos kolejnych przyrostów ≈ 0,9.

## Uruchomienie
```
F:\Python\3.13\python.exe script.py              # render.mp4 + verify_00..03.png (0/33/66/100 % czasu)
F:\Python\3.13\python.exe script.py --preview 17 30   # podgląd PNG wybranych sekund (PREVIEW_DIR)
```
Wymaga matplotlib 3.10, ffmpeg w PATH i czcionki Segoe UI (gdy jej brak, używany jest DejaVu Sans). Nie importuje nic z katalogu `app`.
