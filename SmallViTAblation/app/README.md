# Edytor latentu SD — GUI

Interaktywna aplikacja (PyQt6) do edycji pełnego latentu 600x768 i generowania
obrazów z wytrenowanego Stable Diffusion (`vit_sd_pelny_latent_filmy_eden_20261006`,
checkpoint krok 8500, tryb cross-attention).

## Uruchomienie

```bat
uruchom.bat
```

albo ręcznie: `python app.py` z tego katalogu (`SmallViTAblation/app/`).

Wymagania: torch+CUDA, diffusers 0.39.0, timm, PyQt6, opencv-python, PIL.
Wszystkie wagi są lokalne; pierwsze ładowanie weryfikuje sha256 bazowego UNet/VAE
i może potrwać około minuty.

## Model danych

UNet dostaje warunek 600 tokenów x 768 kanałów. Suwaki edytują jeden wektor
768 wartości; tryb decyduje o jego zastosowaniu do latentu bazowego:

- Offset do wszystkich tokenów: `latent = baza + wektor`,
- Offset do jednego tokena: `latent[t] = baza[t] + wektor`,
- Ustaw token na wektor: `latent[t] = wektor`.

Latent bazowy: domyślnie naturalny latent `before` z
`outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz`; można go wymienić na
naturalny latent ViT z dowolnego obrazu albo na tablicę 600x768 z pliku NPZ.

Suwak ma zakres z per-kanałowych statystyk bazy (min/max + 3 sigma) albo
bipolarny −M..0..+M (M = max(|min|, |max|) kanału, zero dokładnie na środku —
do wyboru w „Zakres suwaków", wygodne dla offsetów dodatnich/ujemnych); pole
liczbowe przyjmuje dowolną wartość spoza zakresu. 768 kanałów w 12 stronach
po 64 suwaki; szybki skok do kanału polem „Skok do". Każda kontrolka ma dymek
z wyjaśnieniem (hover).

Jednostki pól liczbowych: „Absolutne" (surowe wartości latentu — takich oczekuje
model) albo „% zakresu kanału (0–100%)" (0% = minimum kanału w bazie, 100% =
maksimum; suwak odpowiada dokładnie 0–100%). Procenty to tylko sposób
wyświetlania — wektor w pamięci i generacja zawsze używają wartości absolutnych.
Latent to aktywacje po GELU: naturalne wartości są zawsze >= -0,17, górna granica
nie istnieje teoretycznie (w naturalnym latencie bazowym max ≈ 4,0); model był
uczony na surowych wartościach bez normalizacji.

## Limity makrowektorów (2026-10-06)

Makrowektor = jeden z 768 kanałów; każdy ma własny przedział [min_c, max_c] (grupa „Limity makrowektorów”).
Źródła limitów:

- Matematyczne z wag ViT (domyślne): dokładny przedział dla ustalonych wag checkpointu (k1000: min −0,17,
  max 7,02–10,24). Przycisk „Checkpoint ViT...” wybiera inne wagi, np. z innego kroku uczenia.
- Po T krokach uczenia AdamW: twarda granica dla dowolnych danych po T krokach przepisu `maly_vit_od_zera`
  (wspólna dla kanałów; T=100: 17,4, T=1000: 1896; bez limitu kroków granica nie istnieje).
- Statystyki latentu bazowego, plik (NPZ `h_min`/`h_max`, `min`/`max`, `lo`/`hi`; CSV `kanal,min,max`; JSON),
  ręczne wspólne min/max oraz nadpisanie limitu jednego kanału („Ustaw dla kanału”).
- „Zapisz limity” zapisuje NPZ i CSV do `outputs/latent_editor_sd_20261006/`; zapis latentu NPZ dołącza limity.

Wyprowadzenie granic: `experiments/vit_granice_latentu/README.md`.

Zmiana procentowa: jednostka „% zmiany kanału: −100% (do min) .. 0 .. +100% (do max)” — przesunięcie kanału c o
p % × (max_c − min_c); suwak to −100..+100 %. Działa z trybami „Offset do …”. „Przycinaj latent efektywny
do limitów” (domyślnie włączone) przycina każdą wartość do [min_c, max_c]; pasek statystyk podaje, ile
wartości wyszło poza limity. „Zakres suwaków = Limity kanału” ustawia suwak na [min_c, max_c].

Zakładka „Narzędzie: 128 kanałów”: tylko kanały J128 (`KANALY128_PODSUMOWANIE.csv`), posortowane wg delta RMS;
dymek etykiety pokazuje statystyki sztucznej korekty i limit. „Wszystkie 128 o X %” przesuwa każdy z nich
o X % szerokości jego limitu. Obie zakładki edytują ten sam wektor 768 wartości.

## Generowanie

GENERUJ OBRAZ uruchamia DDIM (eta=0, BF16 na CUDA, decode VAE FP32) — identycznie
jak `generate.generate` z eksperymentu — w osobnym wątku, z paskiem postępu.
Kroki 1-200 (domyślnie 50), ziarno ustawiane.

## Generowanie na bieżąco

„Generuj na bieżąco po zmianie” uruchamia generację automatycznie po każdej zmianie suwaka, pola, limitu,
trybu albo bazy, gdy przez „Opóźnienie” (domyślnie 300 ms) nic się nie zmienia. Używa osobnej liczby
kroków DDIM (domyślnie 15). Zmiana w trakcie trwającej generacji jest liczona zaraz po niej, zawsze
z ostatnim stanem latentu. Błąd generacji wyłącza tryb, żeby nie zapętlić okien z błędem.

## Start z zaszumionego zdjęcia (img2img)

Grupa „Start odszumiania”: „Czysty szum” (jak dotąd) albo „Zaszumione zdjęcie (img2img)”. Zdjęcie startowe
(„Zdjęcie startowe...” albo „Jak baza” = zdjęcie, z którego policzono latent bazowy) jest skalowane do 576x384
(INTER_AREA, jak w treningu), kodowane VAE (średnia rozkładu, FP32), dostaje szum DDPM do kroku czasowego
odpowiadającego „Sile szumu”, a potem ostatnie round(kroki x siła) kroków DDIM odszumia je sterowane edytowanym
latentem 600x768. Siła 1,0 = czysty szum. „Podgląd kolejnych kroków co N” pokazuje obraz w trakcie
odszumiania: bieżący stan z szumem albo przewidywany obraz końcowy (pred x0); każdy podgląd to dodatkowy
decode VAE. Zapis latentu NPZ zawiera nazwę zdjęcia startowego i siłę szumu.

## Wybór patcha (tokena) na obrazie

Obraz jest rysowany z siatką 20x30 (600 tokenów). Kliknięcie patcha wybiera token (token = wiersz x 30 +
kolumna) i, jeśli był tryb „Offset do wszystkich tokenów”, przełącza na „Offset do jednego tokena”.
Wybrany patch jest podświetlony; pod obrazem jego numer, wiersz, kolumna i piksele wejścia ViT 480x320.
Siatkę można ukryć.

## Zakładka „Detekcja ViT i auto-szukanie latentu”

1. „Wczytaj zdjęcie...” (etykieta cystotomu z annotation.csv, gdy nazwa wskazuje film i klatkę; inaczej
   „Rysuj etykietę myszą”) albo „Obraz z SD” (ostatni obraz z Edytora, etykieta zdjęcia bazy przeniesiona
   w przybliżeniu). 2. „Checkpoint ViT...” (dowolny krok, słownik z kluczem net; domyślnie k1000).
   „Wykryj”: ramka ViT (zielona), etykieta (cyjan), pewność, IoU, AP@0,5, mAP@0,5:0,95 na tym obrazie.
   „mAP na zdjęciach testowych”: AP@0,5 i mAP@0,5:0,95 na `zdjeciatestowe/` (k1000: 43,5 % i 11,7 %, 18 zdjęć).
3. „Auto-szukanie latentu”: Adam po wartościach latentu tego zdjęcia (128 kanałów J128 albo wszystkie 768),
   strata jak w `experiments/vit_latent128_jedno_zdjecie_90/wyznacz.py` (logit środka etykiety, tłumienie
   reszty, ranking, ramka, GIoU), po każdym kroku przycięcie do limitów makrowektorów z Edytora. Ogon sieci
   od latentu jest dokładny (zgodność z pełnym forwardem 0,0). Wagi i piksele bez zmian.
   Wynik: porównanie PRZED/PO na zdjęciu, mapy latentu 600x768 i 600x128 (przed, po, różnica) i średnie
   kanałów; zapis `outputs/latent_editor_sd_20261006/auto_szukanie_*.npz` + `.json`.
   Zakładka „Makrowektory: 768 wartości globalnie (−/+)”: jedna wartość na kanał zebrana po 600 tokenach
   (średnia lub suma ze znakiem, RMS = siła, maksimum), słupki w górę i w dół od zera na osi −M..+M,
   przed i po, różnica po − przed oraz same J128 w kolejności siły korekty.
   To tylko podsumowanie do oglądania: model nie liczy latentu globalnego, ViT i adapter SD działają na
   każdym z 600 tokenów osobno. Przetasowanie tokenów zostawia makrowektory bez zmian, a zmienia detekcję
   (po auto-szukaniu J128: pewność 0,84 -> 0,35, IoU 1,00 -> 0,73).
   Zakładka „Cząsteczki: okręgi 768 wektorów na patchach” (`czasteczki.py`): w środku każdego z 600 patchy
   okrąg z 768 wektorów; kanał c ma stały kąt 360° · c / 768 i własny kolor z koła barw. Długość = okrąg zera
   + skala · wartość (krótszy od szarego okręgu = ujemny). Brzeg patcha = 99,5. percentyl |wartości|.
   Tryby: przed, po, różnica, przed (szary) i po (kolor) nałożone; kanały: wszystkie, co 2./4./8., tylko J128.
   Nakładki: obraz (przezroczystość) i atencja ViT wybranego bloku (średnia otrzymana albo z klikniętego
   patcha; liczona z obrazu, auto-szukanie jej nie zmienia). Rysunek 768 x 600 wektorów trwa kilka sekund.
   „Latent PO → baza Edytora SD” pozwala wygenerować obraz z zoptymalizowanego latentu.

Przykład (TR_0001_S1_P03 kl. 870, k1000): przed pewność 72,5 %, IoU 66,0 %, mAP@0,5:0,95 40 %;
po J128 w limitach z wag (600 kroków): pewność 84,0 %, IoU 99,98 %, mAP 100 %. Kanały J128 dochodzą do
górnych limitów (8,67). Wszystkie 768 kanałów w zakresie [−0,17; 4,43]: pewność 96,9 %, IoU 99,8 %.

Uruchamiać Pythonem z diffusers (na tym komputerze F:/Python/3.13, pierwszy na PATH).

### Zmienny lr auto-szukania

Pole „Harmonogram lr”: stały; cosinusowy (od lr do „lr min”); liniowy; krokowy (połowa 4 razy w trakcie);
plateau (połowa, gdy strata przez 100 kroków nie spadła o 1 %); restarty (3 etapy po 1/3 kroków z lr x1,
x3, x6, każdy od najlepszego wyniku z nowym Adamem, jak w eksperymencie J128). „Rozgrzewka”: liczba
pierwszych kroków z liniowym wzrostem lr od 0. Bieżące lr jest w pasku postępu i w historii JSON wyniku.
Na TR_0001 kl. 870 (768 kanałów, zakres naturalny, 900 kroków) wszystkie harmonogramy dają pewność
96,3–96,9 % i IoU 99,5–100 %.

### Opcje atencji (widoki cząsteczek 2D i 3D, `opcje_atencji.py`)

Atencja małego ViT nie ma tokenu CLS: w bloku jest macierz [3 głowy, 600, 600]; A[i, j] = ile patch i
patrzy na patch j (ciągnie z niego), wiersz sumuje się do 1.

- Mapa: średnia otrzymana; z wybranego patcha (wiersz); do wybranego patcha (kolumna); skupienie
  (1 − entropia / log 600); zasięg (średnia odległość w patchach); atencja na siebie.
- Głowa: średnia, maksimum albo jedna z 3. Bloki: jeden albo rollout 1..blok (iloczyn (A + I) / 2).
- „Pomiń siebie”: zeruje przekątną i normuje wiersze. Skala: liniowa, logarytmiczna, rangi.
- Strzałki źródło → cel: k celów wybranego patcha, k źródeł wybranego patcha, najsilniejszy cel
  każdego patcha (600 strzałek), k najsilniejszych par w obrazie. Kolor i grubość = waga.

Na TR_0140 kl. 1655 (k1000, średnia po głowach, bez przekątnej) najsilniejszym celem prawie wszystkich
patchy jest jeden patch: token 321 (wiersz 10, kolumna 21). W bloku 3 tak jest dla 300 z 600 patchy,
w bloku 5 dla 584, w bloku 6 dla 591; średnia waga do niego w bloku 6 to 0,129 (równy rozkład dałby
0,0017). Wygląda to na ujście atencji (attention sink); nie sprawdzono, czy leży na narzędziu.

### Zakładka „Cząsteczki 3D: globalny układ (GPU, OpenGL)” (`czasteczki3d.py`)

Wszystkie 600 tokenów w jednym układzie 3D: x = kolumna w pikselach wejścia ViT, y = wiersz od dołu,
z = wysokość nad obrazem. Kanał c ma stały kierunek (spirala Fibonacciego na półsferze nad obrazem albo
na pełnej sferze) i stały kolor; długość = promień sfery zera + skala · wartość. Rysuje pyqtgraph.opengl
(bufory VBO na karcie): obrót kamery nie przelicza danych, pełne przeliczenie 600 x 768 wektorów trwa
około 40 ms (RTX 3070).

- Tryby: przed, po, różnica, nałożone, przejście PRZED → PO z suwakiem i animacją.
- Styl: wektory, końce (chmura punktów) albo oba; maks. długość przycina nieliczne bardzo duże wartości.
- Podłoże: obraz z wtopioną mapą atencji; „podnieś o atencję” unosi środki cząsteczek o wartość mapy.
- Łuki atencji źródło → cel nad obrazem (wyższy łuk = dalszy cel, jaśniejszy koniec = cel).
- Klik bez ruchu wybiera patch; „Tylko wybrany patch” ustawia kamerę na jednej cząsteczce.
- „Zapisz PNG” zapisuje widok do `outputs/latent_editor_sd_20261006/czasteczki3d_*.png`.

## Zakładka „Film: sekwencja klatek, cząsteczki 3D i makrowektory” (`zakladka_filmu.py`, `film.py`)

1. Film: lista 170 filmów Cataract-LMM 3_Object_Tracking z podziałem (train 118, val 25, poza manifestem)
   i liczbą klatek z narzędziem; „Otwórz film” wypakowuje MP4 raz z `videos_part00N.zip` do
   `Dataset/Cataract-LMM/3_Object_Tracking/videos_mp4/`; „Plik MP4…” otwiera dowolny film.
2. Zakres: od min:s do min:s, liczba klatek (domyślnie 100, równo rozłożone), opcjonalnie tylko klatki
   z etykietą narzędzia; źródło MP4 albo JPG z zipa adnotacji. Klatka MP4 i = frame_number i + 1
   (sprawdzone na TR_0001), 29,97 kl./s, 720x480.
3. „Konwertuj zakres na sekwencję”: dla każdej klatki latent ViT (600 x 768, float16), detekcja (ramka,
   pewność, mapa pewności na patchach), etykiety (ramka, obrys cystotome_seg, czubek, rogówka, źrenica),
   IoU i zmiana latentu względem poprzedniej klatki. Klatek nie zapisuje się na dysk.
   TR_0001, 0:20–0:30, 100 klatek: 6 s; średnia pewność 71,0 %, średnie IoU 72,5 %.
4. Klatka: obrysy z etykiety, ramki GT i ViT, czubek, rogówka, źrenica, siatka z patchami maski.
   „Rysuj obwód”: lewy klik = punkt, prawy = zamknij; cofnij, wyczyść, kopiuj na kolejne klatki.
   Maska patchy (który patch należy do narzędzia): z obrysu etykiety, z obwodu ręcznego albo z obu,
   próg pokrycia patcha w %.
5. Przebieg: pewność, IoU i zmiana latentu w czasie; klik przechodzi do klatki.
6. Odtwarzanie: ⏮ ◀ ▶ ▶| ⏭, suwak, kl./s, pętla; porównanie w 3D z poprzednią albo pierwszą klatką.
   Jedna klatka (obraz, 3D z 768 x 600 wektorami, makrowektory) to około 85 ms.
7. „Zapisz sekwencję”: `outputs/latent_editor_sd_20261006/film_<film>_<od>-<do>_<N>kl.npz` (latent,
   ramki, pewności, IoU, maski) i `.json` (ręczne obwody, ustawienia).

Widok 3D nad obrazem dostał obrysy narzędzia na płaszczyźnie obrazu i wybór „tylko patche w obrysie”
/ „tylko poza obrysem”.

### Przestrzeń makrowektorów (`przestrzen_makro.py`)

Kanał c ma stały kierunek d_c (makrowektor). Mikrowektor patcha i w klatce t to (z_tic − μ_c) d_c, więc
położenie cząstki to suma 768 mikrowektorów: p_ti = s (z_ti − μ) D, D: [768, 3]. Kolejne klatki
przesuwają cząstki tam, dokąd pchają je nowe siły kanałów; przelot jest płynny (ms do ustawienia),
ślad pokazuje tor z ostatnich klatek. Kierunki: 3 składowe PCA latentu sekwencji (domyślnie, największy
rozrzut; na TR_0001 0:20–0:30 wyjaśniają 58 % wariancji), sfera Fibonacciego jak w widoku 3D albo losowe.
μ = średni latent sekwencji (można wyłączyć), s stałe w sekwencji (99. percentyl promienia = 100).
Siły z: wszystkich kanałów, tylko J128 albo bez J128. Kolor: położenie patcha, narzędzie / tło (maska),
pewność ViT. Łańcuch mikrowektorów wybranego patcha: 768 wektorów od najsilniejszego, koniec do początku,
kończy się w cząstce.

Uwaga o OpenGL: przy rozłączonej sesji Windows (pulpit zdalny rozłączony, ekran zablokowany) sterownik
nie daje kontekstu OpenGL i widoki 3D są czarne; `WidokGL` pomija wtedy klatki zamiast zgłaszać błąd.

## Pliki

- `silnik.py`: ładowanie modeli, DDIM z postępem, latent z ViT, zakresy kanałów.
- `app.py`: GUI.
- `detekcja.py`: ViT z checkpointu, metryki (IoU, AP, mAP), ogon od latentu, optymalizacja.
- `zakladka_detekcji.py`: zakładka detekcji i auto-szukania.
- `czasteczki.py`, `czasteczki3d.py`: widoki cząsteczek 2D (matplotlib) i 3D (pyqtgraph.opengl).
- `opcje_atencji.py`: wspólny panel opcji atencji (mapy, głowy, rollout, strzałki).
- `film.py`: filmy, zakres czasu na klatki, etykiety i obrysy, maska patchy.
- `zakladka_filmu.py`: zakładka filmu; `przestrzen_makro.py`: przestrzeń makrowektorów.
- `uklad.py`: układ zawijający kontrolki do kolejnych linii.
- Wyniki (obrazy PNG, latenty NPZ) zapisują się do `outputs/latent_editor_sd_20261006/`.

## Ścieżki

- checkpoint: `assets/StableDiffusionModel/best_checkpoint.pt` (sha256 4c450bdd...),
- baza SD: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/ema_020000`,
- VAE: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/sd-vae-ft-mse`,
- ViT: `outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt`.
