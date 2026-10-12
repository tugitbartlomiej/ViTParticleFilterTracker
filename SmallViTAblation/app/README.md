# Edytor latentu SD — GUI

Interaktywna aplikacja (PyQt6) do edycji pełnego latentu 600x768 i generowania
obrazów z wytrenowanego Stable Diffusion (`vit_sd_pelny_latent_filmy_eden_20261006`,
checkpoint krok 8500, tryb cross-attention).

## Uruchomienie

Najprościej: `Edytor latentu.exe` w tym katalogu (dwuklik, bez konsoli). To mały uruchamiacz (48 kB): nie pakuje
Pythona ani torcha, tylko startuje zainstalowany `F:\Python\3.13\pythonw.exe` (torch 2.10 z CUDA 12.6) ze skryptem
`start.py`. Inny Python: pełna ścieżka do `pythonw.exe` w pliku `python.txt` obok exe. Źródło i ikona:
`uruchamiacz/` (przebudowa: `csc.exe /target:winexe /win32icon:ikona.ico /r:System.Windows.Forms.dll
"/out:..\Edytor latentu.exe" Uruchamiacz.cs`, kompilator z .NET Framework 4 w Windows).

Z konsoli: `uruchom.bat` albo `python start.py` z tego katalogu (`SmallViTAblation/app/`).

`start.py` od razu pokazuje okienko ładowania z krokami (torch, silnik SD, okno). Aplikacja startuje pusta:
nie wczytuje żadnej sekwencji, historii ani modelu SD (ciepły start około 3 s do „Gotowe”, potem zero pracy w tle).
Zakładki poza „Edytor SD” budują się przy pierwszym wejściu na kartę (`ZAKLADKI_LENIWE` w `app.py`), więc timm
i torchvision ładują się dopiero przy pierwszej detekcji. Model SD ładuje się przy pierwszym GENERUJ. Błąd startu pokazuje się w oknie i trafia do `start.log`; bez konsoli (exe) wyjście programu idzie do
`wyjscie.log`. `python app.py` też działa, ale bez okienka ładowania.

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
z = wysokość nad obrazem. Kanał c ma stały kierunek (spirala Fibonacciego na pełnej sferze, domyślnie,
ze środkami cząsteczek na płaszczyźnie obrazu, albo na półsferze nad obrazem). Obraz jest półprzezroczysty
(„krycie”, domyślnie 0,5): wektory pod nim widać przez zdjęcie, a te nad nim zostają ostre i stały kolor; długość = promień sfery zera + skala · wartość. Rysuje pyqtgraph.opengl
(bufory VBO na karcie): obrót kamery nie przelicza danych, pełne przeliczenie 600 x 768 wektorów trwa
około 40 ms (RTX 3070).

- Tryby: przed, po, różnica, nałożone, przejście PRZED → PO z suwakiem i animacją.
- Styl: wektory, końce (chmura punktów) albo oba; maks. długość przycina nieliczne bardzo duże wartości.
- Podłoże: obraz z wtopioną mapą atencji; „podnieś o atencję” unosi środki cząsteczek o wartość mapy.
- Łuki atencji źródło → cel nad obrazem (wyższy łuk = dalszy cel, jaśniejszy koniec = cel).
- Klik bez ruchu wybiera patch; „Tylko wybrany patch” ustawia kamerę na jednej cząsteczce.
- „Zapisz PNG” zapisuje widok do `outputs/latent_editor_sd_20261006/czasteczki3d_*.png`.

### Auto-szukanie w wybranych patchach

„Patche” w optymalizacji: latent wolno zmieniać we wszystkich 600 patchach, tylko w ramce etykiety, w ramce
z obwódką 1 patcha, tylko w patchu środka etykiety albo w patchu klikniętym w widoku cząsteczek; w pozostałych
latent zostaje naturalny (`Detektor.optymalizuj(..., tokeny=...)`, None = wszystkie, wynik identyczny jak dotąd).
TR_0001 kl. 870, J128, 1500 kroków: 600 / 40 / 72 / 1 patch dają tę samą pewność 84,1 % i IoU 99,9–100 %.
Latent to wyjście MLP ostatniego bloku, za nim nie ma już atencji, więc patch wpływa tylko na własne wyjście
detekcji: do trafienia wystarcza zmiana w patchu środka, reszta zmienia jedynie pewności tła.

## Zakładka „Film: sekwencja klatek, cząsteczki 3D i makrowektory” (`zakladka_filmu.py`, `film.py`)

1. Film: lista 170 filmów Cataract-LMM 3_Object_Tracking z podziałem (train 118, val 25, poza manifestem)
   i liczbą klatek z narzędziem; „Otwórz film” wypakowuje MP4 raz z `videos_part00N.zip` do
   `Dataset/Cataract-LMM/3_Object_Tracking/videos_mp4/`; „Plik MP4…” otwiera dowolny film.
   „Folder zdjęć…” (źródło „folder zdjęć”): sekwencja z plików PNG / JPG / BMP / TIF w kolejności nazw, np.
   `app/narzedzie_czarne_tlo/<sekwencja>`; zakres ustawia się na cały folder. Gdy folder ma `MANIFEST.json`
   (z `narzedzia/wytnij_narzedzie.py`), etykiety i obrysy są brane z annotation.csv dla właściwych numerów klatek.
   TR_0001 0:20–0:33, 400 klatek samego narzędzia na czarnym tle: konwersja 16 s, średnia pewność ViT 45,1 %,
   średnie IoU 65,0 % (na pełnych klatkach tego filmu 0:20–0:30: 71,0 % i 72,5 %).
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
   „Śledź czubek”: w każdej klatce automatycznie wybrany patch z czubkiem igły (cząstka w 3D, jej mikrowektory
   w przestrzeni makrowektorów, ramka na klatce): z etykiety cystotome_tip (w klatce bez czubka najpewniejszy
   patch ViT) albo najpewniejszy patch ViT. Ręczny klik w patch wyłącza śledzenie. TR_0001 0:20–0:33, co 10.
   klatka: patch z etykiety zmienia się 10 razy na 40 klatek; najpewniejszy patch ViT leży średnio 5,4 patcha
   od czubka (to środek wykrytego narzędzia, nie sam czubek).
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
pewność ViT. Mikrowektory wybranego patcha: domyślnie wszystkie 768 ze wspólnego środka układu plus biała
wypadkowa (ich suma) kończąca się w cząstce; mnożnik „powiększ mikrowektory” wydłuża tylko rysunek mikrowektorów
(TR_0001, klatka 21, patch 315: najdłuższy 7,6, mediana 0,04, wypadkowa 90). Do wyboru też łańcuch (koniec do
początku, od najsilniejszego), oba albo brak; łańcuch i wypadkowa to to samo dodawanie wektorów.

### Wycinanie z obrazu przed analizą ViT (zakładka filmu)

Grupa „Wycinanie z obrazu”: obszary z etykiety klatki (obrys narzędzia, ramka GT, rogówka, źrenica), każdy osobno
albo kilka razem (suma), tryb „zostaw tylko zaznaczone (reszta czarna)” albo „zamaluj zaznaczone na czarno”,
margines w px (dylatacja). Zmieniona klatka idzie do ViT (latent, detekcja, atencja, widoki); zmiana wymaga ponownej
konwersji; ustawienie zapisuje się w JSON sekwencji (`wycinanie`). `film.maska_wycinania`, `film.wytnij`.
TR_0001 0:20–0:30, 30 klatek (JPG), pewność ViT / IoU z etykietą: bez wycinania 72,7 / 73,3 %; tylko obrys 42,9 /
64,7 %; obrys + 8 px 52,0 / 61,6 %; tylko ramka 56,9 / 53,4 %; tylko rogówka 45,3 / 17,5 %; obrys + ramka + rogówka
56,2 / 44,5 %; tylko źrenica 54,1 / 35,3 %; obrys zaczerniony 63,0 / 62,1 %; rogówka zaczerniona 69,8 / 56,4 %.

### Włączanie cząsteczek i widoków

- „Pokaż cząsteczki” (widok 3D, zakładka Cząsteczki) i „Pokaż cząstki” (makrowektory, Wygląd): bez wektorów
  i punktów cząsteczek; w 3D zostaje obraz, mapa atencji i łuki, w makro mikrowektory wybranego patcha.
- Zakładka filmu, „Widoki: widok 3D / makrowektory”: wyłączony widok znika i nie jest przeliczany.
  TR_0001, 400 klatek: wszystko 73 ms na klatkę, 3D bez cząsteczek 22 ms, bez widoku 3D 16 ms, bez obu 12 ms.
- Wykres przebiegu sekwencji pod klatką schowany (niepotrzebny); klatka zajmuje lewą kolumnę.
- Makrowektory, „Kierunki kanałów”: także półsfera Fibonacciego (wszystkie kierunki z > 0).

### Nawigacja i lustrzane odbicie (oba widoki 3D, zakładka opcji „Nawigacja”)

- „Obrót 360°”: kamera krąży dookoła sceny bez końca (prędkość w °/s, kierunek w lewo / w prawo); mysz działa dalej.
- Kroki: obrót o 15° w lewo, w prawo, w górę, w dół; „Bliżej” / „Dalej” (odległość x 0,8 / x 1,25).
- Ujęcia: przód (jak domyślny widok nad obrazem), tył, lewo, prawo, góra, dół (odległość i środek bez zmian).
- „Lustrzane odbicie” w osi X, Y, Z: odbicie całej sceny względem środka widoku w macierzy widoku (`WidokGL.lustro`);
  jedno odbicie zmienia układ prawoskrętny w lewoskrętny. Klik w obraz trafia dalej w ten sam punkt (sprawdzone
  rzutowaniem punktu na ekran i z powrotem przy lustrze X i Y).

### Te same kierunki i kolory mikrowektorów w obu widokach 3D

„Kierunki kanałów: jak widok 3D nad obrazem” (domyślne w zakładce filmu): przestrzeń makrowektorów bierze
z widoku 3D dokładnie te kanały, bazę PCA, kierunki (sfera albo półsfera) i kolory, którymi rysowane są
mikrowektory cząsteczek (`WidokCzasteczek3D.baza_wektorow`); zmiany w widoku 3D przeliczają makro od razu,
a pola „Siły z kanałów” i „PCA wektorów” makro są wtedy wyłączone. Mikrowektor kanału (składowej) ma w obu
widokach ten sam kierunek i kolor, różni się długością (|cos| = 1,000000 dla 768 kanałów, PCA 8, półsfery, J128).
Kamera makro w tej samej orientacji (elewacja 38°, azymut −90°: x w prawo, y w głąb, z w górę, układ
prawoskrętny), przycisk „Kamera jak 3D” kopiuje bieżący kąt widoku 3D.

### Wygładzanie EMA torów (przestrzeń makrowektorów)

„Wygładzanie EMA α” (Wygląd): wygładzone(t) = α · bieżące(t) + (1 − α) · wygładzone(t − 1), od pierwszej klatki;
1 = bez wygładzania (domyślnie). Wygładzane są siły mikrowektorów (EMA jest liniowa), więc mikrowektory wybranej
cząstki i wypadkowa nadal kończą się dokładnie w wygładzonej cząstce (błąd do 3e-5). TR_0001 0:20–0:30: średni
przelot między klatkami 11,6 (α = 1) / 5,4 (0,5) / 2,2 (0,2) / 0,6 (0,05). Opóźnia ruch o około 1/α klatek.

### Atencja w przestrzeni makrowektorów

Zakładka opcji „Atencja” (ten sam panel co w widoku nad obrazem: mapa, głowa, blok, rollout, strzałki):
„Atencja na cząstkach” barwi cząstki mapą atencji bieżącej klatki (inferno) i powiększa je według wartości
(„Powiększ wg mapy”); strzałki źródło -> cel są liniami między cząstkami w 3D (nadążają za przelotem).
Atencja liczona z obrazu każdej klatki tylko przy włączonej opcji; około 106 ms na klatkę razem z resztą widoków.
„Grubość strzałek” (panel atencji, wszystkie widoki): strzałki i łuki grubsze dla silniejszej atencji, w 5 klasach
od 1 px do ustawionej grubości (shader w `czasteczki3d_gpu.py` i `makro_gpu.py`, 5 klas grubości linii).
„Mapa z patchy”: mapy „z wybranego / do wybranego patcha” liczone ze zbioru (średnia): wybrany patch, wszystkie
widoczne cząstki, patche narzędzia z maski klatki albo lista numerów (np. `315, 316, 340-345`).
„Heatmapa całej przestrzeni”: mapa rozlana gradientem po przestrzeni makrowektorów, siatka 48 x 48 x 48, wartość
= gładka średnia mapy pobliskich cząstek (rozmycie Gaussa z normalizacją, „Rozmycie” w % boku), inferno, świecąca
mgła (rysowanie addytywne, nie zasłania cząstek). Zasięg: przy cząstkach (krycie z gęstości) albo cała przestrzeń
(dalej od cząstek wartość najbliższych). Około 50 ms na klatkę przy wyłączonym widoku 3D.

### PCA wektorów (oba widoki 3D)

„PCA wektorów”: zamiast 768 kanałów cząsteczka ma k = 3, 4, 8, 16, 32 albo 128 wektorów, czyli k głównych
składowych latentu. Wektor j to (z − μ) · w_j, gdzie w_j jest j-tą składową, a μ średnim tokenem (sfera zera
w widoku nad obrazem = średni token). Składowe liczy się z kanałów wybranych w „Kanały” / „Siły z kanałów”
(np. tylko J128). Kierunki i kolory: k wektorów równo na sferze, jak dla 768. Baza: w zakładce filmu z całej
sekwencji (ta sama we wszystkich klatkach), w zakładce detekcji z latentu przed i po. Znak składowej jest
ustalony (największy ładunek dodatni), więc ponowne liczenie daje tę samą bazę.

TR_0001, 0:20–0:30, 100 klatek: udział wariancji 3 / 4 / 8 / 16 / 32 / 128 składowych = 57,9 / 63,7 / 75,8 /
86,2 / 92,5 / 98,3 %; błąd odtworzenia latentu klatki 21 (względem odchylenia od średniej) 0,64 / 0,60 / 0,49 /
0,37 / 0,28 / 0,13. W przestrzeni makrowektorów suma k mikrowektorów nadal trafia dokładnie w cząstkę.

„Cząsteczek” / „Cząstek” (oba widoki 3D): co n-ty patch w wierszach i kolumnach siatki, od środka pierwszego
bloku n x n: wszystkie 600, co 2. = 150, co 3. = 70, co 4. = 35, co 5. = 24, co 7. = 12, co 10. = 6. Łączy się
z wyborem patchy w obrysie; PCA, skala i atencja liczą się dalej ze wszystkich 600.

### Odjęcie położenia patcha (przestrzeń makrowektorów, `polozenie.py`)

„Położenie patcha” usuwa z latentu to, co wynika z samego miejsca patcha, zanim policzą się PCA i kierunki.
Latent położenia M_i = średni latent tokenu i po 3000 klatkach walidacji (treść się uśrednia, miejsce zostaje);
dla K1000 położenie to 39,2 % wariancji latentu, a 13 kierunków obejmuje 95 % tego wkładu (pierwsze pięć:
36, 23, 16, 8, 3 %). Liczone raz na checkpoint (15 s na RTX 3070), zapis
`outputs/latent_editor_sd_20261006/POLOZENIE_<checkpoint>.npz`. Sposoby:

- średnia patcha z sekwencji: z − M_i(sekwencja) + g; usuwa też nieruchome tło, zostaje to, co się zmienia,
- średnia patcha z 3000 klatek: z − M_i + g (M_i zawiera też typowy układ sceny, np. źrenicę w środku),
- rzut: z − ((z − g) U) Uᵀ, U = 13 kierunków położenia; reszta latentu bez zmian.

TR_0001, 0:20–0:30, kierunki PCA (3): R² liniowego odczytu kolumny / wiersza patcha z położenia cząstki
(średniego w czasie) 0,76 / 0,72 bez odjęcia; 0,01 / 0,00 po średniej z sekwencji; 0,46 / 0,26 po średniej
z 3000 klatek; 0,39 / 0,23 po rzucie. Rozdział narzędzie / tło (odległość środków grup przez rozrzut) spada
z 1,23 do 0,42 / 0,68 / 0,35: część różnicy narzędzia od tła to jego typowe miejsce w kadrze.

Uwaga o OpenGL: przy rozłączonej sesji Windows (pulpit zdalny rozłączony, ekran zablokowany) sterownik
nie daje kontekstu OpenGL i widoki 3D są czarne; `WidokGL` pomija wtedy klatki zamiast zgłaszać błąd.

### Orientacja: zdjęcie w przestrzeni makrowektorów i kolor wektorów na obrazie

- Przestrzeń makro, „Wygląd”, „Zdjęcie w przestrzeni”: siatka 20 x 30 patchy rozpięta na położeniach
  cząstek (węzeł = cząstka patcha, sąsiedzi z obrazu połączeni), pokolorowana zdjęciem bieżącej klatki,
  kolorami cząstek albo zdjęciem z narzędziem na czerwono. Powierzchnia, linie albo oba; krycie i gęstość
  interpolacji. „Podpisy”: rogi obrazu, środek obrazu, środek narzędzia; brzeg obrazu ma kolor (górny
  niebieski, lewy żółty). Gładka siatka = podobny latent sąsiadów, rozdarcie = skok latentu.
- Widok 3D nad obrazem, „Obraz”, „Koloruj obraz mieszanką kolorów wektorów”: kolor patcha = średnia
  kolorów jego wektorów (kanałów albo składowych PCA) ważona siłą (|wartość|, wartość², wartość⁴ albo
  tylko dodatnie), z automatycznym nasyceniem, opcjonalnie jasność według łącznej siły i gładkie przejścia.
- Przestrzeń makro, „Kolor cząstek: mieszanka kolorów mikrowektorów”: ta sama mieszanka na ruchomych
  cząstkach, liczona w każdej klatce przy rysowaniu (waga, nasycenie, jasność według siły w „Wygląd”).
- Cztery sposoby orientacji w przestrzeni 768 wymiarów (przegląd literatury 2026-10-07):
  1. Zaznaczanie w obie strony: prawy przycisk + przeciągnij na obrazie albo Shift + przeciągnij w przestrzeni
     makro; zaznaczone patche żółte na obrazie, ich cząstki jasne, reszta blada (Rave i in. 2021).
     Mapa podobieństwa: kolor „podobieństwo do wybranego patcha (cos)” w makro i na obrazie widoku 3D.
  2. Kolor „PCA: 3 składowe jako RGB” (jak DINOv2, FeatUp) w makro i na obrazie widoku 3D, z tej samej bazy
     (różnica kolorów 0), więc ten sam kolor = ta sama cząstka.
  3. „Ortonormalne kierunki”: rzut prostokątny przez QR macierzy kierunków (Lehmann i Theisel 2013) oraz
     „Grand Tour” (Li i in. 2020): rama rzutu płynie przez losowe podprzestrzenie; wyłączenie zostawia widok.
  4. „Atlas kanałów”: mapy 20 x 30 każdego kanału (albo składowej PCA) bieżącej klatki, wg zmienności, z paskiem
     koloru kanału; klik podświetla kierunek kanału w makro (`atlas_kanalow.py`).
- Zaznaczanie wielu patchy i śledzenie tylko ich (zakładka „Uczenie ViT”, grupa „Zaznaczone patche”):
  Ctrl + klik na obrazie albo w cząstkę w przestrzeni makro dodaje albo usuwa patch, Ctrl przy prostokącie
  (prawy przycisk na obrazie, Shift + przeciągnij w makro) dołącza zamiast zastępować; przyciski „+ wybrany
  patch”, „patche narzędzia”, „wyczyść”. „Osobne kolory”: każdy zaznaczony patch w swoim kolorze na obrazie,
  w makro i na wykresach. „Śledź tylko zaznaczone”: makro rysuje same zaznaczone cząstki z ich śladami
  (zdjęcie zostaje blade w tle), prognoza Langevina przechodzi na „zaznaczonych patchy (każdy osobno)”
  (najwyżej N modeli) albo „średnią zaznaczonych”. Karta „Zaznaczone patche” pod obrazem: przez całe uczenie
  prędkość w latencie i w makro, odległość i cos do stanu końcowego każdego patcha oraz rozrzut zaznaczonych
  (średnia odległość od ich środka). W zakładce filmu działa to samo Ctrl + klik i Ctrl przy prostokącie.
  Widok 3D („Cząsteczki”) i przestrzeń makro („Wygląd”) mają własne pole „Tylko zaznaczone patche”;
  „śledź tylko zaznaczone” ustawia oba naraz. W panelu równania przycisk „zaznaczone” liczy Langevina, GLE
  i Koopmana osobno dla każdego zaznaczonego patcha (do 20, kolory jak zaznaczenie).
- Policzona sekwencja na dysku: pierwsze wczytanie klatki (około 40-50 s) zapisuje w `app/cache_uczenia`
  latent float16, miary ruchu, detekcję z wag, oscylator i czubek (`*_wyniki.npz`); następne wczytanie bierze
  je z dysku (kilkanaście sekund, głównie odczyt latentu 0,9 GB). Oceny i rozkłady równań cząstek trafiają do
  `*_rownania.pkl` (na patch i parametry). Zmiana latentu albo wag biegu unieważnia zapis.
- „Wczytaj ostatni” (obok „Wczytaj historię”): wybiera ostatnio oglądany bieg i klatkę
  (`cache_uczenia/ostatnie.json`, opis w dymku) i wczytuje je dopiero po kliknięciu. Dawny przełącznik
  „przy starcie” usunięty: program niczego nie wczytuje sam.
- „Równanie krok po kroku” (grupa pod odtwarzaniem, karta wykresów pod obrazem, `rownanie_krokowe.py`): dla
  zaznaczonych patchy (albo wybranego) w każdym kroku t prognoza kroku x(t) -> x(t+1) w pełnych 768 wymiarach
  (Langevin albo GLE) i niedopasowanie: błąd kroku / prawdziwy krok, cos kierunku, błąd po h krokach,
  parametry (bezwładność, sprężyna) w czasie. Dopasowanie kroczące z W ostatnich kroków przed t (uczciwe,
  sprawdzone: zmiana przyszłości nie zmienia prognozy z t) albo stałe okno panelu (porównanie). Stałe okno
  odtwarza parametry panelu równania. W makro strzałki: biała prawdziwy krok, pomarańczowa przewidziany,
  czerwona niedopasowanie; ślad w kolorze wygładzonego błędu. Patch 255, GLE, kroki 700-1000: mediana błędu
  kroku 0,54 (kroczące) i 0,55 (stałe okno), cos 0,86.
- Zakładka „Korekta latentu: cel i dynamika” (`zakladka_korekty.py`, silnik `korekta_latentu.py`, modele
  `dynamika_korekty.py`, gałęzie uczenia `galezie_uczenia.py`; specyfikacja i testy w
  `experiments/stanowisko_korekcji_latentu_gui/`): korekta z etykietą (oracle) latentu po GELU ostatniego MLP przy
  zamrożonym ViT, w pięciu trybach osiągalności (swobodne, zakresy z checkpointu, dolny GELU, wejście MLP: gałąź albo
  wspólny residual), z celami strata / lokalizacja z karą za fałszywe szczyty / najmniejsza znaleziona korekta do progu
  jakości. Zapis prawdziwej trajektorii (δ i gradient w rzeczywistych krokach), losowe kontrole o tej samej normie,
  rozkład korekty na część przechodzącą przez fc2 i niewidoczną dla fc2 (jądro W2 wymiaru 576), import historycznych
  par przed/po J128 z ostrzeżeniem o niezgodnym checkpoincie, ewaluator AP z jawnym protokołem. Porównanie z ruchem
  uczenia obrazu (101 checkpointów, osobna oś czasu) w 3D w stałej bazie; modele dynamiki (persystencja, stała
  prędkość, Langevin, GLE, Koopman, GLE z wejściem optymalizatora) dopasowane tylko na części treningowej;
  kontrolowane gałęzie uczenia od stanu AdamW kroku k (waga obrazu, podmiana obrazu, wzrost zaburzeń; poziom szumu
  z identycznych gałęzi, tryb deterministyczny); obserwator Kalmana przy pomiarze części tokenów. Pomiary w
  `outputs/stanowisko_korekcji_latentu_gui/<czas>_<opis>/`, nic nie liczy się przy otwarciu zakładki.
- Zakładka „Pchnięcie obrazów: SD, DINO i ruch cząstek w kroku” (`zakladka_pchniecia.py`, rachunek `pchniecie.py`,
  stan `stan_uczenia.py`, paczki `paczki_uczenia.py`, kandydaci `kandydaci.py`): dokładne pchnięcie jednego kroku
  AdamW, jakie wywołałby pojedynczy obraz w paczce, na cząstkach sondy gęstej historii. Paczki odtwarzane z ziarna
  (zgodne z nagraniem co do jednej), stan AdamW z powtórki uczenia (98 s raz; wagi przypinane co 10 kroków do zapisu),
  sprawdzian: przewidziany krok sondy = rzeczywisty z gęstej historii w 98 %, suma pchnięć 32 obrazów = pchnięcie
  paczki w 99 %. Kandydaci z łańcucha „sztuczny latent -> SD -> prawdziwe obrazy”: paczka kroku, baza, klatki
  odszukane dla SD (MSE latentu), obrazy SD pętli 4.3 (pseudo-etykieta), klatki DINO dla obrazów SD, klatki przez
  konwerter latent -> DINO. Miary: zmiana straty sondy, zmiana siły tożsamości, kosinus do stanu końcowego (tylko
  wyjaśnienie). Galeria wg miary, mapa na sondzie, wykres tożsamość wobec straty, rozkład w źródłach, zapis JSON
  do `experiments/maly_vit_bezwladnosc_a_ped_adamw/wyniki`.
- „Źródła siły” (przycisk obok „Licz przyczyny ruchu”, okno `okno_zrodel_sily.py`, rachunek `przeplyw_sily.py`):
  skąd pochodzi wektor c (neuron ukrytej warstwy MLP ostatniego bloku) patcha i. Przy wagach atencji i skalach
  LayerNorm z tego przebiegu wszystko między liśćmi (łodyga, położenie sin-cos, wyjścia MLP bloków 0-4, biasy)
  a wartością przed GELU jest liniowe, więc rozkład jest ścisły (sprawdzone: ręczny przebieg = model co do 3e-6,
  suma wkładów = wartość przed GELU, stałe zgodne z przebiegiem z wyzerowanymi liśćmi). Atencja tylko przenosi
  sygnał między patchami, MLP tworzy nowy. Obraz: Integrated Gradients przez łodygę (jedyne przybliżenie, suma
  pikseli = cel z dokładnością 3e-4). Karty: „Przepływ” (płynna animacja sygnału na obrazie przez 14 etapów,
  strzałki przeniesień głów z kropkami płynącymi od źródła do celu, wodospad wkładów), „Warstwy × głowy”
  (mapy 20 x 30: sygnał na wejściu bloku, źródła każdej głowy, nowy sygnał MLP), „Graf przepływu” (rzeka
  sygnału z głowami i MLP jako węzłami), „Wektor patcha” (768 wartości, klik wybiera wektor). Śledzi krok
  i patch zakładki (liczy od nowa przy zmianie checkpointu albo patcha, około 0,2-1,5 s).
- Wygląd ścieżek prognozy Langevina (strona „Langevin”, grupa „Wygląd ścieżek prognozy”): każdy rodzaj ścieżki
  to osobna warstwa z własnym kolorem i grubością (ślad, ślad prognoz z wyprzedzeniem, bieżąca prognoza, stare
  prognozy, dalszy tor, linia błędu); schematy kolorów (kontrastowe, klasyczne, kolor cząstki, automatycznie),
  styl prognozy (ciągła z kropką co krok, ciągła, przerywana), kolor wg horyzontu, grubości, zanikanie starych
  prognoz, „powiększ ruch” (przesunięcia × k wokół bieżącego położenia, sprawdzone: dokładnie × k), linia błędu
  (przewidziane -> rzeczywiste położenie), legenda w rogu widoku, kamera śledząca cząstkę, „zbliż na cząstkę”.
- Strony panelu „Opcje” (widok 3D, makro) przewijają się, zamiast przycinać wiersze. Główne zakładki są
  w przewijanych polach przez pośrednika bez heightForWidth (inaczej strona rosła ponad okno).
- Ustawienia uczenia są stronami panelu „Opcje” przestrzeni makro: „Uczenie” (porównanie w 3D, środek makro,
  śledzenie patcha, strzałki przyczyn), „Langevin”, „Zaznaczone”, „Krok po kroku”; pod odtwarzaniem został
  tylko opis kroku. „Środek makro”: zero latentu (domyślne, surowy latent, środek = wektor zerowy), średnia
  ze wszystkich kroków i tokenów, stan końcowy (na końcu wszystkie cząstki w 0) albo krok 0 (w kroku 0 w 0).
- Panele zakładki „Uczenie ViT” (obraz klatki, wykresy uczenia, widok 3D, makro, równanie; obraz i wykresy osobno) mają pasek z „Odepnij”
  (osobne okno, np. na drugi monitor) i „✕” (chowa panel); pola „Widoki” w górnej grupie pokazują je z powrotem.
- W zakładce „Uczenie ViT” przyciski „Atlas kanałów” i „Krajobraz i tunele” przełączają główny panel (zamiast
  widoku 3D nad obrazem); ponowny klik wraca do widoku 3D. W zakładce filmu otwierają osobne okna.
- „Krajobraz i tunele” (przycisk w przestrzeni makro, `krajobraz.py`): okno z trzema obrazami tego, po czym
  biegają cząstki, śledzące bieżący krok: teren energii U = −ln gęstość na dwóch głównych osiach chmury
  (studnie, jak w TFT `plotly_studnie.py`), wosk z tunelami (izopowierzchnia odwiedzin wokseli, kolor = krok
  pierwszego przejścia; opcja „lity wosk”: blok wydrążany przez cząstki) i otoczka (izopowierzchnia z wybranym
  odsetkiem masy). Gęstość z całej historii, do bieżącego kroku (drążenie) albo z okna N kroków. Pod oknem test
  wchodzenia w ślady poprzedników: klatka 0, krok 600, siatka 64³: 81,5 % odwiedzin w wokselach przebytych
  wcześniej przez inną cząstkę wobec 81,6 % w kontroli z przesuniętym czasem, czyli bez wspólnych kanałów jak
  w szkle; tunele to kształt chmury, a nie wspólne ścieżki.
- Klik (bez przeciągania) w cząstkę w przestrzeni makro wybiera jej patch: biała ramka na obrazie, w widoku
  3D nad obrazem i duża biała cząstka w makro (zakładki filmu i uczenia; wyłącza automatyczne śledzenie).
- Zakładka uczenia: „kroków/s” do 10 000; powyżej szybkości odświeżania ekranu odtwarzanie przeskakuje
  kroki według zegara.

### Wycinanie na żywo i Langevin po klatkach (zakładka filmu)

- **Wycinanie na żywo** (pole „na żywo” w grupie wycinania, domyślnie włączone): zmiana obszarów (obrys narzędzia,
  ramka GT, rogówka, źrenica), trybu albo marginesu od razu przelicza ViT na klatkach wczytanej sekwencji, które są
  w pamięci w wersji oryginalnej (bez ponownego czytania filmu, około 50 ms na klatkę). Klatka, wybrany patch
  i ręczne obwody zostają. Wyłączenie wszystkich obszarów wraca dokładnie do latentu bez wycinania.
- **Langevin** (dolny pasek): liniowy model ruchu każdej cząstki po klatkach filmu,
  v(t+1) = beta v(t) − kappa (z(t) − m(t)), z(t+1) = z(t) + v(t+1), m = średnia ostatnich klatek; beta i kappa
  każdej cząstki z pierwszych 70 % klatek, ocena prognozy na pozostałych (błąd / błąd „bez ruchu”) w opisie klatki.
  W przestrzeni makrowektorów dla wybranej cząstki: pomarańczowa przerywana = tor z modelu (tylko ze stanów do
  bieżącej klatki), turkusowa = rzeczywisty tor. Na zegarze wideo pęd jest bliski zera (beta około 0), a sprężyna
  silna (kappa 0,4-0,7), inaczej niż na zegarze uczenia (beta 0,75-0,87).

## Zakładka „Uczenie ViT: ruch cząsteczek po krokach” (`zakladka_uczenia.py`, `analiza_uczenia.py`, `przyczyny_ruchu.py`)

Oś czasu to krok uczenia, nie klatka filmu. Źródło: bieg z gęstą historią latentu
(`experiments/*/wyniki/*/dane/LATENT_GESTY.npy`, obecnie `maly_vit_ruch_czastek_gesta_historia/pelny`:
1001 kroków co 1, 12 stałych klatek próbnych, wagi co 10 kroków). Pierwsze wczytanie klatki trwa około
30 s i zapisuje kopię float16 w `app/cache_uczenia/` (0,9 GB na klatkę, poza gitem).

- **Animacja:** suwak i odtwarzanie po krokach, widok 3D nad obrazem i przestrzeń makrowektorów
  z tymi samymi kierunkami i kolorami. Baza PCA i środek z puli wszystkich kroków (bez PCA na krok,
  która dawałaby przeskoki osi). „Środek makro”: średnia, stan końcowy (cząstki zbiegają do środka)
  albo krok 0. Detekcja i atencja z najbliższego checkpointu wag (podpisane w tytule klatki).
- **Ruch cząsteczek:** strata i mAP, prędkość tokenów w latencie i cząstek w makro (narzędzie, tło,
  wybrany patch), odległość od stanu końcowego, rozdzielenie narzędzie / tło, efektywny rząd,
  mapa CKA krok x krok (klik = krok).
- **Przyczyny ruchu** (przycisk „Licz przyczyny ruchu”, około 30 s), w przedziałach między checkpointami:
  udział 13 grup wag (łodyga, atencja i MLP każdego bloku) w ruchu cząstek, zmiana wag grup,
  cos(−∂L/∂z, Δz) i siła gradientu straty względem latentu (ramka z obrysu narzędzia), ruch razem
  z tokenami, na które token patrzy (z kontrolą: ruch ze średnią wszystkich), zmiana atencji.
  W makro strzałki dla wybranej cząstki: biała = ruch w przedziale, kolorowe = wkłady grup wag,
  żółta przerywana = kierunek −gradientu.
- **Śledź patch:** domyślnie czubek narzędzia z etykiety klatki (cystotome_tip, czerwone kółko na obrazie),
  dalej najpewniejszy patch ViT i najszybszy patch w kroku.
- **Prognoza Langevina i ślad toru** (osobna grupa pod paskiem odtwarzania): oscylator
  v(t+1) = beta v(t) − kappa (z(t) − m(t)) dla wybranego patcha albo dla średniej cząstek narzędzia
  lub tła (jedna duża kropka, osobno dopasowane beta i kappa). Prognoza startuje co „nowa co” kroków
  i stoi do następnego startu; zostaje kilka starszych prognoz. Pomarańczowa przerywana = prognoza,
  pomarańczowa kropka = gdzie prognoza stawiała cząstkę teraz, turkusowy = rzeczywisty ślad,
  pomarańczowy ciągły = ślad prognoz z wyprzedzeniem, blady jasnoniebieski = rzeczywisty dalszy tor.
  Długość śladu (albo od kroku 0), EMA z α, sposób zanikania i krycie najstarszego odcinka.
  W opisie kroku: błąd prognozy wobec „bez ruchu”.
- **Animuj** w widoku 3D: bez porównania PRZED / PO odtwarza uczenie (jak „Ucz”).
- **Panel równania** (czwarty widok, `panel_rownania.py`, `rownanie_czastki.py`): dla do 8 cząstek z listy
  trzy modele z samej historii cząstki (bez siły paczki): Langevin (oscylator), GLE z jądrem pamięci L kroków
  (jedno jądro dla 768 kanałów; L = 1 daje dokładnie oscylator) i Koopman (Hankel-DMD na r składowych toru
  z d opóźnieniami, reszta: oscylator). Tabela błąd / „bez ruchu” dla h = 1, 5, 20, 50 na krokach testowych,
  równanie z liczbami, udziały części (bezwładność, pamięć, sprężyna, szum), wykresy: błąd vs horyzont, jądro
  pamięci (regresja i Volterra), rozkład kroku na części wokół bieżącego kroku, portret fazowy z polem
  oscylatora, autokorelacja i rozkład szumu, VACF danych i modelu, wartości własne Koopmana.
  „Dla: cząstek z listy (panel równania)” rysuje w makro prognozy wszystkich cząstek z listy w ich kolorach,
  modelem wybranym w panelu. Klatka 0, patch 255: GLE L = 10 daje 0,32 / 0,76 / 0,81 / 0,79 (h = 1 / 5 / 20 / 50)
  wobec oscylatora 0,35 / 0,81 / 0,93 / 0,79; Koopman gorszy na dłuższych horyzontach; szum to około 37 % ruchu
  (jak pchnięcie paczki, 31-35 %) z kurtozą 6-7 (rzadkie silne pchnięcia).

Zastrzeżenia: łodyga zawiera biegnące statystyki BatchNorm (zmieniają się też bez gradientu);
gradient liczony na jednej klatce próbnej, a uczenie szło po paczkach 32 obrazów z AdamW;
wszystko dotyczy tylko stałych klatek próbnych. Plan dalszych etapów: `docs/PLAN_ANIMACJA_UCZENIA.md`.

## Pliki

Struktura katalogu (porządek z 2026-10-12): moduły aplikacji leżą płasko w `app/` pod dotychczasowymi nazwami, bo
skrypty w `experiments/` dopisują `app/` do `sys.path` i importują je wprost (`import ruch_zbioru`, `import app`).

- `start.py`, `uruchom.bat`, `Edytor latentu.exe`, `uruchamiacz/`: start aplikacji (exe szuka `start.py` obok siebie).
- `app.py`: samo okno: pasek zakładek, leniwe budowanie zakładek (`ZAKLADKI_LENIWE`: krótki tytuł karty, pełny
  opis w dymku karty), menu „Ustawienia”, zamykanie.
- `edytor_sd.py`: zakładka „Edytor SD” (`EdytorSD`, klasa bazowa okna: inne zakładki czytają `okno.baza`,
  `okno.wektor`, `okno.efektywny_latent()`, a ścieżki kontrolek w `ustawienia.json` się nie zmieniły).
- `zakladki_glowne.py`: główne zakładki w przewijanych polach i podmiana zastępnika na zbudowaną zakładkę.
- `zakladka_*.py`: zakładki; `zakladka_ruchu.py` ma części w klasach bazowych `zakladka_ruchu_koopman.py`,
  `zakladka_ruchu_fizyka.py` i `zakladka_ruchu_atencja.py` (strony opcji Koopman, Fizyka ruchu, Atencja z obsługą).
- `testy/`: testy pytest (`conftest.py` w `app/` dopisuje katalog aplikacji do ścieżki i ustawia konteksty OpenGL).
- `narzedzia/`: skrypty pomocnicze uruchamiane ręcznie (`sprawdz_dymki.py`, `wytnij_klatki.py`, `wytnij_narzedzie.py`).
- `docs/`: plany (`PLAN_ANIMACJA_UCZENIA.md`); `zasoby/`: luźne pliki wejściowe (klatka TR_0140 kl. 1655).
- `zdjeciatestowe/`, `narzedzie_czarne_tlo/`: dane wejściowe zakładek detekcji i filmu.

- `silnik.py`: ładowanie modeli, DDIM z postępem, latent z ViT, zakresy kanałów.
- `start.py`: start z okienkiem ładowania; `Edytor latentu.exe` i `uruchamiacz/`: uruchamiacz exe.
- `app.py`: okno główne; `edytor_sd.py`: zakładka „Edytor SD”; `widzety_edytora.py`: jej widżety (obraz z siatką,
  lista suwaków) i wątki (model SD, generacja, latent ViT).
- `detekcja.py`: ViT z checkpointu, metryki (IoU, AP, mAP), ogon od latentu, optymalizacja.
- `zakladka_detekcji.py`: zakładka detekcji i auto-szukania.
- `czasteczki.py`, `czasteczki3d.py`: widoki cząsteczek 2D (matplotlib) i 3D nad obrazem; geometria i animacja 3D
  w `czasteczki3d_gpu.py` (shadery OpenGL, sekcja „Obliczenia na GPU”).
- `opcje_atencji.py`: wspólny panel opcji atencji (mapy, głowy, rollout, strzałki).
- `film.py`: filmy, zakres czasu na klatki, etykiety i obrysy, maska patchy.
- `zakladka_filmu.py`: zakładka filmu; `przestrzen_makro.py`: przestrzeń makrowektorów (ruchome warstwy w `makro_gpu.py`).
- `uklad.py`: układ zawijający kontrolki, `dodaj()` (etykieta i pole zawijają się razem), `przycisk()`,
  `pole_wyboru()`, `staly_napis()` i `PanelOpcji` (pasek przycisków widoku plus zakładki z grupami opcji, chowane
  przyciskiem „Opcje”; `zakoncz()` dociska wiersze każdej strony do góry).
- `ustawienia.py`: plik konfiguracyjny `ustawienia.json` (158 list, pól i opcji ze wszystkich zakładek i widoków;
  bez danych i bez przycisków przełączanych), wczytywany przy starcie i zapisywany przy zamknięciu; menu
  „Ustawienia”: zapisz, zapisz jako, wczytaj, przywróć domyślne, zapis przy zamknięciu.
  `EDYTOR_BEZ_USTAWIEN=1` uruchamia bez pliku (testy).
- `uklad.Odpinanie`: przycisk „Odepnij” w widokach 3D (cząsteczki nad obrazem, przestrzeń makrowektorów,
  też w zakładce detekcji) przenosi żywy widok do osobnego okna; dalej odtwarza sekwencję razem z zakładką.
  „Przypnij” albo zamknięcie okna oddaje go na miejsce. Wymaga wspólnych kontekstów OpenGL
  (`AA_ShareOpenGLContexts` w `start.py` i `app.main`). Dawne „Osobne okno” to teraz „Kopia w oknie”.
- `wygoda.py`: kółko myszy zmienia pole, listę albo suwak tylko po kliknięciu (inaczej przewija panel), liczby
  zatwierdzane Enterem albo wyjściem z pola, przyspieszanie przytrzymanej strzałki.
- `styl.py`: hierarchia przycisków (rola „glowny”: główna akcja zakładki, „narzedzie”: małe przyciski widoku).
- `wspolne.py`: siatka patchy (16 px, 20 x 30, wejście 480 x 320), kolory i kierunki 768 kanałów, katalog wyników.
- `polozenie.py`: latent samego położenia patcha (średnia z 3000 klatek walidacji) i jego kierunki.
- `narzedzia/wytnij_narzedzie.py`: sekwencja klatek z samym narzędziem na czarnym tle (`narzedzie_czarne_tlo/`).
- `widok_gl.py`: widoki OpenGL (`WidokGL` odporny na błędy kontekstu, `Widok3D` z kliknięciem w obraz).
- Wyniki (obrazy PNG, latenty NPZ) zapisują się do `outputs/latent_editor_sd_20261006/`.

## Ruch zbioru: cząsteczki i optymalizacja (2026-10-09)

Nowa główna zakładka działa bez serwera na porcie 8774 i bez przeglądarki.
Natywny widok OpenGL rysuje pełny zbiór, w tym 900000 cząsteczek znanego pomiaru.
Shader interpoluje dwa sąsiednie zapisane stany; nie przelicza ani nie interpoluje
metryk detekcji. Kamera, wspólna baza PCA i skala pozostają wspólne dla wszystkich obrazów i kroków.

1. **Gotowy wynik 1500 obrazów** wczytuje zweryfikowane trajektorie K1000 od 0 do 500 co 5 kroków.
   **Wybierz zapis ruchu / Wczytaj ruch** otwiera również folder nowego pomiaru.
2. W **Checkpoint i dane** wybierz checkpoint SmallViT (`.pt` / `.pth`) i kliknij
   **Sprawdź i wczytaj ViT**. Sprawdzane są wagi i zgodność architektury; pokazywane są krok i skrót SHA256.
   Wybierz manifest z etykietami lub COCO i, gdy ścieżki są względne, folder zdjęć.
3. W **Optymalizacja** ustaw liczbę obrazów, budżet kroków, zapis co N kroków,
   learning rate, Adam/SGD, ograniczenie GELU lub swobodną korektę, regularyzację,
   docelową pewność i IoU, ziarno i CUDA/CPU. **Oblicz nowy ruch** liczy w tle;
   **Zatrzymaj obliczenia** zachowuje częściowy wynik. Każdy obraz ma osobną korektę
   600×768 aktywacji, a wagi detektora pozostają zamrożone. To pomiar z etykietą
   (oracle), nie uczenie wag ani dowód jakości na nowych obrazach.
4. Po obliczeniu kliknij **Wczytaj ruch**. W **Animacja** można zmieniać tempo
   podczas odtwarzania, wyłączyć płynne przejścia, włączyć obrót, zmienić rozmiar
   i widoczność punktów oraz oglądać cały zbiór lub jeden obraz. Suwak i przyciski
   **Poprzedni / Następny zapis** wybierają rzeczywiste stany. Pauza zachowuje
   pozycję między nimi. Kliknięcie cząsteczki pokazuje numer obrazu i patcha.

Ścieżki i ustawienia tej zakładki są objęte menu **Ustawienia** oraz automatycznym
zapisem ustawień aplikacji. Nie są zapisywane zdjęcia ani pełne trajektorie 768D:
nowy pomiar zachowuje binarne rzuty 3D, normy ruchu w pełnych 768D, rzeczywiste
metryki, wspólną bazę PCA, konfigurację i pochodzenie checkpointu w
`outputs/ruch_czasteczek_gui/<nowy katalog>/`. Rozmiar tego zapisu jest podany
przed obliczeniem. **Zapisz podgląd PNG** to osobny, dobrowolny eksport widoku.
Przerwany pomiar zachowuje ukończone stany do oglądania, ale nie zapisuje stanu
optymalizatora potrzebnego do wznowienia obliczeń.

Zmiana checkpointu w formularzu nie zmienia pochodzenia gotowych trajektorii.
Przy różnicy SHA256 widok zgłasza niezgodność. Dla innego checkpointu użyj
**Oblicz nowy ruch**. Checkpointy innych architektur nie są konwertowane po cichu.

### Pliki tej zakładki

- `zakladka_ruchu.py`: formularz (`_build` woła po jednej metodzie na stronę opcji), zadania w tle, sterowanie
  odtwarzaniem i metryki; strony Koopman, Fizyka ruchu i Atencja z obsługą w `zakladka_ruchu_koopman.py`,
  `zakladka_ruchu_fizyka.py`, `zakladka_ruchu_atencja.py`.
- `ruch_gpu.py`: compute shadery GLSL 4.30; magazyn stanów osi czasu na karcie (cała oś w budżecie VRAM albo
  pierścień doczytywany w tle), interpolacja, kolory, granice osi, mapa 20 × 30, mediana, wybór punktu, ślady.
  Zastąpił `ruch_gl.py`. Oś czasu wczytuje się dopiero po kliknięciu.
- `ruch_zbioru.py`: czytnik zapisów, weryfikacja danych i silnik nowych pomiarów.
- `ruch_naturalny.py`: rzeczywiste forwardy kolejnych zapisanych modeli i import logów.

### Uczenie naturalne: rzeczywiste checkpointy do K1000

W panelu **Uczenie naturalne** kliknij **Historia prowadząca do K1000**.
Ustawia to odnaleziony lokalny bieg:
`experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/`.
Zawiera 101 checkpointów `k0000.pt`–`k1000.pt`, co 10 aktualizacji, z treningu
na Edenie z 1 października 2026, ziarno 47. Wszystkie SHA256 i zapisane liczniki
zostały zweryfikowane z manifestem transferu. Ostatni checkpoint jest identyczny
z `assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt`.

1. W **Checkpoint i dane** wybierz te same 1500 obrazów i etykiety.
   Domyślny `MANIFEST_1500.jsonl` ma tę samą kolejność co pierwotna baza treningowa.
2. W **Uczenie naturalne** wybierz folder checkpointów, zakres rzeczywistych kroków,
   odstęp między checkpointami, liczbę obrazów i wielkość paczki pomiarowej.
   CUDA/CPU i ziarno wspólnej bazy PCA wybiera się w **Optymalizacja**.
   **Sprawdź historię** weryfikuje zapisane kroki i pliki, bez obliczania ruchu.
3. **Oblicz ruch naturalny** przepuszcza każde zdjęcie przez każdy wybrany model,
   w trybie eval i fp32. Nie uczy wag, nie odtwarza AdamW i nie podmienia aktywacji.
   Wszystkie 600 tokenów każdego obrazu pochodzą z tego samego miejsca sieci co
   w trybie sztucznym: post-GELU ostatniego MLP, po 768 współrzędnych na token.
4. **Wczytaj wynik naturalny** wybiera najnowszy kompletny naturalny pomiar
   dla wybranej liczby obrazów, a gdy jeszcze go nie ma, ukończone stany częściowego pomiaru.
   Stan częściowy jest jawnie oznaczony. Ponowne wczytanie dodaje nowo zapisane checkpointy.
   Inne gotowe zapisy można otworzyć przez **Wczytaj ruch**.
   Suwak pokazuje rzeczywiste numery kroków treningu, także przy zakresie zaczynającym
   się po kroku 0. Ruch liczony jest względem pierwszego wybranego checkpointu.
5. **Wczytaj logi** przyjmuje plik JSON/JSONL z jawnie zapisanymi krokami albo folder
   zawierający `LOG.json` i `OCENY.json`. Oddzielny wykres pokazuje stratę treningową
   lub historyczne mAP50 z walidacji. Nie są to nowe pomiary IoU na 1500 obrazach.
   Logi źródłowe i ich sumy są dołączane do pochodzenia nowego pomiaru.

Domyślny pomiar 1500×600 cząsteczek w 101 stanach zapisuje około 1,45 GB rzutu 3D.
Do dokładnych norm ruchu w 768D potrzebny jest tymczasowy początkowy latent około
2,76 GB, usuwany po zakończeniu lub przerwaniu. Nie zapisuje się zdjęć ani pełnych
trajektorii 768D. Przed uruchomieniem sprawdzane są zasoby. Gotowe dane z tej rodziny
również trafiają do `outputs/ruch_czasteczek_gui/`, z oznaczeniem osi `training_checkpoint`.

Historyczny gęsty zapis co jeden krok dotyczył tylko 12 sond, a nie całych 1500 obrazów.
Dla całego zbioru dostępne są rzeczywiste wagi co 10 kroków. Shader wygładza przejścia
między pomiarami; nie udaje, że ma niezapisane checkpointy. PCA dopasowywane jest raz
na pierwszym wybranym stanie i nie zmienia orientacji w kolejnych krokach. W tej
historycznej współrzędnej obserwowany ruch obejmuje zmiany wag oraz zapisanych
buforów BatchNorm, nie jedynie aktualizacje parametrów.

## Historyczne ścieżki edytora SD

- checkpoint: `assets/StableDiffusionModel/best_checkpoint.pt` (sha256 4c450bdd...),
- baza SD: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/ema_020000`,
- VAE: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/sd-vae-ft-mse`,
- ViT: `outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt`.

## Obliczenia na GPU (2026-10-09)

Animacje cząsteczek liczą compute shadery OpenGL 4.3 (GLSL 4.30, bufory SSBO) w oknie aplikacji, natywnie pod
Windows; bez przeglądarki i WebGL. Dane trafiają na kartę przy zmianie kroku albo opcji, a w każdej klatce CPU podaje
tylko parametry (krok, postęp przelotu, kamera).

| Widok | Moduł GPU | Co liczy karta |
|---|---|---|
| Przestrzeń makrowektorów | `makro_gpu.py` | przelot 600 cząstek, mikrowektory, łańcuch i wypadkowa wybranej cząstki, ślady z położeń wszystkich kroków, zdjęcie na siatce cząstek, brzeg, strzałki atencji |
| Cząsteczki 3D nad obrazem | `czasteczki3d_gpu.py` | wartości PRZED/PO, skala (centyl), mikrowektory i przejście PRZED → PO, łuki atencji, mieszanki kolorów, tekstura obrazu |
| Ruch zbioru (900 000 cząsteczek) | `ruch_gpu.py` | położenia i przejścia między zapisami, kolory, granice osi, mapa ruchu, mediana, wybór punktu, ślady |

Na CPU zostało: dopasowanie baz (PCA, SVD), mapy atencji z ViT (`opcje_atencji.py`), napisy i statyczne elementy
sceny, wykresy matplotlib (aktualizowane przez blitting zamiast pełnego przerysowania).

Wymagania i pułapki:
- OpenGL 4.3 lub nowszy (RTX 3070: 4.6 compatibility). Bez niego błąd jest widoczny w opisie pod widokiem.
- Python 3.13 (`F:\Python\3.13`), bo pod 3.11 `OpenGL_accelerate` nie działa z numpy 2.
- `QOpenGLWidget.repaint()` nie rysuje synchronicznie; testy wymuszają klatkę przez `grabFramebuffer()`.
- Po rysowaniu każdy moduł przywraca stan GL (np. `GL_UNPACK_ALIGNMENT`), inaczej psują się napisy QPainter.

Testy (wszystkie porównują bufory odczytane z karty z referencją CPU/numpy):
`F:\Python\3.13\python.exe -m pytest -q testy` z katalogu `app` (`conftest.py` ustawia wspólne konteksty OpenGL jak `start.py`).
Testy OpenGL wymagają zwykłej platformy Qt (pod `QT_QPA_PLATFORM=offscreen` nie mają kontekstu i padają).
Dymki wszystkich kontrolek sprawdza `narzedzia/sprawdz_dymki.py --podsumowanie`.

### Pomiar naturalny na GPU, uzupełnianie i zapis 16-bitowy (2026-10-09)

- **Pomiar na karcie** (`ruch_naturalny.py`): obrazy dekodowane raz na cały pomiar i trzymane na karcie (dawniej
  każdy z ~101 checkpointów czytał, sprawdzał SHA256 i dekodował od nowa 1500 obrazów). Stan odniesienia (pierwszy
  checkpoint, 2,8 GB) leży w pamięci karty zamiast w pliku czytanym w każdej paczce. Rzut PCA, normy 768D i metryki
  detekcji liczy CUDA dla całej paczki naraz. Klatki bitowo równe dawnej ścieżce CPU. Około 500 obrazów/s na
  RTX 3070 (dawniej około 64), czyli pełny pomiar 101 × 1500 w około 5 minut. Domyślna paczka 32.
- **Uzupełnij brakujące kroki** (przycisk w „Uczenie naturalne”, `ruch_naturalny.uzupelnij`): dolicza na GPU tylko
  brakujące checkpointy przerwanego pomiaru w tym samym folderze i zapisuje je na dysk. Sprawdza SHA256 źródeł,
  wczytuje bazę PCA z pliku i liczy ponownie krok bazowy, który musi zgadzać się bitowo z zapisem. Wynik jest bitowo
  taki sam jak pomiar bez przerwy (test `test_resume_missing_steps_equals_uninterrupted`).
- **Zapis 16-bitowy** (`frame_format = 'q16'`, domyślny dla nowych pomiarów; pliki `step_XXXX.q16`): każda wartość
  jako uint16 w granicach danego kroku, x = lo + q/65535 · span, norma = q/65535 · norm_max; parametry kroku w
  manifeście (`files[krok].quant`). Połowa miejsca na dysku i w pamięci karty (101 kroków: 1,45 GB -> 0,73 GB),
  shader zamienia słowa na liczby (`unpackUnorm2x16`). Błąd najwyżej pół kroku kwantyzacji (rozpiętość osi / 131 070).
  Statystyki, granice i krzywe w manifeście liczy producent z float32 przed kwantyzacją. Dawne pomiary float32
  (`.bin`) czytają się bez zmian; przycisk „Zapisz kopię 16-bitową” (`ruch_zbioru.konwertuj_q16`) robi obok kopię
  `_q16`, oryginał zostaje. Bezstratna kompresja (zstd z tasowaniem bajtów) dawała tylko 24 % mniej miejsca i wolniejszy
  odczyt, dlatego wybrano format 16-bitowy.

### Układy widoku i wygląd w „Ruch zbioru” (2026-10-09)

Wszystko liczą shadery (`ruch_gpu.py`); CPU podaje tylko parametry.

- **Układ** (strona „Animacja”): PCA 3D (dotychczasowy); **siatka zdjęcia + cecha na osi Z** (X, Y = kolumna i wiersz
  patcha, nałożone dla wszystkich obrazów albo mozaika kafli obrazów); **przestrzeń fazowa** (X = cecha, Y = jej
  prędkość między zapisami, Z = norma zmiany 768D). **Cecha**: PC1, PC2, PC3 (rzut wszystkich 768 cech na oś PCA),
  norma zmiany 768D, odległość od środka PCA. **Bez położenia patcha** (domyślnie włączone): od cechy odejmowana jest
  średnia tego patcha po wszystkich obrazach w tym samym kroku, bo położenie na obrazie (kodowanie pozycji, typowy
  układ sceny) pokazują już osie X i Y. Granice osi z maksymalnych odchyleń od średnich, liczonych przy wysyłaniu stanu.
- **Wygląd**: kolory położenia patcha, narzędzia i tła wg ramki GT, pewności ViT obrazu w kroku (metrics.npy) i numeru
  obrazu; co n-ty patch; tylko cząstki narzędzia; zaznaczone patche (Ctrl + klik, wspólne dla wszystkich obrazów);
  zdjęcie w przestrzeni rozpięte na 600 cząsteczkach jednego obrazu (powierzchnia, siatka linii, brzeg; w siatce
  zdjęcia leży na terenie cechy). **Nawigacja** jak w „Uczenie ViT” (obrót 360°, kroki, ujęcia, lustro), „Odepnij”.
- Nie przeniesiono opcji wymagających pełnych latentów 768D każdej cząsteczki (kierunki i siły kanałów, PCA wektorów,
  mikrowektory, atencja, Grand Tour, atlas, krajobraz): zapis trzyma rzut PCA i normę zmiany (pełne latenty
  900 000 cząsteczek × 101 kroków ≈ 280 GB).
- **Układ Koopmana** (strona „Koopman”): jeden liniowy operator dla wszystkich cząsteczek, Hankel-DMD na stanie
  (PC1, PC2, PC3, norma 768D) standaryzowanym i zanurzonym z d opóźnieniami (1–4). Macierze Grama sumuje karta
  (kernele `kstat`, `kgram`), rozkład własny 4d × 4d na CPU, współrzędne cząsteczek to rzuty na lewe wektory własne
  (funkcje własne Koopmana), liczone w shaderze w każdej klatce. X, Y = niezmienniki (λ bliskie 1): dwa najbliższe
  (domyślnie), część rzeczywista i urojona zespolonego, albo λ ≈ +1 i λ ≈ −1. Z = mod najbardziej dynamiczny
  (największe |λ − 1|, |arg λ| albo |ln |λ||) jako część rzeczywista, urojona albo moduł. Widmo z okresami na stronie.
  Na pomiarze naturalnym (d = 3): trzy rzeczywiste niezmienniki λ ≈ 0,98, najbliższy zespolony |λ| = 0,64
  (okres ≈ 270 kroków), mod λ = −0,40 zmieniający znak co zapis. Dopasowanie około 0,8 s na RTX 3070.
- **Ręczny wybór osi Koopmana** (strona „Koopman” i tabela modów w prawej kolumnie, widoczna w układzie
  Koopmana): każda oś X, Y, Z to lista składowych Re ψ_j albo Im ψ_j wybranych modów. Tabela pokazuje Re λ, Im λ,
  |λ|, |λ − 1|, arg λ i okres, z polami wyboru X/Y/Z Re/Im w każdym wierszu. Sortowanie: bliskość do 1
  (niezmienniki najpierw), |λ| malejąco, |arg λ| rosnąco lub malejąco, Re λ, Im λ, |λ − 1| malejąco
  (najbardziej dynamiczne). Filtry: znak Re λ (> 0, < 0) i Im λ (> 0, < 0, = 0) zawężają listę. Szybkie przypisanie: zakres pozycji „od–do” z bieżącej listy, część Re, Im albo obie,
  „Ustaw” zastępuje oś, „Dodaj” dopisuje. Łączenie składowych osi: suma albo moduł √Σ². Osie ręczne zostają do
  „Automatycznie” albo zmiany d. Mody zdegenerowane (wariancja funkcji własnej poniżej 1e-8 śladu) są pomijane,
  bo dają NaN i wybuch skali; ich liczba jest w opisie. Do 126 zestawów wag w buforze KG, zakres osi liczy karta.
- **Osie pod klastry** (strona „Koopman”, `klastry_koopmana.py`): przycisk sam dobiera X, Y i opcjonalnie Z tak,
  żeby cząsteczki tworzyły klastry z jak najmniejszą liczbą outlierów. Karta zwraca próbkę stanów Hankela cząsteczek
  widoku (kernel `kprobka`), potem na karcie przez torch: każda para składowych Re ψ, Im ψ, |ψ| dostaje mieszaninę
  Gaussów z wariancją każdej osi w każdym klastrze, k = 1..k max wybrane przez BIC. Outlier to punkt dalej niż próg
  w σ swojego klastra albo punkt z klastra mniejszego niż 2 procent próbki. Wynik to sylwetka w σ klastra minus kara
  razy udział outlierów. Klastry są istotne, gdy wynik przewyższa o 0,15 gaussowski układ zerowy o tej samej
  kowariancji, podzielony na tyle samo klastrów. Tryb „cały ruch” ocenia tę samą próbkę cząsteczek w 8 zapisach i
  bieżącym osobno i uśrednia, więc nie myli dryfu całej chmury w czasie z klastrami. Model mieszaniny idzie do
  shadera: kolor „Klaster Koopmana”, szare punkty to outliery. Około 1-6 s na RTX 3070.
- **Moc koloru** (obok pola „kolor”): 0 = odcienie szarości, 1 = barwy trybu, do 3 = mocniej nasycone; liczy shader.
- **Trajektorie wybranych zdjęć** (strona „Uczenie naturalne”, na górze; `historie_checkpointow.py`,
  `zbiory_zdjec.py`): pole „historia” wybiera zapisaną historię uczenia (K1000 co 10 kroków albo bieg 35 000 klatek
  co epokę) i wypełnia folder wag, logi, checkpoint referencyjny i zakres; dla historii z epokami „od epoki / do
  epoki” ustawia kroki 1094 · e. Zdjęcia: z manifestu 1500 albo z puli 35 000 klatek uczących (wypakowywane z
  lokalnych zipów z kontrolą SHA-256), wpisane numerami („0-9, 25”), wylosowane (ziarno 2026) albo zaznaczone w
  oknie „Wybierz zdjęcia…” z filtrem nazwy i filmu i podglądem. „Policz trajektorie wybranych” zapisuje manifest
  podzbioru w `outputs/zbiory_zdjec/`, mierzy naturalnym forwardem każdy checkpoint zakresu (q16) i od razu wczytuje
  oś czasu: 600 cząsteczek na zdjęcie. Przykład: 5 zdjęć × 22 epoki w 9 s.
- **Checkpoint z historii w detekcji** (zakładka „Detekcja ViT i auto-szukanie latentu”): obok „Checkpoint ViT…”
  lista historii i kroków albo epok; wybrany model służy detekcji i auto-szukaniu latentu.
- **Fizyka ruchu** (układ „Fizyka ruchu”, strona „Fizyka ruchu”, panel boczny; kernele `ruchk` i `wlasc`): karta
  liczy dla każdej cząsteczki właściwości ruchu w oknie zapisów (cała oś albo ±W wokół bieżącego kroku), w rzucie
  PCA3 osi czasu: temperatura (energia drgań własnych na zapis po odjęciu pola kolektywnego, czyli średniego ruchu
  populacji), wykładnik MSD α = log(MSD(τ)/MSD(1))/log τ, VACF (cos kolejnych przesunięć), prostoliniowość
  (netto/droga), tempo powrotu κ (Ornstein-Uhlenbeck), ruch w rozgrzewce (od kroku 0 do zadanego kroku), zmiana
  768D na końcu okna oraz reżim: balistyczny (VACF i prostoliniowość > próg), dyfuzyjny (α ≥ próg), uwięziony.
  Układ 4 kładzie wybrane trzy właściwości na osiach (temperatura, ruch rozgrzewki i zmiana 768D w skali log).
  Kolory 8-10: temperatura (inferno), reżim, α (viridis). Panel boczny: histogram wybranej właściwości osobno
  dla obszarów oka z AUC względem tła, temperatura i cos kolejnych kroków w czasie dla obszarów oraz energia pola
  kolektywnego, mapa ruchu w rozgrzewce na siatce 20 × 30 z udziałem wariancji tłumaczonym przez położenie patcha.
  Zgodność z numpy: test `test_fizyka_ruchu_zgodna_z_numpy`.
- **Obszary oka** (`obszary_oka.py`, kolor 11, filtr „pokaż” na stronie „Fizyka ruchu”): klasy patchy z wielokątów
  segmentacji `annotation.csv` Cataract-LMM (narzędzie ≥ 25 % pokrycia, źrenica i rogówka ≥ 50 %, reszta tło),
  przeskalowanych do 480 × 320 jak obraz ViT. Liczone raz w tle po wczytaniu osi i zapisywane obok niej
  (`obszary_oka.npz`, 1500 zdjęć w ~40 s, potem 0,03 s). Na historii K1000: tło 83,3 %, rogówka 8,6 %, źrenica
  6,4 %, narzędzie 1,7 % patchy; mediana temperatury rośnie w tej kolejności (0,138 / 0,193 / 0,222 / 0,246),
  AUC temperatury narzędzie vs tło 0,88, źrenica vs tło 0,84.
- **Kolory ruchu chwilowego, częstotliwości i energii.** 12: przyspieszenie chwilowe |v₁ − v₀| wokół bieżącego
  zapisu; 13: prędkość chwilowa (różnica centralna); 14: prędkość względem średniej prędkości tej cząsteczki (log2,
  niebieski wolniej, czerwony zryw). Do właściwości fizyki doszły: okres dominujący ruchu i udział energii w paśmie
  okresów (z DFT przyrostów cząsteczki, najwyżej 128 przyrostów ze środka okna, bez średniego dryfu), energia
  kinetyczna ½⟨|Δx|²⟩ i potencjalna ½κ⟨|x − x̄|²⟩ (sprężyna OU). Kolor 15: wybrana energia (log); 16: udział energii
  w paśmie okresów ustawionym na stronie „Fizyka ruchu”; 17: do 3 zakresów dowolnej właściwości z własnymi kolorami
  (np. okres 2–3 / 3–6 / >6 zapisów, temperatura albo energia; przycisk „Zakresy z tercyli”). Widmo zakłada równe
  odstępy zapisów w oknie: dla historii 35 000 (co 10, potem co 100 kroków i końce epok) wybierz okno w jednej części.
  Na historii K1000: okres dominujący mediana 4,3 zapisu dla tła i 3,4 dla narzędzia i źrenicy (szybsze drgania).
- **Wygładzanie ruchu** (strona „Animacja”): średnia ruchoma wyśrodkowana z n zapisów (bez opóźnienia) albo EMA α
  (przyczynowa od pierwszego zapisu, jak w Uczenie ViT). Liczy shader `stan_g()` z zapisów na karcie, we wszystkich
  układach, śladach, średnich patchy i Koopmanie; napis nad widokiem oznacza STAN WYGŁADZONY. Statystyki, mapa ruchu
  i skale kolorów dalej z zapisanych stanów. Bez wygładzania stan jest bitowo równy zapisowi.
- **Heatmapy atencji** (strona „Atencja”, `atencja_zbioru.py`): mapy ViT z wag checkpointu bieżącego kroku (pomiary
  naturalne), te same definicje co panel atencji Uczenie ViT (zgodność do 1e-4, test `test_atencja_zbioru.py`).
  Jeden obraz: liczone przy każdym kroku (~0,13 s, pierwsze ~5 s z wczytaniem modelu). Cały zbiór: przycisk, 1500 map
  wsadowo na CUDA (~8 s z jednorazowym dekodowaniem zdjęć do pamięci karty). Nakładanie: kolor inferno i rozmiar
  cząsteczek, plama na zdjęciu w przestrzeni albo oba. Wartości w górnych 16 bitach bufora MASK (bit 0 = narzędzie).


### Układ szkła: ruch względem klatki i symulowane szkło koloidalne (2026-10-10)

Porównanie ruchu tokenów z dynamiką szklistą (Weeks i Weitz 2000–2002, Hunter i Weeks 2012) pokazało, że 89 %
przesunięcia względem populacji to wspólny przepływ sąsiedztw, a grzechotanie w klatce to 11 %. Żeby zobaczyć samo
grzechotanie, oś czasu trzeba przenieść do **układu względem klatki** (jak w szkłach 2D, Vivek i in. 2017):
`z_i^CR(t) = z_i(t) − średnia_{j∈N6(i)} [z_j(t) − z_j(300)]`, gdzie N6 to 6 najbliższych tokenów w latencie (PCA32)
w obrębie zdjęcia w kroku 300 (koniec rozgrzewki; od tego kroku lr jest stały).

- **Oś szkła** (strona „Checkpoint i dane”, pole obok „Gotowy wynik 1500 obrazów”): lista folderów
  `outputs/ruch_czasteczek_gui/*szklo*` z opisem z manifestu (`uklad.nazwa`); **Odśwież** czyta listę na nowo,
  **Wczytaj oś szkła** wstawia folder do „zapis ruchu” i wczytuje. Lista powstaje z dysku bez wczytywania czegokolwiek.
- `20261010T_natural_szklo_klatka_35k`: 60 000 cząsteczek biegu 35 000 w układzie klatki, kroki co 2 w 300–2188
  (gęsty zapis co 1 krok) i co 50 dalej, 1362 stany (654 MB, q16). Pewność i IoU w `stats`/`metrics.npy` są
  interpolowane liniowo między 22 checkpointami pomiaru (oznaczone `interpolated_metrics`), obszary oka skopiowane,
  baza ekranu PCA3 z kroków 300, 2188, 10 000. Norma (4. kolumna) = |z^CR(t) − z^CR(300)| w PCA32.
  Producent: `outputs/porownanie_szklo_20261010/eksport_osi_szkla.py`.
- `20261010T_szklo_BD_phi_*` i `20261010T_szklo_BD_stygnace_*`: symulowane szkło koloidalne (dynamika browna
  2048 polidyspersyjnych miękkich kul harmonicznych w pudle periodycznym 3D, `model_szkla.py`): stała temperatura
  i stygnięcie własne T ∝ (1 + t/t0)^−0,79, potem t^−0,30, czyli taki sam zanik energii drgań jak tokenów przy
  stałym lr. Bez zdjęć (`image_count` 0): obszary oka niedostępne, pozostałe kolorowania i fizyka ruchu działają.
- Te same odciski palca (`outputs/porownanie_szklo_20261010/odciski.py`: MSD i wykładniki, rozmiar klatki
  względem odstępu sąsiadów, antykorelacja kolejnych przesunięć, α₂ i ogony, Fs(q,τ), ucieczki z klatki, trwałość
  sąsiedztwa, starzenie) liczone są dla danych ViT w czterech stopniach układu szkła (`uklad_szkla_analiza.py`:
  populacja → klatka → jednakowe cząstki → czas termiczny → bez pędu) i dla symulacji; karta zgodności z literaturą
  w `UKLAD_SZKLA.json`, `MODEL_SZKLA.json` i README tego katalogu.
