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

`start.py` od razu pokazuje okienko ładowania z krokami (torch, silnik SD, ViT z timm i torchvision, OpenGL,
okno). Pierwszy start po włączeniu komputera trwa do minuty (import bibliotek z zimnego dysku), kolejne około
7 s. Błąd startu pokazuje się w oknie i trafia do `start.log`; bez konsoli (exe) wyjście programu idzie do
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
   (z `wytnij_narzedzie.py`), etykiety i obrysy są brane z annotation.csv dla właściwych numerów klatek.
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
od 1 px do ustawionej grubości (`widok_gl.LinieGrube`; OpenGL ma jedną grubość na element linii).
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
- „przy starcie” (obok „Wczytaj historię”): po uruchomieniu programu ostatnio oglądany bieg i klatka
  wczytują się w tle, razem z ostatnio policzonymi równaniami (`cache_uczenia/ostatnie.json`).
  Zmienna środowiskowa `EDYTOR_BEZ_AUTOSTARTU=1` wyłącza to w testach.
- „Równanie krok po kroku” (grupa pod odtwarzaniem, karta wykresów pod obrazem, `rownanie_krokowe.py`): dla
  zaznaczonych patchy (albo wybranego) w każdym kroku t prognoza kroku x(t) -> x(t+1) w pełnych 768 wymiarach
  (Langevin albo GLE) i niedopasowanie: błąd kroku / prawdziwy krok, cos kierunku, błąd po h krokach,
  parametry (bezwładność, sprężyna) w czasie. Dopasowanie kroczące z W ostatnich kroków przed t (uczciwe,
  sprawdzone: zmiana przyszłości nie zmienia prognozy z t) albo stałe okno panelu (porównanie). Stałe okno
  odtwarza parametry panelu równania. W makro strzałki: biała prawdziwy krok, pomarańczowa przewidziany,
  czerwona niedopasowanie; ślad w kolorze wygładzonego błędu. Patch 255, GLE, kroki 700-1000: mediana błędu
  kroku 0,54 (kroczące) i 0,55 (stałe okno), cos 0,86.
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
wszystko dotyczy tylko stałych klatek próbnych. Plan dalszych etapów: `PLAN_ANIMACJA_UCZENIA.md`.

## Pliki

- `silnik.py`: ładowanie modeli, DDIM z postępem, latent z ViT, zakresy kanałów.
- `start.py`: start z okienkiem ładowania; `Edytor latentu.exe` i `uruchamiacz/`: uruchamiacz exe.
- `app.py`: okno główne i logika zakładki „Edytor SD”; `widzety_edytora.py`: jej widżety (obraz z siatką,
  lista suwaków) i wątki (model SD, generacja, latent ViT).
- `detekcja.py`: ViT z checkpointu, metryki (IoU, AP, mAP), ogon od latentu, optymalizacja.
- `zakladka_detekcji.py`: zakładka detekcji i auto-szukania.
- `czasteczki.py`, `czasteczki3d.py`: widoki cząsteczek 2D (matplotlib) i 3D (pyqtgraph.opengl).
- `opcje_atencji.py`: wspólny panel opcji atencji (mapy, głowy, rollout, strzałki).
- `film.py`: filmy, zakres czasu na klatki, etykiety i obrysy, maska patchy.
- `zakladka_filmu.py`: zakładka filmu; `przestrzen_makro.py`: przestrzeń makrowektorów.
- `uklad.py`: układ zawijający kontrolki, `dodaj()` (etykieta i pole zawijają się razem) i `PanelOpcji`
  (pasek przycisków widoku plus zakładki z grupami opcji, chowane przyciskiem „Opcje”).
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
- `wytnij_narzedzie.py`: sekwencja klatek z samym narzędziem na czarnym tle (`narzedzie_czarne_tlo/`).
- `widok_gl.py`: widoki OpenGL (`WidokGL` odporny na błędy kontekstu, `Widok3D` z kliknięciem w obraz).
- Wyniki (obrazy PNG, latenty NPZ) zapisują się do `outputs/latent_editor_sd_20261006/`.

## Ścieżki

- checkpoint: `assets/StableDiffusionModel/best_checkpoint.pt` (sha256 4c450bdd...),
- baza SD: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/ema_020000`,
- VAE: `F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/sd-vae-ft-mse`,
- ViT: `outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt`.
