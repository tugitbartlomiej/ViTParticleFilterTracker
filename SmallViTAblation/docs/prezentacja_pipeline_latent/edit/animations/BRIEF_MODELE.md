# Wspólny brief: film „modele w ruchu” (6 scen Manim)

## O czym jest film
Badacz pokazuje pipeline doboru zdjęć do uczenia małego detektora ViT (2,93 mln parametrów, 600 tokenów na
zdjęcie 480 × 320, siatka 30 × 20, latent 600 × 768, 6 bloków transformera, stem z 4 splotów). Detektor wykrywa
narzędzie chirurgiczne (cystotom) w nagraniach operacji zaćmy. Pierwszy film (napisy i liczby) już istnieje.
Ten drugi film ma pokazać **działanie modeli**: uczenie sieci, wyznaczanie kierunków, ruch w latencie.

## Najważniejsze życzenie użytkownika
„Nie zależy mi na animacji napisów, tylko na animacjach modeli.” Dlatego:
- ruch ma pokazywać mechanizm: przepływ danych przez sieć, gradient płynący wstecz, cząstki w przestrzeni,
  obracający się kierunek, kulkę na krajobrazie straty, trajektorię wag;
- tekst tylko jako: mały nagłówek „MODEL n / 6” i tytuł w lewym górnym rogu (pojawiają się zwykłym FadeIn, bez
  pisania litera po literze), krótkie etykiety obiektów (do 3 słów), osie i liczby w Consolas;
- zero akapitów, zero zdań wyjaśniających na ekranie, zero `Write()` na długich napisach;
- obiekty mają się ruszać płynnie i ciągle (ValueTracker, updatery, always_redraw), nie tylko pojawiać się.

## Styl (taki sam jak w pierwszym filmie)
- Tło `#0E1A24`, panel `#16293A`, linie pomocnicze `#24425A`.
- Tekst `#EAF2F5`, przygaszony `#7C93A3`, morski `#2BB3C0` (główny), pomarańczowy `#F08A2E` (akcent: narzędzie,
  gradient, cel), zielony `#5BC27A` (dobrze), czerwony `#E0565B` (źle).
- Krój `Segoe UI` dla etykiet, `Consolas` dla liczb, `MathTex` tylko gdy wzór jest istotą (LaTeX działa,
  MiKTeX w `F:/Instalki/Latex/miktex/bin/x64`).
- Polski z ogonkami. Bez długich myślników (—) w tekście na ekranie. Słowo „stem” zostaje po angielsku.
- Wygładzanie nieliniowe (`smooth`, `rate_functions.ease_out_cubic`, `ease_in_out_sine`), nigdy `linear` dla
  wejść i wyjść (liniowo wolno tylko ciągły ruch cząstek lub czasu).
- Jedna nowa rzecz naraz; nie wprowadzaj dwóch niezależnych obiektów jednocześnie.
- Scena zaczyna się od pustego tła i kończy wyciszeniem wszystkiego do tła (ostatnie ~0,6 s: FadeOut całości).
- Wzór stylu: `../slot_2/scena.py` (schemat sieci), `../slot_6/scena.py` (wykres odpowiedzi). Przeczytaj je.

## Pauzy (ważne dla odtwarzacza)
Film będzie pokazywany w odtwarzaczu, który zatrzymuje się w wybranych chwilach i czeka na klawisz. Po każdym
ważnym etapie zrób **nieruchome ujęcie co najmniej 1,2 s** (`self.wait(1.2)` lub dłużej), w którym wszystko stoi
(updatery też muszą stać: zatrzymaj ValueTrackery). W ruchu ciągłym (np. cząstki drgają cały czas) nie ma pauz,
więc zatrzymuj go na czas tych ujęć. Zaplanuj 3 do 5 takich ujęć na scenę.

## Dane (prawdziwe pomiary)
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\prezentacja_pipeline_latent\edit\dane_animacji.json`
Każda sekcja ma pole `opis`. Liczby pokazywane jako zmierzone muszą pochodzić z tego pliku albo z briefu sceny.
Schematyczne elementy (chmury punktów, kształt doliny) są dozwolone, ale nie podpisuj ich liczbami, których nie
zmierzono. Wczytaj JSON w `scena.py` przez `json.load` (ścieżka bezwzględna albo względna do pliku sceny).

## Technika
- Manim Community 0.20.1, Python 3.11: `py -3.11 -m manim`. Biblioteki: numpy jest. Nie instaluj niczego.
- Plik: `scena.py` w Twoim folderze slotu, klasa `ModelN` (N = numer sceny).
- Szkic: `PYTHONUTF8=1 py -3.11 -m manim -ql --disable_caching --media_dir media scena.py ModelN`
- Render końcowy (uruchom w folderze slotu):
  `PYTHONUTF8=1 py -3.11 -m manim -r 1920,1080 --fps 30 --disable_caching --media_dir media scena.py ModelN`
  potem skopiuj `media/videos/scena/1080p30/ModelN.mp4` do `render.mp4` w folderze slotu.
- ThreeDScene wolno, jeśli scena tego potrzebuje (wtedy etykiety `add_fixed_in_frame_mobjects`).
- Długi render uruchamiaj w tle albo z dużym limitem czasu; nie przerywaj go.

## Sprawdzenie (obowiązkowe, przed oddaniem)
1. `ffprobe` na `render.mp4`: 1920 × 1080, 30 kl./s, długość zgodna z planem.
2. Wyciągnij kadry co ~2 s: `ffmpeg -y -i render.mp4 -vf "fps=0.5,scale=640:-1,tile=4x4" verify_arkusz.png`
   (albo kilka arkuszy) i **obejrzyj je narzędziem Read**. Szukaj: nachodzących na siebie etykiet, ucięcia przy
   krawędzi, nieczytelnie małego tekstu (min. ~24 px wysokości przy 1080p), pustych lub przypadkowych kadrów,
   błyskania przy przejściach.
3. Popraw i wyrenderuj ponownie. Najwyżej 3 rundy poprawek.
4. Znajdź nieruchome ujęcia: `ffmpeg -i render.mp4 -vf "freezedetect=n=0.001:d=0.8" -map 0:v -f null - 2>&1 | grep freeze_`
   i dla każdego podaj czas początku (to będą punkty zatrzymania w odtwarzaczu).

## Granice
- Pisz wyłącznie w swoim folderze slotu. Nie ruszaj innych slotów, `final.mp4`, `edl.json`, `punkty.js`,
  `odtwarzacz.html`, ani niczego poza folderem `edit\animations\slot_mN`.
- Nie commituj niczego.

## Co oddać (krótko)
Ścieżka `render.mp4`, długość, lista etapów sceny z czasem początku każdego nieruchomego ujęcia
(`m:ss.cc` i 3-5 słów opisu), oraz co poprawiłeś po obejrzeniu kadrów.
