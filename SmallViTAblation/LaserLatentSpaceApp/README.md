# Lasery i koloidy — aplikacja (LaserLatentSpaceApp)

Osobna aplikacja PyQt6 wydzielona 2026-10-22 z zakładki „Lasery i koloidy” edytora latentu (`../app`). Oś czasu uczenia
jednego zdjęcia przez checkpointy małego ViT: sfera reflektorów, pułapka optyczna, studnie, drgania, atencja 6 bloków,
rzuty z 192 wymiarów, dolina pewności i IoU, mechanika reflektora (filmy 2, 12, 16).

## Uruchomienie

- `Lasery i koloidy.exe` (dwuklik, bez konsoli): mały uruchamiacz (48 kB, źródło w `uruchamiacz/`), startuje
  `W:\Python\3.11\pythonw.exe start.py`; inny Python: pełna ścieżka do `pythonw.exe` w `python.txt` obok exe.
  Przebudowa: `csc.exe /codepage:65001 /target:winexe /win32icon:ikona.ico /r:System.Windows.Forms.dll "/out:..\Lasery i koloidy.exe" Uruchamiacz.cs`
  (kompilator z .NET Framework 4 w Windows, `C:\Windows\Microsoft.NET\Framework64\v4.*\csc.exe`).
- Z konsoli: `uruchom.bat` albo `python start.py` (okienko ładowania), `python app.py` (bez niego).
- Wymagania: torch + CUDA, timm, PyQt6, pyqtgraph, PyOpenGL, opencv-python, scipy, numpy. scikit-image jest opcjonalne
  (obrysy 3D ogniska i doliny liczą się bez niego przez `pyqtgraph.isosurface`, `wspolne.izopowierzchnia`).
- Błąd startu: okno z komunikatem i `start.log`; wyjście programu bez konsoli: `wyjscie.log`.
- Testy: `set PYTHONPATH=. && python -m unittest testy.test_etykiety testy.test_rzuty_dane`.

## Układ okna (refaktor GUI 2026-10-22)

- **Pasek osi czasu** u góry: ⏮ ▶, pętla, punkty A–B, suwak, tempo, licznik kroku; „Okienka ▾” = menu okienek.
- **Okienka-doki** z widokami zajmują większość okna. Domyślnie: duża sfera reflektorów (R0–R12 w kartach obok niej),
  obok pułapka optyczna 11 (dolina 12, mechanika 13, przejście P w kartach), na dole niższy wiersz 1 | 2 | 3 | mapy 4–5 |
  wykresy 6–10. Każde okienko można przeciągnąć, połączyć w karty, odpiąć (❐), rozwinąć na ekran (⛶, dwuklik w pasek) albo
  na pełny ekran (F11 pod kursorem, Esc wraca).
- **Parametry** (F2): zwarta lista sekcji po lewej i jedna strona naraz (Dane i etykiety, Sfera, Pułapka, Podkroki,
  Rzuty, Ruch 1–3, Mapy, Dolina, Mechanika, Widok). Panel jest okienkiem-dokiem: można go odpiąć jako osobne okno
  (np. na drugi monitor) albo schować.
- **Podkroki sieci** (F3): osobne okienko z suwakiem przejścia sygnału przez 6 bloków, tempem, synchronizacją z osią
  czasu i przenikaniem uwaga ↔ MLP; można je odpiąć i postawić gdziekolwiek.
- **Prawy klik w okienko** (bez przeciągania; w widokach 3D prawe przeciąganie dalej zbliża): ustawienia tego okienka przy
  kursorze (`panel_ustawien.PopupUstawien`). To te same sekcje co w panelu (wypożyczone na czas otwarcia, w panelu stoi
  zastępnik), więc wszystko działa i zapisuje się tak samo. Przyciski: 📌 przypina popup jako osobne okno, ⇤ pokazuje
  sekcję w panelu, ⧉ odpina okienko, ⛶ maksymalizuje, 🖵 pełny ekran, 🗙 zamyka okienko. Sfera dostaje też wybór
  cząsteczki i reflektora, wykresy 6–10 sekcję „Dane”, okienka R — swoje podsekcje rzutów.
- **Menu „Okienka”**: okienka pogrupowane (ruch i mapy, sfera i pułapka, rzuty), „Układy okienek”: domyślny, sfera +
  pułapka obok siebie, sfera + rzuty R1–R7 obok siebie, sfera na całe okno; Ctrl+Shift+1..3 zapamiętuje bieżący układ,
  Ctrl+1..3 go przywraca (sloty w `ustawienia.json`).
- **Ustawienia**: stan kontrolek i układ okienek w `ustawienia.json` (zapis przy zamknięciu, menu „Ustawienia”). Ustawienia
  dawnej zakładki przeniesiono z `../app/ustawienia.json` (bez układu okienek, który ma nową postać).

## Etykiety: narzędzie i obszary oka (2026-10-22)

Strona „Dane, etykiety”, sekcja „Etykiety”. Które z 600 patchy są narzędziem (`lasery_dane.Historia.ustaw_etykiete`,
`etykiety_narzedzia.TRYBY`, `maski(..., tryb=...)`):

- **obrys: każda łata dotknięta (> 0 %)** — domyślnie: całe narzędzie, także patche w większości z tłem;
- **obrys: pokrycie ≥ próg** — zachowanie z 2026-10-21 (próg 0,5): przy cienkim cystotomie zostają 2–7 patchy;
- **obrys: środek łaty w obrysie**; **ramka GT: środek łaty w ramce** (dawna etykieta sprzed obrysu, dużo tła).

Przykłady (patche narzędzia: dotknięte / pokrycie ≥ 0,5 / środek / ramka): TR_0050 kl. 1629: 16 / 3 / 4 / 18;
TR_0077 kl. 1113: 25 / 7 / 8 / 72; TR_0001 kl. 274: 17 / 2 / 5 / 20.

Etykieta działa we wszystkich miejscach, które czytają `h.narz`: przycisk „narzędzie” (cząsteczka w środku narzędzia),
„Wektory dla: narzędzia”, kolor „narzędzie”, pierścienie w okienkach 1–3, okienko 2 (narzędzie/reszta), podkroki.
Rzuty R1–R12 używają etykiety z chwili liczenia stanów 192D (R9 ma własny próg).

**Obszary oka** (źrenica, rogówka; ViT nie był na nich uczony): z obrysów `pupil_seg` i `cornea_seg` tej samej klatki
Cataract-LMM (`etykieta_obrys.npz` w katalogu historii, `Historia.oko`, `Historia.obszary(prog_oka)`), pierwszeństwo
narzędzie > źrenica > rogówka > tło. Kolor „obszary oka” w „Kolor ścieżek” (sfera) i „Kolor cząsteczek” (okienko P),
„Zaznacz obszar ▾” przy wyborze cząsteczek wpisuje patche obszaru do „+ cząsteczki”.

## Poprawki działania (2026-10-22)

- Historie w cache liczone, gdy projekt był na `F:` (ścieżka wag `F:\...\SmallViTAblation\assets\...`): pułapka optyczna,
  obrys ogniska 3D, dolina, mechanika i podkroki kończyły się błędem `PermissionError`. `lasery_dane.sciezka_lokalna`
  przenosi ścieżkę do bieżącej bazy projektu (tak jak już robiły to rzuty).
- Obrys ogniska 3D i brzeg doliny wymagały scikit-image (brak w Pythonie 3.11): zapas `pyqtgraph.isosurface`.

## Pliki i dane

- `app.py` okno, `start.py` start z okienkiem ładowania, `panel_ustawien.py` panel i popupy, `zakladka_laserow.py` cała
  zawartość okna, `lasery_dane.py` obliczenia i `Historia`, `lasery_gpu.py` shadery, `rzuty_*.py` rzuty 192D,
  `ogniskowa.py`, `pole_podkrokow.py`, `pulapka_obliczenia.py`, `dolina.py`, `reflektor_*.py`, `etykiety_narzedzia.py`,
  `obszary_oka.py`; wspólne z edytorem (kopie): `film.py`, `detekcja.py`, `silnik.py`, `widok_gl.py`, `okna.py`, `uklad.py`,
  `styl.py`, `wygoda.py`, `wspolne.py`, `ustawienia.py`, `historie_checkpointow.py`.
- Cache: `cache_uczenia/lasery/<skrót>/` (dane.npz, Z.npy, rzuty/, przejscia/, etykieta_obrys.npz) i `cache_uczenia/etykiety/`.
  Przeniesione z `../app/cache_uczenia/`; tam zostały połączenia katalogów (junction) do tych samych danych, więc stare
  skrypty eksperymentów z ścieżką `app/cache_uczenia/lasery` działają dalej.
- `../app/lasery_dane.py`, `rzuty_dane.py`, `etykiety_narzedzia.py` zostały w edytorze bez zmian jako biblioteka
  historycznych eksperymentów (`experiments/wykrywanie_grupy_narzedzia_bez_etykiet_*`, `wspoldzialanie_neuronow_*` itd.).

Poniżej opis zakładki przeniesiony z `../app/README.md` (odwołania do „lewego panelu” = dziś strony panelu „Parametry”).

## Opis zakładki „Lasery i koloidy” (2026-10-12; `zakladka_laserow.py`, `lasery_dane.py`, `lasery_gpu.py`)

Interaktywna wersja filmów 2, 12 i 16 z `outputs/porownanie_szklo_20261010/film_krysztal` (zasady w `ZASADY_LASER_KOLOID.md`).

**Dane.** Wybierasz:
- historię checkpointów (`historie_checkpointow.HISTORIE` albo dowolny katalog `k<krok>.pt`);
- zakres kroków i co który checkpoint;
- zdjęcie: „Film” (117 filmów zbioru uczącego biegu 35 000) i „Zdjęcie” (jego klatki), albo grupa „Pliki testowe”
  (TRAIN 0 z filmów, `zdjeciatestowe/`, plik).

Klatka zbioru uczącego (2026-10-13) jest czytana jak w `trening35000.py`: z wiersza `MANIFEST_35000.jsonl`, z archiwum zip
(SHA-256 bajtów sprawdzane), rozciągnięta do 480 × 320 INTER_AREA, ramka z manifestu skalowana osobno w X i Y. Uwaga: 15 z 18
plików `zdjeciatestowe/` (w tym całe TR_0077, TR_0127, TR_0140) NIE należy do zbioru uczącego; opis ramki to mówi.
Ramka prawdziwa dla TRAIN 0 pochodzi z `atencja_train0.npz`, dla pozostałych plików `TR_*` z adnotacji.

**Liczenie.** „Policz / wczytaj” uruchamia `lasery_dane.policz` w wątku, na GPU. Dla każdego checkpointu liczy:
- latent 600 × 768 (`Z.npy`, float16);
- kierunki fc2 rzutowane na 3 główne kierunki z ostatniego checkpointu (d_c = P v̂_c) i położenia p_j = P W₂ (z − z̄);
- atencję 6 bloków (definicja aplikacji i filmu 16);
- najpewniejszą ramkę, pewność i IoU;
- pary przeciwnych reflektorów.

Potem dzieli ruch na trzy składowe jak `eden/szescian_prep.py` i film 2: losowy rzut 128D, centrum = średnia z ±okno kroków, Prokrustes 2D (składowa globalna, lokalna i drgania).

**Cache.** Wynik trafia do `cache_uczenia/lasery/<skrót>/`; ten sam wybór wczytuje się od razu. Pierwsze liczenie pełnej historii 35 000 (806 checkpointów, 8,9 GB wag) trwa kilka minut, bo trzeba odczytać wszystkie pliki z dysku.

**Okienka.** Każde można zamknąć, odpiąć jako osobne okno albo przesunąć. Menu „Okienka ▾” je włącza, a „Układ domyślny” przywraca rozkład.
- **Panele 1–3:** ruch globalny, studnie ze strzałkami od adresów, drgania ze śladami (kamera idzie za narzędziem).
- **Sfera reflektorów:** 768 stałych reflektorów, wiązki z jednym załamaniem na sferze. Można pokazać wszystkie 460 800 wiązek, wiązki do wybranej cząsteczki albo z wybranego reflektora.
- **Panel 4:** mozaika atencji 6 bloków.
- **Panel 5:** mapa na zdjęciu (centra, drgania, atencja, natężenie wybranego reflektora).
- **Wykresy 6–9:** trzy składowe ruchu, IoU, pewność, pary przeciwnych reflektorów.

**Rzuty z 192 wymiarów: okienka R1–R8** (2026-10-21; `rzuty_dane.py`, `rzuty_okienka.py`, `rzuty_gpu.py`; uzasadnienie w
`reports/Rzutowanie wielowymiarowego ruchu do 3D.md`). Zwijana sekcja „Rzuty z 192 wymiarów” w lewym panelu, zaraz pod „Dane”.
„Policz / wczytaj stany 192D” robi raz na historię i zdjęcie jedno przejście w przód na checkpoint i zapisuje w katalogu historii
`rzuty/`:
- stany przed MLP bloku 6 (`X.npy`) i siły bloku 6, czyli sumy 768 wiązek (`F.npy`);
- bazy rzutów i mierniki wierności (`dane.npz`);
- kolumny fc2 w osobnym katalogu `cache_uczenia/lasery/_fc2_<skrót>/V.npy`, wspólnym dla wszystkich zdjęć tej historii.

Wszystko liczy się w 192D, a 3D służy tylko do rysowania. Pod każdym okienkiem jest napis, ile ten rzut pokazuje: rozrzutu chmury,
energii sił, energii wiązek, długości wektora narzędzie–tło Δ i ruchu. Obok jest zero dla losowego 3D: 1,6 % energii, 12,5 % długości.
- **R1 A:** oś Δ narzędzie − tło (100 % separacji) i PCA reszty chmury; baza wspólna albo w każdym checkpoincie (Procrustes osi 2–3).
- **R2 S:** 3 kierunki największej energii sił netto.
- **R3 W:** wiązki ważone aktywacją (następca P3) i wiązki wybranej cząsteczki.
- **R4 B:** rama siły wybranej cząsteczki: oś F̂ i płaszczyzna, w której jej wiązki się znoszą. Rysuje sieć silną (próg energii)
  i resztę jako jedną szarą strzałkę. Kolor wiązki to cos z F̂ w 192D.
- **R5 C:** rama Schura, czyli podprzestrzeń niezmiennicza pełnego jakobianu 192 × 192 (para zespolona i mod rzeczywisty). Pokazuje
  pole lokalnej deformacji, płaszczyznę obrotu, kierunek własny i widmo z zerem Ginibre'a. Liczy się w tle.
- **R6 D:** wędrówka w czasie uczenia (wszystkie, narzędzie, wybrane albo jedna cząsteczka) w bazie przemieszczeń albo w ramie
  cząsteczki (start → koniec). Szare tory zerowe to błądzenie losowe o tych samych krokach.
- **R7 F:** 7 widoków płaszczyzny Fano nad 7 kierunkami (Δ, reszta chmury, siły, wiązki); każda para kierunków jest razem dokładnie raz.
- **R8:** panel bez rzutu: wierność widoków w czasie, separacja wobec permutacji etykiet, kosinusy w 192D, pary wiązek i widmo J.
- **R9 L (2026-10-21):** separacja narzędzia od tła z własną sekcją parametrów. Oś 1 = LDA ze skurczem, czyli różnica średnich
  wybielona kowariancją wewnątrz klas: w ∝ (S_w + λ·tr(S_w)/192·I)⁻¹ Δ, dopasowana na puli ostatnich N checkpointów (stała kamera)
  albo na bieżącym. Osie 2–3 do wyboru: największy lub najmniejszy rozrzut wewnątrz klas w dopełnieniu, PCA tła, PCA narzędzia.
  „Brzeg ramki” pomija skrajne kolumny i wiersze prostokątnej ramki (to piksele tła; rysowane na żółto). W podpisie miary separacji
  w tym rzucie i w 192D: sylwetka narzędzia, trafienia kNN (udział narzędzia wśród 10 sąsiadów), margines (najmniejsza odległość
  narzędzie–tło w odstępach najbliższych sąsiadów tła) i stosunek Fishera, każda z zerem = 95. percentyl dla losowych etykiet
  w tej samej bazie. Uzasadnienie i ograniczenia: `reports/Osie najlepszej separacji narzędzia od tła.md`.

Ślady w okienkach globalnych pokazują wędrówkę przez checkpointy. Opcja „Wygładzanie śladu” uśrednia ostatnie checkpointy, żeby zamiast
szumu SGD widać było dryf. Strzałki to siła netto bloku 6; krycie strzałki to część jej długości widoczna w rzucie. R1–R7 leżą w
kartach obok sfery, R8 obok wykresu 6, a przyciski sekcji wyciągają je na wierzch.

Rysowanie i liczenie są na karcie:
- **Shadery GLSL 4.30 (`rzuty_gpu.RzutGPU`, SSBO jak `lasery_gpu.py`).** Cała historia położeń i sił w bazie okienka trafia do
  bufora raz, przy zmianie bazy, środka, skali albo wygładzania. W klatce idzie tylko czas. Interpolację, ślady, groty strzałek
  zwrócone do kamery i krycie liczy shader.
- **Ramy lokalne R4 i R5** to elementy lokalne tego samego shadera, wysyłane raz na checkpoint i cząsteczkę.
- **Rzutowanie 192D → 3D** całej historii liczy torch na karcie (`DaneRzutow.wsp`).
Testy: `testy/test_rzuty_dane.py`.

**Oś czasu.** Suwak, odtwarzanie (spacja) i strzałki ←/→ (checkpoint). Klik w wykres przenosi do danej chwili. Tempo „jak w filmie 16” to v = min(0,8 + krok/250, 150) kroków na klatkę; tempo stałe liczy się w checkpointach na sekundę.

**Wybór cząsteczki.** Klik w cząsteczkę w panelach 1–3 albo na sferze wybiera patch. Przycisk „narzędzie” bierze patch w środku ramki, a „najzmienniejszy” reflektor o największej wariancji natężeń.

**Rysowanie.** Robią je shadery GLSL 4.30 z buforów SSBO, a cała historia leży na karcie. W klatce CPU podaje tylko czas i opcje. Wyjątek to natężenia z_jc pary sąsiednich checkpointów dla wiązek (600 × 768), wysyłane przy zmianie pary.

**Okienko „Przejście przez sieć: podkroki”** (`lasery_dane.podkroki`, `lasery_gpu.PrzejscieGPU`, `DiagramSieci`).

Dla checkpointu bieżącej chwili osi czasu jedno przejście w przód jest rozłożone na 13 stanów strumienia rezydualnego: stan 0 to łodyga z kodowaniem pozycji, a każdy z 6 bloków dokłada dwa stany, po uwadze i po MLP. Czasem cząstki jest tu głębokość sieci. Rozkład jest dokładny (sprawdzany w każdym liczeniu, błąd około 5·10⁻⁵):
- uwaga: Δx_i = Σ_j f_ij + b_o, gdzie f_ij = Σ_h A^h_ij · W_o v^h_j to siła cząsteczki j na cząsteczkę i;
- MLP: Δx_j = Σ_c z_jc v_c + b₂, czyli wkład 768 laserów.

Diagram jak `diagram_maly_vit.pdf` podświetla bieżący podkrok; klik w uwagę albo MLP bloku przechodzi do tego podkroku. Przycisk „▶ podkroki” animuje przejście sygnału w pętli.

W podkroku uwagi widok pokazuje:
- linie od wybranej cząsteczki do tych, których słucha (albo linie wszystkich par);
- wektory sił f_ij w kolorze cząsteczki j.

W podkroku MLP widok pokazuje 768 wektorów laserów z_jc · v_c dla wybranej cząsteczki i strzałki przesunięcia wszystkich cząsteczek.

Opis pod widokiem podaje |Δx|, trzech najważniejszych partnerów uwagi, udział sąsiadów na siatce w sile, liczbę świecących laserów i najsilniejsze lasery.

Układy widoku, każdy z własnym rzutem PCA 3D:
- znormowany w każdym stanie, jak LayerNorm (domyślny);
- względem środka chmury;
- surowy strumień rezydualny.

Jeden checkpoint liczy się w około 0,1 s, w tle, dla checkpointu najbliższego bieżącej chwili osi czasu.

**Wypadkowe wektory i synchronizacja (2026-10-12).** W trybie podkroków każda cząsteczka ma wypadkowy wektor bieżącego podkroku, czyli przesunięcie od stanu przed podkrokiem do stanu po nim (suma sił albo laserów). Uwaga jest niebieska, MLP pomarańczowe, a wybrana cząsteczka gruba z grotem. Wektory widać na sferze i w rzutach 2–3, z regulowaną długością.

Opis podkroku podaje:
- stosunek rozrzutu chmury r_po / r_przed (r = średnia |x_i − x̄| surowo);
- składową promieniową przesunięcia względem środka chmury (ujemna = do środka).

„Synchronizuj kroki z podkrokami” zmienia odtwarzanie osi czasu: w każdym checkpoincie przechodzą wszystkie 12 podkroków, potem następuje skok o n checkpointów.

Analiza zbiorcza: `outputs/porownanie_szklo_20261010/film_krysztal/analiza_kurczenie_uwaga_mlp.py` → `KURCZENIE_UWAGA_MLP.json` (40 checkpointów × 4 zdjęcia).

**Sfera: koloidy, pułapki i wektory (2026-10-12).**

> **Zmiana nazwy (2026-10-14).** To, co niżej nazywa się „pułapką” (średnie położenie koloidu z ± okna kroków), w aplikacji
> nazywa się teraz **„średnia skoków koloidu”**. Jest liczona z ruchu koloidu i z przyszłych kroków, więc nie jest przyczyną
> ruchu. Słowo **„pułapka”** oznacza od teraz pułapkę optyczną z laserów (ogniskową wiązek, sekcja niżej i okienko 11).

„Pokazuj” wybiera, co widać na sferze: koloidy (tokeny), pułapki albo jedno i drugie. Pułapka koloidu to jego średnie położenie z ± okna kroków, jak centra studni w filmie 2, liczone w tych samych współrzędnych sfery i rysowane jako zielona poświata.

Wektory koloidów (żółte) i pułapek (zielone), osobno dla każdego rodzaju, mają cztery tryby:
- brak;
- z kroku na krok: od poprzedniego checkpointu do teraz;
- od kroku 0: od adresu startu do teraz;
- cała historia: droga przez wszystkie checkpointy od 0 do teraz.

Wektory można pokazać dla wszystkich cząsteczek, dla narzędzia albo dla wybranej cząsteczki, ze skalą długości.

W trybie podkroków pułapka pokrywa się z koloidem: „z kroku na krok” to poprzedni podkrok, a historia to droga przez podkroki. Przy skali chmury dopasowanej do kroku drogi zawierają też zmianę skali między checkpointami; do porównań lepsza jest skala stała.

**Pętla.** Pole „pętla” przy odtwarzaniu osi czasu sprawia, że po ostatnim checkpoincie odtwarzanie wraca do kroku 0, w obu tempach i w trybie synchronizacji z podkrokami.

**Ścieżki pułapek i panel (2026-10-12).**

Opcje ścieżek na sferze:
- **kolor ścieżek:** jednolity, kolor patcha ze zdjęcia, położenie na siatce albo narzędzie;
- **wygładzanie pułapek** wzdłuż checkpointów: średnia ważona krocząca (wagi okno, okno−1, …, 1; tylko przeszłość), EMA (α na checkpoint) albo obie naraz; dotyczy drogi, wektorów i poświaty pułapek;
- **zanik historii:** jasność odcinka exp(−3·wiek/zanik), gdzie wiek liczony jest w krokach uczenia.

Panel po lewej ma sekcje zwijane polem przy tytule, pionowe przewijanie i przeciągany podział szerokości.

**Kolor „odpływ od adresu”** (kolor ścieżek, plasma jak w okienku 2). Każdy punkt drogi i koniec wektora koloidu albo pułapki ma kolor według odległości od położenia w kroku 0 (adresu) w tej chwili: ciemny fiolet oznacza punkt przy adresie, żółty 99. percentyl odpływu (osobno dla koloidów i pułapek, po wygładzeniu).

**Skalowanie w sferze.** Opcje w sekcji sfery:
- wielkość koloidów (px);
- wielkość pułapek (średnica poświaty, px) i jasność pułapek;
- skala chmury w sferze: mnożnik rozmiaru chmury koloidów i pułapek względem szklanej sfery, 1 jak w filmie 12. Wiązki, punkty załamania, wektory, drogi i wybieranie kliknięciem przeliczają się razem z chmurą; kolor odpływu się nie zmienia.

**Pułapka optyczna: ogniskowa wiązek i okienko 11 (2026-10-14).**

Sekcja „Pułapka optyczna” liczy (`ogniskowa.py`, w tle, z cache obok historii: `ogniskowa_t<tok>_G<G>_z<zasięg>.npz`)
pole wszystkich 768 wiązek bloku 6 wokół wybranej cząsteczki, w każdym checkpoincie, prosto z wag:
- przekrój 3D przez koloid rozpięty rzutem sfery P3: `x(u) = x̃_j + P3ᵀu`, pole `f(u) = P3 Σ_c (z_c(x(u)) − z̄_c) v_c`;
  `f(0)` to dokładny skok koloidu w bloku 6 (pomarańczowa strzałka);
- **pułapka** = punkt, w którym pchnięcia wszystkich wiązek się znoszą (`f = 0`, Newton od koloidu, do 3 promieni chmury r99);
  jej charakter z wartości własnych Jakobianu 3 × 3: błękit ściąga, czerwień wypycha, fiolet = wir (okrąg z grotami);
- mgła pola: kolor = wiązki o największym wkładzie (kolory reflektorów jak na sferze), jasna tam, gdzie pchnięcia się znoszą;
- strzałki: pełne pole albo „kształt pułapki”, czyli pole minus skok samego koloidu (jak wiązki rozpychają, ściągają i obracają
  otoczenie koloidu);
- ślad koloidu (kropki = skoki między checkpointami) i pułapki, w układzie laboratorium albo względem pułapki;
- odtwarzanie „najpierw lasery, potem koloid”: w pierwszej połowie przejścia między checkpointami zmienia się pole, pułapka
  i skok, w drugiej koloid przechodzi na nowe miejsce (obie zmiany pochodzą z tego samego kroku uczenia);
- okienko 11 („Pułapka optyczna: koloid skacze w polu wiązek”) z kamerą śledzącą pułapkę, koloid albo środek pola; ta sama
  pułapka opcjonalnie na dużej sferze.

**Zbliżenie na ognisko jednej cząsteczki (2026-10-14, domyślny widok okienka 11).** Duża sfera to cały układ; okienko 11 to mały
wycinek wokół JEDNEJ wybranej cząsteczki: 768 wiązek (po jednej z każdego reflektora) zbiega się w niej z kierunków −d_c
(d_c = P v_c), w kolorach reflektorów. Domyślnie wiązka wchodzi z kierunku, w który naprawdę pcha (wkład (z_jc − z̄_c)·P v_c),
z jasnością |z_jc − z̄_c|·|P v_c|; drugi tryb jak na sferze (kierunek v_c, jasność z_jc, błękit = ujemna dioda). Światło ogniska:
każda wiązka to wiązka gaussowska z przewężeniem w cząsteczce (przewężenie, rozbieżność, jasność), a osie kształtu ogniska
pochodzą z tensora kierunków. Czas okienka: oś uczenia (blok 6) albo podkroki (MLP: 768 wiązek bloku; uwaga: 600 wkładów od innych
cząsteczek), z kontrolą „suma wkładów = ruch” w opisie (błąd ~10⁻⁶). Domyślnie liczone są tylko wiązki wybranej cząsteczki
(806 checkpointów w ~47 s); pole na siatce i szukanie zera to opcja „licz też lokalne pole”. Wkłady (z_jc − z̄_c)·P v_c jako strzałki od cząsteczki albo łańcuch głowa–ogon, którego koniec to dokładnie skok
(pomarańczowa strzałka); długość łańcucha pokazuje, ile pchnięć się znosi. Pole na siatce (mgła, strzałki) jest opcją i domyślnie
lokalne (zasięg 0,05 r99). Kamera domyślnie śledzi cząsteczkę. Dane wiązek zapisuje `ogniskowa.py` od wersji 2 (pv, zj, zbar).
Opcja „reflektory” pokazuje okienko 11 w skali sfery reflektorów (film 12): 768 reflektorów na zewnętrznej sferze, szklaną sferę
i wiązkę każdego reflektora do wybranej cząsteczki (reflektor → punkt załamania na szkle → cząsteczka, ostatni odcinek wzdłuż
kierunku wiązki). Kamera bierze kąty i skalę (piksele na jednostkę) z okna sfery i przestaje śledzić cząsteczkę. W podkrokach MLP
to reflektory danego bloku, a w podkrokach uwagi źródłami są inne cząsteczki.
Opcja „obrys ogniska 3D” rysuje powierzchnię, na której suma światła wiązek gaussowskich (te same wiązki co w rysowanym świetle)
spada do wybranego ułamka jasności w cząsteczce (domyślnie 0,5 = połowa szczytu, jak objętość ogniska w optyce); liczona na GPU
(torch) i metodą marching cubes (`skimage`), około 12 ms na klatkę. Półprzezroczysta powierzchnia z warstwicami: złote = przekroje
wzdłuż najdłuższej osi obrysu, białe = wzdłuż drugiej. Prawdziwy obrys ma 1–2 przewężenia wiązki (kilka pikseli), więc jest rysowany
z lupą (domyślnie ×20). Kształt pochodzi z danych (kierunki i jasności wiązek), rozmiar od parametrów rysunku (przewężenie,
rozbieżność). Przykład (krok 65640, cząsteczka 226, poziom 0,5): blok 4 wydłużenie ×2,1, bloki 1 i 6 ×1,2.

**Pole tylko w komórce tokenu i podgląd na żądanie (2026-10-15, `ogniskowa.py` wersja 4).** Lokalne pole wiązek liczy się tylko
w komórce tokenu: sześcian wokół koloidu o połowie boku r_kom = połowa odległości do najbliższego innego tokenu w rzucie 3D (w każdym
checkpoincie osobno; pole „Zasięg siatki” to ułamek komórki, domyślnie 1). Żaden inny token nie leży w sześcianie (sprawdzone:
0 w 806 checkpointach), siatka nie rośnie do pułapki, a zera pola szuka się tylko w komórce. Kamera „kadruje pole tokenu”: środek
w koloidzie, odległość taka, że sześcian wypełnia okienko 11 w każdym checkpoincie; wiązki, światło i wkłady są wtedy przycięte do
sześcianu. Ślad koloidu i pomarańczowy skok mogą wychodzić poza komórkę, bo token skacze dalej niż jego komórka (cząsteczka 226:
r_kom 0,003–0,355 r99, mediana 0,15 r99; skok 0,3–0,7 r99). Zero pola w komórce: 0 z 806 checkpointów (liniowe oszacowanie
pułapki leży 9–270 r_kom dalej). Kadr okienka 11 ma dwa ustawienia: środek (pułapka albo koloid zawsze na środku okienka) i skala
(swobodna; sześcian zawsze tej samej wielkości na ekranie; stała przez całe uczenie, jedna odległość kamery z 90. percentyla
wielkości komórki, więc widać, jak komórka rośnie i maleje). Przy środku w pułapce rysowana jest też pułapka zastępcza (fioletowa
kropka, najsłabsze pole w komórce), gdy zera pola w komórce nie ma; dla cząsteczki 226 leży ona w rogu sześcianu.
Pole jest pokazywane w KULI komórki (promień r_kom), nie w sześcianie: siatka liczona jest na sześcianie G³, ale mgła i strzałki
poza kulą są ukryte, a zamiast krawędzi pudła rysowane są okręgi kuli. Strzałki pola płyną wzdłuż pola (faza losowa w każdym węźle,
pojawiają się i gasną; także na pauzie, timer tylko gdy okienko 11 jest widoczne), a strzałka wychodząca poza kulę znika. W kadrze
komórki obrys ogniska jest skalowany tak, że jego najdłuższa półoś = promień komórki (lupa automatyczna), więc pole jest w środku obrysu.
Podkroki w okienku 11 (przebieg przez wszystkie 6 bloków, „Czas okienka 11: podkroki”): jedna długość wiązek na całe przejście
checkpointu (1,5 × największy ruch cząsteczki w 12 podkrokach), więc wiązki, światło i obrys nie skaczą między uwagą a MLP; przy
skali kadru innej niż swobodna odległość kamery też jest stała. Tytuł okienka 11 pokazuje tryb oraz bieżący blok i podkrok.
Przełączenie okienka 11 na podkroki włącza samo widok reflektorów w skali sfery (film 12): jedna kamera dla uwagi i MLP,
reflektory bloku świecą w każdym podkroku (w uwadze bez wiązek, z liniami od 600 cząsteczek), a wiązki nie są wygaszane na
początku podkroku. Powrót do osi uczenia wyłącza ten widok, jeśli włączyło go przełączenie.
Pozostałe 599 cząsteczek są rysowane w tle okienka 11, przyciemnione (kolory cząsteczek × jasność tła, domyślnie 0,35): w podkrokach
w tej samej chwili przejścia co koloid, w osi uczenia z historii w chwili koloidu. W uwadze cząsteczki-źródła nie są już osobnymi
jasnymi punktami, świecą tylko linie ich wkładów.
Strzałki pola: domyślnie „pole wiązek” (widać je od razu po policzeniu pola na siatce). Są tylko w osi uczenia (blok 6); w podkrokach
pole na siatce nie jest liczone, co mówi opis pod okienkiem 11.
Gruba zielona strzałka to wypadkowy ruch koloidu w nadchodzącym kroku (położenie w następnym checkpoincie − w bieżącym); opis podaje
jego długość i kąt do pomarańczowego skoku. To dwie różne rzeczy: skok to pchnięcie wiązek w głąb sieci (blok 6, jedno przejście),
zielony ruch to zmiana położenia między checkpointami (uczenie). Dla cząsteczki 226 są prawie prostopadłe: kąt mediana 85°
(kwartyle 55–117°), ruch ≈ 0,5 długości skoku. Po uruchomieniu programu ogniskowa nic nie wczytuje ani nie liczy, dopóki nie zaznaczysz
„pokaż podgląd ogniskowej” albo nie klikniesz „Wczytaj / policz”.

**Zapis przejść przez sieć na dysk (2026-10-15, `przejscia_zapis.py`, sekcja „Zapis przejść przez sieć”).** Policzone przejścia
(podkroki jednego checkpointu) zapisują się do `cache_uczenia/lasery/<historia>/przejscia/k<krok>.npz` i potem wczytują się
zamiast liczenia (0,07 s zamiast 0,1–0,7 s, pierwsze liczenie w sesji ~5 s). Wspólny rzut 3D i przypisanie reflektorów są w
`baza.npz`, więc zapisane przejścia pasują do siebie także między sesjami. Przycisk „Zapisz przejścia wszystkich checkpointów”
liczy je w tle (co n checkpointów, ostatni zawsze, już zapisane pomija). Zapis dokładny float32 (~44 MB na checkpoint, ~35 GB na 806)
albo oszczędny float16 dla atencji, jasności i stanów (~22 MB); limit wielkości katalogu (domyślnie 40 GB). ah_c i z_c są odtwarzane
przy wczytaniu (różnica od policzonych ≤ 2·10⁻⁶). Ogniskowe w osi uczenia zapisywały się już wcześniej
(`ogniskowa_t<tok>_G<G>_z<zasięg>.npz` w katalogu historii).

Kontrola: jasności z mojego przeliczenia zgadzają się z przejściem sieci do 4·10⁻⁶, położenia z cache historii do 3·10⁻⁷.
Wynik na historii s47 (806 checkpointów, cząsteczka narzędzia 226, siatka 17³, zasięg 0,3 r99, liczenie 4,6 min na lokalnym GPU):
zero pola w zasięgu 3 r99 jest tylko w 62 z 806 checkpointów (7,7 %), głównie na początku uczenia, gdy skok koloidu jest
mały. Zwykle koloid nie siedzi w ognisku: wiązki niosą go prawie jednorodnym prądem (skok 0,3–0,7 r99), a wokół niego pole
ma kształt siodła z wirem (np. ściąga −0,19, obraca 0,21); liniowe oszacowanie zera pola leży wtedy ~2 r99 dalej (szara strzałka).

**Płynna synchronizacja (2026-10-12).** Odtwarzanie z synchronizacją kroków i podkroków działa bez pauz:
- Po podkroku 12 następuje od razu płynny powrót (pole „powrót”, w podkrokach; 0 = natychmiast): cząsteczki na sferze i w rzutach 1–3 przechodzą z wyjścia sieci bieżącego checkpointu do wejścia następnego.
- Następne dwa checkpointy liczą się w tle od podkroku 2, razem z rzutami płaskimi i statystykami.
- Wszystkie checkpointy mają wspólny rzut 3D i to samo przypisanie reflektorów do miejsc na sferze (z pierwszego policzonego), więc osie się nie obracają, a reflektory stoją w miejscu jak w filmie 12.
- Animacje odtwarzania liczą czas rzeczywisty (60 kl/s), więc tempo nie zależy od liczby klatek.
- Napisy pod widokami nie zmieniają układu okien. Wcześniej zmiana długości tekstu zmieniała rozmiar widoków OpenGL i dawała przerwy ponad 100 ms.
- Okienko szczegółów podkroków dostaje dane tylko wtedy, gdy jest widoczne.

Pomiar: około 45 kl/s, 95% klatek do 38 ms, przy zmianie checkpointu najwyżej około 60 ms.

**R9: zero z ponownym dopasowaniem i walidacja blokowana (na karcie).** Dwie opcje sekcji R9. „Zero z ponownym dopasowaniem osi”
(DiProPerm): dla każdej losowej etykiety oś LDA jest dopasowywana od nowa, więc zero obejmuje przeuczenie 18 punktów w 192D. „Walidacja
blokowana po checkpointach”: pula jest dzielona na ciągłe bloki, oś z pozostałych bloków jest oceniana na checkpoincie wyłączonego bloku
(chroni przed korelacją czasową, nie przed tymi samymi tokenami). Wszystko liczy karta: wsadowe `solve` i `eigh` w torch, miary z
`cdist` na GPU, float32 i TF32 (zgodne z CPU float64 do 0,2° w osiach), ok. 0,5 s na checkpoint wobec 14 s w numpy. Podczas odtwarzania
liczą się tylko szybkie miary podstawowe, ciężkie dopiero na pauzie.

**Etykieta narzędzia z obrysu, nie z ramki (2026-10-21).** `etykiety_narzedzia.py` czyta wielokąty `<narzędzie>_seg` z
annotation.csv Cataract-LMM 3_Object_Tracking i liczy udział pikseli narzędzia w każdej łacie 16 × 16. Historia (`lasery_dane.Historia`)
odnajduje swoją klatkę po ramce GT (indeks ramek wszystkich filmów w `cache_uczenia/etykiety/indeks_ramek.npz`, przy kilku kandydatach
rozstrzyga obraz) i ustawia `narz` = udział ≥ 0,5 (przy < 2 łatach próg 0,25), `mieszane` = 0 < udział < próg, a starą etykietę
zachowuje jako `narz_ramka`. Bez obrysu zostaje ramka. Cienki cystotom zajmuje w ramce 2 × 9 tylko 3 łaty w co najmniej połowie,
13 to łaty mieszane, a reszta to tło, więc etykieta z ramki była w większości tłem. Rzuty mają `WERSJA = 2` (przeliczają się raz)
i liczą Δ narzędzie–tło bez łat mieszanych.

**R9: etykieta, zero przestrzenne, Fisher jako główna miara.** Sekcja R9 ma wybór etykiety (obrys z progiem udziału albo stara ramka GT
z „brzegiem ramki”). Łaty mieszane są żółte i nie wchodzą ani do dopasowania, ani do miar. „Zero przestrzenne”: ta sama maska narzędzia
przesunięta w każde położenie siatki rozłączne z pikselami narzędzia, oś LDA dopasowana od nowa dla każdego położenia
(`RD.zero_przestrzenne_gpu`, wsadowo na karcie). To właściwe zero, bo dowolny spójny kawałek tła też się „oddziela” (sąsiednie łaty
są podobne); zero z losowych, rozrzuconych tokenów (0,1) było zbyt łagodne wobec ramek tła (sylwetka ok. 0,7). Podpis podaje p =
odsetek położeń tła z miarą ≥ narzędzia. Fisher (odległość klas względem rozrzutu) jest pierwszą liczbą, bo sylwetka nasyca się
przy ok. 0,9 także dla tła. Trafienia kNN mają k = min(10, n_narz − 1): przy 3 łatach narzędzia k = 10 dawałoby najwyżej 0,2.

**R9: grupa narzędzia wykryta bez etykiet (2026-10-21).** Trzecia etykieta w R9: „grupa wykryta bez etykiet”. `RD.wykryj_grupe`
liczy TokenCut (graf kosinusowy aktywacji MLP bloku 6, 600 × 768, τ = 0,2, drugi wektor własny znormalizowanego laplasjanu,
strona z największym |y|, największa spójna składowa) w danym checkpoincie z `h.Z`, bez adnotacji. Oś LDA jest dopasowana do tej
grupy, a obrys służy tylko do sprawdzenia: podpis podaje Dice, precyzję, czułość i p przesunięć grupy po siatce
(`RD.zgodnosc_z_obrysem`); łaty narzędzia z obrysu pominięte przez grupę są żółte. Konfigurację wybrano na filmach DEV (52 filmy,
576 klatek) spośród 448 konfiguracji (cechy bloków 3–6 i MLP, kurtoza, odstawanie, TokenCut). Na filmach TEST (62 filmy, 692 klatki)
dla seedu 3047, epoka 400: Dice 0,47 [0,44–0,50], precyzja 0,62, 94 % tokenów grupy w narzędziu lub ≤ 1 łatę od niego; wagi
nieuczone: Dice 0,04. Eksperyment: `experiments/wykrywanie_grupy_narzedzia_bez_etykiet_20261021__21-42-46`.
Opcja „Oś LDA” (przy tej etykiecie): oś z obrysu (domyślnie; kolor narzędzia = wykryta grupa, więc widać, czy grupa leży tam,
gdzie oznaczone narzędzie) albo oś z wykrytej grupy (widok całkowicie bez etykiet).

**Nawigacja myszą w widokach OpenGL (2026-10-22).** Wspólna dla wszystkich okien (`widok_gl.WidokGL`): LPM obraca (w widoku z góry
przesuwa), ŚPM albo Ctrl+LPM przesuwa w płaszczyźnie widoku, Ctrl+ŚPM przesuwa w pionie, PPM + przeciągnięcie góra/dół zbliża i
oddala, kółko zbliża do punktu pod kursorem, Ctrl+kółko zmienia pole widzenia, podwójny ŚPM wraca do kadru, Shift+LPM rysuje
prostokąt zaznaczenia. Prawy klik bez przeciągania zgłasza sygnał `prawy_px`. Zakładka laserów ma przyciski „Zwiń wszystkie”
i „Rozwiń główne” nad sekcjami lewego panelu.

**Okienka R0 i R10 (2026-10-22).** R0 to zdjęcie klatki z ramką GT (biała kropkowana), ramką modelu (cyjan) z IoU i pewnością w bieżącym
checkpoincie, patchami narzędzia z obrysu (zielone), mieszanymi (żółte), grupą wykrytą bez etykiet (czerwone, TokenCut) i wybraną
cząsteczką (biała); działa bez policzonych stanów 192D. R10 to wykresy z eksperymentu liczone na żywo: LDA narzędzie–tło (oś jak
w R9 albo dopasowana na checkpoincie), obok ta sama maska przesunięta w tło (mediana i najlepsze położenie wg Fishera, LDA od nowa),
w podpisie p zera przestrzennego. Maska w tle liczy się na pauzie (ok. 1 s na GPU). Parametry: sekcja „R0 i R10” oraz sekcja R9
(etykieta, λ, osie).

**Stereo 3D (SBS) dla monitorów 3D, np. Samsung Odyssey 3D (2026-10-22).** Pole „Stereo 3D (SBS)” nad sekcjami lewego panelu zakładki
laserów przełącza wszystkie widoki OpenGL (`WidokGL.ustaw_stereo`) między zwykłym obrazem a half side-by-side: scena jest rysowana
dwa razy (lewe oko w lewej, prawe w prawej połowie okna, każde zgniecione do połowy szerokości, z nożyczkami), z kamerą przesuniętą o
±połowę odstępu w układzie widoku i asymetrycznym frustumem (zerowy rozrzut na odległości środka widoku). „Odstęp” to ułamek tej
odległości (domyślnie 0,035), „zamień oczy” odwraca głębię. Okno z widokiem rozwiń na cały ekran monitora 3D (⛶ albo F11). Napisy
GLTextItem są w stereo ukryte, a klikanie w cząsteczki w trybie stereo nie jest dokładne (picking liczy z kamery środkowej).

**R11: miary separacji w czasie uczenia (2026-10-22).** Cztery wykresy w jednym okienku, z kursorem czasu: Fisher po LDA, Fisher w pełnych
192D, kNN w PCA3 chmury (bez etykiet), sylwetka w 192D. Zielona linia = narzędzie (etykieta z obrysu), szara przerywana = 95. percentyl tej samej
miary dla 48 losowych położeń tej samej maski w tle (zero przestrzenne, LDA od nowa). Liczone na GPU w tle (`RD.miary_w_czasie_gpu`, co 4.
checkpoint, ok. 20 s dla 806 checkpointów) i zapisywane w `rzuty/miary_czas_v1.npz`. R10 ma też podgląd „PCA3 chmury (bez etykiet)”
w pierwszym panelu.

**R12: kNN na chmurze 3D (2026-10-22).** Okienko OpenGL z chmurą 600 tokenów w 3D (PCA3 chmury bez etykiet albo oś LDA jak w R9): duże kule
= tokeny narzędzia, kolor = ułamek ich k najbliższych sąsiadów, którzy też są narzędziem (zielony 1, czerwony 0), linie do sąsiadów
(zielone: sąsiad narzędzie, czerwone: tło). Sąsiedzi liczeni w widocznym 3D albo w pełnych 192D (rysunek zawsze 3D); k auto albo
własne. Kamera mieści każdy token (z histerezą skali). Opcje w sekcji „R0 i R10”.

**F11 = pełny ekran okienka (2026-10-22).** F11 z kursorem myszy nad okienkiem (dokiem) przełącza je do prawdziwego pełnego ekranu na
jego monitorze (bez ramki i paska zadań, potrzebne do stereo SBS); F11 jeszcze raz albo Esc przywraca poprzednie położenie. Przycisk ⛶
w pasku okienka nadal robi zwykłą maksymalizację. Dla okien innych niż doki F11 maksymalizuje jak dotąd.

**Klawiatura w widokach 3D (2026-10-22).** Sterowanie dotyczy widoku pod kursorem myszy (albo klikniętego); pola tekstowe i listy nie tracą
fokusu. Przytrzymanie klawisza działa płynnie (timer 16 ms): W/S obrót w górę/dół (60°/s), A/D obrót w lewo/prawo (90°/s), Q/E (też −/+)
oddalanie i zbliżanie, I/K/J/L przesuwanie w płaszczyźnie widoku, Home albo 0 powrót do kadru; Shift = 3× szybciej, Ctrl = 0,3×.
W widoku z góry WASD przesuwa. Strzałki i spacja zostają dla osi czasu zakładki.
