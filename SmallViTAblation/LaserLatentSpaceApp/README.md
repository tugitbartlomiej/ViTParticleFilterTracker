# Lasery i koloidy — aplikacja (LaserLatentSpaceApp)

Osobna aplikacja PyQt6 wydzielona 2026-10-22 z zakładki „Lasery i koloidy” edytora latentu (`../app`). Oś czasu uczenia
jednego zdjęcia przez checkpointy małego ViT: sfera reflektorów, pułapka optyczna, studnie, drgania, atencja 6 bloków,
rzuty z 192 wymiarów, dolina pewności i IoU, mechanika reflektora (filmy 2, 12, 16).

## Uruchomienie

- `Lasery i koloidy.exe` (dwuklik, bez konsoli): mały uruchamiacz (48 kB, źródło `launcher/Launcher.cs`, ikona
  `launcher/icon.ico`), startuje `W:\Python.11\pythonw.exe run.py`; inny Python: pełna ścieżka do `pythonw.exe` w `python.txt`
  obok exe. Przebudowa (PowerShell, z katalogu aplikacji):
  `& C:\Windows\Microsoft.NET\Framework644.0.30319\csc.exe /nologo /codepage:65001 /target:winexe /win32icon:launcher\icon.ico /r:System.Windows.Forms.dll '/out:Lasery i koloidy.exe' launcher\Launcher.cs`.
- Z konsoli: `run.bat` albo `python run.py` (okienko ładowania), `python -m laser_app` (bez niego).
- Wymagania: torch + CUDA, timm, PyQt6, pyqtgraph, PyOpenGL, opencv-python, scipy, numpy. scikit-image jest opcjonalne
  (obrysy 3D ogniska i doliny liczą się bez niego przez `pyqtgraph.isosurface`, `common.izopowierzchnia`).
- Błąd startu: okno z komunikatem i `run.log`; wyjście programu bez konsoli: `output.log`.
- Testy (z katalogu aplikacji; `test_gui` osobno, na ekranie):
  `$env:PYTHONPATH='.'; python -m unittest tests.test_modules tests.test_tool_labels tests.test_projections tests.test_tooltips`
  i `python -m unittest tests.test_gui`.

## Struktura (refaktor 2026-10-22: pakiet, angielskie nazwy plików i klas)

```
LaserLatentSpaceApp/
  run.py, run.bat, Lasery i koloidy.exe     start (okienko ładowania), uruchamiacz
  laser_app/                                pakiet aplikacji
    __init__.py, __main__.py                PyOpenGL bez glGetError; python -m laser_app
    main_window.py                          MainWindow: menu, ustawienia, zrzuty okienek
    settings.py, paths.py                   zapis ustawień (settings.json), ścieżki aplikacji i projektu
    lasers_tab/                             główna zakładka
      tab.py                                LasersTab: budowa (sekcje panelu, okienka), opcje, ustaw_czas
      layout.py, data_view.py, image_set_view.py, labels_view.py, substeps_view.py,
      trap_view.py, trap_map_view.py, valley_view.py, mechanics_view.py, microlens_view.py, playback.py
                                            mixiny LasersTab (okienka-doki, dane, zestaw zdjęć, etykiety, podkroki,
                                            pułapka 11, mapa pułapek 14, dolina 12, mechanika 13, mikrosoczewki M, oś czasu)
      workers.py, widgets.py, shared.py     wątki (QThread), widżety zakładki, stałe
    ui/                                     widżety Qt
      settings_panel.py, parameter_groups.py, tooltips.py, label_preview.py, widgets.py,
      dock_windows.py, gl_view.py, input_comfort.py, style.py
      projections_view.py                   ProjectionWindows: okienka rzutów R0–R12 (dane, czas, budowa)
      projection_windows/                   mixiny okienek R (zdjęcie i LDA, miary R11, kNN R12, panel R8, ramy B i C,
                                            bazy, rysowanie i opisy), wątki, sceny 3D, stałe
    gpu/                                    elementy OpenGL 4.3 (GLSL 4.30, SSBO)
      base.py                               stałe, kompilacja shaderów, _GPUElement, wspólne funkcje
      sphere_gl.py, focal_field_gl.py, planar_gl.py, image_maps_gl.py, network_passage_gl.py,
      valley_gl.py, schematic_gl.py, trap_map_gl.py, projections_gl.py
    compute/                                obliczenia bez Qt: history, focal_trap, trap_map, substep_field,
                                            trap_measurements, passage_store, projections, tool_labels, valley,
                                            reflector_mechanics, reflector_drawing, checkpoint_histories, common
  tests/                                    unittest (conftest.py, _gui_common.py, test_*.py)
  launcher/                                 źródło exe i ikona
  sample_images/                            zdjęcia testowe
  cache_uczenia/, settings.json, screenshots/, run.log    dane robocze (poza gitem)
```

- Nazwy plików, modułów i klas są angielskie (np. `zakladka_laserow.ZakladkaLaserow` → `lasers_tab/tab.py`, `LasersTab`;
  `lasery_dane.Historia` → `compute/history.py`, `History`; `ogniskowa` → `compute/focal_trap.py`). Funkcje, zmienne, komentarze
  i napisy w GUI zostały po polsku.
- Ustawienia: klucze to nazwy atrybutów (np. `zakladka_laserow.cb_og_G`), więc dawne ustawienia działają; plik
  `ustawienia.json` jest przy pierwszym starcie przenoszony do `settings.json`. Cache (`cache_uczenia/`, nazwy plików npz)
  bez zmian (wskazują na niego junction-y starego edytora w `../app/cache_uczenia`).

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
  kursorze (`settings_panel.SettingsPopup`). To te same sekcje co w panelu (wypożyczone na czas otwarcia, w panelu stoi
  zastępnik), więc wszystko działa i zapisuje się tak samo. Okienko ustawień jest od razu osobnym oknem (zostaje otwarte,
  swobodnie przesuwane, drugi prawy klik w to samo okienko tylko je przywołuje). Grupy ustawień zwija i rozwija klik w ich
  nagłówek (▾ / ▸); ⊟ zwija wszystkie, ⊞ rozwija wszystkie (te same przyciski pod listą stron panelu). Przyciski: ⇤ pokazuje
  sekcję w panelu, ⧉ odpina okienko, ⛶ maksymalizuje, 🖵 pełny ekran, 🗙 zamyka okienko. Sfera dostaje też wybór
  cząsteczki i reflektora, wykresy 6–10 sekcję „Dane”, okienka R — swoje podsekcje rzutów.
- **Menu „Okienka”**: okienka pogrupowane (ruch i mapy, sfera i pułapka, rzuty), „Układy okienek”: domyślny, sfera +
  pułapka obok siebie, sfera + rzuty R1–R7 obok siebie, sfera na całe okno; Ctrl+Shift+1..3 zapamiętuje bieżący układ,
  Ctrl+1..3 go przywraca (sloty w `settings.json`).
- **Ustawienia**: stan kontrolek i układ okienek w `settings.json` (zapis przy zamknięciu, menu „Ustawienia”). Ustawienia
  dawnej zakładki przeniesiono z `../app/settings.json` (bez układu okienek, który ma nową postać).

## Etykiety: narzędzie i obszary oka (2026-10-22)

Strona „Dane, etykiety”, sekcja „Etykiety”. Które z 600 patchy są narzędziem (`history.History.ustaw_etykiete`,
`tool_labels.TRYBY`, `maski(..., tryb=...)`):

- **obrys: każda łata dotknięta (> 0 %)** — domyślnie: całe narzędzie, także patche w większości z tłem;
- **obrys: pokrycie ≥ 50 % (jak dawniej, do 2026-10-21)** — patch, w którym co najmniej połowa pikseli to narzędzie
  (przy cienkim cystotomie 2–7 patchy);
- **obrys: pokrycie ≥ własny próg** — to samo z progiem z pola „próg pokrycia” (aktywne tylko w tym trybie);
- **obrys: środek łaty w obrysie**; **ramka GT: środek łaty w ramce** (dawna etykieta sprzed obrysu, dużo tła).

Przykłady (patche narzędzia: dotknięte / pokrycie ≥ 0,5 / środek / ramka): TR_0050 kl. 1629: 16 / 3 / 4 / 18;
TR_0077 kl. 1113: 25 / 7 / 8 / 72; TR_0001 kl. 274: 17 / 2 / 5 / 20.

**Podgląd łat na zdjęciu** (`laser_app/ui/label_preview.py`, pod wyborem trybu): siatka 30 × 20 łat na zdjęciu, zielone = narzędzie
w wybranym trybie (jaśniej = większy udział pikseli obrysu), żółta obwódka = mieszane (dotknięte obrysem, ale nie narzędzie
w tym trybie), biała = wybrana cząsteczka; najechanie pokazuje udział, klik wybiera cząsteczkę. Nad podglądem liczby łat
narzędzia we wszystkich trybach (np. TR_0077 kl. 1113: dotknięte 25 · ≥ 50 % 7 · środek 8 · ramka 72).

Etykieta działa we wszystkich miejscach, które czytają `h.narz`: przycisk „narzędzie” (cząsteczka w środku narzędzia),
„Wektory dla: narzędzia”, kolor „narzędzie”, pierścienie w okienkach 1–3, okienko 2 (narzędzie/reszta), podkroki.
Rzuty R1–R12 używają etykiety z chwili liczenia stanów 192D (R9 ma własny próg).

**Obszary oka** (źrenica, rogówka; ViT nie był na nich uczony): z obrysów `pupil_seg` i `cornea_seg` tej samej klatki
Cataract-LMM (`etykieta_obrys.npz` w katalogu historii, `History.oko`, `History.obszary(prog_oka)`), pierwszeństwo
narzędzie > źrenica > rogówka > tło. Kolor „obszary oka” w „Kolor ścieżek” (sfera) i „Kolor cząsteczek” (okienko P),
„Zaznacz obszar ▾” przy wyborze cząsteczek wpisuje patche obszaru do „+ cząsteczki”.

## Wydajność: GPU i start (2026-10-22)

- **Tylko widoczne okienka pracują.** `LasersTab.widac(k)`: okienko otwarte i (odpięte albo z niepustym widocznym
  obszarem). Okienko w nieaktywnej karcie albo zamknięte nie liczy, nie wysyła buforów na GPU i nie rysuje; po odsłonięciu
  dostaje bieżącą chwilę (`_odslonieto`). Wcześniej `QDockWidget.isVisible()` było prawdą także dla schowanych kart i rzuty R
  liczyły co klatkę (75 ms CPU na klatkę).
- **Całe natężenia Z na karcie.** Z (K × 600 × 768, float16, ~0,7 GB dla 806 checkpointów) trafia do SSBO `ZH` sfery
  porcjami w tle po wczytaniu historii (`_GPUElement.wyslij_porcjami`: wątek czyta z dysku, każda klatka wysyła do ~8 ms).
  Shader sam wybiera parę checkpointów i interpoluje (`unpackHalf2x16`); do czasu wysłania działa dawna ścieżka (para
  checkpointów w buforze `I`). Sprawdzone: odczyt z karty = Z z dysku bit w bit.
- **Bez synchronizacji CPU–GPU po każdym wywołaniu GL**: `OpenGL.ERROR_CHECKING = False` (laser_app/main_window.py, run.py); szukanie błędów
  GL: zmienna środowiskowa `LASERY_GL_DEBUG=1`.
- **Napisy bez przeliczania układu** (`dock_windows.bez_ukladu`): napisy zmieniane co klatkę (pod widokami, licznik czasu, tytuł
  okienka 11, podkroki) leżą w pojemniku bez układu. Przy skalowaniu ekranu 150 % każde przeliczenie układu gubiło piksel na
  zaokrągleniach doków: doki pełzły o 1–4 px na klatkę, a każdy widok OpenGL dostawał resize i nowy bufor ramki.
- **Pomiar** (RTX 3070, 4K/150 %, historia 35 000, 806 checkpointów, tempo stałe): odtwarzanie 50–55 kl./s w układach
  domyślnym, „sfera na całe okno” i „sfera + rzuty” (przed zmianami 16–18, chwilami 1–3), z pułapką 11 ok. 40 kl./s (przed
  11–21); `ustaw_czas` 0,7 ms (przed 75 ms, p90 950 ms); sfera z 460 800 wiązkami ~5 ms na GPU.
- **Start jak ostatnio**: zapisany układ okienek i położenie/rozmiar okna (`settings.json`, Plik → Zapisz albo autozapis
  przy zamknięciu) stosują się przed pierwszym pokazaniem okna, więc startują tylko okienka, które były otwarte i widoczne;
  widoki w zamkniętych i schowanych w kartach okienkach nie tworzą kontekstu OpenGL. Manifest zbioru uczącego (35 000
  wierszy) z pickla w `cache_uczenia/lasery` (~1 s mniej przy starcie).
- **Menu Plik**: policz/wczytaj historię (Ctrl+O), przerwij liczenie (Ctrl+.), inny katalog wag, zdjęcie z pliku, zapis
  ustawień i układu (Ctrl+S), zapis jako (Ctrl+Shift+S), wczytanie ustawień, obraz okienka pod kursorem do PNG (Ctrl+E,
  folder `screenshots/`), Zamknij (Ctrl+Q). Menu Ustawienia: autozapis przy zamknięciu, przywróć domyślne, stereo 3D.

## Kilka zdjęć naraz na sferze (2026-10-22)

Strona „Kilka zdjęć”: zestaw zdjęć tej samej historii wag (te same checkpointy i okno), domyślny limit 5, najwyżej 32 (jak
paczka uczenia). „+ bieżące zdjęcie” dodaje zdjęcie wybrane w „Dane”, „Policz / wczytaj zestaw” wczytuje historie z cache albo
liczy brakujące po kolei (`ImageSetWorker`; minuty na zdjęcie przy pierwszym liczeniu). Sfera reflektorów pokazuje wtedy
600 koloidów każdego zdjęcia (`history.ImageSetHistory`: położenia sklejone, każde zdjęcie z własnym środkiem chmury;
rzut 3D P3 i reflektory zależą tylko od wag, więc są wspólne — sprawdzone: `dirs` identyczne dla różnych zdjęć) i
600·n·768 wiązek (shader sfery ma `u_T` zamiast stałych 600). Pozostałe okienka (1–13, R, podkroki) i wybór cząsteczki
dotyczą **aktywnego zdjęcia** (lista „Aktywne zdjęcie”; klik w koloid innego zdjęcia na sferze też je przełącza).
Natężenia Z zestawu trafiają na kartę w całości do 3 GB (`gpu.base.LIMIT_ZH`, ok. 4 zdjęcia); większy zestaw wysyła przy
każdej zmianie checkpointu tylko parę checkpointów. Zmiana aktywnego zdjęcia nie wysyła Z ponownie.
Pomiar: 3 zdjęcia (1800 koloidów, 1,38 mln wiązek, Z 2,1 GB na karcie) ok. 32 kl./s przy odtwarzaniu.
Lista zdjęć zestawu zapisuje się w `settings.json` (liczy się dopiero na żądanie).

## Pułapka optyczna: obrys w tle, liczenie tylko widocznego

- Obrys ogniska 3D (`focal_field_gl.siatka_obrysu`) liczy się w wątku w tle (`_OutlineWorker`, zawsze ostatnie zlecenie): objętość
  światła 33³ na GPU (PyTorch), izopowierzchnia, warstwice; klatka rysuje ostatni gotowy obrys przesunięty do bieżącego
  położenia koloidu. Wcześniej liczył się w każdej klatce i odtwarzanie z obrysem traciło płynność.
- Obrys, światło ogniska i wkłady działają też przy wyłączonych liniach wiązek („768 wiązek zbiegających się…” steruje teraz
  tylko liniami wiązek i wiązkami od reflektorów).
- Pułapka (liczenie dla całej historii, potem z pliku) startuje dopiero, gdy okienko 11 albo pułapka na sferze jest widoczna.
- Między checkpointami pułapka jest interpolowana na GPU (pole, kolory, wiązki, koloid: ułamek kroku), więc odtwarzanie
  jest płynne; powolne było tylko pierwsze liczenie z wag (i dawniej obrys w każdej klatce).
- Liczenie przyspieszone (`focal_trap.policz_wiele`): jakobian 3 × 3 w metodzie Newtona liczony analitycznie
  (`focal_trap._jakobian`: pochodna LayerNorm, GELU i rzutu; zgodny z autogradem do 1e-10, test `TestJakobianPulapki`)
  zamiast `torch.autograd.functional.jacobian`. Siatka 17, RTX 3070: 68 ms na checkpoint (wcześniej 100 ms; pierwsze
  uruchomienie z rozgrzewką karty dłużej), ok. 1 min na 806 checkpointów. Wyniki identyczne ze starą wersją (różnice < 1e-3).
- **Newton dla wielu cząsteczek naraz na GPU** (`focal_trap.newton_wsadowo`, `checkpointy`, `_Block6Params`): wspólna część
  (wagi z wyprzedzeniem, przejście sieci, rzut) to generator `checkpointy`; Newton liczy wszystkie cząsteczki jednocześnie
  w maskach (B × 3), jakobian wsadowo (`_jakobian_wsadowo`), układy 3 × 3 wzorem z dopełnień (`rozwiaz3`, bez SVD), a
  cząsteczki, które się zbiegły, wyszły z komórki albo krążą bez zera (po 15 iteracjach |f| > 1e-3 |f(0)|), przestają być
  liczone. Pomiar na 30 checkpointach × 600 cząsteczek: 8 iteracji daje te same zera pola co 60.
- **Dokładność** (lista „Dokładność” w sekcji pułapki): float32 (domyślnie) liczy jakobian, siatkę pola i kolory w float32,
  a resztę f w Newtonie w float64 (w pułapce pchnięcia 768 wiązek się znoszą, więc f to mała różnica dużych liczb). Na
  kartach GeForce float64 jest kilkadziesiąt razy wolniejszy. Porównanie na 12 000 parach checkpoint × cząsteczka: te same
  zera pola co float64, położenia zer identyczne, wartości własne zgodne do 2·10⁻⁵; pole i tak jest zapisywane w float16.
- **Gotowe (bez liczenia)**: lista pułapek już policzonych dla tej historii (pliki `ogniskowa_t…npz`); wybór ustawia
  cząsteczkę, siatkę i zasięg i wczytuje pułapkę z dysku od razu.
- **Licz razem z** (sekcja pułapki): dodatkowe cząsteczki (do 8) liczone w tym samym przejściu: wagi i przejście sieci są
  wspólne, osobno tylko komórka każdej cząsteczki. 3 cząsteczki: 73 ms na checkpoint (1 cząsteczka 68 ms; ok. 30 ms
  to część wspólna: wczytanie wag, przejście sieci). Każda trafia do
  własnego pliku `ogniskowa_t<tok>_…npz`, więc przełączenie „Cząsteczka” albo „▾ policzone” pokazuje ją od razu; wybór
  cząsteczki liczonej właśnie razem z inną nie przerywa liczenia.

## Mapa pułapek: wszystkie 600 cząsteczek (okienko 14)

`laser_app/compute/trap_map.py`, strona panelu „Mapa pułapek (14)”. Dla każdej cząsteczki i każdego checkpointu (bez siatki pola G³,
ta zostaje w okienku 11): skok w bloku 6, zero pola w komórce (Newton wsadowo), wartości własne jakobianu w cząsteczce (λ₀)
i w zerze (λ), liniowe oszacowanie zera, r_kom. Cała historia (806 checkpointów × 600 cząsteczek): ok. 104 s na RTX 3070
(86 ms na checkpoint), wynik `mapa_pulapek_z<zasięg>.npz` (46 MB) obok historii, potem z pliku.

- Kolory: kształt pola w cząsteczce (λ₀: zielony ściąga, żółty siodło, czerwony wypycha, domieszka błękitu = wir), charakter
  zera w komórce, siła skoku, odległość zera od cząsteczki / r_kom; „Wyróżnij” przyciemnia pozostałe (siodła, wiry…).
- Linie od cząsteczek do zer pola w komórce (lupa), wektory skoku, wybrana cząsteczka w białej obwódce; położenia, odcinki i
  kolory interpolowane między checkpointami. Ta sama mapa opcjonalnie na dużej sferze.
- Strzałki pola w całej chmurze (od 2026-10-23, mapa w wersji 2 zapisuje jakobian J₀ w cząsteczce): w komórce każdej z 600
  cząsteczek siatka G³ strzałek (1–5 na bok) pola liniowego f(0) + J₀ u, bez ponownego liczenia sieci. „Kształt pułapki”
  pokazuje samo J₀ u (zbiega się = ściąga, rozbiega = wypycha, siodło, wir), „pole wiązek” pełne pchnięcie. Długość ∝ |f|
  (wspólna skala w checkpoincie) albo jednakowa; kolor strzałek = kolor cząsteczki, więc „Wyróżnij” przygasza też strzałki.
  Starsza mapa (wersja 1) liczy się ponownie przy pierwszym pokazaniu okienka 14.
- Co widać na historii `2cca2743e1c74b14`: zero pola wewnątrz komórki ma tylko 0,005 % par checkpoint × cząsteczka, więc
  informacją jest kształt pola w samej cząsteczce. W checkpoincie 101 prawie wszystkie cząsteczki siedzą w siodle (585 z
  600, 591 z wirem), w checkpoincie 701 większość pól już ściąga (481 ściąga, 119 siodło, 376 z wirem).

## Obliczenia: eksport, import, pamięć na dysku (2026-10-22)

Długie obliczenia liczy się raz; potem aplikacja je wczytuje, a plik eksportu przenosi je na inny komputer albo do innej
ścieżki wag. Okno **Plik → Obliczenia: eksport i import…** (Ctrl+Shift+E; też przyciski „Obliczenia i przebiegi…” przy
wyborze cząsteczek i „⇅ plik…” przy gotowych pułapkach):

- lista obliczeń bieżącej historii z rozmiarem i datą (`compute/exports.lista`): historia uczenia, etykieta, rzuty 192D,
  przejścia przez sieć, pułapki cząsteczek, mapa pułapek, pomiary w podkrokach, pamięć obliczeń na żądanie; stan wybranej
  cząsteczki („Cząsteczka 226: pułapka…; pomiar…; pole w podkrokach: 35 checkpointów”);
- **Eksportuj zaznaczone…**: zip z `manifest.json` (klucz historii, opis, pliki z SHA256), opcjonalnie z przebiegami;
  „Zaznacz obliczenia cząsteczek” zaznacza wszystko, co jest policzone dla cząsteczek z pola „Cząsteczki”;
- **Wczytaj obliczenia z pliku…**: pliki trafiają do historii o tym samym kluczu albo do historii o tych samych
  checkpointach, zdjęciu, ramce i biegu uczenia na innym dysku (klucz przenośny `history.klucz_przenosny`: wersja danych,
  dwa ostatnie człony ścieżki wag, kroki, okno, ramka GT, zdjęcie); sumy SHA256 sprawdzane, istniejące pliki zostają
  (chyba że „nadpisz”); plik z samą historią tworzy ją w cache, plik z samymi obliczeniami bez historii jest odrzucany;
- **Przebiegi → CSV…**: położenia cząsteczek na sferze przez wszystkie checkpointy (`exports.przebiegi`: krok, epoka,
  cząsteczka, rzut P3, ekran sfery, r99, narzędzie) i NPZ obok.

Pamięć na dysku obliczeń liczonych na żądanie (`compute/disk_cache.py`, `<historia>/pamiec/`): pole w podkrokach (okienko 11),
dolina (12) i mechanika (13) dla danego checkpointu i parametrów liczą się raz; ten sam checkpoint potem z pliku (dolina
6,4 s → 0,13 s). Zapis bez pickle: struktura wyniku w JSON, tablice w npz. Historia liczona na innym dysku (F: zamiast W:)
jest znajdowana po kluczu przenośnym i wczytywana zamiast liczona od nowa. „Policz pułapkę w podkrokach…” pyta, gdy pomiar
już jest, zamiast liczyć go od nowa.

## Pomiar pułapki w podkrokach: szybciej

`trap_measurements.policz_historie`: wagi wczytywane raz na checkpoint, z wyprzedzeniem w 4 wątkach (wcześniej dwa razy,
osobno w `history.podkroki` i `substep_field.policz`), bez obrazków łodygi (`podkroki(..., lodyga=False)`: 6 rozkładów
SVD, których pomiar nie używa). 184 → 89 ms na checkpoint (ok. 1,2 min na 806 checkpointów zamiast 2,5).

## Okienko P: widoczność wiązek

Ustawienia okienka P (strona „Podkroki” i prawy klik w okienko), grupa „Widoczność: wszystkie wiązki naraz”: pole „wszystkie
wiązki i linie” chowa naraz linie i wektory uwagi, lasery MLP, przesunięcia w podkroku i tory, a pojedyncze przełączniki zostają
bez zmian. Przyciski: Włącz wszystkie, Wyłącz wszystkie, Tylko uwaga, Tylko lasery, Tylko ruch. Osobno: cząsteczki, wybrana
cząsteczka, grubość linii; uwaga: linie / wektory sił, jedna głowa albo wszystkie, próg atencji; lasery: znak (dodatnie /
ujemne), tylko N najsilniejszych (próg z |z| N-tego lasera, `NetworkPassageGPU._prog_laserow`), próg |z|, krycie;
przesunięcia w podkroku. Tryb sił uwagi (`cb_sily`) jest wspólny ze sferą w trybie podkroków; pozostałe przełączniki
dotyczą tylko okienka P.

## Okienko M: mikrosoczewki (2026-10-23)

Druga wersja wizualizacji przejścia przez sieć, według alegorii z rozmowy 2026-10-23 (`gpu/microlens_gl.py`, `lasers_tab/microlens_view.py`,
menu „Okienka → Sfera, pułapka, …, mikrosoczewki”, strona panelu „Mikrosoczewki (M)”, prawy klik w okienko). Jedna scena 3D w trzech
piętrach, wzdłuż osi Z:

- **na dole zdjęcie** (klatka 480×320) i cztery **płyty łodygi** (wyjścia konwolucji jako obrazy PCA-RGB, z przejścia `lodyga_0..3`):
  zdjęcie nie przechodzi na ekran, lecz przez filtry łodygi **naświetla** półfabrykaty 600 mikrosoczewek (stan 0 = łodyga + kodowanie
  pozycji); promienie naświetlenia biegną od komórki zdjęcia do jej koloidu w sferze (kolor = kolor łaty; narzędzie, wybrane albo wszystkie);
- **w środku sfera reflektorów** (drugi `ReflectorSphere` o tych samych ustawieniach co duża sfera, zawsze w trybie przejścia przez sieć):
  koloid = kształt mikrosoczewki (położenie w 192 wymiarach), lasery bloku i linie uwagi (gwiazda) go formują przez 12 podkroków;
- **na górze lampa, matryca mikrosoczewek i ekran**: jednostajna lampa prześwietla matrycę 30 × 20 (kółko = soczewka, jasność =
  transmisja = sigmoid ciepła z głowy); każda soczewka świeci tylko na własną komórkę ekranu (głowa działa na token, bez mieszania);
  ekran = cień (mapa ciepła), różnica cień − kalka (czerwony za jasno, niebieski za ciemno; to prowadzi uczenie) albo kalka z etykiety
  (Gauss wokół środka narzędzia jak w `model.cele`); ramki: błękitna z głowy (prostokąt plamki najjaśniejszej soczewki: komórka
  szczytu i jej l, t, r, b), biała GT; żółty obrys = wybrane cząsteczki (na matrycy i na ekranie).

Czas okienka = podkrok 0..12 bieżącego checkpointu (suwak podkroków, ▶ podkroki, synchronizacja z osią uczenia), dane = to samo
przejście co okienko P (`history.podkroki`, zlecane także gdy widać okienko M). Przejście niesie teraz klucz `glowa` (13 × 600 × 5:
głowa na każdym z 13 stanów); zapisane wcześniej przejścia bez tego klucza liczą się od nowa (`passage_store.wczytaj`). Domyślnie
ekran pokazuje **sondę**: tę samą głowę przyłożoną do stanu po bieżącym podkroku (jak cień powstaje w głąb sieci; sieć tak nie robi,
czyta tylko stan 12; napis pod widokiem to zaznacza); opcja „głowa czyta tylko stan końcowy (jak sieć)”. Napis pod widokiem: maks.
ciepło i jego komórka, mediana tła, liczba jasnych komórek, IoU ramki szczytu z GT.

**Szkło jako maź formowana polem z jakobianu; ognisko u\* usunięte (2026-10-24).** „Wygląd szkła” w sekcji Mikrosoczewki: szklane
kulki albo (domyślnie) półprzezroczysta maź skupiająca światło. Przejście niesie klucz `jak_n` (6 × 600 × 3 × 3):
`history.jakobiany_mlp` liczy dokładnie z wag (LayerNorm, fc1, GELU′, fc2) jakobian pola MLP każdego bloku w każdym koloidzie
w rzucie `_n`, J = P·∂MLP/∂x·Pᵀ (zgodność z autogradem 6·10⁻⁷; zapisy bez `jak_n` liczą się od nowa). `MicrolensMixin.pole_z_jakobianu`
daje bufor JK (600 × 3 × 4): część symetryczna S (siodło: osie rozciągania λ > 0 i ściskania λ < 0), oś wiru ω z części
antysymetrycznej, |F| względne. Maź (tryb 16, drobiny w FBO gęstości → powierzchnia z gradientu → refrakcja ekranowa światła HDR,
`FS_MAZ`) i soczewki matrycy (tryb 4) są odkształcane S i obracane ω·czas; ruch w podkroku rozciąga maź i zostawia ogon.

Pierwsza wersja rysowała też ognisko pułapki u\* = −J⁻¹F z typem (ściąga / siodło / wypycha / zero poza komórką) jako pierścień
na soczewce i ściągała do niego wnętrze mazi. **Usunięte po pomiarze** na ostatnim checkpoincie historii `90f58d4f8b263d8b`
(k65640; skrypt `diagnostyki_rzutu.py` z sesji 2026-10-24, pełny jakobian 192 × 192 każdego koloidu, U = Pᵀ, Q = I − UUᵀ), blok 4,
mediany po 600 koloidach: widoczna część pola ‖PF‖/‖F‖ = 0,41 (tokeny narzędzia 0,73), defekt niezmienniczości ‖QJU‖/‖JU‖ = 0,80,
defekt zamknięcia ‖PJQ‖/‖PJ‖ = 0,93, ‖PJPᵀ‖/‖J‖ = 0,19 (losowe 3 kierunki: 0,016). Zero zlinearyzowanego rzutu u\* leży
median 55 promieni komórki od koloidu (w programie wszystkie 600 obrysów miało typ „zero poza komórką”), zero pełnego pola
192-D median 450 komórek, a kierunki obu są nieskorelowane (cos median −0,08). W 192 wymiarach każdy koloid jest siodłem
(ok. 92 dodatnie i 98 ujemnych wartości własnych, wszystkie z parą zespoloną), a liniowość w komórce jest dobra (błąd 2 %):
na koloid działa prawie jednorodny prąd laserów z lekkim ścinaniem, nie pułapka. S i ω zostają: są dokładnymi kompresjami
pełnych S₁₉₂ i Ω₁₉₂ (wartości własne S przeplatają się z pełnymi), choć 80 % odpowiedzi ucieka z rzutu. Literatura
(Embree 2017 o niestabilnych modach kompresji, Carden–Embree 2012 o wartościach Ritza macierzy nienormalnych, Horstmeyer–Atay 2016
o lumpability) mówi to samo: kompresja P J Pᵀ nie klasyfikuje punktów stałych pełnego pola.

**Kropla w ścinaniu, skupianie z kontrakcji uwagi, jasność promieni (2026-10-26).** Obraz okienka M nie sugeruje już fizyki,
której pomiary nie potwierdziły. (1) Druga pozycja „Wygląd szkła” to teraz „kropla w ścinaniu (maź odkształcana polem laserów,
światło skupiane kontrakcją uwagi)”, nagłówek grupy „Szkło: kropla w ścinaniu albo kulki, drobiny super szkła”: maź rysuje kroplę
w przepływie liniowym (Taylor 1934, model elipsoidalny Maffettone–Minale 1998), rozciąganą wzdłuż osi rozciągania części
symetrycznej S jakobianu pola MLP, ściskaną wzdłuż osi ściskania i obracaną wirowością ω (kształt z S, ω w JK bez zmian w kodzie).
Pomiar z 24.10.2026 (`experiments/schur_qr_jakobiany_mlp_20261024__10-03-40`): 86 % energii S i 91 % energii ω pochodzi z części
nienormalnej jakobianu (‖N‖/‖J‖ ≈ 0,93 wobec 0,71 losowo), więc to ścinanie, nie wir ani soczewka, a pole nie ma zera w pobliżu
koloidów (brak pułapki). (2) Nowa lista `cb_mk_skup` (opcja sceny `skup_z`, uniform `u_skup_z`): „skupianie mazi: kontrakcja uwagi
(zbliżanie do sąsiadów w półkroku uwagi tego bloku)” (domyślna) albo „skupianie mazi: ciepło głowy (odczyt, jak dotąd)”.
Kontrakcja `MicrolensMixin.kontrakcja_uwagi(xa, xb, k=5, sfera=True, wzgledem_chmury=True)`: xa = `xn[2b]`, xb = `xn[2b+1]`
(pełne stany 192-D przed i po półkroku uwagi bloku b), na sferze chmury (środkowane po tokenach i znormowane per token, więc
liczy się kierunek, a jednorodne skalowanie daje κ = 0); dla każdego tokenu 5 najbliższych sąsiadów w xa, d_przed = średnia
odległość do nich w xa, d_po = do tych samych sąsiadów w xb, κ = 1 − d_po/d_przed, a „względem chmury” dzieli d_po/d_przed
przez średni stosunek wszystkich tokenów. Lista ma trzy pozycje: względem chmury (domyślna), bezwzględna, ciepło głowy.
Powód odniesienia do chmury (k65640, mediany narzędzie / tło, κ bezwzględna na sferze): blok 1 +0,020 / +0,017, blok 2
+0,081 / +0,007, blok 3 +0,021 / +0,002, blok 4 −0,010 / −0,030, blok 5 −0,005 / −0,002, blok 6 −0,020 / −0,005: w blokach
1–3 sąsiedztwo narzędzia ściska się mocniej niż tła, w bloku 4 wszystkie sąsiedztwa się rozchodzą (narzędzia najmniej:
względem chmury +0,025 / +0,006), więc bez odniesienia maź w bloku 4 byłaby cała ciemna. Pierwsza wersja liczyła κ na
stanach `xn` bez normowania per token i dawała w bloku 4 narzędzie −0,004 wobec tła −0,0002 (ruch radialny mieszał się
ze stycznym). `pole_z_jakobianu(J, F, kappa)` wpisuje κ/κ₉₅ (95. percentyl |κ|, obcięte do [−1,5; 1,5]) do `JK[:, 2, 2]` (bez `xn` w starych
zapisach: 0); przy mieszaniu checkpointów κ obu miesza się ułamkiem f. W trybie kontrakcji drobina mazi (tryb 16) ma jasność
i krycie z clamp(κ, 0, 1,5) zamiast ciepła, przy κ < 0 jest szersza (maź ciemniejsza i rozmyta), a `FS_MAZ` mnoży skupianie przez
średnie clamp(κ, 0, 1,5) drobin w pikselu (drugie wyjście FBO mazi, `COLOR_ATTACHMENT1`). Uzasadnienie: pomiar z 25.10.2026
(`experiments/kontrakcja_komutator_polkrokow_20261025__02-12-46`) pokazał, że chmurę ściskają półkroki uwagi (ruch styczny do sfery
97–99 %), a separację klas robi MLP; skupianie narzędzia bierze się ze ściskania przez uwagę, a ciepło jest tylko odczytem głowy.
(3) Nowa opcja `ch_mk_jas_zdj` „jasność promieni ze zdjęcia (tylko ilustracja: sieć nie przenosi jasności łat)” (opcja `jas_zdj`,
uniform `u_jas_zdj`), domyślnie wyłączona: promienie naświetlenia (tryb 2, wszystkie cztery warianty) mają barwę łaty o stałej
luminancji 0,5 (`jas_laty` w `VS_M`), dalej × jasność, krycie, wzmocnienie i ciepło jak dotąd; włączona przywraca jasność ze
zdjęcia. Luminancja łaty nie ma odpowiednika w sieci: łata wchodzi do konwolucji łodygi, nie do szkła.

Okienko M, 2026-10-26: (1) własna siatka szklanej sfery („siatka szklanej sfery”, `sp_mk_siatka`, domyślnie krycie 0,3, opcja
`sfera` w `_mk_opcje_sfery`), niezależna od dużej sfery i widoczna także przy wyłączonej sferze reflektorów (`tylko_siatka`:
`ReflectorSphere._rysuj` rysuje wtedy samą siatkę). (2) Synchronizacja kroków i podkroków: w czasie powrotu (wyjście sieci
checkpointu → wejście następnego) okienko M jedzie razem z dużą sferą (`_mk_start_przejscia`, `_mk_ustaw_przejscie`): koloidy
po prostej, ciepło głowy ze stanu 12 w stan 0 następnego (sonda: w stan 12), wiązki zgaszone; wcześniej stało na stanie 12
i skakało. (3) Płynne podkroki: pole laserów formujące soczewki i maź (`_mk_pole`) w podkroku uwagi bloku b przechodzi
smoothstepem z pola bloku MLP b − 1 w pole bloku b (`bloki_pola`; F, jakobian i κ mieszane), zamiast przeskoku w połowie
podkroku; przy „osobnych podkrokach” jak dawniej. Pomiar: maks. zmiana pola na 0,05 podkroku 0,054 (mediana 0,003), skok
koloidu na granicach powrotu ≤ 0,8 przy medianie ruchu 1,4 na klatkę.

## Pełny ekran i Odyssey 3D Hub (konwersja SBS → 3D)

Hub włącza konwersję, gdy program przejdzie w pełny ekran (okno równe monitorowi; wtedy pokazuje okienko wyboru SBS), a potem
jego okno musi leżeć nad tym programem. Okno OpenGL, którego obszar rysowania pokrywa się z monitorem co do piksela, sterownik
bierze za pełny ekran na wyłączność (SHQueryUserNotificationState = 3, Direct3D full screen) i nic się nad nim nie położy
(YouTube w przeglądarce działa, bo przeglądarka składa obraz przez DWM). Okno większe od ekranu nie jest na wyłączność, ale Hub
nie uznaje go za pełny ekran i się nie włącza (sprawdzone przez użytkownika). Dlatego pełny ekran okienka (🖵, F11) jest
dokładnie na cały monitor, a ma styl WS_BORDER: ramka 1 px zmniejsza obszar rysowania o 1 px z każdej strony (3838 × 2158 na
3840 × 2160), jak Qt-owe setHasBorderInFullScreen (PyQt6 go nie udostępnia). Na czas pełnego ekranu okno traci
FramelessWindowHint, bo w oknie bez ramki Qt nie oddaje ramki z obszaru klienta. Sekcja Stereo 3D: pole „pełny ekran pozwala
nakładkom” (wyłączone = klasyczny pełny ekran bez ramki). Po wyjściu z pełnego ekranu przypięte okienko wraca z kartą na
wierzchu (wcześniej w grupie kart zostawało pod inną kartą). W pełnym ekranie znikają napisy pod widokami (wracają po wyjściu). Próba: `.sessions/Session_2026-10-22_172605/tools/proba_pelnego_ekranu.py 0|1`.

## R12 kNN 3D: bez skoków

R12 liczył PCA3 od nowa w każdym checkpoincie (znaki i kolejność osi przeskakiwały) i brał najbliższy checkpoint. Teraz
(`projections.baza_pca3_wspolna_t`, `baza_pca3_ckpt_t`, `obroc_do_t`, `knn_t`): baza PCA wspólna dla całej historii
(domyślnie), w każdym checkpoincie wyrównana Procrustesem 3 × 3 do wspólnej (z wygładzaniem ± n checkpointów) albo surowa
(dawna); położenia i kolory interpolowane między checkpointami; metryka euklidesowa albo kosinusowa; ułamek sąsiadów
wygładzany ± n checkpointów. Mediana przesunięcia tokenu na krok checkpointu: 0,68–1,03 → 0,14–0,20 promienia chmury.

## Okna, dymki, grupy parametrów

- Odpięte okienka (panel F2, podkroki F3, widoki) i okienka ustawień z prawego kliku to zwykłe okna, nie okna narzędziowe Qt: nie
  leżą zawsze nad oknem głównym i nad sobą, można je schować pod inne i przełączać Alt+Tab (`dock_windows._niezalezne`).
- `laser_app/ui/tooltips.py`: każdy parametr ma dymek z objaśnieniem (co to jest i co zmienia); ten sam dymek ma nazwa wiersza w panelu.
- `laser_app/ui/parameter_groups.py`: nagłówki podgrup w długich sekcjach (pułapka: liczenie, wiązki, światło, obrys, wkłady, mgła pola,
  strzałki pola, oznaczenia, ślad i kadr; sfera, dane, dolina, podkroki, zapis).
- Zwinięcie i rozwinięcie całej sekcji pokazuje z powrotem tylko to, co było widoczne przed zwinięciem: wiersze zwiniętych
  grup i opcje ukryte przez program zostają ukryte (wcześniej nakładały się na nagłówki grup).
- Okienka sterujące (panel parametrów, podkroki) mają zawartość w przewijanym polu (`dock_windows.ScrollWrapper`): po włączeniu
  podkroków pasek podkroków (dwa wiersze opcji i schemat sieci, ponad 1100 px szerokości) nie rozpycha już okna i nie wchodzi
  na panel parametrów ani na widoki; gdy brakuje miejsca, przewija się. Wysokość okienka podkroków dopasowuje się do treści.

## Poprawki działania (2026-10-22)

- Historie w cache liczone, gdy projekt był na `F:` (ścieżka wag `F:\...\SmallViTAblation\assets\...`): pułapka optyczna,
  obrys ogniska 3D, dolina, mechanika i podkroki kończyły się błędem `PermissionError`. `history.sciezka_lokalna`
  przenosi ścieżkę do bieżącej bazy projektu (tak jak już robiły to rzuty).
- Obrys ogniska 3D i brzeg doliny wymagały scikit-image (brak w Pythonie 3.11): zapas `pyqtgraph.isosurface`.

## Pliki i dane

- Kod: pakiet `laser_app/` (opis w „Struktura” wyżej). Obliczenia bez Qt w `compute/`: `history.py` (historia uczenia,
  `History`), `focal_trap.py` (pułapka optyczna, Newton wsadowo), `trap_map.py` (mapa pułapek), `substep_field.py` i
  `trap_measurements.py` (pole i pomiary w podkrokach), `passage_store.py` (zapis przejść), `projections.py` (rzuty 192D),
  `tool_labels.py` (etykieta narzędzia z obrysu, ramka GT, adnotacje Cataract-LMM), `valley.py`, `reflector_mechanics.py`,
  `reflector_drawing.py`, `checkpoint_histories.py` (lista historii), `common.py` (stałe, kolory kanałów, izopowierzchnia,
  `Interrupted`). Rysowanie w `gpu/`, widżety w `ui/`, zakładka w `lasers_tab/`.
- Refaktor 2026-10-22 usunął kod po edytorze latentu, którego ta aplikacja nie używała (detektor, silnik SD, odczyt filmów MP4,
  układ przepływowy, panele opcji, widok 3D makrowektorów; ok. 1300 wierszy).
- Cache: `cache_uczenia/lasery/<skrót>/` (dane.npz, Z.npy, rzuty/, przejscia/, etykieta_obrys.npz) i `cache_uczenia/etykiety/`.
  Przeniesione z `../app/cache_uczenia/`; tam zostały połączenia katalogów (junction) do tych samych danych, więc stare
  skrypty eksperymentów z ścieżką `app/cache_uczenia/lasery` działają dalej.
- `../app/lasery_dane.py`, `../app/rzuty_dane.py`, `../app/etykiety_narzedzia.py` (stare kopie) zostały w edytorze bez zmian jako biblioteka
  historycznych eksperymentów (`experiments/wykrywanie_grupy_narzedzia_bez_etykiet_*`, `wspoldzialanie_neuronow_*` itd.).

Poniżej opis zakładki przeniesiony z `../app/README.md` (odwołania do „lewego panelu” = dziś strony panelu „Parametry”).

## Opis zakładki „Lasery i koloidy” (2026-10-12; `laser_app/lasers_tab/tab.py`, `laser_app/compute/history.py`, pakiet `laser_app/gpu/`)

Interaktywna wersja filmów 2, 12 i 16 z `outputs/porownanie_szklo_20261010/film_krysztal` (zasady w `ZASADY_LASER_KOLOID.md`).

**Dane.** Wybierasz:
- historię checkpointów (`checkpoint_histories.HISTORIE` albo dowolny katalog `k<krok>.pt`);
- zakres kroków i co który checkpoint;
- zdjęcie: „Film” (117 filmów zbioru uczącego biegu 35 000) i „Zdjęcie” (jego klatki), albo grupa „Pliki testowe”
  (TRAIN 0 z filmów, `sample_images/`, plik).

Klatka zbioru uczącego (2026-10-13) jest czytana jak w `trening35000.py`: z wiersza `MANIFEST_35000.jsonl`, z archiwum zip
(SHA-256 bajtów sprawdzane), rozciągnięta do 480 × 320 INTER_AREA, ramka z manifestu skalowana osobno w X i Y. Uwaga: 15 z 18
plików `sample_images/` (w tym całe TR_0077, TR_0127, TR_0140) NIE należy do zbioru uczącego; opis ramki to mówi.
Ramka prawdziwa dla TRAIN 0 pochodzi z `atencja_train0.npz`, dla pozostałych plików `TR_*` z adnotacji.

**Liczenie.** „Policz / wczytaj” uruchamia `history.policz` w wątku, na GPU. Dla każdego checkpointu liczy:
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

**Rzuty z 192 wymiarów: okienka R1–R8** (2026-10-21; `laser_app/compute/projections.py`, `laser_app/ui/projections_view.py`, `laser_app/gpu/projections_gl.py`; uzasadnienie w
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
- **Shadery GLSL 4.30 (`projections_gl.ProjectionGPU`, SSBO jak `laser_app/gpu/base.py`).** Cała historia położeń i sił w bazie okienka trafia do
  bufora raz, przy zmianie bazy, środka, skali albo wygładzania. W klatce idzie tylko czas. Interpolację, ślady, groty strzałek
  zwrócone do kamery i krycie liczy shader.
- **Ramy lokalne R4 i R5** to elementy lokalne tego samego shadera, wysyłane raz na checkpoint i cząsteczkę.
- **Rzutowanie 192D → 3D** całej historii liczy torch na karcie (`ProjectionData.wsp`).
Testy: `tests/test_projections.py`.

**Oś czasu.** Suwak, odtwarzanie (spacja) i strzałki ←/→ (checkpoint). Klik w wykres przenosi do danej chwili. Tempo „jak w filmie 16” to v = min(0,8 + krok/250, 150) kroków na klatkę; tempo stałe liczy się w checkpointach na sekundę.

**Wybór cząsteczki.** Klik w cząsteczkę w panelach 1–3 albo na sferze wybiera patch. Przycisk „narzędzie” bierze patch w środku ramki, a „najzmienniejszy” reflektor o największej wariancji natężeń.

**Rysowanie.** Robią je shadery GLSL 4.30 z buforów SSBO, a cała historia leży na karcie. W klatce CPU podaje tylko czas i opcje. Wyjątek to natężenia z_jc pary sąsiednich checkpointów dla wiązek (600 × 768), wysyłane przy zmianie pary.

**Okienko „Przejście przez sieć: podkroki”** (`history.podkroki`, `network_passage_gl.NetworkPassageGPU`, `NetworkDiagram`).

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
- **wygładzanie pułapek** wzdłuż checkpointów: średnia ważona krocząca (wagi okno, okno−1, …, 1; tylko przeszłość), EMA (α na checkpoint) albo obie naraz; dotyczy drogi, wektorów i poświaty pułapek oraz (od 2026-10-23, „wygładzać”) drogi i wektorów koloidów: osobny bufor ścieżek PS, kropki koloidów zostają w prawdziwych położeniach;
- **zanik historii:** jasność odcinka exp(−3·wiek/zanik), gdzie wiek liczony jest w krokach uczenia.

Panel po lewej ma sekcje zwijane polem przy tytule, pionowe przewijanie i przeciągany podział szerokości.

**Kolor „odpływ od adresu”** (kolor ścieżek, plasma jak w okienku 2). Każdy punkt drogi i koniec wektora koloidu albo pułapki ma kolor według odległości od położenia w kroku 0 (adresu) w tej chwili: ciemny fiolet oznacza punkt przy adresie, żółty 99. percentyl odpływu (osobno dla koloidów i pułapek, po wygładzeniu).

**Skalowanie w sferze.** Opcje w sekcji sfery:
- wielkość koloidów (px);
- wielkość pułapek (średnica poświaty, px) i jasność pułapek;
- skala chmury w sferze: mnożnik rozmiaru chmury koloidów i pułapek względem szklanej sfery, 1 jak w filmie 12. Wiązki, punkty załamania, wektory, drogi i wybieranie kliknięciem przeliczają się razem z chmurą; kolor odpływu się nie zmienia.

**Pułapka optyczna: ogniskowa wiązek i okienko 11 (2026-10-14).**

Sekcja „Pułapka optyczna” liczy (`laser_app/compute/focal_trap.py`, w tle, z cache obok historii: `ogniskowa_t<tok>_G<G>_z<zasięg>.npz`)
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
lokalne (zasięg 0,05 r99). Kamera domyślnie śledzi cząsteczkę. Dane wiązek zapisuje `laser_app/compute/focal_trap.py` od wersji 2 (pv, zj, zbar).
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

**Pole tylko w komórce tokenu i podgląd na żądanie (2026-10-15, `laser_app/compute/focal_trap.py` wersja 4).** Lokalne pole wiązek liczy się tylko
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

**Zapis przejść przez sieć na dysk (2026-10-15, `laser_app/compute/passage_store.py`, sekcja „Zapis przejść przez sieć”).** Policzone przejścia
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
- Ścieżki nie zaczynają się od nowa w kolejnym checkpoincie (2026-10-23): droga koloidów na sferze („cała historia”) i tory okienka P zachowują przejścia ostatnich N checkpointów (pole „ścieżki przez”, domyślnie 50; „od nowa” = dawne zachowanie), razem ze stanami powrotu, wygładzane jak ścieżki. Ręczny skok osi czasu albo wyjście z podkroków czyści ślad.

Pomiar: około 45 kl/s, 95% klatek do 38 ms, przy zmianie checkpointu najwyżej około 60 ms.

**R9: zero z ponownym dopasowaniem i walidacja blokowana (na karcie).** Dwie opcje sekcji R9. „Zero z ponownym dopasowaniem osi”
(DiProPerm): dla każdej losowej etykiety oś LDA jest dopasowywana od nowa, więc zero obejmuje przeuczenie 18 punktów w 192D. „Walidacja
blokowana po checkpointach”: pula jest dzielona na ciągłe bloki, oś z pozostałych bloków jest oceniana na checkpoincie wyłączonego bloku
(chroni przed korelacją czasową, nie przed tymi samymi tokenami). Wszystko liczy karta: wsadowe `solve` i `eigh` w torch, miary z
`cdist` na GPU, float32 i TF32 (zgodne z CPU float64 do 0,2° w osiach), ok. 0,5 s na checkpoint wobec 14 s w numpy. Podczas odtwarzania
liczą się tylko szybkie miary podstawowe, ciężkie dopiero na pauzie.

**Etykieta narzędzia z obrysu, nie z ramki (2026-10-21).** `laser_app/compute/tool_labels.py` czyta wielokąty `<narzędzie>_seg` z
annotation.csv Cataract-LMM 3_Object_Tracking i liczy udział pikseli narzędzia w każdej łacie 16 × 16. Historia (`history.Historia`)
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

**Nawigacja myszą w widokach OpenGL (2026-10-22).** Wspólna dla wszystkich okien (`gl_view.GLView`): LPM obraca (w widoku z góry
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
laserów przełącza wszystkie widoki OpenGL (`GLView.ustaw_stereo`) między zwykłym obrazem a half side-by-side: scena jest rysowana
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

Reset i osie (2026-10-23): ⟲ w pasku tytułu każdego okienka z widokiem 3D (też Home / 0, podwójny ŚPM, prawy klik → ⟲) przywraca
kamerę sprzed obracania, przesuwania i przybliżania (zapamiętaną przy pierwszym ręcznym ruchu; kamery ustawianej przez program to
nie cofa). Strzałka przy ⟲, przyciski X…−Z w menu prawego kliku i klawisze X, Y, Z (Shift: z ujemnej strony), O ustawiają kamerę
wzdłuż osi sceny albo izometrycznie (środek i odległość zostają). Wszystkie widoki naraz: „Okienka ▾ → Resetuj wszystkie widoki 3D”
(Ctrl+Shift+R) i „Wszystkie widoki 3D wzdłuż osi X, Y, Z”.
