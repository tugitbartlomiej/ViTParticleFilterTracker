# SESSION SUMMARY: aplikacja latentu ViT, granice latentu, cząsteczki 3D, film i makrowektory

Data: 2026-10-06, od około 12:45 do 17:16. Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`,
gałąź `codex/eden-nano-training-20260914`. Rozmowa prowadzona z katalogu projektu grantowego
`F:\Studia\MCHTR\Grant\Projekt`, ale cała praca dotyczy repozytorium ViT.

## Cel sesji

1. Sprawdzić konwerter latentu ViT k1000 na cechy DINOv2 (wynik sesji `Session_2026-10-06_124141`) na zdjęciach
   i na sztucznym latencie z prezentacji.
2. Wyznaczyć matematycznie zakres (min, max) wartości latentu, jaki może osiągnąć mały ViT w trakcie uczenia,
   bez korzystania z wytrenowanych wag.
3. Rozbudować aplikację `SmallViTAblation/app` do ręcznej i automatycznej edycji latentu oraz wizualizacji.

## Kontekst

- Mały ViT: dim 192, 6 bloków, 3 głowy, siatka 20 x 30 tokenów, patch 16 px, wejście 480 x 320. Latent = aktywacje
  po GELU ukrytej warstwy MLP ostatniego bloku, 600 x 768. Głowa detekcji działa per token, bez puli globalnej.
- Checkpoint `SmallViTAblation/outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt`.
- Konwerter v2: `SmallViTAblation/outputs/vit_konwerter_dino_filmy_eden/v2/KONWERTER.pt` (poza gitem).
- Dane: `Dataset/Cataract-LMM/3_Object_Tracking` (170 zipów adnotacji z klatkami JPG, 3 zipy MP4),
  manifest `outputs/vit_sd_pelny_latent_filmy_eden_20261006/FILMY.json` (118 train, 25 val).
- Poprzednia sesja: `.sessions/Session_2026-10-06_124141_Konwerter_Latentu_ViT_Na_Cechy_DINO_Filmy_Eden`.

## Wykonane kroki

1. **Sprzątanie Edenu.** Sprawdzono węzeł i GPU po biegach konwertera; nic nie zostało (trap w SLURM działał).
   W katalogu v1 na Edenie został plik STOP (do usunięcia przy okazji).
2. **Test konwertera na zdjęciach** (`experiments/vit_konwerter_dino_test_zdjecia`): 8 zdjęć VAL, konwerter
   przewiduje cechy DINO, szukanie w bazie klatek filmów uczących: 6 z 8 odnalezionych w zbiorze, mediana
   pozycji 172 w bazie 35 tys.
3. **Sztuczny latent z prezentacji** (`experiments/vit_konwerter_dino_sztuczny_latent`): latent J128 z
   `docs/prezentacja_pipeline_latent` jest do 722 odchyleń od rozkładu; maks. 51,35 wobec 4,43 w naturalnym
   latencie (128 kanałów narzędzia: 2,71). Znalezione klatki nie są bliżej celu niż losowe. Wniosek: konwerter
   działa w rozkładzie naturalnym, sztuczny latent trzeba ograniczyć do zakresu osiągalnego.
4. **Granice latentu** (`experiments/vit_granice_latentu`):
   - z wag: wyjście LayerNorm przed skalą ma średnią 0 i normę ≤ √d, więc preaktywacja kanału j leży w
     m_j ± √d‖P(W1_j ⊙ g)‖, potem GELU (minimum −0,16997 przy a = −0,7518);
   - z uczenia: krok AdamW ograniczony przez B_t (Cauchy-Schwarz, B∞ ≈ 7,27), wagi z zanikiem mają punkt
     stały B/λ ≈ 145; γ, β LayerNorm i biasy nie mają zaniku, więc przy nieskończonym uczeniu nie ma
     skończonej granicy. Wykres obwiedni `granice_uczenia.png`.
   - Deep search w Undermind o granicach aktywacji (IBP, Lipschitz, LayerNorm).
5. **Aplikacja, edytor SD** (`app/app.py`, `app/silnik.py`):
   - limity makrowektorów: z wag, z AdamW po T krokach, ze statystyk bazy, z pliku, ręczne; nadpisanie per
     kanał, przycinanie, zapis;
   - jednostka „% zmiany kanału” od −100 % (do min) do +100 % (do max), zakładka tylko J128;
   - generowanie na bieżąco (15 kroków, opóźnienie 300 ms), start z zaszumionego zdjęcia (img2img, PSNR 24,
     23, 22 i 10 dB przy sile 0,1; 0,3; 0,6; 1,0), wybór patcha kliknięciem.
6. **Zakładka detekcji ViT** (`app/detekcja.py`, `app/zakladka_detekcji.py`):
   - wczytanie zdjęcia (etykieta cystotomu z annotation.csv), checkpointu, detekcja z pewnością, IoU, AP@0,5
     i mAP@0,5:0,95, mAP na zdjęciach testowych (k1000: AP50 43,5 %, mAP 11,7 % na 18 zdjęciach);
   - auto-szukanie latentu: Adam z rzutowaniem na limity, strata jak w eksperymencie J128, cele pewności, IoU
     i mAP od 0 do 100 % z objaśnieniami, wagi pewności i ramki;
   - zmienny lr: stały, cosinusowy, liniowy, krokowy, plateau, restarty x1/x3/x6, lr min, rozgrzewka;
     wszystkie harmonogramy na TR_0001 kl. 870 (768 kanałów, zakres naturalny, 900 kroków): pewność
     96,3–96,9 %, IoU 99,5–100 %;
   - wyniki TR_0001 kl. 870: przed 72,5 % / IoU 66 % / mAP 40 %; J128 w limitach z wag 84 % / 99,98 % / 100 %
     (kanały dochodzą do górnego limitu 8,67); 768 kanałów w zakresie naturalnym 97 % / 99,8 %;
   - sprawdzenie „latentu globalnego”: przetasowanie tokenów po auto-szukaniu obniża pewność 0,84 → 0,35
     i IoU 1,0 → 0,73, więc model działa per token; widok makrowektorów opisany jako podsumowanie;
   - widoki latentu: 768, J128, makrowektory (słupki −/+ w skali symlog), cząsteczki 2D, cząsteczki 3D;
     wykresy z przewijaniem, powiększeniem i osobnym oknem.
7. **Cząsteczki 3D** (`app/czasteczki3d.py`): pyqtgraph.opengl (VBO na GPU), 600 tokenów w jednym układzie,
   768 kierunków ze spirali Fibonacciego, tryby przed / po / różnica / nałożone / przejście z animacją,
   wektory albo chmura końców, podłoże z obrazem i atencją, podnoszenie o atencję, łuki atencji, klik na
   patch, zapis PNG. Przeliczenie 600 x 768 wektorów około 40 ms na RTX 3070.
8. **Opcje atencji** (`app/opcje_atencji.py`, wspólne dla 2D i 3D): mapy (średnia otrzymana, z patcha, do
   patcha, skupienie, zasięg, na siebie), głowa (średnia, maks., 1–3), blok albo rollout, pomiń siebie,
   skala, strzałki (k celów, k źródeł, najsilniejszy cel każdego patcha, k par). Obserwacja na TR_0140
   kl. 1655: najsilniejszym celem prawie wszystkich patchy jest token 321 (wiersz 10, kolumna 21): 300 z 600
   w bloku 3, 584 w bloku 5, 591 w bloku 6; wygląda na ujście atencji, nie sprawdzono, czy leży na narzędziu.
9. **Zakładka filmu** (`app/film.py`, `app/zakladka_filmu.py`): lista 170 filmów z podziałem, wypakowanie MP4
   do `Dataset/Cataract-LMM/3_Object_Tracking/videos_mp4/`, zakres od min:s do min:s na N klatek,
   źródło MP4 albo JPG; klatka MP4 i = frame_number i + 1 (sprawdzone na TR_0001), 29,97 kl./s, 720 x 480.
   Latent, detekcja, etykiety (ramka, obrys cystotome_seg, czubek, rogówka, źrenica), ręczny obwód
   narzędzia, maska patchy, przebieg pewności / IoU / zmiany latentu, odtwarzanie, zapis NPZ i JSON.
   TR_0001 0:20–0:30, 100 klatek: konwersja 6 s, średnia pewność 71,0 %, średnie IoU 72,5 %, klatka
   odtwarzania około 85 ms.
10. **Przestrzeń makrowektorów** (`app/przestrzen_makro.py`): kanał c ma stały kierunek d_c; położenie
    patcha p = s (z − μ) D jest sumą 768 mikrowektorów; płynny przelot między klatkami, ślad, łańcuch
    mikrowektorów wybranego patcha. Kierunki PCA (3 składowe wyjaśniają 58 % wariancji na TR_0001
    0:20–0:30), sfera Fibonacciego albo losowe.
11. **Poprawki układu i OpenGL**: `app/uklad.py` (kontrolki zawijają się do kolejnych linii; okno z 3569 px na
    1879 px). Pod koniec sesja Windows była rozłączona (stan Disc, monitory wyłączone) i każdy kontekst OpenGL
    zwracał GL_INVALID_OPERATION; `WidokGL` pomija wtedy klatki. Obraz widoków OpenGL w zakładce filmu nie
    został obejrzany; przestrzeń makrowektorów sprawdzona zastępczo w matplotlib.

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\konwerter-vit-dino-filmy-eden.md`
  i wpis w `MEMORY.md` (pamięć asystenta, nie pliki projektu).

### Zewnętrzne (poza projektem): repozytorium ViT

- Aplikacja `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\`: `app.py`, `silnik.py`,
  `detekcja.py`, `zakladka_detekcji.py`, `czasteczki.py`, `czasteczki3d.py`, `opcje_atencji.py`, `film.py`,
  `zakladka_filmu.py`, `przestrzen_makro.py`, `uklad.py`, `wytnij_klatki.py`, `uruchom.bat`, `README.md`,
  `zdjeciatestowe\MANIFEST.json` (zdjęcia JPG poza commitem, odtwarzalne przez `wytnij_klatki.py`).
- Eksperymenty `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\`:
  `vit_konwerter_dino_test_zdjecia\`, `vit_konwerter_dino_sztuczny_latent\`, `vit_granice_latentu\`.
- Wyniki `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\`:
  `vit_konwerter_dino_test_zdjecia\WYNIKI.json`, `vit_konwerter_dino_sztuczny_latent\WYNIKI.json`,
  `ZAKRES_NATURALNY.json`, `vit_granice_latentu\GRANICE_MODEL_VIT_K1000.json`, `GRANICE_UCZENIA.json`,
  `latent_editor_sd_20261006\auto_szukanie_*.json`, `film_TR_0001_S1_P03_00m20.0-00m30.0_100kl.json`
  (pliki NPZ ignorowane przez git).
- Dane: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking\videos_mp4\TR_0001_S1_P03.mp4`
  (wypakowany z `videos_part001.zip`, 27 MB, poza gitem).
- Ta sesja: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-10-06_171603_Aplikacja_Latentu_Detekcja_Czasteczki_3D_Film_Makrowektory\`.

## Otwarte problemy

- Widoki OpenGL (cząsteczki 3D, przestrzeń makrowektorów) w zakładce filmu nie zostały obejrzane na ekranie,
  bo sesja Windows była rozłączona; trzeba sprawdzić po powrocie do komputera.
- Ujście atencji w tokenie 321 na TR_0140: nie sprawdzono, czy leży na narzędziu ani czy powtarza się na
  innych klatkach.
- J128 w limitach z wag nie przekracza 84 % pewności; kanały siedzą na górnym limicie.
- Plik STOP w katalogu v1 konwertera na Edenie.
- Kierunki PCA w przestrzeni makrowektorów zależą od sekwencji, więc dwie sekwencje nie są wprost porównywalne;
  do porównań lepsze stałe kierunki (sfera albo losowe).

## Następne kroki

- Obejrzeć zakładkę filmu na ekranie (odtwarzanie, przelot cząstek, łuki atencji).
- Sprawdzić, czy patche narzędzia (maska z obrysu etykiety) tworzą w przestrzeni makrowektorów osobne skupisko
  i jak się ono porusza w czasie.
- Zbadać ujście atencji na kilku filmach i klatkach.
- Ewentualnie: stałe kierunki PCA liczone raz na zbiorze uczącym, wspólne dla wszystkich sekwencji.
