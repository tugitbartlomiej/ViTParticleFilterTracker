# Podsumowanie sesji 2026-10-07 23:41

## Cel sesji
1. Wygoda zakładki „Uczenie ViT”: wiele zaznaczonych patchy, szybkie wczytywanie, odpinane i chowane panele,
   czytelne ścieżki prognozy Langevina, porządek ustawień.
2. Równanie ruchu cząstki krok po kroku w 768 wymiarach z niedopasowaniem w każdym kroku.
3. Skąd pochodzi siła (wektor) patcha: przepływ sygnału od obrazu przez warstwy i głowy do wektora.
4. Teoria i pomiar równania ruchu w uczeniu: czy bezwładność to pęd AdamW, rozkład pchnięcia na obrazy paczki,
   połączenie z obrazami SD, wyszukiwaniem prawdziwych klatek i konwerterem do cech DINO (nowa zakładka pomiarowa).

## Kontekst
Repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
Poprzednia sesja: `.sessions\Session_2026-10-07_155955_Uczenie_ViT_Rownanie_Czastki_Krajobraz_Wizualizacje`.
Dane: gęsta historia `SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia` (1001 kroków, 12 klatek,
wagi co 10 kroków, bez stanu AdamW), nagrania sił `maly_vit_rownanie_latentu\nagranie_s47_sily`, `nagranie_s48_sily`.
Wcześniejszy łańcuch SD: `.sessions\2026-10-05-latent-vit-sd-langevin.md` (prognoza pierwszego kroku 88,8-98,6 %,
stała siła na 100 kroków zawiodła; kampania 4.3: ramiona SD najwyższe mAP i najniższa siła tożsamości).

## Wykonane kroki
1. Zaznaczanie wielu patchy (27e05405): Ctrl + klik na obrazie i w makro, Ctrl przy prostokącie, osobne kolory,
   „śledź tylko zaznaczone”, karta wykresów „Zaznaczone patche” (prędkość, odległość i cos do stanu końcowego,
   rozrzut grupy).
2. Zapis i autostart (28f84d99): miary ruchu, detekcja, oscylator i czubek w `app\cache_uczenia\*_wyniki.npz`
   (wczytanie 39 s -> 13 s), równania cząstek w `*_rownania.pkl`, ostatnia klatka wczytuje się w tle po starcie.
   Ramki paneli z „Odepnij” i „✕”, pola „Widoki”; „Tylko zaznaczone patche” w widoku 3D i makro; przycisk
   „zaznaczone” w panelu równania (do 20 cząstek).
3. Równanie krok po kroku (cf7f2509, `app\rownanie_krokowe.py`): dopasowanie kroczące z ostatnich W kroków
   (sprawdzona przyczynowość: zmiana przyszłości nie zmienia prognozy z t) albo stałe okno; patch 255, GLE, kroki
   700-1000: błąd kroku 0,54 (kroczące) i 0,55 (stałe), cos 0,86; bezwładność spada z ~1 do ~0,65 w krokach
   100-250, sprężyna ma szczyt około kroku 450. Strzałki prawdziwy / przewidziany / niedopasowanie i ślad w kolorze
   błędu w makro.
4. Interfejs: panel równania z przeciąganym podziałem (be7390db), obraz i wykresy jako osobne panele (eccb67b8),
   domyślne ustawienia makro użytkownika (41c57e37), ustawienia jako strony „Opcje” makro i środek „zero latentu”
   (549afcfa). Maksymalizacja okna (52d0927f): zakładka detekcji miała minimum 1240 px (okno 1324 px > ekran 1032);
   główne zakładki w przewijanych polach; w c83755bc poprawiony błąd tej poprawki (strona rosła do 3800 px przez
   heightForWidth; pośrednik bez układu).
5. Ścieżki prognozy Langevina (c83755bc): osobne warstwy GL z kolorami i grubościami, schematy kolorów, kropka co
   krok, kolor wg horyzontu, powiększenie ruchu (sprawdzone dokładnie × k), linia błędu, legenda, kamera śledząca;
   strony „Opcje” przewijane.
6. Okno „Źródła siły” (929b4649, `app\przeplyw_sily.py`, `app\okno_zrodel_sily.py`): przy wagach atencji i skalach
   LayerNorm z przebiegu wartość wektora przed GELU jest liniowa w liściach (łodyga, położenie, wyjścia MLP, biasy);
   ręczny przebieg = model co do 3e-6, suma wkładów = wartość przed GELU, stałe zgodne. Integrated Gradients łodygi
   (zupełność 3e-4). Animacja przepływu przez 14 etapów, siatka warstw × głów, graf rzeki sygnału, wybór wektora.
   Patch 255, wektor 22, krok 1000: MLP bloku 4 daje 0,97 z 2,92; głowa 1 bloku 5 przenosi 0,58.
7. Przeglądy literatury (Undermind, przestrzeń „ViT: detekcja końcówki narzędzia”): AdamW jako układ wejście-stan-
   wyjście, pierwszy moment jako nośnik pamięci, błąd atrybucji przy pominięciu AdamW, kolejność paczek jako szum
   pierwszego rzędu, filtry ze wspólnym oszacowaniem stanu i nieznanego wejścia, DMD z nieznanym wymuszeniem,
   teoria tokenów jako cząstek.
8. Odtworzenie paczek (`app\paczki_uczenia.py`, `experiments\maly_vit_bezwladnosc_a_ped_adamw\kod\sprawdz_paczki.py`):
   nagranie 3000/3000 paczek zgodne; gęsta historia: 10 kroków od k0000 = k0010 z różnicą 0,017 % ruchu wag
   (kontrola z innym ziarnem 40 razy więcej).
9. Bezwładność a pęd AdamW (`kod\bezwladnosc.py`): siła pędu = 0,83-0,85 poprzedniej prędkości (trafność 0,87-0,92),
   obserwowana bezwładność 0,82-0,84; bez pędu równanie wyjaśnia 20 % ruchu; pęd 0,88 ruchu, nowa paczka 0,57.
10. Powtórka uczenia (`app\stan_uczenia.py`): 98 s całość, stan dowolnego kroku 3 s; dryf powtórki 9 % ruchu
    w 10 krokach, dlatego wagi przypinane co 10 kroków do zapisu.
11. Pchnięcie obrazów (28088228, `app\pchniecie.py`, `app\kandydaci.py`, `app\zakladka_pchniecia.py`): składnik
    straty obrazu przy normach paczki, ustalony mianownik AdamW, z(θ + d) − z(θ). Przewidziany krok sondy wobec
    rzeczywistego 0,983-0,987 (kroki 100, 500, 900), suma 32 obrazów wobec paczki 0,994-0,999. Kandydaci: paczka,
    baza, odszukane dla SD, obrazy SD (pseudo-etykieta), DINO dla SD, konwerter latent -> DINO.
12. Pierwszy pomiar (060efe46, krok 500, sonda 0): obrazy SD i klatki DINO obniżają stratę sondy i siłę tożsamości;
    klatki odszukane przez MSE latentu podnoszą tożsamość bez zmniejszenia straty. Jeden krok, jedna sonda.
Każda zmiana GUI: test regresji odcisku (24 klucze; różnica tylko w liście zakładek po dodaniu nowej) i test
zrzutem ekranu (widoki GL na zrzutach były czarne przy zablokowanej sesji pulpitu, sprawdzane liczbami).

## Pliki zmienione / utworzone
### W projekcie (ViT)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\`: `zakladka_uczenia.py`,
  `przestrzen_makro.py`, `czasteczki3d.py`, `zakladka_filmu.py`, `panel_rownania.py`, `analiza_uczenia.py`,
  `uklad.py`, `wspolne.py`, `app.py`, `README.md`
- nowe w `...\app\`: `rownanie_krokowe.py`, `przeplyw_sily.py`, `okno_zrodel_sily.py`, `paczki_uczenia.py`,
  `stan_uczenia.py`, `pchniecie.py`, `kandydaci.py`, `zakladka_pchniecia.py`
- nowy eksperyment: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_bezwladnosc_a_ped_adamw\`
  (`WYNIKI.md`, `WYNIKI_bezwladnosc.json`, `WYNIKI_sprawdz_paczki.json`, `kod\`, `wyniki\PCHNIECIE_k0500_sonda00.json`)
- pamięć podręczna (nie w repo): `...\app\cache_uczenia\` (wyniki, równania, migawki AdamW `adamw_*`, około 0,5 GB)
### Zewnętrzne (poza projektem)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\bezwladnosc-czastek-to-ped-adamw.md` i wpis
  w `MEMORY.md`.
- Undermind: dwa nowe przeszukania w przestrzeni „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”.
- Tylko odczyt: nagrania sił, wyniki pętli 4.3 (`maly_vit_petla_sd_rownanie`), `outputs\vit_langevin_cztery_paczki_sd`,
  `outputs\vit_dino_faiss_35000`, `outputs\vit_konwerter_dino_filmy_eden\v2\KONWERTER.pt`, archiwa ZIP Cataract-LMM.

## Otwarte problemy
- Pomiar pchnięcia tylko z jednego kroku i jednej sondy; brak przedziałów niepewności.
- Obrazy SD mają pseudo-etykiety z wag końcowych (mogą im sprzyjać); zmiana straty liczona tylko przez latent.
- Dryf powtórki uczenia (9 % ruchu w 10 krokach) między kotwicami; stan AdamW przybliżony między zapisami wag.
- Zakładka pchnięcia działa tylko dla biegu z kodem uczenia (gęsta historia).
- Widoki GL nie dają się sprawdzić zrzutem przy zablokowanej sesji pulpitu.

## Następne kroki
- Przebieg zbiorczy pchnięcia: wiele kroków (np. 20) × 12 sond, średnie i przedziały w źródłach.
- Prawdziwe etykiety dla obrazów SD (ramka z klatki układu) zamiast pseudo-etykiet.
- Rozwinięcie pędu na historię paczek (ile przeszłych paczek działa w kroku t).
- Plan treningu z doborem paczek według pchnięcia, z ramieniem kontrolnym i oceną mAP, do akceptacji użytkownika.
