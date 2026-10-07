# Podsumowanie sesji 2026-10-07 15:59

## Cel sesji
1. W zakładce filmu: Langevin liczony dla ruchu cząstek na obrazie i włączanie / wyłączanie elementów filmu
   (źrenica, narzędzie itd.) na żywo.
2. W zakładce „Uczenie ViT”: prognoza Langevina, która nie znika od razu, ustawienia ogona historii (EMA,
   zanikanie), średnia cząstek narzędzia jako jedna kropka, śledzenie czubka narzędzia, szybsze odtwarzanie.
3. Kolorowanie obrazu i cząstek mieszanką kolorów wektorów ważoną siłą.
4. Orientacja w przestrzeni: zdeformowane zdjęcie, klik w cząstkę, „ciekawe sposoby wizualizacji przestrzeni
   2D w przestrzeni 768 wektorów” (z przeglądem literatury).
5. Langevin i Koopman dla pojedynczych cząstek z samej ich historii, rozpisanie równania na części (dryf,
   pamięć, szum) i czwarty panel z analizą równania.
6. Wizualizacja powierzchni / krajobrazu, po którym biegają cząstki (analogia wosku z tunelami, szkła),
   wzorowana na „studniach” TFT.

## Kontekst
Repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
Poprzednia sesja: `.sessions\Session_2026-10-07_043248_Uczenie_ViT_Zakladka_Liniowy_Ruch_Czastek` (zakładka
Uczenie ViT, liniowe prawo ruchu v(t+1) = 0,87 v(t) + S_paczki(t), 96-98 %). Dane: lokalna gęsta historia
`SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny` (1001 kroków, 12 klatek).

## Wykonane kroki
1. Film (commit e29332dc): „na żywo” w grupie wycinania przelicza ViT z zapamiętanych oryginałów (około 4 s na
   80 klatek, wyłączenie wszystkich obszarów daje dokładnie latent bez wycinania); Langevin (oscylator) po
   klatkach z prognozą i rzeczywistym torem w makro. Na filmie beta około 0, kappa 0,4-0,7 (inaczej niż w uczeniu).
2. Uczenie ViT (3685a0d6): grupa „Prognoza Langevina i ślad toru”: prognoza co N kroków stoi do następnej,
   stare prognozy, ślad rzeczywisty i ślad prognoz z wyprzedzeniem, EMA, zanikanie, krycie; tryb średniej
   narzędzia / tła (kropka = środek ciężkości, sprawdzone 0,0); „Śledź patch: czubek narzędzia” z cystotome_tip
   (`analiza_uczenia.czubki_biegu`); „Animuj” w 3D bez porównania odtwarza uczenie.
3. Widok 3D (c18fb6ed): „Koloruj obraz mieszanką kolorów wektorów” (waga |wartość|, kwadrat, czwarta potęga,
   tylko dodatnie; automatyczne nasycenie, jasność wg siły, gładko).
4. Makro (c9dda867): zdjęcie zdeformowane do położeń cząstek (siatka 20 x 30, powierzchnia / linie, podpisy rogów,
   środka i narzędzia).
5. Odtwarzanie do 10 000 kroków/s z przeskakiwaniem wg zegara (436143eb): przy 2000 kroków/s całe uczenie < 1 s.
6. Makro (0e862df9): mieszanka kolorów na ruchomych cząstkach liczona przy rysowaniu; przeliczenie makro 15 s -> 7 s.
7. Klik w cząstkę w makro wybiera patch na obrazie i w 3D (09536e80; `WidokGL.na_ekran`, test prawdziwym klikiem).
8. Przegląd literatury (podagent) i cztery wizualizacje (f796c41b): zaznaczanie w obie strony (prawy przycisk
   na obrazie, Shift + przeciągnij w makro), PCA 3 składowe jako RGB w obu widokach (różnica kolorów 0),
   podobieństwo do wybranego patcha (cos), ortonormalne kierunki (QR), Grand Tour, atlas kanałów
   (`atlas_kanalow.py`).
9. Przegląd literatury GLE / Koopman (podagent) i panel równania (cf071f29; `rownanie_czastki.py`,
   `panel_rownania.py`): Langevin, GLE z jądrem L kroków (jedno jądro dla 768 kanałów), Koopman Hankel-DMD
   na r składowych + oscylator na reszcie. Klatka 0, patch 255, błąd / persystencja h = 1 / 5 / 20 / 50:
   oscylator 0,35 / 0,81 / 0,93 / 0,79; GLE L = 10: 0,32 / 0,76 / 0,81 / 0,79; Koopman gorszy na dłuższych.
   GLE L = 1 odtwarza oscylator dokładnie. Szum około 37 % ruchu (jak pchnięcie paczki 31-35 %), kurtoza 6-7.
   Wykresy: błąd vs h, jądro K(s) (regresja i Volterra), rozkład kroku, portret fazowy, ACF i histogram szumu,
   VACF danych i modelu (zgodne), wartości własne Koopmana. Przyspieszenie prognozy GLE do 21 ms.
10. Przegląd fizycznej analogii (podagent; generator studni TFT: `Research\DiagnosticTool\koopman_m5\plotly_studnie.py`)
    i okno „Krajobraz i tunele” (0740d8bd; `krajobraz.py`): teren U = −ln gęstość, wosk z tunelami (kolor =
    wiek), lity wosk, otoczka z odsetkiem masy. Test wchodzenia w ślady poprzedników: 81,5 % wobec 81,6 %
    w kontroli, czyli bez wspólnych kanałów jak w szkle.
11. Atlas i krajobraz przełączają główny panel zakładki Uczenie ViT zamiast osobnego okna (52c6a7b6).
Każda zmiana GUI: test regresji odcisku (24 klucze bez zmian) i test zrzutem ekranu.

## Pliki zmienione / utworzone
### W projekcie (ViT)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\zakladka_filmu.py`, `zakladka_uczenia.py`,
  `przestrzen_makro.py`, `czasteczki3d.py`, `widok_gl.py`, `wspolne.py`, `analiza_uczenia.py`, `README.md`
- nowe: `...\app\atlas_kanalow.py`, `...\app\rownanie_czastki.py`, `...\app\panel_rownania.py`, `...\app\krajobraz.py`
### Zewnętrzne (poza projektem)
- Brak zmian. Tylko odczyt: `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\koopman_m5\nagrania\STUDNIE_3D_*.html`
  i `plotly_studnie.py` (przez podagenta).

## Otwarte problemy
- Korelacja szumu GLE z prawdziwą siłą paczki S_nowa niezmierzona (dane sił na Edenie, około 17 GB).
- Model Kalmana / EM z oszacowaniem nieznanego wejścia (szacunek pchnięcia paczki z historii) niezrobiony.
- Panel równania i atlas tylko w zakładce uczenia; w filmie Langevin bez GLE / Koopmana.
- Pozorne „kratki” w atlasie kanałów: możliwy artefakt położenia ViT, niesprawdzony.

## Następne kroki
- Pobrać wybrane siły z Edenu i policzyć corr(eta, S_nowa) w panelu równania.
- Etap 4 (ulepszenie uczenia: EMA wag, dobór paczek wg siły) po planie i zgodzie.
- Ewentualnie panel równania w zakładce filmu.
