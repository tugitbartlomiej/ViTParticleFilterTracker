# Session: Session_2026-10-23_201556 — od płaszczyzny krok × podkrok do mikrosoczewek (okienko M)

**Created:** 2026-10-23 20:15
**Type:** Mixed (Analysis + Tools)
**Status:** Completed

## Quick Summary
Sesja łączyła trzy rzeczy: (1) pomiary na checkpointach seedu 3047 i s47 (lasery wybiórcze klasowo, temperatura widoczna i
niewidoczna dla głowy, cos kierunku pchania laserów z kierunkiem progu, asymetria jakobianów), (2) długą rozmowę pojęciową o tym,
czym w układzie „lasery i koloidy” jest uwaga, MLP, strata, uczenie, okno odczytu i jaką alegorię fizyczną wybrać,
(3) implementację nowego okienka M „Mikrosoczewki” w `SmallViTAblation/LaserLatentSpaceApp` według tej alegorii.
Droga do wniosków (kolejność pytań, pomiarów i poprawek) jest spisana w `analysis/DROGA_DO_WNIOSKOW.md`.

## Key Results
- Lasery po uczeniu są wybiórcze niemal wyłącznie **tłowo** (blok 4, 400 ep: 40 „tylko tło”, 2 „tylko narzędzie”, 9 „obie”, zero po
  przetasowaniu: 0); 48 laserów wybiórczych robi 73–91 % skoku Fishera w MLP 4 i odsuwa tło od narzędzia (cos −0,23…−0,28).
  Obala to wcześniejszą wersję „lasery zapalają się na narzędziu i je odpychają”.
- Predykcja „sieć uczy się dalej w kierunkach niewidocznych dla głowy” **nie potwierdziła się**: obie składowe drgania stygną
  razem do ~30 ep i są płaskie do 400 ep przy stałym tempie uczenia; kierunki widoczne są 3–4× gorętsze na wymiar niż niewidoczne.
- Lasery nie są pułapkami: jakobiany podkroków mają część antysymetryczną równą symetrycznej (0,86–1,0), cos(v_c, w_c) ≈ 0;
  znak cos odróżnia MLP 1 (rozpycha, +0,07) od bloków 5–6 (ściskają, −0,08…−0,15).
- Nowe okienko M w aplikacji: zdjęcie i płyty łodygi → sfera (formowanie) → matryca mikrosoczewek, lampa i ekran; przejścia
  mają klucz `glowa` (głowa na 13 stanach); 48 testów przechodzi; nic nie zacommitowane.

## Files
- `SESSION_SUMMARY.md` — pełna dokumentacja
- `analysis/DROGA_DO_WNIOSKOW.md` — jak doszliśmy do wniosków, krok po kroku, z poprawkami błędnych wersji
