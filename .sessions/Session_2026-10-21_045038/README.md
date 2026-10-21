# Session: Przeciwne wiązki 0–200 epok, trening do 400 epok, dynamika koloidów i laserów

**Created:** 2026-10-21 04:50 (sesja trwała od 2026-10-20 ok. 22:00)
**Type:** Mixed (Analysis + SSH Eden + Training)
**Status:** Completed

## Quick Summary
Wczytano kontekst starych sesji z `C:\Users\bartl\.claude` i sprawdzono stan Edenu. Kontynuacja treningu seed 3047 do 200 epok została pobrana i przeanalizowana: pary przeciwnych wiązek w epokach 0–200, znoszenie pary 370–391, IoU i pewność. Uruchomiono i ukończono trening do 400 epok. Przeprowadzono weryfikację literaturową (Undermind, Scite) z raportem. Blok 6 opisano też jako układ fizyczny: 600 koloidów, 768 laserów, pułapki.

## Key Results
- Przeciwne wiązki (≥150° na tym samym tokenie) to rzadki produkt uboczny znaku GELU w parach częściowych kopii, a nie mechanizm znoszenia. Liczba par: 0 → 30 (ep. 80–110) → 15 (ep. 200); po szczycie pary przeważnie się rozchodzą, w pełną kopię scaliła się tylko para 370–391 (173°). Jej wyłączenie zmienia IoU o ok. 1e-5.
- IoU stoi od ok. epoki 40 (VAL 0,762 w ep. 50, 0,765 w ep. 200), pewność rośnie także na klatkach bez narzędzia.
- Rzut 3D fabrykuje przeciwne pary na poziomie przypadku (13,5% zamiast 0–9 na blok). Wypadkowa w rzucie P3 aplikacji jest wiarygodna.
- Dynamika koloidów: rozdział faz (96–98% cząstek narzędzia poza chmurą tła), chłodzenie do stałej temperatury, ruch w klatkach (MSD ∝ Δt^0,2–0,34), spłaszczenie chmury do ok. 3 wymiarów. Ostatnia komora działa jak wspólny wiatr i hamulec (−11–13% rozdziału).
- Lasery: kolimacja 154 → 31 kierunków, stały obrót 2,2°/epokę, moc ×3, przesunięcie specjalizacji ku tłu, drużyna hamulcowa 10–19 laserów. Na plateau IoU koreluje ze stanem specjalizacji (ciemne lasery i lasery narzędzia), a nie z temperaturą. To wynik wstępny.
- Trening do 400 epok ukończony (job 1911922, krok 437500).

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
