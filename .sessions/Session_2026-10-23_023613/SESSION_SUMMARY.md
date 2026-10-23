# Session Summary: Kroki × podkroki: eksperyment i tezy artykułu; tempo podkroków do 600; strzałki pola w okienku 14

## Metadata
- **Date:** 2026-10-23
- **Time:** 00:41–02:36 (Europe/Warsaw)
- **Status:** Completed
- **Type:** Mixed (Analysis + aplikacja)
- **Branch:** `feature/rzuty-192d-okienka-gpu` (wypchnięta); commity b441534c, 6da2a60c, 9603507e

## Objective
1. Sprawdzić, co widać tylko na płaszczyźnie krok uczenia × podkrok przejścia (MałyViT s47), czy uwaga jest przekształceniem
   afinicznym i jakie są analogie fizyczne (pułapki optyczne, pola); literatura przez Undermind.
2. Wykresy, raport i krótki wniosek eksperymentu (zamówione).
3. Tezy badawcze pierwszego artykułu z dotychczasowych pomiarów.
4. Aplikacja: większe tempo podkroków; mapa pułapek (okienko 14) jako strzałki pola w całej chmurze 600 cząsteczek.
5. Commit, zapis sesji i wyłączenie komputera za 2 godziny.

## Context
Poprzednia sesja (8821939a): okienko P i pełny ekran dla Odyssey 3D Hub. Zapisane przejścia `cache_uczenia/lasery/7992f2317c64a2e3/przejscia`
(807 plików, 17 GB) pozwalały mierzyć wszystkie 12 podkroków we wszystkich checkpointach bez liczenia sieci.

## Actions Taken
- Eksperyment `SmallViTAblation/experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29/` (Kod_eksperymentu / Opis / Wyniki):
  `pomiar_kroki_podkroki.py` (obserwable 806 × 12, 1247 s), `jakobiany_podkrokow.py` (autograd float64, 7 tokenów × 7 checkpointów),
  `pochodzenie.json`, wpis w `experiments_index.md`.
- Trzy wyszukiwania Undermind (workspace 40294fa6-…); lista prac w `Wyniki/Dane/literatura_undermind.md`.
- Wykresy F1–F9 (PNG + SVG), `Wyniki/Raport/RAPORT.md`, `Wyniki/Krótki_Wniosek/WNIOSEK.txt`.
- Tezy artykułu: `SmallViTAblation/reports/Tezy artykułu o płaszczyźnie krok × podkrok.md` (T0–T4, predykcje P1–P7, szkielet, minimum obliczeń).
- Aplikacja: `sp_pod_tempo` 0,1–600 podkr./s z krokiem adaptacyjnym, pętla animacji z resztą (modulo).
- Aplikacja: `trap_map.py` WERSJA 2 zapisuje J₀ (K, 600, 3, 3); `pole_w_komorkach`, `siatka_komorki`; `trap_map_gl.odcinki_strzalek`
  (trzon z przygaszonym ogonem i grot z 4 odcinków); okienko 14: rodzaj strzałek (pełne pole / kształt pułapki), siatka 1–5 na bok,
  długość ∝ |f| albo jednakowa, mnożnik, krycie; nowy nagłówek grupy w `parameter_groups.py`; test `test_strzalki_pola_w_komorkach`.
- Mapa pułapek historii `44452ed38a748aa7` przeliczona w wersji 2 (218 s).

## Results

### Key Findings
- H1/H2 potwierdzone: udział translacji 1,000 w kroku 0; efektywne źródła 580 → 10–66; wzbogacenie uwagi na narzędziu 40–75×.
- H3 odrzucona: 50–53 % kierunków kurczy w każdym podkroku i w iloczynie; nakładanie podprzestrzeni kurczących = losowe;
  log σ_max 0,85 → 2,1–2,6, log σ_min −1 → −5,5, log|det| −1 → −18,5.
- H4 potwierdzona: wskaźnik balistyczny 0,98 → 2,88 → 0,97; Fisher bloku 4 1,34 → 3,20; korekta sąsiednich podkroków cos do −0,16.
- Okienko 14: kształt pola J₀u w rogu komórki to mediana 4,0 % skoku f(0) w checkpoincie 101 i 9,6 % w 806, więc pełne pole w komórce
  jest prawie równoległe; tryb „kształt pułapki” pokazuje zbieganie (ściąganie) wokół izolowanych cząsteczek.

### Issues Encountered
- Nazwa wyszukiwania Undermind z dwukropkiem odrzucona — zmieniona.
- Git Bash `date` ignoruje TZ — godzina folderu z PowerShell (zapisane w pamięci).
- Git w Bash bez tożsamości — commity z `-c user.name/user.email` jak w historii repozytorium.
- Heredoc złamał `\n` w dymkach — naprawione skryptem.

## Files Generated/Modified
- `SmallViTAblation/experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29/**`, `SmallViTAblation/experiments/experiments_index.md`
- `SmallViTAblation/reports/Tezy artykułu o płaszczyźnie krok × podkrok.md`
- `SmallViTAblation/LaserLatentSpaceApp/laser_app/lasers_tab/substeps_view.py`, `ui/tooltips.py`
- `SmallViTAblation/LaserLatentSpaceApp/laser_app/compute/trap_map.py`, `gpu/trap_map_gl.py`, `lasers_tab/trap_map_view.py`,
  `ui/parameter_groups.py`, `tests/test_modules.py`, `README.md`
- `cache_uczenia/lasery/44452ed38a748aa7/mapa_pulapek_z100.npz` (wersja 2, poza gitem)

## Commands Used
```
W:/Python/3.11/python.exe Kod_eksperymentu/pomiar_kroki_podkroki.py Wyniki/Dane 1
W:/Python/3.11/python.exe Kod_eksperymentu/jakobiany_podkrokow.py Wyniki/Dane
PYTHONPATH=. W:/Python/3.11/python.exe -m unittest tests.test_modules.TestPulapkiWsadowo
git push origin feature/rzuty-192d-okienka-gpu
shutdown /s /t 7200
```

## Next Steps
- [ ] Replikacja tez: seed 3047 i 2–3 inne seedy, 10–20 klatek TEST z większym narzędziem, ~60 tokenów w jakobianach (decyzja użytkownika).
- [ ] Obejrzeć strzałki okienka 14 w działającej aplikacji (sprawdzone tylko zrzutem offscreen).
- [ ] Pozostałe historie z mapą w wersji 1 (np. `2cca2743e1c74b14`) przeliczą się przy pierwszym pokazaniu okienka 14.
- [ ] develop nadal niewypchnięty (czeka na decyzję użytkownika).
