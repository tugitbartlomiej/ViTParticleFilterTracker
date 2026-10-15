# Session Summary: Pułapka optyczna jako ognisko wiązek, zasady układu, pomiary jak w pęsecie

## Metadata
- **Date:** 2026-10-13 → 2026-10-14
- **Time:** zapis 2026-10-14 20:37
- **Status:** Completed
- **Type:** Mixed (Analysis, aplikacja GUI, literatura Undermind/Scite, Eden SSH)
- Worktree: `.claude/worktrees/laser-koloid-zasady` (gałąź codex/eden-nano-training-20260914); pliki kopiowane do głównego folderu.

## Objective
1. Walidacja checkpointów ViT Model Zoo na Imagenette; porządki na Edenie (pobrać lokalnie, usuwać po zgodnym MD5, rejestr).
2. Zrozumieć i nazwać obiekty układu: koloid (token), reflektor (neuron MLP: celownik w_c, wiązka v_c), pułapka.
3. Zaplanować pomiary „jak w pęsecie optycznej” na Edenie.
4. Narysować w aplikacji pułapkę z laserów jako ognisko wiązek, osobne okienko ze śledzeniem, zbliżenie na jedną cząsteczkę,
   pewność, że ognisko przesuwa cząsteczkę w każdym kroku i podkroku.
5. Przenieść i zaktualizować zasady układu do `app/`, opisać mosty ML↔fizyka, sprawdzić literaturę o ogniskach.

## Context
Kontynuacja prac z 2026-10-12/13 (zasady laser-koloid, temperatura bramek Fermi, Model Zoo). Wcześniej słowo „pułapka” w aplikacji
oznaczało średnią położenia z ±300 kroków; użytkownik ustalił nowe znaczenie: pułapka optyczna z laserów.

## Actions Taken
1. **Model Zoo + Imagenette:** pobranie Imagenette val z Edena (3925 zdjęć), skrypt `experiments/vit_modelzoo_20261013/kod/waliduj_imagenette.py`
   (top-1/top-5, 10 klas, temperatura bramek w 12 blokach), commit b6576826.
2. **Eden:** spis plików, wykrycie równoległych porządków innej sesji (rejestr `~/smallvit_runs/USUNIETE_Z_EDENU.txt`), pobranie
   `lr_5e-4_rp` (4,3 GB) z MD5 21/21 i usunięcie na Edenie; rejestr w `.sessions/Session_2026-10-13_165308_Eden_pobrane_usuniete/`.
3. **Wyjaśnienia dla użytkownika:** czym jest czujnik (celownik), wiązka (igła v_c), dlaczego wiązki jednego reflektora są równoległe,
   skąd zestaw zapalonych laserów, CNN a ruch koloidów, chłodzenie ≠ dobra detekcja, czemu chmury na sferze i w podkrokach różnią się
   (inne miejsce w sieci, inny rzut).
4. **Plan Edenu:** `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md` — E1 kalibracja OU, E2 FORMA kontra dryf dokładny,
   E3 wir i prądy, E4 FDT(ω)/Harada–Sasa, E5 entropia, E6 Kramers przełączeń laserów, E7 pary, E8 mikroreologia (deep search 228 prac).
5. **Aplikacja (zakładka „Lasery i koloidy”):**
   - zmiana nazwy „pułapka” → „średnia skoków koloidu”;
   - nowy moduł `app/ogniskowa.py`: pole 768 wiązek bloku 6 wokół cząsteczki (przekrój 3D rzutu sfery), zero pola (Newton), osie
     siodła i wir, kontrola z przejściem sieci; tryb szybki „tylko wiązki” (806 ckpt w 47 s);
   - okienko 11 (`PoleOgniskowej` w `lasery_gpu.py`): zbliżenie na ognisko jednej cząsteczki — 768 wiązek z kierunku, w który pchają,
     w kolorach reflektorów, światło wiązek gaussowskich, osie kształtu ogniska, wkłady (strzałki albo łańcuch), skok i prawdziwy ruch,
     czas: oś uczenia albo podkroki (MLP: wiązki bloku; uwaga: 600 wkładów od innych cząsteczek), kamera śledząca;
   - opcjonalnie lokalne pole (mgła, strzałki) i pułapka na dużej sferze.
6. **Dokumenty:** `app/ZASADY_DZIALANIA_UKLADU.md` (Z1–Z29; stare `film_krysztal/ZASADY_LASER_KOLOID.md` i `NADAZANIE_PULAPKI_S47.md`
   jako archiwum z notką), `app/MOSTY_ML_FIZYKA_PULAPKI.md` (agent w tle: 15 mostów, 63 prace), `app/OGNISKA_FIZYKA_I_SIECI.md`
   (2 deep searche 2026-10-14 + Scite), README aplikacji.
7. **Literatura:** deep searche Undermind „Measurement toolkit for colloidal particles in optical traps…”, „Focal spot formed by many
   converging beams…”, „How individual MLP neuron contributions add up and cancel on a single token…”; Scite (metadane, brak wycofań,
   `report_citations`); lista 28 linków w `literatura_szklo_tokeny_20261011/linki/ogniska-wiazek-wklady-neuronow-2026-10-14.txt`.

## Results

### Key Findings
- ViT-S/16: 76,3 % top-1/1000 na Imagenette po 90. ep.; bloki 7–11 T 0,03–0,12 (zgaszone 88–95 %), bloki 0–5 T 0,4–0,74.
- Ognisko przesuwa cząsteczkę: Σ wkładów = ruch (192D) z błędem ≤ 1,3·10⁻⁶ w 806 checkpointach; podkroki MLP 3–8·10⁻⁷, uwaga 2·10⁻⁶.
- Lasery działają tylko w 6 z 12 podkroków (MLP); w podkrokach uwagi cząsteczkę przesuwają inne cząsteczki.
- Zero pola wiązek w zasięgu 3 r99: 7,7 % checkpointów (głównie początek uczenia); zwykle prąd wiązek + siodło z wirem.
- Sprawność ogniska (3D): start 5 %, później 11–23 % (blok 6); w podkrokach blok 4 68 %, blok 1 32 %, blok 6 16 %; kształt ogniska
  w bloku 4 wydłużony ×1,7, w blokach 1 i 6 prawie kulisty.
- Literatura: luka — brak pomiarów współpracy/znoszenia wkładów neuronów na jednym tokenie i ich ewolucji w uczeniu (szczególnie ViT);
  najbliżej: korekta między warstwami (Patrawala NeurIPS 2025), neurony osłabiające (Gerstner 2026), samonaprawa (Hydra 2023).

### Issues Encountered
- Inna sesja równolegle sprzątała Edena (folder `wyniki` biegu 35000 usunięty bez wpisu w rejestrze) — zamiast równoległego kasowania
  dopisywanie do wspólnego rejestru.
- Pliki aplikacji w głównym folderze były nowsze niż w worktree (zmiany innej sesji) — synchronizacja i osobny commit 16f4e713.
- Pierwsza wersja „pułapki” liczyła globalne pole i rzadko znajdowała zero; użytkownik chciał ogniska jednej cząsteczki — przebudowa
  na zbliżenie z wiązkami; wiązki z wkładem względem średniej były niemal wszystkie błękitne → rysowanie z kierunku pchnięcia.
- Cache historii w wersji 1 (inne położenia) dawał błąd rzutu 10 → test na wersji 3 + dopasowanie rzutu do zapisanych położeń.
- Heredoc w Bash z długim kodem się rozjeżdżał → łatki jako pliki .py w scratchpadzie.

## Files Generated/Modified
- `SmallViTAblation/app/ogniskowa.py` (nowy), `app/lasery_gpu.py`, `app/zakladka_laserow.py`, `app/README.md`
- `SmallViTAblation/app/ZASADY_DZIALANIA_UKLADU.md`, `app/MOSTY_ML_FIZYKA_PULAPKI.md`, `app/OGNISKA_FIZYKA_I_SIECI.md` (nowe)
- `outputs/porownanie_szklo_20261010/film_krysztal/ZASADY_LASER_KOLOID.md`, `NADAZANIE_PULAPKI_S47.md` (notka archiwum)
- `experiments/pomiary_pulapki_fizyka_20261014/PLAN.md`, `experiments/vit_modelzoo_20261013/` (kod + wyniki)
- `assets/Dane_Imagenette/imagenette2-320/val`, `assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/lr_5e-4_rp` (dane, poza git)
- Cache aplikacji: `app/cache_uczenia/lasery/eafb5a973be3d490/ogniskowa_t226_G1_z005.npz` (i wersje G17)
- Commity: b6576826, 6ae5faa5, 6b601c8b, 16f4e713, 406018c6, fb4fb581, aee2f4c4, a8d746a5, e556a5fc (bez push)
- Pamięć: `pulapka-optyczna-znaczenie.md`, `pomiary-pulapki-fizyka-plan-2026-10-14.md`, `vit-modelzoo-imagenette-2026-10-13.md`,
  aktualizacja `zasady-laser-koloid.md`

## Commands Used
- `ssh eden-cluster 'tar cf - ...' | tar xf -` (pobieranie), `md5sum` po obu stronach, `rm -rf` na Edenie tylko po zgodnym MD5
- `F:/Python/3.13/python.exe -X utf8 test_zakladka_ogn.py <app> <historia> <wyj>` (test GUI okienka 11 z renderem)
- `py -3.11 -I -X utf8 waliduj_imagenette.py <przebieg> <val> <json>`
- Undermind: `launch_deep_search`, `inspect_deep_searches`, `get_paper_info`; Scite: `search_literature`, `report_citations`

## Next Steps
- [ ] Opcja w okienku 11: pokazywać też słabe wiązki (wszystkie, które świecą na cząsteczkę), nie tylko główne.
- [ ] Policzyć w 192D: sprawność ogniska η, efektywną liczbę wiązek N i spójność C = η·√N przez całe uczenie (token narzędzia, wszystkie bloki).
- [ ] Test przyczynowy: przeciwlasery a znoszenie wiru (przewidywanie z fizyki wiązek przeciwbieżnych).
- [ ] Ognisko uwagi kontra ognisko MLP w podkrokach (korekta wewnątrz bloku, ViT).
- [ ] Uruchomić pobieranie PDF z listy `ogniska-wiazek-wklady-neuronow-2026-10-14.txt` (czeka na zgodę).
- [ ] Eksperymenty E1–E8 na Edenie wg `PLAN.md`; ewentualnie przyspieszenie `ogniskowa.py` i liczenie od bieżącego checkpointu.
