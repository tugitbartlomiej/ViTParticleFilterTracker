# Session Summary: Okienko 11 — pole w komórce tokenu, podkroki przez 6 bloków, zapis przejść

## Metadata
- **Date:** 2026-10-14 → 2026-10-15
- **Time:** zapis 2026-10-15 04:37
- **Status:** Paused (komputer wyłączany na prośbę użytkownika)
- **Type:** Mixed (aplikacja GUI, analiza)
- Worktree `.claude/worktrees/laser-koloid-zasady`; pliki kopiowane do głównego folderu po sprawdzeniu, że główny = HEAD (bez CRLF).

## Objective
Rozbudowa okienka 11 („Pułapka optyczna”) zakładki „Lasery i koloidy” według kolejnych próśb użytkownika: reflektory, obrys
ogniska, zapis wyników, pole tylko w obrębie jednego tokenu, kadr i skala, ruch w nadchodzącym kroku, przebieg przez wszystkie bloki.

## Context
Sesja 2026-10-14 (`Session_2026-10-14_203707_Pulapka_Optyczna_Ognisko_Wiazek`) zbudowała okienko 11 (ognisko 768 wiązek bloku 6
wokół jednej cząsteczki, kontrola suma wkładów = ruch ~1e-6). Tu dalsze prośby użytkownika na żywo, ze zrzutami ekranu.

## Actions Taken (commity w worktree, wszystkie skopiowane do głównego folderu)
1. ee4aba99 — reflektory w skali sfery (film 12): 768 reflektorów na zewnętrznej sferze, wiązka reflektor → szkło → cząsteczka;
   kamera z okna sfery (ta sama liczba pikseli na jednostkę, pole widzenia pyqtgraph poziome).
2. d12ce244 — obrys ogniska 3D: powierzchnia połowy jasności sumy wiązek gaussowskich (torch na GPU + marching cubes, ~12 ms/klatkę),
   warstwice wzdłuż osi obrysu, lupa.
3. 487df0b1 — `app/przejscia_zapis.py`: zapis przejść przez sieć do `cache_uczenia/lasery/<historia>/przejscia/k<krok>.npz`,
   wspólny rzut `baza.npz`, zapis wszystkich w tle, float32/float16, limit GB, zamek sieci (wspólna `lasery_dane._siec`).
4. 4d8a102c — `ogniskowa.py` wersja 4: pole tylko w komórce tokenu (połowa odległości do najbliższego tokenu w rzucie 3D), zero
   szukane tylko w komórce, siatka nie rośnie; podgląd ogniskowej dopiero po kliknięciu (domyślnie wyłączony).
5. 501bc26a — kadr: środek koloid / pułapka × skala (sześcian tej samej wielkości / stała przez uczenie); pułapka zastępcza.
6. 7c3a6e49 — gruba zielona strzałka: ruch koloidu w nadchodzącym kroku, kąt do skoku w opisie.
7. e44e03c5 — pole w KULI komórki (mgła i strzałki poza kulą ukryte, okręgi kuli zamiast pudła), płynące strzałki pola (timer
   tylko gdy okienko widoczne), obrys wpasowany w komórkę.
8. e76507c8 — podkroki: jedna długość wiązek na całe przejście (bez skoków skali), stały kadr, tytuł okienka z blokiem i podkrokiem.
9. 046bef98 — podkroki: reflektory zawsze zapalone (także w uwadze), jedna skala sfery dla uwagi i MLP, bez wygaszania wiązek.
10. 4e92c705 — pozostałe cząsteczki w tle, przyciemnione.
11. f48abba5 — strzałki pola domyślnie „pole wiązek”; w podkrokach informacja, że pole jest tylko w osi uczenia.
12. (ten zapis) — `app/pole_podkrokow.py`: pole sił w kuli komórki w każdym z 12 podkroków (MLP: 768 laserów bloku; uwaga: próbna
    cząsteczka słucha 600 tokenów, wagi A − ā), f(0) ma być równe ruchowi tokenu. NIEPRZETESTOWANY, niepodłączony do GUI.

## Results

### Key Findings
- Komórka tokenu 226: r_kom 0,003–0,355 r99, mediana 0,15 r99; skok bloku 6 (0,3–0,7 r99) jest większy niż komórka.
- Zero pola wiązek w komórce: 0 z 806 checkpointów; pułapka zastępcza (najsłabsze pole) leży w rogu/na brzegu komórki.
- Ruch w uczeniu ⟂ skok w głąb sieci: kąt mediana 85° (kwartyle 55–117°), tylko 55 % kroków < 90°.
- Obrys ogniska: blok 4 wydłużony ×2,1; bloki 1 i 6 ×1,2.
- Pole na siatce 17³ dla 806 ckpt: 232 s na lokalnym GPU (w cache, `ogniskowa_t226_G17_z100.npz`).

### Issues Encountered
- CUDA OOM przy 6,6 GB „wolnych” — kartę zajmowały inne programy (drugi Python, VoiceDesk); drugie podejście przeszło.
- Pod koniec sesji Windows nie tworzył nowych procesów (kod 107 nawet dla `echo`, `uv_spawn`); po chwili wróciło samo.
- Strażnik worktree blokuje złożone polecenia z git i pętle; łatki jako skrypty .py w scratchpadzie.

## Files Generated/Modified
- `SmallViTAblation/app/lasery_gpu.py`, `zakladka_laserow.py`, `ogniskowa.py`, `README.md`
- nowe: `SmallViTAblation/app/przejscia_zapis.py`, `SmallViTAblation/app/pole_podkrokow.py`
- cache: `app/cache_uczenia/lasery/eafb5a973be3d490/przejscia/` (6 przejść + baza.npz), `ogniskowa_t226_G17_z100.npz`

## Commands Used
- Testy GUI: `F:/Python/3.13/python.exe -u -X utf8 <scratchpad>/test_*.py <app> <historia> <wyjście>`
  (historia `app/cache_uczenia/lasery/eafb5a973be3d490`, token 226)
- Łatki: `py -3.11 -X utf8 <scratchpad>/latka_*.py <app>`

## Next Steps
- [ ] Uruchomić kontrolę `pole_podkrokow.py` (scratchpad `test_pole_pod.py`: f(0) = ruch w 12 podkrokach, czas), potem podłączyć
      do okienka 11 (wątek w tle, druga instancja PoleOgniskowej dla podkroków, płynące strzałki i mgła w każdym bloku).
- [ ] Ewentualnie legenda kolorów strzałek w rogu okienka 11.
- [ ] Ewentualnie pole w osi uczenia dla każdego bloku (nie tylko 6): ~25 min liczenia pola na siatce.
- [ ] Zaległe z 2026-10-14: η, N, C w 192D przez całe uczenie; przeciwlasery a wir; E1–E8 na Edenie; pobieranie PDF z listy ognisk.
