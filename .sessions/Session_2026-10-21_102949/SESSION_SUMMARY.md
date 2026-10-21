# Session Summary: Pole wokół pułapek koloidów, fizyka chmury, okluzja i miejsca publikacji

## Metadata
- **Date:** 2026-10-21
- **Time:** ok. 05:30 – 10:30 (Europe/Warsaw), po restarcie komputera
- **Status:** Completed
- **Type:** Mixed (Analysis + Literature Research)

## Objective
- Zobaczyć, jak w trakcie treningu zmienia się pole wektorowe wokół poszczególnych pułapek koloidów (wiry, siodła), czyli to samo, co pokazuje aplikacja.
- Znaleźć fizyczne odpowiedniki chmury jako całości (Undermind, Scite).
- Ocenić, co w zachowaniu laserów nadaje się do publikacji.
- Odpowiedzieć, czy mikroskop latentu może pomóc w wykrywaniu narzędzia przykrytego tkanką oka.
- Wskazać, co i gdzie publikować, oraz sprawdzić punktację ministerialną.

## Context
Poprzednia część pracy jest w `Session_2026-10-21_045038` (commity `f622d3ff`, `03cc6920`). Obejmowała pary przeciwnych wiązek 0–200 epok, trening do 400 epok, dynamikę koloidów i laserów (eksperyment `koloidy_wiazki_dynamika_0_200_20261021__01-48-10`) oraz raport `SmallViTAblation/reports/Przeciwne wiązki w treningu ViT.md`. Potem komputer został wyłączony na prośbę użytkownika. Sesja-fork (`local_db305e03…`) czeka na decyzję użytkownika.

## Actions Taken
1. Odczyt definicji pola z `app/ogniskowa.py`, czyli f(u) = P3 Σ(z_c − z̄_c)v_c w przekroju P3, oraz zero pola szukane Newtonem w komórce tokenu.
2. `Kod_eksperymentu/pola_pulapek.py` (lokalnie, RTX 3070). Dla 30 koloidów narzędzia i 30 tła na klatkę, 12 klatek, 26 checkpointów:
   - klasa 3×3 w koloidzie: przyciąga / odpycha / siodło, wir;
   - zero pola w komórce;
   - 10 losowych przekrojów 3D;
   - pełny jakobian 192D: udział zespolonych wartości własnych, udział Re>0, stosunek obrót/odkształcenie.
3. `pulapki_newton_diag.py`: Newton bez ograniczenia komórki (epoki 0, 60, 200), czyli położenie zera pola.
4. `Wyniki/Code_Interpretujacy_Wyniki/pokaz_pola_pulapek.py`: tabela faz i trwałość klasy cząstek.
5. Deep research (skill deep-research): 3 wątki z Undermind (6 nowych deep searches) i Scite oraz raport `SmallViTAblation/reports/Fizyka chmury koloidów i okluzja.md`.
6. Uzupełnienie `Opis/OPIS_EKSPERYMENTU.txt` w eksperymencie koloidów.
7. Wyszukanie terminów konferencji (IPCAI, MICCAI, MIDL, ICML 2027) i stanu wykazu ministerialnego.

## Results

### Key Findings
- **Brak prawdziwych pułapek w komórce:**
  - |f(0)| / komórka: 3 (epoka 0) → 12–17 (epoki 60–200);
  - Newton zbiega dla 0,6% / 15–16% koloidów;
  - zero pola leży średnio 37 komórek (2–3,3 R) od koloidu, w komórce w 0% przypadków.
- **Fazy topologii w P3:**
  - krok 0: same siodła;
  - krok 300: narzędzie w 61% odpycha (wybuch);
  - epoki 3–70: przewaga przyciągania;
  - epoki 80–200: narzędzie przyciąga w 43–72%, tło to siodła w 69% → 86–95%.
- **Kontrole:**
  - w losowych przekrojach wiry występują w 41–59%, zanik wirów tła to artefakt P3;
  - pełne 192D: 87–93% zespolonych wartości własnych, ok. 45% Re>0, obrót/odkształcenie 1,00 → 0,85–0,90;
  - dywergencja narzędzia ok. −0,024, tła −0,013…−0,018.
- **Trwałość klasy:** po korekcie na przypadek (kappa, liczona w raporcie) jest słaba i podobna dla narzędzia i tła. Wcześniejsza teza „zamrożone siodła tła wobec migających pułapek narzędzia” jest wycofana.
- **Lasery (poprawka z raportu):** kierunki błądzą po sferze (kąt ∝ odstęp^0,4, 22° po 10 epokach, 66° po 140, zanik pamięci ok. 150 epok). Nie ma stałego obrotu w jedną stronę.
- **Chmura:** najlepiej pasuje stan stacjonarny przy stałym lr z rozdziałem jak w neural collapse. Szkło koloidalne jest niepotwierdzone. Trwałość sąsiedztwa spada przy stałym odstępie, a więc przebudowa przyspiesza, co jest odwrotnością starzenia.
- **Okluzja:** brak zbioru danych zaćmy z potwierdzonymi etykietami okluzji (Cataract-LMM ma klipy z zasłoniętą końcówką, ale bez etykiet). Największe zyski w literaturze daje syntetyczna okluzja w treningu.
- **Publikacje:** 3 artykuły (geometria wiązek, fizyka latentu, mikroskop z okluzją i filtrem cząsteczkowym) oraz publikacja o aplikacji.
  - IPCAI 2027: pełny artykuł do 30.10.2026, long abstract do 15.01.2027.
  - MICCAI 2027: termin nieogłoszony (w 2026 był 26.02).
- **Punktacja:**
  - nowe rozporządzenie z 30.04.2026;
  - nowy wykaz do końca 2026, podpisany w 2027, obowiązuje od publikacji z 2027;
  - punkty od nowa: czasopisma według percentyli Scopus/WoS, konferencje tylko z rankingiem ICORE;
  - IPCAI → IJCARS liczy się jako czasopismo;
  - TMLR: indeksowanie w Scopus/WoS niepotwierdzone;
  - warsztaty raczej bez punktów.

### Issues Encountered
- Newton w komórce nie znajdował zera. Diagnostyka bez ograniczenia pokazała, że to nie błąd, tylko zero leży daleko od komórki.
- Raport skorygował trzy moje wcześniejsze tezy (stały obrót laserów, trwałość klas, zanik wirów tła). Kilka liczb w raporcie („wyliczenie własne”) policzono jednorazowo, bez zapisanego skryptu, więc trzeba je odtworzyć przed artykułem.
- Portal ICORE blokuje automatyczne zapytania (403): rangi MICCAI, MIDL, IPCAI i IEEE VIS trzeba sprawdzić ręcznie.
- Dokładne punkty z wykazu z 5.01.2024 wymagają pobrania pliku XLSX (ok. 9,6 MB). Czeka na zgodę użytkownika.

## Files Generated/Modified
- `SmallViTAblation/experiments/koloidy_wiazki_dynamika_0_200_20261021__01-48-10/Kod_eksperymentu/pola_pulapek.py`, `pulapki_newton_diag.py`
- `.../Wyniki/Code_Interpretujacy_Wyniki/pokaz_pola_pulapek.py`
- `.../Wyniki/Dane/POLA_PULAPEK.csv`, `POLA_PULAPEK_CZASTKI.npz`
- `.../Opis/OPIS_EKSPERYMENTU.txt` (uzupełnienie o pole pułapek)
- `SmallViTAblation/reports/Fizyka chmury koloidów i okluzja.md`
- `SmallViTAblation/research_notes/Fizyka chmury koloidów i okluzja/` (physical_analogues_cloud.md, vortex_saddle_traps.md, occlusion_microscope.md)

## Commands Used
- `W:/Python/3.11/python.exe -I Kod_eksperymentu/pola_pulapek.py <klatki12> <wagi_historia>` (lokalny GPU)
- `W:/Python/3.11/python.exe -I Kod_eksperymentu/pulapki_newton_diag.py <klatki12> <wagi_historia>`
- skill deep-research (agenci z Undermind i Scite), WebSearch/WebFetch (terminy, wykaz ministerialny)

## Next Steps
- [ ] Zdecydować o IPCAI 2027: zgłoszenie zamiaru do 23.10, pełny artykuł do 30.10 albo long abstract do 15.01.2027
- [ ] Sprawdzić ręcznie rangi ICORE: MICCAI, MIDL, IPCAI, IEEE VIS; indeksowanie TMLR w Scopus/WoS
- [ ] (po zgodzie) Pobrać XLSX wykazu z 5.01.2024 i sprawdzić obecne punkty; po publikacji nowego wykazu (koniec 2026) wybrać miejsca
- [ ] Policzyć diagnostyki szkła (antykorelacja klatki, α2, test starzenia, χ4) i gałęzie z innym lr/batch
- [ ] Kontrole pola: przetasowanie kolumn W2, przekroje Schura, 1000+ losowych przekrojów, kontrola z wiązanymi wagami
- [ ] Odtworzyć zapisanym kodem „wyliczenia własne” z raportu
- [ ] Pilot okluzji: pary klatek z syntetycznym przykryciem, ok. 200 ręcznie oznaczonych klatek, kryterium AUROC, warianty filtra cząsteczkowego
- [ ] Replikacja wszystkich analiz na epokach 210–400 i na checkpointach 24 seedów
- [ ] Commit wyników tej sesji (`pola_pulapek`, raport, notatki)
