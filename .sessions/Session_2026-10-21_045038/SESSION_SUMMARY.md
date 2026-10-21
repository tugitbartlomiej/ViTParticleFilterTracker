# Session Summary: Przeciwne wiązki 0–200 epok, trening do 400 epok, dynamika koloidów i laserów

## Metadata
- **Date:** 2026-10-20 / 2026-10-21
- **Time:** ok. 22:00 – 04:50 (Europe/Warsaw)
- **Status:** Completed
- **Type:** Mixed (Analysis + SSH Eden + Training)

## Objective
- Wczytać kontekst poprzednich sesji (stary katalog konfiguracji `C:\Users\bartl\.claude`) i sprawdzić stan Edenu.
- Pobrać kontynuację treningu do 200 epok i zbadać poziom przeciwnych wiązek w ostatnich epokach, szczególnie w epoce 200.
- Zrobić wykresy, raport i krótki wniosek. Sprawdzić znoszenie pary 370–391 oraz IoU i pewność w epokach 0–200.
- Zapuścić trening do 400 epok.
- Zweryfikować wnioski w literaturze (Undermind, Scite) i przygotować wnioski do artykułu.
- Wyjaśnić metaforę sił, koloidów i rzutu 3D oraz przeanalizować blok 6 jako układ fizyczny (koloidy, lasery, pułapki) z korelacjami z IoU i temperaturą.
- Na koniec: commit, zapis sesji, wyłączenie komputera.

## Context
Claude Code przeniesiono 2026-10-20 do `W:\Chaty\Claude\CodeCLI` (`CLAUDE_CONFIG_DIR`); stare rozmowy zostały w `C:\Users\bartl\.claude\projects\...`. Poprzednia sesja (53cd72a2) wykonała pilot kanału Mar25l (H0 3/3 seedy) i przegląd literatury o przeciwnych neuronach. Inny agent uruchomił kontynuację seed 3047 do 200 epok (job 1907130). Część tej sesji zatwierdziła wcześniej sesja-fork (commit 0572525f).

## Actions Taken
1. Wczytanie 4 starych sesji i pamięci; skopiowanie pamięci o Pythonie na W: i dodanie notatki o lokalizacji starych transkryptów.
2. Stan Edenu: brak zadań w kolejce; katalog domowy 184,7 GB z 200 GB (soft), grupa 37,7 GB z 512 GB.
3. Pobranie 14 checkpointów epok 70–200 agentem na Sonnecie, z weryfikacją SHA256 i `DOWNLOAD_MANIFEST.json`.
4. Eksperyment `pary_przeciwne_do_200epok_20261020__23-21-30`:
   - liczenie par T150 na Edenie (job 1911518; kontrola z krokiem 65640 bitowo identyczna z rodzicem);
   - znoszenie pary 370–391, historia jej wag, ablacje na tle 300 par kontrolnych;
   - detekcja na 500 VAL + 500 TEST dla 26 checkpointów (smoke 1911914, pełny bieg 1911919);
   - 9 wykresów, RAPORT.html/.pdf, WNIOSEK.txt;
   - później: rzut 3D vs 192D, równowaga 768 sił (P3 z aplikacji), kopie neuronów, wejścia par T150 i ich los po epoce 110.
5. Eksperyment `vit_kontynuacja_do_400epok_20261020__23-50-57`: pełne wznowienie z k218750 (preflight PASS). Job 1911922 ukończony: krok 437500, 6779 s, 21 checkpointów na Edenie.
6. Deep research (skill deep-research): 4 wątki (nasze dowody, pary/kopie/weight decay, GELU/rzadkość/superpozycja, dynamika/kalibracja/ablacja) z Undermind (nowe deep searches) i Scite. Raport `reports/Przeciwne wiązki w treningu ViT.md` z poprawkami koordynatora.
7. Eksperyment `koloidy_wiazki_dynamika_0_200_20261021__01-48-10`, liczony lokalnie na RTX 3070, 26 checkpointów, 12 klatek: metryki komory, ruch, MSD, kopnięcie a rozdział, pojedyncze lasery, korelacje z IoU i temperaturą, migające lasery.
8. Commit wyników tej sesji (bez plików ignorowanych `*.txt`, `*.npz`).

## Results

### Key Findings
- **Mechanizm przeciwnych wiązek:** żadna para kolumn W2 nie jest antyrównoległa (maks. 142°). Każda para ≥150° to prawie równoległe W2 plus przeciwne znaki GELU. Pary T150 to częściowe kopie (mediana cos W1 0,75–0,81).
- **Cykl życia par:** 0 → 2 (ep. 9) → 25 (ep. 40) → 30 (ep. 80–110) → 15 (ep. 200). Po epoce 110 zniknęło 20 z 30 par: rozeszły się (kąt W2 28° → 40°, cos W1 0,77 → 0,66). W pełną kopię scaliła się tylko para 370–391.
- **Para 370–391:** cos W1 0,96 i kąt W2 7° w epoce 200. Wkłady dodają się na 94% tokenów, częściowo znoszą na 6% (κ 0,19 przy obu aktywnych). Wyłączenie zmienia IoU o ok. 1e-5, w rozrzucie kontroli; iloraz interakcji 1,66 (efekty się sumują).
- **Detekcja:** IoU stoi od ok. epoki 40; pewność rośnie, także na pustych klatkach (VAL 0,63 → 0,68).
- **Rzut 3D:** par przeciwnych na poziomie przypadku jest ok. 13,5% wobec 0–9 w 192D. Wypadkowa w P3 jest dobrze widoczna (47–83% długości); zbiorowa równowaga jest zniekształcona o 20–90%, ze zmiennym znakiem.
- **Kanały Mar25l:** 0 par „kanałowych”; pełne kopie tylko 0–3 w całej sieci.
- **Koloidy:**
  - rozdział faz do 96–98%;
  - prędkość spada ok. 100× do epoki 30, potem jest stała;
  - MSD sub-dyfuzyjne;
  - wymiar chmury 10,6 → 3,2;
  - dywergencja pola ujemna i rosnąca;
  - wiatr 0,58 R plus hamulec −11–13% rozdziału od ok. epoki 80.
- **Lasery:**
  - kolimacja 154 → 31 kierunków;
  - obrót 2,2°/epokę od ep. 50;
  - moc 0,29 → 0,92;
  - lasery tła do ok. 30%, lasery narzędzia maleją 3× po szczycie;
  - elita wymienia ok. 1/3 na 10 epok;
  - połowę hamowania robi 10–19 laserów.
- **Korelacje:** na całym treningu to artefakt trendu. Na plateau temperatura nie wiąże się z IoU. Liczba ciemnych laserów koreluje ze zmianami IoU: r −0,69 (VAL) i −0,79 (TEST); lasery narzędzia +0,61/+0,68. Wynik wstępny (72 testy, 15 zmian, 1 seed).

### Issues Encountered
- Cytowanie w PowerShellu przy ssh: rozwiązane skryptami bash wgrywanymi przez scp (LF, bez BOM).
- Zerwane połączenia ssh (exit 255): ponowienia; job 1911919 zgłoszony drugi raz po sprawdzeniu kolejki.
- Restore checkpointu 200 epok wymagał innego hasha konfiguracji: `restore()` przyjmuje hash rodzica tylko dla źródła.
- `python -I` ignoruje PYTHONIOENCODING: `sys.stdout.reconfigure(encoding='utf-8')`.
- Brak `node` (walidator palety): użyto zwalidowanej palety referencyjnej.
- Moje wcześniejsze tezy poprawione na danych: kopie są rzadkie; spadek par po ep. 110 to rozchodzenie, nie scalanie; migające lasery to nie specjaliści od narzędzia.
- git nie ma w PATH (`W:\Instalki\Git\cmd\git.exe`) ani tożsamości w środowisku CLI: użyto tożsamości z poprzednich commitów (`-c user.name/email`).
- `.gitignore` wyklucza `*.txt` i `*.npz`: opisy `Opis/*.txt` i pliki npz nie trafiają do repozytorium.

## Files Generated/Modified
- `SmallViTAblation/experiments/pary_przeciwne_do_200epok_20261020__23-21-30/` (kod, dane, 9 wykresów, raport, wniosek, analizy 3D i kopii)
- `SmallViTAblation/experiments/vit_kontynuacja_do_400epok_20261020__23-50-57/` (kod, opis; checkpointy na Edenie)
- `SmallViTAblation/experiments/koloidy_wiazki_dynamika_0_200_20261021__01-48-10/` (kod, dane, opis)
- `SmallViTAblation/experiments/vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/` (pobrane checkpointy, logi, manifest)
- `SmallViTAblation/experiments/experiments_index.md`
- `reports/Przeciwne wiązki w treningu ViT.md`, `research_notes/Przeciwne wiązki w treningu ViT/*.md`
- pamięć: `W:\Chaty\Claude\CodeCLI\config\projects\...\memory\` (python-install-location, old-session-transcripts-location)

## Commands Used
- `ssh eden "bash /tmp/<skrypt>.sh"`, `scp -q`, `sbatch` (hopper, gpu:h100:1), `squeue`, `sacct`, `lfs quota`
- `W:\Python\3.11\python.exe -I <skrypt>` (analizy lokalne, RTX 3070)
- headless Edge `--print-to-pdf` (RAPORT.pdf)
- `W:\Instalki\Git\cmd\git.exe add/commit`

## Next Steps
- [ ] Replikacja analiz (pary, koloidy, lasery, korelacje, IoU) na epokach 210–400 (checkpointy na Edenie)
- [ ] Liczenie T150 na checkpointach 24 seedów (epoki 0–60) i 2–3 dodatkowe przebiegi do 200 epok
- [ ] Model zerowy dla liczby par; pełny pakiet ablacyjny (mean/mode, efekt bezpośredni, Δ_ij z CI)
- [ ] Test drużyny hamulcowej: wyłączyć 10–19 laserów hamujących i zmierzyć IoU
- [ ] Przebieg z wd = 0 (kontynuacja z epoki 60)
- [ ] Przeczytać Kwon & Kim 2005, Guo et al. 2017, Fukumizu & Amari 2000
- [ ] Wykresy dla eksperymentu koloidów (nie zamówione)
