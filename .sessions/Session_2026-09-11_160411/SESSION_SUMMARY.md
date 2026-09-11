# Session Summary: Session_2026-09-11_160411

## Metadata
- **Date:** 2026-09-11
- **Time:** 13:35 – 16:04
- **Status:** Completed
- **Type:** Mixed (refaktor narzędzi + porządkowanie archiwum literatury)

## Objective

Trzy cele, kolejno stawiane przez użytkownika:

1. Założyć folder na PDF-y o przetwarzaniu danych przed treningiem sieci i wyciągnąć
   ze skryptów JEPA ogólną formę: zbieranie linków dla dowolnego folderu i pobieranie
   ich tym samym narzędziem.
2. Podpiąć pod te narzędzia istniejący zbiór `2026-09-11_RT-DETR_selekcja_ewaluacja`
   i pobrać do niego literaturę.
3. Przeskanować archiwum `F:\Studia\Articles\Zbiory`, zaplanować refaktor struktury
   i pracy, a potem go wykonać — w tym rozdzielić katalog `Ocr` na surowe wyniki
   i opisy prac, z indeksem.

## Context

Archiwum literatury rosło przez rok bez przyjętej struktury. Kod pobierania PDF-ów
żył wyłącznie w `JEPA\Scripts\PdfDownloader`, ze ścieżkami JEPA wpisanymi w
`config.json`. Ten sam wzorzec powtórzył się w OCR: inna sesja skopiowała
`scan_jepa.py` do zbioru RT-DETR jako `scan_rtdetr.py`, zmieniając wyłącznie ścieżki.
Każdy nowy temat oznaczał kolejną kopię kodu i poprawkę w kilku miejscach.

Przez większość sesji równolegle pracowała druga sesja na tym samym zbiorze RT-DETR
(pisała audyty, budowała bibliografię BibTeX, uruchomiła OCR, przeniosła pliki `.bat`
do `scripts\`). To wymusiło ciągłe sprawdzanie stanu na dysku przed każdą zmianą.

## Actions Taken

### 1. Uogólnienie PdfDownloadera (`Zbiory\_Narzedzia\PdfDownloader`)

- Skopiowanie 8 skryptów z `JEPA\Scripts\PdfDownloader` (pobieranie, raport nieudanych,
  kolejka Primo, Sci-Hub).
- Napisanie modułu `zbior.py`: ustala katalog zbioru w kolejności
  `--zbior` → zmienna `ZBIOR_PDF` → plik `.zbior` → katalog skryptu. Ostatni krok
  zachowuje stare zachowanie, więc układ JEPA działa bez żadnej zmiany.
- Przepisanie `BASE` we wszystkich skryptach na katalog zbioru zamiast katalogu kodu.
- Napisanie `nowa_lista.py` — zbieracz linków: normalizuje gołe DOI i `arXiv:`,
  rozpoznaje alphaXiv, OpenReview, PMC, obcina parametry śledzące, odsiewa powtórki
  po DOI lub identyfikatorze arXiv we wszystkich listach zbioru.
- Napisanie `nowy_zbior.py` — generator zbioru: katalogi, `config.json`, cienkie `.bat`-y,
  README.
- Dokumentacja: `README.md` i `ZBIERANIE_LINKOW.md` (uogólniona konwencja list linków,
  przepisana ze skilla `paper-search`).

### 2. Dwa nowe zbiory

- `Zbiory\DataPreprocessing` — na temat przetwarzania danych i obróbki przed treningiem.
- `Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja` — podpięcie istniejącego katalogu;
  dodana opcja `--folder-pdf`, bo użytkownik chciał `pdf_all` małymi literami.

### 3. Pobieranie literatury do zbioru RT-DETR

- Wciągnięcie dwóch istniejących plików z linkami do `linki\`.
- Pobranie: 18 prac w pierwszym przebiegu, potem uzupełnienia.
- Cztery adresy padły (strona HTML zamiast pliku albo ochrona antybotowa). Każdy
  zastąpiony działającym źródłem: JMLR i CVF bezpośrednim PDF-em, praca Kapoora
  i Narayanana preprintem z arXiv (Cell i PMC odrzucają skrypt). Stare adresy
  wyłączone komentarzem z datą i powodem, nie skasowane.
- Primo okazało się niepotrzebne — wszystkie prace były otwarte.

### 4. Skan całego archiwum

Napisany `skan_zbiory.py` (scratchpad): 82 931 plików, 18,6 GB, 9 zbiorów,
25 grup duplikatów po treści (MD5), rozkład typów i rozmiarów per zbiór.

### 5. Plan refaktoru i jego wykonanie

Plan zapisany w `_Narzedzia\PLAN_REFAKTORU.md`: 8 nazwanych problemów, struktura
docelowa, konwencje, 7 etapów od najbezpieczniejszego, osobna sekcja o refaktorze
sposobu pracy, lista rzeczy do nieruszania.

Wykonane etapy:
- **P1** — usunięty `mineru.sif` (7,3 GB) na polecenie użytkownika, z notatką
  `mineru.sif.USUNIETY.md` opisującą odtworzenie.
- **P8** — skrypty OCR do `_Narzedzia\OcrMinerU` jako `skanuj_ocr.py` i `postep_ocr.py`;
  `zbior.py` przeniesiony poziom wyżej jako moduł wspólny dla wszystkich narzędzi,
  z funkcjami `konfiguracja()`, `ustawienie()`, `folder_pdf()`.
- **Etap 2 w RT-DETR** — notatki do `notatki\`, produkty pochodne do `pochodne\`,
  duplikaty PDF-ów do `pochodne\duplikaty_usuniete\`.
- **P5** — deduplikacja `pdf_all` po treści: zostały nazwy downloadera, zdublowane
  katalogi OCR usunięte.
- **Rozdzielenie `Ocr`** — `raw\` (wynik MinerU), `sources\` (opis jednej pracy na plik),
  `index.md` (katalog całości). Napisany `zbuduj_indeks.py`.
- **Dokończenie `scripts\`** — refaktor rozpoczęty przez drugą sesję zepsuł obieg;
  naprawione wskazania `--zbior "%~dp0..\.."`, dane robocze wróciły do korzenia.

## Results

### Key Findings

**Skan archiwum (stan wyjściowy):**

| co | ile |
|---|---|
| pliki / rozmiar | 82 931 / 18,6 GB |
| `JEPA` | 82 550 plików, 18,5 GB (99,5% całości) |
| `mineru.sif` | 7,3 GB — obraz kontenera, czyli program w folderze danych |
| `Merged_*_Smart` | 4,4 GB sklejek PDF odtwarzalnych w minutę |
| `JEPA\Ocr` | 79 095 plików JPG — obrazy wycięte ze stron |
| duplikaty | 25 grup, 135 MB |

**Zbiór RT-DETR po refaktorze:** 33 linki w trzech listach, 31 PDF-ów, 31 wyników OCR,
31 opisów prac, zero nieudanych pobrań, zero do pobrania.

**Jakość opisów prac:** 28 z 31 ma zdanie „po co ta praca tu jest", złożone
z `ZRODLA.md` albo z notki przy adresie w kolejce linków. Trzy pozostałe czekają
na uzupełnienie — skrypt świadomie nie zmyśla.

### Issues Encountered

| problem | rozwiązanie |
|---|---|
| `DETR\nul` — nazwa zarezerwowana przez Windows, wywala `os.walk` | pominięty w skanie z komunikatem; wpisany do planu jako etap 0 |
| ścieżka względna listy liczona od katalogu kodu, nie zbioru | `rozwiaz_liste()` w `pobierz_pdfy.py`: bezwzględna → zbiór → folder linków → cwd |
| `primo_z_nieudanych.py` wywalał się na świeżym zbiorze | brak raportu nieudanych to normalny stan, nie błąd — zwraca pustą kolejkę |
| `--zbior` trafiał do argumentów pozycyjnych OCR jako folder wejściowy | wywołanie `katalog_zbioru()` przeniesione przed odczyt `sys.argv` |
| duplikaty PDF-ów z dwóch konwencji nazw (dwie sesje pobierały równolegle) | deduplikacja po MD5, zostaje nazwa downloadera |
| druga sesja przeniosła `.bat`-y do `scripts\`, psując obieg | wskazania na `%~dp0..\..`, dane robocze z powrotem do korzenia |
| heredoc w Bashu gubił `\r`, `\a`, `\o` w ścieżkach Windows | skrypty pomocnicze pisane przez Write, nie przez heredoc |
| Pyright zgłaszał brak `zbior` i `mineru` | fałszywe alarmy — moduł leży obok, pakiety są w środowisku MinerU |

### Decyzje wymagające uwagi

- **Język OCR** — wszystkie skany szły z `ocr_jezyk = ch`, choć korpus jest angielski
  (konfiguracja wdrożenia na Eden wybiera `en`). Zostawione `ch`, żeby nowe wyniki
  nie rozjechały się ze starymi. Zmiana wymaga ponownego OCR całości — osobny etap 7 planu.
- **Bibliografia BibTeX** w `pochodne\bibtex\` powstała przed deduplikacją i zna
  starą nazwę `COWAL_WACV2024.pdf`. Do odbudowania.
- **Skill `paper-search`** ma nadal kolejkę JEPA wpisaną na sztywno.

## Files Generated/Modified

### Nowe narzędzia (`F:\Studia\Articles\Zbiory\_Narzedzia\`)

```
zbior.py                          moduł wspólny: katalog zbioru + konfiguracja
PLAN_REFAKTORU.md                 plan refaktoru archiwum, z odhaczonymi etapami
PdfDownloader\
  README.md, ZBIERANIE_LINKOW.md  dokumentacja
  nowy_zbior.py                   generator zbioru tematycznego
  nowa_lista.py                   zbieracz linków
  pobierz_pdfy.py                 przepisany: BASE = katalog zbioru, rozwiaz_liste()
  odswiez_nieudane.py, sprawdz_nieudane_pdfy.py,
  primo_z_nieudanych.py, sprawdz_kolejke.py
  primo\, scihub\
OcrMinerU\
  skanuj_ocr.py                   z scan_jepa.py, ścieżki i język z konfiguracji
  postep_ocr.py                   z check_progress.py
  zbuduj_indeks.py                buduje Ocr\sources\ i Ocr\index.md
```

### Zbiór RT-DETR (`Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\`)

```
README.md, AGENTS.md              przepisane pod nowy układ
Ocr\README.md, Ocr\index.md       nowe
Ocr\raw\        31 katalogów      przeniesione z Ocr\
Ocr\sources\    31 opisów         nowe
notatki\        6 plików .md      przeniesione z korzenia
pochodne\       evidence, search_logs, papers, bibtex, duplikaty_usuniete
scripts\pobieranie\, scripts\ocr\ pliki .bat, poprawione wskazania
linki\          3 listy, 33 linki
pdf_all\        31 PDF-ów
config\config.json                + ocr_jezyk
```

### Pozostałe

- `Zbiory\DataPreprocessing\` — nowy zbiór, gotowy do użycia
- `JEPA\Scripts\OcrMinerU\eden\mineru.sif` — **usunięty** (7,3 GB)
- `JEPA\Scripts\OcrMinerU\eden\mineru.sif.USUNIETY.md` — notatka o odtworzeniu
- `~\.claude\projects\...\memory\pdfdownloader-wspolne-narzedzia.md` — pamięć projektu

## Commands Used

```powershell
# nowy zbiór tematyczny
py -3.11 _Narzedzia\PdfDownloader\nowy_zbior.py "<katalog>" `
    --opis "..." --folder-pdf pdf_all --ocr-jezyk ch --email <adres>

# zbieranie linków
py -3.11 _Narzedzia\PdfDownloader\nowa_lista.py --zbior "<zbiór>" `
    --nowa "Temat" --po-co "..."
py -3.11 ...\nowa_lista.py --zbior "<zbiór>" --do <lista> --sekcja "..." --link "URL # opis"
py -3.11 ...\nowa_lista.py --zbior "<zbiór>" --pokaz

# pobieranie
py -3.11 _Narzedzia\PdfDownloader\pobierz_pdfy.py --zbior "<zbiór>" [--sucho|--przeindeksuj]

# OCR
py -3.11 _Narzedzia\OcrMinerU\postep_ocr.py --zbior "<zbiór>" --krotko
F:\OCR\MinerU\.venv\Scripts\python.exe _Narzedzia\OcrMinerU\skanuj_ocr.py --zbior "<zbiór>"

# indeks prac
py -3.11 _Narzedzia\OcrMinerU\zbuduj_indeks.py --zbior "<zbiór>" [--odswiez|--sucho]

# w samym zbiorze (pliki wsadowe)
scripts\pobieranie\zbierz_linki.bat --pokaz
scripts\pobieranie\pobierz_pdfy.bat
scripts\ocr\skanuj_ocr.bat
```

## Next Steps

- [ ] **Etap 0 planu** (~5 min): skasować `DETR\nul`, kopie `.bak`, 20 starych raportów
      deduplikacji, uporządkować 4 PDF-y luzem w korzeniu `Zbiory`
- [ ] **Etap 1 reszta**: `PdfDeduplicator` i `Merge_PDF_NotebookLM` do `_Narzedzia\`;
      `JEPA\Scripts` odchudzić do samych danych
- [ ] **Etap 2**: `Merged_PDFs_Smart` i `Merged_Books_Smart` (4,4 GB) do `JEPA\pochodne\`
      albo skasować — odtwarzalne w minutę
- [ ] **Etap 3**: ujednolicić `PDF_all` → `pdf_all` w JEPA i DataPreprocessing
- [ ] **Etap 4**: deduplikacja całego drzewa naraz; katalogi `<tytuł>-1` w `JEPA\Ocr`
- [ ] **Etap 5**: `DETR`, `TimesFormer`, `ViT_Med`, `Dino` przez `nowy_zbior.py`
      i odtworzenie list linków z nazw plików
- [ ] **Etap 6**: 79 tysięcy JPG z `JEPA\Ocr` poza główne drzewo albo do archiwów ZIP
- [ ] **Etap 7 (decyzja)**: język OCR `ch` → `en` i ponowny OCR całości
- [ ] Uzupełnić trzy opisy prac bez zdania „po co tu jest" w `Ocr\sources\`
- [ ] Odbudować bibliografię BibTeX po deduplikacji
- [ ] Rozważyć uogólnienie skilla `paper-search` (ma kolejkę JEPA na sztywno)
