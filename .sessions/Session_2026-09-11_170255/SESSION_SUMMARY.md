# Session Summary: Session_2026-09-11_170255

## Metadata
- **Date:** 2026-09-11
- **Time:** 16:04 – 17:02
- **Status:** Completed
- **Type:** Mixed (naprawa i uzupełnienie narzędzi pobierania literatury)

## Objective

1. Rozstrzygnąć, co zrobić z 20 nieudanymi pobraniami w zbiorze RT-DETR.
2. Odpowiedzieć, gdzie w `scripts\pobieranie` jest ścieżka Primo, wzorując się
   na `JEPA\Scripts\PdfDownloader\primo`.
3. Zrobić z pobierania przez zalogowany Edge **osobną metodę**, zamiast
   instrukcji do wyklikania.

## Context

Ciąg dalszy `Session_2026-09-11_160411`, w której skrypty pobierania i OCR
zostały wyjęte z pojedynczego zbioru JEPA i zamienione we wspólne narzędzia.
Uogólnienie okazało się niepełne: przeniosłem `pobierz_pdfy_primo.py`
i `przenies_pobrane.py`, ale **nie** `primo_linki.ps1` — pliku PowerShell,
który zamienia wyszukiwanie w Primo na adres pełnego tekstu. Generowany
`otworz_partie.ps1` ładował go z własnego katalogu, gdzie go nie było, więc
cała ścieżka instytucjonalna była martwa.

Równolegle druga sesja dopisała trzy nowe listy linków (Koopman i DMD,
eksport ResearchRabbit, ważność próbki), z których 20 pozycji nie dało się
pobrać automatem.

## Actions Taken

### 1. Rozpoznanie 22 nieudanych pozycji

- Sprawdzenie każdego DOI w OpenAlex pod kątem `best_oa_location`
  (skrypt `rekonesans_oa.py` w scratchpadzie).
- Wynik: **zero kopii otwartych**. To zamknięte DOI wydawców IEEE, Elsevier,
  APA i ACM, plus klasyka psychologii poznawczej z eksportu ResearchRabbit
  (Ratcliff 1990, McClelland 1995, Rumelhart 1986).
- Wniosek: automat ich nie weźmie, jedyna droga to dostęp instytucjonalny.

### 2. Naprawa ścieżki Primo

- `primo_linki.ps1` i `ciasteczka.przyklad.txt` skopiowane do
  `_Narzedzia\PdfDownloader\primo\`.
- Rozdzielenie kodu od danych w `primo_linki.ps1`: nowa zmienna
  `$PrimoKatalogDanych`, ustawiana przez wołający skrypt. Cache rozwiązanych
  adresów, ostatnia partia i ciasteczka należą do zbioru, nie do katalogu
  z kodem. Bez ustawienia zostaje stare zachowanie, więc JEPA działa dalej.
- `reguly_pdf.json` szukane najpierw w zbiorze, potem w narzędziach.
- `primo_z_nieudanych.py` generuje `otworz_partie.ps1`, który ładuje kod
  z bezwzględnej ścieżki narzędzi, z zapasem na stary układ.
- Trzy launchery jako szablony w `nowy_zbior.py`:
  `primo_otworz_partie.bat`, `primo_pobierz.bat`, `primo_przenies_pobrane.bat`.
- `ciasteczka.przyklad.txt` przepisany: doszła instrukcja krok po kroku, jak
  zebrać ciasteczka z F12, i wyjaśnienie po co one są.

### 3. Próba pobrania i ustalenie ograniczenia

- `primo_otworz_partie.bat rozwiaz` — Primo rozwiązało **10 z 22** pozycji,
  w tym 5 z bezpośrednim adresem PDF. Dwunastu baza PW nie zna (podręczniki,
  notatki wykładowe, prace spoza bazy — `primo\bez_wynikow.txt`).
- `browser_cookie3` na Edge: **`This operation requires admin`**. Edge od
  wersji 127 używa App-Bound Encryption, więc ciasteczek nie da się odczytać
  z dysku bez uprawnień administratora. Potwierdzenie tego, co przewidział
  autor `pobierz_pdfy_primo.py`.
- Otwarcie 5 kart z PDF-ami w Edge — nic nie wpadło do Pobranych, bo Edge
  wyświetlał pliki w karcie.

### 4. Nowa metoda: `przez_edge.py` + `przez_edge.bat`

Jedno wywołanie zamiast czterech kroków:

1. sprawdza `plugins.always_open_pdf_externally` w profilu Edge — bez tego
   przeglądarka wyświetla PDF zamiast go zapisać;
2. odczytuje `download.default_directory`, czyli **dokąd Edge naprawdę
   pobiera** — u użytkownika okazało się to `pdf_all` zbioru, a nie Pobrane;
3. otwiera partię kart (`otworz_partie.ps1 -TylkoPdf`);
4. czeka, aż pobieranie ucichnie (pilnuje `.crdownload`);
5. nadaje nazwy w konwencji downloadera.

Tryb `--ustaw-pobieranie` włącza zapisywanie PDF w Edge, ale **odmawia przy
działającej przeglądarce** — Edge nadpisuje `Preferences` przy wyjściu.
Kopia poprzedniego pliku zostaje jako `.przed-zmiana`.

### 5. Nazywanie plików po faktach, nie po kolejności

Cztery drogi dopasowania pobranego pliku do pozycji w kolejce:

| droga | przykład |
|---|---|
| PII Elseviera w nazwie | `1-s2.0-S1364661399012942-main.pdf` → `10.1016/S1364-6613(99)01294-2` |
| końcówka DOI w nazwie | `978-3-642-88163-3.pdf` → `10.1007/978-3-642-88163-3` |
| nazwa pliku = tytuł | `Exploiting_Typicality_for_...pdf` → `10.1109/TIP.2019.2910634` |
| tytuł z metadanych PDF | `getPDF.pdf` → `10.1109/ACCESS.2018.2825106` |

Czego nie da się pewnie powiązać, zostaje pod nazwą wydawcy i trafia do
raportu. Zgadywanie po kolejności pobrania podpisałoby pracę cudzym DOI.

## Results

### Key Findings

- **Ścieżka Primo była martwa** od momentu uogólnienia narzędzi — brakowało
  jednego pliku PowerShell. Wyszło dopiero przy próbie użycia.
- **Żaden z 22 zamkniętych DOI nie ma kopii otwartej.** Sprawdzone w OpenAlex,
  nie zgadywane.
- **Ciasteczek Edge nie da się odczytać** bez uprawnień administratora
  (App-Bound Encryption od wersji 127). To zamyka drogę „skrypt z sesją" i
  wymusza obieg przez samą przeglądarkę.
- **Edge pobierał do `pdf_all`, nie do Pobranych.** Dlatego pierwsze
  sprawdzenie pokazało pustkę. Skrypt czyta teraz katalog z ustawień, zamiast
  go zakładać.
- Kolejka nieudanych: **22 → 18**.

### Issues Encountered

| problem | rozwiązanie |
|---|---|
| `otworz_partie.ps1` ładował nieistniejący `primo_linki.ps1` | kod do narzędzi, ścieżka bezwzględna w generatorze, zapas na stary układ |
| `primo_linki.ps1` zapisywałby cache do katalogu z kodem | zmienna `$PrimoKatalogDanych` ustawiana przez wołający skrypt |
| `browser_cookie3` wymaga administratora na Edge | droga przez otwieranie kart w zalogowanej przeglądarce |
| Edge wyświetlał PDF zamiast zapisywać | odczyt i ustawianie `always_open_pdf_externally` |
| `przenies_pobrane.py` uznał wszystko za duplikat | Edge pobierał prosto do celu — osobna gałąź „zmiana nazw w miejscu" |
| `rozwiazane.json` ma sygnaturę BOM | odczyt z `utf-8-sig` |
| heredoc w Bashu gubił `\r`, `\a` w ścieżkach Windows | skrypty pomocnicze przez Write, nie przez heredoc |

### Uwaga dla użytkownika

Edge pobiera teraz **wszystko** do `pdf_all` zbioru RT-DETR, także prace
niezwiązane z tematem. Leży tam sześć takich plików (mykotoksyny, aflatoksyny,
kontekstowość kwantowa, preferencje, sieci Hopfielda, szum neuronalny).
Skrypt ich nie ruszył, bo nie potrafi ich powiązać z kolejką.

## Files Generated/Modified

### Wspólne narzędzia (`F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\`)

```
primo\primo_linki.ps1          NOWY (z JEPA) + rozdzielenie kodu od danych
primo\ciasteczka.przyklad.txt  NOWY (z JEPA) + instrukcja krok po kroku
primo\przez_edge.py            NOWY — cały obieg przez zalogowaną przeglądarkę
primo_z_nieudanych.py          generuje ps1 ładujący kod z narzędzi
nowy_zbior.py                  + 4 szablony .bat (3 × Primo, 1 × przez Edge)
```

### Zbiór RT-DETR

```
scripts\pobieranie\primo_otworz_partie.bat      NOWY
scripts\pobieranie\primo_pobierz.bat            NOWY
scripts\pobieranie\primo_przenies_pobrane.bat   NOWY
scripts\pobieranie\przez_edge.bat               NOWY
README.md                                       + punkty 7 i 8 (Primo, przez Edge)
primo\otworz_partie.ps1, rozwiazane.json, bez_wynikow.txt   przebudowane
pdf_all\  + 4 prace zza paywalla pod nazwami doi_*
```

### Repozytorium projektu

Bez zmian — cała praca odbywała się w `F:\Studia\Articles\Zbiory`,
poza repozytorium. W repo powstały tylko katalogi sesji w `.sessions\`.

## Commands Used

```powershell
# ścieżka Primo
scripts\pobieranie\primo_otworz_partie.bat rozwiaz     rozwiąż kolejkę bez kart
scripts\pobieranie\primo_otworz_partie.bat pdfy 1 25   otwórz karty z PDF-ami
scripts\pobieranie\primo_pobierz.bat sucho             plan pobierania

# przez Edge (nowa metoda)
scripts\pobieranie\przez_edge.bat sprawdz              diagnoza ustawień
scripts\pobieranie\przez_edge.bat ustaw                włącz zapisywanie PDF
scripts\pobieranie\przez_edge.bat                      cały obieg
scripts\pobieranie\przez_edge.bat posprzataj           sam porządek nazw

# odświeżenie kolejki po pobraniu
py -3.11 _Narzedzia\PdfDownloader\odswiez_nieudane.py --zbior "<zbiór>"
```

## Next Steps

- [ ] Otworzyć 5 pozycji typu „karta rekordu" z kolejki Primo i ściągnąć je
      ręcznie — `przez_edge.bat` posprząta nazwy
- [ ] 12 pozycji spoza bazy PW: albo szukać ręcznie, albo uznać, że wystarczy
      wpis w liście linków z notką, po co praca była rozważana
- [ ] Rozważyć zmianę katalogu pobierania Edge z powrotem na Pobrane —
      teraz wszystko ląduje w `pdf_all` zbioru RT-DETR
- [ ] OCR dla prac pobranych po ostatnim przebiegu (`scripts\ocr\skanuj_ocr.bat`),
      potem `zbuduj_indeks.py`
- [ ] Pozostałe etapy planu refaktoru z `_Narzedzia\PLAN_REFAKTORU.md`
      (etap 0, reszta etapu 1, etapy 2–7)
