# Session Summary: Deduplikator PDF - przywrócenie plików i naprawa skryptu

## Metadata
- **Date:** 2026-09-11
- **Time:** 17:05:17
- **Status:** Completed
- **Type:** Mixed (narzędzia + weryfikacja danych)
- **Poprzednia sesja tego dnia:** `Session_2026-09-11_160738` (selekcja danych wg literatury)

## Objective
Użytkownik uruchomił świeżo napisany deduplikator PDF-ów i zobaczył trzy komunikaty
`PRZENOSZE DO KOSZA`. Zadanie: sprawdzić, czy to naprawdę były duplikaty, przywrócić
pliki, pokazać dowód i poprawić skrypt.

Cytat polecenia:
> "sprawdz czy aby na pewno to jest to samo przywróc i pokkaz mi to w jakims
> poprawiaj ten sktypt usuwający duplikaty!!!!"

## Context
W sesji `Session_2026-09-11_160738` powstał zbiór literatury
`F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja` z ponad setką PDF-ów
pobranych automatycznie z kilku list linków. Część prac trafiła tam dwa razy pod różnymi
nazwami (`2404.09161.pdf` z jednej listy, `arxiv_2404.09161.pdf` z drugiej), więc
na prośbę użytkownika dopisałem skrypt deduplikujący wzorowany na
`F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDeduplicator`.

Pierwsza wersja miała wadę projektową: **usuwanie było zachowaniem domyślnym**,
a podgląd wymagał flagi `--dry-run`. Uruchomienie bez flag od razu przeniosło pliki
do Kosza, a komunikat pokazywał samą nazwę pliku — nie było z niego widać, że
identyczna praca zostaje obok pod inną nazwą.

## Actions Taken
1. Sprawdziłem stan `pdf_all` — liczba PDF-ów spadła o trzy, bliźniaki nietknięte.
2. Odczytałem raport `raport_duplikatow_2026-09-11_16-16-22.csv` z uruchomienia
   użytkownika i zestawiłem pary usunięty / pozostawiony.
3. Przywróciłem trzy pliki z Kosza Windows przez `Shell.Application`
   (przestrzeń `0xA`, czasownik `Przywróć`/`Restore`).
4. Potwierdziłem identyczność już po przywróceniu: ponownie policzony SHA-256
   oraz `filecmp.cmp(a, b, shallow=False)` dla każdej pary.
5. Przebudowałem skrypt (opis w sekcji „Results").
6. Uruchomiłem poprawiony skrypt bez flag na całym zbiorze — 128 PDF-ów, 5 grup
   duplikatów, zero zmian na dysku.

## Results

### Dowód: usunięte pliki naprawdę były kopiami
| Usunięty (przywrócony) | Pozostawiony | Rozmiar | SHA-256 (początek) | `filecmp` |
|---|---|---|---|---|
| `arxiv_2404.09161.pdf` | `2404.09161.pdf` | 1 123 964 B | `b83d97e675e6ea51` | identyczne |
| `arxiv_2407.17140.pdf` | `2407.17140.pdf` | 88 744 B | `3b712f4d8255e75d` | identyczne |
| `arxiv_2507.00049.pdf` | `2507.00049.pdf` | 30 095 512 B | `bb914620a4946acb` | identyczne |

Tytuły potwierdzają, że to te same prace: *Coreset Selection for Object Detection*,
*RT-DETRv2: Improved Baseline with Bag-of-Freebies*, *AdaDeDup: Adaptive Hybrid Data
Pruning for Efficient Large-Scale Object Detection Training*. Żadna praca nie zniknęła
ze zbioru — zmieniła się tylko liczba kopii.

### Zmiany w skrypcie
1. **Usuwanie wymaga flagi `--usun`.** Uruchomienie bez flag (także dwuklik na `.bat`)
   jest wyłącznie podglądem z raportem. `--dry-run` nadal jest przyjmowane, ale nic
   nie zmienia — stare polecenia dalej działają.
2. **Tytuł z pierwszej strony przy każdym pliku idącym do Kosza**, razem z nazwą
   bliźniaka, który zostaje. Sam skrót mówi tylko „te same bajty"; tytuł zamienia
   komunikat w zdanie, które da się sprawdzić wzrokiem. Tytuł brany jest z metadanych,
   a gdy tych brak — z tekstu pierwszej strony (z odcięciem nagłówka `arXiv:...`).
3. **Powtórne porównanie bajt po bajcie** (`filecmp.cmp(shallow=False)`) tuż przed
   przeniesieniem do Kosza. Skróty liczone są na początku przebiegu, więc ta kontrola
   jest ostatnim, bezpośrednim dowodem. Przy jakiejkolwiek różnicy plik zostaje,
   a powód trafia do raportu jako `NIE USUNIETO`.
4. **Rozstrzyganie grup „ten sam identyfikator, inne bajty"** przez podobieństwo
   pierwszych stron: powyżej 0,85 to dwie wersje jednej pracy, poniżej — dwie różne
   prace pod mylącą nazwą, czyli błąd nazewnictwa do ręcznej poprawy.
5. Raport CSV ma teraz kolumnę `tytul`, a przy plikach do Kosza w kolumnie
   `informacja` stoi nazwa pozostawionego bliźniaka.

### Wynik uruchomienia po poprawce
```
PDF-y przeskanowane: 128
Grupy duplikatow: 5
Dodatkowe kopie: 5
PODGLAD: zadnego pliku nie usunieto.
```
Piąta para to `openaccess.thecvf.com_Wu_Correlation-Aware_Active_Learning...pdf`
i `COWAL_WACV2024.pdf` — ta sama praca COWAL pobrana dwoma drogami.

### Issues Encountered
- **Kosz Windows nie ma API w Pythonie do przywracania.** Rozwiązanie: COM
  `Shell.Application`, przestrzeń nazw `0xA`, dopasowanie czasownika kontekstowego
  wyrażeniem `^(Przywróć|Restore)` po usunięciu znaku `&` z nazwy — nazwa czasownika
  zależy od języka systemu.
- **Odczyt tytułów wymaga `pypdf`.** Gdy biblioteki nie ma, skrypt wypisuje jedno
  ostrzeżenie i działa dalej bez tytułów, zamiast się wywracać.
- **Katalog `papers\` w tym zbiorze nie istnieje** — skrypt wypisuje ostrzeżenie
  i pomija go, zamiast kończyć błędem.

## Files Generated/Modified
- `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\scripts\deduplikacja\usun_duplikaty_pdf.py` — przebudowany
- `...\scripts\deduplikacja\usun_duplikaty_pdf.bat` — dwuklik to teraz podgląd
- `...\scripts\deduplikacja\README.md` — opis dwóch kroków i trzech zabezpieczeń
- `...\pdf_all\arxiv_2404.09161.pdf`, `arxiv_2407.17140.pdf`, `arxiv_2507.00049.pdf` — przywrócone
- `.sessions/Session_2026-09-11_170517/` — ta dokumentacja

## Commands Used
```powershell
# Przywrócenie z Kosza
$cele = @('arxiv_2404.09161.pdf','arxiv_2407.17140.pdf','arxiv_2507.00049.pdf')
$bin = (New-Object -ComObject Shell.Application).Namespace(0xA)
foreach ($item in @($bin.Items())) {
  if ($cele -contains $item.Name) {
    $verb = @($item.Verbs()) | Where-Object { $_.Name -replace '&','' -match '^(Przywróć|Restore)' } | Select-Object -First 1
    if ($verb) { $verb.DoIt() }
  }
}
```
```bash
# Weryfikacja identyczności
py -3.11 -c "import filecmp; print(filecmp.cmp('2404.09161.pdf','arxiv_2404.09161.pdf',shallow=False))"

# Deduplikacja po poprawce
py -3.11 usun_duplikaty_pdf.py            # podgląd
py -3.11 usun_duplikaty_pdf.py --usun     # dopiero to usuwa
```

## Next Steps
- [ ] Zdecydować o pięciu parach duplikatów w `pdf_all` (podgląd gotowy, `--usun` czeka).
- [ ] Przenieść tę samą poprawkę bezpieczeństwa do wzorca
      `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDeduplicator` — ma tę samą wadę.
- [ ] Zaległe z sesji 16:07: zmiany w `protocol.yaml`, `bbox` -> `bboxes`, klatki tła
      poza rankingiem trudności, obniżenie `budgets.tier_b` poniżej 17 832.
