# Session Summary: Session_2026-09-11_170251

## Metadata
- **Date:** 2026-09-11
- **Time:** 17:02:51
- **Status:** Completed
- **Type:** Mixed (Tools + Debug)

## Objective
Użytkownik zobaczył w konsoli trzy linie `PRZENOSZE DO KOSZA` i zażądał kolejno:

1. sprawdzić, czy to na pewno te same pliki, przywrócić je i pokazać w osobnym katalogu,
2. poprawić skrypt, bo „to są zupełnie inne tytuły, to inne artykuły",
3. dodać deduplikatorowi odczyt pierwszej strony, żeby sam rozstrzygał,
   czy dwa PDF-y to ta sama praca, czy dwie różne.

## Context
Wcześniej tego dnia pobrałem 16 PDF-ów do zbioru
`F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja`. Powstały przy tym
trzy duplikaty, a uruchomiony potem `usun_duplikaty_pdf.py` przeniósł je do Kosza.
Komunikat wypisywał same nazwy plików, bez tytułów — i stąd wrażenie, że giną
trzy niezwiązane ze sobą prace.

## Actions Taken

### 1. Weryfikacja: czy to na pewno duplikaty
- Odnalazłem skrypt po treści komunikatu (`grep "PRZENOSZE DO KOSZA"`).
- Przeczytałem oba raporty CSV z tego przebiegu.
- Wylistowałem Kosz Windows przez `Shell.Application` i potwierdziłem, że wszystkie
  trzy pliki tam są, z oryginalną ścieżką.
- Wyjąłem je z Kosza do osobnego katalogu przeglądowego.
- Policzyłem SHA-256 **niezależnie**, poza skryptem, dla par plik-bliźniak.
- Gdy to nie przekonało użytkownika, otworzyłem wszystkie sześć plików i wyciągnąłem
  tytuł z pierwszej strony.

### 2. Szukanie prawdziwej przyczyny
- Prześledziłem `obsluz_link` w `pobierz_pdfy.py` i sprawdziłem, jak działa indeks
  posiadanych (`zbuduj_indeks`, wyrażenie `(\d{4}\.\d{4,5})(v\d+)?`).
- Sprawdziłem wiek indeksu i to, czy zawierał brakujące identyfikatory. Zawierał —
  więc hipoteza o nieświeżym indeksie odpadła.
- Zostało jedno wyjaśnienie: link był DOI wydawcy, w którym numeru arXiv po prostu nie ma.

### 3. Poprawka pobieracza
- Dopisałem w `obsluz_link` drugie sprawdzenie numeru arXiv, już **po** rozwiązaniu adresu.
- Test na sucho ze świeżym indeksem, na tej samej liście linków.

### 4. Odczyt pierwszej strony w deduplikatorze
- Dodałem `pierwsza_strona`, `tytul_pdf`, `podobienstwo_stron`, `werdykt_grupy`.
- Podpiąłem tytuły pod komunikaty konsoli i pod raport CSV.
- Test pozytywny na prawdziwych duplikatach i test negatywny na celowo
  podmienionej nazwie.

## Results

### Key Findings

**Fałszywy alarm — to były duplikaty.** Każda para zgodna co do tytułu i co do bajtu:

| numer arXiv | tytuł | bajty | SHA-256 |
|---|---|---|---|
| 2404.09161 | Coreset Selection for Object Detection | 1 123 964 | zgodny |
| 2407.17140 | RT-DETRv2: Improved Baseline with Bag-of-Freebies… | 88 744 | zgodny |
| 2507.00049 | AdaDeDup: Adaptive Hybrid Data Pruning… | 30 095 512 | zgodny |

Trzy usunięte pliki to trzy **różne** prace i to zmyliło odczyt komunikatu.
Ale każda z nich została w `pdf_all`, tylko pod krótszą nazwą.

**Prawdziwy błąd siedział w pobieraczu.** Sprawdzenie „czy już mam" wyglądało tak:

1. wyciągnij numer arXiv **z adresu na liście** i porównaj z indeksem,
2. rozwiąż adres, a potem porównaj **pełną nazwę** wynikowego pliku.

Dla tych trzech prac link był DOI wydawcy, na przykład
`10.1109/CVPRW63382.2024.00764` dla *Coreset Selection for Object Detection*.
Numeru arXiv tam nie ma, więc krok pierwszy nic nie znalazł. Unpaywall wskazał potem
preprint na arXiv, plik dostał nazwę `arxiv_2404.09161.pdf`, a w zbiorze leżał ten sam
artykuł jako `2404.09161.pdf`. Nazwy różne, więc krok drugi też nie zadziałał.

Po poprawce, na sucho i ze świeżym indeksem: **17 pominiętych jako posiadane,
zero ponownych pobrań**.

**Deduplikator mówi teraz tytułami.** Komunikat wygląda tak:

```
DO KOSZA: pdf_all\arxiv_2404.09161.pdf
    tytul:  Coreset Selection for Object Detection
    ZOSTAJE ta sama praca: pdf_all\2404.09161.pdf
```

W grupach „ten sam numer, inne bajty" skrót nic nie rozstrzyga, a pierwsza strona tak.
Próg podobieństwa 0,85: powyżej to dwie wersje jednej pracy, poniżej to dwie różne
prace pod mylącą nazwą. Test kontrolny (dwa różne PDF-y nazwane tym samym numerem)
dał podobieństwo 0,02 i poprawne ostrzeżenie.

### Issues Encountered

| problem | rozwiązanie |
|---|---|
| Moja pierwsza diagnoza była błędna — podejrzewałem, że skrypt zostawił złą nazwę wbrew `AGENTS.md` | Sprawdziłem wyrażenie indeksujące: numer arXiv jest wyciągany z dowolnej nazwy, więc obie działają. Ostrzeżenie z `AGENTS.md` dotyczy prac spoza arXiv |
| Druga hipoteza (nieświeży indeks) też odpadła | Indeks miał 1,4 h i zawierał wszystkie trzy numery. Dopiero to wskazało na DOI wydawcy jako źródło problemu |
| Heredoc w Bashu wysypał się przy zapisie długiego pliku Markdown | Przejście na narzędzie Write |
| `Format-Table` w PowerShellu na pętli `foreach` — „Pusty element potoku jest niedozwolony" | Wynik pętli przypisany najpierw do zmiennej |
| Odczyt tytułu wymaga `pypdf`, którego może nie być | Skrypt działa dalej, wypisuje jedno ostrzeżenie i pomija tytuły. Jest też `--bez-tytulow` |
| Nagłówek `arXiv:2407.17140v1 [cs.CV] 24 Jul 2024` stoi przed tytułem i psuł odczyt | Wycinany wyrażeniem regularnym przed przycięciem tytułu |

## Files Generated/Modified

**Poza repozytorium projektu** (katalog `F:\Studia\Articles\Zbiory` nie jest repozytorium git):

- `_Narzedzia\PdfDownloader\pobierz_pdfy.py` — drugie sprawdzenie numeru arXiv
  w `obsluz_link`, po rozwiązaniu adresu.
- `2026-09-11_RT-DETR_selekcja_ewaluacja\scripts\deduplikacja\usun_duplikaty_pdf.py` —
  odczyt pierwszej strony, tytuły w konsoli i w raporcie, werdykt dla grup
  o tym samym numerze, flaga `--bez-tytulow`, nowa kolumna `tytul` w CSV.
  W trakcie sesji plik dostał jeszcze dwa zabezpieczenia (zmiana spoza tej rozmowy,
  przyjęta jako stan obowiązujący): uruchomienie **bez flag niczego nie usuwa** —
  trzeba świadomie dopisać `--usun` — a przed przeniesieniem każda para jest
  porównywana bajt po bajcie przez `filecmp.cmp(shallow=False)`, bo sam zgodny
  skrót SHA-256 to wniosek pośredni.
- `…\scripts\deduplikacja\README.md` — opis nowego mechanizmu z przykładami.
- `…\pochodne\duplikaty_do_sprawdzenia_2026-09-11\` — trzy pliki wyjęte z Kosza
  plus `CO_TO_JEST.md` z dowodem i wyjaśnieniem przyczyny.

**W repozytorium projektu:** tylko ta sesja. Kod projektu bez zmian.

## Commands Used

```bash
# gdzie powstaje ten komunikat
grep -rn "PRZENOSZE DO KOSZA" --include=*.py .

# co naprawde lezy w Koszu
$sh = New-Object -ComObject Shell.Application; $sh.Namespace(0xA).Items()

# tytul z pierwszej strony, niezaleznie od skryptu
py -3.11 -c "from pypdf import PdfReader; ..."

# test poprawki pobieracza
py -3.11 pobierz_pdfy.py --zbior "..." --przeindeksuj --sucho "linki\...txt"

# test deduplikatora: przypadek pozytywny i negatywny
py -3.11 usun_duplikaty_pdf.py --dry-run --foldery pdf_all pochodne\duplikaty_do_sprawdzenia_2026-09-11
py -3.11 usun_duplikaty_pdf.py --dry-run --zbior <katalog testowy> --foldery test_pdf
```

Uwaga: po późniejszej zmianie podgląd jest domyślny, więc `--dry-run` nie jest
już potrzebny, a usunięcie wymaga jawnego `--usun`.

## Next Steps

- [ ] Rozważyć wersjonowanie `F:\Studia\Articles\Zbiory\_Narzedzia` w git.
      Dziś poprawki dwóch skryptów nie mają żadnej historii zmian.
- [ ] Uporządkować nazwy w `pdf_all`: obok siebie leżą `2404.09161.pdf`
      i `arxiv_2404.09161.pdf`. Jedna konwencja usunęłaby problem u źródła.
- [ ] Uruchomić `usun_duplikaty_pdf.py` na czysto po tym uporządkowaniu.
- [ ] **Naprawić `.bat` w zbiorze RT-DETR** (zaległość z poprzedniej sesji):
      po przeniesieniu do `scripts\` flaga `--zbior "%~dp0."` wskazuje zły katalog.
- [ ] Uruchomić `odswiez_kolejke.bat` dla ośmiu prac za paywallem.
