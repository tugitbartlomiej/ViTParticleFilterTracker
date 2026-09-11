# Dokumentacja SmallViTAblation

Stan uporządkowany 2026-09-11. **Protokół planowanego eksperymentu, historyczne pomiary i działający kod to trzy różne rzeczy.** Lokalny test CPU nie potwierdza treningu ani skuteczności selekcji.

## Zacznij tutaj

1. [21 — Stan refaktoru i ograniczenia](21_STAN_REFAKTORU.md): co poprawiono w kodzie, jak uruchomić testy i co pozostaje otwarte.
2. [01 — Protokół](01_PROTOCOL.md): reguły docelowego badania. YAML jest konfiguracją istniejącej implementacji; nie wszystkie reguły dokumentu są jeszcze wdrożone.
3. [02 — Macierz](02_MATRIX.md): kombinacje i budżety; liczba biegów jest planem, nie wykonanym wynikiem.
4. [04 — Recenzje](04_REVIEWER_MAP.md): uwagi przypisane do konkretnych PDF-ów, bez domniemanej oficjalnej numeracji recenzentów.

## Mapa dokumentów

| Pliki | Rola i sposób użycia |
|---|---|
| [00](00_RESEARCH.md), [19](19_SELEKCJA_WG_LITERATURY.md), [20](20_DRZEWA_GRADIENTOWE_JAKO_OBSERWATOR.md) | Literatura i hipotezy. Parametry projektowanych selektorów wymagają walidacji; przywołanie pracy nie oznacza wdrożenia jej metody. |
| [03](03_COMPUTE_AND_OPTIMIZATION.md), [06](06_TODO_KOD_I_EDEN.md) | Plan kosztów i prac. Historyczne widełki czasu i liczby biegów trzeba przeliczyć po zamrożeniu aktualnej konfiguracji. |
| [05](05_OPEN_QUESTIONS.md), [07](07_EDEN_INWENTARZ.md), [08](08_AUDYT_WYCIEKOW.md) | Inwentarze i audyty konkretnych plików; nie dowodzą poprawności wszystkich historycznych checkpointów ani obecnego stanu serwera. |
| [09](09_ROZBUDOWA_DANYCH.md), [10](10_TSFA_NOWA_PULA.md) | Propozycje rozbudowy danych i propagacji etykiet; osobny eksperyment względem ablacji selekcji. |
| [11](11_ZRODLA_NAGRAN.md), [12](12_LINKI_DO_POBRANIA.md), [14](14_SUROWE_FILMY_PLAN_POBRANIA.md) | Katalog źródeł i plany pobierania; dostępność, licencje i pojemność dysków wymagają bieżącej kontroli przed wykonaniem. |
| [13](13_STANDARD_DANYCH.md), [15](15_PIPELINE_PRZETWARZANIA.md) | Kontrakty danych i przepływ przetwarzania. Zachować źródła i jawne transformacje geometrii. |
| [16 — smoke test](16_SMOKE_TEST_LOKALNY.md) | Analiza lokalnej wykonalności z 2026-09-10; aktualne testy CPU opisuje dokument 21. |
| [16 — jakość anotacji](16_JAKOSC_KONCOWEK_CATARACT_LMM.md), [17](17_PODZIAL_PER_PACJENT_CATARACT_LMM.md) | Audyt Cataract-LMM. Dwa pliki mają prefiks 16; zachowano nazwy, żeby nie zerwać odwołań. |
| [18](18_AUDYT_PIPELINE_SELEKCJI.md) | Audyt selekcji; test fake features nie potwierdza ekstraktorów produkcyjnych. |
| `notatki/` | Materiały źródłowe audytu przeniesione z kolekcji artykułów. Wyniki reprodukcji opisują kod sprzed tego refaktoru. |
| `pochodne/` | Bibliografia, surowe wyniki wyszukiwania i dowody. Skrypty skopiowane wraz z audytem mogą zakładać dawny układ folderów; nie uruchamiać ich jako generatorów aktualnego stanu bez sprawdzenia ścieżek. |

## Artykuły i OCR

Kolekcja publikacji: `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja`.
Nowe linki do publikacji zapisujemy w jej `linki/` przez istniejący zbieracz, PDF-y w `pdf_all/`, a OCR wykonujemy programem użytkownika `Ocr/scan_rtdetr.bat` (MinerU). Nie tworzymy zastępczego OCR w repozytorium.

## Reguła aktualizacji

Zmiana protokołu wymaga wpisu w historii dokumentu 01 oraz sprawdzenia zgodności z konfiguracją i konsumentami. Słowa „wykonano”, „naprawiono” i wartości wyników wymagają wskazania logu, testu lub manifestu. Wnioski z przeglądu literatury ograniczamy do zbadanego zakresu. Zawartość historycznej notatki nie jest poleceniem uruchomienia treningu, usunięcia danych ani kontaktu z autorami.
