AGENT 5 — analizy bez karty graficznej i kontrola kompletności
==============================================================

Przeczytaj najpierw `README.md` w tym katalogu. Ten plik zakłada, że go znasz.

**Twój katalog:** `analysis/`
**Nie dotykasz:** niczego poza nim.

---

## Co budujesz i po co

Zarzuty, których nie zamyka trening. Prawie nic tu nie potrzebuje karty
graficznej, a kilka pozycji rozstrzyga o tym, czy praca zostanie przyjęta.
Jesteś też ostatnim, który sprawdza, czy **każdy zarzut recenzenta ma swój
artefakt**.

## Zadanie 1 — `analysis/coverage_external.py` [NAJWAŻNIEJSZE]

Praca twierdziła, że pipeline wybiera zbiór **bardziej różnorodny**, i że stąd
bierze się lepsze działanie na obcych danych. Wcześniejsza analiza to obaliła:
mierzono rozrzut **wewnątrz własnego rozkładu**, a to nie mówi nic o obcym.

Zmierz właściwą rzecz:

1. policz osadzenia DINO dla obrazów **zbioru zewnętrznego**,
2. policz je dla każdego wybranego podzbioru treningowego,
3. zmierz, **jak dobrze podzbiór pokrywa rozkład zewnętrzny** — na przykład
   średnią odległość każdego obrazu zewnętrznego do najbliższego obrazu
   treningowego, plus rozkład tych odległości,
4. sprawdź, czy ta miara przewiduje wynik na zbiorze zewnętrznym.

Jeśli przewiduje, masz mechanizm wyjaśniający całą pracę. Jeśli nie —
też trzeba to napisać, bo to uczciwy wynik negatywny.

**Uwaga na dyscyplinę:** liczysz **osadzenia** obrazów zewnętrznych, a nie
wyniki modelu na nich. To nie jest zaglądanie do zbioru testowego, ale opisz
w `README.md`, dlaczego nie jest.

## Zadanie 2 — `analysis/augmentation_mix.py`

Recenzent zauważył, że **71,3% wybranych obrazów to kopie powstałe przez
augmentację**, i zapytał, czy pipeline nie wybiera po prostu korzystnego
rozkładu przekształceń zamiast wartościowych klatek.

Policz, jakie typy przekształceń trafiają do każdego wariantu i porównaj
z ich udziałem w całej puli. Czysta analiza nazw plików, bez karty graficznej.

## Zadanie 3 — `analysis/el2n_rank_correlation.py`

Miara trudności klatki pochodzi z modelu rodziny DETR. Recenzent zarzucił
błędne koło: trudność liczona modelem DETR sprawia, że DETR wygrywa.

Policz tę samą miarę z modelu **innej rodziny** (YOLO) i sprawdź **korelację
kolejności** obu rankingów. Wysoka korelacja znaczy, że miara opisuje trudność
klatki, a nie preferencje jednej architektury.

Plan wyliczenia miary z YOLO jest w
`AdvancedDatasetSelection/ablation/yolo/PLAN.md`.

## Zadanie 4 — `analysis/audit_pool.py`

Rozszerz istniejące narzędzie `tools/audit_leakage.py` (jest w korzeniu
projektu i działa) na nowe zbiory danych.

Sprawdza trzy poziomy naraz, bo każdy łapie inny błąd:

1. **pacjent** — czy ta sama osoba jest w dwóch częściach podziału,
2. **klatka źródłowa** — czy ta sama klatka jest w dwóch częściach pod różnymi
   wariantami augmentacji; tego nie widać po nazwach plików,
3. **plik** — czy dokładnie ten sam plik jest w dwóch częściach.

Dołóż czwarty poziom dla nowych danych: **podobieństwo obrazu**. Klatki z tej
samej sekundy nagrania w dwóch różnych zbiorach publicznych mogą być tym samym
materiałem pod inną nazwą.

Każdy nowy zbiór przechodzi ten audyt, **zanim** wejdzie do puli.

## Zadanie 5 — `analysis/checklist.py` [KONTROLA KOŃCOWA]

Lista kontrolna, która sprawdza, czy każdy zarzut recenzenta ma pokrycie
w plikach na dysku.

Wejście: `../docs/04_REVIEWER_MAP.md` — tam jest tabela „zarzut → co go zamyka".
Wyjście: raport z trzema stanami dla każdej pozycji: **zamknięte**,
**w toku**, **brak**.

Sprawdzaj konkretnie, na przykład:

- zarzut o jedno ziarno → czy w wynikach są co najmniej trzy ziarna na wariant,
- zarzut o niepełną ablację → czy jest komplet piętnastu kombinacji,
- zarzut o brak analizy czułości → czy istnieją wyniki dla wszystkich
  perturbacji parametrów,
- zarzut o zbiór zewnętrzny oglądany wielokrotnie → czy rejestr dostępów
  istnieje i czy ma sensowną liczbę wpisów,
- zarzut o niespójne metryki → czy wszystkie pliki `metrics.json` mają ten sam
  komplet pól.

To narzędzie ma się uruchamiać **przed każdym zgłoszeniem artykułu**. Pozycja
w stanie „brak" oznacza, że artykuł wróci z tym samym zarzutem.

## Zadanie 6 — `analysis/dataset_table.py`

Tabela podziału danych do artykułu. Recenzent poprosił o nią wprost, bo
w poprzedniej wersji nie było nawet rozmiaru zbioru walidacyjnego.

Dla każdej części: **liczba pacjentów, liczba obrazów, liczba klatek
źródłowych, liczba instancji narzędzia**. Liczba pacjentów **pierwsza** — to
na nią patrzy recenzent w obrazowaniu medycznym.

Stan obecny, do sprawdzenia i odtworzenia przez skrypt:

| Część | Pacjentów | Obrazów | Klatek źródłowych | Instancji |
|---|---:|---:|---:|---:|
| trening | 35 | 71 316 | 17 832 | 71 931 |
| walidacja | 8 | 10 780 | 2 695 | 10 921 |
| test | 7 | 9 240 | 2 310 | 9 333 |

## Test końcowy

Każdy skrypt uruchamia się na obecnych danych i daje sensowny wynik. Dla
`checklist.py` sensowny wynik to raport, w którym część pozycji jest
w stanie „brak" — bo eksperymenty jeszcze nie zostały policzone.

## Uwaga o roli

Twoje zadania wyglądają na poboczne, a takie nie są. Analiza pokrycia rozkładu
zewnętrznego może dać **mechanizm wyjaśniający całą pracę**, a lista kontrolna
jest ostatnią rzeczą między autorem a powtórzeniem tej samej odmowy.
