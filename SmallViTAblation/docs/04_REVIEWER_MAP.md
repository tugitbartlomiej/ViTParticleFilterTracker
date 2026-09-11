# 04 — Recenzje: wymagania, dowody i status

Aktualizacja: 2026-09-11. Źródła: folder `DETR_IEEE/odpowiedziRecenzentów`.
Nazwy plików nie potwierdzają oficjalnej numeracji recenzentów. Używamy:

- **A** — `Comments_to_the_Author.pdf`, 17 numerowanych uwag; rekomendacja Reject (do not encourage resubmit).
- **B** — `Reviewer_Comments_to_Authors.pdf`, 10 uwag głównych i 3 akapity drobne; rekomendacja Reject.

W poprzedniej wersji tego dokumentu „recenzent 1” oznaczał B, a „recenzent 2” A.
W [pełnej mapie 30 uwag](notatki/MAPA_UWAG_RECENZENTOW.md) lokalne etykiety są odwrotne:
R1=A, R2=B. Przy cytowaniu identyfikować źródłowy plik i numer uwagi.

Rekomendacja recenzenta nie zastępuje formalnej decyzji redakcji. Miejsce kolejnego
zgłoszenia pozostaje decyzją autora.

## Główne warunki odpowiedzi

| Uwagi źródłowe | Wymagany dowód | Stan po refaktorze |
|---|---|---|
| A2, A15, B1 | Split po operacjach/pacjentach, pełne liczebności, brak kontaktu testu z treningiem i inicjalizacją | Istnieje manifest 35/8/7 i potwierdzenie autora o pacjentach CATARACTS. Główne porównania muszą rzeczywiście korzystać z tego splitu; nie zmienia on historii checkpointów. |
| A4, B6 | Jednakowy COCO AP, PR/F1 z progiem walidacyjnym, analiza transferu progu | Poprawiono zakres obrazów, automatyczny tuning tylko na val, remap klas i wspólne metryki transferu. Potrzebna ocena rzeczywistych modeli i analizy kalibracji. |
| A5, B4 | Pełna macierz cech przy stałych warunkach treningu | Zaplanowana, nie wykonana w tym refaktorze. Historyczny pełny pipeline nie może zastąpić kontrolnego wiersza nowej macierzy. |
| A6 | Powtarzany random z oryginalnych klatek, ten sam budżet | Budżet źródeł skorygowany w dokumencie 01. K=20k nie mieści się w obecnych 17 832 nazwach źródłowych train. |
| A7, A10 | Izolacja efektu selekcji także dla YOLOv8 | Brak potwierdzenia wykonanej macierzy. Transfer kilku wariantów na nowy model nie jest pełną ablacja YOLOv8; zmianę zakresu trzeba uzasadnić. |
| A3, B7 | Współczesny YOLO i co najmniej jeden nowoczesny DETR-style | RT-DETR spełnia kierunek uwagi. Recenzent nie wymagał małego ViT. Istniejący adapter RF-DETR wymaga smoke testu rzeczywistej biblioteki; RT-DETR osobnego adaptera. |
| B8 | Uczciwe porównanie architektur i kosztów | Jednakowe epoki, rozdzielczość i parametry nie gwarantują iso-resource. Raportować efektywny batch, kroki, pretraining, receptę, GPU-czas i budżet strojenia. |
| A1, A13–14, B3, B-drobne | Sens składowych FastSAM, wrażliwość parametrów, PCA, alternatywne selektory i budżety | Hipotezy i macierze w dokumentach 02/19; wymagają wyników walidacyjnych. Korelacja score nie zastępuje downstream AP. |
| A9, B2 | Powtórzenia i niepewność | Oddzielono SD treningów od CI po pacjentach. Nie ustalono z góry wartości p ani nie potwierdzono nowych średnich i SD. |
| B5, B10 | Standardowy odczyt queries i pełna chronologia dostępu do testu | Q81 jako analiza historyczna, główny wynik ze standardowego postprocessingu. Ledger nie przywraca niezależności już analizowanego benchmarku. |
| B9 | Ustawienia i wrażliwość NMS | Poprawiono pomocniczy matching i rozdzielenie klas. Surowe predykcje sprzed NMS, pełna analiza walidacyjna i porównanie Soft-NMS pozostają do wykonania. |
| A8, B-drobne | Pochodzenie puli, zewnętrznego testu i augmentacji | Inwentarze opisują konkretne anotacje. Brakujące 12 wariantów nie dowodzi uszkodzonych filmów; 48 zewnętrznych nazw wymaga audytu historii biegów. |
| A11–12, A16 | Przykłady wizualne, zgodny schemat pipeline'u, jednoznaczne wiersze tabel | Praca redakcyjna po zamrożeniu metody i wyników. |
| A17 | Ostrożna interpretacja kliniczna | Względna poprawa AP nie jest dowodem użyteczności klinicznej; dokładność lokalizacji wymaga osobnej oceny. |

## Czego nie deklarować jako wykonanego

Przekazany `RESPONSE_TO_REVIEWERS_DETAILED.md` zawiera nowe średnie, SD, p<0,001,
analizy wrażliwości i opowieść o 12 uszkodzonych klatkach. Audyt nie potwierdził ich
artefaktami. Nie kopiować tych zdań do odpowiedzi jako wyników wykonanych badań.
Każda przyszła deklaracja wymaga logu/manifestu/wyniku i wskazania zmienionego fragmentu artykułu.

Szczegółowy status kodu i polecenia weryfikacji: [21_STAN_REFAKTORU.md](21_STAN_REFAKTORU.md).
Pełne parafrazy wszystkich 30 uwag oraz lista niepotwierdzonych twierdzeń:
[MAPA_UWAG_RECENZENTOW.md](notatki/MAPA_UWAG_RECENZENTOW.md).

## Historia zmian

- 2026-09-11 — ujednolicono identyfikację źródeł, oddzielono plan od dowodu wykonania;
  usunięto deklaracje, że samo istnienie splitu, tabela planu lub liczba seedów zamyka uwagi.
