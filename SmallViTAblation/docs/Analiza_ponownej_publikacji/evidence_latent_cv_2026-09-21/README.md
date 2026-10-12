# Weryfikacja literatury do planu diagnostyki latentu w widzeniu

Data: 2026-09-21. Dokument docelowy: [plan eksperymentu 37](../37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md).

Użyto wskazanej przez autora umiejętności `C:/Users/bartl/.agents/skills/paper-search/SKILL.md` i dostępnego CLI `C:/Users/bartl/.local/bin/paper-search.exe`. Surowe odpowiedzi zachowano bez poprawiania tytułów i wyników nietrafnych. Pliki `*.stderr.txt` zawierają komunikaty o nieustawionych opcjonalnych kluczach i limicie dostępu anonimowego, bez wartości kluczy.

## Zapytania i rzeczywiste wyniki

| Plik JSON | Polecenie | Wynik |
|---|---|---|
| `paper_search_koopman_video.json` | `paper-search search 'Koopman video object dynamics' -n 3 -s arxiv,semantic,crossref` | 9 rekordów. Semantic Scholar wskazał prace Comasa z 2021 i 2023 oraz pracę o interakcjach obiektów. Część wyników arXiv i Crossref była nietrafna. |
| `paper_search_mori_zwanzig.json` | `paper-search search 'Mori Zwanzig Koopman memory learning' -n 3 -s arxiv,crossref` | 6 rekordów, w tym Lin i in., arXiv:2101.05873v3. Inne trafienia nie zostały automatycznie uznane za dowód dla obrazów. |
| `paper_search_equivariance.json` | `paper-search search 'Understanding Image Representations by Measuring Their Equivariance and Equivalence' -n 2 -s semantic` | 0 rekordów w odpowiedzi CLI. Źródło zweryfikowano przez stronę publikacji. Brak wyniku nie jest dowodem braku pracy. |
| `paper_search_active_video.json` | `paper-search search 'active learning surgical video temporal diversity' -n 3 -s semantic` | 0 rekordów w odpowiedzi CLI. COWAL i LRTD sprawdzono niezależnie na stronach źródeł pierwotnych. |

Pole `errors` w czterech odpowiedziach CLI jest puste. Nie wyciągamy z tego wniosku, że wyniki są kompletne lub że zapytania bez trafień były wyczerpujące.

## Źródła pierwotne i zakres odczytu

| Rekord | URL | Zakres wykorzystany w tej rundzie |
|---|---|---|
| Comas i in. 2021 | https://arxiv.org/abs/2110.00547 | Metadane i abstrakt; informacja o testach syntetycznych. |
| Comas i in. 2023 | https://proceedings.mlr.press/v211/comas23a.html | Metadane PMLR i abstrakt; dynamika obiektów, predykcja i granice przeniesienia. |
| Lenc i Vedaldi 2015 | https://arxiv.org/abs/1411.5908 | Metadane i opis podejścia; odrębna rozszerzona wersja czasopiśmienna była też sprawdzana w poprzednim etapie rozmowy. |
| Lenc i Vedaldi, wersja rozszerzona 2019 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6510825/ | Wybrane fragmenty HTML o ekwiwariancji i zgodności reprezentacji, odczytane w poprzednim etapie rozmowy. Nie liczyć jako niezależnej replikacji wersji CVPR. |
| Lin i in. 2021 | https://arxiv.org/abs/2101.05873 | Metadane i abstrakt; połączenie Koopmana/MZ, estymacja pamięci i przykład Lorenz-96. |
| COWAL, Wu i in. 2024 | https://arxiv.org/abs/2311.08811 | Metadane oraz opis metody i zadania; wybrane fragmenty HTML https://arxiv.org/html/2311.08811v1. |
| LRTD, Shi i in. 2020 | https://arxiv.org/abs/2004.09845 | Metadane i abstrakt; zależności czasowe i rozpoznawanie faz operacji. |
| Brunton i in. 2016 | https://arxiv.org/abs/1510.03007 | Metadane, abstrakt i DOI wersji czasopiśmiennej: https://doi.org/10.1371/journal.pone.0150171. |

Nie pobierano PDF-ów, nie wykonywano OCR i nie reprodukowano kodu publikacji. W planie nie przytaczamy liczbowych przewag z literatury jako oczekiwanej poprawy dla CATARACT-LMM. Przegląd służy uzasadnieniu konstrukcji porównań i ograniczeń wnioskowania.

## Najważniejsze korekty interpretacji

- Rozdzielenie wyglądu obiektu i jego dynamiki ma precedensy w widzeniu maszynowym, ale dobór klatek do detekcji wymaga osobnego eksperymentu.
- COWAL uzasadnia silną kontrolę różnorodność + niepewność; LRTD przestrzega przed zakładaniem, że mała przewidywalność oznacza bezużyteczną klatkę.
- Teoria projekcji uzasadnia możliwość pamięci po redukcji obserwacji; nie gwarantuje fizycznej interpretacji regresji z opóźnieniami.
- Metadane L1 i L2 różnią się tytułem, rokiem i listą autorów. Nie przypisywać tytułu PMLR 2023 do arXiv:2110.00547 bez wyjaśnienia.

## Aktualizacja: nadzorowana selekcja do zwykłego RF-DETR

Dokument docelowy: [plan 39](../39_PLAN_SELEKCJI_KLATEK_RFDETR_I_ROZWOJU_JEPA.md). Etykiety całego train już istnieją; selektor ma poprawiać zbiór końcowego treningu bez udziału w inferencji. Aktualizacja obejmuje audyt kodu, odczyt istniejących raportów i celowane wyszukiwanie. Nie uruchomiono treningów, nie odpytywano kolejki klastra i nie pobierano PDF-ów.

| Plik JSON (towarzyszy mu stderr) | Zapytanie CLI | Źródła i limit | Wyniki |
|---|---|---|---|
| `selection_update_cowal_search.json` | `correlation aware active learning surgery video segmentation` | arxiv,crossref; n=3 | 6 rekordów |
| `selection_update_koopman_search.json` | `Koopman Mori Zwanzig data selection training surgical video` | arxiv,crossref; n=4 | 8 rekordów, część nietrafna |
| `selection_update_novelty_koopman_search.json` | `Panning for Gold Finding Representative Water Quality Dynamic Data Data-Efficient Machine Learning Koopman Analysis` | crossref; n=2 | 2 rekordy |
| `selection_update_rho_loss_search.json` | `Prioritized Training Points Learnable Worth Learning Not Yet Learnt` | arxiv,crossref; n=3 | 6 rekordów |

Źródła zweryfikowane na stronach publikacji i w abstraktach, bez deklaracji lektury pełnych tekstów:

- [OHEM, Shrivastava, Gupta, Girshick, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html): nadzorowany dobór trudnych regionów, nie selektor pełnych klatek RF-DETR.
- [RHO-loss, Mindermann i in., ICML 2022](https://proceedings.mlr.press/v162/mindermann22a.html): priorytetyzacja przykładów pod kątem generalizacji; residual dynamiki nie jest tą samą miarą.
- [GLISTER, Killamsetty i in., AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16988): nadzorowana selekcja podzbioru i wariant aktualizowany w treningu.
- [COWAL](https://arxiv.org/abs/2311.08811), [Lin i in.](https://arxiv.org/abs/2101.05873), [Comas i in. 2023](https://proceedings.mlr.press/v211/comas23a.html): ponowne sprawdzenie zakresu zastosowań, jak w poprzedniej tabeli.
- [Tian i in. 2026, Panning for Gold](https://www.sciencedirect.com/science/article/pii/S3050472426000171): DOI 10.1016/j.wateco.2026.100049, metadane Crossref i abstrakt wydawcy odczytane w przeglądzie; ponowny bezpośredni odczyt ScienceDirect zwrócił 403. Tytuł, autorów i datę online 9 lipca 2026 potwierdza również [lista artykułów wydawcy KeAi](https://www.keaipublishing.com/en/journals/water-and-ecology/recent-articles/). Dobór danych dynamiki przy użyciu Koopmana ma precedens poza CV; bez twierdzeń o liczbowej przewadze przenoszonej do chirurgii.
- [EnKode, Li, Silva, Hsieh, 2024](https://arxiv.org/abs/2410.16605): Koopman i ensemble w aktywnym próbkowaniu pól przepływu.
- [V-JEPA 2, Assran i in., 2025](https://arxiv.org/abs/2506.09985), [SurgTrack, Guo i in., 2024](https://arxiv.org/abs/2409.02598): abstrakty i metadane zweryfikowane podczas rozmowy; tylko podstawa dalszych etapów, nie dowód skuteczności aktualnego selektora.

Przegląd nie wykazał kompletności i nie stanowi systematycznego dowodu nowości. W szczególności nie wolno pisać „pierwsza selekcja danych Koopmanem”.

### Dowody lokalne

`selection_update_local_validation_summary.json` zawiera 36 odczytanych raportów kampanii Static36, ścieżki i SHA256 ich źródeł oraz średnie AP dla `last_epoch`/`best_val` według selektora i budżetu. Wszystkie odczytane raporty mają status passed, ocenę val i test_evaluated=false. Zestawienie nie przelicza AP z predykcji i nie zawiera testu istotności. Same raporty nie dowodzą, że test nigdy nie został użyty w innych analizach.

Sprawdzono także `REPORT_kalman_ut_b2500_best.json`, przykładowy `effective_train_config.json`, manifest podziałów i pierwszy rekord puli. Wyniki w dokumencie 38 mają ograniczenia CV opisane tam w sekcji 4e i w planach 37/39; nie zostały tu naprawione ani ponownie przeliczone. Pozostałe twierdzenia liczbowe z dokumentu 38, np. częstość chybień według ruchu, są jawnie przytoczeniem raportu eksploracyjnego, nie nowym pomiarem.
