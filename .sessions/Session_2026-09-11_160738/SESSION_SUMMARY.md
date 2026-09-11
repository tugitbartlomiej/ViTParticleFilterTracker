# Session Summary: Selekcja danych wg literatury + ważność próbki (RF-DETR / RT-DETR)

## Metadata
- **Date:** 2026-09-11
- **Time:** 16:07:38
- **Status:** Completed
- **Type:** Analysis (weryfikacja literaturowa + porządkowanie bibliografii)

## Objective
Trzy zadania, kolejno zgłaszane przez użytkownika:

1. Sprawdzić, jak **powinna** wyglądać selekcja danych przed treningiem dla detektora
   rodziny DETR, porównując kod `SmallViTAblation/pipeline/`, stary pipeline
   `AdvancedDatasetSelection/` i metodologię artykułu `DETR_IEEE_FINAL.tex` z literaturą.
2. Wyjaśnić, czym jest LightGBM jako „obserwator" ważności danych i czy da się go użyć
   do wyboru obrazów wchodzących do sieci; potem poszerzyć research o inne metody
   ważności próbki dla obrazów.
3. Uporządkować bibliografię zbioru `2026-09-11_RT-DETR_selekcja_ewaluacja`: listy linków,
   BibTeX, automatyczne pobieranie PDF-ów.

## Context
Artykuł IEEE został odrzucony; `docs/01_PROTOCOL.md` (v1, 2026-09-08) to nowy protokół
napisany po recenzjach. Audyt z `docs/18_AUDYT_PIPELINE_SELEKCJI.md` (2026-09-11) wykazał,
że pipeline selekcji **w ogóle się nie uruchamiał** i zostawił cztery decyzje otwarte.
Weryfikacja literaturowa z 2026-09-10 (`AdvancedDatasetSelection/PLAN_v3_RF-DETR.md`)
dotyczyła protokołu z artykułu, a nie nowego kodu. Ta sesja domyka lukę: literatura kontra
**kod, który faktycznie stoi w repozytorium**.

## Actions Taken

1. **Wczytanie stanu**: `pipeline/{features,selection}/`, `configs/protocol.yaml`,
   `docs/{00,01,02,18}`, `PLAN_v3_RF-DETR.md`, sekcja Methodology artykułu (linie 143–270).
2. **Literatura, tura 1** (selekcja dla detekcji): sprawdzenie istniejącego deep searchu
   z 2026-09-10 w workspace ViT zamiast uruchamiania duplikatu; **9 prac przeczytanych
   w całości** przez `read_pdfs`: Zha26e (RT-DETRv2 + DINOv2 + Hungarian), Lee24b (CSOD),
   Kan25 (AdaDeDup), Yag25 (VPS), Cho25 (DUAL), Pau21 (EL2N), Zhe24b (ELFS), Zhe22 (CCS),
   Xie26 (AFSS).
3. **Dokument `docs/19_SELEKCJA_WG_LITERATURY.md`**: rozstrzygnięcie RF-DETR kontra RT-DETR,
   sześć rozbieżności z dowodem w kodzie (plik:linia), tabela realnych zysków z literatury,
   protokół docelowy w siedmiu krokach, kolejność zmian w plikach.
4. **Bibliografia zbioru RT-DETR**: lista linków do PDF-ów użytych w werdykcie; sprawdzenie,
   że lokalny pobieracz ma własną konfigurację z odsiewem duplikatów; sklejony BibTeX
   (29 wpisów, metadane z `doi.org`, klucze `nazwisko+rok`); rozbicie na `bibtex/<klucz>.bib`.
5. **Literatura, tura 2** (LightGBM jako obserwator): deep search „drzewa gradientowe jako
   obserwator waznosci danych obrazowych" (171 prac) + wyszukiwanie w sieci o GOSS;
   **6 prac przeczytanych w całości**: Sch25, Kat18, Kil21, Oka23, Jia21, Jia23, plus
   Kuz24 i An23 pod kątem audytu. Dokument `docs/20_DRZEWA_GRADIENTOWE_JAKO_OBSERWATOR.md`.
6. **Literatura, tura 3** (ważność próbki dla obrazów): cztery wyszukiwania semantyczne,
   **5 kolejnych prac w całości**: Sun25 (DetGain), Eva24 (JEST), Cal25 (DataRater),
   Jin18 (migotanie w wideo), Hua25c (Mimic Score). Dokument
   `docs/22_WAZNOSC_PROBKI_METODY_OBRAZOWE.md`.
7. **Pobieranie**: lista `waznosc-probki-i-selekcja-danych-2026-09-11.txt`; konwersja eksportu
   `untitled.bib` z ResearchRabbit (65 wpisów) na linki — DOI brakujące rozwiązane po tytule
   przez Crossref, API arXiv i OpenAlex przy progu dopasowania 90%.

## Results

### Key Findings

**A. Sześć rozbieżności kodu z literaturą** (`docs/19`, sekcja 3)

| # | Co jest w kodzie | Co mówi literatura |
|---|---|---|
| R1 | `proxy_subset_size: 2500`, jedna inicjalizacja | Pau21: proxy na CAŁEJ puli, średnia z 10 inicjalizacji; 1 init → Spearman 0,75 i wynik jak losowanie |
| R2 | `el2n_from_proxy.py:72` — klatka bez GT dostaje 1.0 | Yag25: brak detekcji = niska wariancja; wybór najtrudniejszych to −5 do −15 pp mAP |
| R3 | `score = 1 − pewność` nazwane EL2N | Pau21: EL2N to norma L2 uśredniona po inicjalizacjach |
| R4 | `k = budget`, centroid | Kan25: K = 100 klastrów + budżet per klaster; Zhe22: 50 warstw + odcięcie β |
| R5 | selekcja z 71 316 obrazów (pula augmentowana) | Zha26e: selekcja na oryginałach, augmentacja po wyborze |
| R6 | DINO CLS całego kadru, 224 px | Lee24b: cechy obiektowe 44,3 kontra cały obraz 38,6/32,3/26,0 AP50 (losowanie 37,9) |

**B. Kalibracja oczekiwań** — zysk selekcji nad losowaniem w literaturze: AdaDeDup nuScenes
0,3728 kontra 0,3663 mAP; VPS na VOC +1,1 / +1,9 / +2,0 pp przy 50/30/10% zostawionych;
CSOD +6,4 pp, ale przy 200 obrazach. Transfer trudności potrafi być ujemny: przy 90%
przycięciu EL2N 18,44 kontra 22,87 dla losowania (Yag25, tab. 3).

**C. Blokada arytmetyczna**: `budgets.tier_b: 20000` przekracza 17 832 kadrów źródłowych
treningowych — poziom B jest niewykonalny bez powtórzeń.

**D. LightGBM** (`docs/20`): mierzy ważność **cech**, nie **próbek**; GOSS (zachowanie
przykładów o największym gradiencie) to jego jedyny mechanizm ważności próbki. Pełnego
łańcucha „drzewo na cechach obrazu → wartość próbki → podzbiór treningowy detektora"
nie ma w 171 sprawdzonych pracach. Najbliżej: Sch25 (ekstrapolacja wyniku przez kNN/graf,
Spearman 0,16–0,70, oszczędność czasu 45–80%), Data-OOB z Jia23 (las losowy — **zawodzi
przy usuwaniu danych**), Kuz24 (las losowy + SHAP jako audyt błędów detekcji).

**E. Ważność próbki dla obrazów** (`docs/21`) — sześć rodzin; trzy rzeczy warte wzięcia:
- **DetGain** [Sun25]: wartość obrazu = przyrost mAP całego zbioru, liczony analitycznie;
  +2 mAP nad losowaniem na sześciu architekturach; **obrazy bez obiektu dają wkład ujemny**,
  co rozstrzyga punkt R2.
- **Migotanie detekcji w wideo** [Jin18]: detekcja o pewności ≥ 0,8 bez potwierdzenia
  w sąsiednich klatkach (IoU < 0,2) to fałszywy alarm; Caltech 23,83% → 18,72% pominięć,
  bez żadnych etykiet — nasza pula daje ten sygnał za darmo.
- **Learnability** [Eva24, JEST]: strata naszego modelu minus strata modelu odniesienia;
  wybór **całymi partiami** bije wybór po jednej próbce o 6% przy odsiewie 90%.

### Issues Encountered

- **`read_pdfs` zwróciło 93 706 znaków** — ponad limit odpowiedzi narzędzia. Wynik trafił
  do pliku w scratchpadzie; odczytałem go w pięciu kawałkach po ~240 linii.
- **Pierwsze uruchomienie pobieracza** wywaliło się na `FileNotFoundError`: równoległa sesja
  przeniosła mój plik z linkami z korzenia zbioru do `linki/`. Okazało się też, że zbiór ma
  **własną kopię pobieracza** z konfiguracją odsiewającą duplikaty po `pdf_all/` i `papers/`.
- **arXiv API zwróciło HTTP 429** po kilkunastu zapytaniach o tytuły z eksportu ResearchRabbit;
  wydłużenie odstępu nie pomogło (blokada okresowa). Obejście: OpenAlex, który dopełnił
  11 z 15 brakujących.
- **Klucze cytowań z DataCite** miały postać `https://doi.org/10.48550/arxiv...` — ukośniki
  łamią część parserów BibTeX, więc przepisałem je na `nazwisko+rok`.
- Katalog zbioru jest **zapisywany równolegle przez drugą sesję** (dokumenty audytowe,
  kopie narzędzi); moje zmiany ograniczyłem do `linki/`, `bibtex/` i plików bibliografii.

## Files Generated/Modified

**Projekt (`SmallViTAblation/docs/`)**
- `19_SELEKCJA_WG_LITERATURY.md` — NOWY: sześć rozbieżności, protokół docelowy, kalibracja
- `20_DRZEWA_GRADIENTOWE_JAKO_OBSERWATOR.md` — NOWY: LightGBM, GOSS, SHAP, werdykt
- `22_WAZNOSC_PROBKI_METODY_OBRAZOWE.md` — NOWY: sześć rodzin metod, ranking zastosowań

**Zbiór `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\`**
- `linki/linki-pdf-selekcja-rt-detr-2026-09-11.txt` — 14 linków aktywnych, 6 zakomentowanych
- `linki/waznosc-probki-i-selekcja-danych-2026-09-11.txt` — 33 linki, 8 sekcji tematycznych
- `linki/researchrabbit-eksport-2026-09-11.txt` — 49 linków z 65 wpisów `untitled.bib`
- `bibliografia-pobrane-pdf-2026-09-11.bib` — 29 wpisów, sklejony do wyszukiwarek
- `bibtex/<klucz>.bib` — 29 plików, po jednym na pracę
- `pdf_all/` — 65 nowych PDF-ów (razem 112)

**Pamięć projektu (`~/.claude/projects/.../memory/`)**
- `selekcja-klatek-weryfikacja-literatura.md` — dopisana sekcja o weryfikacji kodu
- `linki-pdf-zbior-rt-detr.md` — NOWY: gdzie zapisywać linki i zgoda na automatyczne pobieranie
- `MEMORY.md` — dwa wpisy indeksu

## Commands Used

```bash
# literatura (MCP Undermind)
get_orientation / inspect_deep_searches / search_papers / read_pdfs / launch_deep_search

# bibliografia
py -3.11 zbuduj_bib.py          # metadane z doi.org (DataCite/Crossref) -> .bib
py -3.11 porzadkuj_bib.py       # klucze nazwisko+rok, usuniecie smieciowych pol
py -3.11 rozbij_bib.py          # jeden plik .bib na prace
py -3.11 bib_na_linki.py        # untitled.bib -> linki, DOI po tytule z Crossref
py -3.11 bib_arxiv_dopelnij.py  # dopelnienie z API arXiv
py -3.11 bib_openalex.py        # dopelnienie z OpenAlex

# pobieranie (lokalna konfiguracja zbioru)
py -3.11 F:/Studia/Articles/Zbiory/_Narzedzia/PdfDownloader/pobierz_pdfy.py \
    --zbior "F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja" \
    "<nazwa-listy>.txt" --watki 8
```

## Next Steps

- [ ] Rozstrzygnąć: RF-DETR-Small zostaje modelem docelowym, RT-DETRv2 jako trzecia
      architektura transferu (decyzja użytkownika, wpływa na adapter i narrację artykułu)
- [ ] `protocol.yaml`: `proxy_subset_size` → pełna pula, `el2n.n_inits`, `dino_image_size`
      518/704, cechy obiektowe w ramce GT, `cluster_k_divisor`, budżet per klaster, `ccs_beta`
- [ ] `pool_manifest`: `bbox` → `bboxes` (Cataract-LMM ma średnio 1,4 narzędzia na klatkę)
- [ ] `el2n_from_proxy.py`: klatki tła poza rankingiem trudności; zmiana nazwy miary
- [ ] Nowy `pipeline/features/difficulty_from_checkpoints.py` (VPS/DUAL z checkpointów proxy)
- [ ] Obniżyć `budgets.tier_b` albo przedefiniować poziom B (17 832 < 20 000)
- [ ] Policzyć migotanie detekcji w wideo jako darmowy sygnał trudności [Jin18]
- [ ] Dopisać baseline czasowy (uniform-in-time, Random+R) i RS2 do macierzy
