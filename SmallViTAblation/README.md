# SmallViTAblation — przebudowa ablacji na mały ViT i poprawny protokół

**Gałąź:** `feature/small-vit-ablation-matrix`
**Start:** 2026-09-08
**Status:** PLAN — nic jeszcze nie policzone

## Po co ten folder

Artykuł IEEE Access dostał **odrzucenie od obu recenzentów** (jeden dopisał
„do not encourage resubmit"). Powodem nie była architektura ani sam pomysł, tylko
**protokół eksperymentu**: losowy podział klatek, jeden seed, ręcznie dobrane wagi
bez analizy czułości, ablacje trenowane inaczej niż pełny pipeline, niestandardowa
inferencja z jednego zapytania DETR i wielokrotne oglądanie zbioru zewnętrznego.

Ten folder zbiera **plan nowego eksperymentu**, który te zarzuty zamyka. Rdzeń
pomysłu: zejść z drogiego DETR-R50 (setki epok) na **mały detektor ViT**, żeby jeden
bieg kosztował minuty zamiast godzin — i za te same pieniądze policzyć **pełną
macierz ablacji, wiele ziaren losowych i analizy czułości**, których recenzenci
zażądali.

Ten folder zawiera **wyłącznie dokumenty i konfiguracje**. Kod treningu i selekcji
powstaje osobno, w `AdvancedDatasetSelection/` oraz `Eden/Scripts/`, i czyta
konfiguracje stąd.

## Nawigacja

| Plik | Co zawiera |
|---|---|
| `docs/00_RESEARCH.md` | co mówi literatura, jaki model wybieramy i dlaczego, z odnośnikami |
| `docs/01_PROTOCOL.md` | **specyfikacja protokołu** — to jest dokument nadrzędny |
| `docs/02_MATRIX.md` | pełna macierz wariantów, baseline'y, ziarna, budżety |
| `docs/03_COMPUTE_AND_OPTIMIZATION.md` | kosztorys, kalibracja, jak upakować ~140 biegów |
| `docs/04_REVIEWER_MAP.md` | mapowanie: zarzut recenzenta → konkretny eksperyment |
| `docs/05_OPEN_QUESTIONS.md` | co wymaga decyzji albo sprawdzenia w danych |
| `docs/06_TODO_KOD_I_EDEN.md` | **TODO kodu, uruchamianie na Edenie, szacunek czasu** |
| `docs/07_EDEN_INWENTARZ.md` | co już policzone na Edenie pod podziałem po pacjencie i co z tego użyć |
| `docs/08_AUDYT_WYCIEKOW.md` | **audyt wycieków** — który trening był czysty, który nie, z liczbami |
| `docs/09_ROZBUDOWA_DANYCH.md` | jak szybko poprawić zbiór, dokładając zdjęcia (Cataract-1K, negatywy) |
| `docs/10_TSFA_NOWA_PULA.md` | **budowa puli z surowego wideo przez TSFA** — 943 tys. klatek zamiast augmentacji |
| `docs/11_ZRODLA_NAGRAN.md` | **skąd wziąć więcej nagrań** — 6 300 publicznych nagrań, licencje, co pobrać najpierw |
| `configs/protocol.yaml` | jedno źródło prawdy dla hiperparametrów i splitu |
| `configs/variants.yaml` | maszynowa definicja macierzy |
| `tools/build_run_manifest.py` | rozwija konfiguracje do listy biegów |
| `tools/estimate_time.py` | liczy koszt eksperymentu z listy biegów |
| `tools/audit_leakage.py` | sprawdza wycieki między splitami: pacjent, klatka źródłowa, plik |

## Skala w dwóch liczbach

```
py -3.11 tools/build_run_manifest.py   ->  144 biegi
py -3.11 tools/estimate_time.py        ->  250-1000 GPU-godzin
                                           czyli 1,3-5,2 doby na 8 GPU
```

Widełki znikną po biegu kalibracyjnym. Dla porównania: jeden trening DETR
w starym podejściu szedł ponad trzysta epok na pełnym zbiorze.

## Zasada nadrzędna

> **Zmieniamy jedną rzecz na raz.** Wszystko poza badanym składnikiem — podział
> danych, budżet obrazów, inicjalizacja, optymalizator, harmonogram, kryterium
> stopu, reguła wyboru checkpointu, protokół ewaluacji — jest **identyczne dla
> wszystkich wariantów, łącznie z pełnym pipeline'em**.

Złamanie tej zasady w poprzednim podejściu (pełny pipeline trenowany produkcyjnie,
warianty w osobnym frameworku) jest powodem, dla którego [Tab. 5] artykułu nie
dowodzi tego, co miała dowodzić.
