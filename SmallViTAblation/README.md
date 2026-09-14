# SmallViTAblation — przebudowa ablacji na mały ViT i poprawny protokół

**Gałąź:** `feature/small-vit-ablation-matrix`
**Start:** 2026-09-08
**Status:** IMPLEMENTACJA + TESTY LEKKIE — bez treningów produkcyjnych

## Po co ten folder

Artykuł IEEE Access dostał **odrzucenie od obu recenzentów** (jeden dopisał
„do not encourage resubmit"). Powodem nie była architektura ani sam pomysł, tylko
**protokół eksperymentu**: losowy podział klatek, jeden seed, ręcznie dobrane wagi
bez analizy czułości, ablacje trenowane inaczej niż pełny pipeline, niestandardowa
inferencja z jednego zapytania DETR i wielokrotne oglądanie zbioru zewnętrznego.

Ten folder zbiera **plan nowego eksperymentu**, który te zarzuty zamyka. Rdzeń
pomysłu: zejść z drogiego DETR-R50 (setki epok) na **małe detektory**, żeby jeden
bieg kosztował minuty zamiast godzin. YOLOv8n służy do szerokiego screeningu,
a małe transformery do potwierdzenia transferu — dzięki temu można policzyć **pełną
macierz ablacji, wiele ziaren losowych i analizy czułości**, których recenzenci
zażądali.

Ten folder zawiera dokumenty, konfiguracje i wykonywalny kod selekcji, aktywnej
kuracji, diagnostyki latentnej, treningu, ewaluacji i kolejki Eden. Kosztowne
treningi pozostają do uruchomienia po dostarczeniu obrazów Cataract-LMM.

## Nawigacja

Aktualna kolejność startu na klastrze: [RF-DETR-Nano, smoke i statyczna kuracja](docs/Analiza_ponownej_publikacji/30_START_EDEN_RFDETR_NANO_SMOKE.md).
Profile `nano_smoke` i `nano_static_pilot` rozpoczynają badanie bez diagnostyki dynamiki latentnej.

**Zacznij od `PLAN.md`** — plan uporządkowany według zarzutów recenzentów,
z podziałem na to, co już zamknięte, co wymaga obliczeń i w jakiej kolejności.

| Plik | Co zawiera |
|---|---|
| **`PLAN.md`** | **dokument wejściowy: zarzut → co go zamyka → stan → gdzie** |
| **`agents/`** | **instrukcje dla pięciu agentów** — kontrakty, podział pracy, definicja gotowości |
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
| `docs/12_LINKI_DO_POBRANIA.md` | **gotowa lista linków** do zbiorów, które można samemu oznaczyć |
| `docs/13_STANDARD_DANYCH.md` | **jak przygotować dane** — rozdzielczość, pole widzenia, format, metadane |
| `docs/14_SUROWE_FILMY_PLAN_POBRANIA.md` | **surowe filmy** — rozmiary pobrania, budżet dysku, kolejność |
| `docs/15_PIPELINE_PRZETWARZANIA.md` | **potok od filmu do tabeli** — co gdzie leży, na jakim modelu ablacje |
| `docs/16_SMOKE_TEST_LOKALNY.md` | **smoke test na RTX 3070** — co da się udowodnić lokalnie, a czego nie |
| `configs/protocol.yaml` | jedno źródło prawdy dla hiperparametrów i splitu |
| `configs/variants.yaml` | maszynowa definicja macierzy |
| `tools/build_run_manifest.py` | rozwija konfiguracje do listy biegów |
| `tools/estimate_time.py` | liczy koszt eksperymentu z listy biegów |
| `tools/audit_leakage.py` | sprawdza wycieki między splitami: pacjent, klatka źródłowa, plik |
| `configs/profiles.yaml` | ograniczone profile smoke, pilot, confirm i external-once |
| `pipeline/active_learning/` | kumulatywna selekcja klatek przy stałym budżecie |
| `pipeline/latent_dynamics/` | cross-fitowane kontrasty Markow/historia i benchmark syntetyczny |
| `pipeline/transfer/train_transfer.py` | adapter YOLOv8n i oficjalnego RT-DETRv2-S |
| `eden/` | odporna kolejka wielokartowa z heartbeat i tokenem próby |

## Skala w dwóch liczbach

```
py -3.11 tools/build_run_manifest.py              -> bezpiecznie tylko Tier A
py -3.11 tools/build_active_manifest.py --help    -> macierz smoke/pilot/confirm
```

Dokładny koszt zapisuje profil oraz wygenerowany plik blokady eksperymentu.
Dla porównania jeden trening DETR w starym podejściu szedł ponad trzysta epok
na pełnym zbiorze.

## Zasada nadrzędna

> **Zmieniamy jedną rzecz na raz.** Wszystko poza badanym składnikiem — podział
> danych, budżet obrazów, inicjalizacja, optymalizator, harmonogram, kryterium
> stopu, reguła wyboru checkpointu, protokół ewaluacji — jest **identyczne dla
> wszystkich wariantów, łącznie z pełnym pipeline'em**.

Ta równość obowiązuje w obrębie architektury. Profile blokują rozdzielczość
osobno dla modelu, dlatego wyników YOLO, RT-DETRv2 i RF-DETR nie interpretuje się
jako porównania szybkości lub jakości architektur. Między modelami sprawdzamy
wyłącznie, czy kierunek efektu selekcji się przenosi.

Złamanie tej zasady w poprzednim podejściu (pełny pipeline trenowany produkcyjnie,
warianty w osobnym frameworku) jest powodem, dla którego [Tab. 5] artykułu nie
dowodzi tego, co miała dowodzić.
