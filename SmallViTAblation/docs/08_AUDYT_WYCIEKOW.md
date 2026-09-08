# 08 — Audyt wycieków: który trening DETR był czysty, a który nie

Sprawdzone 2026-09-08 skryptem `tools/audit_leakage.py`, uruchomionym na
rzeczywistych plikach anotacji. Skrypt sprawdza trzy poziomy naraz, bo każdy
łapie inny rodzaj błędu:

1. **pacjent** — czy ten sam pacjent jest w dwóch częściach,
2. **klatka źródłowa** — czy ta sama klatka jest w dwóch częściach pod różnymi
   wariantami augmentacji (oryginał w treningu, `_aug_2` w teście); tego nie
   widać po nazwach plików,
3. **plik** — czy dokładnie ten sam plik jest w dwóch częściach.

```bash
py -3.11 SmallViTAblation/tools/audit_leakage.py --all
```

---

## 1. Zbiory ablacji i pełna pula: CZYSTE

| Zbiór | Trening | Walidacja | Test | Werdykt |
|---|---|---|---|---|
| `fullpool_videosplit` | 71 316 obr. / 35 pacjentów | 10 780 / 8 | 9 240 / 7 | **czysty** |
| `ablation/v1` … `v7` | 20 000 obr. / 35 pacjentów | 10 780 / 8 | 9 240 / 7 | **czyste, wszystkie siedem** |

Zero wspólnych pacjentów, zero wspólnych klatek źródłowych, zero wspólnych
plików w każdej z trzech par porównań. Osiem zbiorów sprawdzonych, osiem
czystych.

Liczba klatek źródłowych w treningu różni się między wariantami — od 10 937
(v4) do 13 030 (v5) przy tym samym budżecie 20 000 obrazów. To znaczy, że
warianty różnią się tym, ile kopii augmentowanych tej samej klatki biorą.
Sama w sobie ciekawa obserwacja o zachowaniu pipeline'u.

## 2. Produkcyjny zbiór 20k — ten od modelu z artykułu: NIECZYSTY

To jest zbiór, na którym powstał model `checkpoint_epoch_270.pth`, czyli
headline artykułu (85,74%). Plik: `merged_20k_annotations_fixed.json`.

Sprawdzone przez przyłożenie podziału po pacjencie do jego listy plików:

| Skąd pochodzą klatki | Ile | Udział |
|---|---:|---:|
| pacjenci **treningowi** | 15 648 | 78,2% |
| pacjenci **walidacyjni** | 2 322 | 11,6% |
| pacjenci **testowi** | 1 982 | 9,9% |
| **spoza CATARACTS** (benchmark zewnętrzny) | **48** | 0,2% |

Czyli **21,5% zbioru treningowego to klatki pacjentów, którzy w poprawnym
podziale są zbiorem walidacyjnym i testowym**. Model widział w treningu
prawie co piątą klatkę od osób, na których potem mierzono jego jakość.

Do tego trening używał `random_split()` na poziomie klatek, więc jego własna
walidacja też pochodziła od tych samych pacjentów. Wyciek działał w obie strony.

## 3. Czterdzieści osiem obrazów z benchmarku zewnętrznego w treningu

Te 48 obrazów spoza CATARACTS to **dokładnie ta sama lista**, która leży
w `YOLO_DETR_Benchmarks/scripts/leaked_valid_images.json` (opis w pliku: 48
pozycji, część walidacyjna benchmarku zewnętrznego). Sprawdzone przecięciem
zbiorów: **48 z 48 pasuje**.

To znaczy, że model z artykułu widział w treningu 48 obrazów z benchmarku,
który w tekście jest opisany jako „strict zero-look". Było to już wcześniej
w projekcie wykryte (sesja 2026-01-16) i dlatego istnieje wariant benchmarku
**CLEAN** — FULL minus te 48 obrazów. Ale w artykule liczby headline'owe stoją
na wariancie FULL.

## 4. Wniosek

**Odpowiedź na pytanie „czy trening DETR był na zbiorze per pacjent":**
zależy który. Ablacja tak, produkcja nie.

| Trening | Podział | Wyciek |
|---|---|---|
| ablacja V1–V7 | po pacjencie | brak w danych, **ale start z checkpointu ep170 z wyciekiem** (patrz `07_EDEN_INWENTARZ.md`) |
| `fullpool_vs_scratch` | po pacjencie | brak |
| `scratch_20k/v1` | po pacjencie | brak |
| **produkcyjny 20k, ep270 — headline artykułu** | losowy po klatkach | **21,5% klatek od pacjentów val/test + 48 obrazów z benchmarku** |

Czyli headline artykułu stoi na modelu z podwójnym wyciekiem: pacjenci
i benchmark zewnętrzny. Ablacja miała czyste dane, ale dziedziczyła wyciek
przez inicjalizację z ep170.

**Nowy protokół zamyka oba kanały naraz:** dane budowane wyłącznie z pacjentów
treningowych (`01_PROTOCOL` P2) i start wyłącznie z wag COCO (`configs/protocol.yaml`,
`init: coco_pretrained`).

## 5. Co z tego napisać w artykule

Nie chować. Zdanie w sekcji o protokole: wcześniejsze wyniki na tym zbiorze
powstały przy podziale na poziomie klatek, nowe wyniki używają podziału po
pacjencie i modelu inicjalizowanego wagami COCO. Recenzent, który zobaczy, że
autor sam znalazł i zmierzył wyciek, czyta resztę inaczej niż recenzent, który
go znajduje sam.
