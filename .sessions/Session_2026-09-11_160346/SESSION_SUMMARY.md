# Session Summary: Session_2026-09-11_160346

## Metadata
- **Date:** 2026-09-11
- **Time:** 16:03:46
- **Status:** Completed
- **Type:** Mixed (Analysis + Literature + SSH)

## Objective
Trzy rzeczy, w kolejności zgłoszenia przez użytkownika:

1. Dokończyć w tle transfer Cataract-LMM na `eden-cluster` i potwierdzić kompletność.
2. Znaleźć, jak połączyć analizę koopmanowską latentu
   (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_koopman_hybrydowy_h1`)
   z pipelinem selekcji danych wchodzących do treningu
   (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs`).
3. Zapisać znalezione prace do kolejki PdfDownloadera i pobrać je.

## Context
Sesja jest kontynuacją pracy nad przebudową odrzuconego artykułu do IEEE Access.
Przed nią powstały dokumenty `16`–`20` w `SmallViTAblation/docs`, w tym audyt pipeline'u
selekcji (`18`), weryfikacja protokołu wobec literatury (`19`) i analogiczne pytanie
o LightGBM jako sędziego ważności danych (`20`). Pytanie o Koopmana jest tego samego
rodzaju co pytanie z dokumentu `20`: czy narzędzie z innej dziedziny nadaje się na
tani miernik informatywności klatki.

Transfer na Eden startował w poprzedniej sesji; użytkownik poprosił, żeby szedł w tle
bez meldunków.

## Actions Taken

### 1. Transfer na Eden
Zadanie w tle `b1bsw73a9` czekało na zakończenie lokalnego procesu `tar`, a potem
sprawdziło stan po stronie klastra przez SSH.

### 2. Wyszukiwanie literatury (skill `paper-search`)
- `get_orientation`, `list_workspaces`, `inspect_deep_searches` na obu workspace'ach
  (ViT `40294fa6-…`, grantowy LeJEPA `bb70909d-…`) — sprawdzenie, czy pasująca kwerenda
  już nie istnieje. Nie istniała.
- `launch_deep_search` w workspace ViT: "operator Koopmana na latencie wideo jako
  kryterium selekcji klatek do treningu detektora" → 139 prac.
- Cztery iteracje `search_papers` pod różnymi kątami: residuum operatora jako miara
  informatywności, czas dekorelacji z widma, przewidywalność zamiast niepewności modelu,
  efektywna liczba niezależnych klatek.
- `read_pdfs` — jedenaście prac czytanych w całości w dwóch wsadach
  (Mas21, Rui25, Wu23, Sha25b, Mig24, Bap19 — PDF niedostępny; Kim26, Bi18, Kha24c, Zho25, Ben19).
- Scite jako kontrola krzyżowa: zapytanie łączące Koopmana/DMD z selekcją danych zwróciło
  **zero trafień**; szersze zapytanie dało wyłącznie prace ze sterowania i robotyki,
  gdzie "active learning" znaczy wybór następnego eksperymentu, nie wybór klatek.

### 3. Kolejka i pobieranie
- `nowa_lista.py --zbior … --nowa … --z-pliku …` założyła listę w zbiorze RT-DETR.
- `pobierz_pdfy.py` w tle (zadanie `bk189e4mw`).

## Results

### Key Findings

**Transfer na Eden — kompletny.**

| wielkość | wartość |
|---|---|
| archiwa śledzenia | 170 / 170 |
| klatki segmentacji | 6 094 / 6 094 |
| rozmiar na Edenie | 44 GB |
| wolne miejsce po transferze | 32 GB z 200 GB |
| kod wyjścia | 0 |

**Werdykt literaturowy: bezpośredniego połączenia nie ma.**
Literatura konsekwentnie rozdziela modelowanie dynamiki latentu od selekcji danych.

Najbliżej hipotezy stoi **Swift Sampling** [Kim26]: liczy "temporal surprise" jako normę
residuum ekstrapolacji Taylora na cechach z pierwszej warstwy enkodera wizyjnego,
wybiera lokalne maksima residuum. Narzut 1,02× wobec próbkowania równomiernego,
zysk do +6,8 punktu na długich nagraniach. Praca **nie wspomina** o Koopmanie ani DMD —
to jest dokładnie luka, którą można zająć.

Po stronie operatorowej istnieją mocne analogie, ale żadna nie mierzy jakości detektora:

- [Bi18] DMD do wykrywania cięć montażowych, progowana amplituda modu tła, F1 0,885.
- [Kha24c] CPDMD — residuum rekonstrukcji DMD jako statystyka wykrywania punktu zmiany,
  opóźnienie ~7 kroków, kalibracja przez EWMA.
- [Mig24] DMD w czasie rzeczywistym, macierz 5×5 na okno, ale **wrażliwy na ruch kamery
  i zmiany oświetlenia** — w chirurgii występują oba.
- [Mas21] KIDD uczy operatora Koopmana na latencie wideo, ale residuum używa wyłącznie
  jako składnika straty, nigdy jako oceny klatki.
- [Rui25] pokazuje, że JEPA sama uczy się niezmienników Koopmana, bez estymacji operatora —
  ale też nie wyprowadza z tego żadnej oceny próbki.

**Most jest w równaniu v8 projektu grantowego.** `ROWNANIE.md` zawiera już człon
`ε_{t+1} = z_{t+1} - ẑ_{t+1}`. W projekcie grantowym to błąd prognozy; w pipelinie selekcji
ten sam `ε` jest miarą „ile nowego wnosi klatka". Różnica sprowadza się do źródła latentu:
zamrożony TFT kontra zamrożony DINOv2. Rejestr przesuwający `h` i bramki lokalne przenoszą
się jeden do jednego. Drugi transfer to widmo operatora — wartości własne wyznaczają czas
dekorelacji nagrania, czyli krok próbkowania klatek.

**Skala oczekiwanego zysku — zgodna z dokumentem `19`.**

| praca | dziedzina | przewaga nad losowaniem |
|---|---|---|
| [Wu23] COWAL | segmentacja chirurgiczna | +2,0 AuALC (średnio) |
| [Ben19] temporal coherence | detekcja na wideo | +2,1 mAP |
| [Zho25] | segmentacja laparoskopowa | połowa danych = ten sam mIoU |
| [Sha25b] StepAL | rozpoznawanie kroków zaćmy | +13,7 pkt (ale wybór całych filmów, nie klatek) |
| [Yoo23] | segmentacja instancji | 20% klatek dorównuje pełnemu zbiorowi |

Prace ograniczające: [Mit23] pokazuje zależność zysku od rozkładu danych,
[Aye23] daje fundamentalne granice metod punktowych przy silnej kompresji zbioru.

**Luka badawcza do zajęcia:** porównać residuum operatora Koopmana z residuum Taylora,
z różnicą cech, z optical flow i z losowaniem — mierząc mAP wytrenowanego detektora,
a nie samą trafność wykrycia zmiany.

### Issues Encountered

| problem | rozwiązanie |
|---|---|
| Deep search Undermind trwał ~10 minut, dłużej niż typowe 2–5 min | Czas wykorzystany na cztery `search_papers` i wsad `read_pdfs`, zamiast biernego czekania |
| Scite: pierwsze zapytanie zwróciło 0 wyników, drugie 218 tysięcy znaków (przekroczyło limit) | Wynik zapisany do pliku, tytuły wyciągnięte przez `grep -o` zamiast czytania całości |
| [Bap19] — PDF niedostępny (paywall IEEE TIP) | Praca została w kolejce do pobrania przez Primo |
| 8 z 25 linków nieudanych przy pobieraniu | Wszystkie to paywall IEEE i Elsevier; raport w `nieudane\nieudane_linki.html`, kolejka Primo przez `odswiez_kolejke.bat` |
| Pliki `.bat` zbioru RT-DETR zostały przeniesione do `scripts\`, przez co `--zbior "%~dp0."` wskazuje teraz na `scripts\`, a nie na korzeń zbioru | **Nienaprawione** — patrz Next Steps. Obejście: wołanie `.py` z `_Narzedzia` z jawnym `--zbior` |

## Files Generated/Modified

Utworzone:

- `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\linki\koopman-i-dmd-na-latencie-a-selekcja-klatek-2026-09-11.txt` — 25 linków w czterech sekcjach
- 16 PDF-ów w `…\2026-09-11_RT-DETR_selekcja_ewaluacja\pdf_all\`
- Deep search w workspace Undermind ViT (139 prac)

Bez zmian w kodzie projektu. Stan repozytorium ten sam co na początku sesji.

## Commands Used

```bash
# transfer - sprawdzenie po stronie klastra
ssh eden-cluster 'du -sh $HOME/Cataract-LMM; ls .../annotations/*.zip | wc -l; df -h $HOME'

# kolejka linkow
py -3.11 nowa_lista.py --zbior "...\2026-09-11_RT-DETR_selekcja_ewaluacja" \
    --nowa "Koopman i DMD na latencie a selekcja klatek" --po-co "..." --z-pliku "..."

# pobieranie (.py, nie .bat - launcher konczy sie pause i zawisa w tle)
py -3.11 pobierz_pdfy.py --zbior "...\2026-09-11_RT-DETR_selekcja_ewaluacja" \
    "linki\koopman-i-dmd-na-latencie-a-selekcja-klatek-2026-09-11.txt"
```

## Next Steps

- [ ] **Naprawić `.bat` w zbiorze RT-DETR.** Po przeniesieniu do `scripts\` flaga
      `--zbior "%~dp0."` wskazuje na zły katalog. Albo wrócić plikami do korzenia
      (tak każe `AGENTS.md` tego zbioru), albo poprawić na `%~dp0..`.
- [ ] Uruchomić `odswiez_kolejke.bat`, żeby ośmiu nieudanych pobrać przez Primo —
      w tym [Bi18] i [Yan24], najbardziej domenowo trafne.
- [ ] Napisać `docs/21_KOOPMAN_JAKO_MIARA_INFORMATYWNOSCI.md` na wzór dokumentu `20`,
      z werdyktem i planem eksperymentu porównawczego.
- [ ] Zrobić OCR pobranych PDF-ów (`skanuj_ocr.bat`) — osobne narzędzie, poza tą sesją.

### Zaległości z poprzednich sesji (niezałatwione)

- [ ] Sześć brakujących kluczy `features.*` w `configs/protocol.yaml` (wariant DINOv2,
      rozdzielczość wejścia, wagi i progi FastSAM) — decyzja użytkownika, wchodzi do artykułu.
- [ ] Nazwa i definicja miary trudności nazywanej teraz EL2N.
- [ ] `bbox` → `bboxes` w kontrakcie manifestu, `materialize.py` i `train_el2n_proxy.py`.
- [ ] Rozstrzygnąć punktację klatek bez narzędzia (teraz dostają maksymalną trudność 1,0).
- [ ] Wyjaśnić licencję Cataract-LMM z autorami i poprosić o Global ID dla `3_Object_Tracking`.
- [ ] Zamaskować wypalony stempel czasowy i nazwę szpitala, zanim klatka wejdzie do treningu.
