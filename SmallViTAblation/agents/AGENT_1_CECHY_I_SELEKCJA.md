AGENT 1 — cechy obrazów i wybór podzbiorów
==========================================

Przeczytaj najpierw `README.md` w tym katalogu. Ten plik zakłada, że go znasz.

**Twój katalog:** `pipeline/features/` i `pipeline/selection/`
**Nie dotykasz:** niczego poza nimi.

---

## Co budujesz i po co

Serce badania. Z puli setek tysięcy klatek trzeba wybrać podzbiór o zadanym
rozmiarze, w piętnastu różnych wariantach, i sprawdzić, który wybór daje lepszy
model. Ty budujesz część wybierającą.

Pipeline opisuje każdą klatkę **czterema grupami liczb**:

| Kod | Grupa | Co mierzy | Wymiar | Waga bazowa |
|---|---|---|---|---|
| `D` | DINOv2/v3 | treść sceny, semantyka | 32 po redukcji PCA | 0,35 |
| `E` | EL2N | jak trudna jest klatka dla modelu | 1 | 0,30 |
| `S` | FastSAM | jak złożona jest scena | 9 | 0,20 |
| `F` | Fourier | ostrość i jakość obrazu | 1 | 0,15 |

Te cztery bloki skleja się w jeden wektor na klatkę, mnoży przez wagi, grupuje
metodą K-średnich na tyle grup, ile ma wynieść budżet, i z każdej grupy bierze
jednego przedstawiciela.

## Zadanie 1 — `pipeline/features/build_cache.py`

Policz cechy `D`, `S`, `F` **raz** dla wszystkich klatek z `split == "train"`
i zapisz do cache'u.

- wejście: `pool_manifest.jsonl`, katalog z obrazami,
- wyjście: `feature_cache/dino.pkl`, `sam.pkl`, `fourier.pkl`, każdy jako
  tablica NumPy plus **lista `frame_id` w tej samej kolejności**,
- dopisz `feature_cache/manifest.json`: ile klatek, jakie wymiary, jaki model
  i jego wersja, data.

Wymagania twarde:

- **kolejność klatek identyczna we wszystkich trzech plikach**, inaczej sklejenie
  wektorów da bzdury,
- liczenie **wznawialne** — przerwane w połowie ma dokończyć, a nie zaczynać od
  zera; setki tysięcy klatek to godziny pracy karty,
- postęp wypisywany na bieżąco.

W projekcie są już starsze wersje tych ekstraktorów w
`AdvancedDatasetSelection/feature_extractors/`. Możesz je wykorzystać, ale
**cache musi mieć nowy, jednolity format** opisany wyżej.

## Zadanie 2 — `pipeline/features/train_el2n_proxy.py`

EL2N mierzy, jak trudna jest klatka: liczy się to z **modelu trenowanego
krótko**, wcześnie, a nie z modelu douczonego do końca. To jest istota metody
i poprzednia wersja pracy łamała tę zasadę.

- wytrenuj **model wstępny** (RF-DETR-Small) przez liczbę epok z
  `protocol.yaml → el2n.proxy_epochs` (obecnie 10),
- **wyłącznie na losowym podzbiorze klatek treningowych**, nigdy na `val`/`test`,
- **jeden model wstępny na ziarno losowe**, bo EL2N zależy od ziarna,
- zapisz `feature_cache/el2n_proxy_<seed>.pth`.

## Zadanie 3 — `pipeline/features/el2n_from_proxy.py`

Oceń modelem wstępnym **całą pulę treningową** i zapisz
`feature_cache/el2n_<seed>.pkl` w formacie jak inne cechy.

To jest sama inferencja, więc szybka. Wynik znormalizuj do przedziału od zera
do jedynki i zapisz w metadanych, jak dokładnie.

## Zadanie 4 — `pipeline/selection/select_subset.py`

**Jeden skrypt dla wszystkich piętnastu kombinacji.** Nie pisz piętnastu funkcji.

Wejście: `queue.jsonl` (jeden wiersz to jeden bieg), cache cech, `protocol.yaml`.
Wyjście: `variants/<run_id>.json` w formacie z `README.md`, sekcja 4.2.

Algorytm:

1. weź z wiersza kolejki listę aktywnych grup cech (`groups`) i wagi (`weights`),
2. **reguła wyłączenia grupy**: waga zero, pozostałe **przeskalowane
   proporcjonalnie** tak, by sumowały się do jedynki. Wagi są już policzone
   w `queue.jsonl` — użyj ich, nie licz własnych,
3. każdą grupę cech **wystandaryzuj osobno** (średnia zero, odchylenie jeden),
4. DINO zredukuj metodą PCA do `features.dino_pca_dim` wymiarów,
5. sklej bloki, przemnóż przez wagi,
6. pogrupuj metodą z `features.cluster_algorithm` na tyle grup, ile wynosi
   `budget`, z ziarnem `seed`,
7. z każdej grupy weź przedstawiciela wg `features.cluster_representative`.

Wymagania twarde:

- **wynik ma być powtarzalny**: to samo ziarno daje ten sam podzbiór co do
  klatki,
- liczba wybranych klatek **dokładnie** równa `budget`,
- gdy grupa wyjdzie pusta, uzupełnij deterministyczną regułą i **opisz ją
  w `README.md`**,
- selekcja widzi **tylko** klatki z `split == "train"`.

## Zadanie 5 — `pipeline/selection/baselines.py`

Cztery punkty odniesienia. Trzeci i czwarty to bezpośrednie żądania recenzenta.

| ID | Definicja |
|---|---|
| `B-RANDAUG` | losowe `budget` klatek z puli treningowej |
| `B-RANDSRC` | losuj **klatki źródłowe** (przed augmentacją), weź wszystkie ich warianty aż do `budget` |
| `B-UNIFORM` | równa liczba klatek z każdego pacjenta treningowego |
| `B-FULLPOOL` | wszystkie klatki treningowe, bez wyboru |

`B-RANDSRC` wymaga rozpoznania, które klatki są wariantami tej samej klatki
źródłowej. W obecnych danych poznaje się to po nazwie: `..._aug_1`, `..._aug_2`,
`..._aug_3` to warianty klatki bez tego przyrostka. W nowej puli z wideo tej
augmentacji nie ma — wtedy `B-RANDSRC` to po prostu losowanie po klatkach
źródłowych. Obsłuż oba przypadki i **wypisz w logu, którym poszedłeś**.

## Zadanie 6 — `pipeline/selection/materialize.py`

Z pliku `variants/<run_id>.json` zbuduj `annotations_train.json` w formacie COCO,
gotowy do treningu.

**Nigdy nie kopiuj obrazów.** Anotacje wskazują na obrazy leżące raz, we
wspólnym katalogu. Kopiowanie stu czterdziestu czterech podzbiorów po pięć
tysięcy obrazów to setki gigabajtów bez żadnego zysku.

## Test końcowy

1. `build_cache.py` na 2 500 klatkach → trzy pliki cache o zgodnej kolejności,
2. `select_subset.py` dla `M-DESF` i `M-D` przy budżecie 500 → dwa różne
   podzbiory, każdy po 500 klatek,
3. to samo ziarno uruchomione dwa razy → **identyczny wynik**,
4. `baselines.py` dla wszystkich czterech → cztery pliki,
5. `materialize.py` → poprawny plik COCO, który wczytuje `pycocotools`.

## Pułapki, na które ktoś już wpadł

- **Wagi po wyłączeniu grupy dobierano wcześniej ręcznie**, więc wariant „bez
  Fouriera" różnił się od pełnego także proporcjami pozostałych cech. Dwie
  zmiany naraz. Dlatego reguła proporcjonalnego przeskalowania jest obowiązkowa.
- **EL2N liczono z modelu trenowanego 170 epok na całej puli** — czyli z modelu,
  który widział też dane testowe. To nie jest EL2N i nie wolno tego powtórzyć.
- **Selekcja jechała po całej puli**, łącznie z klatkami pacjentów testowych.
  To był wyciek, o którym recenzenci nawet nie wiedzieli.
