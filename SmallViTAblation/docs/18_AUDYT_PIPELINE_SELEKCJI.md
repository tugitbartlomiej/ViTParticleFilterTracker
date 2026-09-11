# Audyt pipeline'u wyboru danych (`SmallViTAblation/pipeline/`)

Data: 2026-09-11. Zakres: `features/`, `selection/` — czyli droga od surowej
puli do pliku `annotations_train.json`, ktory idzie na trening.

## Werdykt

Pipeline **nie uruchamial sie**. Trzy bledy blokujace, wszystkie znalezione
przez uruchomienie lancucha od poczatku do konca na danych syntetycznych.
Dwa naprawilem, trzeci wymaga decyzji.

Po naprawie caly lancuch przechodzi: cache cech, EL2N, selekcja macierzowa,
cztery baseline'y, materializacja do COCO zweryfikowana przez pycocotools.

## Bledy blokujace

### 1. Selekcja macierzowa zawsze wybierala pusty zbior cech — NAPRAWIONE

`selection/select_subset.py`, linia 211 przed poprawka:

```python
active = [GROUP_KEY[g] for g in groups if GROUP_KEY[g] in GROUP_KEY]
```

`GROUP_KEY` mapuje `{"D": "dino", ...}`. Warunek sprawdzal, czy **wartosc**
(`"dino"`) jest **kluczem** tego samego slownika — a kluczami sa `D, E, S, F`.
Warunek byl wiec zawsze falszywy, `active` zawsze puste, i kazdy z 15 wariantow
macierzy konczyl sie wyjatkiem `need at least one array to concatenate`.

To jest serce eksperymentu: bez tego nie powstaje ani jeden podzbior macierzy.
Poprawka podstawia `if g not in GROUP_KEY` jako jawna walidacje i przepuszcza
wszystkie zadeklarowane grupy.

### 2. Generator danych testowych nie dzialal — NAPRAWIONE

`features/make_fixture.py` wolal `Image.FIND_EDGES`; ta stala mieszka w
`ImageFilter`, nie w `Image`. Skrypt wywalal sie na pierwszym obrazie, wiec
**zadnego testu dymnego nigdy nie dalo sie przeprowadzic** — i dlatego blad
numer 1 przetrwal.

### 3. Brakuje szesciu kluczy konfiguracji — DO DECYZJI

`features/build_cache.py` w trybie produkcyjnym czyta z `features.*`:
`dino_model`, `dino_image_size`, `extract_batch_size`, `fastsam_weights`,
`fastsam_conf`, `fastsam_iou`. **Zadnego z nich nie ma w `configs/protocol.yaml`.**
Dziala tylko `--fake-features`, czyli cechy zastepcze.

Nie dopisalem ich sam, bo to nie jest literowka: trzeba wybrac konkretny wariant
DINOv2, rozdzielczosc wejscia i wagi FastSAM, a te decyzje ida do artykulu.

## Problemy metodologiczne

### EL2N liczy sie czesciowo na danych, ktore proxy widzial

`el2n.proxy_subset_size` to 2500 klatek losowanych z puli treningowej. Proxy
uczy sie na nich, a potem `el2n_from_proxy.py` liczy blad na **calej** puli.
Te 2500 klatek model juz zna, wiec dostana systematycznie nizszy blad niz
reszta. Trudnosc klatki miesza sie z tym, czy trafila do podzbioru proxy.

Skala: przy puli 20 tysiecy to 12 procent klatek z zanizonym wynikiem.
Nierówną ekspozycję trzeba zmierzyć. Możliwe projekty to wczesny trening
proxy na całej puli train (zgodny z ideą EL2N) albo jawny scoring cross-fit.
Nie ma obowiązku oceniania każdej próbki modelem, który jej nie widział.
Sam train-on-train scoring nie oznacza wycieku. Wszystkie warianty pozostają
odseparowane od val/test.

### To, co kod nazywa EL2N, nie jest EL2N

Wzor w kodzie to `1 - pewnosc najlepszej detekcji dopasowanej po IoU`.
Oryginalny EL2N (Paul i wsp., 2021) to norma L2 roznicy miedzy rozkladem
prawdopodobienstwa a wektorem jednostkowym klasy prawdziwej, usredniona po
kilku niezaleznych inicjalizacjach. Tutaj usredniania po inicjalizacjach nie
ma. Dla jednego wyjścia sigmoid dodatniego targetu norma wynosi `1-p`;
dla dwóch wyjść softmax (obiekt/tło) wynosi `sqrt(2)*(1-p)`. Jednoklasowy
softmax bez tła jest degeneracją. Problem obejmuje też thresholding,
dopasowanie po IoU, pomijanie klas i zachowanie przy braku GT.

To moze byc dobra miara trudnosci i moze zostac. Ale **nie wolno jej nazwac
EL2N w artykule** — recenzent sprawdzi definicje w cytowanej pracy. Albo
zmieniamy nazwe na `detection_confidence_error`, albo implementujemy wzor
zgodnie z oryginalem.

### Klatki bez narzedzia dostaja maksymalna trudnosc

`scores` startuje z jedynek i klatka bez ramki GT zostaje z wartoscia 1.0,
czyli najwyzsza mozliwa. To zaciera różnicę między łatwym tłem i tłem z FP.
Przy selekcji centroidów zmienia geometrię, ale nie dowodzi automatycznie
preferowania tła; udział negatywów trzeba zmierzyć. Jesli taki jest zamiar — trzeba to
zapisac w protokole. Jesli nie — tlo powinno dostawac osobna sciezke, nie
maksymalny blad.

### Kontrakt manifestu dopuszcza jedna ramke na klatke

`pool_manifest.jsonl` ma pole `bbox` w liczbie pojedynczej i tak samo czyta je
`materialize.py`. Tymczasem Cataract-LMM ma srednio 1,4 narzedzia na klatke,
a 30 procent klatek ma ich dwa lub wiecej. Przy obecnym kontrakcie drugie
narzedzie znika, a model uczy sie, ze go tam nie ma — czyli dostaje falszywy
negatyw na kazdej takiej klatce. **To moj blad w specyfikacji dla agentow**,
nie ich. Trzeba zmienic pole na liste `bboxes`.

## Co dziala poprawnie

- **Pula wyboru to wylacznie `split == "train"`** (`common.train_pool`),
  sprawdzone na wszystkich czterech wariantach: zero klatek spoza train.
  Regula P2 protokolu jest respektowana.
- **Determinizm**: powtorzona selekcja z tym samym ziarnem daje identyczna
  liste 50 klatek co do pozycji.
- **Wagi dzialaja**: M-DESF i M-D przy tym samym ziarnie maja tylko 22 wspolne
  klatki z 50, a `selection_meta` zapisuje uzyte wagi.
- **Renormalizacja proporcjonalna** przy wylaczaniu grupy jest zaimplementowana
  zgodnie z `disable_rule` w protokole.
- **Materializacja** odrzuca klatki spoza train jawnym wyjatkiem i produkuje
  COCO, ktory przechodzi walidacje pycocotools.
- **Obrazy nie sa kopiowane** — `file_name` wskazuje wspolny katalog. Przy
  144 biegach to roznica miedzy kilkoma gigabajtami a kilkuset.

## Kolejnosc dzialan

1. Uzupelnic szesc kluczy `features.*` w `protocol.yaml` (decyzja o modelach).
2. Rozstrzygnac nazwe i definicje miary trudnosci.
3. Zmienic `bbox` na `bboxes` w kontrakcie manifestu i w `materialize.py`
   oraz `train_el2n_proxy.py`.
4. Zdecydowac, co z klatkami bez narzedzia w scoringu.
5. Dopiero potem liczyc cechy na prawdziwej puli.
