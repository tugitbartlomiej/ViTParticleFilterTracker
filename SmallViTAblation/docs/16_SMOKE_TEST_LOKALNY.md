Smoke test lokalny — czy protokół selekcji w ogóle da się sprawdzić na RTX 3070
==============================================================================

Data: 2026-09-10. Analiza, nie instrukcja. Odpowiada na pytanie: **co da się
udowodnić lokalnie, a czego nie da się w żadnym razie.**

---

## 1. Znalezisko, które zmienia punkt wyjścia

**Obrazów puli nie ma na dysku lokalnym.** W
`Eden/Datasets/datasets_20250606_test/.../images/` leży **dziewięć plików**
i plik anotacji na 36,7 MB. Reszta, czyli 91 336 obrazów, siedzi w archiwum
na Edenie.

Ale **nagrania są lokalnie** — 49 filmów w `E:\Cataract\videos\micro`. I to
wystarcza, bo sprawdziłem, że pulę da się odtworzyć.

### Reguła odwzorowania — zweryfikowana na danych

Nazwa pliku `test01_frame_0000868.jpg` odpowiada klatce wideo o numerze **867**,
czyli numeracja w nazwie jest o jeden wyższa niż indeks klatki w pliku.

Sprawdzone na wszystkich trzech nieaugmentowanych obrazach, jakie są lokalnie:

| Nazwa pliku | Indeks w wideo | Średnia różnica jasności |
|---|---:|---:|
| `test01_frame_0000868.jpg` | 867 | 0,85 |
| `test01_frame_0000869.jpg` | 868 | 0,86 |
| `test01_frame_0000871.jpg` | 870 | 0,86 |

Różnica poniżej jednego stopnia jasności na skali do 255 to szum ponownego
zapisu do JPEG. Sąsiednie klatki dawały od 2,5 do 9, więc dopasowanie jest
jednoznaczne.

**Wniosek:** klatki źródłowe odtwarzasz lokalnie z nagrań, w minutach.
Wycinanie idzie 1 740 klatek na sekundę, a klatek źródłowych w treningu jest
17 832. Kopie augmentowane odtworzą się **nie**, ale do smoke testu i tak ich
nie chcesz — augmentacja tylko zaszumia obraz tego, co mierzysz.

## 2. Sprzęt i czego brakuje

| Rzecz | Stan |
|---|---|
| karta | RTX 3070, 8 GB, ale **wolne tylko 5,7 GB** — 2,3 GB zajmuje pulpit i przeglądarka |
| procesor | 8 rdzeni, 16 wątków |
| pamięć | 32 GB |
| torch, torchvision, transformers, pycocotools, faiss, sklearn, ultralytics, timm | **są** |
| `rfdetr` | **brak, trzeba doinstalować** |

Przed treningiem zamknij przeglądarkę. Przy 5,7 GB wolnego RF-DETR-Nano zmieści
się, ale bez zapasu.

## 3. Sedno: co smoke test może udowodnić, a czego nie

Tu jest pułapka, w którą łatwo wpaść. „Sprawdzę lokalnie, czy selekcja działa"
brzmi sensownie, ale przy małym modelu, małym zbiorze i pięciu epokach
**różnica między dobrym a złym wyborem zdjęć może być mniejsza niż różnica
między dwoma uruchomieniami tego samego wariantu**. Wtedy wynik nic nie znaczy,
a łatwo go odczytać jako „selekcja nie działa".

Dlatego smoke test lokalny to **kalibracja przyrządu, nie mały eksperyment**.
Ma odpowiedzieć na pytanie: czy ten przyrząd w ogóle mierzy.

### Trzy poziomy, każdy odpowiada na co innego

**Poziom 1 — hydraulika.** Czy kod przechodzi całą drogę: cechy, selekcja,
listy plików, trening, metryki. Minuty. Nie mówi nic o nauce, ale wyłapuje
większość błędów.

**Poziom 2 — czy pomiar rozróżnia zbiory.** To jest właściwy cel i to się
pomija najczęściej. Potrzebne są dwie kontrole:

- **Kontrola negatywna — podłoga szumu.** Ten sam wariant, dwa różne ziarna
  losowe. Różnica między wynikami to szum samego treningu. Każdy prawdziwy
  efekt musi być od niej większy, inaczej jest nieodróżnialny od przypadku.
- **Kontrola pozytywna — sufit sygnału.** Zbiór, który **musi** wypaść gorzej.
  Nie losowy, bo losowy bywa zaskakująco dobry. Coś jawnie złego: wszystkie
  klatki z jednego pacjenta albo pięć procent najbardziej rozmytych klatek.
  Jeśli przyrząd nie odróżni dobrego zbioru od takiego, nie odróżni niczego.

Odległość między tymi dwiema liczbami to **zakres pomiarowy** twojego
lokalnego stanowiska. Jeśli wyjdzie bliski zeru, lokalnie protokołu sprawdzić
się nie da i trzeba iść od razu na Eden. To też jest wynik, i to warty
poznania w jeden dzień zamiast w miesiąc.

**Poziom 3 — czy selekcja daje sygnał lokalnie.** Pełny pipeline kontra losowy
wybór. Ma sens **tylko wtedy**, gdy poziom 2 przeszedł.

Uczciwie: przy dwóch i pół tysiąca obrazów i pięciu epokach poziom 3 wyjdzie
prawdopodobnie **nierozstrzygający**. I to jest w porządku. Zadaniem lokalnego
testu są poziomy 1 i 2.

## 4. Jak to zbudować

### Krok 1 — odtworzyć małą pulę z nagrań

Z pliku `augmented_coco_20250417_030014.json` wybierz **wpisy bez przyrostka
`_aug`** i wytnij te klatki z nagrań, pamiętając o przesunięciu o jeden.

Do smoke testu nie potrzebujesz wszystkich 35 pacjentów treningowych.
Weź **dziesięciu**, plus dwóch na walidację i dwóch na test. Podział bierzesz
z istniejącego `video_split.json`, **nigdy nie losujesz własnego** — inaczej
testujesz co innego niż protokół.

Rachunek: 17 832 klatki źródłowe na 35 pacjentów to około 509 na pacjenta.
Dziesięciu pacjentów daje pulę rzędu **pięciu tysięcy klatek**, z której
wybierasz podzbiory po **pięćset**. Przy 704 pikselach to jakieś 150 MB.

### Krok 2 — cechy

DINO, Fourier i FastSAM na tych pięciu tysiącach klatek. To jest jednorazowe
i przy takiej wielkości zejdzie w kilkanaście minut, nie w noc.

EL2N wymaga modelu wstępnego. Na smoke teście **wystarczy model po dwóch
epokach** — chodzi o sprawdzenie, czy potok liczy i zapisuje, nie o jakość
miary trudności.

### Krok 3 — cztery warianty, nie piętnaście

| ID | Co to jest | Rola |
|---|---|---|
| `M-DESF` | pełny pipeline | to, co ma wygrać |
| `B-RANDAUG` | losowy wybór | zwykłe odniesienie |
| **`CTRL-NOISE`** | `M-DESF` z drugim ziarnem | **podłoga szumu** |
| **`CTRL-BAD`** | wszystko z jednego pacjenta albo najbardziej rozmyte klatki | **kontrola pozytywna** |

Cztery warianty razy dwa ziarna to osiem biegów. Przy modelu Nano i pięciuset
obrazach to jest robota na jedno popołudnie.

### Krok 4 — czytanie wyniku

Policz różnicę w mAP między parami:

```
szum      = |M-DESF(ziarno A) - M-DESF(ziarno B)|
sygnal    = |M-DESF - CTRL-BAD|
selekcja  = |M-DESF - B-RANDAUG|
```

Trzy możliwe odpowiedzi:

- **sygnał znacznie większy niż szum** → przyrząd mierzy, można patrzeć na
  trzecią liczbę,
- **sygnał porównywalny z szumem** → lokalnie nie sprawdzisz nic o selekcji;
  hydraulika przetestowana, jedziesz na Eden,
- **selekcja większa niż szum** → dobra wiadomość, ale **nie jest to wynik do
  artykułu**; zbiór za mały, epok za mało, jedno stanowisko.

## 5. Czego lokalnie nie sprawdzisz

Żeby nie tracić czasu na rzeczy, które i tak wymagają klastra:

- **rozdzielczości 704** — przy 5,7 GB wolnej pamięci karty nie zmieści się
  z sensownym batchem,
- **pełnej macierzy piętnastu kombinacji** — to jest sto siedemdziesiąt biegów,
- **liczby, które wejdą do tabel** — każdy wynik z pięciu epok jest wynikiem
  o niedouczonym modelu i nie wolno go pokazywać jako rezultat,
- **wpływu ziarna w sensie statystycznym** — trzy ziarna to minimum, a lokalnie
  starczy czasu na dwa.

## 6. Kolejność, gdyby zaczynać jutro

| # | Krok | Czas |
|---|---|---|
| 1 | doinstalować `rfdetr`, sprawdzić, że model Nano wstaje na karcie | pół godziny |
| 2 | odtworzyć pulę dziesięciu pacjentów z nagrań | minuty |
| 3 | cechy na tej puli | kilkanaście minut |
| 4 | model wstępny EL2N, dwie epoki | kilkadziesiąt minut |
| 5 | cztery warianty razy dwa ziarna | popołudnie |
| 6 | policzyć szum, sygnał i selekcję | minuty |

Punkt piąty ma sens dopiero po pierwszym. Punkt szósty rozstrzyga, czy lokalne
stanowisko jest do czegokolwiek poza hydrauliką.

## 7. Jedna rzecz, którą trzeba zapisać niezależnie od wyniku

**Zmierzony czas jednej epoki.** To jest jedyna liczba z lokalnego testu, która
przenosi się na Eden: pozwala przeliczyć `tools/estimate_time.py` z widełek na
konkret. Zapisz ją razem z rozmiarem batcha i rozdzielczością.
