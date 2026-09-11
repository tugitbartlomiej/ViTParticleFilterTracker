Smoke test lokalny — czy protokół selekcji w ogóle da się sprawdzić na RTX 3070
==============================================================================

Data: 2026-09-10. Analiza, nie instrukcja. Odpowiada na pytanie: **co da się
udowodnić lokalnie, a czego nie da się w żadnym razie.**

---

## 1. Dane są lokalnie — sprawdzone

**Obrazy puli leżą na dysku F**, 90 141 plików JPEG, 37,7 GB:

Stan tego inwentarza pochodzi z 2026-09-10. Aktualne testy CPU i ich zakres:
[21_STAN_REFAKTORU.md](21_STAN_REFAKTORU.md). Dostępna pamięć i czasy poniżej
nie są bieżącym pomiarem ani gwarancją wykonalności konkretnej konfiguracji.

```
Eden/Datasets/datasets_20250606.tar/datasets_20250606/datasets/
    DETR_augmented_dataset_20250218/images/
```

Nazwa `datasets_20250606.tar` myli, bo to jest **katalog**, a nie plik — w środku
leży zarówno samo archiwum (38,3 GB), jak i jego rozpakowana zawartość.

### Ile z tego pokrywa anotacje

| Część | W anotacjach | Brakuje na dysku |
|---|---:|---:|
| trening | 71 316 | 967 (1,4%) |
| walidacja | 10 780 | 128 (1,2%) |
| test | 9 240 | 100 (1,1%) |
| **razem** | **91 336** | **1 195 (1,3%)** |

Braki rozkładają się równo po wszystkich trzech częściach, więc to nie jest
systematyczna dziura, tylko pliki, które kiedyś nie doszły albo zostały usunięte.
Do smoke testu bez znaczenia; do produkcji trzeba je uzupełnić albo wykreślić
z anotacji, świadomie i jednym skryptem.

### Trzecia liczba w sprawie 91 336

Warto to zapisać, bo dotyczy zarzutu recenzenta 2 o pochodzenie liczby obrazów.
Krążą **trzy** wartości:

```
91 348  = 22 837 klatek zrodlowych x 4      (arytmetyka)
91 336  = liczba w anotacjach i w artykule  (minus 12 klatek z trzema wariantami)
90 141  = liczba obrazow FAKTYCZNIE na dysku
```

Ostatnia liczba zgadza się co do sztuki z rozmiarem cache'u cech opisanym
w starszych notatkach (`dino_features.pkl`, 90 141 wierszy). Czyli pula, na
której pipeline naprawdę pracował, miała **90 141 obrazów**, nie 91 336.
Do artykułu idzie liczba rzeczywista, nie deklarowana.

### Odtwarzanie klatek z nagrań — przydaje się gdzie indziej

Przy okazji sprawdziłem, jak nazwy plików odwzorowują się na klatki wideo,
bo to jest potrzebne do budowy nowej puli z nagrań (`10_TSFA_NOWA_PULA.md`).

Nazwa `test01_frame_0000868.jpg` odpowiada klatce wideo o indeksie **867** —
numeracja w nazwie jest o jeden wyższa. Sprawdzone na wszystkich trzech
nieaugmentowanych obrazach z podręcznego katalogu: średnia różnica jasności
0,85–0,86 na skali do 255, czyli szum ponownego zapisu do JPEG, podczas gdy
sąsiednie klatki dawały od 2,5 do 9.

Do smoke testu nie jest to potrzebne — obrazy są gotowe.

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

- **Powtórzenie treningu.** Dwa seedy dają wstępną obserwację zmienności,
  ale pojedyncza różnica nie wyznacza progu istotności ani granicy wykrywalnego efektu.
- **Kontrola z ograniczoną różnorodnością lub jakością.** Jeden pacjent albo
  rozmyte klatki mogą ujawnić problem, lecz nie muszą wypaść gorzej w każdym
  zadaniu. Kontrolę poprawności evaluatora dają syntetyczne GT/predykcje ze
  znanym wynikiem, a treningu — np. overfit małego, poprawnie opisanego zbioru.

Te kontrole są diagnostyczne. Podobne wyniki mogą oznaczać słaby kontrast
zbiorów, niedouczenie, dużą wariancję lub brak efektu. Przeniesienie kodu na
Eden samo nie rozstrzyga między tymi przyczynami.

**Poziom 3 — czy selekcja daje sygnał lokalnie.** Pełny pipeline kontra losowy
wybór. Ma sens **tylko wtedy**, gdy poziom 2 przeszedł.

Uczciwie: przy dwóch i pół tysiąca obrazów i pięciu epokach poziom 3 wyjdzie
prawdopodobnie **nierozstrzygający**. I to jest w porządku. Zadaniem lokalnego
testu są poziomy 1 i 2.

## 4. Jak to zbudować

### Krok 1 — wyciąć małą pulę z tego, co jest

Nie trzeba niczego odtwarzać. Bierzesz podzbiór plików z katalogu obrazów
i podzbiór wpisów z `annotations_train.json`.

Do smoke testu nie potrzebujesz wszystkich 35 pacjentów treningowych.
Weź **dziesięciu**, plus dwóch na walidację i dwóch na test. Podział bierzesz
z istniejącego `video_split.json`, **nigdy nie losujesz własnego** — inaczej
testujesz co innego niż protokół.

Rozważ pominięcie kopii augmentowanych. Klatek źródłowych w treningu jest
17 832, czyli około 509 na pacjenta; dziesięciu pacjentów daje pulę rzędu
**pięciu tysięcy klatek**, z której wybierasz podzbiory po **pięćset**.
Bez augmentacji obraz tego, co mierzysz, jest czystszy.

### Krok 2 — cechy

DINO, Fourier i FastSAM na tych pięciu tysiącach klatek. To jest jednorazowe
i przy takiej wielkości zejdzie w kilkanaście minut, nie w noc.

EL2N wymaga modelu wstępnego. Na smoke teście **wystarczy model po dwóch
epokach** — chodzi o sprawdzenie, czy potok liczy i zapisuje, nie o jakość
miary trudności.

### Krok 3 — cztery warianty, nie piętnaście

| ID | Co to jest | Rola |
|---|---|---|
| `M-DESF` | pełny pipeline | metoda, której efekt mierzymy bez założenia zwycięstwa |
| `B-RANDSRC` | losowy wybór źródeł | główne odniesienie przy tym samym K |
| **`CTRL-NOISE`** | `M-DESF` z drugim ziarnem | **podłoga szumu** |
| **`CTRL-BAD`** | wszystko z jednego pacjenta albo najbardziej rozmyte klatki | **kontrola pozytywna** |

Cztery warianty razy dwa ziarna to osiem biegów. Przy modelu Nano i pięciuset
obrazach to jest robota na jedno popołudnie.

### Krok 4 — czytanie wyniku

Policz różnicę w mAP między parami:

```
szum      = |M-DESF(ziarno A) - M-DESF(ziarno B)|
sygnal    = |M-DESF - CTRL-BAD|
selekcja  = M-DESF - B-RANDSRC  # zachować znak różnicy
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

**Zmierzony czas jednej epoki** wraz z modelem, sprzętem, batch size,
precyzją i rozdzielczością. Czas na RTX 3070 nie przenosi się bezpośrednio na
H100: prognozę Edenu kalibruje się na docelowym węźle. Smoke test pięciu epok
można raportować jako taki, ale nie jako wynik w pełni wytrenowanego modelu.
