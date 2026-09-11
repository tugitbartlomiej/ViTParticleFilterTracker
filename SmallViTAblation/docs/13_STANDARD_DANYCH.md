Standard przygotowania danych — jeden format, ale nie udawanie, że źródła są równe
=================================================================================

Data: 2026-09-09. Odpowiedź na pytanie: czy wszystkie dane mają być w tym samym
formacie i tej samej rozdzielczości.

**Krótko: wejście do modelu — tak, identyczne. Traktowanie źródeł jako
równoważnych — nie.** Obraz natywnie w 1920×1080 zmniejszony do 512 i obraz
natywnie 512×324 powiększony do 512 mają tyle samo pikseli i zupełnie inną
zawartość. Jeśli tego nie zapiszemy, nie odróżnimy spadku wyniku spowodowanego
doborem danych od spadku spowodowanego rozdzielczością źródła.

---

## 1. Rozdzielczość jest tu najważniejszym parametrem, nie drugorzędnym

Policzone na naszej puli treningowej (71 316 obrazów, wszystkie 1920×1080,
71 931 ramek narzędzia). Mediana ramki to **94×128 px**, czyli **0,56%
powierzchni obrazu**.

Co się z nią dzieje przy różnych rozmiarach wejścia, przy zachowaniu proporcji:

| Wejście | Skala | Mediana ramki | Pole | Ramek poniżej 32×32 |
|---|---:|---|---:|---:|
| 384 | 0,200 | 19×26 | 464 px² | **75,8%** |
| 512 | 0,267 | 25×34 | 826 px² | **56,9%** |
| 640 | 0,333 | 31×43 | 1 290 px² | 38,0% |
| 704 | 0,367 | 34×47 | 1 561 px² | 24,6% |
| 960 | 0,500 | 47×64 | 2 903 px² | 5,6% |

Ta tabela opisuje pole bbox po hipotetycznym resize z zachowaniem proporcji.
Nie jest definicją grup COCO AP-small: COCOeval używa `area` w ocenianych
anotacjach, zwykle w układzie oryginalnego obrazu, z granicą pola 32² px².
Małość nie wymaga obu boków poniżej 32 px. Po przeskalowaniu predykcji z
powrotem do układu GT zmiana wejścia sieci nie zmienia kategorii area GT.

To zmienia wybór modelu. Warianty RF-DETR różnią się głównie rozdzielczością,
a prawie nie liczbą parametrów: Nano 30,5 M przy 384, Small 32,1 M przy 512,
Medium 33,7 M przy 576, Large 33,9 M przy 704 (historyczne zestawienie wersji).
Podobna liczba parametrów nie dowodzi identycznej architektury; przed biegiem
trzeba przypiąć wersję i sprawdzić konfigurację wariantu. Stosunek pól 704²/512²
wynosi około 1,89, ale nie jest pomiarem stosunku czasu ani zużycia VRAM.

**Wniosek dla planu:** rozdzielczości nie wybieramy z góry. Bieg kalibracyjny
(`03_COMPUTE`, punkt 2) porównuje 512 i 704 na jednym wariancie i dopiero wynik
decyduje. Jeśli wygra 704, całość drożeje mniej więcej dwukrotnie — z 250–1000
godzin GPU robi się 500–2000, czyli od trzech do dziesięciu dób na ośmiu GPU.

## 2. Rozdzielczości źródeł, które chcemy połączyć

| Źródło | Natywnie | Stosunek do naszych 1920×1080 |
|---|---|---|
| nasze nagrania (CATARACTS) | 1920×1080 | 1,00 |
| SICS-105 | 1920×1080 | 1,00 |
| Cataract-LMM, szpital Noor | 1920×1080 | 1,00 |
| CaDIS | 960×540 | 0,50, ta sama proporcja |
| Cataract-LMM, szpital Farabi | 720×480 | 0,44 w poziomie, **inna proporcja** |
| Cataract-1K, nagrania nieanotowane | 512×324 | 0,27, **inna proporcja** |

Dwa dolne wiersze są problemem. Przy wejściu 512 obraz z Cataract-1K trzeba
**powiększyć**, a powiększenie nie odtwarza szczegółu — to te same piksele
rozciągnięte. Ramka, która u nas ma 94×128 px, tam ma od początku 25×38 px.

Źródła o niższej rozdzielczości mogą być osobnym zbiorem oceny, jeśli należą do
docelowej domeny i mają właściwe anotacje. Nie usuwamy ich tylko dlatego, że
test może być trudniejszy. Raportujemy natywną rozdzielczość i wyniki per źródło;
upscaling nie odtwarza utraconego szczegółu.

## 3. Proporcje: dopełnienie, nie ściskanie

1920×1080 to proporcja 16:9. 512×324 to około 1,58:1. 720×480 to 1,5:1.
Wciśnięcie wszystkiego w kwadrat 512×512 **zniekształca każde źródło inaczej** —
okrągła źrenica staje się inną elipsą w zależności od pochodzenia.

Letterbox jest wariantem preprocessingu do sprawdzenia. Dla każdego detektora
najpierw odtwarzamy jego oficjalne transformacje; zmiana na letterbox jest
osobną decyzją wymagającą walidacji. Zapisujemy skalę, padding i crop, a przed
ewaluacją odwracamy transformację bbox. Unikamy podwójnego resize.

## 4. Rzecz najważniejsza: pole widzenia mikroskopu

Obraz z mikroskopu operacyjnego to jasne koło na czarnym tle. **Każdy zbiór
kadruje to koło inaczej** — inna średnica, inne położenie, inna szerokość
czarnej obwódki.

Obwódka może być sygnałem domeny, ale nie dowodzi, że model wykorzystuje ją
zamiast narzędzia. Crop pola widzenia jest hipotezą do ablacji, nie obowiązkową
naprawą. Może usunąć fragment obiektu; trzeba zweryfikować zachowanie bboxów,
anotacji i przypadków przy brzegu. Regułę crop ustala się na train/val i
stosuje jednakowo, bez strojenia na wyniku zewnętrznego testu.

## 5. Co ujednolicamy, a czego nie ruszamy

**Ujednolicamy kontrakt**: układ katalogów, schemat anotacji i jawny opis
preprocessingu. Rozmiar wejścia wynika z protokołu porównania. Kadrowanie pola
widzenia, zmiana proporcji i ponowna kompresja wymagają uzasadnienia oraz
walidacji; zachowujemy obrazy źródłowe i zapis transformacji geometrii.

**Nie ruszamy** (bo to jest właśnie zmienność, na którą model ma być odporny):
balans bieli, jasność, kontrast, model kamery i mikroskopu, styl operatora,
sposób oświetlenia. Korygowanie tego zabiłoby sens badania nad przenoszeniem
się na obce dane.

**Zapisujemy, nigdy nie zgadujemy:** natywną rozdzielczość, natywną liczbę
klatek na sekundę, źródło, pacjenta, nagranie, numer klatki.

## 6. Liczba klatek na sekundę

FPS wpływa na redundancję i próbkowanie także dla detektora pojedynczych klatek.
Jest również ważny **dla propagacji ramek** z `10_TSFA_NOWA_PULA.md`: ±20 klatek przy 25 klatkach na
sekundę to 0,8 sekundy, a przy 60 klatkach na sekundę tylko 0,33 sekundy.

**Zasięg propagacji definiujemy w sekundach, nie w klatkach.** Inaczej ten sam
parametr znaczy co innego w każdym zbiorze.

## 7. Format zapisu

- Zachowujemy pliki źródłowe. Cache może stosować JPEG o jawnej jakości
  (np. 95), ale ponowny zapis nie wyrównuje historycznych artefaktów kompresji.
- Jeden układ katalogów: `<zbior>/<pacjent>/<nagranie>/<numer_klatki>.jpg`.
- Anotacje w formacie COCO, z **dodatkowymi polami** na każdym obrazie:
  `source_dataset`, `patient_id`, `video_id`, `frame_index`, `native_width`,
  `native_height`, `native_fps`, `fov_crop` (prostokąt pola widzenia),
  a na każdej ramce: `label_origin` (`human` albo `propagated`) oraz
  `seed_distance_s` (odległość od klatki ręcznie oznaczonej, w sekundach).

Bez tych pól nie da się potem policzyć ani wpływu rozdzielczości źródła, ani
wpływu udziału etykiet automatycznych. A jedno i drugie to osobne, ładne
ablacje do artykułu.

## 8. Kolejność kroków

1. zamrozić grupy train/val/test i audyt anotacji; określić preprocessing modelu,
2. opcjonalny crop/resize wdrożyć z zapisem transformacji, zachowaniem źródeł
   i sprawdzeniem boxów; stały crop per film wymaga stabilnego pola widzenia,
3. zapisać pełne metadane każdej klatki,
4. dopiero teraz liczyć cechy i uruchamiać selekcję.

Punkt czwarty jest twardy: **pula musi być gotowa przed liczeniem cech**, bo
cała macierz z `02_MATRIX.md` liczy się na tej puli. Dokładanie danych później
oznacza liczenie wszystkiego od nowa.

## 9. Czego nie robić

- **Nie mieszać źródeł w zbiorach oceny.** Każdy zbiór zewnętrzny raportujemy
  osobno. Uśrednienie po zbiorach o różnej rozdzielczości daje liczbę, która
  może być myląca bez wag i wyników składowych. Agregacja jest możliwa, jeśli
  cel i wagi domen ustalono wcześniej; raportowanie per źródło pozostaje konieczne.
- **Nie skalować w górę do rozmiaru wejścia i nie mówić o tym.** Jeśli źródło
  jest niższej rozdzielczości, ma to być widoczne w tabeli.
- **Nie ustalać rozdzielczości przed kalibracją.** Tabela z punktu 1 pokazuje,
  że to jest wybór między „ponad połowa obiektów jest mała" a „czwarta część
  jest mała", i kosztuje dwa razy tyle czasu.
