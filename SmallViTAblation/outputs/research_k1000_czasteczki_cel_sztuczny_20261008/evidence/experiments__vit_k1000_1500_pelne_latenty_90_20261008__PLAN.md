# Plan: pełne sztuczne latenty K1000 dla dokładnej bazy 1500

## Pytanie badawcze

Czy dla każdego z dokładnych 1500 obrazów użytych jako baza treningowa
zweryfikowanego checkpointa K1000 można znaleźć osobny pełny latent po GELU
ostatniego MLP, o kształcie `600×768`, który przy zamrożonych wagach i buforach
osiąga jednocześnie top-1 confidence co najmniej 0,90 i IoU do etykiety co
najmniej 0,90?

To jest jawnie zamówiony test oracle na tym samym zbiorze. Używa etykiety
każdego badanego obrazu podczas optymalizacji. Nie jest oceną poza próbą.

## Zamrożone wejścia

- Checkpoint: `assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt`.
- Oczekiwany SHA-256 checkpointa:
  `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Adnotacje dokładnej bazy K1000:
  `experiments/maly_vit_od_zera/eden/wejscia/pretrain/train/_annotations.coco.json`.
- Oczekiwany SHA-256 COCO:
  `3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75`.
- Manifest zachowuje kolejność `images` z COCO. Musi zawierać dokładnie 1500
  obrazów i 1500 adnotacji, po jednej na obraz, oraz SHA-256 bajtów każdego
  istniejącego obrazu źródłowego.
- Obrazy są czytane bezpośrednio z istniejących źródeł. Na Edenie czytnik
  otwiera człony istniejącego `tracking_static36.tar` bez ich wypakowywania.

## Aparat

1. Model pracuje w `eval`, FP32, z wyłączonym TF32. Wszystkie parametry mają
   `requires_grad=False`. Hash wag i buforów jest sprawdzany przed i po pracy.
2. Punkt interwencji to wyjście GELU w MLP ostatniego bloku, identyczne jak w
   eksperymencie bazowym `vit_latent128_jedno_zdjecie_90`.
3. Fast tail przelicza dokładnie `fc2`, LayerScale, residual, końcowy LayerNorm
   i głowę. Przed smoke musi przejść test zgodności z rzeczywistym pełnym
   forwardem z hookiem.
4. Optymalizowane są wszystkie `600×768` współrzędne jednego obrazu. Każdy
   obraz ma własny optimizer i własny warunek stopu. Batchowanie obrazów nie
   może mieszać stanów ani korekt.
5. Dolna granica post-GELU wynosi `-0.16997121`, ponieważ to ten sam punkt
   reprezentacji co w eksperymencie bazowym. Po każdym kroku latent jest do
   niej rzutowany.
6. Cel przed pierwszym sukcesem łączy: logit komórki docelowej, tłumienie
   pozostałych komórek, ranking top-1, dopasowanie `ltrb`, GIoU oraz jawną małą
   karę `1e-6 * mean((after-before)^2)`.
7. Sukces wymaga jednocześnie `confidence >= 0.90` i `IoU >= 0.90` w tej samej
   iteracji. Zapisywany jest dokładny pierwszy krok sukcesu.
8. Wszystkie obrazy wykonują wspólny rzeczywisty horyzont 500 kroków. Nie ma
   dopełniania powtórzonym stanem. Po pierwszym sukcesie optimizer nadal
   wykonuje prawdziwe kroki, minimalizując normę korekty pod różniczkowalnymi
   barierami confidence, rankingu top-1 i IoU. Jako `after` wybierany jest
   rzeczywiście odwiedzony stan o najmniejszej normie, który spełnia oba progi.
   Dla braku sukcesu `after` jest najlepszym rzeczywiście odwiedzonym stanem,
   a status brzmi `budget_exhausted`.
9. Wspólny schedule Adam, bez resetu stanu: kroki 1-100 mają LR 0,05, 101-250
   LR 0,15, 251-400 LR 0,05, a 401-500 LR 0,01. Numer `k` zawsze oznacza liczbę
   faktycznie wykonanych aktualizacji danego latentu.

## Zapis i wznowienie

Każdy obraz tworzy atomowo osobny plik latentów i osobny JSON metryk. Plik
latentów zawiera:

- `before`, `after`, `delta` w FP32, każde `600×768`;
- pełne rzeczywiste stany FP32 dla wspólnych kroków 25, 100 i 500; krok 0 jest
  zapisany bez duplikacji jako `before`;
- numery kroków trajektorii;
- identyfikator obrazu i docelowego tokenu.

Strata, confidence, IoU, normy korekty, faza celu, numer top-1 oraz normy
gradientu są zapisywane jako mała telemetria dla każdego kroku 0-500. Dla
deterministycznego panelu 256 cząsteczek zapisujemy w FP32 także gęste
`z_t`, surowy gradient przed `Adam.step` oraz pierwszą i drugą chwilę Adama.
Tylko ten panel może zasilać DMDc; sterowanie w parze `(z_t, u_t, z_{t+1})`
jest wcześniejszym gradientem i stanem optimizera, nigdy ruchem
`z_{t+1}-z_t` przewidującym sam siebie.

Po zapisie `after` jest ponownie wczytywany. Wymagane są: bitowa zgodność FP32,
zgodność fast tail z pełnym forwardem oraz niezmieniony hash wag i buforów.
Istniejący kompletny artefakt jest pomijany przy wznowieniu dopiero po
walidacji zawartości, metadanych, SHA checkpointa i SHA manifestu.

## Smoke i produkcja na Edenie

- Smoke ma osobny katalog wyników i obejmuje reprezentatywne przypadki o
  różnej jakości baseline oraz skrajnych rozmiarach ramek.
- Produkcja jest tablicą niezależnych porcji obrazów, po jednej GPU na zadanie,
  bez DDP. Porcje są rozłączne i razem pokrywają indeksy 0-1499.
- Pełny bieg jest zgłaszany jeden raz po poprawnym smoke, kontroli źródeł,
  wolnego miejsca i braku wcześniejszego biegu produkcyjnego.
- Osobny krok scalający tworzy zbiorcze metryki, listę niespełnionych progów,
  sumy kontrolne i raport końcowy.

## Budżet miejsca

Jeden pełny snapshot FP32 dla `1500×600×768` zajmuje 2 764 800 000 bajtów.
Sto takich stanów to 276,48 GB, a tysiąc 2,7648 TB, więc zapis per-step dla
całej bazy jest zabroniony. Dokładne `before/after/delta` zajmują surowo
8 294 400 000 bajtów. Trzy pełne milestone'y FP32 zajmują kolejne
8 294 400 000 bajtów. Główne tensory wymagają zatem 16 588 800 000 bajtów.
Budżet nie zakłada żadnego zysku z kompresji. Panel sterowania, metryki,
checkpoint i provenance muszą utrzymać całość poniżej 20,54 GB wolnych do
miękkiego limitu projektu Eden, potwierdzonych przed wysłaniem.

## Analiza ruchu i reżimów

Analiza po optymalizacji jest osobnym zależnym zadaniem CPU. Dla wszystkich
900 000 cząsteczek opisuje ruch we wspólnej bazie 768D na podstawie `z0`,
rzeczywistych różnic milestone'ów, kierunku i magnitudy. Reżimy są odkrywane
na cechach zawierających zarówno `z0`, jak i oracle delta. Model przypisania
reżimu dostaje wyłącznie `z0` oraz dostępny przed korektą kontekst tokenu i
obrazu. Przyszła delta nie może wejść do przypisania.

DMD jest analizą opisową gęstego panelu. DMDc dopasowuje zredukowane
`z_{t+1}=A z_t+B u_t`, gdzie `u_t` zawiera surowy gradient i chwile Adama
dostępne przed aktualizacją. Jest to zadanie CPU; skrypt nie przedstawia go
jako akcelerowanego przez GPU. Granice niezależnych trajektorii nigdy nie są
łączone w fałszywe przejścia.

## Przeniesienie korekty i ocena

- Metoda A: średnia oracle delta po obrazach osobno dla każdego indeksu tokenu,
  tensor `600×768`.
- Metoda B: średnia korekta zależna od wykrytego reżimu i indeksu tokenu, z
  kurczeniem do średniej reżimu/metody A przy małej liczności. Reżim dla nowej
  cząsteczki jest przypisywany wyłącznie z `z0` i prawidłowego kontekstu.
- Kontrola: naturalny latent bez zmiany.
- Górne odniesienie: niezależne per-image oracle `after`.

Obie metody są oceniane przez rzeczywisty hook-forward na naturalnych
latentach: confidence, IoU, odsetek obu progów oraz mAP@0.5 na 1500 obrazach.
Raport rozdziela dopasowanie in-sample od GroupKFold po identyfikatorach
filmów, gdzie reżimy i korekty powstają wyłącznie na filmach treningowych.
Analiza reżimów, DMDc i ewaluacja przeniesienia są osobnymi jobami zależnymi
od kompletnego, poprawnie scalonego biegu oracle.

## Podłoga i kryterium obalenia

Baseline to niezmieniony latent tego samego obrazu i ten sam zamrożony K1000.
Test obala powszechną osiągalność, jeśli choć jeden obraz nie spełni obu progów
w ustalonym budżecie albo jeśli geometria głowy ogranicza maksymalny IoU poniżej
0,90. Przypadki niespełnienia pozostają w mianowniku.

## Czego wynik nie dowodzi

Sukces dowodzi wyłącznie istnienia rozwiązania oracle w przestrzeni
post-GELU dla badanego obrazu i tej głowy. Nie dowodzi generalizacji,
naturalności latentu, osiągalności przez wcześniejsze warstwy, jakości
generatora SD ani poprawy detektora po treningu.
