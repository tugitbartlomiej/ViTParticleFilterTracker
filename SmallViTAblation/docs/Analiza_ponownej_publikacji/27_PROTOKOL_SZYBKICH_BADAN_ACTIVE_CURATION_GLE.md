# Protokół szybkich badań: aktywna kuracja i diagnostyka latentna

## Cel i miejsce względem poprawy artykułu

Ten tor **nie zastępuje** naprawy eksperymentów z odrzuconego artykułu. Najpierw
trzeba odtworzyć poprawne porównanie selektorów przy tym samym budżecie danych,
tej samej recepcie treningowej i rozłącznym podziale po operacji. Nowa warstwa
odpowiada na dalsze pytanie: czy stan detektora i dynamika jego reprezentacji
pozwalają wskazać kolejny podzbiór ramek, który poprawia model bardziej niż
losowanie, próbkowanie czasowe i samo pokrycie przestrzeni cech.

## Tezy, które można sprawdzić

1. **H1, kuracja statyczna.** Przy stałej liczbie unikalnych ramek selekcja
   pokryciowa osiąga wyższe COCO mAP niż losowanie i próbkowanie jednostajne.
2. **H2, redundancja filmu.** Zysk nie wynika wyłącznie z mniejszej liczby
   sąsiednich klatek. Selekcja pokryciowa musi pokonać silny baseline
   `uniform_time` z tym samym minimalnym odstępem czasowym.
3. **H3, diagnoza modelu.** Po dodaniu tego samego budżetu kolejnych ramek
   `coverage_residual` albo `coverage_history` daje większą sparowaną poprawę
   niż samo `coverage` oraz `coverage_uncertainty`.
4. **H4, transfer.** Ranking selektorów ustalony na YOLOv8n zachowuje kierunek
   efektu na RT-DETRv2-S. RF-DETR-Nano sprawdza osobno, czy efekt jest obecny
   dla detektora z backbone'em ViT/DINOv2.

H3 jest hipotezą eksploracyjną do czasu przejścia kontrolowanych benchmarków
syntetycznych. Nie należy nazywać wyniku dokładnym jądrem Mori–Zwanziga ani
fizycznym szumem Langevina. Kod estymuje skończone, cross-fitowane kontrasty
błędu predykcji i strukturę reszt w zamrożonym latencie.
Predyktory są cross-fitowane po grupach operacji. Aktualna baza modów reszt jest
jedynie opisowa i powstaje ze wszystkich cross-fitowanych reszt puli treningowej;
jej stabilność trzeba potwierdzić w dodatkowym zewnętrznym podziale grupowym.

## Dane i podział

- Jednostką podziału jest operacja albo źródłowy film. Wszystkie duplikaty i
  klipy pochodzące z tej samej operacji pozostają w jednym splicie.
- Metadane Cataract-LMM nie potwierdzają obecnie tożsamości pacjenta. W tekście
  pracy należy pisać „operation/video-disjoint split”, dopóki tego nie
  zweryfikujemy.
- Selekcja korzysta wyłącznie z puli treningowej. Walidacja służy do zamrożenia
  recepty, progu i checkpointu. Test jest otwierany po zamrożeniu wariantu.
- Zbiór zewnętrzny nie może służyć do wyboru selektora, modelu, epoki ani progu.
- Budżet oznacza liczbę **unikalnych ramek źródłowych**, a nie liczbę obrazów po
  augmentacji.
- W części trackingowej ramka `cystotome_bbox` opisuje instrument, a
  `cystotome_tip` jest punktem. Eksperyment bbox ma więc cel „instrument”, nie
  „instrument tip”. Puste komórki trackingu pozostają etykietą `unknown`, dopóki
  kontrola anotacji nie potwierdzi, że są prawdziwymi negatywami.
- Część instance segmentation z wieloma narzędziami jest osobnym eksperymentem.
  Nie wolno mieszać jej z jednoinstrumentowym trackingiem bez jawnego remapu klas.

Lokalny checkout Cataract-LMM zawiera adnotacje i indeksy, lecz nie zawiera
kompletu obrazów ani archiwów trackingu. Preflight i dry-run można wykonać
lokalnie, a rzeczywisty trening dopiero po pobraniu obrazów.

## Modele i kolejność

1. **YOLOv8n** jest sondą rozwojową. Jest mały, szybko się uczy i ma prosty,
   stabilny interfejs treningowy. To na nim wykonujemy szeroki screening.
2. **RT-DETRv2-S/R18** jest obowiązkowym transferem do głównej historii pracy.
   Oficjalna implementacja podaje około 20 milionów parametrów i konfigurację
   wejścia 640 pikseli: <https://github.com/lyuwenyu/RT-DETR>.
3. **RF-DETR-Nano** jest kontrolą z backbone'em DINOv2 ViT. Nie pełni roli
   najtańszej sondy tylko dlatego, że nazywa się Nano. Dokumentacja modelu:
   <https://rfdetr.roboflow.com/reference/nano/>.
4. D-FINE-S jest opcjonalnym transferem po zakończeniu głównej macierzy.

Profile wykonawcze blokują wejście na 384 piksele dla szybkiego YOLOv8n,
640 pikseli dla oficjalnego RT-DETRv2-S oraz 384 piksele dla RF-DETR-Nano.
W confirm YOLOv8n przechodzi na 512 pikseli. Recepta i rozdzielczość pozostają
identyczne między selektorami tej samej architektury. Różne rozdzielczości
między architekturami są jawne, więc modele służą do sprawdzenia transferu
kierunku efektu, a nie do rankingu architektur.

Profile zapisują również rzeczywistą precyzję: FP16 dla Ultralytics i
oficjalnego RT-DETRv2 oraz BF16 dla RF-DETR. Nie opisujemy wspólnego pola
`precision` jako identycznego między bibliotekami, ponieważ ich mechanizmy AMP
są różne; stałe pozostają ustawienia między selektorami tego samego modelu.

Każdy adapter wyłącza augmentacje online. YOLO dostaje zerowe parametry
transformacji, RT-DETRv2 zastępuje odziedziczoną listę transformacji stałym
Resize, a RF-DETR dostaje `aug_config={}` i `scale_jitter=False`. Dzięki temu
warianty offline nie są mieszane z losową augmentacją biblioteki.

YOLOv8n pozostaje jawnie przypięty jako `yolov8n.pt`, zgodnie z dokumentacją:
<https://docs.ultralytics.com/models/yolov8/>. Zmiana wersji modelu tworzy nowy
eksperyment i nie może nastąpić między wariantami jednego porównania.

## Trzy poziomy kosztu

### Smoke

Najpierw overfit 64 poprawnie zmaterializowanych obrazów. Następnie 10 grup
treningowych, 2 walidacyjne i 2 testowe, budżet 500, dwa seedy. Wystarczą trzy
warunki: `random`, `uniform_time` i `coverage`. Ten etap
sprawdza hydraulikę, mapowanie klas, geometrię bboxów, czas i wariancję. Nie jest
dowodem publikacyjnym.

### Pilot

Na pełnym splicie treningowym wykonujemy krzywe uczenia dla budżetów 500, 1500
i 2500. Screening jednego seeda obejmuje sześć strategii. Po sprawdzeniu logów
i kosztu zamrażamy receptę, po czym powtarzamy wybrane warunki dla trzech
sparowanych seedów. Każdy selector/model w parze dostaje ten sam split, ten sam
startowy zbiór L0, tę samą liczbę aktualizacji optymalizatora i tę samą
inicjalizację COCO.

### Confirm

Do RT-DETRv2-S przenosimy dokładne listy `frame_id`, bez ponownego strojenia
selekcji pod ten model. Minimalny zestaw to `random`, `uniform_time`, `coverage`
i zamrożony zwycięzca diagnostyczny. Trzy seedy są wyłącznie szybkim
sprawdzeniem wariancji. Główne porównanie publikacyjne wykonujemy na sześciu
sparowanych seedach; regułę rozszerzenia do dziesięciu zapisujemy przed testem,
jeżeli przedział ufności pozostaje zbyt szeroki. RF-DETR-Nano powtarza ten sam
test jako kontrola ViT. Drugorzędne kontrole mogą pozostać przy trzech seedach.

## Pętla aktywnej kuracji

Wszystkie strategie rozpoczynają od tego samego L0. Domyślnie L0 ma 500 ramek
wybranych przez `uniform_time`; kolejne porcje mają po 500 ramek. Po każdej
rundzie model jest trenowany od tych samych wag COCO. Warm-start można zbadać
osobno, ale nie wolno mieszać go z porównaniem selektorów.

```text
zamrożony split i pula train
  -> wspólny L0
  -> trening sondy
  -> predykcje i latent dla całej nieoznaczonej puli train
  -> uncertainty + diagnostyka reszt/pamięci
  -> wybór nowej porcji przy ograniczeniu pokrycia i odstępu czasowego
  -> nowy trening od wspólnej inicjalizacji
  -> val; test dopiero po zamrożeniu wariantu
```

Stan aktywnej pętli musi być rzeczywiście aktualizowany po każdej rundzie.
Latent zamrożonego enkodera może pozostać stały jako składnik pokrycia, ale
uncertainty i sygnał zadaniowy muszą pochodzić z aktualnego checkpointu. Inaczej
jest to jednorazowy ranking, a nie aktywne uczenie.

Warianty mają rosnącą złożoność:

- proste: `random`, `uniform_time`;
- średnie: `coverage`, `coverage_uncertainty`;
- zaawansowane: `coverage_residual`, `coverage_history`.

Jeżeli potrzebnego reżimu nie ma w puli, algorytm może opisać brakujący kierunek
reszt, ale nie może wskazać nieistniejącej klatki. Taki wynik staje się listą
cech do pozyskania w nowym nagraniu, a nie automatyczną anotacją.

## Metryki i analiza statystyczna

Główna metryka to COCO `mAP_0.5:0.95` na zamrożonym teście. Raportujemy także
`mAP_0.5`, `mAP_0.75`, `AR@100`, precision, recall i F1 przy progu wybranym na
walidacji. Dla każdego budżetu raportujemy liczbę unikalnych ramek, liczbę
operacji, liczbę instancji, czas selekcji, czas treningu, GPU-hours i liczbę
aktualizacji optymalizatora.

Podstawowym wynikiem jest sparowana różnica względem baseline'u dla każdego
seeda, średnia, odchylenie standardowe i wszystkie wartości surowe. Przedziały
ufności dla jakości detekcji należy liczyć przez bootstrap po operacjach, nie po
klatkach. Krzywą budżetu podsumowujemy również polem pod krzywą mAP względem
liczby ramek. Nie wybieramy modelu na podstawie największego zysku selektora.

Dla gęstego trackingu raportujemy dodatkowo fałszywe alarmy na minutę, długość
luk detekcji i jitter położenia ramki. Są to metryki sekwencyjne; nie zastępują
COCO mAP. Każdy przedział ufności powstaje przez resampling całych operacji.

Dotychczas używany zbiór zewnętrzny był wcześniej oglądany podczas rozwoju
artykułu, dlatego nie jest już dziewiczym testem potwierdzającym. Może pozostać
benchmarkiem historycznym. Ostateczne twierdzenie wymaga nowego nietkniętego
zbioru zewnętrznego albo zagnieżdżonej walidacji grupowej.

## Warunki awansu i negatywne kontrole

Wariant przechodzi z pilota do confirm, jeżeli przed obejrzeniem testu spełnia
ustalone kryterium walidacyjne: dodatnia średnia sparowana względem
`uniform_time`, brak katastrofalnego spadku na którymkolwiek seedzie i koszt
selekcji mieszczący się w zadanym budżecie. Dokładna granica praktyczna musi być
zapisana w `promotion.json` przed testem.

Kontrole obowiązkowe:

- permutacja score'u diagnostycznego między klatkami;
- odwrócony ranking score'u;
- ten sam ranking bez ograniczenia czasowego;
- diagnostyka z historią o długości zero;
- score liczony dla przewidywanej klatki, nigdy dla klatki wejściowej;
- kontrolowany benchmark Markowowski, ukryta pamięć, nieliniowość, wymuszenie
  egzogeniczne i czysty szum.

Pierwsza wersja używa cech całej klatki. Wersja obiektowa jest etapem drugim:
ROI instrumentu lub propozycja detektora, geometria bbox i cecha kontekstu
powinny tworzyć osobne kanały stanu. Nie wolno używać średniej z całego przyszłego
tracku jako „adresu” bieżącej klatki; dozwolona jest wyłącznie historia dostępna
do chwili `t`. Interpretacja stabilnego kanału jako tożsamości pozostaje hipotezą
do sprawdzenia, a nie założeniem metody.

Publikacyjna wartość powstaje dopiero wtedy, gdy diagnostyka wybiera właściwy
rodzaj interwencji i daje powtarzalny przyrost jakości przy stałym budżecie.
Samo połączenie DINO, Fouriera, SAM, Koopmana i równania Langevina nie stanowi
jeszcze dowodu nowości ani poprawności.
