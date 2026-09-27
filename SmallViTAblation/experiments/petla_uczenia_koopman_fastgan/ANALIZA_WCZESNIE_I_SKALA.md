# Czy pętla zadziała lepiej od początku uczenia i na większym modelu? Analiza (2026-09-27)

Pytanie użytkownika po pilocie: czy trenowałem od zera; może przewidywać kierunek uczenia po 1–2 epokach; może
większy model (RF-DETR Medium), bo metoda powinna się bardziej sprawdzić w dużych modelach w pierwszych ~5 epokach.
Źródła: Undermind (workspace „ViT: detekcja końcówki narzędzia”, przeszukanie „when does data selection accelerate
training early phase and model size” oraz wyszukiwania celowane, pełne teksty 9 prac), konfiguracja rfdetr 1.10.1,
nasze biegi 1810662 i pilot pętli (`WYNIKI.md`).

## 1. Co zrobił pilot (fakty)

- Nie od zera: od wag COCO bieg 1810662 uczył się 245 aktualizacji (2,45 epoki po 100 aktualizacji), potem ruszyła
  pętla. Czyli prognoza startowała mniej więcej tam, gdzie proponuje użytkownik („po dwóch epokach”).
- Krzywa 1810662: mAP@0,5 po epoce 0: 0,566; 1: 0,791; 2: 0,860; 3: 0,877; 4: 0,886. Szybka faza uczenia to
  pierwsze 2–3 epoki. Dryf latentu tożsamości gaśnie wykładniczo i kończy się około kroku 500 (sekcja 5
  `koopman_tozsamosc_kierunki`). W pilocie po restarcie optymalizatora zmiana latentu zanikła po 2 obrotach.
- Wniosek: pętla weszła w końcówkę okna, w którym latent jeszcze się rusza. Przez 6 z 8 obrotów Koopman nie
  miał czego przewidywać; klatki wybierał w praktyce obraz obecnego latentu (89 procent wspólnych top-45).

## 2. Wcześnie czy później? Literatura mówi: wcześnie

- **Krytyczne okresy uczenia** (Achille, Rovere, Soatto, *Critical Learning Periods in Deep Networks*,
  arXiv:1711.08856): ingerencja w dane w pierwszych około 15–25 procentach uczenia zmienia wynik końcowy trwale
  (rozmycie obrazów do epoki 40–60 potraja błąd), ta sama ingerencja później prawie nic nie robi. Efekt rośnie z
  głębokością sieci.
- **Wczesna faza uczenia** (Frankle, Schwab, Morcos, *The Early Phase of Neural Network Training*,
  arXiv:2002.10365): struktura sieci ustala się w pierwszych 500–2000 iteracjach (1–5 epok w ich ustawieniu);
  po tym czasie dalsze zmiany kierunku są małe.
- **RHO-LOSS** (Mindermann i in., *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet
  Learnt*, ICML 2022, doi:10.48550/arXiv.2206.07137): dobór „do nauczenia, warte nauczenia, jeszcze nienauczone”
  przyspiesza uczenie 2–18 razy wobec losowego; przewaga jest największa we wczesnej i środkowej fazie, maleje przy
  suficie dokładności. Mały model odniesienia wystarcza.
- Konsekwencja dla nas: pętlę warto uruchomić od początku dostrajania (od wag COCO), z prognozą po pierwszej
  epoce (20 zapisów latentu co 5 aktualizacji wystarczy do dopasowania równania z pamięcią 8), a nie od kroku 245.

- **Ale ostrożnie:** przeszukanie Undermind (190 prac, [link](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/when%20does%20data%20selection%20accelerate%20training%20early%20phase%20and%20model%20size))
  podsumowuje, że wczesna faza jest brzemienna w skutki, ale literatura nie pokazuje, by sam dobór danych bił
  losowanie mocniej, gdy zaczyna się wcześnie, zwłaszcza przy dostrajaniu modelu wstępnie wyuczonego.
  *When Do Curricula Work?* (Wu, Dyer, Neyshabur, ICLR 2021): kolejność danych daje zwykle znikomy zysk nad losową,
  z wyjątkiem krótkiego budżetu czasu i zaszumionych danych — a „jak najszybciej nauczyć” to właśnie krótki budżet.
  *Deep Learning on a Data Diet* (Paul i in., NeurIPS 2021): miary trudności liczone po kilku pierwszych epokach
  (EL2N) wskazują ważne przykłady. *Repeated Random Sampling* (Okanovic i in., 2023): losowanie nowego podzbioru co
  epokę jest bardzo mocnym punktem odniesienia dla czasu do celu — kontrola losowa musi zostać.

## 3. Większy model? Literatura jest niejednoznaczna, a RF-DETR Medium nie jest większym kręgosłupem

- **RF-DETR w rfdetr 1.10.1:** Nano, Small, Base, Medium i Large mają ten sam kręgosłup `dinov2_windowed_small`
  (ViT-S) i hidden_dim 256. Różnią się rozdzielczością (384, 512, 560, 576, 704) i liczbą warstw dekodera (2–4).
  Większy kręgosłup (DINOv2-B) ma tylko wycofany wariant `LargeDeprecated`. Medium to więc ta sama „głowa latentu”,
  ale obraz 576 zamiast 384 pikseli (36 × 36 zamiast 24 × 24 łatek) i głębszy dekoder.
- To może pomóc z innego powodu niż wielkość: końcówka narzędzia zajmuje w Medium 2,25 raza więcej łatek, a pilot
  i teoria (porażka ASAL na małym szczególe) wskazują, że mały szczegół gubi się w dopasowaniu całej sceny.
- **Za większymi modelami:** *Bad Students Make Great Teachers* (Evans i in., doi:10.48550/arXiv.2312.05328):
  dobór danych małym modelem przyspiesza uczenie dużego ViT-L o 26–46 procent aktualizacji.
- **Przeciw:** *Revisiting Active Learning in the Era of Vision Foundation Models* (Gupte i in.,
  doi:10.48550/arXiv.2401.14555): przy większych modelach zysk z doboru jest duży tylko przy bardzo małym budżecie
  i szybko znika. *On Lazy Training in Differentiable Programming* (Chizat, Oyallon, Bach, arXiv:1812.07956):
  im szerszy model, tym mniej jego cechy ruszają się w uczeniu (przy standardowej skali), więc tym mniej dryfu
  latentu do przewidywania. *Beyond neural scaling laws* (Sorscher i in., doi:10.48550/arXiv.2206.14486): przy
  dużej pojemności i małej liczbie danych (nasze 1500 klatek) lepiej dodawać łatwe przykłady, nie trudne.
- *When does loss-based prioritization fail?* (Hu i in., arXiv:2107.07741): dobór po samej trudności zawodzi przy
  szumie etykiet (u nas: część trudnych klatek to graniczne oznaczenia, 7e) — RHO-LOSS jest na to odporny.

## 4. Rekomendacja

1. Najpierw **pętla od początku dostrajania na Nano** (tani test hipotezy „wcześnie”): start z wag COCO, pierwsza
   epoka zwykła (historia do Koopmana), potem obroty od epoki 1, gdy latent jeszcze się mocno przesuwa. Trzy
   ziarna, bo jedno ziarno nie rozróżnia różnic poniżej 0,01.
2. **Cel „przyspieszony”** (propozycja z `PLAN.md`) plus kryterium RHO-LOSS przy wyborze klatek z puli: z klatek
   podobnych do obrazu FastGAN brać te, które detektor jeszcze myli, a mały model odniesienia rozpoznaje
   (do nauczenia i jeszcze nienauczone), zamiast najbliższych sąsiadów.
3. **Stała liczba powtórzeń 1500 klatek bazy** w każdym obrocie we wszystkich pętlach (w pilocie rosnący zbiór
   rozcieńczał bazę z 1,76 do 1,03 powtórzeń na obrót).
4. **Medium dopiero potem**, jako test „wyższej rozdzielczości”, nie „większego modelu”; wymaga przestawienia
   sond i czytnika FastGAN na siatkę 36 × 36 i około 2,3 raza więcej obliczeń na obraz.

Przewidywanie do sprawdzenia: jeżeli metoda ma działać, jej przewaga pojawi się w pierwszych 2–3 obrotach od
początku dostrajania (tam gdzie dryf latentu jest duży) i zniknie, gdy dryf zgaśnie.
