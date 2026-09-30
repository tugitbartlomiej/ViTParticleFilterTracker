# 24. Tożsamość narzędzia w latencie, Koopman i Langevin: pełny opis eksperymentu od początku do końca (21-30 września 2026)

- **Zakres:** cała linia prac od 21 do 30 września 2026 w repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`:
  commity od `e6b24173` (21.09, 13:27) do `255f6668` (30.09, 12:46), zadania na klastrze Eden od 1809131 do 1818496.
- **Stan opisu:** 30 września 2026, po południu. Uczenie Stable Diffusion (zadanie 1818496) w chwili pisania jeszcze trwa;
  jego wynik końcowy jest dopisywany w sekcji F.
- **Źródła:** opisy sesji w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions`, pliki `PLAN`, `WYNIKI`, `DIAGNOZA`,
  `ALGORYTM` i wyniki JSON w folderach `SmallViTAblation\experiments`, dokumenty 36-46 w
  `SmallViTAblation\docs\Analiza_ponownej_publikacji`, dokument 23 w `SmallViTAblation\docs`, historia git i zapis rozmowy.
  Liczby są przepisane ze źródeł. Tam, gdzie źródła się różnią, opis podaje obie wersje, a zbiorczą listę sprzeczności ma
  sekcja G.2.
- **Jak powstał:** opisy pięciu etapów napisano osobno na podstawie źródeł danego okresu, a ramę (sekcje A-G) przy składaniu
  całości. Każdy etap można czytać samodzielnie, dlatego pojęcia są objaśniane przy pierwszym użyciu w każdym etapie.

## Spis treści

- A. Streszczenie całości
- B. Słownik pojęć
- C. Oś czasu
- D. Mapa eksperymentów: folder, pytanie, wynik
- Etap 1. Koopman i Langevin na latencie detektora RF-DETR: dobór klatek, widmo, kod tożsamości, krystalizacja latentu (21-22.09)
- Etap 2. Tożsamość narzędzia w latencie, kryterium domknięcia, test na nietkniętym zbiorze i plany doboru klatek (23.09)
- Etap 3. Koopman z pamięcią, pętla prognoza - FastGAN - wyszukiwanie - uczenie na RF-DETR Nano i Large (25-28.09)
- Etap 4. Diagnoza porażki, mały ViT od zera, wektor i siła tożsamości, FastGAN wzdłuż wektora, pętla w trzech wersjach (28-29.09)
- Etap 5. Horyzont prognozy, literatura o dyfuzji, równanie ruchu latentu, macierze jak w TS, Stable Diffusion na filmach (29-30.09)
- E. Synteza: co wiemy po dziesięciu dniach
- F. Stan na koniec opisu i następne kroki
- G. Źródła i sprzeczności w źródłach

## A. Streszczenie całości

### A.1 Pytanie

Projekt uczy detektor, czyli sieć, która na klatce filmu z operacji zaćmy zaznacza ramką narzędzie chirurgiczne. Sieć
w środku opisuje każdą klatkę wektorem liczb, zwanym latentem. W trakcie uczenia latent się przesuwa, a użytkownik traktuje
jego składowe jak cząstki fizyczne poruszające się w przestrzeni wielowymiarowej. Pytanie całej linii prac brzmiało:

> Czy ruch latentu w trakcie uczenia da się przewidzieć równaniem (operatorem Koopmana albo równaniem Langevina), a jeśli tak,
> czy prognoza „dokąd latent tożsamości narzędzia pójdzie” pozwoli dobrać nowe klatki do uczenia tak, żeby detektor uczył się
> szybciej albo lepiej niż przy klatkach losowych?

Po drodze doszły pytania pomocnicze: czym właściwie jest „tożsamość narzędzia” w latencie i jak ją zmierzyć, czy generator
obrazów (FastGAN, model dyfuzyjny, Stable Diffusion) umie narysować obraz o zadanym latencie, i jaką postać ma równanie ruchu
latentu w trakcie uczenia.

### A.2 Pięć etapów w pięciu akapitach

**Etap 1 (21-22 września), RF-DETR Nano.** Zbudowano aparat do pomiaru latentu dostrajanego detektora RF-DETR Nano i
sprawdzono dobór klatek w kontrolowanej macierzy: 9 ramion, 3 ziarna, budżet 1500 klatek, 27 treningów. Reszta Langevina,
czyli to, czego operator nie przewidział, nie pomogła wybierać klatek (przewaga +0,0027 AP przy progu +0,004), a wszystkie
rozsądne sposoby doboru zmieściły się w 0,005 AP od siebie. Latent „krystalizuje” w trzech pierwszych epokach: położenie
narzędzia daje się z niego odczytać z R2 0,68-0,86, a przed uczeniem wcale. Powstał „kod tożsamości” z niezmienników
operatora, który niesie głównie wydłużenie ramki (85,8 procent). Strata walidacyjna ma dołek w epokach 6-7, a uczenie do
30. epoki kosztuje średnio 0,030 AP.

**Etap 2 (23 września), tożsamość i domknięcie.** Dobór 250 klatek „po kwadratach” przestrzeni tożsamości pokrył ją jak
2000 klatek losowych, ale detektor wyszedł taki sam jak z losowych (0,6296 wobec 0,6319). Powstały teoria i wzór
„domknięcia” tożsamości (dokumenty 43-45) oraz 9 twierdzeń sprawdzonych w Lean; wynika z nich, że gwarancję poprawy daje
tylko przewidziana poprawa straty większa od błędu tego przewidywania, a nie samo widmo operatora. W latencie nie ma
kierunku odpornego na obrót i tło, a „kierunki tożsamości” Koopmana okazały się kątem ułożenia narzędzia. Jest za to jeden
kierunek „tu jest narzędzie” w tokenach kręgosłupa: AUC 0,975 nawet bez etykiet. Na nietkniętym zbiorze testowym (25
operacji) detektor ma mAP 0,681. Zaskoczenie Koopmana nie wskazuje chybień (AUC 0,58), a wybór klatek ze zmianami w
stałych kierunkach Koopmana przegrał z losowym na każdym z trzech ziaren (0,6696 wobec 0,6820).

**Etap 3 (25-28 września), pierwsza pętla z generatorem.** Operator z pamięcią, czyli stan [położenie, ostatnie przyrosty],
przewiduje ruch latentu wyraźnie lepiej niż sam stan (błąd 0,0037 wobec 0,0103), bo ruch w uczeniu ma bezwładność. Pętla
„prognoza latentu - obraz z FastGAN-a - wyszukanie podobnych prawdziwych klatek - dołożenie ich do uczenia” zadziałała
technicznie po naprawie inwersji generatora, ale przewidziane przesunięcie latentu było tak małe, że obraz zmieniał się o
0,1-0,9 procent jasności, a wyszukiwarka zwracała prawie tych samych sąsiadów (54 z 60 wspólnych). Dołożenie 134
dobranych klatek dało tyle, co 134 losowe (+0,0007 wobec +0,0009 mAP@0,5), kampania 5 epok nie spełniła kryterium, pętla 8
obrotów na Nano wypadła najlepiej bez dokładania, a pętla na większym RF-DETR Large przegrała z losowymi w 15 z 18 par.

**Etap 4 (28-29 września), mały ViT od zera.** Diagnoza porażki: w dostrajanym RF-DETR kręgosłup prawie się nie zmienia
(0,026 procent wag w 6 epokach), a prognoza Koopmana była gorsza od „braku zmiany” w 12 z 12 przypadków. Użytkownik
zdecydował uczyć mały ViT od zera (2,93 mln parametrów). Trzy wersje pętli na Edenie (11, 10 i 10 ziaren, potem 20 ziaren
w iteracji 2) pokazały, że metoda uczy szybciej niż losowe klatki (w iteracji 2: 15 z 20 ziaren, p = 0,006), ale ramię z
tymi samymi kotwicami bez prognozy daje prawie to samo, więc zysk pochodzi z wyszukiwania klatek podobnych do narzędzia, a nie
z prognozy. Tożsamość mierzona wektorem LDA i jego „siłą” jest pierwszą wielkością przewidywaną lepiej niż „brak zmiany”;
FastGAN sterowany wzdłuż tego wektora realizuje przewidzianą zmianę siły (znak zgodny 12 z 12). Na parach „to samo miejsce z
narzędziem i bez” powstał wektor `nowy`, który oddziela narzędzie od sceny z d' 16,6 (stary 2,0), a tożsamość zajmuje wąską
wiązkę kierunków (1-2 kierunki z AUC co najmniej 0,9).

**Etap 5 (29-30 września), równanie ruchu i Stable Diffusion.** Siła tożsamości daje się przewidywać lepiej niż „brak
zmiany” do 1000 kroków naprzód, ale tylko o 10-25 procent błędu. Równanie ruchu latentu małego ViT w trakcie uczenia ma
prostą postać: następny krok latentu to 83 procent poprzedniego plus siła bieżącej paczki obrazów; na 10 ziarnach Edenu
wyjaśnia 85 procent ruchu (mediana, każde ziarno powyżej 80). Użytkownik ustalił regułę: prognoza nie może używać przyszłych
klatek, a wyjaśnianie klatkami, które właśnie tworzą krok, jest dozwolone. Prognoza bez przyszłych klatek osiąga 57-67
procent, a duże, uczone macierze w stylu TS tylko około 40 procent, bo uczą się okna na pamięć. Na koniec wysłano na Eden
całe filmy Cataract-LMM z obrysami (49 GB) i uruchomiono dostrajanie Stable Diffusion 1.5 na wszystkich klatkach 118 filmów
treningowych, z maskami narzędzia, rogówki i źrenicy jako warunkiem, na 4 kartach H100.

### A.3 Odpowiedź na dziś

Nie wykazano, że prognoza ruchu latentu pomaga dobierać klatki. W każdym porównaniu, w którym prognoza była jedyną różnicą
między ramionami, wynik mieścił się w szumie ziaren: w macierzy etapu 1, w kampaniach 134 klatek i 5 epok, w pętli na RF-DETR
Large i w wersji 3 pętli na małym ViT (ramię z prognozą 0,4432, ramię bez prognozy 0,4428). Tam, gdzie metoda wygrywała z
losowaniem, zysk dawało wyszukiwanie prawdziwych klatek podobnych do narzędzia.

Powód jest zmierzony, a nie zgadnięty. Przewidziane przesunięcia latentu są za małe, żeby zmienić obraz z generatora i listę
wyszukanych klatek (twierdzenie o stabilności wyboru z etapu 3 mówi, że przy małym przesunięciu wybór najbliższych klatek się
nie zmienia). W dostrajanym RF-DETR kręgosłup prawie się nie rusza, więc nie ma czego przewidywać. FastGAN, trafiając w
przewidzianą siłę tożsamości, zmienia przy okazji resztę latentu około 40 razy mocniej. Dobór „trudnych” albo „brakujących”
klatek traci różnorodność i dlatego przegrywa z losowaniem.

Po drodze powstały jednak rzeczy, które działają i są zmierzone: jednoznaczna miara tożsamości narzędzia (wektor `nowy`,
d' 16,6, AUC 0,997), równanie ruchu latentu w trakcie uczenia (85 procent w trybie wyjaśnienia na 10 ziarnach), prognoza siły
tożsamości lepsza od „braku zmiany” do 1000 kroków, sterowanie FastGAN-em wzdłuż wektora tożsamości (12 z 12) i generator
Stable Diffusion uczony na wszystkich klatkach filmów.

### A.4 Najważniejsze liczby

| co | wynik | etap |
|---|---|---|
| macierz doboru 9 ramion × 3 ziarna, budżet 1500 (RF-DETR Nano) | reszta Langevina +0,0027 AP przy progu +0,004; wszystkie rozsądne ramiona w 0,005 AP; równomierny 0,6866 | 1 |
| krystalizacja latentu | R2 położenia narzędzia z latentu: około 0 przed uczeniem, 0,68-0,86 po; przebudowa w pierwszych 3 epokach | 1 |
| 250 klatek po kwadratach przestrzeni tożsamości | 0,6296 ± 0,0011 wobec losowych 0,6319 ± 0,0023 | 2 |
| wektor „tu jest narzędzie” w tokenach kręgosłupa | AUC 0,976 z etykietami, 0,975 bez etykiet | 2 |
| nietknięty zbiór testowy (25 operacji, 8249 klatek) | mAP 0,681 (najlepsza epoka), 0,654 (ostatnia) | 2 |
| zaskoczenie Koopmana jako wskaźnik chybień | AUC 0,58 (wynik negatywny); pewność detektora 0,88 | 2 |
| klatki ze zmianami w stałych kierunkach Koopmana | 0,6696 ± 0,0015 wobec losowych 0,6820 ± 0,0008 (przegrana na 3 z 3 ziaren) | 2 |
| Koopman ze stanem z pamięcią [z, v] | błąd jednego kroku 0,0037 wobec 0,0103 bez pamięci | 3 |
| 134 klatki dobrane pętlą FastGAN | +0,0007 mAP@0,5, tyle co 134 losowe (+0,0009) | 3 |
| pętla na RF-DETR Large, 6 obrotów × 3 ziarna | koopman - losowe = -0,0030 mAP@0,5, ujemne w 15 z 18 par | 3 |
| zmiana wag kręgosłupa RF-DETR Large w 6 epokach | 0,026 procent; prognoza gorsza od „braku zmiany” w 12 z 12 | 4 |
| mały ViT od zera, wersja 1 (11 ziaren) | lepszy od losowych w 9 z 11 (+0,013, p = 0,005), ale prognoza = „brak zmiany” | 4 |
| mały ViT, wersja 3 (10 ziaren) | ramię z prognozą 0,4432, bez prognozy 0,4428 (5 z 10) | 4 |
| mały ViT, wersja 3, iteracja 2 (20 ziaren) | szybsze uczenie niż losowe w 15 z 20 (p = 0,006); na końcu 11 z 20 | 4 |
| FastGAN sterowany wzdłuż wektora tożsamości | znak zmiany siły zgodny 12 z 12; 85-111 procent przetrwa zaburzenia | 4 |
| wektor tożsamości z par „z narzędziem i bez” | d' 16,6 (stary 2,0), AUC 0,997; 1-2 kierunki z AUC ≥ 0,9, 5-8 z AUC ≥ 0,75 | 4 |
| horyzont prognozy siły tożsamości | lepsza od „braku zmiany” do 1000 kroków o 10-25 procent błędu | 5 |
| równanie ruchu latentu (tryb wyjaśnienia), 10 ziaren | Δz(t+1) = 0,83·Δz(t) + 0,98·S(t): mediana 85,0 (80,5-89,2) | 5 |
| prognoza bez przyszłych klatek | 64-67 (siła z pędu), 57 (bezwładność), około 40 (uczone macierze jak w TS) | 5 |
| Stable Diffusion 1.5 na 118 filmach, 4 × H100 | 209 obrazów/s; strata 0,17 → 0,11 do kroku 6200; wynik końcowy w sekcji F | 5 |


## B. Słownik pojęć

Hasła są pogrupowane tematycznie. Każde objaśnienie mówi, co słowo znaczy w tym eksperymencie, a nie w ogóle. W opisach etapów
te same pojęcia są objaśniane jeszcze raz przy pierwszym użyciu, bo każdy etap można czytać osobno.

### B.1 Słowa wieloznaczne: najpierw przeczytaj to

- **Latent** znaczy w tym dokumencie dwie różne rzeczy. W etapach 1-5 (do równania ruchu włącznie) to wewnętrzny opis obrazu w
  sieci wykrywającej narzędzie: wektor liczb, który sieć wytwarza dla klatki albo dla jednego jej kawałka (tokenu). W Stable
  Diffusion (koniec etapu 5) „latent obrazu” to ściśnięty obraz: klatka 576 × 384 × 3 zapisana jako 72 × 48 × 4 liczb przez koder
  VAE. Te dwa latenty nie mają ze sobą nic wspólnego.
- **Dekoder** też ma trzy znaczenia. W RF-DETR to część detektora, która z cech obrazu wyznacza ramki narzędzia (etap 4, „odczyt
  dekodera”). W VAE to sieć, która z latentu obrazu odtwarza piksele. W planie `maly_vit_dekoder_latentu` to sieć, która miałaby
  rysować obraz wprost z latentu małego ViT.
- **Czas** biegnie w dwóch osiach. Czas filmu to kolejne klatki nagrania (narzędzie się porusza). Czas uczenia to kolejne kroki
  poprawiania wag sieci (sieć się uczy). Operator Koopmana i równanie Langevina stosowano w obu osiach: w etapie 1 głównie wzdłuż
  filmu i wzdłuż epok, od etapu 3 wzdłuż kroków uczenia.
- **Koopman** i **Langevin** zmieniały treść w trakcie prac. „Koopman” to zawsze liniowa reguła przewidująca następny stan z
  obecnego, ale stan był raz współrzędnymi latentu, raz niezmiennikami tokenów, raz jedną liczbą (siłą tożsamości). „Langevin”
  to zawsze ruch z hamowaniem, pamięcią i szumem, ale z różnymi członami. Opisy etapów mówią za każdym razem, o który wariant
  chodzi.
- **Tożsamość narzędzia** to cecha latentu, która odróżnia narzędzie od tła. Mierzono ją kolejno: kodem niezmienników (etap 1),
  wektorem narzędzia w tokenach (etap 2), skalarem tożsamości (etap 3), rdzeniem neuronów, wektorem LDA i siłą tożsamości (etap
  4), wektorem `nowy` z par „z narzędziem i bez” oraz kierunkami cząstkowymi (koniec etapu 4). To nie jest „które narzędzie”
  (zbiór ma jedną klasę narzędzia), tylko „jest narzędzie czy go nie ma”.

### B.2 Dane, zadanie i miary

- **Klatka** - jeden obraz z filmu operacji. **Operacja, film** - jedno nagranie; klatki jednego filmu są do siebie bardzo
  podobne, dlatego podziały na zbiory robi się po filmach, nie po klatkach.
- **Zbiór uczący (train), walidacyjny (val), testowy (test)** - klatki do uczenia, do bieżącej kontroli i do ostatecznej, jednorazowej
  oceny. „Nietknięty zbiór testowy” (etap 2) to 25 operacji, których nie użyto do niczego wcześniej.
- **Pula** - klatki, z których metoda doboru wybiera nowe klatki do uczenia. **Budżet** - ile klatek wolno wybrać (np. 1500).
- **Cataract-LMM** - publiczny zbiór filmów operacji zaćmy; podzbiór `3_Object_Tracking` ma na każdej klatce obrysy cystotomu
  (narzędzia), rogówki i źrenicy.
- **Ramka** - prostokąt wokół narzędzia, który przewiduje detektor. **IoU** - pole części wspólnej ramki przewidzianej i prawdziwej
  podzielone przez pole ich sumy; **chybienie** to IoU poniżej 0,5; **pewne chybienie** - chybienie przy pewności detektora co
  najmniej 0,5. **Precyzja** - trafne wśród zgłoszonych ramek; **czułość** - wykryte wśród narzędzi obecnych na klatkach.
- **Wczesne zatrzymanie** - koniec uczenia po kilkunastu epokach bez poprawy na walidacji. **Mapa zbioru danych** - podział
  przykładów według tego, jak pewność sieci na nich waha się w trakcie uczenia (łatwe, trudne, niejednoznaczne).
- **AP, mAP@0,5, mAP@0,5:0,95** - miary jakości detektora (0 do 1): pole pod krzywą precyzja-czułość; „@0,5” liczy ramkę za
  trafioną przy IoU co najmniej 0,5, „@0,5:0,95” uśrednia po ostrzejszych progach. **F1** - średnia harmoniczna precyzji i czułości.
- **Pole pod krzywą mAP** (w pętlach) - średnie mAP po wszystkich obrotach pętli; mierzy, jak szybko detektor się uczy, a nie
  tylko, gdzie kończy.
- **AUC** - jak dobrze jedna liczba rozdziela dwie grupy: szansa, że losowy przykład z grupy „tak” dostanie wyższy wynik niż losowy
  z grupy „nie”; 0,5 to zgadywanie, 1 to rozdzielenie idealne.
- **d'** - odległość średnich dwóch grup podzielona przez rozrzut; d' = 2 to grupy nachodzące na siebie, d' = 16 to grupy bardzo
  daleko od siebie.
- **R2** - jaka część zmienności wielkości daje się odczytać z latentu (1 = całość, 0 = nic, ujemne = gorzej niż średnia).
- **Cosinus** - zgodność kierunków dwóch wektorów: 1 to ten sam kierunek, 0 to kierunki prostopadłe (0,40 odpowiada około 66°).
- **Korelacja Spearmana ρ** - zgodność kolejności dwóch list liczb. **Jaccard** - udział wspólnych elementów dwóch zbiorów.
- **Test Wilcoxona, test znaków, p** - testy dla par wyników z tych samych ziaren; p to prawdopodobieństwo takiego lub większego
  wyniku, gdyby metoda nie miała żadnego efektu. **Bootstrap** - ocena niepewności przez losowanie operacji ze zwracaniem.
- **Ziarno** - liczba ustalająca losowość uczenia (kolejność klatek, start wag); ten sam eksperyment na kilku ziarnach pokazuje, ile
  różnic to szum.
- **Ramię** - jeden wariant porównania (np. „losowe klatki”, „klatki z prognozy”). **Placebo** - ramię, które przechodzi tę samą
  procedurę, ale z sygnałem wziętym z niewłaściwego miejsca (np. z innego kroku); gdy placebo daje tyle samo, sygnał nie działa.
- **Kryterium ustalone przed wynikiem (bramka)** - warunek sukcesu zapisany przed uruchomieniem, żeby nie dopasować oceny do wyniku.

### B.3 Sieci i uczenie

- **Wagi** - liczby wewnątrz sieci, które zmienia uczenie. **Krok (aktualizacja)** - jedna poprawka wag. **Paczka (batch)** - obrazy
  użyte w jednym kroku. **Epoka** - jedno przejście przez zbiór uczący (w pętlach bywa to umowna porcja kroków zakończona oceną).
- **Strata** - liczba mówiąca, jak bardzo sieć się myli na paczce; uczenie ją zmniejsza. **Strata walidacyjna** - ta sama liczba na
  klatkach walidacyjnych. **Gradient** - kierunek, w którym trzeba zmienić wagi, żeby strata spadła najszybciej.
- **AdamW i pęd** - sposób poprawiania wag, który pamięta średnią z dawnych gradientów; ta pamięć (pęd) daje wagom bezwładność:
  sieć jeszcze przez kilkadziesiąt kroków „jedzie” w kierunku, który wskazywały dawne paczki. **lr** - tempo uczenia, długość kroku.
  **Rozgrzewka** - początkowe kroki z małym tempem, wspólne dla wszystkich ramion.
- **Punkt kontrolny (checkpoint)** - zapisany stan wag. **Kontynuacja treningu** - uczenie od zapisanych wag.
- **Przeuczenie** - dalsze uczenie pogarsza wynik na nowych klatkach, bo sieć dopasowuje się do szczegółów zbioru uczącego.
  **Wkuwanie** (etap 5) - to samo dla równania: dopasowanie przypadkowych szczegółów okresu dopasowania zamiast reguły.
- **Augmentacja** - losowe przekształcenia obrazów uczących (obrót, kolor), żeby sieć nie przywiązywała się do szczegółów.
- **BatchNorm** - warstwa normująca liczby wewnątrz sieci średnimi z paczek. **EMA wag** - wolno uśredniana kopia wag, gładsza od
  bieżących, używana do generowania obrazów. **bf16, TF32** - skrócone formaty liczb, szybsze na kartach graficznych.
- **Dostrajanie** - douczanie gotowych, już wytrenowanych wag zamiast uczenia od zera. **Wagi COCO** - wagi startowe detektora
  wyuczone na ogólnym zbiorze zdjęć COCO.
- **Detektor** - sieć, która na klatce zwraca ramkę narzędzia i pewność. **RF-DETR** (warianty Nano, Large, LargeDeprecated) -
  detektor używany w projekcie. Jego części: **kręgosłup (backbone)** zamienia obraz w cechy, **projektor** łączy kręgosłup z
  dekoderem, **dekoder** wyznacza ramki przez **zapytania (queries)**, czyli wektory, z których każdy „szuka” jednego obiektu.
- **DINOv2** (S mały, B większy) - sieć wyuczona na milionach zdjęć bez etykiet; w RF-DETR pełni rolę kręgosłupa, w pętlach
  służyła też do wyszukiwania podobnych klatek.
- **ViT (Vision Transformer)** - sieć, która tnie obraz na kwadraty (tokeny) i przetwarza je blokami „uwagi” (każdy token patrzy na
  wszystkie inne). **Mały ViT** (etapy 4-5) - własny detektor uczony od zera: łodyga konwolucyjna (pierwsze warstwy splotowe
  zamieniające obraz w siatkę tokenów), 6 bloków, gęsta głowa (każdy token przewiduje środek narzędzia i krawędzie ramki), około
  2,93 mln parametrów, latent 768 liczb na token.
- **MLP** - mała sieć z warstw w pełni połączonych.

### B.4 Latent, tożsamość i jej miary

- **Token, cząstka** - wektor opisujący jeden kwadrat obrazu. W modelu użytkownika każdy token to cząstka poruszająca się w
  przestrzeni latentu (384 wymiary w Nano, 768 w Large i w małym ViT) w miarę uczenia.
- **Sonda** - stały obraz kontrolny, na którym w każdym punkcie kontrolnym mierzy się latent. Sondy miały wersje: oryginał, obrót,
  czarne tło, żółte tło, krótsze i jaśniejsze narzędzie; w etapie 4 doszły pary „to samo miejsce z narzędziem i bez” (narzędzie
  zamalowane, czarne tło).
- **Sonda liniowa** - najprostsza reguła (ważona suma), która próbuje odczytać z latentu wielkość, np. położenie narzędzia; jeśli się
  udaje, latent tę wielkość zawiera.
- **Kotwica** - prawdziwy obraz (wycinek z narzędziem), od którego startuje generowanie albo wyszukiwanie w pętlach.
- **Kod tożsamości** (etap 1) - niezmienniki operatora translacyjnego policzone na tokenach narzędzia. **Skalar tożsamości** (etap
  3) - rzut różnicy „narzędzie minus tło” na kierunek LDA. **Rdzeń tożsamości** (etap 4) - neurony, które zapalają się na narzędziu
  we wszystkich wersjach sondy.
- **Wektor tożsamości** - kierunek w latencie (768 liczb), który najlepiej oddziela narzędzie od tła; wyznaczany przez **LDA z
  Ledoit-Wolfem** (liniowa analiza dyskryminacyjna ze stabilizowaną macierzą rozrzutu). **Siła tożsamości** - rzut latentu sondy na
  ten kierunek, czyli jedna liczba mówiąca, jak mocno sieć „widzi” narzędzie. **Wektor `nowy`** - wektor wyznaczony na parach z
  narzędziem i bez (etap 4), odporny na mylenie narzędzia ze sceną.
- **Kierunek cząstkowy, INLP, energia różnic** - kierunki wyznaczane na pojedynczych tokenach zamiast na średniej; INLP szuka
  kolejnych kierunków rozdzielających po usunięciu poprzednich; energia różnic mówi, w ilu kierunkach mieści się różnica „z
  narzędziem minus bez”.
- **Krystalizacja latentu** - porządkowanie się latentu w czasie uczenia (latent zaczyna nieść położenie i stan narzędzia).
  **Participation ratio** - efektywna liczba kierunków, w których rozkłada się zmienność latentu. **CKA** - podobieństwo kształtu
  dwóch chmur punktów latentu (np. przed i po kolejnej epoce).
- **Domknięcie** - stan, w którym latent tożsamości przestaje się zmieniać (zatrzymanie), nie zależy od nieistotnych zmian obrazu
  (niezmienniczość) i rozdziela narzędzie od reszty (rozdzielenie); **R** to rozrzut wersji jednego obrazu podzielony przez rozrzut
  między obrazami (małe R = domknięte). **Zajętość, nasycenie** (etap 2) - jaka część przestrzeni tożsamości jest już pokryta
  przez klatki uczące i czy dokładanie klatek jeszcze ją poszerza.

### B.5 Dynamika: Koopman, Langevin i równanie ruchu

- **Operator Koopmana** - liniowa reguła (macierz), która z obecnego stanu przewiduje następny. **Mod i wartość własna** - kierunek,
  który operator tylko skaluje (i ewentualnie obraca) przy każdym kroku; moduł wartości własnej mówi, jak szybko mod gaśnie, a kąt
  (część urojona) - jak szybko się obraca. **Niezmiennik** - mod, który prawie nie gaśnie (moduł bliski 1), więc niesie stałą cechę.
- **Operator translacyjny** (etap 1) - operator opisujący przesunięcie narzędzia między sąsiednimi położeniami; **symetryzacja** -
  wymuszenie rzeczywistych wartości własnych (bez obrotów).
- **Deep Koopman (autoenkoder Koopmana)** - sieć ucząca się takich współrzędnych, w których krok jest liniowy.
- **Stan z pamięcią [z, v]** - położenie latentu plus jego ostatnie przyrosty; pozwala uchwycić bezwładność.
- **Rollout** - prognoza na wiele kroków, w której każdy krok startuje z poprzedniej prognozy.
- **„Brak zmiany” (persystencja)** - prognoza „będzie tak jak teraz”; każda metoda musi ją pobić. **Umiejętność U, błąd względem
  braku zmiany** - jaki procent błędu prognozy „bez zmiany” usuwa model (etap 5), albo odwrotnie: błąd modelu jako ułamek błędu
  „bez zmiany” (etap 3, np. 0,272 znaczy 27 procent tego błędu).
- **Równanie Langevina** - ruch cząstki z hamowaniem (dryf ku położeniu równowagi) i losowymi kopnięciami (szum).
  **Uogólnione równanie Langevina (GLE)** - to samo z pamięcią: przeszłe stany wpływają na obecny krok. **Ornstein-Uhlenbeck** -
  najprostszy Langevin z liniowym hamowaniem. **Mori-Zwanzig** - twierdzenie, że gdy obserwujemy tylko część układu, pominięte
  współrzędne wracają jako pamięć i szum.
- **Jądro pamięci** - wagi, z jakimi dawne kroki wpływają na obecny. **Wymuszenie** - człon zewnętrznego sterowania (w TS: plan
  promocji i kalendarz; w małym ViT: obrazy paczek). **Szum kolorowy** - szum skorelowany w czasie, liczony z własnych reszt równania.
- **Siła paczki S(t)** (etap 5) - o ile przesunąłby się latent sond, gdyby wagi zmieniły się tylko o tę część kroku, która pochodzi
  z bieżącej paczki obrazów. **Siła z pędu** - przesunięcie od części kroku, która pochodzi z pamięci optymalizatora, czyli z
  wcześniejszych paczek. **Bezwładność** - część poprzedniego ruchu latentu przenoszona na następny krok (0,83 w małym ViT).
- **Tryb wyjaśnienia a tryb prognozy** (etap 5) - wyjaśnienie może używać klatek, które właśnie tworzą obecny krok; prognoza może
  używać tylko stanów do chwili t, paczek do t-1 i pędu (reguła użytkownika z 29.09).
- **Filtr Kalmana** - sposób łączenia prognozy modelu ruchu z zaszumionym pomiarem; **innowacja** - różnica między pomiarem a
  prognozą filtru; **kurtoza** - miara grubości ogonów rozkładu (duża kurtoza = rzadkie, bardzo duże odchylenia).
- **Regresja grzbietowa** - dopasowanie metodą najmniejszych kwadratów z karą za duże współczynniki. **PCA** - kierunki największej
  zmienności danych. **Wybielenie** - przeskalowanie danych tak, żeby kierunki były nieskorelowane i miały jednakowy rozrzut.
- **EDMD** - praktyczny sposób liczenia operatora Koopmana: dopasowanie macierzy, która z opisu bieżącej klatki przewiduje opis
  następnej. **Para sprzężona** - dwa mody, które razem obracają się jak wskazówka zegara (jeden obrót opisują dwie liczby).
- **Wyciek** (etap 2) - jak bardzo podprzestrzeń tożsamości „wycieka” poza siebie po jednym kroku operatora (sinus kąta między
  nią a jej obrazem); zerowy wyciek to warunek domknięcia. **Kosinusy główne** - zgodność dwóch podprzestrzeni kierunek po kierunku.
- **Związek fluktuacji i dyssypacji** - reguła fizyki łącząca jądro pamięci z samopodobieństwem szumu w czasie; w etapie 2 służył
  do sprawdzenia, czy pamięć policzona dwiema drogami się zgadza. **Podłoga szumu** - wartość miary, jaką daje proces bez żadnej
  pamięci (95. centyl); wynik poniżej podłogi nie jest dowodem pamięci.
- **Odległość Mahalanobisa** - błąd przeskalowany przez rozrzut danych w każdym kierunku. **Zaskoczenie Koopmana** (etap 2) - błąd
  przewidywania następnej klatki przez operator; **reszta Langevina** - ten błąd wyrażony w jednostkach zwykłego szumu.
- **Filtr cząsteczkowy** - zbiór ważonych hipotez o stanie (np. położeniu narzędzia) przesuwanych modelem ruchu i ocenianych
  pomiarem. **Opis Eulera** (etap 1) - cząstką jest stałe miejsce siatki obrazu, a nie poruszające się narzędzie.
- **Test wsteczny** - sprawdzenie prognozy na przeszłości, którą już znamy, ale której model nie widział przy dopasowaniu.

### B.6 Dobór klatek i pętle

- **Dobór klatek** - wybór klatek z puli do dołożenia do uczenia. **Dobór równomierny w czasie** - klatki co stały odstęp w filmie.
  **k-centrum** - kolejno klatka najdalsza od już wybranych. **Dobór D-optymalny** - wybór maksymalizujący wyznacznik macierzy
  informacji, czyli najbardziej zmniejszający niepewność dopasowanego modelu. **GLISTER** - wybór klatek, których gradient
  najbardziej zmniejsza stratę na osobnym zbiorze sterującym. **RHO-LOSS** - wybór klatek trudnych dla bieżącego modelu, a
  łatwych dla modelu odniesienia.
- **Wyszukiwanie najbliższych sąsiadów (FAISS)** - szybkie znajdowanie klatek, których opis (np. cechy DINOv2) jest najbliższy
  zapytaniu. **Top-k (top-50)** - k najpodobniejszych klatek na jedną kotwicę.
- **Obrót pętli** - jeden cykl: prognoza latentu, obraz z generatora, wyszukanie podobnych prawdziwych klatek, dołożenie ich do
  uczenia, porcja uczenia. **Ramiona pętli**: `koopman`/`tozsamosc` (klatki wyszukane pod obraz przewidzianego latentu), `current`
  (te same kotwice, ale obraz bez prognozy), `losowe`, `kolejne` (klatki po kolei z filmów), `bez` (bez dokładania), `trudne`.

### B.7 Generatory obrazów

- **FastGAN** - lekki generator, który z 256 liczb wejścia rysuje wycinek z narzędziem (uczony na wycinkach z narzędziem).
  **Dyskryminator** - sieć oceniająca, czy obraz wygląda na prawdziwy. **Inwersja** - szukanie takiego wejścia generatora, które
  odtwarza daną kotwicę. **Krok Gaussa-Newtona (Levenberga-Marquardta)** - krok poprawiania wejścia oparty na przybliżeniu liniowym
  odwzorowania „wejście generatora - latent detektora”.
- **Sterowanie wzdłuż wektora** - zmiana obrazu tak, żeby latent przesunął się tylko wzdłuż wektora tożsamości. **Zmiana „w bok”** -
  zmiana latentu poza tym kierunkiem (niepożądana).
- **Model dyfuzyjny** - generator, który uczy się zdejmować szum z obrazu; rysuje, zaczynając od czystego szumu i odszumiając go
  krok po kroku. **DDPM, DDIM** - schematy tego odszumiania (DDIM: szybki, deterministyczny, kilkadziesiąt kroków). **ᾱ_t** - udział
  sygnału w obrazie zaszumionym do poziomu t; przy t = 999 prawie zero, co dało błąd kolorów w etapie 5.
- **Prowadzenie (guidance), FreeDoM** - dokładanie w trakcie rysowania gradientu zewnętrznej sieci (np. małego ViT), żeby obraz miał
  zadaną cechę. **ReNO** - poprawianie samego szumu startowego generatora. **Kontrfakt** - minimalnie zmienione zdjęcie, które
  zmienia decyzję sieci. **Inpainting (Telea)** - zamalowanie fragmentu obrazu na podstawie otoczenia. **Wklejanie (copy-paste)** -
  wycięcie narzędzia po obrysie i wklejenie go na inne tło. **Włókno** (praca Rousselota) - zbiór różnych obrazów, które sieć widzi
  z tym samym latentem.
- **Stable Diffusion 1.5** - duży, gotowy model dyfuzyjny, który rysuje w przestrzeni latentu obrazu. **VAE** - koder i dekoder
  ściskające obraz do tego latentu i z powrotem. **UNet** - sieć w kształcie litery U, która przewiduje szum do zdjęcia.
  **CFG** - wzmacnianie warunku przez mieszanie przewidywań z warunkiem i bez. **Offset noise** - drobne stałe przesunięcie
  szumu na cały kanał, pomaga rysować bardzo ciemne i bardzo jasne obrazy. **FD-DINOv2** - odległość rozkładów cech DINOv2
  próbek i prawdziwych klatek (im mniejsza, tym bardziej próbki przypominają prawdziwe klatki).

### B.8 Infrastruktura

- **Eden** - klaster obliczeniowy uczelni; **SLURM** - jego kolejka zadań; każde zadanie ma numer (np. 1818496). **Zadanie
  tablicowe** - jedno zgłoszenie uruchamiające wiele ziaren; **afterok, afterany** - warunek startu po udanym (albo dowolnym)
  zakończeniu innego zadania. **hopper, dgx, short** - partycje (grupy węzłów) z kartami H100 (hopper) i A100 (dgx).
- **Kwota** - limit miejsca projektu na Edenie (200 GB i 200 tys. plików, sprawdzany `lfs quota -p 1299 /mnt/evafs`).
  **`/dev/shm`** - pamięć operacyjna węzła widziana jak dysk (szybki odczyt, znika po zadaniu).
- **DDP (torchrun)** - uczenie na kilku kartach naraz, każda liczy swoją część paczki. **Akumulacja gradientu** - sumowanie gradientów
  kilku mniejszych porcji przed jednym krokiem, żeby uzyskać dużą paczkę na ograniczonej pamięci.
- **SHA256** - skrót pliku; zgodne skróty po wysyłce znaczą, że plik dotarł bez zmian. **RTX 3070** - lokalna karta (8 GB).
- **Lean** - program, który mechanicznie sprawdza dowody matematyczne krok po kroku; twierdzenia „sprawdzone w Lean” nie mają luk
  w rozumowaniu, ale obowiązują tylko przy swoich założeniach. **Codex** - inny asystent programistyczny, który równolegle pisał
  część formalizacji (etap 2).


## C. Oś czasu

Czas lokalny (UTC+2). Numery zadań Eden podano dla najważniejszych kampanii; pełne listy są w opisach etapów.

| kiedy | etap | co się działo | zadania Eden | commity |
|---|---|---|---|---|
| do 20.09 | tło | RF-DETR Nano dostrajany na Edenie (kampanie Nano i Static36), dokument 36 z pomysłem Koopmana i Langevina do doboru klatek | - | - |
| 21.09, 13:27-15:00 | 1 | aparat diagnostyczny latentu, oś budżetu, chybienia według prędkości, filtr Langevina i Kalmana, oś epok, projekt macierzy etapu 1 | 1809131, 1809133, 1809135 (wstrzymane) | `e6b24173`-`97a23880` |
| 21.09, 15:00-21:00 | 1 | macierz etapu 1: 9 ramion × 3 ziarna, wyniki i rozbiór ramienia K4 | 1809161-1809166 | `27bb28d7`, `e3675b42`, `d32d274b` |
| 22.09, 12:46-16:08 | 1 | pilot „czego sieci brakuje”, widmo Koopmana wzdłuż uczenia, dobór D-optymalny, tokeny jako cząstki, kod tożsamości i ramiona K5, widmo bez symetryzacji, deep Koopman, dokument 42, część rzeczywista modów, trzy drogi do tożsamości | 1809374, 1809381, 1809407, 1809408 | `d6cbb58b`-`343af3f7` |
| 22.09, 18:23-21:02 | 1 | krystalizacja latentu, telemetria biegów, strata walidacyjna i moment końca uczenia | 1809553-1809555 | `7bb3484b`, `7c3ddecf`, `7cae6f78` |
| 23.09, 9:00-13:30 | 2 | zajętość i nasycenie przestrzeni tożsamości, 9 biegów po 250 klatek, dokumenty 43 i 44, formalizacja w Lean, pobudzenie kierunków, test obrotu i tła, sonda po każdym kroku | 1809754-1809762 | `197f2ce3`-`2e61f938` |
| 23.09, 14:58-17:30 | 2 | wektor narzędzia bez etykiet, test po obrysie, nietknięty zbiór testowy, zaskoczenie Koopmana, trzy plany (dokument 46), kierunki stałe Koopmana na hopperze, porządki na Edenie | 1809793 i dalsze | `de56106d`-`0ed75a7b` |
| 24.09 | - | przerwa | - | - |
| 25.09, 10:30-18:24 | 3 | Koopman z pamięcią, skalar tożsamości, pełna historia 5-800, pokaz FastGAN i naprawa pętli, 17 nowych klatek | 1814538 | `900cec4e`-`6360929e` |
| 25.09, 18:24-22:31 | 3 | kampania 134 klatek (top-50 na kotwicę), efekt lokalny, trudność klatek, plan 5 epok | 1814611 | `8474b310`, `84747b2d`, `9f45551c` |
| 26.09 | 3 | kampania 5 epok, ocena nowości i szans publikacji, teoria wyboru, równanie Langevina użytkownika dla latentu detektora | 1814835 | `338a40d0` |
| 27.09 | 3 | pętla 8 obrotów na Nano (pilot), analiza „wcześnie i większy model”, pętla od początku dostrajania na RF-DETR LargeDeprecated, diagnoza domknięcia | 1816922-1817676 | `12d3f023`-`e9055b2f` |
| 28.09, do 01:15 | 3 | wyniki pętli Large (6 obrotów × 3 ziarna), zapis sesji | 1817712 (brakujący punkt) | `ebd9db38` |
| 28.09, 13:52-16:30 | 4 | diagnoza porażki, odczyt dekodera, bramki G1 i G2, zwrot ku małemu ViT od zera, wersja 1 lokalnie, sprzątanie Edenu 193,2 → 52,2 GB | 1817930-1817934 (próby) | `5acea332`-`d661a8ee` |
| 28.09, 16:30-20:05 | 4 | wersja 1 na Edenie (11 ziaren), analiza wsteczna, wersja 2 (wektor na 768 neuronach), tożsamość po 21 epokach, FastGAN: diagnoza, literatura, sterowanie wzdłuż wektora | 1817935, 1817948, 1817949 | `f7efafbf`-`6f6bbf06` |
| 28.09, 21:28 - 29.09, 03:45 | 4 | wersja 3 (10 ziaren), zliczenie wygranych, iteracja 2 (20 ziaren) | 1818024, 1818025, 1818050, 1818051 | `d289a8ad`-`97186aa0` |
| 29.09, 12:54-14:52 | 4 | tożsamość na parach z narzędziem i bez, zapis wektorów, kierunki cząstkowe, plan dekodera latentu | - | `fdf82918`-`8d50ae84` |
| 29.09, 14:52-17:51 | 5 | horyzont prognozy, literatura o dyfuzji (10 prac w całości), równanie ruchu latentu (T0-T2), 10 ziaren na Edenie, macierze jak w TS; w rozgałęzieniu rozmowy: model dyfuzyjny wycinków w pikselach | 1818328, 1818331; 1818351-1818356 | `3ce17b1c`-`37a022cd` |
| 29.09 wieczór - 30.09, 02:20 | 5 | reguła „bez przyszłych klatek” i doprecyzowanie, zapis metody (dokument 23), diagnoza kolorów modelu pikselowego, sprzątanie Edenu 117 → 57 GiB, wysyłka Cataract-LMM (49 GB), plan i kod Stable Diffusion | - | `4a0c9a92`-`2bf184b0` |
| 30.09, 08:00-13:05 | 5 | próby Stable Diffusion, latenty wszystkich klatek, start pełnego uczenia | 1818490-1818496 | `056127e9`, `beed3307`, `255f6668` |
| 30.09, po 13:05 | 5 | uczenie Stable Diffusion trwa (koniec około 17:45) | 1818496 | - |

## D. Mapa eksperymentów: folder, pytanie, wynik

Wszystkie foldery leżą w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\`, chyba że podano inaczej.
Foldery oznaczone „(sesja równoległa)” prowadziły inne sesje; w tej linii prac były czytane, a nie zmieniane.

| folder | etap | pytanie | wynik jednym zdaniem | gdzie wyniki |
|---|---|---|---|---|
| `pipeline\koopman_langevin` | 1 | jak mierzyć latent detektora, widmo, tokeny, kod tożsamości | aparat pomiarowy; latent niesie położenie narzędzia po uczeniu, kod tożsamości niesie głównie wydłużenie ramki | `docs\Analiza_ponownej_publikacji\37`-`42` |
| `curated\selection_matrix_stage1` + `pipeline\koopman_langevin\selection_matrix.py` | 1 | czy sygnał Koopmana i Langevina pomaga wybierać 1500 klatek | nie: reszta Langevina +0,0027 przy progu +0,004; K4 +0,0068 nad pokryciem wyglądu, ale K4 było k-centrum kosinusowym | `docs\Analiza_ponownej_publikacji\39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md` |
| `experiments\krystalizacja_latentu` | 1-2 | kiedy latent się porządkuje; tożsamość, sonda, test | krystalizacja w 3 epokach; dołek straty walidacyjnej w epokach 6-7; wektor narzędzia AUC 0,975 bez etykiet; test mAP 0,681; zaskoczenie Koopmana AUC 0,58 | `README.md`, `wyniki\` |
| `docs\Analiza_ponownej_publikacji\43`-`46`, `formalizacja_doboru_klatek` | 2 | kiedy tożsamość jest „domknięta” i co gwarantuje poprawę | wzór domknięcia (jądro pamięci), 9 twierdzeń w Lean: gwarancja tylko przy przewadze przewidzianej poprawy nad błędem przewidywania | dokumenty 43, 44, 45, 46 |
| `experiments\koopman_kierunki_stale` | 2 | czy klatki ze zmianami w stałych kierunkach Koopmana uczą lepiej | nie: 0,6696 wobec 0,6820 losowych, przegrana na 3 z 3 ziaren | `wyniki\` |
| `experiments\wybieracz_tozsamosc_koopman`, `koopman_dynamika_uczenia`, `wlokno_pomylek` | 2 | trzy plany doboru klatek z latentu tożsamości | tylko plany, bez uruchomienia | `PLAN.md` w każdym |
| `experiments\tozsamosc_czastki_bledy` (sesja równoległa) | 2 | tożsamość cząstek a błędy detektora | kod bez uruchomienia | `PLAN.md`, `README.md` |
| `experiments\dobor_cech_adaptacyjny` (sesja równoległa) | 1-2 | GLISTER i dobór po cechach co epokę | żadne ramię nie pobiło losowania (losowe 0,7131) | `WYNIKI_KAMPANII_2026-09-22.md` |
| `experiments\koopman_gesta_historia` (sesja równoległa) | 3 | gęsta historia uczenia dla Koopmana | punkt wyjścia etapu 3; Koopmany tamtej sesji nie biły „braku zmiany” | pliki w folderze |
| `experiments\koopman_z_pamiecia` | 3 | czy stan z pamięcią [z, v] poprawia Koopmana | tak: błąd 0,0037 wobec 0,0103; próg ogólny 5 procent nieosiągnięty | `WYNIKI.md` |
| `experiments\koopman_tozsamosc_kierunki` | 3 | skalar tożsamości, FastGAN, dobór 17 i 134 klatek, 5 epok | prognoza skalara 0,272 błędu „braku zmiany”; dobór klatek = losowe; kryterium 5 epok niespełnione | `WYNIKI.md`, `NOWOSC_I_PUBLIKACJA.md`, `TEORIA_LANGEVIN.md` |
| `experiments\langevin_latent_detektora` | 3 | równanie Langevina użytkownika dla latentu | rollout 0,151 wobec 0,171, ale bramka niespełniona | `WYNIKI.md` |
| `experiments\petla_uczenia_koopman_fastgan` | 3 | pętla 8 obrotów na Nano | najlepsza pętla bez dokładania (0,9072 wobec 0,9055 i 0,9045) | `WYNIKI.md`, `ANALIZA_WCZESNIE_I_SKALA.md` |
| `experiments\petla_od_poczatku_large` | 3-4 | pętla od początku dostrajania na RF-DETR Large | losowe lepsze w 15 z 18 par; diagnoza: kręgosłup prawie się nie zmienia, prognoza gorsza od „braku zmiany” | `WYNIKI.md`, `DIAGNOZA_PORAZKI.md` |
| `experiments\domykanie_tozsamosci_large` | 3 | czy latent tożsamości Large się domyka | staje po 2 obrotach, ale niedomknięty (R około 0,78) | `PLAN.md`, `wyniki\` |
| `experiments\dekoder_sondy_large` | 4 | czy dekoder detektora mierzy postęp lepiej niż kręgosłup | rośnie z mAP w czasie (ρ 0,86), ale nie rozróżnia ramion; bramki G1 i G2 niespełnione | pliki analiz w folderze |
| `experiments\sterowanie_langevin_kalman` | 4 | pętla Langevin + Kalman + FastGAN | tylko plan | `PLAN.md` |
| `experiments\maly_vit_od_zera` | 4 | mały ViT od zera, pętla wersja 1 | 9 z 11 ziaren lepiej niż losowe (p = 0,005), ale prognoza = „brak zmiany” | `PLAN.md`, `PLAN_NASTEPNE_SPRAWDZENIE.md`, `wyniki\` |
| `experiments\maly_vit_wektor_tozsamosci` | 4 | wersja 2: wektor na 768 neuronach, start po 21 epokach | 8 z 10 (p = 0,064); prognoza wektora 0 z 6 obrotów; siła tożsamości przewidywalna o 6-10 procent | `PLAN.md`, `wyniki\` |
| `experiments\fastgan_odwrotnosc_mlp` | 4-5 | czy FastGAN trafia w przewidziany latent | pełny latent najwyżej w 36 procentach drogi; wzdłuż wektora tożsamości 12 z 12; notatka literaturowa | `PLAN.md`, `NOTATKA_LITERATURA.md`, `literatura\` |
| `experiments\maly_vit_petla_sila_tozsamosci` | 4 | wersja 3: prognoza siły tożsamości | = ramię bez prognozy; iteracja 2: szybciej niż losowe w 15 z 20 (p = 0,006) | `ALGORYTM.md`, `PLAN.md`, `wyniki\` |
| `experiments\maly_vit_tozsamosc_tla` | 4 | tożsamość na parach z narzędziem i bez; ile kierunków | wektor `nowy` d' 16,6; 1-2 kierunki z AUC ≥ 0,9 | `PLAN.md`, `wyniki\WEKTORY.npz`, `wyniki\` |
| `experiments\maly_vit_dekoder_latentu` | 4 | rysowanie obrazu wprost z latentu | tylko plan | `PLAN.md` |
| `experiments\maly_vit_horyzont_prognozy` | 5 | jak daleko naprzód działa prognoza siły | do 1000 kroków, o 10-25 procent błędu | `WYNIKI.md`, `wyniki\HORYZONT.json` |
| `experiments\maly_vit_rownanie_latentu` | 5 | równanie ruchu latentu ze sterowaniem danymi | 85 procent w trybie wyjaśnienia na 10 ziarnach | `WYNIKI_T0_T1.md`, `WYNIKI_T2.md`, `WYNIKI_EDEN.md` |
| `experiments\maly_vit_rownanie_macierze` | 5 | uczone macierze jak w TS, bez przyszłych klatek | około 40 procent; prognoza bez przyszłych klatek najlepiej 64-67 | `WYNIKI.md`, `wyniki\` |
| `docs\23_METODA_PRZEWIDYWANIA_LATENTU.md` | 5 | zapis metody | dwa tryby, równania, liczenie siły, tabela trafności | ten plik |
| `experiments\dyfuzja_narzedzie` (rozgałęzienie, potem sesja równoległa) | 5 | własny model dyfuzyjny wycinków w pikselach | 100 000 kroków; złe kolory przez start DDIM od t = 999, start od t = 979 je naprawia (poprawka niewprowadzona) | `PLAN.md`, `wyniki\eden_proba\` |
| `experiments\dyfuzja_sd_filmy` | 5 | Stable Diffusion 1.5 na wszystkich klatkach filmów | uczenie trwa; wynik w sekcji F | `PLAN.md`, `README.md` |


## Etap 1. Koopman i Langevin na latencie detektora RF-DETR: dobór klatek, widmo, kod tożsamości narzędzia, krystalizacja latentu (21-22 września 2026)

Cała praca merytoryczna tego etapu toczyła się w repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker` na gałęzi `codex/eden-nano-training-20260914`: 45 commitów od `e6b24173` (21.09, 13:27) do `7cae6f78` (22.09, 21:02). Projekt grantowy (`F:\Studia\MCHTR\Grant\Projekt`) był tylko źródłem aparatu matematycznego i nie zmieniał się merytorycznie. Poniżej kolejne kroki opisane są chronologicznie, z hashami commitów, numerami zadań na klastrze Eden i ścieżkami do plików z wynikami. Kod, dokumenty i wyniki leżą w katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`; gdy kilka plików leży w jednym folderze, pełną ścieżkę podaję przy pierwszym z nich, a dalej same nazwy.

### 1.1 Tło sprzed 21 września: detektor, dane i kampania Static36

Wszystkie pomiary tego etapu dotyczą jednego detektora obiektów, RF-DETR Nano: małej sieci, która dla pojedynczej klatki filmu zwraca prostokąt otaczający narzędzie (dalej: ramkę) oraz pewność tego wykrycia. Jej pierwsza część, tak zwany backbone (sieć zamieniająca obraz w wektory cech), to DINOv2, transformer wizyjny (ViT), który dzieli obraz na kwadratowe fragmenty i każdy opisuje wektorem (takie wektory nazywa się tokenami). Dane pochodzą z części trackingowej zbioru CATARACT-LMM: 170 archiwów z filmami operacji zaćmy, w których ramką zaznaczono jedno narzędzie, cystotom (igłę do nacięcia przedniej torebki soczewki). Filmy podzielono z ziarnem 42 na 117 grup operacji do uczenia, 25 do walidacji i 25 do testu (grupa to ostrożnie połączone nagrania, a nie potwierdzony pacjent; wyłączono dwa dokładne duplikaty nagrań). Kandydatami są co piąte klatki z dostarczoną ramką: 57 073 uczące, 12 043 walidacyjne i 15 795 testowych; zbioru testowego w całym etapie 1 nie dotykano. 14 września na klastrze obliczeniowym Eden (zadanie 1807600 w SLURM, systemie kolejkowania zadań klastra; trzy niezależne procesy na trzech kartach graficznych H100) skalibrowano przepis uczenia: wejście 384 piksele, 30 epok (epoka to jedno przejście przez cały zbiór uczący), partia 16 obrazów, obliczenia w precyzji BF16, optymalizator AdamW ze współczynnikiem uczenia 1e-4 (dla backbone'u 1e-5), rozgrzewka 2 epoki, w kalibracji bez augmentacji i bez wczesnego zatrzymania, start z wag wyuczonych wcześniej na ogólnym zbiorze zdjęć COCO, trzy ziarna losowe treningu 42, 1337 i 2024 (ziarno ustala losowość inicjalizacji i kolejności przykładów, więc powtórzenie z innym ziarnem mierzy rozrzut samego uczenia) i ten sam podzbiór 500 klatek. Miarą jakości jest AP@[0,50:0,95], czyli średnia precyzja wykrycia uśredniona po progach pokrycia ramki przewidzianej z prawdziwą od 0,50 do 0,95 (w dokumentach także „mAP"; 1 oznacza detektor bezbłędny), a AP50 to ta sama miara przy jednym progu 0,50; ostatnie wagi ziarna 42 z kalibracji osiągnęły AP50 0,880 i AP 0,638 na całej walidacji (dokument 34). 15 września ruszyła kampania Static36 (zadania 1807632, przygotowanie, i 1807634, treningi na najwyżej czterech H100): cztery statyczne selektory klatek (losowanie, klatki równomiernie rozłożone w czasie `uniform_time`, pokrycie przestrzeni cech DINOv2 `dino_coverage` oraz hybryda tego pokrycia z kotwicami czasowymi) przy trzech budżetach (500, 1500 i 2500 klatek) i trzech ziarnach treningu, razem 36 treningów po 30 epok. Średnie AP najlepszego punktu kontrolnego (punkt kontrolny to zapis wag z jednej epoki) wyniosły przy 500 klatkach od 0,6404 (hybryda) do 0,6459 (losowanie), przy 1500 od 0,6818 (losowanie) do 0,6838 (pokrycie DINOv2), a przy 2500 od 0,6899 (`uniform_time`) do 0,6983 (hybryda); wszystkie selektory mieściły się w około pół punktu AP od losowania, przy rozrzucie między ziarnami opisanym w dokumencie 36 jako około 0,002 (średnie wyliczone z plików `validation_report.json` w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\eden_static36_2026-09-15\` i zgodne z liczbami dokumentu 36). Punkt kontrolny `tracking_uniform_time_b500_s42` (plik `ut_b0500_best.pth`, SHA-256 `15e33f6d...`), czyli detektor nauczony na 500 klatkach równomiernych w czasie, stał się w etapie 1 wspólnym punktem startu, nazywanym dalej L0.

### 1.2 Punkt wyjścia zapisany 20 września i wersjonowany w pierwszym commicie etapu (dokument 36, `e6b24173`)

Pierwszy commit etapu, `e6b24173` (21.09, 13:27), wprowadził do repozytorium także pracę z wieczoru 20 września, opisaną w dokumencie `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md`. To jest stan wiedzy, z którym etap się zaczął.

**Aparat.** Katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\` dostał moduły: `tokens.py` (cechy DINOv2 dla każdego fragmentu obrazu na siatce 16 na 16, stały losowy rzut bez dopasowania), `koopman_tool.py` (rozkład DMD okna filmu na część wolną, czyli scenę, i resztę, czyli to, co się rusza), `track_dynamics.py` (Koopman i równanie Langevina na torze narzędzia `[cx, cy, log w, log h]` z prawdziwych ramek), `latent_langevin.py` (rozszerzone równanie Langevina na latencie całej klatki), `select.py` (dobór klatek rundami, zapis wariantów w formacie trenera) i `synthetic_selection_benchmark.py` (test mechanizmu na danych syntetycznych), plus testy.

Dwa pojęcia z tego aparatu wracają w całym etapie. **Operator Koopmana** to liniowy operator, który przesuwa w czasie nie sam stan układu, lecz funkcje stanu (obserwable). Jego funkcje własne, nazywane modami, to kombinacje współrzędnych, które z kroku na krok tylko mnożą się przez stałą, wartość własną λ: moduł |λ| bliski 1 oznacza wielkość trwałą (wolną), moduł bliski 0 wielkość szybko gasnącą, ujemna część rzeczywista wielkość zmieniającą znak co krok, a λ zespolone wielkość, która się obraca. DMD i EDMD to metody dopasowania skończonej macierzy przybliżającej ten operator z danych (DMD na surowych współrzędnych, EDMD na słowniku funkcji). **Rozszerzone równanie Langevina** (w dokumentach także GLE, od uogólnionego równania Langevina) w wersji z projektu grantowego, tu bez wymuszenia zewnętrznego (film z operacji nie ma wejścia znanego z góry), ma postać

    z_{t+1} = z_t + D(z_t, v_t, a_t) + JP(z_{t-1}, z_{t-2}, ...) + N(dawne reszty) + e_{t+1}

gdzie `D` to dryf zależny od bieżącego stanu, prędkości i przyspieszenia, `JP` to jądro pamięci (wpływ dawniejszych stanów), `N` przewidywalna część dawnych reszt, a `e` reszta. Wszystkie człony to regresje grzbietowe (regresja liniowa z karą za duże współczynniki, ang. ridge) dopasowane na innych grupach operacji niż oceniana klatka.

**Twierdzenie porządkujące i poprawka.** W projekcie grantowym udowodniono w Lean 4 (asystencie dowodzenia twierdzeń; plik `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\DiagnostykaInnowacji.lean`), że błąd dowolnego predyktora opartego na historii rozkłada się na zaskoczenie (część nieusuwalną, której żadna wiedza o historii nie przewidzi) i część usuwalną; sama wielkość reszty tych części nie rozróżnia. Dla pojedynczej klatki różnica błędów dwóch predyktorów zawiera jednak człon krzyżowy z zaskoczeniem tej klatki, `|z-h1|^2 - |z-h2|^2 = |m-h1|^2 - |m-h2|^2 + 2<e, h2-h1>`, i pierwsza wersja wyniku kupowała w teście syntetycznym 44 procent klatek szumowych. Obowiązująca ocena nie używa zrealizowanej następnej klatki: `reducible_gain(i) = |h_known(i) - h_pool(i)|^2 / (err_pool(i) + mediana err_pool)`, gdzie `h_pool` to model dopasowany na całej puli (na innych operacjach), a `h_known` ten sam model dopasowany tylko na przejściach w sąsiedztwie już wybranych klatek.

**Wyniki lokalne (RTX 3070 i CPU).**
- Koopman na cechach obrazu (8 filmów, 3653 klatki, krok 5 klatek) trafiał szczytem mapy ruchu w ramkę narzędzia w 35,3 procent klatek, czyli trzy razy częściej niż los (11,0), ale nie lepiej niż trywialne kontrole (mediana w czasie 36,4, średnia 35,8, różnica klatek 39,8); prędkości nie dało się odczytać (korelacja 0,02 do 0,07). Wniosek: stan narzędzia trzeba brać z ramek detektora, nie z samych cech obrazu.
- Tory narzędzia z prawdziwych ramek (170 filmów, 167 grup): równanie przewiduje od 22 do 29 procent następnego kroku względem założenia „narzędzie stoi" (krok 1 klatka: 22,5 z pamięcią; 5 klatek: 28,1; 15 klatek: 29,3), pamięć dokłada 1 do 2 punktów, człon szumu nic, a założenie stałej prędkości jest dużo gorsze (od -168,8 do -180,7), bo kroki ramki są przeciwskorelowane. Mody wolne toru (|λ| powyżej 0,98) leżą na położeniu (72 procent) i rozmiarze (24 procent), szybko gasnące na prędkości (74 procent).
- Równanie na latencie całej klatki (pula Static36: 57 073 klatki, 55 683 okna, 117 grup): umiejętność przewidywania (ang. skill, procent błędu usunięty względem modelu „nic się nie zmienia") 15,7 procent na niewidzianych operacjach. W górnych 10 procentach zaskoczenia narzędzie rusza się 17,6 piksela na krok wobec 8,2 w reszcie, a wejścia i wyjścia narzędzia są 28 razy częstsze.
- Test syntetyczny (10 ziaren, budżet 240 z 5760 klatek): dobór po wielkości reszty kupuje 74 procent klatek szumowych i szkodzi; wersja usuwalna ma błąd 0,90 i bije losowanie w 10 ziarnach z 10; weto zaskoczenia daje 0,79 do 0,80.

**Przygotowany, nieuruchomiony test.** Cztery warianty Langevina (`langevin_reducible`, `langevin_surprise`, `langevin_veto_coverage`, `langevin_veto_uniform`) po trzy ziarna, 12 biegów przy budżecie 1500; pierwsze 500 klatek identyczne z L0. Kryterium zapisane przed biegiem: wariant lepszy, gdy wszystkie trzy ziarna są powyżej średniej `uniform_time` (0,6832) i średnia przewaga przekracza 0,004. Test zablokowało miejsce na Edenie: 1,2 GB wolnego przy limicie 200 GB, a jeden bieg zapisuje około 240 MB.

Pliki: listy wariantów `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\curated\koopman_langevin\tracking_langevin_{reducible,surprise,veto_coverage,veto_uniform}_b{500,1500}_sel42.json`; dowody `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\koopman_tool_report.json`, `synthetic_selection_benchmark.json`, `track_dynamics_stride1.json`, `track_dynamics_stride5.json`, `track_dynamics_stride15.json` w tym samym katalogu.

### 1.3 Aparat diagnostyczny latentu detektora i pierwszy pomiar: oś budżetu (21.09, `e6b24173`, zadanie Eden 1809131)

**Pytanie i motywacja.** Celem sesji było przeniesienie na obrazy aparatu z projektu grantowego (cząstki latentu, rozszerzone równanie Langevina, operator Koopmana, rozkład błędu na część usuwalną i zaskoczenie) oraz odpowiedź na zarzut recenzentów poprzedniego artykułu, że było za mało ablacji i ziaren. Dokument `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md` (sekcja 0) stawia trzy pytania: czy operatorem Koopmana i równaniem Langevina da się (a) pokazać, że sieć w trakcie uczenia buduje w latencie przestrzeń stanów narzędzia (położenie, rozmiar, oś, prędkość), (b) mierzyć to co kilka epok na tych samych klatkach traktowanych jak cząstki, (c) na tej podstawie dobierać klatki do uczenia. Latent oznacza tu wektor liczb, który sieć wytwarza wewnątrz dla danej klatki; cząstka to jedna klatka śledzona jako punkt w przestrzeni latentu, raz wzdłuż filmu (zegar filmu), raz wzdłuż kolejnych punktów kontrolnych wag (zegar treningu).

**Sprostowanie, od którego zależy reszta.** RF-DETR widzi jedną klatkę, więc wektor jednej klatki nie może nieść prędkości narzędzia poza śladem rozmycia ruchu. Prędkość żyje w ciągu latentów kolejnych klatek. Każdy odczyt liczono więc w trzech wariantach: `single` (jedna klatka), `pair` (klatka plus różnica z poprzednią) i `box_pair` (to samo z samych przewidzianych ramek, bez latentu). Jeśli prędkość z `pair` nie przebija `box_pair`, latent nie wnosi nic ponad ramkę.

**Metoda.**
- Porządki na Edenie: na starcie zajęte było 198,9 GB z 200 GB, po sprzątaniu przez użytkownika 87 GB, a na koniec sesji 102 GB. Katalog `DETR` na Edenie (41 GB, w tym dubel 13 GB) porównano z lokalną kopią `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup\DETR`: 4745 plików zgodnych; dociągnięto dwie brakujące wagi (epoki 100 i 110), dłuższy dziennik i plik zdarzeń, sumy SHA-256 zgodne, starsze wersje zachowano z przyrostkiem `.stara_kopia_2026-07-22`.
- `clip_package.py`: paczka 16 gęstych klipów (12 walidacyjnych, 4 uczące, zero testowych; klipy testowe są odrzucane z kodu), po 300 klatek wziętych co 3 klatki, razem 4800 klatek, z torami narzędzia z prawdziwych ramek (dalej: tory GT, od ang. ground truth, prawda z adnotacji).
- `detector_latent.py`: dla każdego punktu kontrolnego i każdej klatki trzy wektory po 256 liczb, bez czytania etykiet: `query` (stan zapytania dekodera o najwyższym wyniku dla narzędzia; dekoder DETR ma zestaw zapytań, z których każde może stać się jednym wykryciem, i z tego wektora głowa odczytuje ramkę i klasę), `box_pool` (średnia mapy cech enkodera, czyli części sieci, która przetwarza cechy backbone'u przed dekoderem, wewnątrz przewidzianej ramki) i `global_pool` (średnia mapy cech z całej klatki), plus przewidziana ramka i jej pewność.
- `epoch_dynamics.py` liczy na każdym punkcie kontrolnym: sondy liniowe (regresja liniowa uczona na zamrożonym latencie jednych operacji i oceniana na innych, żeby sprawdzić, czy dana wielkość jest w latencie zapisana liniowo) dla położenia, rozmiaru, osi i prędkości, z kontrolą z samych ramek i sufitem z torów GT; jakość zapisuję jako R2, czyli udział zmienności wyjaśniony przez odczyt (1 idealnie, 0 nie lepiej niż średnia, wartość ujemna gorzej niż średnia). Dalej: umiejętność równania Langevina z podłogą z przetasowanych kroków; porównanie, ile stanu fizycznego niesie sześć najwolniejszych funkcji własnych Koopmana, a ile ta sama liczba pierwszych kierunków PCA (analiza składowych głównych, czyli rozkład na kierunki największej zmienności); participation ratio (efektywna liczba używanych kierunków, `(Σ s_i)^2 / Σ s_i^2` z wartości własnych kowariancji). Między kolejnymi punktami kontrolnymi: przesunięcie każdej cząstki po najlepszym obrocie całej chmury (analiza Procrustesa) i CKA (miara podobieństwa geometrii dwóch chmur punktów od 0 do 1, niezależna od obrotu i skali). Przesunięcia na klatkę trafiają do plików `*.displacement.npz`.
- Trener dostał opcję `--keep-epoch-checkpoints N` (zachowuje same wagi co N epok; domyślne zachowanie bez zmian) w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\train\train_rfdetr.py` i `run_with_validation.py`; zadania Edenu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_detector_latent.slurm` i `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_epoch_series.slurm`.
- Wagi w pierwszym pomiarze: COCO (przed uczeniem na narzędziu) oraz `uniform_time`, ziarno 42, po 500, 1500 i 2500 klatkach (`best.pth` z kampanii Static36). To jest oś budżetu, nie oś epok. Zadanie Eden 1809131.

**Kryteria zapisane przed wynikiem** (dokument 38, sekcja 4). Hipoteza krystalizacji potwierdzona, jeśli na osi treningu: (1) odczyt położenia i osi z `query` rośnie razem z pokryciem ramek; (2) odczyt prędkości z jednej klatki zostaje poniżej 0,15 na każdym punkcie, a z pary rośnie i na końcu przebija `box_pair` o co najmniej 0,05; (3) umiejętność Langevina z pamięcią rośnie i jest co najmniej 10 punktów nad podłogą z przetasowania; (4) wolne mody Koopmana niosą położenie lepiej niż PCA na wytrenowanym modelu, a nie na COCO. Hipoteza obalona w całości, jeśli nie zachodzą punkty 2 i 4. Dla doboru klatek obowiązywało kryterium z dokumentu 36.

**Wynik** (plik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_budget_axis.json`, kopia robocza w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\REPORT_budget_axis.json`). IoU to średni iloraz pola części wspólnej do pola sumy ramki przewidzianej i prawdziwej na 4800 klatkach; liczby Langevina to procent kroku przewidziany ponad „nic się nie zmienia", w nawiasie podłoga z przetasowania.

| Wagi | IoU | położenie `query` / `box_pool` (R2) | rozmiar | oś | Langevin z pamięcią (podłoga) | zysk pamięci | położenie: wolne mody Koopmana / PCA (`box_pool`) |
|---|---|---|---|---|---|---|---|
| COCO | 0,088 | -0,01 / -0,10 | 0,00 | -0,06 | 31,3 (-1,5) | +4,6 | -0,32 / -0,03 |
| 500 | 0,833 | 0,68 / 0,86 | 0,78 | 0,68 | 22,5 (-1,6) | +4,1 | 0,84 / 0,04 |
| 1500 | 0,839 | 0,76 / 0,83 | 0,80 | 0,73 | 21,5 (-1,6) | +3,6 | 0,80 / 0,17 |
| 2500 | 0,847 | 0,69 / 0,83 | 0,82 | 0,62 | 20,7 (-1,7) | +4,1 | 0,87 / 0,19 |

Ruch cząstek między wagami: COCO do 500 przesunięcie 1,36 i CKA 0,07 dla `query`; 500 do 1500 i 1500 do 2500 przesunięcie około 0,25 i CKA około 0,88; `global_pool` prawie stoi (CKA od 0,85 do 0,96). Z pliku JSON dodatkowo: kontrola z samej ramki detektora czyta położenie z R2 od 0,94 do 0,97, więcej niż latent; participation ratio bloku `box_pool` spada z 7,37 (COCO) do wartości od 5,67 do 6,10 po uczeniu.

**Odczyt względem kryteriów.**
1. Spełnione. Przed uczeniem położenia, rozmiaru i osi nie da się odczytać (R2 około 0), po uczeniu R2 od 0,6 do 0,86. Uczenie przebudowuje latent obszaru narzędzia, a latent całej sceny zostawia w spokoju.
2. Niespełnione i w tej postaci nierozstrzygalne. Prędkości nie czyta żaden wariant: jedna klatka 0,00, para od 0,00 do 0,03, same ramki poniżej zera. Powód leży w celu: 1 procent klatek niesie 58 procent energii prędkości (skoki adnotacji, powrót narzędzia w kadr), a nawet z prawdziwych torów para daje tylko 0,39 według tekstu dokumentu 38 (w pliku JSON 0,35), po obcięciu skoków 0,18. Cel `velocity_trimmed` i sufit z torów GT dodano po pierwszym odczycie; dokument zapisuje to jawnie jako odstępstwo od kryteriów.
3. Niespełnione, kierunek odwrotny. Przewidywalność kroku latentu spada z uczeniem (z 31 do 21 procent) przy podłodze około -1,5, więc to realna dynamika. Latent COCO opisuje wolno zmienną scenę, latent po uczeniu idzie za narzędziem, a ruch narzędzia jest przewidywalny w 22 do 29 procent. Latent dziedziczy przewidywalność obiektu, który śledzi; kryterium 3 było źle postawione, bo miarą krystalizacji nie jest wzrost przewidywalności, tylko jej zbieżność do przewidywalności toru fizycznego.
4. Spełnione dla położenia. Sześć najwolniejszych funkcji własnych Koopmana niesie położenie z R2 od 0,73 do 0,87, ta sama liczba kierunków PCA od 0,04 do 0,45; na COCO obie wartości są około zera. Dla rozmiaru przewagi nie ma.

Między 500 a 2500 klatek prawie nic się nie zmienia (IoU z 0,833 na 0,847), zgodnie z kampanią Static36.

**Przegląd literatury** (dwa podagenty z dostępem do baz Undermind i Scite, 21 prac zweryfikowanych w Scite; dokument 38, sekcja 2). Prędkość i kierunek ruchu są odczytywalne tylko z modeli wideo (Garrido i in. 2025, V-JEPA; Joseph i in. 2026), a Musa i in. 2026 częściowo temu przeczą (lekkie sondy blisko losu). Koopman na wideo i w JEPA jest zawsze uczony razem z siecią albo pracuje w pikselach (Comas, Sznaier, Camps 2021; Ruiz-Morales i in. 2025, gdzie minimum straty JEPA to funkcje własne Koopmana o wartości 1, tylko na danych syntetycznych); nikt nie dopasował Koopmana do latentu gotowego detektora. Koopman i Langevin opisują w literaturze raczej wagi w czasie uczenia (Redman i in. 2024; Kim i in. 2026; Kunin i in. 2023; Simon i in. 2023: uczenie samonadzorowane idzie schodkami). Dobór danych: rozdział błędu na usuwalny i nieusuwalny to koncepcyjnie RHO-loss (Mindermann i in. 2022); wybór według straty kupuje szum (Hu i in. 2021) i zbiorowe przykłady odstające (Karamcheti i in. 2021); przy augmentacji zysk z aktywnego doboru prawie znika (Mittal i in. 2025). Mori-Zwanzig z danych istnieje dla dynamiki molekularnej (Lin i in. 2021; Liu i in. 2025, MEMnets). Za nowe uznano połączenie: Koopman i równanie z pamięcią dopasowane do latentu gotowej sieci i śledzone między punktami kontrolnymi; dokument uczciwie zaznacza, że każdy składnik z osobna istnieje.

### 1.4 Gdzie dobór ma co poprawiać: chybienia według prędkości narzędzia i wstrzymany test 12 biegów (21.09, `631208a3`)

**Pomiar.** Te same klatki walidacyjne paczki (3458 klatek z 12 klipów) podzielono według prędkości środka prawdziwej ramki na krok 3 klatek. Chybienie oznacza ramkę o IoU poniżej 0,5 (dokument 38, sekcja 4b):

| Wagi | wolne 50 procent | od 50 do 80 | od 80 do 95 | najszybsze 5 procent (powyżej 14 px na krok) |
|---|---|---|---|---|
| 500 klatek | 3,5 procent | 3,7 | 5,0 | 13,9 |
| 1500 klatek | 4,5 | 4,1 | 6,0 | 13,3 |
| 2500 klatek | 2,9 | 3,5 | 5,2 | 14,5 |

Detektor chybia w szybkim ruchu cztery do pięciu razy częściej niż w wolnym, a pięć razy więcej zwykłych, równo rozłożonych klatek tego nie naprawia; średnie AP tego nie widzi, bo szybkie klatki to 5 procent zbioru. Zastrzeżenie: część najszybszych klatek to skoki adnotacji i powrót narzędzia w kadr. Wniosek na doborze: obok AP mierzyć udział chybień w najszybszych 5 i 20 procentach klatek.

**Test 12 biegów i decyzja użytkownika.** Ten sam commit dodał `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_langevin_selection.slurm` i zgłoszono test czterech wariantów Langevina z dokumentu 36 (zadanie Eden 1809135) z kryterium dopisanym przed wynikiem: `langevin_surprise` (kupujący klatki zaskoczenia, 28 razy częściej niż losowanie) obniża udział chybień w najszybszych 5 procentach o co najmniej 3 punkty procentowe względem `uniform_time` na wszystkich trzech ziarnach, bez spadku AP poniżej 0,6832 minus 0,004. Użytkownik przerwał uruchamianie generatora ramion słowami: „Najpierw mi napisz tutaj, co będziesz robił i jak to będziesz robił. Jeżeli mi się to spodoba i to zatwierdzę, to wtedy puścisz trening na Eden" (zapis w pamięci `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-plan-przed-treningiem.md`). Od tej chwili obowiązuje reguła: kampanie i nowe ramiona najpierw opisać i czekać na zgodę. Plan został przedstawiony i zatwierdzony, test 1809135 skasowano, a zamiast niego powstała kontrolowana macierz dziewięciu ramion (sekcja 1.8). Dokument 39 odnotowuje, że autor zniósł dla tego etapu zasadę jednego zadania naraz.

### 1.5 Prędkość jako stan ukryty: okno czasu i odporny filtr Langevina (21.09, `1f821f38`)

**Pytanie.** Skoro jedna klatka nie niesie prędkości, czy prędkość da się wydobyć z ciągu latentów i jak dobrze?

**Okno czasu** (`latent_velocity.py`, wynik `REPORT_velocity.json`). Miarą jest korelacja z nachyleniem prawdziwego toru (okno plus minus 2 kroki, 1 procent skoków odcięty). Nachylenie odczytu położenia z latentu `box_pool` po plus minus 1, 2, 4 i 8 krokach: korelacja 0,33, 0,46, 0,57 i 0,62 (wagi 2500); z samych ramek detektora 0,55, 0,68, 0,79 i 0,87; na wagach COCO około 0,03. Prędkość jest w ciągu latentów, ale dopiero na dłuższej podstawie czasu; różnica dwóch klatek tonie w szumie odczytu.

**Filtr** (`kalman_state.py`). Na każdą oś obrazu model `p' = p + v`, `v' = f_v v + a`, `a' = f_a a + w`, `y = p + n`: położenie, prędkość i przyspieszenie z kolorowym wymuszeniem (przyspieszenie to ręka chirurga, której nie widać) i szumem pomiaru. To jest filtr Kalmana (algorytm, który na bieżąco łączy przewidywanie modelu ruchu z zaszumionym pomiarem) zbudowany na rozszerzonym równaniu Langevina. Cztery liczby dopasowano metodą największej wiarygodności na innych operacjach. Pierwsza, gaussowska wersja zawiodła (korelacja od 0,10 do 0,31, gorzej niż zwykła różnica), bo innowacje (różnice między pomiarem a przewidywaniem filtru) mają nadwyżkową kurtozę od 18 do 65 (kurtoza mierzy grubość ogonów rozkładu; dla rozkładu normalnego nadwyżka wynosi 0), więc dopasowanie uciekło w „brak bezwładności, szum 18 px". Wersja odporna używa straty Hubera (kwadratowej dla małych błędów, liniowej dla dużych) i obcina innowacje na 2,5 odchylenia (metoda Masreliez):

| Wagi | Źródło położenia | różnica plus minus 2: korelacja (R2) | filtr przyczynowy | wygładzacz | szum odczytu |
|---|---|---|---|---|---|
| COCO | oba | 0,00 | 0,00 | 0,01 | od 38 do 77 px |
| 500 | latent `box_pool` | 0,37 (-2,89) | 0,17 (+0,01) | 0,42 (-0,61) | 12,0 px |
| 500 | ramka detektora | 0,54 (-2,07) | 0,49 (+0,22) | 0,64 (-0,63) | 2,1 px |
| 2500 | latent `box_pool` | 0,46 (-1,88) | 0,31 (+0,04) | 0,48 (-0,52) | 11,4 px |
| 2500 | ramka detektora | 0,68 (-0,53) | 0,67 (+0,44) | 0,76 (+0,22) | 3,2 px |

Dopasowane parametry dla wag 2500 (plik `REPORT_kalman_ut_b2500_best.json`): dla ramki detektora `f_v` 0,38, `f_a` 0,13, odchylenie wymuszenia 7,4 px, szum pomiaru 3,2 px; dla latentu `f_v` 0,08, `f_a` 0,004, wymuszenie 8,9 px, szum 11,4 px.

**Odczyt.** (1) Filtr zamienia zaszumione położenie w użyteczną prędkość (R2 dodatnie tam, gdzie różnica ma ujemne). (2) Jakość tej prędkości rośnie z uczeniem detektora (korelacja z 0,49 na 0,67), więc nadaje się na miarę krystalizacji stanu. (3) Latent przegrywa z własną ramką detektora: liniowy odczyt położenia z latentu ma błąd 11 px, ramka 3 px; na tym etapie latent nie wnosi o ruchu nic ponad ramkę. (4) Innowacje nie są białe (korelacja z poprzednią od 0,17 do 0,20) i mają bardzo grube ogony (kurtoza 25 dla latentu i 132 dla ramki): wymuszenie przychodzi rzadkimi skokami, jak zakłada HAVOK (model liniowy we współrzędnych opóźnień plus rzadkie wymuszenie; Brunton i in. 2017). (5) Rozmiar innowacji wskazuje chybienia detektora z AUC od 0,60 do 0,68, a pewność samego detektora z AUC od 0,81 do 0,83 (AUC to pole pod krzywą ROC: prawdopodobieństwo, że losowa klatka z błędem dostanie wyższą ocenę niż losowa klatka bez błędu; 0,5 oznacza los, 1 idealne rozróżnienie). Jako samodzielny wykrywacz błędów innowacja przegrywa z pewnością detektora (przy udziale chybień 3,3 procent na wagach 2500).

**Co literatura pozwala obiecać autoenkoderowi Koopmana** (dokument 38, sekcja 4d; autoenkoder Koopmana to sieć, która uczy się współrzędnych, w których krok czasu jest mnożeniem przez jedną macierz, oraz dekodera wracającego do opisu). Lusch, Kutz, Brunton 2018: bez ograniczeń sieć nie zbiega do jednoznacznego, interpretowalnego rozwiązania, więc osie „położenie, prędkość, przyspieszenie" same się nie wyłonią. Proctor, Brunton, Kutz 2018: wymuszenie zewnętrzne psuje mody autonomicznego Koopmana. Wanner i Mezić 2022: zwykły DMD jest obciążony przy zaszumionych obserwablach. Becker i in. 2019 (Recurrent Kalman Networks): z jednego obrazu nie da się uzyskać prędkości, trafia ona do pamięciowej części latentu. Zaproponowana konstrukcja: pierwsze współrzędne to zamrożona liniowa głowica położenia, rozmiaru i osi, reszta to uczone obserwable na oknie opóźnień, pamięć Mori-Zwanziga dopiero na resztach po filtrze. Próg do pobicia: filtr z ramek (korelacja 0,67, R2 0,44) i z latentu (0,31 i 0,04). Autoenkoder nie był w tym etapie uczony na latencie klatki.

Pliki: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_velocity.json`, `REPORT_kalman_coco.json`, `REPORT_kalman_ut_b0500_best.json`, `REPORT_kalman_ut_b2500_best.json`; kod `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\latent_velocity.py`, `kalman_state.py`, test `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\tests\test_kalman_state.py`.

### 1.6 Status pilota wobec planu 37 (21.09, `31c0b31d`)

Tego samego dnia inna sesja napisała w tym samym repozytorium plan `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md` (plik niewersjonowany w git). Plan stawia hipotezy H1 (historia reprezentacji narzędzia poprawia przewidywanie jego stanu ponad elastyczny model bieżącego stanu), H2 (dynamika poprawia wykrywanie błędów detektora ponad proste sygnały), H3 (dynamika poprawia dobór klatek) i H4 (kontrolowane przekształcenia obrazu), etapy od A (poprawność metody) do E, macierz modeli od M0 (trwałość) do M5, progi praktyczne (H1: +1 punkt umiejętności M4 minus M2; H2: +0,02 AUPRC, czyli pola pod krzywą precyzja-czułość; H3: +0,004 AP) oraz bootstrap po grupach operacji z 2000 sparowanych replik i ziarnem 20260921. Commit `31c0b31d` dopisał do dokumentu 38 sekcję 4e: pomiary tej sesji mają status pilota eksploracyjnego, a przy rozbieżności kryteriów wiąże plan 37.

Zarzuty planu, które trafiają w użyty kod: (A1) `LatentLangevin` standaryzuje latent i skalę przyrostów na całej puli przed podziałem na grupy, a PCA w `epoch_dynamics.py` też liczono na wszystkich klatkach, więc liczby „Langevin 21 do 31 procent" wymagają powtórzenia; (A3) `reducible_gain` dzieli przez błąd bieżącego celu, więc jest wynikiem retrospektywnym, nie prognozą; (7.1) „zysk pamięci +4 punkty" to różnica między historią 3 i 8 kroków, a model krótki już używa prędkości i przyspieszenia, więc nie jest to kontrola „bez historii"; właściwy kontrast to M4 minus M2; (8.1) numer zapytania dekodera nie jest trwałą tożsamością obiektu, więc wnioski opierać na `box_pool`; (5.1) pomiary torów z dokumentu 36 czytały wszystkie 167 grup, w tym testowe (paczka 16 klipów testowych nie zawiera); (9) test 12 biegów nie jest etapem D planu, bo nie ma ramion niepewności detektora ani modelu bez historii. Pilot wnosił do planu pięć rzeczy: latent całej sceny nie niesie stanu narzędzia, a latent obszaru narzędzia niesie go dopiero po uczeniu; oś epok i przesunięcie cząstek; prędkość z filtru odpornego na skoki (testy reszt muszą zakładać grube ogony); warstwę oceny według prędkości narzędzia; wstępny sygnał, że innowacja przegrywa z pewnością detektora.

### 1.7 Oś epok: jeden trening z wagami co 3 epoki (21.09, `8f37dd7a`, zadanie Eden 1809133)

**Metoda.** Jeden trening według przepisu kampanii: `uniform_time`, 1500 klatek, ziarno 42, 30 epok, z zachowaniem wag co 3 epoki (punkty po epokach 2, 5, 8, ..., 29) i odczytem latentu każdego punktu na tych samych 4800 klatkach paczki (zadanie 1809133, partycja hopper). Bieg ma w dalszej części etapu nazwę `epochs_uniform_time_b1500_s42`. Wynik: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_epoch_axis.json`; latenty `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\latents_epochs\LATENT_epoch_002.npz` do `LATENT_epoch_029.npz` i `LATENT_ep000_coco.npz`; krzywa jakości `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\epoch_val_map.json`.

| Epoka | IoU | położenie (R2) | oś | wolne mody Koopmana / PCA | Langevin | przesunięcie cząstek od poprzedniego punktu (CKA) | AP walidacji |
|---|---|---|---|---|---|---|---|
| COCO | 0,088 | -0,10 | -0,03 | -0,32 / -0,03 | 29,6 | | |
| 2 | 0,832 | 0,87 | 0,65 | 0,83 / 0,10 | 22,4 | 1,29 (0,09) | 0,665 |
| 5 | 0,841 | 0,85 | 0,67 | 0,81 / 0,20 | 21,9 | 0,13 (0,94) | 0,687 |
| 14 | 0,834 | 0,78 | 0,64 | 0,77 / 0,20 | 22,2 | 0,12 (0,93) | |
| 29 | 0,836 | 0,86 | 0,73 | 0,84 / 0,24 | 22,2 | 0,09 (0,95) | 0,656 |

W pełnej serii z pliku JSON przesunięcie między kolejnymi punktami po epoce 2 wynosi od 0,09 do 0,13 przy CKA od 0,92 do 0,95, a participation ratio `box_pool` rośnie powoli z 5,40 (epoka 2) do 7,02 (epoka 29).

**Odczyt.** (1) Cała przebudowa latentu dzieje się w pierwszych trzech epokach; od epoki 2 odczyt położenia i osi, przewaga modów Koopmana nad PCA i przewidywalność kroku stoją w miejscu. Punkty co 10 do 20 epok niczego by nie pokazały; żeby zobaczyć samo układanie się stanu, trzeba zapisywać wagi wewnątrz epok od 0 do 3. (2) AP walidacji rośnie do epoki 5 (0,687) i spada do 0,656 w epoce 29: przy 1500 klatkach 30 epok to przeuczenie (przeuczenie oznacza spadek jakości na walidacji przy dalszym uczeniu, gdy sieć dopasowuje się do szczegółów zbioru uczącego). Cząstki dalej dryfują (około 0,1 na 3 epoki) bez poprawy odczytu stanu, więc ten dryf nie jest nauką o narzędziu. (3) Dla miary szybkości uczenia liczą się epoki od 0 do 5; późniejsze różnice między ramionami mierzą głównie odporność na przeuczenie, więc trzeba raportować zarówno najlepszy punkt kontrolny (`best_val`), jak i ostatnią epokę (`last_epoch`).

### 1.8 Macierz etapu 1: kontrolowany eksperyment doboru klatek, projekt i przygotowanie (21.09, `f3cbbf3a`, `c8fdd6df`, `97a23880`)

Uwaga o nazwie: „etap 1" w dokumentach repozytorium oznacza pierwszą rundę tej macierzy doboru klatek; w tej części nazywam ją „macierzą etapu 1", żeby nie myliła się z całym Etapem 1 opisu.

**Pytanie.** Czy sygnał Koopmana i Langevina pomaga wybierać klatki do uczenia detektora, przy równym budżecie i jednej recepturze treningu? Projekt, kryteria i wyniki są w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md`. Macierz odpowiada na uwagi recenzentów z mapy `04_REVIEWER_MAP`: A6 (powtarzane losowanie przy tym samym budżecie, ramię B1), A5 i B4 (pełna macierz ablacji przy stałych warunkach: dziewięć ramion, jedna receptura, wspólny start), A9 i B2 (powtórzenia i niepewność: w macierzy etapu 1 trzy ziarna, w etapie 2 pięć ziaren treningu razy trzy ziarna wyboru, bootstrap po operacjach), A2 i B1 (podział po operacjach 117/25/25, test otwierany raz), A13 i A14 (czułość parametrów i alternatywne selektory, etap 3).

**Wspólne warunki.**
- Start L0: 500 klatek `tracking_uniform_time_b500_s42`, identyczny dla wszystkich ramion; każde ramię dokłada 1000 klatek, więc każdy trening widzi 1500.
- Receptura kampanii Static36: RF-DETR Nano, wagi COCO, wejście 384, 30 epok, profil `nano_static_pilot`, protokół `protocol_tracking_calibration.yaml`, to samo archiwum danych `tracking_static36.tar` (sumy SHA-256 w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\queue\stage1.inputs.sha256`).
- Ziarna treningu 42, 1337, 2024; jedno ziarno wyboru 42.
- Wybór nie czyta etykiet. Modele Langevina dopasowano na nieoznaczonej puli uczącej transdukcyjnie (czyli na tych samych nieoznaczonych danych, z których potem się wybiera), co zapisano jawnie. Ocena na walidacji (12 043 klatki); zbioru testowego nie dotykano.
- Pula doboru ma 10 637 klatek, minimalny odstęp między wybranymi klatkami 5 klatek. Z pliku `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\eden_static36\data\tracking_static36\data_manifest.json` (pole `materialized_train_union`) i kodu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\tools\prepare_tracking_static_pilot.py` wynika, że te 10 637 klatek to suma (unia) klatek wybranych przez cztery selektory Static36 przy trzech budżetach, spośród 57 072 unikalnych kandydatów, a nie wszystkie klatki uczące. Dokumenty etapu tego nie mówią wprost; to mój odczyt ze źródeł.
- Modele Langevina (K1 i K2) dopasowano na pamięci podręcznej cech DINOv2 wszystkich 57 073 klatek uczących: 54 576 okien, wymiar 24, umiejętność kroku z historią 18,39 procent (współczynnik regularyzacji w trzech foldach, czyli trzech częściach podziału po grupach operacji, na których na zmianę dopasowuje się i ocenia model: 1,0, 0,1 i 1,0), bez historii 4,57 procent (`MATRIX_MANIFEST.json`). Według audytu z dokumentu 40 (sekcja 1.23) latent K1 to znormalizowany wektor CLS DINOv2 (dodatkowy token podsumowujący cały obraz) sklejony ze znormalizowanym maksimum po tokenach przestrzennych, rzutowany losowo (nie PCA) do 24 wymiarów; model ma dryf zależny od stanu, różnice pierwszego i drugiego rzędu, losowe cechy nieliniowe (RFF, losowe cechy Fouriera `cos(Wx + b)` z losowym `W`, tanie przybliżenie funkcji nieliniowej) i historię 8 kroków; ocena K1 to błąd przewidywania poza grupą dopasowania. Nie było osobnego wymuszenia `W`, korektora dawnych reszt `N` ani członu zdarzeń `Z` z równania grantowego.

**Ramiona.**

| ID | Jak dobiera 1000 klatek | Po co |
|---|---|---|
| B0_consecutive | ciągi sąsiednich klatek od losowego miejsca losowego filmu | punkt odniesienia autora: klatki po kolei, jak przy ręcznym oznaczaniu filmu |
| B1_random | losowo | żądanie recenzenta A6 |
| B2_uniform_time | równomiernie w czasie | kontrola nadmiarowości sąsiednich klatek |
| B3_coverage | k-centrum w przestrzeni cech DINOv2 (zachłannie zawsze klatka najdalsza od już wybranych, co zmniejsza największą lukę pokrycia) | mocny selektor bez dynamiki |
| B4_uncertainty | B3 zmieszane pół na pół z oceną 1 minus pewność detektora L0 (maksymalny wynik narzędzia na klatce) | standard aktywnego uczenia |
| K1_langevin | B3 pół na pół z resztą rozszerzonego równania Langevina z historią 8 kroków | propozycja |
| K2_no_history | to samo, model widzi tylko bieżący latent, ta sama liczba cech nieliniowych | ablacja historii |
| K3_placebo | to samo, oceny K1 przetasowane między klatkami tego samego filmu | placebo: ta sama procedura z sygnałem pozbawionym treści |
| K4_koopman_cover | pokrycie tą samą funkcją k-centrum co B3, ale we współrzędnych 8 najwolniejszych funkcji własnych operatora jednego kroku na `[z_t, z_{t-1}]` latentu K1 (części rzeczywiste i urojone, bez kolumn stałych, ustandaryzowane) | pokrycie przestrzeni stanów |

B4, K1, K2 i K3 przechodzą przez tę samą funkcję `select_batch` z wagami 0,5 dla pokrycia i 0,5 dla oceny pomocniczej; ocena pomocnicza jest zamieniana na rangę percentylową wewnątrz grupy operacji, żeby żadne ramię nie wygrało skalą oceny.

**Kontrasty i kryteria zapisane przed wynikiem.**

| Kontrast | Pytanie | Próg |
|---|---|---|
| K1 minus B3 | czy dynamika dodaje coś ponad wygląd | +0,004 AP |
| K1 minus B4 | czy bije zwykłe aktywne uczenie | +0,004 AP |
| K1 minus K2 | czy zasługa należy do historii | +0,004 AP |
| K1 minus K3 | czy działa sygnał, a nie samo domieszanie drugiego kryterium | +0,004 AP |
| K4 minus B3 | czy pokrycie przestrzeni Koopmana bije pokrycie wyglądu | +0,004 AP |
| każde minus B0 | zysk nad klatkami po kolei | opisowo |

Miara główna: AP@[0,50:0,95] na walidacji z punktu `last_epoch` (ostatnia epoka), `best_val` osobno. Miara druga: udział chybień (IoU poniżej 0,5) w najszybszych 5 i 20 procentach klatek walidacji według prędkości z torów GT, próg: spadek o 3 punkty procentowe. Miara trzecia: AP po każdej epoce, czyli szybkość uczenia. Efekt dodatni wymaga oszacowania punktowego nad progiem oraz 95-procentowego przedziału sparowanej różnicy powyżej zera, z bootstrapem po całych operacjach (bootstrap to wielokrotne losowanie operacji ze zwracaniem w celu oszacowania niepewności; 2000 replik, ziarno 20260921) i korektą Holma dla czterech kontrastów K1 (poprawka progu przy wielu porównaniach naraz). Teza „Langevin pomaga" wymaga wszystkich czterech kontrastów K1. Przy trzech ziarnach macierz etapu 1 jest pilotem: rozstrzyga, czy iść dalej, nie zamyka tezy.

**Plan dalszych etapów.** Etap 2, tylko gdy któryś kontrast K przekroczy próg punktowo: B1, B2, B3, B4, zwycięskie ramię K, K2 i K3; pięć ziaren treningu razy trzy ziarna wyboru; budżety 750, 1000, 1500 i 2500; na końcu jednorazowa ocena na teście. Etap 3: czułość zwycięskiego ramienia (waga domieszki 0,25, 0,5, 0,75; historia 2, 4, 8 kroków; latent DINOv2 wobec latentu detektora; strata odporna na skoki).

**Wykonanie i awaria startu.** Plan przewidywał przygotowanie na jednej karcie A100 (zadanie 1809149), a potem pięciu robotników po jednej karcie, biorących biegi ze wspólnej kolejki: cztery na A100 (1809150 do 1809153) i jeden na H100 (1809154), szacunek około 8 godzin. Pierwsze dwa zgłoszenia przygotowania (1809149 i 1809155) padły po 5 sekundach: gdy `pool_confidence.py` szedł jako skrypt, jego katalog trafiał na ścieżkę importu i plik `koopman_langevin\select.py` przesłaniał moduł `select` biblioteki standardowej, co psuło `import socket` wewnątrz pakietu rfdetr. Zadania zależne anulowały się same, żaden trening nie ruszył. Poprawka w `c8fdd6df`: flaga `-P` interpretera. Trzecie zgłoszenie, 1809161, przeszło w 12 minut na A100 (partycja short): policzyło pewność L0 na 10 637 klatkach puli (`uncertainty_L0.json`, definicja „1 minus maksymalny wynik narzędzia detektora L0", bez etykiet), zbudowało dziewięć ramion i wspólną kolejkę 27 biegów. Pięciu robotników 1809162 do 1809166 (węzły dgx-2 i dgx-4 z kartami A100 oraz hopper z H100; po jednej karcie, 8 rdzeniach i 96 GB pamięci) brało biegi z tej kolejki.

**Stan po przygotowaniu (`97a23880`).** Część wspólna dobranych 1000 klatek z ramieniem K1:

| B0 | B1 | B2 | B3 | B4 | K2 | K3 | K4 |
|---|---|---|---|---|---|---|---|
| 0,11 | 0,11 | 0,10 | 0,46 | 0,33 | 0,83 | 0,26 | 0,17 |

Równanie z historią przewiduje 18,4 procent kroku latentu, bez historii 4,6 procent, a mimo to K1 i K2 wybierają w 83 procentach te same klatki: oba modele uznają za trudne te same miejsca filmu. Zapisano przed wynikiem, że kontrast K1 minus K2 ma z góry małą moc i prawie na pewno wyjdzie nierozstrzygający; rozstrzygające zostają K1 minus B3, K1 minus B4 i K1 minus K3. Uczciwe badanie historii wymagałoby oceny różniącej oba modele, na przykład zysku z historii na klatkę.

Pliki: kod `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\selection_matrix.py`, `pool_confidence.py`, `latent_langevin.py` (dodane ramię bez historii); zadania `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_selection_stage1_prep.slurm`, `run_selection_stage1_worker.slurm`, `build_stage1_queue.py`, `run_selection_stage1.slurm` (wariant na jednym węźle z czterema kartami, nieużyty); listy klatek wszystkich ramion `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\curated\selection_matrix_stage1\matrix_{B0..K4}_*_b1500_sel42.json`, `MATRIX_MANIFEST.json` i `uncertainty_L0.json` (wersjonowane w `e3675b42`); dowody `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\matrix_stage1\variants\MATRIX_MANIFEST.json`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\matrix_stage1\uncertainty_L0.meta.json`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\queue\stage1.inputs.sha256`. Na Edenie katalog `~/smallvit_runs/2026-09-21_koopman_langevin/` (podkatalogi `matrix_stage1`, `queue/stage1`, `results_stage1`).

### 1.9 Macierz etapu 1: wyniki 27 treningów i diagnoza, dlaczego reszta Langevina nie pomogła (21.09, `27bb28d7`, `e3675b42`)

**Przebieg.** O 19:45 skończone były 24 z 27 biegów po 4 godzinach 48 minutach (commit `27bb28d7` o 19:47 z wynikiem ośmiu ramion); po dokończeniu K4 commit `e3675b42` o 20:31 zamknął macierz: 27 z 27 biegów, 0 błędów, całość około 5 godzin. Pojedynczy bieg z oceną trwał od 3049 do 4398 sekund (pole `total_seconds` w `validation_report.json`). Druga miara (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\misses_by_speed.py`) liczy chybienia na 11 940 klatkach walidacji z prędkością z torów GT, progi prędkości 4,6, 9,3 i 19,7 piksela na 3 klatki.

**Tabela główna.** AP to wynik niezależnej oceny zapisanego punktu kontrolnego na całej walidacji, średnia plus minus odchylenie standardowe (populacyjne) z trzech ziaren treningu. „Najlepsze epoki" podane w kolejności ziaren 1337, 2024, 42. Udział chybień w procentach, średnia z trzech ziaren. Wartości oznaczone gwiazdką wyliczyłem z plików źródłowych, bo w dokumencie 39 ich brak; kolumny chybień od 50 do 95 procent i dla najszybszych 20 procent pochodzą z plików JSON.

| Ramię | AP najlepszy punkt | AP ostatnia epoka | średnie AP epok 0-5 | najlepsze epoki | chybienia: wolna połowa | 50-80 | 80-95 | najszybsze 5 | najszybsze 20 |
|---|---|---|---|---|---|---|---|---|---|
| B0_consecutive | 0,6672 ± 0,0030 | 0,6500 ± 0,0031 | 0,6120 | 16, 16, 15 | 6,94 | 7,50 | 11,59 | 35,96 | 17,68 |
| B1_random | 0,6822 ± 0,0011 | 0,6536 ± 0,0107 | 0,6251 | 6, 8, 8 | 6,18 | 6,82 | 10,62 | 33,00 | 16,21 |
| B2_uniform_time | **0,6866** ± 0,0042 | 0,6484 ± 0,0020 | 0,6261 | 7, 7, 7 | 6,17 | 6,88 | 10,73 | 31,77 | 15,99 |
| B3_coverage | 0,6817 ± 0,0013 | 0,6439 ± 0,0023 | 0,6246 | 6, 6, 7 | 7,03 | 7,57 | 11,22 | 32,21 | 16,46 |
| B4_uncertainty | 0,6826 ± 0,0005 | 0,6438 ± 0,0028 | 0,6270 | 8, 6, 9 | 6,36 | 6,95 | 10,83 | 32,33 | 16,20 |
| K1_langevin | 0,6844 ± 0,0020 | 0,6468 ± 0,0042 | 0,6240 | 8, 8, 9 | 6,46 | 6,99 | 10,92 | 31,99 | 16,18 |
| K2_no_history | 0,6859 ± 0,0038 | 0,6479 ± 0,0103 | 0,6243 | 7, 8, 9 | 6,23 | 7,02 | 10,45 | 31,21 | 15,64 |
| K3_placebo | 0,6838 ± 0,0005 | 0,6472 ± 0,0048 | 0,6251 | 8, 8, 8 | 5,64 | 6,49 | 10,53 | 32,16 | 15,93 |
| K4_koopman_cover | **0,6885** ± 0,0019 | 0,6510 ± 0,0112* | 0,6306* | 4, 6, 5 | 5,29 | 6,33 | 10,40 | 31,26 | 15,61 |

Wartości dla poszczególnych ziaren (AP najlepszego punktu, niezależna ocena; ziarna 42, 1337, 2024): B0 0,6637, 0,6710, 0,6667; B1 0,6809, 0,6835, 0,6822; B2 0,6808, 0,6882, 0,6907; B3 0,6820, 0,6799, 0,6830; B4 0,6824, 0,6822, 0,6833; K1 0,6840, 0,6871, 0,6822; K2 0,6808, 0,6870, 0,6899; K3 0,6845, 0,6832, 0,6837; K4 0,6866, 0,6879, 0,6912.

**Kontrasty** (AP najlepszego punktu, jak w dokumencie 39):

| Kontrast | Różnica | Próg | Werdykt |
|---|---|---|---|
| K1 minus B3 | +0,0027 | +0,004 | poniżej progu |
| K1 minus B4 | +0,0018 | +0,004 | poniżej progu |
| K1 minus K2 | -0,0015 | +0,004 | brak (zapowiedziana mała moc, 83 procent wspólnych klatek) |
| K1 minus K3 (placebo) | +0,0006 | +0,004 | brak: przetasowany sygnał działa tak samo jak prawdziwy |
| chybienia w najszybszych 5 procentach, K1 minus B3 | -0,2 punktu | -3 punkty | brak |
| K4 minus B3 | +0,0068 | +0,004 | jedyny kontrast nad progiem |
| każde minus B0 | od +0,015 do +0,019 AP; od 3 do 5 punktów mniej chybień w szybkim ruchu | opisowo | wyraźny efekt |

Dwie uwagi do tego, jak kryteria zastosowano. Po pierwsze, jako miarę główną zapisano przed wynikiem AP z ostatniej epoki, a werdykty dokumentu 39 podano na AP najlepszego punktu (uzasadnienie: 30 epok przeucza, a różnice w ostatniej epoce są w szumie). Przeliczenie na ostatnią epokę z tej samej tabeli daje K1 minus B3 +0,0029, K1 minus B4 +0,0030, K1 minus K2 -0,0011, K1 minus K3 -0,0004 i K4 minus B3 +0,0071, więc werdykty się nie zmieniają; zmienia się jednak porównanie z B0, które w ostatniej epoce (0,6500) nie jest gorsze od pozostałych ramion (od 0,6438 do 0,6536), bo B0 dochodzi do szczytu dopiero w epoce 15 lub 16 i mniej się przeucza. Po drugie, przedziałów bootstrapowych i korekty Holma dla kontrastów AP w źródłach nie ma [nie ustalono w źródłach]; bootstrap po operacjach policzono tylko dla chybień ramienia K4 (sekcja 1.10).

**Odczyt** (dokument 39, sekcja 5b).
1. Teza „sygnał Langevina poprawia dobór klatek" nie znalazła potwierdzenia przy budżecie 1500 z puli 10 637. Żaden kontrast K1 nie przekroczył progu, a placebo dorównuje prawdziwemu sygnałowi, więc drobna przewaga K1 nad B3 pochodzi z samego rozluźnienia pokrycia, nie z dynamiki.
2. Wszystkie rozsądne sposoby doboru mieszczą się w 0,005 AP przy rozrzucie między ziarnami od 0,001 do 0,004; najprostszy dobór równomierny w czasie jest najlepszy lub równy najlepszym. To zgodne z kampanią Static36 i z Mittal 2025.
3. Jedyny wyraźny efekt: klatki po kolei są gorsze. Tracą od 0,015 do 0,019 AP, gubią narzędzie w szybkim ruchu o 3 do 5 punktów częściej i uczą się dwa razy wolniej (najlepsza epoka 16 zamiast 7, niższe AP w epokach od 0 do 5). Dobór ma więc znaczenie, ale wystarcza rozrzucenie klatek.
4. Wynik z ostatniej epoki jest niższy o około 0,035 AP od najlepszego i ma większy rozrzut: 30 epok to przeuczenie.
5. Szybki ruch pozostaje problemem każdego ramienia: około 32 procent chybień w najszybszych 5 procentach klatek wobec około 6 procent w wolnej połowie. Żaden dobór tego nie ruszył, więc to raczej granica modelu jednoklatkowego (rozmycie, wejście narzędzia, błędy adnotacji) niż brak danych.

**Ramię K4** (dokument 39, sekcja 5c). K4 minus B3 = +0,0068 AP, jedyny kontrast nad progiem. K4 ma też najmniej chybień w wolnej połowie klatek (5,29 procent wobec od 6,2 do 7,0) i w najszybszych 20 procentach (15,61), a najlepszą epokę osiąga najwcześniej (4 do 6). Względem doboru równomiernego w czasie (B2) przewaga +0,0019 mieści się w szumie trzech ziaren. Zastrzeżenia zapisane od razu: jedno ziarno wyboru, trzy ziarna treningu, najlepsze z dziewięciu ramion (tak zwana klątwa zwycięzcy: najlepszy z wielu wyników jest zawyżony choćby przez los) i brak kontroli wymiaru, bo K4 pokrywa przestrzeń niskiego wymiaru, więc zysk może pochodzić z samego niskiego wymiaru pokrycia, a nie z tego, że to współrzędne Koopmana. Rozstrzygnąć to miało pokrycie w 16 głównych kierunkach PCA i w 16 losowych rzutach tego samego latentu. Brama do etapu 2 (którykolwiek kontrast K nad progiem punktowo) została formalnie spełniona przez K4, z decyzją pozostawioną autorowi.

**Dlaczego K1 nie pomógł: co ramiona faktycznie wybrały** (prędkość z torów GT puli uczącej, progi 10,2 i 24,6 piksela na 3 klatki; kolumny dotyczą 1000 dokupionych klatek):

| Ramię | udział klatek z najszybszej piątej części | mediana prędkości (px na 3 klatki) | klatki wejścia i wyjścia narzędzia | klatki, na których L0 jest niepewny | filmy |
|---|---|---|---|---|---|
| B1_random | 0,19 | 4,8 | 0,04 | 0,21 | 118 |
| B2_uniform_time | 0,24 | 5,3 | 0,20 | 0,25 | 118 |
| B3_coverage | 0,34 | 6,4 | 0,19 | 0,39 | 117 |
| B4_uncertainty | 0,36 | 7,0 | 0,21 | 0,69 | 118 |
| K1_langevin | **0,40** | **8,0** | 0,09 | 0,32 | 118 |
| K2_no_history | 0,41 | 8,2 | 0,09 | 0,32 | 118 |
| K3_placebo | 0,25 | 5,5 | 0,05 | 0,27 | 117 |
| K4_koopman_cover | 0,25 | 5,3 | 0,09 | 0,24 | 116 |
| B0_consecutive | 0,22 | 5,3 | 0,04 | 0,17 | **58** |

1. Sygnał zadziałał zgodnie z projektem: K1 kupił dwa razy więcej klatek szybkiego ruchu niż losowanie (0,40 wobec 0,19).
2. Te klatki nie leczą błędu, w który celowały: dwa razy więcej przykładów szybkiego ruchu w treningu nie zmieniło chybień w szybkim ruchu na walidacji (około 32 procent w każdym ramieniu). To samo z niepewnością: B4 kupił 69 procent klatek, na których L0 był niepewny, i nie zyskał nic. W języku twierdzenia o rozkładzie błędu duża reszta to tu część nieusuwalna etykietami. Opóźnienia adnotacji w szybkim ruchu nie ma (przesunięcie predykcji wzdłuż ruchu: mediana -0,8 piksela), a przy chybieniu detektor jest pewny siebie (wynik 0,80 wobec 0,89 przy trafieniu), więc trafia w coś innego niż oznaczone narzędzie.
3. Liczy się liczba etykiet i rozrzut po operacjach, nie to, które klatki. Krzywa z kampanii statycznej: 500 klatek 0,646, 1500 klatek 0,682, 2500 klatek 0,693; dołożenie tysiąca dowolnych rozrzuconych klatek daje +0,036, a wybór między nimi najwyżej +0,007. Ramię B0 pokryło dokupionymi klatkami 58 filmów zamiast 118 i straciło 0,017.

Wniosek dokumentu 39: poprawianie równania Langevina albo głębszy autoenkoder Koopmana nie zmieni wyniku K1, bo wąskim gardłem nie jest jakość sygnału, tylko brak usuwalnego błędu we wskazanych klatkach; dynamika pomaga tam, gdzie opisuje stany do pokrycia (K4), a nie zaskoczenia do kupienia (K1). Sygnał Koopmana i Langevina ma na dziś wartość diagnostyczną, nie selekcyjną; miejscem, gdzie selekcja może coś dać, są bardzo małe budżety (100 do 500 etykiet) i pula z wieloma narzędziami. Audyt równoległej sesji z 22.09 (dokument 40, sekcja 1.23) przyjmuje zdanie „ten dobór nie poprawił detektora", ale uznaje mocniejsze zdania (błędy nieusuwalne etykietami, poprawa równania nie może pomóc) za wykraczające poza eksperyment.

Pliki: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\results_stage1\` (27 podkatalogów `matrix_<ramię>_b1500_sel42_s<ziarno>` z `epoch_val_map.json`, `run_manifest.json`, `validation_report.json`), `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\MISSES_BY_SPEED_best_val.json`, `MISSES_BY_SPEED_best_val_K4.json`, `val_speed_px_per_3frames.json`; wagi i archiwa predykcji na Edenie w `~/smallvit_runs/2026-09-21_koopman_langevin/results_stage1/`.

### 1.10 Rozbiór ramienia K4: co naprawdę pokrywa i jaką daje przewagę (21.09 wieczór, `d32d274b`)

**Metoda.** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\koopman_cover_analysis.py`, analiza bez treningu na gotowych biegach; wyniki `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\REPORT_koopman_cover.json` i `MISSES_PER_VIDEO_best_val.json`. Odtworzony wybór K4 pokrywa się z wyborem z Edenu w 100 procentach, więc analiza dotyczy tego samego obiektu.

**Czym są współrzędne Koopmana latentu całej klatki.**

| Współrzędne | wariancja między filmami | wyjaśnione stanem narzędzia (położenie, rozmiar) | wyjaśnione czasem w filmie | korelacja klatka do klatki |
|---|---|---|---|---|
| wolne mody Koopmana | **0,72** | -0,01 | 0,00 | 0,94 |
| główne kierunki PCA | 0,65 | 0,01 | 0,00 | 0,95 |
| losowe rzuty | 0,58 | 0,00 | 0,00 | 0,83 |

To nie jest stan narzędzia. Wolne mody latentu DINOv2 całej klatki to odcisk palca operacji: oko, oświetlenie, ustawienie mikroskopu; w obrębie jednego filmu prawie stoją. K4 rozkłada więc klatki po rodzajach scen, nie po stanach narzędzia; stan narzędzia siedzi w latencie detektora z obszaru narzędzia.

**Przewaga w uczeniu na gotowych 27 biegach.**

| Porównanie | AP najlepszej epoki | odpowiednik w losowych etykietach (nachylenie 0,0107 AP na 1000 klatek) | chybienia, sparowany bootstrap po 25 operacjach walidacji |
|---|---|---|---|
| K4 minus B3 (pokrycie wyglądu) | +0,0072 ± 0,0009 | około +670 klatek | **-1,42 punktu, przedział [-2,73; -0,26]** |
| K4 minus B1 (losowanie) | +0,0072 ± 0,0009 | około +670 | -0,72 punktu, przedział [-1,90; +0,06] |
| K4 minus B2 (równo w czasie) | +0,0023 ± 0,0028 | około +210 | -0,69 punktu, przedział [-1,75; +0,06] |

Różnica między +0,0068 z sekcji 1.9 a +0,0072 tutaj wynika ze źródła liczb: sekcja 5c dokumentu 39 używa niezależnej oceny zapisanego najlepszego punktu (`validation_report.json`), a sekcja 5d wartości AP liczonych w trakcie treningu przez bibliotekę (`epoch_val_map.json`; średnie tam: K4 0,6903, B3 0,6831, B1 0,6831, B2 0,6880). Obie wersje są w dokumencie 39 bez tego objaśnienia. Wartość po znaku ± w tej tabeli odpowiada, według mojego przeliczenia, błędowi standardowemu różnicy dwóch średnich z trzech ziaren, a nie odchyleniu między ziarnami.

Szybkość: K4 przekracza AP 0,675 w epoce 3 we wszystkich trzech ziarnach (pozostałe ramiona w epokach od 3 do 5, klatki po kolei nigdy); w epoce 3 ma 0,683 wobec 0,676 (B2), 0,674 (B1) i 0,668 (B3). Średnie AP z epok od 0 do 9: K4 0,6527, B2 0,6502, B1 0,6479, B3 0,6471.

**Czego nie udało się wykazać.**
1. Że działa akurat składnik Koopmana. Opisy wyboru (równomierność po filmach, pokrycie położenia i rozmiaru narzędzia, czasu, promień pokrycia w przestrzeni wyglądu) są dla K4 prawie takie same jak dla losowania i doboru równomiernego; pokrycie położenia narzędzia (entropia rozkładu wybranych klatek po przedziałach położenia; wyższa znaczy równiej) K4 ma wręcz najniższe (0,785), a najwyższe B4 (0,831), które nie wygrało.
2. Kontrole wymiaru wybierają inne klatki niż K4 (część wspólna: rzut 24-wymiarowy 0,24, PCA 0,21, losowe rzuty 0,13, przy około 0,10 dla czystego przypadku), więc nie wiadomo, czy uczyłyby równie dobrze; to wymaga treningów.
3. Szum samego wyboru jest niezmierzony. Odchylenia w tabelach to ziarna treningu przy jednym wyborze klatek, a dwa różne proste wybory różnią się o kilka tysięcznych (B1 i B2 o 0,0044, K3 i B3 o 0,0021). Przewaga K4 może w części być szczęśliwym losem jednego wyboru.

Wniosek: przewaga K4 jest realna względem pokrycia wyglądu (przedział po operacjach poniżej zera), nierozstrzygnięta względem doboru równomiernego w czasie, a jej przyczyna nieustalona. Etap 2 musi mieć kilka ziaren wyboru, kontrole PCA, losowe rzuty i rzut 24-wymiarowy oraz wariant na latencie narzędzia z detektora.

**Późniejsze korekty opisu K4.** (a) Audyt w dokumencie 40 (22.09, równoległa sesja) zauważa, że raport zawiera 10 rzeczywistych współrzędnych (8 funkcji własnych rozłożonych na części rzeczywiste i urojone, po usunięciu kolumn stałych; pole `koopman_coordinates: 10` w `REPORT_koopman_cover.json`), a nie 16, jak piszą dokument 39 i nazwy kontroli `pca16` i `random16` (kontrolne tablice też mają 10 współrzędnych); ponadto pole `r2_from_tool_state` mierzy kierunek „geometria narzędzia przewiduje współrzędne", a nie odczyt geometrii ze współrzędnych, więc wynik -0,01 nie wyklucza nieliniowej informacji o narzędziu. (b) 22.09 przy budowie ramion K5 okazało się, że K4 w ogóle nie było k-centrum w sensie odległości (sekcja 1.16).

### 1.11 Pilot „czego sieci brakuje": sygnał bez etykiet z latentu samej sieci (22.09, `d6cbb58b`)

**Pytanie autora.** Nie „które klatki pokryć", tylko „czego sieci brakuje do wytrenowania": czy dynamika latentu sieci w trakcie uczenia wskazuje klatki, po których cała sieć lepiej zrozumie narzędzie. Od tego commitu prace szły już w nowej sesji Claude Code (inny identyfikator sesji w commitach).

**Definicja mierzalna.** Brak usuwalny treningiem to klatka, na której detektor z 500 etykiet (L0, `ut_b0500_best`) chybia (IoU poniżej 0,5), a detektor z 2500 etykiet (`ut_b2500_best`) trafia. Na 3499 klatkach walidacji paczki z torami L0 chybia w 4,4 procent, duży detektor w 4,1 procent; tylko 23 procent (dokładnie 22,9) chybień L0 znika po pięciokrotnie większej liczbie etykiet, co przy około 150 chybieniach daje około 35 klatek usuwalnych. To jest sufit każdego doboru: 77 procent błędów małej sieci nie jest brakiem danych.

**Sygnały bez etykiet, liczone na punkcie L0**, i ich AUC (plik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_selection_stage1_2026-09-21\REPORT_defect_pilot.json`; kolumna „zysk IoU" pochodzi tylko z pliku JSON):

| Sygnał | chybienie L0 | usuwalne wśród chybionych | zysk IoU | usuwalne wśród wszystkich | udział usuwalnych w górnych 10 procentach |
|---|---|---|---|---|---|
| 1 minus pewność detektora | 0,796 | 0,758 | 0,773 | **0,928** | 6,9 procent |
| reszta Langevina latentu | 0,772 | 0,591 | 0,641 | 0,798 | |
| reszta minus ruch własnej ramki („defekt") | 0,731 | 0,456 | 0,621 | 0,689 | |
| reszta Koopmana w wolnych modach | 0,768 | 0,672 | 0,635 | 0,839 | 5,2 |
| dryf cząstki między punktami 500 i 1500 (zegar treningu) | 0,774 | 0,710 | 0,683 | 0,864 | 5,7 |
| pewność plus dryf (suma rang) | | | | 0,935 | 7,1 |
| pewność plus dryf plus Koopman | | | | 0,931 | 7,3 |

Ten sam dryf mierzony wewnątrz jednego biegu (epoki od 23 do 29 treningu b1500) wskazuje usuwalny błąd z AUC 0,873, pewność 0,901, suma rang 0,920; dryf wczesny (epoki od 2 do 5) 0,807. Korelacja pewności z dryfem wynosi tylko 0,28, z resztą Langevina 0,38, a reszty z „defektem" 0,68.

**Odczyt.** (1) Dynamika latentu sieci niesie informację o brakach (AUC od 0,84 do 0,87 wobec 0,5 dla losu), ale własna pewność detektora niesie jej więcej (0,93). Odjęcie ruchu ramki od reszty latentu pogarsza sygnał, więc pomaga część zgodna z ruchem narzędzia, nie „defekt reprezentacji". (2) Jedyny sygnał z dynamiki, który coś dokłada do pewności, to zegar treningu (klatki, których cząstka wciąż dryfuje między punktami kontrolnymi): suma rang 0,935 wobec 0,928, a w górnych 10 procentach o 0,4 punktu więcej usuwalnych; przy 35 usuwalnych klatkach mieści się to w szumie. (3) Nawet najlepszy sygnał w górnych 10 procentach trafia w usuwalny błąd tylko w 7 procentach (to precyzja w górnym decylu). Dobór celujący w „braki sieci" kupuje więc głównie klatki, których błędu etykiety nie usuną (mechanizm z Hu i in. 2021), co tłumaczy, czemu B4 i K1 nie pomogły. Wniosek dokumentu 39 (sekcja 5e): sygnał „Koopman wskazuje, czego sieci brakuje" da się zbudować jako sumę rang pewności, dryfu i reszty w wolnych modach, ale sufit 23 procent i precyzja 7 procent oznaczają zysk co najwyżej ułamka tego, co daje dorzucenie rozrzuconych etykiet; większy zapas leży w samej sieci, więc kolejnym sensownym krokiem jest Koopman jako cel uczenia (spójność latentu sąsiednich klatek pod jednym operatorem liniowym), nie selektor. Kod: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\koopman_defect_pilot.py`.

### 1.12 Widmo Koopmana latentu detektora wzdłuż uczenia i granica linearyzacji (22.09, `b7f94383`)

**Metoda.** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\koopman_spectrum_axis.py`, wynik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_spectrum_axis.json`. Latent `box_pool` po PCA do 24 wymiarów podniesiono jednym opóźnieniem (`[z_t, z_{t-1}]`), operator liniowy jednego kroku dopasowano regresją grzbietową (EDMD) na 4 klipach uczących paczki i oceniono na pozostałych 12 (walidacyjnych; w kodzie nazwanych testowymi). Klasy wartości własnych: niezmiennik (|λ| powyżej 0,97 i kąt poniżej 5 stopni, czyli wielkość stała z klatki na klatkę), wolna rzeczywista (|λ| od 0,8 do 0,97), obracająca się (kąt co najmniej 5 stopni), szybka (reszta). Defekt niezmiennika to względny błąd `psi(z_{t+1}) - λ psi(z_t)` na klipach nieużytych do dopasowania.

| Wagi | niezmienniki | wolne | obracające | szybkie | niezmienniki: wariancja między filmami | niezmienniki: R2 ze stanu narzędzia | niezmienniki: defekt poza próbą |
|---|---|---|---|---|---|---|---|
| COCO | 2 | 3 | 31 | 12 | 0,81 | -0,08 | 0,19 |
| epoka 2 (b1500) | 4 | 7 | 29 | 8 | 0,81 | 0,59 | 0,05 |
| 500 klatek | 4 | 6 | 26 | 12 | 0,78 | 0,58 | 0,05 |
| 2500 klatek | 5 | 6 | 26 | 11 | 0,74 | 0,29 | 0,09 |
| epoka 29 (b1500) | 3 | 7 | 28 | 10 | 0,77 | 0,29 | 0,04 |

**Odczyt.** (1) Niezmienniki to tożsamość operacji (od 74 do 81 procent wariancji między filmami), a po uczeniu także stan narzędzia (R2 0,58 tuż po starcie uczenia, 0,29 przy dłuższym treningu). (2) Defekt niezmienników poza próbą spada z 0,19 przed uczeniem do 0,05 po uczeniu: to, co sieć uznaje za stałe, staje się naprawdę stałe z klatki na klatkę. Dokument 38 nazywa to mierzalną krystalizacją bez etykiet i kandydatem na miarę do śledzenia co epokę. (3) Mody obracające się mają okres około 7 klatek i zerowy związek ze stanem narzędzia: to drganie, nie dynamika narzędzia. (4) Zastrzeżenie: 24 wymiary i 4800 klatek dają 48 wartości własnych z próby o ograniczonej długości, więc liczby modów są przybliżone.

**Ile części nieliniowej da się zamienić w liniową** (umiejętność jednokrokowa poza próbą w procentach, latent 2500 klatek):

| wymiar | liniowy, 1 opóźnienie | liniowy, 8 opóźnień (najlepsza regularyzacja) | nieliniowy z historią (RFF plus 8 kroków) |
|---|---|---|---|
| 8 | 7,7 | 17,0 | 21,4 |
| 16 | 2,0 | 5,2 | 21,0 |
| 24 | -1,1 | -2,5 | 20,9 |
| 32 | -10,4 | -10,3 | 21,0 |
| 64 | -17,3 | -19,1 | 20,4 |

Liniowy operator we współrzędnych samego latentu działa tylko przy 8 wymiarach i 8 opóźnieniach (17 procent) i psuje się z wymiarem; model nieliniowy trzyma 21 procent niezależnie od wymiaru, a tor fizyczny narzędzia ma od 22 do 29 procent. Autoenkoder Koopmana miałby do domknięcia lukę 4 do 5 punktów przy małym wymiarze i całą przy dużym; sufit jest fizyczny (ręka chirurga), nie słownikowy.

### 1.13 Człony z teorii sterowania i pilot doboru D-optymalnego (22.09, `abd6d0b5`, `7e701001`)

**Pytanie autora.** W testach narzędzia diagnostycznego projektu grantowego wymuszeniem był ruch kamery (sterowany i znany co do liczby), a w chirurgii kamera mikroskopu stoi, a wejściem jest ręka chirurga, czyli nieobserwowane sterowanie. Co jeszcze z teorii sterowania pasuje do równania i do doboru danych?

**Przegląd literatury** (Undermind plus Scite, dokument 38, sekcja 4h). Dokładne podniesienie układu z wejściem daje stałą macierz `A`, ale macierz wejścia zależną od stanu, więc właściwy człon jest biliniowy (`sum_i u_i N_i z`, iloczyn wejścia i stanu), a nie `W u`; przy nieobserwowanym wejściu identyfikowalny jest tylko iloczyn `B u` (Proctor, Brunton, Kutz 2018; Korda, Mezić 2018; Iacob, Tóth, Schoukens 2022; Bruder i in. 2021; Shang i in. 2026). Nieznane wymuszenie można traktować jako resztę modelu opóźnieniowego (HAVOK) albo jako proces autoregresyjny w stanie rozszerzonym `[z; u]`. Gramian obserwowalności Koopmana `W_o = sum_k (A^k)^T C^T C A^k` mówi, które kierunki stanu widzi wyjście (tu głowa detektora); kierunki o małej wartości własnej `W_o` to te, które latent zmienia, a detekcja nie widzi. Projekt eksperymentu dla operatora (informacja Fishera, czyli miara tego, ile dane mówią o parametrach modelu; T- i D-optymalność; certyfikaty jakości danych) nie był robiony dla wideo; Wu 2026 ostrzega, że wynik zadania końcowego nie jest monotoniczny w żadnym certyfikacie, a You i in. 2026, że błąd jednokrokowy operatora nie mierzy użyteczności modelu latentnego. Ręka steruje przerywanie (korekty wyzwalane progiem błędu, refrakcja 200 do 500 ms, 2 do 3 submovements na sekundę; Gollee i in. 2017, Susilaradeya i in. 2019), co tłumaczy kurtozę innowacji od 25 do 132 z sekcji 1.5; chirurg reaguje na to, co widzi, więc identyfikacja w pętli jest obciążona. Dobór przykładów jako sterowanie stanem uczącej się sieci istnieje w literaturze (machine teaching z zasadą maksimum Pontriagina), ale rozwiązany tylko dla prostych modeli gradientowych ze znanym celem.

Cztery człony zaproponowane do równania, każdy z kontrolą: człon biliniowy z częściowo zmierzonym wejściem (przemieszczenie końcówki z detektora; kontrola: permutacja wejścia w czasie ma wyzerować `N_i`), człon skokowy z wyzwalaniem progowym i refrakcją (identyfikacja odpornym filtrem wielomodelowym IMM; obrazowy odpowiednik członu zdarzeń `Z` z równania grantowego), stan rozszerzony `[z; u]` z warunkiem rangi oraz sprzężenie zwrotne `u_t = -K (z_t - z_ref) + w_t` identyfikowane metodą pośrednią z fazą operacji jako instrumentem. Żadnego z nich w tym etapie nie dopisano do kodu.

**Nowe kryterium doboru: D-optymalne (dźwignia).** Dla regresji operatora kowariancja estymaty jest proporcjonalna do `(Psi_S Psi_S^T + λI)^-1`, więc klatka o największej dźwigni `s_t = psi_t^T (Psi_S Psi_S^T + λI)^-1 psi_t` najbardziej zmniejsza niepewność operatora; zachłanny wybór `argmax s_t` maksymalizuje logarytm wyznacznika macierzy informacji (log det), czyli zmniejsza objętość elipsoidy niepewności parametrów. Test bez treningu zapisany przed wynikiem: przy równym budżecie ślad kowariancji operatora (suma wariancji parametrów; mniej znaczy pewniej) ma spadać monotonicznie dla doboru D-optymalnego, a nie dla doboru po reszcie, a błąd jednokrokowy poza grupami nie może rosnąć. Kod: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\leverage_selection.py`, wynik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_leverage.json`.

Pula 10 637 klatek, z podniesionym stanem 6874; dla każdego ramienia odłożono 39 ze 117 grup operacji (17 827 przejść do oceny); L0 to 335 klatek z podniesionym stanem; λ = 1; dobierano 100, 250, 500 i 1000 klatek. Przy 1000:

| Ramię | log det | ślad kowariancji operatora | umiejętność jednokrokowa na odłożonych grupach (procent) |
|---|---|---|---|
| równo w czasie | 300,4 | 0,210 | 7,3 |
| losowanie | 300,4 | 0,209 | 7,4 |
| reszta Langevina (górne) | 314,8 | 0,126 | 4,8 |
| D-optymalne (dźwignia) | 320,7 | **0,116** | **4,0** |

Przebieg ze wzrostem budżetu (100, 250, 500, 1000): ślad dla D-optymalnego 0,42, 0,27, 0,18, 0,12, dla równego w czasie 0,69, 0,49, 0,34, 0,21; umiejętność dla D-optymalnego -5,5, -2,1, 0,4, 4,0, dla równego w czasie -2,1, 1,6, 4,9, 7,3. Część wspólna wyboru D-optymalnego z wyborem po reszcie 53 procent, z równym w czasie 16 procent.

**Odczyt.** Przewidywanie sprawdziło się w połowie: dobór D-optymalny najmocniej i monotonicznie zmniejsza niepewność operatora, ale dobór po reszcie robi prawie to samo, bo klatki o dużej reszcie są zarazem klatkami o dużej dźwigni. Ważniejsze: operator dopasowany na klatkach D-optymalnych przewiduje przejścia na nowych operacjach gorzej (4,0 wobec 7,3), bo pewność co do operatora kupiono na przypadkach skrajnych, które nie opisują typowego kroku. Certyfikat informacji nie jest monotoniczny w użyteczności (ostrzeżenie Wu 2026 i You 2026). Kryterium D-optymalne zlewa się z kryterium zaskoczenia, które w macierzy etapu 1 nie pomogło, i nie dostało ramienia treningowego; do rozważenia zostaje dźwignia ważona gęstością typowych stanów, czyli w praktyce pokrycie.

### 1.14 Cząstki jako tokeny-patche backbone'u (22.09, `dd066100`, `14ad80d7`; zadania Eden 1809374 i 1809381)

**Pytanie i motywacja.** Na życzenie autora cząstką przestała być cała klatka, a stał się pojedynczy token, czyli wektor opisu jednego kwadratowego fragmentu obrazu (patcha) po ostatniej warstwie backbone'u. Przy wejściu 384 pikseli siatka ma 24 na 24 = 576 tokenów na klatkę. To konwencja narzędzia diagnostycznego z projektu grantowego (rejestrator z `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\test\02_rejestrator.md`, testy 03 i D65).

**Metoda.** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\patch_particles.py` (rejestrator, czyli program zapisujący stany cząstek) i `patch_analysis.py` (analiza). Rusztowanie odczytano z wag końcowej normalizacji warstwy (LayerNorm, operacja, która centruje i skaluje wektor, a potem mnoży go przez wagi `gamma` i dodaje `beta`) backbone'u DINOv2 w RF-DETR: jądro = `beta`, długości = `|gamma|`, K = 384. Stan cząstki to profil `(z - beta)/|gamma|`. Na 4800 klatkach paczki bazę PCA o 32 wymiarach dopasowano raz na 512 klatkach rozłożonych po całym nagraniu (od 66 do 69 procent wariancji). Zapisano pięć wersji wag: COCO, 500 klatek, epoka 5 i epoka 29 biegu b1500 oraz 2500 klatek (zadanie 1809374: zapis, 10 minut, około 700 MB na wersję wag; zadanie 1809381: analiza).

**Strażnicy** (automatyczne testy poprawności zapisu). Udział kierunku martwego 7e-9 (test T3c zdany). Promień profilu 1,000 dla każdej cząstki, więc człon `eps` normy jest pomijalny, a stopień swobody „głośności" (T24c) numerycznie nie istnieje. Determinizm (T35) nie zdany: 50 klatek zakodowanych drugi raz różni się o 0,11 do 0,22 we współrzędnych zredukowanych, czyli o 7 do 14 procent odchylenia współrzędnej. Podejrzana przyczyna: autocast (automatyczne obniżanie precyzji obliczeń) i jądra obliczeniowe mechanizmu uwagi (SDPA) zależne od składu partii w `predict`. Każde porównanie klatka do klatki ma przez to podłogę szumu rzędu 0,1 odchylenia.

**Wynik** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_patch_particles.json`). Basen to skupienie cząstek o podobnym kierunku profilu w obrębie klipu (k = 96 skupień).

| Wagi | baseny: IoU z maską narzędzia (podłoga q95 0,414; klipów nad podłogą z 6) | λ translacyjne: najwolniejszy / ósmy (podłoga) | cosinus podprzestrzeni translacyjnej i czasowej | tożsamość narzędzia, AUC: wolne 8 / szybkie 8 / PCA 8 / pełne 32 |
|---|---|---|---|---|
| COCO | 0,482 (3) | 1,01 / 0,27 (0,01) | 0,51 | 0,776 / 0,808 / 0,867 / 0,894 |
| 500 klatek | 0,522 (5) | 0,80 / 0,41 (0,04) | 0,55 | 0,848 / 0,667 / 0,889 / 0,911 |
| epoka 5 (b1500) | 0,563 (5) | 0,83 / 0,43 (0,04) | 0,55 | 0,851 / 0,806 / 0,905 / 0,921 |
| 2500 klatek | 0,537 (5) | 0,81 / 0,42 (0,04) | 0,56 | 0,899 / 0,789 / 0,913 / 0,924 |
| epoka 29 (b1500) | 0,555 (5) | 0,78 / 0,40 (0,04) | 0,48 | 0,869 / 0,767 / 0,902 / 0,918 |

**Odczyt.**
1. Test basenów zdany po uczeniu, niepewny przed: skupienia cząstek jednego klipu pokrywają maskę narzędzia z IoU od 0,52 do 0,56 wobec podłogi gładkich podziałów losowych 0,41; przed uczeniem 0,48 i tylko 3 klipy z 6 nad podłogą. Narzędzie jest cienkie (3,6 procent klatki), stąd IoU dużo niższe niż 0,81 dla niedźwiedzia z filmów DAVIS (publicznego zbioru krótkich filmów z maskami obiektów) w projekcie grantowym.
2. Operator translacyjny (test D65), czyli operator Koopmana, którego krokiem nie jest następna klatka, lecz przesunięcie narzędzia: para to ten sam obszar narzędzia w dwóch miejscach kadru, wskazany własną ramką detektora, bez etykiet. Dla COCO ramka pokrywa 35 procent klatki (detektor nie zna narzędzia), więc wiersz COCO nie jest porównywalny; po uczeniu ramka to 3,6 procent klatki. Po uczeniu najwolniejszy mod ma λ około 0,8 przy podłodze z zerwanych par 0,04: niezmienniki przesunięcia istnieją i są odróżnialne od losu.
3. Tożsamość narzędzia w niezmiennikach przesunięcia pojawia się z uczeniem: osiem wolnych modów odróżnia token narzędzia od tokenu tła z AUC 0,78 przed uczeniem i 0,90 po 2500 klatkach, a osiem modów szybkich spada z 0,81 do 0,79. Ale to mniej niż osiem głównych kierunków PCA (0,913) i pełne 32 wymiary (0,924): niezmienniki Koopmana dobrze opisują, co sieć zrobiła z tożsamością narzędzia, lecz nie są lepszym kodem niż zwykła wariancja.
4. Podprzestrzenie wolne operatora translacyjnego i czasowego mają średni cosinus około 0,5: „to samo w innym miejscu" i „to samo w następnej klatce" to różne kierunki latentu, zgodnie z opisem Eulera (w zegarze filmu przez stałe miejsce siatki przepływa scena, nie narzędzie).
5. Test rejestrów (T6a) wymaga normy przed normalizacją, której rejestrator nie zapisuje; pozostał do zrobienia.

### 1.15 Z latentu do klatek: dekodowanie niezmienników i wyszukiwanie „dobrze domkniętych" klatek (22.09, `22e5e437`)

**Pomysł autora.** Znaleźć klatki, które poprawnie zakodowały niezmienniki Koopmana, zdekodować z latentu cechy obrazu potrzebne do „domknięcia" tożsamości narzędzia i szukać podobnych klatek w zbiorze uczącym.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\invariant_decode.py`). Na cząstkach-tokenach detektora po 2500 klatkach operator translacyjny dopasowano na 34 336 parach; osiem jego wolnych modów (|λ| od 0,81 do 0,42) to kod tożsamości. Każda klatka dostaje średni kod swoich tokenów narzędzia (tokenów wewnątrz własnej przewidzianej ramki, bez etykiet); prototyp to średni kod pewnych klatek z klipów uczących; domknięcie to odległość kodu klatki od prototypu (4711 klatek z kodem). Etykiety służyły wyłącznie do oceny.

**Wynik** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_decode_b2500.json`). Domknięcie wskazuje chybienia detektora z AUC 0,625 (pewność detektora 0,810; korelacja domknięcia z IoU -0,20). Na małym detektorze (500 klatek) domknięcie wskazuje błąd usuwalny z AUC 0,533, czyli na poziomie losu (pewność 0,899). Osiem najlepiej domkniętych klatek walidacji ma IoU od 0,87 do 0,93 i wszystkie pochodzą z filmu TR_0034; osiem najgorzej domkniętych ma IoU od 0,72 do 0,92 i wszystkie pochodzą z filmu TR_0136. Domknięcie porządkuje więc klatki według klipu i sceny (ostre narzędzie na niebieskiej tęczówce wobec rozmytego przy brzegu), a nie według tego, czego sieci brakuje.

Dekodowanie do pikseli: liniowy dekoder z pełnego profilu tokenu (384 liczby) do jego 16 na 16 na 3 pikseli, dopasowany na 240 klatkach klipów uczących, oceniony na walidacyjnych, daje R2 0,54 na wszystkich tokenach, ale -0,69 na tokenach narzędzia: liniowo odtwarza się tło (tęczówka, kolor), narzędzia nie. Ten sam dekoder na tokenie rzutowanym na osiem wolnych modów daje R2 0,09 (na tokenach narzędzia -0,76), na osiem szybkich 0,09 (-1,05), na osiem kierunków PCA 0,26 (-0,71). Na rysunkach `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\dekodowanie_b2500_klatka.png` i `dekodowanie_b2500_wycinki_domkniecie.png` z niezmienników zostaje jednolita plama z ledwie widoczną tęczówką.

**Wniosek.** Kod tożsamości jest semantyczny: nie jest cechą obrazu, którą dałoby się „odszumić" i wyszukać po wyglądzie, bo z pikseli prowadzi do niego tylko cała sieć. „Proces odwrotny do enkodera" w wersji liniowej nie działa dla narzędzia; wersja nieliniowa (osobny uczony dekoder) pokazałaby, co sieć „wierzy", ale nie dałaby kryterium wyboru innego niż podobieństwo w samym latencie. Żywotna postać pomysłu to pokrycie kodu tożsamości w puli, liczone na tokenach narzędzia z aktualnego detektora, a nie na odcisku sceny jak K4. Rejestrator dostał tryb listy klatek dla rzadkiej puli (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_pool_particles.slurm`).

### 1.16 Kod tożsamości narzędzia na całej puli i ramiona K5 (22.09, `84cab8fd`; zadania Eden 1809407 i 1809408)

**Wejście.** Zadanie 1809407 (3 minuty na H100) przepuściło detektor L0 (`ut_b0500_best.pth`) przez wszystkie 10 637 klatek puli uczącej `tracking_static36` i zapisało tokeny (`patch_particles.py --frame-list`). Zadanie 1809408 (13 sekund na CPU, `identity_code_pool.py`) dopasowało operator translacyjny na 612 682 parach tokenów narzędzia i policzyło kod tożsamości (8 modów, |λ| od 0,64 do 0,32 według dokumentu 39; w `K5_MANIFEST.json` najmniejsza wartość to 0,315) oraz domknięcie każdej klatki. Kod ma 9685 klatek; 952 klatki są bez kodu, bo L0 nie ma tam pewnej ramki (wynik nie wyższy niż 0,5 albo mniej niż 4 tokeny w ramce). Plik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\KOD_TOZSAMOSCI_pool_b0500.npz` (439 KB, poza gitem; na Edenie `~/smallvit_runs/2026-09-21_koopman_langevin/KOD_TOZSAMOSCI_pool_b0500.npz` i nagranie `czastki/CZASTKI_pool_b0500`). Strażnik determinizmu rejestratora nadal nie przechodzi (0,131 jednostki współrzędnej): przy odległościach pokrycia rzędu 1 to nie przeszkadza, ale wyklucza porównania klatka po klatce.

**Cztery ramiona** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\identity_cover_arm.py`; ten sam L0, budżet 1500, odstęp 5 klatek, bez etykiet; listy `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\curated\selection_matrix_stage1\matrix_K5_identity_cover_b1500_sel42.json`, `matrix_K5b_identity_cover_plus_unseen_b1500_sel42.json`, `matrix_K5c_closure_near_b1500_sel42.json`, `matrix_K5d_closure_far_b1500_sel42.json`, opis w `K5_MANIFEST.json`):

| Ramię | Reguła |
|---|---|
| K5_identity_cover | k-centrum euklidesowe w ustandaryzowanym kodzie (zawsze klatka najdalsza od już oznaczonych): rzadkie stany tożsamości najpierw |
| K5b_identity_cover_plus_unseen | jak K5, ale 20 procent dokupionego budżetu (200 klatek) równomiernie w czasie z klatek bez kodu |
| K5c_closure_near | dosłowny pomysł autora: klatki o kodzie najbliższym prototypu (tożsamość zakodowana „poprawnie"); kwota na film proporcjonalna do jego udziału w puli |
| K5d_closure_far | lustro K5c: klatki o kodzie najdalszym od prototypu |

**Pomiar pokrycia bez treningu.** Odległość losowej klatki puli (2000 klatek) od najbliższej klatki oznaczonej (L0 plus dokupione) w przestrzeni kodu, ta sama miara dla ramion macierzy etapu 1 (p90 to 90. percentyl tych odległości, maksimum to największa luka):

| Zbiór oznaczony | mediana | p90 | maksimum | filmy | klatek bez kodu | mediana domknięcia dokupionych |
|---|---|---|---|---|---|---|
| tylko L0 (500) | 1,055 | 1,580 | 2,914 | | | |
| K5_identity_cover | 0,919 | 1,178 | 1,335 | 112 | 0 | 2,117 |
| K5b_identity_cover_plus_unseen | 0,943 | 1,220 | 1,408 | 114 | 200 | 2,145 |
| K5c_closure_near | 0,913 | 1,478 | 2,914 | 118 | 0 | 1,144 |
| K5d_closure_far | 0,938 | 1,307 | 2,206 | 118 | 0 | 2,301 |
| K4_koopman_cover | 0,859 | 1,313 | 2,443 | 116 | 100 | 1,666 |
| B3_coverage | 0,874 | 1,315 | 2,346 | 117 | 180 | 1,680 |
| B2_uniform_time | 0,868 | 1,285 | 2,483 | 118 | 117 | 1,681 |
| B1_random | 0,844 | 1,303 | 2,658 | 118 | 97 | 1,645 |
| K1_langevin | 0,865 | 1,324 | 2,485 | 118 | 142 | 1,641 |
| B4_uncertainty | 0,921 | 1,339 | 2,585 | 118 | 364 | 1,821 |

Mediana domknięcia w całej puli wynosi 1,660; część wspólna K5 z ramionami macierzy etapu 1 od 9 do 13 procent.

**Odczyt.** (1) Żadne ramię macierzy etapu 1 nie różni się od losu w przestrzeni kodu tożsamości (mediana od 0,84 do 0,87, maksimum od 2,3 do 2,7), więc etap 1 nie testował pokrycia stanów tożsamości widzianych przez samą sieć; K5 to nowa oś, a nie wariant B3 czy K4. (2) K5 zmniejsza największą lukę pokrycia o połowę (z 2,9 dla samego L0 do 1,3) i najmocniej ściąga p90; medianę ma nieco gorszą od losu (0,92 wobec 0,84), bo k-centrum wydaje etykiety na rzadkie stany, i kupuje klatki o dużym domknięciu (mediana 2,12 wobec 1,66 w puli), czyli nietypowe stany tożsamości. (3) K5c, czyli dosłownie „klatki, które kodują poprawnie", pokrywa typowe stany jak los, a rzadkich nie dotyka wcale (maksimum 2,914, tyle co sam L0). Kontrast K5 minus K5c rozstrzyga więc, czy sieć lepiej uczy się z rzadkich stanów tożsamości, czy z klatek, które już rozumie. (4) B4 kupiło 364 klatki bez kodu (36 procent dokupionych) i ma najgorsze pokrycie: niepewność kupuje „nie widzę narzędzia", a nie „widzę je w nowym stanie".

**Usterka w ramieniu K4, znaleziona przy okazji.** Funkcja `select_batch(strategy='coverage')` liczy k-centrum kosinusowe (odległość mierzona kątem) i zakłada wektory jednostkowe, a `selection_matrix.py` podał jej współrzędne Koopmana ustandaryzowane, ale nieznormalizowane. K4 było więc regułą „kierunek najbardziej przeciwny do już wybranych" (z normą wzmacniającą klatki skrajne), a nie pokryciem w sensie odległości. Wynik K4 (0,6885) stoi, bo ramię jest deterministyczne i opisane kodem, ale nazwa „pokrycie" jest na wyrost; kontrola K4 z k-centrum euklidesowym trafiła do planu etapu 2. B3 tej wady nie ma (cechy DINOv2 w pamięci podręcznej są jednostkowe), a K5 używa własnego k-centrum euklidesowego.

**Czego to nie dowodzi i propozycja.** Nie było ani jednego treningu; lepsze pokrycie kodu nie musi przełożyć się na AP (w macierzy etapu 1 pokrycie wyglądu nie pobiło doboru równomiernego). Propozycja etapu 2, do zatwierdzenia przez autora i niezgłoszona na Eden: K5, K5c i K5b przy tym samym L0 i budżecie 1500, po 3 ziarna treningu, 12 epok zamiast 30 (najlepsza epoka wypadała około 7); kontrast główny K5 minus B2, pomocniczy K5 minus K5c; próg jak w macierzy etapu 1 (0,004 AP najlepszej epoki, chybienia z bootstrapem po operacjach); koszt 9 biegów po około 25 minut na A100, w pięciu równoległych procesach około godziny.

### 1.17 Co naprawdę niesie kod tożsamości, sprawdzenia poprawności i odniesienie do eksperymentów z niedźwiedziem (22.09, `aff568e8`, `03fd517a`, `76cd61f4`, `a492bd49`)

**Rysunki objaśniające** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\rysunki_k5.py`, `sledzenie_czastek.py`; lokalne nagranie cząstek L0 `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\czastki\CZASTKI_b0500`; rysunki poza gitem w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\rysunki_k5\`: `01_klatka_i_kwadraty.png`, `02_pary_przy_przesunieciu.png`, `03_co_przezywa_przesuniecie.png`, `04_mapa_stanow_i_wybor.png`, `05_czy_czastki_trzymaja_sie_narzedzia.png`, `06_czastka_stoi_narzedzie_plynie.png`). Pokazują siatkę 24 na 24 z tokenami narzędzia w ramce detektora, pary tokenów przy przesunięciu narzędzia, co przeżywa przesunięcie (zgodność par 0,77 dla najstabilniejszego kierunku wobec -0,17 dla najmniej stabilnego) i mapę stanów narzędzia z klatkami, które k-centrum oznaczyłoby jako następne. Korelacja kodu z położeniem ramki w obrębie klipu wynosi 0,28 wobec 0,25 dla kierunków niestabilnych, więc nie wykazano, że kod nie zależy od położenia w klipie; zmierzono tylko, że przeżywa przesunięcie.

**Co niesie osiem liczb kodu** (`co_niesie_kod.py`; predykcja regresją grzbietową z podziałem na foldy po filmach, żeby model nie mógł wygrać rozpoznaniem filmu; etykiety wyłącznie do oceny; 9142 klatki, 117 filmów):

| Wielkość przewidywana z kodu | Wyjaśniona zmienność z kodu (procent) | To samo z geometrii ramki detektora |
|---|---|---|
| wydłużenie prawdziwej ramki (im bardziej wydłużona, tym bardziej narzędzie leży w płaszczyźnie kadru) | 85,8 | 3,0 |
| pole prawdziwej ramki (skala narzędzia) | 29,0 | 0,0 |
| zwrot: końcówka względem środka ramki, poziomo | 0,1 | 2,5 |
| zwrot: końcówka względem środka ramki, pionowo | 0,3 | 0,9 |
| położenie narzędzia w kadrze, poziomo | 10,0 | 100 |
| położenie narzędzia w kadrze, pionowo | 18,5 | 100 |

Udział zmienności kodu między filmami: 58 procent. Kod niesie przede wszystkim geometrię narzędzia (nachylenie względem płaszczyzny obrazu i skalę) plus tożsamość operacji; nie niesie zwrotu (gdzie jest końcówka) i w małym stopniu położenie. Wcześniejsze zdanie z rozmowy, że „osiem liczb mówi, czy narzędzie jest otwarte czy zamknięte", wycofano: w tym podzbiorze jest jedna klasa narzędzia (cystotom, 22 680 klatek) i nie ma etykiety stanu narzędzia.

**Sprawdzenie siatki.** Dziesięć tokenów najpodobniejszych do wzorca narzędzia leży w prawdziwej ramce w 35 procentach przypadków, przy 2 procentach dla losu (tyle wynosi udział pola ramki w kadrze) i 7 procentach przy siatce transponowanej; transpozycja psuje wynik pięciokrotnie, więc kolejność wiersz-kolumna w rejestratorze jest zgodna z obrazem.

**Dopasowanie tokenów między klatkami.** Token wybrany na narzędziu, szukany w kolejnych klatkach po największym podobieństwie kosinusowym profilu, trafia w prawdziwą ramkę w 79 procentach na ośmiu klipach (97 procent na klipie z rysunku 05), przy 3 procentach dla losu. To odpowiednik śledzenia punktów na filmach DAVIS: opis tokenu niesie tożsamość miejsca na narzędziu. Przy tym sama cząstka się nie porusza: w konwencji projektu grantowego (opis Eulera, test T15a) cząstka to stałe miejsce siatki, przez które narzędzie przepływa (rysunek 06), w przeciwieństwie do opisu Lagrange'a, w którym cząstka podąża za obiektem; ruch wchodzi do analizy przez parowanie tokenów. Ograniczenie: tokeny narzędzia pochodzą z wnętrza ramki detektora, a nie z sylwetki narzędzia, więc przy cienkim narzędziu część z nich opisuje tło; dotyczy to też par, z których liczony jest kod.

**Niedźwiedź z projektu grantowego** (podagent przeglądowy, tylko odczyt; `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\test\latki_lagrange.py`, `film_lagrange.py`, `sledzenie.py`, `USTALENIA.md`, `F:\Studia\MCHTR\Grant\Projekt\Research\Tezy\tezy\D104.md`). Autor zarzucił, że w eksperymentach z niedźwiedziem cząstki śledziły obiekt, a tu stoją. Przegląd pokazał, że istnieją oba opisy. W linii DAVIS cząstki zasiane na niedźwiedziu były niesione przepływem optycznym Farnebäcka (polem przesunięć pikseli między klatkami) i po 80 klatkach 76,3 procent wciąż leżało na zwierzęciu (94,4 procent przy sieci śledzącej punkty CoTracker3), przy 7,0 procent dla kontroli z tła. Zysk dla dynamiki był jednak znikomy (R2 z 0,0762 na 0,0783), a na drugim filmie (Wilford) przepływ optyczny przegrał z cząstką nieruchomą (ICC 0,722 wobec 0,947; ICC to współczynnik korelacji wewnątrzklasowej, miara zgodności powtarzanych pomiarów) i został zastąpiony śledzeniem w przestrzeni cech: 40,8 procent cząstek zostaje na zwierzęciu przez cały odcinek wobec 27,6 procent dla przepływu optycznego (teza D104). Test z rysunku 05 jest odpowiednikiem tego zwycięskiego wariantu.

Wszystko to dopisano do sekcji 5f dokumentu 39.

### 1.18 Widmo operatora translacyjnego bez symetryzacji i ślad po przejściu narzędzia (22.09, `14ba6698`, `037e4223`)

**Zarzut autora 1.** W niezmiennikach Koopmana trzeba osobno badać część rzeczywistą dodatnią, rzeczywistą ujemną i urojoną. Zarzut okazał się słuszny: kod tożsamości liczono operatorem po symetryzacji (`generalised_koopman`), czyli zastąpionym wersją symetryczną, której wartości własne są z definicji rzeczywiste, więc cała część obrotowa przepadała.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\widmo_tozsamosci.py`, sekcja 4k dokumentu 38). Ten sam operator translacyjny bez symetryzacji, nagranie L0 na 16 gęstych klipach, 34 165 par, etykiety wyłącznie do oceny, podział po filmach. Z 32 modów 16 to mody trwałe (część rzeczywista dodatnia, bez obrotu), 6 przemiennych (część rzeczywista ujemna: wartość zmienia znak po przesunięciu) i 10 obrotowych (część urojona różna od zera: para wartości obraca się o stały kąt); do odczytu brano do sześciu modów z każdej rodziny (18 razem).

| Rodzina modów | Ile w widmie | Odróżnienie narzędzia od tła (AUC) | Przeżycie przesunięcia | Wydłużenie ramki (procent) | Wielkość ramki |
|---|---|---|---|---|---|
| trwałe | 16 | 0,834 | 0,61 | 71,3 | -1,0 |
| przemienne | 6 | 0,743 | -0,08 | -17,5 | -24,3 |
| obrotowe | 10 | 0,593 | 0,35 | -39,8 | -0,3 |
| trwałe + obrotowe | 12 (w odczycie) | 0,853 | | 71,8 | 18,9 |
| trwałe + przemienne | 12 (w odczycie) | 0,888 | | 79,0 | -12,8 |
| wszystkie trzy rodziny | 18 (w odczycie) | **0,893** | | **79,6** | **30,1** |

**Odczyt.** (1) Osobno liczą się tylko mody trwałe, ale razem pozostałe dwie rodziny dokładają wyraźnie: odróżnienie narzędzia od tła rośnie z 0,834 do 0,893, wydłużenie ramki z 71 do 80 procent, a wielkość ramki z zera do 30 procent. Symetryzacja kasowała więc informację, nie tylko szum. (2) Mody przemienne same nie przeżywają przesunięcia (-0,08), a mimo to dokładają najwięcej do odróżnienia narzędzia; zmiana znaku po przesunięciu to sygnał o strukturze narzędzia wzdłuż jego osi. (3) Kąt obrotu modów zespolonych jest znikomy (okres około 256 przesunięć, w pliku JSON od 219 do 256), więc „obrót" to prawie zdegenerowana para modów rzeczywistych. (4) Zwrotu narzędzia nie niesie żadna rodzina (poniżej 2 procent wszędzie), więc ograniczenie nie leży w symetryzacji, tylko w uśrednianiu tokenów po całym wnętrzu ramki. Konsekwencja: kod tożsamości dla K5 przeliczyć na wszystkich trzech rodzinach (18 liczb zamiast 8) przed etapem 2. Dowód: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_widmo_tozsamosci.json`.

**Zarzut autora 2: czy kwadrat siatki pamięta narzędzie, które przez niego przeszło** (`slad_po_przejsciu.py`, sekcja 4l dokumentu 38). Pomiar bez uczenia: podobieństwo tokenu do wzorca narzędzia w kwadratach, gdzie narzędzia teraz nie ma, wobec kwadratów nietkniętych przez osiem pomiarów; kontrolą jest ten sam rachunek w przyszłość, czyli kwadraty, w których narzędzie dopiero będzie (zbiory rozłączne: „było i nie wróci" wobec „będzie, a nie było"); etykiety wyłącznie do oceny.

| Odstęp | Ślad po przejściu (AUC) | Zapowiedź przed przyjściem (AUC) | liczba próbek |
|---|---|---|---|
| 3 klatki | 0,899 | 0,902 | 2488 |
| 6 klatek | 0,885 | 0,893 | 3928 |
| 9 klatek | 0,874 | 0,881 | 5021 |
| 15 klatek | 0,858 | 0,868 | 6265 |
| 24 klatki | 0,851 | 0,867 | 7488 |

Kwadrat, przez który narzędzie przeszło, wygląda potem „narzędziowo" bardzo wyraźnie (0,899 po trzech klatkach), ale dokładnie tak samo wygląda, zanim narzędzie tam dotrze (0,902). Efekt jest symetryczny w czasie, więc nie jest pamięcią kwadratu, tylko sąsiedztwem toru narzędzia: pole widzenia tokenu sięga poza ramkę. Zgadza się to z ustaleniem U14 projektu grantowego (jądro pamięci na cząstkach Eulera miało głębokość jednego kroku i tłumaczyło od 5 do 6 procent przyrostu). Dowód: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_slad_po_przejsciu.json`.

### 1.19 Niezmienniki wycinka narzędzia klatka po klatce, przegląd siatek i deep Koopman (22.09, `53d80455`, `2a1c978f`, `f821375e`, `743a3684`, `5b699ac5`)

**Pomysł autora.** Zamiast parować pojedyncze tokeny przy przesunięciu, opisać cały wycinek w ramce narzędzia i pytać, co w nim nie zmienia się z klatki na klatkę; potem porównać z niezmiennikami liczonymi po przesunięciu.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\niezmiennik_czasowy.py`). Tokeny leżące w prawdziwej ramce przepisano na stałą siatkę 3 na 3 we współrzędnych względnych ramki (opis nie zależy od wielkości ani położenia ramki), zredukowano do 48 wymiarów (90,3 procent wariancji) i dopasowano operator `x(t+1) = x(t) K` bez symetryzacji, osobno w każdym filmie. Dwie sesje policzyły to niezależnie i otrzymały identyczne liczby (pliki `REPORT_niezmiennik_czasowy_klipy.json` z `5b699ac5` i `REPORT_niezmiennik_czasowy_gesty.json` z `53d80455`).

Zestaw gęsty: 16 klipów, 4698 klatek z narzędziem, 4570 par klatka do następnej (odstęp do 4 klatek przy próbkowaniu co 3).

| Wielkość | Moduły ośmiu najtrwalszych modów |
|---|---|
| operator czasowy | 0,989 0,986 0,972 0,967 0,949 0,935 0,926 0,909 |
| podłoga: czas potasowany w obrębie filmu | 0,966 0,951 0,890 0,839 0,812 0,754 0,688 0,675 |
| podłoga: pary łączące różne filmy | 0,104 0,104 0,101 0,101 0,097 0,097 0,096 0,096 |

Niezmienniki czytają wydłużenie prawdziwej ramki w 88,1 procent, wielkość ramki w 17,8, położenie w kadrze w -0,8 (czyli wcale); 74 procent ich zmienności to różnice między filmami. Kosinusy główne (cosinusy kątów między dwiema podprzestrzeniami; 1 oznacza te same kierunki) z modami operatora translacyjnego: 0,95, 0,90, 0,87, 0,83, potem 0,54, 0,46, 0,37, 0,10. Dwie niezależne procedury trafiają w te same cztery pierwsze wektory latentu, co przemawia za tym, że są one własnością reprezentacji. Ale podłoga z potasowanym czasem jest wysoka (0,966 wobec 0,989): prawie cała trwałość bierze się z bezwładu latentu (wycinek w ogóle wolno się zmienia), jak wcześniej w projekcie grantowym.

Pula ucząca (10 637 klatek, 118 filmów; para tylko przy odstępie najwyżej 6 klatek, stąd 1809 par, bo pula jest próbkowana średnio co 20 klatek):

| Wielkość | Moduły ośmiu najtrwalszych modów |
|---|---|
| operator czasowy | 0,964 0,964 0,937 0,923 0,911 0,894 0,863 0,857 |
| podłoga: czas potasowany | 0,841 0,792 0,779 0,750 0,675 0,675 0,629 0,583 |
| podłoga: pary między filmami | 0,069 0,069 0,069 0,069 0,066 0,066 0,065 0,065 |

Tu niezmienniki czytają wydłużenie w 85,7 procent, wielkość w 41,4, ale też położenie w kadrze w 44,0, czego operator translacyjny z konstrukcji nie robi; zmienność między filmami 64 procent; kosinusy z modami translacyjnymi 0,92, 0,84, 0,67, 0,61, potem poniżej 0,55. Nadwyżka nad potasowanym czasem jest na puli większa (0,964 wobec 0,841), bo odstępy są większe i sam bezwład nie wystarcza. Opis wycinka klatka po klatce miesza tożsamość narzędzia z jego ruchem po kadrze, a operator translacyjny je rozdziela.

**Przegląd siatek** (pytanie autora, czemu 3 na 3, skoro w ramce mieści się około 25 tokenów; pula, te same 1809 par):

| Siatka | Najtrwalszy mod | Podłoga (czas potasowany) | Wydłużenie | Położenie w kadrze | Kosinusy z translacyjnymi (pierwsze dwa) |
|---|---|---|---|---|---|
| 2 na 2 | 0,961 | 0,817 | 85,9 | 37,3 | 0,90 i 0,87 |
| 3 na 3 | 0,964 | 0,841 | 85,7 | 44,0 | 0,92 i 0,84 |
| 4 na 4 | 0,965 | 0,845 | 86,1 | 43,0 | 0,91 i 0,84 |
| 5 na 5 | 0,965 | 0,840 | 85,6 | 29,4 | 0,89 i 0,77 |

Gęstsza siatka nic nie dokłada: najtrwalszy mod zmienia się o 0,004, wydłużenie o pół punktu, a udział wariancji w 48 wymiarach spada z 91 do 75 procent, bo przy 5 na 5 na komórkę przypada jeden token i opis kopiuje jego szum. Zostaje 3 na 3, z 4 na 4 jako kontrolą. Dowody: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_niezmiennik_czasowy_gesty.json`, `REPORT_niezmiennik_czasowy_klipy.json`, `REPORT_niezmiennik_czasowy_pula.json` i warianty `_c2`, `_c4`, `_c5` w katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\`.

**Deep Koopman na opisie wycinka** (`deep_koopman_ramki.py`, sekcja „4n" dokumentu 38; liczone lokalnie na RTX 3070). Deep Koopman to autoenkoder Koopmana: enkoder (sieć z warstw w pełni połączonych) podnosi opis wycinka do 8, 16 albo 32 liczb, w których krok klatki ma być mnożeniem przez jedną macierz, a dekoder wraca do opisu, więc podniesienie nie może zgubić treści. Podział po filmach, wczesne zatrzymanie (przerwanie uczenia, gdy wynik na osobnych filmach przestaje się poprawiać), trzy ziarna, zestaw gęsty (4570 par).

| Model | Umiejętność nad „nic się nie zmienia" (procent) | Podłoga: uczenie na potasowanym czasie | Wydłużenie ramki | Kosinusy z modami liniowymi |
|---|---|---|---|---|
| operator liniowy (odniesienie) | -0,7 | | 88,1 | |
| deep, podniesienie 8 | -58,2 | -57,8 | -27,8 | 0,95 i 0,86 |
| deep, podniesienie 16 | -61,4 | -56,7 | 37,9 | 0,97 i 0,95 |
| deep, podniesienie 32 | -45,5 | -54,1 | 68,1 | 0,98 i 0,94 |

Żaden model nie bije przewidywania „nic się nie zmienia" na nowych filmach (przy odstępie 3 klatek kopiowanie poprzedniej klatki jest mocnym punktem odniesienia; to ta sama pułapka bezwładu co na latencie DINOv2 w projekcie grantowym). Deep nie odróżnia się od własnej podłogi (model uczony na parach bez kolejności czasu wypada tak samo), ale znajduje te same kierunki co operator liniowy (kosinusy 0,98 i 0,94 przy podniesieniu 32). Odczyt wydłużenia jest przy deep gorszy (68 wobec 88 procent) i silnie zależy od rozmiaru podniesienia, co przy 16 klipach wygląda na przeuczenie. Część modów deep ma moduł powyżej jedynki, czyli wielkości rosnące z klatki na klatkę. Do etapu 2 zostaje operator liniowy; deep Koopman ma wrócić dopiero przy dłuższych odcinkach albo z wymuszeniem (ruch kamery jako wejście). Dowód: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_deep_koopman_gesty.json`.

### 1.20 Dokument 42: teoria domknięcia przestrzeni stanów narzędzia a dobór klatek (22.09, `5b699ac5`, poprawka w `e5916f4c`)

**Pytanie autora** (cytat z dokumentu): „jeżeli w latencie udało się odseparować wektory odpowiedzialne za położenie, tożsamość, przesunięcie i tło, to czy da się z tego zbudować pipeline doboru klatek; czy da się zmierzyć, ile brakuje do domknięcia modelu narzędzia i jego przestrzeni stanu; czy da się sterować teorią Mori-Zwanziga, Koopmanem i Langevinem, jądrem pamięci i szumem, i tak szukać próbek poprawiających model". Odpowiedzią jest analiza teoretyczna bez nowych pomiarów i treningu: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md` (opisuję wersję z 22.09; 23.09 dokument dalej zmieniano w `66dd2c7c`).

**Treść.**
- Rozdzielenie jest częściowe: tożsamość to osiem liczb z wolnych modów operatora translacyjnego (odczytuje wydłużenie w 86 procentach), położenie to środek ramki z głowy detektora, przesunięcie to definicja kroku operatora, a tło nie jest rozdzielone (tokeny z wnętrza ramki wchodzą do kodu razem z narzędziem).
- Ramy formalne: przesunięcie narzędzia o wektor `d` indukuje odwzorowanie `S_d`, a operator Koopmana działa na funkcjach stanu, `(T_d f)(z) = f(S_d z)`. Niezmienniki (`T_d f = f`) to funkcje na przestrzeni ilorazowej „latent podzielony przez ruch"; latent tokenu rozkłada się na część niezmienniczą (tożsamość, kształt, tło, operacja), ekwiwariantną (poza: położenie, kąt, skala) i resztę (szum, błąd ramki, rozmycie). Wniosek formalny zapisany tego dnia: dla rodziny przemiennej niezmiennikiem jest kwadrat, dla obrotowej suma kwadratów pary, więc rodziny te powinny wejść do kodu jako wielkości kwadratowe (przewidywanie obalone tego samego dnia, sekcja 1.21).
- Trzy nierównoważne znaczenia „domknięcia", każde z estymatorem bez etykiet: grupowe (czy złożenie dwóch małych przesunięć daje jedno duże: `D1 = ||T_a T_b - T_(a+b)|| / ||T_(a+b)||`, lokalnie `r(x) = |φ(x') - λ φ(x)| / (|φ(x)| + ε)`); wymiarowe (efektywny wymiar części ekwiwariantnej powinien wynosić około 4, bo tyle stopni swobody ma poza narzędzia w kadrze: dwa przesunięcia, obrót i skala; estymator `d_eff = (Σ s_i)^2 / Σ s_i^2`); Markowa (formalizm Mori-Zwanziga: rzutowanie pełnej dynamiki na niepełny zestaw współrzędnych daje równanie `z(t+1) = A z(t) + Σ_k K(k) z(t-k) + ξ(t)` z jądrem pamięci `K` i szumem `ξ`, związanymi twierdzeniem fluktuacyjno-dyssypacyjnym; miara deficytu `Δ_pamięci = R2(stan + historia) - R2(sam stan)` poza grupą dopasowania). Wielkość jądra pamięci mierzy, ile stopni swobody latent pomija, a szum, ile z tego jest nieusuwalne.
- Zlokalizowana dziura: zwrot narzędzia jest niezmienniczy względem przesunięcia, więc powinien być w kodzie, a wychodzi zero. Powód jest rachunkowy: kod klatki to średnia po tokenach, `kod = (1/N) Σ_i φ(z_i)`, symetryczna względem permutacji tokenów, która kasuje każdą funkcję nieparzystą wzdłuż osi narzędzia. Poprawka: moment pierwszy `kod₁ = (1/N) Σ_i u_i φ(z_i)`, gdzie `u_i` to położenie tokenu wzdłuż osi ramki od -1 do +1. To samo dotyczy tła: kod jest mieszaniną `α · kod_narzędzia + (1 - α) · kod_tła`, więc zmienność udziału `α` wchodzi jako fałszywa zmienność stanu; poprawka bez etykiet to miękka maska (ważenie tokenów podobieństwem do wzorca narzędzia).
- Trzy reguły doboru: A, defekt niezmienniczości `r(x)` (ryzyko: może zapaść się do ramienia pewności; test: korelacja z pewnością powyżej 0,6 odrzuca regułę bez treningu); B, informacja o parametrach ważnych dla zadania, czyli deficyt odczytu pozy i tożsamości (w odróżnieniu od D-optymalnego, które maksymalizuje informację o dynamice); C, pokrycie w przestrzeni ilorazowej, czyli K5 z uzasadnieniem formalnym. Naturalną jednostką doboru może być obszar przestrzeni stanów, a nie klatka.
- Dobór jako sterowanie: w czasie treningu wymuszeniem jest zbiór etykiet. Osiągalność (gramian sterowalności: kierunki, w których jest mały, są nieosiągalne przy dowolnej liczbie próbek, więc trzeba zmienić architekturę albo stratę), sufit z szumu (32 procent chybień w szybkim ruchu w każdym ramieniu, usuwalne 23 procent) i obciążenie w pętli zamkniętej, na które lekarstwem jest składnik wzbudzenia, czyli domieszka klatek losowych albo równomiernych (to, co robi K5b).
- Sufit: usuwalne 23 procent chybień, precyzja 7 procent w górnym decylu i najlepszy kontrast 0,0068 AP oznaczają, że reguły A, B i C grają o pojedyncze tysięczne AP. Główny wniosek: struktura niezmiennicza jest znacznie mocniejsza jako człon straty (działa na wszystkich klatkach, także nieoznaczonych) niż jako reguła doboru.
- Sześć falsyfikowalnych przewidywań, od najtańszego: moment pierwszy odzyskuje zwrot; niezmienniki kwadratowe biją liniowe (obalone tego samego dnia, poprawka wpisana w dokument w `e5916f4c`); miękka maska obniża udział zmienności między filmami poniżej 58 procent; deficyt pamięci na współrzędnych niezmienniczych większy niż na surowym latencie; defekt niezmienniczości nie jest przebraną pewnością (korelacja poniżej 0,6); efektywny wymiar części ekwiwariantnej zbliża się do 4 z treningiem. Trenowanie ramion A, B lub C ma sens dopiero, gdy wyjdą przynajmniej trzy pierwsze.

### 1.21 Czy tożsamość siedzi w części rzeczywistej modów: rodziny, mod po modzie i drugi punkt kontrolny (22.09, `e5916f4c`, `4dbcb5be`, `4bfa6fe3`, `87a57eac`)

**Hipoteza autora.** „Tożsamość powinna być zakodowana w części rzeczywistej modów Koopmana"; sprawdzić także mod po modzie oraz osobno mody o dodatniej i ujemnej części rzeczywistej.

**Poziom rodzin** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\czesc_rzeczywista_a_tozsamosc.py`, druga sekcja oznaczona „4m" w dokumencie 38). Dla każdej rodziny modów operatora translacyjnego osobno policzono część rzeczywistą, urojoną, moduł pary i kwadrat. Miary: odróżnienie tokenu narzędzia od tokenu tła (uczenie i ocena na różnych filmach, prawdziwe ramki tylko do oceny), przeżycie przesunięcia i odczyt wydłużenia ramki. Po pięć modów z rodziny, 34 165 par, nagranie L0 na 16 gęstych klipach.

| Rodzina | Postać | Odróżnienie (AUC) | Przeżycie | Wydłużenie (procent) |
|---|---|---|---|---|
| trwałe | Re | **0,825** | 0,65 | 69,1 |
| trwałe | Re do kwadratu | 0,502 | 0,47 | 44,9 |
| przemienne | Re | 0,686 | -0,09 | -16,3 |
| przemienne | Re do kwadratu | **0,723** | 0,03 | -38,2 |
| obrotowe | Re | 0,809 | 0,20 | -15,7 |
| obrotowe | Im | **0,650** | 0,20 | -17,2 |
| obrotowe | moduł pary | **0,844** | 0,14 | -24,0 |
| obrotowe | Re i Im razem | 0,794 | 0,20 | -14,0 |

Zestawy przy równej liczbie kolumn (żeby przewaga nie brała się z liczby cech):

| Zestaw | Kolumn | Odróżnienie | Wydłużenie (procent) |
|---|---|---|---|
| same trwałe Re | 5 | 0,825 | 69,1 |
| trwałe Re + obrotowe moduł | 10 | 0,879 | 68,3 |
| **same części rzeczywiste trzech rodzin** | 15 | **0,887** | **78,3** |
| same części urojone (tylko obrotowe) | 5 | 0,650 | -17,2 |
| trwałe Re + obrotowe Re i Im | 15 | 0,886 | 73,7 |
| trwałe Re + obrotowe moduł + przemienne kwadrat | 15 | 0,882 | 66,7 |
| trwałe Re + obrotowe Re i Im + przemienne kwadrat | 20 | 0,893 | 73,1 |
| wszystko liniowo, jak dotąd | 20 | **0,902** | **79,3** |

**Odczyt.** (1) Hipoteza potwierdzona co do kierunku: w rodzinie obrotowej część rzeczywista niesie wyraźnie więcej tożsamości niż urojona (0,809 wobec 0,650), choć urojona nie jest szumem (dołożona do rzeczywistej podnosi wydłużenie z 68,3 na 73,7 procent). (2) Najwięcej w rodzinie obrotowej niesie moduł pary (0,844), zgodnie z rachunkiem, że dla pary obrotowej niezmiennikiem przesunięcia jest dopiero długość. (3) Przewidywanie dokumentu 42, że postaci kwadratowe pobiją liniowe, jest przy równej liczbie kolumn obalone (0,893 wobec 0,902; wydłużenie 73,1 wobec 79,3): odczyt potrzebuje cechy użytecznej, a nie ściśle niezmienniczej, a moduł wyrzuca fazę niosącą pozę; poprawkę wpisano do dokumentu 42 w miejscu przewidywania. (4) Wyjątkiem jest rodzina przemienna, gdzie kwadrat bije wartość surową (0,723 wobec 0,686), bo wartość zmienia znak po kroku. (5) Moduł pary obrotowej przeżywa przesunięcie gorzej niż jej część rzeczywista (0,14 wobec 0,20), co wraz ze znikomym kątem wskazuje, że te pary są raczej numeryczne niż fizyczne. (6) Same części rzeczywiste trzech rodzin (15 liczb) prawie dorównują pełnemu zestawowi (0,887 wobec 0,902 i 78,3 wobec 79,3 procent), więc na jedną liczbę kodu są wydajniejsze i hipoteza broni się w wersji praktycznej. Uwaga o słowie „tożsamość": mierzone jest odróżnienie tokenu na narzędziu od tokenu tła oraz odczyt kształtu ramki; przy jednej klasie narzędzia tożsamości w sensie „które to narzędzie" nie da się tu zmierzyć. Wniosek praktyczny: zostawić postać liniową ze wszystkimi trzema rodzinami (20 liczb), a przy potrzebie krótkiego kodu wziąć same części rzeczywiste (15 liczb) kosztem 0,015 odróżnienia; nie przechodzić na moduły. Dowód: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_czesc_rzeczywista.json`.

**Poziom pojedynczych modów** (`mod_po_modzie.py`, druga sekcja oznaczona „4n" w dokumencie 38). Każdy z 32 modów wchodzi osobno jako jedna kolumna cech; nagranie L0, 34 165 par.

| Mod | Re wartości własnej | Odróżnienie narzędzia od tła (AUC) | Przeżycie przesunięcia | Odczyt wydłużenia (procent) |
|---|---|---|---|---|
| 0 (największa Re) | 0,786 | 0,616 | **0,77** | **58,6** |
| 4 | 0,523 | 0,739 | 0,51 | -23,3 |
| 14 | 0,155 | 0,629 | 0,17 | -42,1 |
| 28 (najlepszy) | 0,022 | **0,772** | 0,02 | -11,7 |
| 31 | 0,001 | 0,756 | 0,00 | -21,0 |
| 15 (najbardziej ujemny) | -0,148 | 0,525 | -0,15 | -20,6 |

Korelacje rangowe po 32 modach: część rzeczywista a odróżnienie **-0,04**; moduł wartości własnej a odróżnienie -0,23; tylko mody dodatnie, Re a odróżnienie -0,12; tylko mody ujemne, siła zmiany znaku a odróżnienie -0,62; część rzeczywista a odczyt wydłużenia -0,31. Średnie odróżnienie: mody dodatnie 0,569 (24 mody), ujemne 0,566 (8 modów).

**Odczyt.** (1) Hipoteza jako reguła szeregowania jest obalona: korelacja części rzeczywistej z odróżnieniem narzędzia od tła wynosi zero, najlepszy pojedynczy mod (0,772) ma część rzeczywistą 0,022, a mod o największej części rzeczywistej daje tylko 0,616. (2) Hipoteza trafia za to w co innego: mod o największej części rzeczywistej najlepiej przeżywa przesunięcie (0,77) i jako jedyny pojedynczy mod czyta kształt narzędzia dodatnio (58,6 procent). Moduł wartości własnej mierzy trwałość pod ruchem i idzie w parze z kształtem, a nie z odróżnianiem metalu od tkanki; ten pomiar pierwszy raz rozdziela te dwie własności. (3) Podział na dodatnie i ujemne nic nie daje (0,569 wobec 0,566), a wewnątrz ujemnych zależność biegnie odwrotnie (im silniejsza zmiana znaku, tym mniej tożsamości). (4) Siła zestawów z poziomu rodzin bierze się z różnorodności modów, nie z ich trwałości.

**Powtórzenie na drugim punkcie kontrolnym** (uwaga autora w trakcie rundy: wszystko dotąd liczono na L0, sieci słabo wytrenowanej po 500 klatkach; plik `REPORT_mod_po_modzie_b2500.json`):

| Wielkość | L0, 500 klatek | 2500 klatek |
|---|---|---|
| mod o największej części rzeczywistej: Re | 0,786 | 0,796 |
| jego przeżycie przesunięcia | 0,77 | 0,78 |
| jego odczyt wydłużenia | 58,6 procent | 58,4 |
| liczba modów czytających kształt powyżej zera | 1 (ten sam) | 1 (ten sam) |
| korelacja części rzeczywistej z odróżnieniem | -0,04 | -0,02 |
| najlepszy pojedynczy mod: Re i AUC | 0,022 i 0,772 | 0,512 i 0,667 |

Wniosek główny nie zależy od punktu kontrolnego. Zmienia się jedno: mod o wartości własnej bliskiej zeru, który na L0 był najlepszy, po douczeniu znika, więc tamten wynik to własność sieci po 500 klatkach, a nie reprezentacji. Wniosek praktyczny: do kodu tożsamości nie wybierać modów po części rzeczywistej, tylko dwiema listami: trwałość i kształt biorą mody o dużej części rzeczywistej, a odróżnienie narzędzia od tła mody o dużym odróżnieniu, niezależnie od wartości własnej. Dowody: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_mod_po_modzie.json` i `REPORT_mod_po_modzie_b2500.json`.

### 1.22 Które mody niosą stan narzędzia, kształt z niezmienników, trzy drogi jednym metrem, rozkład przez odejmowanie i statyka (22.09, `27342b29`, `b5be0623`, `4a10f6e1`, `343af3f7`)

Te pomiary wykonywała równolegle do sekcji 1.21 druga sesja pracująca w tym samym repozytorium; część liczono lokalnie po ściągnięciu nagrania puli (731 MB), bo krótkie zadanie CPU na Edenie odbił limit konta (`AssocMaxSubmitJobLimit`), a przygotowany skrypt `run_niezm.slurm` nie został zgłoszony.

**Kształt z niezmienników: nie** (`ksztalt_z_niezmiennikow.py`, sekcja „4o" dokumentu 38; zestaw gęsty, 4636 klatek z ramką, operator czasowy na opisie wycinka 3 na 3, uczenie i ocena na różnych filmach). Liniowe odtwarzanie wycinka obrazu (32 na 24 piksele w skali szarości) z szesnastu liczb kodu daje R2 -0,17, czyli gorzej niż podanie jednej uśrednionej klatki; pełny opis (48 współrzędnych) daje -0,29. Na rysunku widać ukośną smugę, ale to przeciętne narzędzie, nie to z danej klatki. Te same szesnaście liczb odtwarza tylko 12 procent opisu tokenów, a szesnaście zwykłych składowych głównych 63 procent. Rysunek i liczby: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\rysunki_k5\REPORT_ksztalt_z_niezmiennikow.png` i `.json` (poza gitem).

**Które mody niosą stan narzędzia** (`ktore_mody_nioso.py`):

| Grupa modów | Ile modów | Wydłużenie (procent) | Wielkość | Położenie w kadrze |
|---|---|---|---|---|
| część rzeczywista bliżej 1 niż o 0,15 | 2 | 42,6 | 0,6 | -9,6 |
| część rzeczywista dodatnia, dalej od 1 | 37 | **69,5** | 12,6 | 29,1 |
| część rzeczywista ujemna | 9 | 37,8 | 12,9 | **42,2** |
| mody z częścią urojoną (obrotowe) | 42 | **74,7** | **33,7** | 34,6 |
| osiem pierwszych (dotychczasowy kod) | 8 | 50,0 | 15,7 | -16,0 |

Najlepszy pojedynczy mod to ten o części rzeczywistej 0,956 (44,6 procent wydłużenia sam jeden), ale hipoteza „stan siedzi w modach najbliższych jedynki" jest trafna tylko w połowie: 37 wolniejszych, ale wciąż dodatnich modów daje razem 69,5, a mody o ujemnej części rzeczywistej niosą przede wszystkim położenie (42,2 procent). Obcięcie kodu do ośmiu najwolniejszych modów wyrzuca połowę informacji o stanie narzędzia; kod na etap 2 należy budować z modów dodatnich i obrotowych, a ujemne trzymać osobno jako współrzędne położenia.

**Trzy drogi do wektora tożsamości jednym metrem** (`porownanie_tozsamosci.py`, sekcja „4p" dokumentu 38). A: przesunięcie narzędzia (operator translacyjny na parach tokenów, kod klatki to średnia po tokenach w ramce detektora); B: widmo czasu na wycinku (operator na opisie całego wycinka, krok to następna klatka); C: deep Koopman; D: wszystkie trzy razem (24 liczby). Na życzenie autora każda droga dostaje osiem liczb, zawsze tylko część rzeczywistą współrzędnych i tylko mody o dodatniej części rzeczywistej. Miary (etykiety wyłącznie do oceny, uczenie i ocena na różnych filmach): odczyt wydłużenia i wielkości prawdziwej ramki (stan narzędzia, im więcej, tym lepiej), odczyt położenia w kadrze i udział zmienności między filmami (zanieczyszczenia, im mniej, tym lepiej), oraz wynik łączny zdefiniowany w skrypcie jako wydłużenie plus wielkość minus |położenie| minus połowa udziału między filmami. Kontrola: osiem zwykłych składowych głównych tego samego opisu. Pula ucząca, 9441 klatek, 118 filmów:

| Metoda | Wydłużenie | Wielkość | Położenie (zanieczyszczenie) | Między filmami | Łącznie |
|---|---|---|---|---|---|
| A: przesunięcie narzędzia | 87,2 | 51,3 | **4,1** | 69 | 100,0 |
| B: widmo czasu na wycinku | 85,7 | 39,9 | 40,5 | 65 | 52,6 |
| C: deep Koopman | 9,8 | 38,7 | 12,4 | 64 | 4,2 |
| D: wszystkie trzy razem | 89,3 | 62,1 | 46,5 | 67 | 71,2 |
| kontrola: osiem składowych głównych | 88,0 | 51,8 | 7,7 | **54** | **104,9** |

Na zestawie gęstym (16 klipów) uszeregowanie jest inne (B wygrywa: wydłużenie 89,5 przy położeniu 3,4, a A ma 73,0), ale przy szesnastu filmach podział na uczenie i ocenę jest za słaby, więc liczy się wynik na puli. Odczyt: (1) żadna z trzech dróg Koopmana nie bije zwykłych składowych głównych, co zapisano jako wynik negatywny dla całej rodziny; (2) wśród dróg Koopmana najczystsza jest A, która prawie nie niesie położenia (4,1 procent), bo z konstrukcji paruje ten sam kawałek narzędzia w dwóch miejscach kadru; (3) B przecieka położeniem (40,5 procent); (4) deep Koopman jest najsłabszy i ma mody o module powyżej jedynki; (5) sklejenie D poprawia tylko wydłużenie, a zanieczyszczenie położeniem rośnie do 46,5.

**Rozkład przez odejmowanie** (pomysł autora, `4a10f6e1`): od każdej współrzędnej B odjęto tę jej część, którą da się odtworzyć z A, i odwrotnie.

| Wariant | Wydłużenie | Wielkość | Położenie | Między filmami |
|---|---|---|---|---|
| B minus A (co zostaje po odjęciu tożsamości) | -2,7 | 1,4 | **28,5** | 56 |
| A minus B (tożsamość po odjęciu widma czasu) | -0,6 | 12,9 | -1,3 | 40 |
| kontrola: składowe główne minus A | -4,9 | 5,1 | 8,7 | 45 |

Reszta B po odjęciu A niesie położenie narzędzia (28,5 procent) i już nie niesie jego stanu (poniżej 3 procent); po odjęciu B z A nie zostaje nic, więc cała tożsamość z A siedzi już w B. Stąd rozkład latentu narzędzia na dwie części liczone bez etykiet: tożsamość to kod z przesunięcia (A), położenie to reszta widma czasu po odjęciu A; pierwsza nadaje się do pokrycia stanów narzędzia, druga do pokrycia jego ruchu po kadrze. Wniosek dla etapu 2: kod tożsamości budować metodą A, z ramieniem odniesienia ze składowych głównych, bez którego żaden zysk z Koopmana nie byłby do obrony. Dowody: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_porownanie_tozsamosci_pula.json` i `REPORT_porownanie_tozsamosci_gesty.json`.

**Statyka w modach Koopmana** (`statyka_w_koopmanie.py`, sekcja „4r" dokumentu 38, `343af3f7`). Zarzut autora: składowe główne mówią tylko, gdzie dane najbardziej się różnią, i nie wiadomo, co niosą, a mody Koopmana są uszeregowane po trwałości, więc to, co przez cały film stoi (narzędzie jako przedmiot, oko, światło, kadr), musi siedzieć w najtrwalszych. Sprawdzenie bez etykiet: każda współrzędna rozkłada się na średnią w danym filmie (statyka) i odchylenie od niej wewnątrz filmu (dynamika). Pula ucząca, 10 109 klatek, 118 filmów; statyka to 31 procent zmienności całego opisu wycinka.

| Zestaw ośmiu kierunków | Udział statyki w tych kierunkach | Trwałość dynamiki | Zatrzymana statyka | Zatrzymana dynamika |
|---|---|---|---|---|
| mody Koopmana (największa część rzeczywista) | **64 procent** | 0,77 | **33** | 9 |
| składowe główne (najsilniejsze) | 52 | 0,70 | 28 | 12 |
| kierunki losowe | 30 | 0,53 | 16 | 17 |

Zależność jest monotoniczna po części rzeczywistej: mod o Re 0,964 ma 73 procent statyki, ósmy o Re 0,857 już 47. Na zestawie gęstym zależność jest ostrzejsza (90 procent statyki w pierwszym modzie wobec średnio 60 procent w składowych głównych). Teza autora potwierdzona: szeregowanie po części rzeczywistej porządkuje kierunki według tego, ile w nich statyki filmu, czego szeregowanie po wariancji nie robi, a składowe główne mieszają statykę z dynamiką. Na punkcie po 2500 klatkach (zestaw gęsty) obraz jest prawie ten sam (statyka w pierwszym modzie 87 wobec 90 procent, trwałość 0,93 wobec 0,91), więc trening tej własności nie zmienia. Zastrzeżenie: „między filmami" obejmuje zarówno narzędzie jako przedmiot, jak i oko, światło i kadr, i tego pomiar nie rozdziela. Dowody: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_statyka_w_koopmanie_pula.json`, `REPORT_statyka_w_koopmanie_gesty_b0500.json`, `REPORT_statyka_w_koopmanie_gesty_b2500.json`.

### 1.23 Równoległa praca innych sesji w tym samym repozytorium (21-22.09, pliki niewersjonowane w git)

Tę część opisuję skrótowo, bo nie należy do listy commitów etapu; pliki leżą w repozytorium jako nieśledzone i były w nim obecne w tych samych dniach.

- **Plan 39 równoległej sesji** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\39_PLAN_SELEKCJI_KLATEK_RFDETR_I_ROZWOJU_JEPA.md`, 21.09, 14:24; nie mylić z wersjonowanym dokumentem 39 z macierzą etapu 1). Zapisuje decyzję autora: Koopman i model z pamięcią inspirowany uogólnionym równaniem Langevina mają służyć do wyboru już oznaczonych klatek, a docelowy RF-DETR nadal analizuje pojedynczy obraz; selekcja i diagnostyka działają podczas przygotowania danych lub uczenia, nie podczas wnioskowania. Głównym punktem odniesienia ma tam być strata detektora plus różnorodność. Na ten plan powołuje się dopisek na początku dokumentu 38.
- **Audyt 40** (`40_TOZSAMOSC_NARZEDZIA_I_ROLA_ROZSZERZONEGO_LANGEVINA.md`, 22.09, 12:46): opisuje dokładnie, czym był K1 (sekcja 1.8), koryguje opis K4 (sekcja 1.10) i uznaje najmocniejsze zdania dokumentu 39 za wykraczające poza eksperyment; rozdziela obecność, typ, ciągłość celu i egzemplarz narzędzia i stwierdza, że obecne dane pozwalają badać tylko obecność i lokalizację oraz ciągłość celu w klipie.
- **Dokument 41** (`41_CZASTKI_TOZSAMOSCI_NARZEDZIA_I_ROWNANIE_LANGEVINA.md`, 22.09, 12:49): uzasadnienie teoretyczne „które cząstki niosą tożsamość narzędzia" przez domknięcie dynamiki (twierdzenia Lean z projektu grantowego i Mori-Zwanzig: „pamięć jest ceną rzutowania") i protokół pomiaru po tokenach, bez nowych pomiarów.
- **Pilot `dobor_klatek_stan_sieci`** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_klatek_stan_sieci\`, 22.09; na Edenie `~/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci`). Cząstką jest pojedynczy token (24 na 24 na klatkę, 384 kanały rzutowane stałą projekcją Gaussa do 24 z zachowaniem indeksu kwadratu); do historii tokenów dopasowano Koopmana i pełne rozszerzone GLE z projektu grantowego (człony D, JP, W, Z i N z własną skalą ruchu i wykładnikami gamma; 24 wymiary zamiast 192, 8 kroków zamiast 56) na 57 073 klatkach uczących, trzy foldy po operacjach, po 20 000 trajektorii na fold. Błąd prognozy tokenu (MSE, błąd średniokwadratowy w kanałach znormalizowanych) w trzech foldach: model „nic się nie zmienia" od 0,410 do 0,414, Koopman od 0,369 do 0,373, GLE od 0,333 do 0,340. Ocena klatki to dodatnia część różnicy błędu odczytu narzędzie-tło z tokenów bieżących i z tokenów przewidzianych z historii (odczyt używa etykiet ramek, więc to dobór z oznaczonej puli). Z puli 10 637 klatek (w tym L0) wybrano 1000 dodatkowych; część wspólna z doborem po samym błędzie odczytu 509, z placebo 418, z losowaniem 79. Pięć treningów (ziarno 42, 10 epok, wspólny start COCO; zadania 1809371 ekstrakcja, 1809379 dopasowanie i wybór, 1809380 treningi, 1809382 diagnostyka punktów kontrolnych; zadanie 1809362 anulowane przed startem kodu przez błąd pobierania środowiska przy `--export`). Wynik (AP zwykłych wag ostatniej epoki na 12 043 klatkach walidacji; w nawiasie AP najlepszej epoki z dziennika treningu): Koopman-GLE 0,67317 (0,6886), losowanie 1500 0,65547 (0,6852), największy błąd odczytu 0,64597 (0,6756), placebo 0,65220 (0,6809), losowanie 3000 0,67122 (0,6951). Autorzy pilota zapisali spełnienie bramki ustalonej przed treningiem, ale też, że to jedno ziarno i brak dowodu statystycznego, a cała procedura z selektorem nie była szybsza (około 27,12 minuty wobec 20,38 dla losowych 3000). Z tabeli wynika, że przewaga nad losowaniem 1500 wynosi +0,0177 w ostatniej epoce, ale tylko +0,0034 w najlepszej epoce. Szczegóły: `WYNIKI_PILOT.md`, `WYNIKI_SELEKCJA.md`, `WYNIKI.json` w tym katalogu.
- **Kampania `dobor_cech_adaptacyjny`** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_cech_adaptacyjny\`): kod z 22.09 po południu, zadania 1809520 (testy) i od 1809521 do 1809535 (15 treningów: pięć ramion razy trzy ziarna, 10 epok, 1500 klatek wybieranych na nowo w każdej epoce) wykonane wieczorem 22.09, wyniki pobrane i opisane dopiero 23.09 (`WYNIKI_KAMPANII_2026-09-22.md`): losowanie 0,7131 ± 0,0042, GLISTER (dobór klatek, których gradient najbardziej zmniejsza stratę na osobnym zbiorze sterującym) 0,7072, ramiona z cechami od 0,6839 do 0,6941, w tym ramię z priorytetem Koopmana-GLE 0,6839; żadne ramię nie pobiło losowania. Te liczby nie są porównywalne z macierzą etapu 1 (inny protokół, około 7500 różnych klatek w uczeniu) i należą do opisu 23 września.

### 1.24 Krystalizacja latentu w trakcie uczenia (22.09, `7bb3484b`)

**Pytanie autora.** Jak zmienia się latent detektora podczas uczenia, czy się krystalizuje, czy można z niego odczytać, które klatki albo grupy klatek niosły informację lepiej poprawiającą model, i czy takie klatki da się wskazać z analizy koopmanowskiej, bezpośrednio albo pośrednio.

**Materiał i miejsce.** Nowy katalog eksperymentu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\` (podkatalogi `dane` i `wyniki`; opisuję stan z 22.09, bo 23.09 dopisano tam kolejne skrypty). Eksperyment nie uruchamia nowego doboru; korzysta z jedenastu eksportów latentu biegu `epochs_uniform_time_b1500_s42` (COCO plus epoki 2, 5, ..., 29) na 4800 klatkach paczki, krzywej jakości walidacyjnej tego biegu (szczyt w epoce 6, wartość 0,6869 według opisu sesji), 27 biegów macierzy etapu 1 i dwóch nagrań cząstek z punktu `ut_b0500_best`.

**Zegar treningu** (`krystalizacja.py`). Dla każdego punktu kontrolnego i czterech bloków latentu (`query`, `box_pool`, `global_pool` oraz `features`, czyli `query` sklejone z `global_pool`; redukcja do 32 wymiarów, 8 modów, 4800 klatek, 4784 par klatka do następnej): participation ratio; udział zmienności między filmami (statyka); liczba modów operatora kroku klatki o części rzeczywistej powyżej 0,9; trwałość, czyli średnia część rzeczywista wartości własnych ośmiu najwolniejszych modów; odczyt położenia, wielkości i wydłużenia narzędzia (uczenie i ocena na różnych filmach). Między sąsiednimi punktami: przesunięcie klatki po najlepszym obrocie, CKA i średnia korelacja kanoniczna (CCA, najwyższe korelacje osiągalne między kombinacjami liniowymi dwóch zestawów zmiennych). Wyniki bloku `query` (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_krystalizacja_query.json`):

| Punkt | AP walidacji | participation ratio | trwałość | modów z Re powyżej 0,9 | odczyt: położenie / wielkość / wydłużenie (procent) |
|---|---|---|---|---|---|
| COCO | | 6,52 | 0,790 | 1 | -86,7 / 7,7 / -39,6 |
| epoka 2 | 0,6650 | 8,57 | 0,949 | 7 | 62,2 / 52,4 / 92,2 |
| epoka 5 | 0,6867 | 9,60 | **0,952** | 8 | 60,3 / 48,6 / 93,9 |
| epoka 8 | 0,6837 | 9,75 | 0,946 | 7 | 48,8 / 53,5 / 91,3 |
| epoka 14 | 0,6676 | 11,66 | 0,940 | 7 | 52,8 / 53,1 / 89,5 |
| epoka 29 | 0,6561 | 16,42 | 0,933 | 6 | 59,2 / 51,1 / 91,3 |

W bloku `box_pool` przesunięcie klatki między sąsiednimi punktami spada z 1,05 (COCO do epoki 2) do 0,26 (epoka 26 do 29), a CKA po epoce 2 trzyma się od 0,82 do 0,87 i nigdy nie dochodzi do jedności. Korelacja trwałości z jakością walidacyjną wynosi od 0,74 (`box_pool`) przez 0,84 (`query`, `features`) do 0,89 (`global_pool`); w bloku `query` udział statyki koreluje z jakością na 0,94, a participation ratio -0,75.

**Odczyt.** Latent ustawia się po pierwszej epoce, a potem tylko dryfuje; pełnej krystalizacji w sensie zbieżności geometrii nie ma. Po szczycie jakości latent się rozlewa (liczba używanych kierunków w `query` rośnie z 8,57 do 16,42), a jakość spada. Trwałość najwolniejszych modów ma szczyt tam, gdzie jakość. Sprawdzono odporność: dla czterech bloków, trzech wymiarów redukcji i trzech liczb modów (36 ustawień) szczyt trwałości wypada w epoce 5, czyli w szczycie jakości, w 32 ustawieniach, a w pozostałych 4 w epoce 2. To kandydat na sygnał końca treningu liczony bez etykiet.

**Które klatki model traci** (`wklad_klatek.py`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_wklad_klatek.json`). Pokrycie ramki przewidzianej z prawdziwą policzono w każdym punkcie dla 4698 klatek z adnotacją. Po szczycie jakości (epoka 5 do 29) 12,0 procent klatek traci ponad 0,05 IoU, a 8,2 procent zyskuje. Próba przewidzenia z latentu trafności w szczycie i jej zmiany po szczycie (uczenie i ocena na różnych filmach) dała R2 w procentach: wolne mody -18,9 i -10,9; osiem składowych głównych 6,8 i -11,3; osiem kierunków losowych -5,1 i -3,3; cały latent -21,0 i -37,6. Wynik negatywny: R2 nigdzie nie przekracza 7 procent. Średnia pewność detekcji rośnie z 0,83 w epoce 5 do 0,93 w epoce 29 mimo spadającej jakości, a udział klatek o niskiej pewności zostaje poniżej 2,5 procent: model staje się pewniejszy siebie przy gorszych wynikach.

**Pokrycie filmu walidacyjnego a pudła** (`pokrycie_a_bledy.py`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_pokrycie_a_bledy.json`). Opis klatek walidacyjnych przepisano do bazy puli (zgodność baz 0,934). Dla czterech ramion (B1, B2, B3, K4) i dwunastu wspólnych filmów walidacyjnych policzono medianę odległości klatki do najbliższej wybranej klatki uczącej w czterech przestrzeniach (wolne mody, składowe główne, kierunki losowe, pełny opis). Surowa korelacja tej odległości z udziałem pudeł w przestrzeni wolnych modów wynosi 0,59 w pliku JSON (0,60 w opisie sesji), w pozostałych przestrzeniach 0,18, 0,25 i 0,16; korelacja rangowa wynosi zero (według opisu sesji i README, w JSON jej nie ma), bo cały efekt robi jeden film odstający (odległość 1,44, pudła 36,8 procent). Po zdjęciu średniej filmu i średniej ramienia korelacje wynoszą -0,12, +0,38, -0,29 i +0,02, czyli mają niespójne znaki. Wynik negatywny.

**Czy po samym zestawie klatek widać wynik** (`co_przewiduje_wynik.py`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_co_przewiduje_wynik.json`). Dla dziewięciu ramion policzono lukę pokrycia puli (p90 i maksimum), rozpiętość chmury wybranych klatek, liczbę filmów i udział szybkich klatek. Korelacja luki p90 z AP przy wszystkich ramionach: wolne mody -0,80, składowe główne -0,84, kierunki losowe -0,67; bez ramienia B0: -0,45, -0,40 i -0,58. Widać więc tylko ramię wyraźnie złe (kolejne klatki filmu), a przestrzeń Koopmana nie ma przewagi nad kierunkami losowymi.

### 1.25 Telemetria kampanii, strata walidacyjna i moment końca uczenia (22.09, `7bb3484b`, `7c3ddecf`; zadania Eden 1809553, 1809554, 1809555)

**Telemetria istniejących biegów.** W środowisku `~/venvs/rfdetr_nano_clean_1_10_1` (RF-DETR 1.10.1, TensorBoard 2.21.0) zrzucono wszystkie wartości liczbowe zapisywane przez TensorBoard (dziennik uczenia z wartościami co epokę) z 28 biegów: 27 z macierzy etapu 1 i biegu osi epok. Plik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\TELEMETRIA_stage1.json` (697 kB, 26 znaczników). Skrypt `telemetria_treningu.py` (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_telemetria.json`) ustalił: szczyt jakości wypada w epoce 8 (mediana, zakres od 4 do 16), a jakość w granicy 0,005 AP od szczytu (w opisie sesji „pół procenta") jest już w epoce 6 (mediana); od szczytu do końca 30 epok jakość spada średnio o 0,028 (w opisie sesji „około 0,03"); strata na zbiorze uczącym spada około pięciokrotnie (od 4,8 do 6,0 razy).

| Ramię | szczyt AP (mediana epoki) | epoka wejścia w granicę 0,005 od szczytu | spadek AP od szczytu do końca | strata ucząca: start / koniec |
|---|---|---|---|---|
| B0_consecutive | 16 | 10 | 0,013 | 6,21 / 1,06 |
| B1_random | 8 | 6 | 0,024 | 6,16 / 1,11 |
| B2_uniform_time | 7 | 6 | 0,029 | 6,20 / 1,10 |
| B3_coverage | 6 | 6 | 0,033 | 6,43 / 1,28 |
| B4_uncertainty | 8 | 6 | 0,033 | 6,69 / 1,34 |
| K1_langevin | 8 | 6 | 0,029 | 6,29 / 1,22 |
| K2_no_history | 8 | 7 | 0,031 | 6,30 / 1,21 |
| K3_placebo | 8 | 6 | 0,030 | 6,27 / 1,20 |
| K4_koopman_cover | 5 | 4 | 0,028 | 6,17 / 1,20 |
| bieg osi epok | 6 | 4 | 0,031 | 6,16 / 1,14 |

K4 dochodzi do szczytu najszybciej (epoka 5), B0 najpóźniej (16).

**Brak straty walidacyjnej.** Żaden bieg kampanii nie zapisywał straty na walidacji (wartości funkcji straty uczenia policzonej na klatkach walidacyjnych, standardowego sygnału przeuczenia): RF-DETR 1.10 ma ustawienie `compute_val_loss: "auto"`, które oznacza pominięcie, gdy nic jej nie pilnuje. Dopisano przełącznik `--val-loss` w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\train\train_rfdetr.py` i `run_with_validation.py`. Nie zmienia on przepisu optymalizacji; `compute_train_metrics` świadomie zostawiono wyłączone, bo zmienia ścieżkę normalizacji straty w kroku uczenia.

**Nowe biegi.** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_telemetria_epok.slurm` powtarza wariant `matrix_B2_uniform_time_b1500_sel42` na trzech ziarnach z pełną telemetrią: strata walidacyjna co epokę, wagi co dwie epoki (punkty po epokach 1, 3, ..., 29), latent każdego zachowanego punktu (po szesnaście eksportów na bieg), zrzut wartości do `TELEMETRIA.json`. Zadania 1809553, 1809554 i 1809555 wykonały się na węzłach dgx-1 i dgx-4 w około 70 minut każde; telemetria ma teraz 40 znaczników zamiast 26, w tym `val/loss` z rozbiciem na człony. Pułapka po drodze: pierwsze trzy zgłoszenia z `sbatch --export=ALL,ZMIENNA=...` scheduler anulował po sekundzie (`CANCELLED by 0`) bez żadnego dziennika, choć minimalne zadanie testowe z tymi samymi zasobami przeszło; rozwiązanie to wyeksportowanie zmiennych w powłoce przed `sbatch`. Według opisu sesji usunięto też `#SBATCH --nodelist=hopper`, żeby móc użyć partycji short (węzeł hopper zajmowała równoległa sesja); w wersjonowanym pliku z `7bb3484b` ta linia nadal jest.

**Wynik: kiedy sieć jest wytrenowana** (`kiedy_przestac.py`; pliki `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\biegi\telemetria_uniform_time_b1500_s{42,1337,2024}\TELEMETRIA.json`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\TRWALOSC_telemetria.json`, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_kiedy_przestac_query.json`, `_features.json`, `_box_pool.json` w katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\`). Epoki liczone od zera, jak w dziennikach.

| Ziarno | dołek straty walidacyjnej (epoka) | szczyt AP (epoka) | AP maksymalne | AP w ostatniej epoce | strata walidacyjna: minimum / koniec | strata ucząca: start / koniec | koszt uczenia do 30. epoki (AP) |
|---|---|---|---|---|---|---|---|
| 42 | 6 | 6 | 0,6846 | 0,6504 | 3,216 / 3,652 | 6,19 / 1,07 | 0,034 |
| 1337 | 6 | 8 | 0,6861 | 0,6551 | 3,228 / 3,603 | 6,21 / 1,10 | 0,031 |
| 2024 | 7 | 7 | 0,6857 | 0,6603 | 3,229 / 3,573 | 6,20 / 1,13 | 0,025 |

Strata na klatkach uczących spada do końca (z około 6,2 do 1,07 dla ziarna 42), a walidacyjna zawraca w epoce 6 lub 7 i rośnie (z 3,22 do 3,65 dla ziarna 42). Uczenie do końca kosztuje średnio 0,030 AP.

**Wynik: sygnał bez etykiet.** Krzywe trwałości dla trzech nowych biegów policzono na klastrze, bo latenty ważą około gigabajta. Epoka szczytu trwałości i strata AP względem maksimum przy zatrzymaniu w tej epoce:

| Blok | ziarno 42 | ziarno 1337 | ziarno 2024 | średni koszt |
|---|---|---|---|---|
| `query` | epoka 7, 0,0011 | epoka 5, 0,0064 | epoka 5, 0,0035 | 0,004 |
| `features` | epoka 7, 0,0011 | epoka 5, 0,0064 | epoka 5, 0,0035 | 0,004 |
| `box_pool` | epoka 7, 0,0011 | epoka 5, 0,0064 | epoka 3, 0,0147 | 0,0074 (wyliczone) |
| `global_pool` | epoka 3, 0,0093 | epoka 3, 0,0148 | epoka 5, 0,0036 | 0,0092 (wyliczone) |

Pójście za sygnałem w blokach `query` i `features` kosztuje średnio 0,004 AP zamiast 0,030, czyli około osiem razy mniej niż trening do trzydziestej epoki. Blok `global_pool` jest zawodny (dwa razy wskazuje epokę 3). Opis sesji tego nie wspomina, ale według pliku `REPORT_kiedy_przestac_box_pool.json` blok `box_pool` też raz (ziarno 2024) wskazał epokę 3 z kosztem 0,0147. Wiersz `global_pool` (osobnego raportu dla tego bloku nie zapisano) i średnie oznaczone jako wyliczone policzyłem z `TRWALOSC_telemetria.json` i `TELEMETRIA.json`; koszty 0,0148 i 0,0093 zgadzają się z liczbami 0,015 i 0,009 z opisu sesji. Zastrzeżenie: wagi zapisywano co dwie epoki, więc rozdzielczość sygnału wynosi dwie epoki; sprawdzono jedno ramię (B2) i jeden zbiór. Nowa przesłanka dla etapu 2: 12 epok w przepisie wystarczy, bo szczyt wypada około szóstej.

### 1.26 Zapisy sesji, równoległe sesje i stan na koniec 22 września (`4f1c2ae7`, `d0912bd9`, `7cae6f78`)

Zapisy pięciu sesji tego etapu trafiły do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\`: `Session_2026-09-22_131716_Koopman_Langevin_Dobor_Klatek_Etap1`, `Session_2026-09-22_150300_Kod_Tozsamosci_Widmo_Slad`, `Session_2026-09-22_160016_Koopman_Czesc_Rzeczywista_Domkniecie`, `Session_2026-09-22_160617_koopman_tozsamosc_narzedzia` i `Session_2026-09-22_210016_krystalizacja_latentu_i_telemetria_treningu`. Trzy pierwsze mają kopie w `F:\Studia\MCHTR\Grant\Projekt\.sessions\` (`Session_2026-09-22_131716_koopman_langevin_vit_dobor_klatek_etap1`, `..._150300_kod_tozsamosci_narzedzia_widmo_i_slad`, `..._160016_czesc_rzeczywista_modow_i_domkniecie_stanu`), identyczne poza dopiskiem o miejscu zapisu. Po południu 22.09 w repozytorium pracowały równolegle dwie sesje: commity `5b699ac5`, `e5916f4c`, `4dbcb5be`, `4bfa6fe3`, `87a57eac` i `4f1c2ae7` należą do sesji „160016", a `27342b29`, `b5be0623`, `4a10f6e1`, `343af3f7` i `d0912bd9` do drugiej. Skutkiem są zdublowane numery sekcji w dokumencie 38 (dwie sekcje „4m" i dwie „4n", brak „4q") oraz dwa niezależne policzenia tych samych niezmienników wycinka (sekcja 1.19).

Stan na koniec etapu: kolejka Edena pusta; katalog roboczy `~/smallvit_runs/2026-09-21_koopman_langevin/` zawiera paczkę 16 klipów, punkty kontrolne, latenty, 27 wyników macierzy etapu 1 z kolejką `queue/stage1`, nagranie `czastki/CZASTKI_pool_b0500`, `KOD_TOZSAMOSCI_pool_b0500.npz`, `results_telemetria/` i `latents_telemetria/` trzech nowych biegów, `TELEMETRIA_stage1.json` i `TRWALOSC_telemetria.json`; wgrane, ale nieuruchomione zostały `code/niezmiennik_czasowy.py`, `patch_analysis.py`, `kalman_state.py`, `pool_manifest.jsonl` i `run_niezm.slurm`. Lokalnie poza gitem: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\detector_probe_clips\` (420 MB) i `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\` (latenty, raporty, nagrania cząstek `czastki\CZASTKI_b0500`, `CZASTKI_b2500`, `CZASTKI_pool_b0500`, rysunki). Żadne ramię K5 ani żadna nowa reguła doboru nie zostały wytrenowane; etap 2 czekał na zatwierdzenie autora. W pamięci asystenta obowiązują od tego etapu reguły „plan przed treningiem" (21.09) oraz uzupełniona 22.09 reguła „bez nazw roboczych" (także obrazowe zwroty własnego autorstwa trzeba objaśnić przed użyciem).

### 1.27 Podsumowanie etapu 1

1. Latent detektora krystalizuje stan narzędzia z uczeniem: przed uczeniem R2 położenia wynosi około 0, po uczeniu od 0,68 do 0,86 (`box_pool` od 0,83 do 0,86); sześć wolnych modów Koopmana niesie położenie z R2 od 0,73 do 0,87 wobec od 0,04 do 0,45 dla PCA. Cała przebudowa dzieje się w pierwszych trzech epokach, a defekt niezmienników Koopmana poza próbą spada z 0,19 do 0,05.
2. Prędkości nie ma w latencie jednej klatki; z ciągu latentów korelacja z prawdziwym nachyleniem toru rośnie do 0,62 przy oknie plus minus 8 kroków. Odporny filtr Kalmana daje z ramek detektora korelację 0,67 (R2 0,44), z latentu 0,31; innowacje mają kurtozę od 25 do 132 i wskazują chybienia gorzej (AUC od 0,60 do 0,68) niż pewność detektora (od 0,81 do 0,83).
3. Macierz etapu 1 (27 treningów: 9 ramion razy 3 ziarna, budżet 1500, 0 błędów): reszta Langevina nie pomaga (K1 minus B3 +0,0027, K1 minus B4 +0,0018, K1 minus placebo +0,0006 przy progu +0,004), wszystkie rozsądne ramiona mieszczą się w 0,005 AP, a dobór równomierny w czasie (0,6866) jest najlepszy lub równy najlepszym. Jedyny wyraźny efekt: klatki po kolei tracą od 0,015 do 0,019 AP w najlepszej epoce.
4. Pokrycie w modach Koopmana (K4) dało 0,6885, +0,0068 nad pokryciem wyglądu (jedyny kontrast nad progiem) i -1,42 punktu chybień (przedział [-2,73; -0,26]), ale tylko +0,0019 nad doborem równomiernym. Jego współrzędne to odcisk operacji (72 procent wariancji między filmami, zero stanu narzędzia), a samo ramię okazało się k-centrum kosinusowym na nieznormalizowanych współrzędnych, więc przyczyna zysku jest nieustalona.
5. Sufit każdego doboru jest niski: tylko 23 procent chybień L0 znika po pięciokrotnie większej liczbie etykiet; pewność detektora wskazuje błąd usuwalny najlepiej (AUC 0,928), sygnały z dynamiki latentu od 0,84 do 0,86, a nawet najlepszy sygnał ma w górnym decylu tylko 7 procent klatek usuwalnych. Szybki ruch daje około 32 procent chybień w każdym ramieniu.
6. Dobór D-optymalny zmniejsza niepewność operatora najbardziej (ślad kowariancji 0,116 wobec 0,210), ale operator dopasowany na tych klatkach przewiduje nowe operacje gorzej (4,0 wobec 7,3 procent); certyfikat informacji nie jest miarą użyteczności.
7. Kod tożsamości narzędzia (niezmienniki operatora translacyjnego na tokenach narzędzia) jest semantyczny (dekoder pikseli: R2 -0,69 na tokenach narzędzia), niesie wydłużenie ramki w 85,8 procent, pole w 29,0, zwrot końcówki w zerowym stopniu, a 58 procent jego zmienności leży między filmami. Ramię K5 zmniejsza największą lukę pokrycia tego kodu z 2,914 do 1,335, podczas gdy żadne ramię macierzy etapu 1 nie różni się tu od losu; K5 nie zostało wytrenowane.
8. Widmo bez symetryzacji: trzy rodziny modów razem odróżniają narzędzie od tła z AUC 0,893 wobec 0,834 dla samych modów trwałych; same części rzeczywiste (15 liczb) dają 0,887 wobec 0,902 dla pełnego kodu. Część rzeczywista nie szereguje modów według tożsamości (korelacja -0,04), za to moduł wartości własnej porządkuje trwałość, kształt (jedyny mod czytający kształt: 58,6 procent) i statykę filmu (64 procent statyki w modach Koopmana wobec 52 w PCA i 30 w kierunkach losowych).
9. Trzy drogi do wektora tożsamości (translacyjna, czasowa, deep Koopman) nie biją ośmiu składowych głównych (88,0 procent wydłużenia przy 7,7 procent przecieku położenia); droga translacyjna jest najczystsza (przeciek 4,1), czasowa przecieka położeniem (40,5), a jej reszta po odjęciu translacyjnej to czysty kod położenia (28,5 procent). Deep Koopman nie bije przewidywania „nic się nie zmienia" (od -45 do -61 procent), a ślad narzędzia w kwadracie siatki jest symetryczny w czasie (0,899 po przejściu wobec 0,902 przed przyjściem), więc to sąsiedztwo, nie pamięć.
10. W czasie uczenia latent ustawia się po pierwszej epoce, dalej dryfuje (CKA około 0,86), a po szczycie jakości rozlewa się (participation ratio `query` z 8,57 do 16,42) przy rosnącej pewności detekcji (z 0,83 do 0,93). Trwałość najwolniejszych modów ma szczyt razem z jakością w 32 z 36 ustawień.
11. Kampania nigdy nie zapisywała straty walidacyjnej; po dopisaniu `--val-loss` trzy nowe biegi pokazały dołek straty walidacyjnej w epokach 6, 6 i 7 i szczyt AP w 6, 8 i 7, a trening do 30. epoki kosztuje średnio 0,030 AP. Sygnał bez etykiet (szczyt trwałości w blokach `query` i `features`) kosztuje średnio 0,004 AP, czyli około osiem razy mniej; w telemetrii 28 biegów szczyt AP wypada w epoce 8 (mediana), stąd propozycja 12 epok w etapie 2.
12. Równolegle inna sesja uzyskała 22.09 w jednoziarnowym pilocie tokenowym Koopman-GLE 0,6732 AP w ostatniej epoce wobec 0,6555 dla losowych 1500 klatek (przy najlepszej epoce przewaga tylko +0,0034); ten dodatni sygnał nie był powtórzony i stoi w napięciu z negatywnym wynikiem macierzy etapu 1 dla reszty Langevina.

### 1.28 Otwarte pytania na koniec etapu 1

- Czy przewaga K4 przetrwa kilka ziaren wyboru, pięć ziaren treningu, kontrole wymiaru (PCA, losowe rzuty, rzut 24-wymiarowy) i poprawione k-centrum euklidesowe? Szum samego wyboru klatek nie był zmierzony.
- Czy pokrycie kodu tożsamości (K5, wobec K5c i B2) przełoży się na AP? Plan etapu 2 (K5, K5b, K5c, B2, B3, poprawione K4 i ramię odniesienia ze składowych głównych; 3 ziarna; budżet 1500; 12 epok; około godziny na A100 przy pięciu procesach) czekał na zatwierdzenie.
- Kod tożsamości wymaga przeliczenia przed etapem 2: trzy rodziny modów (15 albo 20 liczb w postaci liniowej), moment pierwszy odzyskujący zwrot narzędzia, miękka maska zamiast całego wnętrza ramki. Żadna z tych poprawek nie została policzona.
- Strażnik determinizmu rejestratora cząstek nie przechodzi (różnice 0,131 jednostki współrzędnej, wcześniej 7 do 14 procent odchylenia); do tego czasu porównania klatka po klatce są niepewne.
- Operator nie bije przewidywania „nic się nie zmienia" na nowych filmach. Czy dołożenie wymuszenia (ruch kamery z położenia ramki, człon biliniowy, człon skokowy) i liczenie deficytu pamięci Mori-Zwanziga na współrzędnych niezmienniczych da mierzalny zysk?
- Czy Koopman lepiej zadziała jako cel uczenia (strata spójności lub niezmienniczości latentu sąsiednich klatek) niż jako selektor, jak sugeruje dokument 42?
- Około 32 procent chybień w najszybszych 5 procentach klatek niezależnie od doboru: granica modelu jednoklatkowego, rozmycie ruchu czy błędy adnotacji?
- Skąd spadek AP po szczycie, skoro pokrycie ramki na klipach sondujących prawie się nie zmienia, a pewność rośnie? Podejrzewane są fałszywe wykrycia i kalibracja pewności; nie zmierzono.
- Czy sygnał końca uczenia trafia dokładnie w dołek straty przy zapisie wag co epokę, czy działa na innych ramionach, budżetach i modelach, i dlaczego zawodzą bloki `global_pool` (i raz `box_pool`)?
- Zastrzeżenia planu 37 do kodu (skalowanie i PCA na całej puli, retrospektywna ocena, numer zapytania jako tożsamość) nie zostały usunięte; wszystkie pomiary latentu mają status pilota.
- Jak pogodzić negatywny wynik macierzy etapu 1 dla reszty Langevina (DINOv2, 30 epok, trzy ziarna) z dodatnim, jednoziarnowym pilotem tokenowym Koopman-GLE (10 epok, ostatnia epoka)?
- Pomiar „między filmami" nie rozdziela narzędzia jako przedmiotu od oka, światła i kadru, a zbiór ma jedną klasę narzędzia bez etykiety jego stanu, więc tożsamości w sensie „które narzędzie" oraz stanu otwarte czy zamknięte nie da się na tych danych zmierzyć.



## Etap 2. Tożsamość narzędzia w latencie, kryterium domknięcia, test na nietkniętym zbiorze i plany doboru klatek (23 września 2026)

Etap 2 to jeden długi dzień pracy, 23 września 2026, od około 8:50 do 17:30, oraz krótkie zamknięcie w nocy z 23 na 24 września. Wszystkie godziny w tej części są czasem lokalnym (CEST, UTC+2); znaczniki czasu z transkryptów rozmowy, zapisane w UTC, przeliczono. Główna rozmowa z asystentem toczyła się w sesji Claude Code uruchomionej w katalogu projektu grantowego: do 15:11 w transkrypcie `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\d3d4c7f5-5980-4468-900e-4a8a033cbf88.jsonl`, a po skompaktowaniu rozmowy (automatycznym streszczeniu jej wcześniejszej części) w transkrypcie `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\7dec512f-abe0-4cf3-8465-062719d28449.jsonl`. Kod i wyniki trafiały do repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker` (gałąź `codex/eden-nano-training-20260914`), głównie do katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`.

Tego samego dnia pracowały równolegle dwie inne sesje. Sesja Codex (inny asystent programistyczny) napisała formalizację doboru klatek z dowodami i częściowym sprawdzeniem w Lean; jej commity nie mają dopisku `Co-Authored-By: Claude`. Druga sesja przygotowała kod eksperymentu `tozsamosc_czastki_bledy` bez uruchamiania. Obie opisano niżej osobno, bo wpływają na interpretację wyników sesji głównej.

### 2.1 Punkt wyjścia, sprzęt i przebieg dnia

**Co odziedziczono z etapu 1.** Badany jest detektor RF-DETR Nano: sieć typu DETR (detektor oparty na transformerze, który dla obrazu zwraca prostokątną ramkę obiektu i pewność, że obiekt tam jest). Ma dwie części. Backbone to część, która zamienia obraz na siatkę wektorów liczb; tu jest to transformer wizyjny z rodziny DINOv2, który przy rozdzielczości 384 × 384 pikseli dzieli klatkę na 24 × 24 = 576 kwadratowych łatek i dla każdej łatki liczy wektor 384 liczb, zwany tokenem. Dekoder to druga część, która z tokenów buduje kilka „zapytań” (wektorów po 256 liczb), a z najlepszego zapytania wylicza ramkę i pewność. Latent to ogólna nazwa wewnętrznych wektorów sieci, liczonych, zanim sieć poda ramkę; w tej części „latent tokenów” oznacza tokeny backbone, a „latent zapytania” wyjście dekodera. Detektor wykrywa jedno narzędzie, cystotom, w filmach operacji zaćmy ze zbioru Cataract-LMM (część do śledzenia narzędzi).

Podział danych, ustalony wcześniej i niezmieniany (plik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\eden_static36\data\tracking_static36\split_manifest.json`): 117 grup operacji uczących, 25 walidacyjnych i 25 testowych, podział po grupach operacji (seed 42). Walidacja to 12 043 oznaczone klatki z 25 filmów; na niej wybierano najlepszą epokę i liczono wszystkie werdykty etapu 1. Zbiór testowy nie był nigdy eksportowany na klaster (`test_images_exported = 0` w `data_manifest.json`). Pula uczących klatek oznaczonych to 10 637 klatek ze 118 filmów (co trzecia klatka filmu po wstępnej selekcji), z której wybiera się budżet klatek do uczenia. Rozbieżność „117 grup operacji” wobec „118 filmów w puli” nie jest wyjaśniona w źródłach.

Z etapu 1 pochodzą też: odniesienie B2, czyli detektor uczony na 1500 klatkach wybranych równomiernie w czasie, z mAP 0,6866 na walidacji; obserwacja, że uczenie realnie kończy się około epoki 6, a potem sieć się przeucza (strata walidacyjna rośnie z 3,22 do 3,65); oraz nagranie latentu `CZASTKI_pool_b0500`. To nagranie zawiera tokeny 24 × 24 dla 9683 klatek puli ze 118 filmów, z detektora `ut_b0500` (RF-DETR Nano uczony na 500 klatkach wybranych równomiernie w czasie), a każdy token jest zrzutowany na 32 główne składowe wspólnej bazy PCA (analiza składowych głównych, czyli kierunki największej wariancji; plik `baza.npz`); ten 32-liczbowy opis nazywany jest „profilem zredukowanym”. Relacja między 10 637 klatkami puli a 9683 klatkami nagrania nie jest opisana w źródłach etapu 2.

Pojęcia z oceny detektora, używane w całej części: IoU (intersection over union) to pole części wspólnej przewidzianej i prawdziwej ramki podzielone przez pole ich sumy; mAP (mean average precision) to pole pod krzywą precyzja-czułość, a „mAP 0,5:0,95” uśrednia je po progach IoU od 0,5 do 0,95, zaś „mAP 0,5” liczy przy jednym progu IoU 0,5. Epoka to jedno przejście przez wszystkie klatki uczące, krok uczenia to jedna aktualizacja wag po paczce 16 klatek. Ziarno to liczba startowa generatora losowego; ten sam trening powtórzony z innym ziarnem (tu 42, 1337, 2024) pokazuje rozrzut wyniku wynikający z samej losowości.

**Gdzie liczono.** Lokalnie na karcie RTX 3070, w środowisku `F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1` (rfdetr 1.10.1, torch 2.6.0+cu124), oraz na klastrze Eden, zarządzanym przez SLURM (system kolejkowania zadań). Na Edenie używano dwóch partycji, czyli grup węzłów: `short` z węzłami dgx-1 do dgx-4 (karty NVIDIA A100-SXM4-40GB) oraz `hopper` (jeden węzeł z kartami NVIDIA H100 PCIe). Limit dyskowy projektu na Edenie to 200 GB (twardo 220 GB) i 200 tysięcy plików. Dane do każdego treningu pochodzą z archiwum `tracking_static36.tar` (22 680 klatek: 10 637 uczących i 12 043 walidacyjnych), rozpakowywanego przy każdym zadaniu do pamięci węzła i sprzątanego po zakończeniu.

**Przebieg dnia.**

| Godzina | Zdarzenie | Commit |
|---|---|---|
| 08:50-09:21 | objaśnienia wyników etapu 1, pytanie użytkownika o „domknięcie tożsamości”, pierwsze wyszukiwanie literatury | - |
| 09:22-09:29 | zajętość i nasycenie przestrzeni tożsamości | w `197f2ce3` |
| 09:30-09:38 | plan i wysłanie 9 biegów po 250 klatek (zadania 1809754-1809762) | w `197f2ce3` |
| 09:41-09:56 | literatura o wykonalności, dokument 43 (kryterium domknięcia) | `197f2ce3` (09:56) |
| 10:00-10:56 | iloraz ergodyczny, wagi dźwigni, pytanie o dokument 42 | - |
| 11:03-11:25 | dokument 44 „wzór domknięcia” (LaTeX, PDF), wyniki 9 biegów | `38e90068` (11:24) |
| 11:39 | formalizacja Codex (dokument 44 „dobór klatek przez przewidywaną poprawę”) | `66dd2c7c` |
| 11:39-11:50 | mozaiki klatek trzech ramion, wyjaśnienie wyboru K6 | `076872f8` (11:43) |
| 11:52-12:05 | pobudzenie kierunków tożsamości dla pojedynczej klatki, testy domknięcia z literatury | `de3e5196` (11:58) |
| 12:03-12:28 | Codex: Lean, dokument 45, PDF dowodów | `dccc3982`, `e0e7d34e`, `d2639cf5`; w repozytorium grantowym `021e6de01` |
| 12:08-12:26 | test obrotu (zadanie 1809793), sonda uczenia (1809797-1809800) | `684cc3c4` (12:24), `d6fbf761` (12:26) |
| 12:35-13:03 | porządki na Edenie, wynik sondy | `9874e90a` (13:03) |
| 13:06-13:28 | galeria dowodów, artefakt naklejki, kierunek narzędzia w tokenach | `2e61f938` (13:28) |
| 14:35-14:58 | filtr cząsteczkowy, rola Kalmana, wektor narzędzia bez etykiet | `de56106d` (14:58) |
| 14:59-15:10 | test z narzędziem wyciętym po obrysie, lokalnie | `1946d289` (15:09) |
| 15:14-15:29 | ocena detektorów na zbiorze testowym, lokalnie | `9a6bd8a9` (15:28) |
| 15:32-15:36 | zaskoczenie Koopmana wobec chybień (wynik negatywny) | `f758fc35` (15:36) |
| 15:39-16:00 | zastosowania Koopmana, włókno Rousselota, trzy plany eksperymentów | `a8385d7f` (16:00) |
| 15:50 | równoległa sesja: kod `tozsamosc_czastki_bledy` bez uruchomienia | niezacommitowane |
| 15:59-16:05 | wybór 1500 klatek ze zmianami w stałych kierunkach Koopmana | `9fb430ac` (16:04) |
| 16:12-16:28 | próba lokalna, wysłanie 6 biegów na węzeł hopper (1809873-1809878) | - |
| 16:29-16:46 | skan dysku Edenu, kampania doboru cech równoległej sesji, usuwanie plików | - |
| 17:29-17:30 | wynik kierunków stałych, sprzątanie węzła hopper | `0ed75a7b` (17:30) |
| 24.09, 02:02 | polecenie wyłączenia komputera za dwie godziny | - |

### 2.2 Poranna rozmowa: czym ma być „domknięta tożsamość”

**Skąd pytanie.** O 8:50 użytkownik poprosił o opis dotychczasowych wyników (co robi Koopman, czego sieć zdążyła się nauczyć). Po serii wyjaśnień (strata a mAP, podział na zbiór uczący, walidacyjny i testowy, przeuczenie od epoki 6, które „jest takie przeuczenie zwykłe, klasyczne, niezwiązane jeszcze z moim Koopmanem”, jak podsumował użytkownik o 9:01) padło pytanie, jak użyć Koopmana do wyboru klatek. Asystent podał, że pula ma 10 637 oznaczonych klatek ze 118 filmów, a do treningu użyto 1500, czyli około siódmej części.

Operator Koopmana, w wersji używanej w tym projekcie, to macierz K dopasowana metodą najmniejszych kwadratów do par (opis klatki t, opis klatki t + Δ) z tego samego filmu, tak żeby z(t + Δ) ≈ K z(t); ta procedura nazywa się EDMD (extended dynamic mode decomposition). Wektory własne K to mody, a ich wartości własne λ mówią, ile z danego kierunku zostaje po jednym kroku czasu: |λ| bliskie jedności oznacza kierunek trwały, λ zespolone oznacza parę współrzędnych obracających się wokół zera. „Trwałość” w etapie 1 to średnia część rzeczywista ośmiu najwolniejszych wartości własnych.

O 9:06 użytkownik sformułował cel, który prowadzi cały etap: „chciałbym zrobić coś w rodzaju aktywnego uczenia, z tym, że po prostu obserwując, jak się zachowuje latent w środku poprzez operator Kupmana, powiedzieć, że sieć potrzebuje jeszcze takich obrazów, żeby na początku zdobyć dobrą tożsamość tego narzędzia, a potem […] przechodzimy już z operatora Kupmana na operator Langawina, i potem patrzeć […] w jakich pozach potrzebuję jeszcze tego narzędzia chirurgicznego, żeby dopełnić latent tożsamości tego narzędzia”. Aktywne uczenie oznacza tu dobieranie kolejnych klatek do oznaczenia lub uczenia na podstawie stanu sieci, a nie losowo.

**Odpowiedź teoretyczna (9:07-9:10).** Asystent zaproponował dwa mierzalne warunki domknięcia tożsamości. „Domknięcie” oznacza tu, że kilka współrzędnych latentu opisujących narzędzie tworzy samodzielny układ: dokładanie klatek już ich nie zmienia i nie potrzebują one informacji spoza siebie. Warunek wymiaru: liczba modów o trwałości powyżej progu przestaje rosnąć przy dokładaniu klatek. Warunek stabilności: podprzestrzeń tych modów przestaje się obracać między kolejnymi zapisanymi stanami sieci (checkpointami); zmierzone podobieństwo opisów tych samych 4800 klatek z dwóch kolejnych checkpointów zatrzymywało się na 0,86, więc tożsamość w tym sensie nie była domknięta. Źródła różnie nazywają tę miarę: `README.md` eksperymentu `krystalizacja_latentu` podaje podobieństwo geometrii CKA około 0,86, dokument 44 „średnią korelację kanoniczną 0,86”, a dokument 43 korelację kanoniczną wolnych modów od 0,84 do 0,98. Asystent dodał rozróżnienie „sceny” (kierunki trwałe, które opisują oko, światło i kadr) od tożsamości narzędzia oraz rolę Langevina: sprawdzić, co zostaje po odjęciu przewidywania operatora; jeśli ta reszta ma strukturę, brakuje mechanizmu i tam warto dobierać klatki, a jeśli jest białym szumem (nieprzewidywalnym z niczego), dokładanie klatek nic nie da. Użytkownik odpowiedział, że obecność oka, światła i kadru w trwałych modach mu nie przeszkadza, i zapytał, skąd warunek „obracania” (odpowiedź: to zmiana samych kierunków między checkpointami, nie część urojona wartości własnej).

Uwaga o definicji sceny: w tej rozmowie (9:07) asystent opisał tożsamość jako coś stałego w filmie i różnego między filmami, a scenę jako coś stałego we wszystkich filmach. W kodzie liczonym kwadrans później (sekcja 2.3) kryterium jest inne: mod stały przez cały film to scena, a tożsamość musi się w filmie zmieniać, bo narzędzia się wymienia. Obowiązuje drugie kryterium, zapisane też w dokumencie 43.

**Pierwsze wyszukiwanie literatury (9:16-9:18).** Użytkownik poprosił: „Jak to można sformalizować? […] szukaj odpowiednich narzędzi do identyfikacji odpowiednich klatek zbioru uczącego /paper-search”, dołączając długi opis własnej intuicji: tożsamość narzędzia ma być najpierw domknięta tak, żeby sieć na pewno rozpoznawała to konkretne narzędzie, a potem trzeba szukać elementów dynamicznych (pozycja, prędkość, przyspieszenie, obrót w trzech osiach, przy czym obrót wokół osi patrzenia objawia się skracaniem lub wydłużaniem narzędzia w rzucie 2D), co łączył z teorią Moriego-Zwanziga i brakującym wymiarem. Asystent odpowiedział, że gotowej metody domykania tożsamości nie ma, i wskazał: Kim26 („Swift Sampling: Selecting Temporal Surprises via Taylor Series”, wybór klatek, których cechy najbardziej odstają od przewidywania szeregiem Taylora z poprzednich klatek; najbliższy konkurent), She20 („Feature Space Saturation during Training”, nasycenie przestrzeni cech), Ans19 i Rec19 (wymiar wewnętrzny reprezentacji najpierw spada, potem rośnie), Fen21c i Shu22 (dobór klatek pod estymację pozy), Yoo23 (nadmiarowość klatek wideo), Shi20 (słaba zależność czasowa w chirurgii). Pracy o nazwie „Critical Activation Network”, o którą pytał użytkownik, nie odnaleziono. Szkic formalizacji: zamrozić wstępny model, przepuścić przez niego całą pulę, dla filmów dopasować operator kroku, wziąć mody o części rzeczywistej bliskiej jedności, odrzucić te wspólne wszystkim filmom, a w pozostałych współrzędnych liczyć wymiar, obrót między checkpointami i zajętość. O 9:21 użytkownik napisał „tego nie rozumiem” o zajętości; asystent wyjaśnił ją na przykładzie mapy podzielonej na kwadraty (sekcja 2.3).

### 2.3 Zajętość przestrzeni tożsamości: puste kwadraty

**Pytanie.** Użytkownik (9:22): „policz to na naszej puli i pokaż puste kwadraty spróbuj wykonać ten pipeline”. Pytanie badawcze: czy w przestrzeni współrzędnych tożsamości są obszary, w których leżą klatki puli, a nie leży ani jedna klatka już oznaczona i użyta do uczenia; takie „puste kwadraty” miałyby oznaczać nieoglądane układy narzędzia. Zajętość to udział komórek siatki, w których są klatki danego zbioru.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\zajetosc_tozsamosci.py`), bez etykiet w samym rachunku:

1. Opis klatki: tokeny z nagrania `CZASTKI_pool_b0500` z wnętrza ramki przewidzianej przez detektor, uśrednione na siatce 3 × 3 kawałków po 32 liczby (288 liczb), wybielone do 48 wymiarów. Wybielenie to przeskalowanie przez PCA tak, żeby każdy kierunek miał jednostkową wariancję i żadne dwa nie były skorelowane.
2. Operator kroku (EDMD) na parach kolejnych klatek tego samego filmu.
3. Mody wolne: część rzeczywista λ większa niż 0,85. Wyszło ich 10.
4. Odsiew sceny: dla każdego modu udział zmienności wewnątrz filmu (odchylenie od średniej filmu) w zmienności całkowitej. Pięć modów miało ten udział od 27 do 35 procent (scena), pięć od 42 do 53 procent (tożsamość).
5. Osie tożsamości: mody 5, 6 i 7 (części rzeczywiste λ: 0,901, 0,883 i 0,883; udział wewnątrz filmu 0,534, 0,421 i 0,421). Każdą oś dzielono na k przedziałów kwantylowych. Pusty kwadrat to komórka z klatkami puli, ale bez klatek zbioru B2 (1500 klatek równomiernie w czasie). Etykiety posłużyły dopiero na końcu, do opisu zawartości pustych komórek.

**Wyniki** (`...\krystalizacja_latentu\wyniki\REPORT_zajetosc_k8.json`, `_k12`, `_k16`, `_k24`; `REPORT_zajetosc.json` to kopia wariantu k = 24; rysunki `REPORT_zajetosc*.png`). Wiersz k = 6 pochodzi tylko z odpowiedzi w rozmowie (9:25).

| k (przedziałów na oś) | komórek zajętych przez pulę | pustych komórek | klatek w pustych | filmów dotkniętych |
|---|---|---|---|---|
| 6 | 36 z 216 | 0 | 0 | 0 |
| 8 | 64 | 1 | 4 | 3 |
| 12 | 144 | 6 | 37 | 7 |
| 16 | 252 | 17 | 144 | 38 |
| 24 | 558 | 108 | 753 (7,8 procent puli) | 109 ze 118 |

Zawartość pustych komórek przy k = 24 prawie nie różniła się od reszty puli: wydłużenie ramki 0,83 wobec 0,93, wielkość (log pola w pikselach kwadratowych) 9,61 wobec 9,54, prędkość 0,79 wobec 0,83 piksela na klatkę, pewność detekcji 0,915 wobec 0,906. Według rozmowy największa pusta komórka miała 23 klatki z 18 różnych filmów.

**Interpretacja w chwili pomiaru (9:25).** Wynik uznano za negatywny: puste komórki to pojedyncze klatki rozsiane po brzegach chmury, a nie skupiska nieoglądanych układów narzędzia; dobór równomierny w czasie już pokrywa tę przestrzeń. Asystent opisał wtedy chmurę jako „cienką powierzchnię” (36 z 216 komórek przy k = 6).

**Co się później okazało.** Liczby zajętych komórek są dokładnie równe k² (36, 64, 144) albo prawie równe (252 z 256, 558 z 576). Przyczyną był błąd osi: mody 6 i 7 to para sprzężona o tej samej części rzeczywistej, więc osie 6 i 7 były tą samą liczbą (korelacja 1, `REPORT_wzor_domkniecia.json`, pole `k6_korelacja_osi`). Siatka trójwymiarowa była w praktyce dwuwymiarowa, a „cienka powierzchnia” była artefaktem. Błąd wykryto dopiero przy wyprowadzaniu wzoru domknięcia około 11:20 (sekcja 2.7).

### 2.4 Nasycenie: ile klatek obstawia przestrzeń tożsamości

**Skąd pytanie.** O 9:27 użytkownik doprecyzował cel: „chodzi o to, żeby wziąć mniej klatek ze zbioru uczącego, a wiedzieć, że z tych klatek były takie informacje, że wytrenowaliśmy sieć nawet lepiej, niż biorąc wszystkie tam, nie wiem, dziesięć tysięcy klatek. […] trzy tysiące klatek domyka nam tożsamość narzędzia z tych dziesięciu tysięcy. To już jest sukces […] nie chodzi o to, żeby te zbiory się pokrywały, tylko chodzi o to, żeby to działało szybciej, lepiej”. Nasycenie oznacza tu budżet klatek, przy którym przestrzeń tożsamości jest już obstawiona i dokładanie klatek przestaje ją pokrywać bardziej.

**Metoda** (`...\krystalizacja_latentu\nasycenie_tozsamosci.py`): te same trzy osie, siatka 16 przedziałów na oś, 252 komórki zajęte przez pulę 9683 klatek. Pokrycie to udział zajętych komórek, w których jest co najmniej jedna wybrana klatka. Masa to udział klatek puli leżących w komórkach już obstawionych. Cztery sposoby wyboru przy tym samym budżecie: równomiernie w czasie (co n-ta klatka filmu, z dopełnieniem do dokładnego budżetu), losowo, „po kwadratach” (po jednej losowej klatce z każdej komórki, zaczynając od najliczniejszych) oraz zachłanne k-centrum (kolejno klatka najdalsza od już wybranych, kontrola). Werdykt: budżet, przy którym pokrycie osiąga 95 procent.

**Wyniki** (`...\krystalizacja_latentu\wyniki\REPORT_nasycenie.json`, rysunek `REPORT_nasycenie.png`):

| budżet | równomiernie: pokrycie / masa | losowo | po kwadratach | k-centrum |
|---|---|---|---|---|
| 100 | 0,310 / 0,413 | 0,317 / 0,438 | 0,397 / 0,653 | 0,258 / 0,287 |
| 250 | 0,571 / 0,691 | 0,556 / 0,714 | 0,992 / 1,000 | 0,496 / 0,501 |
| 500 | 0,758 / 0,886 | 0,778 / 0,886 | 1,000 / 1,000 | 0,782 / 0,761 |
| 1000 | 0,889 / 0,971 | 0,881 / 0,969 | 1,000 / 1,000 | 0,968 / 0,962 |
| 1500 | 0,929 / 0,988 | 0,929 / 0,984 | 1,000 / 1,000 | 1,000 / 1,000 |
| 2000 | 0,952 / 0,992 | 0,976 / 0,998 | 1,000 / 1,000 | 1,000 / 1,000 |
| 3000 | 0,984 / 0,999 | 0,976 / 0,996 | 1,000 / 1,000 | 1,000 / 1,000 |
| 5000 | 0,988 / 0,999 | 0,992 / 0,999 | 1,000 / 1,000 | 1,000 / 1,000 |

Budżet dla 95 procent pokrycia: równomiernie 2000, losowo 2000, po kwadratach 250, k-centrum 1000. Dobór równomierny przy budżecie 250 dał 248 klatek (dopełnienie dodano później w funkcji `wybor_rownomierny`). W rozmowie (9:29) asystent podał, że chmura zajmuje 252 z 4096 komórek (16³), a najliczniejsza komórka ma 310 klatek; według sekcji 2.3 właściwy mianownik to 256.

**Interpretacja i zastrzeżenie.** Dobór po kwadratach „domyka” przestrzeń przy ośmiokrotnie mniejszym budżecie niż dobór równomierny i losowy, bo bierze po jednej klatce z każdego miejsca, zamiast wracać do gęstych obszarów. Asystent od razu zastrzegł, że to pokrycie w przestrzeni zdefiniowanej przez niego samego, a nie jakość detektora, i że rozstrzygnie dopiero trening. Później (dokument 44, sekcja 2.7) wykazano, że przy stracie lipschitzowskiej we współrzędnych tożsamości przewaga kwadratów jest co najwyżej 0,29 ℓ_max, i tylko jeśli strata zależy głównie od tych współrzędnych.

### 2.5 Trzy ramiona po 250 klatek na Edenie

**Plan i decyzje użytkownika (9:30-9:38).** Użytkownik: „zaplanuj trening 250 klatek po kwadratach kontra 1500 równomiernych”. Asystent zwrócił uwagę, że porównanie 250 z 1500 miesza dwa pytania (czy metoda jest lepsza i czy 250 klatek wystarczy), i zaproponował trzy ramiona przy tym samym budżecie 250: K6 (po kwadratach), B2m (równomiernie w czasie, uczciwa kontrola) i B1m (losowo, podłoga), każde na trzech ziarnach, bo rozrzut między ziarnami wynosi około 0,003 mAP. Kryterium ustalone przed wynikiem: K6 wygrywa, jeśli bije B2m o więcej niż 0,006 (dwa odchylenia ziarna). Asystent zaznaczył też, że przestrzeń tożsamości policzono z modelu już wytrenowanego, więc K6 jest „drugą rundą”, a nie zimnym startem (w tej chwili twierdził błędnie, że chodzi o model z 1500 klatek; o 12:09 poprawił, że nagranie pochodzi z detektora `ut_b0500`, uczonego na 500 klatkach).

Pierwsza wersja planu dokładała warianty ze 180 epokami, żeby wyrównać liczbę kroków uczenia (przy 250 klatkach epoka ma 16 kroków, przy 1500 ma ich 94). Użytkownik zaprotestował: „No ale przy 180 epokach […] będzie przeuczenie, no bo w szczególności gdy weźmiesz 250 klatek”. Asystent zrezygnował ze 180 epok: biegi na 1500 klatkach osiągają szczyt po około 560 krokach, co przy 250 klatkach wypada koło epoki 36, więc ustawił sufit 60 epok (prawie 1000 kroków) i każdy bieg ocenia w jego najlepszym checkpoincie. Użytkownik dodał: „dorzuć jeszcze jakiś patience, po to, żeby po prostu się to uczenie zatrzymało, jak już będzie widać, że jest spadek i przeuczenie”. Ustawiono wczesne zatrzymanie: bieg kończy się, gdy przez 15 epok (cierpliwość, ang. patience) mAP walidacji nie wzrośnie o co najmniej 0,001. Piętnaście, a nie dziesięć, bo przy 250 klatkach krzywa jest poszarpana. Ponieważ zablokowany protokół kampanii ma wczesne zatrzymanie wyłączone i trener pilnuje tego asercją, nowe ramiona dostały osobny protokół `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\configs\protocol_maly_budzet.yaml` i profil `nano_maly_budzet`, a do `pipeline\train\train_rfdetr.py` dopisano przełącznik `--early-stop` (commit `197f2ce3`). O 9:34 użytkownik napisał: „odpal to na edenie”.

**Metoda.** Zestawy klatek zbudował `...\krystalizacja_latentu\buduj_ramiona_250.py` z nagrania `CZASTKI_pool_b0500` (seed selekcji 42), zapisane w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\curated\selection_maly_budzet\` (`maly_K6_identity_cells_b250_sel42.json`, `maly_B2m_uniform_time_b250_sel42.json`, `maly_B1m_random_b250_sel42.json`, `MANIFEST.json`). Pokrycie siatki 16 × 16 × 16: K6 0,992, B2m 0,571, B1m 0,532; liczba filmów: K6 90, B2m 118, B1m 98. Trening: RF-DETR Nano, rozdzielczość 384, paczka 16, precyzja bf16 (16-bitowe liczby zmiennoprzecinkowe), bez augmentacji w trakcie uczenia (pula była augmentowana wcześniej), sufit 60 epok, cierpliwość 15, min_delta 0,001, strata walidacyjna zapisywana co epokę; skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_maly_budzet.slurm` (pliki robocze w `/raid` lub `/dev/shm` węzła, sprzątane pułapką przy wyjściu). Dziewięć zadań 1809754-1809762 ruszyło o 9:38 na węzłach dgx-1, dgx-3 i dgx-4 (A100-SXM4-40GB) i skończyło się między 10:10 a 10:29 (od 2775 do 3891 sekund). Ocena: mAP 0,5:0,95 na walidacji (25 filmów, 12 043 klatki), liczone niezależnie na wagach najlepszej i ostatniej epoki.

**Wyniki** (`...\krystalizacja_latentu\wyniki\REPORT_maly_budzet.json`; metryki biegów w `...\krystalizacja_latentu\dane\biegi\maly_budzet\<bieg>\` z plikami `validation_report.json`, `epoch_val_map.json`, `run_manifest.json`, `TELEMETRIA.json`). Kolejność ziaren: 42, 1337, 2024. P to precyzja (udział trafnych ramek wśród wszystkich zgłoszonych), R to czułość (udział wykrytych narzędzi wśród wszystkich oznaczonych), obie dla wag najlepszej epoki, z pliku `validation_report.json`.

| ramię (250 klatek) | mAP najlepszej epoki | średnia ± odch. | mAP ostatniej epoki (średnia ± odch.) | najlepsza epoka | epok do zatrzymania | epoka minimum straty walidacyjnej | mAP 0,5 najlepszej | P | R |
|---|---|---|---|---|---|---|---|---|---|
| K6 po kwadratach | 0,6290 / 0,6289 / 0,6309 | 0,6296 ± 0,0011 | 0,6138 ± 0,0065 | 13 / 22 / 17 | 29 / 38 / 30 | 13 / 12 / 14 | 0,876 / 0,883 / 0,883 | 0,935 / 0,940 / 0,941 | 0,738 / 0,790 / 0,800 |
| B1m losowo | 0,6344 / 0,6299 / 0,6314 | 0,6319 ± 0,0023 | 0,6030 ± 0,0042 | 6 / 8 / 11 | 22 / 24 / 27 | 10 / 11 / 11 | 0,890 / 0,892 / 0,895 | 0,934 / 0,940 / 0,951 | 0,817 / 0,817 / 0,781 |
| B2m równomiernie | 0,6146 / 0,6238 / 0,6046 | 0,6143 ± 0,0096 | 0,5895 ± 0,0041 | 8 / 6 / 7 | 24 / 22 / 23 | 8 / 10 / 11 | 0,878 / 0,877 / 0,870 | 0,933 / 0,934 / 0,930 | 0,751 / 0,728 / 0,748 |
| odniesienie B2, 1500 klatek (etap 1) | - | 0,6866 | - | - | - | - | - | - | - |

Różnice średnich: K6 − B1m = −0,0023 (remis wobec progu 0,006), K6 − B2m = +0,0152 (wygrana, ale obciążona, niżej), K6 − B2 (1500) = −0,057. Formalnego testu statystycznego nie liczono; kryterium był próg 0,006. Po stronie K6 zostały: najmniejszy rozrzut między ziarnami, najmniejszy spadek od najlepszej do ostatniej epoki, najpóźniejsze maksimum (epoki 13-22 wobec 6-11) i najdłuższe biegi, czyli wolniejsze przeuczanie.

**Co się zepsuło.** Po pierwsze, ramię B2m jest obciążone: przy dwóch klatkach na film funkcja `linspace` bierze pierwszą i ostatnią klatkę filmu w puli, i tak jest ze 155 z 250 klatek. Po drugie, osie K6 zawierały duplikat z pary sprzężonej (sekcja 2.3), więc K6 wybierał w dwóch, a nie w trzech wymiarach (252 zajęte komórki z 256 możliwych, prawie tyle, ile wybierano klatek). Po trzecie, przez część dnia źródło latentu przypisywano detektorowi z 1500 klatek, choć było nim `ut_b0500`. Ostrzeżenie „run_id w variant-file różny od --run-id” w logach zadań było nieszkodliwe.

**Interpretacja.** Wynik przekazany użytkownikowi o 11:37 brzmiał: dobór po kwadratach dał tyle samo co losowe 250 klatek, obie wersje są daleko od 1500 klatek. Na pytanie „no i rozumiem że koopman nie daje lepszych efektów?” (11:49) asystent odpowiedział, że w tym teście nie, a wcześniej, przy 1500 klatkach, wariant z Koopmanem (K4 w etapie 1) wygrał tylko o 0,007, na granicy szumu; test był jednak wadliwy z powodu dwuwymiarowej siatki. Wynik nie dowodzi, że poprawnie zbudowana siatka (część rzeczywista i urojona jednego modu jako dwie osie) też przegra, ani że pełne pokrycie przestrzeni tożsamości jest bezużyteczne; pokazuje tylko, że pokrycie w tych dwóch osiach modelu wstępnego nie przekłada się na mAP. Wagi tych dziewięciu biegów usunięto z Edenu o 16:46 (sekcja 2.22); metryki zostały lokalnie.

### 2.6 Literatura o wykonalności i dokument 43: kryterium domknięcia

**Wykonalność (9:41-9:48).** Użytkownik poprosił o sprawdzenie, „czy takie szukanie tożsamości Koopmanem i rozszerzonym równaniem Langevina w ogóle ma jakieś pokrycie w literaturze i czy w ogóle to ma sens […] czy takie pseudo-aktywne uczenie sieci neuronowej poprzez takiego swipe'a po zbiorze uczącym ma sens” (/paper-search; w tej części dnia asystent pracował na modelu Claude Fable 5.1, co widać w stopce commitu `197f2ce3`). Werdykt asystenta: droga jest wykonalna, bo każdy klocek ma osobne pokrycie, ale nie ma pracy składającej je na detektorze wideo. Klocki: Ber23 (Berman, Naiman, Azencot, „Multifactor Sequential Disentanglement via Structured Koopman Autoencoders”, ICLR 2023) i Mas21 (Comas i in., rozkład wideo na obiekty i ich dynamikę operatorem Koopmana); Con26 (Conradie i in., „Trustworthy Koopman Operator Learning: Invariance Diagnostics and Error Bounds”, miara wycieku podprzestrzeni); Kap25 (Kapadia, Benner, Feng, aktywne uczenie modelu zredukowanego tam, gdzie podprzestrzeń najbardziej się obraca) i Uy21b (dobór próbek pod identyfikację operatora przy szumie); Col19 („Selection via Proxy”, 514 cytowań) i Eva23 („Bad Students Make Great Teachers”), czyli wybór danych małym modelem pomocniczym; Gup23c (Gupta i in., człon pamięci Moriego-Zwanziga dołożony do Koopmana w latencie autoenkodera). Żadna praca nie używa tożsamości z Koopmana do wyboru klatek pod detektor i nie mierzy potem jakości detekcji; najbliżej jest Swift Sampling.

Na pytanie „ale jak oni odkrywają że to są obiekty?” asystent przeczytał pełne teksty (przez Undermind, płatny serwis wyszukiwania i czytania prac naukowych) i odpowiedział, że żadna z dwóch prac nie odkrywa tożsamości operatorem. W Ber23 podział jest wymuszony stratą: projektant z góry ustala liczbę współrzędnych stałych (na przykład 8 z 40), kara ciągnie ich wartości własne do 1, a resztę do koła o promieniu 0,5; weryfikacja polega na podmianie współrzędnych stałych między sekwencjami i klasyfikatorze (kolor włosów przenosi się w 90,6 procent przypadków, błąd identyfikacji mówcy EER 4,46 procent na współrzędnych stałych wobec 26,8 na dynamicznych). W Mas21 Koopman dostaje tylko cztery liczby na obiekt (położenie, skala, proporcje ramki), a wygląd to osobny wektor 50 liczb widziany wyłącznie przez dekoder. U nas wartości własne bliskie jedności wyszły bez wymuszania, a odpowiednikiem weryfikacji był odczyt wydłużenia narzędzia z modów trwałych (85,7 procent). Ber23 ostrzega, że nie da się z góry ustalić, które współrzędne stałe niosą którą cechę, a wymiar podprzestrzeni pojedynczej cechy zmieniał się między partiami danych (1, 2 albo 3). Kap25 wyjaśniono osobno: „obrót” to zmiana samych kierunków podprzestrzeni między dwoma ustawieniami parametru, mierzona wzorem D̂ = sqrt(1 − ‖XᵀY‖²_F / max(p, q)), bez wartości własnych; nowy pomiar stawia się między ustawieniami o największym D̂.

**Dokument 43 (9:50-9:56).** Użytkownik: „przeczytasz wszystkie te artykuły i chcę, żebyś wyciągnął z nich jeden wniosek, co by wskazywało na to, że tożsamość w latencie się domyka dla danego obiektu. I jak to mierzyć przy pomocy Koopmana i jeszcze Langewina […] stwórz sobie pipeline i nawet matematycznie możesz to rozpisać”. Powstał dokument `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\43_KRYTERIUM_DOMKNIECIA_TOZSAMOSCI_KOOPMAN_LANGEVIN.md` (commit `197f2ce3`), synteza jedenastu prac przeczytanych w pełnym tekście (Ber23, Mas21, Con26, Pan20, Lin21b, Gup23c, Uy21, Uy21b, Kap25, She20, Kim26) i czterech znanych ze streszczeń (Ans19, Rec19, Col19, Eva23). Rozwija on dokument 42 z 22 września (`...\42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md`, commit `e5916f4`), który rozróżniał trzy znaczenia domknięcia: grupowe, wymiarowe i Markowa.

Treść dokumentu 43:

- Wniosek główny (w pierwotnej postaci): tożsamość jest domknięta dokładnie wtedy, gdy podprzestrzeń współrzędnych trwałych jest niezmiennicza względem kroku czasu; ten sam fakt ma miarę geometryczną (wyciek) i dynamiczną (zanik jądra pamięci), a pozostały szum musi spełniać związek fluktuacji i dyssypacji.
- Dokładne dyskretne równanie Moriego-Zwanziga (Lin21b): g(k+1) = Ω(0) g(k) + Σ_{l=1..k} Ω(l) g(k−l) + W(k). Ω(0) = C(Δ) C(0)⁻¹ to macierz Markowa, identyczna z macierzą EDMD; Ω(l) to jądra pamięci liczone rekurencyjnie z korelacji dwuczasowych; W(k) to dynamika ortogonalna, czyli szum. Jądro pamięci to zestaw macierzy mnożących przeszłe stany; mówi, ile przyszłość zależy od historii ponad stan bieżący.
- Podprzestrzeń tożsamości: V_toż = span{φ_i : |λ_i − 1| < δ i w_i ≥ w_min}, gdzie w_i to udział zmienności modu wewnątrz filmu (start: Re λ > 0,85, w ≥ 0,35).
- Miara geometryczna (Con26): wyciek(V) = max_j sin θ_j, gdzie θ_j to kąty główne między V a jej obrazem po kroku KV; sin θ_j to norma części, która po kroku wypada poza V. Kąty główne to kolejne najmniejsze kąty między wektorami jednostkowymi z dwóch podprzestrzeni. Próg trzeba wyznaczyć z załamania krzywej błędu (u Con26 około 0,01), nie przepisywać.
- Miara dynamiczna: pamięć(V) = Σ_{l=1..L} ‖Ω_V(l)‖_F / ‖K_V‖_F dla L = 3, kontrola przy L = 6.
- Test spójności przez związek fluktuacji i dyssypacji (Lin21b, równanie 89: Ω(k) = −⟨W(k), W(0)ᵀ⟩ C(−Δ)⁻¹), który wiąże jądro pamięci z autokorelacją szumu. Trzy przypadki: oba małe i zgodne (domknięte, szum nieusuwalny), jądro małe przy skorelowanych resztach (za krótkie L), jądro duże przy białych resztach („brakuje zmiennej, nie danych”, na przykład skrócenia narzędzia przy obrocie w głąb).
- Domknięcie względem danych: stabilność D̂ (Kap25) i nasycenie wymiaru dim V_toż(n) (She20); weryfikacja z etykietami przez R² odczytu cechy fizycznej z V_toż z podziałem po filmach (wydłużenie 85,7 procent, położenie 4,1 procent w kodzie przesunięciowym i 44 procent w kodzie czasowym). R² to udział wariancji jednej wielkości wyjaśniony przez drugą.
- Reguła doboru z brakującego kierunku: kowariancja reszt Σ_W, jej główny wektor własny e₁ i ocena klatki s(t) = |⟨W(t), e₁⟩| razy wskaźnik, że reszta ma strukturę czasową; Kim26 jako podłoga porównawcza, kryterium D-optymalne Uy21b tylko jako kontrola.
- Pipeline w 11 krokach (kod klatki, operator, V_toż, wyciek, pamięć, spójność, stabilność i nasycenie, weryfikacja, dobór, zatrzymanie treningu szczytem trwałości, werdykt: mAP walidacji przy tym samym budżecie, trzy ziarna, próg 0,006). Kroki 1-3 i dobór po kwadratach były policzone; kroki 4-8 wymagały nowych skryptów (`wyciek_tozsamosci.py`, `pamiec_tozsamosci.py`, `stabilnosc_tozsamosci.py`), których w etapie 2 nie napisano.
- Kontrargumenty i luki: mod trwały może być trwały „z nudów”; wyciek zależy od słownika (kontrola przy d = 24 i d = 96); zbiór testowy nie był nigdy uruchamiany.

W sekcji 9 dokument podaje jako model wejściowy „B2 na 1500 klatkach”; faktycznie nagranie pochodziło z `ut_b0500`.

**Lepszy dobór widmowy (10:00-10:02).** Użytkownik zapytał, czy dobór po kwadratach jest jeszcze odpowiednikiem Koopmana. Odpowiedź: tak, zgrubnym. Według Bud12 (Budišić, Mezić, „Geometry of the ergodic quotient reveals coherent structures in flows”) klatki o tym samym zestawie wartości funkcji własnych są dynamicznie równoważne, a kwadrat siatki to dyskretyzacja takiego zbioru (ilorazu ergodycznego). Ci sami autorzy ostrzegają przed równą siatką i zalecają współrzędne dyfuzyjne z jądrem normalizowanym gęstością i dopiero na nich grupowanie. Dla estymacji samego operatora: Uy21b (klatki podnoszące najmniejszą wartość osobliwą macierzy danych) i Eri17 (wagi dźwigni zamiast równomiernego losowania). Dokument 42 pokazał jednak, że dobór optymalny dla operatora (pilot D-optymalny) zmniejszył niepewność operatora z 0,210 do 0,116, ale pogorszył przewidywanie kroku z 7,3 do 4,0. Asystent zaproponował zamianę siatki na iloraz ze współrzędnymi dyfuzyjnymi i ramię kontrolne z wagami dźwigni, najpierw lokalnie na pokryciu. W etapie 2 tego nie policzono.

**Późniejsze korekty dokumentu 43.** Dokument 44 „wzór domknięcia” (sekcja 2.7) wykazał, że „wtedy i tylko wtedy” jest za mocne (kontrprzykład C = 0) i dopisał do dokumentu 43 ramkę „Poprawka (dokument 44, wniosek 4)”. Niezależnie formalizacja Codex (sekcja 2.8) dopisała na początku dokumentów 42 i 43 nagłówek „Korekta matematyczna, 2026-09-23”: równoważność „zerowe jądro pamięci wtedy i tylko wtedy, gdy opis jest domknięty” zostaje wycofana, związek fluktuacji i dyssypacji nie jest uniwersalnym rozstrzygnięciem „braku danych” przeciw „brakowi zmiennej”, a pomiary pozostają wynikami historycznymi bez silniejszych interpretacji.

### 2.7 Dokument 44 „Wzór domknięcia tożsamości”: pamięć jako pętla przez resztę latentu

**Skąd.** O 11:03 użytkownik wkleił fragment odpowiedzi o dokumencie 43 i napisał: „chce mieć jakiś matematyczny wzór który by opisywał to co tutaj analizujemy […] wyprowadź go i zapisz w latex i w pdfie!”. Powstały `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\44_WZOR_DOMKNIECIA_TOZSAMOSCI.tex` oraz `...\44_WZOR_DOMKNIECIA_TOZSAMOSCI.pdf` (PDF poza gitem, bo repozytorium ignoruje pliki PDF), skrypt sprawdzający `...\krystalizacja_latentu\wzor_domkniecia.py` i raport `...\krystalizacja_latentu\wyniki\REPORT_wzor_domkniecia.json`; commit `38e90068` (11:24). Numer 44 okazał się zajęty dwa razy (sekcja 2.8).

**Założenia.** Z1 (wybielenie i stacjonarność): opis klatki φ_n ∈ R^d ma średnią zero i kowariancję jednostkową w obrębie filmu; d = 48, krok Δ = 5 klatek nagrania. Z2 (liniowy model kroku): φ_{n+1} = A φ_n + ε_{n+1}, gdzie szum ε ma kowariancję Σ i jest nieskorelowany z przeszłością; z Z1 wynika Σ = I − AAᵀ oraz E[φ_{n+k} φ_nᵀ] = A^k. W wybielonych współrzędnych A to dokładnie macierz EDMD. Obserwabla liniowa g_a(x) = aᵀφ(x); operator Koopmana działa na współczynniki przez transpozycję, (K g_a)(x) = (Aᵀa)ᵀ φ(x).

**Definicje.** Q (d × p) to baza ortonormalna przestrzeni tożsamości V, Q⊥ baza dopełnienia; y_n = Qᵀφ_n (współrzędne tożsamości), w_n = Q⊥ᵀφ_n (reszta: scena, poza, wszystko inne). Bloki macierzy kroku: M = QᵀAQ (tożsamość do tożsamości, tłumienie i obrót), C = Q⊥ᵀAQ (ile tożsamości przecieka do reszty), D = Q⊥ᵀAQ⊥ (jak długo reszta pamięta), B = QᵀAQ⊥ (ile reszty wraca do tożsamości).

**Twierdzenia i wnioski.**

```
Tw. 1 (dyskretne uogólnione równanie Langevina):
  y_{n+1} = M y_n + Σ_{l=1..n} B D^{l-1} C y_{n-l} + W_{n+1},
  W_{n+1} = B D^n w_0 + Σ_{j=1..n} B D^{n-j} Q⊥ᵀ ε_j + Qᵀ ε_{n+1},   E[W_{n+1} y_0ᵀ] = 0.
Wn. (transmitancja pętli):  Ω̂(z) = Σ_{l≥1} Ω_l z^{-l} = B (zI − D)^{-1} C.
Tw. 2 (jądra z korelacji, rekurencja Lin21b):  C_y(kΔ) = E[y_{n+k} y_nᵀ] = QᵀA^kQ,
  Ω_k = ( C_y((k+1)Δ) − Σ_{l=0..k-1} Ω_l C_y((k−l)Δ) ) C_y(0)^{-1},  Ω_0 = M,  daje Ω_k = B D^{k-1} C.
Wn. (pierwszy wyraz pamięci):  Ω_1 = C_y(2Δ) − C_y(Δ)² = BC;
  brak pamięci ⇔ korelacje tworzą półgrupę: C_y(kΔ) = C_y(Δ)^k.
Tw. 3 (fluktuacja-dyssypacja):  E[W_{k+1} W_1ᵀ] = −Ω_k C_y(Δ)ᵀ,  czyli  Ω_k = −E[W_{k+1} W_1ᵀ] C_y(−Δ)^{-1}.
Tw. 4 (wyciek to blok):  σ_min(QᵀA) sin θ_max ≤ ‖B‖ ≤ σ_max(QᵀA) sin θ_max,
                         σ_min(AQ)  sin φ_max ≤ ‖C‖ ≤ σ_max(AQ)  sin φ_max.
Wn. (nierówność pamięci, przy ‖D‖ < 1):  Σ_{l≥1} ‖Ω_l‖ ≤ ‖A‖² sin θ_max sin φ_max / (1 − ‖D‖).
Wn. 4 (poprawka do dok. 43):  Ω_l = 0 dla wszystkich l ⇔ B D^{l-1} C = 0 dla wszystkich l
  (B zeruje się na przestrzeni Kryłowa span{C, DC, D²C, ...}); B = 0 jest dostateczne, nie konieczne;
  drugi warunek dostateczny to C = 0.
Wn. (wyciek przez obrót):  dla V = span{ψ_R} (sama część rzeczywista funkcji własnej z λ = |λ|e^{iω})
  sin θ_max(V, KV) = |sin ω|.
Tw. 5 (pokrycie):  (1/|P|) Σ_{x∈P} ℓ_S(x) ≤ max_{s∈S} ℓ_S(s) + L Σ_{j∈J(S)} π_j diam_j + ℓ_max (1 − m(S)).
Program doboru:  S* = argmin |S|  przy  Σ‖Ω_l(S)‖ ≤ ε_Ω,  sin θ_max(S) ≤ ε_θ,  D̂(V_S, V_{S∪ΔS}) ≤ ε_D.
```

Tu θ_max to największy kąt między V a jej obrazem Koopmana (wyciek do tożsamości z zewnątrz, blok B), a φ_max to największy kąt między V a AV (wyciek z tożsamości na zewnątrz, blok C). Słowami: pamięć tożsamości to pętla; tożsamość przecieka do reszty latentu (C), krąży tam tak długo, jak reszta pamięta (D), i wraca (B). Uogólnione równanie Langevina (GLE, generalized Langevin equation) to równanie ruchu z dryfem, członem pamięci (sumą po przeszłości z jądrem) i szumem; tu opisuje ruch współrzędnych tożsamości. Związek fluktuacji i dyssypacji (twierdzenie 3) wiąże jądro pamięci z autokorelacją szumu i wynika tu ze stacjonarności Σ = I − AAᵀ; jeśli jądro z rekurencji i jądro z szumu się nie zgadzają, to proces nie jest stacjonarny albo w opisie brakuje zmiennej. Kontrprzykład do dokumentu 43: reszta latentu autonomiczna (C = 0) pcha tożsamość (B ≠ 0); wtedy C_y(kΔ) = M^k, pamięci nie ma, a mimo to sin θ_max > 0. Domknięcie Koopmana (B = 0) jest więc mocniejsze niż domknięcie Markowa (Ω ≡ 0) i potrzebne są obie miary. Wniosek o obrocie tłumaczy błąd K6: z pary sprzężonej trzeba brać ψ_R i ψ_I razem; obrót pary 6-7 był wolny (ω = 0,018 radiana na krok), więc wyciek przez obrót był mały, ale stracono trzecią oś.

Część o nasyceniu: średnia korelacja kanoniczna 0,86 między checkpointami odpowiada D̂ ≈ sqrt(1 − 0,86²) ≈ 0,51, czyli obrotowi podprzestrzeni o około 30 stopni. Gdy prawdziwa przestrzeń jest ustalona, a różnice wynikają tylko z losowania par, D̂ powinno maleć jak N_ef^(−1/2), gdzie efektywna liczba par według przybliżenia Bartletta to N_ef ≈ N (1 − ρ²)/(1 + ρ²) (przy korelacji sąsiednich klatek ρ = 0,9 dziesięć par waży tyle, co jedna niezależna). Przewidywanie do sprawdzenia: nachylenie −1/2 na wykresie log D̂ względem log N_ef znaczy „brakuje tylko danych”, wypłaszczenie znaczy „przestrzeń naprawdę się obraca”. Twierdzenie 5 (strata lipschitzowska w każdej komórce, czyli zmieniająca się nie szybciej niż L razy odległość) daje przy masach 1,00 (po kwadratach), 0,71 (losowo) i 0,69 (równomiernie) przewagę kwadratów co najwyżej 0,29 ℓ_max.

**Sprawdzenie na układzie syntetycznym** (część A skryptu: losowe A o normie mniejszej niż 1, d = 10, p = 3, korelacje dokładne i 400 tysięcy trajektorii Monte Carlo):

| co sprawdzone | wynik |
|---|---|
| rekurencja Lin21b a B D^{l−1} C, l ≤ 6 | największy błąd 2,8·10⁻¹⁶ (zero co do precyzji maszyny) |
| fluktuacja-dyssypacja, k = 1, 2, 3, dokładnie | błędy 6,8·10⁻¹⁷, 4,3·10⁻¹⁷, 1,7·10⁻¹⁷ |
| to samo z Monte Carlo | błędy 0,0024, 0,0031, 0,0031 przy błędzie standardowym około 0,0015 |
| granice ‖B‖ | 0,468 ≤ 0,640 ≤ 0,736 |
| granice ‖C‖ | 0,548 ≤ 0,669 ≤ 0,689 |
| nierówność pamięci | 0,479 ≤ 3,741 (spełniona, luźna) |
| kontrprzykład C = 0 | norma pamięci 5,6·10⁻¹⁷, sin θ_max = 0,319 |
| obrót, ω = 0,02; 0,2; 0,6 | sin θ = 0,0200; 0,1987; 0,5646, równe |sin ω| |

**Pomiar na puli** (część B; nagranie `CZASTKI_pool_b0500`: 9683 klatki ze 118 filmów, 1689 par na odstępie 5 klatek, 1649 na odstępie 10, 1588 na odstępie 15, 315 trójek do testu fluktuacji i dyssypacji). Dziesięć modów wolnych: pięć sceny (0 i 1: para Re 0,9726 ± 0,0073i, udział w filmie 0,270; 2: Re 0,9579, 0,288; 3 i 4: para 0,9360 ± 0,0018i, 0,346) i pięć tożsamości (5: 0,9010, 0,534; 6 i 7: para 0,8833 ± 0,0161i, 0,421; 8: 0,8777, 0,474; 9: 0,8634, 0,495). Przestrzeń tożsamości ma wymiar 5 (mod 5, para 6-7 jako Re i Im, mody 8 i 9). Pamięć to ‖Ω₁‖_F / ‖Ω₀‖_F. Podłoga szumu to 95. centyl tej samej miary na procesie bez pamięci y_{n+1} = Ω₀ y_n + szum, symulowanym na tej samej siatce klatek (200 powtórzeń); przedział to bootstrap po filmach (losowanie filmów ze zwracaniem, 200 losowań, 95 procent).

| przestrzeń | wymiar | sin θ_max, d = 48 | sin θ_max, d = 96 | pamięć z korelacji / podłoga (przedział) | pamięć z szumu / podłoga (przedział) | różnica jąder | miara półgrupy (przedział) |
|---|---|---|---|---|---|---|---|
| tożsamość | 5 | 0,000 | 0,153 | 0,108 / 0,142 (0,107-0,203) | 0,080 / 0,049 (0,074-0,122) | 0,134 | 0,119 (0,117-0,224) |
| scena | 5 | 0,000 | 0,096 | 0,111 / 0,150 (0,096-0,201) | 0,041 / 0,035 (0,035-0,074) | 0,137 | 0,119 (0,101-0,218) |
| osie K6 (same ψ_R) | 2 | 0,009 | 0,117 | 0,055 / 0,129 (0,036-0,164) | 0,073 / 0,038 (0,054-0,118) | 0,064 | 0,061 (0,039-0,181) |

**Interpretacja.** Wyciek w słowniku 48 jest zerem z konstrukcji, bo przestrzeń tożsamości składa się z funkcji własnych operatora policzonego w tym samym słowniku; prawdziwy test daje słownik 96, gdzie tożsamość przecieka mocniej niż scena (0,153 wobec 0,096), czyli około 15 procent obrazu po kroku leży poza tożsamością. Pamięć z korelacji (0,108) jest poniżej podłogi 0,142, więc z 1689 par nie da się jej odróżnić od zera; pamięć z szumu (0,080) jest ponad podłogą 0,049, a oba jądra różnią się o 0,134, bardziej niż same wynoszą. Według twierdzenia 3 taka niezgodność znaczy, że model liniowy i stacjonarny nie pasuje. Najprostsze wyjaśnienie podane w dokumencie: około połowy zmienności osi tożsamości leży między filmami, a stałe przesunięcie filmu działa jak ukryta zmienna o wartości własnej 1, której nie ma w opisie. Następny zaplanowany pomiar (niezrobiony w etapie 2): te same miary po odjęciu średniej filmu. Wyciek w słowniku 96 nie ma własnej podłogi szumu.

Ograniczenia zapisane w dokumencie: wzór z blokami jest dokładny tylko dla modelu liniowego (rekurencja z twierdzenia 2 jest poprawna dla rzutu Moriego zawsze, utożsamienie z pętlą B D^{l−1} C tylko w modelu liniowym); wybielenie liczone na wszystkich klatkach, a korelacje na parach (skrypt liczy kąty w metryce Grama par); nierówność pamięci jest luźna (0,48 wobec 3,74); twierdzenie o pokryciu zakłada lipschitzowską zależność straty od współrzędnych tożsamości, czego wynik z Edenu nie potwierdza; wszystkie werdykty są na walidacji.

**Co się zepsuło i jak naprawiono** (według streszczenia rozmowy): test fluktuacji i dyssypacji w Monte Carlo dawał błąd względny około 1 przez odwracanie źle uwarunkowanej macierzy, więc porównano E[WWᵀ] z −Ω Mᵀ bez odwracania i dodano sprawdzenie dokładne; krok czasowy wyszedł 15 zamiast 5 (mediana różnic numerów klatek jednego filmu), poprawiono na minimum; macierz dla osi K6 była osobliwa przez zduplikowane części rzeczywiste pary sprzężonej, usunięto duplikaty; podłoga szumu sceny dawała NaN (promień spektralny ≥ 1), przeskalowano go do 0,995; heredoc w Bashu zamieniał `\\` na `\` w tabelach LaTeX, poprawiono edycją, a nadmiarowe szerokości przez `xurl`, `emergencystretch` i `gather`.

### 2.8 Równoległa formalizacja: dokument 44 „Dobór klatek przez przewidywaną poprawę”, dokument 45 i dowody w Lean

**Kto i kiedy.** W tym samym czasie sesja Codex („Autor opracowania: Codex, na polecenie autora projektu”; treści polecenia nie ma w źródłach etapu 2) zapisała cztery commity: `66dd2c7c` (11:39, „Formalize conditional guarantees for Koopman Langevin frame selection”), `dccc3982` (12:03, opis cząstek i sprawdzone gwarancje w Lean), `e0e7d34e` (12:19, reszty Langevina i niepewność Kalmana) i `d2639cf5` (12:28, PDF wyprowadzeń). Plik Lean trafił do repozytorium grantowego jako commit `021e6de01` (12:03). Dokumenty: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md` i `...\45_RESZTA_LANGEVINA_KALMAN_I_DOBOR_KLATEK.md`; katalog kontroli `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\` (`README.md`, `sprawdz_wyprowadzenia.py`, `WERYFIKACJA.json`, `sprawdz_lean.ps1`, `LEAN_WERYFIKACJA.json`, `LEAN_WERYFIKACJA.log`, `sprawdz_reszte_kalmana.py`, `RESZTA_KALMAN_WERYFIKACJA.json`, `zbuduj_pdf_langevin.py`). Nie uruchamiano treningu, obrazów, GPU ani Edenu. W efekcie istnieją dwa dokumenty z numerem 44; dokument 43 w nagłówku odsyła do wersji Codex, a `README.md` eksperymentu `krystalizacja_latentu` do wersji „wzór domknięcia”.

**Główny wynik dokumentu 44 (Codex).** „Można zagwarantować poprawę po wybranej partii klatek, jeżeli przewidywana poprawa straty przewyższa ograniczenie błędu tego przewidywania. Można również zagwarantować przewagę nad inną partią, porównując dolne ograniczenie poprawy pierwszej z górnym ograniczeniem poprawy drugiej. Koopman i Langevin mogą dostarczać przybliżeń potrzebnych do takiego obliczenia. Samo widmo, pokrycie modów ani małe jądro pamięci nie wystarczają do gwarancji.”

**Osiem korekt dokumentów 42 i 43**, uznanych za konieczne przed formalizacją: zerowe jądro pamięci nie jest równoważne pełnemu opisowi stanu (może zostać wymuszenie od ukrytego stanu); trwałość nie jest tożsamością narzędzia (stałe tło i stała funkcja też mają λ = 1); część rzeczywista wektora własnego nie ma uprzywilejowanego znaczenia (faza jest dowolna); kwadrat i moduł nie są zawsze niezmiennikami; biała reszta nie dowodzi nieusuwalności błędu detekcji; związek fluktuacji i dyssypacji nie rozstrzyga sam, czy zabrakło danych; brak pokrycia latentu nie jest jeszcze brakiem wiedzy potrzebnej do wykrywania; R² odczytu proporcji prostokąta nie oznacza procentu kompletności przestrzeni stanu.

**Sformułowanie.** Dane rozdzielone po operacjach: U (kandydaci), D (dane sterujące wyborem, z oznaczeniami), E (końcowa ocena). Ryzyko na nowych operacjach R(θ) = E_v E_t ℓ(f_θ(X_{v,t}), Y_{v,t}). Dwa zegary: t (kolejne obrazy przy zamrożonych wagach) i k (kolejne aktualizacje wag); operator Koopmana dopasowany po t nie jest modelem zmian wag po k. Stan treningu s_k = (θ_k, o_k) z pamięcią optymalizatora; dobór partii S_k to sterowanie tym układem, a cel to minimum oczekiwanego ryzyka przy budżecie czasu. Opis przez cząstki (sekcja 4.1a): pole tokenów Z_{v,t} = (z_{v,t,p}), gdzie p to stałe miejsce siatki; redukcja kanałów nie usuwa indeksu cząstki, a dla opisu klatki a = Σ_p w_p z_p przy błędach cząstek e_p zachodzi ‖Σ w_p z_p − Σ w_p z̃_p‖ ≤ Σ |w_p| e_p. Cząstka na stałej siatce nie jest automatycznie punktem metalu narzędzia; śledzenie wymaga jawnej korespondencji.

**Twierdzenia** (wszystkie z jawnymi założeniami):

- T1: znaczenie trwałości: φ_{t+1} − φ_t = (λ − 1)φ_t + r_t, stąd ‖φ_{t+1} − φ_t‖ ≤ |λ − 1| ‖φ_t‖ + ‖r_t‖; kwadrat jest niezmienny tylko dla λ = ±1, moduł dla |λ| = 1; faza pary sprzężonej jest dowolna. Lemat 4.2a: opis w pełni niezmienny względem przesunięcia nie pozwala lokalizować przesuniętych kopii obiektu.
- T2: deficyt informacji o zadaniu dla zamrożonej reprezentacji Z = F_θ(X), przy stracie kwadratowej Δ_zad(Z) = E‖E[Y|X] − E[Y|Z]‖².
- Tożsamość operatorowa Moriego-Zwanziga (6.0), dokładna także dla układu nieliniowego: K^{n+1} = Σ_{j=0..n} K^{n−j} P K (QK)^j + (QK)^{n+1}, gdzie K to operator Koopmana, P to rzut na rozpiętość wybranych obserwabli, a Q = I − P (inne znaczenie Q niż baza w sekcji 2.7); domknięcie przestrzeni obserwabli ma ścisły warunek QKP = 0.
- T3: eliminacja nieobserwowanych współrzędnych w układzie blokowym [c; h]: c_{t+1} = A c_t + Σ_{j=0..t−1} B D_h^j C c_{t−1−j} + B D_h^t h₀. Kontrprzykład: A = 1/2, B = 1, C = 0, D_h = 1/3, czyli c_{t+1} = c_t/2 + 3^{−t} h₀; całe jądro jest zerowe, a c₀ nie wyznacza c₁. Obcięcie pamięci na L_m daje ogon co najwyżej ‖B‖‖C‖M ρ^{L_m}/(1 − ρ). Poprawny przykład stacjonarny dla kontrprzykładu z pliku `MoriZwanzig.lean`: r = 1/4, wielomian widmowy ≥ 3/8.
- T4: ile daje historia: E‖Q − E[Q|G]‖² = E‖Q − E[Q|H]‖² + Δ_hist(Q), Δ_hist ≥ 0; wersja liniowa przez log-wyznacznik, z dowodem M_Q^lin = 0 ⇔ B_c = 0 w pliku `MzLogdetMacierz.lean`.
- T5: dla gradientu L-lipschitzowskiego |G(d) − η⟨g, d⟩| ≤ (Lη²/2)‖d‖², gdzie G(d) = V(θ) − V(θ − ηd), η to długość kroku, a d to faktycznie zastosowany kierunek zmiany wag. Dla optymalizatora AdamW (odmiana spadku gradientu z pamięcią uśrednionych gradientów i zanikiem wag) nie wolno podstawiać za d zwykłego gradientu paczki, jak w SGD (stochastycznym spadku gradientu).
- T6: przy ‖g − ĝ‖ ≤ ε i ‖d_S − d̂_S‖ ≤ δ_S poprawa leży w przedziale [â_S − E_S, â_S + E_S], â_S = η⟨ĝ, d̂_S⟩, E_S = η[ε‖d̂_S‖ + (‖ĝ‖ + ε)δ_S] + (Lη²/2)(‖d̂_S‖ + δ_S)². Przewaga partii S nad B wymaga â_S − E_S > â_B + E_B. Wybór partii o największej przewidywanej poprawie traci wobec najlepszej co najwyżej 2E_max.
- T7: dla liniowego odczytu i straty kwadratowej błąd cech e przenosi się na błąd gradientu co najwyżej (2MR_z + Y_max)e; T7a łączy to z błędem przewidywania równania z pamięcią w pełny łańcuch do przedziału poprawy.
- T8: pokrycie z wagami przybliża gradient całej puli z błędem (L_g/|U|) Σ d(χ_i, χ_{r(i)}); kontrprzykład wag: dziewięć punktów z gradientem +1 i jeden z −1 dają średnią 0,8, dwaj reprezentanci bez wag dają 0.
- T9: samo widmo nie wyznacza użyteczności klatki: dwie kandydatki o identycznym, idealnie stałym opisie Koopmana i zerowej pamięci mają gradienty +1 i −1; pierwsza poprawia stratę o η − η²/2, druga pogarsza o η + η²/2. Biały składnik latentu może nieść całe zadanie (Z_t = Y_t).
- Nieaddytywność grupy (28), zapis doboru jako problemu Bellmana (29).
- T10: przy nierówności Polyaka-Łojasiewicza ‖∇V‖² ≥ 2μ(V − V*) i błędzie kierunku ‖e_k‖ ≤ α‖g_k‖ (α < 1) oraz kroku η ≤ (1 − α)/(L(1 + α)²) zachodzi V(θ_k) − V* ≤ q^k (V(θ₀) − V*), q = 1 − μη(1 − α).
- T11: dla m procedur ustalonych niezależnie od zbioru oceny z n niezależnymi operacjami i stratą w [0, B_ℓ], z prawdopodobieństwem 1 − δ wszystkie różnice ryzyka empirycznego i prawdziwego są nie większe niż B_ℓ sqrt(log(2m/δ)/(2n)) (nierówność Hoeffdinga); tysiące sąsiednich klatek nie są tysiącami niezależnych operacji.
- Certyfikat nasycenia jednokrokowego (35): jeśli przedział T6 obowiązuje jednocześnie dla wszystkich dopuszczonych partii i sup(â_S + E_S) ≤ τ, to najlepsza możliwa poprawa jednym krokiem z tej puli jest ≤ max(0, τ).

Sekcja 12 dokumentu wiąże to z implementacjami: K5/K6 liczą różnorodność w wybranych współrzędnych, bez regularności gradientów w tej metryce; pilot `dobor_klatek_stan_sieci` (równoległa sesja z 22.09) dał w pojedynczym biegu AP 0,67317 dla Koopman-GLE wobec 0,65547 dla losowania, przy czasie selekcji i treningu około 27,12 minuty wobec 20,38 minuty; K6 0,62959 wobec losowych 0,63193 nie wspiera przewagi samego zapełniania komórek. Procedura wynikająca z dowodów (sekcja 13) ma 8 kroków, a cel dla Koopmana-Langevina brzmi: „przy mniejszym koszcie zachować wystarczająco dokładne przewidywanie użyteczności grup klatek, aby spełnić (23) albo kontrolować stratę względem najlepszego wyboru przez (24)”.

**Sprawdzenie obliczeniowe.** `WERYFIKACJA.json` (12:02; SymPy 1.13.1, czyli biblioteka rachunków symbolicznych w Pythonie, i NumPy 2.2.5 do rachunków numerycznych; ziarna 20260923 i 44): 20 kontroli, wszystkie zaliczone. Przykładowe wartości: najmniejszy zapas przedziału T6 na 2400 kierunkach i 300 celach kwadratowych 0,0036; strata wyboru minus ograniczenie najwyżej −0,036; najmniejszy zapas T7 3,40; T10 0,097; niepusty warunek przewagi (ĝ = 1, ε = δ = 0,01, L = 1, η = 0,25): dolna granica wyboru 0,2131 wobec górnej granicy odniesienia −0,2131; T8: średnia pełna 4/5, ważona 4/5, bez wag 0; T7a na 100 układach: najmniejsze zapasy 0,0226 (pamięć), 0,0581 (rekonstrukcja cech), 0,0442 (gradient), 0,0046 (przedział poprawy). Skrypt zastrzega, że skończona liczba przykładów nie dowodzi nierówności uniwersalnych.

**Lean.** Lean to asystent dowodzenia: program, który mechanicznie sprawdza każdy krok dowodu; Mathlib to jego biblioteka matematyczna. Plik `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\ViTFrameSelection.lean` sprawdzono w Lean 4.28.0 z Mathlib w rewizji `8f9d9cff`, 23.09 o 12:02 (12,9 sekundy, status „passed”, bez `sorry` i `admit`, czyli bez pominiętych kroków, a jedyne aksjomaty to standardowe `propext`, `Classical.choice`, `Quot.sound`). Udowodnione twierdzenia, w przestrzeni z iloczynem skalarnym:

| nazwa | co mówi (założenia są hipotezami twierdzenia) |
|---|---|
| `alignment_error` | przy ‖g − ĝ‖ ≤ ε i ‖d − d̂‖ ≤ δ: \|⟨g, d⟩ − ⟨ĝ, d̂⟩\| ≤ ε‖d̂‖ + (‖ĝ‖ + ε)δ |
| `direction_norm`, `direction_norm_sq` | ‖d‖ ≤ ‖d̂‖ + δ oraz ‖d‖² ≤ (‖d̂‖ + δ)² |
| `gain_error` | przedział poprawy T6, przy ZAŁOŻONYM ograniczeniu reszty Taylora \|gain − η⟨g, d⟩\| ≤ curvature·‖d‖² (curvature odpowiada Lη²/2) |
| `certified_improvement` | dodatnie dolne ograniczenie poprawy implikuje mniejszą stratę |
| `certified_comparison` | rozdzielenie dolnej granicy wyboru i górnej granicy odniesienia implikuje przewagę wyboru |
| `selection_regret` | kandydat o nie większej przewidzianej poprawie wygrywa z wybranym co najwyżej o 2E |
| `particle_error` | przy tych samych wagach ‖Σ w_p z_p − Σ w_p z̃_p‖ ≤ Σ \|w_p\| e_p |
| `particle_gradient_error` | przeniesienie tego ograniczenia przez mapę gradientu o ZAŁOŻONEJ stałej Lipschitza |

Nie przeniesiono do Lean: dowodu T5 z gładkości, jawnej stałej T7, całego równania Moriego-Zwanziga, zbieżności T10 ani uogólnienia T11. Dokument sam rozdziela poziomy: udowodnione w modelu matematycznym (T1-T11), sprawdzone obliczeniowo, sprawdzone w Lean (tabela wyżej), zmierzone wcześniej, oraz nadal hipoteza badawcza: „że Koopman-Langevin przewiduje korzyść z klatek dostatecznie tanio i dokładnie, by poprawić AP RF-DETR i czas całej procedury względem mocnych bazowych metod”.

**Dokument 45: reszta rozszerzonego Langevina i niepewność Kalmana** (commit `e0e7d34e`). Wniosek: reszta może dostarczyć informacji o brakach przewidywania, a Kalman pozwala odnieść błąd do przewidywanej niepewności, ale ich połączenie nie daje jeszcze gwarancji przydatności klatki. Filtr Kalmana to rekurencyjny estymator ukrytego stanu liniowego modelu z szumem gaussowskim; w każdym kroku przewiduje stan, porównuje przewidywanie z pomiarem (różnica to innowacja) i poprawia stan oraz jego kowariancję. Treść:

- Dwie reszty w istniejącym kodzie rozszerzonego Langevina (`experiments\dobor_klatek_stan_sieci\model.py`): ẑ⁰_{t+1} = z_t + D_t + JP_t + W_t + Z_t^zd, r⁰ = z_{t+1} − ẑ⁰ (reszta bazowa), N̂ = prognoza z dawnych reszt (deterministyczna, nie nowy szum), e = r⁰ − N̂ (kandydat na innowację). Człony: D (dryf od bieżących cech, prędkości, przyspieszenia), JP (historia), W (wejścia i metadane), Z^zd (korekta zdarzeń przy utracie i odzyskaniu obserwacji), N̂ (przewidywalna część reszt). Duża norma członu to duża już zastosowana poprawka, a nie miara tego, czego brakuje.
- R1: E[‖e‖² | H] = ‖b‖² + tr Σ, gdzie b to błąd warunkowej średniej, Σ kowariancja innowacji. R2: poprawka c z historii pomaga średnio dokładnie wtedy, gdy ‖b − c‖ < ‖b‖. Pułapka: wybór klatek według zaobserwowanej reszty może odwrócić znak pozornej korzyści (przykład: poprawka pogarsza średni błąd o 1, a po wyborze przypadków z ε = 1 pozornie go poprawia o 1).
- Filtr Kalmana w postaci EKF (rozszerzony filtr Kalmana, który dla nieliniowej dynamiki linearyzuje ją w bieżącym oszacowaniu) z aktualizacją kowariancji w postaci Josepha (numerycznie stabilna forma, zachowująca symetrię i dodatnią określoność); trzy różne wielkości: ‖ν‖² (wielkość błędu), tr P⁻ (niepewność stanu), νᵀS⁻¹ν (błąd względem oczekiwanej skali). R3: E[νᵀS⁻¹ν | H] = tr(S⁻¹Σ) + bᵀS⁻¹b; przy S = Σ nadwyżka ponad wymiar q mierzy błąd warunkowej średniej.
- K1: P⁻ − P⁺ = P⁻HᵀS⁻¹HP⁻ ⪰ 0 (pomiar zmniejsza niepewność), ale pomiar samego tła może silnie zmniejszyć ogólną niepewność, nie zmieniając niepewności zadania (przykład P = diag(100, 1)). K2: przy ustalonych parametrach ciąg kowariancji nie zależy od wartości pomiarów. K3: od ograniczenia E[‖a‖² | G] ≤ M przez nierówność Markowa do ograniczenia błędu cząstek, gradientu i przedziału poprawy.
- Audyt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\kalman_state.py`: filtr stanu (położenie, prędkość, przyspieszenie) osobno na każdej osi obrazu; przycina innowacje, dopasowuje parametry kryterium Hubera, więc zwracane kowariancje są nominalne, a nie skalibrowane; pierwszy pomiar inicjuje położenie. Poprawiono tylko komentarze w kodzie, bez zmiany obliczeń. Zapisane wcześniej AUC rozpoznawania nietrafionej ramki (AUC, pole pod krzywą ROC, to tutaj prawdopodobieństwo, że losowo wzięta klatka z chybieniem dostanie wyższą wartość miary niż losowo wzięta klatka trafiona; 0,5 to rzut monetą, 1 to bezbłędne rozróżnienie): `ut_b0500_best` innowacja z odczytu latentu 0,6745, innowacja z ramki detektora 0,6780, minus pewność 0,8133; `ut_b2500_best` odpowiednio 0,6781, 0,6031, 0,8268.
- Kontrola `RESZTA_KALMAN_WERYFIKACJA.json` (12:19, Python 3.11.8): 12 kontroli zaliczonych, między innymi: wybór po reszcie odwraca znak (bezwarunkowo −1, po wyborze +1); postać Josepha na 300 układach (największy błąd tożsamości 1,07·10⁻¹⁴); pomiar tła zmniejsza ślad ogólny o 99,0, a wariancję zadania o 0; pominięcie korelacji cząstek zaniża wariancję czterokrotnie (1,0 wobec 0,25); drugi moment 101 = wariancja 1 + kwadrat obciążenia 100; łańcuch K3 na 100 układach: największy stosunek błędu do ograniczenia 0,324.

PDF z pełnym dokumentem 45 i dodatkiem odtwarzającym dowody T5, T6, T7 zbudował `zbuduj_pdf_langevin.py` (Pandoc i XeLaTeX): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\output\pdf\45_Reszta_Langevina_Kalman_dowody.pdf` z manifestem `...\45_Reszta_Langevina_Kalman_dowody.json` (commit `d2639cf5`); dodatek nie rozszerza zakresu sprawdzenia w Lean.

**Znaczenie dla dalszej pracy.** Formalizacja przesunęła cel z „szukania trwałych modów” na przewidywanie użyteczności klatek dla zadania detektora z kontrolowanym błędem. Żadne z założeń T5-T11 (gładkość RF-DETR, stała L, błąd względny kierunku α, niezależność operacji) nie zostało zmierzone dla detektora, więc gwarancje pozostają warunkowe.

### 2.9 Mozaiki klatek trzech ramion i to, jak naprawdę wybierał K6

**Pytanie.** O 11:39 użytkownik poprosił: „pokaz mi zdjecia które wybierał koopman. i potem pokaz te losowe próbki wybrane do treningu mozesz to zapisac te klatki w jakims folderze.”, wskazując oryginalny zbiór w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM`.

**Metoda.** Skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\pokaz_klatki_250.py` (commit `076872f8`, 11:43) wyjmuje klatki trzech ramion z archiwów adnotacji `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking\annotations\TR_xxxx.zip`. Klatka o identyfikatorze `.../TR_xxxx/000000NN` to plik `TR_xxxx_0000NN.jpg` w archiwum; zgodność sprawdzono skrótem pikseli z `pool_manifest.jsonl`. Na każdej klatce narysowano ramkę z etykiety, wyłącznie do pokazania narzędzia; sam wybór etykiet nie używał. Klatki leżą w `...\krystalizacja_latentu\wyniki\klatki_250\K6_po_kwadratach`, `...\B1m_losowo` i `...\B2m_rownomiernie`, a mozaiki po 250 klatek w `MOZAIKA_K6_po_kwadratach.jpg`, `MOZAIKA_B1m_losowo.jpg`, `MOZAIKA_B2m_rownomiernie.jpg` w tym samym katalogu (pliki JPG są poza gitem).

**Obserwacja (11:43).** Na oko zestawy K6 i losowy wyglądają prawie tak samo: te same oczy, podobne światło, podobne narzędzia, co pasuje do tego samego mAP. K6 sięgnął do 90 filmów, losowanie do 98, więc K6 częściej wracał do tego samego filmu; w ostatnich rzędach jego mozaiki powtarzają się podobne pomarańczowo-czerwone kadry.

**Jak wybierał K6** (odpowiedź z 11:50 na pytanie „a jak wybierałeś klatki koopmanem?”). Klatek nie wybierano bezpośrednio Koopmanem, tylko przez siatkę na tym, co Koopman uznał za tożsamość: wstępny detektor przejrzał pulę bez etykiet (prawie 10 tysięcy klatek), z wnętrza przewidzianej ramki powstał opis 48 liczb, z par klatek tego samego filmu odległych o 5 klatek nagrania wyszło 10 kierunków wracających po kroku prawie takich samych, odrzucono te stałe przez cały film, trzy pozostałe osie podzielono na 16 przedziałów i z każdej zajętej komórki, od najliczniejszych, brano po jednej losowej klatce, aż zebrało się 250. Dwie z trzech osi okazały się tą samą liczbą, więc siatka była płaska i miała 252 zajęte komórki, prawie tyle, ile wybierano klatek. W tej odpowiedzi wstępny detektor opisano jako uczony na 1500 klatkach; o 12:09 asystent poprawił to na detektor `ut_b0500` (500 klatek).

### 2.10 Pobudzenie kierunków tożsamości dla pojedynczej klatki

**Skąd.** O 11:52 użytkownik napisał: „to trzeba jakoś zrobić detektor klatek z tego wstępnie wytrenowanego detektora na 1500 klatkach i koopmana. Jakie masz pomysły? możesz uzywać sieci i /paper-search”. Asystent (11:54) podał cztery pomysły z literatury: spójność w czasie, czyli klatki, na których ramka znika, skacze albo miga pewnością (Ben19, „Temporal Coherence for Active Learning in Videos”); zgodność pod zniekształceniem, czyli różnica ramek dla klatki odbitej lub przeskalowanej (metoda CALD, Yu21, „Consistency-based Active Learning for Object Detection”, 2-3 punkty mAP ponad losowanie); zaskoczenie Koopmana z Kim26 jako podstawą porównania; oraz łączenie każdej oceny z rozrzuceniem wyboru (Hek23, „Non-Redundant Informative Sampling”). Ostrzeżenie z Aye23 („Data pruning and neural scaling laws”): przy budżecie około 2,5 procent puli losowanie jest bardzo trudne do pobicia. Propozycja: tani test offline na puli z etykietami przed jakimkolwiek treningiem. Użytkownik odpowiedział o 11:56: „bardziej mi zalezy na stworzeniu sposobu detekcji obrazu i okresleniu ze ten obraz mocno podbija w latencie kierunki tożsamosci wyznaczone przez koopmana.”

**Metoda** (`...\krystalizacja_latentu\pobudzenie_tozsamosci.py`, commit `de3e5196`, 11:58). Dla jednej klatki, bez etykiet: opis z (48 liczb jak w sekcji 2.3) i baza ortonormalna Q przestrzeni tożsamości złożona z modów 5, 6 (para 6-7 jako część rzeczywista i urojona), 8 i 9 (λ = 0,901; 0,883 ± 0,016i; 0,878; 0,863), wymiar 5. Trzy oceny: amplituda A(x) = |Qᵀ(z − μ)|, czyli jak daleko klatka leży wzdłuż kierunków tożsamości; udział S(x) = |Qᵀ(z − μ)|² / |z − μ|², czyli jaka część odchylenia klatki idzie w tożsamość (dla przypadkowego kierunku 5/48 = 0,104; mediana w puli 0,083); wpływ I(t), czyli pierwszy rząd zmiany wartości własnej modów tożsamości po dołożeniu pary klatek (t, t + 1) do dopasowania (dodatni: para czyni tożsamość trwalszą). Etykiety weszły dopiero na końcu, do opisu różnic. Pula: 9683 klatki. W opisie skryptu jako źródło latentu wymieniono „B2, 1500 klatek”, a faktycznie użyto nagrania `CZASTKI_pool_b0500`.

**Wyniki** (`...\krystalizacja_latentu\wyniki\pobudzenie_tozsamosci\REPORT.json`, `OCENY.npz`, mozaiki `MOZAIKA_udzial_najmocniej.jpg`, `MOZAIKA_udzial_najslabiej.jpg`, `MOZAIKA_wplyw_najmocniej.jpg`, `MOZAIKA_wplyw_najslabiej.jpg`). Korelacje rangowe ocen z cechami ramki (z etykiet):

| ocena | wydłużenie ramki (log długość/szerokość) | wielkość ramki (log px²) | IoU detektora wstępnego | pewność |
|---|---|---|---|---|
| amplituda | −0,39 | 0,22 | 0,09 | −0,01 |
| udział | −0,32 | 0,24 | 0,09 | 0,05 |
| wpływ | −0,03 | 0,09 | 0,04 | 0,05 |

Średnie w grupie ocenionej najwyżej wobec najniżej: log wydłużenia 0,45 wobec 1,40 (amplituda), 0,42 wobec 1,30 (udział), 0,48 wobec 0,65 (wpływ); log wielkości 9,88 wobec 9,37 i 10,03 wobec 9,31; IoU 0,915 wobec 0,883 i 0,912 wobec 0,879; pewność prawie bez różnicy (0,913 wobec 0,910). Według rozmowy najmocniej pobudzają klatki z cystotomem ułożonym ukośnie, najsłabiej te, na których leży poziomo, długi i cienki; ramka w najmocniejszych jest prawie trzy razy mniej wydłużona; z 60 najlepiej ocenionych klatek aż 27 pochodzi z jednego filmu. Ocena „wpływ” jest niezależna od pierwszych dwóch i prawie nie wiąże się z wyglądem narzędzia.

**Interpretacja.** „Kierunki tożsamości” z Koopmana to w praktyce kąt ustawienia narzędzia, zmieszany z wyglądem konkretnej operacji. Rodzaju narzędzia nie dało się sprawdzić, bo w puli jest tylko cystotom.

**Pytanie użytkownika i odpowiedź (12:01-12:02).** Użytkownik: „czyli koopman pokazuje ze siec potrzebuje róznych kierunków (obrotów narzedzia)? i wtedy tożsamosc sie bardziej domyka? […] podczas uczenia rozpoznawania narzedzia nie trzeba nam rozpoznawac operacji ani osoby. Jeżeli jest tylko cystostom to można dac np klatki z filmu gdzie nie ma narzedzia albo zrobic maski ze jest tylko narzedzie i szare tło. Moze to jakoś pomoze ?”. Asystent: tylko w połowie; Koopman pokazuje, że najwolniej zmieniają się kierunki związane z kątem, a nie że sieć potrzebuje więcej kątów; prawdziwa tożsamość nie powinna się zmieniać przy obrocie, a te kierunki się zmieniają. Klatki bez narzędzia raczej nie pomogą: w pakiecie danych ich nie ma (każda z 22 680 klatek ma dokładnie jedno narzędzie), a detektor częściej narzędzie przegapia, niż wskazuje je fałszywie (na walidacji 2824 pominięte wobec 562 fałszywych; źródła nie podają, którego detektora dotyczą te liczby). Zamiast szarego tła, na którym sieć uczyłaby się w warunkach niespotykanych na filmie, zaproponowano copy-paste: wycięcie narzędzia i wklejenie go na klatkę innej operacji, co zrywa związek narzędzia z wyglądem konkretnego oka (Yoo23, „Exploring Video Frame Redundancies”). Zaproponowano też lokalnie Koopmana na opisie po odjęciu średniej filmu oraz na Edenie 250 losowych klatek z wklejaniem wobec tych samych bez wklejania, po zatwierdzeniu. Żadnej z tych dwóch rzeczy nie uruchomiono w etapie 2.

**Jak sprawdzać domknięcie (12:03-12:05).** Użytkownik: „a jak sprawdzac ze tożsamosc na latencie sie domyka? /paper-search”. Odpowiedź: trzy testy. Pierwszy według Len14 (Lenc, Vedaldi, „Understanding Image Representations by Measuring Their Equivariance and Equivalence”): reprezentacja φ jest równoważna (ekwiwariantna) względem przekształcenia obrazu g, gdy φ(gx) ≈ M_g φ(x) dla pewnej macierzy M_g; kierunki, na których M_g działa jak identyczność, są niezmiennicze i tam sieć zapisuje, co to jest, a kierunki przesuwane przez M_g zapisują, jak narzędzie leży. Drugi według Coh14 (Cohen, Welling, „Transformation Properties of Learned Visual Representations”) i Mar22b (Marchetti i in., „Equivariant Representation Learning via Class-Pose Decomposition”): w dobrze ułożonym latencie obrót rozkłada się na część stałą (tożsamość) i pary współrzędnych kręcących się jak wskazówki zegara (położenie); nasze mody tożsamości zanikają o 10-14 procent na krok, więc nie są częścią stałą. Trzeci: czy obie miary przestają się zmieniać przy dokładaniu klatek z nowymi kątami. Rozstrzygnąć może tylko eksperyment.

### 2.11 Test obrotu, wydłużenia i tła na czterech gotowych detektorach

**Skąd.** O 12:08 użytkownik napisał: „to co ja chce robic to szukac tożsamosci narzedzia w latencie. gdzie siec zapisuje w latencie ze to jest to konkretne narzedzie plus jego wariacje: czyli jego różne obroty wydłuzenia skrócenia itd! zrób te dwa testy i sprawdzaj przy kazdej klatce przechodzacej przez siec jak sie uczy i jak sie zmieniają te wartosci w latencie! wykrzystaj te wiedze. Mozesz czytac pdfy w undermind. Rób pomiar co klatkę.” Asystent (12:09) opisał plan: dwa pomiary na gotowych detektorach (test Lenca i Vedaldiego oraz rozkład na bloki) i trzeci pomiar w trakcie treningu (sekcja 2.12), z uwagą, że pomiar dosłownie po każdej klatce nie ma sensu, bo sieć zmienia wagi raz na paczkę, więc najdrobniejszą rozdzielczością jest jeden krok uczenia. O 12:12 użytkownik zatwierdził: „dobra puszczaj na eden”.

**Metoda.** Skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\tozsamosc_obrot.py`, na Edenie przez `...\SmallViTAblation\eden\run_tozsamosc_obrot.slurm` (zadanie 1809793, węzeł dgx-1, 1 minuta 44 sekundy). Dane: 300 klatek walidacyjnych z 25 filmów, których żaden detektor nie widział. Wersje każdej klatki: `orig`; `rot_±15`, `rot_±30`, `rot_±45` (obrót całego obrazu wokół środka ramki narzędzia); `dl_0.7`, `dl_1.4` (ściśnięcie lub rozciągnięcie wzdłuż dłuższego boku ramki, udające skrócenie i wydłużenie); `tlo_a`, `tlo_b` (wycinek z narzędziem wklejony z miękką krawędzią na klatkę innej operacji). Detektory: `ut_b0500`, `ut_b1500` (B2: 1500 klatek równomiernie w czasie), `B1m_250` i `K6_250` (najlepsze wagi biegów z ziarnem 42 z sekcji 2.5). Zapisywano bez etykiet: latent zapytania (256 liczb), tokeny ramki (średnia tokenów backbone w przewidzianej ramce, 384 liczby), dla `ut_b0500` także opis Koopmana (3 × 3 × 32), ramkę i pewność; ramka z etykiety po przekształceniu służyła tylko do oceny.

Analiza (`...\krystalizacja_latentu\analiza_obrotu.py`; raport `...\krystalizacja_latentu\wyniki\REPORT_obrot.json`, dane `...\krystalizacja_latentu\dane\obrot\META.json`, `OBROT_ut_b0500.npz`, `OBROT_ut_b1500.npz`, `OBROT_B1m_250.npz`, `OBROT_K6_250.npz`, podglądy `PODGLAD_0.jpg`-`PODGLAD_3.jpg`; commit `684cc3c4`): (1) IoU i pewność po przekształceniu; (2) zmiana względna R_g = średnia |L(gx) − L(x)|² / średnia |L(x) − średnia|², gdzie 0 znaczy, że latent się nie rusza, a 1, że rusza się tyle, ile różnią się od siebie różne klatki; (3) kierunki niezmiennicze z uogólnionego zagadnienia własnego S_zmian w = λ S_klatek w w 32 składowych PCA (kierunki, które przekształcenia ruszają mało w porównaniu z różnicą między klatkami), dopasowane na połowie filmów i sprawdzane na drugiej, z progiem λ < 0,1; (4) równoważność obrotu: mapa liniowa L(x) → L(gx) dopasowana na połowie filmów, R² na drugiej, wobec mapy tożsamościowej, oraz kąty wartości własnych tej mapy; (5) dla `ut_b0500` współrzędne tożsamości i sceny z Koopmana oraz 5 losowych kierunków jako kontrola.

**Wyniki: wykrywanie po przekształceniu** (IoU z ramką po przekształceniu i pewność detekcji, mediany po 300 klatkach):

| detektor | IoU orig | IoU obrót (6 kątów) | IoU dł. 0,7 / 1,4 | IoU tło a / b | pewność orig | pewność obrót | pewność dł. 0,7 / 1,4 | pewność tło a / b |
|---|---|---|---|---|---|---|---|---|
| ut_b0500 | 0,880 | 0,440-0,691 | 0,871 / 0,825 | 0,786 / 0,779 | 0,902 | 0,868-0,919 | 0,895 / 0,886 | 0,548 / 0,572 |
| ut_b1500 | 0,892 | 0,441-0,698 | 0,878 / 0,831 | 0,813 / 0,810 | 0,867 | 0,852-0,885 | 0,850 / 0,853 | 0,653 / 0,659 |
| B1m_250 | 0,867 | 0,436-0,694 | 0,861 / 0,812 | 0,774 / 0,755 | 0,828 | 0,785-0,847 | 0,814 / 0,821 | 0,380 / 0,366 |
| K6_250 | 0,877 | 0,430-0,689 | 0,851 / 0,802 | 0,769 / 0,762 | 0,905 | 0,889-0,924 | 0,897 / 0,895 | 0,635 / 0,609 |

**Wyniki: zmiana względna latentu R_g**:

| detektor | latent (wymiar) | obrót | wydłużenie | tło |
|---|---|---|---|---|
| ut_b0500 | zapytanie (256) | 1,49 | 0,50 | 2,11 |
| ut_b0500 | tokeny ramki (384) | 1,56 | 0,35 | 1,59 |
| ut_b0500 | opis Koopmana (48) | 1,60 | 0,58 | 1,83 |
| ut_b0500 | współrzędne „tożsamości” Koopmana (5) | 1,31 | 0,28 | 1,35 |
| ut_b0500 | współrzędne sceny Koopmana (5) | 1,25 | 0,15 | 0,79 |
| ut_b0500 | 5 losowych kierunków | 1,58 | 0,61 | 1,63 |
| ut_b1500 | zapytanie / tokeny ramki | 1,46 / 1,54 | 0,46 / 0,30 | 1,87 / 1,35 |
| B1m_250 | zapytanie / tokeny ramki | 1,49 / 1,55 | 0,46 / 0,39 | 2,00 / 1,56 |
| K6_250 | zapytanie / tokeny ramki | 1,57 / 1,54 | 0,58 / 0,40 | 2,26 / 1,54 |

Zmiana zapytania `ut_b0500` rośnie z kątem: −45° 1,83; −30° 1,37; −15° 0,89; +15° 1,05; +30° 1,68; +45° 2,11. Kierunków niezmienniczych (λ < 0,1 na filmach sprawdzających) nie znaleziono w żadnym latencie żadnego detektora: 0 z 32; najlepszy kierunek na filmach sprawdzających miał λ około 0,39 (zapytanie `ut_b0500`) i 0,41 (tokeny). Mapa równoważności obrotu tłumaczy małe obroty, ale nie duże: R² dla zapytania `ut_b0500` wynosi 0,29 i 0,30 przy ±15°, −0,04 przy ±30°, −0,29 i −0,37 przy ±45°, a przy wydłużeniu 0,60 i 0,56 (dla tokenów ramki wszystkich detektorów R² od 0,18 do 0,47 przy ±15° i od −1,13 do −0,25 przy ±45°). Kąty wartości własnych mapy nie odpowiadają kątowi obrotu (na przykład dla −30° pary przy 51,3°, 0,8° i 18,0°).

**Interpretacja** (commit `684cc3c4` i odpowiedzi z 12:35 i 13:14). W żadnym latencie nie ma kierunku stałego przy obrocie narzędzia i zmianie tła. Obrót zmienia latent bardziej niż przejście do innej klatki (R_g > 1), wydłużenie niewiele (0,3-0,6), podmiana tła najmocniej i obniża pewność detekcji (do 0,37-0,65). Kierunki, które Koopman nazwał tożsamością, ruszają się przy obrocie (1,31), tylko trochę mniej niż kierunki losowe (1,58); to potwierdza, że niosą raczej kąt ułożenia niż tożsamość. Sieć nie zapisuje więc tożsamości osobno od ułożenia i od wyglądu operacji, przynajmniej w liniowych kierunkach tych latentów. Zastrzeżenie, które wyszło później (sekcja 2.12): warianty `tlo_a` i `tlo_b` wklejały prostokątny wycinek z własnym tłem, co tworzyło widoczną „naklejkę”; spadek pewności na obcym tle był w dużej części artefaktem.

**Rdzeń klasy w dekoderze (obliczenia doraźne, bez pliku raportu).** O 12:40 użytkownik przedstawił hipotezę: „to ma sens bo koopman znajduje tozsamosc ale inne wektory musza programowac sektory odpowiedzialne za rozpoznawanie narzedzia na roznym tle albo pod roznym katem ale wydaje mi sie ze nawet wtedy czesc wektorów która mowi i stanowi core tozsamosci narzedzia musi byc odpowiedzialna za wykrywanie ze to jest to konkretne narzedzie. Co o tym myslisz?”. Asystent odpowiedział, że w detektorze DETR decyzję „to jest narzędzie” podejmuje jeden kierunek w latencie zapytania, odczytywany przez ostatnią warstwę jako pewność, a Koopman go nie znalazł, bo ten kierunek prawie się nie zmienia między klatkami z narzędziem. Doraźne rachunki na danych `OBROT_*.npz` (poza plikami raportów, tylko w transkrypcie): zmiana względna logitu pewności (log(p/(1 − p))) przy obrocie 1,26-1,88, przy wydłużeniu 0,37-0,61, przy podmianie tła 7,21-7,82 (dla `ut_b0500` 1,58, 0,61 i 7,66; mediana logitu 2,22 na oryginale i 0,19 na tle `tlo_a`). O 13:23: regresja grzbietowa (najmniejsze kwadraty z karą za normę współczynników) logitu na latencie zapytania, dopasowana na połowie filmów, odtwarza logit na drugiej połowie z R² 0,999 (`ut_b0500`), 0,998 (`ut_b1500`), 0,997 (`B1m_250`) i 0,999 (`K6_250`); ten kierunek niesie tylko 1,9-3,3 procent wariancji zapytania (kierunek losowy 0,39 procent) i nie pokrywa się z żadnym głównym kierunkiem (największy cosinus 0,25-0,36); rzut na niego zmienia się przy obrocie o 1,25-1,86, przy wydłużeniu o 0,36-0,61, przy tle o 7,17-7,90. W rozmowie podano to jako: pewność przy obrocie spada z 0,90 do 0,87, na obcym tle do około 0,55, a zmiana rdzenia jest około 7-8 razy większa niż między dwiema zwykłymi klatkami. Użytkownik odrzucił jednak dekoder jako przedmiot badania (sekcja 2.13).

### 2.12 Sonda tożsamości po każdym kroku uczenia

**Metoda.** Sonda to stały zestaw klatek kontrolnych, przepuszczany przez sieć po każdym kroku uczenia bez wpływu na samo uczenie, żeby zapisać, jak zmienia się latent. Skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\sonda_treningu.py` opakowuje `pipeline\train\train_rfdetr.py`: podmienia funkcję `rfdetr.training.build_trainer`, żeby do trenera biblioteki Lightning (biblioteki, która prowadzi pętlę uczenia) doszedł jeden dodatkowy callback, czyli funkcja wywoływana po każdym kroku. Przepis uczenia (optymalizator, harmonogram, augmentacje, wczesne zatrzymanie) zostaje bez zmian. Sonda to 32 klatki walidacyjne w 6 wersjach: `orig`, `rot_-30`, `rot_+30`, `dl_1.4`, `tlo_a` (narzędzie na tle innej operacji) i `nast` (ta sama scena 5 klatek później, prawdziwy ruch). Po każdym kroku optymalizatora sieć w trybie ewaluacji (bez gradientu, na surowych wagach, a nie na ich wykładniczo uśrednionej kopii EMA) liczy dla sondy wyjście dekodera najlepszego zapytania, średnią tokenów backbone w przewidzianej ramce, ramkę i pewność; zapisywane są też identyfikatory klatek uczących z tego kroku. Zapis `SONDA.npz` i `SONDA_META.json` co 100 kroków i na koniec. Bieg: ramię B1m (250 losowych klatek, ziarno 42), protokół `protocol_maly_budzet.yaml`, skrypt `...\SmallViTAblation\eden\run_sonda_treningu.slurm`.

**Awarie i naprawy.** Zadania 1809797 i 1809798 padły po kilku sekundach z błędem `PIL.UnidentifiedImageError: cannot identify image file ...TR_0002_S1_P03_00002381.jpg`, choć plik był poprawnym obrazem JPEG 720 × 480. Przyczyna: plik `select.py` w katalogu `pipeline\koopman_langevin` zasłaniał standardowy moduł Pythona `select`, przez co nie działał moduł `subprocess`, a za nim wtyczka JPEG biblioteki PIL. Naprawa: moduł `tozsamosc_obrot` ładowany przez `importlib` bez dopisywania katalogu do ścieżki importu; skrypt SLURM sprawdza dodatkowo, czy rozpakowano co najmniej 22 680 plików JPG. Zadanie 1809799 padło na łączeniu obrazów w tensor (`torch.stack`), bo część klatek walidacyjnych ma rozmiar 720 × 576; naprawa: każda klatka jest skalowana w PIL do 384 × 384 (commit `d6fbf761`, 12:26). Zadanie 1809800 przeszło: 27 minut 47 sekund, 337 pomiarów (kroków), 21 epok, najlepsze mAP 0,6256 w epoce 5. W planie z 12:09 szacowano około 4000 kroków; wczesne zatrzymanie skróciło bieg do 337.

**Analiza** (`...\krystalizacja_latentu\analiza_sondy.py`; raport `...\krystalizacja_latentu\wyniki\REPORT_sonda_B1m_s42.json` i `REPORT_sonda_B1m_s42.png`; dane `...\krystalizacja_latentu\dane\sonda_B1m_s42\` z `MAPA_ID_KLATEK.json`, gdzie image_id to indeks w `frame_ids` plus 1, oraz `TELEMETRIA.json` i `SONDA_META.json`; commit `9874e90a`, 13:03). Dla każdego kroku zmiana względna R_g każdej wersji; najbardziej niezmienniczy pojedynczy kierunek (uogólnione zagadnienie własne na wspólnej bazie PCA, dopasowane na połowie klatek sondy); pewność i IoU każdej wersji; przypisanie zmian klatkom uczącym: zmiana miary w danym kroku regresowana grzbietowo na obecność klatek w paczce, porównana z podłogą z permutacji.

**Wyniki.** mAP walidacji po kolejnych epokach (krok: mAP): 15: 0,004; 31: 0,381; 47: 0,349; 63: 0,563; 79: 0,614; 95: 0,626 (najlepsze); 111: 0,620; 127: 0,622; 143: 0,623; 159: 0,622; 175: 0,614; 191: 0,610; 207: 0,619; 223: 0,614; 239: 0,610; 255: 0,608; 271: 0,603; 287: 0,610; 303: 0,607; 319: 0,606; 335: 0,606. Epoka ma 16 kroków.

| latent | wersja | R_g na początku | minimum (krok) | R_g na końcu |
|---|---|---|---|---|
| zapytanie | obrót −30° | 2,39 | 1,44 (56) | 1,75 |
| zapytanie | obrót +30° | 1,73 | 1,41 (174) | 1,50 |
| zapytanie | wydłużenie 1,4 | 1,25 | 0,41 (91) | 0,62 |
| zapytanie | obce tło | 1,80 | 1,41 (56) | 2,73 |
| zapytanie | następna klatka | 1,12 | 0,36 (100) | 0,50 |
| zapytanie | najlepszy pojedynczy kierunek | 1,67 | 0,51 | 0,54 |
| tokeny ramki | obrót −30° | 1,94 | 1,26 (204) | 1,38 |
| tokeny ramki | obrót +30° | 1,82 | 1,46 (202) | 1,56 |
| tokeny ramki | wydłużenie 1,4 | 1,05 | 0,28 (104) | 0,33 |
| tokeny ramki | obce tło | 1,81 | 1,42 (70) | 1,72 |
| tokeny ramki | następna klatka | 0,64 | 0,23 (111) | 0,26 |
| tokeny ramki | najlepszy pojedynczy kierunek | 0,92 | 0,42 | 0,54 |

Pewność na końcu: `orig` 0,938, `rot_-30` 0,943, `rot_+30` 0,927, `dl_1.4` 0,940, `tlo_a` 0,256, `nast` 0,946. IoU na końcu: 0,866; 0,578; 0,401; 0,771; 0,665; 0,897. Przypisanie klatkom: największy współczynnik 0,0314 wobec podłogi permutacyjnej (95. centyl) 0,0382, czyli nie wychodzi ponad szum; listy 15 klatek „najbardziej ku niezmienniczości” i „najbardziej od niej” w raporcie nie mają więc znaczenia.

Galeria (`...\krystalizacja_latentu\galeria_sondy.py` z klasą `LeniwyKatalog`, która wyjmuje obrazy z archiwów na żądanie; pliki `...\wyniki\dowody\GALERIA_sondy.jpg`, `PEWNOSC_sondy.png`, `REPORT_galeria_sondy.json`, `WEJSCIE_MODELU_bez_rysunkow.jpg`; commit `2e61f938`): 6 klatek w krokach 0, 45, 56, 95 i 336. Mediana pewności na własnym i obcym tle: krok 0: 0,171 i 0,130; krok 45: 0,178 i 0,123; krok 56: 0,332 i 0,246; krok 95: 0,798 i 0,499; krok 336: 0,938 i 0,256. Klatek z obcym tłem poniżej 0,5: 32 z 32 w kroku 56 i 19 na końcu; z własnym tłem poniżej 0,5 na końcu: 2.

**Interpretacja w chwili pomiaru (13:03).** Sieć „znajduje” narzędzie około 45. kroku i wtedy latent przestaje reagować na wydłużenie i ruch (ich wpływ spada mniej więcej o połowę); na obrót reaguje przez cały trening podobnie mocno, bo nigdy nie widziała obróconych narzędzi; wrażliwość na podmianę tła jest najmniejsza w kroku 56 i potem rośnie do końca prawie dwukrotnie; pewność na obcym tle kończy na 0,26 przy 0,94 na własnym; najlepszy wynik walidacji w kroku 95 (epoka 5) leży blisko minimum wrażliwości. Wniosek sformułowany wtedy: rdzeń „to jest cystotom” powstaje wcześnie, ale im dłużej sieć się uczy, tym bardziej opiera go na oku wokół narzędzia, co tłumaczy przeuczenie po epoce 5 i przemawia za uczeniem z wklejaniem narzędzia na obce tła. Wcześniejszy komunikat z 12:35 (po pierwszych 100 krokach) mówił, że około 40. kroku pewność „skacze z 5 do 80 procent”; mediany galerii (0,18 w kroku 45, 0,33 w kroku 56, 0,80 w kroku 95) pokazują wzrost rozłożony na kilkadziesiąt kroków.

**Co się zepsuło: naklejka.** O 13:06 użytkownik poprosił: „możesz to wykazać na wykresach i poszczególnych zdjeciach i detekcjach modelu? Zrób taką porządną analizę z dowodami ze tak jest”, a o 13:12 zauważył: „Tam widzę problem ze chyba wklejasz zdjęcia narzędzie z ramką detekcji”. Asystent (13:14) wyjaśnił, że ramki są tylko rysowane na galerii, a model dostaje czysty obraz (`WEJSCIE_MODELU_bez_rysunkow.jpg`), ale przyznał, że problem jest: wariant `tlo_a` wkleja prostokątny kawałek z narzędziem i jego własnym tłem, więc model widzi wyraźną naklejkę, a obok niej zamazaną plamę po narzędziu z klatki dawcy; spadek pewności mógł wynikać ze sztuczności naklejki, a nie z obcego otoczenia. Do `tozsamosc_obrot.py` dodano kontrole `tlo_swoje` (ta sama klatka z zamazanym własnym narzędziem i wklejonym z powrotem wycinkiem: szew jest, kontekst ten sam) i `tlo_film` (klatka z tego samego filmu, odległa o co najmniej 250 klatek) i wysłano je w zadaniu 1809823 razem z pomiarem na wagach z kolejnych epok (`...\eden\run_tozsamosc_epoki.slurm`: 45 plików wag z epok z katalogu `results_telemetria` plus wagi sondy, wynik w `results_obrot_epoki`; skrypty analizy `analiza_epok.py` i `galeria_epok.py` napisane w commicie `2e61f938`). Zadanie wysłano o 13:09 do partycji `short`; Eden szacował start na 24.09 około 11:38, ale ruszyło wcześniej i skończyło się 23.09 po południu (16 minut 54 sekundy na dgx-4). Jego wyników w etapie 2 nie przeanalizowano. Test z narzędziem wyciętym po obrysie (sekcja 2.15) pokazał, że spadek pewności do 0,26 był artefaktem naklejki, więc wniosek o rosnącej zależności od tła wymaga powtórzenia z czystym wklejaniem.

### 2.13 Wektor narzędzia w tokenach backbone: z etykietami i bez etykiet

**Skąd.** O 13:22 użytkownik zapytał: „Ale czy ten rdzeń jest widoczny w latencie?”, a po odpowiedzi o dekoderze (sekcja 2.11) doprecyzował o 13:26: „Nie chodzi o to żeby on się zmieniał tylko żeby był zawsze aktywny gdy widziane jest narzedzie! To jest jeden wektor w latencie? Nie dotykam dekodera tylko badam Latent sieci”. Pytanie badawcze: czy w tokenach backbone istnieje jeden kierunek, który zapala się zawsze tam, gdzie widać narzędzie.

**Metoda z etykietami** (`...\krystalizacja_latentu\rdzen_w_tokenach.py`, commit `2e61f938`, 13:28). Latent: tokeny ostatniej warstwy backbone, 24 × 24 łatek, profil 32 liczb z nagrania `CZASTKI_pool_b0500` (detektor `ut_b0500`, bez etykiet przy zapisie); losowo 4000 klatek puli. Etykiety służą tylko do wskazania łatek na narzędziu: łatki, których środek leży blisko odcinka od czubka narzędzia (`tip_point`) do przeciwległego rogu ramki, czyli wzdłuż trzonka; tło to łatki poza ramką powiększoną o połowę. Kierunek wyznacza LDA (liniowa analiza dyskryminacyjna: różnica średnich dwóch klas przeskalowana odwrotnością kowariancji wewnątrz klas, czyli kierunek, wzdłuż którego klasy najlepiej się rozdzielają), dopasowana na połowie filmów (1849 klatek) i sprawdzana na drugiej połowie (1817 klatek z 59 filmów). Wektor jest „zapalony” w klatce, gdy najsilniejsza łatka narzędzia przewyższa 95. centyl łatek tła tej samej klatki; „kontrast” to (średnia aktywacja łatek narzędzia − mediana tła) / odchylenie standardowe tła.

**Wyniki** (`...\krystalizacja_latentu\wyniki\dowody\REPORT_rdzen_tokeny.json`, mapy `RDZEN_tokeny.png`, kierunek `RDZEN_kierunek_32.npy`):

| miara | wartość |
|---|---|
| AUC jednego kierunku, łatki narzędzia wobec tła, na filmach spoza dopasowania | 0,976 |
| AUC drugiego kierunku po usunięciu pierwszego | 0,924 |
| AUC pełnej regresji logistycznej (modelu liniowego zwracającego prawdopodobieństwo klasy) na 32 liczbach | 0,978 |
| cosinus kierunku regresji i kierunku LDA | 0,964 |
| udział klatek z zapalonym wektorem | 0,953 |
| udział zapalonych przy kącie narzędzia 0-30° / 30-60° / 60-90° | 0,950 / 0,954 / 0,968 |
| korelacja kontrastu z kątem / z wydłużeniem | 0,07 / −0,06 |
| najgorszy film | TR_0007_S1_P03, zapalony w 3,1 procent klatek |

Mapy pokazują, że wektor zapala się wzdłuż trzonka cystotomu, a na tle, oku i powiekach jest ciemny. Zastrzeżenia podane w rozmowie (13:28): drugi, prostopadły kierunek też częściowo odróżnia narzędzie (0,924), więc informacja jest zapisana nadmiarowo; pomiar dotyczy detektora z 500 klatek i latentu ściśniętego do 32 liczb; w klatkach ze słabym wektorem zapalał się on na innym narzędziu w kadrze, a ramka z adnotacji była maleńka przy brzegu obrazu, co wygląda raczej na niepełną adnotację niż błąd sieci. Koopman tego wektora nie znalazł, bo liczono go na uśrednionym wycinku ramki (siatka 3 × 3), a nie na pojedynczych łatkach. W rozmowie filmy oceny nazywano „filmami, których sieć nie widziała”; według skryptu są to filmy puli uczącej, niewykorzystane do wyznaczenia kierunku, a detektor `ut_b0500` uczył się na 500 klatkach z tej puli.

**Bez etykiet** (14:57-14:58). Po dyskusji o filtrze cząsteczkowym (sekcja 2.14) użytkownik napisał: „olej kalmana, teraz musze sprawdzic jak odkryć tożsamosc i potem jak ja domykac i jak potem tego użyć do wybierania najlepszych klatek uczacych siec wykrywania narzędzia!”. Skrypt `...\krystalizacja_latentu\rdzen_bez_etykiet.py` (commit `de56106d`, 14:58) wyznacza kierunek z łatek wewnątrz ramki przewidzianej przez sam detektor (pewność > 0,7) wobec łatek daleko poza nią, na połowie filmów; etykiety służą tylko do oceny na drugiej połowie, tak jak wyżej. Wariant „wzorzec” to średnia łatek narzędzia zamiast granicy między klasami, zgodnie z Pah22 („Revisiting Concept Activation Vectors”), gdzie granica między klasami łapie przy okazji sygnały z tła.

| wariant (`...\wyniki\dowody\REPORT_rdzen_bez_etykiet.json`) | AUC na filmach spoza dopasowania | cosinus z kierunkiem z etykiet |
|---|---|---|
| LDA bez etykiet (ramka przewidziana) | 0,975 | 0,877 |
| wzorzec bez etykiet | 0,959 | 0,664 |
| LDA z etykietami | 0,978 | 1 |

Wektor LDA bez etykiet zapisano w `...\wyniki\dowody\RDZEN_bez_etykiet_32.npy`; to on jest „wektorem narzędzia” we wszystkich dalszych testach etapu.

**Segmentacja (15:06-15:08).** Użytkownik przypomniał: „w zbiorze uczacym jest jeszcze segmentacja”. W plikach `annotation.csv` w archiwach Cataract-LMM jest kolumna `cystotome_seg` z wielokątem obrysu narzędzia; według odpowiedzi asystenta brakuje go tylko w 33 z około 2000 sprawdzonych klatek uczących. Na dokładnych obrysach zamiast linii trzonka wektor bez etykiet dał AUC 0,96, a z etykietami 0,969 (liczby tylko z rozmowy, bez pliku raportu). Asystent wskazał trzy zastosowania: czysty pomiar, które łatki leżą na narzędziu; wycinanie i wklejanie narzędzia po kształcie bez naklejki (sekcja 2.15); wklejanie narzędzia na obce tła w uczeniu.

**Interpretacja.** Pierwszy krok planu użytkownika („jak odkryć tożsamość”) działa: wstępny detektor wystarczy, żeby na każdej nowej, nieoznaczonej klatce zobaczyć, gdzie jego latent „widzi” narzędzie. Wynik nie dowodzi, że ten wektor jest tożsamością cystotomu w sensie rozróżniania narzędzi (w puli jest jedno narzędzie; na testach zapalał się też na innych cienkich narzędziach), ani że działa tak samo w innych detektorach.

### 2.14 Jak połączyć wektor, Koopmana i Langevina: filtr cząsteczkowy i rola Kalmana

**Pytanie (14:36).** Po podsumowaniu wniosków o 14:35 użytkownik zapytał: „No to co daje koopman i langevin w takim kontekście ? Jak to połączyć jaka metodę wybrać na szukanie tego narzędzia ? /paper-search”.

**Odpowiedź (14:37).** Podział ról: wektor narzędzia mówi, co i gdzie jest; Koopman mówi, jak narzędzie zmienia się z klatki na klatkę (obrót, skracanie, przesuwanie); Langevin to ruch z szumem i pamięcią, czyli przepis na przewidzenie położenia także wtedy, gdy narzędzie na chwilę zniknie. Błąd dotychczasowego podejścia: Koopmana liczono na opisie ramki mieszającym narzędzie z tłem; należy go liczyć w układzie samego narzędzia, na łatkach zapalanych przez wektor i śledzonych z klatki na klatkę. Wtedy test domknięcia jest prosty: aktywność wektora ma być modem Koopmana o wartości własnej dokładnie 1, a kąt i długość modami, które się kręcą albo zanikają. Jako metodę szukania narzędzia zaproponowano filtr cząsteczkowy, zgodny z nazwą projektu. Filtr cząsteczkowy utrzymuje wiele hipotez stanu (cząstek), każdą z wagą zależną od zgodności z obserwacją, przesuwa je modelem ruchu i losuje na nowo według wag; tu cząstka to hipoteza położenia, kąta i długości narzędzia, mapa wektora ocenia hipotezy, a Langevin z dryfem z Koopmana przesuwa cząstki do następnej klatki. Literatura: Pah22 (wektor jako wzorzec pojęcia), Mas21 (osobny model Koopmana na obiekt), Jun19 („Sequential Monte Carlo Filtering with LSTM Prediction”, filtr cząsteczkowy z wyuczonym ruchem odporny na zgubienie detekcji), Gre21b („Using Kalman Filter The Right Way”, dobrze dostrojony filtr Kalmana jako uczciwa podłoga, bo po dostrojeniu przewaga sieci potrafi zniknąć). Dobór klatek się wtedy zmienia: do oznaczenia brać klatki, na których wektor jest słaby lub niejednoznaczny, a nie klatki wskazane przez mody Koopmana.

**Doprecyzowania.** Na pytanie „filtr czasteczkowy w latencie ? czy na obrazie?” (14:47): cząstki na obrazie, ocena w latencie; cząstka to kilka liczb w pikselach (czubek, kąt, długość), a ocenia się ją odczytem tokenów leżących wzdłuż hipotetycznej linii narzędzia (siatka 24 × 24 jest przyłożona do obrazu). W samym latencie stan miałby setki liczb, głównie tła, a ruch nie miałby prostego sensu fizycznego. Na pytanie „ale dlaczego filtr czasteczkowy?” (14:48): mapa wektora często ma więcej niż jedno jasne miejsce (na przykład drugie narzędzie), a filtr Kalmana z jednym rozkładem gaussowskim uśredniłby dwóch kandydatów w miejsce bez narzędzia; filtr cząsteczkowy trzyma kilka grup hipotez i przy zasłonięciu pod tęczówką rozprasza cząstki, co wprost pokazuje niepewność. Wybór nie jest oczywisty: jeśli mapa prawie zawsze ma jedno maksimum, prostszy Kalman może działać tak samo; rozstrzygnąłby to pomiar, w ilu klatkach mapa ma dwa rozdzielone jasne miejsca (niewykonany).

**Korekta użytkownika (14:49-14:50).** Użytkownik: „ale filtr kalmana miał być do czegos innego. do Langevina zobacz jak próbowałem go wykorzystac w Project!”. Asystent przyznał, że pomieszał role: w `pipeline\koopman_langevin\kalman_state.py` stan narzędzia to położenie, prędkość i przyspieszenie, a przyspieszenie jest kolorowym wymuszeniem (niewidocznym pchnięciem ręki chirurga, które zanika powoli, jak proces Ornsteina-Uhlenbecka); Kalman wylicza z zaszumionych pomiarów położenia to, czego nie widać: prędkość i pchnięcie; jego innowacja wskazywała chybienia detektora lepiej niż przypadek (AUC 0,60-0,68, liczby jak w tabeli dokumentu 45 w sekcji 2.8). W projekcie grantowym (`rnn20_gle_emr_kalmannet`) pamięć równania Langevina zapisuje się jako dodatkowe ukryte zmienne, a Kalman z algorytmem EM (naprzemiennym szacowaniem ukrytych stanów i parametrów modelu) uczy się ich dynamiki. Układ w tym obrazie: wektor narzędzia daje pomiar (położenie trzonka), Langevin z Kalmanem daje ruch, a jego zaskoczenie wskazuje klatki z błędem detektora, Koopman sprawdza domknięcie i daje liniową część ruchu (macierz przejścia). Filtr cząsteczkowy byłby potrzebny tylko przy wielu jasnych miejscach, wtedy w każdej cząstce Kalman dalej liczy Langevina (filtr cząsteczkowy z Kalmanem w środku). Proponowany krok (podać do `kalman_state.py` położenie trzonka z mapy wektora) nie został wykonany, bo o 14:57 użytkownik napisał: „olej kalmana […]” i wrócił do trzech kroków: odkryć tożsamość, sprawdzić jej domknięcie, użyć jej do wyboru klatek. Kolejkowane pytanie użytkownika z 15:01, „a jak wygląda analiza czasteczek w latencie tutaj teraz. RObimy to jakos i czy ona cos daje ?”, nie ma osobnej odpowiedzi w końcowych wypowiedziach asystenta [nie ustalono w źródłach]; tematem cząstek zajmowały się tego dnia sesje równoległe (sekcje 2.8 i 2.23).

### 2.15 Test z narzędziem wyciętym po obrysie (lokalnie, RTX 3070)

**Skąd.** Po wykryciu artefaktu naklejki trzeba było powtórzyć test domknięcia z czystym wklejaniem. O 14:59 użytkownik zapytał: „a nie da sie lokalnie na mojej rtx policzyc ?”. Asystent zainstalował RF-DETR lokalnie w `F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1` (środowisko z dostępem do pakietów systemowych, rfdetr 1.10.1, torch 2.6.0+cu124, CUDA działa; instalacja przez `pip` z podstawieniem `<(echo)` się nie udała, więc użyto pliku ograniczeń wersji `ograniczenia.txt`) i pobrał z Edenu wagi `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\checkpoints\ut_b0500_best.pth`. Dane lokalne to archiwa `...\Dataset\Cataract-LMM\3_Object_Tracking\annotations\TR_xxxx.zip` z plikami `{film}/{film}_{nr:06d}.jpg` i `annotation.csv` (kolumny `cystotome_bbox`, `cystotome_tip`, `cystotome_seg`, obrysy rogówki i źrenicy); zgodność numeracji klatek sprawdzono skrótem pikseli.

**Metoda** (`...\krystalizacja_latentu\test_ksztaltu.py`, uruchomiony lokalnie 15:06-15:09). Narzędzie wycinane dokładnie po wielokącie `cystotome_seg`; własne narzędzie klatki usuwane przez `cv2.inpaint` (inpainting, czyli wypełnienie usuniętego obszaru na podstawie otaczających pikseli). Dane: 300 klatek walidacyjnych z 25 filmów. Wersje: `orig`; `usuniete` (narzędzie usunięte, nic nie wklejone: czy detektor widzi narzędzie tam, gdzie go nie ma); `swoje` (usunięte i wklejone z powrotem po obrysie: kontrola samego sklejania, otoczenie to samo); `obce` (wklejone po obrysie na klatkę INNEJ operacji z usuniętym jej narzędziem); `rot_-30`, `rot_+30` (samo narzędzie obrócone wokół czubka i wklejone na własne, nieruszone tło); `dl_0.7`, `dl_1.3` (samo narzędzie skrócone lub wydłużone wzdłuż osi od czubka). Miary dla `ut_b0500`: mediana pewności, mediana IoU z ramką narzędzia po przekształceniu, udział wykrytych przy IoU ≥ 0,5, oraz aktywacje wektora bez etykiet (`RDZEN_bez_etykiet_32.npy`) na łatkach pod narzędziem wobec tła tej samej klatki: mediana kontrastu i udział klatek z zapalonym wektorem. Do porównania drugi detektor `b1500` = `...\SmallViTAblation\results\eden_static36_2026-09-15\tracking_dino_coverage_b1500_s42\best.pth` (1500 klatek wybranych przez pokrycie w przestrzeni cech DINO, kampania z 15 września). Błąd po drodze: część wierszy `annotation.csv` wskazywała pliki, których nie ma w archiwum; naprawa: filtr przez listę plików archiwum.

**Wyniki** (`...\krystalizacja_latentu\wyniki\dowody\ksztalt\REPORT_ksztalt.json`, `KSZTALT_ut_b0500.npz`, `KSZTALT_b1500.npz`, podglądy wejść `WEJSCIA_0.jpg`-`WEJSCIA_2.jpg`; commit `1946d289`, 15:09):

| wersja | ut_b0500: pewność | IoU | wykryte przy IoU ≥ 0,5 | kontrast wektora | wektor zapalony | b1500: pewność | IoU | wykryte |
|---|---|---|---|---|---|---|---|---|
| orig | 0,905 | 0,887 | 0,870 | 3,16 | 0,977 | 0,887 | 0,897 | 0,913 |
| usunięte | 0,326 | - | 0,000 | 1,61 | 0,857 | 0,491 | - | 0,000 |
| swoje | 0,904 | 0,898 | 0,903 | 3,14 | 0,977 | 0,888 | 0,908 | 0,920 |
| obce | 0,849 | 0,912 | 0,770 | 2,26 | 0,813 | 0,844 | 0,913 | 0,803 |
| obrót −30° | 0,908 | 0,920 | 0,897 | 3,05 | 0,967 | 0,910 | 0,926 | 0,917 |
| obrót +30° | 0,870 | 0,896 | 0,843 | 2,80 | 0,980 | 0,888 | 0,913 | 0,910 |
| skrócenie 0,7 | 0,878 | 0,874 | 0,737 | 3,17 | 0,977 | 0,854 | 0,860 | 0,770 |
| wydłużenie 1,3 | 0,898 | 0,911 | 0,920 | 2,99 | 0,983 | 0,882 | 0,915 | 0,937 |

**Interpretacja (15:10).** Użytkownik miał rację co do naklejki: po wycięciu po obrysie i wklejeniu z powrotem na własną klatkę pewność się nie zmienia (0,90), a kontrast wektora też (3,14 wobec 3,16), więc wcześniejszy spadek do 0,26 był artefaktem prostokątnej naklejki. Wektor narzędzia zapala się w 97-98 procentach klatek po obrocie samego narzędzia o 30° i po zmianie jego długości, czyli jest domknięty względem kąta i długości w tym zakresie. Otoczenie ma wpływ umiarkowany: na tle innej operacji wektor zapala się w 81 procentach klatek, jego kontrast spada o około 30 procent, a pewność z 0,90 do 0,85 (wykrytych 77 wobec 87 procent); detektor `b1500` zachowuje się podobnie (0,84, wykrytych 80 procent). Przy skróceniu 0,7 wektor świeci w 98 procentach, ale ramka przy IoU 0,5 trafia tylko w 74 procentach, więc problemem jest geometria ramki, nie rozpoznanie. Po wymazaniu cystotomu detektor dalej zgłasza coś z pewnością około 0,33 (`b1500`: 0,49), a wektor „świeci” w 86 procentach w miejscu po narzędziu; podgląd pokazuje, że przy brzegu kadru zostaje rękojeść nieobjęta obrysem w adnotacji, plus ślady wypełnienia.

**Czego to nie dowodzi.** Domknięcie zmierzono na sztucznych edycjach samego narzędzia (obrót do 30°, zmiana długości do 30 procent), nie na naturalnej zmienności; wektor pochodzi z latentu jednego detektora (`ut_b0500`); „zapalony” to słabe kryterium (maksimum po łatkach powyżej 95. centyla tła). Wnioski sondy o rosnącej zależności od tła trzeba powtórzyć z tym czystym wklejaniem na wagach z kolejnych epok; w etapie 2 tego nie zrobiono.

### 2.16 Ocena detektorów na nietkniętym zbiorze testowym

**Skąd.** Po kompaktowaniu rozmowy użytkownik napisał o 15:14: „ok to co to wszystko oznacza? Co teraz moge zrobic z koopmanem i langevinem i jak to moge połączyc i zrobic narzedzie - (tutaj to moze byc detektor albo siec mniejsza albo cos innego) co znajduje klatki w których bedzie się poprawiał model podczas uczenia. Może ten aktualny model za szybko sie uczy skoro wystarcza my 6 epoch. Jak dobrze ten model inferuje po wyuczeniu wez checkpoint po najlepszej epoce i zobacz na jakims zbiorze testowym jak idzie mu znajdowanie narzedzia?”. Rano (8:58) asystent ustalił, że zbiór testowy (25 filmów) istnieje i nigdy go nie dotknięto.

**Metoda** (`...\krystalizacja_latentu\ocena_test.py`, lokalnie na RTX 3070, uruchomiony o 15:17 jako odłączony proces po próbie na co 400. klatce; rysunki `...\krystalizacja_latentu\wykresy_test.py`). Zbiór: 25 operacji testowych z `split_manifest.json`; co 10. klatka każdego filmu, razem z klatkami bez narzędzia: 8249 klatek, z czego 7908 z narzędziem i 341 bez. Miary: mAP 0,5:0,95, 0,5 i 0,75 (pycocotools, tylko klatki z narzędziem, jak w walidacji na Edenie); trafiona klatka: najlepsza ramka z pewnością ≥ 0,5 i IoU ≥ 0,5; „brak detekcji”: żadnej ramki z pewnością ≥ 0,5; „zła ramka”: pewna ramka z IoU < 0,5; fałszywy alarm: pewność ≥ 0,5 na klatce, w której adnotacja nie ma cystotomu; wyniki po filmach i najdłuższa seria kolejnych chybień (w próbkowanych klatkach). Dla `ut_b0500` dodatkowo wektor narzędzia: kontrast pod obrysem narzędzia (z etykiet) i „obecność” z samej mapy (najsilniejsza łatka wobec reszty kadru, bez etykiet). Detektory: `ut_b0500` (`results\koopman_langevin_epochs\checkpoints\ut_b0500_best.pth`) oraz z kampanii 15 września (`results\eden_static36_2026-09-15\`): `b500_best` (`tracking_random_b500_s42\best.pth`), `b1500_best` (`tracking_random_b1500_s42\best.pth`), `b1500_last` (`tracking_random_b1500_s42\last.pth`, epoka 30) i `b2500_best` (`tracking_random_b2500_s42\best.pth`); „random” oznacza losowy wybór klatek, ziarno 42.

**Wyniki** (`...\krystalizacja_latentu\wyniki\test\REPORT_test.json`, `TEST_ut_b0500.npz`, `TEST_b500_best.npz`, `TEST_b1500_best.npz`, `TEST_b1500_last.npz`, `TEST_b2500_best.npz`, `log.txt`, `err.txt`, rysunki `TEST_modele.png`, `TEST_przebieg.png`, `TEST_chybienia.jpg`; commit `9a6bd8a9`, 15:28, rysunki dodane mimo reguły ignorowania plików graficznych):

| detektor | mAP 0,5:0,95 | mAP 0,5 | mAP 0,75 | trafione klatki | brak detekcji | zła ramka | fałszywe alarmy bez narzędzia | mediana IoU trafionych | mediana pewności z narzędziem / bez |
|---|---|---|---|---|---|---|---|---|---|
| ut_b0500 (500, równomiernie) | 0,643 | 0,904 | 0,724 | 0,872 | 0,077 | 0,051 | 0,355 | 0,892 | 0,905 / 0,299 |
| b500_best (500, losowo) | 0,642 | 0,902 | 0,727 | 0,875 | 0,078 | 0,047 | 0,349 | 0,896 | 0,877 / 0,186 |
| b1500_best (1500, losowo) | 0,681 | 0,919 | 0,772 | 0,914 | 0,043 | 0,044 | 0,399 | 0,904 | 0,875 / 0,363 |
| b1500_last (ten sam bieg, epoka 30) | 0,654 | 0,900 | 0,737 | 0,907 | 0,027 | 0,065 | 0,499 | 0,896 | 0,948 / 0,499 |
| b2500_best (2500, losowo) | 0,690 | 0,926 | 0,783 | 0,927 | 0,028 | 0,046 | 0,428 | 0,905 | 0,896 / 0,380 |

Rozbicie według standardu COCO dla `ut_b0500` (`log.txt`): AP (średnia precyzja) dla obiektów małych 0,001, średnich 0,511, dużych 0,721; AR (średnia czułość, uśredniona po progach IoU) przy 1, 10 i 100 detekcjach na obraz 0,700, 0,732, 0,739. Najtrudniejszy film to TR_0125_S1_P03: trafione 0,606 klatek dla `b1500_best` (najdłuższa seria chybień 24 próbkowanych klatek), 0,614 dla `b2500_best`, 0,443 dla `b1500_last`, 0,406 dla `b500_best` i 0,371 dla `ut_b0500` (seria 30); kolejne trudne to TR_0097_S1_P03 (od 0,62 dla `b500_best` do 0,77 dla `b2500_best`) oraz TR_0090 (0,60 dla `ut_b0500`), TR_0083 i TR_0024. Klatki bez narzędzia ma 18 z 25 filmów, a udział fałszywych alarmów waha się między filmami od 0 do 1. Wektor narzędzia w latencie `ut_b0500`: zapalony w 99,1 procent klatek z narzędziem, mediana kontrastu 3,25; AUC słabego kontrastu (pod obrysem z etykiet) jako wskaźnika chybienia 0,761; AUC obecności z mapy (narzędzie wobec jego braku) 0,777; AUC pewności detektora w tym samym zadaniu 0,886. Czas oceny: 284 sekundy dla `ut_b0500` (z odczytem latentu), 74-78 sekund dla pozostałych.

**Interpretacja (15:25-15:29).** Najlepsza epoka detektora z 1500 klatek trafia narzędzie w 91 procentach klatek z nowych operacji, a jej mAP na teście (0,681) jest taki sam jak na walidacji (0,682 według etapu 1), więc wybór najlepszej epoki na walidacji nie zawyżył wyniku. Nie dotyczy to wszystkich operacji: na najgorszej trafia w 61 procentach. Ostatnia, trzydziesta epoka jest gorsza (0,654), bardziej pewna siebie w złych miejscach (fałszywe alarmy rosną z 40 do 50 procent), a jej ramki częściej są krzywe (zła ramka 6,5 wobec 4,4 procent); to potwierdza uczenie się klatek na pamięć po około siódmej epoce. Więcej klatek z tych samych filmów niewiele daje: z 2500 klatkami mAP rośnie tylko do 0,690. Galeria chybień (`TEST_chybienia.jpg`, detektor `b1500_best`) pokazała, że najczęściej zaznaczane jest inne cienkie narzędzie, na przykład pęseta albo drugi instrument, czyli sieć nauczyła się „cienkiego metalowego pręta w oku”, a nie samego cystotomu; to tłumaczy około 40 procent alarmów na klatkach bez cystotomu. Drugą przyczyną chybień jest nietypowe tło, na przykład torebka soczewki zabarwiona na niebiesko. Asystent zaproponował wtedy „wybieracz” klatek oparty na trzech liczbach (wektor tożsamości, zaskoczenie Koopmana, poprawka Langevina, z różnorodnością), z zastrzeżeniem, że najpierw trzeba go sprawdzić bez uczenia, oraz dodanie klatek z pęsetą bez cystotomu jako przykładów negatywnych.

**Czego to nie dowodzi i uwaga metodyczna.** Każdy detektor to jeden bieg z jednym ziarnem; oceniono co dziesiątą klatkę, a fałszywe alarmy liczono na tylko 341 klatkach bez narzędzia. Zbiór testowy użyto tu do diagnozy, więc przestał być w pełni nietknięty; w planach z 16:00 zapisano, że dalszy rozwój metod idzie na walidacji, a test zostaje do jednej oceny końcowej.

### 2.17 Literatura kuracji danych przez kierunek w latencie

**Pytanie (15:19).** W trakcie liczenia testu użytkownik uruchomił /paper-search z poleceniem: „szukaj literatury kuracji danych wykorzystującej identyfikacje w latencie tozsamosci i potem dekodowaniu latentu w taki sposób żeby wyznaczyc jakich cech potrzebujemy do domkniecia latentu tożsamosci obiektu”. Kuracja danych to świadome dobieranie lub poprawianie zbioru uczącego.

**Odpowiedź (15:25).** Nikt nie robi dokładnie tego samego, ale trzy elementy mają wzorce:

| praca | co robi | co z niej wynika dla projektu |
|---|---|---|
| Jai22b: Jain i in. 2022, „Distilling Model Failures as Directions in Latent Space” | liniowy klasyfikator „trafił/chybił” w latencie wyznacza kierunek błędów; dobór danych wzdłuż niego | +6,5 punktu na trudnej podgrupie wobec losowych; we własnym latencie uczonego modelu nie działało |
| Sin22b: Singla i in. 2022, „Data-Centric Debugging” | szuka sąsiadów błędnych obrazów w latencie DINO i dokłada je do uczenia | +9,5 punktu wobec losowych danych tej samej liczby |
| Neu25: Neuhaus i in. 2025, FeatInv | dekoder dyfuzyjny z przestrzennej mapy tokenów do obrazu; wycisza jeden kierunek i pokazuje różnicę pikseli | pasuje do siatki 24 × 24; koszt 45-60 godzin na 2 kartach |
| Bor21: Bordes i in. 2021, RCDM | dekoder z wektora cech; co stałe w próbkach, to zakodowane | backbone pamięta kąt, skalę i tło |
| Rou26c: Rousselot i in. 2026, „Show Me What You Don't Know” | generuje obrazy o tym samym latencie bez uczenia dekodera | pokazuje, na co model jest ślepy |
| Dou25: „Encoding-Decoding Direction Pairs” 2025 | oddziela kierunek „gdzie jest pojęcie” od kierunku „jak zostało zapisane” | wektor z LDA to ten pierwszy; do zaburzania potrzebny drugi |
| Yeh19 i Vie23 (MCD): miary kompletności zestawu pojęć | czy zestaw kierunków wystarcza do predykcji | formalna miara „domknięcia” |
| Wu23c: Wu i in. 2023, COWAL (chirurgia) | grupuje podobne klatki, z każdej niepokrytej grupy bierze najbardziej niepewną | zysk wobec losowego mały, około 0,01 |

Najważniejsze ostrzeżenie pochodzi od Jaina: kierunek błędów liczony we własnym latencie uczonego modelu nie działał, bo model zapamiętał trudne klatki uczące. U nas wektor przewidywał chybienia na nowych operacjach z AUC 0,76, więc sygnał jest, ale do wybierania klatek bezpieczniej liczyć go w zamrożonym backbone sprzed dostrajania. Plan w trzech krokach: wektor tożsamości pokazuje, gdzie jest narzędzie; resztę latentu wokół niego (kąt, długość, tło) dzieli się na grupy; grupy ze słabym wektorem i dużym błędem detektora to luki, a FeatInv pokazuje na obrazie, czego w nich brakuje. Pełne głębokie wyszukiwanie w Undermind objęło 287 prac (projekt „kuracja danych przez kierunek tożsamości obiektu w latencie i dekodowanie brakujących wariacji” w przestrzeni roboczej grantu). Pytanie asystenta o zapisanie linków do kolejki PdfDownloadera zostało bez odpowiedzi.

### 2.18 Zaskoczenie Koopmana wobec chybień detektora: wynik negatywny

**Skąd.** Propozycja „wybieracza” z 15:29 zawierała zdanie: „Najpierw bez żadnego uczenia sprawdzasz na zbiorze testowym, czy zaskoczenie Koopmana przewiduje chybienia lepiej niż sam wektor”. O 15:32 użytkownik zapytał najpierw „co to jest AUC?” (odpowiedź: liczba od 0,5 do 1; AUC 0,76 znaczy, że gdy weźmie się losowo jedną klatkę chybioną i jedną trafioną, wektor tożsamości jest słabszy na chybionej w 76 przypadkach na 100), a potem, co znaczy „bez uczenia sieci”. Odpowiedź: detektora nie uczy się od nowa; sprawdza się wybieracz na już wyuczonej sieci, która przeszła przez klatki testowe, więc wiadomo, gdzie trafiła, a gdzie chybiła. Zaskoczenie Koopmana to błąd przewidywania latentu następnej klatki z latentu poprzedniej jedną macierzą liniową, dopasowaną metodą najmniejszych kwadratów w kilka sekund, więc nie jest to uczenie sieci; jest duże, gdy następna klatka wygląda inaczej, niż wynikało z dotychczasowego ruchu. Kryterium ustalone przed wynikiem: jeśli zaskoczenie nie przekroczy AUC 0,76 sięgniętego przez wektor, trening z takim wybieraczem nie ma sensu. O 15:33 użytkownik napisał: „to sprawdź to”.

**Metoda** (`...\krystalizacja_latentu\koopman_zaskoczenie.py`). Ekstrakcja (lokalnie, środowisko rfdetr, GPU): dla każdej klatki testowej stan y_t z tokenów backbone detektora `ut_b0500`, 64 liczby: średnia profilu zredukowanego po całym kadrze (32) i średnia ważona aktywacją wektora narzędzia (32), czyli „jak wygląda miejsce, które sieć uznaje za narzędzie”, bez etykiet (plik `...\wyniki\test\LATENT_test_ut_b0500.npz`). Koopman: y_{t+1} = K y_t + c, najmniejsze kwadraty z małym grzbietem (regularyzacją), dopasowane na filmach z jednej połowy i liczone na drugiej, i odwrotnie. Langevin: błąd mierzony w jednostkach szumu, czyli odległość Mahalanobisa eᵀΣ⁻¹e (błąd przeskalowany odwrotnością kowariancji Σ zwykłych reszt z połowy dopasowania, duży tylko wtedy, gdy wychodzi poza zwykły szum ruchu), oraz wersja względem filmu (podzielona przez medianę danego filmu). Porównanie (AUC: czy miara jest większa na klatkach chybionych niż trafionych, tylko klatki z narzędziem): słaby wektor pod obrysem narzędzia (z etykiet), słaba obecność z mapy wektora (bez etykiet), zwykła zmiana klatki ‖y_t − y_{t−1}‖ (kontrola, czy Koopman daje coś ponad różnicę), trzy wersje Koopmana, niska pewność detektora (częściowo kołowa, bo chybienie obejmuje niską pewność) i średnia rang wektora z Koopmanem-Langevinem. Osobno na „pewnych chybieniach” (pewność ≥ 0,5 przy IoU < 0,5, zwykle inne narzędzie), gdzie pewność detektora z definicji mało mówi. Ten sam stan y_t (z latentu `ut_b0500`) oceniano względem chybień dwóch detektorów.

**Wyniki** (`...\krystalizacja_latentu\wyniki\test\REPORT_koopman_zaskoczenie.json`; commit `f758fc35`, 15:36). Klatek z narzędziem 7889; chybień 1013 dla `ut_b0500` i 682 dla `b1500_best`; pewnych chybień 406 i 346. Korelacja rangowa wektora z zaskoczeniem Koopmana: −0,073.

| miara (AUC) | ut_b0500: wszystkie chybienia | ut_b0500: pewne chybienia | b1500_best: wszystkie | b1500_best: pewne |
|---|---|---|---|---|
| słaby wektor pod obrysem (z etykiet) | 0,761 | 0,716 | 0,774 | 0,714 |
| słaba obecność z mapy (bez etykiet) | 0,601 | 0,526 | 0,598 | 0,536 |
| zwykła zmiana klatki | 0,602 | 0,607 | 0,603 | 0,591 |
| zaskoczenie Koopmana | 0,585 | 0,581 | 0,592 | 0,560 |
| Koopman + Langevin (Mahalanobis) | 0,574 | 0,575 | 0,578 | 0,571 |
| Koopman + Langevin względem filmu | 0,550 | 0,557 | 0,561 | 0,559 |
| niska pewność detektora | 0,882 | 0,705 | 0,827 | 0,658 |
| wektor + Koopman-Langevin (średnia rang) | 0,714 | 0,696 | 0,731 | 0,693 |

**Interpretacja (15:36).** Zaskoczenie Koopmana przewiduje chybienia gorzej niż sam wektor (0,58 wobec 0,76), blisko rzutu monetą; poprawka Langevina niczego nie poprawia (0,57); Koopman przegrywa nawet z najprostszą zmianą klatki (0,60), a dołożenie go do wektora obniża wynik z 0,76 do 0,71. Według rozmowy Koopman przewiduje ruch latentu całkiem dobrze, około 62 procent zmienności (tej liczby nie ma w raporcie), ale jego błędy wypadają tam, gdzie po prostu dzieje się coś szybkiego, a nie tam, gdzie detektor się myli. Asystent poprawił przy tym wcześniejszą sugestię: wektor z AUC 0,76 liczono pod obrysem z adnotacji, którego na nieoznaczonych klatkach nie ma; wersja bez etykiet daje tylko 0,60. Najlepszym sygnałem bez etykiet zostaje więc niepewność samego detektora (0,88), ale i ona słabo działa na najgroźniejszych błędach, gdy detektor pewnie wskazuje inne narzędzie (0,66-0,71). Kryterium ustalone przed wynikiem nie zostało spełnione, więc treningu z tym wybieraczem nie uruchomiono. Wniosek: w tej postaci (Koopman na latencie całego kadru) Koopman nie nadaje się do wybierania klatek; większą szansę ma dokładanie klatek z innymi narzędziami jako przykładów, że to nie jest cystotom. Pomiar zrobiono na zbiorze testowym, co stało się podstawą decyzji, żeby dalszy rozwój prowadzić na walidacji.

### 2.19 Do czego służy Koopman, włókno Rousselota i trzy plany eksperymentów

**Pytanie (15:39).** Użytkownik: „to do czego sie uzywa tego koopmanna i jego niezmienników w sieciach neuronowych????”. Odpowiedź (15:40): Koopman w sieciach służy głównie do opisu i przewidywania ruchu w czasie, a nie do oceny, co sieć już umie. Cztery typowe zastosowania: prognozowanie układów fizycznych (przepływ, pogoda, robot), gdzie sieć uczy się współrzędnych, w których ruch jest liniowy (Lusch, Kutz, Brunton 2018); analiza samego uczenia sieci, gdzie wagi zmieniają się krok po kroku, a Koopman przewiduje, dokąd zmierzają, co pozwala przyspieszyć trening albo porównać metody uczenia (Dogra, Redman 2020); zaglądanie do pamięci sieci sekwencyjnych (jak wcześniej w projekcie grantowym z TFT); niezmienniki, czyli mody o wartości własnej 1, odpowiadające zasadom zachowania, a w wideo na przykład tożsamości obiektu. Dobór klatek nie zadziałał, bo Koopman opisał, jak porusza się narzędzie i zmienia scena, a nie, czego detektor jeszcze nie umie. Naturalne miejsce Koopmana w projekcie to śledzenie: model ruchu dla filtra cząsteczkowego, a jego niezmiennik jako sprawdzian, że w każdej klatce śledzi się to samo narzędzie.

**Prośba o wersję w latencie i włókno (15:46-15:48).** Użytkownik: „no to ja chce własnie coś takiego zrobic tylko w latencie. Mam tożsamość obiektu wiec teraz chce wykorystac i zrobic taki detektor cech obrazów i który pomoze mi lepiej uczyc siec np od zera teraz wiem ze na 6 epoce konczy sie trening. Jak wybrac klatki z filmu które beda przyspieszały trening i go ulepszały. […] Jak to zrobic teraz żeby np wygenerowac z latentu albo braku jego aktywacji wektory błedów i na ich podsawie wygenerowac obrazy albo cechy obrazów które pozwolą mi z prawdziwej puli wybrać te które mocno wzmocnią siec w rozpoznawaniu narzedzia?”, z prośbą o przeczytanie całej pracy Rousselota („Show Me What You Don't Know”). Asystent przeczytał ją w całości. Rdzeniem jest włókno: zbiór wszystkich obrazów, które sieć widzi dokładnie tak samo, czyli z tym samym latentem. Autorzy biorą gotowy model dyfuzyjny (generator, który tworzy obraz, usuwając szum krok po kroku) i przy każdym kroku odszumiania lekko poprawiają obraz tak, żeby jego latent zbliżał się do latentu wzorca; miarą jest strata włókna, kwadrat odległości latentów; metoda nazywa się NDTM i nie wymaga uczenia dekodera, ale wymaga modelu dyfuzyjnego z danej dziedziny (dla zdjęć rentgenowskich wyuczyli model 18 milionów parametrów przez 120 tysięcy kroków; próbka kosztuje około dwóch minut). Jeśli na włóknie pojawia się coś innego niż we wzorcu, sieć jest na to ślepa (przykład z pracy: model nie widzi serca po prawej stronie). U nas wzorcem byłby latent tożsamości, a włókno pokazałoby obrazy, które detektor uznaje za ten sam cystotom; z galerii chybień wiadomo już, co tam trafi: pęseta, inne cienkie narzędzia, rękojeść. Plan od najtańszego: (1) włókno w prawdziwej puli, bez generatora; (2) pełna metoda Rousselota z modelem dyfuzyjnym uczonym na ponad 150 tysiącach klatek Cataract-LMM (kilkudniowe zadanie na Edenie, wymaga zgody); (3) Koopman na przebiegu uczenia, jak u Dogry i Redmana: dla każdej klatki sprawdzić, czy jej aktywacja tożsamości już się ustaliła, czy dalej się waha; ten sam pomysł bez Koopmana to „mapa zbioru danych” (dataset cartography, Swayamdipta i in. 2020), gdzie uczenie na przykładach, które się wahają, poprawia działanie na nowych danych.

**Zapis planów (15:57-16:00).** Użytkownik: „zapisz plany na nastepujace testy: [wklejony opis wybieracza] jako pierwszy; drugi [analiza samego uczenia sieci, Dogra i Redman]; [wklejony opis włókna w prawdziwej puli] […] 3 to bedzie z tym modelem dyfuzyjnym ale na razie bedziemy go ulepszac! zapisz to jako plany ekpserymentów”. Powstały dokument `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\46_PLAN_TRZECH_EKSPERYMENTOW_DOBORU_KLATEK.md` i trzy pliki `PLAN.md` (commit `a8385d7f`, 16:00). Nic z nich nie uruchomiono w etapie 2; każdy trening wymaga osobnej zgody.

Wspólne zasady z dokumentu 46: równe budżety (250 klatek), ta sama pula i trzy ziarna (42, 1337, 2024); odniesienie: losowe 250 klatek, mAP 0,6319; sukces to przewaga większa niż rozrzut między ziarnami; najpierw test offline bez uczenia sieci, trening tylko po jego przejściu i po zgodzie; test zostaje do oceny końcowej.

1. **Eksperyment 1, wybieracz** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\wybieracz_tozsamosc_koopman\PLAN.md`). Pytanie: czy mały wybieracz działający tylko na latencie wstępnie wyuczonego detektora wybiera klatki, które wnoszą coś nowego, i daje lepszy detektor przy tym samym budżecie niż losowanie. Stan wyjściowy to tabela z sekcji 2.18 (dla `ut_b0500`). Zmiany: wektor bez etykiet (kontrast pod ramką przewidzianą i pod maską z progowania mapy); Koopman w układzie narzędzia (tokeny wokół przewidzianej ramki wyrażone względem czubka i osi, z odjętym obrotem i skalą); Langevin z pamięcią (opóźnienia 1-3 kroków, wybielanie kowariancją szumu z innych filmów), porównany z wersją bez pamięci; różnorodność przez grupowanie k-średnich (podział punktów na k grup wokół ich średnich) latentów i jedną klatkę z grupy. Etap A offline na 25 operacjach walidacyjnych (co 10. klatka, dopasowanie na połowie operacji, ocena na drugiej), miara AUC chybień i pewnych chybień `ut_b0500` i `b1500_best`; warunek przejścia: łączny wynik bez etykiet przewyższa niską pewność detektora na pewnych chybieniach (0,66-0,71) albo, dodany do pewności, podnosi AUC o co najmniej 0,02 w obu połowach. Etap B na Edenie: ramiona wybieracz 250, losowe 250 i pewność detektora 250 (klasyczne wybieranie niepewnych klatek), trzy ziarna, trening jak `maly_budzet`, sonda włączona, mAP walidacji w najlepszej epoce i test raz na końcu. Czego nie dowodzi: wysokie AUC offline nie oznacza zysku z uczenia; jeśli etap B zawiedzie, powtórzyć z zamrożonym backbone (ostrzeżenie Jaina). Przed etapem A porównać zakres z `tozsamosc_czastki_bledy` (sekcja 2.23).
2. **Eksperyment 2, Koopman na przebiegu uczenia** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_dynamika_uczenia\PLAN.md`). Pytanie: czy z tego, jak latent każdej klatki uczącej zmienia się w pierwszych krokach uczenia, da się przewidzieć, które klatki sieć opanowała, a które dalej ją uczą, i czy wybór tych drugich przyspiesza trening. Uzasadnienie liczbami z etapu 1: strata uczenia spada dalej (2,6 do 1,1), strata walidacji rośnie (3,2 do 3,7), mAP spada (0,685 do 0,650), a na teście ostatnia epoka daje 0,654 wobec 0,681 najlepszej. Potrzebna nowa sonda na klatkach uczących: po każdym kroku dla każdej z 250 klatek kontrast wektora pod ramką etykiety, pewność i IoU, latent zapytania i strata klatki; bieg 250 losowych klatek, 10 epok, trzy ziarna, lokalnie albo na Edenie. Metoda: trajektorie latentu do końca 2. epoki; wspólny Koopman z(k + 1) = K z(k) z grzbietem; udział trajektorii klatki w wolnych modach i reszta; Langevin rozdzielający resztę na pamięć i szum paczek; przewidywanie latentu po 6. epoce z pierwszych 2 epok (R² i zgodność przewidzianej pewności). Etap A: AUC przewidywania klatek z dużą stratą w epoce 6, porównanie z mapą zbioru danych i z liczbą zapomnień (Toneva i in. 2019; ile razy przykład poprawnie rozpoznawany staje się w trakcie uczenia błędnie rozpoznawany), próg R² > 0,8. Etap B: ramiona Koopman-Langevin, mapa zbioru danych, losowe; miary: mAP najlepszej epoki i epoka, w której osiągnięto 95 procent końcowego mAP (przyspieszenie). Czego nie dowodzi: przeniesienie miary z 250 klatek na pulę to dodatkowe założenie; wersja w latencie to rozszerzenie bez gotowego wzorca w literaturze.
3. **Eksperyment 3, włókno pomyłek** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\wlokno_pomylek\PLAN.md`). Podstawa: Rousselot, Wendebourg, Köthe 2026, arXiv 2603.21782. Włókno φ⁻¹(h) = {x : φ(x) = h}; strata włókna L = ‖φ(x) − φ(x̃)‖², porównywana ze stratą najbliższego sąsiada w danych. NDTM: w każdym kroku odszumiania poprawka x̂_t = x_t + γ_t u_t, gdzie u_t minimalizuje L_terminal + κ‖u_t‖² + τ L_score w kilku krokach gradientu; stan końcowy szacowany wzorem Tweediego E[x₀ | x_t] = x_t + σ_t² ∇ log p(x_t) (oszacowanie czystego obrazu z zaszumionego przez gradient logarytmu gęstości); poprawka stabilności: wzorzec też jest zaszumiany do tego samego poziomu. Ustawienia z pracy: κ = 10⁻⁴, τ = 0, 4-8 kroków na poprawkę, krok uczenia od 2·10⁻³ liniowo do 0, najsilniejsze prowadzenie dla t około 700-400; koszt około 1,8 minuty na próbkę (karta A40, 100 kroków); warunkowy model włókna to około 14 000 minut uczenia i opłaca się od około 9 tysięcy próbek; poprawki lepiej robić w latencie autokodera niż w pikselach, bo w pikselach wychodzą zmiany podobne do ataków adwersarialnych; zastosowania w pracy są wyłącznie diagnostyczne. Etap A (bez generatora, lokalnie): latent tożsamości klatki (tokeny pod przewidzianą ramką ważone aktywacją wektora oraz wersja przestrzenna w układzie narzędzia); wzorce pomyłek (klatki walidacyjne z pewną złą ramką, pewność ≥ 0,5 przy IoU < 0,5, oraz fałszywe alarmy); włókno w puli `CZASTKI_pool_b0500` (klatki o stracie włókna poniżej typowej straty między sąsiednimi trafionymi klatkami filmu); kontrola z etykietami, czy włókno naprawdę zbiera pomyłki; kierunki błędów (różnica „włókno pomyłki” minus „prawdziwy cystotom”, jeden na skupisko: pęseta, rękojeść, niebieskie tło); ocena puli rzutem na te kierunki z różnorodnością; galeria. Trening po etapie A: 250 klatek z kierunków błędów wobec 250 losowych, wariant z pęsetą bez cystotomu jako przykładem negatywnym. Etap B (odłożony zgodnie z poleceniem użytkownika): bezwarunkowy model dyfuzyjny na klatkach Cataract-LMM (około 18 milionów parametrów, rozdzielczość 384, kilka dni na Edenie), NDTM ze stratą włókna na latencie tożsamości; wygenerowane obrazy najpierw służą do nazwania brakujących czynników i jako zapytania do wyszukiwania prawdziwych klatek, a nie wprost do uczenia.

Uwaga o numeracji: eksperyment z sekcji 2.20 nazywa sam siebie w swoim `PLAN.md` „Eksperymentem 4”, ale nie ma go w dokumencie 46.

### 2.20 Klatki ze zmianami w stałych kierunkach Koopmana: 1500 wobec 1500 losowych

**Pomysł użytkownika.** W trakcie zapisywania planów, o 15:59, użytkownik dopisał (wiadomość kolejkowana): „no własnie mozna zrobic tak że sprawdzasz koopmanem kierunki stałe i sprawdzasz w latencie czy w nich jest tozsamosc narzedzia. Sprawdzaj klatki w których wiesz ze jest cystom ale te kierunki w których koopman zazwyczaj jest stały ma mocno zmiany to wtedy te klatki zalisczaj do uczenia. Wyciagnij takie klatki i spawdz czy one poprawiają trening. Szukaj kolejnych 1500 klatek”. Hipoteza: klatki, w których zwykle stałe kierunki latentu nagle się zmieniają, niosą nową informację o narzędziu i poprawią uczenie.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_kierunki_stale\wybor.py`, commit `9fb430ac`, 16:04). Stan klatki: latent `ut_b0500` (profil 32 liczb) w miejscu, gdzie z adnotacji wiadomo, że jest cystotom: opis 3 × 3 łatek wewnątrz ramki etykiety, wybielony do 48 wymiarów; pula: klatki uczące tracking_static36 z ramką cystotomu co najmniej 8 × 8 pikseli (`PLAN.md` mówi o puli 10 637 klatek, a kod czyta nagranie `CZASTKI_pool_b0500` z 9683 klatkami). Koopman (EDMD) na parach klatek tego samego filmu odległych o najwyżej 30 klatek: 6761 par. Kierunki stałe: mody o |λ| ≥ 0,85, wyszło 6 z 48 (|λ| = 0,965; 0,951; 0,925; 0,907; 0,893; 0,853). Czy tożsamość siedzi w kierunkach stałych: R² przewidywania kontrastu wektora tożsamości (pod ramką etykiety) ze współrzędnych Koopmana, połowa klatek na dopasowanie, połowa na ocenę, wobec tylu samo najszybszych kierunków, 20 losowań kierunków losowych i wszystkich 48. Wynik klatki: błąd jednokrokowego przewidywania Koopmana tylko w kierunkach stałych, każdy kierunek w jednostkach swojego typowego szumu (mediana wartości bezwzględnej reszty; to jest tu „poprawka Langevina”, czyli reszta mierzona względem zwykłego szumu), zsumowane kwadraty, większa z wartości dla pary z poprzednią i z następną klatką. Wybór: od największego wyniku, z wykluczeniem klatek bliższych niż 10 klatek od już wybranej w tym samym filmie, budżet 1500; kontrola: 1500 losowych klatek z tej samej puli (ziarno 42), bo pula jest sumą klatek wybranych wcześniej przez inne strategie. Sprawdzenie offline bez uczenia: czy wynik jest większy na klatkach puli, na których `ut_b0500` chybia (pewność < 0,5 albo IoU < 0,5), z pominięciem jego własnych 500 klatek uczących.

**Wyniki offline** (`...\koopman_kierunki_stale\wyniki\REPORT_wybor.json`, `WYNIK_klatek.npz`, listy `tracking_koopman_stale_b1500_sel42.json` i `tracking_losowe_z_puli_b1500_sel42.json`):

| kierunki | R² przewidywania kontrastu wektora tożsamości |
|---|---|
| 6 stałych | 0,247 |
| 6 najszybszych | 0,005 |
| 6 losowych (mediana z 20) | 0,121 |
| wszystkie 48 | 0,587 |

| miara wskazująca chybienia `ut_b0500` | AUC |
|---|---|
| zmiana w kierunkach stałych | 0,643 |
| zmiana w kierunkach szybkich | 0,521 |
| zwykła zmiana latentu | 0,645 |
| słaby wektor tożsamości | 0,784 |
| niska pewność detektora | 0,890 |

Według raportu oceniano 8600 klatek, w tym 880 chybień; `PLAN.md` podaje 3128 klatek i 344 chybienia, prawdopodobnie z wcześniejszego przebiegu przy odstępie par 6 [rozbieżność nie jest wyjaśniona w źródłach]. Skład zestawów: wybrane 1500 klatek zawiera 18,4 procent chybień `ut_b0500` wśród klatek przez niego niewidzianych, losowe 10,6 procent; oba zestawy obejmują 118 filmów; mediana kontrastu wektora 2,84 wobec 3,07; wspólnych klatek ze starym zestawem `random_b1500_s42`: 205 wobec 221. Czułość na odstęp par: przy 6 klatkach tylko 1809 par i 1484 wybrane klatki, przy 60 wynik jak przy 30 (AUC 0,65, chybień w wyborze 20 procent). Tożsamość jest więc wyraźnie bardziej w kierunkach stałych niż szybkich, ale nie cała, a kierunki stałe nie wskazują chybień lepiej niż zwykła zmiana latentu (asystent o 16:05: „obie mają AUC 0,64”).

**Gdzie liczyć (16:12-16:27).** Użytkownik: „a na mojej rtx nie pójdzie ? nie puszczaj jeszcze tylko sprawdz”. Asystent przygotował lokalnie dane (`...\koopman_kierunki_stale\przygotuj_dane_lokalne.py`: 14 834 klatki, czyli klatki obu wariantów i cały zbiór walidacyjny, zgodne z oryginałem co do skrótu pikseli), ale próbny trening zatrzymał się na braku pakietów do uczenia RF-DETR; ich instalacja podniosłaby wersje numpy i opencv w środowisku, w którym liczono wszystkie dotychczasowe pomiary, więc bez zgody jej nie zrobiono. Na RTX 3070 jeden bieg trwałby 2-3 godziny (po każdej epoce ocena 12 tysięcy klatek walidacyjnych), sześć biegów około 15 godzin. Na Edenie zadania szły dotąd do partycji `short`; siedem wolnych kart na dgx-1 było niedostępnych, bo węzeł był w konserwacji, a pozostałe węzły dgx były zajęte; siedem wolnych kart H100 było na węźle `hopper` z osobną partycją otwartą dla grupy. Użytkownik wkleił o 16:18 wynik polecenia `sfree` (hopper 7 z 8 kart wolnych, dgx-1 7 z 8, dgx-2 do dgx-4 po 0) z komentarzem „jest wolnych duzo kart na edenie”, o 16:23 napisał „licz na hooper”, a chwilę później „trening loklany uwal”. Warianty wysłano do `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-21_koopman_langevin/koopman_stale/variants/`, a skrypt `wyslij_ks.sh` wysłał zadania `sbatch -p hopper` z tym samym `eden/run_maly_budzet.slurm` i protokołem `protocol_maly_budzet.yaml` co ramiona 250 (sufit 60 epok, cierpliwość 15, min_delta 0,001; zmienne wyeksportowane w powłoce przed `sbatch`). Zadania: 1809873-1809875 (kierunki stałe, ziarna 42, 1337, 2024) i 1809876-1809878 (losowe z puli), wszystkie uruchomione o 16:27. O 16:28 użytkownik polecił: „po treningu masz wyczyscic to co jest na hooperze”. O 16:38, na pytanie o czas, asystent podał, że po 11 minutach każdy bieg ma za sobą około trzy epoki i skończy się w około godzinę.

**Wyniki treningu** (`...\koopman_kierunki_stale\wyniki\biegi\ks_*\` z plikami `validation_report.json`, `run_manifest.json`, `effective_train_config.json`, `TELEMETRIA.json`; opis w `...\koopman_kierunki_stale\PLAN.md`; commit `0ed75a7b`, 17:30). Węzeł hopper, NVIDIA H100 PCIe, torch 2.9.1+cu128, Python 3.11.15, rfdetr 1.10.1, rozdzielczość 384, paczka 16, bf16; czas zadań 52 minuty 38 sekund do 57 minut 38 sekund; walidacja 25 operacji, 12 043 klatki, wagi najlepszej epoki oceniane niezależnie. Kolejność ziaren: 42, 1337, 2024.

| ramię (1500 klatek) | mAP 0,5:0,95 najlepszej epoki | średnia ± odch. | mAP ostatniej epoki | średnia ostatniej | epoka szczytu | epok do zatrzymania | mAP 0,5 najlepszej | precyzja | czułość |
|---|---|---|---|---|---|---|---|---|---|
| kierunki stałe Koopmana | 0,6708 / 0,6704 / 0,6675 | 0,6696 ± 0,0015 | 0,6366 / 0,6205 / 0,6163 | 0,6245 | 5 / 8 / 7 | 20 / 21 / 21 | 0,897 / 0,898 / 0,892 | 0,931 / 0,924 / 0,925 | 0,874 / 0,890 / 0,894 |
| losowe z tej samej puli | 0,6810 / 0,6822 / 0,6829 | 0,6820 ± 0,0008 | 0,6528 / 0,6627 / 0,6644 | 0,6600 | 8 / 6 / 7 | 22 / 21 / 22 | 0,910 / 0,905 / 0,906 | 0,946 / 0,954 / 0,943 | 0,862 / 0,795 / 0,858 |

Wartości najlepszego mAP zapisane w trakcie treningu (`best_map_50_95_val` w manifestach) są nieco wyższe niż z niezależnej oceny (0,6699-0,6857); tabela podaje ocenę niezależną.

**Werdykt i interpretacja (17:30).** Wybór według stałych kierunków Koopmana przegrywa z losowym o 0,012 na każdym z trzech ziaren, a różnica jest około dziesięć razy większa niż rozrzut między ziarnami; formalnego testu statystycznego nie liczono, kryterium ustalone przed wynikiem (przewaga nad losowymi z puli większa niż rozrzut ziaren na wszystkich trzech ziarnach) wypadło przeciw metodzie. Po szczycie model z wybranych klatek szybciej uczy się na pamięć (ostatnia epoka 0,625 wobec 0,660). Ma wyższą czułość (0,87-0,89 wobec 0,80-0,86), ale niższą precyzję (0,92-0,93 wobec 0,94-0,95), czyli wykrywa więcej narzędzi kosztem większej liczby fałszywych ramek. Losowe klatki z puli (0,682) wypadają tak samo jak dawne losowe z całego zbioru (static36 `random_b1500`, 0,682). Wniosek z `PLAN.md`: zestaw złożony z klatek trudnych (18 procent chybień wobec 11) uczy gorzej niż zestaw typowy, spójnie z pozostałymi próbami (K6 przy 250 klatkach, kampania GLISTER z sekcji 2.21); na tym zbiorze żadna metoda wybierania trudnych albo „brakujących” klatek nie pokonała losowania. Czego to nie dowodzi: wybór używa etykiet (ramka cystotomu wyznacza miejsce opisu), więc to kuracja oznaczonego zbioru, a nie wybór klatek do oznaczenia; trudne klatki mogą też być klatkami z zaszumionymi adnotacjami. Wagi zostały na Edenie w `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-21_koopman_langevin/results_maly_budzet/ks_*`.

**Sprzątanie węzła hopper.** Zgodnie z poleceniem użytkownika po zakończeniu biegów uruchomiono o 17:29 krótkie zadanie `srun` na hopper (1 rdzeń, 2 GB, 3 minuty), które przejrzało `/dev/shm`, `/tmp` i `/raid`. Katalogów roboczych biegów nie było (pułapka sprzątająca zadziałała), został tylko katalog `/tmp/pytest-of-bpiotrowski` (936 KB, z 22 września, po testach równoległej sesji), który usunięto; po nim na węźle nie było żadnych plików użytkownika. W drugiej, równoległej kontynuacji rozmowy zapisanej w pierwszym transkrypcie pojawiła się o 21:34 wiadomość, że „nie było czego usuwać”; jest ona niezgodna z zapisem wykonanych poleceń. Przy okazji potwierdzono, że zadanie `kl_epoki` (1809823) skończyło się (COMPLETED, 16 minut 54 sekundy, dgx-4).

### 2.21 Pytanie o kampanię równoległej sesji „dobór cech adaptacyjny”

**Skąd.** Podczas skanu dysku Edenu (sekcja 2.22) największym katalogiem okazał się `~/smallvit_runs/2026-09-22_dobor_cech_adaptacyjny` (87 GB). O 16:31 użytkownik zapytał: „2026-09-22_dobor_cech_adaptacyjny co tutaj jest?”, a o 16:34: „i jak ta siec sie wytrenowała? Lepiej czy nie ?”.

**Co to jest.** Kampania równoległej sesji z 22 września (katalog lokalny `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_cech_adaptacyjny`, niewersjonowany; zadania Eden 1809520 dla testów i 1809521-1809535 dla kampanii). Wszystkie 15 biegów skończyło się 22.09 wieczorem, ale nikt nie pobrał ani nie opisał wyników, a lokalny `WYNIKI.json` twierdził, że zadania czekają w kolejce. Pięć ramion po trzy ziarna (42, 1337, 2024), RF-DETR Nano, 10 epok, w każdej epoce 1500 klatek wybieranych na nowo: `random` (losowo), `glister` (GLISTER: metoda wybierająca klatki, których gradient najbardziej zmniejsza stratę na zbiorze sterującym), `features_glister` (najpierw potrzeby z cech obrazu, takich jak rozmiar, jasność, rozmycie, położenie ramki, potem GLISTER), `features_kg_glister` (jak wyżej z dodatkowym priorytetem z Koopmana-GLE) i `features_placebo_glister` (kontrola z pozornymi cechami). Miara: mAP 0,5:0,95 na walidacji. Każdy bieg zajmował około 5,5 GB, razem 87 GB; według rozmowy 67 GB stanowiły pliki `.ckpt` (pełne stany treningu z optymalizatorem, potrzebne tylko do wznowienia), a 23 GB wagi `.pth` (suma tych dwóch liczb przekracza 87 GB; źródła nie wyjaśniają różnicy). Usunięcie o 16:45 objęło 539 plików i 90 GB (sekcja 2.22).

**Wyniki** (z `...\dobor_cech_adaptacyjny\WYNIKI_KAMPANII_2026-09-22.md`, zapisanego 23.09 o 16:41 na polecenie użytkownika):

| ramię | najlepsza epoka (średnia ± odch.) | ostatnia epoka | różnych klatek w uczeniu |
|---|---|---|---|
| random | 0,7131 ± 0,0042 | 0,7013 | około 7400-7500 |
| glister | 0,7072 ± 0,0017 | 0,7002 | około 7430-7490 |
| features_placebo_glister | 0,6941 ± 0,0033 | 0,6738 | około 6000-6500 |
| features_glister | 0,6903 ± 0,0083 | 0,6695 | około 5900-6300 |
| features_kg_glister | 0,6839 ± 0,0095 | 0,6807 | około 5900-6300 |

Odpowiedź dla użytkownika (16:38): nie, sieć nie wytrenowała się lepiej; najlepsze jest losowanie; dobór według cech nie tylko nie pomógł, ale lekko szkodził (0,02-0,03 poniżej losowego), prawdziwe cechy nie są lepsze od pozornych (0,690 wobec 0,694), a Koopman-GLE nie pomaga (0,684 wobec 0,690, w granicach rozrzutu). Prawdopodobna przyczyna: ramiona z cechami wracają do podobnych klatek i przez 10 epok widzą około 6 tysięcy różnych klatek wobec około 7,5 tysiąca w losowym; znowu wygrywa różnorodność. Plik zastrzega, że wartości nie są wprost porównywalne z biegami static36 (0,682), bo uczono adaptacyjnie po 1500 klatek, i że cechy nie obejmowały kąta ani zasłonięcia. Plik zawiera też pełną tabelę mAP po każdej z 10 epok dla 15 biegów.

**Decyzje użytkownika.** O 16:39: „to zapisz ze takie wyniki i zapisz ostatnią epoche tamtego treninigu”. Zapisano `WYNIKI_KAMPANII_2026-09-22.md` i zaktualizowano `WYNIKI.json` (status „campaign_completed_results_recorded”), bez commitu, bo katalog należy do równoległej sesji; pobrano metryki, listy klatek i wagi ostatniej, 10. epoki każdego biegu (około 2,4 GB) do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\eden_dobor_cech_adaptacyjny_2026-09-22\<ramię>_s<ziarno>\rounds\epoch_010\weights.pth`. O 16:40 (wiadomość kolejkowana): „potem usuwaj wszsytkie te checkpointy z edena”; usunięcie opisano w sekcji 2.22.

### 2.22 Porządki na Edenie i dyscyplina korzystania z klastra

Tego dnia dysk projektu na Edenie przekroczył miękki limit, więc część czasu poszła na jego porządkowanie. Zasady obowiązujące przy tym (z instrukcji użytkownika i pamięci projektu): niczego nie usuwać bez wyraźnej zgody na konkretny katalog, nie ruszać pracy równoległych sesji, nie zostawiać plików na węzłach.

1. **12:35-12:39.** Asystent zgłosił, że projekt zajmuje 214,5 GB przy limicie 200 i twardej granicy 220, więc kolejne treningi mogą przestać się zapisywać. Użytkownik: „przeskanuj co zajmuje tyle miejsca”. Wynik: `2026-09-22_dobor_cech_adaptacyjny` 86 GB (równoległa sesja zapisywała pełny stan treningu po każdej epoce, 346 MB na epokę); `2026-09-21_koopman_langevin` 31 GB (z tego 13 GB wyniki etapu 1); `grant_tft` 32 GB; `tracking_static36` 13 GB; `dobor_klatek_stan_sieci` 12 GB; środowiska Pythona 9 GB; `.cache` 11 GB (w tym 4 GB pamięci instalatora pip); stare archiwum `20kSelectedImages.tar` 8,5 GB; `.vscode-server` 4 GB.
2. **12:47.** „Dane tracking_static36 zajmują 13 GB co to jest?”: to kampania z 15 września (36 biegów, wyniki 8,2 GB, głównie zapisane przewidywania, wagi już usunięte), dane 4 GB (archiwum z 22 tysiącami klatek i jego rozpakowana kopia 2 GB) oraz 200 MB pamięci modelu DINOv2; archiwum jest potrzebne wszystkim treningom.
3. **12:52-12:58.** „to teraz zidentyfikuj foldery z których nie korzysta nic ?”: około 27 GB nieużywanych (wyniki kampanii z 15.09 z pełną kopią lokalną, 8,2 GB; `20kSelectedImages.tar`, 8,5 GB, nieruszane od grudnia 2025; pamięć pip, 4 GB; katalog `prom_ai`, 7 GB, nieruszany od 12 sierpnia), dwa przypadki niepewne (`2026-09-14_tracking_nano`, 3,7 GB; pamięć modeli Hugging Face, 6,5 GB), a do równoległej sesji należy około 100 GB. Użytkownik: „Czwarty to katalog prom_ai […] usun ten”; `prom_ai` (stare eksperymenty prognoz z lipca i sierpnia) usunięto. Pamięć Hugging Face (serwisu, z którego biblioteki pobierają gotowe sieci) sprawdzono na polecenie „sprawdzaj”: prawie całe 6 GB to katalog `xet` z resztkami przerwanego pobierania dwóch dużych modeli językowych Qwen, a poza nim 400 MB (DETR z ResNet-50 i sam ResNet-50); żaden skrypt projektu z niego nie korzysta; do usunięcia potrzebna zgoda, której w etapie 2 nie było. Zajętość spadła do 187 GB.
4. **16:24-16:30.** Użytkownik (wiadomość kolejkowana): „zrób sprawdzenie co zajmuje tyle miejsca i sprawdz co nie jest w ogoóle używane!”. Wynik: 187 GB z 200, w tym `smallvit_runs` 138 GB, `grant_tft` 29 GB, środowiska 9 GB, VS Code 4 GB; katalogi równoległej sesji 87 GB i 12 GB; własny katalog `2026-09-21_koopman_langevin` 31 GB, prawie cały w użyciu; nieużywane około 10 GB (wagi pierwszej rundy doboru w `results_stage1` 6,5 GB, `2026-09-14_tracking_nano` 3,7 GB, próby rozruchowe poniżej 100 MB).
5. **16:41-16:45.** Po pobraniu wag ostatniej epoki kampanii doboru cech i porównaniu sum kontrolnych (wszystkie 15 plików zgodne co do bajtu z oryginałami) usunięto z Edenu 539 plików punktów kontrolnych tej kampanii, razem 90 GB; zostały metryki, listy klatek i opisy biegów (1,2 GB). Zajętość: 102 GB z 200.
6. **16:43-16:46.** „co jeszcze można usunąć ?”: asystent wskazał około 19 GB bez straty (wagi `results_stage1` 6,5 GB; `2026-09-14_tracking_nano`; wagi dziewięciu biegów po 250 klatek, około 4 GB; `.vscode-server` 4,3 GB; próby rozruchowe) i zaznaczył, czego nie ruszać (środowisko `rfdetr_nano_clean_1_10_1` 9 GB, wagi z epok w `results_telemetria` potrzebne zadaniu `kl_epoki`, dane `tracking_static36`, `grant_tft`, latent cząstek `czastki` 4,1 GB, drugą kampanię równoległej sesji 12 GB). Użytkownik zaznaczył pierwsze trzy pozycje: „usuwaj to”. Usunięto 54 pliki wag z `results_stage1` (6,5 GB), 18 plików wag dziewięciu biegów po 250 klatek (około 2,2 GB) i wagi z `2026-09-14_tracking_nano` (0,7 GB, a nie 3,7 GB, jak podano wcześniej; zostało tam 3 GB, w tym stare archiwum danych kalibracyjnych 1,2 GB), razem około 9,4 GB. Archiwa przewidywań detektora (około 9 GB) zostawiono, bo to zapisane wyniki, a nie wagi.
7. **17:29.** Sprzątanie węzła hopper (sekcja 2.20).

### 2.23 Równoległa sesja: kod eksperymentu `tozsamosc_czastki_bledy` bez uruchomienia

O 15:50 inna sesja zapisała `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-23_155013_Tozsamosc_Czastki_Kod_Bez_Uruchomienia\` (`README.md`, `SESSION_SUMMARY.md`). Użytkownik najpierw poprosił tam o założenie i sprawdzenie eksperymentu, a potem zawęził zakres: „mozesz założyc taki eksperyment i napisac kod na razie niczego nie licz!”. Powstał katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy\` (`PLAN.md`, `README.md`, `WYNIKI.json`, `config.json`, `dane.py`, `modele.py`, `uruchom.py`, `test_aparatu.py`), niewersjonowany. Nie uruchomiono obliczeń, testów, instalacji, enkodera, treningu ani zadań na Edenie; nie wykonano nawet testów składni. `WYNIKI.json`: status „code_only_not_executed”, wyników naukowych brak.

Pytanie w planie: czy przestrzenny rozkład sygnału narzędzia oraz jego dynamika wskazują błędy zamrożonego detektora na innych operacjach lepiej niż pewność detektora i statyczny opis tej samej klatki. Cząstka to token w stałej komórce obrazu (bez zakładania, że śledzi ten sam fragment metalu); maska narzędzia pochodzi z `cystotome_seg`, a nie z prostokąta. Kierunek narzędzia dopasowuje się tylko na części operacji A; każda cząstka rozkłada się na rzut a_p = z_pᵀw i resztę r_p = z_p − a_p w. Podział po operacjach: A (kierunek, skalowanie, Koopman, model z pamięcią, nominalny Kalman), B (predyktor błędu), C (ocena). Drabina porównań ustalona przed wynikiem: U (ujemna pewność), G (pewność i geometria), I (plus sygnał narzędzia), R (plus pozostałe cechy i prosta historia), K (plus błąd prognozy EDMD w słowniku [s, s²] i współrzędne modów), L (plus pamięć), F (plus innowacja nominalnego Kalmana) i kontrola P (dynamika przesunięta cyklicznie wewnątrz filmu, co zrywa zgodność czasową z błędem). Cel: błąd = IoU < 0,5 lub pewność < 0,5, osobno sam błąd lokalizacji; miary AUC, average precision, wynik Briera (średni kwadrat różnicy między przewidzianym prawdopodobieństwem a wynikiem 0 lub 1), odsetek błędów w najwyżej ocenionej części filmu; różnice F − R i F − P z bootstrapem po operacjach. Ustawienia (`config.json`): ziarna 17, 43, 91; 4 foldy; historia 3; krok klatek 3; grzbiet 0,01; C regresji logistycznej 0,1; 4 mody; wariancja obserwacji 0,05; górne 20 procent; 2000 losowań bootstrapu; co najmniej 30 klatek na część; progi IoU i pewności 0,5. Granice zapisane w sesji: kod mierzy przewidywanie błędów, nie poprawę uczenia; EDMD ze stałym słownikiem to nie deep Koopman; przybliżenie pamięci nie identyfikuje pełnego równania Langevina; nominalny Kalman nie daje skalibrowanej niepewności. Plan wybieracza (sekcja 2.19) każe porównać zakres z tym kodem przed swoim etapem A.

### 2.24 Dzień 24 września

O 2:02 w nocy z 23 na 24 września użytkownik napisał „wyłącz za 2h”; wyłączenie komputera zaplanowano na 4:02 (lokalnie nic się już nie liczyło). Tego dnia nie ma żadnych commitów ani nowych pomiarów; rozmowa została tylko wznowiona technicznie o 13:10. Wyniki zadania `kl_epoki` (1809823: test obrotu i tła, z kontrolami `tlo_swoje` i `tlo_film`, na 45 wagach z kolejnych epok i na wagach sondy, katalog `results_obrot_epoki` na Edenie) czekały na analizę; w źródłach etapu 2 jej nie ma. Następny commit w repozytorium pochodzi z 25 września, 13:49.

### 2.25 Podsumowanie etapu 2

1. Dobór 250 klatek „po kwadratach” w przestrzeni tożsamości z Koopmana (K6) obstawił 99 procent komórek siatki (dobór losowy i równomierny potrzebowały do tego 2000 klatek), ale na walidacji zremisował z losowym: 0,6296 ± 0,0011 wobec 0,6319 ± 0,0023, daleko od 0,6866 dla 1500 klatek. Siatka była przez błąd dwuwymiarowa (duplikat z pary sprzężonej), a ramię równomierne obciążone klatkami brzegowymi (155 z 250).
2. Literatura (jedenaście prac w pełnym tekście) dała dokument 43 z kryterium domknięcia (wyciek kątami głównymi, jądro pamięci, związek fluktuacji i dyssypacji, stabilność D̂, nasycenie wymiaru) i 11-krokowy pipeline; jego centralna równoważność „brak pamięci wtedy i tylko wtedy, gdy podprzestrzeń niezmiennicza” została tego samego dnia wycofana przez dwa niezależne dokumenty 44 (kontrprzykład C = 0).
3. Dokument 44 „wzór domknięcia” wyprowadził dla współrzędnych tożsamości dyskretne uogólnione równanie Langevina z jądrem Ω_l = B D^{l−1} C, jądra z korelacji (Ω₁ = C_y(2Δ) − C_y(Δ)²), związek fluktuacji i dyssypacji, granice kątowe i nierówność pamięci; algebra zgadza się na układzie syntetycznym co do precyzji maszyny. Na puli pamięć z korelacji (0,108) nie przekracza podłogi szumu (0,142), jądro z szumu (0,080) ją przekracza (0,049), a oba się nie zgadzają, co wskazuje na brakującą zmienną, najpewniej stałe przesunięcie filmu.
4. Formalizacja Codex (dokumenty 44 i 45, 20 plus 12 kontroli SymPy/NumPy, 9 twierdzeń sprawdzonych w Lean 4.28.0 bez pominiętych kroków) pokazała, że gwarancję poprawy po partii klatek daje tylko przewaga przewidzianej poprawy straty nad ograniczeniem błędu tego przewidywania; samo widmo, pokrycie modów ani małe jądro pamięci jej nie dają (kontrprzykład T9). Założeń tych twierdzeń dla RF-DETR nie zmierzono.
5. W latencie żadnego z czterech detektorów nie ma kierunku niezmienniczego względem obrotu i tła (0 z 32); obrót zmienia latent bardziej niż przejście do innej klatki (R_g 1,3-1,6), a „kierunki tożsamości” Koopmana okazały się kątem ułożenia narzędzia (zmiana przy obrocie 1,31 wobec 1,58 dla kierunków losowych; 27 z 60 najsilniej pobudzających klatek z jednego filmu).
6. W tokenach backbone jest jeden kierunek „tu jest narzędzie”: z etykiet AUC 0,976 (pełna regresja 0,978), bez etykiet 0,975 (cosinus z wersją z etykiet 0,88), zapalony w 95 procentach klatek niezależnie od kąta (0,950-0,968) i wydłużenia. Po wycięciu narzędzia po obrysie wektor zapala się w 97-98 procentach klatek po obrocie o 30° i zmianie długości, a na obcym tle w 81 procentach; wcześniejszy spadek pewności do 0,26 był artefaktem prostokątnej naklejki.
7. Sonda po każdym kroku uczenia (337 kroków) pokazała, że latent przestaje reagować na wydłużenie i ruch po znalezieniu narzędzia, ale nie na obrót; przypisanie zmian pojedynczym klatkom uczącym nie wyszło ponad szum (0,031 wobec podłogi 0,038), a wniosek o rosnącej zależności od tła jest do powtórzenia z czystym wklejaniem.
8. Na nietkniętym zbiorze testowym (25 operacji, 8249 klatek) najlepsza epoka detektora z 1500 losowych klatek ma mAP 0,681 (jak walidacja 0,682) i trafia 91 procent klatek, ostatnia epoka 0,654 przy fałszywych alarmach 50 zamiast 40 procent, a 2500 klatek daje tylko 0,690; główne pomyłki to inne cienkie narzędzia (pęseta) i nietypowe tło.
9. Zaskoczenie Koopmana nie wskazuje chybień detektora (AUC 0,58; z Langevinem 0,57; zwykła zmiana klatki 0,60), a dołożone do wektora obniża wynik z 0,76 do 0,71; bez etykiet najlepszym sygnałem pozostaje pewność detektora (0,88; na pewnych chybieniach 0,66-0,71).
10. Wybór 1500 klatek ze zmianami w stałych kierunkach Koopmana (18 procent trudnych klatek wobec 11) przegrał z 1500 losowymi z tej samej puli na każdym z trzech ziaren: 0,6696 ± 0,0015 wobec 0,6820 ± 0,0008, z szybszym przeuczaniem; kampania GLISTER równoległej sesji dała to samo (losowe 0,7131 najlepsze). Na tym zbiorze nic nie pokonało losowania, a wspólnym wytłumaczeniem jest utrata różnorodności.
11. Zapisano trzy plany (wybieracz, Koopman na przebiegu uczenia, włókno pomyłek; dokument 46) z testami offline przed treningiem, a na Edenie zwolniono miejsce z 214,5 do 102 GB za zgodą użytkownika na konkretne katalogi.

### 2.26 Otwarte pytania na koniec etapu 2

1. Czy poprawnie zbudowana siatka tożsamości (część rzeczywista i urojona jednego modu jako osie, ramię równomierne bez klatek brzegowych) albo iloraz ergodyczny ze współrzędnymi dyfuzyjnymi zmieniłyby wynik 250 klatek, czy remis z losowaniem jest trwały?
2. Czy niezgodność dwóch jąder pamięci zniknie po odjęciu średniej każdego filmu, czyli czy brakującą zmienną jest stałe przesunięcie filmu? Jaka jest podłoga szumu dla wycieku w słowniku 96 i jak zachowuje się D̂ względem efektywnej liczby par (nachylenie −1/2 czy wypłaszczenie)?
3. Czy zależność detektora od tła rośnie w trakcie uczenia, gdy mierzyć ją czystym wklejaniem po obrysie? Co pokazują nieprzeanalizowane wyniki zadania `kl_epoki` z kontrolami `tlo_swoje` i `tlo_film`?
4. Czy wektor narzędzia jest modem Koopmana o wartości własnej 1, gdy Koopmana liczyć w układzie narzędzia (na łatkach zapalanych przez wektor), i czy tak rozumiana tożsamość nasyca się wraz z liczbą klatek? Czy ten sam wektor istnieje w latencie detektora z 1500 klatek i w zamrożonym backbone sprzed dostrajania?
5. Czy jakikolwiek sygnał bez etykiet przebije pewność detektora na pewnych chybieniach (0,66-0,71), co jest warunkiem przejścia wybieracza do treningu? Czy kierunki błędów z włókna pomyłek wskażą klatki z pęsetą i rękojeścią lepiej niż losowanie?
6. Dlaczego wszystkie metody wybierania „trudnych” lub „brakujących” klatek przegrywają z losowaniem: czy decyduje różnorodność, szum adnotacji w trudnych klatkach, czy brak wag reprezentatywności (kontrprzykład T8)? Czy dokładanie przykładów negatywnych (pęseta bez cystotomu) albo wklejanie narzędzia na obce tła poprawi precyzję bez utraty czułości?
7. Czy Koopman na przebiegu uczenia (dynamika latentu klatek w funkcji kroku) przewidzi klatki, które dalej uczą sieć, lepiej niż mapa zbioru danych, i czy przyspieszy dojście do 95 procent końcowego mAP?
8. Czy da się oszacować dla RF-DETR wielkości potrzebne twierdzeniom T5-T11 (stała gładkości, błąd przewidywania gradientu, błąd kierunku AdamW), żeby warunkowe gwarancje doboru klatek stały się sprawdzalne?
9. Jak liczyć dalej, skoro zbiór testowy został użyty do diagnozy: czy zostawić go wyłącznie na jedną ocenę końcową, a wszystkie decyzje podejmować na walidacji?
10. Czy filtr cząsteczkowy na obrazie z oceną przez mapę wektora jest potrzebny (ile klatek ma dwa rozdzielone jasne miejsca na mapie), czy wystarczy Langevin z Kalmanem na położeniu trzonka?


## Etap 3. Koopman z pamięcią, pętla prognoza - FastGAN - wyszukiwanie - uczenie na RF-DETR Nano i Large (25-28 września 2026)

### 3.1 Ramy etapu: detektor, sondy, latent i miary

Etap trwał od południa 25 września do 01:15 28 września 2026 (wszystkie godziny w tej części to czas lokalny, UTC+2). Wszystkie eksperymenty leżą w katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\`, a ich stan zapisuje 14 commitów repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, od `900cec4e` (25.09, 13:49) do `ebd9db38` (28.09, 01:14), na gałęzi `codex/eden-nano-training-20260914`. Obliczenia lokalne szły na karcie RTX 3070 (8 GB pamięci), uczenie detektora na klastrze Eden. Część używanych katalogów należała do równoległej sesji innego agenta (`koopman_gesta_historia`, `koopman_osobno_obraz_token`, `koopman_jeden_wektor`, `koopman_fastgan_petla`). Ta sesja korzystała z ich kodu i danych wyłącznie do odczytu.

Pojęcia, na których opiera się cały etap:

- **Detektor.** RF-DETR Nano z pakietu rfdetr 1.10.1, sieć, która na obrazie zaznacza obiekty prostokątnymi ramkami i podaje dla każdej pewność. Tu wykrywa jedną klasę: cystotom, narzędzie chirurgiczne, na klatkach filmów operacji zaćmy ze zbioru Cataract-LMM.
- **Kręgosłup i tokeny.** Pierwsza część detektora (kręgosłup, ang. backbone) to sieć DINOv2-S. Dzieli obraz 384 × 384 piksele na siatkę 24 × 24 = 576 kwadratowych łatek i każdą opisuje wektorem 384 liczb, zwanym tokenem.
- **Latent.** Wewnętrzny opis obrazu w sieci, tu: tokeny kręgosłupa albo liczby z nich wyliczone. „Tożsamość narzędzia w latencie” oznacza kierunek w tej przestrzeni, wzdłuż którego tokeny narzędzia różnią się od tokenów tła.
- **Aktualizacja, krok, epoka, checkpoint.** Aktualizacja (krok uczenia) to jedna poprawka wag przez optymalizator AdamW na 16 obrazach (porcja 4 obrazy × 4 akumulacje). Epoka to umowna porcja prezentacji obrazów, po której sieć jest oceniana; w biegu odniesienia 1600 prezentacji, czyli 100 aktualizacji. Checkpoint to zapisany stan wag w danej chwili uczenia.
- **Bieg odniesienia 1810662** (Eden, 24.09, uruchomiony przez równoległą sesję w `koopman_gesta_historia`): start od wag COCO (sieć wstępnie wyuczona na ogólnym zbiorze zdjęć COCO), 1500 oznaczonych klatek uczących (dalej „baza” albo „pretrening”), 8 epok po 100 aktualizacji, stała szybkość uczenia 1,67·10⁻⁵ (kręgosłup 1,67·10⁻⁶), wagi i latent zapisywane co 5 aktualizacji: 160 checkpointów (kroki 5-800) i 161 stanów latentu. mAP@0,5 na zbiorze walidacyjnym na końcu kolejnych epok: 0,566; 0,791; 0,860; 0,877; 0,886; 0,891; 0,896; 0,898 (krok 800: 0,8984).
- **Zbiory klatek.** Pula ucząca (train) ma 10 637 klatek, z których 1500 weszło do biegu 1810662. Zbiór walidacyjny (val) ma 12 043 klatki, których sieć nigdy nie ogląda podczas uczenia (manifest `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\lokalny_static36\pool_manifest.jsonl`).
- **Sondy.** Stałe obrazy przepuszczane przez każdy checkpoint, żeby śledzić, jak ich latent zmienia się w trakcie uczenia. Analiza używa 72 sond: 12 wycinków narzędzia (źródeł) × 6 wariantów (oryginał, obrót o 90°, narzędzie skrócone do 0,8 długości, czarne tło, żółte tło, rozjaśnione narzędzie). Podział według źródeł: 48 sond „fit” (8 źródeł, na nich dopasowuje się modele) i 24 „holdout” (4 źródła, tylko do sprawdzenia). Maska sondy mówi, jaki ułamek każdej łatki zajmuje narzędzie. Pliki latentu z uczenia (`LATENT_*.npz`) zawierają 192 sondy; 72 sondy analizy są ich podzbiorem.
- **Miary detekcji.** AP (average precision) to pole pod krzywą precyzja-czułość. Ramka liczy się jako trafiona, gdy IoU, czyli stosunek części wspólnej ramki przewidzianej i prawdziwej do ich sumy, wynosi co najmniej 0,5. Przy jednej klasie mAP@0,5 to po prostu AP@0,5. mAP@0,5:0,95 uśrednia AP po progach IoU od 0,5 do 0,95, więc karze niedokładne położenie ramki. F1 to średnia harmoniczna precyzji i czułości przy progu pewności 0,5.
- **Miary prognozy latentu.** RMSE (pierwiastek z błędu średniokwadratowego) prognozy dzielony przez RMSE założenia „nic się nie zmieni” (brak zmian). Wartość 1,0 znaczy: tak źle jak brak prognozy; 0,27 znaczy: błąd to 27 procent błędu braku zmian. „Rozrzut” to odchylenie standardowe danej współrzędnej latentu między obrazami sond; w tych jednostkach wyrażano przesunięcia latentu.
- **Ziarno, ramię, różnica sparowana.** Ziarno to liczba startowa generatora liczb losowych (kolejność klatek, losowania); powtórzenie uczenia z innym ziarnem pokazuje zwykły szum. Ramię to wariant eksperymentu (np. „bez dodania”, „losowe”). Różnica sparowana to różnica dwóch ramion przy tym samym ziarnie.
- **Bramka.** Kryterium przyjęcia wyniku zapisane przed pomiarem i niezmieniane po nim. W eksperymentach z Koopmanem: RMSE ≤ 0,95 × RMSE ostatniego przyrostu, RMSE ≤ 0,8 × RMSE braku zmian i średni cosinus kąta między przewidzianą a prawdziwą zmianą ≥ 0,5.
- **Eden.** Klaster obliczeniowy z kolejką zadań SLURM. Partycja hopper (karty H100) i partycja short (węzły dgx-1 do dgx-4, m.in. z kartami A100). Kwota dysku projektu 200 GB. Numer zadania SLURM (np. 1814611) identyfikuje każde zgłoszenie.

### 3.2 Punkt wyjścia 25 września: gęsta historia uczenia i nieudane Koopmany równoległej sesji

**Operator Koopmana** to w teorii układów dynamicznych liniowy operator, który przesuwa w czasie dowolne funkcje stanu układu, także nieliniowego. W tym projekcie oznacza w praktyce macierz dopasowaną regresją, która z obecnego (a w wersjach z pamięcią także z przeszłych) stanu latentu przewiduje stan w następnym checkpoincie.

Dane do wszystkich analiz tego dnia pochodzą z biegu 1810662. Równoległa sesja zapisała je w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_gesta_historia\` (plan `PLAN.md`, stan `EDEN_RUN.json`). Pobieranie wyników z Edena przerwała 24.09 awaria połączenia SSH na pliku `pretrain/model_0000370.pth` (`DOWNLOAD_ERROR.json`), więc lokalnie było tylko 73 checkpointy (kroki 5-365). 25.09 użytkownik poprosił o dobranie brakujących („była awaria wczoraj na Edenie i najprawdopodobniej model Astra tego nie pobrał”; „model Astra” to agent równoległej sesji). Ta sesja wznowiła pobieranie skryptem równoległej sesji `pobierz_wynik.py`; skończyło się o 16:28: 906 plików, 23,6 GB, sumy SHA256 zgodne (`...\koopman_gesta_historia\wyniki\run_1810662\VERIFIED.json`, `DOWNLOAD_STATE.json`).

Przed tym etapem (24.09 wieczorem i 25.09 nad ranem) równoległa sesja policzyła kilka wariantów Koopmana na tych danych. Żaden nie pokonał prostych kontroli. Zestawienie z ich plików:

| Analiza (katalog) | Co policzono | Wynik (RMSE, mniej = lepiej) |
|---|---|---|
| `koopman_gesta_historia\wyniki\analiza_czesciowa_full` | jeden obraz, pełny tensor 576 × 384 (221 184 liczby), 161 stanów, macierz afiniczna na położeniu dopasowana do aktualizacji 500, 41 początków prognozy 500-700 | 100 aktualizacji: Koopman 0,0609, brak zmian 0,0474, trend liniowy 0,0476; 5 aktualizacji: Koopman 0,0258, brak zmian 0,0051, ostatni przyrost 0,0042 |
| `...\koopman_przyrostowy_full` | ten sam obraz, operator uczący przyrost (K = I + B) | 5 aktualizacji: 0,0073 wobec 0,0051 braku zmian; 100 aktualizacji: 0,0662 wobec 0,0474 braku zmian |
| `...\macierze_wiele_obrazow` | 73 checkpointy, 72 obrazy, 5184 pary; osobna macierz dla każdego przejścia między checkpointami | holdout: poprzednia macierz 0,0118, ostatni przyrost 0,0043, brak zmian 0,0125 |
| `...\audyt_literatura6` | pełny latent ze stanem [z, z - z_poprzedni], 192 pary, grzbiet 0,1 | 5 aktualizacji: 0,00453 wobec 0,00425 ostatniego przyrostu; 100 aktualizacji: 0,287 wobec 0,173 braku zmian |
| `...\experiments\koopman_jeden_wektor` | jeden token (nr 299), 384 liczby, model „α × ostatni przyrost” | 5 aktualizacji: 0,00526 wobec 0,00541 (przewaga 2,77 procent, bramka 5 procent niezaliczona); 100 aktualizacji: 0,132 wobec 0,219 |
| `...\experiments\koopman_osobno_obraz_token` (25.09, ok. 04:00) | 41 472 osobne macierze 384 × 384 (72 obrazy × 576 tokenów) na samym położeniu | 1 krok: Koopman 0,01033, ostatni przyrost 0,00386, brak zmian 0,00823, model α 0,00372; 100 aktualizacji: 0,232 / 0,117 / 0,136 / 0,071; przez bramkę 0 z 41 472 |

O 12:26 użytkownik wkleił ścieżki do plików równoległej sesji i zapytał: „wczytaj te ścieżki i sprawdz dlaczego liczenie koopmana nie działa? dlaczego on nie wychodzi?”. Ta sesja napisała diagnozę (`diagnoza_przyczyny.py`, wynik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_osobno_obraz_token\DIAGNOZA_PRZYCZYNY.md`, poza gitem, bo katalog należy do równoległej sesji). Próbka: 72 obrazy × co 16. token = 2592 trajektorie, ten sam podział (dopasowanie na 48 parach do kroku 245, test 24 przejść do kroku 365).

| Pomiar | Wartość |
|---|---|
| mediana długości przyrostu tokenu, początek → koniec historii | 0,295 → 0,091 (uczenie zwalnia ponad trzykrotnie) |
| cosinus kolejnych przyrostów (bezwładność ruchu), początek / test | 0,964 / 0,903 |
| udział stanu testowego poza przestrzenią rozpiętą przez stany uczące | 0,04 |
| wyjście testu poza zakres uczenia wzdłuż głównej osi trajektorii | 0,15 długości zakresu |
| średni dryf okna uczenia / prawdziwy przyrost w teście | 1,33 |
| długość prognozy Koopmana / prawdziwy przyrost | 1,62 |
| cosinus (prognoza Koopmana, średni dryf okna uczenia) | 0,66 |

RMSE jednego kroku w tej próbce: Koopman z grzbietem 0,001: 0,0275; z grzbietem 0,1 (wybierany w planie): 0,0100; sam średni dryf: 0,0089; brak zmian: 0,0080; Koopman z grzbietem 10: 0,0070; Koopman na prędkościach (v_{t+1} = A v_t + c): 0,0055; ostatni przyrost: 0,0037; α × ostatni przyrost: 0,0036. (Grzbiet, czyli regresja grzbietowa, to metoda najmniejszych kwadratów z karą za duże współczynniki; im silniejsza kara, tym „mniej robi” macierz.)

Wniosek diagnozy, przekazany użytkownikowi: latent tokenu w czasie uczenia „jedzie jak rozpędzony wózek, który powoli hamuje”. Kolejne przesunięcia są prawie równoległe (bezwładność optymalizatora AdamW), a ich długość maleje. Stała macierz na samym położeniu uczy się średniego tempa z wcześniejszego okna i przewiduje ruch o 62 procent za duży; nowe stany leżą za końcem znanej trajektorii, więc macierz musi ekstrapolować. Stan tokenu nie jest domknięty: ruchem kieruje gradient liczony z całego zbioru i wszystkich wag. W języku formalizmu Mori-Zwanziga (rzutowanie dużego układu na kilka zmiennych: człon chwilowy + jądro pamięci + szum) brakuje członu pamięci, a ostatni przyrost jest jego najprostszym zastępstwem. Zalecenie: stan z opóźnieniem [z_t, z_t - z_{t-1}] i model przyrostów z malejącą skalą. Samo polecenie „no to wylicz to poprawnie!” wykonano już w nowym katalogu tej sesji, na danych równoległej sesji (3.3); liczba „27 procent”, o którą użytkownik pytał po południu, pochodzi z analizy skalara tożsamości (3.4), a nie z plików `koopman_gesta_historia`.

### 3.3 Koopman ze stanem z pamięcią dla każdej pary obraz-token (`koopman_z_pamiecia`, commit `900cec4e`)

**Skąd.** Po diagnozie użytkownik napisał o 13:09: „no to wylicz to poprawnie!”.

**Metoda.** Katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_z_pamiecia\` (`PLAN.md` zapisany przed uruchomieniem, skrypt `licz.py`, liczby `wyniki\REPORT.json`, tablice na jednostkę `wyniki\*.npy`, sumy `wyniki\SHA256.json`). Stan Koopmana to zanurzenie opóźnieniowe s_t = [z_t, v_t], gdzie v_t = z_t - z_{t-1} to ostatni przyrost (jak położenie i pęd w mechanice). Prognoza: z_{t+1} = z_t + v̂_{t+1}. Pięć rodzin operatora, każda osobno dla każdej z 41 472 par obraz-token, na pełnym wektorze 384 liczb, w podwójnej precyzji, bez PCA (analizy głównych składowych) i bez łączenia obrazów:

| Rodzina | Postać | Parametrów na jednostkę |
|---|---|---|
| alpha | v̂ = α v_t (jedno tłumienie) | 1 |
| ar2 | v̂ = a v_t + b v_{t-1} | 2 |
| diag(λ) | osobne tłumienie każdej współrzędnej, ściągane do α siłą λ ∈ {0,1; 1; 10} | 384 |
| vel(ρ) | α v_t + liniowa korekta z v_t (grzbiet, ρ ∈ {1; 10; 100}) | pełna 384 × 384 |
| delay(ρ) | średnia + liniowa funkcja [z_t, v_t] (grzbiet, ρ ∈ {0,1; 1; 10; 100}) | pełna 384 × 768 |

Podział identyczny z `koopman_osobno_obraz_token`: dane `koopman_gesta_historia\wyniki\macierze_wiele_obrazow` (72 obrazy × 576 tokenów × 73 checkpointy, kroki 5-365). Kalibracja: dopasowanie na stanach 0-40, prognozy jednego kroku z początków 40-47; rodzinę i parametr każda jednostka wybiera wyłącznie na tych błędach. Test: dopasowanie na stanach 0-48 (do kroku 245), zamrożenie, 24 przejścia z początków 48-71 (kroki 245-365), każda prognoza dostaje prawdziwe z_t i v_t na starcie. Wariant kroczący (alpha, ar2): współczynniki przeliczane przed każdym początkiem z danych dostępnych do tego momentu. Długi horyzont: 20 checkpointów (100 aktualizacji) od początków 48, 50 i 52, bez prawdziwych stanów pośrednich (tzw. rollout: model karmi się własnymi prognozami). Bramka jak w 3.1, niezmieniana. Czas: około 38 minut na procesorze.

**Wyniki** (wariant główny zamrożony przed testem: rodzina wybierana osobno dla każdej jednostki na kalibracji):

| Model | RMSE 1 kroku (5 aktualizacji) | RMSE 100 aktualizacji | cosinus przyrostu | jednostek przez bramkę |
|---|---|---|---|---|
| Koopman z pamięcią, wybór na kalibracji | 0,003708 | 0,0762 | 0,865 | 7 445 z 41 472 |
| ostatni przyrost | 0,003859 | 0,1170 | 0,864 | 0 |
| brak zmian | 0,008226 | 0,1362 | - | 0 |
| poprzedni Koopman na samym położeniu | 0,010331 | 0,2322 | 0,619 | 0 |

Rodziny stosowane jednakowo do wszystkich jednostek (porównanie po teście, więc eksploracyjne):

| Rodzina | RMSE 1 kroku | przewaga nad ostatnim przyrostem | RMSE 100 aktualizacji | przez bramkę |
|---|---|---|---|---|
| vel_1 | 0,003672 | +4,8 % | 0,0855 | 14 374 |
| vel_10 | 0,003692 | +4,3 % | 0,0712 | 10 637 |
| vel_100 | 0,003714 | +3,8 % | 0,0697 | 5 519 |
| diag_0.1 | 0,003693 | +4,3 % | 0,0760 | 10 753 |
| alpha | 0,003718 | +3,7 % | 0,0711 | 4 375 |
| ar2 | 0,003733 | +3,3 % | 0,0730 | 4 914 |
| alpha kroczące | 0,003713 | +3,8 % | - | 5 630 |
| delay_1 (stan z położeniem) | 0,005308 | -37,6 % | 0,0997 | 2 |

Pozostałe warianty delay: -47,9 % (ρ = 0,1), -56,7 % (ρ = 10), -101,1 % (ρ = 100). Wybór na kalibracji brał najczęściej ar2 (18 293 jednostki), vel_1 (7 219) i vel_10 (6 892), rodzinę z położeniem tylko dla 2 jednostek.

**Interpretacja.** Pamięć naprawia Koopmana: błąd jednego kroku spadł prawie trzykrotnie wobec poprzedniej wersji (0,0103 → 0,0037) i jest o 3,9 procent niższy niż powtórzenie ostatniego przyrostu; na 100 aktualizacji jest o 35 procent niższy od ostatniego przyrostu i o 67 procent od poprzedniego Koopmana. Bramkę przechodzi 18 procent jednostek, wcześniej zero. Średnia przewaga 3,9 procent nie osiąga jednak progu 5 procent dla całości, a progu nie zmieniono. Dodanie samego położenia do stanu psuje prognozę o 38-101 procent. Większość zysku daje samo tłumienie prędkości; pełna macierz 384 × 384 dokłada około procentu. Cosinus przyrostu prawie nie rośnie (0,864 → 0,867): przewidywalne jest tempo i hamowanie ruchu, a nie nowy kierunek.

**Czego to nie dowodzi.** Pula była wcześniej oglądana, porównania rodzin po teście są eksploracyjne, operator nie ma sprzężeń między tokenami. Dobra prognoza przebiegu uczenia nie mówi, które klatki poprawią detektor.

### 3.4 Elementy latentu identyfikujące narzędzie: skalar tożsamości i najlepszy operator (`koopman_tozsamosc_kierunki`, część 1, 73 checkpointy)

**Skąd.** O 15:26 użytkownik uruchomił przegląd literatury (polecenie `/paper-search`) z prośbą: plan w `koopman_gesta_historia\PLAN.md` zakłada „wyznaczone pewne elementy tego latentu, które identyfikują narzędzie”, co nazywa „pełną identyfikacją narzędzia w latencie”; chciał sprawdzić w literaturze, czy Koopman może przewidywać zmianę tych elementów, dlaczego wychodzi gorzej, niż oczekuje, dobrać brakujące checkpointy z Edena i „wyliczyć jak najlepszego [Koopmana], który będzie właśnie te wektory identyfikujące narzędzie przewidywał”.

**Metoda.** Katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_tozsamosc_kierunki\` (`PLAN.md`, `WYNIKI.md` sekcje 1-4, skrypty `licz.py`, `najlepszy.py`, raporty `wyniki\REPORT_K73_S48.json`, `wyniki\REPORT_najlepszy_K73_S48.json`, operatory `wyniki\OPERATORY_K73_S48.npz`). Definicje liczone tylko na sondach fit i checkpointach do kroku 245:

- `w_lda`: kierunek LDA (liniowej analizy dyskryminacyjnej, czyli kierunku najlepiej rozdzielającego dwie grupy) między tokenami narzędzia (maska ≥ 0,5) a tokenami tła (maska = 0), jeden wektor w 384 wymiarach.
- `S_k`: k głównych kierunków różnicy „token narzędzia minus średnie tło obrazu”, k ∈ {1, 4, 8, 16, 32}.
- **Skalar tożsamości** obrazu i: a_i(t) = w_lda · (m_i(t) - tło_i(t)), gdzie m_i to średnia tokenów ważona maską narzędzia.
- Kontrole: losowa podprzestrzeń tego samego wymiaru, dopełnienie, losowe kierunki poza S_32.

Modele na jednostkę: alpha, ar2, prawo hiperboliczne 1/t (z pracy Gur-Ari i in. 2018), pełne macierze na prędkości albo położeniu w S_k. Modele wspólne dla wszystkich obrazów: macierz na prędkości z L = 1-4 opóźnieniami (stan Hankela, czyli L ostatnich przyrostów ułożonych obok siebie), wariant z prawem 1/t. Siłę grzbietu i stałą τ wybierano wyłącznie na końcówce okna obrazów fit. Dopasowanie do kroku 245 (S = 48 zapisów), test 24 przejść do kroku 365, rollout 20 kroków (100 aktualizacji). Miara: RMSE względem braku zmian na obrazach holdout. Bramka ta sama co wcześniej.

**Wyniki.**

1. Kierunek LDA rozdziela tokeny narzędzia od tła na obrazach holdout z AUC 0,995 już w kroku 5 (AUC, pole pod krzywą ROC, to prawdopodobieństwo, że losowy token narzędzia dostanie wyższy wynik niż losowy token tła) i 0,998 w krokach 245 i 365. Tożsamość narzędzia nie powstaje w tym dostrajaniu, jest dziedziczona z wstępnie wyuczonego kręgosłupa i tylko się doostrza. Kierunek obraca się: cosinus z kierunkiem z kroku 5 spada do około 0,75 (obrót o około 40° w pierwszych 100 aktualizacjach). Jest prawie prostopadły do głównych kierunków zmienności różnicy „narzędzie minus tło” (cosinus 0,02-0,10 z podprzestrzeniami PCA k = 1-32).

2. Poziom tokenów narzędzia (1458 jednostek, każda z własnym modelem), RMSE względem braku zmian:

| Podprzestrzeń | alpha | ostatni przyrost | pełna macierz na prędkości | Koopman na położeniu |
|---|---|---|---|---|
| tożsamość S_1 | 0,507 | 0,521 | 1,46 | 1,49 |
| losowa 1-wymiarowa | 0,384 | 0,400 | 1,26 | 1,30 |
| tożsamość S_8 | 0,516 | 0,532 | 1,16 | 1,20 |
| losowa 8-wymiarowa | 0,512 | 0,529 | 1,16 | 1,15 |
| tożsamość S_32 | 0,486 | 0,501 | 0,89 | 1,04 |
| dopełnienie (352 wymiary) | 0,478 | 0,493 | 0,85 | - |
| pełne 384 | 0,479 | 0,494 | - | - |

Na poziomie tokenów podprzestrzeń tożsamości nie jest bardziej przewidywalna niż losowa. Prawo 1/t (τ = 10) nie jest lepsze od stałego α (0,490 wobec 0,479).

3. Poziom obrazu, skalar tożsamości, 72 trajektorie (RMSE względem braku zmian, holdout):

| Model | 1 krok | 20 kroków (100 aktualizacji) |
|---|---|---|
| **wspólny Koopman na prędkości, 4 opóźnienia (L4), 48 obrazów fit** | 0,272 | **0,171** |
| wspólny, 3 opóźnienia | 0,281 | 0,182 |
| wspólny, 1 opóźnienie | 0,272 | 0,238 |
| wspólny z prawem 1/t | 0,277 | 0,238 |
| alpha na obraz | 0,276 | 0,285 |
| hiperboliczne na obraz | 0,281 | 0,445 |
| ostatni przyrost | 0,284 | 0,617 |
| relaksacja afiniczna a_{t+1} = ρ a_t + c (Koopman z punktem stałym) | 0,657 | - |
| brak zmian | 1,0 | 1,0 |

Najlepszy operator: jedna macierz wspólna dla wszystkich obrazów, na czterech ostatnich przyrostach skalara tożsamości, dopasowana na 48 obrazach fit (48 × 47 = 2256 par). Przechodzi bramkę. Wszystkie wartości własne jego macierzy towarzyszącej (macierzy opisującej krok całego stanu z opóźnieniami) mają moduł poniżej 1, największy 0,944, więc prognoza hamuje i się nie rozbiega. (Pierwsze sprawdzenie widma dało 1,081 przez odwróconą kolejność opóźnień; po poprawce wszystkie stabilne.)

4. Opis narzędzia w podprzestrzeni S_k (poziom obrazu): S_1 alpha 0,418 (1 krok) i 0,222 (20 kroków); S_4 alpha / wspólny L4: 0,395 i 0,273 / 0,290; S_8: 0,399 i 0,294; S_16: 0,403 i 0,350; S_32: 0,394 i 0,365; 8 losowych kierunków poza S_32: 0,356 i 0,360; ostatni przyrost na 20 kroków 0,419-0,731; klasyczny Koopman na położeniu 3,5-8,7 (1 krok) i 2,0-2,5 (20 kroków). Im mniej wymiarów, tym lepiej, ale losowe kierunki dają to samo co S_8: pomaga uśrednienie po masce i mały wymiar, nie sama tożsamość.

5. Pięć przyczyn słabszych wyników (sekcja 3 `WYNIKI.md`): liniowość jest w prędkości, nie w położeniu; stan tokenu nie jest domknięty (Mori-Zwanzig); 47 par nie wyznacza macierzy 384 × 384 ani nawet 8 × 8, wspólny operator jest identyfikowalny tylko w małym wymiarze; dla jednej klasy kierunek „zadaniowy” jest jeden (zgodnie z Gur-Arim: górna podprzestrzeń Hessjanu ma wymiar równy liczbie klas), reszta to „bulk”, którego uczenie nie rusza systematycznie; tłumienie jest bliższe wykładniczemu (α ≈ 0,9 na 5 aktualizacji) niż prawu 1/t.

6. Literatura (Undermind, pełne teksty, workspace „ViT: detekcja końcówki narzędzia”; Scite bez not o wycofaniu): 11 prac, m.in. Dogra i Redman 2020 (Koopman na wagach sieci), Redman i in. 2024 (8-64 opóźnień), Manojlović i in. 2020, Tano i in. 2020, Gur-Ari i in. 2018, Xu i in. 2026, Marcotte i in. 2025, Aswani i in. 2024/2025, Henry 2026, Park i in. 2023. Żadna nie przewiduje Koopmanem ewolucji kierunku pojęcia w latencie w trakcie uczenia; prace o Koopmanie na treningu działają na wagach i wszystkie używają opóźnień.

**Rozmowa o znaczeniu wyniku (16:01-16:14).**

- „czyli mam już poprawnie policzonego koopmana?” Odpowiedź: tak, operator jest poprawnie policzony i stabilny (|λ| maks. 0,94), ale na pierwszej połowie uczenia; obrazy holdout to cztery źródła oglądane już w innych eksperymentach, więc potwierdzenie nie jest w pełni niezależne.
- „w jaki sposób ty bierzesz i liczysz teraz tego Koopmana, czy według tej instrukcji?” Odpowiedź: nie w pełni. Najlepszy operator łamie dwa wyraźne zakazy `koopman_gesta_historia\PLAN.md` (uśrednianie tokenów pod maską jako stan, rzut na jeden kierunek), a trzecim odstępstwem jest jedna macierz dla wszystkich obrazów. Zgodne z planem są: pary tylko w obrębie tego samego obrazu, podział chronologiczny, kontrole i bramka, identyfikacja kierunku bez wstępnego PCA. Wersja zgodna z instrukcją (pełne 384 liczby każdego tokenu) też jest policzona i nie daje przewagi.
- „czy ja mogę [...] powiedzieć, że ta analiza [Koopmana] przewiduje, gdzie będzie wektor, który reprezentuje tożsamość narzędzia w następnej chwili T plus 1?” Odpowiedź: tak, ale tylko dla jednej liczby (skalara tożsamości), nie dla całego wektora. Błąd prognozy na obrazach holdout to 27 procent błędu założenia „nic się nie zmieni”, ale powtórzenie ostatniej zmiany daje 28 procent. Przewaga Koopmana ujawnia się dopiero na 100 aktualizacji: 17 procent wobec 62 procent.
- „Błąd tej prognozy [...] wynosi 27 procent tego, co daje założenie »nic się nie zmieni«. [...] czyli on przewiduje lepiej [...], prawda?” Odpowiedź: tak; błąd Koopmana to niecała jedna trzecia błędu braku zmiany, czyli Koopman odgaduje około trzech czwartych rzeczywistego przesunięcia skalara w następnym checkpoincie. Ale „nic się nie zmieni” to słaby przeciwnik, bo skalar stale rośnie; z regułą „przesunie się o tyle, co ostatnio” Koopman wygrywa dopiero na długim horyzoncie, bo ma wbudowane hamowanie.
- „jaka jest kolejna instrukcja do zrobienia?” Odpowiedź według `koopman_gesta_historia\PLAN.md`: pełna identyfikacja wektorów tożsamości na palecie 256 jednolitych teł z kontrolami bez narzędzia, potem zaplanowany test wpływu kierunków na decyzję detektora (hipoteza H2: czy przewidziany przyrost dodany do latentu zmniejsza błąd zamrożonego detektora), na końcu pokaz FastGAN. Na pytanie „to aktualnie mamy już zrobione, prawda?” odpowiedź brzmiała: nie. Jest jeden kierunek na 72 sondach z dwoma tłami (czarnym i żółtym); paleta 3072 obrazów (12 wycinków × 256 teł, `koopman_gesta_historia\wyniki\probe_rgb256`) nie ma policzonego latentu; nie ma kontroli „to samo tło bez narzędzia” ani podziału na wnętrze, brzeg i tło.

### 3.5 Pełna historia 5-800 (część 2, 160 checkpointów)

**Metoda.** Po pobraniu brakujących checkpointów 370-800 tokeny wszystkich 72 sond wyciągnięto lokalnie tym samym kodem co w części 1 (`ekstrakcja.py`, zapis do `...\koopman_tozsamosc_kierunki\wyniki\tokeny_pelne`, 160 plików, 9,5 GB, poza gitem). Dwa okna: wczesne (S = 48, dopasowanie do kroku 245, test 111 przejść 245-795) i późne (S = 96, dopasowanie do kroku 485, test 63 przejść 485-795). Raporty `wyniki\REPORT_K160_S48.json`, `REPORT_K160_S96.json`, `REPORT_najlepszy_K160_S{48,96}.json`, `ROLLOUT_DLUGI_K160_S{48,96}.json` (`rollout_dlugi.py`). Problem techniczny: pełne tokeny zajmują 10 GB pamięci RAM; pierwsze uruchomienie `licz.py` padło na braku pamięci (równolegle liczył się pokaz FastGAN), `najlepszy.py` dla S = 96 nie zaalokował 9,49 GiB. Naprawa: skrypty zwalniają tokeny zaraz po wycięciu potrzebnych trajektorii; powtórzone przy 18,6 GB wolnej pamięci. Wynik zapisano razem z częścią 1 i pierwszym pokazem FastGAN w commicie `654ae124` (17:01).

**Wyniki.**

- AUC LDA na holdout: 0,995 (krok 5), 0,998 (krok 245), 0,998 (krok 800). Kierunek obraca się dalej, powoli: cosinus z kierunkiem z kroku 5 wynosi 0,73 w kroku 365 i 0,64 w kroku 800.
- Średni przyrost skalara tożsamości na zapis: 0,0085 na starcie, około 0,001 w oknie 245-365, około 0 (±0,0008) po kroku 485. **Dryf tożsamości kończy się około kroku 500.** Od kroku 245 do 800 średnia przesuwa się łącznie o 0,128 rozrzutu, od kroku 485 do 800 o 0,028 rozrzutu.

| Okno | Model | 1 krok | 20 kroków |
|---|---|---|---|
| wczesne (dopasowanie do 245, test 245-795) | wspólny L4 | 0,407 | 0,171 |
| | alpha na obraz | 0,421 | 0,285 |
| | ostatni przyrost | 0,433 | 0,617 |
| późne (dopasowanie do 485, test 485-795) | wspólny L4 | 0,682 | 0,454 |
| | alpha na obraz | 0,712 | 0,538 |
| | ostatni przyrost | 0,733 | 0,996 |

Bramka spełniona w obu oknach dla L4, w oknie wczesnym na 1 krok ledwo (0,407 przy progu 0,411). Widmo L4: |λ| maks. 0,944 (wczesne) i 0,939 (późne); suma wag opóźnień 0,92.

Długi rollout od kroku 245 bez prawdziwych stanów pośrednich:

| Horyzont (zapisy) | Krok | L4 | alpha | ostatni przyrost | błąd średniej L4 (rozrzuty) | błąd średniej alpha |
|---|---|---|---|---|---|---|
| 20 | 345 | 0,171 | 0,285 | 0,62 | 0,000 | 0,000 |
| 40 | 445 | 0,254 | 0,344 | 1,04 | 0,000 | -0,009 |
| 60 | 545 | 0,290 | 0,426 | 1,66 | +0,026 | +0,002 |
| 80 | 645 | 0,359 | 0,403 | 2,02 | +0,039 | -0,005 |
| 111 | 800 | 0,417 | 0,451 | 2,85 | +0,078 | -0,002 |

L4 przewiduje lepiej na obraz przez cały horyzont (0,42 wobec 0,45 po 555 aktualizacjach), ale średnią po obrazach przestrzeliwuje po kroku około 500: przewiduje łączny dryf 0,21 rozrzutu wobec rzeczywistego 0,13. Tłumienie alpha każdego obrazu trafia w średnią do końca (błąd ≤ 0,01 rozrzutu). Rollout od kroku 485: L4 0,45 po 20 krokach i 0,58 na końcu, alpha 0,45 na końcu.

Poziom tokenów w oknie późnym jak w części 1: alpha 0,767, ostatni przyrost 0,797, pełna macierz na prędkości 1,02; S_8 0,763 wobec losowej 0,781. Opis w S_k (20 kroków): S_1 alpha 0,426, S_4 L4 0,463, S_32 alpha 0,557, 8 losowych kierunków 0,451.

**Wnioski.** Dryf tożsamości to zjawisko wczesne: hamuje wykładniczo (α ≈ 0,89 na 5 aktualizacji) i kończy się przy kroku około 500. Operator L4 pozostaje najlepszy na obraz, ale jego przewaga maleje, gdy dryf znika (0,17 → 0,45 na 20 kroków). Do przewidywania przyrostu ogólnego na długi horyzont lepsze jest tłumienie alpha. Pokaz FastGAN „od kroku 800 do 900” nie ma sensu, bo Koopman przewiduje tam brak zmian ogólnych.

### 3.6 Pokaz FastGAN na latencie przewidzianym przez Koopmana: pierwsza pętla generatora i jej porażka

**Skąd.** Między 16:17 a 16:25 użytkownik doprecyzował zadanie w czterech wiadomościach: „Ja chcę, żebyś wyliczył przewidywany latent przez [Koopmana] i tak jakby go ustawił jako taki, który już będzie w tym detektorze [...] I teraz na podstawie tego latentu ja chcę wygenerować obraz przy pomocy [FastGAN-a]”; „Nie wydaje mi się, że dla konkretnego obrazu, tylko ogólnie jak zmieni się latent dla kolejnych kroków uczenia”; „To miało być tak, że zamrażasz latent, do którego już jest dodane przewidziane zwiększenie się tego latentu przez Koopmana [...] i przy pomocy [FastGAN-a] generujesz obraz, który będzie odpowiadał [...] temu zamrożonemu detektorowi”; „No to wykonaj to teraz na moim lokalnym kompie. Może działać w tle”. Etap palety teł użytkownik sam odłożył („na razie możemy ominąć to kodowanie tego tła”).

**Pojęcia.** **FastGAN** to lekka sieć generatywna typu GAN (dwie sieci uczone przeciwko sobie: generator tworzy obrazy, dyskryminator odróżnia je od prawdziwych). Tu generator wytrenowany wcześniej na Edenie przez 50 000 kroków (katalog z datą 23.09: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_narzedzie\wyniki\eden_20260923_50k\results_50k\checkpoint_050000.pth`) zamienia wektor wejściowy z (256 liczb) na obraz narzędzia na tle wraz z maską. **Kotwica** to prawdziwy obraz sondy, od którego zaczyna się generowanie i do którego generator jest „przywiązany”; celem jest jej obecny kod plus przewidziany przyrost. **Pętla generatora** zmienia tylko z: generator robi obraz, zamrożony detektor go czyta, liczy się odległość odczytu od celu, a jej gradient płynie wstecz przez detektor i generator do z.

**Metoda** (`pokaz_fastgan.py`, maszyneria `koopman_fastgan_petla\generowanie.py` równoległej sesji, tylko odczyt). Zamrożony detektor `model_0000245.pth` (krok 245), operatory `OPERATORY_K73_S48.npz`, horyzont 20 zapisów, czyli cel = stan w kroku 345, sprawdzalny na prawdziwym checkpoincie. Różniczkowalny czytnik detektora daje opis 1536 liczb (średnia tokenów ważona maską narzędzia, średnia tła, momenty położenia x i y); z niego kod K = 5: skalar tożsamości w_lda · (narzędzie - tło) i 4 kierunki S_4. **Przyrost ogólny** (zgodnie z poprawką użytkownika: jeden dla wszystkich obrazów) to wspólny operator prędkości L4 zastosowany do średniej po obrazach historii prędkości i zsumowany po 20 krokach. Kotwice: TR_0006 (sonda fit nr 0, klatka 746), TR_0082 (fit nr 24, klatka 1656), TR_0134 (holdout nr 48, klatka 341); każdej dodano ten sam przyrost ogólny. Ramiona: `target` (cel Koopmana), `current` (obecny kod, kontrola) i `random_norm` (losowy przyrost tej samej długości). Strata: średnia ((kod - cel)/rozrzut)² + 0,1 · L1(RGB, kotwica) + L1(maska) + 0,001 · średnia(z²); 40 kroków × 2 restarty, start z losowego z. Bramki: błąd skalowany ≤ 0,25, IoU maski ≥ 0,5, L1 do kotwicy ≤ 0,25. Pierwsze uruchomienie padło na wymogu deterministycznych obliczeń macierzowych w bibliotece CuBLAS; po ustawieniu `CUBLAS_WORKSPACE_CONFIG=:4096:8` pokaz przeszedł w 189 sekund. Wyniki: `...\koopman_tozsamosc_kierunki\wyniki\fastgan_K73_krok0245\REPORT.json`, obrazy `kotwica_XX\{target,current,random_norm}\best.png`, przegląd `PRZEGLAD.png`.

| Wielkość | Wartość |
|---|---|
| przyrost ogólny w rozrzutach (5 wymiarów kodu) | 0,063; -0,002; 0,021; 0,018; 0,009 (długość 0,069) |
| błąd prognozy średniej wobec prawdziwego kroku 345 (rozrzuty) | 0,001; 0,011; -0,014; -0,006; 0,012 |
| RMSE prognozy na obraz wobec braku zmian | 0,885 |
| błąd skalowany ramienia `current` (TR_0006 / TR_0082 / TR_0134) | 0,67 / 0,63 / 0,61 (bramka 0,25) |
| błąd skalowany ramienia `target` | 2,40 / 0,94 / 1,47 |
| błąd skalowany ramienia `random_norm` | 0,47 / 0,57 / 0,47 |
| bramki zaliczone | 0 z 9 |

**Interpretacja.** Część Koopmanowska zadziałała: średni przyrost trafił w prawdziwy krok 345 z błędem poniżej 0,015 rozrzutu w każdym wymiarze. Przyrost ogólny jest jednak mały (0,06 rozrzutu wzdłuż tożsamości, ≤ 0,02 w kontekście), a prognoza na obraz jest tylko o 11 procent lepsza od braku zmian (0,885), bo przyrosty poszczególnych obrazów są większe od wspólnego dryfu i mają różne znaki. Zawiódł generator: pętla nie odtwarzała nawet obecnego kodu kotwicy (0,6 rozrzutu przy progu 0,25), najlepszy krok wypadał zwykle w krokach 0-5, potem strata rosła. Różnica 0,06 między `target` i `current` ginęła w szumie optymalizacji; obrazy były rozmytymi próbkami FastGAN (oko, cienka linia narzędzia), nie odtworzeniem kotwicy.

Na pytania użytkownika o 17:21 („gdzie zdjęcia z fastgan?”) i 17:22 („czy to ogólnie działa czy te fastgan generuje zdjęcia dla przewidywanego latentu?”) odpowiedź brzmiała: obrazy leżą w podanym katalogu, ale to jeszcze nie działa, bo to zwykłe próbki generatora, a nie obrazy przewidzianego latentu. Zaproponowano naprawę: start z z dopasowanego najpierw do kotwicy i setki kroków zamiast 40.

### 3.7 Naprawa pętli: inwersja kotwicy i krok Gaussa-Newtona (commit `9f17339d`)

**Skąd.** 17:24: „to napraw tę pętlę i puść jeszcze raz”.

**Inwersja** (`pokaz_fastgan_inwersja.py`, wyniki `...\wyniki\fastgan_K73_krok0245_inwersja\`). Inwersja generatora to znalezienie takiego wejścia z*, dla którego generator odtwarza kod (i wygląd) danej kotwicy. Pierwsza próba (szybkość uczenia 0,01, 300 kroków) grzęzła przy błędzie około 1,8 (np. kotwica 0, restart 1: 1,826). Krajobraz straty okazał się chaotyczny: ten sam start przy szybkości 0,05 kończył na 0,67 albo 2,4 zależnie od celu różniącego się o 0,06 rozrzutu. Zadziałał schemat dwufazowy: 6 startów × 100 kroków (szybkość 0,05), potem doszlifowanie dwóch najlepszych × 200 kroków (0,01), zwycięzca po pełnej stracie. Od z* cztery ramiona optymalizatorem Adam (szybkość 0,01, 100 kroków, ten sam szum warstw): cel Koopmana, obecny kod, losowy przyrost tej samej długości i jawnie oznaczone ramię `target_x10_NIE_PROGNOZA` (przyrost × 10, nie prognoza).

| Kotwica | Inwersja: błąd / IoU / L1 | Cel Koopmana: błąd, cosinus zmiany, L1 RGB do „obecny” | × 10 (nie prognoza): błąd, cosinus, L1 RGB |
|---|---|---|---|
| TR_0006 (fit) | 0,21 / 0,67 / 0,31 | 0,213; 0,31; 0,052 | 0,277; 0,92; 0,078 |
| TR_0082 (fit) | 0,27 / 0,46 / 0,19 | 0,268; ramię nie ruszyło się z z* (L1 RGB 0) | 0,313; 0,84; 0,018 |
| TR_0134 (holdout) | 0,09 / 0,40 / 0,23 | 0,099; 0,10; 0,008 | 0,160; 0,94; 0,021 |

Inwersja działa (błąd kodu 0,09-0,27, bramka kodu zaliczona w 2 z 3), narzędzie stoi w miejscu i orientacji kotwicy; pełne bramki nie przechodzą przez IoU maski i L1 do prawdziwego zdjęcia. Dla przyrostu × 1 ramiona Adama albo stoją, albo ich ruch jest szumem (cosinus 0,10-0,31; losowy przyrost na TR_0134 daje 0,72). Przy × 10 pętla idzie w kierunku Koopmana (cosinus 0,84-0,94, 79-111 procent zamierzonej długości).

**Krok Gaussa-Newtona** (`pokaz_fastgan_gn.py`, wyniki `...\wyniki\fastgan_K73_krok0245_gn\`). Zamiast błądzenia optymalizatora: w punkcie z* liczy się macierz pochodnych kodu po wejściu generatora J (5 × 256, automatyczne różniczkowanie) i wykonuje krok Levenberga-Marquardta o najmniejszej normie, dz = Jᵀ (J Jᵀ + λI)⁻¹ r, gdzie r to brakująca do celu reszta kodu. To linearyzacja odwzorowania z → kod z tłumieniem λ (startowo 0,01 × ślad(J Jᵀ)/5), zwiększanym, gdy reszta nie maleje; 12 iteracji, próg zatrzymania 0,02. Krok jest deterministyczny i lokalny, więc różnice między ramionami wynikają tylko z różnic celów.

| Kotwica | Reszta do „obecny”: z* → po kroku | Cel Koopmana: reszta, cosinus zmiany, udział długości, L1 RGB | × 10 (nie prognoza): cosinus, udział, L1 RGB |
|---|---|---|---|
| TR_0006 (fit) | 0,207 → 0,204 | 0,211; 0,84; 59 %; 0,0018 | 0,85; 48 %; 0,0075 |
| TR_0082 (fit) | 0,273 → 0,127 | 0,157; 0,86; 169 %; 0,0092 | 0,97; 83 %; 0,0090 |
| TR_0134 (holdout) | 0,093 → 0,075 | 0,076; 0,97; 87 %; 0,0013 | 0,93; 80 %; 0,0078 |

**Wnioski pokazu.** Mechanika działa: generator realizuje zmianę latentu w kierunku Koopmana z cosinusem 0,84-0,97 i 59-87 procent zamierzonej długości (169 procent dla TR_0082, bo ramię „obecny” samo się przesunęło). Losowy przyrost tej samej długości jest realizowany podobnie (cosinus 0,37-0,89), więc test nie wyróżnia kierunku Koopmana, tylko pokazuje, że pętla go nie gubi. **Obraz przewidzianego latentu jest nieodróżnialny od obrazu obecnego**: różnica L1 RGB 0,001-0,009 (0,1-0,9 procent zakresu jasności). To nie wada generatora, tylko wielkość zjawiska: dryf tożsamości na 100 aktualizacji to 6 procent rozrzutu między obrazami, do końca uczenia 13 procent. Różnica wzmocniona 30 razy (`PRZEGLAD.png`) skupia się na narzędziu i jego krawędziach. Pętla ma sens dla kierunków o dużej amplitudzie (np. różnic między obrazami), nie dla dryfu uczenia.

**Pokrycie przewidzianego latentu.** Na prośbę z 17:52 („powiedz ile te zdjęcia pokrywają przewidzianego latentu i pokaż które go najbardziej pokrywają”) powstał arkusz `...\fastgan_K73_krok0245_gn\POKRYCIE.png`, posortowany od najlepszego. Najlepiej pokrywa cel obraz dla TR_0134: jego latent leży 0,076 rozrzutu od przewidzianego, kierunek zmiany zgadza się w 97 procentach, zrealizował 87 procent przyrostu. Dalej TR_0082 (0,157) i TR_0006 (0,211). Sam przyrost ma długość 0,069 rozrzutu, więc tylko najlepszy obraz jest bliżej celu niż długość przyrostu; dwa pozostałe ogranicza błąd inwersji.

**TR_0134.** Między 18:10 a 18:13 użytkownik dopytał, czy TR_0134 to prawdziwa klatka i czy wchodziła do treningu testowanego detektora. Odpowiedzi: to prawdziwa klatka z filmu operacji; „holdout” oznaczał tylko wyłączenie z dopasowania Koopmana; oba obrazy obok kotwicy w arkuszu („obecny” i „przewidziany”) to wyjścia FastGAN, różniące się jednym krokiem. Sprawdzenie w `...\koopman_gesta_historia\wyniki\run_1810662\datasets\pretrain\train\_annotations.coco.json`: TR_0134 i TR_0082 były wśród 1500 klatek uczących biegu 1810662 (do kroku 245 sieć obejrzała TR_0134 około 2,5 raza), TR_0006 nie była. Pokaz niczego nie dokładał do treningu.

### 3.8 Nowe klatki pod obrazy FastGAN: wyszukiwanie i pilot 17 klatek (commit `6360929e`)

**Skąd.** 18:14: „chcę, żebyś znalazł [...] klatkę, która nie wchodziła do tej pory do zbioru uczącego tego detektora [...] jeżeli ona wchodziła do pre-trainingu, to ona się nie liczy dla mnie [...] musisz znaleźć nową klatkę, która będzie dopasowana do tych obrazów wygenerowanych przez [FastGAN]”. O 18:16 dopisał: „możesz znaleźć kilka takich klatek, no i potem je włączyć [...] do tego zbioru, który uczył ten detektor i kolejny trening zrobić, do kolejnej epoki [...] Chciałbym zobaczyć, jak te klatki poprawiają detekcję [...] czy może pogarszają. Zrób to, to są kolejne etapy, wykonuj poszczególne etapy”. Traktowano to jako zgodę z góry na etapy: wyszukanie, dodanie, jedna epoka uczenia, ocena.

**Wyszukiwanie** (wyniki `...\koopman_tozsamosc_kierunki\wyniki\dobor_klatek\`). Wyszukiwarka równoległej sesji (`koopman_fastgan_petla\wyszukiwanie.py`, tylko odczyt): sieć DINOv2 ViT-S/14 (ogólna sieć opisująca obrazy, wyuczona bez etykiet) liczy wektor cech kwadratowego wycinka wokół ramki narzędzia (margines 0,1) jako średnią tokenów; biblioteka FAISS (szybkie wyszukiwanie najbliższych sąsiadów po iloczynie skalarnym) przeszukuje indeks 10 637 klatek train (`koopman_fastgan_petla\wyniki\local_1810448\run\index`). Podobieństwo to cosinus kąta między wektorami cech. Zapytania: obrazy `target.png` (i kontrolnie `current.png`) z `fastgan_K73_krok0245_gn\kotwica_XX`. Wykluczone: 1500 klatek pretreningu i 3 kotwice (`wykluczone_pretrain_1500_plus_kotwice.json`, 1501 pozycji); kwalifikuje się 9 136 klatek. Ranking: najlepsza klatka z każdego filmu, potem druga. Równoczesne ładowanie faiss i torch wywołało konflikt dwóch kopii biblioteki wątków OpenMP; obejście `KMP_DUPLICATE_LIB_OK=TRUE`.

| Kotwica | Top-3 (film/klatka, cosinus DINOv2) |
|---|---|
| TR_0006 | TR_0076/1816 0,752; TR_0036/611 0,747; TR_0055/5481 0,747 |
| TR_0082 | TR_0038/1591 0,746; TR_0018/9001 0,745; TR_0081/651 0,729 |
| TR_0134 | TR_0062/391 0,812; TR_0096/5506 0,807; TR_0148/356 0,803 |

Rankingi dla obrazu celu i obrazu obecnego są niemal identyczne (wspólne w top-20: 19/20, 15/20, 20/20, razem 54/60), bo obrazy różnią się o 0,1 procent jasności. Dobór odzwierciedla więc obraz obecnego latentu kotwicy, nie przyrost Koopmana. Do treningu wzięto top-6 na kotwicę: **17 unikalnych klatek z 14 filmów**, cosinus 0,71-0,81 (powyżej progu 0,7 z kampanii równoległej sesji), żadna z filmów kotwic (`kandydaci_top6.json`, przegląd `KANDYDACI.png`).

**Kontynuacja treningu** (`eden\kontynuacja.py`, `eden\kontynuacja.slurm`, zadanie Eden **1814538**, tablica 3 ziaren, partycja hopper). Kontynuacja oznacza, że uczenie startuje z zapisanych wag `model_0000245.pth` biegu 1810662, a nie od nowa; plik zawiera same wagi, więc optymalizator startuje „świeży” (bez zapamiętanych pędów AdamW). Receptura i kod uczenia równoległej sesji bez zmian (`trening.py`: porcja 4 × 4, stała szybkość 1,67·10⁻⁵, obliczenia deterministyczne, zapis sond co epokę). Jedna epoka = 100 aktualizacji = 1600 prezentacji (każda klatka raz plus losowe dopełnienie). Ramiona: `bez_dodania` (1500 klatek), `koopman_fastgan` (1500 + 17 znalezionych), `losowe` (1500 + 17 losowych nieużytych klatek train, ziarno losowania 42). Ziarna uczenia 42, 43, 44. Miara: mAP@0,5, mAP@0,5:0,95 i F1 na całym val po epoce. Odniesienie z biegu 1810662: 0,860 na końcu trzeciej epoki (krok 299), 0,877 na końcu czwartej (krok 399). Założenie przed wynikiem: 17 klatek to 1,1 procent zbioru, więc efekt będzie najwyżej rzędu szumu ziaren. Jedno ramię trwało około 17 minut.

| Ramię | ziarno 42 | ziarno 43 |
|---|---|---|
| bez_dodania (1500) | 0,8667 | 0,8688 |
| koopman_fastgan (1517) | 0,8667 | przerwane |
| losowe (1517) | 0,8663 | przerwane |

W ziarnie 42 trzy ramiona mieszczą się w 0,0004, a sama zmiana ziarna przesuwa wynik o 0,002. Na pytanie „a ile zdjęć dodałeś do treningu?” (19:28) padła odpowiedź: 17, około 1 procent zbioru; żeby efekt był mierzalny, trzeba 150-300 klatek. Użytkownik: „dobierz top-50 na kotwicę i puść jeszcze raz” oraz „tamte treningi anuluj”. Zadanie 1814538 zatrzymano po czterech ramionach; raporty zachowano w `...\wyniki\dobor_klatek\trening\raporty\`.

### 3.9 Kampania 134 klatek: top-50 na kotwicę, zadanie 1814611 (commit `8474b310`)

**Metoda** (`eden\kontynuacja150.slurm`, wyniki `...\wyniki\dobor_klatek\trening150\`). Top-50 oznacza 50 klatek najbardziej podobnych do obrazu celu każdej kotwicy (`retrieval50_kotwica_XX_target.json`, ten sam indeks, wykluczenia i ranking; cosinus 0,66-0,81). Po usunięciu powtórzeń (4, 7 i 8 klatek wspólnych między parami kotwic): **134 unikalne klatki z 84 filmów**, 8,9 procent zbioru 1500. Kontrola: 134 losowe nieużyte klatki train (ziarno losowania 43, 74 filmy). Epoka = 103 aktualizacje (`epoch_size` 1648, bo próbnik danych wymaga rozmiaru epoki nie mniejszego od liczby klatek i podzielnego przez 16), jednakowo dla wszystkich ramion, także bez dodania. Reszta jak w 3.8. Zadanie **1814611** (nazwa kont245b), ziarna 42-44; ziarno 42 liczone na hopper. Użytkownik wkleił stan kolejki („tak ma być?”; ziarna 43 i 44 czekały, bo hopper miał jedną wolną kartę) i listę wolnych kart („licz na dgxach!”), więc oczekujące ziarna przeniesiono poleceniem `scontrol update` na partycję short (węzeł dgx-1). Każde ramię trwało 13-14 minut; skrypt sprzątał katalog roboczy `/dev/shm` pułapką przy wyjściu, na węzłach nic nie zostało. Raporty `trening150\raporty\seed{S}_{ramię}\`, logi `trening150\logi\`, podsumowanie `podsumuj_kontynuacje.py` → `...\wyniki\dobor_klatek\PODSUMOWANIE_KONTYNUACJA.json`, rysunek `KONTYNUACJA.png`.

mAP@0,5 na całym val (12 043 klatki) po jednej epoce od kroku 245:

| Ramię | ziarno 42 | ziarno 43 | ziarno 44 | średnia ± SD | mAP@0,5:0,95 (średnia ± SD) | F1 (średnia) |
|---|---|---|---|---|---|---|
| bez_dodania (1500) | 0,8678 | 0,8688 | 0,8694 | 0,8686 ± 0,0008 | 0,6175 ± 0,0035 | 0,8681 |
| koopman_fastgan (1634) | 0,8696 | 0,8695 | 0,8691 | 0,8694 ± 0,0002 | 0,6178 ± 0,0046 | 0,8697 |
| losowe (1634) | 0,8684 | 0,8698 | 0,8706 | 0,8696 ± 0,0011 | 0,6178 ± 0,0039 | 0,8689 |

Różnice sparowane względem braku dodania, mAP@0,5: dobrane +0,0018, +0,0007, -0,0003 (średnio +0,0007); losowe +0,0006, +0,0010, +0,0012 (średnio +0,0009). mAP@0,5:0,95: obie +0,0003. F1: dobrane +0,0016, losowe +0,0008. SD oznacza odchylenie standardowe po trzech ziarnach.

**Werdykt.** Dołożenie 134 klatek nie pogarsza detekcji i podnosi mAP@0,5 o około 0,001, tyle samo dla klatek dobranych i losowych, w granicach rozrzutu ziaren. Dobór Koopman → FastGAN → wyszukiwarka nie jest lepszy od losowego. Przyczyna znana z 3.8: zapytania do wyszukiwarki niosły tylko „podobieństwo do trzech kotwic”, nie kierunek zmiany. Jedna epoka od kroku 245 daje około 0,869 w każdym ramieniu, czyli zwykły postęp krzywej uczenia. Test nie rozstrzyga efektu lokalnego, wielu epok ani większych porcji; do rozróżnienia różnic 0,001 przy SD 0,001 potrzeba około dziesięciu ziaren na ramię.

### 3.10 Efekt lokalny na klatkach val podobnych do kotwic (commit `84747b2d`)

**Skąd.** 20:42: „policz efekt lokalny na klatkach val podobn[ych]”, czyli: może dobrane klatki pomagają tam, gdzie celowano, a miara na 12 043 klatkach tego nie widzi.

**Metoda** (`efekt_lokalny.py`, lokalnie na RTX 3070, wyniki `...\wyniki\dobor_klatek\efekt_lokalny\`, wagi `FINAL.pth` dziewięciu ramion pobrane z Edena do `trening150\wagi\`, 1,1 GB poza gitem):

1. Podobieństwo: ten sam DINOv2 i to samo przygotowanie wycinków co w indeksie; każda klatka val dostaje maksimum cosinusa do trzech zapytań `target.png` (`PODOBIENSTWO_VAL.npz`). Mediana 0,58; 271 klatek ma cosinus ≥ 0,7; próg top-134 to 0,717. Zapytania `target` i `current` tej samej kotwicy mają między sobą cosinus 0,99-1,00.
2. Predykcje: dziesięć zestawów wag (9 ramion + start `model_0000245.pth`) na wszystkich klatkach val, próg pewności 0,05 (`PREDYKCJE_{nazwa}.npz`). AP w stylu COCO (101 punktów krzywej) liczone lokalnie dla dowolnego podzbioru. Kontrola: na całym val lokalne AP@0,5 = 0,8659 wobec 0,8686 z Edena; różnica 0,003 wynika z odcięcia predykcji poniżej 0,05 i szczegółów implementacji, jest jednakowa dla ramion, więc różnice sparowane są wiarygodne.
3. Podzbiory: top-134, cosinus ≥ 0,7 (271), 10 procent najbliższych (1204), top-50 każdej kotwicy, dolna połowa (6022, kontrola), decyle.

**Wyniki.** Najpierw o samych klatkach: **okolica kotwic jest dla detektora trudna**. AP@0,5 ramienia bez dodania: cały val 0,866, 10 procent najbliższych 0,828, cosinus ≥ 0,7: 0,688, top-134: 0,545; top-50 kotwicy TR_0134 tylko 0,383 (TR_0006: 0,841, TR_0082: 0,803). Po decylach: 0,83 w najbliższym, 0,90-0,93 w środkowych, 0,65 w najdalszym. Start (krok 245) miał na top-134 0,526, więc epoka podnosi ten podzbiór o 0,02 w każdym ramieniu.

Δ AP@0,5 względem braku dodania (średnia po 3 ziarnach ± SD różnic; `EFEKT_LOKALNY.json`, `EFEKT_LOKALNY.png`):

| Podzbiór val (n) | dobrane - bez dodania | losowe - bez dodania |
|---|---|---|
| wszystkie (12 043) | +0,0003 ± 0,0003 | +0,0006 ± 0,0006 |
| dolna połowa (6 022) | +0,0002 ± 0,0006 | -0,0004 ± 0,0011 |
| 10 procent najbliższych (1 204) | -0,0020 ± 0,0039 | +0,0027 ± 0,0043 |
| cosinus ≥ 0,7 (271) | -0,0033 ± 0,0067 | +0,0035 ± 0,0061 |
| top-134 | -0,0028 ± 0,0085 | +0,0054 ± 0,0070 |
| suma top-50 kotwic (138) | -0,0067 ± 0,0113 | +0,0035 ± 0,0064 |
| top-50 TR_0006 | -0,0094 ± 0,0132 | +0,0009 ± 0,0011 |
| top-50 TR_0082 | -0,0117 ± 0,0063 | -0,0067 ± 0,0049 |
| top-50 TR_0134 | +0,0043 ± 0,0245 | +0,0139 ± 0,0168 |

Decyle: dobrane -0,002 ± 0,004 w najbliższym i 0 do +0,002 w pozostałych; losowe +0,003 ± 0,004 i 0 do +0,003. Na top-134 mAP@0,5:0,95 i F1: dobrane -0,001 / -0,001, losowe +0,004 / +0,002.

**Werdykt.** Efektu lokalnego nie ma: dobrane klatki dają na top-134 -0,003, losowe +0,005, oba w szumie ziaren (SD 0,007-0,009 na 134 klatkach), a znak dobranych jest ujemny w trzech z czterech podzbiorów. Jedyne, co pomiar pokazał z zapasem: FastGAN i wyszukiwarka wskazały region, w którym detektor jest słaby (AP 0,38-0,55), ale 134 klatki przez jedną epokę go nie naprawiły. Do rozstrzygnięcia efektu 0,003 na 134 klatkach potrzeba kilkunastu ziaren albo kilkuset klatek val w tym regionie, których nie ma.

### 3.11 Czy dodane klatki były trudne; ramię „trudne” i plan kampanii 5 epok (commit `9f45551c`)

**Skąd.** 22:19 użytkownik zinterpretował słabość detektora przy kotwicach: „jeśli model na tych zdjęciach jest słaby i nie daje rady, to dobrze. To znaczy, że generator wygenerował przypadki, które będą mocno wzmacniać detektor. Trzeba więcej epok do treningu detektora!”.

**Sprawdzenie** (`trudnosc_dodanych.py`, wynik `...\efekt_lokalny\TRUDNOSC_DODANYCH.json`), AP@0,5 na samych klatkach uczących:

| Wagi | 134 dobrane | 134 losowe | 134 z widzianych 1500 (ziarno 7) |
|---|---|---|---|
| start, krok 245 | 0,857 | 0,878 | 0,868 |
| bez_dodania po epoce (klatek nie widział), 3 ziarna | 0,889-0,907 | 0,901-0,905 | 0,900-0,906 |
| koopman_fastgan po epoce (widział je raz) | 0,907-0,912 | 0,899-0,901 | 0,905-0,910 |

Dobrane klatki treningowe nie były trudne: przed treningiem tylko o 0,02 gorsze od losowych, a po epoce bez oglądania ich detektor i tak ma na nich około 0,90. Trudna jest okolica kotwic w walidacji, skupiona w kilku filmach: 134 najbliższe klatki val pochodzą z 10 filmów, 101 z czterech (TR_0100, TR_0002, TR_0060, TR_0056); cały film TR_0060 ma AP@0,5 0,31, TR_0147 0,49, TR_0002 0,73, reszta 0,74-1,00.

**Ramię kontrolne „trudne”** (`dobor_trudnych.py`, `podglad_trudnych.py`, selekcja `...\trening5ep\selekcja_trudne.json`), sprawdza tezę użytkownika bez udziału generatora: 134 nieużyte klatki train, na których detektor z kroku 245 najmniej trafia. Trudność = 1 - maks. (pewność × IoU z prawdziwą ramką), po puli 9136 klatek. Pierwsza wersja z rundami po filmach rozmyła trudność (minimum 0,487, około mediany puli 0,48); ostatecznie najwyżej 2 klatki na film: 83 filmy, trudność 0,945-1,000. Wspólne z dobranymi 0, z losowymi 1. Podgląd 24 klatek (`TRUDNE_PODGLAD.png`): narzędzie przy krawędzi kadru, malutki czubek, detektor zaznacza inne narzędzie (pęsetę, rozwieracz); część to przypadki graniczne oznaczeń, więc ramię może nieść szum etykiet.

**Plan kampanii 5 epok** (zapisany w `PLAN.md` przed wynikiem; pliki wysłane i sprawdzone na Edenie, ale zadań do czasu zgody użytkownika nie zgłoszono): start `model_0000245.pth`, świeży optymalizator, 5 epok × 103 = 515 aktualizacji (koniec około kroku 760 oryginalnego biegu, gdzie jego krzywa dochodzi do 0,896-0,898), walidacja całego val i zapis wag oraz latentu po każdej epoce; ramiona bez_dodania, koopman_fastgan, losowe (te same selekcje co w 3.9) i opcjonalnie trudne; ziarna 42-44; koszt około 45 minut na bieg, 9-12 godzin GPU, około 9 GB na Edenie. **Kryterium ustalone przed wynikiem:** ramię X jest lepsze od losowego, gdy X - losowe > 0 we wszystkich trzech ziarnach na top-134 val i gdy na całym val X - bez_dodania ≥ -0,002. Przy SD ziaren 0,008 na top-134 trzy ziarna rozróżnią dopiero różnice rzędu 0,015. Na pytanie użytkownika „to trzeba będzie porównywać z losowymi? to też zaplanowałeś?” odpowiedź: tak. O 23:00 zgoda: „tak, puść wszystkie 12 biegów” (i prośba o wyłączenie komputera za 3 godziny, na 02:00).

### 3.12 Kampania 5 epok, zadanie 1814835 (commit `338a40d0`)

**Metoda.** `sbatch --array=0-11 kontynuacja5ep.slurm` (zadanie **1814835**, nazwa kont5ep, partycje short i hopper, limit 2:30), start około 23:00; wszystkie 12 biegów naraz (zadania 4-9 na hopper, 0-1 na dgx-3, 2-3 na dgx-1, 10-11 na dgx-4), 47-50 minut każdy. Katalog Eden `~/smallvit_runs/2026-09-25_kontynuacja_5epok/`. Wagi z końca każdej epoki pobrane (60 × 120 MB, `eden\pobierz5ep.sh`), predykcje lokalne (`efekt_lokalny_5ep.py`): epoka 5 na całym val, epoki 1-4 na 10 procentach val najbliższych kotwicom. Wyniki `...\wyniki\dobor_klatek\trening5ep\` (`efekt_lokalny\EFEKT_5EP.json`, `EFEKT_5EP.png`, logi `logi\slurm-1814835_*.out`). Zapisy latentu sond z każdej epoki (`latent\`, 139 MB) pobrano dla etapu 2 równania Langevina (3.15).

Cały val (Eden), mAP@0,5 po 5 epokach:

| Ramię | 42 | 43 | 44 | średnia ± SD | Δ wobec bez_dodania (sparowane: 42, 43, 44) | Δ mAP@0,5:0,95 |
|---|---|---|---|---|---|---|
| bez_dodania | 0,8989 | 0,8958 | 0,8944 | 0,8964 ± 0,0023 | - | - |
| koopman_fastgan | 0,8995 | 0,8954 | 0,8927 | 0,8959 ± 0,0034 | -0,0005 (+0,0006, -0,0004, -0,0017) | -0,0005 |
| losowe | 0,9002 | 0,8977 | 0,8944 | 0,8974 ± 0,0029 | +0,0010 (+0,0013, +0,0019, 0,0000) | -0,0014 |
| trudne | 0,8981 | 0,8982 | 0,8977 | 0,8980 ± 0,0003 | +0,0016 (-0,0009, +0,0025, +0,0033) | -0,0068 |

F1 średnio: 0,9053 / 0,9046 / 0,9049 / 0,8999.

Okolica kotwic (AP@0,5 lokalnie, 134 klatki val najbliższe kotwicom; start 0,526):

| Ramię | po epokach 1-5 (średnia 3 ziaren) | Δ top-134 wobec bez_dodania | Δ top-134 wobec losowych | Δ dolna połowa val wobec bez_dodania |
|---|---|---|---|---|
| bez_dodania | 0,546; 0,565; 0,586; 0,588; 0,590 | - | -0,009 | - |
| koopman_fastgan | 0,539; 0,569; 0,579; 0,585; 0,595 | +0,005 (+0,016, +0,012, -0,013) | -0,004 (+0,006, -0,005, -0,013) | +0,001 |
| losowe | 0,549; 0,562; 0,578; 0,590; 0,600 | +0,009 (+0,011, +0,017, +0,001) | - | +0,003 |
| trudne | 0,544; 0,553; 0,561; 0,564; 0,582 | -0,009 (-0,006, -0,002, -0,018) | -0,018 (-0,016, -0,019, -0,019) | +0,009 (+0,004, +0,009, +0,013) |

**Kryterium niespełnione** dla koopman_fastgan (top-134 wobec losowych +0,006, -0,005, -0,013) i dla trudnych (-0,016, -0,019, -0,019).

**Interpretacja.** Więcej epok pomaga w okolicy kotwic wszystkim ramionom jednakowo (AP top-134 z 0,53 do 0,59-0,60): to efekt samego uczenia, nie dodanych klatek. Globalnie krzywe dochodzą do 0,896-0,898, jak w biegu 1810662. Klatki dobrane przez Koopmana i FastGAN nie odróżniają się od losowych ani globalnie, ani lokalnie. Losowe dają jedyny zgodny we wszystkich ziarnach, choć mały, zysk lokalny (+0,009). Trudne działają odwrotnie, niż zakładała hipoteza użytkownika: pomagają daleko od kotwic (dolna połowa val +0,009 w każdym ziarnie), szkodzą przy kotwicach (-0,018 wobec losowych w każdym ziarnie) i pogarszają dokładność położenia ramki (mAP@0,5:0,95 -0,007, F1 -0,005). Na pytanie rano 26.09 („i jak wyniki?”) użytkownik usłyszał: wyniki są negatywne.

### 3.13 Ocena nowości metody i szans publikacji (`NOWOSC_I_PUBLIKACJA.md`)

**Skąd.** 23:39: „Zrób analizę jak bardzo to jest nowoczesna metoda i jak bardzo to się nadaje na publikację. Szukaj podobnych metod szukania nowych klatek w artykułach naukowych i rób /paper-search”.

**Metoda.** Undermind (płatne wyszukiwanie literatury), workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy” (`40294fa6-c9a3-4469-9a61-d3eddeaee893`): nowe głębokie przeszukanie „generative model synthesizes target image then retrieves real unlabeled samples for training” (103 prace) i trzy wcześniejsze. Pełne teksty: ASAL, DCD, LTG, CLIP-aided diffusion, ATTIC, TRACE, Swift Sampling. Scite: brak not o wycofaniu. Plik: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_tozsamosc_kierunki\NOWOSC_I_PUBLIKACJA.md`.

**Wynik.** Schemat „wygeneruj obraz-zapytanie, potem znajdź najbliższe prawdziwe klatki w puli” nie jest nowy (od 2012-2015; wersja z GAN-em: ASAL, Mayer i Timofte, WACV 2020, doi 10.1109/WACV45572.2020.9093556; wyszukiwanie cechami DINO: Data-Centric Debugging, Singla i in., WACV 2024, doi 10.1109/WACV57701.2024.00014). Nowy jest tylko cel generacji: przyszły stan reprezentacji detektora przewidziany Koopmanem z punktów kontrolnych uczenia; takiego połączenia nie znalazło żadne z czterech przeszukań. Najbliższe prace (tabela 11 pozycji w pliku) obejmują też Jain i in. 2022 (kierunek błędów w przestrzeni CLIP), Longtail-Guided Diffusion (ICML 2025), prace o wyższości prawdziwych obrazów nad syntetycznymi (Geng i in. 2024, Burg i in. 2023), TiDAL i TRACE (dobór z dynamiki uczenia, ale zmierzonej, nie przewidzianej), prace o active learning w wideo chirurgicznym (Peng i in. 2024, Yang i in. 2024, StepAL 2025) i o rzetelnej ocenie active learning (Mittal i in. 2019, Gashi i in. 2024). Porażka ASAL na CelebA-Eyeglasses ma ten sam mechanizm co tu: cechy całej sceny gubią mały szczegół (tam okulary, tu końcówkę narzędzia).

Werdykt: na pracę metodyczną to za mało, bo obraz przewidzianego latentu nie różni się od obecnego, a dobrane klatki nie wygrywają z losowymi. Ścieżki: A. wynik negatywny z diagnozą (warsztat MICCAI albo „I Can't Believe It's Not Better” przy NeurIPS); B. osobna praca o dynamice reprezentacji detektora (sekcje 3.4-3.5); C. metoda na główną konferencję wymagałaby celu o dużej amplitudzie, dopasowania w cechach detektora wokół narzędzia zamiast DINOv2 z całej klatki, ablacji przy równym budżecie (sąsiedzi kotwicy bez Koopmana i generatora, wyszukiwanie od klatek z błędami, niepewność, trudne, losowe), kilku budżetów, co najmniej 3 ziaren i efektu około 0,015 AP@0,5 na top-134.

### 3.14 Teoria: model Langevina i twierdzenie o stabilności wyboru (`TEORIA_LANGEVIN.md`)

**Skąd.** 23:51: „Liczę, że te wyniki będą pozytywne. Trzeba to zamknąć w jakąś teorię i udowodnić, jak nie [Koopmanem], to [równaniem] Langevina”. 26.09 o 16:57 dopytanie: „To jak to wszystko formalizować?”.

**Treść** (`...\koopman_tozsamosc_kierunki\TEORIA_LANGEVIN.md`, commit `338a40d0`):

1. **Model.** Latent tożsamości sondy po aktualizacji t spełnia równanie Langevina z liniowym dryfem (dyskretny proces Ornsteina-Uhlenbecka): z_{t+1} - z* = A (z_t - z*) + σ ξ_t, gdzie A to część deterministyczna (przybliżana Koopmanem), σ ξ_t to szum losowych porcji danych, z* to stan docelowy. Prognoza na h kroków przesuwa się o d_h = -(I - A^h)(z_t - z*). Pomiar: od kroku 245 ‖d_100‖ = 6 procent rozrzutu, do końca uczenia 13 procent.
2. **Twierdzenie 1 (stabilność wyboru).** Wyszukiwarka ocenia klatkę x_i przez s_i(q) = ⟨q, x_i⟩ (q to znormalizowany wektor cech obrazu z generatora). Jeśli przesunięcie zapytania δ = ‖q' - q‖ jest mniejsze niż połowa przerwy między k-tą a (k+1)-szą oceną, to zbiór top-k się nie zmienia. Dowód z nierówności Cauchy'ego-Schwarza: |s_i(q') - s_i(q)| ≤ δ dla wektorów jednostkowych. Wniosek 1: klatka może zmienić przynależność do wyboru tylko, gdy jej ocena leży w pasie [s_(k) - 2δ, s_(k+1) + 2δ]. Wniosek 2: przy małym δ ramię „dobrane” jest w praktyce ramieniem „sąsiedzi obrazu obecnego latentu kotwicy”.
3. **Strona wag.** Uczenie SGD to też równanie Langevina; dodanie zbioru S zmienia stratę na obszarze docelowym R w pierwszym rzędzie o ΔL_R ≈ -η ⟨∇L_R, ∇L_S⟩ (iloczyn gradientów). Łatwe klatki mają mały gradient i prawie nic nie zmieniają; trudne pomagają tylko tam, gdzie ich błędy są podobne do błędów w R.
4. **Konfrontacja z danymi:** małe d_h po kroku 245 → obrazy prawie równe (tak, 0,1-0,9 procent jasności); małe δ → ten sam wybór (tak, 54/60 wspólnych); sąsiedzi kotwicy łatwi → mały efekt (tak, 0,857 na starcie, +0,0007); trudne → znak zależny od zgodności gradientów (tak, +0,009 daleko, -0,018 przy kotwicach).
5. **Przewidywania do sprawdzenia:** P1: wyszukiwanie po samym obrazie obecnym kotwicy da zbiór wspólny w ≥ 90 procentach i ten sam efekt; P2: prognoza zmieni wybór dopiero przy dużym ‖d_h‖ (wcześnie w uczeniu albo przy celu o dużej amplitudzie); P3: cel zbudowany z kierunku straty pobije losowe na top-134, cel z prognozy dryfu po kroku 245 nie.
6. **Koopman czy Langevin:** Koopman to tylko część A; Langevin dodaje szum i daje właściwą miarę, czyli stosunek ‖d_h‖ do szumu prognozy i do rozdzielczości wyszukiwarki. Porażka nie wynika z tego, że operator liniowy źle przewiduje, tylko z tego, że przewidywana zmiana jest za mała, by zmienić wybór.

Na pytanie o formalizację (26.09) odpowiedź zaproponowała trzy warstwy: równanie jako definicję i trzy twierdzenia, każde z pomiarem. Pierwsze (błąd prognozy = szum nie do przewidzenia + brakujący człon równania) jest według tej odpowiedzi już dowiedzione w Lean (programie sprawdzającym dowody krok po kroku) wśród 13 istniejących twierdzeń projektu grantowego; drugie (stabilność wyboru) i trzecie (dodane klatki pomagają, gdy ich błędy przypominają błędy w danym miejscu) są proste do dopisania. W tym etapie ich nie sformalizowano.

### 3.15 Równanie Langevina użytkownika dla latentu detektora (`langevin_latent_detektora`, etap 1)

**Skąd.** 23:53: „Ma być tak, że Koopman przewiduje latent i tyle, chcę, żeby robił to dobrze. Bierz ten mój [równanie Langevina] z [projektu], co ma człon dryf, pamięć, wymuszenie, sterowanie i szum, i to dostosuj. Dryf [można] liczyć Koopmanem”.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\langevin_latent_detektora\`: `PLAN.md` z bramką przed wynikiem, `langevin.py`, `WYNIKI.md`, `WYNIKI.json`, `wyniki\REPORT_S48.json`, `REPORT_S96.json`, commit `338a40d0`). Źródło równania: `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\schemat-diagnostyki-latentu-koopman-langevin-2026-09-18.md` (części A3 i C). Cząstka to skalar tożsamości jednej sondy, krok = 5 aktualizacji:

v̂_{τ+1|τ} = D(a_τ, v_τ) + JP(v_{τ-1}, ..., v_{τ-L+1}) + W(f_τ) + S(u_τ) + N(r_τ, ..., r_{τ-q+1}),  a_{τ+1} = a_τ + v_{τ+1}.

D to dryf: Koopman z wejściem na stanie [v_τ, a_τ - ā_τ, 1]; JP to jądro pamięci (starsze przyrosty, L - 1 opóźnień, człon Mori-Zwanziga); W to wymuszenie znanym harmonogramem (granica epoki, czyli przetasowanie danych co 100 aktualizacji, faza epoki, 1/krok); S to sterowanie (skład danych uczących), w biegu 1810662 stałe, więc nieidentyfikowalne; N to szum kolorowy (model AR(q) na dawnych resztach). Trzy fazy: uczenie parametrów na przejściach do S na 48 sondach fit, prognoza z informacji w chwili τ, obserwacja (reszta aktualizuje bufor N, nigdy nie poprawia prognozy wstecz). Protokół jak w 3.5 (K = 160, okna S = 48 i S = 96), wybór L, grzbietu, q i członów na kalibracji (ostatnie 20 przejść okna fit), test na 24 sondach holdout. **Bramka:** równanie lepsze od L4, gdy rollout h = 20 ≤ 0,95 × L4 w obu oknach i RMSE jednego kroku nie gorsze od L4 o więcej niż 2 procent.

**Wybór na kalibracji:** w obu oknach długa pamięć, 8 ostatnich przyrostów (40 aktualizacji); członu położenia w dryfie i wymuszenia nie wybrano; szum AR(2) tylko w oknie od kroku 485.

| Okno | Model | 1 krok | h = 10 | h = 20 | h = 40 | do końca |
|---|---|---|---|---|---|---|
| od kroku 245 (dryf trwa) | Langevin: dryf Koopmana + pamięć 8 | 0,416 | 0,150 | 0,151 | 0,220 | 0,370 (h = 111) |
| | L4 zapisany | 0,407 | 0,152 | 0,171 | 0,254 | 0,417 |
| | sam dryf (bez pamięci) | 0,417 | 0,376 | 0,482 | 0,555 | 0,695 |
| od kroku 485 (dryf wygasł) | Langevin | 0,682 | 1,414 | 0,557 | 0,367 | 0,461 (h = 63) |
| | L4 zapisany | 0,682 | 1,232 | 0,454 | 0,435 | 0,578 |
| | sam dryf | 0,705 | 1,123 | 0,602 | 0,602 | 0,801 |

**Bramka niespełniona.** Od kroku 245 rollout h = 20 to 0,886 × L4 (warunek spełniony), ale 1 krok gorszy o 2,2 procent (dopuszczalne 2); od kroku 485 rollout h = 20 to 1,23 × L4. Wnioski: ruch latentu niesie pamięć (bez niej błąd na 20 kroków rośnie z 0,15 do 0,48); od kroku 245 reszta jednokrokowa ma autokorelację 0,27 przy opóźnieniu 1, czyli w równaniu brakuje jeszcze członu; od kroku 485 reszty są białe (autokorelacja -0,07 do -0,01), zostaje sam szum porcji danych; reszty od kroku 485 korelują z granicą epoki (-0,24), co wskazuje na dopracowanie członu W. Etap 2 (sterowanie S(u) na 12 biegach kampanii 5 epok) zaplanowano, w tym etapie go nie uruchomiono.

### 3.16 Pętla uczenia 8 obrotów na Nano (`petla_uczenia_koopman_fastgan`, pilot, ziarno 42, commit `12d3f023`)

**Skąd.** 26.09, 17:07: „No to spróbuj zrobić szukanie latentu, generowanie zdjęć [FastGAN-em], szukanie nowych zdjęć, dodawanie do zbioru uczącego, potem uczenie z nowymi zdjęciami i potem znowu [...] przez 8 kroków tej pętli. Zobaczymy, jak się wytrenuje sieć; jeśli się uda ładnie, to [...] uczenie od początku”. Plan rozpisano przed zgłoszeniem zadań; zgoda o 21:38: „No to wykonuj i potem analizuj”, z doprecyzowaniem celu: „Musi być tak, że dzięki temu wiemy, jak najszybciej uczyć sieć!” i „Może wysyłać na Eden na hopper albo dgx”.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_uczenia_koopman_fastgan\`: `PLAN.md`, `obrot_lokalny.py`, `dyrygent.py`, `eden\obrot.slurm`, `eden\kontynuacja_obrot.py`, `eden\config_seed42.json`, `analiza.py`, wyniki `wyniki\ANALIZA.json`, `ANALIZA.png`, `STAN.json`, `wyniki\obroty\kXX\RAPORT.json`). **Obrót pętli** to jeden cykl: prognoza latentu, obraz FastGAN, wyszukanie klatek, dopisanie ich do zbioru, jedna epoka uczenia na Edenie. Pierwsza wersja planu mówiła o obrocie = 103 aktualizacje i prognozie równaniem z `langevin.py`; przy pisaniu kodu, przed wynikiem, zapisano w `PLAN.md` realizację: obrót = 165 aktualizacji (`epoch_size` 2640, bo po 8 obrotach zbiór ma 1500 + 8 × 134 = 2572 klatki), jednakowo dla wszystkich pętli, start z wag końca poprzedniego obrotu (pierwszy: krok 245 biegu 1810662), świeży optymalizator, walidacja całego val; latent 192 sond zapisywany przez `trening.py` co 5 aktualizacji (33 zapisy na obrót), zgodny z lokalnymi tokenami do 10⁻⁶ względnie. Prognoza: wspólny operator na 8 ostatnich przyrostach (grzbiet λ = 0,1), osobno skalar i S_4, dopasowany do całej historii pętli, rollout średniej o 33 zapisy (cel: stan na koniec obrotu); kierunki stałe z `OPERATORY_K73_S48`. FastGAN lokalnie na RTX 3070 (inwersja kotwicy z ciepłym startem z poprzedniego obrotu, krok Gaussa-Newtona), kotwice stałe 00, 24, 48. Wyszukanie: top-45 na kotwicę bez klatek już użytych, około 134 nowe klatki na obrót. Pętle: **koopman**, **losowe** (134 losowe nowe klatki na obrót), **bez** (tylko 1500 klatek bazy). Lokalny „dyrygent” sam prowadził kolejne obroty, wznawiając po przerwach. Zadania Eden (każde to tablica trzech pętli): 1816922, 1816964, 1816992, 1817023, 1817043, 1817091, 1817171, 1817225; pierwsze zgłoszone 26.09 o 21:58, koniec 27.09 o 01:33. Kryterium (przed wynikiem): po 8 obrotach pętla koopman wygrywa z losową na top-134 val i traci na całym val nie więcej niż 0,002 wobec pętli bez.

mAP@0,5 całego val po obrotach (Eden):

| Obrót | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | pole pod krzywą |
|---|---|---|---|---|---|---|---|---|---|
| koopman | 0,8787 | 0,8902 | 0,8962 | 0,9029 | 0,9022 | 0,9048 | 0,9047 | 0,9055 | 0,8981 |
| losowe | 0,8803 | 0,8902 | 0,8953 | 0,8999 | 0,9020 | 0,9045 | 0,9017 | 0,9045 | 0,8973 |
| bez | 0,8811 | 0,8909 | 0,8961 | 0,9003 | 0,9032 | 0,9052 | 0,9073 | 0,9072 | 0,8989 |

Pole pod krzywą to średnia mAP po obrotach, miara tempa uczenia. Tempo (pierwszy obrót z mAP@0,5 ≥ progu): 0,89 wszystkie w obrocie 2; 0,895 wszystkie w 3; 0,90 koopman i bez w 4, losowe w 5. Liczba klatek pętli z dodawaniem: 1634, 1768, ..., 2572.

AP@0,5 na 134 klatkach val najbliższych kotwicom (lokalnie; start 0,526):

| Obrót | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| koopman | 0,553 | 0,554 | 0,570 | 0,583 | 0,599 | 0,591 | 0,594 | 0,593 |
| losowe | 0,557 | 0,576 | 0,578 | 0,584 | 0,600 | 0,582 | 0,579 | 0,563 |
| bez | 0,576 | 0,587 | 0,601 | 0,604 | 0,609 | 0,598 | 0,615 | 0,627 |

Po 8 obrotach mAP@0,5:0,95: koopman 0,6756, losowe 0,6693, bez 0,6665; F1: 0,9158, 0,9123, 0,9146.

**Kryterium formalnie spełnione na jednym ziarnie** (koopman - losowe na top-134: +0,030; koopman - bez na całym val: -0,0017), ale odczyt jest inny niż „Koopman przyspiesza”. Najlepsza jest pętla bez dodawania, globalnie (0,907) i przy kotwicach (0,627); pole pod krzywą: bez > koopman > losowe. Przewaga Koopmana nad losowymi przy kotwicach to „mniej szkodzi”: losowe spadają z 0,600 do 0,563 w obrotach 5-8, koopman trzyma około 0,59. Wyjątek: dokładność położenia ramki najwyższa w pętli koopman (mAP@0,5:0,95 +0,009 wobec bez). Prawdopodobny mechanizm szkody dodawania: stała długość obrotu przy rosnącym zbiorze rozcieńcza powtórzenia klatek bazy (z 1,76 do 1,03 na obrót). Jedno ziarno: różnice globalne rzędu 0,002-0,003 są w szumie.

**Prognoza.** Zmiana skalara tożsamości w obrocie 1: przewidziana 0,127 rozrzutu, zmierzona 0,111; w obrocie 2: 0,062 wobec 0,025; od obrotu 3 zmierzona około 0 (0,007; 0,002; 0,002; 0,000; -0,002). Prognoza obrotu 1, sprawdzona po jego zakończeniu, miała w pięciu śledzonych wymiarach kodu błąd 0,64 błędu braku zmian. Obraz celu różnił się od obecnego o 0,0001-0,008 jasności (L1). Wspólne klatki top-45 dla obrazu celu i obecnego: 30-45, średnio 40 z 45 (89 procent), zgodnie z twierdzeniem 1. Dryf tożsamości gaśnie w pierwszych 2 obrotach (około kroku 550).

**Diagnoza użytkownika (26.09, 23:28):** „my znajdujemy te [klatki], które najlepiej będą budować ten przyszły przewidywany latent, więc on do nauki nic nie wnosi [...] spróbuj znaleźć taki sposób, żeby wygenerować przez [FastGAN] takie przypadki, które jeszcze bardziej przyspieszą uczenie tego latentu”. Odpowiedź: diagnoza jest trafna; propozycja pętli „przyspieszonej” (cel 5 obrotów naprzód, co obrót 3 sondy o największej przewidzianej drodze, czyli najbardziej niedouczone, kryterium: dojście do 0,895 wcześniej niż bez i losowe albo 1,2 razy dłuższa droga tożsamości w obrotach 1-4) trafiła do `PLAN.md` i czekała na zgodę; w tej postaci nigdy jej nie uruchomiono. Po pilocie dopisano, że ma sens dopiero od początku uczenia, bo od kroku 245 latent już stoi.

### 3.17 Analiza: od początku dostrajania i większy model (`ANALIZA_WCZESNIE_I_SKALA.md`, commit `6682ee16`)

**Skąd.** 27.09, 10:29: „A robiłeś trening od zera? W sensie dopiero po na przykład dwóch epokach przewidujesz mniej więcej, w którą stronę będzie wskazywał [...] albo może weźmiesz większą sieć, na przykład RF-DETR [...] medium? Wydaje mi się, że ta metoda by się bardziej sprawdziła w większych modelach [...] przy pierwszych epokach, na przykład pięciu”. Zaraz potem: „Zrób analizę, możesz szukać w internecie, możesz skorzystać z paper search”.

**Wynik** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_uczenia_koopman_fastgan\ANALIZA_WCZESNIE_I_SKALA.md`; Undermind, przeszukanie „when does data selection accelerate training early phase and model size”, 190 prac, pełne teksty 9 prac):

- Pilot nie startował od zera, lecz po 2,45 epoki dostrajania od wag COCO, czyli prawie tam, gdzie proponuje użytkownik. W tym detektorze latent przestaje się ruszać około piątej epoki, więc pętla trafiła w końcówkę ruchu i przez 6 z 8 obrotów Koopman nie miał czego przewidywać.
- Literatura popiera start wcześnie: krytyczne okresy uczenia (Achille i in., arXiv 1711.08856; ingerencja w pierwszych 15-25 procentach uczenia działa trwale), wczesna faza (Frankle i in., arXiv 2002.10365), RHO-LOSS (Mindermann i in., ICML 2022, doi 10.48550/arXiv.2206.07137: wybór przykładów „do nauczenia, wartych nauczenia, jeszcze nienauczonych” przyspiesza uczenie 2-18 razy, najmocniej na początku). Ale literatura nie pokazuje, by dobór bił losowanie mocniej przy wczesnym starcie w dostrajaniu (Wu i in., ICLR 2021: kolejność danych daje zwykle znikomy zysk poza krótkim budżetem czasu i zaszumionymi danymi; Okanovic i in. 2023: losowanie nowego podzbioru co epokę to bardzo mocna kontrola).
- Większy model: za tym Bad Students Make Great Teachers (Evans i in., doi 10.48550/arXiv.2312.05328: 26-46 procent mniej aktualizacji dla ViT-L); przeciw: Gupte i in. (doi 10.48550/arXiv.2401.14555, zysk szybko znika), lazy training (Chizat i in., arXiv 1812.07956: szersze modele mniej ruszają cechami), Sorscher i in. (przy małych danych lepiej łatwe przykłady), Hu i in. (dobór po trudności zawodzi przy szumie etykiet).
- RF-DETR w rfdetr 1.10.1: Nano, Small, Base, Medium i Large mają ten sam kręgosłup DINOv2-S (ViT-S) i hidden_dim 256; różnią się rozdzielczością (384, 512, 560, 576, 704) i liczbą warstw dekodera (2-4). Medium to test wyższej rozdzielczości, nie większego modelu.
- Rekomendacja: pętla od początku dostrajania na Nano, 3 ziarna, stała liczba powtórzeń bazy, wybór w stylu RHO spośród klatek podobnych do obrazu FastGAN; Medium potem. Przewidywanie: jeśli metoda działa, przewaga pojawi się w pierwszych 2-3 obrotach i zniknie, gdy dryf zgaśnie.

Na pytanie „A jest RF-DETR jeszcze większy niż medium?” (13:44) odpowiedź: są X-Large i 2X-Large z kręgosłupem DINOv2-B (około cztery razy więcej parametrów), ale w dodatku `rfdetr[plus]` na licencji Platform Model License, z kontem Roboflow; w zainstalowanym pakiecie jest wycofany wariant `LargeDeprecated` z DINOv2-B i obrazem 560 pikseli, bez dodatkowej licencji. Decyzja użytkownika o 14:47: „To weź ten z DINOv2-B LargeDeprecated. Zrób”.

### 3.18 Pętla od początku dostrajania na RF-DETR LargeDeprecated (`petla_od_poczatku_large`, commity `7190b27b`, `e5fecc11`, `ebd9db38`)

**Model.** `RFDETRLargeDeprecated` z rfdetr 1.10.1 (LargeDeprecated: starsza, wycofana wersja „Large”, zachowana w pakiecie): kręgosłup DINOv2-B (`dinov2_windowed_base`, większa odmiana DINOv2), obraz 560 pikseli, łatka 14 pikseli, siatka 40 × 40 = 1600 tokenów po 768 liczb, hidden_dim 384, 3 warstwy dekodera (części sieci, która z cech robi ramki), 135 mln parametrów; wagi COCO `rf-detr-large.pth` (1,57 GB). Opis sondy ma tu 3072 liczby.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_od_poczatku_large\`: `PLAN.md` z kryterium przed wynikiem, kopie kodu przerobione w `kod\` (`trening.py`, `czastki.py`, `generowanie.py`, `deterministyka.py`, `fastgan_large.py`), `obrot_large.py`, `dyrygent_large.py`, `analiza_large.py`, `eden\obrot_large.slurm`, `eden\fastgan_large.slurm`, `eden\kontynuacja_obrot.py`, `eden\config_seed{42,43,44}.json`; wyniki `wyniki\`, katalog Eden `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-27_petla_large`).

- Obrót = 1 epoka = 2560 próbek = 160 aktualizacji (porcja 4 × 4), szybkości uczenia jak w biegu Nano 1810662. Próbnik ze stałą ekspozycją bazy: każda z 1500 klatek bazy raz na obrót, każda dodana raz, reszta losowe powtórzenia bazy. Latent sond co 5 aktualizacji (32 zapisy na obrót), wagi tylko na końcu obrotu (0,5 GB na zapis).
- **Obrót 0 (rozgrzewka):** od wag COCO na 1500 klatkach bazy, osobno dla ziaren 42, 43, 44; daje historię latentu do dopasowania prognozy.
- **Kierunki tożsamości Large** (etap `kierunki`, z latentu na końcu rozgrzewki): w = znormalizowana średnia (narzędzie - tło) sond fit (tu nie LDA), S_4 = 4 główne kierunki (narzędzie - tło) po usunięciu w; rzut 3072 → 5.
- **Obroty 1-6, trzy pętle na ziarno:** koopman, losowe (+134 losowe nowe klatki na obrót), bez. Pętla koopman: prognoza wspólnym operatorem na 8 ostatnich przyrostach (grzbiet λ = 0,1 na standaryzowanych cechach, osobno skalar i S_4), rollout każdej sondy o 3 obroty naprzód (96 zapisów); kotwice to 3 oryginały (z 12) o największej przewidzianej drodze, czyli „najbardziej niedouczone”. FastGAN na Edenie (odwracanie przez DINOv2-B jest za wolne na RTX 3070): odczyt obecnego kodu, cel = obecny + przyrost, inwersja (6 restartów albo 2 plus ciepły start z poprzedniego z*), krok Gaussa-Newtona do celu i kontrolnie do obecnego kodu. Wybór lokalnie: DINOv2, top-90 na kotwicę dla obrazu celu (i kontrolnie obecnego) jako kandydaci, ocena w stylu RHO-LOSS = trudność dla bieżącego modelu Large minus trudność dla modelu odniesienia (Nano 1810662 po 800 aktualizacjach, który tych klatek nie widział); 134 najwyższe.
- **Kryterium (przed wynikiem):** pętla koopman ma większe pole pod krzywą mAP@0,5 (obroty 1-6) niż losowe i bez w każdym z 3 ziaren, a w obrotach 1-3 jest średnio co najmniej o 0,005 wyżej niż bez. Zatrzymanie, gdyby obrót trwał dłużej niż około 1,5 godziny GPU.

**Przebieg i naprawy.**

- Rozgrzewka, próba 1 (zadania 1817331-1817333): „upsample_bicubic2d_aa_backward_out_cuda does not have a deterministic implementation”; naprawa: tryb deterministyczny z ostrzeżeniem zamiast błędu (`determinizm_ostrzezenie`). Próba 2 (1817334-1817336): przy zapisie po walidacji rfdetr odrzucał wycofaną konfigurację („Deprecated model config 'RFDETRLargeDeprecatedConfig' is no longer supported”, `BestModelCallback`); naprawa: usunięcie tego wywołania zwrotnego, wagi i tak zapisywane są osobno. Próba 3: zadania 1817340-1817342, 49 minut na zadanie. Rozgrzewka dała mAP@0,5 0,8622 (s42), 0,8610 (s43), 0,8591 (s44); Nano po podobnym czasie miał około 0,57.
- Etap wyboru padał (s42 k01, kod 3): faiss-cpu i torch ładowały dwie kopie OpenMP („OMP Error #15”); naprawa `KMP_DUPLICATE_LIB_OK=TRUE`, sprawdzona: top-90 z faiss = top-90 z iloczynu skalarnego w numpy na 20 zapytaniach, różnica wyniku 8·10⁻⁷ (commit `e5fecc11`).
- Kwota Edenu: obrót 1 zapisywał na końcu każdego zadania 3 pliki pełnego stanu (.ckpt, po 1,57 GB) i `FINAL.pth`, kasowane dopiero po walidacji; przy 9 zadaniach naraz chwilowo zajmowało to około 30 GB, a wolne miejsce spadło z 38 do 4 GB. Kontrola kwoty w etapie prognozy zatrzymała dyrygenta ziarna 42 przed obrotem 2 (kod 1). Od obrotu 2 te zapisy wyłączono (`wagi_tylko_koniec`); następny obrót startuje z `model_0000160.pth`, stanu optymalizatora nikt nie czytał, przebieg uczenia bez zmian. Wolne miejsce przed obrotem 2: 33 GB, przed 5: 17,9 GB, przed 6: 12,8 GB, po kampanii około 7 GB.
- s43, pętla bez, obrót 6 (zadanie 1817673_2) padło w 0 s na węźle dgx-3 (`cp -a` do `/dev/shm`: „Numerical result out of range”, usterka węzła); ponowione jako **1817712** z wykluczeniem dgx-3.
- Zadania FastGAN (s42/s43/s44, obroty 1-6): 1817347/1817349/1817350, 1817431/1817433/1817434, 1817485/1817484/1817487, 1817552/1817553/1817560, 1817603/1817610/1817611, 1817667/1817666/1817668. Zadania uczenia: s42 1817351, 1817448, 1817502, 1817562, 1817614, 1817672; s43 1817356, 1817456, 1817499, 1817563, 1817627, 1817673 (bez: 1817712); s44 1817359, 1817458, 1817504, 1817565, 1817628, 1817676. Obrót trwał około 55 minut na zadanie, razem z FastGAN i wyborem około 65 minut; cała kampania około 9 godzin. Stan w `wyniki\s{42,43,44}\STAN.json`.

**Wyniki** (`WYNIKI.md`, `WYNIKI.json`, `wyniki\ANALIZA.json`, `ANALIZA.png`; `wyniki\SHA256.json` dla 2198 plików binarnych, 15,4 GB poza gitem). Stan na koniec etapu: 53 z 54 obrotów pętli (brak `bez` k06 ziarna 43).

mAP@0,5 na val (k00 = wspólna rozgrzewka; obroty 1-6):

| Ziarno | k00 | koopman | losowe | bez |
|---|---|---|---|---|
| 42 | 0,8622 | 0,8953; 0,9002; 0,8948; 0,9057; 0,9048; 0,9098 | 0,8976; 0,9029; 0,8982; 0,9078; 0,9092; 0,9164 | 0,8979; 0,8994; 0,8998; 0,9050; 0,9018; 0,9007 |
| 43 | 0,8610 | 0,8989; 0,9012; 0,9062; 0,9071; 0,9048; 0,9018 | 0,9039; 0,9045; 0,9148; 0,9115; 0,9049; 0,9051 | 0,9001; 0,9028; 0,9040; 0,9018; 0,9073; brak |
| 44 | 0,8591 | 0,8932; 0,8938; 0,8972; 0,9008; 0,9087; 0,9067 | 0,8936; 0,8900; 0,9043; 0,9071; 0,9066; 0,9061 | 0,8931; 0,8999; 0,9044; 0,9065; 0,9016; 0,8972 |

Pole pod krzywą mAP@0,5 (średnia po obrotach 1-6): koopman 0,9018 / 0,9033 / 0,9001; losowe 0,9053 / 0,9075 / 0,9013; bez 0,9008 / 0,9032 (obroty 1-5) / 0,9004. Losowe najwyżej w każdym ziarnie.

| Porównanie (różnice sparowane mAP@0,5) | Obroty 1-3 | Obroty 4-6 | Obroty 1-6 |
|---|---|---|---|
| koopman - bez | -0,0023 (SD 0,0032; 3/9 > 0) | +0,0033 (SD 0,0055; 6/8 > 0) | +0,0004 (SD 0,0052; 9/17 > 0) |
| koopman - losowe | -0,0032 (SD 0,0036; 1/9 > 0) | -0,0027 (SD 0,0031; 2/9 > 0) | -0,0030 (SD 0,0033; 3/18 > 0) |
| losowe - bez | +0,0009 (SD 0,0055; 5/9 > 0) | +0,0060 (SD 0,0057; 7/8 > 0) | +0,0033 (SD 0,0060; 12/17 > 0) |

Tempo (pierwszy obrót z mAP@0,5 ≥ 0,90, ziarna 42/43/44): koopman k2/k2/k4, losowe k2/k1/k3, bez k4/k1/k3.

**Werdykt: kryterium niespełnione w obu częściach.** Pętla koopman nie przyspiesza uczenia modelu Large; koopman - losowe jest ujemne w 15 z 18 par (średnio -0,0030 przy SD 0,0033), więc klatki wybrane pętlą uczą gorzej niż losowe. Dokładanie klatek zaczyna pomagać od obrotu 4 (losowe - bez +0,006), gdy dodanych jest ponad 500.

**Latent i prognoza.** Large porusza latentem znacznie mocniej niż Nano: w rozgrzewce skalar tożsamości wzrósł z 16,8 do 19,1 (0,38 rozrzutu w 160 aktualizacjach); przewidziana droga kotwic na 3 obroty naprzód w obrocie 1 to 1,19-1,67 rozrzutu (pilot Nano: 0,12), w obrocie 2 0,45-0,81, w obrotach 3-6 od 0,11 do 0,58. Ruch skalara na zapis spada z 0,013 (k00) do 0,002 rozrzutu (k03): dryf gaśnie po około dwóch obrotach. Obraz celu FastGAN różnił się od obecnego wyraźniej niż w pilocie (L1 do 0,046 wobec 0,002; w jednej kotwicy s42 k01 generator nie doszedł do celu, reszta 0,60 i L1 0,0010). Wspólne top-45 dla celu i obecnego: w obrotach 1-2 średnio 77-79 procent, w obrotach 4-6 około 90 procent (policzone z `wyniki\ANALIZA.json`; pilot Nano 89 procent). Ocena RHO wybranych klatek spada z około 0,21 w obrocie 1 do około 0 w obrotach 5-6 (bieżący model Large i model odniesienia oceniają je tak samo trudno), a średnia po wszystkich 187-259 kandydatach staje się ujemna od obrotu 2. Pętla koopman przesuwa skalar tożsamości trochę dalej niż pozostałe (k01: +0,03 do +0,10 wobec bez), bez zysku w mAP.

**Czego to nie dowodzi.** Nie dowodzi, że prognoza latentu jest bezużyteczna, tylko że ta zamiana prognozy na klatki (FastGAN, podobieństwo DINOv2, ocena „do nauczenia” według Nano) nie bije losowania na tym zbiorze i modelu. Losowe klatki są różnorodne, a wybór wokół trzech kotwic skupia 134 klatki na wąskim fragmencie danych; porównania z doborem różnorodnym nie robiono. Trzy ziarna, jeden zbiór, jeden model, val bez osobnego testu trudnych klatek. Na pytanie użytkownika o 20:53 („Czyli jak to teraz wygląda?”) odpowiedź po trzech obrotach: kryterium już nie zostanie spełnione, losowe klatki prawdopodobnie lepiej pokrywają dane. Pętlę prowadzono dalej zgodnie z planem do obrotu 6 (około 25 godzin GPU), z propozycją „stop” do decyzji użytkownika. Rozbiór przyczyn tej porażki powstał dopiero 28 września w południe, w następnym etapie (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_od_poczatku_large\DIAGNOZA_PORAZKI.md`, commit `5acea332`).

### 3.19 Domykanie latentu tożsamości: diagnoza i projekt (`domykanie_tozsamosci_large`, commit `e9055b2f`)

**Skąd.** 27.09, 21:14: „To można szukać jakby tych obrazów, które najszybciej domykają latent tożsamości. Tylko musisz dobrze określić warunki domykania i szybkiego uczenia latentu tożsamości!”.

**Diagnoza bez nowego GPU** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\domykanie_tozsamosci_large\diagnoza_domkniecia.py`, wynik `wyniki\DIAGNOZA.json`, liczony na latencie sond pętli Large: 3 ziarna × 3 pętle × końce obrotów; najpierw obroty 0-3, w commicie `ebd9db38` odświeżony do obrotu 6). Sondy: 12 obrazów narzędzia × 6 wariantów, 8 obrazów fit, 4 holdout. Latent tożsamości sondy: skalar a = w · (narzędzie - tło) i 5 liczb (w, S_4). Miary: W1 ruch a na 5 aktualizacji w jednostkach rozrzutu sond; W2 niezmienniczość, czyli R = rozrzut a między wariantami tego samego obrazu podzielony przez rozrzut średnich różnych obrazów, przesunięcia wariantów względem oryginału oraz „rozpoznanie” (odsetek sond, których najbliższy środek obrazu w 5 liczbach to ich własny obraz; przypadek 1/12 ≈ 0,08); W3 rozdzielenie, najmniejsze a po sondach (> 0: narzędzie odróżnione od tła).

| Warunek | Miara | Start (wagi COCO) | Koniec k00 | Koniec k03 | Wniosek |
|---|---|---|---|---|---|
| W1 zatrzymanie | ruch a na 5 aktualizacji / rozrzut sond | 0,007-0,010 | 0,013 | 0,002 | latent stoi od k02 |
| W2 niezmienniczość | R | 1,30 | 0,91 | 0,79 | plateau od k02 |
| W2 niezmienniczość | rozpoznanie (przypadek 0,08) | 0,29 | 0,33 | 0,31 | brak postępu |
| W2 niezmienniczość | przesunięcie wariantu (rozrzut sond 6,1) | | | obrót -3,4; żółte tło +3,5; jasne narzędzie +3,9; czarne tło -0,9; skrócone +0,2 | systematyczne |
| W3 rozdzielenie | min a | 3,0 | 4,4 | 5,5 | spełniony |

Po odświeżeniu do obrotu 6: R około 0,78, rozpoznanie 0,31, przesunięcia: obrót około -3,4, żółte tło +3,3, jasne narzędzie +3,7; W1 około 0,002; min a 5,2-5,7. R liczone tylko na 4 obrazach holdout jest znacznie wyższe i po chwilowym spadku rośnie (na starcie 2,8, po obrocie 1 od 2,2 do 2,5, po obrocie 6 od 3,1 do 3,8), a rozpoznanie na holdout wynosi 0,21. Trzy pętle są nierozróżnialne (różnice R ≤ 0,02). Latent przestaje się ruszać po dwóch obrotach, ale zatrzymuje się w stanie niedomkniętym: obrót narzędzia albo kolor tła zmienia kod bardziej, niż różnią się dwa różne obrazy. Prawdopodobna przyczyna: uczenie ma wyłączone przekształcenia obrazów (`aug_config={}`, bez obrotów, zmian koloru i wielu skal), więc sieć nie dostaje przykładów „to samo narzędzie, inny obrót lub światło”.

**Definicja domknięcia (ustalona przed nowym wynikiem).** Latent tożsamości jest domknięty, gdy naraz: W1 ruch a < 0,005 rozrzutu na 5 aktualizacji; W2 R < 0,5, rozpoznanie ≥ 0,8 i |przesunięcie| < 0,5 (8 procent rozrzutu) dla obrotu, czarnego tła, żółtego tła i jasnego narzędzia; W3 min a > 0. Sam W1 nie wystarcza, bo jest spełniony przy niedomkniętej tożsamości. Tempo domykania: pole pod krzywą R po obrotach 1-3 i liczba aktualizacji do spełnienia W2.

**Projekt metody** (`PLAN.md`, stan: czeka na zatwierdzenie, nic nie zgłoszono). Klatki wybierane po zmierzonym wpływie na domknięcie, a nie po podobieństwie do obrazu FastGAN: cel J(θ) = średnia wariancja a między wariantami / wariancja średnich obrazów (z wagami tych przesunięć, których prognoza Koopmana sama nie zamyka); wynik klatki s(x) = ⟨∇θ L_det(x), ∇θ J⟩, czyli o ile mały krok uczenia na klatce x obniży J (przybliżenie pierwszego rzędu, gradient J przez różniczkowalny odczyt sond, ograniczony do ostatnich bloków kręgosłupa i projektora); kandydaci: 1200 losowych klatek puli na obrót, warstwowo po filmach; 134 najwyższe s(x), najwyżej 10 z filmu. Etap 0 (jedno zadanie hopper, około 30 minut): wagi k01 ziarna 42, 24 kandydatów, prawdziwy krok uczenia na każdej klatce i pomiar zmiany J; warunek przejścia: korelacja rang Spearmana przewidzianej i zmierzonej zmiany J ≥ 0,5, inaczej stop. Etap 1: 3 ziarna × obroty 1-3 od wag k00, ramiona `domykanie` (+134 klatki według s(x)), `augmentacja` (bez dodawania, uczenie z obrotami o 90°/180°/270° i zmianą jasności oraz koloru; tani punkt odniesienia, który metoda musi pobić), `losowe` i `bez` wzięte z kampanii Large. Kryterium: `domykanie` ma mniejsze pole pod krzywą R niż `losowe` i `bez` w każdym ziarnie i wyższe rozpoznanie na holdout, a mAP@0,5 nie niższe niż `bez` o więcej niż 0,005. Koszt: etap 0 około 0,5 godziny GPU; etap 1: 18 zadań × około 55 minut ≈ 16,5 godziny GPU plus 9 zadań oceny kandydatów × 20 minut ≈ 3 godziny; około 10,6 GB na Edenie. O 21:10 wolne było 24,1 GB, a obroty 4-6 bieżącej kampanii miały zająć jeszcze około 15,9 GB, więc obie kampanie się nie mieszczą: do decyzji użytkownika zatrzymanie kampanii Large po obrocie 4 albo zgoda na usunięcie starych katalogów na Edenie. Na pytanie „Czyli ty już sprawdzasz to na Edenie i liczysz?” (21:31) odpowiedź: nie; na Edenie liczy się tylko stara pętla, diagnozę zrobiono lokalnie na pobranym latencie, a nowy test ruszy dopiero po „tak”.

### 3.20 Stan na koniec etapu (28 września, 01:15)

O 01:00 użytkownik poprosił: „Zapisz sesję i wyłącz za 1h komputer”. Wyłączenie zaplanowano na 02:00. Sesje zapisano w repozytorium ViT: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-25_203826_Dobor_Klatek_Kontynuacja_134\` (uzupełniana do 27.09, 01:41) i `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-28_011148_Petla_Large_Koopman_i_Domykanie_Tozsamosci\`.

- Eden: zadanie 1817712 (s43, pętla bez, obrót 6) w kolejce albo w biegu; po nim trzeba ponownie uruchomić `dyrygent_large.py --s 43`, `analiza_large.py`, uzupełnić `WYNIKI.md` i sumy `SHA256.json`, powtórzyć `diagnoza_domkniecia.py`. Zadanie 1810406 (`koopman_fastgan_v1`, wstrzymane przez klaster) należy do równoległej sesji i nie było ruszane.
- Kwota Edenu po kampanii Large: około 7 GB wolnego. Do sprzątnięcia wyłącznie za zgodą użytkownika: `~/smallvit_runs/2026-09-25_kontynuacja_dobor/` (przerwany pilot, 2,4 GB), `2026-09-25_kontynuacja_dobor150/` (2,1 GB, 9 × 120 MB wag), `2026-09-25_kontynuacja_5epok/` (około 9 GB), `2026-09-26_petla_uczenia/` (pilot Nano, 6,9 GB). Wagi i latent tych kampanii pobrano lokalnie.
- Na decyzję czekały: projekt domykania (etap 0 i 1), miejsce na Edenie, sprawdzenie, czy środowisko Edena ma bibliotekę albumentations (ramię `augmentacja`).

### 3.21 Podsumowanie etapu 3

1. Koopman na samym położeniu latentu przegrywał z brakiem zmian, bo ruch latentu w uczeniu ma bezwładność i hamuje. Stan z pamięcią [z, v] naprawił to: dla 41 472 par obraz-token błąd jednego kroku spadł z 0,0103 do 0,0037 (o 3,9 procent poniżej ostatniego przyrostu), na 100 aktualizacji 0,076 wobec 0,117; bramkę przeszło 7445 jednostek (18 procent), ogólny próg 5 procent nie został osiągnięty.
2. Tożsamość narzędzia w latencie Nano jest dziedziczona z wag COCO (AUC 0,995 już w kroku 5). Najlepiej przewidywalny jest jeden skalar tożsamości: wspólny operator na 4 ostatnich przyrostach ma na holdout 0,272 błędu braku zmian na 1 krok (ostatni przyrost 0,284) i 0,171 na 100 aktualizacji (ostatni przyrost 0,617). Dryf tożsamości gaśnie około kroku 500.
3. Przewidziany ogólny przyrost latentu od kroku 245 na 100 aktualizacji to tylko 0,069 rozrzutu. Po naprawie pętli FastGAN (inwersja: błąd 0,09-0,27; krok Gaussa-Newtona: cosinus 0,84-0,97, 59-87 procent długości) obraz przewidzianego latentu różni się od obecnego o 0,1-0,9 procent jasności, a wyszukiwarka zwraca praktycznie tych samych sąsiadów kotwicy (54/60 wspólnych).
4. Dołożenie 17 klatek (pilot) nie zmieniło wyniku (ramiona różnią się o ≤ 0,0004), a 134 dobranych klatek na jedną epokę dało +0,0007 mAP@0,5, tyle co 134 losowe (+0,0009), w szumie ziaren; lokalnie przy kotwicach -0,003 wobec +0,005 losowych. Okolica kotwic w val jest trudna (AP 0,55, wokół TR_0134 0,38), ale dobrane klatki treningowe nie były trudne (AP 0,857 na starcie).
5. Kampania 5 epok (4 ramiona × 3 ziarna): dobrane -0,0005, losowe +0,0010, trudne +0,0016 mAP@0,5 wobec braku dodania; kryterium ustalone przed wynikiem niespełnione. Trudne klatki pomagają daleko od kotwic (+0,009), szkodzą przy nich (-0,018 wobec losowych).
6. Nowość metody sprowadza się do celu z prognozy Koopmana; sam schemat „wygeneruj, potem wyszukaj” istnieje od lat (ASAL 2018/2020, DCD 2024). Teoria (twierdzenie o stabilności wyboru top-k) wyjaśnia, czemu przy małym przesunięciu prognozy wybór się nie zmienia.
7. Równanie Langevina użytkownika z dryfem Koopmana i pamięcią 8 przyrostów poprawiło rollout na 100 aktualizacji od kroku 245 (0,151 wobec 0,171), ale nie przeszło bramki (1 krok gorszy o 2,2 procent, od kroku 485 gorsze od L4).
8. Pętla 8 obrotów na Nano (ziarno 42): najlepsza była pętla bez dodawania (0,9072 wobec 0,9055 koopman i 0,9045 losowe; przy kotwicach 0,627 / 0,593 / 0,563). Prognoza trafna tylko w obrocie 1 (0,127 wobec 0,111), potem latent stoi. Potwierdziło to diagnozę użytkownika: pętla wybiera klatki, do których uczenie i tak dochodzi.
9. Pętla od początku dostrajania na RF-DETR LargeDeprecated (DINOv2-B), 6 obrotów × 3 ziarna, 53 z 54 obrotów: przewidziana droga latentu kotwic w obrocie 1 to 1,19-1,67 rozrzutu wobec 0,12 w Nano, ale kryterium niespełnione; koopman - losowe -0,0030 mAP@0,5, ujemne w 15 z 18 par; pole pod krzywą najwyższe dla losowych w każdym ziarnie.
10. Diagnoza domknięcia: latent tożsamości Large staje po dwóch obrotach, ale niedomknięty (R ≈ 0,78, rozpoznanie wariantów 0,31, obrót i kolor tła przesuwają kod o 3,3-3,9 przy rozrzucie 6,1). Powstała formalna definicja domknięcia (W1-W3) i projekt doboru klatek po iloczynie gradientów, który na koniec etapu czekał na zgodę.

### 3.22 Otwarte pytania na koniec etapu 3

1. Czy prognoza w ogóle może zmienić wybór klatek, gdy cel ma dużą amplitudę (kierunek ku błędom detektora, różnice między obrazami) albo gdy startuje się w fazie silnego dryfu? Przewidywania P2 i P3 z `TEORIA_LANGEVIN.md` pozostały niesprawdzone.
2. Ablacja P1 („wyszukaj po samym obrazie kotwicy, bez Koopmana i FastGAN”) nie została wykonana, choć recenzent zażądałby jej jako pierwszej.
3. Czy dopasowanie w cechach samego detektora wokół narzędzia (poprawka znana z ASAL) zamiast DINOv2 z całej klatki przeniosłoby informację o małym szczególe, czyli końcówce narzędzia?
4. Czy latent sond (cechy kręgosłupa) w ogóle wiąże się z postępem detekcji? W Large latent staje po obrocie 2, a mAP rośnie dalej; tego związku w tym etapie nie zmierzono.
5. Dlaczego losowe klatki uczyły Large lepiej niż wybrane pętlą (15 z 18 par)? Hipoteza o skupieniu wyboru wokół trzech kotwic wobec różnorodności losowania nie została sprawdzona doborem różnorodnym.
6. Czy wynik s(x) (iloczyn gradientów) przewiduje rzeczywistą zmianę niezmienniczości R (etap 0 domykania, korelacja Spearmana ≥ 0,5), i czy wybór klatek domyka tożsamość szybciej niż zwykła augmentacja obrotami i kolorem?
7. Czego brakuje w równaniu Langevina w fazie dryfu (autokorelacja reszt 0,27 od kroku 245) i czy człon wymuszenia granicą epoki (korelacja -0,24 od kroku 485) da się dopasować; etap 2 z członem sterowania S(u) na 12 biegach kampanii 5 epok nie ruszył.
8. Dlaczego R liczone na 4 obrazach holdout jest znacznie wyższe od R na obrazach fit i od obrotu 2 rośnie (po obrocie 1 od 2,2 do 2,5, po obrocie 6 od 3,1 do 3,8)?
9. Pętla „przyspieszona” (cel 5 obrotów naprzód, kotwice niedouczone) na Nano od wag COCO nie była uruchomiona w swojej postaci; w Large użyto celu 3 obroty naprzód.
10. Wynik brakującego punktu (s43, bez, obrót 6, zadanie 1817712) i decyzje użytkownika o sprzątaniu Edenu (około 20 GB starych kampanii) pozostały otwarte.
11. Wszystkie porównania oparto na 3 ziarnach (pilot Nano na jednym); różnice rzędu 0,001-0,003 mAP@0,5 globalnie i 0,01 lokalnie mieszczą się w szumie, a do rozróżnienia 0,015 AP na top-134 potrzeba co najmniej 3 ziaren, dla mniejszych efektów kilkunastu.


## Etap 4. Diagnoza porażki, mały ViT od zera, wektor i siła tożsamości, FastGAN wzdłuż wektora i pętla w trzech wersjach (28-29 września 2026)

Etap trwał od poniedziałku 28 września 2026, godz. 13:52 (pierwsza wiadomość użytkownika po wznowieniu rozmowy), do wtorku 29 września 2026, godz. 14:52 (zapis sesji). Wszystkie godziny podaję w czasie lokalnym (CEST, UTC+2). Commity repozytorium mają czas lokalny, a znaczniki transkryptu rozmowy są w UTC, więc przesunąłem je o dwie godziny. Rozmowę prowadzono z katalogu projektu grantowego `F:\Studia\MCHTR\Grant\Projekt`, ale cała praca merytoryczna dotyczy repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`. Wszystkie eksperymenty tego etapu leżą w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\`. Etap obejmuje 34 commity, od `5acea332` do `8d50ae84`. Zapis sesji: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek\` (`README.md`, `SESSION_SUMMARY.md`).

O 14:01 użytkownik włączył tryb prostych odpowiedzi (`/wyjasniaj-prosto ON`): odpowiedzi miały mieć 2-3 krótkie zdania i objaśniać każdy termin. Dlatego wiele liczb padało w rozmowie w skrócie, a pełne wartości są w plikach wyników, na których opiera się ten opis. Wiadomości użytkownika były w większości dyktowane głosem; w cytatach poprawiłem tylko oczywiste przekłamania transkrypcji i brakujące polskie znaki (np. „latem” na „latent”, „kupman” na „Koopman”, „fastgun” na „FastGAN”), a pominięte fragmenty oznaczyłem „[...]”.

### 4.1 Punkt wyjścia, pojęcia i przebieg w czasie

**Stan na początku etapu.** Etap 3 skończył się porażką pętli „Koopman + FastGAN” na dostrajanym detektorze RF-DETR Large. Detektor to sieć, która na klatce filmu rysuje prostokątne ramki wokół narzędzia i przy każdej podaje pewność, liczbę od 0 do 1. RF-DETR składa się z kręgosłupa (gotowej sieci DINOv2, która zamienia obraz na cechy), projektora (łącznika przerabiającego cechy na wejście dekodera) i dekodera (części, która z cech wyznacza ramki). W kampanii z etapu 3 (6 obrotów, 3 ziarna, trzy ramiona) losowe klatki uczyły lepiej niż klatki wybrane przez Koopmana w 15 z 18 porównań, średnio o 0,0030 mAP@0,5, a kryterium ustalone przed eksperymentem nie zostało spełnione. Brakowało jednego z 54 punktów (ramię bez dokładania, ziarno 43, obrót 6), bo jego zadanie padło przez usterkę węzła Edenu.

**Pojęcia, które wracają w całym etapie** (objaśnienia dotyczą znaczenia w tym projekcie):
- **mAP@0,5** - główna miara jakości detektora, w skali 0-1 (w rozmowie także 0-100). Ramka liczy się jako trafiona, gdy pokrywa się z prawdziwą ramką narzędzia co najmniej w połowie (IoU ≥ 0,5; IoU to pole części wspólnej dwóch prostokątów podzielone przez pole ich sumy). Miara łączy odsetek znalezionych narzędzi i odsetek trafnych ramek, licząc od ramek, których detektor jest najbardziej pewny.
- **Latent** - wewnętrzny opis obrazu w sieci: wektor liczb, który sieć wylicza z obrazu na jednej ze swoich warstw. W małym ViT (od podrozdziału 4.6) jest to 768 liczb na każdy kawałek obrazu.
- **Sonda** - mały, stały obrazek testowy z narzędziem, na którym po każdym kroku pomiaru odczytuje się latent. Sondy pochodzą z prawdziwych klatek z ręcznie zaznaczonym narzędziem (maską), przerobionych na wersje: obrót, czarne tło, żółte tło, krótsze albo jaśniejsze narzędzie i inne.
- **Kotwica** - oryginalny wycinek narzędzia (sonda w wersji „original”), od którego generator FastGAN zaczyna rysowanie obrazu do wyszukiwania klatek.
- **Ziarno** - jedno powtórzenie całego eksperymentu z innym losowaniem (inny start wag, inna kolejność danych). Wynik powtarzalny musi się pokazać w większości ziaren.
- **Obrót pętli** - jeden cykl: prognoza, wybór nowych klatek, dołożenie ich do zbioru uczącego i porcja uczenia (w małym ViT 500 kroków).
- **Ramię** - jeden sposób doboru klatek, liczony z tego samego stanu sieci co pozostałe (np. `koopman`, `losowe`, `bez` dokładania).
- **Pole pod krzywą mAP@0,5** - średnia z ocen mAP@0,5 robionych co 100 kroków po rozgrzewce. To jest miara szybkości uczenia: ramię, które szybciej dochodzi do dobrych wyników, ma większe pole. Nie mylić z AUC w podrozdziałach 4.12 i 4.19-4.21, które jest innym pojęciem.
- **„Brak zmiany”** - najprostsza prognoza: jutro będzie tak jak dziś. Stosunek błędu prognozy do błędu „braku zmiany” poniżej 1 znaczy, że prognoza coś wnosi.
- **Koopman (w tym projekcie)** - model liniowy, który przewiduje następny przyrost stanu z kilku ostatnich przyrostów (regresja grzbietowa, czyli najmniejsze kwadraty z karą za duże współczynniki). Nie ma w nim członu, który hamowałby ruch.
- **Langevin (w tym projekcie)** - model przyrostu stanu z dryfem hamującym w stronę punktu stałego (jak kulka toczona w misce), pamięcią kilku poprzednich przyrostów i szumem. W wersji użytkownika z projektu grantowego dochodzą jeszcze wymuszenie i sterowanie.
- **Baza, pula, walidacja** - baza to 1500 oznaczonych klatek, od których zaczyna się uczenie; pula to klatki treningowe, z których dobiera się nowe klatki do dołożenia; walidacja to klatki z innych filmów, których sieć nigdy nie widzi w uczeniu i które służą tylko do oceny.
- **Krok, epoka, rozgrzewka** - krok to jedna poprawka wag po jednej paczce przykładów; epoka to jedno przejście przez wszystkie klatki bazy; rozgrzewka to wspólny początek uczenia na samej bazie, zanim ramiona zaczną dokładać klatki.
- **DINOv2** - gotowa sieć do opisu obrazów, uczona bez etykiet na milionach zdjęć; w RF-DETR pełni rolę kręgosłupa, a w pętli służy do wyszukiwania w puli klatek podobnych do zadanego obrazu.
- **FastGAN** - generator obrazów typu GAN (sieć uczona w parze z drugą siecią, dyskryminatorem, która ocenia, czy obraz jest prawdziwy); z 256 liczb wejścia z rysuje wycinek 256 × 256 z narzędziem i maską narzędzia. Szczegóły w podrozdziale 4.6.
- **Eden** - klaster obliczeniowy uczelni z kolejką zadań SLURM, z kartami H100 (węzeł `hopper`) i A100 (węzły `dgx-1`, `dgx-4`), z kwotą 200 GB na projekt.

**Przebieg w czasie (commity etapu):**

| Czas | Commit | Co |
|---|---|---|
| 28.09 14:18 | `5acea332` | Diagnoza porażki pętli Koopmana na RF-DETR Large |
| 14:35 | `b68beb06` | Odczyt dekodera na sondach, plan pętli Langevin + Kalman + FastGAN |
| 15:03 | `4ee659fd` | Bramki G1 i G2 lokalnie na wagach Nano (obie niespełnione) |
| 15:09 | `9314fc85` | Arkusze sond i próbki walidacji, poprawka opisu tła walidacji |
| 15:21 | `814201a9` | Plan małego ViT od zera, bramki Z0-Z3 |
| 15:44 | `235dcd80` | Kod pętli wersji 1, kalibracja Z0 |
| 15:55 | `8df08a91` | Analiza Edenu przed sprzątaniem, skrypt analizy pętli |
| 16:07 | `7de1c7c7` | Wersja Eden, zgłoszone 1817930 (próba) i 1817931 |
| 16:21 | `c29df166` | Poprawki (timm, ścieżki Windows), zgłoszone 1817934 i 1817935 |
| 16:25 | `91e97450` | Eden posprzątany: 193,2 -> 52,2 GB |
| 16:28 | `d661a8ee` | Wynik lokalny ziarna 42, kontrola target/current |
| 16:33 | `f7efafbf` | Plan użytkownika na następne sprawdzenie |
| 16:46 | `3a178c23` | Wyniki Eden 1817935 i analiza 11 ziaren |
| 17:04 | `e5897e27` | Analiza wsteczna latentów 11 ziaren |
| 17:19 | `ddfeae69` | Wersja 2, zgłoszone 1817948/1817949 |
| 17:48 | `1ded306b` | Wyniki wersji 2 |
| 18:13 | `6eb7fff0` | Tożsamość po 21 epokach metodą użytkownika |
| 18:20 | `b7a3b308` | Wektor tożsamości po 21 epokach |
| 18:29 | `8deaf528` | Koopman i Langevin na stałym wektorze tożsamości |
| 19:10 | `5f498d37` | FastGAN: diagnoza chybiania, głębsze warstwy, sieć MLP |
| 19:35 | `cf97cf3a` | Przegląd literatury, sterowanie odporne |
| 20:04 | `6f6bbf06` | FastGAN sterowany wzdłuż wektora tożsamości |
| 21:52 | `d289a8ad` | Kod wersji 3, zgłoszone 1818024/1818025 |
| 21:59 | `831bf563` | Skrypt analizy z kryteriami ustalonymi przed wynikiem |
| 22:34 | `d66e6eb8` | Wyniki kampanii 1818025 |
| 22:34 | `b67127d2` | Listy wyszukiwania poza gitem |
| 29.09 00:20 | `8ff21d8e` | Zliczenie wygranych, sondy syntetyczne |
| 01:17 | `e3836c66` | Plan iteracji 2 |
| 01:23 | `c9c8ee4a` | Iteracja 2 zgłoszona (1818050/1818051) |
| 03:45 | `97186aa0` | Wyniki iteracji 2 |
| 13:08 | `fdf82918` | Tożsamość na parach z narzędziem i bez |
| 14:23 | `887e48fb` | Zapis wektorów i ich skupienie na neuronach |
| 14:46 | `238851f7` | Kierunki tożsamości w obrazie cząstkowym |
| 14:51 | `fb899bde` | Plan dekodera latentu |
| 14:52 | `8d50ae84` | Zapis kierunków cząstkowych, porównanie z wektorem `nowy` |

### 4.2 Diagnoza porażki pętli na RF-DETR Large (28.09, 14:04-14:19, `5acea332`)

**Skąd pytanie.** O 14:04 użytkownik zapytał, jak dokładnie liczono na Edenie i dlaczego Koopman nie przyspiesza uczenia. W wklejonym tekście napisał: „chciałbym po prostu wiedzieć, dlaczego to mi nie wychodzi [...] Może to wina jest modelu, może to jest wina tego, że jest za mały epok, może to jest wina tego, że to jest za mały model. Nie wiem, po prostu zweryfikuj mi to, możesz użyć internetu, możesz użyć paper search [...] albo agentów nawet [...] zrób to dokładnie teraz”. Jako cel podał „dobrą metodę identyfikacji tożsamości narzędzia” i przewidywanie, „jakie klatki powinny wchodzić do uczenia”.

**Metoda.** Bez nowego liczenia na karcie graficznej. Skrypt `diagnoza_porazki.py` przeliczył dane kampanii z etapu 3, wynik trafił do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_od_poczatku_large\wyniki\DIAGNOZA_PORAZKI.json`, opis do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_od_poczatku_large\DIAGNOZA_PORAZKI.md`. Równolegle dwa agenty przeglądały literaturę w Undermind i Scite (dla sprawdzonych prac brak not o wycofaniu). Cztery pomiary:
1. błąd prognozy Koopmana 3 obroty naprzód (w opisie skryptu „96 kroków”) wobec „braku zmiany” i „stałej prędkości”, starty z obrotów k1-k4, 3 ziarna, czyli 12 przypadków;
2. względna zmiana wag w czterech grupach: kręgosłup (`backbone.0.encoder`), projektor (reszta `backbone`), dekoder (`transformer`), głowice (reszta);
3. związek skalara tożsamości z mAP@0,5 między pętlami w tym samym ziarnie i obrocie (współczynnik Spearmana ρ, czyli korelacja kolejności, i wartość p, czyli prawdopodobieństwo takiego wyniku przy braku jakiegokolwiek związku). Skalar tożsamości z etapu 3 to jedna liczba na sondę, mówiąca, jak mocno jej latent wskazuje tożsamość narzędzia;
4. różnorodność i pokrycie puli przez dodane klatki, na indeksie DINOv2 z `koopman_fastgan_petla\wyniki\local_1810448\run\index`.

**Receptura uczenia na Edenie (odtworzona w diagnozie).** RF-DETR LargeDeprecated od wag COCO. Jeden obrót to jedna epoka: 2560 próbek, 160 kroków (paczka 4 × akumulacja 4). Optymalizator AdamW (odmiana spadku gradientu, która sama dopasowuje wielkość kroku dla każdej wagi) ze stałym krokiem uczenia, bez rozgrzewki: lr = 1,67e-5 dla dekodera, projektora i głowic oraz lr_encoder = 1,67e-6 dla DINOv2. To 6 i około 90 razy mniej niż wartości domyślne w rfdetr 1.10.1 (1e-4 i 1,5e-4). Krok uczenia (lr) mówi, jak mocno sieć poprawia wagi po każdej porcji przykładów. Bez EMA (uśredniania wag w czasie), bez przekształceń obrazu (`aug_config={}`), bez wielu skal. Każdy obrót startował z samych wag poprzedniego, z wyzerowanym stanem optymalizatora. Próbnik brał każdą klatkę bazy (1500) i każdą dodaną raz na epokę, resztę uzupełniał losowymi powtórzeniami bazy, więc dodane klatki stanowiły 5 % próbek w k01 i 31 % w k06. Razem 7 epok; mAP@0,5 wynosiło 0,86 po pierwszej i 0,90-0,91 po siódmej.

**Wyniki** (`DIAGNOZA_PORAZKI.json`):

| Pomiar | Wynik |
|---|---|
| Błąd prognozy Koopmana 3 obroty naprzód (w rozrzucie sond) | 0,196 wobec 0,078 dla „braku zmiany”; Koopman gorszy w 12 z 12 przypadków |
| Zgodność kierunku przewidzianego i prawdziwego ruchu skalara po sondach | korelacja 0,81-0,94 |
| Ruch netto skalara tożsamości w kolejnych obrotach | 2,30 (k00), 0,99 (k01), 0,10 (k02) |
| Zmiana wag COCO -> k00 | kręgosłup 0,013 %, projektor 0,74 %, dekoder 0,78 %, głowice 0,017 % |
| Zmiana wag k00 -> k06 | kręgosłup 0,026 %, projektor 1,6 %, dekoder 1,3 %, głowice 0,05 % |
| Liczba parametrów grup | kręgosłup 86,58 mln, projektor 32,46 mln, dekoder 14,3 mln, głowice 1,81 mln |
| Skalar tożsamości a mAP@0,5 między pętlami | ρ = 0,07, p = 0,61, n = 53 |
| Niezmienniczość R a mAP@0,5 | ρ = -0,20, p = 0,16 |
| Średni cosinus DINOv2 dodanych klatek | koopman 0,66-0,68, losowe 0,62 |
| Klatki z sąsiadem z tego samego filmu w odstępie ≤ 30 klatek (na 804 dodane) | koopman 138-140, losowe 62-83 |
| Pokrycie puli | koopman 0,893, losowe 0,895, bez dokładania 0,887 |
| Filtr RHO („do nauczenia”) | przepuszcza 134 z około 190-224 kandydatów, czyli prawie nie filtruje |
| Wpływ prognozy na kandydatów | zmienia około 20 % (wspólne top-45, czyli 45 najbardziej podobnych klatek, dla obrazu z prognozą i bez niej około 80 %) |
| Wielkość efektu i szum | efekty około 0,003 mAP przy szumie między ziarnami 0,003-0,006 |

Cosinus to podobieństwo kierunku dwóch wektorów: 1 znaczy ten sam kierunek, 0 kierunki prostopadłe. R to miara z etapu 3, która porównuje zmianę latentu między wersjami tego samego wycinka ze zmianą między różnymi wycinkami (powyżej 1 znaczy, że obrót albo inne tło zmienia kod bardziej niż zmiana obrazu). Filtr RHO (od metody RHO-LOSS) miał przepuszczać tylko klatki, na których sieć ma jeszcze coś do nauczenia.

**Pięć usterek wskazanych w diagnozie.** (1) Prognoza Koopmana przestrzeliwała wielkość ruchu: operator dopasowany do szybkiego ruchu z rozgrzewki nie widział hamowania, więc FastGAN rysował obrazy dla stanów, do których sieć nigdy nie doszła. (2) Latent czytano w kręgosłupie, który w tej recepturze prawie się nie uczył; postęp detekcji robiły projektor i dekoder. (3) Latent nie wiązał się z wynikiem detekcji; stał od k02, a mAP rósł dalej. (4) Wybór skupiał klatki wokół trzech kotwic, z dwa razy większą liczbą prawie identycznych sąsiednich klatek. (5) Dźwignia była słaba, a szum duży; według literatury do wykrycia różnicy rzędu 0,003 trzeba 10-29 sparowanych biegów. Rozmiar modelu nie był przyczyną (Nano i Large dały to samo, a Large ruszał latentem około 10 razy mocniej bez zysku). Liczba epok też nie, bo kręgosłup stał od k02 z powodu kroku uczenia.

**Literatura przytoczona w diagnozie.** Koopman w uczeniu sieci prognozuje wagi, działa w pobliżu atraktora i na krótkim horyzoncie; nikt nie używał go do doboru danych (Dogra i Redman 2020, arXiv:2006.02361; Tano i in. 2020, arXiv:2006.14371; Redman i in., NeurIPS 2023). Szybka zmiana reprezentacji trwa 2-3 epoki (Fort i in. 2020, arXiv:2010.15110); niższe warstwy przy dostrajaniu prawie stoją (Neyshabur i in. 2020, arXiv:2008.11687); ruch cech nie jest miarą postępu (Kumar i in. 2022, arXiv:2202.10054); zamrożony kręgosłup z pojemną szyją dorównuje dostrajaniu w detekcji (Vasconcelos i in., CVPR 2022). Wybór top-k bez różnorodności przegrywa z losowym nawet na cechach DINOv2 (Gupte i in. 2024, arXiv:2401.14555; Kirsch i in. 2021, arXiv:2106.12059); w wideo sąsiednie klatki są nadmiarowe (Bengar i in. 2019); w detekcji dobór po stracie obniża mAP, a pomaga miara zgodna z AP (Sun i in. 2025, arXiv:2511.14197). RHO-LOSS działa przy doborze co paczkę i około 10 % kandydatów (Mindermann i in. 2022, arXiv:2206.07137). Odwracanie generatora przez cechy nie gwarantuje wyglądu, a DINOv2 wyszukuje po wyglądzie (Sabour i in. 2016, arXiv:1511.05122), więc łańcuch FastGAN -> DINOv2 rwie związek z latentem. Różnice w aktywnym uczeniu często mieszczą się w szumie (Munjal i in., CVPR 2022; Bouthillier i in. 2021; Okanovic i in. 2023).

**Zalecenia z diagnozy (w kolejności):** bramka sprawdzająca, czy latent przewiduje wynik zadania; latent tam, gdzie jest uczenie (po projektorze albo z normalnym lr_encoder); prognoza z hamowaniem, używana tylko wtedy, gdy bije „brak zmiany”; wybór zgodny z zadaniem i różnorodny (wpływ gradientowy, jedna klatka na skupisko, wykluczenie sąsiadów ±30 klatek); większa moc statystyczna (co najmniej 10 par ziaren).

**Odpowiedź dla użytkownika (14:19).** Asystent odpowiedział, że winny nie jest ani model, ani liczba epok, tylko trzy usterki metody (prognoza gorsza od „braku zmiany”, latent w kręgosłupie, który się nie uczy, skupiony wybór klatek), i zaproponował naprawę w czterech krokach, zaczynając od sprawdzenia, czy latent w ogóle przewiduje wynik detekcji.

**Czego diagnoza nie dowodzi.** Dotyczy jednej receptury z bardzo małym krokiem uczenia kręgosłupa i trzech ziaren. Nie pokazuje, że prognoza Koopmana nie może działać w ogóle, ani że inny latent będzie lepszy; to sprawdzały kolejne kroki.

### 4.3 Dekoder zamiast kręgosłupa: odczyt dekodera na sondach i projekt pętli sterowania (14:21-14:44, `b68beb06`)

**Skąd pytanie.** O 14:21 użytkownik powtórzył zdanie z diagnozy: „Najpierw trzeba sprawdzić, czy latent w ogóle przewiduje wynik detekcji” i zapytał, co to znaczy oraz czy lepiej skupić się na dekoderze. Asystent wyjaśnił, że „latent przewiduje wynik detekcji” znaczy: tam, gdzie latent przesunął się dalej, detektor powinien działać lepiej; tu związek jest praktycznie zerowy, „jak sterowanie samochodem według prędkościomierza, który nie jest podłączony do kół”. O 14:23 użytkownik zaproponował: „Może my to w ogóle źle robimy, może my powinniśmy patrzeć jeszcze na dekoder [...] dekoder to jest kolejna sieć [...] query key value, czyli to jest de facto znowu attention [...] powinniśmy nie patrzeć na to, jak się zachowuje latent, tylko jak ten latent przechodzi przez dekoder [...] poszukaj i zasugeruj coś”, i uruchomił wyszukiwanie literatury o dynamice dekodera DETR (zapytania obiektowe, mapy uwagi krzyżowej, stabilność dopasowania węgierskiego). O 14:25 dopisał: „pamiętaj, że ja tutaj próbuję wepchnąć jeszcze równanie Koopmana i równanie Langevin'a z poprzedniej pracy [...] ze sterowaniem, z wymuszeniem, z przewidywaniem dryfem i błędem [...] Kalman i odszumianie [...] albo po prostu tutaj teraz fastgana [...] to musi zadziałać”.

**Metoda odczytu dekodera.** Skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dekoder_sondy_large\odczyt_dekodera.py`, bez uczenia. Wagi RF-DETR LargeDeprecated z obrotów k00-k06 łańcucha `koopman`, 3 ziarna (42, 43, 44). 72 sondy: 12 klatek × 6 wersji (oryginał, obrót o 90°, czarne tło, żółte tło, narzędzie skrócone do 0,8, narzędzie rozjaśnione), wycinki 256 × 256 przeskalowane do rozdzielczości modelu. Dekoder RF-DETR ma 3 warstwy i 300 zapytań; zapytanie to jedno z 300 „miejsc” w dekoderze, z których każde proponuje jeden obiekt. Dla każdej sondy i każdej warstwy liczono jakość wykrycia **q** = największy po zapytaniach iloczyn pewności i IoU ramki z prawdziwą ramką (wyznaczoną z maski narzędzia), oraz IoU propozycji przed dekoderem. Analiza: `analiza_dekodera.py`, wynik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dekoder_sondy_large\wyniki\ANALIZA_DEKODERA.json` (surowe odczyty w `DEKODER.json` i `DEKODER.npz`).

**Wyniki na Large (`ANALIZA_DEKODERA.json`):**

| Ziarno | mAP@0,5 k00 -> k06 | Średnie q dekodera k00 -> k06 | Skalar kręgosłupa k00 -> k06 |
|---|---|---|---|
| 42 | 0,862 -> 0,910 | 0,241 -> 0,437 | 19,09 -> 20,18 |
| 43 | 0,861 -> 0,902 | 0,250 -> 0,430 | 19,09 -> 20,27 |
| 44 | 0,859 -> 0,907 | 0,242 -> 0,473 | 19,13 -> 20,34 |

- Średnie q rosło przez wszystkie obroty (0,24 -> 0,37 -> 0,44-0,47), a skalar kręgosłupa stanął po k02 (około 20,2-20,3).
- Związek z mAP@0,5 w czasie, po wycentrowaniu w każdym ziarnie: dekoder ρ = 0,856, kręgosłup ρ = 0,406.
- q według wersji sondy:

| Wersja | k00 | k06 |
|---|---|---|
| oryginał | 0,33 | 0,57 |
| jaśniejsze narzędzie | 0,41 | 0,65 |
| narzędzie skrócone | 0,32 | 0,58 |
| obrót 90° | 0,26 | 0,40 |
| żółte tło | 0,10 | 0,30 |
| czarne tło | 0,05 | 0,18 |

- Stan ukryty wybranego zapytania nie jest niezmienniczy: R = 1,61 (k00), 1,51 (k03), 1,52 (k06); rozpoznanie 0,29 / 0,35 / 0,33.
- Ruch stanu dekodera w jednostkach rozrzutu sond: 0,74 (k00-k01), 0,41 (k01-k02), 0,35 (k02-k03), 0,44 (k03-k06).

**Literatura o dekoderze (z wyszukiwania o 14:23).** Conditional DETR: wolne zbieganie DETR wynika z uwagi krzyżowej dekodera i zapytań treściowych, poprawka daje przyspieszenie 6,7-10 razy. DN-DETR: niestabilność dopasowania węgierskiego (dopasowanie węgierskie to przypisanie w każdym kroku, które zapytanie odpowiada któremu prawdziwemu obiektowi) mierzona liczbą zmian przypisań, poprawka daje przyspieszenie 2 razy. DeLR uznano za nieprzydatny (dotyczy Faster R-CNN). Te pozycje są w streszczeniu rozmowy, nie w pliku planu.

**Projekt pętli sterowania** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\sterowanie_langevin_kalman\PLAN.md`), budowany na prośbę użytkownika, żeby połączyć jego równanie Langevina z Koopmanem, Kalmanem i FastGAN-em:
- stan x_t (co 5 aktualizacji): q każdej sondy po warstwach 1-3 (72 × 3) i IoU propozycji; do równania wchodzi zredukowany wektor: średnie q według wersji (6) i według obrazu (12). Tożsamość domknięta znaczy: q równe we wszystkich wersjach jednego obrazu i wysokie;
- model: `Δx_t = -K (x_t - x*) + Σ_{j=1..m} B_j Δx_{t-j} + W f_t + C u_t + ε_t`, gdzie -K(x_t - x*) to dryf Koopmana z hamowaniem do punktu stałego (proces Ornsteina-Uhlenbecka, OU), Σ B_j Δx to pamięć (w eksperymencie `langevin_latent_detektora` 8 przyrostów poprawiało długi rollout o 11-13 %), W f_t to wymuszenie harmonogramem (granica epoki, reset optymalizatora), C u_t to sterowanie danymi (u_t to skład dodanych klatek), ε_t to szum;
- obserwator Kalmana: filtr, który łączy prognozę z zaszumionym odczytem y_t = x_t + ν_t i odsiewa drgania uczenia; różnica „odczyt minus prognoza” (innowacja) mówi, czego model nie tłumaczy;
- sterownik: wybór klatek minimalizujących przewidywaną lukę między wersjami po H krokach, z ograniczeniami różnorodności (jedna klatka na skupisko DINOv2, wykluczenie sąsiadów ±30 klatek, limit na film);
- macierz sterowania C: wpływ klatki na stan szacowany zgodnością gradientów ΔX ≈ -η ⟨∇θ x, ∇θ L_det(z)⟩ na dekoderze i projektorze;
- aktuator: FastGAN rysuje narzędzie w warunkach luki (inne tło, obrót), a zapytanie do puli idzie po wpływie C; tańszy wariant do porównania: wklejanie narzędzia na różne tła;
- bramki ustalone przed wynikiem: G1 (stan przewiduje zadanie; między pętlami ρ ≥ 0,4 i p < 0,05; koszt około 15 min GPU), G2 (model bije „brak zmiany”: błąd na 1 obrót ≤ 0,8 × „brak zmiany”; około 4 h GPU), G3 (wpływ klatki przewidywalny, Spearman ≥ 0,5; około 1 h GPU), G4 (kampania 5 ziaren × 4 obroty × 3 ramiona `sterowanie`, `losowe z wykluczeniem sąsiadów`, `augmentacja`; około 45 h GPU; kryterium: pole pod krzywą `sterowanie` większe od obu odniesień w ≥ 4 z 5 ziaren i średnio ≥ +0,003). Szacowane zapotrzebowanie G4 na miejsce: około 35 GB na Edenie.

**Rozmowa o uczeniu od zera (14:40-14:44).** Na pytanie, czemu dekoder da lepszy pomiar, asystent odpowiedział, że w tej recepturze prawie całe uczenie działo się w dekoderze i projektorze (wagi zmieniały się około pięćdziesiąt razy mocniej niż w kręgosłupie), a wyjście dekodera jest tuż przy wyniku; kręgosłup porównał do „termometru w pokoju obok”. O 14:43 użytkownik zaproponował: „żeby po prostu uczyć sieć od zera [...] Może wtedy byśmy zobaczyli, że [...] w Encoderze coś się zaczyna dziać i Koopman wtedy zaczyna pracować poprawnie”. Asystent wtedy odradzał uczenie RF-DETR od zera („uczenie od zera to dla nas zła droga”, za mało danych, DETR na COCO potrzebował setek epok) i proponował zamiast tego normalny krok uczenia kręgosłupa (jedno ziarno, dwie epoki). O 14:44 użytkownik poprosił: „policzyć to na podstawie lokalnych wag [...] a potem ewentualnie zgłosić to na Eden”.

**Czego to nie dowodzi.** Korelacja 0,86 była liczona w czasie w obrębie jednej pętli, więc mogła wynikać tylko z tego, że q i mAP rosną razem z uczeniem. To zastrzeżenie asystent zgłosił od razu, a bramka G1 miała je rozstrzygnąć porównaniem pętli w tym samym momencie uczenia.

### 4.4 Bramki G1 i G2 na lokalnych wagach Nano (14:44-15:13, `4ee659fd`, `9314fc85`)

**Metoda.** Zgodnie z prośbą użytkownika bez Edenu, na karcie RTX 3070 (8 GB). Skrypt `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dekoder_sondy_large\odczyt_dekodera_nano.py` odczytał q na tych samych 72 sondach z 244 zapisanych lokalnie wag RF-DETR Nano (kręgosłup DINOv2-S, 384 px, 2 warstwy dekodera; stan = q po ostatniej warstwie). Trzy zestawy:
- `nano5ep`: kampania 5 epok z `koopman_tozsamosc_kierunki\wyniki\dobor_klatek\trening5ep`, 4 ramiona (`bez_dodania`, `losowe`, `trudne`, `koopman_fastgan`) × 3 ziarna × 5 epok = 60 wag;
- `nanopilot`: pilot pętli z `petla_uczenia_koopman_fastgan\wyniki\lancuchy`, 3 pętle × 8 obrotów, ziarno 42 = 24 wagi;
- `nanogesty`: bieg gęsty 1810662 z `koopman_gesta_historia`, wagi co 5 aktualizacji, 160 zapisów (kroki 5-805).

Skrypt `bramki_lokalne.py`: G1 to Spearman między wycentrowanym średnim q a wycentrowanym mAP@0,5 w grupach (ziarno, epoka), między ramionami. G2 na biegu gęstym: dopasowanie na krokach ≤ 400, prognozy z 52 punktów startu na h = 20 zapisów (100 aktualizacji, czyli 1 epoka), modele M0 „brak zmiany”, M1 Koopman na przyrostach (8 ostatnich, bez hamowania), M2 dryf z hamowaniem (OU) z osobną stałą każdej sondy, M3 = M2 + pamięć 8 przyrostów + wymuszenie na granicy epoki, M4 = M3 dopasowany na szeregu odszumionym filtrem Kalmana. Miara: błąd średniokwadratowy prognozy podzielony przez błąd „braku zmiany”. Wynik: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dekoder_sondy_large\wyniki\BRAMKI_LOKALNE.json`.

**Wyniki G1 (warunek ρ ≥ 0,4 i p < 0,05): niespełniona.**

| Zestaw | Odczyt | ρ | p |
|---|---|---|---|
| nano5ep (60 wag, 15 grup) | q ostatniej warstwy | -0,07 | 0,61 |
| nano5ep | średnia warstw / IoU propozycji / najsłabsza wersja | -0,10 / -0,05 / -0,08 | 0,47 / 0,73 / 0,54 |
| nano5ep | pojedyncze wersje | od -0,13 (żółte) do +0,05 (oryginał) | wszystkie ≥ 0,32 |
| nanopilot (24 wagi, 8 grup) | q ostatniej warstwy | -0,51 | 0,01 |
| nanopilot | czarne tło / żółte tło | -0,69 / -0,73 | 0,0002 / 0,00005 |
| nanopilot | IoU propozycji | +0,53 | 0,008 |

W pilocie znak był odwrotny: ramiona z dodanymi klatkami lepiej wykrywały narzędzie na zmienionym tle (q na czarnym i żółtym tle: `bez` 0,257, `koopman` 0,277, `losowe` 0,301), a mAP@0,5 walidacji było w nich minimalnie niższe (0,8989 / 0,8981 / 0,8973). Rozrzut mAP@0,5 między ramionami w grupie wynosił tylko 0,0009-0,0011.

**Wyniki G2 (warunek ≤ 0,8 × „brak zmiany”): niespełniona.** q rosło od 0 (nowa głowica jednoklasowa; średnio 0,0017 na starcie) do około 0,39 w kroku 400, potem stało i drgało (0,39-0,44; 0,433 na końcu). Stosunek błędów przy dopasowaniu na fazie wzrostu: M1 1,00, M2 1,77, M3 1,64, M4 1,64 (współczynnik hamowania -0,0098). Sprawdzenie po wyniku, z dopasowaniem na samym plateau (kroki 300-600, test > 600): dla sond 0,97-1,00; dla średniej sond hamowanie z pamięcią 0,92 (h = 20) i średnia z plateau 0,82 (h = 20), nadal powyżej progu 0,8. Tylko na krótszym horyzoncie h = 4 średnia z plateau dała 0,64 dla średniej sond (dla pojedynczych sond 1,51), ale kryterium dotyczyło horyzontu jednej epoki.

**Pokazanie sond i poprawka błędu.** O 15:07 użytkownik zażądał: „pokaż zdjęcia gdzie to leży! powiedz co dokładnie robiłeś?”, a o 15:08 dopisał: „Ja nie jestem w stanie zrozumieć, dlaczego tutaj wszystko mi jakoś niepoprawnie wychodzi [...] powiedz mi, dlaczego Koopman i Langevin mają złe wyniki”. Skrypt `arkusz_sond.py` (`9314fc85`) utworzył arkusze `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dekoder_sondy_large\wyniki\SONDY_ARKUSZ.png` i `WAL_PROBKA.png` (pliki PNG są poza gitem, odtwarza je skrypt) oraz folder `wyniki\sondy72` z 72 kopiami o czytelnych nazwach (np. `TR_0001_klatka00000576__3_czarne_tlo.png`). Oryginały sond leżą w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_fastgan_petla\wyniki\local_1810448\run\probe\images` (192 pliki = 12 klatek × 16 wersji, obok maski w `masks`). Asystent wcześniej napisał, że zbiór walidacyjny ma „tło z laboratorium”; po obejrzeniu zdjęć przyznał, że to było niesprawdzone zgadywanie („co ty wymyśliłeś?” zapytał użytkownik o 15:13) i poprawił opis: walidacja to 12 043 całe klatki operacji zaćmy z 25 innych filmów, zawsze z okiem w tle.

**Interpretacja.** Równania nie były zepsute; stan dekodera po szybkim starcie nie ma ruchu, który dałoby się przewidzieć. Asystent użył obrazu piłki, która stoczyła się do dołka: gdy stanie, najlepsza prognoza to „zostanie tam”, a dryf w równaniu Langevina spada do zera i zostaje samo drganie od kolejnych porcji danych. Sondy mierzą rozpoznanie narzędzia w zmienionych warunkach (sztuczne tło, obrót), a walidacja tego nie sprawdza; na czarnym tle detektor zwykle trafiał ramką, ale z pewnością około 0,1. Korelacja 0,86 na Large była wspólnym wzrostem obu wielkości w czasie, a nie dowodem, że stan dekodera rozróżnia sposoby doboru danych.

**Co wynikło.** G3 i G4 nie poszły na Eden. Plan dopisał wniosek: miara sukcesu musi obejmować to, co mierzą sondy, a stan modelu trzeba czytać w fazie, w której się rusza (początek uczenia, nie plateau). Plan `domykanie_tozsamosci_large\PLAN.md` z etapu 3 i ten plan zostały bez zgody na kampanię.

### 4.5 Zwrot ku małemu ViT uczonemu od zera: przegląd literatury i projekt (15:12-15:21, `814201a9`)

**Skąd pytanie.** O 15:12 użytkownik wyciągnął wniosek z porażki bramek: „Czyli zmierzamy do tego, że rozwiązaniem byłoby to, że powinniśmy wziąć Vision Transformer jakiś i go uczyć od zera [...] może być jakiś typu RFDETR, albo jeszcze mniejszy Vision Transformer, który występuje gdzieś tam w OpenCV [...] zaproponuj mi jakiegoś Vision Transformera, który by się szybko uczył”. O 15:13 poprosił jeszcze o przypomnienie planu; asystent streścił cztery bramki planu sterowania (dwie pierwsze odpadły, więc trzeciej i czwartej nie zgłasza) oraz sprawy czekające na decyzję (plan domykania tożsamości na Large, miejsce na Edenie, brakujący bieg ziarna 43).

Vision Transformer (ViT) to sieć, która dzieli obraz na małe kawałki (tokeny) i przetwarza je blokami uwagi, w których każdy kawałek „patrzy” na wszystkie pozostałe i bierze z nich informację. Asystent sprawdził sprzęt: w środowisku `F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1` jest biblioteka timm 0.9.16 z gotowymi blokami ViT, lokalne OpenCV 4.11 ma tracker VitTrack, karta RTX 3070 ma 8 GB pamięci. Osobny proces przejrzał literaturę (Undermind, Scite).

**Literatura (zapisana w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\PLAN.md`):**
- Detekcja od zera na małych zbiorach się załamuje: He i in. 2019, „Rethinking ImageNet Pre-training”, przy 10 tys. obrazów uczenie od zera dogania wstępne uczenie (25,9 wobec 26,0 AP, ale 220 tys. zamiast 60 tys. iteracji), przy 1 tys. obrazów 3,5 wobec 9,9 AP. Projekt ma 1500 oznaczonych klatek.
- DETR na 1-3 tys. obrazów ledwo działa nawet z kręgosłupem z ImageNet: Wang i in. 2022 (arXiv:2203.09507), 11,5 AP na Cityscapes (2975 obrazów, 300 epok), około 3,8 AP na 1 % COCO; DETR potrzebuje 500 epok, Deformable DETR 50. Winne jest dopasowanie węgierskie i rzadki sygnał (jedna ramka to jeden cel).
- Mały ViT od zera działa, gdy dostanie wejście konwolucyjne: CCT-7/3x1 (Hassani 2021, arXiv:2104.05704), 3,76 mln parametrów, 96,5 % na CIFAR-10 od zera, 90 % w mniej niż 30 min na RTX 2080 Ti; SPT+LSA (Lee 2021) +3-4 punkty na Tiny-ImageNet; ConViT na 5 % ImageNet 47,8 wobec 34,8 % dla DeiT.
- Samonadzór na własnych danych pomaga najbardziej: El-Nouby 2021 (arXiv:2112.10740), wstępne uczenie na samym zbiorze docelowym Stanford Cars (8144 obrazy) 92,8 wobec 35,3 % od zera; Gani 2022, ViT 2,8 mln z DINO na tym samym zbiorze +6 punktów; Ramesh 2023 (MedIA), wykrywanie narzędzi chirurgicznych 60,0 -> 74,7 mAP przy 5 oznaczonych filmach; MAE dla lekkich ViT ma słabe górne warstwy na małych zbiorach (Wang 2023).
- Fazy uczenia są na początku: Redman 2024 (NeurIPS), widmo Koopmana z pierwszych 100 iteracji różni się od późniejszego i stabilizuje po około 700; Achille, okres krytyczny w pierwszych 40-60 epokach od zera.
- Dobór danych: brak dowodu, że od zera pomaga bardziej. RHO-LOSS przyspiesza w obu reżimach (CIFAR-10 od zera: 90 % w 47 zamiast 177 epok); Sorscher 2022, przy małej puli lepiej zostawiać łatwe przykłady; Gupta 2023 (tylko streszczenie), dla ViT od zera dobór podzbioru prawie nic nie daje. W planie zapisano: „To trzeba zmierzyć, nie założyć”.
- OpenCV: `cv2.TrackerVit` to tracker jednego obiektu (VitTrack, plik ONNX), nie detektor; wymaga ramki startowej i OpenCV go nie uczy. Osobny kod uczenia (repozytorium `lpylpy0514/VitTracker`: wymiar 48, 3 bloki, wejście z 4 konwolucji, 300 epok na dużych zbiorach śledzenia). Liczba około 0,18 mln parametrów to oszacowanie z rozmiaru pliku, niepotwierdzone.

**Propozycja (15:21).** Asystent zmienił stanowisko z 14:43: uczenie od zera ma sens, bo wtedy enkoder naprawdę się zmienia i Koopman ma co opisywać, ale nie RF-DETR (poniżej progu z pracy He; dopasowanie węgierskie dokłada własny szum) i nie VitTrack (inne zadanie). Zaproponował mały hybrydowy ViT, około 4-6 mln parametrów (w rozmowie „około 5 milionów”), napisany na miejscu:
- wejście: cała klatka zmniejszona z 720 × 480 do 480 × 320 (mediana ramki narzędzia 196 × 81 px przechodzi w około 131 × 54 px, czyli około 8 × 3 tokeny);
- **łodyga konwolucyjna**: 4 warstwy splotowe 3 × 3 z krokiem 2. Warstwa splotowa przesuwa po obrazie mały filtr i patrzy tylko na sąsiednie piksele, więc sieć od początku „wie”, że sąsiednie piksele należą do siebie; tego zwykły ViT uczy się długo. Wynik to siatka 30 × 20 = 600 tokenów;
- 6-7 bloków transformera, wymiar 192, 3 głowy uwagi (skala ViT-Tiny i CCT-7);
- **gęsta głowa** zamiast dekodera DETR: każdy token przewiduje, czy leży w nim środek narzędzia (mapa ciepła), i odległości do krawędzi ramki (styl CenterNet/FCOS). Jedna klatka daje wtedy 600 wskazówek zamiast jednej i nie ma dopasowania węgierskiego;
- dwa sposoby uczenia: A, nadzorowane od zera na 1500 klatkach; B, najpierw samonadzór MAE (sieć zgaduje zakryte kawałki własnych klatek) na około 10,6 tys. nieoznaczonych klatkach treningowych, z członem kontrastowym, potem dostrojenie jak A;
- rejestracja od kroku 0: tokeny na sondach co 10 aktualizacji, jakość wykrycia, mAP próbki walidacji, zmiana wag na blok;
- bramki (każda porażka to stop i raport): **Z0** model się uczy (ramię A, 3 ziarna, do 100 epok, mAP@0,5 próbki walidacji ≥ 0,5; jeśli nie, najpierw B); **Z1** latent się rusza i jest przewidywalny (zmiana wag enkodera znacznie powyżej 0,026 % na epokę; Langevin ≤ 0,8 × „brak zmiany” na 1 epokę); **Z2** latent mówi coś o wyniku (3 sposoby doboru × 3 ziarna, ρ ≥ 0,4, p < 0,05); **Z3** kampania 5 ziaren, pole pod krzywą: sterowanie większe od obu odniesień w ≥ 4 z 5 ziaren.
Uczciwe zastrzeżenie w planie: model od zera będzie słabszy niż dostrojony RF-DETR (0,90 mAP@0,5); to stanowisko do badania szybkości doboru klatek, nie docelowy detektor. Asystent ocenił Z0 na około 2 godziny lokalnej karty i czekał na zgodę.

### 4.6 Metoda użytkownika i uruchomienie pętli wersji 1 (15:25-15:44, `235dcd80`)

**Doprecyzowanie metody przez użytkownika (15:25).** Użytkownik opisał, co chce zrobić: „bierzemy model, trenujemy go od zera [...] z tych 1500 klatek, my mieliśmy, patrzeć po trzech epokach, w którą stronę zmierza latent [...] które elementy są odpowiedzialne za pełną tożsamość narzędzia [...] to są te wektory latentu, które [...] odpalają się tylko wtedy, gdy narzędzie jest na ekranie [...] coś takiego jak core latent, czyli jakby wektor, który zawsze się odpala wtedy, gdy mamy narzędzie na obrazku [...] może mieć różne długości, no bo to jest rzut z 3D na 2D, [...] różne kąty, rozbłyski [...] przy pomocy właśnie tego wzoru Koopmana albo Langevina [...] sprawdzić, które i w jaki sposób elementy latentu będą się rozwijać [...] jeżeli doszliśmy do 3-go checkpointu, no to jak to będzie się zachowywało w 4-tym, 5-tym, 6-tym”, a potem FastGAN-em wygenerować obrazy pasujące do przewidzianego latentu, znaleźć do nich prawdziwe klatki w zbiorze uczącym i dołożyć je do uczenia. Zakończył: „możesz to zapuścić! Także puść teraz na mojej karcie graficznej i rób monitoring”.

Warto zapamiętać słowa „wektory latentu” i „core latent, czyli jakby wektor”: w implementacji asystent przełożył je na listy pojedynczych neuronów, co wyszło na jaw dopiero o 18:16 (podrozdział 4.12).

**Implementacja** (katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\`):
- `model.py`, klasa `MalyViT`: łodyga 3 -> 24 -> 48 -> 96 kanałów (trzy warstwy splotowe 3 × 3 z krokiem 2, z normalizacją BatchNorm i funkcją GELU) i czwarta warstwa 96 -> 192 z krokiem 2 (łączny krok 16 px); 6 bloków timm, wymiar 192, 3 głowy, warstwa ukryta MLP 768 (4 × 192); stałe pozycje sin-cos 2D; głowa na każdy token: Linear 192 -> 192, GELU, Linear 192 -> 5 (logit środka narzędzia z początkową wartością odpowiadającą prawdopodobieństwu 0,01 oraz l, t, r, b); strata: ogniskowa strata CenterNet na mapie środków (strata, która zmniejsza wagę łatwych, dobrze już rozpoznanych komórek mapy) plus 2 × (1 - GIoU) na komórkach wewnątrz ramki (GIoU to uogólnione IoU, które daje użyteczny sygnał także dla ramek bez części wspólnej); dekodowanie: szczyty mapy w oknie 3 × 3, do 10 ramek. **2 926 493 parametrów** (w rozmowie „około 3 milionów”, w planie „4-6 mln”). **Latent**: 768 wyjść funkcji GELU w warstwie ukrytej MLP ostatniego bloku, dla każdego tokenu. **Neuron** oznacza tu jedną z tych 768 liczb;
- `dane.py`: baza 1500 klatek, próbka walidacji 3000 z 12 043 klatek walidacji (losowanie z ziarnem 0), pula 9136 klatek treningowych z indeksu wyszukiwania (bez bazy i kotwic), 192 sondy: 12 wycinków narzędzia × 16 wersji, 176 × 176 px, siatka 11 × 11 tokenów. Wersje: oryginał, obroty 90°, 180°, 270°, czarne tło, żółte tło, skala 0,8, narzędzie skrócone do 0,8, przesunięcia w lewo, prawo, górę, dół, narzędzie przyciemnione, rozjaśnione, lekka perspektywa (15 wersji z narzędziem) oraz `empty_black` (czarny obraz bez narzędzia, używany jako negatyw);
- `trening.py`: paczka 32, AdamW lr 5e-4, wd 0,05 (kara za duże wagi), rozgrzewka kroku uczenia 300 kroków, potem stały, obliczenia bf16 (liczby w skróconej precyzji, szybsze na karcie); odbicie poziome i jasność/kontrast ±20 %; latent sond co 10 kroków; mAP@0,5 (AP jednej klasy, 101 punktów) na próbce walidacji;
- `tozsamosc.py`, definicje użytkownika przełożone na neurony: neuron „zapala się” na sondzie, gdy jego aktywacja na tokenach narzędzia przekracza średnią negatywów + 2 odchylenia (negatywy: tło wszystkich sond i obszar narzędzia w `empty_black`); **rdzeń tożsamości**: neurony zapalone w każdej z 15 wersji na ≥ 75 % wycinków, przy zapaleniach na negatywach ≤ 10 %; **pełny latent**: neurony z zapaleniami na negatywach ≤ 10 % i na ≥ 30 % sond z narzędziem; **domknięcie wersji**: udział sond danej wersji, na których zapala się co najmniej połowa neuronów rdzenia (a gdy rdzeń jest pusty, pełnego latentu);
- `prognoza.py`: stan to aktywacje neuronów pełnego latentu (co najmniej 32; gdy pełny latent był mniejszy, Koopman śledził 32 neurony najbardziej wybiórcze, czyli najsilniej reagujące na narzędzie) na 180 sondach z narzędziem; odszumianie średnią z bloków po 5 zapisów (punkt co 50 kroków); PCA do 8 wymiarów (PCA to rzut na kilka kierunków największej zmienności danych); `koopman`: Δy(t+1) = Σ_{j=0..3} A_j Δy(t-j), bez hamowania; `langevin`: Δy(t+1) = K y(t) + Σ_{j=0..3} B_j Δy(t-j) + a_p (hamowanie do punktu stałego, pamięć 4 przyrostów, osobna stała a_p każdej sondy); współczynniki wspólne dla sond, regresja grzbietowa z karą względną 1e-2; cel dla FastGAN-a z modelu `langevin`, 3 obroty naprzód; oba modele raportują test wsteczny (dopasowanie bez ostatnich punktów, prognoza i porównanie z tym, co się stało) względem „braku zmiany”. Test na danych syntetycznych z nasyceniem: `langevin` 0,59-0,80 × „brak zmiany”;
- `generuj.py`: FastGAN, uczony wcześniej od zera na 1044 wycinkach narzędzia (ze 116 operacji, 50 000 kroków), użyty w wersji zamrożonej (uśrednione wagi EMA) z `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_narzedzie\wyniki\eden_20260923_50k\results_50k\checkpoint_050000.pth`. Najpierw **inwersja** kotwicy (szukanie wejścia z, dla którego generator odtwarza wycinek: 4 starty × 160 kroków optymalizatora Adam), potem 250 kroków gradientu po z w stronę przewidzianego latentu (obraz „target”) i, dla kontroli, w stronę obecnego latentu (obraz „current”), z karą za odejście od wyglądu kotwicy;
- `wybor.py`: obraz „target” każdej kotwicy trafia do wyszukiwarki w indeksie puli. Indeks opisuje 10 637 klatek treningowych 384 liczbami z sieci DINOv2 ViT-S/14 (wycinek kwadratowy wokół ramki narzędzia z czarnym dopełnieniem), a podobieństwo liczy iloczyn skalarny (FAISS `IndexFlatIP`). Z każdej listy 90 najbliższych klatek (top-90) brane są klatki na przemian z list 6 kotwic, czyli 6 z 12 oryginalnych wycinków o największej przewidzianej zmianie. Reguła różnorodności: żadnej klatki bliżej niż ±30 klatek tego samego filmu od bazy i od już dodanych; ta sama reguła w ramieniu `losowe`;
- `petla.py`: wspólna rozgrzewka 3 obroty × 500 kroków, potem 6 obrotów × 500 kroków w trzech ramionach (`koopman`, `losowe`, `bez`) z tego samego stanu (wagi i stan optymalizatora); 134 klatki na obrót; ocena co 100 kroków. Tryb `--proba`: 3 obroty rozgrzewki, 2 obroty po 60 kroków, 10 klatek, 2 kotwice, ocena co 30 kroków;
- `kampania.cmd`: ziarna 42-46 po kolei, logi `wyniki\petla_s*.log`.

**Kalibracja Z0** (`z0.py`, wynik `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\wyniki\z0_s42\Z0.json`; ziarno 42, sama baza, 6000 kroków, 563 s na RTX 3070). mAP@0,5 próbki walidacji (w nawiasie krok): 0,000 (0), 0,085 (250), 0,197 (500), 0,237 (750), 0,334 (1000), 0,319 (1250), 0,360 (1500), 0,394 (1750), 0,365 (2000), 0,312 (2250), 0,359 (2500), 0,364 (2750), 0,356 (3000), 0,377 (3250), 0,364 (3500), 0,385 (3750), 0,414 (4000), 0,366 (4250), 0,354 (4500), 0,362 (4750), 0,382 (5000), 0,414 (5250), 0,412 (5500), 0,423 (5750), 0,404 (6000); strata uczenia spadała do 0,29, co przy stojącym mAP oznacza nadmierne dopasowanie do 1500 klatek. **Bramka Z0 (≥ 0,5) niespełniona**; model się uczy i po około 1500 krokach wchodzi w reżim, w którym więcej danych powinno pomagać. Rdzeń tożsamości był pusty na każdym etapie, pełny latent miał 33-55 neuronów, najsłabsze wersje: czarne i żółte tło, obroty. Stąd długość obrotu 500 kroków i rozgrzewka 1500 kroków kończąca fazę wzrostu. (Plan podaje „0,39-0,42 po 3500-6000”, a streszczenie sesji „0,40-0,42 po 6000”; w pliku wartości między krokami 3500 i 6000 wahają się od 0,354 do 0,423.)

**Kryterium kampanii (ustalone przed wynikiem):** pole pod krzywą mAP@0,5 w obrotach 4-9: `koopman` większe niż `losowe` i niż `bez` w ≥ 4 z 5 ziaren; wtórnie domknięcie tożsamości i jakość prognozy względem „braku zmiany”.

**Uruchomienie.** Pętla ruszyła lokalnie o 15:43, mimo niespełnionej bramki Z0 i bez osobnego liczenia bramek Z1 i Z2, bo użytkownik wprost polecił uruchomić uczenie; wariant B (samonadzór MAE) nie był robiony. Asystent zgłosił o 15:44: „sam model od zera dochodzi do 0,42, więc nie osiągnął progu 0,5 z planu. Rdzenia tożsamości jeszcze nie ma”. Ziarno miało trwać około 35 minut, zaplanowano pięć.

### 4.7 Wersja 1 lokalnie: ziarno 42 i kontrola „target kontra current” (15:43-16:28, `d661a8ee`)

**Przebieg ziarna 42** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\wyniki\petla_s42.log`, łącznie 2202 s). mAP@0,5 na końcu kolejnych obrotów:

| Obrót | wspólna rozgrzewka | `koopman` | `losowe` | `bez` |
|---|---|---|---|---|
| k1 / k2 / k3 | 0,1666 / 0,2767 / 0,3932 | | | |
| k4 | | 0,3916 | 0,4103 | 0,3955 |
| k5 | | 0,4083 | 0,3811 | 0,4149 |
| k6 | | 0,4577 | 0,4307 | 0,3974 |
| k7 | | 0,4712 | 0,4039 | 0,3989 |
| k8 | | 0,4329 | 0,4110 | 0,4262 |
| k9 | | 0,4736 | 0,4486 | 0,4358 |

Obrót rozgrzewki trwał około 44 s, obrót `koopman` 237-282 s (FastGAN i wyszukiwanie), obroty `losowe` i `bez` 43-48 s. Zbiór uczący rósł o 134 klatki na obrót, z 1500 do 2304.

**Wyniki ziarna 42** (`PLAN.md`, sekcja „Wyniki lokalne”):
- pole pod krzywą mAP@0,5 (obroty 4-9): `koopman` 0,4415, `losowe` 0,4225, `bez` 0,3916; przewaga nad losowym +0,019 przy rozrzucie kolejnych ocen ±0,03, więc jedno ziarno nic nie rozstrzyga;
- dokładanie klatek w ogóle pomaga (`losowe` - `bez` = +0,031); model od zera jest „głodny danych”, więc dobór ma tu pole do działania (przy dostrajanym RF-DETR różnice były rzędu 0,003);
- rdzeń tożsamości pusty w każdym obrocie i ramieniu; pełny latent 28-62 neuronów; domknięcie wersji 0,18-0,38 bez trendu (średnio `koopman` 0,30, `losowe` 0,28, `bez` 0,26);
- test wsteczny prognozy na 1 obrót, stosunek do „braku zmiany”: `langevin` 0,84; 1,21; 1,14; 1,22; 0,87 (średnio około 1,06); czysty Koopman 4,30; 1,31; 1,04; 0,98; 0,81;
- FastGAN nie trafia w przewidziany latent: odległość obrazu „target” od celu 2,84 jednostki standaryzowanej przy odległości cel-obecny 2,34, czyli obraz jest od celu tak samo daleko jak od stanu obecnego.

**Kontrola „target kontra current”** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\eden\kontrola_target_current.py`, wynik `...\maly_vit_od_zera\wyniki\KONTROLA_TARGET_CURRENT_s42.json`). Pytanie: czy obraz narysowany pod prognozę prowadzi do innych prawdziwych klatek niż obraz narysowany pod stan obecny. Wynik: wspólnych klatek w listach top-90 średnio 18,9 z 90, w listach top-22 (około tyle z każdej listy trafia do dodanych 134 klatek) 2,6 z 22, podobieństwo DINOv2 obrazów target i current 0,78 (dla poszczególnych kotwic top-90 od 3 do 45 wspólnych, cosinus od 0,48 do 0,89). Nie wiadomo było, ile z tej różnicy daje prognoza, a ile losowość optymalizacji FastGAN-a.

**Przerwanie.** O 16:05 użytkownik zapytał o postęp; wtedy `koopman` doszedł w obrotach z 0,39 do 0,47, a bieg bez dokładania stał na około 0,36 (asystent zastrzegł, że dodanie jakichkolwiek klatek też może pomóc). O 16:26, gdy Eden już liczył, użytkownik polecił: „To anuluj liczenie lokalne i zostań na liczeniu na Edenie [...] opisać wnioski na podstawie tego, co lokalnie już się policzyło”. Ziarno 43 przerwano po pierwszym obrocie `koopman` i odrzucono (rozgrzewka 0,1452 / 0,2547 / 0,3214, `koopman` k4 0,3438).

### 4.8 Sprzątanie Edenu (15:45-16:25, `8df08a91`, `91e97450`)

**Prośba.** O 15:45 użytkownik napisał: „możesz usunąć wszystko z eden zostawiając tylko zbiory uczące. Przed usunięciem zrób analizę i mi powiedz co można usunąć?”. Kwota to limit miejsca na dysku klastra dla projektu (200 GB i 200 tys. plików), sprawdzany poleceniem `lfs quota -p 1299`.

**Analiza** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-28.md`, `8df08a91`, nic nie usunięto): stan 192,4 GB z 200 GB i 146 993 plików z 200 000; pełna lista plików (122 tys.) porównana z dyskiem lokalnym po nazwie, rozmiarze i elementach ścieżki, dla plików ≥ 1 MB. Grupy:
- A. zbiory uczące, zostają (około 12 GB), m.in. `tracking_static36.tar` (2,0 GB, 22 680 obrazów Cataract-LMM: pula i walidacja), `tracking_calibration.tar` (1,1 GB), sondy i manifesty, dane Favority projektu grantowego;
- B. do usunięcia bez straty (około 29 GB): `2026-09-24_koopman_gesta_historia` poza sondami (22,5 GB, kopia lokalna), `2026-09-22_dobor_cech_adaptacyjny` (1,2 GB), `~/.vscode-server` (4,1 GB), `~/.roboflow` (0,35 GB), pamięć podręczna (0,2 GB), rozpakowane duplikaty archiwów (3,0 GB);
- C. wyniki pośrednie tylko na Edenie (około 122 GB), m.in. `2026-09-27_petla_large` 35,0 GB, `2026-09-21_koopman_langevin` 26,1 GB, `2026-09-24_koopman_fastgan_petla` 23,1 GB, `2026-09-22_dobor_klatek_stan_sieci` 11,7 GB, `2026-09-25_kontynuacja_5epok` 8,4 GB, `2026-09-26_petla_uczenia` 7,0 GB, `2026-09-23_fastgan` 5,9 GB;
- D. projekt grantowy (około 21 GB, osobna decyzja) i E. narzędzia (środowisko Pythona 9,4 GB, pamięć podręczna, conda; zostają).
Wiszące od 24.09 zadanie 1810406 równoległej sesji wskazywało na jeden z katalogów do usunięcia.

**Wykonanie.** O 15:57 użytkownik wkleił akapit o 29 GB i 122 GB i dopisał „usuwaj to” (grupy B i C). Małe pliki wyników (poniżej 1 MB, pliki tekstowe poniżej 50 MB i archiwa kodu) pobrano do `F:\Studia\PhD_projekt\VIT\eden_archiwum\smallvit_runs_raporty_2026-09-28.tar.gz` (1,3 GB, 10 435 plików, spis sprawdzony: brak 0, zły rozmiar 0; w nim raport zadania 1817712_2, czyli brakującego biegu Large s43 `bez` k06). Końcowy FastGAN skopiowano do katalogu nowego eksperymentu (suma SHA256 zgodna) przed usunięciem `2026-09-23_fastgan`. Usunięto 14 348 plików według jawnej listy oraz `~/.vscode-server` i `~/.roboflow`. Z listy zdjęto dwa archiwa, które okazały się zbiorami uczącymi: `context_train.tar` (57 073 obrazy, 4,9 GB) i `eden_inputs.tar` (1044 wycinki narzędzia dla FastGAN-a). Zadanie 1810406 anulowano na prośbę użytkownika (16:14: „to zadanie możesz anulować!”). Stan po: **52,2 GB z 200 GB** (w opisie wykonania „było 193,2”, w analizie 192,4) i 109 429 plików (było 148 036, w analizie 146 993). Nie ruszano `~/grant_tft`, `~/venvs`, `~/.cache`, `~/.conda`. Zasada zapisana w pamięci asystenta: `eden-po-sprzataniu-2026-09-28.md`.

### 4.9 Wersja 1 na Edenie: dwie nieudane próby i kampania 1817935 (16:05-16:51, `7de1c7c7`, `c29df166`, `3a178c23`)

**Prośba.** O 15:57 użytkownik poprosił: „potem możesz przygotować ten mój lokalny trening na trening na edenie!”, a o 16:05, wklejając stan wolnych kart (`sfree`: m.in. `hopper` 8 z 8 wolnych kart), napisał: „JEŚLI MASZ MOŻLIWOŚĆ PUŚĆ TRENING TEGO MAŁEGO MOJEGO Vita na edenie!”.

**Wersja Eden** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\eden\`): kopie kodu ze ścieżkami ze zmiennych środowiskowych `MALY_VIT_WEJSCIA`, `MALY_VIT_DANE`, `MALY_VIT_CACHE`, `MALY_VIT_WYNIKI`; wyszukiwanie dokładne w numpy zamiast biblioteki FAISS. Przed zgłoszeniem sprawdzono na procesorze: te same piksele bazy, pula 9136, ten sam DINOv2 co w indeksie (sumy SHA256), identyczne listy top-90 jak FAISS dla trzech zapytań z kampanii lokalnej (`test_rownowaznosci.py`). Wejścia (`eden\wejscia`, 155 MB, poza gitem) z sumami w `WEJSCIA_SHA256.json`. Skrypt `petla_eden.slurm`: SLURM to system kolejkowania zadań na klastrze; zadanie tablicowe to jedna definicja uruchamiana dla wielu ziaren (ziarno = pierwszy argument + indeks tablicy), a zależność `afterok` każe kampanii czekać, aż zadanie próbne skończy się bez błędu. Parametry: partycje `hopper,short`, 1 GPU, 8 CPU, 64 GB, limit 2 h 30 min, `--signal=B:TERM@120`; sprawdzenie wejść przed startem; rozpakowanie `tracking_static36.tar` do pamięci `/dev/shm` węzła i sprzątanie jej pułapką (`trap`) na koniec, żeby nic nie zostało na węźle.

**Dwie nieudane próby i poprawki** (próba to krótki bieg sprawdzający, czy kod przechodzi wszystkie etapy pętli, zanim ruszy kampania):
- 1817930 (próba) padła z błędem `ModuleNotFoundError: No module named 'timm'`: środowisko Pythona na Edenie nie ma biblioteki timm. Poprawka: lokalna kopia timm 0.9.16 w `$MY/deps` i `PYTHONPATH`. Czekająca kampania 1817931 została anulowana.
- 1817932 (próba) padła w `wybor._vf` na `int(f)`: nazwy plików w adnotacjach były zapisane w stylu Windows, a `Path(...).stem` na Linuksie zwraca wtedy całą ścieżkę. Poprawka: zamiana ukośników wstecznych na zwykłe przed `Path(...).stem` w `dane.py` wersji Eden. Czekająca kampania 1817933 dostała stan `DependencyNeverSatisfied`; użytkownik zobaczył go o 16:17 w `squeue` i zapytał „co się dzieje?”. Asystent przyznał, że to jego błąd, i anulował 1817933.
- 1817934 (próba) przeszła całą pętlę z FastGAN-em i wyszukiwaniem (16:21-16:23, węzeł `hopper`, karta H100 PCIe); 1817935 (ziarna 47-56) liczyła się od 16:23 do około 16:42, 16-18 min na ziarno (plan podaje „około 16 min”), obrót `koopman` około 128 s, obroty `losowe` i `bez` około 20 s. Osiem ziaren liczyło się na H100 (`hopper`), dwa na A100 (`dgx-4`); plan podaje „hopper” dla wszystkich.
Małe pliki JSON, logi i obrazy FastGAN pobrano do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\wyniki_eden\`, a potem latenty `LATENT.npz` (210 plików, 5,3 GB, około 0,67 GB na ziarno; poza gitem). Analiza: `analiza.py wyniki wyniki_eden` -> `...\maly_vit_od_zera\wyniki\ANALIZA.json` i `ANALIZA.png` (11 ziaren: 42 lokalnie i 47-56).

**Wyniki: pole pod krzywą mAP@0,5 w obrotach 4-9 / mAP@0,5 na końcu (krok 4500):**

| Ziarno | `koopman` | `losowe` | `bez` |
|---|---|---|---|
| 42 | 0,4415 / 0,4736 | 0,4225 / 0,4486 | 0,3916 / 0,4358 |
| 47 | 0,4627 / 0,4898 | 0,4562 / 0,4853 | 0,4243 / 0,4129 |
| 48 | 0,4409 / 0,4453 | 0,4305 / 0,4455 | 0,3909 / 0,4558 |
| 49 | 0,4373 / 0,4774 | 0,4247 / 0,4338 | 0,3928 / 0,4164 |
| 50 | 0,4832 / 0,5061 | 0,4604 / 0,5177 | 0,4496 / 0,4556 |
| 51 | 0,4497 / 0,4963 | 0,4361 / 0,4645 | 0,4162 / 0,4934 |
| 52 | 0,4200 / 0,4890 | 0,3972 / 0,4201 | 0,3750 / 0,4170 |
| 53 | 0,4185 / 0,4745 | 0,4148 / 0,4409 | 0,3855 / 0,3802 |
| 54 | 0,4312 / 0,4638 | 0,4340 / 0,4958 | 0,4092 / 0,4362 |
| 55 | 0,4031 / 0,4873 | 0,4046 / 0,4820 | 0,3702 / 0,3511 |
| 56 | 0,4809 / 0,5239 | 0,4472 / 0,5256 | 0,4431 / 0,4545 |
| średnio (pole) | 0,4426 | 0,4298 | 0,4044 |

Testy na parach ziaren. Test Wilcoxona dla par sprawdza, czy różnice między dwoma ramionami w tych samych ziarnach są systematycznie dodatnie, biorąc pod uwagę ich wielkość; test znaków liczy tylko, ile różnic jest dodatnich. Wartość p to prawdopodobieństwo uzyskania takiego lub mocniejszego wyniku, gdyby ramiona w rzeczywistości były równe.
- `koopman` > `losowe` w 9 z 11 ziaren (przegrane: 54 i 55), średnio +0,013 (+3 % względnie); Wilcoxon p = 0,005, test znaków p = 0,065;
- `koopman` > `bez` w 11 z 11 (+0,038, p = 0,001); `losowe` > `bez` w 11 z 11 (+0,025, p = 0,001);
- kryterium z planu było ustalone dla 5 ziaren (≥ 4 z 5); plan uznaje je za spełnione w proporcji (9 z 11, 82 %), ale pole `kryterium_spelnione` w `ANALIZA.json` ma wartość `false`;
- przewaga nad losowym jest w każdym obrocie 4-9 (średnio od +0,006 do +0,021), bez wyraźnego wzrostu w czasie.

**Prognoza i generator:**
- prognoza nie bije „braku zmiany”: mediana `langevin` / „brak zmiany” w obrotach 5-9: 0,91; 0,99; 1,14; 1,05; 1,13 (poniżej 1 w 9, 6, 1, 2, 2 z 11 ziaren); czysty Koopman 1,10; 1,02; 1,04; 1,00; 1,04; w obrocie 4 prognozy nie było (historia 30 punktów za krótka); prognoza pogarsza się z czasem uczenia;
- FastGAN nie trafia w cel: mediana odległości obrazu „target” od celu 2,24 przy odległości cel-obecny 1,95; bliżej celu niż ta odległość tylko w 39 % par obraz-kotwica;
- ziarna, w których prognoza Langevina była gorsza, miały większy zysk nad losowym (Spearman +0,59, p = 0,056); przy 11 ziarnach to słaby sygnał, ale jego kierunek przeczy temu, że zysk bierze się z prognozy;
- rdzeń tożsamości pusty w każdym ziarnie, ramieniu i obrocie; po obrocie 9 pełny latent: `koopman` 39,7, `losowe` 36,4, `bez` 33,8 neuronów; domknięcie wersji 0,275 / 0,279 / 0,258.

**Interpretacja (16:46, 16:51).** Ramię `koopman` dokładało klatki trochę lepsze od losowych, powtarzalnie, ale nie ma dowodu, że to zasługa prognozy: prognoza była na poziomie „braku zmiany”, FastGAN nie dochodził do celu, a zysk nie rósł tam, gdzie prognoza była lepsza. Najprostsze wyjaśnienie: klatki szukane przez DINOv2 wokół obrazów narzędzia z sond są po prostu bardziej „narzędziowe” niż losowe. Rozstrzygnąć to miało ramię kontrolne `current` (klatki szukane tą samą drogą, ale z obrazu bez prognozy), które wymagało zgody.

### 4.10 Plan użytkownika na następne sprawdzenie i analiza wsteczna latentów 11 ziaren (16:33-17:05, `f7efafbf`, `e5897e27`)

**Plan użytkownika (16:33).** Użytkownik podyktował: „na początku to sieć nic nie wie i ona się bardzo powoli uczy, więc na początku [...] nie będzie w stanie nic przewidywać [...] przy pierwszych epokach [...]. Jak zacznie się uczyć, no to po prostu zaczyna się zawsze przewidywanie i dopiero wtedy możemy obserwować po prostu to, co wychodzi z tożsamości”, oraz: „pierwsza, druga epoka, no to są takie epoki, w których Koopman i Langevin mają prawo się pomylić [...] w kolejnych pomiarach [...] Koopman i Langevin powinny zyskiwać [...] musimy to porównywać do losowego zwiększania różnymi klatkami, albo nawet [...] dokładania kolejnych z rzędu klatek do uczenia”. Poprosił: „To jest plan na kolejne sprawdzenie, więc to sobie zapisz gdzieś”. Zapis: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\PLAN_NASTEPNE_SPRAWDZENIE.md` (słowa użytkownika dosłownie, cztery rzeczy do zmierzenia: jakość prognozy w funkcji czasu uczenia, narodziny tożsamości, sterowanie dopiero po starcie przewidywania, nowe ramię „kolejne z rzędu”).

**Prośba o analizę (16:53).** „zrób analizę wsteczną na latentach z planu następnego sprawdzenia”, z dopiskiem: „pierwsze trzy epoki to są takie właśnie trzy epoki, w których to narzędzie mniej więcej zostanie w iluś tam procentach dobrze wykrywane [...] ale wcześniej musimy w ogóle jeszcze przeprowadzić testy na wektory tożsamości [...] możemy puścić pierwsze trzy epoki na tych 1500 zdjęciach i dopiero od trzeciej epoki zacząć [...] dokładanie [...] Jeżeli twierdzisz, że to powinno być od jakiejś innej epoki, to sprawdź to, przeanalizuj i zaproponuj [...] powiedz mi, czy to ma sens”.

**Przeliczenie jednostek.** Paczka 32 i baza 1500 klatek dają **1 epokę = 47 kroków** (46,875). Rozgrzewka wersji 1 (1500 kroków) to już **32 epoki**, a obrót to około 10,7 epoki. Epoka to jedno przejście przez wszystkie klatki bazy.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\analiza_wsteczna.py`, bez karty graficznej, na `LATENT.npz` ziaren 42 i 47-56; zapis co 10 kroków, bloki po 5, czyli punkt co 50 kroków; dalszy ciąg po kroku 1500 z ramienia `bez`, dla porównania też z `losowe` i `koopman`): (1) narodziny tożsamości co 50 kroków (wielkość pełnego latentu i rdzenia, „odległość do rdzenia” najlepszego neuronu, domknięcie wersji, stabilność zbioru mierzona wskaźnikiem Jaccarda, czyli liczbą wspólnych neuronów podzieloną przez liczbę neuronów w sumie obu zbiorów); (2) jakość prognozy zależnie od chwili startu t (co 250 kroków), z doborem neuronów i standaryzacją tylko z danych do t, na horyzontach 1, 5, 10, 30 bloków (50, 250, 500, 1500 kroków), wobec „braku zmiany” i „trendu” (ostatnie nachylenie z 5 bloków); (3) kierunek: czy prognoza trafia, które neurony urosną, a które zmaleją; (4) mAP@0,5 z ocen co 100 kroków. **Reguła wyboru chwili startu ustalona przed liczeniem:** najwcześniejsza chwila t, w której Langevin albo Koopman na horyzoncie 10 bloków ma medianę błędu po 11 ziarnach poniżej „braku zmiany” i „trendu”, w co najmniej 8 z 11 ziaren poniżej „braku zmiany”, i to samo zachodzi w co najmniej 3 z 4 kolejnych chwil.

**Wyniki** (`...\maly_vit_od_zera\wyniki\ANALIZA_WSTECZNA.json`, `ANALIZA_WSTECZNA.png`):
- trafność wykrywania (mediana 11 ziaren; poziom końcowy po 96 epokach 0,436): 3 epoki (141 kroków) około 0,04, czyli około 9 % poziomu końcowego; 11 epok 0,20 (47 %); 15 epok 0,28 (64 %); 21 epok 0,30 (69 %); 32 epoki 0,387 (89 %);
- tożsamość nie rodzi się, tylko kurczy: pełny latent największy na starcie (80-88 neuronów przy 1-4 epokach, w sieci prawie losowej), 47 przy 32 epokach, 33 przy 96; „odległość do rdzenia” najwyższa na starcie (0,58), potem 0,33; próg rdzenia 0,75 nigdy nieosiągnięty (pojedynczy neuron rdzenia tylko w 1-2 epoce jednego ziarna). Miara wybiórczości łapie na starcie prostą różnicę pikseli (jasny metal kontra tkanka);
- zbiór neuronów tożsamości wymienia się cały czas: Jaccard z chwilą sprzed 250 kroków (około 5 epok) około 0,45 od około 15 epoki do końca, czyli co 5 epok wymienia się połowa neuronów; Jaccard ze zbiorem końcowym 0,07 (start), 0,21 (32 epoki), 0,31 (64 epoki);
- prognoza nie zaczyna działać w żadnej chwili: Koopman / „brak zmiany” około 1,00 we wszystkich chwilach i horyzontach; Langevin 1,02-1,11 na horyzoncie 500 kroków i gorzej na 1500; na krótkich horyzontach (50-250 kroków) pod koniec (75-91 epok) mediana 0,97-0,98 w 8-9 z 11 ziaren, czyli zysk 2-3 %; „trend” 2-5 razy gorszy od „braku zmiany”;
- kierunek (które neurony urosną): zgodność znaku 0,50-0,62 (rzut monetą to 0,50), Spearman 0,0-0,2;
- reguła niespełniona; jedyne „trafienie” (Koopman, `bez`, krok 4000, czyli 85 epok) to ostatnia chwila bez punktów do potwierdzenia, czyli artefakt reguły.

**Sprawdzenie dodatkowe: wektor na wszystkich 768 neuronach** (średnia na narzędziu minus średnia negatywów, w jednostkach odchylenia negatywów; mediana 11 ziaren, ramię `bez`): cosinus z chwilą sprzed 250 kroków 0,72 (11 epok), 0,85 (16 epok), 0,89 (32 epoki), 0,90 (64 epoki), czyli od około 16 epok wektor jest stabilny, dużo stabilniejszy niż lista neuronów; cosinus z wektorem końcowym (96 epok) 0,20 (1 epoka), 0,41 (11), 0,59 (32), 0,75 (64), 0,83 (80), czyli powolny, gładki dryf przez cały trening; zgodność kierunków liczonych osobno dla 15 wersji około 0,65 i stała. Ramiona `losowe` i `koopman` dają te same liczby (±0,05).

**Odpowiedź dla użytkownika (17:05).** Pomysł „najpierw nauczyć, potem dobierać” ma sens i był już zrealizowany (dokładanie startowało po 32 epokach). Trzy epoki w dosłownym sensie to za wcześnie (sieć znajduje wtedy około 9 % tego, co na końcu). Prognoza nie działa w żadnym momencie 1-96 epok, więc przesunięcie startu tego nie naprawi. Przeszkodą jest definicja tożsamości jako listy pojedynczych neuronów; stabilny jest wektor na 768 neuronach. Asystent zaproponował start dokładania nie wcześniej niż około 20 epoki (wykrywanie około 70 % poziomu końcowego) i sprawdzenie prognozy na wektorze bez karty graficznej.

**Czego to nie dowodzi.** Analiza jest wsteczna, na latentach jednej definicji tożsamości i jednego horyzontu obrotu; nie wyklucza, że inny stan (np. inaczej wyznaczony wektor) byłby przewidywalny.

### 4.11 Wersja 2: wektor tożsamości na 768 neuronach, start po 21 epokach (Eden 1817948/1817949; 17:13-17:48, `ddfeae69`, `1ded306b`)

**Zgoda i zmiany.** O 17:13 użytkownik napisał: „Dobra to puść trening na edenie i pomiar na edenie i zobaczmy jak to wyjdzie”. Asystent założył nowy eksperyment `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_wektor_tozsamosci\` i zmienił metodę względem wersji 1 (różnice opisane w nagłówku `kod\petla.py`):
1. stan prognozy to aktywacje narzędzia wszystkich 768 neuronów na 180 sondach, standaryzowane średnią i odchyleniem negatywów z chwili prognozy (zamiast listy neuronów pełnego latentu); cel FastGAN-a też na 768 neuronach;
2. rozgrzewka 2 obroty × 500 kroków = 1000 kroków = 21 epok bazy (było 32), potem 7 obrotów w ramionach, razem 4500 kroków jak w wersji 1; dodanych klatek 7 × 134 = 938 (było 804);
3. horyzont prognozy 1 obrót (500 kroków) zamiast 3, bo klatki dodane teraz działają w najbliższym obrocie;
4. pomiar w każdym ramieniu i obrocie: stabilność wektora tożsamości (cosinus z poprzednim obrotem i z końcem rozgrzewki), test wsteczny na 768 neuronach i na liście neuronów z wersji 1, oraz **sprawdzenie prognozy z poprzedniego obrotu na tym, co się naprawdę stało** (bez podglądania przyszłości), ze zgodnością znaku zmiany neuronów.
Ramiona bez zmian (`koopman`, `losowe`, `bez`), ziarna 47-56 (pary z wersją 1). Nowe ramiona `current` i „kolejne z rzędu” nie weszły, bo czekały na osobną zgodę. W odpowiedzi o 17:19 asystent nazwał dwie główne zmiany (wektor na 768 neuronach, start po 21 epokach), ale nie zaznaczył, że odchodzi od metody użytkownika (tożsamość ze zdjęć sond po 3 epokach, prognoza na tej tożsamości); wyszło to godzinę później (podrozdział 4.12).

**Sprawdzenie przed kampanią** (`test_wektor_offline.py`, wynik `...\maly_vit_wektor_tozsamosci\wyniki\TEST_WEKTOR_OFFLINE_v1.json`): prognoza na 768 neuronach na latentach wersji 1 (ramię `bez`, 11 ziaren, chwile 1000-4000 kroków co 250, horyzont 500 kroków): błąd wektora / „brak zmiany” Koopman 1,00-1,04, Langevin 0,99-1,06 (poniżej 1 najwyżej w 7 z 11 ziaren); błąd sond Koopman 1,00-1,02, Langevin 1,03-1,06; zgodność znaku 0,45-0,59. Asystent uprzedził użytkownika uczciwie: „lokalnie sprawdziłem ten sam wektor na starych latentach i jego prognoza nie była lepsza od założenia „nic się nie zmieni””, bo zmiana wektora w ciągu jednego obrotu to głównie przypadkowe wahania uczenia. To ostrzeżenie padło o 17:19, już po zgłoszeniu zadań; zgodą było wcześniejsze „puść trening na edenie” z 17:13.

**Kryteria ustalone przed wynikiem:** główne, pole pod krzywą mAP@0,5 w obrotach 3-9: `koopman` > `losowe` i > `bez` w ≥ 8 z 10 ziaren; prognoza działa, jeśli w sprawdzeniu z poprzedniego obrotu mediana błędu wektora / „brak zmiany” < 1 oraz < 1 w ≥ 8 z 10 ziaren, w co najmniej 4 z 6 obrotów 4-9; porównanie z wersją 1 średnim mAP@0,5 w oknie kroków 1600-4500 w ramieniu `koopman`.

**Eden.** Katalog `~/smallvit_runs/2026-09-28_maly_vit_wektor/`, wejścia i timm z katalogu wersji 1, skrypt `...\maly_vit_wektor_tozsamosci\petla_eden.slurm` (te same zasoby co w wersji 1). Próba 1817948 (17:19-17:23), kampania 1817949 (17:23-17:47, około 20-24 min na ziarno; 8 ziaren na H100, 2 na A100). W trakcie asystent raportował sprawdzenia prognozy obrót po obrocie: Koopman lepszy od „braku zmiany” w 7 z 10 ziaren (typowo o około 1 %), potem w 4, 3 z 10, a w czwartym sprawdzeniu praktycznie równy „brakowi zmiany” we wszystkich ziarnach; Langevin w pierwszym sprawdzeniu przegrał w 6 z 10.

**Pytanie użytkownika w trakcie (17:37).** „no ale dlaczego koopman źle przewiduje zmiany w latencie? Może nie powinien przewidywać tak daleko? może źle się liczy?”. Asystent zaczął liczyć diagnostykę, a użytkownik przerwał wywołanie narzędzia. Odpowiedź bez nowego liczenia (17:42): to nie wina odległości, bo sprawdzono horyzonty 50-1500 kroków i nawet przy 50 Koopman nie wygrywa; zmiana latentu w obrocie to głównie przypadkowe wahania uczenia. Asystent zgłosił też niesprawdzone podejrzenie błędu w liczeniu: Koopman pracuje na 8 głównych kierunkach PCA wybranych według różnic między sondami, a nie według zmian w czasie, więc może w ogóle nie widzieć ruchu i dlatego przewiduje prawie dokładnie „brak zmiany”. To podejrzenie zostało do końca etapu niesprawdzone.

**Wyniki** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_wektor_tozsamosci\wyniki_eden\ANALIZA.json` i `POMIAR.json`; pole pod krzywą w obrotach 3-9 / mAP@0,5 na końcu):

| Ziarno | `koopman` | `losowe` | `bez` |
|---|---|---|---|
| 47 | 0,4553 / 0,4931 | 0,4433 / 0,5073 | 0,4271 / 0,4622 |
| 48 | 0,4358 / 0,4927 | 0,4358 / 0,4970 | 0,3914 / 0,3697 |
| 49 | 0,4369 / 0,4824 | 0,4444 / 0,4947 | 0,3949 / 0,4297 |
| 50 | 0,4455 / 0,4925 | 0,4559 / 0,4956 | 0,4153 / 0,4291 |
| 51 | 0,4418 / 0,5461 | 0,4255 / 0,4767 | 0,3910 / 0,4333 |
| 52 | 0,4438 / 0,5294 | 0,4276 / 0,4825 | 0,4016 / 0,3819 |
| 53 | 0,4515 / 0,4817 | 0,4116 / 0,4882 | 0,3850 / 0,4238 |
| 54 | 0,4695 / 0,5315 | 0,4651 / 0,5212 | 0,4279 / 0,4980 |
| 55 | 0,3890 / 0,4543 | 0,3869 / 0,4551 | 0,3555 / 0,3239 |
| 56 | 0,4625 / 0,4862 | 0,4446 / 0,4741 | 0,4089 / 0,4095 |
| średnio (pole) | 0,4432 | 0,4341 | 0,3999 |

- **Kryterium główne formalnie spełnione, ale słabo:** `koopman` > obu odniesień w 8 z 10 ziaren; nad losowym +0,009 (Wilcoxon p = 0,064, test znaków p = 0,11). Jedna z ośmiu wygranych (ziarno 48) to różnica 0,00005 (0,43584 wobec 0,43579). Nad `bez` +0,043 w 10 z 10 (p = 0,002); samo dokładanie losowych klatek daje +0,034 (10 z 10, p = 0,002).
- **Prognoza wektora nie działa (0 z 6 obrotów spełnia kryterium).** Ramię `koopman`, sprawdzenie prognozy z poprzedniego obrotu: mediana błędu wektora / „brak zmiany” dla Koopmana 0,989; 1,018; 1,006; 1,004; 1,003; 0,999 (obroty 4-9), poniżej 1 w 7, 4, 3, 4, 4, 5 z 10 ziaren; Langevin 1,003-1,037; zgodność znaku zmiany neuronów 0,52-0,61. W ramionach `losowe` i `bez` to samo.
- Wektor tożsamości między obrotami: cosinus 0,84-0,86, czyli kierunek stabilny, ale jego zmiana w obrocie nieprzewidywalna.
- **Wersja 2 kontra wersja 1** (średnie mAP@0,5 w krokach 1600-4500, 10 par ziaren): `koopman` 0,4427 -> 0,4565 (+0,014, lepsza w 7 z 10), `losowe` +0,015 (7 z 10), `bez` +0,003 (6 z 10). Zysk pochodzi z wcześniejszego i większego dokładania (start po 21 epokach, 938 zamiast 804 klatek), bo `losowe` zyskało tyle samo co `koopman`; przewaga Koopmana nad losowym nie wzrosła (+0,013 w wersji 1, +0,009 w wersji 2).

**Interpretacja (17:48).** Ramię `koopman` znów dobierało klatki minimalnie lepiej niż losowanie, ale prognoza, na której ma się opierać, była przez całą kampanię równa „brakowi zmiany”, więc przewaga nie może z niej pochodzić. Najprostsze wyjaśnienie z wersji 1 (klatki szukane wokół obrazów narzędzia) pozostało.

### 4.12 Spór o metodę i tożsamość po 21 epokach: neurony, wektor, siła tożsamości (18:02-18:30, `6eb7fff0`, `b7a3b308`, `8deaf528`)

**Rozmowa, która ujawniła odejście od metody.** O 18:02 użytkownik zapytał: „A czy przeprowadzałeś po trzech epokach co jest tożsamością narzędzia w LATENCIE?”. Asystent odpowiedział, że tożsamość liczyła się po każdym obrocie i w analizie wstecznej co 50 kroków, że rdzeń nie powstał w żadnym z 11 ziaren, że po trzech obrotach rozgrzewki (32 epokach) pełny latent miał 47 neuronów, a po 3 epokach w dosłownym sensie (141 krokach) około 85 neuronów reagujących tylko na różnicę jasnego metalu i tkanki. O 18:03 użytkownik zapytał, jak w ogóle testować, co jest latentem tożsamości; asystent przyznał, że obecny test jest za słaby, bo przechodzi go nawet sieć nieuczona, i zaproponował trzy testy: szukać kierunku w latencie zamiast pojedynczych neuronów i sprawdzać go na wersjach, których nie widział; próbę przyczynową (usunąć ten kierunek z latentu i zobaczyć, czy sieć przestaje wykrywać narzędzie); powtórzenie na sieci nieuczonej. O 18:05 użytkownik przypomniał testy „wygenerowane przez FastGAN-a” na czarnym i żółtym tle z obrotami, na których po trzech epokach miał wyznaczyć, „które wektory detektora Vision Transformera się zaświecą”, i „na tych wektorach próbować liczyć Kalmana i przewidywać przyrost latentu, albo [...] hamowanie”. Asystent sprostował, że sondy nie pochodzą z FastGAN-a, tylko z prawdziwych klatek z ręcznie zaznaczonym narzędziem (obracanych albo z podmienionym tłem), i że Kalmana jeszcze nie liczył.

Kolejne pytania (18:07-18:10): „ok to wyznaczyłeś tożsamość?”, „i na tej tożsamości liczyłeś koopmana?”, „To miałaś wyznaczyć tożsamość na podstawie tych zdjęć i potem przewidywać koopmanem latent tożsamości!!! Czy tak to zrobiłeś?”, „no to w drugiej kampanii miałeś nauczyć sieć do 3 epoki i potem przeprowadzić testy tożsamości! tak to zrobiłeś?”. Asystent przyznał po kolei: w wersji 1 Koopman liczył na liście neuronów pełnego latentu (dopełnianej do 32 neuronów i wyznaczanej od nowa w każdym obrocie), w wersji 2 na wszystkich 768 neuronach, czyli nie na tożsamości wyznaczonej ze zdjęć; tożsamość wyznaczał po 32 epokach, a nie po 3; nie trzymał jej stałej, choć użytkownik chciał ją wyznaczyć raz i śledzić. O 18:11: „Zmieniłem metodę po swojej analizie wstecznej i Twoje „puść trening” wziąłem za zgodę na te zmiany. To był mój błąd, bo nie powiedziałem wprost, że odchodzę od Twojego planu”. Z tego powstała stała reguła zapisana w pamięci asystenta (`feedback-nie-podmieniac-metody-uzytkownika.md`): przed zgłoszeniem zadania wypisać każdą różnicę względem planu użytkownika i prosić o zgodę na tę konkretną zmianę; gdy dane przemawiają przeciw jego metodzie, najpierw uruchomić jego wersję albo obie obok siebie; „epoka” w ustach użytkownika może znaczyć obrót pętli.

**Krok A: tożsamość po 21 epokach metodą użytkownika (neurony).** Polecenie o 18:11: „no to po 21 epochach wyznacz tożsamość teraz!”. Skrypt `...\maly_vit_wektor_tozsamosci\tozsamosc_21ep.py`, krok 1000 (koniec rozgrzewki wersji 2), średnia 5 ostatnich zapisów, 192 sondy, definicje z `kod\tozsamosc.py`. Wyniki: `...\maly_vit_wektor_tozsamosci\wyniki_eden\TOZSAMOSC_21EP.json` i `TOZSAMOSC_21EP.png` (wykres: który neuron zapala się w której wersji).
- Pełny latent 45-87 neuronów (średnio 66,4). **Rdzeń: 0 w każdym z 10 ziaren.** Najlepszy neuron zapala się we wszystkich 15 wersjach na 25-50 % wycinków (próg rdzenia 75 %).
- Domknięcie wersji (średnio 10 ziaren): rozjaśnione narzędzie 0,44, obrót 180° 0,42, przyciemnione 0,40, oryginał 0,39, lekka perspektywa 0,37, obrót 90° 0,34, skrócone 0,32, przesunięcie w prawo 0,31, obrót 270° 0,30, przesunięcie w dół 0,275, w górę 0,27, w lewo 0,24, skala 0,8 0,18, **czarne tło 0,03, żółte tło 0,00**. Po 21 epokach neurony tożsamości prawie nie zapalają się na narzędziu na sztucznym tle, czyli sieć rozpoznaje narzędzie razem z tłem operacyjnym.
- Bez czarnego i żółtego tła (13 wersji, odejście od definicji, tylko informacyjnie): próg 75 % w każdej wersji spełniają 2 neurony w ziarnie 52 i 1 w ziarnie 55, w pozostałych 0; próg 50 % spełnia 1-5 neuronów na ziarno.
- Sieć nieuczona (krok 0) przechodzi ten sam test lepiej: 231-329 neuronów „pełnego latentu”; 26-58 neuronów z listy po 21 epokach było na niej już w sieci nieuczonej.

**Krok B: wektor tożsamości po 21 epokach.** O 18:16 użytkownik zapytał: „dlaczego neurony a nie wektory latentu?”. Asystent przyznał: „Bo tak przełożyłem Twoją definicję: „wektor, który zawsze się odpala” zamieniłem na zbiór pojedynczych neuronów [...] To był mój wybór, nie Twój, i najpewniej zły, bo sieci zwykle zapisują pojęcie jako kierunek rozłożony na wiele neuronów naraz”. Polecenie o 18:17: „to policz wektor tożsamości po 21 epokach teraz”. Skrypt `...\maly_vit_wektor_tozsamosci\wektor_tozsamosci_21ep.py`, wyniki `...\wyniki_eden\WEKTOR_21EP.json`, `WEKTOR_21EP.npz` (wektory 768 na ziarno), `WEKTOR_21EP.png`.

**Wektor tożsamości** to kierunek w 768-wymiarowej przestrzeni latentu, wzdłuż którego latent narzędzia najbardziej różni się od latentu tła; każdy obraz dostaje jedną liczbę, rzut swojego latentu na ten kierunek. Wyznaczano go metodą **LDA ze ściąganiem Ledoit-Wolfa**: LDA (liniowa analiza dyskryminacyjna) szuka kierunku, w którym średnie dwóch grup (narzędzie i tło) są najdalej od siebie w stosunku do rozrzutu wewnątrz grup; do tego potrzebna jest macierz kowariancji 768 × 768, której z kilkuset przykładów nie da się dobrze oszacować, więc „ściąga” się ją w stronę prostej macierzy diagonalnej z wagą dobraną automatycznie z danych (wzór Ledoita i Wolfa). Próg „zapalenia” wektora ustawiono jak dla neuronów: średnia rzutów negatywów + 2 odchylenia. **AUC** oznacza tu pole pod krzywą ROC: prawdopodobieństwo, że losowo wybrana sonda z narzędziem dostanie wyższy rzut niż losowo wybrany negatyw (0,5 to rzut monetą, 1 to rozdzielenie bez pomyłek). Cztery testy: wersja wyłączona (wektor uczony bez danej wersji, sprawdzany na niej), wycinek wyłączony (narzędzie z wycinka, którego wektor nie widział), prosty wektor (średnia narzędzia minus średnia tła) i sieć nieuczona.

Wyniki:
- wersja wyłączona: narzędzie zapala wektor w 100 % wycinków we wszystkich wersjach poza żółtym tłem (85 %); AUC 1,00 (żółte 0,93). **Rdzeń wektorowy** według progu użytkownika (≥ 75 % w każdej wersji wyłączonej) w **8 z 10 ziaren**; ziarna 49 i 50 zawodzą na żółtym tle (42 % i 50 %, AUC 0,75 i 0,67);
- wycinek wyłączony: 100 % zapaleń, AUC 1,00 w każdym ziarnie;
- tło fałszywie zapala wektor w wersjach wyłączonych w 0-41 % przypadków (najwięcej obrót 90°, średnio 41 %, i przesunięcia: rogi i krawędzie po przekształceniu), przy AUC 1,00, więc problemem jest próg, a nie kierunek;
- prosty wektor słabszy, najsłabsze czarne tło (77 %);
- **sieć nieuczona daje to samo albo lepiej**: 100 % we wszystkich wersjach poza żółtym (97 %) i mniej fałszywych zapaleń tła. Test na tych sondach nie odróżnia więc tożsamości wyuczonej od zwykłej różnicy wyglądu (jasny, cienki metal kontra tkanka).
Wniosek zapisany w planie: wektor po 21 epokach istnieje i przenosi się na nowe wersje i nowe narzędzia, ale do zmierzenia tożsamości wyuczonej potrzebne są trudniejsze negatywy (odblaski, krawędzie tkanek, klatki bez narzędzia) albo próba przyczynowa.

**Krok C: Koopman i Langevin na stałym wektorze.** Polecenie o 18:21: „na tym wektorze policz teraz Koopmana”. Skrypt `...\maly_vit_wektor_tozsamosci\koopman_na_wektorze.py` na latentach wersji 2 pobranych z Edenu (210 plików). Wektor i standaryzacja stałe z kroku 1000. **Siła tożsamości** sondy to rzut jej latentu narzędzia (po standaryzacji) na wektor tożsamości, czyli jedna liczba mówiąca, jak mocno sieć widzi na tej sondzie narzędzie, a nie tło. Stan prognozy to siły 180 sond; chwile startu 1000-4250 co 250 kroków; horyzonty 50, 250, 500 kroków; porównanie z prawdziwą przyszłością w każdym ramieniu. Wyniki: `...\wyniki_eden\KOOPMAN_WEKTOR.json` i `KOOPMAN_WEKTOR.png`.

Pierwsza, wstępna odpowiedź o 18:24 była liczona na 3 ziarnach (reszta latentów się pobierała): Langevin o około 15 % lepszy od „braku zmiany” na 500 kroków i trafny kierunek w 85 %. Użytkownik dopytał: „czyli dążymy do 100% mniejszy?” (odpowiedź: nie, część zmian jest przypadkowa, zależy od kolejności zdjęć i losowych przekształceń; ważniejsze jest trafienie kierunku) i „czyli prognoza jak bardzo się myli albo w ilu procentach jest poprawna? w 75?” (odpowiedź: są dwie miary, kierunek 85 % wobec 50 % rzutu monetą, a błąd 85 % błędu „braku zmiany”; w ramionach z dokładaniem kierunek 59-72 %). O 18:30 asystent poprawił liczby na 10 ziarnach: kierunek 69 %, błąd o 10 % mniejszy.

Wyniki końcowe (10 ziaren, mediany, `_srednio` w `KOOPMAN_WEKTOR.json`):

| Ramię | Błąd sond / „brak zmiany”, 500 kroków: Koopman (odsetek przypadków < 1) | Langevin | Błąd siły średniej: Koopman / Langevin | Trafiony kierunek zmiany siły: Koopman / Langevin |
|---|---|---|---|---|
| `bez` | 0,94 (85 %) | 0,90 (75 %) | 0,95 / 0,86 | 64 % / 69 % |
| `losowe` | 0,93 (85 %) | 0,92 (69 %) | 0,95 / 1,01 | 67 % / 55 % |
| `koopman` | 0,94 (88 %) | 0,94 (64 %) | 0,99 / 1,04 | 54 % / 58 % |

Na horyzontach 50 i 250 kroków stosunki były podobne (0,90-0,95). Siła tożsamości wzdłuż wektora z 21 epok maleje z czasem: średnio 6,13 w kroku 1000 i 5,30 w kroku 4500 w ramieniu `bez`, 4,36 w ramionach `losowe` i `koopman`; Langevin widzi to jako hamowanie (współczynnik k około -0,16 na blok): tożsamość obraca się od kierunku z 21 epok, szybciej przy dokładaniu klatek. **Po raz pierwszy prognoza bije „brak zmiany” powtarzalnie, ale niewiele** (o 6-10 %).

### 4.13 FastGAN do przewidzianego latentu: diagnoza chybiania, głębsze warstwy, sieć MLP (18:28-19:10, `5f498d37`)

**Jak ustalano, co ma przewidywać prognoza.** O 18:28 użytkownik zapytał: „czyli w 85 procent jest ok? to teraz trzeba generować zdjęcia które spełniają te przewidywania [...] wytrenować kilka epok, potem znowu koopman albo langevin przewiduje latent [...] Rozumiesz, powtórz jak to ma być!”. Asystent powtórzył pętlę (uczenie od zera do 21 epoki; stały wektor tożsamości ze zdjęć sond; prognoza zmiany siły na każdej sondzie; FastGAN generuje zdjęcia o przewidzianej sile; wyszukanie podobnych prawdziwych klatek; kilka epok uczenia; powtórka; porównanie z losowymi klatkami i brakiem dokładania) i poprosił o potwierdzenie dwóch założeń: do zbioru idą prawdziwe klatki, nie wygenerowane obrazy; jeden cykl to 500 kroków, czyli około 10 epok bazy. O 18:32 użytkownik doprecyzował: „Koopman i Langevin przewidują, jak zmieni się latent w kolejnych epokach [...] w szczególności tożsamości ma się zmienić, albo wyhamować, albo przyspieszyć, wydłużyć, skrócić [...] Langevin albo Koopman, w zależności od tego, co przewidzi lepiej [...] powinniśmy ten przewidywany latent w całości wziąć, na podstawie tego latentu wygenerować obrazy”. Gdy asystent dodał, że poza kierunkiem tożsamości latent przewiduje się jak „brak zmiany”, więc przewidywany latent będzie się różnił głównie długością tożsamości, użytkownik zaprotestował (18:34): „nie wiem tego! Nie mogę tego zakładać! FastGAN musi w jakimś stopniu dochodzić do zadanego latentu. Wcześniej to robiliśmy w pętli, aż gradient tak wysteruje wejścia FastGAN-a, że on wygeneruje obraz odpowiadający przewidywanemu latentowi przez Koopmana!”. O 18:38: „cały przewidywany latent weźmiemy z Koopmana - na razie tylko z koopmana! [...] to trzeba koniecznie naprawić - może trzeba jakąś sieć skonstruować do tego, małą MLP?”. Asystent zaproponował połączenie: MLP daje punkt startu, a gradient dokręca obraz do celu, ale najpierw test przyczyny chybiania. MLP (perceptron wielowarstwowy) to mała sieć złożona z kilku warstw w pełni połączonych; tu miała się nauczyć przepisu odwrotnego: z zadanego latentu od razu podać wejście FastGAN-a. O 18:40 użytkownik: „no to teraz to zaprojektuj i testuj. Możesz próbować na moim gpu! sprawdzaj i ucz mlp żeby ładnie tłumaczyła latent VIT - (pełny latent) na wejścia do fastgan”. Opis: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_odwrotnosc_mlp\PLAN.md`.

**Problem wyjściowy.** W wersjach 1 i 2 FastGAN sterowany gradientem (250 kroków optymalizatora Adam po wejściu z; strata: odległość latentu obrazu od przewidzianego latentu, plus wierność obrazowi kotwicy i bliskość z) kończył średnio tak samo daleko od celu jak od stanu wyjściowego (wersja 1 lokalnie: 2,84 przy odległości cel-obecny 2,34; na Edenie bliżej celu tylko w 39 % par; w planie FastGAN-a te 39 % przypisano wersji 2, w planie wersji 1 kampanii Eden 1817935).

**Ustawienie wspólne** (`wspolne.py`): mały ViT wersji 1, ziarno 42, krok 1000 (21 epok, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\wyniki\petla_s42\wspolny\k02.pt`); pełny latent to 768 aktywacji uśrednionych po tokenach narzędzia, w jednostkach odchylenia negatywów z kroku 1000; cel to prognoza **Koopmana** na 500 kroków dla 12 kotwic na wszystkich 768 neuronach (funkcja `prognoza_bloku` z wersji 2); FastGAN 50k zamrożony, ze stałym szumem warstw; **postęp** = 1 - odległość(latent obrazu, cel) / odległość(latent startu, cel), 0 to brak ruchu, 1 trafienie. Pomocniczo: część zmiany „wzdłuż” zadanego kroku i część „w bok” (prostopadła do niego). Liczono lokalnie na RTX 3070 środowiskiem `F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1`.

**Krok 1: przyczyna** (`diagnoza.py`, `diagnoza2.py`; wyniki `...\fastgan_odwrotnosc_mlp\wyniki\DIAGNOZA.json`, `DIAGNOZA2.json`, siatki PNG). Przed sterowaniem wyszły dwie przyczyny dawnego chybiania:
- cel był bezwzględny, a sieć jest pojedynczym stanem: prognoza dotyczy średniej 5 zapisów (bloku), a pojedynczy stan sieci w kroku 1000 różni się od tej średniej o 0,84 jednostki, podczas gdy cały krok prognozy to 0,33;
- obraz z inwersji nie jest kotwicą dla ViT: odtwarza ją na oko (różnica pikseli L1 0,074, czyli około 7 % jasności), ale jego latent leży 2,19 od latentu kotwicy, prawie 7 razy dalej niż krok prognozy.
Poprawka: cel = latent obrazu startowego (inwersji) + przewidziany **krok**, na stałych, binarnych tokenach narzędzia (0,16 od zapisu sond, tokeny miękkie dawały 0,24).

Postęp do celu (mediana 12 kotwic; próba 2 ze stałymi tokenami):

| Wariant sterowania | Postęp | Wzdłuż | W bok | Zmiana obrazu L1 |
|---|---|---|---|---|
| A: dawne ustawienia (250 kroków, wierność kotwicy, bliskość z) | 0,00 (w próbie 1 z ruchomą maską -0,75) | | | |
| B: samo wejście z, bez hamulców (w planie 1000 kroków, w wynikach 400) | 0,04 | | | |
| B z krokiem 3 razy większym | 0,05 | | | |
| C: z + przesunięcie cech 8 × 8 generatora, z karą za maskę | 0,20 | | | |
| C1: jak C bez kary, 800 kroków | 0,23 | 0,51 | 0,56 | 0,060 |
| D: z + cechy 8 × 8 + cechy 16 × 16 | 0,36 | 0,63 | 0,49 | 0,075 |
| P: same piksele (górna granica) | 0,61 | 0,84 | 0,30 | 0,022 |
| cele na pewno osiągalne (latent innego obrazu FastGAN-a, odległość około 2,22): B / C | 0,51 / 0,83-0,86 | | | |

„Cechy 8 × 8” i „16 × 16” to wewnętrzne mapy generatora (512 kanałów na siatce 8 × 8 i 256 na 16 × 16); sterowanie nimi daje więcej swobody niż 256 liczb wejścia. Wniosek: sterowanie działa, ale kierunek przewidziany przez Koopmana wymaga zmian w głębi generatora, a samo wejście z go nie wyraża. W trakcie asystent raportował użytkownikowi, że stare ustawienie odepchnęło obraz od celu (mediana -75 % drogi w próbie 1) i że stare pętle z celem bezwzględnym przechodziły 28 % drogi.

**Krok 2: sieć sterująca MLP** (`mlp.py`, wynik `...\wyniki\MLP.json`, `MLP_koopman.png`; wagi `MLP.pt` poza gitem). Pierwotny projekt (768 -> 1024 -> 1024 -> 256, strata cyklu) zmieniono po kroku 1: wejście to latent obrazu startowego l0 (768) i zadany krok Δ (768), wyjście to poprawka wejścia δz (256) i współczynniki 2 × 64 wyuczonych map poprawek cech 8 × 8 i 16 × 16. Strata: ‖latent(G(z0 + δz; cechy + poprawki)) - (l0 + Δ)‖² / ‖Δ‖² na stałych tokenach plus małe kary. Dane: starty z losowego z; kroki Δ z banku 14 400 prawdziwych zmian latentu sond (wersja 1, ziarno 42, horyzont 500 kroków) i 2520 prognoz Koopmana z innych chwil niż krok 1000 (cele oceny wyłączone z uczenia). Po wybuchu straty wprowadzono: zerowy start głowic, obcięcie straty, krok uczenia 2e-4, tylko obrazy z co najmniej 4 tokenami narzędzia, mieszankę 70/30 kroków Koopmana i prawdziwych; paczka miała 12 przykładów. **Cel ustalony przed wynikiem:** mediana postępu ≥ 70 % na 12 celach Koopmana po starcie z MLP i dokręceniu, przy obrazie nadal wyglądającym jak narzędzie.
Wyniki: błąd względny w uczeniu spadł z około 4,4 do około 1,0 (1 to tyle, co bez ruchu), z dużymi skokami (np. 1,76 po 500 krokach, 1,06 po 700, 2,05 po 900), i skończył na 1,03 po 2000 krokach. Na 12 celach Koopmana: sama sieć, postęp 0,01 (zmiana obrazu 0,003: nauczyła się ostrożności, prawie nie rusza obrazu); sieć + 200 kroków dokręcania gradientem, 0,24; samo dokręcanie 200 kroków, 0,25. **Sieć nie pomaga, cel 70 % nieosiągnięty.**

**Stałe kierunki edycji** (`kierunki.py`): kroki Koopmana leżą w 99,6 % w 16 kierunkach latentu (kroki oceny z kroku 1000: 99,2 %), więc zamiast 768 kierunków wystarczyłoby nauczyć generator 16. Uczenie 16 wspólnych dla wszystkich obrazów edycji generatora rozbiegło się dwa razy (błąd względny 4-19 od początku, także przy starcie od zera i mniejszym kroku uczenia) i zostało przerwane; jedna edycja nie działa tak samo na różnych obrazach.

**Wygląd obrazów** (`...\wyniki\DIAGNOZA2.png`: kotwica, inwersja, z + 8 × 8, z + 8 × 8 + 16 × 16, piksele): odtworzenie przez FastGAN oddaje ogólny układ (oko i położenie narzędzia), ale nie szczegóły; sterowanie głębszymi warstwami zostawia narzędzie, ale psuje tło (plamy, drobne jasne punkty), a sterowanie pikselami daje obraz prawie identyczny z inwersją, z drobnym szumem: latent się zmienia, a oko tego nie widzi.

**Wniosek kroku 2 (19:10).** Przewidziana przez Koopmana zmiana latentu to w większości zmiana „niewidoczna”: da się ją wymusić drobnym szumem pikseli (61 % drogi przy zmianie obrazu o 2 %), ale FastGAN nie umie narysować jej jako naturalnej zmiany wyglądu (samo wejście 4 %, głębsze warstwy do 36 % z artefaktami). Najlepsze dostępne narzędzie: dokręcanie gradientem na warstwach 8 × 8 i 16 × 16 obrazu z inwersji, z celem względnym. Asystent zgłosił możliwą zmianę metody wymagającą decyzji: szukać prawdziwych klatek wprost po latencie ViT, bez FastGAN-a.

### 4.14 Literatura („Show Me What You Don't Know”) i sterowanie odporne (19:26-19:54, `cf97cf3a`)

**Prośba.** O 19:26 użytkownik: „możesz to zrobić! Przez Paper Search my znaleźliśmy taką pracę Show me what you don't know. Ja bym chciał tę pracę wykorzystać, bo oni właśnie jakby rysowali to, czego sieć jeszcze nie wie [...] zrób research, przeczytaj ten artykuł, przeczytaj podobne artykuły [...] Underminda możesz użyć i powiedz mi jak użyć tego FastGAN-a, bo to był dobry pomysł, żeby go używać i szukać po prostu podobnych obrazów”. Nie jest jasne, do czego odnosiło się „możesz to zrobić!” (do proponowanego wyszukiwania klatek wprost po latencie?); w źródłach tego etapu nie ma śladu realizacji takiego wyszukiwania.

**Literatura** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_odwrotnosc_mlp\NOTATKA_LITERATURA.md` w wersji z commitu `cf97cf3a`; późniejsze dopiski z 29.09 o Stable Diffusion należą do etapu 5; pełne teksty [Rou26] i [Vat25] czytane przez Undermind):
- **Rousselot, Wendebourg, Köthe 2026, „Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation”** (arXiv:2603.21782): zbiór niezmienniczy („włókno”) to wszystkie obrazy, które sieć widzi tak samo; autorzy próbkują z niego obrazy realistyczne. Dwa sposoby: generator warunkowy uczony dodatkową „stratą włókna” (dokładny, ale drogi, około 14 000 min na A40 dla ImageNet) albo bez uczenia, model dyfuzyjny sterowany w trakcie odszumiania (NDTM) z dynamicznym celem (porównanie z tak samo zaszumioną i odszumioną referencją), około 1,8 min na próbkę. Próbki są znacznie bliżej włókna niż najbliższy sąsiad w zbiorze (DINOv2 na ImageNet: 273 wobec 1225). Najważniejsze dla projektu (dodatek A): dopasowanie reprezentacji gradientem w pikselach daje szum w stylu ataku, więc poprawki trzeba robić w przestrzeni semantycznej generatora; to samo wyszło w kroku 1 (piksele 61 % drogi przy niewidocznej zmianie). Autorzy nie celują w reprezentację przesuniętą o prognozę i nie używają GAN-ów.
- **Väth i in. 2025, „Diffusion Classifier Guidance for Non-robust Classifiers”** (arXiv:2507.00687): sterowanie dyfuzją siecią nieodporną (jak mały ViT) bez poprawek nie działa; działa po liczeniu sieci na jednokrokowej estymacie czystego obrazu i wygładzeniu gradientu średnią wykładniczą (β = 0,99); CelebA: trafność klasy 56 % -> 99,99 %.
- **Nguyen i in. 2016** (arXiv:1605.09304): optymalizacja wejścia generatora jako wzorca naturalnych obrazów zamiast pikseli; to jest obecne podejście z FastGAN-em. **Santurkar i in. 2019**: gradienty sieci odpornych są „percepcyjnie wyrównane”, zwykłych nie; wyjaśnia, dlaczego krok Koopmana jest w dużej części niewidoczny. **Bansal i in. 2023, „Universal Guidance for Diffusion Models”**: sterowanie dyfuzji dowolną siecią bez jej przeuczania. Generowanie zapytania i wyszukanie prawdziwych próbek: Mayer i Timofte (WACV 2020), a prawdziwe wyszukane obrazy lepsze od syntetycznych: Geng i in. 2024, Burg i in. 2023; schemat „FastGAN -> podobne prawdziwe klatki” ma więc oparcie w literaturze.

**Test sterowania odpornego** (`sterowanie3.py`, wynik `...\fastgan_odwrotnosc_mlp\wyniki\STEROWANIE3.json`, `STEROWANIE3.png`; 12 celów Koopmana, 400 kroków, z + cechy 8 × 8 + 16 × 16). „Postęp odporny” to postęp policzony na latencie uśrednionym po 4 zaburzonych kopiach obrazu (jasność i kontrast ±15 %, szum σ 0,02, rozmycie); zmiana, która znika przy drobnym zaburzeniu, jest „niewidoczna”. Dyskryminator FastGAN-a (sieć, która w trakcie uczenia GAN-a oceniała, czy obraz jest prawdziwy) oceniał prawdziwe kotwice wyżej niż inwersje (-0,486 wobec -1,130), ale czysty szum jeszcze wyżej, więc jest wiarygodny tylko blisko danych.

| Wariant | Postęp czysty (wzdłuż) | Postęp odporny (wzdłuż, w bok) | Zmiana obrazu | Ocena dyskryminatora |
|---|---|---|---|---|
| D0 bez ochrony | 0,29 (0,59) | 0,15 (0,54; 0,57) | 0,068 | -1,208 |
| D_E cel na zaburzonych kopiach | 0,01 (0,48) | 0,19 (0,49; 0,58) | 0,080 | -1,385 |
| D_ER + kara dyskryminatora | -0,06 (0,56) | 0,12 (0,53; 0,68) | 0,108 | +3,425 |
| P_E piksele, cel odporny | 0,11 (1,04) | 0,49 (0,79; 0,38) | 0,028 | -1,237 |

Połowa zmiany z D0 znika przy drobnym zaburzeniu obrazu; ochrona podnosi część odporną tylko z 15 do 19 %; kara dyskryminatora została oszukana (ocena -1,13 -> +3,43, silne artefakty) i odrzucona; piksele dają drobne barwne ziarno na całej scenie zamiast naturalnej zmiany narzędzia. **Wniosek kroku 3:** żaden wariant FastGAN-a nie daje naturalnego obrazu, który realizuje krok Koopmana w pełnym latencie (około 30 % drogi kosztem plam, 15-19 % odpornie). Zgodnie z [Rou26] lepszym narzędziem byłby mały model dyfuzyjny (generator, który rysuje obraz stopniowo, usuwając szum krok po kroku) uczony na wycinkach narzędzia i sterowany z dynamicznym celem i stabilizacją jak w [Vat25]; to godziny GPU na Edenie i osobna zgoda. FastGAN z celem względnym miał zostać szybkim wariantem.

**Decyzja użytkownika (19:52-19:54).** Użytkownik napisał: „Na Edenie model dyfuzyjny będziemy trenować, jeżeli tutaj się wszystko uda z Koopmanem i Langevinem”, i zapytał, czy prognoza dała poprawne efekty i czy MLP jest wyuczona. Asystent odpowiedział, że Koopman i Langevin przewidują lepiej niż „brak zmiany” tylko siłę wektora tożsamości (o 6-10 %, kierunek w około dwóch trzecich przypadków), pełnego latentu nie; MLP nie jest wyuczona do dobrego poziomu; testy FastGAN-a celowały w prognozę pełnego latentu, która sama nie jest trafna; następny krok to sterować FastGAN-em tylko wzdłuż wektora tożsamości o zmianę siły przewidzianą przez Langevina. O 19:54 użytkownik: „Ale one nie miały [...] przewidywać całego latentu, tylko właśnie tożsamość narzędzia przewidywać. [...] tak zrób”. To doprecyzowanie stoi w napięciu z jego słowami z 18:32 i 18:38 („przewidywany latent w całości wziąć”, „cały przewidywany latent weźmiemy z Koopmana”), na których opierały się kroki 1-3. Zapisano je jako regułę w pamięci asystenta (`koopman-przewiduje-tylko-tozsamosc.md`): cel prognozy i sterowania to siła wektora tożsamości z 21 epok, a pełny latent służy tylko do pomiaru zmian ubocznych („w bok”).

### 4.15 FastGAN sterowany wzdłuż wektora tożsamości o zmianę siły z Langevina (19:57-20:05, `6f6bbf06`)

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_odwrotnosc_mlp\wzdluz_tozsamosci.py`, lokalnie na RTX 3070, 155 s). Sieć: mały ViT wersji 1, ziarno 42, krok 1000. Wektor tożsamości jak w `wektor_tozsamosci_21ep.py` (rozdzielacz narzędzie/tło na sondach z kroku 1000). Historia siły 180 sond z kroków 50-1000 (bloki po 5 zapisów, 20 bloków), prognoza 10 bloków (500 kroków) naprzód Langevinem, dla porównania także Koopmanem. **Sterowanie wzdłuż wektora** znaczy: zmieniać obraz tak, żeby zmieniła się tylko jego siła tożsamości, a reszta latentu prawie wcale. Cel obrazu: siła obrazu startowego (inwersji kotwicy) + przewidziana zmiana; słaba kara za zmianę latentu w bok; 300 kroków gradientu, zachowany najlepszy stan. Warianty: Z_1x (samo wejście z), D_1x (z + cechy 8 × 8 i 16 × 16), D_1x_E (cel liczony na zaburzonych kopiach), D_3x (zmiana 3 razy większa, żeby było ją widać). Miary: „część” = osiągnięta zmiana siły / przewidziana (1 to dokładnie), „odporna” = to samo na 4 zaburzonych kopiach, „w bok” = norma zmiany latentu poza kierunkiem tożsamości podzielona przez wartość bezwzględną zmiany siły.

**Prognoza.** Siła tożsamości kotwic około 6,27. Langevin przewidział spadek na 11 z 12 kotwic, średnio około 0,7 (zmiany: -0,42; -1,083; -0,726; -0,806; -0,831; -0,165; -0,458; -1,115; -0,415; +0,104; -0,102; -1,848); Koopman zmiany kilka razy mniejsze (od -0,434 do +0,086). **Sprawdzenie na prawdziwym przebiegu** (`sprawdz_prognoze_1500.py`, wspólny odcinek biegu do kroku 1500, `...\wyniki\SPRAWDZ_PROGNOZE_1500.txt`): prawdziwe zmiany kotwic -0,95; -0,89; -1,12; -1,11; -1,23; -1,97; -0,48; -1,11; -0,19; +0,81; -1,58; -1,96, czyli siła rzeczywiście spadła na 11 z 12.

| Model | Kotwice: błąd / „brak zmiany” (zysk), znak | Wszystkie 180 sond: błąd / „brak zmiany” (zysk), znak |
|---|---|---|
| Langevin | 0,35 / 1,11 (54 %), 12 z 12 | 0,65 / 0,96 (16 %), 75 % |
| Koopman | 0,89 / 1,11 (10 %), 75 % | 0,87 / 0,96 (6 %), 67 % |

To jedno ziarno; potwierdza kierunek wyniku z 10 ziaren wersji 2 (6-10 %), ale go nie zastępuje.

**Sterowanie** (`...\wyniki\WZDLUZ.json`, `WZDLUZ.png`: kotwica, inwersja, Z_1x, D_1x, D_1x_E, D_3x; mediany 12 kotwic):

| Wariant | Część zmiany | Odporna | Znak | W bok | Zmiana obrazu L1 |
|---|---|---|---|---|---|
| Z_1x (samo z) | 1,00 | 0,85 | 12 z 12 | 35,3 | 0,068 |
| D_1x (z + 8 × 8 + 16 × 16) | 1,00 | 0,94 | 12 z 12 | 40,7 | 0,092 |
| D_1x_E (cel na zaburzonych kopiach) | 0,99 | 1,11 | 12 z 12 | 37,8 | 0,100 |
| D_3x (3 razy większa zmiana) | 1,00 | 0,93 | 12 z 12 | 13,4 | 0,098 |

Odniesienie dla zmiany w bok (`odniesienie_bok.py`, `...\wyniki\ODNIESIENIE_BOK.txt`): w prawdziwym uczeniu stosunek „w bok / |zmiana siły|” wynosi 58,4 (kotwice 61,0) na 50 kroków, 58,9 (60,8) na 250 kroków i 74,1 (78,3) na 500 kroków, czyli obrazy z FastGAN-a zmieniają resztę latentu mniej niż zwykły krok uczenia (w rozmowie podano „60 do 78”, w planie „58-78”).

**Wygląd.** Z_1x prawie nie różni się od inwersji (zmiana realna, ale subtelna). D_1x, D_1x_E i D_3x obniżają siłę tożsamości, poszarpując narzędzie (łamana, ziarnista krawędź) i dodając jasne drobiny w tle, więc spadek „narzędziowości” jest widoczny, ale częściowo przez artefakty.

**Wniosek (20:05) i zastrzeżenia.** Jeden kierunek FastGAN realizuje: przewidziana zmiana siły jest osiągana w całości, ze znakiem zgodnym na wszystkich kotwicach, i w 85-111 % przetrwa zaburzenia obrazu (wobec 15-19 % dla pełnego latentu w kroku 3). Najczystszy obraz daje samo wejście z. Zastrzeżenia: (1) inwersja sama zmienia siłę tożsamości o ±3 (od 2,5 do 9,5 wobec około 6,2 dla kotwic), więcej niż przewidziana zmiana, więc cel jest względny, ale obraz startowy nie jest kotwicą; (2) warianty z głębszymi warstwami osiągają cel częściowo artefaktami; (3) to jedna sieć (wersja 1, ziarno 42, krok 1000). Następny krok, wymagający zgody: obrazy Z_1x albo D_1x jako zapytania do puli.

### 4.16 Wersja 3: pętla siły tożsamości z pięcioma ramionami (Eden 1818024/1818025; 21:28-22:35, `d289a8ad`, `831bf563`, `d66e6eb8`, `b67127d2`)

**Skąd.** O 21:28 użytkownik uznał: „Czyli mamy już dobry schemat. Możemy puścić testy na edenie”. Asystent odpowiedział (21:32), że jedno ogniwo nie jest sprawdzone: nie wiadomo, czy obraz z prognozą znajduje w puli inne prawdziwe klatki niż obraz bez prognozy. Dlatego zaproponował pięć ramion: cztery użytkownika i jedno własne, kontrolne, nazwane wprost jako spoza listy użytkownika. Wymienił też wybory, które nie wynikają wprost ze słów użytkownika, i czekał na „tak”. O 21:42 użytkownik: „Na edenie trenuj”. Plan: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sila_tozsamosci\PLAN.md`.

**Pętla (schemat użytkownika, zapisany też w `ALGORYTM.md` na jego prośbę z 22:00: „Zapisz w punktach jak jest wykonywany obecny algorytm?”):**
1. Mały ViT uczy się od zera na bazie 1500 klatek przez 1000 kroków (21 epok); rozgrzewka wzięta gotowa z wersji 2, te same ziarna 47-56.
2. Co 10 kroków sieć ogląda 192 sondy i zapisuje 768 liczb na narzędziu i 768 na tle.
3. Z zapisów tuż przed krokiem 1000 liczony jest wektor tożsamości (LDA ze ściąganiem Ledoit-Wolfa: narzędzie 180 sond kontra tło wszystkich 192 sond i obszar narzędzia w `empty_black`, po standaryzacji średnią i odchyleniem negatywów). Wektor zostaje stały do końca biegu.
4. Siła tożsamości każdej ze 180 sond z narzędziem, historia uśredniana w bloki po 50 kroków.
5. Na początku każdego z 7 obrotów (po 500 kroków, około 10,6 epoki): sprawdzenie prognozy z poprzedniego obrotu; dopasowanie Koopmana i Langevina do historii bez ostatnich 500 kroków i wybór tego, który lepiej przewidział te 500 kroków; prognoza zwycięzcy dla 180 sond na 500 kroków naprzód.
6. Ramię `tozsamosc`: z 12 kotwic 6 o największej przewidzianej wartości bezwzględnej zmiany siły; inwersja kotwicy w FastGAN-ie; 300 kroków sterowania samym wejściem z, aż siła obrazu = siła startu + przewidziana zmiana, przy możliwie małej zmianie reszty latentu; obraz do DINOv2, lista 90 najbardziej podobnych prawdziwych klatek z puli (reguła ±30 klatek od bazy i od dodanych); 134 klatki na przemian z 6 list.
7. 500 kroków uczenia na powiększonym zbiorze; mAP@0,5 co 100 kroków na 3000 klatkach walidacji.
8. Powrót do punktu 5; razem 7 obrotów, 938 dodanych klatek, koniec na kroku 4500.

**Ramiona** (ten sam stan po rozgrzewce, ta sama liczba klatek):
- `tozsamosc`: metoda;
- `current`: kontrola zaproponowana przez asystenta: to samo co `tozsamosc`, ale do wyszukiwania idzie obraz startowy z inwersji, bez przesunięcia siły (w `PLAN.md` opisane jako „FastGAN sterowany do obecnej siły, zmiana 0”, w kodzie `sila.generuj(..., sterowac=False)` i w `ALGORYTM.md` jako obraz startowy bez sterowania). Różnica `tozsamosc` - `current` pokazuje, ile daje sama prognoza. Ważne: w obu ramionach prognoza wybiera, które 6 z 12 kotwic wziąć;
- `losowe`: 134 losowe klatki z puli na obrót, reguła ±30;
- `kolejne` (nowe, z listy użytkownika): 134 klatki puli najbliższe w czasie po klatkach bazy (i po już dodanych kolejnych) w tym samym filmie, bez reguły ±30, bo sensem ramienia są klatki sąsiednie (zwykle 5 klatek dalej);
- `bez`: bez dokładania.

**Wybory nazwane przed zgodą:** wektor wyznaczony raz, po 21 epokach, i stały (tak był sprawdzany); horyzont i obrót 500 kroków (użytkownik mówił „kilka epok”, tu około dziesięciu); sterowanie samym wejściem z (najczystszy obraz); rozgrzewka z wersji 2 skopiowana twardymi dowiązaniami (ten sam plik widoczny pod drugą nazwą, bez kopiowania danych i bez ponownego uczenia; pary z wersjami 1 i 2); LDA z Ledoit-Wolfem napisane w numpy, bo na Edenie nie ma biblioteki sklearn.

**Kryteria ustalone przed wynikiem** (skrypt `analiza.py`, `831bf563`): główne, pole pod krzywą mAP@0,5 w obrotach 3-9: `tozsamosc` > `losowe`, > `kolejne` i > `bez` w ≥ 8 z 10 ziaren; rola prognozy: `tozsamosc` > `current` w ≥ 8 z 10 ziaren, inaczej zysk nie pochodzi z prognozy; prognoza w pętli działa, jeśli błąd siły / „brak zmiany” < 1 w ≥ 8 z 10 ziaren w ≥ 4 z 6 obrotów 4-9; FastGAN działa w pętli, jeśli mediana odpornej części zmiany ≥ 0,5 i zgodność znaku ≥ 0,8 w każdym obrocie.

**Sprawdzenia przed zgłoszeniem.** `test_lda_lokalnie.py` -> `...\maly_vit_petla_sila_tozsamosci\wyniki\TEST_LDA.json`: wektor z numpy (`kod\sila.py`) zgodny z sklearn, cosinus 1,000 w kroku 1000 i w kroku 0, największa różnica siły 1,2e-6 i 2,7e-6; siła narzędzia 6,20 wobec negatywów 0,00 (odchylenie negatywów 0,22). Próba lokalna na RTX 3070 (ziarno 42, rozgrzewka z wersji 1 o tych samych krokach 0-1000, tryb `--proba`: 2 obroty po 50 kroków, 10 klatek, 2 kotwice): wszystkie 5 ramion przeszło w 170 s; FastGAN w `tozsamosc`: część zmiany 1,00, odporna 0,96-1,00; wspólne top-90 z obrazem startowym 22-50 z 90, czyli prognoza zmienia wyszukane klatki; `kolejne` bierze klatki 5 klatek po klatkach bazy. Jedyna zmiana wspólnego kodu: `kod\generuj.py` czyta ścieżkę wag FastGAN-a ze zmiennej `MALY_VIT_FASTGAN_CKPT` (tylko dla próby lokalnej, bo lokalnie tych wag nie było w katalogu wejść).

**Eden.** Katalog `~/smallvit_runs/2026-09-28_maly_vit_sila/`, skrypt `...\maly_vit_petla_sila_tozsamosci\petla_eden.slurm` (`hopper,short`, 1 GPU, 8 CPU, 64 GB, limit 3 h, sprzątanie `/dev/shm` pułapką). Próba 1818024 (ziarno 47, 21:52-21:58), kampania 1818025 (ziarna 47-56, `afterok`, 21:58-22:28; 15-25 min na ziarno; pięć ziaren naraz, reszta czekała na karty; obrót metody około 40 s). W trakcie asystent raportował: w ostatnim obrocie prognoza siły była o 2-10 % lepsza od „braku zmiany” w pierwszych czterech ziarnach, o 16 % w piątym, w jednym z kolejnych o 26 %; w dwóch ziarnach metoda dodała w ostatnim obrocie tylko 105 i 110 klatek zamiast 134, bo zabrakło podobnych klatek na listach (w wersji 2 było tak samo: 87-134 w ostatnim obrocie). Pliki: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sila_tozsamosci\wyniki_eden\` (JSON-y i logi; latenty, wagi, obrazy FastGAN-a i listy wyszukiwania `zapytanie_*.json` na Edenie albo lokalnie poza gitem, `b67127d2`), analiza `wyniki_eden\ANALIZA.json`, diagnoza po wyniku `wyniki_eden\DIAG_SILA.json`.

**Wyniki kampanii 1818025: pole pod krzywą mAP@0,5 w obrotach 3-9 / mAP@0,5 na końcu (krok 4500):**

| Ziarno | `tozsamosc` | `current` | `losowe` | `kolejne` | `bez` |
|---|---|---|---|---|---|
| 47 | 0,4513 / 0,4605 | 0,4688 / 0,5140 | 0,4643 / 0,5135 | 0,4464 / 0,4548 | 0,4221 / 0,5062 |
| 48 | 0,4433 / 0,5107 | 0,4482 / 0,4642 | 0,4259 / 0,4682 | 0,4255 / 0,4562 | 0,3916 / 0,4225 |
| 49 | 0,4387 / 0,5028 | 0,4504 / 0,4784 | 0,4403 / 0,4998 | 0,4248 / 0,4558 | 0,4058 / 0,4311 |
| 50 | 0,4536 / 0,5175 | 0,4348 / 0,5119 | 0,4323 / 0,4788 | 0,4458 / 0,4850 | 0,4026 / 0,4467 |
| 51 | 0,4462 / 0,5200 | 0,4318 / 0,5017 | 0,4170 / 0,4708 | 0,4106 / 0,4715 | 0,4020 / 0,4631 |
| 52 | 0,4528 / 0,5000 | 0,4454 / 0,5066 | 0,4168 / 0,4405 | 0,4245 / 0,4959 | 0,3995 / 0,4316 |
| 53 | 0,4350 / 0,5401 | 0,4304 / 0,4649 | 0,4314 / 0,4776 | 0,4180 / 0,5070 | 0,3827 / 0,3997 |
| 54 | 0,4671 / 0,5720 | 0,4710 / 0,5159 | 0,4664 / 0,5280 | 0,4470 / 0,4291 | 0,4336 / 0,4574 |
| 55 | 0,3751 / 0,3929 | 0,3918 / 0,4059 | 0,3868 / 0,4573 | 0,3718 / 0,4172 | 0,3616 / 0,3618 |
| 56 | 0,4694 / 0,5127 | 0,4558 / 0,5500 | 0,4462 / 0,4688 | 0,4405 / 0,4360 | 0,4287 / 0,4342 |
| średnio | 0,4432 / 0,5029 | 0,4428 / 0,4913 | 0,4327 / 0,4803 | 0,4255 / 0,4608 | 0,4030 / 0,4354 |
| klatek dodanych średnio | 930 (929,8) | 929 (928,7) | 938 | 938 | 0 |

Rozrzut między ziarnami (odchylenie standardowe pola, policzone przeze mnie z tej tabeli): 0,026 / 0,023 / 0,024 / 0,023 / 0,022, czyli kilka razy więcej niż różnice między ramionami.

- **Kryterium główne: niespełnione** (`tozsamosc` pokonała wszystkie trzy odniesienia w 7 z 10 ziaren, wymagane 8). `tozsamosc` - `losowe` +0,0105, lepsza w 7 z 10 (Wilcoxon p = 0,11, test znaków p = 0,34); - `kolejne` +0,0177, 10 z 10 (p = 0,002); - `bez` +0,0402, 10 z 10 (p = 0,002).
- **Rola prognozy: niespełniona.** `tozsamosc` - `current` +0,0004, lepsza w 5 z 10 (Wilcoxon p = 0,92, test znaków p = 1,0). Obraz z przewidzianą zmianą siły i obraz bez niej dają tę samą trafność, choć prowadzą do innych klatek (wspólne top-90 z obrazem startowym średnio 27,5-34,9 z 90 w kolejnych obrotach, czyli około dwóch trzecich klatek innych).
- **Prognoza siły w pętli: spełniona** (5 z 6 obrotów z ≥ 8 z 10 ziaren poniżej „braku zmiany”). Mediany błąd / „brak zmiany” w ramieniu `tozsamosc` w obrotach 4-9: 0,972 (9 z 10 ziaren poniżej 1), 0,925 (8), 0,966 (8), 0,970 (5), 0,921 (8), 0,894 (10). Tak samo w ramionach kontrolnych (np. `bez` 0,86-0,98), więc to cecha przebiegu uczenia, nie doboru klatek. Wybierany model: Koopman 40, Langevin 30 razy na 70 obrotów ramienia `tozsamosc`.
- **FastGAN w pętli: spełnione.** Mediana części zmiany 1,00 w każdym obrocie; na zaburzonych kopiach 0,87-1,02; zgodność znaku 0,90-0,97.
- **Diagnoza siły** (`diag_sila_ramion.py` -> `DIAG_SILA.json`, po wyniku, nie kryterium): przewidziane zmiany kotwic były w połowie wzrostami, w połowie spadkami (średnio -0,004; udział spadków 0,49); siła na końcu biegu w `tozsamosc` była wyższa niż w `current` średnio o 0,44, ale tylko w 5 z 10 ziaren. Dobór klatek według prognozy nie przesuwał siły tożsamości sieci w żadną stałą stronę.
- **Porównanie z wersją 2** (te same ziarna i rozgrzewka): `koopman` v2 0,4432 = `tozsamosc` v3 0,4432; `losowe` 0,4341 -> 0,4327; `bez` 0,3999 -> 0,4030.

**Interpretacja (22:35).** Wszystkie ogniwa działały technicznie: prognoza siły biła „brak zmiany” o 3-11 %, FastGAN realizował przewidzianą zmianę, a wyszukane klatki były w dwóch trzecich inne niż bez prognozy. Mimo to prognoza nie poprawiała uczenia: ramię z prognozą i ramię kontrolne bez niej dały identyczną trafność. Przewaga nad losowaniem (+0,01, nieistotna, ta sama co w wersji 2) pochodzi z wybierania klatek podobnych do wycinków narzędzia, nie z prognozy. Kolejne klatki tych samych filmów są gorsze od losowych (-0,007), a każde dokładanie bije brak dokładania.

### 4.17 Zliczenie wygranych, sondy syntetyczne i wyjaśnienia dla użytkownika (29.09, 00:16-01:16, `8ff21d8e`)

**Prośba (00:16).** „podsumuj w ilu ziarnach, ile razy ta metoda wygrywała; jeżeli jest powyżej 80% albo blisko tych 80, no to już jest dobry wynik [...] nawet jeżeli w [...] więcej niż połowie przypadków mieliśmy sytuację, w której metoda nasza wygrywała, no to po prostu coś wymaga dopracowania, ale jest dobry trop”, a także: sprawdzić, „jak wypada detekcja” na syntetycznych klatkach (czarne tło, żółte tło, narzędzie w różnych kątach) i „na ile procent pewny jest detektor w porównaniu do [...] modelu, który był trenowany losowymi klatkami z puli”.

**Zliczenie wygranych `tozsamosc` w 10 ziarnach** (`...\wyniki_eden\ANALIZA.json`):

| Porównanie z | pole pod krzywą (obroty 3-9) | mAP@0,5 na końcu (krok 4500) |
|---|---|---|
| `current` | 5 z 10 | 6 z 10 (+0,012) |
| `losowe` | 7 z 10 | 8 z 10 (+0,023) |
| `kolejne` | 10 z 10 | 9 z 10 |
| `bez` | 10 z 10 | 9 z 10 |

Samo ramię `current` wygrywało z `losowe` w 9 z 10 ziaren (pole pod krzywą), a `losowe` z `kolejne` w 8 z 10. Przewagę nad losowaniem daje więc wybór klatek podobnych do wycinków narzędzia, wspólny dla obu ramion z FastGAN-em; sama prognoza (`tozsamosc` - `current`) jest na poziomie rzutu monetą.

**Sondy syntetyczne** (`ocena_sond.py` -> `...\wyniki_eden\OCENA_SOND.json`, `OCENA_SOND.png`). Końcowe sieci pięciu ramion i stan po rozgrzewce, 10 ziaren, 12 wycinków × 15 wersji + `empty_black`. Miary: pewność trafnego wykrycia (pewność najlepszej ramki o IoU ≥ 0,5 z prawdziwą), trafienie najpewniejszej ramki, fałszywy alarm na `empty_black`. Pewność trafnego wykrycia, średnio 10 ziaren:

| Wersja sondy | `tozsamosc` | `current` | `losowe` | `kolejne` | `bez` | po rozgrzewce | `tozsamosc` lepsza od `losowe` w ziarnach |
|---|---|---|---|---|---|---|---|
| oryginał | 0,147 | 0,117 | 0,121 | 0,126 | 0,128 | 0,108 | 7 |
| obrót 90° | 0,078 | 0,139 | 0,088 | 0,127 | 0,103 | 0,056 | 3 |
| obrót 180° | 0,110 | 0,119 | 0,110 | 0,110 | 0,108 | 0,044 | 5 |
| obrót 270° | 0,203 | 0,200 | 0,185 | 0,180 | 0,198 | 0,098 | 6 |
| czarne tło | 0,161 | 0,176 | 0,188 | 0,173 | 0,163 | 0,069 | 4 |
| żółte tło | 0,022 | 0,031 | 0,020 | 0,027 | 0,022 | 0,015 | 8 |
| skala 0,8 | 0,136 | 0,135 | 0,117 | 0,114 | 0,154 | 0,060 | 4 |
| narzędzie skrócone | 0,158 | 0,159 | 0,139 | 0,139 | 0,156 | 0,097 | 6 |
| przesunięcie w lewo | 0,155 | 0,093 | 0,138 | 0,098 | 0,128 | 0,084 | 7 |
| przesunięcie w prawo | 0,148 | 0,185 | 0,150 | 0,163 | 0,150 | 0,136 | 5 |
| przesunięcie w górę | 0,080 | 0,117 | 0,088 | 0,093 | 0,102 | 0,079 | 5 |
| przesunięcie w dół | 0,187 | 0,210 | 0,171 | 0,150 | 0,161 | 0,118 | 6 |
| przyciemnione | 0,115 | 0,117 | 0,112 | 0,092 | 0,108 | 0,104 | 6 |
| rozjaśnione | 0,184 | 0,167 | 0,156 | 0,154 | 0,171 | 0,123 | 7 |
| lekka perspektywa | 0,116 | 0,113 | 0,129 | 0,111 | 0,123 | 0,103 | 5 |
| średnio wszystkie wersje | 0,133 | 0,139 | 0,128 | 0,124 | 0,132 | 0,086 | 7 |

- `tozsamosc` lepsza (średnio po wersjach) od `current` w 2 ziarnach, od `losowe` w 7, od `kolejne` w 6, od `bez` w 5, od stanu po rozgrzewce w 10;
- najpewniejsze wykrycie trafia w narzędzie w około 20 % sond we wszystkich ramionach (0,17-0,21); na żółtym tle około 0,07;
- na `empty_black` (narzędzia nie ma) detektor zgłasza wykrycie z pewnością 0,24-0,39 (`tozsamosc` 0,258, `current` 0,284, `losowe` 0,240, `kolejne` 0,262, `bez` 0,386, po rozgrzewce 0,189), wyższą niż na prawdziwym narzędziu;
- jedyna wersja, w której `tozsamosc` bije `losowe` w ≥ 8 z 10 ziaren, to żółte tło, gdzie wszystkie ramiona mają około 0,02.
**Wniosek:** sondy są dla tego detektora za daleko od danych uczących (wycinek 256 px zmniejszony do 176 px, sztuczne tło; sieć uczyła się na pełnych klatkach 480 × 320), więc prawie ich nie wykrywa i różnice między ramionami giną w szumie. Do takiego porównania potrzebne są sondy w skali klatek uczących (narzędzie wklejone w pełną klatkę) albo prawdziwe klatki walidacji pogrupowane według kąta i tła.

**Pytania użytkownika o znaczenie wyników (00:56-01:15).**
- 00:56: czy metoda „wypada dużo lepiej” od dokładania kolejnych klatek; losowe klatki są „naturalnie bardzo dobre”, bo różnorodne, więc „jeżeli Koopman i Langevin wychodzą nawet lepiej od losowo dobieranych klatek [...] to jesteśmy w domu”. Odpowiedź (00:57): tak, ale tylko częściowo. Metoda wygrała z kolejnymi klatkami we wszystkich 10 ziarnach, przez cały przebieg o około 4 %, na końcu o około 9 %, a przewaga rosła z każdym obrotem; nie da się jednak przypisać tego Koopmanowi i Langevinowi, bo `current` pokonało kolejne klatki prawie tak samo (9 z 10). Losowe też pokonały kolejne (8 z 10). Z losowaniem metoda wygrała w 7 z 10 w całym przebiegu i w 8 z 10 na końcu; tę przewagę (2-5 %) może jeszcze tłumaczyć przypadek, więc „jesteśmy w domu” nie jest udowodnione.
- 01:11, na prośbę o jaśniejsze wyjaśnienie: jedna sieć po 21 epokach, pięć kopii, każda dostała 938 nowych klatek wybranych inaczej; ocena na 3000 klatkach, których sieć nie widziała; 10 powtórzeń. Wyniki na końcu uczenia w skali 0-100, średnio z 10 powtórzeń: 50,3 (z prognozą), 49,1 (tak samo szukane, bez prognozy), 48 (losowe), 46,1 (kolejne), 43,5 (bez nowych klatek).
- 01:13, „Co oznacza ten wynik?”: to mAP@0,5 na 3000 klatkach; 100 znaczy, że znaleziono każde narzędzie i nie zaznaczono niczego fałszywego.
- 01:14-01:15: czy kopia bez prognozy dostała „te same zdjęcia”? Nie: w metodzie obraz z FastGAN-a jest przesuwany o przewidzianą zmianę siły, w kontroli FastGAN tylko odtwarza wycinek; z każdych 90 znalezionych klatek pokrywa się około 30. Zastrzeżenie: w obu kopiach prognoza wybiera, które 6 z 12 wycinków wziąć, więc kopie różnią się tylko tym, czy obraz przesunięto o prognozę.

### 4.18 Wersja 3, iteracja 2: 14 obrotów, lista 180, 20 ziaren (Eden 1818050/1818051; 01:16-03:45, `e3836c66`, `c9c8ee4a`, `97186aa0`)

**Prośba.** O 01:16 użytkownik: „powiedz mi teraz, jakbyś robił kolejną iterację tego eksperymentu [...] kolejne ziarna [...] kolejne zdjęcia wybierzesz i potem znowu je przetestujesz na tych trzech tysiącach, tylko opisz mi, co byś zrobił na początku, zanim puścisz to na Edenie”.

**Plan (01:17, sekcja „Iteracja 2” w `PLAN.md`).** Cel: sprawdzić, czy przewaga nad losowaniem (7-8 z 10 ziaren, +1 do +2 punktów mAP@0,5) rośnie przy dłuższym dokładaniu i utrzymuje się na 20 ziarnach; próg użytkownika: metoda lepsza w ≥ 80 % ziaren. Cztery zmiany (reszta bez zmian: ramiona, wektor, prognoza, FastGAN, wyszukiwanie, reguła ±30, ocena co 100 kroków na tych samych 3000 klatkach):
1. 14 obrotów zamiast 7: uczenie do kroku 8000, 1876 dodanych klatek zamiast 938 (przewaga nad kolejnymi klatkami rosła z obrotem, od +1,4 do +2,7 punktu);
2. lista podobnych klatek na kotwicę 180 zamiast 90, jednakowo w `tozsamosc` i `current` (przy 90 metoda dostawała w ostatnich obrotach mniej klatek niż kontrole: 930 wobec 938); pula mieści około 3300 klatek przy regule ±30 (sprawdzone losowaniem, 3 próby);
3. ziarna 47-66: 47-56 z rozgrzewką z wersji 2, 57-66 z nową rozgrzewką liczoną w zadaniu (ten sam kod, 1000 kroków, około 3 min);
4. wszystkie ramiona od nowa od kroku 1000, bo kampania 1818025 nie zapisała stanu optymalizatora (pamięci o tym, jak sieć się dotąd uczyła), więc nie dało się jej uczciwie przedłużyć.
Kryteria ustalone przed wynikiem: główne `tozsamosc` > `losowe` w ≥ 16 z 20 ziaren, osobno dla mAP@0,5 na końcu (krok 8000) i dla pola pod krzywą; rola prognozy `tozsamosc` > `current` w ≥ 16 z 20; raport także względem `kolejne` i `bez`, test znaków i stan w kroku 4500. Koszt: około 30 min na ziarno, około 10 GPU-godzin, 1-1,5 h zegara, około 38 GB latentów (kwota: 68 z 200 GB).

**Uruchomienie.** O 01:18 użytkownik: „Ok puszczaj na edenie”. Zgłoszono próbę 1818050 (ziarno 57, nowa rozgrzewka, tryb `proba`, 01:22-01:28) i kampanię 1818051 (ziarna 47-66, `afterok`, 01:28-03:34) w `~/smallvit_runs/2026-09-29_maly_vit_sila_it2/`, skryptem `...\maly_vit_petla_sila_tozsamosci\petla_eden_it2.slurm` (`petla.py --obroty 14 --top-k 180`, warunkowa kopia rozgrzewki wersji 2 twardymi dowiązaniami). Wspólna część list liczona odtąd na pierwszych 90 klatkach (`wspolne_top90_ze_startem`), cała lista osobno; wcześniej przy liście 180 porównywano całe listy (poprawka błędu metryki). Po drodze poprawiono też błąd składni (`global` zadeklarowane po użyciu w `petla.py`). Eden dawał naraz około 6 kart; ziarno trwało 30-48 min; liczono na węzłach `hopper` (H100), `dgx-1` i `dgx-4` (A100). O 02:21 użytkownik poprosił „Wyłącz za 3h komputer”; wyłączenie ustawiono na 5:21, a kampania skończyła się wcześniej.

**Wyniki** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_petla_sila_tozsamosci\wyniki_eden_it2\ANALIZA.json`; latenty i listy wyszukiwania na Edenie). Średnio z 20 ziaren:

| Miara | `tozsamosc` | `current` | `losowe` | `kolejne` | `bez` |
|---|---|---|---|---|---|
| mAP@0,5 na końcu (krok 8000) | 0,545 | 0,548 | 0,536 | 0,534 | 0,456 |
| pole pod krzywą (obroty 3-16) | 0,478 | 0,473 | 0,468 | 0,463 | 0,416 |
| mAP@0,5 w kroku 4500 | 0,496 | 0,480 | 0,475 | 0,477 | 0,421 |
| pole pod krzywą do kroku 4500 (okno jak w kampanii 1818025) | 0,432 | 0,430 | 0,423 | 0,419 | 0,392 |
| klatek dodanych | 1708 | 1721 | 1876 | 1876 | 0 |

Lista 180 też się wyczerpywała, więc metoda dostała około 9 % klatek mniej niż ramiona losowe i kolejne.

Pole pod krzywą / mAP@0,5 na końcu w poszczególnych ziarnach:

| Ziarno | `tozsamosc` | `current` | `losowe` | `kolejne` | `bez` |
|---|---|---|---|---|---|
| 47 | 0,5027 / 0,5317 | 0,5012 / 0,5707 | 0,4962 / 0,5647 | 0,5026 / 0,5838 | 0,4410 / 0,4765 |
| 48 | 0,4731 / 0,5463 | 0,4599 / 0,5275 | 0,4661 / 0,5038 | 0,4659 / 0,5495 | 0,4315 / 0,4919 |
| 49 | 0,4931 / 0,5423 | 0,4771 / 0,5577 | 0,4694 / 0,5418 | 0,4549 / 0,5510 | 0,4074 / 0,4553 |
| 50 | 0,4917 / 0,6103 | 0,4884 / 0,5624 | 0,4849 / 0,6172 | 0,4812 / 0,5808 | 0,4418 / 0,4723 |
| 51 | 0,4671 / 0,5273 | 0,4654 / 0,5148 | 0,4748 / 0,5269 | 0,4628 / 0,4986 | 0,4283 / 0,4813 |
| 52 | 0,4858 / 0,5771 | 0,4712 / 0,5309 | 0,4589 / 0,4804 | 0,4681 / 0,5437 | 0,4242 / 0,4635 |
| 53 | 0,4839 / 0,5528 | 0,4652 / 0,5607 | 0,4564 / 0,4986 | 0,4509 / 0,5018 | 0,4068 / 0,4553 |
| 54 | 0,5027 / 0,5348 | 0,5050 / 0,5682 | 0,5062 / 0,5712 | 0,4953 / 0,5279 | 0,4536 / 0,4845 |
| 55 | 0,4415 / 0,5007 | 0,4272 / 0,4779 | 0,4350 / 0,5326 | 0,4148 / 0,4576 | 0,3734 / 0,3681 |
| 56 | 0,4908 / 0,5148 | 0,5098 / 0,5714 | 0,4890 / 0,5771 | 0,4797 / 0,4706 | 0,4450 / 0,4841 |
| 57 | 0,4540 / 0,5647 | 0,4638 / 0,5075 | 0,4577 / 0,5196 | 0,4572 / 0,5735 | 0,3970 / 0,4231 |
| 58 | 0,4705 / 0,5529 | 0,4749 / 0,5715 | 0,4785 / 0,5795 | 0,4607 / 0,5548 | 0,4143 / 0,5030 |
| 59 | 0,4667 / 0,4941 | 0,4574 / 0,5266 | 0,4316 / 0,4906 | 0,4389 / 0,4822 | 0,4123 / 0,4272 |
| 60 | 0,4873 / 0,5324 | 0,4933 / 0,5650 | 0,4797 / 0,5730 | 0,4825 / 0,5781 | 0,4236 / 0,4270 |
| 61 | 0,5092 / 0,5870 | 0,4860 / 0,5865 | 0,4769 / 0,5073 | 0,4746 / 0,5572 | 0,4225 / 0,5028 |
| 62 | 0,4867 / 0,5661 | 0,4712 / 0,5856 | 0,4711 / 0,5367 | 0,4555 / 0,5168 | 0,3994 / 0,4627 |
| 63 | 0,4552 / 0,5282 | 0,4532 / 0,5580 | 0,4603 / 0,5521 | 0,4527 / 0,5839 | 0,3928 / 0,4090 |
| 64 | 0,4639 / 0,5134 | 0,4683 / 0,5206 | 0,4528 / 0,5444 | 0,4585 / 0,5394 | 0,4030 / 0,4522 |
| 65 | 0,4772 / 0,6094 | 0,4683 / 0,5481 | 0,4671 / 0,5360 | 0,4549 / 0,5473 | 0,4119 / 0,4783 |
| 66 | 0,4471 / 0,5132 | 0,4514 / 0,5413 | 0,4434 / 0,4687 | 0,4505 / 0,4902 | 0,3917 / 0,4055 |

Odchylenie standardowe pola między ziarnami (policzone przeze mnie z tabeli): 0,019 / 0,020 / 0,019 / 0,020 / 0,021.

**Wygrane `tozsamosc` (z 20 ziaren, próg 16):**

| Porównanie z | pole pod krzywą | na końcu (krok 8000) | w kroku 4500 | pole do kroku 4500 |
|---|---|---|---|---|
| `losowe` | 15 (+0,010; Wilcoxon p = 0,006, test znaków p = 0,04) | 11 (+0,009, p = 0,41) | 17 (+0,021) | 16 |
| `current` | 13 (+0,005; p = 0,14, test znaków p = 0,26) | 8 (-0,003) | 12 | 13 |
| `kolejne` | 18 (+0,014; p < 0,001) | 12 | 15 | 18 |
| `bez` | 20 (+0,061; p < 0,001) | 20 | 18 | 20 |

Wartości p dla wyników „na końcu” pochodzą z `PLAN.md`; pozostałe z `ANALIZA.json`. Metoda pokonała jednocześnie `losowe`, `kolejne` i `bez` w 14 z 20 ziaren.

**Kryteria:** z `losowe` niespełnione (15 i 11 z 20, wymagane 16 w obu miarach); z `current` niespełnione; prognoza siły w pętli niespełniona (8 z 13 obrotów z ≥ 16 z 20 ziaren poniżej „braku zmiany”; mediany w ramieniu `tozsamosc` w obrotach 4-16: 0,988; 0,968; 0,957; 0,950; 0,939; 0,904; 0,870; 0,906; 0,908; 0,922; 0,919; 0,884; 0,889, lepsze w późnych obrotach); FastGAN spełnione (odporna część zmiany 0,87-0,98, znak 0,89-0,97; wspólne top-90 z obrazem startowym 31,0-38,5 z 90). Wybierany model w ramieniu `tozsamosc`: Koopman 144, Langevin 136 razy na 280 obrotów.

**Interpretacja (03:43, commit 03:45).** Dobór klatek przez metodę daje szybszą naukę: w połowie drogi (krok 4500) detektor był lepszy od uczonego losowymi klatkami w 17 z 20 ziaren, a przez cały przebieg w 15 z 20 (p = 0,006). Pod koniec, gdy każde ramię ma już około 1700-1900 dodanych klatek (ponad połowę tego, co pula mieści przy regule ±30), ramiona się zrównują (11 z 20). Ramię bez prognozy (`current`) jest prawie tak samo dobre, więc zysk pochodzi głównie z wybierania klatek podobnych do wycinków narzędzia, a udziału samej prognozy Koopmana i Langevina nie wykazano. Metoda wygrywała mimo około 9 % mniejszej liczby klatek.

**Rozmowa następnego dnia.** O 12:44 użytkownik zapytał o wyniki z Edenu, a o 12:51: „Czyli chcesz mi powiedzieć, że rzeczywiście to działa? [...] że Koopman albo Langevin przewidują, w którą stronę pójdzie latent i po prostu dobierają odpowiednie klatki?”. Odpowiedź: „Tak, ale tylko w połowie”. Cała pętla działa (klatki wybrane przez nią uczą szybciej niż losowe w 15 z 20 ziaren), Koopman i Langevin trochę przewidują (około 10 % lepiej od „braku zmiany”), ale nie da się powiedzieć, że to prognoza wybiera lepsze klatki; żeby to pokazać, metoda musiałaby w większości ziaren wygrywać z `current`, a wygrywa w 13 z 20 w całym przebiegu i w 8 z 20 na końcu.

**Czego to nie dowodzi.** Wynik dotyczy detektora od zera z 1500 klatkami bazy, jednego sposobu wyszukiwania (DINOv2 po obrazie FastGAN-a) i miary na próbce 3000 klatek walidacji. Nie pokazuje, że prognoza tożsamości pomaga; nie pokazuje też, że przewaga utrzymałaby się przy innym budżecie klatek.

### 4.19 Tożsamość na parach „z narzędziem i bez” (29.09, 12:54-14:19, `fdf82918`)

**Prośba (12:54).** „Musisz jeszcze wygenerować przykłady na przykład, nie wiem, tylko i wyłącznie czarnego tła albo tła samego oka. Możesz nawet przez FastGAN-a wygenerować takie tło, żeby dobrze zidentyfikować po prostu tożsamość narzędzia w latencie. Trzeba to zrobić teraz jeszcze”. Pomysł odpowiada brakowi z podrozdziału 4.12: dotychczasowe negatywy były za łatwe, więc test przechodziła nawet sieć nieuczona.

**Metoda** (lokalnie na RTX 3070, bez uczenia; katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_tla\`, opis `PLAN.md`):
- `tla.py`: dla każdej ze 180 sond z narzędziem dwa zdjęcia tego samego miejsca bez narzędzia: `oko` (narzędzie zamalowane z otoczenia algorytmem Telei z OpenCV, w masce poszerzonej o około 8 px; zamalowanie, czyli inpainting, wypełnia wskazany obszar, przeciągając do środka kolory i krawędzie z jego brzegu; dla wersji na czarnym i żółtym tle wychodzi jednolite tło tego koloru) oraz `czarne` (całe czarne zdjęcie). 360 obrazów w `tla\images\` (poza gitem), spis `tla\manifest.json`, podgląd `wyniki\TLA_PODGLAD.png` (zostaje lekka smuga po narzędziu). **Wybór wykonawcy nazwany w planie:** zamalowanie zamiast FastGAN-a, bo FastGAN uczono na wycinkach z narzędziem i zawsze je rysuje, a zamalowanie daje parę „to samo zdjęcie bez narzędzia”;
- `wektor_z_tlami.py`: sieci po rozgrzewce (krok 1000, ziarna 47-56); latent narzędzia na tokenach narzędzia sondy i latent zdjęć bez narzędzia na **tych samych tokenach**, więc różnica to samo narzędzie, bez wpływu miejsca na obrazie i reszty sceny. Trzy wektory (LDA z Ledoit-Wolfem): `stary` (jak w pętli: narzędzie kontra tło sond i `empty_black`), `nowy` (narzędzie kontra to samo miejsce bez narzędzia, `oko` i `czarne`), `razem` (wszystkie negatywy). Prognoza siły każdego wektora sprawdzona na zapisanych latentach wersji 2 (kroki 1000-4500, horyzont 500 kroków).
Miara rozdzielenia **d'** (d-prim): różnica średnich rzutów narzędzia i negatywów podzielona przez odchylenie standardowe rzutów negatywów, czyli ile typowych rozrzutów samych zdjęć bez narzędzia dzieli narzędzie od tła.

**Wyniki** (`...\maly_vit_tozsamosc_tla\wyniki\WEKTOR_Z_TLAMI.json` i `WEKTOR_Z_TLAMI.png`, średnio 10 ziaren):

| Miara | `stary` | `nowy` | `razem` |
|---|---|---|---|
| d': narzędzie kontra to samo miejsce bez narzędzia (`oko`) | 2,0 | 16,6 | 15,3 |
| d': narzędzie kontra czarne zdjęcie | 20,1 | 82,9 | 86,3 |
| d': narzędzie kontra tło daleko od narzędzia | 10,7 | 2,8 | 21,7 |
| AUC narzędzie kontra `oko` (te same wycinki) | 0,960 | 1,000 | 1,000 |
| AUC narzędzie kontra `oko` na wycinku wyłączonym | 0,967 | 0,997 | [nie ustalono w źródłach] |

Cosinus `stary`-`nowy` 0,28 (w ziarnie 47: 0,34), czyli prawie inny kierunek w latencie.

Prognoza siły (mediana błąd / „brak zmiany”, w nawiasie liczba ziaren z 10 poniżej 1):

| Ramię wersji 2 | Koopman: `stary` / `nowy` / `razem` | Langevin: `stary` / `nowy` / `razem` |
|---|---|---|
| `bez` | 0,94 (10) / 0,95 (10) / 0,95 (10) | 0,89 (9) / 1,02 (5) / 0,96 (7) |
| `losowe` | 0,93 (10) / 0,95 (10) / 0,95 (10) | 0,92 (8) / 1,00 (5) / 0,98 (7) |
| `koopman` | 0,95 (10) / 0,94 (10) / 0,95 (10) | 0,96 (7) / 1,02 (3) / 0,96 (8) |

Nowy wektor był lepiej przewidywany niż stary tylko w 2-5 z 10 ziaren, łączny w 2-6.

**Wniosek (13:09) i wyjaśnienie „17” (14:19).** Stary wektor tożsamości w dużej części mierzył „miejsce i scenę”, a nie samo narzędzie: zdjęcie oka bez narzędzia w tym samym miejscu dostawało prawie taką siłę jak narzędzie (różnica 2 odchylenia). Wektor z par „z narzędziem i bez” wyróżnia samo narzędzie bardzo wyraźnie (16-17 odchyleń), ale jego zmiany w czasie uczenia Koopman i Langevin przewidują nie lepiej niż zmiany starego. Na pytanie użytkownika „ale co około 17? Masz mi powiedzieć, czy jest rozróżnianie tożsamości narzędzia?” asystent odpowiedział, że sieć już po 21 epokach wyraźnie rozróżnia tożsamość narzędzia: nowy wektor odróżnia zdjęcie z narzędziem od tego samego miejsca bez narzędzia w 99,7 % par na wycinkach niewidzianych przy wyznaczaniu, stary w 96,7 %, ale z bardzo małym zapasem; „17” to wielkość tego zapasu w jednostkach typowej różnicy między zdjęciami bez narzędzia. Użycie nowego wektora w pętli wymaga nowej kampanii i osobnej zgody.

### 4.20 Zapis wektorów i ich rozłożenie na neuronach (14:21-14:23, `887e48fb`)

**Pytanie (14:21).** „czyli mamy te wektory? Ile to jest wektorów z pełnych wektorów latentu?”.

**Wykonanie.** `zapisz_wektory.py` zapisał `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_tla\wyniki\WEKTORY.npz` (poza gitem): dla każdej z 10 sieci (ziarna 47-56, krok 1000) trzy wektory (`stary`, `nowy`, `razem`), każdy to 768 wag plus średnia i rozrzut negatywów (klucze w rodzaju `s47_nowy_w`, `s47_nowy_mu`, `s47_nowy_sd`). Skupienie liczono jako liczbę neuronów, które niosą 50 % i 90 % sumy kwadratów współrzędnych wektora (`wyniki\WEKTORY_SKUPIENIE.json`):

| Wektor | 50 % wagi (średnio 10 sieci) | 90 % wagi |
|---|---|---|
| `stary` | 67,3 neuronów | 301,4 |
| `nowy` | 77,7 | 311,6 |
| `razem` | 80,9 | 312,9 |

(np. ziarno 47: 73/303, 76/314, 73/302; ziarno 48: 60/288, 76/318, 88/328.)

**Odpowiedź (14:23).** Każda sieć ma jeden wektor tożsamości obejmujący wszystkie 768 liczb pełnego latentu; 10 sieci daje 10 wektorów, których nie da się połączyć w jeden, bo w każdej sieci te same liczby znaczą co innego. Tożsamość nie siedzi w kilku neuronach, tylko jest rozłożona na setki.

### 4.21 Obraz cząstkowy: ile kierunków przestrzeni latentu koduje tożsamość (14:24-14:52, `238851f7`, `8d50ae84`)

**Pytanie (14:24).** Użytkownik sprostował: „ale ty mówisz o liczbach, a ja mówię o tym moim modelu cząsteczkowym, gdzie mamy koordynaty i ileś wektorów przestrzeni wielowymiarowej! Jeżeli o tym mówimy, to ile tych wektorów przestrzeni wielowymiarowej latentu koduje tożsamość narzędzia?”. W jego modelu cząsteczkowym każdy token obrazu to **cząstka** o 768 współrzędnych latentu, a tożsamość narzędzia ma być jednym albo kilkoma kierunkami w tej przestrzeni. Liczba neuronów z podrozdziału 4.20 na to nie odpowiada, bo jeden kierunek może się składać z setek neuronów.

**Metoda** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_tla\kierunki_tozsamosci.py`; sieci po rozgrzewce, krok 1000, ziarna 47-56). Cząstki: każdy token narzędzia sondy osobno, bez uśredniania, czyli 1568 cząstek z narzędziem na sieć, porównane z cząstką na tym samym tokenie zdjęcia bez narzędzia (`oko` i `czarne`). Dwie miary:
- **energia różnic**: kierunki główne różnic „cząstka z narzędziem minus ta sama cząstka bez narzędzia” (rozkład własny macierzy drugich momentów tych różnic); ile kierunków niesie 50 % i 90 % całej różnicy;
- **kolejne kierunki rozdzielające** (metoda INLP, Ravfogel i in. 2020): znaleźć LDA-kierunek najlepiej oddzielający cząstki z narzędziem od cząstek bez, usunąć go z przestrzeni (rzut na dopełnienie prostopadłe) i szukać następnego w tym, co zostało, do 40 kierunków albo do spadku AUC poniżej 0,6. AUC każdego kierunku liczone na wycinkach wyłączonych z jego wyznaczania (4 części po 3 wycinki). Liczone są kolejne kierunki z AUC ≥ 0,9 i ≥ 0,75.
Pierwsze uruchomienie w tle przez Basha zostało przerwane (buforowanie wyjścia przez `grep` i 10-minutowy limit zadań w tle), więc powtórzono je jako proces odłączony z logiem. O 14:31 użytkownik zapytał „i jak idzie to liczenie?”; o 14:35 asystent podał wynik pierwszej sieci (jeden kierunek rozdziela w 94 parach na 100, dwa kierunki niosą połowę różnicy, 37 kierunków 90 %, po usunięciu 20 najlepszych kierunków wynik bliski rzutu monetą).

**Wyniki** (`...\maly_vit_tozsamosc_tla\wyniki\KIERUNKI_TOZSAMOSCI.json`, log `kierunki_tozsamosci.log`):

| Ziarno | Energia: 50 % w / 90 % w kierunkach (udział pierwszego) | Kierunki z AUC ≥ 0,9 / ≥ 0,75 | AUC pierwszych 8 kolejnych kierunków (wycinki wyłączone) |
|---|---|---|---|
| 47 | 2 / 37 (42 %) | 1 / 8 | 0,94; 0,89; 0,85; 0,82; 0,80; 0,78; 0,77; 0,75 |
| 48 | 3 / 47 (32 %) | 2 / 7 | 0,96; 0,91; 0,87; 0,84; 0,81; 0,79; 0,77; 0,74 |
| 49 | 2 / 33 (43 %) | 1 / 5 | 0,95; 0,88; 0,83; 0,79; 0,76; 0,74; 0,71; 0,69 |
| 50 | 3 / 39 (32 %) | 1 / 6 | 0,96; 0,90; 0,86; 0,83; 0,79; 0,76; 0,73; 0,70 |
| 51 | 3 / 48 (34 %) | 2 / 7 | 0,96; 0,91; 0,87; 0,84; 0,82; 0,79; 0,77; 0,74 |
| 52 | 2 / 29 (38 %) | 1 / 7 | 0,95; 0,89; 0,85; 0,82; 0,79; 0,77; 0,75; 0,73 |
| 53 | 3 / 39 (40 %) | 1 / 5 | 0,95; 0,88; 0,83; 0,79; 0,76; 0,73; 0,71; 0,70 |
| 54 | 2 / 40 (40 %) | 1 / 7 | 0,96; 0,90; 0,86; 0,83; 0,81; 0,78; 0,75; 0,74 |
| 55 | 2 / 35 (43 %) | 2 / 7 | 0,96; 0,90; 0,86; 0,83; 0,81; 0,79; 0,77; 0,74 |
| 56 | 3 / 37 (35 %) | 1 / 6 | 0,95; 0,88; 0,85; 0,81; 0,78; 0,76; 0,74; 0,72 |
| średnio | 2,5 / 38,4 (38 %) | 1,3 / 6,5 | |

AUC kolejnych kierunków spada łagodnie: po 10 kierunkach 0,66-0,73, po 15 0,60-0,68, po 20 0,60-0,64 (w ziarnie 54 po 25 kierunkach 0,59). W ziarnach 50 i 54 drugi kierunek jest wypisany w logu jako 0,90, ale nie wszedł do liczby kierunków z AUC ≥ 0,9, bo skrypt liczy na pełnej precyzji, więc był tuż poniżej progu; w ziarnie 55 wszedł.

**Zapis kierunków cząstkowych i porównanie z wektorem `nowy`** (`zapisz_kierunki_czastek.py`, `8d50ae84`). **Kierunek cząstkowy** to kierunek wyznaczony na pojedynczych cząstkach (tokenach), w odróżnieniu od wektora `nowy` liczonego na latentach uśrednionych po tokenach sondy. Dla każdej sieci zapisano 6 kolejnych kierunków (wyznaczonych na wszystkich wycinkach) w `...\maly_vit_tozsamosc_tla\wyniki\KIERUNKI_CZASTEK.npz` (klucze `s47_w` o kształcie 6 × 768 w przestrzeni (latent - `s47_mu`) / `s47_sd`, poza gitem). Porównanie (`wyniki\KIERUNKI_CZASTEK.json`, średnio 10 sieci): cosinus pierwszego kierunku cząstkowego z wektorem `nowy` w surowym latencie 0,40 (od 0,32 do 0,52), czyli kąt około 66°; pierwszy kierunek cząstkowy ma AUC 0,999 na cząstkach, z których go wyznaczono (0,95 na wycinkach wyłączonych, tabela wyżej), a wektor `nowy` na tych samych cząstkach 0,96.

**Odpowiedzi dla użytkownika (14:46 i 14:52).** Tożsamość narzędzia koduje głównie jeden kierunek przestrzeni cząstek, a pomaga mu około pięciu słabszych; pierwszy kierunek odróżnia cząstkę narzędzia od cząstki w tym samym miejscu bez narzędzia w około 95 parach na 100, w każdej z 10 sieci, na wycinkach niewidzianych przy wyznaczaniu; po usunięciu pierwszego najlepszy następny trafia w 89 na 100, a po piętnastu usuniętych wynik jest bliski rzutu monetą. Wielkość różnicy jest szerzej rozłożona (90 % w około 38 kierunkach), ale te kierunki opisują głównie wygląd narzędzia, a nie to, czy ono jest na obrazie. Na pytanie „czyli mamy już ten wektor?” (14:50): tak, ale nie ma jednego wspólnego, tylko osobny dla każdej sieci; nie jest to też jedna jedyna strzałka, bo dwa sposoby liczenia (średnia z tokenów sondy i pojedyncze cząstki) dały strzałki rozchylone o około 66°, a obie rozdzielają narzędzie równie dobrze. Tożsamość zajmuje więc wąską wiązkę kierunków, nie jedną linię.

**Czego to nie dowodzi.** Pomiar dotyczy jednej chwili (krok 1000) i sond w skali wycinka; nie ma w nim próby przyczynowej (czy usunięcie tych kierunków psuje detekcję) ani sprawdzenia, jak wiązka obraca się w trakcie uczenia.

### 4.22 Wątki zamykające etap: literatura o Stable Diffusion, plan dekodera latentu, zapis sesji (14:30-14:52, `fb899bde`)

**Rozwidlenie rozmowy.** O 14:30 w transkrypcie odnotowano utworzenie osobnej, równoległej sesji (rozwidlenie rozmowy, w którym nic z dalszego ciągu nie trafia do tej rozmowy). Streszczenie sesji przypisuje commit `fb899bde` tej równoległej sesji, ale transkrypt głównej rozmowy zawiera zapis pliku planu dekodera o 14:51 i odpowiedź, że plan „zapisałem i zatwierdziłem w gicie”.

**Literatura o Stable Diffusion (14:31-14:45).** Użytkownik uruchomił wyszukiwanie „artykułów, które opisują, jak generować obrazy przy pomocy Stable Diffusion, które spełniają latent”; wyszukiwanie pogłębione znalazło 133 prace, a 40 linków trafiło do kolejki pobierania `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDownloader\linki\stable-diffusion-latent-docelowy-sieci-2026-09-29.txt`. Ten wątek i notatkę literaturową z 29.09 opisuje etap 5.

**Plan dekodera latentu** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_dekoder_latentu\PLAN.md`, `fb899bde`; nic nie uruchomiono). Prośby użytkownika: o 14:45 „zrób analizę mojego pipelinu i powiedz, jak mogę ulepszać generowanie obrazów u mnie, biorąc pod uwagę, że mogę wszystko trenować na Edenie”, o 14:46 „dla mnie najlepiej by było rysować prosto z latentu! Wtedy bym mógł rysować i widzieć różnice pomiędzy tym, co jest aktualnie, a tym, co przewiduje Koopman i Langevin”. Analiza 4200 raportów `RAPORT.json` z Edenu (mediany):

| Miara | wersja 3 (420 kotwic × ramię) | iteracja 2 (1680) |
|---|---|---|
| błąd inwersji (L1 obrazu + L1 maski) | 0,104 | 0,105 |
| przewidziana zmiana siły, wartość bezwzględna (odchylenia negatywów) | 0,55 | 0,66 |
| osiągnięta część zmiany (odporna) | 1,00 (0,93) | 1,00 (0,94) |
| zmiana obrazu „po prognozie” względem „teraz”, L1 RGB | 0,064 | 0,060 |
| zmiana latentu w bok od kierunku tożsamości (norma) | 22,3 | 25,1 |
| siła tożsamości startu | 4,96 | 4,20 |

Wniosek planu: FastGAN trafia w przewidzianą siłę, ale zmiana latentu poza kierunkiem tożsamości jest około 40 razy większa niż wzdłuż niego (kara za zmianę w bok ma wagę 1/50 średniej po 768 liczbach, czyli jest słaba), więc obraz „po prognozie” różni się od obrazu „teraz” głównie rzeczami, których prognoza nie dotyczy; to zgadza się z równością ramion `current` i `tozsamosc`. Propozycja: warunkowy model dyfuzyjny uczony raz na Edenie, warunkowany przestrzenną mapą tokenów wycinka (11 × 11 tokenów po 768 liczb), bo według pracy FeatInv (Neukirch 2025) warunkowanie mapą trafia w cel dużo lepiej niż uśrednionym wektorem (cosinus 0,5-0,6 wobec 0,12-0,19). Prognoza byłaby rysowana jako x_prognoza = x_teraz + Δs · w na tokenach narzędzia, z tym samym szumem startowym co obraz „teraz”, więc różnica pikseli pokazałaby tylko przesunięcie przewidziane przez prognozę (zmiana w bok równa zeru z definicji). Trudność: sieć zmienia się co obrót; warianty: A, most przez zamrożony DINOv2 z liniowym przelicznikiem tokenów małego ViT na tokeny DINOv2 dopasowywanym co obrót; B, dekoder wprost na małym ViT z kroku 1000. Plan nazywa cztery różnice względem metody użytkownika (dyfuzja zamiast FastGAN-a; w A obraz z latentu DINOv2 zamiast gradientu małego ViT; w B obraz odpowiada sieci z kroku 1000; możliwość wyszukiwania wprost przesuniętymi cechami DINOv2 jako osobnego ramienia). Model: MiniSD 256 z ControlNet (jak FeatInv) albo mała dyfuzja w pikselach od zera (jak RCDM i Frisch 2023 na CATARACTS); dane: wycinki 256 × 256 z bazy (1500) i puli (9136), bez klatek walidacji; koszt oszacowany (niesprawdzony) na kilka do kilkunastu godzin jednej karty `hopper` (FeatInv: 45-60 h na dwóch L40 dla 1,28 mln obrazów ImageNet). Etapy: 0, lokalny test mostu bez uczenia (R² przelicznika, czyli udział odtworzonej zmienności tokenów, i przeniesienie siły tożsamości); 1, uczenie na Edenie po zgodzie; 2, sprawdzenie w zamkniętej pętli (część zmiany ≥ 0,8, zmiana w bok mniejsza niż 22-25, obraz „teraz” bliżej kotwicy niż inwersja FastGAN-a, L1 0,10); 3, rysunki; 4, ewentualna kampania. Plan sam zaznacza ograniczenie: prognoza to jedna liczba na sondę, mediana przewidzianej zmiany około 0,6 odchylenia przy sile narzędzia 4-5, więc różnica na obrazie będzie subtelna, a prognoza jest tylko trochę lepsza od „braku zmiany” (Koopman 0,94, Langevin 0,89 dla starego wektora).

**Ostatnie prośby i zapis sesji.** O 14:49 użytkownik dopisał prośbę o głębokie wyszukiwanie, jak używać Koopmana i Langevina (szczególnie w formie równania z jego eksperymentów grantowych `tft_langevin_optymalizacja_95`, `tft_langevin_czlon_zdarzen`, `tft_langevin_krok_po_kroku`, `tft_langevin_latent_56`, `tft_langevin_petla_aktywna`, `tft_deep_koopman_mz`, `tft_wlasna_pamiec_ze_stanu`, `tft_korekta_pamiec_roznicowa`) do przewidywania latentu kilka kroków naprzód i na ile kroków się da, osobno dla Koopmana i dla Langevina; asystent uruchomił agenta czytającego te osiem eksperymentów, a sam wątek należy do etapu 5. O 14:52 użytkownik zlecił `/save-session` do repozytorium ViT, co zamknęło etap zapisem `Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek`.

### 4.23 Podsumowanie etapu 4

1. **Diagnoza porażki na RF-DETR Large:** prognoza Koopmana była gorsza od „braku zmiany” w 12 z 12 przypadków (błąd 0,196 wobec 0,078), bo przestrzeliwała wielkość ruchu; kręgosłup DINOv2 przy lr_encoder 1,67e-6 zmienił wagi tylko o 0,026 % w 6 epokach; skalar tożsamości nie wiązał się z mAP (ρ = 0,07); wybór skupiał klatki (138-140 wobec 62-83 sąsiadów).
2. **Dekoder zamiast kręgosłupa:** jakość wykrycia sond z dekodera rosła razem z mAP w czasie (ρ = 0,86 wobec 0,41), ale na lokalnych wagach Nano nie rozróżniała ramion (ρ = -0,07, w pilocie -0,51) i po fazie wzrostu stała, więc żaden model nie zszedł poniżej 0,8 × „brak zmiany” (najlepiej 0,82); nic nie poszło na Eden.
3. **Mały ViT od zera** (2 926 493 parametrów, łodyga konwolucyjna, gęsta głowa, 1500 klatek bazy): uczy się (mAP@0,5 0,36 po 1500 krokach, do 0,42 po 6000), ale nie przeszedł progu 0,5; dokładanie jakichkolwiek klatek pomaga wyraźnie (losowe nad brakiem dokładania: +0,025 pola pod krzywą w 11 z 11 ziaren wersji 1, +0,034 w 10 z 10 ziaren wersji 2).
4. **Wersja 1** (11 ziaren): `koopman` > `losowe` w 9 z 11 (+0,013, Wilcoxon p = 0,005), ale prognoza równała się „brakowi zmiany”, FastGAN nie trafiał w cel (bliżej tylko w 39 % par), rdzeń tożsamości nie powstał, a lista neuronów tożsamości wymieniała się w połowie co około 5 epok.
5. **Wersja 2** (10 ziaren, wektor na 768 neuronach, start po 21 epokach): 8 z 10 (+0,009, p = 0,064; jedna wygrana o 0,00005), prognoza wektora 0 z 6 obrotów; wyszło, że asystent odszedł od metody użytkownika bez nazwania różnic, co dało stałą regułę pracy.
6. **Tożsamość po 21 epokach:** rdzeń neuronowy 0 w 10 ziarnach; wektor LDA przechodzi testy na wersjach wyłączonych (rdzeń wektorowy w 8 z 10), ale sieć nieuczona przechodzi je równie dobrze; siła tożsamości jest pierwszą wielkością przewidywaną lepiej niż „brak zmiany” (o 6-10 %, kierunek 64-69 % w ramieniu `bez`).
7. **FastGAN:** do pełnego przewidzianego latentu dochodzi najwyżej w 36 % drogi (głębsze warstwy, artefakty), samym wejściem w 4 %, sieć MLP nic nie dodaje (0,24 wobec 0,25), a odpornie zostaje 15-19 %; sterowany wzdłuż wektora tożsamości realizuje przewidzianą zmianę siły w całości, znak 12 z 12, 85-111 % odpornie.
8. **Wersja 3** (10 ziaren, 5 ramion): ramię z prognozą równe kontrolnemu `current` (0,4432 wobec 0,4428, 5 z 10); nad losowym 7 z 10 (+0,0105, p = 0,11); nad kolejnymi klatkami 10 z 10; kryterium główne i kryterium roli prognozy niespełnione, kryteria prognozy w pętli i FastGAN-a spełnione.
9. **Iteracja 2** (20 ziaren, 14 obrotów, lista 180): metoda szybciej uczy niż losowe klatki (pole 15 z 20, p = 0,006; krok 4500: 17 z 20), ale na końcu ramiona się zrównują (11 z 20), a wobec `current` wygrywa w 13 z 20 i 8 z 20; zysk daje wyszukiwanie klatek podobnych do narzędzia, udziału prognozy nie wykazano.
10. **Pary z narzędziem i bez:** stary wektor oddzielał narzędzie od tego samego miejsca bez narzędzia tylko na d' = 2,0, nowy na d' = 16,6 (AUC 0,997 na wycinkach wyłączonych), cosinus między nimi 0,28; nowy nie jest lepiej przewidywalny.
11. **Obraz cząstkowy:** tożsamość niosą 1-2 kierunki z AUC ≥ 0,9 i 5-8 z AUC ≥ 0,75; energia różnic 50 % w 2-3, 90 % w 29-48 kierunkach; pierwszy kierunek cząstkowy i wektor `nowy` mają cosinus 0,40 (około 66°), a rozdzielają równie dobrze, więc tożsamość to wąska wiązka kierunków.
12. **Infrastruktura:** Eden posprzątany ze 193,2 do 52,2 GB (archiwum raportów lokalnie, zbiory uczące zostały); dwie nieudane próby (brak timm, ścieżki Windows) naprawione; 4 kampanie Eden (1817935, 1817949, 1818025, 1818051) policzone bez błędów.

### 4.24 Otwarte pytania na koniec etapu 4

- Czy prognoza Koopmana i Langevina wnosi cokolwiek do doboru klatek? W obu kampaniach wersji 3 ramię `current` było prawie równe metodzie, a w obu ramionach prognoza wybierała kotwice; brakowało ramienia, które różniłoby się wyłącznie prognozą (np. losowy wybór kotwic albo prognoza z odwróconym znakiem).
- Dlaczego przewaga nad losowymi klatkami znika pod koniec iteracji 2 (17 z 20 w kroku 4500, 11 z 20 w kroku 8000)? Czy przez wyczerpywanie się puli (około 3300 klatek przy regule ±30, zużyte 1700-1900) i list podobnych klatek (metoda dostała około 9 % klatek mniej)?
- Czy nowy wektor tożsamości (d' 16,6) albo pierwsze kierunki cząstkowe użyte w pętli zmieniłyby wynik? Niesprawdzone, wymaga planu i zgody.
- Czy prognoza na 8 kierunkach PCA wybranych według różnic między sondami, a nie według zmian w czasie, nie „ślepnie” na ruch? Podejrzenie zgłoszone o 17:42, niesprawdzone.
- Czy pełne równanie Langevina użytkownika (dryf, pamięć, wymuszenie, sterowanie, szum, Kalman) da na małym ViT więcej niż prosty Langevin z hamowaniem? Kalmana w tym etapie nie liczono; prośba o głębokie wyszukiwanie przeszła do etapu 5.
- Czy tożsamość wyznaczona na sondach jest tym, czego sieć naprawdę używa? Sieć nieuczona przechodziła testy na wersjach sond; brak próby przyczynowej (usunięcie kierunku z latentu i pomiar detekcji) i trudniejszych negatywów (odblaski, krawędzie tkanek).
- Jak wiązka kierunków tożsamości obraca się w trakcie uczenia? Wektor był stały od kroku 1000, a siła wzdłuż niego malała (6,13 -> 4,36 w ramionach z dokładaniem).
- Jak porównywać tożsamość między ziarnami, skoro wektory różnych sieci nie są porównywalne (każda sieć koduje neurony inaczej)?
- Czy generator może zmieniać obraz tylko wzdłuż tożsamości? Inwersja FastGAN-a sama zmienia siłę o ±3 przy przewidzianej zmianie około 0,6-0,7, a zmiana w bok jest około 40 razy większa od zmiany wzdłuż; plan dekodera latentu i dyfuzji czeka na zgodę.
- Jak zmierzyć pewność detekcji na obrotach i innym tle, skoro sondy syntetyczne są poza rozkładem detektora (trafienie najpewniejszej ramki około 20 %, fałszywe alarmy na czarnym obrazie 0,24-0,39)? Potrzebne sondy w skali pełnej klatki albo prawdziwe klatki walidacji pogrupowane według kąta i tła.
- Czy zgoda „możesz to zrobić!” z 19:26 dotyczyła wyszukiwania klatek wprost po latencie ViT, bez FastGAN-a? Takiego ramienia w tym etapie nie zbudowano.
- Czy efekt z detektora uczonego od zera przenosi się na dostrajany RF-DETR, na którym zaczęła się cała linia badań?



## Etap 5. Horyzont prognozy, literatura o dyfuzji, równanie ruchu latentu, macierze jak w TS i Stable Diffusion na całych filmach (29-30 września 2026)

Etap trwa od 29 września 2026, 14:52, do 30 września 2026, 13:05 (czas lokalny, UTC+2). Cała praca merytoryczna toczyła się w repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker` (gałąź gita `codex/eden-nano-training-20260914`), choć rozmowę prowadzono z katalogu projektu grantowego `F:\Studia\MCHTR\Grant\Projekt`. Etap ma trzy splecione wątki. Pierwszy pyta, na ile kroków naprzód da się przewidzieć ruch latentu małego ViT i jakim równaniem; kończy się jawnym równaniem ruchu latentu, rozdzieleniem „wyjaśnienia” od „prognozy” i zapisaną metodą. Drugi dotyczy generatora obrazów: przegląd literatury o dyfuzji trafiającej w zadany latent, własny mały model dyfuzyjny w przestrzeni pikseli i w końcu Stable Diffusion 1.5 dostrajane na wszystkich klatkach 118 filmów treningowych. Trzeci to porządki na klastrze Eden: sprzątanie kwoty dyskowej i wysyłka pełnego zbioru filmów Cataract-LMM.

Pojęcia wspólne dla całej części (dalej używane bez powtarzania):

- **Mały ViT** - detektor narzędzia uczony od zera w tym projekcie: transformer wizyjny (sieć, która dzieli obraz na kwadratowe kawałki zwane tokenami i w kolejnych blokach miesza informację między nimi) o 6 blokach, wymiarze tokenu 192 i około 3 mln parametrów. Klatkę 480 × 320 dzieli na siatkę 30 × 20 tokenów po 16 pikseli i dla każdego tokenu przewiduje, czy leży w nim środek narzędzia, oraz odległości do krawędzi ramki. Kod: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera\model.py`.
- **Latent** (w wątku równania ruchu) - wektor 768 liczb: aktywacje neuronów warstwy ukrytej MLP (dwuwarstwowej sieci wewnątrz bloku transformera) w ostatnim bloku, po nieliniowości GELU, uśrednione po tokenach, na których leży narzędzie. Drugi kanał to ta sama średnia po tokenach tła. W wątku Stable Diffusion „latent obrazu” znaczy coś innego (5.12).
- **Sonda** - jeden ze 192 stałych obrazów kontrolnych (12 wycinków z narzędziem × 16 wersji), których latent mierzy się po każdym kroku uczenia, jak położenie cząstki próbnej. Jedna z 16 wersji (`empty_black`) nie zawiera narzędzia, więc do pomiarów wchodzi 180 sond. Wycinki 0-7 mają rolę „fit”, wycinki 8-11 rolę „holdout” (narzędzia, których dopasowanie równania nie widziało).
- **Krok** - jedna aktualizacja wag sieci. **Paczka danych** - 32 obrazy uczące użyte w jednym kroku; gradient straty policzony na paczce wyznacza zmianę wag.
- **AdamW i pęd optymalizatora** - AdamW nie przesuwa wag wprost o gradient g(t), tylko o średnią wykładniczą gradientów m(t) = β1·m(t-1) + (1-β1)·g(t), β1 = 0,9, podzieloną po współrzędnych przez pierwiastek ze średniej wykładniczej kwadratów gradientów v(t) (β2 = 0,999). Składnik β1·m(t-1) to **pęd**: pamięć kierunku poprzednich paczek, dzięki której wagi mają bezwładność. Do tego dochodzi **zanik wag** (weight decay 0,05), czyli lekkie ściąganie wag do zera. Mały ViT uczy się z lr (długością kroku) 5·10⁻⁴, z rozgrzewką (liniowym wzrostem lr przez pierwsze 300 kroków), z obcinaniem normy gradientu do 1 i w arytmetyce bf16 (16-bitowy format liczb z szerokim zakresem wykładnika, szybszy od 32-bitowego float32).
- **Siła tożsamości** - jedna liczba na sondę: rzut latentu sondy (wystandaryzowanego statystykami tła) na kierunek tożsamości narzędzia. Kierunek wyznacza liniowa analiza dyskryminacyjna (LDA: reguła liniowa rozdzielająca latent narzędzia od latentu tła), osobno w każdej sieci.
- **Ziarno** - liczba startowa generatora losowego; różne ziarna (tu 47-56) to różne, równie prawdopodobne przebiegi uczenia tej samej sieci.
- **Umiejętność** - miara używana we wszystkich pomiarach ruchu: U = 100 · (1 - Σ‖Δẑ - Δz‖² / Σ‖Δz‖²), gdzie Δz to prawdziwy przyrost, a Δẑ przewidziany. Mówi, jaki procent błędu kwadratowego prognozy „bez zmiany” usuwa model: 0 to tyle co brak zmiany, 100 to trafienie idealne, liczby ujemne są gorsze od braku zmiany. „Trafność 85 procent” w rozmowie znaczy U = 85.
- **Eden i SLURM** - klaster obliczeniowy uczelni; zadania zgłasza się do kolejki SLURM poleceniem `sbatch`, każde dostaje numer. Partycja `hopper` to jeden węzeł z 8 kartami H100, partycja `short` to węzły `dgx-1` do `dgx-4` z kartami A100 40 GB. Katalog domowy: `/home2/faculty/bpiotrowski`.

### 5.1 Przebieg etapu, równoległe gałęzie rozmowy i commity

W pliku rozmowy `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\7dec512f-abe0-4cf3-8465-062719d28449.jsonl` od wspólnego wpisu z 29.09, 14:30:34, biegną dwie niezależne nitki. Rozdzieliłem je po łańcuchu odwołań między wpisami (każdy wpis wskazuje swojego poprzednika) i po treści. Obie podpisują commity tym samym identyfikatorem sesji Claude (`session_014fENohnR1DdC8xZyh9z9jp`), więc sam commit nie mówi, która nitka go zrobiła.

- **Gałąź główna** (dalej: główna): horyzont prognozy, równanie latentu, kampania Eden, macierze jak w TS, a od 23:46 cała reszta etapu (reguła o przyszłych klatkach, zapis metody, diagnoza modelu pikselowego, sprzątanie Edenu, wysyłka filmów, Stable Diffusion). To ona napisała zapis sesji `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-30_124320_Rownanie_Latentu_Eden_i_Stable_Diffusion_Na_Filmach`.
- **Rozgałęzienie o dyfuzji pikselowej** (dalej: rozgałęzienie): trwa od 14:31 do ostatniego wpisu o 17:47 (29.09). Dokończyło liczenie kierunków tożsamości i zapis poprzedniej sesji (koniec etapu 4), przeczytało pełne teksty 10 prac o dyfuzji, wyjaśniło użytkownikowi metodę Farida, a potem zbudowało i uruchomiło na Edenie własny model dyfuzyjny wycinków narzędzia (`dyfuzja_narzedzie`). Gałąź główna nazywa je później „inną sesją” i od 30.09 traktuje jego katalog jako tylko do odczytu.
- **Sesja-fork** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-30_124311_Fork_Kierunki_Tozsamosci_Sprawdzenie_Stanu` (osobny plik rozmowy `439a868c-9324-439e-9729-b83bc92511b5.jsonl`), utworzona poleceniem `/fork` o 14:30:35. Przez cały etap zadała tylko jedno pytanie (5.13).

Kalendarium (czas lokalny; G = główna, R = rozgałęzienie, F = sesja-fork):

| czas | nitka | wydarzenie |
|---|---|---|
| 29.09 14:49 | G | prośba o horyzont prognozy Koopmana i Langevina (tuż przed etapem, 5.2) |
| 14:52-14:55 | R | zapis poprzedniej sesji `Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek` |
| 14:57-15:04 | G | wyszukiwanie pogłębione w Undermind (205 prac) |
| 15:13 | G | horyzont prognozy, commit `3ce17b1c` |
| 15:33 | G | prośba o równanie Langevina dla latentu powyżej 80 procent |
| 15:34-15:40 | R | literatura: notatka `335c8528`, pełne teksty `ddacbc74` |
| 15:44-16:13 | G | lokalne nagrania uczenia ziarna 47 |
| 16:07-16:25 | R | rozmowa o metodzie Farida, decyzja „ta odpada”, propozycje ReNO i FreeDoM |
| 16:29 | R | prośba o plan uczenia modelu dyfuzyjnego (FreeDoM) na Edenie |
| 16:30 | G | wynik 85,8; „możesz to wysyłać na eden zeby policzyc” |
| 16:33-16:43 | G | próba 1818328, commit `019030f4` (16:34), kampania 1818331 |
| 16:48-17:05 | R | dane z filmów, commit `62f5a7ed` |
| 17:00-17:20 | G | wyniki ziaren, pytania o macierze jak w TS, commit `5a2ac3b1` |
| 17:11-17:44 | R | inwentarz Edenu, obrysy, próby 1818351 i 1818352, uczenie 1818354 (+1818355, 1818356), commit `fc7ae09e` |
| 17:27 | G | reguła: bez przyszłych klatek w prognozie |
| 17:36-17:52 | G | macierze jak w TS, commity `c85c75c6`, `37a022cd` |
| 17:47 | R | ostatni wpis rozgałęzienia |
| 23:28 | F | „co było liczone w tym forku?” |
| 23:46 | G | pytanie o przyszłe klatki, ujawnienie |
| 30.09 00:01-00:03 | G | doprecyzowanie reguły, commit `4a0c9a92` |
| 00:24-00:26 | G | zapis metody, commit `1a85dfbc` |
| 00:27-00:33 | G | diagnoza złych kolorów modelu pikselowego |
| 00:43-01:03 | G | pytania o karty i miejsce, analiza, commit `13166fa0` |
| 01:31-01:40 | G | „Usuwaj”, sprzątanie, start wysyłki filmów, commit `1f4de9d1` |
| 01:47-02:20 | G | plan i kod Stable Diffusion, commit `2bf184b0` |
| 04:22 / 04:38 | G | koniec wysyłki / sumy SHA256 zgodne |
| 09:49-10:06 | G | „i jak”, „tak wysyłaj i trenuj” |
| 10:13-10:31 | G | wysyłka kodu i wag, próby 1818490-1818492, zgłoszenie 1818494-1818496, commit `056127e9` |
| 10:34-11:55 | G | pytania o trening i latent, poprawka monitora |
| 10:51 | G | koniec przygotowania latentów, start uczenia 1818496 |
| 12:20 | G | podgląd po 4000 krokach |
| 12:44 / 12:46 | F / G | zapisy sesji: `beed3307` (fork), `255f6668` (główna) |
| 13:04 | - | krok 6200 z 20 000 (stan przekazany przez użytkownika) |

Commity etapu (`git -C F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker log --since="2026-09-29 14:53"`):

| czas | commit | nitka | zawartość (`git show --stat`) |
|---|---|---|---|
| 29.09 15:13 | `3ce17b1c` | G | `maly_vit_horyzont_prognozy`: `horyzont.py`, `WYNIKI.md`, `wyniki/HORYZONT.json` (3 pliki, 2847 linii) |
| 15:35 | `335c8528` | R | `fastgan_odwrotnosc_mlp/NOTATKA_LITERATURA.md` (+42) |
| 15:40 | `ddacbc74` | R | notatka (+73/-1) i `literatura/pelne_teksty_sd_2026-09-29.md` (1499 linii) |
| 16:34 | `019030f4` | G | `maly_vit_rownanie_latentu`: PLAN, WYNIKI_T0_T1, WYNIKI_T2, `dane.py`, `nagranie.py`, `t1_bez_sterowania.py`, `t2_sterowanie.py`, `rownanie_eden.slurm`, `wyniki/T1.json`, `wyniki/T2.json` (11 plików) |
| 17:05 | `62f5a7ed` | R | `dyfuzja_narzedzie`: PLAN, 6 plików kodu, 4 skrypty Eden, manifest danych (13 plików) |
| 17:20 | `5a2ac3b1` | G | WYNIKI_EDEN, `zbierz_eden.py`, `wyniki_eden/T2_s47..56.json`, `ZBIORCZO.json`, `proba_s47.json`, zmiany `nagranie.py` (16 plików) |
| 17:37 | `c85c75c6` | G | `maly_vit_rownanie_macierze`: PLAN, `dane.py`, `model.py`, `uczenie.py`; dopiski reguły w WYNIKI_T2 i WYNIKI_EDEN |
| 17:43 | `fc7ae09e` | R | `dyfuzja_narzedzie`: `przygotuj_obrysy.py`, zmiany `trenuj.py`, manifest 57 073 klatek Edenu, `wyniki/eden_proba` |
| 17:51 | `37a022cd` | G | `maly_vit_rownanie_macierze`: WYNIKI, `liniowe.py`, 8 × `BIEG_*.json`, `LINIOWE.json`, `ODNIESIENIA.json`, 8 × `TRENING_*.csv` (20 plików) |
| 30.09 00:03 | `4a0c9a92` | G | doprecyzowanie reguły w trzech plikach WYNIKI |
| 00:26 | `1a85dfbc` | G | `docs/23_METODA_PRZEWIDYWANIA_LATENTU.md` (95 linii), odnośnik w `docs/README.md` |
| 01:03 | `13166fa0` | G | `docs/EDEN_SPRZATANIE_2026-09-30.md` (analiza, nic nie usunięto) |
| 01:40 | `1f4de9d1` | G | sekcja „Wykonanie” w tym dokumencie, `eden/wyslij_cataract_lmm.sh` |
| 02:20 | `2bf184b0` | G | `dyfuzja_sd_filmy`: PLAN, README, 3 pliki kodu, 3 skrypty Eden (9 plików) |
| 10:31 | `056127e9` | G | README `dyfuzja_sd_filmy` (+15: próba i zgłoszenie) |
| 12:44 | `beed3307` | F | zapis sesji-fork (README, SESSION_SUMMARY) |
| 12:46 | `255f6668` | G | zapis sesji głównej (README, SESSION_SUMMARY) |

### 5.2 Horyzont prognozy siły tożsamości: Koopman i Langevin osobno, do 1000 kroków

**Pytanie i skąd się wzięło.** Etap 4 zakończył się pętlą, w której Koopman lub Langevin przewidywał siłę tożsamości na jeden obrót (500 kroków uczenia), a generator obrazów rysował zapytanie do wyszukiwania klatek. Nie było wiadomo, jak daleko taka prognoza w ogóle działa. O 14:49 użytkownik poprosił: „szukaj głęboko jeszcze jak powinienem używać koopman i langevin (szczególnie takiego równania langevina jakiego użyłem w [ośmiu katalogach `tft_*` projektu grantowego]) … TUtaj szukaj jak można go wykorzystać do przewidywania latentu w kolejnych krokach do przodu i na ile kroków do przodu jest w stanie koopman i langevin przewidywac do przodu. Tutaj można używać koppmana osobno a langevina osobno!”.

**Metoda.** Pomiar lokalny w numpy, bez uczenia żadnej sieci (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_horyzont_prognozy\horyzont.py`, około 12 minut liczenia). Dane: zapisane na Edenie historie siły tożsamości `SILA.npy` (180 sond, zapis co 10 kroków) z pętli `maly_vit_petla_sila_tozsamosci`, w dwóch kampaniach: wersja 3 (10 ziaren, 7 obrotów, do kroku 4500) i iteracja 2 (20 ziaren, 14 obrotów, do kroku 8000), razem 30 przebiegów, wszystkie pięć ramion pętli:

- `bez` - bez dokładania klatek;
- `losowe` - 134 losowe klatki z puli na obrót;
- `kolejne` - kolejne klatki tych samych filmów;
- `tozsamosc` - klatki wyszukane jako podobne do obrazu FastGAN-a sterowanego do przewidzianej siły tożsamości;
- `current` - to samo, ale obraz sterowany do obecnej siły (bez prognozy; ramię kontrolne).

Zapisy uśredniono w blokach po 5 (punkt co 50 kroków), co tłumi szum. Prognoza startuje na początku każdego obrotu (co 500 kroków od kroku 1000), dopasowanie widzi tylko historię do startu, horyzont H = 1-20 bloków (50-1000 kroków). Punktów startu na ziarno: 14 (13 dla H ≥ 11) w iteracji 2 i 7 (6) w wersji 3.

Modele, każdy osobno, z Y(t) = siła tożsamości sondy w bloku t i ΔY(t) = Y(t) - Y(t-1) (kod `prognoza.py` tej samej pętli, co na Edenie):

```
Koopman:   ΔY(t+1) = Σ_{j=0..3} A_j · ΔY(t-j)
Langevin:  ΔY(t+1) = K · Y(t) + Σ_{j=0..3} B_j · ΔY(t-j) + a_p
```

Operator Koopmana to tu liniowa reguła ewolucji na wybranych zmiennych: kolejny przyrost jest kombinacją czterech poprzednich. „Langevin” to tu uproszczone równanie ruchu z hamowaniem do punktu równowagi i pamięcią (pełną postać z projektu TFT opisuje 5.5): człon K·Y(t) + a_p to hamowanie do własnego punktu stałego sondy p (a_p to stała danej sondy), a B_j to pamięć. Współczynniki A_j, K, B_j są wspólne dla wszystkich sond i dopasowane regresją grzbietową (najmniejsze kwadraty z karą λ = 0,01 za duże współczynniki, która stabilizuje dopasowanie). Każdy model liczony dwojako: **rekurencyjnie** (prognoza na jeden blok, potem od niej następny, jak w pętli) i **wprost** (osobne dopasowanie dla każdego H: Y(t+H) - Y(t) = f(cechy w chwili t), bez składania kroków). Odniesienia: brak zmiany oraz średnia zmiana z ostatnich 10 bloków przedłużona liniowo. Kryterium ustalonego przed wynikiem nie było: to pomiar opisowy [nie ustalono w źródłach innego celu].

**Wyniki: iteracja 2, mediana umiejętności po 20 ziarnach** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_horyzont_prognozy\wyniki\HORYZONT.json`):

| ramię | model | 50 kroków | 100 | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|---|
| bez | Koopman | 12,2 | 16,9 | 15,5 | 14,1 | 12,7 | 13,3 |
| bez | Langevin | 12,4 | 17,8 | 18,3 | 15,9 | 7,4 | 8,3 |
| bez | Koopman wprost | 12,2 | 17,6 | 17,0 | 15,4 | 18,6 | 18,2 |
| bez | Langevin wprost | 12,4 | 18,3 | 19,6 | 17,3 | 20,2 | 18,5 |
| losowe | Koopman | 16,2 | 19,1 | 17,5 | 14,5 | 13,9 | 12,8 |
| losowe | Langevin | 18,7 | 24,6 | 24,5 | 14,7 | 10,7 | 7,5 |
| losowe | Koopman wprost | 16,2 | 19,3 | 20,2 | 15,9 | 19,9 | 19,2 |
| losowe | Langevin wprost | 18,7 | 25,1 | 24,5 | 15,4 | 22,3 | 20,6 |
| kolejne | Koopman | 14,5 | 16,3 | 16,5 | 13,9 | 14,0 | 12,1 |
| kolejne | Langevin | 17,6 | 21,4 | 19,2 | 18,1 | 8,9 | 5,7 |
| kolejne | Koopman wprost | 14,5 | 16,2 | 18,2 | 16,1 | 19,7 | 19,2 |
| kolejne | Langevin wprost | 17,6 | 21,3 | 20,6 | 15,6 | 19,6 | 14,9 |
| tozsamosc | Koopman | 15,0 | 17,8 | 16,6 | 14,9 | 13,3 | 11,3 |
| tozsamosc | Langevin | 18,5 | 22,6 | 23,4 | 17,9 | 14,8 | 8,8 |
| tozsamosc | Koopman wprost | 15,0 | 18,2 | 19,5 | 18,0 | 21,9 | 20,2 |
| tozsamosc | Langevin wprost | 18,5 | 22,4 | 24,8 | 20,6 | 24,9 | 23,6 |
| current | Koopman | 12,9 | 15,0 | 14,8 | 13,1 | 10,5 | 9,2 |
| current | Langevin | 15,7 | 20,9 | 18,3 | 15,1 | 4,6 | 1,7 |
| current | Koopman wprost | 12,9 | 15,5 | 16,8 | 14,7 | 17,3 | 15,7 |
| current | Langevin wprost | 15,7 | 20,7 | 18,3 | 15,2 | 19,3 | 16,4 |
| każde | średnia zmiana | -10,3 do -12,9 | -23,2 do -26,1 | -66,1 do -72,6 | -162,9 do -171,2 | -255,7 do -288,7 | -382,2 do -416,9 |

**Wersja 3, mediana po 10 ziarnach** (ten sam obraz, większy szum, bo tylko 6-7 punktów startu na ziarno):

| ramię | Koopman (50 / 250 / 500 / 1000) | Langevin | Koopman wprost | Langevin wprost | średnia zmiana (1000) |
|---|---|---|---|---|---|
| bez | 9,8 / 14,7 / 12,7 / 10,1 | 12,8 / 20,2 / 15,0 / 11,3 | 9,8 / 14,2 / 15,2 / 23,7 | 12,8 / 21,6 / 6,1 / 20,2 | -389,7 |
| losowe | 13,8 / 13,8 / 13,0 / 7,8 | 19,7 / 19,5 / 17,6 / 11,7 | 13,8 / 12,1 / 7,7 / 21,8 | 19,7 / 23,6 / 4,6 / 2,2 | -359,8 |
| kolejne | 16,8 / 12,0 / 13,0 / 8,0 | 20,9 / 20,4 / 9,5 / 10,0 | 16,8 / 11,6 / 8,2 / 18,5 | 20,9 / 18,7 / -1,1 / 13,9 | -331,9 |
| tozsamosc | 16,4 / 16,6 / 13,3 / 9,9 | 22,3 / 23,1 / 18,4 / 14,3 | 16,4 / 16,8 / 13,7 / 19,6 | 22,3 / 22,2 / 14,4 / 5,5 | -370,5 |
| current | 10,8 / 14,5 / 11,4 / 7,5 | 17,7 / 17,5 / 13,9 / 0,0 | 10,8 / 14,0 / 7,4 / 17,0 | 17,7 / 19,7 / 7,1 / 15,6 | -447,0 |

**Rozrzut między ziarnami** (odsetek ziaren lepszych od braku zmiany, iteracja 2, według JSON): Koopman rekurencyjny 100 procent przy każdym H we wszystkich ramionach; Koopman wprost 100 procent poza horyzontami 550 i 650 kroków, gdzie spada do 85-95 procent; Langevin rekurencyjny 80-95 procent przy 500 krokach i 60-75 procent przy 1000 krokach (najsłabiej `current`: 60); Langevin wprost 75-95 procent przy 1000 krokach. `WYNIKI.md` podaje to okrąglej: Koopman i oba warianty „wprost” 100 procent przy każdym H, Langevin rekurencyjny 90-95 procent przy 500 i 70-75 procent przy 1000 krokach, Langevin wprost 90-95 procent przy 1000 krokach. W wersji 3 (mniej punktów startu) najniższy odsetek po wszystkich horyzontach wynosi: Koopman 90-100, Koopman wprost 50-70, Langevin 50-80, Langevin wprost 30-40 procent. Średnia zmiana: 0 procent ziaren przy każdym H.

**Interpretacja** (wnioski z `WYNIKI.md`):
1. Nie ma ostrej granicy horyzontu. Obie metody są lepsze od braku zmiany aż do 1000 kroków (20 bloków), ale tylko o 10-25 procent błędu kwadratowego: to słaby, stały sygnał, a nie prognoza dobra blisko i psująca się dalej.
2. Siła tożsamości nie ma trendu. Przedłużanie ostatniej średniej zmiany jest katastrofalne (około -160 przy 500 krokach). Modele wygrywają, bo przewidują powrót do poziomu sondy i wygaszanie wahań.
3. Langevin rekurencyjny jest najlepszy na 100-250 krokach (18-25), ale składanie kroków psuje go dalej (2-9 przy 1000 krokach). Koopman rekurencyjny jest płaski (12-17; przy 1000 krokach 9-13).
4. Dopasowanie osobno dla każdego horyzontu trzyma 15-25 aż do 1000 kroków, zgodnie z Lin i in. 2021 (składanie wielu małych kroków Koopmana mnoży błąd).
5. Na horyzoncie pętli (500 kroków): Langevin wprost 15-21, Koopman wprost 15-18, Langevin rekurencyjny 15-18, Koopman rekurencyjny 13-15.

**Literatura** (wyszukiwanie pogłębione Undermind „multi-step forecasting of a network latent with Koopman and generalized Langevin equations and the predictability horizon”, 205 prac): Lin i in. 2021 (doi:10.1137/21m1401759; dyskretne uogólnione równanie Langevina z formalizmu Mori-Zwanziga, prognoza w pętli otwartej z szumem ustawionym na zero; przy 1,3 czasu Lapunowa błąd Koopmana składanego z małych kroków około 105-165, Koopmana na docelowym kroku około 73, z pamięcią około 48); Lei, Baker, Li 2016 (doi:10.1073/pnas.1609587113; jądro pamięci, czyli funkcja mówiąca, jak mocno dawne stany wpływają na obecny krok, zapisane jako suma wykładników i zamienione na zmienne pomocnicze; szum barwny, czyli skorelowany w czasie, generowany w nich zgodnie z twierdzeniem fluktuacyjno-dyssypacyjnym); Lien i in. 2024 (doi:10.1103/PhysRevResearch.7.023042; szum barwny poprawia kalibrację rozrzutu prognoz, nie wydłuża horyzontu średniej); Aceituno i in. 2025 (doi:10.48550/arXiv.2506.03889; horyzont uczenia jako hiperparametr, uczenie tylko na jednym kroku rzadko optymalne); Pervez, Locatello 2026 (doi:10.48550/arXiv.2605.08856; przejściowe wzmocnienie operatorów nienormalnych psuje długie prognozy mimo stabilnych wartości własnych); Mamakoukas i in. 2020 (stabilny Koopman, promień spektralny ≤ 1); Dogra, Redman 2020 (Koopman przewiduje wagi sieci 2500 kroków naprzód, ale tylko późno w uczeniu); Tano i in. 2020 (DMD, numeryczne przybliżenie operatora Koopmana, z 14 zapisów najlepiej na około 41 kroków); Kunin i in. 2021 (SGD, czyli spadek gradientu na losowych paczkach, z momentem jako niedotłumione równanie Langevina, czas korelacji około η(1+β)/(1-β) kroków); Pashakhanloo, Koulakov 2023 (szum SGD daje błądzenie reprezentacji wzdłuż kierunków, które nie zmieniają straty). W odpowiedzi dla użytkownika dodano, że jego Koopman na TFT miał kierunek rosnący o prawie 7 procent na krok i dlatego rozjechał się po 3 dniach, oraz że w uczeniu ViT odpowiednikiem członu wymuszenia są zaplanowane dołożenia klatek, a członu zdarzeń początek obrotu.

**Czego to nie dowodzi.** Pomiar dotyczy jednej liczby na sondę, nie całego latentu 768. Starty są tylko na początkach obrotów, więc przy H > 10 prognoza przechodzi przez kolejne dołożenie klatek. Warianty „wprost” używają tych samych cech co pętla; pełnego równania Langevina z projektu TFT (dryf od prędkości i przyspieszenia, jądro pamięci z tłumionymi obrotami, wymuszenie, zdarzenia, szum barwny) tu nie sprawdzano.

**Co z tego wynikło.** Z samej historii siły tożsamości nie da się zbudować dobrej prognozy na setki kroków. Pół godziny później użytkownik zlecił przebudowę równania z TFT tak, żeby przewidywało cały latent krok po kroku, z danymi jako sterowaniem (5.5).

### 5.3 Literatura: generatory, które trafiają w zadany latent (rozgałęzienie)

**Skąd.** Tuż przed etapem (14:31-14:51) gałąź główna przeszukała literaturę poleceniem „szukaj artykułów które opisują jak generować obrazy przy pomocy stable diffusion które spełniają latent!” (Undermind, wyszukiwanie szybkie i pogłębione, 133 prace; Scite nie wykazał wycofań) i dopisała 40 linków w 6 grupach do kolejki pobierania `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDownloader\linki\stable-diffusion-latent-docelowy-sieci-2026-09-29.txt` (pobieranie nie ruszyło). O 15:34 użytkownik wkleił to podsumowanie do rozgałęzienia, a o 15:35 dodał: „mozsz uzyc undermind do czytania całych artykułów od poczatku do konca!”.

**Pojęcia.** **Model dyfuzyjny** uczy się odwracać stopniowe zaszumianie: z obrazu x0 robi się x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε (ε to szum gaussowski, ᾱ_t, „alfa z kreską”, to udział sygnału malejący od 1 do prawie 0 w miarę wzrostu t = 0..999), a sieć uczy się zgadywać ε. Rysowanie zaczyna się od czystego szumu i w wielu krokach zdejmuje szum. **Stable Diffusion** to gotowy, publiczny model tego typu uczony na miliardach zdjęć z podpisami (zbiór LAION), który zdejmuje szum nie z pikseli, lecz ze ściśniętej reprezentacji obrazu; tworzy ją koder VAE, a dekoder VAE odtwarza z niej obraz (szczegóły w 5.12). **DDIM** to szybki, deterministyczny sposób rysowania, który przechodzi przez 1000 poziomów szumu w kilkudziesięciu krokach. **CFG** (classifier-free guidance) to wzmacnianie warunku (np. etykiety) przez mieszanie przewidywań modelu z warunkiem i bez niego. **Prowadzenie** (guidance) to doliczanie w każdym kroku rysowania gradientu straty z zewnętrznej, zamrożonej sieci (u nas: małego ViT), liczonego na przewidywanym czystym obrazie x̂0, żeby rysunek zmierzał do zadanej wartości tej sieci. **Kontrfakt** to zdjęcie zmienione tak, żeby zamrożona sieć zmieniła decyzję, a scena prawie się nie zmieniła.

**Metoda.** Notatka `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_odwrotnosc_mlp\NOTATKA_LITERATURA.md` dostała sekcję z wklejki (commit `335c8528`): trzy sposoby - (1) prowadzenie stratą w trakcie odszumiania bez douczania, (2) optymalizacja szumu startowego (odpowiednik inwersji FastGAN-a), (3) douczony dodatek biorący latent na wejściu. Następnie każdą z 11 prac czytał w całości osobny podagent Undermind (`read_pdfs`) z pytaniami pod pętlę małego ViT; surowe odpowiedzi po angielsku leżą w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\fastgan_odwrotnosc_mlp\literatura\pelne_teksty_sd_2026-09-29.md` (1499 linii, około 123 KB), wnioski w notatce (commit `ddacbc74`). PDF Universal Guidance był dla Undermind nieczytelny.

**Prace i co z nich wynika dla pętli** (dziesięć pierwszych przeczytanych w całości 29.09; Rousselot przeczytany 28.09; Universal Guidance tylko ze streszczenia):

| praca | co robi | liczby | wniosek dla pętli |
|---|---|---|---|
| Shirahama i in. 2025 [Shi25f] | SD 1.4 bez tekstu, 512 × 512, 1000 kroków; w każdym kroku strata L2 między cechą sieci na x̂0 a celem, gradient przeskalowany do normy szumu i obcięty (2-3 sigma), 1000 powtórzeń na 5 pierwszych krokach, 8 na pozostałych | 2,5 h i około 15 GB (CLIP, ResNet-50), 4,7 h i 19,6 GB (ViT-H) na RTX 3090; „cosinus > 0,99” tylko dla celów będących przeskalowanymi cechami prawdziwych zdjęć (× 0,8 i × 1,2); przy ViT-H duże czarne plamy | za wolne: 84 obrazy na ziarno w iteracji 2 (6 na obrót × 14 obrotów) to około 210 h na ziarno; sieć przyjmie też obraz zniekształcony |
| Zhang i in. 2024 [Zha24p] | optymalizuje szum startowy modelu o budowie SD 2.1 (4 × 64 × 64), gradient przez 20 kroków próbkowania, 1500 iteracji Adam, lr 0,1, strata L2 cech + wygładzenie | ViT-base blok 5: PSNR (szczytowy stosunek sygnału do szumu odtworzonego obrazu) 32,5 dB; czasu i pamięci nie podano, kod niewydany | najbliżej „odwracania” cech ViT, ale bez kodu i kosztów |
| Karthikeyan 2026 [Kar26] | model dyfuzyjny w latencie uwarunkowany tokenem CLS (zbiorczym tokenem opisującym cały obraz) sieci DINOv2 ViT-B/14 (768 liczb), ImageNet-100, 256 × 256, 125 epok | +10,76 punktu klasyfikatora (79,26 procent przy 4 × danych syntetycznych wobec 68,5 dla modelu z samą klasą) | trzeba od nowa uczyć duży model; zgodności z celem nie mierzono |
| Farid i in. 2023, LDCE [Far23b] | kontrfakt: prawdziwe zdjęcie w latencie SD zaszumione tylko do kroku 191-250 z 1000, odszumianie z gradientem klasyfikatora na x̂0 i karą L1 za oddalenie; **zgoda kierunków**: gradient sieci zerowany tam, gdzie odbiega o więcej niż 45-55° od kierunku, który sam model dyfuzyjny uważa za zmianę klasy | 39 s na obraz (RTX 3090, 24 GB); kod MIT `lmb-freiburg/ldce`; ostrzeżenie przed domenami spoza modelu (np. medycyna) | lekkie zaszumienie zostawia narzędzie na miejscu, więc nie daje innych położeń; zgoda kierunków to gotowe lekarstwo na nieodporność małego ViT |
| Neukirch i in. 2025, FeatInv [Neu25] | ControlNet (douczana kopia części sieci SD, która wstrzykuje warunek przestrzenny) na MiniSD 256 × 256 uwarunkowany mapą cech ResNet-50, ConvNeXt, SwinV2 | uczenie 45-60 h na 2 × L40; cosinus cech z celem 0,46-0,61 dla ostatniej mapy, 0,96-0,98 dla wczesnych warstw ConvNeXt; sterowanie działa przez osłabianie składowej (× 0,25) | wzmacnianie kierunku wypycha obraz poza rozkład danych, a pętla prosi o wzrost siły |
| Yu i in. 2023, FreeDoM [Yu23] | gradient straty zamrożonej sieci na x̂0 dodawany w każdym kroku; „podróż w czasie” (ponowne zaszumienie i powtórzenie kroku) tylko w fazie 800-500 z 1000, gdzie powstaje semantyka | około 84 s na obraz 512 × 512 (RTX 3090) wobec około 40 min dla Universal Guidance; działa ze SD przez dekoder VAE; kod `vvictoryuki/FreeDoM` | rysuje od szumu, więc każdy start daje inne położenie narzędzia; wybrany jako „mocniejszy wariant” |
| Wallace i in. 2023, DOODL [Wal23b] | optymalizacja szumu startowego przez cały łańcuch 50 kroków SD 1.4, stała pamięć dzięki odwracalnemu próbnikowi EDICT | według [Eyr24] około 20 min i 40 GB na obraz | za drogie |
| Eyring i in. 2024, ReNO [Eyr24] | optymalizacja szumu startowego generatora jednokrokowego (SD-Turbo), 50 iteracji, regularyzacja normy szumu, obcięcie gradientu 0,1, zwracany najlepszy obraz z trajektorii | SD-Turbo 512 × 512: 20 s i 15 GB (A100, fp16); dowolna różniczkowalna strata | pasuje wprost do FastGAN-a (też generator jednokrokowy): start z wielu losowych wejść zamiast z kotwicy |
| Tang i in. 2024, DNO [Tan24d] | optymalizacja szumu startowego i wszystkich szumów wstrzykiwanych w losowym próbniku (DDIM η = 1, czyli z nowym szumem w każdym kroku), regularyzacja prawdopodobieństwa szumu przeciw „oszukiwaniu nagrody” | poniżej 1 min przy 15 krokach, 15-20 GB (A800) | możliwe, ale tylko na Edenie |
| Frisch i in. 2023 [Fri23b] | własny model dyfuzyjny DDIM na CATARACTS, 128 × 128, warunkowany fazą i zestawem narzędzi (CFG, w = 2), 500 epok na 2 × A40 | 30 000 obrazów syntetycznych: F1 (średnia harmoniczna precyzji i czułości) klasyfikatora narzędzi 0,897 → 0,916, w rzadkich fazach do +0,10; sam syntetyk F1 0,299; klinicyści mylili obrazy z prawdziwymi w 61 procent par; FID (odległość Frécheta rozkładów cech sieci Inception, mniej znaczy bliżej prawdziwych zdjęć) 43,7 | wzorzec dla własnego modelu na klatkach zaćmy; tylko obecność narzędzia, bez detekcji i końcówki |
| Rousselot, Wendebourg, Köthe 2026 | próbkowanie ze „zbioru niezmienniczego” φ⁻¹(h): wiele realistycznych obrazów, które sieć widzi tak samo; wariant bez uczenia z dynamicznym celem | około 1,8 min na próbkę (A40); dodatek A: gradient wprost w pikselach daje szum w stylu ataku | celowo rysuje różne obrazy o tej samej reprezentacji, najbliżej celu „różne położenia narzędzia”; tożsamość jako wąska wiązka kierunków to ten sam zbiór niezmienniczy |
| Bansal i in. 2023, Universal Guidance | prowadzenie dowolną siecią bez przeuczania modelu | PDF nieczytelny; około 40 min na obraz według porównania w [Yu23] | nieocenione z pełnego tekstu |

**Wnioski z pełnych tekstów** (z notatki): (1) każdy wariant ze Stable Diffusion 512 × 512 potrzebuje co najmniej 15 GB pamięci karty, więc lokalny RTX 3070 (8 GB) odpada i zostaje Eden; (2) najbliżej pętli jest schemat LDCE (start od prawdziwej kotwicy, częściowe zaszumienie, gradient siły tożsamości do celu s0 + Δs, kara za oddalenie; około 1 h na ziarno przy 84 obrazach); (3) Stable Diffusion nie widziało obrazów z mikroskopu operacyjnego, więc FastGAN uczony na naszych wycinkach ma przewagę domeny; (4) wzmacnianie kierunku jest trudniejsze niż osłabianie; (5) sieć nieodporna przyjmie obraz zniekształcony, więc obraz powinien służyć tylko do wyszukania prawdziwych klatek; (6) żadna praca nie zmienia wyniku wersji 3 pętli: zysk daje samo wyszukiwanie podobnych klatek, a lepszy generator nie doda udziału prognozy. Poprawka do wklejki: „cosinus powyżej 0,99” u Shirahamy dotyczy tylko jednego, łatwego testu.

**Rozmowa o metodzie Farida (16:07-16:25).** Użytkownik dopytywał: „tylko te podpowiedzi sieci, z którymi zgadza się sam generator. tego nie rozumiem”, „jaka mała siec? Generator to fastgan?”, „jak vit może proponować zmiane czegokolwiek?”, „czyli bierzemy jakis obraz i zmieniamy piksele ale to nie zmienia np połozenia narzedzia na zdjeciu?”. Wyjaśnienie: „podpowiedź” sieci to mapa wrażliwości (gradient siły tożsamości po pikselach, liczony jednym przejściem wstecz); generator porównuje odszumianie z etykietą celu i bez niej, a różnica to jego własny kierunek; w każdym kawałku obrazu przechodzi tylko zmiana, której kąt z kierunkiem generatora jest mniejszy niż około 45°. Generatorem w pracy Farida jest Stable Diffusion, a FastGAN się nie nada, bo nie ma przełącznika „z narzędziem / bez narzędzia”. Lekkie zaszumienie celowo zostawia narzędzie na miejscu. Użytkownik: „ok czyli ta metoda nie da takich efektów jakich ja chce czyli np rózne pozycje narzedzia?”, a potem: „no to podaj mi inną metodę ta odpada. Co jeszcze wyczytałeś w literaturze?”. Odpowiedź: najprostsze jest ReNO (dobór szumu startowego FastGAN-a z wielu losowych startów), mocniejsze FreeDoM (model dyfuzyjny rysuje od zera, mały ViT po drodze popycha obraz do celu; model trzeba nauczyć na naszych wycinkach, jak Frisch); Shirahama jest za wolny, Karthikeyan wymaga uczenia dużego modelu. Zastrzeżenie z tej rozmowy: cel z Koopmana i Langevina to tylko jedna liczba, więc położenie narzędzia wybiera generator przypadkiem.

**Co z tego wynikło.** O 16:29 użytkownik wybrał drogę FreeDoM (5.4).

### 5.4 Własny model dyfuzyjny wycinków narzędzia w pikselach (`dyfuzja_narzedzie`, rozgałęzienie)

**Prośba.** 16:29: „opracuj jak wytrenowac taki model a edenie i potem go pobrac i generowac na moim kompie obrazy ewentualnie na edenie podczas testowania jak generowac takie obrazy opracuj jak to zrobic wygenerj kody i zatrzymaj sie jak skonczysz i bedziesz chciał to wysłac na eden i wytrenowac”. W trakcie pracy użytkownik dopisał: „ale mozesz uczyc na danych dostepnych na edenie czyli z filmów i z ich klatek”, „wtedy model nauczy sie dobrze generowac obrazy” (16:48), „jak to eden zawiera tylko wybrane klatki? eden ma cały zestaw treningowy. Nie potrzebujesz końcówki narzedzia tyko etykietę narzedzia i jego obrys!” (16:52), „wydaje mi sie ze nie musisz lokalnie wycinac mozesz to podaczas treningu robic łądowac filmy i adnotacje do poszczególnych klatek i miec w ten sposób wyciete narzedzie z klatki juz bezposrednio z filmu!” (17:05), wreszcie „No i tak zrób i potem roZpicznij trening na edenie tego stable diffusuon” (17:19; asystent zaznaczył, że to nie Stable Diffusion, tylko własny, mniejszy model tego samego rodzaju) oraz „i pusc to a edenie trenin g” i „a potem przygotuj i sprawdzczy wrzucenie filmiw sie zmiesci na edenie i spróbuj kilka filmów wrzuuc na eden” (17:26). Cel według `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dyfuzja_narzedzie\PLAN.md`: zamiennik FastGAN-a, który rysuje różne ułożenia narzędzia.

**Dane: trzy kolejne wersje.**
1. Z klatek puli (`kod/przygotuj_dane.py`): 10 421 okien 240 px ze środkiem w końcówce narzędzia, `dane\okna240.npy` (2,4 GB). Porzucona.
2. Z całych filmów lokalnie (`kod/przygotuj_filmy.py`): co trzecia klatka 118 filmów `train` z `pool_manifest.jsonl` (filmy mają razem 313 359 klatek, 281 348 z maską narzędzia), okno 240 px ze środkiem w środku prostokąta opisanego na obrysie cystotomu, 4 kanały (RGB + obrys), klatka 720 × 480 zmniejszona do 480 × 320 (skala małego ViT). Wynik: 83 091 okien w jednym pliku `okna.bin` (1,59 GB, JPEG jakości 95 + PNG obrysu); w filmach `TR_0019` (307 klatek) i `TR_0057` (183) część adnotacji nie miała obrazu. Próba na 2 filmach: 2547 okien w 13 s, około 18 KB na okno. Wersja zapasowa.
3. **Wariant użyty na Edenie**: sprawdzenie Edenu (17:11-17:16) pokazało, że nie ma tam filmów ani obrysów; jest `~/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/context_train.tar` (4,9 GB, 57 073 klatek = co piąta klatka 118 filmów `train`, bez adnotacji), `tracking_static36` (22 680 klatek, co piąta klatka 143 filmów, z prostokątem i końcówką), `tracking_calibration` (12 543 obrazy) i małe zbiory próbne. Pliki JPEG na Edenie są bajt w bajt takie same jak lokalne (SHA256 trzech klatek). Wysłano więc tylko obrysy: `kod/przygotuj_obrysy.py` → `dane\obrysy\obrysy.npz` (56 005 klatek z obrysem, 1068 bez, 23 MB). Wycinanie w trakcie uczenia: klatka dowolnego rozmiaru → 480 × 320 → maska z wielokąta → losowy kwadrat 176 × 176 zawierający środek obrysu z marginesem 16 px → losowe odbicie poziome.

**Model i uczenie** (`kod/model_dyfuzji.py`, `kod/trenuj.py`). UNet (sieć konwolucyjna w kształcie litery U: zwęża obraz do coraz mniejszych rozdzielczości i rozszerza z powrotem, z połączeniami na skróty między poziomami) o 22 292 420 parametrach: poziomy 176, 88, 44, 22 px, kanały 64, 128, 192, 256, warstwa uwagi na 22 px. Przewiduje szum (schemat DDPM), harmonogram kosinusowy ᾱ_t, T = 1000, bez warunku (zawsze rysuje wycinek z narzędziem i jego obrysem jako czwarty kanał). Paczka 64, AdamW lr 2·10⁻⁴ z rozgrzewką 1000 kroków, bf16, obcięcie normy gradientu 1, EMA wag 0,9995 (EMA: wolno uśredniana kopia wag, w(EMA) ← 0,9995·w(EMA) + 0,0005·w, dająca gładszy model do generowania). Zapisy: `ostatni.pt` co 1000 kroków (wznowienie), wagi EMA co 10 000, podgląd 16 próbek co 5000, `LOG.jsonl`, `STATUS.json`; plik `STOP` tworzony przez SLURM przed końcem czasu powoduje zapis i czyste wyjście. Budżet: 100 000 kroków × 64 = 6,4 mln przykładów.

**Generowanie z prowadzeniem** (`kod/generuj.py`). Od czystego szumu, próbnik DDIM w 50 krokach. W każdym kroku: x̂0 → mały ViT (wagi z pętli, np. `wspolny/k02.pt`) → siła tożsamości na tokenach wskazanych przez wygenerowany obrys → strata ((s - s_cel)/skala)² → gradient przez ViT i UNet → poprawka przewidzianego szumu. Cel: `--cel` albo siła sondy-kotwicy + zmiana z Koopmana lub Langevina. Trzy różnice wobec FreeDoM nazwane do zatwierdzenia: (1) poprawka w przestrzeni szumu z normowaniem gradientu do normy przewidzianego szumu (jak u Shirahamy), waga `--waga` 0,3; (2) „podróż w czasie” (ponowne zaszumienie obrazu o jeden krok wstecz i powtórzenie kroku, żeby prowadzenie się utrwaliło) tylko dla t/T w [0,5; 0,8], 3 powtórzenia; (3) siła liczona na 4 zaburzonych kopiach obrazu (jasność, kontrast, szum, rozmycie), żeby prowadzenie nie szło w niewidoczny wzorek pikseli.

**Sprawdzenia lokalne (RTX 3070, 8 GB).** Uczenie 100 kroków z wznowieniem od kroku 60, paczka 8: 0,27 s na krok, 5,9 GB; dekodowanie danych 4 ms na paczkę. Generowanie modelem nieuczonym (test przepływu gradientu), cel 6,59: bez prowadzenia siła odporna 3,83, waga 0,3 daje 6,27, waga 1,0 daje 7,24; około 1-2 s na obraz przy 20 krokach. W próbie wariantu 3 jeden krok trwał 5,6 s przy 5,6 ms na przykład i 1,6 ms na odczyt JPEG; w tym czasie tę samą kartę zajmowało nagranie ziarna 48 z gałęzi głównej [przyczyny nie rozstrzygnięto w źródłach].

**Eden.** Katalog `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie/`. Wysyłka `eden/wyslij.sh` (kod, obrysy, wektory tożsamości, sumy kontrolne zgodne). Próba 1818351 padła po 35 s: klatka `TR_0027_S1_P03_00000021.jpg` ma 720 × 576, a kod zakładał 720 × 480; poprawka: każda klatka sprowadzana do 480 × 320. Próba 1818352 (500 kroków, 4 min 11 s): strata 0,973 → 0,765 → 0,426 → 0,146 → 0,046 (kroki 100-500), a po 500 krokach w czterech przedziałach poziomu szumu (od t < 250 do t ≥ 750) 0,084 / 0,038 / 0,032 / 0,029; 0,385 s na krok na H100 PCIe; 41,1 GB pamięci karty; na wagach bez uśredniania po 500 krokach widać zarys oka pod szumem. Pełne uczenie: 1818354 (100 000 kroków, paczka 64) i fragmenty 1818355, 1818356 z `--dependency=afterany` (każdy 6 h; szacunek całości 10,7 h). Próbna wysyłka filmów: 3 archiwa (1,38 GB) w 322 s, około 4,3 MB/s, sumy zgodne, w `~/smallvit_runs/2026-09-29_dyfuzja_narzedzie/filmy_zip/`; wszystkie 118 filmów `train` to 28,4 GB (około 1 h 50 min).

**Stan według plików.** Po 17:47 (29.09) w zapisie rozmowy nie ma już wpisów rozgałęzienia; zadania na Edenie działały dalej same, a gałąź główna traktowała katalog jako należący do „innej sesji” (zapis sesji głównej: „Wątek przejęła dalej równoległa sesja”; zapisu takiej dalszej pracy nie ma w źródłach tej części). Lokalnie w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dyfuzja_narzedzie\wyniki\eden_proba\` leżą tylko wyniki próby (`LOG.jsonl`, `STATUS.json`, `ostatni.pt` 357 MB, podglądy). Gałąź główna odczytała Eden 30.09 o 00:28: 1818354 zakończone po 5 h 57 min 53 s, 1818355 działa od 48 min, 1818356 czeka; `STATUS.json` z 00:25:46: krok 62 000 z 100 000, strata 0,0108, 0,388 s na krok, 4,1 h do końca, 41,14 GB; strata w tych samych czterech przedziałach t przy kroku 62 300: 0,0323 / 0,0064 / 0,0026 / 0,0008; zapisane wagi `ckpt_010000` do `ckpt_060000` i podglądy co 5000 do `podglad_060000.png`. O 09:49 gałąź główna podała, że model „skończył uczenie, wszystkie 100 tysięcy kroków”. Czy fragment 1818356 w ogóle ruszył, i jaka jest jakość końcowego modelu: [nie ustalono w źródłach].

**Diagnoza złych kolorów (gałąź główna, 30.09, 00:28-00:33).** Podglądy z kroków 10 000 i 60 000 miały dziwne, przesycone kolory, obecne już przy 10 000 krokach, co wskazywało na rysowanie, a nie uczenie. Siatka kroków DDIM `siatka_krokow(T, kroki) = linspace(T-1, 0, kroki).round()` zaczyna od t = 999 (dalej 979, 958, 938, …). Przewidziany czysty obraz liczy się jako x̂0 = (x_t - √(1-ᾱ_t)·ε̂)/√ᾱ_t, więc błąd przewidzianego szumu jest mnożony przez około 1/√ᾱ_t. Wartości √ᾱ_t z harmonogramu w kodzie: t = 999: 0,00005; 998: 0,00156; 990: 0,01403; 979: 0,03116; 959: 0,0623; 900: 0,15368. Przy t = 999 wzmocnienie wynosi około 20 000 (przeliczenie: 1/√ᾱ_999 ≈ 20 291), obraz po obcięciu do [-1, 1] dostaje losowy, nasycony kolor i zachowuje go do końca; przy t = 979 wzmocnienie to około 32. Test lokalny na pobranych wagach `ckpt_060000.pt` (89 MB), 16 próbek, 50 kroków (średnie nasycenie = średni rozrzut kanałów RGB w pikselu; średnie RGB):

| start rysowania | arytmetyka | nasycenie | średnie RGB |
|---|---|---|---|
| t = 999 | bf16 | 0,568 | 0,528 / 0,484 / 0,461 |
| t = 999 | float32 | 0,616 | 0,537 / 0,506 / 0,468 |
| t = 979 | bf16 | 0,180 | 0,499 / 0,490 / 0,503 |
| t = 959 | bf16 | 0,121 | 0,497 / 0,495 / 0,504 |
| prawdziwe wycinki (podgląd danych) | - | 0,163 | 0,484 / 0,477 / 0,534 |

Precyzja nie jest winna (float32 też przesyca), start od t = 979 daje kolory zbliżone do danych; widać oko, tęczówkę i czasem narzędzie, jeszcze trochę rozmyte. Poprawka to jedna linia w funkcji `siatka_krokow` w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dyfuzja_narzedzie\kod\model_dyfuzji.py` (z tej samej funkcji korzysta `generuj.py`); uczenia nie trzeba przerywać. **Poprawki nie wprowadzono**, bo folder należy do tamtej sesji; do końca etapu pozostaje niewprowadzona.

**Czego to nie dowodzi.** Według planu ani dobra jakość obrazów, ani trafienie w siłę tożsamości nie oznaczają szybszego uczenia detektora; wersja 3 pętli pokazała, że zysk daje samo wyszukiwanie podobnych klatek. Ocena jakości (odległość próbek od najbliższych okien uczących, dobór `--waga`, różnorodność położeń) nie została wykonana w tym etapie.

### 5.5 Równanie ruchu latentu małego ViT ze sterowaniem danymi (T0, T1, T2)

**Prośba.** 15:33: „zajmmij się przmianą tego równania Langevina z TFT na takie które będzie przewidywał latent. Możesz wprowadzac jakos dane jako sterowanie. RAczej nie bedzie wymuszenia ale szum moze byc. Sprobój iteracyjnnie wyprowadzac to rownanie az bedziesz miał chociaz w jednym kroku naprzód w przewidywaniu latentu dokładnosc powyzej 80%”. Katalog: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_rownanie_latentu` (założony według reguły „jeden eksperyment, jeden katalog”; repozytorium ViT nie ma rejestru wyników projektu grantowego, więc deklaracja i opisy leżą w `PLAN.md` i `WYNIKI_*.md`).

**Punkt wyjścia z TFT.** W projekcie grantowym (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56`, `tft_langevin_czlon_zdarzen`) równanie dla latentu TFT ma postać z(t+1) = z(t) + D + JP + W + Z + N. Równanie Langevina to tu równanie ruchu cząstki z bezwładnością, siłami deterministycznymi i siłą losową, rozszerzone o pamięć. Człony: **dryf** D (siła zależna od położenia, prędkości i przyspieszenia), **jądro pamięci** JP (wpływ dawnych prędkości na obecny krok, zapisany jako suma tłumionych, obracających się wykładników), **wymuszenie** W (wpływ znanych z zewnątrz sygnałów, tam kalendarza i promocji), człon **zdarzeń** Z oraz **szum kolorowy** N (szum skorelowany w czasie, modelowany jako warunkowa średnia z własnych, niedawnych reszt równania).

**Rozpoznanie przed planem** (ziarno 47, ramię `bez`, zapis co 10 kroków z Edenu): przyrost latentu na 10 kroków ma normę około 2,0 przy normie latentu około 6; kolejne przyrosty są ujemnie skorelowane (-0,22); najlepsza prosta prognoza z samej historii (powrót do średniej ruchomej) daje około 17; statystyki pikseli paczki (jasność, kontrast, odbicia) odtworzone z generatora losowego nie tłumaczą przyrostu (dopasowanie 7, test ujemny). Wniosek: ruch między zapisami wynika z kolejnych paczek, a bez sterowania danymi 80 jest nieosiągalne.

**Plan i kryteria ustalone przed wynikiem** (`PLAN.md`). Cel: U > 80 na części testowej (dopasowanie na wcześniejszych krokach, test na późniejszych). Podłogi: brak zmiany (0), najlepsze równanie bez sterowania (T1) i **placebo**: te same cechy paczki, ale z kroku przesuniętego o 250, z dopasowaniem od nowa; placebo musi spaść do poziomu T1, inaczej zysk bierze się z przecieku informacji albo nadmiernego dopasowania. Dostęp do informacji: sterowanie widzi tylko to, co znane przed aktualizacją w kroku t (skład paczki i augmentację, latent obrazów paczki w przejściu uczącym, gradient straty względem tego latentu), a nie widzi gradientu sond, samej zmiany wag ani latentu po kroku. Obalenie celu: U testowe ≤ 80 po wyczerpaniu członów z uzasadnieniem fizycznym albo zysk, który daje też placebo.

**T0: nagranie uczenia krok po kroku** (`nagranie.py`). Lokalne uczenie identyczne z Edenem: ziarno 47, 3000 kroków, baza 1500 klatek, paczka 32, AdamW lr 5·10⁻⁴, rozgrzewka 300 kroków, obcinanie gradientu 1, bf16, kolejność paczek z tych samych generatorów losowych co na Edenie (`default_rng(s)` na krokach 1-1000, `default_rng(1000·s + 3)` dalej, kolejka zerowana co 500 kroków). W każdym kroku zapisywano: latent 192 sond (narzędzie i tło, float16), wejście fc1 ostatniego bloku, skład i augmentację paczki, latent obrazów paczki przed aktualizacją i gradient straty względem niego (float32, bo w float16 wartości rzędu 7·10⁻⁶ znikały), stratę, normę gradientu, lr. Co 10 kroków dokładny rozkład przyrostu latentu przez podmianę wag: część od samych wag, od samych buforów BatchNorm (BatchNorm: warstwa normująca aktywacje, której średnie ruchome są przechowywane w „buforach” i zmieniają się bez gradientu) i od samych wag ostatniego bloku. Dwa nagrania na RTX 3070: `nagranie_s47` (367 s) i `nagranie_s47_sily` (984 s, z siłami od kroku 900), kilka GB, poza gitem.

Zgodność z Edenem (ziarno 47, ramię `bez`, kroki od 1000, co 10 kroków; kryterium obalenia T0: inna statystyka):

| wielkość | nagranie | Eden |
|---|---|---|
| norma przyrostu latentu narzędzia na 10 kroków | 2,288 | 2,271 |
| korelacja kolejnych przyrostów (10 kroków) | -0,238 | -0,237 |
| norma latentu | 6,37 | 5,89 |

T0 zdany. Co krok ruch wygląda inaczej: norma przyrostu 0,40, korelacja kolejnych przyrostów +0,75 (pęd AdamW); na 10 krokach ta pamięć gaśnie i odwraca znak. Rozkład przyrostu: same wagi wyjaśniają 97,8 procent, same bufory BatchNorm 0,7 procent, same wagi ostatniego bloku 7,5 procent. Latent porusza się więc przez zmianę wag całej sieci.

**T1: równanie z TFT bez sterowania** (`t1_bez_sterowania.py`, `wyniki\T1.json`). Każda z 768 współrzędnych osobno, współczynniki wspólne dla sond (wariant `diag`) albo jedne dla wszystkich współrzędnych (`glob`), v(t) = Δz(t):

```
Δz(t+1) = D + JP + N
D  = -Σ_τ γ_τ·(z(t) - EMA_τ(z)(t)) + a_v·v(t) + a_a·(v(t) - v(t-1)),   τ = 2, 4, 8, 16, 32, 64, 128
JP = Σ_m [α_m·Re c_m(t) + β_m·Im c_m(t)],   c_m(t) = ρ_m·e^{iω_m}·c_m(t-1) + v(t),
     ρ ∈ {0,5; 0,8; 0,9; 0,97},  ω ∈ {0, π/2, π}
N  = te same jądra na własnych resztach r(t) = v(t) - (D + JP)(t-1)   (drugi etap, jak w TFT)
```

EMA_τ to średnia ruchoma o stałej czasowej τ kroków; c_m to tłumione (ρ < 1), obracające się (ω) filtry prędkości, czyli jądro pamięci. Dopasowanie grzbietowe, λ = 10⁻³. Wymuszenia i zdarzeń nie ma („raczej nie będzie wymuszenia”). Wyniki (umiejętność testowa):

| dane | glob | diag | diag + N | diag na dopasowaniu |
|---|---|---|---|---|
| Eden co 10 kroków, mediana 10 ziaren | 17,59 | 16,79 | 16,35 | 21,33 |
| Eden, zakres po ziarnach (glob) | 16,20-19,76 | | | |
| nagranie co krok | 58,01 | 58,20 | 58,21 | 62,77 |
| nagranie co 10 kroków | 17,47 | 15,60 | 14,60 | 20,72 |

Sama prędkość a·v(t) z a = 0,76: 56,5 na nagraniu co krok; pełne T1 z członem N przeliczonym w T2: 60,2. Wniosek: z samej historii jeden krok da się przewidzieć w około 60 procentach, 10 kroków w około 17.

**T2: sterowanie danymi** (`t2_sterowanie.py`, `wyniki\T2.json`; dopasowanie na krokach 1000-2000, test 2000-2999, nagranie `nagranie_s47_sily`). Kluczowe pojęcia to dwie siły, liczone w pętli uczenia po `loss.backward()` i obcięciu gradientu, a przed `opt.step()`, jednym dodatkowym przejściem sond przez sieć w float32 (tryb ewaluacji, te same bufory BatchNorm, przez `torch.func.functional_call`):

```
S(t)       = z(θ(t) + d_nowa(t)) - z(θ(t))          siła paczki (sterowanie danymi)
S_pęd(t)   = z(θ(t) + d_pęd(t))  - z(θ(t))          siła z pędu
S_zanik(t) = z(θ(t) + d_zanik(t)) - z(θ(t))
d_nowa(t)  = -lr/(1-β1^k) · (1-β1) · g(t)   / M(t)
d_pęd(t)   = -lr/(1-β1^k) · β1 · m(t-1)     / M(t)
d_zanik(t) = -lr · wd · θ(t)
M(t)       = sqrt(β2·v(t-1) + (1-β2)·g(t)²) / sqrt(1-β2^k) + eps
```

Tu θ(t) to wagi po t aktualizacjach, z(θ) latent sond (kanał narzędzia) policzony siecią o wagach θ, g(t) gradient straty paczki t po obcięciu normy do 1, m(t-1) i v(t-1) stan AdamW sprzed kroku, k numer kroku (czynniki 1-β^k poprawiają start średnich od zera), dzielenie przez M(t) po współrzędnych. Suma d_nowa + d_pęd + d_zanik to dokładnie krok AdamW. Słowami: **siła paczki** S(t) mówi, o ile przesunąłby się latent sondy, gdyby sieć wykonała tylko tę część kroku, która pochodzi z gradientu bieżącej paczki; **siła z pędu** S_pęd(t) to przesunięcie od tej części kroku, która pochodzi z pamięci wcześniejszych paczek.

Warianty i wyniki (umiejętność testowa):

| wariant | co widzi | U |
|---|---|---|
| t1 (równanie z TFT bez sterowania) | historia latentu | 60,17 |
| t1 + siła paczki | + S(t) | **85,79** |
| t1 + siła paczki z kroku -250 (placebo, dopasowane od nowa) | + zła paczka | 60,16 |
| t1 + „tanie jądro” bez przeliczania sieci | + Σ_b cos(h_p, h_b)·g_b i jego średnia ruchoma 0,9 (h: wejście fc1 ostatniego bloku, g: gradient po latencie obrazu paczki) | 60,17 |
| t1 + siła paczki + pęd | + S(t) + S_pęd(t) | 95,31 |
| t1 + wszystkie siły | + S_zanik(t) | 95,31 |
| same siły | b·S + c·S_pęd + e·S_zanik, bez historii latentu | 95,36 |

Równanie proste, dwa współczynniki wspólne dla wszystkich współrzędnych i sond:

```
Δz(t+1) = a·Δz(t) + b·S(t) + ε(t),   a = 0,834,  b = 0,985     U testowe 85,4
(sama historia: Δz(t+1) = 0,758·Δz(t),  U = 56,5)
```

Mediana umiejętności po krokach 86,3 (10. percentyl 78,7), mediana po sondach 87,5, najgorsza sonda 71,5. Zanik wag jest pomijalny (norma 0,0015 przy przyroście 0,37). Współczynniki wariantu „same siły” dla pierwszych 8 współrzędnych: 0,97-0,99 przy S, 0,98-0,99 przy S_pęd i od -0,06 do 0,01 przy S_zanik, czyli rzeczywisty krok latentu to prawie dokładnie suma sił (sieć odpowiada liniowo na tak małą zmianę wag).

**Interpretacja fizyczna.** Latent sondy zachowuje się jak cząstka z bezwładnością popychana przez dane (niedotłumiony ruch Langevina w czasie dyskretnym). Człon a·Δz(t) to **bezwładność**: latent zachowuje około 83 procent prędkości z poprzedniego kroku, bo AdamW ma pęd β1 = 0,9, częściowo zjadany przez zmianę wrażliwości sieci i normowanie kroku. Człon b·S(t) to siła danych, wchodząca ze współczynnikiem około 1, czyli wprost. Zamiana historii latentu na zapamiętany pęd optymalizatora daje 95, bo 0,83·Δz(t) to tylko przybliżenie pędu w wagach. Placebo spada dokładnie do 60,2, więc zysk nie jest przeciekiem. Tanie jądro nic nie daje, zgodnie z T0: sam ostatni blok niesie tylko 7,5 procent ruchu.

**Zastrzeżenia (z `WYNIKI_T2.md`).** Jedno ziarno i jedno nagranie lokalne (replikacja w 5.6). Równanie działa na jeden krok: siła dla dalszych kroków wymaga przyszłych wag. Siła wymaga przeliczenia sieci, nie jest zamkniętym wzorem. Tylko ramię bez dokładania klatek. **Najważniejsze zastrzeżenie wyszło później (5.7): S(t) korzysta z paczki kroku t, czyli z klatek, na których sieć uczy się właśnie w przewidywanym kroku. Wynik 85 procent to tryb wyjaśnienia, a nie prognoza z wyprzedzeniem.** Commit `019030f4` (16:34).

### 5.6 Replikacja na 10 ziarnach Edenu (zadania 1818328 i 1818331)

**Prośba i plan.** 16:30: „możesz to wysyłać na eden zeby policzyc”. Plan nazwany przed `sbatch`: to samo nagranie i to samo równanie na ziarnach 47-56, bez nowych ramion i bez zmian metody; ciężkie nagrania (około 6 GB na ziarno) tylko w `/dev/shm` węzła (pamięć RAM widziana jak dysk) i kasowane pułapką (`trap`: polecenie powłoki wykonywane przy każdym zakończeniu skryptu); do katalogu wyników wraca jeden JSON na ziarno. Kryterium z planu: U jednego kroku powyżej 80 na każdym ziarnie i placebo równe wariantowi bez sterowania.

**Aparat.** Skrypt `rownanie_eden.slurm` (1 karta, 8 procesorów, 96 GB RAM, limit 2 h, partycje `hopper,short`, `--signal=B:TERM@120`, `--no-requeue`), katalog `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_rownanie_latentu/` (kod w `kod/`, wejścia z `2026-09-28_maly_vit_od_zera/wejscia`, biblioteka timm 0.9.16 z `deps`, archiwum `tracking_static36.tar`); przed startem sprawdza wszystkie wejścia i pomija ziarno z gotowym wynikiem. Kwota przed zgłoszeniem: 118 558 424 KB (113,1 GiB), 159 863 pliki.

**Próba 1818328** (ziarno 47, 1300 kroków, dopasowanie 1000-1150, test 1150-1299, 182 s liczenia równania): t1 63,09; t1 + paczka 91,58; placebo 63,08; tanie jądro 63,10; + pęd 96,85; same siły 96,87; równanie proste 91,46 (a = 0,8445, b = 0,988); sama historia 58,7 (a = 0,7514); mediana po krokach 92,1 (10. percentyl 87,32); mediana po sondach 92,2; najgorsza sonda 83,4; ruch co 10 kroków: norma 2,2788, korelacja -0,2802. `/dev/shm` po próbie czyste.

**Kampania 1818331** (tablica zadań 0-9, 3000 kroków, dopasowanie 1000-2000, test 2000-2999). Start 16:42:55-16:42:57 dla ośmiu ziaren, ziarna 55 i 56 o 17:00:27-28. Węzły: 6 × H100 PCIe (`hopper`: ziarna 47, 48, 49, 50, 55, 56) i 4 × A100-SXM4-40GB (`dgx-4`: 51; `dgx-2`: 52, 53; `dgx-1`: 54). Nagranie 3000 kroków z siłami trwało około 365 s na H100 (5,6 GB), cały bieg 17-19 min na ziarno (szacunek przed startem mówił o 25 min). Po biegu sprawdzono brak katalogów `rownlat-*` w `/dev/shm` na `hopper`, `dgx-1`, `dgx-2`, `dgx-4`. Wyniki pobrane skryptem `zbierz_eden.py` do `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_rownanie_latentu\wyniki_eden\` (`T2_s47.json` … `T2_s56.json`, `ZBIORCZO.json`, `proba_s47.json`, `logi\rownlat_1818331_0..9.out`), opis `WYNIKI_EDEN.md`, commit `5a2ac3b1`. W `nagranie.py` dodano opcje `--sily-od` i `--wagi-co`.

**Wyniki na każdym ziarnie** (umiejętność testowa):

| ziarno | t1 | t1 + paczka | placebo | t1 + paczka + pęd | same siły | równanie proste | a | b | 10. percentyl kroków | najgorsza sonda |
|---|---|---|---|---|---|---|---|---|---|---|
| 47 | 58,3 | 84,0 | 58,3 | 94,6 | 94,6 | 83,6 | 0,828 | 0,983 | 76,4 | 68,1 |
| 48 | 62,3 | 88,8 | 62,3 | 95,6 | 95,7 | 88,5 | 0,837 | 0,980 | 83,1 | 79,1 |
| 49 | 63,4 | 89,4 | 63,4 | 95,8 | 95,8 | 89,2 | 0,846 | 0,988 | 84,5 | 77,5 |
| 50 | 61,1 | 87,8 | 61,1 | 95,4 | 95,5 | 87,5 | 0,830 | 0,980 | 81,8 | 74,4 |
| 51 | 55,5 | 81,3 | 55,5 | 93,9 | 93,9 | 80,5 | 0,817 | 0,979 | 71,6 | 61,3 |
| 52 | 58,0 | 83,6 | 57,9 | 94,4 | 94,4 | 83,2 | 0,833 | 0,983 | 74,7 | 62,8 |
| 53 | 59,8 | 86,7 | 59,8 | 95,1 | 95,2 | 86,4 | 0,838 | 0,982 | 80,4 | 75,6 |
| 54 | 58,4 | 83,4 | 58,4 | 94,6 | 94,7 | 82,8 | 0,816 | 0,980 | 75,2 | 66,2 |
| 55 | 61,7 | 88,0 | 61,7 | 95,3 | 95,3 | 87,7 | 0,842 | 0,990 | 82,0 | 79,9 |
| 56 | 57,9 | 83,0 | 57,9 | 94,4 | 94,5 | 82,4 | 0,821 | 0,977 | 75,2 | 68,2 |
| **mediana** | 59,1 | **85,4** | 59,1 | 94,9 | 94,9 | **85,0** | 0,831 | 0,981 | 78,4 | 71,3 |
| zakres | 55,5-63,4 | 81,3-89,4 | 55,5-63,4 | 93,9-95,8 | 93,9-95,9 | 80,5-89,2 | 0,816-0,846 | 0,977-0,990 | 71,6-84,5 | 61,3-79,9 |

Dodatkowe liczby z plików `T2_s*.json`:

| ziarno | sama historia a·Δz(t) | a samo | mediana po krokach | mediana po sondach | ruch co 1 krok: norma / korelacja | ruch co 10 kroków: norma / korelacja | sekundy liczenia równania |
|---|---|---|---|---|---|---|---|
| 47 | 55,17 | 0,7546 | 84,96 | 85,51 | 0,393 / 0,749 | 2,252 / -0,243 | 671 |
| 48 | 58,27 | 0,7557 | 88,60 | 88,92 | 0,450 / 0,759 | 2,551 / -0,253 | 673 |
| 49 | 59,44 | 0,7632 | 89,39 | 89,73 | 0,389 / 0,767 | 2,230 / -0,238 | 670 |
| 50 | 57,13 | 0,7430 | 87,93 | 88,50 | 0,410 / 0,749 | 2,274 / -0,264 | 670 |
| 51 | 51,78 | 0,7388 | 82,53 | 84,47 | 0,432 / 0,730 | 2,411 / -0,258 | 660 |
| 52 | 54,62 | 0,7511 | 84,20 | 84,84 | 0,422 / 0,746 | 2,365 / -0,260 | 669 |
| 53 | 55,49 | 0,7500 | 86,70 | 87,26 | 0,455 / 0,748 | 2,551 / -0,272 | 659 |
| 54 | 54,35 | 0,7435 | 84,19 | 84,48 | 0,459 / 0,740 | 2,594 / -0,269 | 630 |
| 55 | 59,06 | 0,7742 | 87,98 | 88,61 | 0,356 / 0,772 | 2,071 / -0,257 | 674 |
| 56 | 54,24 | 0,7401 | 83,36 | 84,13 | 0,456 / 0,738 | 2,580 / -0,266 | 673 |
| mediana (zakres) | 55,33 (51,78-59,44) | 0,751 (0,739-0,774) | 85,83 (82,53-89,39) | 86,39 (84,13-89,73) | | 2,388 (2,071-2,594) / -0,259 | |

Mediana współczynników z 10 ziaren daje równanie Δz(t+1) = 0,83·Δz(t) + 0,98·S(t) + ε(t).

**Werdykt.** Cel spełniony na każdym ziarnie: t1 + paczka 81,3-89,4, równanie proste 80,5-89,2 (10 z 10 ponad 80). Zysk ze sterowania danymi: +26 punktów w medianie. Placebo równe wariantowi bez sterowania w każdym ziarnie (największa różnica 0,01 punktu). Współczynniki stałe między ziarnami (a 0,82-0,85, b 0,98-0,99), czyli ta sama fizyka w każdym przebiegu. Tanie jądro znów nic nie daje. Pęd zamiast historii podnosi wynik do 93,9-95,8.

**Zgodność z biegami pętli** (Eden v2, ten sam start i te same paczki, kroki 1000-2990, co 10 kroków): norma przyrostu 2,39 w nagraniu wobec 2,49 w zapisie pętli (mediana 10 ziaren; zakresy 2,071-2,594 i 2,181-2,671), korelacja -0,259 wobec -0,262. Trajektorie nie są identyczne liczba w liczbę: ziarno 47 na H100 daje 84,0, lokalnie na RTX 3070 85,8, bo różnice numeryczne kart rozchodzą się w trakcie uczenia.

**Czego to nie dowodzi.** Jeden krok naprzód. Tylko ramię bez dokładania klatek i kroki 1000-3000. Dziesięć ziaren to dziesięć przebiegów tego samego uczenia; inne zbiory i architektury nie są sprawdzone. I ponownie: S(t) to siła paczki przewidywanego kroku (5.7).

### 5.7 Reguła o przyszłych klatkach: wyjaśnienie a prognoza

O 17:27 (29.09), w trakcie budowy równania z macierzami, użytkownik ustanowił regułę: „pamietaj ze do równania nie moga wchodzi przyszłe klatki to jest niedozwoone w prognozie”. Asystent przyjął ją ściśle: paczka kroku, który się przewiduje, też jest przyszła; w prognozie wolno używać stanów do chwili t, paczek do t-1 i stanu optymalizatora. Regułę zapisano na stałe w pamięci Claude Code (`C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-bez-przyszlych-klatek-w-prognozie.md`) i dopisano do `WYNIKI_T2.md` i `WYNIKI_EDEN.md` (commit `c85c75c6`).

O 23:46 użytkownik zapytał wprost: „pytanie czy używałeś klatek które były w przyszłości do przewidywania latentu”. Odpowiedź była jawnym ujawnieniem: tak, wynik 85 procent (lokalnie i na Edenie) brał paczkę przewidywanego kroku, czyli klatki, na których sieć uczy się właśnie w tym kroku; to samo dotyczy wariantów 95 procent. Według reguły ten wynik tłumaczy ruch latentu, ale nie jest prognozą. Uczciwe wyniki bez przyszłych klatek to 57-67 procent (5.8). Nazwano też jedyny pozostały ślad paczki t w S_pęd: mianownik M(t) zawiera g(t)² z wagą 1 - β2 = 0,001 (przeciek uznany za pomijalny; czysta wersja wymagałaby mianownika z samego v(t-1)).

O 00:01 (30.09) użytkownik doprecyzował: „ale tłumaczenie aktualnego latentu tymi klatkami które stworzyły ten latent w tej chwili to ok!”. Od tej chwili obowiązują dwa tryby: **wyjaśnienie** (latent z chwili t+1 tłumaczony paczką t, która go stworzyła: dozwolone, 85 procent) i **prognoza z wyprzedzeniem** (bez klatek z kroków, których jeszcze nie było: 57-67 procent). Opisy wyników i reguła w pamięci zostały poprawione (commit `4a0c9a92`, 00:03).

### 5.8 Uczone macierze jak w TS (`maly_vit_rownanie_macierze`)

**Pytanie i skąd się wzięło.** O 17:03 użytkownik zapytał: „czyli jak to może pomóc w ustalaniu latentu na kolejnych krokach uczenia? czy to poprawi generowanie obrazów?” oraz „czy nie lepiej by było jakby trenowac jadro pamieci i macierz wymuszenia tak jak to robiłem w TS ?”. Odpowiedź: równanie działa na jeden krok, więc samo nie ustali latentu na setki kroków; generowania nie poprawi, bo zmiana latentu w jednym kroku jest niewidoczna na obrazie; jądro pamięci daje tylko około 4 punktów ponad bezwładność, a brakuje właśnie wymuszenia (siła paczki daje około 25 punktów); paczki przyszłych kroków są znane z góry (kolejkę losuje się z ziarna), więc wyuczona **macierz wymuszenia** (przełożenie cech paczki obrazów na ruch latentu) mogłaby prognozować wiele kroków bez przeliczania sieci; ryzyko: siła zależy od błędu sieci na paczce, a ten zmienia się z wagami. O 17:07: „chodzi mi o to ze w TS te macierze byly bardzo duze i je sie tez trzeba było trenowac zeby ustawlic parametry. Moze tutaj w równaniu dla latentu i wektorów tożsamości trzeba również zrobic takie macirze? podejrzyj jak to bylo liczone w Project!”. O 17:13: „no to nie mozesz tutaj zrobic takich macierzy do równania langewina dla tych wektorów latentu tożsamowsic?”, co asystent uznał za zgodę na lokalny eksperyment. Od 17:27 obowiązuje reguła o przyszłych klatkach, więc wymuszenie mogło widzieć tylko paczki już użyte.

**Jak liczono w TS** (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56\model.py`). Równanie na latencie TFT (wymiar 192, krok = dzień, okno 56 dni, 85 klatek-dni w sekwencji): z(k+1) = z(k) + D(k) + JP(k) + W(k) + N(k), człony bez wspólnych parametrów, uczone razem jednym błędem kroku.
- **Dryf D**: MLP (LayerNorm, Linear 576 → n_ukr, GELU, Linear, GELU, Linear → 192) na znormalizowanych z, v, a; n_ukr 512, w wariancie szerokim 1024 (D około 1,8 mln parametrów).
- **Jądro pamięci JP**: uwaga (mechanizm ważenia dawnych kroków podobieństwem stanów) po 56 dawnych położeniach, 4 głowy, rząd 48, waga dawnego kroku = zanik ρ_h^l × obrót o ω_h·l × podobieństwo stanów (uwaga liniowa elu+1), początkowe czasy połowicznego zaniku 4, 10, 25, 56 dni, wejście przez Linear 384 → 256.
- **Wymuszenie W**: wariant `Wymuszenie` (plan promocji i kalendarz jutra, 16 wolnych filtrów po 56 dni, MLP 384) albo `WymuszeniePelne` (cały dzień wchodzący do okna TFT: 24 z 74 kanałów, 32 lub 64 wyuczone filtry po 57 dniach, MLP 768 lub 1536). Według `WYNIKI_L2.md`: `L2_dzien` z W 0,97 mln parametrów i `L2_dzien_szeroki` z D 1,8 mln i W 3,2 mln.
- **Szum kolorowy N**: 4 głowy niskiego rzędu (32), jądra g_m(l) = ρ_m^(l-1)·cos(ω_m·(l-1)), l = 1..55, początkowe czasy połowiczne 1, 3, 7, 21 dni, ω = 0, 0, 2π/7, 2π/7; działa na resztach r(j) = (z(j+1) - z(j)) - (D + JP + W)(j).
- Uczenie: AdamW lr 10⁻³, wd 10⁻⁴, partia 1024 cząstek (serii sklep-produkt, 72 600 w uczeniu), kara rozdzielności 0,05 (średni kwadrat cosinusa między wyjściami członów, żeby nie robiły tego samego), EMA wag 0,995 w wariantach szerokich, obcięcie gradientu 1. W TS bez wymuszenia wynik spadał o 30-60 punktów, a jądro pamięci dawało mniej niż punkt (z odpowiedzi asystenta o 17:09).

**Błąd znaleziony w kodzie TS.** W `SzumKolorowy` splot ma dopełnienie OKNO - 1 = 55 przy jądrze długości 55, a potem jeszcze przesunięcie o jeden. W efekcie N(k) = Σ_{l=1..55} g(l)·r(k-1-l), czyli N widzi reszty od dnia k-2, a nie od k-1, jak mówi komentarz w kodzie (impuls reszty w dniu 5 daje pierwsze wyjście w dniu 7; sprawdzone liczbowo). W `tft_langevin_czlon_zdarzen` resztę z dnia k-1 dodają osobno odczepy `a1` (po przekątnej); w biegach L1-L3 `tft_langevin_latent_56` jej brakuje. W wersji dla ViT splot poprawiono (dopełnienie OKNO - 2, pierwsze wyjście w dniu 6). Kodu TS nie zmieniano.

**Przełożenie TS → ViT** (każda różnica nazwana w `PLAN.md`):

| TS (TFT, Favorita) | tu | różnica |
|---|---|---|
| cząstka = seria sklep-produkt, 72 600 w uczeniu | cząstka = sonda, 120 sond z narzędziem z wycinków „fit” | około 600 razy mniej cząstek, wszystkie z jednego przebiegu uczenia |
| krok = dzień | krok = jedna aktualizacja AdamW | - |
| stan = latent TFT 192 | stan = latent narzędzia sondy 768 | - |
| D: MLP z położenia, prędkości, przyspieszenia | to samo | - |
| JP: uwaga po 56 dniach z zanikiem i obrotem | to samo, okno 32 kroki | krótsze, bo pęd AdamW gaśnie do 4 procent po 30 krokach |
| W: MLP z nowego dnia + 32-64 filtry po 57 dniach | uwaga sondy na obrazy paczek t-1..t-16 (podobieństwo sondy do obrazu × sygnał błędu obrazu) z wyuczoną wagą każdej paczki, + MLP | wejściem są paczki już użyte, nie kalendarz; TS widział „jutro” (plan promocji), tu paczki kroku t nie ma; pamięć 16 paczek zamiast 57 dni |
| N: warunkowa średnia szumu z reszt 55 dni | to samo, 31 kroków | - |
| AdamW 10⁻³, wd 10⁻⁴, partia 1024, EMA 0,995, kara 0,05 | to samo; partia = 120 sond × 64 kroki | - |

**Model dla ViT** (`model.py`), z v(t) = z(t) - z(t-1), a(t) = v(t) - v(t-1):

```
z(t+1) = z(t) + D(t) + JP(t) + W(t) + N(t)
D(t)  = MLP(z, v, a)                                   2304 → 1024 → 1024 → 768
JP(t) = Σ_{l=0..31} A_h(t, t-l) · V·n(t-l),   A_h = ρ_h^l · obrót(ω_h·l) · podobieństwo stanów
W(t)  = Wo·A(t) + f_W(A(t), z(t))
A(t)  = Σ_{k=0..15} κ_h(k) · (1/32) · Σ_{b ∈ paczka t-1-k} <q_h(sonda, t), k_h(b)>/√r · v_h(b)
N(t)  = Σ_{l=1..31} Σ_m g_m(l) · U_m·V_m^T · r(t-l),   r = krok - (D + JP + W)
```

W wymuszeniu zapytanie q pochodzi ze stanu sondy (latent narzędzia i tła), klucz k z latentu obrazu paczki (obszar narzędzia i cały obraz) oraz flagi „ma narzędzie”, a wartość v z **sygnału błędu obrazu**, czyli gradientu straty po latencie tego obrazu, zapisanego w kroku, w którym paczka była użyta (w ramieniu `bez_gradientu` wartość to te same cechy co klucz). Bez softmaksu i bez normowania wag: obraz podobny do sondy z dużym błędem pcha ją mocniej, a podobieństwo może mieć znak ujemny. κ_h(k) to wyuczona waga paczki t-1-k, startująca od 0,9^k jak pęd AdamW; 8 głów rzędu 64. JP: 4 głowy rzędu 48, okno 32, początkowe czasy połowiczne 2, 4, 8, 16 kroków. N: 4 głowy rzędu 32, l = 1..31, początkowo 1, 3, 7, 21 kroków i ω = 0, 0, 0,3, 1,0. Parametry: D 4 201 728, JP 689 252 (z projekcją wejścia), W 5 911 170 (5 911 682 w `bez_gradientu`), N 196 616, razem około 11 mln.

**Ramiona, podział i kryteria** (`uczenie.py`, `liniowe.py`). Pierwsza wersja wymuszenia widziała paczkę bieżącego kroku; po regule z 17:27, jeszcze przed pełnym uczeniem, przesunięto ją na paczki t-1..t-16. Ramiona: `pelne` (D + JP + W + N, W z sygnałem błędu), `bez_gradientu` (W tylko z latentów obrazów i flagi), `placebo` (paczki t-251..t-266 zamiast t-1..t-16, uczone od nowa), `bez_W` (D + JP + N). Nagrania: `nagranie_s47_sily` i nowe lokalne `nagranie_s48_sily` (862 s, siły od kroku 2000, wagi sieci co 100 kroków od 1000 w `wagi\k01000.pt` … `k03000.pt`, 21 plików, 4,6 GB). Każde ziarno uczone i oceniane osobno, bo wektory tożsamości ziaren 47 i 48 mają cosinus 0,085 (inny układ neuronów; pierwotny plan przenoszenia z 47 na 48 porzucono). Uczenie na krokach 1000-1800 (120 sond „fit”), wybór epoki na 1800-2000 (według umiejętności latentu, zapisywane wagi EMA), test na 2000-2999 na sondach „fit” i na 60 sondach „holdout”. Miary: umiejętność na pełnym latencie 768, na sile tożsamości (rzut przyrostu na kierunek LDA z kroku 1000, własny dla ziarna) i na „wektorze” (średniej po sondach). Konfiguracja: lr 10⁻³, wd 10⁻⁴, rozgrzewka 200, do 80 epok po 20 iteracji na losowych oknach 64 kroków, cierpliwość 12, obniżanie lr po 3 epokach bez poprawy, kara 0,05, obcięcie 1, EMA 0,995. Odniesienia dopasowane na krokach uczenia ziarna 47 (dla ziarna 48 przeniesione, bo to skalary): bezwładność a·Δz(t), pęd a·Δz(t) + c·S_pęd(t) i, tylko dla porównania, a·Δz(t) + b·S(t) z paczką bieżącą (w pliku oznaczone `paczka_biezaca_NIEDOZWOLONE`). Kryteria obalenia z planu: `pelne` na teście nie wyżej niż `bez_W` albo `placebo` (wymuszenie nic nie wnosi); `pelne` poniżej odniesienia z pędem (uczone wymuszenie nie zastępuje przeliczenia sieci); duży spadek na sondach „holdout” albo na ziarnie 48 (równanie wkuwa sondy albo przebieg). Liczone lokalnie na RTX 3070 jako proces odłączony; jeden bieg 52-137 s, najlepsze epoki 30-37.

**Wyniki na teście (umiejętność; `wyniki\BIEG_s*.json`, `LINIOWE.json`, `ODNIESIENIA.json`):**

| równanie | co widzi | s47 fit: latent / tożsamość / wektor | s47 holdout: latent / tożsamość | s48 fit: latent / tożsamość / wektor | s48 holdout: latent / tożsamość |
|---|---|---|---|---|---|
| duże sieci D+JP+W+N (`pelne`) | stany do t, paczki t-1..t-16 z sygnałem błędu | 39,52 / 16,95 / 37,78 | 38,60 / 17,13 | 40,78 / 32,99 / 42,71 | 39,08 / 24,93 |
| bez W (`bez_W`) | stany do t | 39,62 / 14,85 / 39,05 | 38,46 / 13,27 | 40,87 / 33,69 / 43,55 | 38,90 / 25,66 |
| placebo (paczki sprzed 250 kroków) | - | 39,32 / 15,41 / 37,70 | 38,42 / 14,83 | 40,31 / 31,17 / 42,18 | 38,68 / 23,91 |
| W bez sygnału błędu (`bez_gradientu`) | latenty obrazów paczek t-1..t-16 | 39,60 / 16,26 / 38,71 | 38,63 / 16,56 | 40,52 / 34,61 / 42,85 | 38,72 / 27,16 |
| pełna macierz A·Δz(t) (L1) | stany do t | 52,03 / 42,98 / 52,72 | 50,99 / 41,28 | 50,81 / 47,54 / 52,53 | 50,00 / 46,12 |
| pełne macierze, 3 ostatnie przyrosty (L2) | stany do t | 53,43 / 42,52 / 55,04 | 52,32 / 39,69 | 53,58 / 50,69 / 56,88 | 51,96 / 47,32 |
| A·Δz(t) + P·S_pęd(t) (L3) | + pęd liczony siecią | 60,90 / 53,64 / 56,73 | 59,69 / 51,62 | - (siły s48 dopiero od kroku 2000) | - |
| **jedna liczba: 0,76·Δz(t)** | stany do t | **56,79 / 56,63 / 63,34** | 55,91 / 55,59 | **56,26 / 56,77 / 60,52** | 55,34 / 55,26 |
| **pęd: 0,94·S_pęd(t) - 0,04·Δz(t)** | pęd AdamW liczony siecią | **66,78 / 65,70 / 65,35** | 65,74 / 64,82 | **64,10 / 63,75 / 62,17** | 64,88 / 64,10 |
| wyjaśnienie, nie prognoza: 0,84·Δz(t) + 0,99·S(t) | paczka, która tworzy latent t+1 | 85,31 / 85,67 / 96,33 | 85,43 / 85,27 | 87,84 / 89,06 / 96,73 | 85,21 / 84,97 |

Współczynniki odniesień: bezwładność a = 0,759 (dla s48 dopasowane osobno 0,7497 daje 56,27); pęd: a = -0,0375, c = 0,9354; paczka bieżąca: a = 0,8419, b = 0,9864. Pełne macierze 768 × 768 dopasowane regresją grzbietową z λ wybieraną z {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1} na krokach 1800-2000 (wybrano 0,1; umiejętność na wyborze: L1 55,82 i 54,88, L2 58,94 i 58,36, L3 63,79).

Wyłączanie członów w ocenie (latent, sondy „fit”, test):

| bieg | wszystkie człony | bez D | bez JP | bez W | bez N |
|---|---|---|---|---|---|
| s47 `pelne` | 39,52 | 29,91 | 38,03 | 39,80 | 22,57 |
| s47 `bez_W` | 39,62 | 28,42 | 38,50 | - | 24,41 |
| s47 `placebo` | 39,32 | 30,03 | 37,93 | 39,84 | 21,55 |
| s47 `bez_gradientu` | 39,60 | 29,87 | 37,83 | 39,84 | 24,46 |
| s48 `pelne` | 40,78 | 29,25 | 39,05 | 41,18 | 26,77 |
| s48 `bez_W` | 40,87 | 27,93 | 39,18 | - | 29,71 |
| s48 `placebo` | 40,31 | 28,13 | 38,48 | 40,86 | 27,46 |
| s48 `bez_gradientu` | 40,52 | 27,95 | 38,78 | 40,82 | 29,63 |

Wyuczone jądra (s47 `pelne`; s48 w nawiasach): czasy połowiczne JP 1,42 / 3,42 / 7,38 / 16,17 kroku (1,54 / 3,14 / 7,58 / 15,88), obrót JP prawie zerowy (średnie |ω| 0,004-0,0055); N 0,79 / 2,37 / 5,21 / 15,25 (0,80 / 2,29 / 5,38 / 16,24), ω_N = 0, 0, 0,366, 0,993; wagi paczek κ (średnia po głowach) od 0,88 dla paczki t-1 do 0,08 dla t-16 (s48: od 0,99 do 0,20), czyli wymuszenie nauczyło się zaniku podobnego do pędu. Krzywe uczenia (`TRENING_*.csv`): najlepsza umiejętność na oknie wyboru 48,9-50,5 (latent) i 37,2-40,0 (tożsamość); końcowa strata uczenia 0,10-0,13, co `WYNIKI.md` opisuje jako około 89 procent na krokach uczenia.

**Werdykt (z `WYNIKI.md`, commit `37a022cd`).**
- Duże uczone sieci przegrywają z jedną liczbą bezwładności o 16-17 punktów: na krokach uczenia około 89, na oknie wyboru około 50, na teście około 40. „Wkuwają” okres uczenia (przeuczenie: dopasowują szczegóły jednego odcinka zamiast reguły). To samo co w TS (tam 94 procent na cząstkach uczących i 62 na nowych), tylko ostrzej, bo cały zbiór to jeden przebieg: 800 kroków × 120 sond na 11 mln parametrów.
- Wymuszenie z przeszłych paczek nic nie wnosi: `pelne` = `bez_W` = `placebo` = `bez_gradientu` (różnice względem `pelne` poniżej 0,5 punktu na latencie), a wyłączenie W w ocenie nic nie zmienia. Wpływ dawnych paczek niesie już pęd, a ten widać w prędkości latentu. Z tabeli wyłączania członów (przeliczenie własne, nie z `WYNIKI.md`): najwięcej z uczonych członów wnosi N (bez niego spadek o 11-18 punktów), potem D (9-13), JP tylko 1-2.
- Pełne macierze liniowe też przegrywają ze skalarem (52-54 wobec 57): sprzężenie 768 współrzędnych dopasowuje szum okresu uczenia. Ruch jest w każdej współrzędnej ten sam: 76 procent poprzedniego kroku.
- Najlepsza uczciwa prognoza to pęd optymalizatora przepuszczony przez sieć: 64-67 procent (współczynnik 0,94 przy S_pęd, bezwładność dostaje -0,04, czyli pęd zastępuje ją w całości). Bez przeliczania sieci zostaje bezwładność, 56-57 procent (równanie T1 z jądrami na każdej współrzędnej: 59 w medianie 10 ziaren Edenu).
- Ponad 80 procent daje tylko paczka, która tworzy dany latent: dozwolone jako wyjaśnienie, niedozwolone jako prognoza.
- Sondy „holdout” dają to samo z dokładnością do 1-2 punktów, więc równania nie wkuwają konkretnych sond.

**Czego to nie dowodzi.** Dwa ziarna, jedno ziarno równania na ramię, jedna budowa sieci (z TS); mniejsze sieci albo uczenie na wielu przebiegach naraz mogłyby wkuwać mniej. Tylko jeden krok naprzód i ramię bez dokładania klatek. Siła pędu ma w mianowniku AdamW nowy gradient z wagą 0,001 (nazwane w `dane.py`). Planowany krok dalej, prognoza na 10 i 100 kroków (wymuszenie gaśnie z pamięcią κ, przyszłe paczki nie wchodzą, wagi s48 co 100 kroków do sprawdzeń), nie został wykonany.

### 5.9 Zapis metody przewidywania latentu

O 00:24 (30.09) użytkownik poprosił: „Zapisz ten metodę przewidywania latentu”. Powstał dokument `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md` (commit `1a85dfbc`, 00:26, odnośnik w `docs\README.md`; w pamięci Claude Code notatka `metoda-przewidywania-latentu-vit.md`). Zawiera:

- co się przewiduje (latent sondy 768, krok AdamW, siła wektora tożsamości w układzie danej sieci) i miarę;
- dwa tryby według reguły użytkownika z 29.09 i 30.09;
- trzy równania: wyjaśnienie Δz(t+1) = a·Δz(t) + b·S(t) + szum (a ≈ 0,83, b ≈ 0,98); prognoza Δz(t+1) = c·S_pęd(t) + szum (c ≈ 0,94); prognoza bez przeliczania sieci Δz(t+1) = a₀·Δz(t) + szum (a₀ ≈ 0,76); współczynniki to zwykłe najmniejsze kwadraty, wspólne dla 768 współrzędnych i sond, przenośne między przebiegami (a 0,816-0,846, b 0,977-0,990);
- procedurę liczenia sił w pętli uczenia (funkcje `kierunki()` i `z_fp32()` z `nagranie.py`); koszt: dwa lub trzy dodatkowe przejścia 192 sond na krok; nagranie 3000 kroków z siłami od kroku 900 trwa 16 min na RTX 3070, około 6 min na H100 i 7 min na A100;
- tabelę trafności: wyjaśnienie 85,0 (mediana 10 ziaren, 80,5-89,2; siła tożsamości 85,7 dla s47 i 89,1 dla s48); prognoza z pędem 64,1-66,8 (siła tożsamości 63,8-65,7); bezwładność 56,3-56,8 (56,6-56,8); jądra T1 59,1;
- listę rzeczy, które nie działają (duże sieci jak w TFT 40 procent, wymuszenie z przeszłych paczek równe placebo, macierze 768 × 768 52-54, tanie jądro, statystyki pikseli, bufory BatchNorm 0,7 procent ruchu);
- granice: jeden krok AdamW; na 10 krokach sama historia daje około 17 procent; prognoza wielu kroków z siłą pędu niesprawdzona; siły wymagają dostępu do sieci i optymalizatora w trakcie uczenia.

### 5.10 Eden: karty, kwota i sprzątanie

**Pytania użytkownika (30.09, 00:27-00:48)** i odpowiedzi:
- „A co z tym modelem stable diffusion na edenie?” - to nie Stable Diffusion, tylko mały model pikselowy z rozgałęzienia; stan i diagnoza kolorów w 5.4.
- „Ale jeśli się uczy? Na ilu GPU?” - na jednej karcie H100; skrypt prosi o jedną kartę, a kod uczenia nie umie dzielić pracy; 0,39 s na krok, około 4 h do końca.
- „Zobacz czy inne karty są wolne?” - wolnych było 7 kart: 1 H100 na `hopper` i 6 A100 (4 na `dgx-1`, 2 na `dgx-4`), ale na `dgx-1` i `dgx-4` wolnych było tylko 8-9 procesorów na węzeł, więc zmieściłoby się 1-2 zadania; na wolną H100 czekały zadania innej osoby. Przeróbka na wiele kart zajęłaby prawie tyle, ile dokończenie uczenia; niczego nie zgłoszono.
- „Sprawdź ile jest dobrego miejsca na moim edenie?” - **kwota** (limit miejsca i liczby plików projektu na systemie plików Lustre, sprawdzany poleceniem `lfs quota -p 1299 /mnt/evafs`): zajęte 117 z 200 GiB i 168 499 z 200 000 plików; pliki są bliżej granicy niż gigabajty. Na krótko można przekroczyć oba limity o 10 procent (220 GiB, 220 000 plików), potem zapis zostaje zablokowany.
- „Sprawdź ile tam można usunąć?” - analiza bez usuwania (01:03).

**Analiza** (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-30.md`, commit `13166fa0`). Metoda: `lfs quota -p 1299`, `du` na każdym katalogu (bez `du -sh ~`), pełna lista 84 tys. plików ośmiu katalogów wyników porównana z dyskiem lokalnym po ścieżce i rozmiarze. Od 28.09 przybyło 65 GB, prawie wyłącznie zrzuty latentu pętli małego ViT. Trzy grupy:

| grupa | co | GB | pliki |
|---|---|---|---|
| A, bez straty | rozpakowane `tracking_static36/` (2,04 GB, 22 724 pliki; archiwum `.tar` obok), rozpakowane `tracking_calibration/` (1,13 GB, 12 555), `~/.vscode-server` (1,29 GB, 8550), `2026-09-21_koopman_langevin/package/frames` (0,43 GB, 4800), `~/.triton` (0,11 GB, 8337) | około 4,9 | około 57 tys. |
| B, z kopią lokalną | `2026-09-28_maly_vit_wektor` (7,38 GB, lokalnie 6,10; tylko na Edenie 1,19 GB wag `wspolny/k#.pt`), `2026-09-28_maly_vit_od_zera/wyniki` (7,99 GB, lokalnie 5,80; zostawić `wejscia/` 0,73 GB i `deps/`) | około 11,9 | - |
| C, tylko na Edenie | `2026-09-29_maly_vit_sila_it2` (42,75 GB, w tym 1400 plików `LATENT.npz` po około 26 MB = 36,7 GB), `2026-09-28_maly_vit_sila` (11,59 GB, 9,2 GB zrzutów; zostawić `petla_s*/wspolny/k02.pt`, wagi ViT dla dyfuzji) | około 48 | - |

Na dysku F: zostało 14 GB, więc zrzutów grupy C nie dało się pobrać; ich odtworzenie wymaga powtórzenia kampanii (około 30 min na ziarno). A + B + C to około 65 GB (kwota spadłaby do około 52 GB).

**Wykonanie po „Usuwaj” (01:31).** Przed usunięciem sprawdzono, że żaden skrypt ani trwające zadanie nie korzysta z usuwanych plików i że każdy plik rozpakowanych kopii jest w swoim archiwum. Usunięto: rozpakowane `tracking_static36/` i `tracking_calibration/`, `2026-09-21_koopman_langevin/package/frames`, `~/.triton` oraz katalogi ramion (`bez`, `koopman`, `losowe`, `current`, `kolejne`, `tozsamosc`) w `wyniki/petla_s*` czterech pętli: `2026-09-28_maly_vit_od_zera` (30 katalogów), `2026-09-28_maly_vit_wektor` (30), `2026-09-28_maly_vit_sila` (50), `2026-09-29_maly_vit_sila_it2` (100). Zostawiono celowo: `wspolny/` każdej pętli (wagi rozgrzewki; `wspolny/k02.pt` z `maly_vit_sila` wczytuje `dyfuzja_narzedzie/eden/generuj_eden.slurm`), `WEKTOR.json` i `WEKTOR.npz`, `maly_vit_od_zera/wejscia` i `deps`, `~/.vscode-server` (VS Code był akurat połączony z Edenem). Nietknięte: `~/grant_tft` (29,2 GB), `~/venvs` (9,5 GB), `~/.conda`, `~/.cache`, archiwa zbiorów. Wynik: **117 GiB → 56,7 GiB, 168 501 → 70 504 plików** (lfs: 122 818 840 KB → 59 450 300 KB). Commit `1f4de9d1`; notatki pamięci `eden-po-sprzataniu-2026-09-28.md` i `eden-zbiory-klatek-bez-obrysow.md` zaktualizowane. Surowe zrzuty latentu czterech pętli zostały bezpowrotnie usunięte za zgodą użytkownika.

**Limity kart konta** (sprawdzone 30.09 o 10:14 poleceniem `sacctmgr`): konto `transformers_vsc` ma łączne limity dla wszystkich użytkowników (bfajdek, bpiotrowski, bsiemiatkowska, mlange): 600 procesorów, 17 × A100, **5 × H100**, 3 × tesla, **20 kart razem**, 5 TB pamięci. `hopper` ma 8 × H100 i limit czasu 5 dni, węzły `dgx` po 8 × A100 40 GB i limit 24 h. Ośmiu H100 w jednym zadaniu nie da się więc dostać.

**Odczyty kwoty w etapie** (lfs podaje KB = 1024 bajty; przeliczenia na GiB moje):

| czas | KB | GiB | pliki | kontekst |
|---|---|---|---|---|
| 29.09 16:31 | 118 558 404 | 113,1 | 159 850 | przed kampanią 1818331 (w rozmowie „118,6 z 200 GB”) |
| 29.09 17:41 | 121 834 404 | 116,2 | 168 473 | po wysłaniu danych dyfuzji (w PLAN „121,8 z 200 GB”) |
| 30.09 00:47 | 122 729 916 | 117,0 | 168 499 | przed sprzątaniem |
| 30.09 01:35 | 59 450 300 | 56,7 | 70 504 | po sprzątaniu |
| 30.09 09:49 | 107 471 436 | 102,5 | 70 703 | po wysyłce filmów |
| 30.09 10:13 | 109 767 788 | 104,7 | 73 330 | po wysyłce kodu i wag SD |
| 30.09 11:54 | 138 858 420 | 132,4 | 73 812 | po przygotowaniu latentów (w zapisie sesji „około 139 z 200 GiB”) |

### 5.11 Wysyłka Cataract-LMM 3_Object_Tracking na Eden

**Prośba.** 01:32: „Potem wrzuć filmy z llm cataract”, „Na mój node eden”, „Na mój dysk”.

**Skrypt** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\wyslij_cataract_lmm.sh`: źródło `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking`, cel `/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/` (`videos/`, `annotations/`, `README.md`, `SHA256SUMS`). Trzy równoległe strumienie `scp`; plik o tym samym rozmiarze po stronie Edenu jest pomijany; każdy plik idzie pod nazwą tymczasową `.czesc_*` i dostaje właściwą nazwę dopiero po sprawdzeniu rozmiaru; do 3 ponowień; trzy archiwa wysłane 29.09 do `dyfuzja_narzedzie/filmy_zip` (`TR_0001`, `TR_0003`, `TR_0006`) dowiązano twardo (drugi wpis w katalogu wskazujący te same bajty, bez ponownej wysyłki). Na końcu sumy SHA256 (skrót kryptograficzny pliku; zgodne skróty po obu stronach potwierdzają identyczność bajtów) liczone lokalnie i sprawdzane na Edenie. Skrypt niczego nie usuwa i nie zgłasza zadań.

**Zawartość.** 3 archiwa `videos/videos_part001..003.zip` (170 filmów MP4, 5,6 GB) i 170 archiwów `annotations/<film>.zip` (42 GB): każda klatka jako JPEG 720 × 480 (część filmów 720 × 576) oraz `annotation.csv` z wielokątami trzech obiektów na każdej klatce: cystotomu (`cystotome_seg`, narzędzie), rogówki (`cornea_seg`) i źrenicy (`pupil_seg`); razem 469 118 klatek fazy kapsuloreksji.

**Pomiary** (log `F:\Studia\PhD_projekt\VIT\eden_archiwum\wysylka_cataract_lmm\wysylka.log`, sumy `SHA256SUMS`, 174 linie). Start 01:37:54, koniec wysyłania 04:22:31, koniec sprawdzania sum 04:38:53. Wysłano 170 plików (49 157 MB), 3 dowiązano, 0 ponowień. Archiwa filmów 1656-1978 MB w 991-1134 s (około 1-2 MB/s na strumień), razem około 5 MB/s (przeliczenie: 49 157 MB / 9877 s ≈ 5,0 MB/s). Wynik: `SUMY_ZGODNE`, 3 archiwa filmów i 170 archiwów klatek; kwota po wysyłce 102,5 GiB, 70 703 pliki. Wcześniejsza próba (29.09, rozgałęzienie): 3 archiwa, 1,38 GB w 322 s, około 4,3 MB/s.

### 5.12 Stable Diffusion 1.5 dostrajane na wszystkich klatkach filmów (`dyfuzja_sd_filmy`)

**Prośba i zgoda.** 01:33: „Po wrzuceniu filmów możesz opracować jak uczyć stable diffusion model z całych filmów. Tym razem trening modelu możesz pracować na tylu GPU inne dasz radę na edenie”. Plan i kod powstały przed zgodą (zatrzymanie przed zgłoszeniem), zgoda przyszła o 10:06: „tak wysyłaj i trenuj”. Katalog: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dyfuzja_sd_filmy` (`PLAN.md`, `README.md`, `kod\wspolne.py`, `kod\przygotuj_latenty.py`, `kod\trenuj_sd.py`, `eden\wyslij.sh`, `eden\przygotuj_sd.slurm`, `eden\trenuj_sd.slurm`), na Edenie `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/`.

**Pojęcia.** **Stable Diffusion** to model dyfuzyjny działający w przestrzeni ściśniętej: **VAE** (autokoder wariacyjny: koder ściskający obraz do małej reprezentacji i dekoder odtwarzający z niej obraz) zamienia obraz na **latent obrazu**, 8 razy mniejszy w każdym wymiarze i o 4 kanałach; **UNet** (860 mln parametrów) zdejmuje szum z tego latentu; dekoder VAE zamienia wynik na obraz. W zwykłym użyciu UNet dostaje też opis tekstowy przez koder CLIP; tu tekst nie gra roli. Uczenie zaczyna się od wag wyuczonych na LAION (**dostrajanie**: model już umie rysować tekstury i światło, a douczamy go na klatkach operacji). Latent obrazu nie ma nic wspólnego z latentem małego ViT.

**Dane.** Tylko 118 filmów `train` z `pool_manifest.jsonl` (według zliczenia z 29.09: 313 359 klatek). 25 filmów `val` i 27 filmów spoza manifestu nie wchodzą do uczenia: obrazy z modelu mają potem uczyć detektor, więc model nie może widzieć filmów walidacyjnych (przeciek). Pułapki danych opisane w `wspolne.py`: archiwum `<film>.zip` ma katalog `<film>/` z plikami `<film>_NNNNNN.jpg` i `annotation.csv` (kolumna `filename`, `frame_number` od 1); wielokąty to napis JSON z listą list współrzędnych w pikselach natywnych, pusty jako `''` lub `'[]'`; `cystotome_seg` bywa pusty (narzędzie poza kadrem); uszkodzony JPEG zastępuje czarna klatka.

**Model** (różnice wobec `dyfuzja_narzedzie` nazwane w planie: przestrzeń latentna zamiast pikseli, model gotowy dostrajany zamiast uczonego od zera, całe klatki zamiast wycinków 176 px, wszystkie klatki zamiast co piątej, maska jako warunek wejściowy zamiast rysowanego czwartego kanału, wiele kart zamiast jednej):
- Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, licencja CreativeML OpenRAIL-M) z dekoderem VAE `stabilityai/sd-vae-ft-mse` (MIT); skala latentu 0,18215.
- Rozdzielczość 576 × 384 (proporcje 3:2 jak 720 × 480), latent 72 × 48 × 4. Klatki 720 × 576 ściskane do 576 × 384 tak samo jak przy wejściu małego ViT.
- **Warunek: układ sceny.** Do zaszumionego latentu dokładane są 3 kanały masek w rozdzielczości latentu (narzędzie, rogówka, źrenica; wielokąt rysowany w pełnej rozdzielczości i zmniejszany uśrednianiem pola, więc każda komórka 8 × 8 pikseli ma udział pola 0-255). Pierwsza warstwa UNet przyjmuje 7 zamiast 4 kanałów; wagi dla 4 starych kanałów skopiowane, dla 3 nowych startują od zera (jak w modelach inpaintingu SD). W 20 procentach przykładów maski są zerowane, więc model umie też rysować bez warunku. Po co: przy generowaniu podaje się obrys narzędzia w wybranym miejscu i kącie, więc wiadomo, gdzie są tokeny narzędzia dla siły tożsamości małego ViT.
- Tekst: stałe osadzenie pustego napisu, liczone raz na starcie koderem CLIP, który potem jest zwalniany (plan mówi, że koder „nie jest ładowany”; kod ładuje go na chwilę).
- Harmonogram szumu SD 1.5 (DDPM, 1000 kroków, β od 0,00085 do 0,012 „scaled_linear”, przewidywanie szumu). **Offset noise** 0,05: do szumu dodaje się stałe przesunięcie całego kanału (0,05 × N(0,1) na przykład i kanał), dzięki czemu model uczy się zmieniać średnią jasność i umie rysować ciemne tło klatek.
- Podgląd: próbnik DDIM 50 kroków z wagami EMA; **CFG** (classifier-free guidance: mieszanie przewidywania z warunkiem i bez warunku, e = e0 + w·(e - e0), wzmacniające wpływ masek) jest w kodzie, ale podgląd używa w = 1, czyli bez wzmocnienia. W planie także DPM-Solver 25 kroków.

**Przygotowanie danych** (`przygotuj_latenty.py`, zadanie na kartach): JPEG z archiwum → 576 × 384 → koder VAE (bf16, średnia rozkładu × 0,18215) → `latenty/<film>.npy` (N, 4, 48, 72) float16; maski → `maski/<film>.npy` (N, 3, 48, 72) uint8; `indeks/<film>.json` (nazwy klatek, rozmiar natywny, liczba klatek z narzędziem, statystyki latentu). Około 38 KB na klatkę, około 12 GB i 354 pliki dla zbioru `train`. Filmy dzielone między procesy (jeden na kartę), klatki kawałkami po 512, zapis atomowy (najpierw `.tmp`, potem zmiana nazwy), film z gotowym indeksem pomijany (wznowienie). Wynik poza katalogiem biegu, w `~/smallvit_data/cataract_lmm/sd576_<podział>[_coN]`.

**Uczenie** (`trenuj_sd.py`, `trenuj_sd.slurm`):
- **DDP** (DistributedDataParallel): każda karta ma kopię modelu i własną część danych, a gradienty są uśredniane między kartami; uruchamiane przez `torchrun` na jednym węźle.
- **Akumulacja gradientu**: gradienty kilku kolejnych porcji sumuje się przed jedną aktualizacją wag (podczas porcji pośrednich bez synchronizacji kart, `no_sync`). Paczka globalna stała: 256 = obrazy na kartę × liczba kart × akumulacja, więc wynik nie zależy od liczby wolnych kart.
- AdamW lr 2·10⁻⁵ z rozgrzewką 500 kroków, zanik wag 0,01, obcięcie normy gradientu 1, bf16, TF32 (tryb mnożenia macierzy z krótszą mantysą, szybszy na kartach NVIDIA), EMA wag 0,9999 (ze stopniowym wzrostem współczynnika na starcie: min(0,9999, (1+k)/(10+k))) liczona na karcie 0.
- Strata: błąd średniokwadratowy przewidzianego szumu; t losowane jednostajnie z 0-999; losowe odbicie poziome latentu razem z maskami.
- Budżet: 20 000 kroków × 256 = 5,1 mln przykładów (około 16 przejść przez klatki).
- Zapisy: `ostatni.pt` co 1000 kroków (UNet, EMA, stan optymalizatora, krok; 13,75 GB, nadpisywany atomowo), migawki EMA `ema_XXXXXX` w formacie diffusers fp16 co 5000 kroków (1,7 GB), podgląd co 2000 kroków, `LOG.jsonl` co 50 kroków, `STATUS.json`, `KONFIG.json`.
- Latenty kopiowane do `/dev/shm` węzła (szybki losowy odczyt), katalogi tymczasowe (`TMPDIR`, pamięć podręczna kompilatora jąder GPU Triton i biblioteki torch) też tam, kasowane pułapką.
- Zatrzymanie: SLURM wysyła sygnał 15 min przed końcem limitu 23 h 50 min (`--signal=B:TERM@900`), skrypt tworzy plik `STOP`, Python kończy krok, zapisuje `ostatni.pt` i wychodzi; kolejne zgłoszenie z tymi samymi argumentami wznawia.

**Próby lokalne (RTX 3070, 01:48-02:18).** Przygotowanie: 2 filmy `train` co 20. klatka (575 klatek: `TR_0001` 103, `TR_0003` 472) i 2 filmy `val` co 50. (187: `TR_0002` 150, `TR_0032` 37), 10-13 klatek/s, odchylenie standardowe latentu 0,877-1,031 (SD oczekuje około 1). Pierwsza próba padła na braku pamięci: koder VAE nie zmieścił na 8 GB paczki 48 klatek 576 × 384 (około 1 GB na klatkę w bf16); domyślna paczka zmieniona na 16, lokalnie użyto 4. Rekonstrukcje VAE wierne, kontury masek na narzędziu, rogówce i źrenicy (średni udział pola masek w kontroli: narzędzie 0,8 procent, rogówka 14,6, źrenica 7,7). Uczenie małym UNet (3 mln parametrów, losowe wagi, 4 na kartę × akumulacja 2): 30 kroków, strata 1,11 → 0,72; zapis, migawki EMA, podgląd; wznowienie od kroków 30 i 40; zatrzymanie plikiem STOP z zapisem i usunięciem pliku. Jeden bieg próbny przerwał limit czasu w trakcie końcowego podglądu; czyste zakończenie sprawdzono osobno. Poprawki przed wysyłką: `float(strata)` zamienione na `float(strata.detach())`; w `trenuj_sd.slurm` pętla czekania na proces: pierwotnie `while kill -0 "$PID"; do wait "$PID" || true; done; wait "$PID"; RC=$?`, gdzie końcowe `wait` na procesie już odebranym zwróciłoby błąd, a przy `set -e` przerwało skrypt; teraz kod wyjścia jest zapamiętywany w pętli (`if wait "$PID"; then RC=0; else RC=$?; fi`), a pętla trwa dalej, gdy `wait` przerwie sygnał SIGTERM. Samego SD 1.5 z optymalizatorem nie da się sprawdzić na 8 GB (commit `2bf184b0`).

**Przygotowanie Edenu (10:13).** `eden/wyslij.sh`: kod, manifest, `diffusers==0.39.0` zainstalowane bez zależności do `deps` (54 MB; środowisko `~/venvs/rfdetr_nano_clean_1_10_1` ma torch 2.9.1 i transformers 5.17, ale nie ma diffusers), wagi SD 1.5 w fp16 (UNet, koder tekstu, tokenizer, harmonogram; 1,9 GB) i sd-vae-ft-mse (305 MB) w `modele/`. Węzeł logowania ma dostęp do internetu. Kwota 102,5 → 104,7 GiB.

**Próby na Edenie (10:14-10:29).**
- 1818490 (przygotowanie `train`, 2 filmy, wszystkie klatki, 1 × H100): `TR_0001` 2046 klatek przy 43 kl./s, `TR_0003` 9421 klatek przy 73 kl./s; 177 s; 416 MB (`sd576_train_proba2`). Prędkość ograniczają procesory dekodujące JPEG.
- 1818491 (przygotowanie `val`, 2 filmy, co 10. klatka, 1 × H100): 36 s, 20 MB (`sd576_val_co10_proba2`). Odchylenie latentu 0,876-1,031.
- 1818492 (uczenie, 2 × H100, 16 na kartę × akumulacja 8 = 256, 200 kroków, 11 467 klatek z 2 filmów): **110 obrazów/s** (2,32-2,33 s na krok), 40,04 GB pamięci na karcie 0 (z EMA), strata 0,171 (krok 10) → 0,137 (140) → 0,131 (150) → 0,119 (170), około 0,12 na końcu; zapis `ostatni.pt` 13,75 GB w około 40 s. Podgląd po 200 krokach (`wyniki/proba/podglad_000200.jpg`): realistyczne klatki oka, narzędzie w miejscu obrysu. Na A100 40 GB zmieściłoby się tylko 8 na kartę.

**Pełne zadania (zgłoszone 10:30, commit `056127e9`).** Plan zakładał 8 kart, ale limit konta to 5 × H100, a na węzłach A100 brakowało wolnych procesorów; użyto wszystkich pięciu H100 konta:
- 1818494: przygotowanie wszystkich klatek 118 filmów `train`, 4 × H100, 10:29:55-10:51:07 (21 min 12 s), 68-73 kl./s na proces, wynik `sd576_train` 12 GB;
- 1818495: co 10. klatka 25 filmów `val`, 1 × H100, 3 min 9 s, 186 MB, koniec 10:33:04 (`sd576_val_co10`);
- 1818496: uczenie, 4 × H100, 32 na kartę × akumulacja 2 = 256, 20 000 kroków, z zależnością od obu, start około 10:51. Szacunek z README: około 220 obrazów/s, 5,1 mln przykładów w około 6,5 h.

Zmierzone prędkości razem: wysyłka filmów około 5 MB/s (3 strumienie); przygotowanie latentów 73 kl./s na H100 (próba) i 68-73 kl./s na każdą z 4 kart; uczenie SD 1.5: 1 karta bez osobnego pomiaru (podsumowanie rozmowy podaje 55 obrazów/s na H100, co jest połową wyniku z 2 kart), 2 × H100: 110 obrazów/s, 4 × H100: 208,8-209,6 obrazów/s; lokalnie mały UNet na RTX 3070: 3-7 obrazów/s (tylko test kodu).

**Pytania użytkownika w trakcie.** 10:34, z wklejonym wynikiem `squeue` (1818496 czeka „Dependency”, 1818494 działa): „które z tych zadań to juz własciwy trening?” - właściwe uczenie to 1818496 `sd_trening`, czeka na 1818494 `sd_latenty` (wtedy 17 ze 118 filmów gotowych). 10:35: „a dlaczego on sie uczy z latentu?” - to inny latent niż w małym ViT: ściśnięty obraz, klatka z 663 tys. liczb (576 × 384 × 3) staje się 14 tys. (72 × 48 × 4), 48 razy mniej; tak zbudowane jest Stable Diffusion, gotowe wagi umieją rysować tylko w tej przestrzeni, a klatki ściska się raz przed uczeniem; dekoder odtwarza klatki wiernie razem z cienkim narzędziem, ale najdrobniejsze szczegóły, jak sam czubek narzędzia, mogą wyjść trochę rozmyte. 11:53: „On uczy się z filmów wgranych na eden?” - tak: przygotowanie przeczytało archiwa ze wszystkimi klatkami i obrysami 118 filmów treningowych i skończyło się o 10:51; pliki MP4 nie są potrzebne, bo archiwa mają te same klatki jako obrazy i do tego obrysy.

**Śledzenie i usterki.** Pierwszy monitor (skrypt odpytujący Eden co kilka minut) przerwał się o 10:36 z kodem 4 [przyczyny nie ustalono w źródłach]. Monitor uruchomiony o 11:54 podał „krok 1”, bo wyrażenie `krok.: [0-9]*` dopasowało na końcu ostatniej linii logu fragment `"s_na_krok": 1.224`; poprawione na `{.krok.: [0-9]*` (początek wpisu), od 11:55 odczyt poprawny. Monitory wygasają po 30 min; od 12:25 działała pętla sprawdzająca stan co 5 min i zgłaszająca koniec albo błąd w logu.

**Przebieg do 13:05** (log `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/logi/sd_trening_1818496.out`, wyniki w `/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening`):

| czas | krok | strata | obrazy/s | s na krok | pamięć na kartę |
|---|---|---|---|---|---|
| 11:54 | 2800 | 0,118 | 208,9 | 1,226 | 57,1 GB |
| 11:54 | 2850 | 0,123 | 209,1 | 1,224 | 57,1 GB |
| 11:54 | 2900 | 0,124 | 208,8 | 1,226 | 57,1 GB |
| 11:55 | 2950 | 0,1225 | 209,1 | 1,225 | 57,1 GB |
| około 12:17 | 4000 | 0,114 | 208,9 | 1,225 | 57,1 GB |
| 12:17-12:20 | 4050 | 0,114 | 122,7 | 2,086 (w tym zapis `ostatni.pt` i podglądu po kroku 4000) | 57,1 GB |
| około 12:20 | 4100 | 0,1076 | 209,6 | 1,222 | 57,1 GB |
| około 12:45 | 5050 | 0,107 | - | - | - |
| 13:04 | 6200 | 0,110 | 208,9 | 1,226 | 57,1 GB |

Start straty około 0,17 (próba: 0,17 → 0,12); licznik epok w logu: 2 przy kroku 2800, 3 przy kroku 4000. Zero błędów w logu. Zapisane: `podglad_002000.jpg` (11:35), `podglad_004000.jpg` (12:17), `podglad_006000.jpg`, migawka `ema_005000`, `ostatni.pt` (13,75 GB). **Podgląd po 4000 krokach** to siatka 32 obrazów: według kodu `zapisz_podglad` 16 par, w każdej u góry prawdziwa klatka walidacyjna przepuszczona przez koder i dekoder VAE, a pod nią obraz wygenerowany od szumu z masek tej klatki (kontury: czerwony narzędzie, zielony rogówka, niebieski źrenica); w rozmowie opisano go jako „32 klatki wygenerowane przez model”. Większość obrazów to realistyczne oko w trakcie operacji, narzędzie leży tam, gdzie wskazuje obrys; kilka klatek w dolnym rzędzie jest jeszcze zaszumionych. Maski pochodzą z filmów `val`, więc model rysuje układy scen, których nie widział. Kwota po przygotowaniu latentów: 132,4 GiB (zapis sesji podaje „około 139 z 200 GiB”); każda migawka EMA doda 1,7 GB, a artefakty próby (`wyniki/proba/ostatni.pt` 13,75 GB i migawki EMA próby) czekają na zgodę na usunięcie.

**Ocena zaplanowana, niewykonana** (`PLAN.md`; skrypt `kod/ocena_sd.py` jeszcze nie istnieje): podglądy i próbki z masek filmów `val`; FD-DINOv2 (odległość Frécheta między rozkładami cech DINOv2 ViT-S/14 dla 5000 próbek i 5000 klatek `val`, z odniesieniem `train` wobec `val`; im mniejsza, tym bardziej próbki przypominają prawdziwe klatki); kopiowanie (odległość każdej próbki do najbliższej klatki uczącej w cechach DINOv2; próbka bliżej niż sąsiednie klatki tego samego filmu oznacza kopię); zgodność z maską (czy mały ViT albo prosty segmentator widzi narzędzie tam, gdzie była maska). Dalej, za osobną zgodą: prowadzenie siłą tożsamości małego ViT (FreeDoM, gradient przez dekoder VAE), ewentualnie warstwy czasowe.

**Czego to nie dowodzi.** Dobry wygląd próbek ani niska FD-DINOv2 nie znaczą, że obrazy pomogą detektorowi; użycie generatora w pętli uczenia detektora to osobne ramię i osobna kampania.

**Stan na 2026-09-30 13:05: uczenie trwa.** Zadanie 1818496 (4 × H100, partycja `hopper`, start około 10:51) było przy kroku 6200 z 20 000, strata 0,110, 208,9 obrazów/s, 1,226 s na krok, 57,1 GB pamięci na kartę, bez błędów w logu; szacowany koniec około 17:45. Wyniki końcowe (podglądy, krzywa straty, zgodność z maskami, ocena generatora) nie są znane w chwili zamknięcia tego etapu.

### 5.13 Równoległa gałąź: sesja-fork „kierunki tożsamości”

Sesja-fork powstała o 14:30:35 (29.09) poleceniem `/fork`, w chwili gdy w tle liczył się skrypt `kierunki_tozsamosci.py` (zadanie powłoki `b37opvd2w`): odpowiedź na pytanie użytkownika, ile kierunków 768-wymiarowej przestrzeni cząstek (tokenów) odróżnia token z narzędziem od tego samego tokenu na zdjęciu bez narzędzia. Przez cały etap fork był bezczynny; jedyne pytanie użytkownika padło o 23:28: „co było liczone w tym forku?”. Fork odczytał wyjście swojego zadania: policzonych 5 z 10 sieci (ziarna 47-51), potem proces zniknął (limit Bash w tle albo koniec poprzedniego procesu Claude Code). Sprawdził katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_tla\wyniki`: skrypt uruchomiono ponownie jako osobny proces (`kierunki_tozsamosci.log`, 14:45) i policzono wszystkie 10 sieci, a potem zapisano kierunki cząstkowe (`zapisz_kierunki_czastek.log`, 14:51); wyniki są w commitach `238851f7` i `8d50ae84` (etap 4). Fork nazywa tę nitkę „sesją główną”; według zapisu rozmowy zrobiło to rozgałęzienie opisane w 5.1. Liczby przekazane użytkownikowi (średnio z 10 sieci, 1568 cząstek narzędzia na sieć): 1,3 kierunku z AUC ≥ 0,9 (zakres 1-2) i 6,5 z AUC ≥ 0,75 na wycinkach spoza wyznaczania (AUC: pole pod krzywą ROC, 0,5 to zgadywanie, 1 to idealne rozdzielenie; kolejne kierunki LDA z usuwaniem poprzednich, INLP); połowa energii różnic „z narzędziem minus bez” w 2,5 kierunku, 90 procent w 38,4 kierunku, pierwszy kierunek niesie 38 procent; pierwszy kierunek cząstkowy ma cosinus 0,40 z wektorem `nowy` i rozdziela z AUC 0,999 na cząstkach, na których go wyznaczono, oraz około 0,95 poza nimi (wektor `nowy` 0,961). Częściowe wyjście forka zgadza się liczbowo z pełnym biegiem dla ziaren 47-51. Fork niczego nie zmienił poza własnym zapisem sesji (commit `beed3307`, 12:44, 30.09) i nie ma własnych zadań.

### 5.14 Podsumowanie etapu 5

1. **Horyzont:** Koopman i Langevin, każdy osobno, biją „brak zmiany” aż do 1000 kroków, ale tylko o 10-25 procent błędu kwadratowego; Langevin rekurencyjny najlepszy na 100-250 krokach (18-25), przy 1000 krokach 2-9; dopasowanie wprost na horyzont trzyma 15-25; przedłużanie trendu daje od -382 do -417 przy 1000 krokach.
2. **Równanie ruchu latentu (wyjaśnienie):** Δz(t+1) = 0,83·Δz(t) + 0,98·S(t) + ε(t); na 10 ziarnach Edenu (1818331) mediana 85,0 (80,5-89,2), z pełnym T1 85,4 (81,3-89,4), każde ziarno powyżej 80; placebo różni się od wariantu bez sterowania (mediana 59,1) najwyżej o 0,01 punktu; z pędem optymalizatora 93,9-95,8.
3. **Fizyka ruchu:** latent to cząstka z bezwładnością (83 procent prędkości przenoszone na kolejny krok przez pęd AdamW) popychana siłą paczki ze współczynnikiem około 1; ruch idzie przez wagi całej sieci (97,8 procent), nie przez BatchNorm (0,7) ani sam ostatni blok (7,5).
4. **Reguła i uczciwość:** 85 procent korzysta z siły bieżącej paczki, czyli klatek kroku, który się właśnie dzieje; po regule użytkownika to tryb wyjaśnienia (dozwolony), a nie prognoza. Prognoza bez przyszłych klatek: 64-67 procent (0,94·S_pęd), 56-57 procent (0,76·Δz), 59 procent (jądra T1).
5. **Macierze jak w TS:** uczone człony D + JP + W + N (11 mln parametrów) dają około 40 procent, bo wkuwają okres uczenia; wymuszenie z paczek t-1..t-16 równe placebo; pełne macierze 768 × 768 52-54 wobec 57 skalara; wektory tożsamości dwóch ziaren mają cosinus 0,085, więc równanie trzeba dopasować w każdej sieci osobno. W kodzie TS znaleziono błąd `SzumKolorowy` (reszty od t-2 zamiast t-1).
6. **Literatura:** 10 prac przeczytanych w całości; metoda Farida odpada (nie przesuwa narzędzia), ReNO pasuje do FastGAN-a, FreeDoM do własnego modelu dyfuzyjnego; każdy wariant SD wymaga co najmniej 15 GB karty.
7. **Model pikselowy (rozgałęzienie):** UNet 22,3 mln parametrów, 56 005 klatek z obrysem wycinane w trakcie uczenia, 0,385 s na krok na H100, 100 000 kroków ukończone według odczytu z 09:49; przesycone kolory to wina pierwszego kroku DDIM przy t = 999 (√ᾱ ≈ 5·10⁻⁵, nasycenie 0,568 wobec 0,163 w danych), start od t = 979 daje 0,180; poprawka niewprowadzona.
8. **Eden:** limit konta 5 × H100, 17 × A100, 20 kart dla wszystkich użytkowników; sprzątanie za zgodą 117 → 56,7 GiB i 168 501 → 70 504 plików (utracone surowe zrzuty latentu czterech pętli); wysyłka całego `3_Object_Tracking` (173 archiwa, z czego 170 wysłanych, razem 49 157 MB, i 3 dowiązane) w 2 h 45 min przy około 5 MB/s, sumy SHA256 zgodne.
9. **Stable Diffusion 1.5 na całych filmach:** latenty wszystkich klatek 118 filmów `train` (12 GB w 21 min na 4 × H100), UNet z 7 kanałami wejścia (maski narzędzia, rogówki i źrenicy), paczka 256, 20 000 kroków; 110 obrazów/s na 2 × H100, 209 na 4 × H100; o 13:04 krok 6200, strata 0,110, podglądy realistyczne; koniec około 17:45.

### 5.15 Otwarte pytania i następne kroki

- **Uczenie 1818496 trwa.** Po zakończeniu: pobrać podglądy, `LOG.jsonl`, `STATUS.json` i w razie potrzeby `ema_020000`, obejrzeć obrazy, dopisać wyniki do `dyfuzja_sd_filmy\README.md`, sprawdzić czystość `/dev/shm` na `hopper`; przy przerwaniu przez limit czasu zgłosić ponownie z tymi samymi argumentami (wznowienie).
- **Brak miary jakości generatora:** nie ma jeszcze `ocena_sd.py` (FD-DINOv2, najbliżsi sąsiedzi, zgodność z maską). Nie wiadomo, czy model nie kopiuje klatek i czy narzędzie naprawdę pojawia się w masce (za zgodą).
- **Prowadzenie małym ViT** (FreeDoM przez dekoder VAE) w stronę zadanej siły tożsamości: za osobną zgodą. Otwarte, czy lepszy generator zmieni wynik wersji 3 pętli, gdzie zysk dawało samo wyszukiwanie podobnych klatek.
- **Prognoza wielokrokowa:** równanie działa na jeden krok; siła przyszłych paczek zależy od przyszłych wag. Sprawdzenie prognozy na 10 i 100 kroków z wygasającą pamięcią wymuszenia (T4) na zapisanych wagach ziarna 48 czeka na zgodę. Otwarte też: czy czysta wersja siły pędu (mianownik z samego v(t-1)) zmienia wynik 64-67.
- **Model pikselowy:** poprawka `siatka_krokow` (start DDIM od t = 979) w `dyfuzja_narzedzie` niewprowadzona, bo folder prowadzi inna sesja; ocena jego jakości i los fragmentu 1818356 nieustalone w źródłach.
- **Kwota Edenu:** około 132 GiB (według zapisu sesji około 139) z 200 po latentach; artefakty próby SD (13,75 GB `ostatni.pt` i migawki EMA) czekają na zgodę na usunięcie; każda migawka EMA pełnego uczenia to 1,7 GB.
- **Kod TS:** błąd przesunięcia w `SzumKolorowy` (`tft_langevin_latent_56`, biegi L1-L3) nazwany, ale niepoprawiony w projekcie grantowym; wpływ na tamte wyniki niezmierzony.
- **Zasięg wniosków o równaniu:** tylko ramię bez dokładania klatek, kroki 1000-3000, jedna architektura i jeden zbiór; zachowanie przy dokładaniu klatek (jak w pętli) nie jest sprawdzone.


## E. Synteza: co wiemy po dziesięciu dniach

### E.1 Odpowiedź na pytanie główne

Pytanie brzmiało, czy prognoza ruchu latentu tożsamości pozwala dobrać klatki, na których detektor uczy się szybciej albo lepiej
niż na losowych. Rozstrzygają porównania, w których prognoza jest jedyną różnicą między ramionami. We wszystkich takich
porównaniach wynik mieścił się w szumie ziaren albo był ujemny:

| etap | porównanie | wynik |
|---|---|---|
| 1 | reszta Langevina (K1) wobec tych samych reguł bez niej (B3, B4) i placebo | +0,0027, +0,0018, +0,0006 AP przy progu +0,004 |
| 2 | 250 klatek po kwadratach przestrzeni tożsamości wobec 250 losowych | 0,6296 wobec 0,6319 |
| 2 | 1500 klatek ze zmianami w stałych kierunkach Koopmana wobec 1500 losowych | 0,6696 wobec 0,6820 (przegrana na 3 z 3 ziaren) |
| 3 | 134 klatki dobrane pod obraz przewidzianego latentu wobec 134 losowych | +0,0007 wobec +0,0009 mAP@0,5 |
| 3 | kampania 5 epok: dobrane, losowe, trudne wobec braku dodania | -0,0005, +0,0010, +0,0016 |
| 3 | pętla 8 obrotów na Nano: koopman, losowe, bez dokładania | 0,9055, 0,9045, 0,9072 |
| 3 | pętla na RF-DETR Large: koopman wobec losowych | -0,0030 mAP@0,5, ujemne w 15 z 18 par |
| 4 | wersja 3 na małym ViT: prognoza siły wobec tych samych kotwic bez prognozy (`current`) | 0,4432 wobec 0,4428 (5 z 10 ziaren) |
| 4 | iteracja 2: metoda wobec `current` | 13 z 20 ziaren (pole pod krzywą), 8 z 20 (koniec uczenia) |

Tam, gdzie metoda wygrywała z losowaniem (mały ViT: wersja 1 w 9 z 11 ziaren, wersja 2 w 8 z 10, iteracja 2 wersji 3 w 15 z
20 przy p = 0,006), ta sama przewaga była w ramieniu bez prognozy. Zysk daje więc wyszukiwanie prawdziwych klatek podobnych do
wycinków narzędzia, czyli skupienie uczenia na narzędziu, a nie przewidywanie, dokąd pójdzie latent. Na dostrajanym RF-DETR
nie działało nic, co różniło się od losowania: ani prognoza, ani dobór trudnych klatek, ani GLISTER równoległej sesji
(losowe 0,7131 najlepsze).

### E.2 Dlaczego prognoza nie pomagała: siedem zmierzonych mechanizmów

1. **Przewidziane przesunięcie jest małe wobec różnic między klatkami.** W Nano od kroku 245 prognoza na 100 aktualizacji
   przesuwa latent o 0,069 rozrzutu między sondami (etap 3). Obraz z generatora zmienia się wtedy o 0,1-0,9 procent jasności.
2. **Wybór najbliższych klatek jest stabilny.** Twierdzenie z `TEORIA_LANGEVIN.md` (etap 3) mówi, że przy małym przesunięciu
   zapytania lista top-k się nie zmienia; w pomiarze 54 z 60 sąsiadów było wspólnych dla obrazu „teraz” i „po prognozie”.
3. **Dostrajany kręgosłup prawie się nie rusza.** W RF-DETR Large wagi kręgosłupa DINOv2 zmieniły się o 0,026 procent w 6
   epokach (etap 4), a tożsamość narzędzia w Nano była gotowa już w kroku 5 dzięki wagom COCO (AUC 0,995, etap 3). Latent
   sond to w dużej części cechy odziedziczone, a nie wyuczone na tych danych.
4. **Latent kręgosłupa nie wiąże się z postępem detekcji.** Skalar tożsamości Large nie korelował z mAP (ρ = 0,07); dekoder
   korelował w czasie (ρ = 0,86), ale nie rozróżniał ramion (etap 4). Nawet trafna prognoza takiego latentu nie mówi, które
   klatki poprawią detektor.
5. **Generator nie umie zmienić tylko jednej rzeczy.** FastGAN dochodził do pełnego przewidzianego latentu najwyżej w 36
   procentach drogi, a trafiając w przewidzianą siłę tożsamości, zmieniał resztę latentu około 40 razy mocniej (etap 4).
   Sterowanie wzdłuż jednego wektora działa (12 z 12), ale wtedy cel to jedna liczba, która nie mówi, gdzie ma być narzędzie.
6. **Dobór celowany traci różnorodność.** Klatki „trudne”, „brakujące” albo skupione wokół kilku kotwic przegrywały z losowymi
   (etap 2: kierunki stałe; etap 3: klatki trudne pomagały daleko od kotwic, a szkodziły przy nich; etap 4: w Large wybór skupiał
   klatki, 138-140 wobec 62-83 sąsiadów). Losowanie pokrywa całą pulę, a celowanie nie.
7. **Efekty są rzędu szumu ziaren.** Różnice 0,001-0,003 mAP między ramionami wymagają kilkunastu ziaren; większość kampanii na
   RF-DETR miała 3 ziarna, więc nawet prawdziwy mały efekt byłby niewidoczny.

### E.3 Co działa i czego się nauczyliśmy o latencie

- **Latent krystalizuje wcześnie.** Położenie i stan narzędzia pojawiają się w latencie w pierwszych trzech epokach (R2 0,68-0,86),
  potem latent dryfuje (CKA około 0,86 między kolejnymi epokami), a po szczycie jakości rozlewa się na więcej kierunków
  (etap 1). Dołek straty walidacyjnej wypada w epokach 6-7, a sygnał bez etykiet (szczyt trwałości modów) wskazuje koniec
  uczenia z kosztem 0,004 AP zamiast 0,030 przy uczeniu do 30. epoki.
- **Ruch latentu w uczeniu ma bezwładność.** Stan z pamięcią [z, v] bije sam stan (etap 3), a w małym ViT latent przenosi 83
  procent prędkości na kolejny krok (etap 5). Źródłem jest pęd optymalizatora AdamW, który pamięta dawne paczki.
- **Latent jest popychany przez dane z siłą bliską jedności.** Siła bieżącej paczki S(t) wchodzi do równania ze współczynnikiem
  0,98 (0,977-0,990 na 10 ziarnach), a placebo z niewłaściwej paczki nic nie wnosi (etap 5). Równanie
  Δz(t+1) = 0,83·Δz(t) + 0,98·S(t) wyjaśnia 85 procent ruchu; bez przyszłych klatek prognoza osiąga 57-67 procent.
- **Proste równanie wygrywa z dużym.** Uczone macierze w stylu TS (11 mln parametrów) dają około 40 procent, a pełne macierze
  768 × 768 52-54 procent, bo dopasowują szczegóły okna (etap 5). Tak jak w TS: wymuszenie danymi decyduje, jądro pamięci
  dodaje mało.
- **Tożsamość narzędzia trzeba mierzyć na parach.** Kolejne miary tożsamości mierzyły częściowo scenę albo ułożenie narzędzia:
  kod niezmienników niósł wydłużenie ramki (etap 1), „kierunki tożsamości” Koopmana okazały się kątem ułożenia (etap 2), stary
  wektor LDA oddzielał narzędzie od sceny tylko na d' 2,0, a sieć nieuczona przechodziła testy na wersjach sond (etap 4).
  Dopiero pary „to samo miejsce z narzędziem i bez” dały wektor `nowy` (d' 16,6, AUC 0,997). Tożsamość to wąska wiązka
  kierunków (1-2 kierunki z AUC co najmniej 0,9, 5-8 z AUC co najmniej 0,75), a nie jedna strzałka; wektory różnych sieci nie
  są porównywalne (cosinus 0,085).
- **Siła tożsamości jest przewidywalna, ale słabo.** Koopman i Langevin przewidują ją lepiej niż „brak zmiany” o 6-10 procent
  (etap 4) i do 1000 kroków naprzód o 10-25 procent błędu (etap 5). Langevin krok po kroku najlepiej na 100-250 kroków.
- **Mały ViT od zera to dobre pole doświadczalne.** Uczy się (mAP@0,5 0,36-0,42), każde dokładanie klatek wyraźnie pomaga
  (losowe nad brakiem dokładania w 11 z 11 i 10 z 10 ziaren), a jego latent naprawdę się zmienia w trakcie uczenia.
- **Wyszukiwanie klatek podobnych do narzędzia przyspiesza uczenie małego ViT** (iteracja 2: 15 z 20 ziaren, p = 0,006),
  choć na końcu uczenia ramiona się zrównują.

### E.4 Jak zmieniała się prognoza

| etap | co przewidywano | czym | wynik wobec „braku zmiany” |
|---|---|---|---|
| 1 | współrzędne latentu wzdłuż filmu i epok | operator Koopmana, deep Koopman | nie bije na nowych filmach (deep Koopman od -45 do -61 procent) |
| 2 | następna klatka w filmie | zaskoczenie Koopmana, reszta Langevina | jako wskaźnik chybień AUC 0,57-0,58 |
| 3 | latent sond w krokach uczenia | Koopman ze stanem [z, v] | błąd 0,0037 wobec 0,0103 na jeden krok |
| 3 | skalar tożsamości | wspólny operator na 4 ostatnich przyrostach | 0,272 błędu „braku zmiany” na 1 krok, 0,171 na 100 aktualizacji |
| 3 | latent Large w obrotach pętli | Koopman | gorszy od „braku zmiany” w 12 z 12 (diagnoza z etapu 4) |
| 4 | wektor i rdzeń tożsamości małego ViT | Koopman, Langevin | wektor 0 z 6 obrotów; siła tożsamości lepsza o 6-10 procent |
| 5 | siła tożsamości do 1000 kroków | Koopman, Langevin osobno | lepsze o 10-25 procent błędu |
| 5 | cały latent sond małego ViT na 1 krok | równanie z bezwładnością i siłą paczki | wyjaśnienie 85 procent, prognoza bez przyszłych klatek 57-67 procent |

### E.5 Wnioski metodologiczne

- **Ramię różniące się tylko prognozą jest konieczne.** Bez ramienia `current` wersje 1 i 2 na małym ViT wyglądały na sukces
  prognozy; dopiero ono pokazało, że zysk daje wyszukiwanie.
- **Placebo rozstrzyga o sterowaniu.** Siła z paczki sprzed 250 kroków spadała do poziomu bez sterowania na każdym ziarnie, co
  wyklucza przypadkowe dopasowanie.
- **Wyjaśnienie to nie prognoza.** 85 procent korzysta z klatek kroku, który właśnie się dzieje; uczciwa prognoza (stany do t,
  paczki do t-1, pęd) daje 57-67 procent. Obie liczby trzeba podawać razem.
- **Kryteria przed wynikiem i liczba ziaren.** Kryteria zapisane przed uruchomieniem kilka razy zmieniły wniosek (np. wersja
  3: kryteria prognozy spełnione, kryterium roli prognozy nie). W etapie 1 miarą główną była ostatnia epoka, a werdykty
  wydano na najlepszej; tego nie wolno mieszać.
- **Zbiór testowy został użyty do diagnozy** (etap 2). Dalsze decyzje powinny zapadać na walidacji, a test zostać na jedną
  ocenę końcową.
- **Błędy w kodzie wychodziły dopiero przy rozbiorze wyniku:** k-centrum kosinusowe na nieznormalizowanych danych (K4, etap
  1), dwuwymiarowa siatka z duplikatu pary sprzężonej (K6, etap 2), przesunięcie reszt o jeden krok w `SzumKolorowy` (kod TS,
  etap 5), pierwszy krok DDIM przy t = 999 (etap 5), klatki 720 × 576 zamiast 720 × 480, ścieżki Windows na Linuksie i brak
  `timm` na Edenie (etapy 4-5).

### E.6 Reguły pracy ustalone w tej linii

- Kampanie GPU i nowe ramiona najpierw opisać i czekać na zgodę; zadania w kolejce wstrzymać, nie kasować.
- Nie podmieniać metody użytkownika bez nazwania każdej różnicy przed zgłoszeniem (28.09, po wersji 2, gdy asystent odszedł
  od metody bez nazwania różnic).
- Prognoza tylko ze stanów do t, paczek do t-1 i pędu optymalizatora; wyjaśnianie latentu klatkami, które go w tej chwili
  stworzyły, jest dozwolone (29.09 i 30.09).
- Na Edenie: sprzątanie pułapką, nic nie zostawiać na węzłach, po treningu wyczyścić hopper (23.09), sprawdzać kwotę
  `lfs quota -p 1299 /mnt/evafs` przed zgłoszeniem, usuwać tylko za zgodą na konkretne katalogi.
- Raport po każdym treningu bez pytania; każdy nowy pomiar to osobny folder w `experiments` z planem i wynikami na miejscu.

## F. Stan na koniec opisu i następne kroki

### F.1 Uczenie Stable Diffusion

Stan na 30.09, 13:44: zadanie 1818496 (4 × H100, partycja `hopper`) przy kroku 8100 z 20 000, strata 0,102, bez błędów w
logu, szacowany koniec około 17:45. Wynik końcowy: [dopisywany po zakończeniu uczenia].

### F.2 Co czeka na zgodę użytkownika

- Ocena generatora Stable Diffusion: skrypt `kod/ocena_sd.py` (FD-DINOv2, sprawdzenie kopiowania klatek, zgodność narzędzia z
  maską), potem prowadzenie siłą tożsamości małego ViT (FreeDoM przez dekoder VAE).
- Prognoza wielokrokowa równania ruchu (T4) na zapisanych wagach ziarna 48 w `maly_vit_rownanie_macierze`.
- Usunięcie artefaktów próby Stable Diffusion na Edenie (`wyniki/proba/ostatni.pt`, 13,75 GB, i migawki EMA próby).
- Poprawka `siatka_krokow` (start DDIM od t = 979) w `dyfuzja_narzedzie`, prowadzonym przez inną sesję.
- Poprawka przesunięcia w `SzumKolorowy` w kodzie TS (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56`) i
  pomiar jej wpływu na tamte wyniki.

### F.3 Najważniejsze otwarte pytania

1. **Czy prognoza w ogóle może zmienić wybór klatek?** Potrzebne jest ramię różniące się wyłącznie prognozą (np. prognoza z
   odwróconym znakiem albo losowe kotwice) oraz ablacja „wyszukaj po samym obrazie kotwicy, bez Koopmana i generatora”.
2. **Czy tożsamość wyznaczona na sondach jest tym, czego sieć używa?** Brakuje próby przyczynowej: usunięcia kierunku
   tożsamości z latentu i pomiaru detekcji.
3. **Czy równanie ruchu działa na wiele kroków i przy dokładaniu klatek?** Dziś działa na jeden krok, w ramieniu bez
   dokładania, na krokach 1000-3000 jednej architektury.
4. **Czy Stable Diffusion rysuje narzędzie tam, gdzie jest maska, i nie kopiuje klatek?** Rozstrzygnie ocena z F.2; dopiero
   potem ma sens prowadzenie małym ViT i użycie generatora w pętli.
5. **Czy wynik z małego ViT przenosi się na dostrajany RF-DETR**, na którym zaczęła się cała linia prac?
6. **Dlaczego przewaga nad losowaniem znika pod koniec uczenia** (iteracja 2: 17 z 20 ziaren w kroku 4500, 11 z 20 w kroku
   8000): wyczerpanie puli, mniej klatek w ramieniu metody, czy nasycenie?
7. **Czy nowy wektor tożsamości albo kierunki cząstkowe w pętli zmieniłyby wynik?** Nie sprawdzono.

## G. Źródła i sprzeczności w źródłach

### G.1 Źródła

Opisy sesji w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\`:

- `Session_2026-09-14_205922_RFDETR_Nano_Eden`, `Session_2026-09-15_041500_RFDETR_Static36_Eden` (tło)
- `Session_2026-09-22_131716_Koopman_Langevin_Dobor_Klatek_Etap1`
- `Session_2026-09-22_150300_Kod_Tozsamosci_Widmo_Slad`
- `Session_2026-09-22_160016_Koopman_Czesc_Rzeczywista_Domkniecie`
- `Session_2026-09-22_160617_koopman_tozsamosc_narzedzia`
- `Session_2026-09-22_210016_krystalizacja_latentu_i_telemetria_treningu`
- `Session_2026-09-23_155013_Tozsamosc_Czastki_Kod_Bez_Uruchomienia`
- `Session_2026-09-25_203826_Dobor_Klatek_Kontynuacja_134`
- `Session_2026-09-28_011148_Petla_Large_Koopman_i_Domykanie_Tozsamosci`
- `Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek`
- `Session_2026-09-30_124311_Fork_Kierunki_Tozsamosci_Sprawdzenie_Stanu`
- `Session_2026-09-30_124320_Rownanie_Latentu_Eden_i_Stable_Diffusion_Na_Filmach`

Dokumenty: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\Analiza_ponownej_publikacji\36`-`46`,
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md`,
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-28.md`,
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-30.md`.
Foldery eksperymentów: sekcja D. Kod TS do porównania: `F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56`.
Zapis rozmowy (od 23.09): `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\7dec512f-abe0-4cf3-8465-062719d28449.jsonl`.

### G.2 Sprzeczności i niejasności wykryte przy pisaniu

Źródła są w większości zgodne, ale przy składaniu opisu wyszły poniższe rozbieżności. W opisach etapów podano obie wersje.

**Etap 1.**
- K4 minus B3: +0,0068 przy niezależnej ocenie, +0,0072 według krzywej z treningu; K4 minus B2: 0,0019 i 0,0023.
- K4 według dokumentu 39 ma 16 współrzędnych, według raportu 10; faktycznie było to k-centrum kosinusowe na nieznormalizowanych
  danych.
- Przed wynikiem miarą główną było AP ostatniej epoki, a werdykty wydano na najlepszej; bootstrapu i korekty Holma dla AP brak;
  w ostatniej epoce B0 nie jest gorsze od reszty.
- Sufit prędkości z torów GT: 0,39 w tekście, 0,35 w JSON. Według sesji zawodzi tylko blok `global_pool`, JSON pokazuje, że raz
  zawiódł też `box_pool`.
- Sesja podaje, że usunięto `--nodelist=hopper`, w commicie linia została. Plan wymienia zadania 1809149-1809154, uruchomiono
  1809161-1809166.
- Pula 10 637 klatek to suma wyborów Static36 (ustalone z `data_manifest.json`, dokumenty tego nie mówią). Dwie równoległe
  sesje zdublowały sekcje 4m i 4n w dokumencie 38.

**Etap 2.**
- Dokument 43, `pobudzenie_tozsamosci.py` i rozmowa do 12:09 mówią o detektorze „1500 klatek”, faktycznie użyto `ut_b0500`;
  nazwa „b1500” oznacza w różnych testach trzy różne detektory.
- Są dwa dokumenty nr 44: „wzór domknięcia” i „dobór klatek przez przewidywaną poprawę” (formalizacja Codex).
- `koopman_kierunki_stale`: `PLAN.md` podaje 3128 klatek i 344 chybienia, raport 8600 i 880.
- Sonda: rozmowa mówi o wzroście pewności „około 40. kroku z 5 do 80 procent”, mediany galerii to 0,18 w kroku 45 i 0,80 w
  kroku 95.
- Miara 0,86 jest nazwana CKA (README), korelacją kanoniczną (dokument 44), a dokument 43 podaje 0,84-0,98.
- Pula ma 10 637 klatek, nagranie 9683; podział ma 117 grup uczących, pula 118 filmów. Filmy nazwane „niewidzianymi” przy
  wektorze to filmy puli uczącej.
- Hopper: o 17:30 usunięto `/tmp/pytest-of-bpiotrowski`, wiadomość z 21:34 mówi „nie było czego usuwać”. Rozmiar
  `tracking_nano` poprawiono z 3,7 na 0,7 GB.
- Część liczb jest tylko w rozmowie, bez pliku raportu (AUC na segmentacji 0,96 i 0,969, R2 rdzenia dekodera, „62 procent
  zmienności”, 2824/562). Wyników zadania `kl_epoki` nie przeanalizowano.

**Etap 3.**
- „Wylicz to poprawnie” dotyczyło `koopman_z_pamiecia`, a „27 procent” to skalar tożsamości z `koopman_tozsamosc_kierunki`, nie
  `koopman_gesta_historia`.
- Operator L4 na 1 krok: 0,272 przy teście 245-365, 0,407 przy teście 245-795.
- Obrót kierunku LDA: „cosinus około 0,75 i tam zostaje” wobec 0,73 w kroku 365 i 0,64 w kroku 800.
- Pętla Nano: plan zakładał 103 aktualizacje na obrót i `langevin.py`, było 165 aktualizacji i wspólny operator na 8 przyrostach.
- Krok Levenberga-Marquardta: opis mówi o podwajaniu λ, kod mnoży λ przez 4.
- Domknięcie: `PLAN.md` podaje R 0,79 i przesunięcia +3,5 oraz +3,9, `WYNIKI` i zapis sesji R około 0,78 oraz +3,3 i +3,7.
- Pokrycie top-45 w pętli Large: „około 80 procent” dotyczy obrotów 1-2, średnio 85 procent, w obrotach 4-6 około 90.
- Rozmowa podaje „około 2100 par”, `WYNIKI` 2256. Numery zadań rozgrzewki (1817340-1817342) są tylko w zapisie rozmowy.

**Etap 4.**
- Pierwsza wiadomość etapu 28.09 to 13:52 czasu lokalnego.
- Kryterium wersji 1: plan uznaje je za spełnione „w proporcji” 9 z 11, `ANALIZA.json` ma `kryterium_spelnione: false`.
- „39 procent par”: plan FastGAN-a przypisuje to wersji 2, plan wersji 1 kampanii 1817935.
- Kalibracja Z0: plan 0,39-0,42, streszczenie 0,40-0,42, plik 0,354-0,423. Plan zakładał 4-6 mln parametrów, zbudowano 2,93 mln.
- Węzły wersji 1: plan podaje hopper dla wszystkich ziaren, logi pokazują 2 ziarna na dgx-4.
- Sprzątanie Edenu 28.09: analiza 192,4 GB i 146 993 pliki, wykonanie „było 193,2 GB” i 148 036 plików.
- Wariant B diagnozy FastGAN-a: w projekcie 1000 kroków, w wynikach 400. Zmiana „w bok”: plik 58,4-78,3, rozmowa „60-78”.
- Ramię `current`: plan opisuje sterowanie do obecnej siły, kod i `ALGORYTM.md` biorą obraz z inwersji bez sterowania.
- Wersja 2, „8 z 10”: jedna z wygranych to różnica 0,00005. Trafność kierunku prognozy: wstępne 85 procent (3 ziarna)
  poprawione na 69 procent (10 ziaren). Iteracja 2: p dla końca uczenia jest tylko w `PLAN.md`.
- Użytkownik o 18:32 i 18:38 mówił o „całym przewidywanym latencie”, o 19:54 o „tylko tożsamości”; nie wiadomo, czego dotyczyła
  zgoda „możesz to zrobić!” z 19:26. Commit `fb899bde` zapis sesji przypisuje sesji równoległej, a zapis rozmowy pokazuje go w
  tej rozmowie.

**Etap 5.**
- Od 29.09, 14:30 w zapisie rozmowy biegną dwie nitki: główna (horyzont, równanie, macierze i dalej wszystko) i rozgałęzienie
  (kierunki tożsamości, literatura, model pikselowy, do 17:47); sesja-fork to trzecia, osobna sesja. Fork nazywa rozgałęzienie
  „sesją główną”, a zapis sesji głównej nazywa je „równoległą sesją”.
- Cztery liczby w okolicy 85: lokalnie na ziarnie 47 to 85,8 (T1 plus siła paczki) i 85,4 (równanie proste, a 0,834, b 0,985);
  na 10 ziarnach Edenu to mediany 85,4 (T1 plus siła paczki) i 85,0 (równanie proste, a 0,831, b 0,981). Zapis sesji łączy
  „0,99 S” z wynikiem 85,8.
- `WYNIKI.md` horyzontu podaje odsetki ziaren okrąglej niż JSON (Koopman wprost spada do 85-95 procent ziaren przy 550 i 650
  krokach).
- Kwota: „118,6/121,8 z 200 GB” to w GiB 113,1/116,2; „około 139 GiB” po latentach to według `lfs` 132,4 GiB.
- Podgląd Stable Diffusion: w rozmowie „32 wygenerowane klatki”, według kodu `zapisz_podglad` to 16 par, w każdej prawdziwa
  klatka po koderze i dekoderze VAE nad obrazem wygenerowanym z jej masek.
- Klatki z obrysem: 56 301 (liczba z rozmowy) wobec 56 005 w `obrysy.npz`. Tempo 55 obrazów/s na jednej H100 jest tylko w
  streszczeniu rozmowy, bez osobnego pomiaru.
- Model pikselowy: los fragmentu 1818356 i jakość końcowa nieustalone.
- Plan Stable Diffusion mówi, że koder tekstu nie jest ładowany, kod ładuje go na chwilę (do policzenia pustego opisu).
- Czas na ziarno równania: szacunek 25 min, zmierzone 17-19 min. Przyczyna błędu pierwszego monitora (kod 4) nieznana; zadania
  1818493 nie ma w źródłach.


