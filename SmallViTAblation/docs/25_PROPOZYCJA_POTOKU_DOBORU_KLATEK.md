# 25. Propozycja: dobór klatek jako sterowanie uczeniem (równanie ruchu latentu + Stable Diffusion z maskami)

**Status: PROPOZYCJA do zgody użytkownika, 2026-09-30. Nic z tego nie zostało uruchomione.** Każdy etap (E0-E4)
wymaga osobnego „tak” przed startem, zgodnie z regułą „plan przed treningiem”.

Źródła: opis całej linii prac `24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md`, metoda
`23_METODA_PRZEWIDYWANIA_LATENTU.md`, plan Stable Diffusion `../experiments/dyfuzja_sd_filmy/PLAN.md`, zapisana literatura
(`../experiments/fastgan_odwrotnosc_mlp/NOTATKA_LITERATURA.md`, `.../literatura/pelne_teksty_sd_2026-09-29.md`,
`../experiments/koopman_tozsamosc_kierunki/NOWOSC_I_PUBLIKACJA.md` i `TEORIA_LANGEVIN.md`, dokumenty 19, 22 i 44-46)
oraz baza wiedzy projektu grantowego (schemat diagnostyki z 2026-09-18, dowody Lean `ViTFrameSelection.lean`,
wynik `tft_langevin_petla_aktywna`, prace o pamięci optymalizatora i Koopmanie z wejściem).

## 1. Idea w jednym akapicie

Do tej pory przewidywaliśmy, dokąd latent tożsamości narzędzia pójdzie sam, i szukaliśmy klatek podobnych do obrazu
tego przewidywania. To nie pomagało, bo sam dryf latentu jest mały, a lista najbliższych klatek się od niego nie zmienia.
Nowe równanie mówi coś mocniejszego: ruch latentu w uczeniu to bezwładność plus siła bieżącej paczki, a siłę paczki
umiemy policzyć, zanim sieć się na niej uczy. Proponuję więc odwrócić pytanie: **dla każdej kandydującej klatki, prawdziwej
albo narysowanej przez Stable Diffusion, policzyć, jak przesunie ona detektor, i wybrać różnorodny zestaw klatek, który
najbardziej obniży błąd detekcji tam, gdzie detektor jest słaby.** Stable Diffusion z maskami dostarcza kandydatów w
miejscach, których w prawdziwej puli brakuje, i od razu z etykietą, bo ramka wynika z maski. Równanie Langevina staje się
modelem sterowania: stan to latent i jego prędkość, wejście to skład paczki, a dobór klatek to wybór wejścia.

## 2. Co wiemy (punkt wyjścia z liczbami)

1. Prognoza biernego dryfu latentu nie pomogła w doborze: w każdym porównaniu, w którym prognoza była jedyną różnicą, wynik
   mieścił się w szumie ziaren (dokument 24, sekcja E.1). Przewidziane przesunięcie w Nano to 0,069 rozrzutu na 100
   aktualizacji; lista top-k wyszukiwania zmieniała się w 6 z 60 pozycji.
2. Mały ViT od zera uczy się szybciej, gdy dostaje klatki podobne do wycinków narzędzia (iteracja 2 wersji 3: 15 z 20
   ziaren nad losowymi, p = 0,006), ale ramię bez prognozy daje prawie to samo. Zysk daje skupienie na narzędziu.
3. Na dostrajanym RF-DETR nic nie pobiło losowania; wspólna przyczyna to utrata różnorodności (etapy 1-3).
4. Równanie ruchu latentu małego ViT: `Δz(t+1) = 0,83·Δz(t) + 0,98·S(t)` wyjaśnia 85 procent ruchu na 10 ziarnach, placebo
   (siła z niewłaściwej paczki) nic nie wnosi. Siła paczki S(t) to zmiana latentu sond po „wirtualnym kroku”: wagi
   przesunięte tylko o tę część kroku AdamW, która pochodzi z danej paczki.
5. Współczynnik przy sile to prawie dokładnie 1, więc wirtualny krok wiernie opisuje prawdziwy krok. To jest dowód, że
   skutek klatek da się liczyć przed uczeniem.
6. Bezwładność 0,83 znaczy, że jedna paczka działa przez kilkanaście kroków: jej łączny skutek to około 0,98 / (1 − 0,83)
   ≈ 5,8 raza skutek pierwszego kroku, połowa skutku przychodzi w 3,7 kroku, 90 procent w 12 krokach.
7. Twierdzenie T9 (dokument 44) pokazuje, że reguła oparta wyłącznie na dynamice latentu nie może gwarantować poprawy:
   dwie klatki o identycznym latencie mogą mieć przeciwne gradienty. Gwarancję daje tylko przewidziana poprawa straty
   zadania większa od błędu tego przewidywania (T5, T6; rdzeń sprawdzony w Lean, `ViTFrameSelection.lean`).
8. Literatura (zapisane notatki): dobór bije losowanie tylko wtedy, gdy kryterium jest związane ze stratą zadania i
   zachowuje różnorodność (GLISTER, RHO-LOSS, JEST, Gupte 2024, Bengar 2019); wyszukane prawdziwe obrazy uczą zwykle lepiej
   niż syntetyczne (Wang 2015, Geng 2024, Burg 2023); dane syntetyczne z dyfuzji dają małe zyski, najwięcej w rzadkich
   układach (Frisch 2023 dla zaćmy: F1 0,897 → 0,916, w rzadkich fazach do +0,10).
9. Stable Diffusion 1.5 dostrojone na wszystkich klatkach 118 filmów treningowych z maskami narzędzia, rogówki i źrenicy
   rysuje realistyczne klatki z narzędziem w miejscu obrysu (uczenie 1818496 trwa; ocena jakości jeszcze nie zrobiona).

## 3. Równanie jako model sterowania

Stan detektora w chwili t: latent tożsamości sond `z(t)` i jego prędkość `Δz(t)`, strata detekcji na zbiorze kontrolnym
`L(t)` (osobno w przedziałach układu sceny, rozdział 4.3) i stan optymalizatora (pęd `m`, rozrzut `v`). Wejście: skład
paczki `u(t)`, czyli które klatki idą do uczenia. Model (zmierzony dla latentu, do sprawdzenia dla straty w E0):

    Δz(t+1) = a·Δz(t) + b·S(u(t)) + szum          S(u) = z(θ + d_u) − z(θ)
    ΔL(t+1) = a_L·ΔL(t) + b_L·G(u(t)) + szum      G(u) = L(θ + d_u) − L(θ)

`θ` to wagi, `d_u` to część kroku AdamW wyliczona z gradientu paczki `u` przy obecnym stanie optymalizatora (funkcja
`kierunki()` w `experiments/maly_vit_rownanie_latentu/nagranie.py`), `G(u)` to zmiana straty kontrolnej po wirtualnym
kroku, liczona wprost, bez przybliżenia pierwszego rzędu (ważne dla RF-DETR, gdzie dopasowanie ramek do celów psuje
gładkość straty; dokument 44, uwaga do T5).

To jest liniowy układ z wejściem, czyli najprostsza postać Koopmana ze sterowaniem (Korda-Mezić, DMDc w bazie wiedzy
grantu). Dobór klatek staje się sterowaniem predykcyjnym: w każdym obrocie wybieramy wejście, które według modelu najbardziej
obniży stratę w horyzoncie około 12 kroków (bo tyle trwa 90 procent skutku paczki), a potem sprawdzamy, co naprawdę się
stało, i poprawiamy model.

**Reguła o przyszłych klatkach jest zachowana:** wirtualny krok liczymy w chwili t na obecnych wagach dla kandydatów, a nie
dla przyszłych paczek; dryf z samego pędu używa tylko stanu optymalizatora z chwili t; bank syntetyczny powstaje przed
uczeniem, z filmów treningowych. Prawdziwy skutek po t służy wyłącznie do oceny prognozy.

## 4. Moduły potoku

### 4.1 Pula prawdziwa (rozszerzona)

Wszystkie klatki 118 filmów treningowych Cataract-LMM (na Edenie od 30.09, z obrysami), zamiast dotychczasowej puli z
regułą ±30, w której zostawało około 3300 klatek (w iteracji 2 zużyto 1700-1900). Reguła wykluczenia sąsiadów (±30 klatek od
bazy i od już dodanych) zostaje, bo losowanie z wykluczeniem sąsiadów bije zwykłe losowanie w wideo (Bengar 2019).
Dziesięć filmów treningowych odkładamy jako **zbiór kontrolny** (rozdział 4.3), żeby walidacja 3000 klatek została ta sama
co w wersji 3 i wyniki były porównywalne.

### 4.2 Pula syntetyczna: bank Stable Diffusion

Bank rysujemy raz, przed uczeniem, więc nie zależy od przebiegu uczenia. Trzy przepisy:

- **R1, rzadkie układy** (wzorem Frisch 2023): bierzemy prawdziwą klatkę, zostawiamy jej maski rogówki i źrenicy, a maskę
  narzędzia przekształcamy (obrót do ±60°, przesunięcie w obrębie rogówki, skala 0,8-1,2), losując układy z odwrotności ich
  częstości w danych. Rysujemy od szumu. Tak powstaje narzędzie w miejscach i pod kątami, których pula ma mało.
- **R2, nowy wygląd tego samego układu**: te same trzy maski, nowe losowanie szumu. Działa jak mocna augmentacja; porównamy
  ją z prostym punktem odniesienia (Mixup albo wklejanie narzędzia po obrysie), bo u Karthikeyana sam Mixup dawał prawie
  tyle samo co syntetyki.
- **R3, trudne wersje kotwic (później, opcjonalnie)**: wzorem Farida 2023, prawdziwa klatka zaszumiona do około 200.-250.
  kroku z 1000, odszumiana z prowadzeniem małym ViT w stronę słabszej siły tożsamości, z filtrem zgodności kierunków
  (zmiana sieci wchodzi tylko tam, gdzie zgadza się z generatorem) i karą za oddalenie od oryginału. Siłę małego ViT
  liczymy na estymacie czystego obrazu, gradient wygładzamy (Väth 2025). Dopiero gdy R1 okaże się użyteczne.

Etykieta każdej syntetycznej klatki to ramka opisana na masce narzędzia. Filtry przed wejściem do banku:
zgodność z maską (mały ViT uczony na prawdziwych klatkach znajduje narzędzie z IoU co najmniej 0,5 wobec ramki z maski;
alternatywnie mod Koopmana z DINOv2, który pokrywał się z maską obiektu z AUC 0,98, teza D98), test kopiowania (klatka
bliższa najbliższej klatce uczącej niż sąsiednie klatki tego samego filmu jest odrzucana), filtr środkowych 50 procent
odległości od prawdziwych klatek w cechach DINOv2 (Karthikeyan). Wielkość: około 20 000 klatek przed filtrami.

### 4.3 Mapa braków: gdzie detektor jest słaby

Zbiór kontrolny (klatki 10 odłożonych filmów treningowych) dzielimy na przedziały układu sceny policzone z masek: położenie
końcówki narzędzia względem źrenicy, kąt, długość widocznej części, a osobno klatki z innymi cienkimi narzędziami (pęseta),
które były główną pomyłką na zbiorze testowym. Na początku każdego obrotu liczymy stratę detekcji w każdym przedziale.
Wzorem RHO-LOSS i schematu diagnostyki grantu celujemy w część usuwalną błędu: przedziały, w których strata bieżącego
detektora jest wysoka, a strata detektora odniesienia niska. Detektor odniesienia to mały ViT uczony raz, przed kampanią, na
bazie 1500 klatek plus 3000 losowych klatek puli, potem zamrożony. Waga przedziału w celu to dodatnia część różnicy „strata
bieżąca minus strata odniesienia”; przedziały, w których oba detektory mylą się tak samo (błąd nieusuwalny, np. szum
adnotacji), nie przyciągają wyboru. Wynik wyboru po samej
wielkości błędu był w TS gorszy od braku ważenia (`tft_langevin_petla_aktywna`: 65,8 wobec 78,3), dlatego nie ważymy klatek
wielkością błędu, tylko wybieramy według przewidzianego skutku.

### 4.4 Siła kandydata: wirtualny krok

Dla każdej kandydującej grupy klatek (8-16 klatek; prawdziwych albo syntetycznych) liczymy gradient, część kroku AdamW `d_u`
przy obecnym stanie optymalizatora i dwie wielkości:

- `G(u)`: zmianę straty kontrolnej po wirtualnym kroku, osobno w każdym przedziale mapy braków;
- `S(u)`: zmianę latentu tożsamości sond (wektor `nowy` z par z narzędziem i bez), jako diagnostykę i drugi składnik celu.

Koszt: jedno przejście wstecz grupy i jedno przejście w przód zbioru kontrolnego oraz sond na kandydata. Wstępny odsiew
kandydatów robimy tanio, podobieństwem w cechach obiektowych (tokeny w ramce powiększonej 1,5 raza, jak CSOD: 44,3 wobec
37,9 dla losowania), żeby wirtualny krok liczyć dla kilkuset, a nie dziesiątek tysięcy kandydatów. Tanie przybliżenia bez
przeliczania sieci odpadają, bo ostatni blok niesie tylko 7,5 procent ruchu latentu (dokument 23).

### 4.5 Dryf z pędu: co i tak się stanie

Przed wyborem liczymy, jak zmieni się strata kontrolna bez nowych klatek: wirtualny krok z samej części pędu `d_pęd`
(prognoza dozwolona, `0,94·S_pęd` dla latentu). Cel wyboru to różnica między tym, czego potrzebujemy w każdym przedziale, a
tym, co pęd dowiezie sam. Wielokrokowa wersja tej prognozy (T4) nie była sprawdzona; sprawdzamy ją w E0, razem z testem
resetu pędu (protokół z pracy o pamięci uczenia, arXiv 2601.21624), który niezależnie mierzy bezwładność.

### 4.6 Wybór całej paczki z różnorodnością

Wybieramy zestaw na cały obrót (134 klatki, jak w wersji 3) zachłannie, maksymalizując sumę przewidzianych spadków straty
minus karę za podobieństwo do już wybranych. Ograniczenia, każde z powodem:

- jedna klatka na skupisko w cechach obiektowych i wykluczenie sąsiadów ±30 klatek, bo top-k bez różnorodności przegrywa z
  losowaniem (Gupte 2024) i tak przegrywały nasze ramiona celowane;
- limit klatek z jednego filmu, bo wybór w pętli Large skupiał się wokół kilku kotwic;
- część zestawu (25-50 procent) losowa, bo losowanie pokrywa całą pulę, a efekty celowania są małe;
- najwyżej 30 procent klatek syntetycznych, bo wyszukane prawdziwe obrazy uczą zwykle lepiej niż syntetyczne;
- ocena całego zestawu, a nie pojedynczych klatek, bo skutek grupy nie jest sumą skutków (człon `g_iᵀ H g_j` w dokumencie 44;
  JEST 2024: wybór łączny bije wybór po jednej próbce).

### 4.7 Bezpiecznik: kalibracja i powrót do losowania

Po każdym obrocie porównujemy przewidziany i zmierzony spadek straty kontrolnej. Z tej różnicy szacujemy błąd prognozy `E`.
W następnym obrocie klatki wybrane siłą wchodzą tylko wtedy, gdy ich przewidziany zysk po odjęciu błędu jest większy niż
zysk losowego zestawu po dodaniu błędu (T6: `â_S − E_S > â_B + E_B`); w przeciwnym razie miejsce wypełniamy losowo. Z tego
samego porównania poprawiamy współczynniki `a_L`, `b_L` dla danego ziarna. Z założenia potok nie powinien więc wypaść
gorzej niż losowanie o więcej niż błąd prognozy.

### 4.8 Syntetyki jako zapytania (osobne ramię)

Wzorem ASAL i Data-Centric Debugging klatki z banku w słabych przedziałach mogą też służyć tylko jako zapytania: szukamy do
nich najbliższych prawdziwych klatek w cechach samego detektora (nie DINOv2 z całej klatki, bo mały szczegół jak końcówka
narzędzia ginie w cechach całej sceny). To ramię oddziela zysk z „nowych pikseli” od zysku z „lepszego wyszukiwania”.

## 5. Różnice wobec pętli wersji 3 (metoda użytkownika)

| element | wersja 3 | propozycja |
|---|---|---|
| co przewidujemy | siłę tożsamości 180 sond na 500 kroków, Koopmanem albo Langevinem | skutek każdej kandydującej klatki (wirtualny krok) na stracie kontrolnej i latencie sond; dryf z samego pędu |
| cel | przewidziana zmiana siły tożsamości | spadek straty detekcji w słabych przedziałach; siła tożsamości jako składnik drugi |
| generator | FastGAN, wycinek 256 liczb, zmienia resztę latentu około 40 razy mocniej niż tożsamość | Stable Diffusion z maskami: cała klatka, narzędzie w zadanym miejscu |
| rola generatora | tylko obraz zapytania | zapytanie do wyszukania albo klatka ucząca z ramką z maski (osobne ramiona) |
| wyszukiwanie | DINOv2 z całej klatki, 90 najbliższych na kotwicę | cechy obiektowe i cechy detektora, potem ocena wirtualnym krokiem |
| pula | pula z regułą ±30, około 3300 klatek | wszystkie klatki 118 filmów treningowych bez 10 kontrolnych, plus bank syntetyczny |
| wybór | 134 klatki na przemian z 6 list | cały zestaw 134 klatek z ograniczeniami różnorodności, częścią losową i limitem syntetyków |
| bezpiecznik | brak | kalibracja przewidywanie-pomiar i powrót do losowania (T6) |
| Koopman | wybiera zwycięzcę prognozy siły | tylko w postaci liniowego modelu z wejściem (równanie z rozdziału 3); jako prognoza dryfu nie pomógł w żadnym porównaniu |

Bez zmian: mały ViT od zera, rozgrzewka 1000 kroków, 7 obrotów po 500 kroków, 134 klatki na obrót, mAP@0,5 co 100 kroków na
tych samych 3000 klatkach walidacji, wynik główny to pole pod krzywą mAP@0,5, 20 ziaren, zbiór testowy nietknięty do końca.

## 6. Etapy z bramkami (każdy wymaga osobnej zgody)

### E0. Czy wirtualny krok przewiduje prawdziwy skutek klatek? (lokalnie, RTX 3070, około 1-2 h)

Mały ViT, ziarno 47, zapis pełnego stanu (wagi i optymalizator) w krokach 1000 i 2000. W każdym stanie 64 grupy po 16
klatek: 16 losowych, 16 z wyszukiwania jak w wersji 3, 16 o największej stracie, 16 z najsłabszych przedziałów. Dla każdej
grupy przewidujemy `G(u)` i `S(u)`, a potem od tego samego stanu uczymy 50 kroków paczkami „16 zwykłych + 16 z grupy” i
mierzymy prawdziwy spadek straty kontrolnej po 1, 10 i 50 krokach, wobec przebiegu „16 zwykłych + 16 losowych”.
Kryteria ustalone przed wynikiem:

- korelacja kolejności (Spearman) przewidzianego i prawdziwego skutku co najmniej 0,5 po 10 krokach i co najmniej 0,3 po 50;
- lista 16 najlepszych według siły różni się od listy z wyszukiwania (wspólna część, Jaccard, poniżej 0,5), bo inaczej siła
  nie wnosi nic ponad wyszukiwanie;
- równanie `ΔL(t+1) = a_L·ΔL(t) + b_L·G(u(t))` dla straty wyjaśnia wyraźnie więcej niż „brak zmiany”, a prognoza dryfu z
  samego pędu bije „brak zmiany” i „ostatni przyrost” na 10 krokach (T4); test resetu pędu potwierdza bezwładność.

Jeśli E0 nie przejdzie, cały potok nie ma podstaw i kończymy na nim.

### E1. Ocena Stable Diffusion (Eden, 1 × H100, około 2 h, po zakończeniu 1818496)

Skrypt `experiments/dyfuzja_sd_filmy/kod/ocena_sd.py`: FD-DINOv2 dla 5000 próbek z masek filmów `val` wobec 5000 klatek `val`
(z odniesieniem `train` wobec `val`), test kopiowania, zgodność z maską (mały ViT uczony na prawdziwych klatkach: IoU
co najmniej 0,5 wobec ramki z maski), przegląd 64 próbek przez użytkownika, czas rysowania jednej klatki. Kryterium
ustalone przed wynikiem: zgodność z maską co najmniej 70 procent klatek i odsetek kopii poniżej 1 procent. Tu też dobieramy
siłę wzmocnienia warunku (CFG 1,5 albo 3).

### E2. Bank syntetyczny (Eden, 1-2 × H100, szacunek 1-2 h do zmierzenia w E1)

20 000 klatek w przepisach R1 i R2, filtry z rozdziału 4.2, zapis statystyk (ile odpadło w każdym filtrze, rozkład
układów przed i po).

### E3. Kampania główna na małym ViT (Eden, 20 ziaren, szacunek kilka godzin na 8-10 kartach; najpierw próba 1 ziarna)

Ramiona z tego samego stanu po rozgrzewce, ten sam budżet klatek:

| ramię | co robi | po co |
|---|---|---|
| `losowe` | losowe prawdziwe klatki, reguła ±30 | punkt odniesienia |
| `wyszukiwanie` | jak `current` z wersji 3: klatki podobne do wycinków narzędzia, bez prognozy | obecnie najlepsza metoda |
| `sila` | wybór wirtualnym krokiem z prawdziwej puli, z różnorodnością i bezpiecznikiem | główna metoda bez syntetyków |
| `sila_odwrocona` | ten sam potok, ale klatki o najgorszym przewidzianym skutku | ramię różniące się tylko sygnałem siły: sprawdza, czy siła niesie informację |
| `sila_sd` | wybór wirtualnym krokiem z puli prawdziwej i banku syntetycznego | czy syntetyki pomagają przy dobrym wyborze |
| `sd_losowe` | losowe klatki z banku, ta sama liczba | czy syntetyki pomagają same z siebie |
| `sd_zapytania` | syntetyki ze słabych przedziałów jako zapytania, uczenie na znalezionych prawdziwych klatkach | nowe piksele czy lepsze wyszukiwanie |
| `rho` | RHO-LOSS: wysoka strata bieżącego detektora, niska detektora odniesienia | punkt odniesienia z literatury |
| `bez` | bez dokładania | dolna granica |

Kryteria ustalone przed wynikiem (pole pod krzywą mAP@0,5 na walidacji):

- K1: `sila` lepsze od `wyszukiwanie` w co najmniej 14 z 20 ziaren i test Wilcoxona p < 0,05;
- K2: `sila` lepsze od `sila_odwrocona` w co najmniej 16 z 20 ziaren (siła niesie informację);
- K3: `sila_sd` lepsze od `sila` w co najmniej 14 z 20 ziaren (syntetyki pomagają);
- opisowo: `sd_losowe` wobec `losowe`, `sd_zapytania` wobec `wyszukiwanie`, `rho` wobec `sila`, straty w przedziałach mapy
  braków na końcu uczenia, wskaźnik bezpiecznika (ile klatek wybrała siła, a ile losowanie).

### E4. Przeniesienie na RF-DETR Nano (tylko gdy E3 spełni K1 i K2)

Ten sam potok na dostrajanym RF-DETR Nano, na którym zaczęła się cała linia prac, z wirtualnym krokiem liczonym na stracie
z dopasowaniem ramek. Osobny plan i osobna zgoda.

## 7. Ryzyka i co wtedy

- **Wirtualny krok nie przewiduje skutku po wielu krokach** (bo inne paczki zacierają ślad; wymuszenie musi trwać dłużej niż
  czas relaksacji, Lucarini 2025): E0 to wykryje; wtedy albo trzymamy wybrane klatki w paczkach przez cały obrót (jak dotąd),
  albo rezygnujemy z potoku.
- **Stable Diffusion rysuje narzędzie niezgodnie z maską albo kopiuje klatki**: E1 to wykryje; wtedy syntetyki tylko jako
  zapytania (ramię `sd_zapytania`), bez uczenia na nich.
- **Wzorce w stylu ataku przy prowadzeniu małym ViT** (R3): prowadzenie tylko przez przestrzeń generatora, z filtrem zgodności
  i wczesnym zatrzymaniem (ReNO, Farid); R3 dopiero po R1.
- **Efekty w szumie**: 20 ziaren i ramię `sila_odwrocona`; bez K2 wynik K1 nie przesądza o roli siły.
- **Koszt**: wirtualny krok dla kilkuset kandydatów na obrót to kilka procent czasu uczenia małego ViT; próba 1 ziarna mierzy
  czas przed zgłoszeniem całości.

## 8. Czego to nie dowodzi, nawet gdy się uda

- Wynik dotyczy małego ViT uczonego od zera na tym zbiorze; przeniesienie na RF-DETR to osobny etap (E4).
- Dobry wygląd syntetyków nie znaczy, że pomagają; rozstrzygają ramiona `sd_losowe` i `sila_sd`.
- Zgodność przewidzianego i zmierzonego skutku w krótkim horyzoncie nie dowodzi, że dobór poprawia wynik końcowy uczenia;
  rozstrzyga pole pod krzywą i mAP na końcu, a na samym końcu zbiór testowy, użyty raz.

## 9. Do decyzji użytkownika

1. Zgoda na E0 (lokalnie, bez Edenu, około 1-2 h na RTX 3070).
2. Zgoda na E1 po zakończeniu uczenia Stable Diffusion (1 × H100, około 2 h) i na przegląd 64 próbek.
3. Wybór udziału syntetyków (proponuję najwyżej 30 procent) i części losowej (proponuję 25-50 procent) przed E3.
4. E2, E3 i E4 dopiero po wynikach poprzednich etapów, każdy z osobnym planem liczby kart i czasu.
