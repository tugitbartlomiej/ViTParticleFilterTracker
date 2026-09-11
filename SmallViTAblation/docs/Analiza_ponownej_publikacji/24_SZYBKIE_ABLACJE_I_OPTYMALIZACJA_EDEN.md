# 24 — Szybkie ablacje kuracji: małe detektory, potem transfer

Data: 2026-09-11. **Status: plan optymalizacji i badań, bez nowych pomiarów czasu na Edenie.** Zmienia kolejność prac na życzenie autora: szybkie testy na małych modelach przed szeroką macierzą. Uzupełnia [protokół 01](../01_PROTOCOL.md), [audyt publikacyjny 22](../22_AUDYT_LITERATURY_I_PLAN_PUBLIKACJI.md) i [połączenie z grantem 23](23_POLACZENIE_Z_GRANTEM_DYNAMIKA_LATENTU.md). Nie jest gotową konfiguracją obecnego runnera.

## 1. Decyzja: czego używać najpierw

**Pierwszy detektor roboczy: YOLOv8n z wag COCO.** Pozwala zachować odniesienie do rodziny YOLOv8 z poprzedniej pracy. Najpierw mierzymy jego trening, poprawność lokalizacji końcówek i reakcję na zmianę danych. Do potwierdzenia między rodzinami proponuję **RT-DETRv2-S/R18**. Jeśli potrzebny jest rzeczywisty backbone ViT, dodać do kalibracji **RF-DETR-Nano** i uruchomić go na ograniczonej macierzy.

To rekomendacja do benchmarku, nie obietnica najszybszej zbieżności. Mały model może zbyt słabo lokalizować końcówki albo być bardziej podatny na selekcję niż duży. **Nie wybieramy modelu po tym, na którym różnica na korzyść kuracji jest największa.** Wybieramy go po poprawności, koszcie, stabilności i zdolności do sensownego rozwiązania zadania.

| Model | Rola | Co wiadomo ze źródła, a co trzeba zmierzyć |
|---|---|---|
| **YOLOv8n** | Główny szybki detektor do development i ablacji | Oficjalnie około 3,2 mln parametrów przy referencyjnym COCO; liczba zmienia się z głową/wersją. Docelowy model to detektor bbox. [Dokumentacja](https://docs.ultralytics.com/models/yolov8). |
| YOLO11n | Rezerwa, jeśli benchmark wykaże korzystniejszy koszt/jakość | Oficjalnie 2,6 mln parametrów. Wybór zamiast v8n trzeba ustalić przed porównywaniem selektorów. Nie trzeba trenować pełnej macierzy obu. [Dokumentacja](https://docs.ultralytics.com/models/yolo11). |
| **RT-DETRv2-S/R18** | Wczesna kontrola transferu do rodziny DETR | Oficjalnie 20 mln parametrów, referencyjne wejście 640. R18 jest backbone’em konwolucyjnym, choć detektor zawiera transformery. [Repozytorium autorów](https://github.com/lyuwenyu/RT-DETR). |
| **RF-DETR-Nano** | Opcjonalny detektor z backbone’em ViT, zgodny z zainteresowaniem małymi ViT | Aktualne repo deklaruje DINOv2 ViT, około 30,5 mln parametrów po fuzji i domyślne 384. „Nano” nie znaczy mniej parametrów niż RT-DETRv2-S ani szybszego treningu od YOLO. [Repozytorium](https://github.com/roboflow/rf-detr). |
| YOLOv8s, później YOLOv8l | Transfer skali; v8l łączy się z historycznym modelem | s jest rozsądnym stopniem pośrednim; l uruchamiać dla zamrożonych kontrastów, nie całej siatki. |
| DETR-R50 i większy RT-DETR | Odtworzenie historycznej rodziny oraz osobne potwierdzenie nowoczesnej rodziny | Kilka kluczowych par, uczciwy budżet i czysta inicjalizacja. RT-DETR nie zastępuje automatycznie testu zwykłego DETR-R50. |

Tabele producentów opisują głównie **inferencję**, często TensorRT, FP16 i batch=1. Nie wyliczać z nich czasu treningu na H100. Liczb parametrów z różnych tabel nie traktować jako identycznie mierzonej wielkości; benchmark zapisze parametry całkowite/trenowalne konkretnego obiektu PyTorch.

**Sam DeiT-Tiny, ViT-Tiny lub DINOv2-S nie jest detektorem.** Dołożenie własnej głowy bbox, matcherów i treningu tworzy nową architekturę i dodatkowe źródło błędów. Na tym etapie gotowy mały detektor daje szybszy wynik naukowy niż budowanie kolejnego modelu. Zamrożony ViT nadal może tanio dostarczać cache cech selektora.

## 2. Trzy różne role modeli

1. **Ekstraktor selektora:** DINO lub później V-JEPA. Oblicza cechy raz dla ustalonej puli i preprocessingu.
2. **Proxy trudności:** opcjonalny mały detektor, jeśli testujemy score zależny od loss/predykcji. Jego trening i pochodzenie danych są częścią kosztu kuracji.
3. **Detektor oceniany:** YOLOv8n w development, następnie inne architektury. Każdy bieg startuje ze wspólnych dla swojej architektury wag ogólnych. Nie przejmuje wytrenowanego checkpointu proxy.

Wariant dynamiki z dokumentu 23 na początku potrzebuje tylko roli 1 i taniego ridge na CPU. Wyłączenie proxy z tego wariantu nie oznacza usunięcia go z wiernej kontroli historycznego DESF. [Selection via Proxy](https://arxiv.org/abs/1906.11829) uzasadnia używanie tańszych modeli do selekcji, ale badana tam klasyfikacja nie stanowi dowodu transferu do detekcji końcówek.

## 3. Dane i rozdzielczość: oszczędzać bez zmiany pytania

- Najpierw zamrozić manifest pacjentów/operacji i policzyć **unikalne klatki źródłowe**. Początkowy K=2500 można zastosować tylko, gdy pula na to pozwala. Drugi K=5000 jest warunkowy; wartości nie zastępują aktualnego inwentarza.
- Kandydaci do każdej metody pochodzą z tej samej puli train. Nie dawać dynamice gęstszego filmu niż konkurentom i nie ukrywać kosztu dodatkowo obejrzanych klatek.
- Klatek validation/test nie selekcjonować badanym algorytmem. Pełna, stała walidacja jest preferowana; ewentualny mały monitor development ustalić raz, per operacja, i nie używać go do końcowych tabel.
- YOLO zacząć od **640**, potem sprawdzić 512 w benchmarku rozdzielczości. RF-DETR-Nano porównać przy 384 i wyższej obsługiwanej rozdzielczości, jeśli małe końcówki tracą widoczność. RT-DETRv2-S zacząć od recepty 640.
- Zmierzyć rozkład szerokości/wysokości bbox po rzeczywistym resize/crop/letterbox: medianę, p10 i odsetki bardzo małych obiektów. Próg typu 4 piksele jest diagnostyką, nie uniwersalną normą medyczną. Rozstrzyga również walidacyjny AP75 i recall.
- W obrębie architektury wszystkie selektory mają tę samą rozdzielczość i receptę. Przy różnych rozdzielczościach między architekturami wniosek dotyczy transferu między **receptami modelowymi**; nie izoluje samej architektury.

Augmentacje wykonywać po selekcji. Cache gotowych cech zamrożonego selektora można współdzielić; cache aktywacji trenowanego backbone’u detektora staje się nieaktualny po zmianie wag i nie zastępuje treningu całej sieci. Eksperyment z zamrożonym backbone’em/head-only jest osobnym zadaniem pomocniczym.

## 4. Etapy, które ograniczają liczbę treningów

Liczby poniżej są **planem**, nie wykonanymi biegami ani minimalną liczbą wymaganą przez statystykę. Stara siatka 126 biegów z dokumentu 22 pozostaje projektem szerokiej walidacji; nie uruchamiamy jej automatycznie przed pilotem. Redukcja kosztu wymaga również zawężenia deklaracji publikacji.

### Etap 0 — poprawność i kalibracja sprzętowa

Najpierw zamknąć blokery z audytu 22: pojedynczy/lista bbox, GT i puste obrazy, wybór DINO, k-center, rozbieżność F/S, kompletność COCOeval i adapter właściwej rodziny. Mały overfit na kilkudziesięciu klatkach jest kontrolą implementacji, a nie wynikiem kuracji.

Na jednym typie GPU wykonać krótkie profile YOLOv8n, RT-DETRv2-S i ewentualnie RF-DETR-Nano. Użyć tej samej neutralnej próbki źródeł, np. równomiernie z train. Pomiar obejmuje rozgrzewkę oraz co najmniej kilkaset kroków forward/backward/optimizer; 200 kroków po rozgrzewce to propozycja startowa. Dla malutkiego modelu wydłużyć okno, jeśli pomiar jest niestabilny.

Profil nie odpowiada na pytanie o zbieżność. Następnie doprowadzić 2–3 neutralne kontrole (`random`, `uniform`, pełna pula) do ustalonego limitu i ocenić krzywe development. To pozwala wybrać wspólny krótki i dłuższy budżet dla YOLOv8n bez dopasowywania ich do maksimum zysku nowej metody.

### Etap A — pierwsze porównanie: 9 krótkich biegów

Jeden mały detektor, K=2500, jedno ziarno development, ta sama krótka liczba **optimizer updates**:

| ID | Warunek |
|---|---|
| A0 | Random-global: losowanie źródeł |
| A1 | Uniform-video: równomierne w czasie, jawny przydział między filmami |
| A2 | Pokrycie DINO, prawdziwy k-center, bez sygnału temporalnego |
| A3 | Pokrycie + różnica kolejnych cech (persistence residual) |
| A4 | Pokrycie + residual ridge z dokumentu 23 |
| A5 | Pokrycie + score Taylora inspirowany Swift Sampling, ten sam wrapper wyboru |
| A6 | Sam residual ridge, bez składnika pokrycia; pozostałe quota/odstępy jak A4 |
| A7 | Wierna kontrola historycznego DESF po naprawach, z jawnym kosztem proxy |
| A8 | Pełna pula N, kontekst przy tym samym budżecie kroków |

A2–A6 mają te same quota i reguły odstępu. A5 to kontrola mechanizmu score; pełną adaptację oryginalnego Swift Sampling z lokalnymi maksimami dodać w potwierdzeniu, jeśli wniosek dotyczy przewagi nad tą pracą. Nie nazywać wspólnego wrappera wierną reprodukcją Swift.

Po pełnej lekturze Swift: [dokument 26](26_SWIFT_SAMPLING_ANALIZA_KRYTYCZNA.md) rozdziela resztę bezwzględną z PDF od względnej z kodu, politykę początku sekwencji oraz przypisanie score do targetu. Te wybory trzeba zapisać w A5 przed treningami. Jego osobny program 7/21/12 biegów dotyczy porównania mechanizmu Swift i diagnostyki; nie dodaje się automatycznie do poniższych 51 i nie zastępuje kontroli historycznego DESF.

Jeśli kontrola DESF nie jest gotowa, można rozpocząć development A0–A6/A8, ale odnotować **8 zamiast 9** biegów i otwarty warunek odniesienia historycznego. Nie zastępować DESF innym algorytmem pod starą nazwą.

### Etap B — powtarzalność i dłuższy trening: 18 biegów

Sześć warunków: random, uniform, A2, A4, A7 oraz jeden silny konkurent. Ten ostatni jest wybierany według jawnej reguły na development: z A3/A5 i, przed mocną tezą publikacyjną, odpowiednio zaadaptowanego CSOD/AdaDeDup/COWAL. Wszystkie dodatkowe treningi konkurentów doliczyć; podana liczba 18 zakłada **jedno już wybrane** ramię konkurencyjne.

Sześć warunków × trzy ziarna × jeden K × dłuższy budżet = **18**. Nowa metoda A4 pozostaje w tym etapie nawet wtedy, gdy nie wygrała A; pozwala sprawdzić główną hipotezę bez zastępowania jej zwycięzcą wybranym po wynikach. Warianty bliskie remisu w pilocie nie odpadają na podstawie niewielkiej różnicy jednego seeda.

### Etap C — transfer bez ponownego strojenia: 24 biegi

Po zamrożeniu selektora i manifestów: cztery warunki (random, uniform, A4, silny konkurent) × trzy ziarna × dwie dodatkowe recepty modelowe = **24**. Propozycja: RT-DETRv2-S oraz YOLOv8s; jeśli ważniejsze jest potwierdzenie backbone’u ViT, drugą receptą może być RF-DETR-Nano. Wybór zadeklarować przed obejrzeniem wyników tych selektorów.

Każdy model ma własny, uprzednio skalibrowany budżet; porównujemy **różnice względem jego baseline’ów**. Nie stwierdzamy wyższości architektury z porównania odmiennych recept. Te same wybrane frame IDs muszą przejść między modelami. Ponowne wybranie danych dla każdego modelu bada inny problem.

**A+B+C = 51 treningów docelowych** przy powyższym wariancie. W tym tylko 9 krótkich na pierwszym etapie. Kalibracja, proxy, dobór konkurenta i ponowienia po błędach są dodatkowymi pozycjami kosztu, nie ukrytym zerem. Biegi A nie są automatycznie zaliczane do B, bo budżet i recepta mogą się różnić.

### Rozszerzenia po bramkach

- Drugie K na YOLOv8n: te same sześć warunków × trzy ziarna = **18 dodatkowych** treningów.
- Ścisły most do poprzedniej pracy: DETR-R50 i YOLOv8l × (random, uniform, stary DESF, nowa metoda) × trzy ziarna = **24 dodatkowe**. Dodać odpowiedni silny baseline, jeśli deklaracja dotyczy przewagi nad nim również na dużych modelach.
- Pełna ablacja D/E/S/F na YOLOv8n: 15 kombinacji × trzy ziarna = **45** biegów przed odliczeniem naprawdę identycznych już wykonanych kontroli. Nie powtarzać jej automatycznie na wszystkich dużych modelach.
- Etapy A/B/C plus drugie K i ścisły most to **93 treningi**, nadal bez pełnej siatki D/E/S/F i kosztów pomocniczych. Każde rozszerzenie wymaga dopisania manifestu i kosztu.

Trzy ziarna to pragmatyczny punkt startowy do oszacowania rozrzutu, nie gwarancja mocy. Jeśli efekt jest mały lub pacjentów mało, zwiększenie liczby niezależnych operacji może być ważniejsze niż dokładanie seedów.

## 5. Budżet treningu: kroki, zbieżność i koszt

Początkowe wartości do **kalibracji**, nie gotowe zalecenie empiryczne: `S_fast=1000`, `S_confirm=4000` rzeczywistych aktualizacji optymalizatora. Krzywe kontrolne mogą wymagać innych limitów. Nie zakładać, że stałe 10 czy 30 epok wystarczy każdej rodzinie.

```text
B_eff = micro_batch * grad_accumulation * world_size
ekspozycje ≈ optimizer_updates * B_eff
równoważne_epoki ≈ ekspozycje / liczba_źródeł_w_podzbiorze
```

To przybliżenie pomija końcowe niepełne batche, zmienną akumulację i dodatkowe obrazy z mosaic/mixup. Logować rzeczywiste liczby. Przy B_eff=64 i K=2500: 1000 aktualizacji odpowiada około 25,6, a 4000 około 102,4 przejściom przez podzbiór. Pełna pula widzi przy tych samych krokach mniej przejść. Dlatego równy budżet kroków i równe epoki odpowiadają na różne pytania.

**Tor jakości danych:** stałe K oraz kroki i B_eff w obrębie modelu. Dla krzywej różnych K utrzymać wspólne kroki, pokazując ekspozycję. Harmonogram LR, warmup i wyłączanie augmentacji definiować względem aktualizacji albo stałego udziału budżetu, a nie zmiennej długości epoki.

**Tor oszczędności praktycznej:** dodatkowo jakość względem rzeczywistego czasu GPU i całkowitego kosztu, ewentualnie wspólna liczba epok. Próg jakości i margines niegorszości ustalić na development. Nie uznawać braku istotności za równoważność.

**Audyt adaptera jest konieczny:** framework może mieć automatyczną akumulację, nominalny batch, automatyczny wybór optymalizatora, scheduler zależny od epok albo specyficzne przejścia augmentacji. Przed macierzą zapisać rzeczywisty optimizer, LR, weight decay, B_eff i liczbę `optimizer.step`. Nie zakładać, że `batch=32` oznacza dokładnie jeden update na 32 obrazy. [Parametry YOLO](https://docs.ultralytics.com/modes/train/), [parametry RF-DETR](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/).

Nie zatrzymywać każdego wariantu w innym momencie, a następnie opisywać go jako kontrolowanej ablacji przy stałym koszcie. W takim torze wszystkie kończą wspólny budżet; raportować wynik ostatni i najlepszego checkpointu wybranego tą samą regułą na val. Early stopping może być osobnym eksperymentem praktycznej efektywności, z doliczonym kosztem i jawną regułą.

Jeśli planujemy promować istniejący krótki bieg do dłuższego, jego scheduler musi być z góry zgodny z dłuższym horyzontem i wznowienie zachować optimizer/EMA/RNG. Trening zakończony krótkim cosine schedule, a potem przedłużony, nie jest automatycznie tym samym co trening od początku z dłuższym schedule. Dlatego podstawowy kosztorys liczy A i B oddzielnie.

## 6. Optymalizacja Edena

Inwentarz lokalny z [EDEN_CONSTRAINTS.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/EDEN_CONSTRAINTS.md) opisuje A100/H100/H200 oraz ograniczenia kolejek; pochodzi z wcześniejszych pomiarów. **Nie wykonano teraz odczytu wolnych GPU ani uruchomienia Slurma.** `debug` pozostaje poza planem zgodnie z ograniczeniem projektu.

### 6.1 Domyślnie jeden trening na jednym GPU

Małe modele uruchamiać jako niezależne zadania lub ograniczoną tablicę Slurm, po **jednym GPU na bieg**. Równolegle liczyć różne warianty i ziarna. DDP kilku GPU dla jednego małego modelu wprowadza komunikację; wybrać je dopiero, jeśli pomiar pokaże korzyść dla całego eksperymentu, nie tylko jednego biegu.

Limit równoległości ustalić po sprawdzeniu przydziału i zasad klastra. Osiem GPU w węźle nie oznacza, że mamy je teraz dostępne ani że trzeba rezerwować cały węzeł. Dwa procesy na jednym GPU to oddzielny benchmark: VRAM nie dowodzi zwiększenia sumarycznej przepustowości. Nie włączać współdzielenia domyślnie.

### 6.2 Dane i cache

- Jedna wersjonowana pula obrazów, manifesty wariantów i anotacje; bez 51 kopii obrazów.
- Stage danych na sprawdzony lokalny scratch węzła, z katalogiem zadania. `/raid` jest kandydatem z inwentarza, nie gwarantowaną ścieżką na każdym węźle. Nie zakładać pojemności `/tmp` ani istnienia `SLURM_TMPDIR`.
- Jeśli kilka biegów korzysta z jednego cache na węźle, zapewnić blokadę tworzenia i jawne właścicielstwo. Zakończenie jednego zadania nie może usunąć danych używanych przez inne. Zachować best, last, stan wznowienia, logi i predykcje; dopiero potem sprzątać własny scratch.
- Cechy ekstraktora policzyć raz dla dokładnego hasha wag, obrazów, preprocessingu i warstwy. Uczony/probabilistyczny ekstraktor lub zmieniona rozdzielczość oznacza nowy cache. PCA/scalery dopasowywane w cross-fittingu nie są jednym wspólnym fittem.
- Rezidualy ridge i operacje wyboru wykonywać z cache’u bez ponownej inferencji DINO. Używać bloków/memmap, a nie pełnej macierzy odległości N×N. Dla k-center aktualizować wektor minimalnych odległości; koszt zależy od N, K i wymiaru.
- Wstępny resize do osobnego cache tylko po wykazaniu wąskiego gardła; zachować geometrię bbox i oryginały. PNG/JPEG po resize nadal wymaga dekodowania.

### 6.3 Precyzja i wykorzystanie GPU

Profilować natywny AMP frameworka; bf16 wybrać na zgodnym sprzęcie i adapterze, jeśli smoke test potwierdza stabilność. W innym przypadku użyć wspieranego fp16. FP32 można zachować dla czułych części numerycznych. Regresje/kowariancje selektora nie muszą używać tej samej precyzji co trening detektora.

Najpierw dobrać **fizyczny batch**, np. profil 16/32/64 przy stałej rozdzielczości, i dopiero potem zamrozić go między wariantami. Akumulacja jest sposobem zachowania efektywnego batcha przy ograniczeniu pamięci; nie daje automatycznie wydajności większego fizycznego batcha. BatchNorm również nie widzi akumulowanych batchy jak jednego dużego.

Dostroić liczbę workerów, prefetch, pamięć przypiętą i transfer asynchroniczny do przydzielonych CPU/RAM. `channels_last` sprawdzić dla części konwolucyjnych; `torch.compile` tylko po pomiarze czasu kompilacji, graph breaks i korzyści całego krótkiego biegu. Włączenie gradient checkpointing oszczędza pamięć kosztem przeliczania; używać po stwierdzeniu potrzeby. [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).

Nie nakładać wszystkich optymalizacji naraz. Porównać ze stabilną receptą bazową, kontrolując loss, predykcje i mały bieg walidacyjny; następnie zamrozić ustawienia. Tryb deterministyczny i szybkie niedeterministyczne kernele mają odmienny koszt — wybraną politykę ujawnić, a nie obiecywać bitowej powtarzalności samego seeda.

### 6.4 Walidacja i zapisy

Ograniczyć wykresy i ciężkie logowanie per batch; zachować metryki i czas. Ustalić jednakowy interwał walidacji w krokach oraz obowiązkową walidację na końcu. Większy batch inferencji val można dobrać niezależnie od batcha treningu. Predykcje z wybranych checkpointów zapisać raz w COCO i przeliczać metryki bez ponownej inferencji.

Przerwanie zadania wymaga pełnego resume z optimizerem, schedulerem, EMA, RNG, samplerem i licznikiem kroków. Samo wczytanie wag z best checkpointu jest nowym fine-tuningiem. Każdy bieg ma własny katalog; brak nadpisywania przez kolejny seed.

## 7. Co zmierzyć i jak policzyć czas

| Pole benchmarku | Jednostka / reguła |
|---|---|
| model, commit, checkpoint SHA, wersje, GPU | Faktyczny runtime, bez samych etykiet „nano/small” |
| batch fizyczny i efektywny, rozdzielczość | Rzeczywista konfiguracja, także akumulacja i augmentacje |
| ciepły czas optimizer update | Średnia, p50/p95, pomiar uwzględniający pracę GPU |
| przepustowość wejściowych obrazów | imgs/s; osobno realne źródła dodatkowo użyte przez mosaic/mixup |
| VRAM peak, RAM, data wait | Ustalenie, czy ogranicza GPU, CPU czy I/O |
| start/kompilacja, stage, validation, save | Czas osobno od kroków treningu |
| AP50:95/AP75/AR i krzywa val | Przy tych samych punktach budżetu; profil wydajności sam nie mierzy zbieżności |
| elapsed, GPU-hours, queue wait | Nie zastępować jednego drugim |

```text
T_run = T_start + T_compile + S * t_update + T_validation + T_checkpoint
C_total = C_decode + C_features + C_proxy + C_selector + C_tuning
          + suma(C_target_train + C_evaluation)
wall_time >= max(najdluzszy_bieg, suma_czasow_biegow / realna_rownoleglosc)
```

Ostatnia linia to idealizowane dolne ograniczenie dla zgodnych zasobów, nie prognoza kolejki. Dodać stage/I/O, nierównomierne czasy, awarie i oczekiwanie. GPU-hours H100 i A100 raportować osobno; nominalna suma nie oznacza równej pracy obliczeniowej.

**Przykład rachunkowy, nie benchmark Edena:** przy t_update=0,05 s, S=4000 i 120 s łącznego startu/walidacji/zapisu jeden bieg trwałby 320 s, czyli 5,33 min. Przy 0,20 s byłoby to 920 s, czyli 15,33 min. Te wartości służą tylko sprawdzeniu kalkulatora; aktualne czasy pozostają nieznane.

Warunek opłacalności współdzielonego selektora przy R treningach: `R*(C_full_train-C_subset_train) > C_selection`, jeśli różnica kosztu jest dodatnia, a porównywana jakość spełnia wcześniej ustalone kryterium. W torze równych kroków subset może **nie** przyspieszyć samego treningu, nawet jeśli zawiera mniej obrazów; korzyść może dotyczyć jakości lub anotacji. Raportować koszt pierwszego użycia i amortyzowany, a także wszystkie próby development.

## 8. Reguły naukowej interpretacji

- Główna metryka: COCO AP50:95 na wszystkich obrazach oceny, także bez detekcji. Uzupełnienia: AP50, AP75, AR, wynik per operacja, negatywy i podgrupy trudnych końcówek. F1 tylko z progiem ustalonym na val.
- Porównania sparować po tym samym splicie i blokach seedów. Oddzielić seed selekcji, proxy i treningu; przy A4 bez losowego proxy nie dopisywać mu pozornej zmienności. Te same subset IDs przenosić na inne modele.
- Przedziały niepewności liczyć po pacjentach/operacjach, nie tysiącach klatek. CI przy ustalonych modelach i SD między seedami opisują inne źródła niepewności. Przy bootstrapie COCO powielonym operacjom nadać nowe image/annotation IDs wraz z kopiami predykcji, jak w dokumencie 22.
- Pilot i wielokrotne próby na val są development. Końcowy test musi być zamrożony; jeśli historyczne testy były oglądane, jawnie to opisać i uwzględnić niezależny zbiór zewnętrzny. Nie odtwarza się niezależności testu zmianą nazwy folderu.
- Wczesny ranking sprawdzić na dłuższych krzywych kontrolnych. Jeżeli rankingi się odwracają, nie używać pierwszych epok jako wiarygodnego filtra. Sukces przy małym K nie dowodzi sukcesu przy pełnej puli.
- Zysk tylko na YOLOv8n uzasadnia wąską tezę dla tego modelu. Zysk na większych sieciach wymaga ich treningów. Ujemne transfery są częścią wyniku, nawet jeśli inne modele dają dużą poprawę.
- Historyczne DETR/YOLO służą do opisania poprzedniej metody. Checkpointy ep170 o niezgodnym pochodzeniu nie są inicjalizacją nowych kontroli. AP z innego splitu, preprocessingu i budżetu nie jest parą do nowego AP.

## 9. Kolejność wdrożenia

1. Naprawić blokery danych/ekstrakcji/ewaluacji i zweryfikować jeden adapter YOLOv8n end-to-end na CPU/GPU smoke.
2. Zaimplementować wspólny licznik aktualizacji, manifest i benchmark czasu; skalibrować jeden GPU bez wysyłania całej macierzy.
3. Dodać tani residual ridge i kontrolę persistence na cache’u; sprawdzić granice filmów, długości sekwencji, remisy i dokładne K.
4. Wykonać etap A, zapisać wszystkie wyniki i zdecydować o budżecie potwierdzenia zgodnie z wcześniej ustaloną regułą.
5. Etap B, potem zamrożony transfer C. Dopiero dalej duże modele i pełne ablacje potrzebne do wybranych tez publikacji.

**Ta sesja dostarcza dokumenty i zweryfikowany rachunek planu. Nie dostarcza jeszcze zmierzonego przyspieszenia, działającej nowej kolejki Slurm ani wyników skuteczności kuracji.** Dotychczasowy YAML wymaga osobnej migracji i nie powinien zostać uruchomiony pod etykietą tego planu.
