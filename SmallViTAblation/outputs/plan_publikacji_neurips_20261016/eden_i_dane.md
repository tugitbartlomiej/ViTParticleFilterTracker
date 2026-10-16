# Eden, dane i realistyczna kolejka do badania publikacyjnego

Audyt lokalnych plików: 16 października 2026. **Nie łączono się z Edenem, nie uruchamiano eksperymentów, nie wysyłano zadań.** Odczytano kod, manifesty, zapisane metadane i historyczne logi; prostymi operacjami odtworzono wybór walidacji oraz rozmiary plików. Nie otwierano sekretów. Ta notatka jest planem, nie potwierdzeniem bieżącej dostępności zasobów ani nowych wyników.

## Najważniejsza decyzja

Najpierw pilot pomiaru sześciu bloków, z osobnym rozkładem attention i MLP oraz pomiarem rzeczywistego wpływu na wynik. Następnie replikacja na niezależnych ziarnach i filmach. **Nie skalować od razu pełnych trajektorii wszystkich tokenów i macierzy attention: dysk, a nie sam trening, może być głównym kosztem.** Drugi zbiór i architektura dopiero po przejściu bramki metodologicznej na danych rozwojowych.

Istniejące trzy obrazy i s47 są zbiorem rozwojowym. Wyniki ruch–jakość nie dały mocnej, uniwersalnej sygnatury dobrej detekcji. Plan powinien testować przyrost wartości funkcjonalnej miar współdziałania względem norm/gradientów i kontroli losowych, a nie zakładać prawdziwość analogii pułapki optycznej.

## 1. Co faktycznie jest dostępne

### Dane i podział

1. `experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl`: **35 000 klatek, 117 różnych filmów, wszystkie oznaczone TRAIN**. Każda ma identyfikator filmu/klatki, nazwę elementu ZIP, GT, rozmiary źródłowe i hashe bajtów/pikseli. Dane Cataract-LMM Object Tracking, wejście rozciągane do 480×320; 600 tokenów na siatce 30×20.
2. `docs/Analiza_ponownej_publikacji/evidence_tracking_training_2026-09-14/split_manifest.json`: **117 grup train, 25 val, 25 test**, ziarno 42, jednostka `operation_group`. Audyt wykazał, że wszystkie 117 filmów manifestu 35k należy dokładnie do jego TRAIN. Pole `patient_disjointness_verified` ma wartość **false**. Możemy mówić o podziale według filmów/grup operacji, nie o zweryfikowanej niezależności pacjentów.
3. `experiments/maly_vit_od_zera/eden/wejscia/pretrain/valid/_annotations.coco.json`: pula **12 043** klatek walidacyjnych. Odtworzenie wyboru `default_rng(0).choice(...,3000,replace=False)` z `kod/dane.py` daje **3000 klatek z 25 filmów VAL**, wszystkie zgodne z powyższym splitem. Zbiór filmów TRAIN i tej walidacji jest rozłączny. Historyczny kod dodatkowo usuwa identyczne `frame_id`; w metadanych liczba usuniętych wynosi zero.
4. `wejscia/SONDY_100.jsonl`: 100 sond używanych do obserwacji s47. Dokumentacja wskazuje, że pochodzą z puli treningowej; nie traktować ich jako niezależnego testu generalizacji. Zestaw 1500 zawiera 200 klatek wspólnych z 35k według README eksperymentu.
5. Split w `../Eden/Checkpoints/Ablations/ablation/video_split.json` zawiera identyfikatory `train01`, `test24` itd. To inny manifest, nie automatycznie podział Cataract-LMM `TR_...`. Nie wolno zamieniać ich tylko na podstawie nazwy pliku.

**Przed testem publikacyjnym:** zamrozić manifest testowych filmów, sprawdzić pochodzenie i wcześniejsze użycie ich wyników w projekcie; samo słowo TEST nie dowodzi, że nie były już używane do wyboru metod. Zweryfikować duplikaty hashami i grupowanie operacji/pacjentów, jeśli metadane umożliwiają takie sprawdzenie. Nie przenosić filmu z pierwotnego TRAIN do testu przy ocenie już nauczonego s47. Gdy konieczny jest nowy podział, trzeba trenować model od zera na nowym TRAIN.

### Model i istniejące powtórzenia

- `kod/model.py`: MałyViT, 6 bloków pre-LN, residual **192D**, 3 głowy attention na blok, MLP **192→768→192**, GELU; konwolucyjna łodyga, stałe sin-cos 2D, gęsta głowa na każdy token: logit centrum + cztery odległości do krawędzi.
- Główna długa historia: **seed 47**, batch 32, AdamW lr 5e−4, wd 0,05, warmup 300, później stały lr, bf16, odbicie 0,5 i jasność/kontrast ±20%. 60 epok, 65 640 kroków. `trening35000.py` ma seed jako stałą 47; nie wystarczy zmienić nazwę joba, by uzyskać nowy seed.
- `trening_pokretla.py` przyjmuje `--ziarno`. Zapisane analizy `ANALIZA_lr_5e-4_rp_z48.json`, `...z49.json` oraz analogiczne warianty lr potwierdzają istniejące krótkie analizy 48/49. Nie oznacza to dostępności pełnych 60-epokowych historii ani kompletnego pomiaru attention+MLP sześciu bloków.
- Istnieją analizy `ANALIZA_warstwy_lr_5e-4_rp_w0.json` … `w5.json`; obserwator tych biegów dotyczył aktywacji GELU w rzucie RP128. To zasób pomocniczy, nie gotowe dane pełnego rozkładu attention i residual.
- Historyczny s47 wznowiono po 21. epoce. README jawnie zaznacza, że kolejka permutacji danych zaczyna się po wznowieniu od nowa. Kontynuacja nie jest bitowo identyczna z nieprzerwanym treningiem. Nowe porównania powinny używać wspólnego protokołu wznowień lub zapisywać kolejkę danych i cały stan RNG.
- Waga `k*.pt` przechowuje model i krok; nie jest pełnym stanem AdamW. Stany `stan_e*.pt` obejmują optymalizator, lecz historycznie pozostawiano tylko końcowe. Nie można odtworzyć aktualizacji optymalizatora z każdego checkpointu s47. Nowy pomiar rozdzielający gradient, AdamW i ruch między checkpointami wymaga odpowiedniej telemetrii od początku.

## 2. Eden: konfiguracja historyczna, nie bieżący stan klastra

Źródło ograniczeń: `../Eden/EDEN_CONSTRAINTS.md`, stan z **30 marca 2026**, oraz skrypty z października. Bez aktualnego odczytu klastra nie potwierdzono obecnych limitów, zajętości, uprawnień ani kwoty.

| Pozycja | Dowód w plikach | Znaczenie dla nowego planu |
|---|---|---|
| Konto Slurm | `transformers_vsc` | Historyczne konto; potwierdzić dostęp przed przyszłym startem |
| `hopper` | 8×H100, limit 5 dni w dokumencie | Główny referencyjny GPU; nie wymagać całego węzła dla małego ViT |
| `short` | DGX A100, limit 1 dnia | Alternatywa, czas zmierzyć osobno |
| `hopper-2` | 4×H200, limit 1 dnia | Używane w krótkich biegach pokręteł i CIFAR; stary komentarz „nie hopper-2” dotyczył długich wielodniowych biegów |
| `debug` | Dostęp zablokowany w instrukcji | Nie planować tej partycji |
| `bieg_eden.slurm` dla 35k | 1 GPU, 16 CPU, **128G RAM zamówione**, limit 10 h, `hopper,short` | To request, nie zmierzone peak RAM/VRAM i nie czas potrzebny nowemu eksperymentowi |
| `pokretla_pelny.slurm` | 1 GPU, 16 CPU, 120G, 8 h, hopper-2 | Obsługuje cleanup i domyślnie usuwa część surowych trajektorii po analizie; nie kopiować bez audytu retencji |
| Dane robocze | `/dev/shm/${USER}_vit35k_${SLURM_JOB_ID}`, starsza alternatywa `/raid` | Sprawdzić wolne RAM/scratch. Nie korzystać bezpośrednio z małego `/tmp` |
| Kwota historyczna | 200 GB i 200 000 plików; 107,9 GB po s47; inne notatki pokazują 188,4 GB 12 października | Stan obecny **nieznany**. Przed startem potrzebny odczyt kwoty bajtów i inode; niczego teraz nie kasować |

Ścieżki użyte w historycznym Slurm:

```text
EXP=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
ZIPY=/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations
ARCHIVE=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar
```

Źródła historyczne wskazują PyTorch 2.9.1+cu128 oraz timm 0.9.16 w lokalnej kopii `deps`. Nie zakładać, że środowisko nadal istnieje. W przyszłym preflight sprawdzić wersje i hashe kodu, manifestów, adnotacji oraz dostępność archiwum. Notatka o `--export` sbatch w README pokręteł opisuje historyczne anulowanie przez administratora; najpierw zweryfikować przyjęty sposób przekazywania konfiguracji, nie zgadywać aktualnej polityki.

Stary dokument proponuje kasowanie wszystkich checkpointów poza najlepszym. **Nie stosować automatycznie takiej retencji w badaniu trajektorii:** checkpointy są materiałem badawczym. Zamrozić listę przechowywanych kroków, eksportować z sumami kontrolnymi i uzgadniać ewentualne usuwanie osobno. Obecny audyt nie usuwa niczego.

## 3. Czasy i pojemność: pomiary versus szacunki

### Zapisane pomiary

| Bieg / pomiar | Czas / rozmiar | Źródło i ograniczenie |
|---|---|---|
| Smoke 512 klatek, 2 epoki | 82 s całego joba H100 | STATUS.md, nie benchmark sześciu bloków |
| s47, pierwsze 21 epok | `czas_s=728` w metadata; 13 min 25 s całego joba w STATUS | Metadane i scheduler mają różne zakresy czasu |
| Kontynuacja epok 22–60 | `czas_s=1374` | Osobny job, wznowienie kolejki danych |
| Prędkość kroków od 1000 w LOG.json | mediana **29,46 ms**, percentyle 10/90: **26,12/39,87 ms** | Jeden historyczny H100, czas raportowany w logu; nie latency nowych interwencji |
| 21 epok z gęstym obserwatorem 100 sond | **1222 s**, 4268 stanów, 8 sond pełnego latentu | metadata `wyniki/gesty`; jedna warstwa, inny zakres pomiarów |
| Pełna długa historia wag lokalnie | **806 plików, 9 454 899 870 B**, pojedynczy ok. 11,73 MB | Rozmiar aktualnych lokalnych plików; około 8,81 GiB |
| Surowe RGB 35k przy 320×480×3 uint8 | **15,02 GiB** | Wyliczenie samej tablicy; bez walidacji, celów, kopii, scratch i frameworka |

Suma historycznych `czas_s` dla 60 epok wynosi 2102 s, około 35 minut. Pięć identycznych lekkich przebiegów dawałoby arytmetycznie około 2,92 GPU-godziny. **To wyłącznie orientacja dla bazowego treningu, nie budżet całego badania.** Do nowego czasu należy doliczyć odczyt danych, checkpointy, sześć bloków, gradienty/JVP i liczne interwencje. Nie znaleziono w sprawdzonych plikach peak GPU memory ani Slurm MaxRSS tego biegu, więc nie podaję zmyślonego zużycia pamięci.

### Dlaczego nie zapisywać wszystkiego

Niekompresowane oszacowania, fp16, 647 checkpointów i 100 obrazów na seed:

- 12 tablic residual po 600×192 (np. dwa punkty pomiaru na blok) to **166,6 GiB**. Przy 13 stanach koszt rośnie proporcjonalnie.
- Wszystkie macierze attention 6 bloków × 3 głowy × 600×600 to **780,9 GiB**. Nie obejmuje gradientów ani dodatkowych kopii.
- Pięć pełnych historii wag o rozmiarze historycznego s47 to około **47,3 GB** dziesiętnie; plus kod, stany AdamW, dane, logi i telemetria.
- Mały audyt dokładności: 25 checkpointów × 12 obrazów × 96 wybranych tokenów × 12 residuali × 192 × fp32 to około **0,247 GiB** na seed.

Zalecenie: zapisywać checkpointy oraz metryki/wybrane próbki, agregować pełne tokeny strumieniowo, odtwarzać attention na żądanie. Zachować mały materiał fp32 do audytu dokładności, nie tylko RP/PCA. Wydatki dyskowe liczyć formułą `liczba stanów × obrazów × tokenów × wymiar × punkty pomiaru × bajty`. Bufory GPU i RAM są dodatkowe.

Przed przyjęciem 5 seedów pełnej historii wag potrzebny jawny budżet retencji: samo 47,3 GB to dolne ograniczenie. Roboczy limit 80 GB dla nowych artefaktów byłby rozsądną rezerwą planistyczną przy strumieniowaniu, ale musi zostać zweryfikowany pilotem i wolną kwotą — nie jest stwierdzeniem, że obecnie tyle miejsca istnieje. Nie uruchamiać pięciu jobów naraz przy nieznanej kwocie.

## 4. Kolejka z bramkami

Poniższe pozycje są planem przyszłych zadań. Nic nie zostało wysłane.

### Q0 — audyt danych, implementacji i zasobów

Zamrozić 117/25/25 albo udokumentować nowy split przed uczeniem. Podzielić istniejące 25 filmów VAL na grupy rozwojowe i kalibracyjne, np. 12/13 z zamrożonym ziarnem i kontrolą pokrycia narzędzi; dopiero po sprawdzeniu liczności przyjąć te proporcje. Test 25 filmów zostaje zamknięty do końca, jeśli audyt wcześniejszego użycia to uzasadnia. W testach zachować identyfikator filmu/operacji, nie traktować wielu klatek jednej operacji jako niezależnych replik.

Ustalić reprezentatywną próbkę klatek po czasie i kategorii, np. 4–8 klatek na film na pierwszym etapie; wykrywać duplikaty/niemal identyczne sąsiednie klatki. Zamrozić maski bbox/tła, przedziały czasu i metryki przed wynikami. Decyzje o progach i wyborze metod wyłącznie na development/calibration.

Przyszły preflight klastra: bieżące partycje i ich limity, stan kwoty i inode, wolne scratch/RAM, wersje środowiska, liczba dostępnych GPU, dostępność danych. Logować te dane bez sekretów. Bez preflight nie obiecywać daty ukończenia.

### Q1 — pilot techniczny na istniejącym s47, 1 GPU

12 obrazów z różnych filmów rozwojowych, 25 ustalonych checkpointów obejmujących początek, środek i koniec treningu; 96 tokenów warstwowanych bbox/krawędź/dalekie tło do dokładnego zapisu, agregaty na wszystkich 600. W każdym z 6 bloków osobno mierzyć attention i MLP oraz residual przed/po.

Sprawdzić dokładne odtworzenie forwarda, zgodność sum wkładów, biasy i LayerNorm, kwantyzację fp16 versus fp32, stabilność przy perturbacji bliskiej zeru. **Interwencję wcześniejszego bloku propagować przez cały pozostały model**; lokalny Jacobian ostatniej głowy nie wystarcza dla bloków 1–5. Rozróżniać zmianę ramki stałego tokenu, zmianę zwycięzcy i końcową metrykę detekcji. Attention należy rozkładać zgodnie z wybraną jednostką (głowa/krawędź/wkład value), nie utożsamiać jej automatycznie z neuronem MLP.

Benchmark mierzy oddzielnie: sekundy na checkpoint, obraz, 6 bloków; koszt samego forwarda; koszt gradientów/JVP; koszt jednej oraz grupowych interwencji z kontrolami; GPU peak allocated/reserved, MaxRSS, ilość zapisanych bajtów, czas odczytu i zapisu. Krótkie rozgrzanie oddzielić od pomiaru. To jedyna podstawa kosztorysu właściwego badania.

**Bramka Q1:** zgodność algebraiczna i forwarda w zadanej tolerancji, brak leakage, brak naruszenia splitu, stabilność numeryczna i koszt mieszczący się w przyjętym budżecie. Przy błędzie poprawić pilot, nie rozszerzać liczby seedów.

### Q2 — replikacja trzech seedów, wszystkie 6 bloków, attention + MLP

Docelowo seed 47,48,49 przy identycznym przepisie i zbiorze. Historyczny s47 jest rozwojowy; dla formalnego, nowego protokołu porównywalności należy odtworzyć także 47 lub wyraźnie oddzielić go od nowych replik. Krótkie z48/z49 pokręteł nie zastępują tej serii. Rejestrować osobno seed modelu, tasowania/augmentacji, próbkowania sond i rzutu; seed rzutu nie może udawać niezależnego treningu.

Pełny trening 60 epok jak historyczny punkt odniesienia; nie wprowadzać jednocześnie nowego harmonogramu lr i nowego wskaźnika bez kontrolnego ramienia. Zachować gęstą oś czasu dla podzbioru sond, a skan drogich interwencji wykonać np. w 25 zamrożonych checkpointach. Logować normę gradientu **przed i po clipie**, lr, oba momenty AdamW w ustalonych stanach, batch/RNG/permutację i aktualizację wag. To potrzebne do wyjaśnienia ruchu w czasie uczenia.

W testach funkcjonalnych porównywać miarę współdziałania z największą normą wkładu, wrażliwością/gradientem, losowaniem dopasowanym rozmiarem grupy i normą perturbacji oraz kontrolą niezwiązaną z treścią. Hierarchia jednostek analizy: seed → film → klatka → token; wyniki per-seed/per-film i przedziały grupowane, nie miliony pseudo-niezależnych tokenów.

**Bramka Q2 na development/calibration:** z góry zdefiniowana, praktycznie istotna przewaga nad silną kontrolą funkcjonalną, powtarzalny znak w seedach i filmach, zachowanie wyniku po kontroli czasu/normy/pozycji tokenu, brak dominacji pojedynczego przykładu przełączenia ramki. Próg praktycznej przewagi ustalić przed tą serią na podstawie jednostek zadania i rozdzielczości benchmarku, nie dobrać po otrzymaniu wyników. Jeśli brak przewagi, zawęzić twierdzenie do narzędzia analitycznego i wyniku negatywnego; nie inwestować automatycznie w drugi zbiór.

### Q3 — domknięcie replikacji i niezależny test

Po Q2 dodać seed 50/51, aby uzyskać pięć treningów, o ile budżet pozwala. Jeżeli projekt kończy się na trzech, jasno ujawnić tę liczność. Najpierw zablokować kod, progi i analizę, potem otworzyć 25 testowych filmów. Na początku 4–8 predefiniowanych klatek na film do drogiej analizy, a standardową jakość detekcji ocenić na szerszym zbiorze testowym; nie wybierać jedynie obrazów o wysokim IoU.

Ocena jednocześnie wysokiej jakości powinna obejmować przypadki wysoka pewność/zła ramka oraz dobra ramka/niska pewność. Poprzedni pilot pokazał, że ta rozbieżność jest realna. Osobno zbadać test na nowych filmach i przeniesienie w czasie uczenia; nie mieszać obu osi w jednej losowej walidacji klatek.

### Q4 — drugi zbiór i druga architektura dopiero po bramce

W projekcie są zasoby CIFAR-10 i DeiT-Tiny, więc nie trzeba zaczynać ich przygotowania od zera. `experiments/deit_cifar10_20261013/README.md` opisuje MałyViT i DeiT na CIFAR, ale to zadanie klasyfikacji: może potwierdzić ogólność współdziałania, **nie generalizację wpływu na bounding box**. DeiT ma 12 bloków i 65 tokenów (64 patche+CLS); należy jawnie dostosować jednostkę odczytu i koszty.

Dla wniosku o ramkach potrzebny drugi zbiór detekcyjny z niezależnymi obiektami/scenami oraz porównywalną głową, wybrany dopiero po przeglądzie dostępności/licencji/etykiet. Aby rozdzielić architekturę od zbioru, stosować most: MałyViT na zbiorze A i B, druga architektura na B, a najlepiej także na A. Sama zmiana zbioru i architektury naraz jest niejednoznaczna.

Przed wykorzystaniem istniejącego CIFAR sprawdzić leakage wyboru modelu: jego README wskazuje logowanie test accuracy przez cały trening. Do nowego potwierdzenia potrzebny oddzielny validation split z TRAIN i zablokowany końcowy test. Nie traktować historycznego wyboru najlepszego checkpointu na test accuracy jako czystej generalizacji.

## 5. Obowiązkowa kontrola leakage i kontekst ruch–jakość

W `outputs/wspoldzialanie_neuronow_20261015/ruch_a_jakosc/RAPORT.md` zbadano 647 checkpointów trzech obrazów. Dobre stany: 552/573/0. Po dopasowaniu czasowym różnica prędkości centrum to −1,93%/+0,05%. `vib_bbox/vib_tło` spadało o medianę 14,6%/13,9%, ale tylko w około 59–61% par. Nie jest to wystarczający wskaźnik jakości. Na trzecim obrazie η korelowało z pewnością po odjęciu trendu (r≈0,353), mimo braku stanów jednocześnie IoU≥0,8 i pewność≥0,8. Należy zachować ten wynik negatywny w planie publikacji.

W aplikacji `wezly` są średnią w oknie **±300 kroków**, a projekcja jest dopasowywana z całej historii. To wprost korzysta z przyszłości. `vib` mierzy odległość od tego wygładzenia. Takie dane mogą służyć retrospektywnej ilustracji, ale nie prognozie jakości z wczesnego treningu.

Nowy benchmark predykcyjny musi używać wyłącznie historii do kroku t: średnie jednostronne, PCA/bazy/rzuty i normalizacje ustalone na dozwolonej części treningowej, brak przyszłych checkpointów w imputacji i wyborze cech. Jawnie oddzielić losowy rzut niezależny od danych od PCA uczonego na całej trajektorii. Dodać odstęp czasowy między dopasowaniem a oceną co najmniej tak duży jak użyte okno/pamięć modelu. Podział po filmach dotyczy także kalibracji wskaźnika; dane tych samych filmów w kolejnych checkpointach nie są niezależnym testem.

Nawet korelacja po spline czasu jest analizą retrospektywną, nie zweryfikowanym predyktorem. Dla nowego celu publikacyjnego trzeba odróżnić: opis geometryczny, interwencję przyczynową w zamrożonym modelu oraz przewidywanie jakości na nowych filmach. Każdy z tych celów wymaga innego dowodu.

## 6. Źródła lokalne i ścieżki wynikowe

Główne dowody:

- `../Eden/EDEN_CONSTRAINTS.md` — historyczne limity i zasady.
- `experiments/maly_vit_od_zera_35000_21epok_20261009/{bieg_eden.slurm,README.md,STATUS.md,kod/model.py,kod/dane.py,kod/trening35000.py}`.
- `assets/MalyViT_35000_od_zera_s47/dane/{metadata.json,metadata_do_kroku_22974.json,LOG.json}` oraz `wagi/` — czasy, konfiguracja i realny rozmiar wag.
- `experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/dane/metadata.json` — koszt i zakres starego obserwatora.
- `docs/Analiza_ponownej_publikacji/evidence_tracking_training_2026-09-14/split_manifest.json` i lokalne adnotacje walidacji wskazane wyżej.
- `experiments/maly_vit_szklo_pokretla_20261010/{README.md,FALA5_RECENZENCI_INSTRUKCJE.md,pokretla_pelny.slurm}` oraz `outputs/porownanie_szklo_20261010/eden/wyniki/pokretla/ANALIZA_*.json` — krótkie multi-seed i skan MLP.
- `experiments/deit_cifar10_20261013/{README.md,kod/deit_cifar.slurm}` — zasoby drugiego modelu/zbioru, nie automatyczna replikacja obecnego wyniku detekcyjnego.

Nowy kod w przyszłości: `SmallViTAblation/experiments/`, artefakty: `SmallViTAblation/outputs/`, z osobnymi katalogami i hashami. Ta notatka jest jedynym nowym plikiem zapisanym w ramach tego audytu. Nie modyfikowano aplikacji, danych, skryptów Slurm ani starych raportów.
