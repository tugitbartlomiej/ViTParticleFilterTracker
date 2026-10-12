# Od sztucznego latentu do obrazów i sterowania uczeniem

**Zapis sesji z 5 października 2026 r.**  
Projekt: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker`  
Podstawa: lokalne raporty, JSON-y wyników, manifesty i weryfikacje w `SmallViTAblation`. Statusy odczytano około **18:31–18:33 czasu Europe/Warsaw**. Dokument jest migawką stanu, a nie monitorem procesów.

Najważniejszy wynik sesji to przejście od ręcznej interwencji w aktywacjach do naturalnego odtworzenia części tego celu przez wyuczone wagi ViT. Na jednym zdjęciu zgodność J128 osiągnęła **74,251936 procent**, a osobna generacja SD z czystego szumu, prowadzona nowym modelem, osiągnęła **62,827646 procent**. To potwierdzone wyniki dopasowania do jednego sztucznego celu. Nie stanowią jeszcze dowodu poprawy detektora na nowych zdjęciach.

Druga ważna obserwacja dotyczy Langevina. Prognoza pojedynczego przyrostu latentu była użyteczna, lecz ekstrapolowanie stałej początkowej siły na sto aktualizacji nie opisało rzeczywistej trajektorii. Cztery dobrane paczki dały małe zmiany odległości do celu; wszystkie pogorszyły końcowe IoU i confidence sondy. Następnym zadaniem jest więc wybieranie kolejnej paczki po każdej rzeczywistej aktualizacji i ponownej ocenie stanu sieci.

## 1. Pytanie, które spina sesję

Użytkownik rozwija następujący pipeline:

1. Wybrać checkpoint i zmierzyć naturalny latent konkretnego obrazu.
2. Znaleźć korzystny sztuczny latent, który poprawia wynik detekcji tego obrazu.
3. Wygenerować różnorodne obrazy SD prowadzone w stronę tego celu.
4. Odszukać rzeczywiste obrazy treningowe bliskie naturalnym latentom tych generacji.
5. Ułożyć paczki po 32 obrazy z prawdziwymi etykietami.
6. Z pomocą modelu dynamiki Langevina zaplanować kolejność paczek.
7. Wykonać rzeczywiste aktualizacje wag i sprawdzić zarówno ruch latentu, jak i jakość detekcji.

Sesja dostarczyła artefaktów kilku kolejnych ogniw tego łańcucha oraz wskazała miejsca, w których intuicja wymaga korekty. W szczególności bliskość do sztucznego celu nie gwarantuje lepszej detekcji, a dobry pierwszy krok prognozy nie gwarantuje dobrego długiego horyzontu.

## 2. Co dokładnie mierzymy

Cząsteczka oznacza reprezentację jednego przestrzennego tokenu obrazu. W tym miejscu omawianej sieci jeden obraz daje **600 cząsteczek**, a każda ma **768 współrzędnych** w jednej wspólnej przestrzeni. Pełny układ zapisujemy jako tensor `600×768`. Liczba 600 wynika z tokenizacji obrazu, nie z rozmiaru paczki treningowej.

Makrowektory `e_j` są wspólnymi osiami tej przestrzeni. Mikrowektory konkretnej cząsteczki to składowe `z_j e_j`; plik modelu przechowuje współrzędne `z_j`. J128 oznacza wybrane 128 kolumn dla wszystkich 600 tokenów, czyli `600×128`, a nie 128 tokenów. Pozostałe 640 kolumn służy m.in. do kontroli zachowania pozostałej reprezentacji. Wymiar 768 dotyczy GELU ostatniej MLP; metadane architektury podają również `dim=192`, co opisuje inny poziom sieci. [S1, S10]

Wyniki mają kilka różnych skal:

| Pomiar | Znaczenie | Czego nie oznacza |
|---|---|---|
| MSE128 / MSE768 | Średni kwadrat błędu współrzędnych względem ustalonego celu | Accuracy detektora |
| Zgodność S128 / S768 | Znormalizowana odległość do celu, według wzoru poniżej | Odsetek poprawnych detekcji ani udział identycznych współrzędnych |
| Cosine | Podobieństwo kierunku wektorów | Zgodność ich wartości i norm |
| Confidence | Wynik pewności wybranej predykcji | AP, mAP ani gwarancja poprawnej ramki |
| IoU | Pokrycie ramki z konkretną etykietą GT | Generalizacja na zbiór obrazów |
| AP50 / AP50:.95 | Jakość rankingu detekcji w zadanym protokole | Jakość dopasowania latentu |
| Gain MSE | Procentowa redukcja błędu względem stanu początkowego danego biegu | S128/S768 |
| Skill prognozy przyrostu | Jakość przewidywania zmiany reprezentacji | Procent dotarcia do celu |

W treningu jednego zdjęcia i nowym SD obowiązuje:

```text
S128 = 100 × max(0, 1 − sqrt(MSE128 / mean(target128²)))
S768 = 100 × max(0, 1 − sqrt(MSE768 / mean(target768²)))
gain_MSE = 100 × (MSE_before − MSE_after) / MSE_before
```

Wynik 74,25 procent zgodności jest zatem czymś innym niż cosine 0,966364 i confidence 0,744268. Również poprawa MSE o 0,78 procent w Langevinie nie oznacza zgodności 78 procent. [S7, S10, S11, S12]

Checkpoint jest zapisanym stanem wag, często razem z optymalizatorem. Aktualizacja treningowa jest operacją zmieniającą ten stan. `k1000` identyfikuje stan po tysiącu źródłowych aktualizacji, nie tysiąc epok. Zapisanie kolejnego checkpointu samo nie uczy modelu.

## 3. Punkt odniesienia: pary oryginał–LaMa

Wcześniejsza analiza 400 par stanowiła podstawę interpretacyjną sesji. Jej raport pochodzi z **1 października**, więc nie należy przypisywać całej tej pracy dzisiejszemu przebiegowi. Każda para obejmowała oryginał z narzędziem oraz odpowiadającą mu edycję LaMa po usunięciu narzędzia. Zamrożony k1000 przetworzył 800 obrazów; zapisano pary wektorów 6729 tokenów przecinających maskę i 3200 kontrolnych tokenów tła. [S2]

Podział obejmował 240/75/85 par z 69/23/24 rozłącznych filmów. Osiem mieszanych kierunków SVD zachowało 60,53 procent ważonej energii różnic. SVD z klasyfikatorem liniowym osiągnęło testowe AUROC 0,9686 i zrównoważoną trafność 88,61 procent; regularizowana LDA w 768 współrzędnych odpowiednio 0,9794 i 95,28 procent. Energia różnic nie jest procentem wydobytej tożsamości.

Interwencja w podprzestrzeń obniżała confidence × IoU bardziej niż dopasowane kontrole losowe, także po wyrównaniu normy za fc2. Nie uzyskano jednak rozstrzygającego przedziału dla zmiany AP50. Sham LaMa na 24 obrazach osłabiło proste wyjaśnienie dowolną edycją, ale nie wykluczyło wszystkich artefaktów. Operacyjna „tożsamość” oznaczała obecność anotowanego narzędzia względem jego syntetycznego usunięcia, a nie rozróżnianie typów lub egzemplarzy instrumentów. Filmy odłożono do analizy reprezentacji, lecz obrazy pochodziły z train detektora. [S2]

## 4. Korzystny sztuczny cel i próby odwrócenia go przez SD

Rodzina `vit_latent128_jedno_zdjecie_90` wybrała obraz 29, `TR_0154_S1_P03/00001926`. Przy zamrożonych wagach k1000 zmieniono J128 w aktywacjach wszystkich 600 tokenów. Confidence wzrosło z 0,867869 do **0,931670**, a IoU z 0,856595 do **0,999339**. Pozostałe 640 kanałów zachowano, a zapis i ponowny forward zweryfikowano. Te same piksele bez interwencji nadal dawały wynik początkowy. [S1]

To pokazało, że głowa detektora potrafi wykorzystać takie aktywacje. Nie pokazało, że istnieje naturalny obraz produkujący je przy tych samych wagach. Cztery generacje `vit_sd_cztery_szumy` startowały z niezależnego szumu i miały dopasowane kontrole. Prowadzenie dało jedynie niewielką redukcję pełnego MSE względem kontroli. [S3]

| Seed | MSE768 prowadzone | MSE768 kontrolne | Redukcja względem kontroli | MSE128 prowadzone | Cosine768 |
|---|---:|---:|---:|---:|---:|
| 2026100511 | 14,319647 | 14,338458 | 0,131% | 85,424622 | 0,129833 |
| 2026100523 | 14,287478 | 14,332698 | 0,315% | 85,503624 | 0,141585 |
| 2026100537 | 14,279775 | 14,282085 | 0,016% | 85,404564 | 0,141318 |
| 2026100549 | 14,317536 | 14,338115 | 0,144% | 85,508049 | 0,134358 |

Osobna rodzina `vit_sd_zgodnosc90` ustaliła dla **oryginalnego zamrożonego checkpointu** konieczną dolną granicę MSE768 równą 6,260274259. Przy energii celu 14,545352022 próg S768=90 wymagał MSE nie większego niż 0,145453520. Zapisany certyfikat wykluczał ten próg i dawał górną granicę S768 równą 34,395359238 procent. Najlepszy oceniony obraz uzyskał zaledwie S768=1,187796 procent i MSE768=14,201865919. Granica nie jest dowodem osiągalności swojego maksimum. [S7]

**Po uczeniu norm2/fc1 zmieniły się wagi użyte do wyznaczenia granicy. Stary certyfikat nie ogranicza nowego checkpointu Lorentza.** Nowego wyniku nie wolno więc odrzucać na podstawie granicy starego modelu.

SD nie było wytrenowane jako dekoder tego latentu ViT. W tych próbach latent ViT dostarczał gradientowego celu podczas generacji. Takie prowadzenie nie gwarantuje zgodności, realizmu ani poprawnej detekcji; obrazy wymagają także oceny wizualnej.

## 5. Retrieval: cztery rzeczywiste paczki po 32 obrazy

`vit_sd_cztery_obrazy_paczki32` użyło **naturalnych latentów czterech obrazów SD** jako zapytań. Ranking liczył surowe MSE w `600×768`, z zamrożonym ViT i jednolitym przetwarzaniem RGB 480×320 INTER_AREA. Nie był to ranking bezpośrednio do sztucznego celu. [S4]

Przetworzono **88 469 z 310 257 dostępnych obrazów train**, w 23 rozpoczętych archiwach ZIP. To deterministyczny początek posortowanych archiwów i plików, nie losowa próba ani pełne przeszukanie. Cztery TOP32 są najbliższe wyłącznie w tym podzbiorze. Eksport potwierdził 32 różne obrazy w każdej paczce i **128 różnych obrazów łącznie**, bez przecięć między tymi czterema paczkami. Zachowano ranking, obrazy, latenty i pochodzenie archiwum/member. [S4]

Bliskość w reprezentacji nie gwarantuje podobieństwa wizualnego. Te wybory są materiałem do uczenia, a nie odłożonym testem generalizacji.

## 6. CAV: osobna gałąź diagnostyczna

**CAV nie był częścią opisanego eksperymentu Langevina.** Użytkownik skorygował mieszanie tych gałęzi w interfejsie. Kierunki konceptu, interwencje w aktywacje i siły paczek treningowych mają różne role; ich wyników nie łączymy w jedno doświadczenie.

Adaptowany TCAV wykazał lokalny wpływ kierunków na logit: TCAV po filmach wynosił 93,61 procent dla full768 i historycznej LDA oraz 90,35 procent dla J128. Empiryczne kontrole placebo i losowe nie potwierdziły swoistości konceptu; po korekcie Bonferroniego wszystkie cztery wskazane p wynosiły 1. Wysoki TCAV nie przełożył się na potwierdzoną poprawę jakości detekcji. [S5]

W `vit_cav_maksymalizacja_detekcji` dobierano **wspólny skalar alpha** na walidacji, następnie oceniono zablokowany test. AP50 baseline wynosiło 45,7589 procent, full768 46,6715 procent, historyczna LDA 46,6652 procent, a J128 43,3084 procent. Wszystkie przedziały zmian obejmowały zero; full768 dawało mały, niepewny przyrost, J128 gorszy wynik punktowy. Interwencje korzystały ze znanego ROI i nie musiały odpowiadać żadnemu rzeczywistemu obrazowi. [S6]

Szeroka próba na jednym zdjęciu objęła **817 punktów alpha/r**. Najlepsze q=confidence × IoU wzrosło tylko z 0,74341142 do 0,74373132, a IoU spadło z 0,85659355 do 0,84835923. Single-image AP50:.95 spadło w tym punkcie z 0,8 do 0,7; najlepsze AP w całej siatce nie przekroczyło baseline. Duże przesunięcia obu znaków pogarszały wynik. [S8]

Nowe zadanie Ohma, `vit_cav_pelny_latent_pary400`, dotyczy pełnego układu **`600×768` spłaszczonego do 460 800 współrzędnych**, na wszystkich 400 parach. To inne zadanie niż wcześniejszy CAV z uśrednionego ROI o 768 współrzędnych. Polecenie pochodzi z przekazania sesji; podczas przeglądu nie znaleziono katalogu wyników ani kodu tej rodziny w sprawdzonych lokalnych ścieżkach. **Start i wynik pozostają niepotwierdzone.**

## 7. Lorentz: naturalny latent przez uczenie wag

Kolejne trzy rodziny uczyły to samo zdjęcie, korzystając z niezmienionego sztucznego celu J128 i GT. Zmieniano wyłącznie cztery parametry ostatniego bloku: weight/bias norm2 oraz weight/bias fc1, razem 148 608 parametrów. Zachowano downstream i bufory BN. Kontynuacje przywracały stan Adama zamiast zaczynać od oryginału. [S9, S10]

| Stan modelu | Łączne nowe aktualizacje od oryginału w tej gałęzi | MSE128 | S128 | Cosine128 | Confidence | IoU |
|---|---:|---:|---:|---:|---:|---:|
| Oryginał k1000 | 0 | 85,9211807 | 0,3356007% | 0,1067383 | 0,8678695 | 0,8565947 |
| `vit_wyuczenie_latentu_jedno_zdjecie` | 1000 | 10,9162397 | 64,4756232% | 0,9432904 | 0,7717779 | 0,9999523 |
| `vit_wyuczenie_latentu_jedno_zdjecie_2000` | 2000 | 8,67640686 | 68,3291376% | 0,9510303 | 0,7594470 | 0,9998837 |
| `vit_wyuczenie_latentu_jedno_zdjecie_do85` | 6402 | 5,73468113 | **74,2519364%** | **0,9663640** | **0,7442681** | **0,9999465** |

Ostatnia rodzina wykonała 4402 aktualizacje kontynuacji od stanu 2000 i została zatrzymana na **jawne polecenie użytkownika przed osiągnięciem 85 procent**. Best i final oznaczają zabezpieczony stan 6402. Nie ma podstaw do zapisywania tej próby jako sukcesu progu 85 lub 90. Lorentz zakończył pracę; w tym dokumencie nie wznowiono treningu. [S10]

Świeży proces załadował zapisane wagi i wykonał zwykły forward z pikseli. Hook służył wyłącznie odczytowi i nie zmieniał detektora. Audyt state_dict potwierdził zmiany tylko dozwolonych parametrów. `READY_FOR_SD.json` utrwalił checkpoint, SHA256, target, pipeline i metryki. Wartość GPU peak=0 w końcowym eksporcie nie jest pomiarem zużycia GPU podczas zakończonego treningu. [S10]

To rzeczywiste dopasowanie przez wagi, ale nadal do jednego uprzednio wybranego obrazu i celu optymalizowanego z GT. Confidence maleje mimo dużej poprawy IoU i zgodności latentu. Nie wykonano tu niezależnego testu nowych obrazów.

## 8. Erdős: świeży wynik SD na nowym checkpointcie

Przy przekazaniu sesji zadanie `vit_sd_szum_nowy_checkpoint` było opisane jako ładowanie lub praca w toku. **Świeży odczyt plików zmienił ten status: `RESULT.json` i `PROGRESS.json` mają `complete`, a `VALIDATION.json` potwierdza niezależną weryfikację.** Start zapisano na 16:21:28 UTC; raportowany czas biegu to około 280 sekund. PID w historycznym LAUNCH dokumentuje start, a nie bieżącą żywotność procesu. [S11]

SD użyło zabezpieczonego modelu Lorentza, oryginalnego sztucznego celu i jednego czystego szumu Gaussa. Zdjęcie źródłowe nie było wejściem generacji. Raport podaje pusty tekst, zerowe maski, 80 kroków DDIM i 106 ograniczonych aktualizacji prowadzenia. Nie stosowano img2img, treningu wag ViT, podmiany aktywacji ani optymalizacji obrazu po DDIM. [S11]

| Pomiar wygenerowanego obrazu | Wynik |
|---|---:|
| S128 | **62,827645716%** |
| MSE128 | 11,952540912 |
| MSE640 | 0,056493965 |
| MSE768 | 2,039168456 |
| S768 | 62,557534870% |
| Cosine128 | 0,929508880 |
| Stosunek norm J128 | 0,882977708 |
| Confidence | 0,255850583 |

Świeży plain forward potwierdził S128=62,827648951 procent; maksymalna różnica eksportowanego i odtworzonego latentu wyniosła 0,000026703 w porównaniu float32 CPU/GPU. SHA checkpointu i celu były zgodne, a odczytujący hook pozostawił detektor bitowo bez zmian. [S11]

Próg 90 procent nie został osiągnięty. Jest to jedna generacja bez dodatkowego ramienia kontrolnego, więc nie szacuje przyczynowej korzyści prowadzenia. Bez GT dla wygenerowanego obrazu confidence nie jest accuracy; nie przypisujemy mu IoU ani AP. Podglądy etapów tego samego biegu nie są dodatkowymi niezależnymi próbkami.

## 9. Gibbs: zakończone porównanie czterech paczek i kontroli

`vit_langevin_cztery_paczki_sd` rozpoczęło każde ramię od **oryginalnego k1000 i autentycznego AdamaW**, a nie od wyuczonego modelu Lorentza. Sondą było niezmienione zdjęcie 29, celem oryginalne `after` z `LATENT_PELNY.npz`. Każda paczka miała 32 obrazy z rzeczywistym GT. Sto kroków oznaczało sto powtórzeń tej paczki, nie sto epok całego train. [S12]

Prognozy utrwalono i zahashowano przed rzeczywistym uczeniem. Porównano cztery dobrane paczki, random32 i nochange. W pięciu uczonych ramionach wykonano łącznie 500 aktualizacji i 16 000 ekspozycji treningowych obrazów. To liczba ekspozycji, nie różnych obrazów. Zmieniała się również głowa detektora. [S12]

| Paczka / kontrola | Prognoza redukcji MSE128 po 100 | Rzeczywista redukcja MSE128 | Rzeczywista redukcja MSE768 | Końcowe IoU sondy | Końcowe confidence |
|---|---:|---:|---:|---:|---:|
| 2026100511 | −9,2306% | 0,5222% | 0,2742% | 0,6016 | 0,5651 |
| 2026100523 | −14,7352% | 0,7818% | 0,4496% | 0,5737 | 0,5072 |
| 2026100537 | −13,1607% | 0,0481% | −0,2521% | 0,6579 | 0,7417 |
| 2026100549 | −17,1148% | 0,4438% | 0,0470% | 0,1725 | 0,3887 |
| random32 | −8,6383% | 0,3001% | 0,1621% | 0,8064 | 0,9158 |

Dodatni gain oznacza redukcję MSE; ujemny pogorszenie. Żadne ramię nie przekroczyło pierwszego progu redukcji błędu **5 procent** do kroku 100. Nie wyznaczono wiarygodnej liczby dalszych kroków do celu. Dwie paczki dały lepszą redukcję MSE768 niż random32, a pozostałe nie. Wszystkie cztery pogorszyły confidence i IoU względem oryginału; random32 pogorszyło IoU, choć zwiększyło confidence. [S12]

Skill przewidywania przyrostu pełnego latentu w pierwszym kroku wynosił około **88,8–98,6 procent** przy rzeczywistym BN w raportowanej rodzinie porównań. To jakość prognozy przyrostu, a nie zgodność współrzędnych z celem, cosine stanu ani detekcja. Prognoza kroku pierwszego poprawnie wskazała początkowe oddalanie czterech paczek. Jej utrzymanie ze stałą siłą przez sto kroków zawiodło. [S12]

Końcowa weryfikacja objęła 101 stanów każdej z pięciu trajektorii, końcowe checkpointy z optymalizatorami, baseline oraz seal prognozy. Odczyt końcowego naturalnego latentu odtwarzał zapisany ostatni latent z maksymalnym błędem 0. Zachowano także pierwszy błąd kodowania JSON sprzed treningu. [S12]

## 10. Co przekazano do dalszej pracy

| Osoba / gałąź | Stan według przekazania sesji | Stan potwierdzony lokalnie przy zapisie |
|---|---|---|
| Lorentz, uczenie jednego zdjęcia | Zakończone i zamknięte na polecenie użytkownika | Raport, fresh reload i READY_FOR_SD; final 6402 |
| Erdős, SD z nowego modelu | Wcześniej ładowanie / praca | **Zakończone i zweryfikowane** w świeżych plikach |
| Ohm, pełny CAV pary400 | Przydzielone nowe zadanie `600×768` | Brak lokalnych artefaktów tej rodziny; start niepotwierdzony |
| Gibbs, cztery powtarzane paczki | Zakończone | REPORT, RESULT i FINAL_VERIFICATION |
| Gibbs, planowanie kolejności | Przydzielone `vit_langevin_planowanie_paczek_sd` | Deklaracja WYNIKI.json i pliki modelu/równania; pusty katalog outputs, brak potwierdzenia procesu i wyników |

Następne zadanie Gibbsa wymaga pętli zamkniętej: z bieżących wag i optymalizatora oszacować siły czterech paczek, wybrać kolejną, wykonać rzeczywistą aktualizację, ponownie zmierzyć sondę i przeliczyć wybór. Punktem startowym ma być oryginalny k1000 z AdamW. Porównania powinny obejmować dobór losowy i kolejność cykliczną; należy zapisać wykonaną sekwencję oraz faktyczne przekroczenia progów lub ich brak. Lokalna deklaracja stawia pytanie o 100 i 500 aktualizacji, lecz deklaracja nie dowodzi wykonania tych budżetów. **W tej gałęzi nie ma CAV.** [S13]

Kolejność różnych paczek może zmieniać trajektorię, ponieważ po każdej aktualizacji zmieniają się wagi, momenty optymalizatora i stan używany przez następną paczkę. Sama permutacja przykładów wewnątrz tej samej paczki ze zwykle uśrednianą stratą zasadniczo nie tworzy nowej siły; możliwe różnice od losowości lub numeryki trzeba odróżnić od wyboru innej paczki.

Do rozstrzygnięcia pozostają:

- potwierdzenie startu i potem wyników planowania pętli zamkniętej;
- pełny CAV 460 800D na wszystkich 400 parach, z zachowaniem różnicy względem pooled ROI 768D;
- sprawdzenie, czy poprawa odległości latentu daje korzyść na niezależnym zbiorze detekcyjnym;
- ocena realizmu obrazów SD i powtórzenia z kontrolami oraz wieloma seedami;
- ewentualne poszerzenie retrieval poza deterministyczny podzbiór 88 469 obrazów;
- raportowanie kosztu wyboru paczki obok liczby wykonanych aktualizacji i trafności prognozy.

## 11. Pochodzenie i granice tego zapisu

Manifest wcześniejszego kopiowania potwierdza siedem rodzin przeniesionych z Grant do SmallViTAblation: 744 pliki, 2 407 732 556 bajtów, tryb kopiowania z pozostawieniem źródeł i weryfikacją SHA256. Historyczne ścieżki Grant w JSON-ach i migawkach kodu są śladem pochodzenia. Nie należy przepisywać ich w istniejących artefaktach tylko dla estetyki. [S14]

Kanoniczny checkpoint przekazany do nowego SD:

```text
SmallViTAblation/outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/MODEL_best.pt
SHA256: 841c55b4ce752863d1c25d7e89f0fca452714ee68b5693d2837e3c92f4ca2b2d
```

Niezmieniony oryginalny sztuczny cel:

```text
SmallViTAblation/outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz
Klucz: after
SHA256: 2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df
```

Najmocniejsze lokalne wnioski to: korzystna interwencja w latent jednego obrazu, potwierdzone naturalne dopasowanie tego obrazu przez wagi, zakończona generacja SD nowym modelem oraz negatywny wynik długiej ekstrapolacji stałej siły. Nie wykazano uogólnionej poprawy detektora, osiągnięcia 85/90 procent zgodności ani gwarantowanej drogi do celu przez paczki SD.

Podczas sporządzania tego dokumentu czytano artefakty i zapisano wyłącznie ten nowy plik sesji. Nie uruchamiano GPU, eksperymentów ani treningu, nie wykonywano commitów i nie zmieniano istniejących raportów. Nazwy agentów oraz ich przydziały pochodzą z przekazania użytkownika; lokalne pliki potwierdzają wskazane wyniki, a nie aktualną aktywność agentów. Nowe pliki innych procesów mogą zmienić stan po zapisaniu tej migawki.

## Źródła lokalne

- **S1:** [Sztuczny latent jednego zdjęcia: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_latent128_jedno_zdjecie_90/RAPORT.md), [RESULT.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_latent128_jedno_zdjecie_90/RESULT.json).
- **S2:** [400 par: WYNIKI.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/WYNIKI.md), [PODSUMOWANIE.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/PODSUMOWANIE.md).
- **S3:** [Cztery szumy SD: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_cztery_szumy/RAPORT.md).
- **S4:** [Cztery paczki TOP32: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_cztery_obrazy_paczki32/RAPORT.md), [COVERAGE.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_cztery_obrazy_paczki32/COVERAGE.json).
- **S5:** [TCAV i kontrole: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_tcav_tozsamosc_narzedzia/RAPORT.md).
- **S6:** [Wspólny alpha i detekcja: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_cav_maksymalizacja_detekcji/RAPORT.md).
- **S7:** [SD i granica starego checkpointu: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_zgodnosc90/RAPORT.md), [RESULT.json z certyfikatem](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_zgodnosc90/RESULT.json).
- **S8:** [817 przesunięć jednego zdjęcia: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_cav_jedno_zdjecie_duze_przesuniecia/RAPORT.md).
- **S9:** [Uczenie 1000: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_wyuczenie_latentu_jedno_zdjecie/RAPORT.md), [Kontynuacja 2000: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_wyuczenie_latentu_jedno_zdjecie_2000/RAPORT.md).
- **S10:** [Zatrzymanie w kroku 6402: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/RAPORT.md), [READY_FOR_SD.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/READY_FOR_SD.json).
- **S11:** [Nowy SD: RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_szum_nowy_checkpoint/RAPORT.md), [RESULT.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_szum_nowy_checkpoint/RESULT.json), [VALIDATION.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_szum_nowy_checkpoint/VALIDATION.json), [LAUNCH.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_szum_nowy_checkpoint/LAUNCH.json).
- **S12:** [Langevin, zakończone cztery paczki: REPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_langevin_cztery_paczki_sd/REPORT.md), [FIRST_STEP_VALIDATION.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_langevin_cztery_paczki_sd/FIRST_STEP_VALIDATION.json), [FINAL_VERIFICATION.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_langevin_cztery_paczki_sd/FINAL_VERIFICATION.json).
- **S13:** [Deklaracja nowego planowania: WYNIKI.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_langevin_planowanie_paczek_sd/WYNIKI.json).
- **S14:** [Manifest kopiowania i pochodzenia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/przeniesienie_latent_sd_20261005/MANIFEST.json).
