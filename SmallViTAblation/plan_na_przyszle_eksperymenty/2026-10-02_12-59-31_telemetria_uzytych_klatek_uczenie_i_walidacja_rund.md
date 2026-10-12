# Telemetria rzeczywistego użycia klatek, uczenie rozszerzonej puli i walidacja rund

Utworzono: 2026-10-02T12:59:31+02:00 (Europe/Warsaw). Status: plan do implementacji; bez zmiany kodu treningu.

Priorytet użytkownika: sprawdzalna chronologia dokładnych frame-ID, kiedy dodano każdą nową klatkę i kiedy naprawdę weszła do uczenia. Dodanie do puli nie jest użyciem w batchu.

## 8. Krok szósty: uczenie na rozszerzonej puli prawdziwych klatek i checkpoint

Po rankingu dokładamy wybrane **prawdziwe obrazy train wraz z ich prawdziwymi anotacjami** do dotychczasowej puli. Obrazy SD są zapytaniami, nie nowymi przykładami uczącymi. Stare obrazy i ich anotacje zachowujemy; nowe mają unikalne frame-ID i sprawdzony split. Nie dodajemy val/test. Liczba nowych klatek pozostaje adaptacyjna 0–100 zgodnie z krokiem 5; brak spełniających próg oznacza 0, bez dopełniania.

Przed uczeniem rundy zapisujemy niezmienny snapshot puli: wszystkie frame-ID, pochodzenie film/klatka, źródłowy hash obrazu i anotacji, split, oznaczenie old/new oraz hash całego manifestu. Zapisujemy zdarzenie dodania każdego obrazu: run/arm-ID, round, epoch lub pozycję między epokami, global_optimizer_step, timestamp oraz selection/query-ID, score i metodę wyboru. Zachowujemy źródłowe współrzędne bbox/maski i ich konwencję; transformacje augmentacji mają osobny zapis.

**Dodany do puli nie znaczy użyty do uczenia.** Loader/sampler ma logować image-ID rzeczywiście podane w każdym batchu, ich kolejność i powtórzenia, przypisane do rundy, epoki i globalnego kroku. Dla gradient accumulation rozróżniamy microbatch-ID i optimizer-step; krok optymalizatora liczymy przy faktycznej aktualizacji. Dzięki temu można sprawdzić, która nowa klatka po raz pierwszy trafiła do batcha, ile razy została pokazana i przy którym checkpointcie jej wpływ mógł już wystąpić.

Uczenie prowadzi do checkpointu oznaczonego run/arm/round-ID i global_optimizer_step. Checkpoint zawiera wagi, bufory BatchNorm, identyfikator/hash snapshotu trainpool, hash konfiguracji i mapę kroków oraz, jeśli umożliwia to runtime, stan optymalizatora, schedulera i AMP/scalera. Replay wymaga także stanów RNG i informacji o samplerze; sam seed i lista dodanych klatek nie wystarczają do dokładnego odtworzenia. Wymagania telemetrii z kroku 7 obowiązują **już podczas kroku 6**, nie dopiero po treningu.

## 9. Krok siódmy: walidacja, telemetria pochodzenia i kontrole porównawcze

### 9.1. Co zapisywać, żeby odtworzyć chronologię klatek

Planowany zapis obejmuje snapshoty trainpool oraz append-only zdarzenia JSONL. Minimalny schemat oddziela `selection_events.jsonl` od `batch_samples.jsonl` i `checkpoint_events.jsonl`:

- **Selection/add:** run/arm-ID, source dataset/film/frame-ID, ścieżka i sourcehash obrazu/anotacji, split, old/new, selected_round, added_at, epoch/global_optimizer_step w chwili dodania, SD query-ID, metoda i wersja preprocessingu, cosine/score/prog, rank oraz status accepted/rejected/duplicate/used. Zapis obecności w puli nie jest zapisem ekspozycji na gradient.
- **Actual batch:** round, epoch, global_optimizer_step, microbatch/sample-order, rzeczywiste image-ID z powtórzeniami, seed samplera i dane augmentacji. Logować seed oraz parametry losowych cropów, resize, flipów i przekształceń koloru/geometrycznych, w miarę dostępności runtime; zachować powiązanie transformacji obrazu i anotacji. Sam frame-ID nie wyjaśnia, co rzeczywiście zobaczył model.
- **Checkpoint/replay:** checkpoint-ID/path/hash, run/arm/round, global_optimizer_step, zakres zakończonych batchów i hash puli, stan RNG Python/NumPy/Torch CPU/CUDA i workerów/samplera, optimizer/scheduler/scaler i konfiguracja środowiska, o ile są zapisywane. Niezapisane elementy jawnie oznaczyć; nie obiecywać bitowego replay bez kontroli deterministyczności.

Indeks kroków pozwala odtworzyć film i frame w chronologii źródła oraz chronologię wykorzystania w uczeniu. Nie utożsamiamy porządku klatek filmu z losową kolejnością samplerów. Sumujemy osobno prezentacje old/new, liczbę unikalnych klatek i wszystkie powtórzenia, per runda/epoka/krok i narastająco. Regułę `used_train_pool` aktualizujemy po rzeczywistym użyciu, a klatki już zarezerwowane/dodane oznaczamy dodatkowo, aby kolejna runda nie dodała ich ponownie.

Kontrole integralności sprawdzają: brak val/test w pool i batchach, każdy sampled-ID należy do snapshotu aktywnego przed batch-em, source/anotacje są zgodne, nowe klatki nie występują przed added_at, brak podwójnego dodania, wszystkie checkpointy mają jednoznaczny krok i powiązanie z puli. Ważna jest właśnie odpowiedź „która klatka, kiedy dodana, kiedy rzeczywiście użyta i ile razy”, nie tylko liczba wybranych obrazów.

### 9.2. Walidacja przed i po rundzie

Na stałym zbiorze validation, z tym samym preprocessingu i protokołem, oceniamy model **przed i po każdej rundzie**. Zapisujemy loss oraz mAP i jego ustalone warianty/progi IoU, liczbę przykładów, konfigurację decode/NMS i metryki per obraz oraz agregaty. Checkpoint-ID i globalny krok są częścią wyniku. Nie przypisujemy zmiany metryki do nowych danych bez kontroli ilości wykonanych aktualizacji.

Oddzielny końcowy test pozostaje niewykorzystywany do doboru klatek, progów, hiperparametrów i wyboru modelu. Obrazy validation/test nigdy nie są kandydatami retrieval ani danymi uczącymi. Wszystkie decyzje adaptacyjne i momenty odczytu validation trzeba logować, aby ograniczenie końcowego testu było sprawdzalne.

### 9.3. Jakie ramiona porównujemy

Plan obejmuje pipeline prognoza→SD→retrieval oraz kontrole **random**, **sequential** i **direct real retrieval**. Ostatnia kontrola dobiera prawdziwe klatki bez zapytania SD, według jawnie ustalonego prawdziwego zapytania/reprezentacji i tej samej eligibility; konkretną regułę trzeba ustalić przed wykonaniem, a nie dopisać po obejrzeniu wyników. Pozwala odróżnić korzyść generacji SD od samego wyszukiwania podobnych realnych danych.

Ramiona mają wspólny checkpoint startowy, seed lub zestaw sparowanych seedów, warmup, bazową pulę, protokół validation i budżet aktualizacji. **Każde ramię jest osobnym runem z własnymi checkpointami, puli i logami.** Nie dodajemy po kolei danych różnych ramion do jednej sieci i nie przedstawiamy późniejszych wyników jako kontrolowanego wpływu każdej metody.

Przy adaptacyjnym doborze 0–100 kontrolom należy dopasować rzeczywistą liczbę nowych klatek zaakceptowanych w każdej rundzie, bez wymuszania stałych 134. Gdy pipeline dodaje 0, logujemy 0 także w dopasowanej kontroli doboru; zasady wykonania lub pominięcia aktualizacji rundy muszą być identyczne i ustalone. Rejestrujemy faktyczne prezentacje starych/nowych danych, liczbę aktualizacji i koszt. Gdy nie można wyrównać obu budżetów, pokazujemy krzywe metryk względem liczby przykładów oraz aktualizacji, z jawną różnicą warunków.

To **plan dokumentacyjny**, nie deklaracja zaimplementowanej telemetrii, wykonanych nowych rund czy zatwierdzonych wyników NN. Obowiązująca wersja kroku 5 pozostaje: pełnoobrazowy DINO, cosine>0,85, najwyżej 100, brak quota; FAISS i pozostałe rozszerzenia są opcjonalne.


## Powiązane dokumenty

- [Pipeline kroków1–7](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md).
- [Plan generacji i adaptacyjnego retrieval](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-49-00_sd_z_czystego_szumu_prowadzenie_prognozowanym_latentem_768.md).

## Doprecyzowanie testu zwykłej sieci i pipeline — 2026-10-02 13:02:43 (Europe/Warsaw)

Krok 7 jest porównaniem jakości uczenia; telemetria musi działać już podczas kroku 6. Najnowszy konkretny protokół ramion A–F, etap minimalny A/B/F, dodatkowe D/E/C, mAP learning-AUC, seed-run CI i koszty opisano w [osobnym datowanym planie](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_13-02-43_test_standardowego_vit_i_pipeline_doboru_klatek.md). 10 seedów jest planem, nie wykonanymi replikami. Nie uruchomiono nowego treningu w tej aktualizacji.
