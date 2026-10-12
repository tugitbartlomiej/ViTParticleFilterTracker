# Analiza treningu SD z pełnym latentem ViT K8000

## Metadane i zakres

- Zapis sesji: 2026-10-07 00:25:23 +02:00, czas lokalny Europe/Warsaw.
- Projekt: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker`.
- Cel użytkownika: „przeanalizuj ten trening i powiedz czy dobrze poszło. Jeśli masz pomysł to powiedz jak to ulepszyć”, następnie „potem zapisz sesję”.
- Status: analiza zakończona. Zalecenia pozostają propozycjami; nie wdrożono ich i nie uruchomiono kolejnego treningu.
- Dokument zapisuje aktualną ocenę eksperymentu oraz kontekst potrzebny do jego kontynuacji. Nie jest dosłownym eksportem wszystkich wypowiedzi ani bieżącym monitorem procesów.
- Wcześniejszy etap badań opisano w [sesji z 5 października](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/.sessions/2026-10-05-latent-vit-sd-langevin.md). Zachowano ten dokument bez zmian.

## Kontekst badania

Użytkownik chce przekładać pełny latent detektora ViT na obraz, następnie wykorzystywać takie obrazy lub ich cechy do doboru prawdziwych danych treningowych. Wcześniejsze prace obejmowały sztuczne modyfikacje J128, generację sterowaną gradientem, CAV/TCAV, translator ViT–DINO oraz dobór paczek i prognozy Langevina. Wyników tych etapów nie należy utożsamiać z wynikiem nowego generatora.

Oceniany tutaj eksperyment uczy generator bezpośrednio na naturalnych parach klatka filmu–latent ViT. Użytkownik zażądał losowej inicjalizacji VAE, UNet i adaptera; jedynie zamrożony ViT K8000 był wcześniej wytrenowany. Nie jest to dostrajanie starszego, wytrenowanego SD.

Pełny latent w wybranym miejscu sieci to tensor 600 × 768. J128 oznacza wybrane kanały, a nie 128 tokenów. Obecny generator otrzymuje pełny tensor, bez ograniczenia wejścia do J128.

## Lokalizacje i pochodzenie

- SSH: `eden-cluster`.
- Root eksperymentu na Edenie: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy`.
- Kod lokalny: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_sd_od_zera_k8000_filmy_20261006`.
- Wyniki lokalne: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006`.
- Kopia rzeczywiście użytego kodu: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/kod`.
- Najlepszy model: [best_sd.pt](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/trening/best_sd.pt).
- SHA256 najlepszego SD: `6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7`.
- Rozmiar najlepszego SD: 546 367 845 bajtów.
- SHA256 teachera ViT K8000: `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`.
- SHA256 manifestu: `5be5b519e318a35120d2e3e9107e506bed7890245ba4664b2c24dfbb206f8db7`.

Hash i świeże wczytanie modelu były potwierdzone podczas wcześniejszego testu lokalnego. W bieżącej analizie odczytano raport tego testu; nie wykonywano ponownie generacji.

## Co sprawdzono

1. Raport świeżego lokalnego wczytania i generowania z najlepszego checkpointa.
2. Końcowy raport treningu i historię walidacji.
3. Provenance ze zmianą próbkowania klatek.
4. Kod samplera danych, generowania walidacyjnego, funkcji straty i early stopping.
5. Wyniki kontroli z latentem innego filmu, przy tym samym szumie.

Nie zmieniano wag, optymalizatora, programu, prezentacji ani danych. Nie uruchamiano treningu, indeksowania ani generowania nowych obrazów.

## Potwierdzone wyniki

| Pomiar | Wynik | Interpretacja |
|---|---:|---|
| Rekonstrukcja VAE, średnia po filmach | PSNR 30,4499 dB; SSIM 0,94456 | Dobra rekonstrukcja całego obrazu według tych miar |
| Najlepszy SD, krok 7500 | względny MSE 0,535459; cosine 0,695238 | Najlepszy zapisany wynik walidacji generatora |
| Ostatni SD, krok 19000 | względny MSE 0,581442; cosine 0,686551 | Brak poprawy względem najlepszego modelu |
| Cel jakości SD | względny MSE ≤ 0,10; cosine ≥ 0,90 | Nieosiągnięty |
| Właściwy warunek lepszy niż inny film, najlepszy SD | 19 z 25 przypadków | Warunkowanie działa, ale nierównomiernie |
| Średni względny MSE dla niewłaściwego warunku, krok 7500 | 0,625319 | Właściwy warunek obniżał średni błąd o 14,37% |

Względny MSE jest liczony jako `mean((actual-target)^2) / mean(target^2)`. Nie jest procentową zgodnością. Cosine opisuje podobieństwo kierunku, a nie zgodność wartości. PSNR/SSIM w tej tabeli dotyczą rekonstrukcji VAE, nie obrazów generowanych przez SD od szumu.

Trening SD zakończył się plateau, z `quality=false`. Samo plateau nie dowodzi przeuczenia ani matematycznej niemożliwości dalszej poprawy. Ostatni produkcyjny job 1823430 został wcześniej potwierdzony jako COMPLETED 0:0; tej sesji nie użyto do ponownego audytu kolejki.

## Lokalny test najlepszego modelu

Źródło: [TEST_LOKALNY.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/TEST_LOKALNY.json).

- Świeże, ścisłe wczytanie wag: potwierdzone w raporcie.
- Checkpoint: krok 7500; urządzenie CUDA, lokalny RTX 3070.
- Start: czysty szum Gaussa; seed 20261006; DDIM 50 kroków.
- Naturalny warunek: film walidacyjny TR_0127_S1_P03, klatka 709.
- Kontrola: latent TR_0129_S1_P03 przy identycznym szumie.
- Wygenerowany obraz z właściwym warunkiem: pełny MSE 0,126363; względny MSE 0,383872; cosine 0,787896.
- Niewłaściwy warunek względem tego samego celu: względny MSE 0,868165; cosine 0,488472.
- Wszystkie wartości skończone.
- Confidence: obraz źródłowy 0,948924; generacja z właściwym warunkiem 0,525234.
- IoU obrazu źródłowego z jego etykietą: 0,746950.
- IoU ramki wygenerowanego obrazu z ramką źródłową: 0,184624. To diagnostyka zachowania układu, nie trafność detekcji wobec niezależnie oznaczonej etykiety wygenerowanego obrazu.
- Wcześniejsze oględziny: scena operacyjna i oko rozpoznawalne, obraz nie jest jednolity. Narzędzie, drobne struktury i położenie są zniekształcone. Inny warunek zmienia kompozycję.
- To jeden lokalny przykład, nie oszacowanie jakości całego zbioru.
- Obrazy i latenty testowe utrzymywano w RAM, bez zapisu na dysk.

Test sztucznego celu K8000 pozostaje niepotwierdzony. Znane wcześniejsze cele pochodzą z K1000/K6402; nie wolno ich przedstawiać jako zgodnych celów K8000.

## Diagnoza: fakty oraz hipotezy

### 1. Sposób podawania filmów

Sampler tasował kolejność filmów, ale wewnątrz filmu dostarczał długie sekwencje klatek. Przy najlepszym SD, krok 7500, etap dyfuzji zobaczył 52 ze 118 filmów, czyli 44,07%. Końcowe pokrycie wyniosło 116/118; dwa filmy zostały odrzucone z powodu niespójności danych.

Fakt: pokrycie filmów rosło powoli. Hipoteza: duża korelacja kolejnych paczek i zmiany składu danych utrudniały stabilne uczenie. Sam przebieg nie dowodzi, że jest to jedyna przyczyna.

### 2. Co 600 klatek: zbyt krótka obserwacja po zmianie

Provenance potwierdza `train_frame_stride=600`, wznowienie z kroku 18487 i wcześniejszy stride 1. Pozostało tylko 513 aktualizacji do końca. Nie resetowano best/stale ani wag.

Nie ma podstaw do ogłoszenia nieskuteczności tego próbkowania. Liczba około 567 dopuszczonych pozycji po zmianie dotyczy stałej siatki klatek i nie oznacza, że cały wcześniejszy trening korzystał tylko z tylu obrazów.

### 3. Funkcja straty nie wymusza dokładnie geometrii narzędzia

Główna strata SD to MSE przewidywanego szumu. Dodatkowy błąd pełnego latentu ma współczynnik 0,05, jest rozważany co cztery aktualizacje i tylko dla elementów z timestep < 250. Ramki i etykiety narzędzia nie są używane w tej stracie bezpośrednio.

Hipoteza: poprawa ogólnej sceny może dominować nad małym obszarem narzędzia. Należy zmierzyć ten efekt, a nie zakładać go na podstawie samej funkcji straty.

### 4. Ograniczona walidacja i niepewność diagnozy przeuczenia

VAE sprawdzano na dwóch klatkach na film, około 33% i 67% długości. SD generowało tylko dla pierwszej klatki każdego z 25 filmów, z jednym stałym seedem na przykład. Kontrola używała pierwszego dostępnego innego filmu.

Nie ma porównywalnej, stałej oceny generowania na TRAIN, która pozwalałaby oddzielić przeuczenie od niestabilnej optymalizacji i zmian składu danych. Confidence, cosine i odchylenie przestrzenne obrazu nie zastępują oceny szczegółów narzędzia.

### 5. Optymalizacja i generowanie

Kod stosował AdamW ze stałym learning rate SD 5e-5; nie było EMA ani harmonogramu obniżania learning rate. Lokalny DDIM 50 zaczynał od timestep 980, nie 999. To kandydaci do kontrolowanego sprawdzenia, nie udowodnione źródła błędu.

## Propozycje ulepszeń — niewdrożone

1. Losować klatkę z każdego przedziału 600 klatek, odświeżać losowanie w kolejnych przejściach i mieszać filmy wewnątrz paczki przez ograniczony bufor RAM. Nie zapisywać klatek/latentów.
2. Poszerzyć stałą walidację o kilka klatek i kilka seedów na film. Dodać analogiczny stały pomiar TRAIN oraz metryki obszaru narzędzia. Nadal odkładać całe filmy.
3. Na istniejącym najlepszym modelu porównać DDIM 50/100/200 i harmonogram rozpoczynający od ostatniego poziomu szumu. Utrzymać kompatybilną parametryzację predykcji; nie przełączać gotowych wag epsilon na v-prediction.
4. Sprawdzić, czy sama rekonstrukcja VAE zachowuje reprezentację ViT i geometrię narzędzia. Dobre PSNR/SSIM całego obrazu nie rozstrzygają tej kwestii.
5. Jako osobny, jawny etap można kontynuować własny model z kroku 7500 z zamrożonym własnym VAE i ViT K8000, mniejszym malejącym learning rate i EMA. Wagi mają pochodzenie od losowej inicjalizacji. Best checkpoint nie zawiera optymalizatora, więc wymagałoby to nowego optymalizatora; nie byłoby dokładnym wznowieniem dawnego stanu.
6. Zachować pełne wejście 600 × 768, dodać kontrolowany nacisk na błąd w obszarze prawidłowo oznaczonego narzędzia. Brak etykiety nie oznacza negatywnego przykładu.
7. Jeśli powyższe nie wystarczą, osobno sprawdzić przestrzenne podawanie pełnego układu tokenów do kilku skal UNet. To propozycja nowej architektury, nie wykonana poprawka.

Priorytet: lepsze mieszanie danych i dokładniejsza walidacja narzędzia. Samo wydłużenie niezmienionego biegu nie ma obecnie mocnego uzasadnienia. Porównania wariantów powinny mieć ten sam budżet i jasno określone kryterium; nie zmieniać kilku rzeczy naraz bez kontroli.

## Stan kontynuacji i ograniczenia

- Analiza i zapis tej sesji są zakończone.
- Zakończonego treningu plateau nie należy automatycznie wznawiać na podstawie tego dokumentu.
- Ostatnio zapisany stan monitoringu: wstrzymany z powodu potrzeby decyzji o zgodnym sztucznym celu K8000. Bieżący zapis sesji nie zmienia automatyzacji.
- Ostatni raport finalizacji opisywał zweryfikowaną lokalną kopię modelu i usunięcie zbędnych checkpointów smoke, około 7,69 GB. Szczegóły zachować w raportach finalizacji, nie traktować tej notatki jako aktualnego pomiaru dysku.
- Zachować najlepszy model, provenance, kod, metryki i potrzebne modele odzyskiwania. Nie uruchamiać starszego eksperymentu SD K1000.
- W tej sesji nie tworzono obrazów, latentów, kopii checkpointów, commitów ani pushy.

## Materiały źródłowe

- [Historia walidacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/trening/EVAL_HISTORY.jsonl).
- [Końcowy wynik](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/trening/FINAL.json).
- [Provenance](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/trening/PROVENANCE.json).
- [Test lokalny](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/TEST_LOKALNY.json).
- [Funkcja treningu i walidacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/kod/run.py).
- [Dane i sampler filmów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/kod/dane.py).
- [Architektura](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/kod/model.py).
- [Wcześniejszy plan diagnozy, stan z 6 października](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/DIAGNOZA_PLAN_TRENINGU_20261006.md).
- [Lin i in., Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891). Uzasadnia sprawdzenie zgodności harmonogramów; nie potwierdza przyczyny problemów tego konkretnego modelu.

## Pliki utworzone na polecenie zapisu sesji

- `README.md`: krótki opis i odnośniki.
- `SESSION_SUMMARY.md`: ten dokument, wyniki, diagnoza, zakres niepewności i propozycje.

## Aktualizacja: wdrożenie po poleceniu „No rób to”, 2026-10-07

Poprzednie sekcje dokumentują stan po analizie. Użytkownik następnie autoryzował wykonanie ulepszeń. Utworzono odrębny eksperyment; nie zmieniono ani nie wznowiono zakończonego treningu z 6 października.

### Implementacja i warunki porównania

- Kod: `SmallViTAblation/experiments/vit_sd_k8000_ulepszenia_20261007`.
- Lokalne metryki i dokumentacja: `SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007`.
- Eden: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia`.
- Każdy wariant startuje z tego samego własnego `best_sd.pt`, krok 7500, SHA256 `6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7`. Nowy optymalizator jest celowy, ponieważ model bazowy nie zawiera starego stanu optymalizatora.
- Własny VAE i teacher ViT K8000 są zamrożone. Wymagany SHA256 teachera: `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`. Wejściem jest pełny latent 600 × 768.
- Diagnostyka przed treningiem porównuje DDIM 50/100/200, leading/trailing. Wybór na tej samej próbce walidacyjnej; TRAIN tylko diagnostycznie.
- A_control: dotychczasowy sekwencyjny sampler co 600 klatek, LR 5e-5.
- B_mixed: jedna losowa klatka z każdego okna 600 klatek i przeplatanie filmów, LR 5e-5.
- C_optim: dane jak B, LR malejący od 1e-5 do 2e-6 oraz EMA 0,999.
- D_roi: jak C oraz dodatkowa strata pełnej reprezentacji w obszarze prawidłowo oznaczonego narzędzia; współczynnik 0,05.
- Maksymalnie 4000 nowych aktualizacji na wariant. C kontra B ocenia łącznie zmianę LR i EMA; D kontra C izoluje wpływ dodanej straty ROI.
- Stała walidacja: 25 odłożonych filmów, po dwie klatki (33% i 67%), po dwa seedy. Wyniki uśredniane po seedach, klatkach i filmach. Kontrola innego warunku używa tego samego szumu. Stała mała próbka TRAIN jest oddzielną diagnostyką.
- Wczesne zatrzymanie uwzględnia pokrycie filmów TRAIN. Zachowane osobne best i latest, optymalizator, EMA i RNG. Wznowienie strumienia filmów nie odtwarza bitowo kolejności danych; jest to zapisane w provenance.
- Brak klatek, obrazów i latentów na dysku. RAM buforuje ograniczoną liczbę filmów; zużycie zależy także od rozmiaru MP4. Zapisywane są kod, metryki, logi i checkpointy.

### Weryfikacja i uruchomienia

Testy CPU: 49/49 przeszło. Niezależny przegląd wskazał i poprawiono: ważenie ROI między GPU, odzyskanie FINAL po przerwanym zapisie, sygnał łagodnego zatrzymania oraz wznowienie STOP_TIME na granicy wariantów. Testy regresji obejmują te przypadki.

Pierwszy smoke `1823542` został kontrolowanie zatrzymany po poprawkach z przeglądu; jego wyniki nie potwierdzają ukończenia. Smoke finalnych źródeł `1823543` działa na czterech GPU Hopper. O godz. 04:48 +02:00 ukończył A i B, przechodził do C. To test techniczny, nie dowód poprawy jakości. Pełny trening zostanie zgłoszony dopiero po ukończeniu wszystkich czterech prób, świeżym wczytaniu zapisów i potwierdzeniu hashy kodu.

Godzinny monitoring `monitoruj-przeuczenie-sd-z-latentem-vit` został aktywowany dla wyłącznie nowej rodziny, nazwa joba `sdimpk8`. Wyszukuje bieżący job z submissions.log i SLURM, zapobiega duplikatom i pozostaje cichy przy zwykłym postępie. Po zakończeniu ma porównać warianty, pobrać i sprawdzić model lokalnie oraz posprzątać wyłącznie własne zbędne zapisy po zabezpieczeniu modeli.

Nie wykonywano commitów ani pushy. Nie zatrzymywano zadań innych eksperymentów użytkownika.

### Potwierdzony start produkcyjny, 2026-10-07 04:52 +02:00

Smoke `1823543` zakończył wszystkie cztery warianty oraz świeże wczytania checkpointów. SLURM: COMPLETED 0:0; SMOKE_OK: passed. Potwierdzono zgodność hashy źródeł lokalnych i przetestowanej wersji. Wynik tego krótkiego testu nie jest dowodem jakości.

Uruchomiono raz `deploy.py --submit`. Nowy job **1823544**, nazwa **sdimpk8**, stan **RUNNING**, start **04:51:54 +02:00**, node Hopper, **4 × H100**, 24 CPU, 160 GB RAM. Rozpoczęła się produkcyjna diagnostyka; za nią skrypt wykona cztery warianty po maksymalnie 4000 nowych aktualizacji. Nie ma jeszcze wyników porównania jakości po nowych treningach.

Wyniki biegu: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie`. Lokalne dowody wdrożenia: `outputs/vit_sd_k8000_ulepszenia_20261007/URUCHOMIENIE_ZWERYFIKOWANE.json` i podfolder `smoke_1823543`. Agentom implementującym sampler i ewaluację zakończono pracę po odbiorze kodu. Aktywny pozostaje zlecony monitoring godzinny.

### Kontrola godzinowa 2026-10-07 03:45 UTC / 05:45 +02:00

Job 1823544 działa bez wykrytej awarii. Wariant A_control osiągnął 1160 nowych aktualizacji, zapis recovery jest po 1000. Walidacja obejmuje wszystkie 25 filmów, 50 klatek i 100 generacji. Pełny względny MSE zmienił się z 0,623735 do 0,566433, cosine z 0,639258 do 0,690462; względny błąd ROI wzrósł z 1,101779 do 1,154150. To wynik pośredni wariantu kontrolnego, nie rozstrzygnięcie porównania. Pokrycie TRAIN: 116/118 filmów; dwa znane filmy odrzucono z powodu różnicy liczby klatek MP4 i CSV. Walidacja nie ma odrzuceń.

Diagnostyka porównała sześć wariantów odszumiania; zgodnie z zapisaną regułą spójności między filmami/seedami pozostawiła DDIM 50 leading. B_mixed, C_optim i D_roi jeszcze nie rozpoczęły treningu. Nie wykonano restartu, modyfikacji treningu ani sprzątania. Monitoring pozostaje aktywny; normalny postęp bez dodatkowego powiadomienia. Szczegóły zapisano w MONITORING_20261007_0345UTC.json w wynikach nowego eksperymentu.

### Kontrola 2026-10-07 11:49 UTC / 13:49 +02:00

Job 1823544 nadal RUNNING. A_control i B_mixed zakończyły planowane 4000 nowych kroków; C_optim osiągnął 3860, D_roi jeszcze oczekuje w sekwencji. Brak nowych błędów CUDA/DDP i brak STOP/STOP_TIME. Wszystkie dotychczasowe walidacje zachowały 25 filmów i 100 generacji; dwa znane niespójne filmy TRAIN są odrzucane.

Najlepsze pełne względne MSE w tej samej walidacji: A 0,519988 (3000 nowych kroków), B 0,542292 (4000), C na razie 0,536456 (3000). Błąd ROI odpowiednio 0,976266 / 1,057675 / 1,036664. To wyniki częściowe; nie wybrano końcowego zwycięzcy i nie wykonano jeszcze końcowych oględzin obrazów.

Wolne miejsce na filesystemie: 6.24 GiB. Dodatkowy szczyt dla zapisów wariantu D oszacowany z bieżących rozmiarów modeli: 4.11 GiB, około 2.13 GiB zapasu, o ile inne zapisy nie zajmą miejsca. Niczego nie usuwano podczas aktywnego treningu. Szczegóły w MONITORING_20261007_1147UTC.json. Monitoring pozostaje aktywny, bez powiadomienia o zwykłym postępie.


### Kontrola 2026-10-07 12:45 UTC / 14:45 +02:00

Job 1823544 pozostaje RUNNING. A, B i C zakończyły po 4000 nowych kroków; najlepszy wynik C pozostał przy 3000 krokach, pełny względny MSE 0,536456. D_roi osiągnął 870 nowych kroków, przed pierwszą oceną po treningu (oceny co 1000). Nie ma COMPARISON ani STOP/STOP_TIME; log nie zawiera nowych błędów CUDA/DDP/zapisu. Sprawdzono zgodność teachera K8000, bazy SD oraz pełnego wejścia 600x768 w provenance wszystkich wariantów.

Wolne miejsce: 4,79 GiB. Aktualny początkowy recovery D ma około 1,01 GB; późniejszy pełny recovery z optymalizatorem będzie miał około 1,93 GB, zgodnie z C. Obecny zapas wystarcza na atomowy zapis. Niczego nie usuwano, nie restartowano i nie dublowano. Zapis szczegółów: MONITORING_20261007_1243UTC.json. Monitoring pozostaje aktywny; zwykły postęp bez dodatkowego powiadomienia.


### Kontrola 2026-10-07 13:46 UTC / 15:46 +02:00

Job 1823544 działa; D_roi ma 2160 nowych aktualizacji i poprawny zapis recovery po 2000. A/B/C są zakończone. W D najlepszy dotąd pełny względny MSE wynosi 0,596584 po 1000 kroków; po 2000 wynosi 0,607802, cosine 0,660744, względny błąd ROI 1,143724. Stale=1, degradation=0, brak warunku zatrzymania. Wszystkie 25 filmów walidacyjnych i 100 generacji zachowane, pokrycie TRAIN 116/118. To nie jest końcowy wynik ani dowód przeuczenia.

Brak błędów, STOP, STOP_TIME i końcowego COMPARISON. Wolne 3,93 GiB; po uwzględnieniu tymczasowej kopii recovery przy zapisie pozostaje około 2,13 GiB. Nie zmieniano kodu ani treningu, nie wykonywano restartów, transferu modeli lub sprzątania. Monitoring aktywny, normalny postęp bez dodatkowego powiadomienia. Szczegóły: MONITORING_20261007_1345UTC.json.


### Kontrola 2026-10-07 14:50 UTC / 16:50 +02:00

Job 1823544 działa. D_roi: 3580 nowych aktualizacji; recovery po 3000. Ostatnia walidacja po 3000: pełny względny MSE 0.549792, cosine 0.696403, względny MSE ROI 1.023102. Najlepszy pełny błąd D dotąd 0.549792. Wszystkie 25 filmów walidacyjnych i 100 generacji; A/B/C zakończone. Brak błędów i końcowego COMPARISON. Wolne 3.93 GiB. Bez restartu, zmian i sprzątania; monitoring aktywny. Szczegóły: MONITORING_20261007_1449UTC.json.


### Zakończenie porównania, 2026-10-07 17:13 +02:00

Job 1823544 zakończył się COMPLETED 0:0 po 12:21:32. Wszystkie warianty wykonały po 4000 nowych kroków. Najlepszy według pełnego względnego MSE jest A_control: 0,519988 po 3000 nowych kroków, 10500 łącznie. Pozostałe: B 0,542292, C 0,536456, D 0,544533. Względem bazowego 0,623735 kontrola poprawiła wynik o 16,63%; badane pakiety zmian nie pokonały kontroli. To nie jest dowód poprawnej jakości obrazu ani istotności różnic.

Pobrano metryki, manifest, kod i logi; sprawdzono sumy plików. Najlepszy kompletny checkpoint znajduje się w `SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/eden_kopia/wybrany_model/best_sd.pt`. SHA256: `1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41`.

Finalizacja lokalna: test na dwóch odłożonych filmach i dwóch seedach przygotowany; na tym etapie brak potwierdzonego wyniku. Pierwszy proces wolno ładował biblioteki, drugi zakończył się native access violation podczas diagnostycznego zrzutu faulthandler. Kolejna pojedyncza próba uruchomiona z logowaniem importów, bez faulthandlera. Modele i trening nie zostały zmienione ani wznowione. Do potwierdzenia testu nie usuwamy checkpointów recovery/smoke.

Bezpośrednia kontrola `/dev/shm` na hopper przez SSH nie powiodła się: brak znanego klucza ED25519. Nie obchodzono weryfikacji hosta. W skrypcie treningu jest trap usuwający wyłącznie własny katalog scratch, ale jego skuteczność na nodzie nie została niezależnie potwierdzona.


### Finalizacja lokalna i sprzątanie, 2026-10-07

Świeże wczytanie A_control best10500 na RTX3070 zakończyło się poprawnie. Zastosowano sekwencyjne przenoszenie modułów GPU/CPU; nie zatrzymano innych procesów. Test: TR_0127 klatka709 i TR_0129 klatka172, seedy20261007/20261008, pełny latent600x768, DDIM50leading, czysty szum i naturalny forward ViT bez podmian. Cztery generacje zachowano wyłącznie w RAM. Pełny względny MSE lokalnie0,494920, cosine0,735686; z innym warunkiem MSE0,830999. Względny błądROI1,091135, IoU względem układu źródłowego0,100307 (diagnostyka, nie niezależna prawdziwa etykieta generacji). Naturalny test działa; brak potwierdzonego sztucznego celuK8000 pozostaje osobnym ograniczeniem.

Oględziny przez RAM preview potwierdziły rozpoznawalne obrazy oka, ale widoczne artefakty i brak wiernego odtworzenia położenia/kształtu narzędzia. VAE rekonstruuje źródła dużo wierniej. Podgląd: http://127.0.0.1:62614/ (proces34280 służy już tylko podglądowi, bez treningu). Nie zapisano obrazów ani latentów na dysku. Ładowanie bibliotek w ostatniej próbie zakończyło się bez zmiany modeli; wcześniejszy native access violation wystąpił w próbie z faulthandlerem. Nie twierdzimy, że jego przyczyna jest ostatecznie zdiagnozowana.

Po weryfikacji lokalnego checkpointa, plików metadata i testu usunięto14własnych zbędnych checkpointów smoke/recovery wyłącznie rodziny7października: 17824656483bajty, czyli17,82GB /16,60GiB. NaEdenie pozostały cztery best_sd.pt i małe artefakty; katalog ma2,1GiB. Brak aktywnych/oczekujących zadań. Sprawdzono/dev/shm na hopper przez uwierzytelniony SLURM job1823778 (1CPU,256MB,bezGPU,COMPLETED0:0): brak katalogów scratch tych trzech zadań. Nie obchodzono kontroli kluczySSH.

Raport: `SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/RAPORT_KONCOWY.md`; pomiary: `TEST_LOKALNY.json`; sprzątanie: `RAPORT_SPRZATANIA.json`. Zakończono zakres porównania, testu naturalnego i sprzątania. Monitoring godzinny `monitoruj-przeuczenie-sd-z-latentem-vit` został usunięty po weryfikacji wyniku i sprzątaniu; narzędzie potwierdziło deleteStatus=deleted. Nie wznowiono treningu ani nie zainicjowano nowego eksperymentu.

### Przeniesienie najlepszego modelu do assets, 2026-10-08

Na polecenie użytkownika przeniesiono lokalny best_sd.pt wariantu A_control, krok 10500, do [assets/SD_ViT_K8000_A_control_K10500](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500). W poprzednim katalogu wybrany_model pozostał opis i PRZENIESIENIE.json wskazujący nową lokalizację. Historyczne JSON i kod zachowano bez przepisywania dawnych ścieżek.

Sprawdzono SHA256 1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41 oraz rozmiar 546432766 bajtów. Dodano kartę modelu README.md, PROVENANCE.json, oryginalną kopię PROVENANCE_TRENINGU.json, wyciąg METRYKI.json, SHA256SUMS i dwa zgodne z provenance pliki architektury w kod/. Wagi są identyczne z wcześniej zweryfikowanymi lokalnie. Nie wykonywano nowego treningu ani generacji. Modele innych wariantów i pliki na Edenie pozostały nienaruszone.
