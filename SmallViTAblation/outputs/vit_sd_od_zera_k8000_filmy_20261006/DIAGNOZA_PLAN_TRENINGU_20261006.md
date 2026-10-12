# Jak poprawić trening SD warunkowanego pełnym latentem K8000

Data: 2026-10-06. Analiza kodu i zapisanych pomiarów. Propozycje poniżej nie zostały wdrożone ani potwierdzone nowym eksperymentem. Bieżący job: 1823430, kontynuacja od kroku SD 18487, próbkowanie TRAIN co 600 klatek. Bez ekstrakcji obrazów i latentów na dysk.

## Co wynika z pomiarów

- VAE nauczył się rekonstrukcji: PSNR 30.4499 dB, SSIM 0.94456, 50 klatek z 25 odłożonych filmów. To wynik rekonstrukcji VAE, nie generacji SD. Oględziny jakości i jakości narzędzia wymagają osobnego testu.
- Najlepszy zapisany względny błąd pełnego latentu SD to 0.535459 przy 7500 krokach. Ostatnia odczytana pełna walidacja przy 18500 krokach: 0.675314, plateau=true, stale=22. Spadek straty na bieżącej paczce nie dowodzi poprawy generacji. Strata STATUS jest z jednej paczki/ranku, a nie stałym porównywalnym testem.
- W manifestach TRAIN: 118 filmów i 313359 klatek. Stałe próbkowanie 1,601,1201... daje 577 pozycji. Po wyłączeniu zgłoszonych przez program filmów TR_0007_S1_P03 i TR_0005_S1_P03 pozostaje 567 pozycji. To nie jest liczba klatek już wykorzystanych przez cały historyczny bieg; wcześniej używano wszystkich klatek.
- Odczyt ZIP i dekodowanie MP4 nadal obejmują film. Odrzucenie niewybranych klatek następuje przed konwersją RGB i forwardem modeli. Stride600 zmniejsza obliczenia modeli na przejście filmu, lecz nie gwarantuje szybszego kroku treningu. Pomiar GPU z pomocniczego kroku SLURM bez przydziału GPU nie miał dostępu do urządzeń i nie jest dowodem ich bezczynności.
- Adapter to MLP token po tokenie i uczona pozycja; pełne 600x768 trafia do cross-attention. Brak osobnej gałęzi przenoszącej siatkę 20x30 do map UNet. To cecha konstrukcji, nie udowodniona przyczyna błędu.
- Trening SD optymalizuje MSE predykcji szumu. Dodatkowa strata zgodności ViT występuje co czwarty krok i tylko dla przykładów z t<250. Gradient przez wygenerowany obraz i ViT nie jest odcinany w features(). Nie zmierzono jeszcze norm gradientów ani rzeczywistego udziału poszczególnych strat.
- AdamW ma stałe lr=5e-5; nie ma EMA ani harmonogramu learning rate.
- Walidacja SD generuje jeden obraz z jednego seeda na film. PSNR/SSIM w tym samym rekordzie nadal dotyczą rekonstrukcji VAE. Wrong-condition jest jednym deterministycznie wybranym innym filmem, a nie pełnym testem permutacji warunków.

## Zalecana kolejność

### 1. Krótki test rozstrzygający na zapisanym best_sd

Zanim trenować kolejne tysiące kroków, porównać generację na tych samych kilku klatkach TRAIN i VAL i tych samych seedach. Mierzyć:

1. Naturalny latent referencyjny oraz latent rekonstrukcji VAE: względne MSE/cosine, żeby ustalić błąd wprowadzany przez sam kodek i preprocessing.
2. Błąd przewidywania szumu osobno dla niskiego, średniego i wysokiego poziomu szumu. Porównać właściwy warunek z latentem innego filmu. Sprawdzić, gdzie model faktycznie używa warunku.
3. Generację DDIM przy 50, 100 i 200 krokach; porównać obecną siatkę z trailing przy niezmienionych wagach i beta schedule. W lokalnym diffusers domyślne leading dla 1000/50 zaczyna od t=980; trailing zaczyna od 999. Wersję faktycznie użytej biblioteki na Edenie trzeba potwierdzić przed przypisaniem tego historycznym wynikom. Nie przełączać wytrenowanego epsilon-modelu na v-prediction samą zmianą konfiguracji.
4. Względne MSE pełnego latentu, cosine, jakość obrazu i reakcję na permutację warunku. Ocenić obraz również wzrokowo w RAM. Confidence/IoU tylko tam, gdzie istnieje poprawna etykieta; jakość na TRAIN odróżnić od VAL.

Jeżeli nawet mały zbiór TRAIN nie daje poprawnych generacji, priorytetem jest spójność odszumiania/skalowania i przepływu warunku. Jeżeli TRAIN działa, a VAL nie, priorytetem stają się dane, regularizacja i pojemność. To test diagnostyczny, nie dowód generalizacji.

### 2. Dane: rzadko, lecz za każdym razem inne klatki

Zamiast stale 1,601,1201... losować jedną klatkę w każdym kolejnym przedziale 600 klatek filmu, z nowym losowaniem na następne przejście. Losowanie ma być reprodukowalne z seed/epoka/film. Krótkie filmy również dostarczają co najmniej jedną klatkę. Mieszać filmy w paczkach z ograniczonego bufora RAM; monitorować proporcję klatek z oznaczonym narzędziem. Brak etykiety nie oznacza braku narzędzia.

Oddzielnie mierzyć czas odczytu ZIP, dekodowania, oczekiwania na paczkę, forward/backward oraz walidacji. Następnie dobrać prefetch/bufor RAM. Nie zapisywać klatek/latentów. Stałe klatki VAL pozostają bez zmian.

### 3. Wariant optymalizacji z niezmienioną architekturą

Pierwszy porównywalny wariant: zapisany best_sd z tego samego treningu od zera, zamrożony własny VAE, niezmieniony K8000 i pełny latent. Nowa gałąź eksperymentu, ponieważ świadomie zmienia się receptura. Porównać najpierw samo próbkowanie danych, potem EMA i malejący learning rate. Propozycja startowa do testu: lr 1e-5 do 5e-6, EMA 0.999; nie są to parametry o wykazanej tutaj optymalności. Jeśli best_sd nie zawiera optymalizatora, nowy optymalizator i warmup muszą być jawnie zapisane; to nie jest dokładne wznowienie starego procesu.

Porównania przy takim samym budżecie czasu GPU, klatkach VAL i seedach. Oddzielny licznik patience dla nowej, jawnie opisanej gałęzi; nie kasować historii ani nie obchodzić early stop starego biegu. Pełne treningi dopiero po krótkim wyniku wskazującym korzyść.

### 4. Dopiero jeśli model słabo wykorzystuje warunek

Sprawdzić skalę cech kanałów na TRAIN i normalizację adaptera. Zachować informację o amplitudzie: zwykła normalizacja każdego tokenu może usuwać część informacji istotnej dla sztucznej regulacji latentu; rozważyć stałe statystyki TRAIN albo osobny sygnał normy.

Przetestować dodatkową przestrzenną gałąź warunku: pełne 600x768 ułożyć jako 20x30x768, wyuczyć projekcje do rozdzielczości UNet i dodać je przez początkowo zerowe warstwy wyjściowe. Zachować cross-attention. Każda gałąź przyjmuje cały naturalny latent; nie redukować wejścia do J128 ani globalnej średniej. To własna adaptacja idei warunkowania przestrzennego, a nie dowód, że wyniki ControlNet z pretrained SD przenoszą się do naszego modelu od zera.

Min-SNR weighting jest kolejnym wariantem do porównania po pomiarze błędów według t. Wariant v-prediction + zero-terminal-SNR wymaga zgodnego celu treningowego i osobnego eksperymentu; nie jest prostą poprawką samplera starego checkpointa.

## Decyzja na teraz

Najwyższy priorytet: test odszumiania na TRAIN/VAL oraz losowana pozycja w przedziałach 600 klatek i mieszanie filmów. Zachować wyuczony tutaj VAE i nauczyciela K8000. Nie ma obecnie dowodu, że większy model, więcej GPU albo samo przedłużenie biegu poprawi wynik. Nie uznawać arbitralnych progów latent-MSE/cosine za zamiennik jakości obrazu i narzędzia.

## Dowody i źródła

- Kod: `experiments/vit_sd_od_zera_k8000_filmy_20261006/`.
- `dane.py:267`: stały wybór co frame_stride; `run.py:70`: walidacja; `run.py:204`: optymalizator; `run.py:251`: pomocnicza strata latentu; `model.py:41`: adapter.
- Lokalne `EVAL_HISTORY.jsonl`, `FILMY.json`, `VAE_COMPLETED.json`, log zadania 1823430 i stan SLURM.
- Lin et al., Common Diffusion Noise Schedules and Sample Steps are Flawed: https://arxiv.org/abs/2305.08891 — zgodność punktu startowego generacji i harmonogramu; nie potwierdza przyczyny w naszym biegu.
- Hang et al., Efficient Diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/abs/2303.09556 — uzasadnienie wariantu ważenia strat, bez przenoszenia deklarowanego przyspieszenia na nasz zbiór.
- Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models: https://arxiv.org/abs/2302.05543 — inspiracja warunkowaniem przestrzennym; oryginalna metoda korzysta z pretrained backbone.
