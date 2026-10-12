# Kontynuacja wyuczenia latentu do 2000 aktualizacji

Rodzina wcześniejsza: `experiments/vit_wyuczenie_latentu_jedno_zdjecie` i analogiczny katalog outputs. Wagi wejściowe: MODEL_T1_norm2_fc1_final.pt, krok1000, best=final. Wszystkie wcześniejsze artefakty pozostają nietknięte.

Podłoga zachowuje obraz, GT, target after, J128, architekturę, precision float32, eval BN i frozen downstream. Pierwszy forward musi odtworzyć naturalny latent z poprzedniego biegu i liczby: agreement64.475623, cosine0.94329035, MSE128 10.9162397, confidence0.77177793, IoU0.99995228. Przed update sprawdzamy Adam: cztery stany, każdy step1000, lr3e-5; przy braku zgodności przerywamy zamiast po cichu resetować optimizer.

Uczymy wyłącznie bloki.5.norm2 weight/bias i bloki.5.mlp.fc1 weight/bias. Ta sama loss: MSE128 / energia target128 + 10 MSE640 / energia target640 + 0.02 GT CenterNet/GIoU. Stałe energie pochodzą z oryginalnego niezmienionego targetu after. RGB480x320 INTER_AREA, normalizacja ImageNet wewnątrz oryginalnego modelu. Nie zmieniamy targetu ani forward aktywacji; nie dodajemy nowych parametrów.

Adam przywrócony, lr stały3e-5, bez odtwarzania warmup i bez nowego cosine. Dokładnie 1000 dalszych aktualizacji, logi kroków1000/1001/1010/1025/1050/1100/1250/1500/2000 i co100. Stop wyłącznie po2000 lub błędzie niefinitywności/wykonania, bez earlystop sukcesu lub plateau. Best wybierany jak wcześniej, dodatkowo zapisujemy final. Nie uruchamiamy T2.

Brak agreement >=90 i poprawnej detekcji (IoU>=0.90, confidence>=0.90) oznacza nieosiągnięcie celu w tym budżecie. Porównujemy oryginał0, wyuczone1000 i wyuczone2000 na tym samym obrazie. Każdy wariant inference widzi tylko piksele; trening ma target i GT. To test overfit jednego źródłowego obrazu, bez oceny generalizacji.

Wyniki: pełne naturalne latenty600x768 i CSV J128, before/best/final z Adamem, architekturą i listami zmian, audyt wszystkich wag i buforów wobec oryginału0, rzeczywiste dekodowanie detektora, plain forward w świeżym procesie i odczytujący hook. Osobne initial-checkpoint SHA i source hashes. Atomowy JSON z unikatowym PID/czas i ograniczonym retry Windows OSError; progress niekrytyczny, live-best utrwalany przed progress. RTX3070 limit22 procent, niewielka alokacja; bez zatrzymywania cudzych procesów. Jedyny zakres zapisu to dwa nowe katalogi tej rodziny, bez commitów.
