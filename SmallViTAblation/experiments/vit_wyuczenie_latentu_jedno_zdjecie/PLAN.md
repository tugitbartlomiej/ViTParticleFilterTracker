# Naturalne wyuczenie sztucznego latentu jednego obrazu

Pytanie: czy oryginalne wagi przed mierzonym GELU potrafią odtworzyć dokładnie zapisany cel after dla 600 tokenów? Brak agreement128 >=90 albo top1 IoU >=0.90 i confidence >=0.90 obala sukces w danym budżecie. Agreement = 100 max(0,1-sqrt(MSE128/mean(target128^2))); cosine jest osobnym pomiarem.

Podłoga: oryginalny checkpoint k1000, ten sam RGB z JPEG, resize 480x320 INTER_AREA i ImageNet wewnątrz modelu. Sprawdzamy zgodność z before przy rtol=2e-5, atol=2e-5, hash źródeł oraz baseline confidence/IoU. Pozostałe 640 kanałów mają niezmieniony cel before. Cel after nie jest przeskalowywany ani przycinany.

T1: tylko norm2 affine i fc1 weight/bias ostatniego bloku, Adam lr=0.0003, 1000 aktualizacji, cosine lr do 0.00003. Loss = MSE128 / stała energia target128 + 10 MSE640 / stała energia target640 + 0.02 rzeczywista strata CenterNet/GIoU oryginalnego detektora na GT. Grupy są rozdzielone. Wszystkie inne parametry i bufory BN zamrożone; eval, float32 bez autocast i TF32. Żadna podmiana aktywacji. Pełny forward przy każdej aktualizacji.

Logi 0/1/10/25/50/100/250/500/1000 i co 100 kroków. Stop po sukcesie albo plateau: po >=500 krokach poprawa najlepszego agreement w ostatnich 250 krokach <0.2 punktu procentowego. Jeżeli T1 nie osiągnie celu i wykazuje plateau (także końcowe), opcjonalny T2 startuje z najlepszego T1 i dodaje attention/norm1 ostatniego bloku, z tym samym budżetem i lr. T2 wyraźnie oznaczony. fc2, final norm i dense head nadal zamrożone. Zmiana wejścia residual w T2 jest monitorowana przez rzeczywistą detekcję i stratę.

Warianty widzą podczas treningu jeden obraz, cel after oraz GT. Inference widzi tylko piksele. Ocenę wykonujemy na tym samym obrazie, więc jest to test overfit i osiągalności, bez wniosków o nowych obrazach; T2 ma więcej uczonych parametrów i nie jest porównaniem przy równej pojemności.

Best wybieramy po najwyższym agreement spośród poprawnych detekcji; jeżeli żadna nie spełnia progu, po najwyższym agreement i jawnie raportujemy niespełnienie detekcji. Before/best/final mają pełny state_dict, wersję architektury, listę parametrów, optimizer state i metadane. Niezależny proces ładuje nowe modele i wykonuje plain forward bez hooków; osobny odczytujący hook wyłącznie mierzy latent. Audyt każdego parametru i bufora, pełne latenty i CSV J128, overlay i dokładny czas.

RTX3070 współdzielona: limit alokatora 22 procent, oczekiwanie na >=1800 MiB wolnej pamięci lub CPU po 60 s, pauza 0.03 s między krokami. Nie zatrzymujemy cudzych procesów. Zapis wyłącznie we własnych dwóch katalogach; bez commitów i zmian źródła.

Pierwszy start miał błąd przygotowania GT (Tensor nie obsługuje round), naprawiony przez przekazanie tablicy NumPy do niezmienionej funkcji cele. Drugi start wykonał 96 aktualizacji, następnie Windows zablokował atomowy rename statusu podczas równoległego odczytu. Nie był to OOM. Brak checkpointu z aktualizowanymi wagami w tym starcie, więc następny start jawnie zaczyna od oryginalnego MODEL_before. Dodano unikatowe pliki tymczasowe PID/czas, ograniczone ponawianie OSError i niekrytyczne pomijanie statusu po wyczerpaniu prób. Każde nowe best jest utrwalane przed zapisem progress. Logi nieudanego podejścia pozostają w tej samej rodzinie.
