# SD warunkowane pełnym latentem ViT: zakończony trening i test lokalny

Trening zakończył 20 000 aktualizacji. Najlepszy checkpoint: krok 8500, val_film_mean_mse 0,1641106312. Ostatnia walidacja: 0,1648451703. Walidacja obejmowała 256 klatek z 8 filmów, nie cały podzbiór VAL. Wynik nie dowodzi poprawy detekcji ani przyspieszenia uczenia ViT.

Pobrano best_checkpoint.pt; SHA256 4c450bdd069a8044b6e4c95a922caee242a88f207abf9506c2833bbcf49b85f7 zgadza się z Edenem. Świeże wczytanie sprawdziło SHA wag bazowego UNet, VAE i oryginalnego checkpointa ViT. Generacje odbyły się lokalnie na RTX, po 50 krokach DDIM, od losowego szumu z identycznym seedem 20261006. Warunkiem był pełny latent 600x768. ViT oceniał obrazy naturalnym forward bez podmiany aktywacji.

| Cel | MSE128 | Zgodność wartości J128 | Cosine128 | MSE640 | Confidence |
|---|---:|---:|---:|---:|---:|
| Naturalny: VAL TR_0127_S1_P03, klatka 1 | 0,0596651 | 33,4009% | 0,773022 | 0,0693993 | 0,445949 |
| Pierwotny sztuczny latent | 85,53212 | 0,561494% | 0,175799 | 0,0812940 | 0,242426 |

Zgodność wartości = 100*max(0,1-sqrt(MSE128/mean(target128^2))). Cosine opisuje kąt i nie jest procentową zgodnością wartości.

Oryginalne zdjęcie VAL: confidence 0,792403; IoU wobec prawdziwej etykiety 0,593301. IoU wygenerowanego obrazu wobec ramki referencji wyniosło 0, ale to wyłącznie zgodność układu z oryginałem; nowy obraz nie ma własnej prawdziwej etykiety, więc nie raportujemy tego jako trafności detekcji. Sztuczna generacja również nie ma GT.

Oględziny podglądu: generacja naturalna przedstawia oko i otoczenie operacyjne, ale nie odtwarza narzędzia ani układu oryginału. Generacja sztuczna jest niemal jednolitym szarobrązowym polem. Dwa różne warunki przy tym samym szumie dały różne obrazy; potwierdza to wpływ warunku na wynik, nie poprawność odtworzenia celu. Wszystkie wyjścia tensorowe były skończone.

Obrazy i latenty nie zostały zapisane na dysku. Podgląd przechowywany jest w RAM, dostępny dopóki działa lokalny serwer http://127.0.0.1:8769. Metryki zapisano w TEST_LOKALNY.json. Wynik dotyczy jednego seed oraz dwóch celów; to test działania, nie badanie generalizacji.

Po awarii filmu TR_0007_S1_P03 wdrożono odrzucanie całych filmów z niezgodną liczbą klatek MP4/CSV przed podaniem próbek, z raportowaniem w logach. Manifest 118 filmów oznacza kandydatów, nie potwierdzoną liczbę wykorzystanych poprawnych filmów. Szczegóły wznowienia: NAPRAWA_1823172.json.
