# Regulator przestrzenny J128 z celem AP50 > 0,90

2026-10-05. Zlecenie: znaleźć i uruchomić na RTX metodę ustawiania latentu zamrożonego ViT, z celem ponad 90% detekcji narzędzia. Operacyjnie: AP50 > 0,90. Pewność pojedynczej ramki nie jest tym kryterium. Próg nie jest gwarancją osiągnięcia.

## Pytanie, podłoga i dostęp do informacji

Czy wspólna funkcja ustawiająca J128, uwzględniająca sąsiednie tokeny i destylację treningowych celów oracle, przenosi poprawę na inne filmy? Brak poprawy poza próbą albo AP50 <= 0,90 obala odpowiednio użyteczność lub osiągnięcie celu. Baseline: identyczny zamrożony checkpoint k1000, obrazy, dekoder top10 i metryka AP. Zachowujemy 640 kanałów i wszystkie wagi/bufory ViT.

400 istniejących obrazów: train240/69 filmów, val75/23 filmy, test85/24 filmy. Podziały rozłączne po filmach. Wszystkie te wyniki historycznie oglądano. To adaptacyjna eksploracja starego benchmarku, nie niezależne końcowe potwierdzenie. Żadna etykieta val/test nie służy do gradientu, destylacji ani generowania celów oracle. Test odczytywany dopiero po zamrożeniu wyboru wszystkich konfiguracji. Kolejna iteracja wymaga jawnego oznaczenia powtórnego użycia benchmarku.

Każda metoda dostaje własny latent obrazu bez ID i etykiety. Regulator lokalny widzi wybrane aktywacje; przestrzenny dodatkowo zamrożony stan 192 wymiarów po residualu i sąsiedztwo na siatce20×30. Opcjonalne współrzędne siatki są jawnym, oddzielnym ramieniem. Różne liczby parametrów oceniamy poza próbą. Jedna wyuczona funkcja jest stosowana do wszystkich obrazów; nie oznacza jednak identycznych końcowych aktywacji różnych obrazów.

## Interwencja

Zmiana w tym samym miejscu co historyczny oracle: z'_J=max(z_J+sigma_J D_theta(z_J,Q), -0.16997120), gdzie z_J jest po GELU. Statystyki z i Q tylko z original train. Zakres aktywacji zachowany przez projekcję. Regulator może zmieniać aktywacje w obie strony. Duże zmiany nie dowodzą istnienia naturalnego zdjęcia realizującego te reprezentacje.

Szybka głowa: Q'=Q+ls2(W_fc2[:,J](z'_J-z_J)); norm i głowa detekcyjna pozostają zamrożone. Kontrole pełnego forwardu, gradientów, niezmienności640 kanałów i hashów wykonywane przed uruchomieniem oraz dla wybranych modeli po treningu.

## Dane i cel

Train posiada dodatkowe warianty: rzeczywisty pełny forward odbicia poziomego oraz zmiana jasności/kontrastu z prawidłową ramką. Nie odwracamy po prostu zapisanych cech. Val/test tylko oryginalne. Normalizacja i statystyki kanałów z original train.

Cele detekcyjne: stabilna focal loss przez log-sigmoid, 2GIoU, opcjonalnie center cross-entropy i SmoothL1 log-odległości. Ta ostatnia odmiana ma poprawiać ranking lokalizacji oraz ramkę, nie tylko podnosić logit. Metryka raportowana nadal historycznym dekoderem i AP, bez zmian progów.

Teacher: gradientowa optymalizacja wyłącznie aktywacji train, w tym augmentacji, 600 kroków Adam lr0.05 z projekcją GELU. Original train startuje od swoich dawnych latentów oracle (nigdy z części val/test); augmentacje startują od baseline. Najlepszy stan według nowej straty per obraz. Ten wynik ma etykietę oracle_train, nie generalizacja. Destylacja to SmoothL1 standaryzowanych aktywacji, z wagą0.3 i większą wagą tokenów z obszaru treningowego narzędzia. Bez GT podczas inferencji.

## Przeszukiwanie ustalone przed nowym testem

12 konfiguracji: post-GELU affine (2), lokalny MLP128 (3), przestrzenny CNN64 (4), przestrzenny CNN128 (2), przestrzenny CNN64 ze współrzędnymi (1). Warianty learning rate0.001/0.003 i destylacji0/0.3; dwie rodziny straty (focal+GIoU oraz dodatkowo center/box). CNN zawiera konwolucje3×3 z dylatacją1/2/4, GroupNorm, GELU i dropout0.1. Nie jest to odtworzenie architektury ViT-Adapter z artykułu.

AdamW, weight_decay0.01, batch16, gradient clip5, cosine learning rate z rozgrzewką5 epok, maksymalnie160 epok. Minimum60 epok, zatrzymanie po6 kolejnych pomiarach bez poprawy, pomiar co10. Jeden seed20261005. Kolejność minibatchy taka sama. Kara1e-5 średniej kwadratowej standaryzowanej korekty. Wybór po val AP50, następnie AP50:95, potem stracie. Epoka0/baseline jest kandydatem. Osobny wybór najlepszego statycznego affine. Wyniki wszystkich ramion pozostają zapisane.

## Artefakty i interpretacja

Kod wykonany i hashe, źródła danych/modelu, split, nauczyciel train i jego predykcje, historie, modele, zamrożony wybór przed testem, predykcje, raport, diagnostyka pełnego forwardu. Wskazujemy runtime, pamięć i wynik negatywny. Brak90% jest wynikiem; nie zastępujemy go oracle ani confidence. Testowanie może wybrać kierunek kolejnych badań, ale nie zmienia wybranego modelu tej kampanii.

## Literatura i hipoteza własna

[ViT-Adapter](https://arxiv.org/abs/2205.08534) motywuje zbadanie kontekstu przestrzennego przy zadaniach gęstych. [SSF](https://arxiv.org/abs/2210.08823) motywuje kontrolę wspólnych skal/przesunięć. Własna hipoteza: mały przestrzenny regulator na J128 z destylacją treningowych celów może przewyższać statyczną regulację. Prace nie dowodzą AP50>0,90 dla naszego ViT ani tego benchmarku.
