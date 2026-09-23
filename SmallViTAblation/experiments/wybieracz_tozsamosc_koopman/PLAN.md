# Eksperyment 1. Wybieracz klatek z latentu wstępnie wyuczonego detektora

Status: plan, zapisany 2026-09-23. Część pomiarów offline już wykonano (sekcja „Stan wyjściowy”).
Trening wymaga osobnej zgody użytkownika.

Przegląd wszystkich trzech eksperymentów: `docs/Analiza_ponownej_publikacji/46_PLAN_TRZECH_EKSPERYMENTOW_DOBORU_KLATEK.md`.

## Pytanie

Czy mały wybieracz, działający wyłącznie na latencie wstępnie wyuczonego detektora i bez nowej sieci,
wybiera klatki, które wnoszą coś nowego, zamiast dokładać podobne, i przez to dają lepszy detektor
przy tym samym budżecie klatek niż wybór losowy?

## Pomysł (sformułowanie użytkownika, 2026-09-23)

Wybieracz łączy trzy liczby liczone dla każdej klatki puli:

1. **Wektor tożsamości.** Kierunek w latencie (tokeny backbone), który zapala się tam, gdzie widać narzędzie.
   Gdy zapala się słabo, detektor częściej chybia.
2. **Zaskoczenie Koopmana.** Koopman to liniowy przewidywacz tego, jak latent zmienia się z klatki na klatkę.
   Klatka, której nie umie przewidzieć, pokazuje coś nowego, na przykład wejście pęsety albo zmianę barwy tła.
3. **Poprawka Langevina.** Langevin dzieli błąd przewidywania na część wynikającą z historii ruchu i na zwykły szum.
   Za nowe uznajemy tylko klatki, których błąd wyraźnie przekracza normalny szum danego filmu.

Z tych trzech liczb wybieramy klatki, biorąc po jednej z każdej grupy podobnych, żeby nie powtarzać sąsiadów.

## Stan wyjściowy (pomiary 2026-09-23, commity 9a6bd8a i f758fc3)

Ocena na 25 operacjach testowych, co 10. klatka (8249 klatek, 7908 z narzędziem),
skrypty `experiments/krystalizacja_latentu/ocena_test.py` i `koopman_zaskoczenie.py`.

| Miara (detektor ut_b0500) | AUC chybień | AUC pewnych chybień |
|---|---|---|
| wektor tożsamości pod obrysem z adnotacji (wymaga etykiet) | 0,76 | 0,72 |
| obecność z mapy wektora, bez etykiet | 0,60 | 0,53 |
| zwykła zmiana latentu między klatkami | 0,60 | 0,61 |
| zaskoczenie Koopmana (cały kadr) | 0,58 | 0,58 |
| Koopman + Langevin (Mahalanobis reszty) | 0,57 | 0,58 |
| niska pewność detektora | 0,88 | 0,71 |
| wektor + Koopman–Langevin (średnia rang) | 0,71 | 0,70 |

Wnioski wyjściowe:

- Koopman na latencie całego kadru opisuje ruch sceny, nie niewiedzę detektora. Przegrywa ze zwykłą różnicą klatek.
- Wektor 0,76 wymaga obrysu z adnotacji, więc w tej postaci nie działa na nieoznaczonej puli.
- Główne źródło pewnych chybień to inne cienkie narzędzia (pęseta, drugi instrument) i nietypowe tło
  (galeria `experiments/krystalizacja_latentu/wyniki/test/TEST_chybienia.jpg`).
- **Uwaga metodyczna.** Te pomiary wykonano na zbiorze testowym. Dalsze rozwijanie wybieracza prowadzimy na
  25 operacjach walidacyjnych. Zbiór testowy zostaje do jednej oceny końcowej, żeby nie dopasować metody do testu.

## Co zmieniamy względem stanu wyjściowego

1. **Wektor bez etykiet.** Kontrast wektora liczony pod ramką przewidzianą przez detektor
   (i pod maską z progowania mapy wektora), a nie pod obrysem z adnotacji.
2. **Koopman w układzie narzędzia.** Stan to tokeny w otoczeniu przewidzianej ramki, wyrażone względem czubka
   i osi narzędzia (obrót i skala zdjęte), a nie średnia z całego kadru. Wtedy ruch narzędzia nie jest „nowością”,
   a nowością jest zmiana jego wyglądu lub otoczenia.
3. **Langevin z pamięcią.** Reszta Koopmana z członem pamięci (opóźnienia 1–3 kroków) i wybielaniem przez
   kowariancję szumu dopasowaną na innych filmach. Porównujemy z wersją bez pamięci.
4. **Różnorodność.** Grupowanie latentów (k-średnich na latentach klatek puli) i jedna klatka z grupy,
   najwyżej oceniona łącznym wynikiem.

## Protokół

Etap A, offline, bez uczenia sieci (lokalnie, RTX 3070):

- Dane: 25 operacji walidacyjnych, co 10. klatka. Koopman i Langevin dopasowane na połowie operacji,
  oceniane na drugiej (i odwrotnie).
- Miara: AUC chybień i pewnych chybień detektorów ut_b0500 oraz b1500_best.
- Warunek przejścia do etapu B: łączny wynik bez etykiet przewyższa niską pewność detektora na pewnych chybieniach
  (obecnie 0,66–0,71) albo, po dodaniu do pewności, podnosi AUC o co najmniej 0,02 w obu połowach.

Etap B, trening (Eden, po zgodzie):

- Ramiona: wybieracz 250 klatek, losowe 250, pewność detektora 250 (klasyczne wybieranie niepewnych klatek),
  każde na trzech ziarnach (42, 1337, 2024). Pula: ta sama co w biegach `maly_budzet`.
- Trening jak w `maly_budzet` (RF-DETR Nano, 384), sonda włączona.
- Miara: mAP 0,5:0,95 na walidacji w najlepszej epoce, mAP na teście raz, na końcu.
- Odniesienie z wcześniejszych biegów: losowe 250 klatek dały 0,6319, dobór po kwadratach K6 0,6296 ± 0,0011.
- Kryterium sukcesu: przewaga nad losowym większa niż rozrzut między ziarnami i powtarzalna na trzech ziarnach.

## Czego ten eksperyment nie dowodzi

- Wysokie AUC offline mówi, że miara wskazuje trudne klatki. Nie mówi, że dodanie ich do uczenia poprawi detektor.
  To rozstrzyga dopiero etap B.
- Wektor tożsamości wyznaczono w latencie detektora ut_b0500. Jain i in. (2022) ostrzegają, że kierunki błędów we
  własnym latencie modelu psuje zapamiętanie klatek uczących. Jeśli etap B zawiedzie, powtarzamy z zamrożonym
  backbone sprzed dostrajania.

## Powiązania

- Równoległa sesja przygotowała `experiments/tozsamosc_czastki_bledy` (kod bez uruchomienia) o podobnym pytaniu
  diagnostycznym. Nie ruszamy jej plików. Przed uruchomieniem etapu A porównać zakres, żeby nie dublować pomiaru.
