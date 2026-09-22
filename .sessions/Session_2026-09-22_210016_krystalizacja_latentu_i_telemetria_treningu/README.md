# Sesja 2026-09-22 21:00 - krystalizacja latentu i telemetria treningu

## Cel sesji

Sprawdzic, jak latent detektora RF-DETR Nano zmienia sie w trakcie uczenia, czy sie
krystalizuje, czy da sie po nim poznac, ktore klatki niosly informacje poprawiajaca model,
i czy takie klatki da sie wskazac z analizy koopmanowskiej. Nastepnie uzupelnic brakujace
metryki treningu na Edenie i odpowiedziec, kiedy siec jest naprawde wytrenowana.

## Typ

experiment

## Workerzy uzyci

Brak - sesja solo.

## Co zrobiono (skrot)

- Zalozono nowy eksperyment `SmallViTAblation/experiments/krystalizacja_latentu` z pieciona
  skryptami analizy i wlasnym katalogiem danych oraz wynikow.
- Zmierzono zegar treningu na jedenastu checkpointach: latent ustawia sie po pierwszej epoce,
  potem tylko dryfuje, a po szczycie jakosci rozlewa sie na coraz wiecej kierunkow.
- Znaleziono sygnal konca treningu liczony bez etykiet: szczyt trwalosci najwolniejszych modow
  operatora kroku klatki, zgodny ze szczytem jakosci w 32 z 36 ustawien.
- Zapisano trzy wyniki negatywne: brak przewidywalnosci trafnosci na klatce, brak zwiazku
  pokrycia filmu walidacyjnego z liczba pudel, brak przewagi przestrzeni Koopmana nad
  kierunkami losowymi przy ocenie zestawu klatek.
- Sciagnieto z Edenu pelna telemetrie 28 biegow kampanii i ustalono, ze kampania nigdy nie
  zapisywala straty na walidacji.
- Dopisano przelacznik `--val-loss` do trenera i uruchomiono trzy nowe biegi na Edenie z pelna
  telemetria oraz wagami co dwie epoki.
- Potwierdzono na trzech ziarnach: siec jest wytrenowana po szesciu epokach, a sygnal bez
  etykiet kosztuje osiem razy mniej jakosci niz trening do trzydziestej epoki.

Szczegoly w `SESSION_SUMMARY.md`.
