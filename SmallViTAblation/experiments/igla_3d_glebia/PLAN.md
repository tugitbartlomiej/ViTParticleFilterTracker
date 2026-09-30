# Igła w 3D: głębia z Depth Anything V2 i model ruchu sztywnej igły (2026-09-30)

Prośba użytkownika: „Pamiętaj, że to jest rzut z 3D na 2D! Wczytaj sobie model z folderu Hugging Face do estymacji
głębi i z takich obrazów i mapy generuj model ruchu. Modeluj ruch sztywnej metalowej igły. Szukaj, jak to się robi
w literaturze.” Kontynuacja `../serwowizja_zawias_narzedzia` (osie zbiegają się przy brzegu rogówki, ale obrót wokół
zawiasu w płaszczyźnie obrazu tłumaczy 1/5–1/14 ruchu końcówki).

## Model głębi

`depth-anything/Depth-Anything-V2-Small-hf` z `F:\HuggingFace\hub` (24,8 mln parametrów, tryb offline). Daje głębię
względną (odwrotność odległości z nieznaną skalą i przesunięciem w każdej klatce), nie metryczną.

Liczone na GPU. Karta była zajęta przez `horyzont.py` z sąsiedniej sesji, więc użytkownik polecił: „Poczekaj, aż się
zwolni GPU, i potem puść na GPU”. Start robi `czekaj_i_licz.ps1`: co minutę sprawdza, czy nie działa `horyzont.py`
i czy obciążenie karty jest niskie przez 3 kolejne minuty.

## Etap 1 (próba): czy mapa głębi widzi igłę jako sztywny prosty przedmiot

Trzy filmy, po 300 kolejnych klatek wokół okien z testu zawiasu. Dla każdej klatki:

- głębia na pikselach igły w funkcji odległości od końcówki wzdłuż osi; dopasowanie prostej (sztywna prosta igła
  musi mieć głębię liniową wzdłuż osi) i R² tego dopasowania;
- kontrast głębi igły wobec pierścienia tła tuż wokół niej (czy model w ogóle oddziela igłę od tła);
- odniesienie: tęczówka (rogówka bez źrenicy i bez igły) jako płaszczyzna oka; głębia igły względem niej
  w jednostkach rozrzutu głębi w obrębie rogówki;
- spójność w czasie: czy nachylenie głębi wzdłuż igły zmienia się płynnie między klatkami.

Kryterium przejścia do etapu 2: igła wyraźnie bliżej niż tło w większości klatek, liniowa głębia wzdłuż osi
(mediana R² ≥ 0,5) i nachylenie skorelowane między kolejnymi klatkami (≥ 0,5).

## Etap 2 (po etapie 1 i po wyniku przeglądu literatury)

Model ruchu sztywnej igły: zawias w nacięciu (punkt 3D), dwa kąty osi (w płaszczyźnie obrazu i nachylenie w głąb),
głębokość wsunięcia, obrót wokół własnej osi z zagiętym końcem. Obrót wokół osi przesuwa zagięty koniec w bok bez
ruchu trzonu, co może tłumaczyć wynik testu zawiasu.

## Zastrzeżenia nazwane przed pomiarem

- Mikroskop operacyjny patrzy z około 20 cm na pole kilku centymetrów, a komora przednia ma kilka milimetrów
  głębokości: rzut jest prawie równoległy, więc geometria prawie nie zdradza głębi. Model głębi zgaduje ją
  z cieniowania, rozmycia i wyuczonego kształtu scen naturalnych.
- Igła jest cienka i metaliczna (odblaski), a rogówka przezroczysta: dla takich obiektów modele głębi uczone na
  zdjęciach naturalnych bywają zawodne.
- Głębia względna ma nieznaną skalę w każdej klatce; bez skali nie da się podać kąta nachylenia w stopniach.
