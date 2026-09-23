# Eksperyment 2. Koopman na przebiegu uczenia, w latencie

Status: plan, zapisany 2026-09-23. Nic nie uruchomiono. Trening wymaga osobnej zgody użytkownika.

Przegląd wszystkich trzech eksperymentów: `docs/Analiza_ponownej_publikacji/46_PLAN_TRZECH_EKSPERYMENTOW_DOBORU_KLATEK.md`.

## Pytanie

Czy z tego, jak latent każdej klatki uczącej zmienia się w pierwszych krokach uczenia, da się przewidzieć,
które klatki sieć już opanowała, a które dalej ją czegoś uczą? I czy wybór tych drugich przyspiesza trening
oraz poprawia detektor?

## Skąd pomysł

Dogra i Redman (2020, „Optimizing neural networks via Koopman operator theory”) traktują wagi sieci jako układ
dynamiczny: wagi zmieniają się krok po kroku, a Koopman przewiduje, dokąd zmierzają. Pozwala to przeskoczyć część
uczenia albo porównać metody uczenia. Tutaj robimy to samo, ale nie na wagach, tylko w latencie: układem
dynamicznym jest latent klatki w funkcji kroku uczenia.

Pokrewny wynik bez Koopmana to mapa zbioru danych (Swayamdipta i in. 2020, „Dataset Cartography”): przykłady,
na których pewność modelu waha się między epokami, najbardziej poprawiają działanie na nowych danych.

Dlaczego to ma sens tutaj: trening RF-DETR Nano kończy się realnie w 6–7 epoce. Potem strata uczenia spada dalej
(2,6 do 1,1), a strata walidacji rośnie (3,2 do 3,7) i mAP spada (0,685 do 0,650). Na zbiorze testowym ostatnia
epoka jest gorsza od najlepszej (0,654 wobec 0,681), a fałszywe alarmy rosną z 40 do 50 procent. Całą informację
sieć czerpie więc z pierwszych epok i to je warto opisać.

## Dane

Obecna sonda (`pipeline/koopman_langevin/sonda_treningu.py`) mierzy 32 klatki walidacyjne w 6 wariantach, a nie
klatki uczące. Do tego eksperymentu potrzebna jest sonda na klatkach **uczących**:

- po każdym kroku uczenia (albo co kilka kroków) dla każdej z 250 klatek uczących: wektor tożsamości w tokenach
  backbone (kontrast pod ramką etykiety), pewność i IoU najlepszej detekcji, latent zapytania dekodera,
  strata klatki;
- dodatkowo te same wielkości dla stałej grupy klatek walidacyjnych, żeby łączyć zmiany w uczących z wynikiem.

Bieg: 250 klatek losowych, 10 epok (wystarczy, bo plateau jest w 6–7), trzy ziarna. Możliwy lokalnie na RTX 3070
(Nano, 384, mała paczka) albo na Edenie.

## Metoda

1. Dla każdej klatki trajektoria latentu z(krok), kroki od 0 do końca 2. epoki (okno uczące Koopmana).
2. Koopman wspólny dla wszystkich klatek: z(k+1) = K z(k), dopasowany najmniejszymi kwadratami z grzbietem.
   Wartości własne K mówią, jak szybko poszczególne kierunki latentu się ustalają (bliskie 1 = wolno).
3. Dla każdej klatki: udział jej trajektorii w wolnych modach i reszta przewidywania. Klatka „opanowana” szybko
   zbiega do punktu stałego. Klatka „ucząca” ma duży udział wolnych modów albo dużą resztę.
4. Langevin: reszta przewidywania rozdzielona na część z historii (pamięć) i szum mini-paczek; za „uczącą”
   uznajemy tylko klatkę, której reszta przekracza szum typowy dla jej kroków.
5. Przyspieszenie w duchu Dogry i Redmana: z trajektorii z pierwszych 2 epok przewidzieć latent klatek po 6 epoce.
   Miara: R² przewidywania latentu i zgodność przewidzianej pewności z rzeczywistą.

## Protokół

Etap A, offline po jednym biegu z sondą:

- Czy wolne mody i reszta z pierwszych 2 epok przewidują, które klatki mają dużą stratę w epoce 6 (AUC)?
- Porównanie z mapą zbioru danych (średnia i rozrzut pewności między epokami) i z liczbą zapomnień
  (Toneva i in. 2019), bo to prostsze miary tego samego.
- Czy przewidywanie latentu po 6 epoce z 2 epok ma R² > 0,8 na klatkach niewidzianych przy dopasowaniu K?

Etap B, trening (po zgodzie):

- Pula: ta sama co w `maly_budzet`. Ze wstępnego biegu na 250 klatkach bierzemy miarę „uczących” i rozszerzamy
  ją na pulę przez najbliższych sąsiadów w latencie albo przez krótki bieg na większym podzbiorze.
- Ramiona po 250 klatek, trzy ziarna: Koopman–Langevin, mapa zbioru danych, losowe.
- Miary: mAP najlepszej epoki, **epoka, w której osiągnięto 95 procent końcowego mAP** (przyspieszenie),
  mAP na teście raz na końcu.

## Czego ten eksperyment nie dowodzi

- Koopman na przebiegu uczenia opisuje, jak sieć się uczy na danym zbiorze. Przeniesienie miary z 250 klatek na
  całą pulę jest dodatkowym założeniem, które etap B sprawdza tylko pośrednio.
- Dogra i Redman pracowali na wagach. Wersja w latencie jest naszym rozszerzeniem, bez gotowego wzorca w literaturze.
