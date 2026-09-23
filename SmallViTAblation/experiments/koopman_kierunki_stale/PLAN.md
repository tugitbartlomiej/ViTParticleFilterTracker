# Eksperyment 4. Klatki, w których zwykle stałe kierunki Koopmana mocno się zmieniają

Status: zakończony 2026-09-23. **Wynik negatywny: wybrane klatki dają gorszy detektor niż losowe z tej samej puli.**

## Wynik treningu (Eden, węzeł hopper, H100, 2026-09-23)

Sześć biegów, `eden/run_maly_budzet.slurm`, protokół `protocol_maly_budzet.yaml` (wczesne zatrzymanie),
mAP 0,5:0,95 na walidacji (25 operacji), wagi najlepszej epoki oceniane niezależnie:

| Ramię | s42 | s1337 | s2024 | średnia ± odch. | ostatnia epoka | epoka szczytu |
|---|---|---|---|---|---|---|
| kierunki stałe Koopmana | 0,6708 | 0,6704 | 0,6675 | **0,6696 ± 0,0015** | 0,6245 | 5, 8, 7 |
| losowe z tej samej puli | 0,6810 | 0,6822 | 0,6829 | **0,6820 ± 0,0008** | 0,6600 | 8, 6, 7 |

- Wybór Koopmanem przegrywa o 0,012 na każdym z trzech ziaren; różnica jest ok. 10 razy większa niż rozrzut ziaren.
- Po szczycie przeucza się szybciej: w ostatniej epoce 0,625 wobec 0,660.
- Ma wyższą czułość (0,87–0,89 wobec 0,80–0,86), ale niższą precyzję (0,92–0,93 wobec 0,94–0,95):
  więcej trafionych narzędzi, ale też więcej fałszywych ramek.
- Losowe z puli (0,682) wypadają tak samo jak dawne losowe z całego zbioru (static36 random_b1500, 0,682).

Wniosek: zestaw złożony z klatek trudnych (18 procent chybień wstępnego detektora wobec 11 procent w losowym)
uczy gorzej niż zestaw typowy. Jest spójny z pozostałymi próbami doboru (K6 po kwadratach przy 250 klatkach,
kampania doboru cech z GLISTER): na tym zbiorze żadna metoda wybierania trudnych albo „brakujących” klatek nie
pokonała losowania. Metryki sześciu biegów: `wyniki/biegi/`. Wagi zostały na Edenie w
`results_maly_budzet/ks_*`.

## Pomysł (użytkownik, 2026-09-23)

Koopmanem wyznaczamy kierunki latentu, które z klatki na klatkę zwykle się nie zmieniają (kierunki stałe).
Sprawdzamy, czy siedzi w nich tożsamość narzędzia. Potem bierzemy klatki, o których wiemy z adnotacji, że jest
na nich cystotom, a mimo to kierunki stałe mocno się w nich zmieniają. Takie klatki dokładamy do uczenia
i sprawdzamy, czy poprawiają trening. Budżet: 1500 klatek.

## Jak liczone (`wybor.py`)

- Latent: detektor ut_b0500, tokeny backbone w ramce etykiety cystotomu, opis 3 × 3 kawałków po 32 liczby,
  wybielony do 48 wymiarów. Pula: 10 637 klatek uczących tracking_static36, te same obrazy są na Edenie.
- Koopman (EDMD) na parach klatek tego samego filmu odległych o najwyżej 30 klatek (6761 par).
- Kierunki stałe: |λ| ≥ 0,85, wyszło 6 z 48 (|λ| od 0,85 do 0,965).
- Wynik klatki: błąd przewidywania Koopmana tylko w kierunkach stałych, każdy kierunek w jednostkach swojego
  typowego szumu (poprawka Langevina), większy z porównań z poprzednią i następną klatką.
- Wybór: od największego wyniku, bez klatek bliższych niż 10 klatek od już wybranej w tym samym filmie.

## Wyniki offline (bez uczenia sieci)

Czy tożsamość siedzi w kierunkach stałych (R² przewidywania kontrastu wektora tożsamości, połowa klatek na
dopasowanie, połowa na ocenę):

| Kierunki | R² |
|---|---|
| 6 stałych | 0,25 |
| 6 najszybszych | 0,00 |
| 6 losowych (mediana z 20) | 0,12 |
| wszystkie 48 | 0,59 |

Tożsamość jest więc wyraźnie bardziej w kierunkach stałych niż w szybkich, ale nie cała.

Czy wynik wskazuje chybienia ut_b0500 na klatkach puli, których ten detektor nie widział (3128 klatek, 344 chybienia):

| Miara | AUC |
|---|---|
| zmiana w kierunkach stałych | 0,64 |
| zmiana w kierunkach szybkich | 0,52 |
| zwykła zmiana latentu | 0,64 |
| słaby wektor tożsamości | 0,78 |
| niska pewność detektora | 0,89 |

Wybrane 1500 klatek zawiera 18 procent chybień ut_b0500 (wśród klatek, których nie widział), losowe 1500 z tej
samej puli 11 procent. Wybór jest więc wyraźnie bogatszy w trudne klatki. Kierunki stałe nie wskazują chybień
lepiej niż zwykła zmiana latentu.

Czułość na odstęp par: przy 6 klatkach tylko 1809 par i 1484 wybrane klatki (za mało sąsiadów), przy 60 wynik
jak przy 30 (AUC 0,65, chybień w wyborze 20 procent).

## Plan treningu (czeka na zgodę)

- Ramiona, każde na ziarnach 42, 1337, 2024:
  - `tracking_koopman_stale_b1500_sel42` – 1500 klatek z tej metody,
  - `tracking_losowe_z_puli_b1500_sel42` – 1500 losowych z tej samej puli 10 637 (uczciwa kontrola,
    bo pula jest sumą klatek wybranych wcześniej przez inne strategie).
- Skrypt `eden/run_maly_budzet.slurm`, protokół `configs/protocol_maly_budzet.yaml` (wczesne zatrzymanie).
  Oba ramiona w tym samym protokole.
- Odniesienie: dawne biegi random_b1500 (walidacja 0,682 w najlepszej epoce, ziarno 42), liczone w innym protokole.
- Miary: mAP 0,5:0,95 walidacji w najlepszej epoce, epoka szczytu, mAP na teście raz na końcu (`ocena_test.py`).
- Kryterium: przewaga nad losowymi z puli większa niż rozrzut między ziarnami na wszystkich trzech ziarnach.
- Koszt: 6 biegów po około godzinę na jednej karcie, około 1,5 GB wyników.

## Czego to nie dowodzi

- Wybór używa etykiet (ramka cystotomu wyznacza miejsce opisu), więc to kuracja oznaczonego zbioru, a nie
  wybieranie klatek do oznaczenia.
- Więcej trudnych klatek w zestawie nie gwarantuje lepszego detektora; trudne klatki mogą też być
  zaszumionymi adnotacjami. Rozstrzyga trening.
