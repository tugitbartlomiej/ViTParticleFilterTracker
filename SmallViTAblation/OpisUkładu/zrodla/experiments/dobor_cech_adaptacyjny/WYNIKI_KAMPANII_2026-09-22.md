# Wyniki kampanii adaptacyjnego doboru klatek (Eden, 2026-09-22)

Zapisane 2026-09-23 na polecenie użytkownika. Kampania skończyła się 22.09 wieczorem, ale do 23.09 nikt nie pobrał
ani nie opisał jej wyników (plik `WYNIKI.json` nadal podawał stan „w kolejce”).

## Co porównano

Pięć ramion po trzy ziarna (42, 1337, 2024), RF-DETR Nano, 10 epok, w każdej epoce 1500 klatek wybieranych na nowo:

- `random` – losowo,
- `glister` – GLISTER: klatki, których gradient najbardziej zmniejsza stratę na zbiorze sterującym D,
- `features_glister` – najpierw potrzeby z cech obrazu (rozmiar, jasność, rozmycie, położenie ramki), potem GLISTER,
- `features_kg_glister` – jak wyżej, z dodatkowym priorytetem z Koopmana–GLE,
- `features_placebo_glister` – kontrola: cechy pozorne, ta sama procedura.

Miara: mAP 0,5:0,95 na oficjalnej walidacji (25 operacji). Zbiór testowy nie był używany.

## Wynik

| Ramię | najlepsza epoka (średnia ± odch.) | ostatnia epoka | różnych klatek w uczeniu |
|---|---|---|---|
| random | **0,7131 ± 0,0042** | 0,7013 | ok. 7400–7500 |
| glister | 0,7072 ± 0,0017 | 0,7002 | ok. 7430–7490 |
| features_placebo_glister | 0,6941 ± 0,0033 | 0,6738 | ok. 6000–6500 |
| features_glister | 0,6903 ± 0,0083 | 0,6695 | ok. 5900–6300 |
| features_kg_glister | 0,6839 ± 0,0095 | 0,6807 | ok. 5900–6300 |

mAP w kolejnych epokach (walidacja):

```
random_s42                     0.657 0.702 0.687 0.690 0.701 0.707 0.668 0.703 0.705 0.705
random_s1337                   0.673 0.699 0.694 0.700 0.704 0.710 0.681 0.681 0.716 0.698
random_s2024                   0.645 0.690 0.686 0.672 0.705 0.682 0.716 0.692 0.707 0.700
glister_s42                    0.678 0.674 0.686 0.691 0.665 0.696 0.695 0.697 0.707 0.699
glister_s1337                  0.678 0.651 0.672 0.680 0.676 0.665 0.693 0.687 0.709 0.700
glister_s2024                  0.667 0.689 0.696 0.691 0.675 0.683 0.701 0.705 0.695 0.702
features_glister_s42           0.663 0.688 0.662 0.646 0.648 0.675 0.663 0.678 0.682 0.685
features_glister_s1337         0.671 0.649 0.662 0.653 0.653 0.676 0.678 0.670 0.682 0.646
features_glister_s2024         0.661 0.686 0.648 0.674 0.701 0.691 0.693 0.661 0.687 0.677
features_kg_glister_s42        0.664 0.641 0.683 0.675 0.680 0.695 0.679 0.665 0.663 0.685
features_kg_glister_s1337      0.681 0.658 0.673 0.655 0.635 0.673 0.681 0.663 0.683 0.686
features_kg_glister_s2024      0.668 0.651 0.663 0.636 0.653 0.637 0.641 0.670 0.671 0.671
features_placebo_glister_s42   0.672 0.665 0.663 0.656 0.669 0.670 0.682 0.693 0.686 0.678
features_placebo_glister_s1337 0.677 0.669 0.652 0.685 0.654 0.684 0.670 0.691 0.682 0.668
features_placebo_glister_s2024 0.642 0.666 0.615 0.656 0.645 0.699 0.634 0.679 0.679 0.675
```

## Wnioski

- **Żadna metoda doboru nie pokonała losowania.** Najlepsze jest ramię losowe; GLISTER jest o 0,006 niżej.
- **Dobór według cech szkodzi.** Wszystkie trzy ramiona z cechami są 0,02–0,03 poniżej losowego.
- **Prawdziwe cechy nie są lepsze od pozornych** (0,690 wobec 0,694), więc strata nie wynika z treści cech,
  tylko z samego mechanizmu potrzeb.
- **Koopman–GLE nie pomaga** (0,684 wobec 0,690 bez niego), w granicach rozrzutu ziaren.
- Najbardziej prawdopodobna przyczyna: ramiona z cechami wracają do podobnych klatek i przez 10 epok widzą
  około 6 tys. różnych klatek wobec około 7,5 tys. w losowym. Różnorodność wygrywa, jak w pozostałych próbach
  doboru (K6 po kwadratach 0,6296 wobec losowych 0,6319 przy 250 klatkach).

## Czego to nie dowodzi

- Wynik dotyczy tej implementacji i tych cech (tylko cechy z ramki i jasności; bez kąta i zasłonięcia).
- Trzy ziarna i jeden protokół (10 epok, adaptacyjnie po 1500 klatek). Wartości nie są wprost porównywalne
  z biegami static36 (0,682), które uczyły się na stałym zestawie 1500 klatek.

## Pliki

- Lokalna kopia metryk, list klatek i wag ostatniej (10.) epoki każdego biegu:
  `results/eden_dobor_cech_adaptacyjny_2026-09-22/<ramię>_s<ziarno>/rounds/epoch_010/weights.pth`.
- Pełne stany treningu (`*.ckpt`) i wagi pośrednich epok usunięte z Edena 2026-09-23 na polecenie użytkownika,
  po sprawdzeniu lokalnej kopii ostatniej epoki.
