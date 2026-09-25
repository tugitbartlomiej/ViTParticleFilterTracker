# Koopman ze stanem z pamięcią, osobno dla każdego obrazu i tokenu

Data: 2026-09-25. Plan zapisany przed uruchomieniem. Polecenie użytkownika: „wylicz to poprawnie” po diagnozie
`koopman_osobno_obraz_token/DIAGNOZA_PRZYCZYNY.md`.

## Dlaczego nowa wersja

Poprzedni operator działał na samym położeniu wektora z_t. Diagnoza pokazała, że następny krok zależy głównie od
poprzedniego kroku (bezwładność optymalizatora, cos kolejnych przyrostów 0,90), a długość kroków maleje w czasie.
Stan bez pamięci nie jest domknięty, a stała macierz na położeniu ekstrapoluje średni dryf z wcześniejszego okna.

Poprawny stan Koopmana to stan z opóźnieniem: s_t = [z_t, v_t], gdzie v_t = z_t − z_{t−1}. Dla wektorów kolumnowych

```
z_{t+1} = z_t + v̂_{t+1},   v̂_{t+1} = A v_t (+ B v_{t−1} albo + C z_t)
K = [[I, A], [0, A]]  (rodzina z samą prędkością)
```

Rodziny operatora A (każda osobno dla pary obraz–token, bez mieszania obrazów i pozycji):

| nazwa | postać | parametrów na jednostkę |
|---|---|---|
| alpha | v̂ = α v_t | 1 |
| ar2 | v̂ = a v_t + b v_{t−1} | 2 |
| diag(λ) | v̂ = diag(a_i) v_t, a_i ściągane do α siłą λ ∈ {0,1; 1; 10} | 384 |
| vel(ρ) | v̂ = α v_t + korekta liniowa z v_t (grzbiet dualny, ρ ∈ {1; 10; 100}) | pełna 384×384 |
| delay(ρ) | v̂ = średnia + liniowa funkcja [z_t, v_t] (grzbiet dualny, ρ ∈ {0,1; 1; 10; 100}) | pełna 384×768 |

## Podział (identyczny z koopman_osobno_obraz_token)

- Dane: `koopman_gesta_historia/wyniki/macierze_wiele_obrazow`, 72 obrazy × 576 tokenów × 73 checkpointy (kroki 5..365).
- Kalibracja: dopasowanie na stanach 0..40, prognozy jednego kroku z początków 40..47. Wybór rodziny i parametru
  osobno dla każdej jednostki, tylko na tych błędach.
- Test: dopasowanie na stanach 0..48 (do kroku 245), zamrożenie, 24 przejścia z początków 48..71.
  Każda prognoza dostaje prawdziwe z_t i v_t na początku.
- Dodatkowo wersja krocząca (tylko alpha, ar2): współczynniki przeliczane przed każdym początkiem z danych
  dostępnych do tego początku.
- Długi horyzont: 20 checkpointów (100 aktualizacji) od początków 48, 50, 52, bez prawdziwych stanów pośrednich.

## Kontrole i bramka (bez zmian względem poprzedniego eksperymentu)

Brak zmian, ostatni przyrost. Bramka jednostki: RMSE ≤ 0,95 × RMSE ostatniego przyrostu, ≤ 0,8 × RMSE braku zmian,
średni cos przyrostu ≥ 0,5. Raportujemy każdą rodzinę stosowaną jednakowo do wszystkich jednostek oraz wariant
z wyborem rodziny na kalibracji. Progi nie zmieniają się po wyniku.

## Czego to nie dowodzi

Znana wcześniej pula. Dobra prognoza przebiegu uczenia nie oznacza, że wskazuje, które klatki poprawią detektor.
