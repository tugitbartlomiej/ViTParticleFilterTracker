# Koopman ze stanem z pamięcią: wynik (2026-09-25)

Pełne dane: 72 obrazy × 576 tokenów = 41 472 osobne operatory, każdy na pełnym wektorze 384 liczb, float64, bez PCA.
Podział identyczny jak w `koopman_osobno_obraz_token`: dopasowanie do kroku 245, test 24 przejść do kroku 365.
Czas: ok. 38 min na CPU. Skrypt `licz.py`, liczby `wyniki/REPORT.json`, tablice per jednostka `wyniki/*.npy`
(sumy w `wyniki/SHA256.json`).

## Wynik główny (zamrożony przed testem: rodzina wybierana osobno dla każdej jednostki na kalibracji)

| Model | RMSE 1 krok | RMSE 100 aktualizacji | cos przyrostu | jednostek przez bramkę |
|---|---|---|---|---|
| **Koopman z pamięcią, wybór na kalibracji** | **0,003708** | **0,0762** | 0,865 | **7 445 z 41 472** |
| ostatni przyrost | 0,003859 | 0,1170 | 0,864 | 0 |
| brak zmian | 0,008226 | 0,1362 | – | 0 |
| poprzedni Koopman na samym położeniu | 0,010331 | 0,2322 | 0,619 | 0 |

- Błąd jednego kroku spadł prawie trzykrotnie względem poprzedniego Koopmana i jest o 3,9 procent niższy niż
  powtórzenie ostatniego przyrostu.
- Na 100 aktualizacji naprzód błąd jest o 35 procent niższy niż ostatni przyrost i o 67 procent niższy niż
  poprzedni Koopman.
- Bramkę (≥5 procent lepiej od ostatniego przyrostu, ≥20 procent od braku zmian, cos ≥0,5) przechodzi 18 procent
  jednostek; poprzednio 0.
- Średnia przewaga 3,9 procent nie spełnia progu 5 procent dla całości; wynik ogłaszamy bez zmiany progu.

## Rodziny stosowane jednakowo do wszystkich jednostek (eksploracyjnie, po teście)

| Rodzina | RMSE 1 krok | vs ostatni | 100 aktualizacji | bramka |
|---|---|---|---|---|
| vel_1: α v + pełna korekta 384×384 z v | 0,003672 | +4,8 % | 0,0855 | 14 374 |
| vel_10 | 0,003692 | +4,3 % | 0,0712 | 10 637 |
| vel_100 | 0,003714 | +3,8 % | **0,0697** | 5 519 |
| diag_0.1: osobne tłumienie każdej współrzędnej | 0,003693 | +4,3 % | 0,0760 | 10 753 |
| alpha: jedno tłumienie | 0,003718 | +3,7 % | 0,0711 | 4 375 |
| ar2: dwa ostatnie przyrosty | 0,003733 | +3,3 % | 0,0730 | 4 914 |
| alpha kroczące | 0,003713 | +3,8 % | – | 5 630 |
| delay_1: stan [z, v] z położeniem | 0,005308 | −37,6 % | 0,0997 | 2 |

Wybór na kalibracji najczęściej brał ar2 (18 293 jednostek), vel_1 (7 219) i vel_10 (6 892); rodzina z położeniem
prawie nigdy (2 jednostki).

## Wnioski

1. **Pamięć naprawia Koopmana.** Stan musi zawierać ostatni krok. Wtedy operator przewiduje ruch latentu podczas
   uczenia lepiej niż wszystkie proste kontrole, na jednym kroku i na 100 aktualizacjach.
2. **Położenie szkodzi.** Dodanie samego z_t do stanu (rodzina delay) psuje prognozę o 38–100 procent, zgodnie
   z diagnozą: położenie wymusza ekstrapolację poza znaną trajektorię.
3. **Pełna macierz na prędkości pomaga, ale mało.** Korekta 384×384 do tłumienia daje dodatkowy procent na jednym
   kroku; na długim horyzoncie wygrywa silniej ściśnięta wersja (vel_100). Większość zysku daje samo tłumienie.
4. Przewidywalny jest tempo i kierunek dalszego ruchu, nie nowy kierunek: cos przyrostu prawie nie rośnie
   (0,864 → 0,867).

## Ograniczenia

Znana wcześniej pula; porównania rodzin po teście są eksploracyjne. Brak sprzężeń między tokenami w operatorze.
Dobra prognoza przebiegu uczenia nie mówi jeszcze, które klatki poprawią detektor.
