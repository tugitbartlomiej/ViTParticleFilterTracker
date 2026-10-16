# Cel koloidu w uczeniu i fizyka ruchu (bieg s47, zdjęcie z historii eafb5a973be3d490)

806 checkpointów, kroki 0–65640. Token 226: heat 1.00 (środek ramki).

## 1. Cel: pułapka, ruch w uczeniu, kto zmniejsza błąd

### cząsteczka 226 (środek ramki)
- błąd: krok 0 4.7 → krok 16400 0.0937 → krok 65640 0.0479
- początek: pułapka w stronę celu cos +0.06 (dodatni 72%), pchnięcie obniża błąd w 90% (mediana 19.9% błędu); ruch w uczeniu wzdłuż −∇L cos +0.18 (dodatni 79%); zmiana błędu: koloid -3.07, cel -4.24, razem -4.6, współdziałanie +2.7
- środek: pułapka w stronę celu cos +0.02 (dodatni 57%), pchnięcie obniża błąd w 70% (mediana 8.7% błędu); ruch w uczeniu wzdłuż −∇L cos +0.12 (dodatni 74%); zmiana błędu: koloid -0.144, cel -0.335, razem -0.0377, współdziałanie +0.442
- koniec: pułapka w stronę celu cos +0.03 (dodatni 63%), pchnięcie obniża błąd w 76% (mediana 13.3% błędu); ruch w uczeniu wzdłuż −∇L cos +0.11 (dodatni 73%); zmiana błędu: koloid -0.118, cel -0.116, razem -0.00823, współdziałanie +0.226

### tokeny w ramce (8)
- błąd: krok 0 6.18 → krok 16400 0.382 → krok 65640 0.268
- początek: pułapka w stronę celu cos -0.00 (dodatni 49%), pchnięcie obniża błąd w 57% (mediana 3.3% błędu); ruch w uczeniu wzdłuż −∇L cos +0.12 (dodatni 74%); zmiana błędu: koloid -5.59, cel -6.82, razem -5.8, współdziałanie +6.61
- środek: pułapka w stronę celu cos +0.01 (dodatni 52%), pchnięcie obniża błąd w 64% (mediana 7.8% błędu); ruch w uczeniu wzdłuż −∇L cos +0.11 (dodatni 74%); zmiana błędu: koloid -1.27, cel -1.68, razem -0.0895, współdziałanie +2.86
- koniec: pułapka w stronę celu cos -0.01 (dodatni 46%), pchnięcie obniża błąd w 63% (mediana 8.0% błędu); ruch w uczeniu wzdłuż −∇L cos +0.10 (dodatni 72%); zmiana błędu: koloid -0.646, cel -0.868, razem -0.0245, współdziałanie +1.49

### tło (heat < 0,05)
- błąd: krok 0 0.00117 → krok 16400 0.000582 → krok 65640 4.7e-07
- początek: pułapka w stronę celu cos +0.03 (dodatni 68%), pchnięcie obniża błąd w 63% (mediana 40.1% błędu); ruch w uczeniu wzdłuż −∇L cos +0.00 (dodatni 29%); zmiana błędu: koloid -1.3, cel +1.57, razem -0.000584, współdziałanie -0.27
- środek: pułapka w stronę celu cos +0.00 (dodatni 62%), pchnięcie obniża błąd w 85% (mediana 2511.9% błędu); ruch w uczeniu wzdłuż −∇L cos +0.00 (dodatni 1%); zmiana błędu: koloid -0.0033, cel -0.00304, razem -0.000581, współdziałanie +0.00576
- koniec: pułapka w stronę celu cos +0.00 (dodatni 70%), pchnięcie obniża błąd w 81% (mediana 869.4% błędu); ruch w uczeniu wzdłuż −∇L cos +0.00 (dodatni 0%); zmiana błędu: koloid -0.00178, cel -0.00152, razem -5.3e-08, współdziałanie +0.0033

Cały obraz, całe uczenie: koloidy -9.08, cel (głowa) -7.43, razem -5.92, współdziałanie +10.6.

## 2. Zapadanie na podprzestrzeń głowy (losowo 5/192 = 2,6 %)
- początek: 5 głównych składowych 75.7%; podprzestrzeń odczytu głowy 14.1%; token 226 w podprzestrzeni odczytu 25.9%
- środek: 5 głównych składowych 75.1%; podprzestrzeń odczytu głowy 12.0%; token 226 w podprzestrzeni odczytu 23.7%
- koniec: 5 głównych składowych 76.2%; podprzestrzeń odczytu głowy 12.0%; token 226 w podprzestrzeni odczytu 25.1%

## 3. Relaksacja do położenia końcowego
- cząsteczka 226 (środek ramki): potęgowe d ~ t^-0.43 (R² 0.990); wykładnicze τ = 17576 kroków (R² 0.796)
- tokeny w ramce (8): potęgowe d ~ t^-0.44 (R² 0.991); wykładnicze τ = 17035 kroków (R² 0.807)
- tło (heat < 0,05): potęgowe d ~ t^-0.34 (R² 0.989); wykładnicze τ = 22837 kroków (R² 0.744)

## 4. Odpowiedź na siłę błędu z opóźnieniem (checkpointy co 100 kroków od 1100)
- 226: C(τ) dla τ = 0, 5, …, 30 checkpointów: [0.11999999731779099, 0.003000000026077032, 0.008999999612569809, -0.0, 0.004999999888241291, -0.0010000000474974513, -0.01600000075995922]; A(τ) dla τ = 1, 6, …, 26: [-0.40799999237060547, 0.0020000000949949026, 0.004000000189989805, -0.006000000052154064, -0.0020000000949949026, 0.014999999664723873]
- w ramce: C(τ) dla τ = 0, 5, …, 30 checkpointów: [0.10700000077486038, 0.004000000189989805, 0.004999999888241291, -0.0010000000474974513, 0.0, -0.0010000000474974513, -0.008999999612569809]; A(τ) dla τ = 1, 6, …, 26: [-0.40700000524520874, -0.003000000026077032, 0.003000000026077032, -0.00800000037997961, -0.0020000000949949026, 0.012000000104308128]

## 5. Dyfuzja w późnym uczeniu (kroki ≥ 20 000, co 100)
- 226: MSD ~ τ^0.13 (τ = 1–64 checkpointów po 100 kroków)
- w ramce: MSD ~ τ^0.13 (τ = 1–64 checkpointów po 100 kroków)
