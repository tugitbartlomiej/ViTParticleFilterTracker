# Predykcje P1–P7 zapisane przed pomiarem (2026-10-23, 02:17, przed uruchomieniem skryptów)

Źródło predykcji: `reports/Tezy artykułu o płaszczyźnie krok × podkrok.md` (sekcja 4), sformułowane na biegu s47 (klatka TRAIN 0).
Tu dostosowane do siatki checkpointów seedu 3047 (46 checkpointów: kroki 0, 300, 1000, 3000, 6000, 10 000, 16 000, 24 000, 33 000,
44 000, 55 000, 65 640 i co 10 epok do 437 500; epoka = 1093,75 kroku) i do 10 klatek TEST. Progi liczbowe są ustalone tutaj
i zakodowane w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`; po pomiarze nie będą zmieniane. Etykieta narzędzia:
cystotom (kategoria, na której trenowano detektor), udział pikseli obrysu w łacie ≥ 0,25 (s47 miał 0,5 przy 3 łatach; tu próg niższy,
bo cystotom jest cienki, a wymagamy ≥ 8 łat), tło = udział 0, łaty mieszane pominięte.

- **P1 (kolaps uwagi przed detekcją).** Niech k* = pierwszy checkpoint, w którym średnie IoU na 10 klatkach przekracza 0,3.
  W k* mediana po klatkach efektywnej liczby źródeł uwagi (1/ΣA², mediana po tokenach, średnia po głowach) w blokach 4 i 5 jest < 100,
  a w kroku 1000 < 50.
- **P2 (gwiazda na narzędziu).** Dla checkpointów ≥ 10 000 (≥ 9 ep) i bloków 4–5: trzy tokeny o największej odbieranej uwadze
  (średnia A_ij po zapytaniach i głowach) leżą w obrysie narzędzia (udział > 0) albo ≤ 1 łatę od niego w ≥ 90 % par (klatka, checkpoint,
  blok). Dodatkowo wzbogacenie tło → narzędzie w blokach 4 i 5 wynosi ≥ 10 w ≥ 8 z 10 klatek na każdym checkpoincie ≥ 10 000.
- **P3 (ruch prosty, potem zygzak).** Wskaźnik balistyczny 12 podkroków (mediana po tokenach, średnia po klatkach): w kroku 0 w
  przedziale [0,8; 1,2]; maksimum na krokach {300, 1000, 3000} ≥ 2,0; wartość ≤ 1,2 na wszystkich checkpointach ≥ 24 000 (≥ 21,9 ep).
- **P4 (złożenie siodeł nie kurczy).** Dla każdej (klatka, checkpoint, token) z jakobianów: udział kierunków o σ < 1 iloczynu 12 map
  własnych ∈ [0,45; 0,55]; średnie po 5 parach bloków nakładanie podprzestrzeni kurczących minus wartość losowa ≤ 0,03.
- **P5 (front rozdziału w bloku 3–4, blok 6 nie podnosi).** Dla każdego checkpointu ≥ 24 000: największy przyrost Fishera między
  kolejnymi stanami przypada na podkrok bloku 3 lub 4 (u3, m3, u4, m4) w ≥ 6 z 10 klatek; Fisher po MLP 6 ≤ 1,05 · Fisher po uwadze 6
  w ≥ 6 z 10 klatek. Oba warunki muszą zachodzić w ≥ 80 % checkpointów ≥ 24 000.
- **P6 (uwaga nie wygładza po siatce).** Mediana masy uwagi na 8 sąsiadach siatki < 0,026 (2 × jednostajna) dla każdego bloku,
  checkpointu i klatki; na każdym checkpoincie ≥ 1000 suma zmian energii Dirichleta w 6 podkrokach MLP (średnia po klatkach) jest ujemna
  i mniejsza niż suma zmian w 6 podkrokach uwagi.
- **P7 (przejście ściska, narzędzie mocniej).** Σ_s log|det J_s| po 12 podkrokach (średnia po tokenach narzędzia i klatkach) < 0 na
  wszystkich checkpointach jakobianów ≥ 10 000; na checkpointach jakobianów ≥ 24 000 wartość dla narzędzia jest mniejsza niż dla tła
  w ≥ 6 z 10 klatek.

Checkpointy jakobianów (10): 0, 300, 1000, 3000, 10 000, 24 000, 65 640, 109 375 (100 ep), 218 750 (200 ep), 437 500 (400 ep).
Tokeny jakobianów na klatkę: 3 narzędzia (największy udział) i 3 tła (losowe, ziarno = numer klatki, ≥ 5 łat od obrysu, wnętrze siatki);
pełny jakobian uwagi dla pierwszego tokenu narzędzia i pierwszego tokenu tła.

Werdykt: PASS, gdy wszystkie warunki predykcji spełnione; FAIL w przeciwnym razie, z podaniem liczb. Wynik negatywny P4 byłby
istotniejszy niż pozytywny (ogniskowanie naprzemienne).
