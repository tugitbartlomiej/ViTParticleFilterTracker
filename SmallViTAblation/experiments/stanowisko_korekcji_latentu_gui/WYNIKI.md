# Stanowisko korekcji latentu: pierwszy pomiar (2026-10-08)

Zakładka „Korekta latentu: cel i dynamika” w aplikacji (`app/zakladka_korekty.py`, silnik `app/korekta_latentu.py`,
modele `app/dynamika_korekty.py`, gałęzie `app/galezie_uczenia.py`). Specyfikacja: `SPECYFIKACJA_DLA_CLAUDE_CODE.md`.
Testy akceptacji: `testy/test_stanowisko.py`, 18/18 (`testy/RAPORT_TESTOW.json`). Pomiar: `kod/pierwszy_pomiar.py`,
wynik `outputs/stanowisko_korekcji_latentu_gui/20261008_001815_pierwszy_pomiar/PODSUMOWANIE.json` (6 minut, RTX 3070).
Wszystkie korekty to oracle: optymalizacja aktywacji jednego obrazu z jego etykietą przy zamrożonym ViT.

## 1. Kontrole poprawności

- Zerowa interwencja odtwarza pełny forward dokładnie (błąd 0) we wszystkich 5 trybach; hash wag bez zmian.
- Tryby wejścia MLP zgodne z prawdziwą siecią z hakiem (gałąź: wejście norm2; wspólne: residual) do 2e-5.
- fc2: W2 [192 x 768] ma rząd 192 (tolerancja 1,0e-4), jądro 576. Zaburzenie z jądra o normie całego latentu obrazu
  zmienia wyjście o 1e-6, zaburzenie tej samej normy z przestrzeni wierszowej o 0,53. W2 ograniczona do J128 ma rząd
  128 i zerowe jądro.
- AP 400 obrazów z DETECTIONS_FULL400 tym ewaluatorem = RESULT.json co do 1e-6 (0,437555 / 0,884748 / 0,873660).

## 2. Korekta oracle i fc2

Tryb „zakresy z checkpointu”, J128, wszystkie tokeny, Adam lr 0,3, 200 kroków, zapis co krok.

| ckpt, obraz | IoU najpewniejszej | strata | niewidoczna dla fc2 | kontrola izotropowa (strata) |
|---|---|---|---|---|
| k1000, 5 | 0,541 -> 0,750 | 0,855 -> 0,457 | 72,9 % | 0,91-0,97 |
| k1000, 60 | 0,593 -> 0,999 | 1,064 -> 0,258 | 72,7 % | 1,03-1,08 |
| k1000, 250 | 0,716 -> 0,956 | 1,738 -> 0,245 | 71,4 % | 1,56-1,61 |
| k500, 5 | 0,372 -> 0,707 | 0,972 -> 0,590 | 73,3 % | 1,17-1,26 |
| k500, 60 | 0,476 -> 0,736 | 1,430 -> 0,598 | 73,5 % | 1,40-1,45 |
| k500, 250 | 0,384 -> 0,745 | 1,553 -> 0,596 | 73,0 % | 1,79-1,93 |

- Około 72 % sumy kwadratów korekty leży w jądrze fc2, czyli końcówka jej nie widzi. Historyczne korekty J128 też:
  72-74 % na 6 obrazach.
- Historyczne cele J128 (tylko dolne ograniczenie GELU) wychodzą poza zakresy z checkpointu na 0,9-2,5 %
  współrzędnych, z przekroczeniem do 58. Nowe korekty w trybie „zakresy” mieszczą się w nich z definicji.
- Losowe kierunki o tej samej normie nie poprawiają albo pogarszają stratę. Losowe znaki z modułami oracle
  poprawiają częściowo, ale ta kontrola zależy od etykiety.

## 3. Uczenie a korekta (dwie osie czasu)

- Ruch uczenia jest prawie prostopadły do korekty oracle. Prędkość uczenia tuż przed k_c ma kosinus z korektą
  −0,017 do +0,030. Przy k500 przyszły ruch z(k) − z(500) do k1000 ma kosinus od −0,045 do +0,015, a współczynnik
  rzutu α wynosi |α| < 0,008. Ten pomiar jest retrospektywny.
- Prawdziwe uczenie poprawia te obrazy (IoU przy k500 -> k1000: 0,37 -> 0,54, 0,48 -> 0,59, 0,38 -> 0,72), ale innym
  kierunkiem niż najlepsza znaleziona korekta. Uczenie zmienia też wagi i residual, więc ten sam efekt nie musi
  wymagać ruchu latentu w kierunku oracle.

## 4. Modele dynamiki (rel = błąd / błąd persystencji, część testowa)

- Tor korekty (latent_optimization): stała prędkość, Langevin i GLE dają 0,02-0,07 przy h = 1 i 0,09-0,42 przy
  h = 20. Koopman (Hankel-DMD, r = 8) jest gorszy (0,18-0,30 przy h = 1, do 1,2 przy h = 20).
- Wejście optymalizatora: przy swobodnej korekcie krok Adama z zapisanych gradientów przewiduje ruch co do 4e-12
  (test). W trybie „zakresy” nie pomaga (z L = 5: 0,0222 -> 0,0225), a bez pamięci szkodzi (do 2,7), bo rzut na
  zakresy obcina krok w innych współrzędnych niż na części treningowej.
- Tor uczenia (network_training, co 10 kroków, J128 600 tokenów): Langevin i GLE 0,80-0,88 przy h = 1, 0,71-0,77
  przy h = 5. Stała prędkość zawodzi (2,6 przy h = 1, 170-200 przy h = 20): ruch co 10 kroków nie jest gładki.

## 5. Gałęzie uczenia (k = 500, H = 20, 3 przyszłe sekwencje paczek)

- Niedeterminizm GPU: dwie identyczne gałęzie bez interwencji rozchodzą się. ||Δz|| sond wynosi 0,17 po 2 krokach,
  0,86 po 5 i 4,0 po 20, czyli tyle co odpowiedź na interwencję (4,0 po 20). W trybie deterministycznym (atencja MATH,
  deterministyczne algorytmy) różnica wynosi dokładnie 0. To prawdopodobnie główne źródło dryfu powtórki uczenia
  (9 % ruchu wag w 10 krokach).
- Waga obrazu w slocie 0 × 1,5: po 1 kroku ||Δz|| = 0,25 przy liniowości 0,03 (2ε daje prawie dokładnie 2 razy więcej).
  Po 5 krokach liniowość wynosi 0,61, po 20 krokach 0,81. Odpowiedź przestaje być liniowa po kilku krokach.
- Podmiana slotu 0 na obraz m0005: zmiana straty tego obrazu po 5 krokach od −0,065 do +0,012 w zależności od
  przyszłych paczek. Znak zależy od przyszłości.
- Zaburzenie wag ε = 1e-4: d(20)/d(1) = 22 (prawdziwe paczki) i 17 (inna sekwencja); ε = 1e-2: 1,5-1,6. To wzrost
  odchylenia reprezentacji w tokenach sond, nie wykładnik Lapunowa.

## 6. Obserwator Kalmana (klatka próbna 0, 20 tokenów narzędzia, co krok, test kroki 601-1000, r = 6)

| pomiar | koszt | błąd modów: otwarta pętla / prior / posterior / sam pomiar | pokrycie 90 %: prior / posterior |
|---|---|---|---|
| 25 % tokenów co krok | 0,25 | 12,8 / 2,05 / 1,84 / 1,80 | 0,80 / 0,65 |
| 10 % tokenów co krok | 0,10 | 12,8 / 3,56 / 3,39 / 3,34 | 0,65 / 0,51 |
| 25 % tokenów co 5 | 0,05 | 12,8 / 2,78 / 2,42 / 1,79 | 0,92 / 0,81 |
| 25 % tokenów co 20 | 0,0125 | 12,8 / 4,39 / 4,19 / 1,81 | 0,97 / 0,88 |

- Przy pomiarze co krok filtr nie poprawia samego pomiaru, bo model AR(2) w modach jest słaby.
- Przy rzadkim pomiarze filtr daje 1/3 błędu otwartej pętli przy 1-5 % kosztu.
- Przedziały po obserwacji są za wąskie (0,51-0,88 zamiast 0,90).
- W pełnym latencie błąd zdominowany jest przez ruch poza 6 modami części treningowej: rzut daje 21,9 wobec
  persystencji 22,5.

## Czego to nie dowodzi

- Jeden checkpoint (k1000), jeden bieg uczenia, 3 obrazy korekty i jedna klatka obserwatora.
- Korekty oracle nie dowodzą istnienia zdjęć, które je realizują. Nie dowodzą też generalizacji.
- Prostopadłość ruchu uczenia do korekty dotyczy tej korekty (J128, zakresy), a nie każdej możliwej.
