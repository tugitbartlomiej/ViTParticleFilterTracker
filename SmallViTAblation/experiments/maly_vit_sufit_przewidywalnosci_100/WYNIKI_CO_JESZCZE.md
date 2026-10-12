# Wynik: czy lepsze równanie z samej historii da 85 procent na 100 kroków

Data: 2026-10-03. Trzy pomiary na istniejących danych, bez treningu. Kod w `kod/`, liczby w `wyniki/`.

## Werdykt

Nie. Po rozgrzewce żadne równanie z samej przeszłości latentu nie dojdzie do 85 procent na 100 kroków.
Najlepsze z historii: 0,23–0,35 (MSE). Przyczyna leży w danych: przyrosty latentu pamiętają tylko około
10 kroków, a dalej latent błądzi losowo, napędzany losowym składem paczek. Wspólny ruch wszystkich tokenów
to tylko około 20 procent ruchu, więc nawet dokładna znajomość przyszłości wspólnego wektora 768 nie wystarcza.

## 1. Bogatsze modele z historii (`kod/co_jeszcze_z_historii.py`, `wyniki/co_jeszcze/CO_JESZCZE.json`)

Dopasowanie na celach do kroku 700, obrazy 0–5 / 6–11 na krzyż, test: starty 700, 720, ..., 900.
Jednostki: tokeny narzędzia plus 30 losowych na obraz (578 jednostek, 768 współrzędnych).

| model | h=10 | h=50 | h=100 |
|---|---:|---:|---:|
| ridge, 8 cech, współczynniki wspólne | 0,23 | 0,27 | 0,23 |
| ridge, osobne współczynniki dla każdej z 768 współrzędnych | 0,22 | 0,27 | 0,23 |
| jak wyżej plus cechy wspólnego ruchu s_t | 0,22 | 0,27 | 0,23 |
| wyrocznia: prognoza bez zmian plus PRAWDZIWY przyszły ruch wspólny | 0,22 | 0,21 | 0,20 |
| udział ruchu wspólnego w błędzie prognozy bez zmian | 0,22 | 0,21 | 0,20 |

768 razy więcej współczynników nic nie daje. Cechy wspólnego ruchu nic nie dają. Znajomość przyszłości
wspólnego ruchu usuwa tylko 20 procent błędu, bo 80 procent ruchu jest własne dla tokenu.
Wiersze OU w pierwszym biegu miały błąd znaku (κ = 0); poprawiony bieg trwa, OU z `prognoza_900` daje 0,22.

## 2. Pamięć przyrostów (`kod/autokorelacja_przyrostow.py`, `wyniki/autokorelacja/ACF.json`)

Autokorelacja przyrostów tokenów narzędzia: 0,69 przy opóźnieniu 1, ujemna od opóźnienia 3, minimum −0,24
przy 4, zero od około 10 (kroki 300–1000; kroki 0–300 tak samo). Suma do opóźnienia 100: 0,56. Ruch na 100
kroków ma więc wariancję około 12 procent tego, co dałyby niezależne przyrosty: krótkie pchnięcie, odbicie,
a potem powolne błądzenie bez pamięci. Nie ma regularnych drgań o długim okresie, które równanie
z bezwładnością mogłoby przedłużyć.

## 3. Wczesna faza, 10 ziaren (`kod/wczesna_faza_10_ziaren.py`, `wyniki/wczesna_faza/WCZESNA_FAZA.json`)

Latent 192 sond co 10 kroków, kroki 0–500, ziarna 47–56 (`maly_vit_od_zera/wyniki_eden/petla_s*/wspolny/k01`).
Model: przedłużenie prędkości z opóźnień 10, 20, 30 kroków ze współczynnikami zależnymi od kroku startu,
dopasowanymi na 5 ziarnach, test na 5 pozostałych, horyzont 100 kroków.

| starty | 30–100 | 100–200 | 200–300 | 300–400 | razem |
|---|---:|---:|---:|---:|---:|
| wynik | 0,08 | 0,06 | 0,15 | 0,18 | 0,11 |

Przesunięcie w 100 krokach do 10 razy przesunięcia w 10 krokach (kwadraty): 0,2–0,6. Czysty dryf dałby 10,
białe drgania 1. Latent w rozgrzewce też zawraca, więc sufit 0,93–0,96 z wyroczni był zawyżony: wyrocznia
widziała zakręty, których z przeszłości nie widać.

## Co stąd wynika dla celu 85 procent

Cel wymaga informacji spoza historii latentu albo zmiany uczenia:

1. Siła paczki w równaniu (jak w `docs/23`, 85 procent na 1 krok). Na 100 kroków wymaga paczek z przyszłości
   i przeliczania sieci w każdym kroku, czyli odtworzenia uczenia. Łamie regułę „bez przyszłych klatek”, chyba
   że przyszłe paczki są planem sterowania pętli.
2. Mniejsze losowe pchnięcia: większa paczka. Rozrzut pchnięć maleje jak 1 przez rozmiar paczki, dryf zostaje.
   Z wzoru na sufit: 10 razy większa paczka podnosi stosunek dryfu do szumu 10 razy, czyli z około 0,3 do
   okolic 0,85–0,9. To przewidywanie do sprawdzenia nowym biegiem, nie wynik.
   **Sprawdzone 2026-10-03 i obalone** (`../maly_vit_gesta_historia_paczka/WYNIKI.md`): paczka 128 i 320 nie
   zmniejsza drgań ani nie poprawia prognozy na 10 kroków (0,20 → 0,24). Drgania pochodzą z normalizacji kroku
   w AdamW, nie z losowania paczki.
3. Zmiana celu prognozy: równanie Langevina uczciwie przewiduje dryf i rozrzut, nie położenie. Żądanie 85 procent
   dla położenia jest żądaniem przewidzenia szumu.

## Czego to NIE dowodzi

- Nie sprawdzono modeli nieliniowych ani uczonych sieci; wcześniejsze próby w projekcie (TFT, 11 mln
  parametrów) dały 40 procent i uczyły się okresu dopasowania na pamięć (`docs/23`).
- Jedno ziarno dla gęstej historii; 10 ziaren tylko dla wczesnej fazy co 10 kroków.
- Przewidywanie z punktu 2 opiera się na wzorze i teorii dyfuzji SGD, nie na pomiarze.
