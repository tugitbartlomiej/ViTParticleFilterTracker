# Kroki × podkroki: afiniczność uwagi, pole MLP i złożenie siodeł (bieg s47, klatka TRAIN 0)

**Data:** 2026-10-23, 00:41 (Europe/Warsaw). **Typ:** pomiar na zapisanych danych i wagach, bez treningu. **Rodzice:** `pochodzenie.json`.

## Cel

Pytanie użytkownika: czy w układzie „lasery i koloidy” jest coś, co widać **tylko** na płaszczyźnie krok uczenia (806 checkpointów)
× podkrok przejścia (12 podkroków: uwaga i MLP sześciu bloków)? Czy uwaga jest przekształceniem afinicznym chmury cząsteczek i czy
to widać na cząsteczkach? Jakie są analogie do układów fizycznych (pułapki optyczne, pola sił)?

## Hipotezy

1. **Uwaga jest afiniczna w chmurze, nie w cząsteczce.** Przy ustalonych wagach A przesunięcie Δx_i = Σ_j A_ij W v_j + b jest liniowe
   w całej macierzy stanów X (kombinacja wypukła wektorów wartości), ale nie jest funkcją własnego położenia x_i. MLP jest odwrotnie:
   dokładnie funkcją własnego położenia, ale nieliniową. Przewidywanie: CV-R² afinicznego dopasowania Δ ~ M x_i + c jest niskie dla uwagi
   i wysokie dla MLP tam, gdzie GELU działa prawie liniowo; udział części „barycentrycznej” (zamrożone A) w pełnym jakobianie uwagi jest
   duży, a jakobian własny uwagi bliski jedności.
2. **Na starcie uwaga jest czystą translacją** (A prawie jednostajne, 600 efektywnych źródeł): wszystkie cząsteczki dostają to samo
   przesunięcie; uczenie zamienia translację na selektywne mieszanie (spadek efektywnej liczby źródeł), najszybciej w blokach 4–5.
3. **Złożenie siodeł kurczy.** Każdy blok z osobna jest siodłem z wirem (Z15), ale iloczyn 12 jakobianów po głębokości może kurczyć
   większość kierunków, jeśli podprzestrzenie kurczące kolejnych bloków się nie nakładają (analogia do ogniskowania naprzemiennego
   w akceleratorach: sekwencja soczewek skupiających i rozpraszających ogniskuje netto). Test: udział kierunków o σ < 1 iloczynu wobec
   pojedynczych map oraz nakładanie podprzestrzeni kurczących kolejnych bloków wobec losowego.
4. **Profil głębokości zmienia się w uczeniu**: rozdział narzędzie–tło powstaje w określonym podkroku i określonym kroku uczenia; filtr
   dolnoprzepustowy uwagi (gładkość po siatce) i rozpychanie MLP mają fazę w uczeniu.

## Model i dane

- MałyViT biegu `assets/MalyViT_35000_od_zera_s47` (ziarno 47), 806 checkpointów 0–65 640 (60 epok). SHA256 w `pochodzenie.json`.
- Jedna klatka TRAIN 0 (TR_0050_S1_P03, klatka 1629), 600 tokenów; etykieta z obrysu: 3 tokeny narzędzia (udział ≥ 0,5), 13 mieszanych,
  584 tła. Zapisane przejścia `7992f2317c64a2e3/przejscia/k*.npz` (807 plików, float32; `xn · rms` odtwarza dokładne wycentrowane stany 192D).

## Procedura

1. `Kod_eksperymentu/pomiar_kroki_podkroki.py Wyniki/Dane 1` — obserwable na wszystkich 806 checkpointach × 12 podkrokach (lista w
   docstringu): r_po/r_przed, |Δ|, składowa promieniowa, CV-R² afiniczne (192D, 2 fałdy po 300 tokenów, zero z przetasowania), R² w rzucie
   3D, udział translacji, wskaźnik balistyczny po głębokości, cosinusy między podkrokami, efektywne źródła uwagi, masa na sobie / 8
   sąsiadach / średnia odległość, wzbogacenie uwagi na narzędziu, lasery (zapalone, efektywna liczba, top-10), Fisher i czystość kNN
   narzędzia w 13 stanach, energia Dirichleta na siatce.
2. `Kod_eksperymentu/jakobiany_podkrokow.py Wyniki/Dane` — autograd (float64) jakobianów 12 podkroków dla 7 tokenów (223, 260, 224
   narzędzie; 226 mieszana; 3 tła) w 7 checkpointach: mapa własna (192 × 192), pełny jakobian uwagi (192 × 600 × 192) i jego wersja
   z zamrożonymi A; wartości osobliwe map własnych, bloków i iloczynu po głębokości; nakładanie podprzestrzeni kurczących.
3. Literatura: trzy wyszukiwania Undermind (workspace „ViT: detekcja końcówki narzędzia…”): uwaga jako mapa barycentryczna i zmiana profilu
   głębokości w uczeniu; sekwencje siodeł, które ogniskują netto (ogniskowanie naprzemienne, siodło obrotowe, pułapka Paula); bloki jako
   rozszczepienie operatorowe uwaga/MLP.

## Metryki

Jak wyżej; wszystkie liczone w pełnych 192D (rzut 3D tylko tam, gdzie pytanie dotyczy obrazu w aplikacji). Zera: przetasowane Δ (R²),
udział narzędzia w chmurze (kNN), k_b·k_{b+1}/192 (nakładanie podprzestrzeni), 1 (Dirichlet, cechy niezależne od sąsiedztwa).

## Ograniczenia

- Jeden bieg (s47) i jedna klatka; 3 tokeny narzędzia (cienki cystotom), więc Fisher i kNN są miarami orientacyjnymi.
- Jakobiany własne uwagi trzymają inne cząsteczki w miejscu (pole „w miejscu cząsteczki” przy zamrożonym otoczeniu); pełny jakobian
  obejmuje innych tylko dla 7 tokenów i 7 checkpointów.
- CV-R² afiniczne w 192D ma 193 parametry na współrzędną i 300 tokenów uczących: zero z przetasowania podane obok.
- Wykresy (F1–F9), raport i krótki wniosek przygotowane 2026-10-23 na wyraźne zamówienie użytkownika; paleta i zasady wykresów ze skill
  dataviz (kategorie w stałej kolejności, jeden odcień dla wielkości, niebieski ↔ czerwony dla wielkości ze znakiem).

## Odtwarzanie

```
cd experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29
W:/Python/3.11/python.exe Kod_eksperymentu/pomiar_kroki_podkroki.py Wyniki/Dane 1
W:/Python/3.11/python.exe Kod_eksperymentu/jakobiany_podkrokow.py Wyniki/Dane
```
Wyniki: `Wyniki/Dane/kroki_podkroki.npz`, `kroki_podkroki_tabele.json`, `jakobiany_podkrokow.npz`, `jakobiany_podkrokow.json`, logi.

## Stan (2026-10-23)

Pomiar ukończony. Wyniki: `Wyniki/Dane/` (npz, json, tabele, literatura), `Wyniki/Wykresy/F1…F9` (PNG + SVG), `Wyniki/Raport/RAPORT.md`,
`Wyniki/Krótki_Wniosek/WNIOSEK.txt`. Hipotezy: 1 potwierdzona (afiniczność w chmurze na starcie, potem afiniczny wygląd przy nieliniowym
mechanizmie), 2 potwierdzona (translacja → gwiazda do narzędzia, kolaps w blokach 5 i 4 w krokach 120–450), 3 odrzucona (złożenie siodeł
nie kurczy; nakładanie podprzestrzeni kurczących = losowe; przejście to mapa ściskająca), 4 potwierdzona (front rozdziału przesuwa się z bloku
3 do 4; uwaga nie wygładza po siatce, MLP tak).

