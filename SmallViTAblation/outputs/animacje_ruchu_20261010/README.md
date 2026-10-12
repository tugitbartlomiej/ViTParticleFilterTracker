# Typy ruchu cząsteczek w latencie ViT i co z nich czytamy

Data: 2026-10-10. Animacje nagrane narzędziami aplikacji (zakładka „Ruch zbioru”, przycisk „Nagraj film…”
i metoda `ZakladkaRuchu.nagraj_film`, skrypt `generuj_animacje.py`) oraz z danych policzonych bezpośrednio
z gęstego zapisu trajektorii (`edit/dane/licz_trajektorie.py`, animacje w `edit/animations/`). Montaż wszystkich
klipów: `edit/final.mp4` (`edit/montaz.py`). Wskaźniki: `WSKAZNIKI.json`, `edit/dane/PODSUMOWANIE.json`.

„Cząsteczka” = wektor jednego tokenu (patch 16 × 16) w 768D po GELU ostatniego bloku MLP, dla stałego zdjęcia,
śledzony po kolejnych checkpointach uczenia. Oś czasu to krok uczenia, nie klatki filmu.

Dane: historia K1000 (1500 zdjęć × 600 tokenów = 900 000 cząsteczek, 101 checkpointów co 10 kroków, uczenie od
zera na bazie 1500 klatek), bieg 35 000 klatek (100 zdjęć uczących, końce 21 epok oraz gęsty zapis co 1 krok dla
tych samych 100 zdjęć do kroku 2188, potem co 10), segmentacje narzędzia, źrenicy i rogówki na patchach.

## Sześć typów ruchu (animacje)

| Plik | Typ ruchu | Co widać | Jak czytać |
|---|---|---|---|
| `A1_rozgrzewka_przeplyw_kolektywny.mp4` | przepływ kolektywny (balistyczny) | kroki 0–200, 225 000 cząsteczek; wszystkie jadą razem w jednym kierunku, kolor reżimu żółty, potem przełącza się na fiolet | w rozgrzewce lr (300 kroków) 80–90 % energii ruchu to wspólny wektor; kolejne kroki zgodne (cos 0,9); to sieć zmieniająca wszystkie reprezentacje naraz, nie rozróżnianie treści |
| `A3_rozgrzewka_na_siatce_zdjecia.mp4` | ruch zależny od pozycji | siatka zdjęcia, Z = norma zmiany latentu; powierzchnia faluje gładko niezależnie od treści | położenie patcha tłumaczy 84 % wariancji ruchu w rozgrzewce: pierwsze kroki reagują na stałe kodowanie pozycji sin-cos, nie na obraz |
| `A2_drgania_w_klatce_jeden_obraz.mp4` | drgania w klatce (uwięzienie) | 600 cząsteczek jednego zdjęcia ze śladami, kroki 300–700 | cząsteczka zawraca co zapis wokół wolno dryfującego położenia: α ≈ 0,3 (co 10 kroków), VACF < 0; to dynamika szklista, nie gaz ani ciecz |
| `edit/animations/slot_1/render.mp4` | trzy reżimy w czasie opóźnienia τ | MSD(τ) log-log z gęstego zapisu co 1 krok | α ≈ 1,8 przy τ = 1–3 kroków (balistyczny), plateau α ≈ 0,1 przy τ = 30–100 (klatka), α ≈ 1 przy τ ≈ 1000 (ucieczka z klatki); VACF +0,73 po 1 kroku, −0,1 po 10 |
| `A4_temperatura_narzedzie_vs_tlo.mp4` | podział po temperaturze | układ fizyki (X temperatura, Y zmiana 768D, Z α), lewa: tło, prawa: narzędzie, okno ±5 zapisów | narzędzie drga mocniej (AUC temperatury 0,88 w kroku 100–300), ale w sumie przesuwa się mniej niż tło (AUC zmiany 768D 0,10); źrenica i rogówka pośrednio |
| `A5_atencja_i_IoU_rozpoznanie_narzedzia.mp4` | skąd wiemy, że sieć rozpoznaje narzędzie | jedno zdjęcie, heatmapa atencji z wag każdego checkpointu, Z = zmiana latentu bez pozycji, w napisie pewność i IoU | atencja skupia się na patchach narzędzia, IoU zdjęcia 0,44 (krok 100) → 0,94 (krok 1000); średnie IoU na 1500 zdjęciach 0,26 → 0,60 |
| `A6_epoki_kondensacja_chmury.mp4` | kondensacja chmury | 60 000 cząsteczek przez 21 epok, położenia w PCA3 | mediana promienia chmury 1,92 (epoka 1) → 0,96 (epoka 8), potem stała; mAP walidacji 0,45 → 0,60 → plateau 0,62–0,65 od ~8. epoki |
| `edit/animations/slot_2/render.mp4` | zegar uczenia | serie czasowe z ruchu: udział pola kolektywnego, temperatura obszarów, cos kolejnych kroków, strata, mAP | fazy uczenia widać w samym ruchu, bez etykiet |

## Wskaźniki wytrenowania (K1000, 900 000 cząsteczek; `WSKAZNIKI.json`)

| Krok | IoU średnie (1500 zdjęć) | AUC temperatury narzędzie vs tło | reżim bal./dyf./uw. |
|---|---|---|---|
| 30 | 0,15 | 0,44 | 91 / 9 / 0 % |
| 60 | 0,20 | 0,56 | 0 / 49 / 51 % |
| 100 | 0,26 | 0,76 | 0 / 15 / 85 % |
| 300 | 0,40 | 0,85 | 0 / 4 / 96 % |
| 600 | 0,53 | 0,79 | 0 / 8 / 92 % |
| 1000 | 0,60 | 0,74 | 2 / 29 / 69 % |

Czytanie: w rozgrzewce (krok 30) ruch narzędzia nie różni się od tła (AUC 0,44, czyli poniżej 0,5), bo wszystko
jedzie razem. Między krokiem 60 a 100 ruch przechodzi z balistycznego w uwięziony i w tym samym czasie temperatura
zaczyna odróżniać narzędzie od tła (AUC 0,76 → 0,85 do kroku 300). To moment, w którym sieć zaczyna traktować
patche narzędzia inaczej niż tło. IoU rośnie wolniej i dalej, do 0,60 w kroku 1000.

## Gęsty zapis co 1 krok (bieg 35 000, 60 000 cząsteczek; `edit/dane/PODSUMOWANIE.json`)

- Udział pola kolektywnego w energii ruchu: 0,80–0,88 w krokach 5–100, 0,47 w kroku 300, 0,22 w kroku 1000.
- Temperatura (energia drgań własnych po odjęciu pola, PCA32) w kroku 1000: tło 0,024, rogówka 0,037, źrenica
  0,047, narzędzie 0,058. Szczyt około kroku 240–290, potem powolne chłodzenie (do kroku 2188 około 4 razy).
- Przewaga temperatury narzędzia nad tłem jest cechą wczesnego uczenia: narzędzie/tło w PCA32 wynosi 1,9 (krok
  300), 2,4 (krok 1000), 1,7 (krok 2000) i tylko 1,04 od około 5. epoki do końca; w pełnych 768D późno narzędzie
  jest nawet chłodniejsze od tła (0,7–0,8). Źrenica zostaje cieplejsza przez cały bieg (1,4 w PCA32, 1,1–1,2 w
  768D). W historii K1000 (1000 kroków) jesteśmy jeszcze w gorącej fazie: AUC temperatury 0,85 w kroku 300 i 0,74
  w kroku 1000. Interpretacja (jeden bieg, do sprawdzenia na innych ziarnach): patche narzędzia drgają najmocniej,
  gdy sieć uczy się narzędzia, a stygną, gdy jego reprezentacja się ustali.
- VACF po 1 kroku +0,73 (silna pamięć prędkości w skali jednego kroku), po 10 krokach −0,07 … −0,12 (zawracanie).
- Lokalny wykładnik MSD: 1,75–1,79 (τ = 2), 0,7 (τ = 7), 0,4 (τ = 15), 0,03–0,13 (τ = 35–80), 0,5 (τ ≈ 430),
  0,96–1,07 (τ ≈ 1000): balistyczny → klatka → dyfuzja.
- Efekt paczki: gdy zdjęcie sondy jest w paczce, energia kroku jej cząsteczek jest tylko 1,11 razy większa niż
  w pozostałych krokach. Ruch cząsteczki wyznacza wspólna aktualizacja wag (pole średnie), nie własne zdjęcie.
- Korelacja Spearmana energii ruchu populacji ze spadkiem straty na paczce po rozgrzewce: 0,34; z normą
  aktualizacji wag: 0,18.

## Skąd wiemy, że sieć nauczyła się rozpoznawać narzędzie

1. **Metryki detekcji** (zapisane przy każdym checkpoincie): pewność i IoU z etykietą na mierzonych zdjęciach rosną
   i wypłaszczają się; walidacja mAP@0,5 na 3000 klatkach spoza treningu osiąga plateau 0,62–0,65 od ~8. epoki
   biegu 35 000.
2. **Atencja**: heatmapa z wag kolejnych checkpointów skupia się na patchach narzędzia (A5).
3. **Ruch cząsteczek narzędzia zaczyna się różnić od tła**: AUC temperatury 0,44 → 0,85 między krokiem 30 a 300;
   narzędzie jest cieplejsze, ale przesuwa się w sumie mniej (AUC zmiany 768D 0,10). Kolejność temperatur
   tło < rogówka < źrenica < narzędzie trwa przez pierwsze ~2 epoki; potem narzędzie stygnie do poziomu tła
   (w 768D poniżej), a źrenica zostaje cieplejsza. Ostygnięcie narzędzia zbiega się z nasyceniem IoU i mAP.
4. **Kondensacja chmury**: mediana promienia chmury w PCA3 maleje do ~8. epoki i potem stoi; walidacja w tym samym
   czasie przestaje rosnąć. Dalsze epoki tylko dopasowują model do zbioru uczącego (strata maleje, mAP nie).
5. **Czego ruch nie mówi**: cząsteczki nie dzielą się na osobne skupienia po właściwościach ruchu (GMM, ARI 0,01);
   podział jest ciągły. Ruch w rozgrzewce to kodowanie pozycji, nie treść. Zdjęcia z gęstego zapisu i z historii
   K1000 są zdjęciami uczącymi.

## Montaż

`edit/final.mp4` (1920 × 1080, 30 kl./s, 168 s): karta tytułowa, A1, A3, A2, MSD (slot 1), A4, A5, A6, zegar
uczenia (slot 2), karta końcowa. Klipy osobno przeskalowane z letterboxem i połączone bezstratnie (`edit/montaz.py`,
segmenty w `edit/segmenty/`). Kontrola: `edit/verify/final_arkusz.png`. Animacje slotów mają własne README i klatki
kontrolne (`edit/animations/slot_*/`); zegar uczenia zaznacza skok temperatury przy kroku 2188 jako zmianę
gęstości zapisu (co 1 → co 10 kroków), nie zjawisko fizyczne.

## Jak odtworzyć

```powershell
F:/Python/3.13/python.exe -u generuj_animacje.py          # A1–A6 (aplikacja, karta graficzna, ~8 min)
F:/Python/3.13/python.exe -u edit/dane/licz_trajektorie.py  # dane z gęstego zapisu (~4 min, 10 GB RAM)
F:/Python/3.13/python.exe edit/animations/slot_1/script.py  # animacja MSD
F:/Python/3.13/python.exe edit/animations/slot_2/script.py  # zegar uczenia
F:/Python/3.13/python.exe edit/montaz.py                     # edit/final.mp4
```

W aplikacji: „Ruch zbioru” → strona „Animacja” → „Nagraj film…” nagrywa zakres „od–do” z bieżącymi ustawieniami.
