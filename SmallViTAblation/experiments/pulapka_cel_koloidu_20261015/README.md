# Cel koloidu w uczeniu i fizyka pułapki (2026-10-15)

Pytanie użytkownika: „podczas nauki pchamy tą pułapką optyczną koloid i do czegoś dążymy — do czego? Zmierzyć i znaleźć w literaturze,
czym to jest fizycznie”. Koloid = token 226 (środek ramki narzędzia) jednego zdjęcia, bieg s47 (`MalyViT_35000_od_zera_s47`,
806 checkpointów, kroki 0–65 640). Dane wejściowe: historia `app/cache_uczenia/lasery/eafb5a973be3d490` i wagi biegu.

## Pliki
- `pomiar.py` — siła błędu ∂L/∂x na koloid, pchnięcie MLP bloku 6, ruch w uczeniu, rozkład zmiany błędu (ruch koloidu / ruch głowy),
  zapadanie na podprzestrzeń głowy; zapis `outputs/pulapka_cel_koloidu_20261015/pomiar.npz` (~3 min na GPU).
- `analiza.py` → `outputs/pulapka_cel_koloidu_20261015/WYNIKI.md` — cel, zapadanie, prawo relaksacji, odpowiedź z opóźnieniem, dyfuzja.
- `pulapka_uczenia.py` → `outputs/pulapka_cel_koloidu_20261015/PULAPKA_UCZENIA.md` — koloid w czasie uczenia jako proces Ornsteina–Uhlenbecka wokół dryfującego środka.
- Pomiary pola wiązek w podkrokach (sztywność, wir, dywergencja): `app/pulapka_obliczenia.py`,
  wynik `app/cache_uczenia/lasery/eafb5a973be3d490/pulapka_podkrokow_t226_G5.npz/.csv`.
Uruchomienie: `python pomiar.py <katalog app> <katalog historii>`, potem `python analiza.py`, `python pulapka_uczenia.py`.

## Wyniki (cząsteczka 226 = token środka ramki)
1. **Cel jest znany z kodu uczenia**: głowa(LN(x)) ma dać „tu jest środek” i odległości do krawędzi ramki. Błąd tokenu 4,7 → 0,094
   (krok 16 400) → 0,048 (koniec).
2. **Uczenie przesuwa i koloid, i cel.** Sam ruch koloidu obniżyłby błąd (początek) o 3,1, sam ruch głowy o 4,2, oba razem o 4,6;
   cały obraz przez całe uczenie: koloidy −9,1, głowa −7,4, razem −5,9 (dublują się: współdziałanie +10,6). Pod koniec wkłady równe.
3. **Ruch w uczeniu idzie w stronę spadku błędu tylko słabo**: cos(ΔX, −∇L) mediana +0,18 → +0,11, dodatni w ~75 % kroków
   (losowo w 192 wymiarach ±0,07). Odpowiedź jest natychmiastowa: przy opóźnieniu ≥ 500 kroków korelacja ≈ 0.
4. **Pułapka wiązek (MLP bloku 6) pomaga celowi słabo**: obniża błąd tokenu w 63–90 % checkpointów, pod koniec o ~13 % błędu.
   Pole tej pułapki nie trzyma koloidu (brak zera pola w komórce, siodło, MLP rozprasza; pulapka_podkrokow).
5. **Prawdziwa pułapka działa w czasie uczenia**: koloid drga wokół wolno dryfującego środka jak cząstka Browna w pęsecie
   (proces Ornsteina–Uhlenbecka). Korelacja kolejnych przyrostów −0,415 (OU przewiduje −0,42); czas relaksacji 8–12 kroków
   na początku, ~30–50 kroków pod koniec (krótszy niż odstęp 100 kroków między checkpointami, więc tylko oszacowanie);
   „sztywność” 1/wariancja drgań rośnie ~5× (10–13 → 56–70 w jednostkach 1/rms²); środek pułapki zwalnia ~30×.
   MSD w późnym uczeniu ~ τ^0,13 (koloid uwięziony, prawie plateau).
6. **Dojście do położenia końcowego jest potęgowe, nie wykładnicze**: odległość ~ t^−0,43 (R² 0,99; wykładnicze R² 0,80),
   tło t^−0,34 — spowalnianie jak starzenie (aging) w układach szklistych.
7. **Zapadanie na podprzestrzeń głowy**: 5 głównych składowych niesie ~76 % wariancji cech (od początku), podprzestrzeń odczytu
   głowy 12–14 % (5× więcej niż losowo 2,6 %), bez narastania w czasie — brak postępującego zapadania (neural collapse) na tym zdjęciu.

## Czym to jest fizycznie
- Pole wiązek w jednym przejściu (MLP) = pole sił z pamięci klucz–wartość (Geva i in. 2021), w obrazie transformera jako układu
  cząstek: MLP = dryf, uwaga = oddziaływanie (Lu i in. 2019). Brak równowagi, siodło, wir, dywergencja > 0 → nie pęseta, tylko
  układ soczewek przepychający cząstkę (optyczne twierdzenie Earnshawa, Ashkin i Gordon 1983; rdzeń pułapki, Huang i in. 2022).
- Ruch w czasie uczenia = cząstka Browna w ruchomej pułapce harmonicznej: szum = losowe paczki danych (SGD/AdamW jako proces OU
  wokół minimum, Mandt i in. 2017), środek pułapki dryfuje potęgowo i zwalnia (uczenie jak układ szklisty: zwalnianie, dyfuzja
  na dnie krajobrazu, Baity-Jesi i in. 2019). Najbliższy obraz z fizyki: koloid przeciągany przez wolno przesuwaną pęsetę
  (van Zon i Cohen 2003; Imparato i in. 2007), z tą różnicą, że cel (głowa) też się uczy (rozkład reprezentacja–odczyt, Chou i in. 2026).

## Zastrzeżenia
- Jedno zdjęcie, jeden token środka, jeden bieg; rms-y w przestrzeni strumienia rezydualnego (baza nie jest zakotwiczona).
- τ późnego uczenia słabo rozdzielone (odstęp 100 kroków > τ); dokładniej wymagałoby gęstszych checkpointów.
- Dla tokenów tła błąd jest bliski zera i siły zapisane w float16 — ich cos i udziały są niewiarygodne.
- Kolekcja Scite z pracami odniesienia i tymi wynikami: „ViT: pułapka optyczna koloidu (pole wiązek MLP, cel w uczeniu)”.
