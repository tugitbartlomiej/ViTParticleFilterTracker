# Od sztucznego latentu do zdjęć uczących (2026-10-08)

Zlecenie użytkownika: zoptymalizować latent wielu zdjęć sztucznie, pod wiele celów detekcji (pewność, IoU, ranking,
GIoU jak w zakładce „Detekcja ViT”). Zobaczyć ruch cząstek. Zaplanować kolejne kroki zmian latentu (punkty pośrednie
toru). Dla każdego kroku znaleźć zdjęcia, które do niego pasują: przez konwerter latent -> DINO, przez SD z latentu
i przez prawdziwy latent kandydatów. Potem podmienić zdjęcia w zbiorze uczącym i porównać checkpointy z oryginałem.
Narzędzia: aplikacja (`app/korekta_latentu.py`, `pchniecie.py`, `stan_uczenia.py`, `kandydaci.py`, `silnik.py`).

## Etap 1. Cele latentu (oracle, oś latent_optimization)
- Checkpoint k1000 gęstej historii, bo SD i konwerter są uczone na latencie tego checkpointu (ten sam hash 3df42a6a).
- 24 obrazy manifestu J128, po jednym z filmu, najsłabsze przy k1000 (IoU najpewniejszej ramki).
- Cel „detekcja” (wielocelowy, pewność 0,95). Tryb „zakresy z checkpointu”, wszystkie 768 kanałów i 600 tokenów,
  Adam lr 0,05, 300 kroków. Zapis trajektorii co 50 kroków.
- Punkty pośrednie toru: kroki 100, 200 i 300, czyli z_w = z + δ(w).

## Etap 2. Zdjęcia dla punktów pośrednich
- Konwerter: z_w -> KONWERTER.pt v2 -> CLS DINO -> 20 najbliższych z 35 000 klatek. Pomijam filmy walidacji
  (wal3000) i samą klatkę celu. Potem przelicznie: prawdziwy latent ViT k1000 kandydata wobec z_w.
- SD: obraz z warunku z_w (DDIM 30 kroków, ziarno 0). Latent obrazu SD wobec z_w, detekcja na obrazie SD.
  Potem SD -> latent -> konwerter -> 3 klatki.
- Pchnięcie (pchniecie.py) w kroku 990 na obrazach celu jako sondach. Czy dodanie zdjęcia do paczki przesuwa latent
  celu w stronę δ(w): kosinus i zmiana straty celu w pierwszym rzędzie. Kontrole: losowe klatki z tej samej puli, sam
  obraz celu (z etykietą) i obraz SD z etykietą obrazu celu (pseudo).

## Etap 3. Uczenie z podmienionymi zdjęciami (nowe ramiona, zlecone wprost przez użytkownika)
Przepis biegu gęstej historii bez zmian: trening.Uczen(47), default_rng(47), baza 1500, paczka 32, AdamW 5e-4,
rozgrzewka 300, bf16, 1000 kroków. Różnice, nazwane wprost:
1. Tryb deterministyczny (atencja MATH, deterministyczne algorytmy). Oryginalny bieg go nie miał, a bez niego dwa
   identyczne biegi się rozchodzą. Dlatego porównuję z powtórzonym oryginałem w tym samym trybie (ramię A), a nie
   z zapisanymi wagami.
2. Podmiana 96 stałych pozycji bazy (ziarno 123) na nowe zdjęcia z etykietami zbioru. Liczebność bazy bez zmian.
3. Bez zapisu latentu sond co krok (niepotrzebny tutaj).
Ramiona:
- A: oryginał, powtórzony deterministycznie;
- B: zdjęcia z konwertera po przeliczeniu prawdziwym latentem (najbliższe punktom pośrednim);
- C: kontrola, losowe klatki z tej samej puli i z tymi samymi filtrami;
- D: zdjęcia z drogi SD -> konwerter (jeśli starczy czasu);
- E: same 24 obrazy celu z etykietą (odniesienie górne, 24 pozycje).
Ocena co 100 kroków: IoU najpewniejszej ramki i AP50 na 24 obrazach celu; mAP@0,5 na wal3000 przy 500 i 1000.
Jeden przebieg na ramię (jedno ziarno): różnice wymagają potwierdzenia na kolejnych ziarnach.
