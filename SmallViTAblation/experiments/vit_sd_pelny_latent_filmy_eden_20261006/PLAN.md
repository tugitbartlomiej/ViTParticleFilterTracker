# Pytanie i zakres

Czy SD EMA020000 dostrojony przez adapter i cross-attention potrafi odszumiać
klatkę przy warunku jej pełnego naturalnego latentu ViT600x768, przy zamrożonych
ViT/VAE i bez przechowywania danych pośrednich? 20000 aktualizacji, 4 GPU,
microbatch 2, akumulacja 4, global batch 32. Zadania rodzica obejmują deploy/submit.

## Kontrole i warunki obalenia

1. Smoke: ten sam noisy latent i timestep, warunek naturalny, zerowy i tokeny
   przesunięte. Zerowa zmiana predykcji obala zależność od warunku. Noise, model,
   liczba tokenów i rozdzielczość pozostają identyczne.
2. Gradient: skończony loss i gradient, niezerowy adapter i attn2, brak gradientów
   parametrów ViT/VAE. Gałąź pomocnicza musi przeprowadzać gradient przez obraz.
3. Checkpoint: po dwóch rzeczywistych aktualizacjach wagi i AdamW odczytywane
   z dysku; po celowej zmianie adaptera odtworzenie daje tę samą predykcję.
   Nie dowodzi wznowienia strumienia na klastrze; trzeba sprawdzić osobny restart.
4. Held-out: zachowujemy istniejący podział całych filmów TRAIN/VAL. Nie mieszamy
   klatek filmu między splitami i nie używamy unknown 27. Raportujemy skalary
   epsilon MSE i film mean, rozmiar próby oraz ograniczenie liczby paczek.
5. Zgodność i detekcja: cel wyłącznie z naturalnego forwardu oryginalnej klatki,
   bbox jako supervision dla has_tool=True. Nie podmieniamy aktywacji ViT.
   Gałąź pomocnicza jest kontrolowalna przez aux_every=0; porównanie z nią wymaga
   tego samego splitu, seeda, budżetu i zakresu uczonych parametrów, nowego output.

## Czego ten bieg nie rozstrzyga

Spadek DDPM loss ani poprawa zgodności latentu nie dowodzą jakości fotorealistycznej,
niezależności tożsamości narzędzia ani poprawności sztucznie sterowanego latentu.
Pretrained SD widział już TRAIN w poprzednim treningu. ViT jest checkpointem
konkretnego eksperymentu, nie uniwersalnym modelem. Wynik na jednym seedzie
i części VAL nie stanowi pełnej walidacji generalizacji. Nie zapisujemy obrazów
do przeglądu. Nie wyciągamy tez badawczych przed zakończeniem i oceną biegu.

## Ograniczenia operacyjne

Strumień odtwarza w RAM wszystkie klatki przypisanych TRAIN filmów, także bez
adnotacji narzędzia. Brak adnotacji nie jest negatywną etykietą detektora.
Powtarzane shardy filmów na rank/worker nie synchronizują epok; raport liczby
filmów oznacza dostępność, nie potwierdzenie ekspozycji każdej klatki.
Replay iteratora przy resume kosztuje czas dekodowania. Dwa checkpointy i optimizer,
brak EMA, brak foto/video/latent/npy/npz outputs. Full UNet jest opcją wymagającą
ponownej oceny pamięci i miejsca. Rzeczywisty smoke i restart na Eden pozostają
weryfikacją rodzica; ten agent nie wykonuje SSH ani submit.
