# Równanie Langevina latentu małego ViT z uczonymi macierzami, jak w TS (2026-09-29)

Prośba użytkownika: „w TS te macierze były bardzo duże i trzeba je było trenować, żeby ustawić parametry. Może tutaj
w równaniu dla latentu i wektorów tożsamości trzeba również zrobić takie macierze? … nie możesz tutaj zrobić takich
macierzy do równania Langevina dla tych wektorów latentu tożsamości?”

## Reguła: żadnych przyszłych klatek

Użytkownik (2026-09-29): „pamiętaj, że do równania nie mogą wchodzić przyszłe klatki, to jest niedozwolone
w prognozie”. Przyjmuję ściśle: paczka kroku t (klatki, na których sieć uczy się w przewidywanym kroku) też jest
przyszła. Równanie widzi stany sond do chwili t, paczki t−1 i wcześniejsze (już użyte w uczeniu) z ich sygnałem
błędu z chwili użycia, oraz stan optymalizatora (pęd = suma gradientów paczek do t−1). Siła paczki bieżącej S_t
z `../maly_vit_rownanie_latentu` (85 procent) zostaje tylko jako odniesienie „z przyszłą klatką”.

## Pytanie

Czy równanie o budowie z TFT (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56\model.py`:
dryf D, jądro pamięci JP, wymuszenie W, szum kolorowy N, wszystkie człony jako sieci uczone razem jednym błędem
kroku) przewiduje krok latentu sond i siły wektora tożsamości w uczeniu małego ViT, gdy wymuszeniem jest paczka
obrazów, i czy uczone wymuszenie zastępuje siłę paczki liczoną wprost przez sieć (`../maly_vit_rownanie_latentu`,
85 procent w medianie 10 ziaren).

## Przełożenie TS → ViT (każda różnica nazwana)

| TS (TFT, Favorita) | tutaj | różnica |
|---|---|---|
| cząstka = seria sklep–produkt, 72 600 w uczeniu | cząstka = sonda (stały obraz), 120 sond z narzędziem z wycinków „fit” | ~600 razy mniej cząstek, wszystkie z jednego przebiegu uczenia |
| krok = dzień | krok = jedna aktualizacja AdamW | — |
| stan = latent TFT 192 | stan = latent narzędzia sondy, 768 (wyjście GELU ostatniego bloku) | — |
| D: MLP z położenia, prędkości, przyspieszenia | to samo | — |
| JP: uwaga po 56 dniach z zanikiem ρ^l i obrotem ω·l | to samo, okno 32 kroki | okno krótsze, bo pęd AdamW (β1 = 0,9) gaśnie do 4 procent po 30 krokach |
| W: MLP z nowego dnia + 32–64 wyuczone filtry po 57 dniach | uwaga sondy na obrazy paczek t−1..t−16: podobieństwo sondy do obrazu × sygnał błędu obrazu, z wyuczoną wagą dla każdej paczki, + MLP | wejściem są paczki obrazów już użyte w uczeniu, nie kalendarz; TS widział „jutro” (plan promocji), tu paczki kroku t nie ma; pamięć wymuszenia 16 paczek zamiast 57 dni |
| N: warunkowa średnia szumu z reszt 55 dni | to samo, 31 kroków | — |
| uczenie AdamW 1e-3, wd 1e-4, partia 1024 cząstek, EMA wag 0,995, kara rozdzielności 0,05 | to samo; partia = 120 sond × 64 kroki | — |

Sygnał błędu obrazu to gradient straty względem latentu tego obrazu (`paczka_g`), policzony w kroku, w którym paczka
była użyta. Wariant bez gradientu widzi tylko latenty obrazów tych paczek i to, czy mają narzędzie.

## Ramiona (`uczenie.py --ramiona`)

- `pelne`: D + JP + W + N, W z sygnałem błędu obrazów.
- `bez_gradientu`: W tylko z latentów obrazów paczki i flagi narzędzia.
- `placebo`: jak `pelne`, ale zamiast paczek t−1..t−16 paczki t−251..t−266 (uczone od nowa). Podłoga dla W: zachowuje wszystko
  poza związkiem paczki z krokiem.
- `bez_W`: D + JP + N, uczone bez wymuszenia (sama duża pamięć, odpowiednik T1).

## Podział i ocena (poza próbą)

- Uczenie: nagranie ziarna 47 (`../maly_vit_rownanie_latentu/nagranie_s47_sily`), kroki 1000–1800, 120 sond
  z narzędziem z wycinków 0–7 (rola „fit” w manifeście sond).
- Wybór epoki: kroki 1800–2000, te same sondy.
- Test 1 (nowe kroki): ziarno 47, kroki 2000–2999, sondy uczące i osobno 60 sond z wycinków 8–11 (rola „holdout”,
  nowe narzędzia, których równanie nie widziało).
- Test 2 (nowy przebieg uczenia): ziarno 48 (`nagranie_s48_sily`, nagrane lokalnie, siły od kroku 2000), kroki
  2000–2999, bez żadnego dopasowania do tego ziarna.
- Miary: umiejętność = 100 · (1 − SSE / SS) przyrostu względem braku zmiany, na (a) pełnym latencie 768, (b) sile
  wektora tożsamości (rzut przyrostu, w standaryzacji negatywami z kroku 1000, na kierunek LDA liczony tak jak
  w `../maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py`, osobno dla każdego ziarna), (c) wektorze = średniej
  po sondach.
- Odniesienia na tych samych krokach i sondach, dopasowane na krokach uczenia ziarna 47: bezwładność a · Δz_t;
  pęd a · Δz_t + c · S_pęd (siła pędu AdamW liczona wprost, dozwolona; mianownik AdamW zawiera nowy gradient z wagą
  0,001, przeciek pomijalny); tylko dla porównania a · Δz_t + b · S_t z paczką bieżącego kroku (przyszłe klatki).
- Wyłączanie członów w ocenie (jak w TS): bez D, bez JP, bez W, bez N.

## Co obali

- `pelne` na teście nie wyżej niż `bez_W` albo nie wyżej niż `placebo` — wymuszenie z paczki nic nie wnosi.
- `pelne` poniżej odniesienia z pędem liczonym wprost — uczone wymuszenie nie zastępuje przeliczenia sieci.
- Duży spadek na sondach „holdout” albo na ziarnie 48 — równanie wkuwa sondy albo przebieg.

## Dalej (osobny krok, po wyniku)

Prognoza na 10 i 100 kroków: od chwili t równanie nie dostaje żadnych nowych paczek, więc wymuszenie gaśnie z pamięcią
ostatnich paczek (kappa). Przyszłe paczki nie wchodzą. Wagi ziarna 48 co 100 kroków (`wagi/`) służą tylko do sprawdzeń.

## Czego to nie rozstrzygnie

Jedno ziarno uczące, jedno ziarno testowe, jedno ziarno równania. Tylko ramię bez dokładania klatek.
