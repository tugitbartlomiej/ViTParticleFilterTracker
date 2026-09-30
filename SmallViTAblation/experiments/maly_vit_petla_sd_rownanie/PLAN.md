# Pętla małego ViT, wersja 4.1: metoda użytkownika z równaniem ruchu latentu i Stable Diffusion

Prośby użytkownika (2026-09-30):
- „pobrać checkpoint Stable Diffusion, zastąpić FastGAN-a i zaimplementować nowe równanie; zobaczmy, co da nowa pętla”;
- „prognoza ma za zadanie przewidzieć, jak się będzie rozwijał latent w przyszłości, który identyfikuje narzędzie!”;
- „porównuję z kontrolą bez prognozy, czyli np. losowe albo następujące po sobie klatki!”.

Pierwsza wersja kodu (4.0, próba lokalna o 16:20) zmieniła cel prognozy na spadek błędu detekcji; użytkownik to
zatrzymał. Wersja 4.1 przywraca jego układ: prognoza latentu tożsamości, generator rysuje obraz o przewidzianym latencie,
wyszukanie podobnych prawdziwych klatek, uczenie. Wybór przez spadek błędu detekcji zostaje tylko jako ramiona dodatkowe
(`*_strata`, propozycja z `../../docs/25_PROPOZYCJA_POTOKU_DOBORU_KLATEK.md`), poza porównaniem głównym.

## Obrót pętli (kod `kod/petla4.py`)

1. **Prognoza latentu tożsamości** (`kod/prognoza4.py`, równanie z `docs/23_METODA_PRZEWIDYWANIA_LATENTU.md`). Siła
   tożsamości s każdej ze 180 sond z narzędziem (rzut latentu narzędzia na wektor tożsamości, jak w wersji 3) porusza się
   jak cząstka z bezwładnością: Δs(t+1) = 0,83·Δs(t) + 0,98·F. Siła paczki F: zmiana s po wirtualnym kroku AdamW z
   gradientu paczki, średnio z 8 paczek losowanych z obecnego zbioru uczącego. Prędkość startowa: nachylenie s z ostatnich
   100 kroków. Prognoza zmiany na H kroków przy stałej sile:
   Δs(H) = v0·a(1 − a^H)/(1 − a) + b·F/(1 − a)·(H − a(1 − a^H)/(1 − a)), liczona na 100 i 500 kroków.
   Reguła o przyszłych klatkach: tylko stan do chwili t, klatki już obecne w zbiorze i obecny stan optymalizatora.
2. **Obraz o przewidzianym latencie** (zamiast FastGAN-a): 6 z 12 kotwic o największej przewidzianej zmianie siły; dla
   każdej 8 wersji ze Stable Diffusion z maskami tej samej klatki (4 × ten sam układ z innym szumem, 4 × narzędzie
   przestawione: obrót do ±35°, skala 0,85-1,15, przesunięcie do ±20 px). Mały ViT mierzy siłę tożsamości każdej wersji
   na wycinku 176 × 176 (jak sonda) i wybiera wersję najbliższą: siła kotwicy + przewidziana zmiana na 500 kroków.
3. **Wyszukanie**: wycinek wybranej wersji → DINOv2 → lista podobnych prawdziwych klatek puli; 134 klatki na przemian z 6
   list, reguła ±30 klatek filmu (jak wersja 3).
4. **Uczenie** 500 kroków; mAP@0,5 co 100 kroków na 3000 klatkach walidacji.
5. **Trafność prognozy** po obrocie, we wszystkich ramionach: przewidziana siła kontra zmierzona na 100 i 500 kroków,
   jako RMSE względem „braku zmiany” (poniżej 1 = lepiej niż brak zmiany) i zgodność znaku zmiany; obok prognoza
   Koopmana albo Langevina z wersji 3.

## Ramiona

| ramię | co robi | rola |
|---|---|---|
| `sd_prognoza` | kroki 1-4 | metoda |
| `losowe` | losowe klatki puli, reguła ±30 | kontrola bez prognozy (wskazana przez użytkownika) |
| `kolejne` | klatki następujące po klatkach bazy w tym samym filmie (jak wersja 3) | kontrola bez prognozy (wskazana przez użytkownika) |
| `bez` | bez dokładania | dolna granica |
| `sd_teraz` | kroki 1-4, ale wersja SD najbliższa obecnej sile kotwicy | pomocnicza: jedyna różnica wobec metody to prognoza |
| `sd_strata`, `sd_strata_losowo`, `kotwice_strata` | wybór przez przewidziany spadek błędu detekcji (docs/25) | dodatkowe, poza porównaniem głównym, tylko na życzenie (`--ramiona`) |

## Różnice wobec wersji 3 (metoda użytkownika), każda nazwana

1. **Prognoza:** Koopman albo Langevin dopasowany do historii siły (wersja 3) zastąpiony równaniem ruchu z bezwładnością i
   siłą paczek liczoną przez sieć; Koopman/Langevin liczony dalej, ale tylko do porównania trafności.
2. **Generator:** FastGAN sterowany wzdłuż wektora tożsamości (300 kroków optymalizacji wejścia) zastąpiony przez Stable
   Diffusion: 8 wersji kotwicy z masek tej samej klatki i wybór wersji o sile najbliższej celowi (bez sterowania).
   Osiągalna zmiana siły zależy od rozrzutu wersji; zapisujemy „część zmiany” jak w wersji 3.
3. **Obraz zapytania:** wycinek 256 px wokół narzędzia z pełnej klatki SD (576 × 384), a nie wycinek FastGAN-a.
4. **Pula:** bez 10 filmów zbioru kontrolnego (używanego tylko przez ramiona `*_strata`); ta sama pula w każdym ramieniu.
5. **Ramię `current` z wersji 3** zastąpione przez `sd_teraz` (ten sam generator, bez prognozy).
Bez zmian: model, rozgrzewka, wektor tożsamości (LDA na sondach z końca rozgrzewki), 7 obrotów po 500 kroków, 134 klatki
na obrót, 6 kotwic, reguła ±30, wybór na przemian z list, walidacja, miara główna.

## Kryteria ustalone przed wynikiem (kampania wieloziarnowa, pole pod krzywą mAP@0,5 po rozgrzewce)

- K1: `sd_prognoza` lepsze od `losowe` w co najmniej 70 procentach ziaren i test Wilcoxona p < 0,05.
- K2: `sd_prognoza` lepsze od `kolejne` w co najmniej 70 procentach ziaren.
- K3 (rola prognozy): `sd_prognoza` lepsze od `sd_teraz` w co najmniej 70 procentach ziaren.
- Prognoza: równanie lepsze od „braku zmiany” (RMSE poniżej 1) na 100 i 500 kroków w większości obrotów i ziaren oraz lepsze
  od Koopmana/Langevina z wersji 3.
- Opisowo: część przewidzianej zmiany zrealizowana przez wybraną wersję SD, rozrzut siły wersji, mAP ramion w czasie.

## Etapy i zgody

1. Próba kodu lokalnie (`--proba`) - sprawdza tylko działanie kodu.
2. Pełne ziarno lokalnie i kampania wieloziarnowa na Edenie - **każde po osobnym „tak”** z liczbą ziaren, kart i czasu.

Uruchomienie lokalne: `uruchom_lokalnie.cmd kod\petla4.py --ziarno 47 [--proba] [--ramiona ...]`.
