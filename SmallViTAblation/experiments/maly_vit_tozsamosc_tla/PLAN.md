# Tożsamość narzędzia na parach „z narzędziem i bez” (2026-09-29)

Prośba użytkownika: wygenerować przykłady samego tła — czarnego i samego oka — żeby dobrze wyznaczyć tożsamość
narzędzia w latencie. Pomiar lokalny (RTX 3070), bez uczenia sieci.

## Co zrobiono

1. `tla.py`: dla każdej ze 180 sond z narzędziem dwa zdjęcia bez narzędzia w tym samym miejscu —
   `oko` (narzędzie zamalowane z otoczenia, OpenCV inpainting Telea w masce poszerzonej o ~8 px) i `czarne`.
   **Wybór wykonawcy:** zamalowanie zamiast FastGAN-a, bo FastGAN uczono na wycinkach z narzędziem i zawsze je rysuje;
   zamalowanie daje parę „to samo zdjęcie bez narzędzia”. Podgląd: `wyniki/TLA_PODGLAD.png` (zostaje lekka smuga).
2. `wektor_z_tlami.py`: sieci po rozgrzewce (krok 1000, ziarna 47–56), latent narzędzia na tokenach narzędzia sondy,
   latent zdjęć bez narzędzia na tych samych tokenach. Trzy wektory: `stary` (jak w pętli: narzędzie kontra tło sond
   i `empty_black`), `nowy` (narzędzie kontra to samo miejsce bez narzędzia), `razem` (wszystkie negatywy).
   Prognoza siły każdego wektora sprawdzona na zapisanych latentach wersji 2 (kroki 1000–4500, horyzont 500 kroków).

## Wyniki (`wyniki/WEKTOR_Z_TLAMI.json`, średnio 10 ziaren)

- Rozdzielenie narzędzie kontra to samo miejsce bez narzędzia (`oko`), d' w sd negatywów: stary 2,0, nowy 16,6,
  razem 15,3. Na wycinku wyłączonym z uczenia AUC: stary 0,967, nowy 0,997.
- Narzędzie kontra tło daleko od narzędzia: stary 10,7, nowy 2,8, razem 21,7.
- Cosinus stary–nowy 0,28: to prawie inny kierunek w latencie.
- Prognoza siły (błąd / brak zmiany, mediana; ramię `bez`): Koopman stary 0,94, nowy 0,95, razem 0,95;
  Langevin stary 0,89, nowy 1,02, razem 0,96. Nowy i łączny wektor są przewidywalne co najwyżej tak samo jak stary
  (lepsze od starego w 2–6 z 10 ziaren).

## Wniosek

Stary wektor tożsamości w dużej części mierzył „miejsce i scenę”, a nie samo narzędzie: zdjęcie oka bez narzędzia
w tym samym miejscu dostaje prawie taką siłę jak narzędzie (różnica 2 sd). Wektor z par „z narzędziem i bez” wyróżnia
samo narzędzie bardzo wyraźnie (16–17 sd), ale jego zmiany w czasie uczenia Koopman i Langevin przewidują nie lepiej
niż zmiany starego wektora. Użycie go w pętli wymaga nowej kampanii (osobna zgoda).

## Zapisane wektory i ich skupienie (`zapisz_wektory.py`)

`wyniki/WEKTORY.npz`: dla każdej z 10 sieci (ziarna 47–56, krok 1000) trzy wektory (`stary`, `nowy`, `razem`), każdy
to 768 wag plus średnia i rozrzut negatywów (`s47_nowy_w`, `s47_nowy_mu`, `s47_nowy_sd` itd.). Wektory różnych sieci
nie są porównywalne, bo neurony każdej sieci znaczą co innego.
Skupienie (`wyniki/WEKTORY_SKUPIENIE.json`, średnio 10 sieci): połowę wagi nowego wektora niesie ~78 z 768 liczb
latentu, 90 % wagi ~312 (stary: 67 i 301) — tożsamość jest rozłożona na setki neuronów, nie na kilka.

## Ile kierunków latentu koduje tożsamość — obraz cząstkowy (`kierunki_tozsamosci.py`)

Pytanie użytkownika: każdy token to cząstka o 768 współrzędnych; ile kierunków (wektorów) tej przestrzeni koduje
tożsamość narzędzia? Liczba neuronów z poprzedniej sekcji na to nie odpowiada, bo jeden kierunek może się składać
z setek neuronów. Cząstki: każdy token narzędzia sondy osobno (1568 cząstek na sieć), porównane z cząstką na tym
samym tokenie zdjęcia bez narzędzia (`oko`, `czarne`). Sieci po rozgrzewce (krok 1000), ziarna 47–56.

Wynik (`wyniki/KIERUNKI_TOZSAMOSCI.json`, średnio 10 sieci):

- Kolejne kierunki rozdzielające (LDA, usunięcie kierunku, następny w reszcie przestrzeni; AUC liczone na wycinkach
  wyłączonych z wyznaczania kierunku, 4 części po 3 wycinki): kierunków z AUC ≥ 0,9 jest 1–2 (średnio 1,3),
  z AUC ≥ 0,75 — 5–8 (średnio 6,5). Pierwszy kierunek ma AUC 0,94–0,96 w każdej sieci, drugi 0,88–0,91, dalej
  spadek łagodny; po ~15–20 kierunkach AUC schodzi do 0,6 (bliżej rzutu monetą).
- Energia różnic w parach (cząstka z narzędziem − ta sama cząstka bez narzędzia): 50 % w 2–3 kierunkach,
  90 % w 29–48 (średnio 38); pierwszy kierunek niesie 32–43 %.

Wniosek: tożsamość narzędzia (to, czy w tym miejscu jest narzędzie) siedzi głównie w jednym kierunku przestrzeni
cząstek, z kilkoma słabszymi kierunkami pomocniczymi. Szczegóły wyglądu narzędzia (różnice między wersjami sond)
rozkładają się na kilkadziesiąt kierunków, ale te same w sobie słabo odróżniają narzędzie od tła. Pojedyncza
cząstka jest głośniejsza niż średnia z tokenów sondy (AUC 0,95 zamiast 0,997 wektora `nowy`).
