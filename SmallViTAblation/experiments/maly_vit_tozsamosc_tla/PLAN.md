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
