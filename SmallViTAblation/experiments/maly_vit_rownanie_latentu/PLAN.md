# Równanie Langevina dla latentu małego ViT w trakcie uczenia (2026-09-29)

## Pytanie

Prośba użytkownika: „zajmij się przemianą tego równania Langevina z TFT na takie, które będzie przewidywało latent.
Możesz wprowadzać jakoś dane jako sterowanie. Raczej nie będzie wymuszenia, ale szum może być. Spróbuj iteracyjnie
wyprowadzać to równanie, aż będziesz miał choćby w jednym kroku naprzód w przewidywaniu latentu dokładność powyżej 80 %.”

Równanie z TFT (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_latent_56`, `tft_langevin_czlon_zdarzen`):
z_{t+1} = z_t + D + JP + W + Z + N — dryf od stanu, prędkości i przyspieszenia; pamięć jako suma tłumionych,
obracających się jąder; wymuszenie; zdarzenia; szum barwny (warunkowa średnia z własnych reszt).

## Obiekt

Cząstka = sonda (192 obrazy: 12 wycinków × 16 wersji), latent = średnia aktywacji 768 neuronów MLP ostatniego bloku
po tokenach narzędzia (kanał tła osobno). Krok = jedna aktualizacja wag (nagranie lokalne) albo 10 aktualizacji
(zapis z Edena, `maly_vit_wektor_tozsamosci/wyniki_eden`, 10 ziaren, kroki 0–4500).

## Miara (jak w TFT)

Umiejętność = 100 · (1 − Σ‖Δẑ − Δz‖² / Σ‖Δz‖²) — procent błędu kwadratowego przyrostu usunięty względem braku
zmiany. Cel: > 80 na części testowej. Ocena poza próbą: dopasowanie na wcześniejszych krokach, test na późniejszych
(podział w czasie); człony o wspólnych parametrach dodatkowo z wyłączonymi całymi wycinkami (12 źródeł, po 3 w teście).

## Rozpoznanie przed planem (liczone 2026-09-29, ziarno 47, ramię `bez`, zapis co 10 kroków)

- Przyrost latentu na 10 kroków ma normę ~2,0 przy normie latentu ~6; kolejne przyrosty są ujemnie skorelowane (−0,22).
- Z samej historii latentu najlepsza prosta prognoza (powrót do średniej ruchomej) daje umiejętność ~17.
- Statystyki pikseli paczki (jasność, kontrast, odbicia) odtworzone z generatora losowego nie tłumaczą przyrostu
  (dopasowanie 7, test ujemny).
- Wniosek: ruch latentu między zapisami wynika z kolejnych paczek danych; bez sterowania danymi 80 jest nieosiągalne.

## Testy

- **T0 `nagranie.py`**: jedno uczenie lokalne identyczne z Edenem (ziarno 47, 3000 kroków), zapis w każdym kroku:
  latent sond, skład paczki, latent obrazów paczki i gradient straty względem niego, dokładny rozkład przyrostu na
  część od wag, od buforów BatchNorm i od samego ostatniego bloku (co 10 kroków).
  Obali: inna statystyka przyrostów niż na Edenie.
- **T1 `t1_bez_sterowania.py`**: przekład równania TFT bez sterowania (dryf do średniej, pamięć z tłumionymi obrotami,
  szum barwny) — to jest podłoga dla członów sterujących.
- **T2 i dalej**: człony sterowania danymi, dokładane po jednym; po każdym `WYNIKI_T*.md`.

## Podłogi

- brak zmiany (0) i najlepsze równanie bez sterowania (T1);
- **placebo sterowania**: te same cechy paczki, ale z innego kroku (przesunięcie o 250 kroków) — umiejętność musi
  spaść do poziomu T1, inaczej zysk bierze się z przecieku albo nadmiernego dopasowania.

## Dostęp do informacji (równy dla porównywanych wariantów)

Sterowanie widzi tylko to, co jest znane przed aktualizacją w kroku t: skład paczki i augmentację, latent obrazów
paczki w przejściu uczącym oraz gradient straty względem tego latentu (siła, z jaką dane ciągną latent). Nie widzi:
gradientu sond, samej zmiany wag, latentu po kroku. Rozkład T0 (od wag / od buforów) służy tylko do diagnozy.

## Co obali cel

Umiejętność testowa ≤ 80 po wyczerpaniu członów z uzasadnieniem fizycznym albo zysk, który placebo też daje.
Wynik negatywny opisujemy tak samo starannie.

## Uwagi do konwencji

Repo ViT nie ma rejestru `src.narzedzia` z projektu grantowego; deklarację i opis trzymamy w tym katalogu
(PLAN.md, WYNIKI_*.md), duże nagrania poza gitem (`.gitignore`).
