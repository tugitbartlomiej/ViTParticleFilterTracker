# Sesja 2026-10-07 00:21:39: małe ViT o różnej wielkości latentu, trening na Edenie

## Cel sesji

Ciąg dalszy sesji `Session_2026-10-06_220807_Aplikacja_PCA_Czubek_GUI_Ustawienia_Odpinanie` (ta sama rozmowa).
Lista ViT według wielkości latentu oraz trening na Edenie kilku małych ViT o różnej wielkości latentu
(od 600 x 128 do 600 x 1536 i wersje z 8 tokenami TokenLearner) na zbiorze zmniejszonym do 5000 klatek,
po zgodzie użytkownika na uruchomienie.

## Typ

experiment

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Eksperyment `SmallViTAblation/experiments/maly_vit_rozne_latenty_5000`: siedem ramion (femto, pico, nano,
  tiny, small, tiny_tl8, nano_tl8), wspólny przepis, zbiór 5000 klatek (baza 1500 + 3500 równo po 118 filmach).
- Kod sprawdzony lokalnie na RTX 3070 (próba 20 kroków), wysłany na Eden (sumy SHA256 zgodne).
- Eden: próba 1823464 i bieg 1823465 (7 x 5000 kroków), wszystko COMPLETED w około 20 minut.
- mAP@0,5 (val 3000, koniec / najlepszy): 0,559 / 0,585 / 0,612 / 0,654 / 0,589 (0,624) / 0,595 (0,628) /
  0,542 (0,608); w nano_tl8 jeden token skupia mapę na narzędziu (15,6 x).
- Commity `f2addbd1` (eksperyment) i `77b5dec6` (wyniki).

Szczegóły: `SESSION_SUMMARY.md`.
