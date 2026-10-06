# Sesja 2026-10-06 17:16:03: aplikacja latentu ViT (limity, detekcja, auto-szukanie, cząsteczki 3D, film, makrowektory)

## Cel sesji

Ciąg dalszy sesji `Session_2026-10-06_124141` (konwerter latentu ViT na cechy DINO). Najpierw testy konwertera
na zdjęciach i na sztucznym latencie z prezentacji, potem matematyczne granice latentu małego ViT, a na końcu
rozbudowa aplikacji `SmallViTAblation/app` (edytor latentu SD) o limity makrowektorów, detekcję ViT
z auto-szukaniem latentu, widoki cząsteczek 2D i 3D, opcje atencji, sekwencje klatek z filmów i przestrzeń
makrowektorów.

## Typ

experiment

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Test konwertera v2 na 8 zdjęciach VAL: 6 z 8 odnalezionych w zbiorze, mediana pozycji 172 w bazie 35 tys. klatek.
- Sztuczny latent J128 z prezentacji jest daleko poza rozkładem (do 722 odchyleń, maks. 51,35 przy naturalnym 4,43);
  konwerter na nim zawodzi.
- Granice latentu: dokładna granica z wag (LayerNorm + GELU, minimum −0,17), granica uczenia AdamW
  (B∞ ≈ 7,27 na krok, punkt stały wag z zanikiem ≈ 145); dla nieskończonego uczenia brak skończonej granicy,
  bo γ, β LayerNorm i biasy nie mają zaniku wag.
- Aplikacja: limity per kanał i suwaki od −100 % do +100 %, zakładka J128, generowanie na bieżąco, img2img,
  wybór patcha, zakładka detekcji ViT (pewność, IoU, mAP) z auto-szukaniem latentu, celami 0–100 %,
  zmiennym lr, porównaniem przed i po, makrowektorami (podsumowanie, model działa per token),
  cząsteczkami 2D, cząsteczkami 3D na OpenGL, opcjami atencji (głowy, rollout, strzałki źródło → cel),
  zakładką filmu (zakres min:s → N klatek, etykiety i obrysy, ręczny obwód, odtwarzanie) i przestrzenią
  makrowektorów (cząstka = suma 768 mikrowektorów).

Szczegóły: `SESSION_SUMMARY.md`.
