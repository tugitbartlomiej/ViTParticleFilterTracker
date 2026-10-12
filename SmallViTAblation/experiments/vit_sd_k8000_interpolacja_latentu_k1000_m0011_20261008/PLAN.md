# Plan: bezpośrednia generacja z interpolowanych latentów m0011

## Pytanie

Jak zmienia się obraz generowany przez model `SD_ViT_K8000_A_control_K10500`, gdy od sztucznego latentu obrazu `m0011` odejmujemy kolejno po jednej dziesiątej pełnego przyrostu aż do latentu oryginalnego obrazu?

## Wejścia

- Oryginalny latent: `z_baza` z `KONIEC.npz` dla `m0011`.
- Sztuczny latent: `zb_m0011_minmax.npy`.
- Generator: `assets/SD_ViT_K8000_A_control_K10500/best_sd.pt`.
- Stały seed szumu: 20261008.
- DDIM: 50 kroków, `leading`, eta 0.

## Konstrukcja serii

`delta = z_sztuczny - z_oryginalny`.

`z_k = z_sztuczny - (k / 10) * delta`, dla `k = 0..10`.

Powstaje 11 punktów, ponieważ oba końce są jawnie zachowane: bezpośredni sztuczny latent oraz 10 kolejnych odejmowań, z których ostatnie daje dokładnie latent oryginalny. Każdy punkt używa identycznego początkowego szumu.

## Podłoga i ograniczenie interpretacji

Kontrolą jest końcowy punkt `k=10`, czyli latent oryginalny tego samego obrazu, przy identycznym seedzie i samplerze. Jedyną zmienną między obrazami jest warunek latentowy.

Latenty m0011 pochodzą z ViT K1000, a generator był uczony na latencie ViT K8000. Tensor ma zgodny kształt 600 × 768, lecz osie reprezentacji nie mają potwierdzonej zgodności. Test mierzy więc zachowanie generatora dla wejścia poza jego potwierdzonym rozkładem. Nie dowodzi poprawnego odwrócenia sztucznego latentu K8000.

## Kryteria techniczne

- Hash generatora musi zgadzać się z kartą modelu.
- Oba końce interpolacji muszą być numerycznie identyczne z wejściami.
- Każdy obraz musi mieć skończone wartości i niezerowe odchylenie pikseli.
- Nie nadpisujemy wcześniejszego etapu 10 wykonanego innym generatorem.

