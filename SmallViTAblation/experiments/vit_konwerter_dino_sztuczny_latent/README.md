# Konwerter v2 na sztucznym latencie J128

Uruchomienie: `F:/Python/python.exe -X utf8 experiments/vit_konwerter_dino_sztuczny_latent/test.py`.
Wyniki: `outputs/vit_konwerter_dino_sztuczny_latent` (`WYNIKI.json`, `znalezione_zdjecia.png`,
`zmiana_latentu_i_odpowiedzi.png`).

Sztuczny latent: `outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz` (before/after), zdjęcie TR_0154_S1_P03
kl. 1926 (film TRAIN). Konwerter `outputs/vit_konwerter_dino_filmy_eden/v2/KONWERTER.pt`, baza 35000 klatek TRAIN
`outputs/vit_dino_faiss_35000/features.npy`. Każde z 32 znalezionych zdjęć przechodzi przez zamrożony ViT:
cos128 i MSE128 jego naturalnego latentu do sztucznego celu na 128 kanałach oraz IoU detekcji do etykiety.

## Wynik 2026-10-06

- Latent przed zmianą (naturalny, film TRAIN): konwerter daje cos 0,902 do prawdziwego DINO, 0,857 po odjęciu
  średniej. Nasz przepis obrazu odtwarza zapisany latent z błędem względnym 0,0003.
- Latent po zmianie leży daleko poza rozkładem uczącym: maks. 722 odchylenia standardowe (przed: 24), średni
  moduł na 128 kanałach 27,4 odchylenia (przed: 0,69).
- Odpowiedź konwertera na sztuczny latent: cos do odpowiedzi przed zmianą 0,40, tokeny 0,11. Konwerter nie robi
  małej poprawki, tylko wyrzuca inny obraz.

| 32 zdjęcia znalezione | cos128 do celu (średnio / maks.) | IoU detekcji | filmów |
|---|---|---|---|
| ze sztucznego latentu | 0,130 / 0,157 | 0,15 | 2 (głównie TR_0036) |
| z latentu przed zmianą | 0,116 / 0,136 | 0,80 | 1 (TR_0154) |
| prawdziwe DINO źródła | 0,109 / 0,128 | 0,77 | 1 (TR_0154) |
| losowe | 0,140 / 0,202 | 0,52 | 26 |
| odniesienie: źródło przed zmianą | 0,107 | 0,857 | - |

Wnioski: z naturalnego latentu konwerter działa (cos 0,90, znajduje sąsiednie klatki tego samego filmu).
Ze sztucznego nie: zdjęcia znalezione przez niego nie są bliżej celu na 128 kanałach niż losowe, a detekcja
jest na nich gorsza. Żadne sprawdzone zdjęcie nie zbliża się do celu (cos128 najwyżej 0,20), co zgadza się
z wcześniejszym ustaleniem, że cel J128 jest nieosiągalny dla naturalnych zdjęć.
