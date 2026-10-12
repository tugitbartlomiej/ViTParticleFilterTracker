# Optymalizacja 128 kanałów: wynik

Promień wybrany wyłącznie na walidacji: 1.0 SD.

Oracle oznacza osobną optymalizację obrazu z jego etykietą; nie jest testem generalizacji. Shared oznacza jedną korektę uczoną na train i zamrożoną po val.

| Promień | Metoda | AP50 test | Strata test | Confidence×IoU test |
|---|---|---:|---:|---:|
| 0.25 | baseline | 0.226160 | 2.140136 | 0.236764 |
| 0.25 | shared | 0.226345 | 2.137338 | 0.237974 |
| 0.25 | shared_random11 | 0.226182 | 2.139917 | 0.236757 |
| 0.25 | shared_random22 | 0.226331 | 2.140453 | 0.236633 |
| 0.25 | shared_random33 | 0.224628 | 2.140164 | 0.236540 |
| 0.25 | oracle | 0.244550 | 2.107035 | 0.238401 |
| 0.25 | oracle_random11 | 0.226149 | 2.139944 | 0.236471 |
| 0.25 | oracle_random22 | 0.226418 | 2.140267 | 0.236498 |
| 0.25 | oracle_random33 | 0.224551 | 2.140399 | 0.236243 |
| 0.5 | baseline | 0.226160 | 2.140136 | 0.236764 |
| 0.5 | shared | 0.226342 | 2.135102 | 0.238894 |
| 0.5 | shared_random11 | 0.226079 | 2.139677 | 0.236823 |
| 0.5 | shared_random22 | 0.226380 | 2.140760 | 0.236511 |
| 0.5 | shared_random33 | 0.224631 | 2.140229 | 0.236595 |
| 0.5 | oracle | 0.257449 | 2.074622 | 0.241984 |
| 0.5 | oracle_random11 | 0.226051 | 2.139759 | 0.236425 |
| 0.5 | oracle_random22 | 0.240433 | 2.140412 | 0.236540 |
| 0.5 | oracle_random33 | 0.224493 | 2.140662 | 0.236219 |
| 1.0 | baseline | 0.226160 | 2.140136 | 0.236764 |
| 1.0 | shared | 0.226157 | 2.132306 | 0.239924 |
| 1.0 | shared_random11 | 0.226074 | 2.139181 | 0.236985 |
| 1.0 | shared_random22 | 0.226374 | 2.141459 | 0.236268 |
| 1.0 | shared_random33 | 0.224763 | 2.140357 | 0.236422 |
| 1.0 | oracle | 0.276814 | 2.011786 | 0.245162 |
| 1.0 | oracle_random11 | 0.226155 | 2.139405 | 0.236001 |
| 1.0 | oracle_random22 | 0.240828 | 2.140733 | 0.236173 |
| 1.0 | oracle_random33 | 0.234738 | 2.141214 | 0.235821 |

Kryterium wstępnej użyteczności wspólnej korekty: niespełnione.
95% CI różnicy straty po filmach: [-0.010826896316564038, -0.004749667427192133].
95% CI różnicy confidence×IoU po filmach: [0.0024909371367978148, 0.003794924120503267].

Wartości kanałów: TARGETS_ORACLE.npz (per obraz/token), SHARED.npz (korekta wspólna). Optymalizacja lokalna nie gwarantuje maksimum globalnego ani istnienia naturalnego obrazu z takim latentem. Jeden checkpoint, znane wcześniej filmy, etykieta jednej ramki narzędzia. Nie wykonano generowania obrazów ani aktualizacji wag ViT.
