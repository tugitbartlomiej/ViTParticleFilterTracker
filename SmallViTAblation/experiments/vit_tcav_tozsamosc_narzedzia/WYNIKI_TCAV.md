# Wynik adaptowanego TCAV ROI/token

Wykryto lokalny wpływ kierunków na logit, lecz nie wykazano poprawy jakości detekcji przy zadeklarowanych małych epsilon. Historyczna LDA ma TCAV po filmach 93.61 procent (CI 86.11–99.17 procent); full768 także 93.61, J128 90.35 procent. Pięć obrazów ma dokładnie zerową pochodną, ponieważ komórka środka bbox wypada poza wsparciem maski. Przy największym epsilon LDA plus zwiększa filmową średnią IoU o 0.00001692 (CI obejmuje zero), minus zmniejsza o 0.00001678 (CI obejmuje zero). AP50 zmienia się odpowiednio o -0.00022529 i -0.00001986; recall, precision i F1 przy progach .5/.75 nie zmieniają się. Znak konceptu nie daje gwarancji korzyści dla bbox.

Paper-style test t względem .5 daje małe p, lecz empiryczne null nie potwierdzają swoistości: full768 placebo p=0.2941, random p=0.2549; J128 placebo p=0.4510, random p=0.3137 (po Bonferronim wszystkie 1). Są to dwustronne porównania odległości TCAV od .5. Film sign-flip względem .5 także nie jest testem specyficzności konceptu. Dobre rozpoznanie oryginał/LaMa przez sondę nie jest dowodem poprawy detektora.

Eksperyment rozszerza wcześniejszą LDA o wpływ na detektor. SVD 8D nie jest kierunkiem LDA. Kierunki, sieć i podział są zamrożone; nie optymalizowano obrazów ani latentu.

## Aparat
Kim et al. 2018 §§3-4: separator, pochodna logitu, frakcja dodatnia i test względem 0.5. Tutaj uczymy wspólny 768D CAV ze średniej ROI; lift ma normę Frobeniusa 1. W historycznej LDA normalna jest już w raw 768D, więc nie dzielimy jej przez SD sondy SVD.

## Wynik
| Kierunek | TCAV po filmach | Średnia pochodna | Zero |
|---|---:|---:|---:|
|full768|0.93611|0.00515173|5|
|J128|0.90347|0.00149163|5|
|historical_LDA|0.93611|0.00397259|5|

TCAV i BA/AP sondy są odrębnymi pomiarami. Pełne probe scores, magnitudes i CI są w RESULT.json i PROBE_METRICS.json.

## Detekcja: wynik przed i po
| Ramię | Delta AP50 | Delta średniej IoU | Poprawa/pogorszenie IoU |
|---|---:|---:|---:|
|J128_eps0_-1|0.000000|-5.03037e-08|8/7|
|J128_eps0_+1|0.000000|1.6723e-08|7/7|
|J128_eps1_-1|0.000000|-4.76339e-07|22/30|
|J128_eps1_+1|0.000000|4.19344e-07|30/22|
|J128_eps2_-1|0.000000|-2.19407e-06|26/30|
|J128_eps2_+1|-0.000173|2.19785e-06|30/26|
|J128_eps3_-1|0.000015|-4.62973e-06|26/32|
|J128_eps3_+1|-0.000245|4.69066e-06|31/26|
|J128_permuted_eps0_-1|0.000000|3.99199e-08|7/7|
|J128_permuted_eps0_+1|0.000000|-6.96618e-08|5/6|
|J128_permuted_eps1_-1|0.000000|5.77207e-07|26/23|
|J128_permuted_eps1_+1|0.000000|-6.08275e-07|24/27|
|J128_permuted_eps2_-1|0.000000|2.94855e-06|31/29|
|J128_permuted_eps2_+1|0.000000|-2.94092e-06|29/31|
|J128_permuted_eps3_-1|0.000000|7.4525e-06|31/29|
|J128_permuted_eps3_+1|0.000000|-7.42588e-06|29/31|
|J128_random_eps0_-1|0.000000|-8.94251e-09|16/11|
|J128_random_eps0_+1|0.000000|-1.72414e-08|9/13|
|J128_random_eps1_-1|0.000000|6.18698e-08|31/23|
|J128_random_eps1_+1|0.000000|-9.64367e-08|23/32|
|J128_random_eps2_-1|-0.000173|3.6509e-07|32/26|
|J128_random_eps2_+1|0.000000|-3.4658e-07|26/32|
|J128_random_eps3_-1|-0.000173|1.33889e-06|34/26|
|J128_random_eps3_+1|0.000000|-1.2797e-06|26/34|
|full768_eps0_-1|0.000000|-3.34171e-07|13/24|
|full768_eps0_+1|0.000000|2.89701e-07|22/12|
|full768_eps1_-1|0.000000|-3.09092e-06|25/33|
|full768_eps1_+1|0.000000|3.04398e-06|33/24|
|full768_eps2_-1|0.000015|-1.53377e-05|25/35|
|full768_eps2_+1|-0.000245|1.53367e-05|35/25|
|full768_eps3_-1|-0.000161|-3.02058e-05|25/35|
|full768_eps3_+1|-0.000245|3.02884e-05|35/25|
|full768_permuted_eps0_-1|0.000000|1.09294e-07|9/6|
|full768_permuted_eps0_+1|0.000000|-1.61609e-07|7/14|
|full768_permuted_eps1_-1|0.000000|1.30858e-06|27/25|
|full768_permuted_eps1_+1|0.000000|-1.31868e-06|23/27|
|full768_permuted_eps2_-1|0.000000|6.49996e-06|32/27|
|full768_permuted_eps2_+1|0.000000|-6.49778e-06|27/32|
|full768_permuted_eps3_-1|0.000000|1.48616e-05|32/28|
|full768_permuted_eps3_+1|-0.000072|-1.4799e-05|28/32|
|full768_random_eps0_-1|0.000000|3.01029e-08|8/6|
|full768_random_eps0_+1|0.000000|-3.63129e-08|5/7|
|full768_random_eps1_-1|0.000000|2.1649e-07|32/24|
|full768_random_eps1_+1|0.000000|-2.17805e-07|24/30|
|full768_random_eps2_-1|0.000000|1.10253e-06|32/25|
|full768_random_eps2_+1|-0.000173|-1.11558e-06|25/32|
|full768_random_eps3_-1|0.000000|2.21487e-06|33/25|
|full768_random_eps3_+1|-0.000173|-2.17765e-06|25/32|
|historical_LDA_eps0_-1|0.000000|-2.09203e-07|19/20|
|historical_LDA_eps0_+1|0.000000|1.78172e-07|22/19|
|historical_LDA_eps1_-1|0.000000|-2.06915e-06|26/30|
|historical_LDA_eps1_+1|0.000000|2.06392e-06|30/26|
|historical_LDA_eps2_-1|0.000015|-1.03514e-05|27/33|
|historical_LDA_eps2_+1|-0.000245|1.03449e-05|33/27|
|historical_LDA_eps3_-1|-0.000020|-1.67825e-05|27/33|
|historical_LDA_eps3_+1|-0.000225|1.69224e-05|33/27|

Nie utożsamiamy dodatniej zmiany logitu ani TCAV z poprawą detekcji. Raport obejmuje obie strony, wszystkie epsilon, także pogorszenie i zerowe zmiany. Dokładne bbox/IoU, score responses i błędy finite differences są w FINITE_DIFFERENCES.json.

## Podłoga
50 film-bootstrap separatorów, 50 pair-label placebo z flipem całego filmu wyłącznie train i 50 losowych kierunków na dopasowanym wsparciu/normie. Film-bootstrap CI i testy sign-flip nie liczą klatek jako niezależnych filmów. Paper-style t-test jest opisowy; Bonferroni uwzględnia rodzinę porównań.

## Zastrzeżenia i czego wynik nie dowodzi
Known original ROI and single annotated tool bbox
Historical J128 and LDA selection reuse these films; not independent validation
LaMa artefact confounding
Local hidden perturbations need not correspond to images
Paper ttest across shared-film repetitions is descriptive; primary grouping by film
Empirical null resolution 1/51; Bonferroni4 prevents tiny p with only 50 nulls

Nie wykazujemy nowego odkrycia konceptu, semantycznej czystości LDA, identyfikacji egzemplarza narzędzia ani generalizacji na niezależne filmy.

## Audyt i wagi filmów

Historyczna LDA miała BA ważone po filmach; BA tabeli PROBE_METRICS jest nieważone po obrazach. AUDIT.json i RESULT.probe.test_film_weighted zawierają porównywalne metryki po filmach oraz CI dla BA, AP i AUROC. Zapisany oryginalny LATENT_PELNY.before zgadza się z cache tokenów ROI. Kierunek LDA nie został refitowany.
