# Jeden fixed full768 CAV: szeroki signed alpha

Przeszukano817 punktów r od -100 do100 i lokalnie doprecyzowano najlepszy q. Najlepszy dodatni punkt r=0.62183158, alpha=18.13375569: q wzrasta z0.74341142 do0.74373132 (zaledwie+0.00031990), confidence rośnie z0.86786950 do0.87667030, lecz IoU maleje z0.85659355 do0.84835923. AP50 iAP75 pozostają1.0; single-image meanAP(.50:.95) spada z0.8 do0.7. Nie wykazano poprawyAP ani lepszej geometrii w punkcie wybranym wedługq.

Najlepsze meanAP w całej siatce nadal wynosi0.8, czyli tyle co baseline. Wśród474 remisów wybór przezq daje r=0.44487828. Duże r=+100 daje q=0.28291574, a r=-100 daje q=0.03051965: oba wyraźnie pogarszają detekcję. Najlepsza strona ujemna leży praktycznie przy zerze (r=-0.00010719), a przyrostq około jednej milionowej nie stanowi przekonującej korzyści wobec błędów numerycznych FP32.

Badano wskazany oryginalny obraz29, TR_0154_S1_P03/00001926, na oryginalnym zamrożonym k1000. Jeden CAV ma768 stałych współczynników. Alpha przesuwa obrazowe600x768 aktywacje w jednym ustalonym kierunku na oryginalnym ROI. J128 służy tylko do oznaczania i pomiaru części zmiany; nie optymalizowano128 ani768 współczynników.

Primary: q_star=maximum(confidence_j * IoU(box_j,GT)) po10 decodedboxes. To oracle optymalizacja tego samego obrazu z GT, nie generalizacja. Obraz nie został zmodyfikowany ani generowany przez SD.

| Wariant | r | alpha | confidence matched | IoU matched | q_star | AP50 | AP75 | AP50:.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|baseline|0|0|0.867869|0.856594|0.743411|1.000000|1.000000|0.800000|
|best_overall|0.621832|18.1338|0.876670|0.848359|0.743731|1.000000|1.000000|0.700000|
|best_positive|0.621832|18.1338|0.876670|0.848359|0.743731|1.000000|1.000000|0.700000|
|best_negative|-0.000107189|-0.00312583|0.867868|0.856597|0.743412|1.000000|1.000000|0.800000|
|best_AP50|0.621832|18.1338|0.876670|0.848359|0.743731|1.000000|1.000000|0.700000|
|best_AP75|0.621832|18.1338|0.876670|0.848359|0.743731|1.000000|1.000000|0.700000|
|best_AP50_95|0.444878|12.9735|0.874650|0.850296|0.743712|1.000000|1.000000|0.800000|

Najlepsza zmiana q względem naturalnego baseline: +0.00031990. Zakres zbadany r=[-100.0, 100.0]; liczba punktów 817. Najlepszy punkt w zbadanej siatce z lokalnym refinement, nie dowód globalnego maksimum.

## AP i korelacje
AP50/AP75/meanAP(.50:.95) liczone źródłową101punktową metryką na jednymGT. To single-image AP diagnostic, nie populacyjne mAP. Wiele punktów ma identyczne0/1 lub1/rankAP; APwinner wybierany osobno, remisy rozstrzyga q. Primary/refinement nie zmieniono na AP po obejrzeniu wyniku.

Pearson/Spearman tylko na zadeklarowanych21 coarsepoints, osobno signed21 oraz11positive/11negative z zerem. Wszystkie koeficjenty w COARSE_CORRELATIONS.json. To zależne próbki jednej deterministycznej funkcji, zależne od grid, saturacji, rank changes i nieliniowości, nie korelacja populacyjna ani wniosek o monotoniczności.

## Aparat i zastrzeżenia
Fastforward zachowuje oryginalnyresidualq, fc2, norm i glowa. AUDIT zawiera pełny forward z rzeczywistymhookGELU dla małych/dużych±punktów i wybranych najlepszych wyników. Modelstate i źródłowe hashe niezmienne.

Duże perturbacje nie są obcinane. Wartości poniżej minimalnegoGELU są zapisane, a sztuczne najlepsze aktywacje nie muszą być realizowalne przez obraz. Wybrany obraz należy do historycznej puli treningowej i wcześniejszej selekcji. Oryginalny knownROI i użycieGT do wyborualpha sprawiają, że to eksperymentoracle jednej klatki. Nie dowodzi to poprawy detektora na zbiorze ani możliwości generowania obrazu z tym latentem.

Best q: fraction belowGELUfloor=0.210286, delta normJ128=7.635324, other640=16.447948.

CURVE.csv/json: wszystkie punkty i10boxes. RAW_MAPS: surowe logits5kanałów. BASE_BEST_LATENTS: naturalbaseline oraz artificialbest600x768, jednostkowy lift, maskweights i indeksyJ128. CAV_COEFFICIENTS.csv: wszystkie768 współczynników z flagąJ128. AUDIT/PROVENANCE/HASHES i executedsource umożliwiają odtworzenie.


## Korelacje coarsegrid (bez pvalue)

| Zakres alpha | Metryka | Pearson | Spearman |
|---|---|---:|---:|
|signed_all|confidence_top1|0.350569|0.418182|
|signed_all|IoU_top1|0.205436|-0.138961|
|signed_all|q_star|0.214999|0.327273|
|positive_with_zero|confidence_top1|-0.932542|-0.254545|
|positive_with_zero|IoU_top1|-0.716700|-0.990909|
|positive_with_zero|q_star|-0.833425|-0.490909|
|negative_with_zero|confidence_top1|0.964155|1.000000|
|negative_with_zero|IoU_top1|0.856070|0.490909|
|negative_with_zero|q_star|0.887377|1.000000|

Na dodatniej stronie większe alpha w szerokim zakresie zwykle pogarsza IoU (Spearman około-0.991); confidence początkowo wzrasta, potem spada. Wniosek zależy od21 zadeklarowanychpunktów i nie jest prawem populacyjnym.
