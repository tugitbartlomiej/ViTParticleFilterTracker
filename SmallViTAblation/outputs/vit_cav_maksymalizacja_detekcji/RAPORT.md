# Wspólny skalar CAV a rzeczywista detekcja

Na zablokowanym teście AP50: baseline 45.7589 procent; full768 46.6715 procent (+0.9127 punktu), historyczna LDA 46.6652 procent (+0.9064 punktu), J128 43.3084 procent (-2.4504 punktu). Wszystkie CI zmian względem baseline i najlepszego po val random obejmują zero. Nie wykazano pewnej przewagi jakości detekcji.

Każdy główny kierunek wybrał r=+0.5, alpha=14.61536598, czyli górną granicę zadeklarowanej siatki. Nie określono optimum poza tym zakresem. Przy tym alpha średni udział współrzędnych ROI poniżej teoretycznej dolnej granicy GELU wynosi full768 17.42 procent, LDA 18.72 procent, J128 4.18 procent. Nie przycinano współrzędnych, aby zachować stały kierunek. To interwencje diagnostyczne w aktywacjach, bez dowodu realizowalności przez obraz.

Primary objective: źródłowe AP50 jednej anotowanej klasy, interpolacja 101-punktowa. To diagnostyka oracle ROI, nie wdrażalny detektor. Skalar wybrano po75 obrazach/23filmach walidacji, zapisano LOCK, następnie jednokrotnie oceniono85 obrazów/24filmów testu. Wagi i kierunki pozostały zamrożone.

Baseline test AP50=0.45758871, AP75=0.02250294, AP50:.95=0.11427454.

| Kierunek | r val | alpha | val AP50 | test AP50 | delta AP50 | CI95 delta AP50 |
|---|---:|---:|---:|---:|---:|---|
|full768|0.5|14.6154|0.425442|0.466715|0.009127|[-0.002226, 0.024335]|
|J128|0.5|14.6154|0.413388|0.433084|-0.024504|[-0.074224, 0.010846]|
|historical_LDA|0.5|14.6154|0.425831|0.466652|0.009064|[-0.002068, 0.023296]|
|random768_s202610051|0.5|14.6154|0.413705|0.461980|0.004391|[-0.000805, 0.013555]|
|random768_s202610052|-0.25|-7.30768|0.411000|0.459564|0.001975|[-0.002973, 0.006508]|
|random768_s202610053|0.1|2.92307|0.411806|0.457229|-0.000360|[-0.003709, 0.001597]|
|randomJ128_s202610051|0.5|14.6154|0.410667|0.436393|-0.021196|[-0.058430, 0.004207]|
|randomJ128_s202610052|0.25|7.30768|0.411546|0.457577|-0.000012|[-0.007070, 0.003402]|
|randomJ128_s202610053|0.25|7.30768|0.412311|0.456621|-0.000968|[-0.004716, 0.000149]|

## Wynik i podłogi
full768: testowa zmiana AP50=+0.00912663, CI[-0.00222608,+0.02433496]. Względem najlepszego po val dopasowanego random (random768_s202610051): +0.00473528, CI[-0.00453739,+0.01799178]. Nie wykazano dodatniej przewagi nad baseline z CI wykluczającym zero.
J128: testowa zmiana AP50=-0.02450431, CI[-0.07422431,+0.01084638]. Względem najlepszego po val dopasowanego random (randomJ128_s202610053): -0.02353632, CI[-0.07267734,+0.01208282]. Nie wykazano dodatniej przewagi nad baseline z CI wykluczającym zero.
historical_LDA: testowa zmiana AP50=+0.00906361, CI[-0.00206774,+0.02329611]. Względem najlepszego po val dopasowanego random (random768_s202610051): +0.00467226, CI[-0.00422069,+0.01764714]. Nie wykazano dodatniej przewagi nad baseline z CI wykluczającym zero.

## Aparat i ograniczenia
Wszystkie9 kierunków przeszły tę samą15-punktową siatkę. Alpha0 zapobiega pogorszeniu wybranej walidacji, lecz nie gwarantuje poprawy testu. Bestrandom wybierano tylko po val spośród3seedów, nie po teście. Film-bootstrap2000 na zapisanych predykcjach utrzymuje parowanie zmian; CI warunkowe na kierunki i wybór, bez ponownej selekcji.

Norma lift=1, wspólny alpha=r*trainmedianROI. Żadnego fittingu alpha do obrazu, clippingu, poprawiania wag ani kierunków. Fastforward obejmuje fc2, residual, norm i densehead, zgodność z hookiem GELU i bbox sprawdzono na3 walidacyjnych obrazach.

Punkty poniżej dolnej granicy GELU zapisano jako diagnostykę; interwencje nie muszą być realizowalne przez obraz. Oracle GT ROI i tylko jedna anotowana ramka mogą zawyżać użyteczność i zaniżać precision dla innych obecnych narzędzi.

To najlepszy punkt w zadeklarowanej siatce, nie globalne maksimum. Oryginały były w puli treningowej ViT, J128 historycznie wybrano z tej puli i test już wcześniej wykorzystano. Brak dowodu patient-disjoint, detector-training-disjoint i niezależnej generalizacji. LaMa może konfudować semantykę CAV. Nie mylimy AP50 z confidence ani BA sondy.

Pliki: CALIBRATION, SPLITS_GT, CAV_FIXED, DIRECTIONS_768/J128.csv, SMOKE, VALIDATION_CURVES, VAL_PRED_*, LOCK, TEST_PRED_* wraz z mapami i GT, TEST_INTERVENTIONS, BOOTSTRAP_SAMPLES, RESULT, AUDIT, PROVENANCE, HASHES.
