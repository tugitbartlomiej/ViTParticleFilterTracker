# Langevin/GLE ruchu cząsteczek na kierunkach identyfikacji

Ukończono 2026-10-02T01:10:40.099745+02:00. Jeden stały obraz, 31 odrębnych tokenów, 1001 stanów (0..1000), czyli 1000 aktualizacji ViT/BatchNorm. To dynamika podczas treningu, odrębna od wcześniejszej reakcji na usunięcie narzędzia.

**Główny wynik:** autonomiczny GLE z L=4, ridge=0.01 wybrany na walidacji daje na H1 skill **60.24 procent**, znak wszystkich składowych **78.61 procent**, po odrzuceniu bardzo małych rzeczywistych zmian, z progiem wyznaczonym na fit, **81.59 procent**. Na H50 skill jest ujemny: **-5.03 procent**. Nie osiągnięto 80 procent ani redukcji MSE, ani znaku wszystkich zmian. Filtr osiąga 81.59 procent tylko na H1 i nie jest trafnością całych wektorów.

## Obraz, tokeny i przyczynowość kierunków

Sonda wybrana przez roota przed wynikami tego zadania: indeks od 0=2, `cataract_lmm/tracking/TR_0057_S1_P03/00001431`. Faktyczne source role = fit i split = train. Wcześniejsze określenie jako pierwsza holdout sonda było pomyłką, skorygowaną przed testem. Nie zmieniliśmy obrazu po obejrzeniu wyników. Bieg seed47 i jego późniejsze stany były analizowane wcześniej: temporalny test jest chronologiczny dla bieżącego algorytmu, ale nie stanowi nowego nietkniętego potwierdzenia ani generalizacji na nowy obraz.

![Stały obraz i śledzone tokeny](CZASTECZKI_NA_STALYM_OBRAZIE.png)

Token i to pełny z_i∈R^768; mikroruch to zmiana współrzędnych względem wspólnej bazy e_j. Maska oryginału, coverage i indeksy 31 tokenów są stałe. Wykresy pokazują indywidualne cząsteczki; pooling jest wyłącznie pomocniczy. CZASTECZKI.csv podaje numery od 0 i od 1, row/column, coverage i wagę. Granicę/interior wybrano według min/max coverage, bez oglądania trajektorii.

U500 (768×8) wyznaczono na checkpointcie 500 wyłącznie z 240 par train i wagami film→obraz→coverage. Rank 8 zadeklarowano wcześniej. Pozostałe pary nie wpływały na bazę lub ranking osi. Pięć osi wybranych na train 500: **359,594,472,512,138** (od 0:358,593,471,511,137).

Diagnostyczny probe C=1, próg 0.5: identyfikacja na 85 parach testowych AUROC 95.53 procent, balanced accuracy 90.00 procent. To **image-weighted**, znany ROI oryginał/LaMa; nie porównujemy bezpośrednio z wcześniejszym film-weighted U1000/progiem wybieranym na val.

Lokalny evalforward k500 na dokładnym obrazy.npy[2] zgadza się z LATENT_GESTY[500,2]: relativeL2 4.3e-07, maxabs 1.79e-06, cosine 0.999999999999908. Tolerancja relative 1e-4, pełne 600×768. U500 i zapisane kanały są zgodne numerycznie. Źródłowy pomiar A100/PyTorch2.9 versus lokalny RTX3070/PyTorch2.6 daje drobne różnice float32.

U1000 oraz poprzednie osie 500 /472/735/512/647 są wyłącznie opisem retrospektywnym, w osobnym NPZ. Nie użyto ich do prognozy. Root wyznaczył kąty główne U500 /U1000 w porownanie_podprzestrzeni/: 27.69,30.56,31.41,35.54,42.72,44.10,58.60,83.95 stopnia; mean cos²=.531748. To geometria zmiany dwóch ponownie identyfikowanych baz, bez zmiany cech tego testu. Nie interpretujemy U jako stałej uniwersalnej tożsamości.

![Ruch tokenów w U500](RUCH_KIERUNKI_K500.png)

Pionowe linie: 500 oznacza checkpoint identyfikacji U500 i anchor; 700 koniec pierwotnego fit/początek walidacji; 800 początek originów testowych. Po wyborze modelu końcowy refit obejmował cele do 799.

![Mikroruch pięciu osi500](RUCH_OSIE_K500.png)

## Równanie, estymacja i jednostki

```text
q_i(t)=U500ᵀ z_i(t),  x_i(t)=q_i(t)−q_i(500)
Δx_i(t)=x_i(t)−x_i(t−dt)
x_i(t+dt)=x_i(t)+b+A x_i(t)
            +Σ_(ell=0..L−1) K_ell Δx_i(t−ell*dt)+epsilon_i(t+dt)
```

b,A,K są wspólne dla tokenów i estymowane JEDNOCZEŚNIE ridge, z równymi wagami targetów i coverage-weighted tokenami. Stały tokenowy anchor usuwa różnice wyjściowych położeń. W surowym q wyraz wolny jest b_i=b−Aq_i(500). Jest to empiryczny GLE/AR z pamięcią i skończona affine aproksymacja Koopmana na obserwablach q plus stała. Nie dowodzi dokładnego Mori–Zwanziga, globalnej liniowości ani Brownowskiego szumu.

Źródłowe wejścia treningu: paczka32, AdamW lr=.0005, weight_decay=.05, warmup300. Bieg nie zapisywał faktycznych batch-ID/augmentacji/gradientów każdej aktualizacji. Dlatego ten model **nie ma jawnego członu sterowania**; nie przeniesiono telemetrii innych biegów. Ruch w eval obejmuje zmiany wag i biegnących statystyk BatchNorm; nie przypisujemy go wyłącznie gradientowi ani semantyce.

Skale stanu s_x i przyrostu s_d, intercept i ridge pochodzą tylko z fit. NumPy używa wierszy F@B; równanie wyżej używa kolumn. Jeśli B_x i B_lag są blokami fitted slope, macierze w fizycznych współrzędnych to:

```text
b = diag(s_d) intercept
A = diag(s_d) B_xᵀ diag(1/s_x)
K_ell = diag(s_d) B_lag,ellᵀ diag(1/s_d)
```

Na wszystkich regresorach fit skalowane i fizyczne równanie są numerycznie identyczne: maks. błąd dla wybranego dt1 4.44e-16. b/epsilon mają jednostki aktywacji na krok dt; A,K są bezwymiarowe dla tej dyskretyzacji. W q macierze stosujemy bez ukrytej normalizacji. To przyrost latentu, nie pęd AdamW. Gram rank 40/40, regularized condition 353, companion spectral radius 0.980263. Ridge stabilizuje predykcję, nie dowodzi fizycznej identyfikowalności A/K.

Macierze, skale, anchor i próg selected_epsilon do filtrowanej metryki znaku oraz wszystkie współczynniki do autonomicznego rollout zapisane w MACIERZE_DT1/DT10.npz; metadane MODELE_DT*.json. Ridge gamma = lambda*trace(Gram)/liczba_features, Gram z sumą wag 1, intercept niepenalizowany. Lambda 1e-4/1e-3/1e-2/.1/1, L0/1/2/4/8. Żadnego Kalmana. Nazwa selected_epsilon w NPZ oznacza 8D próg near-zero, a nie realizację reszty epsilon_i(t) z równania; reszty są zapisane osobno w RESZTY_DT*.npz.

## Podział, wybór i autonomiczny test

dt1: fit targets 501..700, wspólne wszystkim L (L8 korzysta z poznanej historii 492..500). Validation origins 700..749; H1/10/50, target<=799. Wybór maximum MEAN skill trzech horyzontów; nie surowego pooledMSE. Po wyborze refit targets <=799. Test origins 800..950, wszystkie modele te same151originów dla H1/2/5/10/20/50. H100 ma osobny zestaw101 originów 800..900, bez CI.

Wybrane dt1: L4, ridge0.01; validation mean skill 24.92 procent. Osobno zadeklarowany bestH1 to L2/.01, testH1 skill 60.59 procent. Nie zmieniono głównego modelu po teście.

dt10: targets 510..700 co 10, validation origins 700..740 co 10, równoważnie średni skill H10/50; refit do 790. Model L1/ridge1 wybrany przed testem,16 originów 800..950 co 10. H1/2/5 niedostępne, nie nazywamy dt10 jedną aktualizacją. Artefakt best_H1 dla tego modelu oznacza bestH10. dt1 dodatkowo oceniono na tych samych co 10 originach.

W rollout używano własnych przewidzianych przyrostów; żadnych przyszłych q, paczek, obrazów, wag, gradientów lub reszt. Aktualny stan originu i dostępna historia są wejściem rolling forecast. Osobne testy dokładnego prefixu i zatrutego NaN przyszłego sufiksu dały identyczne prognozy dla wszystkich modeli przy origin 800/950, H50. Skale i parametry pozostały zamrożone. Test wykonano po audycie Astry, jednokrotnie.

## Wyniki fizycznego błędu i kierunku

MSE = średnia po originach i 8 współrzędnych kwadratu błędu, z coverage-weighted tokenami. Skill=1−MSE/MSEpersistence. Cosine dotyczy wektorów ZMIANY forecast versus actual, nie poziomu. Znak oznacza poprawność pojedynczych współrzędnych. Persistence ma zerową zmianę: cosine nieokreślony, a nie idealny.

| H [aktualizacje] | MSE GLE | MSE brak zmiany | Skill | Znak wszystkich | Znak po filtrze | Cosine zmian | Większość znaku train |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.002960|0.007444|60.24%|78.61%|81.59%|0.7402|50.09%|
|2|0.014394|0.025831|44.28%|73.83%|75.19%|0.6232|49.95%|
|5|0.084308|0.096176|12.34%|63.87%|64.24%|0.3831|49.41%|
|10|0.160399|0.176483|9.11%|65.34%|65.74%|0.3673|49.89%|
|20|0.241420|0.266457|9.40%|66.62%|66.91%|0.4128|52.77%|
|50|0.304237|0.289670|-5.03%|63.84%|64.08%|0.3931|56.61%|
|100|0.343992|0.395599|13.05%|65.72%|65.94%|0.4338|58.23%|

H100 jest oddzielną diagnostyką na innych originach: 101 dla dt1 i 11 dla dt10. Nie jest kolejnym punktem tego samego testowego zbioru ani bezpośrednim porównaniem obu dyskretyzacji; odbicie skilla nie dowodzi poprawy przy dłuższym horyzoncie. H1 validcosine=4681/4681, H50=4681/4681. Wektorowa cosine H1=.74024 nie oznacza 74 procent poprawnych całych wektorów.

Filtr: epsilon_j=.1×ważona SD przyrostów fit do 799. Stały dla wszystkich H. Odsiew abs(rzeczywista przyszła zmiana)>epsilon jest wykonywany dopiero podczas oceny po poznaniu celu. To warunkowa metryka, nie dostępna online bramka pewności prognozy. H1 zachowuje **33417/37448** porównań składowych (89.24% bez ważenia; 89.00% ważone coverage). 81.59 procent dotyczy tej przefiltrowanej metryki, nie wszystkich zmian i nie całych wektorów. Maksymalny badany horyzont osiągający 80 procent tej metryki to H1; cały znak i skill nie osiągają 80 na żadnym badanym H.

| Ramię dt1 | H1 skill | H10 skill | H50 skill |
|---|---:|---:|---:|
|L0|-0.31%|7.60%|-3.63%|
|L1|42.95%|13.29%|-3.13%|
|L2|43.91%|13.03%|-2.46%|
|L4|60.24%|9.11%|-5.03%|
|L8|58.25%|9.20%|-4.43%|
|selected|60.24%|9.11%|-5.03%|
|best_H1|60.59%|11.10%|-5.11%|
|velocity|47.77%|-385.69%|-6533.42%|
|persistence|0.00%|0.00%|0.00%|

![Autonomiczne prognozy indywidualnych tokenów](PROGNOZY_H1_H10_H50.png)

![Horyzont i błędy](HORYZONTY_DT1.png)

## Checkpoint co 10, uczciwe porównanie originów

| H | dt1 selected skill, co 10 originy | dt10 L0 skill | dt10 selected L1 skill | dt10 selected znak |
|---:|---:|---:|---:|---:|
|10|12.27%|15.75%|16.12%|63.03%|
|20|5.90%|19.08%|20.12%|65.25%|
|50|-4.25%|10.45%|13.28%|63.78%|

![Modele na wspólnych co 10 originach](CHECKPOINTY_CO10.png)

Pamięć wnosi główną poprawę na H1/dt1: Markov jest blisko braku zmiany, a velocity daje 47.77 procent skilla wobec 60.24 GLE. Przy dt10 / H10 L0 jest niemal równie dobry jak L1; nie wykazujemy przewagi pamięci w każdej skali.

## Niepewność i kontrola wspólnej dynamiki

Circular moving block bootstrap 300, blok 50 aktualizacji, wspólne losowanie całego kompletu tokenów/modeli; dt10 blok 5 originów. Tylko około 3 efektywne bloki, więc CI są **opisowe**, nie niezależne potwierdzenie z wieloma ziarnami. H100 nie ma CI (około 1 blok). Bieg i obraz były wcześniej analizowane.

95% CI dt1: H1 skill[56.45, 64.46]%, znak wszystkich [77.49, 79.72]%, po filtrze[80.20, 82.98]%. H10 skill[1.75, 17.17]%; H50 skill[-26.07, 11.71]%. dt10 H50punktowo 13.28%, CI[-7.96, 29.94]%, obejmuje 0. Brak gwarancji monotonicznego wyniku z H.

Transfer tego samego modelu do 8 stałych farbackgroundtokens (zmieniono tylko anchor, operator bez refitu): H1 skill57.60%, znak 77.55%. Jest to zbliżone do tooltokens, więc przewidywalność nie jest wyjątkowa dla narzędzia. H10/H50 background skill: -29.11/-133.57%. Model opisuje także wspólne zmiany sieci/BN; nie rozstrzygamy przyczyny.

## Reszta: kolor, kształt rozkładu i niedomknięcie

Reszta one-step z ciągłego target 801..1000 (200 próbek) używa rzeczywistej historii do kolejnego originu; jest osobna od autonomicznych błędów H. ACF/Ljung–Box badane w czasie per token / per coordinate i dla shared residual. Gaussowskość osobno przez skew/kurtosis/Jarque–Bera. Nie spłaszczono tokenów do fałszywej jednej serii.

Raw Ljung–Box p<.05 dla 134/256 szeregów; Jarque–Bera p<.05 dla 52/256. Bez korekty wielokrotnych testów i degrees of freedom dopasowanych parametrów: diagnostyczne wyniki, nie certyfikat white/Gaussian. Dla shared q2 JB p=0.5532 (nie odrzuca), Ljung–Box p=7.57e-05 (korelacje); brak odrzucenia gaussowskości nie implikuje białości.

Cross-token mediany korelacji reszt w tej samej współrzędnej: 0.722, 0.896, 0.652, 0.690, 0.765, 0.816, 0.769, 0.558. Są dodatnie i duże: reszta zawiera wspólny składnik aktualizacji sieci. Zmiany wariancji połów i pełne ACF w SZUM_DT*.json. Nie narzucamy FDT dla AdamW ani fizycznego white Brownian noise; niedomknięcie pozostaje.

![Reszty i ACF](RESZTY_ACF.png)

## Co uzyskano i czego nie

Wyznaczono przyczynowo dostępne kierunki 500, pełne indywidualne trajektorie 31 tokenów i wspólny empiryczny GLE z pamięcią. Poprawiono krótką prognozę względem persistence/velocity/Markov; nie wykazano 80 procent na wszystkich zmianach, dobrego H50 dla głównego modelu, wyjątkowej dynamiki semantycznej narzędzia ani dokładnego równania fizycznego. Brak telemetrii minibatchy ogranicza interpretację dryfu/reszt. Potrzebne nowe niezależne biegi i obrazy oraz jawne historyczne wejścia, bez prawdziwych przyszłych paczek.

## Artefakty i odtwarzanie

PLAN.md; IDENTYFIKACJA_K500.npz/json (U500); TRAJEKTORIE_PRZYCZYNOWE_K500.npz; osobno TRAJEKTORIE_RETROSPEKTYWNE_K1000.npz; MACIERZE_DT1/DT10.npz; WYBOR_WALIDACJA_DT*.json; METRYKI_DT*.json; PROGNOZY_DT*.npz; SZUM/RESZTY_DT*; AUDYT_PREFIX_DT*; ZGODNOSC_K500.json; CZASTECZKI.csv; AUDYT_ASTRA.md. Root ma własność porownanie_podprzestrzeni/ i PODSUMOWANIE.md.

F:/Python/python.exe, kolejno identyfikacja500.py, trajektorie.py, prognoza.py --stage validation; audit Astry przed prognoza.py --stage test; raport.py. Bez kopiowania 22 GB, pełne wektory 768D przez mmap tylko w RAM, małe NPZ. Bez pobierania, treningu i commitów.
