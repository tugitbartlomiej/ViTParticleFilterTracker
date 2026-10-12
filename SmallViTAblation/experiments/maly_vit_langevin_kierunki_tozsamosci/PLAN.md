# GLE ruchu cząsteczek na zamrożonych kierunkach identyfikacji

Utworzono 2026-10-02 Europe/Warsaw, przed obliczeniami. Rodzina badawcza maly_vit_langevin_kierunki_tozsamosci. Bez treningu ViT, pobierania i commitów. Katalog <=150 MB, minimum 1 GB wolnego miejsca. Praca nie modyfikuje dawnych eksperymentów.

## Dane i granica interpretacji

Źródłowy bieg seed47 ma 1001 stanów (aktualizacje0..1000), stałe obrazy i LATENT_GESTY.npy (1001,12,600,768), float32. Wagi co10 aktualizacji. Pomiar w eval obejmuje wagi ORAZ biegnące statystyki BatchNorm. Brak zapisu faktycznych batch-ID, augmentacji i gradientów na każdym kroku: bez jawnego sterowania, nie zastępujemy go danymi innego biegu. Reszty zawierają nieznane paczki, AdamW, BN i niedomknięcie modelu.

Sonda ustalona przez roota przed wynikami tej analizy: indeks od0=2, cataract_lmm/tracking/TR_0057_S1_P03/00001431. Faktyczne source-role='fit', split='train'; wcześniejsze nazwanie pierwszą odłożoną sondą było pomyłką. Nie zmieniamy obrazu po rozpoczęciu pracy. Maska i coverage oryginału stałe. Każdy maskowany token jest osobną cząsteczką z_i w wspólnej przestrzeni R^768, e_j wspólnymi makrowektorami. Pooling służy wyłącznie dodatkowym wykresom/diagnostyce.

Kierunki główne: k0500, wyłącznie istniejące240trainpar po filmach. Rank8 ustalony z góry, brak wyboru na późniejszej trajektorii. Przetworzymy400par w evalfp32 (brakTF32/autocast, zamrożoneBN, fullimage480×320AREA), ale tylko train ustali ważony drugi moment delty i top5osi. Testowepary służą opisowi identyfikacji. Wagi film→obraz→coverage; top5osi wg abs(meanpairedDelta/SD) train. Wyznaczymy U500 i q_i(t)=U500ᵀz_i(t). U500 znane od anchor500.

U1000 i poprzednie osie1-based500/472/735/512/647 wyłącznie retrospektywnie. Żadnej prognozy ocenianej jako przyczynowa w tych kierunkach wyznaczonych z końca źródłowej trajektorii. Root ma własność porownanie_podprzestrzeni/ i PODSUMOWANIE.md; reviewer AUDYT_ASTRA.md.

## Równanie i estymacja

Główny model dla każdego tokenu i, wspólne b,A,K:

```text
q_i(t) = U500ᵀ z_i(t)
x_i(t) = q_i(t) - q_i(500)          # stały, token-specific anchor
Δx_i(t) = x_i(t) - x_i(t-dt)
x_i(t+dt) = x_i(t) + b + A x_i(t)
           + Σ_(ell=0..L-1) K_ell Δx_i(t-ell*dt) + epsilon_i(t+dt)
```

Macierze wspólne dopasowujemy JEDNOCZEŚNIE ridge na przyszrostach. To empiryczny GLE/AR z pamięcią i skończona aproksymacja affine Koopmana/EDMD na zadeklarowanych liniowych obserwablach; nie dokładny Mori–Zwanzig ani dowód globalnej liniowości. L0 oznacza Markov, L1/2/4/8 pamięć. Dwie dodatkowe podłogi: brak zmiany oraz stała ostatnia prędkość. Zero przyszłych q, paczek, gradientów, wag i reszt w rollout; tylko historia do originu. Brak Kalmana.

Anchor tokenu jest stały, poznany w500. Skale poziomu i przyrostu estymujemy wyłącznie z zakończonych par fit; przekładamy wszystkie współczynniki do fizycznych współrzędnych q. NumPy regressor-row @ coef, równanie zapisane dla kolumn, transpozycje jawne. W q wyraz wolny b_i=b-Aq_i(500). Jednostki b/epsilon: aktywacja na jeden krok dt; A,K bezwymiarowe dla tej dyskretyzacji. Nie utożsamiać przyrostu z pędem AdamW.

Regresory: x/state_scale, opóźnione Δx/delta_scale. Cel Δx/delta_scale. Weighted mean intercept niepenalizowany. Gram normalizowany do sumy wag1; ridgegamma = lambda*trace(Gram)/liczba_regresorów. Lambda grid1e-4/1e-3/1e-2/0.1/1. Wagi każdego celu czasowego równe, tokenów coverage z sumą1. Skale, średnie i thresholdnearzero tylko fit. Condition/rank raportowane; ridge nie zapewnia fizycznej identyfikowalności współczynników.

## Przyczynowy podział czasu i wybór

dt1: anchor500, fit targets501..700 dla wszystkichL (L8 korzysta z dostępnych stanów492..500). Validation origins700..749, cele <=799. Główny wybór L/lambda maksymalizuje ŚREDNI skill względem persistence dla H1/10/50, równa waga horyzontów. Osobno zapisujemy najlepszyL0 i najlepszy H1, oba z tej samej siatki validation. Nie dobieramy agregacji po teście. Po wyborze refit końcowy target<=799, bez retuning. Test origins800..950, H1/2/5/10/20/50, jednakowe dla modeli na danymH. H100: osobna diagnostyka origins800..900.

dt10 (checkpoint): anchor500, fit targets510..700co10, valorigins700..740co10. Wybór średni skill H10/50; dodatkowy bestH10; refit targets<=790 (ostatni dostępny przed799). Test origins800..950co10 dla H10/20/50; H100 diagnostyka800..900co10. H1/2/5 niedostępne dla modelu co10: nie nazywać kroku10 jednym krokiem uczenia. Modeldt1 raportować także na wspólnych co10 originach dla porównania dyskretyzacji.

Test uruchamiamy jednokrotnie po audycie kodu i zamrożeniu wyboru. Prefix i zatrucie przyszłego sufiksu muszą dać identyczny rollout. U500 współczynniki/features nie korzystają z obserwacji testu. Stan originu i cała wcześniejsza historia to dozwolone wejście rolling forecast; w obrębie rollout nie ma teacher forcing. Ten bieg i jego późniejsze stany analizowano wcześniej: podział jest chronologiczny dla bieżącego algorytmu, ale nie jest nowym nietkniętym testem ani niezależnym obrazem. Wyniki pozostają eksploracyjne.

## Metryki i szum

MSE ważony coverage, uśredniony po originach i8współrzędnych w FIZYCZNYM q; skill1−MSE/persistence. Cosine wektorów zmian, a nie poziomów, z liczbą ważnych token-originów; zerowe mianowniki jawnie wyłączone. Persistence ma nieokreślony cosine. Signaccuracy wszystkich współrzędnych (zero predykcji nie jest dodatnim/ujemnym znakiem); osobno po filtrze |actualΔ_H,j|>epsilon_j. Epsilon_j=0.1×ważonaSD przyrostów fit, stały dla wszystkichH danej dyskretyzacji. Dodatkowa kontrola znaku: większość znaków przyrostów fit; bez strojenia na teście. Tokeny nie są niezależnymi replikacjami. Per-token błędy i wykresy,8stałych odległych backgroundcontrols opisowo i transfer tego samego modelu.

Moving circular block bootstrap originów300powtórzeń, wspólne losowanie całego kompletu tokenów i modeli. Blok co najmniej50aktualizacji i co najmniejH, dt10 oznacza5sampleoriginów dla50aktualizacji. Przy151originach mamy około3efektywne bloki: CI tylko opisowe dla jednego biegu/sondy. H100: około1efektywnyblok, bez CI; diagnostyka punktowa. Żadnej gwarancji80procent ani monotoniczności zH.

Residualone-step po granicy testu: ciągłe targets801..1000 dla dt1 i810..1000co10 dla dt10. To warunkowy one-step residual z rzeczywistą historią do każdego kroku, osobny od autonomicznych błędówH. ACF i Ljung–Box pertoken/percoordinate w czasie oraz dla ważonego sharedresidual. Gaussowskość osobno: skew/excesskurtosis/Jarque–Bera. Sharednoise/cross-token korelacje percoordinate, nie spłaszczamy tokenów w jedną serię. Raportujemy zmianę wariancji między połówkami, bez FDT lub interpretacji fizycznego whiteBrownian noise.

## Artefakty

U500/top5 oraz dokładne model/datahashes; mały NPZ trajectory q500/top5axes500 i osobny NPZ retrospektywnyU1000/oldaxes; macierze ridge/skale/anchor; wybór validation i JSON metryk; wykresy torów na stałym obrazie, forecastH1/10/50 i error/skillversusH. Pełne toolvectors w RAM przez mmap, żadnej kopii22GB. README odtwarzania, WYNIKI.md z negatywami. Maks150MB łącznie i minimum1GBfree. Ruch obejmuje changingweights/BN; tożsamość semantyczna pozostaje operacyjną hipotezą poprzednich kontroli.
