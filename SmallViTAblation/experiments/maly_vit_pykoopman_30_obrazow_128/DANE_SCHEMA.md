# Schemat danych ekstrakcji

`LATENT_J128.npy`: float32 `[101,34,600,128]`, order C, mmap read-only. Pierwsza oś odpowiada checkpoint_steps z DANE.json: 0,10,...,1000 aktualizacji. Drugą osią jest indeks obrazu w MANIFEST.json; trzecia to token y*30+x, czwarta to kanał channels0[j] w oryginalnym 768D wyjściu GELU. Brak PCA, nowego rankingu ani uśredniania przestrzennych cząsteczek.

MANIFEST.json jest obiektem z records, fit_indices i holdout_indices. Fit: `[0,1,2,3,4,5,6,7,12,...,33]`, 30 obrazów. Holdout: dokładnie `[8,9,10,11]`, te same cztery frame-ID co wcześniej. Tablica nie jest sortowana na fit/test, aby nie zmienić dawnych indeksów. Dotychczasowe12 są zachowane; 22 nowe obrazy to odrębne filmy spoza wszystkich12 dawnych filmów.

`MASKI.npz`: tool, background, tool25 bool `[34,600]`; coverage oraz area_coverage float32 `[34,600]`; mask_input bool i mask_input_area float32 `[34,320,480]`; token_xy int32 `[600,2]`. Coverage jest udziałem binarnej maski w patchu16×16, area_coverage uwzględnia uśrednienie maski natywnej podczas zmiany rozmiaru. Tool to wszystkie patchy z oryginalną maską. Background odpowiada source-rule: poza dylatacją maski; nie jest dowodem braku innych narzędzi w klatce.

Przy modelowaniu cząsteczek wybierz ids=flatnonzero(tool[image]), pozostaw q[:,ids,:] jako wszystkie współrzędne każdego tokenu. Wagi tokenowe: coverage[image,ids]/sum(coverage[image,ids]), jak w poprzednich eksperymentach. Każdy obraz może mieć równą wagę nadrzędną. Średnia tła, jeśli potrzebna, to oddzielnie mean(q[:,background[image],:],axis=1). Średnia cząsteczek nie zastępuje ich indywidualnych trajektorii.

`KANALY_J128.npz`: channels0 i channels1=channels0+1, 128 indeksów w ustalonej kolejności. `OBRAZY.npy`: RGB uint8 `[34,320,480,3]`. PNG w obrazy/ są dokładną wersją tego wejścia. Source model sam normalizuje tensor obrazów RGB/255 mean/std ImageNet. J zidentyfikowano przy k500; historyczna analiza przed500 w tych kanałach jest retrospektywna.

DANE.json zostaje zapisane dopiero po zakończeniu wszystkich101 checkpointów i sprawdzeniu finite. Zawiera checkpoint_steps, shape, fit_indices, holdout_indices, channels0/1, sumySHA256, hash architektury i listę hashy checkpointów, preprocessing, sprzęt/wersje oraz zgodność12 dawnych sond z LATENT_GESTY. EKSTRAKCJA_POSTEP.json nie oznacza gotowości całego pliku. Nie wczytuj niezakończonego mmap do fitu.

Brak zapisanych rzeczywistych batch-ID/augmentacji/gradientów dla tego source run. Pliki net+krok co10 zachowują wagi i bufory BatchNorm. Ekstrakcja jest eval/no-grad, nie nowym treningiem. Model dynamics nie może zobaczyć przyszłych rzeczywistych pomiarów podczas rollout.
