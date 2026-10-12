# Jeden obraz SD z celu J128, 2026-10-04

## Pytanie i aparat

Wygenerowano jedną klatkę z SD 1.5 Cataract-LMM EMA 20000, prowadząc odszumianie zapisanym celem J128 z eksperymentu o AP50 0,88474795. Wzięto pierwszy rekord manifestu: TR_0001_S1_P03, klatka 26. Start img2img z tego obrazu, z prawdziwymi maskami narzędzia, rogówki i źrenicy. Zamrożony ViT k1000, pełny latent 600 × 768, cele na 128 kanałach każdego tokenu, pozostałe współrzędne jako składnik zachowania obrazu. Ziarno 20261004. Lokalna RTX 3070; obliczenia i ładowanie około 86 sekund.

## Wynik

- `outputs/vit_latent128_obraz_sd/OBRAZ_SD.png`: 576 × 384, RGB, jedna wygenerowana klatka.
- MSE na J128 względem celu, obraz źródłowy: 19,464159.
- MSE końcowego PNG podzielony przez MSE źródła: 0,99938804. Redukcja błędu: **0,061196 procent**, czyli znikoma.
- Względny MSE pozostałych kanałów względem ich energii w źródle: 0,0281072.
- Pewność najwyżej ocenionej detekcji: źródło 0,832321; PNG z SD 0,854640; bezpośrednie podstawienie docelowego latentu w obrazie źródłowym 0,869089.
- To są wyniki jednej klatki, nie AP50 ani mAP. Wzrost pewności nie wystarcza do oceny poprawności ramki. Ramkę źródłową można porównywać tylko przy założeniu zachowanej geometrii; nie ma niezależnej adnotacji wygenerowanego obrazu.
- Oględziny: zachowana scena oka i narzędzia; zmieniony wygląd i szczegóły, zniekształcone napisy w dolnej części kadru.

## Punkt odniesienia

Punktem odniesienia jest obraz źródłowy, ten sam ViT i preprocessing. Końcową ocenę wykonano ponownie z zapisanego PNG uint8, przez rzeczywisty przebieg ViT, bez podmieniania jego aktywacji. Przed startem zgodność oryginalnych J128 ze zapisanym plikiem celu sprawdzono numerycznie. Gradient prowadzenia był skończony i niezerowy we wszystkich aktualizacjach. Brak kontroli SD bez prowadzenia, więc nie przypisujemy zmiany pewności samej metodzie prowadzenia.

## Wniosek i ograniczenia

Generowanie z prowadzeniem latentem działa wykonawczo, ale **ta próba nie odtworzyła istotnie zoptymalizowanego stanu latentu**. Przy tym celu dotychczasowy przepis SD nie przenosi dużej poprawy uzyskanej przez bezpośrednie modyfikowanie wnętrza ViT. To nie dowód niemożliwości inwersji: jeden obraz i jeden zestaw parametrów nie rozstrzygają osiągalności celu. Skokowy krok prowadzenia nie gwarantuje monotonicznego spadku energii; przebieg zapisano, wraz ze wzrostami błędu. Obraz źródłowy, maski i prior SD współtworzą wynik, więc nie jest to jednoznaczne dekodowanie samego wektora.

## Artefakty i pochodzenie

W katalogu wyników: `RESULT.json`, `BIEG_sd_0000.json`, `LATENT_cel_0000.npz`, `LATENT_obrazu_sd.npz`, `MASKI_0000.npy`, `OBRAZ_zrodlowy.png`, `PROGRESS.json`. JSON zawiera parametry, urządzenie, seed, sumy SHA256 checkpointów, celu oraz finalnego PNG. Kod i sposób odtworzenia są w bieżącym katalogu eksperymentu.
