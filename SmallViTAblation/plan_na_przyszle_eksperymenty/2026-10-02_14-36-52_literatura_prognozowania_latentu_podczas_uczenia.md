# Prognozowanie latentu podczas uczenia ViT: alternatywy dla NNDMD i Koopmana

Data: 2026-10-02. Dokument jest przeglądem i propozycją porównań, nie raportem nowych obliczeń GPU. Nie zmienia uzgodnionego treningu ani kodu.

## Najważniejszy wniosek

Istnieją prace przewidujące zmiany **wag uczonej sieci** oraz prace modelujące **niepełne obserwacje układów dynamicznych**. Nie znaleziono w sprawdzonym materiale dowodu, że konkretna metoda przewyższa NNDMD w naszym dokładnym zadaniu: ten sam obraz, zmieniające się checkpointy ViT, pełne reprezentacje tokenów i prognoza bez przyszłych pomiarów. Najbardziej uzasadnione porównanie obejmuje mały model bezpośredniej prognozy wielohoryzontowej, model z historią oraz lokalny operator Koopmana. Pamięć Mori–Zwanziga uzasadnia badanie historii, ale sama teoria nie gwarantuje trafności.

## Zakres i rzeczywisty stan wyszukiwania

Użyto skilla projektu `paper-search`, wyszukiwania internetowego źródeł pierwotnych i Undermind DeepSearch. DeepSearch **zakończył się**, zwracając 108 uszeregowanych publikacji. Nazwa: *Alternatives to Koopman for fixed image latent evolution during neural network training*.

[Wyniki Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Alternatives%20to%20Koopman%20for%20fixed%20image%20latent%20evolution%20during%20neural%20network%20training).

Nie wszystkie 108 prac przeczytano w całości. Ukierunkowane odczyty PDF przez Undermind obejmowały Dogra–Redman, Koopman Message Passing, NiNo, Gradient Flow Matching, Mori–Zwanzig, MZ-AE i Koopa. Dla części pozostałych źródeł sprawdzono abstrakt, metadane i stronę pierwotną. Liczby dotyczące Dogra–Redman dodatkowo zweryfikowano w HTML oryginału. Automatyczne odpowiedzi czytnika potraktowano jako materiał pomocniczy: np. brak wielu trajektorii jest ograniczeniem dowodów dla GFM, a nie matematycznym zakazem trenowania dowolnej adaptacji na jednej historii.

## Co dokładnie prognozujemy

Stan `z[i,p,k]` to 768 współrzędnych cząsteczki `p` obrazu `i` przy checkpointcie `k`. Makrowektory to wspólne osie; ruch oznacza zmianę współrzędnych tej samej cząsteczki. Nie prognozujemy ruchu narzędzia między kolejnymi klatkami filmu. Obraz pozostaje stały, zmieniają się wagi ViT.

Zgodnie z aktualnym planem: dokładnie 1500 obrazów źródłowego treningu, 600 tokenów na obraz, 101 checkpointów o indeksach 0–100, odstęp dziesięciu aktualizacji ViT. Model dostaje pełne 768 kanałów. Zestaw 128 oryginalnych kanałów tożsamości służy do wydzielonej oceny, bez PCA. Batch to 32 **obrazy**, a nie 32 tokeny. Trening korzysta z celów do checkpointu 90; końcowy test to zamrożona prognoza 90→100.

Sama reprezentacja tokenu nie musi być zamkniętym stanem Markowa. Jej następny ruch zależy również od wag sieci, momentów AdamW, learning rate, kolejności batchy i gradientów. Dwa podobne latenty mogą mieć inną przyszłość przy różnych ukrytych stanach uczenia. Większy zbiór obrazów zwiększa pokrycie obserwacji, ale 1500 obrazów tego samego biegu nie stanowi 1500 niezależnych przebiegów treningu.

## Źródła bezpośrednio związane z uczeniem sieci

| Publikacja, autorzy, rok | Co bada i dlaczego ma znaczenie | Granica przeniesienia |
| --- | --- | --- |
| [Optimizing Neural Networks via Koopman Operator Theory](https://arxiv.org/abs/2006.02361), A. S. Dogra, W. T. Redman, 2020 | Przewidywanie wag i biasów; lokalne operatory dla grup parametrów związanych z neuronami. | Małe sieci feedforward; wynik sukcesu dotyczy obniżenia loss, nie dokładności 768 kanałów tokenu. |
| [Learning Linear Embeddings for Non-Linear Network Dynamics with Koopman Message Passing](https://arxiv.org/abs/2305.09060), K. F. Yeh, P. Flood, W. T. Redman, P. Liò, 2023 | Uczony operator na grafie parametrów uwzględnia ich współzależność. | Eksperymenty używają wielu niezależnych małych sieci: 10000 trajektorii na konfigurację. Nie jest to 10000 tokenów jednego ViT. |
| [Learning to Boost Training by Periodic Nowcasting Near Future Weights](https://proceedings.mlr.press/v202/jang23b.html), J. Jang i współautorzy, 2023 | WNN przewiduje bliskie przyszłe wagi, okresowo skracając trening. | Prognoza parametrów; nie potwierdza przewagi dla tokenów obserwowanych na stałym obrazie. |
| [Accelerating Training with Neuron Interaction and Nowcasting Networks](https://arxiv.org/abs/2409.04434), B. Knyazev, A. Moudgil, G. Lajoie, E. Belilovsky, S. Lacoste-Julien, 2024/ICLR 2025 | NiNo wykorzystuje pięć historycznych stanów parametrów i interakcje neuronów; uczy bezpośrednie prognozy wielu horyzontów. | Meta-trening na wielu modelach; transfer całej metody do naszej jednej historii byłby nowym eksperymentem. |
| [Gradient Flow Matching for Learning Update Dynamics in Neural Network Training](https://arxiv.org/abs/2505.20221), X. Shou, Y. Ding, J. Gao, 2025, preprint | Uczy pole zmian parametrów ze zbioru trajektorii; prognoza bez przyszłych gradientów. | Wejście pola to wagi i czas; nie ma jawnego wejścia z momentami optymalizatora. Reprezentacje tokenów nie są wagami. |
| [How Do the Architecture and Optimizer Affect Representation Learning?](https://arxiv.org/abs/2405.17377), Y. Sharon, Y. Dar, 2024, preprint | Bezpośrednio analizuje ewolucję reprezentacji podczas uczenia, wpływ architektury i optymalizatora. | CKA i sondy opisują zmianę reprezentacji; nie są walidacją przyszłej trajektorii pojedynczego tokenu. |
| [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720), J. Lee i współautorzy, 2019 | Uzasadnia liniową aproksymację odpowiedzi sieci w określonym reżimie szerokości i treningu. | Nie gwarantuje liniowości skończonego ViT z AdamW. JVP może służyć do lokalnej diagnostyki. |
| [Measuring Structured Predictability in Neural Training Dynamics: A Cross-Regime Study](https://arxiv.org/abs/2608.15483), F. Wang, W. Tang, H. Qi, 2026, preprint | Pokazuje zróżnicowaną przewidywalność fragmentów parametrów i reżimów treningu. | Dotyczy wag; nie uprawnia do założenia, że wszystkie kanały naszych cząsteczek są równie przewidywalne. |
| [Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW](https://arxiv.org/abs/2608.19762), K. Liu, S. Li, 2026, preprint | Analizuje odpowiedź AdamW z uwzględnieniem wag i momentów na perturbacje batchy. | Analiza odpowiedzi przy określonym przyszłym przebiegu nie jest bezwarunkową prognozą przy nieznanych przyszłych batchach. |

### Audyt horyzontów Dogra–Redman

Źródło: [oryginalny tekst](https://arxiv.org/html/2006.02361), sekcje 4–5 i S9–S10. Nie przeliczamy epok na aktualizacje bez informacji z publikacji.

| Eksperyment | Historia do wyznaczenia operatora | Prognoza | Optymalizator i batch |
| --- | --- | --- | --- |
| Solver ODE, 1:10:10:2 | Aktualizacje 35000–45000: okno 10000 iteracji, późny etap | 2500 iteracji | Adam, Adagrad lub Adadelta; stały batch, jego rozmiar niepodany w tej tabeli |
| MNIST, 784:20:20:20:10 | Początek epoki 3 do końca 5; publikacja podaje 938 iteracji/epokę | Dwie epoki po epoce 5 | Stochastic Adadelta; batch treningowy 64 |

W §4.1 „success” oznacza `T_eq/T > 0`: przewidziane wagi obniżyły loss względem chwili rozpoczęcia prognozy. Raportowane 100 procent sukcesu MNIST nie oznacza 100 procent zgodności znaków ani małego RMSE wszystkich parametrów. §5 wyraźnie rozróżnia podobną jakość loss i dokładne odtworzenie wag. Algorytm używa przybliżenia `U = F' F^+`, bez wykazanego tu doboru ridge. Warunki lokalnej dynamiki i tysiące gęstych obserwacji istotnie różnią się od naszej historii 101 zapisów. Architektonicznego grupowania wag nie należy utożsamiać z patchami obrazu.

## Źródła metodologiczne: użyteczne analogie

| Publikacja, autorzy, rok | Mechanizm do sprawdzenia | Co faktycznie walidowano |
| --- | --- | --- |
| [Data-driven learning for the Mori-Zwanzig formalism](https://arxiv.org/abs/2101.05873), Y. T. Lin, Y. Tian, M. Anghel, D. Livescu, 2021 | Rozdzielenie członu chwilowego, pamięci i nierozwiązanego wpływu. | Układy dynamiczne, m.in. Lorenz-96; nie latent uczonego ViT. |
| [Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder](https://arxiv.org/abs/2310.10745), P. Gupta, P. J. Schmid, D. Sipp, T. Sayadi, G. Rigas, 2023, preprint | Nieliniowy autoencoder, operator Koopmana i domknięcie pamięci, także RNN/LSTM. | Dynamika płynów i Kuramoto–Sivashinsky; poprawne statystyki długoterminowe nie oznaczają trafnej pojedynczej trajektorii. |
| [Operator Inference of Non-Markovian Terms](https://doi.org/10.1007/s10915-021-01580-2), W. I. Uy, B. Peherstorfer, 2021 | Regresja członów pamięci przy częściowej obserwacji stanu. | Modele zredukowane; dowód dla naszego zadania wymaga osobnego testu. |
| [Online dynamic mode decomposition for time-varying systems](https://arxiv.org/abs/1707.02876), H. Zhang, C. W. Rowley, E. A. Deem, L. N. Cattafesta, 2017/2019 | Lokalny operator, okno lub zapominanie, aktualizacje rekurencyjne. | Układy zmienne w czasie. Aktualizacja z nowego prawdziwego pomiaru jest innym zadaniem niż zamrożona prognoza. |
| [Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://arxiv.org/abs/2305.18803), Y. Liu, C. Li, J. Wang, M. Long, 2023 | Rozdzielenie stabilnej i lokalnie zmiennej dynamiki. | Zwykłe szeregi czasowe. Wariant Operator Adaptation korzysta z napływających obserwacji; nie wolno podawać ich w naszym zamrożonym teście. |
| [Next generation reservoir computing](https://arxiv.org/abs/2106.07688), D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, 2021 | Tania nieliniowa autoregresja z opóźnieniami i ridge. | Fizyczne układy chaotyczne; wszystkie kwadratowe interakcje 768 kanałów byłyby kosztowne. |

## Pięć modeli w kolejności testowania

1. **Bezpośrednia prognoza wielohoryzontowa przyrostu.** Mały MLP lub TCN przyjmuje pełne 768 kanałów, kilka ostatnich przyrostów i znaną fazę treningu. Zwraca dziesięć prognoz przyrostu względem początku okna. Nie musi wielokrotnie stosować tej samej błędnej macierzy. To tania kontrola hipotezy, czy ograniczenie jednej macierzy przejścia utrudnia prognozę.
2. **Model z historią: strukturalna autoregresja i mały GRU.** Najpierw tania regresja historii, potem nieliniowy wariant. Pamięć obejmuje przykładowo 3–5 poprzednich stanów i przyrostów. Dodać przyczynowy wskaźnik wspólnego ruchu sieci obliczony wyłącznie z dostępnych reprezentacji. Nie wykonywać PCA wybranych kanałów tożsamości.
3. **Lokalny lub warunkowany czasem Koopman.** Operator bazowy z małą korektą zależną od historii i znanego learning rate albo kilka lokalnych reżimów. Macierz nie jest aktualizowana prawdziwymi stanami 91–100 podczas testu. Sprawdzić normy potęg i wzmacnianie błędu, nie tylko promień spektralny.
4. **Koopman z domknięciem pamięci MZ-AE/GLE.** Nieliniowy encoder i decoder, liniowy człon chwilowy oraz uczona korekta historyczna; wspólna strata dziesięciu kroków. Ablacja bez pamięci rozstrzyga, czy pamięć pomaga. Nazwa Mori–Zwanzig nie dowodzi, że wyuczona sieć realizuje dokładne jądro z formalnej projekcji.
5. **Model informowany optymalizatorem i odpowiedzią Jacobianu.** Lokalnie `Delta z ≈ J_theta z · Delta theta`. Porównać udział zmian wag, momentów AdamW i fazy treningu. W przyszłości trzeba prognozować aktualizacje lub dostarczyć legalnie znane sterowanie. Użycie rzeczywistych przyszłych aktualizacji wag daje test z informacją uprzywilejowaną, który musi być podpisany jako oracle.

Transformer, rozbudowany SSM i neural ODE nie są automatycznie lepsze. Przy jednej historii o 101 zapisach najpierw warto sprawdzić powyższe małe modele i koszt danych. Zwiększenie macierzy nie zastępuje brakujących informacji o układzie.

## Uczciwy protokół porównania

Używać tego samego manifestu 1500 obrazów, stałego preprocessingu, tej samej warstwy ViT i pozycji tokenów. Normalizacja oraz wszelkie skalowanie muszą powstać na historii dostępnej przed daną prognozą. Obrazy należące do treningu ViT są zgodne z wymaganiem użytkownika; odłożenie przyszłego czasu sprawdza przyszłość znanych obrazów, nie generalizację na nowe sceny.

Ustawienia wybierać na wcześniejszych początkach prognoz, np. 50→60, 60→70 i 70→80, z dopasowaniem tylko na dostępnych prefiksach. Wyników 90→100 nie używać do strojenia. Po wyborze konfiguracji dopasować finalny model na oknach `k→k+10`, których wszystkie cele są nie późniejsze niż 90; następnie zamrozić.

Każde okno uczy cały rollout: encoder koduje początek, kolejne stany są prognozowane bez podmiany stanami rzeczywistymi. Cele treningowe służą do straty. Przykładowa strata to średnia MSE po wszystkich dziesięciu horyzontach, 768 kanałach, tokenach i 32 obrazach. To średnia **błędów**, nie średni latent batcha. Walidacja nie aktualizuje wag. Można porównać równe wagi horyzontów i dodatkową wagę końca okna, wybierając ją wcześniej.

Raportować RMSE pełnych 768 kanałów oraz oddzielnie zestawu 128; każdy horyzont 1–10, wynik końcowy i wynik całej trajektorii. RMSE globalny to pierwiastek ze wspólnego MSE, a nie średnia pierwiastków z poszczególnych obrazów. Dodać RMSE przyrostów, podobieństwo kierunku, trafność znaku z kontrolą niemal zerowych zmian, normy stanów i względną poprawę wobec braku zmiany. Sama trafność znaku nie mierzy wielkości ruchu.

Kontrole obowiązkowe: brak zmiany, ostatni przyrost, tłumiony trend, powrót do historycznej średniej, strukturalny model liniowy z historią i obecny NNDMD z tym samym budżetem. Wszystkie muszą mieć to samo dostępne sterowanie. Nie podawać przyszłych batchy, przyszłych latentów ani przyszłych momentów AdamW tylko jednemu wariantowi. Dodatkowy test online z korektami po obserwacji można wykonać osobno i tak go nazwać.

Przy RTX 3070 8 GB używać fragmentów tokenów i akumulacji gradientów zachowującej równy udział obrazów. Pełny eksport FP32 ma około 279 GB dziesiętnych: `1500*101*600*768*4`. Potrzebne są shardy, streaming lub jawnie zwalidowany format FP16; nie zakładać, że cały zbiór zmieści się na obecnym dysku. Koszt ekstrakcji checkpointów należy uwzględnić w porównaniu całego pipeline'u.

Powtórzenia seedów modelu prognozującego mierzą warunkową stabilność dopasowania. Przedziały niepewności z tysięcy skorelowanych tokenów nie zastępują niezależnych seedów treningu ViT. Docelowo potrzebne są inne przebiegi ViT. Obecny test dziesięciu checkpointów obejmuje sto aktualizacji ViT; nie potwierdzi celu pięciuset aktualizacji. Nie ma podstaw do gwarantowania 85 procent trafności przed obliczeniami.

## Co literatura zmienia w diagnozie

Słaby wynik nie rozstrzyga, że Koopman jest niewłaściwy. Może wynikać z braku domknięcia stanu, zmiany reżimu treningu, niestabilnego rolloutu, niedopasowanej straty lub przecenienia liczby niezależnych danych. Nie rozstrzyga też, że większa sieć poprawi wynik. Pierwszy konkretny test to porównanie pełnostanowego NNDMD ze wspólną historią i prostą bezpośrednią prognozą dziesięciu horyzontów, przy identycznym podziale czasu i budżecie. Dopiero ten wynik uzasadni wybór dalszego modelu Langevina, Koopmana albo modelu nieliniowego.
