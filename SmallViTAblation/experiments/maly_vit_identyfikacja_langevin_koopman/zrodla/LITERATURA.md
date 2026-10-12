# Literatura: identyfikacja dynamiki ViT z historii

Sprawdzono 2026-10-01. Użyto `paper-search` (arXiv/Crossref) i stron pierwotnych: autorzy, wydawcy oraz instytucje autorów. Nie pobierano PDF. Zakres sprawdzenia jest jawny w `SOURCES.json`; abstrakt nie oznacza audytu całego dowodu.

## Synteza i granice uzasadnienia

**Wniosek:** model opóźnionego stanu `s` z historycznymi paczkami `u` jest uzasadnioną hipotezą identyfikacyjną. Literatura nie dowodzi, że skończony wektor tożsamości ViT zamyka dynamikę AdamW, ani że pamięć i wymuszenie można zastąpić białym szumem.

DMDc [1] oddziela regresyjnie dynamikę stanu od wpływu obserwowanego wejścia. KIC [2] rozszerza teorię Koopmana na wejścia i sterowanie, w tym nieliniową zależność wejście-wyjście. Uzasadnia to baseline `z_(t+1)=Az_t+Bu_t` i porównanie z nieliniowymi interakcjami. Samo dopisanie paczki do cech nie gwarantuje identyfikacji wpływu paczki: przy silnej korelacji `s` i `u` różne parametry mogą tłumaczyć tę samą trajektorię. Historyczne wejście musi odpowiadać aktualizacji prowadzącej do etykiety, z jednoznacznym indeksem czasu. To aplikacja teorii do eksperymentu, nie wynik autorów o ViT.

Brunton i współautorzy [3] pokazują ograniczenia skończenie wymiarowych podprzestrzeni niezmienniczych zawierających jawnie stan. Ograniczenia dotyczą dokładnej reprezentacji i zakresu globalnego rozpatrywanych systemów; nie wykluczają użytecznych lokalnych aproksymacji ani wybranych obserwabli. Mały błąd jednokrokowy nie ustanawia dokładnego liftingu, globalnej linearyzacji ani stabilnego długiego rollout'u. Trzeba badać błąd wielokrokowy i zależność od rzędu modelu.

HAVOK [4] motywuje opóźnienia i rozdzielenie dominujących współrzędnych od wymuszenia. W przykładach wymuszenie ma ciężkie ogony; jego kolejność czasowa jest istotna. Autorzy zaznaczają, że symulacja wymaga sygnału wymuszenia dostępnego online. Wynik ten nie uzasadnia prognozy bez przyszłych wejść ani automatycznej obserwowalności stanu optymalizatora na podstawie dowolnego embeddingu. Hankel i transformacje mają być dopasowane na treningu, a każde okno przy predykcji musi kończyć się w `t`.

Mori [5] i data-driven MZ [6] uzasadniają, że redukcja pełnego stanu może pozostawić składnik pamięci i dynamikę ortogonalną. Ukryte momenty AdamW, zmienne harmonogramu i historia próbkowania są zatem sensownymi kandydatami na źródło nie-Markowowskich reszt. To wniosek projektowy: źródła nie identyfikują tych składników dla naszej sieci. Skończone opóźnienia lub pomocniczy stan AR mogą aproksymować pamięć, ale ich skuteczność musi zostać wykazana na danych.

Lei, Baker i Li [7] konstruują aproksymacje jądra pamięci oraz osadzenie GLE w rozszerzonym modelu stochastycznym. Ich relacja fluktuacja-dysypacja wykorzystuje określone statystyki równowagowe. AdamW [8] stosuje adaptacyjne aktualizacje i odłączony weight decay; praca nie dowodzi termicznej równowagi ani FDT. **Nie wolno narzucać FDT dla AdamW bez osobnego sprawdzenia założeń. Gaussowski nie znaczy biały:** proces gaussowski może mieć niezerową autokowariancję dla różnych czasów. Test kształtu histogramu nie zastępuje testu autokorelacji, korelacji z wejściem i zmian wariancji.

Kalman [9] rozróżnia obserwowalność, sterowalność i minimalną realizację: relacja wejście-wyjście ujawnia tylko odpowiednią część stanu liniowego systemu. Willems i współautorzy [10] pokazują rolę dostatecznie bogatego pobudzenia dla sterowalnych systemów LTI. Dla ViT potrzebne są diagnostyki rzędu/SVD macierzy regresorów, kondycji i stabilności współczynników. Twierdzenie LTI nie jest dowodem persistent excitation ani identyfikowalności nieliniowego treningu. Ljung [11] jest odniesieniem metodologicznym do system identification; sprawdzono tutaj tylko metadane książki na stronie autora.

## Konsekwencje dla protokołu

1. Predyktor w `t` korzysta wyłącznie z `s_(<=t)`, `u_(<=t)` i polityki znanej w `t`. Przyszłe stany służą jako etykiety.
2. Wielokrokowa prognoza wymaga osobnego modelu przyszłego wymuszenia: `p(u_(t+1:t+h)|historia do t,polityka do t)`. Znany harmonogram można uwzględnić; realizacje przyszłych losowanych paczek i gradienty są niedostępne. Zapisany seed pozwalający odtworzyć przyszłą sekwencję oznacza wariant warunkowy/oracle, jeśli nie jest elementem jawnej polityki dostępnej w zadaniu.
3. Wynik z rzeczywistymi przyszłymi paczkami raportować oddzielnie jako oracle. Nie mieszać go z uczciwą prognozą z historii. DMDc nie dostarcza automatycznie modelu przyszłego wejścia.
4. Porównać persistence, model bez `u`, model z historycznym `u`, model opóźnień i model z kolorowymi resztami. Dodać ablacją permutacji paczek i długości historii. Dobór wariantu tylko na walidacji; test poza doborem.
5. Raportować błędy po horyzoncie, autokorelację reszt i kalibrację niepewności. Przewaga predykcyjna nie dowodzi tożsamości fizycznej, przyczynowości paczek ani dokładnego GLE.

## Bibliografia pierwotna

| ID | Autorzy, rok, tytuł | DOI / źródło |
|---|---|---|
| 1 | J. L. Proctor, S. L. Brunton, J. N. Kutz (2016), Dynamic Mode Decomposition with Control | [10.1137/15M1013857](https://doi.org/10.1137/15M1013857); [preprint](https://arxiv.org/abs/1409.6358) |
| 2 | J. L. Proctor, S. L. Brunton, J. N. Kutz (2018), Generalizing Koopman Theory to Allow for Inputs and Control | [10.1137/16M1062296](https://doi.org/10.1137/16M1062296); [preprint](https://arxiv.org/abs/1602.07647) |
| 3 | S. L. Brunton, B. W. Brunton, J. L. Proctor, J. N. Kutz (2016), Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control | [10.1371/journal.pone.0150171](https://doi.org/10.1371/journal.pone.0150171) |
| 4 | S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, J. N. Kutz (2017), Chaos as an intermittently forced linear system | [10.1038/s41467-017-00030-8](https://www.nature.com/articles/s41467-017-00030-8) |
| 5 | H. Mori (1965), Transport, Collective Motion, and Brownian Motion | [10.1143/PTP.33.423](https://academic.oup.com/ptp/article/33/3/423/1925580) |
| 6 | Y. T. Lin, Y. Tian, M. Anghel, D. Livescu (2021), Data-Driven Learning for the Mori-Zwanzig Formalism: A Generalization of the Koopman Learning Framework | [10.1137/21M1401759](https://doi.org/10.1137/21M1401759); [preprint](https://arxiv.org/abs/2101.05873) |
| 7 | H. Lei, N. A. Baker, X. Li (2016), Data-driven parameterization of the generalized Langevin equation | [10.1073/pnas.1609587113](https://doi.org/10.1073/pnas.1609587113); [preprint](https://arxiv.org/abs/1606.02596) |
| 8 | I. Loshchilov, F. Hutter (2019), Decoupled Weight Decay Regularization, ICLR | DOI wydawniczy nieustalony; [preprint z 2017](https://arxiv.org/abs/1711.05101), DOI repozytorium `10.48550/arXiv.1711.05101` |
| 9 | R. E. Kalman (1963), Mathematical Description of Linear Dynamical Systems | [10.1137/0301010](https://doi.org/10.1137/0301010) |
| 10 | J. C. Willems, P. Rapisarda, I. Markovsky, B. L. M. De Moor (2005), A note on persistency of excitation | [10.1016/j.sysconle.2004.09.003](https://doi.org/10.1016/j.sysconle.2004.09.003); [instytucja autora](https://researchportal.vub.be/en/publications/a-note-on-persistency-of-excitation/) |
| 11 | L. Ljung (1999), System Identification: Theory for the User, wyd. 2, Prentice Hall | DOI nieustalony; [strona autora](https://www.rt.isy.liu.se/en/books/sysid/) |
