# Literatura: lifting Koopmana i przyczynowy obserwator pamięci

Sprawdzono 2026-10-01 przez `paper-search`, web oraz callable Undermind (orientation, lookup, metadane i abstrakty). Nie uruchamiano deep search, czytających subagentów, eksperymentów ani testów. Nie pobierano PDF. Zakres weryfikacji każdej pozycji opisuje `SOURCES_OBSERWATOR.json`; część dotyczy tylko abstraktu/metadanych. Numery poniżej są lokalne dla tego dokumentu.

## EDMD a uczony autoencoder

**Klasyczne EDMD nie jest autoencoderem.** Williams, Kevrekidis i Rowley [1] przyjmują pary snapshotów i zadany słownik obserwabli `psi`. Aproksymują operator na przestrzeni tego słownika. Nieliniowe funkcje bazowe nie stają się przez to uczonym encoderem, a regresyjny odczyt stanu nie oznacza architektury AE. Baseline z opóźnieniami, kwadratami, iloczynami i cechami historycznych obrazów należy opisać jako EDMD z ustalonym słownikiem; transformacje i standaryzacja muszą pochodzić z treningu.

Lusch, Kutz i Brunton [2] uczą encoder, decoder i dynamikę latentną, łącząc rekonstrukcję, zgodność ewolucji latentu i predykcję wielokrokową. To inna rodzina modelu. Sama dobra rekonstrukcja nie wystarcza do identyfikacji predykcyjnych współrzędnych. Ich wariant dla ciągłego widma używa również parametrów zależnych od latentu; nie należy utożsamiać całej pracy z jedną stałą macierzą `K`. Wyniki demonstracyjne nie gwarantują dokładnej linearyzacji treningu ViT ani zamknięcia dowolnego krótkiego okna historii.

## Drift, pamięć i obserwacja

GLE może być osadzone w rozszerzonym stanie dzięki przybliżeniu jądra pamięci i zmiennym pomocniczym [3]. Dlatego sensowną hipotezą jest stan `x_t=(z_t,m_t)`, gdzie `z` reprezentuje drift Koopmana, a `m` przechowuje pamięć/kolorową resztę. Przykładowa, projektowa aproksymacja to:

```text
z_(t+1) = A z_t + B u_t + C m_t + w_t
m_(t+1) = F m_t + G z_t + J u_t + eta_t
y_t     = H z_t + D m_t + v_t
```

`u_t` opisuje dostępną historyczną paczkę prowadzącą do następnej aktualizacji; indeks trzeba uzgodnić z logiem. `y_t` jest pomiarem, a nie przyszłą etykietą. Augmentacja nie zapewnia obserwowalności ukrytej pamięci. Dla liniowej wersji trzeba sprawdzić rangę/kondycję odpowiedniej macierzy obserwowalności i diagnozować innowacje. Bryson i Henrikson [4] pokazują, że sekwencyjnie skorelowany szum wymaga specjalnego traktowania oraz że augmentacja może powodować złe uwarunkowanie. Ich praca dotyczy szumu obserwacji; przeniesienie jej na procesową resztę treningu wymaga osobnego modelu generatywnego.

Historyczne cechy obrazów mogą być wejściem albo dodatkową obserwacją zależnie od protokołu. Nie wolno bez analizy traktować cech, stanu i etykiet pochodzących z tego samego obrazu jako niezależnych pomiarów. Ekstraktor musi być zamrożony lub aktualizowany wyłącznie z informacji dostępnej do `t`; użycie przyszłego checkpointu do kodowania historii jest przeciekiem.

Julier i Uhlmann [5] motywują UKF ograniczeniami linearyzacji EKF: propagacja średniej i kowariancji przez nieliniowość wymaga aproksymacji. Lokalny EKF z Jacobianem albo UKF ze sigma-points mogą obsłużyć nieliniowy drift/decoder, ale nie stają się przez to dokładnym filtrem dowolnego wielomodalnego posterioru. Książka Särkkä i Svensson [6] stanowi odniesienie do filtracji, wygładzania i estymacji parametrów; tu sprawdzono jej zakres i metadane, nie wszystkie algorytmy.

Schön, Wills i Ninness [7] pokazują ML/EM dla nieliniowych modeli stanu, z particle smoothing w etapie estymacji. Smoothing offline na zbiorze treningowym może pomagać identyfikacji. Podczas oceny prognozy trzeba jednak inicjalizować stan z filtra warunkowanego wyłącznie obserwacjami do początku prognozy. Estymata wygładzona z przyszłych pomiarów jest innym zadaniem.

## Reguła przyczynowości i marginalny rollout

Reszta `r_(tau,H)=y_(tau+H)-forecast_H(tau)` istnieje dopiero po zakończeniu etykiety. W chwili `t` obserwator może użyć jej tylko gdy `tau+H<=t`. Filtrowanie reszty dla `tau+H>t` przecieka przyszłość, także gdy zapisano ją przy indeksie `tau`. Przy rollout'cie od `t` kolejne kroki wykonują tylko prediction; correction wymaga nowego rzeczywiście dostępnego pomiaru. Nie wolno korygować symulacji rzeczywistymi przyszłymi resztami ani etykietami. Nakładanie okien H-krokowych może samo wytworzyć autokorelację błędów; nie dowodzi pamięci GLE.

Uczciwa prognoza marginalizuje posterior stanu/pamięci i rozkład przyszłych wymuszeń warunkowany historią i polityką znaną w `t`. Można propagować wspólne próbki paczek, pamięci i innowacji; ich zależności muszą odpowiadać oszacowanemu modelowi. Plug-in średniej wejścia przy nieliniowej dynamice nie jest ogólnie średnią prognozy. Rzeczywiste przyszłe paczki oznaczają osobny wariant oracle.

Żadne z tych źródeł nie ustanawia FDT dla AdamW. Gaussowskość procesu nie implikuje białości; szum biały w augmentowanym modelu jest założeniem o innowacji, a obserwowana reszta może pozostawać kolorowa. Nazwa GLE wymaga jawnego jądra lub osadzenia pamięci; sam AR/Kalman nie dowodzi fizycznego mechanizmu Langevina.

## Bibliografia

| ID | Autorzy, rok, tytuł | DOI / pierwotne treści |
|---|---|---|
| 1 | M. O. Williams, I. G. Kevrekidis, C. W. Rowley, 2015. A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition | [10.1007/s00332-015-9258-5](https://doi.org/10.1007/s00332-015-9258-5); [abstrakt autorów](https://arxiv.org/abs/1408.4408) |
| 2 | B. Lusch, J. N. Kutz, S. L. Brunton, 2018. Deep learning for universal linear embeddings of nonlinear dynamics | [10.1038/s41467-018-07210-0](https://www.nature.com/articles/s41467-018-07210-0) |
| 3 | H. Lei, N. A. Baker, X. Li, 2016. Data-driven parameterization of the generalized Langevin equation | [10.1073/pnas.1609587113](https://doi.org/10.1073/pnas.1609587113); [abstrakt autorów](https://arxiv.org/abs/1606.02596) |
| 4 | A. E. Bryson Jr., L. J. Henrikson, 1968. Estimation using sampled data containing sequentially correlated noise | [10.2514/3.29327](https://doi.org/10.2514/3.29327); [NASA](https://ntrs.nasa.gov/citations/19680049568) |
| 5 | S. J. Julier, J. K. Uhlmann, 2004. Unscented filtering and nonlinear estimation | [10.1109/JPROC.2003.823141](https://doi.org/10.1109/JPROC.2003.823141) |
| 6 | S. Särkkä, L. Svensson, 2023. Bayesian Filtering and Smoothing, wyd. 2 | [10.1017/9781108917407](https://doi.org/10.1017/9781108917407); [instytucja autorów](https://research.aalto.fi/en/publications/bayesian-filtering-and-smoothing-3/) |
| 7 | T. B. Schön, A. Wills, B. Ninness, 2011. System identification of nonlinear state-space models | [10.1016/j.automatica.2010.10.013](https://www.sciencedirect.com/science/article/pii/S0005109810004279) |
