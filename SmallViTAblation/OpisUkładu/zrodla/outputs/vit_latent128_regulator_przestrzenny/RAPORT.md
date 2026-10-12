# Przestrzenne ustawianie J128: wynik

Wspólne parametry regulatora; checkpoint k1000 i640 pozostałych kanałów zamrożone. Wybór na val przed testem. Benchmark historycznie oglądany, wynik eksploracyjny.

| Wariant | Split | AP50 | AP50:95 | Loss |
|---|---|---:|---:|---:|
| baseline | train | 0.453273 | 0.134857 | 2.330511 |
| baseline | val | 0.402075 | 0.112166 | 2.575332 |
| baseline | test | 0.457589 | 0.114275 | 2.485679 |
| baseline | all | 0.437555 | 0.120881 | 2.411157 |
| selected | train | 0.644604 | 0.272832 | 0.836495 |
| selected | val | 0.489307 | 0.132588 | 2.510699 |
| selected | test | 0.425228 | 0.121785 | 2.471075 |
| selected | all | 0.572220 | 0.214222 | 1.506638 |
| static | train | 0.438725 | 0.136739 | 2.204512 |
| static | val | 0.451874 | 0.115655 | 2.449351 |
| static | test | 0.445213 | 0.113506 | 2.390016 |
| static | all | 0.434026 | 0.121935 | 2.291438 |

Wybrany regulator: spatial64_focal, epoka 20.
Cel AP50>0,90 na test: False.
Czas kampanii: 1510.8 s.

Oracle train ma własny raport TEACHER_RESULT.json i wykorzystuje etykiety tylko treningu. Jego wynik nie jest skutecznością regulatora na nowych obrazach. Nie znaleziono naturalnych zdjęć realizujących zadane latenty. Nie wykazano optimum globalnego ani domknięcia tożsamości.

## Porównanie i ograniczenia

Podłoga używa tych samych obrazów, modelu, dekodera i AP. Regulator warunkowy przewiduje korektę z cech, a static używa wspólnych skal/przesunięć. Przedziały ufności resamplują całe filmy i nie usuwają wpływu wielokrotnego oglądania starego benchmarku.

## Przedziały 95% różnicy wobec baseline

| Wariant | Metryka | Dolna | Górna |
|---|---|---:|---:|
| selected | AP50 | -0.138220 | 0.088548 |
| selected | AP50_95 | -0.022045 | 0.040803 |
| static | AP50 | -0.036295 | 0.010630 |
| static | AP50_95 | -0.013444 | 0.006974 |
