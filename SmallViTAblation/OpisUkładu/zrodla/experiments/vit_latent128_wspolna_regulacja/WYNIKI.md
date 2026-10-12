# Wspólna regulacja J128: wynik pilota

Zamrożony ViT k1000, stałe J128 z k500. Jeden wspólny regulator na wszystkie obrazy i tokeny. Parametry wybierane na val; żadnych etykiet testu w interwencji.

| Metoda | Split | AP50 | AP50:95 | Loss (po filmach) |
|---|---|---:|---:|---:|
| baseline | train | 0.453273 | 0.134857 | 2.330511 |
| baseline | val | 0.402075 | 0.112166 | 2.575332 |
| baseline | test | 0.457589 | 0.114275 | 2.485679 |
| baseline | all | 0.437555 | 0.120881 | 2.411157 |
| selected | train | 0.509102 | 0.163148 | 1.496559 |
| selected | val | 0.507648 | 0.141723 | 1.905237 |
| selected | test | 0.367857 | 0.106870 | 1.969341 |
| selected | all | 0.468564 | 0.142983 | 1.675407 |
| static | train | 0.452715 | 0.138552 | 2.102852 |
| static | val | 0.465949 | 0.124368 | 2.348232 |
| static | test | 0.465775 | 0.117622 | 2.277830 |
| static | all | 0.449718 | 0.125120 | 2.187707 |

Wybór ogólny: adapter16_lr001, epoka 20.
Najlepszy statyczny regulator: affine_lr001, epoka 40.
Bramka pilota (AP50 i AP50:95 test > baseline): False.
Czas całości: 440.9 s; urządzenie: NVIDIA GeForce RTX 3070.

## Znaczenie i ograniczenia

Stały regulator to skale/przesunięcia wspólne dla wszystkich danych. Adapter jest osobną rodziną: wspólna funkcja zależna od wartości tokenu, bez identyfikatora obrazu i GT. Wagi bazowego ViT i pozostałe640 kanałów niezmienione. Wybrane wartości regulatora zapisano w modelach; KANALY_STATYCZNE.csv opisuje jego skale i przesunięcia.
Nie jest to ten sam rodzaj wyniku co oracle AP50=0.884748 z etykietą każdego obrazu. Nie gwarantuje optimum globalnego, pełnej tożsamości narzędzia ani istnienia zdjęcia odtwarzającego dany latent. Historyczne grupy były wcześniej oglądane; to pilot jednego seeda, nie niezależne potwierdzenie końcowe.

## Kontrole

PREFLIGHT.json: zgodność szybkiego tail z pełnym forwardem i gradientami, zakres GELU, niezmienność640 kanałów, rozłączność filmów. VERIFICATION.json: kontrola wybranej interwencji w pełnym forwardzie na nowych przykładach, hash wag i buforów.
SELECTION.json powstaje przed obliczeniem metryk testu. Historia wszystkich konfiguracji i wyniki negatywne zachowane. Przedziały ufności resamplują całe filmy i są warunkowe względem już wybranej konfiguracji; nie usuwają efektu wielokrotnego oglądania historycznego benchmarku.

## Przedziały ufności różnicy względem baseline

| Regulator | Metryka | Dolna granica 95% | Górna granica 95% |
|---|---|---:|---:|
| selected | AP50 | -0.201090 | 0.025938 |
| selected | AP50_95 | -0.036121 | 0.021975 |
| selected | loss | -0.875497 | -0.165258 |
| static | AP50 | -0.016814 | 0.053704 |
| static | AP50_95 | -0.008794 | 0.014283 |
| static | loss | -0.293246 | -0.124172 |
