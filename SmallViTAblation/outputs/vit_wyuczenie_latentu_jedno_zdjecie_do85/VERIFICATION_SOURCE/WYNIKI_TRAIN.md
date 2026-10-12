# Naturalny latent do85 procent

Primary agreement>=85: NIEOSIĄGNIĘTE.

| Pomiar | Oryginał0 | Wyuczone1000 | Wyuczone2000 | Final | Best |
|---|---:|---:|---:|---:|---:|
| MSE128 | 85.9211807 | 10.9162397 | 8.67640686 | 5.73468113 | 5.73468113 |
| MSE640 | 0 | 4.18178752e-06 | 3.36482549e-06 | 1.89167304e-06 | 1.89167304e-06 |
| MSE768 | 14.3201971 | 1.81937671 | 1.44607067 | 0.955781758 | 0.955781758 |
| agreement128 | 0.335600708 | 64.4756232 | 68.3291376 | 74.2519364 | 74.2519364 |
| cosine128 | 0.106738292 | 0.943290353 | 0.951030314 | 0.966364026 | 0.966364026 |
| confidence | 0.867869496 | 0.771777928 | 0.759447038 | 0.74426806 | 0.74426806 |
| iou | 0.856594683 | 0.999952275 | 0.999883696 | 0.999946511 | 0.999946511 |
| GTloss | 0.490047604 | 0.154576004 | 0.151910305 | 0.132014871 | 0.132014871 |

## Pytanie i aparat

Czy dalsze uczenie tych samych norm2/fc1 od kroku2000 osiąga naturalny target J128 na85 procent? Agreement =100 max(0,1-sqrt(MSE128/mean(target128²))). Cosine/confidence/IoU są osobno, bez warunku sukcesu. Nie normalizujemy wektorów, nie skalujemy ani nie przycinamy targetu.

Adam przywrócony z MODEL_final.pt rodziny2000. Best krok6402, final6402; nowych aktualizacji4402. Stop: explicit_human_stop_before85. Czas trenera wraz z eksportem 510.182 s; weryfikacja 1.143 s; GPU peak 0 bajtów.

Tylko norm2 weight/bias i fc1 weight/bias ostatniego bloku. Loss bez zmian: MSE128/energia128 +10 MSE640/energia640 +0.02 GT CenterNet/GIoU. Float32, eval BN, bez autocast/TF32, pełny forward. Początkowy LR3e-5; każde zdarzenie zmiany zapisano w LR_EVENTS.

{"seed": 20261005, "steps": 50000, "start_step": 2000, "lr": 3e-05, "min_lr": 3e-05, "lr_schedule": "constant_continuation", "retention": 10.0, "detection_weight": 0.02, "pause_seconds": 0.03, "agreement_threshold": 85.0, "iou_threshold": 0.9, "confidence_threshold": 0.9, "plateau_window": 2000, "plateau_gain_pp": 0.1, "precision": "float32", "autocast": false, "tf32": false, "chunk_steps": 1000, "stagnation_window": 4000, "stagnation_gain_pp": 0.02, "minimum_plateau_lr": 1e-05}

[]

## Podłoga i audyt

Przed pierwszym update odtworzono naturalny latent2000: max błąd 0; Adam cztery step2000. Energia128 86.5008011, energia640 0.154263243. Ten sam RGB480x320 INTER_AREA i ImageNet wewnątrz modelu.

Nowy proces utworzył modele before/best/final i wykonał plain forward bez hooków. Oddzielny hook jest wyłącznie odczytujący; jego wpływ na detektor wynosi bitowo zero. Oryginał0 również świeżo sprawdzono. Audyt każdego state_dict potwierdza identyczne zamrożone parametry i BN, tylko cztery dozwolone zmiany. Sprawdzono optimizer steps, źródłowe SHA oraz nienaruszony checkpoint2000.

## Zakres wyniku

Jeden uprzednio korzystnie wybrany obraz i GT. Target pochodzi z optymalizacji GT. Trening widzi target/GT, inference tylko piksele i wagi. To overfit, bez dowodu generalizacji lub poprawy na nowych obrazach. Żadnych lookup, biasfield, nowych warstw, zmian architektury, downstream lub trenowalnych parametrów. Nieosiągnięcie progu w budżecie/plateau nie dowodzi niemożliwości.

## Artefakty

MODEL_before/best/final.pt z Adamem i metadanymi; best_live/latest oraz chunk checkpoints; naturalne latenty600x768 i pełne CSV J128; HISTORY.jsonl/json/csv; audyty, LR_EVENTS, FRESH_RELOAD i overlay. EXECUTED_SOURCE: kod z chwili startu. VERIFICATION_SOURCE: końcowa weryfikacja. Deklaracja rodziny lokalna, bez zapisu poza dwoma dozwolonymi katalogami.
