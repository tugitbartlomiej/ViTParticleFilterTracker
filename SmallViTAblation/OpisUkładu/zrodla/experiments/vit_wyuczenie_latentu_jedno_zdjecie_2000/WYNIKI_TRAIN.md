# Kontynuacja do 2000 aktualizacji

Łączne kryterium: NIEOSIĄGNIĘTE.

| Pomiar | Oryginał0 | Wyuczone1000 | Wyuczone2000 | Best kontynuacji |
|---|---:|---:|---:|---:|
| MSE128 | 85.9211807 | 10.9162397 | 8.67640686 | 8.67640686 |
| MSE640 | 0 | 4.18178752e-06 | 3.36482549e-06 | 3.36482549e-06 |
| MSE768 | 14.3201971 | 1.81937671 | 1.44607067 | 1.44607067 |
| agreement128 | 0.335600708 | 64.4756232 | 68.3291376 | 68.3291376 |
| cosine128 | 0.106738292 | 0.943290353 | 0.951030314 | 0.951030314 |
| confidence | 0.867869496 | 0.771777928 | 0.759447038 | 0.759447038 |
| iou | 0.856594683 | 0.999952275 | 0.999883696 | 0.999883696 |
| GTloss | 0.490047604 | 0.154576004 | 0.151910305 | 0.151910305 |

## Pytanie i aparat

Czy dalsze1000 aktualizacji tych samych wag i tego samego celu poprawi naturalne dopasowanie GELU? Adam przywrócony z MODEL_T1_norm2_fc1_final.pt rodziny vit_wyuczenie_latentu_jedno_zdjecie; wszystkie cztery stany step1000 na wejściu, step2000 na końcu. LR stały3e-5, bez warmup i bez nowego cosine. Żadnego restartu od oryginału.

Uczone tylko norm2 affine i fc1 weight/bias ostatniego bloku. Loss = MSE128/energia target128 + 10 MSE640/energia target640 + 0.02 GT CenterNet/GIoU. Float32, eval BN, bez autocast/TF32, frozen downstream.

{"seed": 20261005, "steps": 1000, "start_step": 1000, "lr": 3e-05, "min_lr": 3e-05, "lr_schedule": "constant_continuation", "retention": 10.0, "detection_weight": 0.02, "pause_seconds": 0.03, "agreement_threshold": 90.0, "iou_threshold": 0.9, "confidence_threshold": 0.9, "plateau_window": 250, "plateau_gain_pp": 0.2, "precision": "float32", "autocast": false, "tf32": false}

Czas kontynuacji z przygotowaniem i eksportem: 130.139 s. Weryfikacja: 1.107 s. Szczytowa alokacja GPU: 54858240 bajtów.

1000 nowych aktualizacji; łącznie 2000; best krok 2000; stop budget2000; 148608 uczonych parametrów; pętla 126.304 s.

## Podłoga i audyt

Przed pierwszą aktualizacją odtworzono naturalny checkpoint1000 i jego latent; max błąd 0. Ten sam RGB480x320 INTER_AREA i normalizacja ImageNet wewnątrz modelu. Stała energia target128 86.5008011, target640 0.154263243.

Świeży proces utworzył modele i wykonał plain forward bez hooków. Oddzielny hook wyłącznie odczytujący nie zmienił mapy ani o jeden bit. Oryginał0 również zweryfikowano ponownie. Pełne audyty porównują każdy tensor state_dict do oryginału0: zmieniły się tylko dozwolone cztery parametry, bufory BN identyczne. Checkpoint wznowienia, pierwotny checkpoint, NPZ i obrazy zachowały SHA256. Wszystkie optimizer steps i zmienione parametry zapisano i sprawdzono.

Writer JSON ma unikatowe PID/czas, ograniczone retry OSError i niekrytyczny progress. Best jest zapisywany przed progress. Nie było zatrzymującego błędu w kontynuacji. T2 i earlystop nie były stosowane.

## Zastrzeżenia

Jeden uprzednio korzystnie wybrany obraz i jego GT. Target również pochodzi z optymalizacji GT. Trening widzi target i GT, inference tylko piksele i wyuczone wagi. To dopasowanie wewnątrz próby; nie dowodzi generalizacji ani poprawy na nowych zdjęciach. Nie zmieniono architektury, parametrówki, objective ani targetu. Bez lookup, biasfield, aktywacji zastępujących forward, rescale i clipping celu. Nieosiągnięcie90 procent jest wynikiem w zadanym budżecie, nie dowodem niemożliwości.

## Artefakty

MODEL_before/best/final.pt (before=krok1000) z Adamem, architekturą/wersjami i changed_parameters; LATENT_NATURAL_*.npz600x768; VEKTORY_J128_*.csv600x128; HISTORY; AUDIT i FRESH_RELOAD; OVERLAY i POROWNANIE_0_1000_2000.png. RUN_CONFIG zawiera konfigurację, PID, hashes, optimizer i źródło. EXECUTED_SOURCE jest migawką z chwili startu, VERIFICATION_SOURCE końcową weryfikacją. Deklaracja rodziny jest lokalna w granicy wyłącznej własności.
