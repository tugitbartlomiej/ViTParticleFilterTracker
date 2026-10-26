# Predykcje zapisane przed pomiarem (2026-10-26, 19:10, Europe/Warsaw)

Zapisane przed uruchomieniem jakiegokolwiek zadania tego eksperymentu na Edenie. Progi nie będą zmieniane po pomiarze; uwagi po fakcie
będą oznaczone jako takie. Źródło predykcji: wyniki rodzica `mosty_fizyki_24seedy_ablacje_20261026__13-04-31` (24 seedy + 18 ablacji,
`Wyniki/Raport/RAPORT.md`). Miary jak w rodzicu (`Kod_eksperymentu/pomiar_mosty_eden.py`), dla CIFAR w `pomiar_mosty_cifar.py`;
ocena w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`.

Konwencje: bloki numerowane 1–NB; „koniec” = ostatni checkpoint (65 640 kroków w części K, 93 780 w części C), „k0” = wagi losowe.
Mediany po tokenach i obrazach. Test znaku po biegach: PASS, gdy warunek zachodzi w co najmniej podanej liczbie biegów.

## Część K: kontrola „oba zamrożone” (wariant frozen_both, seedy 3047, 4047, 5047)

Uczą się tylko łodyga (4 konwolucje), norma końcowa i głowa; uwaga, LN₁, MLP i LN₂ każdego bloku zostają na inicjalizacji.
Porównania z biegami base, frozen_mlp, frozen_attn **tego samego seedu** z rodzica (ten sam kod, sprzęt A100 i siatka checkpointów).
Pytanie: ile detekcji daje sama łodyga z losowym, stałym mieszaniem przez resztkę, czyli jaki jest punkt odniesienia dla kompensacji.

| nr | predykcja | warunek liczbowy (per seed, koniec) | próg |
|---|---|---|---|
| K1 | bez uczonych bloków detekcja wyraźnie spada | AP@0,5 < 0,80 **i** AP ≤ AP(frozen_mlp) − 0,03 **i** AP ≤ AP(frozen_attn) − 0,03 | 3/3 |
| K2 | zamrożone bloki nie budują separacji | Σ Δsep_u po wszystkich 12 półkrokach < 0,5 (base: 1,0–1,5) | 3/3 |
| K3 | separację przejmuje łodyga | sep_u stanu 0 > 1,10 (base 0,93–0,94, frozen_mlp 1,02–1,03) | 3/3 |
| K4 | uwaga o zamrożonych wagach staje się prawie identycznością (jak w frozen_attn) | iloczyn stosunków D_u po półkrokach uwagi > 0,85 (k0: 0,62–0,66) | 3/3 |
| K5 | nienormalność pola MLP zostaje losowa | ‖N‖/‖J‖ bloku 4 < 0,80 | 3/3 |

Uwaga do K1: jeżeli AP ≥ 0,80, wniosek brzmi, że detekcję cystotomu robi w większości łodyga i głowa, a bloki (uczone czy nie) dodają
niewiele; wtedy „kompensacja” z rodzica jest w dużej mierze złudzeniem pomiaru na blokach.

## Część C: drugi zbiór i druga architektura (CIFAR-10, biegi z `2026-10-13_deit_cifar10`)

Biegi: D1 = `s47_obciecie` (DeiT-Tiny, 12 bloków, CLS), D2 = `s47_bez_obciecia` (DeiT bez obcięcia gradientu), M1 = `maly_s47_obciecie`
(MałyViT 6 bloków na 128 px), M2 = `maly_s48_obciecie`, M3 = `maly_s47_wd0` (bez weight decay), M0 = `maly_s47_losowe_etykiety`
(kontrola zerowa). „Uczone” = D1, D2, M1, M2, M3 (5 biegów). Blok środkowy b* = NB // 2 + 1 (DeiT: 7, MałyViT: 4). Tokeny do A: 16 losowych
tokenów obrazu na obraz (CLS osobno, nie wchodzi do median). Separacja klas sep = `sep_u_mean` (średnia tokenów obrazu na sferze).

| nr | predykcja | warunek liczbowy | próg |
|---|---|---|---|
| C1 | trening instaluje nienormalność MLP niezależnie od zbioru i architektury | ‖N‖/‖J‖ bloku b* na końcu > 0,85 **i** przy k0 < 0,80 | 5/5 uczonych |
| C1z | także przy losowych etykietach (jak A6 rodzica) | M0: ‖N‖/‖J‖ bloku b* na końcu > 0,85 | 1/1 |
| C2 | sygnał decyduje o skali pola, nie o kształcie | ‖J‖ bloku b*: koniec > k0 w biegach uczonych; M0: koniec < 0,5 · k0 | 5/5 i 1/1 |
| C3 | nienormalność nie rośnie monotonicznie z głębokością | ‖N‖/‖J‖ ostatniego bloku < maks. po blokach 2 … NB−1 (koniec) | 5/5 |
| C4a | uwaga ściska na starcie | iloczyn stosunków D_u po półkrokach uwagi przy k0 < 0,9 | 6/6 |
| C4b | trening przenosi ściskanie z uwagi do MLP | iloczyn uwagi: koniec > k0 **i** iloczyn MLP: koniec < k0 | 5/5 |
| C5 | ruch w uwadze jest styczny do sfery | mediana udziału stycznego > 0,9 w każdym półkroku uwagi poza ostatnim blokiem (koniec) | 5/5 |
| C6 | separację klas buduje MLP | Σ Δsep po półkrokach MLP > Σ Δsep po półkrokach uwagi (koniec) | 3/3 MałyViT; 2/2 DeiT osobno |
| C7 | zero kontrolne | przy k0: Σ Δsep po wszystkich półkrokach < 0,2 **i** acc < 0,15 | 6/6 |
| C8 | losowe etykiety nie budują separacji | M0: acc < 0,15 **i** Σ Δsep (MLP) < ½ · Σ Δsep (MLP) w M1 | 1/1 |
| C9a | defekt komutatora nie siedzi na brzegach | argmax po blokach mediany ‖S−P‖/‖S−X‖ ∉ {1, NB} (koniec) | 5/5 |
| C9b | kolejność uwaga → MLP nie decyduje o wyniku | acc po zamianie bloku b na równoległy (P) ≥ acc − 0,05 dla każdego b ≥ 2 (koniec) | 5/5 |

Uwaga do C6 (DeiT): odczyt jest z tokenu CLS, więc informacja o klasie musi przejść przez uwagę do CLS; separacja średniej tokenów
obrazu może rosnąć głównie po uwadze. C6 dla DeiT zapisuję jako test, nie jako oczekiwanie pewne; jeżeli nie zajdzie, wniosek brzmi,
że podział pracy „MLP rozdziela” zależy od sposobu odczytu (średnia tokenów vs CLS), a nie od architektury bloku.

Uwaga do C2: w rodzicu ‖J‖ bloku 4 spadało w pierwszych 1000 krokach (3,2 → 1,5) i rosło do 4,2 na końcu; przy losowych etykietach
spadło do 0,5. Predykcja przenosi to na CIFAR bez zmian.

## Co uznam za potwierdzenie

- Teza A (nienormalne ścinanie jako kształt nadawany przez trening, skala z sygnału): C1, C1z, C2, C3, K5.
- Teza B (uwaga ściska i jest styczna, MLP rozdziela, trening przenosi ściskanie): C4a, C4b, C5, C6, C7, C8, K2–K4.
- Kompensacja (bloki są wymienne, łodyga nie wystarcza): K1; jeżeli K1 nie zajdzie, kompensacja z rodzica wymaga przepisania (patrz uwaga).
- Teza C (kolejność) pozostaje odrzucona; C9a i C9b sprawdzają tylko, czy jej odrzucenie przenosi się na CIFAR.
