# Predykcje zapisane przed pomiarem (2026-10-26, 13:30, Europe/Warsaw)

Zapisane przed uruchomieniem jakiegokolwiek zadania na Edenie. Progi nie będą zmieniane po pomiarze; ewentualne uwagi po fakcie
będą oznaczone jako takie. Wszystkie miary są zdefiniowane w `Kod_eksperymentu/pomiar_mosty_eden.py` (docstring), ocena
w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`.

Skąd biorą się predykcje: pomiary na jednym biegu (seed 47, historia `90f58d4f8b263d8b`, 806 checkpointów, jedna klatka TRAIN)
w eksperymentach `schur_qr_jakobiany_mlp_20261024__10-03-40` i `kontrakcja_komutator_polkrokow_20261025__02-12-46`.
Liczby tam zmierzone (nienormalność 0,93, kontrakcja uwagi, +0,54 vs +0,12 separacji, defekt komutatora w blokach 4–5, korelacja
separacji z IoU 0,76) są tu hipotezami, nie wynikami.

Konwencje: bloki numerowane 1–6 (indeksy 0–5 w kodzie); „k65640” = ostatni checkpoint (60 epok), „k0” = wagi losowe.
Mediany po tokenach danej klasy i po klatkach, chyba że napisano inaczej. Test znaku po seedach: predykcja PASS, gdy warunek
zachodzi w co najmniej podanej liczbie z 24 seedów (lub z 3 seedów w ablacjach). Dla 24 seedów próg 20/24 odpowiada p ≈ 0,0008
w dwustronnym teście znaku przy p₀ = 0,5; próg 18/24 p ≈ 0,02.

## Część 1: replikacja na 24 seedach (12 checkpointów, 500 klatek TEST, 50 diagnostycznych)

| nr | predykcja | warunek liczbowy (per seed) | próg |
|---|---|---|---|
| P1 | trening instaluje nienormalność pola MLP | mediana ‖N‖/‖J‖ (tokeny narzędzia i tła razem) w bloku 4 przy k65640 > 0,85 **i** przy k0 < 0,80 | 20/24 |
| P2 | nienormalność nie rośnie monotonicznie z głębokością | mediana ‖N‖/‖J‖ bloku 4 ≥ bloku 6 przy k65640 | 18/24 |
| P3 | wokół narzędzia pole bardziej ściska | mediana tr J tokenów narzędzia < mediana tr J tła w bloku 4 przy k65640 | 18/24 |
| P4 | uwaga ściska, MLP 1 rozciąga | przy k65640: stosunek median D_u (s+1)/(s) < 1 dla ≥ 4 z 6 półkroków uwagi **i** > 1 dla półkroku MLP bloku 1 | 20/24 |
| P5 | ruch w uwadze jest styczny do sfery | mediana udziału stycznego („styczny”) w każdym z 6 półkroków uwagi > 0,9 przy k65640 | 20/24 |
| P6 | separację klas robi MLP | Σ Δsep_u po półkrokach MLP > Σ Δsep_u po półkrokach uwagi przy k65640 (mediany sep_u po klatkach) | 20/24 |
| P7a | separacja przewiduje IoU w obrębie seedu | Spearman między sep_u w stanie 12 a IoU ramki szczytu, po parach (checkpoint, klatka), 12 × 500 punktów: ρ > 0,5 | 20/24 |
| P7b | separacja przewiduje IoU między seedami | przy k65640: Spearman między medianą sep_u w stanie 12 a AP@0,5 po 24 seedach: ρ > 0,4 (jeden test, 24 punkty) | 1/1 |
| P8 | trening przenosi ściskanie z uwagi do MLP | iloczyn stosunków D_u po półkrokach uwagi: k65640 > k0 **i** iloczyn po półkrokach MLP: k65640 < k0 | 20/24 |
| P9 | kolejność liczy się w blokach 4–5 | argmax po blokach mediany ‖S−P‖/‖S−X‖ przy k65640 ∈ {4, 5} | 18/24 |
| P10 | zero kontrolne (wagi losowe) | przy k0: ‖N‖/‖J‖ bloku 4 < 0,80 **i** Σ Δsep_u (MLP) < 0,1 **i** \|ρ\| z P7a liczone na samym k0 po klatkach < 0,3 | 20/24 |

Dodatkowe, opisowe (bez progu PASS/FAIL): wszystkie 12 półkroków, wszystkie bloki, wszystkie 12 checkpointów, trend z treningiem,
różnice narzędzie / tło dla każdej miary. Będą podane w tabelach niezależnie od wyniku predykcji.

## Część 2: ablacje (6 wariantów × seedy 3047, 4047, 5047; kontrola = wariant base z tych samych zadań)

| nr | wariant | predykcja | warunek liczbowy | próg |
|---|---|---|---|---|
| A1 | base | zgodność z częścią 1 | P1, P4, P6, P9 zachodzą dla base (ten sam sprzęt i kod co ablacje) | 3/3 |
| A2 | parallel | kolejność uwaga→MLP jest potrzebna | AP@0,5 przy k65640 niższe od base o ≥ 0,02 dla tego samego seedu | 3/3 |
| A3 | frozen_mlp | separację robi uczony MLP | Σ Δsep_u (MLP) < Σ Δsep_u (uwaga) przy k65640 **i** AP@0,5 < 0,40 | 3/3 |
| A4 | frozen_attn | przeniesienie ściskania wymaga uczenia uwagi | \|iloczyn uwagi (k65640) − iloczyn uwagi (k0)\| < 0,05, podczas gdy w base różnica > 0,05 | 3/3 |
| A5 | no_wd | nienormalność pochodzi z zadania, nie z weight decay | ‖N‖/‖J‖ bloku 4 przy k65640 > 0,85 | 3/3 |
| A6 | shuffled_labels | A i B wymagają sygnału | ‖N‖/‖J‖ bloku 4 przy k65640 < 0,85 **i** Σ Δsep_u (MLP) < 0,1 **i** AP@0,5 < 0,10 | 3/3 |

Uwaga do A2: to jedyna predykcja, której kierunek nie wynika wprost z pomiaru na seedzie 47 (defekt komutatora mówi, że kolejność
zmienia wyjście, nie że sieć nie nauczyłaby się wariantu równoległego). Zapisuję ją jako test, nie jako oczekiwanie pewne. Jeżeli
A2 nie zajdzie, wniosek brzmi: defekt komutatora jest cechą wyuczoną przy sekwencyjnym bloku, a nie koniecznością zadania.

## Co uznam za potwierdzenie tez

- Teza A (nienormalne ścinanie wyuczone): P1, P2, P3, P10, A5, A6.
- Teza B (uwaga ściska, MLP rozdziela, trening przenosi ściskanie, separacja przewiduje IoU): P4–P8, P10, A3, A4, A6.
- Teza C (kolejność liczy się tam, gdzie zapada decyzja): P9, A2.
- Teza D (rzut 3D nie klasyfikuje punktów stałych): nie testowana tu ponownie; zmierzona 24.10 i poparta literaturą.

Część tez uznam za potwierdzoną, gdy zajdą wszystkie jej predykcje z części 1; ablacje wzmacniają wniosek przyczynowy, ich
niezajście nie unieważnia części 1, ale zmienia opis mechanizmu (zostanie opisane osobno).
