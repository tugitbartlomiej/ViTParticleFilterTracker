# Sesja: dobór klatek do obrazów przewidzianego latentu i kontynuacja treningu

Katalog eksperymentu: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_tozsamosc_kierunki`.
Pełny opis i liczby: `WYNIKI.md`, sekcje 6b, 6c i 7 (7a wyszukiwanie, 7b pilot 17 klatek, 7c kampania 134 klatek).

## Polecenia użytkownika (w kolejności)

1. Pokazać obrazy FastGAN dla przewidzianego latentu i naprawić pętlę generatora („to napraw tę pętlę i puść jeszcze raz”).
2. Znaleźć klatki, które nie weszły do treningu detektora, dopasowane do obrazów FastGAN; dołożyć je do zbioru
   uczącego; trenować kolejną epokę; sprawdzić, czy detekcja się poprawia czy pogarsza. Etapy zatwierdzone z góry.
3. Po pilocie z 17 klatkami: „dobierz top-50 na kotwicę i puść jeszcze raz”, „tamte treningi anuluj”.
4. „licz na dgxach!” - oczekujące ziarna przeniesione na partycję short (dgx-1).

## Co zrobiono

- Naprawiona pętla FastGAN (`pokaz_fastgan_inwersja.py`, `pokaz_fastgan_gn.py`): najpierw inwersja kotwicy
  (błąd 0,09-0,27 rozrzutu), potem deterministyczny krok Gaussa-Newtona w stronę celu Koopmana (cos 0,84-0,97,
  59-87 procent długości). Obrazy przewidzianego i obecnego latentu różnią się o 0,001-0,009 L1, gołym okiem nic.
- Kotwice pokazu TR_0082 i TR_0134 były w 1500 klatkach pretrainingu biegu 1810662; TR_0006 nie. Nowe klatki
  szukane wyłącznie poza tymi 1500 i poza filmami kotwic (indeks DINOv2 równoległej sesji, tylko odczyt).
- Pilot: top-6 na kotwicę = 17 klatek z 14 filmów; zadanie 1814538 (3 ziarna x 3 ramiona) przerwane na polecenie
  użytkownika po 4 ramionach. Ziarno 42: 0,8667 / 0,8667 / 0,8663 (bez dodania / dobrane / losowe).
- Kampania: top-50 na kotwicę = 134 klatki z 84 filmów (cos 0,66-0,81), losowa kontrola 134 klatek; zadanie
  1814611 (`eden/kontynuacja150.slurm`, `eden/kontynuacja.py`), jedna epoka = 103 aktualizacje od
  `model_0000245.pth`, świeży optymalizator, `trening.py` równoległej sesji bez zmian, walidacja na całym val.
- Podsumowanie: `podsumuj_kontynuacje.py` -> `wyniki/dobor_klatek/PODSUMOWANIE_KONTYNUACJA.json`, `KONTYNUACJA.png`.

## Wynik (mAP@0,5, val 12 043 klatki, średnia po ziarnach 42/43/44)

- bez dodania (1500): 0,8686 +- 0,0008
- dobrane Koopman+FastGAN (1634): 0,8694 +- 0,0002
- losowe (1634): 0,8696 +- 0,0011

Różnice sparowane względem braku dodania: dobrane +0,0007, losowe +0,0009; mAP@0,5:0,95 obie +0,0003;
F1 dobrane +0,0016, losowe +0,0008. Wszystko w rozrzucie ziaren.

Werdykt: dołożenie 134 klatek nie szkodzi i daje około +0,001, niezależnie od sposobu doboru. Dobór przez
Koopman -> FastGAN -> wyszukiwarkę nie jest lepszy od losowego. Powód znany z 7a: obrazy przewidzianego i obecnego
latentu są praktycznie identyczne, więc wyszukiwarka dostała „podobieństwo do trzech kotwic”, nie kierunek zmiany.

## Granice

- Miara globalna; efekt lokalny (klatki val podobne do kotwic) niezmierzony. Możliwy następny krok: policzyć
  mAP zachowanych wag `FINAL.pth` na podzbiorze val najbliższym kotwicom. Nieuruchomiony.
- Trzy ziarna rozdzielają różnice rzędu 0,002-0,003, nie 0,001.
- Jedna epoka i 8,9 procent zbioru; większe porcje i więcej epok nietestowane.

## Eden (do sprzątnięcia za zgodą)

- `~/smallvit_runs/2026-09-25_kontynuacja_dobor150/results/seed{42,43,44}_{ramię}/trening/FINAL.pth` (9 x 120 MB).
- `~/smallvit_runs/2026-09-25_kontynuacja_dobor/` - katalog przerwanego pilotu (4 komplety wag).
- Zadanie 1810406 (`koopman_fastgan_v1`, held) należy do równoległej sesji; nie ruszane.
- Węzły czyste: scratch `/dev/shm/kont245b-*` usuwany przez trap, logi potwierdzają zakończenie wszystkich ramion.

## Commity (repo ViTParticleFilterTracker)

`654ae12` (K=160 + pokaz), `9f17339` (inwersja + Gauss-Newton), `6360929` (17 klatek + projekt kontynuacji),
następny: kampania 134 klatek, sekcja 7c, ta notatka.
