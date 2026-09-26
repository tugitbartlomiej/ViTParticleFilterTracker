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

- Efekt lokalny (polecenie „policz efekt lokalny na klatkach val podobnych”, sekcja 7d, `efekt_lokalny.py`):
  wagi `FINAL.pth` pobrane z Edena (1,1 GB, poza gitem), DINOv2 równoległej sesji policzył podobieństwo 12 043
  klatek val do kotwic, RF-DETR lokalnie przewidział ramki dla 10 zestawów wag, AP COCO na podzbiorach.
  Wynik: na klatkach najbardziej podobnych do kotwic dobrane klatki −0,003 AP@0,5, losowe +0,005, oba w szumie
  (SD 0,007-0,009 na 134 klatkach). Okolica kotwic jest trudna: AP 0,55 na top-134, 0,38 wokół TR_0134.
- Trzy ziarna rozdzielają różnice rzędu 0,002-0,003 globalnie i 0,01 na małych podzbiorach.
- Jedna epoka i 8,9 procent zbioru; większe porcje i więcej epok nietestowane.

- Kampania 5 epok (7f, zadanie 1814835, 4 ramiona x 3 ziarna, polecenie „więcej epok”, dodatkowe ramię „trudne”):
  globalnie dobrane −0,0005 mAP@0,5 względem braku dodania, losowe +0,001, trudne +0,0016, wszystko w szumie.
  Okolica kotwic: więcej epok podnosi AP top-134 z 0,53 do 0,59–0,60 we wszystkich ramionach; dobrane wobec
  losowych +0,006, −0,005, −0,013, trudne −0,018 w każdym ziarnie (pomagają za to daleko od kotwic, +0,009).
  Kryterium ustalone przed wynikiem niespełnione. Wagi epok, predykcje i zapisy latentu sond pobrane (poza gitem).
- Literatura (Undermind, Scite, `NOWOSC_I_PUBLIKACJA.md`): generate-then-retrieve istnieje od 2015/2018 (ASAL,
  WACV 2020), wyszukiwanie cechami DINO od DCD (WACV 2024); nowy jest tylko cel z prognozy Koopmana. Porażka
  ASAL na CelebA-Eyeglasses ma ten sam mechanizm co u nas (cała scena zamiast małego szczegółu).
- Teoria (`TEORIA_LANGEVIN.md`): model Langevina z liniowym dryfem i twierdzenie o stabilności wyboru top-k
  (Cauchy-Schwarz); przy małym przesunięciu prognozy wybór klatek jest ten sam co dla kotwicy (54/60 wspólnych).
- Polecenie „Koopman ma dobrze przewidywać latent, dostosuj moje równanie Langevina”: nowy eksperyment
  `experiments/langevin_latent_detektora` (dryf Koopmana, pamięć, wymuszenie, szum kolorowy; sterowanie w etapie 2).
  Od kroku 245 rollout 100 aktualizacji 0,151 wobec 0,171 dla L4 (pamięć 8 przyrostów), ale 1 krok gorszy o
  2,2 procent, a od kroku 485 gorszy od L4; bramka niespełniona. Ruch niesie pamięć; po kroku 485 zostaje szum.

- Pętla uczenia 8 obrotów (2026-09-26/27, `experiments/petla_uczenia_koopman_fastgan`, polecenie „wykonuj i potem
  analizuj”, cel: jak najszybciej uczyć sieć): obrót = 165 aktualizacji; pętle koopman (prognoza latentu Koopmanem
  z pamięcią → FastGAN → DINOv2 → +134 klatki), losowe, bez; ziarno 42. Po 8 obrotach mAP@0,5: bez 0,9072, koopman
  0,9055, losowe 0,9045; przy kotwicach bez 0,627, koopman 0,593, losowe 0,563. Kryterium formalnie spełnione
  (koopman − losowe +0,030 lokalnie, −0,0017 globalnie wobec bez), ale dodawanie klatek nie przyspiesza uczenia:
  najlepsza pętla bez dodawania. Prognoza Koopmana trafna w obrocie 1 (0,127 wobec 0,111), potem dryf gaśnie.
  Diagnoza użytkownika: pętla szuka klatek, do których uczenie i tak dojdzie. Propozycja pętli przyspieszonej
  (cel 5 obrotów naprzód, kotwice niedouczone, najlepiej od kroku 0) czeka na zgodę.

## Eden (do sprzątnięcia za zgodą)

- `~/smallvit_runs/2026-09-25_kontynuacja_dobor150/results/seed{42,43,44}_{ramię}/trening/FINAL.pth` (9 x 120 MB).
- `~/smallvit_runs/2026-09-25_kontynuacja_dobor/` - katalog przerwanego pilotu (4 komplety wag).
- Zadanie 1810406 (`koopman_fastgan_v1`, held) należy do równoległej sesji; nie ruszane.
- Węzły czyste: scratch `/dev/shm/kont245b-*` usuwany przez trap, logi potwierdzają zakończenie wszystkich ramion.
- `~/smallvit_runs/2026-09-25_kontynuacja_5epok/` (kampania 5 epok, około 9 GB): wagi pobrane lokalnie; do sprzątnięcia za zgodą.
- `~/smallvit_runs/2026-09-26_petla_uczenia/` (pętla 8 obrotów, około 6 GB): wagi i latent pobrane lokalnie; do sprzątnięcia za zgodą.

## Commity (repo ViTParticleFilterTracker)

`654ae12` (K=160 + pokaz), `9f17339` (inwersja + Gauss-Newton), `6360929` (17 klatek + projekt kontynuacji),
`8474b31` (kampania 134 klatek, sekcja 7c, ta notatka), `84747b2` (7d), `9f45551` (7e), następny: 7f,
teoria, nowość, równanie Langevina.
