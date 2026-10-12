# Pętla uczenia: prognoza latentu → FastGAN → wyszukanie → uczenie, 8 obrotów (pilot, ziarno 42)

Data: 2026-09-26/27. Plan i kryterium: `PLAN.md` (zatwierdzony „No to wykonuj i potem analizuj”). Kod:
`obrot_lokalny.py` (prognoza, FastGAN, wyszukanie), `dyrygent.py` (8 obrotów, zadania Edena 1816922 …),
`eden/obrot.slurm`, `eden/kontynuacja_obrot.py`, `analiza.py`. Liczby: `wyniki/ANALIZA.json`, rysunek
`wyniki/ANALIZA.png`, stan pętli `wyniki/STAN.json`, raporty obrotów `wyniki/obroty/kXX/RAPORT.json`.

Każdy obrót: 165 aktualizacji od wag końca poprzedniego obrotu (start: krok 245 biegu 1810662), świeży optymalizator,
walidacja całego val na Edenie. Pętle: **koopman** (+134 klatki na obrót z wyszukiwarki DINOv2 dla obrazów FastGAN
latentu przewidzianego na koniec obrotu), **losowe** (+134 losowe nieużyte), **bez** (1500 klatek bazy).

## Wynik

mAP@0,5 całego val po obrotach (Eden):

| obrót | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | pole pod krzywą |
|---|---|---|---|---|---|---|---|---|---|
| koopman | 0,8787 | 0,8902 | 0,8962 | **0,9029** | 0,9022 | 0,9048 | 0,9047 | 0,9055 | 0,8981 |
| losowe | 0,8803 | 0,8902 | 0,8953 | 0,8999 | 0,9020 | 0,9045 | 0,9017 | 0,9045 | 0,8973 |
| bez | 0,8811 | 0,8909 | 0,8961 | 0,9003 | 0,9032 | 0,9052 | **0,9073** | **0,9072** | **0,8989** |

Tempo (pierwszy obrót z mAP@0,5 ≥ próg): 0,89 → wszystkie w obrocie 2; 0,895 → wszystkie w 3; 0,90 → koopman
i bez w 4, losowe w 5.

AP@0,5 na 134 klatkach val najbliższych kotwicom (lokalnie; start 0,526):

| obrót | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| koopman | 0,553 | 0,554 | 0,570 | 0,583 | 0,599 | 0,591 | 0,594 | 0,593 |
| losowe | 0,557 | 0,576 | 0,578 | 0,584 | 0,600 | 0,582 | 0,579 | 0,563 |
| bez | 0,576 | 0,587 | 0,601 | 0,604 | 0,609 | 0,598 | 0,615 | **0,627** |

Po 8 obrotach mAP@0,5:0,95: koopman 0,6756, losowe 0,6693, bez 0,6665; F1: 0,9158, 0,9123, 0,9146.

## Kryterium (ustalone przed wynikiem)

Koopman − losowe na top-134 po 8 obrotach: **+0,030** (> 0); koopman − bez na całym val: **−0,0017** (≥ −0,002).
Formalnie spełnione na jednym ziarnie. Ale odczyt jest inny niż „Koopman przyspiesza”:

1. **Najlepsza jest pętla bez dodawania** globalnie (0,907) i przy kotwicach (0,627). Dodawanie klatek, jakichkolwiek,
   nie przyspieszyło uczenia; pole pod krzywą bez > koopman > losowe.
2. Przewaga Koopmana nad losowymi przy kotwicach to „mniej szkodzi”, nie „pomaga”: losowe spadają przy kotwicach
   z 0,600 do 0,563 w obrotach 5–8, koopman trzyma około 0,59, bez rośnie do 0,627.
3. Wyjątek: dokładność położenia ramki (mAP@0,5:0,95) jest najwyższa w pętli koopman (+0,009 wobec bez).
4. Prawdopodobny mechanizm szkody: stała długość obrotu (2640 próbek) przy rosnącym zbiorze (do 2572 klatek)
   rozcieńcza powtórzenia 1500 klatek bazy (z 1,76 do 1,03 na obrót). Pętla bez widzi bazę częściej.
   Test: pętla z tą samą liczbą powtórzeń bazy (nie sprawdzone).
5. Jedno ziarno: rozrzut między ziarnami w kampanii 5 epok to około 0,002–0,003 globalnie i 0,008 lokalnie na
   ziarno; różnice przy kotwicach 0,03–0,06 są od niego większe, globalne nie.

## Prognoza Koopmana i teoria

- Prognoza zmiany skalara tożsamości (w rozrzutach między obrazami): obrót 1 przewidziana 0,127, zmierzona 0,111;
  obrót 2: 0,062 wobec 0,025; od obrotu 3 zmiana zmierzona ≈ 0 (0,007; 0,002; 0,002; 0,000; −0,002), a prognozy
  drobnych ruchów są gorsze od braku zmian. Dryf tożsamości kończy się w pierwszych 2 obrotach (około kroku 550),
  zgodnie z sekcją 5 `koopman_tozsamosc_kierunki` (koniec dryfu około kroku 500).
- Obraz celu różni się od obrazu obecnego o 0,0001–0,008 jasności (L1), coraz mniej z obrotami.
- Wspólne klatki top-45 (obraz celu vs obraz obecny): 30–45, średnio 40 z 45 (89 procent) — zgodnie z twierdzeniem
  o stabilności wyboru (`../koopman_tozsamosc_kierunki/TEORIA_LANGEVIN.md`): po wygaśnięciu dryfu pętla Koopmana
  wybiera w praktyce sąsiadów kotwicy.
- Diagnoza użytkownika (23:20): pętla szuka klatek zgodnych z latentem, do którego uczenie i tak dojdzie, więc nie
  wnosi nic nowego. Wynik jest z nią zgodny. Propozycja pętli „przyspieszonej” (cel 5 obrotów naprzód, kotwice
  najbardziej niedouczone) w `PLAN.md`, czeka na zatwierdzenie.
