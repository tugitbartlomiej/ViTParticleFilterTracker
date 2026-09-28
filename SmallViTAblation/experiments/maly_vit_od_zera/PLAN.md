# Mały Vision Transformer uczony od zera: latent, który się rusza

Data: 2026-09-28. Status: **uruchomione lokalnie (RTX 3070) 2026-09-28 15:43** po zgodzie użytkownika:
„możesz to zapuścić! Także puść teraz na mojej karcie graficznej i rób monitoring tego wszystkiego”.
Zrealizowana wersja pętli: sekcja „Wykonanie” na końcu.

Polecenie użytkownika: skoro w dostrajanym RF-DETR kręgosłup stoi, wziąć Vision Transformera bez
pre-treningu, uczyć od zera, tak żeby enkoder się ruszał i Koopman z Langevinem miały co opisywać;
zaproponować ViT, który uczy się szybko (RF-DETR, coś mniejszego, np. z OpenCV).

## Skąd ten pomysł (pomiary)

- Dostrajany RF-DETR: wagi DINOv2 zmieniają się o 0,026 % w 6 epokach (`../petla_od_poczatku_large/DIAGNOZA_PORAZKI.md`).
- Stan dekodera Nano: wzrost od 0 do ~0,40 w pierwszych ~400 aktualizacjach, potem postój i drganie;
  żaden model (Koopman, OU, Langevin, Kalman) nie bije „braku zmiany” (`../sterowanie_langevin_kalman/PLAN.md`,
  sekcja „Wynik bramek”). Dynamika jest tylko w fazie uczenia, a po pre-treningu ta faza trwa chwilę.
- Sondy (wycinki 256 × 256, warianty tła i obrotu) mierzą co innego niż val (całe kadry operacji zaćmy,
  12 043 klatki z 25 filmów) — `../dekoder_sondy_large/wyniki/SONDY_ARKUSZ.png`, `WAL_PROBKA.png`.

## Co mówi literatura (przegląd 2026-09-28: Undermind, Scite; bez not o wycofaniu)

- **Detekcja od zera na małych zbiorach się załamuje.** He i in. 2019 „Rethinking ImageNet Pre-training”
  (10.1109/ICCV.2019.00502): od zera dogania pre-trening przy 10 tys. obrazów (25,9 vs 26,0 AP, ale 220 tys.
  zamiast 60 tys. iteracji); przy 1 tys. obrazów 3,5 vs 9,9 AP. My mamy 1500 oznaczonych klatek.
- **DETR na ~1–3 tys. obrazów ledwo działa nawet z kręgosłupem z ImageNet**: Wang i in. 2022
  (arXiv:2203.09507) — 11,5 AP na Cityscapes (2975 obrazów, 300 epok), ~3,8 AP na 1 % COCO. DETR potrzebuje
  500 epok (10.1007/978-3-030-58452-8_13), Deformable DETR 50 (arXiv:2010.04159). Winne jest dopasowanie
  węgierskie i rzadki sygnał (jedna ramka = jeden cel na obraz).
- **Mały ViT od zera na małych zbiorach działa, jeśli dostanie „konwolucyjne” wejście:** CCT-7/3x1, 3,76 mln
  parametrów, 96,5 % CIFAR-10 od zera, 90 % w < 30 min na RTX 2080 Ti (Hassani 2021, arXiv:2104.05704);
  SPT+LSA (Lee 2021, arXiv:2112.13492) +3–4 pkt na Tiny-ImageNet; ConViT (10.1088/1742-5468/ac9830) na 5 %
  ImageNet 47,8 vs 34,8 % DeiT. Wejście konwolucyjne zamiast cięcia na łatki daje sieci od początku wiedzę,
  że sąsiednie piksele należą do siebie — tego ViT sam uczy się długo.
- **Samonadzór tylko na własnych danych pomaga najbardziej:** El-Nouby 2021 (arXiv:2112.10740) — pre-trening
  na samym zbiorze docelowym (Stanford Cars, 8144 obrazy): 92,8 vs 35,3 % od zera; Gani 2022 BMVC
  (arXiv:2210.07240) — ViT 2,8 mln, DINO na tym samym zbiorze + dostrojenie: +6 pkt na Tiny-ImageNet.
  Chirurgia: Ramesh 2023 MedIA (10.1016/j.media.2023.102844), wykrywanie narzędzi 60,0 → 74,7 mAP przy 5
  oznaczonych filmach (tam start z ImageNet). MAE dla lekkich ViT (Wang 2023, arXiv:2205.14443) ma słabe górne
  warstwy na małych zbiorach — potrzebny człon destylacji albo kontrastu.
- **Fazy uczenia są na początku:** Redman 2024 NeurIPS (10.52202/079017-0744) — widmo Koopmana z pierwszych
  100 iteracji różni się od późniejszego, stabilizuje się po ~700; Achille (arXiv:1711.08856) — okres krytyczny
  w pierwszych 40–60 epokach od zera.
- **Dobór danych: brak dowodu, że od zera pomaga bardziej niż przy dostrajaniu.** RHO-LOSS (arXiv:2206.07137)
  przyspiesza w obu reżimach (od zera CIFAR-10: 90 % w 47 zamiast 177 epok); Sorscher 2022
  (arXiv:2206.14486) — przy małej puli lepiej zostawiać łatwe przykłady; Gupta 2023 (arXiv:2303.02095, tylko
  streszczenie) — dla ViT od zera dobór podzbioru prawie nic nie daje. To trzeba zmierzyć, nie założyć.
- **OpenCV:** `cv2.TrackerVit` (jest w lokalnym OpenCV 4.11) to tracker jednego obiektu (VitTrack, plik ONNX
  z `opencv_zoo`), nie detektor: wymaga ramki startowej, OpenCV go nie uczy. Kod uczenia istnieje osobno
  (`github.com/lpylpy0514/VitTracker`, konfiguracja od zera: wymiar 48, 3 bloki, wejście z 4 konwolucji,
  300 epok na LaSOT/GOT-10k/COCO/TrackingNet). Liczba parametrów ~0,18 mln to oszacowanie z rozmiaru pliku,
  niepotwierdzone. Bierzemy z niego tylko pomysł wejścia konwolucyjnego.

## Propozycja modelu

**Nie RF-DETR od zera** (DINOv2 + DETR na 1500 klatkach: poniżej progu z pracy He, dopasowanie węgierskie
dokłada własny szum do trajektorii latentu). **Nie VitTrack z OpenCV** (inne zadanie, uczenie poza OpenCV,
wymaga dużych zbiorów śledzenia).

**Proponuję: mały hybrydowy ViT z gęstą głową, ~4–6 mln parametrów, napisany u nas (timm 0.9.16 jest w
środowisku `F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1`):**
- wejście: cała klatka zmniejszona do 480 × 320 (proporcje 3:2 zachowane; mediana ramki narzędzia 196 × 81 px
  w 720 × 480 → ~131 × 54 px, czyli ~8 × 3 tokeny);
- „łodyga” z 4 konwolucji 3 × 3 z krokiem 2 (jak CCT, VitTrack, Hong 2024 IJCV 10.1007/s11263-024-01988-x)
  → siatka 30 × 20 = 600 tokenów;
- 6–7 bloków transformera, wymiar 192, 3 głowy uwagi (skala ViT-Tiny/CCT-7);
- **gęsta głowa zamiast DETR**: każdy token przewiduje, czy jest środkiem narzędzia (mapa ciepła) oraz
  wymiary ramki (styl CenterNet/FCOS). Każda klatka daje 600 celów zamiast jednego, bez dopasowania węgierskiego
  — to jest źródło szybkiego uczenia;
- ocena: mAP@0,5 na val (ta sama, co RF-DETR; na co dzień próbka 2000 klatek, pełna na końcu).

**Dwa sposoby uczenia (dwa ramiona startowe):**
- A: nadzorowane od zera na 1500 klatkach;
- B: najpierw samonadzór MAE (sieć zgaduje zakryte kawałki obrazu) na nieoznaczonych klatkach z filmów
  treningowych (lokalnie ~10,6 tys. z puli, bez klatek val/test — żeby nie było wycieku), z członem
  kontrastowym dla górnych warstw, potem dostrojenie jak A.

## Co rejestrujemy (dla Koopmana i Langevina)

Co 10 aktualizacji: tokeny enkodera na 72 sondach i na stałej próbce 200 klatek val (średnia po tokenach
narzędzia z ramki i po tle), jakość q per obraz (jak w bramkach), mAP próbki val co epokę, względna zmiana wag
per blok. Wszystko od kroku 0, bo fazy mieszczą się w pierwszych setkach kroków.

## Bramki (każda porażka = stop i raport)

- **Z0 — model w ogóle się uczy (lokalnie, ~1–2 h):** ramię A, 3 ziarna, do 100 epok. Warunek: mAP@0,5 próbki
  val ≥ 0,5. Jeśli nie — najpierw B (samonadzór), bo literatura przewiduje załamanie przy 1500 obrazach.
- **Z1 — latent się rusza i da się go przewidzieć (bez nowego GPU):** na trajektorii z Z0 zmiana wag enkodera
  ≫ 0,026 % na epokę; model Langevina (dryf z hamowaniem, pamięć, wymuszenie, Kalman) bije „brak zmiany”
  ≤ 0,8 × na 1 epokę naprzód w fazie uczenia (ten sam test co G2, dopasowanie na wcześniejszej części).
- **Z2 — latent mówi coś o wyniku (lokalnie, ~3 h):** 3 sposoby doboru klatek (koopman, losowe z wykluczeniem
  sąsiadów ±30, bez dodawania) × 3 ziarna; między ramionami w tej samej epoce latent/q koreluje z mAP val,
  ρ ≥ 0,4, p < 0,05.
- **Z3 — kampania (lokalnie albo Eden, po zgodzie):** 5 ziaren; pole pod krzywą mAP@0,5 w czasie: sterowanie
  > oba punkty odniesienia w ≥ 4 z 5 ziaren.

## Koszt i miejsce

Model ~5 mln parametrów, 600 tokenów, paczka 32: szacuję sekundy na epokę 1500 klatek na RTX 3070 (8 GB);
dokładny czas mierzy Z0. Całość Z0–Z2 mieści się lokalnie, bez Edenu i bez kwoty. Jeden bieg naraz,
jako proces odłączony z logiem.

## Uczciwie

Model od zera będzie słabszy niż dostrojony RF-DETR (0,90 mAP@0,5) — to stanowisko do zbadania, czy dobór
klatek sterowany dynamiką latentu przyspiesza uczenie, nie docelowy detektor. Literatura nie obiecuje, że
dobór danych pomaga bardziej od zera. Po fazie uczenia plateau wróci; zysk może być tylko w szybkości dojścia.

## Wykonanie (2026-09-28)

Doprecyzowanie metody przez użytkownika (przed startem): uczymy od zera na 1500 klatkach; po trzech checkpointach
szukamy w latencie elementów odpowiedzialnych za pełną tożsamość narzędzia — **rdzeń** (odpala się zawsze, gdy
narzędzie jest na obrazie, przy każdym kącie, długości, tle, rozbłysku) i **pełny latent**; Koopman (potem Langevin)
przewiduje, jak te elementy urosną lub zmaleją w obrotach 4, 5, 6…; FastGAN generuje obrazy pasujące do przewidzianego
latentu; do nich szukamy prawdziwych klatek w zbiorze uczącym, dodajemy je i uczymy dalej.

Kod (ten katalog):
- `model.py` — ViT od zera: 4 konwolucje z krokiem 2 (krok 16 px), 6 bloków timm, wymiar 192, 3 głowy,
  pozycje sin-cos 2D, gęsta głowa (mapa środków + l, t, r, b), strata ogniskowa CenterNet + (1 − GIoU);
  **2,93 mln parametrów**. Latent: 768 neuronów MLP ostatniego bloku (po GELU).
- `dane.py` — baza 1500 klatek (720×480 → 480×320), próbka val 3000 z 12 043 (ziarno 0), pula 9136 klatek
  treningowych z indeksu (bez bazy i kotwic), 192 sondy (12 źródeł × 16 wariantów, 176×176, siatka 11×11).
- `trening.py` — paczka 32, AdamW lr 5e-4, wd 0,05, rozgrzewka lr 300 kroków, potem stały lr, bf16; odbicie
  poziome i jasność/kontrast ±20 %; latent sond co 10 kroków; mAP@0,5 (AP jednej klasy, 101 punktów) na próbce val.
- `tozsamosc.py` — neuron „zapala się”, gdy aktywacja na tokenach narzędzia > średnia negatywów + 2 odchylenia
  (negatywy: tło wszystkich sond i obszar narzędzia w sondach `empty_black`); rdzeń: w każdym z 15 wariantów na
  ≥ 75 % źródeł, negatywy ≤ 10 %; pełny latent: negatywy ≤ 10 %, ≥ 30 % sond z narzędziem.
- `prognoza.py` — stan: aktywacje neuronów pełnego latentu (min. 32) na 180 sondach z narzędziem; odszumianie
  średnią z bloków po 5 zapisów (punkt co 50 kroków); PCA do 8 wymiarów; `koopman` (przyrosty, 4 opóźnienia, bez
  hamowania) i `langevin` (hamowanie K y_t + pamięć 4 przyrostów + stała każdej sondy); grzbiet względny 1e-2.
  Cel FastGAN z modelu `langevin`, 3 obroty naprzód; oba modele raportują test wsteczny względem „braku zmiany”.
  Test na danych syntetycznych z nasyceniem: langevin 0,59–0,80 × brak zmiany.
- `generuj.py` — FastGAN 50k (EMA, zamrożony): inwersja kotwicy (4 starty × 160 kroków Adam), potem 250 kroków do
  przewidzianego latentu (target) i do obecnego (current, kontrola), z wiernością obrazowi kotwicy.
- `wybor.py` — wyszukiwanie DINOv2 w indeksie puli (top-90 na kotwicę), klatki na przemian z list 6 kotwic
  (sondy „original” o największej przewidzianej zmianie); reguła różnorodności ±30 klatek tego samego filmu od
  bazy i już dodanych — **ta sama w ramieniu `losowe`**.
- `petla.py` — rozgrzewka 3 obroty × 500 kroków (wspólna), potem 6 obrotów × 500 kroków w trzech ramionach
  (`koopman`, `losowe`, `bez`) z tego samego stanu (wagi i optymalizator); 134 klatki na obrót; ocena co 100 kroków.
- `kampania.cmd` — ziarna 42–46 po kolei; logi `wyniki/petla_s*.log`.

**Z0 (kalibracja, ziarno 42, sama baza, 6000 kroków):** mAP@0,5 próbki val 0,085 po 250 krokach, 0,36 po 1500,
0,39–0,42 po 3500–6000 przy spadającej stracie uczenia (nadmierne dopasowanie do 1500 klatek). Bramka Z0 (≥ 0,5)
**niespełniona**, ale model się uczy i po ~1500 krokach wchodzi w reżim, w którym więcej danych powinno pomagać.
Rdzeń tożsamości (wg definicji wyżej) w Z0 pusty na każdym etapie; pełny latent 33–55 neuronów; najsłabsze
warianty: czarne i żółte tło, obroty. Stąd długość obrotu 500 kroków: rozgrzewka (1500 kroków) kończy fazę wzrostu.

Kryterium kampanii (ustalone przed wynikiem): pole pod krzywą mAP@0,5 w obrotach 4–9 (średnia ocen co 100 kroków):
`koopman` > `losowe` i > `bez` w ≥ 4 z 5 ziaren. Wtórnie: domknięcie tożsamości (udział sond z zapalonym pełnym
latentem/rdzeniem per wariant) i jakość prognozy względem „braku zmiany”.

## Wyniki lokalne (2026-09-28, przerwane na prośbę użytkownika po ziarnie 42; dalej Eden 1817935, ziarna 47–56)

Ziarno 42 pełne (3 ramiona × 6 obrotów), ziarno 43 przerwane po pierwszym obrocie `koopman` (odrzucone).
- Pole pod krzywą mAP@0,5 (obroty 4–9): `koopman` 0,4415, `losowe` 0,4225, `bez` 0,3916. Koopman wygrywa w tym
  ziarnie (+0,019 nad losowym); rozrzut ocen co 100 kroków ±0,03, więc jedno ziarno nic nie rozstrzyga.
- Dodawanie klatek w ogóle pomaga (`losowe` − `bez` = +0,031): model od zera jest głodny danych, więc dobór ma tu
  pole do działania (przy dostrajanym RF-DETR różnice były rzędu 0,003).
- Rdzeń tożsamości pusty w każdym obrocie i ramieniu; pełny latent 28–62 neuronów; domknięcie wariantów 0,18–0,38
  bez trendu (średnio koopman 0,30, losowe 0,28, bez 0,26); najsłabsze: czarne/żółte tło, obroty.
- Prognoza (test wsteczny, 1 obrót): langevin / brak zmiany = 0,84; 1,21; 1,14; 1,22; 0,87 — średnio ≈ 1,06;
  czysty Koopman 4,30; 1,31; 1,04; 0,98; 0,81.
- FastGAN nie trafia w przewidziany latent: reszta obrazu „target” do celu 2,84 z-jednostki przy odległości
  cel–obecny 2,34 (obraz jest tak samo daleko od celu jak od stanu obecnego).
- Mimo to obraz „target” prowadzi do innych prawdziwych klatek niż obraz „current”: wspólne top-90 18,9/90,
  top-22 2,6/22, podobieństwo DINOv2 obrazów target–current 0,78 (`eden/kontrola_target_current.py`,
  `wyniki/KONTROLA_TARGET_CURRENT_s42.json`). Nie wiadomo, ile z tej różnicy to prognoza, a ile losowość optymalizacji.
