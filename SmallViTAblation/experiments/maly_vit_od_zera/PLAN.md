# Mały Vision Transformer uczony od zera: latent, który się rusza

Data: 2026-09-28. Status: **projekt, czeka na zatwierdzenie** (nic nie uruchomione).

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
