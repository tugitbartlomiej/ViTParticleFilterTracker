# Pętla od początku dostrajania na większym modelu: RF-DETR LargeDeprecated (DINOv2-B)

Data: 2026-09-27. Polecenie użytkownika: „To weź ten z DINOv2-B, LargeDeprecated. Zrób” — po analizie
`../petla_uczenia_koopman_fastgan/ANALIZA_WCZESNIE_I_SKALA.md` (rekomendacja: pętla od początku dostrajania, prognoza
po pierwszej epoce, 3 ziarna, stała ekspozycja bazy, dobór „do nauczenia i jeszcze nienauczone”). Hipoteza
użytkownika: metoda sprawdzi się lepiej na większym modelu i w pierwszych epokach.

## Model

`RFDETRLargeDeprecated` z rfdetr 1.10.1: kręgosłup `dinov2_windowed_base` (DINOv2-B), 560 px, łatka 14, siatka
40 × 40, 768 cech, hidden_dim 384, 3 warstwy dekodera, 135 mln parametrów. Wagi COCO `rf-detr-large.pth`
(storage.googleapis.com/rfdetr, 1,57 GB, SHA256 86879a31…, MD5 zgodne z pakietem). Nano: DINOv2-S, 384 px, 24 × 24 × 384.

## Kod (kopie przerobione w `kod/`, oryginały równoległej sesji nietknięte)

- `kod/trening.py` (z `koopman_gesta_historia`): wybór modelu (`cfg.model`), konfiguracja odczytu sond pod siatkę
  modelu, sampler ze stałą ekspozycją bazy (`cfg.baza_n`: każda z 1500 klatek bazy raz na obrót, każda dodana raz,
  reszta losowe powtórzenia bazy), ziarno obrotu (`cfg.obrot`), wagi zapisywane tylko na końcu obrotu (`wagi_tylko_koniec`;
  0,5 GB na zapis, gęste zapisy przekroczyłyby kwotę Edena). Latent sond (LATENT_*.npz) co 5 aktualizacji.
- `kod/czastki.py` (z `koopman_fastgan_petla`): siatka, liczba cech i rozdzielczość przez `configure()`;
  sprawdzone lokalnie: opis sondy 3072 liczby, tokeny 1600 × 768.
- `eden/obrot_large.slurm`, `eden/kontynuacja_obrot.py`, `eden/config_seed{42,43,44}.json`.

## Przebieg

- Obrót = 1 epoka = 2560 próbek = 160 aktualizacji (batch 4 × 4), receptura LR jak bieg Nano 1810662.
- **Obrót 0 (wspólna rozgrzewka):** od wag COCO na 1500 klatkach bazy, bez dodawania, osobno dla ziaren 42, 43, 44.
  Daje historię latentu do dopasowania równania (32 punkty) i mierzy czas obrotu modelu Large.
- **Obroty 1–6:** trzy pętle na ziarno: koopman (cel FastGAN = latent przewidziany na koniec obrotu k+3, kotwice
  najbardziej niedouczone; spośród klatek podobnych do obrazów FastGAN wybór wg „do nauczenia i jeszcze
  nienauczone”: trudne dla bieżącego modelu, łatwe dla modelu odniesienia — Nano po pełnym uczeniu, który tych klatek
  nie widział), losowe, bez. 134 nowe klatki na obrót.
- FastGAN dla modelu Large na Edenie (odwracanie przez DINOv2-B jest za wolne na RTX 3070), reszta lokalnie.

## Miary i kryterium (ustalone przed wynikiem)

Tempo: obrót dojścia do progów mAP@0,5; pole pod krzywą mAP@0,5 po obrotach 1–6; AP przy kotwicach; tempo latentu.
Kryterium: pętla koopman ma większe pole pod krzywą mAP@0,5 niż losowe i bez w każdym z 3 ziaren, a w obrotach 1–3
średnio co najmniej o 0,005 wyżej niż bez. Przewidywanie literatury: jeśli przewaga jest, pojawi się w pierwszych
obrotach i zniknie, gdy dryf latentu zgaśnie.

## Etapy i koszt

1. Teraz: obrót 0 dla 3 ziaren (3 zadania). Wynik mierzy koszt; jeśli obrót trwa dłużej niż około 1,5 h na GPU,
   zatrzymuję się i raportuję przed obrotami 1–6.
2. Kod pętli dla Large (prognoza w nowych kierunkach tożsamości liczonych z latentu rozgrzewki, FastGAN na Edenie,
   ocena „do nauczenia”), próba na sucho, potem obroty 1–6.

## Uwagi z przebiegu (2026-09-27)

- Rozgrzewka k00: mAP@0,5 0,8622 (s42), 0,8610 (s43), 0,8591 (s44); 49 min na zadanie (hopper), w planie.
- Obrót 1: zadanie pętli 50–55 min (hopper, dgx-2), razem z FastGAN i wyborem około 65 min na obrót.
- Wybór klatek: faiss-cpu i torch z F:\Python ładują dwa środowiska OpenMP (OMP Error #15) w każdej kolejności
  importu; etap `wybor` ustawia KMP_DUPLICATE_LIB_OK=TRUE. Sprawdzone: top-90 z faiss = top-90 z iloczynu
  skalarnego w numpy na 20 zapytaniach, różnica wyniku 8e-7.
- Kwota Edenu: obrót 1 zapisywał na końcu każdego zadania FULL_*.ckpt, last.ckpt i FINAL.ckpt (3 × 1,57 GB) oraz
  FINAL.pth; sprzątanie usuwało je dopiero po walidacji, więc chwilowo zajęte było ~30 GB (wolne spadło do 4 GB).
  Od obrotu 2 (`wagi_tylko_koniec`) te pliki nie powstają: następny obrót startuje z `model_0000160.pth`,
  stanu optymalizatora nikt nie czyta, a zapis nie zmienia przebiegu uczenia. Przed każdym obrotem `prognoza`
  sprawdza kwotę projektu 1299 i zatrzymuje dyrygenta poniżej 8 GB wolnego.
