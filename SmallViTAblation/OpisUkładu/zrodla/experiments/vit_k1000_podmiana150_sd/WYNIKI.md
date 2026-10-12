# Wyniki: K1000 + podmiana 150 klatek (A150) wobec oryginalnej bazy (C0)

Bieg: Eden, zadanie **1824868** (H100, 2026-10-08 22:41-22:44; próba kodu 1824863). Każde ramię: 235 kroków
(5 epok × 47), uczenie 23 s. Plan i różnice wobec oryginalnego przepisu: [PLAN.md](PLAN.md).
Wyniki lokalnie: `outputs/vit_k1000_podmiana150_sd/eden/pelny/{C0,A150,C0b}/` (POMIARY.npz, METRYKI.json,
SZCZEGOLY_GRUP.json, LOG.json, metadata.json, końcowe `pelny/k1235.pt` dla C0 i A150; SHA256SUMS sprawdzone 19/19),
analiza `outputs/vit_k1000_podmiana150_sd/ANALIZA.json`, wykres `outputs/vit_k1000_podmiana150_sd/WYKRES.png`
(`kod/analiza.py`). Pełne stany po każdej epoce (k1000-k1235, wagi + AdamW) zostały na Edenie w
`~/smallvit_runs/2026-10-08_k1000_podmiana150_sd/wyniki/pelny/<ramię>/pelny/`.

## Najkrócej

- **Tryb deterministyczny działa:** C0b jest identyczne z C0 na każdym checkpoincie (różnica latentu 0,0, te same
  mAP i pewności). Każda różnica A150 − C0 jest więc skutkiem podmiany, a nie szumu GPU.
- **Podmiana mocno zmienia latent celu, ale prawie nie w stronę z\*:** ||z_A150 − z_C0|| wynosi 71-135, tyle co cały
  ruch uczenia w ramieniu (~100). Kosinus tej różnicy z δ = z\*_K1000 − z0 to −0,06 / 0,01 / 0,11 / 0,01 / 0,04
  (część widoczna dla fc2: −0,17 / 0,03 / 0,17 / 0,07 / 0,13). Postęp celu wzdłuż δ po 5 epokach: A150 0,5 %,
  C0 −0,1 % (najwięcej w K1141: 2,5 % wobec 0,6 %).
- **Detekcja celu bez zmiany:** IoU najpewniejszej ramki na 0055/4019 zostaje 0 w obu ramionach na wszystkich
  checkpointach. Pewność tej (złej) ramki w A150 0,57, w C0 0,35 na końcu.
- **mAP@0,5 wal3000:** start 0,308; koniec C0 0,369, A150 0,340. Po epokach: C0 0,351 / 0,309 / 0,370 / 0,336 / 0,369,
  A150 0,339 / 0,330 / 0,326 / 0,362 / 0,340. Wahania między epokami jednego ramienia (±0,03) są tego samego
  rzędu co różnica ramion; jedno ziarno nie rozstrzyga.
- **Własne klatki się uczą:** IoU na 150 nowych klatkach A150 0,56 → 0,74 (C0 0,51); na 150 usuniętych A150 0,62
  (C0 0,68).

## Tabela (zdjęcie celu i walidacja)

| krok | postęp C0 | postęp A150 | cos(D, δ) | cos(D, δ) fc2 | ‖D‖ | pewność celu C0 / A150 | IoU celu | mAP C0 / A150 |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1000 | 0 | 0 | - | - | 0 | 0,509 / 0,509 | 0 / 0 | 0,308 / 0,308 |
| 1047 | −0,25 % | −0,88 % | −0,061 | −0,166 | 71,5 | 0,521 / 0,639 | 0 / 0 | 0,351 / 0,339 |
| 1094 | 0,85 % | 1,02 % | 0,011 | 0,029 | 108,8 | 0,599 / 0,535 | 0 / 0 | 0,309 / 0,330 |
| 1141 | 0,59 % | 2,51 % | 0,106 | 0,174 | 122,9 | 0,453 / 0,353 | 0 / 0 | 0,370 / 0,326 |
| 1188 | 1,55 % | 1,84 % | 0,015 | 0,066 | 135,2 | 0,445 / 0,605 | 0 / 0 | 0,336 / 0,362 |
| 1235 | −0,08 % | 0,49 % | 0,035 | 0,132 | 108,5 | 0,352 / 0,567 | 0 / 0 | 0,369 / 0,340 |

Postęp = ⟨z(e) − z(K1000), δ⟩ / ||δ||², ||δ|| = 2,83 ||z0||. D = z_A150 − z_C0 (latent 600 × 768 zdjęcia celu).
Część widoczna liczona przez fc2 K1000 (po uczeniu fc2 się zmienia, to przybliżenie). Losowy kierunek w 460 800
wymiarach dałby |cos| rzędu 0,0015, ale to nie jest właściwa kontrola: brak ramienia z losowymi 150 klatkami.

## Sprawdzenia

- Pełny stan K1000 na Edenie: net + opt (AdamW, 2 grupy, 89 stanów) + krok 1000, SHA256 e7a7727c…; każdy tensor net
  równy assetowi MODEL_VIT_K1000.pt (sprawdzane w `bieg.py`). mAP na starcie 0,3084, jak w oryginalnym biegu (0,308).
- Latent celu na starcie z GPU (Eden) wobec CPU (lokalnie): różnica względna 7,7e-6.
- Wejścia: 0088/536 z listy PODOBNE było w bazie; zastąpione 0088/1484 (następna w rankingu DINO bez bazy).
- Na Edenie po sprzątaniu 2026-10-08 brakowało `od_zera/deps/timm` (katalogi bez plików), `od_zera/wejscia/pretrain/valid`
  i `probe/manifest.json`. Bieg używa własnych kopii w katalogu biegu: timm 0.9.16 z `F:\Python` oraz adnotacji COCO
  z `maly_vit_od_zera/eden/wejscia` (SHA256 zgodne z WEJSCIA_SHA256.json). Chronionych katalogów nie zmieniano.

## Czego to nie dowodzi

- 150 klatek dobrano pod sztuczny latent K8000, a uczenie i pomiar są w przestrzeni K1000 (polecenie użytkownika).
- z\*_K1000 słabszy niż w K8000: korekta daje IoU 0,70 przy pewności 0,06 (K8000: 0,999 / 0,81), „poza” 19,7.
- Jedno ziarno kontynuacji. Brak kontroli „losowe 150 klatek z puli”, więc nie wiadomo, czy mały dodatni kosinus to
  efekt wyboru przez SD i DINO, czy dowolnej podmiany; 143 ze 150 klatek pochodzi z filmu 0088.
