# DETR — 20kDataset small LR: wnioski z treningu (2025-12-13 17:54)

## Źródła
- Raport: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_sessions\2025-12-13_20kDataset_small_LR\detr_training_session_report.md`
- JSON: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_sessions\2025-12-13_20kDataset_small_LR\detr_training_session_analysis.json`
- Sesja: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR`

## Streszczenie (co się teraz dzieje)
- To jest **kontynuacja fine-tuningu po wznowieniu**: trening wznowił się z `final_checkpoint.pth` na epoce `172`.
- Nastąpił **reset learning rate** (fine-tuning): `1e-4 → 5e-5`, backbone `1e-5 → 5e-6`, a scheduler cosine został przeinicjalizowany na pozostałe ~`128` epok.
- W analizowanym oknie **model dalej się poprawia**: `val loss` spadł z `0.3410` (E173) do `0.1614` (E185), a najlepszy checkpoint to `checkpoint_epoch_185.pth`.
- Trening zakończył się **w połowie epoki**: `E186` na `993/1125` (~`88%`), więc sensownie jest wznawiać od E185.

## Kluczowe obserwacje z metryk
- **Trend lossów (E173→E185):**
  - train: `0.3431 → 0.1787` (spadek `-0.1644`)
  - val: `0.3410 → 0.1614` (spadek `-0.1796`)
  - średnia różnica train–val ~`0.009` (czasem `val < train`, co jest typowe gdy trening ma augmentacje, a walidacja nie).
- **Jednorazowy “spike” na walidacji** w `E183`: `0.1891 → 0.2320 → 0.1718` (E182→E183→E184).
  - Interpretacja: wygląda na szum/niestabilność walidacji lub zmienność batchy/seedów, nie na trwałą degradację (bo E184 i E185 wracają na trend i biją najlepszy wynik).
- **Gradient norm** spadł z ~`112.5` do ~`45.7` i ustabilizował się (brak NaN/Inf w checkpointach).
- **Składniki loss (epoch-level):**
  - `CE` ~0 (klasyfikacja jest praktycznie “łatwa”),
  - poprawa wynika głównie z **lokalizacji**: spadek `bbox` i `GIoU` (szczególnie `GIoU`).

## Stabilność / zasoby
- Brak errorów typu NaN/Inf w checkpointach; waga modelu rośnie płynnie (bez “eksplozji” norm).
- Pamięć GPU: w TensorBoard widać skok “reserved” do ~`6.4 GB` po początku okna (najpewniej alokacje/caching), bez dalszego narastania.
- Czas epoki stabilny: ~`3387 s` (~`56 min`).

## Rekomendacje (co robić dalej)
1) **Wznowić trening** od `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_185.pth`.
2) **Zweryfikować, czy walidacja jest deterministyczna i bez augmentacji** (spike E183 może wynikać z augmentacji w val / losowego samplingu / niedeterministycznego DataLoadera).
3) Dla interpretacji “co się poprawia” (poza samym lossem): policzyć mAP/AR na stałym val/test dla checkpointów `E170/E175/E180/E185` oraz porównać to z trendem `val loss`.

