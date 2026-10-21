# Opposite beams (przeciwne / przeciwstawne wiązki) in a small ViT during training, epochs 0–200: the project's own evidence

Compiled 2026-10-21 by reading local files only. Nothing was modified, re-run or submitted.

The project root is `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`. Every path below starts from it, using these abbreviations:

| Tag | Absolute path |
|---|---|
| [ATLAS] | `…\SmallViTAblation\experiments\vit_wiazkiprzeciwstawne_20261020__18-10-00` |
| [P200] | `…\SmallViTAblation\experiments\pary_przeciwne_do_200epok_20261020__23-21-30` |
| [MAR] | `…\SmallViTAblation\experiments\kanal_mar25l_checkpointy_20261020__18-38-50` |
| [K200] | `…\SmallViTAblation\experiments\vit_kontynuacja_do_200epok_20261020__19-06-42` |
| [K400] | `…\SmallViTAblation\experiments\vit_kontynuacja_do_400epok_20261020__23-50-57` |
| [FUN24] | `…\SmallViTAblation\outputs\kompensacja_funkcjonalna_dgx2_20261017` |
| [PRUNE] | `…\SmallViTAblation\outputs\kompensacja_pruning_dynamics_20261018` |
| [RESTORE] | `…\SmallViTAblation\outputs\kompensacja_restore_all_20261018` |
| [REL] | `…\SmallViTAblation\outputs\kompensacja_wklad_wzgledny_20261019` |
| [S3] | `…\SmallViTAblation\outputs\przeciwstawne_eden_20261017_3seeds` (plus `outputs\przeciwstawne_eden_ciag_20261017`) |
| [SFERA] | `…\SmallViTAblation\outputs\wiazki_sfera_abcd_20261019` |
| [AUDYT] | `…\SmallViTAblation\outputs\opposition_sphere_audit_20261019` |
| [WSP] | `…\SmallViTAblation\outputs\wspoldzialanie_neuronow_20261015` |
| [LIT] | `…\SmallViTAblation\plan_na_przyszle_eksperymenty\2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md` and `…\SmallViTAblation\docs\2026-10-20_literatura_przeciwstawne_neurony\PRACE_PELNE_TEKSTY.md` |

Labels used throughout:
- "derived" means I computed the value myself from the cited CSVs; the experiment files do not state it.
- "Inference" means my own interpretation, not a measured result.
- Numbers are copied exactly, except values that I printed rounded from CSVs; those say "rounded".

## Takeaway

The project uses two different notions of "opposition":
- **Same-token signed beams at ≥150°.** This is the main object of the 0–200-epoch experiments. It is measured for one seed (3047).
- **Anti-parallel W2 columns (cos < −0.5, i.e. >120°).** This is the basis of the older campaigns, which have 24 seeds but cover only epochs 0–60 and the last MLP.

The two sets do not overlap. In epochs 60–200, every pair that reached ≥150° had a W2 angle ≤ 29.98° (derived), so these pairs are never anti-parallel in W2.

What holds in seed 3047 over epochs 0–200:
- Opposite beams appear only after detection IoU has mostly saturated.
- They never exceed 173.0°.
- They come from near-parallel W2 columns combined with opposite GELU signs.
- They peak at 30 pairs (epochs 80–110) and fall to 15 by epoch 200.
- The single ≥170° pair is a pair of copies (kopie), not antagonists. Ablating it has no effect on detection.

Across the older 24-seed work, functional compensation exists and is output-specific, but switching pairs off or restoring them does not improve IoU.

---

## Observations

### O1. Opposite beam pairs (≥150°) are absent at initialization, appear around epoch 9, peak at epochs 80–110 and then decline

**Statement.** No pair of beams on the same token reaches ≥150° in any block from step 0 to step 6000 (epoch 5.49).
- The first 2 pairs appear at step 10000 (epoch 9.14), in blocks 2 and 5.
- Summed over the 6 blocks, the count reaches 25 at epochs 40–50 and 24 at epoch 60.
- It peaks at 30 pairs at epochs 80, 90 and 110 (29 at epoch 100).
- It then falls: 21 at epoch 120 and 15 at epochs 180–200.

**Exact numbers (all blocks summed; per-block ≥150° counts listed as b1/b2/b3/b4/b5/b6).**

| Step | Epoch | ≥150° | ≥160° | ≥170° | ≥175° | Max angle (°) | ≥150° per block | Pair-token occurrences ≥150° | Same, both beams ≥5% | Unique pairs ≥150°, both ≥5% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0 | 0 | 0 | 0 | 110.231 | 0/0/0/0/0/0 | 0 | 0 | 0 |
| 3000 | 2.74 | 0 | 0 | 0 | 0 | 145.000 | 0/0/0/0/0/0 | 0 | 0 | 0 |
| 6000 | 5.49 | 0 | 0 | 0 | 0 | 147.697 | 0/0/0/0/0/0 | 0 | 0 | 0 |
| 10000 | 9.14 | 2 | 0 | 0 | 0 | 150.261 | 0/1/0/0/1/0 | 9556 | 3290 | 2 |
| 16000 | 14.63 | 9 | 0 | 0 | 0 | 154.549 | 0/1/0/1/5/2 | 36048 | 12287 | 9 |
| 24000 | 21.94 | 13 | 0 | 0 | 0 | 158.547 | 1/1/0/1/4/6 | 60096 | 21384 | 13 |
| 33000 | 30.17 | 18 | 1 | 0 | 0 | 162.035 | 3/1/0/1/6/7 | 74837 | 17851 | 18 |
| 44000 | 40.23 | 25 | 2 | 0 | 0 | 164.765 | 6/1/0/2/6/10 | 115240 | 18929 | 25 |
| 55000 | 50.29 | 25 | 2 | 0 | 0 | 165.021 | 6/1/0/1/6/11 | 89059 | 12994 | 25 |
| 65640 | 60.01 | 24 | 2 | 0 | 0 | 164.196 | 7/1/0/1/6/9 | 95092 | 12662 | 23 |
| 76563 | 70.00 | 28 | 2 | 0 | 0 | 164.263 | 8/1/0/3/5/11 | 99571 | 13454 | 28 |
| 87500 | 80.00 | 30 | 2 | 0 | 0 | 165.753 | 8/0/0/4/6/12 | 147647 | 19600 | 30 |
| 98438 | 90.00 | 30 | 2 | 0 | 0 | 165.080 | 8/0/0/3/7/12 | 149318 | 13407 | 30 |
| 109375 | 100.00 | 29 | 2 | 0 | 0 | 165.034 | 6/0/0/4/6/13 | 128391 | 13540 | 29 |
| 120313 | 110.00 | 30 | 4 | 0 | 0 | 164.460 | 6/0/0/4/7/13 | 162725 | 17934 | 30 |
| 131250 | 120.00 | 21 | 2 | 0 | 0 | 162.796 | 6/0/0/3/4/8 | 104496 | 9906 | 21 |
| 142188 | 130.00 | 22 | 1 | 0 | 0 | 160.917 | 4/0/0/3/6/9 | 105070 | 5789 | 22 |
| 153125 | 140.00 | 25 | 1 | 0 | 0 | 160.573 | 3/0/0/3/7/12 | 118386 | 2084 | 25 |
| 164063 | 150.00 | 21 | 3 | 0 | 0 | 164.975 | 3/0/0/3/6/9 | 93392 | 4679 | 21 |
| 175000 | 160.00 | 16 | 3 | 0 | 0 | 167.809 | 2/0/0/3/6/5 | 84473 | 5268 | 16 |
| 185938 | 170.00 | 18 | 2 | 1 | 0 | 170.412 | 2/0/0/4/6/6 | 86791 | 3746 | 18 |
| 196875 | 180.00 | 15 | 1 | 1 | 0 | 170.960 | 0/0/0/3/5/7 | 85366 | 2258 | 15 |
| 207813 | 190.00 | 15 | 1 | 1 | 0 | 171.467 | 0/0/0/4/5/6 | 69422 | 2961 | 15 |
| 218750 | 200.00 | 15 | 1 | 1 | 0 | 173.000 | 1/0/0/2/5/7 | 86808 | 4249 | 14 |

Rows for steps 300 and 1000 are all zero; their maximum angles are 110.052° and 124.529°.
- Source: `[P200]\Wyniki\Dane\polaczone_0_200\SUMA_BLOKOW_0_200.csv`.
- The two "both beams ≥5%" columns are summed from `COUNTS_0_200.csv` (derived).
- The headline numbers also appear in `[P200]\Wyniki\Raport\RAPORT.html` §3.1 and `[P200]\Wyniki\Krótki_Wniosek\WNIOSEK.txt`.

**Per-block notes.**
- **Block 3** never has a pair ≥150° at any of the 26 checkpoints. Its maximum is 148.29° at epoch 200.
- **Block 2** has exactly one pair, (87,112), from epoch 9.14 to epoch 70. It peaks at 156.26° at step 33000 and drops to 141.86° by epoch 200.
- **Block 1** falls from 8 pairs to 1 after epoch 90.
- **Block 6** has the most pairs at most checkpoints from epoch 21.94 onward. The exceptions are epoch 160 (block 5 has 6, block 6 has 5) and epoch 170 (a tie at 6).
- At epoch 200, the maximum angle per block is 150.1°, 141.9°, 148.3°, 159.4°, 158.4° and 173.0° (`RAPORT.html` §3.1; `COUNTS_0_200.csv`).
- For epochs 0–60 the same numbers also appear in `[ATLAS]\Wyniki\Dane\opposite_pair_counts_by_epoch\PLOT_SERIES.csv`, `[ATLAS]\Wyniki\Dane\opposite_pair_counts_20261020__18-52-50\COUNTS_PER_BLOCK.csv` and `[ATLAS]\Wyniki\Krótki_Wniosek\WNIOSEK_PARY_PRZECIWNE_0_55000_65640.txt` (0 pairs at step 0, 25 at step 55000, 24 at step 65640).

**Sample size.**
- 1 seed (3047), 26 checkpoints.
- The 0–60 range has 12 irregular checkpoints (0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640). After that there is one every 10 epochs.
- 50 fixed TEST frames (2 per film from 25 films), 600 tokens each, giving 30,000 tokens.
- All 294,528 unordered neuron pairs in each of the 6 blocks.

**Robustness: moderate.**
- Within the seed the count is exhaustive (every pair) and does not depend on which frames were sampled. At every checkpoint the observed number of unique pairs at ≥150° equals the weight-based upper bound: `possible_signed_unique_pairs_all_blocks` equals `unique_pairs_all_blocks` in SUMA_BLOKOW_0_200.csv, and `LOCAL_QC.json` reports `"observed_unique_equals_global_possible_at_all_thresholds": true` for steps 0, 55000 and 65640.
- Weakness: one training run only. There are no confidence intervals on the counts, and the trajectory is not smooth (for example 30 → 21 between epochs 110 and 120, then 25 at epoch 140).

**Caveats.**
- A unique pair counts once if it reaches the threshold on at least one token.
- The pair-token occurrence columns depend on the 50 frames. The active-5% columns depend on the activity threshold.
- A large angle does not by itself mean cancellation, because the two strengths may be unequal (`[ATLAS]\Wyniki\Krótki_Wniosek\WNIOSEK_PARY_PRZECIWNE_0_55000_65640.txt`).

### O2. No pair ever reaches 175°; only one pair reaches ≥170°, and only from epoch 170

**Statement.**
- Across all 26 checkpoints and all blocks: 0 pairs at ≥175°, ≥179° and ≥179.999999°.
- ≥170° appears first at epoch 170 (step 185938), with 1 pair and an angle of 170.412°. It is the same pair at epochs 180, 190 and 200, where it reaches 170.960°, 171.467° and 173.000°.
- ≥160° counts stay between 0 and 4 pairs. The first ≥160° pair appears at epoch 30.17 in block 5.
- The largest angle over the whole run (the maximum over blocks) grows as follows: 110.23° at step 0, 124.53° at 0.91 epochs, 145.00° at 2.74, 150.26° at 9.14, 162.04° at 30.17, 165.02° at 50.29, 165.75° at 80, 160.57° at 140, 167.81° at 160, 173.00° at 200.

**Sources.**
- `[P200]\Wyniki\Dane\polaczone_0_200\SUMA_BLOKOW_0_200.csv`
- `[P200]\Wyniki\Raport\RAPORT.html` ("Pierwsza w historii para ≥170° pojawia się w epoce 170, a w epoce 200 ma 173,0°. Progu ≥175° nie osiąga żadna para")
- For epochs 0–60: `[ATLAS]\Wyniki\Dane\opposite_pair_counts_by_epoch\PLOT_QC.json` (`"no_pairs_ge170_at_any_saved_checkpoint": true`)

**Sample size.** As in O1.

**Robustness: moderate.**
- The absence of ≥170° pairs up to epoch 160 is a property of the weights, not of the frame sample. At each checkpoint the bound max(α, 180°−α), computed from W2 angles under both possible GELU signs, stays below the threshold for every pair (`[ATLAS]\Opis\LICZENIE_PAR_0_BEST_FINAL.txt`; `WNIOSEK_PARY_PRZECIWNE_0_55000_65640.txt`).
- This is still a single seed.

**Caveat.**
- Before epoch 170, the "maximum angle" is set by different pairs in different blocks. Pair identities are given in the `maximum_observed_pair` column of `COUNTS_0_200.csv`.

### O3. Mechanism: opposite beams come from near-parallel W2 columns combined with opposite GELU signs, never from anti-parallel W2

**Statement.**
- No pair of W2 columns has an angle ≥150° at any checkpoint: `weight_only_unique_pairs_all_blocks` is 0 at every step.
- The largest W2 column angle, taken over all blocks, is:

  | Epoch | 0 | 9.14 | 50.29 | 60 | 200 |
  |---|---|---|---|---|---|
  | Max W2 angle (°) | 110.23 | 132.56 | 140.42 | 139.63 | 142.31 |

- For W2 angle α, the two possible beam angles are α (same GELU signs) and 180°−α (opposite signs). Since α < 150° always, a pair reaches ≥150° only when α ≤ 30° and the GELU signs are opposite on that token.
- Checked directly on the epoch 60–200 data:
  - All 339 candidate rows reaching ≥150° (`CANDIDATE_PAIRS_60_200.csv`) have a W2 angle ≤ 29.98° (derived).
  - The 15 pairs at epoch 200 have W2 angles from 7.00° (pair 370–391) to 29.89° (derived).
  - For each of them, the number of tokens at ≥150° equals the number of tokens with opposite signs and nonzero beams (`count_ge_150` = `opposite_sign_nonzero_tokens`, derived).
- The report states: "Żadna para nie ma przeciwnych wag W2: największy kąt kolumn W2 to ok. 142°. Wszystkie przeciwne wiązki biorą się z prawie równoległych W2 i przeciwnych znaków GELU." (`[P200]\Wyniki\Raport\RAPORT.html` §3.1)

**Supporting evidence from other campaigns.**
- **Last MLP, 24 seeds, final checkpoint.** For functionally selected pairs vs matched controls, both GELU activations are negative in 88.39% vs 84.83% of cases, and the activation signs are opposite in 10.50% vs 14.19% (`[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`, table "Geometria wag a faktyczne wkłady").
- **Sphere pilot (seed 3047, step 65640).** No pair-token among the 19,575 pairs selected for being 3D-opposite reaches a signed 150° in full 192D ("In full 192D, no pair-token reaches a signed 150°"; `[SFERA]\RAPORT_PL.md`).
  - Inference: this is consistent with O3. Those pairs are anti-parallel in the 3D projection of W2, but in 192D their W2 angle is <140°, so neither α nor 180°−α can reach 150°.

**Sample size.**
- 1 seed, 26 checkpoints, all pairs in all blocks.
- The supporting GELU-sign percentages come from 24 seeds, last MLP, epochs 0–60.

**Robustness.**
- Strong within seed 3047: the result follows directly from the weights and holds at every checkpoint.
- Not tested on other seeds for epochs 60–200.

**Caveats.**
- "Near-parallel" here means W2 angles ≤30°, i.e. cos ≥ 0.866. As a result, the ≥150° beam pairs and the W2-anti-parallel family used in the older campaigns (cos < −0.5) are disjoint sets (derived; see O12 and the inconsistency notes in the Gaps section).

### O4. Opposite-beam occurrences on tokens are rare and fall after epoch 110, especially when both beams are active

**Statement.**
- Occurrences of pair-tokens at ≥150° (all blocks, 30,000 tokens per block) peak at 162,725 at epoch 110 and are 86,808 at epoch 200.
- When both beams must exceed 5% of the strongest beam on the token, the count:
  - peaks at 21,384 at epoch 21.94;
  - is 19,600 at epoch 80 and 17,934 at epoch 110;
  - then drops to 2,084 at epoch 140 and 4,249 at epoch 200.
- The report summarizes this as: "Przy obu wiązkach aktywnych spadek po epoce 110 jest wyraźny".
- Nearly every ≥150° pair has at least one token where both beams are active: 14 of 15 at epoch 200 and 23 of 24 at epoch 60. The exception at epoch 200 is block-6 pair 81–258, which has 146 opposite-sign tokens and 0 active ones (derived from `CANDIDATE_PAIRS_60_200.csv`).

**Sources.**
- `[P200]\Wyniki\Dane\polaczone_0_200\COUNTS_0_200.csv` and `SUMA_BLOKOW_0_200.csv`
- `[P200]\Wyniki\Raport\RAPORT.html` §3.1 and Rys. 4

**Sample size.** 1 seed, 50 frames.

**Robustness: preliminary to moderate.** Unlike the unique-pair counts, these occurrence counts depend on the frame sample.

**Caveat.** A pair-token is counted per token and per block, so one pair present on many tokens dominates the count.

### O5. The identities of opposite pairs turn over substantially

**Statement.**
- Of the 24 pairs present at epoch 60, only 5 survive to epoch 200 ("Skład par się wymienia: z 24 par z epoki 60 do epoki 200 przetrwało 5"; `[P200]\Wyniki\Raport\RAPORT.html` §3.1).
- The figures below are derived from `[ATLAS]\Wyniki\Dane\opposite_pair_counts_by_epoch\INTERMEDIATE_PAIRS.csv`, `[ATLAS]\Wyniki\Dane\opposite_pair_counts_20261020__18-52-50\CANDIDATE_PAIRS.csv` and `[P200]\Wyniki\Dane\pary_epoki_60_200\CANDIDATE_PAIRS_60_200.csv`. The per-step totals reproduce the official counts exactly.
  - **Distinct identities.** 73 distinct (block, neuron, neuron) identities reach ≥150° at some point across the 26 checkpoints.
  - **Survival to epoch 200.**

    | From epoch | 30 | 50 | 60 | 110 |
    |---|---|---|---|---|
    | Pairs surviving to epoch 200 | 3 of 18 | 5 of 25 | 5 of 24 | 10 of 30 |

  - **Adjacent checkpoints in late training.** Persistence is 12–16 of 15–21, for example 13 of 18 from epoch 170 to 180 and 12 of 15 from 190 to 200.
  - **Longest-lived pairs.**

    | Pair (block: neurons) | Checkpoints present (of 26) | Range |
    |---|---|---|
    | 5: 81–108 | 21 | epochs 9.14–200 |
    | 5: 429–443 | 20 | epochs 14.6–200 |
    | 5: 249–373 | 18 | epochs 14.6–180 |

  - **Pair 370–391** (block 6) reaches ≥150° only from epoch 140 (step 153125).
- The older 24-seed W2-geometric campaign also reports turnover: between steps 55000 and 65640, on average 299.13 pairs cross the cos < −0.5 threshold towards opposition and 340.25 leave it, while about 424 neurons keep at least one opponent (`[PRUNE]\RAPORT_KONCOWY_PL.md`; `[PRUNE]\analysis\DYNAMICS_INTERPRETATION_PL.md` L44: 299.125 and 340.250).

**Sample size.**
- Beams: 1 seed.
- W2 turnover: 24 seeds.

**Robustness.**
- Beam turnover: preliminary. Only the "5 of 24" figure is in the files; the rest is derived by me.
- W2 turnover: strong, but it uses a different definition.

### O6. Opposite pairs appear after detection IoU has largely saturated

**Statement.**

| | Epoch 2.74 | Epoch 9.14 | Epoch 21.94 | Epochs 40–200 |
|---|---|---|---|---|
| ≥150° pairs | 0 (max angle 145.0°) | 2 (first appearance) | 13 | 25 → 30 → 15 |
| VAL IoU | 0.6420 | 0.7207 | 0.7452 | between 0.748 and 0.775 |
| Share of epoch-200 VAL IoU (0.7654, derived) | 84% | 94% | 97% | — |

- The ≥170° pair appears at epoch 170, with no change in IoU.
- The 50-frame diagnostic panel (DIAG50) used for pair counting shows the same pattern: IoU 0.6720 at epoch 2.74, 0.7293 at epoch 50.29, 0.7401 at epoch 200.

**Sources.**
- `[P200]\Wyniki\Dane\detekcja_0_200\DETEKCJA_0_200.csv`
- `[P200]\Wyniki\Dane\polaczone_0_200\SUMA_BLOKOW_0_200.csv`
- `[P200]\Wyniki\Raport\RAPORT.html` §3.2 ("Od epoki 40 do 200 IoU waha się w wąskim pasie: VAL 0,748–0,775, TEST 0,736–0,762")
- `[ATLAS]\Wyniki\Raport\RAPORT.html` §3 ("Duża część poprawy IoU następuje już do kroku 3000")

**Multi-seed evidence (W2 definition, last MLP, 24 seeds, epochs 0–60).**
- The largest gains in IoU and score occur in the 0–300-step interval in all 24 seeds.
- At steps 0 and 300, no neuron has a W2 partner with cos < −0.5.
- The largest absolute increase in the number of neurons with an opponent falls in 10000–16000 for 20/24 seeds and in 16000–24000 for 4/24.
- After controlling for seed and interval, all six planned correlations between changes in IoU and changes in opposition (first differences or rates; lags −1, 0, +1) have confidence intervals that include zero. At lag 0: r = 0.0704, CI [−0.0400; +0.1613] for differences; r = 0.0101, CI [−0.0187; +0.0380] for rates.
- Over 108 analyses, 8 confidence intervals exclude zero, without correction for multiple comparisons.
- Sources: `[PRUNE]\RAPORT_KONCOWY_PL.md`, section "Czy przeciwstawność pojawia się nagle wraz z uczeniem?"; `[PRUNE]\analysis\DYNAMICS_INTERPRETATION_PL.md` L7, L40, L72–74.

**Dense single-run series** (806 checkpoints, W2 definition; the seed is not stated):
- After detrending, the six correlations of P with IoU and confidence range from −0.027 to 0.052.
- The first positive share of opposed neurons appears at step 3200 (epoch 2.925).
- Source: `[WSP]\przeciwne_reflektory\RAPORT.md`.

**Mar25l pilot** (3 seeds):
- The largest IoU gain is in 0–300 in all 3 seeds.
- The largest growth in opposed pairs is in 10000–16000, 16000–24000 and 24000–33000 for seeds 3047, 4047 and 5047 respectively.
- Source: `[MAR]\Wyniki\Dane\eden\redukcja_pilot\HYPOTHESES_AND_SUMMARY.json` (`timing_per_seed`).

**Robustness.**
- For the W2 definition: strong (24 seeds, pre-planned detrended correlations).
- For the same-token beam definition: moderate (1 seed, no statistical test of timing).

**Caveats.**
- Checkpoints are sparse early in training, so onset is known only as an interval.
- Co-occurrence in time does not establish whether one causes the other.

### O7. Detection over epochs 0–200: IoU plateaus from about epoch 40 while confidence keeps rising, including on frames with no tool

**Statement.** The table below is rounded to 4 decimals from `DETEKCJA_0_200.csv`. "Score" is the mean sigmoid score on frames with a tool; "Empty" is the mean top-1 score on frames without one.

| Epoch | VAL IoU | VAL score | VAL IoU≥0.5 share | VAL empty | TEST IoU | TEST score | TEST empty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0019 | 0.0129 | 0.0000 | 0.0129 | 0.0012 | 0.0129 | 0.0129 |
| 0.27 | 0.3939 | 0.3994 | 0.3240 | 0.3300 | 0.3578 | 0.3880 | 0.3078 |
| 2.74 | 0.6420 | 0.7469 | 0.7983 | 0.5263 | 0.6644 | 0.7442 | 0.4181 |
| 9.14 | 0.7207 | 0.7922 | 0.8970 | 0.5573 | 0.7036 | 0.7760 | 0.4338 |
| 30.17 | 0.7459 | 0.8419 | 0.9142 | 0.6795 | 0.7322 | 0.8169 | 0.5136 |
| 50.29 | 0.7619 | 0.8634 | 0.9249 | 0.6292 | 0.7450 | 0.8620 | 0.5407 |
| 60.01 | 0.7600 | 0.8811 | 0.9292 | 0.6502 | 0.7470 | 0.8753 | 0.5182 |
| 80 | 0.7632 | 0.8984 | 0.9270 | 0.6377 | 0.7514 | 0.8910 | 0.5496 |
| 110 | 0.7738 | 0.8909 | 0.9292 | 0.7221 | 0.7584 | 0.8845 | 0.6274 |
| 120 | 0.7573 | 0.8800 | 0.8927 | 0.6239 | 0.7405 | 0.8677 | 0.5131 |
| 130 | 0.7754 | 0.8819 | 0.9313 | 0.6851 | 0.7566 | 0.8689 | 0.5257 |
| 150 | 0.7722 | 0.8956 | 0.9249 | 0.6760 | 0.7448 | 0.8908 | 0.5485 |
| 180 | 0.7735 | 0.9132 | 0.9142 | 0.7132 | 0.7596 | 0.9071 | 0.5754 |
| 200 | 0.7654 | 0.8992 | 0.9120 | 0.6747 | 0.7559 | 0.8868 | 0.5787 |

**95% confidence intervals for IoU** (film-level bootstrap, 2000 resamples; `RAPORT.html` §3.2):

| Epoch | VAL IoU [95% CI] | TEST IoU [95% CI] |
|---|---|---|
| 9.1 | 0.721 [0.685; 0.753] | 0.704 [0.666; 0.737] |
| 50 | 0.762 [0.725; 0.795] | 0.745 [0.712; 0.777] |
| 110 | 0.774 [0.743; 0.800] | — |
| 200 | 0.765 [0.730; 0.796] | 0.756 [0.719; 0.791] |

- "Przedziały wszystkich epok się nakładają" (all epochs' intervals overlap).
- Confidence on tool frames rises from 0.863 to 0.899 on VAL and from 0.862 to 0.887 on TEST (epoch 50 → 200).
- Confidence on empty frames rises from 0.629 to 0.675 on VAL and from 0.541 to 0.579 on TEST.
- The report reads this as "Dłuższy trening daje większą pewność bez lepszej lokalizacji". It calls it "typowy objaw przeuczenia kalibracji" but says this was not tested separately ("Nie sprawdzaliśmy go osobno, na przykład ECE").

**Sources.**
- `[P200]\Wyniki\Dane\detekcja_0_200\DETEKCJA_0_200.csv`, `DETEKCJA_BOOTSTRAP.csv`, `BOOTSTRAP_MANIFEST.json` (2000 resamples, seed 3047, film-level cluster bootstrap, 95% percentile intervals)
- `[P200]\Wyniki\Raport\RAPORT.html` §3.2
- `[P200]\Wyniki\Krótki_Wniosek\WNIOSEK.txt`

**Sample size.**
- 1 seed, 26 checkpoints.
- VAL: 500 frames (466 with a tool, 34 empty). VAL was used earlier to choose the checkpoint.
- TEST: 500 frames (472 with a tool, 28 empty). TEST is used for reporting only, but had been viewed in earlier campaigns.
- 25 films per split. The empty frames come from 11 VAL films (`DETEKCJA_BOOTSTRAP.csv`, films = 11).

**Robustness.**
- IoU plateau: moderate (one seed; confidence intervals overlap).
- Rising confidence on empty frames: preliminary. There are few empty frames, and the trend is not monotonic: VAL empty-frame confidence is 0.7221 at epoch 110 and 0.6239 at epoch 120.

**QC.** VAL IoU and score for the 12 checkpoints from step 0 to 65640 are identical to the parent's checkpoint-selection record (max_abs_iou_diff 0.0; `[P200]\Wyniki\Dane\detekcja_0_200\QC_VAL_VS_PARENT_SELECTION.json`).

**Caveat: the "best checkpoint" label.** Step 55000 (VAL IoU 0.7619) was the VAL-best only within epochs 0–60. Over epochs 0–200 the VAL maximum is 0.7754 at epoch 130, with overlapping confidence intervals.

### O8. Pair 370–391 (block 6, the only pair ≥160° at epoch 200) becomes a pair of copies (kopie) after epoch 100

**Statement.**
- Up to epoch 100 the two neurons are independent: W2 angle between 72° and 93°, input weight cosine W1 ≤ 0.31.
- Between epochs 100 and 150 they converge: W2 angle falls to 15° and W1 cosine rises to 0.90.
- At epoch 200: W2 angle 7.0°, W1 cosine 0.964.
- W2 norms grow, and the effective biases (b̂, see Definitions) move negative.
- At epoch 200 both neurons sit in the negative GELU tail together on 91% of tokens (27,150 of 29,896). The correlation of their activations is 0.80 at epoch 200 and 0.03 at epoch 60.

**Exact history** (`[P200]\Wyniki\Dane\znoszenie_370_391\PARA_WAGI_0_200.csv`). W1 cosine is computed after folding in the LayerNorm.

| Epoch | W2 angle (°) | W1 cosine | W2 norm, 370 | W2 norm, 391 | b̂, 370 | b̂, 391 |
|---|---|---|---|---|---|---|
| 0 | 87.95 | −0.0009 | 0.312 | 0.288 | −0.041 | +0.043 |
| 9.14 | 85.09 | 0.228 | 0.320 | 0.353 | −0.082 | −0.055 |
| 50.29 | 86.78 | 0.283 | 0.587 | 0.893 | −0.346 | −0.333 |
| 60.01 | 84.84 | 0.281 | 0.602 | 0.890 | −0.440 | −0.434 |
| 70 | 76.62 | 0.122 | 0.623 | 0.871 | −0.413 | −0.543 |
| 80 | 75.52 | 0.194 | 0.583 | 0.874 | −0.554 | −0.455 |
| 90 | 77.67 | 0.276 | 0.586 | 0.863 | −0.539 | −0.536 |
| 100 | 72.28 | 0.307 | 0.647 | 0.869 | −0.815 | −0.628 |
| 110 | 61.71 | 0.412 | 0.810 | 0.817 | −0.717 | −0.745 |
| 120 | 49.47 | 0.565 | 1.194 | 0.882 | −0.928 | −0.810 |
| 130 | 30.98 | 0.779 | 1.529 | 1.104 | −1.073 | −0.994 |
| 140 | 21.46 | 0.867 | 1.738 | 1.277 | −1.100 | −0.953 |
| 150 | 15.02 | 0.903 | 1.850 | 1.395 | −1.197 | −1.093 |
| 160 | 12.19 | 0.928 | 1.943 | 1.518 | −1.200 | −1.120 |
| 170 | 9.59 | 0.941 | 2.105 | 1.704 | −1.261 | −1.031 |
| 180 | 9.04 | 0.958 | 2.093 | 1.750 | −1.314 | −1.175 |
| 190 | 8.53 | 0.955 | 2.104 | 1.775 | −1.397 | −1.056 |
| 200 | 7.00 | 0.964 | 2.138 | 1.822 | −1.473 | −1.106 |

All values rounded. The raw W1 cosine at epoch 200 is 0.963.

**Sources.**
- The table above.
- `[P200]\Wyniki\Raport\RAPORT.html` §3.3 ("Historia wag", including the 91% negative-tail and correlation figures)
- `[P200]\Wyniki\Krótki_Wniosek\WNIOSEK.txt` ("Para 370-391 to dwie kopie, a nie przeciwnicy")

**Sample size.** One pair, 1 seed, 26 checkpoints. Token statistics use the 50 diagnostic frames.

**Robustness: preliminary.** This is a single pair, though it is the most extreme one.

**Caveats.**
- This is the only pair whose weight history (W1 cosine, norms, biases) was measured. `CANDIDATE_PAIRS*.csv` stores W2 angles for the other pairs but not W1 cosines.
- The report gives weight decay as the "probable mechanism" ("kara L2 woli dwie zgodne połówki niż dwie duże przeciwne wagi"), and explicitly labels this interpretation as untested ("Poniższa interpretacja nie była osobno testowana").

### O9. Pair 370–391 at epoch 200: real but local and small cancellation on 6% of tokens; its beams add up on the other 94%

**Statement** (29,896 tokens on which both beams are nonzero). κ is the energy-cancellation index (2 = equal and aligned, 1 = independent, 0 = full cancellation); "residual" is what is left of the sum of the two strengths.

| Token set | n | Angle | Strength ratio (median) | Aggregate κ | Residual (median) | Share of MLP output (median) |
|---|---|---|---|---|---|---|
| All | 29,896 | – | 0.66 | 1.73 | 1.00 | 1.5% |
| Same signs | 28,107 (94%) | 7.0° | 0.68 | 1.80 | 1.00 | 1.6% |
| Opposite signs | 1,789 (6%) | 173.0° | 0.38 | 0.54 | 0.45 | 0.56% |
| Opposite signs, both beams ≥5% active | 269 | 173.0° | 0.75 | 0.19 | 0.16 | 0.32% |

- On opposite-sign tokens the negative neuron is almost always 391: 1,718 of 1,789 tokens, and all 269 active tokens.
- These tokens rarely lie inside the tool box: 1.3%, against an average of 4.3%.
- Where both beams are active, the pair is about 2% of the MLP output on that token, and 0.3% after cancellation.
- The short conclusion gives "ok. 0,3–0,6% wyjścia MLP".

**Sources.**
- `[P200]\Wyniki\Raport\RAPORT.html` §3.3 (token table)
- `[P200]\Wyniki\Dane\znoszenie_370_391\PARA_TOKENY_0_200.csv` (per-checkpoint token statistics, categories both_nonzero / same_sign / opposite_sign / *_active05)
- `[P200]\Wyniki\Krótki_Wniosek\WNIOSEK.txt`

**Sample size.** 1 pair, 1 seed, 50 frames.

**Robustness: preliminary.**

**Caveat.** The fc2 bias is excluded from the denominator of the MLP-output share.

### O10. Ablating pair 370–391 does not change detection; its logit effects are additive, as expected for copies

**Statement: epoch-200 ablation results.** Ablation means zeroing the GELU output on all tokens.

| Ablated | ΔIoU, VAL [95% CI] | ΔIoU, TEST [95% CI] | Δscore, VAL | Δscore, TEST | Winner token changed |
|---|---|---|---|---|---|
| 370 | −1.0·10⁻⁵ [−2.1; −0.1]·10⁻⁵ | +0.6·10⁻⁵ [−0.2; +1.9]·10⁻⁵ | −2.1·10⁻⁵ | −4.9·10⁻⁵ | 0/1000 |
| 391 | −0.6·10⁻⁵ [−3.6; +2.6]·10⁻⁵ | +2.4·10⁻⁵ [−1.8; +8.9]·10⁻⁵ | +0.1·10⁻⁵ | −0.7·10⁻⁵ | 0/1000 |
| Both | −1.6·10⁻⁵ [−5.4; +2.2]·10⁻⁵ | +3.1·10⁻⁵ [−2.1; +10.7]·10⁻⁵ | −1.7·10⁻⁵ | −6.7·10⁻⁵ | 1/1000 |

- Maximum absolute ΔIoU for any frame after ablating both: 0.00233 on VAL and 0.00796 on TEST (`ABLACJA_BOOTSTRAP_k218750.csv`).

**Comparison with controls.** The pair is compared with 300 control pairs from block 6, 100 in each of three groups: random, similar strength, and most parallel in W2.
- **ΔIoU.** The pair falls at the 20th–34th percentile of controls on VAL and the 75th–77th on TEST.
- **Interaction ratio.** This is the mean |Δlogit| of the pair ablation divided by the larger single-neuron ablation, over 30,000 tokens.
  - The pair's ratio is 1.66.
  - Control medians are 1.07 (random), 1.24 (similar strength) and 1.29 (near-parallel).
  - The pair sits at the 97th, 99th and 86th percentile of these groups respectively.
- **Single-neuron ranks.** Ranked by effect on logits, neurons 370 and 391 are 253rd and 341st of 768.

**Interaction-ratio history** (`ABLACJA_PARY_0_200.csv`): below 1 only at epochs 60, 80 and 90 (0.960, 0.982, 0.926), before the neurons became copies.

| Epoch | 0 | 50.29 | 100 | 110 | 140 | 170 | 190 | 200 |
|---|---|---|---|---|---|---|---|---|
| Interaction ratio | 1.421 | 1.114 | 1.042 | 1.487 | 1.871 | 1.713 | 1.777 | 1.658 |

**Sources.**
- `[P200]\Wyniki\Dane\znoszenie_370_391\ABLACJA_BOOTSTRAP_k218750.csv`, `KONTROLA_k218750.csv` (301 rows: target plus 300 controls), `POJEDYNCZE_k218750.csv` (768 single-neuron ablations), `ABLACJA_PARY_0_200.csv`
- `[P200]\Wyniki\Raport\RAPORT.html` §3.3 and Rys. 7

**QC.**
- In the last block, ablating the pair must shift the residual stream by exactly −(u1+u2). This holds at all 26 checkpoints, with a maximum error of 1.0785·10⁻⁵ (at step 196875); the tolerance is 10⁻⁴ (`[P200]\Wyniki\Dane\znoszenie_370_391\QC.json`).
- Smoke test job 1911914; full job 1911919, 246.9 s (`PROVENANCE.json`).

**Sample size.**
- 1 pair, 1 seed.
- Ablation on 500 VAL + 500 TEST frames at each of 26 checkpoints.
- Controls and single-neuron ablations at epoch 200 only.
- Interaction ratio computed on logits of the 50 diagnostic frames.

**Robustness: moderate** for "no detection effect" (wide frame base, 300 controls); **preliminary** for the copy interpretation.

**Caveats.**
- Zero ablation is a mild intervention for neurons that mostly sit in the negative GELU tail; the report calls it "łagodna interwencja". No mean or resample ablation was done.
- For neuron 370 alone, the VAL confidence interval [−2.05·10⁻⁵; −1.2·10⁻⁶] excludes zero, although its size is negligible.
- Interaction ratios above 1 are common even at initialization (1.42 at step 0) and among random controls (median 1.07).

### O11. Martinelli et al. 2025 ("Mar25l", "flat channels to infinity") is not supported: input weights of opposed pairs are no more similar than random, and input-similar pairs are copies, not channels

**Pre-registered decision (3-seed pilot).**
- H1 (channel present) was accepted in 0/3 seeds; H0 (anti-parallel outputs without a channel) in 3/3 seeds.
- This holds on both the TEST and VAL panels.
- The channel-subpopulation hypothesis H1s was accepted in 0/3 seeds.
- Verdict: "H0_no_channel" (`[MAR]\Wyniki\Dane\eden\redukcja_pilot\HYPOTHESES_AND_SUMMARY.json`, `decisions`).
- The primary cell was block index 5 (0-based, i.e. the last block, "blok 6" in the other experiments), step 65640 (epoch 60), TEST panel.

**Per-seed numbers (seeds 3047 / 4047 / 5047), last block, step 65640.**

| Quantity | 3047 | 4047 | 5047 | Pre-set criterion |
|---|---|---|---|---|
| Opposed pairs (c_out < −0.5) | 655 | 641 | 687 | – |
| Neurons with an opponent | 422 | 388 | 415 | – |
| Median input similarity c_in_eff, opposed pairs | 0.0778 | 0.0779 | 0.0667 | H1(a) needs ≥ 0.9 |
| Median c_in_eff, random pairs | 0.0781 | 0.0632 | 0.0590 | – |
| 95th percentile of c_in_eff, random pairs | 0.2723 | 0.2590 | 0.2639 | H0: opposed median ≤ this |
| Near-channel pairs (c_in_eff > 0.9 and c_out < −0.5) | 0 | 0 | 0 | H0: fraction < 1% |
| Pairs meeting the literal Mar25l criteria | 0 | 0 | 0 | – |
| Median κ, opposed pairs (TEST) | 0.9749 | 0.9959 | 0.9999 | H1(c) needs ≤ 0.3 |
| Median GLU approximation error, opposed pairs (TEST) | 0.6352 | 0.6120 | 0.6582 | H1(d) needs ≤ 0.2 |
| Median GLU approximation error, random pairs | 0.690 | 0.683 | 0.728 | – |
| ρ (norm of pair sum / sum of norms), opposed (TEST) | 0.8016 | 0.8414 | 0.8476 | – |
| ρ, random | 0.8618 | 0.8722 | 0.8740 | – |
| "Copy" pairs (c_in_eff > 0.9 and c_out > 0.5) | 0 | 1 | 0 | – |
| Input-similar pairs (c_in_eff > 0.5): opposed vs aligned (c_out > 0.5) | 17 vs 441 | 8 vs 382 | 6 vs 336 | – |
| Spearman correlation of c_in_eff with c_out | +0.0515 | +0.0490 | +0.0587 | – |
| Largest pair parameter norm | 3.007 | 3.075 | 2.926 | Literal criterion > 10³ |
| Mean channel-update fraction f_ch, opposed pairs | 0.0190 | 0.0242 | 0.0179 | Literal criterion > 0.9 |
| f_ch, random pairs | 0.0230 | 0.0227 | 0.0197 | Uniform expectation 2/770 = 0.002597 |

- Literal Mar25l pairs are 0 in every one of the 6 blocks (`literal_all_tracked_by_block`).
- The one copy pair (seed 4047) has c_in_eff 0.954, κ 1.81 and GLU error 0.059.
- At step 10000, the channel-update fraction for opposed pairs is somewhat higher: 0.1049 / 0.0975 / 0.0854, against 0.0140 / 0.0166 / 0.0161 for random pairs.
- Seed×film bootstrap for opposed pairs: κ on TEST 0.9857 [0.9683; 0.9961]; GLU error 0.6213 [0.5961; 0.6496]; co-activation 0.521 [0.446; 0.611], against 0.591 [0.519; 0.661] for random pairs.
- Opposed pairs per block (0–5) at step 65640:

  | Seed | b0 | b1 | b2 | b3 | b4 | b5 (last) |
  |---|---|---|---|---|---|---|
  | 3047 | 39 | 6 | 32 | 56 | 201 | 655 |
  | 4047 | 45 | 0 | 3 | 19 | 390 | 641 |
  | 5047 | 43 | 6 | 21 | 29 | 234 | 687 |

**Sources.**
- `[MAR]\Wyniki\Dane\eden\redukcja_pilot\HYPOTHESES_AND_SUMMARY.json`
- `POPULATION.csv` (quadrant counts `q_cin0.5_*`, `q_cin0.9_*`; `spearman_cin_eff_cout`; `max_pair_param_norm`)
- `SETS_WEIGHTS.csv` (`f_channel_mean`)
- `POPULATION_DATA.csv`
- Pre-registered hypotheses: `[MAR]\Opis\OPIS_EKSPERYMENTU.txt` §4

**QC.**
- The local and Eden reductions agree. The worst column difference is 2.22·10⁻¹⁶ in SETS_WEIGHTS and 8.3·10⁻⁵ in co-activation. Decisions and timing are equal (`[MAR]\Wyniki\Dane\eden\POROWNANIE_EDEN_LOKALNIE.json`).
- 10 unit tests and a local smoke test passed (`OPIS` §5).
- The per-seed count of neurons with an opponent (422 for seed 3047) matches the 24-seed campaign (`[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`, per-seed table: 3047 → 422).

**Sample size.**
- 3 seeds, 12 checkpoints, epochs 0–60 only.
- Weight geometry covers all 294,528 pairs in every block.
- Token measures: 10 TEST + 10 VAL frames (6,000 tokens per panel).
- The full 24-seed campaign was prepared but not submitted (`[MAR]\Opis\OPIS_EKSPERYMENTU.txt` §8a; `experiments_index.md`).

**Robustness: preliminary.** The decision is unanimous in 3/3 seeds and has a wide margin, but this is a pilot.

**Caveats.**
- With weight decay 0.05, the literal Mar25l criteria (norms → ∞) were expected a priori to fail (`OPIS` §2, §4).
- The pilot window ends at epoch 60, so it does not cover the late copy formation of pair 370–391 (epochs 100–200).
- No training without weight decay was run.
- Pairs selected by final weights (opposed_final) are retrospective.

### O12. W2-geometric opposition (cos < −0.5, >120°) grows from 0 to about half the last-MLP neurons by epoch 60 — a different and much larger set than the ≥150° beam pairs

**Statement.**
- **24 seeds, last MLP.** At initialization no neuron has a W2 partner with cos < −0.5. At step 65640, 387–474 of 768 neurons have one (mean 424.04, i.e. 55.21%).
  - This corresponds to 739.83 unordered pairs on average, out of 294,528 (0.251%) (`[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`, "Geometria wag a faktyczne wkłady").
  - The mean number of neurons with a partner rises: 0 at steps 0 and 300, 92.17 at step 10000, 210.50 at 16000, 424.04 at the end (`[PRUNE]\RAPORT_KONCOWY_PL.md`).
  - Mean opposed pairs: 99.88 at step 10000, 307.58 at 16000, 780.96 at 55000, 739.83 at 65640 (`[PRUNE]\analysis\DYNAMICS_INTERPRETATION_PL.md` L25–36).
- **3-seed cohort (seeds 47 / 1047 / 2047).** 408, 433 and 380 of 768 neurons at the end (53.125%, 56.380%, 49.479%), with 844, 746 and 658 opposed pairs. The first nonzero values appear at step 6000, 6000 and 3000; these are "not exact onset moments" (`[S3]\RAPORT_WYNIKOW.md`).
- **Dense single run** (806 checkpoints).
  - 408/768 = 53.125% at the end, with a maximum of 56.771% at step 60100.
  - First positive value at step 3200 (3 of 768 neurons).
  - The fastest growth is in epochs 5–15, at 2.400 percentage points per epoch.
  - Threshold sensitivity at the end: 87.37% at cos < −0.4, 53.125% at < −0.5, 17.188% at < −0.6.
  - Source: `[WSP]\przeciwne_reflektory\RAPORT.md`.
- **Mar25l pilot (3 seeds).** The last block holds by far the most opposed pairs (see the per-block table in O11).

**Sample size.** 24 seeds (main), 3 seeds, and 1 dense run. All cover epochs 0–60.

**Robustness: strong** across seeds.

**Caveats.**
- Cos < −0.5 is an angle >120°, not 180°. In the files' own words: "Posiadanie partnera nie oznacza jednoczesnej aktywności ani roli w detekcji".
- By O3 this family is disjoint from the same-token ≥150° beam pairs, because those require a W2 angle ≤30°.
- The dense run's seed is not stated. Its end values (408/768, 844 pairs) are identical to seed 47 in [S3], so it is probably seed 47 (inference). Seed 47 had a different resumption history: it was resumed after epoch 21 ([S3]; [WSP]).

### O13. Weight geometry is not functional cancellation

**Statement: overlap between the pair families.**
- Of the 1,327 functional pairs across 24 models, 1,257 fail W2 cos < −0.5; only 70 meet both criteria, and 298 have a non-negative W2 cosine (`[AUDYT]\RAPORT_PL.md`).
- Of the 479,248 pairs that are ≥150° apart in the 3D sphere projection (24 final models), only 10,828 have W2 cos < −0.5 in 192D, and 78,327 have a non-negative 192D cosine (`[SFERA]\RAPORT_PL.md`).

**Catalogue means for the last MLP, final checkpoint, 24 seeds** (`[PRUNE]\RAPORT_KONCOWY_PL.md` table; `[PRUNE]\analysis\PAIR_MECHANISM_PL.md` L33–35):

| Family | W2 cosine | Signed z·v cosine | Co-activation | Edge-change cosine | Edge q |
|---|---|---|---|---|---|
| Functional | −0.152 | −0.135 | 48.14% | −0.640 | 0.568 |
| Geometric | −0.549 | −0.446 | 22.57% | −0.363 | 0.804 |

- Edge q is the compensation index on the four box edges; smaller q means stronger compensation.

**Co-activation of geometric pairs** (threshold 0.05 × the neuron's 95th percentile): both neurons are active together in 42.38% of cases inside the box and 69.56% in far background. The file says this is not tool-specific (`[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`, "Koaktywacja geometrycznych par").

**The same 16 fast-changing pairs followed from step 0 to 65640** (50-frame panel):
- W2 cosine goes from 0.02775 to −0.56027.
- Signed cosine goes from −0.00185 to −0.47198.
- Co-activation falls from 0.86167 to 0.17729.
- q goes from 0.79691 to 0.82046.
- In short, the weights become more opposed while co-activation falls and cancellation does not strengthen (`[PRUNE]\analysis\DYNAMICS_INTERPRETATION_PL.md` L103–106; `[PRUNE]\RAPORT_KONCOWY_PL.md`: "koaktywacja … maleje z około 86,17% do 17,73%").

**Sample size.** 24 seeds; 1,327 functional pairs. Pairs are not independent units.

**Robustness: strong** that geometry and function dissociate.

**Caveat.** The families have different selection rules, so the comparisons between them are descriptive.

### O14. Functional compensation exists and is output-specific, but switching pairs off or restoring them does not improve IoU; removal lowers score

**Pre-planned contrasts, 24 seeds** (5000 seed×film bootstrap resamples; 97.5% intervals, Bonferroni-corrected).
- **P1 = 0.159633 [0.131831; 0.190173], positive in 24/24 seeds.** P1 is the paired difference q_control − q_pair, for pairs selected at step 33000 and evaluated at 65640.
- **P2 = 0.521615 [0.479007; 0.562913], positive in 24/24.** P2 compares the change in advantage from initialization to the end for late-selected pairs against pairs selected at initialization.
- Most of P2 comes from initialization-selected pairs losing their effect: their advantage goes from 0.440336 to −0.002540. Pairs selected at step 33000 go from 0.011457 to 0.090196.
- Source: `[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`.

**Removing whole families from the last MLP, clean-winner token** (`[PRUNE]\RAPORT_KONCOWY_PL.md`; `[PRUNE]\analysis\PRUNING_INTERPRETATION_PL.md` L33–35):

| Family | ΔIoU [95% CI] | Δscore [95% CI] | Seeds with lower score |
|---|---|---|---|
| Functional | −0.0000149 [−0.000278; +0.000264] | −0.01523 [−0.01772; −0.01293] | 24/24 |
| Geometric | −0.0008653 [−0.003279; +0.001906] | −0.07550 [−0.08554; −0.06595] | 24/24 |
| Fast | −0.0008153 [−0.003137; +0.001886] | −0.07292 [−0.08248; −0.06374] | 24/24 |

- Cumulative pruning curves: no point beats the clean model on both IoU and score for the functional family in any seed (0/24). Such points exist in 2/24 seeds for geometric and 8/24 for fast, and the files describe these as selected post hoc.
- With a fixed GT-anchored token, IoU improves slightly: +0.000423 [+0.000113; +0.000761] for functional and +0.002875 for geometric, while the anchor's score falls. In the files' words: "Lepszy wynik tej kotwicy nie jest automatycznie lepszym wynikiem całego detektora."

**Restoring pairs after removing the whole union** (`[RESTORE]\RAPORT_KONCOWY.md` L21–41):
- IoU / score: clean 0.7442270 / 0.8465622; all-off 0.7442121 / 0.8313294; off + one pair 0.7442107 / 0.8321347.
- Restoring one pair raises score by 0.000805 in 24/24 seeds.
- IoU moves in either direction: 12/24 positive.
- Target minus control: IoU 0.00000226 [−0.00000583; 0.00001001]; score 0.00014065 [0.00008232; 0.00020050], 19/24 seeds.

**Compensation already exists at initialization.**
- Dense run: q of selected compensating pairs is 0.1187 at step 0 and 0.0857 at step 65640; "Silna kompensacja występuje już w checkpoincie 0" (`[WSP]\RAPORT.md`).
- 3 seeds: adaptive q_gain is about 0.558 at initialization and about 0.585 at the end. The end-minus-initialization change for frozen pairs is 0.09407 [−0.07536; 0.20821] (`[S3]\RAPORT_WYNIKOW.md`).

**Sample size.**
- 24 seeds, last MLP, final checkpoint.
- 500 VAL frames for selection, 500 TEST frames for evaluation.

**Robustness: strong** (24/24 seeds; confidence intervals).

**Caveats.**
- TEST had already been viewed.
- One architecture and one dataset.
- Score is a sigmoid output and is not calibrated.
- Pairs were selected for an effect on box edges, which partly guarantees opposed edge effects ([RESTORE] `WNIOSKI_PUBLIKACYJNE_I_WYBOR_VENUE.md` L30).

### O15. Opposite effects on box edges coexist with both neurons supporting the detection logit

**Statement.**
- **Restoration audit.** Restoring A and restoring B move the box edges in opposite directions in 87.53% of cases (cosine −0.6471).
  - q is 0.5706317273 for all target pairs and 0.56828 for matched target pairs, against 0.82147 for controls.
  - Both neurons raise the logit in 58.96% of cases, and both conditions hold together in 54.47% (controls: 22.1553%).
  - The non-additive residual is 0.0001185 px RMS, against single effects of about 0.026 px, so the effects are almost additive.
  - Sources: `[RESTORE]\RAPORT_KONCOWY.md` L72–78; `PAIR_RESTORATION_MECHANICS.md` L11–21.
- **First functional pair, ablated from the clean model.**
  - Edge cosine < −0.5 in 24/24 seeds.
  - Δscore after ablating both is −0.0010014, negative in 24/24, CI [−0.0012534; −0.0007630].
  - ΔIoU is +0.00000664, CI [−0.00001374; +0.00002932], with 14 seeds positive and 10 negative.
  - Source: `[PRUNE]\analysis\PAIR_MECHANISM_PL.md` L51–57.

**Sample size.** 24 seeds; 662,056 valid pair×image cells (restoration audit).

**Robustness: strong** within the last MLP of this model family.

**Caveat.** The pair selection rule favours opposed edge effects.

### O16. Relative size of a pair's contribution: about 2% of the MLP output norm

**Statement.**
- The resultant of the first functional pair is 2.043% of the full MLP output norm, including the bias (per-model range 1.380–3.098%).
- Across the 1,200 model×image observations: median 1.899%, P5–P95 0.592–3.557%, maximum 20.521%.
- Switching the pair off rotates the MLP output by 1.096° on average.
- Latent cancellation ‖sum‖/(‖u_A‖+‖u_B‖) is 0.706.
- Over all 1,327 functional pairs: 1.682% per pair.
- Only 17 observations (3 models) are simultaneously co-active, signed-opposed (cos < −0.5) and balanced (norm ratio ≤ 1.25).
- Sources: `[REL]\WNIOSKI_PL.md`; `[REL]\RAPORT_PL.md`.
- This is consistent with pair 370–391 (O9): median share 1.5% over all tokens and 0.32% on active opposite-sign tokens.

**Sample size.** 24 seeds, 50 frames, final checkpoint only.

**Robustness: moderate.**

**Caveat.** In the files' words, a share of the norm is not causal responsibility: "procent normy nie jest procentem odpowiedzialności przyczynowej".

### O17. "Fading" (wygaszanie) of beams is mostly loss of relative importance, not true decay

**Statement (seed 3047, step 0 → 55000, 50 frames).**
- **Fewer beams above 5% of the token maximum**, per token (of 768): 554.8 → 113.7 in block 1, and 588.7 → 64.4 in block 4.
- **N_eff also falls**: 334.2 → 159.9 in block 1 and 353.2 → 87.1 in block 4.
- **Exact zeros stay rare** at step 55000: 0.01–3.31 per token.
- **More beams above a fixed absolute threshold.** Beams with strength > 0.05 per token: 236.7 → 539.6 (block 1), 264.9 → 429.2 (block 4), 255.0 → 444.2 (block 6).
- **Total strength and median strength rise.** Sum of beam strengths per token: 53.28 → 123.71 in block 4. Median strength: 0.0440 → 0.0654 in block 4.
- **Only the weakest decile shrinks**, in blocks 2–6: the 10th percentile goes 0.0105 → 0.0015 in block 4. Block 1 is the exception (0.0103 → 0.0135).

**Beams that drop below the 5% threshold** (above at the start, below at the end):

| Block | Truly weaker | ≥10× weaker |
|---|---|---|
| 1 | 47.5% | 8.8% |
| 2 | 63.7% | 23.9% |
| 3 | 62.8% | 23.8% |
| 4 | 59.3% | 21.8% |
| 5 | 64.1% | 23.3% |
| 6 | 60.4% | 20.7% |

- Beams that stay above the threshold strengthen 4–13×. In block 4 their mean strength goes from 0.090 to 1.173.

**Sources.**
- `[ATLAS]\Wyniki\Raport\RAPORT.html` §1
- `[ATLAS]\Wyniki\Dane\SUMMARY.json` (`blocks[*].activity`, `endpoint_transitions`)
- `[ATLAS]\Wyniki\Krótki_Wniosek\WNIOSEK.txt` points 1–2

**Sample size.** 1 seed, 11 checkpoints (0–55000), 50 frames. Confidence intervals come from a 25-film bootstrap (2000 resamples) and are conditional on one model.

**Robustness: moderate.**

**Caveats.**
- Not measured after epoch 50.
- The relative threshold rises as the dominant beams grow.
- "Nie dowodzi, że słabsze wiązki można bezkarnie usunąć" (it does not show that weaker beams can be removed without cost).

### O18. The angle distribution between persistent beams widens in both directions

**Statement.**
- The mean angle within the same token falls only slightly, for example 89.98° → 87.28° in block 4 (paired Δ −2.70°, CI [−2.93; −2.45]).
- The mean distance from 90° rises: 3.31° → 8.79° in block 4, CI on Δ [5.28; 5.67]; 3.30° → 5.21° in block 1.
- Both directions of change are common: 46.8–50.8% of pairs decrease by more than 1° and 41.0–44.8% increase by more than 1°.
- The share of persistent pair-token instances at ≥150° at step 55000 is tiny: 3.57·10⁻⁶ in block 4.
- Sources: `[ATLAS]\Wyniki\Raport\RAPORT.html` §2; `[ATLAS]\Wyniki\Dane\SUMMARY.json` (`angles.persistent`).

**Sample size.** 1 seed, 50 frames, 25-film bootstrap.

**Robustness: moderate.**

**Caveat.** The persistent cohort is defined by the 5% threshold at both endpoints. Threshold variants (1%, 10%) keep the direction of the change but alter its size.

### O19. Quality-control checks that passed

- **Angle computation**: matches direct dot products within 2.842·10⁻¹⁴ on synthetic and real data, over all 294,528 pairs (`[ATLAS]\...\SYNTHETIC_QC.json`; `[ATLAS]\Wyniki\Dane\SUMMARY.json` qc).
- **Re-run at step 65640**: recomputed from the continuation checkpoint, it is bitwise identical to the parent in weights, sign masks, counts and angles (`[P200]\Wyniki\Dane\pary_epoki_60_200\QC_65640_VS_PARENT.json`).
- **VAL detection** at 0–65640 is identical to the parent (O7).
- **Ablation residual check** passes at 26/26 checkpoints (O10).
- **Continuation training**: a two-step PREFLIGHT confirmed bitwise agreement of the next batch, model, optimizer, sampler queue and RNG. The 15 checkpoints were SHA256-verified (`[K200]\Opis\ODTWARZANIE_I_CHECKPOINTY.txt`; `[K200]\Wyniki\Dane\CONFIG_QC.json`; `experiments_index.md`).
- **Atlas QA**: `[ATLAS]\Wyniki\Dane\FINAL_QA.json` and `PLOT_QC.json` both have status PASS.
- **Mar25l**: local and Eden results are identical (O11).

---

## Definitions & setup

**Terminology.** In older documents, "reflektor" means a neuron, "wiązka" (beam) means one neuron's contribution on one token, and "cząsteczka" (particle) means a token ([AUDYT], [WSP]).

**Beam (wiązka).**
- u[t,c] = z[t,c] · W2[:,c] ∈ ℝ¹⁹², where z[t,c] = GELU(fc1(LN(x)))[t,c] is the activation of hidden neuron c on token t.
- Strength (siła): a[t,c] = |z[t,c]| · ‖W2[:,c]‖.
- There are 768 beams per token per block. The fc2 bias is a separate term, not assigned to any beam.
- Angles are computed in full 192D, keeping the activation sign; there is no 3D projection. A negative GELU flips the beam's direction.
- Source: `[ATLAS]\Opis\OPIS_EKSPERYMENTU.txt`.

**Activity thresholds.**
- A beam is "significant" (znacząca) if its strength exceeds 5% of the strongest beam on the same token. Controls use relative thresholds of 1% and 10% and absolute thresholds of 0.01, 0.05 and 0.1.
- Beams with norm ≤ 10⁻¹² count as zero; their angle is undefined.
- Effective number of beams: N_eff = (Σa)² / Σa².

**Same-token pair angle and the counting rule.**
- For neurons c ≠ d in the same block: the signed angle between u[t,c] and u[t,d] on the same image and token.
- A pair of neuron IDs counts once per block and checkpoint if, on at least one of the 30,000 tokens (50 frames × 600 tokens), its angle reaches the threshold.
- Thresholds: 150°, 160°, 170°, 175°, 179° and 179.999999°.
- Also stored for each threshold:
  - the total number of pair-token occurrences;
  - an "active05" variant where both beams exceed 5% of the token maximum;
  - W2-only angles α;
  - a frame-independent bound max(α, 180°−α).
- Sources: `[ATLAS]\Opis\LICZENIE_PAR_0_BEST_FINAL.txt`; `[P200]\Opis\OPIS_EKSPERYMENTU.txt`; code `[ATLAS]\Kod_eksperymentu\count_opposite_pairs.py` (SHA256 392e943a…, reused by [P200] after an SHA check).

**Pair metrics (pair 370–391)** (`[P200]\Kod_eksperymentu\konfiguracja_znoszenie.json`):
- Strength ratio = min(s1, s2) / max(s1, s2).
- κ_token = ‖u_a+u_b‖² / (‖u_a‖²+‖u_b‖²). 2 means aligned and equal, 1 orthogonal, 0 perfect cancellation.
- Cancellation = 1 − ‖u_a+u_b‖ / (s_a+s_b).
- Share of MLP output = ‖u_a+u_b‖ / ‖z·W2ᵀ‖ on the same token, with the fc2 bias excluded.
- Effective bias b̂ = b1 + W1·β (the LayerNorm shift folded into the bias), as in `[MAR]\Opis\OPIS_EKSPERYMENTU.txt` §3A.
- Ablation = zeroing the GELU outputs of the given neurons in block 6 on all tokens.
- Interaction ratio = mean |Δlogit| for the pair / the larger of the two single-neuron effects. Below 1 means the two partly cancel.

**Mar25l metrics** (`[MAR]\Opis\OPIS_EKSPERYMENTU.txt` §3):
- c_out = cos between the two fc2 columns.
- c_in_eff = cos between the effective input weights after folding in the LayerNorm affine (centred W1⊙γ together with b̂/√D). c_in_raw is the literal version.
- Population κ_ij = Σ_t‖u_i+u_j‖² / Σ_t(‖u_i‖²+‖u_j‖²) = 1 + c_g·c_out·β.
- ρ = Σ‖y‖ / Σ(‖u_i‖+‖u_j‖).
- GLU approximation error; channel-update fraction f_ch (random expectation 2/770).
- Literal criteria: pair parameter norm > 10³, input cosine distance < 10⁻³, c_out < −0.9, f_ch > 0.9.

**Older-campaign definitions.**
- **Geometric opposition**: cos(W2 columns) < −0.5, i.e. an angle >120°.
- **Functional pairs**: signed JVP on the 4 box edges with cos < −0.5 and norm ratio ≤ 1.25 on VAL, ranked by real A/B/AB ablations.
- **Fast pairs**: opposed in the final geometry with an initial cos ≥ −0.1.
- **q** = ‖Δedges_AB‖ / (‖Δedges_A‖ + ‖Δedges_B‖) on the 4 box-edge coordinates. Smaller means stronger compensation.
- Sources: [PRUNE], [FUN24], [AUDYT].

**Detection.**
- Winner token = argmax logit per frame.
- IoU = maximum IoU against the GT boxes.
- Score (pewność) = sigmoid of the winner's logit; it is not calibrated.
- Source: `[P200]\Wyniki\Raport\RAPORT.html` §2.3.

**Model: MalyViT** (`[MAR]\Opis\OPIS_EKSPERYMENTU.txt` §2; `[FUN24]\RAPORT_WYNIKOW_24SEEDY.md`).
- Convolutional stem, then 6 timm blocks (pre-LN, dim 192, 3 heads, MLP 768 with erf GELU), then a final LayerNorm and a dense head.
- The head outputs, per token, a centre-map logit and 4 distances to the box edges (l, t, r, b).
- Input 480×320, giving 600 tokens.
- Loss: l_heat + 2·l_box (`[K200]\Wyniki\Dane\logi\train_1907130.log`, model.py:114).
- Task: detecting the tip of a surgical tool in cataract-surgery video (the cataract_lmm `3_Object_Tracking` annotations).

**Training.**
- AdamW, lr 5·10⁻⁴, 300 warmup steps, then **constant lr** (no decay).
- Betas (0.9, 0.999).
- Weight decay 0.05 on weights of rank ≥2 (including fc1 and fc2); 0 on biases and norms.
- Gradient clipping 1.0; bf16 autocast; batch 32.
- Augmentation: horizontal flip and contrast/brightness changes.
- 35,000 TRAIN images, split by film.
- Epoch = step × 32 / 35,000. Step 55,000 = 50.2857 epochs; step 65,640 = 60.0137 epochs.
- Sources: `[K200]\Opis\OPIS_EKSPERYMENTU.txt`, `[K200]\Kod_eksperymentu\config.json`, `[K400]\Opis\OPIS_EKSPERYMENTU.txt`.

**Seeds and runs.**
- **Seed 3047** is one of the 24-seed cohort: seeds 3047 + 1000·i, i = 0…23 (3047–26047), trained 65,640 steps on H100 in `kompensacja_funkcjonalna_dgx2_20261017`.
  - Epochs 0–60 are measured on its saved weights at steps 0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640.
  - The step-55000 weights have SHA256 9578761b…; this was the VAL-best checkpoint at that time.
- **Continuation to epoch 200** ([K200]):
  - full restart from step 65640, restoring the model, AdamW moments and counters, the sampler queue and all RNG states;
  - warmup not reset; global step preserved;
  - 153,110 extra steps to step 218,750; Eden job 1907130, 4718 s;
  - checkpoints every 10 epochs at step ceil(epoch·35000/32).
- **Continuation to epoch 400** ([K400]): restarts from the step-218750 checkpoint (SHA256 bfa43a3e…) towards step 437,500 (job 1911922). It was still training on 2026-10-21 and has not been analysed.
- **Older cohort**: seeds 47 / 1047 / 2047 ([S3]).
- **Dense run**: one run with 806 checkpoints ([WSP]); the seed is not stated.

**Measurement sets.**
- **VAL**: 500 frames from 25 films (466 with a tool, 34 empty). Used for checkpoint selection and pair discovery.
- **TEST**: 500 frames from 25 other films (472 with a tool, 28 empty). Used for reporting, but already viewed in earlier campaigns.
- **DIAG50**: 50 fixed TEST frames, 2 per film from 25 films, all with a tool. Used for beam and pair counts.
- **Mar25l pilot**: 10 TEST + 10 VAL frames.
- All inference in eval mode, FP32, TF32 off, deterministic algorithms ([P200] OPIS).

---

## Gaps

### What has not been shown yet

1. **Replication across seeds of the beam-level (same-token ≥150°) results.**
   - All 0–200-epoch beam counts, the decline after epoch 110, the ≥170° pair and the copy story rest on seed 3047 only.
   - The 24-seed checkpoints (epochs 0–60) exist, but same-token pair counts were never run on them; only W2 geometry was.
   - No second architecture or dataset.
2. **Very late training.** The epoch 200–400 checkpoints are not analysed ([K400] still training). It is open whether the 370–391 angle keeps approaching 180° and whether the count keeps falling.
3. **A causal role for training settings.**
   - No run without weight decay.
   - No lr decay schedule; lr is constant after warmup.
   - No regularizer penalising opposed or parallel pairs.
   - The weight-decay explanation for copies and the cause of the post-epoch-110 decline are interpretations only ("nie była osobno testowana").
4. **Weight histories for pairs other than 370–391.** Only that pair has W1 cosine, norm and bias histories. Whether the other 14 epoch-200 pairs (W2 angles 20.6–29.9°) are also converging copies is untested.
5. **Ablations.**
   - The ≥150° pairs were never ablated jointly or block by block, including blocks 1–5.
   - Only zero ablation was used for 370–391; no mean, resample or dose ablation.
   - The interaction ratio is computed on 50-frame logits, not on IoU.
   - Controls and single-neuron ablations were run only at epoch 200.
6. **Detection metrics.**
   - No calibration metrics (ECE, reliability curves) and no AP/mAP over epochs 0–200.
   - Only 34 and 28 empty frames.
   - No fresh, unviewed TEST set: the current TEST was used in earlier hypothesis generation.
7. **Strength, N_eff and angle-distribution analysis after epoch 50.** The atlas stops at step 55,000.
8. **Mar25l.**
   - The full 24-seed campaign was never submitted.
   - The pilot covers epochs 0–60 with 10+10 frames, so the late copy formation (epochs 100–200) was not tested against the channel/copy criteria.
9. **Null model for pair counts** (for example random matrices with a matched spectrum, or rotation controls). This was suggested in [LIT] §6 and not done. The only baseline is initialization: maximum angle 110.23° at step 0.
10. **Statistics on the trajectory itself.** There are no confidence intervals on pair counts, no trend test, and no correction for the many thresholds and blocks. Onset is known only as an interval because checkpoints are sparse: 12 points over epochs 0–60.
11. **Meaning of the opposite-sign tokens.** Only the share inside the GT box was measured (1.3% vs 4.3%). Location relative to tool edges and information content were not.
12. **Older family names.** The family names "opposite_beams_all_blocks_best", "near180" and "cooling" were not found locally. I checked folder names in experiments\ and outputs\ and grepped *.md and *.txt files. "beam_survival_angles_training_20261020" exists only as an Eden path; its results were copied into [ATLAS]\Wyniki\Dane.

### Known inconsistencies and caveats across experiments

- **"Opposite" means four different things.**

  | Definition | Scale | Sources |
  |---|---|---|
  | W2 cos < −0.5 (>120°) | about 424 of 768 neurons and 739.83 pairs per seed in the last MLP | [FUN24], [PRUNE] |
  | Same-token signed beam angle ≥150° | 0–30 pairs over all 6 blocks | [ATLAS], [P200] |
  | Functional (JVP on box edges) | 1,327 pairs | [AUDYT] |
  | 3D sphere angle ≥150° | 479,248 pairs | [SFERA] |

  The ≥150° beam pairs have W2 angles ≤ 29.98° (derived), so they are disjoint from the W2-anti-parallel family. Only 70 of 1,327 functional pairs are W2-opposed. The six literature-assessed observations refer to the W2/functional definitions, not to the ≥150° beams.
- **Sphere pilot vs the beam atlas.** The sphere pilot found no signed 150° pair-token in 192D (2 frames, 19,575 3D-selected pairs, seed 3047, step 65640). The atlas finds 9 block-6 pairs ≥150° at the same step. Both are reconciled by O3 (inference): the 3D-selected pairs are near-anti-parallel in 3D and have 192D W2 angles of at most 137.87° in block 6 at step 65640, so they cannot reach 150°.
- **"Best checkpoint."** Step 55000 is VAL-best only within epochs 0–60 (0.7619). Over epochs 0–200 the VAL maximum is 0.7754 at epoch 130 (O7).
- **Block indexing.** [MAR] uses 0-based blocks: primary_block 5 = the last block = "blok 6" in [ATLAS] and [P200].
- **Mar25l summary metadata.** `HYPOTHESES_AND_SUMMARY.json` has `"smoke": true`. Its `crossed_bootstrap` entries list `"films": 25`, while the pilot used 10 films per panel (`finite_cells: 30` = 3 seeds × 10). With n = 3, the seed-bootstrap confidence intervals are close to the min–max range.
- **Dense run seed.** The 15-October dense run does not state its seed. It matches seed 47 (408/768, 844 pairs), and seed 47 had a different resumption history ([S3]; [WSP]).
- **Transfer of frozen pairs.** In the 15-October run, frozen pairs did not transfer (q 0.887 vs 0.898 for controls; [WSP]\RAPORT.md). Later campaigns found transfer (3-seed q_gain 0.26033 [0.16734; 0.35208]; 24-seed P1 0.159633). The protocols differ.
- **q is not comparable across campaigns**:
  - 0.0841 = medians, adaptive, local ablation ([WSP]);
  - 0.55151 = means, frozen, global ablation ([S3]);
  - restore_all computes q relative to the all-off state and drops it only for a zero denominator;
  - pruning_dynamics computes q relative to the clean model and drops it when the denominator is ≤ 0.001 px.
- **Mixed columns in the restore report.** `[RESTORE]\RAPORT_KONCOWY.md` mixes columns: 87.53% and cos −0.6471 come from "all target pairs", while q 0.56828 comes from "matched only". The all-pairs q is 0.5706317273.
- **Small, minor mismatches.**
  - Pair 370–391: the report summarizes the confidence intervals as including zero, but the VAL interval for neuron 370 alone excludes zero (magnitude about 10⁻⁵).
  - Active05 unique counts are 23 vs 24 (epoch 60) and 14 vs 15 (epoch 200), because one pair has opposite-sign tokens but no active ones.
- **Earlier claims corrected by the files themselves.**
  - "D=C aliasuje" was corrected to C = clean, D_A = A, D_B = B (`[SFERA]\ERRATA_PILOT_V2.md`).
  - An earlier statement that "gasiliśmy całe reflektory" (we switched off whole reflectors) was narrowed in `[AUDYT]\AUDYT_ZAKRESU_WIAZEK_PL.md`. Local A/B/AB beam zeroing was done only on the 50-frame diagnostic panel. The main 24-seed contrasts used global neuron masks.

### The six observations in the literature notes

Assessments as recorded in [LIT] (Undermind "Prior art check", 62 papers). All six concern the older W2/functional campaigns: 24 seeds, last MLP, epochs 0–60.

| # | Observation in [LIT] | Assessment | Closest prior work listed | Corresponds to |
|---|---|---|---|---|
| 1 | Neurons with an anti-parallel partner grow 0 → 424/768; partners turn over at the end | partly anticipated (częściowo antycypowane) | Mar25l, Kwo05b; condensation Xu25ad, Zho21h | O12, O5 (W2 part) |
| 2 | Opposition grows after the main quality gain; no correlation after detrending | probably new (prawdopodobnie nowe) | none; methods Ran23c, Saf26 | O6 |
| 3 | Weight geometry ≠ functional cancellation | partly anticipated | Lan22, Fak24, Ahm26c | O13 |
| 4 | Opposite edge effects with joint logit support | anticipated at task level, new at pair level | Wu19d, Zha23bf, Hut25 | O15 |
| 5 | Removing pairs does not improve the detector | partly anticipated | Cas19e, Yu22f, Mor18c | O14 |
| 6 | Opposition rises while co-activation falls (86% → 18%) | partly anticipated | Mar25l, Pat25d, Mcg23, Rus24b, Ahm26c | O13 |

The literature notes are dated 2026-10-20 and were written before the 0–200-epoch results. None of the following has been assessed against the literature:
- the same-token ≥150° beam counts and their peak-and-decline (O1, O2, O4);
- the near-parallel-W2 plus opposite-GELU-sign mechanism (O3);
- the copy pair 370–391 (O8–O10);
- the Mar25l pilot verdict (O11);
- the confidence-without-IoU trend (O7);
- the relative-importance view of beam fading (O17, O18).

The notes proposed Mar25l channels as the ready mechanistic hypothesis, and the pilot (O11) did not support it.

---

## Figures that best illustrate each observation

- **O1, O2** (counts and maximum angle, 0–200):
  - `[P200]\Wyniki\Wykresy\01_pary_przeciwne_0_200_progi.png`, `02_pary_przeciwne_bloki.png`, `03_maksymalny_kat_bloki.png`
  - for 0–60: `[ATLAS]\Wyniki\Wykresy\PARY_PRZECIWNE_PRZEBIEG_EPOKI_BLOKI.png` and `PARY_PRZECIWNE_0_55000_65640.png`
- **O4** (occurrences on tokens): `[P200]\Wyniki\Wykresy\04_wystapienia_par_tokeny.png`
- **O3** (W2 weights and signs): `[ATLAS]\Wyniki\Wykresy\16_wagi_i_znaki.png` and `17_rzadkie_katy.png` (signs, rare large angles)
- **O6, O7** (detection vs time):
  - `[P200]\Wyniki\Wykresy\08_iou_i_pewnosc_0_200.png` and `09_iou_i_pewnosc_od_40_epoki.png`
  - `[ATLAS]\Wyniki\Wykresy\19_iou_score.png`
  - multi-seed timing: `[PRUNE]\figures_20261019\trajectories_gt_anchor.png` and `correlations_gt_anchor_difference.png`
  - dense run: `[WSP]\przeciwne_reflektory\przebieg_w_czasie.png` and `korelacje_po_trendzie.png`
- **O8** (pair 370–391 weight history): `[P200]\Wyniki\Wykresy\05_para_370_391_historia.png`
- **O9** (pair 370–391 tokens at epoch 200): `[P200]\Wyniki\Wykresy\06_para_370_391_tokeny_epoka200.png`
- **O10** (ablation vs 300 controls): `[P200]\Wyniki\Wykresy\07_wylaczenie_pary_kontrola.png`
- **O11** (Mar25l): no figures exist; `[MAR]\Wyniki\Wykresy`, `Raport` and `Krótki_Wniosek` are empty. Use the tables in `[MAR]\Wyniki\Dane\eden\redukcja_pilot\`.
- **O12** (W2 opposition growth):
  - `[S3]\geometria_ostatni_mlp.png` and `geometria_szesc_blokow.png`
  - `[PRUNE]\figures_20261019\trajectories_clean_winner.png`
  - `[WSP]\przeciwne_reflektory\przebieg_w_czasie.png`
- **O13** (geometry vs function, co-activation):
  - `[PRUNE]\figures_20261019\group_retrospective_fast16_clean_winner.png`
  - `[WSP]\przeciwne_aktywacje\wspolaktywacja_w_czasie.png`
  - `[FUN24]\figures\directions_confidence_24seeds.png`
- **O14, O15** (compensation, pruning and restoration):
  - `[FUN24]\figures\functional_compensation_24seeds.png`
  - `[PRUNE]\figures_20261019\pruning_functional_clean_winner.png` and `contrasts_clean_winner.png`
  - `[RESTORE]\figures\poziomy_clean_winner.png` and `narastajaco_exhaustive_clean_winner.png`
  - `[S3]\frozen_q_gain.png`
- **O17** (concentration vs zeroing): `[ATLAS]\Wyniki\Wykresy\01_podsumowanie.png`, `02_aktywnosc_wzgledna.png`, `03_aktywnosc_absolutna.png`, `04_efektywna_liczba.png`, `07_zera.png`, `08_przejscia.png`
- **O18** (angle distributions): `[ATLAS]\Wyniki\Wykresy\09_mean_angle.png`, `10_mean_distance_from90.png`, `11_rozklad_katow.png`, `13_rozklad_zmian_katow.png`
- Every [ATLAS] and [P200] figure also exists as SVG and PDF. The full atlas is `[ATLAS]\Wyniki\Wykresy\ATLAS_WYKRESOW.pdf`. The detailed reports are `[ATLAS]\Wyniki\Raport\RAPORT.pdf` and `[P200]\Wyniki\Raport\RAPORT.pdf`.
- Do not cite `[RESTORE]\validation_plot\synthetic_figures\*`; those plots are built from synthetic inputs.
