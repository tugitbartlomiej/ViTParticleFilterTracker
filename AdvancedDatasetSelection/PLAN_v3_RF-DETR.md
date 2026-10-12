# Propozycja protokołu selekcji klatek v3 — pod RF-DETR (2026-09-10)

Źródło: weryfikacja literaturowa z 2026-09-10 (Undermind deep search, 50 PDF przeczytanych w całości,
workspace ViT `40294fa6-c9a3-4469-9a61-d3eddeaee893`; linki: `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDownloader\linki\selekcja-klatek-koreset-detekcja-2026-09-10.txt`).

## 1. Diagnoza obecnego protokołu (v2, artykuł DETR_IEEE_FINAL)

| # | Słaby punkt | Dowód | Literatura |
|---|-------------|-------|------------|
| 1 | Selekcja z puli PO augmentacji: 75,3 % z 20 k wybranych to kopie `_aug_N`; 4 227 kadrów źródłowych wybranych >1 raz, czyli 23,4 % wyboru to duplikaty | `selection_reasons.json`, skrypt z sesji | Kuchnik & Smith 2019 (arXiv 1810.05222); Liu & Mirzasoleiman 2022 (10.48550/arXiv.2210.08363) |
| 2 | Trudność EL2N z jednego zapytania (Q81): nieprzenośna na RF-DETR (brak specjalizacji zapytań, zapytania uczą się położeń) | RF-DETR, sekcja 3 | Robinson i in. 2026 (10.48550/arXiv.2511.09554); Kwok i in. 2024 (10.48550/arXiv.2401.01867) |
| 3 | DINOv3 CLS opisuje całą scenę, końcówka to ułamek kadru; koresety z cech całego obrazu bywają gorsze od losowania w detekcji | brak | Lee i in. 2024 CSOD (10.1109/CVPRW63382.2024.00764); Popp i in. 2024 OFDS (arXiv 2412.10032); Xie i in. 2023 FreeSel (10.48550/arXiv.2309.17342) |
| 4 | k = B i najbliższy centroidowi: same typowe kadry, brak rzadkich póz; brak budżetu per klaster | brak | Abbas i in. 2024 DBP (10.48550/arXiv.2401.04578); Kang i in. 2025 AdaDeDup (10.48550/arXiv.2507.00049); Zheng i in. 2023 CCS (10.48550/arXiv.2210.15809) |
| 5 | Ablacja: random 0,23 % vs pipeline 26,04 %; skala 100x nie występuje w literaturze (typowo kilka pp); pełny pipeline trenowany innym harmonogramem | `DETR_IEEE_FINAL.tex`, sekcja Ablation | Guo i in. 2022 DeepCore (10.48550/arXiv.2204.08499); Ayed & Hayou 2023 (10.48550/arXiv.2302.06960) |
| 6 | Brak baseline'u czasowego (uniform-in-time, Random+R); na wideo laparoskopowym uniform wygrywa do 33 % budżetu | brak | Shen i in. 2025 (10.48550/arXiv.2505.18302); Bengar i in. 2019 (10.1109/ICCVW.2019.00120) |

## 2. Prace-wzorce (co z której wziąć)

| Rola | Praca | Co przejąć |
|------|-------|------------|
| Szkielet całości (detekcja, hybryda dedup + feedback modelu, budżet per klaster) | **AdaDeDup**, Kang i in. 2025, 10.48550/arXiv.2507.00049 | dwa etapy: klastry VLM, proxy, porównanie loss kept/pruned w klastrze, korekta budżetu |
| Najbliższy domenowo (RT-DETRv2 + DINOv2 obiektowe + Hungarian) | **Zhang i in. 2026**, arXiv 2607.16671 | cechy obiektowe DINOv2 + niepewność z dopasowania Hungarian; novelty score względem prototypów |
| Selekcja obiektowa jednoprzebiegowa | **OFDS**, Popp i in. 2024, arXiv 2412.10032 | k-means na cechach OBIEKTÓW, jeden obiekt na klaster najbliższy centroidowi; jawne porównanie obiekt vs obraz; Deformable DETR |
| Wideo chirurgiczne | **COWAL**, Wu i in. 2024, 10.1109/WACV57701.2024.00201 | najpierw klastry (k-means z zamrożonymi centroidami etykietowanych), potem najbardziej niepewna klatka w klastrze |
| Trudność z trajektorii dla detekcji + transfer na inny detektor | **VPS**, Yagi 2025, 10.48550/arXiv.2505.17245 | std(IoU), std(conf) po epokach; wybór o pośredniej trudności; tab. 3: Faster R-CNN do YOLOv5 |
| Trudność x niepewność bez pełnego treningu | **DUAL**, Cho i in. 2025, 10.48550/arXiv.2502.06905 | (1 - mean p) * std(p) w oknie epok + Beta-sampling zależny od stopnia przycięcia |
| Protokół oceny transferu | **CSOD**, Lee i in. 2024, 10.1109/CVPRW63382.2024.00764 | tab. 7: subset wybrany jednym detektorem, trenowane inne |
| Budżet per klaster | **DBP**, Abbas i in. 2024, 10.48550/arXiv.2401.04578 | C_j = d_inter * d_intra, softmax(C / tau), QP z ograniczeniem M_j |
| Czas w wideo | **Bengar i in. 2019**, 10.1109/ICCVW.2019.00120 | Random+R: nie wybieraj k sąsiadów już wybranej klatki |
| Uzasadnienie cech częstotliwościowych | **FAST**, Cui i in. 2025, 10.48550/arXiv.2511.19476 | częstotliwość (faza) uzupełnia embeddingi sieci; brak dowodu detekcyjnego, traktować jako ablację |

## 3. Protokół v3 — kroki

0. **Pula = kadry źródłowe** (~22,5 k z 50 wideo), bez augmentacji. Metadane: `video_id`, `t` (czas znormalizowany 0–1), ramka GT.
1. **Cechy** (wszystkie na kadrach źródłowych):
   - a) *obiektowe*: DINOv3 ViT-L/16 patch tokens uśrednione wewnątrz ramki GT powiększonej 1,5x, 1024 wymiary, PCA do 32 (OFDS, Zhang 2026, Sokolov 2025, CSOD);
   - b) *kontekst sceny*: CLS, PCA do 16, mała waga (0,10);
   - c) *trudność trajektoryjna, bez Q81*: dla każdego kadru i każdego zapisanego checkpointu DETR (ep100–170, już istnieją) zapytanie dopasowane Hungarianem do GT daje `p_tooltip` i `IoU`; `VPS = std_ep(IoU)`, `DUAL = (1 - mean_ep p) * std_ep(p)`. **Dla RF-DETR**: 10–20 epok warm-up na całej puli źródłowej z checkpointem co epokę, ta sama miara na docelowej architekturze (Kwok 2024, Coleman 2020);
   - d) Fourier 9-D i FastSAM 1-D: zostają jako grupa o łącznej wadze do 0,15, wyłącznie do ablacji;
   - e) *czas*: nie jako cecha, lecz jako ograniczenie (krok 4).
2. **Klastrowanie**: K-Means (FAISS) na [a, b, d] z alfa = sqrt(w / d) jak dotąd, ale **k = B/3 … B/5** (np. 1 500 klastrów dla B = 5 000 kadrów), nie k = B.
3. **Budżet per klaster** (nowość N1): `b_c ~ C_c * T_c * U_c`, gdzie `C_c` = złożoność DBP (d_intra * d_inter), `T_c` = rozpiętość czasowa klastra (liczba różnych wideo + zakres `t`), `U_c` = średni DUAL w klastrze; softmax z temperaturą i ograniczeniem `b_c <= |c|` (QP jak w DBP). Klaster z jednego ujęcia jednego wideo dostaje 1; klaster obejmujący wiele faz lub wideo dostaje więcej. Opcjonalnie pętla AdaDeDup: po krótkim treningu proxy porównać loss kept/pruned w klastrze i skorygować `b_c`.
4. **Wybór w klastrze**: stratyfikacja po DUAL (CCS): odrzuć górne 5 % najtrudniejszych (szum, Mindermann 2022), resztę podziel na pasma i losuj po równo; dodatkowo **min. odstęp dt** między wybranymi z tego samego wideo (Random+R).
5. **Augmentacja PO wyborze**: RF-DETR tylko natywne (flip, crop); dodatkowe kopie tylko dla kadrów z górnego kwartyla DUAL (Kuchnik 2019: augmentuj trudne). Dla DETR i YOLOv8 analogicznie, żeby porównanie było uczciwe.
6. **Ewaluacja** (najważniejsze dla recenzentów):
   - identyczny harmonogram treningu dla WSZYSTKICH wariantów (ten sam checkpoint startowy, LR, epoki, stop), 3 ziarna, przedziały ufności;
   - split per wideo (już jest w `ablation/detr/PLAN.md`), ocena same-distribution + zewnętrzna;
   - baseline'y: random, **Random+R (czasowy)**, **uniform-in-time**, SemDeDup-lite (k-means + próg cos), k-center, CCS na DUAL, v2 (obecny), v3;
   - budżety 10 / 20 / 30 / 50 % kadrów źródłowych (optymalna strategia zależy od budżetu, Sorscher 2022);
   - **transfer**: subset wybrany sygnałami DETR trenuje RF-DETR i YOLOv8 (tabela jak CSOD tab. 7 i VPS tab. 3); odczyt: RF-DETR = max po zapytaniach; DETR raportować Q81 i all-query;
   - oczekiwany efekt: kilka pp mAP@0.5 nad random, nie rzędy wielkości.

## 4. Co w tym jest nowe (ostrożnie)

- **N1** budżet per klaster zależny od rozpiętości czasowej i trudności trajektoryjnej: DBP używa gęstości, AdaDeDup lossu proxy, Bengar wyklucza sąsiadów; połączenia dla wideo nie znalazłem.
- **N2** trudność z dopasowanego (Hungarian) zapytania po trajektorii checkpointów dla rodziny DETR, bez stałego zapytania: VPS był dla Faster R-CNN; DetGain (Sun 2025) używa Deformable DETR, ale z nauczycielem, nie z trajektorii.
- **N3** cechy obiektowe DINOv3 w ramce GT dla jednoklasowej, drobnej końcówki: OFDS i Zhang 2026 robią to z Grounding DINO i SAM2; tu GT istnieje, więc prościej i taniej.
- **N4** pełna tabela transferu DETR, RF-DETR, YOLOv8 na wideo chirurgicznym, z baseline'ami czasowymi: wkład empiryczny.

## 5. Co da się zrobić od razu z istniejących artefaktów

- duplikaty augmentacji: policzone (23,4 %); skrypt: strip `_aug_\d+` w `selection_reasons.json`;
- checkpointy DETR ep100–170 istnieją, więc VPS/DUAL na puli źródłowej bez nowego treningu;
- cache DINOv3 CLS istnieje; patch tokens wymagają jednego przebiegu po ~22,5 k kadrów;
- `ablation/detr/PLAN.md` ma już split per wideo i wspólny harmonogram; rozszerzyć o baseline'y czasowe i budżety.

## 6. Ryzyka

- efekt v3 może być mały względem Random+R i uniform-in-time; wtedy wynikiem jest rzetelna ocena z transferem, publikowalna, jeśli protokół jest czysty;
- odczyt Q81 w obecnej ewaluacji zawyża lub zaniża zależnie od tego, które zapytanie przejęło klasę po fine-tuningu; raportować oba odczyty.
