# Zweryfikowane źródła: selekcja i ewaluacja RT-DETR

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data dostępu: 2026-09-11. Wyszukiwanie: CLI paper-search (arXiv, Semantic Scholar, Crossref) oraz internet. Semantic Scholar częściowo zwracał 429; wykorzystano pozostałe źródła. Surowe wyniki nie są bibliografią: część jest nietrafna. Nie przeczytano wszystkich prac w całości; zakres lektury podano jawnie. 19 publikacji i 8 źródeł implementacyjnych.

| ID | Tytuł i źródło | Autorzy | Rok / venue | Zastosowanie i ograniczenie |
|---|---|---|---|---|
| S01 | [Metrics Reloaded: Recommendations for image analysis validation](https://doi.org/10.1038/s41592-023-02151-z) | Maier-Hein et al. | 2024, Nature Methods | Dobór metryk do zadania, małych obiektów i struktury pacjent/klatka. Nie zastępuje definicji COCO. |
| S02 | [Understanding metric-related pitfalls in image analysis validation](https://doi.org/10.1038/s41592-023-02150-0) | Reinke et al. | 2024, Nature Methods | Pułapki agregowania i interpretowania metryk; uzupełnienie S01. |
| S03 | [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) | Bouthillier et al. | 2021, MLSys | Zmienność doboru danych, inicjalizacji i hiperparametrów; nie ustanawia magicznej liczby seedów. |
| S04 | [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html) | Cawley and Talbot | 2010, JMLR | Wybór modeli może przeuczyć kryterium walidacyjne. Test musi pozostać poza wyborem. |
| S05 | [Machine learning for medical imaging: methodological failures and recommendations for the future](https://doi.org/10.1038/s41746-022-00592-y) | Varoquaux and Cheplygina | 2022, npj Digital Medicine | Walidacja na danych medycznych, źródła stronniczości i niepewność wyników. |
| S06 | [Leakage and the reproducibility crisis in machine-learning-based science](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/) | Kapoor and Narayanan | 2023, Patterns | Przecieki danych i dokumentowanie protokołu; nie dowodzi historii konkretnych lokalnych checkpointów. |
| S07 | [Coreset Selection for Object Detection](https://arxiv.org/abs/2404.09161) | Lee, Kim, Lee, Yoo and Kwak | 2024, CVPR Workshop DDCV | CSOD: reprezentacje obiektów z GT, reprezentatywność i różnorodność; Faster R-CNN, transfer do RetinaNet/FCOS. |
| S08 | [Single-Pass Object-Focused Data Selection](https://arxiv.org/abs/2412.10032) | Popp, Zhang, Metzen, Hein and Schott | 2024 preprint; v2 June 2025 | OFDS: obiektowe cechy modeli foundation, selekcja przed etykietowaniem. Nasze cropy GT to adaptacja do już opisanej puli. |
| S09 | [Correlation-Aware Active Learning for Surgery Video Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.html) | Wu, Márquez-Neila, Zheng, Rafii-Tari and Sznitman | 2024, WACV | COWAL: klastry i niepewność przy skorelowanych klatkach. Segmentacja i AL, nie bezpośrednia walidacja selekcji RT-DETR. |
| S10 | [AdaDeDup: Adaptive Hybrid Data Pruning for Efficient Large-Scale Object Detection Training](https://arxiv.org/abs/2507.00049) | Kang et al. | 2025, preprint | Klastry/dedup oraz feedback proxy. Budżet per klaster; nie gotowa recepta na zachowanie 22% danych chirurgicznych. |
| S11 | [Foundation-Assisted Active Learning for Object Detection Annotation](https://arxiv.org/abs/2607.16671) | Zhang, Zumbrun, Lin and Lu | 2026, preprint | RTDETRV2, DINOv2, SAM2, UPN. Hungarian dopasowuje dwie pule propozycji, a nie GT. Zdjęcia lotnicze; ograniczenia opisu eksperymentu w raporcie. |
| S12 | [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://arxiv.org/abs/2107.07075) | Paul, Ganguli and Dziugaite | 2021, NeurIPS; corrected arXiv v2 2023 | EL2N: oczekiwana norma błędu klasyfikacji we wczesnym treningu. Ocena train przez model uczony na train jest dopuszczalna. |
| S13 | [Selection via Proxy: Efficient Data Selection for Deep Learning](https://arxiv.org/abs/1906.11829) | Coleman et al. | 2020, ICLR; preprint 2019 | Mały proxy może być użyteczny do selekcji; przydatność transferu trzeba sprawdzić w naszej domenie. |
| S14 | [DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning](https://arxiv.org/abs/2204.08499) | Guo, Zhao and Bai | 2022, ECCV | Random jest mocnym baseline; praca głównie o klasyfikacji. |
| S15 | [Coverage-centric Coreset Selection for High Pruning Rates](https://arxiv.org/abs/2210.15809) | Zheng, Liu, Lai and Prakash | 2023, ICLR; preprint 2022 | Znaczenie pokrycia przy silnej redukcji; brak gwarancji przewagi dowolnej mieszanki cech. |
| S16 | [Temporal Coherence for Active Learning in Videos](https://arxiv.org/abs/1908.11757) | Bengar et al. | 2019, ICCV Workshop | AL detekcji w wideo i redundancja czasowa; uzasadnia kontrolę czasową jako punkt odniesienia. |
| S17 | [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) | Zhao et al. | 2024, CVPR; preprint 2023 | RT-DETR: end-to-end detector, dynamiczna selekcja zapytań; nie przenosić stałego slotu Q81. |
| S18 | [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140) | Lv et al. | 2024, arXiv technical report | Dokładny wariant, schedule augmentacji i pomiar TensorRT/FP16 trzeba zdefiniować osobno od RF-DETR. |
| S19 | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554) | Robinson, Robicheaux, Popov, Ramanan and Peri | 2026, ICLR; preprint 2025 | RF-DETR i RT-DETR to różne modele; obecny SmallViTAblation jest kierowany na RF-DETR. |
| C01 | [COCO API: reference evaluator](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py) | COCO authors | Official code, retrieved 2026-09-11 | Definicje AP/AR, pełne imgIds, IoU, maxDets, sortowanie i matching. |
| C02 | [RT-DETR official implementation](https://github.com/lyuwenyu/RT-DETR) | lyuwenyu/RT-DETR | Official repository, retrieved 2026-09-11 | Podstawa implementacji adaptera RT-DETRv2. |
| C03 | [RT-DETRv2 Hungarian matcher](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/matcher.py) | RT-DETR authors | Official code | Matcher treningowy dla raw logits i boxów; odrębny od matchingu ewaluacyjnego COCO. |
| C04 | [RT-DETRv2 postprocessor](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py) | RT-DETR authors | Official code | Sigmoid focal, top-k, konwersja cxcywh i skalowanie do obrazu. |
| C05 | [RF-DETR training and result checkpoints](https://rfdetr.roboflow.com/learn/train/) | Roboflow | Official documentation, retrieved 2026-09-11 | Layout danych, ładowanie wag, różne checkpointy best/last. |
| C06 | [RF-DETR API source 1.2.1](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/detr.py) | Roboflow | Versioned official source | API odniesienia w audycie; nie dowód wersji zainstalowanej na Edenie. |
| C07 | [RF-DETR configuration 1.2.1](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/config.py) | Roboflow | Versioned official source | Nazwy konfiguracji, których sam **kwargs nie waliduje. |
| C08 | [RT-DETRv2 custom COCO dataset loader](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py) | RT-DETR authors | Official code | Jawne mapowanie custom category_id do etykiet 0..C-1; samo wyłączenie remap_mscoco nie mapuje id=1 do 0. |

## Co rzeczywiście sprawdzono

- **S01**: Full-text sections on object detection, aggregation, empty outputs, confidence intervals. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/arxiv_2206.01653.pdf)
- **S02**: Publisher full-text web passages.
- **S03**: Abstract and full-text experimental/recommendation sections. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2103.03098.pdf)
- **S04**: Abstract on JMLR.
- **S05**: Publisher full-text web passages.
- **S06**: Abstract and authors project page.
- **S07**: Methods 3.1-3.3 and experiments 4.1/4.8. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2404.09161.pdf)
- **S08**: Method and appendix A on Deformable DETR. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2412.10032.pdf)
- **S09**: Method and experiment sections. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/openaccess.thecvf.com_Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.pdf)
- **S10**: Sections 3.3 and 4. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2507.00049.pdf)
- **S11**: Sections III and IV-A from full HTML/PDF. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2607.16671.pdf)
- **S12**: Definition 2.3 and section 3, correction notice. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2107.07075.pdf)
- **S13**: Abstract.
- **S14**: Abstract.
- **S15**: Abstract.
- **S16**: Abstract.
- **S17**: Official abstract, PDF methods, official code. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2304.08069.pdf)
- **S18**: Sections 2-3 and official configurations. [PDF](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/pdf_all/2407.17140.pdf)
- **S19**: Abstract and official documentation.
- **C01**: Code and local synthetic COCOeval experiments.
- **C02**: Matcher, postprocessor, criterion, dataset and model config.
- **C03**: Code.
- **C04**: Code.
- **C05**: Documentation.
- **C06**: Code.
- **C07**: Code.
- **C08**: Code.
