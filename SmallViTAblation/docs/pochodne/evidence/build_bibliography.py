"""Curated, verified references; raw search hits are stored separately."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
# id, title, authors, year/venue, url, local PDF, evidence read, relevance/limitations
ROWS = [
('S01','Metrics Reloaded: Recommendations for image analysis validation','Maier-Hein et al.','2024, Nature Methods','https://doi.org/10.1038/s41592-023-02151-z','2206.01653.pdf','Full-text sections on object detection, aggregation, empty outputs, confidence intervals','Dobór metryk do zadania, małych obiektów i struktury pacjent/klatka. Nie zastępuje definicji COCO.'),
('S02','Understanding metric-related pitfalls in image analysis validation','Reinke et al.','2024, Nature Methods','https://doi.org/10.1038/s41592-023-02150-0','','Publisher full-text web passages','Pułapki agregowania i interpretowania metryk; uzupełnienie S01.'),
('S03','Accounting for Variance in Machine Learning Benchmarks','Bouthillier et al.','2021, MLSys','https://arxiv.org/abs/2103.03098','2103.03098.pdf','Abstract and full-text experimental/recommendation sections','Zmienność doboru danych, inicjalizacji i hiperparametrów; nie ustanawia magicznej liczby seedów.'),
('S04','On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation','Cawley and Talbot','2010, JMLR','https://www.jmlr.org/papers/v11/cawley10a.html','','Abstract on JMLR','Wybór modeli może przeuczyć kryterium walidacyjne. Test musi pozostać poza wyborem.'),
('S05','Machine learning for medical imaging: methodological failures and recommendations for the future','Varoquaux and Cheplygina','2022, npj Digital Medicine','https://doi.org/10.1038/s41746-022-00592-y','','Publisher full-text web passages','Walidacja na danych medycznych, źródła stronniczości i niepewność wyników.'),
('S06','Leakage and the reproducibility crisis in machine-learning-based science','Kapoor and Narayanan','2023, Patterns','https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/','','Abstract and authors project page','Przecieki danych i dokumentowanie protokołu; nie dowodzi historii konkretnych lokalnych checkpointów.'),
('S07','Coreset Selection for Object Detection','Lee, Kim, Lee, Yoo and Kwak','2024, CVPR Workshop DDCV','https://arxiv.org/abs/2404.09161','2404.09161.pdf','Methods 3.1-3.3 and experiments 4.1/4.8','CSOD: reprezentacje obiektów z GT, reprezentatywność i różnorodność; Faster R-CNN, transfer do RetinaNet/FCOS.'),
('S08','Single-Pass Object-Focused Data Selection','Popp, Zhang, Metzen, Hein and Schott','2024 preprint; v2 June 2025','https://arxiv.org/abs/2412.10032','2412.10032.pdf','Method and appendix A on Deformable DETR','OFDS: obiektowe cechy modeli foundation, selekcja przed etykietowaniem. Nasze cropy GT to adaptacja do już opisanej puli.'),
('S09','Correlation-Aware Active Learning for Surgery Video Segmentation','Wu, Márquez-Neila, Zheng, Rafii-Tari and Sznitman','2024, WACV','https://openaccess.thecvf.com/content/WACV2024/html/Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.html','COWAL_WACV2024.pdf','Method and experiment sections','COWAL: klastry i niepewność przy skorelowanych klatkach. Segmentacja i AL, nie bezpośrednia walidacja selekcji RT-DETR.'),
('S10','AdaDeDup: Adaptive Hybrid Data Pruning for Efficient Large-Scale Object Detection Training','Kang et al.','2025, preprint','https://arxiv.org/abs/2507.00049','2507.00049.pdf','Sections 3.3 and 4','Klastry/dedup oraz feedback proxy. Budżet per klaster; nie gotowa recepta na zachowanie 22% danych chirurgicznych.'),
('S11','Foundation-Assisted Active Learning for Object Detection Annotation','Zhang, Zumbrun, Lin and Lu','2026, preprint','https://arxiv.org/abs/2607.16671','2607.16671.pdf','Sections III and IV-A from full HTML/PDF','RTDETRV2, DINOv2, SAM2, UPN. Hungarian dopasowuje dwie pule propozycji, a nie GT. Zdjęcia lotnicze; ograniczenia opisu eksperymentu w raporcie.'),
('S12','Deep Learning on a Data Diet: Finding Important Examples Early in Training','Paul, Ganguli and Dziugaite','2021, NeurIPS; corrected arXiv v2 2023','https://arxiv.org/abs/2107.07075','2107.07075.pdf','Definition 2.3 and section 3, correction notice','EL2N: oczekiwana norma błędu klasyfikacji we wczesnym treningu. Ocena train przez model uczony na train jest dopuszczalna.'),
('S13','Selection via Proxy: Efficient Data Selection for Deep Learning','Coleman et al.','2020, ICLR; preprint 2019','https://arxiv.org/abs/1906.11829','','Abstract','Mały proxy może być użyteczny do selekcji; przydatność transferu trzeba sprawdzić w naszej domenie.'),
('S14','DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning','Guo, Zhao and Bai','2022, ECCV','https://arxiv.org/abs/2204.08499','','Abstract','Random jest mocnym baseline; praca głównie o klasyfikacji.'),
('S15','Coverage-centric Coreset Selection for High Pruning Rates','Zheng, Liu, Lai and Prakash','2023, ICLR; preprint 2022','https://arxiv.org/abs/2210.15809','','Abstract','Znaczenie pokrycia przy silnej redukcji; brak gwarancji przewagi dowolnej mieszanki cech.'),
('S16','Temporal Coherence for Active Learning in Videos','Bengar et al.','2019, ICCV Workshop','https://arxiv.org/abs/1908.11757','','Abstract','AL detekcji w wideo i redundancja czasowa; uzasadnia kontrolę czasową jako punkt odniesienia.'),
('S17','DETRs Beat YOLOs on Real-time Object Detection','Zhao et al.','2024, CVPR; preprint 2023','https://arxiv.org/abs/2304.08069','2304.08069.pdf','Official abstract, PDF methods, official code','RT-DETR: end-to-end detector, dynamiczna selekcja zapytań; nie przenosić stałego slotu Q81.'),
('S18','RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer','Lv et al.','2024, arXiv technical report','https://arxiv.org/abs/2407.17140','2407.17140.pdf','Sections 2-3 and official configurations','Dokładny wariant, schedule augmentacji i pomiar TensorRT/FP16 trzeba zdefiniować osobno od RF-DETR.'),
('S19','RF-DETR: Neural Architecture Search for Real-Time Detection Transformers','Robinson, Robicheaux, Popov, Ramanan and Peri','2026, ICLR; preprint 2025','https://arxiv.org/abs/2511.09554','','Abstract and official documentation','RF-DETR i RT-DETR to różne modele; obecny SmallViTAblation jest kierowany na RF-DETR.'),
('C01','COCO API: reference evaluator','COCO authors','Official code, retrieved 2026-09-11','https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py','','Code and local synthetic COCOeval experiments','Definicje AP/AR, pełne imgIds, IoU, maxDets, sortowanie i matching.'),
('C02','RT-DETR official implementation','lyuwenyu/RT-DETR','Official repository, retrieved 2026-09-11','https://github.com/lyuwenyu/RT-DETR','','Matcher, postprocessor, criterion, dataset and model config','Podstawa implementacji adaptera RT-DETRv2.'),
('C03','RT-DETRv2 Hungarian matcher','RT-DETR authors','Official code','https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/matcher.py','','Code','Matcher treningowy dla raw logits i boxów; odrębny od matchingu ewaluacyjnego COCO.'),
('C04','RT-DETRv2 postprocessor','RT-DETR authors','Official code','https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py','','Code','Sigmoid focal, top-k, konwersja cxcywh i skalowanie do obrazu.'),
('C05','RF-DETR training and result checkpoints','Roboflow','Official documentation, retrieved 2026-09-11','https://rfdetr.roboflow.com/learn/train/','','Documentation','Layout danych, ładowanie wag, różne checkpointy best/last.'),
('C06','RF-DETR API source 1.2.1','Roboflow','Versioned official source','https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/detr.py','','Code','API odniesienia w audycie; nie dowód wersji zainstalowanej na Edenie.'),
('C07','RF-DETR configuration 1.2.1','Roboflow','Versioned official source','https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/config.py','','Code','Nazwy konfiguracji, których sam **kwargs nie waliduje.'),
('C08','RT-DETRv2 custom COCO dataset loader','RT-DETR authors','Official code','https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py','','Code','Jawne mapowanie custom category_id do etykiet 0..C-1; samo wyłączenie remap_mscoco nie mapuje id=1 do 0.'),
]
FIELDS=['id','title','authors','year_venue','url','local_pdf','read_scope','relevance_limitations']
def main():
    data=[dict(zip(FIELDS,row)) for row in ROWS]
    (OUT/'sources.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'sources.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(data)
    (OUT/'linki_artykuly.txt').write_text('\n'.join(r['url'] for r in data if r['id'].startswith('S'))+'\n',encoding='utf-8')
    lines=['# Zweryfikowane źródła: selekcja i ewaluacja RT-DETR','',
        'Data dostępu: 2026-09-11. Wyszukiwanie: CLI paper-search (arXiv, Semantic Scholar, Crossref) oraz internet. Semantic Scholar częściowo zwracał 429; wykorzystano pozostałe źródła. Surowe wyniki nie są bibliografią: część jest nietrafna. Nie przeczytano wszystkich prac w całości; zakres lektury podano jawnie. 19 publikacji i 8 źródeł implementacyjnych.','',
        '| ID | Tytuł i źródło | Autorzy | Rok / venue | Zastosowanie i ograniczenie |','|---|---|---|---|---|']
    for r in data:
        lines.append(f"| {r['id']} | [{r['title']}]({r['url']}) | {r['authors']} | {r['year_venue']} | {r['relevance_limitations']} |")
    lines+=['','## Co rzeczywiście sprawdzono','']
    for r in data:
        local=f" [PDF](pdf_all/{r['local_pdf']})" if r['local_pdf'] else ''
        lines.append(f"- **{r['id']}**: {r['read_scope']}.{local}")
    (OUT/'ZRODLA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved',len(data),'curated sources;',sum(r['id'].startswith('S') for r in data),'papers.')
if __name__=='__main__':main()
