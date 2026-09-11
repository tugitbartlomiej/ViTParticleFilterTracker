"""Read-only project audit, PDF extraction and a COCO metric counterexample."""
import contextlib
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import fitz

OUT = Path(__file__).resolve().parents[1]
ROOT = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker')
PAPER = Path('F:/Studia/Articles/Moj/IEEE/Overleaf/DETR_IEEE/Uwagi/Final')

def save(name, obj):
    (OUT / 'evidence' / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    files = list((ROOT / 'SmallViTAblation/pipeline').rglob('*.py'))
    files += [ROOT / 'SmallViTAblation/configs/protocol.yaml', PAPER/'DETR_IEEE_FINAL.tex', PAPER/'DETR_IEEE_FINAL.pdf']
    save('file_hashes.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    # Literature PDFs were moved to pdf_all. Use the user's Ocr/scan_rtdetr.bat
    # for subsequent document processing; retain papers/*.txt as audit history.
    with fitz.open(PAPER/'DETR_IEEE_FINAL.pdf') as doc:
        pageinfo=[]
        for i,page in enumerate(doc):
            t=page.get_text()
            pageinfo.append({'page':i+1,'chars':len(t)})
            if 'Random selection' in t or ('EVALUATION METRICS' in t):
                page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'evidence'/f'paper_page_{i+1}.png')
        save('pdf_pages.json',pageinfo)
    split=json.loads((ROOT/'Eden/Checkpoints/Ablations/ablation/video_split.json').read_text(encoding='utf-8'))
    save('split_structure.json',split)
    selected=json.loads((ROOT/'AdvancedDatasetSelection/output/selected_dataset1/merged_20k_annotations_fixed.json').read_text(encoding='utf-8'))
    leaks=json.loads((ROOT/'YOLO_DETR_Benchmarks/scripts/leaked_valid_images.json').read_text(encoding='utf-8'))
    names=[Path(x['file_name']).name for x in selected['images']]
    sources=Counter(re.sub(r'_aug_\d+(?=\.[^.]+$)','',n) for n in names)
    leak_list = leaks if isinstance(leaks,list) else leaks.get('images', leaks.get('leaked_images', []))
    leaked_names={Path(x if isinstance(x,str) else x.get('file_name','')).name for x in leak_list}
    membership={}
    for key in ('train','val','test'):
        vals=split.get(key,split.get(key+'_videos',[]))
        if isinstance(vals,list):
            for v in vals: membership[str(v)]=key
    counts=Counter(membership.get(n.split('_frame_')[0],'unknown') for n in names)
    save('production_pool_audit.json',{'annotation_file':str(ROOT/'AdvancedDatasetSelection/output/selected_dataset1/merged_20k_annotations_fixed.json'),'images':len(names),'unique_sources':len(sources),'augmented_images':sum(bool(re.search(r'_aug_\d+\.',n)) for n in names),'redundant_source_copies':len(names)-len(sources),'membership_current_video_split':dict(counts),'leak_file_keys':list(leaks) if isinstance(leaks,dict) else 'list','matched_leaked_filenames': sorted(set(names)&leaked_names),'note':'Filename/source-name audit only; not a pixel-hash test or proof of historical checkpoint provenance.'})
    normalized=[]
    for p in (OUT/'search_logs').glob('0*.txt'):
        s=p.read_text(encoding='utf-8-sig',errors='replace')
        start=s.find('{')
        if start<0:continue
        try:d=json.loads(s[start:])
        except json.JSONDecodeError:continue
        for item in d.get('papers',[]):
            normalized.append({'query':d.get('query'), 'log':p.name, **item})
    (OUT/'search_results.json').write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Extracted papers and saved project evidence; search records:',len(normalized))

def metric_demo():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    def ap(score,threshold):
        with contextlib.redirect_stdout(io.StringIO()):
            gt=COCO()
            gt.dataset={'info':{},'images':[{'id':1,'width':100,'height':100}], 'categories':[{'id':1,'name':'tip'}], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,10,10,10],'area':100,'iscrowd':0}]}
            gt.createIndex()
            detections=[{'image_id':1,'category_id':1,'bbox':[10,10,10,10],'score':score}] if score>=threshold else []
            if detections:dt=gt.loadRes(detections)
            else:
                dt=COCO(); dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
            ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
            return {'ap50':float(ev.stats[1]),'ap50_95':float(ev.stats[0])}
    save('coco_threshold_demo.json',{'perfect_box_score_0.2_cutoff_0.001':ap(0.2,0.001),'same_box_cutoff_0.3':ap(0.2,0.3),'same_ranking_score_0.8_cutoff_0.001':ap(0.8,0.001),'interpretation':'Confidence filtering alone can erase a correct detection. AP uses score ordering; a strictly monotonic score transformation without changing the retained detections does not change AP.'})
    print('COCO threshold counterexample saved.')

if __name__=='__main__':
    main()
    metric_demo()
