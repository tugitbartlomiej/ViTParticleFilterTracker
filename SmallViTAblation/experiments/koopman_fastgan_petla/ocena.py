"""Ocena zamrożonego checkpointu; test uruchamiany dopiero po decyzjach rund."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from dane import read_json,write_json

def matching(boxes,truth,threshold=.5):
    """Greedy w kolejności pewności; jedno trafienie na każdą adnotację."""
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    truth=np.asarray(truth,dtype=float).reshape(-1,4)
    used=set(); hits=0
    for box in boxes:
        best,score=None,0.
        for j,gt in enumerate(truth):
            if j in used: continue
            lt=np.maximum(box[:2],gt[:2]); rb=np.minimum(box[2:],gt[2:])
            inter=np.maximum(rb-lt,0).prod()
            union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(gt[2:]-gt[:2],0).prod()-inter
            iou=inter/max(union,1e-12)
            if iou>score: best,score=j,iou
        if best is not None and score>=threshold: used.add(best);hits+=1
    return dict(tp=hits,fp=len(boxes)-hits,fn=len(truth)-hits,objects=len(truth))

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    annotations=read_json(Path(args.dataset)/('test' if args.split=='test' else 'valid')/'_annotations.coco.json')
    if not annotations['images']:
        write_json(out/'REPORT.json',dict(state='unavailable',split=args.split,
            reason='No images in the predeclared split; validation is not relabelled as test.',
            metrics=None,counts=None,per_frame=[]))
        return
    from rfdetr import RFDETRNano
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    model=RFDETRNano(pretrain_weights=args.weights,num_classes=1,resolution=384,amp=False,
                      device=args.device,trust_checkpoint=True)
    metrics=model.evaluate(dataset_dir=args.dataset,split=args.split,batch_size=4,num_workers=0,
                           accelerator='gpu' if args.device!='cpu' else 'cpu',devices=1,
                           progress_bar=None,tensorboard=False,wandb=False,mlflow=False,
                           output_dir=str(out),use_ema=False)
    annotations=read_json(Path(args.dataset)/('test' if args.split=='test' else 'valid')/'_annotations.coco.json')
    mapping=read_json(Path(args.dataset)/'mapping.json')['test' if args.split=='test' else 'val']
    byimage={}
    for a in annotations['annotations']:
        x,y,w,h=a['bbox'];byimage.setdefault(a['image_id'],[]).append([x,y,x+w,y+h])
    rows=[]
    for image in annotations['images']:
        with Image.open(image['file_name']) as im: predictions=model.predict(im.convert('RGB'),threshold=.5)
        order=np.argsort(-predictions.confidence)
        boxes=predictions.xyxy[order]
        rows.append(dict(**mapping[image['id']],**matching(boxes,byimage.get(image['id'],[]))))
    totals={key:sum(row[key] for row in rows) for key in ['tp','fp','fn','objects']}
    write_json(out/'REPORT.json',dict(split=args.split,metrics=metrics,counts=totals,
        missed_fraction=totals['fn']/max(1,totals['objects']),false_positives_per_frame=totals['fp']/max(1,len(rows)),
        confidence_threshold=.5,iou_threshold=.5,per_frame=rows))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['weights','dataset','out']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--split',choices=['val','test'],default='test');ap.add_argument('--device',default='cuda')
    run(ap.parse_args())
