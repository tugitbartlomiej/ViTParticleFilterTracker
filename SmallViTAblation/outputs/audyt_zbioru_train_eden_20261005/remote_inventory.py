"""Executed through SSH stdin with python -B. No remote writes/imports of training code."""
import json, tarfile, hashlib, re, math
from pathlib import Path
from collections import Counter

B=Path('/home2/faculty/bpiotrowski/smallvit_runs')
T=B/'2026-09-15_tracking_static36/data/tracking_static36.tar'
W=B/'2026-09-28_maly_vit_od_zera/wejscia'
def norm(s):
    m=re.search(r'(TR_\d{4}_S\d+_P\d+)[/_](\d+)',str(s).replace('\\','/'))
    return m.group(1)+'/'+str(int(m.group(2))) if m else None
def sha(b): return hashlib.sha256(b).hexdigest()
out={'host':'eden-cluster','tar_path':str(T),'input_root':str(W),'remote_writes':False}
with tarfile.open(T,'r:') as t:
    members=t.getmembers()
    out['tar_members']=len(members)
    out['image_members']=[norm(m.name) for m in members if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
    out['image_file_count']=len(out['image_members'])
    out['metadata']={}
    for name in ['split_manifest.json','data_manifest.json','selector_manifest.json','pool_manifest.jsonl']:
        found=[m for m in members if Path(m.name).name==name]
        if len(found)!=1: out['metadata'][name]={'matches':len(found)}; continue
        raw=t.extractfile(found[0]).read()
        if name.endswith('.jsonl'):
            rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
            data=[{'id':norm(r['frame_id']),'split':r['split'],'path':r['path'],'annotations':r.get('annotations',[]),'pixel_sha256':r.get('pixel_sha256')} for r in rows]
        else: data=json.loads(raw)
        out['metadata'][name]={'member':found[0].name,'sha256':sha(raw),'data':data}
out['coco']={}
for split in ['train','valid']:
    p=W/'pretrain'/split/'_annotations.coco.json'; raw=p.read_bytes(); d=json.loads(raw)
    anns={}
    for a in d['annotations']: anns.setdefault(a['image_id'],[]).append(a)
    out['coco'][split]={'path':str(p),'sha256':sha(raw),'categories':d.get('categories'), 'images':[{'id':norm(i['file_name']),'file_name':i['file_name'],'width':i.get('width'),'height':i.get('height'),'annotations':anns.get(i['id'],[])} for i in d['images']]}
p=W/'index/manifest.jsonl'; raw=p.read_bytes()
out['index']={'path':str(p),'sha256':sha(raw),'rows':[{'id':norm(r['frame_id']),'split':r['split'],'annotations':r.get('annotations',[])} for r in (json.loads(l) for l in raw.splitlines() if l.strip())]}
out['source_checksums']={}
for p in [B/'2026-10-01_maly_vit_ruch_czastek_gesta_historia/bieg_eden.slurm',B/'2026-10-01_maly_vit_ruch_czastek_gesta_historia/kod/dane.py',B/'2026-10-01_maly_vit_ruch_czastek_gesta_historia/kod/bieg.py']:
    out['source_checksums'][str(p)]=sha(p.read_bytes())
print(json.dumps(out,separators=(',',':')))
