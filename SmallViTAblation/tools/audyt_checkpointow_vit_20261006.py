import json, sys
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
out=root/'outputs/audyt_checkpointow_vit_20261006'
out.mkdir(parents=True,exist_ok=True)
files=[]
for area in ['results','outputs','output','curated']:
    folder=root/area
    if folder.exists():
        files.extend(p for p in folder.rglob('*.pth') if p.name in ['best.pth','last.pth','best_model.pth','final_model.pth'])
rows=[]
for p in files:
    try:
        ck=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(ck,dict): continue
        scalars={k:v for k,v in ck.items() if isinstance(v,(int,float,str,bool)) or v is None}
        for k in ['metrics','val_metrics','config','args','best_metrics']:
            if isinstance(ck.get(k),dict):
                scalars[k]={a:b for a,b in ck[k].items() if isinstance(b,(int,float,str,bool)) or b is None}
        state=next((ck[k] for k in ['model','model_state_dict','state_dict','net'] if isinstance(ck.get(k),dict)),ck)
        shapes={k:list(v.shape) for k,v in state.items() if isinstance(v,torch.Tensor) and any(s in k for s in ['patch_embed.proj.weight','mlp.fc1.weight','glowa.weight','head.weight'])}
        row=dict(path=str(p),bytes=p.stat().st_size,metadata=scalars,keys=list(ck)[:15],representative_shapes=dict(list(shapes.items())[:5]))
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        del ck,state
    except Exception as e:
        rows.append(dict(path=str(p),error=str(e)[:250]))
(out/'LOKALNE.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('DONE',len(rows))
