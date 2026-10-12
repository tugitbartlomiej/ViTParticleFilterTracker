import json
from pathlib import Path
root=Path('/home2/faculty/bpiotrowski/smallvit_runs')
rows=[]
for run in root.iterdir():
    if not any(s in run.name for s in ['maly_vit','petla43']):continue
    for p in run.rglob('wagi_koncowe.pth'):
        q=p.parent/'OCENY.json'
        if not q.exists():continue
        try:
            scores=json.loads(q.read_text())
            scores=[x for x in scores if isinstance(x,dict) and 'mAP50' in x and x.get('ramie')==p.parent.name]
            if not scores:continue
            last=max(scores,key=lambda x:x.get('krok',0));best=max(scores,key=lambda x:x['mAP50'])
            rows.append(dict(path=str(p),bytes=p.stat().st_size,last=last,best_observed=best,report=str(q)))
        except Exception as e:rows.append(dict(path=str(p),error=str(e)))
print(json.dumps(dict(root=str(root),candidates=rows),ensure_ascii=False))
