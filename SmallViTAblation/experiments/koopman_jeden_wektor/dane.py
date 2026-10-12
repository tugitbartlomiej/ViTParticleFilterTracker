"""Same-image checkpoint states; transformed images are not independent groups.

Only eight spatial positions are cached, each with ALL 384 coordinates.
Position selection uses fit masks only. This is explicitly a token diagnostic,
not a compressed replacement for the full-image operator.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'koopman_gesta_historia/wyniki/macierze_wiele_obrazow'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacing an open status file.
    for attempt in range(9):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==8:raise
            time.sleep(.05*(attempt+1))

def metadata():
    cfg=json.loads((SOURCE/'CONFIG.json').read_text())
    rows=cfg['rows'];steps=[w['step'] for w in cfg['weights']]
    groups=sorted({r['group_id'] for r in rows if r['role']=='fit'})
    inner=np.array([i for i,r in enumerate(rows) if r['role']=='fit' and r['group_id'] in groups[:6]])
    val=np.array([i for i,r in enumerate(rows) if r['role']=='fit' and r['group_id'] in groups[6:]])
    fit=np.array([i for i,r in enumerate(rows) if r['role']=='fit'])
    hold=np.array([i for i,r in enumerate(rows) if r['role']=='holdout'])
    assert (len(inner),len(val),len(fit),len(hold))==(36,12,48,24)
    assert set(rows[i]['group_id'] for i in fit).isdisjoint(rows[i]['group_id'] for i in hold)
    mask=np.load(SOURCE/'MASK_WEIGHTS.npy')
    coverage=mask[fit].mean(0)
    tokens=np.argsort(-coverage,kind='stable')[:8]
    return cfg,rows,steps,inner,val,fit,hold,mask,tokens

def load_candidates(steps,tokens,status):
    data=np.empty((len(steps),72,len(tokens),384),dtype=np.float32)
    proofs=[]
    for j,step in enumerate(steps):
        p=SOURCE/f'TOKENS_{step:07d}.npy'
        expected=json.loads(p.with_suffix('.json').read_text())['sha256']
        actual=sha(p);assert actual==expected,f'Bad SHA256: {p}'
        a=np.load(p,mmap_mode='r');assert a.shape==(72,576,384)
        data[j]=a[:,tokens,:]
        proofs.append(dict(step=step,sha256=actual))
        status('reading_candidates',step=step)
    return data,proofs
