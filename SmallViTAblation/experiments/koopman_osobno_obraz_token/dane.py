"""Independent trajectories: fixed image and token position, time along checkpoints.

Images include correlated variants. Their operators remain distinct; counts
are descriptive, not independent statistical replications. Float32 is the
original stored precision; model fitting promotes all coordinates to float64.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'koopman_gesta_historia/wyniki/macierze_wiele_obrazow'
OUT=HERE/'wyniki'

def write(p,data):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(9):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==8:raise
            time.sleep(.05*(attempt+1))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def metadata():
    cfg=json.loads((SOURCE/'CONFIG.json').read_text())
    steps=[r['step'] for r in cfg['weights']]
    if steps!=list(range(5,370,5)):raise ValueError('Source grid changed; protocol needs review')
    if len(cfg['rows'])!=72:raise ValueError('Expected 72 images')
    return cfg,steps,np.load(SOURCE/'MASK_WEIGHTS.npy')

def load_all(steps,status):
    data=np.empty((72,576,73,384),dtype=np.float32)
    proofs=[]
    for t,step in enumerate(steps):
        p=SOURCE/f'TOKENS_{step:07d}.npy';expected=json.loads(p.with_suffix('.json').read_text())['sha256']
        actual=sha(p)
        if actual!=expected:raise RuntimeError(f'SHA256 mismatch {p}')
        a=np.load(p,mmap_mode='r')
        if a.shape!=(72,576,384) or not np.isfinite(a).all():raise ValueError(f'Invalid full tensor {p}')
        data[:,:,t,:]=a
        proofs.append(dict(step=step,sha256=actual))
        if t%8==0 or t==72:status('loading',step=step)
    write(OUT/'INPUT_SHA256.json',proofs)
    return data

def array(name,shape,dtype='float64'):
    p=OUT/(name+'.npy')
    if p.exists():
        a=np.load(p,mmap_mode='r+')
        if a.shape!=shape or a.dtype!=np.dtype(dtype):raise ValueError(f'Wrong saved array {name}')
        return a
    if next(OUT.glob('IMAGE_*_DONE.json'),None) is not None:
        raise RuntimeError(f'Missing {name} despite completed-image markers; refusing unsafe resume')
    return np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
