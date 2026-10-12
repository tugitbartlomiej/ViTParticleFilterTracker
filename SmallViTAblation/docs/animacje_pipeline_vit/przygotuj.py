"""Przygotowuje niewielkie zasoby filmu z istniejących pomiarów, bez treningu."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
EXP=BASE/'experiments/maly_vit_ruch_czastek_maska_1px'
AS=HERE/'assets'
AS.mkdir(exist_ok=True)
imgs=np.load(EXP/'wyniki/dane/obrazy.npy')
masks=dict(np.load(EXP/'wyniki/dane/maski.npz'))
z=np.load(EXP/'wyniki/dane/latent.npy',mmap_mode='r')
dirs=np.load(EXP/'wyniki/KIERUNKI_RUCHU.npz')
for i in range(len(imgs)):
    Image.fromarray(imgs[i]).save(AS/f'klatka_{i:02d}.png')
    # Maska pozostaje maską, nie modyfikujemy zdjęcia wejściowego.
    Image.fromarray((masks['mask_input'][i]*255).astype('uint8')).save(AS/f'maska_{i:02d}.png')
coord=np.asarray(z[:,0],dtype=float)@dirs['directions'][:,:3]
center=coord.mean((0,1)); coord-=center
scale=np.max(np.abs(coord),axis=(0,1)); coord/=np.maximum(scale,1e-9)
S=np.load(BASE/'experiments/maly_vit_petla_sila_tozsamosci/wyniki_eden/petla_s47/bez/SILA.npy')
assert S.shape[0]>=151
raw=S[:151,0].astype(float)
blocks=raw[1:101].reshape(20,5).mean(1)
def predict(series,model):
    hist=list(series.astype(float)); diff=list(np.diff(hist)); X=[]; targets=[]
    for t in range(4,len(hist)-1):
        row=[diff[t-1-j] for j in range(4)]
        if model=='langevin': row=[hist[t]]+row+[1.0]
        X.append(row); targets.append(diff[t])
    X=np.array(X); targets=np.array(targets)
    gram=X.T@X
    coef=np.linalg.solve(gram+.01*np.diag(np.diag(gram))+1e-9*np.eye(len(gram)),X.T@targets)
    preds=[]
    for _ in range(50):
        row=[diff[-1-j] for j in range(4)]
        if model=='langevin': row=[hist[-1]]+row+[1.0]
        d=float(np.dot(row,coef)); diff.append(d);hist.append(hist[-1]+d);preds.append(hist[-1])
    return dict(values=preds,coefficients=coef.tolist(),fit_samples=len(targets))
forecast={m:predict(raw[1:101],m) for m in ['koopman','langevin']}
for v in forecast.values():
    err=np.asarray(v['values'])-raw[101:151]
    v['rmse']=float(np.sqrt(np.mean(err**2)))
    v['rmse_persistence']=float(np.sqrt(np.mean((raw[100]-raw[101:151])**2)))
report=json.loads((EXP/'wyniki/REPORT.json').read_text(encoding='utf-8'))
meta=json.loads((EXP/'wyniki/dane/metadata.json').read_text(encoding='utf-8'))
payload=dict(coords=coord.tolist(),tool=masks['tool'][0].astype(int).tolist(),
             tool25=masks['tool25'][0].astype(int).tolist(),coverage=masks['coverage'][0].tolist(),
             token_counts=masks['tool'].sum(1).tolist(),steps=meta['steps'],
             raw_history=raw.tolist(),raw_steps=(np.arange(len(raw))*10).tolist(),
             averaged_history=blocks.tolist(),averaged_steps=(np.arange(1,21)*50).tolist(),
             forecast=forecast,
             spectrum=np.cumsum(dirs['eigenvalues'][:64]).__truediv__(dirs['eigenvalues'].sum()).tolist(),
             metrics=report['results']['mask_1px']['summary'])
(AS/'dane.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
sources=[BASE/'experiments/maly_vit_od_zera/model.py',BASE/'experiments/maly_vit_od_zera/dane.py',
         BASE/'experiments/maly_vit_petla_sila_tozsamosci/kod/sila.py',BASE/'experiments/maly_vit_petla_sila_tozsamosci/kod/prognoza.py',
         BASE/'experiments/maly_vit_rownanie_latentu/nagranie.py',BASE/'experiments/maly_vit_rownanie_latentu/WYNIKI_EDEN.md',
         EXP/'wyniki/REPORT.json',EXP/'wyniki/dane/metadata.json',
         BASE/'experiments/maly_vit_ruch_czastek_gesta_historia/README.md']
manifest=[]
for p in sources:
    manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(HERE/'ZRODLA.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Zasoby gotowe. Dane realne:',tuple(z.shape),'historia LDA:',S.shape)
print('Prognoza jednej sondy ze 100 surowych pomiarow:', {m:(v['rmse'],v['rmse_persistence']) for m,v in forecast.items()})
