import json, sys
import numpy as np
from pathlib import Path
ROOT = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker'); sys.path.insert(0, str(ROOT))
from SmallViTAblation.pipeline.koopman_langevin.niezmiennik_czasowy import crop_descriptor, edmd, skill
from SmallViTAblation.pipeline.koopman_langevin.patch_analysis import load_recording
PKG = ROOT/'SmallViTAblation/feature_cache/detector_probe_clips'
rec = load_recording(ROOT/'SmallViTAblation/results/koopman_langevin_epochs/czastki/CZASTKI_b0500')
grid = tuple(rec['meta']['grid']); fids = rec['frame_ids']; at={f:i for i,f in enumerate(fids)}
tracks = json.loads((PKG/'tracks.json').read_text(encoding='utf-8'))
box = np.full((len(fids),4), np.nan)
for clip,t in tracks.items():
    for i,n in enumerate(t['frame']):
        j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
        if j is not None and t['box'][i]: box[j]=t['box'][i]
x = crop_descriptor(rec['profil_red'].astype(np.float64), box, grid, 3)
ok = np.isfinite(x).all(1)&np.isfinite(box).all(1); rows=np.flatnonzero(ok)
video=np.array([f.split('/')[-2] for f in fids]); frame=np.array([int(f.split('/')[-1]) for f in fids],float)
xm=x[rows].astype(np.float64); xm=xm-xm.mean(0)
_,s,vt=np.linalg.svd(xm,full_matrices=False)
base=vt[:48].T/np.maximum(s[:48]/np.sqrt(len(xm)),1e-9); z=xm@base
pairs=[]
for v in np.unique(video[rows]):
    idx=rows[video[rows]==v]; idx=idx[np.argsort(frame[idx])]; pos={r:k for k,r in enumerate(idx)}
    d=np.diff(frame[idx])
    for i in np.flatnonzero(d<=4): pairs.append((pos[idx[i]],pos[idx[i+1]]))
pairs=np.array(pairs)
lam, modes, mu = edmd(z[pairs[:,0]], z[pairs[:,1]])
b=box[rows]; vid=video[rows]
wy=np.log(b[:,2]/np.maximum(b[:,3],1)); wi=np.log(b[:,2]*b[:,3]); px=b[:,0]+b[:,2]/2
c=(z-mu)@modes
print(' nr   Re(lam)  Im(lam)  |lam|   wydluzenie  wielkosc  polozenie')
res=[]
for k in range(48):
    f = np.column_stack([c[:,k].real, c[:,k].imag]) if abs(lam[k].imag)>1e-9 else c[:,[k]].real
    a=skill(f,wy,vid); bb=skill(f,wi,vid); cc=skill(f,px,vid)
    res.append((lam[k], a, bb, cc))
    if k<20 or a>10: print(f'{k:3d} {lam[k].real:9.3f} {lam[k].imag:8.3f} {abs(lam[k]):7.3f} {a:10.1f}% {bb:9.1f}% {cc:9.1f}%')
L=np.array([r[0] for r in res]); A=np.array([r[1] for r in res]); B=np.array([r[2] for r in res]); C=np.array([r[3] for r in res])
def grupa(m,n):
    if m.sum()==0: print(f'{n:34s} brak'); return
    print(f'{n:34s} modow {m.sum():2d}  wydluzenie razem {skill(np.column_stack([c[:,m].real, c[:,m].imag]),wy,vid):6.1f}%  '
          f'wielkosc {skill(np.column_stack([c[:,m].real, c[:,m].imag]),wi,vid):6.1f}%  polozenie {skill(np.column_stack([c[:,m].real, c[:,m].imag]),px,vid):6.1f}%')
print()
grupa(np.abs(L.real-1)<0.15, 'Re blisko 1 (roznica ponizej 0,15)')
grupa((np.abs(L.real-1)>=0.15)&(L.real>0), 'Re dodatnia, ale dalej od 1')
grupa(L.real<0, 'Re ujemna')
grupa(np.abs(L.imag)>1e-9, 'z czescia urojona (obrotowe)')
grupa(np.arange(48)<8, 'osiem pierwszych (dotychczasowy kod)')
print('\nnajlepszy pojedynczy mod dla wydluzenia: nr', int(np.argmax(A)), 'Re', round(float(L[np.argmax(A)].real),3), 'wynik', round(float(A.max()),1))
