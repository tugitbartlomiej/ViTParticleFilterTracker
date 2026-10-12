"""Exact full-population density histograms in a single fixed PCA basis."""
import hashlib,json,functools
from pathlib import Path
import numpy as np
from dense_timeline import DenseTimeline,COUNT
ROOT=Path(__file__).resolve().parents[2]/'outputs/vit_k1000_taniec_czasteczek_20261009/data'
TARGET=ROOT/'global_density.npz'
PLANES=((0,1),(0,2),(1,2))
G=64

def histogram(xyz,axes,bounds):
    a,b=axes
    h=np.histogram2d(xyz[:,b],xyz[:,a],bins=G,range=[bounds[b],bounds[a]])[0]
    if int(h.sum())!=len(xyz):raise ValueError('Punkty poza wspólnymi osiami')
    return h

def smooth(h):
    h=np.asarray(h,dtype=np.float64)
    w=np.exp(-np.arange(-3,4,dtype=float)**2/3.125);w/=w.sum()
    for axis in (-1,-2):
        pads=[(0,0)]*h.ndim;pads[axis]=(3,3);p=np.pad(h,pads)
        result=np.zeros_like(h)
        for j,weight in enumerate(w):
            sl=[slice(None)]*h.ndim;sl[axis]=slice(j,j+h.shape[axis]);result+=weight*p[tuple(sl)]
        h=result
    return h

def build():
    dense=DenseTimeline(ROOT/'dense');m=dense.load()
    if not m:raise ValueError('Brak kompletnego zweryfikowanego zapisu')
    bounds=np.asarray(m['bounds']['natural']);counts=np.empty((3,len(m['steps']),G,G),np.uint32)
    for t,step in enumerate(m['steps']):
        xyz=np.frombuffer(dense.binary(step),dtype='<f4').reshape(COUNT,4)[:,:3]
        for plane,axes in enumerate(PLANES):counts[plane,t]=histogram(xyz,axes,bounds)
        if t%10==0:print(f'{t+1}/{len(m["steps"])} · {COUNT} punktów/stany',flush=True)
    provenance={'point_count':COUNT,'image_count':1500,'steps':m['steps'],'grid':G,
                'pca_basis_sha256':m['pca_basis_sha256'],'source_manifest_sha256':hashlib.sha256((ROOT/'dense/manifest.json').read_bytes()).hexdigest(),
                'sampling':False,'interpolation':False,'density':'all particles; histogram y rows, x columns; fixed natural PCA bounds'}
    with TARGET.with_suffix('.part').open('wb') as f:np.savez_compressed(f,counts=counts,bounds=bounds,provenance=np.array(json.dumps(provenance)))
    TARGET.with_suffix('.part').replace(TARGET)
    print('COMPLETE',TARGET,flush=True)

@functools.lru_cache(maxsize=3)
def payload(plane):
    if plane not in range(3):raise ValueError('Nieprawidłowa płaszczyzna')
    with np.load(TARGET,allow_pickle=False) as f:
        p=json.loads(str(f['provenance']));bounds=f['bounds'];h=f['counts'][plane].astype(float)
    source=ROOT/'dense/manifest.json'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=p['source_manifest_sha256']:raise ValueError('Niezgodne źródło gęstości')
    if not np.all(h.sum(axis=(-1,-2))==COUNT):raise ValueError('Niepełna populacja histogramu')
    field=smooth(h);peak=float(field.max());z=-np.log((field+.02)/(peak+.02))
    total=smooth(h.sum(axis=0));whole=-np.log((total+.02)/(float(total.max())+.02))
    a,b=PLANES[plane]
    def centers(axis):lo,hi=bounds[axis];return (lo+(np.arange(G)+.5)*(hi-lo)/G).tolist()
    return {**p,'plane':plane,'x':centers(a),'y':centers(b),'fields':z.round(4).tolist(),
            'whole':whole.round(4).tolist(),'zmax':float(max(z.max(),whole.max())),
            'whole_observation_count':int(COUNT*len(p['steps'])),'peak_density':peak}

def request(query):
    if set(query)!={'terrain','plane'} or query['terrain']!=['global'] or len(query['plane'])!=1 or query['plane'][0] not in ('0','1','2'):raise ValueError('Wymagane terrain=global i plane=0..2')
    return payload(int(query['plane'][0]))

if __name__=='__main__':build()
