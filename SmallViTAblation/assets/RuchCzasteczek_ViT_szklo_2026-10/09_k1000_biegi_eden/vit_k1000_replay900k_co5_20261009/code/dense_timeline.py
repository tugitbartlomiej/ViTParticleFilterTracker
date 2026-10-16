"""Read only verified genuine dense states, never interpolate missing steps."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np

COUNT=900000
BASIS_SHA='8d3cb28ad2b1581efeca89fab659b9d6fa518bffe4d452b64c68ea7dcd1b9cca'

class DenseTimeline:
    def __init__(self,root):self.root=Path(root);self.manifest=None;self.signature=None
    def load(self):
        p=self.root/'manifest.json'
        if not p.exists():return None
        signature=(p.stat().st_mtime_ns,p.stat().st_size)
        if signature==self.signature:return self.manifest
        m=json.loads(p.read_text(encoding='utf-8'))
        if m.get('status')!='complete':return None
        if m.get('point_count')!=COUNT or m.get('pca_basis_sha256')!=BASIS_SHA:raise ValueError('Niezgodna tożsamość osi lub liczba cząsteczek')
        steps=m['steps'];stride=m['stride']
        if not isinstance(stride,int) or not 1<=stride<=10 or 500%stride or steps!=list(range(0,501,stride)):raise ValueError('Niepełna rzeczywista oś kroków 0..500')
        if m.get('interpolation') is not False:raise ValueError('Gęsty zapis nie może być interpolacją')
        for step in steps:
            record=m['files'][str(step)];path=self.path(record['name'])
            if record['bytes']!=COUNT*16 or not path.is_file() or path.stat().st_size!=COUNT*16:raise ValueError(f'Niekompletny stan {step}')
            if str(step) not in m['stats']:raise ValueError('Brak rzeczywistych statystyk')
        self.manifest=m;self.signature=signature;self.binary.cache_clear();return m
    def path(self,name):
        p=(self.root/name).resolve();p.relative_to(self.root.resolve())
        if Path(name).name!=name or p.suffix!='.bin':raise ValueError('Niewłaściwa ścieżka stanu')
        return p
    @lru_cache(maxsize=4)
    def binary(self,step):
        m=self.load()
        if m is None or step not in m['steps']:raise ValueError('Brak zweryfikowanego rzeczywistego kroku')
        record=m['files'][str(step)];blob=self.path(record['name']).read_bytes()
        if len(blob)!=COUNT*16 or hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError(f'Niezgodna suma zapisu kroku {step}')
        if not np.isfinite(np.frombuffer(blob,dtype='<f4')).all():raise ValueError('Niefinitywne położenia')
        return blob
    def frame(self,step,movement,image=0,geometry=False):
        m=self.load()
        if m is None or step not in m['steps']:raise ValueError('Brak kroku w gęstym zapisie')
        stats=m['stats'][str(step)]
        result={**stats,'step':step,'mode':'global','image':image,'count':COUNT,'total':COUNT,'xyz':[],'norm':[],'images':[],'tokens':[],'lines':[],'bar_tokens':[],'bar_norm':[], 'bounds':m['bounds']['movement' if movement else 'natural'],'norm_scale':m['norm_scale'],'heat_scale':m['heat_scale'],'selected_image_count':1500,'selected_both90':stats['both90']}
        if geometry:
            packed=np.frombuffer(self.binary(step),dtype='<f4').reshape(COUNT,4)
            idx=np.random.default_rng(20261009).choice(COUNT,8000,replace=False)
            xyz=packed[idx,:3].copy()
            if movement:xyz-=np.frombuffer(self.binary(0),dtype='<f4').reshape(COUNT,4)[idx,:3]
            result.update(xyz=xyz.astype(float).round(5).tolist(),norm=packed[idx,3].astype(float).round(5).tolist(),images=(idx//600).tolist(),tokens=(idx%600).tolist(),count=len(idx))
        return result
