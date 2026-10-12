"""Read annotated archive images, preserving canonical film splits."""
import ast,csv,io,json,zipfile,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch,cv2
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[2]
SPLIT=ROOT/'feature_cache/eden_static36/data/tracking_static36/split_manifest.json'
ARCHIVES=ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
PRE=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
def inventory(path,split):
    rows=[]
    with zipfile.ZipFile(path) as z:
        images={};seen=set()
        for n in z.namelist():
            if n.lower().endswith(('.jpg','.jpeg','.png')):images.setdefault(Path(n).name,[]).append(n)
        annotations=[n for n in z.namelist() if Path(n).name=='annotation.csv']
        if len(annotations)!=1:raise ValueError(f'Invalid annotation source {path}')
        raw=z.read(annotations[0]);digest=hashlib.sha256(raw).hexdigest()
        for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            members=images.get(r.get('filename',''),[])
            if len(members)!=1 or members[0] in seen:continue
            boxes=[]
            for k,v in r.items():
                if k and k.endswith('_bbox') and k not in ['cornea_bbox','pupil_bbox']:
                    try:
                        b=list(map(float,ast.literal_eval(v)))
                        if len(b)==4 and np.isfinite(b).all() and min(b[:2])>=0 and min(b[2:])>0:boxes.append(b)
                    except (ValueError,TypeError,SyntaxError):pass
            if not boxes:continue
            try:frame=int(r['frame_number']);assert frame==int(Path(r['filename']).stem.rsplit('_',1)[-1])
            except (ValueError,KeyError,AssertionError):continue
            seen.add(members[0]);info=z.getinfo(members[0])
            rows.append(dict(id=path.stem+'/'+str(frame),film=path.stem,split=split,frame=frame,archive=str(path),member=members[0],boxes=boxes,annotation_sha256=digest,crc=info.CRC))
    return sorted(rows,key=lambda r:r['frame'])

def sample(rows,count):
    films=sorted({r['film'] for r in rows});by={f:[r for r in rows if r['film']==f] for f in films};chosen=[]
    quotas={f:0 for f in films}
    while sum(quotas.values())<min(count,len(rows)):
        for f in films:
            if quotas[f]<len(by[f]) and sum(quotas.values())<count:quotas[f]+=1
    for f in films:
        ix=np.linspace(0,len(by[f])-1,quotas[f],dtype=int)
        chosen.extend(by[f][i] for i in ix)
    return chosen

class Reader:
    def __init__(self):self.archives=OrderedDict()
    def read(self,r):
        p=r['archive']
        if p not in self.archives:self.archives[p]=zipfile.ZipFile(p)
        self.archives.move_to_end(p)
        while len(self.archives)>8:self.archives.popitem(last=False)[1].close()
        z=self.archives[p];assert z.getinfo(r['member']).CRC==r['crc']
        with Image.open(io.BytesIO(z.read(r['member']))) as im:im.load();im=im.convert('RGB')
        w,h=im.size
        if any(x+bw>w+1e-4 or y+bh>h+1e-4 for x,y,bw,bh in r['boxes']):raise ValueError('Annotation outside image')
        a=cv2.resize(np.asarray(im),(480,320),interpolation=cv2.INTER_AREA)
        vit=torch.from_numpy(a).permute(2,0,1).float()/255
        return vit,PRE(im)
    def close(self):
        for z in self.archives.values():z.close()
        self.archives.clear()

def order(rows,seed):
    # Shuffle temporal rows, then blocks, reducing ZIP I/O without discarding data.
    rng=np.random.default_rng(seed);by={}
    for i,r in enumerate(rows):by.setdefault(r['film'],[]).append(i)
    blocks=[]
    for ix in by.values():
        ix=rng.permutation(ix)
        blocks.extend(ix[j:j+64] for j in range(0,len(ix),64))
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
