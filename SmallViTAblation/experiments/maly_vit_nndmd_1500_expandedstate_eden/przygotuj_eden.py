"""Exact1500 provenance + bounded scratch image cache. No GPU, no training."""
from pathlib import Path
import os,json,hashlib,tarfile,time
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path(os.environ.get('EDEN_BASE','/home2/faculty/bpiotrowski'))
SOURCE=BASE/'smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
SCRATCH=Path(os.environ.get('EXPANDED_SCRATCH','/dev/shm/UNCONFIGURED_EXPANDED'))
IMAGES=SCRATCH/'baza1500.npy'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def js(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def prepare():
 import cv2
 start=time.time();manifest=read(HERE/'MANIFEST_1500.json');assert manifest['count']==1500
 coco=BASE/'smallvit_runs/2026-09-28_maly_vit_od_zera/wejscia/pretrain/train/_annotations.coco.json'
 # The complete manifest hashes source pixels, preserving exact source order.
 assert sha(coco)=='3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75'
 SCRATCH.mkdir(parents=True,exist_ok=True)
 js(HERE/'DANE_EDEN.json',dict(status='building',timestamp=stamp(),manifest_sha256=sha(HERE/'MANIFEST_1500.json')))
 names={Path(r['source_path'].replace('\\','/')).name:r for r in manifest['records']};assert len(names)==1500
 arr=np.lib.format.open_memmap(IMAGES,mode='w+',dtype=np.uint8,shape=(1500,320,480,3));found=set()
 archive=BASE/'smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar'
 with tarfile.open(archive,'r:') as tar:
  for member in tar:
   name=Path(member.name).name
   if name not in names or not member.isfile():continue
   assert name not in found,'duplicate tar source';row=names[name];raw=tar.extractfile(member).read()
   assert hashlib.sha256(raw).hexdigest()==row['source_sha256'],name
   image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);assert image is not None
   rgb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
   assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['input_rgb_sha256'],name
   arr[row['index']]=rgb;found.add(name)
 assert len(found)==1500
 arr.flush();del arr
 for cp in manifest['checkpoints']:
  path=SOURCE/'wagi'/('k%04d.pt'%cp['optimizer_step']);assert path.is_file() and sha(path)==cp['sha256']
 js(HERE/'DANE_EDEN.json',dict(status='complete',timestamp=stamp(),count=1500,shape=[1500,320,480,3],dtype='uint8',images_path=str(IMAGES),images_sha256=sha(IMAGES),manifest_sha256=sha(HERE/'MANIFEST_1500.json'),source_coco_sha256=sha(coco),checkpoints_verified=101,source_pixel_hashes_verified=1500,segmentation_masks_available=False,seconds=time.time()-start))
if __name__=='__main__':prepare()
