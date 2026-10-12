"""Testuje exact1500/all600/all768 provenance; niekompletne cache lub rozjazd źródła obalają zgodność."""
from pathlib import Path
import json,hashlib,numpy as np,cv2,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
BASE=EXP/'maly_vit_od_zera'
SOURCE=EXP/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
COCO=BASE/'eden/wejscia/pretrain/train/_annotations.coco.json'
IMAGES=BASE/'cache/baza1500.npy'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stamp():return datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
def frameid(path):
 s=Path(path.replace(chr(92),'/')).stem
 return 'cataract_lmm/tracking/'+s[:14]+'/'+s[15:]
def prepare():
 start=time.time();c=read(COCO);X=np.load(IMAGES,mmap_mode='r');assert X.shape==(1500,320,480,3) and X.dtype==np.uint8
 by={}
 for ann in c['annotations']:by.setdefault(ann['image_id'],[]).append(ann)
 records=[]
 for i,row in enumerate(c['images']):
  path=Path(row['file_name']);assert path.is_file(),path
  raw=cv2.imread(str(path));assert raw is not None
  rgb=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA);assert np.array_equal(rgb,X[i]),i
  fid=frameid(str(path));records.append(dict(index=i,coco_image_id=row['id'],frame_id=fid,film=fid.split('/')[-2],frame=fid.split('/')[-1],split='train',source_path=str(path),source_sha256=sha(path),input_rgb_sha256=hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest(),annotations=by.get(row['id'],[]),bbox_format='source native XYWH',segmentation_available=any('segmentation' in a for a in by.get(row['id'],[])),cache_pixels_match_source=True))
 assert len(records)==len({r['frame_id'] for r in records})==1500
 checkpoints=[]
 for slot,step in enumerate(range(0,1001,10)):
  p=SOURCE/'wagi'/f'k{step:04d}.pt';assert p.is_file()
  checkpoints.append(dict(index=slot,optimizer_step=step,path=str(p),sha256=sha(p)))
 arch=EXP/'maly_vit_ruch_czastek_gesta_historia/kod/model.py';local=HERE/'model_snapshot.py'
 if not local.exists():local.write_bytes(arch.read_bytes())
 assert sha(local)==sha(arch)
 old=read(SOURCE/'dane/images.json');oldX=np.load(SOURCE/'dane/obrazy.npy');index={x['frame_id']:x['index'] for x in records};matched=[]
 for j,row in enumerate(old):
  if row['frame_id'] in index:
   i=index[row['frame_id']];assert np.array_equal(X[i],oldX[j]);matched.append(dict(index1500=i,source_probe_index=j,frame_id=row['frame_id']))
 js(HERE/'MANIFEST_1500.json',dict(status='verified',count=1500,records=records,source_coco=str(COCO),source_coco_sha256=sha(COCO),imagecache=str(IMAGES),imagecache_sha256=sha(IMAGES),imagecache_shape=list(X.shape),exact_source_order=True,checkpoint_count=101,checkpoints=checkpoints,model_arch_sha256=sha(local),matched_source_probes=matched,segmentation_available=False,all600_loss_weights='1/600 within each image, each image equal',created=stamp(),seconds=time.time()-start))
 js(HERE/'RESOURCE_AUDIT.json',dict(created=stamp(),free_disk_bytes=shutil.disk_usage(HERE).free,full_dump_bytes=279244800000,window11_bytes=30412800000,cache_strategy='11 modulo slots; reuse only current complete checkpoint',image_microbatch=4,no_existing_data_deleted=True))
 print('VERIFIED1500',len(matched),'source probes',round(time.time()-start,2),flush=True)
if __name__=='__main__':prepare()
