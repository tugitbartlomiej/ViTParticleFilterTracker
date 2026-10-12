"""Deterministyczne 30 fit + 4 identyczne source holdout; bez latentów/fitu."""
from pathlib import Path
import numpy as np,json,hashlib,cv2,shutil,time,collections
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
SOURCE=EXP/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
BASE=EXP/'maly_vit_od_zera';PAIRS=EXP/'maly_vit_test_orientacji_narzedzia_400'
JROOT=EXP/'maly_vit_tozsamosc90_bez_pca_langevin'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(p,v):
 def convert(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,Path):return str(x)
  raise TypeError(type(x))
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')
def fid(path):
 s=Path(path.replace(chr(92),'/')).stem
 return 'cataract_lmm/tracking/'+s[:14]+'/'+s[15:]
def main():
 start=time.time();assert not (HERE/'MANIFEST.json').exists()
 coco_path=BASE/'eden/wejscia/pretrain/train/_annotations.coco.json';coco=read(coco_path);base_ids=[fid(r['file_name']) for r in coco['images']]
 assert len(base_ids)==len(set(base_ids))==1500
 index={s:i for i,s in enumerate(base_ids)};cache=np.load(BASE/'cache/baza1500.npy',mmap_mode='r');assert cache.shape==(1500,320,480,3) and cache.dtype==np.uint8
 old=read(SOURCE/'dane/images.json');oldx=np.load(SOURCE/'dane/obrazy.npy');oldm=dict(np.load(SOURCE/'dane/maski.npz'))
 sourcehold=[r['frame_id'] for r in old[8:12]];forbidden={r['film'] for r in old};oldids={r['frame_id'] for r in old}
 rows=read(PAIRS/'manifest.json');pm=dict(np.load(PAIRS/'TOKEN_MASKI.npz'))
 candidates=[(j,r) for j,r in enumerate(rows) if r['frame_id'] in index and r['frame_id'] not in oldids and r['video_id'] not in forbidden]
 seed=20261002;rng=np.random.default_rng(seed);tie={r['frame_id']:float(rng.random()) for _,r in sorted(candidates,key=lambda a:a[1]['frame_id'])}
 chosen=[];used=set();bins=collections.Counter()
 while len(chosen)<22:
  possible=[(j,r) for j,r in candidates if r['video_id'] not in used]
  if not possible:raise RuntimeError('insufficient unique films')
  j,r=min(possible,key=lambda a:(bins[a[1]['angle_bin']],a[1]['axis_uncertain'],tie[a[1]['frame_id']]))
  chosen.append((j,r));used.add(r['video_id']);bins[r['angle_bin']]+=1
 chosen.sort(key=lambda a:(a[1]['angle_bin'],a[1]['video_id'],a[1]['frame_index']))
 X=np.empty((34,320,480,3),np.uint8);X[:12]=oldx;out={k:np.empty((34,)+v.shape[1:],v.dtype) for k,v in oldm.items() if v.shape[0]==12};manifest=[]
 for k,v in out.items():v[:12]=oldm[k]
 (HERE/'obrazy').mkdir(exist_ok=True);(HERE/'maski').mkdir(exist_ok=True)
 for i,r in enumerate(old):
  record=dict(r,index=i,role='holdout' if i in (8,9,10,11) else 'fit',preserved_source_index=i,base1500_index=index.get(r['frame_id']),new_selection=False,input_origin=str(SOURCE/'dane/obrazy.npy'))
  manifest.append(record)
 pixel_checks=[]
 for i,(j,r) in enumerate(chosen,12):
  bi=index[r['frame_id']];X[i]=cache[bi]
  image=cv2.imread(str(PAIRS/r['image_path']));assert image is not None
  expected=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
  diff=int(np.max(np.abs(expected.astype(int)-X[i].astype(int))));pixel_checks.append(dict(index=i,maxabs_cache_vs_original_resize=diff))
  assert diff==0,'cache versus source pixels differ'
  mask=cv2.imread(str(PAIRS/r['mask_480_path']),0)>0
  area=cv2.resize(cv2.imread(str(PAIRS/r['mask_native_path']),0).astype(np.float32)/255.,(480,320),interpolation=cv2.INTER_AREA)
  for k in out:
   if k=='tool':out[k][i]=pm['tool'][j]
   elif k=='background':out[k][i]=pm['background'][j]
   elif k=='coverage':out[k][i]=pm['tool_coverage'][j]
   elif k=='area_coverage':out[k][i]=pm['area_coverage'][j]
   elif k=='mask_input_area':out[k][i]=area
   elif k=='mask_input':out[k][i]=mask
   elif k=='tool25':out[k][i]=pm['tool25'][j]
   else:raise KeyError('unhandled original mask field '+k)
  record=dict(r,index=i,role='fit',film=r['video_id'],preserved_source_index=None,base1500_index=bi,new_selection=True,selection_candidate_index400=j,input_origin=str(BASE/'cache/baza1500.npy'),source_image_absolute=str(PAIRS/r['image_path']),source_mask_absolute=str(PAIRS/r['mask_480_path']))
  manifest.append(record)
 for i,r in enumerate(manifest):
  ip=HERE/'obrazy'/f'{i:02d}.png';mp=HERE/'maski'/f'{i:02d}.png';assert cv2.imwrite(str(ip),cv2.cvtColor(X[i],cv2.COLOR_RGB2BGR));assert cv2.imwrite(str(mp),out['mask_input'][i].astype(np.uint8)*255)
  r.update(input_png=str(ip.relative_to(HERE)),input_png_sha256=sha(ip),input_mask_png=str(mp.relative_to(HERE)),input_mask_png_sha256=sha(mp),tool_token_count=int(out['tool'][i].sum()),background_token_count=int(out['background'][i].sum()))
 out['token_xy']=oldm['token_xy'];np.save(HERE/'OBRAZY.npy',X);np.savez_compressed(HERE/'MASKI.npz',**out)
 with np.load(JROOT/'KANALY_MODEL.npz') as z:J=z['channels0'].copy()
 np.savez_compressed(HERE/'KANALY_J128.npz',channels0=J,channels1=J+1)
 assert len(manifest)==len({r['frame_id'] for r in manifest})==34
 assert [manifest[i]['frame_id'] for i in (8,9,10,11)]==sourcehold
 assert sum(r['role']=='fit' for r in manifest)==30
 js(HERE/'MANIFEST.json',dict(count=34,fit_count=30,holdout_count=4,fit_indices=list(range(8))+list(range(12,34)),holdout_indices=[8,9,10,11],seed=seed,records=manifest))
 js(HERE/'DOBOR_AUDYT.json',dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),base_count=1500,base_coco_sha256=sha(coco_path),base_cache_sha256=sha(BASE/'cache/baza1500.npy'),candidate_count=len(candidates),candidate_films=len(set(r['video_id'] for _,r in candidates)),new_count=22,new_films=len(used),new_angle_bins=dict(bins),excluded_films=sorted(forbidden),holdout_ids=sourcehold,pixel_checks=pixel_checks,old_images_exact=True,old_masks_exact=True,channels_source_sha256=sha(JROOT/'KANALY_MODEL.npz'),manifest_sha256=sha(HERE/'MANIFEST.json'),seconds=time.time()-start))
 print('PREPARED34:30fit4holdout',time.time()-start,flush=True)
if __name__=='__main__':main()
