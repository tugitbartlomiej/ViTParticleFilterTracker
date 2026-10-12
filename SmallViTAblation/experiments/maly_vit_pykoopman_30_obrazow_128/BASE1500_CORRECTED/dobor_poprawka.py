import sys,copy,collections,time,json
from pathlib import Path
import numpy as np,cv2
OLD=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD))
from dobor import read,sha,js,BASE,PAIRS,fid
HERE=Path(__file__).resolve().parent
m=read(OLD/'MANIFEST.json');coco=BASE/'eden/wejscia/pretrain/train/_annotations.coco.json'
index={fid(x['file_name']):i for i,x in enumerate(read(coco)['images'])};assert len(index)==1500
replace=[i for i in m['fit_indices'] if m['records'][i]['frame_id'] not in index];assert replace==[0,2,3,5,6]
films={r['film'] for r in m['records']};ids={r['frame_id'] for r in m['records']}
rows=read(PAIRS/'manifest.json');pm=dict(np.load(PAIRS/'TOKEN_MASKI.npz'))
candidates=[(j,r) for j,r in enumerate(rows) if r['frame_id'] in index and r['frame_id'] not in ids and r['video_id'] not in films]
rng=np.random.default_rng(202610021);tie={r['frame_id']:float(rng.random()) for j,r in sorted(candidates,key=lambda a:a[1]['frame_id'])};chosen=[];used=set();bins=collections.Counter()
while len(chosen)<5:
 j,r=min([(j,r) for j,r in candidates if r['video_id'] not in used],key=lambda a:(bins[a[1]['angle_bin']],a[1]['axis_uncertain'],tie[a[1]['frame_id']]))
 chosen.append((j,r));used.add(r['video_id']);bins[r['angle_bin']]+=1
chosen.sort(key=lambda a:(a[1]['angle_bin'],a[1]['video_id'],a[1]['frame_index']))
X=np.load(OLD/'OBRAZY.npy');mask=dict(np.load(OLD/'MASKI.npz'));cache=np.load(BASE/'cache/baza1500.npy',mmap_mode='r');records=copy.deepcopy(m['records']);(HERE/'obrazy').mkdir(exist_ok=True);(HERE/'maski').mkdir(exist_ok=True);mapping=[]
for slot,(j,r) in zip(replace,chosen):
 bi=index[r['frame_id']];X[slot]=cache[bi]
 image=cv2.imread(str(PAIRS/r['image_path']));expected=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA);assert np.array_equal(expected,X[slot])
 area=cv2.resize(cv2.imread(str(PAIRS/r['mask_native_path']),0).astype(np.float32)/255.,(480,320),interpolation=cv2.INTER_AREA)
 for k in mask:
  if k=='tool':mask[k][slot]=pm['tool'][j]
  elif k=='background':mask[k][slot]=pm['background'][j]
  elif k=='coverage':mask[k][slot]=pm['tool_coverage'][j]
  elif k=='area_coverage':mask[k][slot]=pm['area_coverage'][j]
  elif k=='mask_input_area':mask[k][slot]=area
  elif k=='mask_input':mask[k][slot]=cv2.imread(str(PAIRS/r['mask_480_path']),0)>0
  elif k=='tool25':mask[k][slot]=pm['tool25'][j]
  elif k!='token_xy':raise KeyError(k)
 record=dict(r,index=slot,role='fit',film=r['video_id'],base1500_index=bi,new_selection=True,selection_candidate_index400=j,replaced_old_frame_id=records[slot]['frame_id'],source_image_absolute=str(PAIRS/r['image_path']),source_mask_absolute=str(PAIRS/r['mask_480_path']),tool_token_count=int(mask['tool'][slot].sum()))
 records[slot]=record
for i,record in enumerate(records):
 ip=HERE/'obrazy'/f'{i:02d}.png';mp=HERE/'maski'/f'{i:02d}.png';cv2.imwrite(str(ip),cv2.cvtColor(X[i],cv2.COLOR_RGB2BGR));cv2.imwrite(str(mp),mask['mask_input'][i].astype(np.uint8)*255)
 record.update(input_png=str(ip),input_png_sha256=sha(ip),input_mask_png=str(mp),input_mask_png_sha256=sha(mp),base1500_index=index.get(record['frame_id']),base1500_member=record['frame_id'] in index)
 mapping.append(dict(index=i,path=str(HERE/'LATENT_NOWE5_768.npy') if i in replace else str(OLD/'LATENT_768.npy'),image_index=replace.index(i) if i in replace else i))
assert all(records[i]['base1500_member'] for i in m['fit_indices']);assert len({r['frame_id'] for r in records})==34
assert [records[i]['frame_id'] for i in m['holdout_indices']]==[m['records'][i]['frame_id'] for i in m['holdout_indices']]
np.save(HERE/'OBRAZY.npy',X);np.savez_compressed(HERE/'MASKI.npz',**mask)
with np.load(OLD/'KANALY_J128.npz') as z:np.savez(HERE/'KANALY_J128.npz',**dict(z))
js(HERE/'MANIFEST.json',dict(count=34,fit_count=30,fit_base1500_count=30,holdout_count=4,fit_indices=m['fit_indices'],holdout_indices=m['holdout_indices'],replaced_indices=replace,records=records,cache_sources=mapping,selection_seed=202610021,base_coco_sha256=sha(coco),source_manifest_sha256=sha(OLD/'MANIFEST.json'),holdout_membership=[dict(index=i,frame_id=records[i]['frame_id'],base1500_member=records[i]['base1500_member'],base1500_index=records[i]['base1500_index']) for i in m['holdout_indices']]))
js(HERE/'DOBOR_AUDYT.json',dict(status='complete',fit_base1500_count=30,replaced_indices=replace,new_films=list(sorted(used)),excluded_films=list(sorted(films)),pixel_matches=True,base_cache_sha256=sha(BASE/'cache/baza1500.npy'),images_sha256=sha(HERE/'OBRAZY.npy'),masks_sha256=sha(HERE/'MASKI.npz'),manifest_sha256=sha(HERE/'MANIFEST.json')))
print('PREPARED corrected30fit; replace',replace,flush=True)
