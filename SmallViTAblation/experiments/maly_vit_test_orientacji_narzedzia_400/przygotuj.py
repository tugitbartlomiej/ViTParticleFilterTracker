"""Czy400 rzeczywistych train ma zgodne maski i zróżnicowane osie?
Brak400 poprawnych unikalnych train lub niezgodność źródła przerywa export.
Nie zmienia źródeł ani nie wytwarza syntetyków i nie wywołuje modelu.
"""
from pathlib import Path
import csv,io,json,zipfile,hashlib,collections,time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
POOL=ROOT/'feature_cache/lokalny_static36/pool_manifest.jsonl'
ZIPS=ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SEED=20261001

def sha(b):return hashlib.sha256(b).hexdigest()
def js(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def raster(polygons,w,h):
    mask=np.zeros((h,w),np.uint8)
    for p in polygons:
        pts=np.asarray(p,float).reshape(-1,2)
        if len(pts)<3 or not np.isfinite(pts).all() or (pts<0).any() or (pts[:,0]>=w).any() or (pts[:,1]>=h).any():raise ValueError('invalid polygon')
        cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],1)
    if mask.sum()<30:raise ValueError('small mask')
    return mask

def orientation(mask):
    m=cv2.moments(mask,True);area=m['m00'];center=[m['m10']/area,m['m01']/area]
    cov=np.array([[m['mu20'],m['mu11']],[m['mu11'],m['mu02']]])/area
    eig,V=np.linalg.eigh(cov);vec=V[:,-1];ratio=float(eig[-1]/max(eig[0],1e-9))
    return float(np.degrees(np.arctan2(vec[1],vec[0]))%180),ratio,center

def coverage(mask):return mask.reshape(20,16,30,16).mean((1,3),dtype=np.float32).ravel()

def main():
    start=time.time();HERE.mkdir(parents=True,exist_ok=True)
    allrows=[json.loads(x) for x in POOL.read_text().splitlines()]
    assert len(set(r['frame_id'] for r in allrows))==len(allrows)
    train=[r for r in allrows if r['split']=='train' and r['annotations']]
    groups=collections.defaultdict(dict)
    for r in train:groups[r['video_id']][int(r['frame_index'])]=r
    candidates=[];reject=collections.Counter();source={};ann_cache={}
    for video,frames in sorted(groups.items()):
        archive=ZIPS/(video+'.zip')
        if not archive.exists():reject['missing_archive']+=len(frames);continue
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith('/annotation.csv')];assert len(names)==1
            member=names[0];raw=z.read(member);source[video]=dict(archive=str(archive),annotation_member=member,annotation_csv_sha256=sha(raw))
            seen=set()
            for a in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
                frame=int(a['frame_number'])
                if frame not in frames:continue
                assert frame not in seen;seen.add(frame);r=frames[frame]
                try:
                    polygons=json.loads(a.get('cystotome_seg') or '[]');w=r['native_width'];h=r['native_height']
                    mask=raster(polygons,w,h);angle,ratio,center=orientation(mask)
                    ys,xs=np.nonzero(mask);bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
                    given=json.loads(a['cystotome_bbox']);assert np.allclose(given,r['annotations'][0]['bbox'])
                    x,y,bw,bh=given
                    if xs.min()<x-2 or ys.min()<y-2 or xs.max()>x+bw+2 or ys.max()>y+bh+2:raise ValueError('mask_bbox_mismatch')
                    image_member=str(Path(member).parent.as_posix()+'/'+a['filename']);info=z.getinfo(image_member)
                    candidates.append(dict(frame_id=r['frame_id'],video_id=video,group_id=r['group_id'],frame_index=frame,
                        split='train',width=w,height=h,angle_axis_deg=angle,angle_bin=int(angle//15),eigenvalue_ratio=ratio,
                        axis_uncertain=ratio<3,centroid_xy=center,mask_area_native=int(mask.sum()),bbox_mask_native=bbox,
                        bbox_source_native=given,tip_source_native=json.loads(a['cystotome_tip']) if a.get('cystotome_tip') else None,
                        archive_member=image_member,image_bytes=info.file_size,archive_crc32=info.CRC))
                    ann_cache[r['frame_id']]=polygons
                except (ValueError,AssertionError,KeyError) as e:reject[str(e) or type(e).__name__]+=1
            reject['missing_csv_frame']+=len(frames)-len(seen)
        if len(source)%10==0:print('audit films',len(source),'candidates',len(candidates),flush=True)
    audit=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),source_manifest=str(POOL),
        source_manifest_sha256=sha(POOL.read_bytes()),train_rows=len(train),eligible=len(candidates),
        eligible_axis_certain=sum(not r['axis_uncertain'] for r in candidates),rejected=dict(reject),source_archives=source,
        angle_bins_available=dict(sorted(collections.Counter(r['angle_bin'] for r in candidates).items())),seed=SEED)
    js(HERE/'AUDYT_ZRODEL.json',audit)
    if len(candidates)<400:raise RuntimeError(f'Only {len(candidates)} eligible train')
    rng=np.random.default_rng(SEED)
    for r in candidates:r['tie']=float(rng.random())
    selected=[];ids=set();pixel_seen=set();byte_seen=set();filmcount=collections.Counter();bincount=collections.Counter();filmframes=collections.defaultdict(list)
    for gap in (50,20,5,0):
        while len(selected)<400:
            eligible=[r for r in candidates if r['frame_id'] not in ids and all(abs(r['frame_index']-f)>=gap for f in filmframes[r['video_id']])]
            if not eligible:break
            r=min(eligible,key=lambda a:(bincount[a['angle_bin']],a['axis_uncertain'],filmcount[a['video_id']],a['tie']))
            ids.add(r['frame_id']);video=r['video_id'];src=source[video]
            with zipfile.ZipFile(src['archive']) as z:imgbytes=z.read(r['archive_member'])
            im=cv2.imdecode(np.frombuffer(imgbytes,np.uint8),cv2.IMREAD_COLOR)
            if im is None or im.shape[:2]!=(r['height'],r['width']):reject['image_dimensions']+=1;continue
            bhash=sha(imgbytes);phash=sha(im.tobytes())
            if phash in pixel_seen or bhash in byte_seen:reject['duplicate_pixels_or_bytes']+=1;continue
            pixel_seen.add(phash);byte_seen.add(bhash);r.update(image_sha256=bhash,pixel_sha256=phash,selection_min_gap=gap)
            selected.append(r);filmcount[video]+=1;bincount[r['angle_bin']]+=1;filmframes[video].append(r['frame_index'])
        if len(selected)==400:break
    if len(selected)!=400:raise RuntimeError(f'Only {len(selected)} unique train')
    selected.sort(key=lambda a:(a['angle_bin'],a['video_id'],a['frame_index']))
    for p in ('obrazy_oryginalne','maski_natywne','maski_480x320','obrazy_bez_narzedzia'): (HERE/p).mkdir(exist_ok=True)
    coco=dict(info=dict(description='400 real train cystotome images; native unmodified JPEG'),images=[],annotations=[],categories=[dict(id=1,name='cystotome')])
    manifests=[];pairs=[];binary_cov=[];area_cov=[];background=[]
    for i,r in enumerate(selected):
        video=r['video_id'];polys=ann_cache[r['frame_id']];src=source[video]
        name=f'{video}_{r["frame_index"]:08d}';ip=HERE/'obrazy_oryginalne'/(name+'.jpg')
        if ip.exists():raise FileExistsError(ip)
        with zipfile.ZipFile(src['archive']) as z:imgbytes=z.read(r['archive_member'])
        assert sha(imgbytes)==r['image_sha256'];ip.write_bytes(imgbytes)
        mask=raster(polys,r['width'],r['height']);mp=HERE/'maski_natywne'/(name+'.png');assert cv2.imwrite(str(mp),mask*255)
        area=cv2.resize(mask.astype(np.float32),(480,320),interpolation=cv2.INTER_AREA);binary=area>0
        msmall=HERE/'maski_480x320'/(name+'.png');assert cv2.imwrite(str(msmall),binary.astype(np.uint8)*255)
        cov=coverage(binary);acov=coverage(area);dil=cv2.dilate(binary.astype(np.uint8),np.ones((17,17),np.uint8))
        binary_cov.append(cov);area_cov.append(acov);background.append(coverage(dil)==0)
        r.pop('tie');r.update(source_category='cystotome',source=src,source_polygons=polys,
            image_path=str(ip.relative_to(HERE)),mask_native_path=str(mp.relative_to(HERE)),mask_480_path=str(msmall.relative_to(HERE)),
            mask_native_sha256=sha(mp.read_bytes()),mask_480_sha256=sha(msmall.read_bytes()),tool_tokens=int((cov>0).sum()),
            tool_tokens_25=int((cov>=.25).sum()),background_tokens=int((coverage(dil)==0).sum()),token_row=i)
        manifests.append(r);pairs.append(dict(pair_id=name,frame_id=r['frame_id'],split='train',original=r['image_path'],
            synthetic=f'obrazy_bez_narzedzia/{name}.png',synthetic_status='pending_model_decision',
            comparison_token_row=i,comparison_mask=r['mask_480_path'],indices_rule='same original-mask tokens in both images'))
        coco['images'].append(dict(id=i+1,file_name=r['image_path'],width=r['width'],height=r['height'],source_frame_id=r['frame_id'],split='train'))
        coco['annotations'].append(dict(id=i+1,image_id=i+1,category_id=1,segmentation=polys,bbox=r['bbox_mask_native'],
            source_bbox=r['bbox_source_native'],area=r['mask_area_native'],iscrowd=0))
    js(HERE/'manifest.json',manifests);js(HERE/'PARY.json',pairs);js(HERE/'annotations.coco.json',coco)
    np.savez_compressed(HERE/'TOKEN_MASKI.npz',frame_ids=np.array([r['frame_id'] for r in manifests]),
        tool_coverage=np.stack(binary_cov),area_coverage=np.stack(area_cov),tool=np.stack(binary_cov)>0,
        tool25=np.stack(binary_cov)>=.25,background=np.stack(background),grid_hw=np.array([20,30]),patch_wh=np.array([16,16]))
    # Contact sheet is a visualization only; dataset JPEG remains byte-for-byte.
    picks=np.unique(np.rint(np.linspace(0,399,48)).astype(int));sheet=Image.new('RGB',(8*240,6*180),'white');draw=ImageDraw.Draw(sheet)
    for j,k in enumerate(picks):
        r=manifests[k];im=Image.open(HERE/r['image_path']).convert('RGB');im.thumbnail((236,150))
        x=(j%8)*240;y=(j//8)*180;sheet.paste(im,(x,y))
        draw.text((x+3,y+152),f'{r["angle_axis_deg"]:.1f} deg {r["video_id"]}',fill='black')
        draw.text((x+3,y+165),f'frame {r["frame_index"]} axis ratio {r["eigenvalue_ratio"]:.1f}',fill='black')
    sheet.save(HERE/'PLANSZA_48.jpg',quality=92)
    verified=verify()
    report=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),count_original=400,count_synthetic=0,pairs_pending=400,
        films=len(filmcount),angle_counts=dict(sorted(bincount.items())),film_counts=dict(sorted(filmcount.items())),
        selection_gap_counts=dict(collections.Counter(r['selection_min_gap'] for r in manifests)),
        axis_uncertain=sum(r['axis_uncertain'] for r in manifests),angle_range_deg=[min(r['angle_axis_deg'] for r in manifests),max(r['angle_axis_deg'] for r in manifests)],
        candidate_rejects=dict(reject),seed=SEED,seconds=time.time()-start,verification=verified,
        bytes_total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
    js(HERE/'WYNIKI.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

def verify():
    rows=json.loads((HERE/'manifest.json').read_text());pool={r['frame_id']:r for r in [json.loads(s) for s in POOL.read_text().splitlines()]}
    assert len(rows)==400 and len({r['frame_id'] for r in rows})==400
    assert len({r['image_sha256'] for r in rows})==400 and len({r['pixel_sha256'] for r in rows})==400
    for r in rows:
        assert pool[r['frame_id']]['split']=='train'
        im=cv2.imread(str(HERE/r['image_path']));mask=cv2.imread(str(HERE/r['mask_native_path']),0)
        assert im.shape[:2]==mask.shape==(r['height'],r['width']) and mask.any()
        assert sha((HERE/r['image_path']).read_bytes())==r['image_sha256']
        assert np.array_equal(mask>0,raster(r['source_polygons'],r['width'],r['height'])>0)
        assert sha((HERE/r['mask_native_path']).read_bytes())==r['mask_native_sha256']
    z=np.load(HERE/'TOKEN_MASKI.npz');assert z['tool_coverage'].shape==(400,600)
    assert np.isfinite(z['tool_coverage']).all() and ((z['tool_coverage']>=0)&(z['tool_coverage']<=1)).all()
    assert np.array_equal(z['frame_ids'],[r['frame_id'] for r in rows])
    assert ((z['tool']&z['background']).sum()==0)
    return dict(exactly400=True,unique_frame_ids=True,unique_image_bytes=True,unique_pixels=True,all_train=True,
        dimensions=True,native_polygon_rasterization=True,image_and_mask_checksums=True,token_grid_400x600=True,
        original_images_unmodified=True,synthetic_count=len(list((HERE/'obrazy_bez_narzedzia').glob('*.png'))))

if __name__=='__main__':main()
