"""Exact tracking-frame polygons from original ZIPs, no extraction or bbox fallback."""
import functools, json, re, zipfile
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
ANNOTATIONS=Path(__file__).resolve().parents[3]/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'

@functools.lru_cache(maxsize=2)
def archive(video):
    with zipfile.ZipFile(ANNOTATIONS/f'{video}.zip') as z:
        return json.loads(z.read(f'{video}/annotation_full.json'))

def patch_membership(mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(320,480):raise ValueError('Maska musi być w układzie 480×320')
    padded=np.pad(mask,1)
    interior=mask.copy()
    for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
        interior &= padded[dy:dy+320,dx:dx+480]
    edge=mask & ~interior
    def tiles(a):return a.reshape(20,16,30,16).transpose(0,2,1,3).reshape(600,256)
    return {'tool_patches':np.flatnonzero(tiles(mask).any(axis=1)).tolist(),
            'contour_patches':np.flatnonzero(tiles(edge).any(axis=1)).tolist(),
            'coverage':tiles(mask).mean(axis=1).tolist()}

def payload(row):
    match=re.fullmatch(r'(TR_\d{4}_S\d_P\d{2})_(\d+)\.jpg',row['image_name'])
    if not match:raise ValueError('Brak zgodnej tożsamości klatki tracking')
    video,number=match.groups();frame_number=int(number);d=archive(video)
    matches=[(v,k) for v in d['videos'] for k,n in enumerate(v['file_names'])
             if re.fullmatch(re.escape(video)+r'_0*'+str(frame_number)+r'\.jpg',n)]
    if len(matches)!=1:raise ValueError('Brak jednoznacznej klatki w adnotacjach')
    v,k=matches[0];polygons=[];instances=[]
    names={c['id']:c['name'] for c in d['categories']}
    for a in d['annotations']:
        if a['video_id']!=v['id'] or names[a['category_id']].lower() in ('cornea','pupil'):continue
        seg=a['segmentations'][k]
        if not seg:continue
        if not isinstance(seg,list):raise ValueError('Ten format maski nie jest obsługiwany; nie zastąpiono go ramką')
        parts=[]
        for p in seg:
            if len(p)<6 or len(p)%2 or not np.isfinite(p).all():raise ValueError('Niepoprawny wielokąt maski')
            parts.append([[float(p[j])*480/v['width'],float(p[j+1])*320/v['height']] for j in range(0,len(p),2)])
        polygons.extend(parts);instances.append({'id':a['id'],'category':names[a['category_id']]})
    mask=Image.new('1',(480,320));draw=ImageDraw.Draw(mask)
    for p in polygons:draw.polygon([tuple(x) for x in p],fill=1)
    return {'image':row['index'],'frame_id':row['frame_id'],'source':str(ANNOTATIONS/f'{video}.zip'),
            'annotation_frame_index':k,'polygons':polygons,'instances':instances,
            'mask_available':bool(polygons),**patch_membership(mask)}
