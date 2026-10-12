"""Control whether LaMa edits alone provoke candidate concept responses.

Translate the actual edit mask into a nearby disjoint region, preserve its
native area and shape, and leave the annotated cystotome intact. This controls
the editing mechanism imperfectly: image content and spatial position differ.
No model or detector results are consulted when choosing masks or images.
"""
from pathlib import Path
import sys, json, hashlib, time, argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'maly_vit_test_orientacji_narzedzia_400'
sys.path.insert(0, str(DATA))
import lama_inpaint as lama

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def translated_mask(edit, tool):
    h, w = edit.shape
    forbidden = cv2.dilate(tool.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    # Coarse fixed grid, shortest displacement first; no output-based choices.
    offsets = [(dx, dy) for dx in range(-w+16, w, 24)
               for dy in range(-h+16, h, 24)
               if dx*dx+dy*dy >= 48**2]
    offsets.sort(key=lambda v: (v[0]**2+v[1]**2, v[1], v[0]))
    area = int(edit.sum())
    for dx, dy in offsets:
        shifted = cv2.warpAffine(edit.astype(np.uint8),
            np.array([[1, 0, dx], [0, 1, dy]], np.float32), (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        if int(shifted.sum()) == area and not np.any(shifted & forbidden):
            return shifted, [dx, dy]
    return None, None

def fill_image(session, image, edit):
    h, w = image.shape[:2]
    scale = min(512/w, 512/h)
    nw, nh = round(w*scale), round(h*scale)
    top, left = (512-nh)//2, (512-nw)//2
    bottom, right = 512-nh-top, 512-nw-left
    x = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(edit.astype(np.uint8), (nw,nh), interpolation=cv2.INTER_NEAREST)
    x = cv2.copyMakeBorder(x,top,bottom,left,right,cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    result = session.run(['output'], {'image':x.transpose(2,0,1)[None].astype(np.float32)/255,
                                     'mask':mask[None,None].astype(np.float32)})[0]
    fill = result[0].transpose(1,2,0)[top:top+nh,left:left+nw]
    fill = cv2.resize(fill,(w,h),interpolation=cv2.INTER_LINEAR)
    fill = np.clip(np.rint(fill),0,255).astype(np.uint8)
    out = image.copy()
    out[edit] = fill[edit]
    assert np.array_equal(out[~edit], image[~edit])
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    output_root = HERE/'heldout' if args.test_only else HERE
    output_root.mkdir(exist_ok=True)
    rows = json.loads((DATA/'manifest.json').read_text(encoding='utf-8-sig'))
    allowed = set(range(len(rows)))
    if args.test_only:
        split = json.loads((HERE.parent/'PODZIAL.json').read_text(encoding='utf-8-sig'))
        allowed = {int(r['source_index']) for r in split['records'] if r['split']=='test'}
    selected=[]
    for angle_bin in range(12):
        # Two deterministic examples of each source-native orientation bin.
        available = [i for i,r in enumerate(rows) if r['angle_bin']==angle_bin and i in allowed]
        assert len(available) >= 2, (angle_bin, len(available))
        selected.extend([available[len(available)//3],available[2*len(available)//3]])
    session, meta = lama.session()
    (output_root/'images').mkdir(exist_ok=True)
    (output_root/'masks').mkdir(exist_ok=True)
    records=[]
    start=time.time()
    for source_index in selected:
        row=rows[source_index]
        image=cv2.cvtColor(cv2.imread(str(DATA/row['image_path'])),cv2.COLOR_BGR2RGB)
        tool=cv2.imread(str(DATA/row['mask_native_path']),0)>0
        edit=cv2.imread(str(DATA/row['mask_edit_path']),0)>0
        shifted, offset=translated_mask(edit,tool)
        if shifted is None:
            records.append({'source_index':source_index,'frame_id':row['frame_id'],'status':'no_valid_translation'})
            continue
        output_path=output_root/'images'/f'sham_{source_index:03d}.png'
        mask_path=output_root/'masks'/f'sham_{source_index:03d}.png'
        if output_path.exists():
            out=cv2.cvtColor(cv2.imread(str(output_path)),cv2.COLOR_BGR2RGB)
        else:
            out=fill_image(session,image,shifted)
            assert cv2.imwrite(str(output_path),cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
            assert cv2.imwrite(str(mask_path),shifted.astype(np.uint8)*255)
        assert np.array_equal(out[~shifted],image[~shifted])
        assert np.array_equal(out[tool],image[tool])
        frac=cv2.resize(shifted.astype(np.float32),(480,320),interpolation=cv2.INTER_AREA)
        coverage=frac.reshape(20,16,30,16).mean(axis=(1,3)).reshape(600)
        records.append(dict(source_index=source_index,frame_id=row['frame_id'],video_id=row['video_id'],
            angle_bin=row['angle_bin'],status='generated',synthetic_path=str(output_path),mask_edit_path=str(mask_path),
            translated_pixels_xy=offset,edit_pixels=int(shifted.sum()),original_edit_pixels=int(edit.sum()),
            edit_mask_overlaps_tool=False,outside_mask_changed_pixels=0,tool_pixels_changed=0,
            image_sha256=sha(output_path),mask_sha256=sha(mask_path),coverage=coverage.tolist()))
        print('sham',len(records),'/',len(selected),flush=True)
    report=dict(status='complete',records=records,count=sum(r['status']=='generated' for r in records),
        source_dataset=str(DATA),model_revision=meta['revision'],seconds=time.time()-start,
        selection='two deterministic source indices per native angle bin, without model outputs',
        split_scope='test_only' if args.test_only else 'mixed',
        limitations=['different spatial position and content versus tool erasure',
                     'background means no annotated cystotome; another unannotated instrument is not excluded',
                     'not independent additional original images; paired controls of existing dataset'])
    (output_root/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    good=[r for r in records if r['status']=='generated'][:8]
    sheet=Image.new('RGB',(960,len(good)*224),'white')
    draw=ImageDraw.Draw(sheet)
    for k,r in enumerate(good):
        row=rows[r['source_index']]
        original=Image.open(DATA/row['image_path']).convert('RGB').resize((320,213))
        edited=Image.open(r['synthetic_path']).convert('RGB').resize((320,213))
        mask=Image.open(r['mask_edit_path']).convert('RGB').resize((320,213))
        sheet.paste(original,(0,k*224+11));sheet.paste(edited,(320,k*224+11));sheet.paste(mask,(640,k*224+11))
        draw.text((2,k*224),f"{r['source_index']}: original / sham LaMa / translated mask; tool kept",fill='black')
    sheet.save(output_root/'SHAM_PREVIEW.jpg',quality=90)
    print('complete',report['count'],'seconds',round(report['seconds'],1),flush=True)

if __name__=='__main__':main()
