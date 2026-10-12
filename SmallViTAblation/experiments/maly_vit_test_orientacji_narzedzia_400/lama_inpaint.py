"""Czy lokalna LaMa usuwa maskowany cystotome, zachowując obraz poza maską?
Zmiana zewnętrznych pikseli lub niezgodne IO obala poprawność pary.
Użytkownik jawnie autoryzował programową lokalną inferencję400par LaMa.
"""
import os
os.environ['HF_HOME']='F:/HuggingFace'
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2,numpy as np,onnxruntime as ort
from PIL import Image,ImageDraw
from przygotuj import HERE,js,sha

PILOT=[0,65,130,200,265,335]
RADIUS=5

def session():
    meta=json.loads((HERE/'model_lama/MODEL.json').read_text())
    path=meta['files']['lama_fp32.onnx']['cache_path']
    assert sha(Path(path).read_bytes())==meta['files']['lama_fp32.onnx']['sha256']
    opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
    s=ort.InferenceSession(path,sess_options=opt,providers=['CPUExecutionProvider'])
    assert [x.name for x in s.get_inputs()]==['image','mask']
    assert all(x.shape[-2:]==[512,512] for x in s.get_inputs())
    return s,meta

def inpaint(s,row):
    image=cv2.cvtColor(cv2.imread(str(HERE/row['image_path'])),cv2.COLOR_BGR2RGB)
    original=cv2.imread(str(HERE/row['mask_native_path']),0)>0
    edit=cv2.dilate(original.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*RADIUS+1,2*RADIUS+1)))>0
    h,w=image.shape[:2];scale=min(512/w,512/h);nw,nh=round(w*scale),round(h*scale)
    top=(512-nh)//2;bottom=512-nh-top;left=(512-nw)//2;right=512-nw-left
    im=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    mk=cv2.resize(edit.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)
    im=cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_REFLECT_101)
    mk=cv2.copyMakeBorder(mk,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    X=im.astype(np.float32).transpose(2,0,1)[None]/255;M=mk.astype(np.float32)[None,None]
    tic=time.time();Y=s.run(['output'],dict(image=X,mask=M))[0];seconds=time.time()-tic
    assert Y.shape==(1,3,512,512) and np.isfinite(Y).all()
    fill=Y[0].transpose(1,2,0);bounds=[float(fill.min()),float(fill.max())]
    assert bounds[1]>1.5 and bounds[0]>-20 and bounds[1]<300,'unexpected output scale'
    fill=fill[top:top+nh,left:left+nw]
    fill=cv2.resize(fill,(w,h),interpolation=cv2.INTER_LINEAR)
    fill=np.clip(np.rint(fill),0,255).astype(np.uint8)
    out=image.copy();out[edit]=fill[edit]
    assert np.array_equal(out[~edit],image[~edit])
    name=Path(row['image_path']).stem
    ip=HERE/'obrazy_bez_narzedzia'/(name+'.png');mp=HERE/'maski_edycji'/(name+'.png')
    assert not ip.exists() and not mp.exists()
    assert cv2.imwrite(str(ip),cv2.cvtColor(out,cv2.COLOR_RGB2BGR)) and cv2.imwrite(str(mp),edit.astype(np.uint8)*255)
    return dict(frame_id=row['frame_id'],image_path=str(ip.relative_to(HERE)),mask_edit_path=str(mp.relative_to(HERE)),
        synthetic_sha256=sha(ip.read_bytes()),mask_edit_sha256=sha(mp.read_bytes()),
        original_sha256=row['image_sha256'],seconds=seconds,raw_output_range=bounds,
        shape_native_wh=[w,h],resized_wh=[nw,nh],pad_tblr=[top,bottom,left,right],
        dilation_native_radius=RADIUS,dilation_reference480_radius_xy=[RADIUS*480/w,RADIUS*320/h],
        edit_pixels=int(edit.sum()),original_mask_pixels=int(original.sum()),outside_edit_changed_pixels=0,
        original_mask_covered=True,synthetic=True,qc='automatic_geometry_pass; visual audit separate')

def sheet(rows,records,filename):
    byid={r['frame_id']:r for r in records};chosen=[r for r in rows if r['frame_id'] in byid]
    grid=Image.new('RGB',(1440,len(chosen)*344),'white');draw=ImageDraw.Draw(grid)
    for j,r in enumerate(chosen):
        rec=byid[r['frame_id']]
        for col,path in enumerate([r['image_path'],rec['image_path'],rec['mask_edit_path']]):
            im=Image.open(HERE/path).convert('RGB');im=im.resize((480,320));grid.paste(im,(col*480,j*344+24))
        draw.text((4,j*344+3),f'{r["video_id"]}/{r["frame_index"]} axis {r["angle_axis_deg"]:.1f} | original / LaMa synthetic / edit mask',fill='black')
    grid.save(HERE/filename,quality=93)

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');args=p.parse_args()
    rows=json.loads((HERE/'manifest.json').read_text());indices=list(range(400)) if args.all else PILOT
    (HERE/'maski_edycji').mkdir(exist_ok=True);s,meta=session();start=time.time()
    cfg=dict(repo=meta['repo'],revision=meta['revision'],model_sha256=meta['files']['lama_fp32.onnx']['sha256'],
        model_cache_path=meta['files']['lama_fp32.onnx']['cache_path'],runtime=ort.__version__,providers=s.get_providers(),
        threads=4,batch=1,image_values='RGB CHW float32 0..1',mask_values='1 erase 0 keep',output_values='raw 0..255',
        preprocessing='aspect-preserving long side512; round height; REFLECT_101 symmetric image padding; constant0 mask padding; nearest mask',
        postprocessing='remove padding; linear fill resize to each native source dimensions; round/clipuint8; composite edit only; losslessPNG',
        dilation='native elliptical radius5 pixels; reference radius_xy depends on original size, saved per image',seed=None,
        deterministic_inference=True,reference_token_masks='unchanged original',
        license='Apache2.0; README/license saved; third-partyONNXmirror of Carve',
        hardware_audit='RTX3070 8192MiB1531MiBused,0%util; CPUruntimeexisting, noGPUinference claim')
    js(HERE/'INPAINT_CONFIG.json',cfg)
    path=HERE/'INPAINT_QC.json';records=json.loads(path.read_text()) if path.exists() else []
    known={r['frame_id']:r for r in records};pairs=json.loads((HERE/'PARY.json').read_text())
    for k in indices:
        row=rows[k]
        if row['frame_id'] in known:
            rec=known[row['frame_id']];assert sha((HERE/rec['image_path']).read_bytes())==rec['synthetic_sha256'];continue
        rec=inpaint(s,row);rec.update(model_revision=meta['revision'],model_sha256=cfg['model_sha256'])
        records.append(rec);known[row['frame_id']]=rec;js(path,records)
        pair=pairs[k];assert pair['frame_id']==row['frame_id']
        pair.update(synthetic_status='generated_local_LaMa',synthetic_sha256=rec['synthetic_sha256'],
            mask_edit=rec['mask_edit_path'],mask_edit_sha256=rec['mask_edit_sha256'],qc=rec['qc'],model_revision=meta['revision'])
        js(HERE/'PARY.json',pairs)
        if len(records)%10==0 or not args.all:print('generated',len(records),'/400','last seconds',round(rec['seconds'],3),flush=True)
    if not args.all:sheet([rows[k] for k in PILOT],records,'PILOT_LAMA_6.jpg')
    else:sheet([rows[k] for k in np.rint(np.linspace(0,399,12)).astype(int)],records,'PARY_LAMA_12.jpg')
    js(HERE/'INPAINT_STATUS.json',dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),count_generated=len(records),
        count_original=400,count_total_images=400+len(records),pending=400-len(records),elapsed_current_seconds=time.time()-start,
        all_images_outside_edit_unchanged=all(r['outside_edit_changed_pixels']==0 for r in records),
        no_ViT_or_detector_forward=True,synthetic_background_not_ground_truth=True))
    print('completed current phase',len(records),flush=True)

if __name__=='__main__':main()
