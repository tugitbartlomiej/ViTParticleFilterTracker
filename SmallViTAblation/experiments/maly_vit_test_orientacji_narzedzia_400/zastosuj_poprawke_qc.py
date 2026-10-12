"""Zapis jednej jawnej poprawki po pilotowej ocenie widocznego śladu trzonu."""
import json,shutil
import cv2,numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from przygotuj import HERE,js,sha

fid='cataract_lmm/tracking/TR_0022_S1_P03/00003266'
records=json.loads((HERE/'INPAINT_QC.json').read_text())
rec=next(r for r in records if r['frame_id']==fid)
pairs=json.loads((HERE/'PARY.json').read_text())
pair=next(p for p in pairs if p['frame_id']==fid)
candidate=next(r for r in json.loads((HERE/'pilot_poprawka/QC.json').read_text()) if r['dilation_native_radius']==10)
archive=HERE/'pilot_poprawka/original_radius5';archive.mkdir(exist_ok=True)
for key in ('image_path','mask_edit_path'):
    dst=HERE/rec[key]
    archived=archive/(key+'_'+dst.name)
    if not archived.exists():
        if key=='image_path' and (archive/dst.name).exists():shutil.copy2(archive/dst.name,archived)
        elif key=='mask_edit_path':
            row=next(r for r in json.loads((HERE/'manifest.json').read_text()) if r['frame_id']==fid)
            native=(cv2.imread(str(HERE/row['mask_native_path']),0)>0).astype(np.uint8)
            cv2.imwrite(str(archived),cv2.dilate(native,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))*255)
        else:shutil.copy2(dst,archived)
    assert sha((HERE/rec[key]).read_bytes()) in (rec['synthetic_sha256'] if key=='image_path' else rec['mask_edit_sha256'],candidate['synthetic_sha256'] if key=='image_path' else candidate['mask_edit_sha256'])
    shutil.copy2(candidate[key],dst)
old_seconds=rec.get('initial_radius5_seconds',rec['seconds'])
for key in ('synthetic_sha256','mask_edit_sha256','dilation_native_radius','dilation_reference480_radius_xy','edit_pixels','raw_output_range'):
    rec[key]=candidate[key]
rec.update(initial_radius5_seconds=old_seconds,correction_seconds=candidate['seconds'],
    qc='geometry_pass; native visual QC corrected dark shaft-like line with radius10; radius15 not selected',
    correction_date=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),initial_output_archive='pilot_poprawka/original_radius5')
pair.update(synthetic_sha256=rec['synthetic_sha256'],mask_edit_sha256=rec['mask_edit_sha256'],
    visual_qc='native visual review: dark shaft-like line reduced after explicit radius10 correction',
    correction='QC radius10; initial radius5 archived; original source mask and token indices unchanged')
js(HERE/'INPAINT_QC.json',records);js(HERE/'PARY.json',pairs)
cfg=json.loads((HERE/'INPAINT_CONFIG.json').read_text());cfg['qc_mask_exceptions']=[dict(frame_id=fid,native_radius=10,reason='visible dark shaft-like line in radius5; minimum of tested10/15 selected')]
js(HERE/'INPAINT_CONFIG.json',cfg)
print('One canonical pair corrected; original radius5 and two candidates retained separately')
