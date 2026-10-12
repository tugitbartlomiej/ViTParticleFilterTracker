"""Czy400 par LaMa zachowuje source i wszystkie piksele poza masceedycji?
Niekompletne800obrazy, zmiana poza maską lub zmienione indeksy obalą zbiór.
"""
import json,time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2,numpy as np
from przygotuj import HERE,verify,sha,js,coverage

def main():
    base=verify();rows=json.loads((HERE/'manifest.json').read_text());pairs=json.loads((HERE/'PARY.json').read_text())
    qc=json.loads((HERE/'INPAINT_QC.json').read_text());byid={r['frame_id']:r for r in qc}
    assert len(qc)==len(byid)==len(pairs)==400
    assert len({r['synthetic_sha256'] for r in pairs})==400
    token=np.load(HERE/'TOKEN_MASKI.npz');edit_cov=[];edit_area=[];outside=[];changes=[]
    for i,(r,pair) in enumerate(zip(rows,pairs)):
        assert r['frame_id']==pair['frame_id']==token['frame_ids'][i] and pair['comparison_token_row']==i
        a=cv2.imread(str(HERE/r['image_path']));b=cv2.imread(str(HERE/pair['synthetic']))
        assert a.shape==b.shape==(r['height'],r['width'],3)
        original=cv2.imread(str(HERE/r['mask_native_path']),0)>0
        edit=cv2.imread(str(HERE/pair['mask_edit']),0)>0
        radius=byid[r['frame_id']]['dilation_native_radius']
        assert radius in (5,10)
        expected=cv2.dilate(original.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
        assert np.array_equal(edit,expected) and np.all(edit[original])
        assert np.array_equal(a[~edit],b[~edit])
        assert sha((HERE/pair['synthetic']).read_bytes())==pair['synthetic_sha256']
        assert sha((HERE/pair['mask_edit']).read_bytes())==pair['mask_edit_sha256']
        assert sha((HERE/r['image_path']).read_bytes())==r['image_sha256']
        binary=cv2.imread(str(HERE/r['mask_480_path']),0)>0
        from przygotuj import orientation
        angle,ratio,center=orientation(binary.astype(np.uint8))
        r.update(bbox_source_480=[r['bbox_source_native'][j]*(480/r['width'] if j%2==0 else 320/r['height']) for j in range(4)],
            tip_source_480=[r['tip_source_native'][0]*480/r['width'],r['tip_source_native'][1]*320/r['height']] if r['tip_source_native'] else None,
            angle_axis_480_deg=angle,eigenvalue_ratio_480=ratio,
            source_to_reference_scale_xy=[480/r['width'],320/r['height']])
        byid[r['frame_id']]['dilation_reference480_radius_xy']=[radius*480/r['width'],radius*320/r['height']]
        byid[r['frame_id']].pop('dilation_reference480_radius',None)
        assert np.array_equal(coverage(binary),token['tool_coverage'][i])
        area=cv2.resize(edit.astype(np.float32),(480,320),interpolation=cv2.INTER_AREA)
        edit_cov.append(coverage(area>0));edit_area.append(coverage(area))
        outside.append(int(np.any(a!=b,axis=2)[~edit].sum()));changes.append(int(np.any(a!=b,axis=2)[edit].sum()))
        pair.setdefault('visual_qc','pilot6_and_contact_sample; not exhaustive')
        pair.update(geometry_qc='passed_exact_outside_mask_and_source',
            comparison_token_indices_unchanged=True,synthetic_status='generated_local_LaMa_verified_geometry')
        r.update(synthetic_path=pair['synthetic'],synthetic_sha256=pair['synthetic_sha256'],
            mask_edit_path=pair['mask_edit'],mask_edit_sha256=pair['mask_edit_sha256'],paired_synthetic=True)
    assert min(changes)>0
    js(HERE/'PARY.json',pairs);js(HERE/'manifest.json',rows)
    js(HERE/'INPAINT_QC.json',qc)
    np.savez_compressed(HERE/'TOKEN_MASKI_EDYCJI.npz',frame_ids=token['frame_ids'],
        edit_coverage=np.stack(edit_cov),edit_area_coverage=np.stack(edit_area),grid_hw=np.array([20,30]))
    record=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),original_count=400,synthetic_count=400,total_images=800,
        pairs=400,source_verification=base,outside_edit_changed_pixels_total=sum(outside),
        all_edit_masks_are_declared_dilation=True,dilation_radius_counts={str(radius):sum(q['dilation_native_radius']==radius for q in qc) for radius in (5,10)},all_edit_masks_cover_original=True,
        all_source_hashes_preserved=True,all_synthetic_and_edit_mask_hashes_checked=True,
        unique_synthetic_file_hashes=True,
        all_reference_token_indices_unchanged=True,all_pairs_changed_inside_edit=True,
        visual_review_scope='17 unique pairs in pilot6 and supplementary contact12; one native radius10 correction; not all400 at native resolution',
        limitations=['synthetic hidden background not true background','other instruments outside cystotome mask remain',
                    'natural angles not controlled rotations','train probe set notheldout detector evaluation'],
        token_file_sha256=sha((HERE/'TOKEN_MASKI.npz').read_bytes()),
        inference_seconds_sum=sum(r['seconds'] for r in qc),additional_correction_inference_seconds=sum(r.get('correction_seconds',0) for r in qc),runtime='onnxruntime1.19.2CPU4threadsbatch1')
    js(HERE/'WERYFIKACJA_PAR.json',record)
    old=json.loads((HERE/'WYNIKI.json').read_text());old.update(count_synthetic=400,count_original=400,pairs_pending=0,
        total_images=800,pairs_verified=400,inpaint_verification=record,
        bytes_total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
    js(HERE/'WYNIKI.json',old)
    cfg=json.loads((HERE/'CONFIG.json').read_text());cfg.update(synthetic_count=400,synthetic_status='all400_local_LaMa_complete',
        native_size_wh=None,native_size_policy='preserve each source size; see manifest',
        synthetic_output='PNG at same native size as source; exact pixels outside edit',
        prospective_model=None,model_metadata='model_lama/MODEL.json',inpaint_protocol='INPAINT_CONFIG.json')
    js(HERE/'CONFIG.json',cfg)
    print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
