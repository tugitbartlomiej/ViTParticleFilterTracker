"""Rozstrzyga zgodność 400 oryginałów i ramek z CSV źródłowym. Każda różnica przerywa audyt."""
import csv,io,json,zipfile,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs'/HERE.name
start=time.time();rows=json.loads((ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400/manifest.json').read_text(encoding='utf-8-sig'));groups={}
for row in rows:groups.setdefault(row['video_id'],[]).append(row)
checked=[]
for film,rr in groups.items():
    src=rr[0]['source']
    with zipfile.ZipFile(src['archive']) as z:
        raw=z.read(src['annotation_member']);assert hashlib.sha256(raw).hexdigest()==src['annotation_csv_sha256']
        annotations={int(a['frame_number']):a for a in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}
        for r in rr:
            a=annotations[r['frame_index']]
            assert json.loads(a['cystotome_bbox'])==r['bbox_source_native'],r['frame_id']
            assert json.loads(a['cystotome_seg'])==r['source_polygons'],r['frame_id']
            assert Path(r['archive_member']).name==a['filename']
            assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['image_sha256']
        checked.append(dict(film=film,images=len(rr),annotation_sha256=src['annotation_csv_sha256']))
result=dict(pass_all=True,images=len(rows),films=len(groups),seconds=time.time()-start,checked=checked,checks=['CSV SHA256','frame_number','bbox cystotome','source polygons','filename','original archive JPEG SHA256'])
(OUT/'SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checked'}))
