"""Read-only inventory of compressed video and training runtime."""
import json, zipfile, importlib.util
from pathlib import Path
from collections import Counter
b=Path('/home2/faculty/bpiotrowski')
data=b/'smallvit_data/cataract_lmm/3_Object_Tracking'
old=b/'smallvit_runs/2026-09-30_dyfuzja_sd_filmy'
splits={}
for line in (old/'wejscia/pool_manifest.jsonl').read_text().splitlines():
    r=json.loads(line); splits[r['video_id']]=r['split']
videos=[]
for zp in sorted((data/'videos').glob('*.zip')):
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            if m.filename.lower().endswith('.mp4'):
                film=Path(m.filename).stem
                videos.append(dict(film=film,zip_path=str(zp),member=m.filename,split=splits.get(film,'unknown'),bytes=m.file_size,annotation_exists=(data/'annotations'/f'{film}.zip').exists()))
import av,io,csv
first=videos[0]
with zipfile.ZipFile(first['zip_path']) as z:
    blob=z.read(first['member'])
with av.open(io.BytesIO(blob)) as c:
    s=c.streams.video[0]; frame=next(c.decode(video=0))
    info=dict(stream_frames=s.frames,width=frame.width,height=frame.height,fps=str(s.average_rate),first_pts=frame.pts)
with zipfile.ZipFile(data/'annotations'/f"{first['film']}.zip") as z:
    csv_name=next(x for x in z.namelist() if x.endswith('annotation.csv'))
    rows=list(csv.DictReader(io.StringIO(z.read(csv_name).decode('utf-8-sig'))))
info.update(annotation_rows=len(rows),annotation_first=rows[0],annotation_last=rows[-1])
print(json.dumps(dict(videos=len(videos),splits=dict(Counter(x['split'] for x in videos)),examples=videos[:3],alignment_example=info,dependencies={x:importlib.util.find_spec(x) is not None for x in ['torch','av','cv2','timm','diffusers']},train_missing_annotations=[x['film'] for x in videos if x['split']=='train' and not x['annotation_exists']]),indent=2))
