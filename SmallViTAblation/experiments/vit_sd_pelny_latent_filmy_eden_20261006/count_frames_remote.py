"""Count CSV annotation rows in video TRAIN without opening JPEG members."""
import csv,io,json,zipfile
from pathlib import Path
root=Path('/home2/faculty/bpiotrowski')
splits={}
for line in (root/'smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wejscia/pool_manifest.jsonl').read_text().splitlines():
    r=json.loads(line);splits[r['video_id']]=r['split']
csv.field_size_limit(32*1024*1024)
stats=[]
for film in sorted(f for f,s in splits.items() if s=='train'):
    with zipfile.ZipFile(root/'smallvit_data/cataract_lmm/3_Object_Tracking/annotations'/f'{film}.zip') as z:
        member=next(x for x in z.namelist() if x.endswith('annotation.csv'))
        with z.open(member) as f,io.TextIOWrapper(f,encoding='utf-8-sig',newline='') as text:
            n=tool=0
            for r in csv.DictReader(text):
                n+=1; tool+=bool(r.get('cystotome_bbox','').strip() not in ('','[]'))
        stats.append(dict(film=film,frames=n,tool_boxes=tool))
print(json.dumps(dict(train_films=len(stats),all_train_frames=sum(x['frames'] for x in stats),train_frames_with_tool_box=sum(x['tool_boxes'] for x in stats),films=stats)))
