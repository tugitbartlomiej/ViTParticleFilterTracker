import json
from pathlib import Path
import dane
root=Path('/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy')
row=next(r for r in dane.load_manifest(root/'wejscia/FILMY.json') if r['video_id']=='TR_0007_S1_P03')
try:
    next(dane.iter_video(row))
except dane.FrameCountMismatch as e:
    print(json.dumps({'film':row['video_id'],'preflight_rejected':True,'yielded_samples':0,'reason':str(e)}),flush=True)
else:
    raise AssertionError('Expected preflight rejection before any sample')
import torch
c=torch.load(root/'wyniki/trening/checkpoint_00002500.pt',map_location='cpu',weights_only=True)
assert c['step']==2500 and c['optimizer'] and c['trainable']
print(json.dumps({'checkpoint_step':c['step'],'checkpoint_readable':True,'rank_states':len(c['ranks'])}),flush=True)
