"""Czy większa jawna maska ogranicza zaobserwowany cień/trzon LaMa?
Osobne kandydaty: bez nadpisywania 400 par lub parametrów kampanii.
"""
import json
from pathlib import Path
import cv2
import lama_inpaint as lama
from przygotuj import HERE, js

rows=json.loads((HERE/'manifest.json').read_text())
row=next(r for r in rows if 'TR_0022_S1_P03/00003266' in r['frame_id'])
session,meta=lama.session()
records=[]
for radius in (10,15):
    target=HERE/'pilot_poprawka'/f'radius_{radius}'
    (target/'obrazy_bez_narzedzia').mkdir(parents=True,exist_ok=True)
    (target/'maski_edycji').mkdir(exist_ok=True)
    local=dict(row,image_path=str(HERE/row['image_path']),mask_native_path=str(HERE/row['mask_native_path']))
    lama.HERE=target;lama.RADIUS=radius
    rec=lama.inpaint(session,local)
    rec['image_path']=str(target/rec['image_path']);rec['mask_edit_path']=str(target/rec['mask_edit_path'])
    records.append(rec)
js(HERE/'pilot_poprawka/QC.json',records)
print('Two separate diagnostic candidates saved; main 400 unchanged')
