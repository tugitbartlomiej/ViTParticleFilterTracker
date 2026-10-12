from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2];out=root/'outputs/vit_k1000_taniec_czasteczek_20261009'
proof=json.loads((out/'WERYFIKACJA_SHADERA.json').read_text(encoding='utf-8'))
proof['browser_validation']={'renderer':'ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, D3D11)','all_900000_vertices_drawn':True,'verified_steps':[100,500],'natural_positions_and_displacements':True,'webgl_errors':0,'console_errors':[],'observed_auto_rotation_fps':60,'fps_scope':'Observed requestAnimationFrame throughput during automatic rotation of 900000 vertices; not an isolated GPU time benchmark or a guarantee.','buffer_bytes_after_step500':43200000,'exact128_playback_visible':True}
proof['verified_utc']=datetime.now(timezone.utc).isoformat()
(out/'WERYFIKACJA_SHADERA.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
path=out/'WYNIKI.json';card=json.loads(path.read_text(encoding='utf-8-sig'))
card['gpu_renderer']={'all_particles':900000,'sampling':False,'binary_float32':True,'recorded_steps':[0,100,500],'proof':'WERYFIKACJA_SHADERA.json','observed_auto_rotation_fps':60,'vertex_buffers_bytes':43200000}
for a in card['artifacts']:
    p=out/a['filename']
    a.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
path.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
print('Zapisano weryfikację GPU i aktualne sumy plików.')
