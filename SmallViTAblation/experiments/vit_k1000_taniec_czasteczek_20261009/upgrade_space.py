from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'
s=p.read_text(encoding='utf-8')
if 'id="space-view"' in s:raise RuntimeError('Space controls already installed')
s=s.replace('<div class="inspect-controls" aria-label="Kolorowanie', '''<div class="inspect-controls"><label for="space-view">Widok przestrzeni</label><select id="space-view"><option value="pca">PCA · przestrzeń 3D</option><option value="pc12">Płaszczyzna PC1–PC2</option><option value="pc13">Płaszczyzna PC1–PC3</option><option value="pc23">Płaszczyzna PC2–PC3</option><option value="grid">Siatka patchy + ruch 768D · bez PCA</option><option value="raw">Oryginalne kanały latentu · bez PCA</option></select><label><input type="checkbox" id="show-axes" checked> Osie i podziałka</label><div id="raw-axes" class="inspect-controls hidden"><label>X <input id="axis-x" type="number" min="0" max="767" value="0" style="width:75px"></label><label>Y <input id="axis-y" type="number" min="0" max="767" value="1" style="width:75px"></label><label>Z <input id="axis-z" type="number" min="0" max="767" value="2" style="width:75px"></label><button id="apply-axes">Zastosuj kanały</button></div></div><div id="space-note" class="small">Stałe wspólne osie PCA. Wybierz inny widok, żeby zobaczyć siatkę ruchu lub surowe kanały.</div>
<div class="inspect-controls" aria-label="Kolorowanie''')
s=s.replace('<div id="selected-metrics"', '<div id="profile-panel" class="hidden"><h2 style="margin-top:20px">Profil 768 wymiarów wybranej cząsteczki</h2><div id="latent-profile" class="chart-small"></div><p id="raw-provenance" class="note"></p></div><div id="selected-metrics"')
s=s.replace("const availableSteps=()=>", "const availableSteps=()=>spaceView()==='raw'?[0,100,500]:")
s=s.replace("dense=el('mode').value!=='exact'&&META?.dense_ready", "dense=spaceView()!=='raw'&&el('mode').value!=='exact'&&META?.dense_ready")
s=s.replace('async function loadFrame(o){','async function loadFrame(o){if(spaceView()===\'raw\')return rawFrame(o);')
s=s.replace('const d=provided||await loadFrame(o);','let d=provided||await loadFrame(o);d=transformSpace(d,o);')
s=s.replace("gpuCloud.colorMode=colorMode;", "gpuCloud.colorMode=colorMode;gpuCloud.showAxes=el('show-axes').checked;if(gpuCloud.spaceView!==spaceCode())gpuCloud.setSpace(spaceCode());")
s=s.replace("title:`${o.movement?'Δ ':''}PC${i+1}`", "title:spaceNames()[i],visible:el('show-axes').checked&&(spaceCode()===0||spaceCode()===4||i<2||spaceView()==='raw')")
s=s.replace("camera:lastCamera||{eye:{x:1.6,y:1.4,z:1.1}}", "camera:lastCamera||{eye:spaceCode()>=1&&spaceCode()<=3?{x:0,y:0,z:2.5}:{x:1.6,y:1.4,z:1.1}}")
s=s.replace('await updateInspection(d.step);', 'await updateInspection(d.step,d);')
s=s.replace("async function updateInspection(requestedStep){", "async function updateInspection(requestedStep,frame=null){if(spaceView()==='raw'){await updateRawInspection(frame||await rawFrame(opts()));return;}el('profile-panel').classList.add('hidden');")
s=s.replace('// Stable identities:',Path(__file__).with_name('space-inline.js').read_text(encoding='utf-8')+'\n// Stable identities:')
s=s.replace("el('mode').onchange=async()=>{stop();", "el('mode').onchange=async()=>{stop();if(spaceView()==='raw'&&el('mode').value!=='image'){el('space-view').value='pca';el('raw-axes').classList.add('hidden');el('profile-panel').classList.add('hidden');el('space-note').textContent='PCA wspólna dla całej chmury.';}")
p.write_text(s,encoding='utf-8')
Path(__file__).with_name('viewer_inline_check.js').write_text(s.split('<script>')[-1].split('</script>')[0],encoding='utf-8')
