"""Integrate image/patch inspection into the existing, measured trajectory viewer."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HTML=ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'

def main():
    s=HTML.read_text(encoding='utf-8')
    if 'id="color-image"' in s:
        raise RuntimeError('Selection UI already installed')
    s=s.replace('</style>', '''
.inspect-controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}.inspect-controls input{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:7px}.inspect-controls button[aria-pressed="true"]{background:#23757c;border-color:var(--cyan)}
#source-preview{width:100%;height:auto;display:block;cursor:crosshair;border-radius:8px;border:1px solid var(--line)}#source-name{overflow-wrap:anywhere}#selected-quality{height:220px}#image-search{width:210px}#patch-select{width:90px}
</style>''')
    s=s.replace('<div id="frame-kind"', '''<div class="inspect-controls" aria-label="Kolorowanie cząsteczek"><span>Kolor:</span><button id="color-motion" aria-pressed="true">Długość ruchu</button><button id="color-image" aria-pressed="false">Według zdjęcia</button><button id="color-patch" aria-pressed="false">Według patcha</button><label><input type="checkbox" id="focus-image" checked> Wyróżnij 600 cząsteczek wybranego zdjęcia</label></div>
<div class="inspect-controls"><label for="image-search">Szukaj zdjęcia</label><input id="image-search" type="search" placeholder="Numer, film lub nazwa"><button id="prev-image" aria-label="Poprzednie zdjęcie">Poprzednie</button><button id="next-image" aria-label="Następne zdjęcie">Następne</button><button id="only-image">Pokaż tylko to zdjęcie</button></div>
<div id="color-note" class="small"></div>
<div id="frame-kind"''')
    s=s.replace('<section class="panel"><h2 id="metric-title">', '''<section class="panel"><h2>Wybrane zdjęcie i jego 600 patchy</h2><div id="source-name" class="small"></div><div class="inspect-controls"><label for="patch-select">Patch 0–599</label><input id="patch-select" type="number" min="0" max="599" value="0"><label><input id="show-grid" type="checkbox" checked> Siatka 20 × 30</label></div><canvas id="source-preview" width="480" height="320" role="img" aria-label="Zdjęcie źródłowe i wybór patcha w siatce 20 na 30"></canvas><div id="selection-detail" class="note" role="status"></div><p class="note">Kliknij patch na zdjęciu albo cząsteczkę w chmurze. Biała ramka i duży biały punkt oznaczają tę samą parę: zdjęcie + patch. Zielona ramka na zdjęciu to etykieta narzędzia, nie nowa predykcja.</p><h2 style="margin-top:20px">Detekcja wybranego zdjęcia</h2><div id="selected-quality" role="img" aria-label="Pewność i IoU wybranego zdjęcia w kolejnych krokach"></div><div id="selected-metrics" class="small"></div><h2 style="margin-top:24px" id="metric-title">''')
    s=s.replace("el('mode').value==='global'?(META?.global_steps", "el('mode').value!=='exact'?(META?.global_steps")
    s=s.replace("dense=el('mode').value==='global'&&META?.dense_ready", "dense=el('mode').value!=='exact'&&META?.dense_ready")
    s=s.replace("el('image-wrap').classList.toggle('hidden',mode==='global');", "el('image-wrap').classList.remove('hidden');")
    s=s.replace("ids.includes(old)&&old!==0?old:ids.find(i=>i!==0)||ids[0]", "ids.includes(old)?old:ids[0]")
    s=s.replace("if(!gpuCloud)gpuCloud=new ParticleCloudGPU(el('gpu-cloud'),el('gpu-status'),el('point-detail'));", "if(!gpuCloud){gpuCloud=new ParticleCloudGPU(el('gpu-cloud'),el('gpu-status'),el('point-detail'));gpuCloud.onSelect=chooseParticle;}gpuCloud.colorMode=colorMode;gpuCloud.setSelection(selectedImage(),selectedPatch,el('focus-image').checked);")
    s=s.replace("color:d.norm,cmin:", "color:colorMode?d.tokens.map((t,j)=>identityColor(colorMode===1?d.images[j]:t)):d.norm,cmin:")
    s=s.replace("const axis=(i)=>", "const selectedIndex=d.tokens.findIndex((t,j)=>t===selectedPatch&&d.images[j]===selectedImage());if(selectedIndex>=0)traces.push({x:[d.xyz[selectedIndex][0]],y:[d.xyz[selectedIndex][1]],z:[d.xyz[selectedIndex][2]],type:'scatter3d',mode:'markers',marker:{size:9,color:'#ffffff',line:{color:'#142131',width:2}},showlegend:false,customdata:[[selectedImage(),selectedPatch]],hovertemplate:'Wybrany obraz %{customdata[0]} · patch %{customdata[1]}<extra></extra>'});\nconst axis=(i)=>")
    s=s.replace("el('cloud').dataset.eventsAttached='true';", "el('cloud').on('plotly_click',event=>{const v=event.points[0].customdata;if(v)chooseParticle(v[0],v[1]);});el('cloud').dataset.eventsAttached='true';")
    s=s.replace("await Plotly.relayout('quality',", "await updateInspection(d.step);await Plotly.relayout('quality',")
    s=s.replace("if(performance.now()-lastPanelTime>250", "if(performance.now()-lastPanelTime>250")
    s=s.replace("el('image').onchange=async()=>{stop();lastCamera=null;await drawCurves();await draw()}", "el('image').onchange=async()=>{stop();lastCamera=null;await drawSelectedCurve();await drawCurves();await draw()}")
    s=s.replace("await drawCurves();await draw();await waitForView();el('play').disabled=false;", "await drawSelectedCurve();await drawCurves();await draw();await waitForView();el('play').disabled=false;")
    extension=Path(__file__).with_name('selection-inline.js').read_text(encoding='utf-8')
    s=s.replace("async function init(){", extension+"\nasync function init(){")
    HTML.write_text(s,encoding='utf-8')

if __name__=='__main__':main()
