"""One-time bounded upgrade of the existing viewer; data remain unchanged."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
path=root/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'
s=path.read_text(encoding='utf-8')
def replace(old,new):
    global s
    assert old in s,old[:100]
    s=s.replace(old,new,1)
replace('<script src="/plotly.min.js"></script>','<script src="/plotly.min.js"></script>\n<script src="/cloud-webgl.js"></script>')
replace('<option value="global">','<option value="global" selected>')
replace('<option value="exact" selected>','<option value="exact">')
replace('max="500" step="1" value="100"','max="2" step="1" value="1"')
replace('<div id="point-detail">','<div id="gpu-cloud" class="chart hidden" style="position:relative" role="img" aria-label="Cała chmura 900 tysięcy cząsteczek"></div><div id="gpu-controls" class="hidden"><label><input id="spin" type="checkbox"> Obracaj automatycznie</label> <label>Widoczność punktów <input id="opacity" type="range" min="0.01" max="0.3" step="0.005" value="0.065" aria-label="Widoczność punktów"></label><div id="gpu-status" class="badge" role="status"></div></div><div id="point-detail">')
replace('Podgląd globalny pokazuje stałą próbkę 8000;','Podgląd globalny rysuje wszystkie 900 000 punktów przez shader GPU;')
replace('const actualStep=','let gpuCloud=null;\nconst actualStep=')
replace('trails:o.trails?1:0})','trails:o.trails?1:0,geometry:o.mode===\'global\'?0:1})')
replace("const traces=[];if(o.mode==='exact'){", "el('cloud').classList.toggle('hidden',o.mode==='global');el('gpu-controls').classList.toggle('hidden',o.mode!=='global');\nif(o.mode==='global'){if(!gpuCloud)gpuCloud=new ParticleCloudGPU(el('gpu-cloud'),el('gpu-status'),el('point-detail'));await gpuCloud.show(d,o.movement);el('point-detail').textContent='Kliknij cząsteczkę, aby odczytać jej obraz i numer patcha. Obrót: przeciągnij. Przybliżenie: kółko myszy.';}else{if(gpuCloud)gpuCloud.hide();\nconst traces=[];if(o.mode==='exact'){")
replace("await Plotly.react('cloud',traces,layout,CONFIG);", "await Plotly.react('cloud',traces,layout,CONFIG);\n}")
replace('Stała, deterministyczna próbka 8000 z 900 000 cząsteczek.','Wszystkie 900 000 cząsteczek, bez próbkowania. Shader obraca i koloruje rzeczywiste pozycje; nie zmienia latentu.')
replace("async function init(){", "el('spin').onchange=()=>{if(gpuCloud){gpuCloud.spinning=el('spin').checked;gpuCloud.frames=[];gpuCloud.requestDraw();}};el('opacity').oninput=()=>{if(gpuCloud){gpuCloud.opacity=Number(el('opacity').value);gpuCloud.requestDraw();}};\nasync function init(){")
# Plotly events become available after the first non-global draw.
replace("el('cloud').on('plotly_relayout'", "if(el('cloud').on)el('cloud').on('plotly_relayout'")
replace("el('cloud').on('plotly_hover'", "if(el('cloud').on)el('cloud').on('plotly_hover'")
path.write_text(s,encoding='utf-8')
print('Zapisano widok GPU wszystkich 900000 cząsteczek.')
