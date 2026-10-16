"""Samodzielny widok Plotly offline; wszystkie pokazywane punkty policzono w 192D."""
from pathlib import Path
import base64
import io
import json

import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/dolina_odpowiedzi_t226_20261015'
d=np.load(OUT/'geometria.npz')
buf=io.BytesIO();Image.fromarray(d['rgb']).save(buf,format='PNG')
keys=['sheet_xyz','sheet_o','sheet_p','sheet_iou','path_xyz','path_o','path_p','path_iou','path_E',
      'normal_s','normal_o','normal_p','normal_iou','normal_E','N','null_residual','gt','target','energy_u','energy_delta','energy_metrics']
data={k:np.round(d[k],10).tolist() for k in keys}
# W przeglądarce pokazujemy co czwartą próbkę chmury, bez interpolacji odpowiedzi.
data['volume_xyz']=np.round(d['volume_xyz'][:,::4].reshape(-1,3),7).tolist()
data['volume_metrics']=np.round(d['volume_metrics'][:,::4].reshape(-1,3),9).tolist()
data['volume_o']=np.round(d['volume_o'][:,::4].reshape(-1,5),9).tolist()
data['image']='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
template=r'''<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dolina odpowiedzi cząsteczki 226</title>
<style>
:root{color-scheme:dark;--bg:#060b13;--panel:#0b141f;--line:#26394a;--cyan:#45dee3;--gold:#ffda70;--muted:#98aec1;--orange:#ff8d57}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#e7f0f8;font:16px 'Segoe UI',sans-serif}main{max-width:1600px;margin:auto;padding:28px}
h1{font-size:32px;margin:0 0 9px}h2{font-size:23px;margin:0 0 10px}p{line-height:1.55;color:var(--muted);margin:8px 0 16px}.lead{max-width:1150px}
.layout{display:grid;grid-template-columns:minmax(0,2.1fr) minmax(310px,1fr);gap:18px}.panel{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:20px;min-width:0}
#spatial{height:610px}#energy{height:430px}#components{height:270px}.caption{color:var(--muted);font-size:14px;margin-top:8px}.kpis{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.kpis small{display:block;color:var(--muted);font-size:13px}.kpis strong{display:block;font-size:37px;color:var(--cyan)}
canvas{display:block;width:100%;height:auto;border-radius:5px}label{display:block;margin:12px 0 4px;color:var(--muted)}input[type=range]{width:100%;accent-color:var(--cyan)}select,button{font:inherit;color:#e7f0f8;background:#162637;border:1px solid #34516b;border-radius:5px;padding:9px 13px}button{cursor:pointer}button:hover{border-color:var(--cyan)}.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.row label{margin:0}.small{font-size:14px}.controls{margin-top:10px;padding-top:14px;border-top:1px solid var(--line)}.thresholds{display:grid;grid-template-columns:1fr 1fr;gap:20px}.section{margin-top:20px}.pill{color:var(--gold);font-size:14px}details{margin-top:16px}summary{cursor:pointer;color:var(--cyan)}.dirgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(37px,1fr));gap:4px;margin:12px 0}.dirgrid button{padding:5px 0;font-size:12px}.dirgrid button.active{background:#166d75;border-color:var(--cyan)}.outputs{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.outputs small{color:var(--muted)}.outputs b{display:block;margin-top:5px;font-size:20px}.status{font-size:14px;color:var(--gold);min-height:23px}.legend{font-size:14px;color:var(--muted);margin:12px 0}.legend span{margin-right:15px}.muted{color:var(--muted)}footer{margin:25px 0;color:var(--muted);font-size:14px}.statline{display:flex;gap:18px;flex-wrap:wrap;color:var(--cyan);margin:12px 0}
@media(max-width:950px){main{padding:14px}.layout{grid-template-columns:1fr}#spatial{height:530px}.kpis{margin:15px 0}.panel{padding:14px}h1{font-size:27px}}
</style><script>__PLOTLY__</script></head><body><main>
<h1>Jedna odpowiedź. Wiele położeń cząsteczki.</h1>
<p class="lead">Błękitna powierzchnia pokazuje rzeczywiste położenia tokenu 226 dające <b style="color:var(--cyan)">pewność 99,5% i tę samą ramkę</b>. To wybrany wycinek zbioru w przestrzeni 192D. Dwa kierunki styczne mieszczą się na ekranie; lokalnie jest ich 187.</p>
<div class="layout"><section class="panel"><h2>Położenia w latencie</h2>
<div id="spatial" aria-label="Obracany wycinek zbioru stałej odpowiedzi detektora"></div>
<div class="caption">Obracaj myszą. Kliknij punkt powierzchni albo złotą próbkę, aby zobaczyć jej odpowiedź. Wysokość to rzut na kierunek n, a nie błąd.</div>
<div class="controls"><div class="row"><label for="route">Ruch cząsteczki</label><select id="route"><option value="sheet">Po powierzchni — stała odpowiedź</option><option value="normal">W poprzek — zmienna odpowiedź</option></select><button id="play">▶ Odtwórz</button></div>
<label for="position">Położenie na ścieżce <span id="positionLabel"></span></label><input id="position" type="range" min="0" max="360" value="0" aria-label="Położenie cząsteczki">
<div class="row"><label><input id="cloud" type="checkbox"> Pokaż próbki dobrej odpowiedzi</label><span id="count" class="pill"></span></div>
<div class="thresholds"><div><label for="probThreshold">Minimalna pewność: <b id="probLabel">99,00%</b></label><input id="probThreshold" type="range" min="0.95" max="0.999" step="0.0005" value="0.99"></div><div><label for="iouThreshold">Minimalne IoU: <b id="iouLabel">0,900</b></label><input id="iouThreshold" type="range" min="0.7" max="1" step="0.005" value="0.9"></div></div>
<div class="caption">Chmura jest próbką rodziny trzech parametrów. Nie pokazuje całej objętości dopuszczalnych położeń w 192D. Progi są wybrane do eksploracji.</div></div></section>
<aside class="panel"><h2>Odpowiedź tej samej głowy</h2><p class="small">Zbliżenie zdjęcia. Wagi głowy przez cały pokaz pozostają zamrożone.</p><canvas id="image" width="690" height="315" aria-label="Narzędzie z ramką oznaczoną i ramką detektora"></canvas>
<div class="legend"><span style="color:white">━ ramka oznaczona</span><span style="color:var(--cyan)">━ ramka detektora</span></div>
<div class="kpis"><div><small>PEWNOŚĆ</small><strong id="pValue">99,50%</strong></div><div><small>IoU Z OZNACZENIEM</small><strong id="iouValue">1,0000</strong></div></div>
<div id="status" class="status"></div><label>Odległości do krawędzi · kratki</label><div class="outputs"><div><small>Lewa</small><b id="l"></b></div><div><small>Góra</small><b id="t"></b></div><div><small>Prawa</small><b id="r"></b></div><div><small>Dół</small><b id="b"></b></div></div>
<p class="small" style="margin-top:25px">Niezgodność odpowiedzi: <b id="Evalue"></b></p>
<hr style="border:0;border-top:1px solid var(--line);margin:24px 0"><p class="small">Po treningu, przed dodatkową optymalizacją tokenu: <b>94,95% · IoU 0,8858</b>.</p><p class="small">Powierzchnię skonstruowano wokół przykładowego położenia o pewności 99,5% i idealnej ramce. Ten punkt nie jest jedynym celem ani dowiedzionym najbliższym rozwiązaniem. Nie sprawdzamy tu, czy cały ViT potrafi go wytworzyć dla tego zdjęcia.</p>
<div class="statline"><b>192 współrzędne</b><b>5 liczb wyjścia</b><b>187 kierunków stycznych</b></div></aside></div>
<section class="panel section"><h2>Wszystkie 187 kierunków: ich rzeczywiste składowe w 192D</h2><p>Wybierz kierunek. Wykres pokazuje jego 192 współrzędne w tej samej bazie stanu tokenu. Kierunki d₁ i d₂ tworzą pokazany wycinek powierzchni. Baza jest obliczona w jej środku; przy dalszym ruchu kierunki styczne mogą się zmieniać.</p>
<div class="row"><label for="direction">Kierunek</label><select id="direction" aria-label="Wybierz jeden ze 187 kierunków"></select><span id="nullCheck" class="pill"></span></div><div id="components"></div>
<details><summary>Pokaż 187 przycisków kierunków</summary><div id="dirgrid" class="dirgrid"></div></details>
<p class="small">Prosta styczna zachowuje odpowiedź tylko w pierwszym przybliżeniu. Błękitna powierzchnia uwzględnia krzywiznę: pięć dodatkowych współrzędnych jest korygowanych tak, by wszystkie pięć liczb wyjścia pozostało stałe.</p></section>
<section class="panel section"><h2>Dolina niezgodności odpowiedzi</h2><p>Teraz wysokość oznacza miarę niezgodności E, a nie położenie w latencie. Wzdłuż błękitnego dna odpowiedź jest zgodna z wybranym celem. Ruch poprzeczny podnosi E. Pokazujemy jeden kierunek wzdłuż powierzchni i jedną normalną.</p><div id="energy" aria-label="Dolina miary niezgodności odpowiedzi"></div>
<p class="small">E = (logit − logit celu)² + suma kwadratów względnych błędów czterech odległości. To miara na potrzeby wizualizacji, a nie oryginalna strata uczenia ani energia pola laserów MLP.</p></section>
<footer>MalyViT s47 · checkpoint 65 640 · zdjęcie z historii eafb5a973be3d490 · token 226 · stan po ostatnim bloku, przed końcową LayerNorm (192D). Wewnętrzne aktywacje MLP mają w tym modelu 768 składowych. 1089 próbek powierzchni, błąd pięciu równań poniżej 10⁻¹². Współrzędne i pełna precyzja: geometria.npz i POMIAR.json.</footer>
</main><script>
'use strict';
const D=__DATA__;
const $=id=>document.getElementById(id),fmt=(v,n=2)=>Number(v).toFixed(n).replace('.',',');
const baseLayout={paper_bgcolor:'#0b141f',plot_bgcolor:'#0b141f',font:{family:'Segoe UI, sans-serif',color:'#d9e8f3'},margin:{l:0,r:0,t:10,b:0}};
const ax=(title)=>({title:{text:title},gridcolor:'#26394a',zerolinecolor:'#426079',backgroundcolor:'#0b141f',showbackground:true});
const config={responsive:true,displaylogo:false};
const sx=D.sheet_xyz.map(r=>r.map(p=>p[0])),sy=D.sheet_xyz.map(r=>r.map(p=>p[1])),sz=D.sheet_xyz.map(r=>r.map(p=>p[2]));
const initSheet=D.target;
let lastO=initSheet,lastP=.995,lastI=1,lastE=0,playing=false,animationHandle=null;
const sheetTrace={type:'surface',x:sx,y:sy,z:sz,colorscale:[[0,'#175d69'],[1,'#45dee3']],surfacecolor:sz.map(r=>r.map(()=>.6)),showscale:false,opacity:.9,name:'Ta sama odpowiedź',customdata:D.sheet_o.map((r,j)=>r.map((o,i)=>[j,i])),hovertemplate:'d₁=%{x:.2f}<br>d₂=%{y:.2f}<br>rzut n=%{z:.3f}<br>p=99,5% · IoU=1<extra></extra>',contours:{x:{show:true,color:'#77d4df',width:1},y:{show:true,color:'#77d4df',width:1}}};
const cloudTrace={type:'scatter3d',mode:'markers',x:[],y:[],z:[],marker:{size:2,color:'#ffda70',opacity:.2},name:'Dobra odpowiedź',hovertemplate:'Próbka dobrej odpowiedzi<extra></extra>'};
const particle={type:'scatter3d',mode:'markers',x:[0],y:[0],z:[0],marker:{size:7,color:'#ffda70',line:{color:'#fff9db',width:2}},name:'Cząsteczka 226',hoverinfo:'skip'};
const pathTrace={type:'scatter3d',mode:'lines',x:D.path_xyz.map(p=>p[0]),y:D.path_xyz.map(p=>p[1]),z:D.path_xyz.map(p=>p[2]),line:{color:'#ffda70',width:3},name:'Ścieżka po powierzchni',hoverinfo:'skip'};
const arrows=[[[0,0,0],[2.7,0,0],'d₁','#45dee3'],[[0,0,0],[0,2.7,0],'d₂','#45dee3'],[[0,0,0],[0,0,3],'n','#b991fb']].map(([a,b,label,col])=>({type:'scatter3d',mode:'lines+text',x:[a[0],b[0]],y:[a[1],b[1]],z:[a[2],b[2]],text:['',label],textfont:{size:20,color:col},line:{color:col,width:5},showlegend:false,hoverinfo:'skip'}));
const spatialLayout={...baseLayout,uirevision:'fixed-camera',legend:{x:0,y:1,bgcolor:'#0b141f99'},scene:{xaxis:ax('d₁ · współrzędna'),yaxis:ax('d₂ · współrzędna'),zaxis:{...ax('rzut na n'),range:[-.5,3.2]},aspectmode:'data',camera:{eye:{x:1.5,y:-1.8,z:1.2}}}};
const plotReady=Plotly.newPlot('spatial',[sheetTrace,cloudTrace,particle,pathTrace,...arrows],spatialLayout,config);
function setOutput(o,p,iou,E){lastO=o;lastP=p;lastI=iou;lastE=E;$('pValue').textContent=fmt(p*100)+'%';$('iouValue').textContent=fmt(iou,4);$('Evalue').textContent=E<1e-10?'≈ 0':fmt(E,5);['l','t','r','b'].forEach((id,i)=>$(id).textContent=fmt(Math.exp(o[i+1]),3));const pg=p>=+$('probThreshold').value,ig=iou>=+$('iouThreshold').value;$('pValue').style.color=pg?'#45dee3':'#ff8d57';$('iouValue').style.color=ig?'#45dee3':'#ff8d57';$('status').textContent=pg&&ig?'Spełnia oba wybrane progi.':pg?'Pewność spełnia próg, ale IoU jest za niskie.':ig?'IoU spełnia próg, ale pewność jest za niska.':'Pewność i IoU są poniżej progów.';paintImage();}
const photo=new Image();photo.onload=()=>paintImage();photo.src=D.image;
function paintImage(){if(!photo.complete||!photo.naturalWidth)return;const cv=$('image'),c=cv.getContext('2d');c.drawImage(photo,150,75,230,105,0,0,690,315);const bb=[264-Math.exp(lastO[1])*16,120-Math.exp(lastO[2])*16,264+Math.exp(lastO[3])*16,120+Math.exp(lastO[4])*16];const rect=(b,col,width)=>{c.strokeStyle=col;c.lineWidth=width;c.strokeRect((b[0]-150)*3,(b[1]-75)*3,(b[2]-b[0])*3,(b[3]-b[1])*3);};rect(D.gt,'#fff',4);rect(bb,'#45dee3',2);c.fillStyle='#ffda70';c.beginPath();c.arc((264-150)*3,(120-75)*3,4,0,Math.PI*2);c.fill();}
function move(){const idx=+$('position').value,normal=$('route').value==='normal';let xyz,o,p,iou,E;if(normal){const s=D.normal_s[idx];xyz=[0,0,s];o=D.normal_o[idx];p=D.normal_p[idx];iou=D.normal_iou[idx];E=D.normal_E[idx];$('positionLabel').textContent='· przesunięcie '+fmt(s);}else{xyz=D.path_xyz[idx];o=D.path_o[idx];p=D.path_p[idx];iou=D.path_iou[idx];E=D.path_E[idx];$('positionLabel').textContent='· próbka '+(idx+1)+' / 361';}Plotly.restyle('spatial',{x:[[xyz[0]]],y:[[xyz[1]]],z:[[xyz[2]]]},[2]);setOutput(o,p,iou,E);}
$('position').addEventListener('input',move);
$('route').addEventListener('change',()=>{$('position').max=$('route').value==='normal'?240:360;$('position').value=$('route').value==='normal'?120:0;updateRange();move();});
function updateRange(){const normal=$('route').value==='normal',cloud=$('cloud').checked;Plotly.relayout('spatial',{'scene.zaxis.range':normal?[-12.5,12.5]:cloud?[-8.5,11]:[-.5,3.2]});}
let thresholdTimer;
function updateCloud(){const pmin=+$('probThreshold').value,imin=+$('iouThreshold').value;$('probLabel').textContent=fmt(100*pmin)+'%';$('iouLabel').textContent=fmt(imin,3);const inds=[];D.volume_metrics.forEach((m,i)=>{if(m[0]>=pmin&&m[1]>=imin)inds.push(i);});$('count').textContent=inds.length+' / '+D.volume_metrics.length+' próbek';const show=$('cloud').checked;Plotly.restyle('spatial',{x:[show?inds.map(i=>D.volume_xyz[i][0]):[]],y:[show?inds.map(i=>D.volume_xyz[i][1]):[]],z:[show?inds.map(i=>D.volume_xyz[i][2]):[]],customdata:[inds]},[1]);setOutput(lastO,lastP,lastI,lastE);updateRange();}
for(const id of ['probThreshold','iouThreshold'])$(id).addEventListener('input',()=>{clearTimeout(thresholdTimer);thresholdTimer=setTimeout(updateCloud,100);});$('cloud').addEventListener('change',updateCloud);
function stop(){playing=false;clearTimeout(animationHandle);$('play').textContent='▶ Odtwórz';}
function tick(){if(!playing)return;const s=$('position');s.value=(+s.value+2)%(+s.max+1);move();animationHandle=setTimeout(tick,65);}
$('play').addEventListener('click',()=>{if(playing)stop();else{playing=true;$('play').textContent='Ⅱ Zatrzymaj';tick();}});
plotReady.then(()=>{$('spatial').on('plotly_click',ev=>{stop();const p=ev.points[0];if(p.curveNumber===0&&Array.isArray(p.customdata)){const [j,i]=p.customdata,xyz=D.sheet_xyz[j][i];Plotly.restyle('spatial',{x:[[xyz[0]]],y:[[xyz[1]]],z:[[xyz[2]]]},[2]);setOutput(D.sheet_o[j][i],D.sheet_p[j][i],D.sheet_iou[j][i],0);}else if(p.curveNumber===1){const i=p.customdata,m=D.volume_metrics[i];Plotly.restyle('spatial',{x:[[D.volume_xyz[i][0]]],y:[[D.volume_xyz[i][1]]],z:[[D.volume_xyz[i][2]]]},[2]);setOutput(D.volume_o[i],m[0],m[1],m[2]);}});updateCloud();move();});
const componentLayout={...baseLayout,margin:{l:55,r:15,t:12,b:50},xaxis:{...ax('współrzędna stanu tokenu'),range:[.5,192.5]},yaxis:ax('składowa kierunku'),showlegend:false};
Plotly.newPlot('components',[{type:'bar',x:Array.from({length:192},(_,i)=>i+1),y:D.N.map(row=>row[0]),marker:{color:'#45dee3'},hovertemplate:'Współrzędna %{x}<br>składowa %{y:.5f}<extra></extra>'}],componentLayout,config);
for(let i=0;i<187;i++){const op=document.createElement('option');op.value=i;op.textContent='d'+(i+1);$('direction').appendChild(op);const b=document.createElement('button');b.textContent=i+1;b.id='dir'+i;b.setAttribute('aria-label','Pokaż kierunek '+(i+1));b.addEventListener('click',()=>{$('direction').value=i;selectDirection();});$('dirgrid').appendChild(b);}
function selectDirection(){const i=+$('direction').value;Plotly.restyle('components',{y:[D.N.map(row=>row[i])]},[0]);$('nullCheck').textContent='J · d'+(i+1)+' ≈ 0 · kierunek jednostkowy w 192D';for(let j=0;j<187;j++)$('dir'+j).classList.toggle('active',i===j);}
$('direction').addEventListener('change',selectDirection);selectDirection();
const eu=D.energy_u,ed=D.energy_delta,ez=D.energy_metrics.map(row=>row.map(m=>m[2]));
Plotly.newPlot('energy',[{type:'surface',x:eu,y:ed,z:ez,colorscale:[[0,'#45dee3'],[.5,'#446c87'],[1,'#ff8d57']],showscale:false,hovertemplate:'wzdłuż d₁=%{x:.2f}<br>poprzecznie=%{y:.2f}<br>E=%{z:.5f}<extra></extra>'},{type:'scatter3d',mode:'lines',x:eu,y:eu.map(()=>0),z:eu.map(()=>0),line:{color:'#81f8f5',width:7},name:'Dno: ta sama odpowiedź'}],{...baseLayout,scene:{xaxis:ax('wzdłuż d₁'),yaxis:ax('poprzecznie · δ'),zaxis:ax('niezgodność E'),aspectmode:'manual',aspectratio:{x:1.5,y:1.4,z:.8},camera:{eye:{x:1.5,y:-1.9,z:1.1}}}},config);
window.dolinaReady=true;
</script></body></html>'''
html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
dest=OUT/'dolina_interaktywna.html'
dest.write_text(html,encoding='utf-8')
print(dest,round(dest.stat().st_size/1024/1024,2),'MiB')
