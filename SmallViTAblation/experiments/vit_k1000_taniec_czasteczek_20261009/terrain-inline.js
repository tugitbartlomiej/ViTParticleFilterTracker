// Landscape derived exclusively from recorded projected trajectories, not a loss landscape.
let terrainData=null,terrainTicket=0,terrainCamera=null;
let terrainStep=0,terrainPlaying=false,terrainLastTime=0,terrainMarkerState=null,terrainBusy=false;
function terrainBracket(steps,step){let a=Math.max(0,steps.findLastIndex(s=>s<=step)),b=Math.min(a+1,steps.length-1);return {a,b,u:b===a?0:(step-steps[a])/(steps[b]-steps[a])};}
function terrainPause(){terrainPlaying=false;el('terrain-play').textContent='Odtwórz';}
async function terrainMove(step){
 terrainStep=Math.max(0,Math.min(500,step));el('terrain-time').value=terrainStep;el('terrain-time-label').textContent=`Krok ${terrainStep.toFixed(1)} / 500`;
 if(typeof trajectoryActive!=='undefined'&&trajectoryActive){await trajectoryGPU.setStep(terrainStep);return;}
 if(typeof globalDensityData!=='undefined'&&el('terrain-scope')?.value==='global'){await moveGlobalDensity();return;}
 const s=terrainMarkerState,d=terrainData;if(!s||!d||terrainBusy)return;
 const {a,b,u}=terrainBracket(d.steps,terrainStep),current={x:[],y:[],z:[],text:[]};
 for(const p of s.selected){const pos=d.frames[a].xyz[p].map((v,j)=>v+(d.frames[b].xyz[p][j]-v)*u),x=s.ribbon?terrainStep:pos[s.ids[0]],y=s.ribbon?p:pos[s.ids[1]],z=s.ribbon?d.frames[a].norm[p]+(d.frames[b].norm[p]-d.frames[a].norm[p])*u:s.field.sample(x,y);
  current.x.push(x);current.y.push(y);current.z.push(z);current.text.push(`Obraz ${d.image} · patch ${p} · krok ${terrainStep.toFixed(1)}<br>${u===0?'Rzeczywisty zapis':'Interpolacja widoku między '+d.steps[a]+' i '+d.steps[b]}`);
 }
 terrainBusy=true;try{await Plotly.restyle('terrain-plot',{x:[current.x],y:[current.y],z:[current.z],text:[current.text]},[s.trace]);el('terrain-plot').dataset.step=String(terrainStep);el('terrain-status').textContent=`Obraz ${d.image} · 600 cząsteczek × ${d.steps.length} rzeczywistych stanów · ${s.selected.length} śladów · krok ${terrainStep.toFixed(1)} · ${u===0?'rzeczywisty zapis':'interpolacja między '+d.steps[a]+' i '+d.steps[b]}.`;}finally{terrainBusy=false;}
}
async function terrainTick(now){
 if(!terrainPlaying)return;const dt=Math.min((now-terrainLastTime)/1000,.2);terrainLastTime=now;
 await terrainMove(terrainStep+dt*Number(el('terrain-speed').value));
 if(terrainStep>=500){terrainPause();return;}requestAnimationFrame(terrainTick);
}
const terrainColors=['#aebdca','#62d4dd','#ffb967','#d88fe5','#a9d776','#8da6ff','#f0909d'];
const terrainFamilies=['Niewielkie przemieszczenie','+PC1','−PC1','+PC2','−PC2','+PC3','−PC3'];
function terrainQuantile(a,q){const s=a.slice().sort((x,y)=>x-y);return s[Math.floor((s.length-1)*q)]||0;}
function terrainDynamics(frames){
 const T=frames.length,N=600,velocity=Array.from({length:T},()=>new Float32Array(N)),turn=Array.from({length:T},()=>new Float32Array(N));
 for(let t=1;t<T;t++)for(let p=0;p<N;p++){
  const d=frames[t].xyz[p].map((v,j)=>v-frames[t-1].xyz[p][j]);const len=Math.hypot(...d);velocity[t][p]=len/(frames[t].step-frames[t-1].step);
  if(t>1){const prev=frames[t-1].xyz[p].map((v,j)=>v-frames[t-2].xyz[p][j]);const norm=Math.hypot(...prev);if(len>1e-8&&norm>1e-8)turn[t][p]=Math.acos(Math.max(-1,Math.min(1,d.reduce((s,v,j)=>s+v*prev[j],0)/(len*norm))))*180/Math.PI;}
 }
 const final=frames[T-1].xyz.map((v,p)=>v.map((x,j)=>x-frames[0].xyz[p][j])),cut=terrainQuantile(final.map(v=>Math.hypot(...v)),.1);
 const families=final.map(v=>{if(Math.hypot(...v)<=cut)return 0;let j=0;for(let k=1;k<3;k++)if(Math.abs(v[k])>Math.abs(v[j]))j=k;return 1+j*2+(v[j]<0?1:0);});
 return {velocity,turn,families,cut};
}
async function buildTerrain(){
 if(typeof closeTrajectories==='function')closeTrajectories();
 if(el('terrain-scope')?.value==='global'){await buildGlobalDensity();return;}
 if(!META)return;stop();terrainPause();terrainMarkerState=null;const ticket=++terrainTicket,image=selectedImage(),button=el('terrain-build');button.disabled=true;el('terrain-status').textContent='Odczyt zapisanych trajektorii…';
 try{
  const steps=META.global_steps||[0,100,500],frames=new Array(steps.length);
  for(let start=0;start<steps.length;start+=4){const ids=steps.slice(start,start+4);const batch=await Promise.all(ids.map(step=>api('/api/frame?'+new URLSearchParams({mode:'image',image,step,movement:0,trails:0}))));batch.forEach((f,j)=>{if(f.step!==ids[j]||f.xyz.length!==600||f.norm.length!==600)throw Error('Niezgodny zapis trajektorii');frames[start+j]=f;});if(ticket!==terrainTicket)return;el('terrain-status').textContent=`Wczytano ${Math.min(start+4,steps.length)} / ${steps.length} zapisów obrazu ${image}.`;}
  terrainData={image,steps,frames,...terrainDynamics(frames)};terrainCamera=null;if(typeof loadToolMask==='function')await loadToolMask(image);await renderTerrain();
 }catch(e){el('terrain-status').textContent='Nie udało się odczytać trajektorii: '+e.message;}finally{button.disabled=false;}
}
function terrainField(data,ids,kind){
 const G=64,points=[];let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
 for(let t=0;t<data.frames.length;t++)for(let p=0;p<600;p++){const a=data.frames[t].xyz[p];const x=a[ids[0]],y=a[ids[1]];lo[0]=Math.min(lo[0],x);lo[1]=Math.min(lo[1],y);hi[0]=Math.max(hi[0],x);hi[1]=Math.max(hi[1],y);points.push([x,y,data.velocity[t][p]]);}
 const span=lo.map((v,j)=>Math.max(hi[j]-v,.01));lo=lo.map((v,j)=>v-span[j]*.035);hi=hi.map((v,j)=>v+span[j]*.035);const dx=(hi[0]-lo[0])/(G-1),dy=(hi[1]-lo[1])/(G-1),count=new Float64Array(G*G),sum=new Float64Array(G*G);
 for(const [x,y,v] of points){const ix=Math.max(0,Math.min(G-1,Math.round((x-lo[0])/dx))),iy=Math.max(0,Math.min(G-1,Math.round((y-lo[1])/dy)));count[iy*G+ix]++;sum[iy*G+ix]+=v;}
 const c=new Float64Array(G*G),s=new Float64Array(G*G);
 for(let y=0;y<G;y++)for(let x=0;x<G;x++){const n=count[y*G+x];if(!n)continue;for(let a=-3;a<=3;a++)for(let b=-3;b<=3;b++){if(x+b<0||x+b>=G||y+a<0||y+a>=G)continue;const w=Math.exp(-(a*a+b*b)/3.125),i=(y+a)*G+x+b;c[i]+=n*w;s[i]+=sum[y*G+x]*w;}}
 const maximum=Math.max(...c),z=Array.from({length:G},(_,y)=>Array.from({length:G},(_,x)=>{const i=y*G+x;return kind==='speed'?(c[i]>.01?s[i]/c[i]:null):-Math.log((c[i]+.02)/(maximum+.02));}));
 const sample=(x,y)=>{const fx=Math.max(0,Math.min(G-1,(x-lo[0])/dx)),fy=Math.max(0,Math.min(G-1,(y-lo[1])/dy)),a=Math.min(G-2,Math.floor(fx)),b=Math.min(G-2,Math.floor(fy)),u=fx-a,v=fy-b;const zs=[z[b][a],z[b][a+1],z[b+1][a],z[b+1][a+1]];if(zs.some(k=>k===null))return null;return zs[0]*(1-u)*(1-v)+zs[1]*u*(1-v)+zs[2]*(1-u)*v+zs[3]*u*v;};
 return {x:Array.from({length:G},(_,i)=>lo[0]+i*dx),y:Array.from({length:G},(_,i)=>lo[1]+i*dy),z,sample};
}
async function renderTerrain(){
 if(typeof closeTrajectories==='function'&&trajectoryActive)closeTrajectories();
 if(el('terrain-scope')?.value==='global'){await renderGlobalDensity();return;}
 const d=terrainData;if(!d)return;terrainPause();const kind=el('terrain-kind').value,ids=el('terrain-plane').value.split(',').map(Number),chosen=selectedPatch,limit=Number(el('terrain-tracks').value),filter=Number(el('terrain-family').value);let selected=[chosen];
 for(let p=0;p<600&&selected.length<limit;p+=Math.max(1,Math.floor(600/limit)))if(p!==chosen&&(filter<0||d.families[p]===filter))selected.push(p);
 const maskSet=typeof toolPatchSet==='function'?toolPatchSet(d.image):new Set(),maskMode=el('tool-mode')?.value||'off';
 if(maskMode==='only')selected=[...maskSet];else if(maskMode==='highlight')selected=[...new Set([...selected,...maskSet])];
 const labels=ids.map(i=>'PC'+(i+1)),ribbon=kind==='ribbon',field=ribbon?null:terrainField(d,ids,kind),traces=[];
 const ztitle=ribbon?'Ruch L2 w pełnych 768D':kind==='speed'?'Prędkość rzutu PCA / krok':'U = −ln(gęstości względnej)';
 if(ribbon){traces.push({type:'surface',name:'Ruch 768D',x:d.steps,y:Array.from({length:600},(_,i)=>i),z:Array.from({length:600},(_,p)=>d.frames.map(f=>f.norm[p])),colorscale:[[0,'#1d3b6e'],[.35,'#2f6f8f'],[.7,'#6fa8a0'],[1,'#d8d2b0']],opacity:.8,showscale:false,hovertemplate:'Krok %{x}<br>Patch %{y}<br>Ruch 768D %{z:.3f}<extra></extra>'});}
 else traces.push({type:'surface',name:'Krajobraz ruchu',x:field.x,y:field.y,z:field.z,colorscale:[[0,'#1d3b6e'],[.35,'#2f6f8f'],[.7,'#6fa8a0'],[1,'#d8d2b0']],opacity:.72,showscale:false,contours:{z:{show:true,color:'#f0e6c0',width:1,project:{z:false}}},hovertemplate:ztitle+' %{z:.3f}<extra></extra>'});
 const bundles=new Map(),events={x:[],y:[],z:[],text:[]},current={x:[],y:[],z:[],text:[],color:[]};let bad=0;
 let index=d.steps.findLastIndex(s=>s<=actualStep());index=Math.max(0,index);
 for(const p of selected){const xs=[],ys=[],zs=[],texts=[],v=d.velocity.slice(1).map(a=>a[p]),median=terrainQuantile(v,.5),group=d.families[p];
  for(let t=0;t<d.frames.length;t++){const pos=d.frames[t].xyz[p],x=ribbon?d.steps[t]:pos[ids[0]],y=ribbon?p:pos[ids[1]],z=ribbon?d.frames[t].norm[p]:field.sample(x,y);xs.push(x);ys.push(y);zs.push(z);texts.push(`Obraz ${d.image} · patch ${p} · krok ${d.steps[t]}<br>${terrainFamilies[group]}<br>Prędkość rzutu ${d.velocity[t][p].toFixed(4)} · zwrot ${d.turn[t][p].toFixed(1)}°`);
   if(t>1&&z!==null&&(d.turn[t][p]>90||(median>1e-8&&d.velocity[t][p]>3*median))){bad++;if(el('terrain-events').checked){events.x.push(x);events.y.push(y);events.z.push(z);events.text.push(texts[t]);}}
  }
  const tool=maskMode!=='off'&&maskSet.has(p),bundleGroup=tool?7:group;
  if(p===chosen)traces.push({type:'scatter3d',mode:'lines',name:`Wybrany patch ${p}${tool?' · narzędzie':''}`,x:xs,y:ys,z:zs,text:texts,hovertemplate:'%{text}<extra></extra>',line:{width:7,color:tool?'#ffdc55':'#fff5dd'}});
  else {if(!bundles.has(bundleGroup))bundles.set(bundleGroup,{x:[],y:[],z:[],text:[]});const bundle=bundles.get(bundleGroup);for(const [key,values] of [['x',xs],['y',ys],['z',zs],['text',texts]]){bundle[key].push(...values);bundle[key].push(null);}}

  if(zs[index]!==null){current.x.push(xs[index]);current.y.push(ys[index]);current.z.push(zs[index]);current.text.push(texts[index]);current.color.push(tool?'#ffdc55':p===chosen?'#fff5dd':terrainColors[group]);}
 }
 for(const [group,bundle] of bundles)traces.push({type:'scatter3d',mode:'lines',name:group===7?'Patche narzędzia':terrainFamilies[group],...bundle,hovertemplate:'%{text}<extra></extra>',line:{width:group===7?4:2.5,color:group===7?'#ffdc55':terrainColors[group]},opacity:.72});
 const markerTrace=traces.length;
 traces.push({type:'scatter3d',mode:'markers',name:'Bieżące położenia',...current,marker:{size:5,color:current.color},hovertemplate:'%{text}<extra></extra>'});
 if(events.x.length)traces.push({type:'scatter3d',mode:'markers',name:'Zwroty / skoki prędkości',...events,marker:{size:3,color:'#ff556d',symbol:'diamond'},hovertemplate:'%{text}<extra></extra>'});
 const axis={gridcolor:C.grid,color:C.text,zerolinecolor:'#70838c',showbackground:false},layout={...commonLayout(),height:660,uirevision:`terrain-${d.image}-${kind}-${ids}`,legend:{orientation:'h',y:1.13},margin:{l:0,r:0,t:55,b:0},scene:{xaxis:{...axis,title:ribbon?'Krok optymalizacji':labels[0]},yaxis:{...axis,title:ribbon?'Numer patcha':labels[1]},zaxis:{...axis,title:ztitle},aspectmode:'manual',aspectratio:{x:1,y:1,z:.55},camera:terrainCamera||{eye:{x:1.5,y:-1.5,z:.9}}}};
 await Plotly.react('terrain-plot',traces,layout,CONFIG);const plot=el('terrain-plot');plot.dataset.image=String(d.image);plot.dataset.frames=String(d.steps.length);plot.dataset.tracks=String(selected.length);plot.dataset.kind=kind;plot.dataset.events=String(bad);plot.dataset.step=String(d.steps[index]);plot.dataset.families=JSON.stringify(d.families.reduce((s,k)=>(s[k]=(s[k]||0)+1,s),{}));
 plot.removeAllListeners?.('plotly_click');plot.on('plotly_click',event=>{const text=event.points?.[0]?.text,match=typeof text==='string'&&text.match(/patch (\d+)/);if(match){chooseParticle(d.image,Number(match[1]));void renderTerrain();}});
 el('terrain-status').textContent=`Obraz ${d.image} · 600 cząsteczek × ${d.steps.length} stanów · ${selected.length} śladów · znacznik kroku ${d.steps[index]} · ${bad} zdarzeń na pokazanych śladach.`;
 el('terrain-note').textContent=ribbon?'Powierzchnia pokazuje zmierzoną normę ruchu 768D. Patch jest numerem przestrzennym, nie reżimem.':kind==='speed'?'Wysokość: wygładzona średnia prędkość rzutu PCA w danym miejscu. Brak danych pozostawia przerwy w powierzchni.':'Dolina oznacza wysoką gęstość zapisanych położeń w rzucie; to empiryczna mapa odwiedzin, nie zmierzony potencjał fizyczny ani funkcja błędu. Ślady są podniesione na powierzchnię przez interpolację jej wysokości.';
 el('terrain-note').textContent+=' Kolory rodzin: dominująca oś końcowego przesunięcia w PCA; szary: norma do 10. percentyla. Czerwone zdarzenia: zwrot >90° lub prędkość >3× mediana danej trajektorii. To opisowe podziały i anomalie rzutu, nie dowód reżimów Koopmana ani matematycznych osobliwości. Kliknij ślad, by wskazać odpowiadający patch.';
 terrainMarkerState={selected,ids,ribbon,field,trace:markerTrace};await terrainMove(terrainStep);
}
el('terrain-build').onclick=buildTerrain;for(const id of ['terrain-kind','terrain-tracks','terrain-family','terrain-events'])el(id).onchange=()=>void renderTerrain();
el('terrain-plane').onchange=()=>{if(typeof trajectoryActive!=='undefined'&&trajectoryActive)trajectoryGPU.draw();else void renderTerrain();};
el('terrain-update').onclick=()=>{terrainStep=actualStep();void renderTerrain();};
el('terrain-play').onclick=()=>{if(terrainPlaying){terrainPause();return;}if(!terrainData&&!(typeof globalDensityData!=='undefined'&&globalDensityData)&&!(typeof trajectoryActive!=='undefined'&&trajectoryActive))return;stop();if(terrainStep>=500)terrainStep=0;terrainPlaying=true;terrainLastTime=performance.now();el('terrain-play').textContent='Pauza';requestAnimationFrame(terrainTick);};
el('terrain-reset').onclick=()=>{terrainPause();void terrainMove(0);};
el('terrain-time').oninput=()=>{terrainPause();void terrainMove(Number(el('terrain-time').value));};
let terrainDebounce;for(const id of ['step','patch-select'])el(id).addEventListener('input',()=>{clearTimeout(terrainDebounce);terrainDebounce=setTimeout(()=>{if(terrainData)void renderTerrain();},150);});
