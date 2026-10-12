let globalDensityData=null,globalDensityTicket=0;
const globalDensityCache=new Map();
const globalPlane=()=>({'0,1':0,'0,2':1,'1,2':2}[el('terrain-plane').value]);
function globalDensityZ(data,step,whole){
 if(whole)return data.whole;
 const {a,b,u}=terrainBracket(data.steps,step);
 return data.fields[a].map((row,y)=>row.map((z,x)=>z+(data.fields[b][y][x]-z)*u));
}
async function buildGlobalDensity(){
 terrainPause();stop();const plane=globalPlane(),ticket=++globalDensityTicket;
 el('terrain-status').textContent='Wczytywanie gęstości wszystkich 1500 zdjęć…';
 try{const data=globalDensityCache.get(plane)||await api('/api/raw?terrain=global&plane='+plane);if(ticket!==globalDensityTicket)return;
  if(data.point_count!==900000||data.image_count!==1500||data.steps.length!==101||data.sampling!==false)throw Error('Niepełna populacja gęstości');
  globalDensityCache.set(plane,data);globalDensityData=data;await renderGlobalDensity();
 }catch(e){el('terrain-status').textContent='Nie można wczytać wspólnej gęstości: '+e.message;}
}
async function renderGlobalDensity(){
 terrainPause();if(!globalDensityData||globalDensityData.plane!==globalPlane()){await buildGlobalDensity();return;}
 const d=globalDensityData,ids=el('terrain-plane').value.split(',').map(Number),axis={gridcolor:C.grid,color:C.text,showbackground:false};
 const z=globalDensityZ(d,terrainStep,el('terrain-density-time').value==='whole');
 el('terrain-plot').removeAllListeners?.('plotly_click');
 await Plotly.react('terrain-plot',[{type:'surface',x:d.x,y:d.y,z,colorscale:[[0,'#1d3b6e'],[.35,'#2f6f8f'],[.7,'#6fa8a0'],[1,'#d8d2b0']],cmin:0,cmax:d.zmax,showscale:true,colorbar:{title:'−ln gęstości'},contours:{z:{show:true,color:'#f0e6c0',width:1}},hovertemplate:'PC '+(ids[0]+1)+': %{x:.3f}<br>PC '+(ids[1]+1)+': %{y:.3f}<br>−ln gęstości: %{z:.3f}<extra>Wszystkie 1500 zdjęć</extra>'}],{...commonLayout(),height:660,uirevision:'global-density-'+d.plane,margin:{l:0,r:30,t:20,b:0},scene:{xaxis:{...axis,title:'PC'+(ids[0]+1),range:[d.x[0],d.x.at(-1)]},yaxis:{...axis,title:'PC'+(ids[1]+1),range:[d.y[0],d.y.at(-1)]},zaxis:{...axis,title:'U = −ln gęstości względnej',range:[0,d.zmax]},aspectmode:'manual',aspectratio:{x:1,y:1,z:.55},camera:{eye:{x:1.5,y:-1.5,z:.9}}}},CONFIG);
 el('terrain-note').textContent='Wspólna mapa gęstości wszystkich 1500 zdjęć, bez próbkowania cząsteczek. Te same osie PCA i ta sama siatka 64×64 we wszystkich krokach. Studnie są empiryczną gęstością położeń, nie funkcją błędu. Mapa całego ruchu sumuje odwiedziny od 0 do 500; pojedyncza cząsteczka występuje w niej 101 razy. Zaznaczenie maski po prawej dotyczy wybranego zdjęcia i nie filtruje tej globalnej mapy.';
 await moveGlobalDensity();
}
async function moveGlobalDensity(){
 const d=globalDensityData;if(!d||terrainBusy||d.plane!==globalPlane())return;
 const whole=el('terrain-density-time').value==='whole';terrainBusy=true;
 try{if(!whole)await Plotly.restyle('terrain-plot',{z:[globalDensityZ(d,terrainStep,false)]},[0]);
 const {a,b,u}=terrainBracket(d.steps,terrainStep);
 el('terrain-status').textContent=whole?`Wszystkie 1500 zdjęć · 900 000 cząsteczek · 101 rzeczywistych stanów · ${d.whole_observation_count.toLocaleString('pl-PL')} położeń w całym ruchu 0–500.`:`Wszystkie 1500 zdjęć · 900 000 cząsteczek · krok ${terrainStep.toFixed(1)} · ${u===0?'rzeczywista gęstość zapisu':'interpolacja powierzchni między '+d.steps[a]+' i '+d.steps[b]}.`;
 Object.assign(el('terrain-plot').dataset,{scope:'global',pointCount:'900000',imageCount:'1500',step:String(terrainStep),sampling:'false',densityTime:whole?'whole':'current'});
 }finally{terrainBusy=false;}
}
el('terrain-scope').onchange=()=>{
 if(typeof closeTrajectories==='function')closeTrajectories();
 terrainPause();const global=el('terrain-scope').value==='global';el('terrain-build').textContent=global?'Wczytaj studnie wszystkich 1500 zdjęć':'Wczytaj trajektorie wybranego zdjęcia';
 el('terrain-density-time').disabled=!global;
 for(const id of ['terrain-kind','terrain-tracks','terrain-family','terrain-events'])el(id).disabled=global;
 if(global){el('terrain-kind').value='density';void buildGlobalDensity();}else{++globalDensityTicket;el('terrain-plot').dataset.scope='image';if(terrainData)void renderTerrain();else{Plotly.purge('terrain-plot');el('terrain-note').textContent='';el('terrain-status').textContent='Wczytaj trajektorie wybranego zdjęcia.';}}
};
el('terrain-density-time').onchange=()=>{if(el('terrain-scope').value==='global')void renderGlobalDensity();};
