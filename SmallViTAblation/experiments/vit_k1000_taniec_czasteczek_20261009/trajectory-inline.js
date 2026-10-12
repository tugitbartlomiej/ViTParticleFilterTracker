let trajectoryGPU=null,trajectoryActive=false,trajectoryTicket=0;
function showTrajectoryHost(show){el('terrain-traj-plot').classList.toggle('hidden',!show);el('terrain-plot').classList.toggle('hidden',show);}
function trajectorySelectedDefault(){if(!el('trajectory-images').value.trim())el('trajectory-images').value=String(selectedImage());}
async function buildTrajectories(){
 terrainPause();stop();const ticket=++trajectoryTicket,scope=el('trajectory-scope').value;trajectorySelectedDefault();
 try{
  if(!trajectoryGPU)trajectoryGPU=new ParticleTrajectoriesGPU(el('terrain-traj-plot'),el('terrain-status'));
  trajectoryActive=true;showTrajectoryHost(true);trajectoryGPU.setColor(el('trajectory-color').value);
  if(scope==='selected'){
   const images=parseTrajectoryImages(el('trajectory-images').value),plan=trajectoryPlan(scope,images.length,10);await trajectoryGPU.selected(images.join(','));if(ticket!==trajectoryTicket)return;
   Object.assign(el('terrain-traj-plot').dataset,{plan:JSON.stringify(plan),scope:'selected'});
  }else{
   const tail=Number(el('trajectory-tail').value),plan=trajectoryPlan(scope,0,tail),meta=await api('/api/frame?mode=global&image=0&step=0&movement=0&trails=0&geometry=0');if(ticket!==trajectoryTicket)return;
   meta.steps=META.global_steps;await trajectoryGPU.global(meta,terrainStep,tail);if(ticket!==trajectoryTicket)return;
   Object.assign(el('terrain-traj-plot').dataset,{plan:JSON.stringify(plan),scope:'global'});
  }
  el('terrain-note').textContent=scope==='selected'?'Pełne trajektorie wszystkich 600 patchy każdego wybranego zdjęcia. Ciemniejsza linia pokazuje całą drogę 0–500, a jaśniejsza część — drogę przebytą do bieżącego kroku.':'Każdy widoczny odcinek jest rzeczywistym ruchem jednej z 900 000 cząsteczek między zapisami co 5 kroków. Regulowany ogon zachowuje poprzednie przedziały. Nie zastosowano próbkowania cząsteczek.';
 }catch(error){trajectoryActive=false;showTrajectoryHost(false);el('terrain-status').textContent='Nie można wczytać trajektorii: '+error.message;}
}
function closeTrajectories(){trajectoryActive=false;++trajectoryTicket;showTrajectoryHost(false);}
el('trajectory-build').onclick=()=>void buildTrajectories();
el('trajectory-scope').onchange=()=>{const selected=el('trajectory-scope').value==='selected';el('trajectory-images-wrap').classList.toggle('hidden',!selected);el('trajectory-tail-wrap').classList.toggle('hidden',selected);el('trajectory-build').textContent=selected?'Wczytaj pełne trajektorie wybranych zdjęć':'Wczytaj trajektorie wszystkich 1500 zdjęć';};
el('trajectory-color').onchange=()=>{trajectoryGPU?.setColor(el('trajectory-color').value);};
el('trajectory-tail').onchange=()=>{if(trajectoryActive&&el('trajectory-scope').value==='global')void buildTrajectories();};
