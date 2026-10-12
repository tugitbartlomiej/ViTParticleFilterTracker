let toolMask=null,toolMaskTicket=0,toolPreview=null;
function toolPatchSet(image){return new Set(toolMask&&toolMask.image===image?(el('tool-region').value==='contour'?toolMask.contour_patches:toolMask.tool_patches):[]);}
async function loadToolMask(image){
 const ticket=++toolMaskTicket;toolMask=null;toolPreview=null;el('tool-status').textContent='Odczyt maski z archiwum adnotacji…';
 try{const data=await api('/api/tool-mask?image='+image);if(ticket!==toolMaskTicket)return;toolMask=data;el('tool-status').textContent=data.mask_available?`Obraz ${image}: ${data.tool_patches.length} patchy narzędzia, ${data.contour_patches.length} patchy obrysu. ${data.instances.map(x=>x.category).join(', ')}.`:'Dla tej klatki brak maski narzędzia. Nie użyto ramki zastępczej.';
  const img=new Image();img.onload=()=>{if(ticket!==toolMaskTicket)return;toolPreview=img;paintToolPreview();};img.src='/api/image?image='+image;
 }catch(e){if(ticket!==toolMaskTicket)return;el('tool-status').textContent='Brak zgodnej maski: '+e.message;}
 syncToolCloud();paintToolPreview();
}
function paintToolPreview(){
 const canvas=el('tool-preview'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,480,320);ctx.fillStyle='#0c1723';ctx.fillRect(0,0,480,320);
 if(!toolMask||!toolPreview)return;
 const maskOnly=el('tool-preview-mode').value==='mask';if(!maskOnly)ctx.drawImage(toolPreview,0,0,480,320);
 ctx.fillStyle=maskOnly?'#ffdc55':'#ffdc5588';ctx.strokeStyle='#ffdc55';ctx.lineWidth=1;
 for(const polygon of toolMask.polygons){ctx.beginPath();polygon.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.stroke();}
 for(const p of toolPatchSet(toolMask.image)){const x=p%30*16,y=Math.floor(p/30)*16;ctx.fillStyle='#ffdc5522';ctx.fillRect(x,y,16,16);ctx.strokeStyle='#ffdc55';ctx.lineWidth=1;ctx.strokeRect(x,y,16,16);}
 const x=selectedPatch%30*16,y=Math.floor(selectedPatch/30)*16;ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.strokeRect(x,y,16,16);
 canvas.dataset.image=toolMask.image;canvas.dataset.toolCount=toolPatchSet(toolMask.image).size;
}
function syncToolCloud(){if(gpuCloud){gpuCloud.toolImage=toolMask?.image??-1;gpuCloud.toolPatches=el('tool-mode').value==='off'?[]:[...toolPatchSet(toolMask?.image)];gpuCloud.requestDraw();}}
for(const id of ['tool-mode','tool-region'])el(id).onchange=()=>{syncToolCloud();paintToolPreview();if(terrainData)void renderTerrain();};
el('tool-preview-mode').onchange=paintToolPreview;
el('tool-load').onclick=async()=>{await loadToolMask(terrainData?.image??selectedImage());if(terrainData)await renderTerrain();};
el('tool-preview').onclick=e=>{if(!toolMask)return;const r=e.currentTarget.getBoundingClientRect(),p=Math.min(19,Math.floor((e.clientY-r.top)/r.height*20))*30+Math.min(29,Math.floor((e.clientX-r.left)/r.width*30));chooseParticle(toolMask.image,p);paintToolPreview();if(terrainData)void renderTerrain();};
el('tool-mosaic-build').onclick=async()=>{
 const host=el('tool-mosaic');host.replaceChildren();const start=Math.floor((terrainData?.image??selectedImage())/8)*8;
 for(let image=start;image<Math.min(1500,start+8);image++){
  const button=document.createElement('button'),canvas=document.createElement('canvas'),label=document.createElement('div');canvas.width=240;canvas.height=160;canvas.style.width='100%';label.textContent='Obraz '+image;button.append(canvas,label);host.append(button);
  button.onclick=async()=>{chooseParticle(image,selectedPatch);await buildTerrain();paintToolPreview();};
  try{const mask=await api('/api/tool-mask?image='+image),img=new Image();img.onload=()=>{const c=canvas.getContext('2d');c.drawImage(img,0,0,240,160);c.scale(.5,.5);c.fillStyle='#ffdc5588';c.strokeStyle='#ffdc55';for(const poly of mask.polygons){c.beginPath();poly.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.closePath();c.fill();}for(const p of mask.tool_patches)c.strokeRect(p%30*16,Math.floor(p/30)*16,16,16);};img.src='/api/image?image='+image;if(!mask.mask_available)label.textContent+=' · brak maski';}catch(e){label.textContent+=' · brak adnotacji';}
 }
};
