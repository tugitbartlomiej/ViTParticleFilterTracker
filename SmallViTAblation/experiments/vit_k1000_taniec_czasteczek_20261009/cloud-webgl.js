/* All 900,000 particles; interpolation changes the display only, never recorded data. */
'use strict';
class ParticleCloudGPU {
  constructor(host, status, detail) {
    this.host=host;this.status=status;this.detail=detail;
    this.canvas=document.createElement('canvas');this.canvas.setAttribute('aria-label','Wszystkie 900 tysięcy cząsteczek, rzut PCA 3D');
    this.canvas.style.cssText='width:100%;height:100%;display:block;touch-action:none;cursor:grab';host.append(this.canvas);
    const gl=this.gl=this.canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:false});
    if(!gl)throw new Error('Ten podgląd wymaga WebGL 2; wybierz widok jednego obrazu lub panel kontrolny.');
    this.program=this.makeProgram();this.cache=new Map();this.pending=new Map();this.count=900000;
    this.yaw=.7;this.pitch=-.42;this.zoom=1;this.opacity=.065;this.pointSize=1.5;this.visible=false;this.spinning=false;this.raf=null;this.frames=[];this.uploadMs=0;
    this.next=null;this.targetMeta=null;this.blend=0;this.transition=null;this.viewBlend=0;
    this.colorMode=0;this.selectedImage=0;this.selectedPatch=0;this.highlight=true;
    this.spaceView=0;this.axisNames=['PC1','PC2','PC3'];this.axisBounds=null;this.showAxes=true;
    const ext=gl.getExtension('WEBGL_debug_renderer_info');this.renderer=ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
    host.dataset.renderer=this.renderer;
    this.installControls();new ResizeObserver(()=>{if(this.visible)this.requestDraw();}).observe(host);
    this.canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.pauseTransition();this.spinning=false;status.textContent='Kontekst GPU został utracony. Odśwież podgląd.';});
    document.addEventListener('visibilitychange',()=>{if(this.transition)this.transition.last=null;if(document.hidden){cancelAnimationFrame(this.raf);this.raf=null;}else if(this.visible)this.requestDraw();});
  }
  makeProgram(){
    const gl=this.gl;
    const vs=`#version 300 es
precision highp float;
precision highp int;
in vec3 a_position;
in vec3 a_next;
in vec3 a_initial;
in float a_norm;
in float a_nextnorm;
uniform vec3 u_center, u_scale;
uniform float u_radius, u_yaw, u_pitch, u_zoom, u_aspect, u_size, u_blend;
uniform bool u_movement;
uniform int u_pass, u_view;
out float v_norm;
flat out int v_id;
vec3 rotatePoint(vec3 p){float c=cos(u_yaw),s=sin(u_yaw);p=vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);c=cos(u_pitch);s=sin(u_pitch);return vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);}
void main(){vec3 p=mix(a_position,a_next,u_blend);v_norm=mix(a_norm,a_nextnorm,u_blend);if(u_movement)p-=a_initial;if(u_view==1)p=vec3(p.xy,0.);if(u_view==2)p=vec3(p.x,p.z,0.);if(u_view==3)p=vec3(p.y,p.z,0.);if(u_view==4)p=vec3(float(gl_VertexID%30),float((gl_VertexID%600)/30),v_norm);p=rotatePoint((p-u_center)/u_scale);gl_Position=vec4(p.x*u_zoom/u_aspect,p.y*u_zoom,-p.z*.1,1.);gl_PointSize=u_size*(u_pass==2?7.:u_pass==1?2.5:1.);v_id=gl_VertexID+1;}`;
    const fs=`#version 300 es
precision highp float;
precision highp int;
in float v_norm;
flat in int v_id;
uniform float u_maxnorm, u_alpha;
uniform bool u_pick;
uniform int u_color, u_pass;
uniform bool u_focus;
out vec4 color;
vec3 identityColor(int id){float h=fract(float(id)*.61803398875);return .25+.75*clamp(abs(fract(h+vec3(0.,.6666667,.3333333))*6.-3.)-1.,0.,1.);}
void main(){if(length(gl_PointCoord-.5)>.5)discard;if(u_pick){int id=v_id;color=vec4(float(id&255),float((id>>8)&255),float((id>>16)&255),255.)/255.;return;}float t=clamp(v_norm/u_maxnorm,0.,1.);vec3 a=vec3(.325,.447,.545),b=vec3(.384,.831,.867),c=vec3(1.,.725,.404);vec3 rgb=t<.45?mix(a,b,t/.45):mix(b,c,(t-.45)/.55);if(u_color>0)rgb=identityColor(u_color==1?(v_id-1)/600:(v_id-1)%600);if(u_pass==2)rgb=vec3(1.,1.,1.);color=vec4(rgb,u_pass>0?.95:u_alpha*(u_focus?.3:1.));}`;
    const compile=(kind,source)=>{const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader;};
    const p=gl.createProgram(),v=compile(gl.VERTEX_SHADER,vs),f=compile(gl.FRAGMENT_SHADER,fs);gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;
  }
  async buffer(step){
    if(this.cache.has(step))return this.cache.get(step);if(this.pending.has(step))return this.pending.get(step);
    const work=(async()=>{const response=await fetch(`/api/cloud.bin?step=${step}`);if(!response.ok)throw new Error('Nie udało się wczytać całej chmury');const raw=await response.arrayBuffer();if(raw.byteLength!==this.count*16)throw new Error('Nieprawidłowy rozmiar chmury');const t=performance.now(),gl=this.gl,b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,raw,gl.STATIC_DRAW);this.uploadMs=performance.now()-t;const result={buffer:b,data:new Float32Array(raw),bytes:raw.byteLength};this.cache.set(step,result);this.pruneCache(step);return result;})();
    this.pending.set(step,work);try{return await work;}finally{this.pending.delete(step);}
  }
  async show(meta,movement){
    this.pauseTransition();
    this.status.textContent='Wczytywanie wszystkich 900 000 punktów do GPU…';
    const [initial,current]=await Promise.all([this.buffer(0),this.buffer(meta.step)]);
    this.initial=initial;this.current=current;this.next=current;this.targetMeta=meta;this.blend=0;this.viewBlend=0;this.meta=meta;this.movement=movement;
    // Bounded streaming cache: initial/current/next/lookahead, never all 101 states.
    const active=this.cache.get(meta.step);this.cache.delete(meta.step);this.cache.set(meta.step,active);
    this.pruneCache();
    this.configureSpace();
    this.visible=true;this.host.classList.remove('hidden');this.frames=[];this.requestDraw();
  }
  pruneCache(keepStep=null){for(const [step,entry] of this.cache){if(this.cache.size<=6)break;if(step!==0&&step!==keepStep&&entry!==this.current&&entry!==this.next){this.gl.deleteBuffer(entry.buffer);this.cache.delete(step);}}}
  prefetch(step){if(step!=null&&!this.cache.has(step)&&this.pending.size<2)this.buffer(step).catch(()=>{});}
  pauseTransition(){const old=this.transition;this.transition=null;this.host.dataset.animating='false';this.frames=[];if(old)old.resolve(false);}
  async transitionTo(meta,getRate,getSmooth,onProgress,isActive){
    const next=await this.buffer(meta.step);
    if(!isActive())return false;
    // Resume a paused pair at its existing fraction. A seek uses show() to reset it.
    if(this.targetMeta?.step!==meta.step){this.blend=0;this.viewBlend=0;}
    this.next=next;this.targetMeta=meta;this.pruneCache();
    return new Promise(resolve=>{this.transition={getRate,getSmooth,onProgress,resolve,last:null};this.requestDraw();});
  }
  hide(){this.pauseTransition();this.visible=false;this.host.classList.add('hidden');cancelAnimationFrame(this.raf);this.raf=null;}
  requestDraw(){if(this.visible&&!document.hidden&&this.raf===null)this.raf=requestAnimationFrame(t=>{this.raf=null;this.draw(t);});}
  bind(){const gl=this.gl,p=this.program;gl.useProgram(p);
    const attr=(name,b,size,offset)=>{const loc=gl.getAttribLocation(p,name);gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,16,offset);};
    attr('a_position',this.current.buffer,3,0);attr('a_norm',this.current.buffer,1,12);attr('a_next',(this.next||this.current).buffer,3,0);attr('a_nextnorm',(this.next||this.current).buffer,1,12);attr('a_initial',this.initial.buffer,3,0);
    const u=name=>gl.getUniformLocation(p,name);
    gl.uniform3fv(u('u_center'),this.center);gl.uniform1f(u('u_radius'),Math.max(this.radius,1e-6));gl.uniform1f(u('u_yaw'),this.yaw);gl.uniform1f(u('u_pitch'),this.pitch);gl.uniform1f(u('u_zoom'),this.zoom);gl.uniform1f(u('u_aspect'),this.canvas.width/this.canvas.height);gl.uniform1f(u('u_size'),this.pointSize*this.dpr);gl.uniform1i(u('u_movement'),this.movement?1:0);gl.uniform1f(u('u_maxnorm'),this.meta.norm_scale);gl.uniform1f(u('u_alpha'),this.opacity);gl.uniform1i(u('u_pick'),0);
    gl.uniform1f(u('u_blend'),this.viewBlend);
    gl.uniform1i(u('u_color'),this.colorMode);gl.uniform1i(u('u_pass'),0);gl.uniform1i(u('u_focus'),this.highlight?1:0);
    gl.uniform1i(u('u_view'),this.spaceView);
    gl.uniform3fv(u('u_scale'),this.spaceView===4?this.axisBounds.map(r=>Math.max((r[1]-r[0])*.7,1e-6)):[this.radius,this.radius,this.radius]);
  }
  setSelection(image,patch,highlight=true){this.selectedImage=image;this.selectedPatch=patch;this.highlight=highlight;this.requestDraw();}
  configureSpace(){if(!this.meta)return;const b=this.meta.bounds,prefix=this.movement?'Δ ':'';if(this.spaceView===4){this.axisBounds=[[-.5,29.5],[-.5,19.5],[0,this.meta.norm_scale]];this.axisNames=['Kolumna patcha','Wiersz patcha','Ruch L2 w 768D'];}else{const ids=this.spaceView===2?[0,2]:this.spaceView===3?[1,2]:[0,1];this.axisBounds=this.spaceView===0?b:[b[ids[0]],b[ids[1]],[-.01,.01]];this.axisNames=this.spaceView===0?[0,1,2].map(i=>prefix+'PC'+(i+1)):[...ids.map(i=>prefix+'PC'+(i+1)),'Płaszczyzna 2D'];}this.center=this.axisBounds.map(r=>(r[0]+r[1])/2);this.radius=Math.sqrt(this.axisBounds.reduce((s,r)=>s+Math.pow((r[1]-r[0])/2,2),0))*1.07;}
  setSpace(view){this.spaceView=view;this.configureSpace();if(view>=1&&view<=3){this.yaw=0;this.pitch=0;}else{this.yaw=.7;this.pitch=-.42;}this.zoom=1;this.requestDraw();}
  selectedPasses(){const gl=this.gl,u=gl.getUniformLocation(this.program,'u_pass');if(this.highlight){gl.uniform1i(u,1);if(!this.toolPatches?.length)gl.drawArrays(gl.POINTS,this.selectedImage*600,600);gl.uniform1i(u,2);gl.drawArrays(gl.POINTS,this.selectedImage*600+this.selectedPatch,1);}if(this.toolPatches?.length&&this.toolImage>=0){gl.uniform1i(u,2);for(const patch of this.toolPatches)gl.drawArrays(gl.POINTS,this.toolImage*600+patch,1);}gl.uniform1i(u,0);}
  draw(now){if(!this.visible||!this.current)return;
    const transition=this.transition;
    if(transition){
      const span=Math.max(1,this.targetMeta.step-this.meta.step);
      if(transition.last!==null)this.blend=Math.min(1,this.blend+Math.max(0,Math.min(200,now-transition.last))*transition.getRate()/1000/span);
      transition.last=now;this.viewBlend=transition.getSmooth()?this.blend:(this.blend>=1?1:0);
      transition.onProgress(this.meta.step+span*this.viewBlend,this.meta.step,this.targetMeta.step,this.viewBlend);
    }
    const gl=this.gl;this.dpr=Math.min(window.devicePixelRatio||1,1.5);
    const w=Math.max(1,Math.round(this.host.clientWidth*this.dpr)),h=Math.max(1,Math.round(this.host.clientHeight*this.dpr));
    if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}
    if(this.spinning)this.yaw+=.008;
    gl.viewport(0,0,w,h);gl.clearColor(.078,.129,.192,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);this.bind();gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS,0,this.count);
    this.selectedPasses();
    const err=gl.getError();if(err!==gl.NO_ERROR){this.pauseTransition();this.spinning=false;this.status.textContent=`Błąd renderowania GPU: ${err}`;return;}
    const mb=[...this.cache.values()].reduce((s,b)=>s+b.bytes,0)/1e6;
    this.frames.push(now);if(this.frames.length>61)this.frames.shift();const fps=this.frames.length>5?1000*(this.frames.length-1)/(now-this.frames[0]):null;
    this.host.dataset.drawCount=String(this.count);this.host.dataset.step=String(this.meta.step+(this.targetMeta.step-this.meta.step)*this.viewBlend);this.host.dataset.observedStep=String(this.meta.step);this.host.dataset.nextStep=String(this.targetMeta.step);this.host.dataset.blend=String(this.viewBlend);this.host.dataset.interpolated=String(this.viewBlend>0&&this.viewBlend<1);this.host.dataset.animating=String(!!transition);this.host.dataset.movement=String(this.movement);this.host.dataset.bufferBytes=String(mb*1e6);this.host.dataset.glError='0';this.host.dataset.yaw=String(this.yaw);
    this.status.textContent=`WebGL 2 · 900 000 punktów · bufory GPU ${mb.toFixed(1)} MB${(this.spinning||transition)&&fps?` · ${fps.toFixed(0)} kl./s`:''}`;
    this.host.dataset.colorMode=String(this.colorMode);this.host.dataset.selectedImage=String(this.selectedImage);this.host.dataset.selectedPatch=String(this.selectedPatch);this.host.dataset.highlight=String(this.highlight);
    this.host.dataset.spaceView=String(this.spaceView);
    this.axesLabels();
    if(transition&&this.blend>=1){this.current=this.next;this.meta=this.targetMeta;this.blend=0;this.viewBlend=0;this.transition=null;this.host.dataset.animating='false';this.host.dataset.observedStep=String(this.meta.step);this.host.dataset.blend='0';transition.resolve(true);this.pruneCache();}
    if(this.spinning||this.transition)this.requestDraw();
  }
  axesLabels(){let labels=this.host.querySelector('.gpu-axes');if(!labels){labels=document.createElement('div');labels.className='gpu-axes';labels.style.cssText='position:absolute;left:12px;bottom:12px;pointer-events:none;background:#142131cc;padding:6px 9px;border-radius:6px;font-size:12px;color:#aebdca';this.host.append(labels);}labels.textContent=`${this.movement?'Δ ':''}PC1 / PC2 / PC3 · wspólna skala, obrót kamery`;
    // Explicit numeric bounds make the preserved PCA coordinates inspectable.
    labels.title=this.meta.bounds.map((r,i)=>`PC${i+1}: ${r[0]} … ${r[1]}`).join('\n');
    labels.textContent=this.axisNames.filter((_,i)=>this.spaceView===0||this.spaceView===4||i<2).join(' / ')+' · stałe osie';
    let svg=this.host.querySelector('svg');if(!svg){svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.style.cssText='position:absolute;inset:0;width:100%;height:100%;pointer-events:none';svg.setAttribute('aria-hidden','true');this.host.append(svg);}
    svg.style.display=this.showAxes?'block':'none';if(!this.showAxes)return;
    const width=this.host.clientWidth,height=this.host.clientHeight;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const project=p=>{p=p.map((v,i)=>(v-this.center[i])/(this.spaceView===4?Math.max((this.axisBounds[i][1]-this.axisBounds[i][0])*.7,1e-6):this.radius));const c=Math.cos(this.yaw),s=Math.sin(this.yaw),x=c*p[0]+s*p[2],z=-s*p[0]+c*p[2],y=Math.cos(this.pitch)*p[1]-Math.sin(this.pitch)*z;return [width/2+x*this.zoom*height/2,height/2-y*this.zoom*height/2];};
    const bounds=this.axisBounds||this.meta.bounds;labels.title=bounds.map((r,i)=>`${this.axisNames[i]}: ${r[0]} … ${r[1]}`).join('\n');
    const corners=Array.from({length:8},(_,j)=>project(bounds.map((r,i)=>r[(j>>i)&1])));let drawing='';
    for(let j=0;j<8;j++)for(let i=0;i<3;i++)if(!(j&(1<<i))){const a=corners[j],b=corners[j|(1<<i)];drawing+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#567086" stroke-opacity=".4"/>`;}
    const colors=['#ffb967','#a9d776','#8da6ff'],axes=(this.spaceView>=1&&this.spaceView<=3)?2:3;
    for(let i=0;i<axes;i++){const a=corners[0],b=corners[1<<i],c=colors[i];drawing+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${c}" stroke-width="2"/>`;
      for(let k=0;k<=4;k++){const t=k/4,value=bounds[i][0]+t*(bounds[i][1]-bounds[i][0]),point=bounds.map(r=>r[0]);point[i]=value;const p=project(point);drawing+=`<line x1="${p[0]-3}" y1="${p[1]-3}" x2="${p[0]+3}" y2="${p[1]+3}" stroke="${c}"/><text x="${p[0]+7}" y="${p[1]+12}" fill="${c}" stroke="#142131" stroke-width="3" paint-order="stroke" font-size="11">${value.toLocaleString('pl-PL',{maximumFractionDigits:2})}</text>`;}
      drawing+=`<text x="${b[0]}" y="${b[1]-14}" fill="${c}" stroke="#142131" stroke-width="3" paint-order="stroke" text-anchor="middle" font-size="13">${this.axisNames[i]}</text>`;}
    svg.innerHTML=drawing;}
  installControls(){let previous=null,travel=0;
    this.canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;this.canvas.setPointerCapture(e.pointerId);previous=[e.clientX,e.clientY];travel=0;this.frames=[];});
    this.canvas.addEventListener('pointermove',e=>{if(!previous)return;const dx=e.clientX-previous[0],dy=e.clientY-previous[1];travel+=Math.abs(dx)+Math.abs(dy);this.yaw+=dx*.008;this.pitch=Math.max(-1.5,Math.min(1.5,this.pitch+dy*.008));previous=[e.clientX,e.clientY];this.requestDraw();});
    this.canvas.addEventListener('pointerup',e=>{if(previous&&travel<4)this.pick(e);previous=null;});this.canvas.addEventListener('pointercancel',()=>{previous=null;});
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.2,Math.min(30,this.zoom*Math.exp(-e.deltaY*.001)));this.requestDraw();},{passive:false});
  }
  pick(e){if(!this.current)return;const gl=this.gl,w=this.canvas.width,h=this.canvas.height;
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    const fb=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,texture,0);
    try{if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('Bufor wyboru punktu niegotowy');gl.disable(gl.BLEND);gl.disable(gl.DITHER);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);this.bind();gl.uniform1i(gl.getUniformLocation(this.program,'u_pick'),1);gl.drawArrays(gl.POINTS,0,this.count);this.selectedPasses();const rect=this.canvas.getBoundingClientRect(),px=new Uint8Array(4),x=Math.floor((e.clientX-rect.left)*w/rect.width),y=h-1-Math.floor((e.clientY-rect.top)*h/rect.height);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);const id=px[0]+256*px[1]+65536*px[2]-1;
      const norm=id>=0&&id<this.count?this.current.data[id*4+3]*(1-this.viewBlend)+(this.next||this.current).data[id*4+3]*this.viewBlend:0;
      this.detail.textContent=id>=0&&id<this.count?`Obraz ${Math.floor(id/600)} · patch ${id%600} · ${this.viewBlend>0&&this.viewBlend<1?'interpolowany kolor ruchu':'ruch 768D'}: ${norm.toFixed(4)}`:'Kliknij punkt chmury, aby odczytać obraz i patch.';
      if(id>=0&&id<this.count)this.onSelect?.(Math.floor(id/600),id%600);
    }catch(error){this.detail.textContent=error.message;}finally{gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.deleteFramebuffer(fb);gl.deleteTexture(texture);gl.enable(gl.DITHER);this.requestDraw();}
  }
}
