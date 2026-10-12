"""Shared low-rank coupling: global foreground is ALWAYS derived from particles."""
import numpy as np

def layout(L,coupled,bg=False):
 names=['bg','fg'] if bg else (['own','fg','bg'] if coupled else ['own'])
 return [(name,-1) for name in names]+[(name,lag) for lag in range(L) for name in names]

def fit_operator(features,Y,w,ridge):
 D=Y.shape[-1];X=np.concatenate(features,axis=-1);shape=X.shape;X=X.reshape(-1,X.shape[-1]);Y=Y.reshape(-1,D)
 wm=w/w.sum();mx=wm@X;my=wm@Y;sx=np.maximum(np.sqrt(wm@(X-mx)**2),1e-10);sy=np.maximum(np.sqrt(wm@(Y-my)**2),1e-10)
 XX=(X-mx)/sx;YY=(Y-my)/sy;G=XX.T@(XX*wm[:,None]);R=XX.T@(YY*wm[:,None]);gamma=ridge*np.trace(G)/len(G);standard=np.linalg.solve(G+gamma*np.eye(len(G)),R)
 B=standard*sy[None,:]/sx[:,None];b=my-mx@B
 error=float(np.max(np.abs(X@B+b-((X-mx)/sx@standard)*sy-my)))
 assert error<1e-9
 ev=np.linalg.eigvalsh(G);return dict(B=B,b=b,scale_x=sx,scale_y=sy,center_x=mx,center_y=my,standardized_slope=standard,
  ridge_gamma=gamma,gram_rank=int((ev>max(ev[-1],1e-30)*1e-10).sum()),features=len(ev),condition=float((ev[-1]+gamma)/(max(ev[0],0)+gamma)),native_max_error=error)

def fit(s,bg,w,L,ridge,dt,end,coupled):
 D=s.shape[-1];anchor=s[500].copy();anchorbg=bg[500].copy();x=s-anchor;g=bg-anchorbg;fg=np.einsum('tnd,n->td',x,w)
 targets=np.arange(500+dt,end+1,dt);t=targets-dt;T=len(t);N=len(w);names=layout(L,coupled);fgfeatures=[]
 values={'own':x,'fg':fg,'bg':g}
 for name,lag in names:
  v=values[name];f=v[t] if lag<0 else v[t-lag*dt]-v[t-(lag+1)*dt]
  fgfeatures.append(f if name=='own' else np.broadcast_to(f[:,None],(T,N,D)))
 model=dict(fg=fit_operator(fgfeatures,x[targets]-x[t],np.tile(w/T,T),ridge),bg=None,L=L,ridge=ridge,dt=dt,D=D,coupled=coupled,anchor=anchor,anchor_bg=anchorbg,layout_fg=names,layout_bg=layout(L,True,True),fit_targets=targets)
 if coupled:
  features=[]
  for name,lag in model['layout_bg']:
   v=values[name];features.append(v[t] if lag<0 else v[t-lag*dt]-v[t-(lag+1)*dt])
  model['bg']=fit_operator(features,g[targets]-g[t],np.ones(T)/T,ridge)
 delta=x[targets]-x[t];mean=np.einsum('tnd,n->d',delta,w)/T
 sd=np.sqrt(np.einsum('tnd,n->d',(delta-mean)**2,w)/T);maj=np.sign(np.einsum('tnd,n->d',np.sign(delta),w));maj[maj==0]=1
 model['epsilon_q']=.1*sd[:8];model['majority_q']=maj[:8];return model

def delta_operator(op,blocks,values):
 D=op['b'].size;out=op['b'].copy()
 for j,block in enumerate(blocks):out=out+values[tuple(block)]@op['B'][j*D:(j+1)*D]
 return out

def rollout(s,bg,w,origins,H,model):
 dt=model['dt'];L=model['L'];origins=np.array(origins,int);assert H%dt==0
 # Transfer changes only anchors, all fitted operators stay fixed.
 x=s[origins]-s[500];g=bg[origins]-bg[500];fg=np.einsum('ond,n->od',x,w)
 histx=[s[origins-j*dt]-s[origins-(j+1)*dt] for j in range(L)];histg=[bg[origins-j*dt]-bg[origins-(j+1)*dt] for j in range(L)]
 out={0:s[origins,:,:8].copy()};maxnorm=0.
 for h in range(dt,H+1,dt):
  values={('own',-1):x,('fg',-1):fg[:,None],('bg',-1):g[:,None]}
  for j in range(L):values[('own',j)]=histx[j];values[('fg',j)]=np.einsum('ond,n->od',histx[j],w)[:,None];values[('bg',j)]=histg[j][:,None]
  dx=delta_operator(model['fg'],model['layout_fg'],values)
  if model['coupled']:
   vb={('fg',-1):fg,('bg',-1):g}
   for j in range(L):vb[('fg',j)]=values[('fg',j)][:,0];vb[('bg',j)]=histg[j]
   dg=delta_operator(model['bg'],model['layout_bg'],vb)
  else:dg=np.zeros_like(g)
  # Synchronous snapshot: both increments computed before either state update.
  x=x+dx;g=g+dg;fg=np.einsum('ond,n->od',x,w)
  if L:histx=[dx]+histx[:-1];histg=[dg]+histg[:-1]
  if not np.isfinite(x).all() or not np.isfinite(g).all():raise FloatingPointError('nonfinite autonomous rollout')
  maxnorm=max(maxnorm,float(np.max(np.abs(x))),float(np.max(np.abs(g))));out[h]=(x+s[500])[:,:,:8].copy()
 return out,maxnorm

def spectrum(model):
 # Decomposition: deviations obey own operator; foreground mean and background form 2D coupled system.
 D=model['D'];L=model['L'];blocks={tuple(k):model['fg']['B'][j*D:(j+1)*D].T for j,k in enumerate(model['layout_fg'])}
 def companion(A,K):
  d=len(A);M=np.zeros((d*(L+1),d*(L+1)));M[:d,:d]=np.eye(d)+A
  if L:
   M[d:2*d,:d]=A
   for j in range(L):M[:d,(j+1)*d:(j+2)*d]=K[j];M[d:2*d,(j+1)*d:(j+2)*d]=K[j]
   for j in range(1,L):M[(j+1)*d:(j+2)*d,j*d:(j+1)*d]=np.eye(d)
  return float(np.max(np.abs(np.linalg.eigvals(M))))
 own=companion(blocks[('own',-1)],[blocks[('own',j)] for j in range(L)])
 if not model['coupled']:return dict(deviation_radius=own,mean_background_radius=None)
 bblocks={tuple(k):model['bg']['B'][j*D:(j+1)*D].T for j,k in enumerate(model['layout_bg'])}
 A=np.block([[blocks[('own',-1)]+blocks[('fg',-1)],blocks[('bg',-1)]],[bblocks[('fg',-1)],bblocks[('bg',-1)]]])
 K=[np.block([[blocks[('own',j)]+blocks[('fg',j)],blocks[('bg',j)]],[bblocks[('fg',j)],bblocks[('bg',j)]]]) for j in range(L)]
 return dict(deviation_radius=own,mean_background_radius=companion(A,K))
