"""Pełne natywne współrzędne: affine Koopman, finite/EMA GLE. Bez PCA."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve

FIRST_TARGET=5

def configs():
 out=[]
 for coupled in (False,True):
  for ridge in (.1,1.,10.):out.append(dict(family='koopman',coupled=coupled,ridge=ridge,L=0))
  for L in (1,2,4):
   for ridge in (1.,10.):out.append(dict(family='finite_gle',coupled=coupled,ridge=ridge,L=L))
  for bank in ((2,10,50),(10,50,200)):
   for ridge in (1.,10.):out.append(dict(family='exp_gle',coupled=coupled,ridge=ridge,bank=list(bank),L=0))
 for ridge in (.1,1.,10.):
  for rho in (.995,1.):out.append(dict(family='stable_koopman',coupled=False,ridge=ridge,L=0,rho=rho))
 for i,c in enumerate(out):c['id']='linear_%02d'%i
 return out

def ema(x,bank):
 result=[]
 for half in bank:
  a=2.**(-1./half);e=np.zeros_like(x)
  for t in range(1,len(x)):e[t]=a*e[t-1]+(1-a)*(x[t]-x[t-1])
  result.append(e)
 return result

def blocks(x,g,w,c,t):
 fg=np.einsum('tnd,n->td',x,w);vals={'own':x,'fg':fg,'bg':g}
 names=['own','fg','bg'] if c['coupled'] else ['own']
 T=len(t);N=len(w);D=x.shape[-1]
 f=[];layout=[]
 for name in names:
  v=vals[name][t];f.append(v if name=='own' else np.broadcast_to(v[:,None,:],(T,N,D)));layout.append((name,'level',0))
 if c['family'].startswith('exp_gle'):
  memories={name:ema(vals[name],c['bank']) for name in names}
  for j in range(len(c['bank'])):
   for name in names:
    v=memories[name][j][t];f.append(v if name=='own' else np.broadcast_to(v[:,None,:],(T,N,D)));layout.append((name,'ema',j))
 else:
  for lag in range(c.get('L',0)):
   for name in names:
    v=vals[name][t-lag]-vals[name][t-lag-1];f.append(v if name=='own' else np.broadcast_to(v[:,None,:],(T,N,D)));layout.append((name,'lag',lag))
 return f,layout

def regression(X,Y,w,ridge):
 w=w/w.sum();mx=w@X;my=w@Y
 sx=np.maximum(np.sqrt(w@(X-mx)**2),1e-10);sy=np.maximum(np.sqrt(w@(Y-my)**2),1e-10)
 Z=(X-mx)/sx;R=(Y-my)/sy;G=Z.T@(Z*w[:,None]);cross=Z.T@(R*w[:,None])
 gamma=ridge*np.trace(G)/len(G);Bstd=cho_solve(cho_factor(G+gamma*np.eye(len(G)),check_finite=False),cross,check_finite=False)
 B=Bstd*sy[None,:]/sx[:,None];b=my-mx@B
 assert np.max(np.abs(X@B+b-((X-mx)/sx@Bstd*sy+my)))<1e-8
 return dict(B=B,b=b,center_x=mx,center_y=my,scale_x=sx,scale_y=sy,gamma=gamma,rows=len(X),features=X.shape[-1],gram=G)

def fit(qs,bgs,weights,end,config):
 c=dict(config);D=qs[0].shape[-1];assert D==128
 xs=[];ys=[];ws=[];xb=[];yb=[];wb=[]
 for q,bg,w in zip(qs,bgs,weights):
  q=np.asarray(q[:end+1],float);bg=np.asarray(bg[:end+1],float)
  x=q-q[0];g=bg-bg[0];t=np.arange(FIRST_TARGET-1,end)
  ff,layout=blocks(x,g,w,c,t);T=len(t);N=len(w)
  xs.append(np.concatenate(ff,axis=-1).reshape(T*N,-1));ys.append((x[t+1]-x[t]).reshape(T*N,D));ws.append(np.tile(w/T/len(qs),T))
  if c['coupled']:
   fg=np.einsum('tnd,n->td',x,w);vv={'bg':g,'fg':fg};features=[];bl=[]
   for name in ('bg','fg'):features.append(vv[name][t]);bl.append((name,'level',0))
   if c['family'].startswith('exp_gle'):
    mm={name:ema(vv[name],c['bank']) for name in vv}
    for j in range(len(c['bank'])):
     for name in ('bg','fg'):features.append(mm[name][j][t]);bl.append((name,'ema',j))
   else:
    for j in range(c.get('L',0)):
     for name in ('bg','fg'):features.append(vv[name][t-j]-vv[name][t-j-1]);bl.append((name,'lag',j))
   xb.append(np.concatenate(features,axis=-1));yb.append(g[t+1]-g[t]);wb.append(np.ones(T)/T/len(qs))
 op=regression(np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),c['ridge'])
 model=dict(config=c,D=D,end=end,first_target=FIRST_TARGET,fg=op,bg=None,layout_fg=layout,layout_bg=None)
 if c['coupled']:model['bg']=regression(np.concatenate(xb),np.concatenate(yb),np.concatenate(wb),c['ridge']);model['layout_bg']=bl
 if c['family'].startswith('stable_koopman'):
  F=np.eye(D)+op['B'].T;ev,V=np.linalg.eig(F);clipped=ev*np.minimum(1.,c['rho']/np.maximum(np.abs(ev),1e-30))
  new=(V@np.diag(clipped)@np.linalg.inv(V));imag=float(np.max(np.abs(new.imag)));assert imag<1e-7
  op['B']=(new.real-np.eye(D)).T
  # Preserve fitted mean increment at the training feature mean.
  op['b']=op['center_y']-op['center_x']@op['B']
  model['stabilization']=dict(original_radius=float(np.max(abs(ev))),clipped_radius=float(np.max(abs(clipped))),imag_error=imag,actual_radius=float(np.max(abs(np.linalg.eigvals(new.real)))),eigenvector_condition=float(np.linalg.cond(V)))
 return model

def predict(q,bg,w,origin,max_h,model):
 c=model['config'];D=model['D'];assert q.shape[-1]==D==128
 # Slice BEFORE all differencing/EMA operations: no future observations.
 q=np.asarray(q[:origin+1],float);bg=np.asarray(bg[:origin+1],float)
 xhistory=q-q[0];ghistory=bg-bg[0];x=xhistory[-1].copy();g=ghistory[-1].copy()
 L=c.get('L',0);hx=[xhistory[-1-j]-xhistory[-2-j] for j in range(L)];hg=[ghistory[-1-j]-ghistory[-2-j] for j in range(L)]
 if c['family'].startswith('exp_gle'):ex=[a[-1].copy() for a in ema(xhistory,c['bank'])];eg=[a[-1].copy() for a in ema(ghistory,c['bank'])]
 out=np.empty((max_h+1,len(w),D));out[0]=q[-1]
 for h in range(1,max_h+1):
  fg=np.einsum('nd,n->d',x,w);values={('own','level',0):x,('fg','level',0):fg,('bg','level',0):g}
  for j in range(L):values[('own','lag',j)]=hx[j];values[('fg','lag',j)]=w@hx[j];values[('bg','lag',j)]=hg[j]
  if c['family'].startswith('exp_gle'):
   for j in range(len(ex)):values[('own','ema',j)]=ex[j];values[('fg','ema',j)]=w@ex[j];values[('bg','ema',j)]=eg[j]
  def apply(op,layout):
   result=op['b'].copy()
   for j,key in enumerate(layout):result=result+values[tuple(key)]@op['B'][j*D:(j+1)*D]
   return result
  dx=apply(model['fg'],model['layout_fg']);dg=apply(model['bg'],model['layout_bg']) if c['coupled'] else np.zeros_like(g)
  x=x+dx;g=g+dg
  if L:hx=[dx]+hx[:-1];hg=[dg]+hg[:-1]
  if c['family'].startswith('exp_gle'):
   for j,half in enumerate(c['bank']):
    a=2.**(-1./half);ex[j]=a*ex[j]+(1-a)*dx;eg[j]=a*eg[j]+(1-a)*dg
  out[h]=x+q[0]
 return out
