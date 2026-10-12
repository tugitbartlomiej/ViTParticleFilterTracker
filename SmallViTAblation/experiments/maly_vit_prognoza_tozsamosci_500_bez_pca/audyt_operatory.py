"""Diagnostyka homogenicznych operatorów; bez fitu/zmiany modeli."""
import badanie as b
import dane as data
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.linalg import svdvals

def companion(A,K,c):
 d=len(A);n=len(K);M=np.zeros((d*(n+1),d*(n+1)));M[:d,:d]=np.eye(d)+A
 for j,k in enumerate(K):M[:d,(j+1)*d:(j+2)*d]=k
 if c['family'].startswith('exp_gle'):
  for j,half in enumerate(c['bank']):
   a=2.**(-1./half);r=slice((j+1)*d,(j+2)*d);M[r,:d]=(1-a)*A
   for k,kk in enumerate(K):M[r,(k+1)*d:(k+2)*d]=(1-a)*kk
   M[r,r]+=a*np.eye(d)
 else:
  if n:
   M[d:2*d,:d]=A
   for j,k in enumerate(K):M[d:2*d,(j+1)*d:(j+2)*d]=k
   for j in range(1,n):M[(j+1)*d:(j+2)*d,j*d:(j+1)*d]=np.eye(d)
 return M

def operators(m):
 D=128;c=m['config'];op=m['fg'];blocks={tuple(k):op['B'][j*D:(j+1)*D].T for j,k in enumerate(m['layout_fg'])}
 kind='ema' if c['family'].startswith('exp_gle') else 'lag';n=len(c.get('bank',[])) if kind=='ema' else c.get('L',0)
 own=companion(blocks[('own','level',0)],[blocks[('own',kind,j)] for j in range(n)],c)
 result={'relative_or_local':own}
 if c['coupled']:
  ob=m['bg'];bb={tuple(k):ob['B'][j*D:(j+1)*D].T for j,k in enumerate(m['layout_bg'])}
  def joint(kind,j):return np.block([[blocks[('own',kind,j)]+blocks[('fg',kind,j)],blocks[('bg',kind,j)]],[bb[('fg',kind,j)],bb[('bg',kind,j)]]])
  result['mean_foreground_background']=companion(joint('level',0),[joint(kind,j) for j in range(n)],c)
 return result

def main():
 threadpool_limits(4);models=b.load_models();records=[]
 for ident,m in models.items():
  if not ident.startswith('linear_'):
   records.append(dict(id=ident,status='not_matrix_diagnosed',reason='Observer/trend formula documented separately; no common fitted native companion.'));continue
  for name,F in operators(m).items():
   ev,V=np.linalg.eig(F);rho=float(np.max(np.abs(ev)));condition=float(np.linalg.cond(V));dt=m.get('driver_dt',1)
   requested=sorted(set([1]+[(h+dt-1)//dt for h in (10,50,100,500)]));R=np.random.default_rng(26102).normal(size=(len(F),8));R/=np.linalg.norm(R,axis=0);gains={}
   for t in range(1,max(requested)+1):
    R=F@R
    if t in requested:gains[str(t*dt)]=float(np.max(np.linalg.norm(R,axis=0)))
   rec=dict(id=ident,block=name,dimensions=len(F),model_dt=dt,spectral_radius=rho,eigenvector_condition=condition,one_step_spectral_norm=float(svdvals(F,check_finite=False)[0]),sampled_transient_gain_lower_bounds=gains,homogeneous_only=True,affine_constant_excluded=True,spectrum_full=True)
   records.append(rec);print('operator',ident,name,rho,flush=True)
 data.save_json(data.HERE/'AUDYT_OPERATOROW.json',dict(completed=b.now(),model_sha=data.sha(data.HERE/'MACIERZE_PRZED_TESTEM.npz'),records=records,warning='Sampled gains are lower bounds, not exact operator norms; rho<=1 does not imply contraction or absence of transient growth. Native coordinates and complete memory retained. No model changes/refits.'))
if __name__=='__main__':main()
