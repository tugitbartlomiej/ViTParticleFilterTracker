"""Metryki skumulowanego przemieszczenia: wszystkie J128, bez odsiewu."""
import numpy as np

def measure(actual,predicted,initial,w):
 actual=np.asarray(actual,float);predicted=np.asarray(predicted,float)
 da=actual-initial;dp=predicted-initial;w=np.asarray(w,float);w=w/w.sum()
 mse=float(w@np.mean((predicted-actual)**2,axis=-1));base=float(w@np.mean(da**2,axis=-1))
 finite=bool(np.isfinite(predicted).all() and np.isfinite(mse))
 if not finite:
  return dict(status='failure',reason='nonfinite prediction or squared error',finite=False,**{k:None for k in ['mse','persistence_mse','skill','sign_accuracy','balanced_sign_accuracy','cosine','norm_absolute_error','norm_signed_bias','norm_ratio','actual_norm_mean','predicted_norm_mean']})
 sa=np.sign(da);sp=np.sign(dp);sign=float(w@np.mean(sa==sp,axis=-1))
 balanced=[]
 for lab in (-1,1):
  mask=sa==lab;den=float(np.sum(w[:,None]*mask))
  balanced.append(float(np.sum(w[:,None]*mask*(sp==sa))/den) if den else None)
 norm_a=np.linalg.norm(da,axis=-1);norm_p=np.linalg.norm(dp,axis=-1);valid=(norm_a>0)&(norm_p>0)&np.isfinite(norm_p)
 cosine=float(np.sum(w[valid]*np.einsum('nd,nd->n',da[valid],dp[valid])/(norm_a[valid]*norm_p[valid]))/w[valid].sum()) if valid.any() else None
 return dict(mse=mse,persistence_mse=base,skill=1-mse/base if base else None,sign_accuracy=sign,balanced_sign_accuracy=float(np.mean(balanced)) if all(v is not None for v in balanced) else None,sign_class_recalls=balanced,actual_zero_coordinate_weight=float(np.sum(w[:,None]*(sa==0))/da.shape[-1]),cosine=cosine,cosine_valid_tokens=int(valid.sum()),cosine_total_tokens=len(w),norm_absolute_error=float(w@np.abs(norm_p-norm_a)),norm_signed_bias=float(w@(norm_p-norm_a)),status='ok',actual_norm_mean=float(w@norm_a),predicted_norm_mean=float(w@norm_p),norm_ratio=float(w@norm_p/(w@norm_a)) if w@norm_a>0 else None,finite=finite)

def aggregate(records):
 keys=['mse','persistence_mse','skill','sign_accuracy','balanced_sign_accuracy','cosine','norm_absolute_error','norm_signed_bias','norm_ratio','actual_norm_mean','predicted_norm_mean']
 if not all(r['finite'] for r in records):return dict(status='failure',finite=False,images=len(records),**{k:None for k in keys})
 out={k:float(np.mean([r[k] for r in records if r[k] is not None])) if any(r[k] is not None for r in records) else None for k in keys}
 out['finite']=all(r['finite'] for r in records);out['images']=len(records)
 return out
