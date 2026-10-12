"""Explicitly distinguish interval length from separation of one-step increments."""
from analyze import load,vacf,save
s,gs=load('dense')
out={name:{str(dt):vacf(g,s,300,1094,dt,12) for dt in [1,2,5,10,20,50]} for name,g in gs.items()}
save('DENSE_MULTISCALE_VACF.json',out)
for name,r in out.items():print(name,[(dt,round(v['mean'][1],4)) for dt,v in r.items()],flush=True)
