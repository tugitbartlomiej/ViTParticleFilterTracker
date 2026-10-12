"""Independent direct-coordinate checks of the Gram-based formulas."""
import numpy as np
from analyze import dist, vacf, save, timeavg
rng=np.random.default_rng(47)
x=rng.normal(size=(8,401,10,12)).cumsum(axis=1)
flat=x.reshape(8,401,-1)
g=flat@flat.transpose(0,2,1)/10
s=np.arange(401)
maxerr=0.
for lag in [1,5,50,100]:
    direct=((x[:,lag:]-x[:,:-lag])**2).sum(-1).mean((1,2))
    via=dist(g,np.arange(401-lag),np.arange(lag,401)).mean(1)
    maxerr=max(maxerr,float(np.max(np.abs(direct-via)/direct)))
v=vacf(g,s,0,400,1,10)
d=np.diff(x,axis=1);e=(d*d).sum(-1).mean((1,2))
for lag in range(1,11):
    direct=(d[:,lag:]*d[:,:-lag]).sum(-1).mean((1,2))/e
    maxerr=max(maxerr,float(np.max(np.abs(direct-v['per_image'][:,lag]))))
assert maxerr<1e-11,maxerr
# Directly verify that local image translation removal equals subtracting centroid Gram.
c=x.mean(2);gcent=c@c.transpose(0,2,1)
r=x-c[:,:,None,:];rg=r.reshape(8,401,-1)@r.reshape(8,401,-1).transpose(0,2,1)/10
centerr=float(np.max(np.abs(rg-(g-gcent))))
assert centerr<1e-10
# Analytical stationary OU counterexample: low exponent and negative adjacent increments,
# with NO age dependence at all. This is not a simulation of structural glass.
rho=np.exp(-1/30);ou=rho**np.abs(s[:,None]-s[None,:]);og=np.repeat(ou[None],8,axis=0)
ov=vacf(og,s,0,400,1,10)['mean'][1]
assert abs(ov+(1-rho)/2)<1e-12
t,y,_=timeavg(og,s,0,400,1,np.arange(1,201))
alpha=float(np.polyfit(np.log(t[t>=100]),np.log(y.mean(0)[t>=100]),1)[0])
save('VALIDATION.json',dict(direct_formula_max_relative_error=maxerr,centroid_gram_max_error=centerr,
    stationary_OU=dict(tau=30,adjacent_vacf=float(ov),msd_exponent_100_200=alpha,aging=False),passed=True))
print('Validated direct MSD, VACF, centroid removal and analytic OU.',flush=True)
