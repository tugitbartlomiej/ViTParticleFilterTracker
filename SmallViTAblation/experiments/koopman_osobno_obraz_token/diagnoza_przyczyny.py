import numpy as np, json
from pathlib import Path
S=Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/koopman_gesta_historia/wyniki/macierze_wiele_obrazow')
steps=list(range(5,370,5)); tok=np.arange(0,576,16)
Z=np.stack([np.load(S/f'TOKENS_{s:07d}.npy',mmap_mode='r')[:,tok,:] for s in steps],2).astype(np.float64)  # 72 x 36 x 73 x 384
Z=Z.reshape(-1,73,384); U=len(Z); print('jednostek',U)
V=np.diff(Z,axis=1)                         # 72 przyrostow
n=np.linalg.norm(V,axis=2)
print('mediana |przyrostu| wg kroku (co 5 punktow):',np.round(np.median(n,0)[::5],4).tolist())
c=(V[:,1:]*V[:,:-1]).sum(2)/(n[:,1:]*n[:,:-1])
print('cos(v_t,v_t-1) mediana: pocz',round(float(np.median(c[:,:20])),3),'test',round(float(np.median(c[:,48:])),3))
# okno dopasowania 48 par (stany 0..48), test przejscia 48..71
X=Z[:,:48]; Y=Z[:,1:49]; mx=X.mean(1,keepdims=True); md=(Y-X).mean(1,keepdims=True); Xc=X-mx; Dc=(Y-X)-md
# 1) ekstrapolacja: udzial testowego stanu poza przestrzenia rozpieta przez Xc
poza=[]; wzdluz=[]
for u in range(U):
    Q,_=np.linalg.qr(Xc[u].T)                 # 384 x 48
    d=Z[u,48:72]-mx[u,0]
    pr=d@Q@Q.T
    poza.append(np.median(np.linalg.norm(d-pr,axis=1)/np.linalg.norm(d,axis=1)))
    # glowny kierunek trajektorii i polozenie testu wzgledem zakresu uczenia
    _,_,vt=np.linalg.svd(Xc[u],full_matrices=False); p=vt[0]
    s_tr=Xc[u]@p; s_te=d@p; rng=s_tr.max()-s_tr.min()
    wzdluz.append(np.median(np.maximum(0,np.maximum(s_te-s_tr.max(),s_tr.min()-s_te))/rng))
print('udzial testowego stanu POZA przestrzenia uczenia (mediana):',round(float(np.median(poza)),3))
print('jak daleko test wychodzi poza zakres uczenia wzdluz glownej osi (w dlugosciach zakresu):',round(float(np.median(wzdluz)),3))
# 2) dryf sredni z okna uczenia wobec prawdziwych przyrostow w tescie
vt_=V[:,48:72]
print('|srednia dryfu okna uczenia| / |przyrost w tescie| mediana:',round(float(np.median(np.linalg.norm(md[:,0],axis=1)[:,None]/np.linalg.norm(vt_,axis=2))),3))
print('cos(sredni dryf uczenia, przyrost testowy) mediana:',round(float(np.median((vt_*md).sum(2)/(np.linalg.norm(vt_,axis=2)*np.linalg.norm(md,axis=2)))),3))
# 3) modele jednokrokowe na tescie
def koop(rr):
    G=Xc@Xc.transpose(0,2,1); lam=rr*np.trace(G,axis1=1,axis2=2)/48
    C=np.linalg.solve(G+lam[:,None,None]*np.eye(48),Dc)
    q=(Z[:,48:72]-mx)@Xc.transpose(0,2,1)
    return Z[:,48:72]+md+q@C
tru=Z[:,49:73]; z=Z[:,48:72]
def rmse(p): return float(np.sqrt(((p-tru)**2).mean()))
wyn={'koopman_ridge0.1':rmse(koop(.1)),'koopman_ridge0.001':rmse(koop(.001)),'koopman_ridge10':rmse(koop(10.)),
     'sam_dryf_sredni(z+md)':rmse(z+md),'brak_zmian':rmse(z),'ostatni_przyrost':rmse(z+V[:,47:71])}
a=[]
for u in range(U):
    v=V[u,:47]; a.append((v[:-1]*v[1:]).sum()/(v[:-1]**2).sum())
a=np.array(a)[:,None,None]; wyn['alpha*ostatni']=rmse(z+a*V[:,47:71])
# predkosc jako stan: v_{t+1} = A v_t (osobno, grzbiet, postac dualna) + dryf
Va=V[:,:46]; Vb=V[:,1:47]; mva=Va.mean(1,keepdims=True); mvb=Vb.mean(1,keepdims=True)
Vc=Va-mva; G=Vc@Vc.transpose(0,2,1); lam=.1*np.trace(G,axis1=1,axis2=2)/46
Cv=np.linalg.solve(G+lam[:,None,None]*np.eye(46),(Vb-mvb))
vpred=mvb+((V[:,47:71]-mva)@Vc.transpose(0,2,1))@Cv
wyn['koopman_na_predkosci']=rmse(z+vpred)
print(json.dumps({k:round(v,6) for k,v in wyn.items()},indent=1))
pk=koop(.1)-z
print('|prognoza Koopmana| / |prawdziwy przyrost| mediana:',round(float(np.median(np.linalg.norm(pk,axis=2)/np.linalg.norm(tru-z,axis=2))),3),
      ' cos(prognoza Koopmana, sredni dryf):',round(float(np.median((pk*md).sum(2)/(np.linalg.norm(pk,axis=2)*np.linalg.norm(md,axis=2)))),3))
