"""Dodatkowa analiza masek bbox, dodana na prośbę użytkownika po analizie głównej."""
from ruch_a_jakosc import *

def main():
    frames=[];comparisons=[];cors=[];masks=[];validation=[]
    p=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz')
    ts=np.arange(1000,65601,100);steps=p['steps'];idx=np.searchsorted(steps,ts);prev=np.searchsorted(steps,ts-100)
    pts=np.c_[(np.arange(600)%30+.5)*16,(np.arange(600)//30+.5)*16]
    main_history=pd.read_csv(OUT/'historia.csv')
    for im,name in enumerate(NAMES):
        h=np.load(ROOT/'app/cache_uczenia/lasery'/name/'dane.npz');gt=h['gt'];inside=((pts>=gt[:2])&(pts<=gt[2:])).all(1)
        dist=np.linalg.norm(np.maximum(np.maximum(gt[:2]-pts,pts-gt[2:]),0),axis=1);far=dist>32
        assert np.array_equal(inside,h['narz']) and not np.any(inside&far)
        masks.append(dict(image=im+1,bbox_tokens=int(inside.sum()),far_background_tokens=int(far.sum()),
            primary_center_inside=int(inside[p['ids'][im,:8]].sum()),primary_background_far=int(far[p['ids'][im,16:24]].sum()),
            bbox_ids=np.flatnonzero(inside).tolist(),far_ids=np.flatnonzero(far).tolist()))
        vals=dict(image=np.full(len(ts),im+1),step=ts)
        # POS używa projekcji residual-pre 192D do 3D po odjęciu środka wszystkich tokenów w każdym stanie.
        pos=h['pos'].astype(float);vpos=np.linalg.norm(pos[idx]-pos[prev],axis=-1)/100
        nodes=h['wezly'].astype(float);vn=np.linalg.norm(nodes[idx]-nodes[prev],axis=-1)/100
        for label,mask in [('bbox',inside),('background',far)]:
            vals['vib_'+label]=h['vib'][idx][:,mask].mean(1).astype(float)
            vals['node_speed_'+label]=vn[:,mask].mean(1)
            vals['projected_speed_'+label]=vpos[:,mask].mean(1)
        for m in ['vib','node_speed','projected_speed']:
            assert np.all(vals[m+'_background']>0)
            vals[m+'_ratio']=vals[m+'_bbox']/vals[m+'_background']
        frame=pd.DataFrame(vals);frames.append(frame)
        original=main_history[main_history.image==im+1];high=original.high.to_numpy()
        for radius in [1000,2000]:
            pairs=match(ts,high,radius)
            for m in list(vals)[2:]:
                a=vals[m];rec=dict(image=im+1,radius=radius,metric=m,n_pairs=len(pairs))
                if len(pairs):
                    lo=a[pairs[:,0]];hi=a[pairs[:,1]]
                    rec.update(median_paired_pct=float(np.median(100*(hi-lo)/lo)),fraction_high_larger=float(np.mean(hi>lo)))
                comparisons.append(rec)
        for m in list(vals)[2:]:
            for target in ['iou','confidence']:
                for k in [4,6,10]:
                    cors.append(dict(image=im+1,metric=m,target=target,knots=k,
                        residual=corr(residual(ts,vals[m],k),residual(ts,original[target].to_numpy(),k))))
        # Niezależna kontrola IoU i pomiaru ruchu w tabeli głównej.
        b=h['ramka'][idx].astype(float);inter=np.maximum(np.minimum(b[:,2:],gt[2:])-np.maximum(b[:,:2],gt[:2]),0).prod(1)
        area=np.maximum(b[:,2:]-b[:,:2],0).prod(1);gtarea=np.prod(gt[2:]-gt[:2]);iou=inter/(area+gtarea-inter)
        err=float(np.max(abs(iou-h['iou'][idx])))
        assert err<1e-6
        probe=np.arange(0,647,57);x=p['x_pre'].astype(float)
        independent=np.array([np.mean([np.linalg.norm(x[idx[j],im,t]-x[prev[j],im,t])/100 for t in range(8)]) for j in probe])
        speed_error=float(np.max(abs(independent-original.speed_center.to_numpy()[probe])))
        assert speed_error<1e-10
        validation.append(dict(image=im+1,iou_max_error=err,speed_max_error=speed_error))
    pd.concat(frames).to_csv(OUT/'historia_etykiety.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'porownania_etykiety.csv',index=False)
    pd.DataFrame(cors).to_csv(OUT/'korelacje_etykiety.csv',index=False)
    (OUT/'ETYKIETY_I_KONTROLE.json').write_text(json.dumps(dict(masks=masks,checks=validation),indent=2),encoding='utf-8')
    print(pd.DataFrame(comparisons).query('radius==1000').to_string(index=False))
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for im,(ax,name) in enumerate(zip(axes,NAMES)):
        h=np.load(ROOT/'app/cache_uczenia/lasery'/name/'dane.npz');ax.imshow(h['rgb'])
        mi=masks[im];a=pts[mi['bbox_ids']];b=pts[mi['far_ids']]
        ax.scatter(b[:,0],b[:,1],s=3,color='cyan',alpha=.5,label='Dalekie tło >32 px od bbox')
        ax.scatter(a[:,0],a[:,1],s=9,color='lime',label='Środek tokenu w bbox')
        c=pts[p['ids'][im,:8]];ax.scatter(c[:,0],c[:,1],s=35,facecolors='none',edgecolors='red',label='8 tokenów centrum')
        ax.set_title(f'Obraz {im+1}: bbox={mi["bbox_tokens"]}, tło={mi["far_background_tokens"]}');ax.axis('off')
    axes[0].legend(fontsize=6,loc='lower left');fig.suptitle('Etykieta ramki nie jest maską segmentacji narzędzia');fig.tight_layout()
    fig.savefig(OUT/'maski_obszarow.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
