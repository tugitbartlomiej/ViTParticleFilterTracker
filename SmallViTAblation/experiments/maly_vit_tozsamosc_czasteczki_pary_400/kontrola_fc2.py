"""Czy przewaga q pozostaje po norm-matching fc2? Kontrola oddziela wzmocnienie."""
import json,time
import numpy as np,torch
from dane import HERE,DATA,load,image,tensor,init_net,weights,js
import interwencje as I

def main():
    start=time.time();rows,t=load();cache=dict(np.load(HERE/'TOKENY_PARY.npz'));fit=dict(np.load(HERE/'PODPRZESTRZEN.npz'));maps=dict(np.load(HERE/'MAPY_INTERWENCJI.npz'));U=fit['U'];rank=U.shape[1];test=maps['source_indices'];net=init_net()
    old=json.loads((HERE/'DETEKCJA_OBRAZY.json').read_text(encoding='utf-8'));records=[r for r in old if r['method'] in ('nohook','identity')]
    I.METHODS=['nohook','identity','fc2random11','fc2random22','fc2random33'];pred={m:[] for m in I.METHODS};gtall=[]
    outputs={m:[] for m in I.METHODS if m.startswith('fc2')};R={s:np.linalg.qr(np.random.default_rng(s).normal(size=(768,rank)))[0].astype(np.float32) for s in (11,22,33)}
    with torch.no_grad():
        for j,i in enumerate(test):
            bb=np.array(rows[i]['bbox_source_480']);gt=np.array([[bb[0],bb[1],bb[0]+bb[2],bb[1]+bb[3]]],np.float32);gtall.append(gt)
            for m in ('nohook','identity'):
                _,d=I.quality(torch.tensor(maps[m][j:j+1]),gt);pred[m].append(d)
            a,b=cache['offsets'][i:i+2];x=tensor([image(DATA/rows[i]['image_path'])]);ids=cache['token_ids'][a:b]
            for seed in (11,22,33):
                mode=f'fc2random{seed}';diag={'R':R[seed]};hook=net.bloki[-1].mlp.act.register_forward_hook(I.make_hook(ids,cache['states'][a:b,1],U,mode,diag,net))
                o,_=net(x);hook.remove();assert np.isfinite(o.cpu().numpy()).all()
                unchanged=np.ones(600,bool);unchanged[ids]=False
                assert np.max(np.abs(o[0].cpu().numpy().reshape(600,5)[unchanged]-maps['nohook'][j].reshape(600,5)[unchanged]))<2e-5
                quality,d=I.quality(o,gt);diag.pop('R');records.append(dict(source_index=int(i),frame_id=rows[i]['frame_id'],film=rows[i]['video_id'],method=mode,**quality,**diag));pred[mode].append(d);outputs[mode].append(o[0].cpu().numpy())
            if (j+1)%25==0:print('fc2-matched controls',j+1,'/',len(test),flush=True)
    boot=I.bootstrap_detection(rows,test,records,pred,gtall);w=weights(rows,test);summary={}
    for mode in I.METHODS:
        rec=[r for r in records if r['method']==mode];summary[mode]=dict(AP50=I.ap50(pred[mode],gtall),**{k:float(w@np.array([r[k] for r in rec])) for k in ('loss','q','max_confidence')},hidden_norm=float(w@np.array([r.get('hidden_norm',0) for r in rec])),fc2_norm=float(w@np.array([r.get('functional_fc2_norm',0) for r in rec])),**boot[mode])
    identity=np.array([r['q'] for r in records if r['method']=='identity']);random=np.mean([[r['q'] for r in records if r['method']==f'fc2random{s}'] for s in (11,22,33)],axis=0)
    fd=np.array([(identity-random)[[j for j,i in enumerate(test) if rows[i]['video_id']==f]].mean() for f in sorted({rows[i]['video_id'] for i in test})]);rng=np.random.default_rng(913);draw=np.array([fd[rng.integers(0,len(fd),len(fd))].mean() for _ in range(500)])
    report=dict(summary=summary,identity_minus_mean_fc2random_q=float(w@(identity-random)),paired_film_bootstrap95=list(np.quantile(draw,[.025,.975])),
        max_fc2_token_norm_mismatch=max(r.get('max_fc2_token_norm_mismatch',0) for r in records),zero_denominators=sum(r.get('zero_random_denominators',0) for r in records),
        original_U_unchanged=True,forward_count=3*len(test),seconds=time.time()-start,all_finite=True)
    js(HERE/'DETEKCJA_FC2_OBRAZY.json',records);js(HERE/'DETEKCJA_FC2.json',report)
    np.savez_compressed(HERE/'MAPY_INTERWENCJI_FC2.npz',source_indices=test,**{m:np.array(outputs[m]) for m in outputs})
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
