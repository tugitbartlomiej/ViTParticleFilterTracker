from pathlib import Path
import importlib.util,json,time,csv,sys,gc,hashlib
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_kontrolowane_obrazy_20261008/pilot_v1'
FACTORS=['pozycja','kolor','proporcje']

def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def put_tool(texture,alpha,dx=0,dy=0,sx=1.,color=None):
    # Integer displacement and fixed-background compositing; pixel correspondences explicit.
    rgb=texture.copy()
    if color is not None:
        lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        lab[...,1]+=color[0];lab[...,2]+=color[1]
        rgb=np.rint(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB).clip(0,1)*255).astype(np.uint8)
    sy=1/sx;M=np.float32([[sx,0,240*(1-sx)+dx],[0,sy,160*(1-sy)+dy]])
    # Warp premultiplied pixels to avoid dark fringes; no inpainting needed on known background.
    a=cv2.warpAffine(alpha.astype(np.float32),M,(480,320))
    fg=cv2.warpAffine(rgb.astype(np.float32)*alpha[...,None],M,(480,320))
    image=np.rint(fg+127*(1-a[...,None])).clip(0,255).astype(np.uint8)
    return image,a

def make_controls():
    colors={'black':(0,0,0),'white':(255,255,255),'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255),'gray32':(32,32,32),'gray64':(64,64,64),'gray127':(127,127,127),'gray192':(192,192,192)}
    result=[(n,np.full((320,480,3),c,np.uint8)) for n,c in colors.items()]
    grad=np.tile(np.linspace(0,255,480,dtype=np.uint8)[None,:,None],(320,1,3));result.append(('gradient_x',grad));result.append(('gradient_y',np.tile(np.linspace(0,255,320,dtype=np.uint8)[:,None,None],(1,480,3))))
    for name,im in result:cv2.imwrite(str(OUT/'images'/f'control_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    return result

def make_anchor(rgb,mask):
    ys,xs=np.where(mask>.5);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
    scale=min(240/(x1-x0),120/(y1-y0));w=max(2,round((x1-x0)*scale));h=max(2,round((y1-y0)*scale))
    texture=np.zeros((320,480,3),np.uint8);alpha=np.zeros((320,480),np.float32)
    x=240-w//2;y=160-h//2
    texture[y:y+h,x:x+w]=cv2.resize(rgb[y0:y1,x0:x1],(w,h))
    alpha[y:y+h,x:x+w]=cv2.resize(mask[y0:y1,x0:x1].astype(np.float32),(w,h))
    return texture,alpha

def main():
    if (OUT/'RESULT.json').exists():raise RuntimeError('Completed run exists; choose a fresh output path before rerunning.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    started=time.time();prior=module_at('previous_semantics',ROOT/'experiments/vit_semantyka_ruchu_20261008/run.py')
    source=prior.SOURCE;images=np.load(source/'dane/obrazy.npy');records=json.loads((source/'dane/images.json').read_text(encoding='utf8'))
    anchors=[];manifest=[];boxes=[]
    for i,(im,record) in enumerate(zip(images,records)):
        mask,box=prior.mask_box(record);boxes.append(box);texture,alpha=make_anchor(im,mask)
        anchors.append((texture,alpha));manifest.append(dict(anchor=i,domain='real_cutout',role=record['role'],film=record['film'],true_annotation=True))
    controls=make_controls()
    # Isolated, bounded generator inference; no training or generator input optimization.
    import torch
    torch.set_num_threads(4);torch.manual_seed(47)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fastgan={}
    try:
        directory=ROOT/'experiments/koopman_fastgan_petla';sys.path.insert(0,str(directory))
        generation=module_at('controlled_fastgan',directory/'generowanie.py')
        checkpoint=ROOT/'experiments/fastgan_narzedzie/wyniki/eden_20260923_50k/results_50k/checkpoint_050000.pth'
        vendor=ROOT/'experiments/fastgan_narzedzie/vendor/deepfake_gi_fastGAN'
        generator,fastgan=generation.load_fastgan(checkpoint,vendor,'cpu')
        with torch.inference_mode():
            for j in range(4):
                seed=470+j;torch.manual_seed(seed);z=torch.randn(1,256)
                rgba=generation.fixed_generator_output(generator,z,seed)[0].permute(1,2,0).cpu().numpy()
                rgb=np.rint(rgba[...,:3]*255).astype(np.uint8);mask=(rgba[...,3]>.5).astype(np.uint8)
                cv2.imwrite(str(OUT/'images'/f'fastgan_raw_{seed}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));np.save(OUT/'images'/f'fastgan_mask_{seed}.npy',mask)
                if mask.sum()<16:continue
                texture,alpha=make_anchor(rgb,mask);index=len(anchors);anchors.append((texture,alpha));manifest.append(dict(anchor=index,domain='fastgan',role='holdout',seed=seed,true_annotation=False))
        del generator;gc.collect();fastgan['status']='generated';fastgan['accepted_masks']=len(anchors)-12
    except Exception as error:
        fastgan={'status':'failed','error':str(error)}
    (OUT/'GENERATOR.json').write_text(json.dumps(dict(fastgan=fastgan,sd={'status':'not_loaded','reason':'Concurrent workloads occupy GPU and system memory; controlled measurements do not require SD.'}),indent=2),encoding='utf8')
    operations=[dict(factor='baseline',dx=0,dy=0,sx=1.,color=None)]
    for dx,dy in [(32,0),(-32,0),(0,32),(0,-32)]:operations.append(dict(factor='pozycja',dx=dx,dy=dy,sx=1.,color=None))
    for color in [(12,0),(-12,0),(0,12),(0,-12)]:operations.append(dict(factor='kolor',dx=0,dy=0,sx=1.,color=color))
    for sx in [1.25,.8]:operations.append(dict(factor='proporcje',dx=0,dy=0,sx=sx,color=None))
    variants=[];selections=[];qa={}
    for i,(texture,alpha) in enumerate(anchors):
        ims=[];masks=[]
        for j,op in enumerate(operations):
            im,m=put_tool(texture,alpha,**{k:v for k,v in op.items() if k!='factor'});ims.append(im);masks.append(m)
            cv2.imwrite(str(OUT/'images'/f'anchor{i:02d}_variant{j:02d}_{op["factor"]}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        for j in range(5,9):assert np.array_equal(ims[j][masks[0]==0],ims[0][masks[0]==0]);assert np.array_equal(masks[j],masks[0])
        assert np.array_equal(ims[1][:,32:],ims[0][:,:-32])
        variants.append(ims)
        union=np.maximum.reduce(masks);region=cv2.resize(union,(30,20),interpolation=cv2.INTER_AREA).reshape(-1)>.01
        selections.append(region)
    qa.update(color_mask_equal=True,color_background_equal=True,integer_translation_equal=True)
    (OUT/'IMAGES.json').write_text(json.dumps(dict(anchors=manifest,operations=operations,controls=[n for n,_ in controls]),indent=2,ensure_ascii=False),encoding='utf8')
    variants=np.array(variants,np.uint8);selections=np.array(selections)
    np.save(OUT/'CONTROLLED_IMAGES.npy',variants);np.save(OUT/'TOKEN_REGIONS.npy',selections)
    del anchors;gc.collect()
    # Keep per-forward memory bounded; CPU avoids interference with occupied GPU.
    device='cpu';model_module=prior.load_model_module();model=model_module.MalyViT().eval();capture={};replace={'z':None}
    def hook(m,i,o):
        capture['z']=o.detach().clone()
        if replace['z'] is not None:return replace['z']
    handle=model.bloki[-1].mlp.act.register_forward_hook(hook)
    @torch.inference_mode()
    def forward(im):
        inp=torch.tensor(im.transpose(2,0,1).copy(),dtype=torch.float32)[None]/255
        p,_=model(inp);return p,capture['z'][0]
    slices=[range(1,5),range(5,9),range(9,11)];fit_indices=[i for i,m in enumerate(manifest) if m['role']=='fit'];test_indices=[i for i,m in enumerate(manifest) if m['role']=='holdout']
    probes=[];motion=[];controlrows=[];detections=[];basis_archive=[];past=None;past_q=None;past_iou=None
    steps=[0,250,500,750,1000]
    # Common random baseline, rank and edit norm matched; same across time/factors.
    torch.manual_seed(48);random_q=torch.linalg.qr(torch.randn(768,8)).Q
    real_region=[]
    for record in records:
        mask,_=prior.mask_box(record);real_region.append(cv2.resize(cv2.dilate(mask,np.ones((33,33),np.uint8)),(30,20),interpolation=cv2.INTER_AREA).reshape(-1)>0)
    for step in steps:
        w=torch.load(source/f'wagi/k{step:04d}.pt',map_location='cpu',weights_only=False);model.load_state_dict(w['net']);del w
        responses=[[] for _ in FACTORS]
        for i in range(len(manifest)):
            _,z0=forward(variants[i,0]);region=torch.tensor(selections[i])
            for f,indices in enumerate(slices):
                diffs=[]
                for j in indices:
                    _,z=forward(variants[i,j]);diffs.append((z-z0)[region])
                responses[f].append(torch.cat(diffs))
        # Compare direct SVD against contrastive generalized eigendirections in 768 channel coordinates.
        cov=[]
        for res in responses:
            stack=torch.cat([res[i] for i in fit_indices]);c=stack.T@stack;cov.append(c/c.trace().clamp(min=1e-12))
        bases={};import scipy.linalg
        for f,name in enumerate(FACTORS):
            vals,vecs=torch.linalg.eigh(cov[f]);bases[('svd',name)]=vecs[:,-8:]
            denominator=sum(cov[g] for g in range(3) if g!=f)+torch.eye(768)*(.01/768)
            eigenvalues,eigenvectors=scipy.linalg.eigh(cov[f].numpy().astype(np.float64),denominator.numpy().astype(np.float64),subset_by_index=[760,767])
            bases[('contrast',name)]=torch.linalg.qr(torch.tensor(eigenvectors,dtype=torch.float32)).Q
        basis_archive.append(np.stack([q.numpy() for q in bases.values()]))
        for (method,f),q in bases.items():
            for g,res in zip(FACTORS,responses):
                for i in test_indices:
                    rr=res[i];fraction=float((rr@q).square().sum()/rr.square().sum().clamp(min=1e-12))
                    probes.append(dict(step=step,method=method,basis=f,perturbation=g,anchor=i,domain=manifest[i]['domain'],capture=fraction))
        del responses;gc.collect()
        latent=[];qualities=[]
        for i,im in enumerate(images):
            pred,z=forward(im);b,s=model_module.dekoduj(pred);gt=torch.tensor(boxes[i])[None];quality=float(prior.iou(b[:,0],gt)[0]);latent.append(z);qualities.append(quality)
            detections.append(dict(step=step,image=i,role=records[i]['role'],iou=quality,score=float(s[0,0])))
        current=torch.stack(latent);del latent
        if past is not None:
            delta=current-past
            for (method,f),q in past_q.items():
                edit=(delta@q)@q.T;rnd=(delta@random_q)@random_q.T
                scales=edit.flatten(1).norm(dim=1)/rnd.flatten(1).norm(dim=1).clamp(min=1e-12);rnd=rnd*scales[:,None,None]
                for i in range(8,12):
                    d=delta[i,torch.tensor(real_region[i])];fraction=float((d@q).square().sum()/d.square().sum().clamp(min=1e-12))
                    vals=[]
                    for e in [edit,rnd]:
                        replace['z']=(current[i]-e[i])[None];p,_=forward(images[i]);replace['z']=None;b,_=model_module.dekoduj(p);vals.append(float(prior.iou(b[:,0],torch.tensor(boxes[i])[None])[0]))
                    motion.append(dict(step=step,previous_step=step-250,image=i,method=method,factor=f,motion_fraction=fraction,delta_iou=qualities[i]-past_iou[i],rollback_effect=qualities[i]-vals[0],random_norm_effect=qualities[i]-vals[1]))
        past=current;past_q=bases;past_iou=qualities
        for name,im in controls:
            p,_=forward(im);b,s=model_module.dekoduj(p);controlrows.append(dict(step=step,control=name,max_score=float(s[0,0]),detections_above_point5=int((s>.5).sum())))
        csvwrite('PROBES.csv',probes);csvwrite('DETECTIONS.csv',detections);csvwrite('CONTROLS.csv',controlrows)
        if motion:csvwrite('MOTION.csv',motion)
        print(f'checkpoint {step}, elapsed={time.time()-started:.1f}s, mean natural holdout IoU={np.mean(qualities[8:]):.4f}',flush=True)
    # Identity substitution check at actual final checkpoint.
    original,z=forward(images[8]);replace['z']=z[None];identical,_=forward(images[8]);replace['z']=None
    qa['zero_intervention_equal']=bool(torch.equal(original,identical));assert qa['zero_intervention_equal']
    handle.remove();np.savez_compressed(OUT/'BASES.npz',steps=steps,bases=np.stack(basis_archive),labels=np.array([f'{m}:{f}' for m,f in bases]))
    (OUT/'QA.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
    result=dict(status='complete',steps=steps,anchors=len(manifest),controlled_images=int(variants.shape[0]*variants.shape[1]),controls=len(controls),device=device,elapsed_seconds=time.time()-started,fastgan=fastgan,rank=8,source=str(source),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf8');(OUT/'run_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
