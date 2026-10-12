"""Freeze modified online runner for the user-requested random10k scope."""
from pathlib import Path
import shutil,hashlib,json,ast
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments/vit_translator_mlp_calydataset'
source=(SOURCE/'run.py').read_text(encoding='utf-8');text=source
text=text.replace("initial=ROOT/'outputs/vit_translator_mlp_pelny_latent/TRANSLATOR_MLP.pt'","initial=ROOT/'outputs/vit_translator_mlp_calydataset/MODEL_latest.pt'")
text=text.replace("optimizer='fresh AdamW; pilot checkpoint contains no optimizer state'","optimizer='continue AdamW from full-dataset step500 checkpoint',dataset_limit=10000,selection_seed=20261005,no_saved_vit_latents=True")
start=text.index("    groups=json.loads(SPLIT.read_text())")
end=text.index("    train=[r for r in rows",start)
replacement='''    manifest=OUT/'MANIFEST.jsonl'
    full_manifest=ROOT/'outputs/vit_translator_mlp_calydataset/MANIFEST.jsonl'
    if manifest.exists():rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    else:
        rng=np.random.default_rng(config['selection_seed']);reservoir=[];held=[];seen=0
        with full_manifest.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['split']=='train':
                    seen+=1
                    if len(reservoir)<10000:reservoir.append(r)
                    else:
                        j=int(rng.integers(seen))
                        if j<10000:reservoir[j]=r
                else:held.append(r)
        assert seen==283577 and len(reservoir)==10000
        rows=reservoir+sample([r for r in held if r['split']=='val'],512)+sample([r for r in held if r['split']=='test'],512)
        tmp=manifest.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r)+'\\n')
        replace(tmp,manifest)
        js('SELECTION.json',dict(method='uniform reservoir sample without replacement, Algorithm R',seed=config['selection_seed'],selected=10000,source_eligible_train=seen,source_manifest_sha256=sha(full_manifest),initial_checkpoint_step=500,prior_training_exposures=4000))
'''
text=text[:start]+replacement+text[end:]
text=text.replace("assert len(train)>35000", "assert len(train)==10000")
text=text.replace("validate_every=2000", "validate_every=500")
text=text.replace("model,old=load(initial,'cuda');hidden=old['hidden'];del old", "model,old=load(initial,'cuda');hidden=old['hidden'];initial_optimizer=old['optimizer'];del old")
needle="    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])"
assert needle in text
text=text.replace(needle,needle+"\n    optimizer.load_state_dict(initial_optimizer);del initial_optimizer")
# No per-image ViT latents or DINO arrays are written in this run.
text=text.replace("    np.savez(OUT/'TEST_PREDICTIONS.npz',predicted=p,actual=y,ids=np.array(ids));js('RESULT.json',", "    js('RESULT.json',")
start=text.index("    # The artificial target remains tied")
end=text.index("if __name__=='__main__'",start)
text=text[:start]+"    reader.close();progress('complete',test=m,valid_images_seen=valid_seen)\n"+text[end:]
assert 'np.save(' not in text and 'np.savez(' not in text
ast.parse(text)
(HERE/'run.py').write_text(text,encoding='utf-8');shutil.copy2(SOURCE/'dane.py',HERE/'dane.py')
(HERE/'SOURCE_LINEAGE.json').write_text(json.dumps(dict(parent=str(SOURCE/'run.py'),parent_sha256=hashlib.sha256(source.encode()).hexdigest(),no_saved_latent_arrays=True),indent=2),encoding='utf-8')
