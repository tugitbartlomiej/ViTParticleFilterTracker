"""Fresh dense history; inference/generation remain a separate local stage."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from dane import read_json, write_json, sha256, load_pool, make_dataset
from czastki import extract_probe  # Validate recorder dependencies before allocating training work.

def main():
    p=argparse.ArgumentParser()
    for key in ('config','pool','data-root','selection','weights','probe-source','out','archive'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(a.config)
    write_json(out/'config.json',cfg)
    write_json(out/'STATUS.json',dict(state='preparing'))
    try:
        rows=load_pool(a.pool)
        ids=read_json(a.selection)['frame_ids']
        assert len(ids)==cfg['initial_budget']
        dataset=make_dataset(rows,a.data_root,ids,out/'datasets/pretrain')
        shutil.copytree(a.probe_source,out/'probe')
        probe=out/'probe/manifest.json'
        byid={r['frame_id']:r for r in rows}
        for r in read_json(probe):
            assert byid[r['frame_id']]['split']=='train'
            assert byid[r['frame_id']]['group_id']==r['group_id']
            for k in ('image','mask'): assert (probe.parent/r[k]).is_file()
        source=Path(__file__).parent
        write_json(out/'INPUTS.json',dict(arguments=vars(a),
            code_sha256={f.name:sha256(f) for f in source.glob('*.py')},
            archive_sha256=sha256(a.archive),weights_sha256=sha256(a.weights),
            selection_sha256=sha256(a.selection),pool_sha256=sha256(a.pool)))
        steps=cfg['pretrain_epochs']*cfg['epoch_size']//(cfg['batch_size']*cfg['grad_accum_steps'])
        write_json(out/'STATUS.json',dict(state='training',target_step=steps))
        with (out/'pretrain.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-B',str(source/'trening.py'),
                '--config',str(out/'config.json'),'--dataset',str(dataset),'--probe',str(probe),
                '--initial-weights',a.weights,'--out',str(out/'pretrain'),
                '--stop-step',str(steps),'--device','cuda'],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read_json(out/'pretrain/STATUS.json')['global_step']==steps
        expected=cfg['pretrain_epochs']*cfg['snapshots_per_epoch']+1
        assert len(list((out/'pretrain').glob('LATENT_*.npz')))==expected
        image_root=Path(a.data_root).resolve()
        hashes={}
        for split in ('train','valid'):
            for im in read_json(dataset/split/'_annotations.coco.json')['images']:
                path=Path(im['file_name']).resolve()
                hashes[path.relative_to(image_root).as_posix()]=sha256(path)
        write_json(out/'pretrain/IMAGE_BYTES.json',dict(data_root=str(image_root),files=hashes,
            checkpoint_sha256=sha256(out/'pretrain/FINAL.ckpt')))
        write_json(out/'REPORT.json',dict(state='completed',global_step=steps,latent_snapshots=expected,
            validation=read_json(out/'pretrain/VALIDATION.json'),
            next_stage='local Koopman analysis; FastGAN only after forecast validation'))
        write_json(out/'STATUS.json',dict(state='completed',global_step=steps))
    except BaseException as e:
        write_json(out/'STATUS.json',dict(state='failed',error=repr(e)))
        raise

if __name__=='__main__': main()
