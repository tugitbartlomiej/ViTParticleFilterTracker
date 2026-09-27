"""Kopia trening.py z koopman_gesta_historia (petla_od_poczatku_large): wybor modelu (cfg.model: nano albo
large_deprecated = DINOv2-B 560 px), odczyt sond dopasowany do siatki modelu, sampler ze stala ekspozycja bazy
(cfg.baza_n: kazda klatka bazy raz na epoke, kazda dodana raz, reszta losowe powtorzenia bazy) i ziarno obrotu
(cfg.obrot). Bez tych kluczy zachowanie jak w oryginale.

Oryginalny opis: wznawialne galezie o rownym budzecie; obala je rozny stan startowy lub niesparowany strumien danych."""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from dane import read_json, write_json, sampling_indices,sha256


def rng_state():
    nr=np.random.get_state()
    return dict(python=random.getstate(), numpy=[nr[0],nr[1].tolist(),int(nr[2]),int(nr[3]),float(nr[4])], torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    nr=state['numpy'];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def run(args):
    from pytorch_lightning import Callback, seed_everything
    import rfdetr
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.trainer import build_trainer
    from torch.utils.data import DataLoader, Sampler
    cfg = read_json(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg['seed'], workers=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if cfg.get('determinizm_ostrzezenie'): os.environ['DETERMINIZM_OSTRZEZENIE']='1'
    torch.use_deterministic_algorithms(True, warn_only=bool(cfg.get('determinizm_ostrzezenie')))
    model_name = cfg.get('model', 'nano')
    if model_name == 'nano':
        wrapper = rfdetr.RFDETRNano(pretrain_weights=args.initial_weights, num_classes=1, resolution=384,
                                    amp=False, device=args.device)
    elif model_name == 'large_deprecated':
        wrapper = rfdetr.RFDETRLargeDeprecated(pretrain_weights=args.initial_weights, num_classes=1,
                                               amp=False, device=args.device)
    else:
        raise ValueError(f'Nieznany model {model_name}')
    mc = wrapper.model_config
    import czastki
    kanaly = {'dinov2_windowed_small': 384, 'dinov2_windowed_base': 768}[mc.encoder]
    czastki.configure(mc.resolution, kanaly, (mc.resolution // mc.patch_size,) * 2)
    total_epochs = cfg['pretrain_epochs']+cfg['rounds']
    tc = wrapper.get_train_config(dataset_dir=args.dataset, output_dir=str(out), epochs=total_epochs,
        batch_size=cfg['batch_size'], grad_accum_steps=cfg['grad_accum_steps'], lr=cfg['lr'],
        lr_encoder=cfg['lr_encoder'], weight_decay=cfg['weight_decay'], warmup_epochs=0,
        lr_scheduler='torch.optim.lr_scheduler.ConstantLR', lr_scheduler_kwargs={'factor':1.,'total_iters':1},
        lr_scheduler_interval='step', seed=cfg['seed'], use_ema=False, early_stopping=False,
        compute_val_loss=True, num_workers=0, eval_interval=1, run_test=False, aug_config={},
        augmentation_backend='cpu', multi_scale=False, scale_jitter=False, expanded_scales=False,
        do_random_resize_via_padding=False, tensorboard=False, wandb=False, mlflow=False,
        save_dataset_grids=False, progress_bar=None, checkpoint_interval=100,
        num_sanity_val_steps=0, accelerator='cpu' if args.device=='cpu' else 'gpu', devices=1)
    write_json(out/'effective_config.json',tc.model_dump(mode='json'))
    write_json(out/'model_config.json',mc.model_dump(mode='json'))
    args_dict=vars(wrapper.model.args)
    means, stds = wrapper.means, wrapper.stds
    del wrapper
    mapping=read_json(Path(args.dataset)/'mapping.json')['train']
    n=len(mapping)-(1 if args.added else 0)
    epoch_size=int(cfg['epoch_size'])
    effective=cfg['batch_size']*cfg['grad_accum_steps']
    if epoch_size % (effective*cfg['snapshots_per_epoch']):
        raise ValueError('Need equally spaced snapshots including epoch boundaries')
    steps_epoch=epoch_size//effective
    if args.stop_step%steps_epoch or args.stop_step>steps_epoch*total_epochs:
        raise ValueError('Stop must be a reachable complete epoch boundary')
    annotations=read_json(Path(args.dataset)/'train'/'_annotations.coco.json')
    probe_files={r[k]:sha256(Path(args.probe).parent/r[k]) for r in read_json(args.probe) for k in ['image','mask']}
    contract=dict(recipe=cfg,model_config=mc.model_dump(mode='json'),probe_sha256=sha256(args.probe),
                  probe_files=probe_files,validation_sha256=sha256(Path(args.dataset)/'valid'/'_annotations.coco.json'),
                  mapping=mapping,annotations=annotations)

    class PairedSampler(Sampler):
        def __init__(self, dm): self.dm=dm
        def __len__(self): return epoch_size
        def __iter__(self):
            epoch=self.dm.trainer.current_epoch
            if cfg.get('baza_n'):
                n0=int(cfg['baza_n']); rng=np.random.default_rng(cfg['seed']+1000*int(cfg.get('obrot',0))+epoch*100003)
                if n<n0 or epoch_size<n: raise ValueError('epoch_size musi pomiescic baze i wszystkie dodane')
                idx=list(range(n))+rng.integers(0,n0,size=epoch_size-n).tolist()
                indices=[int(i) for i in rng.permutation(idx)]; slots=[]
            else:
                indices,slots=sampling_indices(n,epoch_size,cfg['seed'],epoch,n if args.added else None,cfg['presentations'])
            write_json(out/f'samples_epoch{epoch:03d}.json',dict(indices=indices,replaced_slots=slots,
                frame_ids=[mapping[i]['frame_id'] for i in indices]))
            return iter(indices)

    class PairedData(RFDETRDataModule):
        def train_dataloader(self):
            # Own sampler fixes the old sample order across the two branches.
            return DataLoader(self._dataset_train, batch_size=cfg['batch_size'], sampler=PairedSampler(self),
                collate_fn=self._collate_fn, num_workers=0, drop_last=True,
                generator=torch.Generator().manual_seed(cfg['seed']),pin_memory=args.device!='cpu')

    class History(Callback):
        def __init__(self): self.loaded_rng=None; self.last_step=-1;self.final_boundary=False
        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint['experiment_rng']=rng_state()
            checkpoint['experiment_recipe']=cfg
            checkpoint['experiment_mapping']=mapping
            checkpoint['experiment_contract']=contract
            if self.final_boundary:
                from granica import canonical_boundary
                canonical_boundary(checkpoint,steps_epoch)
        def on_load_checkpoint(self, trainer, module, checkpoint):
            if 'optimizer_states' not in checkpoint or 'experiment_rng' not in checkpoint:
                raise ValueError('Need full experiment checkpoint to resume')
            if checkpoint.get('experiment_boundary')!='before_next_epoch_v1':raise ValueError('Not a canonical epoch boundary')
            old=checkpoint.get('experiment_contract',{})
            if any(old.get(k)!=contract[k] for k in ['recipe','model_config','probe_sha256','probe_files','validation_sha256']):
                raise ValueError('Resume changes experiment recipe, model or fixed probe')
            expected=mapping[:-1] if args.added else mapping
            if old.get('mapping')!=expected:raise ValueError('Resume changes existing frame order')
            old_ann=old['annotations'];limit=len(expected)
            if (annotations['images'][:limit]!=old_ann['images'] or
                annotations['categories']!=old_ann['categories'] or
                [x for x in annotations['annotations'] if x['image_id']<limit]!=old_ann['annotations']):
                raise ValueError('Resume changes previous annotations or image paths')
            step=int(checkpoint['global_step'])
            if step%steps_epoch or not step<args.stop_step or not checkpoint.get('lr_schedulers') or not checkpoint.get('loops'):
                raise ValueError('Parent must contain optimizer/scheduler/loops at earlier epoch boundary')
            self.loaded_rng=checkpoint['experiment_rng']
        def on_train_start(self,trainer,module):
            if self.loaded_rng: restore_rng(self.loaded_rng)
            self.record(trainer,module)
        def record(self, trainer,module):
            step=int(trainer.global_step)
            if self.last_step==step: return
            state=rng_state(); modes={m:m.training for m in module.model.modules()}
            try:
                from czastki import extract_probe
                module.model.eval()
                # Probe inference never changes weights or consumes training randomness.
                with torch.no_grad(), torch.autocast(device_type=module.device.type, enabled=False):
                    result=extract_probe(module.model, means, stds, args.probe, device=str(module.device), save_particles=bool(cfg.get('czastki_zapis', True)))
                # Worker API is adapted once at integration; dict required for metadata.
                desc=result['descriptors'] if isinstance(result,dict) else result
                if self.last_step < 0:
                    repeated=extract_probe(module.model,means,stds,args.probe,device=str(module.device),save_particles=False)['descriptors']
                    repeat_error=float(np.linalg.norm(repeated-desc)/max(np.linalg.norm(desc),1e-12))
                    write_json(out/'REPEATABILITY.json',dict(relative_error=repeat_error,threshold=1e-5,passed=repeat_error<=1e-5))
                    if repeat_error>1e-5: raise RuntimeError('Probe is not repeatable; refusing Koopman history')
                np.savez_compressed(out/f'LATENT_{step:07d}.npz',features=np.asarray(desc),step=step)
                if isinstance(result,dict) and 'tokens' in result:
                    np.savez_compressed(out/f'CZASTKI_{step:07d}.npz',tokens=result['tokens'])
                write_json(out/'STATUS.json',dict(state='running',global_step=step,epoch=int(trainer.current_epoch)))
                self.last_step=step
            finally:
                for m,mode in modes.items(): m.training=mode
                restore_rng(state)
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            step=int(trainer.global_step)
            if step and step%(steps_epoch//cfg['snapshots_per_epoch'])==0 and step!=self.last_step:
                self.record(trainer,module)
                # Dense files carry weights; full resumable states kept at epoch ends only.
                if cfg.get('wagi_tylko_koniec') and step!=args.stop_step: return   # duzy model: wagi tylko na koncu (kwota Edena)
                torch.save({'model':module.model.state_dict(),'args':args_dict,'model_config':mc.model_dump(mode='json'),
                            'step':step},out/f'model_{step:07d}.pth')
        def on_train_epoch_end(self,trainer,module):
            trainer.save_checkpoint(out/f'FULL_{int(trainer.global_step):07d}.ckpt')
        def on_train_end(self,trainer,module):
            self.record(trainer,module)

    module=RFDETRModelModule(mc,tc)
    module.strict_loading=True
    dm=PairedData(mc,tc)
    trainer=build_trainer(tc,mc,max_steps=args.stop_step,precision='32-true',deterministic=True,
                          enable_progress_bar=False)
    history_callback=History();trainer.callbacks.append(history_callback)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    write_json(out/'NUMERICS.json',dict(matmul_precision=torch.get_float32_matmul_precision(),
        tf32=torch.backends.cuda.matmul.allow_tf32,determinism='strict; math SDPA; grid_sample CPU with autograd'))
    from deterministyka import diagnostic_kernels
    with diagnostic_kernels():
        trainer.fit(module, datamodule=dm, ckpt_path=args.resume)
    if int(trainer.global_step)!=args.stop_step:raise RuntimeError('Actual update budget differs from target')
    history_callback.final_boundary=True
    trainer.save_checkpoint(out/'FINAL.ckpt')
    torch.save({'model':module.model.state_dict(),'args':args_dict,'model_config':mc.model_dump(mode='json'),
                'global_step':int(trainer.global_step)},out/'FINAL.pth')
    # Explicit final validation; unrelated historic best-score callback states are not used.
    with diagnostic_kernels():
        metrics=trainer.validate(module, datamodule=dm, verbose=False)
    write_json(out/'VALIDATION.json',metrics)
    write_json(out/'STATUS.json',dict(state='completed',global_step=int(trainer.global_step),epoch=int(trainer.current_epoch)))
    return out


def main():
    ap=argparse.ArgumentParser()
    for key in ['config','dataset','probe','out','initial-weights']:
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--resume'); ap.add_argument('--added',action='store_true')
    ap.add_argument('--stop-step',type=int,required=True); ap.add_argument('--device',default='cuda')
    run(ap.parse_args())

if __name__=='__main__': main()
