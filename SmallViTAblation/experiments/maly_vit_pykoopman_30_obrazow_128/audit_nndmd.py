"""Read-only diagnostics after frozen NNDMD evaluation; never selects or trains."""
from pathlib import Path
import json
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'4')
import numpy as np
import torch
import pykoopman as pk
from nndmd_fit import install_loss_and_rollout_adapter
from pykoopman_fit import write_json,sha256,now


def main():
    root=Path(__file__).parent
    result=json.loads((root/'WYNIKI_NNDMD.json').read_text(encoding='utf-8'))
    frozen=json.loads((root/'ZAMROZENIE_NNDMD.json').read_text(encoding='utf-8'))
    assert sha256(root/frozen['model_file'])==frozen['model_sha256']
    cp=torch.load(root/frozen['model_file'],map_location='cpu'); config=cp['config']
    a=cp['A_learned'].astype(np.float64); g=cp['generator'].astype(np.float64)
    assert a.shape==(768,768) and np.isfinite(a).all()
    eig=np.linalg.eigvals(a); singular=np.linalg.svd(a,compute_uv=False)
    decoder=cp['state_dict']['_decoder.layers.0.weight'].numpy().astype(np.float64)
    decoder_s=np.linalg.svd(decoder,compute_uv=False)
    decoder_rank=int(np.sum(decoder_s>decoder_s[0]*max(decoder.shape)*np.finfo(np.float32).eps))
    a50=np.linalg.matrix_power(a,50)
    stability={'spectral_radius_A':float(np.abs(eig).max()),'operator_norm2_A':float(singular[0]),
               'operator_norm2_A_power50':float(np.linalg.norm(a50,2)),
               'min_singular_value_A':float(singular[-1]),'condition_number2_A':float(singular[0]/singular[-1]),
               'max_real_generator_eigenvalue':float(np.linalg.eigvals(g).real.max()),
               'finite_A_power50':bool(np.isfinite(a50).all()),'coefficients_A':int(a.size),
               'note':'Stability is in learned coordinates; spectral radius alone does not bound transient amplification or decoded prediction error.'}
    stability['decoder_linear_C']={'rank_at_float32_relative_tolerance':decoder_rank,
             'tolerance':float(decoder_s[0]*max(decoder.shape)*np.finfo(np.float32).eps),
             'min_singular_value':float(decoder_s[-1]),'operator_norm2':float(decoder_s[0]),
             'condition_number2':float(decoder_s[0]/decoder_s[-1]),
             'note':'If C is invertible, decoded rollout is (C A C^-1)^h R(x0), R=s C E(x0/s). A nonlinear encoder and a square linear decoder do not by themselves establish physical nonlinear conjugacy.'}
    reg=pk.regression.NNDMD(dt=1,look_forward=config['look_forward'],config_encoder=config['encoder'],config_decoder=config['decoder'])
    install_loss_and_rollout_adapter(reg._regressor,np.array(config['channels0']),config['identity_beta'])
    reg._regressor.load_state_dict(cp['state_dict']); module=reg._regressor.cuda().eval()
    device=module.device; scale=config['scale']; matrix=torch.tensor(cp['A_learned'],device=device)
    latent=np.load(root/'LATENT_768.npy',mmap_mode='r'); masks=np.load(root/'MASKI.npz')
    closure={}
    for label,images,first,last in [('fit30_prefix500',frozen['fit_indices'],0,50),('holdout_future_diagnostic',[8,9,10,11],50,100)]:
        error=0.; persistence=0.; target_energy=0.
        for image in images:
            ids=np.flatnonzero(masks['tool'][image]&(masks['coverage'][image]>0))
            w=masks['coverage'][image,ids].astype(np.float64);w/=w.sum()
            x=np.asarray(latent[first:last+1,image,ids],dtype=np.float32)
            flat=x.reshape(-1,768)
            with torch.no_grad():
                encoded=torch.cat([module._encoder(torch.tensor(flat[k:k+512],device=device)/scale) for k in range(0,len(flat),512)],0).reshape(x.shape)
                e=(encoded[1:]-encoded[:-1]@matrix.T).cpu().numpy().astype(np.float64)
                d=(encoded[1:]-encoded[:-1]).cpu().numpy().astype(np.float64)
                energy=encoded[1:].cpu().numpy().astype(np.float64)
            error+=float(np.sum(e*e*w[None,:,None])/(len(images)*(last-first)*768))
            persistence+=float(np.sum(d*d*w[None,:,None])/(len(images)*(last-first)*768))
            target_energy+=float(np.sum(energy*energy*w[None,:,None])/(len(images)*(last-first)*768))
        closure[label]={'learned_one_step_rmse':float(np.sqrt(error)),'learned_persistence_rmse':float(np.sqrt(persistence)),
                        'learned_rmse_ratio_vs_persistence':float(np.sqrt(error/persistence)),
                        'relative_rmse_to_learned_state_rms':float(np.sqrt(error/target_energy)),
                        'images':images,'source_updates':[first*10,last*10],
                        'post_test_diagnostic_only':True,'units':'learned normalized coordinates; not raw native channels'}
    full=np.load(root/'NNDMD_PROGNOZA_HOLDOUT_FULL768.npy',mmap_mode='r')
    j=np.load(root/'NNDMD_PROGNOZA_HOLDOUT_J128.npy',mmap_mode='r')
    channels=np.array(config['channels0']); same=all(np.array_equal(full[h][...,channels],j[h]) for h in range(50))
    assert same
    baselines=json.loads((root/'WALIDACJA_LAMBDA_3.json').read_text(encoding='utf-8'))['aggregate']
    baseline_rows=[{'horizon_updates':h,'static_EDMD_rmse':float(np.sqrt(baselines[h//10-1]['mse_state'])),
                    'persistence_rmse':float(np.sqrt(baselines[h//10-1]['mse_persistence']))} for h in [10,20,50,100,200]]
    report={'created':now(),'model_sha256':frozen['model_sha256'],'no_tuning':True,'stability':stability,'closure':closure,
            'decoded_J128_bitwise_equal_to_full_forecast_slice':same,'internal_validation_baselines':{
                'origin_updates':300,'images':[28,29,30,31,32,33],'rows':baseline_rows,
                'static_EDMD_mean5_rmse':float(np.mean([r['static_EDMD_rmse'] for r in baseline_rows])),
                'persistence_mean5_rmse':float(np.mean([r['persistence_rmse'] for r in baseline_rows]))},
            'reconstruction_origin':result['reconstruction_origin'],
            'interpretation':'Reconstruction RMSE is a measured encoder/decoder error, not a rigorous lower bound on forecast RMSE. Closure residual uses true future encoder states only after test; no forecast consumes them.'}
    write_json(root/'NNDMD_AUDYT_OPERATORA.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
