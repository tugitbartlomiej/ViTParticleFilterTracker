"""Verify saved factors and held-out errors using independent CPU reconstruction."""
import os
os.environ['OMP_NUM_THREADS']='4'
import json,numpy as np,torch
from dane import HERE,SOURCE,OUT,write,sha
from operator_lokalny import Operators

torch.set_num_threads(4)
def run():
    if json.loads((OUT/'REPORT.json').read_text())['state']!='completed':raise RuntimeError('Incomplete run')
    units=[(0,0),(0,299),(66,299),(71,575)]
    states=np.stack([np.stack([np.load(SOURCE/f'TOKENS_{s:07d}.npy',mmap_mode='r')[i,t] for i,t in units]) for s in range(5,370,5)],axis=1).astype('float64')
    rel=np.load(OUT/'RELATIVE_RIDGE.npy',mmap_mode='r');rr=torch.tensor([rel[i,t] for i,t in units])
    x=torch.tensor(states)
    m=Operators().fit(x[:,:72],x[:,1:73],rr)
    saved_c=np.load(OUT/'C_FINAL.npy',mmap_mode='r');saved_err=np.load(OUT/'FIT_MSE.npy',mmap_mode='r')
    records=[]
    for k,(i,t) in enumerate(units):
        np.testing.assert_allclose(saved_c[i,t],m.coeff[k].numpy(),atol=1e-9,rtol=1e-7)
        mse=float(((m.predict(x[:,:72])[k]-x[k,1:73])**2).mean())
        np.testing.assert_allclose(mse,saved_err[i,t,1],atol=1e-12,rtol=1e-8)
        records.append(dict(image=i,token=t,full_fit_mse=mse,coefficient_max_abs_diff=float(abs(saved_c[i,t]-m.coeff[k].numpy()).max())))
    test=Operators().fit(x[:,:48],x[:,1:49],rr);pred=test.predict(x[:,48:72]);err=((pred-x[:,49:73])**2).mean(-1).numpy()
    saved_test=np.load(OUT/'MSE_ONE_STEP.npy',mmap_mode='r')
    for k,(i,t) in enumerate(units):np.testing.assert_allclose(err[k],saved_test[i,t,:,0],atol=1e-12,rtol=1e-8)
    a=np.load(OUT/'EXAMPLE_EVALUATION_MATRIX.npz');p=np.load(OUT/'EXAMPLE_FORECASTS.npz')
    np.testing.assert_allclose(p['z']@a['K']+a['b'],p['prediction'],atol=1e-10,rtol=1e-8)
    f=np.load(OUT/'MATRIX_final_066_299.npz');K,b=m.explicit()
    np.testing.assert_allclose(f['K'],K[2].numpy(),atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(f['b'],b[2].numpy(),atol=1e-9,rtol=1e-8)
    write(OUT/'VERIFICATION.json',dict(all_passed=True,independent_cpu_units=records,
        heldout_errors_reproduced=True,explicit_evaluation_and_final_matrices_verified=True,
        script_sha256=sha(__file__),historical_code_snapshot='source_snapshot/',
        post_run_guard_changes='Resume/export/status guards changed after launch; algebra unchanged'))
    print('Verified CPU/GPU factors, fit and held-out errors, both explicit matrices.')

if __name__=='__main__':
    with torch.no_grad():run()
