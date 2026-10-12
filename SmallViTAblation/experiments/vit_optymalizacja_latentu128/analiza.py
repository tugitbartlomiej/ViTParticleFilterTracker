"""Sprawdza pobrane artefakty i eksportuje 128 kanałów; to nie maksima semantyczne."""
from pathlib import Path
import numpy as np,json,csv,hashlib,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik
ROOT=HERE.parents[1]/'outputs/vit_optymalizacja_latentu128/eden'
def main():
    for line in (ROOT/'ARTIFACTS.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1);p=ROOT/name.strip();h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==digest,name
    report=json.loads((ROOT/'SUMMARY.json').read_text());r=report['selected_radius'];folder=ROOT/f'r{r}'
    with np.load(folder/'TARGETS_ORACLE.npz') as n:
        channels=n['channels0'];base=n['base'];delta=n['delta_raw'];target=n['target'];standard=n['delta_standardized']
        assert base.shape==target.shape==delta.shape==(400,600,128)
        assert np.isfinite(target).all() and np.isfinite(delta).all()
        np.testing.assert_allclose(base+delta,target,rtol=0,atol=0)
        assert np.max(np.abs(standard))<=float(r)+1e-6
        means=target.mean((0,1));mins=target.min((0,1));maxs=target.max((0,1));rms=np.sqrt((delta**2).mean((0,1)));dm=delta.mean((0,1))
    with np.load(folder/'SHARED.npz') as n:shared=n['delta_raw'].reshape(-1)
    with (ROOT/'KANALY128.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['channel0','channel1','shared_delta_raw','oracle_mean_delta','oracle_rms_delta','oracle_target_mean','oracle_target_min_observed','oracle_target_max_observed'])
        for j,c in enumerate(channels):w.writerow([int(c),int(c+1),float(shared[j]),float(dm[j]),float(rms[j]),float(means[j]),float(mins[j]),float(maxs[j])])
    result=report['results'][r]
    zapisz_wynik(__file__,'wynik_eden_20261004',dict(ziarno=20261004,status='complete',selected_radius=r,metrics=result['methods'],CI=result['paired_film_CI95'],hashes_verified=True,shapes_verified=True,oracle_not_generalization=True,source_jobs=[1822385,1822386,1822387]))
    print('PASS: hashes, shape, target=base+delta, bounds; KANALY128.csv saved')
if __name__=='__main__':main()
