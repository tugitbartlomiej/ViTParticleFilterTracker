"""Sprawdza brak przyszłych danych przy dopasowaniu i przy prognozie.
Zmiana modelu po zatruciu sufiksu albo akceptacja niedojrzałych etykiet obala test.
"""
import json
import numpy as np
from pilot import load, fit, causal_tests, HERE


def main():
    d=load(47)
    spec=dict(kind='legacy_gle',ridge=.001,desc=True,effect=True,ped=True)
    t=np.arange(1000,1301,10)
    a=fit(d,t,500,spec,1800)
    mutated={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
    for k in ('S','P','N','q'):mutated[k][1801:]=1e9
    b=fit(mutated,t,500,spec,1800)
    assert np.array_equal(a['W'],b['W'])
    assert np.array_equal(a['mean'],b['mean'])
    rejected=False
    try:fit(d,np.array([1301]),500,spec,1800)
    except AssertionError:rejected=True
    assert rejected
    out=causal_tests(d,a)
    # Direct source alignment: the ongoing row 1000 must not be in N[1000].
    from pathlib import Path
    root=HERE.parent/'maly_vit_prognoza_tozsamosci_z_historii'/'wyniki'
    z=np.load(root/'seria_s47.npz')
    assert np.array_equal(d['N'][1000],z['nowa'][999])
    assert np.array_equal(d['P'][1000],z['ped'][999])
    out.update(training_suffix_invariance=True,immature_targets_rejected=True,
               exact_source_shift=True,all_passed=True)
    (HERE/'wyniki/TESTY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':main()
