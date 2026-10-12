"""Testy błędów naukowych: granice par, leakage, sterowanie i kontrakt 768D."""
import importlib
from pathlib import Path
import numpy as np
import pytest

def core():
    assert (Path(__file__).parent/'temporal_core.py').exists(), 'Brak implementacji kontraktu temporalnego'
    return importlib.import_module('temporal_core')

def test_paired_public_dmdc_recovers_known_system_without_image_bridges():
    # Niezależne pary, nie trajektoria; sklejenie obrazów zniszczy identyfikację.
    c=core(); rng=np.random.default_rng(71)
    x=rng.normal(size=(300,2));u=rng.normal(size=(300,1))
    A=np.array([[.8,.2],[-.1,.9]]); B=np.array([[.5],[-.3]])
    y=x@A.T+u@B.T
    fitted=c.fit_pydmd_pairs(x,y,u)
    assert fitted['backend']=='pydmd.DMDc.fit'
    np.testing.assert_allclose(fitted['A'],A,atol=1e-9)
    np.testing.assert_allclose(fitted['B'],B,atol=1e-9)
    known=c.fit_pydmd_pairs(x,y,u,known_B=B)
    np.testing.assert_allclose(known['A'],A,atol=1e-9)
    assert fitted['pairs']==300 and fitted['physical_lag']==1

def test_scaler_does_not_refit_on_extreme_test_data():
    c=core(); s=c.Scaler.fit(np.array([[0.,10.],[2.,14.]]))
    expected_mean=s.mean.copy();expected_scale=s.scale.copy()
    assert np.allclose(s.transform(np.array([[1.,12.]])),0)
    s.transform(np.full((10,2),1e10))
    np.testing.assert_array_equal(s.mean,expected_mean)
    np.testing.assert_array_equal(s.scale,expected_scale)

def test_split_keeps_every_frame_of_a_film_together():
    c=core(); movies=['a','a','b','c','d','d','e','f','g','h','i','j']
    labels=c.film_split(movies)
    assert labels[0]==labels[1] and labels[4]==labels[5]
    assert set(labels)=={'train','validation','test'}

def test_rollout_windows_do_not_cross_phase_or_trajectory_boundary():
    c=core(); phase=np.array([0,0,0,1,1,1,1])
    # 7 transitions, first three fit. Window len=2 starts 0,1,3,4,5.
    np.testing.assert_array_equal(c.window_starts(phase,2),[0,1,3,4,5])
    np.testing.assert_array_equal(c.window_starts(phase,20),[])

def test_adam_input_is_pre_step_and_clamp_is_applied():
    c=core(); z=np.array([[-.169,1.]])
    g=np.array([[1.,-1.]]);m=np.zeros_like(g);v=np.zeros_like(g)
    dz=c.adam_input(z,g,m,v,np.array([1]),np.array([.05]))
    np.testing.assert_allclose(dz,[[-.00097121,.05]],atol=1e-8)

def test_contract_rejects_pca3_and_missing_moments():
    c=core()
    with pytest.raises(ValueError,match='768'):
        c.validate_arrays({'control_z':np.zeros((501,8,3))})
    with pytest.raises(ValueError,match='control_gradient'):
        c.validate_arrays({'control_z':np.zeros((501,8,768))})

def test_deeptime_real_paired_dmd_predicts_linear_dynamics():
    c=core();rng=np.random.default_rng(123);x=rng.normal(size=(200,2))
    A=np.array([[.8,-.2],[.1,.7]])
    fitted=c.fit_deeptime_pairs(x,x@A.T)
    np.testing.assert_allclose(fitted['A'],A,atol=1e-9)

def test_metrics_cannot_hide_diverging_rollout():
    c=core(); a=np.array([[0.,1.],[1.,2.]])
    m=c.error_metrics(a+100,a,np.zeros_like(a))
    assert m['rmse']==100 and m['n']==2


def test_basis_never_reads_validation_or_test(monkeypatch):
    import analiza_temporalna as pipeline
    rng=np.random.default_rng(888);z=rng.normal(size=(4,2,768))
    seen=[]
    def loader(row):
        seen.append(row['image_index'])
        assert row['analysis_split']=='train', 'Leakage: heldout data read during basis fit'
        return {'control_z':z}
    monkeypatch.setattr(pipeline,'load',loader)
    rows=[{'image_index':1,'analysis_split':'train'}, {'image_index':2,'analysis_split':'test'}, {'image_index':3,'analysis_split':'validation'}]
    C,info=pipeline.training_basis(rows,2)
    assert seen==[1] and info['fit_images']==[1] and C.shape==(2,768)


def test_control_scaling_preserves_tiny_adam_second_moment():
    c=core();x=np.array([[1e-15],[3e-15]])
    np.testing.assert_allclose(c.Scaler.fit(x).transform(x),[[-1],[1]])
