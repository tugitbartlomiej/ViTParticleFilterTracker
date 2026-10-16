"""Scientific gates for scalar-control whole-population DMDc; no scheduler calls."""
import importlib
import numpy as np
import pytest

def module():return importlib.import_module('population_dmdc_supplement')

def fixture():
    a={'steps':np.arange(0,21,5),'telemetry_learning_rate':np.arange(21,dtype=float)/100,
       'interval_gradient_l2_start':np.full((4,2),2.),'interval_adam_m1_l2_start':np.full((4,2),3.),'interval_adam_m2_l2_start':np.full((4,2),4.),
       'interval_gradient_l2_mean':np.full((4,2),999.),'interval_gradient_l2_last':np.full((4,2),888.),'interval_increment_norm':np.full((4,2),777.)}
    return a

def test_controls_only_use_pre_interval_amplitudes_lr_next_update_clock():
    m=module();a=fixture();u=m.causal_controls(a)
    np.testing.assert_allclose(u[:,0,:3],np.tile([2,3,4],(4,1)))
    np.testing.assert_allclose(u[:,0,3],[.01,.06,.11,.16])
    np.testing.assert_allclose(u[:,0,4],[0,.01,.02,.03])
    for k in ['interval_gradient_l2_mean','interval_gradient_l2_last','interval_increment_norm']:a[k][:]=np.nan
    np.testing.assert_array_equal(m.causal_controls(a),u)

def test_frozen_start_input_cannot_see_future_controls():
    m=module();A=np.eye(2);B=np.ones((2,1));initial=np.zeros((1,2));u=np.array([[[1.]],[[2.]],[[3.]],[[4.]]])
    frozen=m.rollout(A,B,initial,u,4,True);teacher=m.rollout(A,B,initial,u,4,False)
    np.testing.assert_allclose(frozen,[[4,4]]);np.testing.assert_allclose(teacher,[[10,10]])
    u[1:]=1000
    np.testing.assert_array_equal(m.rollout(A,B,initial,u,4,True),frozen)

def test_mixed_phase_is_retained_as_distinct_stratum():
    m=module();fraction=np.array([0.,0.,.4,1.,1.,1.])
    np.testing.assert_array_equal(m.phase_windows(fraction,1),[0,0,-1,1,1,1])
    np.testing.assert_array_equal(m.phase_windows(fraction,4),[-1,-1,-1])

def test_full768_omitted_energy_not_called_zero_error():
    m=module();a={'analysis_increment_residual_norm':np.array([[2.],[3.],[4.],[5.]]),'analysis_state_residual_norm':np.array([[0.],[2.],[5.],[9.],[14.]])}
    lo,hi=m.residual_bounds(a,1);np.testing.assert_array_equal(lo,a['analysis_increment_residual_norm']);np.testing.assert_array_equal(lo,hi)
    lo,hi=m.residual_bounds(a,4);np.testing.assert_array_equal(lo,[[14.]]);np.testing.assert_array_equal(lo,hi)

def test_training_image_sampler_excludes_heldout_films():
    m=module();rows=[{'image_index':i,'split':s} for i,s in enumerate(['test','train','validation','train','train'])]
    selected=m.select_fit_rows(rows,2)
    assert [r['image_index'] for r in selected]==[1,4]

def test_real_pydmd_controlled_pairs_recover_known_coarse_dynamics():
    m=module();rng=np.random.default_rng(121);x=rng.normal(size=(500,2));u=rng.normal(size=(500,2))
    A=np.array([[.9,.1],[-.2,.8]]);B=np.array([[.2,.5],[.3,-.1]]);y=x@A.T+u@B.T
    fitted=m.fit_shared_models(x,y,u)
    sx=fitted['state_scaler'];su=fitted['control_scaler'];us=np.column_stack([su.transform(u),np.ones(len(u))])
    predicted=sx.inverse(sx.transform(x)@fitted['dmdc']['A'].T+us@fitted['dmdc']['B'].T)
    np.testing.assert_allclose(predicted,y,atol=1e-9)
    assert fitted['dmdc']['physical_lag']==5 and fitted['dmdc']['backend']=='pydmd.DMDc.fit'
    before=su.mean.copy();su.transform(np.full((10,2),1e10));np.testing.assert_array_equal(su.mean,before)

def test_population_evaluation_counts_all_tokens_and_full768_residual():
    m=module();a={'steps':np.arange(0,501,5),'telemetry_learning_rate':np.full(501,.05),
      'analysis_positions_pca32':np.zeros((101,600,32)), 'analysis_increment_residual_norm':np.full((100,600),2.),
      'analysis_state_residual_norm':2*np.arange(101)[:,None]*np.ones((1,600)), 'interval_increment_norm':np.full((100,600),np.sqrt(5)),
      'interval_phase_min_norm_fraction':np.r_[np.zeros(50),np.ones(50)],'token_role_mask':np.r_[np.uint8(1),np.zeros(599,dtype=np.uint8)]}
    a['analysis_positions_pca32'][:,:,0]=np.arange(101)[:,None]
    for key in m.CONTROL_FIELDS:a[key]=np.ones((100,600))
    fit={'state_scaler':m.Scaler(np.zeros(32),np.ones(32)),'control_scaler':m.Scaler(np.zeros(5),np.ones(5)),
      'control_train_min':np.zeros(5),'control_train_max':np.ones(5),'dmdc':{'A':np.eye(32),'B':np.zeros((32,6))},'dmd':{'A':np.eye(32),'b':np.zeros(32)}}
    class Writer:
        def __init__(self):self.rows=[]
        def writerow(self,r):self.rows.append(r)
    writer=Writer();totals={}
    m.evaluate_one(a,{'image_index':0,'film':'f','split':'test'},{'status':'success'},fit,totals,writer)
    selected=[r for r in writer.rows if r['method']=='persistence' and r['horizon']==5][0]
    assert selected['n']==60000 and selected['rmse768_lower']==pytest.approx(np.sqrt(5/768))
    assert selected['retained_energy_h5']==pytest.approx(.2)
    assert sum(v['n'] for k,v in totals.items() if k[4]==5 and k[5]=='persistence')==60000
    assert any(k[2]==-1 and k[4]==20 for k in totals)
