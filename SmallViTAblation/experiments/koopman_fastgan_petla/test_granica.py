import pytest
from granica import canonical_boundary

def fixture():
    return {'global_step':4,'optimizer_states':[{'moment':123}], 'loops':{'fit_loop':{
        'epoch_progress':{'current':{'completed':1}},
        'epoch_loop.batch_progress':{'total':{'completed':16},'current':{'completed':16},'is_last_batch':True},
        'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{'step':{'total':{'completed':4},'current':{'completed':4}}}},
        'epoch_loop.val_loop.batch_progress':{'total':{'completed':2},'current':{'completed':2},'is_last_batch':True}}}}

def test_boundary_preserves_lifetime_update_and_optimizer_state():
    c=canonical_boundary(fixture(),4);f=c['loops']['fit_loop']
    assert f['epoch_loop.batch_progress']['current']['completed']==0
    assert f['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed']==4
    assert c['optimizer_states']==[{'moment':123}]
    assert c['experiment_raw_loops']['fit_loop']['epoch_loop.batch_progress']['is_last_batch']

def test_mid_epoch_is_not_canonicalized():
    c=fixture();c['global_step']=3
    with pytest.raises(ValueError):canonical_boundary(c,4)
