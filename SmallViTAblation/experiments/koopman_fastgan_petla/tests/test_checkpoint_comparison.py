import torch
import pytest
from verify_states import weight_error,equivalent

def test_native_state_dict_allows_empty_registers_and_bool_buffers():
    a={'registers':torch.empty(0,384),'mask':torch.tensor([True,False]),'weight':torch.ones(3)}
    b={k:v.clone() for k,v in a.items()}
    assert weight_error(a,b)==0
    assert equivalent(a,b)
    b['mask'][0]=False
    assert weight_error(a,b)==1
    assert not equivalent(a,b)

def test_nan_model_cannot_pass_as_identical():
    with pytest.raises(AssertionError,match='Nonfinite'):
        weight_error({'w':torch.tensor([float('nan')])},{'w':torch.ones(1)})
