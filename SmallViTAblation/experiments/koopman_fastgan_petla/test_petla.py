import json
import numpy as np
import pytest
from petla import history,map_score,verify_schedules

def test_conflicting_boundary_is_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    np.savez(a/'LATENT_1.npz',step=1,features=np.ones((2,3)))
    np.savez(b/'LATENT_1.npz',step=1,features=np.zeros((2,3)))
    with pytest.raises(ValueError,match='resume boundary'):history([a,b])

def test_nan_cannot_win_validation(tmp_path):
    f=tmp_path/'score.json';f.write_text('[{"val/mAP_50_95": NaN}]')
    with pytest.raises(ValueError,match='Nonfinite'):map_score(f)

def test_hidden_extra_replacement_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'samples_epoch001.json').write_text(json.dumps(dict(frame_ids=['a','b','c'],replaced_slots=[])))
    (b/'samples_epoch001.json').write_text(json.dumps(dict(frame_ids=['x','b','x'],replaced_slots=[0])))
    with pytest.raises(ValueError,match='Unpaired'):verify_schedules(a,b,'x',1)
