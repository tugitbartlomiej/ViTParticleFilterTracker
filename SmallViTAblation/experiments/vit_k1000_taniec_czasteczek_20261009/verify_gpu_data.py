"""Verify all GPU vertices against the source, including order and full-D norm."""
import hashlib,json,time
import numpy as np
import serve
start=time.perf_counter();a=serve.load()['arrays'];checks=[]
for k,step in enumerate((0,100,500)):
    raw=serve.cloud_binary(step);packed=np.frombuffer(raw,dtype='<f4').reshape(1500,600,4)
    assert len(raw)==900000*16
    assert np.array_equal(packed[:,:,:3],a['positions'][:,k])
    assert np.array_equal(packed[:,:,3],a['movement_norm'][:,k])
    assert np.isfinite(packed).all()
    for movement in (False,True):
        d=serve.frame('global',0,step,movement,False,False)
        assert d['count']==d['total']==900000 and not d['xyz']
        assert np.isclose(d['norm_median'],np.median(a['movement_norm'][:,k]))
        xy=a['positions']-a['positions'][:,:1] if movement else a['positions']
        assert d['bounds']==serve.limits(xy)
    checks.append({'step':step,'count':900000,'bytes':len(raw),'bitwise_positions_and_norm':True,'sha256':hashlib.sha256(raw).hexdigest()})
individual=serve.frame('image',100,500,False,False)
assert individual['count']==600
np.testing.assert_allclose(individual['xyz'],a['positions'][100,2],atol=5.1e-6,rtol=0)
panel=serve.frame('exact',-1,100,False,True)
assert panel['count']==128 and len(panel['lines'])==128
np.testing.assert_allclose(panel['xyz'],a['control_positions'][:,100].reshape(128,3),atol=5.1e-6,rtol=0)
report={'status':'passed','milestones':checks,'order':'image * 600 + patch','dtype':'little-endian float32','stride_bytes':16,'max_gpu_vertex_buffers_bytes':43200000,'common_axes_and_time_bounds':True,'all_points_verified':True,'individual600_and_panel128_preserved':True,'seconds':time.perf_counter()-start,'source_npz_sha256':hashlib.sha256((serve.DATA/'projection.npz').read_bytes()).hexdigest()}
(serve.OUT/'WERYFIKACJA_SHADERA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
