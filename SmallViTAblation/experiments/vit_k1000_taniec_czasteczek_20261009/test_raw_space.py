"""RAM-only raw channel inspection. Set RAW_SPACE_REAL_SMOKE=1 for one
authenticated SSH fetch of actual image 1469, without inference or disk output.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import raw_space as raw
import serve


def blob(arrays):
    stream=io.BytesIO()
    np.savez_compressed(stream,**arrays)
    return stream.getvalue()


def tree(files):
    return hashlib.sha256(''.join(f"{r['relative_path']}\0{r['sha256']}\n" for r in
                        sorted(files,key=lambda r:r['relative_path'])).encode()).hexdigest()


@pytest.fixture
def sample(tmp_path):
    before=(np.arange(600,dtype=np.float32)[:,None]/1000+np.arange(768,dtype=np.float32)[None,:]/768)
    direction=np.arange(1,769,dtype=np.float32)/768
    arrays={'before':before,'trajectory':np.stack((before+.1*direction,before-.2*direction)),
            'trajectory_steps':np.array([100,500],np.int32)}
    payload=blob(arrays)
    files=[{'index':i,'relative_path':f'results/production/latents/{i:06d}.npz',
            'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)} for i in range(1500)]
    frozen={'source_root':raw.REMOTE_ROOT,'files':files,'tree_sha256':tree(files)}
    path=tmp_path/'SOURCE_FREEZE.json'
    def save():path.write_text(json.dumps(frozen),encoding='utf-8')
    save()
    rows=[{'index':i,'source_npz_sha256':files[i]['sha256'],'image_name':f'image_{i}.jpg'} for i in range(1500)]
    confidence=np.full((1500,501),.7,np.float32);iou=np.full((1500,501),.6,np.float32)
    confidence[:,100]=.91;iou[:,100]=.93
    state={'metadata':{'images':rows},'arrays':{'confidence':confidence,'iou':iou}}
    calls=[]
    def fetch(image,size):
        calls.append((image,size))
        return payload
    engine=raw.RawSpace(tmp_path,fetch)
    return {'engine':engine,'state':state,'calls':calls,'arrays':arrays,'blob':payload,
            'files':files,'frozen':frozen,'save':save,'root':tmp_path}


def test_original_channels_profiles_full_norm_fixed_bounds_and_metrics(sample):
    axes=(767,0,345);patch=599
    states=np.concatenate((sample['arrays']['before'][None],sample['arrays']['trajectory']))
    expected_norm=np.linalg.norm(states.astype(np.float64)-states[0].astype(np.float64),axis=-1)
    natural=[];movement=[]
    for k,step in enumerate((0,100,500)):
        f=sample['engine'].frame(sample['state'],1,step,patch,axes,False)
        delta=states[k].astype(np.float64)-states[0].astype(np.float64)
        np.testing.assert_array_equal(f['xyz'],states[k][:,list(axes)].astype(np.float64))
        np.testing.assert_allclose(f['norm'],expected_norm[k],rtol=0,atol=0)
        np.testing.assert_allclose(f['heatmap'],expected_norm[k].reshape(20,30),rtol=0,atol=0)
        np.testing.assert_array_equal(f['profile']['before'],states[0,patch])
        np.testing.assert_array_equal(f['profile']['current'],states[k,patch])
        np.testing.assert_array_equal(f['profile']['delta'],delta[patch])
        assert f['tokens']==list(range(600)) and f['images']==[1]*600
        assert f['indices']==list(range(600,1200)) and f['channel_axes']==list(axes)
        assert f['actual_steps']==[0,100,500] and f['count']==f['total']==600
        assert f['norm_scale']==f['heat_scale']==float(expected_norm.max())
        assert f['confidence']==float(sample['state']['arrays']['confidence'][1,step])
        assert f['iou']==float(sample['state']['arrays']['iou'][1,step])
        assert f['both90']==(1500 if step==100 else 0)
        assert f['selected_both90']==(1 if step==100 else 0)
        assert f['source_sha256']==hashlib.sha256(sample['blob']).hexdigest()
        assert f['metadata']['source_origin']=='original_full768_npz'
        assert f['metadata']['source_path'].endswith('/000001.npz')
        assert f['lines']==f['bar_tokens']==f['bar_norm']==[]
        natural.append(f['bounds'])
        moved=sample['engine'].frame(sample['state'],1,step,patch,axes,True)
        np.testing.assert_array_equal(moved['xyz'],delta[:,list(axes)])
        assert moved['norm']==f['norm']
        movement.append(moved['bounds'])
    assert natural[0]==natural[1]==natural[2] and movement[0]==movement[1]==movement[2]
    assert len(sample['calls'])==1


@pytest.mark.parametrize('kwargs',[{'image':-1},{'image':1500},{'image':True},{'image':'0'},
    {'step':5},{'step':True},{'patch':-1},{'patch':600},{'patch':2.0},
    {'axes':(0,0,2)},{'axes':(0,1,768)},{'axes':(-1,1,2)},{'axes':(False,1,2)},
    {'axes':(0,1)},{'axes':'0,1,2'},{'movement':1}])
def test_invalid_selection_fails_before_fetch(sample,kwargs):
    args={'image':0};args.update(kwargs)
    with pytest.raises(ValueError):sample['engine'].frame(sample['state'],**args)
    assert sample['calls']==[]


def test_lru_two_and_concurrent_fetch_deduplication(sample):
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda _:sample['engine'].frame(sample['state'],3,100),range(4)))
    assert len(sample['calls'])==1 and all(f['step']==100 for f in result)
    for image in (4,3,5):sample['engine'].frame(sample['state'],image)
    assert len(sample['engine'].cache)==2
    assert [key[0] for key in sample['engine'].cache]==[3,5]
    sample['engine'].frame(sample['state'],4)
    assert len(sample['engine'].cache)==2 and len(sample['calls'])==4


def test_frame_is_ram_only_and_cache_excludes_archive_and_other_trajectories(sample,monkeypatch):
    before={p.name:p.read_bytes() for p in sample['root'].iterdir()}
    def fail(*args,**kwargs):raise AssertionError('disk write forbidden')
    monkeypatch.setattr(Path,'write_bytes',fail);monkeypatch.setattr(Path,'write_text',fail)
    sample['engine'].frame(sample['state'],0,500)
    assert before=={p.name:p.read_bytes() for p in sample['root'].iterdir()}
    entry=next(iter(sample['engine'].cache.values()))
    assert entry['states'].shape==(3,600,768) and not entry['states'].flags.writeable
    assert not any(isinstance(v,bytes) for v in entry.values())
    assert 'control_z' not in entry and 'after' not in entry


@pytest.mark.parametrize('mutation',['hash','metadata_hash','path','duplicate','tree','root','size'])
def test_source_freeze_identity_validation(sample,mutation):
    if mutation=='hash':sample['engine'].fetcher=lambda i,n:b'x'*n
    if mutation=='metadata_hash':sample['state']['metadata']['images'][0]['source_npz_sha256']='f'*64
    if mutation=='path':sample['files'][0]['relative_path']='../../different.npz'
    if mutation=='duplicate':sample['files'][1]['index']=0
    if mutation=='root':sample['frozen']['source_root']='/another/run'
    if mutation=='size':sample['files'][0]['bytes']=raw.MAX_SOURCE_BYTES+1
    sample['frozen']['tree_sha256']=tree(sample['files'])
    if mutation=='tree':sample['frozen']['tree_sha256']='f'*64
    sample['save']()
    with pytest.raises(ValueError):sample['engine'].frame(sample['state'],0)
    assert not sample['engine'].cache


@pytest.mark.parametrize('mutation',['before_shape','before_dtype','trajectory_shape','nonfinite','steps',
                                   'float_steps','telemetry_axis','telemetry_nan','incomplete_telemetry'])
def test_npz_schema_finite_steps_telemetry(sample,mutation):
    arrays=dict(sample['arrays'])
    if mutation=='before_shape':arrays['before']=np.zeros((599,768),np.float32)
    if mutation=='before_dtype':arrays['before']=arrays['before'].astype(np.float64)
    if mutation=='trajectory_shape':arrays['trajectory']=np.zeros((3,600,768),np.float32)
    if mutation=='nonfinite':arrays['trajectory']=arrays['trajectory'].copy();arrays['trajectory'][1,599,767]=np.nan
    if mutation=='steps':arrays['trajectory_steps']=np.array([100,499],np.int32)
    if mutation=='float_steps':arrays['trajectory_steps']=np.array([100.,500.])
    if mutation.startswith('telemetry') or mutation=='incomplete_telemetry':
        arrays.update(telemetry_step=np.arange(501),telemetry_confidence=np.full(501,.7),telemetry_iou=np.full(501,.6))
        if mutation=='telemetry_axis':arrays['telemetry_step'][1]=2
        if mutation=='telemetry_nan':arrays['telemetry_iou'][0]=np.nan
        if mutation=='incomplete_telemetry':del arrays['telemetry_iou']
    payload=blob(arrays)
    sample['files'][0].update(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    sample['state']['metadata']['images'][0]['source_npz_sha256']=sample['files'][0]['sha256']
    sample['frozen']['tree_sha256']=tree(sample['files']);sample['save']()
    sample['engine'].fetcher=lambda i,n:payload
    with pytest.raises(ValueError):sample['engine'].frame(sample['state'],0)


def test_npz_metrics_used_and_checked_against_original_export(sample):
    arrays=dict(sample['arrays'])
    arrays.update(telemetry_step=np.arange(501),
        telemetry_confidence=sample['state']['arrays']['confidence'][0],
        telemetry_iou=sample['state']['arrays']['iou'][0])
    payload=blob(arrays)
    sample['files'][0].update(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    sample['state']['metadata']['images'][0]['source_npz_sha256']=sample['files'][0]['sha256']
    sample['frozen']['tree_sha256']=tree(sample['files']);sample['save']()
    sample['engine'].fetcher=lambda i,n:payload
    f=sample['engine'].frame(sample['state'],0,100)
    assert f['metadata']['metric_origin']=='original_npz_telemetry'
    sample['state']['arrays']['confidence'][0,100]=.3
    with pytest.raises(ValueError,match='provenance mismatch'):sample['engine'].frame(sample['state'],0,100)


def test_fixed_shell_false_ssh_capped_stdout_and_timeout(sample,monkeypatch):
    calls=[]
    class Process:
        def __init__(self,args,**kwargs):
            calls.append((args,kwargs));self.stdout=io.BytesIO(sample['blob']);self.code=0
        def wait(self,timeout=None):return self.code
        def poll(self):return self.code
        def kill(self):raise AssertionError('successful SSH must not be killed')
    monkeypatch.setattr(raw.subprocess,'Popen',Process)
    assert raw.fetch(1469,len(sample['blob']))==sample['blob']
    args,kwargs=calls[0]
    assert args==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','eden-cluster','cat','--',
        raw.REMOTE_LATENTS+'/001469.npz']
    assert kwargs['shell'] is False and kwargs['stdout']==subprocess.PIPE
    assert not any('StrictHostKeyChecking=no' in a or 'KnownHostsFile' in a for a in args)
    with pytest.raises(ValueError):raw.fetch('1469; rm',1)
    assert len(calls)==1


def test_ssh_timeout_stops_only_own_subprocess(monkeypatch):
    stop=threading.Event();instances=[]
    class Stream:
        def read(self,n):stop.wait();return b''
        def close(self):pass
    class Process:
        def __init__(self,*args,**kwargs):self.stdout=Stream();self.code=None;instances.append(self)
        def poll(self):return self.code
        def kill(self):self.code=-1;stop.set()
        def wait(self,timeout=None):return self.code
    monkeypatch.setattr(raw.subprocess,'Popen',Process)
    monkeypatch.setattr(raw,'TIMEOUT',.01)
    with pytest.raises(TimeoutError):raw.fetch(1469,1)
    assert len(instances)==1 and instances[0].code==-1


def http(path):
    handler=serve.Handler.__new__(serve.Handler)
    handler.path=path;handler.wfile=io.BytesIO()
    response={}
    handler.send_response=lambda status:response.update(status=status)
    handler.send_header=lambda *args:None;handler.end_headers=lambda:None
    handler.send_error=lambda status,message=None,explain=None:response.update(status=status,error=explain or message)
    handler.do_GET()
    if response['status']==200:response['frame']=json.loads(handler.wfile.getvalue())
    return response


def test_http_raw_contract_without_server_restart(sample,monkeypatch):
    monkeypatch.setattr(serve,'load',lambda:sample['state'])
    monkeypatch.setattr(raw,'RAW',sample['engine'])
    response=http('/api/raw?image=0&step=100&patch=599&axes=767,1,345&movement=1')
    assert response['status']==200
    f=response['frame']
    assert f['channel_axes']==[767,1,345] and f['actual_steps']==[0,100,500]
    assert len(f['xyz'])==len(f['norm'])==600 and len(f['profile']['current'])==768
    for query in ('image=-1','image=0&step=5','image=0&axes=0,0,2','image=0&axes=0,1,768',
                  'image=0&image=1','image=0&movement=true','image=0&file=../a','image=0&patch=600'):
        assert http('/api/raw?'+query)['status']==400
    assert len(sample['calls'])==1


@pytest.mark.skipif(os.environ.get('RAW_SPACE_REAL_SMOKE')!='1',reason='explicit one original Eden NPZ in RAM')
def test_actual_1469_endpoint_three_states_no_inference_or_disk(monkeypatch):
    # Only metric arrays and provenance are needed from the small local bundle.
    meta=json.loads((raw.DATA/'metadata.json').read_text(encoding='utf-8'))
    with np.load(raw.DATA/'projection.npz',allow_pickle=False) as z:
        state={'metadata':meta,'arrays':{key:z[key] for key in ('confidence','iou')}}
    calls=[]
    def fetch(image,size):calls.append(image);return raw.fetch(image,size)
    engine=raw.RawSpace(fetcher=fetch)
    monkeypatch.setattr(raw,'RAW',engine);monkeypatch.setattr(serve,'load',lambda:state)
    for step in (0,100,500):
        response=http(f'/api/raw?image=1469&step={step}&patch=599&axes=0,1,767&movement=1')
        assert response['status']==200,response
        f=response['frame']
        assert f['source_sha256']==meta['images'][1469]['source_npz_sha256']
        assert f['actual_steps']==[0,100,500] and len(f['profile']['before'])==768
        assert f['confidence']==float(state['arrays']['confidence'][1469,step])
        assert f['iou']==float(state['arrays']['iou'][1469,step])
        if step==0:assert np.count_nonzero(f['xyz'])==np.count_nonzero(f['norm'])==0
    assert calls==[1469] and len(engine.cache)==1
