"""Read-only photo selection and measured dense image/token identity tests.

No inference, production server restart, frontend edits or production writes.
Generated JPEGs and dense fixtures live only in pytest temporary directories.
"""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import serve
import dense_timeline


@pytest.fixture
def selection(tmp_path,monkeypatch):
    images=tmp_path/'images';images.mkdir()
    rows=[{'index':i,'image_name':f'image_{i}.jpg','image_sha256':'0'*64} for i in range(1500)]
    for i in (0,1,2,1499):
        yy,xx=np.mgrid[:24,:36]
        pixels=np.stack(((xx*7+i)%256,(yy*11+i)%256,((xx+yy)*5+i)%256),axis=2).astype(np.uint8)
        path=images/rows[i]['image_name']
        Image.fromarray(pixels).save(path,format='JPEG',quality=95)
        rows[i]['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    step=np.arange(501)[None,:]
    image=np.arange(1500)[:,None]
    confidence=(.1+image/5000+step/2000).astype(np.float32)
    iou=(.2+image/5000+step/3000).astype(np.float32)
    confidence[[2,1499],5]=.95;iou[[2,1499],5]=.97
    state={'metadata':{'images':rows},'arrays':{'confidence':confidence,'iou':iou}}
    monkeypatch.setattr(serve,'IMAGE_ROOT',images)
    monkeypatch.setattr(serve,'load',lambda:state)
    # Keep the genuine DenseTimeline verification/cache, shrinking only the
    # synthetic point count. Production DenseTimeline still gates all 900k.
    monkeypatch.setattr(dense_timeline,'COUNT',1800)
    root=tmp_path/'dense';root.mkdir()
    m={'status':'complete','point_count':1800,'pca_basis_sha256':dense_timeline.BASIS_SHA,
       'stride':5,'steps':list(range(0,501,5)),'interpolation':False,'files':{},'stats':{},
       'bounds':{'natural':[[-100,10000]]*3,'movement':[[-50,50]]*3},'norm_scale':1234.,'heat_scale':567.}
    indices=np.arange(1800)
    for t in m['steps']:
        array=np.empty((1800,4),dtype='<f4')
        array[:,0]=indices//600*7+t*.01
        array[:,1]=indices%600+t*.02
        array[:,2]=indices+t*.03
        array[:,3]=t*(1+(indices%600)/1000+(indices//600)/100)
        name=f'step_{t:03d}.bin';raw=array.tobytes();(root/name).write_bytes(raw)
        m['files'][str(t)]={'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        # Deliberately unrelated global values: selection must use actual
        # per-image/per-step metrics, not these means or sparse milestones.
        m['stats'][str(t)]={'both90':999,'confidence':.88,'iou':.77}
    (root/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
    reader=dense_timeline.DenseTimeline(root)
    monkeypatch.setattr(serve,'DENSE',reader)
    serve.sparse_curves.cache_clear()
    yield {'root':images,'rows':rows,'state':state,'reader':reader,'manifest':m}
    reader.binary.cache_clear()
    serve.sparse_curves.cache_clear()


@pytest.mark.parametrize('image',[0,1,2,1499])
def test_original_jpeg_same_bytes_decoded_photo_and_no_image_writes(selection,image):
    path=selection['root']/selection['rows'][image]['image_name']
    before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in selection['root'].iterdir()}
    actual=serve.image_binary(image)
    assert actual==path.read_bytes()
    with Image.open(path) as original,Image.open(io.BytesIO(actual)) as delivered:
        assert delivered.format=='JPEG' and delivered.size==(36,24)
        np.testing.assert_array_equal(np.asarray(original),np.asarray(delivered))
    assert before=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in selection['root'].iterdir()}


@pytest.mark.parametrize('image',[-1,1500,10**9,True,False,1.0,'1',None])
def test_invalid_indices_rejected_without_clamping_or_negative_index(selection,image):
    with pytest.raises(ValueError):serve.image_binary(image)
    with pytest.raises(ValueError):serve.frame('image',image,5,False,False)
    with pytest.raises(ValueError):serve.curves('image',image)


@pytest.mark.parametrize('name',['../image_0.jpg','..\\image_0.jpg','/image_0.jpg',
    'C:/image_0.jpg','C:\\image_0.jpg','C:image_0.jpg','dir/image_0.jpg',
    '%2e%2e%2fimage_0.jpg','image_0.jpg:stream','image_0.jpg\x00','image_0.png',''])
def test_metadata_name_rejects_traversal_drives_streams_and_non_jpeg(selection,name):
    selection['rows'][0]['image_name']=name
    with pytest.raises(ValueError,match='nazwa'):serve.image_binary(0)


def test_symlink_escape_is_rejected_even_with_matching_hash(selection,tmp_path):
    external=tmp_path/'outside.jpg'
    source=selection['root']/'image_0.jpg'
    external.write_bytes(source.read_bytes())
    link=selection['root']/'escape.jpg'
    try:link.symlink_to(external)
    except OSError:pytest.skip('Windows symlink privilege unavailable')
    selection['rows'][0]['image_name']='escape.jpg'
    with pytest.raises(ValueError,match='poza katalog'):serve.image_binary(0)


def test_hash_change_is_rechecked_on_every_request(selection):
    path=selection['root']/'image_0.jpg'
    before=serve.image_binary(0)
    # Same length and same path must not accidentally pass an image cache.
    path.write_bytes(bytes([before[0]^1])+before[1:])
    with pytest.raises(ValueError,match='image_sha256'):serve.image_binary(0)
    path.write_bytes(before)
    selection['rows'][0]['image_sha256']='bad'
    with pytest.raises(ValueError,match='image_sha256'):serve.image_binary(0)


def test_matching_hash_does_not_bypass_decode_check(selection):
    path=selection['root']/'image_0.jpg'
    path.write_bytes(b'not a JPEG')
    selection['rows'][0]['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='zdekodować'):serve.image_binary(0)
    Image.new('RGB',(10,10)).save(path,format='PNG')
    selection['rows'][0]['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='nie jest.*JPEG'):serve.image_binary(0)


def test_metadata_order_missing_file_and_missing_hash(selection):
    selection['rows'][0]['index']=1
    with pytest.raises(ValueError,match='kolejność'):serve.image_binary(0)
    selection['rows'][0]['index']=0
    del selection['rows'][0]['image_sha256']
    with pytest.raises(ValueError,match='image_sha256'):serve.image_binary(0)
    selection['rows'][0]['image_sha256']='0'*64
    (selection['root']/'image_0.jpg').unlink()
    with pytest.raises(FileNotFoundError):serve.image_binary(0)


def request(path):
    """Exercise HTTP routing/headers without opening or restarting any server."""
    h=serve.Handler.__new__(serve.Handler)
    response={'headers':{},'body':io.BytesIO()}
    h.path=path;h.wfile=response['body']
    h.send_response=lambda status:response.update(status=status)
    h.send_header=lambda name,value:response['headers'].update({name:str(value)})
    h.end_headers=lambda:None
    h.send_error=lambda status,message=None,explain=None:response.update(status=status,error=explain or message)
    h.do_GET()
    response['body']=response['body'].getvalue()
    return response


def test_image_http_jpeg_content_type_length_and_readonly_cache_policy(selection):
    response=request('/api/image?image=1499')
    assert response['status']==200
    assert response['body']==serve.image_binary(1499)
    assert response['headers']['Content-Type']=='image/jpeg'
    assert int(response['headers']['Content-Length'])==len(response['body'])
    assert response['headers']['Cache-Control']=='no-store'


@pytest.mark.parametrize('query',['','image=','image=-1','image=1500','image=1.0','image=True',
    'image=00','image=%2B1','image=0&image=1','image=0&name=../other.jpg','image=../0'])
def test_http_image_query_is_single_strict_integer(selection,query):
    assert request('/api/image?'+query)['status']==400


def test_real_http_error_serialization_keeps_polish_out_of_reason_phrase():
    # Exercise BaseHTTPRequestHandler's real header encoder without a socket
    # or a server restart. Non-Latin-1 descriptions must stay in the body.
    handler=serve.Handler.__new__(serve.Handler)
    handler.path='/api/image?image=1500'
    handler.requestline='GET /api/image?image=1500 HTTP/1.1'
    handler.request_version='HTTP/1.1'
    handler.command='GET'
    handler.wfile=io.BytesIO()
    handler.do_GET()
    headers,body=handler.wfile.getvalue().split(b'\r\n\r\n',1)
    assert headers.startswith(b'HTTP/1.0 400 Invalid request\r\n')
    assert 'liczbą całkowitą 0..1499' in body.decode('utf-8')


def test_dense_image_identity_at_every_true_step(selection):
    dense=selection['reader']
    assert dense.load()['steps']==list(range(0,501,5))
    for step in range(0,501,5):
        selected=serve.frame('image',2,step,False,False)
        expected=np.frombuffer(dense.binary(step),dtype='<f4').reshape(1800,4)[1200:1800]
        np.testing.assert_allclose(selected['xyz'],expected[:,:3],atol=5.1e-6,rtol=0)
        np.testing.assert_allclose(selected['norm'],expected[:,3],atol=5.1e-6,rtol=0)
        np.testing.assert_allclose(selected['heatmap'],expected[:,3].reshape(20,30),atol=5.1e-6,rtol=0)
        assert selected['tokens']==list(range(600))
        assert selected['images']==[2]*600 and selected['indices']==list(range(1200,1800))
        assert selected['step']==step and selected['mode']=='image'
        assert selected['count']==selected['total']==600
        assert selected['confidence']==float(selection['state']['arrays']['confidence'][2,step])
        assert selected['iou']==float(selection['state']['arrays']['iou'][2,step])
        assert dense.binary.cache_info().currsize<=4


def test_movement_relative_binary_zero_and_shared_scales_not_refit_per_image(selection):
    for image in (0,1,2):
        selected=serve.frame('image',image,255,True,True)
        dense=selection['reader'];span=slice(image*600,(image+1)*600)
        current=np.frombuffer(dense.binary(255),dtype='<f4').reshape(1800,4)[span]
        initial=np.frombuffer(dense.binary(0),dtype='<f4').reshape(1800,4)[span]
        np.testing.assert_allclose(selected['xyz'],current[:,:3]-initial[:,:3],atol=5.1e-6,rtol=0)
        np.testing.assert_allclose(selected['norm'],current[:,3],atol=5.1e-6,rtol=0)
        assert selected['bounds']==selection['manifest']['bounds']['movement']
        assert selected['norm_scale']==selected['heat_scale']==1234.
        assert selected['lines']==[]  # no fabricated trails/interpolated history
        zero=serve.frame('image',image,0,True,False)
        assert np.count_nonzero(zero['xyz'])==0
        assert np.count_nonzero(zero['norm'])==0


@pytest.mark.parametrize('step',[-1,1,4,6,499,501,True,5.0])
def test_unsaved_steps_refused_instead_of_interpolated_or_clamped(selection,step):
    with pytest.raises(ValueError,match='rzeczywistego kroku'):serve.frame('image',0,step,False,False)


def test_geometry_false_keeps_true_image_heatmap_metrics_and_global_success_count(selection):
    selected=serve.frame('image',2,5,False,False,geometry=False)
    assert selected['xyz']==selected['norm']==selected['tokens']==selected['images']==selected['indices']==[]
    assert np.asarray(selected['heatmap']).shape==(20,30)
    assert selected['confidence']==float(selection['state']['arrays']['confidence'][2,5])
    assert selected['both90']==2 and selected['selected_both90']==1 and selected['selected_image_count']==1
    assert selected['norm_median']>0 and selected['norm_max']>0


def test_curves_real_all_501_steps_and_bounded_cache(selection):
    for image in range(100):
        c=serve.curves('image',image)
        assert c['steps']==list(range(501))
        assert c['confidence']==selection['state']['arrays']['confidence'][image].tolist()
        assert c['iou']==selection['state']['arrays']['iou'][image].tolist()
        assert serve.sparse_curves.cache_info().currsize<=80


def test_http_frame_image_dense_contract(selection):
    response=request('/api/frame?mode=image&image=2&step=5&movement=1')
    assert response['status']==200
    result=json.loads(response['body'])
    assert result['indices']==list(range(1200,1800)) and result['step']==5
    assert result['confidence']==float(selection['state']['arrays']['confidence'][2,5])
    assert request('/api/frame?mode=image&image=-1&step=5')['status']==400
    assert request('/api/frame?mode=image&image=2&step=6')['status']==400


def test_actual_local_photo_is_exact_original_readonly(monkeypatch):
    path=serve.DATA/'metadata.json'
    if not path.is_file():pytest.skip('local real metadata unavailable')
    meta=json.loads(path.read_text(encoding='utf-8'))
    original=serve.IMAGE_ROOT/meta['images'][0]['image_name']
    if not original.is_file():pytest.skip('local real JPEG unavailable')
    monkeypatch.setattr(serve,'load',lambda:{'metadata':meta})
    raw=serve.image_binary(0)
    assert raw==original.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['images'][0]['image_sha256']
    with Image.open(original) as source,Image.open(io.BytesIO(raw)) as delivered:
        np.testing.assert_array_equal(np.asarray(source),np.asarray(delivered))


def test_actual_dense_last_image_slice_and_real_metrics_without_inference(monkeypatch):
    if not (serve.DATA/'dense/manifest.json').is_file() or not (serve.DATA/'projection.npz').is_file():
        pytest.skip('local real dense timeline unavailable')
    with np.load(serve.DATA/'projection.npz',allow_pickle=False) as f:
        state={'arrays':{k:f[k] for k in ('confidence','iou')}}
    monkeypatch.setattr(serve,'load',lambda:state)
    reader=dense_timeline.DenseTimeline(serve.DATA/'dense')
    monkeypatch.setattr(serve,'DENSE',reader)
    try:
        assert reader.load()['steps']==list(range(0,501,5))
        for step in (0,5,500):
            result=serve.frame('image',1499,step,True,False)
            packed=np.frombuffer(reader.binary(step),dtype='<f4').reshape(900000,4)[1499*600:1500*600]
            initial=np.frombuffer(reader.binary(0),dtype='<f4').reshape(900000,4)[1499*600:1500*600]
            np.testing.assert_allclose(result['xyz'],packed[:,:3]-initial[:,:3],atol=5.1e-6,rtol=0)
            np.testing.assert_allclose(result['norm'],packed[:,3],atol=5.1e-6,rtol=0)
            assert result['confidence']==float(state['arrays']['confidence'][1499,step])
            assert result['iou']==float(state['arrays']['iou'][1499,step])
    finally:reader.binary.cache_clear()
