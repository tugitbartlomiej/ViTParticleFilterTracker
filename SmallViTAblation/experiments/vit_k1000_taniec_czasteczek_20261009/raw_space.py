"""Original three full-768D states, fetched once per image into bounded RAM.

No model, training, interpolation, disk downloads or projections. Only the
trusted original NPZ is read over authenticated SSH. The two-entry LRU keeps
extracted states, not compressed archives or full control histories.
"""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import zipfile

import numpy as np

REMOTE_ROOT='/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008'
REMOTE_LATENTS=REMOTE_ROOT+'/results/production/latents'
SSH_HOST='eden-cluster'
TIMEOUT=60
MAX_SOURCE_BYTES=64<<20
STEPS=(0,100,500)
DATA=Path(__file__).resolve().parents[2]/'outputs/vit_k1000_taniec_czasteczek_20261009/data'


def validate(image,step=0,patch=0,axes=(0,1,2),movement=False):
    for name,value,maximum in [('image',image,1499),('patch',patch,599)]:
        if type(value) is not int or not 0<=value<=maximum:
            raise ValueError(f'{name} must be an integer in 0..{maximum}')
    if type(step) is not int or step not in STEPS:
        raise ValueError('Only actual original raw steps 0,100,500 are available')
    if (not isinstance(axes,(tuple,list)) or len(axes)!=3 or
            any(type(a) is not int or not 0<=a<768 for a in axes) or len(set(axes))!=3):
        raise ValueError('axes must be three distinct integer channels in 0..767')
    if type(movement) is not bool:
        raise ValueError('movement must be boolean')


def fetch(image,expected_bytes):
    """Fixed SSH command, capped stdout and a total 60-second deadline.

    The only process stopped on error is this request's own SSH subprocess.
    No host-key/authentication bypass, shell=True, temp files or scp.
    """
    validate(image)
    if type(expected_bytes) is not int or not 0<expected_bytes<=MAX_SOURCE_BYTES:
        raise ValueError('Invalid bounded source size')
    remote=f'{REMOTE_LATENTS}/{image:06d}.npz'
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',SSH_HOST,'cat','--',remote]
    started=time.monotonic()
    process=subprocess.Popen(args,shell=False,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    result={}
    def read():
        try:result['blob']=process.stdout.read(expected_bytes+1)
        except BaseException as exc:result['error']=exc
    reader=threading.Thread(target=read,daemon=True)
    try:
        reader.start()
        reader.join(TIMEOUT)
        if reader.is_alive():raise TimeoutError('Original NPZ SSH read exceeded 60 seconds')
        if 'error' in result:raise result['error']
        blob=result['blob']
        if len(blob)>expected_bytes:raise ValueError('Remote NPZ exceeds frozen source size')
        remaining=TIMEOUT-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Original NPZ SSH timeout')
        try:code=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:raise TimeoutError('Original NPZ SSH timeout') from exc
        if code:raise RuntimeError(f'Authenticated SSH read failed (exit {code})')
        if len(blob)!=expected_bytes:raise ValueError('Remote NPZ does not match frozen source size')
        return blob
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join()
        process.stdout.close()


def _bounds(states):
    flat=states.reshape(-1,768)
    low,high=flat.min(axis=0),flat.max(axis=0)
    pad=np.maximum((high-low)*.04,.03)
    return np.column_stack((low-pad,high+pad))


class RawSpace:
    def __init__(self,data=DATA,fetcher=None):
        self.data=Path(data)
        self.fetcher=fetcher or fetch
        self.lock=threading.Lock()
        self.cache=OrderedDict()

    def _record(self,image,row):
        raw=(self.data/'SOURCE_FREEZE.json').read_bytes()
        frozen=json.loads(raw)
        files=frozen.get('files')
        if frozen.get('source_root')!=REMOTE_ROOT or not isinstance(files,list):
            raise ValueError('Wrong original source freeze root/schema')
        tree=hashlib.sha256(''.join(f"{r['relative_path']}\0{r['sha256']}\n" for r in
                                 sorted(files,key=lambda r:r['relative_path'])).encode()).hexdigest()
        if tree!=frozen.get('tree_sha256'):
            raise ValueError('Source freeze tree SHA256 mismatch')
        by_index={}
        for record in files:
            if 'index' not in record:continue
            i=record['index']
            if type(i) is not int or not 0<=i<1500 or i in by_index:
                raise ValueError('Invalid/duplicate source freeze index')
            if record.get('relative_path')!=f'results/production/latents/{i:06d}.npz':
                raise ValueError('Invalid original NPZ path')
            by_index[i]=record
        if len(by_index)!=1500:raise ValueError('Source freeze must cover exactly 1500 images')
        record=by_index[image]
        digest=record.get('sha256')
        if (not isinstance(row,dict) or type(row.get('index')) is not int or row['index']!=image or
                not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or
                row.get('source_npz_sha256')!=digest):
            raise ValueError('Original NPZ SHA256/metadata image identity mismatch')
        if type(record.get('bytes')) is not int or not 0<record['bytes']<=MAX_SOURCE_BYTES:
            raise ValueError('Invalid bounded original NPZ size')
        return record,hashlib.sha256(raw).hexdigest(),tree

    def _parse(self,blob,image,row,record):
        if len(blob)!=record['bytes'] or hashlib.sha256(blob).hexdigest()!=record['sha256']:
            raise ValueError('Original NPZ bytes/SHA256 mismatch')
        # Bound decompression before touching .npy arrays. Never unpack an NPZ
        # to disk or extract large, unrelated control_z trajectories.
        caps={'before':600*768*4+65536,'trajectory':2*600*768*4+65536,
              'trajectory_steps':65536,'metadata_json_utf8':1<<20,
              'telemetry_step':65536,'telemetry_confidence':65536,'telemetry_iou':65536}
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate NPZ member')
            for key,cap in caps.items():
                if key+'.npy' in names and archive.getinfo(key+'.npy').file_size>cap:
                    raise ValueError('Oversized NPZ member: '+key)
        try:
            with np.load(io.BytesIO(blob),allow_pickle=False) as z:
                before=z['before'];trajectory=z['trajectory'];steps=z['trajectory_steps']
                if before.shape!=(600,768) or before.dtype!=np.float32:
                    raise ValueError('Expected fp32 before[600,768]')
                if trajectory.shape!=(2,600,768) or trajectory.dtype!=np.float32:
                    raise ValueError('Expected fp32 trajectory[2,600,768]')
                if steps.shape!=(2,) or steps.dtype.kind not in 'iu' or not np.array_equal(steps,[100,500]):
                    raise ValueError('Only genuine trajectory_steps [100,500] accepted')
                states=np.concatenate((before[None],trajectory))
                if not np.isfinite(states).all():raise ValueError('Nonfinite original raw states')
                if 'metadata_json_utf8' in z.files:
                    embedded=json.loads(z['metadata_json_utf8'].tobytes().decode('utf-8'))
                    for key in ('index','image_name','image_sha256','checkpoint_sha256','manifest_sha256'):
                        if key in row and embedded.get(key)!=row[key]:
                            raise ValueError('Original embedded metadata mismatch: '+key)
                present=[k in z.files for k in ('telemetry_step','telemetry_confidence','telemetry_iou')]
                metrics=None
                if any(present):
                    if not all(present):raise ValueError('Incomplete original telemetry')
                    axis=z['telemetry_step']
                    if axis.shape!=(501,) or axis.dtype.kind not in 'iu' or not np.array_equal(axis,np.arange(501)):
                        raise ValueError('Invalid actual original telemetry steps')
                    metrics={}
                    for key in ('confidence','iou'):
                        values=z['telemetry_'+key]
                        if (values.shape!=(501,) or values.dtype.kind not in 'fi' or not np.isfinite(values).all()
                                or ((values<0)|(values>1)).any()):
                            raise ValueError('Invalid original telemetry '+key)
                        metrics[key]=values.copy()
        except (KeyError,zipfile.BadZipFile) as exc:
            raise ValueError('Invalid original NPZ schema') from exc
        delta=states.astype(np.float64)-states[0].astype(np.float64)
        norm=np.linalg.norm(delta,axis=-1)
        entry={'states':states,'norm':norm,'natural_bounds':_bounds(states.astype(np.float64)),
               'movement_bounds':_bounds(delta),'norm_scale':max(float(norm.max()),1e-6),'metrics':metrics}
        for key,value in entry.items():
            if isinstance(value,np.ndarray):value.flags.writeable=False
        return entry

    def frame(self,state,image,step=0,patch=0,axes=(0,1,2),movement=False):
        validate(image,step,patch,axes,movement)
        if not state or not isinstance(state.get('metadata',{}).get('images'),list) or len(state['metadata']['images'])!=1500:
            raise ValueError('Original run metadata for 1500 images required')
        row=state['metadata']['images'][image]
        # Lock includes fetch+validation+admission, so concurrent requests for
        # one image cannot duplicate SSH reads or decompress extra archives.
        with self.lock:
            record,freeze_sha,tree_sha=self._record(image,row)
            cache_key=(image,record['sha256'],record['bytes'],tuple(row.get(key) for key in
                       ('image_name','image_sha256','checkpoint_sha256','manifest_sha256')))
            if cache_key not in self.cache:
                entry=self._parse(self.fetcher(image,record['bytes']),image,row,record)
                self.cache[cache_key]=entry
                while len(self.cache)>2:self.cache.popitem(last=False)
            self.cache.move_to_end(cache_key)
            entry=self.cache[cache_key]
        k=STEPS.index(step)
        current=entry['states'][k].astype(np.float64)
        before=entry['states'][0].astype(np.float64)
        delta=current-before
        xyz=(delta if movement else current)[:,list(axes)]
        norm=entry['norm'][k]
        arrays=state.get('arrays',{})
        metrics=entry['metrics']
        if metrics is None:
            for key in ('confidence','iou'):
                if key not in arrays or arrays[key].shape!=(1500,501):
                    raise ValueError('Missing matching original-run metric arrays')
            metrics={key:arrays[key][image] for key in ('confidence','iou')}
        confidence=float(metrics['confidence'][step]);iou=float(metrics['iou'][step])
        if not all(np.isfinite(v) and 0<=v<=1 for v in (confidence,iou)):
            raise ValueError('Nonfinite/out-of-range actual original metrics')
        both=None
        if all(key in arrays and arrays[key].shape==(1500,501) for key in ('confidence','iou')):
            for key in ('confidence','iou'):
                if metrics is not None and not np.isclose(metrics[key][step],arrays[key][image,step],rtol=0,atol=1e-7):
                    raise ValueError('Original NPZ/exported metric provenance mismatch')
            both=int(((arrays['confidence'][:,step]>=.9)&(arrays['iou'][:,step]>=.9)).sum())
        bounds=entry['movement_bounds' if movement else 'natural_bounds'][list(axes)]
        return {'xyz':xyz.tolist(),'norm':norm.tolist(),'tokens':list(range(600)),'images':[image]*600,
            'indices':list(range(image*600,(image+1)*600)),'heatmap':norm.reshape(20,30).tolist(),
            'bounds':bounds.tolist(),'norm_scale':entry['norm_scale'],'heat_scale':entry['norm_scale'],
            'norm_median':float(np.median(norm)),'norm_max':float(norm.max()),
            'count':600,'total':600,'image':image,'step':step,'mode':'image','space':'raw_channels',
            'channel_axes':list(axes),'actual_steps':list(STEPS),'interpolation':False,
            'source_sha256':record['sha256'],
            'confidence':confidence,'iou':iou,'both90':both,'selected_both90':int(confidence>=.9 and iou>=.9),
            'selected_image_count':1,'lines':[],'bar_tokens':[],'bar_norm':[],
            'profile':{'patch':patch,'row':patch//30,'col':patch%30,'before':before[patch].tolist(),
                       'current':current[patch].tolist(),'delta':delta[patch].tolist()},
            'metadata':{'source_origin':'original_full768_npz','source_host':SSH_HOST,
                'source_path':f'{REMOTE_LATENTS}/{image:06d}.npz','source_npz_sha256':record['sha256'],
                'source_bytes':record['bytes'],'source_freeze_sha256':freeze_sha,'source_tree_sha256':tree_sha,
                'image_name':row.get('image_name'),'checkpoint_sha256':row.get('checkpoint_sha256'),
                'metric_origin':'original_npz_telemetry' if entry['metrics'] is not None else 'original_run_projection_arrays',
                'bounds_scope':'all 600 patches of this image at original steps 0,100,500; axes unchanged',
                'norm_definition':'L2(z(step)-z(0)) in original full 768D','downloads_persisted':False}}


RAW=RawSpace()


def request(state,query):
    if 'terrain' in query:
        from global_terrain import request as density_request
        return density_request(query)
    allowed={'image','step','patch','axes','movement'}
    if set(query)-allowed or 'image' not in query or any(len(v)!=1 for v in query.values()):
        raise ValueError('Expected single raw image/step/patch/axes/movement parameters')
    def integer(key,default):
        value=query.get(key,[default])[0]
        if not isinstance(value,str) or not re.fullmatch(r'0|[1-9][0-9]{0,3}',value):
            raise ValueError('Invalid integer '+key)
        return int(value)
    raw_axes=query.get('axes',['0,1,2'])[0]
    if not isinstance(raw_axes,str) or not re.fullmatch(r'(?:0|[1-9][0-9]{0,2})(?:,(?:0|[1-9][0-9]{0,2})){2}',raw_axes):
        raise ValueError('axes must be three comma-separated original channel indices')
    movement=query.get('movement',['0'])[0]
    if movement not in ('0','1'):raise ValueError('movement must be 0 or 1')
    image=integer('image','0');step=integer('step','0');patch=integer('patch','0')
    axes=tuple(int(v) for v in raw_axes.split(','))
    validate(image,step,patch,axes,movement=='1')
    return RAW.frame(state,image,step,patch,axes,movement=='1')
