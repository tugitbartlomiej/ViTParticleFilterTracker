import warnings
import dane
rows=[dict(zip_path='unused.zip',member=f'TR_{i:04d}_S1_P03.mp4',video_id=f'TR_{i:04d}_S1_P03',split='train',annotation_zip='unused.zip') for i in (7,8)]
def fake(row, cache):
    if row['video_id']=='TR_0007_S1_P03':
        raise dane.FrameCountMismatch('MP4/CSV mismatch')
    yield {'film':row['video_id']}
dane.iter_video=fake
with warnings.catch_warnings(record=True) as caught:
    output=list(dane.VideoFrames(rows,repeat=False))
assert output==[{'film':'TR_0008_S1_P03'}]
assert len(caught)==1 and 'film_rejected_before_training' in str(caught[0].message)
def broken(row, cache):
    raise RuntimeError('unexpected decode failure')
    yield
dane.iter_video=broken
try:
    list(dane.VideoFrames(rows,repeat=False))
except RuntimeError as e:
    assert str(e)=='unexpected decode failure'
else:
    raise AssertionError('Other failures must not be silently ignored')
print('PASS: mismatched film rejected; unexpected errors propagate')
