from ocena import matching

def test_one_prediction_cannot_cover_two_objects():
    assert matching([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]])==dict(tp=1,fp=0,fn=1,objects=2)

def test_duplicates_are_false_positives():
    assert matching([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]])==dict(tp=1,fp=1,fn=0,objects=1)

def test_empty_image():
    assert matching([],[])==dict(tp=0,fp=0,fn=0,objects=0)


def test_missing_test_split_is_unavailable_not_perfect_score(tmp_path):
    from types import SimpleNamespace
    from dane import write_json, read_json
    from ocena import run
    write_json(tmp_path/'data/test/_annotations.coco.json',dict(images=[],annotations=[]))
    run(SimpleNamespace(dataset=str(tmp_path/'data'),out=str(tmp_path/'out'),split='test'))
    report=read_json(tmp_path/'out/REPORT.json')
    assert report['state']=='unavailable'
    assert report['metrics'] is None and report['counts'] is None
