"""Przygotowuje niezmienny pakiet 400 obrazów i pochodzenie danych."""
from pathlib import Path
import json,hashlib,shutil,tarfile
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments')
OUT=HERE.parents[1]/'outputs/vit_optymalizacja_latentu128'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pack=OUT/'input';pack.mkdir(parents=True,exist_ok=True)
    data=BASE/'maly_vit_test_orientacji_narzedzia_400'
    rows=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    split=json.loads((BASE/'maly_vit_tozsamosc_czasteczki_pary_400/PODZIAL.json').read_text(encoding='utf-8'))
    dest=[]
    for i,row in enumerate(rows):
        src=data/row['image_path'];assert sha(src)==row['image_sha256']
        name=f'images/{i:04d}{src.suffix}';(pack/'images').mkdir(exist_ok=True)
        shutil.copy2(src,pack/name)
        rec=split['records'][i];assert rec['frame_id']==row['frame_id']
        dest.append(dict(index=i,image=name,film=row['video_id'],frame_id=row['frame_id'],split=rec['split'],bbox_xywh=row['bbox_source_480'],sha256=sha(src)))
    (pack/'manifest.json').write_text(json.dumps(dest,indent=2),encoding='utf-8')
    shutil.copy2(BASE/'maly_vit_tozsamosc90_bez_pca_langevin/KANALY_MODEL.npz',pack/'channels.npz')
    shutil.copy2(BASE/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k0500.pt',pack/'checkpoint.pt')
    assert len(np.load(pack/'channels.npz')['channels0'])==128
    sums={str(p.relative_to(pack)).replace('\\','/'):sha(p) for p in pack.rglob('*') if p.is_file() and p.name!='HASHES.json'}
    (pack/'HASHES.json').write_text(json.dumps(sums,indent=2),encoding='utf-8')
    with tarfile.open(OUT/'input.tar.gz','w:gz') as tf:tf.add(pack,arcname='input')
    print('package',OUT/'input.tar.gz',(OUT/'input.tar.gz').stat().st_size)
if __name__=='__main__':main()
