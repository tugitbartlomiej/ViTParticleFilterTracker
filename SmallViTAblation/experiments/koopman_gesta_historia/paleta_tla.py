"""Fixed 256-colour diagnostic expansion; does not alter the running probe."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
from dane import read_json, write_json, sha256

def palette():
    # A deterministic RGB cube, including black, white and saturated primaries.
    return np.array(list(itertools.product(np.rint(np.linspace(0,255,8)).astype(int),
        np.rint(np.linspace(0,255,8)).astype(int),np.rint(np.linspace(0,255,4)).astype(int))),dtype=np.uint8)

def compose(rgb,mask,color):
    alpha=mask.astype(float)[...,None]/255
    return np.rint(rgb*alpha+np.asarray(color)*(1-alpha)).astype(np.uint8)

def run(source,out):
    source=Path(source);out=Path(out)
    originals=[r for r in read_json(source) if r['variant']=='original']
    assert originals and all(r['split']=='train' for r in originals)
    colors=palette(); assert colors.shape==(256,3) and len(np.unique(colors,axis=0))==256
    out.mkdir(parents=True,exist_ok=False)
    (out/'images').mkdir();(out/'masks').mkdir()
    records=[]
    for i,r in enumerate(originals):
        ip=source.parent/r['image'];mp=source.parent/r['mask']
        assert sha256(ip)==r['image_sha256'] and sha256(mp)==r['mask_sha256']
        rgb=np.array(Image.open(ip).convert('RGB')); mask=np.array(Image.open(mp).convert('L'))
        maskname=f'masks/{i:02d}.png';Image.fromarray(mask).save(out/maskname)
        background=1-mask.astype(float)/255
        for j,color in enumerate(colors):
            image=compose(rgb,mask,color)
            assert np.array_equal(image[mask==255],rgb[mask==255])
            assert np.all(image[mask==0]==color)
            name=f'images/{i:02d}_rgb_{j:03d}.png';Image.fromarray(image).save(out/name)
            labels=list(r['labels'])
            luminance=image.astype(float)@np.array([.2126,.7152,.0722])/255
            labels[r['label_names'].index('backgroundbrightness')]=float((luminance*background).sum()/background.sum())
            rec=dict(r,image=name,mask=maskname,variant=f'background_rgb_{j:03d}',labels=labels,
                background_rgb=color.tolist(),palette_index=j,photometric_scope='background_only_alpha_composite',
                image_sha256=sha256(out/name),mask_sha256=sha256(out/maskname))
            records.append(rec)
    write_json(out/'manifest.json',records)
    write_json(out/'REPORT.json',dict(state='completed',palette=colors.tolist(),colors=256,
        images=len(records),source_groups=len(originals),source_manifest_sha256=sha256(source),
        foreground_opaque_pixels_unchanged=True,soft_boundary='alpha blending from original mask',
        running_training_probe_modified=False,latent_extraction_done=False,
        note='Background sensitivity diagnostic, not proof of causal identity. Uniform backgrounds are synthetic.'))
    swatch=np.zeros((256,256,3),dtype=np.uint8)
    for j,c in enumerate(colors):swatch[(j//16)*16:(j//16+1)*16,(j%16)*16:(j%16+1)*16]=c
    Image.fromarray(swatch).save(out/'PALETA.png')
    print(f'Created {len(records)} images, {len(originals)} sources, 256 colours')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
