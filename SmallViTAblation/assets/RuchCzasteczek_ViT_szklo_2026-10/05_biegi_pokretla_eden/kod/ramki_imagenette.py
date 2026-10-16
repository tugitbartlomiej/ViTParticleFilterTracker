"""Ramki obiektów ImageNet (ILSVRC2012_bbox_train_v2.tar.gz, ILSVRC2012_bbox_val_v3.tgz, MD5 zgodne ze stroną image-net.org)
dopasowane do plików Imagenette (imagenette2-320) po nazwie pliku. Archiwa czytane strumieniowo, bez rozpakowywania XML
(kwota plików na Edenie). Ramki w XML są w rozmiarze, w jakim obraz pokazano anotatorowi (pole <size>); przeliczamy je na
piksele pliku imagenette2-320 (krótszy bok 320) skalą osobno w X i Y, a także na środkowe wycięcie 320×320 używane w treningu
(trening_klasyfikacja.wczytaj_imagenette), z obcięciem do kadru.
Wyjście: imagenette_bbox.json {plik: {split, klasa, rozmiar_xml, rozmiar_pliku, ramki_xml, ramki_plik, ramki_wyciecie320}} i podsumowanie.
Uruchomienie: python ramki_imagenette.py <imagenette2-320> <katalog z archiwami> <wyjście.json>"""
import json
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROZMIAR = 320


def parsuj(dane):
    r = ET.fromstring(dane)
    s = r.find('size')
    w, h = int(s.find('width').text), int(s.find('height').text)
    ramki = []
    for o in r.findall('object'):
        b = o.find('bndbox')
        ramki.append(dict(wnid=o.find('name').text, xyxy=[float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
    return w, h, ramki


def main():
    root, arch, wy = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    pliki = {}
    for split in ('train', 'val'):
        for p in sorted((root / split).glob('*/*.JPEG')):
            pliki[p.stem] = (split, p.parent.name, p)
    klasy = sorted({k for _, k, _ in pliki.values()})
    xml = {}
    for a in ('ILSVRC2012_bbox_train_v2.tar.gz', 'ILSVRC2012_bbox_val_v3.tgz'):
        with tarfile.open(arch / a, 'r:gz') as t:
            for m in t:
                if not m.isfile() or not m.name.endswith('.xml'):
                    continue
                stem = Path(m.name).stem
                if stem in pliki:
                    xml[stem] = t.extractfile(m).read()
    wynik, bez = {}, {'train': 0, 'val': 0}
    for stem, (split, kl, p) in pliki.items():
        if stem not in xml:
            bez[split] += 1
            continue
        wx, hx, ramki = parsuj(xml[stem])
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        sx, sy = w / wx, h / hx
        s = ROZMIAR / min(h, w)
        W, Hh = (max(ROZMIAR, round(w * s)), max(ROZMIAR, round(h * s))) if abs(s - 1) > 1e-6 else (w, h)
        x0, y0 = (W - ROZMIAR) // 2, (Hh - ROZMIAR) // 2
        rp, rw = [], []
        for r in ramki:
            a, b, c, d = r['xyxy']
            f = [a * sx, b * sy, c * sx, d * sy]
            rp.append(dict(wnid=r['wnid'], xyxy=[round(v, 1) for v in f]))
            g = [f[0] * W / w - x0, f[1] * Hh / h - y0, f[2] * W / w - x0, f[3] * Hh / h - y0]
            g = [min(max(g[0], 0), ROZMIAR), min(max(g[1], 0), ROZMIAR), min(max(g[2], 0), ROZMIAR), min(max(g[3], 0), ROZMIAR)]
            if g[2] - g[0] >= 2 and g[3] - g[1] >= 2:
                rw.append(dict(wnid=r['wnid'], xyxy=[round(v, 1) for v in g]))
        wynik[str(p.relative_to(root)).replace('\\', '/')] = dict(split=split, klasa=kl, rozmiar_xml=[wx, hx], rozmiar_pliku=[w, h],
                                                                    ramki_xml=ramki, ramki_plik=rp, ramki_wyciecie320=rw)
    pods = {}
    for split in ('train', 'val'):
        for k in klasy:
            n = sum(1 for v in pliki.values() if v[0] == split and v[1] == k)
            z = [v for v in wynik.values() if v['split'] == split and v['klasa'] == k]
            pods[f'{split}/{k}'] = dict(obrazow=n, z_ramka=len(z), ramek=sum(len(v['ramki_plik']) for v in z),
                                        w_wycieciu=sum(1 for v in z if v['ramki_wyciecie320']))
    meta = dict(opis=__doc__.split('\n')[0], zrodlo='https://image-net.org/data/ILSVRC/2012/ (ILSVRC2012_bbox_train_v2.tar.gz MD5 9271167e2176350e65cfe4e546f14b17, '
                'ILSVRC2012_bbox_val_v3.tgz MD5 f4cd18b5ea29fe6bbea62ec9c20d80f0)', obrazow=len(pliki), z_ramka=len(wynik), bez_ramki=bez,
                wspolrzedne='xyxy w pikselach: ramki_xml w rozmiarze z XML, ramki_plik w pliku imagenette2-320, ramki_wyciecie320 w środkowym wycięciu 320×320',
                podsumowanie=pods)
    json.dump(dict(meta=meta, obrazy=wynik), open(wy, 'w', encoding='utf-8'), ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k != 'podsumowanie'}, ensure_ascii=False))
    for k, v in pods.items():
        print(k, v)


if __name__ == '__main__':
    main()
