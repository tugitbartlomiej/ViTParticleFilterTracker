"""CIFAR-10 z HuggingFace (`uoft-cs/cifar10`, plain_text, parquet z PNG) → jeden plik cifar10_uint8.npz:
X_train (50000, 32, 32, 3) uint8, y_train (50000,), X_test (10000, 32, 32, 3), y_test (10000,), klasy (10,).
Oryginalne archiwum z cs.toronto.edu pobiera się bardzo wolno (~40 KB/s); jeśli jest pełne, `--sprawdz TAR` porównuje piksele
i etykiety z nim (kolejność obrazów może się różnić, więc porównanie po odciskach obrazów).
Uruchomienie: python przygotuj_cifar10.py <katalog z parquet> [--sprawdz cifar-10-python.tar.gz]"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

KLASY = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def wczytaj(p):
    t = pq.read_table(p).to_pydict()
    kol_img = 'img' if 'img' in t else 'image'
    X = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB')) for d in t[kol_img]])
    return X.astype(np.uint8), np.asarray(t['label'], np.int64)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('katalog')
    a.add_argument('--sprawdz', default=None)
    a = a.parse_args()
    k = Path(a.katalog)
    Xtr, ytr = wczytaj(k / 'train-00000-of-00001.parquet')
    Xte, yte = wczytaj(k / 'test-00000-of-00001.parquet')
    assert Xtr.shape == (50000, 32, 32, 3) and Xte.shape == (10000, 32, 32, 3), (Xtr.shape, Xte.shape)
    assert np.bincount(ytr).tolist() == [5000] * 10 and np.bincount(yte).tolist() == [1000] * 10
    out = k / 'cifar10_uint8.npz'
    np.savez(out, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, klasy=np.array(KLASY))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    opis = dict(zrodlo='https://huggingface.co/datasets/uoft-cs/cifar10 (plain_text)', train=list(Xtr.shape), test=list(Xte.shape),
                klasy=KLASY, sha256=sha)
    if a.sprawdz:
        import pickle
        import tarfile
        with tarfile.open(a.sprawdz) as t:
            d = pickle.load(t.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')
        Xo = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); yo = np.asarray(d[b'labels'])
        odc = lambda X: [hashlib.md5(x.tobytes()).hexdigest() for x in X]
        mapa = {h: int(y) for h, y in zip(odc(Xte), yte)}
        zgodne = sum(mapa.get(h) == int(y) for h, y in zip(odc(Xo), yo))
        opis['zgodnosc_z_oryginalem_test'] = f'{zgodne}/{len(yo)}'
    (k / 'cifar10_uint8.json').write_text(json.dumps(opis, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(opis, ensure_ascii=False))


if __name__ == '__main__':
    main()
