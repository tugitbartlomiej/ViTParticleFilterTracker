"""Serwer wersji GPU tańca cząsteczek K1000.

Serwer NIE liczy ruchu. Podaje surowe tablice z zamrożonego projection.npz
(bajty little-endian, kolejność C, bez przestawiania osi) oraz pliki strony.
Pozycje w czasie, przesunięcia, prędkości, granice, mapy, mediany i krzywe
liczą compute shadery WGSL w przeglądarce. Numpy służy tu wyłącznie jako
niezależny sędzia w trybie weryfikacji (/api/verify/*).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

FAMILY = 'vit_k1000_taniec_czasteczek_gpu_20261009'
SOURCE_FAMILY = 'vit_k1000_taniec_czasteczek_20261009'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / 'outputs' / SOURCE_FAMILY
DATA = SRC / 'data'
OUT = ROOT / 'outputs' / FAMILY
NPZ_SHA256 = '95d61d173ddb404abf7deadde10f112a3e3fc9d12fa3e63abb74a3ca7321f36a'
N, TOKENS, STATES, HORIZON = 1500, 600, 3, 500
MILESTONES = (0, 100, 500)

# Tablice wysyłane do GPU: nazwa -> (klucz NPZ, dtype docelowy, oczekiwany kształt).
ARRAYS = {
    'positions': ('positions', '<f4', (N, STATES, TOKENS, 3)),
    'movement_norm': ('movement_norm', '<f4', (N, STATES, TOKENS)),
    'control_positions': ('control_positions', '<f4', (16, HORIZON + 1, 8, 3)),
    'control_norm': ('control_norm', '<f4', (16, HORIZON + 1, 8)),
    'control_step_norm': ('control_step_norm', '<f4', (16, HORIZON + 1, 8)),
    'confidence': ('confidence', '<f4', (N, HORIZON + 1)),
    'iou': ('iou', '<f4', (N, HORIZON + 1)),
    'control_images': ('control_images', '<u4', (16,)),
    'control_tokens': ('control_tokens', '<u4', (16, 8)),
}

LOCK = threading.Lock()
STATE: dict = {}
BLOBS: dict[str, bytes] = {}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load() -> dict:
    with LOCK:
        if STATE:
            return STATE
        npz = DATA / 'projection.npz'
        digest = file_sha(npz)
        if digest != NPZ_SHA256:
            raise RuntimeError(f'projection.npz zmienił się: {digest}')
        transfer = json.loads((DATA / 'LOCAL_TRANSFER.json').read_text(encoding='utf-8'))
        if not transfer.get('all_original_remote_bundle_hashes_match'):
            raise RuntimeError('Transfer danych nie jest zweryfikowany')
        with np.load(npz, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        arrays, binary = {}, {}
        for name, (key, dtype, shape) in ARRAYS.items():
            a = raw[key]
            if a.shape != shape:
                raise ValueError(f'{key}: {a.shape} != {shape}')
            if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
                raise ValueError(f'{key}: wartości nieskończone')
            if dtype == '<u4' and (a < 0).any():
                raise ValueError(f'{key}: ujemne indeksy')
            cast = np.ascontiguousarray(a.astype(dtype, copy=False))
            if not np.array_equal(cast, a):
                raise ValueError(f'{key}: rzutowanie zmieniło wartości')
            arrays[name] = cast
            binary[name] = cast.tobytes(order='C')
        meta = json.loads((DATA / 'metadata.json').read_text(encoding='utf-8'))
        images = [{'frame_id': r.get('frame_id'), 'first_success_step': r.get('first_success_step'),
                   'after_step': r.get('after_step')} for r in meta['images']]
        STATE.update(arrays=arrays, binary=binary, images=images,
                     pca_variance=meta['fit']['explained_variance_ratio'], npz_sha256=digest)
        return STATE


def meta_payload() -> dict:
    s = load()
    return {
        'ready': True, 'family': FAMILY, 'source_family': SOURCE_FAMILY,
        'npz_sha256': s['npz_sha256'], 'images': s['images'],
        'control_images': s['arrays']['control_images'].tolist(),
        'control_tokens': s['arrays']['control_tokens'].tolist(),
        'pca_variance': s['pca_variance'], 'milestones': list(MILESTONES),
        'arrays': {k: {'shape': list(v.shape), 'dtype': v.dtype.str, 'bytes': v.nbytes,
                       'sha256': hashlib.sha256(s['binary'][k]).hexdigest()}
                   for k, v in s['arrays'].items()},
    }


# ---------------------------------------------------------------- sędzia numpy
def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f4').tobytes()).hexdigest()


def _kin_milestone(a, image, ka, kb, s, movement, color):
    """Referencja kernela kin_milestone (float32, te same reguły końców odcinka)."""
    P, Nm = a['positions'], a['movement_norm']
    sel = slice(None) if image < 0 else slice(image, image + 1)
    pa, pb, p0 = P[sel, ka], P[sel, kb], P[sel, 0]
    if s <= 0:
        x = pa.copy()
    elif s >= 1:
        x = pb.copy()
    else:
        x = (pa + (pb - pa) * np.float32(s)).astype(np.float32)
    if movement:
        x = x - p0
    if color == 0:
        na, nb = Nm[sel, ka], Nm[sel, kb]
        c = na if s <= 0 else nb if s >= 1 else (na + (nb - na) * np.float32(s))
    elif color == 1:
        d = (pb - pa).astype(np.float64)
        c = np.sqrt((d * d).sum(-1)) / (MILESTONES[kb] - MILESTONES[ka])
    else:
        d = (x if movement else x - p0).astype(np.float64)
        c = np.sqrt((d * d).sum(-1))
    out = np.empty(x.shape[:-1] + (4,), np.float32)
    out[..., :3] = x
    out[..., 3] = c
    return out.reshape(-1, 4)


def _kin_control(a, panel, step, movement, color):
    CP, CN, CS = a['control_positions'], a['control_norm'], a['control_step_norm']
    sel = slice(None) if panel < 0 else slice(panel, panel + 1)
    x = CP[sel, step].copy()
    if movement:
        x = x - CP[sel, 0]
    if color == 0:
        c = CN[sel, step]
    elif color == 1:
        c = CS[sel, step]
    elif color == 2:
        d = (x if movement else x - CP[sel, 0]).astype(np.float64)
        c = np.sqrt((d * d).sum(-1))
    else:
        c = np.broadcast_to(np.arange(8, dtype=np.float32), x.shape[:-1])
    out = np.empty(x.shape[:-1] + (4,), np.float32)
    out[..., :3] = x
    out[..., 3] = c
    return out.reshape(-1, 4)


def _trail(a, panel, step, movement, length=40):
    CP = a['control_positions']
    sel = slice(None) if panel < 0 else slice(panel, panel + 1)
    steps = np.clip(np.arange(step - length, step + 1), 0, None)
    x = CP[sel][:, steps]                       # slot, h, j, 3
    if movement:
        x = x - CP[sel, 0][:, None]
    x = x.transpose(0, 2, 1, 3).reshape(-1, length + 1, 3)  # (slot*8+j), h, 3
    out = np.zeros(x.shape[:-1] + (4,), np.float32)
    out[..., :3] = x
    out[..., 3] = (np.arange(length + 1, dtype=np.float32) / length)[None]
    return out.reshape(-1, 4)


def _stats_milestone(a, image, movement):
    P, Nm = a['positions'], a['movement_norm']
    sel = slice(None) if image < 0 else slice(image, image + 1)
    x = P[sel] - P[sel, :1] if movement else P[sel]
    flat = x.reshape(-1, 3)
    return {'lo': flat.min(0).tolist(), 'hi': flat.max(0).tolist(),
            'norm_max': float(Nm[sel].max())}


def _stats_control(a, panel, movement):
    CP, CN = a['control_positions'], a['control_norm']
    sel = slice(None) if panel < 0 else slice(panel, panel + 1)
    x = CP[sel] - CP[sel, :1] if movement else CP[sel]
    flat = x.reshape(-1, 3)
    return {'lo': flat.min(0).tolist(), 'hi': flat.max(0).tolist(), 'norm_max': float(CN[sel].max())}


def _order_stats(v):
    v = np.sort(np.asarray(v, np.float32).ravel())
    n = len(v)
    return {'n': n, 'low': float(v[(n - 1) // 2]), 'high': float(v[n // 2]),
            'median': float(np.median(v)), 'max': float(v[-1])}


def _curves(a, kind, image):
    idx = (np.arange(N) if kind == 0 else a['control_images'].astype(int) if kind == 1
           else np.array([image]))
    c, q = a['confidence'][idx], a['iou'][idx]
    both = (c >= np.float32(.9)) & (q >= np.float32(.9))
    return {'confidence': c.astype(np.float64).mean(0), 'iou': q.astype(np.float64).mean(0),
            'both_count': both.sum(0)}


def _cmp(got, expected, rtol, atol):
    got = np.asarray(got, np.float64)
    expected = np.asarray(expected, np.float64)
    if got.shape != expected.shape:
        return {'passed': False, 'reason': f'kształt {got.shape} != {expected.shape}'}
    err = np.abs(got - expected)
    ok = bool(np.all(err <= atol + rtol * np.abs(expected)))
    return {'passed': ok, 'max_abs_error': float(err.max()) if err.size else 0.0,
            'max_rel_error': float((err / np.maximum(np.abs(expected), 1e-30)).max()) if err.size else 0.0,
            'rtol': rtol, 'atol': atol, 'n': int(err.size)}


def judge(report: dict) -> dict:
    """Porównuje odczyty z GPU z niezależną referencją numpy."""
    a = load()['arrays']
    results = []
    for case in report['cases']:
        kind, p = case['kind'], case['params']
        r = {'name': case['name'], 'kind': kind, 'params': p}
        try:
            if kind == 'kin_milestone':
                ref = _kin_milestone(a, p['image'], p['ka'], p['kb'], p['s'], p['movement'], p['color'])
                blob = np.frombuffer(BLOBS[case['blob']], '<f4').reshape(-1, 4)
                if p['bitwise']:
                    same = bool(np.array_equal(blob, ref))
                    hashes = case.get('sha256') == _sha(ref)
                    r.update(passed=same and hashes, compare='bitwise + sha256',
                             gpu_sha256=case.get('sha256'), numpy_sha256=_sha(ref), n=len(ref))
                else:
                    r.update(compare='tolerance xyz', **_cmp(blob[:, :3], ref[:, :3], 2e-6, 2e-5))
                    col = _cmp(blob[:, 3], ref[:, 3], 2e-6, 1e-6)
                    r['color'] = col
                    r['passed'] = r['passed'] and col['passed']
            elif kind == 'kin_control':
                ref = _kin_control(a, p['panel'], p['step'], p['movement'], p['color'])
                blob = np.frombuffer(BLOBS[case['blob']], '<f4').reshape(-1, 4)
                if p['color'] == 2:
                    col = _cmp(blob[:, 3], ref[:, 3], 2e-6, 1e-6)
                    r.update(compare='xyz bitwise + |Δ| tolerance', color=col,
                             passed=bool(np.array_equal(blob[:, :3], ref[:, :3])) and col['passed'])
                else:
                    r.update(compare='bitwise', passed=bool(np.array_equal(blob, ref)), n=len(ref))
            elif kind == 'trail':
                ref = _trail(a, p['panel'], p['step'], p['movement'])
                blob = np.frombuffer(BLOBS[case['blob']], '<f4').reshape(-1, 4)
                r.update(compare='bitwise xyz', passed=bool(np.array_equal(blob[:, :3], ref[:, :3])), n=len(ref))
            elif kind in ('stats_milestone', 'stats_control'):
                ref = (_stats_milestone(a, p['image'], p['movement']) if kind == 'stats_milestone'
                       else _stats_control(a, p['panel'], p['movement']))
                g = case['result']
                ok = (np.array_equal(np.float32(g['lo']), np.float32(ref['lo'])) and
                      np.array_equal(np.float32(g['hi']), np.float32(ref['hi'])) and
                      np.float32(g['norm_max']) == np.float32(ref['norm_max']))
                r.update(compare='bitwise min/max', passed=bool(ok), gpu=g, numpy=ref)
            elif kind == 'heat':
                Nm = a['movement_norm']
                ref = (Nm.astype(np.float64).mean(0) if p['image'] < 0 else Nm[p['image']].astype(np.float64))
                r.update(compare='tolerance', **_cmp(np.asarray(case['result']).reshape(3, 600), ref, 1e-5, 1e-9))
            elif kind == 'order_stats':
                if p['source'] == 'milestone':
                    Nm = a['movement_norm']
                    v = Nm[:, p['state']] if p['image'] < 0 else Nm[p['image'], p['state']]
                else:
                    CN = a['control_norm']
                    v = CN[:, p['step']] if p['panel'] < 0 else CN[p['panel'], p['step']]
                ref = _order_stats(v)
                g = case['result']
                ok = (g['n'] == ref['n'] and np.float32(g['low']) == np.float32(ref['low']) and
                      np.float32(g['high']) == np.float32(ref['high']) and np.float32(g['max']) == np.float32(ref['max']) and
                      abs(g['median'] - ref['median']) <= 1e-6 * max(abs(ref['median']), 1e-30))
                r.update(compare='bitwise order statistics', passed=bool(ok), gpu=g, numpy=ref)
            elif kind == 'curves':
                ref = _curves(a, p['kind'], p['image'])
                g = np.asarray(case['result']).reshape(HORIZON + 1, 4)
                c1 = _cmp(g[:, 0], ref['confidence'], 1e-5, 1e-7)
                c2 = _cmp(g[:, 1], ref['iou'], 1e-5, 1e-7)
                counts = bool(np.array_equal(g[:, 3].astype(np.int64), ref['both_count']))
                r.update(compare='tolerance + exact counts', passed=c1['passed'] and c2['passed'] and counts,
                         confidence=c1, iou=c2, both_counts_exact=counts)
            elif kind == 'bars':
                CN = a['control_norm']
                v = CN[:, p['step']] if p['panel'] < 0 else CN[p['panel'], p['step']][None]
                r.update(compare='tolerance', **_cmp(case['result'], v.astype(np.float64).mean(0), 1e-6, 1e-9))
            elif kind == 'pick':
                r.update(compare='identyfikator wybranego punktu', passed=case['result']['id'] == p['expected_id'],
                         gpu=case['result'])
            else:
                r.update(passed=False, reason='nieznany przypadek')
        except Exception as e:  # noqa: BLE001 - raportujemy każdy błąd sędziego
            r.update(passed=False, reason=f'{type(e).__name__}: {e}')
        results.append(r)
    passed = bool(results) and all(r['passed'] for r in results)
    out = {'status': 'passed' if passed else 'failed', 'family': FAMILY,
           'judge': 'numpy na CPU, niezależnie od kerneli WGSL',
           'source_npz_sha256': load()['npz_sha256'], 'cases_total': len(results),
           'cases_passed': sum(r['passed'] for r in results), 'cases': results,
           'gpu': report.get('gpu'), 'timings': report.get('timings'),
           'verified_utc': datetime.now(timezone.utc).isoformat()}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / 'WERYFIKACJA_GPU.json.tmp'
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(OUT / 'WERYFIKACJA_GPU.json')
    return out


STATIC = {
    '/': (HERE / 'web' / 'index.html', 'text/html; charset=utf-8'),
    '/app.js': (HERE / 'web' / 'app.js', 'text/javascript; charset=utf-8'),
    '/gpu-engine.js': (HERE / 'web' / 'gpu-engine.js', 'text/javascript; charset=utf-8'),
    '/shaders.js': (HERE / 'web' / 'shaders.js', 'text/javascript; charset=utf-8'),
    '/selftest.js': (HERE / 'web' / 'selftest.js', 'text/javascript; charset=utf-8'),
    '/plotly.min.js': (SRC / 'plotly.min.js', 'text/javascript; charset=utf-8'),
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, mime: str, code: int = 200):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, code=200):
        self._send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                   'application/json; charset=utf-8', code)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            if u.path == '/api/meta':
                return self._json(meta_payload())
            if u.path.startswith('/api/array/'):
                name = u.path.rsplit('/', 1)[1]
                if name not in ARRAYS:
                    return self.send_error(404)
                return self._send(load()['binary'][name], 'application/octet-stream')
            if u.path in STATIC:
                path, mime = STATIC[u.path]
                return self._send(path.read_bytes(), mime)
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self.send_error(500, str(e)[:200])

    def do_POST(self):
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            length = int(self.headers.get('Content-Length', '0'))
            if length > 64 << 20:
                return self.send_error(413)
            body = self.rfile.read(length)
            if u.path == '/api/verify/blob':
                name = q['name'][0]
                BLOBS[name] = body
                return self._json({'stored': name, 'bytes': len(body)})
            if u.path == '/api/snapshot':
                name = ''.join(ch for ch in q.get('name', ['klatka'])[0] if ch.isalnum() or ch in '_-')[:80] or 'klatka'
                if not body.startswith(b'\x89PNG\r\n\x1a\n'):
                    return self.send_error(400, 'Oczekiwano PNG')
                folder = OUT / 'klatki'
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f'{name}.png').write_bytes(body)
                return self._json({'saved': f'klatki/{name}.png', 'bytes': len(body)})
            if u.path == '/api/verify/finish':
                result = judge(json.loads(body))
                BLOBS.clear()
                return self._json(result)
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self.send_error(500, str(e)[:200])

    def log_message(self, format, *args):  # noqa: A002 - sygnatura klasy bazowej
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8775)
    args = p.parse_args()
    t = time.perf_counter()
    load()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'SERVER.json').write_text(json.dumps({'pid': os.getpid(), 'url': f'http://127.0.0.1:{args.port}/',
                                                 'source': str(Path(__file__).resolve())}), encoding='utf-8')
    print(f'Dane gotowe w {time.perf_counter() - t:.1f} s. Podgląd GPU: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
