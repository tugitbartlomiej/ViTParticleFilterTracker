import numpy as np

def iou(b, g):
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)

def ap50(detekcje, ramki_gt):
    """AP@0,5 jednej klasy, interpolacja 101-punktowa jak w COCO."""
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        for b, s in zip(bx, sc):
            wszystkie.append((float(s), i, b))
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (s, i, b) in enumerate(wszystkie):
        g = ramki_gt[i]
        if len(g):
            v = np.array([iou(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= .5:
                tp[j] = 1; uzyte[i][k] = True
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101