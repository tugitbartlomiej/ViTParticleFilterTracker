import numpy as np

def iou(b, g):
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)

def average_precision(detekcje, ramki_gt, iou_threshold):
    """AP jednej klasy; zerowe pozycje dopełniające dekoder nie są detekcjami."""
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        for b, s in zip(bx, sc):
            if not np.isfinite(s) or not np.isfinite(b).all():
                raise ValueError("Niefinitywna detekcja")
            if float(s) <= 0.0:
                continue
            wszystkie.append((float(s), i, b))
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (s, i, b) in enumerate(wszystkie):
        g = ramki_gt[i]
        if len(g):
            v = np.array([iou(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= float(iou_threshold):
                tp[j] = 1; uzyte[i][k] = True
    if n_gt == 0 or len(wszystkie) == 0:
        return 0.0
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        ap += prec[k] if k < len(prec) else 0.0
    return float(ap / 101)


def ap50(detekcje, ramki_gt):
    """AP@0,5 jednej klasy, interpolacja 101-punktowa jak w COCO."""
    return average_precision(detekcje, ramki_gt, 0.5)


def detection_map(detekcje, ramki_gt):
    """AP50, AP75 i mAP po progach IoU 0,50:0,05:0,95 dla jednej klasy."""
    thresholds = np.linspace(0.50, 0.95, 10)
    values = np.asarray(
        [average_precision(detekcje, ramki_gt, threshold) for threshold in thresholds],
        dtype=np.float64,
    )
    return {
        "AP50": float(values[0]),
        "AP75": float(values[5]),
        "mAP50_95": float(values.mean()),
        "AP_by_IoU": {
            f"{threshold:.2f}": float(value)
            for threshold, value in zip(thresholds, values)
        },
        "matching": "global score ranking; one-to-one per image; 101-point interpolation",
        "decoder_padding": "detections with score <= 0 are excluded",
    }
