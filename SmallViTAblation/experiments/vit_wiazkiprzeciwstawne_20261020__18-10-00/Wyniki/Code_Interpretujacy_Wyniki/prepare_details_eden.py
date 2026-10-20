"""Read saved measurements; export small diagnostic arrays, without training."""
from pathlib import Path
import json, csv, hashlib, time
import numpy as np

SOURCE = Path("/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation/outputs/beam_survival_angles_training_20261020/run_v1")
DEST = Path(__file__).resolve().parents[1] / "Dane"
DEST.mkdir(parents=True, exist_ok=True)
cfg = json.loads((SOURCE / "config.json").read_text())
steps = cfg["steps"]
rows = list(csv.DictReader((SOURCE / "results/DETECTION.csv").open()))
frame = cfg["rows"][0]["id"]
token = int(next(r["token"] for r in rows if r["id"] == frame and int(r["step"]) == 55000))
metadata = dict(source=str(SOURCE), frame=frame, frame_index=0, fixed_token=token,
                token_row=token // 30, token_col=token % 30, steps=steps,
                selection="First frozen frame; token of its top detection at step55000. Fixed throughout. Diagnostic example, not representative sampling.",
                input_files={}, blocks=[])
out = {}
def track(path):
    metadata["input_files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
def unpack(x):
    return np.unpackbits(x, axis=-1, count=768).astype(bool)
def angles(v):
    v = v.astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-30)
    return np.degrees(np.arccos((v @ v.T).clip(-1, 1)))

for b in range(1, 7):
    t0 = time.monotonic()
    path = SOURCE / f"states/b{b}_ENDPOINT_STRENGTHS.npz"
    track(path)
    with np.load(path) as f:
        initial = f["initial"]
        final = f["best"]
    out[f"b{b}_strength0"] = initial[0, token].copy()
    out[f"b{b}_strength1"] = final[0, token].copy()
    bins = np.linspace(-12, 2, 141)
    out["strength_log10_edges"] = bins
    for tag, a in [("initial", initial), ("best", final)]:
        nonzero = a[a > 0]
        hist, _ = np.histogram(np.log10(nonzero), bins=bins)
        out[f"b{b}_strength_hist_{tag}"] = hist / a.size
        out[f"b{b}_strength_outside_{tag}"] = np.array([(a == 0).sum(), (nonzero < 1e-12).sum(), (nonzero > 100).sum(), a.size])
    del initial, final
    masks, signs, valid, vectors = [], [], [], []
    for step in steps:
        path = SOURCE / f"states/b{b}_k{step:05d}.npz"
        track(path)
        with np.load(path) as f:
            masks.append(unpack(f["active_0.05"][0, token]))
            signs.append(unpack(f["positive"][0, token]))
            valid.append(unpack(f["nonzero"][0, token]))
            vectors.append(f["V"])
            if step in [0, 55000]:
                out[f"b{b}_spatial_effective_{step}"] = f["effective"][0].reshape(20, 30)
                out[f"b{b}_spatial_active_{step}"] = unpack(f["active_0.05"][0]).sum(-1).reshape(20, 30)
    masks, signs, valid = map(np.array, (masks, signs, valid))
    out[f"b{b}_active"] = masks
    out[f"b{b}_positive"] = signs
    m = masks[0] & masks[-1] & valid.all(0)
    i, j = np.triu_indices(768, 1)
    keep = m[i] & m[j]
    i, j = i[keep], j[keep]
    a0 = angles(vectors[0])[i, j]
    a1 = angles(vectors[-1])[i, j]
    a0 = np.where(signs[0, i] == signs[0, j], a0, 180-a0)
    a1 = np.where(signs[-1, i] == signs[-1, j], a1, 180-a1)
    chosen = [np.argmax(a1), np.argmin(a1-a0), np.argmax(a1-a0)]
    pair_ids = np.array([(i[k], j[k]) for k in chosen])
    histories = []
    for k, v in enumerate(vectors):
        a = angles(v)[pair_ids[:, 0], pair_ids[:, 1]]
        histories.append(np.where(signs[k, pair_ids[:, 0]] == signs[k, pair_ids[:, 1]], a, 180-a))
    out[f"b{b}_pairs"] = pair_ids
    out[f"b{b}_pair_angles"] = np.array(histories)
    metadata["blocks"].append(dict(block=b, persistent_beams=int(m.sum()),
        pair_ids=pair_ids.tolist(), pair_selection=["largest final angle", "largest angle decrease", "largest angle increase"]))
    print("DETAILS", b, round(time.monotonic()-t0, 1), flush=True)
np.savez_compressed(DEST / "DETAILS.npz", **out)
(DEST / "DETAILS_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("DONE", flush=True)
