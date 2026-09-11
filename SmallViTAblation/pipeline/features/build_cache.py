"""Liczy cechy DINO / FastSAM / Fourier dla klatek split=='train' z pool_manifest.

Wyjscie (kontrakt Agent 1, zadanie 1):
    feature_cache/dino.pkl      (N, D_dino) float32 + frame_ids w tej samej kolejnosci
    feature_cache/sam.pkl       (N, 9)
    feature_cache/fourier.pkl   (N, 1)
    feature_cache/manifest.json  ile klatek, wymiary, model + wersja, data

Liczenie jest WZNAWIALNE: stan czastkowy w <grupa>.partial.pkl (checkpoint co
features.checkpoint_every_frames klatek); przerwanie i ponowne uruchomienie
kontynuuje, a nie zaczyna od zera. Postep wypisywany na biezaco.

Modele produkcyjne (DINOv3 przez torch.hub, FastSAM przez ultralytics) laduja
sie LENIWIE - import i wagi dopiero przy pierwszej klatce. Flagi --fake-features
uzywa sie WYLACZNIE w testach dymnych (deterministyczne funkcje pikseli).

Uzycie:
    py -3.11 pipeline/features/build_cache.py \
        --manifest pool_manifest.jsonl --images-root . [--fake-features]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.features.common import (  # noqa: E402
    CheckpointedGroup, GROUP_NAMES, load_feature_pkl, load_protocol,
    read_manifest, resolve_under_root, train_pool, utc_now, write_cache_manifest)


# ------------------------------------------------------------ extractors ---

class FakeDinoExtractor:
    """FIXTURE: zastepcze cechy 'semantyczne' - deterministyczne histogramy."""

    def __init__(self, dim: int):
        self.dim = dim

    def meta(self) -> dict:
        return {"model": f"fake-dino-histogram-{self.dim}d", "version": "fixture",
                "fake": True}

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        per_ch = (self.dim - 16) // 3 if self.dim >= 19 else self.dim // 3
        for i, p in enumerate(paths):
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            gray = arr.mean(axis=2)
            feats = [np.histogram(gray, bins=16, range=(0, 255),
                                  density=True)[0]]
            for c in range(3):
                if per_ch > 0:
                    feats.append(np.histogram(arr[..., c], bins=per_ch,
                                              range=(0, 255), density=True)[0])
            v = np.concatenate(feats)
            v = v / (np.linalg.norm(v) + 1e-10)
            out[i, :v.shape[0]] = v[:self.dim]
        return out


class FakeSamExtractor:
    """FIXTURE: 9 deterministycznych statystyk pikseli udajacych zlozonosc."""

    DIM = 9

    def meta(self) -> dict:
        return {"model": "fake-sam-pixelstats-9d", "version": "fixture",
                "fake": True}

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        out = np.zeros((len(paths), self.DIM), dtype=np.float32)
        for i, p in enumerate(paths):
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            gray = arr.mean(axis=2) / 255.0
            gx = np.abs(np.diff(gray, axis=1))
            gy = np.abs(np.diff(gray, axis=0))
            hist = np.histogram(gray, bins=32, range=(0, 1), density=True)[0]
            hist = hist / (hist.sum() + 1e-10)
            entropy = float(-(hist * np.log2(hist + 1e-12)).sum() / 5.0)
            edge = float((gx > 0.1).mean())
            out[i] = [gray.mean(), gray.std(),
                      float(arr[..., 0].mean()), float(arr[..., 1].mean()),
                      float(arr[..., 2].mean()),
                      float(gx.mean() + gy.mean()), edge, entropy,
                      float(np.percentile(gray, 95) - np.percentile(gray, 5))]
        return out


class DinoExtractor:
    """PRODUKCJA: DINOv2/v3 przez torch.hub, token CLS. Leniwe ladowanie."""

    def __init__(self, model_name: str, image_size: int, batch_size: int):
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self._model = None
        self._torch = None
        self._transform = None
        self._version = "unknown"

    def _init(self) -> None:
        if self._model is not None:
            return
        import torch  # leniwie
        from torchvision import transforms
        repos = {"dinov2": "facebookresearch/dinov2:main",
                 "dinov3": "facebookresearch/dinov3:main",
                 "dino": "facebookresearch/dino:main"}
        family = "".join(ch for ch in self.model_name.split("_")[0] if not ch.isdigit())
        repo = repos.get(family, repos["dinov2"])
        self._torch = torch
        cache_dir = str(resolve_under_root("External/Models/torch_hub"))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(cache_dir)
        model = torch.hub.load(repo, self.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self._device = device
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        try:
            self._version = str(torch.hub.get_dir()) + f"/{repo}#{self.model_name}"
        except Exception:
            pass

    def meta(self) -> dict:
        return {"model": f"torch.hub {self.model_name}", "version": self._version,
                "image_size": self.image_size, "fake": False}

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        self._init()
        from PIL import Image
        torch = self._torch
        tensors = [self._transform(Image.open(p).convert("RGB")) for p in paths]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)


class FastSamExtractor:
    """PRODUKCJA: FastSAM (ultralytics) -> 9 metryk zlozonosci sceny.

    Wektor 9-wymiarowy (kolejnosc stal):
      0 log1p(num_segments), 1 avg_segment_size, 2 segment_size_std,
      3 coverage_ratio, 4 edge_density, 5 avg_confidence, 6 min_confidence,
      7 complexity_score (0-1), 8 perimeter_density
    """

    DIM = 9

    def __init__(self, weights: str, conf: float, iou: float):
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self._model = None
        self._version = "unknown"

    def _init(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # leniwie
        import ultralytics
        self._model = YOLO(self.weights)
        self._version = f"ultralytics {ultralytics.__version__}, {self.weights}"

    def meta(self) -> dict:
        return {"model": f"FastSAM ({self.weights})", "version": self._version,
                "conf": self.conf, "iou": self.iou, "fake": False}

    def _complexity(self, masks, confs, shape) -> np.ndarray:
        h, w = shape[:2]
        total = float(h * w)
        if masks is None or len(masks) == 0:
            return np.zeros(self.DIM, dtype=np.float32)
        areas = masks.sum(axis=(1, 2)).astype(np.float64)
        coverage = float(min(areas.sum(), total) / total)
        avg_size = float(areas.mean() / total)
        std_size = float(areas.std() / total) if len(areas) > 1 else 0.0
        edge_pixels = 0.0
        perim_pixels = 0.0
        for m in masks:
            mu = m.astype(np.uint8)
            pad = np.pad(mu, 1)
            nb = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2]
                  + pad[1:-1, 2:])
            border = (mu == 1) & (nb < 4)
            perim_pixels += float(border.sum())
            edge_pixels += float(((nb > 0) & (nb < 4)).sum())
        edge_density = edge_pixels / total
        perim_density = perim_pixels / total
        c = np.asarray(confs, dtype=np.float64)
        avg_conf, min_conf = float(c.mean()), float(c.min())
        n = len(areas)
        score = (0.25 * min(n / 50.0, 1.0) + 0.20 * min(std_size * 10, 1.0)
                 + 0.20 * min(edge_density * 10, 1.0) + 0.20 * coverage
                 + 0.15 * (1.0 - avg_conf))
        return np.array([np.log1p(n), avg_size, std_size, coverage,
                         edge_density, avg_conf, min_conf,
                         float(np.clip(score, 0, 1)), perim_density],
                        dtype=np.float32)

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        self._init()
        from PIL import Image
        out = np.zeros((len(paths), self.DIM), dtype=np.float32)
        results = self._model([str(p) for p in paths], verbose=False,
                              conf=self.conf, iou=self.iou)
        for i, r in enumerate(results):
            masks = None
            confs: list[float] = []
            if r.masks is not None and r.masks.data is not None:
                masks = (r.masks.data.cpu().numpy() > 0.5)
                confs = (r.boxes.conf.cpu().numpy().tolist()
                         if r.boxes is not None else [1.0] * len(masks))
            shape = (np.asarray(Image.open(paths[i])).shape
                     if r.orig_shape is None else (r.orig_shape[0], r.orig_shape[1], 3))
            out[i] = self._complexity(masks, confs, shape)
        return out


class FourierExtractor:
    """PRODUKCJA i fixture: 1 cecha - udzial energii wysokich czestotliwosci.

    Miara ostrosci: E(high) / E(all) z log-widma FFT (pasmo z protocol.yaml
    features.fourier_high_band). Czysta numpy - dziala bez ciezkich zaleznosci.
    """

    DIM = 1

    def __init__(self, high_band: tuple[float, float], fake: bool = False):
        self.high_band = tuple(high_band)
        self.fake = fake

    def meta(self) -> dict:
        return {"model": "fourier-highband-ratio",
                "version": "numpy-fft", "high_band": list(self.high_band),
                "fake": self.fake}

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        lo, hi = self.high_band
        out = np.zeros((len(paths), 1), dtype=np.float32)
        for i, p in enumerate(paths):
            gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            rows, cols = gray.shape
            mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
            cy, cx = rows // 2, cols // 2
            y, x = np.ogrid[:rows, :cols]
            rmax = float(np.sqrt(cy ** 2 + cx ** 2)) + 1e-10
            radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / rmax
            total = float(mag.sum()) + 1e-10
            high = float(mag[(radius >= lo) & (radius < hi)].sum())
            out[i, 0] = high / total
        return out


def build_extractors(protocol: dict, fake: bool) -> dict:
    feat = protocol["features"]
    if fake:
        return {
            "dino": FakeDinoExtractor(int(feat.get("fake_dino_dim", 64))),
            "sam": FakeSamExtractor(),
            "fourier": FourierExtractor(
                tuple(feat.get("fourier_high_band", [0.5, 1.0])), fake=True),
        }
    return {
        "dino": DinoExtractor(str(feat["dino_model"]),
                              int(feat["dino_image_size"]),
                              int(feat["extract_batch_size"])),
        "sam": FastSamExtractor(str(feat["fastsam_weights"]),
                                float(feat["fastsam_conf"]),
                                float(feat["fastsam_iou"])),
        "fourier": FourierExtractor(
            tuple(feat.get("fourier_high_band", [0.5, 1.0]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="pool_manifest.jsonl (domyslnie z protocol.yaml paths)")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="katalog glowny, wzgledem ktorego licza sie 'path'")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--groups", default="dino,sam,fourier")
    ap.add_argument("--fake-features", action="store_true",
                    help="FIXTURE/test dymny: cechy zastepcze zamiast modeli")
    ap.add_argument("--limit", type=int, default=None,
                    help="tylko pierwsze N klatek puli (testy)")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    feat = protocol["features"]
    paths_cfg = protocol.get("paths", {})

    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))
    images_root = args.images_root or resolve_under_root(
        paths_cfg.get("images_root", "."))
    cache_dir = args.cache_dir or resolve_under_root(
        paths_cfg.get("feature_cache_dir", "feature_cache"))

    records = read_manifest(manifest)
    pool = train_pool(records)
    if args.limit:
        pool = pool[:args.limit]
    print(f"Pula train: {len(pool)} klatek (manifest: {manifest})")
    if not pool:
        sys.exit("Pusta pula train - nic do policzenia.")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in groups:
        if g not in GROUP_NAMES:
            sys.exit(f"Nieznana grupa '{g}' (dostepne: {GROUP_NAMES})")

    extractors = build_extractors(protocol, args.fake_features)
    batch_size = int(feat.get("extract_batch_size", 32))
    checkpoint_every = int(feat.get("checkpoint_every_frames", 500))

    manifest_entries: dict = {}
    for name in groups:
        ext = extractors[name]
        dim = getattr(ext, "dim", None) or getattr(ext, "DIM")
        ckpt = CheckpointedGroup(cache_dir, name, int(dim), checkpoint_every)
        todo = [r for r in pool if r["frame_id"] not in ckpt.done_ids]
        print(f"\n=== {name}: do policzenia {len(todo)}/{len(pool)} "
              f"(wznowiono {len(pool) - len(todo)}) ===")
        done_total = len(pool) - len(todo)
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            bpaths = [Path(images_root) / r["path"] for r in batch]
            feats = ext.extract_batch(bpaths)
            ckpt.append([r["frame_id"] for r in batch], feats)
            done_total += len(batch)
            print(f"[{name}] {done_total}/{len(pool)} klatek "
                  f"({100.0 * done_total / len(pool):.1f}%)", flush=True)
        out = ckpt.finalize([r["frame_id"] for r in pool], ext.meta())
        print(f"[{name}] zapisano {out}")
        meta = ext.meta()
        manifest_entries[name] = {
            "n_frames": len(pool),
            "dim": int(load_feature_pkl(out)["features"].shape[1]),
            "file": out.name, **meta,
        }

    mpath = write_cache_manifest(cache_dir, manifest_entries)
    print(f"\nManifest cache: {mpath} ({utc_now()})")
    print("Kolejnosc frame_id identyczna we wszystkich plikach "
          "(sort rosnaco po frame_id).")


if __name__ == "__main__":
    main()
