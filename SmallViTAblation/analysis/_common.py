#!/usr/bin/env python3
"""
Wspolne funkcje dla analiz Agenta 5 (analysis/).

Zasada nadrzedna: zero zaszytych wartosci konfiguracyjnych — wszystko czytane
z configs/protocol.yaml (sekcja `analysis`) i configs/variants.yaml.
Sciezki w protocol.yaml sa wzgledem korzenia repozytorium (jak split.file).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("Brak pyyaml. Zainstaluj: py -3.11 -m pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[1]   # SmallViTAblation/
REPO = ROOT.parent                            # ViTParticleFilterTracker/
CONFIGS = ROOT / "configs"

# Przyrostek augmentacji: "_aug_3" albo "_aug_flip_3" (typ zakodowany w nazwie).
AUG_RE = re.compile(r"_aug_(?:([A-Za-z][A-Za-z0-9]*)?_)?(\d+)$")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def load_protocol(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIGS / "protocol.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_variants(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIGS / "variants.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def analysis_cfg(protocol: dict) -> dict:
    cfg = protocol.get("analysis")
    if not cfg:
        raise SystemExit("Brak sekcji `analysis` w protocol.yaml — dopisz sciezki (README sekcja 5).")
    return cfg


def resolve(p: str | Path) -> Path:
    """Sciezka wzgledna w protocol.yaml -> wzgledem korzenia repozytorium."""
    pp = Path(p)
    return pp if pp.is_absolute() else REPO / pp


def cfg_path(cfg: dict, *keys: str, default=None) -> Path | None:
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    if node is None:
        return default
    return resolve(node)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i} — zly JSON: {e}") from e
    return out


def load_pool_manifest(path: Path) -> list[dict]:
    recs = load_jsonl(path)
    for r in recs:
        if "frame_id" not in r or "split" not in r:
            raise SystemExit(f"{path} — wiersz bez frame_id/split (kontrakt README 4.1)")
    return recs


def split_aug(stem: str) -> tuple[str, str | None, int | None]:
    """'p01_frame_3_aug_flip_2' -> ('p01_frame_3', 'flip', 2); bez augmentacji: (stem, None, None)."""
    m = AUG_RE.search(stem)
    if not m:
        return stem, None, None
    return stem[: m.start()], m.group(1), int(m.group(2))


def stem_of(name: str) -> str:
    return Path(name).stem if Path(name).suffix.lower() in IMAGE_EXTS else name.rsplit(".", 1)[0] if "." in Path(name).name else name


def iter_variant_files(variants_dir: Path) -> list[tuple[str, dict]]:
    out = []
    d = Path(variants_dir)
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_id = data.get("run_id", f.stem)
        out.append((run_id, data))
    return out


# ---------------------------------------------------------------- osadzenia

def fake_embedding(path: Path, cfg: dict) -> np.ndarray:
    """Deterministyczna cecha zastepcza z pikseli (bez modelu, bez etykiet).

    Siatka fake_grid x fake_grid w skali szarosci, srednia odjeta, L2=1.
    Gdy pliku nie ma lub PIL nie czyta: wektor z ziarna = sha256 zawartosci/sciezki.
    """
    grid = int(cfg.get("fake_grid", 8))
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((grid, grid), Image.BILINEAR)
        v = np.asarray(img, dtype=np.float64).ravel()
    except Exception:
        h = hashlib.sha256(str(path).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.standard_normal(grid * grid)
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def embed_dino(paths: list[Path], cfg: dict) -> np.ndarray:
    """Osadzenia DINOv2 przez torch.hub — lazy import, tylko w trybie prawdziwym."""
    import torch
    from PIL import Image

    cov = cfg.get("coverage", {})
    model_name = cov.get("dino_model", "dinov2_vits14")
    size = int(cov.get("dino_image_size", 224))
    batch = int(cov.get("dino_batch_size", 32))

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            tensors = []
            for p in paths[i : i + batch]:
                img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                tensors.append((t - mean) / std)
            vecs.append(model(torch.stack(tensors)).cpu().numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)


def compute_embeddings(entries: list[tuple[str, Path]], cfg: dict, mode: str,
                       cache_path: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Osadzenia dla listy (klucz, sciezka), opcjonalnie z pamiecia podreczna npz.

    mode: "fake" (cechy z pikseli) albo "dino" (torch.hub, lazy).
    Zwraca (keys, matrix [n, d]).
    """
    keys_all = [k for k, _ in entries]
    have: dict[str, np.ndarray] = {}
    emb_mode = mode

    if cache_path and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=False)
        emb_mode = str(z["mode"])
        if emb_mode != mode:
            print(f"  [cache] tryb w pliku ({emb_mode}) != zadany ({mode}) — licze od nowa")
            have = {}
        else:
            for k, v in zip(z["keys"].tolist(), z["emb"]):
                have[str(k)] = v
            print(f"  [cache] {len(have)} osadzen z {cache_path}")

    missing = [(k, p) for k, p in entries if k not in have]
    if missing:
        if mode == "fake":
            mat = np.stack([fake_embedding(p, cfg) for _, p in missing])
        elif mode == "dino":
            mat = embed_dino([p for _, p in missing], cfg)
        else:
            raise SystemExit(f"Nieznany tryb osadzen: {mode}")
        for (k, _), v in zip(missing, mat):
            have[k] = np.asarray(v, dtype=np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 keys=np.array(keys_all, dtype=object).astype(str),
                 emb=np.stack([have[k] for k in keys_all]),
                 mode=np.array(mode))
        print(f"  [cache] zapisano {len(keys_all)} osadzen -> {cache_path}")

    return keys_all, np.stack([have[k] for k in keys_all]).astype(np.float32)


def l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def min_distances(query: np.ndarray, ref: np.ndarray, distance: str) -> np.ndarray:
    """Odleglosc kazdego wiersza query do NAJBLIZSZEGO wiersza ref."""
    if distance == "cosine":
        d = 1.0 - l2norm(query) @ l2norm(ref).T
    elif distance == "euclidean":
        diff = query[:, None, :] - ref[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
    else:
        raise SystemExit(f"Nieznana miara odleglosci: {distance}")
    return d.min(axis=1)


# ---------------------------------------------------------------- korelacje

def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def correlations(x, y) -> dict:
    """Pearson + Spearman; p-value z scipy jesli jest (lazy), inaczej None."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    out = {"n": n, "pearson_r": None, "pearson_p": None,
           "spearman_r": None, "spearman_p": None}
    if n < 2 or np.std(x) == 0 or np.std(y) == 0:
        return out
    out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    out["spearman_r"] = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    try:
        from scipy import stats
        out["pearson_p"] = float(stats.pearsonr(x, y).pvalue)
        out["spearman_p"] = float(stats.spearmanr(x, y).pvalue)
    except ImportError:
        pass
    return out


# ---------------------------------------------------------------- perceptual hash

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path, hash_size: int = 8) -> np.ndarray | None:
    """Roznicowy hash perceptualny (dhash) w numpy/PIL, bez ciezkich zaleznosci.

    Zwraca uint8[8] (64 bity) albo None gdy nie dalo sie wczytac obrazu.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.int16)
    except Exception:
        return None
    bits = (a[:, :-1] < a[:, 1:]).ravel()
    return np.packbits(bits)


def hamming_matrix(h1: np.ndarray, h2: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Macierz odleglosci Hamminga miedzy zestawami hashow uint8[n, 8]."""
    out = np.empty((len(h1), len(h2)), dtype=np.int16)
    for i in range(0, len(h1), chunk):
        x = np.bitwise_xor(h1[i : i + chunk, None, :], h2[None, :, :])
        out[i : i + chunk] = _POPCOUNT_LUT[x].sum(axis=2, dtype=np.int16)
    return out


# ---------------------------------------------------------------- misc

def write_json(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {path}")


def external_metric_from(metrics: dict) -> tuple[float | None, str | None]:
    """mAP_0.5 ze zbioru zewnętrznego (external_clean preferowany) — TYLKO do korelacji
    post-hoc z gotowych metrics.json; nigdy do wyboru epoki/progu."""
    res = metrics.get("results", {})
    for block in ("external_clean", "external_full"):
        if block in res and isinstance(res[block], dict):
            v = res[block].get("mAP_0.5")
            if v is not None:
                return float(v), block
    return None, None
