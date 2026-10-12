from __future__ import annotations

import hashlib
import html
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation")
QUERY_DIR = ROOT / "outputs" / "latent_cel_do_obrazow" / "sd_k8000_m0011_interpolacja"
INDEX_DIR = ROOT / "outputs" / "vit_dino_faiss_35000"
OUT = ROOT / "outputs" / "latent_cel_do_obrazow" / "sd_k8000_m0011_dino_train_retrieval"
NATURAL = OUT / "naturalne_top5"
TOP_K = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_archive_image(row: dict) -> Image.Image:
    with zipfile.ZipFile(row["archive"], "r") as zf:
        with zf.open(row["member"], "r") as f:
            return Image.open(f).convert("RGB")


def fit_thumb(image: Image.Image, size=(320, 220)) -> Image.Image:
    canvas = Image.new("RGB", size, "#fff7df")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    NATURAL.mkdir(parents=True, exist_ok=True)

    config = json.loads((INDEX_DIR / "CONFIG.json").read_text(encoding="utf-8"))
    spec = config["spec"]
    manifest = [
        json.loads(line)
        for line in (INDEX_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest) != 35000 or {row.get("split") for row in manifest} != {"TRAIN"}:
        raise RuntimeError("Indeks nie jest oczekiwanym indeksem 35 000 obrazów TRAIN")

    features = np.asarray(np.load(INDEX_DIR / "features.npy", mmap_mode="r"), dtype=np.float32)
    if features.shape != (35000, 384):
        raise RuntimeError(f"Nieoczekiwany kształt cech: {features.shape}")

    query_files = sorted(QUERY_DIR.glob("k??_*.png"))
    if len(query_files) != 11:
        raise RuntimeError(f"Oczekiwano 11 zapytań, znaleziono {len(query_files)}")
    control = QUERY_DIR / "referencja_m0011.png"
    if not control.exists():
        raise FileNotFoundError(control)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(spec["hub_repo"], spec["model"], source="local", pretrained=False)
    state = torch.load(spec["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    all_queries = query_files + [control]
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in all_queries]).to(device)
    with torch.inference_mode():
        q = model.forward_features(batch)["x_norm_clstoken"].float()
        q = torch.nn.functional.normalize(q, dim=1).cpu().numpy().astype(np.float32)

    records = []
    natural_cache: dict[int, Image.Image] = {}
    for qi, path in enumerate(all_queries):
        scores = features @ q[qi]
        indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        matches = []
        query_tag = path.stem if path != control else "kontrola_referencja"
        for rank, idx_raw in enumerate(indices, 1):
            idx = int(idx_raw)
            row = manifest[idx]
            if idx not in natural_cache:
                natural_cache[idx] = load_archive_image(row)
            image = natural_cache[idx]
            filename = f"{query_tag}__r{rank:02d}__{row['film']}__f{int(row['frame']):06d}.jpg"
            image.save(NATURAL / filename, quality=94, subsampling=0)
            matches.append(
                {
                    "rank": rank,
                    "index_row": idx,
                    "cosine": float(scores[idx]),
                    "film": row["film"],
                    "frame": int(row["frame"]),
                    "id": row["id"],
                    "categories": [g["category"] for g in row.get("gt", [])],
                    "source_archive": row["archive"],
                    "source_member": row["member"],
                    "saved_file": f"naturalne_top5/{filename}",
                }
            )
        records.append(
            {
                "query": path.name,
                "query_sha256": sha256(path),
                "control": path == control,
                "matches": matches,
            }
        )

    generated_records = [r for r in records if not r["control"]]
    top1_scores = [r["matches"][0]["cosine"] for r in generated_records]
    top1_ids = [r["matches"][0]["id"] for r in generated_records]
    all_ids = [m["id"] for r in generated_records for m in r["matches"]]
    adjacent_top5_overlap = []
    for left, right in zip(generated_records, generated_records[1:]):
        a = {m["id"] for m in left["matches"]}
        b = {m["id"] for m in right["matches"]}
        adjacent_top5_overlap.append(len(a & b))

    result = {
        "state": "complete",
        "meaning": "nearest natural TRAIN frames in DINOv2 ViT-S/14 CLS cosine space",
        "queries": len(generated_records),
        "top_k": TOP_K,
        "index": {
            "path": str(INDEX_DIR),
            "count": len(manifest),
            "splits": dict(Counter(row["split"] for row in manifest)),
            "feature_shape": list(features.shape),
            "manifest_sha256": sha256(INDEX_DIR / "manifest.jsonl"),
            "features_sha256": sha256(INDEX_DIR / "features.npy"),
            "model": spec["model"],
            "weights": spec["weights"],
            "weights_sha256": sha256(Path(spec["weights"])),
        },
        "summary": {
            "top1_cosine_mean": float(np.mean(top1_scores)),
            "top1_cosine_min": float(np.min(top1_scores)),
            "top1_cosine_max": float(np.max(top1_scores)),
            "unique_top1": len(set(top1_ids)),
            "unique_among_55_matches": len(set(all_ids)),
            "adjacent_top5_overlap_mean": float(np.mean(adjacent_top5_overlap)),
            "adjacent_top5_overlap": adjacent_top5_overlap,
        },
        "records": records,
        "caveat": "DINO cosine measures visual-semantic similarity; it is not equality of the full ViT latent or evidence of the same training gradient.",
    }
    (OUT / "WYNIK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Zbiorczy obraz: dla każdego kroku zapytanie i najlepsza naturalna klatka.
    font = ImageFont.load_default(size=18)
    label_font = ImageFont.load_default(size=15)
    cell_w, image_h, label_h = 320, 220, 62
    sheet = Image.new("RGB", (cell_w * 4, (image_h + label_h) * 6), "#fff7df")
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(generated_records):
        row = i // 2
        pair = i % 2
        x0 = pair * 2 * cell_w
        y0 = row * (image_h + label_h)
        query_image = fit_thumb(Image.open(QUERY_DIR / rec["query"]).convert("RGB"), (cell_w, image_h))
        match = rec["matches"][0]
        natural_image = fit_thumb(Image.open(OUT / match["saved_file"]).convert("RGB"), (cell_w, image_h))
        sheet.paste(query_image, (x0, y0))
        sheet.paste(natural_image, (x0 + cell_w, y0))
        draw.text((x0 + 8, y0 + image_h + 5), f"{rec['query'][:24]}\nobraz SD", fill="#17252a", font=label_font)
        draw.text((x0 + cell_w + 8, y0 + image_h + 5), f"TRAIN: {match['film']} f{match['frame']}\ncosine {match['cosine']:.4f}", fill="#17252a", font=label_font)
    sheet.save(OUT / "POROWNANIE_TOP1_11.jpg", quality=94, subsampling=0)

    sections = []
    for rec in generated_records:
        cards = [
            f'<article class="card query"><img src="../sd_k8000_m0011_interpolacja/{html.escape(rec["query"])}"><b>Zapytanie SD</b><small>{html.escape(rec["query"])}</small></article>'
        ]
        for m in rec["matches"]:
            cards.append(
                f'<article class="card"><img src="{html.escape(m["saved_file"])}"><b>#{m["rank"]} · cosine {m["cosine"]:.4f}</b>'
                f'<small>{html.escape(m["film"])} · klatka {m["frame"]}<br>{html.escape(", ".join(m["categories"]))}</small></article>'
            )
        sections.append(f'<section><h2>{html.escape(rec["query"])}</h2><div class="row">{"".join(cards)}</div></section>')

    page = f"""<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Naturalne odpowiedniki obrazów SD</title>
<style>body{{font-family:Arial,sans-serif;background:#fff7df;color:#17252a;margin:24px}}h1{{font-size:34px}}h2{{margin-top:40px}}.summary{{padding:18px;background:#e4f0e8;border-left:7px solid #2f7f78}}.row{{display:grid;grid-template-columns:repeat(6,minmax(180px,1fr));gap:12px}}.card{{background:white;border:1px solid #c8c8b8;padding:8px}}.query{{border:4px solid #e8862c}}img{{width:100%;aspect-ratio:3/2;object-fit:contain;background:#111}}b,small{{display:block;margin-top:7px}}small{{line-height:1.35}}@media(max-width:1100px){{.row{{grid-template-columns:repeat(3,1fr)}}}}</style></head><body>
<h1>Naturalne odpowiedniki 11 obrazów SD</h1><div class="summary">Przeszukano dokładnie 35 000 naturalnych klatek TRAIN. Miara: cosine w przestrzeni DINOv2 ViT-S/14 CLS. Średni cosine top-1: {result['summary']['top1_cosine_mean']:.4f}; zakres: {result['summary']['top1_cosine_min']:.4f}–{result['summary']['top1_cosine_max']:.4f}. To podobieństwo wizualne, nie zgodność pełnego latentu ViT.</div>
{''.join(sections)}</body></html>"""
    (OUT / "GALERIA.html").write_text(page, encoding="utf-8")

    summary_md = f"""# Wyniki

Przeszukano 35 000 naturalnych klatek ze splitu TRAIN dla 11 obrazów wygenerowanych przez SD.

- średni cosine top-1: {result['summary']['top1_cosine_mean']:.4f}
- zakres cosine top-1: {result['summary']['top1_cosine_min']:.4f}–{result['summary']['top1_cosine_max']:.4f}
- unikalne obrazy top-1: {result['summary']['unique_top1']} z 11
- unikalne obrazy w 55 trafieniach: {result['summary']['unique_among_55_matches']}
- średnia liczba wspólnych trafień top-5 między sąsiednimi krokami: {result['summary']['adjacent_top5_overlap_mean']:.2f} z 5

Interpretacja: DINO znajduje naturalne klatki podobne wizualnie do kolejnych obrazów SD. Wynik nie oznacza, że znalezione klatki mają taki sam pełny latent ViT ani że wywołają tę samą zmianę podczas uczenia.
"""
    (OUT / "WYNIKI.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
