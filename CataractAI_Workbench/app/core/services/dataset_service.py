"""Dataset analysis service -- parse COCO annotations, compute statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

from .base import AnalysisService


class DatasetAnalysisService(AnalysisService):
    """Parse COCO annotations and compute dataset statistics."""

    def analyze(self, data: dict) -> dict:
        """Expect ``{coco_data, img_dir?, selected_category?}``.

        Returns ``{composition, bboxes, images, recommendations, category_names}``.
        """
        coco = data["coco_data"]
        img_dir = data.get("img_dir")
        selected_cat = data.get("selected_category", "All")

        images = coco.get("images", [])
        annotations = list(coco.get("annotations", []))
        categories = coco.get("categories", [])

        # Filter by category
        if selected_cat != "All":
            cat_id = None
            for c in categories:
                if c.get("name") == selected_cat:
                    cat_id = c["id"]
                    break
            if cat_id is not None:
                annotations = [a for a in annotations if a.get("category_id") == cat_id]

        # Count images on disk
        dir_img_count = 0
        if img_dir and Path(img_dir).is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            dir_img_count = sum(
                1 for f in Path(img_dir).iterdir() if f.suffix.lower() in exts
            )

        n_images = len(images) if images else dir_img_count
        n_annotations = len(annotations)
        cat_names = [c.get("name", f"id={c.get('id')}") for c in categories]

        img_ann_count = Counter(a.get("image_id") for a in annotations)
        all_img_ids = set(img["id"] for img in images) if images else set()
        zero_box_count = len(all_img_ids - set(img_ann_count.keys()))

        box_counts = list(img_ann_count.values())
        one_box = sum(1 for c in box_counts if c == 1)
        multi_box = sum(1 for c in box_counts if c >= 2)
        avg_boxes = n_annotations / max(n_images, 1)

        comp_lines = [
            f"Total images:       {n_images}",
            f"Total annotations:  {n_annotations}",
            f"Categories:         {', '.join(cat_names) if cat_names else 'N/A'}",
            "",
            f"Avg boxes/image:    {avg_boxes:.2f}",
            f"Images with 0 boxes: {zero_box_count}"
            f" ({zero_box_count / max(n_images, 1) * 100:.1f}%)",
            f"Images with 1 box:   {one_box}"
            f" ({one_box / max(n_images, 1) * 100:.1f}%)",
            f"Images with 2+ boxes: {multi_box}"
            f" ({multi_box / max(n_images, 1) * 100:.1f}%)",
        ]

        if len(categories) > 1:
            comp_lines.append("")
            comp_lines.append("Per-category annotations:")
            cat_counts = Counter(
                a.get("category_id") for a in coco.get("annotations", [])
            )
            for cat in categories:
                cid = cat["id"]
                cname = cat.get("name", f"id={cid}")
                comp_lines.append(f"  {cname}: {cat_counts.get(cid, 0)}")

        # Extract bbox array
        import numpy as np

        bboxes_list = [a["bbox"] for a in annotations if len(a.get("bbox", [])) == 4]
        bboxes = np.array(bboxes_list) if bboxes_list else np.empty((0, 4))

        recommendations = self._generate_recommendations(
            n_images, n_annotations, zero_box_count, bboxes, categories, coco,
        )

        return {
            "composition": "\n".join(comp_lines),
            "bboxes": bboxes,
            "images": images,
            "recommendations": "\n\n".join(recommendations),
            "category_names": cat_names,
        }

    # -- recommendations ---------------------------------------------

    @staticmethod
    def _generate_recommendations(
        n_images: int,
        n_annotations: int,
        zero_box_count: int,
        bboxes,
        categories: list,
        coco_data: dict,
    ) -> List[str]:
        import numpy as np

        recs: List[str] = []

        # Background-heavy dataset
        if n_images > 0:
            bg_pct = zero_box_count / n_images * 100
            if bg_pct > 80:
                recs.append(
                    f"[!] {bg_pct:.0f}% images have no annotations -- typical for"
                    f" background training but verify this is intentional."
                )
            elif bg_pct > 50:
                recs.append(
                    f"[~] {bg_pct:.0f}% images have no annotations."
                    f" Ensure background-to-object ratio is intentional."
                )

        # Box size consistency
        if len(bboxes) > 0:
            areas = bboxes[:, 2] * bboxes[:, 3]
            cv = float(np.std(areas) / max(np.mean(areas), 1))
            if cv < 0.5:
                recs.append(f"[OK] Box sizes are consistent (CV={cv:.2f}).")
            elif cv < 1.0:
                recs.append(
                    f"[~] Moderate variation in box sizes (CV={cv:.2f})."
                    f" Consider multi-scale augmentation."
                )
            else:
                recs.append(
                    f"[!] High variation in box sizes (CV={cv:.2f})."
                    f" Use multi-scale training or anchor optimization."
                )

        # Small dataset warning
        if 0 < n_images < 100:
            recs.append(
                f"[!] Very small dataset ({n_images} images). Heavy augmentation"
                f" is essential. Consider collecting more data."
            )
        elif 0 < n_images < 500:
            recs.append(
                f"[~] Small dataset ({n_images} images). Consider adding"
                f" augmentation for better generalization."
            )
        elif n_images >= 500:
            recs.append(f"[OK] Dataset size ({n_images} images) is reasonable.")

        # Annotation density
        if n_images > 0 and n_annotations > 0:
            density = n_annotations / n_images
            if density > 20:
                recs.append(
                    f"[~] High annotation density ({density:.1f} boxes/image)."
                    f" Ensure the detector can handle dense scenes."
                )
            elif density < 0.5:
                recs.append(
                    f"[~] Low annotation density ({density:.2f} boxes/image)."
                    f" Many images may be unannotated backgrounds."
                )

        # Category balance
        if len(categories) > 1:
            cat_counts = Counter(
                a.get("category_id") for a in coco_data.get("annotations", [])
            )
            counts = list(cat_counts.values())
            if counts:
                ratio = max(counts) / max(min(counts), 1)
                if ratio > 10:
                    recs.append(
                        f"[!] Severe category imbalance (max/min ratio: {ratio:.1f}x)."
                        f" Use class-weighted loss or oversampling."
                    )
                elif ratio > 3:
                    recs.append(
                        f"[~] Category imbalance (max/min ratio: {ratio:.1f}x)."
                        f" Consider balancing strategies."
                    )
                else:
                    recs.append("[OK] Categories are reasonably balanced.")

        if not recs:
            recs.append("[OK] No issues detected.")

        return recs
