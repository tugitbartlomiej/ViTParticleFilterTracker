from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as M


EPS = 1e-12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path: Path, device: torch.device) -> M.MalyViT:
    model = M.MalyViT().to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    model.zapisuj_latent(True)
    return model


@torch.inference_mode()
def latent(model: M.MalyViT, image: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(image).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    _, z = model(x)
    if z is None or tuple(z.shape) != (1, 600, 768):
        raise RuntimeError(f"Nieoczekiwany latent: {None if z is None else tuple(z.shape)}")
    return z[0].float()


def state_metrics(z: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is not None:
        z, target = z[mask], target[mask]
    delta = z - target
    return {
        "rmse": float(delta.square().mean().sqrt().item()),
        "mae": float(delta.abs().mean().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(z.flatten(), target.flatten(), dim=0).item()),
    }


def direction_metrics(start: torch.Tensor, end: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is not None:
        start, end, target = start[mask], end[mask], target[mask]
    move = (end - start).flatten()
    desired = (target - start).flatten()
    dot = torch.dot(move, desired)
    return {
        "movement_rmse": float(move.square().mean().sqrt().item()),
        "cosine_with_direction_to_target": float(torch.nn.functional.cosine_similarity(move, desired, dim=0).item()),
        "signed_target_fraction": float((dot / desired.square().sum().clamp_min(EPS)).item()),
        "rmse_progress": float(
            (start - target).square().mean().sqrt().item() - (end - target).square().mean().sqrt().item()
        ),
    }


def cosine_vectors(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--probe-config", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--target-origin", type=Path, required=True)
    ap.add_argument("--target-result", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bgr = cv2.imread(str(args.probe), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.probe)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (320, 480):
        image = cv2.resize(image, (480, 320), interpolation=cv2.INTER_AREA)

    models = {
        "base_k8000": load_model(args.base, device),
        "control500": load_model(args.control, device),
        "variant8_500": load_model(args.variant, device),
    }
    z = {name: latent(model, image, device) for name, model in models.items()}
    target_np = np.load(args.target).astype(np.float32)
    origin_pack = np.load(args.target_origin)
    old_origin_np = origin_pack["z_baza"].astype(np.float32)
    old_image_np = origin_pack["obraz"]
    if target_np.shape != (600, 768) or old_origin_np.shape != (600, 768):
        raise RuntimeError(f"Nieoczekiwany kształt celu: {target_np.shape}, źródła: {old_origin_np.shape}")
    if not np.array_equal(old_image_np, image):
        raise RuntimeError("Obraz w pliku źródłowym celu nie jest identyczny z aktualną sondą m0011")
    target = torch.from_numpy(target_np).to(device)
    old_origin = torch.from_numpy(old_origin_np).to(device)

    cfg = json.loads(args.probe_config.read_text(encoding="utf-8"))
    x0, y0, x1, y1 = cfg["gt_xyxy_480x320"][0]
    yy, xx = torch.meshgrid(torch.arange(20), torch.arange(30), indexing="ij")
    centers_x = (xx.flatten() + 0.5) * M.KROK
    centers_y = (yy.flatten() + 0.5) * M.KROK
    roi = ((centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)).to(device)
    outside = ~roi

    target_result = json.loads(args.target_result.read_text(encoding="utf-8"))
    target_record = next(
        row for row in target_result["wyniki"] if row["cel"] == 11 and row["poziom"] == "minmax"
    )
    states = {name: state_metrics(value, target) for name, value in z.items()}
    states["old_k1000_natural_origin"] = state_metrics(old_origin, target)
    directions = {
        "base_k8000_to_control500": direction_metrics(z["base_k8000"], z["control500"], target),
        "base_k8000_to_variant8_500": direction_metrics(z["base_k8000"], z["variant8_500"], target),
        "control500_to_variant8_500_replacement_effect": direction_metrics(z["control500"], z["variant8_500"], target),
    }
    direction_old = target - old_origin
    movement_control = z["control500"] - z["base_k8000"]
    movement_variant = z["variant8_500"] - z["base_k8000"]
    movement_replacement = z["variant8_500"] - z["control500"]
    alignment = {
        "control_movement_vs_old_k1000_artificial_delta": cosine_vectors(movement_control, direction_old),
        "variant_movement_vs_old_k1000_artificial_delta": cosine_vectors(movement_variant, direction_old),
        "replacement_effect_vs_old_k1000_artificial_delta": cosine_vectors(movement_replacement, direction_old),
    }

    regional = {}
    for region, mask in {"roi": roi, "outside_roi": outside}.items():
        regional[region] = {
            "tokens": int(mask.sum().item()),
            "states": {name: state_metrics(value, target, mask) for name, value in z.items()},
            "directions": {
                "base_to_control": direction_metrics(z["base_k8000"], z["control500"], target, mask),
                "base_to_variant": direction_metrics(z["base_k8000"], z["variant8_500"], target, mask),
                "control_to_variant_replacement_effect": direction_metrics(z["control500"], z["variant8_500"], target, mask),
            },
            "replacement_effect_vs_old_k1000_artificial_delta_cosine": cosine_vectors(movement_replacement, direction_old, mask),
        }

    origin_compatibility = {
        "same_probe_pixels": True,
        "k1000_origin_vs_k8000_base": state_metrics(old_origin, z["base_k8000"]),
        "k1000_origin_to_artificial_target_rmse": state_metrics(old_origin, target)["rmse"],
        "checkpoint_relationship": "K1000 target and K8000 detector come from different documented runs; coordinate compatibility is not established",
    }
    clean = directions["control500_to_variant8_500_replacement_effect"]
    report = {
        "state": "complete",
        "question": "Czy trening z ośmioma podmianami przesunął naturalny latent m0011 w stronę sztucznego celu użytego do generacji i wyszukania obrazów?",
        "answer_for_saved_target": "closer" if clean["rmse_progress"] > 0 else "farther_or_not_closer",
        "target": {
            "path": str(args.target.resolve()),
            "sha256": sha(args.target),
            "provenance_checkpoint": "K1000",
            "provenance_checkpoint_sha256": "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e",
            "target_quality_under_k1000": target_record,
        },
        "models": {
            "base_k8000": {"path": str(args.base.resolve()), "sha256": sha(args.base)},
            "control500": {"path": str(args.control.resolve()), "sha256": sha(args.control)},
            "variant8_500": {"path": str(args.variant.resolve()), "sha256": sha(args.variant)},
        },
        "states_to_saved_target": states,
        "directions_to_saved_target": directions,
        "alignment_with_original_k1000_artificial_delta": alignment,
        "regional": regional,
        "origin_compatibility": origin_compatibility,
        "scientific_conclusion": (
            "The numerical diagnostic can say whether the K8000 branch moved toward the saved K1000 array, "
            "but it cannot establish movement toward a valid K8000 artificial target. A compatible m0011 target "
            "must be optimized with the exact K8000 checkpoint."
        ),
    }
    output_json = args.output / "TARGET_DIRECTION_K1000_DIAGNOSTIC.json"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    names = ["K8000", "kontrola 500", "wariant 8"]
    rmses = [states[key]["rmse"] for key in ("base_k8000", "control500", "variant8_500")]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bars = ax.bar(names, rmses, color=["#4C78A8", "#72B7B2", "#E45756"])
    ax.set_ylabel("RMSE do zapisanego sztucznego celu K1000")
    ax.set_title("Diagnostyczne porównanie kierunku — cel niezgodny z K8000")
    for bar, value in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.6f}", ha="center", va="bottom")
    fig.savefig(args.output / "TARGET_DIRECTION_K1000_DIAGNOSTIC.png", dpi=180)
    plt.close(fig)

    progress = clean["rmse_progress"]
    md = f"""# Kierunek ruchu względem zapisanego sztucznego celu m0011

## Wynik liczbowy

W porównaniu z gałęzią kontrolną wariant z ośmioma podmianami {('zbliżył' if progress > 0 else 'nie zbliżył')} naturalny latent m0011 do zapisanej tablicy sztucznego celu. RMSE kontrola-cel wynosi {states['control500']['rmse']:.8f}, a wariant-cel {states['variant8_500']['rmse']:.8f}. Zmiana odległości wynosi {progress:+.8f}. Cosine dodatkowego ruchu `kontrola -> wariant` z kierunkiem od kontroli do celu wynosi {clean['cosine_with_direction_to_target']:.8f}.

Cosine dodatkowego ruchu z pierwotną sztuczną korektą K1000 wynosi {alignment['replacement_effect_vs_old_k1000_artificial_delta']:.8f}.

## Ograniczenie

To jest wyłącznie diagnostyka numeryczna. Sztuczny cel pochodzi z checkpointa K1000 o SHA256 `3df42a6a...649e`, natomiast trening podmian zaczynał się z K8000 o SHA256 `0844a20a...490b`. Są to różne udokumentowane biegi. Naturalne latenty tego samego obrazu mają między nimi RMSE {origin_compatibility['k1000_origin_vs_k8000_base']['rmse']:.8f} i cosine {origin_compatibility['k1000_origin_vs_k8000_base']['cosine']:.8f}. Dlatego wynik nie jest dowodem ruchu ku poprawnemu celowi K8000.

Aby odpowiedzieć bez tego ograniczenia, trzeba wyznaczyć sztuczny cel m0011 ponownie na dokładnym K8000, a następnie powtórzyć to samo porównanie `kontrola -> wariant`.
"""
    (args.output / "TARGET_DIRECTION_K1000_DIAGNOSTIC.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "answer": report["answer_for_saved_target"],
        "control_rmse": states["control500"]["rmse"],
        "variant_rmse": states["variant8_500"]["rmse"],
        "progress": progress,
        "replacement_direction_cosine": clean["cosine_with_direction_to_target"],
        "replacement_vs_old_delta_cosine": alignment["replacement_effect_vs_old_k1000_artificial_delta"],
        "origin_mismatch_rmse": origin_compatibility["k1000_origin_vs_k8000_base"]["rmse"],
        "origin_mismatch_cosine": origin_compatibility["k1000_origin_vs_k8000_base"]["cosine"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
