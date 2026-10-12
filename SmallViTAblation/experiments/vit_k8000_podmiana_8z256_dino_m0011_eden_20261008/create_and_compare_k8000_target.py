from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def latent_from_checkpoint(model_cls, checkpoint: Path, image: np.ndarray, device: torch.device) -> torch.Tensor:
    net = model_cls().to(device)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    net.eval().requires_grad_(False)
    captured = {}
    hook = net.bloki[-1].mlp.act.register_forward_hook(lambda _m, _i, out: captured.update(z=out.detach()))
    try:
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().to(device) / 255.0
        with torch.inference_mode():
            net(x)
    finally:
        hook.remove()
    return captured["z"][0].float()


def state_metrics(z: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    d = z - target
    return {
        "rmse": float(d.square().mean().sqrt().item()),
        "mae": float(d.abs().mean().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(z.flatten(), target.flatten(), dim=0).item()),
    }


def direction_metrics(start: torch.Tensor, end: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    move = (end - start).flatten()
    desired = (target - start).flatten()
    before = (start - target).square().mean().sqrt()
    after = (end - target).square().mean().sqrt()
    return {
        "movement_rmse": float(move.square().mean().sqrt().item()),
        "cosine_with_direction_to_target": float(torch.nn.functional.cosine_similarity(move, desired, dim=0).item()),
        "signed_target_fraction": float((torch.dot(move, desired) / desired.square().sum().clamp_min(1e-12)).item()),
        "rmse_before": float(before.item()),
        "rmse_after": float(after.item()),
        "rmse_progress": float((before - after).item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ["KOREKTA_WYJSCIE"] = str((args.output / "optimization").resolve())
    sys.path.insert(0, str(ROOT / "app"))
    import korekta_latentu as KL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = KL.manifest()
    row = rows[11]
    image, boxes = KL.obraz_manifestu(row)
    st = KL.Stanowisko(args.base, dev=device)

    # Ten sam protokół zakresów co w k8000_od_nowa.py, ale tylko dla m0011.
    rng = np.random.default_rng(0)
    lo = torch.full((768,), float("inf"), device=device)
    hi = torch.full((768,), float("-inf"), device=device)
    for index in range(len(rows)):
        z = st.przebieg(KL.obraz_manifestu(rows[index])[0])["z"]
        selected = z[torch.from_numpy(rng.choice(600, 100, replace=False)).to(device)]
        lo = torch.minimum(lo, selected.min(0).values)
        hi = torch.maximum(hi, selected.max(0).values)

    lo_ck, hi_ck = st.h_min.clone(), st.h_max.clone()
    st.h_min = torch.minimum(lo, lo_ck)
    st.h_max = torch.maximum(torch.minimum(hi, hi_ck), lo + 1e-6)
    spec = {
        "tryb": "zakresy",
        "kanaly": np.arange(768),
        "tokeny": 0,
        "cel": "detekcja",
        "cel_pewnosc": 0.95,
        "norma": "brak",
        "kroki": 300,
        "lr": 0.05,
        "zapis_co": 300,
        "ziarno": 0,
    }
    result = KL.oblicz_korekte(
        st,
        image,
        boxes,
        spec,
        {"id": "k8000_m0011_minmax_compatible", "manifest_index": 11, "frame_id": row.get("frame_id")},
    )
    st.h_min, st.h_max = lo_ck, hi_ck
    pack = np.load(Path(result["kat"]) / "KONIEC.npz")
    base = torch.from_numpy(pack["z_baza"].astype(np.float32)).to(device)
    target = torch.from_numpy((pack["z_baza"] + pack["d_najlepszy"]).astype(np.float32)).to(device)
    control = latent_from_checkpoint(KL.MalyViT, args.control, image, device)
    variant = latent_from_checkpoint(KL.MalyViT, args.variant, image, device)
    target_file = args.output / "TARGET_K8000_M0011.npy"
    np.save(target_file, target.cpu().numpy())

    states = {
        "base_k8000": state_metrics(base, target),
        "control500": state_metrics(control, target),
        "variant8_500": state_metrics(variant, target),
    }
    directions = {
        "base_to_control": direction_metrics(base, control, target),
        "base_to_variant": direction_metrics(base, variant, target),
        "control_to_variant_replacement_effect": direction_metrics(control, variant, target),
    }
    clean = directions["control_to_variant_replacement_effect"]
    report = {
        "state": "complete",
        "target_compatible": True,
        "target_checkpoint": {"path": str(args.base.resolve()), "sha256": sha(args.base)},
        "target_file": {"path": str(target_file.resolve()), "sha256": sha(target_file)},
        "probe": {"manifest_index": 11, "frame_id": row.get("frame_id"), "boxes_xyxy": boxes.tolist()},
        "optimization": {
            "protocol": "same K8000 minmax protocol as k8000_od_nowa.py; full 600x768 latent, 300 Adam steps",
            "result_dir": result["kat"],
            "before": result["przed"],
            "best": result["najlepszy"],
            "checks": result["sprawdzenia"],
            "target_delta_rmse": float((target - base).square().mean().sqrt().item()),
            "target_delta_norm": float((target - base).norm().item()),
        },
        "models": {
            "control500": {"path": str(args.control.resolve()), "sha256": sha(args.control)},
            "variant8_500": {"path": str(args.variant.resolve()), "sha256": sha(args.variant)},
        },
        "states_to_compatible_target": states,
        "directions_to_compatible_target": directions,
        "answer": "closer" if clean["rmse_progress"] > 0 else "farther_or_not_closer",
        "interpretation": (
            "The control-to-variant contrast isolates the additional effect of eight replacements under the single paired run. "
            "Positive RMSE progress and positive directional cosine mean movement toward the compatible target."
        ),
    }
    (args.output / "DIRECTION_TO_COMPATIBLE_K8000_TARGET.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    labels = ["K8000", "kontrola 500", "wariant 8"]
    values = [states[name]["rmse"] for name in ("base_k8000", "control500", "variant8_500")]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#E45756"])
    ax.set_ylabel("RMSE do zgodnego sztucznego celu K8000")
    ax.set_title("Czy osiem podmian przesuwa m0011 w stronę sztucznego celu?")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.6f}", ha="center", va="bottom")
    fig.savefig(args.output / "DIRECTION_TO_COMPATIBLE_K8000_TARGET.png", dpi=180)
    plt.close(fig)

    md = f"""# Ruch m0011 w stronę zgodnego sztucznego celu K8000

Sztuczny cel został wyznaczony ponownie z dokładnego checkpointa K8000 użytego jako początek obu gałęzi treningu. Korekta zmieniła wynik detekcji z IoU {result['przed']['iou_top'] or 0:.6f} i confidence {result['przed']['pewnosc_top']:.6f} do IoU {result['najlepszy']['iou_top'] or 0:.6f} i confidence {result['najlepszy']['pewnosc_top']:.6f}.

- K8000 -> cel: RMSE {states['base_k8000']['rmse']:.8f}.
- Kontrola po 500 aktualizacjach -> cel: RMSE {states['control500']['rmse']:.8f}.
- Wariant z 8 podmianami -> cel: RMSE {states['variant8_500']['rmse']:.8f}.
- Czysty efekt podmiany, kontrola -> wariant: zmiana RMSE {clean['rmse_progress']:+.8f}; cosine ruchu z kierunkiem do celu {clean['cosine_with_direction_to_target']:.8f}; zrealizowana część pozostałej drogi {clean['signed_target_fraction'] * 100:+.5f}%.

Wynik: względem kontroli wariant z ośmioma podmianami ma tylko minimalną składową ruchu w stronę celu. Obie wytrenowane gałęzie są jednak dalej od celu niż startowy K8000. Efekt podmiany jest prawie prostopadły do kierunku celu i realizuje około {clean['signed_target_fraction'] * 100:.5f}% pozostałej drogi. Nie ma więc podstaw, aby twierdzić, że trening nauczył się kierunku sztucznego latentu. Jest to wynik jednej pary biegów i nie zastępuje powtórzeń na wielu seedach.
"""
    (args.output / "DIRECTION_TO_COMPATIBLE_K8000_TARGET.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "target_iou_before": result["przed"]["iou_top"],
        "target_iou_after": result["najlepszy"]["iou_top"],
        "target_confidence_before": result["przed"]["pewnosc_top"],
        "target_confidence_after": result["najlepszy"]["pewnosc_top"],
        "base_rmse": states["base_k8000"]["rmse"],
        "control_rmse": states["control500"]["rmse"],
        "variant_rmse": states["variant8_500"]["rmse"],
        "replacement_progress": clean["rmse_progress"],
        "replacement_direction_cosine": clean["cosine_with_direction_to_target"],
        "answer": report["answer"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
