from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as M


W, H = 480, 320
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path: Path, device: torch.device) -> M.MalyViT:
    net = M.MalyViT().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    net.zapisuj_latent(True)
    return net


def load_images(data: Path, ids: list[str], by_id: dict[str, dict]) -> np.ndarray:
    images = []
    for image_id in ids:
        row = by_id[image_id]
        image = cv2.imread(str(data / row["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(data / row["file"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA))
    return np.stack(images)


@torch.inference_mode()
def latents(models: dict[str, M.MalyViT], images: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    result = {}
    for name, net in models.items():
        _, latent = net(x)
        if latent is None or latent.shape[1:] != (600, 768):
            raise RuntimeError(f"Nieoczekiwany latent {name}: {None if latent is None else tuple(latent.shape)}")
        result[name] = latent.detach().float()
    return result


def pair_per_image(a: torch.Tensor, b: torch.Tensor) -> dict[str, np.ndarray]:
    d = b - a
    dims = (1, 2)
    rmse = d.square().mean(dims).sqrt()
    mae = d.abs().mean(dims)
    ref_rms = a.square().mean(dims).sqrt()
    cosine = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
    token_cosine = torch.nn.functional.cosine_similarity(a, b, dim=2).mean(1)
    return {
        "rmse": rmse.cpu().numpy(),
        "mae": mae.cpu().numpy(),
        "relative_rms": (rmse / ref_rms.clamp_min(EPS)).cpu().numpy(),
        "cosine": cosine.cpu().numpy(),
        "mean_token_cosine": token_cosine.cpu().numpy(),
    }


def movement_per_image(base: torch.Tensor, control: torch.Tensor, variant: torch.Tensor) -> dict[str, np.ndarray]:
    dc = control - base
    dv = variant - base
    residual = variant - control
    flat_dc = dc.flatten(1)
    flat_dv = dv.flatten(1)
    flat_r = residual.flatten(1)
    rms_dc = flat_dc.square().mean(1).sqrt()
    rms_dv = flat_dv.square().mean(1).sqrt()
    rms_r = flat_r.square().mean(1).sqrt()
    return {
        "cos_control_vs_variant_movement": torch.nn.functional.cosine_similarity(flat_dc, flat_dv, dim=1).cpu().numpy(),
        "cos_replacement_vs_control_movement": torch.nn.functional.cosine_similarity(flat_r, flat_dc, dim=1).cpu().numpy(),
        "replacement_rms_over_control_rms": (rms_r / rms_dc.clamp_min(EPS)).cpu().numpy(),
        "replacement_rms_over_variant_rms": (rms_r / rms_dv.clamp_min(EPS)).cpu().numpy(),
    }


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def pair_subset(a: torch.Tensor, b: torch.Tensor, token_mask: torch.Tensor) -> dict[str, float]:
    aa = a[:, token_mask]
    bb = b[:, token_mask]
    d = bb - aa
    return {
        "tokens": int(token_mask.sum().item()),
        "rmse": float(d.square().mean().sqrt().item()),
        "mae": float(d.abs().mean().item()),
        "relative_rms": float((d.square().mean().sqrt() / aa.square().mean().sqrt().clamp_min(EPS)).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(aa.flatten(), bb.flatten(), dim=0).item()),
        "mean_token_cosine": float(torch.nn.functional.cosine_similarity(aa, bb, dim=2).mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--guard-limit", type=int, default=512)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = json.loads((args.data / "pool.json").read_text(encoding="utf-8"))
    records = json.loads((args.data / "records.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in records}
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    removed = comparison["combined_8"]["removed_ids"]
    inserted = comparison["combined_8"]["inserted_ids"]

    models = {
        "base_k8000": load_model(args.base, device),
        "control500": load_model(args.control, device),
        "variant8_500": load_model(args.variant, device),
    }
    checkpoint_info = {
        "base_k8000": {"path": str(args.base.resolve()), "sha256": sha256(args.base)},
        "control500": {"path": str(args.control.resolve()), "sha256": sha256(args.control)},
        "variant8_500": {"path": str(args.variant.resolve()), "sha256": sha256(args.variant)},
    }

    groups = {
        "guard_512": list(pool["guard_512"][: args.guard_limit]),
        "inserted_8": list(inserted),
        "removed_8": list(removed),
    }
    pair_names = {
        "base_to_control": ("base_k8000", "control500"),
        "base_to_variant8": ("base_k8000", "variant8_500"),
        "control_to_variant8_replacement_effect": ("control500", "variant8_500"),
    }

    group_results: dict[str, dict] = {}
    per_image_rows: list[dict] = []
    guard_axis_sums = {
        "control": np.zeros(768, np.float64),
        "variant": np.zeros(768, np.float64),
        "replacement": np.zeros(768, np.float64),
    }
    guard_axis_count = 0

    for group_name, ids in groups.items():
        pair_values = {name: {k: [] for k in ("rmse", "mae", "relative_rms", "cosine", "mean_token_cosine")} for name in pair_names}
        move_values = {k: [] for k in (
            "cos_control_vs_variant_movement",
            "cos_replacement_vs_control_movement",
            "replacement_rms_over_control_rms",
            "replacement_rms_over_variant_rms",
        )}
        for start in range(0, len(ids), args.batch):
            batch_ids = ids[start : start + args.batch]
            image_batch = load_images(args.data, batch_ids, by_id)
            z = latents(models, image_batch, device)
            batch_pairs = {}
            for pair_name, (left, right) in pair_names.items():
                batch_pairs[pair_name] = pair_per_image(z[left], z[right])
                for key, values in batch_pairs[pair_name].items():
                    pair_values[pair_name][key].append(values)
            batch_move = movement_per_image(z["base_k8000"], z["control500"], z["variant8_500"])
            for key, values in batch_move.items():
                move_values[key].append(values)
            if group_name == "guard_512":
                dc = z["control500"] - z["base_k8000"]
                dv = z["variant8_500"] - z["base_k8000"]
                dr = z["variant8_500"] - z["control500"]
                guard_axis_sums["control"] += dc.square().sum((0, 1)).cpu().numpy()
                guard_axis_sums["variant"] += dv.square().sum((0, 1)).cpu().numpy()
                guard_axis_sums["replacement"] += dr.square().sum((0, 1)).cpu().numpy()
                guard_axis_count += int(dc.shape[0] * dc.shape[1])
            for index, image_id in enumerate(batch_ids):
                row = {"group": group_name, "id": image_id}
                for pair_name in pair_names:
                    for metric, values in batch_pairs[pair_name].items():
                        row[f"{pair_name}.{metric}"] = float(values[index])
                for metric, values in batch_move.items():
                    row[f"movement.{metric}"] = float(values[index])
                per_image_rows.append(row)
            del z
        group_results[group_name] = {
            "images": len(ids),
            "unique_films": len({by_id[image_id]["film"] for image_id in ids}),
            "pairs": {
                pair_name: {metric: describe(np.concatenate(chunks)) for metric, chunks in metrics.items()}
                for pair_name, metrics in pair_values.items()
            },
            "movement": {metric: describe(np.concatenate(chunks)) for metric, chunks in move_values.items()},
        }

    probe_cfg = json.loads((args.data / "probe.json").read_text(encoding="utf-8"))
    probe_bgr = cv2.imread(str(args.data / probe_cfg["file"]), cv2.IMREAD_COLOR)
    if probe_bgr is None:
        raise FileNotFoundError(args.data / probe_cfg["file"])
    probe_rgb = cv2.cvtColor(probe_bgr, cv2.COLOR_BGR2RGB)
    probe_rgb = cv2.resize(probe_rgb, (W, H), interpolation=cv2.INTER_AREA)[None]
    probe_z = latents(models, probe_rgb, device)
    x0, y0, x1, y1 = probe_cfg["gt_xyxy_480x320"][0]
    yy, xx = torch.meshgrid(torch.arange(H // M.KROK), torch.arange(W // M.KROK), indexing="ij")
    centers_x = (xx.flatten() + 0.5) * M.KROK
    centers_y = (yy.flatten() + 0.5) * M.KROK
    roi_mask = ((centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)).to(device)
    outside_mask = ~roi_mask
    probe_result = {
        "id": probe_cfg["id"],
        "gt_xyxy": [x0, y0, x1, y1],
        "pairs_full": {},
        "pairs_roi": {},
        "pairs_outside_roi": {},
        "movement": {key: float(value[0]) for key, value in movement_per_image(
            probe_z["base_k8000"], probe_z["control500"], probe_z["variant8_500"]
        ).items()},
    }
    for pair_name, (left, right) in pair_names.items():
        probe_result["pairs_full"][pair_name] = {
            key: float(value[0]) for key, value in pair_per_image(probe_z[left], probe_z[right]).items()
        }
        probe_result["pairs_roi"][pair_name] = pair_subset(probe_z[left], probe_z[right], roi_mask)
        probe_result["pairs_outside_roi"][pair_name] = pair_subset(probe_z[left], probe_z[right], outside_mask)

    axis_rms = {name: np.sqrt(values / guard_axis_count) for name, values in guard_axis_sums.items()}
    residual_energy = axis_rms["replacement"] ** 2
    order = np.argsort(residual_energy)[::-1]
    axis_summary = {
        "guard_token_count": guard_axis_count,
        "top_20_replacement_axes": [
            {
                "axis": int(j),
                "replacement_rms": float(axis_rms["replacement"][j]),
                "control_rms": float(axis_rms["control"][j]),
                "variant_rms": float(axis_rms["variant"][j]),
            }
            for j in order[:20]
        ],
        "replacement_energy_share_top_128_axes": float(residual_energy[order[:128]].sum() / residual_energy.sum()),
    }

    residual_per_image = np.asarray([
        row["control_to_variant8_replacement_effect.rmse"] for row in per_image_rows if row["group"] == "guard_512"
    ])
    probe_result["replacement_rmse_percentile_among_guard_512"] = float(
        100.0 * np.mean(residual_per_image <= probe_result["pairs_full"]["control_to_variant8_replacement_effect"]["rmse"])
    )

    report = {
        "state": "complete",
        "device": str(device),
        "latent_definition": "bloki[-1].mlp.act po GELU, przed fc2; 600 tokenow x 768 wspolrzednych",
        "causal_contrast": "control500 i variant8_500: ten sam K8000, seed, kolejnosc, optymalizator i 500 aktualizacji; roznica to 8 podmienionych obrazow",
        "checkpoints": checkpoint_info,
        "groups": group_results,
        "probe_m0011": probe_result,
        "axis_summary": axis_summary,
        "limitations": [
            "To mierzy zmiane naturalnego latentu wywolywanego przez te same obrazy, bez recznej podmiany aktywacji.",
            "Nie ocenia kierunku do starego sztucznego celu m0011, bo zapisany cel pochodzi z K1000 i nie ma potwierdzonej zgodnosci reprezentacji z K8000.",
            "Jest to jedna deterministyczna para biegow; niepewnosc miedzy seedami wymaga powtorzen treningu.",
        ],
    }
    (args.output / "LATENT_COMPARISON.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if per_image_rows:
        with (args.output / "LATENT_PER_IMAGE.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0]))
            writer.writeheader()
            writer.writerows(per_image_rows)

    guard = report["groups"]["guard_512"]
    labels = ["K8000→kontrola", "K8000→8 podmian", "kontrola→8 podmian"]
    keys = ["base_to_control", "base_to_variant8", "control_to_variant8_replacement_effect"]
    rms_values = [guard["pairs"][key]["rmse"]["mean"] for key in keys]
    cos_values = [guard["pairs"][key]["cosine"]["mean"] for key in keys]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#E45756"]
    axes[0, 0].bar(labels, rms_values, color=colors)
    axes[0, 0].set_title("Średni ruch pełnego latentu na 512 obrazach")
    axes[0, 0].set_ylabel("RMSE współrzędnych")
    axes[0, 0].tick_params(axis="x", rotation=14)
    for i, value in enumerate(rms_values):
        axes[0, 0].text(i, value, f"{value:.5f}", ha="center", va="bottom")

    axes[0, 1].bar(labels, [1.0 - value for value in cos_values], color=colors)
    axes[0, 1].set_title("Zmiana kierunku reprezentacji")
    axes[0, 1].set_ylabel("1 − cosine")
    axes[0, 1].tick_params(axis="x", rotation=14)

    x = np.arange(768)
    axes[1, 0].plot(x, axis_rms["control"], label="K8000→kontrola", linewidth=1.1)
    axes[1, 0].plot(x, axis_rms["variant"], label="K8000→8 podmian", linewidth=1.1)
    axes[1, 0].plot(x, axis_rms["replacement"], label="efekt 8 podmian", linewidth=1.4)
    axes[1, 0].set_title("RMS zmiany każdej z 768 współrzędnych")
    axes[1, 0].set_xlabel("Indeks osi latentu")
    axes[1, 0].set_ylabel("RMS zmiany")
    axes[1, 0].legend()

    axes[1, 1].hist(residual_per_image, bins=32, color="#E45756", alpha=0.85)
    probe_residual = probe_result["pairs_full"]["control_to_variant8_replacement_effect"]["rmse"]
    axes[1, 1].axvline(probe_residual, color="black", linestyle="--", label=f"m0011: {probe_residual:.5f}")
    axes[1, 1].set_title("Efekt podmiany ośmiu zdjęć na poszczególnych obrazach")
    axes[1, 1].set_xlabel("RMSE: kontrola → wariant 8")
    axes[1, 1].set_ylabel("Liczba obrazów")
    axes[1, 1].legend()
    fig.suptitle("Zmiana naturalnego latentu po 500 aktualizacjach", fontsize=16, fontweight="bold")
    fig.savefig(args.output / "LATENT_POROWNANIE.png", dpi=180)
    plt.close(fig)

    g_pairs = guard["pairs"]
    effect = g_pairs["control_to_variant8_replacement_effect"]
    p_effect = probe_result["pairs_full"]["control_to_variant8_replacement_effect"]
    p_roi = probe_result["pairs_roi"]["control_to_variant8_replacement_effect"]
    p_out = probe_result["pairs_outside_roi"]["control_to_variant8_replacement_effect"]
    md = f"""# Porównanie latentów K8000, kontroli i wariantu z 8 podmianami

## Odpowiedź

Tak: naturalny latent zaczął się zmieniać po dalszym treningu, a osiem podmienionych zdjęć wywołało dodatkową, mierzalną zmianę względem gałęzi kontrolnej. Najczystsza miara wpływu podmiany to porównanie `control500 → variant8_500`, ponieważ oba modele zaczynały z tego samego K8000 i wykonały tyle samo aktualizacji.

## 512 odłożonych obrazów

- Zwykły dalszy trening, `K8000 → kontrola`: średni RMSE = {g_pairs['base_to_control']['rmse']['mean']:.8f}, cosine = {g_pairs['base_to_control']['cosine']['mean']:.8f}.
- Trening z 8 podmianami, `K8000 → wariant`: średni RMSE = {g_pairs['base_to_variant8']['rmse']['mean']:.8f}, cosine = {g_pairs['base_to_variant8']['cosine']['mean']:.8f}.
- Dodatkowy efekt samych 8 podmian, `kontrola → wariant`: średni RMSE = {effect['rmse']['mean']:.8f}, cosine = {effect['cosine']['mean']:.8f}.
- Norma dodatkowego ruchu stanowi średnio {guard['movement']['replacement_rms_over_control_rms']['mean'] * 100:.2f}% normy ruchu kontrolnego.
- Kierunek całkowitego ruchu kontroli i wariantu ma średni cosine {guard['movement']['cos_control_vs_variant_movement']['mean']:.8f}.
- Na samych ośmiu dodanych obrazach dodatkowy RMSE wynosi {report['groups']['inserted_8']['pairs']['control_to_variant8_replacement_effect']['rmse']['mean']:.8f}, czyli {report['groups']['inserted_8']['pairs']['control_to_variant8_replacement_effect']['rmse']['mean'] / effect['rmse']['mean']:.2f} razy więcej niż średnio na odłożonych obrazach.
- Najsilniejsze 128 z 768 osi niesie {axis_summary['replacement_energy_share_top_128_axes'] * 100:.2f}% energii dodatkowej zmiany; są to osie wybrane po pomiarze, a nie wcześniejszy zestaw J128.

## Obraz-sonda m0011

- Dodatkowy efekt 8 podmian: RMSE = {p_effect['rmse']:.8f}, cosine = {p_effect['cosine']:.8f}.
- W tokenach wewnątrz ramki narzędzia: RMSE = {p_roi['rmse']:.8f} ({p_roi['tokens']} tokenów).
- Poza ramką: RMSE = {p_out['rmse']:.8f} ({p_out['tokens']} tokenów).
- Stosunek ruchu dodatkowego do ruchu kontrolnego: {probe_result['movement']['replacement_rms_over_control_rms'] * 100:.2f}%.
- Wielkość dodatkowej zmiany m0011 leży na około {probe_result['replacement_rmse_percentile_among_guard_512']:.0f}. percentylu rozkładu 512 odłożonych obrazów.

## Interpretacja

Latent nie pozostał stały. Większość przesunięcia wynika jednak ze wspólnych 500 aktualizacji na prawie tym samym zbiorze. Różnica `kontrola → wariant` pokazuje mniejszy, ale realny wpływ ośmiu podmienionych obrazów. Ten pomiar nie mówi jeszcze, że latent przesunął się w stronę starego sztucznego celu m0011, ponieważ tamten cel pochodzi z K1000, a tutaj badamy K8000.
"""
    (args.output / "LATENT_ANALIZA.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "guard_replacement_rmse": effect["rmse"]["mean"],
        "guard_replacement_cosine": effect["cosine"]["mean"],
        "probe_replacement_rmse": p_effect["rmse"],
        "probe_roi_rmse": p_roi["rmse"],
        "probe_outside_rmse": p_out["rmse"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
