"""Rozstrzyga, jak SD K8000 reaguje na liniowe cofanie sztucznego latentu K1000 m0011.

Test obali tezę o widocznej reakcji na warunek, jeśli przy identycznym szumie obrazy
pozostaną praktycznie takie same. Nie rozstrzyga wierności względem przestrzeni K8000,
ponieważ wejściowe latenty zostały wyznaczone przez checkpoint K1000.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "latent_cel_do_obrazow"
RUN = SOURCE / "sd_k8000_m0011_interpolacja"
ASSET = ROOT / "assets" / "SD_ViT_K8000_A_control_K10500"
MODEL = ASSET / "best_sd.pt"
MODEL_SHA256 = "1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41"
K1000_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
BASE_NPZ = SOURCE / "korekty" / "20261008_004248_zakresy_detekcja_m0011" / "KONIEC.npz"
BASE_CONFIG = BASE_NPZ.with_name("CONFIG.json")
ARTIFICIAL = SOURCE / "zb_m0011_minmax.npy"
SEED = 20261008
STEPS = 50
START = time.time()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def status(phase: str, **values) -> None:
    record = dict(phase=phase, elapsed_seconds=time.time() - START, pid=os.getpid(), **values)
    write_json(RUN / "STATUS.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def finite(tensor, label: str):
    import torch
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Nieskończone wartości: {label}")
    return tensor


def save_image(path: Path, tensor) -> dict:
    import numpy as np
    from PIL import Image
    array = tensor[0].permute(1, 2, 0).float().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray(np.uint8(np.rint(array * 255))).save(path)
    return dict(mean=float(array.mean()), std=float(array.std()), minimum=float(array.min()), maximum=float(array.max()))


def make_contact_sheet(paths: list[Path], labels: list[str], destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    columns, label_height = 3, 34
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 7), label, fill="black", font=font)
    sheet.save(destination, quality=94)
    for image in images:
        image.close()


def make_gallery(records: list[dict]) -> None:
    cards = "".join(
        f'<figure><figcaption>k={r["k"]}, przyrost={100*r["alpha"]:.0f}%</figcaption>'
        f'<img src="{r["file"]}"><p>std={r["image"]["std"]:.4f}</p></figure>'
        for r in records
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>SD K8000 · m0011</title>
<style>body{{font:18px Arial;background:#f5f1dd;color:#193239;margin:28px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{font-weight:bold;margin:8px 0}}.warn{{max-width:1100px;background:#fff0bf;padding:14px}}</style>
<h1>SD K8000: od sztucznego latentu m0011 do oryginału</h1>
<p class="warn">Warunek pochodzi z ViT K1000, a generator był uczony na K8000. To test wejścia poza potwierdzonym rozkładem. Wszystkie obrazy zaczynają od identycznego szumu.</p>
<p>k=0: pełny sztuczny przyrost. Każdy kolejny krok odejmuje 1/10 przyrostu. k=10: dokładny latent oryginalny.</p>
<div class="grid">{cards}</div>"""
    (RUN / "GALERIA.html").write_text(html, encoding="utf-8")


def main() -> None:
    import numpy as np
    import torch
    from diffusers import DDIMScheduler

    RUN.mkdir(parents=True, exist_ok=True)
    status("verify_inputs")
    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError("Niezgodny checkpoint generatora")
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8-sig"))
    if config["checkpoint_sha256"] != K1000_SHA256:
        raise ValueError("Niezgodne pochodzenie latentu bazowego")
    archive = np.load(BASE_NPZ)
    original = np.asarray(archive["z_baza"], dtype=np.float32)
    artificial = np.asarray(np.load(ARTIFICIAL), dtype=np.float32)
    if original.shape != (600, 768) or artificial.shape != original.shape:
        raise ValueError(f"Nieprawidłowe kształty: {original.shape}, {artificial.shape}")
    if not np.isfinite(original).all() or not np.isfinite(artificial).all():
        raise FloatingPointError("Latenty zawierają wartości niefinitywne")
    delta = artificial - original
    changed = int(np.count_nonzero(delta))
    conditions = [artificial - np.float32(k / 10) * delta for k in range(11)]
    if not np.array_equal(conditions[0], artificial):
        raise AssertionError("Początek nie jest sztucznym latentem")
    if not np.array_equal(conditions[10], original):
        # Konstrukcja odejmowania może dać ostatni bit inny; endpoint ustawiamy jawnie.
        conditions[10] = original.copy()

    sys.path.insert(0, str(ASSET / "kod"))
    from model import Denoiser, new_vae, schedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Ten bieg wymaga lokalnego CUDA")
    torch.set_num_threads(4)
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    if checkpoint["step"] != 10500 or checkpoint["signature"]["arm"] != "A_control":
        raise ValueError("Niezgodne metadane checkpointa")
    net, vae = Denoiser().eval(), new_vae().eval()
    net.load_state_dict(checkpoint["net"], strict=True)
    vae.load_state_dict(checkpoint["vae"], strict=True)
    net.requires_grad_(False)
    vae.requires_grad_(False)
    scale = float(checkpoint["scale"])
    del checkpoint

    scheduler_config = schedule().config
    initial_noise = torch.randn((1, 4, 48, 72), generator=torch.Generator(device=device).manual_seed(SEED), device=device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    final_states = []
    net.to(device)
    with torch.inference_mode():
        for k, condition_np in enumerate(conditions):
            status("generating", k=k, alpha=1 - k / 10)
            scheduler = DDIMScheduler.from_config(scheduler_config, timestep_spacing="leading")
            scheduler.set_timesteps(STEPS, device=device)
            state = initial_noise.clone() * scheduler.init_noise_sigma
            condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
            for timestep in scheduler.timesteps:
                with torch.autocast("cuda", dtype=amp_dtype):
                    prediction = net(scheduler.scale_model_input(state, timestep), timestep, condition)
                state = finite(scheduler.step(prediction.float(), timestep, state, eta=0.0).prev_sample, "DDIM")
            final_states.append(state.cpu())
            del condition, prediction, state
    net.cpu()
    del net, initial_noise
    gc.collect()
    torch.cuda.empty_cache()

    vae.to(device)
    records, image_paths, labels, previous = [], [], [], None
    with torch.inference_mode():
        for k, state in enumerate(final_states):
            alpha = 1 - k / 10
            status("decoding", k=k, alpha=alpha)
            image = ((finite(vae.decode(state.to(device) / scale).sample.float(), "VAE") + 1) / 2).clamp(0, 1).cpu()
            suffix = "sztuczny" if k == 0 else "oryginalny_latent" if k == 10 else f"minus_{k}_dziesiatych"
            path = RUN / f"k{k:02d}_przyrost_{int(round(alpha*100)):03d}pct_{suffix}.png"
            stats = save_image(path, image)
            change = None if previous is None else float((image - previous).abs().mean())
            records.append(dict(k=k, alpha=alpha, removed_fraction=k / 10, file=path.name,
                                image=stats, mean_absolute_change_from_previous=change))
            image_paths.append(path)
            labels.append(f"k={k} | przyrost {int(round(alpha*100))}%")
            previous = image
    vae.cpu()
    del vae, final_states, previous
    gc.collect()
    torch.cuda.empty_cache()

    reference = RUN / "referencja_m0011.png"
    from PIL import Image
    Image.fromarray(np.asarray(archive["obraz"], dtype=np.uint8)).save(reference)
    make_contact_sheet(image_paths, labels, RUN / "PASEK_11_KROKOW.jpg")
    make_gallery(records)
    result = dict(
        state="complete",
        model=str(MODEL), model_sha256=MODEL_SHA256, model_step=10500,
        source_image="m0011", source_reference_file=reference.name,
        original_latent_source=str(BASE_NPZ), artificial_latent_source=str(ARTIFICIAL),
        latent_provenance_checkpoint="K1000", latent_provenance_sha256=K1000_SHA256,
        generator_training_checkpoint="K8000",
        compatibility="shape-compatible but representation compatibility is not established",
        formula="z_k = z_artificial - (k/10) * (z_artificial-z_original), k=0..10",
        latent_shape=list(original.shape), changed_coordinates=changed,
        delta=dict(norm=float(np.linalg.norm(delta)), minimum=float(delta.min()), maximum=float(delta.max()),
                   mean=float(delta.mean()), std=float(delta.std()), one_tenth_norm=float(np.linalg.norm(delta) / 10)),
        endpoint_checks=dict(k0_exact_artificial=bool(np.array_equal(conditions[0], artificial)),
                             k10_exact_original=bool(np.array_equal(conditions[10], original))),
        sampling=dict(seed=SEED, ddim_steps=STEPS, spacing="leading", eta=0, same_initial_noise=True,
                      direct_conditioning=True, img2img=False),
        generated_images=len(records), records=records,
        caveat="This does not validate inversion of a K8000 artificial latent because both endpoints came from K1000.",
    )
    write_json(RUN / "WYNIK.json", result)
    status("complete", generated_images=len(records), gallery=str(RUN / "GALERIA.html"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            RUN.mkdir(parents=True, exist_ok=True)
            status("failed", error_type=type(error).__name__, error=str(error))
        finally:
            raise

