"""Validated trajectory selection and exact particle-major packing."""
import re
import numpy as np


def parse_images(value, maximum=24, total=1500):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Podaj numery zdjęć, np. 12, 40-45")
    result = []
    seen = set()
    for token in value.split(","):
        token = token.strip()
        if re.fullmatch(r"\d+", token):
            numbers = [int(token)]
        elif re.fullmatch(r"\d+-\d+", token):
            start, stop = map(int, token.split("-"))
            if start > stop:
                raise ValueError("Zakres zdjęć musi rosnąć")
            numbers = range(start, stop + 1)
        else:
            raise ValueError("Niepoprawna lista zdjęć")
        for image in numbers:
            if not 0 <= image < total:
                raise ValueError("Numer zdjęcia musi należeć do 0..1499")
            if image not in seen:
                seen.add(image)
                result.append(image)
            if len(result) > maximum:
                raise ValueError(f"Pełne ślady można wczytać dla najwyżej {maximum} zdjęć")
    return result


def pack_selected(dense, images, particles_per_image=600):
    manifest = dense.load()
    if not manifest:
        raise ValueError("Brak zweryfikowanej osi trajektorii")
    steps = manifest["steps"]
    count = manifest["point_count"]
    if count % particles_per_image:
        raise ValueError("Liczba cząsteczek nie pasuje do liczby patchy obrazu")
    out = np.empty((len(images) * particles_per_image, len(steps), 4), dtype="<f4")
    for time_index, step in enumerate(steps):
        state = np.frombuffer(dense.binary(step), dtype="<f4").reshape(count, 4)
        for slot, image in enumerate(images):
            start = image * particles_per_image
            stop = start + particles_per_image
            if stop > count:
                raise ValueError("Obraz wychodzi poza zapis cząsteczek")
            out[slot * particles_per_image:(slot + 1) * particles_per_image, time_index] = state[start:stop]
    if not np.isfinite(out).all():
        raise ValueError("Trajektorie zawierają wartości niefinitywne")
    return out, {"images": list(images), "steps": list(steps), "particles": len(images) * particles_per_image}


def selected_payload(dense, value, particles_per_image=600):
    images = parse_images(value)
    packed, meta = pack_selected(dense, images, particles_per_image)
    manifest = dense.load()
    positions = packed[:, :, :3].reshape(-1, 3)
    low, high = positions.min(axis=0), positions.max(axis=0)
    pad = np.maximum((high - low) * .04, .03)
    meta.update(shape=list(packed.shape), sampling=False, dtype="float32-le",
                bounds=np.column_stack((low - pad, high + pad)).round(5).tolist(),
                norm_scale=manifest.get("norm_scale"))
    return packed.tobytes(order="C"), meta
