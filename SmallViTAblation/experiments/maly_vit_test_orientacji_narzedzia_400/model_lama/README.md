---
license: apache-2.0
library_name: onnx
pipeline_tag: image-to-image
tags:
- lama
- inpainting
- object-removal
- onnx
---

# LaMa (ONNX) — object-removal inpainting

`lama_fp32.onnx` — an ONNX export of **LaMa** (big-lama) for image inpainting
(erasing a masked object and filling the hole). Runs client-side with
onnxruntime-web on the WASM or WebGPU execution providers.

A verbatim mirror of
[`Carve/LaMa-ONNX`](https://huggingface.co/Carve/LaMa-ONNX)'s `lama_fp32.onnx`,
hosted so a single, immutable, CORS-enabled copy backs the Edge Tools
"Remove Object" tool.

## Inputs / outputs

- **Inputs:** `image` float32 `[1, 3, 512, 512]`, `mask` float32 `[1, 1, 512, 512]`
- **Output:** `output` float32 `[1, 3, 512, 512]` — the inpainted image
- RGB, channels-first (CHW). Image values in 0..1; **mask polarity: `1` = erase
  (the hole), `0` = keep.** The output is the raw fill (0..255), composited over
  the original by the caller so only the hole changes. Spatial size is **fixed at
  512×512** (resize to it, then back).

## Provenance & license

- **Algorithm / weights:** **LaMa** — *Resolution-robust Large Mask Inpainting
  with Fourier Convolutions* (Suvorov et al., WACV 2022),
  [advimman/lama](https://github.com/advimman/lama). The `big-lama` weights are
  released under the **Apache-2.0 license**.
- **ONNX export:**
  [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`).
- The full **Apache-2.0 `LICENSE`** is included in this repo.
