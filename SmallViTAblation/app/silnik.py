"""Silnik edytora latentu: wytrenowany SD z checkpointu, DDIM z postępem, naturalny latent z ViT.

Warunek dla UNet to pełny latent 600x768 (tokeny x kanały), dokładnie jak w treningu
(`vit_sd_pelny_latent_filmy_eden_20261006`). Ten moduł tylko ładuje modele i liczy;
zapisem obrazów/latentów zajmuje się warstwa GUI (app.py).
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import sys
from pathlib import Path
import numpy as np

# torch importowany dopiero w funkcjach: start okna nie czeka na PyTorch (kilka sekund na zimnym dysku)

ROOT = Path(__file__).resolve().parent.parent          # .../ViTParticleFilterTracker/SmallViTAblation (dowolny dysk)
EKSPERYMENT_SD = ROOT/'experiments/vit_sd_pelny_latent_filmy_eden_20261006'
sys.path.insert(0, str(EKSPERYMENT_SD))

CHECKPOINT = ROOT/'assets/StableDiffusionModel/best_checkpoint.pt'
SD_ROOT = ROOT.parents[1]/'eden_archiwum/sd_modele/ema_020000'
VAE_ROOT = ROOT.parents[1]/'eden_archiwum/sd_modele/sd-vae-ft-mse'
VIT_CHECKPOINT = ROOT/'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
DOMYSLNY_LATENT = ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
WYJSCIE = ROOT/'outputs/latent_editor_sd_20261006'

TOKENY = 600
KANALY = 768

KANALY_NARZEDZIA_CSV = ROOT/'outputs/vit_latent128_jedno_zdjecie_90/KANALY128_PODSUMOWANIE.csv'


def kanaly_narzedzia():
    """128 kanałów tożsamości narzędzia (J128) z eksperymentu vit_latent128_jedno_zdjecie_90.

    Sortowane wg delta_rms malejąco (jak tabela w RAPORT.md). Zwraca listę dict:
    kanal (0-based), delta_rms, before_mean, after_mean, delta_roi_mean, after_max.
    after_max służy do poszerzenia zakresu suwaka (zmiany J128 sięgały ~50).
    """
    import csv
    wiersze = []
    try:
        with open(KANALY_NARZEDZIA_CSV, newline='', encoding='utf-8-sig') as f:
            for w in csv.DictReader(f):
                wiersze.append(dict(kanal=int(w['channel0']),
                                    delta_rms=float(w['delta_rms']),
                                    before_mean=float(w['before_mean']),
                                    after_mean=float(w['after_mean']),
                                    delta_roi_mean=float(w['delta_roi_mean']),
                                    after_max=float(w['after_max'])))
    except OSError:
        dane = np.load(DOMYSLNY_LATENT)
        wiersze = [dict(kanal=int(c), delta_rms=0.0, before_mean=float('nan'),
                        after_mean=float('nan'), delta_roi_mean=float('nan'),
                        after_max=1.0) for c in sorted(dane['channels0'].tolist())]
    wiersze.sort(key=lambda w: -w['delta_rms'])
    return wiersze


def zaladuj_model(urzadzenie='cuda'):
    """Bazowy zamrożony UNet/VAE + wytrenowane attn2 i adapter z checkpointu.

    load_trained weryfikuje sha256 bazowych wag UNet/VAE względem sygnatury
    checkpointu, więc pierwsze ładowanie czyta kilka GB z dysku.
    """
    from generate import load_trained
    trained = load_trained(str(SD_ROOT), str(VAE_ROOT), str(CHECKPOINT), device=urzadzenie)
    trained.vae.enable_tiling()
    return trained


def obraz_startowy(sciezka):
    """Zdjęcie startowe dla img2img: RGB float32 [3,384,576] w [0,1], skalowane INTER_AREA jak w treningu SD."""
    import cv2
    from PIL import Image
    rgb = np.asarray(Image.open(sciezka).convert('RGB'))
    rgb = cv2.resize(rgb, (576, 384), interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255).transpose(2, 0, 1).copy()


def _dekoduj(trained, sample):
    import torch
    with torch.autocast(trained.device.type, enabled=False):
        rgb = (trained.vae.decode(sample.float() / float(trained.vae.config.scaling_factor)).sample + 1) / 2
    return rgb.float().clamp(0, 1)[0].cpu().numpy()


def generuj(*a, **k):
    """`_generuj` w torch.inference_mode (opis tam)."""
    import torch
    with torch.inference_mode():
        return _generuj(*a, **k)


def _generuj(trained, latent, kroki=50, ziarno=0, postep=None, start=None, sila=1.0,
             podglad=None, podglad_co=0, podglad_x0=False):
    """DDIM (eta=0) z warunku 600x768 do RGB; postep(i, n) po każdym kroku.

    start=None albo sila>=1: zgodne 1:1 z generate.generate (czysty szum, wszystkie `kroki`).
    start = obraz [3,384,576] w [0,1] (img2img): VAE koduje zdjęcie (średnia rozkładu, FP32, jak
    w treningu obraz*2-1 i * scaling_factor), szum DDPM dodawany do kroku czasowego odpowiadającego
    `sila`, potem odszumianie przez ostatnie round(kroki*sila) kroków DDIM tego samego harmonogramu.
    podglad(i, n, rgb) co `podglad_co` kroków (i=0: zaszumiony start): bieżący stan z szumem
    albo, gdy podglad_x0, przewidywany obraz końcowy (pred_original_sample).
    Zwraca ndarray float32 [3,384,576] w [0,1].
    """
    import torch
    from diffusers import DDIMScheduler
    warunek = torch.as_tensor(np.asarray(latent), dtype=torch.float32)[None]
    if warunek.shape[1:] != (TOKENY, KANALY) or not torch.isfinite(warunek).all():
        raise ValueError(f'Oczekiwano skończonego latentu 600x768, jest {tuple(warunek.shape)}')
    if not 1 <= int(kroki) <= 1000:
        raise ValueError('kroki muszą być w [1,1000]')
    device = trained.device
    scheduler = DDIMScheduler.from_config(trained.scheduler_config, clip_sample=False,
                                          set_alpha_to_one=False,
                                          steps_offset=0 if kroki == 1000 else 1)
    scheduler.set_timesteps(int(kroki), device=device)
    rng = torch.Generator(device=device).manual_seed(int(ziarno))
    szum = torch.randn((1, 4, 48, 72), device=device, dtype=torch.float32, generator=rng)
    kroki_czasu = scheduler.timesteps
    if start is None or float(sila) >= 1.0:
        sample = szum * scheduler.init_noise_sigma
    else:
        if not 0.0 < float(sila) < 1.0:
            raise ValueError('siła szumu musi być w (0, 1]')
        x = torch.as_tensor(np.asarray(start), dtype=torch.float32)[None].to(device)
        if x.shape[1:] != (3, 384, 576):
            raise ValueError(f'Zdjęcie startowe musi mieć 3x384x576, jest {tuple(x.shape)}')
        with torch.autocast(device.type, enabled=False):
            x0 = trained.vae.encode(x * 2 - 1).latent_dist.mean * float(trained.vae.config.scaling_factor)
        n = max(1, min(int(kroki), round(int(kroki) * float(sila))))
        kroki_czasu = kroki_czasu[int(kroki) - n:]
        sample = scheduler.add_noise(x0, szum, kroki_czasu[:1])
    n = len(kroki_czasu)
    if podglad is not None and podglad_co:
        podglad(0, n, _dekoduj(trained, sample))
    warunek = warunek.to(device)
    for i, t in enumerate(kroki_czasu):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            pred = trained.model(scheduler.scale_model_input(sample, t), t, warunek)
        wynik = scheduler.step(pred.float(), t, sample, eta=0.)
        sample = wynik.prev_sample
        if postep is not None:
            postep(i + 1, n)
        if podglad is not None and podglad_co and (i + 1) % int(podglad_co) == 0 and i + 1 < n:
            podglad(i + 1, n, _dekoduj(trained, wynik.pred_original_sample if podglad_x0 else sample))
    return _dekoduj(trained, sample)


def latent_z_obrazu(sciezka, urzadzenie='cuda'):
    """Naturalny latent ViT 600x768 z dowolnego obrazu (skalowany do 480x320 jak w treningu).

    ViT jest ładowany leniwo i zwalniany po użyciu, żeby nie trzymać go w VRAM.
    """
    import cv2
    from PIL import Image
    import torch
    from model import load_vit, vit_forward
    vit = load_vit(str(VIT_CHECKPOINT), torch.device(urzadzenie))
    try:
        obraz = np.asarray(Image.open(sciezka).convert('RGB')).astype(np.float32) / 255
        obraz = cv2.resize(obraz, (480, 320), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(obraz).permute(2, 0, 1)[None].to(urzadzenie)
        _, latent = vit_forward(vit, x)
        return latent[0].float().cpu().numpy()
    finally:
        del vit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def wczytaj_latent_npz(sciezka):
    """Pierwsza tablica 600x768 z NPZ; preferuje klucze latent/after/before. Zwraca (latent, klucz)."""
    dane = np.load(sciezka)
    for klucz in ('latent', 'after', 'before'):
        if klucz in dane and dane[klucz].shape == (TOKENY, KANALY):
            return dane[klucz].astype(np.float32), klucz
    for klucz in dane.files:
        if dane[klucz].shape == (TOKENY, KANALY):
            return dane[klucz].astype(np.float32), klucz
    raise ValueError(f'Brak tablicy 600x768 w {sciezka}; klucze: {dane.files}')


def domyslna_baza():
    """Naturalny latent `before` ze zdjęcia źródłowego starego eksperymentu; zera gdy brak pliku."""
    if DOMYSLNY_LATENT.exists():
        dane = np.load(DOMYSLNY_LATENT)
        klucz = 'before' if 'before' in dane else dane.files[0]
        if dane[klucz].shape == (TOKENY, KANALY):
            return dane[klucz].astype(np.float32), f'{DOMYSLNY_LATENT.name} [{klucz}]'
    return np.zeros((TOKENY, KANALY), np.float32), 'zera'


def zakresy_kanalow(baza):
    """Per-kanał [lo, hi] dla suwaków: obserwowany zakres bazy poszerzony o 3 sigma."""
    mu = baza.mean(axis=0)
    sd = baza.std(axis=0)
    lo = np.minimum(baza.min(axis=0), mu - 3 * sd)
    hi = np.maximum(baza.max(axis=0), mu + 3 * sd)
    waskie = (hi - lo) < 1e-3
    lo[waskie] -= 1.0
    hi[waskie] += 1.0
    return lo.astype(np.float32), hi.astype(np.float32)


# ---------------------------------------------------------------- limity makrowektorów
# Makrowektor = jeden z 768 kanałów (oś latentu); każdy ma własny przedział [min, max].
# Wyprowadzenie granic: experiments/vit_granice_latentu/README.md.

GELU_MIN = -0.16997120747990244          # minimum GELU (w a = -0,7518), twarde dla każdego kanału
GELU_A0 = -0.7517916


def _gelu(a):
    import math
    return 0.5 * a * (1 + math.erf(a / math.sqrt(2)))


def granice_z_wag(checkpoint=VIT_CHECKPOINT):
    """Dokładny przedział każdego z 768 kanałów dla ustalonych wag ViT (supremum po wszystkich wejściach bloku).

    h_j = GELU(W1_j . LN(x) + c_j); wyjście LN przed skalą ma średnią 0 i normę <= sqrt(d), więc
    pobudzenie a_j leży w [m_j - r_j, m_j + r_j], m_j = W1_j . b + c_j, r_j = sqrt(d) ||P(W1_j * g)||.
    Zwraca (h_min, h_max) float32 [768].
    """
    import torch
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
    sd = sd.get('net', sd)
    blok = max(int(k.split('.')[1]) for k in sd if k.startswith('bloki.'))
    g = sd[f'bloki.{blok}.norm2.weight'].double()
    b = sd[f'bloki.{blok}.norm2.bias'].double()
    W = sd[f'bloki.{blok}.mlp.fc1.weight'].double()
    c = sd[f'bloki.{blok}.mlp.fc1.bias'].double()
    Wg = W * g
    r = (g.numel() ** .5) * (Wg - Wg.mean(1, keepdim=True)).norm(dim=1)
    m = W @ b + c
    a_lo, a_hi = (m - r).numpy(), (m + r).numpy()
    h_max = np.array([_gelu(a) for a in a_hi])
    h_lo = np.array([_gelu(a) for a in a_lo])
    h_min = np.where((a_lo < GELU_A0) & (a_hi > GELU_A0), GELU_MIN, np.minimum(h_lo, h_max))
    return h_min.astype(np.float32), h_max.astype(np.float32)


def granica_po_krokach(kroki, lr=5e-4, wd=0.05, rozgrzewka=300, b1=0.9, b2=0.999, d=192):
    """Twarda granica latentu po `kroki` krokach AdamW (przepis maly_vit_od_zera), dla dowolnych danych.

    Wspólna dla wszystkich kanałów: start nn.Linear U(+-1/sqrt(d)), LayerNorm g=1, b=0; krok Adama
    |m_hat/sqrt(v_hat)| <= B_t (Cauchy-Schwarz); decay tylko macierze. Bez limitu kroków granica nie
    istnieje (g, b LayerNorm i bias fc1 bez weight decay). Zwraca (h_min, h_max) float32 [768].
    """
    import math
    R0 = 1 / math.sqrt(d)
    Rw, Rc, Rg, Rb, s = R0, R0, 0.0, 0.0, 0.0
    for t in range(1, int(kroki) + 1):
        lr_t = lr * min(1.0, t / rozgrzewka)
        s = s * (b1 * b1 / b2) + 1.0
        B = (1 - b1) / math.sqrt(1 - b2) * math.sqrt(s) * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        Rw = (1 - lr_t * wd) * Rw + lr_t * B
        Rc, Rg, Rb = Rc + lr_t * B, Rg + lr_t * B, Rb + lr_t * B
    a = d * Rw * (Rb + 1 + Rg) + Rc
    return (np.full(KANALY, GELU_MIN, np.float32), np.full(KANALY, _gelu(a), np.float32))


def wczytaj_limity(sciezka):
    """Limity 768 kanałów z NPZ (h_min/h_max, min/max albo lo/hi), CSV (kolumny kanal,min,max) lub JSON."""
    import csv, json
    sciezka = Path(sciezka)
    pary = (('h_min', 'h_max'), ('min', 'max'), ('lo', 'hi'))
    if sciezka.suffix.lower() == '.npz':
        dane = np.load(sciezka)
        for a, b in pary:
            if a in dane and b in dane:
                lo, hi = dane[a], dane[b]
                break
        else:
            raise ValueError(f'NPZ bez par h_min/h_max, min/max ani lo/hi: {dane.files}')
    elif sciezka.suffix.lower() == '.json':
        dane = json.loads(sciezka.read_text(encoding='utf-8'))
        for a, b in pary:
            if a in dane and b in dane:
                lo, hi = np.asarray(dane[a]), np.asarray(dane[b])
                break
        else:
            raise ValueError('JSON bez par h_min/h_max, min/max ani lo/hi')
    else:
        lo, hi = np.full(KANALY, np.nan), np.full(KANALY, np.nan)
        with open(sciezka, newline='', encoding='utf-8-sig') as f:
            for w in csv.DictReader(f):
                k = int(w.get('kanal', w.get('channel0')))
                lo[k], hi[k] = float(w['min']), float(w['max'])
    lo, hi = np.asarray(lo, np.float32).reshape(-1), np.asarray(hi, np.float32).reshape(-1)
    if lo.shape != (KANALY,) or hi.shape != (KANALY,) or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError('Limity muszą mieć 768 skończonych wartości min i max')
    if (hi <= lo).any():
        raise ValueError(f'max <= min w kanałach: {np.flatnonzero(hi <= lo)[:10].tolist()}')
    return lo, hi


def zapisz_limity(lo, hi, zrodlo):
    """Zapis limitów do WYJSCIE jako NPZ (h_min, h_max, zrodlo) i CSV (kanal,min,max). Zwraca ścieżkę NPZ."""
    from datetime import datetime
    WYJSCIE.mkdir(parents=True, exist_ok=True)
    znacznik = datetime.now().strftime('%Y%m%d_%H%M%S')
    npz = WYJSCIE / f'limity_{znacznik}.npz'
    np.savez(npz, h_min=lo, h_max=hi, zrodlo=str(zrodlo))
    with open(npz.with_suffix('.csv'), 'w', encoding='utf-8') as f:
        f.write('kanal,min,max\n')
        for k in range(KANALY):
            f.write(f'{k},{lo[k]:.6f},{hi[k]:.6f}\n')
    return npz
