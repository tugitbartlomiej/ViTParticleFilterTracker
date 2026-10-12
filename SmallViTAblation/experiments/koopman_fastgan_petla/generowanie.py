"""Diagnostyczne odwracanie celu Koopmana przez zamrożone FastGAN i RF-DETR.

API generate(checkpointRFpth, generatorckpt, vendor, probe_anchor,
directions_npz, koopmantargetnpy, out, device, maxsteps=40, restarts=2,
expected_current_npy=None).
probe_anchor: manifest sondy (pierwszy original), (manifest, indeks) albo
JSON pojedynczego rekordu z image/mask względnymi do tego JSON.
Kierunki NPZ: mean[1536], scale[1536], projection[1536,K] (alias directions),
opcjonalnie target_scale[K], domyślnie jedynki. Cel NPY: pełny kod [K].

Całe RGB pochodzi z generatora. Ręczna maska kotwicy jest stałym obszarem
odczytu; maska z kanału czwartego służy wyłącznie ograniczeniu kształtu.
Wyjścia są diagnostic, nigdy labeled_generated_training. Zgodność maski
ani osiągnięcie celu nie dowodzą istnienia poprawnego metalu w RGB.
Import nie ładuje torch, Pillow, RF-DETR ani kodu poprzedniego eksperymentu.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

if __package__:
    from .czastki import DESCRIPTOR_DIM, DESCRIPTOR_VERSION, DifferentiableReader
else:
    from czastki import DESCRIPTOR_DIM, DESCRIPTOR_VERSION, DifferentiableReader

# Tolerancja dotyczy średniego kwadratu błędu w stałej skali celu.
FEATURE_GRADIENT_ERROR_EPS = 1e-12
EXPECTED_CURRENT_SCALED_RMSE_TOL = 1e-5


def _finite_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name}: wymagane liczby rzeczywiste')
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: wymagane skończone liczby')
    return result


class DescriptorProjection:
    """Stała baza i skalowanie dopasowane poza generatorem, przed prognozą."""
    def __init__(self, mean, scale, directions, target_scale=None):
        self.mean = _finite_array(mean, 'mean')
        self.scale = _finite_array(scale, 'scale')
        self.directions = _finite_array(directions, 'projection')
        if (self.mean.shape != (DESCRIPTOR_DIM,) or self.scale.shape != self.mean.shape or
                np.any(self.scale <= 0) or self.directions.ndim != 2 or
                self.directions.shape[0] != DESCRIPTOR_DIM or self.directions.shape[1] < 1):
            raise ValueError('Niezgodna baza: mean/scale[1536], scale>0, projection[1536,K]')
        k = self.directions.shape[1]
        self.target_scale = _finite_array(np.ones(k) if target_scale is None else target_scale, 'target_scale')
        if self.target_scale.shape != (k,) or np.any(self.target_scale <= 0):
            raise ValueError('target_scale musi mieć K dodatnich wartości')

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            key = 'projection' if 'projection' in data else 'directions'
            if not {'mean', 'scale', key}.issubset(data.files):
                raise ValueError('NPZ wymaga mean, scale, projection (lub directions)')
            if 'projection' in data and 'directions' in data and not np.array_equal(data['projection'], data['directions']):
                raise ValueError('Sprzeczne projection i directions')
            return cls(data['mean'], data['scale'], data[key], data['target_scale'] if 'target_scale' in data else None)

    def project_numpy(self, descriptors):
        descriptors = _finite_array(descriptors, 'descriptors')
        if descriptors.ndim < 1 or descriptors.shape[-1] != DESCRIPTOR_DIM:
            raise ValueError('Ostatnia oś deskryptorów musi mieć 1536 cech')
        return ((descriptors-self.mean)/self.scale) @ self.directions

    def project_torch(self, descriptors):
        if descriptors.shape[-1] != DESCRIPTOR_DIM:
            raise ValueError('Ostatnia oś deskryptorów musi mieć 1536 cech')
        return ((descriptors-descriptors.new_tensor(self.mean))/descriptors.new_tensor(self.scale)) @ descriptors.new_tensor(self.directions)


def make_control_targets(current, target, target_scale, seed=42):
    """Losowy przyrost ma normę równą delcie Koopmana w metryce straty."""
    current, target, scale = (_finite_array(v, n) for v, n in
                              ((current, 'current'), (target, 'target'), (target_scale, 'target_scale')))
    if current.ndim != 1 or current.size == 0 or current.shape != target.shape or scale.shape != target.shape or np.any(scale <= 0):
        raise ValueError('Cele i dodatnie skale muszą mieć zgodny kształt [K]')
    norm = np.linalg.norm((target-current)/scale)
    random = np.random.default_rng(seed).normal(size=current.shape)
    random = random/np.linalg.norm(random)*norm
    return dict(target=target, current=current, random_norm=current+random*scale)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_fastgan(checkpoint, vendor, device):
    """Załaduj wyłącznie zweryfikowaną strukturę wytrenowanego EMA nc4/nz256.

    Sprawdzamy step>0, zapis konfiguracji, skrót lokalnego models.py,
    wejście/wyjście wag i strict state_dict. To weryfikacja pochodzenia i
    architektury, nie jakości wygenerowanych narzędzi.
    """
    import torch

    checkpoint, vendor = Path(checkpoint), Path(vendor)
    source = vendor/'models.py' if vendor.is_dir() else vendor
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = payload.get('config', {})
    step = payload.get('step', 0)
    ema = payload.get('ema')
    if not isinstance(step, int) or step <= 0 or not isinstance(ema, dict) or not ema:
        raise ValueError('Wymagane EMA z wytrenowanego checkpointu, step>0')
    if any(config.get(k) != v for k, v in {'channels': 4, 'size': 256, 'width': 64}.items()):
        raise ValueError('Checkpoint musi opisywać nc=4, im_size=256, ngf=64')
    architecture_sha = _sha(source)
    if config.get('architecture_sha256') != architecture_sha:
        raise ValueError('Skrót models.py nie zgadza się z architekturą treningu')
    data_sha = config.get('data_manifest_sha256', '')
    if not isinstance(data_sha, str) or len(data_sha) != 64:
        raise ValueError('Brak identyfikatora danych treningowych generatora')
    latent_weight = ema.get('init.init.0.weight_orig')
    rgb_weight = ema.get('to_big.weight_orig')
    if (latent_weight is None or latent_weight.ndim != 4 or latent_weight.shape[0] != 256 or
            rgb_weight is None or rgb_weight.ndim != 4 or rgb_weight.shape[0] != 4):
        raise ValueError('Wagi EMA nie potwierdzają nz=256 i nc=4')
    spec = importlib.util.spec_from_file_location(f'koopman_fastgan_vendor_{architecture_sha[:12]}', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Inicjalizacja konstruktorów nie powinna zmieniać zewnętrznego RNG CPU.
    with torch.random.fork_rng(devices=[]):
        net = module.Generator(ngf=64, nz=256, nc=4, im_size=256)
    net.load_state_dict(ema, strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in net.state_dict().values() if v.is_floating_point()):
        raise ValueError('Nieskończone wagi EMA')
    net = net.to(device).eval().requires_grad_(False)
    return net, dict(step=step, weights='ema', nc=4, nz=256, ngf=64, im_size=256,
                     checkpoint_sha256=_sha(checkpoint), architecture_sha256=architecture_sha,
                     data_manifest_sha256=data_sha)


def fixed_generator_output(generator, z, seed=42):
    """RGBA w [0,1], stałe szumy warstw i zachowany zewnętrzny stan RNG.

    Vendor losuje NoiseInjection na CPU i przenosi na urządzenie cech.
    Zachowujemy także RNG konkretnego CUDA, gdy podano takie urządzenie.
    Nie używamy no_grad: graf prowadzi przez generator aż do z.
    """
    import torch

    devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        value = generator(z)
        raw = value[0] if isinstance(value, (tuple, list)) else value
        if raw.ndim != 4 or raw.shape[1] != 4 or raw.shape[0] != z.shape[0]:
            raise ValueError('Generator musi zwracać Bx4xHxW')
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Generator zwrócił NaN/Inf')
        return (raw.clamp(-1, 1)+1)/2


@contextmanager
def _frozen_eval(*models):
    modes = {m: m.training for model in models for m in model.modules()}
    flags = {p: p.requires_grad for model in models for p in model.parameters()}
    try:
        for model in models:
            model.eval().requires_grad_(False)
        yield
    finally:
        for p, flag in flags.items():
            p.requires_grad_(flag)
        for module, mode in modes.items():
            module.training = mode


def shape_iou(generated_mask, anchor_mask):
    """Zgodność progowanego kanału maski z kotwicą; nie ocena metalu w RGB."""
    import torch
    if generated_mask.shape != anchor_mask.shape:
        raise ValueError('Maski muszą mieć identyczny kształt')
    if not bool(torch.isfinite(generated_mask).all() and torch.isfinite(anchor_mask).all()):
        raise ValueError('Maski muszą być skończone')
    a, b = generated_mask >= .5, anchor_mask >= .5
    union = (a | b).sum().item()
    return float((a & b).sum().item()/union) if union else 0.


def optimize_latent(generator, reader, anchor_image, anchor_mask, projection, target, z0,
                    *, steps=40, lr=.05, noise_seed=42):
    """Optymalizuj tylko z; mały wariant tej samej funkcji służy testom CPU.

    L = mean(((c-target)/target_scale)^2) + .1*L1(RGB,anchor)
        + L1(mask_G,mask_hand) + .001*mean(z^2).
    Zachowuje najlepszy punkt wg pełnej straty, także inicjalizację. Historia
    ma steps+1 ocen, wykonujemy dokładnie steps aktualizacji i testów gradientu.
    Osobno badamy gradient samej straty cech w każdej ocenie, także końcowej.
    Brak połączenia z z oznacza zerowy gradient; przy błędzie celu > 1e-12
    taki gradient przerywa próbę. Zprior ani L1 nie zastępują tej kontroli.
    Nie przełącza celu, nie dopasowuje skali na obrazach generowanych.
    """
    import torch
    import torch.nn.functional as functional

    if not isinstance(steps, int) or steps < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('steps i lr muszą być dodatnie')
    if not torch.is_grad_enabled():
        raise RuntimeError('Generowanie wymaga aktywnego autograd, bez no_grad/inference_mode')
    if anchor_image.ndim != 4 or anchor_image.shape[:2] != (1, 3) or anchor_mask.shape != (1, 1, *anchor_image.shape[-2:]):
        raise ValueError('Wymagane pojedyncze anchor RGB B3HW i maska B1HW')
    if not bool(torch.isfinite(anchor_image).all() and torch.isfinite(anchor_mask).all()):
        raise ValueError('Kotwica musi być skończona')
    if bool(((anchor_image < 0) | (anchor_image > 1)).any() or ((anchor_mask < 0) | (anchor_mask > 1)).any()):
        raise ValueError('RGB i maska muszą być w [0,1]')
    if not 0 < float(anchor_mask.sum()) < anchor_mask.numel():
        raise ValueError('Kotwica wymaga ręcznego wsparcia i tła')
    if z0.ndim != 2 or z0.shape[0] != 1 or not bool(torch.isfinite(z0).all()):
        raise ValueError('z0 musi być skończonym [1,nz]')
    target = _finite_array(target, 'target')
    if target.shape != projection.target_scale.shape:
        raise ValueError('Cel musi mieć kształt [K] w bazie kierunków')
    z = z0.detach().clone().requires_grad_(True)
    anchor_image, anchor_mask = anchor_image.detach(), anchor_mask.detach()
    target_tensor = z.new_tensor(target)[None]
    scale = z.new_tensor(projection.target_scale)[None]
    optimizer = torch.optim.Adam([z], lr=lr)
    history, best, best_rgba, best_z, best_code = [], None, None, None, None
    max_gradient_norm = 0.
    feature_gradient_max_norm = 0.
    feature_gradient_finite = True
    feature_gradient_required = False
    with _frozen_eval(generator, reader.net):
        for step in range(steps+1):
            rgba = fixed_generator_output(generator, z, seed=noise_seed)
            if rgba.shape[-2:] != anchor_image.shape[-2:]:
                raise ValueError('Generator i kotwica muszą mieć ten sam rozmiar wycinka')
            descriptors = reader.descriptors(rgba[:, :3], anchor_mask)
            code = projection.project_torch(descriptors)
            feature = ((code-target_tensor)/scale).square().mean()
            anchor = functional.l1_loss(rgba[:, :3], anchor_image)
            mask = functional.l1_loss(rgba[:, 3:4], anchor_mask)
            prior = z.square().mean()
            loss = feature+.1*anchor+mask+.001*prior
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nieskończona strata generowania')
            stats = dict(step=step, total=float(loss.detach()), feature=float(feature.detach()),
                         anchor_l1=float(anchor.detach()), mask_l1=float(mask.detach()), z_prior=float(prior.detach()))
            # allow_unused obsługuje kod wymagający gradientu, ale odłączony
            # od z. Całkiem odłączony tensor nie ma grad_fn: nie wolno wtedy
            # wywoływać autograd.grad nawet z allow_unused=True.
            feature_gradient = None
            if feature.requires_grad:
                feature_gradient, = torch.autograd.grad(feature, z, retain_graph=True, allow_unused=True)
            feature_norm = 0. if feature_gradient is None else float(feature_gradient.norm())
            feature_finite = (np.isfinite(feature_norm) and
                              (feature_gradient is None or bool(torch.isfinite(feature_gradient).all())))
            if not feature_finite:
                raise FloatingPointError(f'Nieskończony gradient straty cech względem z w kroku {step}')
            required = stats['feature'] > FEATURE_GRADIENT_ERROR_EPS
            if required and feature_norm == 0:
                raise RuntimeError('Brak niezerowego gradientu straty cech względem z '
                                   f'przy błędzie celu {stats["feature"]:.6g} > {FEATURE_GRADIENT_ERROR_EPS:g}')
            feature_gradient_max_norm = max(feature_gradient_max_norm, feature_norm)
            feature_gradient_finite = feature_gradient_finite and bool(feature_finite)
            feature_gradient_required = feature_gradient_required or required
            stats.update(feature_gradient_norm=feature_norm, feature_gradient_finite=bool(feature_finite),
                         feature_gradient_required=required, feature_gradient_connected=feature_gradient is not None)
            history.append(stats)
            if best is None or stats['total'] < best['total']:
                best = stats.copy()
                best_rgba, best_z, best_code = rgba.detach().clone(), z.detach().clone(), code.detach().clone()
            if step == steps:
                break
            gradient, = torch.autograd.grad(loss, z)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError('Nieskończony gradient względem z')
            norm = float(gradient.norm())
            if not np.isfinite(norm):
                raise FloatingPointError('Nieskończona norma gradientu')
            max_gradient_norm = max(max_gradient_norm, norm)
            stats['gradient_norm'] = norm
            optimizer.zero_grad(set_to_none=True)
            z.grad = gradient
            optimizer.step()
    return dict(best=best, history=history, rgba=best_rgba, z=best_z, code=best_code,
                initial_z=z0.detach().clone(), gradients_finite=True, gradient_steps=steps,
                max_gradient_norm=max_gradient_norm, feature_gradient_max_norm=feature_gradient_max_norm,
                feature_gradient_finite=feature_gradient_finite, feature_gradient_required=feature_gradient_required)


def _load_anchor(probe_anchor, device):
    import torch
    from PIL import Image

    index = None
    if isinstance(probe_anchor, (tuple, list)) and len(probe_anchor) == 2:
        probe_anchor, index = probe_anchor
    path = Path(probe_anchor)
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        if index is None:
            index = next((i for i, r in enumerate(data) if r['variant'] == 'original'), None)
        if not isinstance(index, int) or not 0 <= index < len(data):
            raise ValueError('Brak prawidłowego indeksu kotwicy original')
        row = data[index]
    elif isinstance(data, dict):
        row = data
    else:
        raise ValueError('probe_anchor musi wskazywać manifest lub rekord JSON')
    if row.get('split', 'train') != 'train' or row.get('role') not in ('fit', 'holdout') or row['labels'][0] != 1:
        raise ValueError('Kotwica wymaga rzeczywistego narzędzia z puli sondy train')
    root = path.parent.resolve()
    images = []
    for key, mode in [('image', 'RGB'), ('mask', 'L')]:
        local = (root/row[key]).resolve()
        if Path(row[key]).is_absolute() or not local.is_relative_to(root):
            raise ValueError('Ścieżka kotwicy wychodzi poza katalog sondy')
        if key+'_sha256' in row and _sha(local) != row[key+'_sha256']:
            raise ValueError(f'Zmieniony plik kotwicy: {key}')
        with Image.open(local) as image:
            images.append(np.array(image.convert(mode), copy=True))
    rgb, mask = images
    if rgb.shape != (256, 256, 3) or mask.shape != (256, 256):
        raise ValueError('Wytrenowany FastGAN wymaga kotwicy 256x256; przygotuj sondę z takich wycinków')
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device)/255
    support = torch.from_numpy(mask)[None, None].float().to(device)/255
    return image, support, dict(record=row, manifest_sha256=_sha(path), row_index=index)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def _save_image(path, value):
    from PIL import Image
    a = value.detach().cpu().numpy()
    a = np.moveaxis(a[0], 0, -1)
    if a.shape[-1] == 1:
        a = a[..., 0]
    Image.fromarray(np.rint(np.clip(a, 0, 1)*255).astype(np.uint8)).save(path)


@contextmanager
def _generation_numerics(device):
    """Te same kernele i FP32 co trening oraz pomiar zamrożonej referencji.

    Import pozostaje leniwy. Kontekst obejmuje również backward wszystkich
    kontroli i przywraca ustawienia procesu także po nieudanej generacji.
    """
    import torch
    if __package__:
        from .deterministyka import diagnostic_kernels
    else:
        from deterministyka import diagnostic_kernels

    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        with diagnostic_kernels(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield dict(amp=False, dtype='float32',
                       matmul_precision=torch.get_float32_matmul_precision(),
                       tf32=torch.backends.cuda.matmul.allow_tf32,
                       determinism='strict; math SDPA; grid_sample CPU with autograd',
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                       deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       sdpa_backend='math', grid_sample='cpu_fallback_with_autograd',
                       torch_version=str(torch.__version__))
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def generate(checkpointRFpth, generatorckpt, vendor, probe_anchor, directions_npz,
             koopmantargetnpy, out, device, maxsteps=40, restarts=2, *, expected_current_npy=None):
    """Wykonaj trzy sparowane kontrole i zapisz diagnostyczne obrazy oraz raport.

    RF-DETR: RFDETRNano(num_classes=1, resolution=384,
    pretrain_weights=checkpointRFpth, amp=False), własne means/stds, końcowy eksport pth.
    Odczyt kotwicy i pełny autograd wszystkich kontroli używają diagnostic_kernels
    oraz FP32/highest bez TF32; konfiguracja trafia do NUMERICS.json i raportu.
    expected_current_npy opcjonalnie wskazuje zapisany kod kotwicy [K]
    z tego samego checkpointu i bazy. Scaled RMSE względem odczytu musi być
    <=1e-5; rozbieżność zapisujemy i przerywamy przed optymalizacją z.
    Ziarno 42; jednakowe z0 i szumy warstw per restart dla wszystkich kontroli.
    Progi diagnostyczne są stałe: scaled RMSE<=.25, shape IoU>=.5, L1<=.25.
    Raport JSON nigdy nie certyfikuje semantycznej poprawności obrazu.
    """
    import torch

    if not isinstance(maxsteps, int) or maxsteps < 1 or not isinstance(restarts, int) or restarts < 1:
        raise ValueError('maxsteps i restarts muszą być dodatnimi liczbami całkowitymi')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    checkpointRFpth = Path(checkpointRFpth)
    if not checkpointRFpth.is_file():
        raise FileNotFoundError(checkpointRFpth)
    projection = DescriptorProjection.from_npz(directions_npz)
    target = _finite_array(np.load(koopmantargetnpy, allow_pickle=False), 'pełny cel Koopmana')
    if target.shape != projection.target_scale.shape:
        raise ValueError('NPY celu musi mieć kształt [K], nie NxK ani surowe 1536 cech')
    expected_current = None
    if expected_current_npy is not None:
        expected_current = _finite_array(np.load(expected_current_npy, allow_pickle=False), 'expected_current')
        if expected_current.shape != projection.target_scale.shape:
            raise ValueError('expected_current musi mieć kształt [K] dla wybranej kotwicy')
    anchor_image, anchor_mask, anchor_meta = _load_anchor(probe_anchor, device)
    generator, generator_meta = load_fastgan(generatorckpt, vendor, device)
    from rfdetr import RFDETRNano
    wrapper = RFDETRNano(num_classes=1, resolution=384, pretrain_weights=str(checkpointRFpth),
                        trust_checkpoint=True, amp=False)
    net = wrapper.model.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    out.mkdir(parents=True, exist_ok=False)
    _save_image(out/'anchor.png', anchor_image)
    _save_image(out/'anchor_support.png', anchor_mask)
    report = dict(classification='diagnostic', labeled_generated_training=False,
                  semantic_metal_verified=False, human_review_required=True,
                  note='Kształt kanału maski nie jest prawdą odniesienia metalu w RGB. Brak wniosku o nowej geometrii 3D.',
                  descriptor_version=DESCRIPTOR_VERSION, anchor=anchor_meta, generator=generator_meta,
                  detector_checkpoint_sha256=_sha(checkpointRFpth), directions_sha256=_sha(directions_npz),
                  target_sha256=_sha(koopmantargetnpy), means=list(wrapper.means), stds=list(wrapper.stds),
                  device=str(device), seed=42, maxsteps=maxsteps, restarts=restarts,
                  loss_weights=dict(feature=1., anchor_l1=.1, mask_l1=1., z_prior=.001),
                  thresholds=dict(scaled_rmse=.25, shape_iou=.5, anchor_l1=.25,
                                  feature_gradient_error_eps=FEATURE_GRADIENT_ERROR_EPS,
                                  expected_current_scaled_rmse=EXPECTED_CURRENT_SCALED_RMSE_TOL),
                  random_norm_metric='L2 po podzieleniu delty przez target_scale', arms={})
    _write_json(out/'status.json', dict(state='running', classification='diagnostic'))
    try:
        with _generation_numerics(device) as numerics, DifferentiableReader(net, wrapper.means, wrapper.stds) as reader:
            report['numerics'] = numerics
            _write_json(out/'NUMERICS.json', numerics)
            with torch.no_grad():
                current = projection.project_torch(reader.descriptors(anchor_image, anchor_mask))[0].cpu().numpy()
            current = _finite_array(current, 'odczyt kotwicy')
            np.save(out/'current.npy', current)
            check = dict(enabled=expected_current is not None, anchor_index=anchor_meta['row_index'],
                         threshold=EXPECTED_CURRENT_SCALED_RMSE_TOL, scaled_rmse=None, passed=None,
                         metric='sqrt(mean(((current-expected_current)/target_scale)^2))')
            if expected_current is not None:
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    error = (current-expected_current)/projection.target_scale
                    rmse = float(np.linalg.norm(error)/np.sqrt(error.size))
                finite = bool(np.isfinite(rmse))
                check.update(scaled_rmse=rmse if finite else None, finite=finite,
                             passed=finite and rmse <= EXPECTED_CURRENT_SCALED_RMSE_TOL,
                             expected_current_sha256=_sha(expected_current_npy),
                             expected_current=expected_current.tolist(), current=current.tolist(),
                             target_scale=projection.target_scale.tolist())
            report['expected_current_check'] = check
            _write_json(out/'CURRENT_CHECK.json', check)
            _write_json(out/'report.json', report)
            if expected_current is not None and not check['passed']:
                raise RuntimeError('Odczyt kotwicy nie zgadza się z expected_current: '
                                   f'scaled RMSE={check["scaled_rmse"]}, '
                                   f'tolerancja={EXPECTED_CURRENT_SCALED_RMSE_TOL:g}')
            targets = make_control_targets(current, target, projection.target_scale, seed=42)
            np.savez_compressed(out/'targets.npz', **targets, target_scale=projection.target_scale)
            for arm, code_target in targets.items():
                arm_dir = out/arm
                arm_dir.mkdir()
                attempts = []
                for restart in range(restarts):
                    seed = 42+restart
                    local_rng = torch.Generator(device='cpu').manual_seed(seed)
                    z0 = torch.randn((1, 256), generator=local_rng).to(device)
                    result = optimize_latent(generator, reader, anchor_image, anchor_mask, projection,
                                             code_target, z0, steps=maxsteps, noise_seed=1042+restart)
                    stem = f'restart_{restart:02d}'
                    _save_image(arm_dir/f'{stem}.png', result['rgba'][:, :3])
                    _save_image(arm_dir/f'{stem}_mask_diagnostic.png', result['rgba'][:, 3:4])
                    np.savez_compressed(arm_dir/f'{stem}.npz', z=result['z'].cpu().numpy(),
                                        z0=z0.cpu().numpy(), code=result['code'].cpu().numpy(),
                                        target=code_target, noise_seed=np.asarray(1042+restart))
                    best = result['best']
                    iou = shape_iou(result['rgba'][:, 3:4], anchor_mask)
                    gates = dict(finite_gradient=result['gradients_finite'], nonzero_gradient=result['max_gradient_norm'] > 0,
                                 feature_gradient_finite=result['feature_gradient_finite'],
                                 feature_gradient_nonzero=(result['feature_gradient_max_norm'] > 0 or
                                                           not result['feature_gradient_required']),
                                 target_reached=best['feature']**.5 <= .25, shape_consistency=iou >= .5,
                                 anchor_similarity=best['anchor_l1'] <= .25)
                    stats = dict(restart=restart, image=f'{arm}/{stem}.png', best=best, shape_iou=iou,
                                 gates=gates, numerical_gates_pass=all(gates.values()), semantic_metal_verified=False,
                                 gradient_steps=result['gradient_steps'], max_gradient_norm=result['max_gradient_norm'],
                                 feature_gradient_max_norm=result['feature_gradient_max_norm'],
                                 feature_gradient_finite=result['feature_gradient_finite'],
                                 feature_gradient_required=result['feature_gradient_required'],
                                 history=result['history'], seed=seed, noise_seed=1042+restart)
                    _write_json(arm_dir/f'{stem}.json', stats)
                    attempts.append(stats)
                winner = min(attempts, key=lambda a: a['best']['total'])
                winner_stem = f"restart_{winner['restart']:02d}"
                for source, destination in [(f'{winner_stem}.png', 'best.png'),
                                            (f'{winner_stem}_mask_diagnostic.png', 'best_mask_diagnostic.png')]:
                    (arm_dir/destination).write_bytes((arm_dir/source).read_bytes())
                report['arms'][arm] = dict(target=code_target.tolist(), attempts=attempts,
                                           best_restart=winner['restart'], best_image=f'{arm}/best.png',
                                           gradient_budget=maxsteps*restarts,
                                           numerical_gates_pass=winner['numerical_gates_pass'])
                _write_json(out/'report.json', report)
        _write_json(out/'status.json', dict(state='complete', classification='diagnostic'))
    except Exception as exc:
        _write_json(out/'status.json', dict(state='failed', classification='diagnostic', error=str(exc)))
        raise
    return report
