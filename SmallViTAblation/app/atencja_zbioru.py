"""Mapy atencji ViT dla wielu obrazów naraz (CUDA), te same definicje co panel opcji_atencji.OpcjeAtencji.

Zakładka „Ruch zbioru” nakłada je na cząsteczki (kolor i rozmiar) albo na zdjęcie rozpięte w przestrzeni. Wagi
atencji liczone jak w detekcja.Detektor.atencja: softmax(q k^T · skala) z wejścia bloku atencji, ale dla całej
paczki obrazów. Mapa [600] na obraz; skala (liniowa, log, ranga) normowana osobno w każdym obrazie do 0..1.
"""
import numpy as np

WIERSZE, KOLUMNY = 20, 30


def _odleglosci(dev):
    import torch
    r, c = np.divmod(np.arange(WIERSZE * KOLUMNY), KOLUMNY)
    d = np.hypot(r[:, None] - r[None], c[:, None] - c[None]).astype(np.float32)
    return torch.as_tensor(d, device=dev)


def ustawienia(opcje):
    """Ustawienia z panelu OpcjeAtencji jako słownik (blok od 1, głowa, rollout, bez siebie, mapa, skala)."""
    return dict(blok=opcje.spin_blok.value(), glowa=opcje.combo_glowa.currentIndex(),
                rollout=opcje.combo_bloki.currentIndex() != 0, bez_siebie=opcje.chk_bez_siebie.isChecked(),
                mapa=opcje.combo_mapa.currentIndex(), skala=opcje.combo_skala.currentIndex())


def _macierze_blokow(net, x, bloki):
    """Wagi atencji [B, głowy, 600, 600] wybranych bloków dla paczki x [B, 3, H, W] (float 0..1)."""
    import torch
    pam = {}
    uchwyty = [net.bloki[b].attn.register_forward_hook(lambda m, a, o, b=b: pam.__setitem__(b, a[0].detach()))
               for b in bloki]
    try:
        with torch.no_grad():
            net(x)
    finally:
        for h in uchwyty:
            h.remove()
        net._latent = None
    wynik = {}
    for b in bloki:
        at, xin = net.bloki[b].attn, pam[b]
        B, N, C = xin.shape
        with torch.no_grad():
            qkv = at.qkv(xin).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q = getattr(at, 'q_norm', torch.nn.Identity())(q)
            k = getattr(at, 'k_norm', torch.nn.Identity())(k)
            skala = getattr(at, 'scale', (C // at.num_heads) ** -0.5)
            wynik[b] = ((q * skala) @ k.transpose(-2, -1)).softmax(-1).float()
    return wynik


def _jeden(A, u):
    """[B, H, 600, 600] -> [B, 600, 600] według głowy (średnia, maksimum, numer) i opcji „bez siebie”."""
    import torch
    g = u['glowa']
    M = A.mean(1) if g == 0 else A.max(1).values if g == 1 else A[:, min(g - 2, A.shape[1] - 1)]
    if g == 1 or u['bez_siebie']:
        M = M.clone()
        if u['bez_siebie']:
            M.diagonal(dim1=1, dim2=2).zero_()
        M = M / M.sum(-1, keepdim=True).clamp_min(1e-12)
    return M


def macierz(net, x, u):
    """Macierz atencji [B, 600, 600] dla paczki (wiersz = kto patrzy, kolumna = na kogo), jak OpcjeAtencji.macierz."""
    import torch
    b = u['blok'] - 1
    if not u['rollout']:
        return _jeden(_macierze_blokow(net, x, [b])[b], u)
    A = _macierze_blokow(net, x, list(range(b + 1)))
    I = torch.eye(600, device=x.device)
    M = I.expand(x.shape[0], 600, 600).clone()
    for i in range(b + 1):
        Ai = .5 * _jeden(A[i], u) + .5 * I
        M = (Ai / Ai.sum(-1, keepdim=True)) @ M
    if u['bez_siebie']:
        M.diagonal(dim1=1, dim2=2).zero_()
        M = M / M.sum(-1, keepdim=True).clamp_min(1e-12)
    return M


def mapy(net, x, u, token=None):
    """Mapy [B, 600] w 0..1 (normowane w każdym obrazie) albo None, gdy mapa potrzebuje wybranego patcha."""
    import torch
    m = u['mapa']
    if m in (1, 2) and token is None:
        return None
    M = macierz(net, x, u)
    tok = None if token is None else torch.as_tensor(np.atleast_1d(token), device=x.device, dtype=torch.long)
    if m == 0:
        v = M.mean(1)
    elif m == 1:
        v = M[:, tok].mean(1)
    elif m == 2:
        v = M[:, :, tok].mean(-1)
    elif m == 3:
        v = 1 + (M * M.clamp_min(1e-12).log()).sum(-1) / np.log(600)
    elif m == 4:
        v = (M * _odleglosci(x.device)).sum(-1)
    else:
        v = M.diagonal(dim1=1, dim2=2)
    s = u['skala']
    if s == 1:
        v = torch.log10(torch.maximum(v, v.max(1, keepdim=True).values.clamp_min(1e-12) * 1e-4))
    elif s == 2:
        v = v.argsort(1).argsort(1).float()
    lo, hi = v.min(1, keepdim=True).values, v.max(1, keepdim=True).values
    return ((v - lo) / (hi - lo).clamp_min(1e-12)).cpu().numpy()


def mapy_zbioru(net, obrazy, u, token=None, paczka=24, postep=lambda i, n: None, przerwij=lambda: False):
    """Mapy dla wszystkich obrazów: obrazy uint8 [N, 3, H, W] (tensor na karcie albo w RAM). Zwraca [N, 600]."""
    import torch
    dev = next(net.parameters()).device
    n = obrazy.shape[0]
    wynik = np.zeros((n, 600), np.float32)
    for a in range(0, n, paczka):
        if przerwij():
            raise InterruptedError('przerwano liczenie atencji')
        x = obrazy[a:a + paczka].to(device=dev, dtype=torch.float32) / 255
        v = mapy(net, x, u, token)
        if v is None:
            return None
        wynik[a:a + len(v)] = v
        postep(min(a + paczka, n), n)
    return wynik
